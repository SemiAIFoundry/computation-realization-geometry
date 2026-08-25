# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .compare import compare_file


@dataclass
class Result:
    key: str
    status: str
    elapsed_s: float
    details: list[str]
    workdir: str | None = None


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run(command: list[str], cwd: Path, log: Path) -> None:
    env = os.environ.copy()
    env.setdefault("MPLBACKEND", "Agg")
    # Isolate Matplotlib's cache and lock files per study workspace. This avoids
    # cross-study font-cache contention during aggregate validation runs.
    mpl_config = cwd / ".mplconfig"
    mpl_config.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_config)
    env.setdefault("PYTHONHASHSEED", "0")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
    if process.returncode:
        tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-80:]
        raise RuntimeError(f"command failed ({process.returncode}): {' '.join(command)}\n" + "\n".join(tail))


def _temp_copy(source: Path, keep: bool, prefix: str) -> tuple[Path, Path]:
    parent = repo_root() / ".runs" if keep else Path(tempfile.mkdtemp(prefix=f"crg-{prefix}-"))
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{prefix}-", dir=parent)) if keep else parent / prefix
    shutil.copytree(source, work, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache", "*.pyc"))
    return parent, work


def _finish(key: str, start: float, parent: Path, work: Path, keep: bool, details: list[str], error: Exception | None = None) -> Result:
    if error is None:
        result = Result(key, "pass", time.perf_counter() - start, details, str(work) if keep else None)
    else:
        result = Result(key, "fail", time.perf_counter() - start, [str(error)], str(work) if keep else None)
    if not keep:
        shutil.rmtree(parent, ignore_errors=True)
    return result


def run_theorem_audit(keep: bool = False) -> Result:
    start = time.perf_counter(); source = repo_root() / "validation/theorem_audit"
    parent, work = _temp_copy(source, keep, "theorem-audit")
    try:
        _run([sys.executable, "audit_checks.py"], work, work / "audit_checks.log")
        _run([sys.executable, "build_audit_matrix.py"], work, work / "build_audit_matrix.log")
        return _finish("theorem-audit", start, parent, work, keep, ["audit checks and matrix regeneration passed"])
    except Exception as exc:
        return _finish("theorem-audit", start, parent, work, keep, [], exc)


def run_retention(keep: bool = False) -> Result:
    start = time.perf_counter(); source = repo_root() / "validation/retention"
    parent, work = _temp_copy(source, keep, "retention")
    try:
        _run([sys.executable, "reanalyze.py", "--verify"], work, work / "retention_verify.log")
        _run([sys.executable, "-m", "pytest", "-q", "tests"], work, work / "retention_pytest.log")
        return _finish("retention", start, parent, work, keep, ["frozen-ledger recalculation passed"])
    except Exception as exc:
        return _finish("retention", start, parent, work, keep, [], exc)


def run_routed_hi(keep: bool = False) -> Result:
    start = time.perf_counter(); source = repo_root() / "validation/routed_heterogeneous_integration"
    parent, work = _temp_copy(source, keep, "routed-hi")
    try:
        baseline = source / "outputs"
        _run([sys.executable, "run_study.py"], work, work / "run_study.log")
        _run([sys.executable, "-m", "pytest", "-q", "tests"], work, work / "routed_pytest.log")
        errors = []
        for rel in ["study_summary.json", "design_comparison.csv", "net_metrics.csv"]:
            errors.extend(f"{rel}: {item}" for item in compare_file(baseline / rel, work / "outputs" / rel))
        if errors:
            raise RuntimeError("; ".join(errors[:20]))
        return _finish("routed-hi", start, parent, work, keep, ["routed compact-model outputs match retained baselines"])
    except Exception as exc:
        return _finish("routed-hi", start, parent, work, keep, [], exc)


def _run_manuscript_once(source: Path, work: Path) -> list[str]:
    logs = work / "logs"
    logs.mkdir(exist_ok=True)

    for dirname in ["renormalization_results_v1", "figures_renormalization_v1"]:
        shutil.rmtree(work / dirname, ignore_errors=True)
    _run([sys.executable, "renormalization_validation_v1.py"], work, logs / "scaling.log")

    for dirname in ["flagship_results_v2", "figures_flagship_v2"]:
        shutil.rmtree(work / dirname, ignore_errors=True)
    _run([sys.executable, "flagship_validation_v2.py"], work, logs / "components.log")

    _run([sys.executable, "northstar_transformer_model_v3.py", "--self-test-only"], work, logs / "transformer_selftest.log")
    test_class = "test_northstar_transformer_model_v3.NorthstarV3IndependentTests"
    transformer_tests = [
        sys.executable, "-m", "unittest", "-v",
        f"{test_class}.test_gqa_projection_shape_count",
        f"{test_class}.test_executed_pair_oracles",
        f"{test_class}.test_review4_dense_padded_point_from_direct_algebra",
        f"{test_class}.test_default_certificate_from_direct_algebra",
        f"{test_class}.test_vectorized_audit_rejects_nonintegral_contexts",
        f"{test_class}.test_h_pipeline_conserves_every_resource",
        f"{test_class}.test_general_layout_assignment_against_bruteforce_small_p",
    ]
    _run(transformer_tests, work, logs / "transformer_unittest.log")
    generated = work / "northstar_results_vNext4"
    _run([
        sys.executable, "-m", "unittest", "-v",
        f"{test_class}.test_generated_result_manifest",
    ], work, logs / "transformer_manifest.log")

    for dirname in ["epfl_results_v2", "figures_epfl_v2", "results_epfl_v3"]:
        shutil.rmtree(work / dirname, ignore_errors=True)
    netlists = repo_root() / "third_party" / "epfl" / "netlists"
    shutil.copytree(netlists, work / "third_party_epfl_netlists", dirs_exist_ok=True)
    shutil.copytree(source / "epfl_results_v2", work / "epfl_results_v2")
    _run([sys.executable, "epfl_portable_check.py"], work, logs / "epfl_portable.log")
    for dirname in ["figures_epfl_v2", "results_epfl_v3"]:
        shutil.copytree(source / dirname, work / dirname)

    _run([sys.executable, "trilogy_validation_v3.py"], work, logs / "integration.log")

    errors: list[str] = []
    comparisons = [
        "renormalization_results_v1/summary.json",
        "flagship_results_v2/summary.json",
        "northstar_results_vNext4/northstar_results.json",
        "trilogy_validation_v3_results.json",
    ]
    for rel in comparisons:
        actual = generated / rel.split("/", 1)[1] if rel.startswith("northstar_results_vNext4/") else work / rel
        errors.extend(f"{rel}: {item}" for item in compare_file(source / rel, actual))
    return errors


def run_manuscript(keep: bool = False) -> Result:
    start = time.perf_counter()
    source = repo_root() / "validation/manuscript_artifacts"
    parent, work = _temp_copy(source, keep, "manuscript")
    try:
        errors = _run_manuscript_once(source, work)
        if errors:
            raise RuntimeError("; ".join(errors[:20]))
        return _finish(
            "manuscript",
            start,
            parent,
            work,
            keep,
            ["manuscript-connected public profile matches retained baselines"],
        )
    except Exception as exc:
        return _finish("manuscript", start, parent, work, keep, [], exc)


def run_all(keep: bool = False) -> list[Result]:
    """Run every public validation target in independent work directories."""
    jobs = [run_theorem_audit, run_retention, run_routed_hi, run_manuscript]
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = [pool.submit(job, keep) for job in jobs]
        return [future.result() for future in futures]


def serialize(results: list[Result]) -> str:
    return json.dumps([asdict(result) for result in results], indent=2, sort_keys=True)
