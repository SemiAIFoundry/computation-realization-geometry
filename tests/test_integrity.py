# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from crg_validation.runner import run_theorem_audit


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_renormalization_nested_checksums() -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "validation" / "manuscript_artifacts"
    sums = base / "renormalization_results_v1" / "SHA256SUMS.txt"
    for line in sums.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        assert _sha256(base / relative) == digest


def test_northstar_run_binds_retained_results() -> None:
    root = Path(__file__).resolve().parents[1]
    base = root / "validation" / "manuscript_artifacts"
    run = json.loads((base / "northstar_run_vNext4.json").read_text(encoding="utf-8"))
    aggregate = (base / "northstar_results_vNext4" / "RESULTS_SHA256.txt").read_text(encoding="utf-8").strip()
    assert run["results_sha256"] == aggregate


def test_theorem_audit_rejects_a_retained_output_that_does_not_regenerate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "validation" / "theorem_audit"
    target = tmp_path / "validation" / "theorem_audit"
    shutil.copytree(source, target)

    (tmp_path / "manuscripts").mkdir()
    (tmp_path / "manuscripts" / "DOI_CROSSWALK.tsv").write_text("key\tdoi\n", encoding="utf-8")
    (tmp_path / "VERSION").write_text("1.0.0\n", encoding="utf-8")
    (tmp_path / "MANIFEST.tsv").write_text("path\tsize\tsha256\n", encoding="utf-8")

    assert run_theorem_audit(root=tmp_path).status == "pass"
    (target / "audit_checks_results.json").write_text("{}\n", encoding="utf-8")

    result = run_theorem_audit(root=tmp_path)
    assert result.status == "fail"
    assert "audit_checks_results.json" in result.details[0]

    shutil.copy2(source / "audit_checks_results.json", target / "audit_checks_results.json")
    (target / "audit_checks.py").write_text("pass\n", encoding="utf-8")

    result = run_theorem_audit(root=tmp_path)
    assert result.status == "fail"
    assert "missing generated file" in result.details[0]
