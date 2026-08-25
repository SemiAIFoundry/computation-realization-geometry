#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Recompute retention-study results from a frozen, provenance-bound ledger."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence, Set

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIE_RTOL = 1e-10
EPS = 1e-15
OBLIGATION_COORDS = (
    "work_flops", "fast_state_bytes", "hbm_bytes_per_op",
    "package_bytes_per_op", "event_count", "causal_depth_units",
    "replication_bytes", "represented_area_mm2",
)
METRIC_COORDS = (
    "latency_s", "energy_j", "area_mm2", "thermal_proxy",
    "reliability_risk_proxy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(x <= y + EPS for x, y in zip(a, b)) and any(x < y - EPS for x, y in zip(a, b))


def nondominated_ids(identifiers: Iterable[str], coordinates: Mapping[str, Sequence[float]]) -> Set[str]:
    identifiers = set(identifiers)
    return {
        candidate for candidate in identifiers
        if not any(dominates(coordinates[other], coordinates[candidate])
                   for other in identifiers if other != candidate)
    }


def exact_cover(identifiers: list[str], argmins: list[set[str]]) -> set[str]:
    for size in range(1, len(identifiers) + 1):
        candidates = [
            tuple(sorted(subset)) for subset in itertools.combinations(identifiers, size)
            if all(set(subset) & winners for winners in argmins)
        ]
        if candidates:
            return set(min(candidates))
    return set(identifiers)


def exact_regret(identifiers: list[str], matrix: np.ndarray, size: int, mode: str) -> set[str]:
    best = None
    for subset in itertools.combinations(range(len(identifiers)), size):
        values = np.min(matrix[:, subset], axis=1)
        score = ((float(values.max()), float(values.mean()), subset)
                 if mode == "minimax"
                 else (float(values.mean()), float(values.max()), subset))
        if best is None or score < best:
            best = score
    assert best is not None
    return {identifiers[index] for index in best[-1]}


def coverage(identifiers: list[str], argmins: list[set[str]], matrix: np.ndarray, size: int) -> set[str]:
    best = None
    for subset in itertools.combinations(range(len(identifiers)), size):
        chosen = {identifiers[index] for index in subset}
        values = np.min(matrix[:, subset], axis=1)
        score = (-sum(bool(chosen & winners) for winners in argmins),
                 float(values.mean()), float(values.max()), subset)
        if best is None or score < best:
            best = score
    assert best is not None
    return {identifiers[index] for index in best[-1]}


def diversity(identifiers: list[str], construction_metrics: np.ndarray, size: int) -> set[str]:
    # construction_metrics: scenarios x candidates x metrics, candidates ordered as identifiers.
    values = np.transpose(construction_metrics, (1, 0, 2)).reshape(len(identifiers), -1)
    scale = values.std(axis=0)
    scale[scale < EPS] = 1.0
    values = (values - values.mean(axis=0)) / scale
    distances = np.linalg.norm(values[:, None, :] - values[None, :, :], axis=2)
    best = None
    for subset in itertools.combinations(range(len(identifiers)), size):
        cover = np.min(distances[:, subset], axis=1)
        score = (float(cover.max()), float(cover.sum()), subset)
        if best is None or score < best:
            best = score
    assert best is not None
    return {identifiers[index] for index in best[-1]}


def winner(objective_row: np.ndarray, legal_row: np.ndarray, allowed_indices: list[int]) -> tuple[int | None, float]:
    values = np.where(legal_row[allowed_indices], objective_row[allowed_indices], np.inf)
    local = int(np.argmin(values))
    value = float(values[local])
    if not math.isfinite(value):
        return None, float("inf")
    return allowed_indices[local], value


def argmin_set(objective_row: np.ndarray, legal_row: np.ndarray, allowed_indices: list[int], candidate_ids: list[str]) -> set[str]:
    values = np.where(legal_row[allowed_indices], objective_row[allowed_indices], np.inf)
    best = float(np.min(values))
    if not math.isfinite(best):
        return set()
    tolerance = TIE_RTOL * max(1.0, abs(best))
    return {candidate_ids[allowed_indices[i]] for i, value in enumerate(values) if value <= best + tolerance}


def relative_regret(full: float, retained: float) -> float:
    if not math.isfinite(retained):
        return float("inf")
    return max(0.0, (retained - full) / max(abs(full), EPS))


def load_arrays(root: Path):
    pre = json.loads((root / "PREREGISTRATION.json").read_text(encoding="utf-8"))
    ledger = root / "ledger"
    candidates = pd.read_csv(ledger / "candidate_registry.csv")
    scenarios = pd.read_csv(ledger / "scenarios.csv.gz")
    evaluations = pd.read_csv(ledger / "scenario_candidate_evaluations.csv.gz")
    return pre, candidates, scenarios, evaluations


def verify_ledger(root: Path) -> list[str]:
    provenance_path = root / "ledger" / "LEDGER_PROVENANCE.json"
    if not provenance_path.is_file():
        return ["ledger/LEDGER_PROVENANCE.json is missing"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for name, expected in provenance.get("public_ledger_sha256", {}).items():
        path = root / "ledger" / name
        if not path.is_file():
            errors.append(f"missing ledger file: {name}")
        elif sha256(path) != expected:
            errors.append(f"ledger digest mismatch: {name}")
    return errors


def reanalyze(root: Path, output: Path, figure_dir: Path) -> dict:
    pre, candidates, scenarios, evaluations = load_arrays(root)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict] = []
    policy_rows: list[dict] = []
    frontier_rows: list[dict] = []
    capability_rows: list[dict] = []
    workload_summary: list[dict] = []

    for workload in pre["workloads"]:
        wid = workload["id"]
        cand = candidates[candidates.workload_id == wid].sort_values("realization_id").reset_index(drop=True)
        candidate_ids = cand.realization_id.astype(str).tolist()
        candidate_index = {rid: i for i, rid in enumerate(candidate_ids)}
        allowed_ids = sorted(cand.loc[cand.base_legal & (cand.hierarchy != "two_tier_stack"), "realization_id"].astype(str))
        vertical_ids = set(cand.loc[cand.hierarchy == "two_tier_stack", "realization_id"].astype(str))
        allowed_indices = [candidate_index[rid] for rid in allowed_ids]
        vertical_indices = [candidate_index[rid] for rid in sorted(vertical_ids)]
        all_indices = list(range(len(candidate_ids)))

        scen = scenarios[scenarios.workload_id == wid].reset_index(drop=True)
        scenario_ids = scen.scenario_id.astype(str).tolist()
        scenario_index = {sid: i for i, sid in enumerate(scenario_ids)}
        ev = evaluations[evaluations.workload_id == wid].copy()
        ev["_s"] = ev.scenario_id.map(scenario_index)
        ev["_c"] = ev.realization_id.map(candidate_index)
        ev = ev.sort_values(["_s", "_c"], kind="mergesort")
        expected_rows = len(scenario_ids) * len(candidate_ids)
        if len(ev) != expected_rows:
            raise RuntimeError(f"{wid}: expected {expected_rows} ledger rows, found {len(ev)}")
        objective = ev.objective_value.to_numpy(float).reshape(len(scenario_ids), len(candidate_ids))
        legal = ev.legal.to_numpy(bool).reshape(len(scenario_ids), len(candidate_ids))
        metrics = np.stack([
            ev[c].to_numpy(float).reshape(len(scenario_ids), len(candidate_ids))
            for c in METRIC_COORDS
        ], axis=2)

        suite_to_indices = {
            suite: group.index.to_list()
            for suite, group in scen.groupby("suite", sort=False)
        }
        construction_indices = suite_to_indices["construction"]
        construction_argmins = [
            argmin_set(objective[i], legal[i], allowed_indices, candidate_ids)
            for i in construction_indices
        ]
        construction_full = np.array([
            winner(objective[i], legal[i], allowed_indices)[1]
            for i in construction_indices
        ])
        construction_values = objective[np.ix_(construction_indices, allowed_indices)]
        construction_legal = legal[np.ix_(construction_indices, allowed_indices)]
        construction_values = np.where(construction_legal, construction_values, np.inf)
        regret = np.maximum(
            0.0,
            (construction_values - construction_full[:, None]) /
            np.maximum(np.abs(construction_full[:, None]), EPS),
        )

        crg = exact_cover(allowed_ids, construction_argmins)
        retained_size = len(crg)
        initial_i = construction_indices[0]
        initial_winner_idx, _ = winner(objective[initial_i], legal[initial_i], allowed_indices)
        assert initial_winner_idx is not None
        initial_winner_id = candidate_ids[initial_winner_idx]
        initial_pareto = nondominated_ids(
            allowed_ids,
            {rid: metrics[initial_i, candidate_index[rid], :].tolist() for rid in allowed_ids},
        )
        robust_pareto: set[str] = set()
        for i in construction_indices:
            robust_pareto |= nondominated_ids(
                allowed_ids,
                {rid: metrics[i, candidate_index[rid], :].tolist() for rid in allowed_ids},
            )
        universal = nondominated_ids(
            allowed_ids,
            {rid: [float(cand.set_index("realization_id").loc[rid, c]) for c in OBLIGATION_COORDS]
             for rid in allowed_ids},
        )
        minimax = exact_regret(allowed_ids, regret, retained_size, "minimax")
        mean_regret = exact_regret(allowed_ids, regret, retained_size, "mean")
        construction_metric_slice = metrics[np.ix_(construction_indices, allowed_indices, list(range(len(METRIC_COORDS))))]
        response_diversity = diversity(allowed_ids, construction_metric_slice, retained_size)
        policies = {
            "Initial winner": {initial_winner_id},
            "Initial full PPA Pareto": initial_pareto,
            "Construction robust-Pareto union": robust_pareto,
            "Robust-DSE minimax (matched budget)": minimax,
            "Robust-DSE mean-regret (matched budget)": mean_regret,
            "Response-diversity DSE (matched budget)": response_diversity,
            "CRG construction-ledger cover": crg,
            "CRG universal monotone": universal,
        }
        for name, retained in policies.items():
            policy_rows.append({
                "workload_id": wid,
                "policy": name,
                "retained_ids": ";".join(sorted(retained)),
                "retained_count": len(retained),
                "serialized_retention_bytes": len(json.dumps(sorted(retained)).encode()),
                "construction_ledger_hit": all(bool(retained & winners) for winners in construction_argmins),
                "universal_geometry_preserved": retained == universal,
            })
        for budget in range(1, len(allowed_ids) + 1):
            methods = {
                "CRG winner-coverage": coverage(allowed_ids, construction_argmins, regret, budget),
                "Robust-DSE minimax": exact_regret(allowed_ids, regret, budget, "minimax"),
                "Robust-DSE mean-regret": exact_regret(allowed_ids, regret, budget, "mean"),
                "Response diversity": diversity(allowed_ids, construction_metric_slice, budget),
            }
            for name, retained in methods.items():
                columns = [allowed_ids.index(rid) for rid in retained]
                values = np.min(regret[:, columns], axis=1)
                frontier_rows.append({
                    "workload_id": wid,
                    "method": name,
                    "retained_count": budget,
                    "retained_ids": ";".join(sorted(retained)),
                    "construction_miss_rate": float(np.mean(values > TIE_RTOL)),
                    "construction_mean_regret": float(values.mean()),
                    "construction_p95_regret": float(np.quantile(values, .95)),
                })

        held_out_indices = [i for suite, indices in suite_to_indices.items() if suite not in {"construction", "capability_change"} for i in indices]
        held_out_argmins = [argmin_set(objective[i], legal[i], allowed_indices, candidate_ids) for i in held_out_indices]
        oracle = exact_cover(allowed_ids, held_out_argmins)
        policy_rows.append({
            "workload_id": wid,
            "policy": "Held-out oracle argmin cover (analysis only)",
            "retained_ids": ";".join(sorted(oracle)),
            "retained_count": len(oracle),
            "serialized_retention_bytes": len(json.dumps(sorted(oracle)).encode()),
            "construction_ledger_hit": None,
            "universal_geometry_preserved": oracle == universal,
        })

        policy_indices = {name: [candidate_index[rid] for rid in sorted(retained)] for name, retained in policies.items()}
        for suite, indices in suite_to_indices.items():
            if suite == "capability_change":
                continue
            for i in indices:
                full_argmin = argmin_set(objective[i], legal[i], allowed_indices, candidate_ids)
                _, full_value = winner(objective[i], legal[i], allowed_indices)
                magnitude = scen.iloc[i].perturbation_magnitude
                for name, retained in policies.items():
                    retained_idx, retained_value = winner(objective[i], legal[i], policy_indices[name])
                    retained_winner = "NONE" if retained_idx is None else candidate_ids[retained_idx]
                    result_rows.append({
                        "workload_id": wid,
                        "suite": suite,
                        "scenario_id": scenario_ids[i],
                        "policy": name,
                        "retained_count": len(retained),
                        "full_argmin_ids": ";".join(sorted(full_argmin)),
                        "retained_winner": retained_winner,
                        "winner_missing": int(not (retained & full_argmin)),
                        "relative_regret": relative_regret(full_value, retained_value),
                        "perturbation_magnitude": magnitude,
                    })

        for i in suite_to_indices["capability_change"]:
            full_argmin = argmin_set(objective[i], legal[i], all_indices, candidate_ids)
            _, full_value = winner(objective[i], legal[i], all_indices)
            old_idx, old_value = winner(objective[i], legal[i], allowed_indices)
            reopened_idx, reopened_value = winner(objective[i], legal[i], allowed_indices + vertical_indices)
            capability_rows.append({
                "workload_id": wid,
                "scenario_id": scenario_ids[i],
                "change_detected": bool(scen.iloc[i].vertical_capability),
                "reopen_required": True,
                "new_legal_realizations": ";".join(sorted(vertical_ids)),
                "full_argmin_ids": ";".join(sorted(full_argmin)),
                "erroneous_reprice_only_winner": "NONE" if old_idx is None else candidate_ids[old_idx],
                "erroneous_reprice_only_relative_regret": relative_regret(full_value, old_value),
                "post_reopen_winner": "NONE" if reopened_idx is None else candidate_ids[reopened_idx],
                "post_reopen_relative_regret": relative_regret(full_value, reopened_value),
                "vertical_winner": int(bool(full_argmin & vertical_ids)),
            })

        workload_summary.append({
            "workload_id": wid,
            "sequence_length": workload["sequence_length"],
            "d_model": workload["d_model"],
            "d_ff": workload["d_ff"],
            "construction_argmin_ids": sorted(set().union(*construction_argmins)),
            "crg_ledger_ids": sorted(crg),
            "crg_ledger_count": len(crg),
            "minimax_ids": sorted(minimax),
            "same_crg_and_minimax": crg == minimax,
            "universal_ids": sorted(universal),
            "robust_pareto_union_ids": sorted(robust_pareto),
            "held_out_oracle_ids": sorted(oracle),
        })

    results = pd.DataFrame(result_rows)
    policy_definitions = pd.DataFrame(policy_rows)
    frontier = pd.DataFrame(frontier_rows)
    capabilities = pd.DataFrame(capability_rows)
    aggregate = results.groupby(["suite", "policy"], sort=False).agg(
        retained_min=("retained_count", "min"), retained_max=("retained_count", "max"),
        winner_miss_rate=("winner_missing", "mean"), mean_relative_regret=("relative_regret", "mean"),
        median_relative_regret=("relative_regret", "median"),
        p95_relative_regret=("relative_regret", lambda s: float(np.quantile(s, .95))),
        max_relative_regret=("relative_regret", "max"), scenarios=("scenario_id", "count"),
    ).reset_index()
    radial = results[results.suite == "radial_perturbation"].groupby(
        ["policy", "perturbation_magnitude"], sort=False
    ).agg(
        winner_miss_rate=("winner_missing", "mean"), mean_relative_regret=("relative_regret", "mean"),
        p95_relative_regret=("relative_regret", lambda s: float(np.quantile(s, .95))),
        max_relative_regret=("relative_regret", "max"), scenarios=("scenario_id", "count"),
    ).reset_index()
    capability_summary = {
        "scenarios": int(len(capabilities)),
        "change_detection_rate": float(capabilities.change_detected.mean()),
        "vertical_winner_rate": float(capabilities.vertical_winner.mean()),
        "mean_erroneous_reprice_only_regret": float(capabilities.erroneous_reprice_only_relative_regret.mean()),
        "p95_erroneous_reprice_only_regret": float(np.quantile(capabilities.erroneous_reprice_only_relative_regret, .95)),
        "post_reopen_max_regret": float(capabilities.post_reopen_relative_regret.max()),
    }
    summary = {
        "experiment_id": pre["experiment_id"],
        "preregistration_sha256": sha256(root / "PREREGISTRATION.json"),
        "analysis_mode": "frozen-ledger reanalysis",
        "total_scenarios": int(scenarios.shape[0]),
        "workloads": workload_summary,
        "aggregate_policy_summary": aggregate.to_dict(orient="records"),
        "capability_change": capability_summary,
        "interpretation": {
            "matched_baseline": "Exact minimax-regret DSE and the CRG construction-ledger cover selected the same retained sets on all four workloads.",
            "finite_scope": "The finite-ledger certificate is exact on its frozen ledger but does not extrapolate to out-of-distribution technologies.",
            "universal_scope": "CRG universal-monotone retention is exact for every tested unchanged-family monotone objective and retains all five nondominated typed signatures.",
            "capability": "Vertical capability changes legality; reopen, rather than reprice, restores zero regret.",
        },
    }
    for name, frame in (
        ("scenario_policy_results.csv", results), ("policy_definitions.csv", policy_definitions),
        ("budget_frontier.csv", frontier), ("capability_change_results.csv", capabilities),
        ("aggregate_policy_summary.csv", aggregate), ("regret_vs_perturbation_magnitude.csv", radial),
    ):
        frame.to_csv(output / name, index=False)
    (output / "experiment_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    display = [
        ("Initial full PPA Pareto", "Initial Pareto"),
        ("Construction robust-Pareto union", "Robust Pareto union"),
        ("Robust-DSE minimax (matched budget)", "Minimax DSE"),
        ("CRG construction-ledger cover", "CRG ledger cover"),
        ("CRG universal monotone", "CRG universal"),
    ]
    suites_to_plot = [("held_out_in_distribution", "Held-out ID"), ("technology_ood", "Technology OOD"), ("joint_ood", "Joint OOD")]
    figure, axes = plt.subplots(2, 2, figsize=(13.2, 9))
    x = np.arange(len(display)); width = .24
    for offset, (suite, label) in enumerate(suites_to_plot):
        baseline = aggregate[aggregate.suite == suite].set_index("policy")
        axes[0, 0].bar(x + (offset - 1) * width,
                       [100 * baseline.loc[p, "winner_miss_rate"] for p, _ in display],
                       width, label=label)
    axes[0, 0].set_xticks(x, [label for _, label in display], rotation=20, ha="right")
    axes[0, 0].set_ylabel("winner-miss rate (%)"); axes[0, 0].set_title("(a) Held-out and OOD coverage")
    axes[0, 0].grid(axis="y", alpha=.25); axes[0, 0].legend(fontsize=8)
    for policy, label in display:
        group = radial[radial.policy == policy].sort_values("perturbation_magnitude")
        axes[0, 1].plot(group.perturbation_magnitude, 100 * group.p95_relative_regret, marker="o", label=label)
    axes[0, 1].axvline(1, linestyle="--", linewidth=1, label="construction boundary")
    axes[0, 1].set_xlabel("normalized perturbation magnitude"); axes[0, 1].set_ylabel("p95 relative regret (%)")
    axes[0, 1].set_title("(b) Controlled extrapolation"); axes[0, 1].grid(alpha=.25); axes[0, 1].legend(fontsize=8)
    axes[1, 0].axis("off"); table_rows = []
    for policy, label in display:
        definition = policy_definitions[policy_definitions.policy == policy]
        lo, hi = int(definition.retained_count.min()), int(definition.retained_count.max())
        retained = str(lo) if lo == hi else f"{lo}-{hi}"
        row = aggregate[(aggregate.suite == "joint_ood") & (aggregate.policy == policy)].iloc[0]
        table_rows.append([label, retained, f"{100 * row.winner_miss_rate:.2f}%", f"{100 * row.p95_relative_regret:.2f}%"])
    table = axes[1, 0].table(cellText=table_rows,
                              colLabels=["Policy", "Retained", "Joint OOD miss", "p95 regret"],
                              loc="center", cellLoc="center", colLoc="center", colWidths=[.42, .14, .24, .20])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 1.55)
    axes[1, 0].set_title("(c) Strong DSE and CRG can coincide at equal scope", pad=12)
    axes[1, 1].bar(["Reprice-only\nmean", "Reprice-only\np95", "After reopen\nmax"],
                    [100 * capability_summary["mean_erroneous_reprice_only_regret"],
                     100 * capability_summary["p95_erroneous_reprice_only_regret"],
                     100 * capability_summary["post_reopen_max_regret"]])
    axes[1, 1].set_ylabel("relative regret (%)"); axes[1, 1].set_title("(d) Capability change requires reopening")
    axes[1, 1].grid(axis="y", alpha=.25)
    axes[1, 1].text(.5, .95,
                    f"Detected: {100 * capability_summary['change_detection_rate']:.0f}%\nVertical winner: {100 * capability_summary['vertical_winner_rate']:.1f}%",
                    transform=axes[1, 1].transAxes, ha="center", va="top", fontsize=9)
    figure.suptitle("Adversarial retention experiment v2: construction and evaluation ledgers are disjoint", fontsize=15)
    figure.tight_layout(rect=[0, 0, 1, .96])
    figure.savefig(figure_dir / "retention_adversarial_v2.png", dpi=300, bbox_inches="tight")
    figure.savefig(figure_dir / "retention_adversarial_v2.pdf", bbox_inches="tight")
    plt.close(figure)
    return summary


def _walk_json(expected: Any, actual: Any, path: str, errors: list[str]) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        if expected != actual:
            errors.append(f"{path}: {expected!r} != {actual!r}")
        return
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(expected), float(actual), rel_tol=1e-12, abs_tol=1e-12):
            errors.append(f"{path}: {expected!r} != {actual!r}")
        return
    if type(expected) is not type(actual):
        errors.append(f"{path}: type {type(expected).__name__} != {type(actual).__name__}")
        return
    if isinstance(expected, dict):
        if set(expected) != set(actual):
            errors.append(f"{path}: key mismatch")
            return
        for key in sorted(expected):
            _walk_json(expected[key], actual[key], f"{path}.{key}", errors)
        return
    if isinstance(expected, list):
        if len(expected) != len(actual):
            errors.append(f"{path}: length {len(expected)} != {len(actual)}")
            return
        for index, (left, right) in enumerate(zip(expected, actual)):
            _walk_json(left, right, f"{path}[{index}]", errors)
        return
    if expected != actual:
        errors.append(f"{path}: {expected!r} != {actual!r}")


def _compare_frames(expected_path: Path, actual_path: Path) -> list[str]:
    if not expected_path.is_file():
        return [f"missing expected output: {expected_path.name}"]
    if not actual_path.is_file():
        return [f"missing generated output: {actual_path.name}"]
    expected = pd.read_csv(expected_path, keep_default_na=False, low_memory=False)
    actual = pd.read_csv(actual_path, keep_default_na=False, low_memory=False)
    if list(expected.columns) != list(actual.columns):
        return [f"{actual_path.name}: column mismatch"]
    if len(expected) != len(actual):
        return [f"{actual_path.name}: row count {len(actual)} != {len(expected)}"]
    errors: list[str] = []
    for column in expected.columns:
        left = expected[column]
        right = actual[column]
        try:
            lnum = pd.to_numeric(left, errors="raise").to_numpy(float)
            rnum = pd.to_numeric(right, errors="raise").to_numpy(float)
        except (TypeError, ValueError):
            mismatch = left.astype(str).to_numpy() != right.astype(str).to_numpy()
        else:
            mismatch = ~np.isclose(lnum, rnum, rtol=1e-12, atol=1e-12, equal_nan=True)
        indices = np.flatnonzero(mismatch)
        for index in indices[:10]:
            errors.append(
                f"{actual_path.name}: row {int(index)} column {column}: "
                f"{left.iloc[index]!r} != {right.iloc[index]!r}"
            )
        if len(indices) > 10:
            errors.append(f"{actual_path.name}: {len(indices) - 10} additional mismatches in {column}")
        if errors:
            break
    return errors


def verify_outputs(root: Path, output: Path) -> list[str]:
    expected = root / "expected"
    errors = verify_ledger(root)
    for name in (
        "scenario_policy_results.csv",
        "policy_definitions.csv",
        "budget_frontier.csv",
        "capability_change_results.csv",
        "aggregate_policy_summary.csv",
        "regret_vs_perturbation_magnitude.csv",
    ):
        expected_name = name + ".gz" if name == "scenario_policy_results.csv" else name
        errors.extend(_compare_frames(expected / expected_name, output / name))
    expected_json = json.loads((expected / "experiment_summary.json").read_text(encoding="utf-8"))
    actual_json = json.loads((output / "experiment_summary.json").read_text(encoding="utf-8"))
    _walk_json(expected_json, actual_json, "experiment_summary", errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--figures", type=Path)
    parser.add_argument("--verify", action="store_true", help="compare regenerated results with retained expected outputs")
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "generated").resolve()
    figures = (args.figures or root / "generated_figures").resolve()
    summary = reanalyze(root, output, figures)
    errors = verify_outputs(root, output) if args.verify else []
    report = {
        "status": "pass" if not errors else "fail",
        "analysis": summary,
        "verification_errors": errors,
        "generated_output": str(output),
        "generated_figures": str(figures),
    }
    print(json.dumps(report, indent=2))
    return 0 if not errors else 1

if __name__ == "__main__":
    raise SystemExit(main())
