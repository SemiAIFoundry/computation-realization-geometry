# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_ledger_digests_and_record_counts() -> None:
    provenance = json.loads((ROOT / "ledger/LEDGER_PROVENANCE.json").read_text())
    for name, expected in provenance["public_ledger_sha256"].items():
        assert _sha256(ROOT / "ledger" / name) == expected
    assert len(pd.read_csv(ROOT / "ledger/workloads.csv")) == 4
    assert len(pd.read_csv(ROOT / "ledger/candidate_registry.csv")) == 24
    assert len(pd.read_csv(ROOT / "ledger/scenarios.csv.gz")) == 26624
    assert len(pd.read_csv(ROOT / "ledger/scenario_candidate_evaluations.csv.gz")) == 159744


def test_expected_summary_contract() -> None:
    summary = json.loads((ROOT / "expected/experiment_summary.json").read_text())
    assert summary["experiment_id"] == "crg.retention.adversarial.v2"
    assert summary["total_scenarios"] == 26624
    assert len(summary["workloads"]) == 4
    assert summary["capability_change"]["post_reopen_max_regret"] == 0.0
