# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from pathlib import Path
import json


def test_routed_study_outputs() -> None:
    root = Path(__file__).resolve().parents[1]
    summary = json.loads((root / "outputs" / "study_summary.json").read_text())
    assert len(summary) == 2
    names = list(summary)
    d25 = summary[[name for name in names if "2p5D" in name][0]]
    d3 = summary[[name for name in names if "3D" in name][0]]
    assert d25["routing"]["routed_bundles"] > 0
    assert d3["link_timing_shoreline"]["critical_link_latency_ns"] < d25["link_timing_shoreline"]["critical_link_latency_ns"]
    assert d3["thermal"]["max_temperature_c"] > d25["thermal"]["max_temperature_c"]
    assert d3["pdn"]["vertical_delivery_drop_mv"] > 0
