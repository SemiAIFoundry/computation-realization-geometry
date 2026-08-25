# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Focused portable check for the retained EPFL manuscript artifacts.

The complete ten-netlist hierarchy reconstruction is intentionally reserved
for the registered strict-release environment.  This portable check verifies
all third-party input bytes, reconstructs the full adder hierarchy summary
from source, and compares it with the retained baseline.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from epfl_rccg_validation import (
    NETLIST_DIR,
    level_metrics_internal,
    parse_blif,
    projection_adjacency,
    rcm_hierarchy,
    summary,
)

EXPECTED_HASHES = {
    "adder.blif": "3126ec3f60fa4a37ac94fa5e70aa8706aac79ccf316ddaa09ed70db33625d472",
    "arbiter.blif": "bf808364cacf4ccf38f9fdbfec39e33a407681b0cdf65783613ad8466b1c1499",
    "bar.blif": "cbda4fc7dffda03ace50aa1d329ca81b0b597c6f52c480f12c5dfa3b93437a4b",
    "div.blif": "4c19702b927f23fd99940b283833ad65ec2694deecea7730444fc1ee1691a456",
    "i2c.blif": "9a600ae50bbfd998251310c0432374b8bfad5adc1c8c89ea75e1e640e70cd999",
    "log2.blif": "c0d052af4e95de4c1327a2ceddd855518a052a8f3a3960e6d58c5b5ca65c0dde",
    "multiplier.blif": "9f66477f16653748f444fb00ff076330d51da706607253e98d2ddfdbe69edd75",
    "router.blif": "eabb05688bbb01fc623ce1c1d231d3356b47e82948a6b9537b39eab2d8f7aa0c",
    "sqrt.blif": "7c5a28925fb2a6b3f1d0979ceaa93eafabea39fa418ec717e09cb4ff3b882107",
    "square.blif": "a5ffcd4b148c690817d0391bbe0894686ac983aaa0f43d24281bccdb2287b8f5",
}


def _same_number(actual: float, expected: float) -> bool:
    if math.isnan(actual) and math.isnan(expected):
        return True
    return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def main() -> int:
    failures: list[str] = []
    for name, expected in EXPECTED_HASHES.items():
        path = NETLIST_DIR / name
        if not path.is_file():
            failures.append(f"missing EPFL input: {name}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"EPFL input hash mismatch: {name}: {actual} != {expected}")

    root = Path(__file__).resolve().parent
    retained = json.loads((root / "epfl_results_v2" / "summary.json").read_text(encoding="utf-8"))
    expected_row = next(
        row for row in retained
        if row["name"] == "adder" and row["hierarchy"] == "RCM-fixed"
    )

    netlist = parse_blif(NETLIST_DIR / "adder.blif")
    adjacency = projection_adjacency(netlist)
    levels = rcm_hierarchy(adjacency, len(netlist.vertices), min_size=8)
    actual_row = summary(level_metrics_internal(netlist, levels, "RCM-fixed"))

    for key, expected_value in expected_row.items():
        actual_value = actual_row.get(key)
        if isinstance(expected_value, float):
            if actual_value is None or not _same_number(float(actual_value), expected_value):
                failures.append(f"adder summary mismatch {key}: {actual_value!r} != {expected_value!r}")
        elif actual_value != expected_value:
            failures.append(f"adder summary mismatch {key}: {actual_value!r} != {expected_value!r}")

    report = {
        "status": "pass" if not failures else "fail",
        "input_hashes_verified": len(EXPECTED_HASHES) - sum("hash mismatch" in item or "missing EPFL" in item for item in failures),
        "portable_reconstruction": "adder/RCM-fixed",
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
