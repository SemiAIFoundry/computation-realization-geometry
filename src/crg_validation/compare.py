# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _equal(a: float, b: float, rtol: float, atol: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return True
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def _walk(a: Any, b: Any, path: str, rtol: float, atol: float, errors: list[str]) -> None:
    if isinstance(a, bool) or isinstance(b, bool):
        if a != b:
            errors.append(f"{path}: {a!r} != {b!r}")
        return
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not _equal(float(a), float(b), rtol, atol):
            errors.append(f"{path}: {a!r} != {b!r}")
        return
    if type(a) is not type(b):
        errors.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
        return
    if isinstance(a, dict):
        if set(a) != set(b):
            errors.append(f"{path}: key mismatch")
            return
        for key in sorted(a):
            _walk(a[key], b[key], f"{path}.{key}", rtol, atol, errors)
        return
    if isinstance(a, list):
        if len(a) != len(b):
            errors.append(f"{path}: length {len(a)} != {len(b)}")
            return
        for index, (left, right) in enumerate(zip(a, b)):
            _walk(left, right, f"{path}[{index}]", rtol, atol, errors)
        return
    if a != b:
        errors.append(f"{path}: {a!r} != {b!r}")


def compare_file(expected: Path, actual: Path, rtol: float = 1e-12, atol: float = 1e-12) -> list[str]:
    if not expected.is_file():
        return [f"missing baseline: {expected}"]
    if not actual.is_file():
        return [f"missing generated file: {actual}"]
    if expected.suffix.lower() == ".json":
        errors: list[str] = []
        _walk(json.loads(expected.read_text()), json.loads(actual.read_text()), "$", rtol, atol, errors)
        return errors
    if expected.suffix.lower() == ".csv":
        with expected.open(newline="", encoding="utf-8") as handle:
            left = list(csv.DictReader(handle))
        with actual.open(newline="", encoding="utf-8") as handle:
            right = list(csv.DictReader(handle))
        if len(left) != len(right):
            return [f"row count {len(left)} != {len(right)}"]
        errors = []
        for index, (a, b) in enumerate(zip(left, right)):
            if set(a) != set(b):
                errors.append(f"row {index}: columns differ")
                continue
            for key in a:
                try:
                    x, y = float(a[key]), float(b[key])
                except (TypeError, ValueError):
                    if a[key] != b[key]:
                        errors.append(f"row {index}.{key}: {a[key]!r} != {b[key]!r}")
                else:
                    if not _equal(x, y, rtol, atol):
                        errors.append(f"row {index}.{key}: {x!r} != {y!r}")
        return errors
    return [] if expected.read_bytes() == actual.read_bytes() else ["byte mismatch"]
