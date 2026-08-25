# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import importlib.metadata
import platform
import sys

EXPECTED = {
    "numpy": "2.3.5",
    "pandas": "2.2.3",
    "matplotlib": "3.10.8",
    "scipy": "1.17.0",
    "pytest": "9.0.2",
}


def report() -> dict:
    packages = {}
    differences = []
    for name, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(name).split("+", 1)[0]
        except importlib.metadata.PackageNotFoundError:
            actual = None
        packages[name] = actual
        if actual != expected:
            differences.append(f"{name}: {actual!r} != {expected!r}")
    python_supported = sys.version_info >= (3, 12)
    if not python_supported:
        differences.append(f"Python {sys.version_info.major}.{sys.version_info.minor} is below 3.12")
    return {
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "python_supported": python_supported,
        "dependency_match": not differences,
        "differences": differences,
        "packages": packages,
    }
