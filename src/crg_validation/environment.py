# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import importlib.metadata
import platform
import sys

EXPECTED = {
    "numpy": "2.5.2",
    "pandas": "3.0.5",
    "tzdata": "2026.3",
    "matplotlib": "3.10.8",
    "scipy": "1.18.1",
    "pytest": "9.1.1",
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
    version = sys.version_info[:2]
    implementation = platform.python_implementation()
    python_supported = implementation == "CPython" and (3, 12) <= version < (3, 14)
    if not python_supported:
        differences.append(
            f"{implementation} {sys.version_info.major}.{sys.version_info.minor} "
            "is outside the supported CPython 3.12-3.13 range"
        )
    return {
        "python": sys.version,
        "implementation": implementation,
        "platform": platform.platform(),
        "python_supported": python_supported,
        "dependency_match": not differences,
        "differences": differences,
        "packages": packages,
    }
