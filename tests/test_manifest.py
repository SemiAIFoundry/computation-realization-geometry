# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from pathlib import Path

from crg_validation.manifest import verify


def test_release_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    assert verify(root) == []
