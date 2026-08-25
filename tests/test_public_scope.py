# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from pathlib import Path

from crg_validation.scope import check


def test_public_scope_is_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    assert check(root, include_transient=False) == []
