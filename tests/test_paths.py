# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from pathlib import Path

import pytest

from crg_validation.paths import CorpusRootError, discover_root


def test_discovers_checkout_from_child_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root / "validation" / "theorem_audit")
    assert discover_root() == root


def test_explicit_invalid_root_fails(tmp_path: Path) -> None:
    with pytest.raises(CorpusRootError, match="not a CRG corpus root"):
        discover_root(tmp_path)
