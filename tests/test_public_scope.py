# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from pathlib import Path

from crg_validation.scope import check


def test_public_scope_is_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    assert check(root, include_transient=False) == []


def test_checkout_metadata_is_not_public_payload(tmp_path: Path) -> None:
    git_file = tmp_path / ".git" / "config"
    git_file.parent.mkdir()
    git_file.write_text("local checkout metadata", encoding="utf-8")
    assert all(".git" not in error for error in check(tmp_path, include_transient=False))
