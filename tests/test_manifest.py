# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from pathlib import Path
import hashlib
import shutil

import pytest

from crg_validation.manifest import verify


def test_release_manifest() -> None:
    root = Path(__file__).resolve().parents[1]
    assert verify(root) == []


def test_manifest_tampering_is_detected(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("MANIFEST.tsv", "SHA256SUMS.txt"):
        shutil.copy2(root / name, tmp_path / name)
    (tmp_path / "MANIFEST.tsv").write_text(
        (tmp_path / "MANIFEST.tsv").read_text(encoding="utf-8").replace("\tsha256\n", "\tbroken\n", 1),
        encoding="utf-8",
    )
    assert "invalid MANIFEST.tsv header" in verify(tmp_path)


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    digest = "0" * 64
    (tmp_path / "MANIFEST.tsv").write_text(
        f"path\tbytes\tsha256\n../outside\t0\t{digest}\n",
        encoding="utf-8",
    )
    (tmp_path / "SHA256SUMS.txt").write_text(
        f"{digest}  ../outside\n",
        encoding="utf-8",
    )
    errors = verify(tmp_path)
    assert any("unsafe MANIFEST.tsv path" in error for error in errors)
    assert any("unsafe SHA256SUMS.txt path" in error for error in errors)


@pytest.mark.parametrize(
    "relative",
    (".git/config", "build/file.txt", ".venv/state.json", ".DS_Store", "payload.pyc"),
)
def test_manifest_rejects_attested_excluded_paths(tmp_path: Path, relative: str) -> None:
    payload = b"excluded public payload"
    digest = hashlib.sha256(payload).hexdigest()
    path = tmp_path / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    (tmp_path / "MANIFEST.tsv").write_text(
        f"path\tbytes\tsha256\n{relative}\t{len(payload)}\t{digest}\n",
        encoding="utf-8",
    )
    (tmp_path / "SHA256SUMS.txt").write_text(
        f"{digest}  {relative}\n",
        encoding="utf-8",
    )

    assert f"attested path is excluded from the public payload: {relative}" in verify(tmp_path)
