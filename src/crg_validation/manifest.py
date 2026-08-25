# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import hashlib
from pathlib import Path

EXCLUDED_FILES = {"SHA256SUMS.txt", "MANIFEST.tsv"}
EXCLUDED_DIRS = {".git", ".runs", ".generated", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", ".venv", "generated", "generated_figures", "generated_transformer", "logs", "build", "dist"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.name in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        if path.suffix in EXCLUDED_SUFFIXES or path.name == ".DS_Store":
            continue
        yield path, rel


def generate(root: Path) -> tuple[Path, Path]:
    rows = []
    sums = []
    for path, rel in iter_files(root):
        digest = sha256(path)
        rows.append((rel.as_posix(), path.stat().st_size, digest))
        sums.append(f"{digest}  {rel.as_posix()}")
    manifest = root / "MANIFEST.tsv"
    manifest.write_text(
        "path\tbytes\tsha256\n" + "\n".join(f"{p}\t{s}\t{d}" for p, s, d in rows) + "\n",
        encoding="utf-8",
    )
    sums_path = root / "SHA256SUMS.txt"
    sums_path.write_text("\n".join(sums) + "\n", encoding="utf-8")
    return manifest, sums_path


def verify(root: Path) -> list[str]:
    sums_path = root / "SHA256SUMS.txt"
    if not sums_path.is_file():
        return ["SHA256SUMS.txt missing"]
    expected = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        if line:
            digest, rel = line.split("  ", 1)
            expected[rel] = digest
    errors: list[str] = []
    actual_paths = {rel.as_posix(): path for path, rel in iter_files(root)}
    for rel, digest in expected.items():
        path = root / rel
        if not path.is_file():
            errors.append(f"missing: {rel}")
        elif sha256(path) != digest:
            errors.append(f"digest mismatch: {rel}")
    extras = sorted(set(actual_paths) - set(expected))
    for rel in extras:
        errors.append(f"unlisted file: {rel}")
    return errors
