# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

ROOT_INTEGRITY_FILES = {"SHA256SUMS.txt", "MANIFEST.tsv"}
EXCLUDED_DIRS = {".git", ".runs", ".generated", ".pytest_cache", "__pycache__", ".mypy_cache", ".ruff_cache", ".venv", "generated", "generated_figures", "generated_transformer", "logs", "build", "dist"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return (
        bool(value)
        and "\\" not in value
        and not path.is_absolute()
        and path.as_posix() == value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


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
        if len(rel.parts) == 1 and rel.name in ROOT_INTEGRITY_FILES:
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
    manifest_path = root / "MANIFEST.tsv"
    sums_path = root / "SHA256SUMS.txt"
    errors: list[str] = []
    if not manifest_path.is_file():
        errors.append("MANIFEST.tsv missing")
    if not sums_path.is_file():
        errors.append("SHA256SUMS.txt missing")
    if errors:
        return errors

    expected: dict[str, str] = {}
    for number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            digest, rel = line.split("  ", 1)
        except ValueError:
            errors.append(f"malformed SHA256SUMS.txt line {number}")
            continue
        if not _safe_relative_path(rel):
            errors.append(f"unsafe SHA256SUMS.txt path at line {number}: {rel}")
            continue
        if SHA256_RE.fullmatch(digest) is None:
            errors.append(f"invalid SHA-256 in SHA256SUMS.txt line {number}")
            continue
        if rel in expected:
            errors.append(f"duplicate SHA256SUMS.txt path: {rel}")
        expected[rel] = digest

    manifest: dict[str, tuple[int, str]] = {}
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "path\tbytes\tsha256":
        errors.append("invalid MANIFEST.tsv header")
    for number, line in enumerate(lines[1:], start=2):
        try:
            rel, size_text, digest = line.split("\t")
            size = int(size_text)
        except (ValueError, TypeError):
            errors.append(f"malformed MANIFEST.tsv line {number}")
            continue
        if not _safe_relative_path(rel):
            errors.append(f"unsafe MANIFEST.tsv path at line {number}: {rel}")
            continue
        if size < 0:
            errors.append(f"negative MANIFEST.tsv size at line {number}")
            continue
        if SHA256_RE.fullmatch(digest) is None:
            errors.append(f"invalid SHA-256 in MANIFEST.tsv line {number}")
            continue
        if rel in manifest:
            errors.append(f"duplicate MANIFEST.tsv path: {rel}")
        manifest[rel] = (size, digest)

    if set(manifest) != set(expected):
        for rel in sorted(set(manifest) - set(expected)):
            errors.append(f"MANIFEST.tsv path missing from SHA256SUMS.txt: {rel}")
        for rel in sorted(set(expected) - set(manifest)):
            errors.append(f"SHA256SUMS.txt path missing from MANIFEST.tsv: {rel}")
    for rel in sorted(set(manifest) & set(expected)):
        if manifest[rel][1] != expected[rel]:
            errors.append(f"integrity records disagree: {rel}")

    actual_paths = {rel.as_posix(): path for path, rel in iter_files(root)}
    for rel in sorted(set(expected) - set(actual_paths)):
        path = root / rel
        if path.exists():
            errors.append(f"attested path is excluded from the public payload: {rel}")
        else:
            errors.append(f"missing: {rel}")
    for rel in sorted(set(expected) & set(actual_paths)):
        digest = expected[rel]
        path = actual_paths[rel]
        try:
            path.resolve().relative_to(root.resolve())
        except ValueError:
            errors.append(f"path resolves outside corpus root: {rel}")
            continue
        actual_digest = sha256(path)
        if actual_digest != digest:
            errors.append(f"digest mismatch: {rel}")
        if rel in manifest and path.stat().st_size != manifest[rel][0]:
            errors.append(f"size mismatch: {rel}")
    extras = sorted(set(actual_paths) - set(expected))
    for rel in extras:
        errors.append(f"unlisted file: {rel}")
    return errors
