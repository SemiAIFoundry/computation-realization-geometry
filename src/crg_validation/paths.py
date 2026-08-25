# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import os
from pathlib import Path

CORPUS_ROOT_ENV = "CRG_CORPUS_ROOT"


class CorpusRootError(RuntimeError):
    """Raised when the public research corpus cannot be located."""


def _is_corpus_root(path: Path) -> bool:
    return (
        (path / "VERSION").is_file()
        and (path / "MANIFEST.tsv").is_file()
        and (path / "manuscripts" / "DOI_CROSSWALK.tsv").is_file()
        and (path / "validation").is_dir()
    )


def discover_root(explicit: str | Path | None = None, *, start: Path | None = None) -> Path:
    """Locate a corpus checkout without depending on package installation layout.

    Resolution order is an explicit path, ``CRG_CORPUS_ROOT``, the current
    directory and its parents, then the source checkout when running editable.
    """
    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if _is_corpus_root(candidate):
            return candidate
        raise CorpusRootError(f"not a CRG corpus root: {candidate}")

    configured = os.environ.get(CORPUS_ROOT_ENV)
    if configured:
        candidate = Path(configured).expanduser().resolve()
        if _is_corpus_root(candidate):
            return candidate
        raise CorpusRootError(f"{CORPUS_ROOT_ENV} is not a CRG corpus root: {candidate}")

    origin = (start or Path.cwd()).resolve()
    candidates = (origin, *origin.parents)
    source = Path(__file__).resolve()
    candidates += tuple(source.parents)
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if _is_corpus_root(candidate):
            return candidate

    raise CorpusRootError(
        "CRG corpus not found; run from a corpus checkout or provide "
        "--root PATH / CRG_CORPUS_ROOT"
    )
