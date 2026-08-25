# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

ALLOWED_ROOT_FILES = {
    ".gitattributes", ".gitignore", "CITATION.cff", "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md", "GOVERNANCE.md", "LICENSE", "LICENSE_SCOPE.tsv",
    "MANIFEST.tsv", "NOTICE", "README.md", "SECURITY.md",
    "SHA256SUMS.txt", "VERSION", "pyproject.toml", "release-metadata.json",
}
ALLOWED_EXACT = {
    ".github/dependabot.yml",
    ".github/workflows/validation.yml",
    "LICENSES/CC-BY-NC-4.0.txt",
    "LICENSES/EPFL-MIT.txt",
    "LICENSES/PolyForm-Noncommercial-1.0.0.txt",
    "manuscripts/DOI_CROSSWALK.tsv",
    "manuscripts/PUBLICATION_CORPUS.bib",
    "manuscripts/PUBLICATION_CORPUS.json",
    "requirements/audit.txt",
    "requirements/audit.lock",
    "requirements/build.txt",
    "requirements/build.lock",
    "requirements/validation.txt",
    "requirements/validation.lock",
    "src/crg_validation/__init__.py",
    "src/crg_validation/cli.py",
    "src/crg_validation/compare.py",
    "src/crg_validation/environment.py",
    "src/crg_validation/manifest.py",
    "src/crg_validation/paths.py",
    "src/crg_validation/runner.py",
    "src/crg_validation/scope.py",
    "tests/test_manifest.py",
    "tests/test_dependencies.py",
    "tests/test_integrity.py",
    "tests/test_metadata.py",
    "tests/test_paths.py",
    "tests/test_public_scope.py",
    "third_party/epfl/PROVENANCE.json",
    "tools/regenerate_release.py",
    "tools/verify_wheel.py",
}
ALLOWED_PREFIXES = (
    "third_party/epfl/netlists/",
    "validation/manuscript_artifacts/",
    "validation/retention/",
    "validation/routed_heterogeneous_integration/",
    "validation/theorem_audit/",
)
TEXT_SUFFIXES = {
    ".md", ".txt", ".py", ".sh", ".json", ".yaml", ".yml", ".toml",
    ".tsv", ".csv", ".cff", ".bib", ".ini", ".cfg", ".tex",
}
TRANSIENT_PARTS = {
    ".git", ".runs", ".generated", ".pytest_cache", "__pycache__",
    ".mypy_cache", ".ruff_cache", ".venv", "generated", "generated_figures",
    "generated_transformer", "logs", "build", "dist",
}
FORBIDDEN_NAMES = {
    "PRIVATE_MIGRATION_INVENTORY.tsv", "PUBLIC_PACKAGE_BUILD_REPORT.md",
    "RELEASE_CHECKLIST.md", "VERIFICATION_REPORT.md", "run_stdout.txt",
    "audit_checks_stdout.json",
}
LOCAL_MARKERS = (
    "/" + "mnt/data/", "/" + "home/oai/", "/" + "Users/",
    "C:" + "\\\\Users\\\\", "/" + "tmp/", "file:" + "//",
)
PROHIBITED_TEXT = (
    "release" + "-candidate",
    "v1.0.0" + "-rc1",
    "un" + "published release",
    "to be " + "minted",
    "internal " + "handoff",
    "private " + "source",
    "not for " + "distribution",
    "do not " + "publish",
    "open technology" + " disclosure",
    "invention" + " disclosure",
    "OID-" + "CRG",
    "OID-" + "ETQ",
)
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC |DSA |PGP )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{24,}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|client[_-]?secret|access[_-]?token|password)\b"
        r"\s*[:=]\s*[\"'][^\"'\n]{8,}[\"']"
    ),
)
EXPECTED_DOIS = {
    "canonical": "10.5281/zenodo.22048090",
    "foundations": "10.5281/zenodo.22050676",
    "rent": "10.5281/zenodo.22050605",
    "design": "10.5281/zenodo.22050638",
    "closure": "10.5281/zenodo.22050654",
    "transformer": "10.5281/zenodo.22058422",
}
REQUIRED_MARKDOWN = {
    "CODE_OF_CONDUCT.md", "CONTRIBUTING.md", "GOVERNANCE.md", "README.md",
    "SECURITY.md",
}


def _allowed(rel: Path) -> bool:
    value = rel.as_posix()
    if len(rel.parts) == 1 and value in ALLOWED_ROOT_FILES:
        return True
    if value in ALLOWED_EXACT:
        return True
    return any(value.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def _iter_public_files(root: Path, include_transient: bool, errors: list[str]):
    """Walk public payload while pruning checkout and generated directories."""
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        retained_directories = []
        for name in directories:
            path = current_path / name
            rel = path.relative_to(root)
            value = rel.as_posix()
            if name == ".git":
                continue
            if path.is_symlink():
                errors.append(f"symbolic link present: {value}")
                continue
            if name in TRANSIENT_PARTS or name.endswith(".egg-info"):
                if include_transient:
                    errors.append(f"transient directory present: {value}")
                continue
            retained_directories.append(name)
        directories[:] = retained_directories

        for name in filenames:
            path = current_path / name
            rel = path.relative_to(root)
            value = rel.as_posix()
            if path.is_symlink():
                errors.append(f"symbolic link present: {value}")
                continue
            if path.suffix.lower() in {".pyc", ".pyo"} or name == ".DS_Store":
                if include_transient:
                    errors.append(f"transient file present: {value}")
                continue
            yield path, rel


def check(root: Path, *, include_transient: bool = True) -> list[str]:
    errors: list[str] = []
    markdown = []
    public_files = list(_iter_public_files(root, include_transient, errors))
    for path, rel in public_files:
        value = rel.as_posix()
        if not _allowed(rel):
            errors.append(f"non-allowlisted file present: {value}")
        if path.name in FORBIDDEN_NAMES:
            errors.append(f"forbidden file present: {value}")
        if path.suffix.lower() == ".md":
            markdown.append(value)
        if path.suffix.lower() in {".zip", ".tar", ".tgz", ".7z"}:
            errors.append(f"nested archive present: {value}")
    if set(markdown) != REQUIRED_MARKDOWN:
        errors.append(
            "public policy-document set differs from the required set; "
            f"found {sorted(markdown)}"
        )

    scope_source = Path(__file__).resolve()
    for path, rel in public_files:
        if path.resolve() == scope_source:
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or path.stat().st_size > 10 * 1024 * 1024:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lowered = text.lower()
        for marker in LOCAL_MARKERS:
            if marker.lower() in lowered:
                errors.append(f"absolute or local path present in {rel.as_posix()}: {marker}")
        for marker in PROHIBITED_TEXT:
            if marker.lower() in lowered:
                errors.append(f"development-only marker present in {rel.as_posix()}: {marker}")
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                errors.append(f"possible credential or private key present: {rel.as_posix()}")

    for path in list((root / "src").rglob("*.py")) + list((root / "tools").rglob("*.py")) + list((root / "tests").rglob("*.py")) + list((root / "validation").rglob("*.py")):
        text = path.read_text(encoding="utf-8", errors="replace")
        if "SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0" not in text:
            errors.append(f"missing PolyForm SPDX identifier: {path.relative_to(root).as_posix()}")

    required = {
        "README.md", "SECURITY.md", "CONTRIBUTING.md", "CODE_OF_CONDUCT.md",
        "GOVERNANCE.md", "LICENSE", "NOTICE", "LICENSE_SCOPE.tsv",
        "CITATION.cff", "VERSION", "release-metadata.json",
        "manuscripts/DOI_CROSSWALK.tsv",
    }
    for name in sorted(required):
        if not (root / name).is_file():
            errors.append(f"required public file missing: {name}")

    doi_path = root / "manuscripts/DOI_CROSSWALK.tsv"
    if doi_path.is_file():
        with doi_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        observed = {row.get("key", ""): row.get("doi", "") for row in rows}
        if len(rows) != len(EXPECTED_DOIS) or observed != EXPECTED_DOIS:
            errors.append(f"DOI crosswalk differs from the six verified records: {observed}")
        for row in rows:
            doi = row.get("doi", "")
            if not doi.startswith("10.5281/zenodo.") or any(token in doi.lower() for token in ("tbd", "placeholder", "pending")):
                errors.append(f"invalid or unfinished DOI record: {row.get('key', '<unknown>')}")
    return errors
