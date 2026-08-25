# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import csv
import json
from pathlib import Path
import re
import tomllib

import crg_validation


EXPECTED = {
    "canonical": "10.5281/zenodo.22048090",
    "foundations": "10.5281/zenodo.22050676",
    "rent": "10.5281/zenodo.22050605",
    "design": "10.5281/zenodo.22050638",
    "closure": "10.5281/zenodo.22050654",
    "transformer": "10.5281/zenodo.22058422",
}
PUBLISHER = "Semi AI Foundry, LLC"


def test_publication_metadata_is_synchronized() -> None:
    root = Path(__file__).resolve().parents[1]
    with (root / "manuscripts" / "DOI_CROSSWALK.tsv").open(encoding="utf-8", newline="") as handle:
        crosswalk = list(csv.DictReader(handle, delimiter="\t"))
    corpus = json.loads((root / "manuscripts" / "PUBLICATION_CORPUS.json").read_text(encoding="utf-8"))
    release = json.loads((root / "release-metadata.json").read_text(encoding="utf-8"))

    assert {row["key"]: row["doi"] for row in crosswalk} == EXPECTED
    assert {row["key"]: row["doi"] for row in corpus["records"]} == EXPECTED
    assert release["publication_records"] == corpus["records"]
    corpus_by_key = {row["key"]: row for row in corpus["records"]}
    for row in crosswalk:
        assert row["title"] == corpus_by_key[row["key"]]["title"]
        assert row["doi_url"] == f"https://doi.org/{row['doi']}"
        assert corpus_by_key[row["key"]]["doi_url"] == row["doi_url"]
    assert all(row["publisher"] == PUBLISHER for row in corpus["records"])
    assert release["release_date"] is None
    assert release["status"] == "pre-release"

    public_text = "\n".join(
        (root / path).read_text(encoding="utf-8")
        for path in (
            "README.md",
            "CITATION.cff",
            "manuscripts/PUBLICATION_CORPUS.bib",
        )
    )
    for doi in EXPECTED.values():
        assert doi in public_text
    assert "10.5281/zenodo.22051089" not in public_text
    assert "10.5281/zenodo.22051129" not in public_text
    assert "Structural Theorems for Computation-Realization Geometry" in public_text
    assert "Closure Theorems for Computation-Realization Geometry" not in public_text


def test_polyform_required_notice_is_present() -> None:
    root = Path(__file__).resolve().parents[1]
    prefix = "Required Notice: CRG Research and Validation Corpus 1.0.0."
    assert prefix in (root / "LICENSE").read_text(encoding="utf-8")
    assert prefix in (root / "NOTICE").read_text(encoding="utf-8")


def test_version_identity_is_synchronized() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = (root / "VERSION").read_text(encoding="utf-8").strip()
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    release = json.loads((root / "release-metadata.json").read_text(encoding="utf-8"))
    citation = (root / "CITATION.cff").read_text(encoding="utf-8")
    cited = re.search(r'^version: "([^"]+)"$', citation, flags=re.MULTILINE)

    assert expected == "1.0.0"
    assert project["project"]["version"] == expected
    assert release["version"] == expected
    assert cited is not None and cited.group(1) == expected
    assert crg_validation.__version__ == expected
