# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

from pathlib import Path
import re
import tomllib

from crg_validation.environment import report


PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)")


def _pins(lines: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in lines:
        match = PIN.match(line.strip())
        if match:
            name, version = match.groups()
            result[name.lower().replace("_", "-")] = version
    return result


def test_direct_requirements_and_hash_lock_are_synchronized() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    direct_file = (root / "requirements" / "validation.txt").read_text(encoding="utf-8")
    lock_file = (root / "requirements" / "validation.lock").read_text(encoding="utf-8")

    project_pins = _pins(project["project"]["dependencies"])
    direct_pins = _pins(direct_file.splitlines())
    locked_pins = _pins(lock_file.splitlines())

    assert project_pins == direct_pins
    assert project_pins
    assert all(locked_pins.get(name) == version for name, version in project_pins.items())
    assert "--only-binary" in lock_file and ":all:" in lock_file
    assert "--hash=sha256:" in lock_file
    assert "/" + "Users/" not in lock_file
    assert "/" + "tmp/" not in lock_file
    assert "file:" + "//" not in lock_file


def test_auditor_requirement_and_hash_lock_are_synchronized() -> None:
    root = Path(__file__).resolve().parents[1]
    direct_file = (root / "requirements" / "audit.txt").read_text(encoding="utf-8")
    lock_file = (root / "requirements" / "audit.lock").read_text(encoding="utf-8")

    direct_pins = _pins(direct_file.splitlines())
    locked_pins = _pins(lock_file.splitlines())

    assert direct_pins == {"pip-audit": "2.10.1"}
    assert locked_pins.get("pip-audit") == direct_pins["pip-audit"]
    assert "--only-binary" in lock_file and ":all:" in lock_file
    assert "--hash=sha256:" in lock_file
    assert "/" + "Users/" not in lock_file
    assert "/" + "tmp/" not in lock_file
    assert "file:" + "//" not in lock_file


def test_build_requirements_and_hash_lock_are_synchronized() -> None:
    root = Path(__file__).resolve().parents[1]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    direct_file = (root / "requirements" / "build.txt").read_text(encoding="utf-8")
    lock_file = (root / "requirements" / "build.lock").read_text(encoding="utf-8")

    project_pins = _pins(project["build-system"]["requires"])
    direct_pins = _pins(direct_file.splitlines())
    locked_pins = _pins(lock_file.splitlines())

    assert project_pins == direct_pins == {"setuptools": "84.0.0", "wheel": "0.48.0"}
    assert all(locked_pins.get(name) == version for name, version in direct_pins.items())
    assert "--only-binary" in lock_file and ":all:" in lock_file
    assert "--hash=sha256:" in lock_file
    assert "/" + "Users/" not in lock_file
    assert "/" + "tmp/" not in lock_file
    assert "file:" + "//" not in lock_file


def test_current_validation_runtime_is_supported() -> None:
    environment = report()
    assert environment["python_supported"] is True
    assert environment["dependency_match"] is True
