# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .environment import report
from .manifest import generate, verify
from .runner import run_all, run_manuscript, run_retention, run_routed_hi, run_theorem_audit, serialize
from .scope import check as scope_check


def root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(prog="crg-validation", description="Validate the public CRG research corpus")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("doctor")
    sub.add_parser("scope-check")

    run = sub.add_parser("run")
    run.add_argument("target", choices=["smoke", "theorem-audit", "retention", "routed-hi", "manuscript", "all"])
    run.add_argument("--keep", action="store_true")

    manifest = sub.add_parser("manifest")
    manifest.add_argument("--verify", action="store_true")

    args = parser.parse_args()
    if args.command == "doctor":
        print(json.dumps(report(), indent=2, sort_keys=True)); return 0
    if args.command == "scope-check":
        errors = scope_check(root())
        print(json.dumps({"status": "pass" if not errors else "fail", "errors": errors}, indent=2))
        return 0 if not errors else 1
    if args.command == "manifest":
        if args.verify:
            errors = verify(root())
            print(json.dumps({"status": "pass" if not errors else "fail", "errors": errors}, indent=2))
            return 0 if not errors else 1
        paths = generate(root())
        print("\n".join(str(path) for path in paths)); return 0

    if args.target == "smoke":
        errors = scope_check(root())
        results = [{"key": "scope", "status": "pass" if not errors else "fail", "details": errors}]
        print(json.dumps(results, indent=2)); return 0 if not errors else 1
    if args.target == "theorem-audit": results = [run_theorem_audit(args.keep)]
    elif args.target == "retention": results = [run_retention(args.keep)]
    elif args.target == "routed-hi": results = [run_routed_hi(args.keep)]
    elif args.target == "manuscript": results = [run_manuscript(args.keep)]
    else: results = run_all(args.keep)
    print(serialize(results))
    return 0 if all(result.status == "pass" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
