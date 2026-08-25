#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from pathlib import Path

from crg_validation.manifest import generate
from crg_validation.scope import check

ROOT = Path(__file__).resolve().parents[1]
errors = check(ROOT)
if errors:
    raise SystemExit("public scope check failed:\n" + "\n".join(errors))
manifest, sums = generate(ROOT)
print(manifest)
print(sums)
