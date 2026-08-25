# Computation-Realization Geometry

## Research and Validation Corpus

> **Communication before commitment.**

This repository is the public scientific companion to **A Communication Theory
of Computation Realization** and **Computation-Realization Geometry (CRG)**. It
contains the finalized manuscript citation record, study-specific reproduction
programs, benchmark inputs, retained numerical results, figures, theorem checks,
negative controls, and integrity records.

The **[CRG Systems implementation](https://github.com/SemiAIFoundry/crg-systems) is distributed separately**. This
repository contains neither the operational system nor reusable system
components, interfaces, deployment assets, or product documentation.

## Manuscript corpus

| Work | DOI |
|---|---|
| *A Communication Theory of Computation Realization: Resource-Conditioned Geometry of Legal Realizations and the Safe-Commitment Law* | [10.5281/zenodo.22048090](https://doi.org/10.5281/zenodo.22048090) |
| *Computation-Realization Geometry: Semantic Interface Laws, Decision-Sufficient Abstraction, and Cost Duality* | [10.5281/zenodo.22050676](https://doi.org/10.5281/zenodo.22050676) |
| *A Generalized Rent Law from Computation-Realization Geometry: Scaling Fixed Points, Hierarchy Phase, and Finite-Scale Diagnostics* | [10.5281/zenodo.22050605](https://doi.org/10.5281/zenodo.22050605) |
| *Designing with Computation-Realization Geometry: Certified Architecture Selection, Technology Matching, and Reference Physical Realization* | [10.5281/zenodo.22050638](https://doi.org/10.5281/zenodo.22050638) |
| *Structural Theorems for Computation-Realization Geometry: Interaction, Safe Commitment, Network Embedding, Conservation, and Scaling* | [10.5281/zenodo.22050654](https://doi.org/10.5281/zenodo.22050654) |
| *Certified Nonseparable Co-Design of a Long-Context Transformer Superblock: A Computation-Realization Geometry Case Study* | [10.5281/zenodo.22058422](https://doi.org/10.5281/zenodo.22058422) |

Machine-readable citations are provided in `CITATION.cff` and `manuscripts/`.

## Repository contents

| Path | Public record |
|---|---|
| `manuscripts/` | Final manuscript DOI crosswalk, BibTeX, and JSON metadata |
| `validation/manuscript_artifacts/` | Manuscript-connected scaling, component, Transformer, EPFL, and integrated checks |
| `validation/theorem_audit/` | Project-led theorem ledger and executable finite or numerical checks |
| `validation/retention/` | Preregistered frozen candidate/scenario ledger, reanalysis, expected outputs, and figure |
| `validation/routed_heterogeneous_integration/` | Deterministic synthetic routed 2.5D/3D compact-model study |
| `third_party/epfl/` | Pinned EPFL benchmark inputs and machine-readable provenance |
| `src/crg_validation/` | Narrow corpus validation, boundary, and integrity tooling |

## Reproduce and verify

CPython 3.12 and 3.13 are supported. Direct dependencies are pinned in
`requirements/validation.txt` and `pyproject.toml`; the fully resolved public
validation environment is hash-locked in `requirements/validation.lock`, and
the packaging toolchain is separately locked in `requirements/build.lock`.
Retained figure fidelity is bound to Matplotlib 3.10.8; rendering with another
minor release is a separately reviewed visual-corpus revision, not a routine
environment substitution.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m crg_validation.cli scope-check

python -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements/validation.lock
python -m pip install --require-hashes -r requirements/build.lock
python -m pip install --no-build-isolation --no-deps .

crg-validation doctor
crg-validation run smoke
crg-validation run all
pytest -q
crg-validation manifest --verify
```

The installed wheel contains the validation CLI, not a duplicate of the
research corpus. Run commands from the repository checkout. From another
directory, pass `--root /path/to/computation-realization-geometry` before the
subcommand or set `CRG_CORPUS_ROOT`.

`crg-validation run all` performs the portable public profile in isolated work
areas: theorem checks and matrix regeneration, exact reanalysis of the frozen
retention ledger, regeneration of the routed compact-model study, and
manuscript-connected consistency checks. The individual study programs remain
available for focused or more exhaustive reruns.

## Evidence boundary

The corpus supports the claims and finite studies identified by the manuscripts
and released artifacts. In particular:

- the theorem ledger is a project-led audit and executable check set, not
  external peer review or a priority determination;
- the retention study is an exact reanalysis of a preregistered finite ledger
  containing 26,624 scenario instances across four workloads, not a claim about
  every industrial design-space exploration process;
- the routed heterogeneous-integration study uses deterministic synthetic
  compact models and is not package signoff, qualification, fabricated
  hardware, or measured product performance;
- retained Transformer and public-netlist records are scoped to their declared
  contracts and benchmark inputs; and
- the public artifacts do not establish completeness outside the released
  candidate, scenario, workload, technology, or model families.

## Licensing

Licenses are artifact-specific:

- project-created validation software, tests, and executable workflows:
  **PolyForm Noncommercial License 1.0.0**;
- project-created documentation, metadata, figures, retained results, and data:
  **Creative Commons Attribution-NonCommercial 4.0 International**;
- EPFL benchmark inputs: upstream **MIT License**;
- manuscripts: the license recorded by each canonical Zenodo manuscript entry.

Commercial use of project-created materials requires a separate written license
from Semi AI Foundry, LLC. The controlling summary is `LICENSE`; exact path
classification is in `LICENSE_SCOPE.tsv`; standard terms are reproduced in
`LICENSES/`.

Required attribution:

> Copyright © 2026 Semi AI Foundry, LLC. Developed by Fitih M. Cinnor for
> semiAIfoundry Research, a research group within Semi AI Foundry, LLC.

This is a source-available research corpus, not an OSI-approved open-source
software release.

## Citation

The preferred citation is:

> Fitih M. Cinnor, *A Communication Theory of Computation Realization:
> Resource-Conditioned Geometry of Legal Realizations and the Safe-Commitment
> Law*, 2026. DOI: 10.5281/zenodo.22048090.

Use the specific manuscript DOI when citing a specialist result. Repository
issues may be used to report a reproduction discrepancy, correction, or
provenance problem. Do not report suspected vulnerabilities in a public issue;
use the private process in `SECURITY.md`. Outside code contributions are not
accepted through this research corpus.
