#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""RCCG Northstar Transformer model v3.1: Round-5 semantic closure.

This file intentionally preserves ``northstar_transformer_model_v2.py``.
Version 3 makes the workload, causal-execution, peak-throughput, fabric, and
schedule contracts explicit.  It reports exact results *under those declared
contracts*; it is not a production-PPA claim.

Version 3.1 replaces the monotonicity-assuming context root search with a
complete integer dominance audit.  It reports every S-vs-U interval, the
earliest S-no-slower point, and the first point sustained through the declared
audit horizon separately.

Named default contract
----------------------
``llama31-gqa-block-rounded-causal128-dense-bf16-shared-package-aggregate-v1``

* Llama-3.1-8B-class GQA projections (32 query heads, 8 KV heads);
* block-rounded causal attention with 128-token square tensor tiles;
* published dense BF16 peak coordinate of 5 PFLOP/s (the 10-PF coordinate is
  retained only as a structured-sparse sensitivity);
* one conserved aggregate package-payload budget shared by split-stage
  collectives and the semantic inter-stage bridge.

The auxiliary operations outside the two tensor contractions (softmax, RoPE,
RMSNorm, residuals, SwiGLU elementwise work, control, and synchronization) are
typed exclusions.  Consequently the reported operation coordinate is
``modeled_tensor_matmul_flops``, not "full exact attention FLOPs."
"""
from __future__ import annotations

import argparse
import csv
import functools
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

MIB = 2**20
GIB = 2**30
DOMINANCE_AUDIT_MAX_CONTEXT = 1048576
DOMINANCE_SCAN_CHUNK = 65536


class PairPolicy(str, Enum):
    """Declared physical tensor-pair execution policy for causal prefill."""

    DENSE_PADDED = "dense-padded"
    IDEAL_CAUSAL = "ideal-causal"
    BLOCK_ROUNDED = "block-rounded-causal"


class PeakBasis(str, Enum):
    """Which published peak coordinate prices modeled tensor work."""

    DENSE = "dense-bf16"
    SPARSE = "sparse-bf16-sensitivity"


class PackageAccounting(str, Enum):
    """How internal collectives and the semantic bridge consume fabric."""

    SHARED = "shared-package-aggregate"
    SEPARATE = "separate-internal-local-fabric"


@dataclass(frozen=True)
class Workload:
    name: str = "Llama-3.1-8B-class-GQA"
    batch: int = 1
    d: int = 4096
    d_ff: int = 14336
    heads: int = 32
    kv_heads: int = 8
    layers: int = 32
    bytes_per_element: int = 2
    max_context: int = 131072


@dataclass(frozen=True)
class System:
    compute_chiplets: int = 32
    hbm_stacks: int = 8
    dense_bf16_flops_s: float = 5e15
    sparse_bf16_flops_s: float = 10e15
    raw_hbm_bytes_s: float = 9.6e12
    effective_hbm_bytes_s: float = 8e12
    local_fabric_bytes_s: float = 4e12
    fast_state_mib_per_chiplet: float = 32.0


@dataclass(frozen=True)
class ExecutionContract:
    pair_policy: PairPolicy = PairPolicy.BLOCK_ROUNDED
    causal_tile_tokens: int = 128


@dataclass(frozen=True)
class PeakContract:
    basis: PeakBasis = PeakBasis.DENSE


@dataclass(frozen=True)
class FlatCoordinates:
    specialized_utilization: float = 0.893
    specialized_over_flash_speed_ratio: float = 4.1
    flash_hbm_multiplier: float = 16.0
    evidence: str = (
        "Imported FlatAttention maxima from a different MHA many-PE platform; "
        "treated as swept portability hypotheses, not product guarantees."
    )


@dataclass(frozen=True)
class ModelContract:
    name: str = (
        "llama31-gqa-block-rounded-causal128-dense-bf16-"
        "shared-package-aggregate-v1"
    )
    execution: ExecutionContract = ExecutionContract()
    peak: PeakContract = PeakContract()
    package_accounting: PackageAccounting = PackageAccounting.SHARED
    eta_universal: float = 0.80
    hybrid_capability_tax: float = 0.02


@dataclass(frozen=True)
class AttentionMode:
    name: str
    fast_state_mib: float
    utilization: float
    hbm_multiplier: float
    d2d_multiplier: float
    evidence: str


@dataclass(frozen=True)
class MLPMode:
    name: str
    fast_state_mib: float
    utilization: float
    hbm_multiplier: float
    d2d_multiplier: float
    evidence: str


@dataclass(frozen=True)
class Package:
    name: str
    payload_bytes_s: float
    energy_pj_bit: float
    capability: str
    evidence: str


@dataclass
class Candidate:
    architecture: str
    technology: str
    workload_kind: str
    context: int
    contract_name: str
    initiation_interval_s: float
    latency_s: float
    attention_s: float
    mlp_s: float
    bridge_s: float
    attention_chiplets: int
    mlp_chiplets: int
    attention_hbm_stacks: int
    mlp_hbm_stacks: int
    attention_mode: str
    mlp_mode: str
    schedule: str
    bottleneck: str
    bridge_bytes: float
    package_bytes_per_interval: float
    link_energy_j: float
    fast_state_mib: float
    compute_share_sum: float
    hbm_share_sum: float
    fabric_share_sum: float
    allocation_witnesses_tested: int
    notes: str


WORKLOAD = Workload()
SYSTEM = System()
DEFAULT_CONTRACT = ModelContract()
DEFAULT_FLAT = FlatCoordinates()

MLP_MODES = (
    MLPMode(
        "2D-low-replication", 8.0, 0.85, 1.0, 1.0,
        "declared constructive dense-GEMM mode",
    ),
    MLPMode(
        "2.5D-c4", 24.0, 0.85, 0.5, 0.5,
        "declared c=4 replication mode calibrated to prior certified GEMM frontier",
    ),
)

PACKAGES = (
    Package(
        "UCIe-S-declared", 64e9, 0.50, "standard-package aggregate payload",
        "declared cluster calibrated to UCIe rate classes",
    ),
    Package(
        "UCIe-A32-declared", 256e9, 0.25, "advanced-package aggregate payload",
        "declared cluster calibrated to UCIe rate classes",
    ),
    Package(
        "UCIe-A64-declared", 512e9, 0.25, "advanced-package aggregate payload",
        "declared cluster calibrated to the UCIe 64-GT/s class",
    ),
    Package(
        "3D-vertical-declared", 4e12, 0.05, "hybrid-bonded vertical aggregate payload",
        "declared sensitivity point; not an official UCIe throughput guarantee",
    ),
)


def attention_modes(flat: FlatCoordinates = DEFAULT_FLAT) -> Tuple[AttentionMode, ...]:
    if not (0.0 < flat.specialized_utilization <= 1.0):
        raise ValueError("specialized utilization must be in (0,1]")
    if flat.specialized_over_flash_speed_ratio < 1.0:
        raise ValueError("specialized/flash speed ratio must be >= 1")
    if flat.flash_hbm_multiplier < 1.0:
        raise ValueError("flash HBM multiplier must be >= 1")
    return (
        AttentionMode(
            "Flash-local",
            8.0,
            flat.specialized_utilization / flat.specialized_over_flash_speed_ratio,
            flat.flash_hbm_multiplier,
            1.0,
            "FlatAttention portability hypothesis: reported relative maxima",
        ),
        AttentionMode(
            "Flat-collective",
            28.0,
            flat.specialized_utilization,
            1.0,
            0.5,
            "28-MiB declared tile plus swept imported utilization",
        ),
    )


def selected_peak_flops_s(
    contract: ModelContract = DEFAULT_CONTRACT,
    system: System = SYSTEM,
) -> float:
    if contract.peak.basis == PeakBasis.DENSE:
        return system.dense_bf16_flops_s
    if contract.peak.basis == PeakBasis.SPARSE:
        return system.sparse_bf16_flops_s
    raise ValueError(f"unsupported peak basis: {contract.peak.basis}")


def executed_attention_pairs(
    n: int,
    execution: ExecutionContract = DEFAULT_CONTRACT.execution,
) -> int:
    """Return physically executed QK/AV pairs under the declared policy."""
    if n <= 0:
        raise ValueError("context length must be positive")
    if execution.pair_policy == PairPolicy.DENSE_PADDED:
        return n * n
    if execution.pair_policy == PairPolicy.IDEAL_CAUSAL:
        return n * (n + 1) // 2
    if execution.pair_policy == PairPolicy.BLOCK_ROUNDED:
        block = execution.causal_tile_tokens
        if block <= 0:
            raise ValueError("causal tile size must be positive")
        tiles = math.ceil(n / block)
        # The declared kernel pads both Q and K to complete square tiles and
        # executes every tile on and below the block-causal diagonal.
        return block * block * tiles * (tiles + 1) // 2
    raise ValueError(f"unsupported pair policy: {execution.pair_policy}")


def workload_obligations(
    n: int,
    kind: str,
    contract: ModelContract = DEFAULT_CONTRACT,
    workload: Workload = WORKLOAD,
) -> Dict[str, float]:
    """Typed tensor-work and traffic obligations for one Transformer block."""
    w = workload
    b, d, dff, batch = w.bytes_per_element, w.d, w.d_ff, w.batch
    kv_ratio = w.kv_heads / w.heads
    d_kv = d * kv_ratio
    weight_att = (2.0 + 2.0 * kv_ratio) * d * d * b
    weight_mlp = 3.0 * d * dff * b
    projection_per_token = (4.0 + 4.0 * kv_ratio) * d * d

    if kind == "prefill":
        semantic_pairs = n * (n + 1) // 2
        physical_pairs = executed_attention_pairs(n, contract.execution)
        activation = batch * n * d * b
        projection_flops = batch * n * projection_per_token
        pair_flops = 4.0 * batch * physical_pairs * d
        f_att = projection_flops + pair_flops
        f_mlp = 6.0 * batch * n * d * dff
        q_att_base = weight_att + 4.0 * activation
        q_mlp_base = weight_mlp + 6.0 * activation
        bridge = 2.0 * activation
        d2d = 2.0 * activation
    elif kind == "decode":
        semantic_pairs = n
        physical_pairs = n
        activation = batch * d * b
        projection_flops = batch * projection_per_token
        pair_flops = 4.0 * batch * physical_pairs * d
        f_att = projection_flops + pair_flops
        f_mlp = 6.0 * batch * d * dff
        compulsory_kv = 2.0 * batch * n * d_kv * b
        q_att_base = weight_att + compulsory_kv + 4.0 * activation
        q_mlp_base = weight_mlp + 6.0 * activation
        bridge = 2.0 * activation
        d2d = 2.0 * activation
    else:
        raise ValueError(f"unsupported workload kind: {kind}")

    return {
        "activation_bytes": float(activation),
        "attention_projection_flops": float(projection_flops),
        "attention_pair_flops": float(pair_flops),
        "attention_flops": float(f_att),
        "mlp_flops": float(f_mlp),
        "semantic_attention_pairs": float(semantic_pairs),
        "executed_attention_pairs": float(physical_pairs),
        "attention_hbm_base": float(q_att_base),
        "mlp_hbm_base": float(q_mlp_base),
        "bridge_bytes": float(bridge),
        "internal_d2d_base": float(d2d),
        "attention_weight_bytes": float(weight_att),
        "mlp_weight_bytes": float(weight_mlp),
    }


def _components(
    flops: float,
    hbm_bytes: float,
    d2d_bytes: float,
    utilization: float,
    contract: ModelContract,
    *,
    peak_scale: float = 1.0,
    hbm_scale: float = 1.0,
    fabric_bytes_s: float | None = None,
) -> Dict[str, float]:
    peak = selected_peak_flops_s(contract) * peak_scale * utilization
    hbm = SYSTEM.effective_hbm_bytes_s * hbm_scale
    fabric = SYSTEM.local_fabric_bytes_s if fabric_bytes_s is None else fabric_bytes_s
    return {
        "compute": flops / peak,
        "hbm": hbm_bytes / hbm,
        "fabric": d2d_bytes / fabric,
    }


def _allocated_component(value: float, share: float) -> float:
    return math.inf if share <= 0.0 else value / share


def _argmax_label(items: Mapping[str, float]) -> str:
    return max(items, key=items.__getitem__)


def _candidate_key(
    candidate: Candidate,
) -> Tuple[float, float, str, str, int, int]:
    return (
        candidate.initiation_interval_s,
        candidate.latency_s,
        candidate.architecture,
        candidate.schedule,
        candidate.attention_chiplets,
        candidate.attention_hbm_stacks,
    )


def best(items: Iterable[Candidate]) -> Candidate:
    materialized = list(items)
    if not materialized:
        raise ValueError("candidate set is empty")
    return min(materialized, key=_candidate_key)


def allocation_optimum_summary(
    candidates: Iterable[Candidate],
    selected: Candidate,
    *,
    atol: float = 1e-15,
) -> Dict[str, object]:
    """Report the full II-optimal allocation set and deterministic tie-break."""
    materialized = list(candidates)
    ties = [
        c for c in materialized
        if abs(c.initiation_interval_s - selected.initiation_interval_s) <= atol
    ]
    compute = sorted({
        (c.attention_chiplets, c.mlp_chiplets) for c in ties
    })
    hbm = sorted({
        (c.attention_hbm_stacks, c.mlp_hbm_stacks) for c in ties
    })
    return {
        "objective": "minimum steady-state initiation interval",
        "optimal_witness_count": len(ties),
        "optimal_compute_partitions": [list(x) for x in compute],
        "optimal_hbm_partitions": [list(x) for x in hbm],
        "reported_witness": {
            "attention_chiplets": selected.attention_chiplets,
            "mlp_chiplets": selected.mlp_chiplets,
            "attention_hbm_stacks": selected.attention_hbm_stacks,
            "mlp_hbm_stacks": selected.mlp_hbm_stacks,
        },
        "tie_break": (
            "minimum latency, then architecture/schedule and lexicographic "
            "attention compute/HBM allocation"
        ),
    }


def homogeneous_candidates(
    n: int,
    kind: str,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
    fast_state_mib: float = SYSTEM.fast_state_mib_per_chiplet,
    *,
    include_continuous_pipeline_envelope: bool = False,
) -> List[Candidate]:
    """Static fused H.

    The registered certificate uses sequential full-resource reuse.  An
    optional continuous-resource chunk envelope can be emitted as a lower-bound
    sensitivity, but is excluded from architecture competition because no
    integer/common-allocation streaming implementation is declared.
    """
    o = workload_obligations(n, kind, contract)
    out: List[Candidate] = []
    for am in attention_modes(flat):
        for fm in MLP_MODES:
            if am.fast_state_mib + fm.fast_state_mib > fast_state_mib + 1e-12:
                continue
            ac = _components(
                o["attention_flops"],
                o["attention_hbm_base"] * am.hbm_multiplier,
                o["internal_d2d_base"] * am.d2d_multiplier,
                am.utilization,
                contract,
            )
            fc = _components(
                o["mlp_flops"],
                o["mlp_hbm_base"] * fm.hbm_multiplier,
                o["internal_d2d_base"] * fm.d2d_multiplier,
                fm.utilization,
                contract,
            )
            ta, tf = max(ac.values()), max(fc.values())
            sequential = ta + tf
            out.append(Candidate(
                "H-static-fused", "local-fabric", kind, n, contract.name,
                sequential, sequential, ta, tf, 0.0,
                SYSTEM.compute_chiplets, SYSTEM.compute_chiplets,
                SYSTEM.hbm_stacks, SYSTEM.hbm_stacks,
                am.name, fm.name, "sequential-full-resource-reuse",
                f"A:{_argmax_label(ac)}+F:{_argmax_label(fc)}",
                0.0, 0.0, 0.0, am.fast_state_mib + fm.fast_state_mib,
                1.0, 1.0, 1.0, 1,
                "Stages reuse the complete compute/HBM/fabric budgets in time; "
                "no simultaneous full-peak claim.",
            ))

            if not include_continuous_pipeline_envelope:
                continue

            # If each resource can be divided independently and continuously,
            # the conserved lower envelope is max_j(a_j+f_j), not
            # max(T_A,T_F) at two full peaks.  This is not a registered
            # constructive witness without an integer/common-allocation stream.
            totals = {resource: ac[resource] + fc[resource] for resource in ac}
            pipe_ii = max(totals.values())
            shares_a = {
                resource: ac[resource] / totals[resource] if totals[resource] else 0.5
                for resource in totals
            }
            shares_f = {resource: 1.0 - shares_a[resource] for resource in totals}
            pipe_ta = max(
                _allocated_component(ac[resource], shares_a[resource])
                for resource in totals
            )
            pipe_tf = max(
                _allocated_component(fc[resource], shares_f[resource])
                for resource in totals
            )
            out.append(Candidate(
                "H-static-fused", "local-fabric", kind, n, contract.name,
                pipe_ii, pipe_ta + pipe_tf, pipe_ta, pipe_tf, 0.0,
                round(SYSTEM.compute_chiplets * shares_a["compute"]),
                round(SYSTEM.compute_chiplets * shares_f["compute"]),
                round(SYSTEM.hbm_stacks * shares_a["hbm"]),
                round(SYSTEM.hbm_stacks * shares_f["hbm"]),
                am.name, fm.name, "chunk-pipeline-continuous-resource-lower-envelope",
                _argmax_label(totals), 0.0, 0.0, 0.0,
                am.fast_state_mib + fm.fast_state_mib,
                shares_a["compute"] + shares_f["compute"],
                shares_a["hbm"] + shares_f["hbm"],
                shares_a["fabric"] + shares_f["fabric"],
                1,
                "Conserved continuous-resource lower-envelope sensitivity. "
                "Excluded from class competition until an integer/common-"
                "allocation chunk schedule is declared.",
            ))
    return out


def universal_fused_candidates(
    n: int,
    kind: str,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
    fast_state_mib: float = SYSTEM.fast_state_mib_per_chiplet,
) -> List[Candidate]:
    """Time-multiplexed universal fused U; full resources are reused in time."""
    o = workload_obligations(n, kind, contract)
    out: List[Candidate] = []
    for am in attention_modes(flat):
        for fm in MLP_MODES:
            if max(am.fast_state_mib, fm.fast_state_mib) > fast_state_mib + 1e-12:
                continue
            att_util = contract.eta_universal if am.name == "Flat-collective" else am.utilization
            ac = _components(
                o["attention_flops"],
                o["attention_hbm_base"] * am.hbm_multiplier,
                o["internal_d2d_base"] * am.d2d_multiplier,
                att_util,
                contract,
            )
            fc = _components(
                o["mlp_flops"],
                o["mlp_hbm_base"] * fm.hbm_multiplier,
                o["internal_d2d_base"] * fm.d2d_multiplier,
                fm.utilization,
                contract,
            )
            ta, tf = max(ac.values()), max(fc.values())
            total = ta + tf
            out.append(Candidate(
                "U-universal-fused", "local-fabric", kind, n, contract.name,
                total, total, ta, tf, 0.0,
                SYSTEM.compute_chiplets, SYSTEM.compute_chiplets,
                SYSTEM.hbm_stacks, SYSTEM.hbm_stacks,
                f"{am.name}@eta={att_util:.6f}", fm.name,
                "sequential-full-resource-reuse",
                f"A:{_argmax_label(ac)}+F:{_argmax_label(fc)}",
                0.0, 0.0, 0.0, max(am.fast_state_mib, fm.fast_state_mib),
                1.0, 1.0, 1.0, 1,
                "Shared fast-state pool; compute/HBM/fabric are conserved by "
                "sequential full-resource reuse.",
            ))
    return out


def _split_candidate(
    n: int,
    kind: str,
    package: Package,
    contract: ModelContract,
    flat: FlatCoordinates,
    pa: int,
    ha: int,
    *,
    architecture: str = "S-heterogeneous-split",
    attention_peak_scale: float = 1.0,
    mlp_peak_scale: float = 1.0,
    attention_hbm_scale: float = 1.0,
    attention_utilization: float | None = None,
    attention_hbm_multiplier: float | None = None,
    attention_d2d_multiplier: float | None = None,
    mlp_d2d_multiplier: float | None = None,
    attention_mode_name: str | None = None,
    fast_state_mib: float | None = None,
    witnesses_tested: int = 217,
) -> Candidate:
    o = workload_obligations(n, kind, contract)
    am = attention_modes(flat)[1]
    fm = MLP_MODES[1]
    pf = SYSTEM.compute_chiplets - pa
    hf = SYSTEM.hbm_stacks - ha
    cshare_a, cshare_f = pa / SYSTEM.compute_chiplets, pf / SYSTEM.compute_chiplets
    hshare_a, hshare_f = ha / SYSTEM.hbm_stacks, hf / SYSTEM.hbm_stacks
    ahbm = am.hbm_multiplier if attention_hbm_multiplier is None else attention_hbm_multiplier
    ad2d = am.d2d_multiplier if attention_d2d_multiplier is None else attention_d2d_multiplier
    fd2d = fm.d2d_multiplier if mlp_d2d_multiplier is None else mlp_d2d_multiplier

    autil = am.utilization if attention_utilization is None else attention_utilization
    ac_full = _components(
        o["attention_flops"], o["attention_hbm_base"] * ahbm,
        o["internal_d2d_base"] * ad2d, autil, contract,
        peak_scale=attention_peak_scale, hbm_scale=attention_hbm_scale,
        fabric_bytes_s=package.payload_bytes_s
        if contract.package_accounting == PackageAccounting.SHARED else None,
    )
    fc_full = _components(
        o["mlp_flops"], o["mlp_hbm_base"] * fm.hbm_multiplier,
        o["internal_d2d_base"] * fd2d, fm.utilization, contract,
        peak_scale=mlp_peak_scale,
        fabric_bytes_s=package.payload_bytes_s
        if contract.package_accounting == PackageAccounting.SHARED else None,
    )
    stage_terms = {
        "attention-compute": _allocated_component(ac_full["compute"], cshare_a),
        "attention-hbm": _allocated_component(ac_full["hbm"], hshare_a),
        "mlp-compute": _allocated_component(fc_full["compute"], cshare_f),
        "mlp-hbm": _allocated_component(fc_full["hbm"], hshare_f),
    }
    tb = o["bridge_bytes"] / package.payload_bytes_s

    if contract.package_accounting == PackageAccounting.SHARED:
        package_bytes = (
            o["internal_d2d_base"] * ad2d
            + o["internal_d2d_base"] * fd2d
            + o["bridge_bytes"]
        )
        package_floor = package_bytes / package.payload_bytes_s
        ii_terms = {**stage_terms, "shared-package": package_floor}
        isolated_ta = max(
            stage_terms["attention-compute"],
            stage_terms["attention-hbm"],
            ac_full["fabric"],
        )
        isolated_tf = max(
            stage_terms["mlp-compute"],
            stage_terms["mlp-hbm"],
            fc_full["fabric"],
        )
        fabric_sum = 1.0
    else:
        # Internal collectives use an explicitly separate local fabric, divided
        # in the same physical proportions as the chiplet pools.
        a_local = _allocated_component(ac_full["fabric"], cshare_a)
        f_local = _allocated_component(fc_full["fabric"], cshare_f)
        ii_terms = {
            **stage_terms,
            "attention-local-fabric": a_local,
            "mlp-local-fabric": f_local,
            "bridge-package": tb,
        }
        package_bytes = o["bridge_bytes"]
        isolated_ta = max(stage_terms["attention-compute"], stage_terms["attention-hbm"], a_local)
        isolated_tf = max(stage_terms["mlp-compute"], stage_terms["mlp-hbm"], f_local)
        fabric_sum = cshare_a + cshare_f

    ii = max(ii_terms.values())
    energy = package_bytes * 8.0 * package.energy_pj_bit * 1e-12
    return Candidate(
        architecture, package.name, kind, n, contract.name,
        ii, isolated_ta + isolated_tf + tb, isolated_ta, isolated_tf, tb,
        pa, pf, ha, hf,
        attention_mode_name or am.name, fm.name,
        "steady-pipeline-disjoint-conserved-pools",
        _argmax_label(ii_terms), o["bridge_bytes"], package_bytes, energy,
        fast_state_mib if fast_state_mib is not None else max(am.fast_state_mib, fm.fast_state_mib),
        cshare_a + cshare_f, hshare_a + hshare_f, fabric_sum,
        witnesses_tested,
        "All 31x7 legal integer compute/HBM allocations are re-swept at "
        "each (N, workload, technology, contract) point.",
    )


def split_candidates(
    n: int,
    kind: str,
    package: Package,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
) -> List[Candidate]:
    out: List[Candidate] = []
    witnesses = (SYSTEM.compute_chiplets - 1) * (SYSTEM.hbm_stacks - 1)
    for pa in range(1, SYSTEM.compute_chiplets):
        for ha in range(1, SYSTEM.hbm_stacks):
            out.append(_split_candidate(
                n, kind, package, contract, flat, pa, ha,
                witnesses_tested=witnesses,
            ))
    return out


def memory_centric_candidate(
    n: int,
    kind: str,
    package: Package,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
) -> Candidate:
    return _split_candidate(
        n, kind, package, contract, flat, 8, 5,
        architecture="M-memory-centric-attention",
        attention_peak_scale=0.25,
        attention_hbm_scale=1.2,
        attention_utilization=0.75,
        attention_hbm_multiplier=0.5,
        attention_d2d_multiplier=0.25,
        attention_mode_name="near-HBM-attention",
        fast_state_mib=24.0,
        witnesses_tested=1,
    )


def vertical_candidates(
    n: int,
    kind: str,
    package: Package,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
) -> List[Candidate]:
    out: List[Candidate] = []
    witnesses = (SYSTEM.compute_chiplets - 1) * (SYSTEM.hbm_stacks - 1)
    for pa in range(1, SYSTEM.compute_chiplets):
        for ha in range(1, SYSTEM.hbm_stacks):
            out.append(_split_candidate(
                n, kind, package, contract, flat, pa, ha,
                architecture="V-vertical-specialized",
                attention_peak_scale=2.0,
                mlp_peak_scale=2.0,
                attention_hbm_multiplier=0.5,
                attention_d2d_multiplier=0.1,
                mlp_d2d_multiplier=0.1,
                attention_mode_name="vertical-flat",
                fast_state_mib=64.0,
                witnesses_tested=witnesses,
            ))
    return out


def hybrid_candidate(
    n: int,
    kind: str,
    package: Package,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
) -> Candidate:
    """A legitimate split/universal fallback class, not a fictitious new schedule.

    The physical class contains a package bypass plus reconfigurable fallback
    on the split pools.  It may choose either already-registered U or S mode,
    and pays a declared capability/routing tax.  Therefore it is an adversarial
    registry closure but cannot beat the untaxed union of its constituent
    witnesses without a new concurrent schedule.
    """
    u = best(universal_fused_candidates(n, kind, contract, flat))
    s = best(split_candidates(n, kind, package, contract, flat))
    source = min((u, s), key=_candidate_key)
    scale = 1.0 / (1.0 - contract.hybrid_capability_tax)
    return Candidate(
        "X-dual-mode-split-with-universal-fallback",
        package.name,
        kind,
        n,
        contract.name,
        source.initiation_interval_s * scale,
        source.latency_s * scale,
        source.attention_s * scale,
        source.mlp_s * scale,
        source.bridge_s * scale,
        source.attention_chiplets,
        source.mlp_chiplets,
        source.attention_hbm_stacks,
        source.mlp_hbm_stacks,
        source.attention_mode,
        source.mlp_mode,
        f"dual-mode-envelope-from-{source.architecture}",
        source.bottleneck,
        source.bridge_bytes,
        source.package_bytes_per_interval,
        source.link_energy_j,
        max(28.0, 24.0),
        source.compute_share_sum,
        source.hbm_share_sum,
        source.fabric_share_sum,
        source.allocation_witnesses_tested,
        f"Legitimate U/S fallback hardware with declared "
        f"{100*contract.hybrid_capability_tax:.1f}% whole-path capability tax; "
        "included to challenge class closure, not to manufacture a new win.",
    )


def scenario_candidates(
    n: int,
    kind: str,
    package: Package,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
    *,
    allow_vertical: bool = False,
    include_hybrid: bool = True,
) -> List[Candidate]:
    candidates = [
        best(homogeneous_candidates(n, kind, contract, flat)),
        best(universal_fused_candidates(n, kind, contract, flat)),
        best(split_candidates(n, kind, package, contract, flat)),
        memory_centric_candidate(n, kind, package, contract, flat),
    ]
    if include_hybrid:
        candidates.append(hybrid_candidate(n, kind, package, contract, flat))
    if allow_vertical:
        candidates.append(best(vertical_candidates(n, kind, package, contract, flat)))
    return candidates


def nested_layout_movement(p: int, c1: int, c2: int) -> float:
    """Closed form, valid only when the sequence partitions are nested."""
    if p % c1 or p % c2:
        raise ValueError("partition factors must divide P")
    if c1 % c2 and c2 % c1:
        raise ValueError("closed form requires C|C' or C'|C")
    return 1.0 - min(c1, c2) / max(c1, c2)


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def general_layout_retained_fraction(p: int, c1: int, c2: int) -> float:
    """Exact aligned-layout retention as a maximum-weight assignment.

    Each of the P source shards is relabeled bijectively to one of the P target
    shards.  Edge weight is the 2-D intersection area of the two aligned
    rectangles.  The assignment sum is the globally retained data fraction.
    This remains valid when C and C' are not nested.
    """
    if p <= 0 or c1 <= 0 or c2 <= 0 or p % c1 or p % c2:
        raise ValueError("positive C and C' must divide positive P")
    t1, t2 = p // c1, p // c2
    weights = np.zeros((p, p), dtype=float)
    source = [(i, j) for i in range(c1) for j in range(t1)]
    target = [(i, j) for i in range(c2) for j in range(t2)]
    for si, (sr, sh) in enumerate(source):
        sr0, sr1 = sr / c1, (sr + 1) / c1
        sh0, sh1 = sh / t1, (sh + 1) / t1
        for ti, (tr, th) in enumerate(target):
            tr0, tr1 = tr / c2, (tr + 1) / c2
            th0, th1 = th / t2, (th + 1) / t2
            weights[si, ti] = (
                _interval_overlap(sr0, sr1, tr0, tr1)
                * _interval_overlap(sh0, sh1, th0, th1)
            )
    rows, cols = linear_sum_assignment(-weights)
    return float(weights[rows, cols].sum())


def aligned_layout_movement(p: int, c1: int, c2: int) -> float:
    return 1.0 - general_layout_retained_fraction(p, c1, c2)


def universal_efficiency_boundary(
    n: int,
    package: Package,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
) -> float | None:
    """Eta_U at which best U ties best S, if bracketed in (0,1]."""
    split = best(split_candidates(n, "prefill", package, contract, flat)).initiation_interval_s

    def delta(eta: float) -> float:
        c = replace(contract, eta_universal=eta)
        u = best(universal_fused_candidates(n, "prefill", c, flat)).initiation_interval_s
        return u - split

    lo, hi = 1e-4, 1.0
    if delta(lo) <= 0.0 or delta(hi) >= 0.0:
        return None
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if delta(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return hi


def _prefill_u_s_times_array(
    contexts: np.ndarray,
    package: Package,
    contract: ModelContract,
    flat: FlatCoordinates,
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact vectorized U/S initiation intervals for positive integer contexts.

    This is the same declared algebra as ``universal_fused_candidates`` and
    ``split_candidates`` specialized to prefill.  It exists so a complete
    integer audit through one million tokens does not instantiate 217 Python
    candidate objects at every context.
    """
    raw_contexts = np.asarray(contexts)
    if raw_contexts.ndim != 1 or raw_contexts.size == 0:
        raise ValueError(
            "contexts must be a nonempty vector of positive integers"
        )
    if (
        np.issubdtype(raw_contexts.dtype, np.bool_)
        or not np.issubdtype(raw_contexts.dtype, np.integer)
    ):
        raise ValueError(
            "contexts must use an integer dtype; nonintegral values are invalid"
        )
    if (
        np.issubdtype(raw_contexts.dtype, np.unsignedinteger)
        and np.any(raw_contexts > np.iinfo(np.int64).max)
    ):
        raise ValueError("contexts exceed the supported int64 range")
    n_int = raw_contexts.astype(np.int64, copy=False)
    if np.any(n_int <= 0):
        raise ValueError(
            "contexts must be a nonempty vector of positive integers"
        )
    n = n_int.astype(np.float64)
    w = WORKLOAD
    activation = n * w.batch * w.d * w.bytes_per_element
    projection = n * w.batch * (
        4.0 + 4.0 * w.kv_heads / w.heads
    ) * w.d**2

    if contract.execution.pair_policy == PairPolicy.DENSE_PADDED:
        pairs = n_int * n_int
    elif contract.execution.pair_policy == PairPolicy.IDEAL_CAUSAL:
        pairs = n_int * (n_int + 1) // 2
    elif contract.execution.pair_policy == PairPolicy.BLOCK_ROUNDED:
        block = contract.execution.causal_tile_tokens
        if block <= 0:
            raise ValueError("causal tile size must be positive")
        tiles = (n_int + block - 1) // block
        pairs = block * block * tiles * (tiles + 1) // 2
    else:
        raise ValueError(
            f"unsupported pair policy: {contract.execution.pair_policy}"
        )

    f_attention = projection + 4.0 * pairs.astype(np.float64) * w.d
    f_mlp = n * w.batch * 6.0 * w.d * w.d_ff
    weight_attention = (
        (2.0 + 2.0 * w.kv_heads / w.heads)
        * w.d**2
        * w.bytes_per_element
    )
    weight_mlp = 3.0 * w.d * w.d_ff * w.bytes_per_element
    q_attention = weight_attention + 4.0 * activation
    q_mlp = weight_mlp + 6.0 * activation
    d2d = 2.0 * activation
    bridge = 2.0 * activation
    peak = selected_peak_flops_s(contract)

    # U: enumerate the same four legal attention/MLP mode pairs and take the
    # exact lower envelope of their sequential full-resource schedules.
    u_time = np.full(n.shape, np.inf, dtype=np.float64)
    for am in attention_modes(flat):
        for fm in MLP_MODES:
            if max(am.fast_state_mib, fm.fast_state_mib) > (
                SYSTEM.fast_state_mib_per_chiplet + 1e-12
            ):
                continue
            attention_utilization = (
                contract.eta_universal
                if am.name == "Flat-collective"
                else am.utilization
            )
            attention_time = np.maximum.reduce((
                f_attention / (peak * attention_utilization),
                q_attention * am.hbm_multiplier / SYSTEM.effective_hbm_bytes_s,
                d2d * am.d2d_multiplier / SYSTEM.local_fabric_bytes_s,
            ))
            mlp_time = np.maximum.reduce((
                f_mlp / (peak * fm.utilization),
                q_mlp * fm.hbm_multiplier / SYSTEM.effective_hbm_bytes_s,
                d2d * fm.d2d_multiplier / SYSTEM.local_fabric_bytes_s,
            ))
            np.minimum(u_time, attention_time + mlp_time, out=u_time)

    # S: compute/fabric and HBM allocations are independent conserved
    # partitions.  Therefore the 31x7 exhaustive minimum factorizes exactly
    # into a 31-way and a 7-way lower envelope before the final maximum.
    am = attention_modes(flat)[1]
    fm = MLP_MODES[1]
    attention_compute = f_attention / (peak * am.utilization)
    mlp_compute = f_mlp / (peak * fm.utilization)
    attention_hbm = (
        q_attention * am.hbm_multiplier / SYSTEM.effective_hbm_bytes_s
    )
    mlp_hbm = q_mlp * fm.hbm_multiplier / SYSTEM.effective_hbm_bytes_s
    attention_fabric = d2d * am.d2d_multiplier
    mlp_fabric = d2d * fm.d2d_multiplier

    compute_fabric_envelope = np.full(n.shape, np.inf, dtype=np.float64)
    for pa in range(1, SYSTEM.compute_chiplets):
        pf = SYSTEM.compute_chiplets - pa
        terms = [
            attention_compute / (pa / SYSTEM.compute_chiplets),
            mlp_compute / (pf / SYSTEM.compute_chiplets),
        ]
        if contract.package_accounting == PackageAccounting.SEPARATE:
            terms.extend((
                attention_fabric
                / SYSTEM.local_fabric_bytes_s
                / (pa / SYSTEM.compute_chiplets),
                mlp_fabric
                / SYSTEM.local_fabric_bytes_s
                / (pf / SYSTEM.compute_chiplets),
            ))
        allocation_time = np.maximum.reduce(terms)
        np.minimum(
            compute_fabric_envelope, allocation_time,
            out=compute_fabric_envelope,
        )

    hbm_envelope = np.full(n.shape, np.inf, dtype=np.float64)
    for ha in range(1, SYSTEM.hbm_stacks):
        hf = SYSTEM.hbm_stacks - ha
        allocation_time = np.maximum(
            attention_hbm / (ha / SYSTEM.hbm_stacks),
            mlp_hbm / (hf / SYSTEM.hbm_stacks),
        )
        np.minimum(hbm_envelope, allocation_time, out=hbm_envelope)

    if contract.package_accounting == PackageAccounting.SHARED:
        package_floor = (
            attention_fabric + mlp_fabric + bridge
        ) / package.payload_bytes_s
    elif contract.package_accounting == PackageAccounting.SEPARATE:
        package_floor = bridge / package.payload_bytes_s
    else:
        raise ValueError(
            f"unsupported package accounting: {contract.package_accounting}"
        )
    s_time = np.maximum.reduce((
        compute_fabric_envelope,
        hbm_envelope,
        package_floor,
    ))
    return u_time, s_time


@functools.lru_cache(maxsize=64)
def _exact_context_dominance_runs(
    package: Package,
    contract: ModelContract,
    flat: FlatCoordinates,
    min_context: int,
    max_context: int,
) -> Tuple[Tuple[int, int, bool], ...]:
    """Enumerate and run-length encode every integer U/S comparison."""
    if min_context <= 0 or max_context < min_context:
        raise ValueError("require 0 < min_context <= max_context")
    runs: List[Tuple[int, int, bool]] = []
    for low in range(min_context, max_context + 1, DOMINANCE_SCAN_CHUNK):
        high = min(max_context, low + DOMINANCE_SCAN_CHUNK - 1)
        contexts = np.arange(low, high + 1, dtype=np.int64)
        u_time, s_time = _prefill_u_s_times_array(
            contexts, package, contract, flat
        )
        # The declared comparison is exact at the model's float64 arithmetic:
        # ties belong to S because the question is whether S is no slower.
        states = s_time <= u_time
        changes = np.flatnonzero(states[1:] != states[:-1]) + 1
        starts = np.concatenate((np.array([0]), changes))
        stops = np.concatenate((changes, np.array([states.size])))
        for start_index, stop_index in zip(starts, stops):
            start = low + int(start_index)
            end = low + int(stop_index) - 1
            state = bool(states[int(start_index)])
            if runs and runs[-1][2] == state and runs[-1][1] + 1 == start:
                prior = runs[-1]
                runs[-1] = (prior[0], end, state)
            else:
                runs.append((start, end, state))
    return tuple(runs)


def _dominance_analysis_from_runs(
    runs: Sequence[Tuple[int, int, bool]],
    package_name: str,
    contract_name: str,
    pair_policy: str,
    min_context: int,
    max_context: int,
) -> Dict[str, object]:
    if not runs:
        raise ValueError("dominance run set cannot be empty")
    intervals = [
        {
            "start_context": start,
            "end_context": end,
            "relation": (
                "S-no-slower" if s_no_slower else "U-strictly-faster"
            ),
        }
        for start, end, s_no_slower in runs
    ]
    s_intervals = [
        {
            "start_context": start,
            "end_context": end,
        }
        for start, end, s_no_slower in runs
        if s_no_slower
    ]
    monotone = not any(
        left[2] and not right[2]
        for left, right in zip(runs, runs[1:])
    )
    earliest = s_intervals[0]["start_context"] if s_intervals else None
    sustained = (
        runs[-1][0]
        if runs[-1][2] and runs[-1][1] == max_context
        else None
    )
    return {
        "comparison": "best-S initiation interval <= best-U initiation interval",
        "tie_rule": "an exact model tie is assigned to S-no-slower",
        "enumeration_method": (
            "complete float64 evaluation at every integer context; "
            "vectorized in bounded chunks and run-length encoded"
        ),
        "package": package_name,
        "contract_name": contract_name,
        "pair_policy": pair_policy,
        "min_context": min_context,
        "max_context": max_context,
        "integer_contexts_audited": max_context - min_context + 1,
        "dominance_intervals": intervals,
        "s_no_slower_intervals": s_intervals,
        "earliest_s_no_slower_context": earliest,
        "first_sustained_s_no_slower_context": sustained,
        "sustained_scope": (
            "S remains no slower through max_context; no claim beyond the "
            "audited range"
        ),
        "s_no_slower_is_monotone_nondecreasing": monotone,
        "monotonicity_established_by_complete_integer_audit": monotone,
        "transition_count": len(runs) - 1,
    }


def context_dominance_analysis(
    package: Package,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
    *,
    min_context: int = 1,
    max_context: int = WORKLOAD.max_context,
) -> Dict[str, object]:
    """Return the complete integer S-vs-U dominance partition for prefill."""
    runs = _exact_context_dominance_runs(
        package, contract, flat, min_context, max_context
    )
    return _dominance_analysis_from_runs(
        runs,
        package.name,
        contract.name,
        contract.execution.pair_policy.value,
        min_context,
        max_context,
    )


def restrict_context_dominance_analysis(
    analysis: Mapping[str, object],
    max_context: int,
) -> Dict[str, object]:
    """Restrict a complete dominance audit to a smaller common lower range."""
    min_context = int(analysis["min_context"])
    source_max = int(analysis["max_context"])
    if max_context < min_context or max_context > source_max:
        raise ValueError("restricted maximum must lie inside the source audit")
    runs: List[Tuple[int, int, bool]] = []
    for item in analysis["dominance_intervals"]:  # type: ignore[index]
        start = int(item["start_context"])
        if start > max_context:
            break
        end = min(int(item["end_context"]), max_context)
        runs.append((start, end, item["relation"] == "S-no-slower"))
    return _dominance_analysis_from_runs(
        runs,
        str(analysis["package"]),
        str(analysis["contract_name"]),
        str(analysis["pair_policy"]),
        min_context,
        max_context,
    )


def find_monotone_context_overtake(
    package: Package,
    contract: ModelContract = DEFAULT_CONTRACT,
    flat: FlatCoordinates = DEFAULT_FLAT,
    *,
    max_context: int = WORKLOAD.max_context,
) -> int | None:
    """Return the first overtake only for a monotone audited comparison.

    The complete integer audit must first establish that the S-no-slower
    predicate is monotone nondecreasing.  Block-rounded staircases generally
    violate that premise and raise; callers must use
    ``context_dominance_analysis`` and distinguish the earliest from the first
    sustained overtake.
    """
    analysis = context_dominance_analysis(
        package, contract, flat, max_context=max_context
    )
    if not analysis["s_no_slower_is_monotone_nondecreasing"]:
        raise ValueError(
            "S-vs-U dominance is nonmonotone on the audited range; use "
            "context_dominance_analysis()"
        )
    value = analysis["earliest_s_no_slower_context"]
    return None if value is None else int(value)


def _winner_name(
    package: Package,
    contract: ModelContract,
    flat: FlatCoordinates,
    *,
    allow_vertical: bool,
) -> str:
    return best(scenario_candidates(
        16384, "prefill", package, contract, flat,
        allow_vertical=allow_vertical,
    )).architecture


def phase_boundary_distance(
    package: Package,
    contract: ModelContract,
    flat: FlatCoordinates,
    *,
    allow_vertical: bool,
) -> Dict[str, object]:
    """Nearest winner-change boundaries along B_pkg and eta_U axes."""
    current = _winner_name(package, contract, flat, allow_vertical=allow_vertical)

    def winner_at_bandwidth(log_bw: float) -> str:
        p = replace(package, payload_bytes_s=math.exp(log_bw), name="boundary-sweep")
        return _winner_name(p, contract, flat, allow_vertical=allow_vertical)

    log0 = math.log(package.payload_bytes_s)
    grid = np.linspace(math.log(1e9), math.log(16e12), 241)
    bandwidth_boundaries: List[float] = []
    prior_x, prior_w = float(grid[0]), winner_at_bandwidth(float(grid[0]))
    for x in grid[1:]:
        x = float(x)
        w = winner_at_bandwidth(x)
        if w != prior_w and (w == current or prior_w == current):
            lo, hi = prior_x, x
            left_w = prior_w
            for _ in range(55):
                mid = (lo + hi) / 2.0
                if winner_at_bandwidth(mid) == left_w:
                    lo = mid
                else:
                    hi = mid
            bandwidth_boundaries.append(math.exp((lo + hi) / 2.0))
        prior_x, prior_w = x, w
    if bandwidth_boundaries:
        bw_boundary = min(
            bandwidth_boundaries,
            key=lambda b: abs(math.log2(b / package.payload_bytes_s)),
        )
        bw_distance = abs(math.log2(bw_boundary / package.payload_bytes_s))
    else:
        bw_boundary, bw_distance = None, None

    def winner_at_eta(eta: float) -> str:
        return _winner_name(
            package, replace(contract, eta_universal=eta), flat,
            allow_vertical=allow_vertical,
        )

    eta_grid = np.linspace(0.10, 1.0, 181)
    eta_boundaries: List[float] = []
    prior_eta, prior_w = float(eta_grid[0]), winner_at_eta(float(eta_grid[0]))
    for eta in eta_grid[1:]:
        eta = float(eta)
        w = winner_at_eta(eta)
        if w != prior_w and (w == current or prior_w == current):
            lo, hi = prior_eta, eta
            left_w = prior_w
            for _ in range(55):
                mid = (lo + hi) / 2.0
                if winner_at_eta(mid) == left_w:
                    lo = mid
                else:
                    hi = mid
            eta_boundaries.append((lo + hi) / 2.0)
        prior_eta, prior_w = eta, w
    if eta_boundaries:
        eta_boundary = min(eta_boundaries, key=lambda x: abs(x - contract.eta_universal))
        eta_distance = abs(eta_boundary - contract.eta_universal)
    else:
        eta_boundary, eta_distance = None, None

    normalized_candidates: List[Tuple[str, float]] = []
    if bw_distance is not None:
        normalized_candidates.append(("package_payload_log2", bw_distance / math.log2(16e12 / 1e9)))
    if eta_distance is not None:
        normalized_candidates.append(("universal_efficiency", eta_distance / 0.90))
    nearest = min(normalized_candidates, key=lambda x: x[1]) if normalized_candidates else (None, None)
    return {
        "winner": current,
        "package_boundary_bytes_s": bw_boundary,
        "package_boundary_distance_log2_ratio": bw_distance,
        "eta_boundary": eta_boundary,
        "eta_boundary_distance": eta_distance,
        "nearest_normalized_axis": nearest[0],
        "nearest_normalized_distance": nearest[1],
        "search_domain": {
            "package_bytes_s": [1e9, 16e12],
            "eta_universal": [0.10, 1.0],
        },
    }


def certificate_scenario(
    name: str,
    package: Package,
    contract: ModelContract,
    flat: FlatCoordinates,
    *,
    allow_vertical: bool,
    include_boundary: bool = True,
) -> Dict[str, object]:
    candidates = scenario_candidates(
        16384, "prefill", package, contract, flat,
        allow_vertical=allow_vertical,
    )
    ordered = sorted(candidates, key=_candidate_key)
    winner, second = ordered[:2]
    nonhybrid = sorted(
        [c for c in candidates if not c.architecture.startswith("X-")],
        key=_candidate_key,
    )
    runner_nonhybrid = nonhybrid[1] if nonhybrid[0].architecture == winner.architecture else nonhybrid[0]
    result: Dict[str, object] = {
        "scenario": name,
        "package": package.name,
        "winner": winner.architecture,
        "winner_ms": winner.initiation_interval_s * 1e3,
        "second_best": second.architecture,
        "second_best_ms": second.initiation_interval_s * 1e3,
        "all_class_gap_ms": (second.initiation_interval_s - winner.initiation_interval_s) * 1e3,
        "all_class_relative_gap": (
            second.initiation_interval_s / winner.initiation_interval_s - 1.0
        ),
        "nonhybrid_runner_up": runner_nonhybrid.architecture,
        "nonhybrid_gap_ms": (
            runner_nonhybrid.initiation_interval_s - winner.initiation_interval_s
        ) * 1e3,
        "class_times_ms": {
            c.architecture: c.initiation_interval_s * 1e3 for c in candidates
        },
        "winner_signature": asdict(winner),
    }
    if winner.architecture == "S-heterogeneous-split":
        allocation_pool = split_candidates(
            16384, "prefill", package, contract, flat
        )
    elif winner.architecture == "V-vertical-specialized":
        allocation_pool = vertical_candidates(
            16384, "prefill", package, contract, flat
        )
    else:
        allocation_pool = [winner]
    result["winner_allocation_optima"] = allocation_optimum_summary(
        allocation_pool, winner
    )
    if include_boundary:
        result["phase_boundary_distance"] = phase_boundary_distance(
            package, contract, flat, allow_vertical=allow_vertical,
        )
    return result


def contract_with(
    pair_policy: PairPolicy,
    peak_basis: PeakBasis,
    package_accounting: PackageAccounting,
) -> ModelContract:
    block = 128
    name = (
        f"llama31-gqa-{pair_policy.value}"
        f"{block if pair_policy == PairPolicy.BLOCK_ROUNDED else ''}-"
        f"{peak_basis.value}-{package_accounting.value}-v1"
    )
    return ModelContract(
        name=name,
        execution=ExecutionContract(pair_policy, block),
        peak=PeakContract(peak_basis),
        package_accounting=package_accounting,
        eta_universal=0.80,
        hybrid_capability_tax=0.02,
    )


def _flat_sensitivity_rows(contract: ModelContract) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for eta_flat in (0.65, 0.70, 0.75, 0.80, 0.85, 0.893, 0.93):
        for speed_ratio in (2.0, 2.5, 3.0, 3.5, 4.1, 4.5):
            for hbm_ratio in (4.0, 8.0, 12.0, 16.0, 20.0):
                flat = FlatCoordinates(eta_flat, speed_ratio, hbm_ratio)
                sc = certificate_scenario(
                    "A64 FlatAttention portability sweep", PACKAGES[2],
                    contract, flat, allow_vertical=False, include_boundary=False,
                )
                signature = sc["winner_signature"]
                rows.append({
                    "specialized_utilization": eta_flat,
                    "specialized_over_flash_speed_ratio": speed_ratio,
                    "flash_hbm_multiplier": hbm_ratio,
                    "winner": sc["winner"],
                    "winner_ms": sc["winner_ms"],
                    "second_best": sc["second_best"],
                    "all_class_gap_ms": sc["all_class_gap_ms"],
                    "winner_attention_chiplets": signature["attention_chiplets"],
                    "winner_mlp_chiplets": signature["mlp_chiplets"],
                    "winner_attention_hbm_stacks": signature["attention_hbm_stacks"],
                    "winner_mlp_hbm_stacks": signature["mlp_hbm_stacks"],
                    "winner_bottleneck": signature["bottleneck"],
                })
    return rows


def _contract_sensitivity_rows() -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[Dict[str, object]],
]:
    rows: List[Dict[str, object]] = []
    interval_rows: List[Dict[str, object]] = []
    analyses: List[Dict[str, object]] = []
    for policy in PairPolicy:
        for peak in PeakBasis:
            for accounting in PackageAccounting:
                contract = contract_with(policy, peak, accounting)
                standard = certificate_scenario(
                    "standard contract sensitivity", PACKAGES[0], contract,
                    DEFAULT_FLAT, allow_vertical=False, include_boundary=False,
                )
                a64 = certificate_scenario(
                    "A64 contract sensitivity", PACKAGES[2], contract,
                    DEFAULT_FLAT, allow_vertical=False, include_boundary=False,
                )
                vertical = certificate_scenario(
                    "vertical contract sensitivity", PACKAGES[3], contract,
                    DEFAULT_FLAT, allow_vertical=True, include_boundary=False,
                )
                audited = context_dominance_analysis(
                    PACKAGES[0], contract, DEFAULT_FLAT,
                    max_context=DOMINANCE_AUDIT_MAX_CONTEXT,
                )
                declared = restrict_context_dominance_analysis(
                    audited, WORKLOAD.max_context
                )
                boundary = universal_efficiency_boundary(
                    16384, PACKAGES[2], contract, DEFAULT_FLAT
                )
                rows.append({
                    "contract_name": contract.name,
                    "pair_policy": policy.value,
                    "tile_tokens": contract.execution.causal_tile_tokens
                    if policy == PairPolicy.BLOCK_ROUNDED else "",
                    "peak_basis": peak.value,
                    "selected_peak_flops_s": selected_peak_flops_s(contract),
                    "package_accounting": accounting.value,
                    "contract_admissibility": (
                        "primary-dense-workload"
                        if peak == PeakBasis.DENSE
                        else "structured-sparse-sensitivity-only"
                    ),
                    "standard_winner": standard["winner"],
                    "standard_winner_ms": standard["winner_ms"],
                    "standard_gap_ms": standard["all_class_gap_ms"],
                    "a64_winner": a64["winner"],
                    "a64_winner_ms": a64["winner_ms"],
                    "a64_second_best": a64["second_best"],
                    "a64_gap_ms": a64["all_class_gap_ms"],
                    "a64_nonhybrid_gap_ms": a64["nonhybrid_gap_ms"],
                    "vertical_winner": vertical["winner"],
                    "vertical_winner_ms": vertical["winner_ms"],
                    "vertical_gap_ms": vertical["all_class_gap_ms"],
                    "eta_u_boundary": boundary,
                    "dominance_audit_max_context": DOMINANCE_AUDIT_MAX_CONTEXT,
                    "declared_earliest_s_no_slower_context": declared[
                        "earliest_s_no_slower_context"
                    ],
                    "declared_first_sustained_s_no_slower_context": declared[
                        "first_sustained_s_no_slower_context"
                    ],
                    "declared_s_no_slower_is_monotone_nondecreasing": declared[
                        "s_no_slower_is_monotone_nondecreasing"
                    ],
                    "audited_earliest_s_no_slower_context": audited[
                        "earliest_s_no_slower_context"
                    ],
                    "audited_first_sustained_s_no_slower_context": audited[
                        "first_sustained_s_no_slower_context"
                    ],
                    "audited_s_no_slower_is_monotone_nondecreasing": audited[
                        "s_no_slower_is_monotone_nondecreasing"
                    ],
                    "audited_s_no_slower_interval_count": len(
                        audited["s_no_slower_intervals"]
                    ),
                    "audited_s_no_slower_intervals_json": json.dumps(
                        audited["s_no_slower_intervals"],
                        separators=(",", ":"),
                    ),
                })
                analyses.append({
                    "contract_name": contract.name,
                    "pair_policy": policy.value,
                    "peak_basis": peak.value,
                    "package_accounting": accounting.value,
                    "declared_context": declared,
                    "audited_context": audited,
                })
                for index, item in enumerate(audited["dominance_intervals"], 1):
                    interval_rows.append({
                        "contract_name": contract.name,
                        "pair_policy": policy.value,
                        "peak_basis": peak.value,
                        "package_accounting": accounting.value,
                        "audit_min_context": audited["min_context"],
                        "audit_max_context": audited["max_context"],
                        "interval_index": index,
                        "start_context": item["start_context"],
                        "end_context": item["end_context"],
                        "relation": item["relation"],
                    })
    return rows, interval_rows, analyses


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, object]],
    fields: Sequence[str] | None = None,
) -> None:
    if not rows:
        return
    selected_fields = list(fields or rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=selected_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_tex_macros(
    path: Path,
    *,
    contract: ModelContract,
    scenarios: Sequence[Mapping[str, object]],
    obligations: Mapping[str, float],
    eta_boundary: float | None,
    declared_dominance: Mapping[str, object],
    audited_dominance: Mapping[str, object],
) -> None:
    """Write the small canonical value surface consumed by the manuscripts.

    All repeated Northstar numbers are generated from the same result object.
    The manuscripts retain the model definitions and interpretation, but do
    not hand-copy certificate values.
    """

    by_name = {str(item["scenario"]): item for item in scenarios}
    standard = by_name["standard package / N=16k"]
    advanced = by_name["advanced package / N=16k"]
    vertical = by_name["vertical package / N=16k"]
    standard_times = standard["class_times_ms"]
    advanced_times = advanced["class_times_ms"]
    vertical_times = vertical["class_times_ms"]
    advanced_signature = advanced["winner_signature"]
    advanced_optima = advanced["winner_allocation_optima"]
    advanced_compute = advanced_optima["optimal_compute_partitions"][0]
    advanced_hbm = advanced_optima["reported_witness"]

    def ms(value: object) -> str:
        return f"{float(value):.6f}"

    def integer(value: float) -> str:
        return f"{int(round(value)):,}".replace(",", "{,}")

    def winner_code(value: object) -> str:
        return str(value).split("-", 1)[0]

    lines = [
        "% Generated by northstar_transformer_model_v3.py; do not edit.",
        f"\\newcommand{{\\NorthstarContractName}}{{\\texttt{{{contract.name}}}}}",
        f"\\newcommand{{\\NorthstarDensePeakPF}}{{{selected_peak_flops_s(contract)/1e15:.0f}}}",
        f"\\newcommand{{\\NorthstarTileTokens}}{{{contract.execution.causal_tile_tokens}}}",
        f"\\newcommand{{\\NorthstarStandardWinner}}{{{winner_code(standard['winner'])}}}",
        f"\\newcommand{{\\NorthstarStandardTimeMs}}{{{ms(standard['winner_ms'])}}}",
        f"\\newcommand{{\\NorthstarStandardGapMs}}{{{ms(standard['nonhybrid_gap_ms'])}}}",
        f"\\newcommand{{\\NorthstarStandardAllGapMs}}{{{ms(standard['all_class_gap_ms'])}}}",
        f"\\newcommand{{\\NorthstarAdvancedWinner}}{{{winner_code(advanced['winner'])}}}",
        f"\\newcommand{{\\NorthstarAdvancedTimeMs}}{{{ms(advanced['winner_ms'])}}}",
        f"\\newcommand{{\\NorthstarAdvancedGapMs}}{{{ms(advanced['nonhybrid_gap_ms'])}}}",
        f"\\newcommand{{\\NorthstarAdvancedAllGapMs}}{{{ms(advanced['all_class_gap_ms'])}}}",
        f"\\newcommand{{\\NorthstarVerticalWinner}}{{{winner_code(vertical['winner'])}}}",
        f"\\newcommand{{\\NorthstarVerticalTimeMs}}{{{ms(vertical['winner_ms'])}}}",
        f"\\newcommand{{\\NorthstarVerticalGapMs}}{{{ms(vertical['nonhybrid_gap_ms'])}}}",
        f"\\newcommand{{\\NorthstarHTimeMs}}{{{ms(standard_times['H-static-fused'])}}}",
        f"\\newcommand{{\\NorthstarUTimeMs}}{{{ms(standard_times['U-universal-fused'])}}}",
        f"\\newcommand{{\\NorthstarStandardSTimeMs}}{{{ms(standard_times['S-heterogeneous-split'])}}}",
        f"\\newcommand{{\\NorthstarStandardXTimeMs}}{{{ms(standard_times['X-dual-mode-split-with-universal-fallback'])}}}",
        f"\\newcommand{{\\NorthstarMTimeMs}}{{{ms(standard_times['M-memory-centric-attention'])}}}",
        f"\\newcommand{{\\NorthstarAdvancedSTimeMs}}{{{ms(advanced_times['S-heterogeneous-split'])}}}",
        f"\\newcommand{{\\NorthstarAdvancedXTimeMs}}{{{ms(advanced_times['X-dual-mode-split-with-universal-fallback'])}}}",
        f"\\newcommand{{\\NorthstarVTimeMs}}{{{ms(vertical_times['V-vertical-specialized'])}}}",
        (
            "\\newcommand{\\NorthstarAdvancedComputeAllocation}{"
            f"{advanced_compute[0]}/{advanced_compute[1]}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarAdvancedHBMRepresentative}{"
            f"{advanced_hbm['attention_hbm_stacks']}/"
            f"{advanced_hbm['mlp_hbm_stacks']}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarAdvancedHBMOptimaCount}{"
            f"{advanced_optima['optimal_witness_count']}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarAdvancedBridgeTimeMs}{"
            f"{float(advanced_signature['bridge_s'])*1e3:.6f}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarAdvancedPackageMiB}{"
            f"{float(advanced_signature['package_bytes_per_interval'])/MIB:.0f}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarAdvancedLinkEnergymJ}{"
            f"{float(advanced_signature['link_energy_j'])*1e3:.6f}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarAdvancedLatencyMs}{"
            f"{float(advanced_signature['latency_s'])*1e3:.6f}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarPackageBoundaryGBs}{"
            f"{float(advanced['phase_boundary_distance']['package_boundary_bytes_s'])/1e9:.6f}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarEtaDistance}{"
            f"{float(advanced['phase_boundary_distance']['eta_boundary_distance']):.6f}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarEtaBoundary}{"
            + ("undefined" if eta_boundary is None else f"{eta_boundary:.6f}")
            + "}"
        ),
        (
            "\\newcommand{\\NorthstarDeclaredEarliestSNoSlower}{"
            + (
                "none"
                if declared_dominance["earliest_s_no_slower_context"] is None
                else integer(float(
                    declared_dominance["earliest_s_no_slower_context"]
                ))
            )
            + "}"
        ),
        (
            "\\newcommand{\\NorthstarDeclaredSustainedSNoSlower}{"
            + (
                "none"
                if declared_dominance[
                    "first_sustained_s_no_slower_context"
                ] is None
                else integer(float(
                    declared_dominance[
                        "first_sustained_s_no_slower_context"
                    ]
                ))
            )
            + "}"
        ),
        (
            "\\newcommand{\\NorthstarAuditedEarliestSNoSlower}{"
            + (
                "none"
                if audited_dominance["earliest_s_no_slower_context"] is None
                else integer(float(
                    audited_dominance["earliest_s_no_slower_context"]
                ))
            )
            + "}"
        ),
        (
            "\\newcommand{\\NorthstarAuditedSustainedSNoSlower}{"
            + (
                "none"
                if audited_dominance[
                    "first_sustained_s_no_slower_context"
                ] is None
                else integer(float(
                    audited_dominance[
                        "first_sustained_s_no_slower_context"
                    ]
                ))
            )
            + "}"
        ),
        (
            "\\newcommand{\\NorthstarAuditedSIntervalCount}{"
            f"{len(audited_dominance['s_no_slower_intervals'])}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarAuditedSMonotone}{"
            + (
                "yes"
                if audited_dominance[
                    "s_no_slower_is_monotone_nondecreasing"
                ]
                else "no"
            )
            + "}"
        ),
        (
            "\\newcommand{\\NorthstarBridgeMiB}{"
            f"{obligations['bridge_bytes']/MIB:.0f}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarAttentionTFLOP}{"
            f"{obligations['attention_flops']/1e12:.6f}"
            "}"
        ),
        (
            "\\newcommand{\\NorthstarMLPTFLOP}{"
            f"{obligations['mlp_flops']/1e12:.6f}"
            "}"
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_internal_consistency_tests() -> Dict[str, object]:
    """Fast invariant suite; an independent unittest oracle is also released."""
    checks: Dict[str, object] = {}

    # Direct GQA shape accounting.
    d, n = WORKLOAD.d, 16384
    dkv = d * WORKLOAD.kv_heads // WORKLOAD.heads
    direct_projection = 2 * n * (d * d + d * dkv + d * dkv + d * d)
    dense_contract = contract_with(
        PairPolicy.DENSE_PADDED, PeakBasis.SPARSE, PackageAccounting.SEPARATE
    )
    dense_ob = workload_obligations(n, "prefill", dense_contract)
    assert dense_ob["attention_projection_flops"] == direct_projection
    assert dense_ob["attention_flops"] == dense_ob["mlp_flops"]
    checks["gqa_projection_shape_oracle"] = "pass"
    checks["dense_padded_fa_equals_ff_at_n_4d"] = "pass"

    # Executed-pair ordering and exact block formula.
    ideal = executed_attention_pairs(n, ExecutionContract(PairPolicy.IDEAL_CAUSAL, 128))
    block = executed_attention_pairs(n, ExecutionContract(PairPolicy.BLOCK_ROUNDED, 128))
    dense = executed_attention_pairs(n, ExecutionContract(PairPolicy.DENSE_PADDED, 128))
    assert ideal < block < dense
    assert block == n * (n + 128) // 2
    checks["executed_pair_policy_order"] = "pass"

    # Review-4 compatibility contract reproduces the corrected headline points.
    u = best(universal_fused_candidates(n, "prefill", dense_contract, DEFAULT_FLAT))
    s = best(split_candidates(n, "prefill", PACKAGES[2], dense_contract, DEFAULT_FLAT))
    assert abs(u.initiation_interval_s * 1e3 - 1.4006646287661175) < 1e-12
    assert abs(s.initiation_interval_s * 1e3 - 1.3582202460762354) < 1e-12
    assert (s.attention_chiplets, s.mlp_chiplets) == (16, 16)
    eta = universal_efficiency_boundary(n, PACKAGES[2], dense_contract, DEFAULT_FLAT)
    assert eta is not None and abs(eta - 0.85) < 1e-12
    assert (
        find_monotone_context_overtake(
            PACKAGES[0], dense_contract, DEFAULT_FLAT
        )
        == 99641
    )
    checks["review4_compatibility_numbers"] = "pass"

    # Block-rounded execution is nonmonotone near the package transition.
    # The complete audit must expose the transient interval separately from
    # the terminal interval; a local root check cannot establish minimality.
    dominance = context_dominance_analysis(
        PACKAGES[0], DEFAULT_CONTRACT, DEFAULT_FLAT,
        max_context=DOMINANCE_AUDIT_MAX_CONTEXT,
    )
    assert dominance["s_no_slower_intervals"] == [
        {"start_context": 198913, "end_context": 198926},
        {
            "start_context": 199041,
            "end_context": DOMINANCE_AUDIT_MAX_CONTEXT,
        },
    ]
    assert dominance["earliest_s_no_slower_context"] == 198913
    assert dominance["first_sustained_s_no_slower_context"] == 199041
    assert not dominance["s_no_slower_is_monotone_nondecreasing"]
    try:
        find_monotone_context_overtake(
            PACKAGES[0], DEFAULT_CONTRACT, DEFAULT_FLAT,
            max_context=DOMINANCE_AUDIT_MAX_CONTEXT,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("nonmonotone block audit must reject scalar crossover")
    checks["block_staircase_complete_dominance_intervals"] = "pass"

    # Every split point is reoptimized over the complete integer allocation set.
    assert s.allocation_witnesses_tested == 31 * 7
    assert s.initiation_interval_s <= _split_candidate(
        n, "prefill", PACKAGES[2], dense_contract, DEFAULT_FLAT, 17, 1
    ).initiation_interval_s
    checks["split_complete_allocation_sweep"] = "pass"

    # H never grants both stages a full resource concurrently.
    h_candidates = homogeneous_candidates(
        n, "prefill", dense_contract, DEFAULT_FLAT,
        include_continuous_pipeline_envelope=True,
    )
    for h in h_candidates:
        assert h.compute_share_sum <= 1.0 + 1e-12
        assert h.hbm_share_sum <= 1.0 + 1e-12
        assert h.fabric_share_sum <= 1.0 + 1e-12
    checks["h_resource_conservation"] = "pass"

    # Nested formula remains exact; nonnested 4->6 counterexample is corrected.
    for c1 in (1, 2, 4, 8, 16, 32):
        for c2 in (1, 2, 4, 8, 16, 32):
            assert abs(
                aligned_layout_movement(32, c1, c2)
                - nested_layout_movement(32, c1, c2)
            ) < 1e-12
    assert abs(general_layout_retained_fraction(12, 4, 6) - 0.5) < 1e-12
    assert abs(aligned_layout_movement(12, 4, 6) - 0.5) < 1e-12
    checks["general_layout_assignment_and_counterexample"] = "pass"

    # The registered hybrid is physically distinct but cannot beat the untaxed
    # constituent envelope under its declared positive capability tax.
    x = hybrid_candidate(n, "prefill", PACKAGES[2], dense_contract, DEFAULT_FLAT)
    assert x.initiation_interval_s > min(u.initiation_interval_s, s.initiation_interval_s)
    checks["hybrid_envelope_dominance"] = "pass"

    return checks


def generate_results(outdir: Path, contract: ModelContract = DEFAULT_CONTRACT) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    contexts = [2048, 4096, 8192, 13312, 16384, 32768, 65536, 131072]

    architecture_rows: List[Dict[str, object]] = []
    for kind in ("prefill", "decode"):
        for n in contexts:
            for package, allow_vertical in (
                (PACKAGES[0], False),
                (PACKAGES[1], False),
                (PACKAGES[2], False),
                (PACKAGES[3], True),
            ):
                candidates = scenario_candidates(
                    n, kind, package, contract, DEFAULT_FLAT,
                    allow_vertical=allow_vertical,
                )
                for candidate in candidates:
                    architecture_rows.append({
                        "scenario_package": package.name,
                        "vertical_class_enabled": allow_vertical,
                        **asdict(candidate),
                    })
    _write_csv(outdir / "architecture_sweep.csv", architecture_rows)

    scenario_defs = (
        ("standard package / N=16k", PACKAGES[0], False),
        ("advanced package / N=16k", PACKAGES[2], False),
        ("vertical package / N=16k", PACKAGES[3], True),
    )
    scenarios = [
        certificate_scenario(
            name, package, contract, DEFAULT_FLAT, allow_vertical=allow_vertical
        )
        for name, package, allow_vertical in scenario_defs
    ]
    scenario_rows = [{
        "scenario": sc["scenario"],
        "winner": sc["winner"],
        "winner_ms": sc["winner_ms"],
        "second_best": sc["second_best"],
        "second_best_ms": sc["second_best_ms"],
        "all_class_gap_ms": sc["all_class_gap_ms"],
        "all_class_relative_gap": sc["all_class_relative_gap"],
        "nonhybrid_runner_up": sc["nonhybrid_runner_up"],
        "nonhybrid_gap_ms": sc["nonhybrid_gap_ms"],
        "allocation_optimal_witness_count": sc[
            "winner_allocation_optima"
        ]["optimal_witness_count"],
        "optimal_compute_partitions": json.dumps(
            sc["winner_allocation_optima"]["optimal_compute_partitions"],
            separators=(",", ":"),
        ),
        "optimal_hbm_partitions": json.dumps(
            sc["winner_allocation_optima"]["optimal_hbm_partitions"],
            separators=(",", ":"),
        ),
        "package_boundary_bytes_s": sc["phase_boundary_distance"]["package_boundary_bytes_s"],
        "package_boundary_distance_log2_ratio": sc["phase_boundary_distance"]["package_boundary_distance_log2_ratio"],
        "eta_boundary": sc["phase_boundary_distance"]["eta_boundary"],
        "eta_boundary_distance": sc["phase_boundary_distance"]["eta_boundary_distance"],
        "nearest_normalized_axis": sc["phase_boundary_distance"]["nearest_normalized_axis"],
        "nearest_normalized_distance": sc["phase_boundary_distance"]["nearest_normalized_distance"],
    } for sc in scenarios]
    _write_csv(outdir / "certificate_scenarios.csv", scenario_rows)

    flat_rows = _flat_sensitivity_rows(contract)
    _write_csv(outdir / "flatattention_sensitivity.csv", flat_rows)
    contract_rows, dominance_interval_rows, contract_dominance = (
        _contract_sensitivity_rows()
    )
    _write_csv(outdir / "contract_sensitivity.csv", contract_rows)
    _write_csv(
        outdir / "context_dominance_intervals.csv",
        dominance_interval_rows,
    )

    layout_rows: List[Dict[str, object]] = []
    for p in (12, 32):
        factors = [c for c in range(1, p + 1) if p % c == 0]
        for c1 in factors:
            for c2 in factors:
                nested = (c1 % c2 == 0 or c2 % c1 == 0)
                retained = general_layout_retained_fraction(p, c1, c2)
                layout_rows.append({
                    "P": p,
                    "source_C": c1,
                    "target_C": c2,
                    "nested": nested,
                    "retained_fraction": retained,
                    "moved_fraction": 1.0 - retained,
                    "nested_closed_form_if_applicable": (
                        nested_layout_movement(p, c1, c2) if nested else ""
                    ),
                })
    _write_csv(outdir / "layout_conversion.csv", layout_rows)
    _write_csv(outdir / "technology_library.csv", [asdict(x) for x in PACKAGES])

    obligations_16k = workload_obligations(16384, "prefill", contract)
    audited_dominance = context_dominance_analysis(
        PACKAGES[0], contract, DEFAULT_FLAT,
        max_context=DOMINANCE_AUDIT_MAX_CONTEXT,
    )
    declared_dominance = restrict_context_dominance_analysis(
        audited_dominance, WORKLOAD.max_context
    )
    tests = run_internal_consistency_tests()
    eta_boundary = universal_efficiency_boundary(
        16384, PACKAGES[2], contract, DEFAULT_FLAT
    )
    result = {
        "schema_version": "northstar-vNext4.1-model-v3.1",
        "default_contract": {
            **asdict(contract),
            "selected_peak_flops_s": selected_peak_flops_s(contract),
            "peak_interpretation": (
                "5 PFLOP/s is the dense BF16 coordinate; 10 PFLOP/s is retained "
                "only as a structured-sparse sensitivity and is not silently "
                "applied to the dense named workload."
            ),
            "operation_scope": "modeled tensor-matmul FLOPs",
            "typed_exclusions": [
                "softmax", "RoPE", "RMSNorm", "residual adds",
                "SwiGLU elementwise activation", "kernel launch", "control",
                "synchronization", "ECC",
                "additional nonideal utilization beyond declared mode factors",
            ],
        },
        "workload": asdict(WORKLOAD),
        "system": asdict(SYSTEM),
        "flatattention_coordinates": asdict(DEFAULT_FLAT),
        "attention_modes": [asdict(x) for x in attention_modes(DEFAULT_FLAT)],
        "mlp_modes": [asdict(x) for x in MLP_MODES],
        "packages": [asdict(x) for x in PACKAGES],
        "n16384_prefill_obligations": obligations_16k,
        "certificate_scenarios": scenarios,
        "ucie_s_dominance_within_declared_context": declared_dominance,
        "ucie_s_dominance_within_audited_context": audited_dominance,
        "universal_specialization_eta_boundary_a64_n16384": eta_boundary,
        "flatattention_sweep_summary": {
            "rows": len(flat_rows),
            "winner_counts": {
                name: sum(1 for row in flat_rows if row["winner"] == name)
                for name in sorted({str(row["winner"]) for row in flat_rows})
            },
        },
        "contract_sensitivity_rows": len(contract_rows),
        "contract_dominance_analyses": contract_dominance,
        "context_dominance_interval_rows": len(dominance_interval_rows),
        "layout_theorem": {
            "general_statement": (
                "retention is the maximum-weight perfect-assignment sum over "
                "source/target shard intersections"
            ),
            "nested_corollary": (
                "if C divides C' or C' divides C, moved fraction is "
                "1-min(C,C')/max(C,C')"
            ),
            "p12_c4_to_c6_retained_fraction": general_layout_retained_fraction(12, 4, 6),
            "p12_c4_to_c6_moved_fraction": aligned_layout_movement(12, 4, 6),
        },
        "internal_consistency_tests": tests,
        "unresolved_modeling_choices": [
            "Actual causal executed-pair count requires an implementation trace; "
            "dense-padded, ideal-causal, and block-rounded results are all emitted.",
            "The shared aggregate package-fabric contract is conservative; a "
            "separate internal-local-fabric comparator is emitted rather than hidden.",
            "FlatAttention coordinates remain imported MHA-platform maxima and are "
            "swept as portability hypotheses.",
            "The 5-PF dense peak is a ceiling, not an achieved end-to-end rate; "
            "auxiliary operations and synchronization remain typed exclusions.",
            "The continuous-resource H chunk envelope is excluded from class "
            "competition until an integer/common-allocation stream is declared; "
            "the sequential conserved witness is the registered H schedule.",
            "Usable fast-state derating and fragmentation require calibration; "
            "capacity-wall witnesses have zero certified slack.",
            "The X hybrid capability tax is declared rather than measured.",
            "Routed contention, buffers, clocks, ECC, energy, area, thermal, "
            "yield, and cost are outside the current algebraic certificate.",
            "Memory-centric and vertical delivered capabilities remain declared "
            "sensitivity coordinates.",
        ],
    }
    result_path = outdir / "northstar_results.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_tex_macros(
        outdir / "northstar_macros.tex",
        contract=contract,
        scenarios=scenarios,
        obligations=obligations_16k,
        eta_boundary=eta_boundary,
        declared_dominance=declared_dominance,
        audited_dominance=audited_dominance,
    )

    digest = hashlib.sha256()
    for path in sorted(outdir.glob("*")):
        if path.name == "RESULTS_SHA256.txt" or not path.is_file():
            continue
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    (outdir / "RESULTS_SHA256.txt").write_text(digest.hexdigest() + "\n", encoding="utf-8")
    result["results_sha256"] = digest.hexdigest()
    return result


def _contract_from_args(args: argparse.Namespace) -> ModelContract:
    policy = PairPolicy(args.pair_policy)
    peak = PeakBasis(args.peak_basis)
    accounting = PackageAccounting(args.package_accounting)
    contract = contract_with(policy, peak, accounting)
    if policy == PairPolicy.BLOCK_ROUNDED and args.tile_tokens != 128:
        contract = replace(
            contract,
            name=contract.name.replace("128", str(args.tile_tokens), 1),
            execution=ExecutionContract(policy, args.tile_tokens),
        )
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "northstar_results_vNext4",
    )
    parser.add_argument(
        "--pair-policy",
        choices=[x.value for x in PairPolicy],
        default=PairPolicy.BLOCK_ROUNDED.value,
    )
    parser.add_argument("--tile-tokens", type=int, default=128)
    parser.add_argument(
        "--peak-basis",
        choices=[x.value for x in PeakBasis],
        default=PeakBasis.DENSE.value,
    )
    parser.add_argument(
        "--package-accounting",
        choices=[x.value for x in PackageAccounting],
        default=PackageAccounting.SHARED.value,
    )
    parser.add_argument(
        "--self-test-only",
        action="store_true",
        help="run invariant checks without regenerating result artifacts",
    )
    args = parser.parse_args()
    if args.self_test_only:
        print(json.dumps(run_internal_consistency_tests(), indent=2, sort_keys=True))
        return
    contract = _contract_from_args(args)
    result = generate_results(args.out.resolve(), contract)

    def dominance_summary(item: Mapping[str, object]) -> Dict[str, object]:
        return {
            "min_context": item["min_context"],
            "max_context": item["max_context"],
            "dominance_intervals": item["dominance_intervals"],
            "earliest_s_no_slower_context": item[
                "earliest_s_no_slower_context"
            ],
            "first_sustained_s_no_slower_context": item[
                "first_sustained_s_no_slower_context"
            ],
            "s_no_slower_is_monotone_nondecreasing": item[
                "s_no_slower_is_monotone_nondecreasing"
            ],
            "sustained_scope": item["sustained_scope"],
        }

    print(json.dumps({
        "default_contract": result["default_contract"]["name"],
        "certificate_winners": [
            {
                "scenario": sc["scenario"],
                "winner": sc["winner"],
                "winner_ms": sc["winner_ms"],
                "all_class_gap_ms": sc["all_class_gap_ms"],
                "nonhybrid_gap_ms": sc["nonhybrid_gap_ms"],
            }
            for sc in result["certificate_scenarios"]
        ],
        "ucie_s_dominance_within_declared_context": dominance_summary(
            result["ucie_s_dominance_within_declared_context"]
        ),
        "ucie_s_dominance_within_audited_context": dominance_summary(
            result["ucie_s_dominance_within_audited_context"]
        ),
        "eta_boundary": result[
            "universal_specialization_eta_boundary_a64_n16384"
        ],
        "internal_tests": result["internal_consistency_tests"],
        "results_sha256": result["results_sha256"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
