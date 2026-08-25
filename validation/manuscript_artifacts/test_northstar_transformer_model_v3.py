#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Independent consistency oracles for northstar_transformer_model_v3.

These tests deliberately recompute key quantities from shape arithmetic and
small enumerations rather than treating a second call to the model as an
independent verification.
"""
from __future__ import annotations

import csv
import hashlib
import itertools
import math
import unittest
from pathlib import Path

import numpy as np

import northstar_transformer_model_v3 as model


def independent_all_contract_dominance_runs(
    *,
    max_context: int = 1_048_576,
    chunk_size: int = 65536,
) -> dict[tuple[str, str, str], list[tuple[int, int, bool]]]:
    """Full direct oracle with a literal, unfactorized 31x7 S sweep.

    The constants below are the declared ledger coordinates, intentionally
    repeated rather than imported from production helpers.  In particular,
    the S minimum is taken only after jointly enumerating every legal
    compute/HBM pair; this oracle does not reuse the production factorization
    or dominance implementation.
    """
    d, d_ff, bytes_per_element = 4096, 14336, 2
    attention_weight_bytes = 2.5 * d**2 * bytes_per_element
    mlp_weight_bytes = 3.0 * d * d_ff * bytes_per_element
    results: dict[tuple[str, str, str], list[tuple[int, int, bool]]] = {}

    for pair_policy in (
        "dense-padded",
        "ideal-causal",
        "block-rounded-causal",
    ):
        for peak_basis, peak_flops_s in (
            ("dense-bf16", 5e15),
            ("sparse-bf16-sensitivity", 10e15),
        ):
            for accounting in (
                "shared-package-aggregate",
                "separate-internal-local-fabric",
            ):
                runs: list[tuple[int, int, bool]] = []
                for low in range(1, max_context + 1, chunk_size):
                    high = min(max_context, low + chunk_size - 1)
                    n_int = np.arange(low, high + 1, dtype=np.int64)
                    n = n_int.astype(np.float64)
                    if pair_policy == "dense-padded":
                        pairs = n_int * n_int
                    elif pair_policy == "ideal-causal":
                        pairs = n_int * (n_int + 1) // 2
                    else:
                        tiles = (n_int + 127) // 128
                        pairs = 128**2 * tiles * (tiles + 1) // 2

                    activation = n * d * bytes_per_element
                    f_attention = (
                        5.0 * n * d**2
                        + 4.0 * pairs.astype(np.float64) * d
                    )
                    f_mlp = 6.0 * n * d * d_ff
                    q_attention = attention_weight_bytes + 4.0 * activation
                    q_mlp = mlp_weight_bytes + 6.0 * activation
                    d2d = 2.0 * activation
                    bridge = 2.0 * activation

                    # U: independently enumerate both attention modes and
                    # both MLP modes.
                    u_time = np.full(n.shape, np.inf)
                    for attention_util, attention_hbm, attention_d2d in (
                        (0.893 / 4.1, 16.0, 1.0),
                        (0.80, 1.0, 0.5),
                    ):
                        attention_time = np.maximum.reduce((
                            f_attention
                            / (peak_flops_s * attention_util),
                            q_attention * attention_hbm / 8e12,
                            d2d * attention_d2d / 4e12,
                        ))
                        for mlp_hbm, mlp_d2d in (
                            (1.0, 1.0),
                            (0.5, 0.5),
                        ):
                            mlp_time = np.maximum.reduce((
                                f_mlp / (peak_flops_s * 0.85),
                                q_mlp * mlp_hbm / 8e12,
                                d2d * mlp_d2d / 4e12,
                            ))
                            u_time = np.minimum(
                                u_time, attention_time + mlp_time
                            )

                    # S: enumerate all 217 joint witnesses without taking
                    # separate compute and HBM lower envelopes.
                    s_time = np.full(n.shape, np.inf)
                    for pa in range(1, 32):
                        pf = 32 - pa
                        attention_compute = (
                            f_attention
                            / (peak_flops_s * 0.893 * pa / 32)
                        )
                        mlp_compute = (
                            f_mlp / (peak_flops_s * 0.85 * pf / 32)
                        )
                        if accounting == (
                            "separate-internal-local-fabric"
                        ):
                            attention_local = (
                                0.5 * d2d / (4e12 * pa / 32)
                            )
                            mlp_local = (
                                0.5 * d2d / (4e12 * pf / 32)
                            )
                        for ha in range(1, 8):
                            hf = 8 - ha
                            terms = [
                                attention_compute,
                                mlp_compute,
                                q_attention / (8e12 * ha / 8),
                                0.5 * q_mlp / (8e12 * hf / 8),
                            ]
                            if accounting == "shared-package-aggregate":
                                terms.append((d2d + bridge) / 64e9)
                            else:
                                terms.extend((
                                    attention_local,
                                    mlp_local,
                                    bridge / 64e9,
                                ))
                            s_time = np.minimum(
                                s_time, np.maximum.reduce(terms)
                            )

                    states = s_time <= u_time
                    changes = (
                        np.flatnonzero(states[1:] != states[:-1]) + 1
                    )
                    starts = np.concatenate((np.array([0]), changes))
                    stops = np.concatenate((
                        changes, np.array([states.size])
                    ))
                    for start_index, stop_index in zip(starts, stops):
                        start = low + int(start_index)
                        end = low + int(stop_index) - 1
                        state = bool(states[int(start_index)])
                        if (
                            runs
                            and runs[-1][2] == state
                            and runs[-1][1] + 1 == start
                        ):
                            prior = runs[-1]
                            runs[-1] = (prior[0], end, state)
                        else:
                            runs.append((start, end, state))
                results[(pair_policy, peak_basis, accounting)] = runs
    return results


class NorthstarV3IndependentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.compat = model.contract_with(
            model.PairPolicy.DENSE_PADDED,
            model.PeakBasis.SPARSE,
            model.PackageAccounting.SEPARATE,
        )

    def test_gqa_projection_shape_count(self) -> None:
        w, n = model.WORKLOAD, 16384
        head_dim = w.d // w.heads
        kv_width = w.kv_heads * head_dim
        # Two FLOPs per multiply-add for Q, K, V, and O.
        direct = 2 * n * (
            w.d * w.d
            + w.d * kv_width
            + w.d * kv_width
            + w.d * w.d
        )
        obligations = model.workload_obligations(n, "prefill", self.compat)
        self.assertEqual(obligations["attention_projection_flops"], direct)
        self.assertEqual(
            direct,
            (4 + 4 * w.kv_heads / w.heads) * n * w.d**2,
        )

    def test_executed_pair_oracles(self) -> None:
        n, block = 1031, 128
        ideal_oracle = sum(row + 1 for row in range(n))
        dense_oracle = sum(n for _ in range(n))
        tiles = math.ceil(n / block)
        block_oracle = sum((tile + 1) * block * block for tile in range(tiles))
        self.assertEqual(
            model.executed_attention_pairs(
                n, model.ExecutionContract(model.PairPolicy.IDEAL_CAUSAL, block)
            ),
            ideal_oracle,
        )
        self.assertEqual(
            model.executed_attention_pairs(
                n, model.ExecutionContract(model.PairPolicy.DENSE_PADDED, block)
            ),
            dense_oracle,
        )
        self.assertEqual(
            model.executed_attention_pairs(
                n, model.ExecutionContract(model.PairPolicy.BLOCK_ROUNDED, block)
            ),
            block_oracle,
        )

    def test_review4_dense_padded_point_from_direct_algebra(self) -> None:
        w, n = model.WORKLOAD, 16384
        peak = 10e15
        f_attention = (
            (4 + 4 * w.kv_heads / w.heads) * n * w.d**2
            + 4 * n**2 * w.d
        )
        f_mlp = 6 * n * w.d * w.d_ff
        self.assertEqual(f_attention, f_mlp)

        u_oracle = f_attention / (peak * 0.80) + f_mlp / (peak * 0.85)
        u = model.best(model.universal_fused_candidates(
            n, "prefill", self.compat, model.DEFAULT_FLAT
        ))
        self.assertAlmostEqual(u.initiation_interval_s, u_oracle, places=15)

        # Independent enumeration of the compute allocation floor.  HBM and
        # fabric are checked below to be nonbinding at the selected witness.
        split_compute = []
        for pa in range(1, 32):
            pf = 32 - pa
            ta = f_attention / (peak * 0.893 * pa / 32)
            tf = f_mlp / (peak * 0.85 * pf / 32)
            split_compute.append((max(ta, tf), pa, pf))
        oracle, pa, pf = min(split_compute)
        self.assertEqual((pa, pf), (16, 16))
        s = model.best(model.split_candidates(
            n, "prefill", model.PACKAGES[2], self.compat, model.DEFAULT_FLAT
        ))
        self.assertEqual((s.attention_chiplets, s.mlp_chiplets), (16, 16))
        self.assertAlmostEqual(s.initiation_interval_s, oracle, places=15)
        self.assertEqual(s.bottleneck, "mlp-compute")

    def test_review4_boundary_and_monotone_dense_overtake(self) -> None:
        eta = model.universal_efficiency_boundary(
            16384, model.PACKAGES[2], self.compat, model.DEFAULT_FLAT
        )
        self.assertAlmostEqual(eta, 0.85, places=14)
        analysis = model.context_dominance_analysis(
            model.PACKAGES[0], self.compat, model.DEFAULT_FLAT
        )
        self.assertTrue(analysis["s_no_slower_is_monotone_nondecreasing"])
        self.assertEqual(analysis["s_no_slower_intervals"], [{
            "start_context": 99641,
            "end_context": model.WORKLOAD.max_context,
        }])
        crossing = model.find_monotone_context_overtake(
            model.PACKAGES[0], self.compat, model.DEFAULT_FLAT
        )
        self.assertEqual(crossing, 99641)
        # Scalar boundary checks independently confirm the exact audited run.
        def delta(n: int) -> float:
            u = model.best(model.universal_fused_candidates(
                n, "prefill", self.compat, model.DEFAULT_FLAT
            )).initiation_interval_s
            s = model.best(model.split_candidates(
                n, "prefill", model.PACKAGES[0], self.compat, model.DEFAULT_FLAT
            )).initiation_interval_s
            return u - s
        self.assertLess(delta(crossing - 1), 0.0)
        self.assertGreaterEqual(delta(crossing), 0.0)

    def test_default_certificate_from_direct_algebra(self) -> None:
        """Reconstruct the default headline without calling model components."""
        w, n, block = model.WORKLOAD, 16384, 128
        peak = 5e15
        tiles = n // block
        executed_pairs = block**2 * tiles * (tiles + 1) // 2
        activation = n * w.d * w.bytes_per_element
        f_attention = (
            (4 + 4 * w.kv_heads / w.heads) * n * w.d**2
            + 4 * executed_pairs * w.d
        )
        f_mlp = 6 * n * w.d * w.d_ff

        u_oracle = (
            f_attention / (peak * 0.80)
            + f_mlp / (peak * 0.85)
        )
        u = model.best(model.universal_fused_candidates(
            n, "prefill", model.DEFAULT_CONTRACT, model.DEFAULT_FLAT
        ))
        self.assertAlmostEqual(u.initiation_interval_s, u_oracle, places=15)

        compute_floors = []
        for pa in range(1, 32):
            pf = 32 - pa
            compute_floors.append((
                max(
                    f_attention / (peak * 0.893 * pa / 32),
                    f_mlp / (peak * 0.85 * pf / 32),
                ),
                pa,
                pf,
            ))
        compute_oracle, pa, pf = min(compute_floors)
        self.assertEqual((pa, pf), (12, 20))

        weight_attention = (
            2 + 2 * w.kv_heads / w.heads
        ) * w.d**2 * w.bytes_per_element
        weight_mlp = 3 * w.d * w.d_ff * w.bytes_per_element
        q_attention = weight_attention + 4 * activation
        q_mlp = 0.5 * (weight_mlp + 6 * activation)
        package_floor = 4 * activation / 512e9
        tied_hbm_partitions = []
        for ha in range(1, 8):
            hf = 8 - ha
            hbm_floor = max(
                q_attention / (8e12 * ha / 8),
                q_mlp / (8e12 * hf / 8),
            )
            if max(compute_oracle, hbm_floor, package_floor) == compute_oracle:
                tied_hbm_partitions.append((ha, hf))
        self.assertEqual(tied_hbm_partitions, [
            (1, 7), (2, 6), (3, 5), (4, 4), (5, 3), (6, 2), (7, 1)
        ])

        s_pool = model.split_candidates(
            n, "prefill", model.PACKAGES[2],
            model.DEFAULT_CONTRACT, model.DEFAULT_FLAT,
        )
        s = model.best(s_pool)
        self.assertAlmostEqual(s.initiation_interval_s, compute_oracle, places=15)
        optimum = model.allocation_optimum_summary(s_pool, s)
        self.assertEqual(optimum["optimal_witness_count"], 7)
        self.assertEqual(optimum["optimal_compute_partitions"], [[12, 20]])
        self.assertEqual(
            optimum["optimal_hbm_partitions"],
            [list(x) for x in tied_hbm_partitions],
        )

        standard = model.certificate_scenario(
            "standard", model.PACKAGES[0], model.DEFAULT_CONTRACT,
            model.DEFAULT_FLAT, allow_vertical=False, include_boundary=False,
        )
        advanced = model.certificate_scenario(
            "advanced", model.PACKAGES[2], model.DEFAULT_CONTRACT,
            model.DEFAULT_FLAT, allow_vertical=False, include_boundary=False,
        )
        vertical = model.certificate_scenario(
            "vertical", model.PACKAGES[3], model.DEFAULT_CONTRACT,
            model.DEFAULT_FLAT, allow_vertical=True, include_boundary=False,
        )
        self.assertEqual(standard["winner"], "U-universal-fused")
        self.assertEqual(advanced["winner"], "S-heterogeneous-split")
        self.assertEqual(vertical["winner"], "V-vertical-specialized")
        self.assertAlmostEqual(
            vertical["winner_ms"] / 1e3, compute_oracle / 2.0, places=15
        )

    def test_default_boundaries_and_all_contract_winners(self) -> None:
        eta = model.universal_efficiency_boundary(
            16384, model.PACKAGES[2],
            model.DEFAULT_CONTRACT, model.DEFAULT_FLAT,
        )
        self.assertAlmostEqual(eta, 0.8812003968253968, places=14)
        declared = model.context_dominance_analysis(
            model.PACKAGES[0], model.DEFAULT_CONTRACT, model.DEFAULT_FLAT
        )
        self.assertEqual(declared["dominance_intervals"], [{
            "start_context": 1,
            "end_context": model.WORKLOAD.max_context,
            "relation": "U-strictly-faster",
        }])
        audited = model.context_dominance_analysis(
            model.PACKAGES[0], model.DEFAULT_CONTRACT, model.DEFAULT_FLAT,
            max_context=model.DOMINANCE_AUDIT_MAX_CONTEXT,
        )
        self.assertEqual(audited["s_no_slower_intervals"], [
            {"start_context": 198913, "end_context": 198926},
            {
                "start_context": 199041,
                "end_context": model.DOMINANCE_AUDIT_MAX_CONTEXT,
            },
        ])
        self.assertEqual(audited["earliest_s_no_slower_context"], 198913)
        self.assertEqual(
            audited["first_sustained_s_no_slower_context"], 199041
        )
        self.assertFalse(
            audited["s_no_slower_is_monotone_nondecreasing"]
        )
        with self.assertRaises(ValueError):
            model.find_monotone_context_overtake(
                model.PACKAGES[0], model.DEFAULT_CONTRACT, model.DEFAULT_FLAT,
                max_context=model.DOMINANCE_AUDIT_MAX_CONTEXT,
            )

        def delta(n: int) -> float:
            u = model.best(model.universal_fused_candidates(
                n, "prefill", model.DEFAULT_CONTRACT, model.DEFAULT_FLAT
            )).initiation_interval_s
            s = model.best(model.split_candidates(
                n, "prefill", model.PACKAGES[0],
                model.DEFAULT_CONTRACT, model.DEFAULT_FLAT,
            )).initiation_interval_s
            return u - s

        for start, end in ((198913, 198926), (199041, 199041)):
            self.assertLess(delta(start - 1), 0.0)
            self.assertGreaterEqual(delta(start), 0.0)
            if end < 199041:
                self.assertGreaterEqual(delta(end), 0.0)
                self.assertLess(delta(end + 1), 0.0)

        for policy in model.PairPolicy:
            for peak in model.PeakBasis:
                for accounting in model.PackageAccounting:
                    contract = model.contract_with(policy, peak, accounting)
                    expected = (
                        (model.PACKAGES[0], False, "U-universal-fused"),
                        (model.PACKAGES[2], False, "S-heterogeneous-split"),
                        (model.PACKAGES[3], True, "V-vertical-specialized"),
                    )
                    for package, allow_vertical, winner in expected:
                        actual = model.best(model.scenario_candidates(
                            16384, "prefill", package, contract,
                            model.DEFAULT_FLAT, allow_vertical=allow_vertical,
                        ))
                        self.assertEqual(
                            actual.architecture, winner,
                            msg=(policy, peak, accounting, package.name),
                        )

    def test_all_emitted_dominance_partitions_against_full_oracle(self) -> None:
        result_path = (
            Path(__file__).resolve().parent
            / "northstar_results_vNext4"
            / "context_dominance_intervals.csv"
        )
        emitted: dict[
            tuple[str, str, str], list[tuple[int, int, bool]]
        ] = {}
        with result_path.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                key = (
                    row["pair_policy"],
                    row["peak_basis"],
                    row["package_accounting"],
                )
                self.assertEqual(int(row["audit_min_context"]), 1)
                self.assertEqual(
                    int(row["audit_max_context"]), 1_048_576
                )
                emitted.setdefault(key, []).append((
                    int(row["start_context"]),
                    int(row["end_context"]),
                    row["relation"] == "S-no-slower",
                ))

        oracle = independent_all_contract_dominance_runs()
        self.assertEqual(len(oracle), 12)
        self.assertEqual(set(emitted), set(oracle))
        for key, expected_runs in oracle.items():
            self.assertEqual(emitted[key], expected_runs, msg=key)

    def test_dense_and_ideal_policies_establish_monotonicity_before_root(self) -> None:
        for policy in (
            model.PairPolicy.DENSE_PADDED,
            model.PairPolicy.IDEAL_CAUSAL,
        ):
            for peak in model.PeakBasis:
                for accounting in model.PackageAccounting:
                    contract = model.contract_with(policy, peak, accounting)
                    analysis = model.context_dominance_analysis(
                        model.PACKAGES[0], contract, model.DEFAULT_FLAT,
                        max_context=model.DOMINANCE_AUDIT_MAX_CONTEXT,
                    )
                    self.assertTrue(
                        analysis["s_no_slower_is_monotone_nondecreasing"],
                        msg=(policy, peak, accounting),
                    )
                    self.assertEqual(
                        model.find_monotone_context_overtake(
                            model.PACKAGES[0], contract, model.DEFAULT_FLAT,
                            max_context=model.DOMINANCE_AUDIT_MAX_CONTEXT,
                        ),
                        analysis["earliest_s_no_slower_context"],
                    )
                    self.assertLessEqual(
                        len(analysis["s_no_slower_intervals"]), 1
                    )

    def test_vectorized_audit_matches_scalar_candidate_enumeration(self) -> None:
        contexts = np.array([
            1, 127, 128, 129, 16384, 37141, 73985, 74113,
            99641, 198913, 198927, 199041, 224641, 449153,
        ], dtype=np.int64)
        for policy in model.PairPolicy:
            for peak in model.PeakBasis:
                for accounting in model.PackageAccounting:
                    contract = model.contract_with(policy, peak, accounting)
                    u_vector, s_vector = model._prefill_u_s_times_array(
                        contexts, model.PACKAGES[0], contract,
                        model.DEFAULT_FLAT,
                    )
                    for n, u_value, s_value in zip(
                        contexts, u_vector, s_vector
                    ):
                        u_scalar = model.best(
                            model.universal_fused_candidates(
                                int(n), "prefill", contract,
                                model.DEFAULT_FLAT,
                            )
                        ).initiation_interval_s
                        s_scalar = model.best(model.split_candidates(
                            int(n), "prefill", model.PACKAGES[0], contract,
                            model.DEFAULT_FLAT,
                        )).initiation_interval_s
                        self.assertEqual(u_value, u_scalar)
                        self.assertEqual(s_value, s_scalar)

    def test_vectorized_audit_rejects_nonintegral_contexts(self) -> None:
        for contexts in (
            np.array([1.5, 2.0]),
            np.array([np.nan]),
            np.array(["128"]),
            np.array([True]),
        ):
            with self.subTest(dtype=str(contexts.dtype)):
                with self.assertRaisesRegex(ValueError, "integer dtype"):
                    model._prefill_u_s_times_array(
                        contexts,
                        model.PACKAGES[0],
                        model.DEFAULT_CONTRACT,
                        model.DEFAULT_FLAT,
                    )

    def test_h_pipeline_conserves_every_resource(self) -> None:
        candidates = model.homogeneous_candidates(
            16384, "prefill", model.DEFAULT_CONTRACT, model.DEFAULT_FLAT,
            include_continuous_pipeline_envelope=True,
        )
        pipeline = [
            c for c in candidates
            if c.schedule == "chunk-pipeline-continuous-resource-lower-envelope"
        ]
        self.assertTrue(pipeline)
        for candidate in pipeline:
            self.assertAlmostEqual(candidate.compute_share_sum, 1.0)
            self.assertAlmostEqual(candidate.hbm_share_sum, 1.0)
            self.assertAlmostEqual(candidate.fabric_share_sum, 1.0)

    def test_general_layout_assignment_against_bruteforce_small_p(self) -> None:
        # Independent exhaustive assignment for P=4, C=1 -> C'=2.
        p, c1, c2 = 4, 1, 2
        t1, t2 = p // c1, p // c2
        source = [(i, j) for i in range(c1) for j in range(t1)]
        target = [(i, j) for i in range(c2) for j in range(t2)]

        def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
            return max(0.0, min(a1, b1) - max(a0, b0))

        weights = []
        for sr, sh in source:
            row = []
            for tr, th in target:
                row.append(
                    overlap(sr / c1, (sr + 1) / c1, tr / c2, (tr + 1) / c2)
                    * overlap(sh / t1, (sh + 1) / t1, th / t2, (th + 1) / t2)
                )
            weights.append(row)
        brute = max(
            sum(weights[i][perm[i]] for i in range(p))
            for perm in itertools.permutations(range(p))
        )
        self.assertAlmostEqual(
            model.general_layout_retained_fraction(p, c1, c2), brute
        )
        # The reviewer counterexample: the old formula predicts 2/3 retained;
        # the exact assignment retains only 1/2.
        self.assertAlmostEqual(
            model.general_layout_retained_fraction(12, 4, 6), 0.5
        )

    def test_generated_result_manifest(self) -> None:
        out = Path(__file__).resolve().parent / "northstar_results_vNext4"
        if not out.exists():
            self.skipTest("result directory has not yet been generated")
        digest = hashlib.sha256()
        for path in sorted(out.glob("*")):
            if path.name == "RESULTS_SHA256.txt" or not path.is_file():
                continue
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
        expected = (out / "RESULTS_SHA256.txt").read_text(encoding="utf-8").strip()
        self.assertEqual(digest.hexdigest(), expected)

        with (out / "architecture_sweep.csv").open(
            newline="", encoding="utf-8"
        ) as fh:
            rows = list(csv.DictReader(fh))
        self.assertTrue(rows)
        self.assertTrue(all(row["scenario_package"] for row in rows))
        self.assertEqual(len(rows), len({
            tuple(sorted(row.items())) for row in rows
        }))
        self.assertFalse(any(
            row["scenario_package"] == "UCIe-A64-declared"
            and row["architecture"].startswith("V-")
            for row in rows
        ))
        self.assertTrue(any(
            row["scenario_package"] == "3D-vertical-declared"
            and row["architecture"].startswith("V-")
            for row in rows
        ))

        with (out / "contract_sensitivity.csv").open(
            newline="", encoding="utf-8"
        ) as fh:
            contracts = list(csv.DictReader(fh))
        self.assertEqual(len(contracts), 12)
        self.assertTrue(all(
            row["standard_winner"] == "U-universal-fused"
            and row["a64_winner"] == "S-heterogeneous-split"
            and row["vertical_winner"] == "V-vertical-specialized"
            for row in contracts
        ))
        self.assertNotIn(
            "ucie_s_extrapolated_crossover_do_not_claim", contracts[0]
        )
        self.assertTrue(all(
            row["audited_first_sustained_s_no_slower_context"]
            for row in contracts
        ))

        with (out / "context_dominance_intervals.csv").open(
            newline="", encoding="utf-8"
        ) as fh:
            intervals = list(csv.DictReader(fh))
        groups: dict[str, list[dict[str, str]]] = {}
        for row in intervals:
            groups.setdefault(row["contract_name"], []).append(row)
        self.assertEqual(len(groups), 12)
        for contract_name, group in groups.items():
            self.assertEqual(int(group[0]["start_context"]), 1)
            self.assertEqual(
                int(group[-1]["end_context"]),
                model.DOMINANCE_AUDIT_MAX_CONTEXT,
            )
            for left, right in zip(group, group[1:]):
                self.assertEqual(
                    int(left["end_context"]) + 1,
                    int(right["start_context"]),
                    msg=contract_name,
                )
                self.assertNotEqual(
                    left["relation"], right["relation"],
                    msg=contract_name,
                )
        default_group = groups[model.DEFAULT_CONTRACT.name]
        self.assertEqual([
            (
                int(row["start_context"]),
                int(row["end_context"]),
                row["relation"],
            )
            for row in default_group
        ], [
            (1, 198912, "U-strictly-faster"),
            (198913, 198926, "S-no-slower"),
            (198927, 199040, "U-strictly-faster"),
            (
                199041,
                model.DOMINANCE_AUDIT_MAX_CONTEXT,
                "S-no-slower",
            ),
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
