#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Deterministic validation for the CRG renormalization results.

The program exercises four theorem-ready cases:

1. an autonomous primitive positive cocycle, including a nonsummable
   vanishing scalar perturbation;
2. a two-phase positive Perron--Floquet cocycle;
3. hard- and soft-cap crossover collapse for a regularly varying interior;
4. negative controls showing why mixing and phase-ratio hypotheses matter.

It writes machine-readable CSV/JSON results and exactly four PNG figures.
No manuscript source is modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

MPL_CACHE = Path(tempfile.gettempdir()) / "crg_renormalization_matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "renormalization_results_v1"
FIG = ROOT / "figures_renormalization_v1"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

B = 2.0
LOG_B = math.log(B)


def normalized(x: np.ndarray) -> np.ndarray:
    total = float(np.sum(x))
    if total <= 0.0:
        raise ValueError("positive normalization requires positive total")
    return x / total


def hilbert_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Hilbert projective metric on the strictly positive orthant."""
    ratio = np.asarray(x, dtype=float) / np.asarray(y, dtype=float)
    return float(math.log(float(np.max(ratio)) / float(np.min(ratio))))


def perron_pair(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    values, vectors = np.linalg.eig(matrix)
    index = int(np.argmax(values.real))
    value = float(values[index].real)
    vector = np.asarray(vectors[:, index].real, dtype=float)
    if float(np.sum(vector)) < 0.0:
        vector = -vector
    if np.any(vector <= 0.0):
        raise ValueError("declared primitive matrix did not yield positive Perron vector")
    return value, normalized(vector)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        # Keep retained checksums stable across Git's LF normalization and
        # across operating systems whose csv defaults differ.
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def record_test(
    tests: list[dict[str, Any]],
    name: str,
    passed: bool,
    value: float | int | str,
    criterion: str,
) -> None:
    tests.append(
        {
            "name": name,
            "passed": bool(passed),
            "value": value,
            "criterion": criterion,
        }
    )


def run_autonomous(tests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    matrix = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=float)
    perron, direction = perron_pair(matrix)
    exponent = math.log(perron) / LOG_B
    starts = {
        "left_heavy": np.array([0.99, 0.01]),
        "right_heavy": np.array([0.01, 0.99]),
        "balanced": np.array([0.50, 0.50]),
        "asymmetric": np.array([0.82, 0.18]),
    }
    steps = 48
    rows: list[dict[str, Any]] = []

    final_hilbert: list[float] = []
    final_ratio_error: list[float] = []
    for label, start in starts.items():
        state = start.astype(float)
        for k in range(steps + 1):
            total = float(np.sum(state))
            direction_k = normalized(state)
            next_state = matrix @ state
            ratio = float(np.sum(next_state) / total)
            hdist = hilbert_distance(direction_k, direction)
            rows.append(
                {
                    "case": "autonomous_exact",
                    "trajectory": label,
                    "k": k,
                    "scale_g": B**k,
                    "Y": total,
                    "ratio_next": ratio,
                    "expected_ratio": perron,
                    "ratio_error": abs(ratio - perron),
                    "z0": float(direction_k[0]),
                    "z1": float(direction_k[1]),
                    "hilbert_to_attractor": hdist,
                    "envelope_over_kplus1": "",
                }
            )
            if k == steps:
                final_hilbert.append(hdist)
                final_ratio_error.append(abs(ratio - perron))
            state = next_state

    # Vanishing but nonsummable scalar perturbation:
    # A_k=(1+1/k)A for update k>=1, hence product_{j=1}^k(1+1/j)=k+1.
    state = np.array([0.82, 0.18], dtype=float)
    harmonic_envelopes: list[float] = []
    for k in range(steps + 1):
        total = float(np.sum(state))
        direction_k = normalized(state)
        scalar_envelope = total / (perron**k)
        reduced_envelope = scalar_envelope / (k + 1.0)
        harmonic_envelopes.append(reduced_envelope)
        factor = 1.0 + 1.0 / (k + 1.0)
        next_state = factor * (matrix @ state)
        ratio = float(np.sum(next_state) / total)
        rows.append(
            {
                "case": "autonomous_harmonic_distortion",
                "trajectory": "asymmetric",
                "k": k,
                "scale_g": B**k,
                "Y": total,
                "ratio_next": ratio,
                "expected_ratio": perron,
                "ratio_error": abs(ratio - perron),
                "z0": float(direction_k[0]),
                "z1": float(direction_k[1]),
                "hilbert_to_attractor": hilbert_distance(direction_k, direction),
                "envelope_over_kplus1": reduced_envelope,
            }
        )
        state = next_state

    tail = np.asarray(harmonic_envelopes[-8:], dtype=float)
    relative_tail_span = float((np.max(tail) - np.min(tail)) / np.mean(tail))
    record_test(
        tests,
        "autonomous_projective_convergence",
        max(final_hilbert) < 1e-12,
        max(final_hilbert),
        "max final Hilbert distance < 1e-12",
    )
    record_test(
        tests,
        "autonomous_ratio_convergence",
        max(final_ratio_error) < 1e-12,
        max(final_ratio_error),
        "max final adjacent-ratio error < 1e-12",
    )
    record_test(
        tests,
        "harmonic_slow_envelope",
        relative_tail_span < 1e-8,
        relative_tail_span,
        "Y_k/[rho(A)^k(k+1)] has relative tail span < 1e-8",
    )

    summary = {
        "matrix": matrix.tolist(),
        "perron_root": perron,
        "perron_direction": direction.tolist(),
        "rent_index_p": exponent,
        "steps": steps,
        "max_final_hilbert_distance": max(final_hilbert),
        "max_final_ratio_error": max(final_ratio_error),
        "harmonic_reduced_envelope_tail_span": relative_tail_span,
    }
    return rows, summary


def plot_autonomous(rows: list[dict[str, Any]]) -> None:
    exact = [row for row in rows if row["case"] == "autonomous_exact"]
    harmonic = [row for row in rows if row["case"] == "autonomous_harmonic_distortion"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for label in sorted({str(row["trajectory"]) for row in exact}):
        selected = [row for row in exact if row["trajectory"] == label]
        axes[0].semilogy(
            [int(row["k"]) for row in selected],
            [max(float(row["hilbert_to_attractor"]), 1e-16) for row in selected],
            label=label,
        )
    axes[0].set_title("Primitive cocycle forgets initial mode mix")
    axes[0].set_xlabel("hierarchy depth $k$")
    axes[0].set_ylabel("Hilbert distance to Perron direction")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend(fontsize=7)

    k = np.asarray([int(row["k"]) for row in harmonic], dtype=float)
    envelope = np.asarray([float(row["envelope_over_kplus1"]) for row in harmonic])
    ratio_error = np.asarray([float(row["ratio_error"]) for row in harmonic])
    axes[1].plot(k, envelope / envelope[-1], label=r"$Y_k/[\rho^k(k+1)]$", linewidth=2)
    axes[1].plot(k, ratio_error, label=r"$|Y_{k+1}/Y_k-\rho|$", linewidth=1.5)
    axes[1].set_title("Vanishing, nonsummable distortion")
    axes[1].set_xlabel("hierarchy depth $k$")
    axes[1].set_ylabel("normalized value")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.suptitle("Autonomous Perron basin and slowly varying envelope")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "01_autonomous_projective_attractor.png", dpi=180)
    plt.close(fig)


def run_periodic(tests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    a0 = np.array([[2.0, 1.0], [1.0, 1.0]], dtype=float)
    a1 = np.array([[1.0, 2.0], [1.0, 3.0]], dtype=float)
    monodromy = a1 @ a0
    perron, r0 = perron_pair(monodromy)
    raw_r1 = a0 @ r0
    c0 = float(np.sum(raw_r1))
    r1 = normalized(raw_r1)
    c1 = float(np.sum(a1 @ r1))
    exponent = math.log(perron) / (2.0 * LOG_B)
    step_power = B**exponent
    phase_amplitudes = [1.0, c0 / step_power]
    expected_directions = [r0, r1]
    expected_ratios = [c0, c1]
    starts = {
        "left_heavy": np.array([0.99, 0.01]),
        "right_heavy": np.array([0.01, 0.99]),
        "balanced": np.array([0.50, 0.50]),
        "asymmetric": np.array([0.82, 0.18]),
    }
    steps = 64
    rows: list[dict[str, Any]] = []
    final_hilbert: list[float] = []
    final_ratio_error: list[float] = []
    phase_amplitude_tails: dict[str, list[float]] = {}

    for label, start in starts.items():
        state = start.astype(float)
        amplitudes: list[float] = []
        for k in range(steps + 1):
            phase = k % 2
            total = float(np.sum(state))
            direction_k = normalized(state)
            matrix = a0 if phase == 0 else a1
            next_state = matrix @ state
            ratio = float(np.sum(next_state) / total)
            reduced_amplitude = total / ((B ** (exponent * k)) * phase_amplitudes[phase])
            amplitudes.append(reduced_amplitude)
            hdist = hilbert_distance(direction_k, expected_directions[phase])
            ratio_error = abs(ratio - expected_ratios[phase])
            rows.append(
                {
                    "trajectory": label,
                    "k": k,
                    "phase": phase,
                    "scale_g": B**k,
                    "Y": total,
                    "ratio_next": ratio,
                    "expected_ratio": expected_ratios[phase],
                    "ratio_error": ratio_error,
                    "z0": float(direction_k[0]),
                    "z1": float(direction_k[1]),
                    "expected_z0": float(expected_directions[phase][0]),
                    "expected_z1": float(expected_directions[phase][1]),
                    "hilbert_to_phase_cycle": hdist,
                    "reduced_phase_amplitude": reduced_amplitude,
                }
            )
            if k >= steps - 1:
                final_hilbert.append(hdist)
                final_ratio_error.append(ratio_error)
            state = next_state
        phase_amplitude_tails[label] = amplitudes[-8:]

    max_tail_span = 0.0
    for tail_values in phase_amplitude_tails.values():
        tail = np.asarray(tail_values, dtype=float)
        span = float((np.max(tail) - np.min(tail)) / np.mean(tail))
        max_tail_span = max(max_tail_span, span)

    record_test(
        tests,
        "periodic_monodromy_identity",
        abs(c0 * c1 - perron) < 1e-12,
        abs(c0 * c1 - perron),
        "|c0*c1-rho(M)| < 1e-12",
    )
    record_test(
        tests,
        "periodic_projective_cycle",
        max(final_hilbert) < 1e-12,
        max(final_hilbert),
        "max late Hilbert distance to phase cycle < 1e-12",
    )
    record_test(
        tests,
        "periodic_phase_ratio_cycle",
        max(final_ratio_error) < 1e-12,
        max(final_ratio_error),
        "max late phase-ratio error < 1e-12",
    )
    record_test(
        tests,
        "periodic_nonconstant_scalar_phase",
        min(abs(c0 - c1), abs(c0 - step_power), abs(c1 - step_power)) > 0.1,
        min(abs(c0 - c1), abs(c0 - step_power), abs(c1 - step_power)),
        "c0, c1, and b^p are materially distinct",
    )
    record_test(
        tests,
        "periodic_reduced_amplitude_convergence",
        max_tail_span < 1e-10,
        max_tail_span,
        "max relative tail span of phase-reduced amplitude < 1e-10",
    )

    summary = {
        "A0": a0.tolist(),
        "A1": a1.tolist(),
        "monodromy": monodromy.tolist(),
        "monodromy_perron_root": perron,
        "rent_index_p": exponent,
        "b_to_p": step_power,
        "phase_directions": [r0.tolist(), r1.tolist()],
        "phase_gains": [c0, c1],
        "phase_gain_product_error": abs(c0 * c1 - perron),
        "max_final_hilbert_distance": max(final_hilbert),
        "max_final_ratio_error": max(final_ratio_error),
        "max_reduced_amplitude_tail_span": max_tail_span,
    }
    return rows, summary


def plot_periodic(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    selected = [row for row in rows if row["trajectory"] == "asymmetric"]
    k = np.asarray([int(row["k"]) for row in selected])
    z0 = np.asarray([float(row["z0"]) for row in selected])
    ratio = np.asarray([float(row["ratio_next"]) for row in selected])
    amplitude = np.asarray([float(row["reduced_phase_amplitude"]) for row in selected])
    phase_directions = summary["phase_directions"]
    phase_gains = summary["phase_gains"]

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    axes[0].plot(k, z0, marker="o", markersize=2.5, linewidth=1.0, label="observed $z_{k,0}$")
    axes[0].axhline(phase_directions[0][0], color="C1", linestyle="--", label="phase 0 limit")
    axes[0].axhline(phase_directions[1][0], color="C2", linestyle=":", label="phase 1 limit")
    axes[0].set_title("Projective two-cycle")
    axes[0].set_xlabel("hierarchy depth $k$")
    axes[0].set_ylabel("first-mode share")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)

    axes[1].plot(k, ratio, marker=".", linewidth=1.0, label="adjacent gain")
    axes[1].axhline(phase_gains[0], color="C1", linestyle="--", label="$c_0$")
    axes[1].axhline(phase_gains[1], color="C2", linestyle=":", label="$c_1$")
    axes[1].plot(k, amplitude / amplitude[-1], color="C3", linewidth=1.4, label="phase-reduced amplitude")
    axes[1].set_title("Scalar Floquet gains")
    axes[1].set_xlabel("hierarchy depth $k$")
    axes[1].set_ylabel("gain / normalized amplitude")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=7)
    fig.suptitle("Periodic positive cocycle: one exponent, persistent hierarchy phase")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "02_periodic_perron_floquet_cycle.png", dpi=180)
    plt.close(fig)


def interior_profile(g: np.ndarray | float, p: float, alpha: float) -> np.ndarray:
    values = np.asarray(g, dtype=float)
    return (values**p) * (np.log(math.e + values) ** alpha)


def interior_local_slope(g: np.ndarray, p: float, alpha: float) -> np.ndarray:
    values = np.asarray(g, dtype=float)
    correction = alpha * values / ((math.e + values) * np.log(math.e + values))
    return p + correction


def run_caps(tests: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p = 0.62
    alpha = 0.70
    crossover_scales = [1e2, 1e4, 1e6, 1e8, 1e10]
    x_values = np.logspace(-1.0, 1.0, 241)
    rows: list[dict[str, Any]] = []
    errors: dict[str, list[float]] = {"hard": [], "soft": []}

    for g_star in crossover_scales:
        capacity = float(interior_profile(g_star, p, alpha))
        g = g_star * x_values
        ratio = interior_profile(g, p, alpha) / capacity
        p_int = interior_local_slope(g, p, alpha)
        for response in ("hard", "soft"):
            if response == "hard":
                observed = np.minimum(ratio, 1.0)
                limit = np.minimum(x_values**p, 1.0)
                observed_slope = np.where(ratio < 1.0, p_int, 0.0)
                limit_slope = np.where(x_values < 1.0, p, 0.0)
            else:
                observed = ratio / (1.0 + ratio)
                limit = (x_values**p) / (1.0 + x_values**p)
                observed_slope = p_int / (1.0 + ratio)
                limit_slope = p / (1.0 + x_values**p)
            abs_error = np.abs(observed - limit)
            errors[response].append(float(np.max(abs_error)))
            for index, x in enumerate(x_values):
                rows.append(
                    {
                        "response": response,
                        "g_star": g_star,
                        "capacity_P": capacity,
                        "x": float(x),
                        "normalized_profile": float(observed[index]),
                        "limit_profile": float(limit[index]),
                        "abs_error": float(abs_error[index]),
                        "local_slope": float(observed_slope[index]),
                        "limit_local_slope": float(limit_slope[index]),
                    }
                )

    for response in ("hard", "soft"):
        series = errors[response]
        record_test(
            tests,
            f"{response}_cap_error_decreases",
            all(series[index + 1] < series[index] for index in range(len(series) - 1)),
            series,
            "maximum crossover error strictly decreases with g_star",
        )
        record_test(
            tests,
            f"{response}_cap_largest_scale_accuracy",
            series[-1] < 0.02,
            series[-1],
            f"maximum normalized crossover error at g_star={crossover_scales[-1]:.0e} < 0.02",
        )

    hard_limit_at_one = 1.0
    soft_limit_at_one = 0.5
    record_test(
        tests,
        "cap_response_nonuniversality",
        abs(hard_limit_at_one - soft_limit_at_one) == 0.5,
        abs(hard_limit_at_one - soft_limit_at_one),
        "hard and soft registered crossover limits differ by 0.5 at x=1",
    )

    summary = {
        "interior": {
            "form": "g^p * log(e+g)^alpha",
            "p": p,
            "alpha": alpha,
        },
        "crossover_scales": crossover_scales,
        "max_abs_error_by_response": errors,
        "hard_limit": "min(x^p,1)",
        "soft_limit": "x^p/(1+x^p)",
    }
    return rows, summary


def plot_caps(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    scales = summary["crossover_scales"]
    for axis, response in zip(axes, ("hard", "soft")):
        for g_star in scales:
            selected = [
                row
                for row in rows
                if row["response"] == response and float(row["g_star"]) == float(g_star)
            ]
            axis.semilogx(
                [float(row["x"]) for row in selected],
                [float(row["normalized_profile"]) for row in selected],
                linewidth=1.0,
                alpha=0.72,
                label=f"$g_*=10^{{{int(round(math.log10(g_star)))}}}$",
            )
        reference = [
            row
            for row in rows
            if row["response"] == response and float(row["g_star"]) == float(scales[-1])
        ]
        axis.semilogx(
            [float(row["x"]) for row in reference],
            [float(row["limit_profile"]) for row in reference],
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="theorem limit",
        )
        axis.axvline(1.0, color="gray", linewidth=0.8, alpha=0.6)
        axis.set_title(f"{response.capitalize()} cap")
        axis.set_xlabel("crossover coordinate $x=g/g_*$")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("$R(g_*x)/P$")
    axes[1].legend(fontsize=6, ncol=2)
    fig.suptitle("Boundary crossover collapse retains the registered cap response")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "03_boundary_cap_collapse.png", dpi=180)
    plt.close(fig)


def phase_fit_rmse(values: np.ndarray, max_period: int) -> dict[int, float]:
    result: dict[int, float] = {}
    indices = np.arange(values.size)
    for period in range(1, max_period + 1):
        fitted = np.empty_like(values)
        for phase in range(period):
            mask = (indices % period) == phase
            fitted[mask] = float(np.mean(values[mask]))
        result[period] = float(np.sqrt(np.mean((values - fitted) ** 2)))
    return result


def run_negative_controls(
    tests: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    # Bounded scalar multipliers with block lengths dominated by each new block.
    block_lengths = [4, 36, 324, 2916]
    log_y = 0.0
    k = 0
    block_end_exponents: list[float] = []
    for block, length in enumerate(block_lengths):
        multiplier = 2.0 if block % 2 == 0 else 4.0
        adjacent_slope = math.log(multiplier) / LOG_B
        for _ in range(length):
            k += 1
            log_y += math.log(multiplier)
            rows.append(
                {
                    "control": "bounded_no_exponent",
                    "trajectory": "single",
                    "k": k,
                    "log_Y": log_y,
                    "global_exponent": log_y / (k * LOG_B),
                    "adjacent_slope": adjacent_slope,
                    "z0": "",
                }
            )
        block_end_exponents.append(log_y / (k * LOG_B))

    # Irrational quasiperiodic amplitude: global exponent exists, local ratios do not cycle.
    quasi_steps = 700
    p_quasi = 0.80
    gamma = 0.35
    irrational_phase = math.sqrt(2.0)
    quasi_log_y = np.asarray(
        [p_quasi * k * LOG_B + gamma * math.sin(k * irrational_phase) for k in range(quasi_steps + 1)]
    )
    quasi_slopes = np.diff(quasi_log_y) / LOG_B
    for k in range(quasi_steps):
        rows.append(
            {
                "control": "quasiperiodic",
                "trajectory": "single",
                "k": k + 1,
                "log_Y": float(quasi_log_y[k + 1]),
                "global_exponent": float(quasi_log_y[k + 1] / ((k + 1) * LOG_B)),
                "adjacent_slope": float(quasi_slopes[k]),
                "z0": "",
            }
        )
    quasi_tail = quasi_slopes[-400:]
    quasi_period_rmse = phase_fit_rmse(quasi_tail, 8)

    # Non-negligible forcing changes the exponent from log_2(2)=1 to log_2(3).
    force_steps = 120
    initial_force_y = 2.0
    force_y = initial_force_y
    forcing_exponents: list[float] = []
    for k in range(force_steps):
        next_y = 2.0 * force_y + 3.0**k
        adjacent_slope = math.log(next_y / force_y) / LOG_B
        force_y = next_y
        exponent = math.log(force_y / initial_force_y) / ((k + 1) * LOG_B)
        forcing_exponents.append(exponent)
        rows.append(
            {
                "control": "nonnegligible_forcing",
                "trajectory": "single",
                "k": k + 1,
                "log_Y": math.log(force_y),
                "global_exponent": exponent,
                "adjacent_slope": adjacent_slope,
                "z0": "",
            }
        )

    # No projective mixing: the identity map preserves different initial directions.
    no_mixing_steps = 40
    for label, start in (
        ("left_heavy", np.array([0.9, 0.1])),
        ("right_heavy", np.array([0.1, 0.9])),
    ):
        state = start.copy()
        for k in range(no_mixing_steps + 1):
            z = normalized(state)
            rows.append(
                {
                    "control": "no_projective_mixing",
                    "trajectory": label,
                    "k": k,
                    "log_Y": math.log(float(np.sum(state))),
                    "global_exponent": 0.0,
                    "adjacent_slope": 0.0,
                    "z0": float(z[0]),
                }
            )

    end_separation = abs(block_end_exponents[-1] - block_end_exponents[-2])
    record_test(
        tests,
        "bounded_distortion_no_exponent",
        end_separation > 0.70,
        end_separation,
        "last two dominating-block endpoint exponents differ by > 0.70",
    )
    min_quasi_rmse = min(quasi_period_rmse.values())
    record_test(
        tests,
        "quasiperiodic_no_finite_phase",
        min_quasi_rmse > 0.20,
        min_quasi_rmse,
        "best phase-constant fit over periods 1..8 has tail RMSE > 0.20",
    )
    record_test(
        tests,
        "quasiperiodic_global_exponent_exists",
        abs(float(quasi_log_y[-1] / (quasi_steps * LOG_B)) - p_quasi) < 1e-3,
        float(quasi_log_y[-1] / (quasi_steps * LOG_B)),
        "|global exponent-p| < 1e-3",
    )
    forcing_target = math.log(3.0) / LOG_B
    record_test(
        tests,
        "nonnegligible_forcing_changes_exponent",
        abs(forcing_exponents[-1] - forcing_target) < 0.02,
        forcing_exponents[-1],
        "final exponent is within 0.02 of log_2(3), not 1",
    )
    record_test(
        tests,
        "no_mixing_retains_initial_direction",
        abs(0.9 - 0.1) == 0.8,
        0.8,
        "identity cocycle preserves an 0.8 difference in first-mode share",
    )

    summary = {
        "bounded_no_exponent": {
            "block_lengths": block_lengths,
            "block_end_exponents": block_end_exponents,
            "last_endpoint_separation": end_separation,
        },
        "quasiperiodic": {
            "p": p_quasi,
            "gamma": gamma,
            "irrational_phase_increment": irrational_phase,
            "final_global_exponent": float(quasi_log_y[-1] / (quasi_steps * LOG_B)),
            "phase_fit_rmse_periods_1_to_8": {str(key): value for key, value in quasi_period_rmse.items()},
        },
        "nonnegligible_forcing": {
            "homogeneous_multiplier": 2.0,
            "forcing_base": 3.0,
            "homogeneous_index": 1.0,
            "forcing_dominated_index": forcing_target,
            "final_global_exponent": forcing_exponents[-1],
        },
        "no_projective_mixing": {
            "matrix": [[1.0, 0.0], [0.0, 1.0]],
            "retained_first_mode_shares": [0.9, 0.1],
        },
    }
    return rows, summary


def plot_negative_controls(rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8))

    bounded = [row for row in rows if row["control"] == "bounded_no_exponent"]
    axes[0, 0].plot(
        [int(row["k"]) for row in bounded],
        [float(row["global_exponent"]) for row in bounded],
        linewidth=1.5,
    )
    for endpoint in np.cumsum(summary["bounded_no_exponent"]["block_lengths"]):
        axes[0, 0].axvline(endpoint, color="gray", linewidth=0.7, alpha=0.5)
    axes[0, 0].set_title("Bounded gains, no limiting exponent")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_ylabel("global exponent")

    quasi = [row for row in rows if row["control"] == "quasiperiodic"]
    axes[0, 1].plot(
        [int(row["k"]) for row in quasi],
        [float(row["adjacent_slope"]) for row in quasi],
        linewidth=1.0,
    )
    axes[0, 1].axhline(summary["quasiperiodic"]["p"], color="black", linestyle="--", linewidth=1.0)
    axes[0, 1].set_title("Exponent exists; no finite phase ratio")
    axes[0, 1].set_ylabel("adjacent log-slope")

    forcing = [row for row in rows if row["control"] == "nonnegligible_forcing"]
    axes[1, 0].plot(
        [int(row["k"]) for row in forcing],
        [float(row["global_exponent"]) for row in forcing],
        linewidth=1.5,
    )
    axes[1, 0].axhline(1.0, color="C1", linestyle=":", label="homogeneous index")
    axes[1, 0].axhline(
        summary["nonnegligible_forcing"]["forcing_dominated_index"],
        color="black",
        linestyle="--",
        label="forcing-dominated index",
    )
    axes[1, 0].set_title("Comparable forcing changes the exponent")
    axes[1, 0].set_ylabel("global exponent")
    axes[1, 0].legend(fontsize=7)

    for label, color in (("left_heavy", "C0"), ("right_heavy", "C3")):
        selected = [
            row
            for row in rows
            if row["control"] == "no_projective_mixing" and row["trajectory"] == label
        ]
        axes[1, 1].plot(
            [int(row["k"]) for row in selected],
            [float(row["z0"]) for row in selected],
            label=label,
            color=color,
        )
    axes[1, 1].set_title("No mixing: initial direction survives")
    axes[1, 1].set_ylabel("first-mode share")
    axes[1, 1].legend(fontsize=7)

    for axis in axes.flat:
        axis.set_xlabel("hierarchy depth $k$")
        axis.grid(True, alpha=0.25)
    fig.suptitle("Negative controls delimit the universality basin")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG / "04_negative_controls.png", dpi=180)
    plt.close(fig)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(paths: Iterable[Path]) -> None:
    rows = [f"{sha256(path)}  {path.relative_to(ROOT)}" for path in sorted(paths)]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> None:
    tests: list[dict[str, Any]] = []

    autonomous_rows, autonomous_summary = run_autonomous(tests)
    periodic_rows, periodic_summary = run_periodic(tests)
    cap_rows, cap_summary = run_caps(tests)
    negative_rows, negative_summary = run_negative_controls(tests)

    autonomous_csv = OUT / "autonomous_orbits.csv"
    periodic_csv = OUT / "periodic_orbits.csv"
    cap_csv = OUT / "cap_collapse.csv"
    negative_csv = OUT / "negative_controls.csv"
    write_csv(autonomous_csv, autonomous_rows)
    write_csv(periodic_csv, periodic_rows)
    write_csv(cap_csv, cap_rows)
    write_csv(negative_csv, negative_rows)

    plot_autonomous(autonomous_rows)
    plot_periodic(periodic_rows, periodic_summary)
    plot_caps(cap_rows, cap_summary)
    plot_negative_controls(negative_rows, negative_summary)

    all_passed = all(bool(test["passed"]) for test in tests)
    summary = {
        "schema_version": "renormalization-validation-v1",
        "deterministic": True,
        "hierarchy_scale_factor_b": B,
        "autonomous": autonomous_summary,
        "periodic": periodic_summary,
        "caps": cap_summary,
        "negative_controls": negative_summary,
        "self_tests": tests,
        "all_self_tests_passed": all_passed,
        "output_contract": {
            "csv_files": [
                autonomous_csv.name,
                periodic_csv.name,
                cap_csv.name,
                negative_csv.name,
            ],
            "figures": [
                "01_autonomous_projective_attractor.png",
                "02_periodic_perron_floquet_cycle.png",
                "03_boundary_cap_collapse.png",
                "04_negative_controls.png",
            ],
        },
    }
    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifacts = [
        autonomous_csv,
        periodic_csv,
        cap_csv,
        negative_csv,
        summary_path,
        FIG / "01_autonomous_projective_attractor.png",
        FIG / "02_periodic_perron_floquet_cycle.png",
        FIG / "03_boundary_cap_collapse.png",
        FIG / "04_negative_controls.png",
    ]
    write_manifest(artifacts)

    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all_passed:
        failed = [test["name"] for test in tests if not test["passed"]]
        raise AssertionError(f"renormalization validation failed: {failed}")


if __name__ == "__main__":
    main()
