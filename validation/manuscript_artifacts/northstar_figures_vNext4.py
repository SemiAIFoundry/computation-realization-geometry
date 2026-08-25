#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""Generate vNext4.1 Northstar figures using the stable vNext4 artifact names."""
from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "crg-matplotlib"))

import matplotlib.pyplot as plt
import numpy as np

import northstar_transformer_model_v3 as model


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "northstar_results_vNext4"
FIGURES = ROOT / "figures_northstar_vNext4"


def _save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def _load_json() -> dict:
    return json.loads((RESULTS / "northstar_results.json").read_text())


def class_certificates(data: dict) -> None:
    scenarios = data["certificate_scenarios"]
    order = [
        "H-static-fused",
        "U-universal-fused",
        "S-heterogeneous-split",
        "X-dual-mode-split-with-universal-fallback",
        "M-memory-centric-attention",
        "V-vertical-specialized",
    ]
    labels = {"H-static-fused": "H", "U-universal-fused": "U",
              "S-heterogeneous-split": "S",
              "X-dual-mode-split-with-universal-fallback": "X",
              "M-memory-centric-attention": "M",
              "V-vertical-specialized": "V"}
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.8))
    for ax, scenario in zip(axes, scenarios):
        values = [(name, scenario["class_times_ms"][name])
                  for name in order if name in scenario["class_times_ms"]]
        bars = ax.bar(
            range(len(values)),
            [value for _, value in values],
            color=["#8c8c8c" if name != scenario["winner"] else "#087e8b"
                   for name, _ in values],
        )
        ax.set_xticks(range(len(values)), [labels[name] for name, _ in values])
        ax.set_title(scenario["scenario"].replace(" / ", "\n"), fontsize=9)
        ax.set_ylabel("Initiation interval (ms)")
        ax.grid(axis="y", alpha=0.22)
        ax.bar_label(bars, fmt="%.3f", fontsize=7, padding=2)
        ax.text(
            0.98, 0.96,
            f"winner {scenario['winner'].split('-', 1)[0]}\n"
            f"nonhybrid gap {scenario['nonhybrid_gap_ms']:.3f} ms",
            ha="right", va="top", transform=ax.transAxes, fontsize=7.5,
        )
    fig.suptitle(
        "Default-contract class certificates at N=16,384\n"
        "GQA · block-causal 128 · 5-PF dense · shared aggregate package",
        fontsize=11,
    )
    fig.tight_layout()
    _save(fig, "class_certificates_vNext4")


def context_phase() -> None:
    contexts = np.array([2048, 4096, 8192, 13312, 16384, 32768, 65536,
                         131072])
    curves: dict[str, list[float]] = {
        "U universal fused": [],
        "S standard package": [],
        "S advanced package": [],
        "V vertical": [],
    }
    for n in contexts:
        u = model.best(model.universal_fused_candidates(
            int(n), "prefill", model.DEFAULT_CONTRACT, model.DEFAULT_FLAT
        ))
        s_std = model.best(model.split_candidates(
            int(n), "prefill", model.PACKAGES[0],
            model.DEFAULT_CONTRACT, model.DEFAULT_FLAT
        ))
        s_adv = model.best(model.split_candidates(
            int(n), "prefill", model.PACKAGES[2],
            model.DEFAULT_CONTRACT, model.DEFAULT_FLAT
        ))
        v = model.best(model.vertical_candidates(
            int(n), "prefill", model.PACKAGES[3],
            model.DEFAULT_CONTRACT, model.DEFAULT_FLAT
        ))
        curves["U universal fused"].append(u.initiation_interval_s * 1e3)
        curves["S standard package"].append(
            s_std.initiation_interval_s * 1e3
        )
        curves["S advanced package"].append(
            s_adv.initiation_interval_s * 1e3
        )
        curves["V vertical"].append(v.initiation_interval_s * 1e3)

    fig, ax = plt.subplots(figsize=(7.3, 4.6))
    styles = {
        "U universal fused": ("#087e8b", "o"),
        "S standard package": ("#d95f02", "s"),
        "S advanced package": ("#7570b3", "^"),
        "V vertical": ("#1b9e77", "D"),
    }
    for label, values in curves.items():
        color, marker = styles[label]
        ax.plot(contexts, values, marker=marker, color=color, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Context length N")
    ax.set_ylabel("Initiation interval (ms)")
    ax.set_title(
        "Context–package phase under the named default contract\n"
        "U is strictly faster than S inside the declared 128K domain"
    )
    ax.legend(fontsize=8)
    ax.grid(which="both", alpha=0.18)
    fig.tight_layout()
    _save(fig, "context_package_phase_vNext4")


def specialization_phase() -> None:
    etas = np.linspace(0.55, 0.97, 91)
    bandwidths = np.geomspace(32e9, 2048e9, 91)
    phase = np.empty((len(bandwidths), len(etas)), dtype=int)
    for row, bandwidth in enumerate(bandwidths):
        package = replace(
            model.PACKAGES[2],
            name="phase-sweep",
            payload_bytes_s=float(bandwidth),
        )
        split = model.best(model.split_candidates(
            16384, "prefill", package,
            model.DEFAULT_CONTRACT, model.DEFAULT_FLAT
        )).initiation_interval_s
        for col, eta in enumerate(etas):
            contract = replace(
                model.DEFAULT_CONTRACT,
                eta_universal=float(eta),
            )
            universal = model.best(model.universal_fused_candidates(
                16384, "prefill", contract, model.DEFAULT_FLAT
            )).initiation_interval_s
            phase[row, col] = 0 if universal <= split else 1

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    image = ax.pcolormesh(
        etas, bandwidths / 1e9, phase, shading="auto",
        cmap=plt.matplotlib.colors.ListedColormap(["#087e8b", "#7570b3"]),
        vmin=0, vmax=1,
    )
    ax.set_yscale("log")
    ax.set_xlabel("Universal attention efficiency")
    ax.set_ylabel("Package payload (GB/s)")
    ax.set_title(
        "Universal–specialized phase map at N=16,384\n"
        "GQA · block-causal 128 · 5-PF dense · shared aggregate package"
    )
    ax.axvline(0.80, color="white", linestyle="--", linewidth=1.2,
               label=r"declared $\eta_U=0.80$")
    ax.axhline(512, color="white", linestyle=":", linewidth=1.2,
               label="declared advanced payload")
    ax.scatter([0.80], [512], s=36, c="white", edgecolors="black", zorder=5)
    ax.legend(fontsize=8, loc="lower left")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0.25, 0.75])
    colorbar.ax.set_yticklabels(["U universal", "S specialized"])
    fig.tight_layout()
    _save(fig, "specialization_efficiency_phase_vNext4")


def flatattention_portability() -> None:
    rows: list[dict[str, str]] = []
    with (RESULTS / "flatattention_sensitivity.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows.extend(csv.DictReader(stream))
    filtered = [row for row in rows
                if math.isclose(float(row["flash_hbm_multiplier"]), 16.0)]
    utils = sorted({float(row["specialized_utilization"]) for row in filtered})
    ratios = sorted({
        float(row["specialized_over_flash_speed_ratio"]) for row in filtered
    })
    phase = np.empty((len(utils), len(ratios)), dtype=int)
    for i, utilization in enumerate(utils):
        for j, ratio in enumerate(ratios):
            row = next(
                item for item in filtered
                if math.isclose(float(item["specialized_utilization"]),
                                utilization)
                and math.isclose(
                    float(item["specialized_over_flash_speed_ratio"]), ratio
                )
            )
            phase[i, j] = 0 if row["winner"].startswith("U-") else 1

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    image = ax.imshow(
        phase, origin="lower", aspect="auto",
        cmap=plt.matplotlib.colors.ListedColormap(["#087e8b", "#7570b3"]),
        vmin=0, vmax=1,
    )
    ax.set_xticks(range(len(ratios)), [f"{value:g}" for value in ratios])
    ax.set_yticks(range(len(utils)), [f"{value:.3f}" for value in utils])
    ax.set_xlabel("Imported specialized / Flash speed ratio")
    ax.set_ylabel("Imported specialized utilization")
    ax.set_title(
        "FlatAttention portability sensitivity at N=16,384 and 512 GB/s\n"
        "Slice at Flash HBM multiplier = 16"
    )
    ax.scatter(
        [ratios.index(4.1)], [utils.index(0.893)],
        s=70, facecolors="none", edgecolors="white", linewidths=1.7,
        label="imported point",
    )
    ax.legend(fontsize=8, loc="lower right")
    colorbar = fig.colorbar(image, ax=ax, ticks=[0.25, 0.75])
    colorbar.ax.set_yticklabels(["U universal", "S specialized"])
    fig.tight_layout()
    _save(fig, "flatattention_portability_vNext4")


def layout_assignment() -> None:
    p = 12
    factors = [1, 2, 3, 4, 6, 12]
    moved = np.array([
        [model.aligned_layout_movement(p, c1, c2) for c2 in factors]
        for c1 in factors
    ])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    image = ax.imshow(moved, origin="lower", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(factors)), factors)
    ax.set_yticks(range(len(factors)), factors)
    ax.set_xlabel("Target sequence partitions C′")
    ax.set_ylabel("Source sequence partitions C")
    ax.set_title(
        "Exact assignment-based layout movement, P=12\n"
        "Nonnested factors expose the limit of the closed form"
    )
    for i, c1 in enumerate(factors):
        for j, c2 in enumerate(factors):
            suffix = "*" if not (c1 % c2 == 0 or c2 % c1 == 0) else ""
            ax.text(
                j, i, f"{moved[i, j]:.2f}{suffix}",
                ha="center", va="center", fontsize=7.5,
                color="white" if moved[i, j] > 0.42 else "black",
            )
    ax.text(
        0.01, -0.15, "* nonnested: maximum-weight assignment required",
        transform=ax.transAxes, fontsize=8,
    )
    fig.colorbar(image, ax=ax, label="Moved fraction")
    fig.tight_layout()
    _save(fig, "layout_assignment_vNext4")


def contract_sensitivity() -> None:
    rows: list[dict[str, str]] = []
    with (RESULTS / "contract_sensitivity.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows.extend(csv.DictReader(stream))
    rows = [row for row in rows if row["peak_basis"] == "dense-bf16"]
    labels = []
    values = []
    within = []
    for row in rows:
        policy = row["pair_policy"].replace("-causal", "").replace("-", " ")
        fabric = "shared" if row["package_accounting"].startswith("shared") \
            else "separate"
        labels.append(f"{policy}\n{fabric}")
        raw = row["audited_first_sustained_s_no_slower_context"]
        values.append(float(raw) / 1000.0)
        within.append(bool(
            row["declared_first_sustained_s_no_slower_context"]
        ))
    colors = ["#087e8b" if flag else "#b3b3b3" for flag in within]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    bars = ax.bar(range(len(values)), values, color=colors)
    ax.axhline(131.072, color="#d95f02", linestyle="--", linewidth=1.3,
               label="declared 128K limit")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_ylabel("First sustained S≤U point (thousand tokens)")
    ax.set_title(
        "Sustained standard-package dominance is contract dependent\n"
        "Dense BF16 peak; sustained means through the 1,048,576-token audit limit"
    )
    ax.bar_label(bars, fmt="%.1f", fontsize=7.5, padding=2)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    _save(fig, "contract_crossover_sensitivity_vNext4")


def main() -> None:
    data = _load_json()
    class_certificates(data)
    context_phase()
    specialization_phase()
    flatattention_portability()
    layout_assignment()
    contract_sensitivity()
    print(f"generated {len(list(FIGURES.glob('*')))} figure files in {FIGURES}")


if __name__ == "__main__":
    main()
