# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from designs import build_designs
from package_models import PackageDesign
from physical_models import link_and_timing_metrics, solve_pdn, solve_thermal, yield_test_reliability
from router import route_design


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
FIG = ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def plot_routes(design: PackageDesign, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    for c in design.chiplets:
        if c.tier != 0:
            continue
        rect = plt.Rectangle((c.rect.x, c.rect.y), c.rect.w, c.rect.h, fill=False, linewidth=1.5)
        ax.add_patch(rect)
        ax.text(c.rect.x + c.rect.w / 2, c.rect.y + c.rect.h / 2, c.name, ha="center", va="center", fontsize=8)
    for net in design.nets:
        if net.vertical:
            p = design.ports[net.source]
            ax.scatter([p.x], [p.y], marker="x", s=90, label=net.name)
            continue
        if net.routed_path:
            xs = [i * design.grid_mm for i, _ in net.routed_path]
            ys = [j * design.grid_mm for _, j in net.routed_path]
            ax.plot(xs, ys, linewidth=max(1.0, 0.6 + net.lane_units / 18), label=net.name)
    ax.set_xlim(0, design.package_w_mm)
    ax.set_ylim(0, design.package_h_mm)
    ax.set_aspect("equal")
    ax.set_xlabel("Package x (mm)")
    ax.set_ylabel("Package y (mm)")
    ax.set_title(f"Negotiated grid routes: {design.name}")
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_map(array: np.ndarray, title: str, label: str, path: Path) -> None:
    if array.ndim == 3:
        for layer in range(array.shape[0]):
            fig, ax = plt.subplots(figsize=(7.4, 5.6))
            im = ax.imshow(array[layer], origin="lower", aspect="auto")
            fig.colorbar(im, ax=ax, label=label)
            ax.set_title(f"{title} - tier {layer}")
            ax.set_xlabel("x cell")
            ax.set_ylabel("y cell")
            fig.tight_layout()
            fig.savefig(path.with_name(f"{path.stem}_tier{layer}{path.suffix}"), dpi=220)
            plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7.4, 5.6))
        im = ax.imshow(array, origin="lower", aspect="auto")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_title(title)
        ax.set_xlabel("x cell")
        ax.set_ylabel("y cell")
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


def main() -> None:
    designs = build_designs()
    all_results: Dict[str, object] = {}
    net_rows = []
    summary_rows = []
    target_op_rate = 0.45e6

    for name, design in designs.items():
        routing = route_design(design)
        link = link_and_timing_metrics(design, target_op_rate)
        # Link power contributes to the base tier or nearest physical owner.
        if design.chiplets:
            design.chiplets[0].power_w += link["total_link_power_w"]
        thermal = solve_thermal(design)
        pdn = solve_pdn(design)
        ytr = yield_test_reliability(
            design,
            float(thermal["max_temperature_c"]),
            float(routing["weighted_route_length_lane_mm"]),
        )
        silicon_area = sum(c.rect.area for c in design.chiplets if c.tier == 0)
        stacked_silicon_area = sum(c.rect.area for c in design.chiplets)
        total_power = sum(c.power_w for c in design.chiplets)
        result = {
            "design": design.to_dict(),
            "routing": routing,
            "link_timing_shoreline": link,
            "thermal": {k: v for k, v in thermal.items() if k != "temperature_map_c"},
            "pdn": {k: v for k, v in pdn.items() if k != "voltage_map_v"},
            "yield_test_reliability": ytr,
            "physical_ppa": {
                "package_area_mm2": design.package_w_mm * design.package_h_mm,
                "base_tier_silicon_area_mm2": silicon_area,
                "total_active_silicon_area_mm2": stacked_silicon_area,
                "total_power_w": total_power,
                "critical_link_latency_ns": link["critical_link_latency_ns"],
                "max_temperature_c": thermal["max_temperature_c"],
                "max_ir_drop_mv": pdn["max_ir_drop_mv"],
                "yield_proxy": ytr["negative_binomial_system_yield_proxy"],
                "test_score": ytr["test_observability_score"],
                "reliability_risk_proxy": ytr["reliability_risk_proxy"],
            },
            "evidence_scope": {
                "routing": "actual deterministic bundle routing on the released grid model",
                "timing": "serialization + route-length propagation + declared protocol latency",
                "thermal": "compact steady-state resistance network",
                "power_delivery": "resistive package mesh plus vertical bump drop",
                "yield": "negative-binomial die/interposer and declared assembly proxy",
                "test": "test-access observability proxy",
                "reliability": "Arrhenius temperature acceleration plus interconnect/stack penalty",
                "not_claimed": "signoff routing, electromigration, SI/PI signoff, measured silicon, production yield or product reliability",
            },
        }
        all_results[name] = result
        safe = name.replace("-", "_").replace(".", "_")
        (OUT / f"{safe}.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
        plot_routes(design, FIG / f"{safe}_routes.png")
        plot_map(np.asarray(thermal["temperature_map_c"]), f"Temperature map: {name}", "Temperature (C)", FIG / f"{safe}_thermal.png")
        plot_map(np.asarray(pdn["voltage_map_v"]), f"PDN voltage map: {name}", "Voltage (V)", FIG / f"{safe}_pdn.png")
        for row in link["nets"]:
            net_rows.append({"design": name, **row})
        summary_rows.append({"design": name, **result["physical_ppa"], **routing, "semantic_bridge_latency_ns": link["semantic_bridge_latency_ns"], "worst_memory_latency_ns": link["worst_memory_latency_ns"], "timing_slack_ns": link["timing_slack_ns"], "required_shoreline_mm": link["required_shoreline_mm"], "timing_feasible": link["timing_feasible"]})

    pd.DataFrame(net_rows).to_csv(OUT / "net_metrics.csv", index=False)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT / "design_comparison.csv", index=False)
    (OUT / "study_summary.json").write_text(json.dumps(all_results, indent=2, default=float), encoding="utf-8")

    # Comparison figure with normalized metrics (lower is better except yield/test).
    metrics = [
        "package_area_mm2",
        "critical_link_latency_ns",
        "max_temperature_c",
        "max_ir_drop_mv",
        "required_shoreline_mm",
        "reliability_risk_proxy",
    ]
    norm = summary_df.set_index("design")[metrics].copy()
    norm = norm / norm.max(axis=0)
    fig, ax = plt.subplots(figsize=(9.2, 5.7))
    x = np.arange(len(metrics))
    width = 0.35
    for idx, (design, row) in enumerate(norm.iterrows()):
        ax.bar(x + (idx - 0.5) * width, row.values, width=width, label=design)
    ax.set_xticks(x, [m.replace("_", "\n") for m in metrics], fontsize=8)
    ax.set_ylabel("Normalized burden (lower is better)")
    ax.set_title("2.5D and 3D exchange routing/shoreline for thermal and integration risk")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "design_tradeoff_comparison.png", dpi=220)
    plt.close(fig)

    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
