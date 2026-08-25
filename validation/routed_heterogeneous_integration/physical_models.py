# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from package_models import Chiplet, NetBundle, PackageDesign


K_B_EV = 8.617333262e-5


def _cell_index(layer: int, i: int, j: int, nx: int, ny: int) -> int:
    return layer * nx * ny + j * nx + i


def solve_thermal(design: PackageDesign, ambient_c: float = 35.0) -> Dict[str, object]:
    dx = 1.0  # mm compact-model cell
    nx = int(math.ceil(design.package_w_mm / dx))
    ny = int(math.ceil(design.package_h_mm / dx))
    layers = max(1, design.stack_depth)
    n = nx * ny * layers
    A = lil_matrix((n, n), dtype=float)
    b = np.zeros(n, dtype=float)
    power = np.zeros((layers, ny, nx), dtype=float)

    for c in design.chiplets:
        layer = min(layers - 1, c.tier)
        cells = []
        for i in range(nx):
            x = (i + 0.5) * dx
            for j in range(ny):
                y = (j + 0.5) * dx
                if c.rect.contains(x, y):
                    cells.append((j, i))
        if not cells:
            continue
        per = c.power_w / len(cells)
        for j, i in cells:
            power[layer, j, i] += per

    g_lat = 0.22
    g_vert = 0.30
    for layer in range(layers):
        g_amb = 0.050 if layer == layers - 1 else 0.020
        if layers == 1:
            g_amb = 0.055
        for j in range(ny):
            for i in range(nx):
                idx = _cell_index(layer, i, j, nx, ny)
                diag = g_amb
                b[idx] = power[layer, j, i] + g_amb * ambient_c
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ii, jj = i + di, j + dj
                    if 0 <= ii < nx and 0 <= jj < ny:
                        nidx = _cell_index(layer, ii, jj, nx, ny)
                        A[idx, nidx] = -g_lat
                        diag += g_lat
                if layers > 1:
                    for ll in (layer - 1, layer + 1):
                        if 0 <= ll < layers:
                            nidx = _cell_index(ll, i, j, nx, ny)
                            A[idx, nidx] = -g_vert
                            diag += g_vert
                A[idx, idx] = diag
    temps = spsolve(A.tocsr(), b).reshape((layers, ny, nx))
    per_chip = {}
    for c in design.chiplets:
        layer = min(layers - 1, c.tier)
        vals = []
        for i in range(nx):
            x = (i + 0.5) * dx
            for j in range(ny):
                y = (j + 0.5) * dx
                if c.rect.contains(x, y):
                    vals.append(temps[layer, j, i])
        per_chip[c.name] = {
            "mean_c": float(np.mean(vals)) if vals else ambient_c,
            "max_c": float(np.max(vals)) if vals else ambient_c,
        }
    return {
        "temperature_map_c": temps,
        "max_temperature_c": float(np.max(temps)),
        "mean_temperature_c": float(np.mean(temps)),
        "per_chiplet": per_chip,
        "model": "steady-state finite-difference compact thermal-resistance network",
    }


def solve_pdn(design: PackageDesign, supply_v: float = 0.8) -> Dict[str, object]:
    dx = 1.0
    nx = int(math.ceil(design.package_w_mm / dx))
    ny = int(math.ceil(design.package_h_mm / dx))
    n = nx * ny
    fixed = set()
    for i in range(nx):
        fixed.add((i, 0)); fixed.add((i, ny - 1))
    for j in range(ny):
        fixed.add((0, j)); fixed.add((nx - 1, j))
    unknown = [(i, j) for j in range(ny) for i in range(nx) if (i, j) not in fixed]
    pos = {node: k for k, node in enumerate(unknown)}
    A = lil_matrix((len(unknown), len(unknown)), dtype=float)
    b = np.zeros(len(unknown), dtype=float)
    current = np.zeros((ny, nx), dtype=float)
    for c in design.chiplets:
        if c.tier != 0:
            continue
        cells = []
        for i in range(nx):
            x = (i + 0.5) * dx
            for j in range(ny):
                y = (j + 0.5) * dx
                if c.rect.contains(x, y):
                    cells.append((j, i))
        if cells:
            per = c.power_w / supply_v / len(cells)
            for j, i in cells:
                current[j, i] += per

    r_edge = 0.0014
    g = 1.0 / r_edge
    for (i, j), row in pos.items():
        diag = 0.0
        b[row] = -current[j, i]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (i + di, j + dj)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                continue
            diag += g
            if nb in fixed:
                b[row] += g * supply_v
            else:
                A[row, pos[nb]] = -g
        A[row, row] = diag
    v_unknown = spsolve(A.tocsr(), b)
    vmap = np.full((ny, nx), supply_v)
    for node, row in pos.items():
        vmap[node[1], node[0]] = v_unknown[row]

    # Buried tier adds a declared vertical-delivery drop.
    upper_power = sum(c.power_w for c in design.chiplets if c.tier > 0)
    power_bumps = 1600 if design.stack_depth > 1 else 1
    vertical_drop = (upper_power / supply_v / power_bumps) * 0.018 if upper_power else 0.0
    min_v = float(np.min(vmap) - vertical_drop)
    return {
        "voltage_map_v": vmap,
        "supply_v": supply_v,
        "min_voltage_v": min_v,
        "max_ir_drop_mv": (supply_v - min_v) * 1e3,
        "vertical_delivery_drop_mv": vertical_drop * 1e3,
        "model": "resistive package mesh with edge supply and declared vertical-bump resistance",
    }


def link_and_timing_metrics(design: PackageDesign, target_op_rate: float) -> Dict[str, object]:
    net_rows = []
    total_power = 0.0
    max_latency = 0.0
    total_signal_bumps = 0
    semantic_bridge_latency = 0.0
    worst_memory_latency = 0.0
    for net in design.nets:
        allocated_bw = max(net.bandwidth_GBs, net.lane_units * 2.0 * 0.85)
        serialization_ns = net.bytes_per_operation / (allocated_bw * 1e9) * 1e9
        propagation_ns = (0.015 * net.routed_length_mm) if not net.vertical else 0.02
        latency_ns = serialization_ns + propagation_ns + net.protocol_latency_ns
        actual_data_rate_GBs = net.bytes_per_operation * target_op_rate / 1e9
        power_w = actual_data_rate_GBs * 1e9 * 8 * net.energy_pj_per_bit * 1e-12
        total_power += power_w
        max_latency = max(max_latency, latency_ns)
        if "semantic_bridge" in net.name:
            semantic_bridge_latency = latency_ns
        if net.kind == "memory":
            worst_memory_latency = max(worst_memory_latency, latency_ns)
        bumps = 2 * net.lane_units + 8
        total_signal_bumps += bumps
        net_rows.append(
            {
                "net": net.name,
                "kind": net.kind,
                "provisioned_bandwidth_GBs": net.bandwidth_GBs,
                "actual_data_rate_GBs": actual_data_rate_GBs,
                "lane_units": net.lane_units,
                "length_mm": net.routed_length_mm,
                "serialization_ns": serialization_ns,
                "propagation_ns": propagation_ns,
                "protocol_ns": net.protocol_latency_ns,
                "latency_ns": latency_ns,
                "link_power_w": power_w,
                "max_edge_utilization": net.max_edge_utilization,
                "overflow_units": net.overflow_units,
                "signal_bumps": bumps,
            }
        )
    shoreline_mm = total_signal_bumps * design.bump_pitch_um / 1000.0 / design.port_rows
    required_interval_ns = 1e9 / target_op_rate
    return {
        "nets": net_rows,
        "total_link_power_w": total_power,
        "critical_link_latency_ns": max_latency,
        "target_operation_rate_ops_s": target_op_rate,
        "target_operation_interval_ns": required_interval_ns,
        "timing_slack_ns": required_interval_ns - max_latency,
        "timing_feasible": max_latency <= required_interval_ns,
        "semantic_bridge_latency_ns": semantic_bridge_latency,
        "worst_memory_latency_ns": worst_memory_latency,
        "total_signal_bumps": total_signal_bumps,
        "required_shoreline_mm": shoreline_mm,
    }


def yield_test_reliability(
    design: PackageDesign,
    max_temp_c: float,
    route_length_lane_mm: float,
    defect_density_cm2: float = 0.08,
    alpha: float = 3.0,
) -> Dict[str, float]:
    die_yields = []
    for c in design.chiplets:
        area_cm2 = c.rect.area / 100.0
        y = (1.0 + defect_density_cm2 * area_cm2 / alpha) ** (-alpha)
        die_yields.append(y)
    interposer_area_cm2 = design.package_w_mm * design.package_h_mm / 100.0
    interposer_yield = (1.0 + 0.025 * interposer_area_cm2 / alpha) ** (-alpha)
    attach_yield = 0.995 ** max(0, len(design.chiplets) - 1)
    bond_pairs = sum(1 for n in design.nets if n.vertical)
    vertical_bond_yield = 0.992 ** bond_pairs
    system_yield = float(np.prod(die_yields) * interposer_yield * attach_yield * vertical_bond_yield)

    total_test_lanes = sum(c.test_access_lanes for c in design.chiplets)
    required = 12 * len(design.chiplets) + 16 * max(0, design.stack_depth - 1)
    test_score = min(1.0, total_test_lanes / required) * math.exp(-0.10 * (design.stack_depth - 1))

    t_ref = 85.0 + 273.15
    t = max_temp_c + 273.15
    arrhenius_af = math.exp(0.7 / K_B_EV * (1 / t_ref - 1 / t))
    interconnect_penalty = 0.00002 * route_length_lane_mm + 0.08 * max(0, design.stack_depth - 1)
    reliability_risk = math.log10(max(arrhenius_af, 1e-12)) + interconnect_penalty
    return {
        "negative_binomial_system_yield_proxy": system_yield,
        "test_observability_score": test_score,
        "arrhenius_acceleration_factor_vs_85c": arrhenius_af,
        "reliability_risk_proxy": reliability_risk,
        "defect_density_cm2": defect_density_cm2,
        "cluster_parameter_alpha": alpha,
    }
