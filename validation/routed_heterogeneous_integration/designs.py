# SPDX-FileCopyrightText: 2026 Semi AI Foundry LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import math
from typing import Dict, Tuple

from package_models import Chiplet, NetBundle, PackageDesign, Port, Rect


def _lanes(bw_GBs: float) -> int:
    return max(1, math.ceil(bw_GBs / 2.0))


def build_designs() -> Dict[str, PackageDesign]:
    bridge_bw = 16.384
    attn_hbm_each = 37.8
    ffn_hbm_each = 52.2
    bridge_bytes = 32768
    attn_bytes_each = 75600
    ffn_bytes_each = 104400

    chip25 = [
        Chiplet("attention", Rect(5, 12, 10, 12), 92, tier=0, test_access_lanes=24),
        Chiplet("feed_forward", Rect(23, 12, 10, 12), 112, tier=0, test_access_lanes=24),
        Chiplet("controller_io", Rect(16, 2, 6, 6), 18, tier=0, test_access_lanes=20),
        Chiplet("hbm0", Rect(1, 27, 7, 9), 14, tier=0, process="memory", test_access_lanes=12),
        Chiplet("hbm1", Rect(9, 27, 7, 9), 14, tier=0, process="memory", test_access_lanes=12),
        Chiplet("hbm2", Rect(22, 27, 7, 9), 14, tier=0, process="memory", test_access_lanes=12),
        Chiplet("hbm3", Rect(30, 27, 7, 9), 14, tier=0, process="memory", test_access_lanes=12),
    ]
    ports25 = {
        "attn_bridge": Port("attn_bridge", "attention", 15.5, 18),
        "ffn_bridge": Port("ffn_bridge", "feed_forward", 22.5, 18),
        "attn_h0": Port("attn_h0", "attention", 7.5, 24.5),
        "attn_h1": Port("attn_h1", "attention", 12.5, 24.5),
        "ffn_h2": Port("ffn_h2", "feed_forward", 25.5, 24.5),
        "ffn_h3": Port("ffn_h3", "feed_forward", 30.5, 24.5),
        "h0": Port("h0", "hbm0", 4.5, 26.5),
        "h1": Port("h1", "hbm1", 12.5, 26.5),
        "h2": Port("h2", "hbm2", 25.5, 26.5),
        "h3": Port("h3", "hbm3", 33.5, 26.5),
        "ctrl_a": Port("ctrl_a", "controller_io", 16, 8.5),
        "ctrl_b": Port("ctrl_b", "attention", 9, 11.5),
        "ctrl_c": Port("ctrl_c", "feed_forward", 29, 11.5),
    }
    nets25 = [
        NetBundle("semantic_bridge", "attn_bridge", "ffn_bridge", bridge_bw, bridge_bytes, _lanes(bridge_bw), "activation", 0.35, 2.0),
        NetBundle("attention_hbm0", "attn_h0", "h0", attn_hbm_each, attn_bytes_each, _lanes(attn_hbm_each), "memory", 0.45, 2.5),
        NetBundle("attention_hbm1", "attn_h1", "h1", attn_hbm_each, attn_bytes_each, _lanes(attn_hbm_each), "memory", 0.45, 2.5),
        NetBundle("ffn_hbm2", "ffn_h2", "h2", ffn_hbm_each, ffn_bytes_each, _lanes(ffn_hbm_each), "memory", 0.45, 2.5),
        NetBundle("ffn_hbm3", "ffn_h3", "h3", ffn_hbm_each, ffn_bytes_each, _lanes(ffn_hbm_each), "memory", 0.45, 2.5),
        NetBundle("control_attention", "ctrl_a", "ctrl_b", 1.0, 256, 2, "control_test", 0.60, 4.0),
        NetBundle("control_ffn", "ctrl_a", "ctrl_c", 1.0, 256, 2, "control_test", 0.60, 4.0),
    ]
    d25 = PackageDesign(
        name="CRG-2p5D-side-by-side",
        package_w_mm=38,
        package_h_mm=38,
        grid_mm=0.5,
        route_layers=4,
        edge_capacity_lane_units_per_layer=24,
        chiplets=chip25,
        ports=ports25,
        nets=nets25,
        bump_pitch_um=55,
        port_rows=4,
        stack_depth=1,
        notes=["Synthetic open interposer floorplan", "Side-by-side specialist chiplets", "Four HBM stacks"],
    )

    chip3 = [
        Chiplet("base_feed_forward_io", Rect(10, 10, 12, 12), 126, tier=0, test_access_lanes=28),
        Chiplet("upper_attention", Rect(10, 10, 12, 12), 92, tier=1, test_access_lanes=16),
        Chiplet("hbm0", Rect(1, 23, 7, 8), 14, tier=0, process="memory", test_access_lanes=12),
        Chiplet("hbm1", Rect(8.5, 23, 7, 8), 14, tier=0, process="memory", test_access_lanes=12),
        Chiplet("hbm2", Rect(18.5, 23, 7, 8), 14, tier=0, process="memory", test_access_lanes=12),
        Chiplet("hbm3", Rect(26, 23, 7, 8), 14, tier=0, process="memory", test_access_lanes=12),
    ]
    ports3 = {
        "base_h0": Port("base_h0", "base_feed_forward_io", 11, 22.5),
        "base_h1": Port("base_h1", "base_feed_forward_io", 14, 22.5),
        "base_h2": Port("base_h2", "base_feed_forward_io", 18, 22.5),
        "base_h3": Port("base_h3", "base_feed_forward_io", 21, 22.5),
        "h0": Port("h0", "hbm0", 4.5, 22.5),
        "h1": Port("h1", "hbm1", 12, 22.5),
        "h2": Port("h2", "hbm2", 22, 22.5),
        "h3": Port("h3", "hbm3", 29.5, 22.5),
        "vertical_lower": Port("vertical_lower", "base_feed_forward_io", 16, 16, tier=0),
        "vertical_upper": Port("vertical_upper", "upper_attention", 16, 16, tier=1),
    }
    nets3 = [
        NetBundle("vertical_semantic_bridge", "vertical_upper", "vertical_lower", 128.0, bridge_bytes, _lanes(128.0), "activation", 0.08, 0.6, vertical=True, routed_length_mm=0.05),
        NetBundle("stack_hbm0", "base_h0", "h0", attn_hbm_each, attn_bytes_each, _lanes(attn_hbm_each), "memory", 0.40, 2.0),
        NetBundle("stack_hbm1", "base_h1", "h1", attn_hbm_each, attn_bytes_each, _lanes(attn_hbm_each), "memory", 0.40, 2.0),
        NetBundle("stack_hbm2", "base_h2", "h2", ffn_hbm_each, ffn_bytes_each, _lanes(ffn_hbm_each), "memory", 0.40, 2.0),
        NetBundle("stack_hbm3", "base_h3", "h3", ffn_hbm_each, ffn_bytes_each, _lanes(ffn_hbm_each), "memory", 0.40, 2.0),
    ]
    d3 = PackageDesign(
        name="CRG-3D-face-to-face",
        package_w_mm=34,
        package_h_mm=33,
        grid_mm=0.5,
        route_layers=4,
        edge_capacity_lane_units_per_layer=24,
        chiplets=chip3,
        ports=ports3,
        nets=nets3,
        bump_pitch_um=25,
        port_rows=6,
        stack_depth=2,
        notes=["Synthetic open face-to-face active-tier stack", "Vertical semantic bridge", "Four HBM stacks"],
    )
    return {d25.name: d25, d3.name: d3}
