# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return self.x - margin <= x <= self.x2 + margin and self.y - margin <= y <= self.y2 + margin


@dataclass
class Chiplet:
    name: str
    rect: Rect
    power_w: float
    tier: int = 0
    process: str = "logic"
    test_access_lanes: int = 16
    available_shoreline_mm: Optional[float] = None

    def __post_init__(self) -> None:
        if self.available_shoreline_mm is None:
            self.available_shoreline_mm = 2 * (self.rect.w + self.rect.h)


@dataclass
class Port:
    name: str
    chiplet: str
    x: float
    y: float
    tier: int = 0


@dataclass
class NetBundle:
    name: str
    source: str
    sink: str
    bandwidth_GBs: float
    bytes_per_operation: int
    lane_units: int
    kind: str
    energy_pj_per_bit: float
    protocol_latency_ns: float
    vertical: bool = False
    routed_path: List[Tuple[int, int]] = field(default_factory=list)
    routed_length_mm: float = 0.0
    max_edge_utilization: float = 0.0
    overflow_units: float = 0.0


@dataclass
class PackageDesign:
    name: str
    package_w_mm: float
    package_h_mm: float
    grid_mm: float
    route_layers: int
    edge_capacity_lane_units_per_layer: int
    chiplets: List[Chiplet]
    ports: Dict[str, Port]
    nets: List[NetBundle]
    bump_pitch_um: float
    port_rows: int
    stack_depth: int
    notes: List[str]

    def to_dict(self):
        return asdict(self)
