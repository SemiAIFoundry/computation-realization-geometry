# SPDX-FileCopyrightText: 2026 Semi AI Foundry, LLC
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
from __future__ import annotations

import heapq
import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from package_models import NetBundle, PackageDesign


GridNode = Tuple[int, int]
GridEdge = Tuple[GridNode, GridNode]


def canonical_edge(a: GridNode, b: GridNode) -> GridEdge:
    return (a, b) if a <= b else (b, a)


def nearest_node(x: float, y: float, grid: float, nx: int, ny: int) -> GridNode:
    return (min(nx - 1, max(0, int(round(x / grid)))), min(ny - 1, max(0, int(round(y / grid)))))


def obstacle_nodes(design: PackageDesign, tier: int = 0) -> Set[GridNode]:
    nx = int(round(design.package_w_mm / design.grid_mm)) + 1
    ny = int(round(design.package_h_mm / design.grid_mm)) + 1
    blocked: Set[GridNode] = set()
    for c in design.chiplets:
        if c.tier != tier:
            continue
        for i in range(nx):
            x = i * design.grid_mm
            for j in range(ny):
                y = j * design.grid_mm
                if c.rect.contains(x, y, margin=0.15 * design.grid_mm):
                    blocked.add((i, j))
    # Ports are legal endpoints and must be opened.
    for p in design.ports.values():
        if p.tier == tier:
            blocked.discard(nearest_node(p.x, p.y, design.grid_mm, nx, ny))
    return blocked


def astar_route(
    start: GridNode,
    goal: GridNode,
    nx: int,
    ny: int,
    blocked: Set[GridNode],
    usage: Dict[GridEdge, float],
    demand: float,
    capacity: float,
) -> List[GridNode]:
    def h(n: GridNode) -> float:
        return abs(n[0] - goal[0]) + abs(n[1] - goal[1])

    open_heap: List[Tuple[float, float, GridNode]] = [(h(start), 0.0, start)]
    came: Dict[GridNode, GridNode] = {}
    gscore: Dict[GridNode, float] = {start: 0.0}
    closed: Set[GridNode] = set()
    while open_heap:
        _, g, node = heapq.heappop(open_heap)
        if node in closed:
            continue
        if node == goal:
            path = [node]
            while node in came:
                node = came[node]
                path.append(node)
            return list(reversed(path))
        closed.add(node)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (node[0] + dx, node[1] + dy)
            if not (0 <= nb[0] < nx and 0 <= nb[1] < ny):
                continue
            if nb in blocked and nb not in (start, goal):
                continue
            edge = canonical_edge(node, nb)
            util_after = (usage.get(edge, 0.0) + demand) / capacity
            congestion = 8.0 * util_after**2
            overflow = 120.0 * max(0.0, util_after - 1.0) ** 2
            edge_cost = 1.0 + congestion + overflow
            tentative = g + edge_cost
            if tentative < gscore.get(nb, float("inf")):
                gscore[nb] = tentative
                came[nb] = node
                heapq.heappush(open_heap, (tentative + h(nb), tentative, nb))
    raise RuntimeError(f"No route from {start} to {goal}")


def route_design(design: PackageDesign) -> Dict[str, float]:
    nx = int(round(design.package_w_mm / design.grid_mm)) + 1
    ny = int(round(design.package_h_mm / design.grid_mm)) + 1
    usage: Dict[GridEdge, float] = defaultdict(float)
    capacity = design.route_layers * design.edge_capacity_lane_units_per_layer
    blocked = obstacle_nodes(design, tier=0)

    routed = sorted((n for n in design.nets if not n.vertical), key=lambda n: n.lane_units, reverse=True)
    for net in routed:
        ps = design.ports[net.source]
        pt = design.ports[net.sink]
        start = nearest_node(ps.x, ps.y, design.grid_mm, nx, ny)
        goal = nearest_node(pt.x, pt.y, design.grid_mm, nx, ny)
        path = astar_route(start, goal, nx, ny, blocked, usage, net.lane_units, capacity)
        net.routed_path = path
        net.routed_length_mm = max(0, len(path) - 1) * design.grid_mm
        utils = []
        overflow = 0.0
        for a, b in zip(path[:-1], path[1:]):
            edge = canonical_edge(a, b)
            usage[edge] += net.lane_units
            util = usage[edge] / capacity
            utils.append(util)
            overflow += max(0.0, usage[edge] - capacity)
        net.max_edge_utilization = max(utils, default=0.0)
        net.overflow_units = overflow

    total_weighted_length = sum(n.routed_length_mm * n.lane_units for n in design.nets if not n.vertical)
    max_util = max((u / capacity for u in usage.values()), default=0.0)
    overflow = sum(max(0.0, u - capacity) for u in usage.values())
    return {
        "grid_nodes": nx * ny,
        "routed_bundles": len(routed),
        "weighted_route_length_lane_mm": total_weighted_length,
        "max_edge_utilization": max_util,
        "total_overflow_lane_units": overflow,
        "edge_capacity_lane_units": capacity,
    }
