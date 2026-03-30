#!/usr/bin/env python3
"""
Dynamic map overlay helpers for runtime-changing maps.

This module uses a strategy registry keyed by location name so different maps
can use different runtime collision logic:

- Mauville Gym: live metatile reads from the map buffer (tile-level overlay)
- Fortree Gym: marks gate interaction zones with '?' for visual hint; actual
  collision is handled as a pathfinding edge constraint (see pathfinding.py)
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from utils.mapping.map_formatter import format_tile_to_symbol

logger = logging.getLogger(__name__)

# GBA map buffers include a 7-tile border around the real map.
GBA_MAP_BORDER_OFFSET = 7

# ── Fortree rotating-gate constants (shared with pathfinding) ──────────
# Script var IDs for rotating-gate orientation bytes.
VAR_TEMP_0_ID = 0x4000

FORTREE_GATE_CONFIG = [
    {"x": 6, "y": 7, "shape": "T2", "initial_orientation": 1},
    {"x": 9, "y": 15, "shape": "T2", "initial_orientation": 2},
    {"x": 3, "y": 19, "shape": "T2", "initial_orientation": 1},
    {"x": 2, "y": 6, "shape": "T1", "initial_orientation": 1},
    {"x": 9, "y": 12, "shape": "T1", "initial_orientation": 0},
    {"x": 6, "y": 23, "shape": "T1", "initial_orientation": 0},
    {"x": 12, "y": 22, "shape": "T1", "initial_orientation": 0},
    {"x": 6, "y": 3, "shape": "L4", "initial_orientation": 2},
]

FORTREE_ARM_LAYOUTS: Dict[str, List[Tuple[int, int]]] = {
    "T1": [(1, 0), (1, 0), (1, 0), (0, 0)],
    "T2": [(1, 1), (1, 0), (1, 0), (0, 0)],
    "L4": [(1, 1), (1, 1), (0, 0), (0, 0)],
}

ARM_DIR_VECTORS: List[Tuple[int, int]] = [
    (0, -1),  # north
    (1, 0),   # east
    (0, 1),   # south
    (-1, 0),  # west
]

DIR_NORTH = 0
DIR_SOUTH = 1
DIR_WEST = 2
DIR_EAST = 3

ROTATE_NONE = 0
ROTATE_ANTICLOCKWISE = 1
ROTATE_CLOCKWISE = 2

GATE_ARM_NORTH = 0
GATE_ARM_EAST = 1
GATE_ARM_SOUTH = 2
GATE_ARM_WEST = 3

ARM_POSITIONS_CLOCKWISE: List[Tuple[int, int]] = [
    (0, -1), (1, -2), (0, 0), (1, 0),
    (-1, 0), (-1, 1), (-1, -1), (-2, -1),
]
ARM_POSITIONS_ANTICLOCKWISE: List[Tuple[int, int]] = [
    (-1, -1), (-1, -2), (0, -1), (1, -1),
    (0, 0), (0, 1), (-1, 0), (-2, 0),
]

PRESERVE_SYMBOLS = {"P", "N", "I", "S", "D", "T", "K", "V"}

OverlayFn = Callable[[Dict[str, Any], Any, str], bool]


# ── Fortree gate helpers (used by overlay + pathfinding) ───────────────

def _gate_rot(rotation_direction: int, arm: int, is_long_arm: int) -> Tuple[int, int, int]:
    return (rotation_direction, arm, is_long_arm)


ROTATION_INFO_BY_DIRECTION: Dict[int, List[Tuple[int, int, int] | None]] = {
    DIR_NORTH: [
        None, None, None, None,
        _gate_rot(ROTATE_CLOCKWISE, GATE_ARM_WEST, 1),
        _gate_rot(ROTATE_CLOCKWISE, GATE_ARM_WEST, 0),
        _gate_rot(ROTATE_ANTICLOCKWISE, GATE_ARM_EAST, 0),
        _gate_rot(ROTATE_ANTICLOCKWISE, GATE_ARM_EAST, 1),
        None, None, None, None,
        None, None, None, None,
    ],
    DIR_SOUTH: [
        None, None, None, None,
        None, None, None, None,
        _gate_rot(ROTATE_ANTICLOCKWISE, GATE_ARM_WEST, 1),
        _gate_rot(ROTATE_ANTICLOCKWISE, GATE_ARM_WEST, 0),
        _gate_rot(ROTATE_CLOCKWISE, GATE_ARM_EAST, 0),
        _gate_rot(ROTATE_CLOCKWISE, GATE_ARM_EAST, 1),
        None, None, None, None,
    ],
    DIR_WEST: [
        None,
        _gate_rot(ROTATE_ANTICLOCKWISE, GATE_ARM_NORTH, 1),
        None, None, None,
        _gate_rot(ROTATE_ANTICLOCKWISE, GATE_ARM_NORTH, 0),
        None, None, None,
        _gate_rot(ROTATE_CLOCKWISE, GATE_ARM_SOUTH, 0),
        None, None, None,
        _gate_rot(ROTATE_CLOCKWISE, GATE_ARM_SOUTH, 1),
        None, None,
    ],
    DIR_EAST: [
        None, None,
        _gate_rot(ROTATE_CLOCKWISE, GATE_ARM_NORTH, 1),
        None, None, None,
        _gate_rot(ROTATE_CLOCKWISE, GATE_ARM_NORTH, 0),
        None, None, None,
        _gate_rot(ROTATE_ANTICLOCKWISE, GATE_ARM_SOUTH, 0),
        None, None, None,
        _gate_rot(ROTATE_ANTICLOCKWISE, GATE_ARM_SOUTH, 1),
        None,
    ],
}


def _is_impassable(grid: List[List[str]], x: int, y: int) -> bool:
    if y < 0 or y >= len(grid):
        return True
    if x < 0 or x >= len(grid[y]):
        return True
    return grid[y][x] == "#"


def _get_rotation_info(direction: int, center_x: int, center_y: int) -> Tuple[int, int, int] | None:
    table = ROTATION_INFO_BY_DIRECTION.get(direction)
    if table is None:
        return None
    if not (0 <= center_x < 4 and 0 <= center_y < 4):
        return None
    return table[center_y * 4 + center_x]


def _fortree_gate_has_arm(shape: str, orientation: int, arm: int, is_long_arm: int) -> bool:
    layout = FORTREE_ARM_LAYOUTS.get(shape)
    if not layout:
        return False
    arm_orientation = (arm - orientation + 4) % 4
    short_arm, long_arm = layout[arm_orientation]
    return bool(long_arm if is_long_arm else short_arm)


def _fortree_gate_can_rotate(
    gate: Dict[str, Any],
    orientation: int,
    rotation_direction: int,
    static_grid: List[List[str]],
) -> bool:
    if rotation_direction == ROTATE_ANTICLOCKWISE:
        arm_positions = ARM_POSITIONS_ANTICLOCKWISE
    elif rotation_direction == ROTATE_CLOCKWISE:
        arm_positions = ARM_POSITIONS_CLOCKWISE
    else:
        return False

    shape = str(gate["shape"])
    layout = FORTREE_ARM_LAYOUTS.get(shape)
    if not layout:
        return False

    gx = int(gate["x"])
    gy = int(gate["y"])

    for arm in range(4):
        for seg in range(2):
            if layout[arm][seg]:
                arm_index = 2 * ((orientation + arm) % 4) + seg
                dx, dy = arm_positions[arm_index]
                if _is_impassable(static_grid, gx + dx, gy + dy):
                    return False
    return True


def _read_fortree_gate_orientations(memory_reader: Any) -> List[int]:
    """Read 8 gate orientations from VAR_TEMP_0..VAR_TEMP_3 (packed bytes)."""
    orientations: List[int] = []
    var_word_cache: Dict[int, int | None] = {}

    for gate_id, gate in enumerate(FORTREE_GATE_CONFIG):
        var_id = VAR_TEMP_0_ID + (gate_id // 2)

        if var_id not in var_word_cache:
            read_var_fn = getattr(memory_reader, "read_var", None)
            var_word_cache[var_id] = read_var_fn(var_id) if callable(read_var_fn) else None

        word = var_word_cache[var_id]
        if word is None:
            orientations.append(int(gate["initial_orientation"]) % 4)
            continue

        byte_shift = 0 if (gate_id % 2) == 0 else 8
        orientation = (int(word) >> byte_shift) & 0xFF
        orientations.append(orientation % 4)

    return orientations


def fortree_movement_blocked(
    direction: int,
    target_x: int,
    target_y: int,
    orientations: List[int],
    static_grid: List[List[str]],
) -> bool:
    """Port of CheckForRotatingGatePuzzleCollisionWithoutAnimation.

    Public so pathfinding.py can call this as an edge constraint.
    """
    for gate_id, gate in enumerate(FORTREE_GATE_CONFIG):
        gx = int(gate["x"])
        gy = int(gate["y"])

        if not (gx - 2 <= target_x <= gx + 1 and gy - 2 <= target_y <= gy + 1):
            continue

        center_x = target_x - gx + 2
        center_y = target_y - gy + 2
        rotation_info = _get_rotation_info(direction, center_x, center_y)
        if rotation_info is None:
            continue

        rotation_direction, arm, is_long_arm = rotation_info
        orientation = orientations[gate_id] % 4
        shape = str(gate["shape"])
        if _fortree_gate_has_arm(shape, orientation, arm, is_long_arm):
            if not _fortree_gate_can_rotate(gate, orientation, rotation_direction, static_grid):
                return True
            return False

    return False


def get_fortree_gate_zone_cells() -> set[Tuple[int, int]]:
    """Return the set of all (x, y) cells inside any gate's 4x4 interaction zone."""
    cells: set[Tuple[int, int]] = set()
    for gate in FORTREE_GATE_CONFIG:
        gx = int(gate["x"])
        gy = int(gate["y"])
        for y in range(gy - 2, gy + 2):
            for x in range(gx - 2, gx + 2):
                cells.add((x, y))
    return cells


# ── Generic helpers ────────────────────────────────────────────────────

def is_dynamic_location(location_name: str) -> bool:
    """Return True if the given location needs a dynamic overlay."""
    return bool(location_name) and location_name.upper() in DYNAMIC_MAP_OVERLAYS


def _read_live_tiles(memory_reader: Any, width: int, height: int, location_name: str) -> List[List[Any]] | None:
    if memory_reader is None:
        logger.warning("Dynamic overlay skipped for %s: memory_reader unavailable", location_name)
        return None

    if not getattr(memory_reader, "_map_buffer_addr", None):
        find_buffer_fn = getattr(memory_reader, "_find_map_buffer_addresses", None)
        if not callable(find_buffer_fn) or not find_buffer_fn():
            logger.warning("Dynamic overlay skipped for %s: failed to locate map buffer", location_name)
            return None

    live_raw_tiles = memory_reader.read_map_metatiles(
        x_start=GBA_MAP_BORDER_OFFSET,
        y_start=GBA_MAP_BORDER_OFFSET,
        width=width,
        height=height,
    )
    if not live_raw_tiles:
        logger.warning("Dynamic overlay skipped for %s: live metatile read returned empty", location_name)
        return None
    return live_raw_tiles


def _build_live_grid(live_raw_tiles: List[List[Any]], location_name: str) -> List[List[str]]:
    live_grid: List[List[str]] = []
    for y, row in enumerate(live_raw_tiles):
        grid_row: List[str] = []
        for x, tile in enumerate(row):
            grid_row.append(format_tile_to_symbol(tile, x=x, y=y, location_name=location_name))
        live_grid.append(grid_row)
    return live_grid


def _clone_grid_as_lists(grid: Any) -> List[List[str]]:
    return [list(row) for row in grid]


def _update_ascii_if_present(map_payload: Dict[str, Any], grid: List[List[str]]) -> None:
    if "ascii" in map_payload:
        map_payload["ascii"] = "\n".join("".join(row) for row in grid)


# ── Mauville City Gym: live metatile overlay ──────────────────────────

def _apply_live_metatile_overlay_payload(map_payload: Dict[str, Any], memory_reader: Any, location_name: str) -> bool:
    """Overlay strategy for metatile-changing maps (e.g. Mauville Gym)."""
    dims = map_payload.get("dimensions") or {}
    width = int(dims.get("width", 0) or 0)
    height = int(dims.get("height", 0) or 0)
    if width <= 0 or height <= 0:
        logger.warning("Dynamic overlay skipped for %s: missing dimensions", location_name)
        return False

    live_raw_tiles = _read_live_tiles(memory_reader, width, height, location_name)
    if live_raw_tiles is None:
        return False

    live_grid = _build_live_grid(live_raw_tiles, location_name)
    map_payload["raw_tiles"] = live_raw_tiles
    map_payload["grid"] = live_grid
    _update_ascii_if_present(map_payload, live_grid)
    logger.info("Dynamic metatile overlay applied for %s (%sx%s)", location_name, width, height)
    return True


# ── Fortree City Gym: visual '?' marker for gate zones ────────────────

def _apply_fortree_gate_zone_markers(map_payload: Dict[str, Any], memory_reader: Any, location_name: str) -> bool:
    """Mark gate interaction-zone cells with '?' so the agent knows to verify walkability visually.

    Also stores ``fortree_gate_orientations`` in the payload so the pathfinder
    can enforce directional gate collision as an edge constraint.
    """
    grid = map_payload.get("grid")
    if not grid:
        logger.warning("Fortree gate-zone markers skipped: missing base grid")
        return False

    mutable_grid = _clone_grid_as_lists(grid)
    height = len(mutable_grid)
    width = len(mutable_grid[0]) if mutable_grid else 0
    if width <= 0 or height <= 0:
        return False

    if memory_reader is None:
        orientations = [int(g["initial_orientation"]) % 4 for g in FORTREE_GATE_CONFIG]
    else:
        orientations = _read_fortree_gate_orientations(memory_reader)

    map_payload["fortree_gate_orientations"] = orientations

    gate_cells = get_fortree_gate_zone_cells()
    marked = 0
    for x, y in gate_cells:
        if 0 <= y < height and 0 <= x < width:
            sym = mutable_grid[y][x]
            if sym not in PRESERVE_SYMBOLS and sym != "#":
                mutable_grid[y][x] = "?"
                marked += 1

    map_payload["grid"] = mutable_grid
    _update_ascii_if_present(map_payload, mutable_grid)
    logger.info(
        "Fortree gate-zone markers applied (%d cells marked '?', orientations=%s)",
        marked, orientations,
    )
    return True


# ── Registry ───────────────────────────────────────────────────────────

DYNAMIC_MAP_OVERLAYS: Dict[str, OverlayFn] = {
    "MAUVILLE CITY GYM": _apply_live_metatile_overlay_payload,
    "FORTREE CITY GYM": _apply_fortree_gate_zone_markers,
}


# ── Public API ─────────────────────────────────────────────────────────

def apply_live_overlay_to_json_map(json_map: Dict[str, Any], memory_reader: Any, location_name: str) -> bool:
    """Apply per-location dynamic overlay to a json_map payload."""
    if not is_dynamic_location(location_name):
        return False
    overlay_fn = DYNAMIC_MAP_OVERLAYS[location_name.upper()]
    return overlay_fn(json_map, memory_reader, location_name.upper())


def apply_live_metatile_overlay(state: Dict[str, Any], env: Any, location_name: str) -> bool:
    """Apply per-location dynamic overlay to state['map']['porymap'].

    Returns True if overlay was applied, otherwise False.
    """
    if not is_dynamic_location(location_name):
        return False

    porymap = state.get("map", {}).get("porymap", {})
    memory_reader = getattr(env, "memory_reader", None)
    overlay_fn = DYNAMIC_MAP_OVERLAYS[location_name.upper()]
    applied = overlay_fn(porymap, memory_reader, location_name.upper())
    if applied:
        state.setdefault("map", {})["porymap"] = porymap
    return applied
