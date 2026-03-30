# PokéAgent speedrun — system design (maintainers)

Short architecture notes for contributors. For install, run commands, and benchmarks, see the top-level [README.md](../README.md). For module detail, see `server/README.md`, `agents/README.md`, `pokemon_env/README.md`, and `utils/README.md`.

## High-level shape

- **Headless game server** (`server/app.py`): emulator, `/state`, actions, MCP HTTP tools. Agents and the web UI are clients.
- **Memory + Porymap**: `pokemon_env/memory_reader.py` reads WRAM; `pokemon_env/porymap/` holds decompilation map/tileset data. `utils/mapping/*` turns that into grids, ASCII, and A* pathfinding consumed by the server and agent prompts.

## Dynamic map overlay (strategy pattern)

### Problem

Different maps have different runtime-collision mechanisms. A single “read live metatiles” approach works for maps that mutate metatile collision bits (Mauville Gym), but fails for maps where collision is computed from puzzle state (Fortree rotating gates). The overlay system now dispatches per-location strategies.

### Allowlist

Only a small set of locations opt in. Uppercase ROM location names are keys in `DYNAMIC_MAP_OVERLAYS` in `utils/mapping/dynamic_map_overlay.py`.

- `MAUVILLE CITY GYM` -> live metatile overlay strategy
- `FORTREE CITY GYM` -> gate-zone `?` markers + orientations for pathfinding (see [implementation gaps — Fortree](architecture/implementation_gaps.md#4-fortree-city-gym-rotating-gates-vs-2d-map-representation))

### Data flow

1. `_format_porymap_info` builds a static `json_map` from Porymap.
2. `apply_live_overlay_to_json_map` dispatches by `location_name.upper()` using `DYNAMIC_MAP_OVERLAYS`.
3. Strategy mutates `grid` (and optionally `raw_tiles` / `ascii`) on `json_map` or `state["map"]["porymap"]`.
4. Updated grid then flows into pathfinding, state text formatting, and `/state` visual-map generation.

### Strategy classes

- **Metatile strategy (Mauville):**
  - Reads live map buffer with `read_map_metatiles(x_start=7, y_start=7, width, height)` (7-tile GBA border).
  - Rebuilds symbols via `format_tile_to_symbol`.
  - Replaces `grid`, `raw_tiles`, and `ascii` (when present).

- **Fortree (partial / prototype):**
  - Reads gate orientation bytes from `VAR_TEMP_0..VAR_TEMP_3` via `memory_reader.read_var(var_id)` (8 gates packed into 4 words).
  - Marks each gate’s interaction zone with `?` on the grid so the agent is nudged to confirm walkability from the game view.
  - Stores `fortree_gate_orientations` on the porymap payload; pathfinding applies an edge constraint in `_can_move_to` (`fortree_movement_blocked`). This is **not** a full tile-level model of arms (see [implementation gaps](architecture/implementation_gaps.md#4-fortree-city-gym-rotating-gates-vs-2d-map-representation)).

### Key APIs (`dynamic_map_overlay.py`)

- `is_dynamic_location(location_name)` — gate on strategy registry.
- `apply_live_overlay_to_json_map(json_map, memory_reader, location_name)` — used when formatting from Porymap’s `json_map`.
- `apply_live_metatile_overlay(state, env, location_name)` — used when navigation builds a fresh `state` slice for A*.
- `pokemon_env/memory_reader.py: read_var(var_id)` — reads normal script vars (`0x4000`-`0x40FF`) from `SaveBlock1 + 0x139C`.

### Integration points

| Area | Role |
|------|------|
| `server/game_tools.py` — `navigate_to_direct` | After static porymap is loaded into `state`, applies `apply_live_metatile_overlay` so the pathfinder grid matches RAM. |
| `server/game_tools.py` — `get_game_state_direct` | Sets `state["_memory_reader"]` before `format_state_for_llm`, then **pops** it before returning `raw_state` so serialized JSON stays clean. |
| `utils/mapping/porymap_state.py` — `_format_porymap_info` | Optional `memory_reader`: after static `json_map` is built, calls `apply_live_overlay_to_json_map` so ASCII map, `grid`, and `raw_tiles` (including elevation filtering) see live data. |
| `utils/state_formatter.py` — `_format_map_info` | Passes `memory_reader=state_data.get("_memory_reader")` into `_format_porymap_info`. |
| `server/app.py` — `/state` | Passes `env.memory_reader` into `_format_porymap_info` so `visual_map` / `porymap.grid` in the API match the same overlay path as the agent. |

### Extending

Add a new strategy function and register it in `DYNAMIC_MAP_OVERLAYS` under the map's uppercase ROM location. Choose strategy by puzzle mechanics:

- Metatile mutation puzzle -> metatile strategy
- Script-var/object-state puzzle -> custom strategy that projects puzzle state into walkability

## Related code map

- Pathfinding: `utils/mapping/pathfinding.py` (consumes `state["map"]["porymap"]["grid"]`).
- Porymap build + ROM name mapping: `utils/mapping/porymap_json_builder.py`, `utils/mapping/porymap_state.py` (`ROM_TO_PORYMAP_MAP`).
- Live metatile + vars read: `pokemon_env/memory_reader.py` (`read_map_metatiles`, `read_var`).
