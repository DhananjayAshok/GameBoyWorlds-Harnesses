import heapq
import re
from collections import deque
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple

import numpy as np

from execution.game_api import HARD_MAX_STEPS

from execution.perception.pokemon.grid import LEFT_COL, TOP_ROW

Cell = Tuple[int, int]

WALKABLE = "walkable"
BLOCKED = "blocked"
LEDGE = "ledge"
LEDGE_LEFT = "ledge_left"

WALKABLE_TAGS = {
    "ground", "ground decoration", "tall grass", "entrance", "staircase", "elevator",
    "special ground tile", "player",
}
WARP_TAGS = {"entrance", "staircase"}
LEDGE_TAGS = {"ledge"}
FURNITURE_OVERHANG_TOP_TAGS = {"interactable object", "obstacle"}
OVERHANG_TOP_TAGS = {"interactable object", "obstacle", "npc", "item", "sign"}
QUARTERS = ("top_left", "top_right", "bottom_left", "bottom_right")
FURNITURE_OVERHANG_DESC_PATTERN = re.compile(r"\b(?:table|desk|counter|shelf|bookshelf|machine|computer|pc|tv|television|sink|stove|bed)\b", re.IGNORECASE)
RIDGE_DESCRIPTION_PATTERN = re.compile(
    r"\b(?:ledge|cliff|ridge)\b|rocky\s+fence|horizontal\s+fence",
    re.IGNORECASE,
)
WEST_LEDGE_DESCRIPTION_PATTERN = re.compile(
    r"\bledge\b|rocky\s+fence",
    re.IGNORECASE,
)
WATER_BORDER_DESCRIPTION_PATTERN = re.compile(
    r"\b(?:water|water's|shoreline)\b",
    re.IGNORECASE,
)

X_RANGE = range(LEFT_COL, LEFT_COL + 10)
Y_RANGE = range(TOP_ROW - 8, TOP_ROW + 1)
DIRECTIONS: Dict[str, Cell] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
DIRECTION_ALIASES: Dict[str, str] = {
    "up": "up", "down": "down", "left": "left", "right": "right",
    "north": "up", "south": "down", "west": "left", "east": "right",
}
DIRECTION_PREFIX_PATTERN = re.compile(r"^\s*(up|down|left|right|north|south|west|east)\b", re.IGNORECASE)
MAX_STEPS_PER_MOVE = HARD_MAX_STEPS
COORDINATE_PATTERN = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def in_bounds(cell: Cell) -> bool:
    return cell[0] in X_RANGE and cell[1] in Y_RANGE


def is_top_overhang_floor_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
) -> bool:
    cell_q = quarter_tags.get(cell, {})
    if len(cell_q) < 4:
        return False
    bottom_tags = {cell_q["bottom_left"], cell_q["bottom_right"]}
    top_tags = {cell_q["top_left"], cell_q["top_right"]}
    if not (bottom_tags <= WALKABLE_TAGS) or (top_tags <= WALKABLE_TAGS) or bool(bottom_tags & (WARP_TAGS | {"elevator"})):
        return False
    if not (top_tags <= WALKABLE_TAGS | FURNITURE_OVERHANG_TOP_TAGS):
        return False
    if quarter_descriptions is not None:
        cell_d = quarter_descriptions.get(cell, {})
        for q in ("top_left", "top_right"):
            if cell_q.get(q) in FURNITURE_OVERHANG_TOP_TAGS:
                desc = cell_d.get(q) or ""
                if not FURNITURE_OVERHANG_DESC_PATTERN.search(desc):
                    return False
    above_q = quarter_tags.get((cell[0], cell[1] + 1), {})
    above_bottom = {above_q[q] for q in ("bottom_left", "bottom_right") if q in above_q}
    if above_bottom:
        return bool(above_bottom & (FURNITURE_OVERHANG_TOP_TAGS | {"item"}))
    return True


def is_ridge_quarter(tag: Optional[str], description: Optional[str] = None) -> bool:
    if tag in LEDGE_TAGS:
        return True
    if tag != "obstacle" or not description:
        return False
    if WATER_BORDER_DESCRIPTION_PATTERN.search(description):
        return False
    return bool(RIDGE_DESCRIPTION_PATTERN.search(description))


def is_west_ledge_quarter(tag: Optional[str], description: Optional[str] = None) -> bool:
    if tag in LEDGE_TAGS:
        return True
    if tag != "obstacle" or not description:
        return False
    if WATER_BORDER_DESCRIPTION_PATTERN.search(description):
        return False
    return bool(WEST_LEDGE_DESCRIPTION_PATTERN.search(description))


def is_south_ledge_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
) -> bool:
    cell_q = quarter_tags.get(cell, {})
    if len(cell_q) < 4:
        return False
    cell_d = (quarter_descriptions or {}).get(cell, {})
    top_tags = {cell_q["top_left"], cell_q["top_right"]}
    if not (top_tags <= WALKABLE_TAGS | LEDGE_TAGS):
        return False
    bot_left_ridge = is_ridge_quarter(cell_q["bottom_left"], cell_d.get("bottom_left"))
    bot_right_ridge = is_ridge_quarter(cell_q["bottom_right"], cell_d.get("bottom_right"))
    if bot_left_ridge and bot_right_ridge:
        return True
    if cell == (0, 1) and (bot_left_ridge or bot_right_ridge):
        bottom_tags = {cell_q["bottom_left"], cell_q["bottom_right"]}
        if bottom_tags <= WALKABLE_TAGS | LEDGE_TAGS | OVERHANG_TOP_TAGS:
            return True
    return False


def is_west_ledge_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
) -> bool:
    cell_q = quarter_tags.get(cell, {})
    if len(cell_q) < 4:
        return False
    cell_d = (quarter_descriptions or {}).get(cell, {})
    right_tags = {cell_q["top_right"], cell_q["bottom_right"]}
    if not (right_tags <= WALKABLE_TAGS):
        return False
    return (
        is_west_ledge_quarter(cell_q["top_left"], cell_d.get("top_left"))
        and is_west_ledge_quarter(cell_q["bottom_left"], cell_d.get("bottom_left"))
    )


STAIR_STEP_DESCRIPTION_PATTERN = re.compile(
    r"\b(?:step|steps|stair|stairs|ladder)\b",
    re.IGNORECASE,
)
ELEVATION_STEP_MEAN_DIFF = 60.0


def _uniform_ground_quarter_image(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
    allow_player_cap: bool = False,
) -> Optional[np.ndarray]:
    if quarter_images is None:
        return None
    cell_q = quarter_tags.get(cell, {})
    if len(cell_q) < 4 or set(cell_q.values()) != {"ground"}:
        return None
    cell_d = (quarter_descriptions or {}).get(cell, {})
    for desc in cell_d.values():
        if desc and STAIR_STEP_DESCRIPTION_PATTERN.search(desc):
            return None
    cell_imgs = quarter_images.get(cell, {})
    if len(cell_imgs) < 4 or any(cell_imgs.get(q) is None for q in QUARTERS):
        return None
    tl = np.squeeze(np.asarray(cell_imgs["top_left"]))
    tr = np.squeeze(np.asarray(cell_imgs["top_right"]))
    bl = np.squeeze(np.asarray(cell_imgs["bottom_left"]))
    br = np.squeeze(np.asarray(cell_imgs["bottom_right"]))
    if tl.size == 0 or int(tl.min()) == int(tl.max()) or not np.array_equal(tl, tr):
        return None
    if np.array_equal(tl, bl) and np.array_equal(tl, br):
        return tl
    if allow_player_cap and cell == (0, 1):
        if np.array_equal(tl[:4, :], bl[:4, :]) and np.array_equal(tl[:4, :], br[:4, :]):
            return tl
    return None


def _is_east_elevation_step_barrier_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
    allow_player_cap: bool = False,
) -> bool:
    img_here = _uniform_ground_quarter_image(
        cell, quarter_tags, quarter_descriptions, quarter_images, allow_player_cap=allow_player_cap
    )
    if img_here is None:
        return False
    shades_here = {int(v) for v in np.unique(img_here)}
    if 0 in shades_here:
        return False
    west = (cell[0] - 1, cell[1])
    west_q = quarter_tags.get(west, {})
    if allow_player_cap and west == (0, 0) and "player" in set(west_q.values()):
        west_candidates = [(-1, 0), (0, 1), (0, -1)]
    else:
        west_candidates = [west]
    for wc in west_candidates:
        img_west = _uniform_ground_quarter_image(
            wc, quarter_tags, quarter_descriptions, quarter_images, allow_player_cap=allow_player_cap
        )
        if img_west is None:
            continue
        shades_west = {int(v) for v in np.unique(img_west)}
        if 0 in shades_west and float(np.mean(img_here)) - float(np.mean(img_west)) >= ELEVATION_STEP_MEAN_DIFF:
            return True
    return False


def is_east_cliff_barrier_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
    allow_player_cap: bool = False,
) -> bool:
    cell_q = quarter_tags.get(cell, {})
    if len(cell_q) < 4:
        return False
    cell_d = (quarter_descriptions or {}).get(cell, {})
    left_tags = {cell_q["top_left"], cell_q["bottom_left"]}
    if not (left_tags <= WALKABLE_TAGS):
        return False
    if (
        is_ridge_quarter(cell_q["top_right"], cell_d.get("top_right"))
        and is_ridge_quarter(cell_q["bottom_right"], cell_d.get("bottom_right"))
    ):
        return True
    return _is_east_elevation_step_barrier_cell(
        cell, quarter_tags, quarter_descriptions, quarter_images, allow_player_cap=allow_player_cap
    )


def is_player_overhang_floor_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
) -> bool:
    if cell != (0, 1):
        return False
    cell_q = quarter_tags.get(cell, {})
    if len(cell_q) < 4:
        return False
    if (
        is_south_ledge_cell(cell, quarter_tags, quarter_descriptions)
        or is_west_ledge_cell(cell, quarter_tags, quarter_descriptions)
        or is_east_cliff_barrier_cell(cell, quarter_tags, quarter_descriptions)
    ):
        return False
    top_tags = {cell_q["top_left"], cell_q["top_right"]}
    bottom_tags = {cell_q["bottom_left"], cell_q["bottom_right"]}
    if not (top_tags <= WALKABLE_TAGS) or (bottom_tags <= WALKABLE_TAGS):
        return False
    if not (bottom_tags <= WALKABLE_TAGS | OVERHANG_TOP_TAGS):
        return False
    cell_d = (quarter_descriptions or {}).get(cell, {})
    for q in ("bottom_left", "bottom_right"):
        if is_ridge_quarter(cell_q.get(q), cell_d.get(q)):
            return False
    if cell_q["bottom_left"] == "obstacle" and cell_q["bottom_right"] == "obstacle":
        return False
    return True


def is_player_overhang_ledge_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
) -> bool:
    if cell != (0, 1):
        return False
    return is_south_ledge_cell(cell, quarter_tags, quarter_descriptions)


def is_npc_head_overhang_floor_cell(cell: Cell, quarter_tags: Dict[Cell, Dict[str, str]]) -> bool:
    if cell == (0, 1):
        return False
    cell_q = quarter_tags.get(cell, {})
    if len(cell_q) < 4:
        return False
    top_tags = {cell_q["top_left"], cell_q["top_right"]}
    bottom_tags = {cell_q["bottom_left"], cell_q["bottom_right"]}
    if not (top_tags <= WALKABLE_TAGS) or (bottom_tags <= WALKABLE_TAGS):
        return False
    if not (bottom_tags <= WALKABLE_TAGS | {"npc"}) or "npc" not in bottom_tags:
        return False
    below_q = quarter_tags.get((cell[0], cell[1] - 1), {})
    below_top = {below_q[q] for q in ("top_left", "top_right") if q in below_q}
    return "npc" in below_top


def is_overhang_floor_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
) -> bool:
    return (
        is_top_overhang_floor_cell(cell, quarter_tags, quarter_descriptions)
        or is_player_overhang_floor_cell(cell, quarter_tags, quarter_descriptions)
        or is_npc_head_overhang_floor_cell(cell, quarter_tags)
    )


def cell_kinds(
    identified,
    ledge_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None,
    walk_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None,
    people: Optional[Dict[Cell, str]] = None,
    left_ledge_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None,
    blocked_cells: Optional[Collection[Cell]] = None,
) -> Dict[Cell, str]:
    from execution.perception.pokemon.tiles import filter_cell_warp_quarter_tags, normalize_tabletop_tag

    # ``ledge_patterns`` / ``walk_patterns``: quarter hashes (QUARTERS order) learned from movement.
    tags: Dict[Cell, set] = {}
    quarter_tags: Dict[Cell, Dict[str, str]] = {}
    quarter_descriptions: Dict[Cell, Dict[str, Optional[str]]] = {}
    quarter_images: Dict[Cell, Dict[str, Optional[Any]]] = {}
    hashes: Dict[Cell, Dict[str, Optional[str]]] = {}
    for (x, y, quarter), tile in (identified or {}).items():
        desc = getattr(tile, "description", None)
        img = getattr(tile, "image", None)
        norm_tag = normalize_tabletop_tag(tile.tag, desc, img)
        quarter_tags.setdefault((x, y), {})[quarter] = norm_tag
        quarter_descriptions.setdefault((x, y), {})[quarter] = desc
        quarter_images.setdefault((x, y), {})[quarter] = img
        hashes.setdefault((x, y), {})[quarter] = tile.hash
    for cell in list(quarter_tags):
        quarter_tags[cell] = filter_cell_warp_quarter_tags(quarter_tags[cell])
        tags[cell] = set(quarter_tags[cell].values())
    confirmed_occupied: set = set()
    for cell, label in (people or {}).items():
        if cell == (0, 0):
            continue
        lower = (label or "").lower()
        if "remembered here" in lower or "seen only in part" in lower:
            continue
        confirmed_occupied.add(cell)
    empirical_blocked = set(blocked_cells or ())
    kinds = {}
    for cell, cell_tags in tags.items():
        pattern = tuple(hashes[cell].get(q) for q in QUARTERS)
        if cell in confirmed_occupied:
            kinds[cell] = BLOCKED
        elif cell != (0, 0) and cell in empirical_blocked and not (cell_tags & WARP_TAGS):
            kinds[cell] = BLOCKED
        elif is_east_cliff_barrier_cell(
            cell, quarter_tags, quarter_descriptions, quarter_images, allow_player_cap=True
        ):
            kinds[cell] = BLOCKED
        elif (
            cell_tags & WARP_TAGS
            or cell_tags <= WALKABLE_TAGS
            or is_overhang_floor_cell(cell, quarter_tags, quarter_descriptions)
        ):
            kinds[cell] = WALKABLE
        elif is_south_ledge_cell(cell, quarter_tags, quarter_descriptions) or (
            cell_tags <= WALKABLE_TAGS | LEDGE_TAGS
        ):
            kinds[cell] = LEDGE
        elif is_west_ledge_cell(cell, quarter_tags, quarter_descriptions):
            kinds[cell] = LEDGE_LEFT
        elif ledge_patterns and pattern in ledge_patterns:
            kinds[cell] = LEDGE
        elif left_ledge_patterns and pattern in left_ledge_patterns:
            kinds[cell] = LEDGE_LEFT
        elif (
            walk_patterns
            and pattern in walk_patterns
            and "npc" not in cell_tags
            and "unknown" not in cell_tags
            and "water" not in cell_tags
            and (
                bool(cell_tags & WALKABLE_TAGS)
                or (len(pattern) == 4 and len(set(pattern)) == 1 and pattern[0] is not None)
            )
        ):
            kinds[cell] = WALKABLE
        else:
            kinds[cell] = BLOCKED
    for cell, cell_q in quarter_tags.items():
        if (
            len(cell_q) == 4
            and cell_q.get("top_left") == "entrance"
            and cell_q.get("top_right") == "entrance"
            and cell_q.get("bottom_left") != "entrance"
            and cell_q.get("bottom_right") != "entrance"
        ):
            ext_hashes = {
                hashes.get(cell, {}).get("bottom_left"),
                hashes.get(cell, {}).get("bottom_right"),
            } - {None}
            if len(ext_hashes) == 1:
                ext_h = next(iter(ext_hashes))
                for nb in ((cell[0], cell[1] - 1), (cell[0] - 1, cell[1]), (cell[0] + 1, cell[1])):
                    if nb != (0, 0) and tuple(hashes.get(nb, {}).get(q) for q in QUARTERS) == (ext_h,) * 4:
                        kinds[nb] = BLOCKED
    return kinds


def neighbours(cell: Cell, direction_order: Iterable[str] = DIRECTIONS) -> List[Tuple[str, Cell]]:
    return [(d, (cell[0] + DIRECTIONS[d][0], cell[1] + DIRECTIONS[d][1])) for d in direction_order]


def _ordered_directions_for_goals(start: Cell, goals: set, prefer_lateral: bool = False) -> Tuple[str, ...]:
    if not prefer_lateral or len(goals) != 1:
        return tuple(DIRECTIONS)
    target = next(iter(goals))
    dx = target[0] - start[0]
    dy = target[1] - start[1]
    if dx == 0 or dy == 0:
        return tuple(DIRECTIONS)
    primary = []
    if abs(ty := target[1]) == max(abs(min(Y_RANGE)), abs(max(Y_RANGE))) and abs(tx := target[0]) < max(abs(min(X_RANGE)), abs(max(X_RANGE))):
        primary.append("right" if dx > 0 else "left")
        primary.append("up" if dy > 0 else "down")
    elif abs(target[0]) in (abs(min(X_RANGE)), abs(max(X_RANGE))) and abs(target[1]) < max(abs(min(Y_RANGE)), abs(max(Y_RANGE))):
        primary.append("up" if dy > 0 else "down")
        primary.append("right" if dx > 0 else "left")
    else:
        primary.append("right" if dx > 0 else "left")
        primary.append("up" if dy > 0 else "down")
    return tuple(primary + [d for d in DIRECTIONS if d not in primary])


TRAINER_LOS_MAX_RANGE = 4
FACING_LABEL_PATTERN = re.compile(r"\bfacing\s+(up|down|left|right)\b", re.IGNORECASE)


def parse_trainer_los_from_label(label: Optional[str]) -> Tuple[Optional[str], bool]:
    if not label or not isinstance(label, str) or label.startswith("sign:"):
        return None, False
    lower = label.lower()
    if "remembered here" in lower:
        return None, False
    match = FACING_LABEL_PATTERN.search(lower)
    facing = match.group(1).lower() if match else None
    talked = "talked to here" in lower and "not talked to here" not in lower
    return facing, (facing is not None and not talked)


def trainer_los_cells(
    kinds: Optional[Dict[Cell, str]],
    people: Optional[Dict[Cell, str]] = None,
    max_range: int = TRAINER_LOS_MAX_RANGE,
) -> Dict[Cell, List[Tuple[Cell, str]]]:
    los: Dict[Cell, List[Tuple[Cell, str]]] = {}
    if not kinds or not people:
        return los
    for npc_cell, label in people.items():
        if npc_cell == (0, 0) or not in_bounds(npc_cell):
            continue
        facing, active = parse_trainer_los_from_label(label)
        if not active or facing not in DIRECTIONS:
            continue
        dx, dy = DIRECTIONS[facing]
        for step in range(1, max_range + 1):
            cell = (npc_cell[0] + dx * step, npc_cell[1] + dy * step)
            if not in_bounds(cell) or kinds.get(cell, BLOCKED) != WALKABLE:
                break
            los.setdefault(cell, []).append((npc_cell, facing))
    return los


def trainer_los_ray(
    kinds: Optional[Dict[Cell, str]],
    npc_cell: Cell,
    label: Optional[str],
    max_range: int = TRAINER_LOS_MAX_RANGE,
) -> Tuple[Optional[str], List[Cell]]:
    if not kinds or npc_cell == (0, 0) or not in_bounds(npc_cell):
        return None, []
    facing, active = parse_trainer_los_from_label(label)
    if not active or facing not in DIRECTIONS:
        return facing, []
    dx, dy = DIRECTIONS[facing]
    ray: List[Cell] = []
    for step in range(1, max_range + 1):
        cell = (npc_cell[0] + dx * step, npc_cell[1] + dy * step)
        if not in_bounds(cell) or kinds.get(cell, BLOCKED) != WALKABLE:
            break
        ray.append(cell)
    return facing, ray


def is_ledge_roofed_edge_cell(
    cell: Cell,
    direction: str,
    kinds: Dict[Cell, str],
) -> bool:
    if direction in ("left", "right"):
        x, y0 = cell
        if y0 >= 0:
            return False
        for y in range(y0 + 1, 1):
            k = kinds.get((x, y), BLOCKED)
            if k == LEDGE:
                return True
            if k != WALKABLE:
                return False
        return False
    if direction == "down":
        x0, y = cell
        if x0 >= 0:
            return False
        for x in range(x0 + 1, 1):
            k = kinds.get((x, y), BLOCKED)
            if k == LEDGE_LEFT:
                return True
            if k != WALKABLE:
                return False
        return False
    return False


def shortest_path(
    kinds: Dict[Cell, str],
    start: Cell,
    goals: Iterable[Cell],
    prefer_lateral: bool = False,
    avoid_cells: Optional[Collection[Cell]] = None,
    people: Optional[Dict[Cell, str]] = None,
) -> Optional[List[Tuple[str, str, Cell]]]:
    goal_set = set(goals)
    if start in goal_set:
        return []
    if not goal_set:
        return None
    direction_order = _ordered_directions_for_goals(start, goal_set, prefer_lateral=prefer_lateral)
    dir_rank = {d: i for i, d in enumerate(direction_order)}
    avoid_set = (set(avoid_cells) - goal_set - {start}) if avoid_cells else set()
    los_set = set(trainer_los_cells(kinds, people)) if people else set()
    best: Dict[Cell, Tuple[int, int, int, int]] = {start: (0, 0, 0, 0)}
    previous: Dict[Cell, Optional[Tuple[Cell, str, str]]] = {start: None}
    counter = 0
    heap: List[Tuple[int, int, int, int, int, int, Cell]] = [(0, 0, 0, 0, 0, 0, start)]
    while heap:
        jumps, avoids, los_hits, steps, first_rank, _, cell = heapq.heappop(heap)
        if (jumps, avoids, los_hits, steps) > best.get(cell, (jumps + 1, 0, 0, 0)):
            continue
        if cell in goal_set:
            path = []
            cur = cell
            while previous[cur] is not None:
                parent, d, m = previous[cur]
                path.append((d, m, cur))
                cur = parent
            return path[::-1]
        for direction, nxt in neighbours(cell, direction_order):
            if not in_bounds(nxt):
                continue
            kind = kinds.get(nxt, BLOCKED)
            if kind == WALKABLE:
                landing, move = nxt, "walk"
                n_jumps = jumps
            elif (kind == LEDGE and direction == "down") or (kind == LEDGE_LEFT and direction == "left"):
                dx, dy = DIRECTIONS[direction]
                landing, move = (nxt[0] + dx, nxt[1] + dy), "jump"
                if not in_bounds(landing) or kinds.get(landing, BLOCKED) != WALKABLE:
                    continue
                n_jumps = jumps + 1
            else:
                continue
            step_avoid = int((nxt in avoid_set and nxt not in goal_set) or (landing in avoid_set and landing not in goal_set))
            n_avoids = avoids + step_avoid
            n_los = los_hits + (1 if landing in los_set else 0)
            n_steps = steps + 1
            cost = (n_jumps, n_avoids, n_los, n_steps)
            if cost < best.get(landing, (n_jumps + 1, 0, 0, 0)):
                best[landing] = cost
                previous[landing] = (cell, direction, move)
                counter += 1
                rank = dir_rank[direction] if cell == start else first_rank
                heapq.heappush(heap, (n_jumps, n_avoids, n_los, n_steps, rank, counter, landing))
    return None


def compress_path(path: List[Tuple[str, str, Cell]]) -> List[Tuple[str, int, str]]:
    runs: List[Tuple[str, int, str]] = []
    for direction, move, _ in path:
        if move == "walk" and runs and runs[-1][0] == direction and runs[-1][2] == "walk" and runs[-1][1] < MAX_STEPS_PER_MOVE:
            runs[-1] = (direction, runs[-1][1] + 1, "walk")
        else:
            runs.append((direction, 1, move))
    return runs


COUNTER_SEPARATOR_DESC_PATTERN = re.compile(
    r"\b(?:counter|desk|table|window|booth)\b",
    re.IGNORECASE,
)


def is_counter_separator_cell(
    cell: Cell,
    kinds: Dict[Cell, str],
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
) -> bool:
    if not in_bounds(cell) or kinds.get(cell) != BLOCKED:
        return False
    if identified is None:
        return True
    from execution.perception.pokemon.tiles import normalize_tabletop_tag

    has_counter_desc = False
    for q in QUARTERS:
        tile = identified.get((cell[0], cell[1], q))
        if tile is None:
            continue
        desc = getattr(tile, "description", None) or ""
        tag = normalize_tabletop_tag(tile.tag, desc, getattr(tile, "image", None))
        if tag in ("npc", "sign"):
            return False
        if COUNTER_SEPARATOR_DESC_PATTERN.search(desc):
            has_counter_desc = True
    return has_counter_desc


def _is_npc_target_cell(
    target: Cell,
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
    people: Optional[Dict[Cell, str]] = None,
) -> bool:
    if people and target in people:
        label = str(people[target] or "")
        if not label.startswith("sign:"):
            return True
    if identified is not None:
        for q in QUARTERS:
            tile = identified.get((target[0], target[1], q))
            if tile is not None and getattr(tile, "tag", None) == "npc":
                return True
    return identified is None and people is None


def approach_cells(
    kinds: Dict[Cell, str],
    target: Cell,
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
    people: Optional[Dict[Cell, str]] = None,
) -> Dict[Cell, str]:
    cells = {}
    for direction, cell in neighbours(target):
        if in_bounds(cell) and kinds.get(cell) == WALKABLE:
            facing = next(d for d, (dx, dy) in DIRECTIONS.items() if (cell[0] + dx, cell[1] + dy) == target)
            cells[cell] = facing
    if _is_npc_target_cell(target, identified, people):
        for direction, (dx, dy) in DIRECTIONS.items():
            mid = (target[0] + dx, target[1] + dy)
            far = (target[0] + 2 * dx, target[1] + 2 * dy)
            if not (in_bounds(mid) and in_bounds(far)):
                continue
            if kinds.get(far) != WALKABLE:
                continue
            if not is_counter_separator_cell(mid, kinds, identified):
                continue
            facing = next(d for d, (ox, oy) in DIRECTIONS.items() if (ox, oy) == (-dx, -dy))
            cells[far] = facing
    return cells


def edge_cells(kinds: Dict[Cell, str], direction: str) -> List[Cell]:
    if direction == "up":
        cells = [(x, max(Y_RANGE)) for x in X_RANGE]
    elif direction == "down":
        cells = [(x, min(Y_RANGE)) for x in X_RANGE]
    elif direction == "left":
        cells = [(min(X_RANGE), y) for y in reversed(Y_RANGE)]
    else:
        cells = [(max(X_RANGE), y) for y in reversed(Y_RANGE)]
    return [cell for cell in cells
            if kinds.get(cell) == WALKABLE and shortest_path(kinds, (0, 0), [cell]) is not None]


def is_on_edge(cell: Cell, direction: str) -> bool:
    return {
        "up": cell[1] == max(Y_RANGE),
        "down": cell[1] == min(Y_RANGE),
        "left": cell[0] == min(X_RANGE),
        "right": cell[0] == max(X_RANGE),
    }[direction]


AVOID_MARKER_PATTERN = re.compile(
    r"\b(?:"
    r"avoid(?:ing|s)?"
    r"|do\s+not|don't|not|never"
    r"|without"
    r"|except|excluding"
    r"|instead\s+of|rather\s+than|other\s+than"
    r"|away\s+from|clear\s+of"
    r"|stay(?:ing)?\s+(?:off|away\s+from|clear\s+of)"
    r"|keep(?:ing)?\s+(?:off|away\s+from|clear\s+of)"
    r"|bypass(?:ing)?"
    r"|skip(?:ping)?"
    r"|ignor(?:e|ing)"
    r")\b",
    re.IGNORECASE,
)

POSITIVE_ROUTE_BREAK_PATTERN = re.compile(
    r"\b(?:"
    r"via|through|along|toward|towards"
    r"|by\s+(?:walking|moving|going|stepping|heading|continuing|taking|using)"
    r"|to\s+(?:exit|reach|enter|take|walk|move|go|step|head|continue|leave|pass|cross|get)"
    r")\b",
    re.IGNORECASE,
)

COMPLETED_AVOID_OBJECT_THEN_ROUTE_PATTERN = re.compile(
    r"\b(?:the|a|an|this|that|any)\s+[a-z][\w\s-]*?\b(?:"
    r"via|through|along|toward|towards"
    r"|by\s+(?:walking|moving|going|stepping|heading|continuing|taking|using)"
    r"|to\s+(?:exit|reach|enter|take|walk|move|go|step|head|continue|leave|pass|cross|get)"
    r")\b",
    re.IGNORECASE,
)

COORD_NP_CONTINUATION_PATTERN = re.compile(
    r"^\s*(?:,\s*)?(?:(?:and|or|nor)\s+)?(?:,\s*)?(?:(?:the|a|an|any|another)\s+[\w\s-]{1,30}?\s+)?(?:(?:at|on|in)\s+)?$",
    re.IGNORECASE,
)

CLAUSE_CONJUNCTION_SPLIT_PATTERN = re.compile(
    r"\b(?:and|but|then|while)\b",
    re.IGNORECASE,
)


def classify_task_cells(text: Optional[str]) -> Tuple[List[Cell], List[Cell]]:
    if not text:
        return [], []
    positive: List[Cell] = []
    avoided: List[Cell] = []
    prev_end = 0
    prev_avoided = False
    for match in COORDINATE_PATTERN.finditer(text):
        cell = (int(match.group(1)), int(match.group(2)))
        prefix = text[prev_end:match.start()]
        prev_end = match.end()

        if (
            prev_avoided
            and not AVOID_MARKER_PATTERN.search(prefix)
            and not POSITIVE_ROUTE_BREAK_PATTERN.search(prefix)
            and COORD_NP_CONTINUATION_PATTERN.match(prefix)
        ):
            is_avoided = True
        else:
            clause = re.split(r"[,;.:()\[\]]", prefix)[-1]
            conj_parts = CLAUSE_CONJUNCTION_SPLIT_PATTERN.split(clause)
            if len(conj_parts) > 1:
                last_part = conj_parts[-1]
                if AVOID_MARKER_PATTERN.search(last_part):
                    after_avoid = AVOID_MARKER_PATTERN.split(last_part)[-1]
                    is_avoided = not bool(COMPLETED_AVOID_OBJECT_THEN_ROUTE_PATTERN.search(after_avoid))
                elif (
                    AVOID_MARKER_PATTERN.search(conj_parts[-2])
                    and not POSITIVE_ROUTE_BREAK_PATTERN.search(last_part)
                    and COORD_NP_CONTINUATION_PATTERN.match(last_part)
                ):
                    is_avoided = True
                else:
                    is_avoided = False
            else:
                if AVOID_MARKER_PATTERN.search(clause):
                    after_avoid = AVOID_MARKER_PATTERN.split(clause)[-1]
                    is_avoided = not bool(COMPLETED_AVOID_OBJECT_THEN_ROUTE_PATTERN.search(after_avoid))
                else:
                    is_avoided = False

        prev_avoided = is_avoided
        if is_avoided:
            avoided.append(cell)
        else:
            positive.append(cell)
    return positive, avoided


def parse_cell(text: Optional[str]) -> Optional[Cell]:
    if not text:
        return None
    match = COORDINATE_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_all_cells(text: Optional[str], include_avoided: bool = False) -> List[Cell]:
    if not text:
        return []
    if include_avoided:
        return [(int(m.group(1)), int(m.group(2))) for m in COORDINATE_PATTERN.finditer(text)]
    positive, _ = classify_task_cells(text)
    return positive


def parse_avoided_cells(text: Optional[str]) -> List[Cell]:
    if not text:
        return []
    _, avoided = classify_task_cells(text)
    return avoided


def parse_edge_direction(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    match = DIRECTION_PREFIX_PATTERN.match(text)
    if match is None:
        return None
    return DIRECTION_ALIASES.get(match.group(1).lower())


def edge_runs(cells: Iterable[Cell], direction: str) -> List[List[Cell]]:
    axis = 0 if direction in ("up", "down") else 1
    ordered = sorted(cells, key=lambda c: c[axis])
    runs: List[List[Cell]] = []
    for cell in ordered:
        if runs and cell[axis] - runs[-1][-1][axis] == 1:
            runs[-1].append(cell)
        else:
            runs.append([cell])
    return runs


def filter_forward_edge_cells(cells: Iterable[Cell], direction: str) -> List[Cell]:
    cells_list = list(cells)
    dx, dy = DIRECTIONS[direction]
    forward = [c for c in cells_list if (c[0] * dx + c[1] * dy) >= 0]
    return forward if forward else cells_list


LOOKAHEAD_STEPS = 4


def _offscreen_continuation_depth(
    seg: Dict[Cell, Any],
    start: Cell,
    direction: str,
    max_depth: int = LOOKAHEAD_STEPS,
) -> int:
    dx, dy = DIRECTIONS[direction]
    gx_min, gx_max = start[0] - max(X_RANGE) + min(X_RANGE), start[0]
    if direction == "up":
        bound = start[1]
        in_forward = lambda g: g[1] >= bound
        depth_of = lambda g: g[1] - bound
    elif direction == "down":
        bound = start[1]
        in_forward = lambda g: g[1] <= bound
        depth_of = lambda g: bound - g[1]
    elif direction == "left":
        bound = start[0]
        in_forward = lambda g: g[0] <= bound
        depth_of = lambda g: bound - g[0]
    else:
        bound = start[0]
        in_forward = lambda g: g[0] >= bound
        depth_of = lambda g: g[0] - bound

    first = (start[0] + dx, start[1] + dy)
    fc = seg.get(first)
    if fc is None or getattr(fc, "kind", None) == "unknown":
        return max_depth
    if (getattr(fc, "kind", None) == LEDGE and direction == "down") or (
        getattr(fc, "kind", None) == LEDGE_LEFT and direction == "left"
    ):
        first = (start[0] + 2 * dx, start[1] + 2 * dy)
        fc = seg.get(first)
        if fc is None or getattr(fc, "kind", None) == "unknown":
            return max_depth
        if getattr(fc, "kind", None) != WALKABLE:
            return 0
    elif getattr(fc, "kind", None) != WALKABLE:
        return 0

    best_depth = depth_of(first)
    visited = {start, first}
    queue = deque([(first, 1)])
    while queue:
        cur, steps = queue.popleft()
        d_cur = depth_of(cur)
        if d_cur > best_depth:
            best_depth = d_cur
        if best_depth >= max_depth or steps >= max_depth + 2:
            return max_depth
        for d_name, (nx_d, ny_d) in DIRECTIONS.items():
            nxt = (cur[0] + nx_d, cur[1] + ny_d)
            if not in_forward(nxt):
                continue
            nc = seg.get(nxt)
            if nc is None or getattr(nc, "kind", None) == "unknown":
                return max_depth
            nkind = getattr(nc, "kind", None)
            if "entrance" in getattr(nc, "tags", ()) or "staircase" in getattr(nc, "tags", ()):
                return max_depth
            if nkind == WALKABLE:
                landing = nxt
            elif (nkind == LEDGE and d_name == "down") or (nkind == LEDGE_LEFT and d_name == "left"):
                landing = (nxt[0] + nx_d, nxt[1] + ny_d)
                if not in_forward(landing):
                    continue
                lc = seg.get(landing)
                if lc is None or getattr(lc, "kind", None) == "unknown":
                    return max_depth
                if getattr(lc, "kind", None) != WALKABLE:
                    continue
            else:
                continue
            if landing not in visited:
                visited.add(landing)
                queue.append((landing, steps + 1))
    return min(best_depth, max_depth)


def _is_warp_edge_cell(
    cell: Cell,
    seg: Optional[Dict[Cell, Any]] = None,
    px: int = 0,
    py: int = 0,
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
) -> bool:
    if seg is not None:
        sc = seg.get((px + cell[0], py + cell[1]))
        if sc is not None and any(t in WARP_TAGS for t in getattr(sc, "tags", ())):
            return True
    if identified is not None:
        from execution.perception.pokemon.tiles import filter_cell_warp_quarter_tags, normalize_tabletop_tag

        cell_q = {
            q: normalize_tabletop_tag(
                identified[(cell[0], cell[1], q)].tag,
                getattr(identified[(cell[0], cell[1], q)], "description", None),
                getattr(identified[(cell[0], cell[1], q)], "image", None),
            )
            for q in QUARTERS
            if (cell[0], cell[1], q) in identified and hasattr(identified[(cell[0], cell[1], q)], "tag")
        }
        if set(filter_cell_warp_quarter_tags(cell_q).values()) & WARP_TAGS:
            return True
    return False


def warp_cells(
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
    global_map: Optional[Any] = None,
) -> set:
    warps = set()
    seg = None
    px = py = 0
    if global_map is not None and getattr(global_map, "current", None) and getattr(global_map, "pos", None) is not None:
        try:
            _, (px, py) = global_map.resolve(global_map.segment, global_map.pos)
            seg = global_map.current
        except Exception:  # noqa: BLE001
            seg = None
    if seg is None and identified is None:
        return warps
    for x in X_RANGE:
        for y in Y_RANGE:
            c = (x, y)
            if _is_warp_edge_cell(c, seg=seg, px=px, py=py, identified=identified):
                warps.add(c)
    return warps


def filter_open_lane_edge_cells(
    cells: Iterable[Cell],
    direction: str,
    kinds: Dict[Cell, str],
    global_map: Optional[Any] = None,
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
    people: Optional[Dict[Cell, str]] = None,
) -> List[Cell]:
    cells_list = list(cells)
    if not cells_list:
        return []
    dx, dy = DIRECTIONS[direction]
    seg = None
    px = py = 0
    if global_map is not None and getattr(global_map, "current", None) and getattr(global_map, "pos", None) is not None:
        try:
            _, (px, py) = global_map.resolve(global_map.segment, global_map.pos)
            seg = global_map.current
        except Exception:  # noqa: BLE001
            seg = None

    non_warp = [
        c for c in cells_list
        if not _is_warp_edge_cell(c, seg=seg, px=px, py=py, identified=identified)
        and (seg is None or _offscreen_continuation_depth(seg, (px + c[0], py + c[1]), direction, 1) > 0)
    ]
    if non_warp:
        cells_list = non_warp

    if seg is not None:
        depths = {
            c: _offscreen_continuation_depth(seg, (px + c[0], py + c[1]), direction, LOOKAHEAD_STEPS)
            for c in cells_list
        }
        deep_clear = [c for c in cells_list if depths[c] >= LOOKAHEAD_STEPS]
        if deep_clear:
            cells_list = deep_clear
        else:
            max_d = max(depths.values(), default=0)
            if max_d > 0:
                cells_list = [c for c in cells_list if depths[c] == max_d]

    inward_clear = [
        c for c in cells_list
        if kinds.get((c[0] - dx, c[1] - dy)) == WALKABLE
        or (kinds.get((c[0] - dx, c[1] - dy)) == LEDGE and direction == "down")
        or (kinds.get((c[0] - dx, c[1] - dy)) == LEDGE_LEFT and direction == "left")
    ]
    if inward_clear:
        cells_list = inward_clear

    hazard_cells = set(trainer_los_cells(kinds, people)) if people else set()
    non_jump_cells: List[Cell] = []
    unroofed_non_jump_cells: List[Cell] = []
    los_free_non_jump_cells: List[Cell] = []
    los_free_inward_non_jump_cells: List[Cell] = []
    los_free_unroofed_non_jump_cells: List[Cell] = []
    los_free_inward_unroofed_non_jump_cells: List[Cell] = []
    los_free_any_cells: List[Cell] = []
    for c in cells_list:
        path = shortest_path(kinds, (0, 0), [c], people=people)
        if path is None:
            continue
        has_jump = any(m == "jump" for _, m, _ in path)
        has_los = bool(hazard_cells) and any(step_cell in hazard_cells for _, _, step_cell in path)
        inward_los = bool(hazard_cells) and ((c[0] - dx, c[1] - dy) in hazard_cells)
        roofed = is_ledge_roofed_edge_cell(c, direction, kinds)
        if not has_jump:
            non_jump_cells.append(c)
            if not roofed:
                unroofed_non_jump_cells.append(c)
            if not has_los:
                los_free_non_jump_cells.append(c)
                if not roofed:
                    los_free_unroofed_non_jump_cells.append(c)
                if not inward_los:
                    los_free_inward_non_jump_cells.append(c)
                    if not roofed:
                        los_free_inward_unroofed_non_jump_cells.append(c)
        if not has_los:
            los_free_any_cells.append(c)

    if los_free_inward_unroofed_non_jump_cells:
        return los_free_inward_unroofed_non_jump_cells
    if los_free_unroofed_non_jump_cells:
        return los_free_unroofed_non_jump_cells
    if unroofed_non_jump_cells:
        return unroofed_non_jump_cells
    if los_free_inward_non_jump_cells:
        return los_free_inward_non_jump_cells
    if los_free_non_jump_cells:
        return los_free_non_jump_cells
    if non_jump_cells:
        return non_jump_cells
    if los_free_any_cells:
        return los_free_any_cells
    return cells_list


def is_boundary_exit_direction(kinds: Dict[Cell, str], cell: Cell, direction: str) -> bool:
    dx, dy = DIRECTIONS[direction]
    cur = (cell[0] + dx, cell[1] + dy)
    while in_bounds(cur):
        if kinds.get(cur, BLOCKED) != BLOCKED:
            return False
        cur = (cur[0] + dx, cur[1] + dy)
    return True


def boundary_exit_direction(kinds: Dict[Cell, str], identified: Optional[Dict[Tuple[int, int, str], object]],
                            cell: Cell, preferred: Optional[str] = None) -> Optional[str]:
    from execution.perception.pokemon.tiles import filter_cell_warp_quarter_tags, normalize_tabletop_tag

    if identified is None:
        return None
    cell_q = {
        q: normalize_tabletop_tag(
            identified[(cell[0], cell[1], q)].tag,
            getattr(identified[(cell[0], cell[1], q)], "description", None),
            getattr(identified[(cell[0], cell[1], q)], "image", None),
        )
        for q in QUARTERS
        if (cell[0], cell[1], q) in identified and hasattr(identified[(cell[0], cell[1], q)], "tag")
    }
    tags = set(filter_cell_warp_quarter_tags(cell_q).values())
    if "entrance" not in tags:
        return None
    order = ("down", "up", "left", "right")
    if preferred in DIRECTIONS:
        order = (preferred,) + tuple(d for d in order if d != preferred)
    for direction in order:
        if is_boundary_exit_direction(kinds, cell, direction):
            return direction
    return None


def format_path_runs(runs: List[Tuple[str, int, str]]) -> str:
    parts = []
    for direction, steps, move in runs:
        if move == "jump":
            parts.append(f"jump {direction}")
        else:
            parts.append(f"{direction} {steps}")
    return ", ".join(parts)


def edge_hazard_annotations(
    cells: List[Cell],
    kinds: Dict[Cell, str],
    people: Optional[Dict[Cell, str]] = None,
    direction: Optional[str] = None,
) -> str:
    if not cells:
        return ""
    hazard_cells = set(trainer_los_cells(kinds, people)) if people else set()
    jump_only: List[Cell] = []
    roofed_only: List[Cell] = []
    los_only: List[Cell] = []
    for cell in cells:
        path = shortest_path(kinds, (0, 0), [cell], people=people)
        if path is None:
            continue
        has_jump = any(m == "jump" for _, m, _ in path)
        is_roofed = bool(direction) and is_ledge_roofed_edge_cell(cell, direction, kinds)
        if has_jump:
            jump_only.append(cell)
        elif is_roofed:
            roofed_only.append(cell)
        if hazard_cells and any(step_cell in hazard_cells for _, _, step_cell in path):
            los_only.append(cell)
    notes = ""
    if jump_only and len(jump_only) < len(cells):
        j_str = ", ".join(f"({x}, {y})" for x, y in jump_only)
        notes += f" [requires one-way ledge jump: {j_str}]"
    if roofed_only and len(roofed_only) < len(cells):
        r_str = ", ".join(f"({x}, {y})" for x, y in roofed_only)
        notes += f" [drops below one-way ledge: {r_str}]"
    if los_only and len(los_only) < len(cells):
        l_str = ", ".join(f"({x}, {y})" for x, y in los_only)
        notes += f" [crosses line of sight: {l_str}]"
    return notes


def verbalize_edges(
    kinds: Dict[Cell, str],
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
    people: Optional[Dict[Cell, str]] = None,
    global_map: Optional[Any] = None,
) -> str:
    warps = warp_cells(identified=identified, global_map=global_map)
    lines = []
    for direction in DIRECTIONS:
        cells = edge_cells(kinds, direction)
        if cells:
            open_cells = filter_open_lane_edge_cells(
                cells, direction, kinds, global_map=global_map, identified=identified, people=people
            )
            listed = ", ".join(f"({x}, {y})" for x, y in cells)
            path = shortest_path(kinds, (0, 0), open_cells or cells, avoid_cells=warps, people=people)
            if path:
                target = path[-1][2]
                lat_path = shortest_path(
                    kinds,
                    (0, 0),
                    [target],
                    prefer_lateral=not (abs(target[0]) >= 4 and abs(target[1]) == 4),
                    avoid_cells=warps,
                    people=people,
                )
                runs_text = format_path_runs(compress_path(lat_path or path))
                listed += f" (shortest path to {target}: {runs_text})"
            listed += edge_hazard_annotations(cells, kinds, people=people, direction=direction)
        else:
            listed = "none"
        lines.append(f"- {direction} edge: {listed}")
    return "\n".join(lines)


def diagonal_barrier_pairs(kinds: Dict[Cell, str], goals: Iterable[Cell]) -> List[Tuple[Cell, Cell]]:
    goal_set = set(goals)
    if not goal_set:
        return []
    r_player = {c for c in kinds if c == (0, 0) or shortest_path(kinds, (0, 0), [c]) is not None}
    r_target = set(goal_set)
    queue = deque(goal_set)
    while queue:
        cur = queue.popleft()
        for _, nxt in neighbours(cur):
            if in_bounds(nxt) and kinds.get(nxt) == WALKABLE and nxt not in r_target:
                r_target.add(nxt)
                queue.append(nxt)
    pinches: List[Tuple[Cell, Cell]] = []
    for (px, py) in sorted(r_player):
        for dx in (-1, 1):
            for dy in (-1, 1):
                tx, ty = px + dx, py + dy
                if (tx, ty) in r_target:
                    b1, b2 = (px + dx, py), (px, py + dy)
                    if kinds.get(b1, BLOCKED) != WALKABLE and kinds.get(b2, BLOCKED) != WALKABLE:
                        pair = tuple(sorted((b1, b2)))
                        if pair not in pinches:
                            pinches.append(pair)
    return pinches


def describe_target_reachability(
    kinds: Optional[Dict[Cell, str]],
    target: Cell,
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
    people: Optional[Dict[Cell, str]] = None,
) -> str:
    if not kinds or target == (0, 0):
        return ""
    if kinds.get(target) == WALKABLE:
        goals = [target]
    else:
        goals = list(approach_cells(kinds, target, identified=identified, people=people))
    if not goals:
        return " [no walkable approach tile adjacent on current screen]"
    path = shortest_path(kinds, (0, 0), goals, people=people)
    if path is not None:
        if not path:
            if abs(target[0]) + abs(target[1]) == 2:
                return " [adjacent across counter]"
            return " [adjacent to player]"
        runs_text = format_path_runs(compress_path(path))
        return f" [on-screen path: {runs_text}]"
    pinches = diagonal_barrier_pairs(kinds, goals)
    if pinches:
        b1, b2 = pinches[0]
        return (
            f" [no walkable path on current screen — separated by an obstacle barrier "
            f"(obstacles at {b1} and {b2} meet diagonally with no orthogonal passage); "
            f"MoveTo will fail with no_path, use MoveOff to route around the obstacle]"
        )
    return (
        " [no walkable path on current screen — separated by an obstacle barrier; "
        "MoveTo will fail with no_path, use MoveOff to route around the obstacle]"
    )
