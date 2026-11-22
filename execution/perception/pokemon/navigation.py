import re
from collections import deque
from typing import Any, Collection, Dict, Iterable, List, Optional, Tuple

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
    if not (bottom_tags <= WALKABLE_TAGS) or (top_tags <= WALKABLE_TAGS):
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


def is_east_cliff_barrier_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
) -> bool:
    cell_q = quarter_tags.get(cell, {})
    if len(cell_q) < 4:
        return False
    cell_d = (quarter_descriptions or {}).get(cell, {})
    left_tags = {cell_q["top_left"], cell_q["bottom_left"]}
    if not (left_tags <= WALKABLE_TAGS):
        return False
    return (
        is_ridge_quarter(cell_q["top_right"], cell_d.get("top_right"))
        and is_ridge_quarter(cell_q["bottom_right"], cell_d.get("bottom_right"))
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
) -> Dict[Cell, str]:
    from execution.perception.pokemon.tiles import normalize_tabletop_tag

    # ``ledge_patterns`` / ``walk_patterns``: quarter hashes (QUARTERS order) learned from movement.
    tags: Dict[Cell, set] = {}
    quarter_tags: Dict[Cell, Dict[str, str]] = {}
    quarter_descriptions: Dict[Cell, Dict[str, Optional[str]]] = {}
    hashes: Dict[Cell, Dict[str, Optional[str]]] = {}
    for (x, y, quarter), tile in (identified or {}).items():
        desc = getattr(tile, "description", None)
        norm_tag = normalize_tabletop_tag(tile.tag, desc)
        tags.setdefault((x, y), set()).add(norm_tag)
        quarter_tags.setdefault((x, y), {})[quarter] = norm_tag
        quarter_descriptions.setdefault((x, y), {})[quarter] = desc
        hashes.setdefault((x, y), {})[quarter] = tile.hash
    confirmed_occupied: set = set()
    for cell, label in (people or {}).items():
        if cell == (0, 0):
            continue
        lower = (label or "").lower()
        if "remembered here" in lower or "seen only in part" in lower:
            continue
        confirmed_occupied.add(cell)
    kinds = {}
    for cell, cell_tags in tags.items():
        pattern = tuple(hashes[cell].get(q) for q in QUARTERS)
        if cell in confirmed_occupied:
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
            and bool(cell_tags & WALKABLE_TAGS)
            and not is_east_cliff_barrier_cell(cell, quarter_tags, quarter_descriptions)
        ):
            kinds[cell] = WALKABLE
        else:
            kinds[cell] = BLOCKED
    return kinds


def neighbours(cell: Cell) -> List[Tuple[str, Cell]]:
    return [(d, (cell[0] + dx, cell[1] + dy)) for d, (dx, dy) in DIRECTIONS.items()]


def shortest_path(kinds: Dict[Cell, str], start: Cell, goals: Iterable[Cell]) -> Optional[List[Tuple[str, str, Cell]]]:
    goals = set(goals)
    if start in goals:
        return []
    previous: Dict[Cell, Tuple[Cell, str, str]] = {start: None}
    queue = deque([start])
    while queue:
        cell = queue.popleft()
        for direction, nxt in neighbours(cell):
            if not in_bounds(nxt):
                continue
            kind = kinds.get(nxt, BLOCKED)
            if kind == WALKABLE:
                landing, move = nxt, "walk"
            elif (kind == LEDGE and direction == "down") or (kind == LEDGE_LEFT and direction == "left"):
                dx, dy = DIRECTIONS[direction]
                landing, move = (nxt[0] + dx, nxt[1] + dy), "jump"
                if not in_bounds(landing) or kinds.get(landing, BLOCKED) != WALKABLE:
                    continue
            else:
                continue
            if landing in previous:
                continue
            previous[landing] = (cell, direction, move)
            if landing in goals:
                path = []
                while previous[landing] is not None:
                    parent, d, m = previous[landing]
                    path.append((d, m, landing))
                    landing = parent
                return path[::-1]
            queue.append(landing)
    return None


def compress_path(path: List[Tuple[str, str, Cell]]) -> List[Tuple[str, int, str]]:
    runs: List[Tuple[str, int, str]] = []
    for direction, move, _ in path:
        if move == "walk" and runs and runs[-1][0] == direction and runs[-1][2] == "walk" and runs[-1][1] < MAX_STEPS_PER_MOVE:
            runs[-1] = (direction, runs[-1][1] + 1, "walk")
        else:
            runs.append((direction, 1, move))
    return runs


def approach_cells(kinds: Dict[Cell, str], target: Cell) -> Dict[Cell, str]:
    cells = {}
    for direction, cell in neighbours(target):
        if in_bounds(cell) and kinds.get(cell) == WALKABLE:
            facing = next(d for d, (dx, dy) in DIRECTIONS.items() if (cell[0] + dx, cell[1] + dy) == target)
            cells[cell] = facing
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


def parse_cell(text: Optional[str]) -> Optional[Cell]:
    if not text:
        return None
    match = COORDINATE_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def parse_all_cells(text: Optional[str]) -> List[Cell]:
    if not text:
        return []
    return [(int(m.group(1)), int(m.group(2))) for m in COORDINATE_PATTERN.finditer(text)]


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


def filter_open_lane_edge_cells(
    cells: Iterable[Cell],
    direction: str,
    kinds: Dict[Cell, str],
    global_map: Optional[Any] = None,
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

    if seg is not None:
        beyond_clear = []
        for c in cells_list:
            beyond = seg.get((px + c[0] + dx, py + c[1] + dy))
            if beyond is None:
                beyond_clear.append(c)
            elif beyond.kind == WALKABLE:
                beyond_clear.append(c)
            elif (beyond.kind == LEDGE and direction == "down") or (beyond.kind == LEDGE_LEFT and direction == "left"):
                landing = seg.get((px + c[0] + 2 * dx, py + c[1] + 2 * dy))
                if landing is None or landing.kind == WALKABLE:
                    beyond_clear.append(c)
        if beyond_clear:
            cells_list = beyond_clear

    inward_clear = [
        c for c in cells_list
        if kinds.get((c[0] - dx, c[1] - dy)) == WALKABLE
        or (kinds.get((c[0] - dx, c[1] - dy)) == LEDGE and direction == "down")
        or (kinds.get((c[0] - dx, c[1] - dy)) == LEDGE_LEFT and direction == "left")
    ]
    if inward_clear:
        cells_list = inward_clear
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
    if identified is None:
        return None
    tags = {
        identified[(cell[0], cell[1], q)].tag
        for q in QUARTERS
        if (cell[0], cell[1], q) in identified and hasattr(identified[(cell[0], cell[1], q)], "tag")
    }
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


def verbalize_edges(kinds: Dict[Cell, str]) -> str:
    lines = []
    for direction in DIRECTIONS:
        cells = edge_cells(kinds, direction)
        if cells:
            listed = ", ".join(f"({x}, {y})" for x, y in cells)
            path = shortest_path(kinds, (0, 0), cells)
            if path:
                target = path[-1][2]
                runs_text = format_path_runs(compress_path(path))
                listed += f" (shortest path to {target}: {runs_text})"
        else:
            listed = "none"
        lines.append(f"- {direction} edge: {listed}")
    return "\n".join(lines)
