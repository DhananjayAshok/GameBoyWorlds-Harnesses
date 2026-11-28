import heapq
import re
from collections import deque
from typing import Any, Collection, Dict, Iterable, List, Optional, Sequence, Tuple

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
SEATING_QUARTER_DESC_PATTERN = re.compile(
    r"\b(?:stool|stools|chair|chairs|cushion|cushions|seat|seats)\b",
    re.IGNORECASE,
)
SEATING_TABLETOP_REJECT_DESC_PATTERN = re.compile(
    r"\b(?:flask|flasks|plate|plates|cup|cups|book|books|map|trophy|statue|pot|plant|tv|television|sink|stove|bed)\b",
    re.IGNORECASE,
)
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


def _is_top_cliff_border_overhang_floor_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
) -> bool:
    if quarter_images is None:
        return False
    cell_q = quarter_tags.get(cell, {})
    cell_imgs = quarter_images.get(cell, {})
    if len(cell_q) < 4 or len(cell_imgs) < 4:
        return False
    if cell_q.get("top_left") != "obstacle" or cell_q.get("top_right") != "obstacle":
        return False
    if (
        cell_q.get("bottom_left") not in ("ground", "ground decoration")
        or cell_q.get("bottom_right") not in ("ground", "ground decoration")
    ):
        return False
    if any(cell_imgs.get(q) is None for q in QUARTERS):
        return False
    tl = np.squeeze(np.asarray(cell_imgs["top_left"]))
    tr = np.squeeze(np.asarray(cell_imgs["top_right"]))
    bl = np.squeeze(np.asarray(cell_imgs["bottom_left"]))
    br = np.squeeze(np.asarray(cell_imgs["bottom_right"]))
    if tl.shape != (8, 8) or tr.shape != (8, 8) or bl.shape != (8, 8) or br.shape != (8, 8):
        return False
    bl_sample = bl
    br_sample = br
    if not np.array_equal(bl, br) or (0 in np.unique(bl[4:, :]) and 0 not in np.unique(bl[:4, :])):
        bl_sample = bl[:4, :]
        br_sample = br[:4, :]
    if not (np.array_equal(tl, tr) and np.array_equal(bl_sample, br_sample)):
        return False
    ground_shades = {int(v) for v in np.unique(bl_sample)}
    if 0 in ground_shades or len(ground_shades) < 2:
        return False
    top_lower_shades = {int(v) for v in np.unique(tl[4:, :])}
    if not (top_lower_shades <= ground_shades) or 0 not in np.unique(tl[:2, :]):
        return False
    if abs(float(tl[4:, :].mean()) - float(bl_sample.mean())) > 25.0:
        return False
    return True


def is_top_overhang_floor_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
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
    is_cliff_overhang = _is_top_cliff_border_overhang_floor_cell(cell, quarter_tags, quarter_images)
    if not is_cliff_overhang and quarter_descriptions is not None:
        cell_d = quarter_descriptions.get(cell, {})
        for q in ("top_left", "top_right"):
            if cell_q.get(q) in FURNITURE_OVERHANG_TOP_TAGS:
                desc = cell_d.get(q) or ""
                if not FURNITURE_OVERHANG_DESC_PATTERN.search(desc):
                    return False
    above_q = quarter_tags.get((cell[0], cell[1] + 1), {})
    above_bottom = {above_q[q] for q in ("bottom_left", "bottom_right") if q in above_q}
    if above_bottom and not is_cliff_overhang:
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
    if cell_q["bottom_left"] in ("obstacle", "interactable object") and cell_q["bottom_right"] in ("obstacle", "interactable object"):
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


def _is_bottom_quarter_npc_head_overhang_image(
    img: Optional[Any],
    walkable_arrays: Sequence[np.ndarray],
) -> bool:
    arr = _quarter_array(img)
    if arr is None:
        return False
    top4 = arr[:4, :]
    bot4 = arr[4:, :]
    top_black = int((top4 == 0).sum())
    bot_black = int((bot4 == 0).sum())
    if bot_black <= 0:
        return False
    if top_black == 0:
        return True
    for w_arr in walkable_arrays:
        if w_arr is not None and w_arr.shape == (8, 8):
            if np.array_equal(top4, w_arr[:4, :]) or np.array_equal(top4, w_arr[4:, :]):
                if bot_black > top_black or int((bot4 >= 240).sum()) > int((top4 >= 240).sum()):
                    return True
    return False


def is_npc_head_overhang_floor_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
    global_map: Optional[Any] = None,
) -> bool:
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
    if "npc" in below_top:
        return True
    if quarter_images is not None:
        cell_imgs = quarter_images.get(cell, {})
        walkable_arrays = [
            arr
            for q in QUARTERS
            if cell_q.get(q) in WALKABLE_TAGS
            and (arr := _quarter_array(cell_imgs.get(q))) is not None
        ]
        npc_bottom_qs = [q for q in ("bottom_left", "bottom_right") if cell_q.get(q) == "npc"]
        if npc_bottom_qs and all(
            _is_bottom_quarter_npc_head_overhang_image(cell_imgs.get(q), walkable_arrays)
            for q in npc_bottom_qs
        ):
            return True
    if cell[1] != min(Y_RANGE) or below_q:
        return False
    if (
        global_map is not None
        and getattr(global_map, "current", None)
        and getattr(global_map, "pos", None) is not None
    ):
        try:
            si, (px, py) = global_map.resolve(global_map.segment, global_map.pos)
            seg = global_map.segments[si] if 0 <= si < len(global_map.segments) else {}
            abs_cell = (px + cell[0], py + cell[1])
            mc = seg.get(abs_cell)
            if mc is not None and getattr(mc, "kind", None) == WALKABLE:
                return True
        except Exception:  # noqa: BLE001
            pass
    return False


PATTERNED_FLOOR_OBSTACLE_REJECT_DESC_PATTERN = re.compile(
    r"\b(?:ledge|cliff|ridge|fence|tree|bush|plant|table|desk|counter|chair|stool|bed|shelf|bookshelf|cabinet|water|shoreline|sign|signpost|house|building|roof|door|window|porthole|rock|boulder)\b",
    re.IGNORECASE,
)


def _quarter_array(img: Optional[Any]) -> Optional[np.ndarray]:
    if img is None or getattr(img, "size", 0) == 0:
        return None
    arr = np.squeeze(np.asarray(img))
    return arr if arr.shape == (8, 8) else None


def _dominant_ground_shades(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
) -> set:
    if quarter_images is None:
        return set()
    cell_q = quarter_tags.get(cell, {})
    cell_imgs = quarter_images.get(cell, {})
    cell_d = (quarter_descriptions or {}).get(cell, {})
    if "obstacle" in set(cell_q.values()):
        for q in QUARTERS:
            arr = _quarter_array(cell_imgs.get(q))
            if arr is not None and int(arr.min()) != int(arr.max()):
                if PATTERNED_FLOOR_OBSTACLE_REJECT_DESC_PATTERN.search(cell_d.get(q) or ""):
                    return set()
    doms: set = set()
    for q in QUARTERS:
        if cell_q.get(q) not in ("ground", "ground decoration"):
            continue
        arr = _quarter_array(cell_imgs.get(q))
        if arr is None:
            continue
        vals, counts = np.unique(arr, return_counts=True)
        if len(vals) < 2 or int(vals.min()) < 56:
            continue
        best_idx = int(np.argmax(counts))
        if float(counts[best_idx]) / arr.size >= 0.50:
            doms.add(int(vals[best_idx]))
    return doms


def _uniform_obstacle_floor_shade(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
) -> Optional[int]:
    if quarter_images is None:
        return None
    cell_q = quarter_tags.get(cell, {})
    cell_imgs = quarter_images.get(cell, {})
    if len(cell_q) < 4 or len(cell_imgs) < 4:
        return None
    tag_set = set(cell_q.values())
    if "obstacle" not in tag_set or not (tag_set <= {"ground", "ground decoration", "obstacle"}):
        return None
    cell_d = (quarter_descriptions or {}).get(cell, {})
    obst_shades: set = set()
    uniform_ground_shades: set = set()
    has_multishade_ground = False
    for q in QUARTERS:
        arr = _quarter_array(cell_imgs.get(q))
        if arr is None:
            return None
        if int(arr.min()) != int(arr.max()):
            if cell_q.get(q) == "obstacle":
                return None
            if PATTERNED_FLOOR_OBSTACLE_REJECT_DESC_PATTERN.search(cell_d.get(q) or ""):
                return None
            has_multishade_ground = True
        else:
            s = int(arr.min())
            if s < 88:
                return None
            if cell_q.get(q) == "obstacle":
                obst_shades.add(s)
            else:
                uniform_ground_shades.add(s)
    if len(obst_shades) != 1:
        return None
    s = next(iter(obst_shades))
    if uniform_ground_shades and uniform_ground_shades != {s}:
        return None
    if has_multishade_ground and s not in _dominant_ground_shades(
        cell, quarter_tags, quarter_descriptions, quarter_images
    ):
        return None
    return s


def is_patterned_floor_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
    min_ground_quarters: int = 2,
) -> bool:
    if quarter_images is None:
        return False
    cell_q = quarter_tags.get(cell, {})
    cell_imgs = quarter_images.get(cell, {})
    if len(cell_q) < 4 or len(cell_imgs) < 4:
        return False
    tag_set = set(cell_q.values())
    if not ("obstacle" in tag_set and bool(tag_set & {"ground", "ground decoration"})):
        return False
    if not (tag_set <= {"ground", "ground decoration", "obstacle"}):
        return False
    ground_qs = [q for q in QUARTERS if cell_q.get(q) in ("ground", "ground decoration")]
    obst_qs = [q for q in QUARTERS if cell_q.get(q) == "obstacle"]
    if not ground_qs or not obst_qs:
        return False
    uni_shade = _uniform_obstacle_floor_shade(cell, quarter_tags, quarter_descriptions, quarter_images)
    if uni_shade is not None and uni_shade in _dominant_ground_shades(
        cell, quarter_tags, quarter_descriptions, quarter_images
    ):
        return True
    if len(ground_qs) < min_ground_quarters:
        return False
    cell_d = (quarter_descriptions or {}).get(cell, {})
    for q in QUARTERS:
        desc = cell_d.get(q) or ""
        if PATTERNED_FLOOR_OBSTACLE_REJECT_DESC_PATTERN.search(desc):
            return False
    ground_shades: set = set()
    ground_means: List[float] = []
    for q in ground_qs:
        arr = _quarter_array(cell_imgs.get(q))
        if arr is None:
            return False
        vals = np.unique(arr)
        if int(arr.min()) < 56:
            return False
        ground_shades.update(int(v) for v in vals)
        ground_means.append(float(arr.mean()))
    if len(ground_shades) < 2 or (max(ground_shades) - min(ground_shades)) > 96:
        return False
    mean_ground = sum(ground_means) / len(ground_means)
    for q in obst_qs:
        arr = _quarter_array(cell_imgs.get(q))
        if arr is None:
            return False
        vals, counts = np.unique(arr, return_counts=True)
        shades = {int(v) for v in vals}
        if shades != ground_shades:
            return False
        if int(counts[0]) > 12:
            return False
        if abs(float(arr.mean()) - mean_ground) > 12.0:
            return False
    return True


def infer_spatial_patterned_floor_cells(
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
    hashes: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
) -> set:
    inferred: set = set()
    for cell in quarter_tags:
        if is_patterned_floor_cell(cell, quarter_tags, quarter_descriptions, quarter_images, min_ground_quarters=2):
            inferred.add(cell)
    if quarter_images is None:
        return inferred
    dom_by_cell = {
        cell: _dominant_ground_shades(cell, quarter_tags, quarter_descriptions, quarter_images)
        for cell in quarter_tags
    }
    if (0, 0) not in dom_by_cell or not dom_by_cell.get((0, 0)):
        player_doms: set = set()
        for _, nb in neighbours((0, 0)):
            player_doms.update(dom_by_cell.get(nb, set()))
        if player_doms:
            dom_by_cell[(0, 0)] = player_doms
    uni_by_cell = {
        cell: _uniform_obstacle_floor_shade(cell, quarter_tags, quarter_descriptions, quarter_images)
        for cell in quarter_tags
    }
    queue = deque()
    for cell, s in uni_by_cell.items():
        if s is None:
            continue
        if s in dom_by_cell.get(cell, set()) or any(s in dom_by_cell.get(nb, set()) for _, nb in neighbours(cell)):
            inferred.add(cell)
            queue.append(cell)
    while queue:
        cur = queue.popleft()
        s = uni_by_cell.get(cur)
        for _, nb in neighbours(cur):
            if nb not in inferred and uni_by_cell.get(nb) == s:
                inferred.add(nb)
                queue.append(nb)
    if not hashes:
        return inferred
    by_pattern: Dict[Tuple[Optional[str], ...], List[Cell]] = {}
    for cell in quarter_tags:
        if cell in inferred:
            continue
        pat = tuple(hashes.get(cell, {}).get(q) for q in QUARTERS)
        if len(pat) == 4 and None not in pat:
            if is_patterned_floor_cell(
                cell, quarter_tags, quarter_descriptions, quarter_images, min_ground_quarters=1
            ):
                by_pattern.setdefault(pat, []).append(cell)
    for pat, cells in by_pattern.items():
        xs = {c[0] for c in cells}
        ys = {c[1] for c in cells}
        if len(cells) >= 4 and len(xs) >= 2 and len(ys) >= 2:
            inferred.update(cells)
    return inferred


def is_walkable_seating_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
) -> bool:
    if quarter_descriptions is None:
        return False
    cell_q = quarter_tags.get(cell, {})
    if len(cell_q) < 4:
        return False
    tag_set = set(cell_q.values())
    if not (tag_set & {"obstacle", "interactable object"}) or not (
        tag_set <= {"ground", "ground decoration", "obstacle", "interactable object"}
    ):
        return False
    cell_d = quarter_descriptions.get(cell, {})
    has_seating = False
    for q in QUARTERS:
        t = cell_q.get(q)
        desc = (cell_d.get(q) or "").strip()
        if t in ("ground", "ground decoration"):
            continue
        if not desc or not SEATING_QUARTER_DESC_PATTERN.search(desc):
            return False
        if SEATING_TABLETOP_REJECT_DESC_PATTERN.search(desc):
            return False
        has_seating = True
    return has_seating


def is_overhang_floor_cell(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
    global_map: Optional[Any] = None,
) -> bool:
    return (
        is_top_overhang_floor_cell(cell, quarter_tags, quarter_descriptions, quarter_images)
        or is_player_overhang_floor_cell(cell, quarter_tags, quarter_descriptions)
        or is_npc_head_overhang_floor_cell(cell, quarter_tags, quarter_images=quarter_images, global_map=global_map)
        or is_walkable_seating_cell(cell, quarter_tags, quarter_descriptions)
    )


def is_walkable_pattern_learnable(
    tags: Iterable[str],
    hashes: Optional[Tuple[Optional[str], ...]] = None,
    pixels: Optional[Any] = None,
    quarter_tags: Optional[Dict[str, str]] = None,
    quarter_descriptions: Optional[Dict[str, Optional[str]]] = None,
    quarter_images: Optional[Dict[str, Optional[Any]]] = None,
) -> bool:
    tag_set = set(quarter_tags.values()) if quarter_tags else set(tags)
    if "npc" in tag_set or "unknown" in tag_set or "water" in tag_set:
        return False
    if quarter_tags and len(quarter_tags) == 4:
        q_map = {(0, 0): quarter_tags}
        d_map = {(0, 0): quarter_descriptions or {}} if quarter_descriptions is not None else None
        i_map = {(0, 0): quarter_images or {}} if quarter_images is not None else None
        if is_top_overhang_floor_cell((0, 0), q_map, d_map, i_map):
            return True
        if is_walkable_seating_cell((0, 0), q_map, d_map):
            return True
        if is_patterned_floor_cell((0, 0), q_map, d_map, i_map, min_ground_quarters=1):
            return True
    if hashes is not None and len(hashes) == 4 and len(set(hashes)) == 1 and hashes[0] is not None:
        sample = pixels
        if (sample is None or getattr(sample, "size", 0) == 0) and quarter_images:
            sample = next((img for img in quarter_images.values() if img is not None), None)
        if sample is not None and getattr(sample, "size", 0) > 0:
            arr = np.asarray(sample)
            min_s, max_s = int(arr.min()), int(arr.max())
            if min_s == max_s:
                return False
            if "obstacle" in tag_set and (min_s < 56 or (max_s - min_s) > 72):
                return False
        return True
    if "obstacle" in tag_set:
        return False
    if bool(tag_set & WALKABLE_TAGS):
        return True
    return False


def classify_cell_from_quarters(
    cell: Cell,
    quarter_tags: Dict[Cell, Dict[str, str]],
    quarter_descriptions: Optional[Dict[Cell, Dict[str, Optional[str]]]] = None,
    quarter_images: Optional[Dict[Cell, Dict[str, Optional[Any]]]] = None,
    hashes: Optional[Tuple[Optional[str], ...]] = None,
    patterned_floor_cells: Optional[Collection[Cell]] = None,
    ledge_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None,
    left_ledge_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None,
    walk_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None,
    allow_player_overhang: bool = True,
    allow_player_cap: bool = True,
    global_map: Optional[Any] = None,
) -> Tuple[str, Tuple[str, ...]]:
    cell_q = quarter_tags.get(cell, {})
    cell_d = (quarter_descriptions or {}).get(cell, {})
    cell_imgs = (quarter_images or {}).get(cell, {})
    cell_tag_set = set(cell_q.values())
    raw_tags = tuple(sorted(cell_tag_set))
    if is_east_cliff_barrier_cell(
        cell, quarter_tags, quarter_descriptions, quarter_images, allow_player_cap=allow_player_cap
    ):
        return BLOCKED, raw_tags
    if is_top_overhang_floor_cell(cell, quarter_tags, quarter_descriptions, quarter_images):
        tags = tuple(sorted({cell_q[q] for q in ("bottom_left", "bottom_right") if q in cell_q})) or ("ground",)
        return WALKABLE, tags
    if is_walkable_seating_cell(cell, quarter_tags, quarter_descriptions):
        tags = tuple(sorted({t for t in cell_q.values() if t in WALKABLE_TAGS})) or ("ground",)
        return WALKABLE, tags
    if (
        allow_player_overhang and is_player_overhang_floor_cell(cell, quarter_tags, quarter_descriptions)
    ) or is_npc_head_overhang_floor_cell(cell, quarter_tags, quarter_images=quarter_images, global_map=global_map):
        tags = tuple(sorted({cell_q[q] for q in ("top_left", "top_right") if q in cell_q})) or ("ground",)
        return WALKABLE, tags
    in_patterned = (
        cell in patterned_floor_cells
        if patterned_floor_cells is not None
        else is_patterned_floor_cell(cell, quarter_tags, quarter_descriptions, quarter_images)
    )
    if in_patterned:
        tags = tuple(sorted({t for t in cell_q.values() if t in WALKABLE_TAGS})) or ("ground",)
        return WALKABLE, tags
    if is_south_ledge_cell(cell, quarter_tags, quarter_descriptions):
        tags = tuple(sorted({cell_q[q] for q in ("top_left", "top_right") if q in cell_q} | {"ledge"}))
        return LEDGE, tags
    if is_west_ledge_cell(cell, quarter_tags, quarter_descriptions):
        tags = tuple(sorted({cell_q[q] for q in ("top_right", "bottom_right") if q in cell_q} | {"ledge"}))
        return LEDGE_LEFT, tags
    if "npc" in cell_tag_set:
        return "occupied", raw_tags
    if cell_tag_set <= WALKABLE_TAGS:
        return WALKABLE, raw_tags
    if cell_tag_set <= WALKABLE_TAGS | LEDGE_TAGS:
        return LEDGE, raw_tags
    if "unknown" in cell_tag_set and (cell_tag_set - {"unknown"}) <= WALKABLE_TAGS | LEDGE_TAGS:
        return "unknown", raw_tags
    if hashes is not None and ledge_patterns and hashes in ledge_patterns:
        return LEDGE, raw_tags
    if hashes is not None and left_ledge_patterns and hashes in left_ledge_patterns:
        return LEDGE_LEFT, raw_tags
    if (
        hashes is not None
        and walk_patterns
        and hashes in walk_patterns
        and is_walkable_pattern_learnable(
            cell_tag_set,
            hashes=hashes,
            quarter_tags=cell_q,
            quarter_descriptions=cell_d,
            quarter_images=cell_imgs,
        )
    ):
        return WALKABLE, raw_tags
    return BLOCKED, raw_tags


def cell_kinds(
    identified,
    ledge_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None,
    walk_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None,
    people: Optional[Dict[Cell, str]] = None,
    left_ledge_patterns: Optional[Collection[Tuple[Optional[str], ...]]] = None,
    blocked_cells: Optional[Collection[Cell]] = None,
    global_map: Optional[Any] = None,
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
    patterned_floor_cells = infer_spatial_patterned_floor_cells(
        quarter_tags, quarter_descriptions, quarter_images, hashes
    )
    confirmed_occupied: set = set()
    for cell, label in (people or {}).items():
        if cell == (0, 0):
            continue
        lower = (label or "").lower()
        if "remembered here" in lower or "seen only in part" in lower:
            continue
        if lower.startswith("sign:"):
            cell_q = quarter_tags.get(cell, {})
            if len(cell_q) == 4 and set(cell_q.values()) <= WALKABLE_TAGS:
                continue
        confirmed_occupied.add(cell)
    empirical_blocked = set(blocked_cells or ())
    kinds = {}
    for cell, cell_tags in tags.items():
        pattern = tuple(hashes[cell].get(q) for q in QUARTERS)
        if cell in confirmed_occupied:
            kinds[cell] = BLOCKED
            continue
        if cell != (0, 0) and cell in empirical_blocked and not (cell_tags & WARP_TAGS):
            kinds[cell] = BLOCKED
            continue
        if cell_tags & WARP_TAGS and not is_east_cliff_barrier_cell(
            cell, quarter_tags, quarter_descriptions, quarter_images, allow_player_cap=True
        ):
            kinds[cell] = WALKABLE
            continue
        kind, _ = classify_cell_from_quarters(
            cell,
            quarter_tags,
            quarter_descriptions,
            quarter_images,
            hashes=pattern,
            patterned_floor_cells=patterned_floor_cells,
            ledge_patterns=ledge_patterns,
            left_ledge_patterns=left_ledge_patterns,
            walk_patterns=walk_patterns,
            allow_player_overhang=True,
            allow_player_cap=True,
            global_map=global_map,
        )
        kinds[cell] = BLOCKED if kind in ("occupied", "unknown") else kind
    if (
        global_map is not None
        and getattr(global_map, "current", None)
        and getattr(global_map, "pos", None) is not None
    ):
        try:
            si, (px, py) = global_map.resolve(global_map.segment, global_map.pos)
            seg = global_map.segments[si] if 0 <= si < len(global_map.segments) else {}
            visits = global_map.visits[si] if 0 <= si < len(getattr(global_map, "visits", ())) else {}
            seg_blocked = (
                set(global_map.blocked_cells[si])
                if 0 <= si < len(getattr(global_map, "blocked_cells", ()))
                else set()
            )
            gatehouse_road_cells: set = set()
            if hasattr(global_map, "side_gatehouse_doorways"):
                for dw in global_map.side_gatehouse_doorways(si):
                    gatehouse_road_cells.update(dw.get("road_cells", ()))
            for cell, cell_tags in tags.items():
                if cell == (0, 0) or kinds.get(cell) != BLOCKED:
                    continue
                if cell in confirmed_occupied or cell in empirical_blocked:
                    continue
                if cell_tags & {"npc", "water", "sign", "interactable object", "item", "ledge", "right_ledge", "left_ledge"}:
                    continue
                abs_cell = (px + cell[0], py + cell[1])
                if abs_cell in seg_blocked:
                    continue
                mc = seg.get(abs_cell)
                if mc is not None and getattr(mc, "kind", None) == WALKABLE:
                    if visits.get(abs_cell, 0) > 0 or abs_cell in gatehouse_road_cells:
                        kinds[cell] = WALKABLE
        except Exception:  # noqa: BLE001
            pass
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
    best: Dict[Tuple[Cell, Optional[str]], Tuple[int, int, int, int, int, int]] = {(start, None): (0, 0, 0, 0, 0, 0)}
    previous: Dict[Tuple[Cell, Optional[str]], Optional[Tuple[Tuple[Cell, Optional[str]], str, str]]] = {(start, None): None}
    counter = 0
    heap: List[Tuple[int, int, int, int, int, int, int, Cell, Optional[str]]] = [(0, 0, 0, 0, 0, 0, 0, start, None)]
    while heap:
        jumps, avoids, los_hits, steps, turns, first_rank, _, cell, prev_dir = heapq.heappop(heap)
        state = (cell, prev_dir)
        if (jumps, avoids, los_hits, steps, turns, first_rank) > best.get(state, (jumps + 1, 0, 0, 0, 0, 0)):
            continue
        if cell in goal_set:
            path = []
            cur = state
            while previous[cur] is not None:
                parent, d, m = previous[cur]
                path.append((d, m, cur[0]))
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
            n_turns = turns + (1 if (prev_dir is not None and direction != prev_dir) else 0)
            rank = dir_rank[direction] if cell == start else first_rank
            cost = (n_jumps, n_avoids, n_los, n_steps, n_turns, rank)
            next_state = (landing, direction)
            if cost < best.get(next_state, (n_jumps + 1, 0, 0, 0, 0, 0)):
                best[next_state] = cost
                previous[next_state] = (state, direction, move)
                counter += 1
                heapq.heappush(heap, (n_jumps, n_avoids, n_los, n_steps, n_turns, rank, counter, landing, direction))
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
    r"\b(?:counter|desk|table|window|booth|chair|stool|bench|seat|cushion)\b",
    re.IGNORECASE,
)


def _cell_has_counter_desc(
    cell: Cell,
    kinds: Dict[Cell, str],
    identified: Dict[Tuple[int, int, str], Any],
) -> Tuple[bool, bool, set]:
    if not in_bounds(cell) or kinds.get(cell) != BLOCKED:
        return False, True, set()
    from execution.perception.pokemon.tiles import normalize_tabletop_tag

    has_counter_desc = False
    cell_hashes: set = set()
    for q in QUARTERS:
        tile = identified.get((cell[0], cell[1], q))
        if tile is None:
            continue
        desc = getattr(tile, "description", None) or ""
        tag = normalize_tabletop_tag(tile.tag, desc, getattr(tile, "image", None))
        if tag in ("npc", "sign", "water", "tall grass", "entrance", "staircase"):
            return False, True, set()
        h = getattr(tile, "hash", None)
        if h:
            cell_hashes.add(h)
        if COUNTER_SEPARATOR_DESC_PATTERN.search(desc):
            has_counter_desc = True
    return has_counter_desc, False, cell_hashes


def is_counter_separator_cell(
    cell: Cell,
    kinds: Dict[Cell, str],
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
) -> bool:
    if not in_bounds(cell) or kinds.get(cell) != BLOCKED:
        return False
    if identified is None:
        return True
    has_desc, disqualified, cell_hashes = _cell_has_counter_desc(cell, kinds, identified)
    if disqualified:
        return False
    if has_desc:
        return True
    for axis_nbs in (
        ((cell[0] - 1, cell[1]), (cell[0] + 1, cell[1])),
        ((cell[0], cell[1] - 1), (cell[0], cell[1] + 1)),
    ):
        nb_results = [_cell_has_counter_desc(nb, kinds, identified) for nb in axis_nbs]
        if all(r[0] and not r[1] for r in nb_results):
            return True
        for nb_has_desc, nb_disq, nb_hashes in nb_results:
            if nb_has_desc and not nb_disq and (cell == (0, 1) or bool(cell_hashes & nb_hashes)):
                return True
    return False


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


APPROACH_SIDE_TO_FACING = {
    "north": "down",
    "above": "down",
    "top": "down",
    "south": "up",
    "below": "up",
    "bottom": "up",
    "east": "left",
    "right": "left",
    "west": "right",
    "left": "right",
}

APPROACH_DIRECTION_CLAUSE_PATTERN = re.compile(
    r"(?:"
    r"\bfrom\s+(?:the\s+|directly\s+|its\s+)*(?:behind\s*\(\s*)?(north|south|east|west|left|right|above|below|top|bottom)(?:\s+or\s+(north|south|east|west|left|right|above|below|top|bottom))?"
    r"|\b(?:on|to)\s+(?:the\s+|its\s+)+(north|south|east|west|left|right|top|bottom)\s+side\b"
    r"|\bstand(?:ing)?\s+(?:directly\s+)?(north|south|east|west|left|right|above|below)\s+of\b"
    r"|\bfacing\s+(up|down|left|right|north|south|east|west)\b"
    r")",
    re.IGNORECASE,
)

FACING_WORD_TO_DIR = {
    "up": "up",
    "north": "up",
    "down": "down",
    "south": "down",
    "left": "left",
    "west": "left",
    "right": "right",
    "east": "right",
}

STAND_AT_COORD_PREFIX_PATTERN = re.compile(
    r"\b(?:walk\s+to|move\s+to|step\s+to|stand\s+(?:at|on)|from\s+(?:tile|cell|position)?)\s*$",
    re.IGNORECASE,
)

def is_bottom_edge_overhang_only_cell(
    cell: Cell,
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
    global_map: Optional[Any] = None,
) -> bool:
    if identified is None or cell[1] != min(Y_RANGE):
        return False
    from execution.perception.pokemon.tiles import filter_cell_warp_quarter_tags, normalize_tabletop_tag

    q_tags: Dict[Cell, Dict[str, str]] = {}
    q_imgs: Dict[Cell, Dict[str, Optional[Any]]] = {}
    for c in (cell, (cell[0], cell[1] - 1)):
        for q in QUARTERS:
            tile = identified.get((c[0], c[1], q))
            if tile is not None:
                img = getattr(tile, "image", None)
                q_tags.setdefault(c, {})[q] = normalize_tabletop_tag(
                    tile.tag, getattr(tile, "description", None), img
                )
                q_imgs.setdefault(c, {})[q] = img
        if c in q_tags:
            q_tags[c] = filter_cell_warp_quarter_tags(q_tags[c])
    return is_npc_head_overhang_floor_cell(cell, q_tags, quarter_images=q_imgs, global_map=global_map)


def resolve_interaction_target_and_constraints(
    task: Optional[str],
    kinds: Optional[Dict[Cell, str]] = None,
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
    people: Optional[Dict[Cell, str]] = None,
    global_map: Optional[Any] = None,
) -> Tuple[Optional[Cell], set, Optional[Cell]]:
    positive_cells = [c for c in parse_all_cells(task) if c != (0, 0)]
    allowed_facings: set = set()
    if task:
        for match in APPROACH_DIRECTION_CLAUSE_PATTERN.finditer(task):
            prefix = task[max(0, match.start() - 40):match.start()]
            clause = re.split(r"[,;.:()\[\]]", prefix)[-1]
            if AVOID_MARKER_PATTERN.search(clause):
                continue
            side1, side2, side3, side4, face_word = match.groups()
            for s in (side1, side2, side3, side4):
                if s:
                    f_dir = APPROACH_SIDE_TO_FACING.get(s.lower())
                    if f_dir:
                        allowed_facings.add(f_dir)
            if face_word:
                f_dir = FACING_WORD_TO_DIR.get(face_word.lower())
                if f_dir:
                    allowed_facings.add(f_dir)

    valid_cells: List[Cell] = []
    for c in positive_cells:
        if is_bottom_edge_overhang_only_cell(c, identified=identified, global_map=global_map):
            c = (c[0], c[1] - 1)
        if in_bounds(c) or (
            kinds is not None and bool(approach_cells(kinds, c, identified=identified, people=people))
        ):
            if c not in valid_cells:
                valid_cells.append(c)

    target_cell: Optional[Cell] = None
    stand_cell: Optional[Cell] = None
    if len(valid_cells) == 1:
        target_cell = valid_cells[0]
    elif len(valid_cells) >= 2:
        # Check if one cell is an explicit stand/walk-to cell and another is the target adjacent to it.
        coord_matches = list(COORDINATE_PATTERN.finditer(task or ""))
        stand_candidates: List[Cell] = []
        for m in coord_matches:
            c = (int(m.group(1)), int(m.group(2)))
            if is_bottom_edge_overhang_only_cell(c, identified=identified, global_map=global_map):
                c = (c[0], c[1] - 1)
            prefix = (task or "")[max(0, m.start() - 32):m.start()]
            if STAND_AT_COORD_PREFIX_PATTERN.search(prefix) and c in valid_cells:
                stand_candidates.append(c)
        non_stand = [c for c in valid_cells if c not in stand_candidates]
        if stand_candidates and non_stand:
            stand_cell = stand_candidates[0]
            target_cell = non_stand[0]
        else:
            # Prefer a cell that is an NPC/object target over a walkable approach neighbor
            npc_or_blocked = [
                c for c in valid_cells
                if _is_npc_target_cell(c, identified, people) or (kinds is not None and kinds.get(c) != WALKABLE)
            ]
            walkable_cands = [c for c in valid_cells if kinds is not None and kinds.get(c) == WALKABLE]
            if len(npc_or_blocked) == 1:
                target_cell = npc_or_blocked[0]
                for wc in walkable_cands:
                    if wc != target_cell and abs(wc[0] - target_cell[0]) + abs(wc[1] - target_cell[1]) in (1, 2):
                        stand_cell = wc
                        break
            else:
                target_cell = valid_cells[-1]
                if valid_cells[0] != target_cell and abs(valid_cells[0][0] - target_cell[0]) + abs(valid_cells[0][1] - target_cell[1]) in (1, 2):
                    stand_cell = valid_cells[0]

    if stand_cell is not None and target_cell is not None:
        dx = target_cell[0] - stand_cell[0]
        dy = target_cell[1] - stand_cell[1]
        for d_name, (ox, oy) in DIRECTIONS.items():
            if (dx, dy) == (ox, oy) or (dx, dy) == (2 * ox, 2 * oy):
                allowed_facings.add(d_name)
                break
    return target_cell, allowed_facings, stand_cell


def approach_cells(
    kinds: Dict[Cell, str],
    target: Cell,
    identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
    people: Optional[Dict[Cell, str]] = None,
    allowed_facings: Optional[Collection[str]] = None,
    preferred_stand_cell: Optional[Cell] = None,
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
    if preferred_stand_cell is not None:
        if preferred_stand_cell in cells:
            return {preferred_stand_cell: cells[preferred_stand_cell]}
        if not allowed_facings:
            return {}
    if allowed_facings:
        allowed_set = set(allowed_facings)
        return {c: f for c, f in cells.items() if f in allowed_set}
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


def _gmap_blocked_sets(global_map: Optional[Any]) -> Tuple[set, Dict[Cell, set]]:
    if global_map is None or getattr(global_map, "segment", None) is None:
        return set(), {}
    try:
        si = global_map.segment
        b_cells = (
            set(global_map.blocked_cells[si])
            if si < len(getattr(global_map, "blocked_cells", ()))
            else set()
        )
        if hasattr(global_map, "stationary_npc_squares"):
            b_cells |= set(global_map.stationary_npc_squares(si))
        b_moves = (
            {k: set(v) for k, v in global_map.segment_blocked_moves(si).items()}
            if hasattr(global_map, "segment_blocked_moves")
            else {}
        )
        for src, dirs in b_moves.items():
            for d_name in dirs:
                if d_name in DIRECTIONS:
                    ox, oy = DIRECTIONS[d_name]
                    b_cells.add((src[0] + ox, src[1] + oy))
        return b_cells, b_moves
    except Exception:  # noqa: BLE001
        return set(), {}


FRONT_TARGET_TAGS = {"npc", "interactable object", "sign", "item"}


def has_offscreen_front_target(
    global_map: Optional[Any],
    cell: Cell,
    direction: str,
    *,
    include_direct: bool = True,
) -> bool:
    if (
        global_map is None
        or not getattr(global_map, "current", None)
        or getattr(global_map, "pos", None) is None
        or direction not in DIRECTIONS
    ):
        return False
    try:
        si, (px, py) = global_map.resolve(global_map.segment, global_map.pos)
        seg = global_map.current
    except Exception:  # noqa: BLE001
        return False
    if not seg or (px, py) not in seg:
        return False
    stationary_npcs = (
        set(global_map.stationary_npc_squares(si))
        if hasattr(global_map, "stationary_npc_squares")
        else set()
    )
    dx, dy = DIRECTIONS[direction]
    start = (px + cell[0], py + cell[1])
    first = (start[0] + dx, start[1] + dy)
    fc = seg.get(first)
    if fc is None:
        return False
    f_kind = getattr(fc, "kind", None)
    f_tags = set(getattr(fc, "tags", ()))
    if include_direct and (f_kind == "occupied" or first in stationary_npcs or bool(f_tags & FRONT_TARGET_TAGS)):
        return True
    if f_kind == BLOCKED:
        second = (start[0] + 2 * dx, start[1] + 2 * dy)
        sc = seg.get(second)
        if sc is not None:
            s_kind = getattr(sc, "kind", None)
            s_tags = set(getattr(sc, "tags", ()))
            if s_kind == "occupied" or second in stationary_npcs or bool(s_tags & FRONT_TARGET_TAGS):
                return True
        perp = ((1, 0), (-1, 0)) if direction in ("up", "down") else ((0, 1), (0, -1))
        for ox, oy in perp:
            nb1 = seg.get((first[0] + ox, first[1] + oy))
            nb2 = seg.get((second[0] + ox, second[1] + oy))
            if (
                nb1 is not None
                and getattr(nb1, "kind", None) == BLOCKED
                and "interactable object" in getattr(nb1, "tags", ())
                and nb2 is not None
                and (
                    getattr(nb2, "kind", None) == "occupied"
                    or (second[0] + ox, second[1] + oy) in stationary_npcs
                    or bool(set(getattr(nb2, "tags", ())) & FRONT_TARGET_TAGS)
                )
            ):
                return True
    return False


def has_offscreen_continuation(
    global_map: Optional[Any],
    cell: Cell,
    direction: str,
) -> bool:
    if (
        global_map is None
        or not getattr(global_map, "current", None)
        or getattr(global_map, "pos", None) is None
        or direction not in DIRECTIONS
    ):
        return True
    try:
        _, (px, py) = global_map.resolve(global_map.segment, global_map.pos)
        seg = global_map.current
    except Exception:  # noqa: BLE001
        return True
    if not seg or (px, py) not in seg:
        return True
    b_cells, b_moves = _gmap_blocked_sets(global_map)
    return (
        _offscreen_continuation_depth(
            seg,
            (px + cell[0], py + cell[1]),
            direction,
            1,
            blocked_cells=b_cells,
            blocked_moves=b_moves,
        )
        > 0
    )


def _offscreen_continuation_depth(
    seg: Dict[Cell, Any],
    start: Cell,
    direction: str,
    max_depth: int = LOOKAHEAD_STEPS,
    blocked_cells: Optional[set] = None,
    blocked_moves: Optional[Dict[Cell, set]] = None,
) -> int:
    dx, dy = DIRECTIONS[direction]
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

    if blocked_moves and direction in blocked_moves.get(start, ()):
        return 0
    first = (start[0] + dx, start[1] + dy)
    if blocked_cells and first in blocked_cells:
        return 0
    fc = seg.get(first)
    if fc is None or getattr(fc, "kind", None) == "unknown":
        return max_depth
    if (getattr(fc, "kind", None) == LEDGE and direction == "down") or (
        getattr(fc, "kind", None) == LEDGE_LEFT and direction == "left"
    ):
        first = (start[0] + 2 * dx, start[1] + 2 * dy)
        if blocked_cells and first in blocked_cells:
            return 0
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
        cur_blocked_dirs = blocked_moves.get(cur, ()) if blocked_moves else ()
        for d_name, (nx_d, ny_d) in DIRECTIONS.items():
            if d_name in cur_blocked_dirs:
                continue
            nxt = (cur[0] + nx_d, cur[1] + ny_d)
            if not in_forward(nxt) or (blocked_cells and nxt in blocked_cells):
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
                if not in_forward(landing) or (blocked_cells and landing in blocked_cells):
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
    global_map: Optional[Any] = None,
) -> bool:
    if seg is not None:
        sc = seg.get((px + cell[0], py + cell[1]))
        if sc is not None and any(t in WARP_TAGS for t in getattr(sc, "tags", ())):
            return True
    if global_map is not None and _is_known_map_exit_cell(global_map, cell)[0]:
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
            if _is_warp_edge_cell(c, seg=seg, px=px, py=py, identified=identified, global_map=global_map):
                warps.add(c)
    return warps


def _straight_offscreen_depth(
    seg: Dict[Cell, Any],
    start: Cell,
    direction: str,
    max_depth: int = LOOKAHEAD_STEPS,
    blocked_cells: Optional[set] = None,
    blocked_moves: Optional[Dict[Cell, set]] = None,
) -> int:
    dx, dy = DIRECTIONS[direction]
    cur = start
    depth = 0
    while depth < max_depth:
        if blocked_moves and direction in blocked_moves.get(cur, ()):
            break
        nxt = (cur[0] + dx, cur[1] + dy)
        if blocked_cells and nxt in blocked_cells:
            break
        nc = seg.get(nxt)
        if nc is None or getattr(nc, "kind", None) == "unknown":
            return max_depth
        nkind = getattr(nc, "kind", None)
        if "entrance" in getattr(nc, "tags", ()) or "staircase" in getattr(nc, "tags", ()):
            return max_depth
        if nkind == WALKABLE:
            cur = nxt
            depth += 1
        elif (nkind == LEDGE and direction == "down") or (nkind == LEDGE_LEFT and direction == "left"):
            landing = (nxt[0] + dx, nxt[1] + dy)
            if blocked_cells and landing in blocked_cells:
                break
            lc = seg.get(landing)
            if lc is None or getattr(lc, "kind", None) == "unknown":
                return max_depth
            if getattr(lc, "kind", None) != WALKABLE:
                break
            cur = landing
            depth += 2
        else:
            break
    return min(depth, max_depth)


def run_pocket_bounds(
    global_map: Optional[Any],
    direction: str,
    run_cells: List[Cell],
) -> Optional[Tuple[set, set]]:
    if global_map is None or not getattr(global_map, "current", None) or getattr(global_map, "pos", None) is None or not run_cells:
        return None
    try:
        _, (px, py) = global_map.resolve(global_map.segment, global_map.pos)
        seg = global_map.current
    except Exception:  # noqa: BLE001
        return None
    gx_min, gx_max = px + min(X_RANGE), px + max(X_RANGE)
    gy_min, gy_max = py + min(Y_RANGE), py + max(Y_RANGE)

    cur_seg = global_map.resolve(global_map.segment, (0, 0))[0]
    door_leads: Dict[Cell, Optional[int]] = {}
    if hasattr(global_map, "reachable_door_list"):
        try:
            for d in global_map.reachable_door_list(cur_seg):
                for cell in d.get("cells", ()):
                    door_leads[cell] = d.get("leads_to")
        except Exception:  # noqa: BLE001
            door_leads = {}

    def _is_door_cell(g: Cell, c: Any) -> bool:
        if c is None or not bool(set(getattr(c, "tags", ())) & (WARP_TAGS | {"elevator"})):
            return False
        if g not in door_leads:
            return True
        leads = door_leads[g]
        if leads is None:
            return True
        try:
            if getattr(global_map.info(leads), "explored", None) is not None and not global_map.frontier(leads):
                return False
            if hasattr(global_map, "branch_exploration") and global_map.branch_exploration(cur_seg, leads).get("exhausted"):
                return False
        except Exception:  # noqa: BLE001
            return True
        return True

    def in_screen(g: Cell) -> bool:
        return gx_min <= g[0] <= gx_max and gy_min <= g[1] <= gy_max

    def in_half_plane(g: Cell) -> bool:
        if direction == "up":
            return g[1] >= gy_max
        if direction == "down":
            return g[1] <= gy_min
        if direction == "left":
            return g[0] <= gx_min
        return g[0] >= gx_max

    def strictly_beyond(g: Cell) -> bool:
        if direction == "up":
            return g[1] > gy_max
        if direction == "down":
            return g[1] < gy_min
        if direction == "left":
            return g[0] < gx_min
        return g[0] > gx_max

    seeds = []
    for lx, ly in run_cells:
        eg = (px + lx, py + ly)
        c = seg.get(eg)
        if c is None or getattr(c, "kind", None) == "unknown" or _is_door_cell(eg, c):
            return None
        if getattr(c, "kind", None) == WALKABLE:
            seeds.append(eg)
    if not seeds:
        return None

    visited = set(seeds)
    queue = deque(seeds)
    beyond_cells = set()
    half_plane_open = False

    while queue:
        cur = queue.popleft()
        if strictly_beyond(cur):
            beyond_cells.add(cur)
        c = seg.get(cur)
        if c is not None and _is_door_cell(cur, c):
            half_plane_open = True
            break
        for _, (nx_d, ny_d) in DIRECTIONS.items():
            nxt = (cur[0] + nx_d, cur[1] + ny_d)
            if in_screen(nxt):
                if nxt not in seeds:
                    continue
            if not in_half_plane(nxt):
                nc = seg.get(nxt)
                if nc is None or getattr(nc, "kind", None) in (WALKABLE, "unknown") or _is_door_cell(nxt, nc):
                    half_plane_open = True
                    break
                continue
            nc = seg.get(nxt)
            if nc is None or getattr(nc, "kind", None) == "unknown" or _is_door_cell(nxt, nc):
                half_plane_open = True
                break
            if getattr(nc, "kind", None) == WALKABLE and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
                if len(visited) > 80:
                    half_plane_open = True
                    break
        if half_plane_open:
            break
    if not half_plane_open:
        return visited, beyond_cells

    for min_offset in (1, 2):
        def in_forward_offset(g: Cell, off: int = min_offset) -> bool:
            if direction == "up":
                return g[1] >= py + off
            if direction == "down":
                return g[1] <= py - off
            if direction == "left":
                return g[0] <= px - off
            return g[0] >= px + off

        visited2 = set(seeds)
        queue2 = deque(seeds)
        beyond2 = set()
        open2 = False
        while queue2:
            cur = queue2.popleft()
            if strictly_beyond(cur):
                beyond2.add(cur)
            c = seg.get(cur)
            if c is not None and _is_door_cell(cur, c):
                open2 = True
                break
            for _, (nx_d, ny_d) in DIRECTIONS.items():
                nxt = (cur[0] + nx_d, cur[1] + ny_d)
                if not in_forward_offset(nxt):
                    continue
                nc = seg.get(nxt)
                if nc is None or getattr(nc, "kind", None) == "unknown" or _is_door_cell(nxt, nc):
                    open2 = True
                    break
                if getattr(nc, "kind", None) == WALKABLE and nxt not in visited2:
                    visited2.add(nxt)
                    queue2.append(nxt)
                    if len(visited2) > 25:
                        open2 = True
                        break
            if open2:
                break
        if not open2:
            return visited2, beyond2
    return None


def edge_pocket_note(
    global_map: Optional[Any],
    direction: str,
    local_edge_cells: List[Cell],
) -> Optional[str]:
    if global_map is None or not getattr(global_map, "current", None) or getattr(global_map, "pos", None) is None or not local_edge_cells:
        return None
    try:
        _, (px, py) = global_map.resolve(global_map.segment, global_map.pos)
    except Exception:  # noqa: BLE001
        return None
    runs = edge_runs(local_edge_cells, direction)
    closed_runs = []
    open_runs = []
    visited_all: set = set()
    beyond_all: set = set()
    for r in runs:
        bounds = run_pocket_bounds(global_map, direction, r)
        if bounds is not None:
            closed_runs.append(r)
            visited_all |= bounds[0]
            beyond_all |= bounds[1]
        else:
            open_runs.append(r)
    if not closed_runs:
        return None
    xs = [c[0] - px for c in visited_all]
    ys = [c[1] - py for c in visited_all]
    if not open_runs:
        return (
            f" [mapped closed dead-end pocket beyond {direction} edge: "
            f"{len(beyond_all)} off-screen walkable tile(s), relative x in [{min(xs)}, {max(xs)}], "
            f"y in [{min(ys)}, {max(ys)}], no unexplored tiles or doors]"
        )
    closed_cells = [c for r in closed_runs for c in r]
    closed_str = ", ".join(f"({x}, {y})" for x, y in closed_cells)
    return (
        f" [dead-end alcove at {closed_str}: leads into a mapped closed pocket "
        f"(relative x in [{min(xs)}, {max(xs)}], y in [{min(ys)}, {max(ys)}], no unexplored tiles or doors)]"
    )


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

    if global_map is not None:
        open_run_cells = [
            c for r in edge_runs(cells_list, direction)
            if run_pocket_bounds(global_map, direction, r) is None
            for c in r
        ]
        if open_run_cells:
            cells_list = open_run_cells

    b_cells, b_moves = _gmap_blocked_sets(global_map) if seg is not None else (set(), {})
    non_warp = [
        c for c in cells_list
        if not _is_warp_edge_cell(c, seg=seg, px=px, py=py, identified=identified)
        and (
            seg is None
            or _offscreen_continuation_depth(
                seg, (px + c[0], py + c[1]), direction, 1, blocked_cells=b_cells, blocked_moves=b_moves
            )
            > 0
        )
    ]
    if non_warp:
        cells_list = non_warp

    if seg is not None:
        depths = {
            c: _offscreen_continuation_depth(
                seg, (px + c[0], py + c[1]), direction, LOOKAHEAD_STEPS, blocked_cells=b_cells, blocked_moves=b_moves
            )
            for c in cells_list
        }
        deep_clear = [c for c in cells_list if depths[c] >= LOOKAHEAD_STEPS]
        if deep_clear:
            cells_list = deep_clear
        else:
            max_d = max(depths.values(), default=0)
            if max_d > 0:
                cells_list = [c for c in cells_list if depths[c] == max_d]
        straight_depths = {
            c: _straight_offscreen_depth(
                seg, (px + c[0], py + c[1]), direction, LOOKAHEAD_STEPS, blocked_cells=b_cells, blocked_moves=b_moves
            )
            for c in cells_list
        }
        straight_deep = [c for c in cells_list if straight_depths[c] >= LOOKAHEAD_STEPS]
        if straight_deep:
            cells_list = straight_deep
        else:
            max_sd = max(straight_depths.values(), default=0)
            if max_sd > 0:
                cells_list = [c for c in cells_list if straight_depths[c] == max_sd]

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


def _is_known_map_exit_cell(global_map: Optional[Any], cell: Cell) -> Tuple[bool, Optional[str]]:
    if global_map is None or not getattr(global_map, "current", None) or getattr(global_map, "pos", None) is None:
        return False, None
    try:
        si, (px, py) = global_map.resolve(global_map.segment, global_map.pos)
        info = global_map.info(si)
        if getattr(info, "kind", None) == "outdoors":
            return False, None
        seg = global_map.segments[si]
        abs_cell = (px + cell[0], py + cell[1])
        labels = {p: b for (s, p), b in global_map.door_labels().items() if s == si and b != si}
        if not labels:
            return False, None
        matched_warp = abs_cell in labels
        if not matched_warp:
            mc = seg.get(abs_cell)
            mc_hashes = getattr(mc, "hashes", None)
            if mc_hashes and None not in mc_hashes:
                for p in labels:
                    if abs(p[0] - abs_cell[0]) + abs(p[1] - abs_cell[1]) == 1:
                        pc = seg.get(p)
                        if pc is not None and getattr(pc, "hashes", None) == mc_hashes:
                            matched_warp = True
                            break
        if not matched_warp:
            return False, None
        pref_dir = None
        from execution.strategist.pokemon.global_map.routes import OPPOSITE
        for (_, a, pa, b, pb), w_dir in zip(getattr(global_map, "warps", ()), getattr(global_map, "warp_moves", ())):
            ra, rpa = global_map.resolve(a, pa)
            rb, rpb = global_map.resolve(b, pb)
            if ra == si and rb != si and abs(rpa[0] - abs_cell[0]) + abs(rpa[1] - abs_cell[1]) <= 1 and w_dir in DIRECTIONS:
                pref_dir = w_dir
                break
            if rb == si and ra != si and abs(rpb[0] - abs_cell[0]) + abs(rpb[1] - abs_cell[1]) <= 1 and w_dir in OPPOSITE:
                pref_dir = OPPOSITE[w_dir]
                break
        return True, pref_dir
    except Exception:  # noqa: BLE001
        return False, None


def boundary_exit_direction(
    kinds: Dict[Cell, str],
    identified: Optional[Dict[Tuple[int, int, str], object]],
    cell: Cell,
    preferred: Optional[str] = None,
    global_map: Optional[Any] = None,
) -> Optional[str]:
    from execution.perception.pokemon.tiles import filter_cell_warp_quarter_tags, normalize_tabletop_tag

    has_entrance_tag = False
    if identified is not None:
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
        has_entrance_tag = "entrance" in tags
    is_map_exit, map_pref_dir = _is_known_map_exit_cell(global_map, cell)
    if not has_entrance_tag and not is_map_exit:
        return None
    eff_preferred = preferred if preferred in DIRECTIONS else map_pref_dir
    order = ("down", "up", "left", "right")
    if eff_preferred in DIRECTIONS:
        order = (eff_preferred,) + tuple(d for d in order if d != eff_preferred)
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
                    prefer_lateral=True,
                    avoid_cells=warps,
                    people=people,
                )
                runs_text = format_path_runs(compress_path(lat_path or path))
                listed += f" (shortest path to {target}: {runs_text})"
            listed += edge_hazard_annotations(cells, kinds, people=people, direction=direction)
            note = edge_pocket_note(global_map, direction, cells)
            if note:
                listed += note
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
        if _is_npc_target_cell(target, identified, people) and (abs(target[0]) + abs(target[1])) >= 2:
            if target[0] == 0 or target[1] == 0:
                return (
                    " [separated by impassable water/obstacle — no adjacent walkable tile from here, "
                    "and player is already on the aligned tile (0, 0) without triggering a challenge; "
                    "cannot be triggered or interacted with by turning or pressing A across water — move on]"
                )
            return (
                " [separated by impassable water/obstacle — no adjacent walkable tile from here; "
                "if stepping onto the aligned walkway tile did not trigger a trainer challenge, "
                "cannot be triggered or interacted with from this path]"
            )
        return " [no walkable approach tile adjacent on current screen]"
    path = shortest_path(kinds, (0, 0), goals, people=people)
    if path is not None:
        if not path:
            if abs(target[0]) + abs(target[1]) == 2:
                return " [adjacent across counter]"
            return " [adjacent to player]"
        runs_text = format_path_runs(compress_path(path))
        end_cell = path[-1][2]
        across_note = " (across counter)" if abs(target[0] - end_cell[0]) + abs(target[1] - end_cell[1]) == 2 else ""
        return f" [on-screen path: {runs_text}{across_note}]"
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
