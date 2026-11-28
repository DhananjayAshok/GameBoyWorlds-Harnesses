import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.matching import cell_key, shade_bank, shade_candidates, split_quarters
from execution.perception.pokemon.battle import (BANNER_TOP, TILE_SIZE, find_modal_preview_boxes,
                                                 has_bottom_dialogue_box, has_bottom_location_banner)
from execution.perception.pokemon.grid import GRID_SKIP, cell_box, draw_box, split_grid
from execution.perception.pokemon.navigation import (FURNITURE_OVERHANG_TOP_TAGS, OVERHANG_TOP_TAGS,
                                                     WALKABLE, Y_RANGE,
                                                     approach_cells,
                                                     describe_target_reachability,
                                                     in_bounds,
                                                     infer_spatial_patterned_floor_cells,
                                                     is_counter_separator_cell,
                                                     is_east_cliff_barrier_cell, is_npc_head_overhang_floor_cell,
                                                     is_overhang_floor_cell, is_patterned_floor_cell,
                                                     is_player_overhang_floor_cell,
                                                     is_south_ledge_cell, is_top_overhang_floor_cell,
                                                     is_walkable_seating_cell,
                                                     is_west_ledge_cell, trainer_los_ray)
from execution.perception.pokemon.prompts import PLAYER_SIDES, TILE_IN_CELL_PROMPT, TILE_TAGS, fill_player_note
from utils import log_error, parse_key_value

TileKey = Tuple[int, int, str]

PLAYER_CELL = (0, 0)
QUARTER_OFFSETS = {
    "top_left": (0, 0),
    "top_right": (GRID_SKIP // 2, 0),
    "bottom_left": (0, GRID_SKIP // 2),
    "bottom_right": (GRID_SKIP // 2, GRID_SKIP // 2),
}


@dataclass
class Tile:
    image: Optional[np.ndarray]
    hash: Optional[str]
    tag: str
    description: Optional[str] = None


UNKNOWN_TILE = Tile(image=None, hash=None, tag="unknown", description=None)
PLAYER_TILE = Tile(image=None, hash=None, tag="player", description="the player")
NON_OBJECT_TAGS = ("ground", "ground decoration", "obstacle", "player", "unknown")
HALVES = {
    frozenset(("top_left", "top_right")): "top half",
    frozenset(("bottom_left", "bottom_right")): "bottom half",
    frozenset(("top_left", "bottom_left")): "left half",
    frozenset(("top_right", "bottom_right")): "right half",
}


def parse_tag(output: str) -> str:
    tag = (parse_key_value(output, "Category") or "unknown").strip().lower()
    return tag if tag in TILE_TAGS else "unknown"


def parse_description(output: str) -> Optional[str]:
    description = parse_key_value(output, "Description")
    if description is None or description.strip().lower() in ("none", ""):
        return None
    return description.strip().strip('"')


def cell_part(quarters) -> str:
    quarters = frozenset(quarters)
    if len(quarters) == 4:
        return ""
    if quarters in HALVES:
        return f" ({HALVES[quarters]})"
    names = [q.replace("_", "-") for q in ("top_left", "top_right", "bottom_left", "bottom_right") if q in quarters]
    return f" ({', '.join(names)} quarter{'s' if len(names) > 1 else ''})"


SIGN_OBJECT_TAGS = ("sign", "interactable object", "item")
WARP_QUARTER_TAGS = ("entrance", "staircase", "elevator")
REACHABILITY_OBJECT_TAGS = ("entrance", "staircase", "elevator", "interactable object", "item", "npc", "sign")
MIN_WARP_QUARTERS = 2
TABLETOP_OBJECT_PATTERN = re.compile(
    r"\b(?:table|desk|counter)\b.*\b(?:with|holding|containing)\b"
    r"(?!.*\b(?:potted\s+plant|plant|flower\s*pot|flowerpot|stool|stools|chair|chairs|cushion|cushions|seat|seats)\b)",
    re.IGNORECASE,
)
CUTTABLE_OBJECT_PATTERN = re.compile(
    r"\bcuttable\b",
    re.IGNORECASE,
)
WALKABLE_GROUND_OVERRIDE_PATTERN = re.compile(
    r"^(?:boundary\s+hedge|hedge\s+border|roadside\s+sign)$",
    re.IGNORECASE,
)
INDOOR_LADDER_SIGN_DESC_PATTERN = re.compile(
    r"^(?:wooden\s+sign(?:post)?|freestanding\s+sign(?:post)?|signpost)$",
    re.IGNORECASE,
)
TABLETOP_SURFACE_DESC_PATTERN = re.compile(
    r"\b(?:table|desk|counter)\b",
    re.IGNORECASE,
)
ITEM_BALL_DESC_PATTERN = re.compile(
    r"\b(?:ball|item)\b",
    re.IGNORECASE,
)
SPECKLED_GROUND_OBSTACLE_DESC_PATTERN = re.compile(
    r"\b(?:rocky|cave|cliff|plateau|stone|ground|floor|indoor|interior|wall|brick|mat|tile|tiled)\b",
    re.IGNORECASE,
)
ROCKY_PLATEAU_TOP_EDGE_DESC_PATTERN = re.compile(
    r"\b(?:plateau|cliff|ridge)\b",
    re.IGNORECASE,
)
STAIR_OR_STEP_DESC_PATTERN = re.compile(
    r"\b(?:step|steps|stair|stairs|ladder)\b",
    re.IGNORECASE,
)


def is_solid_shade_piece(image: Optional[np.ndarray]) -> bool:
    return image is not None and image.size > 0 and int(image.min()) == int(image.max())


def is_speckled_ground_quarter(
    tag: str, description: Optional[str], image: Optional[np.ndarray] = None
) -> bool:
    if tag != "obstacle" or not description or image is None or image.size == 0:
        return False
    if TABLETOP_SURFACE_DESC_PATTERN.search(description):
        return False
    arr = np.squeeze(np.asarray(image))
    if arr.shape != (8, 8):
        return False
    vals, counts = np.unique(arr, return_counts=True)
    min_s, max_s = int(arr.min()), int(arr.max())
    dom = float(counts.max()) / arr.size
    row_means = [float(arr[r, :].mean()) for r in range(8)]
    col_means = [float(arr[:, c].mean()) for c in range(8)]
    row_span = max(row_means) - min(row_means)
    col_span = max(col_means) - min(col_means)
    if (
        len(vals) == 2
        and min_s >= 200
        and max_s >= 240
        and 0.45 <= dom <= 0.85
        and row_span <= 30.0
        and col_span <= 24.0
    ):
        return True
    if not SPECKLED_GROUND_OBSTACLE_DESC_PATTERN.search(description):
        return False
    if len(vals) < 3:
        return False
    if min_s < 60 or (max_s - min_s) > 72:
        return False
    if not (0.50 <= dom <= 0.85):
        return False
    if row_span <= 24.0 and col_span <= 24.0:
        return True
    if len(vals) == 3 and int(counts[0]) <= 10 and (int(counts[1]) + int(counts[2])) >= 54:
        return row_span <= 42.0 and col_span <= 42.0
    return False


def is_rocky_plateau_top_edge_quarter(
    tag: str, description: Optional[str], image: Optional[np.ndarray] = None
) -> bool:
    if tag != "ground" or not description or image is None or image.size == 0:
        return False
    if not ROCKY_PLATEAU_TOP_EDGE_DESC_PATTERN.search(description):
        return False
    if STAIR_OR_STEP_DESC_PATTERN.search(description):
        return False
    arr = np.squeeze(np.asarray(image))
    if arr.shape != (8, 8):
        return False
    return int(arr[2:, :].min()) - int(arr[:2, :].min()) >= 80


def is_indoor_floor_ladder_quarter(
    tag: str, description: Optional[str], image: Optional[np.ndarray] = None
) -> bool:
    if tag != "sign" or not description or image is None or image.size == 0:
        return False
    if not INDOOR_LADDER_SIGN_DESC_PATTERN.search(description.strip()):
        return False
    arr = np.squeeze(np.asarray(image))
    if arr.shape != (8, 8):
        return False
    if int(arr.min()) < 56 or int(arr.max()) > 216:
        return False
    return not bool(np.any((arr > 60) & (arr < 150)))


def is_overworld_item_ball_quarter(
    tag: str, description: Optional[str], image: Optional[np.ndarray] = None
) -> bool:
    if image is None or image.size == 0:
        return False
    arr = np.squeeze(np.asarray(image))
    if arr.shape != (8, 8):
        return False
    if description and TABLETOP_SURFACE_DESC_PATTERN.search(description):
        return False

    def _top_left_dome(a: np.ndarray) -> bool:
        border = ((0, 3), (1, 3), (2, 2), (3, 2), (4, 2), (5, 2), (6, 3), (7, 3))
        if not all(int(a[r, c]) == 0 for r, c in border):
            return False
        interior = [int(a[r, c]) for r in range(8) for c in range(4 if r in (0, 1, 6, 7) else 3, 8)]
        if not all(v >= 150 for v in interior):
            return False
        if not all(int(a[r, c]) >= 240 for r in (5, 6, 7) for c in range(4, 8)):
            return False
        bg = a[2:8, 0:2]
        if bool(np.all(bg == bg[0:1, :])):
            return False
        return True

    def _bottom_left_rim(a: np.ndarray) -> bool:
        if not (int(a[0, 4]) == 0 and int(a[0, 5]) == 0 and int(a[0, 6]) >= 240 and int(a[0, 7]) >= 240):
            return False
        if not (int(a[1, 6]) == 0 and int(a[1, 7]) == 0 and int(a[1, 4]) > 0):
            return False
        if not (int(a[2, 6]) > 0 and int(a[2, 7]) > 0):
            return False
        sub = a[2:8, 2:8]
        if bool(np.all(sub == sub[0:1, :])):
            return False
        if any(int(a[r, c]) == 0 and int(a[r, c + 1]) == 0 for r in range(2, 8) for c in range(2, 7)):
            return False
        return True

    return (
        _top_left_dome(arr)
        or _top_left_dome(arr[:, ::-1])
        or _bottom_left_rim(arr)
        or _bottom_left_rim(arr[:, ::-1])
    )


def normalize_tabletop_tag(tag: str, description: Optional[str], image: Optional[np.ndarray] = None) -> str:
    if tag in WARP_QUARTER_TAGS and is_solid_shade_piece(image):
        return "ground"
    if is_indoor_floor_ladder_quarter(tag, description, image):
        return "staircase"
    if is_overworld_item_ball_quarter(tag, description, image):
        return "item"
    if is_speckled_ground_quarter(tag, description, image):
        return "ground"
    if is_rocky_plateau_top_edge_quarter(tag, description, image):
        return "obstacle"
    if description:
        stripped = description.strip()
        if tag == "obstacle" and (
            TABLETOP_OBJECT_PATTERN.search(stripped) or CUTTABLE_OBJECT_PATTERN.search(stripped)
        ):
            return "interactable object"
        if tag in ("obstacle", "sign") and WALKABLE_GROUND_OVERRIDE_PATTERN.search(stripped):
            return "ground"
    return tag


def filter_cell_warp_quarter_tags(cell_quarter_tags: Dict[str, str]) -> Dict[str, str]:
    warp_counts: Dict[str, int] = {}
    for tag in cell_quarter_tags.values():
        if tag in WARP_QUARTER_TAGS:
            warp_counts[tag] = warp_counts.get(tag, 0) + 1
    if not any(count < MIN_WARP_QUARTERS for count in warp_counts.values()):
        return dict(cell_quarter_tags)
    return {
        quarter: ("ground" if tag in WARP_QUARTER_TAGS and warp_counts.get(tag, 0) < MIN_WARP_QUARTERS else tag)
        for quarter, tag in cell_quarter_tags.items()
    }


def _quarter_tag_map(
    identified: Dict[TileKey, Tile]
) -> Tuple[
    Dict[Tuple[int, int], Dict[str, str]],
    Dict[Tuple[int, int], Dict[str, Optional[str]]],
    Dict[Tuple[int, int], Dict[str, Optional[np.ndarray]]],
]:
    quarter_tags: Dict[Tuple[int, int], Dict[str, str]] = {}
    quarter_descriptions: Dict[Tuple[int, int], Dict[str, Optional[str]]] = {}
    quarter_images: Dict[Tuple[int, int], Dict[str, Optional[np.ndarray]]] = {}
    for (x, y, quarter), tile in (identified or {}).items():
        img = getattr(tile, "image", None)
        quarter_tags.setdefault((x, y), {})[quarter] = normalize_tabletop_tag(
            tile.tag, tile.description, img
        )
        quarter_descriptions.setdefault((x, y), {})[quarter] = tile.description
        quarter_images.setdefault((x, y), {})[quarter] = img
    for cell in list(quarter_tags):
        quarter_tags[cell] = filter_cell_warp_quarter_tags(quarter_tags[cell])
    return quarter_tags, quarter_descriptions, quarter_images


def verbalize_tiles(
    identified: Dict[TileKey, Tile],
    people: Optional[Dict[Tuple[int, int], str]] = None,
    kinds: Optional[Dict[Tuple[int, int], str]] = None,
) -> str:
    quarter_tags, quarter_descriptions, quarter_images = _quarter_tag_map(identified)
    top_overhang_cells = {
        cell
        for cell in quarter_tags
        if is_top_overhang_floor_cell(cell, quarter_tags, quarter_descriptions, quarter_images)
    }
    player_overhang_cells = {
        cell for cell in quarter_tags if is_player_overhang_floor_cell(cell, quarter_tags, quarter_descriptions)
    }
    npc_head_overhang_cells = {
        cell
        for cell in quarter_tags
        if is_npc_head_overhang_floor_cell(cell, quarter_tags, quarter_images=quarter_images)
    }
    seating_floor_cells = {
        cell
        for cell in quarter_tags
        if is_walkable_seating_cell(cell, quarter_tags, quarter_descriptions)
    }
    south_ledge_cells = {
        cell for cell in quarter_tags if is_south_ledge_cell(cell, quarter_tags, quarter_descriptions)
    }
    west_ledge_cells = {
        cell for cell in quarter_tags if is_west_ledge_cell(cell, quarter_tags, quarter_descriptions)
    }
    east_cliff_cells = {
        cell
        for cell in quarter_tags
        if is_east_cliff_barrier_cell(
            cell, quarter_tags, quarter_descriptions, quarter_images, allow_player_cap=True
        )
    }
    by_cell: Dict[Tuple[int, int], Dict[str, Dict[str, Any]]] = {}
    for cell in south_ledge_cells:
        by_cell.setdefault(cell, {})["ledge"] = {
            "quarters": {"bottom_left", "bottom_right"},
            "descriptions": ["one-way ledge (jump down only; blocks moving up)"],
        }
    for cell in west_ledge_cells:
        by_cell.setdefault(cell, {})["ledge"] = {
            "quarters": {"top_left", "bottom_left"},
            "descriptions": ["one-way vertical ledge (jump left only; blocks moving right)"],
        }
    for cell in east_cliff_cells:
        by_cell.setdefault(cell, {})["obstacle"] = {
            "quarters": {"top_right", "bottom_right"},
            "descriptions": ["vertical cliff wall (blocks moving left or right)"],
        }
    for (x, y, quarter), tile in identified.items():
        if (x, y) in south_ledge_cells or (x, y) in west_ledge_cells or (x, y) in east_cliff_cells:
            continue
        tag = quarter_tags.get((x, y), {}).get(
            quarter, normalize_tabletop_tag(tile.tag, tile.description, getattr(tile, "image", None))
        )
        if (x, y) in top_overhang_cells and quarter in ("top_left", "top_right") and tag in FURNITURE_OVERHANG_TOP_TAGS:
            continue
        if (x, y) in player_overhang_cells and quarter in ("bottom_left", "bottom_right") and tag in OVERHANG_TOP_TAGS:
            continue
        if (x, y) in npc_head_overhang_cells and quarter in ("bottom_left", "bottom_right") and tag == "npc":
            continue
        if (x, y) in seating_floor_cells:
            continue
        if tag in NON_OBJECT_TAGS:
            continue
        entry = by_cell.setdefault((x, y), {}).setdefault(tag, {"quarters": set(), "descriptions": []})
        entry["quarters"].add(quarter)
        desc = (
            "Poke Ball on the ground"
            if tag == "item" and (not tile.description or not ITEM_BALL_DESC_PATTERN.search(tile.description))
            else tile.description
        )
        if desc and desc not in entry["descriptions"]:
            entry["descriptions"].append(desc)
    for cell, label in (people or {}).items():
        if not in_bounds(cell) and (
            kinds is None or not approach_cells(kinds, cell, identified=identified, people=people)
        ):
            continue
        if isinstance(label, str) and label.startswith("sign:"):
            sign_note = label[len("sign:"):].strip()
            if kinds is not None and kinds.get(cell) == WALKABLE and not any(
                tag in by_cell.get(cell, {}) for tag in SIGN_OBJECT_TAGS
            ):
                continue
            cell_entries = by_cell.setdefault(cell, {})
            matched = False
            for tag in SIGN_OBJECT_TAGS:
                if tag in cell_entries:
                    if sign_note not in cell_entries[tag]["descriptions"]:
                        cell_entries[tag]["descriptions"].append(sign_note)
                    matched = True
            if not matched:
                cell_entries["sign"] = {"quarters": set(QUARTER_OFFSETS), "descriptions": [sign_note]}
            continue
        by_cell.setdefault(cell, {})["npc"] = {"quarters": set(QUARTER_OFFSETS), "descriptions": [label]}
        above = by_cell.get((cell[0], cell[1] + 1), {})
        if ((cell[0], cell[1] + 1) not in people and "npc" in above
                and above["npc"]["quarters"] <= {"bottom_left", "bottom_right"}):
            del above["npc"]  # the top of that person's sprite, drawn above its square
    for cell in sorted(npc_head_overhang_cells):
        if cell[1] != min(Y_RANGE):
            continue
        off_cell = (cell[0], cell[1] - 1)
        if off_cell in by_cell:
            continue
        if kinds is not None and not approach_cells(kinds, off_cell, identified=identified, people=people):
            continue
        overhang_descs: List[str] = []
        for q in ("bottom_left", "bottom_right"):
            t = identified.get((cell[0], cell[1], q))
            if t is not None and getattr(t, "tag", None) == "npc" and t.description:
                if t.description not in overhang_descs:
                    overhang_descs.append(t.description)
        by_cell[off_cell] = {
            "npc": {
                "quarters": set(QUARTER_OFFSETS),
                "descriptions": overhang_descs or ["person just below the bottom edge of the screen"],
            }
        }
    top_counter_cell = (0, max(Y_RANGE))
    behind_top_counter_cell = (0, max(Y_RANGE) + 1)
    if (
        kinds is not None
        and top_counter_cell not in by_cell
        and behind_top_counter_cell not in by_cell
        and kinds.get((0, max(Y_RANGE) - 1)) == WALKABLE
        and is_counter_separator_cell(top_counter_cell, kinds, identified)
    ):
        counter_descs: List[str] = []
        counter_qs: set = set()
        for q in QUARTER_OFFSETS:
            t = identified.get((top_counter_cell[0], top_counter_cell[1], q))
            if t is not None and t.description and TABLETOP_SURFACE_DESC_PATTERN.search(t.description):
                counter_qs.add(q)
                if t.description not in counter_descs:
                    counter_descs.append(t.description)
        by_cell[top_counter_cell] = {
            "interactable object": {
                "quarters": counter_qs or set(QUARTER_OFFSETS),
                "descriptions": counter_descs or ["reception counter"],
            }
        }
    lines = ["You are at (0, 0). x increases to the right, y increases upward. Each (x, y) is one grid cell; "
             "ground and obstacles are not listed."]
    if not by_cell:
        lines.append("No objects, entrances or other notable things are visible.")
    for (x, y) in sorted(by_cell, key=lambda c: (-c[1], c[0])):
        for tag, entry in by_cell[(x, y)].items():
            details = f": {'; '.join(entry['descriptions'])}" if entry["descriptions"] else ""
            reach = (
                describe_target_reachability(kinds, (x, y), identified=identified, people=people)
                if tag in REACHABILITY_OBJECT_TAGS
                else ""
            )
            los_note = ""
            if tag == "npc" and people and (x, y) in people:
                facing, ray = trainer_los_ray(kinds, (x, y), people.get((x, y)))
                if facing and ray:
                    ray_str = ", ".join(f"({rx}, {ry})" for rx, ry in ray)
                    los_note = f" [line of sight ({facing}): {ray_str}]"
            lines.append(f"({x}, {y}){cell_part(entry['quarters'])} {tag}{details}{reach}{los_note}")
    return "\n".join(lines)


STATE_SCREEN_DESCRIPTIONS = {
    "in_dialogue": (
        "Game state: in_dialogue (a dialogue box is open on screen; "
        "overworld grid perception is not applicable until returned to free_roam)."
    ),
    "in_menu": (
        "Game state: in_menu (a menu, choice box, or info overlay is open on screen; "
        "overworld grid perception is not applicable until returned to free_roam)."
    ),
    "in_battle": (
        "Game state: in_battle (a battle screen is open; "
        "overworld grid perception is not applicable until returned to free_roam)."
    ),
}


def describe_state_screen(state: Any) -> str:
    name = getattr(state, "name", str(state)).lower()
    return STATE_SCREEN_DESCRIPTIONS.get(
        name,
        f"Game state: {name} (overworld grid perception is not applicable until returned to free_roam).",
    )


PLAYER_NEIGHBOURS = {
    (-1, 1): "diagonally up and to the left of",
    (0, 1): "directly above",
    (1, 1): "diagonally up and to the right of",
    (-1, 0): "directly to the left of",
    (1, 0): "directly to the right of",
    (-1, -1): "diagonally down and to the left of",
    (0, -1): "directly below",
    (1, -1): "diagonally down and to the right of",
}
COUNTER_NEIGHBOURS = {
    (0, 2): "two cells above (across a 1-tile counter/desk directly above)",
    (0, -2): "two cells below (across a 1-tile counter/desk directly below)",
    (-2, 0): "two cells to the left (across a 1-tile counter/desk directly to the left)",
    (2, 0): "two cells to the right (across a 1-tile counter/desk directly to the right)",
}


def verbalize_cell(identified: Dict[TileKey, Tile], cell: Tuple[int, int]) -> str:
    quarter_tags, quarter_descriptions, quarter_images = _quarter_tag_map(identified)
    if is_south_ledge_cell(cell, quarter_tags, quarter_descriptions):
        return "ledge (one-way ledge, jump down only; blocks moving up) in the bottom half"
    if is_west_ledge_cell(cell, quarter_tags, quarter_descriptions):
        return "ledge (one-way vertical ledge, jump left only; blocks moving right) in the left half"
    if is_east_cliff_barrier_cell(
        cell, quarter_tags, quarter_descriptions, quarter_images, allow_player_cap=True
    ):
        return "obstacle (vertical cliff wall; blocks moving left or right) in the right half"
    is_top_overhang = is_top_overhang_floor_cell(cell, quarter_tags, quarter_descriptions, quarter_images)
    is_player_overhang = is_player_overhang_floor_cell(cell, quarter_tags, quarter_descriptions)
    is_npc_head_overhang = is_npc_head_overhang_floor_cell(cell, quarter_tags, quarter_images=quarter_images)
    patterned_cells = infer_spatial_patterned_floor_cells(
        quarter_tags, quarter_descriptions, quarter_images
    )
    is_patterned_floor = cell in patterned_cells or is_patterned_floor_cell(
        cell, quarter_tags, quarter_descriptions, quarter_images
    )
    groups: Dict[Tuple[str, Optional[str]], set] = {}
    for quarter in QUARTER_OFFSETS:
        tile = identified.get((cell[0], cell[1], quarter))
        if tile is not None:
            tag = quarter_tags.get(cell, {}).get(
                quarter, normalize_tabletop_tag(tile.tag, tile.description, getattr(tile, "image", None))
            )
            if is_top_overhang and quarter in ("top_left", "top_right") and tag in FURNITURE_OVERHANG_TOP_TAGS:
                continue
            if is_player_overhang and quarter in ("bottom_left", "bottom_right") and tag in OVERHANG_TOP_TAGS:
                continue
            if is_npc_head_overhang and quarter in ("bottom_left", "bottom_right") and tag == "npc":
                continue
            norm_solid_ground = False
            if is_patterned_floor and tag == "obstacle":
                tag = "ground"
                norm_solid_ground = is_solid_shade_piece(getattr(tile, "image", None))
            desc = (
                "plain ground"
                if norm_solid_ground
                else (
                    "Poke Ball on the ground"
                    if tag == "item" and (not tile.description or not ITEM_BALL_DESC_PATTERN.search(tile.description))
                    else tile.description
                )
            )
            groups.setdefault((tag, desc), set()).add(quarter)
    parts = []
    for (tag, description), quarters in groups.items():
        where = cell_part(quarters).strip().strip("()")
        parts.append(tag + (f" ({description})" if description else "") + (f" in the {where}" if where else ""))
    return "; ".join(parts) if parts else "nothing recognised"


def verbalize_neighbourhood(identified: Dict[TileKey, Tile]) -> str:
    lines = [
        f"- ({x}, {y}), {position} the player: {verbalize_cell(identified, (x, y))}"
        for (x, y), position in PLAYER_NEIGHBOURS.items()
    ]
    for (x, y), position in COUNTER_NEIGHBOURS.items():
        cell_text = verbalize_cell(identified, (x, y))
        if "npc" in cell_text:
            lines.append(f"- ({x}, {y}), {position} the player: {cell_text}")
    return "\n".join(lines)


class TileRecognizer(StrategistArtifact):
    # Not picklable: callers must call set_vlm_call() again after load(). The shade caches are rebuilt on demand.
    _TRANSIENT = ("vlm_call", "_shade_bank", "_shade_bank_size", "_shade_aliases", "_shade_misses")
    # Fewest shades both pieces need for a shade match (see shade_candidates).
    SHADE_MIN_SHADES = 3
    # Once this many pieces on a screen matched by shade, the screen is recoloured and fewer shades suffice.
    SHADE_SCREEN_EVIDENCE = 8
    SHADE_RECOLOURED_MIN_SHADES = 2
    # On a recoloured screen, also accept recolourings that reorder shades (some night palettes swap light and dark).
    SHADE_RECOLOURED_UNORDERED = True

    def __init__(
        self,
        name: str,
        game: str = "pokemon_red",
        parameters: Optional[dict] = None,
        max_new_tokens: int = 4000,
    ):
        self.vlm_call: Optional[Callable[..., Any]] = None
        self.name = name
        self.game = game
        self.parameters = parameters
        self.max_new_tokens = max_new_tokens
        self.tiles: List[Tile] = []
        self._by_hash: Dict[str, Tile] = {}

    def _shade_match(self, tile_hash: Optional[str], piece: np.ndarray,
                     min_shades: Optional[int] = None, ordered: bool = True) -> Optional[Tile]:
        # Palettes can recolour the whole map (time of day), so an exact hash alone would miss known tiles.
        min_shades = self.SHADE_MIN_SHADES if min_shades is None else min_shades
        if tile_hash is None or not self.tiles:
            return None
        if getattr(self, "_shade_bank_size", None) != len(self.tiles):
            self._shade_bank = shade_bank([tile.image for tile in self.tiles])
            self._shade_bank_size, self._shade_misses = len(self.tiles), {}
        if getattr(self, "_shade_aliases", None) is None:
            self._shade_aliases = {}
        if tile_hash in self._shade_aliases:
            return self._shade_aliases[tile_hash]
        misses = self._shade_misses.setdefault((min_shades, ordered), set())
        if tile_hash in misses:
            return None
        candidates = [
            self.tiles[i]
            for i in shade_candidates(self._shade_bank, piece, min_shades, ordered)
            if self.tiles[i].tag != "unknown"
        ]
        if not candidates or len({tile.tag for tile in candidates}) != 1:
            misses.add(tile_hash)
            return None
        self._shade_aliases[tile_hash] = candidates[0]
        return candidates[0]

    def _water_shift_match(self, piece: np.ndarray) -> Optional[Tile]:
        if piece is None or piece.size == 0 or not self.tiles:
            return None
        arr = np.squeeze(np.asarray(piece))
        if arr.shape != (8, 8):
            return None
        for tile in self.tiles:
            if tile.tag != "water" or tile.image is None:
                continue
            warr = np.squeeze(np.asarray(tile.image))
            if warr.shape != (8, 8) or not any(len(np.unique(warr[r, :])) >= 2 for r in range(8)):
                continue
            for shift in range(1, 8):
                if np.array_equal(np.roll(warr, shift, axis=1), arr):
                    return tile
        return None

    def _reconcile_water_shift_tiles(self) -> None:
        water_tiles: List[Tuple[Tile, np.ndarray]] = []
        for tile in getattr(self, "tiles", ()):
            if tile.tag == "water" and tile.image is not None:
                warr = np.squeeze(np.asarray(tile.image))
                if warr.shape == (8, 8) and any(len(np.unique(warr[r, :])) >= 2 for r in range(8)):
                    water_tiles.append((tile, warr))
        if not water_tiles:
            return
        for tile in getattr(self, "tiles", ()):
            if tile.tag in ("ground", "ground decoration", "obstacle", "unknown") and tile.image is not None:
                arr = np.squeeze(np.asarray(tile.image))
                if arr.shape != (8, 8):
                    continue
                for wtile, warr in water_tiles:
                    if any(np.array_equal(np.roll(warr, shift, axis=1), arr) for shift in range(1, 8)):
                        tile.tag = "water"
                        tile.description = wtile.description or "water"
                        break

    def _resolve_screen(self, pieces: Dict[TileKey, np.ndarray]) -> Dict[TileKey, Optional[Tile]]:
        found: Dict[TileKey, Optional[Tile]] = {}
        pending: Dict[TileKey, Tuple[str, np.ndarray]] = {}
        for key, piece in pieces.items():
            tile_hash = cell_key(piece)
            tile = self._by_hash.get(tile_hash)
            found[key] = tile
            if tile is None:
                pending[key] = (tile_hash, piece)
        recoloured = set()
        for key, (tile_hash, piece) in pending.items():
            tile = self._shade_match(tile_hash, piece)
            if tile is not None:
                found[key] = tile
                recoloured.add(tile_hash)
            else:
                wtile = self._water_shift_match(piece)
                if wtile is not None:
                    found[key] = wtile
        if len(recoloured) >= self.SHADE_SCREEN_EVIDENCE:
            for key, (tile_hash, piece) in pending.items():
                if found[key] is None:
                    found[key] = self._shade_match(tile_hash, piece, self.SHADE_RECOLOURED_MIN_SHADES)
                if found[key] is None and self.SHADE_RECOLOURED_UNORDERED:
                    found[key] = self._shade_match(tile_hash, piece, self.SHADE_RECOLOURED_MIN_SHADES, ordered=False)
        return found

    def canonical_hash(self, piece: np.ndarray) -> Optional[str]:
        tile_hash = cell_key(piece)
        if tile_hash in self._by_hash:
            return tile_hash
        tile = self._shade_match(tile_hash, piece)
        return tile.hash if tile is not None else tile_hash

    def set_vlm_call(self, vlm_call: Callable[..., Any]) -> None:
        self.vlm_call = vlm_call

    def diff(self, old: "TileRecognizer") -> Dict[str, Any]:
        old_hashes = {tile.hash for tile in old.tiles}
        new_tiles = [tile for tile in self.tiles if tile.hash not in old_hashes]
        by_tag: Dict[str, int] = {}
        for tile in new_tiles:
            by_tag[tile.tag] = by_tag.get(tile.tag, 0) + 1
        return {"new_tiles": new_tiles, "new_tile_count_by_tag": by_tag}

    def _split(self, frame: np.ndarray) -> Dict[TileKey, np.ndarray]:
        pieces = {}
        for (x, y), cell in split_grid(frame).items():
            for quarter, piece in split_quarters(cell).items():
                pieces[(x, y, quarter)] = piece
        return pieces

    @staticmethod
    def overlay_tile_keys(frame: np.ndarray) -> set:
        if has_bottom_dialogue_box(frame):
            return {(x, y, q) for x in range(-4, 6) for y in (-4, -3, -2) for q in QUARTER_OFFSETS}
        masked = set()
        if has_bottom_location_banner(frame):
            for x in range(-4, 6):
                for y in range(-4, 5):
                    _, y0, _, _ = cell_box(x, y)
                    for q, (_, dy) in QUARTER_OFFSETS.items():
                        if y0 + dy + TILE_SIZE > BANNER_TOP:
                            masked.add((x, y, q))
        for top_ty, bot_ty, left_tx, right_tx in find_modal_preview_boxes(frame):
            box_x0, box_x1 = left_tx * TILE_SIZE, (right_tx + 1) * TILE_SIZE
            box_y0, box_y1 = top_ty * TILE_SIZE, (bot_ty + 1) * TILE_SIZE
            for x in range(-4, 6):
                for y in range(-4, 5):
                    x0, y0, _, _ = cell_box(x, y)
                    for q, (dx, dy) in QUARTER_OFFSETS.items():
                        qx0, qy0 = x0 + dx, y0 + dy
                        if qx0 + TILE_SIZE > box_x0 and qx0 < box_x1 and qy0 + TILE_SIZE > box_y0 and qy0 < box_y1:
                            masked.add((x, y, q))
        return masked

    def _backfill_tile_from_map_cell(self, map_cell: Any, quarter: str) -> Tile:
        from execution.perception.pokemon.navigation import BLOCKED, LEDGE, LEDGE_LEFT, QUARTERS, WALKABLE

        hashes = getattr(map_cell, "hashes", None) or ()
        if quarter in QUARTERS:
            qi = QUARTERS.index(quarter)
            if qi < len(hashes) and hashes[qi]:
                tile = self._by_hash.get(hashes[qi])
                if tile is not None and tile.tag != "unknown":
                    return tile
        tags = tuple(t for t in (getattr(map_cell, "tags", None) or ()) if t != "unknown")
        kind = getattr(map_cell, "kind", None)
        if "tall grass" in tags:
            return Tile(image=None, hash=None, tag="tall grass", description="patch of tall grass")
        if "entrance" in tags:
            return Tile(image=None, hash=None, tag="entrance", description="doorway")
        if "staircase" in tags:
            return Tile(image=None, hash=None, tag="staircase", description="staircase")
        if "item" in tags:
            return Tile(image=None, hash=None, tag="item", description="Poke Ball on the ground")
        if "sign" in tags:
            return Tile(image=None, hash=None, tag="sign", description="signpost")
        if "npc" in tags or kind == "occupied":
            return Tile(image=None, hash=None, tag="npc", description="person")
        if kind == LEDGE:
            tag = "ledge" if quarter in ("bottom_left", "bottom_right") else "ground"
            return Tile(image=None, hash=None, tag=tag, description=None)
        if kind == LEDGE_LEFT:
            tag = "ledge" if quarter in ("top_left", "bottom_left") else "ground"
            return Tile(image=None, hash=None, tag=tag, description=None)
        if kind == WALKABLE:
            tag = "ground decoration" if "ground decoration" in tags else "ground"
            return Tile(image=None, hash=None, tag=tag, description=None)
        if kind == BLOCKED:
            return Tile(image=None, hash=None, tag="obstacle", description=None)
        return UNKNOWN_TILE

    def backfill_masked_tiles(
        self,
        identified: Dict[TileKey, Tile],
        frame: np.ndarray,
        global_map: Optional[Any] = None,
    ) -> Dict[TileKey, Tile]:
        if (
            global_map is None
            or has_bottom_dialogue_box(frame)
            or find_modal_preview_boxes(frame)
            or not has_bottom_location_banner(frame)
        ):
            return identified
        seg = getattr(global_map, "current", None)
        pos = getattr(global_map, "pos", None)
        if not seg or pos is None or getattr(global_map, "faded", False):
            return identified
        px, py = pos
        masked = self.overlay_tile_keys(frame)
        if not masked:
            return identified
        out = dict(identified)
        for (x, y, quarter) in masked:
            if (x, y) == PLAYER_CELL:
                continue
            cur = out.get((x, y, quarter))
            if cur is not None and cur.tag != "unknown":
                continue
            map_cell = seg.get((px + x, py + y))
            if map_cell is None or getattr(map_cell, "kind", None) == "unknown":
                continue
            backfilled = self._backfill_tile_from_map_cell(map_cell, quarter)
            if backfilled.tag != "unknown":
                out[(x, y, quarter)] = backfilled
        return out

    def identify_tiles(self, frame: np.ndarray, global_map: Optional[Any] = None) -> Dict[TileKey, Tile]:
        all_pieces = self._split(frame)
        masked = self.overlay_tile_keys(frame)
        pieces = {key: piece for key, piece in all_pieces.items()
                  if key[:2] != PLAYER_CELL and key not in masked}
        identified = {key: tile or UNKNOWN_TILE for key, tile in self._resolve_screen(pieces).items()}
        for key in masked:
            if key[:2] != PLAYER_CELL and key in all_pieces:
                identified[key] = UNKNOWN_TILE
        for key in all_pieces:
            if key[:2] == PLAYER_CELL:
                identified[key] = PLAYER_TILE
        if global_map is not None:
            identified = self.backfill_masked_tiles(identified, frame, global_map)
        return identified

    def record_tiles(self, frame: np.ndarray, episode_number: int) -> None:
        if has_bottom_dialogue_box(frame) or find_modal_preview_boxes(frame):
            return
        pieces = self._split(frame)
        masked = self.overlay_tile_keys(frame)
        resolved = self._resolve_screen({key: piece for key, piece in pieces.items()
                                         if key[:2] != PLAYER_CELL and key not in masked})
        new_groups: Dict[str, List[TileKey]] = {}
        for key, tile in resolved.items():
            if tile is None:
                new_groups.setdefault(cell_key(pieces[key]), []).append(key)
        if not new_groups:
            return
        if self.vlm_call is None:
            log_error(f"Tile recognizer {self.name!r} has {len(new_groups)} new tiles to classify but no vlm_call. "
                      "Call set_vlm_call first.", self.parameters)

        cells = split_grid(frame)
        hashes, texts, images = [], [], []
        for tile_hash, members in new_groups.items():
            x, y, quarter = next((m for m in members if m[:2] not in PLAYER_SIDES), members[0])
            x0, y0, _, _ = cell_box(x, y)
            dx, dy = QUARTER_OFFSETS[quarter]
            left, top = x0 + dx, y0 + dy
            size = GRID_SKIP // 2
            hashes.append(tile_hash)
            texts.append(fill_player_note(TILE_IN_CELL_PROMPT.replace("[QUARTER]", quarter.replace("_", "-")), (x, y)))
            images.append([
                draw_box(frame, left, top, left + size - 1, top + size - 1),
                pieces[(x, y, quarter)],
                cells[(x, y)],
            ])

        outputs = self.vlm_call(texts=texts, images=images, max_new_tokens=self.max_new_tokens)
        self._add_tiles([
            Tile(image=np.array(pieces[new_groups[tile_hash][0]]), hash=tile_hash,
                 tag=normalize_tabletop_tag(
                     parse_tag(output), parse_description(output), pieces[new_groups[tile_hash][0]]
                 ),
                 description=parse_description(output))
            for tile_hash, output in zip(hashes, outputs)
        ], episode_number=episode_number)

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for tile in getattr(self, "tiles", ()):
            new_tag = normalize_tabletop_tag(tile.tag, tile.description, getattr(tile, "image", None))
            if tile.tag == "sign" and new_tag == "staircase":
                tile.description = "indoor floor ladder"
            elif new_tag == "item" and (not tile.description or not ITEM_BALL_DESC_PATTERN.search(tile.description)):
                tile.description = "Poke Ball on the ground"
            tile.tag = new_tag
        self._reconcile_water_shift_tiles()

    @update
    def _add_tiles(self, tiles: List[Tile]) -> None:
        for tile in tiles:
            new_tag = normalize_tabletop_tag(tile.tag, tile.description, getattr(tile, "image", None))
            if tile.tag == "sign" and new_tag == "staircase":
                tile.description = "indoor floor ladder"
            elif new_tag == "item" and (not tile.description or not ITEM_BALL_DESC_PATTERN.search(tile.description)):
                tile.description = "Poke Ball on the ground"
            tile.tag = new_tag
            self.tiles.append(tile)
            self._by_hash[tile.hash] = tile
        self._reconcile_water_shift_tiles()
