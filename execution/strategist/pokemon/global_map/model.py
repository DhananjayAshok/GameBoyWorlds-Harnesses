from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.matching import cell_key, shade_bank, shade_candidates, split_quarters
from execution.perception.pokemon.battle import (
    BANNER_TOP,
    has_bottom_dialogue_box,
    has_bottom_location_banner,
    has_modal_preview_box,
    is_battle_screen,
    is_wavy_scanline_wipe_against,
)
from execution.perception.pokemon.grid import GRID_SKIP
from execution.perception.pokemon.navigation import (LEDGE_LEFT, LEDGE_TAGS, OVERHANG_TOP_TAGS, WALKABLE_TAGS,
                                                     classify_cell_from_quarters,
                                                     infer_spatial_patterned_floor_cells,
                                                     is_east_cliff_barrier_cell, is_npc_head_overhang_floor_cell,
                                                     is_overhang_floor_cell, is_patterned_floor_cell,
                                                     is_player_overhang_floor_cell,
                                                     is_player_overhang_ledge_cell, is_south_ledge_cell,
                                                     is_top_overhang_floor_cell,
                                                     is_walkable_pattern_learnable, is_west_ledge_cell)
from execution.perception.pokemon.tiles import (TileRecognizer, filter_cell_warp_quarter_tags,
                                                normalize_tabletop_tag)

from .registration import (BLACK_LEVEL, _local_cells, _screen, _still_lit, grid_offset, grid_residual, has_structure,
                           pixel_diff, register)

Cell = Tuple[int, int]

QUARTERS = ("top_left", "top_right", "bottom_left", "bottom_right")
DIRECTIONS: Dict[str, Cell] = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}

WALKABLE, BLOCKED, LEDGE, UNKNOWN, OCCUPIED = "walkable", "blocked", "ledge", "unknown", "occupied"
# What kind of place an area is, as the strategist judged when naming it.
OUTDOORS, BUILDING, CAVE = "outdoors", "building", "cave"
AREA_KINDS = (OUTDOORS, BUILDING, CAVE)

BLACKOUT_CELLS = 3
WHITE_LEVEL = 240
DARK_LEVEL = 100
BRIGHT_LEVEL = 150
TRANSITION_FLASH_CELLS = 40
IRIS_WIPE_CELLS = 30
MIN_MID_TONE_CELLS = 10
MAX_STEP_FILL = 8
# A cell counts as unchanged by a wipe if at most this fraction of its pixels changed.
WIPE_CELL_CHANGE = 0.2
# A wipe leaves at least this many unchanged cells for every changed one; a new room changes nearly all.
WIPE_SAME_RATIO = 2
# Re-localisation needs this many matching cells (fewer on a screen with little content).
RELOCALISE_MIN_CELLS = 30
# A cell picture votes for at most this many places it appears in a segment.
RELOCALISE_MAX_FANOUT = 25
# Re-localisation needs this fraction of the screen cells that land on mapped cells to match.
RELOCALISE_MIN_FRACTION = 0.7
# This many of the best-voted placements are checked cell by cell before one is accepted.
RELOCALISE_CANDIDATES = 5
# Placements matching within this fraction of the best are a close call.
RELOCALISE_CLOSE_CALL = 0.8
# A close call is settled only by a placement matching this many more distinctive cells than every other.
DISTINCTIVE_LEAD = 2
# A cell picture is distinctive if it appears at most this many times in the segment.
DISTINCTIVE_MAX_COPIES = 2
# Two ends of door links this close (in squares, either way) count as the same door.
DOOR_REACH = 1
# A non-door warp endpoint snaps to a unique nearby door component within this radius.
DOOR_SNAP_REACH = 4
# Re-localisation onto a known warp destination accepts this overall fraction when the core around the door matches.
RELOCALISE_WARP_MIN_FRACTION = 0.6
RELOCALISE_WARP_CORE_RADIUS = 2
# A verdict on merging two differently named areas stands until their overlap grows this many times larger.
MERGE_REJUDGE_GROWTH = 1.5
# Consecutive unplaceable free-roam frames before a new segment is opened.
LOST_PATIENCE = 3
# A segment younger than this many placements keeps checking whether it overlaps an older one.
MERGE_WINDOW = 40
# Two segments are merged at the end of an episode only if they share this many matching cells.
CONSOLIDATE_MIN_CELLS = 30
# Merging also needs at least this fraction of the overlap to match.
CONSOLIDATE_MIN_AGREEMENT = 0.6
# Merging also needs the best offset to have this many times the votes of any other.
CONSOLIDATE_MARGIN = 1.5
# Failing that margin (repeating scenery), the best offset needs at least this many votes.
CONSOLIDATE_MIN_VOTES = 10
# Failing that margin, the best offset's agreement must also beat the rival offset's by this much.
CONSOLIDATE_AGREEMENT_LEAD = 0.2
DOOR_TAGS = {"entrance", "staircase", "elevator"}
NEIGHBOURS_8 = tuple((ox, oy) for ox in (-1, 0, 1) for oy in (-1, 0, 1) if (ox, oy) != (0, 0))
# A segment is only named once the player has been placed in it this many times.
MIN_NAME_OBSERVATIONS = 3
# A segment is (re)named at the end of an episode once its map has grown by this many pixels: about two screens.
RENAME_GROWTH_PIXELS = 2 * 90 * 16 * 16


def is_door(c: "MapCell") -> bool:
    tags = set(c.tags)
    if not (tags & DOOR_TAGS) or "obstacle" in tags:
        return False
    if tags == {"staircase"} and c.pixels is not None and c.pixels.size == GRID_SKIP * GRID_SKIP:
        p = np.squeeze(c.pixels)
        if int((p <= BLACK_LEVEL).sum()) >= 110 and float((p == p[:, ::-1]).mean()) < 0.45 and float((p == p[::-1, :]).mean()) < 0.45:
            return False
    return True


_QUARTER_RANK_CACHE: Dict[bytes, bytes] = {}
_QUARTER_RECOLOUR_CACHE: Dict[Tuple[bytes, bytes], bool] = {}
_MAX_QUARTER_CACHE = 32768


def _quarter_rank_signature(pixels: Optional[np.ndarray]) -> Optional[Tuple[bytes, ...]]:
    if pixels is None or pixels.size != GRID_SKIP * GRID_SKIP:
        return None
    p = np.squeeze(pixels)
    if int(p.min()) == int(p.max()):
        return None
    if len(_QUARTER_RANK_CACHE) > _MAX_QUARTER_CACHE:
        _QUARTER_RANK_CACHE.clear()
    quarters = split_quarters(p)
    sigs = []
    for q in QUARTERS:
        qp = quarters[q]
        key = qp.tobytes()
        sig = _QUARTER_RANK_CACHE.get(key)
        if sig is None:
            u, idx = np.unique(qp.reshape(-1), return_inverse=True)
            if len(u) == 1:
                sig = b"D" if int(u[0]) <= BLACK_LEVEL else b"L"
            else:
                idx2d = idx.astype(np.uint8).reshape(qp.shape)
                min_roll = min(np.roll(idx2d, s, axis=1).tobytes() for s in range(idx2d.shape[1]))
                sig = bytes([len(u)]) + min_roll
            _QUARTER_RANK_CACHE[key] = sig
        sigs.append(sig)
    return tuple(sigs)


def _quarter_recolour_matches(
    h1: Optional[str], h2: Optional[str], p1: Optional[np.ndarray], p2: Optional[np.ndarray]
) -> bool:
    if h1 is not None and h1 == h2:
        return True
    if p1 is None or p2 is None:
        return False
    b1 = np.asarray(p1).tobytes()
    b2 = np.asarray(p2).tobytes()
    if b1 == b2:
        return True
    key = (b1, b2)
    cached = _QUARTER_RECOLOUR_CACHE.get(key)
    if cached is not None:
        return cached
    if len(_QUARTER_RECOLOUR_CACHE) > _MAX_QUARTER_CACHE:
        _QUARTER_RECOLOUR_CACHE.clear()
    a, b = np.asarray(p1).reshape(-1), np.asarray(p2).reshape(-1)
    u1, idx1 = np.unique(a, return_inverse=True)
    u2, idx2 = np.unique(b, return_inverse=True)
    if len(u1) == 1 and len(u2) == 1:
        res = (int(u1[0]) <= BLACK_LEVEL) == (int(u2[0]) <= BLACK_LEVEL)
    elif len(u1) < 2 or len(u2) < 2:
        res = False
    elif len(u1) == len(u2) and np.array_equal(idx1, idx2):
        res = True
    else:
        a2d = np.squeeze(np.asarray(p1))
        b2d = np.squeeze(np.asarray(p2))
        if (
            len(u1) == len(u2)
            and a2d.ndim == 2
            and a2d.shape == b2d.shape
            and any(
                np.array_equal(
                    np.roll(idx2.reshape(b2d.shape), s, axis=1),
                    idx1.reshape(a2d.shape),
                )
                for s in range(1, b2d.shape[1])
            )
        ):
            res = True
        else:
            bank = shade_bank([p2])
            res = len(shade_candidates(bank, p1, min_shades=2, ordered=True)) > 0
    _QUARTER_RECOLOUR_CACHE[key] = res
    _QUARTER_RECOLOUR_CACHE[(b2, b1)] = res
    return res


def _cells_match(
    hashes1: Tuple[Optional[str], ...],
    pixels1: Optional[np.ndarray],
    cell2: "MapCell",
) -> bool:
    if cell2.hashes == hashes1:
        return True
    if pixels1 is None or cell2.pixels is None:
        return False
    q1 = split_quarters(pixels1)
    q2 = split_quarters(cell2.pixels)
    return all(
        _quarter_recolour_matches(hashes1[i], cell2.hashes[i], q1[q], q2[q])
        for i, q in enumerate(QUARTERS)
    )


def classify(tags: Iterable[str]) -> str:
    tags = set(tags)
    if "npc" in tags:
        return OCCUPIED
    if tags <= WALKABLE_TAGS:
        return WALKABLE
    if tags <= WALKABLE_TAGS | LEDGE_TAGS:
        return LEDGE
    if "unknown" in tags and (tags - {"unknown"}) <= WALKABLE_TAGS | LEDGE_TAGS:
        return UNKNOWN
    return BLOCKED


@dataclass
class SegmentInfo:
    name: Optional[str] = None
    description: Optional[str] = None
    episode: Optional[int] = None
    step: Optional[int] = None
    aliases: List[Tuple[str, Optional[str]]] = field(default_factory=list)
    named_pixels: int = 0
    explored: Optional[int] = None
    kind: Optional[str] = None
    needs_merge_rename: bool = False


@dataclass
class MapCell:
    kind: str
    tags: Tuple[str, ...]
    hashes: Tuple[Optional[str], ...]
    first_seen: int
    last_seen: int
    pixels: Optional[np.ndarray] = None
    prior: Optional[Tuple[Any, ...]] = None


class GlobalMap(StrategistArtifact):
    """Segments of global grid, the player's place on them, and the bookkeeping to explain both."""

    # Set by MappingEnvironment to the tile recognizer's palette-independent hash; not saved.
    _TRANSIENT = ("canonical_hash", "_seg_index_cache", "_overlap_cache", "_npcs", "_tile_lookup")
    canonical_hash: Optional[Callable[[np.ndarray], Optional[str]]] = None
    _npcs: Optional[Any] = None
    _tile_lookup: Optional[Dict[str, Any]] = None

    def attach_npcs(self, npcs: Optional[Any]) -> None:
        self._npcs = npcs

    def cell_facility_sign_label(self, segment: Optional[int], cell: Cell) -> Optional[str]:
        lookup = getattr(self, "_tile_lookup", None)
        if not lookup:
            return None
        si = self.segment if segment is None else segment
        si = self.resolve(si, (0, 0))[0]
        seg = self.segments[si] if 0 <= si < len(self.segments) else {}
        c = seg.get(cell)
        if c is None or not c.hashes:
            return None
        descs = [
            (getattr(lookup.get(h), "description", "") or "").lower()
            for h in c.hashes
            if h and h in lookup
        ]
        if any("mart" in d for d in descs):
            return "Poké Mart sign"
        if any("pokemon center" in d or "pokémon center" in d or "poke center" in d for d in descs):
            return "Pokémon Center sign"
        if any("gym" in d for d in descs):
            return "Gym sign"
        return None

    def stationary_npc_squares(self, segment: Optional[int] = None) -> set:
        npcs = getattr(self, "_npcs", None)
        if npcs is None or not hasattr(npcs, "map_people"):
            return set()
        si = self.segment if segment is None else segment
        try:
            si = self.resolve(si, (0, 0))[0]
            seg = self.segments[si] if 0 <= si < len(self.segments) else {}
            visits = self.visits[si] if 0 <= si < len(self.visits) else {}
            cur_pos = (
                self.resolve(self.segment, self.pos)[1]
                if getattr(self, "pos", None) is not None and self.resolve(self.segment, self.pos)[0] == si
                else None
            )
            out: set = set()
            for mark in npcs.map_people(self, si):
                if getattr(mark, "stale", False):
                    continue
                earlier = getattr(mark, "earlier", None)
                if earlier and getattr(mark, "kind", None) != "trainer":
                    continue
                sq = getattr(mark, "square", None)
                if sq is None or sq == cur_pos or visits.get(sq, 0) > 0:
                    continue
                c = seg.get(sq)
                if c is not None and is_door(c):
                    continue
                out.add(sq)
            return out
        except Exception:  # noqa: BLE001
            return set()

    def _cell_hashes(self, quarters: Dict[str, np.ndarray]) -> Tuple[Optional[str], ...]:
        hash_of = self.canonical_hash or cell_key
        return tuple(hash_of(quarters[q]) for q in QUARTERS)

    def __init__(self) -> None:
        self.segments: List[Dict[Cell, MapCell]] = [{}]
        self.segment = 0
        self.pos: Cell = (0, 0)
        self.anchor: Optional[np.ndarray] = None
        self.residual: Cell = (0, 0)
        self.lost = 0
        self.faded = False
        self.in_battle = False
        self.battle_place: Optional[Tuple[int, Cell]] = None
        self.battle_start_step: int = 0
        self.since_battle_moved: bool = True
        self.merged_into: Dict[int, Tuple[int, Cell]] = {}
        self.pending: Optional[Tuple[str, int, Optional[int]]] = None
        self.step = 0
        self.visits: List[Counter] = [Counter()]
        self.trail: List[Tuple[int, int, Cell]] = []
        self.new_cells_by_episode: Dict[int, int] = {}
        self._new_this_episode = 0
        self.stats: Counter = Counter()
        self.exploring_until: Optional[int] = None
        self.segment_born = 0
        self.segment_born_id = 0
        self.warps: List[Tuple[int, int, Cell, int, Cell]] = []
        self.warp_moves: List[Optional[str]] = []
        self.fade_direction: Optional[str] = None
        self.routes: List[Dict[str, Any]] = []
        self.segment_info: Dict[int, SegmentInfo] = {0: SegmentInfo()}
        self.naming_queue: List[Dict[str, Any]] = []
        self.merge_verdicts: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.seen_entrances: Optional[set] = set()
        self.defeats: List[Dict[str, Any]] = []
        self.defeat_return: Optional[Dict[str, Any]] = None
        self.ledge_patterns: set = set()
        self.left_ledge_patterns: set = set()
        self.walk_patterns: set = set()
        self.blocked_cells: List[set] = [set()]
        self.blocked_moves: List[Dict[Cell, set]] = [{}]
        self.healed_shifts: List[Dict[str, Any]] = []

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self.__dict__.setdefault("defeats", [])
        self.__dict__.setdefault("defeat_return", None)
        self.__dict__.setdefault("seen_entrances", None)
        self.__dict__.setdefault("exploring_until", None)
        self.__dict__.setdefault("segment_born", 0)
        self.__dict__.setdefault("segment_born_id", len(self.segments) - 1)
        self.__dict__.setdefault("warps", [])
        self.__dict__.setdefault("warp_moves", [None] * len(self.warps))
        self.__dict__.setdefault("in_battle", False)
        self.__dict__.setdefault("battle_place", None)
        self.__dict__.setdefault("battle_start_step", 0)
        self.__dict__.setdefault("since_battle_moved", True)
        self.__dict__.setdefault("fade_direction", None)
        self.__dict__.setdefault("routes", [])
        self.__dict__.setdefault("segment_info", {})
        self.__dict__.setdefault("naming_queue", [])
        self.__dict__.setdefault("merge_verdicts", {})
        self.__dict__.setdefault("ledge_patterns", set())
        self.__dict__.setdefault("left_ledge_patterns", set())
        self.__dict__.setdefault("walk_patterns", set())
        self.__dict__.setdefault("blocked_cells", [set() for _ in self.segments])
        while len(self.blocked_cells) < len(self.segments):
            self.blocked_cells.append(set())
        self.__dict__.setdefault("blocked_moves", [{} for _ in self.segments])
        while len(self.blocked_moves) < len(self.segments):
            self.blocked_moves.append({})
        self.__dict__.setdefault("healed_shifts", [])
        self._heal_post_defeat_shifts()
        for si in range(len(self.segments)):
            self._heal_visited_ground(si)

    def _heal_visited_ground(self, si: int) -> int:
        seg = self.segments[si] if 0 <= si < len(self.segments) else None
        if not seg or si >= len(self.visits):
            return 0
        seg_blocked = self.blocked_cells[si] if si < len(self.blocked_cells) else set()
        for _, t_si, t_pos in getattr(self, "trail", ()):
            rs, rp = self.resolve(t_si, t_pos)
            if rs != si or rp in seg_blocked or self.visits[si].get(rp, 0) > 0:
                continue
            mc = seg.get(rp)
            if mc is not None and not is_door(mc) and (
                mc.kind in (WALKABLE, UNKNOWN)
                or "ground" in mc.tags
                or self._can_learn_walkable(mc.tags, mc.hashes, mc.pixels)
            ):
                self.visits[si][rp] = 1
        if not self.visits[si]:
            return 0
        restored = 0
        for g, cnt in self.visits[si].items():
            if cnt <= 0 or g in seg_blocked:
                continue
            c = seg.get(g)
            if (
                c is not None
                and c.kind == BLOCKED
                and not is_door(c)
                and "ground" in c.tags
                and "obstacle" not in c.tags
                and self._can_learn_walkable(c.tags, c.hashes, c.pixels)
                and not (set(c.tags) & {"sign", "interactable object", "pc", "water", "ledge", "right_ledge", "left_ledge"})
            ):
                c.kind = WALKABLE
                c.tags = ("ground",)
                restored += 1
            elif (
                c is not None
                and c.kind == OCCUPIED
                and "ground" in c.tags
                and getattr(c, "prior", None) is not None
                and c.prior[0] == WALKABLE
            ):
                c.kind = WALKABLE
                c.tags = tuple(t for t in c.tags if t != "npc") or ("ground",)
                restored += 1
        if 0 <= si < len(getattr(self, "blocked_moves", ())):
            seg_moves = self.blocked_moves[si]
            for b_pos in list(seg_moves):
                kept_dirs = {
                    d for d in seg_moves[b_pos]
                    if d in DIRECTIONS
                    and not self._is_uniform_walk_pattern_cell(
                        seg.get((b_pos[0] + DIRECTIONS[d][0], b_pos[1] + DIRECTIONS[d][1]))
                    )
                }
                if kept_dirs:
                    seg_moves[b_pos] = kept_dirs
                else:
                    seg_moves.pop(b_pos, None)
        if self.info(si).kind != OUTDOORS:
            return restored
        seen_box = {
            (vx + lx, vy + ly)
            for (vx, vy) in self.visits[si]
            for lx in range(-4, 6)
            for ly in range(-4, 5)
        }
        for _ in range(2):
            added_pass = 0
            for g in sorted(seen_box):
                if g in seg or g in seg_blocked:
                    continue
                nbrs = [seg.get((g[0] + dx, g[1] + dy)) for dx, dy in DIRECTIONS.values()]
                walkable_nbrs = sum(1 for c in nbrs if c is not None and c.kind == WALKABLE)
                mapped_nbrs = sum(1 for c in nbrs if c is not None)
                if walkable_nbrs >= 2 and mapped_nbrs >= 3:
                    seg[g] = MapCell(WALKABLE, ("ground",), (None, None, None, None), self.step, self.step, None)
                    restored += 1
                    added_pass += 1
            if not added_pass:
                break
        return restored

    @staticmethod
    def _can_learn_walkable(
        tags: Iterable[str],
        hashes: Optional[Tuple[Optional[str], ...]] = None,
        pixels: Optional[np.ndarray] = None,
        quarter_tags: Optional[Dict[str, str]] = None,
        quarter_descriptions: Optional[Dict[str, Optional[str]]] = None,
        quarter_images: Optional[Dict[str, Optional[np.ndarray]]] = None,
    ) -> bool:
        return is_walkable_pattern_learnable(
            tags,
            hashes=hashes,
            pixels=pixels,
            quarter_tags=quarter_tags,
            quarter_descriptions=quarter_descriptions,
            quarter_images=quarter_images,
        )

    def _classify(self, tags: Iterable[str], hashes: Tuple[Optional[str], ...]) -> str:
        tag_set = set(tags)
        kind = classify(tag_set)
        if kind == BLOCKED and hashes in self.ledge_patterns:
            return LEDGE
        if kind == BLOCKED and hashes in self.left_ledge_patterns:
            return LEDGE_LEFT
        if kind == BLOCKED and hashes in self.walk_patterns and self._can_learn_walkable(tag_set, hashes):
            return WALKABLE
        return kind

    def _learn_walkable(self, cell: "MapCell") -> int:
        pattern = cell.hashes
        if any(h is None for h in pattern) or not self._can_learn_walkable(cell.tags, pattern, cell.pixels):
            return 0
        self.ledge_patterns.discard(pattern)
        self.left_ledge_patterns.discard(pattern)
        relabelled = 0
        if pattern not in self.walk_patterns:
            self.walk_patterns.add(pattern)
            for si_other, other in enumerate(self.segments):
                seg_blk = self.blocked_cells[si_other] if si_other < len(self.blocked_cells) else set()
                for g_other, c in other.items():
                    if g_other not in seg_blk and c.kind == BLOCKED and c.hashes == pattern:
                        c.kind = WALKABLE
                        relabelled += 1
            self.stats["walk_patterns_learned"] += 1
            self.stats["walk_cells_relabelled"] += relabelled
        elif cell.kind == BLOCKED:
            cell.kind = WALKABLE
            relabelled += 1
        return relabelled

    def _learn_ledges(self, start: Cell, end: Cell, direction: Optional[str]) -> int:
        # a move down or left across exactly one cell mapped as blocked went over a ledge
        seg = self.segments[self.segment]
        if direction == "down" and start[0] == end[0] and start[1] - end[1] >= 2:
            path = [seg.get((start[0], y)) for y in range(start[1] - 1, end[1] - 1, -1)]
            target_set, target_kind = self.ledge_patterns, LEDGE
        elif direction == "left" and start[1] == end[1] and start[0] - end[0] >= 2:
            path = [seg.get((x, start[1])) for x in range(start[0] - 1, end[0] - 1, -1)]
            target_set, target_kind = self.left_ledge_patterns, LEDGE_LEFT
        else:
            return 0
        blocked = [i for i, c in enumerate(path) if c is not None and c.kind == BLOCKED]
        # the hop lands on the cell after the ledge, so the ledge is never the last cell of the move
        if len(blocked) != 1 or blocked[0] == len(path) - 1:
            return 0
        pattern = path[blocked[0]].hashes
        if any(h is None for h in pattern) or pattern in target_set:
            return 0
        target_set.add(pattern)
        self.walk_patterns.discard(pattern)
        relabelled = 0
        for other in self.segments:
            for c in other.values():
                if c.kind == BLOCKED and c.hashes == pattern:
                    c.kind = target_kind
                    relabelled += 1
        self.stats["ledges_learned"] += 1
        self.stats["ledge_cells_relabelled"] += relabelled
        return relabelled

    def record_blocked_step(
        self,
        direction: Optional[str] = None,
        cell: Optional[Cell] = None,
        source_cell: Optional[Cell] = None,
    ) -> Optional[Cell]:
        if not self.current or self.pos is None:
            return None
        while len(self.blocked_cells) <= self.segment:
            self.blocked_cells.append(set())
        while len(self.blocked_moves) <= self.segment:
            self.blocked_moves.append({})
        src = self.pos if source_cell is None else source_cell
        if cell is None:
            if direction not in DIRECTIONS:
                return None
            dx, dy = DIRECTIONS[direction]
            target = (src[0] + dx, src[1] + dy)
        else:
            target = cell
            if direction not in DIRECTIONS and src is not None:
                for d_name, (dx, dy) in DIRECTIONS.items():
                    if (src[0] + dx, src[1] + dy) == target:
                        direction = d_name
                        break
        if target == self.pos and source_cell is None:
            return None
        seg = self.segments[self.segment]
        c = seg.get(target)
        if c is not None and (is_door(c) or c.kind in (LEDGE, LEDGE_LEFT)):
            return None
        if self._is_uniform_walk_pattern_cell(c):
            return None
        if direction in DIRECTIONS and src is not None and (c is None or c.kind != OCCUPIED):
            self.blocked_moves[self.segment].setdefault(src, set()).add(direction)
        if self.visits[self.segment].get(target, 0) > 0:
            if (
                c is None
                or not c.hashes
                or len(c.hashes) != 4
                or None in c.hashes
                or (c.hashes in self.walk_patterns and self._can_learn_walkable(c.tags, c.hashes, c.pixels))
            ):
                return None
            self.visits[self.segment].pop(target, None)
        self.blocked_cells[self.segment].add(target)
        if c is not None and c.kind == WALKABLE:
            c.kind = BLOCKED
            self.stats["empirical_blocked_cells"] += 1
        return target

    def clear_traversed_step(self, prev_pos: Cell, new_pos: Cell, segment: Optional[int] = None) -> None:
        si = self.segment if segment is None else segment
        if not (0 <= si < len(self.segments)):
            return
        dx, dy = new_pos[0] - prev_pos[0], new_pos[1] - prev_pos[1]
        if (dx == 0 and dy == 0) or ((dx == 0) == (dy == 0)):
            return
        steps = abs(dx) + abs(dy)
        if steps > MAX_STEP_FILL:
            return
        sx = (dx > 0) - (dx < 0)
        sy = (dy > 0) - (dy < 0)
        step_dir = next((d for d, delta in DIRECTIONS.items() if delta == (sx, sy)), None)
        seg_blocked = self.blocked_cells[si] if si < len(self.blocked_cells) else None
        seg_moves = self.blocked_moves[si] if si < len(self.blocked_moves) else None
        cur = prev_pos
        for _ in range(steps):
            if seg_moves and step_dir in DIRECTIONS and cur in seg_moves:
                seg_moves[cur].discard(step_dir)
                if not seg_moves[cur]:
                    seg_moves.pop(cur, None)
            cur = (cur[0] + sx, cur[1] + sy)
            if seg_blocked is not None:
                seg_blocked.discard(cur)

    def _clear_blocked_moves_to(self, segment: int, target: Cell) -> None:
        if not (0 <= segment < len(getattr(self, "blocked_moves", ()))):
            return
        seg_moves = self.blocked_moves[segment]
        for d_name, (dx, dy) in DIRECTIONS.items():
            src = (target[0] - dx, target[1] - dy)
            if src in seg_moves and d_name in seg_moves[src]:
                seg_moves[src].discard(d_name)
                if not seg_moves[src]:
                    seg_moves.pop(src, None)

    def _is_uniform_walk_pattern_cell(self, c: Optional["MapCell"]) -> bool:
        return bool(
            c is not None
            and c.hashes
            and len(c.hashes) == 4
            and len(set(c.hashes)) == 1
            and c.hashes[0] is not None
            and c.hashes in self.walk_patterns
            and self._can_learn_walkable(c.tags, c.hashes, c.pixels)
        )

    def segment_blocked_moves(
        self,
        segment: Optional[int] = None,
        extra_blocked_moves: Optional[Dict[Cell, Iterable[str]]] = None,
    ) -> Dict[Cell, set]:
        si = self.segment if segment is None else segment
        seg = self.segments[si] if 0 <= si < len(self.segments) else {}
        out: Dict[Cell, set] = {}
        if 0 <= si < len(getattr(self, "blocked_moves", ())):
            for b_pos, dirs in self.blocked_moves[si].items():
                valid_dirs = {
                    d for d in dirs
                    if d in DIRECTIONS
                    and not self._is_uniform_walk_pattern_cell(
                        seg.get((b_pos[0] + DIRECTIONS[d][0], b_pos[1] + DIRECTIONS[d][1]))
                    )
                }
                if valid_dirs:
                    out.setdefault(b_pos, set()).update(valid_dirs)
        if extra_blocked_moves:
            for b_pos, dirs in extra_blocked_moves.items():
                valid_dirs = {
                    d for d in dirs
                    if d in DIRECTIONS
                    and not self._is_uniform_walk_pattern_cell(
                        seg.get((b_pos[0] + DIRECTIONS[d][0], b_pos[1] + DIRECTIONS[d][1]))
                    )
                }
                if valid_dirs:
                    out.setdefault(b_pos, set()).update(valid_dirs)
        return out

    def local_blocked_cells(self) -> set:
        if not self.current or self.pos is None:
            return set()
        from .routes import blocked_target_cells

        px, py = self.pos
        out = set()
        if self.segment < len(self.blocked_cells):
            for gx, gy in self.blocked_cells[self.segment]:
                lx, ly = gx - px, gy - py
                if -4 <= lx <= 5 and -4 <= ly <= 4 and (lx, ly) != (0, 0):
                    out.add((lx, ly))
        seg_moves = self.segment_blocked_moves(self.segment)
        if seg_moves:
            for gx, gy in blocked_target_cells(self, self.segment, seg_moves):
                lx, ly = gx - px, gy - py
                if -4 <= lx <= 5 and -4 <= ly <= 4 and (lx, ly) != (0, 0):
                    out.add((lx, ly))
        return out

    def names_differ(self, a: int, b: int) -> bool:
        def norm(name: Optional[str]) -> Optional[str]:
            return None if name is None else " ".join(re.sub(r"[^\w\s]", "", name.lower()).split())
        na, nb = norm(self.info(a).name), norm(self.info(b).name)
        return na is not None and nb is not None and na != nb

    def info(self, segment: int) -> "SegmentInfo":
        return self.segment_info.setdefault(segment, SegmentInfo())

    def set_name(self, segment: int, name: Optional[str], description: Optional[str], kind: str,
                 episode: int) -> bool:
        info = self.info(segment)
        info.named_pixels = self.filled_pixels(segment)
        info.needs_merge_rename = False
        name, description = name or info.name, description or info.description
        if (name, description, kind) == (info.name, info.description, info.kind):
            return False
        if info.name is not None and name != info.name:
            info.aliases.append((info.name, info.description))
        info.name, info.description, info.kind, info.episode, info.step = name, description, kind, episode, self.step
        return True

    def parent(self, segment: int) -> Optional[int]:
        from .routes import _exits
        segment = self.resolve(segment, (0, 0))[0]
        if self.info(segment).kind != BUILDING:
            return None
        exits, seen, queue = _exits(self), {segment}, [segment]
        while queue:
            for e in exits.get(queue.pop(0), []):
                to = self.resolve(e["to"], (0, 0))[0]
                if to in seen:
                    continue
                seen.add(to)
                kind = self.info(to).kind
                if kind == OUTDOORS:
                    return to
                if kind == BUILDING:
                    queue.append(to)
        return None

    def place_label(self, segment: Optional[int] = None) -> str:
        segment = self.resolve(self.segment if segment is None else segment, (0, 0))[0]
        parent = self.parent(segment)
        return self.area_label(segment) + (f" (inside {self.area_label(parent)})" if parent is not None else "")

    def prune_routes(self) -> None:
        self.routes = [r for r in self.routes
                       if self.segment not in {self.resolve(t, (0, 0))[0] for t in r["targets"]}]

    def filled_pixels(self, segment: int) -> int:
        return sum(c.pixels.size for c in self.segments[segment].values() if c.pixels is not None)

    def clear_routes(self) -> None:
        self.routes = []

    def take_naming_queue(self) -> List[Dict[str, Any]]:
        queue, self.naming_queue = self.naming_queue, []
        return queue

    def count(self, stat: str, by: int = 1) -> None:
        self.stats[stat] += by

    def return_to_defeat(self, defeat: Dict[str, Any], episode: int) -> None:
        self.defeat_return = {"defeat": defeat, "episode": episode}

    def _visited_cells_count(self, segment: int) -> int:
        visits = self.visits[segment]
        if len(visits) <= 1 or len(visits) >= MIN_NAME_OBSERVATIONS:
            return len(visits)
        from .routes import walked_squares
        return max(len(visits), len(walked_squares(self, segment)))

    def _has_min_observations(self, segment: int) -> bool:
        return (
            sum(self.visits[segment].values()) >= MIN_NAME_OBSERVATIONS
            and self._visited_cells_count(segment) >= MIN_NAME_OBSERVATIONS
        )

    def queue_grown(self) -> None:
        queued = {entry["segment"] for entry in self.naming_queue}
        for si, seg in enumerate(self.segments):
            info = self.info(si)
            if (seg and si not in queued and si not in self.merged_into
                    and self._has_min_observations(si)
                    and (info.name is None or getattr(info, "needs_merge_rename", False))
                    and self.filled_pixels(si) - info.named_pixels >= RENAME_GROWTH_PIXELS):
                self.naming_queue.append({"segment": si, "exit": None, "step": self.step, "marker": None,
                                          "reason": "grew"})
                queued.add(si)
        if (self.current and self.segment not in queued and self.info(self.segment).name is None
                and self._has_min_observations(self.segment)):
            self.naming_queue.append({"segment": self.segment, "exit": None, "step": self.step, "marker": None,
                                      "reason": "here"})

    def mark_explored(self, episode: int) -> None:
        for si, seg in enumerate(self.segments):
            if not seg or si in self.merged_into:
                continue
            info = self.info(si)
            reach = self.reachable_cells(si)
            unentered = sum(1 for d in self.reachable_door_list(si, reach) if d["leads_to"] is None)
            done = bool(reach) and not self.frontier(si, reachable=reach) and unentered == 0
            if done and info.explored is None:
                info.explored = episode
                self.stats["segments_explored"] += 1
            elif not done:
                info.explored = None

    def observe(self, frame: np.ndarray, identified: Dict[Tuple[int, int, str], Any],
                shift: Cell = (0, 0), can_learn_walkable: bool = False,
                record_visit: bool = True) -> int:
        seg = self.segments[self.segment]
        a = _screen(frame)
        px, py = self.pos
        added = 0
        if shift != (0, 0) or has_bottom_dialogue_box(frame) or has_modal_preview_box(frame):
            if record_visit:
                self.visits[self.segment][self.pos] += 1
                self.trail.append((self.step, self.segment, self.pos))
            return 0
        masked_cells = {key[:2] for key in TileRecognizer.overlay_tile_keys(frame)} if shift == (0, 0) else set()
        quarter_tags: Dict[Cell, Dict[str, str]] = {}
        quarter_descriptions: Dict[Cell, Dict[str, Optional[str]]] = {}
        quarter_images: Dict[Cell, Dict[str, Optional[np.ndarray]]] = {}
        if shift == (0, 0):
            for (ix, iy, iq), tile in (identified or {}).items():
                img = getattr(tile, "image", None)
                quarter_tags.setdefault((ix, iy), {})[iq] = normalize_tabletop_tag(
                    tile.tag, tile.description, img
                )
                quarter_descriptions.setdefault((ix, iy), {})[iq] = tile.description
                quarter_images.setdefault((ix, iy), {})[iq] = img
            for cell in list(quarter_tags):
                quarter_tags[cell] = filter_cell_warp_quarter_tags(quarter_tags[cell])
        local_hashes_by_cell = {
            cell: {q: getattr(identified.get((cell[0], cell[1], q)), "hash", None) for q in QUARTERS}
            for cell in quarter_tags
        }
        patterned_floor_cells = infer_spatial_patterned_floor_cells(
            quarter_tags, quarter_descriptions, quarter_images, local_hashes_by_cell
        )
        for (x, y), piece in _local_cells(a, shift):
            if (x, y) in masked_cells and (x, y) != (0, 0):
                continue
            pixels = piece.copy()
            quarters = split_quarters(piece)
            hashes = self._cell_hashes(quarters)
            cell_q = quarter_tags.get((x, y))
            cell_d = quarter_descriptions.get((x, y))
            cell_imgs = quarter_images.get((x, y))
            if (x, y) == (0, 0):
                tags, kind = ("player",), WALKABLE
            elif shift != (0, 0):
                tags, kind = ("unknown",), UNKNOWN
            elif cell_q:
                kind, tags = classify_cell_from_quarters(
                    (x, y),
                    quarter_tags,
                    quarter_descriptions,
                    quarter_images,
                    hashes=hashes,
                    patterned_floor_cells=patterned_floor_cells,
                    ledge_patterns=self.ledge_patterns,
                    left_ledge_patterns=self.left_ledge_patterns,
                    walk_patterns=self.walk_patterns,
                    allow_player_overhang=True,
                    allow_player_cap=True,
                )
            else:
                tags = tuple(sorted({
                    normalize_tabletop_tag(
                        identified[(x, y, q)].tag,
                        identified[(x, y, q)].description,
                        getattr(identified[(x, y, q)], "image", None),
                    )
                    for q in QUARTERS if (x, y, q) in identified
                }))
                kind = self._classify(tags, hashes)
            g = (px + x, py + y)
            while len(self.blocked_cells) <= self.segment:
                self.blocked_cells.append(set())
            seg_blocked = self.blocked_cells[self.segment]
            is_uniform_piece = int(pixels.min()) == int(pixels.max())
            cell_learnable = self._can_learn_walkable(
                tags,
                hashes=hashes,
                pixels=pixels,
                quarter_tags=cell_q,
                quarter_descriptions=cell_d,
                quarter_images=cell_imgs,
            )
            old = seg.get(g)
            if (x, y) == (0, 0):
                seg_blocked.discard(g)
                self._clear_blocked_moves_to(self.segment, g)
            elif g in seg_blocked and kind == WALKABLE and not bool(set(tags) & DOOR_TAGS):
                if (
                    (len(set(hashes)) == 1 and None not in hashes and hashes in self.walk_patterns and cell_learnable)
                    or (old is not None and old.hashes != hashes and ("obstacle" in old.tags or "interactable object" in old.tags))
                ):
                    seg_blocked.discard(g)
                    self._clear_blocked_moves_to(self.segment, g)
                else:
                    kind = BLOCKED
            elif old is not None and old.kind == BLOCKED and kind == WALKABLE and old.hashes != hashes:
                self._clear_blocked_moves_to(self.segment, g)
            if old is None:
                if (
                    is_uniform_piece
                    and self.info(self.segment).kind == OUTDOORS
                    and (kind != WALKABLE or int(pixels.max()) <= BLACK_LEVEL)
                ):
                    continue
                seg[g] = MapCell(kind, tags, hashes, self.step, self.step, pixels)
                added += 1
                continue
            prev_last_seen = old.last_seen
            old.last_seen = self.step
            if is_uniform_piece and old.pixels is not None and int(old.pixels.min()) != int(old.pixels.max()):
                if old.kind == BLOCKED and kind == WALKABLE and (x, y) in patterned_floor_cells and g not in seg_blocked:
                    old.kind, old.tags = kind, tags
                continue
            # an NPC on a cell, or a tile not yet recognised, never overwrites what the cell was known to be
            if kind in (OCCUPIED, UNKNOWN) and old.kind not in (OCCUPIED, UNKNOWN):
                continue
            if is_door(old) and not bool(set(tags) & DOOR_TAGS) and any(
                self.resolve(a, pa) == (self.segment, g) or self.resolve(b, pb) == (self.segment, g)
                for _, a, pa, b, pb in self.warps
            ):
                continue
            if (x, y) == (0, 0) and old.kind != OCCUPIED:
                if can_learn_walkable and old.kind == BLOCKED:
                    self._learn_walkable(old)
                continue
            if (x, y) == (0, 1) and old.kind == WALKABLE and kind == WALKABLE:
                # keep clean ground hashes/pixels at (0, 1) rather than overwriting with the player's cap overhang
                continue
            if (
                old.kind == WALKABLE
                and not is_door(old)
                and self.visits[self.segment].get(g, 0) > 0
                and g not in seg_blocked
                and kind == BLOCKED
                and not bool(set(tags) & DOOR_TAGS)
                and cell_learnable
            ):
                continue
            if (
                old.kind == BLOCKED
                and "obstacle" in old.tags
                and kind == WALKABLE
                and not bool(set(tags) & DOOR_TAGS)
                and len(set(hashes)) == 1
                and hashes[0] is not None
                and self.visits[self.segment].get(g, 0) == 0
                and self.visits[self.segment].get(self.pos, 0) == 0
                and not (abs(x) <= 2 and abs(y) <= 2)
                and old.hashes != hashes
                and old.pixels is not None
                and int(old.pixels.min()) < int(old.pixels.max())
            ):
                continue
            if (
                old.kind not in (OCCUPIED, UNKNOWN)
                and old.hashes != hashes
                and prev_last_seen < self.step
            ):
                existing_prior = getattr(old, "prior", None)
                if existing_prior is None or (self.step - int(existing_prior[4]) > 600):
                    old.prior = (
                        old.kind,
                        old.tags,
                        old.hashes,
                        old.first_seen,
                        prev_last_seen,
                        old.pixels.copy() if old.pixels is not None else None,
                    )
            if (
                kind == BLOCKED
                and not cell_learnable
                and "obstacle" in tags
                and self.visits[self.segment].get(g, 0) > 0
            ):
                self.visits[self.segment].pop(g, None)
            old.kind, old.tags, old.hashes, old.pixels = kind, tags, hashes, pixels
        if record_visit:
            self.visits[self.segment][self.pos] += 1
            self.trail.append((self.step, self.segment, self.pos))
        self._new_this_episode += added
        self.stats["cells_added"] += added
        return added

    def on_step(self, frame: np.ndarray, identified, free_roam: bool, direction: Optional[str] = None,
                max_steps: int = 0, reported: Optional[int] = None, was_free_roam: bool = True,
                faded: bool = False, in_battle: bool = False,
                passed_frames: Optional[Iterable[np.ndarray]] = None) -> str:
        self.step += 1
        if in_battle or (not free_roam and is_battle_screen(frame)):
            if not self.in_battle or self.battle_place is None:
                self.battle_place = self.resolve(self.segment, self.pos)
                self.battle_start_step = self.step
            self.in_battle = True
            self.since_battle_moved = False
            self.faded = False
            self.fade_direction = None
            self.lost = 0
        elif free_roam and direction in DIRECTIONS and not self.in_battle:
            self.since_battle_moved = True
        if faded and (
            (not self.in_battle and self.since_battle_moved and (free_roam or was_free_roam))
            or (self.in_battle and free_roam and self.battle_place is None)
        ):
            # after a blank screen the position must be recognised from the map, not slid from the last frame
            self.faded = True
            if direction in DIRECTIONS:
                self.fade_direction = direction
        if not free_roam:
            # remember a move under way so the next free-roam frame is placed relative to the last good one
            if was_free_roam and direction in DIRECTIONS:
                self.pending = (direction, max_steps, reported)
            self.fade_direction = None
            self.stats["not_free_roam"] += 1
            return "not_free_roam"
        if not has_structure(frame):
            if direction in DIRECTIONS:
                self.pending = (direction, max_steps, reported)
            self.stats["blank_frames"] += 1
            return "blank"
        if self.anchor is None:
            shift = self._grid_shift(frame)
            self.anchor, self.residual = _screen(frame), grid_residual(shift)
            self._anchor_place = (self.segment, self.pos)
            self._anchor_pending = None
            self.in_battle = False
            self.observe(frame, identified, shift)
            return "start"
        was_in_battle = bool(self.in_battle)
        had_pending = self.pending is not None
        pre_battle_place = self.resolve(*self.battle_place) if (was_in_battle and self.battle_place is not None) else None
        use_passed = passed_frames if (not was_in_battle and self.pending is None) else None
        anchor_place = getattr(self, "_anchor_place", None)
        anchor_pending = getattr(self, "_anchor_pending", None)
        if self.pending is not None and direction not in DIRECTIONS:
            direction, max_steps, reported = self.pending
        elif anchor_pending is not None and direction not in DIRECTIONS and anchor_place is not None and anchor_place[0] == self.segment and anchor_place[1] != self.pos:
            direction, max_steps, reported = anchor_pending
        base_pos = (
            anchor_place[1]
            if (anchor_place is not None and anchor_place[0] == self.segment and anchor_pending is not None)
            else self.pos
        )
        if (
            not self.faded
            and direction not in DIRECTIONS
            and self.pos is not None
            and 0 <= self.segment < len(self.segments)
            and self.segments[self.segment]
            and self._blacks_out(
                frame,
                self.segment,
                self.pos,
                tuple((-r) % GRID_SKIP for r in self.residual),
            )
        ):
            self.stats["blackout_frames"] += 1
            return "blank"
        placed = None if self.faded else register(
            self.anchor,
            frame,
            direction,
            max_steps,
            reported,
            self.residual,
            passed_frames=use_passed,
        )
        if placed is not None:
            delta, residual, structural = placed
            prev_pos = self.pos
            prev_residual = self.residual
            pos = (base_pos[0] + delta[0], base_pos[1] + delta[1])
            step_delta = (pos[0] - prev_pos[0], pos[1] - prev_pos[1])
            shift = tuple((-r) % GRID_SKIP for r in residual)
            if self._blacks_out(frame, self.segment, pos, shift):
                if direction in DIRECTIONS and self.pending is None:
                    self.pending = (direction, max_steps, reported)
                self.stats["blackout_frames"] += 1
                return "blank"
            was_off_grid = prev_residual != (0, 0) or residual != (0, 0)
            if residual != (0, 0) and not structural and grid_offset(frame) == (0, 0):
                residual = (0, 0)
                shift = (0, 0)
            self.pos, self.residual = pos, residual
            if not structural and self.residual == (0, 0):
                snapped = self._snap_within_segment(frame, identified, was_off_grid=was_off_grid)
                if snapped is not None and snapped != self.pos:
                    self.pos = snapped
                    step_delta = (self.pos[0] - prev_pos[0], self.pos[1] - prev_pos[1])
            placement_rejected = False
            if (
                not structural
                and self.residual == (0, 0)
                and (was_in_battle or had_pending)
                and not has_bottom_dialogue_box(frame)
                and not has_modal_preview_box(frame)
            ):
                m_pl, ov_pl = self._placement_screen_agreement(frame, self.segment, self.pos, (0, 0))
                if ov_pl >= 16 and m_pl < 0.40 * ov_pl:
                    self.pos, self.residual = prev_pos, prev_residual
                    placed = None
                    placement_rejected = True
            if not placement_rejected:
                if not structural:
                    self._learn_ledges(base_pos, pos, direction)
                    if (delta[0] == 0) != (delta[1] == 0) and 1 < abs(delta[0]) + abs(delta[1]) <= MAX_STEP_FILL:
                        sx = (delta[0] > 0) - (delta[0] < 0)
                        sy = (delta[1] > 0) - (delta[1] < 0)
                        for i in range(1, abs(delta[0]) + abs(delta[1])):
                            mid = (base_pos[0] + sx * i, base_pos[1] + sy * i)
                            c_mid = self.segments[self.segment].get(mid)
                            if c_mid is None or c_mid.kind not in (BLOCKED, LEDGE, LEDGE_LEFT):
                                self.visits[self.segment][mid] += 1
                if not structural and step_delta != (0, 0):
                    self.clear_traversed_step(prev_pos, self.pos, self.segment)
                placed_steps = abs(delta[0]) + abs(delta[1])
                if structural:
                    if direction in DIRECTIONS and delta != (0, 0):
                        if anchor_place is None or anchor_place[0] != self.segment:
                            self._anchor_place = (self.segment, prev_pos)
                        self._anchor_pending = (direction, max_steps, reported)
                    if direction in DIRECTIONS and reported is not None and placed_steps < reported:
                        rem_reported = reported - placed_steps
                        rem_max = max(rem_reported + 1, max_steps - placed_steps)
                        self.pending = (direction, rem_max, rem_reported)
                    else:
                        self.pending = None
                else:
                    self.pending = None
                    self._anchor_pending = None
                self.lost = 0
                if not structural:
                    self.anchor = _screen(frame)
                    self._anchor_place = (self.segment, self.pos)
                if not structural:
                    self.in_battle = False
                self.stats["placed_by_edges" if structural else "placed"] += 1
                if structural:
                    # recoloured (a flash or fade): the position holds, the colours do not
                    self.stats["recoloured_frames"] += 1
                elif self.residual == (0, 0):
                    self.observe(frame, identified, can_learn_walkable=(step_delta != (0, 0)))
                    self._try_merge(frame)
                else:
                    self.observe(frame, identified, shift)
                    self.stats["off_grid_frames"] += 1
                return "moved" if step_delta != (0, 0) else "stayed"
        shift = self._grid_shift(frame)
        warp_targets = self._known_warp_targets(direction, max_steps, reported) if self.faded else None
        allow_fn = lambda si, q: not self.arrival_breaks_doors(si, q)
        found = None
        if (
            was_in_battle
            and not self.faded
            and pre_battle_place is not None
            and 0 <= pre_battle_place[0] < len(self.segments)
            and self.segments[pre_battle_place[0]]
        ):
            b_seg, b_pos = pre_battle_place
            found = self.relocalise(
                frame,
                shift=shift,
                allowed=lambda si, q: (
                    si == b_seg
                    and abs(q[0] - b_pos[0]) + abs(q[1] - b_pos[1]) <= MAX_STEP_FILL + 1
                    and allow_fn(si, q)
                ),
                warp_targets=warp_targets,
            )
            if found is None and not self._blacks_out(frame, b_seg, b_pos, shift) and not self._wipe_closing(frame):
                m_b, ov_b = self._placement_screen_agreement(frame, b_seg, b_pos, shift)
                far_found = self.relocalise(
                    frame,
                    shift=shift,
                    allowed=allow_fn,
                    warp_targets=warp_targets,
                )
                if far_found is not None and far_found != (b_seg, b_pos):
                    f_seg, f_pos = far_found
                    f_map = self.segments[f_seg] if 0 <= f_seg < len(self.segments) else {}
                    m_f, ov_f = self._placement_screen_agreement(frame, f_seg, f_pos, shift)
                    has_respawn_door = any(
                        (f_pos[0] + dx, f_pos[1] + dy) in f_map and is_door(f_map[(f_pos[0] + dx, f_pos[1] + dy)])
                        for dx, dy in ((0, 1), (0, 0), (0, -1), (-1, 0), (1, 0))
                    )
                    if (
                        ov_f >= 12
                        and (ov_b == 0 or m_b < 0.45 * ov_b)
                        and (
                            m_f >= 0.75 * ov_f
                            or (has_respawn_door and m_f >= 0.70 * ov_f)
                        )
                    ):
                        found = far_found
                        self.battle_place = None
                if found is None:
                    if ov_b >= 12 and m_b < 0.45 * ov_b:
                        self.segment, self.pos = b_seg, b_pos
                        self.lost += 1
                        self.stats["lost_frames"] += 1
                        return "lost"
                    found = (b_seg, b_pos)
        elif not self.faded and 0 <= self.segment < len(self.segments) and self.segments[self.segment]:
            found = self.relocalise(
                frame,
                shift=shift,
                allowed=lambda si, q: si == self.segment and allow_fn(si, q),
                warp_targets=warp_targets,
            )
        if found is None and not (was_in_battle and not self.faded and pre_battle_place is not None):
            found = self.relocalise(
                frame,
                shift=shift,
                allowed=allow_fn,
                warp_targets=warp_targets,
            )
            if (
                found is None
                and not self.faded
                and 0 <= self.segment < len(self.segments)
                and self.segments[self.segment]
                and self.pos is not None
                and not self._blacks_out(frame, self.segment, self.pos, shift)
                and not self._wipe_closing(frame)
            ):
                m_cur, ov_cur = self._placement_screen_agreement(frame, self.segment, self.pos, shift)
                if ov_cur >= 16 and m_cur >= 0.65 * ov_cur:
                    found = (self.segment, self.pos)
        if found is not None and self._blacks_out(frame, *found, shift):
            if direction in DIRECTIONS and self.pending is None:
                self.pending = (direction, max_steps, reported)
            self.stats["blackout_frames"] += 1
            return "blank"
        if found is not None:
            prev_segment = self.segment
            if (
                found == (self.segment, self.pos)
                and not self.faded
                and direction in DIRECTIONS
                and reported is not None
                and reported > 0
            ):
                dx, dy = DIRECTIONS[direction]
                seg = self.segments[self.segment]
                seg_blocked = self.blocked_cells[self.segment] if self.segment < len(self.blocked_cells) else set()
                adv_pos = self.pos
                seg_moves = self.segment_blocked_moves(self.segment)
                for _ in range(reported):
                    if direction in seg_moves.get(adv_pos, ()):
                        break
                    nxt = (adv_pos[0] + dx, adv_pos[1] + dy)
                    c_nxt = seg.get(nxt)
                    if nxt in seg_blocked or (c_nxt is not None and c_nxt.kind in (BLOCKED, LEDGE, LEDGE_LEFT)):
                        break
                    adv_pos = nxt
                if adv_pos != self.pos:
                    a_scr = _screen(frame)
                    masked_adv = {key[:2] for key in TileRecognizer.overlay_tile_keys(frame)} if shift == (0, 0) else set()
                    local_adv: Dict[Cell, Tuple[Optional[str], ...]] = {}
                    lp_adv: Dict[Cell, np.ndarray] = {}
                    blank_adv: List[Cell] = []
                    for (lx, ly), piece in _local_cells(a_scr, shift):
                        if (lx, ly) == (0, 0) or (lx, ly) in masked_adv:
                            continue
                        if int(piece.min()) == int(piece.max()):
                            blank_adv.append((lx, ly))
                            continue
                        local_adv[(lx, ly)] = self._cell_hashes(split_quarters(piece))
                        lp_adv[(lx, ly)] = piece
                    m_cur, _ = self._screen_agreement(seg, local_adv, self.pos, blank_adv, lp_adv)
                    m_adv, ov_adv = self._screen_agreement(seg, local_adv, adv_pos, blank_adv, lp_adv)
                    if ov_adv == 0 or (m_adv >= RELOCALISE_MIN_FRACTION * ov_adv and m_adv >= m_cur - 2):
                        cur_step = self.pos
                        while cur_step != adv_pos:
                            cur_step = (cur_step[0] + dx, cur_step[1] + dy)
                            self.visits[self.segment][cur_step] += 1
                        found = (self.segment, adv_pos)
            if found != (self.segment, self.pos):
                self._link(*found, departure_pos=self._warp_departure_pos(direction, max_steps, reported))
            self.segment, self.pos = found
            self.residual, self.pending, self.lost, self.anchor = grid_residual(shift), None, 0, _screen(frame)
            self._anchor_place = (self.segment, self.pos)
            self._anchor_pending = None
            self.faded = False
            self.in_battle = False
            self.stats["relocalised"] += 1
            self.observe(frame, identified, shift)
            if prev_segment != self.segment:
                self.prune_transient_segments()
            return "relocalised"
        if self._wipe_closing(frame):
            if direction in DIRECTIONS and self.pending is None:
                self.pending = (direction, max_steps, reported)
            self.stats["wipe_frames"] += 1
            return "blank"
        # often one bad frame, still fading in: wait a few frames before deciding this is somewhere new
        self.lost += 1
        if direction in DIRECTIONS and self.pending is None:
            self.pending = (direction, max_steps, reported)
        if self.lost < LOST_PATIENCE:
            self.stats["lost_frames"] += 1
            return "lost"
        prev_segment = self.segment
        self.segments.append({})
        self.visits.append(Counter())
        self.blocked_cells.append(set())
        self.blocked_moves.append({})
        self.segment_info[len(self.segments) - 1] = SegmentInfo()
        self._link(
            len(self.segments) - 1,
            (0, 0),
            departure_pos=self._warp_departure_pos(direction, max_steps, reported),
        )
        self.segment, self.pos = len(self.segments) - 1, (0, 0)
        self.residual, self.pending, self.lost, self.anchor = grid_residual(shift), None, 0, _screen(frame)
        self._anchor_place = (self.segment, self.pos)
        self._anchor_pending = None
        self.faded = False
        self.in_battle = False
        self.segment_born = self.stats["placed"]
        self.segment_born_id = self.segment
        self.stats["new_segment"] += 1
        self.observe(frame, identified, shift)
        if prev_segment != self.segment:
            self.prune_transient_segments()
        return "new_segment"

    def _warp_departure_pos(self, direction: Optional[str], max_steps: int = 0,
                            reported: Optional[int] = None) -> Cell:
        if not self.faded or self.pos is None:
            return self.pos
        seg = self.segments[self.segment] if 0 <= self.segment < len(self.segments) else None
        if not seg:
            return self.pos
        near_here = self._door_at(self.segment, self.pos)
        if near_here in seg and is_door(seg[near_here]):
            return near_here
        move_dir = direction if direction in DIRECTIONS else (
            self.fade_direction if self.fade_direction in DIRECTIONS else None
        )
        if move_dir not in DIRECTIONS:
            return self.pos
        dx, dy = DIRECTIONS[move_dir]
        if reported is not None and reported > 0:
            candidate = (self.pos[0] + dx * reported, self.pos[1] + dy * reported)
            door_c = self._door_at(self.segment, candidate)
            if door_c in seg and is_door(seg[door_c]):
                return door_c
        reach = max(int(reported or 0), int(max_steps or 0), MAX_STEP_FILL)
        for k in range(1, min(reach, MAX_STEP_FILL) + 1):
            candidate = (self.pos[0] + dx * k, self.pos[1] + dy * k)
            c = seg.get(candidate)
            if c is not None and is_door(c):
                return candidate
        return self.pos

    def _snap_warp_door_endpoint(self, segment: int, pos: Cell) -> Cell:
        seg = self.segments[segment] if 0 <= segment < len(self.segments) else {}
        if not seg:
            return pos
        c = seg.get(pos)
        if c is not None and is_door(c):
            return pos
        d = self._door_at(segment, pos)
        if d != pos and d in seg and is_door(seg[d]) and max(abs(d[0] - pos[0]), abs(d[1] - pos[1])) <= DOOR_REACH:
            return d
        return pos

    def _link(self, segment: int, pos: Cell, departure_pos: Optional[Cell] = None) -> None:
        if not self.faded or self.in_battle or not self.since_battle_moved or segment == self.segment:
            return
        dep = self.pos if departure_pos is None else departure_pos
        move = self.fade_direction if self.fade_direction in DIRECTIONS else None
        if not self._is_scripted_cutscene_warp(self.segment, dep, segment, pos, move):
            dep = self._snap_warp_door_endpoint(self.segment, dep)
            pos = self._snap_warp_door_endpoint(segment, pos)
        if move is None:
            seg_a = self.segments[self.segment] if 0 <= self.segment < len(self.segments) else {}
            seg_b = self.segments[segment] if 0 <= segment < len(self.segments) else {}
            da = self._door_at(self.segment, dep)
            db = self._door_at(segment, pos)
            ca = seg_a.get(da)
            cb = seg_b.get(db)
            has_door_a = ca is not None and is_door(ca) and max(abs(da[0] - dep[0]), abs(da[1] - dep[1])) <= DOOR_REACH
            has_door_b = cb is not None and is_door(cb) and max(abs(db[0] - pos[0]), abs(db[1] - pos[1])) <= DOOR_REACH
            if (
                not has_door_a
                and not has_door_b
                and ca is not None
                and ca.kind not in (OCCUPIED, UNKNOWN)
                and "unknown" not in ca.tags
                and (cb is None or (cb.kind not in (OCCUPIED, UNKNOWN) and "unknown" not in cb.tags))
            ):
                return
        self.warps.append((self.step, self.segment, dep, segment, pos))
        self.warp_moves.append(move)
        left = self.segment
        if (self.info(left).name is None and self._has_min_observations(left)
                and all(entry["segment"] != left for entry in self.naming_queue)):
            self.naming_queue.append({"segment": left, "exit": dep, "step": self.step, "marker": None,
                                      "reason": "door"})

    def _snap_within_segment(
        self,
        frame: np.ndarray,
        identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
        was_off_grid: bool = False,
    ) -> Optional[Cell]:
        if not (0 <= self.segment < len(self.segments)):
            return None
        seg = self.segments[self.segment]
        if not seg or self.pos is None or has_bottom_dialogue_box(frame) or has_modal_preview_box(frame):
            return None
        a = _screen(frame)
        masked_cells = {key[:2] for key in TileRecognizer.overlay_tile_keys(frame)}
        local: Dict[Cell, Tuple[Optional[str], ...]] = {}
        local_pixels: Dict[Cell, np.ndarray] = {}
        blank_local: List[Cell] = []
        for (x, y), piece in _local_cells(a, (0, 0)):
            if (x, y) == (0, 0) or (x, y) in masked_cells:
                continue
            if int(piece.min()) == int(piece.max()):
                blank_local.append((x, y))
                continue
            local[(x, y)] = self._cell_hashes(split_quarters(piece))
            local_pixels[(x, y)] = piece
        if len(local) < 20:
            return None
        live_kinds: Dict[Cell, str] = {}
        if identified:
            quarter_tags: Dict[Cell, Dict[str, str]] = {}
            quarter_descriptions: Dict[Cell, Dict[str, Optional[str]]] = {}
            quarter_images: Dict[Cell, Dict[str, Optional[np.ndarray]]] = {}
            for (ix, iy, iq), tile in identified.items():
                img = getattr(tile, "image", None)
                quarter_tags.setdefault((ix, iy), {})[iq] = normalize_tabletop_tag(
                    tile.tag, tile.description, img
                )
                quarter_descriptions.setdefault((ix, iy), {})[iq] = tile.description
                quarter_images.setdefault((ix, iy), {})[iq] = img
            for cell in list(quarter_tags):
                quarter_tags[cell] = filter_cell_warp_quarter_tags(quarter_tags[cell])
            local_hashes_by_cell = {
                cell: {q: getattr(identified.get((cell[0], cell[1], q)), "hash", None) for q in QUARTERS}
                for cell in quarter_tags
            }
            patterned_floor_cells = infer_spatial_patterned_floor_cells(
                quarter_tags, quarter_descriptions, quarter_images, local_hashes_by_cell
            )
            for (x, y), h in local.items():
                if (x, y) in quarter_tags:
                    lk, _ = classify_cell_from_quarters(
                        (x, y),
                        quarter_tags,
                        quarter_descriptions,
                        quarter_images,
                        hashes=h,
                        patterned_floor_cells=patterned_floor_cells,
                        ledge_patterns=self.ledge_patterns,
                        left_ledge_patterns=self.left_ledge_patterns,
                        walk_patterns=self.walk_patterns,
                        allow_player_overhang=True,
                        allow_player_cap=True,
                    )
                    live_kinds[(x, y)] = lk
        visits = self.visits[self.segment] if self.segment < len(self.visits) else Counter()
        px, py = self.pos
        scores: Dict[Cell, Dict[str, Any]] = {}
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 2:
                    continue
                cand = (px + dx, py + dy)
                if (dx, dy) != (0, 0) and self._blacks_out(frame, self.segment, cand, (0, 0)):
                    continue
                m, ov = self._screen_agreement(seg, local, cand, blank_local, local_pixels)
                vis_match = vis_total = 0
                lm_match = lm_mismatch = 0
                vis_blk = vis_wlk = 0
                for (lx, ly), h in local.items():
                    g = (cand[0] + lx, cand[1] + ly)
                    v = visits.get(g, 0)
                    if v > 0 and (lx, ly) in live_kinds:
                        lk = live_kinds[(lx, ly)]
                        if lk == BLOCKED:
                            vis_blk += 1
                        elif lk == WALKABLE:
                            vis_wlk += 1
                    c = seg.get(g)
                    if c is None or c.kind == OCCUPIED:
                        continue
                    matched = _cells_match(h, local_pixels.get((lx, ly)), c)
                    if v > 0:
                        vis_total += 1
                        if matched:
                            vis_match += 1
                    if is_door(c) or "sign" in c.tags:
                        if matched:
                            lm_match += 1
                        else:
                            lm_mismatch += 1
                cand_cell = seg.get(cand)
                on_blocked = (
                    cand_cell is not None
                    and cand_cell.kind == BLOCKED
                    and "obstacle" in cand_cell.tags
                    and visits.get(cand, 0) == 0
                )
                scores[(dx, dy)] = {
                    "pos": cand,
                    "m": m,
                    "ov": ov,
                    "frac": m / ov if ov else 0.0,
                    "vis_match": vis_match,
                    "vis_total": vis_total,
                    "lm_match": lm_match,
                    "lm_mismatch": lm_mismatch,
                    "vis_blk": vis_blk,
                    "vis_wlk": vis_wlk,
                    "on_blocked": on_blocked,
                }
        s0 = scores.get((0, 0))
        if s0 is None:
            return None
        cands = []
        for d, sc in scores.items():
            if d == (0, 0) or sc["ov"] < 20 or sc["on_blocked"]:
                continue
            self_per = pixel_diff(a, a, d[0] * GRID_SKIP, -d[1] * GRID_SKIP)
            if self_per is not None and self_per <= 0.01:
                continue
            dist = abs(d[0]) + abs(d[1])
            qualifies = False
            if sc["frac"] >= 0.85 and (
                sc["frac"] - s0["frac"] >= 0.08
                or (sc["lm_match"] > s0["lm_match"] and sc["lm_mismatch"] < s0["lm_mismatch"] and sc["frac"] - s0["frac"] >= 0.04)
                or (sc["vis_total"] >= 4 and sc["vis_match"] == sc["vis_total"] and s0["vis_match"] < s0["vis_total"] and sc["frac"] - s0["frac"] >= 0.05)
                or (s0["on_blocked"] and sc["vis_blk"] <= s0["vis_blk"] and sc["frac"] - s0["frac"] >= 0.04)
            ):
                qualifies = True
            elif (
                was_off_grid
                and dist == 1
                and sc["frac"] - s0["frac"] >= 0.08
                and sc["vis_total"] >= 3
                and sc["vis_match"] == sc["vis_total"]
                and s0["vis_match"] < s0["vis_total"]
            ):
                qualifies = True
            elif (
                dist == 1
                and s0["vis_blk"] >= 3
                and sc["vis_blk"] == 0
                and sc["vis_wlk"] > s0["vis_wlk"]
            ):
                qualifies = True
            if qualifies:
                cands.append(((sc["frac"], -sc["vis_blk"], sc["vis_wlk"], sc["m"], -dist), sc["pos"]))
        if not cands:
            return None
        cands.sort(reverse=True)
        self.stats["within_segment_snaps"] += 1
        return cands[0][1]

    def _blacks_out(self, frame: np.ndarray, segment: int, pos: Cell, shift: Cell = (0, 0)) -> bool:
        seg = self.segments[segment]
        n = 0
        for (x, y), piece in _local_cells(_screen(frame), shift):
            old = seg.get((pos[0] + x, pos[1] + y))
            if old is not None and old.pixels is not None:
                n += not _still_lit(piece, old.pixels).all()
        return n >= BLACKOUT_CELLS

    def _wipe_closing(self, frame: np.ndarray) -> bool:
        seg, shift = self.segments[self.segment], tuple((-r) % GRID_SKIP for r in self.residual)
        a = _screen(frame)
        cells_by_xy = dict(_local_cells(a, shift))
        half = GRID_SKIP // 2
        top_black = sum(1 for x in range(-4, 6) if (x, 4) in cells_by_xy and cells_by_xy[(x, 4)].max() <= BLACK_LEVEL)
        bot_black = sum(1 for x in range(-4, 6) if (x, -4) in cells_by_xy and cells_by_xy[(x, -4)].max() <= BLACK_LEVEL)
        left_black = sum(1 for y in range(-4, 5) if (-4, y) in cells_by_xy and cells_by_xy[(-4, y)].max() <= BLACK_LEVEL)
        right_black = sum(1 for y in range(-4, 5) if (5, y) in cells_by_xy and cells_by_xy[(5, y)].max() <= BLACK_LEVEL)
        black = white = darkened = compressed = 0
        best_same = 0
        best_changed = 999
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                same = changed = b = w = d = c = 0
                for (x, y), piece in cells_by_xy.items():
                    if (x, y) == (0, 0):
                        continue
                    old = seg.get((self.pos[0] + dx + x, self.pos[1] + dy + y))
                    if old is None or old.pixels is None:
                        if int(piece.max()) > BLACK_LEVEL:
                            changed += 1
                        continue
                    new, prior = piece.astype(int), np.squeeze(old.pixels).astype(int)
                    lit = _still_lit(new, prior)
                    if not lit.all():
                        b += 1
                    if new.min() >= WHITE_LEVEL and new.min() == new.max() and prior.min() < WHITE_LEVEL:
                        w += 1
                    if new.max() <= DARK_LEVEL and prior.max() >= BRIGHT_LEVEL:
                        d += 1
                    old_span = int(prior.max()) - int(prior.min())
                    new_span = int(new.max()) - int(new.min())
                    if old_span >= 24 and new_span <= old_span - 16:
                        c += 1
                    lit_quarters = [
                        (slice(qy, qy + half), slice(qx, qx + half))
                        for qy in (0, half)
                        for qx in (0, half)
                        if new[qy:qy + half, qx:qx + half].max() > BLACK_LEVEL
                        and prior[qy:qy + half, qx:qx + half].max() > BLACK_LEVEL
                    ]
                    if not lit_quarters:
                        continue
                    if all(np.mean(new[q] != prior[q]) < WIPE_CELL_CHANGE for q in lit_quarters):
                        same += 1
                    else:
                        changed += 1
                if (dx, dy) == (0, 0):
                    black, white, darkened, compressed = b, w, d, c
                if (same, -changed) > (best_same, -best_changed):
                    best_same, best_changed = same, changed
        if black >= BLACKOUT_CELLS and best_same >= WIPE_SAME_RATIO * best_changed:
            return True
        if top_black >= 8 and bot_black >= 8 and left_black >= 7 and right_black >= 7 and black >= IRIS_WIPE_CELLS:
            return True
        if white >= TRANSITION_FLASH_CELLS:
            return True
        p2, p98 = np.percentile(a, (2, 98))
        if darkened >= TRANSITION_FLASH_CELLS and p98 <= DARK_LEVEL:
            return True
        if compressed >= TRANSITION_FLASH_CELLS and (p98 - p2) <= 96:
            return True
        mid_tone = sum(
            1 for piece in cells_by_xy.values()
            if int(piece.max()) > BLACK_LEVEL
            and int(piece.min()) < int(piece.max())
            and (int(piece.min()) > BLACK_LEVEL or np.any((piece > BLACK_LEVEL) & (piece < WHITE_LEVEL)))
        )
        if cells_by_xy and mid_tone < MIN_MID_TONE_CELLS:
            return True
        if self.anchor is not None and is_wavy_scanline_wipe_against(self.anchor, a):
            return True
        return False

    def _grid_shift(self, frame: np.ndarray) -> Cell:
        shift = grid_offset(frame) or (0, 0)
        if shift != (0, 0):
            self.stats["off_grid_unplaced"] += 1
        return shift

    def _known_warp_targets(self, direction: Optional[str] = None, max_steps: int = 0,
                            reported: Optional[int] = None, for_merge: bool = False) -> set:
        targets = set()
        if not for_merge:
            if not self.faded or self.pos is None:
                return targets
            cur_seg, cur_pos = self.resolve(self.segment, self.pos)
            dep_pos = self._warp_departure_pos(direction, max_steps, reported)
            for idx, (_, a, pa, b, pb) in enumerate(self.warps):
                m = self.warp_moves[idx] if idx < len(self.warp_moves) else None
                if self._is_scripted_cutscene_warp(a, pa, b, pb, m):
                    continue
                ra, rpa = self.resolve(a, pa)
                rb, rpb = self.resolve(b, pb)
                if ra == rb:
                    continue
                if ra == cur_seg and (
                    self._same_door(cur_seg, rpa, cur_pos) or self._same_door(cur_seg, rpa, dep_pos)
                ):
                    targets.add((rb, self._door_at(rb, rpb)))
                if rb == cur_seg and (
                    self._same_door(cur_seg, rpb, cur_pos) or self._same_door(cur_seg, rpb, dep_pos)
                ):
                    targets.add((ra, self._door_at(ra, rpa)))
            return targets
        cur_seg = self.segment
        for idx, (_, a, pa, b, pb) in enumerate(self.warps):
            m = self.warp_moves[idx] if idx < len(self.warp_moves) else None
            if self._is_scripted_cutscene_warp(a, pa, b, pb, m):
                continue
            ra, rpa = self.resolve(a, pa)
            rb, rpb = self.resolve(b, pb)
            if rb != cur_seg or ra == cur_seg:
                continue
            for idx2, (_, a2, pa2, b2, pb2) in enumerate(self.warps):
                m2 = self.warp_moves[idx2] if idx2 < len(self.warp_moves) else None
                if self._is_scripted_cutscene_warp(a2, pa2, b2, pb2, m2):
                    continue
                ra2, rpa2 = self.resolve(a2, pa2)
                rb2, rpb2 = self.resolve(b2, pb2)
                if ra2 == rb2 or cur_seg in (ra2, rb2):
                    continue
                if ra2 == ra and self._same_door(ra, rpa, rpa2):
                    db2 = self._door_at(rb2, rpb2)
                    targets.add((rb2, (db2[0] + self.pos[0] - rpb[0], db2[1] + self.pos[1] - rpb[1])))
                if rb2 == ra and self._same_door(ra, rpa, rpb2):
                    da2 = self._door_at(ra2, rpa2)
                    targets.add((ra2, (da2[0] + self.pos[0] - rpb[0], da2[1] + self.pos[1] - rpb[1])))
        return targets

    def _try_merge(self, frame: np.ndarray) -> None:
        if (
            self.segment == 0
            or self.segment != getattr(self, "segment_born_id", len(self.segments) - 1)
            or bool(self.info(self.segment).name)
            or self.stats["placed"] - self.segment_born > MERGE_WINDOW
            or getattr(self, "_last_merge_try_pos", None) == (self.segment, self.pos)
        ):
            return
        self._last_merge_try_pos = (self.segment, self.pos)
        warp_targets = self._known_warp_targets(for_merge=True)
        found = self.relocalise(frame, exclude=self.segment, warp_targets=warp_targets)
        if found is None:
            return
        target, target_pos = found
        if self.names_differ(self.segment, target):
            return
        offset = (target_pos[0] - self.pos[0], target_pos[1] - self.pos[1])
        if self.merge_breaks_doors(self.segment, target, offset):
            alt_offset = None
            for t_seg, t_pos in warp_targets:
                if t_seg == target:
                    cand = (t_pos[0] - self.pos[0], t_pos[1] - self.pos[1])
                    if not self.merge_breaks_doors(self.segment, target, cand):
                        alt_offset = cand
                        break
            if alt_offset is None:
                ov = self.overlap(self.segment, target)
                if (
                    ov is not None
                    and ov["agreement"] >= CONSOLIDATE_MIN_AGREEMENT
                    and not self.merge_breaks_doors(self.segment, target, ov["offset"])
                ):
                    alt_offset = ov["offset"]
            if alt_offset is None:
                self.stats["merge_door_vetoes"] += 1
                return
            offset = alt_offset
        self._merge(self.segment, target, offset)
        if self.segment == target and self.pos != target_pos:
            self.pos = target_pos

    def _merge(self, source: int, target: int, offset: Cell) -> None:
        dx, dy = offset
        src, dst = self.segments[source], self.segments[target]
        while len(self.blocked_cells) <= max(source, target):
            self.blocked_cells.append(set())
        while len(self.blocked_moves) <= max(source, target):
            self.blocked_moves.append({})
        for (x, y), c in src.items():
            g = (x + dx, y + dy)
            old = dst.get(g)
            if old is not None and (is_door(old) or old.kind != OCCUPIED) and (
                c.kind in (OCCUPIED, UNKNOWN) or not c.tags or "player" in c.tags
            ):
                continue
            if old is None or old.kind in (OCCUPIED, UNKNOWN) or c.last_seen >= old.last_seen:
                dst[g] = c
        for (x, y), v in self.visits[source].items():
            self.visits[target][(x + dx, y + dy)] += v
        for (x, y) in self.blocked_cells[source]:
            g = (x + dx, y + dy)
            if self.visits[target].get(g, 0) == 0:
                self.blocked_cells[target].add(g)
        for (x, y), dirs in self.blocked_moves[source].items():
            g = (x + dx, y + dy)
            valid_dirs = {d for d in dirs if d in DIRECTIONS}
            if valid_dirs:
                self.blocked_moves[target].setdefault(g, set()).update(valid_dirs)
        self.trail = [(t, target, (p[0] + dx, p[1] + dy)) if si == source else (t, si, p)
                      for t, si, p in self.trail]

        def moved(si: int, p: Optional[Cell]) -> Tuple[int, Optional[Cell]]:
            if si != source or p is None:
                return si, p
            return target, (p[0] + dx, p[1] + dy)

        for entry in self.naming_queue:
            entry["segment"], entry["exit"] = moved(entry["segment"], entry["exit"])
        absorbed, kept = self.segment_info.pop(source, SegmentInfo()), self.info(target)
        if kept.name is None:
            absorbed.aliases += kept.aliases
            self.segment_info[target] = absorbed
        elif absorbed.name is not None:
            kept.aliases.append((absorbed.name, absorbed.description))
            if absorbed.name != kept.name:
                kept.needs_merge_rename = True
        self.merged_into[source] = (target, (dx, dy))
        self.segments[source] = {}
        self.visits[source] = Counter()
        self.blocked_cells[source] = set()
        self.blocked_moves[source] = {}
        if self.segment == source:
            self.segment, self.pos = target, (self.pos[0] + dx, self.pos[1] + dy)
        self._clean_warps()
        self.stats["merged"] += 1

    def _clean_warps(self) -> None:
        new_warps: List[Tuple[int, int, Cell, int, Cell]] = []
        new_moves: List[Optional[str]] = []
        for (t, a, pa, b, pb), m in zip(self.warps, self.warp_moves):
            ra, rpa = self.resolve(a, pa)
            rb, rpb = self.resolve(b, pb)
            if ra == rb:
                continue
            if not self._is_scripted_cutscene_warp(ra, rpa, rb, rpb, m):
                rpa = self._snap_warp_door_endpoint(ra, rpa)
                rpb = self._snap_warp_door_endpoint(rb, rpb)
            new_warps.append((t, ra, rpa, rb, rpb))
            new_moves.append(m)
        self.warps, self.warp_moves = new_warps, new_moves

    @staticmethod
    def _block_index(seg: Dict[Cell, MapCell]) -> Dict[Tuple[Any, ...], List[Cell]]:
        idx: Dict[Tuple[Any, ...], List[Cell]] = defaultdict(list)
        for (x, y), c in seg.items():
            if c.kind == OCCUPIED:
                continue
            c1, c2, c3 = seg.get((x + 1, y)), seg.get((x, y + 1)), seg.get((x + 1, y + 1))
            if c1 is not None and c2 is not None and c3 is not None:
                if c1.kind != OCCUPIED and c2.kind != OCCUPIED and c3.kind != OCCUPIED:
                    idx[(c.hashes, c1.hashes, c2.hashes, c3.hashes)].append((x, y))
        return idx

    @staticmethod
    def _collect_offset_votes(local_hashes: Dict[Cell, Tuple[Optional[str], ...]],
                              index: Dict[Tuple[Optional[str], ...], List[Cell]],
                              block_index: Dict[Tuple[Any, ...], List[Cell]]) -> Counter:
        votes: Counter = Counter()
        for (x, y), h in local_hashes.items():
            places = index.get(h, ())
            if len(places) <= RELOCALISE_MAX_FANOUT:
                for p in places:
                    votes[(p[0] - x, p[1] - y)] += 1
            if (x + 1, y) in local_hashes and (x, y + 1) in local_hashes and (x + 1, y + 1) in local_hashes:
                sig = (h, local_hashes[(x + 1, y)], local_hashes[(x, y + 1)], local_hashes[(x + 1, y + 1)])
                b_places = block_index.get(sig, ())
                if len(b_places) <= RELOCALISE_MAX_FANOUT:
                    for p in b_places:
                        votes[(p[0] - x, p[1] - y)] += 2
        if not votes:
            b_counts = [
                len(block_index[sig]) for (x, y), h in local_hashes.items()
                if (x + 1, y) in local_hashes and (x, y + 1) in local_hashes and (x + 1, y + 1) in local_hashes
                and (sig := (h, local_hashes[(x + 1, y)], local_hashes[(x, y + 1)],
                             local_hashes[(x + 1, y + 1)])) in block_index
            ]
            if b_counts:
                cap = min(b_counts) * 2
                for (x, y), h in local_hashes.items():
                    if (x + 1, y) in local_hashes and (x, y + 1) in local_hashes and (x + 1, y + 1) in local_hashes:
                        sig = (h, local_hashes[(x + 1, y)], local_hashes[(x, y + 1)], local_hashes[(x + 1, y + 1)])
                        b_places = block_index.get(sig, ())
                        if len(b_places) <= cap:
                            for p in b_places:
                                votes[(p[0] - x, p[1] - y)] += 1
        return votes

    def _transient_offset(self, si: int, target_seg: int, fallback_pos: Cell) -> Cell:
        from .render import render_segment

        seg_si = self.segments[si]
        seg_dst = self.segments[target_seg]
        if not seg_si or not seg_dst:
            return fallback_pos
        canvas = render_segment(self, si, trail=False, doors=False)
        if canvas is None:
            return fallback_pos
        x0_si, y1_si = min(g[0] for g in seg_si), max(g[1] for g in seg_si)
        a_si = canvas[:, :, 0]
        index: Dict[Tuple[Optional[str], ...], List[Cell]] = defaultdict(list)
        for g, mc in seg_dst.items():
            if mc.kind != OCCUPIED:
                index[mc.hashes].append(g)
        best = (0, fallback_pos)
        for dy in (0, 4, 8, 12):
            for dx in (0, 4, 8, 12):
                cells: Dict[Cell, Tuple[Optional[str], ...]] = {}
                for r in range((a_si.shape[0] - dy) // GRID_SKIP):
                    for c in range((a_si.shape[1] - dx) // GRID_SKIP):
                        piece = a_si[dy + r * GRID_SKIP:dy + (r + 1) * GRID_SKIP,
                                     dx + c * GRID_SKIP:dx + (c + 1) * GRID_SKIP]
                        if piece.min() == piece.max():
                            continue
                        cells[(x0_si + c, y1_si - r)] = self._cell_hashes(split_quarters(piece))
                if not cells:
                    continue
                votes: Counter = Counter()
                for (gx, gy), h in cells.items():
                    for p in index.get(h, ())[:50]:
                        votes[(p[0] - gx, p[1] - gy)] += 1
                for off, _ in votes.most_common(RELOCALISE_CANDIDATES):
                    m = sum(1 for (gx, gy), h in cells.items()
                            if (gx + off[0], gy + off[1]) in seg_dst
                            and seg_dst[(gx + off[0], gy + off[1])].hashes == h)
                    if m > best[0] and m >= 15:
                        best = (m, off)
        return best[1]

    def prune_transient_segments(self) -> int:
        changed = 0
        for si in range(len(self.segments)):
            if si == self.segment or si in self.merged_into or not self.segments[si]:
                continue
            if self._has_min_observations(si):
                continue
            if self.info(si).name is not None and len(self.visits[si]) > 1:
                continue
            si_steps = [t for t, s, _ in self.trail if s == si]
            if not si_steps:
                continue
            t_min, t_max = min(si_steps), max(si_steps)
            before = [(t, s, p) for t, s, p in self.trail if s != si and t < t_min]
            after = [(t, s, p) for t, s, p in self.trail if s != si and t > t_max]
            target_sp = (after[0][1], after[0][2]) if after else ((before[-1][1], before[-1][2]) if before else None)
            if target_sp is None:
                continue
            target_seg, fallback_pos = self.resolve(*target_sp)
            if before and after and self.resolve(before[-1][1], before[-1][2])[0] == target_seg:
                offset = fallback_pos
            else:
                offset = self._transient_offset(si, target_seg, fallback_pos)
            dx, dy = offset
            self.trail = [(t, target_seg, (p[0] + dx, p[1] + dy)) if s == si else (t, s, p)
                          for t, s, p in self.trail]
            self.naming_queue = [e for e in self.naming_queue if e.get("segment") != si]
            self.segment_info.pop(si, None)
            self.merged_into[si] = (target_seg, (dx, dy))
            self.segments[si] = {}
            self.visits[si] = Counter()
            if si < len(self.blocked_cells):
                self.blocked_cells[si] = set()
            if si < len(self.blocked_moves):
                self.blocked_moves[si] = {}
            self._clean_warps()
            self.stats["transient_pruned"] += 1
            changed += 1
        self._clean_warps()
        return changed

    def _segment_version(self, si: int) -> Tuple[int, int, int]:
        seg = self.segments[si] if 0 <= si < len(self.segments) else {}
        if not seg:
            return (0, 0, 0)
        return (
            len(seg),
            sum(c.last_seen for c in seg.values()),
            sum(1 for c in seg.values() if c.kind == OCCUPIED),
        )

    def _segment_index(self, si: int) -> Tuple[
        Dict[Cell, Tuple[Optional[str], ...]],
        Dict[Cell, Tuple[bytes, ...]],
        Dict[Cell, bool],
        Dict[Tuple[Any, ...], List[Cell]],
        Dict[Tuple[Any, ...], List[Cell]],
        Tuple[int, int, int],
    ]:
        seg = self.segments[si] if 0 <= si < len(self.segments) else {}
        ver = self._segment_version(si)
        cache = getattr(self, "_seg_index_cache", None)
        if cache is None:
            cache = {}
            self._seg_index_cache = cache
        entry = cache.get(si)
        if entry is not None and entry[5] == ver:
            return entry
        content_hashes: Dict[Cell, Tuple[Optional[str], ...]] = {}
        content_ranks: Dict[Cell, Tuple[bytes, ...]] = {}
        uniform: Dict[Cell, bool] = {}
        index: Dict[Tuple[Any, ...], List[Cell]] = defaultdict(list)
        for g, c in seg.items():
            if c.kind == OCCUPIED:
                continue
            is_u = c.pixels is not None and int(c.pixels.min()) == int(c.pixels.max())
            uniform[g] = is_u
            if not is_u:
                content_hashes[g] = c.hashes
                index[c.hashes].append(g)
                rank_sig = _quarter_rank_signature(c.pixels)
                if rank_sig is not None:
                    content_ranks[g] = rank_sig
                    index[rank_sig].append(g)
        block_index: Dict[Tuple[Any, ...], List[Cell]] = defaultdict(list)
        for (x, y), h in content_hashes.items():
            h1 = content_hashes.get((x + 1, y))
            h2 = content_hashes.get((x, y + 1))
            h3 = content_hashes.get((x + 1, y + 1))
            if h1 is not None and h2 is not None and h3 is not None:
                block_index[(h, h1, h2, h3)].append((x, y))
        entry = (content_hashes, content_ranks, uniform, index, block_index, ver)
        cache[si] = entry
        return entry

    def overlap(self, source: int, target: int) -> Optional[Dict[str, Any]]:
        seg_src = self.segments[source] if 0 <= source < len(self.segments) else {}
        seg_dst = self.segments[target] if 0 <= target < len(self.segments) else {}
        if not seg_src or not seg_dst:
            return None
        src_hashes, src_rank, src_uniform, _, _, ver_src = self._segment_index(source)
        _, _, dst_uniform, index, block_index, ver_dst = self._segment_index(target)
        ov_cache = getattr(self, "_overlap_cache", None)
        if ov_cache is None:
            ov_cache = {}
            self._overlap_cache = ov_cache
        cache_key = (source, target, ver_src, ver_dst)
        if cache_key in ov_cache:
            return ov_cache[cache_key]
        votes = self._collect_offset_votes(src_hashes, index, block_index)
        if src_rank:
            rank_votes = self._collect_offset_votes(src_rank, index, {})
            for off, cnt in rank_votes.items():
                if cnt > votes[off]:
                    votes[off] = cnt
        if not votes:
            ov_cache[cache_key] = None
            return None

        def agreement(offset: Cell) -> Tuple[int, int]:
            matches = total = 0
            ox, oy = offset
            for g, u_src in src_uniform.items():
                g_dst = (g[0] + ox, g[1] + oy)
                u_dst = dst_uniform.get(g_dst)
                if u_dst is None:
                    continue
                if u_src and u_dst:
                    continue
                total += 1
                if not u_src and not u_dst:
                    c = seg_src[g]
                    c_dst = seg_dst[g_dst]
                    if _cells_match(c.hashes, c.pixels, c_dst):
                        matches += 1
            return matches, total

        ranked = sorted(
            ((agreement(o), v, o) for o, v in votes.most_common(RELOCALISE_CANDIDATES)),
            key=lambda item: (item[0][0], item[0][0] / max(1, item[0][1]), item[1]),
            reverse=True,
        )
        (matches, shared), best_votes, (dx, dy) = ranked[0]
        rivals = [(n, o) for o, n in votes.items() if max(abs(o[0] - dx), abs(o[1] - dy)) > 2]
        rival, rival_offset = max(rivals, default=(0, None))
        rival_matches, rival_shared = agreement(rival_offset) if rival_offset is not None else (0, 0)
        res = {"offset": (dx, dy), "matches": matches, "shared": shared,
               "agreement": matches / shared if shared else 0.0,
               "votes": best_votes, "rival": rival,
               "rival_agreement": rival_matches / rival_shared if rival_shared else 0.0}
        ov_cache[cache_key] = res
        return res

    def _judged_same(self, source: int, target: int, overlap: Dict[str, Any],
                     judge: Optional[Callable[[int, int, Dict[str, Any]], Optional[bool]]]) -> bool:
        if not self.names_differ(source, target):
            return True
        if overlap.get("matches", 0) >= 60 and overlap.get("agreement", 0.0) >= 0.9:
            def base_name(name: Optional[str]) -> Optional[str]:
                if not name:
                    return None
                cleaned = re.sub(r"\([^)]*\)", " ", name.lower())
                return " ".join(re.sub(r"[^\w\s]", " ", cleaned).split())
            na, nb = base_name(self.info(source).name), base_name(self.info(target).name)
            if na and nb and (na == nb or na in nb or nb in na):
                return True
        key = (min(source, target), max(source, target))
        verdict = self.merge_verdicts.get(key)
        if verdict is not None and overlap["matches"] < MERGE_REJUDGE_GROWTH * verdict["matches"]:
            return verdict["same"]
        if (
            overlap.get("matches", 0) >= 60
            and overlap.get("agreement", 0.0) >= 0.9
            and any(
                self._is_doorless_inplace_warp(a, pa, b, pb, source, target, overlap["offset"])
                for _, a, pa, b, pb in self.warps
            )
        ):
            return True
        if (
            overlap.get("trail_anchored")
            and overlap.get("matches", 0) >= 18
            and overlap.get("agreement", 0.0) >= 0.9
            and (self.info(source).kind in (OUTDOORS, None) or self.info(target).kind in (OUTDOORS, None))
            and self.info(source).kind != BUILDING
            and self.info(target).kind != BUILDING
        ):
            return True
        if judge is None:
            return False
        same = judge(source, target, overlap)
        self.merge_verdicts[key] = {"same": bool(same), "matches": overlap["matches"], "step": self.step,
                                    "names": (self.info(source).name, self.info(target).name)}
        self.stats["merges_judged_same" if same else "merges_judged_different"] += 1
        return bool(same)

    def _trail_anchored_overlap(self, source: int, target: int) -> Optional[Dict[str, Any]]:
        src_seg = self.segments[source] if 0 <= source < len(self.segments) else {}
        dst_seg = self.segments[target] if 0 <= target < len(self.segments) else {}
        if not src_seg or not dst_seg:
            return None
        _, _, src_uniform, _, _, _ = self._segment_index(source)
        _, _, dst_uniform, _, _, _ = self._segment_index(target)
        base_offsets: List[Cell] = []
        for idx in range(1, len(self.trail)):
            t_prev, s_prev, p_prev = self.trail[idx - 1]
            t_cur, s_cur, p_cur = self.trail[idx]
            if s_prev == s_cur:
                continue
            rs_prev, rp_prev = self.resolve(s_prev, p_prev)
            rs_cur, rp_cur = self.resolve(s_cur, p_cur)
            if {rs_prev, rs_cur} != {source, target}:
                continue
            has_warp = any(
                t_prev <= wt <= t_cur
                and (
                    self.resolve(wa, (0, 0))[0] in (source, target)
                    or self.resolve(wb, (0, 0))[0] in (source, target)
                )
                for wt, wa, _, wb, _ in self.warps
            )
            if has_warp:
                continue
            has_defeat = any(t_prev <= int(d.get("step", 0)) <= t_cur for d in getattr(self, "defeats", ()))
            if has_defeat:
                continue
            if rs_prev == target and rs_cur == source:
                base_off = (rp_prev[0] - rp_cur[0], rp_prev[1] - rp_cur[1])
            else:
                base_off = (rp_cur[0] - rp_prev[0], rp_cur[1] - rp_prev[1])
            if base_off not in base_offsets:
                base_offsets.append(base_off)
        if not base_offsets:
            return None
        best = None
        for bx, by in base_offsets:
            for dx in range(-2, 3):
                for dy in range(-10, 11):
                    if abs(dx) + abs(dy) > 10:
                        continue
                    ox, oy = bx + dx, by + dy
                    matches = total = 0
                    for g, u_src in src_uniform.items():
                        if g == (0, 0) and src_seg[g].tags == ("player",):
                            continue
                        g_dst = (g[0] + ox, g[1] + oy)
                        u_dst = dst_uniform.get(g_dst)
                        if u_dst is None or (u_src and u_dst):
                            continue
                        total += 1
                        if not u_src and not u_dst:
                            c = src_seg[g]
                            c_dst = dst_seg[g_dst]
                            if _cells_match(c.hashes, c.pixels, c_dst):
                                matches += 1
                    needed_shared = min(CONSOLIDATE_MIN_CELLS, max(15, int(0.4 * len(src_uniform))))
                    high_precision_collar = (
                        abs(dx) + abs(dy) <= 10
                        and (
                            (total >= 20 and matches >= 20 and matches >= 0.85 * total)
                            or (total >= 15 and matches >= 15 and matches >= 0.90 * total)
                        )
                    )
                    if (
                        abs(dx) + abs(dy) <= 5
                        and total >= needed_shared
                        and matches >= 0.65 * total
                    ) or high_precision_collar:
                        score = (matches, matches / total, -abs(dx) - abs(dy))
                        if best is None or score > best[0]:
                            best = (score, (ox, oy), matches, total)
        if best is None:
            return None
        _, offset, matches, shared = best
        return {
            "offset": offset,
            "matches": matches,
            "shared": shared,
            "agreement": matches / shared,
            "votes": matches,
            "rival": 0,
            "rival_agreement": 0.0,
            "trail_anchored": True,
        }

    def _warp_anchored_overlap(self, source: int, target: int) -> Optional[Dict[str, Any]]:
        src_seg = self.segments[source] if 0 <= source < len(self.segments) else {}
        dst_seg = self.segments[target] if 0 <= target < len(self.segments) else {}
        if not src_seg or not dst_seg or self.info(source).name is not None:
            return None
        for idx, (_, a, pa, b, pb) in enumerate(self.warps):
            m = self.warp_moves[idx] if idx < len(self.warp_moves) else None
            if self._is_scripted_cutscene_warp(a, pa, b, pb, m):
                continue
            ra, rpa = self.resolve(a, pa)
            rb, rpb = self.resolve(b, pb)
            if rb == source and ra != source:
                src_entry = rpb
                dep_seg, dep_door = ra, self._door_at(ra, rpa)
            elif ra == source and rb != source:
                src_entry = rpa
                dep_seg, dep_door = rb, self._door_at(rb, rpb)
            else:
                continue
            for idx2, (_, a2, pa2, b2, pb2) in enumerate(self.warps):
                m2 = self.warp_moves[idx2] if idx2 < len(self.warp_moves) else None
                if self._is_scripted_cutscene_warp(a2, pa2, b2, pb2, m2):
                    continue
                ra2, rpa2 = self.resolve(a2, pa2)
                rb2, rpb2 = self.resolve(b2, pb2)
                if ra2 == dep_seg and rb2 == target and self._same_door(dep_seg, dep_door, rpa2):
                    dst_door = self._door_at(target, rpb2)
                elif rb2 == dep_seg and ra2 == target and self._same_door(dep_seg, dep_door, rpb2):
                    dst_door = self._door_at(target, rpa2)
                else:
                    continue
                offset = (dst_door[0] - src_entry[0], dst_door[1] - src_entry[1])
                matches = shared = core_matches = core_shared = 0
                for (x, y), c in src_seg.items():
                    if c.kind == OCCUPIED or (c.pixels is not None and int(c.pixels.min()) == int(c.pixels.max())):
                        continue
                    cd = dst_seg.get((x + offset[0], y + offset[1]))
                    if cd is None or cd.kind == OCCUPIED or (
                        cd.pixels is not None and int(cd.pixels.min()) == int(cd.pixels.max())
                    ):
                        continue
                    shared += 1
                    matched = _cells_match(c.hashes, c.pixels, cd)
                    if matched:
                        matches += 1
                    if max(abs(x - src_entry[0]), abs(y - src_entry[1])) <= RELOCALISE_WARP_CORE_RADIUS:
                        core_shared += 1
                        if matched:
                            core_matches += 1
                needed_shared = min(
                    CONSOLIDATE_MIN_CELLS,
                    max(
                        10,
                        int(
                            0.6
                            * sum(
                                1
                                for c in src_seg.values()
                                if c.kind != OCCUPIED
                                and not (c.pixels is not None and int(c.pixels.min()) == int(c.pixels.max()))
                            )
                        ),
                    ),
                )
                if (
                    shared >= needed_shared
                    and matches >= RELOCALISE_WARP_MIN_FRACTION * shared
                    and core_shared > 0
                    and core_matches >= RELOCALISE_MIN_FRACTION * core_shared
                ):
                    return {
                        "offset": offset,
                        "matches": matches,
                        "shared": shared,
                        "agreement": matches / shared,
                        "votes": matches,
                        "rival": 0,
                        "rival_agreement": 0.0,
                    }
        return None

    def _heal_aliased_merged_doors(self) -> int:
        healed = 0
        for src in list(self.merged_into):
            target, (dx, dy) = self.resolve(src, (0, 0))
            seg = self.segments[target] if 0 <= target < len(self.segments) else None
            if not seg:
                continue
            by_depart: Dict[Tuple[int, Cell], List[Cell]] = defaultdict(list)
            for _, a, pa, b, pb in self.warps:
                ra, rpa = self.resolve(a, pa)
                rb, rpb = self.resolve(b, pb)
                if ra == rb:
                    continue
                if rb == target:
                    by_depart[(ra, self._door_at(ra, rpa))].append(rpb)
                elif ra == target:
                    by_depart[(rb, self._door_at(rb, rpb))].append(rpa)
            for pts in by_depart.values():
                door_cells = [
                    (p[0] + ox, p[1] + oy)
                    for p in pts
                    for ox in range(-DOOR_SNAP_REACH, DOOR_SNAP_REACH + 1)
                    for oy in range(-DOOR_SNAP_REACH, DOOR_SNAP_REACH + 1)
                    if (p[0] + ox, p[1] + oy) in seg and is_door(seg[(p[0] + ox, p[1] + oy)])
                ]
                unique_doors = sorted(set(door_cells), key=lambda g: (seg[g].first_seen, seg[g].last_seen))
                if len(unique_doors) > 1 and (dx, dy) in unique_doors:
                    aliased = (dx, dy)
                    canonical = next(
                        (d for d in unique_doors if d != aliased and seg[d].hashes == seg[aliased].hashes),
                        None,
                    )
                    if (
                        canonical is not None
                        and max(abs(canonical[0] - aliased[0]), abs(canonical[1] - aliased[1])) <= DOOR_SNAP_REACH
                    ):
                        nb_grounds = [
                            seg[nb]
                            for ox, oy in DIRECTIONS.values()
                            for nb in [(aliased[0] + ox, aliased[1] + oy)]
                            if nb in seg and seg[nb].kind == WALKABLE and not is_door(seg[nb])
                        ]
                        if nb_grounds:
                            ref = nb_grounds[0]
                            seg[aliased] = MapCell(
                                WALKABLE, ("ground",), ref.hashes, ref.first_seen, ref.last_seen, ref.pixels
                            )
                            self.warps = [
                                (
                                    t,
                                    a,
                                    (canonical if self.resolve(a, pa) == (target, aliased) else pa),
                                    b,
                                    (canonical if self.resolve(b, pb) == (target, aliased) else pb),
                                )
                                for t, a, pa, b, pb in self.warps
                            ]
                            healed += 1
        for target, seg in enumerate(self.segments):
            if not seg or target in self.merged_into:
                continue
            warp_doors = set()
            for _, a, pa, b, pb in self.warps:
                ra, rpa = self.resolve(a, pa)
                rb, rpb = self.resolve(b, pb)
                if ra == rb:
                    continue
                if ra == target:
                    warp_doors.add(self._door_at(ra, rpa))
                if rb == target:
                    warp_doors.add(self._door_at(rb, rpb))
            door_cells = sorted(
                [d for d in warp_doors if d in seg and is_door(seg[d])],
                key=lambda g: (seg[g].first_seen, seg[g].last_seen),
            )
            if len(door_cells) <= 1:
                continue
            for i, d1 in enumerate(door_cells):
                if d1 not in seg or not is_door(seg[d1]):
                    continue
                for d2 in door_cells[i + 1:]:
                    if d2 not in seg or not is_door(seg[d2]):
                        continue
                    if seg[d1].hashes != seg[d2].hashes or None in seg[d1].hashes:
                        continue
                    if max(abs(d1[0] - d2[0]), abs(d1[1] - d2[1])) > DOOR_SNAP_REACH:
                        continue
                    dests_1 = {
                        self.resolve(b, pb)[0]
                        for _, a, pa, b, pb in self.warps
                        if self.resolve(a, pa)[0] == target and self._door_at(target, self.resolve(a, pa)[1]) == d1
                    } | {
                        self.resolve(a, pa)[0]
                        for _, a, pa, b, pb in self.warps
                        if self.resolve(b, pb)[0] == target and self._door_at(target, self.resolve(b, pb)[1]) == d1
                    }
                    dests_2 = {
                        self.resolve(b, pb)[0]
                        for _, a, pa, b, pb in self.warps
                        if self.resolve(a, pa)[0] == target and self._door_at(target, self.resolve(a, pa)[1]) == d2
                    } | {
                        self.resolve(a, pa)[0]
                        for _, a, pa, b, pb in self.warps
                        if self.resolve(b, pb)[0] == target and self._door_at(target, self.resolve(b, pb)[1]) == d2
                    }
                    same_dest = False
                    for s1 in dests_1:
                        for s2 in dests_2:
                            if s1 == s2:
                                same_dest = True
                            else:
                                ov = self.overlap(max(s1, s2), min(s1, s2))
                                if ov and ov.get("matches", 0) >= 60 and ov.get("agreement", 0.0) >= 0.9:
                                    same_dest = True
                    if not same_dest:
                        continue
                    canonical, aliased = (d2, d1) if seg[d2].last_seen >= seg[d1].last_seen else (d1, d2)
                    nb_grounds = [
                        seg[nb]
                        for ox, oy in DIRECTIONS.values()
                        for nb in [(aliased[0] + ox, aliased[1] + oy)]
                        if nb in seg and seg[nb].kind == WALKABLE and not is_door(seg[nb])
                    ]
                    nb_obs = [
                        seg[nb]
                        for ox, oy in DIRECTIONS.values()
                        for nb in [(aliased[0] + ox, aliased[1] + oy)]
                        if nb in seg and seg[nb].kind == BLOCKED and not is_door(seg[nb])
                    ]
                    if nb_grounds:
                        ref = nb_grounds[0]
                        seg[aliased] = MapCell(
                            WALKABLE, ("ground",), ref.hashes, ref.first_seen, ref.last_seen, ref.pixels
                        )
                    elif nb_obs:
                        ref = nb_obs[0]
                        seg[aliased] = MapCell(
                            BLOCKED, ("obstacle",), ref.hashes, ref.first_seen, ref.last_seen, ref.pixels
                        )
                        self.visits[target].pop(aliased, None)
                    else:
                        continue
                    self.warps = [
                        (
                            t,
                            (target if self.resolve(a, pa)[0] == target and self._door_at(target, self.resolve(a, pa)[1]) == aliased else a),
                            (canonical if self.resolve(a, pa)[0] == target and self._door_at(target, self.resolve(a, pa)[1]) == aliased else pa),
                            (target if self.resolve(b, pb)[0] == target and self._door_at(target, self.resolve(b, pb)[1]) == aliased else b),
                            (canonical if self.resolve(b, pb)[0] == target and self._door_at(target, self.resolve(b, pb)[1]) == aliased else pb),
                        )
                        for t, a, pa, b, pb in self.warps
                    ]
                    healed += 1
        return healed

    def _heal_unwarped_segment_splits(self) -> int:
        healed = 0
        if len(self.trail) < 2:
            return 0
        while True:
            split_idx = None
            for idx in range(len(self.trail) - 1, 0, -1):
                t_prev, s_prev, p_prev = self.trail[idx - 1]
                t_cur, s_cur, p_cur = self.trail[idx]
                if s_prev == s_cur:
                    continue
                rs_prev, rp_prev = self.resolve(s_prev, p_prev)
                rs_cur, rp_cur = self.resolve(s_cur, p_cur)
                if rs_prev == rs_cur:
                    continue
                has_warp = any(
                    t_prev <= wt <= t_cur
                    and (
                        self.resolve(wa, (0, 0))[0] in (rs_prev, rs_cur)
                        or self.resolve(wb, (0, 0))[0] in (rs_prev, rs_cur)
                    )
                    for wt, wa, _, wb, _ in self.warps
                )
                if has_warp:
                    continue
                if any(self.resolve(s, (0, 0))[0] != rs_cur for _, s, _ in self.trail[idx:]):
                    continue
                earlier_in_cur = [t for t, s, _ in self.trail[:idx] if self.resolve(s, (0, 0))[0] == rs_cur]
                if not earlier_in_cur:
                    continue
                base_ox = rp_prev[0] - rp_cur[0]
                base_oy = rp_prev[1] - rp_cur[1]
                seg_cur = self.segments[rs_cur]
                seg_prev = self.segments[rs_prev]
                if not seg_cur or not seg_prev:
                    continue
                new_cells = {g: c for g, c in seg_cur.items() if c.last_seen >= t_cur}
                best_off = None
                best_score = (-1, -1)
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if abs(dx) + abs(dy) > 2:
                            continue
                        ox, oy = base_ox + dx, base_oy + dy
                        m = tot = 0
                        for g, c in new_cells.items():
                            c2 = seg_prev.get((g[0] + ox, g[1] + oy))
                            if c2 is None or c.kind == OCCUPIED or c2.kind == OCCUPIED:
                                continue
                            u1 = len(set(c.hashes)) == 1
                            u2 = len(set(c2.hashes)) == 1
                            if u1 and u2:
                                continue
                            tot += 1
                            if not u1 and not u2 and _cells_match(c.hashes, c.pixels, c2):
                                m += 1
                        score = (m, -abs(dx) - abs(dy))
                        if tot >= 5 and m / tot >= 0.5 and score > best_score:
                            best_score = score
                            best_off = (ox, oy)
                if best_off is None:
                    continue
                split_idx = (idx, t_cur, rs_cur, rs_prev, best_off)
                break
            if split_idx is None:
                break
            idx, t_cur, rs_cur, rs_prev, (ox, oy) = split_idx
            seg_cur = self.segments[rs_cur]
            seg_prev = self.segments[rs_prev]
            for g, c in list(seg_cur.items()):
                if c.last_seen >= t_cur:
                    g_dst = (g[0] + ox, g[1] + oy)
                    if g_dst not in seg_prev:
                        seg_prev[g_dst] = c
                    if c.first_seen >= t_cur:
                        seg_cur.pop(g, None)
            post_pts = [self.resolve(s, p)[1] for _, s, p in self.trail[idx:] if self.resolve(s, (0, 0))[0] == rs_cur]
            post_filled = set()
            for i, p1 in enumerate(post_pts):
                post_filled.add(p1)
                if i + 1 < len(post_pts):
                    p2 = post_pts[i + 1]
                    if (p1[0] == p2[0]) != (p1[1] == p2[1]) and 1 < abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) <= MAX_STEP_FILL:
                        sx = (p2[0] > p1[0]) - (p2[0] < p1[0])
                        sy = (p2[1] > p1[1]) - (p2[1] < p1[1])
                        for k in range(1, abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])):
                            post_filled.add((p1[0] + sx * k, p1[1] + sy * k))
            pre_pts = [self.resolve(s, p)[1] for _, s, p in self.trail[:idx] if self.resolve(s, (0, 0))[0] == rs_cur]
            pre_visits: Counter = Counter()
            for i, p1 in enumerate(pre_pts):
                pre_visits[p1] += 1
                if i + 1 < len(pre_pts):
                    p2 = pre_pts[i + 1]
                    if (p1[0] == p2[0]) != (p1[1] == p2[1]) and 1 < abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) <= MAX_STEP_FILL:
                        sx = (p2[0] > p1[0]) - (p2[0] < p1[0])
                        sy = (p2[1] > p1[1]) - (p2[1] < p1[1])
                        for k in range(1, abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])):
                            mid = (p1[0] + sx * k, p1[1] + sy * k)
                            c_mid = seg_cur.get(mid)
                            if c_mid is None or c_mid.kind not in (BLOCKED, LEDGE, LEDGE_LEFT):
                                pre_visits[mid] += 1
            for g in post_filled:
                old_v = self.visits[rs_cur].get(g, 0)
                keep_v = pre_visits.get(g, 0)
                moved_v = max(0, old_v - keep_v)
                if keep_v > 0:
                    self.visits[rs_cur][g] = keep_v
                else:
                    self.visits[rs_cur].pop(g, None)
                if moved_v > 0:
                    self.visits[rs_prev][(g[0] + ox, g[1] + oy)] += moved_v
            warp_doors_cur = [
                self._door_at(rs_cur, self.resolve(s, p)[1])
                for _, wa, pa, wb, pb in self.warps
                for s, p in ((wa, pa), (wb, pb))
                if self.resolve(s, (0, 0))[0] == rs_cur
            ]
            ref_ground = next(
                (
                    mc for gp, mc in seg_cur.items()
                    if mc.first_seen < t_cur and mc.kind == WALKABLE and not is_door(mc) and pre_visits.get(gp, 0) > 0
                ),
                None,
            )
            for g, c in list(seg_cur.items()):
                if (
                    c.first_seen < t_cur <= c.last_seen
                    and is_door(c)
                    and pre_visits.get(g, 0) > 0
                    and not any(max(abs(g[0] - wd[0]), abs(g[1] - wd[1])) <= 2 for wd in warp_doors_cur)
                ):
                    seg_cur[g] = MapCell(
                        WALKABLE,
                        ("ground",),
                        ref_ground.hashes if ref_ground is not None else c.hashes,
                        c.first_seen,
                        c.last_seen,
                        ref_ground.pixels if ref_ground is not None else c.pixels,
                    )
            if rs_cur < len(self.blocked_moves) and self.blocked_moves[rs_cur]:
                while len(self.blocked_moves) <= rs_prev:
                    self.blocked_moves.append({})
                for b_pos, dirs in list(self.blocked_moves[rs_cur].items()):
                    if b_pos in post_filled and b_pos not in pre_visits:
                        self.blocked_moves[rs_prev].setdefault((b_pos[0] + ox, b_pos[1] + oy), set()).update(dirs)
                        self.blocked_moves[rs_cur].pop(b_pos, None)
            self.trail = [
                (t, rs_prev, (self.resolve(s, p)[1][0] + ox, self.resolve(s, p)[1][1] + oy))
                if (i >= idx and self.resolve(s, (0, 0))[0] == rs_cur)
                else (t, s, p)
                for i, (t, s, p) in enumerate(self.trail)
            ]
            if self.segment == rs_cur and self.pos is not None:
                self.segment = rs_prev
                self.pos = (self.pos[0] + ox, self.pos[1] + oy)
                if getattr(self, "_anchor_place", None) is not None and self._anchor_place[0] == rs_cur:
                    ap = self._anchor_place[1]
                    self._anchor_place = (rs_prev, (ap[0] + ox, ap[1] + oy))
            self.naming_queue = [e for e in self.naming_queue if e.get("segment") != rs_cur]
            info_cur = self.info(rs_cur)
            if info_cur.aliases:
                old_name, old_desc = info_cur.aliases.pop(0)
                info_cur.name, info_cur.description = old_name, old_desc
            if not self.frontier(rs_cur) and info_cur.explored is None:
                info_cur.explored = info_cur.episode
            self._seg_index_cache = {}
            self._overlap_cache = {}
            healed += 1
        return healed

    def _heal_post_defeat_shifts(self) -> int:
        if not self.defeats or not self.trail:
            return 0
        last_defeat = self.defeats[-1]
        d_step = int(last_defeat.get("step", 0))
        if d_step <= 0 or self.step - d_step > 600:
            return 0
        d_seg, d_pos = self.resolve(last_defeat["segment"], last_defeat["pos"])
        if not (0 <= d_seg < len(self.segments)) or not self.segments[d_seg]:
            return 0
        post_idx = next((i for i, (t, _, _) in enumerate(self.trail) if t >= d_step), None)
        if post_idx is None:
            return 0
        _, first_s, first_p = self.trail[post_idx]
        if self.resolve(first_s, first_p) != (d_seg, d_pos):
            return 0
        if any(self.resolve(s, (0, 0))[0] != d_seg for _, s, _ in self.trail[post_idx:]):
            return 0
        seg = self.segments[d_seg]
        post_cells = {g: c for g, c in seg.items() if c.last_seen >= d_step}
        pre_cells = {g: c for g, c in seg.items() if c.last_seen < d_step}
        if not post_cells or not pre_cells:
            return 0
        best_off = None
        best_score = (-1, 0.0)
        cand_offsets = set()
        post_doors = [g for g, c in post_cells.items() if is_door(c)]
        pre_doors = [g for g, c in pre_cells.items() if is_door(c)]
        for pd in post_doors:
            for qd in pre_doors:
                off = (qd[0] - pd[0], qd[1] - pd[1])
                if off != (0, 0):
                    cand_offsets.add(off)
        for ox in range(-25, 26):
            for oy in range(-15, 16):
                if (ox, oy) != (0, 0):
                    cand_offsets.add((ox, oy))
        for ox, oy in cand_offsets:
            respawn_pos = (d_pos[0] + ox, d_pos[1] + oy)
            has_respawn_door = any(
                (respawn_pos[0] + dx, respawn_pos[1] + dy) in pre_cells
                and is_door(pre_cells[(respawn_pos[0] + dx, respawn_pos[1] + dy)])
                for dx, dy in ((0, 1), (0, 0), (0, -1), (-1, 0), (1, 0))
            )
            if not has_respawn_door:
                continue
            m = tot = 0
            for g, c in post_cells.items():
                c2 = pre_cells.get((g[0] + ox, g[1] + oy))
                if c2 is None or c.kind == OCCUPIED or c2.kind == OCCUPIED:
                    continue
                u1 = len(set(c.hashes)) == 1
                u2 = len(set(c2.hashes)) == 1
                if u1 and u2:
                    continue
                tot += 1
                if not u1 and not u2 and _cells_match(c.hashes, c.pixels, c2):
                    m += 1
            if tot >= 15 and m / tot >= 0.70:
                score = (m, m / tot)
                if score > best_score:
                    best_score = score
                    best_off = (ox, oy)
        if best_off is None:
            return 0
        ox, oy = best_off
        post_snapshot = {
            g: MapCell(
                c.kind,
                c.tags,
                c.hashes,
                c.first_seen,
                c.last_seen,
                c.pixels.copy() if c.pixels is not None else None,
                getattr(c, "prior", None),
            )
            for g, c in post_cells.items()
        }
        trail_by_t = {t: self.resolve(s, p)[1] for t, s, p in self.trail[post_idx:] if self.resolve(s, (0, 0))[0] == d_seg}
        post_pts = [self.resolve(s, p)[1] for _, s, p in self.trail[post_idx:] if self.resolve(s, (0, 0))[0] == d_seg]
        post_filled = set()
        for i, p1 in enumerate(post_pts):
            post_filled.add(p1)
            if i + 1 < len(post_pts):
                p2 = post_pts[i + 1]
                if (p1[0] == p2[0]) != (p1[1] == p2[1]) and 1 < abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) <= MAX_STEP_FILL:
                    sx = (p2[0] > p1[0]) - (p2[0] < p1[0])
                    sy = (p2[1] > p1[1]) - (p2[1] < p1[1])
                    for k in range(1, abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])):
                        post_filled.add((p1[0] + sx * k, p1[1] + sy * k))
        pre_pts = [self.resolve(s, p)[1] for _, s, p in self.trail[:post_idx] if self.resolve(s, (0, 0))[0] == d_seg]
        pre_visits: Counter = Counter()
        for i, p1 in enumerate(pre_pts):
            pre_visits[p1] += 1
            if i + 1 < len(pre_pts):
                p2 = pre_pts[i + 1]
                if (p1[0] == p2[0]) != (p1[1] == p2[1]) and 1 < abs(p2[0] - p1[0]) + abs(p2[1] - p1[1]) <= MAX_STEP_FILL:
                    sx = (p2[0] > p1[0]) - (p2[0] < p1[0])
                    sy = (p2[1] > p1[1]) - (p2[1] < p1[1])
                    for k in range(1, abs(p2[0] - p1[0]) + abs(p2[1] - p1[1])):
                        mid = (p1[0] + sx * k, p1[1] + sy * k)
                        c_mid = seg.get(mid)
                        if c_mid is None or c_mid.kind not in (BLOCKED, LEDGE, LEDGE_LEFT):
                            pre_visits[mid] += 1
        for g in post_filled:
            old_v = self.visits[d_seg].get(g, 0)
            keep_v = pre_visits.get(g, 0)
            moved_v = max(0, old_v - keep_v)
            if keep_v > 0:
                self.visits[d_seg][g] = keep_v
            else:
                self.visits[d_seg].pop(g, None)
            if moved_v > 0:
                self.visits[d_seg][(g[0] + ox, g[1] + oy)] += moved_v
        for g, cnt in pre_visits.items():
            if cnt > self.visits[d_seg].get(g, 0):
                self.visits[d_seg][g] = cnt
        warp_doors = {
            self._door_at(d_seg, self.resolve(s, p)[1])
            for _, wa, pa, wb, pb in self.warps
            for s, p in ((wa, pa), (wb, pb))
            if self.resolve(s, (0, 0))[0] == d_seg
        }
        ref_ground = next(
            (mc for mc in pre_cells.values() if mc.kind == WALKABLE and not is_door(mc) and len(set(mc.hashes)) == 1),
            None,
        )
        ref_obstacle = next(
            (
                mc for mc in pre_cells.values()
                if mc.kind == BLOCKED
                and mc.tags == ("obstacle",)
                and not is_door(mc)
                and len(mc.hashes) == 4
                and None not in mc.hashes
            ),
            None,
        )
        for g, c_snap in list(post_snapshot.items()):
            p_last = trail_by_t.get(c_snap.last_seen, d_pos)
            g_src = (g[0] + ox, g[1] + oy)
            prior_snap = getattr(c_snap, "prior", None)
            if prior_snap is not None and int(prior_snap[4]) < d_step:
                pk, pt, ph, pfs, pls, ppix = prior_snap
                seg[g] = MapCell(pk, pt, ph, pfs, pls, ppix)
                continue
            if c_snap.first_seen >= d_step:
                seg.pop(g, None)
                continue
            if (g[0] - ox, g[1] - oy) not in post_snapshot and p_last != d_pos:
                seg.pop(g, None)
                continue
            if is_door(c_snap) and g not in warp_doors and g_src in seg and is_door(seg[g_src]):
                nb_obs = next(
                    (
                        seg[nb]
                        for dx, dy in ((-1, 0), (1, 0), (0, 1))
                        for nb in [(g[0] + dx, g[1] + dy)]
                        if nb in seg and seg[nb].kind == BLOCKED and "obstacle" in seg[nb].tags and not is_door(seg[nb])
                    ),
                    ref_obstacle,
                )
                seg[g] = MapCell(
                    BLOCKED,
                    ("obstacle",),
                    nb_obs.hashes if nb_obs is not None else (None, None, None, None),
                    c_snap.first_seen,
                    c_snap.last_seen,
                    nb_obs.pixels if nb_obs is not None else None,
                )
                continue
            if (
                self.visits[d_seg].get(g, 0) > 0
                and (c_snap.kind in (BLOCKED, OCCUPIED) or "sign" in c_snap.tags)
                and not (is_door(c_snap) and g in warp_doors)
            ):
                seg[g] = MapCell(
                    WALKABLE,
                    ("ground",),
                    ref_ground.hashes if ref_ground is not None else (None, None, None, None),
                    c_snap.first_seen,
                    c_snap.last_seen,
                    ref_ground.pixels if ref_ground is not None else None,
                )
        for g, c_snap in sorted(post_snapshot.items(), key=lambda kv: (kv[1].last_seen, kv[1].first_seen)):
            g_dst = (g[0] + ox, g[1] + oy)
            c_old = seg.get(g_dst)
            first_s = min(c_old.first_seen, c_snap.first_seen) if c_old is not None else c_snap.first_seen
            last_s = max(c_old.last_seen, c_snap.last_seen) if c_old is not None else c_snap.last_seen
            if c_old is not None and is_door(c_old) and g_dst in warp_doors and not is_door(c_snap):
                c_old.last_seen = last_s
                continue
            if self.visits[d_seg].get(g_dst, 0) > 0 and c_snap.kind in (OCCUPIED, BLOCKED) and not is_door(c_snap):
                if c_old is not None and c_old.kind == WALKABLE:
                    c_old.last_seen = last_s
                    continue
                seg[g_dst] = MapCell(
                    WALKABLE,
                    ("ground",),
                    ref_ground.hashes if ref_ground is not None else c_snap.hashes,
                    first_s,
                    last_s,
                    ref_ground.pixels if ref_ground is not None else c_snap.pixels,
                )
                continue
            seg[g_dst] = MapCell(c_snap.kind, c_snap.tags, c_snap.hashes, first_s, last_s, c_snap.pixels)
        self.trail = [
            (t, d_seg, (self.resolve(s, p)[1][0] + ox, self.resolve(s, p)[1][1] + oy))
            if (i >= post_idx and self.resolve(s, (0, 0))[0] == d_seg)
            else (t, s, p)
            for i, (t, s, p) in enumerate(self.trail)
        ]
        old_pos = self.pos
        if self.segment == d_seg and self.pos is not None:
            self.pos = (self.pos[0] + ox, self.pos[1] + oy)
            if getattr(self, "_anchor_place", None) is not None and self._anchor_place[0] == d_seg:
                ap = self._anchor_place[1]
                self._anchor_place = (d_seg, (ap[0] + ox, ap[1] + oy))
        if getattr(self, "battle_place", None) is not None:
            bs, bp = self.resolve(*self.battle_place)
            if bs == d_seg and bp != d_pos:
                self.battle_place = (d_seg, (bp[0] + ox, bp[1] + oy))
        self.__dict__.setdefault("healed_shifts", [])
        if old_pos is not None and self.pos is not None and old_pos != self.pos:
            self.healed_shifts.append({
                "segment": d_seg,
                "from_step": d_step,
                "to_step": self.step,
                "ox": ox,
                "oy": oy,
                "old_pos": old_pos,
                "new_pos": self.pos,
            })
        self._seg_index_cache = {}
        self._overlap_cache = {}
        return 1

    def consolidate(self, judge: Optional[Callable[[int, int, Dict[str, Any]], Optional[bool]]] = None
                    ) -> List[Dict[str, Any]]:
        self._heal_post_defeat_shifts()
        self._heal_unwarped_segment_splits()
        self.prune_transient_segments()
        self._heal_aliased_merged_doors()
        merged = []
        while True:
            live = [si for si, seg in enumerate(self.segments) if seg and si not in self.merged_into]
            best = None
            for i, target in enumerate(live):
                for source in live[i + 1:]:
                    o = self.overlap(source, target)
                    valid_o = (
                        o is not None
                        and o["matches"] >= CONSOLIDATE_MIN_CELLS
                        and o["agreement"] >= CONSOLIDATE_MIN_AGREEMENT
                        and (
                            o["votes"] >= CONSOLIDATE_MARGIN * o["rival"]
                            or (
                                o["votes"] >= CONSOLIDATE_MIN_VOTES
                                and o["agreement"] - o["rival_agreement"] >= CONSOLIDATE_AGREEMENT_LEAD
                            )
                        )
                    )
                    if not valid_o:
                        o = self._warp_anchored_overlap(source, target)
                        if o is None:
                            o = self._trail_anchored_overlap(source, target)
                        if o is None:
                            continue
                    if best is not None and o["matches"] <= best["matches"]:
                        continue
                    if self.merge_breaks_doors(source, target, o["offset"]):
                        self.stats["merge_door_vetoes"] += 1
                        continue
                    if not self._judged_same(source, target, o, judge):
                        continue
                    best = dict(o, source=source, target=target)
            if best is None:
                self._heal_aliased_merged_doors()
                return merged
            self._merge(best["source"], best["target"], best["offset"])
            self.stats["consolidated"] += 1
            merged.append(best)

    def relocalise(self, frame: np.ndarray, exclude: Optional[int] = None, shift: Cell = (0, 0),
                   allowed: Optional[Callable[[int, Cell], bool]] = None,
                   warp_targets: Optional[set] = None) -> Optional[Tuple[int, Cell]]:
        a = _screen(frame)
        masked_cells = {key[:2] for key in TileRecognizer.overlay_tile_keys(frame)} if shift == (0, 0) else set()
        local: Dict[Cell, Tuple[Optional[str], ...]] = {}
        local_pixels: Dict[Cell, np.ndarray] = {}
        local_ranks: Dict[Cell, Tuple[bytes, ...]] = {}
        blank_local: List[Cell] = []
        for (x, y), piece in _local_cells(a, shift):
            if (x, y) == (0, 0) or (x, y) in masked_cells:
                continue
            # blank cells match everywhere, so they do not vote
            if int(piece.min()) == int(piece.max()):
                blank_local.append((x, y))
                continue
            quarters = split_quarters(piece)
            local[(x, y)] = self._cell_hashes(quarters)
            local_pixels[(x, y)] = piece
            rank_sig = _quarter_rank_signature(piece)
            if rank_sig is not None:
                local_ranks[(x, y)] = rank_sig
        core_local = {
            k: v for k, v in local.items()
            if max(abs(k[0]), abs(k[1])) <= RELOCALISE_WARP_CORE_RADIUS
        }
        core_blank = [
            k for k in blank_local
            if max(abs(k[0]), abs(k[1])) <= RELOCALISE_WARP_CORE_RADIUS
        ]
        best = None
        for si, seg in enumerate(self.segments):
            if si == exclude or not seg:
                continue
            _, _, _, index, block_index, _ = self._segment_index(si)
            votes = self._collect_offset_votes(local, index, block_index)
            if local_ranks:
                rank_votes = self._collect_offset_votes(local_ranks, index, {})
                for off, cnt in rank_votes.items():
                    if cnt > votes[off]:
                        votes[off] = cnt
            seg_warps = [wp for w_si, wp in (warp_targets or ()) if w_si == si]
            for wp in seg_warps:
                if wp not in votes:
                    votes[wp] = 1
            if not votes:
                continue
            cand_offsets = [o for o, _ in votes.most_common(RELOCALISE_CANDIDATES)]
            for wp in seg_warps:
                if wp not in cand_offsets:
                    cand_offsets.append(wp)
            ranked = sorted(
                (self._screen_agreement(seg, local, o, blank_local, local_pixels), o)
                for o in cand_offsets
            )[::-1]
            best_m, best_ov = ranked[0][0]
            best_frac = best_m / max(1, best_ov)
            close = [
                (o, overlap, m)
                for (m, overlap), o in ranked
                if m >= RELOCALISE_CLOSE_CALL * best_m
                or (
                    seg_warps
                    and overlap > 0
                    and any(max(abs(o[0] - wp[0]), abs(o[1] - wp[1])) <= DOOR_REACH for wp in seg_warps)
                    and m / overlap >= RELOCALISE_CLOSE_CALL * best_frac
                )
            ]
            if len(close) > 1 and seg_warps:
                warp_scored = []
                for o, overlap, m in close:
                    if not any(max(abs(o[0] - wp[0]), abs(o[1] - wp[1])) <= DOOR_REACH for wp in seg_warps):
                        continue
                    cm, cov = self._screen_agreement(seg, core_local, o, core_blank, local_pixels)
                    if cov > 0 and cm >= RELOCALISE_MIN_FRACTION * cov:
                        min_dist = min(abs(o[0] - wp[0]) + abs(o[1] - wp[1]) for wp in seg_warps)
                        on_door = int(o in seg and is_door(seg[o]))
                        warp_scored.append(((on_door, -min_dist, cm / cov, m / max(1, overlap), m), o, overlap, m))
                if warp_scored:
                    warp_scored.sort(reverse=True)
                    _, w_off, w_ov, w_m = warp_scored[0]
                    close = [(w_off, w_ov, w_m)]
            if len(close) > 1:
                distinctive = sorted(
                    ((self._distinctive_matches(seg, index, local, o, block_index), o, overlap, m)
                     for o, overlap, m in close),
                    reverse=True,
                )
                if distinctive[0][0] - distinctive[1][0] >= DISTINCTIVE_LEAD:
                    close = [(distinctive[0][1], distinctive[0][2], distinctive[0][3])]
                else:
                    top_o, top_ov, top_m = close[0]
                    second_m = ranked[1][0][0] if len(ranked) > 1 else 0
                    lead_needed = max(6, int(0.08 * max(1, top_ov)))
                    if (
                        top_ov > 0
                        and top_m >= 0.92 * top_ov
                        and top_m - second_m >= lead_needed
                    ):
                        close = [(top_o, top_ov, top_m)]
                    elif (
                        not self.faded
                        and si == self.segment
                        and self.pos is not None
                    ):
                        near_cur = [
                            (abs(o[0] - self.pos[0]) + abs(o[1] - self.pos[1]), -m, -(m / overlap), o, overlap, m)
                            for o, overlap, m in close
                            if overlap > 0
                            and m >= 0.85 * overlap
                            and abs(o[0] - self.pos[0]) + abs(o[1] - self.pos[1]) <= MAX_STEP_FILL
                        ]
                        if near_cur:
                            near_cur.sort()
                            if len(near_cur) == 1 or near_cur[0][0] < near_cur[1][0] or near_cur[0][1] < near_cur[1][1]:
                                _, _, _, n_o, n_ov, n_m = near_cur[0]
                                close = [(n_o, n_ov, n_m)]
                            else:
                                continue
                        else:
                            continue
                    else:
                        continue
            offset, overlap, n = close[0]
            # a small room shows few content cells: ask for most of them rather than the fixed minimum
            needed = min(RELOCALISE_MIN_CELLS, max(10, int(0.6 * len(local))))
            min_fraction = RELOCALISE_MIN_FRACTION
            if seg_warps and any(max(abs(offset[0] - wp[0]), abs(offset[1] - wp[1])) <= DOOR_REACH for wp in seg_warps):
                cm, cov = self._screen_agreement(seg, core_local, offset, core_blank, local_pixels)
                if cov > 0 and cm >= RELOCALISE_MIN_FRACTION * cov:
                    min_fraction = RELOCALISE_WARP_MIN_FRACTION
            if n >= needed and n >= min_fraction * overlap:
                if self._blacks_out(frame, si, offset, shift):
                    continue
                if allowed is not None and not allowed(si, offset):
                    self.stats["relocalise_door_vetoes"] += 1
                    continue
                if best is None or n > best[0]:
                    best = (n, si, offset)
        return None if best is None else (best[1], best[2])

    def _placement_screen_agreement(
        self,
        frame: np.ndarray,
        segment: int,
        pos: Cell,
        shift: Cell = (0, 0),
    ) -> Tuple[int, int]:
        if not (0 <= segment < len(self.segments)) or not self.segments[segment]:
            return 0, 0
        seg = self.segments[segment]
        a_scr = _screen(frame)
        masked = {key[:2] for key in TileRecognizer.overlay_tile_keys(frame)} if shift == (0, 0) else set()
        local: Dict[Cell, Tuple[Optional[str], ...]] = {}
        lp: Dict[Cell, np.ndarray] = {}
        blank: List[Cell] = []
        for (lx, ly), piece in _local_cells(a_scr, shift):
            if (lx, ly) == (0, 0) or (lx, ly) in masked:
                continue
            if int(piece.min()) == int(piece.max()):
                blank.append((lx, ly))
                continue
            local[(lx, ly)] = self._cell_hashes(split_quarters(piece))
            lp[(lx, ly)] = piece
        return self._screen_agreement(seg, local, pos, blank, lp)

    @staticmethod
    def _screen_agreement(seg: Dict[Cell, MapCell], local: Dict[Cell, Tuple[Optional[str], ...]],
                          offset: Cell, blank_local: Iterable[Cell] = (),
                          local_pixels: Optional[Dict[Cell, np.ndarray]] = None) -> Tuple[int, int]:
        placed = [
            (seg.get((offset[0] + x, offset[1] + y)), h, (local_pixels or {}).get((x, y)))
            for (x, y), h in local.items()
        ]
        placed = [(c, h, p) for c, h, p in placed if c is not None and c.kind != OCCUPIED]
        blank_disagreements = sum(
            1 for (x, y) in blank_local
            if (c := seg.get((offset[0] + x, offset[1] + y))) is not None
            and c.kind != OCCUPIED
            and c.pixels is not None
            and int(c.pixels.min()) != int(c.pixels.max())
        )
        return sum(1 for c, h, p in placed if _cells_match(h, p, c)), len(placed) + blank_disagreements

    @staticmethod
    def _distinctive_matches(seg: Dict[Cell, MapCell], index: Dict[Tuple[Optional[str], ...], List[Cell]],
                             local: Dict[Cell, Tuple[Optional[str], ...]], offset: Cell,
                             block_index: Optional[Dict[Tuple[Any, ...], List[Cell]]] = None) -> int:
        single = sum(1 for (x, y), h in local.items()
                     if len(index.get(h, ())) <= DISTINCTIVE_MAX_COPIES
                     and (c := seg.get((offset[0] + x, offset[1] + y))) is not None and c.hashes == h)
        if block_index is None:
            return single
        blocks = sum(
            1 for (x, y), h in local.items()
            if (x + 1, y) in local and (x, y + 1) in local and (x + 1, y + 1) in local
            and len(block_index.get((h, local[(x + 1, y)], local[(x, y + 1)], local[(x + 1, y + 1)]), ()))
            <= DISTINCTIVE_MAX_COPIES
            and all((c := seg.get((offset[0] + x + dx, offset[1] + y + dy))) is not None
                    and c.hashes == local[(x + dx, y + dy)]
                    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)))
        )
        return single + blocks

    def _same_door(self, segment: int, p: Cell, q: Cell) -> bool:
        if max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= DOOR_REACH:
            return True
        if 0 <= segment < len(self.segments) and self.segments[segment]:
            dp, dq = self._door_at(segment, p), self._door_at(segment, q)
            return max(abs(dp[0] - dq[0]), abs(dp[1] - dq[1])) <= DOOR_REACH
        return False

    @staticmethod
    def door_conflicts(warps: List[Tuple[int, int, Cell, int, Cell]], moves: List[Optional[str]],
                       same_door: Optional[Callable[[int, Cell, Cell], bool]] = None) -> int:
        def near(s: int, p: Cell, q: Cell) -> bool:
            if same_door is not None:
                return same_door(s, p, q)
            return max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= DOOR_REACH

        conflicts, ends = 0, []
        for (_, a, pa, b, pb), move in zip(warps, moves):
            if a == b:
                conflicts += 1
                continue
            ends += [(a, pa, b, pb), (b, pb, a, pa)]
        for i, (s, p, other, q) in enumerate(ends):
            conflicts += any(s == s2 and near(s, p, p2) and (other != other2 or not near(other, q, q2))
                             for s2, p2, other2, q2 in ends[i + 1:])
        return conflicts

    def _is_doorless_inplace_warp(
        self,
        a: int,
        pa: Cell,
        b: int,
        pb: Cell,
        source: int,
        target: int,
        offset: Cell,
    ) -> bool:
        if {a, b} != {source, target}:
            return False
        dx, dy = offset
        src_p = pa if a == source else pb
        dst_p = pb if a == source else pa
        src_door = self._door_at(source, src_p)
        dst_door = self._door_at(target, dst_p)
        src_seg = self.segments[source] if 0 <= source < len(self.segments) else {}
        dst_seg = self.segments[target] if 0 <= target < len(self.segments) else {}
        src_has_door = src_door in src_seg and is_door(src_seg[src_door])
        dst_has_door = dst_door in dst_seg and is_door(dst_seg[dst_door])
        if src_has_door or dst_has_door:
            return False
        c_src = src_seg.get(src_p)
        c_dst = dst_seg.get(dst_p)
        if c_src is None or c_dst is None:
            return False
        if c_src.kind in (OCCUPIED, UNKNOWN) or c_dst.kind in (OCCUPIED, UNKNOWN):
            return False
        if "unknown" in c_src.tags or "unknown" in c_dst.tags:
            return False
        trans_src_p = (src_p[0] + dx, src_p[1] + dy)
        return max(abs(trans_src_p[0] - dst_p[0]), abs(trans_src_p[1] - dst_p[1])) <= MAX_STEP_FILL

    def _is_scripted_cutscene_warp(
        self,
        a: int,
        pa: Cell,
        b: int,
        pb: Cell,
        move: Optional[str],
        source: Optional[int] = None,
        target: Optional[int] = None,
        offset: Cell = (0, 0),
    ) -> bool:
        if move in DIRECTIONS:
            return False
        ra, rpa = self.resolve(a, pa)
        if source is not None and target is not None and ra == source:
            ra, rpa = target, (rpa[0] + offset[0], rpa[1] + offset[1])
        rb, rpb = self.resolve(b, pb)
        if source is not None and target is not None and rb == source:
            rb, rpb = target, (rpb[0] + offset[0], rpb[1] + offset[1])
        if ra == rb or not self._has_min_observations(ra):
            return False
        seg_a = self.segments[ra] if 0 <= ra < len(self.segments) else {}
        da = self._door_at(ra, rpa)
        ca = seg_a.get(da)
        if ca is None or is_door(ca) or ca.kind in (OCCUPIED, UNKNOWN):
            return False
        for (_, a2, pa2, b2, pb2), m2 in zip(self.warps, self.warp_moves):
            if (a2, pa2, b2, pb2, m2) == (a, pa, b, pb, move):
                continue
            ra2, rpa2 = self.resolve(a2, pa2)
            if source is not None and target is not None and ra2 == source:
                ra2, rpa2 = target, (rpa2[0] + offset[0], rpa2[1] + offset[1])
            rb2, rpb2 = self.resolve(b2, pb2)
            if source is not None and target is not None and rb2 == source:
                rb2, rpb2 = target, (rpb2[0] + offset[0], rpb2[1] + offset[1])
            if ra2 == rb2:
                continue
            if (ra2 == ra and self._same_door(ra, rpa, rpa2)) or (rb2 == ra and self._same_door(ra, rpa, rpb2)):
                return False
        seg_b = self.segments[rb] if 0 <= rb < len(self.segments) else {}
        cb = seg_b.get(rpb)
        if cb is not None and is_door(cb):
            return False
        near_b_door = any(
            (rpb[0] + ox, rpb[1] + oy) in seg_b and is_door(seg_b[(rpb[0] + ox, rpb[1] + oy)])
            for ox in (-1, 0, 1)
            for oy in (-1, 0, 1)
        )
        return not near_b_door

    def _is_round_trip_same_door_warp(
        self,
        a: int,
        pa: Cell,
        b: int,
        pb: Cell,
        move: Optional[str],
        source: int,
        target: int,
        offset: Cell,
    ) -> bool:
        if {a, b} != {source, target}:
            return False
        if self._is_doorless_inplace_warp(a, pa, b, pb, source, target, offset):
            return True
        dx, dy = offset
        src_p = pa if a == source else pb
        dst_p = pb if a == source else pa
        src_door = self._door_at(source, src_p)
        dst_door = self._door_at(target, dst_p)
        src_seg = self.segments[source] if 0 <= source < len(self.segments) else {}
        dst_seg = self.segments[target] if 0 <= target < len(self.segments) else {}
        src_has_door = src_door in src_seg and is_door(src_seg[src_door])
        dst_has_door = dst_door in dst_seg and is_door(dst_seg[dst_door])
        if not src_has_door and move in DIRECTIONS and a == source:
            mdx, mdy = DIRECTIONS[move]
            for k in range(1, MAX_STEP_FILL + 1):
                cand = (src_p[0] + mdx * k, src_p[1] + mdy * k)
                if cand in src_seg and is_door(src_seg[cand]):
                    src_door = cand
                    src_has_door = True
                    break
        trans_src_door = (src_door[0] + dx, src_door[1] + dy)
        if not dst_has_door and trans_src_door in dst_seg and is_door(dst_seg[trans_src_door]):
            if max(abs(dst_p[0] - trans_src_door[0]), abs(dst_p[1] - trans_src_door[1])) <= 2:
                dst_door = trans_src_door
                dst_has_door = True
        if not (src_has_door and dst_has_door):
            return False
        return max(abs(trans_src_door[0] - dst_door[0]), abs(trans_src_door[1] - dst_door[1])) <= DOOR_REACH

    def merge_breaks_doors(self, source: int, target: int, offset: Cell) -> bool:
        def moved(si: int, p: Cell) -> Tuple[int, Cell]:
            return (target, (p[0] + offset[0], p[1] + offset[1])) if si == source else (si, p)

        src_seg = self.segments[source] if 0 <= source < len(self.segments) else {}
        dst_seg = self.segments[target] if 0 <= target < len(self.segments) else {}
        for g, c_src in src_seg.items():
            if not is_door(c_src):
                continue
            g_dst = (g[0] + offset[0], g[1] + offset[1])
            c_dst = dst_seg.get(g_dst)
            if c_dst is not None and c_dst.kind != OCCUPIED and not is_door(c_dst):
                snapped = self._door_at(target, g_dst)
                if snapped != g_dst and max(abs(snapped[0] - g_dst[0]), abs(snapped[1] - g_dst[1])) > DOOR_REACH:
                    return True

        after: List[Tuple[int, int, Cell, int, Cell]] = []
        after_moves: List[Optional[str]] = []
        before_warps: List[Tuple[int, int, Cell, int, Cell]] = []
        for (t, a, pa, b, pb), m in zip(self.warps, self.warp_moves):
            if self._is_round_trip_same_door_warp(a, pa, b, pb, m, source, target, offset):
                continue
            if self._is_scripted_cutscene_warp(a, pa, b, pb, m, source, target, offset):
                continue
            after.append((t, *moved(a, pa), *moved(b, pb)))
            after_moves.append(m)
            before_warps.append((t, a, pa, b, pb))

        overlap_cache: Dict[Tuple[int, int], Optional[Dict[str, Any]]] = {}

        def coupled_same(other1: int, q1: Cell, other2: int, q2: Cell) -> bool:
            if other1 == other2:
                return self._same_door(other1, q1, q2)
            if self.names_differ(other1, other2):
                return False
            cache_key = (other1, other2)
            if cache_key not in overlap_cache:
                overlap_cache[cache_key] = self.overlap(other1, other2)
            ov = overlap_cache[cache_key]
            if (
                ov is not None
                and ov["matches"] >= CONSOLIDATE_MIN_CELLS
                and ov["agreement"] >= CONSOLIDATE_MIN_AGREEMENT
            ):
                dq1 = self._door_at(other1, q1)
                trans_q1 = (dq1[0] + ov["offset"][0], dq1[1] + ov["offset"][1])
                if self._same_door(other2, trans_q1, q2):
                    return True
            return False

        before_conflicts, ends = 0, []
        for _, a, pa, b, pb in before_warps:
            if a == b:
                before_conflicts += 1
                continue
            ends += [(a, pa, b, pb), (b, pb, a, pa)]
        for i, (s, p, other, q) in enumerate(ends):
            before_conflicts += any(
                s == s2
                and self._same_door(s, p, p2)
                and {other, other2} != {source, target}
                and not coupled_same(other, q, other2, q2)
                for s2, p2, other2, q2 in ends[i + 1:]
            )
        after_conflicts, after_ends = 0, []
        for _, a, pa, b, pb in after:
            if a == b:
                after_conflicts += 1
                continue
            after_ends += [(a, pa, b, pb), (b, pb, a, pa)]
        for i, (s, p, other, q) in enumerate(after_ends):
            after_conflicts += any(
                s == s2
                and self._same_door(s, p, p2)
                and not coupled_same(other, q, other2, q2)
                for s2, p2, other2, q2 in after_ends[i + 1:]
            )
        return after_conflicts > before_conflicts

    def arrival_breaks_doors(self, segment: int, pos: Cell) -> bool:
        if not self.faded:
            return False
        if self.in_battle and self.battle_place is None:
            return False
        move = self.fade_direction if self.fade_direction in DIRECTIONS else None
        if self._is_scripted_cutscene_warp(self.segment, self.pos, segment, pos, move):
            return False
        cur_seg, _ = self.resolve(self.segment, self.pos)
        dst_seg, dst_pos = self.resolve(segment, pos)
        if (
            cur_seg != dst_seg
            and self._has_min_observations(cur_seg)
            and self.info(dst_seg).kind != OUTDOORS
        ):
            doors = self.door_list(dst_seg)
            if doors and not any(
                max(abs(dst_pos[0] - c[0]), abs(dst_pos[1] - c[1])) <= DOOR_REACH
                for d in doors
                for c in d["cells"]
            ):
                return True
        base_warps: List[Tuple[int, int, Cell, int, Cell]] = []
        base_moves: List[Optional[str]] = []
        for (t, a, pa, b, pb), m in zip(self.warps, self.warp_moves):
            if self._is_scripted_cutscene_warp(a, pa, b, pb, m):
                continue
            base_warps.append((t, a, pa, b, pb))
            base_moves.append(m)
        after = base_warps + [(self.step, self.segment, self.pos, segment, pos)]
        return (self.door_conflicts(after, base_moves + [move], self._same_door)
                > self.door_conflicts(base_warps, base_moves, self._same_door))

    def refresh(self, tile_lookup: Dict[str, Any]) -> int:
        self._tile_lookup = tile_lookup
        self.prune_transient_segments()
        def _pattern_learnable(pattern: Tuple[Optional[str], ...], cell_pixels: Optional[np.ndarray] = None) -> bool:
            if not pattern or len(pattern) != 4 or None in pattern:
                return False
            tiles = [tile_lookup.get(h) for h in pattern]
            if not all(t is not None for t in tiles):
                return True
            raw_qtags = {
                q: normalize_tabletop_tag(t.tag, t.description, getattr(t, "image", None))
                for q, t in zip(QUARTERS, tiles)
            }
            qtags = filter_cell_warp_quarter_tags(raw_qtags)
            qdescs = {q: t.description for q, t in zip(QUARTERS, tiles)}
            qimgs = {q: getattr(t, "image", None) for q, t in zip(QUARTERS, tiles)}
            tag_set = set(qtags.values())
            t_img = cell_pixels if cell_pixels is not None else getattr(tiles[0], "image", None)
            return self._can_learn_walkable(
                tag_set,
                hashes=pattern,
                pixels=t_img,
                quarter_tags=qtags,
                quarter_descriptions=qdescs,
                quarter_images=qimgs,
            )

        if self.walk_patterns:
            self.walk_patterns = {
                pattern for pattern in self.walk_patterns if _pattern_learnable(pattern)
            }
        for si, seg in enumerate(self.segments):
            if not seg or si >= len(self.visits) or not self.visits[si]:
                continue
            seg_blocked = self.blocked_cells[si] if si < len(self.blocked_cells) else set()
            visited_by_pattern: Dict[Tuple[Optional[str], ...], List[MapCell]] = defaultdict(list)
            for g, cnt in self.visits[si].items():
                if cnt <= 0 or g in seg_blocked:
                    continue
                c = seg.get(g)
                if (
                    c is not None
                    and c.hashes
                    and len(c.hashes) == 4
                    and None not in c.hashes
                ):
                    visited_by_pattern[c.hashes].append(c)
            for pat, cells_for_pat in visited_by_pattern.items():
                if len(cells_for_pat) >= 2 and _pattern_learnable(pat, cells_for_pat[0].pixels):
                    self._learn_walkable(cells_for_pat[0])
        while len(self.blocked_cells) < len(self.segments):
            self.blocked_cells.append(set())
        while len(self.blocked_moves) < len(self.segments):
            self.blocked_moves.append({})
        changed = 0
        for si, seg in enumerate(self.segments):
            if not seg:
                continue
            seg_blocked = self.blocked_cells[si]
            if self.info(si).kind == OUTDOORS:
                for g in list(seg):
                    c = seg[g]
                    if (
                        c.pixels is not None
                        and int(c.pixels.min()) == int(c.pixels.max())
                        and (c.kind != WALKABLE or int(c.pixels.max()) <= BLACK_LEVEL)
                    ):
                        del seg[g]
                        seg_blocked.discard(g)
                        changed += 1
                changed += self._heal_visited_ground(si)
            quarter_tags: Dict[Cell, Dict[str, str]] = {}
            quarter_descriptions: Dict[Cell, Dict[str, Optional[str]]] = {}
            quarter_images: Dict[Cell, Dict[str, Optional[np.ndarray]]] = {}
            for g, c in seg.items():
                tiles = [tile_lookup.get(h) for h in c.hashes]
                if all(t is not None for t in tiles) or ("unknown" in c.tags and any(t is not None for t in tiles)):
                    raw_qtags = {
                        q: (
                            normalize_tabletop_tag(t.tag, t.description, getattr(t, "image", None))
                            if t is not None
                            else "unknown"
                        )
                        for q, t in zip(QUARTERS, tiles)
                    }
                    quarter_tags[g] = filter_cell_warp_quarter_tags(raw_qtags)
                    quarter_descriptions[g] = {
                        q: (t.description if t is not None else None) for q, t in zip(QUARTERS, tiles)
                    }
                    quarter_images[g] = {
                        q: (getattr(t, "image", None) if t is not None else None) for q, t in zip(QUARTERS, tiles)
                    }
            seg_hashes_by_cell = {
                g: {q: h for q, h in zip(QUARTERS, seg[g].hashes or ())}
                for g in quarter_tags
            }
            seg_patterned_floor_cells = infer_spatial_patterned_floor_cells(
                quarter_tags, quarter_descriptions, quarter_images, seg_hashes_by_cell
            )
            for g, c in seg.items():
                if g not in quarter_tags or (c.kind == OCCUPIED and "player" in c.tags):
                    continue
                new_kind, new_tags = classify_cell_from_quarters(
                    g,
                    quarter_tags,
                    quarter_descriptions,
                    quarter_images,
                    hashes=c.hashes,
                    patterned_floor_cells=seg_patterned_floor_cells,
                    ledge_patterns=self.ledge_patterns,
                    left_ledge_patterns=self.left_ledge_patterns,
                    walk_patterns=self.walk_patterns,
                    allow_player_overhang=False,
                    allow_player_cap=False,
                )
                learnable = _pattern_learnable(c.hashes, c.pixels)
                if g in seg_blocked:
                    if (
                        c.hashes
                        and len(c.hashes) == 4
                        and len(set(c.hashes)) == 1
                        and c.hashes[0] is not None
                        and c.hashes in self.walk_patterns
                        and learnable
                    ):
                        seg_blocked.discard(g)
                    elif new_kind == WALKABLE and not bool(set(new_tags) & DOOR_TAGS):
                        new_kind = BLOCKED
                if (
                    si < len(self.visits)
                    and self.visits[si].get(g, 0) > 0
                    and g not in seg_blocked
                    and new_kind in (BLOCKED, UNKNOWN)
                    and not is_door(c)
                    and not bool(set(new_tags) & DOOR_TAGS)
                    and (learnable or (new_kind == UNKNOWN and "ground" in new_tags))
                    and (c.kind == WALKABLE or "ground" in c.tags or "ground" in new_tags)
                    and not (set(new_tags) & {"sign", "interactable object", "pc", "water", "ledge", "right_ledge", "left_ledge"})
                ):
                    new_kind = WALKABLE
                    if "obstacle" in new_tags or "unknown" in new_tags:
                        new_tags = tuple(t for t in new_tags if t not in ("obstacle", "unknown")) or ("ground",)
                elif (
                    si < len(self.visits)
                    and self.visits[si].get(g, 0) > 0
                    and new_kind == BLOCKED
                    and "obstacle" in new_tags
                    and not learnable
                ):
                    self.visits[si].pop(g, None)
                if (c.kind, c.tags) != (new_kind, new_tags):
                    c.kind, c.tags = new_kind, new_tags
                    changed += 1
            if self.info(si).kind == OUTDOORS:
                changed += self._heal_visited_ground(si)
        merged = self.consolidate(judge=None)
        if merged:
            changed += len(merged)
        if self.seen_entrances is not None:
            valid = {(si, g) for si, seg in enumerate(self.segments)
                     if si not in self.merged_into and self.info(si).kind in (OUTDOORS, None)
                     for g, c in seg.items() if "entrance" in c.tags}
            self.seen_entrances = {key for key in self.seen_entrances if self.resolve(*key) in valid}
        return changed

    @update
    def end_episode(self, episode: int,
                    judge: Optional[Callable[[int, int, Dict[str, Any]], Optional[bool]]] = None) -> int:
        self.consolidate(judge)
        self.mark_explored(episode)
        self.new_cells_by_episode[episode] = self._new_this_episode
        self._new_this_episode = 0
        return self.new_cells_by_episode[episode]

    def _door_at(self, segment: int, pos: Cell) -> Cell:
        seg = self.segments[segment] if 0 <= segment < len(self.segments) else {}
        if not seg:
            return pos
        near = [(max(abs(ox), abs(oy)), abs(ox) + abs(oy), g) for ox, oy in NEIGHBOURS_8 + ((0, 0),)
                for g in [(pos[0] + ox, pos[1] + oy)] if g in seg and is_door(seg[g])]
        if near:
            return min(near)[2]
        cand = [
            (max(abs(ox), abs(oy)), abs(ox) + abs(oy), (pos[0] + ox, pos[1] + oy))
            for ox in range(-DOOR_SNAP_REACH, DOOR_SNAP_REACH + 1)
            for oy in range(-DOOR_SNAP_REACH, DOOR_SNAP_REACH + 1)
            if (pos[0] + ox, pos[1] + oy) in seg and is_door(seg[(pos[0] + ox, pos[1] + oy)])
        ]
        if not cand:
            return pos
        doors_in_range = {g for _, _, g in cand}
        seed = min(cand)[2]
        comp, stack = set(), [seed]
        while stack:
            h = stack.pop()
            if h in comp:
                continue
            comp.add(h)
            for ox, oy in NEIGHBOURS_8:
                nb = (h[0] + ox, h[1] + oy)
                if nb in seg and is_door(seg[nb]) and nb not in comp:
                    stack.append(nb)
        return seed if doors_in_range <= comp else pos

    def door_labels(self) -> Dict[Tuple[int, Cell], int]:
        labels: Dict[Tuple[int, Cell], int] = {}
        for idx, (_, a, pa, b, pb) in enumerate(self.warps):
            m = self.warp_moves[idx] if idx < len(self.warp_moves) else None
            if self._is_scripted_cutscene_warp(a, pa, b, pb, m):
                continue
            ra, rpa = self.resolve(a, pa)
            rb, rpb = self.resolve(b, pb)
            if ra == rb:
                continue
            da, db = self._door_at(ra, rpa), self._door_at(rb, rpb)
            seg_a = self.segments[ra] if 0 <= ra < len(self.segments) else {}
            seg_b = self.segments[rb] if 0 <= rb < len(self.segments) else {}
            if da == rpa and da in seg_a and not is_door(seg_a[da]):
                same_a = [
                    p for (s, p), dst in labels.items()
                    if s == ra and dst == rb and p in seg_a and is_door(seg_a[p])
                    and max(abs(p[0] - da[0]), abs(p[1] - da[1])) <= DOOR_SNAP_REACH
                ]
                if len(same_a) == 1:
                    da = same_a[0]
            if db == rpb and db in seg_b and not is_door(seg_b[db]):
                same_b = [
                    p for (s, p), dst in labels.items()
                    if s == rb and dst == ra and p in seg_b and is_door(seg_b[p])
                    and max(abs(p[0] - db[0]), abs(p[1] - db[1])) <= DOOR_SNAP_REACH
                ]
                if len(same_b) == 1:
                    db = same_b[0]
            labels[(ra, da)] = rb
            labels[(rb, db)] = ra
        for i in range(len(self.trail) - 1):
            _, s1, p1 = self.trail[i]
            _, s2, p2 = self.trail[i + 1]
            rs1, rp1 = self.resolve(s1, p1)
            rs2, rp2 = self.resolve(s2, p2)
            if rs1 == rs2 and self.info(rs1).kind != OUTDOORS and abs(rp1[0] - rp2[0]) + abs(rp1[1] - rp2[1]) > MAX_STEP_FILL:
                seg = self.segments[rs1] if 0 <= rs1 < len(self.segments) else {}
                d1, d2 = self._door_at(rs1, rp1), self._door_at(rs1, rp2)
                if d1 != d2 and d1 in seg and d2 in seg and is_door(seg[d1]) and is_door(seg[d2]):
                    labels.setdefault((rs1, d1), rs1)
                    labels.setdefault((rs1, d2), rs1)
        return labels

    def area_label(self, segment: int) -> str:
        name = self.info(segment).name
        return name if name else f"area #{segment} (not named yet)"

    def side_gatehouse_doorways(self, segment: Optional[int] = None) -> List[Dict[str, Any]]:
        si = self.segment if segment is None else segment
        if not (0 <= si < len(self.segments)):
            return []
        seg = self.segments[si]
        if not seg or self.info(si).kind != OUTDOORS:
            return []
        blocked_targets = set(self.blocked_cells[si]) if si < len(self.blocked_cells) else set()
        seg_moves = self.segment_blocked_moves(si)
        doorways: List[Dict[str, Any]] = []
        seen_road: set = set()
        for x0, y0 in sorted(seg):
            block = [(x0 + dx, y0 + dy) for dx in range(4) for dy in range(4)]
            if not all(b in seg and seg[b].kind == BLOCKED and seg[b].tags == ("obstacle",) for b in block):
                continue
            if any(len(seg[b].hashes) != 4 or None in seg[b].hashes or len(set(seg[b].hashes)) == 1 for b in block):
                continue
            if not all(seg[(x0 + 1, y0 + dy)].hashes == seg[(x0 + 2, y0 + dy)].hashes for dy in (0, 2, 3)):
                continue
            if len({seg[(x0 + 1, y0 + dy)].hashes for dy in range(4)}) != 4:
                continue
            if any(seg[(x0, y0 + dy)].hashes == seg[(x0 + 1, y0 + dy)].hashes for dy in range(4)):
                continue
            if any(seg[(x0 + 3, y0 + dy)].hashes == seg[(x0 + 1, y0 + dy)].hashes for dy in range(4)):
                continue
            for wall_x, road_x, step_dir in ((x0 + 3, x0 + 4, "left"), (x0, x0 - 1, "right")):
                road_cells = [(road_x, y0), (road_x, y0 + 1)]
                wall_cells = [(wall_x, y0), (wall_x, y0 + 1)]
                if not all(rc in seg and seg[rc].kind == WALKABLE and rc not in blocked_targets for rc in road_cells):
                    continue
                if all(step_dir in seg_moves.get(rc, ()) for rc in road_cells):
                    continue
                seen_road.add(tuple(road_cells))
                doorways.append({
                    "road_cells": road_cells,
                    "wall_cells": wall_cells,
                    "direction": step_dir,
                })
        for wx, y0 in sorted(seg):
            col = [(wx, y0 + dy) for dy in range(4)]
            if not all(b in seg and seg[b].kind == BLOCKED and seg[b].tags == ("obstacle",) for b in col):
                continue
            if any(len(seg[b].hashes) != 4 or None in seg[b].hashes or len(set(seg[b].hashes)) == 1 for b in col):
                continue
            h0, h1, h2, h3 = [seg[(wx, y0 + dy)].hashes for dy in range(4)]
            if len({h0, h1, h2, h3}) != 4 or h3[2:4] != h2[0:2]:
                continue
            for side, road_x, inner_step, step_dir in (("right_wall", wx + 1, -1, "left"), ("left_wall", wx - 1, 1, "right")):
                road_cells = [(road_x, y0), (road_x, y0 + 1)]
                if tuple(road_cells) in seen_road:
                    continue
                wall_cells = [(wx, y0), (wx, y0 + 1)]
                if not all(rc in seg and seg[rc].kind == WALKABLE and rc not in blocked_targets for rc in road_cells):
                    continue
                if all(step_dir in seg_moves.get(rc, ()) for rc in road_cells):
                    continue
                used_h0_door = False
                if side == "right_wall":
                    if h0[0] == h1[0] and h0[1] == h1[1] == h1[3] and h0[0] != h0[1] and h1[2] != h1[0]:
                        pass
                    elif (
                        h0[0] == h1[0] == h1[2]
                        and h0[1] == h1[1] == h1[3]
                        and h0[0] != h0[1]
                        and h0[2] != h0[0]
                        and h0[3] != h0[1]
                    ):
                        used_h0_door = True
                    else:
                        continue
                else:
                    if h0[1] == h1[1] and h0[0] == h1[0] == h1[2] and h0[0] != h0[1] and h1[3] != h1[1]:
                        pass
                    elif (
                        h0[0] == h1[0] == h1[2]
                        and h0[1] == h1[1] == h1[3]
                        and h0[0] != h0[1]
                        and h0[2] != h0[0]
                        and h0[3] != h0[1]
                    ):
                        used_h0_door = True
                    else:
                        continue
                inner_block = [(wx + inner_step * k, y0 + dy) for k in range(1, 4) for dy in range(4)]
                if not any(b in seg for b in inner_block):
                    continue
                if not used_h0_door and all(b in seg for b in inner_block):
                    continue
                if any(
                    b in seg
                    and (
                        (not used_h0_door and is_door(seg[b]))
                        or (
                            not (seg[b].kind == BLOCKED and seg[b].tags == ("obstacle",))
                            and not (used_h0_door and b[1] == y0 and is_door(seg[b]))
                        )
                    )
                    for b in inner_block
                ):
                    continue
                seen_road.add(tuple(road_cells))
                doorways.append({
                    "road_cells": road_cells,
                    "wall_cells": wall_cells,
                    "direction": step_dir,
                })
        return doorways

    def side_gatehouse_step_direction(self, cell: Cell, segment: Optional[int] = None) -> Optional[str]:
        si = self.segment if segment is None else segment
        for dw in self.side_gatehouse_doorways(si):
            if cell in dw["road_cells"] or cell in dw["wall_cells"]:
                return dw["direction"]
        return None

    def door_list(self, segment: Optional[int] = None) -> List[Dict[str, Any]]:
        si = self.segment if segment is None else segment
        seg = self.segments[si]
        labels = {p: b for (s, p), b in self.door_labels().items() if s == si}
        order = {key: i for i, key in enumerate(labels)}
        side_door_cells: set = set()
        for dw in self.side_gatehouse_doorways(si):
            if not any(rc in labels or wc in labels for rc in dw["road_cells"] for wc in dw["wall_cells"]):
                side_door_cells.update(dw["road_cells"])
        cells = {g for g, c in seg.items() if is_door(c)} | set(labels) | side_door_cells
        out, seen = [], set()
        for g in sorted(cells):
            if g in seen:
                continue
            group, stack = [], [g]
            while stack:
                h = stack.pop()
                if h in seen or h not in cells:
                    continue
                seen.add(h)
                group.append(h)
                stack += [(h[0] + ox, h[1] + oy) for ox, oy in NEIGHBOURS_8]
            known = sorted((order[h], labels[h]) for h in group if h in labels)
            out.append({"cells": sorted(group), "leads_to": known[-1][1] if known else None})
        return out

    def resolve(self, segment: int, pos: Cell) -> Tuple[int, Cell]:
        while segment in self.merged_into:
            segment, (dx, dy) = self.merged_into[segment]
            pos = (pos[0] + dx, pos[1] + dy)
        return segment, pos

    def diff(self, old: "GlobalMap") -> Dict[str, Any]:
        known: Dict[Tuple[int, Cell], MapCell] = {}
        for si, seg in enumerate(old.segments):
            for g, c in seg.items():
                known[self.resolve(si, g)] = c
        new_cells: Dict[int, List[Cell]] = defaultdict(list)
        reclassified: Dict[int, List[Cell]] = defaultdict(list)
        for si, seg in enumerate(self.segments):
            for g, c in seg.items():
                before = known.get((si, g))
                if before is None:
                    new_cells[si].append(g)
                elif before.kind == UNKNOWN and c.kind != UNKNOWN:
                    reclassified[si].append(g)
        return {
            "cells": self.n_cells - old.n_cells,
            "segments": len(self.segments) - len(old.segments),
            "new_cells": dict(new_cells),
            "reclassified": dict(reclassified),
            "opened": list(range(len(old.segments), len(self.segments))),
            "merged": {s: m for s, m in self.merged_into.items() if s not in old.merged_into},
            "steps": self.step - old.step,
            "position": ((old.segment, old.pos), (self.segment, self.pos)),
            "exploring": (old.exploring_until, self.exploring_until),
            "stats": {k: self.stats[k] - old.stats[k] for k in set(self.stats) | set(old.stats)
                      if self.stats[k] != old.stats[k]},
            "doors": self._new_doors(old),
        }

    def record_defeat(self, place: Optional[Tuple[int, Cell]] = None) -> None:
        if place is None:
            place = self.battle_place if self.battle_place is not None else (self.segment, self.pos)
        seg, pos = self.resolve(place[0], place[1])
        if self.defeats:
            last = self.defeats[-1]
            last_seg, last_pos = self.resolve(last["segment"], last["pos"])
            if (last_seg, last_pos) == (seg, pos) and last.get("step", 0) >= max(1, self.battle_start_step):
                return
        if self.battle_start_step > 0 and self.warps:
            kept_warps: List[Tuple[int, int, Cell, int, Cell]] = []
            kept_moves: List[Optional[str]] = []
            for w, m in zip(self.warps, self.warp_moves):
                t, a, pa, b, pb = w
                ra, rpa = self.resolve(a, pa)
                if t >= self.battle_start_step and (ra, rpa) == (seg, pos):
                    continue
                kept_warps.append(w)
                kept_moves.append(m)
            self.warps, self.warp_moves = kept_warps, kept_moves
        self.defeats.append({"step": self.step, "segment": seg, "pos": pos})
        self.stats["defeats"] += 1
        self.faded = True
        self.fade_direction = None
        self.battle_place = None
        self.pending = None

    def defeat_route(self, defeat: Dict[str, Any]) -> str:
        from .routes import plan_route, render_plan
        segment, pos = self.resolve(defeat["segment"], defeat["pos"])
        where = f"{self.area_label(segment)}, at {pos} on that area's map"
        if segment == self.segment:
            dx, dy = pos[0] - self.pos[0], pos[1] - self.pos[1]
            return f"{where}: you are in that area now; it is at ({dx}, {dy}) from you (x = steps right, y = steps up)."
        return f"{where}.\n" + render_plan(self, [segment], plan_route(self, [segment]))

    def entrances(self) -> set:
        return {(si, g) for si, seg in enumerate(self.segments)
                if si not in self.merged_into and self.info(si).kind == OUTDOORS
                for g, c in seg.items() if "entrance" in c.tags}

    def new_entrances(self, seen_towns: Sequence[str] = ()) -> int:
        if self.seen_entrances is None or not self.segments:
            return 0
        si = self.resolve(self.segment, (0, 0))[0]
        info = self.info(si)
        if info.kind != OUTDOORS or not self._segment_may_have_unseen_town(info.name, seen_towns):
            return 0
        doors = self.door_list(si)
        if len(doors) < 2:
            return 0
        seen_here = {pos for seg_id, pos in (self.resolve(*key) for key in self.seen_entrances) if seg_id == si}
        new_groups = 0
        for door in doors:
            if door["leads_to"] is not None:
                continue
            cells = [c for c in door["cells"] if c in self.segments[si] and "entrance" in self.segments[si][c].tags]
            if not cells:
                continue
            if any(max(abs(c[0] - s[0]), abs(c[1] - s[1])) <= 1 for c in cells for s in seen_here):
                continue
            new_groups += 1
        return new_groups

    @staticmethod
    def _segment_may_have_unseen_town(name: Optional[str], seen_towns: Sequence[str]) -> bool:
        if not name or not name.strip():
            return True
        known = {t.strip().lower() for t in seen_towns if t and t.strip()}
        if not known:
            return True
        name_l = name.strip().lower()
        if name_l in known:
            return False
        parts = [p.strip() for chunk in name.split(",") for p in chunk.split("&") if p.strip()]
        has_known = any(any(k in p.lower() for k in known) for p in parts)
        if not has_known:
            return True
        town_words = ("town", "city", "village", "island", "plateau", "hamlet", "port")
        unknown_town_parts = [
            p for p in parts
            if not any(k in p.lower() for k in known)
            and any(w in p.lower() for w in town_words)
        ]
        return bool(unknown_town_parts)

    def mark_entrances_seen(self) -> None:
        seen = {self.resolve(*key) for key in (self.seen_entrances or set())}
        valid_segments = {si for si in range(len(self.segments)) if si not in self.merged_into}
        self.seen_entrances = {key for key in seen if key[0] in valid_segments} | self.entrances()

    def _new_doors(self, old: "GlobalMap") -> Dict[Tuple[int, Cell], int]:
        before = {self.resolve(*key): self.resolve(b, (0, 0))[0] for key, b in old.door_labels().items()}
        return {k: b for k, b in self.door_labels().items() if before.get(k) != b}

    @property
    def current(self) -> Dict[Cell, MapCell]:
        return self.segments[self.segment]

    @property
    def n_cells(self) -> int:
        return sum(len(s) for s in self.segments)

    @property
    def n_areas(self) -> int:
        return sum(1 for s in self.segments if s)

    def episodes_without_new_cells(self, episodes: Iterable[int]) -> int:
        count = 0
        for ep in sorted(episodes, reverse=True):
            if self.new_cells_by_episode.get(ep, 0) > 0:
                break
            count += 1
        return count

    def reachable_cells(self, segment: Optional[int] = None) -> set:
        from collections import deque
        from .routes import blocked_target_cells, walked_squares

        si = self.segment if segment is None else segment
        seg = self.segments[si] if 0 <= si < len(self.segments) else {}
        if not seg:
            return set()
        seg_moves = self.segment_blocked_moves(si)
        blocked_targets = set(self.blocked_cells[si]) if si < len(self.blocked_cells) else set()
        if seg_moves:
            blocked_targets |= blocked_target_cells(self, si, seg_moves)
        if self.segment == si and self.pos is not None:
            blocked_targets.discard(self.pos)
        seeds = set(walked_squares(self, si)) - blocked_targets
        if self.segment == si and self.pos in seg:
            seeds.add(self.pos)
        if not seeds:
            return {g for g, c in seg.items() if c.kind == WALKABLE and g not in blocked_targets}
        visited = set(seeds)
        queue = deque(seeds)
        while queue:
            cell = queue.popleft()
            cell_blocked_dirs = seg_moves.get(cell, ())
            for direction, (dx, dy) in DIRECTIONS.items():
                if direction in cell_blocked_dirs:
                    continue
                nxt = (cell[0] + dx, cell[1] + dy)
                if nxt in blocked_targets:
                    continue
                c = seg.get(nxt)
                if c is None:
                    continue
                if (c.kind == LEDGE and direction == "down") or (c.kind == LEDGE_LEFT and direction == "left"):
                    land = (nxt[0] + dx, nxt[1] + dy)
                    if land in blocked_targets:
                        continue
                    lc = seg.get(land)
                    if lc is not None and lc.kind in (WALKABLE, UNKNOWN, OCCUPIED) and land not in visited:
                        visited.add(land)
                        queue.append(land)
                elif is_door(c):
                    if nxt not in visited:
                        visited.add(nxt)
                        if nxt in seeds:
                            queue.append(nxt)
                elif c.kind in (WALKABLE, UNKNOWN, OCCUPIED):
                    if nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)
        return visited

    MAX_DOOR_UNEXPLORED_GAP = 6

    def is_door_reachable(
        self,
        segment: int,
        door: Dict[str, Any],
        reachable: Optional[set] = None,
    ) -> bool:
        from collections import deque
        from .routes import blocked_target_cells

        if door.get("leads_to") is not None:
            return True
        seg = self.segments[segment] if 0 <= segment < len(self.segments) else {}
        if not seg:
            return False
        reach = self.reachable_cells(segment) if reachable is None else reachable
        if not reach:
            return True
        for c in door["cells"]:
            if c in reach or any((c[0] + dx, c[1] + dy) in reach for dx, dy in DIRECTIONS.values()):
                return True
        frontier_cells = set(self.frontier(segment, reachable=reach))
        if not frontier_cells:
            return False
        seg_moves = self.segment_blocked_moves(segment)
        blocked_targets = set(self.blocked_cells[segment]) if segment < len(self.blocked_cells) else set()
        if seg_moves:
            blocked_targets |= blocked_target_cells(self, segment, seg_moves)
        door_comp = set(door["cells"])
        comp_queue = deque(door["cells"])
        while comp_queue:
            cur = comp_queue.popleft()
            for dx, dy in DIRECTIONS.values():
                nb = (cur[0] + dx, cur[1] + dy)
                if nb in door_comp or nb in blocked_targets:
                    continue
                mc = seg.get(nb)
                if mc is not None and mc.kind in (WALKABLE, UNKNOWN):
                    door_comp.add(nb)
                    comp_queue.append(nb)
        dist = {f: 0 for f in frontier_cells}
        queue = deque(frontier_cells)
        while queue:
            cur = queue.popleft()
            d = dist[cur]
            if cur in door_comp:
                return True
            cur_blocked = seg_moves.get(cur, ())
            for d_name, (dx, dy) in DIRECTIONS.items():
                if d_name in cur_blocked:
                    continue
                nb = (cur[0] + dx, cur[1] + dy)
                if nb in blocked_targets:
                    continue
                if nb in door_comp:
                    return True
                if d < self.MAX_DOOR_UNEXPLORED_GAP and seg.get(nb) is None and nb not in dist:
                    dist[nb] = d + 1
                    queue.append(nb)
        return False

    def reachable_door_list(
        self,
        segment: Optional[int] = None,
        reachable: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        si = self.segment if segment is None else segment
        reach = self.reachable_cells(si) if reachable is None else reachable
        return [d for d in self.door_list(si) if self.is_door_reachable(si, d, reach)]

    def branch_exploration(self, from_segment: int, target_segment: int) -> Dict[str, Any]:
        from collections import deque

        start_seg = self.resolve(from_segment, (0, 0))[0]
        root_target = self.resolve(target_segment, (0, 0))[0]
        if root_target == start_seg or not (0 <= root_target < len(self.segments)) or not self.segments[root_target]:
            return {"exhausted": False, "segments": [], "frontiers": 0, "unentered_doors": 0}
        seen = {start_seg, root_target}
        queue = deque([root_target])
        order: List[int] = []
        total_frontiers = 0
        total_unentered = 0
        while queue:
            si = queue.popleft()
            order.append(si)
            reach = self.reachable_cells(si)
            total_frontiers += len(self.frontier(si, reachable=reach))
            for d in self.reachable_door_list(si, reach):
                leads = d.get("leads_to")
                if leads is None:
                    total_unentered += 1
                else:
                    nxt = self.resolve(leads, (0, 0))[0]
                    if nxt not in seen and 0 <= nxt < len(self.segments) and self.segments[nxt]:
                        seen.add(nxt)
                        queue.append(nxt)
        return {
            "exhausted": total_frontiers == 0 and total_unentered == 0,
            "segments": order,
            "frontiers": total_frontiers,
            "unentered_doors": total_unentered,
        }

    def frontier(self, segment: Optional[int] = None, reachable: Optional[set] = None) -> List[Cell]:
        si = self.segment if segment is None else segment
        seg = self.segments[si] if 0 <= si < len(self.segments) else {}
        if not seg:
            return []
        reach = self.reachable_cells(si) if reachable is None else reachable
        info = self.info(si)
        check_void = info.kind not in (OUTDOORS, CAVE)
        void_cols = (
            {
                g[0]
                for g, mc in seg.items()
                if mc.pixels is not None
                and int(mc.pixels.max()) <= BLACK_LEVEL
                and int(mc.pixels.min()) == int(mc.pixels.max())
            }
            if check_void
            else set()
        )
        void_rows = (
            {
                g[1]
                for g, mc in seg.items()
                if mc.pixels is not None
                and int(mc.pixels.max()) <= BLACK_LEVEL
                and int(mc.pixels.min()) == int(mc.pixels.max())
            }
            if check_void
            else set()
        )
        has_void = bool(void_cols)
        visits = self.visits[si] if si < len(self.visits) else Counter()
        cave_void_cells = (
            [
                g
                for g, mc in seg.items()
                if mc.pixels is not None
                and int(mc.pixels.max()) <= BLACK_LEVEL
                and int(mc.pixels.min()) == int(mc.pixels.max())
            ]
            if info.kind == CAVE
            else []
        )
        cave_visited_walk = (
            [g for g, mc in seg.items() if mc.kind == WALKABLE and visits.get(g, 0) > 0]
            if cave_void_cells
            else []
        )
        from .routes import blocked_target_cells

        seg_moves = self.segment_blocked_moves(si)
        blocked_targets = set(self.blocked_cells[si]) if si < len(self.blocked_cells) else set()
        if seg_moves:
            blocked_targets |= blocked_target_cells(self, si, seg_moves)
        out = []
        for g, c in seg.items():
            if c.kind != WALKABLE or g in blocked_targets or (reach and g not in reach):
                continue
            g_blocked_dirs = seg_moves.get(g, ())
            has_open_unmapped = False
            for d_name, (dx, dy) in DIRECTIONS.items():
                if d_name in g_blocked_dirs:
                    continue
                u = (g[0] + dx, g[1] + dy)
                if u in seg or u in blocked_targets:
                    continue
                if has_void and visits.get(g, 0) == 0:
                    if dx != 0 and (g[0] in void_cols or u[0] in void_cols):
                        continue
                    if dy != 0 and (g[1] in void_rows or u[1] in void_rows):
                        continue
                if cave_void_cells and visits.get(g, 0) == 0:
                    f_void = sum(
                        1
                        for v in cave_void_cells
                        if (dx > 0 and v[0] >= u[0])
                        or (dx < 0 and v[0] <= u[0])
                        or (dy > 0 and v[1] >= u[1])
                        or (dy < 0 and v[1] <= u[1])
                    )
                    if f_void >= 4 and not any(
                        (dx > 0 and w[0] >= g[0])
                        or (dx < 0 and w[0] <= g[0])
                        or (dy > 0 and w[1] >= g[1])
                        or (dy < 0 and w[1] <= g[1])
                        for w in cave_visited_walk
                    ):
                        continue
                if visits.get(g, 0) == 0:
                    enclosed_corner = False
                    if dx != 0:
                        for oy in (-1, 1):
                            c_orth = seg.get((g[0], g[1] + oy))
                            c_diag = seg.get((u[0], g[1] - oy))
                            c_diag2 = seg.get((u[0], g[1] - 2 * oy))
                            if (
                                c_orth is not None
                                and c_orth.kind == BLOCKED
                                and c_diag is not None
                                and c_diag.kind == BLOCKED
                                and (c_diag2 is None or c_diag2.kind == BLOCKED)
                            ):
                                enclosed_corner = True
                                break
                    elif dy != 0:
                        for ox in (-1, 1):
                            c_orth = seg.get((g[0] + ox, g[1]))
                            c_diag = seg.get((g[0] - ox, u[1]))
                            c_diag2 = seg.get((g[0] - 2 * ox, u[1]))
                            if (
                                c_orth is not None
                                and c_orth.kind == BLOCKED
                                and c_diag is not None
                                and c_diag.kind == BLOCKED
                                and (c_diag2 is None or c_diag2.kind == BLOCKED)
                            ):
                                enclosed_corner = True
                                break
                    if enclosed_corner:
                        continue
                has_open_unmapped = True
                break
            if has_open_unmapped:
                out.append(g)
        return out
