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
)
from execution.perception.pokemon.grid import GRID_SKIP
from execution.perception.pokemon.navigation import (LEDGE_LEFT, LEDGE_TAGS, OVERHANG_TOP_TAGS, WALKABLE_TAGS,
                                                     is_east_cliff_barrier_cell, is_npc_head_overhang_floor_cell,
                                                     is_overhang_floor_cell, is_player_overhang_floor_cell,
                                                     is_player_overhang_ledge_cell, is_south_ledge_cell,
                                                     is_top_overhang_floor_cell, is_west_ledge_cell)
from execution.perception.pokemon.tiles import (TileRecognizer, filter_cell_warp_quarter_tags,
                                                normalize_tabletop_tag)

from .registration import (BLACK_LEVEL, _local_cells, _screen, _still_lit, grid_offset, grid_residual, has_structure,
                           register)

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


def _quarter_rank_signature(pixels: Optional[np.ndarray]) -> Optional[Tuple[bytes, ...]]:
    if pixels is None or pixels.size != GRID_SKIP * GRID_SKIP:
        return None
    p = np.squeeze(pixels)
    if int(p.min()) == int(p.max()):
        return None
    sigs = []
    for q in QUARTERS:
        qp = split_quarters(p)[q].reshape(-1)
        u, idx = np.unique(qp, return_inverse=True)
        if len(u) == 1:
            sigs.append(b"D" if int(u[0]) <= BLACK_LEVEL else b"L")
        else:
            sigs.append(bytes([len(u)]) + idx.astype(np.uint8).tobytes())
    return tuple(sigs)


def _quarter_recolour_matches(
    h1: Optional[str], h2: Optional[str], p1: Optional[np.ndarray], p2: Optional[np.ndarray]
) -> bool:
    if h1 is not None and h1 == h2:
        return True
    if p1 is None or p2 is None:
        return False
    a, b = np.asarray(p1).reshape(-1), np.asarray(p2).reshape(-1)
    u1, idx1 = np.unique(a, return_inverse=True)
    u2, idx2 = np.unique(b, return_inverse=True)
    if len(u1) == 1 and len(u2) == 1:
        return (int(u1[0]) <= BLACK_LEVEL) == (int(u2[0]) <= BLACK_LEVEL)
    if len(u1) < 2 or len(u2) < 2:
        return False
    if len(u1) == len(u2) and np.array_equal(idx1, idx2):
        return True
    bank = shade_bank([p2])
    return len(shade_candidates(bank, p1, min_shades=2, ordered=True)) > 0


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


@dataclass
class MapCell:
    kind: str
    tags: Tuple[str, ...]
    hashes: Tuple[Optional[str], ...]
    first_seen: int
    last_seen: int
    pixels: Optional[np.ndarray] = None


class GlobalMap(StrategistArtifact):
    """Segments of global grid, the player's place on them, and the bookkeeping to explain both."""

    # Set by MappingEnvironment to the tile recognizer's palette-independent hash; not saved.
    _TRANSIENT = ("canonical_hash",)
    canonical_hash: Optional[Callable[[np.ndarray], Optional[str]]] = None

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

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self.__dict__.setdefault("defeats", [])
        self.__dict__.setdefault("defeat_return", None)
        self.__dict__.setdefault("seen_entrances", None)
        self.__dict__.setdefault("exploring_until", None)
        self.__dict__.setdefault("segment_born", 0)
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
        for si in range(len(self.segments)):
            self._heal_visited_ground(si)

    def _heal_visited_ground(self, si: int) -> int:
        seg = self.segments[si] if 0 <= si < len(self.segments) else None
        if not seg or si >= len(self.visits) or not self.visits[si]:
            return 0
        seg_blocked = self.blocked_cells[si] if si < len(self.blocked_cells) else set()
        restored = 0
        for g, cnt in self.visits[si].items():
            if cnt <= 0:
                continue
            seg_blocked.discard(g)
            c = seg.get(g)
            if (
                c is not None
                and c.kind == BLOCKED
                and not is_door(c)
                and "ground" in c.tags
                and not (set(c.tags) & {"sign", "interactable object", "pc", "water", "ledge", "right_ledge", "left_ledge"})
            ):
                c.kind = WALKABLE
                c.tags = ("ground",)
                restored += 1
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
    ) -> bool:
        tag_set = set(tags)
        if "npc" in tag_set or "unknown" in tag_set or "water" in tag_set:
            return False
        if bool(tag_set & WALKABLE_TAGS):
            return True
        if hashes is not None and len(hashes) == 4 and len(set(hashes)) == 1 and hashes[0] is not None:
            if pixels is not None and pixels.size > 0 and int(pixels.min()) == int(pixels.max()):
                return False
            return True
        return False

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
            for other in self.segments:
                for c in other.values():
                    if c.kind == BLOCKED and c.hashes == pattern:
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

    def record_blocked_step(self, direction: Optional[str] = None, cell: Optional[Cell] = None) -> Optional[Cell]:
        if not self.current or self.pos is None:
            return None
        while len(self.blocked_cells) <= self.segment:
            self.blocked_cells.append(set())
        if cell is None:
            if direction not in DIRECTIONS:
                return None
            dx, dy = DIRECTIONS[direction]
            target = (self.pos[0] + dx, self.pos[1] + dy)
        else:
            target = cell
        if target == self.pos:
            return None
        if self.visits[self.segment].get(target, 0) > 0:
            return None
        seg = self.segments[self.segment]
        c = seg.get(target)
        if c is not None and (is_door(c) or c.kind in (LEDGE, LEDGE_LEFT)):
            return None
        if (
            c is not None
            and c.hashes
            and len(c.hashes) == 4
            and len(set(c.hashes)) == 1
            and c.hashes[0] is not None
            and c.hashes in self.walk_patterns
        ):
            return None
        self.blocked_cells[self.segment].add(target)
        if c is not None and c.kind == WALKABLE:
            c.kind = BLOCKED
            self.stats["empirical_blocked_cells"] += 1
        return target

    def local_blocked_cells(self) -> set:
        if not self.current or self.pos is None or self.segment >= len(self.blocked_cells):
            return set()
        px, py = self.pos
        out = set()
        for gx, gy in self.blocked_cells[self.segment]:
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
            if (seg and si not in queued and si not in self.merged_into
                    and self._has_min_observations(si)
                    and self.filled_pixels(si) - self.info(si).named_pixels >= RENAME_GROWTH_PIXELS):
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
                shift: Cell = (0, 0), can_learn_walkable: bool = False) -> int:
        seg = self.segments[self.segment]
        a = _screen(frame)
        px, py = self.pos
        added = 0
        if shift != (0, 0) or has_bottom_dialogue_box(frame) or has_modal_preview_box(frame):
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
        top_overhang_cells = {
            cell for cell in quarter_tags if is_top_overhang_floor_cell(cell, quarter_tags, quarter_descriptions)
        }
        player_overhang_cells = {
            cell for cell in quarter_tags if is_player_overhang_floor_cell(cell, quarter_tags, quarter_descriptions)
        }
        npc_head_overhang_cells = {
            cell for cell in quarter_tags if is_npc_head_overhang_floor_cell(cell, quarter_tags)
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
        for (x, y), piece in _local_cells(a, shift):
            if (x, y) in masked_cells and (x, y) != (0, 0):
                continue
            pixels = piece.copy()
            quarters = split_quarters(piece)
            hashes = self._cell_hashes(quarters)
            if (x, y) == (0, 0):
                tags, kind = ("player",), WALKABLE
            elif shift != (0, 0):
                tags, kind = ("unknown",), UNKNOWN
            elif (x, y) in top_overhang_cells:
                tags = tuple(sorted({
                    quarter_tags[(x, y)][q] for q in ("bottom_left", "bottom_right") if q in quarter_tags[(x, y)]
                }))
                kind = WALKABLE
            elif (x, y) in player_overhang_cells or (x, y) in npc_head_overhang_cells:
                tags = tuple(sorted({
                    quarter_tags[(x, y)][q] for q in ("top_left", "top_right") if q in quarter_tags[(x, y)]
                }))
                kind = WALKABLE
            elif (x, y) in south_ledge_cells:
                tags = tuple(sorted(
                    {quarter_tags[(x, y)][q] for q in ("top_left", "top_right") if q in quarter_tags[(x, y)]}
                    | {"ledge"}
                ))
                kind = LEDGE
            elif (x, y) in west_ledge_cells:
                tags = tuple(sorted(
                    {quarter_tags[(x, y)][q] for q in ("top_right", "bottom_right") if q in quarter_tags[(x, y)]}
                    | {"ledge"}
                ))
                kind = LEDGE_LEFT
            else:
                cell_q = quarter_tags.get((x, y))
                if cell_q:
                    tags = tuple(sorted(set(cell_q.values())))
                else:
                    tags = tuple(sorted({
                        normalize_tabletop_tag(
                            identified[(x, y, q)].tag,
                            identified[(x, y, q)].description,
                            getattr(identified[(x, y, q)], "image", None),
                        )
                        for q in QUARTERS if (x, y, q) in identified
                    }))
                kind = BLOCKED if (x, y) in east_cliff_cells else self._classify(tags, hashes)
            g = (px + x, py + y)
            while len(self.blocked_cells) <= self.segment:
                self.blocked_cells.append(set())
            seg_blocked = self.blocked_cells[self.segment]
            is_uniform_piece = int(pixels.min()) == int(pixels.max())
            if (x, y) == (0, 0) or self.visits[self.segment].get(g, 0) > 0:
                seg_blocked.discard(g)
            elif g in seg_blocked and kind == WALKABLE and not bool(set(tags) & DOOR_TAGS):
                if (can_learn_walkable or len(set(hashes)) == 1) and None not in hashes and hashes in self.walk_patterns:
                    seg_blocked.discard(g)
                else:
                    kind = BLOCKED
            old = seg.get(g)
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
            old.last_seen = self.step
            if is_uniform_piece and old.pixels is not None and int(old.pixels.min()) != int(old.pixels.max()):
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
                and kind == BLOCKED
                and not bool(set(tags) & DOOR_TAGS)
            ):
                continue
            old.kind, old.tags, old.hashes, old.pixels = kind, tags, hashes, pixels
        self.visits[self.segment][self.pos] += 1
        self.trail.append((self.step, self.segment, self.pos))
        self._new_this_episode += added
        self.stats["cells_added"] += added
        return added

    def on_step(self, frame: np.ndarray, identified, free_roam: bool, direction: Optional[str] = None,
                max_steps: int = 0, reported: Optional[int] = None, was_free_roam: bool = True,
                faded: bool = False, in_battle: bool = False) -> str:
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
        if faded and not self.in_battle and self.since_battle_moved and (free_roam or was_free_roam):
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
            self.in_battle = False
            self.observe(frame, identified, shift)
            return "start"
        if self.pending is not None and direction not in DIRECTIONS:
            direction, max_steps, reported = self.pending
        placed = None if self.faded else register(self.anchor, frame, direction, max_steps, reported,
                                                  self.residual)
        if placed is not None:
            delta, residual, structural = placed
            prev_pos = self.pos
            pos = (self.pos[0] + delta[0], self.pos[1] + delta[1])
            shift = tuple((-r) % GRID_SKIP for r in residual)
            if self._blacks_out(frame, self.segment, pos, shift):
                if direction in DIRECTIONS and self.pending is None:
                    self.pending = (direction, max_steps, reported)
                self.stats["blackout_frames"] += 1
                return "blank"
            if not structural:
                self._learn_ledges(prev_pos, pos, direction)
                if (delta[0] == 0) != (delta[1] == 0) and 1 < abs(delta[0]) + abs(delta[1]) <= MAX_STEP_FILL:
                    sx = (delta[0] > 0) - (delta[0] < 0)
                    sy = (delta[1] > 0) - (delta[1] < 0)
                    for i in range(1, abs(delta[0]) + abs(delta[1])):
                        mid = (prev_pos[0] + sx * i, prev_pos[1] + sy * i)
                        c_mid = self.segments[self.segment].get(mid)
                        if c_mid is None or c_mid.kind not in (BLOCKED, LEDGE, LEDGE_LEFT):
                            self.visits[self.segment][mid] += 1
            was_off_grid = self.residual != (0, 0) or residual != (0, 0)
            if residual != (0, 0) and not structural and grid_offset(frame) == (0, 0):
                residual = (0, 0)
                shift = (0, 0)
            self.pos, self.residual = pos, residual
            if not structural and self.residual == (0, 0):
                snapped = self._snap_within_segment(frame, identified, was_off_grid=was_off_grid)
                if snapped is not None and snapped != self.pos:
                    self.pos = snapped
                    delta = (self.pos[0] - prev_pos[0], self.pos[1] - prev_pos[1])
            placed_steps = abs(delta[0]) + abs(delta[1])
            if structural and direction in DIRECTIONS and reported is not None and placed_steps < reported:
                rem_reported = reported - placed_steps
                rem_max = max(rem_reported + 1, max_steps - placed_steps)
                self.pending = (direction, rem_max, rem_reported)
            else:
                self.pending = None
            self.lost = 0
            if not structural:
                self.anchor = _screen(frame)
            if not structural:
                self.in_battle = False
            self.stats["placed_by_edges" if structural else "placed"] += 1
            if structural:
                # recoloured (a flash or fade): the position holds, the colours do not
                self.stats["recoloured_frames"] += 1
            elif self.residual == (0, 0):
                self.observe(frame, identified, can_learn_walkable=(delta != (0, 0)))
                self._try_merge(frame)
            else:
                self.observe(frame, identified, shift)
                self.stats["off_grid_frames"] += 1
            return "moved" if delta != (0, 0) else "stayed"
        shift = self._grid_shift(frame)
        warp_targets = self._known_warp_targets(direction, max_steps, reported) if self.faded else None
        found = self.relocalise(
            frame,
            shift=shift,
            allowed=lambda si, q: not self.arrival_breaks_doors(si, q),
            warp_targets=warp_targets,
        )
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
                adv_pos = self.pos
                for _ in range(reported):
                    nxt = (adv_pos[0] + dx, adv_pos[1] + dy)
                    c_nxt = seg.get(nxt)
                    if c_nxt is not None and c_nxt.kind in (BLOCKED, LEDGE, LEDGE_LEFT):
                        break
                    adv_pos = nxt
                    self.visits[self.segment][adv_pos] += 1
                found = (self.segment, adv_pos)
            if found != (self.segment, self.pos):
                self._link(*found, departure_pos=self._warp_departure_pos(direction, max_steps, reported))
            self.segment, self.pos = found
            self.residual, self.pending, self.lost, self.anchor = grid_residual(shift), None, 0, _screen(frame)
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
        self.segment_info[len(self.segments) - 1] = SegmentInfo()
        self._link(
            len(self.segments) - 1,
            (0, 0),
            departure_pos=self._warp_departure_pos(direction, max_steps, reported),
        )
        self.segment, self.pos = len(self.segments) - 1, (0, 0)
        self.residual, self.pending, self.lost, self.anchor = grid_residual(shift), None, 0, _screen(frame)
        self.faded = False
        self.in_battle = False
        self.segment_born = self.stats["placed"]
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

    def _link(self, segment: int, pos: Cell, departure_pos: Optional[Cell] = None) -> None:
        if not self.faded or self.in_battle or not self.since_battle_moved or segment == self.segment:
            return
        dep = self.pos if departure_pos is None else departure_pos
        self.warps.append((self.step, self.segment, dep, segment, pos))
        self.warp_moves.append(self.fade_direction if self.fade_direction in DIRECTIONS else None)
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
            for (x, y), h in local.items():
                qtags = {
                    normalize_tabletop_tag(
                        identified[(x, y, q)].tag,
                        identified[(x, y, q)].description,
                        getattr(identified[(x, y, q)], "image", None),
                    )
                    for q in QUARTERS
                    if (x, y, q) in identified
                }
                if qtags:
                    live_kinds[(x, y)] = self._classify(qtags, h)
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
                }
        s0 = scores.get((0, 0))
        if s0 is None:
            return None
        cands = []
        for d, sc in scores.items():
            if d == (0, 0) or sc["ov"] < 20:
                continue
            dist = abs(d[0]) + abs(d[1])
            qualifies = False
            if sc["frac"] >= 0.85 and (
                sc["frac"] - s0["frac"] >= 0.08
                or (sc["lm_match"] > s0["lm_match"] and sc["lm_mismatch"] < s0["lm_mismatch"] and sc["frac"] - s0["frac"] >= 0.04)
                or (sc["vis_total"] >= 4 and sc["vis_match"] == sc["vis_total"] and s0["vis_match"] < s0["vis_total"] and sc["frac"] - s0["frac"] >= 0.05)
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
            for _, a, pa, b, pb in self.warps:
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
        for _, a, pa, b, pb in self.warps:
            ra, rpa = self.resolve(a, pa)
            rb, rpb = self.resolve(b, pb)
            if rb != cur_seg or ra == cur_seg:
                continue
            for _, a2, pa2, b2, pb2 in self.warps:
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
        if self.segment == 0 or self.stats["placed"] - self.segment_born > MERGE_WINDOW:
            return
        warp_targets = self._known_warp_targets(for_merge=True)
        found = self.relocalise(frame, exclude=self.segment, warp_targets=warp_targets)
        if found is None:
            return
        target, target_pos = found
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
        if self.names_differ(self.segment, target):
            return
        self._merge(self.segment, target, offset)
        if self.segment == target and self.pos != target_pos:
            self.pos = target_pos

    def _merge(self, source: int, target: int, offset: Cell) -> None:
        dx, dy = offset
        src, dst = self.segments[source], self.segments[target]
        while len(self.blocked_cells) <= max(source, target):
            self.blocked_cells.append(set())
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
        self.merged_into[source] = (target, (dx, dy))
        self.segments[source] = {}
        self.visits[source] = Counter()
        self.blocked_cells[source] = set()
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
            self._clean_warps()
            self.stats["transient_pruned"] += 1
            changed += 1
        self._clean_warps()
        return changed

    def overlap(self, source: int, target: int) -> Optional[Dict[str, Any]]:
        def is_uniform(c: MapCell) -> bool:
            return c.pixels is not None and int(c.pixels.min()) == int(c.pixels.max())

        def content(seg):
            return {g: c for g, c in seg.items() if c.kind != OCCUPIED and not is_uniform(c)}

        seg_src, seg_dst = self.segments[source], self.segments[target]
        src, dst = content(seg_src), content(seg_dst)
        index: Dict[Tuple[Any, ...], List[Cell]] = defaultdict(list)
        for g, c in dst.items():
            index[c.hashes].append(g)
            rank_sig = _quarter_rank_signature(c.pixels)
            if rank_sig is not None:
                index[rank_sig].append(g)
        block_index = self._block_index(dst)
        src_hashes = {g: c.hashes for g, c in src.items()}
        votes = self._collect_offset_votes(src_hashes, index, block_index)
        src_rank = {g: s for g, c in src.items() if (s := _quarter_rank_signature(c.pixels)) is not None}
        if src_rank:
            rank_votes = self._collect_offset_votes(src_rank, index, {})
            for off, cnt in rank_votes.items():
                if cnt > votes[off]:
                    votes[off] = cnt
        if not votes:
            return None

        def agreement(offset: Cell) -> Tuple[int, int]:
            matches = total = 0
            for g, c in seg_src.items():
                if c.kind == OCCUPIED:
                    continue
                c_dst = seg_dst.get((g[0] + offset[0], g[1] + offset[1]))
                if c_dst is None or c_dst.kind == OCCUPIED:
                    continue
                u_src, u_dst = is_uniform(c), is_uniform(c_dst)
                if u_src and u_dst:
                    continue
                total += 1
                if not u_src and not u_dst and _cells_match(c.hashes, c.pixels, c_dst):
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
        return {"offset": (dx, dy), "matches": matches, "shared": shared,
                "agreement": matches / shared if shared else 0.0,
                "votes": best_votes, "rival": rival,
                "rival_agreement": rival_matches / rival_shared if rival_shared else 0.0}

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
        if judge is None:
            return False
        same = judge(source, target, overlap)
        self.merge_verdicts[key] = {"same": bool(same), "matches": overlap["matches"], "step": self.step,
                                    "names": (self.info(source).name, self.info(target).name)}
        self.stats["merges_judged_same" if same else "merges_judged_different"] += 1
        return bool(same)

    def _warp_anchored_overlap(self, source: int, target: int) -> Optional[Dict[str, Any]]:
        src_seg = self.segments[source] if 0 <= source < len(self.segments) else {}
        dst_seg = self.segments[target] if 0 <= target < len(self.segments) else {}
        if not src_seg or not dst_seg or self.info(source).name is not None:
            return None
        for _, a, pa, b, pb in self.warps:
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
            for _, a2, pa2, b2, pb2 in self.warps:
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
        return healed

    def consolidate(self, judge: Optional[Callable[[int, int, Dict[str, Any]], Optional[bool]]] = None
                    ) -> List[Dict[str, Any]]:
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
            index: Dict[Tuple[Any, ...], List[Cell]] = defaultdict(list)
            for g, c in seg.items():
                if c.kind != OCCUPIED:
                    index[c.hashes].append(g)
                    rank_sig = _quarter_rank_signature(c.pixels)
                    if rank_sig is not None:
                        index[rank_sig].append(g)
            block_index = self._block_index(seg)
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
                if distinctive[0][0] - distinctive[1][0] < DISTINCTIVE_LEAD:
                    continue
                close = [(distinctive[0][1], distinctive[0][2], distinctive[0][3])]
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
        return max(abs(trans_src_p[0] - dst_p[0]), abs(trans_src_p[1] - dst_p[1])) <= DOOR_REACH

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
            after.append((t, *moved(a, pa), *moved(b, pb)))
            after_moves.append(m)
            before_warps.append((t, a, pa, b, pb))

        def coupled_same(other1: int, q1: Cell, other2: int, q2: Cell) -> bool:
            if other1 == other2:
                return self._same_door(other1, q1, q2)
            ov = self.overlap(other1, other2)
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
        move = self.fade_direction if self.fade_direction in DIRECTIONS else None
        after = self.warps + [(self.step, self.segment, self.pos, segment, pos)]
        return (self.door_conflicts(after, self.warp_moves + [move], self._same_door)
                > self.door_conflicts(self.warps, self.warp_moves, self._same_door))

    def refresh(self, tile_lookup: Dict[str, Any]) -> int:
        self.prune_transient_segments()
        if self.walk_patterns:
            cleaned_patterns = set()
            for pattern in self.walk_patterns:
                tiles = [tile_lookup.get(h) for h in pattern]
                if all(t is not None for t in tiles):
                    qtags = {
                        normalize_tabletop_tag(t.tag, t.description, getattr(t, "image", None)) for t in tiles
                    }
                    t_img = getattr(tiles[0], "image", None)
                    if not self._can_learn_walkable(qtags, pattern, t_img):
                        continue
                cleaned_patterns.add(pattern)
            self.walk_patterns = cleaned_patterns
        for si, seg in enumerate(self.segments):
            if not seg or si >= len(self.visits) or not self.visits[si]:
                continue
            seg_blocked = self.blocked_cells[si] if si < len(self.blocked_cells) else set()
            uniform_visited: Dict[Tuple[Optional[str], ...], List[MapCell]] = defaultdict(list)
            for g, cnt in self.visits[si].items():
                if cnt <= 0 or g in seg_blocked:
                    continue
                c = seg.get(g)
                if (
                    c is not None
                    and c.hashes
                    and len(c.hashes) == 4
                    and len(set(c.hashes)) == 1
                    and c.hashes[0] is not None
                ):
                    uniform_visited[c.hashes].append(c)
            for cells_for_pat in uniform_visited.values():
                if len(cells_for_pat) >= 2:
                    self._learn_walkable(cells_for_pat[0])
        while len(self.blocked_cells) < len(self.segments):
            self.blocked_cells.append(set())
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
            for g, c in seg.items():
                if g not in quarter_tags or c.kind == OCCUPIED:
                    continue
                if is_top_overhang_floor_cell(g, quarter_tags, quarter_descriptions):
                    new_tags = tuple(sorted({quarter_tags[g]["bottom_left"], quarter_tags[g]["bottom_right"]}))
                    new_kind = WALKABLE
                elif is_npc_head_overhang_floor_cell(g, quarter_tags):
                    new_tags = tuple(sorted({quarter_tags[g]["top_left"], quarter_tags[g]["top_right"]}))
                    new_kind = WALKABLE
                elif is_south_ledge_cell(g, quarter_tags, quarter_descriptions):
                    new_tags = tuple(sorted({quarter_tags[g]["top_left"], quarter_tags[g]["top_right"], "ledge"}))
                    new_kind = LEDGE
                elif is_west_ledge_cell(g, quarter_tags, quarter_descriptions):
                    new_tags = tuple(sorted({quarter_tags[g]["top_right"], quarter_tags[g]["bottom_right"], "ledge"}))
                    new_kind = LEDGE_LEFT
                elif is_east_cliff_barrier_cell(g, quarter_tags, quarter_descriptions, quarter_images):
                    new_tags = tuple(sorted(set(quarter_tags[g].values())))
                    new_kind = BLOCKED
                else:
                    new_tags = tuple(sorted(set(quarter_tags[g].values())))
                    new_kind = self._classify(new_tags, c.hashes)
                if g in seg_blocked:
                    if (
                        (si < len(self.visits) and self.visits[si].get(g, 0) > 0)
                        or (
                            c.hashes
                            and len(c.hashes) == 4
                            and len(set(c.hashes)) == 1
                            and c.hashes[0] is not None
                            and c.hashes in self.walk_patterns
                        )
                    ):
                        seg_blocked.discard(g)
                    elif new_kind == WALKABLE and not bool(set(new_tags) & DOOR_TAGS):
                        new_kind = BLOCKED
                if (
                    si < len(self.visits)
                    and self.visits[si].get(g, 0) > 0
                    and new_kind == BLOCKED
                    and not is_door(c)
                    and not bool(set(new_tags) & DOOR_TAGS)
                    and (c.kind == WALKABLE or "ground" in c.tags or "ground" in new_tags)
                    and not (set(new_tags) & {"sign", "interactable object", "pc", "water", "ledge", "right_ledge", "left_ledge"})
                ):
                    new_kind = WALKABLE
                    if "obstacle" in new_tags:
                        new_tags = tuple(t for t in new_tags if t != "obstacle") or ("ground",)
                if (c.kind, c.tags) != (new_kind, new_tags):
                    c.kind, c.tags = new_kind, new_tags
                    changed += 1
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
        for _, a, pa, b, pb in self.warps:
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

    def door_list(self, segment: Optional[int] = None) -> List[Dict[str, Any]]:
        si = self.segment if segment is None else segment
        seg = self.segments[si]
        labels = {p: b for (s, p), b in self.door_labels().items() if s == si}
        order = {key: i for i, key in enumerate(labels)}
        cells = {g for g, c in seg.items() if is_door(c)} | set(labels)
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
        from .routes import walked_squares

        si = self.segment if segment is None else segment
        seg = self.segments[si] if 0 <= si < len(self.segments) else {}
        if not seg:
            return set()
        seeds = set(walked_squares(self, si))
        if self.segment == si and self.pos in seg:
            seeds.add(self.pos)
        if not seeds:
            return {g for g, c in seg.items() if c.kind == WALKABLE}
        visited = set(seeds)
        queue = deque(seeds)
        while queue:
            cell = queue.popleft()
            for direction, (dx, dy) in DIRECTIONS.items():
                nxt = (cell[0] + dx, cell[1] + dy)
                c = seg.get(nxt)
                if c is None:
                    continue
                if (c.kind == LEDGE and direction == "down") or (c.kind == LEDGE_LEFT and direction == "left"):
                    land = (nxt[0] + dx, nxt[1] + dy)
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

    def is_door_reachable(
        self,
        segment: int,
        door: Dict[str, Any],
        reachable: Optional[set] = None,
    ) -> bool:
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
        out = []
        for g, c in seg.items():
            if c.kind != WALKABLE or (reach and g not in reach):
                continue
            has_open_unmapped = False
            for dx, dy in DIRECTIONS.values():
                u = (g[0] + dx, g[1] + dy)
                if u in seg:
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
