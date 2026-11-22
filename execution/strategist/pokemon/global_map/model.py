from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.matching import cell_key, split_quarters
from execution.perception.pokemon.battle import has_bottom_dialogue_box, has_modal_preview_box, is_battle_screen
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
    return bool(tags & DOOR_TAGS) and "obstacle" not in tags


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

    @staticmethod
    def _can_learn_walkable(tags: Iterable[str]) -> bool:
        tag_set = set(tags)
        return bool(tag_set & WALKABLE_TAGS) and "npc" not in tag_set and "unknown" not in tag_set

    def _classify(self, tags: Iterable[str], hashes: Tuple[Optional[str], ...]) -> str:
        tag_set = set(tags)
        kind = classify(tag_set)
        if kind == BLOCKED and hashes in self.ledge_patterns:
            return LEDGE
        if kind == BLOCKED and hashes in self.left_ledge_patterns:
            return LEDGE_LEFT
        if kind == BLOCKED and hashes in self.walk_patterns and self._can_learn_walkable(tag_set):
            return WALKABLE
        return kind

    def _learn_walkable(self, cell: "MapCell") -> int:
        pattern = cell.hashes
        if any(h is None for h in pattern) or not self._can_learn_walkable(cell.tags):
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
            if not seg:
                continue
            info = self.info(si)
            done = any(c.kind == WALKABLE for c in seg.values()) and not self.frontier(si)
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
        if has_bottom_dialogue_box(frame) or has_modal_preview_box(frame):
            self.visits[self.segment][self.pos] += 1
            self.trail.append((self.step, self.segment, self.pos))
            return 0
        masked_cells = {key[:2] for key in TileRecognizer.overlay_tile_keys(frame)} if shift == (0, 0) else set()
        quarter_tags: Dict[Cell, Dict[str, str]] = {}
        quarter_descriptions: Dict[Cell, Dict[str, Optional[str]]] = {}
        if shift == (0, 0):
            for (ix, iy, iq), tile in (identified or {}).items():
                quarter_tags.setdefault((ix, iy), {})[iq] = normalize_tabletop_tag(
                    tile.tag, tile.description, getattr(tile, "image", None)
                )
                quarter_descriptions.setdefault((ix, iy), {})[iq] = tile.description
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
            cell for cell in quarter_tags if is_east_cliff_barrier_cell(cell, quarter_tags, quarter_descriptions)
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
            if (x, y) == (0, 0):
                seg_blocked.discard(g)
            elif g in seg_blocked and kind == WALKABLE and not bool(set(tags) & DOOR_TAGS):
                kind = BLOCKED
            old = seg.get(g)
            if old is None:
                if is_uniform_piece and self.info(self.segment).kind == OUTDOORS:
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
            if (x, y) == (0, 0) and old.kind != OCCUPIED:
                if can_learn_walkable and old.kind == BLOCKED:
                    self._learn_walkable(old)
                continue
            if (x, y) == (0, 1) and old.kind == WALKABLE and kind != OCCUPIED:
                # keep clean ground hashes/pixels at (0, 1) rather than overwriting with the player's cap overhang
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
            self.in_battle = True
            self.faded = False
            self.fade_direction = None
            self.lost = 0
        elif faded and not self.in_battle:
            # after a blank screen the position must be recognised from the map, not slid from the last frame
            self.faded = True
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
            self.pos, self.residual = pos, residual
            self.pending, self.lost = None, 0
            if not structural or delta != (0, 0):
                self.anchor = _screen(frame)
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
        found = self.relocalise(frame, shift=shift, allowed=lambda si, q: not self.arrival_breaks_doors(si, q))
        if found is not None and self._blacks_out(frame, *found, shift):
            if direction in DIRECTIONS and self.pending is None:
                self.pending = (direction, max_steps, reported)
            self.stats["blackout_frames"] += 1
            return "blank"
        if found is not None:
            prev_segment = self.segment
            if found != (self.segment, self.pos):
                self._link(*found)
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
        self._link(len(self.segments) - 1, (0, 0))
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

    def _link(self, segment: int, pos: Cell) -> None:
        if not self.faded or self.in_battle or segment == self.segment:
            return
        self.warps.append((self.step, self.segment, self.pos, segment, pos))
        self.warp_moves.append(self.fade_direction if self.fade_direction in DIRECTIONS else None)
        left = self.segment
        if (self.info(left).name is None and self._has_min_observations(left)
                and all(entry["segment"] != left for entry in self.naming_queue)):
            self.naming_queue.append({"segment": left, "exit": self.pos, "step": self.step, "marker": None,
                                      "reason": "door"})

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
        top_black = sum(1 for x in range(-4, 6) if (x, 4) in cells_by_xy and cells_by_xy[(x, 4)].max() <= BLACK_LEVEL)
        bot_black = sum(1 for x in range(-4, 6) if (x, -4) in cells_by_xy and cells_by_xy[(x, -4)].max() <= BLACK_LEVEL)
        left_black = sum(1 for y in range(-4, 5) if (-4, y) in cells_by_xy and cells_by_xy[(-4, y)].max() <= BLACK_LEVEL)
        right_black = sum(1 for y in range(-4, 5) if (5, y) in cells_by_xy and cells_by_xy[(5, y)].max() <= BLACK_LEVEL)
        black = white = darkened = 0
        best_same = 0
        best_changed = 999
        for dx in (-2, -1, 0, 1, 2):
            for dy in (-2, -1, 0, 1, 2):
                same = changed = b = w = d = 0
                for (x, y), piece in cells_by_xy.items():
                    old = seg.get((self.pos[0] + dx + x, self.pos[1] + dy + y))
                    if old is None or old.pixels is None or (x, y) == (0, 0):
                        continue
                    new, prior = piece.astype(int), np.squeeze(old.pixels).astype(int)
                    lit = _still_lit(new, prior)
                    if not lit.all():
                        b += 1
                    if new.min() >= WHITE_LEVEL and prior.min() < WHITE_LEVEL:
                        w += 1
                    if new.max() <= DARK_LEVEL and prior.max() >= BRIGHT_LEVEL:
                        d += 1
                    if not lit.any():
                        continue
                    if np.mean(np.abs(new - prior)[lit] > BLACK_LEVEL) < WIPE_CELL_CHANGE:
                        same += 1
                    else:
                        changed += 1
                if (dx, dy) == (0, 0):
                    black, white, darkened = b, w, d
                if (same, -changed) > (best_same, -best_changed):
                    best_same, best_changed = same, changed
        if black >= BLACKOUT_CELLS and best_same >= WIPE_SAME_RATIO * best_changed:
            return True
        if top_black >= 8 and bot_black >= 8 and left_black >= 7 and right_black >= 7 and black >= IRIS_WIPE_CELLS:
            return True
        if white >= TRANSITION_FLASH_CELLS:
            return True
        if black + white >= TRANSITION_FLASH_CELLS and best_same == 0:
            return True
        if darkened >= TRANSITION_FLASH_CELLS and np.percentile(a, 98) <= DARK_LEVEL:
            return True
        mid_tone = sum(1 for piece in cells_by_xy.values() if np.any((piece > BLACK_LEVEL) & (piece < WHITE_LEVEL)))
        if cells_by_xy and mid_tone < MIN_MID_TONE_CELLS:
            return True
        return False

    def _grid_shift(self, frame: np.ndarray) -> Cell:
        shift = grid_offset(frame) or (0, 0)
        if shift != (0, 0):
            self.stats["off_grid_unplaced"] += 1
        return shift

    def _try_merge(self, frame: np.ndarray) -> None:
        if self.segment == 0 or self.stats["placed"] - self.segment_born > MERGE_WINDOW:
            return
        found = self.relocalise(frame, exclude=self.segment)
        if found is None:
            return
        target, target_pos = found
        offset = (target_pos[0] - self.pos[0], target_pos[1] - self.pos[1])
        if self.merge_breaks_doors(self.segment, target, offset):
            self.stats["merge_door_vetoes"] += 1
            return
        if self.names_differ(self.segment, target):
            return
        self._merge(self.segment, target, offset)

    def _merge(self, source: int, target: int, offset: Cell) -> None:
        dx, dy = offset
        src, dst = self.segments[source], self.segments[target]
        while len(self.blocked_cells) <= max(source, target):
            self.blocked_cells.append(set())
        for (x, y), c in src.items():
            g = (x + dx, y + dy)
            old = dst.get(g)
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
        def content(seg):
            return {g: c for g, c in seg.items()
                    if c.kind != OCCUPIED and not (c.pixels is not None and c.pixels.min() == c.pixels.max())}
        src, dst = content(self.segments[source]), content(self.segments[target])
        index: Dict[Tuple[Optional[str], ...], List[Cell]] = defaultdict(list)
        for g, c in dst.items():
            index[c.hashes].append(g)
        block_index = self._block_index(dst)
        src_hashes = {g: c.hashes for g, c in src.items()}
        votes = self._collect_offset_votes(src_hashes, index, block_index)
        if not votes:
            return None
        def agreement(offset: Cell) -> Tuple[int, int]:
            shared = [(c, dst[(g[0] + offset[0], g[1] + offset[1])]) for g, c in src.items()
                      if (g[0] + offset[0], g[1] + offset[1]) in dst]
            return sum(1 for a, b in shared if a.hashes == b.hashes), len(shared)

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
        key = (min(source, target), max(source, target))
        verdict = self.merge_verdicts.get(key)
        if verdict is not None and overlap["matches"] < MERGE_REJUDGE_GROWTH * verdict["matches"]:
            return verdict["same"]
        if judge is None:
            return False
        same = judge(source, target, overlap)
        self.merge_verdicts[key] = {"same": bool(same), "matches": overlap["matches"], "step": self.step,
                                    "names": (self.info(source).name, self.info(target).name)}
        self.stats["merges_judged_same" if same else "merges_judged_different"] += 1
        return bool(same)

    def consolidate(self, judge: Optional[Callable[[int, int, Dict[str, Any]], Optional[bool]]] = None
                    ) -> List[Dict[str, Any]]:
        self.prune_transient_segments()
        merged = []
        while True:
            live = [si for si, seg in enumerate(self.segments) if seg and si not in self.merged_into]
            best = None
            for i, target in enumerate(live):
                for source in live[i + 1:]:
                    o = self.overlap(source, target)
                    if (o is None or o["matches"] < CONSOLIDATE_MIN_CELLS
                            or o["agreement"] < CONSOLIDATE_MIN_AGREEMENT):
                        continue
                    if (o["votes"] < CONSOLIDATE_MARGIN * o["rival"]
                            and (o["votes"] < CONSOLIDATE_MIN_VOTES
                                 or o["agreement"] - o["rival_agreement"] < CONSOLIDATE_AGREEMENT_LEAD)):
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
                return merged
            self._merge(best["source"], best["target"], best["offset"])
            self.stats["consolidated"] += 1
            merged.append(best)

    def relocalise(self, frame: np.ndarray, exclude: Optional[int] = None, shift: Cell = (0, 0),
                   allowed: Optional[Callable[[int, Cell], bool]] = None) -> Optional[Tuple[int, Cell]]:
        a = _screen(frame)
        local: Dict[Cell, Tuple[Optional[str], ...]] = {}
        for (x, y), piece in _local_cells(a, shift):
            # blank cells match everywhere, so they do not vote
            if (x, y) == (0, 0) or piece.min() == piece.max():
                continue
            quarters = split_quarters(piece)
            local[(x, y)] = self._cell_hashes(quarters)
        best = None
        for si, seg in enumerate(self.segments):
            if si == exclude or not seg:
                continue
            index: Dict[Tuple[Optional[str], ...], List[Cell]] = defaultdict(list)
            for g, c in seg.items():
                if c.kind != OCCUPIED:
                    index[c.hashes].append(g)
            block_index = self._block_index(seg)
            votes = self._collect_offset_votes(local, index, block_index)
            if not votes:
                continue
            ranked = sorted((self._screen_agreement(seg, local, o), o)
                            for o, _ in votes.most_common(RELOCALISE_CANDIDATES))[::-1]
            close = [(o, overlap) for (m, overlap), o in ranked if m >= RELOCALISE_CLOSE_CALL * ranked[0][0][0]]
            if len(close) > 1:
                distinctive = sorted(
                    ((self._distinctive_matches(seg, index, local, o, block_index), o, overlap)
                     for o, overlap in close),
                    reverse=True,
                )
                if distinctive[0][0] - distinctive[1][0] < DISTINCTIVE_LEAD:
                    continue
                close = [(distinctive[0][1], distinctive[0][2])]
            offset, overlap = close[0]
            n = self._screen_agreement(seg, local, offset)[0]
            # a small room shows few content cells: ask for most of them rather than the fixed minimum
            needed = min(RELOCALISE_MIN_CELLS, max(10, int(0.6 * len(local))))
            if n >= needed and n >= RELOCALISE_MIN_FRACTION * overlap:
                if allowed is not None and not allowed(si, offset):
                    self.stats["relocalise_door_vetoes"] += 1
                    continue
                if best is None or n > best[0]:
                    best = (n, si, offset)
        return None if best is None else (best[1], best[2])

    @staticmethod
    def _screen_agreement(seg: Dict[Cell, MapCell], local: Dict[Cell, Tuple[Optional[str], ...]],
                          offset: Cell) -> Tuple[int, int]:
        placed = [(seg.get((offset[0] + x, offset[1] + y)), h) for (x, y), h in local.items()]
        placed = [(c, h) for c, h in placed if c is not None]
        return sum(1 for c, h in placed if c.kind != OCCUPIED and c.hashes == h), len(placed)

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
            if a == b and move is not None:
                conflicts += 1
                continue
            if a == b and near(a, pa, pb):
                continue
            ends += [(a, pa, b, pb), (b, pb, a, pa)]
        for i, (s, p, other, q) in enumerate(ends):
            conflicts += any(s == s2 and near(s, p, p2) and (other != other2 or not near(other, q, q2))
                             for s2, p2, other2, q2 in ends[i + 1:])
        return conflicts

    def merge_breaks_doors(self, source: int, target: int, offset: Cell) -> bool:
        def moved(si: int, p: Cell) -> Tuple[int, Cell]:
            return (target, (p[0] + offset[0], p[1] + offset[1])) if si == source else (si, p)

        after = [(t, *moved(a, pa), *moved(b, pb)) for t, a, pa, b, pb in self.warps]
        return (self.door_conflicts(after, self.warp_moves, self._same_door)
                > self.door_conflicts(self.warps, self.warp_moves, self._same_door))

    def arrival_breaks_doors(self, segment: int, pos: Cell) -> bool:
        if not self.faded:
            return False
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
                    if not self._can_learn_walkable(qtags):
                        continue
                cleaned_patterns.add(pattern)
            self.walk_patterns = cleaned_patterns
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
                    if c.pixels is not None and int(c.pixels.min()) == int(c.pixels.max()):
                        del seg[g]
                        seg_blocked.discard(g)
                        changed += 1
            quarter_tags: Dict[Cell, Dict[str, str]] = {}
            quarter_descriptions: Dict[Cell, Dict[str, Optional[str]]] = {}
            for g, c in seg.items():
                tiles = [tile_lookup.get(h) for h in c.hashes]
                if any(t is not None for t in tiles):
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
                elif is_east_cliff_barrier_cell(g, quarter_tags, quarter_descriptions):
                    new_tags = tuple(sorted(set(quarter_tags[g].values())))
                    new_kind = BLOCKED
                else:
                    new_tags = tuple(sorted(set(quarter_tags[g].values())))
                    new_kind = self._classify(new_tags, c.hashes)
                if g in seg_blocked and new_kind == WALKABLE and not bool(set(new_tags) & DOOR_TAGS):
                    new_kind = BLOCKED
                if (c.kind, c.tags) != (new_kind, new_tags):
                    c.kind, c.tags = new_kind, new_tags
                    changed += 1
        if self.seen_entrances is not None:
            valid = self.entrances()
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
        seg = self.segments[segment]
        near = [(max(abs(ox), abs(oy)), abs(ox) + abs(oy), g) for ox, oy in NEIGHBOURS_8 + ((0, 0),)
                for g in [(pos[0] + ox, pos[1] + oy)] if g in seg and is_door(seg[g])]
        return min(near)[2] if near else pos

    def door_labels(self) -> Dict[Tuple[int, Cell], int]:
        labels: Dict[Tuple[int, Cell], int] = {}
        for _, a, pa, b, pb in self.warps:
            ra, rpa = self.resolve(a, pa)
            rb, rpb = self.resolve(b, pb)
            if ra == rb:
                continue
            labels[(ra, self._door_at(ra, rpa))] = rb
            labels[(rb, self._door_at(rb, rpb))] = ra
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

    def record_defeat(self) -> None:
        self.defeats.append({"step": self.step, "segment": self.segment, "pos": self.pos})
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
        return {(si, g) for si, seg in enumerate(self.segments) if si not in self.merged_into
                for g, c in seg.items() if "entrance" in c.tags}

    def new_entrances(self) -> int:
        if self.seen_entrances is None:
            return 0
        return len(self.entrances() - {self.resolve(*key) for key in self.seen_entrances})

    def mark_entrances_seen(self) -> None:
        self.seen_entrances = self.entrances()

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

    def frontier(self, segment: Optional[int] = None) -> List[Cell]:
        seg = self.current if segment is None else self.segments[segment]
        out = []
        for g, c in seg.items():
            if c.kind != WALKABLE:
                continue
            if any((g[0] + dx, g[1] + dy) not in seg for dx, dy in DIRECTIONS.values()):
                out.append(g)
        return out
