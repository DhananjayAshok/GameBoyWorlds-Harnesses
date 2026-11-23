# Places are stored raw as (segment, square) and resolved through the global map's merge records on every read.

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from execution.artifact import StrategistArtifact, update
from execution.perception.pokemon import sprites as S
from execution.perception.pokemon.grid import LEFT_COL, TOP_ROW, cell_box
from execution.strategist.pokemon.global_map import (DIRECTIONS, OCCUPIED, PX_SLACK, GlobalMap, StepEvent,
                                                     grid_offset, grid_residual)
from execution.strategist.pokemon import npc_model
from execution.strategist.pokemon.npc_map import LABEL_NOTE, WHERE_NOTES, maybe_lines
from execution.strategist.pokemon.npc_model import display_name
from execution.strategist.pokemon.npc_names import draw_name
from utils import log_info, log_warn

Cell = Tuple[int, int]
FACINGS = ("down", "up", "left", "right")
KINDS = ("person", "trainer", "pokemon", "object", "unknown")

PLACED_STATUS = {"start", "moved", "stayed", "relocalised", "new_segment"}
# The same look within this many squares of where a person was last seen is that person.
NEAR = 4
# A new person is compared with at most this many candidates by a model.
MAX_CANDIDATES = 8
# A person standing still is noted again after this many map steps.
SIGHTING_EVERY = 10
# Fewest dark pixels for a sprite caught mid-step or half hidden to count as a person.
MIN_TAGGED_BLACK = 12
# Map-square tags a conversation can be with besides a person.
SIGN_TAGS = {"sign", "interactable object", "item"}
# At most this many dialogue pictures of one conversation are read by the model.
MAX_OCR_CROPS = 12
OCR_PROMPT = "If there is no text in the image, just say NONE. Otherwise, perform OCR and state the text in this image:"
# People not seen for more than this many episodes are left out of the strategist's list.
STALE_EPISODES = 20
# A half-recognised person needs this many person-tagged quarters, this many dark pixels and this many map steps.
WEAK_QUARTERS = 2
WEAK_BLACK = 60
WEAK_FRAMES = 2
# A person not seen for more than this many episodes is drawn grey on the map.
STALE_MARK_EPISODES = 3
# A sprite-like square tagged as a person is shown to a model after this many map steps, MAX_CONFIRM per episode.
CONFIRM_FRAMES = 2
MAX_CONFIRM = 6
# Squares kept around such a square, in every direction.
CONFIRM_REACH = 2


@dataclass
class View:
    pixels: np.ndarray
    key: np.ndarray
    facing: Optional[str]
    mirrored_of: Optional[int]
    first_seen: Tuple[int, int]
    n_seen: int = 1
    partial: bool = False


@dataclass
class Look:
    id: str
    views: List[View]
    kind_hint: Optional[str] = None
    links: List[Tuple[int, int, str]] = field(default_factory=list)


@dataclass
class Sighting:
    episode: int
    map_step: int
    segment: int
    square: Cell
    view: int
    offset: Cell = (0, 0)
    track: int = -1
    flipped: bool = False


@dataclass
class Conversation:
    id: str
    npc: Optional[str]
    sign: Optional[str]
    episode: int
    map_step_start: int
    map_step_end: Optional[int]
    segment: int
    square: Cell
    target: Optional[Cell]
    trigger: str
    crops: List[np.ndarray] = field(default_factory=list)
    text: str = ""
    summary: str = ""
    speaker_name: Optional[str] = None
    outcome: Optional[str] = None
    led_to: Optional[str] = None
    complete: bool = False
    read: bool = False
    context: Optional[np.ndarray] = None
    summarised: bool = False
    name_type: Optional[str] = None
    tries: int = 0
    how: str = "seen"
    confidence: str = "high"


@dataclass
class NPC:
    id: str
    name: str
    look: str
    kind: str = "unknown"
    description: str = ""
    real_name: Optional[str] = None
    name_type: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    sightings: List[Sighting] = field(default_factory=list)
    conversations: List[str] = field(default_factory=list)
    needs: Optional[str] = None
    candidates: List[str] = field(default_factory=list)
    created: Tuple[int, int] = (0, 0)
    merged_into: Optional[str] = None
    context: Optional[np.ndarray] = None
    context_cell: Optional[Cell] = None
    tries: int = 0
    hidden: Optional[str] = None


@dataclass
class Sign:
    id: str
    segment: int
    square: Cell
    conversations: List[str] = field(default_factory=list)
    text: str = ""
    episode: int = 0


@dataclass
class MaybeSquare:
    """A sprite-like square tagged as a person where nobody was recognised yet."""
    frames: int = 0
    last_step: int = -1
    crop: Optional[bytes] = None
    box: Cell = (0, 0)
    distance: int = 99
    seen: Tuple[int, int] = (0, 0)
    verdict: Optional[str] = None
    tries: int = 0


@dataclass
class Track:
    id: int
    npc: str
    segment: int
    square: Cell
    view: int
    last_step: int
    steps_seen: int = 1
    ambiguous: bool = False
    last_sight_step: int = -10_000
    last_sight_square: Optional[Cell] = None
    last_sight_flipped: bool = False


@dataclass
class PersonMark:
    """One person on the map for one area at one map step, derived on every read and never stored."""
    npc_id: str
    number: int
    name: str
    kind: str
    square: Cell
    facing: Optional[str]
    last_seen: Tuple[int, int]
    on_screen: bool
    talked: str
    times_talked: int
    same_words: bool
    earlier: List[Cell]
    spread: Optional[Tuple[int, int, int, int]]
    stale: bool
    when: str


def frame_residual(frame: np.ndarray) -> Optional[Cell]:
    shift = grid_offset(frame)
    if shift is None:
        return None
    residual = grid_residual(shift)
    return residual if max(abs(residual[0]), abs(residual[1])) <= PX_SLACK else None


def _manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


class NPCRegistry(StrategistArtifact):
    """Every person the player has seen in one run."""

    _TRANSIENT = ("_gmap", "_episode", "_tracks", "_last_people", "_last_people_at", "_stack", "_stack_ref",
                  "_look_merges", "_weak", "_facing", "_open", "_frame", "_free_frame", "_free_identified",
                  "_live", "_a_step", "_a_facing", "_resume", "_last_closed")

    def __init__(self, game: str, run_name: str) -> None:
        self.game = game
        self.run_name = run_name
        self.looks: Dict[str, Look] = {}
        self.npcs: Dict[str, NPC] = {}
        self.signs: Dict[str, Sign] = {}
        self.conversations: Dict[str, Conversation] = {}
        self.unattributed: List[str] = []
        self.look_aliases: Dict[str, str] = {}
        # Two people seen in one frame are never the same person.
        self.covisible: Set[FrozenSet[str]] = set()
        self.next_ids: Counter = Counter()
        self.stats: Counter = Counter()
        # (raw segment, square) tagged as a person -> whether it was ever drawn like a sprite there.
        self.tagged_squares: Dict[Tuple[int, Cell], bool] = {}
        self.maybe_squares: Dict[Tuple[int, Cell], MaybeSquare] = {}
        # The conversation the last episode ended in the middle of; the next one may open in its text box.
        self.interrupted: Optional[str] = None
        self.pending_press: Optional[Tuple[int, Optional[str], int, Cell]] = None
        self.last_facing: Optional[str] = None
        self.version = 1
        self._reset_transient()

    def _reset_transient(self) -> None:
        self._gmap: Optional[GlobalMap] = None
        self._episode = 0
        self._tracks: Dict[int, Track] = {}
        self._last_people: Dict[Cell, str] = {}
        self._last_people_at: Optional[Tuple[int, int]] = None
        self._stack: Optional[np.ndarray] = None
        self._stack_ref: List[Tuple[str, int]] = []
        self._look_merges: Set[Tuple[str, str]] = set()
        self._facing: Optional[str] = None
        self._open: Optional[str] = None
        self._weak: Dict[Tuple[int, Cell], Set[int]] = {}
        self._frame: Optional[np.ndarray] = None
        self._free_frame: Optional[np.ndarray] = None
        self._free_identified: Dict[Any, Any] = {}
        self._live: Optional[Callable[[str], Callable[..., Any]]] = None
        self._a_step: Optional[int] = None
        self._a_facing: Optional[str] = None
        self._resume: Optional[str] = None
        self._last_closed: Optional[str] = None

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        self._reset_transient()
        for name, default in (("look_aliases", dict), ("covisible", set), ("unattributed", list),
                              ("signs", dict), ("conversations", dict), ("stats", Counter),
                              ("next_ids", Counter), ("version", lambda: 1), ("tagged_squares", dict),
                              ("maybe_squares", dict), ("interrupted", lambda: None),
                              ("pending_press", lambda: None), ("last_facing", lambda: None)):
            self.__dict__.setdefault(name, default())

    @property
    def _p(self) -> Dict[str, Any]:
        return S.params(self.game)

    def _next(self, prefix: str) -> str:
        self.next_ids[prefix] += 1
        return f"{prefix}_{self.next_ids[prefix]:03d}"

    def _taken_names(self) -> List[str]:
        return [x for n in self.npcs.values() for x in (n.name, n.real_name, *n.aliases) if x]

    def get(self, npc_id: str) -> NPC:
        npc = self.npcs[npc_id]
        while npc.merged_into is not None:
            npc = self.npcs[npc.merged_into]
        return npc

    def live_npcs(self) -> List[NPC]:
        return [n for n in self.npcs.values() if n.merged_into is None]

    def seen_whole(self, npc: NPC) -> bool:
        return any(not v.partial for v in self.look_of(npc).views)

    def look_of(self, npc: NPC) -> Look:
        look_id = npc.look
        while look_id in self.look_aliases:
            look_id = self.look_aliases[look_id]
        return self.looks[look_id]

    def place(self, gmap: GlobalMap, sighting: Sighting) -> Tuple[int, Cell]:
        return gmap.resolve(sighting.segment, sighting.square)

    def last_seen(self, gmap: GlobalMap, npc: NPC) -> Optional[Tuple[int, Cell, Sighting]]:
        if not npc.sightings:
            return None
        newest = max(npc.sightings, key=lambda s: s.map_step)
        segment, square = self.place(gmap, newest)
        return segment, square, newest

    def squares_in(self, gmap: GlobalMap, npc: NPC, segment: int) -> List[Cell]:
        out: List[Cell] = []
        for s in sorted(npc.sightings, key=lambda s: -s.map_step):
            seg, square = self.place(gmap, s)
            if seg == segment and square not in out:
                out.append(square)
        return out

    def _forget_screen(self) -> None:
        self._last_people, self._last_people_at = {}, None

    def on_screen(self, gmap: GlobalMap) -> Dict[Cell, str]:
        # Keyed by square relative to the player's current position.
        if self._last_people_at is None or self._last_people_at[0] != gmap.segment:
            return {}
        px, py = gmap.pos
        return {(x - px, y - py): npc for (x, y), npc in self._last_people.items()}

    def in_view(self, gmap: GlobalMap, square: Cell) -> bool:
        if self._last_people_at is None or self._last_people_at[0] != gmap.segment:
            return False
        dx, dy = square[0] - gmap.pos[0], square[1] - gmap.pos[1]
        return LEFT_COL <= dx <= LEFT_COL + 9 and TOP_ROW - 8 <= dy <= TOP_ROW

    def begin_episode(self, episode: int, gmap: GlobalMap) -> None:
        self._gmap, self._episode = gmap, episode
        self._tracks = {}
        self._forget_screen()
        self._facing, self._open, self._last_closed = self.last_facing, None, None
        self._a_step, self._a_facing = None, None
        if self.pending_press is not None:
            p_step, p_facing, p_seg, p_pos = self.pending_press
            if gmap.resolve(gmap.segment, gmap.pos) == gmap.resolve(p_seg, p_pos):
                self._a_step, self._a_facing = p_step, p_facing
                if p_facing is not None:
                    self._facing = p_facing
        self.pending_press = None
        self._resume, self.interrupted = self.interrupted, None

    def on_opening_screen(self, frame: np.ndarray, identified: Dict[Any, Any], free: bool) -> None:
        if free and self._gmap is not None:
            self._process(frame, identified, "start", 0)
            self._free_frame = frame
            self._free_identified = dict(identified or {})

    def on_env_step(self, ev: StepEvent) -> None:
        if self._gmap is None:
            return
        # Conversations first: dialogue steps are not free.
        try:
            self._conversation_step(ev)
        except Exception as e:
            log_warn(f"NPC registry: conversation step skipped: {type(e).__name__}: {e}")
        if not ev.free:
            return
        if ev.status not in PLACED_STATUS:
            self._forget_screen()
            return
        self._process(ev.frame, ev.identified, ev.status, ev.max_steps)
        self._free_frame = ev.frame
        self._free_identified = dict(ev.identified or {})

    def _view_stack(self) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        if self._stack is None:
            keys, ref = [], []
            for look in self.looks.values():
                for i, view in enumerate(look.views):
                    if not view.partial:
                        keys.append(view.key)
                        ref.append((look.id, i))
            self._stack = np.stack(keys) if keys else np.zeros((0, 16, 16), dtype=bool)
            self._stack_ref = ref
        return self._stack, self._stack_ref

    def _best_view(self, mask: np.ndarray) -> Optional[Tuple[int, str, int]]:
        keys, ref = self._view_stack()
        if len(keys) == 0:
            return None
        distances, _ = S.stack_distances(keys, mask)
        index = int(distances.argmin())
        return int(distances[index]), ref[index][0], ref[index][1]

    def _detect(self, frame: np.ndarray, identified: Dict[Any, Any]) -> List[Dict[str, Any]]:
        p = self._p
        level = S.black_level(frame)
        cells = S.find_npc_cells(identified, p["min_quarters"])
        found: List[Dict[str, Any]] = []
        for cell in cells:
            window = S.sprite_window(frame, cell, (0, 0), p["y_offset"])
            if window is None:
                continue
            mask = S.black_mask(window, level)
            if int(mask.sum()) < MIN_TAGGED_BLACK:
                continue
            found.append(dict(cell=cell, offset=(0, 0), window=window, mask=mask,
                              full=S.full_window_ok(identified, cell) and S.complete_sprite(mask)))
        for cell in S.find_npc_cells(identified, WEAK_QUARTERS):
            if cell in cells:
                continue
            window = S.sprite_window(frame, cell, (0, 0), p["y_offset"])
            if window is None:
                continue
            mask = S.black_mask(window, level)
            if int(mask.sum()) >= WEAK_BLACK:
                found.append(dict(cell=cell, offset=(0, 0), window=window, mask=mask, full=False, weak=True))
        keys, _ = self._view_stack()
        if len(keys):
            for cell, offset, _, _ in S.scan_for_views(frame, keys, S.scan_cells(identified, cells), level,
                                                       p["scan_range"], p["scan_step"], p["auto"], p["y_offset"],
                                                       p["min_black"]):
                window = S.sprite_window(frame, cell, offset, p["y_offset"])
                if window is not None:
                    found.append(dict(cell=cell, offset=offset, window=window, mask=S.black_mask(window, level),
                                      full=False))
        return found

    @staticmethod
    def _one_per_sprite(detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Detections whose windows overlap are one sprite: keep the whole, tag-backed one.
        def origin(det):
            x0, y0, _, _ = cell_box(*det["cell"])
            return x0 + det["offset"][0], y0 + det["offset"][1]
        ranked = sorted(detections, key=lambda d: (bool(d.get("weak")), not d["full"], -int(d["mask"].sum())))
        kept: List[Dict[str, Any]] = []
        for det in ranked:
            ox, oy = origin(det)
            if all(abs(ox - origin(k)[0]) >= 12 or abs(oy - origin(k)[1]) >= 12 for k in kept):
                kept.append(det)
        return kept

    def _confirm_weak(self, detections: List[Dict[str, Any]], segment: int, step: int) -> List[Dict[str, Any]]:
        kept = [d for d in detections if not d.get("weak")]
        claimed = [d["square"] for d in kept]
        for det in sorted((d for d in detections if d.get("weak")), key=lambda d: -int(d["mask"].sum())):
            if any(_manhattan(det["square"], square) <= 1 for square in claimed):
                continue
            claimed.append(det["square"])
            seen = self._weak.setdefault((segment, det["square"]), set())
            seen.add(step)
            if len(seen) >= WEAK_FRAMES:
                det["full"] = True
                kept.append(det)
        return kept

    def _process(self, frame: np.ndarray, identified: Dict[Any, Any], status: str, max_steps: int) -> None:
        gmap = self._gmap
        self._frame = frame
        if status in ("relocalised", "new_segment"):
            self._tracks = {}
        if gmap.residual != (0, 0) and frame_residual(frame) != (0, 0):
            # Tags are for the unshifted grid; the map's residual can be stale, so the frame's own grid decides.
            self._forget_screen()
            return
        pos, segment, step = gmap.pos, gmap.segment, gmap.step
        like = self._judge_tagged(frame, identified, segment, pos)
        detections = self._detect(frame, identified)
        for det in detections:
            det["square"] = (pos[0] + det["cell"][0], pos[1] + det["cell"][1])
        detections = self._confirm_weak(self._one_per_sprite(detections), segment, step)
        live ={t.id: t for t in self._tracks.values() if t.segment == segment}
        allowance = max(1, min(max_steps, 4))

        # Detections close to each other could be confused, so their tracks never link views.
        crowded = {i for i, a in enumerate(detections) for j, b in enumerate(detections)
                   if i != j and _manhattan(a["square"], b["square"]) <= 2}

        pairs = []
        for i, det in enumerate(detections):
            for track in live.values():
                moved = _manhattan(det["square"], track.square)
                if moved <= allowance:
                    view = self.look_of(self.npcs[track.npc]).views[track.view]
                    pairs.append((moved, S.mask_distance(det["mask"], view.key)[0], i, track.id))
        pairs.sort()
        assigned: Dict[int, int] = {}
        taken: Set[int] = set()
        for _, _, i, tid in pairs:
            if i not in assigned and tid not in taken:
                assigned[i] = tid
                taken.add(tid)

        now: Dict[int, Track] = {}
        for i, tid in assigned.items():
            track = live[tid]
            self._update_track(track, detections[i], i in crowded, step)
            now[track.id] = track
        busy = {t.npc for t in now.values()}
        for i, det in enumerate(detections):
            if i in assigned:
                continue
            track = self._start_track(det, busy, segment, step)
            if track is not None:
                track.ambiguous = i in crowded
                now[track.id] = track
                busy.add(track.npc)
        self._tracks = now
        people = [t.npc for t in now.values()]
        self.covisible.update(frozenset(pair) for pair in _pairs(people))
        squares = {d["square"] for d in detections}
        self._last_people = {t.square: t.npc for t in now.values() if t.square in squares}
        self._last_people_at = (segment, step)
        self.stats["frames_with_people"] += bool(now)
        self._note_maybe(frame, like, {t.square for t in now.values()}, segment, pos, step)

    def _judge_tagged(self, frame: np.ndarray, identified: Dict[Any, Any], segment: int, pos: Cell) -> List[Cell]:
        like = []
        for cell in {(x, y) for (x, y, _), tile in identified.items() if tile.tag == "npc" and (x, y) != (0, 0)}:
            key = (segment, (pos[0] + cell[0], pos[1] + cell[1]))
            if S.sprite_like(frame, cell, self._p["y_offset"]):
                self.tagged_squares[key] = True
                like.append(cell)
            else:
                self.tagged_squares.setdefault(key, False)
        return like

    def _known_squares(self, segment: int) -> List[Cell]:
        out = []
        for npc in self.live_npcs():
            for sighting in npc.sightings:
                seg, square = self.place(self._gmap, sighting)
                if seg == segment:
                    out.append(square)
        return out

    def _note_maybe(self, frame: np.ndarray, like: List[Cell], tracked: Set[Cell], segment: int, pos: Cell,
                    step: int) -> None:
        import cv2

        if not like:
            return
        resolved_segment = self._gmap.resolve(segment, pos)[0]
        known = None
        screen = S._screen(frame)
        for cell in like:
            square = (pos[0] + cell[0], pos[1] + cell[1])
            if any(_manhattan(square, t) <= 1 for t in tracked):
                continue
            entry = self.maybe_squares.get((segment, square))
            if entry is not None and entry.verdict is not None:
                continue
            if known is None:
                known = self._known_squares(resolved_segment)
            if any(_manhattan(self._gmap.resolve(segment, square)[1], k) <= 1 for k in known):
                continue
            entry = self.maybe_squares.setdefault((segment, square), MaybeSquare())
            if entry.last_step != step:
                entry.frames, entry.last_step = entry.frames + 1, step
            distance = abs(cell[0]) + abs(cell[1])
            if entry.crop is None or distance < entry.distance:
                x0, y0, _, _ = cell_box(*cell)
                reach = CONFIRM_REACH * 16
                left, top = max(0, x0 - reach), max(0, y0 - reach)
                crop = screen[top:min(screen.shape[0], y0 + 16 + reach), left:min(screen.shape[1], x0 + 16 + reach)]
                entry.crop = cv2.imencode(".png", np.ascontiguousarray(crop, dtype=np.uint8))[1].tobytes()
                entry.box, entry.distance, entry.seen = (x0 - left, y0 - top), distance, (self._episode, step)

    def background_squares(self, gmap: GlobalMap) -> Set[Tuple[int, Cell]]:
        return ({gmap.resolve(*key) for key, sprite in self.tagged_squares.items() if not sprite}
                | {gmap.resolve(*key) for key, m in self.maybe_squares.items() if m.verdict == "nothing"})

    def hidden_person(self, segment: int, square: Cell, why: str, when: Tuple[int, int],
                      window: Optional[np.ndarray] = None, level: int = 0, needs: Optional[str] = "card") -> NPC:
        # The look is marked partial so it never matches anyone; square is raw.
        if window is None or window.shape != (16, 16):
            window = np.full((16, 16), 255, np.uint8)
        look = Look(id=self._next("look"), views=[View(np.array(window, copy=True), S.black_mask(window, level), None,
                                                       None, when, partial=True)])
        self.looks[look.id] = look
        npc = self._new_npc(look, needs, [], when[1])
        npc.created, npc.hidden = when, why
        npc.sightings.append(Sighting(when[0], when[1], segment, square, 0))
        self.stats[f"hidden_by_{why}"] += 1
        return npc

    def _hidden_at(self, offset: Cell) -> Optional[str]:
        g = self._gmap
        square = (g.pos[0] + offset[0], g.pos[1] + offset[1])
        segment, resolved = g.resolve(g.segment, square)
        entry = self.maybe_squares.get((g.segment, square))
        if entry is not None and entry.verdict == "nothing":
            return None
        if not (self.tagged_squares.get((g.segment, square)) or resolved in self.unconfirmed(g, segment)):
            return None
        if any(_manhattan(resolved, k) <= 1 for k in self._known_squares(segment)):
            return None
        window, level = None, 0
        if self._free_frame is not None:
            window = S.sprite_window(self._free_frame, offset, (0, 0), self._p["y_offset"])
            level = S.black_level(self._free_frame)
        npc = self.hidden_person(g.segment, square, "conversation", (self._episode, g.step), window, level)
        if self._free_frame is not None:
            npc.context, npc.context_cell = np.array(self._free_frame, copy=True), offset
        if entry is not None:
            entry.verdict, entry.crop = "person", None
        return npc.id

    def confirm_candidates(self, gmap: GlobalMap) -> List[Tuple[Tuple[int, Cell], MaybeSquare]]:
        out, done = [], set()
        for key, m in sorted(self.maybe_squares.items(), key=lambda kv: -kv[1].frames):
            if m.verdict is not None or m.crop is None:
                continue
            segment, square = gmap.resolve(*key)
            if (segment, square) in done:
                continue
            if m.frames < CONFIRM_FRAMES and square not in self.unconfirmed(gmap, segment):
                continue
            if any(_manhattan(square, k) <= 1 for k in self._known_squares(segment)):
                continue
            done.add((segment, square))
            out.append((key, m))
        return out[:MAX_CONFIRM]

    def _new_look(self, det: Dict[str, Any], step: int) -> Look:
        look = Look(id=self._next("look"), views=[View(det["window"].copy(), det["mask"].copy(), None, None,
                                                       (self._episode, step))])
        self.looks[look.id] = look
        self._stack = None
        self.stats["looks_created"] += 1
        return look

    def _new_npc(self, look: Look, needs: str, candidates: List[str], step: int) -> NPC:
        npc_id = self._next("npc")
        npc = NPC(id=npc_id, name=draw_name(self.game, self.run_name, npc_id, self._taken_names()), look=look.id,
                  needs=needs, candidates=candidates[:MAX_CANDIDATES], created=(self._episode, step))
        self.npcs[npc_id] = npc
        self.stats["npcs_created"] += 1
        return npc

    def _near(self, segment: int, square: Cell, exclude: Set[str], look_id: Optional[str] = None) -> List[NPC]:
        gmap = self._gmap
        found = []
        for npc in self.live_npcs():
            if npc.id in exclude or (look_id is not None and self.look_of(npc).id != look_id):
                continue
            seen = self.last_seen(gmap, npc)
            if seen is None or seen[0] != segment:
                continue
            distance = _manhattan(seen[1], square)
            if distance <= NEAR:
                found.append((distance, -seen[2].map_step, npc))
        return [npc for _, _, npc in sorted(found, key=lambda x: (x[0], x[1]))]

    def _static_at(self, segment: int, square: Cell, busy: Set[str]) -> Optional[NPC]:
        gmap = self._gmap
        for npc in self.live_npcs():
            if npc.id in busy:
                continue
            seen = self.last_seen(gmap, npc)
            if seen is None or seen[0] != segment or seen[1] != square:
                continue
            if len({self.place(gmap, s) for s in npc.sightings}) == 1:
                return npc
        return None

    def _start_track(self, det: Dict[str, Any], busy: Set[str], segment: int, step: int) -> Optional[Track]:
        p, square = self._p, det["square"]
        best = self._best_view(det["mask"])
        npc: Optional[NPC] = None
        view_index = 0
        if best is not None and best[0] <= p["auto"]:
            look = self.looks[best[1]]
            if look.kind_hint == "object":
                self.stats["object_hits"] += 1
                return None
            same = [n for n in self.live_npcs() if self.look_of(n).id == look.id and n.id not in busy]
            near = self._near(segment, square, busy, look.id)
            static = None if near else self._static_at(segment, square, busy)
            if near:
                npc, view_index = near[0], best[2]
            elif static is not None:
                # Someone who never moves was last seen here: they turned, even if the picture is a look-alike's.
                npc, view_index = static, self._turned(static, det, step)
                if det["full"] and self.look_of(static).id != look.id:
                    self._look_merges.add((self.look_of(static).id, look.id))
            elif det["full"] or int(det["mask"].sum()) >= p["min_black"]:
                # A look-alike elsewhere is another person until a model or a conversation says otherwise.
                recent = sorted(same, key=lambda n: -max((s.map_step for s in n.sightings), default=-1))
                npc = self._new_npc(look, "judge" if same else "card", [n.id for n in recent], step)
                view_index = best[2]
            else:
                return None
        else:
            if not det["full"]:
                npc = self._static_at(segment, square, busy)
                if npc is None or not npc.hidden or not S.sprite_like(self._frame, det["cell"], p["y_offset"]):
                    self.stats["ignored_partial"] += 1
                    return None
                # Already found but never seen whole, still there: a partial picture.
                view_index = self._partial_view(npc, det, step)
                return self._begin_track(npc, det, segment, step, view_index)
            npc = self._static_at(segment, square, busy)
            if npc is not None:
                view_index = self._turned(npc, det, step)
            else:
                look = self._new_look(det, step)
                candidates = [n.id for n in self._near(segment, square, busy)]
                if best is not None and best[0] <= p["maybe"]:
                    candidates += [n.id for n in self.live_npcs()
                                   if self.look_of(n).id == best[1] and n.id not in busy and n.id not in candidates]
                npc = self._new_npc(look, "card", candidates, step)
        return self._begin_track(npc, det, segment, step, view_index)

    def _begin_track(self, npc: NPC, det: Dict[str, Any], segment: int, step: int, view_index: int) -> Track:
        if npc.context is None and self._frame is not None:
            npc.context, npc.context_cell = np.array(self._frame, copy=True), det["cell"]
        self.stats["tracks_started"] += 1
        self.next_ids["track"] += 1
        track = Track(id=self.next_ids["track"], npc=npc.id, segment=segment, square=det["square"], view=view_index,
                      last_step=step)
        self._note_sighting(track, det, step)
        return track

    def _partial_view(self, npc: NPC, det: Dict[str, Any], step: int) -> int:
        look = self.look_of(npc)
        partial = [(S.mask_distance(det["mask"], v.key)[0], i) for i, v in enumerate(look.views) if v.partial]
        if partial and min(partial)[0] <= self._p["auto"]:
            look.views[min(partial)[1]].n_seen += 1
            return min(partial)[1]
        look.views.append(View(det["window"].copy(), det["mask"].copy(), None, None, (self._episode, step),
                               partial=True))
        look.links.append((0, len(look.views) - 1, "partial"))
        self.stats["partial_views"] += 1
        return len(look.views) - 1

    def _turned(self, npc: NPC, det: Dict[str, Any], step: int) -> int:
        look = self.look_of(npc)
        last = self.last_seen(self._gmap, npc)[2]
        if not det["full"]:
            return last.view
        look.views.append(View(det["window"].copy(), det["mask"].copy(), None, None, (self._episode, step)))
        look.links.append((last.view, len(look.views) - 1, "same_square"))
        self._stack = None
        self.stats["views_added"] += 1
        self.stats["same_square"] += 1
        return len(look.views) - 1

    def _update_track(self, track: Track, det: Dict[str, Any], crowded: bool, step: int) -> None:
        p = self._p
        npc = self.get(track.npc)
        look = self.look_of(npc)
        current = look.views[track.view]
        distance, _ = S.mask_distance(det["mask"], current.key)
        track.square, track.last_step = det["square"], step
        track.steps_seen += 1
        track.ambiguous = track.ambiguous or crowded
        if distance <= p["auto"]:
            current.n_seen += 1
        elif det["full"] and not track.ambiguous:
            nearest = sorted((S.mask_distance(det["mask"], v.key)[0], i) for i, v in enumerate(look.views))[0]
            if nearest[0] <= p["auto"]:
                track.view = nearest[1]
                look.views[nearest[1]].n_seen += 1
            else:
                look.views.append(View(det["window"].copy(), det["mask"].copy(), None, None,
                                       (self._episode, step)))
                look.links.append((track.view, len(look.views) - 1, "track"))
                track.view = len(look.views) - 1
                self._stack = None
                self.stats["views_added"] += 1
            other = self._best_view(det["mask"])
            if other is not None and other[1] != look.id and other[0] <= p["auto"]:
                self._look_merges.add((look.id, other[1]))
        self._note_sighting(track, det, step)

    def _note_sighting(self, track: Track, det: Dict[str, Any], step: int) -> None:
        # The picture comparison allows a left/right mirror, so a turn is recorded per sighting and makes a new one.
        views = self.look_of(self.npcs[track.npc]).views
        flipped = bool(track.view < len(views) and S.mask_distance(det["mask"], views[track.view].key)[1])
        if (track.last_sight_square == det["square"] and step - track.last_sight_step < SIGHTING_EVERY
                and flipped == track.last_sight_flipped):
            return
        self.npcs[track.npc].sightings.append(Sighting(self._episode, step, track.segment, det["square"], track.view,
                                                       det["offset"], track.id, flipped))
        track.last_sight_step, track.last_sight_square, track.last_sight_flipped = step, det["square"], flipped
        self.stats["sightings"] += 1

    @staticmethod
    def _is_a_press(action: Any, kwargs: Dict[str, Any]) -> bool:
        from execution.game_api import InteractAction, LowLevelAction, LowLevelActions, PassDialogueAction

        if isinstance(action, type) and issubclass(action, (InteractAction, PassDialogueAction)):
            return True
        return (isinstance(action, type) and issubclass(action, LowLevelAction)
                and kwargs.get("low_level_action") == LowLevelActions.PRESS_BUTTON_A)

    @staticmethod
    def _step_crops(ev: StepEvent) -> List[np.ndarray]:
        new: List[np.ndarray] = []
        if not ev.info:
            return new
        for region in (ev.info.get("ocr") or {}).get("transition_ocr_regions") or []:
            stack = region.get("dialogue") if isinstance(region, dict) else None
            if stack is not None and len(stack):
                new.extend(np.asarray(c) for c in stack)
        details = (ev.info.get("core") or {}).get("previous_action_details")
        if isinstance(details, (tuple, list)) and len(details) >= 3 and isinstance(details[2], list):
            for state in details[2]:
                if not isinstance(state, dict):
                    continue
                stack = ((state.get("ocr") or {}).get("ocr_regions") or {}).get("dialogue")
                if stack is not None and len(stack):
                    new.extend(np.asarray(c) for c in stack)
        return new

    def _conversation_step(self, ev: StepEvent) -> None:
        from execution.game_api import AgentState

        state = ev.info.get("pokemon_core", {}).get("agent_state") if ev.info else None
        if self._resume is not None:
            self._continue_conversation(ev)
        if ev.direction is not None and ev.was_free:
            self._facing = ev.direction
        elif ev.status in ("relocalised", "new_segment"):
            self._facing = None
        elif ev.status == "moved" and ev.direction is None:
            self._facing = None
        self.last_facing = self._facing

        a_press = self._is_a_press(ev.action, ev.kwargs)
        # A 1-2 tick delayed text box can open after the A press and up to 2 settle WaitActions.
        pressed = a_press or (self._a_step is not None and 0 < self._gmap.step - self._a_step <= 3)
        if a_press and ev.was_free:
            self._a_step = self._gmap.step
            self._a_facing = self._facing
        elif not ev.free or (self._a_step is not None and self._gmap.step - self._a_step > 3) or (
                ev.was_free and ev.free and ev.direction is not None):
            self._a_step = None
            self._a_facing = None

        crops = self._step_crops(ev)
        # Text opens a conversation: either entering IN_DIALOGUE from free roam, or a single-step dialogue with crops.
        if self._open is None and ev.was_free and (
                state == AgentState.IN_DIALOGUE
                or (crops and state in (AgentState.FREE_ROAM, AgentState.IN_MENU))):
            if not pressed and self._try_reopen_pre_battle(ev):
                pass
            else:
                self._open_conversation(ev, pressed)
            self._a_step = None
            self._a_facing = None
        if self._open is not None:
            self._collect_crops(ev, crops)
            if state == AgentState.FREE_ROAM:
                self._close_conversation("free_roam", True)
            elif state == AgentState.IN_BATTLE:
                self._close_conversation("battle", True)
        elif state == AgentState.IN_BATTLE:
            self._upgrade_recent_to_battle(crops)

    def _recent_pre_battle_conversation(self) -> Optional[Conversation]:
        if self._last_closed is None or self._gmap is None:
            return None
        conv = self.conversations.get(self._last_closed)
        if conv is None or conv.episode != self._episode or conv.trigger == "battle_start":
            return None
        if conv.led_to != "free_roam" or conv.sign is not None:
            return None
        end_step = conv.map_step_end if conv.map_step_end is not None else conv.map_step_start
        if self._gmap.step - end_step > 4:
            return None
        if self._gmap.resolve(self._gmap.segment, self._gmap.pos) != self._gmap.resolve(conv.segment, conv.square):
            return None
        return conv

    def _try_reopen_pre_battle(self, ev: StepEvent) -> bool:
        conv = self._recent_pre_battle_conversation()
        if conv is None or not conv.crops:
            return False
        conv.complete, conv.led_to, conv.map_step_end = False, None, None
        conv.text, conv.read, conv.summarised, conv.tries = "", False, False, 0
        self._open = conv.id
        self.stats["conversations_continued"] += 1
        return True

    def _upgrade_recent_to_battle(self, crops: Optional[List[np.ndarray]] = None) -> None:
        from execution.pokemon.executors import drop_prefix_crops

        conv = self._recent_pre_battle_conversation()
        if conv is None:
            return
        if crops:
            drop_prefix_crops(crops, conv.crops)
        conv.led_to, conv.map_step_end = "battle", self._gmap.step
        if conv.npc in self.npcs and self.get(conv.npc).kind in ("unknown", "person"):
            self.get(conv.npc).kind = "trainer"
        if not conv.crops and not conv.text and conv.npc is None and conv.sign is None:
            self._mark_battle_start(conv)

    def _continue_conversation(self, ev: StepEvent) -> None:
        conv = self.conversations.get(self._resume)
        self._resume = None
        g = self._gmap
        if (conv is None or self._open is not None or ev.was_free
                or g.resolve(g.segment, g.pos) != g.resolve(conv.segment, conv.square)):
            return
        conv.complete, conv.led_to, conv.map_step_end = False, None, None
        conv.text, conv.read, conv.summarised, conv.tries = "", False, False, 0
        self._open = conv.id
        self.stats["conversations_continued"] += 1

    def _cell_tags(self, target: Cell, ev: Optional[StepEvent] = None) -> Set[str]:
        g = self._gmap
        square = (g.pos[0] + target[0], g.pos[1] + target[1])
        tags: Set[str] = set()
        cell = g.segments[g.segment].get(square)
        if cell is not None and cell.tags:
            tags.update(cell.tags)
        for identified in ((ev.identified if ev is not None else None), self._free_identified):
            if not identified:
                continue
            tags.update(tile.tag for (x, y, _), tile in identified.items() if (x, y) == target)
        return tags

    def _sign_at(self, target: Cell, ev: Optional[StepEvent] = None) -> Optional[Cell]:
        g = self._gmap
        square = (g.pos[0] + target[0], g.pos[1] + target[1])
        return square if (self._cell_tags(target, ev) & SIGN_TAGS) else None

    def _across_counter_speaker(self, target: Cell, on_screen: Dict[Cell, str],
                                ev: Optional[StepEvent] = None) -> Tuple[Optional[str], str, str]:
        if self._cell_tags(target, ev) & {"sign", "item"}:
            return None, "seen", "high"
        across = (2 * target[0], 2 * target[1])
        speaker = on_screen.get(across)
        if speaker is not None:
            return speaker, "seen", "high"
        speaker = self._remembered_at(across)
        if speaker is not None:
            return speaker, "remembered", "high"
        speaker = self._hidden_at(across)
        if speaker is not None:
            return speaker, "hidden", "low"
        return None, "seen", "high"

    def _emote_cell(self, frame: Optional[np.ndarray]) -> Optional[Cell]:
        from execution.perception.pokemon.battle import DARK_INK, LIGHT_BACKGROUND, _grey_frame

        if frame is None:
            return None
        img = _grey_frame(frame)
        h, w = img.shape[:2]
        if h < 120 or w < 160:
            return None
        max_y = min(h - 9, 96)
        dark = img < DARK_INK
        light = img >= LIGHT_BACKGROUND
        for x0 in range(0, w - 15, 4):
            bot_rows = dark[8:max_y + 8, x0 + 3:x0 + 13].all(axis=1)
            for y_core in np.flatnonzero(bot_rows):
                y_core = int(y_core)
                y_bot = y_core + 8
                if not (dark[y_core:y_core + 7, x0 + 1].all() and dark[y_core:y_core + 7, x0 + 14].all()):
                    continue
                if not (dark[y_bot - 1, x0 + 2] and dark[y_bot - 1, x0 + 13]
                        and light[y_bot - 1, x0 + 3:x0 + 13].all()):
                    continue
                if float(light[y_core:y_core + 7, x0 + 2:x0 + 7].mean()) < 0.85:
                    continue
                if float(light[y_core:y_core + 7, x0 + 9:x0 + 14].mean()) < 0.85:
                    continue
                if not (int(dark[y_core:y_core + 7, x0 + 6:x0 + 10].sum()) >= 6
                        and int(light[y_core:y_core + 7, x0 + 6:x0 + 10].sum()) >= 6):
                    continue
                if y_core >= 3 and not dark[y_core - 3, x0 + 3:x0 + 13].all():
                    continue
                for cx in range(LEFT_COL, LEFT_COL + 10):
                    for cy in range(TOP_ROW - 8, TOP_ROW + 1):
                        bx0, by0, _, _ = cell_box(cx, cy)
                        if abs(bx0 - x0) <= 4 and abs(by0 - (y_bot + 8)) <= 4:
                            return (cx, cy)
        return None

    def _adjacent_sprite_person(self, facing: Optional[str],
                                ev: Optional[StepEvent] = None) -> Tuple[Optional[str], str, str]:
        frame = self._free_frame if self._free_frame is not None else (ev.frame if ev is not None else None)
        if frame is None or self._gmap is None:
            return None, "none", "low"
        g = self._gmap
        y_off = self._p["y_offset"]
        level = S.black_level(frame)
        allowed_tags = {"npc", "ground", "ground decoration", "tall grass", "unknown"}
        candidates: List[Tuple[int, int, Cell, np.ndarray, np.ndarray]] = []
        for d, cell in DIRECTIONS.items():
            tags = self._cell_tags(cell, ev)
            if tags and not (tags <= allowed_tags):
                continue
            if not S.sprite_like(frame, cell, y_off):
                continue
            window = S.sprite_window(frame, cell, (0, 0), y_off)
            if window is None:
                continue
            mask = S.black_mask(window, level)
            black = int(mask.sum())
            if black < 50 or not S.complete_sprite(mask):
                continue
            has_npc_tag = 1 if "npc" in tags else 0
            is_facing = 1 if (facing is not None and DIRECTIONS.get(facing) == cell) else 0
            candidates.append((has_npc_tag, is_facing, cell, window, mask))
        if len(candidates) != 1:
            return None, "none", "low"
        _, _, cell, window, mask = candidates[0]
        rem = self._remembered_at(cell)
        if rem is not None:
            return rem, "remembered", "low"
        hid = self._hidden_at(cell)
        if hid is not None:
            return hid, "hidden", "low"
        square = (g.pos[0] + cell[0], g.pos[1] + cell[1])
        seg, resolved = g.resolve(g.segment, square)
        for npc in self.live_npcs():
            seen = self.last_seen(g, npc)
            if seen is not None and seen[0] == seg and _manhattan(seen[1], resolved) <= 1:
                return npc.id, "remembered", "low"
        det = dict(cell=cell, offset=(0, 0), window=window, mask=mask, full=True, square=square)
        track = self._start_track(det, set(), g.segment, g.step)
        if track is not None:
            return track.npc, "seen", "low"
        return None, "none", "low"

    def _open_conversation(self, ev: StepEvent, a_press: bool) -> None:
        g = self._gmap
        speaker: Optional[str] = None
        extra_speakers: List[str] = []
        sign_square: Optional[Cell] = None
        facing = self._a_facing or self._facing
        on_screen = self.on_screen(g)
        neighbour_cells = [(c, n) for c, n in on_screen.items() if abs(c[0]) + abs(c[1]) == 1]
        neighbours = [n for _, n in neighbour_cells]
        how, confidence = "seen", "high"
        if a_press and facing is not None:
            target = DIRECTIONS[facing]
            speaker = on_screen.get(target)
            if speaker is None:
                speaker = self._remembered_at(target)
                if speaker is not None:
                    how = "remembered"
            if speaker is None:
                speaker = self._hidden_at(target)
                if speaker is not None:
                    how, confidence = "hidden", "low"
            if speaker is None:
                speaker, how, confidence = self._across_counter_speaker(target, on_screen, ev)
            if speaker is None:
                sign_square = self._sign_at(target, ev)
        elif not a_press:
            emote_cell = self._emote_cell(self._free_frame if self._free_frame is not None else ev.frame)
            if emote_cell is not None and emote_cell != (0, 0):
                speaker = on_screen.get(emote_cell) or self._remembered_at(emote_cell)
                if speaker is not None:
                    how = "seen" if emote_cell in on_screen else "remembered"
            if speaker is None and len(set(neighbours)) == 1:
                speaker = neighbours[0]
        if speaker is None and sign_square is None and not a_press:
            if len(set(neighbours)) > 1:
                seg = g.resolve(g.segment, g.pos)[0]
                distinct = list(dict.fromkeys(neighbours))
                untalked = [nid for nid in distinct if not self._talks_in(g, self.get(nid), seg)]
                ordered = untalked + [nid for nid in distinct if nid not in untalked]
                speaker, extra_speakers = ordered[0], ordered[1:]
                how, confidence = "seen", "low"
            else:
                # Only check adjacent distance-1 cells (never arbitrary NPCs up to 5 tiles away).
                adj_rem = []
                if facing is not None and self._remembered_at(DIRECTIONS[facing]) is not None:
                    adj_rem.append(self._remembered_at(DIRECTIONS[facing]))
                for d_cell in DIRECTIONS.values():
                    r_id = self._remembered_at(d_cell)
                    if r_id is not None and r_id not in adj_rem:
                        adj_rem.append(r_id)
                if len(adj_rem) == 1:
                    speaker, how, confidence = adj_rem[0], "remembered", "low"
                elif len(adj_rem) > 1:
                    seg = g.resolve(g.segment, g.pos)[0]
                    untalked = [nid for nid in adj_rem if not self._talks_in(g, self.get(nid), seg)]
                    ordered = untalked + [nid for nid in adj_rem if nid not in untalked]
                    speaker, extra_speakers = ordered[0], ordered[1:]
                    how, confidence = "remembered", "low"
                else:
                    speaker, how, confidence = self._adjacent_sprite_person(facing, ev)
        if speaker is not None:
            # Memory of someone seen on several squares is weak, and gone once they roamed.
            squares = self.squares_in(g, self.get(speaker), g.resolve(g.segment, g.pos)[0])
            if _spread(squares) > 1 and how != "seen":
                speaker = None
            elif len(squares) > 1:
                confidence = "low"
        if speaker is None and sign_square is None:
            how = "none"
        conv = Conversation(
            id=self._next("conv"), npc=speaker, sign=None, episode=self._episode, map_step_start=g.step,
            map_step_end=None, segment=g.segment, square=g.pos, target=sign_square,
            trigger="a_press" if a_press else "npc_started", how=how, confidence=confidence)
        if speaker is not None:
            npc = self.get(speaker)
            conv.npc = npc.id
            if conv.id not in npc.conversations:
                npc.conversations.append(conv.id)
            seen = self.last_seen(g, npc)
            conv.target = seen[1] if seen else None
            for extra_id in extra_speakers:
                extra_npc = self.get(extra_id)
                if conv.id not in extra_npc.conversations:
                    extra_npc.conversations.append(conv.id)
        elif sign_square is not None:
            sign = self._sign_for(g.segment, sign_square)
            conv.sign = sign.id
            if conv.id not in sign.conversations:
                sign.conversations.append(conv.id)
            sign.episode = self._episode
        else:
            self.unattributed.append(conv.id)
        if self._free_frame is not None:
            conv.context = np.array(self._free_frame, copy=True)
        self.conversations[conv.id] = conv
        self._open = conv.id
        self.stats["conversations"] += 1

    def _people_nearby(self) -> List[Tuple[int, Cell, NPC]]:
        # Returns (distance, offset from the player, npc) for every known person last seen in the player's area.
        g = self._gmap
        segment, here = g.resolve(g.segment, g.pos)
        out = []
        for npc in self.live_npcs():
            if self.look_of(npc).kind_hint == "object":
                continue
            seen = self.last_seen(g, npc)
            if seen is not None and seen[0] == segment:
                offset = (seen[1][0] - here[0], seen[1][1] - here[1])
                out.append((abs(offset[0]) + abs(offset[1]), offset, npc))
        return out

    def _remembered_at(self, offset: Cell) -> Optional[str]:
        for _, where, npc in self._people_nearby():
            if where == offset:
                return npc.id
        return None

    def _nearest_person(self) -> Optional[str]:
        # Only an adjacent known person at distance 1; with two or more in reach, nobody.
        people = [p for p in self._people_nearby() if p[0] == 1]
        return people[0][2].id if len(people) == 1 else None

    def _sign_for(self, segment: int, square: Cell) -> Sign:
        g = self._gmap
        here = g.resolve(segment, square)
        for sign in self.signs.values():
            if g.resolve(sign.segment, sign.square) == here:
                return sign
        sign = Sign(id=self._next("sign"), segment=segment, square=square, episode=self._episode)
        self.signs[sign.id] = sign
        return sign

    def _collect_crops(self, ev: StepEvent, crops: Optional[List[np.ndarray]] = None) -> None:
        from execution.pokemon.executors import drop_prefix_crops

        conv = self.conversations[self._open]
        new = self._step_crops(ev) if crops is None else crops
        if new:
            drop_prefix_crops(new, conv.crops)

    def _detach_conversation(self, conv: Conversation) -> None:
        for npc in self.npcs.values():
            while conv.id in npc.conversations:
                npc.conversations.remove(conv.id)
        for sign in self.signs.values():
            while conv.id in sign.conversations:
                sign.conversations.remove(conv.id)
        conv.npc, conv.sign, conv.target, conv.how = None, None, None, "none"
        if conv.id in self.unattributed:
            self.unattributed.remove(conv.id)

    def _mark_battle_start(self, conv: Conversation) -> None:
        was_battle_start = conv.trigger == "battle_start"
        self._detach_conversation(conv)
        conv.trigger, conv.read, conv.summarised = "battle_start", True, True
        conv.text, conv.summary, conv.speaker_name, conv.name_type = "", "", None, None
        if not was_battle_start:
            self.stats["battle_starts"] += 1
            self.stats["conversations"] -= 1

    def _close_conversation(self, led_to: str, complete: bool) -> None:
        conv = self.conversations[self._open]
        conv.led_to, conv.complete, conv.map_step_end = led_to, complete, self._gmap.step
        self.interrupted = conv.id if led_to == "episode_end" else None
        if led_to == "battle" and conv.npc in self.npcs and self.get(conv.npc).kind in ("unknown", "person"):
            self.get(conv.npc).kind = "trainer"
        if led_to == "battle" and conv.npc is None and conv.sign is None:
            # Nobody known spoke and a battle followed at once: a battle start, never read, summarised or counted.
            self._mark_battle_start(conv)
        self._last_closed = conv.id
        self._open = None

    def attach_model(self, vlm_caller: Optional[Callable[[str], Callable[..., Any]]]) -> None:
        self._live = vlm_caller

    def _attribute_by_text(self, conv: Conversation) -> None:
        if self._gmap is None or conv.sign is not None:
            return
        gmap = self._gmap
        conv_seg = gmap.resolve(conv.segment, conv.square)[0]
        candidates = [npc for npc, _ in self.people_here(gmap)
                      if self.last_seen(gmap, npc) and self.last_seen(gmap, npc)[0] == conv_seg]
        if not candidates:
            return
        names_in_text: Set[str] = set()
        if conv.speaker_name:
            names_in_text.add(conv.speaker_name.strip().casefold())
        for line in (conv.text or "").splitlines():
            m = npc_model.TAG.match(line)
            if m:
                names_in_text.add(m.group(1).strip().casefold())
        matched: List[NPC] = []
        for tag_name in names_in_text:
            if not tag_name or len(tag_name) < 2:
                continue
            for npc in candidates:
                real = (npc.real_name or "").strip().casefold()
                desc = (npc.description or "").casefold()
                if (real and (real == tag_name or tag_name in real.split())) or (
                        tag_name in desc.split() or f" {tag_name}" in desc or f"({tag_name})" in desc):
                    if npc not in matched:
                        matched.append(npc)
        if not matched:
            return
        if conv.npc is None or (conv.how == "nearest" and conv.confidence == "low"
                                and self.get(conv.npc) not in matched):
            old_npc = self.get(conv.npc) if conv.npc in self.npcs else None
            if old_npc is not None and conv.id in old_npc.conversations:
                old_npc.conversations.remove(conv.id)
            primary = matched[0]
            conv.npc = primary.id
            conv.how = "tag" if conv.how in ("none", "nearest") else conv.how
            if conv.id in self.unattributed:
                self.unattributed.remove(conv.id)
        for npc in matched:
            if conv.id not in npc.conversations:
                npc.conversations.append(conv.id)

    def _update_sign_text(self, conv: Conversation) -> None:
        if conv.sign and conv.sign in self.signs:
            sign = self.signs[conv.sign]
            sign.episode = max(sign.episode, conv.episode)
            text = (conv.summary or conv.text or "").strip()
            if text:
                sign.text = text

    def _summarise_now(self, conv: Conversation) -> None:
        if self._live is None or self._gmap is None:
            return
        try:
            if not conv.text and conv.crops and not conv.read:
                self.read_conversations(self._live("npc_ocr"), only=[conv])
            if conv.text.strip() and not conv.summarised:
                self.run_summaries(self._live("npc_summary"), self._gmap, only=[conv])
                self.apply_real_names()
            self._attribute_by_text(conv)
            self._update_sign_text(conv)
        except Exception as e:
            log_warn(f"NPC registry: summarising {conv.id} on the spot failed ({type(e).__name__}: {e}); "
                     f"it is retried at the end of the episode")

    def finish_conversations(self) -> None:
        if self._gmap is not None and self._a_step is not None and 0 <= self._gmap.step - self._a_step <= 2:
            self.pending_press = (self._a_step, self._a_facing or self._facing, self._gmap.segment, self._gmap.pos)
        else:
            self.pending_press = None
        self.last_facing = self._facing
        if self._open is not None and self._gmap is not None:
            self._close_conversation("episode_end", False)

    def read_conversations(self, vlm_call: Any, only: Optional[List[Conversation]] = None) -> None:
        from execution.pokemon.executors import drop_prefix_crops
        from utils.vlm import cache_lookup_ocr, cache_store_ocr

        todo = [c for c in (only if only is not None else self.conversations.values())
                if c.crops and not c.text and not c.read]
        if not todo:
            return
        crops_by_conv: List[List[np.ndarray]] = []
        cached_by_conv: List[List[Optional[str]]] = []
        missing_crops: List[np.ndarray] = []
        missing_refs: List[Tuple[int, int]] = []
        for ci, c in enumerate(todo):
            deduped = drop_prefix_crops(c.crops)[:MAX_OCR_CROPS]
            crops_by_conv.append(deduped)
            row: List[Optional[str]] = []
            for j, crop in enumerate(deduped):
                cached = cache_lookup_ocr(crop)
                row.append(cached)
                if cached is None:
                    missing_crops.append(crop)
                    missing_refs.append((ci, j))
            cached_by_conv.append(row)
        if missing_crops:
            images = [[crop] for crop in missing_crops]
            out = vlm_call(texts=[OCR_PROMPT] * len(images), images=images, max_new_tokens=4000)
            out = out if isinstance(out, list) else [out]
            for (ci, j), crop, raw in zip(missing_refs, missing_crops, out):
                cached_by_conv[ci][j] = cache_store_ocr(crop, raw)
        for c, texts in zip(todo, cached_by_conv):
            pieces = [t for t in texts if t and t.lower() != "none"]
            if not pieces and only is not None:
                continue
            merged_text = "\n".join(ordered_merge(pieces))
            if npc_model.is_battle_start_text(merged_text):
                self._mark_battle_start(c)
                continue
            stripped = npc_model.strip_battle_start_lines(merged_text)
            if stripped != merged_text:
                merged_text = stripped
                if c.led_to == "free_roam":
                    c.led_to = "battle"
                if c.npc in self.npcs and self.get(c.npc).kind in ("unknown", "person"):
                    self.get(c.npc).kind = "trainer"
            if not merged_text.strip() and c.led_to == "battle":
                self._mark_battle_start(c)
                continue
            c.text, c.read = merged_text, True
            self._update_sign_text(c)
            self._attribute_by_text(c)
            self.stats["conversations_read"] += 1

    def merge_npcs(self, keep: NPC, drop: NPC) -> None:
        keep.sightings += drop.sightings
        keep.conversations += drop.conversations
        if drop.name not in keep.aliases:
            keep.aliases.append(drop.name)
        keep.aliases += [a for a in drop.aliases if a not in keep.aliases]
        if drop.real_name and not keep.real_name:
            keep.real_name, keep.name_type = drop.real_name, drop.name_type
        for conversation in drop.conversations:
            if conversation in self.conversations:
                self.conversations[conversation].npc = keep.id
        self.covisible = {frozenset(keep.id if x == drop.id else x for x in pair) for pair in self.covisible}
        self.covisible = {pair for pair in self.covisible if len(pair) == 2}
        drop.sightings, drop.conversations, drop.merged_into, drop.needs = [], [], keep.id, None
        self.stats["npcs_merged"] += 1

    def merge_looks(self, keep_id: str, drop_id: str) -> None:
        keep, drop = self.looks[keep_id], self.looks[drop_id]
        shift = len(keep.views)
        keep.views += drop.views
        keep.links += [(a + shift, b + shift, why) for a, b, why in drop.links]
        keep.links.append((0, shift, "track"))
        for npc in self.npcs.values():
            if self.look_aliases.get(npc.look, npc.look) == drop_id or npc.look == drop_id:
                npc.look = keep_id
                for s in npc.sightings:
                    s.view += shift
        self.look_aliases[drop_id] = keep_id
        del self.looks[drop_id]
        self._stack = None
        self.stats["looks_merged"] += 1

    def consolidate(self) -> None:
        gmap = self._gmap
        for a, b in sorted(self._look_merges):
            a, b = self.look_aliases.get(a, a), self.look_aliases.get(b, b)
            if a != b and a in self.looks and b in self.looks:
                self.merge_looks(a, b)
        self._look_merges = set()
        people = sorted(self.live_npcs(), key=lambda n: n.created)
        for i, older in enumerate(people):
            if older.merged_into is not None:
                continue
            for newer in people[i + 1:]:
                if newer.merged_into is not None or self.look_of(newer).id != self.look_of(older).id:
                    continue
                if frozenset((older.id, newer.id)) in self.covisible:
                    continue
                a, b = self.last_seen(gmap, older), self.last_seen(gmap, newer)
                if a and b and a[0] == b[0] and _manhattan(a[1], b[1]) <= NEAR:
                    self.merge_npcs(older, newer)

    @update
    def end_episode(self, gmap: GlobalMap, vlm_caller: Optional[Callable[[str], Callable[..., Any]]] = None) -> None:
        self._gmap = gmap
        self.finish_conversations()
        if vlm_caller is not None:
            try:
                self.read_conversations(vlm_caller("npc_ocr"))
            except Exception as e:
                log_warn(f"NPC registry: reading conversations failed and was skipped: {type(e).__name__}: {e}")
        self.consolidate()
        if vlm_caller is not None:
            self._model_steps(vlm_caller, gmap)
        self._tracks = {}
        self._forget_screen()

    def _model_steps(self, vlm_caller: Callable[[str], Callable[..., Any]], gmap: GlobalMap) -> None:
        for label, step in (("cards and judges", lambda: self.run_model(vlm_caller("npc_card"),
                                                                       vlm_caller("npc_judge"), gmap)),
                            ("summaries", lambda: self.run_summaries(vlm_caller("npc_summary"), gmap)),
                            ("real names", self.apply_real_names),
                            ("text attribution", self.attribute_conversations)):
            try:
                step()
            except Exception as e:
                log_warn(f"NPC registry: {label} skipped: {type(e).__name__}: {e}")

    def attribute_conversations(self) -> None:
        for conv in self.conversations.values():
            self._update_sign_text(conv)
            self._attribute_by_text(conv)

    def run_model(self, vlm_card: Callable[..., Any], vlm_judge: Callable[..., Any], gmap: GlobalMap) -> None:
        npc_model.run_model(self, vlm_card, vlm_judge, self.game, gmap)

    def run_summaries(self, vlm_call: Callable[..., Any], gmap: GlobalMap,
                      only: Optional[List[Conversation]] = None) -> None:
        npc_model.run_summaries(self, vlm_call, self.game, gmap, only=only)
        for conv in (only if only is not None else self.conversations.values()):
            self._update_sign_text(conv)
            self._attribute_by_text(conv)

    @staticmethod
    def _infer_view_facing(view: View) -> Optional[str]:
        m = np.asarray(view.key, dtype=bool)
        px = np.asarray(view.pixels)
        if m.shape != (16, 16) or px.shape != (16, 16) or not m.any():
            return None
        m12 = m[4:, :]
        sym12 = int((m12 ^ m12[:, ::-1]).sum())
        mh = m[3:9, 2:14]
        sym_h = int((mh ^ mh[:, ::-1]).sum())
        if sym12 <= 16 or sym_h <= 14:
            light_face = int((px[5:10, 4:12] >= 180).sum())
            cheeks = int((px[6:9, 5:7] >= 180).sum() + (px[6:9, 9:11] >= 180).sum())
            if light_face >= 14 and cheeks >= 6:
                return "down"
            return "up"
        l_light = int((px[5:10, 3:8] >= 180).sum())
        r_light = int((px[5:10, 8:13] >= 180).sum())
        if l_light > r_light:
            return "left"
        if r_light > l_light:
            return "right"
        return None

    @staticmethod
    def sighting_facing(look: Look, sighting: Sighting) -> Optional[str]:
        if sighting.view >= len(look.views):
            return None
        view = look.views[sighting.view]
        facing = view.facing
        if facing is None and not view.partial:
            for other in look.views:
                if other.facing is not None and not other.partial:
                    dist, fl = S.mask_distance(view.key, other.key)
                    if dist <= 12:
                        if fl and other.facing in ("left", "right"):
                            facing = "left" if other.facing == "right" else "right"
                        else:
                            facing = other.facing
                        break
        if facing is None:
            facing = NPCRegistry._infer_view_facing(view)
        if sighting.flipped and facing in ("left", "right"):
            return "left" if facing == "right" else "right"
        return facing

    def visits(self, npc: NPC) -> List[Tuple[int, Conversation, Optional[int]]]:
        # Returns (visit number, conversation, earlier visit with the same words or None), oldest first.
        talks = sorted((self.conversations[c] for c in npc.conversations if c in self.conversations),
                       key=lambda c: (c.episode, c.map_step_start))
        seen: Dict[str, int] = {}
        out = []
        for k, c in enumerate(talks, 1):
            words = npc_model._same_text(c.text)
            out.append((k, c, seen.get(words) if words else None))
            if words:
                seen.setdefault(words, k)
        return out

    def apply_real_names(self) -> None:
        npc_model.apply_real_names(self)

    def people_here(self, gmap: GlobalMap) -> List[Tuple[NPC, Cell]]:
        found = []
        here = gmap.resolve(gmap.segment, gmap.pos)[0]
        for npc in self.live_npcs():
            seen = self.last_seen(gmap, npc)
            if seen is not None and seen[0] == here:
                found.append((npc, seen[1]))
        return found

    def signs_here(self, gmap: GlobalMap) -> List[Tuple[Sign, Cell]]:
        found = []
        here = gmap.resolve(gmap.segment, gmap.pos)[0]
        for sign in self.signs.values():
            seg, square = gmap.resolve(sign.segment, sign.square)
            if seg == here and sign.conversations:
                found.append((sign, square))
        return found

    def _sign_heard(self, sign: Sign) -> Tuple[int, str]:
        convs = [self.conversations[c] for c in sign.conversations if c in self.conversations]
        ep = max((c.episode for c in convs), default=sign.episode)
        text = (sign.text or "").strip()
        if not text and convs:
            last = convs[-1]
            text = (last.summary or last.text or "").strip()
        return ep, text.replace("\n", " ")

    def when_seen(self, gmap: GlobalMap, npc: NPC, square: Cell) -> str:
        seen = self.last_seen(gmap, npc)
        if npc.id in self.on_screen(gmap).values():
            return "on screen now"
        steps = gmap.step - seen[2].map_step
        if seen[2].episode == self._episode:
            when = f"last seen {steps} step(s) ago"
        else:
            when = f"last seen {self._episode - seen[2].episode} episode(s) ago"
        squares = self.squares_in(gmap, npc, seen[0])
        if len(squares) > 1:
            when += f", seen on {len(squares)} different squares (walks about)"
        if self.in_view(gmap, square):
            when += ("; not recognised on that square, which is on screen now: may have moved" if self.seen_whole(npc)
                     else "; only ever seen in part (hidden, for example in tall grass)")
        return when

    def people_block(self, gmap: GlobalMap, max_people: int = 8, max_chars: int = 1800, where: str = "here") -> str:
        # where: "here" when the current screen was placed, else a key of npc_map.WHERE_NOTES.
        here = [(npc, square) for npc, square in self.people_here(gmap)
                if self._episode - self.last_seen(gmap, npc)[2].episode <= STALE_EPISODES
                and self.look_of(npc).kind_hint != "object"]
        maybe = [] if where == "away" else self.unconfirmed(gmap, gmap.resolve(gmap.segment, gmap.pos)[0])
        signs = [] if where == "away" else self.signs_here(gmap)
        if not here and not maybe and not signs:
            return ""
        px, py = gmap.pos
        here.sort(key=lambda r: abs(r[1][0] - px) + abs(r[1][1] - py))
        lines = []
        for npc, square in here[:max_people]:
            dx, dy = square[0] - px, square[1] - py
            when = self.when_seen(gmap, npc, square)
            talks = self._talks_in(gmap, npc, gmap.resolve(gmap.segment, gmap.pos)[0])
            if talks:
                last = talks[-1]
                heard = last.summary or last.text[:160].strip().replace("\n", " ")
                said = f": \"{heard}\"" if heard else ""
                status = f"talked to here (episode {last.episode}){said}"
            else:
                status = "not talked to yet"
            seen_info = self.last_seen(gmap, npc)
            facing = self.sighting_facing(self.look_of(npc), seen_info[2]) if seen_info is not None else None
            facing_note = f", facing {facing}" if facing else ""
            what = ", ".join(x for x in (npc.kind if npc.kind != "unknown" else "", npc.description[:70]) if x)
            at = "" if where == "away" else f" at ({dx}, {dy})"
            lines.append(f'- "{display_name(npc)}"{f" ({what})" if what else ""}{at}{facing_note}, {when}: {status}.')
        if len(here) > max_people:
            lines.append(f"...and {len(here) - max_people} more.")
        lines += maybe_lines(gmap, maybe)
        if signs:
            signs.sort(key=lambda r: abs(r[1][0] - px) + abs(r[1][1] - py))
            lines.append("Signs and objects already inspected in this area:")
            for sign, square in signs[:max_people]:
                dx, dy = square[0] - px, square[1] - py
                ep, heard = self._sign_heard(sign)
                said = f': "{heard[:120]}"' if heard else ""
                lines.append(f"- at ({dx}, {dy}): already inspected (episode {ep}){said}.")
        note = WHERE_NOTES[where] + "\n" if where in WHERE_NOTES else ""
        block = ("PEOPLE YOU HAVE SEEN IN THIS AREA (positions are steps from you: x right, y up). The quoted names are "
                 "labels for this list (unless one is a real name the person gave): the player does not know the labels, "
                 "so describe a person by where they stand, not by name, and never write a label into a task. Talking "
                 "to someone you have not talked to yet can give information or items.\n" + note
                 + "\n".join(lines))
        return "\n\n" + block[:max_chars]

    def talked_in(self, gmap: GlobalMap, npc: NPC) -> Set[int]:
        found = set()
        for conversation_id in npc.conversations:
            c = self.conversations.get(conversation_id)
            if (c is not None and c.trigger != "battle_start"
                    and (c.text or (c.crops and not c.read) or c.trigger == "a_press")):
                found.add(gmap.resolve(c.segment, c.square)[0])
        return found

    def _talks_in(self, gmap: GlobalMap, npc: NPC, segment: int) -> List[Conversation]:
        target_seg = gmap.resolve(segment, (0, 0))[0]
        return [self.conversations[c] for c in npc.conversations
                if c in self.conversations
                and self.conversations[c].trigger != "battle_start"
                and (self.conversations[c].text or (self.conversations[c].crops and not self.conversations[c].read)
                     or self.conversations[c].trigger == "a_press")
                and gmap.resolve(self.conversations[c].segment, self.conversations[c].square)[0] == target_seg]

    def screen_labels(self, gmap: GlobalMap) -> Dict[Cell, str]:
        # Keyed by square relative to the player.
        out: Dict[Cell, str] = {}
        people = dict(self.on_screen(gmap))
        remembered = set()
        segment = gmap.resolve(gmap.segment, gmap.pos)[0]
        for mark in self.map_people(gmap, segment):
            cell = (mark.square[0] - gmap.pos[0], mark.square[1] - gmap.pos[1])
            if (not mark.on_screen and self.in_view(gmap, mark.square) and not mark.earlier and cell not in people
                    and self._episode - mark.last_seen[0] <= STALE_MARK_EPISODES):
                people[cell] = mark.npc_id
                remembered.add(cell)
        for cell, npc_id in people.items():
            npc = self.get(npc_id)
            if self.look_of(npc).kind_hint == "object":
                continue
            talks = self._talks_in(gmap, npc, segment)
            if talks:
                heard = (talks[-1].summary or talks[-1].text)[:50].strip().replace("\n", " ")
                status = f'talked to here in episode {talks[-1].episode}' + (f': "{heard}"' if heard else "")
            else:
                status = "not talked to here"
                elsewhere = [seg for seg in self.talked_in(gmap, npc) if seg != segment]
                if elsewhere:
                    status += "; met before in another area"
            if cell in remembered:
                status = ("remembered here, not recognised on this frame: may have moved; " if self.seen_whole(npc)
                          else "seen only in part (hidden, for example in tall grass); ") + status
            seen_info = self.last_seen(gmap, npc)
            facing = self.sighting_facing(self.look_of(npc), seen_info[2]) if seen_info is not None else None
            facing_note = f", facing {facing}" if facing else ""
            kind = npc.kind if npc.kind != "unknown" else "person"
            name = (f'who gave the name "{npc.real_name}"' if npc.real_name
                    else f'labelled "{npc.name}" (the label is {LABEL_NOTE})')
            out[cell] = f'{kind} {name}{facing_note} ({status})'[:320]
        for sign, square in self.signs_here(gmap):
            cell = (square[0] - gmap.pos[0], square[1] - gmap.pos[1])
            if cell in out:
                continue
            if LEFT_COL <= cell[0] <= LEFT_COL + 9 and TOP_ROW - 8 <= cell[1] <= TOP_ROW:
                ep, heard = self._sign_heard(sign)
                said = f': "{heard[:60]}"' if heard else ""
                out[cell] = f"sign:already inspected in episode {ep}{said}"[:320]
        return out

    def map_people(self, gmap: GlobalMap, segment: int, upto_step: Optional[int] = None) -> List[PersonMark]:
        upto = float("inf") if upto_step is None else upto_step
        live = [n for n in self.live_npcs() if self.look_of(n).kind_hint != "object"]
        now = self._episode if upto_step is None else max(
            (s.episode for n in live for s in n.sightings if s.map_step <= upto), default=0)
        on_screen = set(self.on_screen(gmap).values()) if upto_step is None else set()
        marks = []
        for npc in sorted(live, key=lambda n: n.id):
            seen = sorted((s for s in npc.sightings if s.map_step <= upto), key=lambda s: -s.map_step)
            if not seen or self.place(gmap, seen[0])[0] != segment:
                continue
            squares: List[Cell] = []
            for sighting in seen:
                where = self.place(gmap, sighting)
                if where[0] == segment and where[1] not in squares:
                    squares.append(where[1])
            square, newest = squares[0], seen[0]
            views = self.look_of(npc).views
            talks = [c for c in self._talks_in(gmap, npc, segment) if c.map_step_start <= upto]
            sure = [c for c in talks if c.how in ("seen", "remembered") and getattr(c, "confidence", "high") == "high"]
            words = {npc_model._same_text(c.text) for c in talks}
            stale = now - newest.episode > STALE_MARK_EPISODES
            if upto_step is None:
                when = self.when_seen(gmap, npc, square)
                stale = stale or (npc.id not in on_screen and self.in_view(gmap, square) and self.seen_whole(npc))
            else:
                when = f"last seen in episode {newest.episode}"
            xs, ys = [q[0] for q in squares], [q[1] for q in squares]
            marks.append(PersonMark(
                npc_id=npc.id, number=int(npc.id.split("_")[1]), name=display_name(npc), kind=npc.kind,
                square=square, facing=self.sighting_facing(self.look_of(npc), newest),
                last_seen=(newest.episode, newest.map_step), on_screen=npc.id in on_screen,
                talked="yes" if sure else "probably" if talks else "no", times_talked=len(talks),
                same_words=len(talks) > 1 and len(words) == 1 and "" not in words, earlier=squares[1:4],
                spread=(min(xs), min(ys), max(xs), max(ys)) if len(squares) >= 3 else None, stale=stale, when=when))
        return marks

    def unconfirmed(self, gmap: GlobalMap, segment: int) -> List[Cell]:
        background = {square for seg, square in self.background_squares(gmap) if seg == segment}
        known = []
        for npc in self.live_npcs():
            for sighting in npc.sightings:
                seg, square = self.place(gmap, sighting)
                if seg == segment:
                    known.append(square)
        out = []
        for square, cell in gmap.segments[segment].items():
            if (cell.kind == OCCUPIED and "npc" in cell.tags and square not in background
                    and all(_manhattan(square, k) > 1 for k in known)):
                out.append(square)
        return sorted(out)

    def has_noteworthy_conversation(self, episode: int) -> bool:
        for c in self.conversations.values():
            if c.episode != episode or c.trigger == "battle_start" or not c.text.strip():
                continue
            if not npc_model.is_routine_conversation(self, c):
                return True
        return False

    def digest_lines(self, episode: int, gmap: GlobalMap) -> str:
        lines = []
        for c in self.conversations.values():
            if c.episode != episode or c.trigger == "battle_start" or not c.text.strip():
                continue
            if npc_model.is_battle_start_text(c.text):
                continue
            who = (f'"{display_name(self.get(c.npc))}"' if c.npc in self.npcs else "a sign" if c.sign else "someone")
            lines.append(f'- {who} in {npc_model._area(self, gmap, gmap.resolve(c.segment, c.square)[0])}: '
                         f'"{c.text[:400].strip()}"')
        if not lines:
            return ""
        return ("\n\nConversations this episode (the quoted names are labels for a list, not names the game knows: "
                "to say who spoke, describe the person and where they stood):\n" + "\n".join(lines))

    def markers(self, gmap: GlobalMap, segment: int) -> List[Tuple[Cell, str, bool]]:
        # Returns (square, number, talked_to_here).
        return [(m.square, str(m.number), m.talked != "no") for m in self.map_people(gmap, segment)]

    def diff(self, old: "NPCRegistry") -> Dict[str, Any]:
        new_npcs = [i for i in self.npcs if i not in old.npcs]
        return {
            "new_npcs": new_npcs,
            "new_looks": [i for i in self.looks if i not in old.looks],
            "new_views": {i: len(l.views) - len(old.looks[i].views) for i, l in self.looks.items()
                          if i in old.looks and len(l.views) != len(old.looks[i].views)},
            "new_conversations": [i for i in self.conversations if i not in old.conversations],
            "merged": {i: n.merged_into for i, n in self.npcs.items()
                       if n.merged_into and (i not in old.npcs or old.npcs[i].merged_into is None)},
            "renamed": [(i, old.npcs[i].name, n.name) for i, n in self.npcs.items()
                        if i in old.npcs and old.npcs[i].name != n.name],
            "kind_changed": [(i, old.npcs[i].kind, n.kind) for i, n in self.npcs.items()
                             if i in old.npcs and old.npcs[i].kind != n.kind],
            "new_signs": [i for i in self.signs if i not in old.signs],
        }


def ordered_merge(pieces: List[str]) -> List[str]:
    lines: List[str] = []
    for piece in pieces:
        new = [ln.replace("▼", "").replace("▶", "").strip() for ln in piece.splitlines()]
        new = [ln for ln in new if ln]
        for k in range(min(len(lines), len(new)), 0, -1):
            if lines[-k:] == new[:k]:
                new = new[k:]
                break
        if new and not any(lines[i:i + len(new)] == new for i in range(len(lines) - len(new) + 1)):
            lines.extend(new)
    return lines


def _spread(squares: List[Cell]) -> int:
    return max((_manhattan(a, b) for a in squares for b in squares), default=0)


def _pairs(items: List[str]) -> List[Tuple[str, str]]:
    return [(a, b) for i, a in enumerate(items) for b in items[i + 1:] if a != b]
