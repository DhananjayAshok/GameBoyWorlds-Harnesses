from collections import deque
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from execution.game_api import AgentState, WaitAction

from execution.executors import Executor
from execution.perception.pokemon.battle import (has_bottom_dialogue_box, has_bottom_location_banner,
                                                 has_emote_bubble,
                                                 has_menu_cursor, has_modal_preview_box,
                                                 has_top_dialogue_box, is_battle_screen,
                                                 is_blank_or_fade_frame,
                                                 is_cutscene_frame_changing, is_overworld_pack_menu,
                                                 is_pack_screen, is_party_menu_screen,
                                                 read_battle_hud_hp)
from execution.perception.pokemon.navigation import (DIRECTIONS, LEDGE, LEDGE_LEFT, WALKABLE, X_RANGE, Y_RANGE,
                                                     boundary_exit_direction, cell_kinds, compress_path, edge_cells,
                                                     edge_runs, filter_open_lane_edge_cells, format_path_runs,
                                                     in_bounds, is_boundary_exit_direction, is_on_edge,
                                                     parse_all_cells, parse_edge_direction, shortest_path,
                                                     verbalize_edges, warp_cells)
from execution.perception.pokemon.tiles import TileRecognizer, describe_state_screen, verbalize_tiles
from execution.pokemon.executors import (CombatExecutor, DialogueExecutor, FreeExecutor,
                                         InteractionExecutor, MenuExecutor,
                                         BATTLE_ENDED, BATTLE_LEFT_TO_DIALOGUE, BATTLE_LEFT_TO_MENU,
                                         DIALOGUE_CLEARED, DIALOGUE_LEFT_TO_BATTLE, DIALOGUE_LEFT_TO_MENU,
                                         FREE_OK, LOCATION_FAIL_CODES,
                                         MENU_LEFT_TO_BATTLE, MENU_LEFT_TO_DIALOGUE, MENU_OK,
                                         MOVE_OK, MOVE_TARGET_AMBIGUOUS,
                                         OffScreenMoveExecutor, OnScreenMoveExecutor,
                                         effective_agent_state, settle_overworld_free_roam,
                                         settle_overworld_menu_state)
from execution.pokemon.prompts import (PLAYTHROUGH_ACHIEVABLE_PROMPT, PLAYTHROUGH_AFTER_BLOCK,
                                       PLAYTHROUGH_AREA_MAP_BLOCK, PLAYTHROUGH_ATTEMPTS_BLOCK,
                                       PLAYTHROUGH_BATTLE_INSTRUCTIONS_PROMPT, PLAYTHROUGH_BATTLE_TIPS_BLOCK,
                                       PLAYTHROUGH_CLARIFY_TARGET_PROMPT, PLAYTHROUGH_DECISION_RETRY_BLOCK,
                                       PLAYTHROUGH_GUIDANCE_BLOCK,
                                       PLAYTHROUGH_JUDGE_PROMPT, PLAYTHROUGH_LOG_BLOCK,
                                       PLAYTHROUGH_MENU_OBJECTIVE_PROMPT, PLAYTHROUGH_PLAN_PROMPT,
                                       PLAYTHROUGH_REMAINING_SUBGOALS_BLOCK, PLAYTHROUGH_REMAINING_SUBGOALS_LINE,
                                       PLAYTHROUGH_SUBGOAL_DECISION_PROMPT, PLAYTHROUGH_TASK_CHECK_PROMPT,
                                       PLAYTHROUGH_TEXT_BLOCK, PRIMITIVES)
from execution.strategist.pokemon.global_map.model import OCCUPIED, UNKNOWN, is_door
from execution.strategist.pokemon.global_map.routes import blocked_target_cells
from execution.strategist.pokemon.global_map.text import LEGEND, ascii_window, door_lines, frontier_targets
from execution.report import ExecutorReport
from execution.supervisors.base import Supervisor
from utils import log_error, log_warn, parse_int, parse_key_value, parse_list, parse_yes_no

ACTIONS: Dict[str, Tuple[str, bool, Optional[str]]] = {
    "MoveTo": ("on_screen_move", True, None),
    "MoveOff": ("off_screen_move", True, None),
    "Interact": ("interaction", True, None),
    "Battle": ("combat", False, "win the current battle"),
    "Menu": ("menu", True, None),
    "Dialogue": ("dialogue", False, "clear the dialogue on screen"),
    "Free": ("free", True, None),
}
MOVE_TAGS = ("on_screen_move", "off_screen_move")
EXECUTOR_STEPS = {
    "on_screen_move": 20,
    "off_screen_move": 20,
    "interaction": 3,
    "menu": 25,
    "combat": 60,
    "dialogue": 10,
    "free": 15,
}
ACTION_PATTERN = re.compile(r"^\s*`?(" + "|".join(ACTIONS) + r")\s*\((.*)\)\s*`?\s*\.?\s*$", re.IGNORECASE | re.DOTALL)


FREE_TERMINATION_PATTERN = re.compile(r"[;,.]?\s*stop\s+when\s*:?\s*", re.IGNORECASE)

_FREE_DIR_WORD_MAP = {
    "up": "up", "north": "up",
    "down": "down", "south": "down",
    "left": "left", "west": "left",
    "right": "right", "east": "right",
}
_FREE_NUM_WORD_MAP = {
    "once": 1, "one": 1,
    "twice": 2, "two": 2,
    "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
}
_FREE_COUNT_TOKEN = r"(?:\d+|once|twice|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
_FREE_DIR_TOKEN = r"(?:up|down|left|right|north|south|west|east)"
_FREE_COUNTED_DIR_PATTERN = re.compile(
    rf"\b(?:"
    rf"(?P<dir1>{_FREE_DIR_TOKEN})\s+(?P<cnt1>{_FREE_COUNT_TOKEN})(?:\s+(?:times?|steps?|tiles?|squares?))?\b"
    rf"|(?P<cnt2>{_FREE_COUNT_TOKEN})\s+(?:times?|steps?|tiles?|squares?)\s+(?:to\s+the\s+)?(?P<dir2>{_FREE_DIR_TOKEN})\b"
    rf")",
    re.IGNORECASE,
)
_FREE_CLAUSE_DIR_PATTERN = re.compile(
    rf"\b(?:press(?:ing)?|walk(?:ing)?|mov(?:e|ing)|go(?:ing)?|step(?:ping)?|head(?:ing)?)\s+(?P<dir>{_FREE_DIR_TOKEN})\b",
    re.IGNORECASE,
)


def _parse_count_token(token: str) -> int:
    t = token.strip().lower()
    if t.isdigit():
        return int(t)
    return _FREE_NUM_WORD_MAP.get(t, 1)


def parse_free_directional_legs(instruction: str) -> List[Tuple[str, int]]:
    legs: List[Tuple[str, int]] = []
    for m in _FREE_COUNTED_DIR_PATTERN.finditer(instruction):
        d_raw = m.group("dir1") or m.group("dir2")
        c_raw = m.group("cnt1") or m.group("cnt2")
        if d_raw and c_raw:
            d = _FREE_DIR_WORD_MAP[d_raw.lower()]
            c = _parse_count_token(c_raw)
            if c > 0:
                if legs and legs[-1][0] == d:
                    legs[-1] = (d, legs[-1][1] + c)
                else:
                    legs.append((d, c))
    if legs:
        return legs
    for m in _FREE_CLAUSE_DIR_PATTERN.finditer(instruction):
        prefix = instruction[max(0, m.start() - 4):m.start()].lower()
        if "to " in prefix and legs:
            continue
        d = _FREE_DIR_WORD_MAP[m.group("dir").lower()]
        if not legs or legs[-1][0] != d:
            legs.append((d, 1))
    return legs


def split_free_argument(argument: str) -> Tuple[str, Optional[str]]:
    parts = FREE_TERMINATION_PATTERN.split(argument, maxsplit=1)
    instruction = parts[0].strip().strip("\"'")
    termination = parts[1].strip().strip("\"'") if len(parts) > 1 else None
    return instruction or argument, termination


def parse_decision(output: str) -> Optional[Tuple[str, Optional[str]]]:
    line = parse_key_value(output, "Action")
    if line is None:
        return None
    match = ACTION_PATTERN.match(line)
    if match is None:
        return None
    name = next(a for a in ACTIONS if a.lower() == match.group(1).lower())
    argument = match.group(2).strip().strip("\"'") or None
    requires_argument = ACTIONS[name][1]
    if requires_argument and argument is None:
        return None
    return name, argument


class PokemonPlayThroughSupervisor(Supervisor):
    EXECUTORS: Dict[str, Type[Executor]] = {
        "interaction": InteractionExecutor,
        "on_screen_move": OnScreenMoveExecutor,
        "off_screen_move": OffScreenMoveExecutor,
        "combat": CombatExecutor,
        "menu": MenuExecutor,
        "dialogue": DialogueExecutor,
        "free": FreeExecutor,
    }
    CLEAR_EVENT_INSTRUCTION = ("Clear the event that is going on, most likely by pressing A or B until the player "
                               "has control of their character again and can walk around.")
    CLEAR_EVENT_TERMINATION = ("the player is standing on the map with no text box, menu or battle on screen, "
                               "free to walk again")
    MAX_CLARIFICATIONS = 2
    MAX_SUBGOAL_ATTEMPTS = 4
    MAX_SUBGOAL_ITERATIONS = 12
    MAX_REPLANS = 2
    MAX_CONTINUATION_REPLANS = 3
    MAX_TOTAL_EXECUTOR_CALLS = 30
    MAX_INTERRUPTION_CHAIN = 6
    MAX_WILD_MOVE_RESUMES = 2
    WILD_FLEE_INSTRUCTION = "Run away from the wild Pokemon immediately to continue your task."

    def __init__(self, task: str, env, game: str, max_steps: int,
                 tile_recognizer: TileRecognizer, episode_number: int, guidance: Optional[str] = None,
                 battle_tips: Optional[List[str]] = None,
                 screen_people: Optional[Callable[[], Dict[Tuple[int, int], str]]] = None,
                 initial_battle_hp: Optional[Tuple[int, int]] = None,
                 initial_healing_bag_empty: bool = False,
                 initial_blocked_moves: Optional[Dict[Tuple[int, Tuple[int, int]], Dict[str, List[str]]]] = None,
                 **kwargs: Any) -> None:
        self._screen_people = screen_people
        self._current_tag: Optional[str] = None
        self._guidance = guidance
        self._battle_tips = list(battle_tips or [])
        self._last_seen_battle_hp: Optional[Tuple[int, int]] = initial_battle_hp
        self._healing_bag_empty: bool = bool(initial_healing_bag_empty)
        self._losses_at_start = env.get_info()["pokemon_core"].get("n_battles_lost", 0)
        gmap_init = getattr(env, "_map", None)
        self._defeats_at_start = len(getattr(gmap_init, "defeats", ()) or ())
        self._tile_recognizer = tile_recognizer
        self._episode_number = episode_number
        self._screen_tiles: Optional[str] = None
        self._identified = None
        self._calls = 0
        self._total_calls = 0
        self._max_total_calls = max(self.MAX_TOTAL_EXECUTOR_CALLS, max_steps * 3)
        self._start_place: Optional[Tuple[int, Tuple[int, int]]] = None
        self._leg_stops: List[Tuple[int, Tuple[int, int]]] = []
        self._blocked_moves: Dict[Tuple[int, Tuple[int, int]], Dict[str, List[str]]] = {
            k: {d: list(notes) for d, notes in by_dir.items()}
            for k, by_dir in (initial_blocked_moves or {}).items()
        }
        self._last_settled_step: Optional[int] = None
        self.last_executor = None
        self.plan: List[str] = []
        super().__init__(task=task, executor_class=None, env=env, game=game,
                         max_steps=max_steps, **kwargs)
        self._tile_recognizer.set_vlm_call(self._vlm_caller("perception"))
        self._record_map_stop()

    def _run_config(self) -> dict:
        config = super()._run_config()
        config["guidance"] = self._guidance
        config["battle_tips"] = list(self._battle_tips)
        config["tile_recognizer"] = self._tile_recognizer.name
        config["episode_number"] = self._episode_number
        config["max_executor_calls"] = self._max_steps
        config["max_total_executor_calls"] = self._max_total_calls
        config["max_subgoal_attempts"] = self.MAX_SUBGOAL_ATTEMPTS
        config["max_subgoal_iterations"] = self.MAX_SUBGOAL_ITERATIONS
        config["max_replans"] = self.MAX_REPLANS
        return config

    def _current_frame(self):
        return self._env.get_info()["core"]["current_frame"]

    def _agent_state(self):
        return self._env.get_info()["pokemon_core"]["agent_state"]

    def _effective_state(self, frame=None, raw_state=None) -> AgentState:
        raw = self._agent_state() if raw_state is None else raw_state
        if raw in (AgentState.IN_MENU, AgentState.FREE_ROAM):
            frame = self._current_frame() if frame is None else frame
            return effective_agent_state(raw, frame)
        return raw

    def _is_in_dialogue(self, frame=None) -> bool:
        return self._effective_state(frame) == AgentState.IN_DIALOGUE

    def _global_map(self):
        return getattr(self._env, "_map", None)

    def _current_map_place(self) -> Optional[Tuple[int, Tuple[int, int]]]:
        gmap = self._global_map()
        if gmap is None or not getattr(gmap, "current", None) or getattr(gmap, "pos", None) is None:
            return None
        try:
            return gmap.resolve(gmap.segment, gmap.pos)
        except Exception:  # noqa: BLE001
            return None

    def _record_map_stop(self) -> Optional[Tuple[int, Tuple[int, int]]]:
        place = self._current_map_place()
        if place is None:
            return None
        if self._start_place is None:
            self._start_place = place
        if not self._leg_stops or self._leg_stops[-1] != place:
            self._leg_stops.append(place)
        return place

    def _current_segment_blocked_moves(self) -> Dict[Tuple[int, int], set]:
        gmap = self._global_map()
        place = self._current_map_place()
        if gmap is None or place is None:
            return {}
        cur_seg, _ = place
        out: Dict[Tuple[int, int], set] = {}
        if hasattr(gmap, "segment_blocked_moves"):
            for r_pos, dirs in gmap.segment_blocked_moves(cur_seg).items():
                if dirs:
                    out.setdefault(r_pos, set()).update(dirs)
        for (b_seg, b_pos), by_dir in self._blocked_moves.items():
            try:
                r_seg, r_pos = gmap.resolve(b_seg, b_pos)
            except Exception:  # noqa: BLE001
                continue
            if r_seg == cur_seg and by_dir:
                out.setdefault(r_pos, set()).update(by_dir.keys())
        return out

    def _local_blocked_cells(self) -> set:
        local_blocked = set(getattr(self._env, "blocked_cells", None) or ())
        gmap = self._global_map()
        place = self._current_map_place()
        if gmap is None or place is None:
            return local_blocked
        cur_seg, (px, py) = place
        seg_blocked_moves = self._current_segment_blocked_moves()
        if not seg_blocked_moves:
            return local_blocked
        for target in blocked_target_cells(gmap, cur_seg, seg_blocked_moves):
            rel = (target[0] - px, target[1] - py)
            if rel != (0, 0) and in_bounds(rel):
                local_blocked.add(rel)
        return local_blocked

    def _run_pocket_bounds(
        self,
        direction: str,
        run_cells: List[Tuple[int, int]],
    ) -> Optional[Tuple[set, set]]:
        gmap = self._global_map()
        place = self._current_map_place()
        if gmap is None or place is None or not run_cells:
            return None
        seg = gmap.current
        cur_seg, (px, py) = place
        seg_blocked_moves = self._current_segment_blocked_moves()
        blocked_targets = blocked_target_cells(gmap, cur_seg, seg_blocked_moves)
        gx_min, gx_max = px + min(X_RANGE), px + max(X_RANGE)
        gy_min, gy_max = py + min(Y_RANGE), py + max(Y_RANGE)

        def in_screen(g: Tuple[int, int]) -> bool:
            return gx_min <= g[0] <= gx_max and gy_min <= g[1] <= gy_max

        def in_half_plane(g: Tuple[int, int]) -> bool:
            if direction == "up":
                return g[1] >= gy_max
            if direction == "down":
                return g[1] <= gy_min
            if direction == "left":
                return g[0] <= gx_min
            return g[0] >= gx_max

        def strictly_beyond(g: Tuple[int, int]) -> bool:
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
            if eg in blocked_targets:
                continue
            c = seg.get(eg)
            if c is None or c.kind == UNKNOWN or is_door(c):
                return None
            if c.kind == WALKABLE:
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
            if c is not None and is_door(c):
                half_plane_open = True
                break
            cur_blocked_dirs = seg_blocked_moves.get(cur, set())
            for d_name, (nx_d, ny_d) in DIRECTIONS.items():
                if d_name in cur_blocked_dirs:
                    continue
                nxt = (cur[0] + nx_d, cur[1] + ny_d)
                if nxt in blocked_targets:
                    continue
                if in_screen(nxt):
                    if nxt not in seeds:
                        continue
                if not in_half_plane(nxt):
                    nc = seg.get(nxt)
                    if nc is None or nc.kind in (WALKABLE, UNKNOWN) or is_door(nc):
                        half_plane_open = True
                        break
                    continue
                nc = seg.get(nxt)
                if nc is None or nc.kind == UNKNOWN or is_door(nc):
                    half_plane_open = True
                    break
                if nc.kind == WALKABLE and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
                    if len(visited) > 60:
                        half_plane_open = True
                        break
            if half_plane_open:
                break
        if not half_plane_open:
            return visited, beyond_cells

        for min_offset in (1, 2):
            def in_forward_offset(g: Tuple[int, int], off: int = min_offset) -> bool:
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
                if c is not None and is_door(c):
                    open2 = True
                    break
                cur_blocked_dirs = seg_blocked_moves.get(cur, set())
                for d_name, (nx_d, ny_d) in DIRECTIONS.items():
                    if d_name in cur_blocked_dirs:
                        continue
                    nxt = (cur[0] + nx_d, cur[1] + ny_d)
                    if nxt in blocked_targets or not in_forward_offset(nxt):
                        continue
                    nc = seg.get(nxt)
                    if nc is None or nc.kind == UNKNOWN or is_door(nc):
                        open2 = True
                        break
                    if nc.kind == WALKABLE and nxt not in visited2:
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

    def _edge_pocket_note(self, direction: str, local_edge_cells: List[Tuple[int, int]]) -> Optional[str]:
        place = self._current_map_place()
        if place is None or not local_edge_cells:
            return None
        px, py = place[1]
        runs = edge_runs(local_edge_cells, direction)
        closed_runs = []
        open_runs = []
        visited_all: set = set()
        beyond_all: set = set()
        for r in runs:
            bounds = self._run_pocket_bounds(direction, r)
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
            return (f" [mapped closed dead-end pocket beyond {direction} edge: "
                    f"{len(beyond_all)} off-screen walkable tile(s), relative x in [{min(xs)}, {max(xs)}], "
                    f"y in [{min(ys)}, {max(ys)}], no unexplored tiles or doors]")
        closed_cells = [c for r in closed_runs for c in r]
        closed_str = ", ".join(f"({x}, {y})" for x, y in closed_cells)
        return (f" [dead-end alcove at {closed_str}: leads into a mapped closed pocket "
                f"(relative x in [{min(xs)}, {max(xs)}], y in [{min(ys)}, {max(ys)}], no unexplored tiles or doors)]")

    def _describe_edges(
        self,
        kinds: Dict[Tuple[int, int], str],
        identified: Optional[Dict[Tuple[int, int, str], Any]] = None,
        people: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> str:
        from execution.perception.pokemon.navigation import (
            edge_hazard_annotations,
            filter_open_lane_edge_cells,
            warp_cells,
        )
        gmap = self._global_map()
        warps = warp_cells(identified=identified, global_map=gmap)
        lines = []
        for direction in DIRECTIONS:
            cells = edge_cells(kinds, direction)
            if cells:
                runs = edge_runs(cells, direction)
                open_cells = [c for r in runs if self._run_pocket_bounds(direction, r) is None for c in r]
                open_lane_cells = filter_open_lane_edge_cells(
                    open_cells or cells, direction, kinds, gmap, identified=identified, people=people
                )
                listed = ", ".join(f"({x}, {y})" for x, y in cells)
                path = shortest_path(
                    kinds, (0, 0), open_lane_cells or open_cells or cells, avoid_cells=warps, people=people
                )
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
                note = self._edge_pocket_note(direction, cells)
                if note:
                    listed += note
            else:
                listed = "none"
            lines.append(f"- {direction} edge: {listed}")
        return "\n".join(lines)

    def _area_map_block(self, local_kinds: Optional[Dict[Tuple[int, int], str]] = None) -> str:
        gmap = self._global_map()
        place = self._current_map_place()
        if gmap is None or place is None:
            return ""
        if self._agent_state() != AgentState.FREE_ROAM or self._is_in_dialogue():
            return ""
        cur_seg, cur_pos = place
        lines = []
        remaining_calls = max(0, self._max_steps - self._calls)
        if self._start_place is not None:
            s_seg, s_pos = gmap.resolve(self._start_place[0], self._start_place[1])
            if s_seg == cur_seg:
                dx, dy = cur_pos[0] - s_pos[0], cur_pos[1] - s_pos[1]
                lines.append(
                    f"Current area coordinate: {cur_pos} (episode start: {s_pos}; "
                    f"net displacement: ({dx:+d}, {dy:+d}), where +x is right/east and +y is up/north; "
                    f"episode action budget used: {self._calls}/{self._max_steps}, {remaining_calls} remaining)."
                )
            else:
                lines.append(
                    f"Current area coordinate: {cur_pos} (entered a new area since episode start; "
                    f"episode action budget used: {self._calls}/{self._max_steps}, {remaining_calls} remaining)."
                )
        else:
            lines.append(
                f"Current area coordinate: {cur_pos} (+x is right/east, +y is up/north; "
                f"episode action budget used: {self._calls}/{self._max_steps}, {remaining_calls} remaining)."
            )
        same_seg_stops = []
        for seg, pos in self._leg_stops:
            r_seg, r_pos = gmap.resolve(seg, pos)
            if r_seg == cur_seg and (not same_seg_stops or same_seg_stops[-1] != r_pos):
                same_seg_stops.append(r_pos)
        if not same_seg_stops or same_seg_stops[-1] != cur_pos:
            same_seg_stops.append(cur_pos)
        if len(same_seg_stops) > 1:
            xs = [p[0] for p in same_seg_stops]
            ys = [p[1] for p in same_seg_stops]
            trail_str = " -> ".join(f"({x}, {y})" for x, y in same_seg_stops)
            lines.append(
                f"Leg positions visited in this area during this episode: {trail_str} "
                f"(x span [{min(xs)}, {max(xs)}], y span [{min(ys)}, {max(ys)}])."
            )
        seg_blocked_moves = self._current_segment_blocked_moves()
        lines.append(
            "Map of what has been seen in this area around the player "
            "(@ is the player at (0, 0) on the current screen, north is up, one character per tile):"
        )
        lines.append(ascii_window(gmap, blocked_moves=seg_blocked_moves))
        lines.append("Legend: " + LEGEND)
        blocked_by_pos: Dict[Tuple[int, int], Dict[str, List[str]]] = {}
        for r_pos, dirs in seg_blocked_moves.items():
            for d in sorted(dirs):
                blocked_by_pos.setdefault(r_pos, {}).setdefault(d, [])
        for (b_seg, b_pos), by_dir in self._blocked_moves.items():
            r_seg, r_pos = gmap.resolve(b_seg, b_pos)
            if r_seg == cur_seg:
                for d, notes in by_dir.items():
                    dst_notes = blocked_by_pos.setdefault(r_pos, {}).setdefault(d, [])
                    for n_str in notes:
                        if n_str not in dst_notes:
                            dst_notes.append(n_str)
        blocked_lines = []
        for r_pos in sorted(blocked_by_pos):
            by_dir = blocked_by_pos[r_pos]
            if not by_dir:
                continue
            here_tag = " [HERE]" if r_pos == cur_pos else ""
            dir_summaries = [
                f"{d} ({'; '.join(notes[-2:])})" if notes else d
                for d, notes in by_dir.items()
            ]
            blocked_lines.append(f"- at {r_pos}{here_tag}: blocked moving {', '.join(dir_summaries)}")
        if blocked_lines:
            lines.append("Blocked movement directions recorded in this area during this episode (do NOT repeat these from the same coordinate):")
            lines.extend(blocked_lines)
        lines += door_lines(gmap, local_kinds, blocked_moves=seg_blocked_moves)
        frontier_count = len(gmap.frontier())
        targets, _ = frontier_targets(gmap, local_kinds, blocked_moves=seg_blocked_moves)
        if targets:
            lines.append(
                f"Nearest unexplored map edges in this area ({frontier_count} reachable unexplored edge square(s) "
                f"remaining in this area; x = steps right, y = steps up from the player):"
            )
            for t in targets[:6]:
                where = "on screen, reachable" if t["on_screen"] else "off screen"
                leg_note = f" [next on-screen leg: {t['next_leg']}]" if t.get("next_leg") else ""
                if t.get("path_moves"):
                    route_note = f"{where}; mapped route ({t['path_steps']} moves): {t['path_moves']}{leg_note}"
                elif t.get("path_steps") is None:
                    route_note = f"{where}; no direct mapped path yet"
                else:
                    route_note = where
                lines.append(f"- ({t['dx']}, {t['dy']}): {t['steps']} steps {t['sector']} ({route_note})")
        elif frontier_count == 0:
            lines.append("All reachable map edges in this area have been explored (0 unexplored edge squares remaining).")
        hp_block = self._party_hp_block()
        if hp_block:
            lines.insert(1 if lines else 0, hp_block)
        return PLAYTHROUGH_AREA_MAP_BLOCK.replace("[AREA_MAP]", "\n".join(lines))

    def _format_leg_movement(
        self,
        before_place: Optional[Tuple[int, Tuple[int, int]]],
        after_place: Optional[Tuple[int, Tuple[int, int]]],
        prior_stops: List[Tuple[int, Tuple[int, int]]],
        is_move_action: bool = False,
    ) -> str:
        gmap = self._global_map()
        if gmap is None or before_place is None or after_place is None:
            return ""
        b_seg, b_pos = gmap.resolve(before_place[0], before_place[1])
        a_seg, a_pos = gmap.resolve(after_place[0], after_place[1])
        if b_seg != a_seg:
            return f"[Area coordinate: {b_pos} -> {a_pos} (entered new area)]"
        if b_pos == a_pos and not is_move_action:
            return ""
        dx, dy = a_pos[0] - b_pos[0], a_pos[1] - b_pos[1]
        prior_same = [gmap.resolve(seg, pos)[1] for seg, pos in prior_stops if gmap.resolve(seg, pos)[0] == a_seg]
        revisit = ""
        if a_pos != b_pos and a_pos in prior_same:
            revisit = "; revisited an earlier coordinate from this episode"
        if self._start_place is not None:
            s_seg, s_pos = gmap.resolve(self._start_place[0], self._start_place[1])
            if s_seg == a_seg:
                net_dx, net_dy = a_pos[0] - s_pos[0], a_pos[1] - s_pos[1]
                return (
                    f"[Area coordinate: {b_pos} -> {a_pos} "
                    f"(step delta ({dx:+d}, {dy:+d}); net from episode start {s_pos}: ({net_dx:+d}, {net_dy:+d}){revisit})]"
                )
        return f"[Area coordinate: {b_pos} -> {a_pos} (step delta ({dx:+d}, {dy:+d}){revisit})]"

    def _perceive(self, frame) -> str:
        self._record_map_stop()
        state = self._effective_state(frame)
        if state != AgentState.FREE_ROAM:
            self._identified = {}
            self._screen_tiles = describe_state_screen(state)
            return self._screen_tiles
        self._tile_recognizer.record_tiles(frame, self._episode_number)
        gmap = self._global_map()
        if gmap is not None:
            gmap.refresh(self._tile_recognizer._by_hash)
        self._identified = self._tile_recognizer.identify_tiles(frame, global_map=gmap)
        people = self._people_now()
        kinds = cell_kinds(self._identified, getattr(self._env, "ledge_patterns", None),
                           getattr(self._env, "walk_patterns", None), people,
                           getattr(self._env, "left_ledge_patterns", None),
                           self._local_blocked_cells())
        banner_note = (
            "\nNote: The thin 2-tile-tall bordered box at the bottom of the screen is a non-blocking area/location "
            "name popup banner (NOT a dialogue box). The player is already in free roam and can walk normally right "
            "now (the banner disappears automatically as you walk)—do NOT plan a dialogue-clearing subgoal or call "
            "Dialogue() or Free() to dismiss it; proceed directly with MoveTo, MoveOff, or Interact."
            if has_bottom_location_banner(frame) and not has_bottom_dialogue_box(frame)
            else ""
        )
        self._screen_tiles = (verbalize_tiles(self._identified, people, kinds)
                              + "\nWalkable cells on each edge of the screen:\n"
                              + self._describe_edges(kinds, identified=self._identified, people=people)
                              + banner_note)
        return self._screen_tiles

    def _people_now(self) -> Optional[Dict[Tuple[int, int], str]]:
        if self._screen_people is None:
            return None
        try:
            return self._screen_people()
        except Exception as e:  # noqa: BLE001
            log_warn(f"People on screen could not be named: {type(e).__name__}: {e}", self._parameters)
            return None

    def _log_block(self) -> str:
        if not self.report.narrative:
            return ""
        entries = "\n".join(f"- {entry}" for entry in self.report.narrative)
        return PLAYTHROUGH_LOG_BLOCK.replace("[LOG]", entries)

    def _current_local_kinds(self) -> Optional[Dict[Tuple[int, int], str]]:
        if self._effective_state() == AgentState.FREE_ROAM and self._identified:
            return cell_kinds(
                self._identified,
                getattr(self._env, "ledge_patterns", None),
                getattr(self._env, "walk_patterns", None),
                self._people_now(),
                getattr(self._env, "left_ledge_patterns", None),
                self._local_blocked_cells(),
            )
        return None

    def _unfinished_room_exploration_note(
        self,
        text: str,
        local_kinds: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> Optional[str]:
        if max(0, self._max_steps - self._calls) < 2:
            return None
        if self._agent_state() != AgentState.FREE_ROAM or self._is_in_dialogue():
            return None
        gmap = self._global_map()
        place = self._current_map_place()
        if gmap is None or place is None:
            return None
        if not re.search(r"\bexplore\b", text or "", re.IGNORECASE):
            return None
        frontier_cells = gmap.frontier()
        if not (1 <= len(frontier_cells) <= 16):
            return None
        targets, _ = frontier_targets(gmap, local_kinds, blocked_moves=self._current_segment_blocked_moves())
        if not targets:
            return None
        nearest = targets[0]
        dist = nearest.get("path_steps") if nearest.get("path_steps") is not None else nearest.get("steps")
        if dist is None or dist > 15:
            return None
        leg_str = f" [next on-screen leg: {nearest['next_leg']}]" if nearest.get("next_leg") else ""
        route_str = f"; mapped route ({nearest['path_steps']} moves): {nearest['path_moves']}{leg_str}" if nearest.get("path_moves") else ""
        return (
            f"{len(frontier_cells)} reachable unexplored edge square(s) still remain in this room "
            f"(nearest at ({nearest['dx']}, {nearest['dy']}), {nearest['steps']} steps {nearest['sector']}{route_str}) "
            f"— walk to reveal the remaining unexplored edge squares before finishing"
        )

    def _fill_prompt(
        self,
        prompt: str,
        local_kinds_override: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> str:
        if self._screen_tiles is None:
            log_error("Supervisor prompt built before the screen was perceived.", self._parameters)
        guidance_block = PLAYTHROUGH_GUIDANCE_BLOCK.replace("[GUIDANCE]", self._guidance) if self._guidance else ""
        plan = "\n".join(f"{n + 1}. {subgoal}" for n, subgoal in enumerate(self.plan))
        raw_state = self._effective_state()
        state = raw_state.name.lower().replace("_", " ")
        local_kinds = local_kinds_override if local_kinds_override is not None else self._current_local_kinds()
        return (prompt.replace("[GAME]", self._game)
                .replace("[TASK]", self._task)
                .replace("[GUIDANCE_BLOCK]", guidance_block)
                .replace("[LOG_BLOCK]", self._log_block())
                .replace("[SCREEN_TILES]", self._screen_tiles)
                .replace("[AREA_MAP_BLOCK]", self._area_map_block(local_kinds))
                .replace("[AGENT_STATE]", state)
                .replace("[PRIMITIVES]", PRIMITIVES)
                .replace("[PLAN]", plan))

    def call_executor(self, tag: str, task: str, *, hint: Optional[str] = None,
                      allow_self_termination: bool = False,
                      max_steps: Optional[int] = None, counted: bool = True,
                      **executor_kwargs: Any) -> Any:
        if tag not in self.EXECUTORS:
            log_error(f"Unknown executor tag {tag!r}. Options: {list(self.EXECUTORS)}",
                      self._parameters)
        run_kwargs = dict(self._executor_kwargs)
        run_kwargs.update(executor_kwargs)
        run_kwargs["hint"] = hint
        run_kwargs["allow_self_termination"] = allow_self_termination
        self._current_tag = tag
        self._total_calls += 1
        if counted:
            self._calls += 1
        try:
            executor = self.EXECUTORS[tag](
                env=self._env,
                task=task,
                game=self._game,
                max_steps=EXECUTOR_STEPS[tag] if max_steps is None else max_steps,
                parameters=self._parameters,
                **run_kwargs,
            )
            self.last_executor = executor
            self.report.event_log.append(executor.report)
            hook = getattr(self, "_on_event_logged", None)
            if callable(hook):
                hook()
            return self.process_executor_return(executor.report)
        finally:
            self._current_tag = None

    def process_executor_return(self, report: ExecutorReport) -> Any:
        processors = {
            "interaction": self._process_interaction,
            "on_screen_move": self._process_on_screen_move,
            "off_screen_move": self._process_off_screen_move,
            "combat": self._process_combat,
            "menu": self._process_menu,
            "dialogue": self._process_dialogue,
            "free": self._process_free,
        }
        if self._current_tag not in processors:
            log_error(f"process_executor_return called with no processor for tag "
                      f"{self._current_tag!r}", self._parameters)
        return processors[self._current_tag](report)

    def _result(self, tag: str, report: ExecutorReport, succeeded: bool, **extra) -> Dict[str, Any]:
        return {
            "tag": tag,
            "report": report,
            "succeeded": succeeded,
            "code": report.outcome,
            "reason": report.termination_reason,
            "notes": report.notes,
            **extra,
        }

    def _process_interaction(self, report: ExecutorReport) -> Dict[str, Any]:
        succeeded = report.outcome in (0, 1, 2)
        return self._result("interaction", report, succeeded,
                            dialogue=report.notes if succeeded else None)

    def _process_on_screen_move(self, report: ExecutorReport) -> Dict[str, Any]:
        return self._result("on_screen_move", report, report.outcome == MOVE_OK,
                            location_failed=report.outcome in LOCATION_FAIL_CODES,
                            ambiguous=report.outcome == MOVE_TARGET_AMBIGUOUS)

    def _process_off_screen_move(self, report: ExecutorReport) -> Dict[str, Any]:
        return self._result("off_screen_move", report, report.outcome == MOVE_OK,
                            location_failed=report.outcome in LOCATION_FAIL_CODES,
                            ambiguous=report.outcome == MOVE_TARGET_AMBIGUOUS)

    def _update_party_hp_from_combat(self, report: ExecutorReport) -> None:
        if getattr(self.last_executor, "_bag_unavailable", False):
            self._healing_bag_empty = True
        hp = CombatExecutor.extract_player_hp_from_text(report.notes or "")
        if hp is not None:
            self._last_seen_battle_hp = hp

    def _update_party_hp_from_menu(self, report: ExecutorReport) -> None:
        notes = report.notes or ""
        if "No healing item was used in menu" in notes:
            self._healing_bag_empty = True
            return
        menu_hp = CombatExecutor.extract_player_hp_from_text(notes)
        m = re.search(r"\b(?:recovered(?:\s+by)?|restored)\s+(\d+)(?:\s*HP)?\b", notes, re.IGNORECASE)
        if m:
            self._healing_bag_empty = False
            if menu_hp is not None:
                self._last_seen_battle_hp = menu_hp
            elif self._last_seen_battle_hp is not None:
                delta = int(m.group(1))
                cur_hp, max_hp = self._last_seen_battle_hp
                self._last_seen_battle_hp = (min(max_hp, cur_hp + delta), max_hp)
        elif re.search(
            r"\b(?:fighting\s+fit|fully\s+healed|restored\s+(?:your\s+POK[eé]MON\s+)?to\s+full\s+health)\b",
            notes,
            re.IGNORECASE,
        ):
            if menu_hp is not None:
                self._last_seen_battle_hp = (menu_hp[1], menu_hp[1])
            elif self._last_seen_battle_hp is not None:
                _, max_hp = self._last_seen_battle_hp
                self._last_seen_battle_hp = (max_hp, max_hp)
        elif menu_hp is not None:
            self._last_seen_battle_hp = menu_hp

    def _party_hp_block(self) -> str:
        if self._last_seen_battle_hp is None and not self._healing_bag_empty:
            return ""
        lines: List[str] = []
        if self._last_seen_battle_hp is not None:
            cur_hp, max_hp = self._last_seen_battle_hp
            low = cur_hp * 2 <= max_hp
            bag_note = (
                " (healing bag was already checked and has no usable healing items)"
                if self._healing_bag_empty else ""
            )
            if low and not self._healing_bag_empty:
                lines.append(
                    f"Active Pokemon last seen battle HP: {cur_hp}/{max_hp} HP (LOW — at or below 50% max HP; "
                    f"before walking further along a route/cave or talking to a trainer, use Menu(open the ITEM menu "
                    f"and use a Potion to heal your active Pokemon, then close the menu) first)."
                )
            elif low and self._healing_bag_empty:
                lines.append(
                    f"Active Pokemon last seen battle HP: {cur_hp}/{max_hp} HP (LOW — at or below 50% max HP){bag_note}."
                )
            else:
                lines.append(f"Active Pokemon last seen battle HP: {cur_hp}/{max_hp} HP{bag_note}.")
        elif self._healing_bag_empty:
            lines.append("Bag healing item status: No usable healing items in bag (checked earlier).")
        return "\n".join(lines)

    def _process_combat(self, report: ExecutorReport) -> Dict[str, Any]:
        self._update_party_hp_from_combat(report)
        return self._result("combat", report, report.outcome == BATTLE_ENDED,
                            dialogue=report.notes,
                            left_battle=report.outcome in (BATTLE_LEFT_TO_DIALOGUE, BATTLE_LEFT_TO_MENU))

    def _process_menu(self, report: ExecutorReport) -> Dict[str, Any]:
        self._update_party_hp_from_menu(report)
        succeeded = (report.outcome in (MENU_OK, MENU_LEFT_TO_DIALOGUE, MENU_LEFT_TO_BATTLE)
                     and bool(report.steps))
        return self._result("menu", report,
                            succeeded,
                            dialogue=report.notes,
                            left_menu=report.outcome in (MENU_LEFT_TO_DIALOGUE, MENU_LEFT_TO_BATTLE))

    def _process_dialogue(self, report: ExecutorReport) -> Dict[str, Any]:
        return self._result("dialogue", report,
                            report.outcome in (DIALOGUE_CLEARED, DIALOGUE_LEFT_TO_MENU, DIALOGUE_LEFT_TO_BATTLE),
                            dialogue=report.notes,
                            left_dialogue=report.outcome in (DIALOGUE_LEFT_TO_MENU, DIALOGUE_LEFT_TO_BATTLE))

    def _process_free(self, report: ExecutorReport) -> Dict[str, Any]:
        dialogue = getattr(self.last_executor, "dialogue_text", None) or None
        return self._result("free", report,
                            report.outcome == FREE_OK and bool(report.steps),
                            dialogue=dialogue)

    def _plan_subgoals(self, frame) -> List[str]:
        output = self._vlm_call("plan", texts=self._fill_prompt(PLAYTHROUGH_PLAN_PROMPT), images=[frame])
        subgoals = [line.strip() for line in parse_list(output, "Subgoals") if line.strip()]
        if not subgoals:
            log_warn(f"Could not parse a plan from: {output!r}", self._parameters)
        return subgoals

    MAX_DECISION_RETRY_CHARS = 1500

    def _best_open_edge_target(
        self,
        direction: str,
        local_kinds: Dict[Tuple[int, int], str],
    ) -> Optional[Tuple[int, int]]:
        cells = edge_cells(local_kinds, direction)
        if not cells:
            return None
        runs = edge_runs(cells, direction)
        open_cells = [c for r in runs if self._run_pocket_bounds(direction, r) is None for c in r]
        if not open_cells:
            return None
        gmap = self._global_map()
        people = self._people_now()
        warps = warp_cells(identified=self._identified, global_map=gmap)
        open_lane_cells = filter_open_lane_edge_cells(
            open_cells, direction, local_kinds, gmap, identified=self._identified, people=people
        )
        candidates = open_lane_cells or open_cells
        path = shortest_path(local_kinds, (0, 0), candidates, avoid_cells=warps, people=people)
        return path[-1][2] if path else candidates[0]

    def _coerce_free_navigation_decision(
        self,
        argument: Optional[str],
        frame,
    ) -> Optional[Tuple[str, str]]:
        if not argument:
            return None
        if (
            self._effective_state(frame) != AgentState.FREE_ROAM
            or self._is_in_dialogue(frame)
            or has_bottom_dialogue_box(frame)
            or has_top_dialogue_box(frame)
            or has_emote_bubble(frame)
            or is_blank_or_fade_frame(frame)
            or has_menu_cursor(frame)
            or is_battle_screen(frame)
        ):
            return None
        instruction, termination = split_free_argument(argument)
        if re.search(
            r"\b(?:a|b|start|menu|talk|speak|interact|dialogue|text|cutscene|wait|evolve|evolution)\b",
            instruction,
            re.IGNORECASE,
        ):
            return None
        if re.search(
            r"\b(?:up\s+and\s+down|down\s+and\s+up|left\s+and\s+right|right\s+and\s+left|back\s+and\s+forth|tall\s+grass|grind|level\s+up)\b",
            instruction,
            re.IGNORECASE,
        ):
            return None
        if termination and re.search(r"\bwild\s+pokemon\b", termination, re.IGNORECASE) and not re.search(
            r"\b(?:or|walked|reached|passed|past|cleared|entered|exited|chamber|corridor|passage|pillar|wall)\b",
            termination,
            re.IGNORECASE,
        ):
            return None
        local_kinds = self._current_local_kinds()
        if not local_kinds:
            return None
        if boundary_exit_direction(
            local_kinds, self._identified, (0, 0), global_map=self._global_map()
        ) is not None:
            return None
        legs = parse_free_directional_legs(instruction)
        if not legs:
            return None
        if len(legs) == 1 and legs[0][1] <= 2 and is_boundary_exit_direction(local_kinds, (0, 0), legs[0][0]):
            return None
        if len(legs) < 2 and legs[0][1] < 4:
            return None
        gmap = self._global_map()
        warps = warp_cells(identified=self._identified, global_map=gmap)
        if len(legs) >= 2:
            cur = (0, 0)
            prefix_ok = True
            for d_leg, n_leg in legs[:-1]:
                dx_l, dy_l = DIRECTIONS[d_leg]
                for _ in range(n_leg):
                    nxt = (cur[0] + dx_l, cur[1] + dy_l)
                    if not in_bounds(nxt) or local_kinds.get(nxt) != WALKABLE or nxt in warps:
                        prefix_ok = False
                        break
                    cur = nxt
                if not prefix_ok:
                    break
            if prefix_ok:
                dir_last, n_last = legs[-1]
                dx_last, dy_last = DIRECTIONS[dir_last]
                probe = cur
                reached_edge_cell = None
                for _ in range(max(n_last, 10)):
                    if is_on_edge(probe, dir_last):
                        reached_edge_cell = probe
                        break
                    nxt = (probe[0] + dx_last, probe[1] + dy_last)
                    if not in_bounds(nxt) or local_kinds.get(nxt) != WALKABLE or nxt in warps:
                        break
                    probe = nxt
                    if is_on_edge(probe, dir_last):
                        reached_edge_cell = probe
                        break
                if reached_edge_cell is not None and reached_edge_cell in edge_cells(local_kinds, dir_last):
                    return "MoveOff", f"{dir_last} via {reached_edge_cell}"
            for d_leg, _ in legs:
                target = self._best_open_edge_target(d_leg, local_kinds)
                if target is not None:
                    return "MoveOff", f"{d_leg} via {target}"
        elif len(legs) == 1 and legs[0][1] >= 4:
            target = self._best_open_edge_target(legs[0][0], local_kinds)
            if target is not None:
                return "MoveOff", f"{legs[0][0]} via {target}"
        return None

    def _decide(self, subgoal: str, attempts: List[str], frame) -> Tuple[Optional[Tuple[str, Optional[str]]], str]:
        attempts_block = ""
        if attempts:
            attempts_block = PLAYTHROUGH_ATTEMPTS_BLOCK.replace(
                "[ATTEMPTS]", "\n".join(f"- {attempt}" for attempt in attempts))
        prompt = (self._fill_prompt(PLAYTHROUGH_SUBGOAL_DECISION_PROMPT)
                  .replace("[SUBGOAL]", subgoal)
                  .replace("[ATTEMPTS_BLOCK]", attempts_block))
        output, records = self._vlm_infer("decision", texts=prompt, images=[frame])
        decision = parse_decision(output)
        if decision is None:
            self.report.event_log.extend(records)
            hook = getattr(self, "_on_event_logged", None)
            if callable(hook):
                hook()
            retry_prompt = prompt + PLAYTHROUGH_DECISION_RETRY_BLOCK.replace(
                "[PREVIOUS]", (output or "")[-self.MAX_DECISION_RETRY_CHARS:]
            )
            output, records = self._vlm_infer("decision", texts=retry_prompt, images=[frame])
            decision = parse_decision(output)
        if decision is None:
            log_warn(f"Could not parse a decision from: {output!r}", self._parameters)
        elif decision[0] == "Free":
            coerced = self._coerce_free_navigation_decision(decision[1], frame)
            if coerced is not None:
                decision = coerced
                if records:
                    records[-1].response = re.sub(
                        r"(?im)^(\s*Action\s*:\s*).*$",
                        f"\1{coerced[0]}({coerced[1]})",
                        str(records[-1].response),
                    )
        self.report.event_log.extend(records)
        hook = getattr(self, "_on_event_logged", None)
        if callable(hook):
            hook()
        reasoning = (parse_key_value(output, "Reasoning") or "").strip()
        return decision, reasoning

    def _judge(self, subgoal: str, request: str, result: Dict[str, Any],
               subgoal_index: int = 0, move_tag: str = "",
               decision_reasoning: str = "") -> Tuple[bool, str, int, bool]:
        report = result["report"]
        before = report.initial_state["core"]["current_frame"]
        after = report.final_state["core"]["current_frame"]
        after_state = self._effective_state(after, raw_state=report.final_state["pokemon_core"]["agent_state"])
        after_kinds = None
        if after_state == AgentState.FREE_ROAM:
            self._tile_recognizer.record_tiles(after, self._episode_number)
            gmap = self._global_map()
            if gmap is not None:
                gmap.refresh(self._tile_recognizer._by_hash)
            after_identified = self._tile_recognizer.identify_tiles(after, global_map=gmap)
            after_people = self._people_now()
            after_kinds = cell_kinds(after_identified, getattr(self._env, "ledge_patterns", None),
                                     getattr(self._env, "walk_patterns", None), after_people,
                                     getattr(self._env, "left_ledge_patterns", None),
                                     self._local_blocked_cells())
            after_banner_note = (
                "\nNote: The thin 2-tile-tall bordered box at the bottom of the screen is a non-blocking area/location "
                "name popup banner (NOT a dialogue box); the player is in free roam."
                if has_bottom_location_banner(after) and not has_bottom_dialogue_box(after)
                else ""
            )
            after_tiles = (verbalize_tiles(after_identified, after_people, after_kinds)
                           + "\nWalkable cells on each edge of the screen:\n"
                           + self._describe_edges(after_kinds, identified=after_identified, people=after_people)
                           + after_banner_note)
            if move_tag:
                after_tiles += f"\n{move_tag}"
            after_block = PLAYTHROUGH_AFTER_BLOCK.replace("[AFTER_TILES]", after_tiles)
        else:
            after_desc = describe_state_screen(after_state)
            if move_tag:
                after_desc += f"\n{move_tag}"
            after_block = PLAYTHROUGH_AFTER_BLOCK.replace("[AFTER_TILES]", after_desc)
        text = result.get("dialogue") or ""
        text_block = PLAYTHROUGH_TEXT_BLOCK.replace("[TEXT]", text) if text.strip() else ""
        executor_report = f"{result['reason']} (code {result['code']})"
        if result.get("notes") and result.get("notes") != text:
            executor_report += f": {result['notes']}"
        remaining = [f"{n + 1}. {sg}" for n, sg in enumerate(self.plan) if n > subgoal_index]
        if remaining:
            remaining_block = PLAYTHROUGH_REMAINING_SUBGOALS_BLOCK.replace(
                "[REMAINING_SUBGOALS]", "\n".join(remaining))
            remaining_line = PLAYTHROUGH_REMAINING_SUBGOALS_LINE
        else:
            remaining_block = ""
            remaining_line = ""
        request_with_context = request
        if decision_reasoning:
            request_with_context = f"{request}\nWhy you chose that action: {decision_reasoning}"
        prompt = (self._fill_prompt(PLAYTHROUGH_JUDGE_PROMPT, local_kinds_override=after_kinds)
                  .replace("[SUBGOAL_NUMBER]", str(subgoal_index + 1))
                  .replace("[SUBGOAL]", subgoal)
                  .replace("[REMAINING_SUBGOALS_BLOCK]", remaining_block)
                  .replace("[REMAINING_SUBGOALS_LINE]", remaining_line)
                  .replace("[REQUEST]", request_with_context)
                  .replace("[AFTER_BLOCK]", after_block)
                  .replace("[TEXT_BLOCK]", text_block)
                  .replace("[EXECUTOR_REPORT]", executor_report))
        output = self._vlm_call("judge", texts=prompt, images=[before, after])
        summary = (parse_key_value(output, "Summary") or "").strip()
        if not summary:
            summary = f"{request} -> {executor_report}"
        complete = bool(parse_yes_no(output, "Subgoal complete"))
        replan_needed = bool(parse_yes_no(output, "Replan needed")) and not complete
        reached = parse_int(output, "Reached subgoal", lo=subgoal_index + 1, hi=len(self.plan)) if remaining else None
        if not complete and reached is not None and reached > subgoal_index + 1:
            complete = True
            replan_needed = False
        reached_index = (reached - 1) if (complete and reached is not None) else subgoal_index
        if complete and reached_index >= len(self.plan) - 1:
            unfinished_note = self._unfinished_room_exploration_note(subgoal, after_kinds)
            if unfinished_note:
                complete = False
                replan_needed = False
                summary = f"{summary} [Note: {unfinished_note}.]"
        return complete, summary, reached_index, replan_needed

    def _reconsider(self, subgoal: str, attempts: List[str], frame) -> Tuple[str, Optional[str], List[str]]:
        prompt = (self._fill_prompt(PLAYTHROUGH_ACHIEVABLE_PROMPT)
                  .replace("[SUBGOAL]", subgoal)
                  .replace("[ATTEMPTS]", "\n".join(f"- {attempt}" for attempt in attempts)))
        output = self._vlm_call("reconsider", texts=prompt, images=[frame])
        decision = (parse_key_value(output, "Decision") or "").strip().lower()
        reason = (parse_key_value(output, "Reason") or "").strip()
        subgoals = [line.strip() for line in parse_list(output, "Subgoals") if line.strip()]
        if "quit" in decision or not subgoals:
            return "quit", reason or "No new plan was given.", []
        return "replan", None, subgoals

    def _unfinished_corridor_traversal_note(
        self,
        local_kinds: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> Optional[str]:
        remaining_calls = max(0, self._max_steps - self._calls)
        if remaining_calls < 4:
            return None
        if self._agent_state() != AgentState.FREE_ROAM or self._is_in_dialogue():
            return None
        gmap = self._global_map()
        place = self._current_map_place()
        if gmap is None or place is None or self._start_place is None:
            return None
        cur_seg, cur_pos = place
        s_seg, s_pos = gmap.resolve(self._start_place[0], self._start_place[1])
        if cur_seg != s_seg:
            return None
        move_legs = sum(
            1 for rep in self.report.executor_reports
            if rep.executor_name in ("OffScreenMoveExecutor", "OnScreenMoveExecutor")
        )
        if move_legs == 0 or move_legs >= 3:
            return None
        net_steps = abs(cur_pos[0] - s_pos[0]) + abs(cur_pos[1] - s_pos[1])
        if net_steps >= 15:
            return None
        task_text = (self._task or "") + " " + (self._guidance or "")
        if re.search(r"\b(?:heal|potion|menu|poke\s*mart|pokemon\s*center|talk|speak|interact|pick\s+up|collect)\b",
                     self._task or "", re.IGNORECASE):
            return None
        dir_map = {
            "north": "up", "up": "up",
            "south": "down", "down": "down",
            "west": "left", "left": "left",
            "east": "right", "right": "right",
        }
        task_dirs = []
        for m in re.finditer(r"\b(north|south|east|west|up|down|left|right)\b", self._task or "", re.IGNORECASE):
            d = dir_map[m.group(1).lower()]
            if d not in task_dirs:
                task_dirs.append(d)
        if not task_dirs:
            return None
        if local_kinds is not None:
            for d in DIRECTIONS:
                for (lx, ly), kind in local_kinds.items():
                    if abs(lx) + abs(ly) <= 6 and kind == "item":
                        return None
        targets, _ = frontier_targets(gmap, local_kinds, blocked_moves=self._current_segment_blocked_moves())
        doors_here = gmap.reachable_door_list()
        unentered_doors = [d for d in doors_here if d["leads_to"] is None]
        if not targets and not unentered_doors:
            return None
        if local_kinds is not None:
            for (lx, ly), info in (self._identified or {}).items():
                tags = info if isinstance(info, (set, list, tuple)) else getattr(info, "tags", ())
                if any(t in ("door", "staircase", "entrance", "exit") for t in (tags or ())):
                    if (lx, ly) != (0, 0) and abs(lx) + abs(ly) <= 6:
                        return None
        open_task_dirs = []
        if local_kinds is not None:
            for d in task_dirs:
                cells = edge_cells(local_kinds, d)
                if not cells:
                    continue
                runs = edge_runs(cells, d)
                if any(self._run_pocket_bounds(d, r) is None for r in runs):
                    open_task_dirs.append(d)
        if not open_task_dirs:
            return None
        if targets:
            nearest = targets[0]
            dist = nearest.get("path_steps") if nearest.get("path_steps") is not None else nearest.get("steps")
            if dist is not None and dist <= 6 and nearest.get("on_screen"):
                return None
            leg_str = f" [next on-screen leg: {nearest['next_leg']}]" if nearest.get("next_leg") else ""
            route_info = (
                f"nearest unexplored edge is at ({nearest['dx']}, {nearest['dy']}), {nearest['steps']} steps {nearest['sector']}"
                + (f" (mapped route, {nearest['path_steps']} moves: {nearest['path_moves']}{leg_str})" if nearest.get("path_moves") else "")
            )
        else:
            route_info = f"{len(unentered_doors)} unentered door(s)/stairs remain in this area"
        return (
            f"only {move_legs} movement leg(s) ({net_steps} net steps from {s_pos} to {cur_pos}) completed so far "
            f"with {remaining_calls} action calls remaining in this episode's budget, while the {', '.join(open_task_dirs)} "
            f"corridor edge continues open and {route_info} — continue advancing along the corridor/route in this episode"
        )

    def _continuation_unfinished_note(
        self,
        local_kinds: Optional[Dict[Tuple[int, int], str]] = None,
        *,
        can_continue: bool = True,
    ) -> Optional[str]:
        if not can_continue:
            return None
        if local_kinds is None:
            local_kinds = self._current_local_kinds()
        unfinished_note = self._unfinished_room_exploration_note(self._task, local_kinds)
        if unfinished_note is None:
            unfinished_note = self._unfinished_corridor_traversal_note(local_kinds)
        return unfinished_note

    def _task_check(self, frame, *, can_continue: bool = True) -> Tuple[bool, str]:
        remaining_calls = max(0, self._max_steps - self._calls)
        local_kinds = self._current_local_kinds()
        unfinished_note = self._continuation_unfinished_note(local_kinds, can_continue=can_continue)
        if can_continue and remaining_calls > 0:
            extra = f" ({unfinished_note}.)" if unfinished_note else ""
            continuation_note = (
                f"\nNote: A continuation replan is available ({remaining_calls} action call(s) remaining in this episode's budget) "
                f"if the task needs further movement steps.{extra}"
            )
        else:
            continuation_note = (
                "\nNote: No further continuation replan will occur in this episode. Judge strictly whether the stages "
                "named in \"Your task\" have already been achieved on screen; do not answer no merely to use remaining "
                "action calls for extra forward movement."
            )
        prompt = self._fill_prompt(PLAYTHROUGH_TASK_CHECK_PROMPT).replace("[CONTINUATION_NOTE]", continuation_note)
        output = self._vlm_call("task_check", texts=prompt, images=[frame])
        summary = (parse_key_value(output, "Summary") or "").strip()
        complete = bool(parse_yes_no(output, "Task complete"))
        if complete and unfinished_note:
            complete = False
            summary = f"{summary} ({unfinished_note}.)"
        return complete, summary

    def _clarify_target(self, target: str, candidates: Optional[str], frame) -> Optional[str]:
        prompt = (self._fill_prompt(PLAYTHROUGH_CLARIFY_TARGET_PROMPT)
                  .replace("[TARGET]", target)
                  .replace("[CANDIDATES]", candidates or "(the candidates were not listed)"))
        output = self._vlm_call("clarify", texts=prompt, images=[frame])
        clarified = parse_key_value(output, "Target")
        if not clarified or not clarified.strip():
            log_warn(f"Could not parse a clarified target from: {output!r}", self._parameters)
            return None
        return clarified.strip().strip("\"'")

    def _menu_objective(self, instruction: str, frame) -> Tuple[str, bool]:
        state = self._effective_state(frame).name.lower().replace("_", " ")
        prompt = (self._fill_prompt(PLAYTHROUGH_MENU_OBJECTIVE_PROMPT)
                  .replace("[INSTRUCTION]", instruction)
                  .replace("[AGENT_STATE]", state))
        output = self._vlm_call("menu_objective", texts=prompt, images=[frame])
        objective = parse_key_value(output, "Objective")
        if not objective or not objective.strip():
            log_warn(f"Could not parse a menu objective, using the instruction as given: {output!r}", self._parameters)
            objective = instruction
        return objective.strip().strip("\"'"), bool(parse_yes_no(output, "Start menu"))

    def _tips_text(self) -> str:
        return "\n".join(f"- {tip}" for tip in self._battle_tips)

    def _battle_instructions(self, frame) -> str:
        tips = PLAYTHROUGH_BATTLE_TIPS_BLOCK.replace("[TIPS]", self._tips_text()) if self._battle_tips else ""
        hp_line = self._party_hp_block()
        if hp_line:
            tips = f"\n{hp_line}\n" + tips
        output = self._vlm_call("battle_instructions",
                                texts=self._fill_prompt(PLAYTHROUGH_BATTLE_INSTRUCTIONS_PROMPT)
                                .replace("[BATTLE_TIPS_BLOCK]", tips),
                                images=[frame])
        instructions = (parse_key_value(output, "Instructions") or "").strip().strip("\"'")
        if not instructions or instructions.lower() == "none":
            return "win the current battle"
        return instructions

    @staticmethod
    def _is_wild_encounter_dialogue(dialogue_text: Optional[str]) -> bool:
        if not dialogue_text:
            return False
        text = " ".join(str(dialogue_text).split())
        if re.search(r"\b(?:wants\s+to\s+(?:fight|battle)|sent\s+out)\b", text, re.IGNORECASE):
            return False
        return bool(re.search(r"\bwild\b.{1,30}\bappeared\b", text, re.IGNORECASE))

    def _task_wants_wild_battles(self) -> bool:
        combined = f"{self._task or ''} {self._guidance or ''}"
        if re.search(
            r"\b(?:flee|run(?:\s+away)?)\s+from\s+wild\b|\bavoid\s+wild\b",
            combined,
            re.IGNORECASE,
        ):
            if not re.search(
                r"\b(?:train|grind|level\s+up|farm\s+exp|catch|capture)\b",
                self._task or "",
                re.IGNORECASE,
            ):
                return False
        return bool(
            re.search(
                r"\b(?:train|grind|level\s+up|farm\s+exp|gain\s+(?:exp|levels?)|catch|capture)\b"
                r"|\b(?:fight|battle|defeat)\s+wild\b",
                combined,
                re.IGNORECASE,
            )
        )

    def _active_pokemon_hp_healthy(self, frame=None) -> bool:
        hp = read_battle_hud_hp(frame) if frame is not None else None
        if hp is None:
            hp = self._last_seen_battle_hp
        if hp is None:
            return False
        cur_hp, max_hp = hp
        return cur_hp > 0 and max_hp > 0 and (cur_hp * 2 > max_hp)

    def _combat(self, instructions: str, *, counted: bool = True, **executor_kwargs: Any) -> Any:
        task = instructions + (f"\nYour battle tips:\n{self._tips_text()}" if self._battle_tips else "")
        return self.call_executor("combat", task, counted=counted, **executor_kwargs)

    def _record_leg_blocked_moves(self, request: str) -> Dict[Any, Any]:
        exec_blocked = getattr(self.last_executor, "_blocked_at_place", None) or {}
        gmap_now = self._global_map()
        for b_place, b_dirs in exec_blocked.items():
            for b_dir in b_dirs:
                detail = f"{request} -> blocked moving {b_dir}"
                dir_list = self._blocked_moves.setdefault(b_place, {}).setdefault(b_dir, [])
                if detail not in dir_list:
                    dir_list.append(detail)
                if gmap_now is not None and isinstance(b_place, tuple) and len(b_place) == 2:
                    try:
                        r_seg, r_pos = gmap_now.resolve(b_place[0], b_place[1])
                        if r_seg == gmap_now.segment:
                            gmap_now.record_blocked_step(direction=b_dir, source_cell=r_pos)
                    except Exception:  # noqa: BLE001
                        pass
        return dict(exec_blocked)

    def _prepare_wild_move_resume(
        self,
        action: str,
        current_task: str,
        leg_start_place: Optional[Tuple[int, Tuple[int, int]]],
        after_place: Optional[Tuple[int, Tuple[int, int]]],
        state_chain: str,
        handled: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], bool]:
        if (
            not handled.get("succeeded")
            or handled.get("reason") != "battle_ended"
            or "in_battle" not in state_chain
            or self._agent_state() != AgentState.FREE_ROAM
            or self._is_in_dialogue()
            or self._battles_lost()
            or self._env_done()
            or not self._is_wild_encounter_dialogue(handled.get("dialogue"))
            or self._task_wants_wild_battles()
            or not self._active_pokemon_hp_healthy()
        ):
            return False, None, False
        gmap = self._global_map()
        if gmap is None or leg_start_place is None or after_place is None:
            return False, None, False
        s_seg, s_pos = gmap.resolve(leg_start_place[0], leg_start_place[1])
        a_seg, a_pos = gmap.resolve(after_place[0], after_place[1])
        if s_seg != a_seg:
            return False, None, False
        dx = a_pos[0] - s_pos[0]
        dy = a_pos[1] - s_pos[1]
        if action == "MoveOff":
            direction = parse_edge_direction(current_task)
            if direction not in DIRECTIONS:
                return False, None, False
            coords = [c for c in parse_all_cells(current_task) if in_bounds(c)]
            if len(coords) == 1 and (dx, dy) != (0, 0):
                old_c = coords[0]
                new_c = (old_c[0] - dx, old_c[1] - dy)
                if is_on_edge(old_c, direction) and in_bounds(new_c) and is_on_edge(new_c, direction):
                    return True, f"{direction} via {new_c}", False
                return True, direction, False
            return True, current_task, False
        if action == "MoveTo":
            coords = [c for c in parse_all_cells(current_task) if in_bounds(c)]
            if len(coords) == 1:
                old_c = coords[0]
                new_c = (old_c[0] - dx, old_c[1] - dy)
                if new_c == (0, 0):
                    return False, None, True
                if in_bounds(new_c):
                    pattern = re.compile(rf"\(\s*{old_c[0]}\s*,\s*{old_c[1]}\s*\)")
                    updated = pattern.sub(f"({new_c[0]}, {new_c[1]})", current_task, count=1)
                    return True, updated, False
                return False, None, False
            if (dx, dy) == (0, 0):
                return True, current_task, False
        return False, None, False

    def _battles_lost(self) -> int:
        env_losses = self._env.get_info()["pokemon_core"].get("n_battles_lost", 0) - self._losses_at_start
        gmap = self._global_map()
        map_losses = (len(gmap.defeats) - self._defeats_at_start) if gmap is not None else 0
        return max(env_losses, map_losses)

    def _run_action(self, action: str, argument: Optional[str], frame) -> Tuple[Dict[str, Any], str]:
        tag, _, default_task = ACTIONS[action]
        task = argument or default_task
        if tag in MOVE_TAGS:
            result = self._move(tag, task, counted=True)
            clarifications = 0
            while (action == "MoveTo" and result.get("ambiguous")
                   and clarifications < self.MAX_CLARIFICATIONS):
                clarified = self._clarify_target(task, result["notes"], frame)
                if clarified is None:
                    break
                clarifications += 1
                task = clarified
                result = self._move(tag, task, counted=False)
        elif tag == "interaction":
            kinds = cell_kinds(self._identified, getattr(self._env, "ledge_patterns", None),
                               getattr(self._env, "walk_patterns", None), self._people_now(),
                               getattr(self._env, "left_ledge_patterns", None),
                               self._local_blocked_cells())
            result = self.call_executor(tag, task, identified=self._identified,
                                        cell_kinds=kinds, people_fn=self._people_now)
        elif tag == "menu":
            task, is_start_menu = self._menu_objective(task, frame)
            result = self.call_executor(tag, task, is_start_menu=is_start_menu)
        elif tag == "combat":
            task = argument or self._battle_instructions(frame)
            result = self._combat(task)
        elif tag == "free":
            task, termination = split_free_argument(task)
            result = self.call_executor(tag, task, termination=termination)
        else:
            result = self.call_executor(tag, task)
        return result, task

    def _move(self, tag: str, task: str, *, counted: bool = True) -> Any:
        kinds = cell_kinds(self._identified, getattr(self._env, "ledge_patterns", None),
                           getattr(self._env, "walk_patterns", None), self._people_now(),
                           getattr(self._env, "left_ledge_patterns", None),
                           self._local_blocked_cells())
        return self.call_executor(tag, task, counted=counted, screen_tiles=self._screen_tiles,
                                  cell_kinds=kinds,
                                  identified=self._identified,
                                  people_fn=self._people_now)

    @staticmethod
    def _merge_handled(earlier: Optional[Dict[str, Any]], later: Dict[str, Any]) -> Dict[str, Any]:
        if earlier is None:
            return later
        combined_text = "\n".join(filter(None, [earlier.get("dialogue"), later.get("dialogue")]))
        if combined_text:
            earlier["dialogue"] = combined_text
            later["dialogue"] = combined_text
        earlier["report"].final_state = later["report"].final_state
        earlier["succeeded"] = bool(later.get("succeeded"))
        earlier["code"] = later.get("code")
        earlier["reason"] = later.get("reason")
        if later.get("notes"):
            earlier["notes"] = "\n".join(filter(None, [earlier.get("notes"), later.get("notes")]))
        return earlier

    def _handle_interruption(self, *, dialogue_task: str = "clear the dialogue that interrupted what the player was doing",
                             initial_dialogue_crops: Optional[List[Any]] = None,
                             initial_battle_instructions: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
        states_seen: List[str] = []
        combined: Optional[Dict[str, Any]] = None
        pending_crops = list(initial_dialogue_crops) if initial_dialogue_crops else None
        pre_battle_dialogue: Optional[str] = None
        active_battle_instructions: Optional[str] = initial_battle_instructions
        for step_idx in range(self.MAX_INTERRUPTION_CHAIN):
            if self._env_done() or self._battles_lost() or self._total_calls >= self._max_total_calls:
                break
            if self._agent_state() == AgentState.IN_MENU:
                self._settle_menu_state()
            frame = self._current_frame()
            state = self._effective_state(frame)
            if state == AgentState.FREE_ROAM and combined is not None:
                break
            state_label = state.name.lower()
            if not states_seen or states_seen[-1] != state_label:
                states_seen.append(state_label)
            if state == AgentState.IN_DIALOGUE:
                task = dialogue_task if step_idx == 0 else "clear the dialogue that followed"
                handled = self.call_executor("dialogue", task, counted=False,
                                             initial_dialogue_crops=pending_crops)
                diag_crops = getattr(self.last_executor, "dialogue_crops", None)
                pending_crops = list(diag_crops) if diag_crops else None
                if handled.get("dialogue"):
                    pre_battle_dialogue = (
                        f"{pre_battle_dialogue}\n{handled['dialogue']}"
                        if pre_battle_dialogue
                        else str(handled["dialogue"])
                    )
            elif state == AgentState.IN_BATTLE:
                is_wild_flee = (
                    self._is_wild_encounter_dialogue(pre_battle_dialogue)
                    and not self._task_wants_wild_battles()
                    and self._active_pokemon_hp_healthy(frame)
                )
                instructions = (
                    self.WILD_FLEE_INSTRUCTION
                    if is_wild_flee
                    else self._battle_instructions(frame)
                )
                active_battle_instructions = instructions
                handled = self._combat(
                    instructions,
                    counted=False,
                    initial_dialogue_crops=pending_crops,
                    skip_intro_ocr=is_wild_flee,
                )
                pending_crops = None
            elif state == AgentState.IN_MENU:
                from_battle_menu = bool(
                    active_battle_instructions
                    and (combined is None or combined.get("reason") == "left_battle_to_menu")
                )
                if (
                    from_battle_menu
                    and (is_pack_screen(frame) or is_party_menu_screen(frame))
                    and not is_overworld_pack_menu(frame)
                ):
                    handled = self._combat(
                        active_battle_instructions,
                        counted=False,
                        initial_dialogue_crops=pending_crops,
                    )
                else:
                    if from_battle_menu:
                        hp_line = self._party_hp_block()
                        hp_context = f" ({hp_line})" if hp_line else ""
                        menu_instruction = (
                            f"complete the in-battle menu action for the ongoing battle "
                            f"(active battle instructions: \"{active_battle_instructions}\"){hp_context}"
                        )
                    else:
                        menu_instruction = "deal with the menu that opened while the player was doing something else"
                    objective, is_start_menu = self._menu_objective(menu_instruction, frame)
                    handled = self.call_executor("menu", objective, is_start_menu=is_start_menu, counted=False)
                pending_crops = None
            else:
                handled = self.call_executor("free", self.CLEAR_EVENT_INSTRUCTION,
                                             termination=self.CLEAR_EVENT_TERMINATION, counted=False)
                pending_crops = None
            combined = self._merge_handled(combined, handled)
            if self._env_done() or self._battles_lost():
                break
            if self._agent_state() == AgentState.FREE_ROAM:
                step_before_settle = getattr(getattr(self._env, "_emulator", None), "step_count", None)
                self._settle_free_roam()
                step_after_settle = getattr(getattr(self._env, "_emulator", None), "step_count", None)
                if step_after_settle != step_before_settle and combined is not None:
                    combined["report"].final_state = self._env.get_info()
            next_frame = self._current_frame()
            next_state = self._effective_state(next_frame)
            if next_state == AgentState.FREE_ROAM:
                break
            if not handled.get("succeeded") and next_state == state:
                break
        if combined is None:
            frame = self._current_frame()
            state = self._effective_state(frame)
            handled = self.call_executor("free", self.CLEAR_EVENT_INSTRUCTION,
                                         termination=self.CLEAR_EVENT_TERMINATION, counted=False)
            return state.name.lower(), handled
        return " -> ".join(states_seen), combined

    def _note(self, entry: str) -> None:
        self.report.narrative.append(entry)

    def _out_of_calls(self) -> bool:
        return self._calls >= self._max_steps or self._total_calls >= self._max_total_calls

    def _env_done(self) -> bool:
        return bool(self._env._emulator.check_if_done())

    def _step_wait_action(self):
        before = self._current_frame()
        self._env.step_high_level_action(WaitAction)
        return before, self._current_frame()

    def _settle_free_roam(
        self,
        ticks: int = 2,
        max_cutscene_ticks: int = 30,
        post_motion_ticks: int = 5,
        post_warp_banner_ticks: int = 6,
    ) -> None:
        env_settled = getattr(self._env, "_last_settled_step", None)
        if env_settled is not None and (self._last_settled_step is None or env_settled > self._last_settled_step):
            self._last_settled_step = env_settled
        if self._last_settled_step is not None and env_settled is None:
            try:
                self._env._last_settled_step = self._last_settled_step
            except Exception:  # noqa: BLE001
                pass
        settle_overworld_free_roam(
            env=self._env,
            get_state_fn=self._agent_state,
            get_frame_fn=self._current_frame,
            is_done_fn=self._env_done,
            step_wait_fn=self._step_wait_action,
            settle_wait=ticks,
            max_cutscene_wait=max_cutscene_ticks,
            post_motion_wait=post_motion_ticks,
            post_warp_banner_wait=post_warp_banner_ticks,
            include_top_box=True,
            record_dialogue_settled=False,
            respect_last_settled=True,
            settle_menu_fn=lambda: self._settle_menu_state(
                max_cutscene_ticks=max_cutscene_ticks,
                post_motion_ticks=post_motion_ticks,
            ),
        )
        self._last_settled_step = getattr(self._env, "_last_settled_step", self._last_settled_step)

    def _settle_menu_state(self, ticks: int = 1, max_cutscene_ticks: int = 24, post_motion_ticks: int = 4) -> AgentState:
        return settle_overworld_menu_state(
            get_state_fn=self._agent_state,
            get_frame_fn=self._current_frame,
            is_done_fn=self._env_done,
            step_wait_fn=self._step_wait_action,
            min_wait=ticks,
            settle_wait=ticks,
            max_cutscene_wait=max_cutscene_ticks,
            post_motion_wait=post_motion_ticks,
        )

    @staticmethod
    def _dialogue_snippet(handled: Dict[str, Any], limit: int = 160) -> str:
        text = " ".join((handled.get("dialogue") or "").split())
        return f": \"{text[:limit]}\"" if text else ""

    def _clear_pending_dialogue(self, attempts: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        if not self._is_in_dialogue() or self._out_of_calls() or self._env_done():
            return None
        state_chain, handled = self._handle_interruption(dialogue_task="clear the dialogue on screen")
        label = "open dialogue" if state_chain == "in_dialogue" else f"open dialogue ({state_chain})"
        note = f"Cleared {label} ({handled['reason']}){self._dialogue_snippet(handled)}"
        self._note(note)
        if attempts is not None:
            attempts.append(note)
        return handled

    def _clear_pending_interruption(self, attempts: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        if self._out_of_calls() or self._env_done():
            return None
        if not (self._is_in_dialogue() or self._agent_state() != AgentState.FREE_ROAM):
            return None
        frame = self._current_frame()
        self._perceive(frame)
        if self._is_in_dialogue(frame):
            return self._clear_pending_dialogue(attempts)
        state_chain, handled = self._handle_interruption()
        note = f"Cleared pending {state_chain} ({handled['reason']}){self._dialogue_snippet(handled)}"
        self._note(note)
        if attempts is not None:
            attempts.append(note)
        return handled

    def _made_new_subgoal_progress(
        self,
        before_place: Optional[Tuple[int, Tuple[int, int]]],
        after_place: Optional[Tuple[int, Tuple[int, int]]],
        subgoal_places: List[Tuple[int, Tuple[int, int]]],
    ) -> bool:
        gmap = self._global_map()
        if gmap is None or before_place is None or after_place is None:
            return False
        b_seg, b_pos = gmap.resolve(before_place[0], before_place[1])
        a_seg, a_pos = gmap.resolve(after_place[0], after_place[1])
        if b_seg != a_seg:
            return True
        if a_pos == b_pos:
            return False
        visited = {gmap.resolve(seg, pos) for seg, pos in subgoal_places}
        return (a_seg, a_pos) not in visited

    def _pursue(self, subgoal: str, subgoal_index: int = 0) -> Tuple[str, List[str], int]:
        attempts: List[str] = []
        subgoal_places: List[Tuple[int, Tuple[int, int]]] = []
        start_subgoal_place = self._current_map_place()
        if start_subgoal_place is not None:
            subgoal_places.append(start_subgoal_place)
        failed_attempts = 0
        iterations = 0
        while failed_attempts < self.MAX_SUBGOAL_ATTEMPTS and iterations < self.MAX_SUBGOAL_ITERATIONS:
            if self._env_done():
                return "env_done", attempts, subgoal_index
            if self._out_of_calls():
                return "out_of_calls", attempts, subgoal_index
            if self._total_calls > 0:
                self._settle_free_roam()
            if self._is_in_dialogue() or self._agent_state() != AgentState.FREE_ROAM:
                before = self._current_frame()
                self._perceive(before)
                if self._is_in_dialogue():
                    handled = self._clear_pending_dialogue(attempts)
                    request_label = "Dialogue(clear open dialogue)"
                else:
                    state_chain, handled = self._handle_interruption()
                    note = f"Cleared pending {state_chain} ({handled['reason']}){self._dialogue_snippet(handled)}"
                    self._note(note)
                    attempts.append(note)
                    request_label = f"Interrupt({state_chain})"
                if self._battles_lost():
                    self._note("the player lost a battle and blacked out")
                    return "lost_battle", attempts, subgoal_index
                if self._env_done():
                    return "env_done", attempts, subgoal_index
                if handled and handled.get("succeeded"):
                    complete, summary, reached, replan_needed = self._judge(
                        subgoal, request_label, handled, subgoal_index)
                    self._note(summary)
                    attempts.append(f"Dialogue(): {summary}")
                    if complete:
                        return "complete", attempts, reached
                    if replan_needed:
                        return "stuck", attempts, subgoal_index
                if self._out_of_calls():
                    return "out_of_calls", attempts, subgoal_index
            iterations += 1
            frame = self._current_frame()
            self._perceive(frame)
            before_place = self._current_map_place()
            if before_place is not None and (not subgoal_places or subgoal_places[-1] != before_place):
                subgoal_places.append(before_place)
            prior_stops = list(self._leg_stops)
            decision, decision_reasoning = self._decide(subgoal, attempts, frame)
            if decision is None:
                attempts.append("the reply could not be read, so nothing was done")
                failed_attempts += 1
                continue
            action, argument = decision
            is_move_action = ACTIONS[action][0] in MOVE_TAGS
            leg_initial_state = None
            result, task = self._run_action(action, argument, frame)
            if "report" in result and getattr(result["report"], "initial_state", None) is not None:
                leg_initial_state = result["report"].initial_state
            after_place = self._record_map_stop()
            move_tag = self._format_leg_movement(before_place, after_place, prior_stops, is_move_action=is_move_action)
            request = f"{action}({task})"
            exec_blocked = self._record_leg_blocked_moves(request)
            if self._battles_lost():
                note = f"{request}: the player lost a battle and blacked out"
                self._note(note)
                attempts.append(note)
                return "lost_battle", attempts, subgoal_index

            if self._env_done():
                note = f"{request} used up the last of the environment's steps"
                self._note(note)
                attempts.append(note)
                return "env_done", attempts, subgoal_index

            wild_resumes = 0
            leg_start_place = before_place
            current_task = task
            while (
                is_move_action
                and not result["succeeded"]
                and (self._agent_state() != AgentState.FREE_ROAM or self._is_in_dialogue())
                and wild_resumes < self.MAX_WILD_MOVE_RESUMES
                and not self._env_done()
                and not self._battles_lost()
                and self._total_calls < self._max_total_calls
            ):
                if self._calls > 0:
                    self._calls -= 1
                seed_crops = getattr(self.last_executor, "dialogue_crops", None)
                state, handled = self._handle_interruption(initial_dialogue_crops=seed_crops)
                if leg_initial_state is not None:
                    handled["report"].initial_state = leg_initial_state
                after_place = self._record_map_stop()
                move_tag = self._format_leg_movement(
                    before_place, after_place, prior_stops, is_move_action=is_move_action)
                note = (f"{request} was interrupted while the game was in {state}; "
                        f"dealt with it ({handled['reason']}){self._dialogue_snippet(handled)}")
                if move_tag:
                    note = f"{move_tag} {note}"
                self._note(note)
                attempts.append(note)
                if self._battles_lost():
                    self._note("the player lost a battle and blacked out")
                    return "lost_battle", attempts, subgoal_index
                if self._env_done():
                    return "env_done", attempts, subgoal_index
                if after_place is not None and (not subgoal_places or subgoal_places[-1] != after_place):
                    subgoal_places.append(after_place)
                can_resume, resumed_task, already_completed = self._prepare_wild_move_resume(
                    action, current_task, leg_start_place, after_place, state, handled
                )
                if already_completed:
                    handled["succeeded"] = True
                    result = handled
                    break
                if not can_resume or resumed_task is None:
                    result = {"_interruption_handled": (state, handled)}
                    break
                wild_resumes += 1
                leg_start_place = after_place
                current_task = resumed_task
                resume_frame = self._current_frame()
                self._perceive(resume_frame)
                move_tag_name = ACTIONS[action][0]
                result = self._move(move_tag_name, current_task, counted=True)
                if leg_initial_state is not None and "report" in result:
                    result["report"].initial_state = leg_initial_state
                if handled.get("dialogue"):
                    combined_diag = "\n".join(filter(None, [handled.get("dialogue"), result.get("dialogue")]))
                    if combined_diag:
                        result["dialogue"] = combined_diag
                after_place = self._record_map_stop()
                move_tag = self._format_leg_movement(
                    before_place, after_place, prior_stops, is_move_action=is_move_action)
                exec_blocked.update(self._record_leg_blocked_moves(request))
                if self._battles_lost():
                    note = f"{request}: the player lost a battle and blacked out"
                    self._note(note)
                    attempts.append(note)
                    return "lost_battle", attempts, subgoal_index
                if self._env_done():
                    note = f"{request} used up the last of the environment's steps"
                    self._note(note)
                    attempts.append(note)
                    return "env_done", attempts, subgoal_index

            if "_interruption_handled" in result:
                state, handled = result["_interruption_handled"]
                if handled.get("succeeded"):
                    complete, summary, reached, replan_needed = self._judge(
                        subgoal, f"{request} (cleared {state})", handled, subgoal_index,
                        move_tag=move_tag, decision_reasoning=decision_reasoning)
                    if move_tag and move_tag not in summary:
                        summary = f"{move_tag} {summary}"
                    self._note(summary)
                    attempts.append(f"{request}: {summary}")
                    if complete:
                        return "complete", attempts, reached
                    if replan_needed:
                        return "stuck", attempts, subgoal_index
                if not is_move_action or not handled.get("succeeded"):
                    failed_attempts += 1
                continue

            if result["succeeded"] and self._agent_state() == AgentState.FREE_ROAM and not self._env_done():
                step_before_settle = getattr(getattr(self._env, "_emulator", None), "step_count", None)
                self._settle_free_roam()
                step_after_settle = getattr(getattr(self._env, "_emulator", None), "step_count", None)
                if step_after_settle != step_before_settle:
                    result["report"].final_state = self._env.get_info()
                    after_place = self._record_map_stop()
                    move_tag = self._format_leg_movement(
                        before_place, after_place, prior_stops, is_move_action=is_move_action)
                if self._env_done():
                    note = f"{request} used up the last of the environment's steps"
                    self._note(note)
                    attempts.append(note)
                    return "env_done", attempts, subgoal_index

            if (result["succeeded"]
                    and (self._is_in_dialogue() or self._agent_state() != AgentState.FREE_ROAM)
                    and not self._env_done()
                    and self._total_calls < self._max_total_calls):
                _, follow_up = self._handle_interruption(
                    dialogue_task="clear the dialogue that followed the action")
                combined_text = "\n".join(filter(None, [result.get("dialogue"), follow_up.get("dialogue")]))
                if combined_text:
                    result["dialogue"] = combined_text
                result["report"].final_state = follow_up["report"].final_state
                if result["tag"] == "menu":
                    follow_up["report"].initial_state = result["report"].initial_state
                    combined_notes = "\n".join(filter(None, [result.get("notes"), follow_up.get("notes")]))
                    if combined_notes:
                        follow_up["notes"] = combined_notes
                        follow_up["report"].notes = combined_notes
                    if combined_text:
                        follow_up["dialogue"] = combined_text
                    result = follow_up
                if self._battles_lost():
                    note = f"{request}: the player lost a battle and blacked out"
                    self._note(note)
                    attempts.append(note)
                    return "lost_battle", attempts, subgoal_index
                if self._env_done():
                    note = f"{request} used up the last of the environment's steps"
                    self._note(note)
                    attempts.append(note)
                    return "env_done", attempts, subgoal_index
                after_place = self._record_map_stop()
                move_tag = self._format_leg_movement(
                    before_place, after_place, prior_stops, is_move_action=is_move_action)

            if not result["succeeded"] and (self._agent_state() != AgentState.FREE_ROAM or self._is_in_dialogue()):
                if is_move_action and self._calls > 0:
                    self._calls -= 1
                seed_crops = getattr(self.last_executor, "dialogue_crops", None) if is_move_action else None
                seed_battle_instructions = task if (action == "Battle" and result.get("reason") == "left_battle_to_menu") else None
                state, handled = self._handle_interruption(
                    initial_dialogue_crops=seed_crops,
                    initial_battle_instructions=seed_battle_instructions,
                )
                if leg_initial_state is not None:
                    handled["report"].initial_state = leg_initial_state
                after_place = self._record_map_stop()
                move_tag = self._format_leg_movement(
                    before_place, after_place, prior_stops, is_move_action=is_move_action)
                note = (f"{request} was interrupted while the game was in {state}; "
                        f"dealt with it ({handled['reason']}){self._dialogue_snippet(handled)}")
                if move_tag:
                    note = f"{move_tag} {note}"
                self._note(note)
                attempts.append(note)
                if self._battles_lost():
                    self._note("the player lost a battle and blacked out")
                    return "lost_battle", attempts, subgoal_index
                if self._env_done():
                    return "env_done", attempts, subgoal_index
                if handled.get("succeeded"):
                    complete, summary, reached, replan_needed = self._judge(
                        subgoal, f"{request} (cleared {state})", handled, subgoal_index,
                        move_tag=move_tag, decision_reasoning=decision_reasoning)
                    if move_tag and move_tag not in summary:
                        summary = f"{move_tag} {summary}"
                    self._note(summary)
                    attempts.append(f"{request}: {summary}")
                    if complete:
                        return "complete", attempts, reached
                    if replan_needed:
                        return "stuck", attempts, subgoal_index
                if after_place is not None and (not subgoal_places or subgoal_places[-1] != after_place):
                    subgoal_places.append(after_place)
                if not is_move_action or not handled.get("succeeded"):
                    failed_attempts += 1
                continue

            if not result["succeeded"]:
                if is_move_action and not result["report"].steps and self._calls > 0:
                    self._calls -= 1
                if result["reason"] in ("move_blocked", "out_of_steps"):
                    if not exec_blocked:
                        move_dir = None
                        if result.get("notes"):
                            m_dir = re.search(
                                r"(?:Blocked|Stopped|Could not finish).*moving\s+(up|down|left|right)",
                                str(result.get("notes")),
                                re.IGNORECASE,
                            )
                            if m_dir:
                                move_dir = m_dir.group(1).lower()
                        if move_dir is None and action == "MoveOff":
                            move_dir = parse_edge_direction(task)
                        if move_dir is None and action == "Free":
                            m_dir = re.search(r"\b(up|down|left|right|north|south|west|east)\b", task, re.IGNORECASE)
                            if m_dir:
                                aliases = {"north": "up", "south": "down", "west": "left", "east": "right"}
                                move_dir = aliases.get(m_dir.group(1).lower(), m_dir.group(1).lower())
                        if move_dir:
                            recorder = getattr(self._env, "record_blocked_step", None)
                            if callable(recorder):
                                try:
                                    recorder(direction=move_dir)
                                except Exception:  # noqa: BLE001
                                    pass
                            detail = f"{request} -> {result['reason']}"
                            target_places = [after_place] if after_place is not None else [before_place]
                            for place_key in (p for p in target_places if p is not None):
                                dir_list = self._blocked_moves.setdefault(place_key, {}).setdefault(move_dir, [])
                                if detail not in dir_list:
                                    dir_list.append(detail)
                note = f"{request} did not work: {result['reason']}" + (
                    f" ({result['notes']})" if result.get("notes") else "")
                if move_tag:
                    note = f"{move_tag} {note}"
                self._note(note)
                attempts.append(note)
                failed_attempts += 1
                continue

            complete, summary, reached, replan_needed = self._judge(
                subgoal, request, result, subgoal_index, move_tag=move_tag,
                decision_reasoning=decision_reasoning)
            if move_tag and move_tag not in summary:
                summary = f"{move_tag} {summary}"
            self._note(summary)
            attempts.append(f"{request}: {summary}")
            if complete:
                return "complete", attempts, reached
            if replan_needed:
                return "stuck", attempts, subgoal_index
            made_progress = is_move_action and self._made_new_subgoal_progress(before_place, after_place, subgoal_places)
            if after_place is not None and (not subgoal_places or subgoal_places[-1] != after_place):
                subgoal_places.append(after_place)
            if not made_progress:
                failed_attempts += 1
        return "stuck", attempts, subgoal_index

    def _digest(self, status: str, reason: Optional[str], summary: Optional[str]) -> str:
        lines = [f"Task: {self._task}"]
        if self._guidance:
            lines.append(f"Guidance: {self._guidance}")
        lines.append(f"Outcome: {status}" + (f" ({reason})" if reason else ""))
        if summary:
            lines.append(f"Final check: {summary}")
        if self.plan:
            lines.append("Last plan:")
            lines += [f"{n + 1}. {subgoal}" for n, subgoal in enumerate(self.plan)]
        lines.append("What happened, in order:")
        lines += [f"- {entry}" for entry in self.report.narrative] or ["- (nothing was recorded)"]
        texts = [report for report in self.report.executor_reports if report.notes and report.notes.strip()]
        if texts:
            lines.append("Text seen along the way, in order:")
            lines += [f"- while doing \"{report.task}\": {report.notes.strip()}" for report in texts]
        return "\n".join(lines)

    def _out_of_calls_reason(self) -> str:
        if self._calls >= self._max_steps:
            return f"Used all {self._max_steps} executor calls."
        return f"Used all {self._max_total_calls} total executor calls."

    def _finish(self, status: str, reason: Optional[str] = None, summary: Optional[str] = None) -> dict:
        self.report.digest = self._digest(status, reason, summary)
        return {
            "status": status,
            "reason": reason,
            "summary": summary,
            "plan": list(self.plan),
            "narrative": list(self.report.narrative),
            "executor_calls": self._calls,
            "total_executor_calls": self._total_calls,
        }

    def _evaluate(self) -> Optional[dict]:
        self._settle_free_roam()
        self._clear_pending_interruption()
        if self._battles_lost():
            return self._finish("lost_battle", "The player lost a battle and blacked out.")
        frame = self._current_frame()
        self._perceive(frame)
        self.plan = self._plan_subgoals(frame)
        if not self.plan:
            return self._finish("no_plan", "The planner did not produce any subgoals.")

        replans = 0
        continuation_replans = 0
        calls_at_plan_start = self._calls
        all_subgoals_judged = True
        index = 0
        while index < len(self.plan):
            subgoal = self.plan[index]
            outcome, attempts, reached_index = self._pursue(subgoal, index)

            if outcome == "complete":
                if reached_index > index:
                    all_subgoals_judged = False
                index = max(index, reached_index) + 1
                if index < len(self.plan):
                    if not self._out_of_calls() and not self._env_done():
                        continue
                    self._settle_free_roam()
                    frame = self._current_frame()
                    self._perceive(frame)
                    complete, summary = self._task_check(frame, can_continue=False)
                    if complete:
                        return self._finish("completed", None, summary)
                    if self._env_done():
                        return self._finish("env_done",
                                            "The environment ran out of steps before the plan was finished.",
                                            summary)
                    return self._finish("out_of_calls", self._out_of_calls_reason(), summary)
                self._settle_free_roam()
                self._clear_pending_interruption()
                if self._battles_lost():
                    return self._finish("lost_battle", "The player lost a battle and blacked out.")
                frame = self._current_frame()
                self._perceive(frame)
                can_continue = (
                    continuation_replans < self.MAX_CONTINUATION_REPLANS
                    and not self._out_of_calls()
                    and not self._env_done()
                    and self._calls > calls_at_plan_start
                )
                local_kinds = self._current_local_kinds()
                unfinished_note = self._continuation_unfinished_note(local_kinds, can_continue=can_continue)
                if (
                    unfinished_note is None
                    and (len(self.plan) == 1 or all_subgoals_judged)
                    and self._effective_state(frame) == AgentState.FREE_ROAM
                    and not self._is_in_dialogue(frame)
                    and self.report.narrative
                ):
                    judge_summary = re.sub(
                        r"^\[Area coordinate:[^\]]*\]\s*",
                        "",
                        self.report.narrative[-1],
                    ).strip() or self.report.narrative[-1]
                    return self._finish("completed", None, judge_summary)
                complete, summary = self._task_check(frame, can_continue=can_continue)
                if complete:
                    return self._finish("completed", None, summary)
                if not can_continue:
                    return self._finish("plan_done_task_incomplete",
                                        "Worked through the plan without finishing the task.",
                                        summary)
                self._note(f"The plan was finished but the task is not done: {summary}")
                calls_at_plan_start = self._calls
                self.plan = self._plan_subgoals(frame)
                continuation_replans += 1
                all_subgoals_judged = True
                if not self.plan:
                    return self._finish("plan_done_task_incomplete",
                                        "Worked through the plan without finishing the task.",
                                        summary)
                index = 0
                continue

            if outcome == "env_done":
                frame = self._current_frame()
                self._perceive(frame)
                complete, summary = self._task_check(frame, can_continue=False)
                return self._finish("completed" if complete else "env_done",
                                    None if complete else "The environment ran out of steps "
                                                          "before the plan was finished.",
                                    summary)

            if outcome == "out_of_calls":
                summary = None
                if self.report.narrative and not self._env_done():
                    self._settle_free_roam()
                    frame = self._current_frame()
                    if self._agent_state() == AgentState.FREE_ROAM and not self._is_in_dialogue(frame):
                        self._perceive(frame)
                        complete, summary = self._task_check(frame, can_continue=False)
                        if complete:
                            return self._finish("completed", None, summary)
                return self._finish("out_of_calls", self._out_of_calls_reason(), summary)

            if outcome == "lost_battle":
                return self._finish("lost_battle", "The player lost a battle and blacked out.")

            if replans >= self.MAX_REPLANS:
                return self._finish("stuck", f"Could not achieve: {subgoal}")
            frame = self._current_frame()
            self._perceive(frame)
            decision, reason, subgoals = self._reconsider(subgoal, attempts, frame)
            replans += 1
            if decision == "quit":
                return self._finish("quit", reason)
            calls_at_plan_start = self._calls
            self.plan = subgoals
            index = 0

        return self._finish("no_plan", "The plan became empty while working through it.")
