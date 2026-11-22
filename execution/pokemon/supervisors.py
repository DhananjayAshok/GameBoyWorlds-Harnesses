from collections import deque
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from execution.game_api import AgentState, WaitAction

from execution.executors import Executor
from execution.perception.pokemon.battle import (has_bottom_dialogue_box, has_menu_cursor,
                                                 has_modal_preview_box, has_top_dialogue_box,
                                                 is_cutscene_frame_changing)
from execution.perception.pokemon.navigation import (DIRECTIONS, LEDGE, LEDGE_LEFT, WALKABLE, X_RANGE, Y_RANGE,
                                                     cell_kinds, compress_path, edge_cells,
                                                     format_path_runs, shortest_path, verbalize_edges)
from execution.perception.pokemon.tiles import TileRecognizer, describe_state_screen, verbalize_tiles
from execution.pokemon.executors import (CombatExecutor, DialogueExecutor, FreeExecutor,
                                         InteractionExecutor, MenuExecutor,
                                         BATTLE_ENDED, BATTLE_LEFT_TO_DIALOGUE, BATTLE_LEFT_TO_MENU,
                                         DIALOGUE_CLEARED, DIALOGUE_LEFT_TO_BATTLE, DIALOGUE_LEFT_TO_MENU,
                                         FREE_OK, LOCATION_FAIL_CODES,
                                         MENU_LEFT_TO_BATTLE, MENU_LEFT_TO_DIALOGUE, MENU_OK,
                                         MOVE_OK, MOVE_TARGET_AMBIGUOUS,
                                         OffScreenMoveExecutor, OnScreenMoveExecutor,
                                         effective_agent_state)
from execution.pokemon.prompts import (PLAYTHROUGH_ACHIEVABLE_PROMPT, PLAYTHROUGH_AFTER_BLOCK,
                                       PLAYTHROUGH_AREA_MAP_BLOCK, PLAYTHROUGH_ATTEMPTS_BLOCK,
                                       PLAYTHROUGH_BATTLE_INSTRUCTIONS_PROMPT, PLAYTHROUGH_BATTLE_TIPS_BLOCK,
                                       PLAYTHROUGH_CLARIFY_TARGET_PROMPT, PLAYTHROUGH_GUIDANCE_BLOCK,
                                       PLAYTHROUGH_JUDGE_PROMPT, PLAYTHROUGH_LOG_BLOCK,
                                       PLAYTHROUGH_MENU_OBJECTIVE_PROMPT, PLAYTHROUGH_PLAN_PROMPT,
                                       PLAYTHROUGH_REMAINING_SUBGOALS_BLOCK, PLAYTHROUGH_REMAINING_SUBGOALS_LINE,
                                       PLAYTHROUGH_SUBGOAL_DECISION_PROMPT, PLAYTHROUGH_TASK_CHECK_PROMPT,
                                       PLAYTHROUGH_TEXT_BLOCK, PRIMITIVES)
from execution.strategist.pokemon.global_map.model import UNKNOWN, is_door
from execution.strategist.pokemon.global_map.text import LEGEND, ascii_window, frontier_targets
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
    "menu": 15,
    "combat": 60,
    "dialogue": 10,
    "free": 15,
}
ACTION_PATTERN = re.compile(r"^\s*`?(" + "|".join(ACTIONS) + r")\s*\((.*)\)\s*`?\s*\.?\s*$", re.IGNORECASE | re.DOTALL)


FREE_TERMINATION_PATTERN = re.compile(r"[;,.]?\s*stop\s+when\s*:?\s*", re.IGNORECASE)


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
    MAX_TOTAL_EXECUTOR_CALLS = 30
    MAX_INTERRUPTION_CHAIN = 6

    def __init__(self, task: str, env, game: str, max_steps: int,
                 tile_recognizer: TileRecognizer, episode_number: int, guidance: Optional[str] = None,
                 battle_tips: Optional[List[str]] = None,
                 screen_people: Optional[Callable[[], Dict[Tuple[int, int], str]]] = None, **kwargs: Any) -> None:
        self._screen_people = screen_people
        self._current_tag: Optional[str] = None
        self._guidance = guidance
        self._battle_tips = list(battle_tips or [])
        self._losses_at_start = env.get_info()["pokemon_core"].get("n_battles_lost", 0)
        self._tile_recognizer = tile_recognizer
        self._episode_number = episode_number
        self._screen_tiles: Optional[str] = None
        self._identified = None
        self._calls = 0
        self._total_calls = 0
        self._max_total_calls = max(self.MAX_TOTAL_EXECUTOR_CALLS, max_steps * 3)
        self._start_place: Optional[Tuple[int, Tuple[int, int]]] = None
        self._leg_stops: List[Tuple[int, Tuple[int, int]]] = []
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

    def _edge_pocket_note(self, direction: str, local_edge_cells: List[Tuple[int, int]]) -> Optional[str]:
        gmap = self._global_map()
        place = self._current_map_place()
        if gmap is None or place is None or not local_edge_cells:
            return None
        seg = gmap.current
        px, py = place[1]
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
        for lx, ly in local_edge_cells:
            eg = (px + lx, py + ly)
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

        while queue:
            cur = queue.popleft()
            if strictly_beyond(cur):
                beyond_cells.add(cur)
            c = seg.get(cur)
            if c is not None and is_door(c):
                return None
            for d, (nx_d, ny_d) in DIRECTIONS.items():
                nxt = (cur[0] + nx_d, cur[1] + ny_d)
                if in_screen(nxt):
                    if nxt not in seeds:
                        continue
                if not in_half_plane(nxt):
                    nc = seg.get(nxt)
                    if (
                        nc is None
                        or nc.kind in (WALKABLE, UNKNOWN)
                        or (nc.kind == LEDGE and d == "down")
                        or (nc.kind == LEDGE_LEFT and d == "left")
                    ):
                        return None
                    continue
                nc = seg.get(nxt)
                if nc is None or nc.kind == UNKNOWN or is_door(nc):
                    return None
                if nc.kind == WALKABLE:
                    landing = nxt
                elif (nc.kind == LEDGE and d == "down") or (nc.kind == LEDGE_LEFT and d == "left"):
                    landing = (nxt[0] + nx_d, nxt[1] + ny_d)
                    if in_screen(landing) and landing not in seeds:
                        continue
                    if not in_half_plane(landing):
                        return None
                    lc = seg.get(landing)
                    if lc is None or lc.kind == UNKNOWN or is_door(lc):
                        return None
                    if lc.kind != WALKABLE:
                        continue
                else:
                    continue
                if landing not in visited:
                    visited.add(landing)
                    queue.append(landing)

        xs = [c[0] - px for c in visited]
        ys = [c[1] - py for c in visited]
        return (f" [mapped closed dead-end pocket beyond {direction} edge: "
                f"{len(beyond_cells)} off-screen walkable tile(s), relative x in [{min(xs)}, {max(xs)}], "
                f"y in [{min(ys)}, {max(ys)}], no unexplored tiles or doors]")

    def _describe_edges(self, kinds: Dict[Tuple[int, int], str]) -> str:
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
        if self._start_place is not None:
            s_seg, s_pos = gmap.resolve(self._start_place[0], self._start_place[1])
            if s_seg == cur_seg:
                dx, dy = cur_pos[0] - s_pos[0], cur_pos[1] - s_pos[1]
                lines.append(
                    f"Current area coordinate: {cur_pos} (episode start: {s_pos}; "
                    f"net displacement: ({dx:+d}, {dy:+d}), where +x is right/east and +y is up/north)."
                )
            else:
                lines.append(f"Current area coordinate: {cur_pos} (entered a new area since episode start).")
        else:
            lines.append(f"Current area coordinate: {cur_pos} (+x is right/east, +y is up/north).")
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
        lines.append(
            "Map of what has been seen in this area around the player "
            "(@ is the player at (0, 0) on the current screen, north is up, one character per tile):"
        )
        lines.append(ascii_window(gmap))
        lines.append("Legend: " + LEGEND)
        targets, _ = frontier_targets(gmap, local_kinds)
        if targets:
            lines.append("Nearest unexplored map edges in this area (x = steps right, y = steps up from the player):")
            for t in targets[:6]:
                where = "on screen, reachable" if t["on_screen"] else "off screen"
                lines.append(f"- ({t['dx']}, {t['dy']}): {t['steps']} steps {t['sector']} ({where})")
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
        self._identified = self._tile_recognizer.identify_tiles(frame)
        people = self._people_now()
        kinds = cell_kinds(self._identified, getattr(self._env, "ledge_patterns", None),
                           getattr(self._env, "walk_patterns", None), people,
                           getattr(self._env, "left_ledge_patterns", None))
        self._screen_tiles = (verbalize_tiles(self._identified, people)
                              + "\nWalkable cells on each edge of the screen:\n"
                              + self._describe_edges(kinds))
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

    def _fill_prompt(self, prompt: str) -> str:
        if self._screen_tiles is None:
            log_error("Supervisor prompt built before the screen was perceived.", self._parameters)
        guidance_block = PLAYTHROUGH_GUIDANCE_BLOCK.replace("[GUIDANCE]", self._guidance) if self._guidance else ""
        plan = "\n".join(f"{n + 1}. {subgoal}" for n, subgoal in enumerate(self.plan))
        raw_state = self._effective_state()
        state = raw_state.name.lower().replace("_", " ")
        local_kinds = None
        if raw_state == AgentState.FREE_ROAM and self._identified:
            local_kinds = cell_kinds(self._identified, getattr(self._env, "ledge_patterns", None),
                                     getattr(self._env, "walk_patterns", None), self._people_now(),
                                     getattr(self._env, "left_ledge_patterns", None))
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

    def _process_combat(self, report: ExecutorReport) -> Dict[str, Any]:
        return self._result("combat", report, report.outcome == BATTLE_ENDED,
                            dialogue=report.notes,
                            left_battle=report.outcome in (BATTLE_LEFT_TO_DIALOGUE, BATTLE_LEFT_TO_MENU))

    def _process_menu(self, report: ExecutorReport) -> Dict[str, Any]:
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

    def _decide(self, subgoal: str, attempts: List[str], frame) -> Optional[Tuple[str, Optional[str]]]:
        attempts_block = ""
        if attempts:
            attempts_block = PLAYTHROUGH_ATTEMPTS_BLOCK.replace(
                "[ATTEMPTS]", "\n".join(f"- {attempt}" for attempt in attempts))
        prompt = (self._fill_prompt(PLAYTHROUGH_SUBGOAL_DECISION_PROMPT)
                  .replace("[SUBGOAL]", subgoal)
                  .replace("[ATTEMPTS_BLOCK]", attempts_block))
        output = self._vlm_call("decision", texts=prompt, images=[frame])
        decision = parse_decision(output)
        if decision is None:
            log_warn(f"Could not parse a decision from: {output!r}", self._parameters)
        return decision

    def _judge(self, subgoal: str, request: str, result: Dict[str, Any],
               subgoal_index: int = 0, move_tag: str = "") -> Tuple[bool, str, int]:
        report = result["report"]
        before = report.initial_state["core"]["current_frame"]
        after = report.final_state["core"]["current_frame"]
        after_state = self._effective_state(after, raw_state=report.final_state["pokemon_core"]["agent_state"])
        if after_state == AgentState.FREE_ROAM:
            self._tile_recognizer.record_tiles(after, self._episode_number)
            after_identified = self._tile_recognizer.identify_tiles(after)
            after_people = self._people_now()
            after_kinds = cell_kinds(after_identified, getattr(self._env, "ledge_patterns", None),
                                     getattr(self._env, "walk_patterns", None), after_people,
                                     getattr(self._env, "left_ledge_patterns", None))
            after_tiles = (verbalize_tiles(after_identified, after_people)
                           + "\nWalkable cells on each edge of the screen:\n"
                           + self._describe_edges(after_kinds))
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
        prompt = (self._fill_prompt(PLAYTHROUGH_JUDGE_PROMPT)
                  .replace("[SUBGOAL_NUMBER]", str(subgoal_index + 1))
                  .replace("[SUBGOAL]", subgoal)
                  .replace("[REMAINING_SUBGOALS_BLOCK]", remaining_block)
                  .replace("[REMAINING_SUBGOALS_LINE]", remaining_line)
                  .replace("[REQUEST]", request)
                  .replace("[AFTER_BLOCK]", after_block)
                  .replace("[TEXT_BLOCK]", text_block)
                  .replace("[EXECUTOR_REPORT]", executor_report))
        output = self._vlm_call("judge", texts=prompt, images=[before, after])
        summary = (parse_key_value(output, "Summary") or "").strip()
        if not summary:
            summary = f"{request} -> {executor_report}"
        complete = bool(parse_yes_no(output, "Subgoal complete"))
        reached = parse_int(output, "Reached subgoal", lo=subgoal_index + 1, hi=len(self.plan)) if remaining else None
        if not complete and reached is not None and reached > subgoal_index + 1:
            complete = True
        reached_index = (reached - 1) if (complete and reached is not None) else subgoal_index
        return complete, summary, reached_index

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

    def _task_check(self, frame) -> Tuple[bool, str]:
        output = self._vlm_call("task_check", texts=self._fill_prompt(PLAYTHROUGH_TASK_CHECK_PROMPT),
                                images=[frame])
        summary = (parse_key_value(output, "Summary") or "").strip()
        return bool(parse_yes_no(output, "Task complete")), summary

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
        output = self._vlm_call("battle_instructions",
                                texts=self._fill_prompt(PLAYTHROUGH_BATTLE_INSTRUCTIONS_PROMPT)
                                .replace("[BATTLE_TIPS_BLOCK]", tips),
                                images=[frame])
        instructions = (parse_key_value(output, "Instructions") or "").strip().strip("\"'")
        if not instructions or instructions.lower() == "none":
            return "win the current battle"
        return instructions

    def _combat(self, instructions: str, *, counted: bool = True) -> Any:
        task = instructions + (f"\nYour battle tips:\n{self._tips_text()}" if self._battle_tips else "")
        return self.call_executor("combat", task, counted=counted)

    def _battles_lost(self) -> int:
        return self._env.get_info()["pokemon_core"].get("n_battles_lost", 0) - self._losses_at_start

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
            result = self.call_executor(tag, task, identified=self._identified)
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
                           getattr(self._env, "left_ledge_patterns", None))
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

    def _handle_interruption(self, *, dialogue_task: str = "clear the dialogue that interrupted what the player was doing") -> Tuple[str, Dict[str, Any]]:
        states_seen: List[str] = []
        combined: Optional[Dict[str, Any]] = None
        for step_idx in range(self.MAX_INTERRUPTION_CHAIN):
            if self._env_done() or self._battles_lost() or self._total_calls >= self._max_total_calls:
                break
            frame = self._current_frame()
            state = self._effective_state(frame)
            if state == AgentState.FREE_ROAM and combined is not None:
                break
            state_label = state.name.lower()
            if not states_seen or states_seen[-1] != state_label:
                states_seen.append(state_label)
            if state == AgentState.IN_DIALOGUE:
                task = dialogue_task if step_idx == 0 else "clear the dialogue that followed"
                handled = self.call_executor("dialogue", task, counted=False)
            elif state == AgentState.IN_BATTLE:
                handled = self._combat(self._battle_instructions(frame), counted=False)
            elif state == AgentState.IN_MENU:
                objective, is_start_menu = self._menu_objective(
                    "deal with the menu that opened while the player was doing something else", frame)
                handled = self.call_executor("menu", objective, is_start_menu=is_start_menu, counted=False)
            else:
                handled = self.call_executor("free", self.CLEAR_EVENT_INSTRUCTION,
                                             termination=self.CLEAR_EVENT_TERMINATION, counted=False)
            combined = self._merge_handled(combined, handled)
            if self._env_done() or self._battles_lost():
                break
            if self._agent_state() == AgentState.FREE_ROAM:
                self._settle_free_roam()
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

    def _settle_free_roam(self, ticks: int = 2, max_cutscene_ticks: int = 16, post_motion_ticks: int = 2) -> None:
        waited = 0
        post_motion_remaining = 0
        while (self._agent_state() == AgentState.FREE_ROAM
               and not self._env_done()
               and waited < max_cutscene_ticks):
            before = self._current_frame()
            box_open = has_bottom_dialogue_box(before) or (has_top_dialogue_box(before) and not has_menu_cursor(before))
            if waited >= ticks and not box_open and post_motion_remaining <= 0:
                break
            self._env.step_high_level_action(WaitAction)
            waited += 1
            if self._agent_state() != AgentState.FREE_ROAM or self._env_done():
                break
            after = self._current_frame()
            if (is_cutscene_frame_changing(before, after)
                    or has_bottom_dialogue_box(after)
                    or (has_top_dialogue_box(after) and not has_menu_cursor(after))):
                post_motion_remaining = post_motion_ticks
            elif post_motion_remaining > 0:
                post_motion_remaining -= 1

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
                    complete, summary, reached = self._judge(
                        subgoal, request_label, handled, subgoal_index)
                    self._note(summary)
                    attempts.append(f"Dialogue(): {summary}")
                    if complete:
                        return "complete", attempts, reached
                if self._out_of_calls():
                    return "out_of_calls", attempts, subgoal_index
            iterations += 1
            frame = self._current_frame()
            self._perceive(frame)
            before_place = self._current_map_place()
            if before_place is not None and (not subgoal_places or subgoal_places[-1] != before_place):
                subgoal_places.append(before_place)
            prior_stops = list(self._leg_stops)
            decision = self._decide(subgoal, attempts, frame)
            if decision is None:
                attempts.append("the reply could not be read, so nothing was done")
                failed_attempts += 1
                continue
            action, argument = decision
            is_move_action = ACTIONS[action][0] in MOVE_TAGS
            result, task = self._run_action(action, argument, frame)
            after_place = self._record_map_stop()
            move_tag = self._format_leg_movement(before_place, after_place, prior_stops, is_move_action=is_move_action)
            request = f"{action}({task})"
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
                state, handled = self._handle_interruption()
                handled["report"].initial_state = result["report"].initial_state
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
                    complete, summary, reached = self._judge(
                        subgoal, f"{request} (cleared {state})", handled, subgoal_index, move_tag=move_tag)
                    if move_tag and move_tag not in summary:
                        summary = f"{move_tag} {summary}"
                    self._note(summary)
                    attempts.append(f"{request}: {summary}")
                    if complete:
                        return "complete", attempts, reached
                if after_place is not None and (not subgoal_places or subgoal_places[-1] != after_place):
                    subgoal_places.append(after_place)
                if not is_move_action or not handled.get("succeeded"):
                    failed_attempts += 1
                continue

            if not result["succeeded"]:
                note = f"{request} did not work: {result['reason']}" + (
                    f" ({result['notes']})" if result.get("notes") else "")
                if move_tag:
                    note = f"{move_tag} {note}"
                self._note(note)
                attempts.append(note)
                failed_attempts += 1
                continue

            complete, summary, reached = self._judge(subgoal, request, result, subgoal_index, move_tag=move_tag)
            if move_tag and move_tag not in summary:
                summary = f"{move_tag} {summary}"
            self._note(summary)
            attempts.append(f"{request}: {summary}")
            if complete:
                return "complete", attempts, reached
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
        index = 0
        while index < len(self.plan):
            subgoal = self.plan[index]
            outcome, attempts, reached_index = self._pursue(subgoal, index)

            if outcome == "complete":
                index = max(index, reached_index) + 1
                if index < len(self.plan):
                    if not self._out_of_calls() and not self._env_done():
                        continue
                    self._settle_free_roam()
                    frame = self._current_frame()
                    self._perceive(frame)
                    complete, summary = self._task_check(frame)
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
                complete, summary = self._task_check(frame)
                if complete:
                    return self._finish("completed", None, summary)
                if (replans >= self.MAX_REPLANS or self._out_of_calls()
                        or self._env_done()):
                    return self._finish("plan_done_task_incomplete",
                                        "Worked through the plan without finishing the task.",
                                        summary)
                self._note(f"The plan was finished but the task is not done: {summary}")
                self.plan = self._plan_subgoals(frame)
                replans += 1
                if not self.plan:
                    return self._finish("plan_done_task_incomplete",
                                        "Worked through the plan without finishing the task.",
                                        summary)
                index = 0
                continue

            if outcome == "env_done":
                frame = self._current_frame()
                self._perceive(frame)
                complete, summary = self._task_check(frame)
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
                        complete, summary = self._task_check(frame)
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
            self.plan = subgoals
            index = 0

        return self._finish("no_plan", "The plan became empty while working through it.")
