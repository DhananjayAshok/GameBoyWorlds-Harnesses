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
                                                 is_cutscene_frame_changing, is_healing_item_name,
                                                 is_overworld_menu_or_ui_screen,
                                                 is_overworld_pack_menu, is_pack_screen,
                                                 is_party_menu_screen, is_status_heal_item_name,
                                                 is_unrevealed_battle_intro_frame,
                                                 read_battle_hud_hp, read_move_learning_dialogue_status,
                                                 read_tile_text)
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
                                         drop_prefix_crops, effective_agent_state,
                                         extract_bottom_dialogue_crop, settle_overworld_free_roam,
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
from execution.strategist.pokemon.party_state import PartyInventoryState
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
                 party_state: Optional[PartyInventoryState] = None,
                 initial_battle_hp: Optional[Tuple[int, int]] = None,
                 initial_healing_bag_empty: bool = False,
                 initial_pokemon_status: Optional[str] = None,
                 initial_status_heal_bag_empty: bool = False,
                 initial_zero_pp_moves: Optional[List[str]] = None,
                 initial_known_bag_items: Optional[List[str]] = None,
                 initial_known_bag_complete: bool = False,
                 initial_blocked_moves: Optional[Dict[Tuple[int, Tuple[int, int]], Dict[str, List[str]]]] = None,
                 **kwargs: Any) -> None:
        self._screen_people = screen_people
        self._current_tag: Optional[str] = None
        self._guidance = guidance
        self._battle_tips = list(battle_tips or [])
        if party_state is None:
            party_state = PartyInventoryState(
                items=list(initial_known_bag_items or []),
                saw_top=bool(initial_known_bag_complete),
                saw_bottom=bool(initial_known_bag_complete),
                healing_exhausted=bool(initial_healing_bag_empty),
                status_heal_exhausted=bool(initial_status_heal_bag_empty),
                last_battle_hp=initial_battle_hp,
                active_pokemon_status=initial_pokemon_status or None,
                zero_pp_moves=list(initial_zero_pp_moves or []),
            )
        self.party_state: PartyInventoryState = party_state
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
        self._leg_departures: List[Tuple[Tuple[int, Tuple[int, int]], str, Tuple[int, Tuple[int, int]]]] = []
        self._blocked_moves: Dict[Tuple[int, Tuple[int, int]], Dict[str, List[str]]] = {
            k: {d: list(notes) for d, notes in by_dir.items()}
            for k, by_dir in (initial_blocked_moves or {}).items()
        }
        self._last_settled_step: Optional[int] = None
        self._dialogue_clear_failed: bool = False
        self._dialogue_executor_failed: bool = False
        self._free_in_dialogue_failed: bool = False
        self._dialogue_fail_count: int = 0
        self._zero_step_dialogue_failures: int = 0
        self._dialogue_failed_step: Optional[int] = None
        self._last_failed_dialogue_text: Optional[str] = None
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
        if raw in (AgentState.IN_MENU, AgentState.FREE_ROAM, AgentState.IN_DIALOGUE):
            frame = self._current_frame() if frame is None else frame
            eff = effective_agent_state(raw, frame)
            if raw == AgentState.IN_MENU and eff == AgentState.IN_BATTLE and is_overworld_menu_or_ui_screen(frame):
                return AgentState.IN_MENU
            return eff
        return raw

    def _is_in_dialogue(self, frame=None) -> bool:
        return self._effective_state(frame) == AgentState.IN_DIALOGUE

    def _current_step_count(self) -> Optional[int]:
        return getattr(getattr(self._env, "_emulator", None), "step_count", None)

    def _reset_dialogue_failure_state(self) -> None:
        self._dialogue_clear_failed = False
        self._dialogue_executor_failed = False
        self._free_in_dialogue_failed = False
        self._dialogue_fail_count = 0
        self._zero_step_dialogue_failures = 0
        self._dialogue_failed_step = None
        self._last_failed_dialogue_text = None

    def _sync_dialogue_failure_state(self) -> None:
        if not self._is_in_dialogue():
            self._reset_dialogue_failure_state()

    @staticmethod
    def _trailing_dialogue_steps_stalled(report: ExecutorReport, min_stalled: int = 3) -> bool:
        steps = getattr(report, "steps", None) or []
        if not steps:
            return True
        if len(steps) < min_stalled:
            return not any(getattr(s, "frame_changed", False) for s in steps)
        tail = steps[-min_stalled:]
        return all(
            not getattr(s, "frame_changed", False) or getattr(s, "action_success", 0) == -1
            for s in tail
        )

    def _record_dialogue_executor_outcome(
        self,
        tag: str,
        executor: Any,
        result: Dict[str, Any],
        was_in_dialogue: bool,
        step_before: Optional[int],
        step_after: Optional[int],
    ) -> None:
        if not self._is_in_dialogue():
            self._reset_dialogue_failure_state()
            return
        if result.get("succeeded"):
            return
        report = result.get("report")
        dialogue_text = " ".join(str(result.get("dialogue") or getattr(executor, "dialogue_text", "") or "").split())
        zero_steps = step_before is not None and step_after == step_before
        tail_stalled = report is not None and self._trailing_dialogue_steps_stalled(report)
        same_text = bool(
            dialogue_text
            and self._last_failed_dialogue_text
            and dialogue_text == self._last_failed_dialogue_text
        )
        if tag == "dialogue":
            self._dialogue_clear_failed = True
            self._dialogue_fail_count += 1
            self._dialogue_failed_step = step_after
            if (
                zero_steps
                or tail_stalled
                or same_text
                or result.get("reason") == "dialogue_stuck"
                or self._dialogue_fail_count >= 2
            ):
                self._dialogue_executor_failed = True
            if zero_steps or tail_stalled:
                self._zero_step_dialogue_failures += 1
            else:
                self._zero_step_dialogue_failures = 0
            if dialogue_text:
                self._last_failed_dialogue_text = dialogue_text
        elif tag == "free" and result.get("reason") != "move_interrupted" and (was_in_dialogue or zero_steps or tail_stalled):
            self._dialogue_clear_failed = True
            self._dialogue_failed_step = step_after
            presses = [str(p).lower() for p in (getattr(executor, "presses", None) or ())]
            a_tail = len(presses) >= 3 and all(p == "a" for p in presses[-3:])
            if (zero_steps or tail_stalled or same_text) and (was_in_dialogue or a_tail):
                if a_tail and "b" not in presses[-5:]:
                    self._dialogue_executor_failed = True
                if was_in_dialogue:
                    self._free_in_dialogue_failed = True
            if zero_steps or tail_stalled:
                self._zero_step_dialogue_failures += 1
            else:
                self._zero_step_dialogue_failures = 0
            if dialogue_text:
                self._last_failed_dialogue_text = dialogue_text

    def _is_dialogue_unrecoverable(self) -> bool:
        return (
            self._is_in_dialogue()
            and self._dialogue_executor_failed
            and self._free_in_dialogue_failed
            and self._zero_step_dialogue_failures >= 2
        )

    @staticmethod
    def _is_dialogue_closing_subgoal(text: str) -> bool:
        if not text:
            return False
        has_dialogue_term = bool(
            re.search(
                r"\b(?:dialogue|text\s*box|conversation|cutscene|call|message|banner)\b",
                text,
                re.IGNORECASE,
            )
        )
        has_close_term = bool(
            re.search(
                r"\b(?:close[ds]?|clear(?:ed)?|dismiss(?:ed)?|finish(?:ed)?|end(?:ed)?|complete[ds]?|back\s+in\s+free\s+roam|return(?:ed)?\s+to\s+free\s+roam)\b",
                text,
                re.IGNORECASE,
            )
        )
        return has_dialogue_term and has_close_term

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

    @property
    def _last_seen_battle_hp(self) -> Optional[Tuple[int, int]]:
        return self.party_state.last_battle_hp

    @_last_seen_battle_hp.setter
    def _last_seen_battle_hp(self, value: Optional[Tuple[int, int]]) -> None:
        self.party_state.last_battle_hp = value

    @property
    def _healing_bag_empty(self) -> bool:
        return self.party_state.healing_bag_empty

    @_healing_bag_empty.setter
    def _healing_bag_empty(self, value: bool) -> None:
        self.party_state.healing_bag_empty = bool(value)

    @property
    def _active_pokemon_status(self) -> Optional[str]:
        return self.party_state.active_pokemon_status

    @_active_pokemon_status.setter
    def _active_pokemon_status(self, value: Optional[str]) -> None:
        self.party_state.active_pokemon_status = value or None

    @property
    def _status_heal_bag_empty(self) -> bool:
        return self.party_state.status_heal_bag_empty

    @_status_heal_bag_empty.setter
    def _status_heal_bag_empty(self, value: bool) -> None:
        self.party_state.status_heal_bag_empty = bool(value)

    @property
    def _zero_pp_moves(self) -> List[str]:
        return self.party_state.zero_pp_moves

    @_zero_pp_moves.setter
    def _zero_pp_moves(self, value: List[str]) -> None:
        self.party_state.zero_pp_moves = list(value or [])

    @property
    def _learned_moves(self) -> List[str]:
        return getattr(self.party_state, "learned_moves", [])

    def _scan_frame_for_party_move_status(self, frame) -> None:
        st = read_move_learning_dialogue_status(frame)
        if st is None:
            return
        kind, pokemon, move = st
        if kind in ("learned", "knows"):
            self.party_state.record_learned_move(move, pokemon, episode_number=self._episode_number)
        elif kind == "forgot":
            self.party_state.record_forgotten_move(move, pokemon, episode_number=self._episode_number)

    @property
    def _known_bag_items(self) -> List[str]:
        return self.party_state.items

    @_known_bag_items.setter
    def _known_bag_items(self, value: List[str]) -> None:
        self.party_state.items = [
            PartyInventoryState.normalize_item_name(x) for x in (value or [])
            if PartyInventoryState.normalize_item_name(x)
        ]

    @property
    def _known_bag_complete(self) -> bool:
        return self.party_state.bag_complete

    @_known_bag_complete.setter
    def _known_bag_complete(self, value: bool) -> None:
        self.party_state.bag_complete = bool(value)

    @property
    def _poison_step_remainder(self) -> int:
        return self.party_state.poison_step_remainder

    @_poison_step_remainder.setter
    def _poison_step_remainder(self, value: int) -> None:
        self.party_state.poison_step_remainder = int(value)

    def _apply_overworld_poison_step_decay(
        self,
        prev_place: Tuple[int, Tuple[int, int]],
        next_place: Tuple[int, Tuple[int, int]],
    ) -> None:
        if self._active_pokemon_status != "PSN" or self._last_seen_battle_hp is None:
            return
        gmap = self._global_map()
        if gmap is None:
            return
        try:
            p_seg, p_pos = gmap.resolve(prev_place[0], prev_place[1])
            n_seg, n_pos = gmap.resolve(next_place[0], next_place[1])
        except Exception:  # noqa: BLE001
            return
        steps_walked = (
            abs(n_pos[0] - p_pos[0]) + abs(n_pos[1] - p_pos[1])
            if p_seg == n_seg
            else 1
        )
        self.party_state.record_overworld_steps(steps_walked)

    def _record_map_stop(self) -> Optional[Tuple[int, Tuple[int, int]]]:
        place = self._current_map_place()
        if place is None:
            return None
        if self._start_place is None:
            self._start_place = place
        if not self._leg_stops or self._leg_stops[-1] != place:
            if self._leg_stops:
                self._apply_overworld_poison_step_decay(self._leg_stops[-1], place)
            self._leg_stops.append(place)
        return place

    def _current_segment_blocked_moves(self) -> Dict[Tuple[int, int], set]:
        gmap = self._global_map()
        place = self._current_map_place()
        if gmap is None or place is None:
            return {}
        cur_seg, _ = place
        extra: Dict[Tuple[int, int], set] = {}
        for (b_seg, b_pos), by_dir in self._blocked_moves.items():
            try:
                r_seg, r_pos = gmap.resolve(b_seg, b_pos)
            except Exception:  # noqa: BLE001
                continue
            if r_seg == cur_seg and by_dir:
                extra.setdefault(r_pos, set()).update(by_dir.keys())
        if hasattr(gmap, "segment_blocked_moves"):
            return gmap.segment_blocked_moves(cur_seg, extra)
        return extra

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

        door_leads: Dict[Tuple[int, int], Optional[int]] = {}
        if hasattr(gmap, "reachable_door_list"):
            try:
                for d in gmap.reachable_door_list(cur_seg):
                    for cell in d.get("cells", ()):
                        door_leads[cell] = d.get("leads_to")
            except Exception:  # noqa: BLE001
                door_leads = {}

        episode_visited_here: set = set()
        for s_seg, s_pos in getattr(self, "_leg_stops", ()):
            try:
                r_seg, r_pos = gmap.resolve(s_seg, s_pos)
                if r_seg == cur_seg:
                    episode_visited_here.add(r_pos)
            except Exception:  # noqa: BLE001
                pass

        offscreen_entity_cells: set = set()
        if re.search(
            r"\b(?:leader|trainer|gym|person|people|npc|nurse|clerk|talk|speak|challenge|battle|defeat|interact|item|ball)\b",
            self._task or "",
            re.IGNORECASE,
        ):
            npc_reg = getattr(gmap, "_npcs", None) or getattr(self._env, "_npcs", None)
            if npc_reg is not None and hasattr(npc_reg, "people_here"):
                try:
                    for _, sq in npc_reg.people_here(gmap):
                        if strictly_beyond(sq):
                            offscreen_entity_cells.add(sq)
                except Exception:  # noqa: BLE001
                    pass
            for g_cell, mc in seg.items():
                if strictly_beyond(g_cell) and (
                    getattr(mc, "kind", None) in ("occupied", "item")
                    or bool(set(getattr(mc, "tags", ())) & {"npc", "item", "interactable object"})
                ):
                    offscreen_entity_cells.add(g_cell)

        def _has_unvisited_offscreen_entity(g: Tuple[int, int]) -> bool:
            if not offscreen_entity_cells or g not in offscreen_entity_cells:
                return False
            return not any(
                (g[0] + dx, g[1] + dy) in episode_visited_here
                for dx, dy in DIRECTIONS.values()
            )

        def _is_open_door_cell(g: Tuple[int, int], c: Any) -> bool:
            if c is None or not is_door(c):
                return False
            if g not in door_leads:
                return True
            leads = door_leads[g]
            if leads is None:
                return True
            try:
                if getattr(gmap.info(leads), "explored", None) is not None and not gmap.frontier(leads):
                    return False
                if hasattr(gmap, "branch_exploration") and gmap.branch_exploration(cur_seg, leads).get("exhausted"):
                    return False
            except Exception:  # noqa: BLE001
                return True
            return True

        seeds = []
        for lx, ly in run_cells:
            eg = (px + lx, py + ly)
            if eg in blocked_targets:
                continue
            c = seg.get(eg)
            if c is None or c.kind == UNKNOWN or _is_open_door_cell(eg, c):
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
            if _is_open_door_cell(cur, c):
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
                    if nc is None or nc.kind in (WALKABLE, UNKNOWN) or _is_open_door_cell(nxt, nc):
                        half_plane_open = True
                        break
                    continue
                nc = seg.get(nxt)
                if nc is None or nc.kind == UNKNOWN or _is_open_door_cell(nxt, nc) or _has_unvisited_offscreen_entity(nxt):
                    half_plane_open = True
                    break
                if nc.kind == WALKABLE and nxt not in visited:
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
                if _is_open_door_cell(cur, c):
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
                    if nc is None or nc.kind == UNKNOWN or _is_open_door_cell(nxt, nc) or _has_unvisited_offscreen_entity(nxt):
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
        prior_departures = []
        for dep_place, dep_req, dst_place in self._leg_departures:
            try:
                d_seg, d_pos = gmap.resolve(dep_place[0], dep_place[1])
                t_seg, t_pos = gmap.resolve(dst_place[0], dst_place[1])
            except Exception:  # noqa: BLE001
                continue
            if d_seg == cur_seg and d_pos == cur_pos and (d_seg, d_pos) != (t_seg, t_pos):
                entry = f"{dep_req} -> {t_pos}" if t_seg == cur_seg else f"{dep_req} -> entered another area"
                if entry not in prior_departures:
                    prior_departures.append(entry)
        if prior_departures:
            lines.append(
                f"Actions already taken from your current coordinate {cur_pos} earlier in this episode "
                f"(do NOT repeat a leg that led into a dead-end pocket): {'; '.join(prior_departures)}."
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

    def _sync_screen_to_global_map(self, frame, identified, gmap) -> None:
        if (
            gmap is None
            or not getattr(gmap, "current", None)
            or getattr(gmap, "pos", None) is None
            or getattr(gmap, "residual", None) != (0, 0)
            or getattr(gmap, "faded", False)
        ):
            return
        try:
            gmap.observe(frame, identified, record_visit=False)
            cur_seg, (px, py) = gmap.resolve(gmap.segment, gmap.pos)
            seg = gmap.segments[cur_seg]
            seg_blocked = gmap.blocked_cells[cur_seg] if cur_seg < len(getattr(gmap, "blocked_cells", ())) else set()
            for (b_seg, b_pos), by_dir in list(self._blocked_moves.items()):
                r_seg, r_pos = gmap.resolve(b_seg, b_pos)
                if r_seg != cur_seg or r_pos != (px, py):
                    continue
                for d_name in list(by_dir):
                    if d_name not in DIRECTIONS:
                        continue
                    dx, dy = DIRECTIONS[d_name]
                    target = (px + dx, py + dy)
                    c = seg.get(target)
                    if c is not None and c.kind == WALKABLE and target not in seg_blocked:
                        by_dir.pop(d_name, None)
                if not by_dir:
                    self._blocked_moves.pop((b_seg, b_pos), None)
        except Exception:  # noqa: BLE001
            pass

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
        self._sync_screen_to_global_map(frame, self._identified, gmap)
        people = self._people_now()
        kinds = cell_kinds(self._identified, getattr(self._env, "ledge_patterns", None),
                           getattr(self._env, "walk_patterns", None), people,
                           getattr(self._env, "left_ledge_patterns", None),
                           self._local_blocked_cells(), global_map=gmap)
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
                global_map=self._global_map(),
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
        if tag in ("combat", "menu"):
            run_kwargs.setdefault("party_state", self.party_state)
        run_kwargs.update(executor_kwargs)
        run_kwargs["hint"] = hint
        run_kwargs["allow_self_termination"] = allow_self_termination
        self._sync_dialogue_failure_state()
        was_in_dialogue = self._is_in_dialogue()
        step_before = self._current_step_count()
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
            result = self.process_executor_return(executor.report)
            step_after = self._current_step_count()
            self._record_dialogue_executor_outcome(
                tag, executor, result, was_in_dialogue, step_before, step_after
            )
            return result
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

    def _update_party_from_dialogue_text(self, notes: Optional[str]) -> None:
        self.party_state.update_from_dialogue_notes(notes, episode_number=self._episode_number)

    def _process_interaction(self, report: ExecutorReport) -> Dict[str, Any]:
        succeeded = report.outcome in (0, 1, 2)
        if succeeded:
            self._update_party_from_dialogue_text(report.notes)
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

    def _update_bag_status_from_notes(self, notes: str, task: Optional[str] = None) -> bool:
        return self.party_state.update_from_bag_notes(notes, task)

    def _update_party_hp_from_combat(self, report: ExecutorReport) -> None:
        self.party_state.update_from_combat_report(
            report,
            executor=self.last_executor,
            episode_number=self._episode_number,
        )

    def _update_party_hp_from_menu(self, report: ExecutorReport) -> None:
        self.party_state.update_from_menu_report(
            report,
            episode_number=self._episode_number,
        )

    def _party_hp_block(self) -> str:
        if (
            self._last_seen_battle_hp is None
            and not self._healing_bag_empty
            and not self._active_pokemon_status
            and not self._zero_pp_moves
            and not self._known_bag_items
            and not self._learned_moves
        ):
            return ""
        lines: List[str] = []
        status_note = ""
        if self._active_pokemon_status:
            if self._status_heal_bag_empty:
                status_note = (
                    f" [status: {self._active_pokemon_status} — bag was already checked and has NO Antidote or "
                    f"status-curing item; do NOT ask for an Antidote in the menu]"
                )
            else:
                status_note = f" [status: {self._active_pokemon_status}]"
        if self._last_seen_battle_hp is not None:
            cur_hp, max_hp = self._last_seen_battle_hp
            low = cur_hp * 2 <= max_hp
            bag_note = (
                " (healing bag was already checked and has NO usable healing items — "
                "do NOT open the ITEM menu or instruct the player to use a Potion, even if task guidance mentions Potions)"
                if self._healing_bag_empty else ""
            )
            if low and not self._healing_bag_empty:
                lines.append(
                    f"Active Pokemon last seen battle HP: {cur_hp}/{max_hp} HP{status_note} (LOW — at or below 50% max HP; "
                    f"before walking further along a route/cave or talking to a trainer, use Menu(open the ITEM menu "
                    f"and use a Potion to heal your active Pokemon, then close the menu) first)."
                )
            elif low and self._healing_bag_empty:
                lines.append(
                    f"Active Pokemon last seen battle HP: {cur_hp}/{max_hp} HP{status_note} (LOW — at or below 50% max HP){bag_note}."
                )
            else:
                lines.append(f"Active Pokemon last seen battle HP: {cur_hp}/{max_hp} HP{status_note}{bag_note}.")
        else:
            if self._active_pokemon_status:
                lines.append(f"Active Pokemon status: {self._active_pokemon_status}{status_note}.")
            if self._healing_bag_empty:
                lines.append(
                    "Bag healing item status: No usable healing items in bag (checked earlier — "
                    "do NOT open the ITEM menu or instruct the player to use a Potion)."
                )
        if self._zero_pp_moves:
            lines.append(
                f"Active Pokemon moves at 0 PP (cannot be used until healed at a Pokemon Center): "
                f"{', '.join(self._zero_pp_moves)}."
            )
        if self._known_bag_items:
            complete_note = " (all bag slots checked)" if self._known_bag_complete else ""
            lines.append(f"Bag items currently known: {', '.join(self._known_bag_items)}{complete_note}.")
        if self._learned_moves:
            formatted_moves = self.party_state.format_learned_moves()
            lines.append(
                f"Party moves learned / confirmed known: {', '.join(formatted_moves)} "
                f"(already learned — do NOT open the Pack or ITEM menu to teach {', '.join(self._learned_moves)} again)."
            )
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
        self._update_party_from_dialogue_text(report.notes)
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
            r"\b(?:start|menu|dialogue|text|cutscene|wait|evolve|evolution)\b"
            r"|(?:\b(?:press|pressing|hit|push|tap|hold)\s+(?:the\s+)?b\b|\bb\s+button\b)",
            instruction,
            re.IGNORECASE,
        ):
            return None
        wants_interact = bool(
            re.search(
                r"\b(?:talk|speak|interact)\b"
                r"|(?:\b(?:press|pressing|hit|push|tap|hold|then|and)\s+(?:the\s+)?a\b|\ba\s+button\b)",
                instruction,
                re.IGNORECASE,
            )
        )
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
        people = self._people_now()
        warps = warp_cells(identified=self._identified, global_map=gmap)
        if wants_interact:
            explicit_coords = [c for c in parse_all_cells(instruction) if in_bounds(c) and c != (0, 0)]
            if len(explicit_coords) == 1:
                tc = explicit_coords[0]
                goals = approach_cells(local_kinds, tc, identified=self._identified, people=people)
                if goals and shortest_path(local_kinds, (0, 0), goals, avoid_cells=warps) is not None:
                    return "Interact", f"person at {tc}"
            cur_i = (0, 0)
            for d_leg, n_leg in legs:
                dx_l, dy_l = DIRECTIONS[d_leg]
                for _ in range(n_leg):
                    nxt = (cur_i[0] + dx_l, cur_i[1] + dy_l)
                    if in_bounds(nxt) and local_kinds.get(nxt) == WALKABLE and nxt not in warps:
                        cur_i = nxt
                    else:
                        break
            if cur_i != (0, 0) and legs:
                last_dx, last_dy = DIRECTIONS[legs[-1][0]]
                cand_target = (cur_i[0] + last_dx, cur_i[1] + last_dy)
                if in_bounds(cand_target) and cand_target != (0, 0) and local_kinds.get(cand_target) == BLOCKED:
                    goals = approach_cells(local_kinds, cand_target, identified=self._identified, people=people)
                    if goals and shortest_path(local_kinds, (0, 0), goals, avoid_cells=warps) is not None:
                        return "Interact", f"person at {cand_target}"
            return None
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
                exact_end = cur
                exact_ok = True
                for _ in range(n_last):
                    nxt = (exact_end[0] + dx_last, exact_end[1] + dy_last)
                    if not in_bounds(nxt) or local_kinds.get(nxt) != WALKABLE or nxt in warps:
                        exact_ok = False
                        break
                    exact_end = nxt
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
                if exact_ok and exact_end != (0, 0):
                    if shortest_path(local_kinds, (0, 0), [exact_end], avoid_cells=warps, people=people) is not None:
                        return "MoveTo", f"{exact_end}"
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
        effective_attempts = list(attempts)
        if self._is_in_dialogue(frame) and self._dialogue_executor_failed:
            snippet = f" (\"{self._last_failed_dialogue_text[:120]}\")" if self._last_failed_dialogue_text else ""
            warn_note = (
                f"Dialogue() already ran out of steps without closing the currently open dialogue box{snippet} — "
                f"do NOT choose Dialogue() again; choose Free(press b and a alternately to close the open dialogue box; "
                f"stop when: the dialogue box disappears and the player is in free roam)"
            )
            if not any("do NOT choose Dialogue() again" in a for a in effective_attempts):
                effective_attempts.append(warn_note)
        if effective_attempts:
            attempts_block = PLAYTHROUGH_ATTEMPTS_BLOCK.replace(
                "[ATTEMPTS]", "\n".join(f"- {attempt}" for attempt in effective_attempts))
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
        elif decision[0] == "Dialogue" and self._is_in_dialogue(frame) and self._dialogue_executor_failed:
            coerced = (
                "Free",
                "press b and a alternately to close the open dialogue box; "
                "stop when: the dialogue box disappears and the player is in free roam",
            )
            decision = coerced
            if records:
                records[-1].response = re.sub(
                    r"(?im)^(\s*Action\s*:\s*).*$",
                    f"\1{coerced[0]}({coerced[1]})",
                    str(records[-1].response),
                )
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
            self._sync_screen_to_global_map(after, after_identified, gmap)
            after_people = self._people_now()
            after_kinds = cell_kinds(after_identified, getattr(self._env, "ledge_patterns", None),
                                     getattr(self._env, "walk_patterns", None), after_people,
                                     getattr(self._env, "left_ledge_patterns", None),
                                     self._local_blocked_cells(), global_map=gmap)
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
        if re.search(
            r"\b(?:all\s+potions\s+(?:in\s+the\s+bag\s+)?have\s+(?:now\s+)?been\s+used\s+up"
            r"|used\s+up\s+(?:all\s+)?(?:our\s+|the\s+)?remaining\s+potions"
            r"|using\s+up\s+(?:all\s+)?(?:our\s+|the\s+)?remaining\s+potions"
            r"|no\s+(?:usable\s+)?healing\s+items\s+(?:left\s+|remaining\s+)?in\s+(?:the\s+)?bag"
            r"|healing\s+bag\s+is\s+empty)\b",
            f"{output}\n{summary}",
            re.IGNORECASE,
        ):
            self._healing_bag_empty = True
        complete = bool(parse_yes_no(output, "Subgoal complete"))
        replan_needed = bool(parse_yes_no(output, "Replan needed")) and not complete
        reached = parse_int(output, "Reached subgoal", lo=subgoal_index + 1, hi=len(self.plan)) if remaining else None
        if not complete and reached is not None and reached > subgoal_index + 1:
            complete = True
            replan_needed = False
        reached_index = (reached - 1) if (complete and reached is not None) else subgoal_index
        combined_report_text = f"{text}\n{executor_report}"
        if (
            self._is_unconditional_move_teaching_subgoal(subgoal)
            and "[No move was taught in menu:" in combined_report_text
            and not self._subgoal_target_move_is_learned(subgoal)
            and not re.search(r"\[Move (?:taught|already known) in menu:", combined_report_text, re.IGNORECASE)
        ):
            if complete or reached_index > subgoal_index:
                complete = False
                reached_index = subgoal_index
                replan_needed = False
                summary = (
                    f"{summary} [Note: no move was taught to any party Pokemon before the menu closed — "
                    f"reopen the menu and complete teaching the move on the party screen.]"
                )
        elif (
            self._is_pure_move_teaching_subgoal(subgoal)
            and after_state == AgentState.FREE_ROAM
            and self._subgoal_target_move_is_learned(subgoal)
            and not complete
        ):
            complete = True
            replan_needed = False
            learned_str = ", ".join(self.party_state.format_learned_moves())
            summary = f"{summary} [Note: {learned_str} is already confirmed learned and the menu is closed in free roam.]"
        elif (
            not self._is_pure_move_teaching_subgoal(subgoal)
            and after_state == AgentState.FREE_ROAM
            and request.startswith("Menu(")
            and (self._subgoal_target_move_is_learned(request) or self._subgoal_target_move_is_learned(subgoal))
        ):
            replan_needed = False
            if re.search(
                r"\b(?:without\s+teaching|not\s+(?:yet\s+)?(?:been\s+)?taught|has\s+not\s+learned|no\s+party\s+pok[eé]mon\s+knows)\b",
                summary,
                re.IGNORECASE,
            ):
                learned_str = ", ".join(self.party_state.format_learned_moves())
                summary = (
                    f"{summary} [Note: {learned_str} is already confirmed learned on the party — "
                    f"do NOT reopen the Pack or ITEM menu to teach it again; interact directly with the obstacle or use the field move.]"
                )
        elif (
            not complete
            and after_state == AgentState.FREE_ROAM
            and request.startswith("Interact(")
            and self._is_completed_room_gift_conversation(subgoal, text, summary)
        ):
            complete = True
            replan_needed = False
        if complete and reached_index >= len(self.plan) - 1:
            unfinished_note = self._unfinished_room_exploration_note(subgoal, after_kinds)
            if unfinished_note is None:
                unfinished_note = self._unfinished_open_corridor_target_note(subgoal, after_kinds)
            if unfinished_note is None:
                unfinished_note = self._unfinished_corridor_traversal_note(after_kinds, subgoal=subgoal)
            if unfinished_note is None:
                unfinished_note = self._budget_exhausted_unfinished_subgoal_note(subgoal, output, after_kinds)
            if unfinished_note:
                complete = False
                replan_needed = False
                summary = f"{summary} [Note: {unfinished_note}.]"
        return complete, summary, reached_index, replan_needed

    @staticmethod
    def _is_move_teaching_subgoal(text: str) -> bool:
        if not text:
            return False
        return bool(
            re.search(r"\b(?:teach|taught|learn(?:ed)?)\b", text, re.IGNORECASE)
            and re.search(
                r"\b(?:hm\s*\d*|tm\s*\d*|tm\s*pocket|pack|bag|item|move|cut|flash|surf|strength|fly|whirlpool|waterfall|rock\s+smash|headbutt)\b",
                text,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _is_unconditional_move_teaching_subgoal(cls, text: str) -> bool:
        if not cls._is_move_teaching_subgoal(text):
            return False
        return not bool(
            re.search(
                r"\b(?:if\s+(?:present|available|you\s+have|in\s+(?:the\s+)?bag)|check\s+(?:your\s+|the\s+)?(?:item|bag|pack)\b.*\bfor)\b",
                text,
                re.IGNORECASE,
            )
        )

    @classmethod
    def _is_pure_move_teaching_subgoal(cls, text: str) -> bool:
        if not cls._is_move_teaching_subgoal(text):
            return False
        return not bool(
            re.search(
                r"\b(?:cut\s+down|chop\s+down|walk(?:ed)?\s+(?:to|north|south|east|west|up|down|left|right|past|through|along)"
                r"|stand(?:ing)?\s+(?:at|below|above|beside|in\s+front)|talk(?:ed)?\s+to|speak|spoken|defeat(?:ed)?)\b",
                text,
                re.IGNORECASE,
            )
        )

    def _subgoal_target_move_is_learned(self, text: str) -> bool:
        if not text or not self._learned_moves:
            return False
        for move in self._learned_moves:
            if re.search(r"\b" + re.escape(move) + r"\b", text, re.IGNORECASE):
                return True
        return False

    @staticmethod
    def _is_completed_room_gift_conversation(subgoal: str, dialogue_text: str, judge_summary: str) -> bool:
        if not subgoal or not (dialogue_text or judge_summary):
            return False
        is_generic_room_talk = bool(
            re.search(
                r"\b(?:talk(?:ed)?|speak|spoke(?:n)?|chat(?:ted)?|converse[ds]?)\s+(?:to|with)\s+"
                r"(?:each\s+of\s+|all\s+(?:of\s+)?)?(?:the\s+)?(?:people|npcs|everyone|residents|folks)\b",
                subgoal,
                re.IGNORECASE,
            )
        )
        if not is_generic_room_talk:
            return False
        combined_text = f"{dialogue_text or ''}\n{judge_summary or ''}"
        received_gift = bool(
            re.search(
                r"\b(?:received\s+(?:tm|hm)\d+|received\s+[a-z0-9][a-z0-9\s'\-]{1,24}[!.]"
                r"|put\s+the\s+[a-z0-9][a-z0-9\s'\-]{1,24}\s+in\s+the\s+(?:tm|item|key\s+item|ball)\s+pocket"
                r"|obtained\s+(?:tm|hm)\d+|gave\s+(?:the\s+player\s+)?(?:tm|hm)\d+)\b",
                combined_text,
                re.IGNORECASE,
            )
        )
        if not received_gift:
            return False
        if re.search(
            r"\b(?:pok[eé]mon\s+beside|pok[eé]mon\s+behind|npc/pok[eé]mon|companion\s+pok[eé]mon|bird|butterfly|butterfree|pikachu|clefairy|jigglypuff|nidoran|slowpoke|fearow|pidgey|murkrow)\b",
            judge_summary or "",
            re.IGNORECASE,
        ) or not re.search(
            r"\b(?:guard|clerk|receptionist|nurse|captain|elder|leader|professor|aide)\b",
            judge_summary or "",
            re.IGNORECASE,
        ):
            return True
        return False

    def _reconsider(self, subgoal: str, attempts: List[str], frame) -> Tuple[str, Optional[str], List[str]]:
        prompt = (self._fill_prompt(PLAYTHROUGH_ACHIEVABLE_PROMPT)
                  .replace("[SUBGOAL]", subgoal)
                  .replace("[ATTEMPTS]", "\n".join(f"- {attempt}" for attempt in attempts)))
        output = self._vlm_call("reconsider", texts=prompt, images=[frame])
        decision = (parse_key_value(output, "Decision") or "").strip().lower()
        reason = (parse_key_value(output, "Reason") or "").strip()
        subgoals = [line.strip() for line in parse_list(output, "Subgoals") if line.strip()]
        if (
            subgoals
            and self._effective_state(frame) == AgentState.FREE_ROAM
            and not self._is_in_dialogue(frame)
            and self._learned_moves
        ):
            while (
                len(subgoals) > 1
                and self._is_pure_move_teaching_subgoal(subgoals[0])
                and self._subgoal_target_move_is_learned(subgoals[0])
            ):
                subgoals.pop(0)
        if "quit" in decision or not subgoals:
            return "quit", reason or "No new plan was given.", []
        return "replan", None, subgoals

    def _unfinished_open_corridor_target_note(
        self,
        subgoal: str,
        local_kinds: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> Optional[str]:
        remaining_calls = max(0, self._max_steps - self._calls)
        if remaining_calls < 1 or local_kinds is None:
            return None
        if self._agent_state() != AgentState.FREE_ROAM or self._is_in_dialogue():
            return None
        combined = f"{self._task or ''} {subgoal or ''}"
        if not re.search(r"\b(?:ladder|stair(?:s|case)?|door(?:way)?|exit)\b", combined, re.IGNORECASE):
            return None
        gmap = self._global_map()
        if gmap is not None and self._start_place is not None:
            cur_place = self._current_map_place()
            if cur_place is not None:
                c_seg, _ = gmap.resolve(cur_place[0], cur_place[1])
                s_seg, _ = gmap.resolve(self._start_place[0], self._start_place[1])
                if c_seg != s_seg:
                    return None
        for key, info in (self._identified or {}).items():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            lx, ly = key[0], key[1]
            tag = getattr(info, "tag", None)
            tags = (tag,) if tag else (info if isinstance(info, (set, list, tuple)) else getattr(info, "tags", ()))
            if any(t in ("door", "staircase", "entrance", "exit") for t in (tags or ())):
                if (lx, ly) != (0, 0) and shortest_path(local_kinds, (0, 0), [(lx, ly)], people=self._people_now()) is not None:
                    return None
        last_move_dir = None
        for _, dep_req, _ in reversed(self._leg_departures):
            if dep_req.startswith("MoveOff("):
                last_move_dir = parse_edge_direction(dep_req[len("MoveOff("):-1])
                if last_move_dir in DIRECTIONS:
                    break
        if last_move_dir not in DIRECTIONS:
            return None
        cells = edge_cells(local_kinds, last_move_dir)
        if not cells:
            return None
        runs = edge_runs(cells, last_move_dir)
        if not any(self._run_pocket_bounds(last_move_dir, r) is None for r in runs):
            return None
        return (
            f"the target ladder/stairs/door has not been reached on the current screen while the {last_move_dir} "
            f"screen edge ({', '.join(str(c) for c in cells[:3])}) continues open and {remaining_calls} action "
            f"call(s) remain — continue along the open {last_move_dir} screen edge toward the target"
        )

    def _budget_exhausted_unfinished_subgoal_note(
        self,
        subgoal: str,
        judge_output: str,
        local_kinds: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> Optional[str]:
        if not self._out_of_calls():
            return None
        if not re.search(
            r"\b(?:action\s+budget|budget\s*\(\s*\d+\s*/\s*\d+\s*\)|exhausted|out\s+of\s+(?:action\s+)?calls|no\s+actions?\s+remaining|0\s+action\s+calls?\s+remaining)\b",
            judge_output or "",
            re.IGNORECASE,
        ):
            return None
        combined = f"{self._task or ''} {subgoal or ''}"
        gmap = self._global_map()
        if gmap is not None and re.search(r"\b(?:explore|reveal|sweep|unexplored)\b", combined, re.IGNORECASE):
            targets, _ = frontier_targets(gmap, local_kinds, blocked_moves=self._current_segment_blocked_moves())
            if targets:
                nearest = targets[0]
                return (
                    f"exhausting the episode action budget ({self._calls}/{self._max_steps}) does not complete an "
                    f"exploration/reveal subgoal while reachable unexplored map edges remain "
                    f"(nearest at ({nearest['dx']}, {nearest['dy']}), {nearest['steps']} steps {nearest['sector']})"
                )
        if re.search(r"\b(?:catch|catching|capture|capturing)\b", combined, re.IGNORECASE):
            if not re.search(r"\b(?:caught|captured|gotcha)\b", judge_output or "", re.IGNORECASE):
                return (
                    f"exhausting the episode action budget ({self._calls}/{self._max_steps}) without catching the "
                    f"target wild Pokemon does not complete the subgoal"
                )
        m_away = re.search(
            r"\b(\d+)\s+(?:moves?|steps?|tiles?)\s+(?:away\s+from|north\s+of|south\s+of|east\s+of|west\s+of|short\s+of)\s+"
            r"(?:the\s+)?[^.\n]{0,40}\b(?:door(?:way)?|entrance|center|mart|gym|house|cottage|ladder|stairs|staircase|gatehouse)\b",
            judge_output or "",
            re.IGNORECASE,
        )
        if m_away and int(m_away.group(1)) >= 2:
            return (
                f"exhausting the episode action budget ({self._calls}/{self._max_steps}) while still "
                f"{m_away.group(0)} does not complete the destination task"
            )
        return None

    def _unfinished_corridor_traversal_note(
        self,
        local_kinds: Optional[Dict[Tuple[int, int], str]] = None,
        *,
        subgoal: Optional[str] = None,
    ) -> Optional[str]:
        remaining_calls = max(0, self._max_steps - self._calls)
        if remaining_calls < 2:
            return None
        if self._agent_state() != AgentState.FREE_ROAM or self._is_in_dialogue():
            return None
        if self._last_seen_battle_hp is not None:
            cur_hp, max_hp = self._last_seen_battle_hp
            if max_hp > 0 and cur_hp * 2 <= max_hp:
                return None
        gmap = self._global_map()
        place = self._current_map_place()
        if gmap is None or place is None or self._start_place is None:
            return None
        cur_seg, cur_pos = place
        s_seg, s_pos = gmap.resolve(self._start_place[0], self._start_place[1])
        same_seg = cur_seg == s_seg
        cur_kind = gmap.info(cur_seg).kind
        start_kind = gmap.info(s_seg).kind
        if cur_kind == "building" or start_kind == "building":
            return None
        move_legs = 0
        net_steps = 0
        for b_place, _, a_place in self._leg_departures:
            try:
                b_seg, b_pos = gmap.resolve(b_place[0], b_place[1])
                a_seg, a_pos = gmap.resolve(a_place[0], a_place[1])
            except Exception:  # noqa: BLE001
                continue
            if (b_seg, b_pos) == (a_seg, a_pos):
                continue
            move_legs += 1
            if b_seg == a_seg:
                net_steps += abs(a_pos[0] - b_pos[0]) + abs(a_pos[1] - b_pos[1])
            else:
                net_steps += 5
        if same_seg:
            net_steps = abs(cur_pos[0] - s_pos[0]) + abs(cur_pos[1] - s_pos[1])
        if move_legs == 0:
            return None
        if re.search(
            r"\b(?:heal|potion|menu|poke\s*mart|pokemon\s*center|talk|speak|interact|pick\s+up|collect|cut|teach|learn|buy|sell)\b",
            self._task or "",
            re.IGNORECASE,
        ):
            return None
        is_ongoing_route_task = bool(
            re.search(
                r"\b(?:continue\s+(?:walking\s+|heading\s+)?(?:north|south|east|west|up|down|left|right|along|through|into)"
                r"|follow\s+the\s+.*\b(?:route|path|corridor|bridge)"
                r"|battl(?:e|ing)\s+(?:any\s+|the\s+)?trainers)\b",
                self._task or "",
                re.IGNORECASE,
            )
        )
        if move_legs >= 3 or net_steps >= 15:
            if not is_ongoing_route_task:
                return None
            if move_legs >= 6 and not re.search(
                r"\bbattl(?:e|ing)\s+(?:any\s+|the\s+)?trainers\b",
                self._task or "",
                re.IGNORECASE,
            ) and remaining_calls < 3:
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
        active_dirs = list(task_dirs)
        if subgoal and (move_legs >= 3 or len(task_dirs) > 1):
            sg_dirs = []
            for m in re.finditer(r"\b(north|south|east|west|up|down|left|right)\b", subgoal, re.IGNORECASE):
                d = dir_map[m.group(1).lower()]
                if d not in sg_dirs:
                    sg_dirs.append(d)
            if sg_dirs:
                active_dirs = sg_dirs
        if local_kinds is not None:
            for (lx, ly), kind in local_kinds.items():
                if abs(lx) + abs(ly) <= 6 and kind == "item":
                    return None
        targets, _ = frontier_targets(gmap, local_kinds, blocked_moves=self._current_segment_blocked_moves())
        doors_here = gmap.reachable_door_list(cur_seg)
        unentered_doors = [d for d in doors_here if d.get("leads_to") is None]
        for d in unentered_doors:
            for gx, gy in d.get("cells") or ():
                rel = (gx - cur_pos[0], gy - cur_pos[1])
                if in_bounds(rel) or (abs(rel[0]) + abs(rel[1]) <= 8):
                    return None
        if not targets and not unentered_doors:
            return None
        if local_kinds is not None:
            for key, info in (self._identified or {}).items():
                if not isinstance(key, tuple) or len(key) < 2:
                    continue
                lx, ly = key[0], key[1]
                tag = getattr(info, "tag", None)
                tags = (tag,) if tag else (info if isinstance(info, (set, list, tuple)) else getattr(info, "tags", ()))
                if any(t in ("door", "staircase", "entrance", "exit") for t in (tags or ())):
                    if (lx, ly) != (0, 0) and abs(lx) + abs(ly) <= 6:
                        return None
        open_task_dirs = []
        if local_kinds is not None:
            for d in active_dirs:
                cells = edge_cells(local_kinds, d)
                if not cells:
                    continue
                runs = edge_runs(cells, d)
                if any(self._run_pocket_bounds(d, r) is None for r in runs):
                    open_task_dirs.append(d)
        if not open_task_dirs:
            return None

        def _target_matches_dirs(t: Dict[str, Any], dirs: List[str]) -> bool:
            tdx, tdy = t.get("dx", 0), t.get("dy", 0)
            n_leg = t.get("next_leg") or ""
            for d in dirs:
                if is_on_edge((tdx, tdy), d) or n_leg.startswith(f"MoveOff({d} via "):
                    return True
                if d == "right" and tdx > 0:
                    return True
                if d == "left" and tdx < 0:
                    return True
                if d == "up" and tdy > 0:
                    return True
                if d == "down" and tdy < 0:
                    return True
            return False

        forward_targets = [t for t in targets if _target_matches_dirs(t, open_task_dirs)]
        if targets and not forward_targets and not unentered_doors:
            return None
        chosen_targets = forward_targets or targets
        if chosen_targets:
            nearest = chosen_targets[0]
            tdx, tdy = nearest.get("dx", 0), nearest.get("dy", 0)
            dist = nearest.get("path_steps") if nearest.get("path_steps") is not None else nearest.get("steps")
            on_open_edge = any(is_on_edge((tdx, tdy), d) for d in open_task_dirs)
            if (
                same_seg
                and dist is not None
                and dist <= 6
                and nearest.get("on_screen")
                and not on_open_edge
                and not any(
                    not t.get("on_screen") or any(is_on_edge((t.get("dx", 0), t.get("dy", 0)), d) for d in open_task_dirs)
                    for t in chosen_targets
                )
            ):
                return None
            next_leg = nearest.get("next_leg")
            if on_open_edge:
                for d in open_task_dirs:
                    if is_on_edge((tdx, tdy), d):
                        next_leg = f"MoveOff({d} via ({tdx}, {tdy}))"
                        break
            leg_str = f" [next on-screen leg: {next_leg}]" if next_leg else ""
            route_info = (
                f"nearest unexplored edge is at ({tdx}, {tdy}), {nearest['steps']} steps {nearest['sector']}"
                + (f" (mapped route, {nearest['path_steps']} moves: {nearest['path_moves']}{leg_str})" if nearest.get("path_moves") else leg_str)
            )
        else:
            route_info = f"{len(unentered_doors)} unentered door(s)/stairs remain in this area"
        span_desc = f"from {s_pos} to {cur_pos}" if same_seg else f"to {cur_pos} across area boundary"
        return (
            f"only {move_legs} movement leg(s) ({net_steps} net steps {span_desc}) completed so far "
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
            unfinished_note = self._unfinished_corridor_traversal_note(
                local_kinds,
                subgoal=self.plan[-1] if self.plan else None,
            )
        if unfinished_note is None and self.plan:
            unfinished_note = self._unfinished_open_corridor_target_note(self.plan[-1], local_kinds)
        return unfinished_note

    def _unreached_task_destination_note(
        self,
        check_output: str,
        local_kinds: Optional[Dict[Tuple[int, int], str]] = None,
    ) -> Optional[str]:
        final_subgoal = self.plan[-1] if self.plan else ""
        budget_note = self._budget_exhausted_unfinished_subgoal_note(final_subgoal, check_output, local_kinds)
        if budget_note is not None:
            return budget_note
        m_away = re.search(
            r"\b(\d+)\s+(?:moves?|steps?|tiles?)\s+(?:away\s+from|north\s+of|south\s+of|east\s+of|west\s+of|short\s+of)\s+"
            r"(?:the\s+)?[^.\n]{0,40}\b(?:door(?:way)?|entrance|center|mart|gym|house|cottage|ladder|stairs|staircase|gatehouse)\b",
            check_output or "",
            re.IGNORECASE,
        )
        if m_away and int(m_away.group(1)) >= 2:
            return f"the player is still {m_away.group(0)} and has not reached the task's destination"
        return None

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
                "action calls for extra forward movement, and do not answer yes if a specific destination doorway, building "
                "entrance, ladder, or staircase named in \"Your task\" has not yet been reached."
            )
        prompt = self._fill_prompt(PLAYTHROUGH_TASK_CHECK_PROMPT).replace("[CONTINUATION_NOTE]", continuation_note)
        output = self._vlm_call("task_check", texts=prompt, images=[frame])
        summary = (parse_key_value(output, "Summary") or "").strip()
        complete = bool(parse_yes_no(output, "Task complete"))
        if complete and unfinished_note is None:
            unfinished_note = self._unreached_task_destination_note(f"{output}\n{summary}", local_kinds)
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
        self._scan_frame_for_party_move_status(frame)
        state = self._effective_state(frame).name.lower().replace("_", " ")
        hp_line = self._party_hp_block()
        inst_text = f"{instruction}\n({hp_line})" if hp_line else instruction
        prompt = (self._fill_prompt(PLAYTHROUGH_MENU_OBJECTIVE_PROMPT)
                  .replace("[INSTRUCTION]", inst_text)
                  .replace("[AGENT_STATE]", state))
        output = self._vlm_call("menu_objective", texts=prompt, images=[frame])
        objective = parse_key_value(output, "Objective")
        if not objective or not objective.strip():
            log_warn(f"Could not parse a menu objective, using the instruction as given: {output!r}", self._parameters)
            objective = instruction
        return objective.strip().strip("\"'"), bool(parse_yes_no(output, "Start menu"))

    def _tips_text(self) -> str:
        return "\n".join(f"- {tip}" for tip in self._battle_tips)

    MAX_BATTLE_INTRO_SETTLE_TICKS = 10

    def _settle_battle_intro_frame(
        self,
        frame,
        pending_crops: Optional[List[Any]] = None,
    ) -> Tuple[Any, Optional[List[Any]]]:
        crops = list(pending_crops) if pending_crops else []
        for _ in range(self.MAX_BATTLE_INTRO_SETTLE_TICKS):
            if (
                self._env_done()
                or self._agent_state() != AgentState.IN_BATTLE
                or not is_unrevealed_battle_intro_frame(frame)
            ):
                break
            _, frame = self._step_wait_action()
            crop = extract_bottom_dialogue_crop(frame)
            if crop is not None:
                drop_prefix_crops([crop], crops)
        crop = extract_bottom_dialogue_crop(frame)
        if crop is not None:
            drop_prefix_crops([crop], crops)
        return frame, (crops if crops else pending_crops)

    def _battle_instructions(self, frame) -> str:
        if self._agent_state() == AgentState.IN_BATTLE and is_unrevealed_battle_intro_frame(frame):
            frame, _ = self._settle_battle_intro_frame(frame)
        self._scan_frame_for_party_move_status(frame)
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

    @classmethod
    def _is_wild_encounter_crops(cls, crops: Optional[List[Any]], current_frame=None) -> bool:
        from utils.vlm import cache_lookup_ocr

        if crops:
            cached = [cache_lookup_ocr(c) for c in crops]
            joined = "\n".join(t for t in cached if t)
            if joined and cls._is_wild_encounter_dialogue(joined):
                return True
        for candidate in list(crops or ()) + ([current_frame] if current_frame is not None else []):
            if candidate is None or not hasattr(candidate, "shape"):
                continue
            h = candidate.shape[0]
            if h >= 144:
                row_text = read_tile_text(candidate, 14, 1, 19)
                full_str = "".join(
                    read_tile_text(candidate, 14, tx, tx + 1) or " "
                    for tx in range(1, 19)
                )
            elif h >= 32:
                row_text = read_tile_text(candidate, 1, 0, 18)
                full_str = "".join(
                    read_tile_text(candidate, 1, tx, tx + 1) or " "
                    for tx in range(0, min(18, candidate.shape[1] // 8))
                )
            else:
                continue
            if row_text.startswith("WILD") or (
                row_text == "W" and len(full_str) >= 6 and full_str[0] == "W" and full_str[4] == " " and full_str[5].isalpha()
            ):
                return True
        return False

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
        if self._zero_pp_moves and "initial_zero_pp_moves" not in executor_kwargs:
            executor_kwargs["initial_zero_pp_moves"] = list(self._zero_pp_moves)
        if (
            self._healing_bag_empty
            and (not self._active_pokemon_status or self._status_heal_bag_empty)
            and "initial_bag_unavailable" not in executor_kwargs
        ):
            executor_kwargs["initial_bag_unavailable"] = True
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
        *,
        allow_async_dialogue: bool = False,
        current_argument: Optional[str] = None,
    ) -> Tuple[bool, Optional[str], bool]:
        if (
            not handled.get("succeeded")
            or self._agent_state() != AgentState.FREE_ROAM
            or self._is_in_dialogue()
            or self._battles_lost()
            or self._env_done()
        ):
            return False, None, False
        is_wild_battle_clear = (
            handled.get("reason") == "battle_ended"
            and "in_battle" in state_chain
            and self._is_wild_encounter_dialogue(handled.get("dialogue"))
            and not self._task_wants_wild_battles()
            and self._active_pokemon_hp_healthy()
        )
        is_async_dialogue_clear = (
            allow_async_dialogue
            and handled.get("reason") == "dialogue_cleared"
            and "in_battle" not in state_chain
            and "in_menu" not in state_chain
        )
        if not (is_wild_battle_clear or is_async_dialogue_clear):
            return False, None, False
        if action == "Free":
            resumed_arg = current_argument if current_argument is not None else current_task
            if not resumed_arg:
                return False, None, False
            gmap = self._global_map()
            if gmap is not None and leg_start_place is not None and after_place is not None:
                s_seg, _ = gmap.resolve(leg_start_place[0], leg_start_place[1])
                a_seg, _ = gmap.resolve(after_place[0], after_place[1])
                if s_seg != a_seg:
                    return False, None, True
            return True, resumed_arg, False
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
            frame_now = self._current_frame()
            if self._effective_state(frame_now) != AgentState.FREE_ROAM or self._is_in_dialogue(frame_now):
                return False, None, False
            self._perceive(frame_now)
            kinds_now = self._current_local_kinds() or {}
            cells_now = edge_cells(kinds_now, direction)
            if not cells_now:
                return False, None, False
            people_now = self._people_now()
            warps_now = warp_cells(identified=self._identified, global_map=gmap)
            coords = [c for c in parse_all_cells(current_task) if in_bounds(c)]
            if len(coords) == 1 and (dx, dy) != (0, 0):
                old_c = coords[0]
                new_c = (old_c[0] - dx, old_c[1] - dy)
                if (
                    is_on_edge(old_c, direction)
                    and in_bounds(new_c)
                    and is_on_edge(new_c, direction)
                    and new_c in cells_now
                    and shortest_path(kinds_now, (0, 0), [new_c], avoid_cells=warps_now, people=people_now) is not None
                ):
                    return True, f"{direction} via {new_c}", False
            elif len(coords) == 1 and (dx, dy) == (0, 0):
                old_c = coords[0]
                if (
                    old_c in cells_now
                    and shortest_path(kinds_now, (0, 0), [old_c], avoid_cells=warps_now, people=people_now) is not None
                ):
                    return True, current_task, False
            best_c = self._best_open_edge_target(direction, kinds_now)
            if (
                best_c is not None
                and shortest_path(kinds_now, (0, 0), [best_c], avoid_cells=warps_now, people=people_now) is not None
            ):
                return True, f"{direction} via {best_c}", False
            return False, None, False
        if action == "MoveTo":
            coords = [c for c in parse_all_cells(current_task) if in_bounds(c)]
            if len(coords) == 1:
                old_c = coords[0]
                new_c = (old_c[0] - dx, old_c[1] - dy)
                wants_warp = bool(
                    re.search(
                        r"\b(?:ladder|stair(?:s|case)?|door(?:way)?|entrance|exit|elevator|warp)\b",
                        current_task,
                        re.IGNORECASE,
                    )
                )
                if wants_warp:
                    frame_now = self._current_frame()
                    self._perceive(frame_now)
                    kinds_now = self._current_local_kinds() or {}
                    warps_now = warp_cells(identified=self._identified, global_map=gmap) - {(0, 0)}
                    if new_c not in warps_now:
                        nearby_warps = sorted(
                            (
                                abs(wx - new_c[0]) + abs(wy - new_c[1]),
                                abs(wx) + abs(wy),
                                (wx, wy),
                            )
                            for (wx, wy) in warps_now
                            if abs(wx - new_c[0]) + abs(wy - new_c[1]) <= 2
                            and shortest_path(kinds_now, (0, 0), [(wx, wy)], people=self._people_now()) is not None
                        )
                        if nearby_warps:
                            new_c = nearby_warps[0][2]
                        elif new_c == (0, 0):
                            return False, None, False
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
                               self._local_blocked_cells(), global_map=self._global_map())
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
                           self._local_blocked_cells(), global_map=self._global_map())
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
            self._scan_frame_for_party_move_status(frame)
            raw_cur_state = self._agent_state()
            state = self._effective_state(frame, raw_state=raw_cur_state)
            if (
                raw_cur_state == AgentState.IN_MENU
                and state == AgentState.IN_BATTLE
                and not active_battle_instructions
                and (is_overworld_menu_or_ui_screen(frame) or not is_battle_screen(frame))
            ):
                state = AgentState.IN_MENU
            if state == AgentState.FREE_ROAM and combined is not None:
                break
            state_label = state.name.lower()
            if not states_seen or states_seen[-1] != state_label:
                states_seen.append(state_label)
            if state == AgentState.IN_DIALOGUE:
                task = dialogue_task if step_idx == 0 else "clear the dialogue that followed"
                handled = self.call_executor("dialogue", task, counted=False,
                                             initial_dialogue_crops=pending_crops,
                                             defer_ocr_on_battle=True)
                diag_crops = getattr(self.last_executor, "dialogue_crops", None)
                pending_crops = list(diag_crops) if diag_crops else None
                if handled.get("dialogue"):
                    pre_battle_dialogue = (
                        f"{pre_battle_dialogue}\n{handled['dialogue']}"
                        if pre_battle_dialogue
                        else str(handled["dialogue"])
                    )
            elif state == AgentState.IN_BATTLE:
                if is_unrevealed_battle_intro_frame(frame):
                    frame, pending_crops = self._settle_battle_intro_frame(frame, pending_crops)
                is_wild_flee = (
                    (
                        self._is_wild_encounter_dialogue(pre_battle_dialogue)
                        or self._is_wild_encounter_crops(pending_crops, frame)
                    )
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
            self._scan_frame_for_party_move_status(next_frame)
            next_state = self._effective_state(next_frame)
            if next_state == AgentState.FREE_ROAM:
                break
            if (
                not handled.get("succeeded")
                and next_state == state
                and handled.get("reason") not in ("left_battle_to_menu", "left_battle_to_dialogue")
            ):
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
        raw = handled.get("dialogue") or ""
        if not raw:
            return ""
        non_press_lines = [
            ln.strip()
            for ln in str(raw).splitlines()
            if ln.strip() and not ln.strip().lower().startswith("presses:")
        ]
        text = " ".join(" ".join(non_press_lines if non_press_lines else [str(raw)]).split())
        return f": \"{text[:limit]}\"" if text else ""

    def _clear_pending_dialogue(self, attempts: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        self._sync_dialogue_failure_state()
        if not self._is_in_dialogue() or self._dialogue_clear_failed or self._out_of_calls() or self._env_done():
            return None
        state_chain, handled = self._handle_interruption(dialogue_task="clear the dialogue on screen")
        label = "open dialogue" if state_chain == "in_dialogue" else f"open dialogue ({state_chain})"
        prefix = "Cleared" if handled.get("succeeded") else "Could not clear"
        note = f"{prefix} {label} ({handled['reason']}){self._dialogue_snippet(handled)}"
        self._note(note)
        if attempts is not None:
            attempts.append(note)
        return handled

    def _clear_pending_interruption(self, attempts: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
        self._sync_dialogue_failure_state()
        if self._out_of_calls() or self._env_done():
            return None
        if not (self._is_in_dialogue() or self._effective_state() != AgentState.FREE_ROAM):
            return None
        frame = self._current_frame()
        if self._is_in_dialogue(frame) and self._dialogue_clear_failed:
            return None
        self._perceive(frame)
        if self._is_in_dialogue(frame):
            return self._clear_pending_dialogue(attempts)
        state_chain, handled = self._handle_interruption()
        prefix = "Cleared" if handled.get("succeeded") else "Could not clear"
        note = f"{prefix} pending {state_chain} ({handled['reason']}){self._dialogue_snippet(handled)}"
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

    MAX_DETERMINISTIC_CORRIDOR_CHAIN = 3

    @staticmethod
    def _normalize_ascii_text(text: Optional[str]) -> str:
        import unicodedata

        return unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii").lower()

    def _projected_door_route_next_leg(
        self,
        gmap,
        a_seg: int,
        a_pos: Tuple[int, int],
        after_kinds: Dict[Tuple[int, int], str],
        subgoal: str,
    ) -> Optional[str]:
        from execution.strategist.pokemon.global_map.routes import _snap, path_in_segment, walked_squares
        from execution.strategist.pokemon.global_map.text import project_route_next_leg

        doors = gmap.reachable_door_list(a_seg)
        if not doors:
            return None
        plan_text = " ".join(self.plan) if self.plan else ""
        raw_context = f"{self._task or ''} {self._guidance or ''} {subgoal or ''} {plan_text}"
        combined = self._normalize_ascii_text(raw_context)
        subgoal_and_task = self._normalize_ascii_text(f"{self._task or ''} {subgoal or ''} {plan_text}")
        mentioned_coords = set(parse_all_cells(raw_context))
        start_seg = gmap.resolve(self._start_place[0], self._start_place[1])[0] if self._start_place is not None else None
        start_pos = gmap.resolve(self._start_place[0], self._start_place[1])[1] if self._start_place is not None else None
        seg_blocked = self._current_segment_blocked_moves()
        walked = walked_squares(gmap, a_seg)
        candidates: List[Tuple[int, int, str]] = []
        stop_words = {"area", "the", "and", "overworld", "interior", "floor"}
        for door in doors:
            leads_to = door.get("leads_to")
            if leads_to is not None and gmap.resolve(leads_to, (0, 0))[0] == a_seg:
                continue
            goal_cell = min(door["cells"], key=lambda g: abs(g[0] - a_pos[0]) + abs(g[1] - a_pos[1]))
            if start_seg == a_seg and start_pos is not None and abs(goal_cell[0] - start_pos[0]) + abs(goal_cell[1] - start_pos[1]) <= 2:
                continue
            dx, dy = goal_cell[0] - a_pos[0], goal_cell[1] - a_pos[1]
            if in_bounds((dx, dy)) or abs(dx) + abs(dy) <= 6:
                continue
            rel_start = (
                (goal_cell[0] - start_pos[0], goal_cell[1] - start_pos[1])
                if (start_seg == a_seg and start_pos is not None)
                else None
            )
            coord_matched = (
                (dx, dy) in mentioned_coords
                or (rel_start is not None and rel_start in mentioned_coords)
                or goal_cell in mentioned_coords
            )
            priority = 4
            if coord_matched:
                priority = 0
            elif leads_to is None:
                priority = 2
            else:
                try:
                    label = self._normalize_ascii_text(gmap.area_label(leads_to))
                    primary_label = label.split("(")[0].strip()
                    primary_words = [
                        w for w in re.findall(r"[a-z0-9]+", primary_label)
                        if len(w) >= 3 and w not in stop_words
                    ]
                    all_words = [
                        w for w in re.findall(r"[a-z0-9]+", label)
                        if len(w) >= 3 and w not in stop_words
                    ]
                    if primary_words and all(w in subgoal_and_task for w in primary_words):
                        priority = 1
                    elif primary_words and any(w in subgoal_and_task for w in primary_words):
                        priority = 2
                    elif all_words and any(w in combined for w in all_words):
                        priority = 3
                    elif any(k in combined for k in ("gatehouse", "door", "entrance", "exit", "route", "town", "city")):
                        priority = 3
                except Exception:  # noqa: BLE001
                    pass
            if priority >= 4:
                continue
            snapped = _snap(gmap, a_seg, goal_cell, walked)
            if snapped is None:
                continue
            found = path_in_segment(gmap, a_seg, a_pos, snapped, walked, blocked_moves=seg_blocked)
            if found is None or not found[1] or len(found[1]) <= 6:
                continue
            n_leg = project_route_next_leg(gmap, found[1], local_kinds=after_kinds, goal_rel=(dx, dy))
            if n_leg and n_leg.startswith("MoveOff("):
                candidates.append((priority, len(found[1]), n_leg[len("MoveOff("):-1]))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        if len(candidates) > 1 and candidates[0][0] == candidates[1][0] and candidates[0][2] != candidates[1][2]:
            return None
        return candidates[0][2]

    def _try_deterministic_corridor_continuation(
        self,
        *,
        subgoal: str,
        subgoal_index: int,
        action: str,
        task: str,
        request: str,
        result: Dict[str, Any],
        before_place: Optional[Tuple[int, Tuple[int, int]]],
        after_place: Optional[Tuple[int, Tuple[int, int]]],
        move_tag: str,
        made_progress: bool,
        chain_count: int = 0,
    ) -> Optional[Tuple[str, Optional[Tuple[str, str]], str]]:
        if (
            action != "MoveOff"
            or not result.get("succeeded")
            or result.get("reason") != "left_screen"
            or not made_progress
            or chain_count >= self.MAX_DETERMINISTIC_CORRIDOR_CHAIN
            or self._out_of_calls()
            or max(0, self._max_steps - self._calls) < 1
        ):
            return None
        if self._agent_state() != AgentState.FREE_ROAM or self._is_in_dialogue():
            return None
        if self._last_seen_battle_hp is not None:
            cur_hp, max_hp = self._last_seen_battle_hp
            if max_hp > 0 and cur_hp * 2 <= max_hp and not self._healing_bag_empty:
                return None
        gmap = self._global_map()
        if gmap is None or before_place is None or after_place is None:
            return None
        b_seg, b_pos = gmap.resolve(before_place[0], before_place[1])
        a_seg, a_pos = gmap.resolve(after_place[0], after_place[1])
        if b_seg != a_seg:
            return None
        notes_str = str(result.get("notes") or "")
        if "further movement" in notes_str.lower() and "blocked" in notes_str.lower():
            return None
        move_dir = parse_edge_direction(task)
        if move_dir not in DIRECTIONS:
            return None

        after_frame = self._current_frame()
        if self._effective_state(after_frame) != AgentState.FREE_ROAM or self._is_in_dialogue(after_frame):
            return None
        self._tile_recognizer.record_tiles(after_frame, self._episode_number)
        gmap.refresh(self._tile_recognizer._by_hash)
        after_identified = self._tile_recognizer.identify_tiles(after_frame, global_map=gmap)
        self._sync_screen_to_global_map(after_frame, after_identified, gmap)
        self._identified = after_identified
        after_people = self._people_now()
        after_kinds = cell_kinds(
            after_identified,
            getattr(self._env, "ledge_patterns", None),
            getattr(self._env, "walk_patterns", None),
            after_people,
            getattr(self._env, "left_ledge_patterns", None),
            self._local_blocked_cells(),
            global_map=gmap,
        )
        combined_goal_text = f"{self._task or ''} {subgoal or ''}"
        wants_item_pickup = bool(
            re.search(
                r"\b(?:item|poke\s*ball|pick\s+up|collect|ether|potion|antidote|tm\d*|hm\d*)\b",
                combined_goal_text,
                re.IGNORECASE,
            )
        )
        seg_map = gmap.segments[a_seg] if 0 <= a_seg < len(gmap.segments) else {}
        for (lx, ly), kind in (after_kinds or {}).items():
            if kind != "item":
                continue
            dist_item = abs(lx) + abs(ly)
            if dist_item <= 2:
                return None
            if dist_item <= 6 and wants_item_pickup:
                return None
            if dist_item <= 4:
                sp_item = shortest_path(after_kinds, (0, 0), [(lx, ly)], people=after_people)
                if sp_item is not None:
                    crosses_grass = any(
                        "tall grass" in getattr(seg_map.get((a_pos[0] + px_s, a_pos[1] + py_s)), "tags", ())
                        for _, _, (px_s, py_s) in sp_item
                    )
                    if not crosses_grass:
                        return None
        for key, info in (after_identified or {}).items():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            lx, ly = key[0], key[1]
            tag = getattr(info, "tag", None)
            tags = (tag,) if tag else (info if isinstance(info, (set, list, tuple)) else getattr(info, "tags", ()))
            if any(t in ("door", "staircase", "entrance", "exit") for t in (tags or ())):
                if (lx, ly) != (0, 0) and shortest_path(after_kinds, (0, 0), [(lx, ly)], people=after_people) is not None:
                    return None
        for door in gmap.reachable_door_list(a_seg):
            if door.get("leads_to") is None:
                for gx, gy in door.get("cells", ()):
                    rel = (gx - a_pos[0], gy - a_pos[1])
                    if in_bounds(rel) and rel != (0, 0):
                        return None

        dir_aliases = {
            "north": "up", "up": "up", "top": "up",
            "south": "down", "down": "down", "bottom": "down",
            "west": "left", "left": "left",
            "east": "right", "right": "right",
            "northeast": ("up", "right"), "north-east": ("up", "right"),
            "northwest": ("up", "left"), "north-west": ("up", "left"),
            "southeast": ("down", "right"), "south-east": ("down", "right"),
            "southwest": ("down", "left"), "south-west": ("down", "left"),
        }
        subgoal_dirs: List[str] = []
        for m in re.finditer(
            r"\b(north-?east|north-?west|south-?east|south-?west|north|south|east|west|up|down|left|right|top|bottom)\b",
            subgoal or "",
            re.IGNORECASE,
        ):
            mapped_d = dir_aliases[m.group(1).lower()]
            for d in (mapped_d if isinstance(mapped_d, tuple) else (mapped_d,)):
                if d not in subgoal_dirs:
                    subgoal_dirs.append(d)
        is_corridor_subgoal = bool(
            re.search(
                r"\b(?:corridor|path|route|bridge|tunnel|hallway|passage|lane|street|road|forest|cave|gym"
                r"|continue[ds]?|follow(?:ed|ing)?|deeper|along|through|around|toward|until|past|winding)\b",
                subgoal or "",
                re.IGNORECASE,
            )
        )
        if not subgoal_dirs and not is_corridor_subgoal:
            return None

        single_step_only = bool(
            re.search(
                r"\b(?:walked|stepped|moved)\s+(?:north|south|east|west|up|down|left|right)\s+off\s+the\s+"
                r"(?:top|bottom|left|right|up|down|upper|lower)\s+(?:screen\s+)?edge\b",
                subgoal or "",
                re.IGNORECASE,
            )
            and not re.search(
                r"\b(?:continue[ds]?|all\s+the\s+way|along|middle|end|toward|until|past|through|corridor|bridge|route|around|bend|corner|turn)\b",
                subgoal or "",
                re.IGNORECASE,
            )
        )
        if single_step_only:
            return None

        if re.search(
            r"\b(?:bend|corner|turn(?:s|ed|ing)?|junction|intersection)\b",
            subgoal or "",
            re.IGNORECASE,
        ) and not re.search(
            r"\b(?:around\s+the\s+(?:bend|corner|pond|wall|trees)|past\s+the\s+(?:bend|corner))\b",
            subgoal or "",
            re.IGNORECASE,
        ):
            check_cells = edge_cells(after_kinds, move_dir)
            check_runs = edge_runs(check_cells, move_dir) if check_cells else []
            if not any(self._run_pocket_bounds(move_dir, r) is None for r in check_runs):
                return None

        opposite_dir = {"up": "down", "down": "up", "left": "right", "right": "left"}[move_dir]
        open_runs_by_dir: Dict[str, List[List[Tuple[int, int]]]] = {}
        for d_cand in DIRECTIONS:
            d_cells = edge_cells(after_kinds, d_cand)
            if not d_cells:
                continue
            d_runs = edge_runs(d_cells, d_cand)
            d_open = [r for r in d_runs if self._run_pocket_bounds(d_cand, r) is None]
            if d_open:
                open_runs_by_dir[d_cand] = d_open

        if not open_runs_by_dir:
            return None

        targets, _ = frontier_targets(gmap, after_kinds, blocked_moves=self._current_segment_blocked_moves())
        if any(t.get("on_screen") and (t.get("path_steps") or t.get("steps") or 0) <= 4 for t in targets):
            return None

        next_dir: Optional[str] = None
        mapped_next_arg: Optional[str] = None

        wants_frontier_exploration = bool(
            re.search(r"\b(?:explore|reveal|sweep|unexplored)\b", combined_goal_text, re.IGNORECASE)
        )
        effective_subgoal_dirs = list(subgoal_dirs)
        if (
            len(effective_subgoal_dirs) > 1
            and move_dir in open_runs_by_dir
            and move_dir in effective_subgoal_dirs
            and effective_subgoal_dirs[-1] == move_dir
        ):
            effective_subgoal_dirs = [move_dir]

        door_arg = self._projected_door_route_next_leg(gmap, a_seg, a_pos, after_kinds, subgoal)
        if door_arg is not None and not wants_frontier_exploration:
            cand_dir = parse_edge_direction(door_arg)
            if (
                cand_dir in open_runs_by_dir
                and cand_dir != opposite_dir
                and (not effective_subgoal_dirs or cand_dir in effective_subgoal_dirs)
            ):
                next_dir = cand_dir
                mapped_next_arg = door_arg

        if mapped_next_arg is None:
            for t in targets:
                n_leg = t.get("next_leg") or ""
                dist = t.get("path_steps") or t.get("steps") or 0
                if not n_leg.startswith("MoveOff(") or dist <= 6 or t.get("on_screen"):
                    continue
                cand_arg = n_leg[len("MoveOff("):-1]
                cand_dir = parse_edge_direction(cand_arg)
                if (
                    cand_dir in open_runs_by_dir
                    and cand_dir != opposite_dir
                    and (not effective_subgoal_dirs or cand_dir in effective_subgoal_dirs)
                ):
                    next_dir = cand_dir
                    mapped_next_arg = cand_arg
                    break

        if mapped_next_arg is None and door_arg is not None:
            cand_dir = parse_edge_direction(door_arg)
            if (
                cand_dir in open_runs_by_dir
                and cand_dir != opposite_dir
                and (not effective_subgoal_dirs or cand_dir in effective_subgoal_dirs)
            ):
                next_dir = cand_dir
                mapped_next_arg = door_arg

        if mapped_next_arg is None and move_dir in open_runs_by_dir:
            open_runs = open_runs_by_dir[move_dir]
            if (not subgoal_dirs or move_dir in subgoal_dirs) and (len(open_runs) == 1 or bool(targets)):
                best_target = self._best_open_edge_target(move_dir, after_kinds)
                if best_target is not None:
                    next_dir = move_dir
                    mapped_next_arg = f"{move_dir} via {best_target}"

        if mapped_next_arg is None and move_dir not in open_runs_by_dir:
            perp_open_dirs = [
                d_cand for d_cand in DIRECTIONS
                if d_cand not in (move_dir, opposite_dir) and d_cand in open_runs_by_dir
            ]
            if len(perp_open_dirs) == 1:
                turn_dir = perp_open_dirs[0]
                turn_runs = open_runs_by_dir[turn_dir]
                if (not subgoal_dirs or turn_dir in subgoal_dirs) and len(turn_runs) == 1:
                    best_target = self._best_open_edge_target(turn_dir, after_kinds)
                    if best_target is not None:
                        next_dir = turn_dir
                        mapped_next_arg = f"{turn_dir} via {best_target}"

        if mapped_next_arg is None or next_dir is None:
            return None

        dx, dy = a_pos[0] - b_pos[0], a_pos[1] - b_pos[1]
        steps_taken = abs(dx) + abs(dy)
        cardinal_name = {"up": "north", "down": "south", "left": "west", "right": "east"}[move_dir]
        if next_dir == move_dir:
            cont_clause = f"the {move_dir} corridor continues open via {mapped_next_arg}"
        else:
            cont_clause = f"the corridor turns {next_dir} via {mapped_next_arg}"
        summary = (
            f"The player walked {steps_taken} step{'s' if steps_taken != 1 else ''} {cardinal_name} "
            f"from {b_pos} to {a_pos} ({task}); {cont_clause}."
        )
        next_reasoning = (
            f"The player advanced to {a_pos} in free roam and {cont_clause} "
            f"toward the subgoal destination."
        )
        return summary, ("MoveOff", mapped_next_arg), next_reasoning

    def _pursue(self, subgoal: str, subgoal_index: int = 0) -> Tuple[str, List[str], int]:
        attempts: List[str] = []
        subgoal_places: List[Tuple[int, Tuple[int, int]]] = []
        start_subgoal_place = self._current_map_place()
        if start_subgoal_place is not None:
            subgoal_places.append(start_subgoal_place)
        failed_attempts = 0
        iterations = 0
        det_chain_count = 0
        queued_decision: Optional[Tuple[Tuple[str, Optional[str]], str]] = None
        while failed_attempts < self.MAX_SUBGOAL_ATTEMPTS and iterations < self.MAX_SUBGOAL_ITERATIONS:
            if self._env_done():
                return "env_done", attempts, subgoal_index
            if self._out_of_calls():
                return "out_of_calls", attempts, subgoal_index
            if self._total_calls > 0:
                self._settle_free_roam()
            self._sync_dialogue_failure_state()
            in_dialogue_now = self._is_in_dialogue()
            if (in_dialogue_now and not self._dialogue_clear_failed) or (
                not in_dialogue_now and self._effective_state() != AgentState.FREE_ROAM
            ):
                queued_decision = None
                before = self._current_frame()
                self._perceive(before)
                if self._is_in_dialogue(before):
                    handled = self._clear_pending_dialogue(attempts)
                    request_label = "Dialogue(clear open dialogue)"
                else:
                    state_chain, handled = self._handle_interruption()
                    prefix = "Cleared" if handled.get("succeeded") else "Could not clear"
                    note = f"{prefix} pending {state_chain} ({handled['reason']}){self._dialogue_snippet(handled)}"
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
            before_effective_state = self._effective_state(frame)
            before_place = self._current_map_place()
            if before_place is not None and (not subgoal_places or subgoal_places[-1] != before_place):
                subgoal_places.append(before_place)
            prior_stops = list(self._leg_stops)
            if queued_decision is not None:
                decision, decision_reasoning = queued_decision
                queued_decision = None
            else:
                decision, decision_reasoning = self._decide(subgoal, attempts, frame)
            if decision is None:
                attempts.append("the reply could not be read, so nothing was done")
                failed_attempts += 1
                continue
            action, argument = decision
            is_move_action = ACTIONS[action][0] in MOVE_TAGS
            leg_initial_state = None
            result, task = self._run_action(action, argument, frame)
            if action == "Free" and result.get("reason") == "move_interrupted":
                is_move_action = True
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
            current_argument = argument if action == "Free" else task
            while (
                is_move_action
                and not result["succeeded"]
                and (
                    (self._is_in_dialogue() and not self._dialogue_clear_failed)
                    or (not self._is_in_dialogue() and self._effective_state() != AgentState.FREE_ROAM)
                )
                and wild_resumes < self.MAX_WILD_MOVE_RESUMES
                and not self._env_done()
                and not self._battles_lost()
                and self._total_calls < self._max_total_calls
            ):
                if self._calls > 0:
                    self._calls -= 1
                interrupt_frame = self._current_frame()
                was_top_box_interrupt = (
                    self._agent_state() == AgentState.IN_MENU
                    and has_top_dialogue_box(interrupt_frame)
                    and not has_menu_cursor(interrupt_frame)
                )
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
                    action,
                    current_task,
                    leg_start_place,
                    after_place,
                    state,
                    handled,
                    allow_async_dialogue=bool(
                        not exec_blocked
                        and (was_top_box_interrupt or (action == "Free" and result.get("reason") == "move_interrupted"))
                    ),
                    current_argument=current_argument,
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
                if action == "Free":
                    current_argument = resumed_task
                    result, current_task = self._run_action(action, current_argument, resume_frame)
                else:
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
                    and (self._is_in_dialogue() or self._effective_state() != AgentState.FREE_ROAM)
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

            after_in_dialogue = self._is_in_dialogue()
            should_handle_interruption = (
                not result["succeeded"]
                and (self._effective_state() != AgentState.FREE_ROAM or after_in_dialogue)
                and self._total_calls < self._max_total_calls
                and not (
                    after_in_dialogue
                    and (
                        action == "Dialogue"
                        or before_effective_state == AgentState.IN_DIALOGUE
                        or self._dialogue_clear_failed
                    )
                )
            )
            if should_handle_interruption:
                if is_move_action and self._calls > 0:
                    self._calls -= 1
                seed_crops = getattr(self.last_executor, "dialogue_crops", None) if is_move_action else None
                seed_battle_instructions = task if (action == "Battle" and result.get("reason") == "left_battle_to_menu") else None
                state, handled = self._handle_interruption(
                    initial_dialogue_crops=seed_crops,
                    initial_battle_instructions=seed_battle_instructions,
                )
                combined_text = "\n".join(filter(None, [result.get("dialogue"), handled.get("dialogue")]))
                if combined_text:
                    handled["dialogue"] = combined_text
                combined_notes = "\n".join(filter(None, [result.get("notes"), handled.get("notes")]))
                if combined_notes:
                    handled["notes"] = combined_notes
                    handled["report"].notes = combined_notes
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
                if self._is_dialogue_unrecoverable():
                    return "stuck", attempts, subgoal_index
                continue

            if (
                not result["succeeded"]
                and is_move_action
                and before_place is not None
                and after_place is not None
                and before_place[0] != after_place[0]
            ):
                result["succeeded"] = True
                result["reason"] = "arrived"

            if is_move_action and before_place is not None and after_place is not None and before_place != after_place:
                self._leg_departures.append((before_place, request, after_place))

            if not result["succeeded"]:
                if is_move_action and not result["report"].steps and self._calls > 0:
                    self._calls -= 1
                if (
                    result["reason"] in ("move_blocked", "out_of_steps")
                    and before_effective_state == AgentState.FREE_ROAM
                    and not after_in_dialogue
                    and self._agent_state() == AgentState.FREE_ROAM
                ):
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
                note_detail = " ".join(str(result.get("notes") or "").split())
                note = f"{request} did not work: {result['reason']}" + (
                    f" ({note_detail[:160]})" if note_detail else "")
                if move_tag:
                    note = f"{move_tag} {note}"
                self._note(note)
                attempts.append(note)
                failed_attempts += 1
                if self._is_dialogue_unrecoverable():
                    return "stuck", attempts, subgoal_index
                continue

            made_progress = is_move_action and self._made_new_subgoal_progress(before_place, after_place, subgoal_places)
            det_cont = (
                self._try_deterministic_corridor_continuation(
                    subgoal=subgoal,
                    subgoal_index=subgoal_index,
                    action=action,
                    task=task,
                    request=request,
                    result=result,
                    before_place=before_place,
                    after_place=after_place,
                    move_tag=move_tag,
                    made_progress=made_progress,
                    chain_count=det_chain_count,
                )
                if is_move_action
                else None
            )
            if det_cont is not None:
                det_chain_count += 1
                summary, next_decision, next_reasoning = det_cont
                if move_tag and move_tag not in summary:
                    summary = f"{move_tag} {summary}"
                self._note(summary)
                attempts.append(f"{request}: {summary}")
                if after_place is not None and (not subgoal_places or subgoal_places[-1] != after_place):
                    subgoal_places.append(after_place)
                if next_decision is not None:
                    queued_decision = (next_decision, next_reasoning)
                continue

            det_chain_count = 0
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

            if (
                replans >= self.MAX_REPLANS
                or self._is_dialogue_unrecoverable()
                or (
                    self._is_in_dialogue()
                    and self._dialogue_clear_failed
                    and self._is_dialogue_closing_subgoal(subgoal)
                )
            ):
                return self._finish("stuck", f"Could not achieve: {subgoal}")
            frame = self._current_frame()
            self._perceive(frame)
            decision, reason, subgoals = self._reconsider(subgoal, attempts, frame)
            replans += 1
            if decision == "quit":
                return self._finish("quit", reason)
            if (
                self._is_in_dialogue()
                and self._dialogue_clear_failed
                and self._is_dialogue_closing_subgoal(subgoal)
                and subgoals
                and self._is_dialogue_closing_subgoal(subgoals[0])
            ):
                return self._finish("stuck", f"Could not achieve: {subgoal}")
            calls_at_plan_start = self._calls
            self.plan = subgoals
            index = 0

        return self._finish("no_plan", "The plan became empty while working through it.")
