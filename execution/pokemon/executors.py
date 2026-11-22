from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from execution.game_api import (AgentState, BattleMenuAction, InteractAction, LowLevelAction, LowLevelActions,
                                MenuAction, MoveStepsAction, OpenMenuAction, PassDialogueAction, PickAttackAction,
                                WaitAction)

from execution.executors import PolicyExecutor
from execution.perception.pokemon.battle import (has_bottom_dialogue_box, has_menu_cursor,
                                                 has_modal_preview_box, has_top_dialogue_box,
                                                 is_battle_decision_screen, is_battle_screen,
                                                 is_cutscene_frame_changing, is_same_battle_frame)
from execution.perception.pokemon.navigation import (BLOCKED, DIRECTIONS, LEDGE, LEDGE_LEFT, WALKABLE, approach_cells,
                                                     boundary_exit_direction, cell_kinds, compress_path,
                                                     diagonal_barrier_pairs, edge_cells, edge_runs,
                                                     filter_forward_edge_cells, filter_open_lane_edge_cells, in_bounds,
                                                     is_boundary_exit_direction, parse_all_cells, parse_cell,
                                                     parse_edge_direction, shortest_path, verbalize_edges)
from execution.pokemon.executor_actions import CheckInteractionAction
from execution.report import EnvironmentStepRecord
from utils import log_error, ocr, parse_int, parse_key_value, parse_yes_no


DIALOGUE_BACKGROUND = 248
CURSOR_MARGIN = 12


def is_blank_crop(crop: np.ndarray) -> bool:
    if crop.size == 0 or crop.shape[1] <= CURSOR_MARGIN:
        return True
    return bool(np.all(crop[:, :-CURSOR_MARGIN] >= DIALOGUE_BACKGROUND))


def is_same_crop(earlier: np.ndarray, later: np.ndarray) -> bool:
    if earlier.shape != later.shape:
        return False
    a = earlier[:, :-CURSOR_MARGIN] >= DIALOGUE_BACKGROUND
    b = later[:, :-CURSOR_MARGIN] >= DIALOGUE_BACKGROUND
    return bool(np.array_equal(a, b))


SCROLL_BORDER_MARGIN = 3
SCROLL_SHIFTS = (4, 8, 12, 16)


def is_prefix_crop(earlier: np.ndarray, later: np.ndarray) -> bool:
    if earlier.shape != later.shape:
        return False
    a, b = earlier[:, :-CURSOR_MARGIN], later[:, :-CURSOR_MARGIN]
    differ = a != b
    return bool(np.all(a[differ] >= DIALOGUE_BACKGROUND))


def is_scrolled_prefix_crop(earlier: np.ndarray, later: np.ndarray) -> bool:
    if (earlier.shape != later.shape
            or earlier.shape[0] <= 2 * SCROLL_BORDER_MARGIN + 8
            or earlier.shape[1] <= CURSOR_MARGIN + SCROLL_BORDER_MARGIN):
        return False
    a = earlier[:, SCROLL_BORDER_MARGIN:-CURSOR_MARGIN] >= DIALOGUE_BACKGROUND
    b = later[:, SCROLL_BORDER_MARGIN:-CURSOR_MARGIN] >= DIALOGUE_BACKGROUND
    top = SCROLL_BORDER_MARGIN + 1
    bot = a.shape[0] - (SCROLL_BORDER_MARGIN + 1)
    if bot <= top:
        return False
    for dy in SCROLL_SHIFTS:
        if top + dy >= bot:
            continue
        a_body = a[top + dy:bot]
        b_body = b[top:bot - dy]
        differ = a_body != b_body
        if bool(np.all(a[top:top + dy])) and bool(np.all(a_body[differ])):
            return True
    return False


def captured_dialogue_crops(records: List[EnvironmentStepRecord]) -> List[Any]:
    crops = []
    for record in records:
        for state in record.transition_states:
            step_crops = state.get("ocr", {}).get("ocr_regions", {}).get("dialogue")
            if step_crops is None or len(step_crops) == 0:
                continue
            crops.extend(step_crops)
    return crops


def drop_prefix_crops(crops: List[Any], kept: Optional[List[Any]] = None) -> List[Any]:
    kept = [] if kept is None else kept
    for crop in crops:
        if is_blank_crop(crop):
            continue
        if kept and is_same_crop(kept[-1], crop):
            continue
        if kept and (is_prefix_crop(kept[-1], crop) or is_scrolled_prefix_crop(kept[-1], crop)):
            kept[-1] = crop
        else:
            kept.append(crop)
    return kept


INTERACT_SETTLE_WAIT = 2
INTERACT_MODAL_MAX_PRESSES = 8
MAX_INTERACT_DIALOGUE_STEPS = 10


class InteractionExecutor(PolicyExecutor):
    CARDINAL_DIRECTIONS = {cell: name for name, cell in CheckInteractionAction.cardinals.items()}

    def __init__(self, env, task, max_steps: int = 3, identified=None, **kwargs):
        self.identified = identified
        self.check_result: Optional[dict] = None
        self.dialogue_text: Optional[str] = None
        self.dialogue_crops: List[Any] = []
        self.captured_crops: List[Any] = []
        super().__init__(env, task, max_steps, **kwargs)

    def _fail(self, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return -1

    def _agent_state(self):
        return self._get_state()["pokemon_core"]["agent_state"]

    def _current_frame(self):
        return self._get_state()["core"]["current_frame"]

    def _has_interaction_overlay(self, frame=None) -> bool:
        frame = self._current_frame() if frame is None else frame
        return has_bottom_dialogue_box(frame) or has_modal_preview_box(frame)

    def _wait_for_interaction(self, records: List[EnvironmentStepRecord]):
        state = self._agent_state()
        for _ in range(INTERACT_SETTLE_WAIT):
            if state != AgentState.FREE_ROAM or self._last_terminated or self._last_truncated:
                break
            wait_record = self._take_action(WaitAction)
            records.append(wait_record)
            state = self._agent_state()
            if captured_dialogue_crops([wait_record]) or self._has_interaction_overlay(wait_record.frame_after):
                break
        return state

    def _advance_interaction_overlay(self, records: List[EnvironmentStepRecord]):
        state = self._agent_state()
        for _ in range(INTERACT_MODAL_MAX_PRESSES):
            if state != AgentState.FREE_ROAM or self._last_terminated or self._last_truncated:
                break
            if not self._has_interaction_overlay():
                break
            btn_record = self._take_action_string("button(a)")
            if btn_record is None:
                break
            records.append(btn_record)
            if self._last_terminated or self._last_truncated:
                break
            state = self._agent_state()
            if state == AgentState.FREE_ROAM:
                state = self._wait_for_interaction(records)
        return state

    def _clear_interaction_dialogue(self, records: List[EnvironmentStepRecord]) -> None:
        for _ in range(MAX_INTERACT_DIALOGUE_STEPS):
            if self._agent_state() != AgentState.IN_DIALOGUE or self._last_terminated or self._last_truncated:
                break
            pass_record = self._take_action(PassDialogueAction)
            records.append(pass_record)
            if self._last_terminated or self._last_truncated:
                break
            if self._agent_state() == AgentState.FREE_ROAM:
                self._wait_for_interaction(records)
                if self._agent_state() == AgentState.FREE_ROAM and self._has_interaction_overlay():
                    self._advance_interaction_overlay(records)
                if self._agent_state() != AgentState.IN_DIALOGUE:
                    break
            elif pass_record.action_success == -1:
                break

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        info = self._get_state()
        if info["pokemon_core"]["agent_state"] != AgentState.FREE_ROAM:
            return self._fail("not_free_roam")

        check = CheckInteractionAction(self._vlm_call, self._parameters)
        result, code = check.execute(info, target=self._task, identified=self.identified)
        self.check_result = result
        if code is None:
            return self._fail("invalid_check", result["invalid"])
        if code == -1:
            return self._fail("target_not_found")
        if code == 1:
            return self._fail("target_only_diagonal",
                              f"Target found at {result['found_cells_str']}, which cannot be faced")

        direction = result["direction"]
        records = []
        self._begin_scripted(f"face {direction} and interact with {self._task}")
        if getattr(self._env, "facing", None) != direction:
            turn = self._take_action(MoveStepsAction, direction=direction, steps=1)
            records.append(turn)
            if self._last_terminated or self._last_truncated:
                return self._finish(records)
            if self._agent_state() != AgentState.FREE_ROAM:
                if self._agent_state() == AgentState.IN_DIALOGUE:
                    self._clear_interaction_dialogue(records)
                return self._finish(records)
            action_return = turn.transition_states[-1]["core"].get("action_return") or {}
            if action_return.get("n_steps_taken"):
                return self._fail("target_cell_walkable",
                                  f"Pressed {direction} to face the target at {result['found_cells_str']}, "
                                  "but the player walked onto that cell instead, so nothing is there")

        interact = self._take_action(InteractAction)
        records.append(interact)
        if self._last_terminated or self._last_truncated:
            return self._finish(records)
        state = self._agent_state()
        if state == AgentState.FREE_ROAM:
            state = self._wait_for_interaction(records)
            if self._last_terminated or self._last_truncated:
                return self._finish(records)
        saw_overlay = any(self._has_interaction_overlay(r.frame_after) for r in records)
        if state == AgentState.FREE_ROAM and self._has_interaction_overlay():
            saw_overlay = True
            state = self._advance_interaction_overlay(records)
            if self._last_terminated or self._last_truncated:
                return self._finish(records)
        if (interact.action_success != 1
                and state == AgentState.FREE_ROAM
                and not captured_dialogue_crops(records)
                and not saw_overlay):
            return self._fail("interaction_did_nothing")

        if state == AgentState.IN_DIALOGUE and not (self._last_terminated or self._last_truncated):
            self._clear_interaction_dialogue(records)
        return self._finish(records)

    def _finish(self, records: List[EnvironmentStepRecord]) -> int:
        self.dialogue_text = self._read_dialogue(records)
        self.report.notes = self.dialogue_text
        final_state = self._get_state()["pokemon_core"]["agent_state"]
        if self._last_terminated:
            self.report.termination_reason = "terminated"
            return 1
        if self._last_truncated:
            self.report.termination_reason = "truncated"
            return 2
        if final_state == AgentState.IN_DIALOGUE:
            self.report.termination_reason = "dialogue_not_cleared"
        elif final_state == AgentState.FREE_ROAM:
            self.report.termination_reason = "dialogue_finished"
        else:
            self.report.termination_reason = f"dialogue_ended_in_{final_state.name.lower()}"
        return 0

    def _read_dialogue(self, records: List[EnvironmentStepRecord]) -> str:
        self.captured_crops.extend(captured_dialogue_crops(records))
        frames = drop_prefix_crops(self.captured_crops)
        self.dialogue_crops = frames
        if not frames:
            return ""
        return "\n".join(ocr(frames, self._vlm, do_merge=True, parameters=self._parameters))


MOVE_OK = 0
MOVE_TERMINATED = 1
MOVE_TRUNCATED = 2
MOVE_NOT_FREE_ROAM = -1
MOVE_TARGET_NOT_FOUND = -2
MOVE_TARGET_AMBIGUOUS = -3
MOVE_BAD_LOCATION = -4
MOVE_NO_PATH = -5
MOVE_BLOCKED = -6
MOVE_INTERRUPTED = -7
MOVE_OUT_OF_STEPS = -8
LOCATION_FAIL_CODES = (MOVE_TARGET_NOT_FOUND, MOVE_TARGET_AMBIGUOUS, MOVE_BAD_LOCATION)
MOVE_SETTLE_WAIT = 2
MOVE_MAX_CUTSCENE_WAIT = 16
MOVE_POST_MOTION_WAIT = 2

GRID_DESCRIPTION = """The screen is a grid of cells. The player is at (0, 0). x increases to the right, from -4 at the left edge to 5 at the right edge. y increases upward, from 4 at the top edge to -4 at the bottom edge."""


class GridMoveExecutor(PolicyExecutor):
    LOCATE_TAG = "locate"
    MAX_NEW_TOKENS = 4000
    MAX_REROUTES = 2

    def __init__(self, env, task, max_steps: int = 10, screen_tiles: Optional[str] = None,
                 cell_kinds: Optional[Dict[Tuple[int, int], str]] = None, identified: Optional[Any] = None,
                 people_fn=None, **kwargs):
        self.screen_tiles = screen_tiles
        self.cell_kinds = cell_kinds
        self.identified = identified
        self.people_fn = people_fn
        self.target_cell: Optional[Tuple[int, int]] = None
        self.plan: Optional[List[Tuple[str, int, str]]] = None
        self.dialogue_crops: List[Any] = []
        self._n_moves = 0
        self._steps_taken = 0
        self._displacement: Tuple[int, int] = (0, 0)
        self._last_step_taken = 0
        self._saw_cutscene_activity = False
        super().__init__(env, task, max_steps, screen_tiles=screen_tiles, cell_kinds=cell_kinds,
                         identified=identified, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _locate(self, prompt: str) -> str:
        frame = self._get_state()["core"]["current_frame"]
        return self._vlm_call(self.LOCATE_TAG, texts=prompt, images=[frame], max_new_tokens=self.MAX_NEW_TOKENS)

    def _check_inputs(self) -> Optional[int]:
        if self.screen_tiles is None or self.cell_kinds is None:
            log_error(f"{self.__class__.__name__} needs screen_tiles and cell_kinds from the supervisor.",
                      self._parameters)
        if self._get_state()["pokemon_core"]["agent_state"] != AgentState.FREE_ROAM:
            return self._fail(MOVE_NOT_FREE_ROAM, "not_free_roam")
        return None

    def _current_map_place(self) -> Optional[Tuple[int, Tuple[int, int]]]:
        gmap = getattr(self._env, "_map", None)
        if gmap is None or not getattr(gmap, "current", None) or getattr(gmap, "pos", None) is None:
            return None
        try:
            return gmap.resolve(gmap.segment, gmap.pos)
        except Exception:  # noqa: BLE001
            return None

    def _record_empirical_block(self, direction: str) -> None:
        recorder = getattr(self._env, "record_blocked_step", None)
        if callable(recorder):
            try:
                recorder(direction=direction)
            except Exception:  # noqa: BLE001
                pass

    def _refresh_local_perception(self, blocked_neighbor: Optional[Tuple[int, int]] = None) -> bool:
        tiles = getattr(self._env, "_tiles", None)
        if tiles is None or not hasattr(tiles, "identify_tiles"):
            if blocked_neighbor is not None and self.cell_kinds is not None:
                self.cell_kinds = dict(self.cell_kinds)
                self.cell_kinds[blocked_neighbor] = BLOCKED
            return False
        frame = self._get_state()["core"]["current_frame"]
        self.identified = tiles.identify_tiles(frame)
        people = None
        if callable(self.people_fn):
            try:
                people = self.people_fn()
            except Exception:  # noqa: BLE001
                people = None
        kinds = cell_kinds(
            self.identified,
            getattr(self._env, "ledge_patterns", None),
            getattr(self._env, "walk_patterns", None),
            people,
            getattr(self._env, "left_ledge_patterns", None),
            getattr(self._env, "blocked_cells", None),
        )
        if blocked_neighbor is not None:
            kinds[blocked_neighbor] = BLOCKED
        self.cell_kinds = kinds
        return True

    def _settle_free_roam(self):
        state = self._get_state()["pokemon_core"]["agent_state"]
        self._saw_cutscene_activity = False
        waited = 0
        post_motion_remaining = 0
        while (state == AgentState.FREE_ROAM
               and not (self._last_terminated or self._last_truncated)
               and waited < MOVE_MAX_CUTSCENE_WAIT):
            frame = self._get_state()["core"]["current_frame"]
            box_open = has_bottom_dialogue_box(frame)
            if box_open:
                self._saw_cutscene_activity = True
            if waited >= MOVE_SETTLE_WAIT and not box_open and post_motion_remaining <= 0:
                break
            wait_record = self._take_action(WaitAction)
            drop_prefix_crops(captured_dialogue_crops([wait_record]), self.dialogue_crops)
            waited += 1
            state = self._get_state()["pokemon_core"]["agent_state"]
            if state != AgentState.FREE_ROAM or self._last_terminated or self._last_truncated:
                break
            if (is_cutscene_frame_changing(wait_record.frame_before, wait_record.frame_after)
                    or has_bottom_dialogue_box(wait_record.frame_after)):
                self._saw_cutscene_activity = True
                post_motion_remaining = MOVE_POST_MOTION_WAIT
            elif post_motion_remaining > 0:
                post_motion_remaining -= 1
        return state

    ARROW_ACTIONS = {
        "up": LowLevelActions.PRESS_ARROW_UP,
        "down": LowLevelActions.PRESS_ARROW_DOWN,
        "left": LowLevelActions.PRESS_ARROW_LEFT,
        "right": LowLevelActions.PRESS_ARROW_RIGHT,
    }

    def _pre_turn(self, direction: str) -> Optional[int]:
        if getattr(self._env, "facing", None) == direction:
            return None
        arrow = self.ARROW_ACTIONS.get(direction)
        if arrow is None:
            return None
        before_place = self._current_map_place()
        record = self._take_action(LowLevelAction, low_level_action=arrow)
        drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
        if self._last_terminated:
            return self._fail(MOVE_TERMINATED, "terminated")
        if self._last_truncated:
            return self._fail(MOVE_TRUNCATED, "truncated")
        after_place = self._current_map_place()
        taken = 0
        if before_place is not None and after_place is not None:
            if before_place[0] != after_place[0]:
                taken = 1
            else:
                step_dx = after_place[1][0] - before_place[1][0]
                step_dy = after_place[1][1] - before_place[1][1]
                if (step_dx, step_dy) != (0, 0):
                    taken = max(1, abs(step_dx) + abs(step_dy))
                    self._displacement = (self._displacement[0] + step_dx, self._displacement[1] + step_dy)
        self._steps_taken += taken
        self._last_step_taken += taken
        state = self._get_state()["pokemon_core"]["agent_state"]
        if state != AgentState.FREE_ROAM or has_bottom_dialogue_box(self._get_state()["core"]["current_frame"]):
            return self._fail(MOVE_INTERRUPTED, "move_interrupted",
                              f"Left free roam after {taken} steps {direction}: {state.name}")
        return taken

    def _step(self, direction: str, steps: int, move_kind: str = "walk") -> Optional[int]:
        remaining = steps
        attempts = 0
        retried_after_cutscene = False
        self._last_step_taken = 0
        dx, dy = DIRECTIONS[direction]
        pre_turn = self._pre_turn(direction)
        if pre_turn is not None:
            if pre_turn < 0 or self._last_terminated or self._last_truncated:
                return pre_turn
            remaining = max(0, remaining - pre_turn)
        while remaining > 0:
            if attempts >= steps + 2:
                return self._fail(MOVE_BLOCKED, "move_blocked", f"Could not finish moving {direction} {steps}")
            if self._n_moves >= self._max_steps:
                return self._fail(MOVE_OUT_OF_STEPS, "out_of_steps")
            attempts += 1
            self._n_moves += 1
            before_place = self._current_map_place()
            record = self._take_action(MoveStepsAction, direction=direction, steps=remaining)
            drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
            if self._last_terminated:
                return self._fail(MOVE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._fail(MOVE_TRUNCATED, "truncated")
            action_return = (record.transition_states[-1]["core"].get("action_return") or {}) if record.transition_states else {}
            taken = action_return.get("n_steps_taken", 0)
            self._steps_taken += taken
            self._last_step_taken += taken
            after_place = self._current_map_place()
            if before_place is not None and after_place is not None and before_place[0] == after_place[0]:
                step_dx = after_place[1][0] - before_place[1][0]
                step_dy = after_place[1][1] - before_place[1][1]
            else:
                mult = 2 if (move_kind == "jump" and taken > 0) else 1
                step_dx, step_dy = dx * taken * mult, dy * taken * mult
            self._displacement = (self._displacement[0] + step_dx, self._displacement[1] + step_dy)
            state = self._get_state()["pokemon_core"]["agent_state"]
            if record.action_success == 2 or state != AgentState.FREE_ROAM:
                return self._fail(MOVE_INTERRUPTED, "move_interrupted",
                                  f"Left free roam after {taken} steps {direction}: {state.name}")
            remaining -= taken
            if (taken == 0 and not action_return.get("rotated")) or (
                    record.action_success in (-1, 1) and remaining > 0 and taken > 0):
                step_moved_screen = (
                    is_cutscene_frame_changing(record.frame_before, record.frame_after)
                    or has_bottom_dialogue_box(record.frame_after)
                )
                state = self._settle_free_roam()
                if self._last_terminated:
                    return self._fail(MOVE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._fail(MOVE_TRUNCATED, "truncated")
                if state != AgentState.FREE_ROAM or has_bottom_dialogue_box(self._get_state()["core"]["current_frame"]):
                    return self._fail(MOVE_INTERRUPTED, "move_interrupted",
                                      f"Left free roam after {taken} steps {direction}: {state.name}")
                if ((step_moved_screen or self._saw_cutscene_activity)
                        and not retried_after_cutscene
                        and self._n_moves < self._max_steps):
                    retried_after_cutscene = True
                    continue
                self._record_empirical_block(direction)
                if taken == 0 and not action_return.get("rotated"):
                    return self._fail(MOVE_BLOCKED, "move_blocked",
                                      f"Blocked moving {direction} with {remaining} steps left")
                return self._fail(MOVE_BLOCKED, "move_blocked",
                                  f"Stopped moving {direction} with {remaining} steps left")
        return None

    def _walk(self, goals: Dict[Tuple[int, int], Optional[str]]) -> Optional[int]:
        current_goals = dict(goals)
        reroutes = 0
        while True:
            path = shortest_path(self.cell_kinds, (0, 0), current_goals)
            if path is None:
                if reroutes > 0:
                    return self._fail(MOVE_BLOCKED, "move_blocked",
                                      f"Blocked before reaching any of {sorted(current_goals)}")
                pinches = diagonal_barrier_pairs(self.cell_kinds, current_goals)
                if pinches:
                    b1, b2 = pinches[0]
                    note = (
                        f"No walkable path from (0, 0) to any of {sorted(current_goals)} — "
                        f"separated by an impassable obstacle barrier (obstacles at {b1} and {b2} meet diagonally "
                        f"with no orthogonal passage); use MoveOff to route off screen around the obstacle rather than MoveTo or Free"
                    )
                else:
                    note = (
                        f"No walkable path from (0, 0) to any of {sorted(current_goals)} — "
                        f"separated by an impassable obstacle barrier on the current screen; "
                        f"use MoveOff to route off screen around the obstacle rather than MoveTo or Free"
                    )
                return self._fail(MOVE_NO_PATH, "no_path", note)
            self.plan = compress_path(path)
            end = path[-1][2] if path else (0, 0)
            facing = current_goals[end]
            description = ", ".join(f"{m} {d} {n}" for d, n, m in self.plan) or "stay"
            prefix = "reroute to" if reroutes > 0 else "walk to"
            self._begin_scripted(f"{prefix} {end} ({description})" + (f" and face {facing}" if facing else ""))
            start_disp = self._displacement
            leg_steps_before = self._steps_taken
            failed_code = None
            failed_dir = None
            for direction, n_steps, move in self.plan:
                code = self._step(direction, n_steps, move_kind=move)
                if code is not None:
                    failed_code = code
                    failed_dir = direction
                    break
            if failed_code is not None:
                leg_taken = self._steps_taken - leg_steps_before
                if (failed_code == MOVE_BLOCKED
                        and leg_taken > 0
                        and reroutes < self.MAX_REROUTES
                        and self._n_moves < self._max_steps
                        and self._get_state()["pokemon_core"]["agent_state"] == AgentState.FREE_ROAM
                        and not has_bottom_dialogue_box(self._get_state()["core"]["current_frame"])):
                    dx = self._displacement[0] - start_disp[0]
                    dy = self._displacement[1] - start_disp[1]
                    if (dx, dy) != (0, 0):
                        shifted_goals = {
                            (gx - dx, gy - dy): f
                            for (gx, gy), f in current_goals.items()
                            if in_bounds((gx - dx, gy - dy))
                        }
                        if (0, 0) in shifted_goals:
                            self.report.termination_reason = None
                            self.report.notes = None
                            facing = shifted_goals[(0, 0)]
                            break
                        if shifted_goals:
                            blocked_neighbor = DIRECTIONS.get(failed_dir)
                            self._refresh_local_perception(blocked_neighbor=blocked_neighbor)
                            if shortest_path(self.cell_kinds, (0, 0), shifted_goals) is not None:
                                current_goals = shifted_goals
                                reroutes += 1
                                self.report.termination_reason = None
                                self.report.notes = None
                                continue
                return failed_code
            break
        if facing is not None:
            pre_turn = self._pre_turn(facing)
            if pre_turn is not None and (pre_turn < 0 or self._last_terminated or self._last_truncated):
                return pre_turn
        return None


class OnScreenMoveExecutor(GridMoveExecutor):
    LOCATE_TAG = "locate_target"
    LOCATE_PROMPT = """You are helping a player move to something on the screen in a Game Boy Pokemon game.

The player wants to move to: "[TARGET]"

The image is the current game screen. [GRID_DESCRIPTION] Here is what has been recognised on the screen:
[SCREEN_TILES]

Find the exact cell of the target.
- If exactly one thing on the screen matches the target, give its coordinates. If it covers several cells, give the one closest to the player.
- If more than one thing on the screen could be the target, answer Ambiguous and list every candidate.
- If nothing on the screen matches the target, answer Not Found.

Respond in exactly this format:
Reasoning: <what matches the target and where>
Location: <(x, y), Ambiguous, or Not Found>
Candidates: <if Ambiguous, each candidate with its coordinates and what sets it apart from the others; otherwise None>
[STOP]"""

    def _resolve_explicit_target(self) -> Optional[Tuple[int, int]]:
        coords = [c for c in parse_all_cells(self._task) if in_bounds(c)]
        if len(coords) != 1:
            return None
        target = coords[0]
        if target == (0, 0):
            return target
        if self.cell_kinds.get(target) == WALKABLE:
            if shortest_path(self.cell_kinds, (0, 0), [target]) is not None:
                return target
        else:
            goals = approach_cells(self.cell_kinds, target)
            if goals and shortest_path(self.cell_kinds, (0, 0), goals) is not None:
                return target
        return target

    def _execute(self) -> int:
        code = self._check_inputs()
        if code is not None:
            return code
        target = self._resolve_explicit_target()
        if target is None:
            prompt = (self.LOCATE_PROMPT.replace("[TARGET]", self._task.strip())
                      .replace("[GRID_DESCRIPTION]", GRID_DESCRIPTION)
                      .replace("[SCREEN_TILES]", self.screen_tiles))
            output = self._locate(prompt)
            location = (parse_key_value(output, "Location") or "").strip()
            lowered = location.lower()
            if "ambiguous" in lowered:
                return self._fail(MOVE_TARGET_AMBIGUOUS, "target_ambiguous", parse_key_value(output, "Candidates"))
            if "not found" in lowered:
                return self._fail(MOVE_TARGET_NOT_FOUND, "target_not_found", parse_key_value(output, "Reasoning"))
            target = parse_cell(location)
            if target is None or not in_bounds(target):
                return self._fail(MOVE_BAD_LOCATION, "bad_location", f"Unusable location {location!r}")
        self.target_cell = target
        if target == (0, 0):
            return MOVE_OK
        exit_dir = None
        if self.cell_kinds.get(target) == WALKABLE:
            exit_dir = boundary_exit_direction(self.cell_kinds, self.identified, target)
            goals = {target: None}
        else:
            goals = approach_cells(self.cell_kinds, target)
            if not goals:
                return self._fail(MOVE_NO_PATH, "no_path", f"Nothing walkable next to the target at {target}")
        code = self._walk(goals)
        if code is not None:
            return code
        if exit_dir is not None and self._get_state()["pokemon_core"]["agent_state"] == AgentState.FREE_ROAM:
            exit_code = self._step(exit_dir, 1)
            if exit_code is not None and exit_code != MOVE_BLOCKED:
                return exit_code
        self._settle_free_roam()
        if self._last_terminated:
            return self._fail(MOVE_TERMINATED, "terminated")
        if self._last_truncated:
            return self._fail(MOVE_TRUNCATED, "truncated")
        self.report.termination_reason = "arrived"
        self.report.notes = f"Moved to the target at {target}"
        return MOVE_OK


class OffScreenMoveExecutor(GridMoveExecutor):
    LOCATE_TAG = "locate_edge"
    MAX_DETERMINISTIC_CORRIDOR_WIDTH = 3

    def _at_exterior_boundary(self, direction: str) -> bool:
        from execution.perception.pokemon.grid import split_grid

        dx, dy = DIRECTIONS[direction]
        cells = split_grid(self._get_state()["core"]["current_frame"])
        cur = (dx, dy)
        saw_neighbor = False
        while in_bounds(cur):
            piece = cells.get(cur)
            if piece is None or int(piece.min()) != int(piece.max()):
                return False
            saw_neighbor = True
            cur = (cur[0] + dx, cur[1] + dy)
        return saw_neighbor

    def _screen_scrolled_along(self, before_frame: np.ndarray, after_frame: np.ndarray, direction: str) -> bool:
        from execution.strategist.pokemon.global_map.registration import register

        dx, dy = DIRECTIONS[direction]
        if (self._displacement[0] * dx + self._displacement[1] * dy) > 0:
            return True
        placed = register(before_frame, after_frame, direction, max(1, self._steps_taken))
        if placed is None:
            return False
        shift_x, shift_y = placed[0]
        return (shift_x * dx + shift_y * dy) > 0

    def _can_step_from_current(self, direction: str) -> Optional[bool]:
        tiles = getattr(self._env, "_tiles", None)
        if tiles is None or not hasattr(tiles, "identify_tiles"):
            return None
        frame = self._get_state()["core"]["current_frame"]
        identified = tiles.identify_tiles(frame)
        people = None
        if callable(self.people_fn):
            try:
                people = self.people_fn()
            except Exception:  # noqa: BLE001
                people = None
        kinds = cell_kinds(
            identified,
            getattr(self._env, "ledge_patterns", None),
            getattr(self._env, "walk_patterns", None),
            people,
            getattr(self._env, "left_ledge_patterns", None),
            getattr(self._env, "blocked_cells", None),
        )
        dx, dy = DIRECTIONS[direction]
        nxt = (dx, dy)
        kind = kinds.get(nxt)
        if kind == WALKABLE:
            return True
        if (kind == LEDGE and direction == "down") or (kind == LEDGE_LEFT and direction == "left"):
            return kinds.get((2 * dx, 2 * dy)) == WALKABLE
        if kind in (BLOCKED, LEDGE, LEDGE_LEFT):
            return False
        return None

    def _resolve_deterministic_edge(self) -> Optional[Tuple[str, Tuple[int, int]]]:
        direction = parse_edge_direction(self._task)
        if direction not in DIRECTIONS:
            return None
        cells = edge_cells(self.cell_kinds, direction)
        if not cells:
            return None
        forward_cells = filter_forward_edge_cells(cells, direction)
        gmap = getattr(self._env, "_map", None)
        open_cells = filter_open_lane_edge_cells(forward_cells, direction, self.cell_kinds, gmap)

        coords = [c for c in parse_all_cells(self._task) if in_bounds(c)]
        on_edge_coords = [c for c in coords if c in cells]
        if on_edge_coords:
            preferred = [c for c in on_edge_coords if c in open_cells] or [c for c in on_edge_coords if c in forward_cells] or on_edge_coords
            path = shortest_path(self.cell_kinds, (0, 0), preferred)
            target = path[-1][2] if path else preferred[0]
            return direction, target

        if len(coords) == 1:
            waypoint = coords[0]
            axis = 0 if direction in ("up", "down") else 1
            aligned = [c for c in open_cells if c[axis] == waypoint[axis]] or [c for c in forward_cells if c[axis] == waypoint[axis]]
            if len(aligned) == 1:
                return direction, aligned[0]

        stripped = self._task.strip().lower()
        is_bare_direction = stripped in ("up", "down", "left", "right", "north", "south", "west", "east")
        runs = edge_runs(open_cells, direction)
        if len(runs) == 1:
            run = runs[0]
            if is_bare_direction or len(run) <= self.MAX_DETERMINISTIC_CORRIDOR_WIDTH:
                path = shortest_path(self.cell_kinds, (0, 0), run)
                target = path[-1][2] if path else run[0]
                return direction, target
        elif is_bare_direction and open_cells:
            path = shortest_path(self.cell_kinds, (0, 0), open_cells)
            if path:
                return direction, path[-1][2]
        return None

    LOCATE_PROMPT = """You are helping a player walk off the edge of the screen in a Game Boy Pokemon game.

The player wants to: "[TARGET]"

The image is the current game screen. [GRID_DESCRIPTION] Here is what has been recognised on the screen:
[SCREEN_TILES]

These cells on each edge of the screen can be walked on:
[EDGE_CELLS]

Choose the edge cell the player should walk to so that they can then keep walking off the screen as described. Pick a cell from the list above on the edge in the direction the player wants to go.
- Look carefully at the image along the chosen edge: if a cell on that edge is visibly aligned with or right below/above/beside a signpost, fence post, tree, ledge, or narrow corridor wall at the edge, pick an adjacent open edge cell in the clear lane or gap instead of the cell aligned with the obstacle.
- If no listed cell fits the description, answer Not Found.

Respond in exactly this format:
Reasoning: <which edge and which cell, and why>
Direction: <up, down, left or right>
Location: <(x, y) or Not Found>
[STOP]"""

    def _execute(self) -> int:
        code = self._check_inputs()
        if code is not None:
            return code
        parsed_dir = parse_edge_direction(self._task)
        if parsed_dir in DIRECTIONS and not edge_cells(self.cell_kinds, parsed_dir):
            if is_boundary_exit_direction(self.cell_kinds, (0, 0), parsed_dir):
                self._begin_scripted(f"step {parsed_dir} at the boundary")
                step_code = self._step(parsed_dir, 1)
                if step_code is None:
                    self.report.termination_reason = "left_screen"
                    self.report.notes = f"Stepped {parsed_dir} across the boundary"
                    return MOVE_OK
                if step_code != MOVE_BLOCKED:
                    return step_code
            return self._fail(MOVE_TARGET_NOT_FOUND, "edge_not_found",
                              f"No walkable cells on the {parsed_dir} edge")

        resolved = self._resolve_deterministic_edge()
        if resolved is not None:
            direction, target = resolved
        else:
            prompt = (self.LOCATE_PROMPT.replace("[TARGET]", self._task.strip())
                      .replace("[GRID_DESCRIPTION]", GRID_DESCRIPTION)
                      .replace("[SCREEN_TILES]", self.screen_tiles)
                      .replace("[EDGE_CELLS]", verbalize_edges(self.cell_kinds)))
            output = self._locate(prompt)
            location = (parse_key_value(output, "Location") or "").strip()
            direction = (parse_key_value(output, "Direction") or "").strip().lower()
            direction_aliases = {"north": "up", "south": "down", "west": "left", "east": "right"}
            direction = direction_aliases.get(direction, direction)
            if parsed_dir in DIRECTIONS:
                direction = parsed_dir
            if "not found" in location.lower():
                if direction in DIRECTIONS and not edge_cells(self.cell_kinds, direction):
                    if is_boundary_exit_direction(self.cell_kinds, (0, 0), direction):
                        self._begin_scripted(f"step {direction} at the boundary")
                        step_code = self._step(direction, 1)
                        if step_code is None:
                            self.report.termination_reason = "left_screen"
                            self.report.notes = f"Stepped {direction} across the boundary"
                            return MOVE_OK
                        if step_code != MOVE_BLOCKED:
                            return step_code
                return self._fail(MOVE_TARGET_NOT_FOUND, "edge_not_found", parse_key_value(output, "Reasoning"))
            target = parse_cell(location)
            if direction not in DIRECTIONS:
                return self._fail(MOVE_BAD_LOCATION, "bad_direction", f"Unusable direction {direction!r}")
            cells = edge_cells(self.cell_kinds, direction)
            forward_cells = filter_forward_edge_cells(cells, direction)
            candidate_cells = forward_cells or cells
            if target is not None and target not in candidate_cells and candidate_cells:
                target = min(candidate_cells, key=lambda c: (c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2)
            if target is None or target not in cells:
                return self._fail(MOVE_BAD_LOCATION, "bad_location",
                                  f"{location!r} is not a walkable cell on the {direction} edge")
            gmap = getattr(self._env, "_map", None)
            open_cells = filter_open_lane_edge_cells(candidate_cells, direction, self.cell_kinds, gmap)
            if target not in open_cells and open_cells:
                axis = 0 if direction in ("up", "down") else 1
                adjacent_open = [c for c in open_cells if abs(c[axis] - target[axis]) <= 2]
                if adjacent_open:
                    target = min(adjacent_open, key=lambda c: (abs(c[axis] - target[axis]), abs(c[axis])))

        self.target_cell = target
        initial_frame = self._get_state()["core"]["current_frame"]
        code = self._walk({target: None})
        if code is not None:
            return code
        after_walk_frame = self._get_state()["core"]["current_frame"]
        scrolled_to_edge = self._steps_taken > 0 and (
            self._screen_scrolled_along(initial_frame, after_walk_frame, direction)
            or self._at_exterior_boundary(direction)
        )
        can_step = self._can_step_from_current(direction)
        if can_step is False:
            self._record_empirical_block(direction)
            if scrolled_to_edge:
                self._settle_free_roam()
                if self._last_terminated:
                    return self._fail(MOVE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._fail(MOVE_TRUNCATED, "truncated")
                self.report.termination_reason = "left_screen"
                self.report.notes = f"Walked to {target} at the {direction} edge"
                return MOVE_OK
            return self._fail(MOVE_BLOCKED, "move_blocked", f"Blocked at {target} moving {direction}")
        code = self._step(direction, 1)
        if code is not None:
            if code == MOVE_BLOCKED and scrolled_to_edge:
                self._settle_free_roam()
                if self._last_terminated:
                    return self._fail(MOVE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._fail(MOVE_TRUNCATED, "truncated")
                self.report.termination_reason = "left_screen"
                self.report.notes = f"Walked to {target} at the {direction} boundary"
                return MOVE_OK
            return code
        self._settle_free_roam()
        if self._last_terminated:
            return self._fail(MOVE_TERMINATED, "terminated")
        if self._last_truncated:
            return self._fail(MOVE_TRUNCATED, "truncated")
        self.report.termination_reason = "left_screen"
        self.report.notes = f"Walked to {target} and stepped off the screen {direction}"
        return MOVE_OK

BATTLE_ENDED = 0
BATTLE_TERMINATED = 1
BATTLE_TRUNCATED = 2
BATTLE_LEFT_TO_DIALOGUE = 3
BATTLE_LEFT_TO_MENU = 4
BATTLE_NOT_IN_BATTLE = -1
BATTLE_UNREADABLE = -2
BATTLE_ACTION_FAILED = -3
BATTLE_OUT_OF_STEPS = -4

BATTLE_EXIT_SETTLE_WAIT = 4
MAX_TURN_PROGRESS_STEPS = 30
MAX_IDLE_PROGRESS_STEPS = 3
MAX_IDLE_INTRO_STEPS = 6
MAX_STAT_DISMISS_STEPS = 4
BATTLE_OPTIONS = ("fight", "bag", "pokemon", "run", "progress")
MENU_PRESSES = ("up", "down", "left", "right", "confirm", "back")
LEFT_BATTLE_CODES = {
    AgentState.FREE_ROAM: BATTLE_ENDED,
    AgentState.IN_DIALOGUE: BATTLE_LEFT_TO_DIALOGUE,
    AgentState.IN_MENU: BATTLE_LEFT_TO_MENU,
}


def effective_agent_state(raw_state: AgentState, frame: np.ndarray) -> AgentState:
    if raw_state == AgentState.IN_MENU:
        if not has_menu_cursor(frame):
            if is_battle_screen(frame):
                return AgentState.IN_BATTLE
            if has_top_dialogue_box(frame):
                return AgentState.IN_DIALOGUE
    elif raw_state == AgentState.FREE_ROAM:
        if (has_bottom_dialogue_box(frame)
                or has_modal_preview_box(frame)
                or (has_top_dialogue_box(frame) and not has_menu_cursor(frame))):
            return AgentState.IN_DIALOGUE
    return raw_state


class CombatExecutor(PolicyExecutor):
    MAX_NEW_TOKENS = 4000
    MAX_INVALID = 3
    STEP_TAG = "battle_step"

    STEP_PROMPT = """You are playing a Game Boy Pokemon game and are in a battle.

[INSTRUCTIONS]
The image is the current battle screen.
[HISTORY]
Choose what to do next:
- fight: attack, and say which of the four moves to use (1 is the top one)
- progress: press through battle text, or continue when there is no choice to make
- run: try to escape. This never works against another trainer
- bag: open the bag to use an item; only choose it if an item is needed
- pokemon: open your team to switch; only choose it if switching is needed

Respond in exactly this format:
Reasoning: <what the screen shows and what to do about it>
Action: <one of fight, progress, run, bag, pokemon>
Move: <1, 2, 3 or 4 if the action is fight, otherwise None>
[STOP]"""

    MOVE_TAG = "battle_move"
    MOVE_ATTEMPTS = 3
    MAX_PREVIOUS_CHARS = 1500

    MOVE_PROMPT = """You are playing a Game Boy Pokemon game and chose to fight in a battle.

[INSTRUCTIONS]
The image is the current battle screen, showing your moves.
[HISTORY]
Your previous reply was:
[PREVIOUS]

That reply could not be used: it did not give the number of a move. Choose which of the four moves to use, where 1 is the top one.

Respond in exactly this format:
Reasoning: <which move to use and why>
Move: <1, 2, 3 or 4>
[STOP]"""

    MENU_STEP_TAG = "battle_menu_step"

    MENU_STEP_PROMPT = """You are playing a Game Boy Pokemon game and a menu or choice box is open during a battle.

[INSTRUCTIONS]
The image is the current screen.
[HISTORY]
Choose the single button press that best moves towards completing this menu or choice box so the battle can continue (or press back to cancel and return to the battle menu if nothing is needed here).

The presses you can make:
- up, down, left, right: move the cursor
- confirm: press A to choose what the cursor is on
- back: press B to go back, say no, or close the menu

Respond in exactly this format:
Reasoning: <what the menu or choice box shows, and what should be pressed next>
Press: <one of up, down, left, right, confirm, back>
[STOP]"""

    def __init__(self, env, task, max_steps: int = 60, **kwargs):
        self.history: List[str] = []
        self.dialogue_crops: List[Any] = []
        self.dialogue_text: List[str] = []
        self._exited_to_free_roam: bool = False
        super().__init__(env, task, max_steps, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _agent_state(self):
        return self._get_state()["pokemon_core"]["agent_state"]

    def _current_frame(self) -> np.ndarray:
        return self._get_state()["core"]["current_frame"]

    def _ensure_scripted(self, description: str) -> None:
        if self._current_call is None:
            self._begin_scripted(description)

    def _read_new_text(self, records: List[EnvironmentStepRecord], *, new_turn: bool = False) -> List[str]:
        if isinstance(records, EnvironmentStepRecord):
            records = [records]
        crops = captured_dialogue_crops(records)
        if not crops:
            return []
        if new_turn:
            fresh = drop_prefix_crops(crops)
            self.dialogue_crops.extend(fresh)
        else:
            before = len(self.dialogue_crops)
            last_before = self.dialogue_crops[-1] if self.dialogue_crops else None
            drop_prefix_crops(crops, self.dialogue_crops)
            start = before
            if last_before is not None and self.dialogue_crops[before - 1] is not last_before:
                start = before - 1
            fresh = self.dialogue_crops[start:]
        if not fresh:
            return []
        texts = ocr(fresh, self._vlm, do_merge=True, parameters=self._parameters)
        self.dialogue_text.extend(texts)
        return texts

    def _confirm_battle_exit(self, records: List[EnvironmentStepRecord]) -> bool:
        self._ensure_scripted("settle battle exit")
        state = self._agent_state()
        for _ in range(BATTLE_EXIT_SETTLE_WAIT):
            if state != AgentState.FREE_ROAM or self._last_terminated or self._last_truncated:
                break
            wait_record = self._take_action(WaitAction)
            records.append(wait_record)
            state = self._agent_state()
        if state != AgentState.IN_BATTLE and not is_battle_screen(self._current_frame()):
            self._exited_to_free_roam = True
            return True
        return False

    def _advance_until_decision(self, records: List[EnvironmentStepRecord]) -> Tuple[int, int]:
        self._ensure_scripted("advance battle text")
        progress_presses = 0
        stat_dismisses = 0
        consecutive_dismisses = 0
        idle_steps = 0
        for _ in range(MAX_TURN_PROGRESS_STEPS):
            if self._last_terminated or self._last_truncated:
                break
            state = self._agent_state()
            frame = self._current_frame()
            if state == AgentState.FREE_ROAM:
                if self._confirm_battle_exit(records):
                    break
                continue
            if state == AgentState.IN_MENU:
                if has_menu_cursor(frame):
                    break
                in_battle_intro = is_battle_screen(frame)
                if not in_battle_intro and consecutive_dismisses >= MAX_STAT_DISMISS_STEPS:
                    break
                extra = self._take_action(MenuAction, menu_action="confirm")
                records.append(extra)
                if in_battle_intro:
                    progress_presses += 1
                    consecutive_dismisses = 0
                else:
                    stat_dismisses += 1
                    consecutive_dismisses += 1
                if extra.action_success == -1:
                    break
                if is_same_battle_frame(extra.frame_before, extra.frame_after):
                    idle_steps += 1
                    if idle_steps >= (MAX_IDLE_INTRO_STEPS if in_battle_intro else MAX_IDLE_PROGRESS_STEPS):
                        break
                else:
                    idle_steps = 0
                continue
            if state == AgentState.IN_DIALOGUE:
                extra = self._take_action(PassDialogueAction)
                records.append(extra)
                progress_presses += 1
                consecutive_dismisses = 0
                idle_steps = 0
                if extra.action_success == -1:
                    break
                continue
            if is_battle_decision_screen(frame):
                break
            extra = self._take_action(BattleMenuAction, option="progress")
            records.append(extra)
            if self._last_terminated or self._last_truncated:
                break
            if extra.action_success == -1:
                break
            progress_presses += 1
            consecutive_dismisses = 0
            if is_same_battle_frame(extra.frame_before, extra.frame_after):
                idle_steps += 1
                if idle_steps >= MAX_IDLE_PROGRESS_STEPS:
                    break
            else:
                idle_steps = 0
        return progress_presses, stat_dismisses

    @staticmethod
    def _format_auto_suffix(progress_presses: int, stat_dismisses: int) -> str:
        parts = []
        if progress_presses:
            parts.append(f"pressed progress {progress_presses} time{'s' if progress_presses > 1 else ''}")
        if stat_dismisses:
            parts.append(f"dismissed {stat_dismisses} stat box{'es' if stat_dismisses > 1 else ''}")
        return (", then " + " and ".join(parts)) if parts else ""

    def _history_block(self) -> str:
        if not self.history:
            return ""
        return "\nWhat has happened in this battle so far:\n" + "\n".join(self.history) + "\n"

    def _instructions_block(self) -> str:
        task = (self._task or "").strip()
        if not task or task.lower() in ("win the current battle", "none"):
            return "Win this battle.\n"
        return f"Your instructions for this battle: \"{task}\"\n"

    def _choose_move(self, output: str) -> Optional[int]:
        move = parse_int(output, "Move", lo=1, hi=4)
        for _ in range(self.MOVE_ATTEMPTS):
            if move is not None:
                return move
            self._record_invalid(output, reason="no move number")
            frame = self._current_frame()
            prompt = (self.MOVE_PROMPT.replace("[INSTRUCTIONS]", self._instructions_block())
                      .replace("[HISTORY]", self._history_block())
                      .replace("[PREVIOUS]", output[:self.MAX_PREVIOUS_CHARS]))
            output = self._vlm_call(self.MOVE_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            move = parse_int(output, "Move", lo=1, hi=4)
        if move is None:
            self._record_invalid(output, reason="no move number")
        return move

    def _record_outcome(self, state) -> int:
        code = LEFT_BATTLE_CODES.get(state, BATTLE_ENDED)
        reason = {
            BATTLE_ENDED: "battle_ended",
            BATTLE_LEFT_TO_DIALOGUE: "left_battle_to_dialogue",
            BATTLE_LEFT_TO_MENU: "left_battle_to_menu",
        }[code]
        self.report.termination_reason = reason
        self.report.notes = "\n".join(self.dialogue_text)
        return code

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        self._exited_to_free_roam = False
        if self._agent_state() != AgentState.IN_BATTLE and not is_battle_screen(self._current_frame()):
            return self._fail(BATTLE_NOT_IN_BATTLE, "not_in_battle",
                              f"The player is {self._agent_state().name.lower()}, not in a battle.")

        if not is_battle_decision_screen(self._current_frame()):
            self._begin_scripted("advance through opening battle text")
            intro_records: List[EnvironmentStepRecord] = []
            presses, dismisses = self._advance_until_decision(intro_records)
            if self._last_terminated:
                return self._fail(BATTLE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._fail(BATTLE_TRUNCATED, "truncated")
            texts = self._read_new_text(intro_records, new_turn=True)
            if presses or dismisses or texts:
                entry = "- battle intro" + self._format_auto_suffix(presses, dismisses).replace(", then ", ": ")
                if texts:
                    entry += "\n  text on screen: " + " / ".join(texts)
                self.history.append(entry)
            if self._exited_to_free_roam:
                return self._record_outcome(AgentState.FREE_ROAM)

        consecutive_invalid = 0
        consecutive_failures = 0
        for step in range(self._max_steps):
            state = self._agent_state()
            if self._exited_to_free_roam or state == AgentState.FREE_ROAM:
                if not self._exited_to_free_roam:
                    check_records: List[EnvironmentStepRecord] = []
                    exited = self._confirm_battle_exit(check_records)
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    texts = self._read_new_text(check_records)
                    if texts:
                        self.history.append(f"- step {step + 1}: settled battle exit\n  text on screen: "
                                            + " / ".join(texts))
                    if not exited:
                        continue
                return self._record_outcome(AgentState.FREE_ROAM)

            if state == AgentState.IN_DIALOGUE:
                self._ensure_scripted("advance battle dialogue")
                turn_records: List[EnvironmentStepRecord] = []
                presses, dismisses = self._advance_until_decision(turn_records)
                if self._last_terminated:
                    return self._fail(BATTLE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._fail(BATTLE_TRUNCATED, "truncated")
                texts = self._read_new_text(turn_records)
                if presses or dismisses or texts:
                    entry = f"- step {step + 1}: advanced battle dialogue" + self._format_auto_suffix(presses, dismisses)
                    if texts:
                        entry += "\n  text on screen: " + " / ".join(texts)
                    self.history.append(entry)
                if self._exited_to_free_roam:
                    return self._record_outcome(AgentState.FREE_ROAM)
                if self._agent_state() == AgentState.IN_DIALOGUE and not presses:
                    return self._record_outcome(AgentState.IN_DIALOGUE)
                continue

            if state == AgentState.IN_MENU:
                frame = self._current_frame()
                if not has_menu_cursor(frame):
                    self._ensure_scripted("dismiss battle stat overlay")
                    turn_records = []
                    presses, dismisses = self._advance_until_decision(turn_records)
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    texts = self._read_new_text(turn_records)
                    if presses or dismisses or texts:
                        entry = f"- step {step + 1}: dismissed stat overlay" + self._format_auto_suffix(presses, dismisses)
                        if texts:
                            entry += "\n  text on screen: " + " / ".join(texts)
                        self.history.append(entry)
                    if self._exited_to_free_roam:
                        return self._record_outcome(AgentState.FREE_ROAM)
                    if self._agent_state() != AgentState.IN_MENU:
                        continue
                    frame = self._current_frame()

                prompt = (self.MENU_STEP_PROMPT.replace("[INSTRUCTIONS]", self._instructions_block())
                          .replace("[HISTORY]", self._history_block()))
                output = self._vlm_call(self.MENU_STEP_TAG, texts=prompt, images=[frame],
                                        max_new_tokens=self.MAX_NEW_TOKENS)
                press = (parse_key_value(output, "Press") or "").strip().lower()
                press = next((name for name in MENU_PRESSES if name in press), None)
                if press is None:
                    self._record_invalid(output)
                    consecutive_invalid += 1
                    if consecutive_invalid >= self.MAX_INVALID:
                        return self._fail(BATTLE_UNREADABLE, "action_unreadable", output)
                    continue
                consecutive_invalid = 0
                record = self._take_action(MenuAction, menu_action=press)
                if self._last_terminated:
                    return self._fail(BATTLE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._fail(BATTLE_TRUNCATED, "truncated")
                turn_records = [record]
                entry = f"- step {step + 1}: pressed {press} in battle menu"
                if record.action_success == -1:
                    entry += " (the game did not accept it)"
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                presses, dismisses = 0, 0
                if self._agent_state() != AgentState.IN_MENU:
                    presses, dismisses = self._advance_until_decision(turn_records)
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                entry += self._format_auto_suffix(presses, dismisses)
                texts = self._read_new_text(turn_records, new_turn=True)
                if texts:
                    entry += "\n  text on screen: " + " / ".join(texts)
                self.history.append(entry)
                if self._exited_to_free_roam:
                    return self._record_outcome(AgentState.FREE_ROAM)
                if consecutive_failures >= self.MAX_INVALID:
                    return self._fail(BATTLE_ACTION_FAILED, "actions_not_accepted", self._history_block())
                continue

            frame = self._current_frame()
            prompt = (self.STEP_PROMPT.replace("[INSTRUCTIONS]", self._instructions_block())
                      .replace("[HISTORY]", self._history_block()))
            output = self._vlm_call(self.STEP_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            option = (parse_key_value(output, "Action") or "").strip().lower()
            option = next((name for name in BATTLE_OPTIONS if name in option), None)
            if option is None:
                self._record_invalid(output)
                consecutive_invalid += 1
                if consecutive_invalid >= self.MAX_INVALID:
                    return self._fail(BATTLE_UNREADABLE, "action_unreadable", output)
                continue
            consecutive_invalid = 0

            record = self._take_action(BattleMenuAction, option=option)
            if self._last_terminated:
                return self._fail(BATTLE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._fail(BATTLE_TRUNCATED, "truncated")
            turn_records = [record]
            entry = f"- step {step + 1}: chose {option}"
            if record.action_success in (1, 2):
                entry += " (could not escape)"
                consecutive_failures = 0
            elif record.action_success == -1:
                entry += " (the game did not accept it)"
                consecutive_failures += 1
            else:
                consecutive_failures = 0

            presses, dismisses = 0, 0
            if option == "fight":
                attack = None
                if record.action_success == 0:
                    move = self._choose_move(output)
                    if move is None:
                        entry += ", but no move number was given"
                    else:
                        attack = self._take_action(PickAttackAction, option=move)
                elif record.action_success == -1:
                    move = parse_int(output, "Move", lo=1, hi=4)
                    if move is not None:
                        attack = self._take_action(PickAttackAction, option=move)
                        if attack.action_success != -1:
                            consecutive_failures = 0
                            entry = f"- step {step + 1}: chose fight"
                if attack is not None:
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    turn_records.append(attack)
                    entry += f", used move {move}"
                    if attack.action_success == 1:
                        entry += " (no PP left)"
                    elif attack.action_success == -1:
                        entry += " (the game did not accept it)"
                    else:
                        presses, dismisses = self._advance_until_decision(turn_records)
                        if self._last_terminated:
                            return self._fail(BATTLE_TERMINATED, "terminated")
                        if self._last_truncated:
                            return self._fail(BATTLE_TRUNCATED, "truncated")
            elif option in ("progress", "run") and record.action_success != -1:
                if (option == "progress"
                        and is_same_battle_frame(record.frame_before, record.frame_after)
                        and is_battle_decision_screen(self._current_frame())):
                    entry += " (a battle menu is already open waiting for a choice; choose fight, pokemon, bag, or run)"
                else:
                    presses, dismisses = self._advance_until_decision(turn_records)
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")

            entry += self._format_auto_suffix(presses, dismisses)
            texts = self._read_new_text(turn_records, new_turn=(option != "progress"))
            if texts:
                entry += "\n  text on screen: " + " / ".join(texts)
            self.history.append(entry)
            if self._exited_to_free_roam:
                return self._record_outcome(AgentState.FREE_ROAM)
            if consecutive_failures >= self.MAX_INVALID:
                return self._fail(BATTLE_ACTION_FAILED, "actions_not_accepted", self._history_block())

        return self._fail(BATTLE_OUT_OF_STEPS, "out_of_steps", self._history_block())

MENU_OK = 0
MENU_TERMINATED = 1
MENU_TRUNCATED = 2
MENU_LEFT_TO_DIALOGUE = 3
MENU_LEFT_TO_BATTLE = 4
MENU_NOT_IN_MENU = -1
MENU_OPEN_FAILED = -2
MENU_UNREADABLE = -3
MENU_OUT_OF_STEPS = -4

MENU_STATES = (AgentState.IN_MENU, AgentState.IN_BATTLE, AgentState.IN_DIALOGUE)
LEFT_MENU_OUTCOMES = {
    AgentState.FREE_ROAM: (MENU_OK, "menu_closed"),
    AgentState.IN_DIALOGUE: (MENU_LEFT_TO_DIALOGUE, "left_menu_to_dialogue"),
    AgentState.IN_BATTLE: (MENU_LEFT_TO_BATTLE, "left_menu_to_battle"),
}


class MenuExecutor(PolicyExecutor):
    MAX_NEW_TOKENS = 4000
    MAX_INVALID = 3
    OPEN_TAG = "menu_open"
    STEP_TAG = "menu_step"

    OPEN_PROMPT = """You are playing a Game Boy Pokemon game and are about to open the main menu with the Start button.

Objective: "[TASK]"

The image is the current game screen.

Which entry of the main menu should be opened first?
- pokedex: the Pokedex
- pokemon: your Pokemon team, for checking, reordering, healing or using their moves outside battle
- bag: your items
- trainer: your trainer card

Respond in exactly this format:
Reasoning: <which entry the objective needs and why>
Entry: <one of pokedex, pokemon, bag, trainer>
[STOP]"""

    STEP_PROMPT = """You are playing a Game Boy Pokemon game and are using the game's menus.

Objective: "[TASK]"

The image is the current screen.
[HISTORY]
First decide whether the objective for the menu or choice box on screen has already been achieved (including if the menu, info screen, or yes/no box has already closed into plain dialogue or the overworld). If it has not, choose the single button press that best moves towards it.

The presses you can make:
- up, down, left, right: move the cursor
- confirm: press A to choose what the cursor is on
- back: press B to go back, or to close the current menu

Respond in exactly this format:
Reasoning: <what is on the screen, and what should be pressed next>
Complete: <yes or no>
Press: <one of up, down, left, right, confirm, back>
[STOP]"""

    def __init__(self, env, task, max_steps: int = 15, is_start_menu: bool = False, **kwargs):
        self.is_start_menu = is_start_menu
        self.presses: List[str] = []
        self.dialogue_crops: List[Any] = []
        self.dialogue_text: str = ""
        super().__init__(env, task, max_steps, is_start_menu=is_start_menu, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _finish(self, code: int, reason: str) -> int:
        if self.dialogue_crops:
            self.dialogue_text = "\n".join(ocr(self.dialogue_crops, self._vlm, do_merge=True,
                                               parameters=self._parameters))
        self.report.termination_reason = reason
        self.report.notes = "\n".join(filter(None, [f"Presses: {', '.join(self.presses) or 'none'}",
                                                    self.dialogue_text]))
        return code

    def _agent_state(self):
        return self._get_state()["pokemon_core"]["agent_state"]

    def _open_main_menu(self) -> Optional[int]:
        frame = self._get_state()["core"]["current_frame"]
        output = self._vlm_call(self.OPEN_TAG, texts=self.OPEN_PROMPT.replace("[TASK]", self._task.strip()),
                                images=[frame], max_new_tokens=self.MAX_NEW_TOKENS)
        entry = (parse_key_value(output, "Entry") or "").strip().lower()
        entry = next((option for option in OpenMenuAction.options if option in entry), None)
        if entry is None:
            return self._fail(MENU_OPEN_FAILED, "menu_entry_unreadable", output)
        record = self._take_action(OpenMenuAction, option=entry)
        drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
        if self._last_terminated:
            return self._finish(MENU_TERMINATED, "terminated")
        if self._last_truncated:
            return self._finish(MENU_TRUNCATED, "truncated")
        if record.action_success != 0 or self._agent_state() not in MENU_STATES:
            return self._fail(MENU_OPEN_FAILED, "menu_did_not_open",
                              f"Tried to open {entry}, success={record.action_success}, state={self._agent_state().name}")
        return None

    def _history_block(self) -> str:
        if not self.presses:
            return ""
        return f"\nPresses made so far, in order: {', '.join(self.presses)}.\n"

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        state = self._agent_state()
        if state not in MENU_STATES:
            if not self.is_start_menu:
                return self._fail(MENU_NOT_IN_MENU, "not_in_menu",
                                  f"The objective needs a menu that is already open, but the player is {state.name.lower()}.")
            if state != AgentState.FREE_ROAM:
                return self._fail(MENU_NOT_IN_MENU, "not_free_roam",
                                  f"Cannot open the main menu while {state.name.lower()}.")
            self._begin_scripted(f"open the main menu for: {self._task}")
            code = self._open_main_menu()
            if code is not None:
                return code

        consecutive_invalid = 0
        for _ in range(self._max_steps):
            frame = self._get_state()["core"]["current_frame"]
            state = effective_agent_state(self._agent_state(), frame)
            if state != AgentState.IN_MENU:
                code, reason = LEFT_MENU_OUTCOMES.get(state, (MENU_OK, "menu_closed"))
                return self._finish(code, reason)
            prompt = (self.STEP_PROMPT.replace("[TASK]", self._task.strip())
                      .replace("[HISTORY]", self._history_block()))
            output = self._vlm_call(self.STEP_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            if parse_yes_no(output, "Complete") and (self.presses or self.is_start_menu):
                return self._finish(MENU_OK, "objective_complete")
            press = (parse_key_value(output, "Press") or "").strip().lower()
            press = next((option for option in MENU_PRESSES if option in press), None)
            if press is None:
                self._record_invalid(output)
                consecutive_invalid += 1
                if consecutive_invalid >= self.MAX_INVALID:
                    return self._fail(MENU_UNREADABLE, "press_unreadable", output)
                continue
            consecutive_invalid = 0
            self.presses.append(press)
            record = self._take_action(MenuAction, menu_action=press)
            drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
            if self._last_terminated:
                return self._finish(MENU_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(MENU_TRUNCATED, "truncated")
            state = effective_agent_state(self._agent_state(), self._get_state()["core"]["current_frame"])
            if state != AgentState.IN_MENU:
                code, reason = LEFT_MENU_OUTCOMES.get(state, (MENU_OK, "menu_closed"))
                return self._finish(code, reason)
        return self._finish(MENU_OUT_OF_STEPS, "out_of_steps")


DIALOGUE_CLEARED = 0
DIALOGUE_TERMINATED = 1
DIALOGUE_TRUNCATED = 2
DIALOGUE_LEFT_TO_MENU = 3
DIALOGUE_LEFT_TO_BATTLE = 4
DIALOGUE_NOT_IN_DIALOGUE = -1
DIALOGUE_STUCK = -2

DIALOGUE_SETTLE_WAIT = 6
DIALOGUE_MAX_CUTSCENE_WAIT = 24
DIALOGUE_POST_MOTION_WAIT = 4
DIALOGUE_FANFARE_RETRY_LIMIT = 4
DIALOGUE_BUTTON_SETTLE_WAIT = 2
DIALOGUE_TOP_BOX_PRESSES = 4
FREE_SETTLE_WAIT = 4
MENU_SETTLE_WAIT = 1

LEFT_DIALOGUE_CODES = {
    AgentState.FREE_ROAM: DIALOGUE_CLEARED,
    AgentState.IN_MENU: DIALOGUE_LEFT_TO_MENU,
    AgentState.IN_BATTLE: DIALOGUE_LEFT_TO_BATTLE,
}


class _DialogueSettlingExecutor(PolicyExecutor):
    def _agent_state(self):
        return self._get_state()["pokemon_core"]["agent_state"]

    def _current_frame(self):
        return self._get_state()["core"]["current_frame"]

    def _is_top_box_dialogue(self, frame=None) -> bool:
        frame = self._current_frame() if frame is None else frame
        return has_top_dialogue_box(frame) and not has_menu_cursor(frame)

    def _effective_state(self, frame=None, raw_state=None) -> AgentState:
        raw = self._agent_state() if raw_state is None else raw_state
        frame = self._current_frame() if frame is None else frame
        return effective_agent_state(raw, frame)

    def _wait_for_next_box(self, settle_wait: int = DIALOGUE_SETTLE_WAIT):
        state = self._agent_state()
        waited = 0
        post_motion_remaining = 0
        while (state == AgentState.FREE_ROAM
               and not (self._last_terminated or self._last_truncated)
               and waited < DIALOGUE_MAX_CUTSCENE_WAIT):
            frame = self._current_frame()
            box_open = has_bottom_dialogue_box(frame) or self._is_top_box_dialogue(frame)
            if waited >= settle_wait and not box_open and post_motion_remaining <= 0:
                break
            wait_record = self._take_action(WaitAction)
            waited += 1
            drop_prefix_crops(captured_dialogue_crops([wait_record]), self.dialogue_crops)
            state = self._agent_state()
            if state != AgentState.FREE_ROAM or self._last_terminated or self._last_truncated:
                break
            if (is_cutscene_frame_changing(wait_record.frame_before, wait_record.frame_after)
                    or has_bottom_dialogue_box(wait_record.frame_after)
                    or self._is_top_box_dialogue(wait_record.frame_after)):
                post_motion_remaining = DIALOGUE_POST_MOTION_WAIT
            elif post_motion_remaining > 0:
                post_motion_remaining -= 1
        return state

    def _settle_state(self):
        state = self._agent_state()
        for _ in range(MENU_SETTLE_WAIT):
            if state != AgentState.IN_MENU or self._last_terminated or self._last_truncated:
                break
            wait_record = self._take_action(WaitAction)
            drop_prefix_crops(captured_dialogue_crops([wait_record]), self.dialogue_crops)
            state = self._agent_state()
        return state

    def _settle_after_step(self, settle_wait: int = DIALOGUE_SETTLE_WAIT):
        state = self._agent_state()
        if state == AgentState.FREE_ROAM:
            state = self._wait_for_next_box(settle_wait=settle_wait)
        if state == AgentState.IN_MENU:
            state = self._settle_state()
        return state

    def _advance_dialogue_box(self, button: str = "a") -> bool:
        if self._agent_state() == AgentState.IN_MENU and self._is_top_box_dialogue():
            menu_action = "back" if button == "b" else "confirm"
            progressed = False
            for _ in range(DIALOGUE_TOP_BOX_PRESSES):
                record = self._take_action(MenuAction, menu_action=menu_action)
                drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
                if record.action_success != -1 or record.frame_changed:
                    progressed = True
                if self._last_terminated or self._last_truncated:
                    return True
                if self._agent_state() != AgentState.IN_MENU or not self._is_top_box_dialogue():
                    self._settle_after_step()
                    return True
            return progressed
        record = self._take_action(PassDialogueAction)
        drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
        if self._last_terminated or self._last_truncated:
            return True
        state = self._settle_after_step()
        if self._last_terminated or self._last_truncated or state != AgentState.IN_DIALOGUE:
            return True
        if record.action_success != -1:
            return True
        for _ in range(DIALOGUE_FANFARE_RETRY_LIMIT):
            before_btn = self._current_frame()
            btn_record = self._take_action_string(f"button({button})")
            if btn_record is None:
                return False
            drop_prefix_crops(captured_dialogue_crops([btn_record]), self.dialogue_crops)
            if self._last_terminated or self._last_truncated:
                return True
            state = self._settle_after_step()
            if self._last_terminated or self._last_truncated or state != AgentState.IN_DIALOGUE:
                return True
            if btn_record.frame_changed or is_cutscene_frame_changing(before_btn, self._current_frame()):
                for _ in range(DIALOGUE_BUTTON_SETTLE_WAIT):
                    if self._agent_state() != AgentState.IN_DIALOGUE or self._last_terminated or self._last_truncated:
                        break
                    wait_record = self._take_action(WaitAction)
                    drop_prefix_crops(captured_dialogue_crops([wait_record]), self.dialogue_crops)
                if self._last_terminated or self._last_truncated:
                    return True
                self._settle_after_step()
                return True
            retry = self._take_action(PassDialogueAction)
            drop_prefix_crops(captured_dialogue_crops([retry]), self.dialogue_crops)
            if self._last_terminated or self._last_truncated:
                return True
            state = self._settle_after_step()
            if self._last_terminated or self._last_truncated or state != AgentState.IN_DIALOGUE:
                return True
            if retry.action_success != -1 or retry.frame_changed:
                return True
        return False


class DialogueExecutor(_DialogueSettlingExecutor):
    def __init__(self, env, task, max_steps: int = 10, initial_dialogue_crops: Optional[List[Any]] = None, **kwargs):
        self.dialogue_crops: List[Any] = list(initial_dialogue_crops) if initial_dialogue_crops else []
        self.dialogue_text: str = ""
        super().__init__(env, task, max_steps, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _finish(self, code: int, reason: str) -> int:
        if self.dialogue_crops:
            self.dialogue_text = "\n".join(ocr(self.dialogue_crops, self._vlm, do_merge=True,
                                               parameters=self._parameters))
        self.report.termination_reason = reason
        self.report.notes = self.dialogue_text
        return code

    def _in_dialogue_loop_state(self, state: AgentState) -> bool:
        return state == AgentState.IN_DIALOGUE or (state == AgentState.IN_MENU and self._is_top_box_dialogue())

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        self._begin_scripted(f"pass the dialogue: {self._task}")
        state = self._agent_state()
        if state == AgentState.FREE_ROAM:
            state = self._wait_for_next_box()
            if self._last_terminated:
                return self._finish(DIALOGUE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(DIALOGUE_TRUNCATED, "truncated")
        if not self._in_dialogue_loop_state(state):
            frame = self._current_frame()
            if state == AgentState.FREE_ROAM and (has_bottom_dialogue_box(frame)
                                                  or has_modal_preview_box(frame)
                                                  or self._is_top_box_dialogue(frame)):
                a_record = self._take_action_string("button(a)")
                if a_record is not None:
                    drop_prefix_crops(captured_dialogue_crops([a_record]), self.dialogue_crops)
                if self._last_terminated:
                    return self._finish(DIALOGUE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._finish(DIALOGUE_TRUNCATED, "truncated")
                state = self._settle_after_step()
                if self._last_terminated:
                    return self._finish(DIALOGUE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._finish(DIALOGUE_TRUNCATED, "truncated")
                if not self._in_dialogue_loop_state(state):
                    eff_state = self._effective_state(raw_state=state)
                    code = LEFT_DIALOGUE_CODES.get(eff_state, DIALOGUE_CLEARED)
                    return self._finish(code, {
                        DIALOGUE_CLEARED: "dialogue_cleared",
                        DIALOGUE_LEFT_TO_MENU: "left_dialogue_to_menu",
                        DIALOGUE_LEFT_TO_BATTLE: "left_dialogue_to_battle",
                    }[code])
            else:
                return self._fail(DIALOGUE_NOT_IN_DIALOGUE, "not_in_dialogue",
                                  f"The player is {state.name.lower()}, not in dialogue.")

        for _ in range(self._max_steps):
            advanced = self._advance_dialogue_box("a")
            if self._last_terminated:
                return self._finish(DIALOGUE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(DIALOGUE_TRUNCATED, "truncated")
            state = self._agent_state()
            if not self._in_dialogue_loop_state(state):
                eff_state = self._effective_state(raw_state=state)
                code = LEFT_DIALOGUE_CODES.get(eff_state, DIALOGUE_CLEARED)
                return self._finish(code, {
                    DIALOGUE_CLEARED: "dialogue_cleared",
                    DIALOGUE_LEFT_TO_MENU: "left_dialogue_to_menu",
                    DIALOGUE_LEFT_TO_BATTLE: "left_dialogue_to_battle",
                }[code])
            if not advanced:
                return self._finish(DIALOGUE_STUCK, "dialogue_stuck")
        return self._finish(DIALOGUE_STUCK, "out_of_steps")


FREE_OK = 0
FREE_TERMINATED = 1
FREE_TRUNCATED = 2
FREE_UNREADABLE = -1
FREE_OUT_OF_STEPS = -2
FREE_BUTTONS = ("a", "b", "start", "up", "down", "left", "right")


class FreeExecutor(_DialogueSettlingExecutor):
    MAX_NEW_TOKENS = 4000
    MAX_INVALID = 3
    STEP_TAG = "free_step"

    FIRST_STEP_PROMPT = """You are playing a Game Boy Pokemon game, pressing one button at a time.

What you have been asked to do: "[TASK]"

Stop when: [TERMINATION]

The image is the current screen.
Choose the single button press that best moves towards what you have been asked to do.

The buttons are: a, b, start, up, down, left, right.

Respond in exactly this format:
Reasoning: <what is on the screen, and which button to press first>
Press: <one of a, b, start, up, down, left, right>
[STOP]"""

    STEP_PROMPT = """You are playing a Game Boy Pokemon game, pressing one button at a time.

What you have been asked to do: "[TASK]"

Stop when: [TERMINATION]

The image is the current screen.
[HISTORY]
First decide, from this screen, whether the condition to stop at has been met (a bordered dialogue box at the bottom of the screen—even if blank or mid-text—means dialogue is still open). If it has not, choose the single button press that best moves towards it.

The buttons are: a, b, start, up, down, left, right.

Respond in exactly this format:
Reasoning: <what is on the screen, and whether the stopping condition is met>
Complete: <yes or no>
Press: <one of a, b, start, up, down, left, right>
[STOP]"""

    DEFAULT_TERMINATION = "what you were asked to do has visibly happened on screen"

    MAX_REPEATED_BLOCKED_PRESSES = 2

    def __init__(self, env, task, max_steps: int = 15, termination: Optional[str] = None, **kwargs):
        self.presses: List[str] = []
        self.dialogue_crops: List[Any] = []
        self.dialogue_text: str = ""
        self._blocked_at_place: Dict[Any, List[str]] = {}
        self._repeated_blocked_presses: int = 0
        self.termination = (termination or "").strip() or self.DEFAULT_TERMINATION
        super().__init__(env, task, max_steps, termination=self.termination, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _history_block(self) -> str:
        if not self.presses:
            return ""
        lines = [f"Buttons pressed so far, in order: {', '.join(self.presses)}."]
        cur_place = self._current_map_place()
        blocked_here = self._blocked_at_place.get(cur_place, []) if cur_place is not None else []
        if blocked_here:
            lines.append(
                f"Blocked directions at current position {cur_place[1]} (pressing these did NOT move the player — "
                f"do NOT press them again from here): {', '.join(blocked_here)}."
            )
        return "\n" + "\n".join(lines) + "\n"

    def _finish(self, code: int, reason: str) -> int:
        if self.dialogue_crops:
            self.dialogue_text = "\n".join(ocr(self.dialogue_crops, self._vlm, do_merge=True,
                                               parameters=self._parameters))
        self.report.termination_reason = reason
        cur_place = self._current_map_place()
        blocked_here = self._blocked_at_place.get(cur_place, []) if cur_place is not None else []
        blocked_note = f"Blocked moving {', '.join(blocked_here)} at {cur_place[1]}" if blocked_here else ""
        self.report.notes = "\n".join(filter(None, [f"Presses: {', '.join(self.presses) or 'none'}",
                                                    blocked_note,
                                                    self.dialogue_text]))
        return code

    @staticmethod
    def _default_first_press(task: str) -> Optional[str]:
        lowered = task.lower()
        for btn in ("up", "down", "left", "right", "start", "a", "b"):
            if f"press {btn}" in lowered or f"pressing {btn}" in lowered:
                return btn
        return None

    def _current_map_place(self) -> Optional[Tuple[int, Tuple[int, int]]]:
        gmap = getattr(self._env, "_map", None)
        if gmap is None or not getattr(gmap, "current", None) or getattr(gmap, "pos", None) is None:
            return None
        try:
            return gmap.resolve(gmap.segment, gmap.pos)
        except Exception:  # noqa: BLE001
            return None

    def _step_press(self, press: str) -> bool:
        state = self._agent_state()
        if press in ("a", "b") and (state == AgentState.IN_DIALOGUE
                                    or (state == AgentState.IN_MENU and self._is_top_box_dialogue())):
            self._advance_dialogue_box(press)
            return True
        was_free = (
            state == AgentState.FREE_ROAM
            and not has_bottom_dialogue_box(self._current_frame())
            and not self._is_top_box_dialogue()
        )
        before_place = self._current_map_place() if (was_free and press in DIRECTIONS) else None
        was_facing = getattr(self._env, "facing", None) if (was_free and press in DIRECTIONS) else None
        record = self._take_action_string(f"button({press})") if press in FREE_BUTTONS else None
        if record is None:
            return False
        drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
        if self._last_terminated or self._last_truncated:
            return True
        after_place = self._current_map_place() if before_place is not None else None
        if before_place is not None and after_place != before_place:
            self._repeated_blocked_presses = 0
        elif (
            before_place is not None
            and ((self.presses and self.presses[-1] == press) or was_facing == press)
            and self._agent_state() == AgentState.FREE_ROAM
            and not has_bottom_dialogue_box(self._current_frame())
            and not self._is_top_box_dialogue()
            and after_place == before_place
        ):
            blocked_list = self._blocked_at_place.setdefault(before_place, [])
            if press in blocked_list:
                self._repeated_blocked_presses += 1
            else:
                blocked_list.append(press)
            recorder = getattr(self._env, "record_blocked_step", None)
            if callable(recorder):
                try:
                    recorder(direction=press)
                except Exception:  # noqa: BLE001
                    pass
        if press in ("a", "b") and self._agent_state() == AgentState.FREE_ROAM:
            self._settle_after_step(settle_wait=FREE_SETTLE_WAIT)
        elif self._agent_state() == AgentState.IN_MENU:
            self._settle_state()
        return True

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        consecutive_invalid = 0
        for _ in range(self._max_steps):
            frame = self._get_state()["core"]["current_frame"]
            template = self.STEP_PROMPT if self.presses else self.FIRST_STEP_PROMPT
            prompt = (template.replace("[TASK]", self._task.strip())
                      .replace("[TERMINATION]", self.termination)
                      .replace("[HISTORY]", self._history_block()))
            output = self._vlm_call(self.STEP_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            if self.presses and parse_yes_no(output, "Complete"):
                return self._finish(FREE_OK, "termination_met")
            answer = (parse_key_value(output, "Press") or "").strip().lower().strip(".")
            press = answer.split()[0] if answer else ""
            if press not in FREE_BUTTONS and not self.presses:
                press = self._default_first_press(self._task) or ""
            if not self._step_press(press):
                self._record_invalid(output)
                consecutive_invalid += 1
                if consecutive_invalid >= self.MAX_INVALID:
                    return self._finish(FREE_UNREADABLE, "press_unreadable")
                continue
            consecutive_invalid = 0
            self.presses.append(press)
            if self._last_terminated:
                return self._finish(FREE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(FREE_TRUNCATED, "truncated")
            if self._repeated_blocked_presses >= self.MAX_REPEATED_BLOCKED_PRESSES:
                return self._finish(FREE_OUT_OF_STEPS, "out_of_steps")
        return self._finish(FREE_OUT_OF_STEPS, "out_of_steps")
