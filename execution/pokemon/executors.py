import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from execution.game_api import (AgentState, BattleMenuAction, InteractAction, LowLevelAction, LowLevelActions,
                                MenuAction, MoveStepsAction, OpenMenuAction, PassDialogueAction, PickAttackAction,
                                WaitAction)

from execution.executors import PolicyExecutor
from execution.perception.pokemon.battle import (has_bottom_dialogue_box, has_bottom_location_banner,
                                                 has_emote_bubble, has_menu_cursor, has_modal_preview_box,
                                                 has_player_pokemon_fainted_text, has_quantity_selector_box,
                                                 has_top_dialogue_box,
                                                 is_battle_command_menu, is_battle_decision_screen,
                                                 is_battle_defeat_text, is_battle_menu_crop,
                                                 is_battle_opponent_critical_hp,
                                                 is_battle_screen, is_blank_or_fade_frame,
                                                 is_cutscene_frame_changing,
                                                 is_screen_wide_cutscene_change,
                                                 is_empty_battle_bag_menu, is_evolution_screen,
                                                 is_false_overworld_menu_state,
                                                 is_gen1_item_list_screen, is_move_selection_screen,
                                                 is_no_pp_dialogue_frame, is_overworld_confirmation_subbox,
                                                 is_overworld_menu_or_ui_screen, is_overworld_pack_menu,
                                                 is_pack_action_subbox_screen, is_pack_item_list_screen,
                                                 is_pack_screen, is_party_menu_screen,
                                                 has_party_subbox,
                                                 is_party_menu_selection_screen, is_same_battle_frame,
                                                 is_start_menu_screen, is_tm_hm_item_name,
                                                 is_tm_hm_pack_pocket, is_tm_hm_startup_dialogue_frame,
                                                 healing_item_restore_amount,
                                                 is_healing_item_name, is_item_use_subbox_screen,
                                                 is_party_boost_item_name, is_party_consumable_item_name,
                                                 is_status_heal_item_for_status,
                                                 is_status_heal_item_name,
                                                 pack_action_box_height,
                                                 read_bag_cursor_row, read_bag_cursor_row_index,
                                                 read_bag_item_list,
                                                 read_bag_item_quantities, read_battle_hud_hp,
                                                 read_battle_hud_level, read_battle_hud_status,
                                                 read_battle_opponent_hud_level, read_dialogue_box_lines,
                                                 read_menu_cursor_item_name,
                                                 read_battle_opponent_hud_name,
                                                 read_move_menu_cursor_slot,
                                                 read_move_menu_names, read_move_menu_pp,
                                                 read_move_menu_type,
                                                 read_party_menu_compatibility,
                                                 read_party_menu_cursor_slot, read_party_menu_hp,
                                                 read_party_menu_level,
                                                 read_party_menu_members, read_party_menu_status,
                                                 read_move_learning_dialogue_status,
                                                 read_party_move_forget_list, read_quantity_selector_value,
                                                 read_tm_hm_pocket_rows)
from execution.perception.pokemon.navigation import (BLOCKED, DIRECTIONS, LEDGE, LEDGE_LEFT, WALKABLE, approach_cells,
                                                     boundary_exit_direction, cell_kinds, compress_path,
                                                     diagonal_barrier_pairs, edge_cells, edge_runs,
                                                     filter_forward_edge_cells, filter_open_lane_edge_cells, in_bounds,
                                                     is_bottom_edge_overhang_only_cell, is_boundary_exit_direction,
                                                     is_ledge_roofed_edge_cell, is_on_edge, parse_all_cells,
                                                     parse_avoided_cells, parse_cell, parse_edge_direction,
                                                     resolve_interaction_target_and_constraints, shortest_path,
                                                     verbalize_edges, warp_cells)
from execution.executor_action import coords_to_string
from execution.pokemon.executor_actions import CheckInteractionAction
from execution.report import EnvironmentStepRecord, ExecutorVLMCallRecord
from execution.strategist.pokemon.party_state import PartyInventoryState
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


DIALOGUE_CROP_TOP = 100
DIALOGUE_CROP_BOTTOM = 140
DIALOGUE_CROP_LEFT = 5
DIALOGUE_CROP_RIGHT = 155


def extract_bottom_dialogue_crop(frame: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if frame is None or not hasattr(frame, "shape") or frame.ndim < 2:
        return None
    if frame.shape[0] < DIALOGUE_CROP_BOTTOM or frame.shape[1] < DIALOGUE_CROP_RIGHT:
        return None
    if not has_bottom_dialogue_box(frame) or is_top_box_hangup_or_blank_bottom_box(frame):
        return None
    grey = frame[:, :, 0:1] if frame.ndim == 3 else frame[:, :, None]
    crop = np.ascontiguousarray(
        grey[DIALOGUE_CROP_TOP:DIALOGUE_CROP_BOTTOM, DIALOGUE_CROP_LEFT:DIALOGUE_CROP_RIGHT]
    )
    if is_blank_crop(crop) or is_battle_menu_crop(crop):
        return None
    return crop


def is_top_box_hangup_or_blank_bottom_box(frame: Optional[np.ndarray]) -> bool:
    if frame is None or not hasattr(frame, "shape") or frame.ndim < 2:
        return True
    if not has_bottom_dialogue_box(frame):
        return True
    grey = frame[:, :, 0] if frame.ndim == 3 else frame
    if grey.shape[0] < 136 or grey.shape[1] < 104:
        return True
    ink_mask = grey[104:136, 8:104] < 128
    ink_count = int(ink_mask.sum())
    if ink_count <= 16:
        return True
    ys = np.where(ink_mask)[0]
    return len(ys) == 0 or (int(ys.max()) - int(ys.min()) + 1) <= 2


def captured_dialogue_crops(records: List[EnvironmentStepRecord]) -> List[Any]:
    crops = []
    for record in records:
        rec_crops = []
        for state in record.transition_states:
            step_crops = state.get("ocr", {}).get("ocr_regions", {}).get("dialogue")
            if step_crops is None or len(step_crops) == 0:
                continue
            rec_crops.extend(step_crops)
        if not rec_crops:
            for candidate_frame in (getattr(record, "frame_before", None), getattr(record, "frame_after", None)):
                if (
                    candidate_frame is not None
                    and has_top_dialogue_box(candidate_frame)
                    and not has_menu_cursor(candidate_frame)
                ):
                    crop = extract_bottom_dialogue_crop(candidate_frame)
                    if crop is not None:
                        rec_crops.append(crop)
        crops.extend(rec_crops)
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
INTERACT_POST_DIALOGUE_SETTLE_WAIT = 5
INTERACT_MAX_CUTSCENE_WAIT = 30
INTERACT_POST_MOTION_WAIT = 5
INTERACT_POST_WARP_BANNER_WAIT = 6
PURE_DIFF_MAX_CUTSCENE_WAIT = 10
INTERACT_MODAL_MAX_PRESSES = 8
MAX_INTERACT_DIALOGUE_STEPS = 10


def settle_overworld_menu_state(
    *,
    get_state_fn,
    get_frame_fn,
    is_done_fn,
    step_wait_fn,
    min_wait: int = 1,
    settle_wait: int = 2,
    max_cutscene_wait: int = INTERACT_MAX_CUTSCENE_WAIT,
    post_motion_wait: int = INTERACT_POST_MOTION_WAIT,
    require_waited_for_ready_check: bool = False,
) -> AgentState:
    """Shared overworld IN_MENU cutscene settling loop across executors and supervisor."""
    state = get_state_fn()
    waited = 0
    post_motion_remaining = 0
    while state == AgentState.IN_MENU and not is_done_fn() and waited < max_cutscene_wait:
        frame = get_frame_fn()
        ready = (
            has_menu_cursor(frame)
            or is_battle_screen(frame)
            or (has_top_dialogue_box(frame) and not has_menu_cursor(frame))
            or has_top_dialogue_box(frame)
            or is_false_overworld_menu_state(frame)
        )
        if (not require_waited_for_ready_check or waited > 0) and ready:
            break
        if waited >= min_wait and post_motion_remaining <= 0:
            if ready or waited >= settle_wait:
                break
        before_frame, after_frame = step_wait_fn()
        waited += 1
        state = get_state_fn()
        if state != AgentState.IN_MENU or is_done_fn():
            break
        if (
            has_menu_cursor(after_frame)
            or is_battle_screen(after_frame)
            or has_top_dialogue_box(after_frame)
            or is_false_overworld_menu_state(after_frame)
        ):
            break
        if is_cutscene_frame_changing(before_frame, after_frame):
            post_motion_remaining = post_motion_wait
        elif post_motion_remaining > 0:
            post_motion_remaining -= 1
    return state


def settle_overworld_free_roam(
    *,
    env,
    get_state_fn,
    get_frame_fn,
    is_done_fn,
    step_wait_fn,
    settle_wait: int = 2,
    max_cutscene_wait: int = INTERACT_MAX_CUTSCENE_WAIT,
    post_motion_wait: int = INTERACT_POST_MOTION_WAIT,
    post_warp_banner_wait: int = INTERACT_POST_WARP_BANNER_WAIT,
    stop_on_overlay: bool = False,
    include_top_box: bool = True,
    record_dialogue_settled: bool = False,
    respect_last_settled: bool = False,
    settle_menu_fn=None,
) -> Tuple[AgentState, int, bool]:
    """Shared overworld FREE_ROAM cutscene and pre-battle settling state machine."""
    state = get_state_fn()
    cur_step = getattr(getattr(env, "_emulator", None), "step_count", None)
    if respect_last_settled and state == AgentState.FREE_ROAM and cur_step is not None:
        env_settled = getattr(env, "_last_settled_step", None)
        if cur_step == env_settled:
            before = get_frame_fn()
            if not (
                has_bottom_dialogue_box(before)
                or has_emote_bubble(before)
                or is_blank_or_fade_frame(before)
                or (include_top_box and has_top_dialogue_box(before) and not has_menu_cursor(before))
            ):
                return state, 0, False
        env_dialogue_settled = getattr(env, "_last_dialogue_settled_step", None)
        if cur_step == env_dialogue_settled:
            settle_wait = min(settle_wait, 1)

    waited = 0
    post_motion_remaining = 0
    saw_fade = False
    saw_overlay_or_fade = False
    saw_emote_or_dialogue = False
    saw_activity = False
    pure_move_settle = not record_dialogue_settled and not stop_on_overlay
    seen_frames: List[np.ndarray] = []

    def _is_overlay(fr) -> bool:
        return (
            has_bottom_dialogue_box(fr)
            or has_modal_preview_box(fr)
            or (has_top_dialogue_box(fr) and not has_menu_cursor(fr))
        )

    while state == AgentState.FREE_ROAM and not is_done_fn() and waited < max_cutscene_wait:
        frame = get_frame_fn()
        if stop_on_overlay and waited > 0 and (_is_overlay(frame) or is_battle_screen(frame)):
            break
        fade_open = is_blank_or_fade_frame(frame)
        if fade_open:
            saw_fade = True
        emote_or_box_open = (
            has_emote_bubble(frame)
            or (
                not stop_on_overlay
                and (
                    has_bottom_dialogue_box(frame)
                    or (include_top_box and has_top_dialogue_box(frame) and not has_menu_cursor(frame))
                )
            )
        )
        if emote_or_box_open:
            saw_emote_or_dialogue = True
        box_open = emote_or_box_open or fade_open
        if box_open:
            saw_activity = True
            saw_overlay_or_fade = True
        elif frame is not None and not seen_frames:
            seen_frames.append(frame)
        if pure_move_settle and not box_open and not saw_emote_or_dialogue and waited >= min(max_cutscene_wait, PURE_DIFF_MAX_CUTSCENE_WAIT):
            break
        if waited >= settle_wait and not box_open and post_motion_remaining <= 0:
            break
        before_frame, after_frame = step_wait_fn()
        waited += 1
        state = get_state_fn()
        if state == AgentState.IN_MENU and is_false_overworld_menu_state(after_frame):
            state = AgentState.FREE_ROAM
        if state == AgentState.IN_MENU and settle_menu_fn is not None and not is_done_fn():
            state = settle_menu_fn()
            after_frame = get_frame_fn()
            if state == AgentState.IN_MENU and is_false_overworld_menu_state(after_frame):
                state = AgentState.FREE_ROAM
        if state != AgentState.FREE_ROAM or is_done_fn():
            break
        if stop_on_overlay and (_is_overlay(after_frame) or is_battle_screen(after_frame)):
            break
        fade_now = is_blank_or_fade_frame(before_frame) or is_blank_or_fade_frame(after_frame)
        if fade_now:
            saw_fade = True
        emote_or_dialogue_now = (
            has_emote_bubble(before_frame)
            or has_emote_bubble(after_frame)
            or (not stop_on_overlay and has_bottom_dialogue_box(after_frame))
            or (not stop_on_overlay and include_top_box and has_top_dialogue_box(after_frame) and not has_menu_cursor(after_frame))
        )
        if emote_or_dialogue_now:
            saw_emote_or_dialogue = True
        overlay_or_emote_now = emote_or_dialogue_now or fade_now
        if overlay_or_emote_now:
            saw_overlay_or_fade = True
        elif after_frame is not None:
            seen_frames.append(after_frame)
        frame_changing = is_cutscene_frame_changing(before_frame, after_frame, seen_frames=seen_frames)
        screen_wide = frame_changing and is_screen_wide_cutscene_change(before_frame, after_frame)
        if screen_wide:
            saw_overlay_or_fade = True
        active_now = frame_changing or overlay_or_emote_now
        if active_now:
            saw_activity = True
            if saw_fade and has_bottom_location_banner(after_frame):
                post_motion_remaining = max(post_motion_remaining, post_warp_banner_wait)
            elif pure_move_settle and not saw_emote_or_dialogue:
                if fade_now or screen_wide:
                    post_motion_remaining = max(post_motion_remaining, post_motion_wait)
                else:
                    post_motion_remaining = max(post_motion_remaining - 1, 1)
            else:
                post_motion_remaining = max(post_motion_remaining, post_motion_wait)
        elif post_motion_remaining > 0:
            post_motion_remaining -= 1

    if state == AgentState.IN_MENU and settle_menu_fn is not None and not is_done_fn():
        state = settle_menu_fn()
    if state == AgentState.FREE_ROAM and not is_done_fn():
        end_step = getattr(getattr(env, "_emulator", None), "step_count", None)
        if end_step is not None:
            try:
                if record_dialogue_settled:
                    if waited >= max(5, settle_wait):
                        env._last_settled_step = end_step
                    elif waited >= settle_wait:
                        env._last_dialogue_settled_step = end_step
                else:
                    env._last_settled_step = end_step
            except Exception:  # noqa: BLE001
                pass
    return state, waited, saw_activity


class InteractionExecutor(PolicyExecutor):
    CARDINAL_DIRECTIONS = {cell: name for name, cell in CheckInteractionAction.cardinals.items()}
    MAX_ALIGN_STEPS = 10

    def __init__(self, env, task, max_steps: int = 3, identified=None, cell_kinds=None, people_fn=None, **kwargs):
        self.identified = identified
        self.cell_kinds = cell_kinds
        self.people_fn = people_fn
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
        return (has_bottom_dialogue_box(frame)
                or has_modal_preview_box(frame)
                or (has_top_dialogue_box(frame) and not has_menu_cursor(frame)))

    def _step_wait_record(self, records: List[EnvironmentStepRecord]) -> Tuple[np.ndarray, np.ndarray]:
        wait_record = self._take_action(WaitAction)
        records.append(wait_record)
        return wait_record.frame_before, wait_record.frame_after

    def _wait_for_interaction(self, records: List[EnvironmentStepRecord], settle_wait: int = INTERACT_SETTLE_WAIT):
        state, _, _ = settle_overworld_free_roam(
            env=self._env,
            get_state_fn=self._agent_state,
            get_frame_fn=self._current_frame,
            is_done_fn=lambda: bool(self._last_terminated or self._last_truncated),
            step_wait_fn=lambda: self._step_wait_record(records),
            settle_wait=settle_wait,
            max_cutscene_wait=INTERACT_MAX_CUTSCENE_WAIT,
            post_motion_wait=INTERACT_POST_MOTION_WAIT,
            post_warp_banner_wait=INTERACT_POST_WARP_BANNER_WAIT,
            stop_on_overlay=True,
            record_dialogue_settled=True,
        )
        return state

    def _task_allows_affirmative_confirmation(self) -> bool:
        raw = self._task or ""
        return not bool(
            re.search(
                r"\b(?:select\s+no|choose\s+no|say\s+no|decline|refuse|cancel)\b",
                raw,
                re.IGNORECASE,
            )
        )

    def _confirm_interaction_prompt_if_applicable(self, records: List[EnvironmentStepRecord]) -> AgentState:
        state = self._agent_state()
        if (
            state != AgentState.IN_MENU
            or self._last_terminated
            or self._last_truncated
            or not self._task_allows_affirmative_confirmation()
        ):
            return state
        frame = self._current_frame()
        if not is_overworld_confirmation_subbox(frame):
            return state
        confirm_rec = self._take_action(MenuAction, menu_action="confirm")
        records.append(confirm_rec)
        if self._last_terminated or self._last_truncated:
            return self._agent_state()
        state = self._agent_state()
        if state == AgentState.FREE_ROAM:
            self._wait_for_interaction(records, settle_wait=INTERACT_POST_DIALOGUE_SETTLE_WAIT)
            if self._agent_state() == AgentState.FREE_ROAM and self._has_interaction_overlay():
                self._advance_interaction_overlay(records)
            state = self._agent_state()
        if state == AgentState.IN_MENU and not (self._last_terminated or self._last_truncated):
            state = settle_overworld_menu_state(
                get_state_fn=self._agent_state,
                get_frame_fn=self._current_frame,
                is_done_fn=lambda: bool(self._last_terminated or self._last_truncated),
                step_wait_fn=lambda: self._step_wait_record(records),
                min_wait=MENU_SETTLE_WAIT,
                settle_wait=INTERACT_POST_DIALOGUE_SETTLE_WAIT,
                max_cutscene_wait=INTERACT_MAX_CUTSCENE_WAIT,
                post_motion_wait=INTERACT_POST_MOTION_WAIT,
            )
        return self._agent_state()

    def _settle_interaction_menu_state(self, records: List[EnvironmentStepRecord]):
        state = settle_overworld_menu_state(
            get_state_fn=self._agent_state,
            get_frame_fn=self._current_frame,
            is_done_fn=lambda: bool(self._last_terminated or self._last_truncated),
            step_wait_fn=lambda: self._step_wait_record(records),
            min_wait=MENU_SETTLE_WAIT,
            settle_wait=INTERACT_POST_DIALOGUE_SETTLE_WAIT,
            max_cutscene_wait=INTERACT_MAX_CUTSCENE_WAIT,
            post_motion_wait=INTERACT_POST_MOTION_WAIT,
        )
        if state == AgentState.IN_MENU and not (self._last_terminated or self._last_truncated):
            state = self._confirm_interaction_prompt_if_applicable(records)
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
                state = self._wait_for_interaction(records, settle_wait=INTERACT_POST_DIALOGUE_SETTLE_WAIT)
        return state

    def _settle_after_interaction_dialogue_step(self, records: List[EnvironmentStepRecord]) -> AgentState:
        state = self._agent_state()
        if state == AgentState.FREE_ROAM:
            self._wait_for_interaction(records, settle_wait=INTERACT_POST_DIALOGUE_SETTLE_WAIT)
            if self._agent_state() == AgentState.FREE_ROAM and self._has_interaction_overlay():
                self._advance_interaction_overlay(records)
            if self._agent_state() == AgentState.IN_MENU:
                self._settle_interaction_menu_state(records)
        elif state == AgentState.IN_MENU:
            self._settle_interaction_menu_state(records)
        return self._agent_state()

    def _clear_interaction_dialogue(self, records: List[EnvironmentStepRecord]) -> None:
        for _ in range(MAX_INTERACT_DIALOGUE_STEPS):
            if self._agent_state() != AgentState.IN_DIALOGUE or self._last_terminated or self._last_truncated:
                break
            pass_record = self._take_action(PassDialogueAction)
            records.append(pass_record)
            if self._last_terminated or self._last_truncated:
                break
            state = self._settle_after_interaction_dialogue_step(records)
            if self._last_terminated or self._last_truncated or state != AgentState.IN_DIALOGUE:
                break
            if pass_record.action_success != -1:
                continue
            fanfare_progressed = False
            for _ in range(DIALOGUE_FANFARE_RETRY_LIMIT):
                before_btn = self._current_frame()
                btn_record = self._take_action_string("button(a)")
                if btn_record is None:
                    break
                records.append(btn_record)
                if self._last_terminated or self._last_truncated:
                    fanfare_progressed = True
                    break
                state = self._settle_after_interaction_dialogue_step(records)
                if self._last_terminated or self._last_truncated or state != AgentState.IN_DIALOGUE:
                    fanfare_progressed = True
                    break
                if btn_record.frame_changed or is_cutscene_frame_changing(before_btn, self._current_frame()):
                    for _ in range(DIALOGUE_BUTTON_SETTLE_WAIT):
                        if self._agent_state() != AgentState.IN_DIALOGUE or self._last_terminated or self._last_truncated:
                            break
                        wait_record = self._take_action(WaitAction)
                        records.append(wait_record)
                    if self._last_terminated or self._last_truncated:
                        fanfare_progressed = True
                        break
                    self._settle_after_interaction_dialogue_step(records)
                    fanfare_progressed = True
                    break
                retry = self._take_action(PassDialogueAction)
                records.append(retry)
                if self._last_terminated or self._last_truncated:
                    fanfare_progressed = True
                    break
                state = self._settle_after_interaction_dialogue_step(records)
                if self._last_terminated or self._last_truncated or state != AgentState.IN_DIALOGUE:
                    fanfare_progressed = True
                    break
                if retry.action_success != -1 or retry.frame_changed:
                    fanfare_progressed = True
                    break
            if not fanfare_progressed:
                break

    def _current_map_place(self) -> Optional[Tuple[int, Tuple[int, int]]]:
        gmap = getattr(self._env, "_map", None)
        if gmap is None or not getattr(gmap, "current", None) or getattr(gmap, "pos", None) is None:
            return None
        try:
            return gmap.resolve(gmap.segment, gmap.pos)
        except Exception:  # noqa: BLE001
            return None

    def _pre_turn_align(self, direction: str, records: List[EnvironmentStepRecord]) -> Optional[int]:
        if getattr(self._env, "facing", None) == direction:
            return 0
        arrow = GridMoveExecutor.ARROW_ACTIONS.get(direction)
        if arrow is None:
            return 0
        before_place = self._current_map_place()
        rec = self._take_action(LowLevelAction, low_level_action=arrow)
        records.append(rec)
        if self._last_terminated or self._last_truncated:
            return 0
        after_place = self._current_map_place()
        taken = 0
        dx, dy = DIRECTIONS[direction]
        if before_place is not None and after_place is not None:
            if before_place[0] != after_place[0]:
                taken = 1
            else:
                step_dx = after_place[1][0] - before_place[1][0]
                step_dy = after_place[1][1] - before_place[1][1]
                if step_dx * dx + step_dy * dy > 0:
                    taken = 1
        if taken == 0 and rec.frame_before is not None and rec.frame_after is not None:
            from execution.strategist.pokemon.global_map.registration import _sprite_turned, register

            if not _sprite_turned(rec.frame_before, rec.frame_after):
                placed = register(rec.frame_before, rec.frame_after, direction, 1)
                if placed is not None and (placed[0][0] * dx + placed[0][1] * dy) > 0:
                    taken = 1
        return taken

    def _align_to_target(
        self,
        target_cell: Tuple[int, int],
        records: List[EnvironmentStepRecord],
        *,
        allowed_facings: Optional[set] = None,
        preferred_stand_cell: Optional[Tuple[int, int]] = None,
    ) -> Optional[str]:
        people = None
        if callable(self.people_fn):
            try:
                people = self.people_fn()
            except Exception:  # noqa: BLE001
                people = None
        gmap = getattr(self._env, "_map", None)
        kinds = self.cell_kinds
        if kinds is None and self.identified is not None:
            kinds = cell_kinds(
                self.identified,
                getattr(self._env, "ledge_patterns", None),
                getattr(self._env, "walk_patterns", None),
                people,
                getattr(self._env, "left_ledge_patterns", None),
                getattr(self._env, "blocked_cells", None),
                global_map=gmap,
            )
        if not kinds:
            return None
        if is_bottom_edge_overhang_only_cell(target_cell, identified=self.identified, global_map=gmap):
            target_cell = (target_cell[0], target_cell[1] - 1)
        all_goals = approach_cells(kinds, target_cell, identified=self.identified, people=people)
        goals = approach_cells(
            kinds,
            target_cell,
            identified=self.identified,
            people=people,
            allowed_facings=allowed_facings,
            preferred_stand_cell=preferred_stand_cell,
        )
        if not goals and (allowed_facings or preferred_stand_cell is not None):
            empirical_blocked = set(getattr(self._env, "blocked_cells", None) or ())
            candidate_Kinds = dict(kinds)
            probed_goals: Dict[Tuple[int, int], str] = {}
            for d_name, (dx, dy) in DIRECTIONS.items():
                cand = (target_cell[0] - dx, target_cell[1] - dy)
                if not in_bounds(cand) or cand in empirical_blocked:
                    continue
                if preferred_stand_cell is not None and cand != preferred_stand_cell:
                    continue
                if allowed_facings and d_name not in allowed_facings:
                    continue
                if people and cand in people and not str(people[cand] or "").startswith("sign:"):
                    continue
                if self.identified is not None and any(
                    (t := self.identified.get((cand[0], cand[1], q))) is not None
                    and getattr(t, "tag", None) in ("npc", "water", "ledge", "right_ledge", "left_ledge")
                    for q in ("top_left", "top_right", "bottom_left", "bottom_right")
                ):
                    continue
                candidate_Kinds[cand] = WALKABLE
                probed_goals[cand] = d_name
            if probed_goals:
                kinds = candidate_Kinds
                goals = probed_goals
        if not goals:
            return None
        base_avoid = {c for c in parse_avoided_cells(self._task) if in_bounds(c)} | warp_cells(
            self.identified, gmap
        )
        non_goal_adjacent = {c for c in all_goals if c not in goals and c != (0, 0)}
        avoid = base_avoid | (non_goal_adjacent if (allowed_facings or preferred_stand_cell is not None) else set())
        path = shortest_path(kinds, (0, 0), goals, avoid_cells=avoid)
        if (path is None or len(path) > self.MAX_ALIGN_STEPS) and non_goal_adjacent:
            path = shortest_path(kinds, (0, 0), goals, avoid_cells=base_avoid)
        if path is None or len(path) > self.MAX_ALIGN_STEPS:
            return None
        if not path:
            return goals.get((0, 0))
        runs = compress_path(path)
        end_cell = path[-1][2]
        facing = goals[end_cell]
        self._begin_scripted(f"walk to {end_cell} to align with {self._task} at {target_cell}")
        for move_dir, n_steps, _ in runs:
            pre_taken = self._pre_turn_align(move_dir, records)
            if self._last_terminated or self._last_truncated:
                return facing
            if self._agent_state() != AgentState.FREE_ROAM or self._has_interaction_overlay() or has_emote_bubble(self._current_frame()):
                if self._agent_state() == AgentState.FREE_ROAM:
                    self._wait_for_interaction(records)
                return facing
            remaining = max(0, n_steps - (pre_taken or 0))
            attempts = 0
            while remaining > 0:
                if attempts >= n_steps + 2:
                    return None
                attempts += 1
                step_rec = self._take_action(MoveStepsAction, direction=move_dir, steps=remaining)
                records.append(step_rec)
                if self._last_terminated or self._last_truncated:
                    return facing
                if self._agent_state() != AgentState.FREE_ROAM:
                    return facing
                action_return = (step_rec.transition_states[-1]["core"].get("action_return") or {}) if step_rec.transition_states else {}
                taken = action_return.get("n_steps_taken", 0)
                remaining -= taken
                if (
                    step_rec.action_success == 2
                    or self._has_interaction_overlay()
                    or has_emote_bubble(self._current_frame())
                ):
                    self._wait_for_interaction(records)
                    if self._agent_state() != AgentState.FREE_ROAM or self._has_interaction_overlay():
                        return facing
                if remaining > 0 and taken == 0 and not action_return.get("rotated"):
                    self._wait_for_interaction(records, settle_wait=MOVE_SETTLE_WAIT)
                    if self._agent_state() != AgentState.FREE_ROAM or self._has_interaction_overlay():
                        return facing
                    return None
        if self._agent_state() == AgentState.FREE_ROAM and (
            self._has_interaction_overlay() or has_emote_bubble(self._current_frame())
        ):
            self._wait_for_interaction(records)
        return facing

    def _refresh_live_perception(self) -> Tuple[dict, Optional[Dict[Tuple[int, int], str]], Optional[Dict[Tuple[int, int], str]]]:
        info = self._get_state()
        frame = info["core"]["current_frame"]
        people = None
        if callable(self.people_fn):
            try:
                people = self.people_fn()
            except Exception:  # noqa: BLE001
                people = None
        tiles = getattr(self._env, "_tiles", None)
        gmap = getattr(self._env, "_map", None)
        if tiles is not None and hasattr(tiles, "identify_tiles"):
            if hasattr(tiles, "record_tiles") and getattr(tiles, "vlm_call", None) is not None:
                ep = getattr(self._env, "_episode", 0)
                try:
                    tiles.record_tiles(frame, ep)
                    if gmap is not None and hasattr(gmap, "refresh"):
                        gmap.refresh(tiles._by_hash)
                except Exception:  # noqa: BLE001
                    pass
            try:
                self.identified = tiles.identify_tiles(frame, global_map=gmap)
            except Exception:  # noqa: BLE001
                pass
        if self.identified is not None:
            self.cell_kinds = cell_kinds(
                self.identified,
                getattr(self._env, "ledge_patterns", None),
                getattr(self._env, "walk_patterns", None),
                people,
                getattr(self._env, "left_ledge_patterns", None),
                getattr(self._env, "blocked_cells", None),
                global_map=gmap,
            )
        return info, self.cell_kinds, people

    def _resolve_post_walk_direction(
        self,
        shifted_target: Optional[Tuple[int, int]],
        kinds: Optional[Dict[Tuple[int, int], str]],
        people: Optional[Dict[Tuple[int, int], str]],
        allowed_facings: Optional[set] = None,
    ) -> Optional[str]:
        if shifted_target is None or shifted_target == (0, 0):
            return None
        if shifted_target in self.CARDINAL_DIRECTIONS:
            cand_dir = self.CARDINAL_DIRECTIONS[shifted_target]
            if not allowed_facings or cand_dir in allowed_facings:
                return cand_dir
            return None
        if kinds is not None:
            goals = approach_cells(
                kinds,
                shifted_target,
                identified=self.identified,
                people=people,
                allowed_facings=allowed_facings,
            )
            if (0, 0) in goals:
                return goals[(0, 0)]
        return None

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        info = self._get_state()
        if info["pokemon_core"]["agent_state"] != AgentState.FREE_ROAM:
            return self._fail("not_free_roam")

        records = []
        people = None
        if callable(self.people_fn):
            try:
                people = self.people_fn()
            except Exception:  # noqa: BLE001
                people = None
        gmap = getattr(self._env, "_map", None)
        kinds = self.cell_kinds
        if kinds is None and self.identified is not None:
            kinds = cell_kinds(
                self.identified,
                getattr(self._env, "ledge_patterns", None),
                getattr(self._env, "walk_patterns", None),
                people,
                getattr(self._env, "left_ledge_patterns", None),
                getattr(self._env, "blocked_cells", None),
                global_map=gmap,
            )
        target_cell, allowed_facings, preferred_stand_cell = resolve_interaction_target_and_constraints(
            self._task,
            kinds=kinds,
            identified=self.identified,
            people=people,
            global_map=gmap,
        )
        direction = None
        need_walk_align = target_cell is not None and (
            target_cell not in CheckInteractionAction.cardinals.values()
            or (preferred_stand_cell is not None and preferred_stand_cell != (0, 0))
            or (
                bool(allowed_facings)
                and self.CARDINAL_DIRECTIONS.get(target_cell) not in allowed_facings
            )
        )
        if need_walk_align and target_cell is not None:
            before_place = self._current_map_place()
            direction = self._align_to_target(
                target_cell,
                records,
                allowed_facings=allowed_facings,
                preferred_stand_cell=preferred_stand_cell,
            )
            if self._last_terminated or self._last_truncated:
                return self._finish(records)
            if self._agent_state() != AgentState.FREE_ROAM or self._has_interaction_overlay():
                if self._agent_state() == AgentState.FREE_ROAM and self._has_interaction_overlay():
                    self._advance_interaction_overlay(records)
                if self._agent_state() == AgentState.IN_MENU:
                    self._settle_interaction_menu_state(records)
                if self._agent_state() == AgentState.IN_DIALOGUE:
                    self._clear_interaction_dialogue(records)
                return self._finish(records)
            if records:
                after_place = self._current_map_place()
                if (
                    before_place is not None
                    and after_place is not None
                    and before_place[0] == after_place[0]
                ):
                    dx = after_place[1][0] - before_place[1][0]
                    dy = after_place[1][1] - before_place[1][1]
                    target_cell = (target_cell[0] - dx, target_cell[1] - dy)
                info, kinds, people = self._refresh_live_perception()
                if direction is None:
                    direction = self._resolve_post_walk_direction(
                        target_cell, kinds, people, allowed_facings=allowed_facings
                    )

        if direction is None:
            if records:
                info, kinds, people = self._refresh_live_perception()
            check_target = self._task
            if records and target_cell is not None:
                check_target = re.sub(
                    r"\(\s*-?\d+\s*,\s*-?\d+\s*\)",
                    f"({target_cell[0]}, {target_cell[1]})",
                    self._task,
                )
            check = CheckInteractionAction(self._vlm_call, self._parameters)
            result, code = check.execute(
                info,
                target=check_target,
                identified=self.identified,
                cell_kinds=self.cell_kinds,
            )
            self.check_result = result
            if code is None:
                return self._fail("invalid_check", result["invalid"])
            if code == -1:
                return self._fail("target_not_found")
            if result.get("cell") is not None:
                target_cell = result.get("cell")
            if code == 1 or (
                code == 0
                and result.get("direction") is not None
                and allowed_facings
                and result.get("direction") not in allowed_facings
            ):
                cell = result.get("cell")
                if cell is not None:
                    before_place = self._current_map_place()
                    rec_before = len(records)
                    direction = self._align_to_target(
                        cell,
                        records,
                        allowed_facings=allowed_facings,
                        preferred_stand_cell=preferred_stand_cell,
                    )
                    if self._last_terminated or self._last_truncated:
                        return self._finish(records)
                    if self._agent_state() != AgentState.FREE_ROAM or self._has_interaction_overlay():
                        if self._agent_state() == AgentState.FREE_ROAM and self._has_interaction_overlay():
                            self._advance_interaction_overlay(records)
                        if self._agent_state() == AgentState.IN_MENU:
                            self._settle_interaction_menu_state(records)
                        if self._agent_state() == AgentState.IN_DIALOGUE:
                            self._clear_interaction_dialogue(records)
                        return self._finish(records)
                    if len(records) > rec_before:
                        after_place = self._current_map_place()
                        if (
                            before_place is not None
                            and after_place is not None
                            and before_place[0] == after_place[0]
                        ):
                            dx = after_place[1][0] - before_place[1][0]
                            dy = after_place[1][1] - before_place[1][1]
                            target_cell = (cell[0] - dx, cell[1] - dy)
                        info, kinds, people = self._refresh_live_perception()
                        if direction is None:
                            direction = self._resolve_post_walk_direction(
                                target_cell, kinds, people, allowed_facings=allowed_facings
                            )
                if direction is None:
                    return self._fail("target_only_diagonal",
                                      f"Target found at {result['found_cells_str']}, which cannot be faced")
            else:
                direction = result["direction"]
                if records and result.get("cell") is not None:
                    target_cell = result.get("cell")

        aligned_walked = bool(records)
        if aligned_walked:
            info, kinds, people = self._refresh_live_perception()
        self._begin_scripted(f"face {direction} and interact with {self._task}")
        if getattr(self._env, "facing", None) != direction:
            adj_cell = DIRECTIONS.get(direction)
            if (
                aligned_walked
                and adj_cell is not None
                and kinds is not None
                and kinds.get(adj_cell) == WALKABLE
                and target_cell != adj_cell
                and target_cell not in CheckInteractionAction.counter_cardinals.values()
            ):
                pre_taken = self._pre_turn_align(direction, records)
                if self._last_terminated or self._last_truncated:
                    return self._finish(records)
                if self._agent_state() != AgentState.FREE_ROAM:
                    if self._agent_state() == AgentState.IN_DIALOGUE:
                        self._clear_interaction_dialogue(records)
                    return self._finish(records)
                if pre_taken:
                    found_cells_str = (
                        (self.check_result or {}).get("found_cells_str")
                        or coords_to_string([target_cell] if target_cell is not None else [])
                    )
                    return self._fail("target_cell_walkable",
                                      f"Pressed {direction} to face the target at {found_cells_str}, "
                                      "but the player walked onto that cell instead, so nothing is there")
            else:
                turn = self._take_action(MoveStepsAction, direction=direction, steps=1)
                records.append(turn)
                if self._last_terminated or self._last_truncated:
                    return self._finish(records)
                if self._agent_state() != AgentState.FREE_ROAM:
                    if self._agent_state() == AgentState.IN_DIALOGUE:
                        self._clear_interaction_dialogue(records)
                    return self._finish(records)
                action_return = turn.transition_states[-1]["core"].get("action_return") or {}
                steps_taken = action_return.get("n_steps_taken", 0)
                if steps_taken and not (
                    not aligned_walked
                    and steps_taken == 1
                    and target_cell in CheckInteractionAction.counter_cardinals.values()
                ):
                    found_cells_str = (
                        (self.check_result or {}).get("found_cells_str")
                        or coords_to_string([target_cell] if target_cell is not None else [])
                    )
                    return self._fail("target_cell_walkable",
                                      f"Pressed {direction} to face the target at {found_cells_str}, "
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
        if state == AgentState.IN_MENU and not (self._last_terminated or self._last_truncated):
            state = self._settle_interaction_menu_state(records)
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
        if is_battle_defeat_text([self.dialogue_text]):
            recorder = getattr(self._env, "record_defeat", None)
            if callable(recorder):
                recorder()
        self.report.notes = self.dialogue_text
        raw_state = self._get_state()["pokemon_core"]["agent_state"]
        frame = self._current_frame()
        final_state = effective_agent_state(raw_state, frame)
        if final_state == AgentState.FREE_ROAM and is_battle_screen(frame):
            final_state = AgentState.IN_BATTLE
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
MOVE_MAX_CUTSCENE_WAIT = 30
MOVE_POST_MOTION_WAIT = 5
MOVE_POST_WARP_BANNER_WAIT = 6

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
        self._blocked_at_place: Dict[Any, List[str]] = {}
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
        place = self._current_map_place()
        if place is not None and direction in DIRECTIONS:
            blocked_list = self._blocked_at_place.setdefault(place, [])
            if direction not in blocked_list:
                blocked_list.append(direction)
        recorder = getattr(self._env, "record_blocked_step", None)
        if callable(recorder):
            try:
                recorder(direction=direction)
            except Exception:  # noqa: BLE001
                pass

    def _local_blocked_cells_from_history(self) -> set:
        local_blocked = set(getattr(self._env, "blocked_cells", None) or ())
        place = self._current_map_place()
        gmap = getattr(self._env, "_map", None)
        if place is None or gmap is None or not self._blocked_at_place:
            return local_blocked
        cur_seg, (px, py) = place
        for b_place, dirs in self._blocked_at_place.items():
            try:
                r_seg, r_pos = gmap.resolve(b_place[0], b_place[1])
            except Exception:  # noqa: BLE001
                continue
            if r_seg != cur_seg:
                continue
            for d in dirs:
                if d not in DIRECTIONS:
                    continue
                dx, dy = DIRECTIONS[d]
                rel = (r_pos[0] + dx - px, r_pos[1] + dy - py)
                if rel != (0, 0) and in_bounds(rel):
                    local_blocked.add(rel)
        return local_blocked

    def _refresh_local_perception(self, blocked_neighbor: Optional[Tuple[int, int]] = None) -> bool:
        tiles = getattr(self._env, "_tiles", None)
        if tiles is None or not hasattr(tiles, "identify_tiles"):
            if self.cell_kinds is not None:
                self.cell_kinds = dict(self.cell_kinds)
                for rel in self._local_blocked_cells_from_history():
                    self.cell_kinds[rel] = BLOCKED
                if blocked_neighbor is not None:
                    self.cell_kinds[blocked_neighbor] = BLOCKED
            return False
        frame = self._get_state()["core"]["current_frame"]
        gmap = getattr(self._env, "_map", None)
        self.identified = tiles.identify_tiles(frame, global_map=gmap)
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
            self._local_blocked_cells_from_history(),
            global_map=gmap,
        )
        if blocked_neighbor is not None:
            kinds[blocked_neighbor] = BLOCKED
        self.cell_kinds = kinds
        return True

    def _step_wait_with_crops(self) -> Tuple[np.ndarray, np.ndarray]:
        wait_record = self._take_action(WaitAction)
        drop_prefix_crops(captured_dialogue_crops([wait_record]), self.dialogue_crops)
        return wait_record.frame_before, wait_record.frame_after

    def _settle_free_roam(self):
        state, _, saw_activity = settle_overworld_free_roam(
            env=self._env,
            get_state_fn=lambda: self._get_state()["pokemon_core"]["agent_state"],
            get_frame_fn=lambda: self._get_state()["core"]["current_frame"],
            is_done_fn=lambda: bool(self._last_terminated or self._last_truncated),
            step_wait_fn=self._step_wait_with_crops,
            settle_wait=MOVE_SETTLE_WAIT,
            max_cutscene_wait=MOVE_MAX_CUTSCENE_WAIT,
            post_motion_wait=MOVE_POST_MOTION_WAIT,
            post_warp_banner_wait=MOVE_POST_WARP_BANNER_WAIT,
            include_top_box=False,
            record_dialogue_settled=False,
            respect_last_settled=True,
            settle_menu_fn=self._settle_menu_state,
        )
        self._saw_cutscene_activity = saw_activity
        return state

    def _settle_menu_state(self):
        return settle_overworld_menu_state(
            get_state_fn=lambda: self._get_state()["pokemon_core"]["agent_state"],
            get_frame_fn=lambda: self._get_state()["core"]["current_frame"],
            is_done_fn=lambda: bool(self._last_terminated or self._last_truncated),
            step_wait_fn=self._step_wait_with_crops,
            min_wait=MENU_SETTLE_WAIT,
            settle_wait=MOVE_SETTLE_WAIT,
            max_cutscene_wait=MOVE_MAX_CUTSCENE_WAIT,
            post_motion_wait=MOVE_POST_MOTION_WAIT,
        )

    def _check_post_settle_interruption(self, state: AgentState, context: str) -> Optional[int]:
        frame = self._get_state()["core"]["current_frame"]
        if (state != AgentState.FREE_ROAM
                or has_bottom_dialogue_box(frame)
                or is_battle_screen(frame)
                or (has_top_dialogue_box(frame) and not has_menu_cursor(frame))):
            eff_state = effective_agent_state(state, frame)
            if eff_state == AgentState.FREE_ROAM and is_battle_screen(frame):
                eff_state = AgentState.IN_BATTLE
            return self._fail(
                MOVE_INTERRUPTED,
                "move_interrupted",
                f"Left free roam {context}: {eff_state.name}",
            )
        return None

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
        dx, dy = DIRECTIONS[direction]
        if before_place is not None and after_place is not None:
            if before_place[0] != after_place[0]:
                taken = 1
            else:
                step_dx = after_place[1][0] - before_place[1][0]
                step_dy = after_place[1][1] - before_place[1][1]
                along = step_dx * dx + step_dy * dy
                if along > 0:
                    taken = 1
                    self._displacement = (self._displacement[0] + dx, self._displacement[1] + dy)
        if taken == 0 and record.frame_before is not None and record.frame_after is not None:
            from execution.strategist.pokemon.global_map.registration import register

            placed = register(record.frame_before, record.frame_after, direction, 1)
            if placed is not None and (placed[0][0] * dx + placed[0][1] * dy) > 0:
                taken = 1
                self._displacement = (self._displacement[0] + dx, self._displacement[1] + dy)
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
            after_place = self._current_map_place()
            warped_area = (
                before_place is not None
                and after_place is not None
                and before_place[0] != after_place[0]
            )
            if taken == 0 and before_place is not None and after_place is not None:
                if warped_area:
                    taken = 1
                else:
                    along = (
                        (after_place[1][0] - before_place[1][0]) * dx
                        + (after_place[1][1] - before_place[1][1]) * dy
                    )
                    if along > 0:
                        taken = along
            self._steps_taken += taken
            self._last_step_taken += taken
            mult = 2 if (move_kind == "jump" and taken > 0) else 1
            if (
                before_place is not None
                and after_place is not None
                and before_place[0] == after_place[0]
                and (after_place[1] != before_place[1] or taken == 0)
            ):
                step_dx = after_place[1][0] - before_place[1][0]
                step_dy = after_place[1][1] - before_place[1][1]
            else:
                step_dx, step_dy = dx * taken * mult, dy * taken * mult
            self._displacement = (self._displacement[0] + step_dx, self._displacement[1] + step_dy)
            state = self._get_state()["pokemon_core"]["agent_state"]
            if record.action_success == 2 or state != AgentState.FREE_ROAM:
                return self._fail(MOVE_INTERRUPTED, "move_interrupted",
                                  f"Left free roam after {taken} steps {direction}: {state.name}")
            if warped_area:
                return None
            remaining = max(0, remaining - taken)
            if taken > 0 and remaining > 0 and self._n_moves < self._max_steps and attempts < steps + 2:
                continue
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
                settled_place = self._current_map_place()
                if (
                    (before_place is not None and settled_place is not None and settled_place[0] != before_place[0])
                    or bool(getattr(getattr(self._env, "_map", None), "faded", False))
                    or int(getattr(getattr(self._env, "_map", None), "lost", 0) or 0) > 0
                ):
                    return None
                if ((step_moved_screen or self._saw_cutscene_activity)
                        and not retried_after_cutscene
                        and self._n_moves < self._max_steps):
                    retried_after_cutscene = True
                    continue
                if (
                    taken > 0
                    and before_place is not None
                    and settled_place == before_place
                ):
                    gmap = getattr(self._env, "_map", None)
                    if gmap is not None and getattr(gmap, "pos", None) is not None:
                        expected_pos = (
                            before_place[1][0] + dx * taken * mult,
                            before_place[1][1] + dy * taken * mult,
                        )
                        gmap.pos = expected_pos
                        if gmap.segment < len(gmap.visits):
                            gmap.visits[gmap.segment][expected_pos] += 1
                        self._displacement = (
                            self._displacement[0] - step_dx + dx * taken * mult,
                            self._displacement[1] - step_dy + dy * taken * mult,
                        )
                self._record_empirical_block(direction)
                if taken == 0 and not action_return.get("rotated"):
                    return self._fail(MOVE_BLOCKED, "move_blocked",
                                      f"Blocked moving {direction} with {remaining} steps left")
                return self._fail(MOVE_BLOCKED, "move_blocked",
                                  f"Stopped moving {direction} with {remaining} steps left")
        return None

    def _people_now(self) -> Optional[Dict[Tuple[int, int], str]]:
        if not callable(self.people_fn):
            return None
        try:
            return self.people_fn()
        except Exception:  # noqa: BLE001
            return None

    def _walk(self, goals: Dict[Tuple[int, int], Optional[str]], prefer_lateral: bool = False) -> Optional[int]:
        current_goals = dict(goals)
        current_avoid = {c for c in parse_avoided_cells(self._task) if in_bounds(c)} | warp_cells(
            self.identified, getattr(self._env, "_map", None)
        )
        reroutes = 0
        while True:
            people = self._people_now()
            path = shortest_path(
                self.cell_kinds,
                (0, 0),
                current_goals,
                prefer_lateral=prefer_lateral,
                avoid_cells=current_avoid,
                people=people,
            )
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
                        shifted_avoid = {
                            (ax - dx, ay - dy)
                            for (ax, ay) in current_avoid
                            if in_bounds((ax - dx, ay - dy))
                        }
                        if shifted_goals:
                            blocked_neighbor = DIRECTIONS.get(failed_dir)
                            self._refresh_local_perception(blocked_neighbor=blocked_neighbor)
                            if shortest_path(
                                self.cell_kinds,
                                (0, 0),
                                shifted_goals,
                                prefer_lateral=prefer_lateral,
                                avoid_cells=shifted_avoid,
                                people=self._people_now(),
                            ) is not None:
                                current_goals = shifted_goals
                                current_avoid = shifted_avoid
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

    def _is_usable_target_cell(self, cell: Optional[Tuple[int, int]], people=None) -> bool:
        if cell is None:
            return False
        if in_bounds(cell):
            return True
        if self.cell_kinds is None:
            return False
        return bool(approach_cells(self.cell_kinds, cell, identified=self.identified, people=people))

    def _resolve_explicit_target(self) -> Optional[Tuple[int, int]]:
        people = self._people_now()
        coords = [c for c in parse_all_cells(self._task) if self._is_usable_target_cell(c, people=people)]
        if len(coords) != 1:
            return None
        target = coords[0]
        if target == (0, 0):
            return target
        if self.cell_kinds.get(target) == WALKABLE:
            if shortest_path(self.cell_kinds, (0, 0), [target], people=people) is not None:
                return target
        else:
            goals = approach_cells(self.cell_kinds, target, identified=self.identified, people=people)
            if goals and shortest_path(self.cell_kinds, (0, 0), goals, people=people) is not None:
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
            if not self._is_usable_target_cell(target, people=self._people_now()):
                return self._fail(MOVE_BAD_LOCATION, "bad_location", f"Unusable location {location!r}")
        self.target_cell = target
        if target == (0, 0):
            return MOVE_OK
        gmap = getattr(self._env, "_map", None)
        side_door_dir = None
        if gmap is not None and getattr(gmap, "pos", None) is not None and hasattr(gmap, "side_gatehouse_step_direction"):
            try:
                si, (px, py) = gmap.resolve(gmap.segment, gmap.pos)
                side_door_dir = gmap.side_gatehouse_step_direction((px + target[0], py + target[1]), si)
            except Exception:  # noqa: BLE001
                side_door_dir = None
        exit_dir = None
        if self.cell_kinds.get(target) == WALKABLE:
            exit_dir = boundary_exit_direction(
                self.cell_kinds, self.identified, target, global_map=gmap
            )
            if exit_dir is None and side_door_dir is not None:
                exit_dir = side_door_dir
            goals = {target: None}
        else:
            goals = approach_cells(self.cell_kinds, target, identified=self.identified, people=self._people_now())
            if not goals:
                return self._fail(MOVE_NO_PATH, "no_path", f"Nothing walkable next to the target at {target}")
            if side_door_dir is not None:
                exit_dir = side_door_dir
        code = self._walk(goals)
        if code is not None:
            return code
        if exit_dir is not None and self._get_state()["pokemon_core"]["agent_state"] == AgentState.FREE_ROAM:
            exit_code = self._step(exit_dir, 1)
            if exit_code is not None and exit_code != MOVE_BLOCKED:
                return exit_code
        state = self._settle_free_roam()
        if self._last_terminated:
            return self._fail(MOVE_TERMINATED, "terminated")
        if self._last_truncated:
            return self._fail(MOVE_TRUNCATED, "truncated")
        interrupted = self._check_post_settle_interruption(state, f"at target {target}")
        if interrupted is not None:
            return interrupted
        self.report.termination_reason = "arrived"
        self.report.notes = f"Moved to the target at {target}"
        return MOVE_OK


class OffScreenMoveExecutor(GridMoveExecutor):
    LOCATE_TAG = "locate_edge"
    MAX_DETERMINISTIC_CORRIDOR_WIDTH = 3
    MAX_EDGE_ALIGNMENT_SHIFT = 2

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

    @staticmethod
    def _can_step_from_cell_in_kinds(
        kinds: Dict[Tuple[int, int], str],
        cell: Tuple[int, int],
        direction: str,
        avoid_warps: Optional[set] = None,
    ) -> Optional[bool]:
        dx, dy = DIRECTIONS[direction]
        nxt = (cell[0] + dx, cell[1] + dy)
        if avoid_warps and nxt in avoid_warps:
            return False
        kind = kinds.get(nxt)
        if kind == WALKABLE:
            return True
        if (kind == LEDGE and direction == "down") or (kind == LEDGE_LEFT and direction == "left"):
            landing = (cell[0] + 2 * dx, cell[1] + 2 * dy)
            if avoid_warps and landing in avoid_warps:
                return False
            return kinds.get(landing) == WALKABLE
        if kind in (BLOCKED, LEDGE, LEDGE_LEFT):
            return False
        return None

    def _live_perception(self) -> Tuple[Optional[Dict[Tuple[int, int], str]], Optional[Dict[Any, Any]]]:
        tiles = getattr(self._env, "_tiles", None)
        if tiles is None or not hasattr(tiles, "identify_tiles"):
            return None, None
        frame = self._get_state()["core"]["current_frame"]
        gmap = getattr(self._env, "_map", None)
        if hasattr(tiles, "record_tiles") and getattr(tiles, "vlm_call", None) is not None:
            ep = getattr(self._env, "_episode", 0)
            try:
                tiles.record_tiles(frame, ep)
                if gmap is not None and hasattr(gmap, "refresh"):
                    gmap.refresh(tiles._by_hash)
            except Exception:  # noqa: BLE001
                pass
        identified = tiles.identify_tiles(frame, global_map=gmap)
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
            global_map=gmap,
        )
        return kinds, identified

    def _live_cell_kinds(self) -> Optional[Dict[Tuple[int, int], str]]:
        kinds, _ = self._live_perception()
        return kinds

    def _task_targets_warp(self) -> bool:
        return bool(
            re.search(
                r"\b(?:ladder|stair(?:s|case)?|door(?:way)?|entrance|exit|elevator|warp)\b",
                self._task,
                re.IGNORECASE,
            )
        )

    def _task_targets_front_entity(self) -> bool:
        return bool(
            re.search(
                r"\b(?:person|npc|nurse|clerk|trainer|guard|sign(?:post)?|counter|desk|table|pc|computer|machine|item|ball|talk|interact|speak|heal|approach)\b",
                self._task,
                re.IGNORECASE,
            )
        )

    def _can_step_from_current(self, direction: str) -> Optional[bool]:
        kinds, identified = self._live_perception()
        if kinds is None:
            return None
        gmap = getattr(self._env, "_map", None)
        avoid_warps = None if self._task_targets_warp() else warp_cells(identified=identified, global_map=gmap)
        return self._can_step_from_cell_in_kinds(kinds, (0, 0), direction, avoid_warps=avoid_warps)

    def _at_front_target(
        self,
        direction: str,
        kinds: Optional[Dict[Tuple[int, int], str]] = None,
        identified: Optional[Dict[Any, Any]] = None,
    ) -> bool:
        from execution.perception.pokemon.navigation import (
            FRONT_TARGET_TAGS,
            QUARTERS,
            _is_npc_target_cell,
            has_offscreen_front_target,
        )

        include_direct = self._task_targets_front_entity()
        gmap = getattr(self._env, "_map", None)
        if has_offscreen_front_target(gmap, (0, 0), direction, include_direct=include_direct):
            return True
        if kinds is None and identified is None:
            kinds, identified = self._live_perception()
        people = self._people_now()
        dx, dy = DIRECTIONS[direction]
        first = (dx, dy)
        second = (2 * dx, 2 * dy)
        if include_direct and _is_npc_target_cell(first, identified, people):
            return True
        if kinds is not None and kinds.get(first) == BLOCKED and _is_npc_target_cell(second, identified, people):
            return True
        if identified is not None:
            if include_direct:
                first_tags = {
                    getattr(identified.get((first[0], first[1], q)), "tag", None)
                    for q in QUARTERS
                    if (first[0], first[1], q) in identified
                }
                if first_tags & FRONT_TARGET_TAGS:
                    return True
            if kinds is not None and kinds.get(first) == BLOCKED:
                second_tags = {
                    getattr(identified.get((second[0], second[1], q)), "tag", None)
                    for q in QUARTERS
                    if (second[0], second[1], q) in identified
                }
                if second_tags & FRONT_TARGET_TAGS:
                    return True
        return False

    def _sidestep_targets_at_edge(
        self,
        target: Tuple[int, int],
        target_run: List[Tuple[int, int]],
        direction: str,
    ) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        kinds, identified = self._live_perception()
        if kinds is None:
            return []
        if self._steps_taken > 0 and self._at_front_target(direction, kinds=kinds, identified=identified):
            return []
        gmap = getattr(self._env, "_map", None)
        avoid_warps = set() if self._task_targets_warp() else warp_cells(identified=identified, global_map=gmap)
        avoided_set = {c for c in parse_avoided_cells(self._task) if in_bounds(c)}
        explicit_coords = {c for c in parse_all_cells(self._task) if in_bounds(c)}
        limit_shift = self._steps_taken > 0 and target in explicit_coords
        axis = 0 if direction in ("up", "down") else 1
        candidates = []
        for cell in target_run:
            if cell == target or cell in avoided_set:
                continue
            rel = (cell[0] - target[0], cell[1] - target[1])
            if limit_shift and abs(rel[axis]) > self.MAX_EDGE_ALIGNMENT_SHIFT:
                continue
            if not in_bounds(rel) or kinds.get(rel) != WALKABLE or rel in avoid_warps:
                continue
            if self._can_step_from_cell_in_kinds(kinds, rel, direction, avoid_warps=avoid_warps) is not True:
                continue
            if shortest_path(kinds, (0, 0), [rel], avoid_cells=avoid_warps) is not None:
                candidates.append((abs(rel[axis]), abs(cell[axis]), rel, cell))
        candidates.sort()
        return [(rel, cell) for _, _, rel, cell in candidates]

    def _rebind_direction_for_coords(
        self,
        direction: str,
        coords: List[Tuple[int, int]],
    ) -> str:
        if not coords:
            return direction
        gmap = getattr(self._env, "_map", None)
        people = self._people_now()
        cells = edge_cells(self.cell_kinds, direction)
        forward_cells = filter_forward_edge_cells(cells, direction)
        open_cells = filter_open_lane_edge_cells(
            forward_cells, direction, self.cell_kinds, gmap, identified=self.identified, people=people
        )
        if any(c in open_cells for c in coords):
            return direction
        # If an explicit corner or edge coordinate has an open exit on another edge that it touches
        # (e.g. "MoveOff(right through the tall grass at (5, 4))" where (5, 4) is blocked on the right
        # edge or inward-blocked on the up edge, or where the task text also mentions another direction),
        # prefer the edge where that coordinate is an open exit or has the best offscreen continuation.
        lower_task = self._task.lower()
        mentioned_dirs = []
        for word, alias in (("up", "up"), ("north", "up"), ("down", "down"), ("south", "down"),
                            ("left", "left"), ("west", "left"), ("right", "right"), ("east", "right")):
            if re.search(rf"\b{word}\b", lower_task) and alias not in mentioned_dirs:
                mentioned_dirs.append(alias)
        candidate_dirs = list(DIRECTIONS)
        best_dir = direction
        best_score = (-1, -1, -1)
        seg = getattr(gmap, "current", None) if gmap is not None else None
        px = py = 0
        if seg is not None and getattr(gmap, "pos", None) is not None:
            try:
                _, (px, py) = gmap.resolve(gmap.segment, gmap.pos)
            except Exception:  # noqa: BLE001
                seg = None
        from execution.perception.pokemon.navigation import (
            LOOKAHEAD_STEPS,
            _gmap_blocked_sets,
            _offscreen_continuation_depth,
        )
        b_cells, b_moves = _gmap_blocked_sets(gmap) if seg is not None else (set(), {})
        for d in candidate_dirs:
            d_cells = edge_cells(self.cell_kinds, d)
            matching = [c for c in coords if c in d_cells]
            if not matching:
                continue
            d_forward = filter_forward_edge_cells(d_cells, d)
            d_open = filter_open_lane_edge_cells(
                d_forward, d, self.cell_kinds, gmap, identified=self.identified, people=people
            )
            in_open = int(any(c in d_open for c in matching))
            depth = 0
            if seg is not None:
                depth = max(
                    _offscreen_continuation_depth(
                        seg, (px + c[0], py + c[1]), d, LOOKAHEAD_STEPS, blocked_cells=b_cells, blocked_moves=b_moves
                    )
                    for c in matching
                )
            else:
                depth = LOOKAHEAD_STEPS if in_open else 0
            is_mentioned = int(d in mentioned_dirs)
            is_Vertical_corner = int(d in ("up", "down") and any(abs(c[0]) >= 4 and abs(c[1]) == 4 for c in matching))
            score = (in_open if depth > 0 else 0, depth, is_Vertical_corner + is_mentioned)
            if score > best_score:
                best_score = score
                best_dir = d
        return best_dir

    def _resolve_deterministic_edge(self) -> Optional[Tuple[str, Tuple[int, int]]]:
        direction = parse_edge_direction(self._task)
        if direction not in DIRECTIONS:
            return None
        coords = [c for c in parse_all_cells(self._task) if in_bounds(c)]
        avoided_set = {c for c in parse_avoided_cells(self._task) if in_bounds(c)}
        if coords:
            direction = self._rebind_direction_for_coords(direction, coords)
        cells = edge_cells(self.cell_kinds, direction)
        if not cells:
            return None
        if avoided_set:
            non_avoided_cells = [c for c in cells if c not in avoided_set]
            if non_avoided_cells:
                cells = non_avoided_cells
        forward_cells = filter_forward_edge_cells(cells, direction)
        gmap = getattr(self._env, "_map", None)
        people = self._people_now()
        path_avoid = avoided_set | warp_cells(self.identified, gmap)
        open_cells = filter_open_lane_edge_cells(
            forward_cells, direction, self.cell_kinds, gmap, identified=self.identified, people=people
        )
        if avoided_set:
            non_avoided_open = [c for c in open_cells if c not in avoided_set]
            if non_avoided_open:
                open_cells = non_avoided_open

        def _is_ledge_trap_cell(c: Tuple[int, int]) -> bool:
            if is_ledge_roofed_edge_cell(c, direction, self.cell_kinds):
                return True
            p = shortest_path(self.cell_kinds, (0, 0), [c], avoid_cells=path_avoid, people=people)
            return bool(p and any(m == "jump" for _, m, _ in p))

        unroofed_open_exists = any(not _is_ledge_trap_cell(oc) for oc in open_cells)
        task_allows_ledge = bool(re.search(r"\b(?:ledge|jump)\b", self._task, re.IGNORECASE))

        from execution.perception.pokemon.navigation import (
            has_offscreen_continuation,
            has_offscreen_front_target,
        )

        include_direct_front = self._task_targets_front_entity()
        on_edge_coords = [c for c in coords if c in cells]
        if on_edge_coords:
            preferred = [c for c in on_edge_coords if c in open_cells]
            front_targets = [
                c for c in on_edge_coords
                if has_offscreen_front_target(gmap, c, direction, include_direct=include_direct_front)
            ]
            if not preferred and front_targets:
                preferred = front_targets
            if not preferred and open_cells:
                axis = 0 if direction in ("up", "down") else 1
                all_runs = edge_runs(cells, direction)
                same_run_open = []
                for c in on_edge_coords:
                    c_run = next((r for r in all_runs if c in r), [])
                    same_run_open.extend(
                        oc
                        for oc in c_run
                        if oc in open_cells
                        and abs(oc[axis] - c[axis]) <= self.MAX_EDGE_ALIGNMENT_SHIFT
                        and oc not in same_run_open
                    )
                if same_run_open:
                    ref = on_edge_coords[0]
                    nearest = min(same_run_open, key=lambda c: (abs(c[axis] - ref[axis]), abs(c[axis])))
                    preferred = [nearest]
            if not preferred:
                candidate_explicit = [
                    c for c in ([c for c in on_edge_coords if c in forward_cells] or on_edge_coords)
                    if has_offscreen_continuation(gmap, c, direction)
                    or has_offscreen_front_target(gmap, c, direction)
                ]
                if not candidate_explicit and open_cells:
                    preferred = open_cells
                elif unroofed_open_exists and not task_allows_ledge:
                    safe_explicit = [c for c in (candidate_explicit or on_edge_coords) if not _is_ledge_trap_cell(c)]
                    preferred = safe_explicit or open_cells
                else:
                    preferred = candidate_explicit or open_cells or on_edge_coords
            path = shortest_path(self.cell_kinds, (0, 0), preferred, avoid_cells=path_avoid, people=people)
            if path is None and open_cells:
                path = shortest_path(self.cell_kinds, (0, 0), open_cells, avoid_cells=path_avoid, people=people)
            target = path[-1][2] if path else preferred[0]
            return direction, target

        if len(coords) == 1:
            waypoint = coords[0]
            if self.cell_kinds.get(waypoint) in (WALKABLE, LEDGE, LEDGE_LEFT):
                axis = 0 if direction in ("up", "down") else 1
                aligned = (
                    [c for c in open_cells if c[axis] == waypoint[axis]]
                    or [c for c in forward_cells if c[axis] == waypoint[axis]]
                )
                if len(aligned) == 1:
                    return direction, aligned[0]

        unreached_edge_coords = [c for c in coords if is_on_edge(c, direction)]
        if unreached_edge_coords and (open_cells or forward_cells or cells):
            candidates = open_cells or forward_cells or cells
            path = shortest_path(self.cell_kinds, (0, 0), candidates, avoid_cells=path_avoid, people=people)
            if path:
                return direction, path[-1][2]

        stripped = self._task.strip().lower()
        is_bare_direction = stripped in ("up", "down", "left", "right", "north", "south", "west", "east")
        runs = edge_runs(open_cells, direction)
        if len(runs) == 1:
            run = runs[0]
            if is_bare_direction or len(run) <= self.MAX_DETERMINISTIC_CORRIDOR_WIDTH:
                path = shortest_path(self.cell_kinds, (0, 0), run, avoid_cells=path_avoid, people=people)
                target = path[-1][2] if path else run[0]
                return direction, target
        elif open_cells and (is_bare_direction or not coords):
            path = shortest_path(self.cell_kinds, (0, 0), open_cells, avoid_cells=path_avoid, people=people)
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
                      .replace("[EDGE_CELLS]", verbalize_edges(
                          self.cell_kinds,
                          identified=self.identified,
                          people=self._people_now(),
                          global_map=getattr(self._env, "_map", None),
                      )))
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
            avoided_set = {c for c in parse_avoided_cells(self._task) if in_bounds(c)}
            cells = edge_cells(self.cell_kinds, direction)
            if avoided_set:
                non_avoided_cells = [c for c in cells if c not in avoided_set]
                if non_avoided_cells:
                    cells = non_avoided_cells
            forward_cells = filter_forward_edge_cells(cells, direction)
            candidate_cells = forward_cells or cells
            if target is not None and target not in candidate_cells and candidate_cells:
                target = min(candidate_cells, key=lambda c: (c[0] - target[0]) ** 2 + (c[1] - target[1]) ** 2)
            if target is None or target not in cells:
                return self._fail(MOVE_BAD_LOCATION, "bad_location",
                                  f"{location!r} is not a walkable cell on the {direction} edge")
            gmap = getattr(self._env, "_map", None)
            people = self._people_now()
            open_cells = filter_open_lane_edge_cells(
                candidate_cells, direction, self.cell_kinds, gmap, identified=self.identified, people=people
            )
            if avoided_set:
                non_avoided_open = [c for c in open_cells if c not in avoided_set]
                if non_avoided_open:
                    open_cells = non_avoided_open
            from execution.perception.pokemon.navigation import (
                has_offscreen_continuation,
                has_offscreen_front_target,
            )

            if (
                target not in open_cells
                and open_cells
                and not has_offscreen_front_target(
                    gmap, target, direction, include_direct=self._task_targets_front_entity()
                )
            ):
                axis = 0 if direction in ("up", "down") else 1
                adjacent_open = [c for c in open_cells if abs(c[axis] - target[axis]) <= self.MAX_EDGE_ALIGNMENT_SHIFT]
                if adjacent_open:
                    target = min(adjacent_open, key=lambda c: (abs(c[axis] - target[axis]), abs(c[axis])))
                elif not has_offscreen_continuation(gmap, target, direction):
                    safe_open_path = shortest_path(self.cell_kinds, (0, 0), open_cells, people=people)
                    if safe_open_path:
                        target = safe_open_path[-1][2]
                    else:
                        target = min(open_cells, key=lambda c: (abs(c[axis] - target[axis]), abs(c[axis])))
                elif not re.search(r"(?:ledge|jump)", self._task, re.IGNORECASE):
                    target_path = shortest_path(self.cell_kinds, (0, 0), [target], people=people)
                    target_is_trap = is_ledge_roofed_edge_cell(target, direction, self.cell_kinds) or bool(
                        target_path and any(m == "jump" for _, m, _ in target_path)
                    )
                    if target_is_trap:
                        safe_open_path = shortest_path(self.cell_kinds, (0, 0), open_cells, people=people)
                        if safe_open_path:
                            target = safe_open_path[-1][2]
        initial_cells = edge_cells(self.cell_kinds, direction)
        initial_forward = filter_forward_edge_cells(initial_cells, direction)
        initial_open = filter_open_lane_edge_cells(
            initial_forward or initial_cells,
            direction,
            self.cell_kinds,
            getattr(self._env, "_map", None),
            identified=self.identified,
            people=self._people_now(),
        ) or initial_forward or initial_cells
        initial_edge_runs = edge_runs(initial_cells, direction)
        target_run = next((r for r in initial_edge_runs if target in r), [target])
        self.target_cell = target
        initial_frame = self._get_state()["core"]["current_frame"]
        code = self._walk({target: None}, prefer_lateral=True)
        if code == MOVE_BLOCKED:
            if is_on_edge(self._displacement, direction) and self._can_step_from_current(direction) is not False:
                target = self._displacement
                target_run = next((r for r in initial_edge_runs if target in r), [target])
                self.target_cell = target
                self.report.termination_reason = None
                self.report.notes = None
                code = None
            elif (
                self._n_moves < self._max_steps
                and self._get_state()["pokemon_core"]["agent_state"] == AgentState.FREE_ROAM
                and not has_bottom_dialogue_box(self._get_state()["core"]["current_frame"])
            ):
                dx_d, dy_d = self._displacement
                fallback_map = {
                    (oc[0] - dx_d, oc[1] - dy_d): oc
                    for oc in initial_open
                    if oc != target and in_bounds((oc[0] - dx_d, oc[1] - dy_d))
                }
                if fallback_map:
                    self._refresh_local_perception()
                    fb_avoid = warp_cells(self.identified, getattr(self._env, "_map", None))
                    fb_path = shortest_path(
                        self.cell_kinds,
                        (0, 0),
                        list(fallback_map.keys()),
                        prefer_lateral=True,
                        avoid_cells=fb_avoid,
                        people=self._people_now(),
                    )
                    if fb_path is not None:
                        rel_fb = fb_path[-1][2]
                        orig_fb = fallback_map[rel_fb]
                        self.report.termination_reason = None
                        self.report.notes = None
                        code = self._walk({rel_fb: None}, prefer_lateral=True)
                        if code is None:
                            target = orig_fb
                            target_run = next((r for r in initial_edge_runs if target in r), [target])
                            self.target_cell = target
                        elif (
                            code == MOVE_BLOCKED
                            and is_on_edge(self._displacement, direction)
                            and self._can_step_from_current(direction) is not False
                        ):
                            target = self._displacement
                            target_run = next((r for r in initial_edge_runs if target in r), [target])
                            self.target_cell = target
                            self.report.termination_reason = None
                            self.report.notes = None
                            code = None
        if code is not None:
            return code
        after_walk_frame = self._get_state()["core"]["current_frame"]
        scrolled_to_edge = self._steps_taken > 0 and (
            self._screen_scrolled_along(initial_frame, after_walk_frame, direction)
            or self._at_exterior_boundary(direction)
        )
        can_step = self._can_step_from_current(direction)
        if can_step is False:
            dx_dir, dy_dir = DIRECTIONS[direction]
            step_cell = (dx_dir, dy_dir)
            _, live_identified = self._live_perception()
            step_is_warp = step_cell in warp_cells(
                identified=live_identified, global_map=getattr(self._env, "_map", None)
            )
            if not step_is_warp:
                self._record_empirical_block(direction)
            sidesteps = self._sidestep_targets_at_edge(target, target_run, direction)
            if (
                sidesteps
                and self._n_moves < self._max_steps
                and self._get_state()["pokemon_core"]["agent_state"] == AgentState.FREE_ROAM
                and not has_bottom_dialogue_box(self._get_state()["core"]["current_frame"])
            ):
                rel_target, new_edge_target = sidesteps[0]
                self._refresh_local_perception(blocked_neighbor=None if step_is_warp else DIRECTIONS.get(direction))
                sidestep_code = self._walk({rel_target: None}, prefer_lateral=True)
                if sidestep_code is not None:
                    return sidestep_code
                target = new_edge_target
                self.target_cell = target
                can_step = self._can_step_from_current(direction)
                if can_step is False:
                    _, live_identified2 = self._live_perception()
                    if step_cell not in warp_cells(
                        identified=live_identified2, global_map=getattr(self._env, "_map", None)
                    ):
                        self._record_empirical_block(direction)
                    if scrolled_to_edge:
                        state = self._settle_free_roam()
                        if self._last_terminated:
                            return self._fail(MOVE_TERMINATED, "terminated")
                        if self._last_truncated:
                            return self._fail(MOVE_TRUNCATED, "truncated")
                        interrupted = self._check_post_settle_interruption(state, f"at {target} on the {direction} edge")
                        if interrupted is not None:
                            return interrupted
                        self.report.termination_reason = "left_screen"
                        self.report.notes = f"Walked to {target} at the {direction} edge (further movement {direction} blocked at {target})"
                        return MOVE_OK
                    return self._fail(MOVE_BLOCKED, "move_blocked", f"Blocked at {target} moving {direction}")
            else:
                if scrolled_to_edge:
                    state = self._settle_free_roam()
                    if self._last_terminated:
                        return self._fail(MOVE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(MOVE_TRUNCATED, "truncated")
                    interrupted = self._check_post_settle_interruption(state, f"at {target} on the {direction} edge")
                    if interrupted is not None:
                        return interrupted
                    self.report.termination_reason = "left_screen"
                    self.report.notes = f"Walked to {target} at the {direction} edge (further movement {direction} blocked at {target})"
                    return MOVE_OK
                return self._fail(MOVE_BLOCKED, "move_blocked", f"Blocked at {target} moving {direction}")
        code = self._step(direction, 1)
        if code is not None:
            if code == MOVE_BLOCKED and scrolled_to_edge:
                state = self._settle_free_roam()
                if self._last_terminated:
                    return self._fail(MOVE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._fail(MOVE_TRUNCATED, "truncated")
                interrupted = self._check_post_settle_interruption(state, f"at {target} on the {direction} boundary")
                if interrupted is not None:
                    return interrupted
                self.report.termination_reason = "left_screen"
                self.report.notes = f"Walked to {target} at the {direction} boundary"
                return MOVE_OK
            return code
        state = self._settle_free_roam()
        if self._last_terminated:
            return self._fail(MOVE_TERMINATED, "terminated")
        if self._last_truncated:
            return self._fail(MOVE_TRUNCATED, "truncated")
        interrupted = self._check_post_settle_interruption(state, f"after stepping {direction} from {target}")
        if interrupted is not None:
            return interrupted
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

BATTLE_EXIT_SETTLE_WAIT = 2
MAX_TURN_PROGRESS_STEPS = 135
MAX_BATTLE_DIALOGUE_STEPS = 90
MAX_IDLE_PROGRESS_STEPS = 18
MAX_IDLE_INTRO_STEPS = 12
MAX_IDLE_BATTLE_DIALOGUE_STEPS = 30
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
            if is_false_overworld_menu_state(frame):
                return AgentState.FREE_ROAM
    elif raw_state == AgentState.IN_DIALOGUE:
        if (
            (is_overworld_menu_or_ui_screen(frame) or is_tm_hm_startup_dialogue_frame(frame))
            and not is_battle_screen(frame)
            and not (has_top_dialogue_box(frame) and not has_menu_cursor(frame))
        ):
            return AgentState.IN_MENU
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
- fight: attack, and say which of the four moves to use (1 is the top one; use your strongest non-resisted damaging move with PP left rather than non-damaging accuracy/stat moves against lower-level opponents you can defeat directly, and prefer STAB damaging moves over low-power non-STAB utility HMs like CUT when neither is resisted; if the opponent is using an escalating multi-turn attack like ROLLOUT or FURY CUTTER that doubles in power each consecutive hit and you cannot knock it out in one hit, use an accuracy-lowering move like SMOKESCREEN, SAND-ATTACK, or FLASH immediately if available and not yet applied; if a move was already shown to be "not very effective" or had no effect against the current opponent and another damaging move with PP left is available that is not resisted, switch to that non-resisted damaging move rather than repeating the resisted move, even if a general battle tip prefers the resisted move)
- progress: press through battle text, or continue when there is no choice to make
- run: try to escape. This never works against another trainer
- bag: open the bag to use an item; choose this in a trainer battle whenever your active Pokemon's HP is at or below 50% of max HP (do not gamble on knocking out a healthy opponent first when your HP is at or below 50%), unless the opponent's HP bar is already at critical red (nearly empty) and you can finish it off immediately on this turn with a damaging move, or the opponent's attack deals as much or more damage per turn than your healing item restores (or leaves you in 1-hit KO range, making bag healing futile/net-negative), or a note below says the bag has no usable item or says not to heal from the bag
- pokemon: open your team to switch; only choose it if switching is needed

Respond in exactly this format:
Reasoning: <what the screen shows (including your active Pokemon's current/max HP, e.g. 12/21 HP) and what to do about it>
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

That reply could not be used: it did not give the number of a usable move with PP left. Choose which of the four moves to use, where 1 is the top one (never choose a move listed as having 0 PP left).

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
If you opened the bag/pack to heal your active Pokemon:
- Check the description box at the bottom of the Pack/Item screen to see which row the cursor is currently on (when the cursor is on an item, the bottom box describes that item, e.g. "Restores POKeMON HP by 20."; when the bottom description box is blank while items are listed above CANCEL, the cursor is on CANCEL—press up to move the cursor onto an item instead of pressing confirm on CANCEL).
- Check your active Pokemon's current/max HP and missing HP shown below: when your active Pokemon is missing 30 or more HP and a stronger healing item (such as one restoring 50 HP instead of 20 HP) is listed in the bag, move the cursor to the stronger healing item and press confirm so you recover enough HP to survive the opponent's next hit in one turn.
- Once the 2-option USE / QUIT action box opens on the healing item, press confirm on USE.
- Once the party selection screen opens, move the cursor to the Pokemon that needs healing (usually your active lead Pokemon at the top) and press confirm to use the item on it.
If a move-learning prompt asks whether to delete/forget an older move to make room for a new move:
- To learn the new move: select YES (press confirm when the cursor is on YES), then move the cursor to the weakest or least useful move in the move list and press confirm to forget it.
- To skip learning the new move: select NO (press back) on deleting an older move, and then when asked whether to stop/abandon learning the move, select YES (press confirm while the cursor is on YES). Never press back or select NO on both prompts, as that repeats the move-learning question in an endless loop.

The presses you can make:
- up, down, left, right: move the cursor
- confirm: press A to choose what the cursor is on
- back: press B to go back, say no, or close the menu

Respond in exactly this format:
Reasoning: <what the menu or choice box shows, and what should be pressed next>
Press: <one of up, down, left, right, confirm, back>
[STOP]"""

    def __init__(
        self,
        env,
        task,
        max_steps: int = 60,
        initial_dialogue_crops: Optional[List[Any]] = None,
        skip_intro_ocr: bool = False,
        initial_zero_pp_moves: Optional[List[str]] = None,
        initial_bag_unavailable: bool = False,
        party_state: Optional[PartyInventoryState] = None,
        **kwargs,
    ):
        self._party_state: Optional[PartyInventoryState] = party_state
        self.history: List[str] = []
        self.dialogue_crops: List[Any] = (
            drop_prefix_crops([c for c in initial_dialogue_crops if not is_battle_menu_crop(c)])
            if initial_dialogue_crops
            else []
        )
        self._pending_initial_ocr: bool = bool(self.dialogue_crops)
        self.dialogue_text: List[str] = []
        self._skip_intro_ocr: bool = bool(skip_intro_ocr)
        self._exited_to_free_roam: bool = False
        task_says_bag_empty = bool(
            re.search(
                r"\b(?:do\s+not\s+check\s+the\s+bag|no\s+(?:usable\s+)?healing\s+items\s+(?:left\s+)?in\s+(?:the\s+)?bag|bag\s+is\s+empty)\b",
                str(task or ""),
                re.IGNORECASE,
            )
        )
        task_wants_ball = bool(
            re.search(
                r"\b(?:catch|capture|poke\s*ball|poké\s*ball|great\s*ball|ultra\s*ball)\b",
                str(task or ""),
                re.IGNORECASE,
            )
        )
        state_bag_empty = bool(
            party_state is not None
            and party_state.healing_bag_empty
            and (not party_state.active_pokemon_status or party_state.status_heal_bag_empty)
            and not task_wants_ball
        )
        self._bag_unavailable: bool = bool(initial_bag_unavailable or task_says_bag_empty or state_bag_empty)
        self._in_bag_menu: bool = False
        self._in_pokemon_menu: bool = False
        self._party_effect_pending: bool = False
        self._saw_battle_state: bool = False
        self._battle_place: Optional[Tuple[int, Tuple[int, int]]] = None
        self.battle_lost: bool = False
        self._move_names: Dict[int, str] = (
            dict(party_state.active_move_names) if (party_state is not None and getattr(party_state, "active_move_names", None)) else {}
        )
        self._move_types: Dict[int, str] = (
            dict(party_state.active_move_types) if (party_state is not None and getattr(party_state, "active_move_types", None)) else {}
        )
        self._move_pp: Dict[int, Tuple[int, int]] = {}
        self._zero_pp_slots: Dict[int, str] = {}
        self._known_zero_pp_names: set = set()
        self._current_opponent_name: Optional[str] = None
        self._resisted_moves_for_opponent: Dict[int, str] = {}
        self._immune_moves_for_opponent: Dict[int, str] = {}
        self._opponent_escalating_move: Optional[str] = None
        self._opponent_accuracy_lowered: int = 0
        self._last_opponent_damage: Optional[int] = None
        self._max_opponent_damage: int = 0
        self._last_heal_amount: Optional[int] = None
        self._last_turn_was_net_negative_heal: bool = False
        self._turn_start_hp: Optional[Tuple[int, int]] = None
        self._seen_bag_items: List[str] = list(party_state.items) if party_state is not None else []
        self._bag_item_quantities: Dict[str, Optional[int]] = (
            dict(party_state.quantities) if party_state is not None else {}
        )
        self._saw_bag_top: bool = False
        self._saw_bag_bottom: bool = False
        self._saw_bag_end: bool = False
        self._bag_scrolled_in_opening: bool = False
        self._opened_bag_in_battle: bool = False
        self._healing_items_exhausted: bool = bool(
            party_state.healing_exhausted if party_state is not None else False
        )
        self._bag_cursor_idx: Optional[int] = None
        self._selected_bag_item: Optional[str] = None
        seed_zero_pp = (
            initial_zero_pp_moves
            if initial_zero_pp_moves is not None
            else (party_state.zero_pp_moves if party_state is not None else ())
        )
        for entry in (seed_zero_pp or ()):
            text = str(entry).strip()
            if not text:
                continue
            m_slot = re.search(r"\bmove\s*([1-4])\b", text, re.IGNORECASE)
            m_name = re.search(r"\(([A-Z0-9 -]+)\)", text)
            name_val = m_name.group(1).strip().upper() if m_name else ""
            if not name_val and not m_slot:
                name_val = text.upper()
            if m_slot:
                slot_num = int(m_slot.group(1))
                self._zero_pp_slots[slot_num] = name_val or self._zero_pp_slots.get(slot_num, "")
            if name_val:
                self._known_zero_pp_names.add(name_val)
        super().__init__(env, task, max_steps, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _agent_state(self):
        return self._get_state()["pokemon_core"]["agent_state"]

    def _current_frame(self) -> np.ndarray:
        return self._get_state()["core"]["current_frame"]

    MENU_LOW_LEVEL_ACTIONS = {
        "up": LowLevelActions.PRESS_ARROW_UP,
        "down": LowLevelActions.PRESS_ARROW_DOWN,
        "left": LowLevelActions.PRESS_ARROW_LEFT,
        "right": LowLevelActions.PRESS_ARROW_RIGHT,
        "confirm": LowLevelActions.PRESS_BUTTON_A,
        "back": LowLevelActions.PRESS_BUTTON_B,
    }

    def _is_in_battle_sub_screen(self, frame: np.ndarray) -> bool:
        if is_gen1_item_list_screen(frame) or is_pack_item_list_screen(frame) or is_pack_action_subbox_screen(frame):
            return bool(self._in_bag_menu or self._saw_battle_state)
        if is_party_menu_selection_screen(frame) and not self._party_effect_pending:
            return bool(self._in_bag_menu or self._in_pokemon_menu or self._saw_battle_state)
        return False

    def _has_battle_menu_cursor(self, frame: np.ndarray) -> bool:
        if self._is_in_battle_sub_screen(frame):
            return True
        if is_gen1_item_list_screen(frame) or is_pack_item_list_screen(frame):
            return bool(self._in_bag_menu or not self._saw_battle_state)
        return has_menu_cursor(frame)

    def _is_in_battle_choice_menu(self, frame: np.ndarray) -> bool:
        return (
            self._has_battle_menu_cursor(frame)
            and not is_battle_command_menu(frame)
            and not is_move_selection_screen(frame)
        )

    def _take_battle_menu_press(self, press: str) -> EnvironmentStepRecord:
        if self._agent_state() == AgentState.IN_MENU:
            record = self._take_action(MenuAction, menu_action=press)
            if (
                record.action_success == -1
                and not record.frame_changed
                and self._agent_state() != AgentState.IN_MENU
            ):
                return self._take_action(
                    LowLevelAction, low_level_action=self.MENU_LOW_LEVEL_ACTIONS[press]
                )
            return record
        return self._take_action(
            LowLevelAction, low_level_action=self.MENU_LOW_LEVEL_ACTIONS[press]
        )

    def _non_battle_menu_outcome(self, frame: np.ndarray) -> Optional[AgentState]:
        if effective_agent_state(AgentState.IN_MENU, frame) == AgentState.IN_DIALOGUE:
            return AgentState.IN_DIALOGUE
        if is_start_menu_screen(frame) or (
            not (self._in_bag_menu or self._in_pokemon_menu or self._saw_battle_state)
            and is_overworld_pack_menu(frame)
        ):
            return AgentState.IN_MENU
        if not self._saw_battle_state and (
            is_party_menu_screen(frame) or is_pack_screen(frame) or has_menu_cursor(frame)
        ):
            return AgentState.IN_MENU
        return None

    def _current_map_place(self) -> Optional[Tuple[int, Tuple[int, int]]]:
        gmap = getattr(self._env, "_map", None)
        if gmap is None or not getattr(gmap, "current", None) or getattr(gmap, "pos", None) is None:
            return None
        try:
            if getattr(gmap, "battle_place", None) is not None and getattr(gmap, "in_battle", False):
                return gmap.resolve(*gmap.battle_place)
            return gmap.resolve(gmap.segment, gmap.pos)
        except Exception:  # noqa: BLE001
            return None

    def _ensure_scripted(self, description: str) -> None:
        if self._current_call is None:
            self._begin_scripted(description)

    def _read_new_text(self, records: List[EnvironmentStepRecord], *, new_turn: bool = False) -> List[str]:
        if isinstance(records, EnvironmentStepRecord):
            records = [records]
        crops = [crop for crop in captured_dialogue_crops(records) if not is_battle_menu_crop(crop)]
        if self._pending_initial_ocr:
            self._pending_initial_ocr = False
            if crops:
                drop_prefix_crops(crops, self.dialogue_crops)
            fresh = list(self.dialogue_crops)
        elif not crops:
            return []
        elif new_turn:
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
        if is_battle_defeat_text(texts):
            self.battle_lost = True
        return texts

    def _confirm_battle_exit(self, records: List[EnvironmentStepRecord]) -> bool:
        self._ensure_scripted("settle battle exit")
        state = self._agent_state()
        waited = 0
        for _ in range(BATTLE_EXIT_SETTLE_WAIT):
            if state != AgentState.FREE_ROAM or self._last_terminated or self._last_truncated:
                break
            wait_record = self._take_action(WaitAction)
            records.append(wait_record)
            waited += 1
            state = self._agent_state()
        if state != AgentState.IN_BATTLE and not is_battle_screen(self._current_frame()):
            self._exited_to_free_roam = True
            if (
                state == AgentState.FREE_ROAM
                and waited >= BATTLE_EXIT_SETTLE_WAIT
                and not (self._last_terminated or self._last_truncated)
            ):
                cur_step = getattr(getattr(self._env, "_emulator", None), "step_count", None)
                if cur_step is not None:
                    try:
                        self._env._last_settled_step = cur_step
                    except Exception:  # noqa: BLE001
                        pass
            return True
        return False

    def _advance_until_decision(self, records: List[EnvironmentStepRecord]) -> Tuple[int, int]:
        self._ensure_scripted("advance battle text")
        progress_presses = 0
        stat_dismisses = 0
        consecutive_dismisses = 0
        idle_steps = 0
        turn_steps = 0
        dialogue_steps = 0
        dialogue_idle_steps = 0
        while turn_steps < MAX_TURN_PROGRESS_STEPS and dialogue_steps < MAX_BATTLE_DIALOGUE_STEPS:
            if self._last_terminated or self._last_truncated:
                break
            state = self._agent_state()
            if state == AgentState.IN_BATTLE:
                self._saw_battle_state = True
            frame = self._current_frame()
            if state == AgentState.FREE_ROAM:
                if self._confirm_battle_exit(records):
                    break
                turn_steps += 1
                continue
            if state == AgentState.IN_DIALOGUE or (not has_menu_cursor(frame) and is_evolution_screen(frame)):
                if is_battle_decision_screen(frame):
                    break
                extra = self._take_action(LowLevelAction, low_level_action=LowLevelActions.PRESS_BUTTON_A)
                records.append(extra)
                progress_presses += 1
                dialogue_steps += 1
                consecutive_dismisses = 0
                idle_steps = 0
                if is_same_battle_frame(extra.frame_before, extra.frame_after):
                    dialogue_idle_steps += 1
                    if dialogue_idle_steps >= MAX_IDLE_BATTLE_DIALOGUE_STEPS:
                        break
                else:
                    dialogue_idle_steps = 0
                continue
            if state == AgentState.IN_MENU:
                if self._non_battle_menu_outcome(frame) is not None or self._has_battle_menu_cursor(frame):
                    self._party_effect_pending = False
                    break
                if self._party_effect_pending and is_party_menu_screen(frame) and not has_bottom_dialogue_box(frame):
                    wait_rec = self._take_action(WaitAction)
                    records.append(wait_rec)
                    turn_steps += 1
                    if is_same_battle_frame(wait_rec.frame_before, wait_rec.frame_after):
                        idle_steps += 1
                        if idle_steps >= 2:
                            self._party_effect_pending = False
                    else:
                        idle_steps = 0
                    continue
                if self._party_effect_pending and is_party_menu_screen(frame) and has_bottom_dialogue_box(frame):
                    wait_rec = self._take_action(WaitAction)
                    records.append(wait_rec)
                    turn_steps += 1
                    if not is_same_battle_frame(wait_rec.frame_before, wait_rec.frame_after):
                        idle_steps = 0
                        continue
                in_battle_intro = (
                    is_battle_screen(frame)
                    or is_party_menu_screen(frame)
                    or (has_bottom_dialogue_box(frame) and not self._has_battle_menu_cursor(frame))
                )
                if not in_battle_intro and consecutive_dismisses >= MAX_STAT_DISMISS_STEPS:
                    break
                extra = self._take_action(MenuAction, menu_action="confirm")
                records.append(extra)
                turn_steps += 1
                if in_battle_intro:
                    progress_presses += 1
                    consecutive_dismisses = 0
                else:
                    stat_dismisses += 1
                    consecutive_dismisses += 1
                if not in_battle_intro and extra.action_success == -1 and self._agent_state() == AgentState.IN_MENU:
                    break
                if is_same_battle_frame(extra.frame_before, extra.frame_after):
                    idle_steps += 1
                    if idle_steps >= (MAX_IDLE_INTRO_STEPS if in_battle_intro else MAX_IDLE_PROGRESS_STEPS):
                        break
                else:
                    idle_steps = 0
                continue
            if is_battle_decision_screen(frame):
                break
            extra = self._take_action(LowLevelAction, low_level_action=LowLevelActions.PRESS_BUTTON_B)
            records.append(extra)
            turn_steps += 1
            if self._last_terminated or self._last_truncated:
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

    ACCURACY_REDUCTION_MOVE_NAMES = (
        "SMOKESCREEN",
        "SMOKESCR",
        "SAND-ATTACK",
        "SAND ATTACK",
        "FLASH",
        "KINESIS",
        "MUD-SLAP",
    )

    LOW_POWER_UTILITY_HM_MOVES = (
        "CUT",
        "ROCK SMASH",
        "WHIRLPOOL",
    )

    def _reset_opponent_combat_stats(self) -> None:
        self._resisted_moves_for_opponent.clear()
        self._immune_moves_for_opponent.clear()
        self._opponent_escalating_move = None
        self._opponent_accuracy_lowered = 0
        self._last_opponent_damage = None
        self._max_opponent_damage = 0
        self._last_turn_was_net_negative_heal = False

    def _update_opponent_tracking(self, frame: Optional[np.ndarray] = None, texts: Optional[List[str]] = None) -> None:
        if texts:
            joined = "\n".join(texts)
            if re.search(r"\b(?:sent\s+out|fainted!|about\s+to\s+use)\b", joined, re.IGNORECASE):
                self._reset_opponent_combat_stats()
            m_esc = re.search(
                r"\b(?:Enemy|Wild|Foe)\s+[A-Z0-9♀♂.'-]+\s+used\s+(ROLLOUT|FURY\s+CUTTER|RAGE|BIDE)!",
                joined,
                re.IGNORECASE,
            )
            if m_esc:
                self._opponent_escalating_move = " ".join(m_esc.group(1).upper().split())
            if re.search(
                r"\b(?:Enemy|Wild|Foe)\s+[A-Z0-9♀♂.'-]+(?:'s|’s)?\s+accuracy\s+fell!",
                joined,
                re.IGNORECASE,
            ):
                self._opponent_accuracy_lowered += 1
        if frame is not None:
            lvl = read_battle_hud_level(frame) or read_party_menu_level(frame)
            if lvl is not None and self._party_state is not None:
                self._party_state.record_active_level(lvl)
            opp = read_battle_opponent_hud_name(frame)
            if opp:
                if self._current_opponent_name and opp != self._current_opponent_name:
                    self._reset_opponent_combat_stats()
                self._current_opponent_name = opp

    def _best_available_heal_amount(self, max_hp: int) -> Optional[int]:
        amounts = [
            amt
            for it in self._seen_bag_items
            if is_healing_item_name(it)
            for amt in (healing_item_restore_amount(it, max_hp),)
            if amt is not None
        ]
        if amounts:
            return max(amounts)
        if self._last_heal_amount is not None:
            return self._last_heal_amount
        task_text = (self._task or "").upper()
        task_amounts = [
            amt
            for it in (
                "FULL RESTORE",
                "MAX POTION",
                "HYPER POTION",
                "MOOMOO MILK",
                "LEMONADE",
                "SODA POP",
                "SUPER POTION",
                "FRESH WATER",
                "GOLD BERRY",
                "POTION",
                "BERRY",
            )
            if it in task_text
            for amt in (healing_item_restore_amount(it, max_hp),)
            if amt is not None
        ]
        if task_amounts:
            return max(task_amounts)
        return None

    def _get_bag_healing_futility(self, cur_hp: int, max_hp: int) -> Optional[Tuple[int, int, int]]:
        if max_hp <= 0 or cur_hp >= max_hp:
            return None
        obs_dmg = self._last_opponent_damage if self._last_opponent_damage is not None else self._max_opponent_damage
        if not obs_dmg or obs_dmg <= 0:
            return None
        best_heal = self._best_available_heal_amount(max_hp)
        if best_heal is None:
            if self._last_turn_was_net_negative_heal:
                best_heal = self._last_heal_amount or 20
            else:
                return None
        healed_hp = min(max_hp, cur_hp + best_heal)
        actual_gain = max(0, healed_hp - cur_hp)
        if (
            self._last_turn_was_net_negative_heal
            or actual_gain <= obs_dmg
            or healed_hp <= obs_dmg
        ):
            return (obs_dmg, best_heal, healed_hp)
        return None

    def _record_turn_hp_transition(
        self,
        pre_turn_hp: Optional[Tuple[int, int]],
        turn_records: List[EnvironmentStepRecord],
        texts: List[str],
        *,
        used_heal_item: Optional[str] = None,
    ) -> None:
        joined = "\n".join(texts or ())
        if re.search(r"\b(?:sent\s+out|fainted!|about\s+to\s+use)\b", joined, re.IGNORECASE):
            return
        raw_pairs: List[Tuple[int, int]] = []
        if pre_turn_hp is not None:
            raw_pairs.append(pre_turn_hp)
        for rec in turn_records or ():
            for fr in (rec.frame_before, rec.frame_after):
                hp = read_battle_hud_hp(fr) or read_party_menu_hp(fr)
                if hp is not None:
                    raw_pairs.append(hp)
        cur_after = read_battle_hud_hp(self._current_frame()) or read_party_menu_hp(self._current_frame())
        if cur_after is not None:
            raw_pairs.append(cur_after)
        if len(raw_pairs) < 2:
            return
        target_max = pre_turn_hp[1] if pre_turn_hp is not None else raw_pairs[-1][1]
        if target_max <= 0:
            return
        hps = [hp[0] for hp in raw_pairs if hp[1] == target_max]
        if len(hps) < 2:
            return
        start_hp = hps[0]
        end_hp = hps[-1]
        peak_hp = max(hps)
        m_rec = re.search(r"\b(?:recovered\s+by|restored|recovered)\s+(\d{1,3})\b", joined, re.IGNORECASE)
        text_amt = int(m_rec.group(1)) if m_rec else None
        item_amt = (
            healing_item_restore_amount(used_heal_item, target_max)
            if used_heal_item and is_healing_item_name(used_heal_item)
            else None
        )
        healed_on_turn = (peak_hp > start_hp) or (text_amt is not None) or (item_amt is not None)
        if healed_on_turn:
            eff_peak = max(peak_hp, min(target_max, start_hp + (text_amt or item_amt or 0)))
            obs_heal = max(eff_peak - start_hp, text_amt or 0, item_amt or 0)
            if obs_heal > 0:
                self._last_heal_amount = obs_heal
            if end_hp < eff_peak:
                dmg = eff_peak - end_hp
                self._last_opponent_damage = dmg
                self._max_opponent_damage = max(self._max_opponent_damage, dmg)
            self._last_turn_was_net_negative_heal = end_hp <= start_hp
        else:
            self._last_turn_was_net_negative_heal = False
            if end_hp < start_hp:
                dmg = start_hp - end_hp
                self._last_opponent_damage = dmg
                self._max_opponent_damage = max(self._max_opponent_damage, dmg)

    def _record_move_effectiveness(self, move_slot: Optional[int], texts: List[str]) -> str:
        if move_slot is None or not texts:
            return ""
        joined = "\n".join(texts)
        move_name = self._move_names.get(move_slot, f"Move {move_slot}")
        if re.search(r"\b(?:doesn['’]t\s+affect|it\s+had\s+no\s+effect)\b", joined, re.IGNORECASE):
            self._immune_moves_for_opponent[move_slot] = move_name
            return " (had NO EFFECT on the current opponent — switch to a different move)"
        if re.search(r"\bnot\s+very\s+effective\b", joined, re.IGNORECASE):
            self._resisted_moves_for_opponent[move_slot] = move_name
            return " (was NOT VERY EFFECTIVE against the current opponent — prefer another damaging move if available)"
        if re.search(r"\bsuper\s+effective\b", joined, re.IGNORECASE):
            self._resisted_moves_for_opponent.pop(move_slot, None)
            self._immune_moves_for_opponent.pop(move_slot, None)
        return ""

    def _scan_move_menu_frame(self, frame: Optional[np.ndarray]) -> None:
        if frame is None:
            return
        self._update_opponent_tracking(frame=frame)
        names = read_move_menu_names(frame)
        if names:
            for slot, name in names.items():
                self._move_names[slot] = name
                if self._party_state is not None:
                    self._party_state.active_move_names[slot] = name
                if slot in self._zero_pp_slots and not self._zero_pp_slots[slot]:
                    self._zero_pp_slots[slot] = name
                for known_name in list(self._known_zero_pp_names):
                    if name == known_name or name.startswith(known_name) or known_name.startswith(name):
                        self._zero_pp_slots[slot] = name
        slot = read_move_menu_cursor_slot(frame)
        pp = read_move_menu_pp(frame)
        mtype = read_move_menu_type(frame)
        if slot is not None and mtype:
            self._move_types[slot] = mtype
            if self._party_state is not None:
                self._party_state.active_move_types[slot] = mtype
        if slot is not None and pp is not None:
            self._move_pp[slot] = pp
            if pp[0] == 0:
                move_name = self._move_names.get(slot, "")
                self._zero_pp_slots[slot] = move_name
                if move_name:
                    self._known_zero_pp_names.add(move_name)
            elif slot in self._zero_pp_slots:
                old_name = self._zero_pp_slots.pop(slot, "")
                if old_name:
                    self._known_zero_pp_names.discard(old_name)

    def _scan_record_move_frames(self, record: Optional[EnvironmentStepRecord]) -> None:
        if record is None:
            return
        self._scan_move_menu_frame(record.frame_before)
        self._scan_move_menu_frame(record.frame_after)
        last_ev = getattr(self._env, "last_step_event", None)
        passed = getattr(last_ev, "passed_frames", None) if last_ev is not None else None
        if passed is not None:
            for fr in passed:
                self._scan_move_menu_frame(fr)

    def _mark_zero_pp_slot(self, slot: int) -> None:
        name = self._move_names.get(slot, "")
        self._zero_pp_slots[slot] = name
        if name:
            self._known_zero_pp_names.add(name)
        if slot in self._move_pp:
            self._move_pp[slot] = (0, self._move_pp[slot][1])

    def _format_zero_pp_list(self) -> List[str]:
        items: List[str] = []
        covered_names: set = set()
        for slot in sorted(self._zero_pp_slots):
            name = self._zero_pp_slots[slot] or self._move_names.get(slot, "")
            if name:
                items.append(f"Move {slot} ({name})")
                covered_names.add(name)
            else:
                items.append(f"Move {slot}")
        for name in sorted(self._known_zero_pp_names):
            if not any(
                name == cov or name.startswith(cov) or cov.startswith(name)
                for cov in covered_names
            ):
                items.append(name)
        return items

    def _dismiss_no_pp_dialogue(self, turn_records: List[EnvironmentStepRecord]) -> int:
        dismissed = 0
        for _ in range(4):
            frame = self._current_frame()
            if is_move_selection_screen(frame) or not is_no_pp_dialogue_frame(frame):
                break
            extra = self._take_action(LowLevelAction, low_level_action=LowLevelActions.PRESS_BUTTON_B)
            turn_records.append(extra)
            dismissed += 1
            if self._last_terminated or self._last_truncated:
                break
        self._scan_move_menu_frame(self._current_frame())
        return dismissed

    @staticmethod
    def _merge_visible_bag_items(seen: List[str], visible: List[str]) -> List[str]:
        return PartyInventoryState.merge_visible_items(seen, visible)

    def _scan_bag_menu_frame(self, frame: Optional[np.ndarray]) -> Optional[Tuple[List[str], bool, Optional[str]]]:
        if frame is None:
            return None
        info = read_bag_item_list(frame)
        if info is None:
            return None
        self._opened_bag_in_battle = True
        items, has_more = info
        quantities = dict(read_bag_item_quantities(frame) or ())
        cursor_idx = read_bag_cursor_row_index(frame, tracked_index=self._bag_cursor_idx)
        cursor_row = read_bag_cursor_row(frame, tracked_index=self._bag_cursor_idx)
        if cursor_row == "CANCEL":
            self._bag_cursor_idx = len(items)
        elif cursor_row in items:
            self._bag_cursor_idx = items.index(cursor_row)
        elif cursor_idx is not None:
            self._bag_cursor_idx = cursor_idx
        max_items_with_cancel = 3 if is_gen1_item_list_screen(frame) else 4
        if not has_more:
            self._saw_bag_bottom = True
            if len(items) < max_items_with_cancel:
                self._saw_bag_top = True
        if (
            not self._bag_scrolled_in_opening
            and cursor_idx == 0
            and bool(items)
            and (
                not self._seen_bag_items
                or items[0] not in self._seen_bag_items
                or self._seen_bag_items.index(items[0]) == 0
            )
        ):
            self._saw_bag_top = True
        if self._party_state is not None:
            self._seen_bag_items = self._party_state.observe_bag_window(
                items,
                quantities=quantities,
                has_more=has_more,
                at_top=self._saw_bag_top,
                at_bottom=self._saw_bag_bottom,
                max_items_with_cancel=max_items_with_cancel,
            )
            self._bag_item_quantities = dict(self._party_state.quantities)
        else:
            self._seen_bag_items = self._merge_visible_bag_items(self._seen_bag_items, items)
            for name in items:
                if name in quantities and quantities[name] is not None:
                    self._bag_item_quantities[name] = quantities[name]
            if not has_more and len(items) < max_items_with_cancel:
                for existing in list(self._seen_bag_items):
                    if existing not in items:
                        self._seen_bag_items.remove(existing)
                        self._bag_item_quantities.pop(existing, None)
        self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)
        if self._healing_items_exhausted and any(is_healing_item_name(it) for it in self._seen_bag_items):
            self._healing_items_exhausted = False
        return items, has_more, cursor_row

    def _consume_selected_bag_item(self, item_name: Optional[str]) -> None:
        if not item_name or item_name == "CANCEL":
            return
        if self._party_state is not None:
            exhausted = self._party_state.consume_item(item_name)
            self._seen_bag_items = list(self._party_state.items)
            self._bag_item_quantities = dict(self._party_state.quantities)
            if self._party_state.healing_exhausted:
                self._healing_items_exhausted = True
            if exhausted:
                has_heal_left = any(is_healing_item_name(it) for it in self._seen_bag_items)
                if (self._saw_bag_end or self._healing_items_exhausted or self._party_state.healing_bag_empty) and not has_heal_left:
                    live_status = self._last_seen_player_status()
                    if not live_status or not any(
                        is_status_heal_item_for_status(it, live_status) for it in self._seen_bag_items
                    ):
                        self._bag_unavailable = True
            return
        qty = self._bag_item_quantities.get(item_name)
        if qty is not None and qty > 1:
            self._bag_item_quantities[item_name] = qty - 1
        elif qty == 1:
            self._bag_item_quantities.pop(item_name, None)
            if item_name in self._seen_bag_items:
                self._seen_bag_items.remove(item_name)
            has_heal_left = any(is_healing_item_name(it) for it in self._seen_bag_items)
            if is_healing_item_name(item_name) and not has_heal_left:
                self._healing_items_exhausted = True
            if (self._saw_bag_end or self._healing_items_exhausted) and not has_heal_left:
                live_status = self._last_seen_player_status()
                if not live_status or not any(
                    is_status_heal_item_for_status(it, live_status) for it in self._seen_bag_items
                ):
                    self._bag_unavailable = True

    @staticmethod
    def _is_ball_item_name(name: str) -> bool:
        up = (name or "").strip().upper()
        return bool(up and up != "CANCEL" and ("BALL" in up or up == "POK"))

    def _select_best_in_battle_bag_target(
        self,
        candidates: List[str],
        *,
        wants_ball: bool,
        needs_heal: bool,
        low_hp: bool,
        deficit: int,
        live_status: Optional[str],
    ) -> Optional[str]:
        if wants_ball and not low_hp:
            for it in candidates:
                if self._is_ball_item_name(it):
                    return it
        if needs_heal:
            heal_items = [it for it in candidates if is_healing_item_name(it)]
            if heal_items:
                if deficit > 20:
                    stronger = [
                        it for it in heal_items
                        if not any(w in it.upper() for w in ("POTION", "BERRY"))
                        or any(w in it.upper() for w in ("SUPER", "HYPER", "MAX", "GOLD"))
                    ]
                    if stronger:
                        return stronger[0]
                return heal_items[0]
        if live_status:
            for it in candidates:
                if is_status_heal_item_for_status(it, live_status):
                    return it
        if wants_ball:
            for it in candidates:
                if self._is_ball_item_name(it):
                    return it
        return None

    def _auto_navigate_in_battle_bag(
        self,
        turn_records: List[EnvironmentStepRecord],
    ) -> Tuple[str, int, int]:
        cur_frame = self._current_frame()
        bag_info = self._scan_bag_menu_frame(cur_frame)
        if bag_info is None:
            return "", 0, 0
        live_hp = (
            read_party_menu_hp(cur_frame)
            or read_battle_hud_hp(cur_frame)
            or self._last_seen_player_hp()
        )
        live_status = self._last_seen_player_status(cur_frame)
        needs_heal = live_hp is None or (live_hp[1] > 0 and live_hp[0] < live_hp[1])
        low_hp = live_hp is not None and live_hp[1] > 0 and live_hp[0] * 2 <= live_hp[1]
        deficit = max(0, live_hp[1] - live_hp[0]) if live_hp is not None else 20
        wants_ball = bool(
            re.search(
                r"\b(?:catch|capture|poke\s*ball|poké\s*ball|great\s*ball|ultra\s*ball)\b",
                self._task or "",
                re.IGNORECASE,
            )
        )

        for _ in range(24):
            if self._last_terminated or self._last_truncated:
                return "", 0, 0
            cur_frame = self._current_frame()
            bag_info = self._scan_bag_menu_frame(cur_frame)
            if bag_info is None:
                break
            vis_items, has_more, cursor_row = bag_info
            vis_target = self._select_best_in_battle_bag_target(
                vis_items,
                wants_ball=wants_ball,
                needs_heal=needs_heal,
                low_hp=low_hp,
                deficit=deficit,
                live_status=live_status,
            )
            if vis_target is not None:
                if cursor_row == vis_target:
                    break
                if cursor_row == "CANCEL":
                    step_dir = "up"
                elif cursor_row in vis_items:
                    step_dir = "down" if vis_items.index(vis_target) > vis_items.index(cursor_row) else "up"
                else:
                    step_dir = "down" if has_more else "up"
            else:
                seen_target = self._select_best_in_battle_bag_target(
                    self._seen_bag_items,
                    wants_ball=wants_ball,
                    needs_heal=needs_heal,
                    low_hp=low_hp,
                    deficit=deficit,
                    live_status=live_status,
                )
                if seen_target is not None and vis_items and seen_target in self._seen_bag_items:
                    first_vis = vis_items[0]
                    if first_vis in self._seen_bag_items and self._seen_bag_items.index(seen_target) < self._seen_bag_items.index(first_vis):
                        step_dir = "up"
                    else:
                        step_dir = "down"
                elif not self._saw_bag_bottom:
                    step_dir = "down"
                elif not self._saw_bag_top:
                    step_dir = "up"
                else:
                    break

            self._ensure_scripted("navigate in-battle bag")
            self._bag_scrolled_in_opening = True
            n_vis = len(vis_items)
            max_idx = n_vis if not has_more else max(0, n_vis - 1)
            if self._bag_cursor_idx is not None:
                if step_dir == "down":
                    self._bag_cursor_idx = min(max_idx, self._bag_cursor_idx + 1)
                elif step_dir == "up":
                    self._bag_cursor_idx = max(0, self._bag_cursor_idx - 1)
            rec = self._take_battle_menu_press(step_dir)
            turn_records.append(rec)
            if self._last_terminated or self._last_truncated:
                return "", 0, 0
            after_info = self._scan_bag_menu_frame(self._current_frame())
            if after_info is None:
                break
            after_items, after_more, after_cursor = after_info
            if step_dir == "down" and (
                not after_more
                or (after_items == vis_items and after_cursor == cursor_row)
                or is_same_battle_frame(rec.frame_before, rec.frame_after)
            ):
                self._saw_bag_bottom = True
                if self._party_state is not None:
                    self._party_state.saw_bottom = True
                self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)
            elif step_dir == "up" and (
                (after_items == vis_items and after_cursor == cursor_row)
                or is_same_battle_frame(rec.frame_before, rec.frame_after)
            ):
                self._saw_bag_top = True
                if self._party_state is not None:
                    self._party_state.saw_top = True
                self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)

        cur_frame = self._current_frame()
        bag_info = self._scan_bag_menu_frame(cur_frame)
        if bag_info is None:
            return "", 0, 0
        vis_items, has_more, cursor_row = bag_info
        target_now = self._select_best_in_battle_bag_target(
            vis_items,
            wants_ball=wants_ball,
            needs_heal=needs_heal,
            low_hp=low_hp,
            deficit=deficit,
            live_status=live_status,
        )
        if target_now is not None and cursor_row == target_now:
            if not wants_ball and (
                (needs_heal and is_healing_item_name(cursor_row))
                or (live_status and is_status_heal_item_for_status(cursor_row, live_status))
            ):
                self._ensure_scripted(f"confirm {cursor_row} in battle bag")
                self._selected_bag_item = cursor_row
                confirm_rec = self._take_battle_menu_press("confirm")
                turn_records.append(confirm_rec)
                if self._last_terminated or self._last_truncated:
                    return "", 0, 0
                auto_steps = self._complete_in_battle_item_application(cursor_row, turn_records)
                if self._last_terminated or self._last_truncated:
                    return "", 0, 0
                b_presses, b_dismisses = 0, 0
                after_frame = self._current_frame()
                if (
                    self._agent_state() != AgentState.IN_MENU
                    or not self._has_battle_menu_cursor(after_frame)
                ):
                    b_presses, b_dismisses = self._advance_until_decision(turn_records)
                    if self._last_terminated or self._last_truncated:
                        return "", 0, 0
                    self._party_effect_pending = False
                    if (
                        self._agent_state() != AgentState.IN_MENU
                        or is_battle_command_menu(self._current_frame())
                    ):
                        self._in_bag_menu = False
                if b_presses > 0 or auto_steps:
                    self._consume_selected_bag_item(cursor_row)
                self._selected_bag_item = None
                step_desc = f", {' and '.join(auto_steps)}" if auto_steps else ""
                return f" (selected {cursor_row}{step_desc})", b_presses, b_dismisses
            return f" (scrolled bag cursor to {cursor_row})", 0, 0

        any_seen_target = self._select_best_in_battle_bag_target(
            self._seen_bag_items,
            wants_ball=wants_ball,
            needs_heal=needs_heal,
            low_hp=low_hp,
            deficit=deficit,
            live_status=live_status,
        )
        if any_seen_target is None and self._saw_bag_end and not wants_ball:
            self._ensure_scripted("close battle bag without usable healing item")
            back_rec = self._take_battle_menu_press("back")
            turn_records.append(back_rec)
            if self._last_terminated or self._last_truncated:
                return "", 0, 0
            self._bag_unavailable = True
            self._in_bag_menu = False
            return (
                " (checked all bag slots — no usable healing item in bag, so closed bag; do not choose bag again)",
                0,
                0,
            )
        return "", 0, 0

    def _complete_in_battle_item_application(
        self,
        selected_item: Optional[str],
        turn_records: List[EnvironmentStepRecord],
    ) -> List[str]:
        auto_steps: List[str] = []
        if not selected_item or selected_item == "CANCEL":
            return auto_steps
        if not (is_healing_item_name(selected_item) or is_status_heal_item_name(selected_item)):
            return auto_steps
        cur_frame = self._current_frame()
        if is_pack_action_subbox_screen(cur_frame):
            self._ensure_scripted("confirm USE on in-battle bag item")
            use_rec = self._take_battle_menu_press("confirm")
            turn_records.append(use_rec)
            auto_steps.append("confirmed USE")
            if self._last_terminated or self._last_truncated:
                return auto_steps
            cur_frame = self._current_frame()
        if is_party_menu_selection_screen(cur_frame):
            party_hp = read_party_menu_hp(cur_frame)
            party_st = read_party_menu_status(cur_frame)
            needs_heal = is_healing_item_name(selected_item) and (
                party_hp is None or party_hp[0] < party_hp[1]
            )
            needs_status = is_status_heal_item_name(selected_item) and bool(
                party_st or self._last_seen_player_status()
            )
            if needs_heal or needs_status:
                self._ensure_scripted("confirm active Pokemon on party screen")
                target_rec = self._take_battle_menu_press("confirm")
                turn_records.append(target_rec)
                auto_steps.append("confirmed active Pokemon on party screen")
                if self._last_terminated or self._last_truncated:
                    return auto_steps
                after_target = self._current_frame()
                if is_party_menu_screen(after_target) and not has_menu_cursor(after_target):
                    self._party_effect_pending = True
        return auto_steps

    def _history_block(self) -> str:
        cur_frame = self._current_frame()
        self._scan_move_menu_frame(cur_frame)
        bag_info = self._scan_bag_menu_frame(cur_frame)
        lines = list(self.history)
        live_hp = (
            read_party_menu_hp(cur_frame)
            or read_battle_hud_hp(cur_frame)
            or self._last_seen_player_hp()
        )
        live_status = self._last_seen_player_status(cur_frame)
        futility_info: Optional[Tuple[int, int, int]] = None
        if live_hp is not None:
            cur_hp, max_hp = live_hp
            deficit = max(0, max_hp - cur_hp)
            pct = int(round(100.0 * cur_hp / max_hp)) if max_hp > 0 else 100
            status_suffix = f", status: {live_status}" if live_status else ""
            opp_critical = is_battle_opponent_critical_hp(cur_frame)
            futility_info = self._get_bag_healing_futility(cur_hp, max_hp)
            low_hp_note = ""
            if opp_critical:
                low_hp_note = (
                    " — Opponent HP is CRITICAL (nearly empty HP bar): if you have a damaging move with PP left "
                    "that can knock out the opponent this turn, choose fight to finish off the opponent immediately before healing"
                )
            elif futility_info is not None:
                obs_dmg, best_heal, healed_hp = futility_info
                low_hp_note = (
                    f" — DO NOT HEAL FROM BAG: opponent's attack deals ~{obs_dmg} HP per hit while your best bag healing item "
                    f"restores at most {best_heal} HP (to {healed_hp}/{max_hp} HP, net-negative and within 1-hit KO range); "
                    f"choose fight to attack with your strongest non-resisted damaging move"
                )
            elif max_hp > 0 and cur_hp * 2 <= max_hp and not self._bag_unavailable:
                low_hp_note = (
                    " — LOW HP (<= 50% max HP): in a trainer battle, choose bag to heal now "
                    "rather than gambling on fight"
                )
            lines.append(
                f"Active Pokemon HP: {cur_hp}/{max_hp} HP ({pct}%, missing {deficit} HP{status_suffix}){low_hp_note}."
            )
        elif live_status:
            lines.append(f"Active Pokemon status: {live_status}.")
        if bag_info is not None:
            vis_items, has_more, cursor_row = bag_info
            formatted_rows = [
                f"{it} (x{self._bag_item_quantities[it]})"
                if self._bag_item_quantities.get(it) is not None
                else it
                for it in vis_items
            ]
            if not has_more:
                formatted_rows.append("CANCEL")
            more_parts: List[str] = []
            if has_more:
                more_parts.append("more items below")
            if not self._saw_bag_top:
                more_parts.append("top bag slots above this window not yet checked — press up to scroll up")
            more_note = f" ({'; '.join(more_parts)})" if more_parts else ""
            if formatted_rows:
                lines.append(f"Visible bag rows in order: {', '.join(formatted_rows)}{more_note}.")
            if cursor_row == "CANCEL":
                lines.append(
                    "Current bag cursor row: CANCEL (bottom description box is blank — "
                    "press up to move onto an item above CANCEL; do NOT press confirm on CANCEL)."
                )
            elif cursor_row:
                lines.append(f"Current bag cursor row: {cursor_row}.")
        if self._move_names:
            move_Descs: List[str] = []
            for s in sorted(self._move_names):
                m_name = self._move_names[s]
                details: List[str] = []
                if s in self._move_types:
                    details.append(f"type: {self._move_types[s]}")
                if m_name in self.LOW_POWER_UTILITY_HM_MOVES:
                    details.append("low-power utility HM — prefer STAB damaging moves when unresisted")
                if s in self._move_pp:
                    details.append(f"PP: {self._move_pp[s][0]}/{self._move_pp[s][1]}")
                elif s in self._zero_pp_slots:
                    details.append("PP: 0")
                if s in self._immune_moves_for_opponent:
                    details.append("NO EFFECT against current opponent")
                elif s in self._resisted_moves_for_opponent:
                    details.append("NOT VERY EFFECTIVE against current opponent")
                detail_str = f" [{', '.join(details)}]" if details else ""
                move_Descs.append(f"Move {s}: {m_name}{detail_str}")
            lines.append(f"Known active Pokemon moves: {'; '.join(move_Descs)}.")
        if self._opponent_escalating_move and not is_battle_opponent_critical_hp(cur_frame):
            acc_slots = [
                f"Move {s} ({self._move_names[s]})"
                for s in sorted(self._move_names)
                if s not in self._zero_pp_slots
                and any(
                    self._move_names[s] == a or self._move_names[s].startswith(a[:7])
                    for a in self.ACCURACY_REDUCTION_MOVE_NAMES
                )
            ]
            if acc_slots and self._opponent_accuracy_lowered < 2:
                lines.append(
                    f"Note: Opponent is using escalating multi-turn attack {self._opponent_escalating_move} "
                    f"(power doubles on every consecutive hit) — use {', '.join(acc_slots)} immediately to lower "
                    f"the opponent's accuracy and break the {self._opponent_escalating_move} chain before it snowballs."
                )
        if self._immune_moves_for_opponent or self._resisted_moves_for_opponent:
            bad_parts: List[str] = []
            for s in sorted(set(self._immune_moves_for_opponent) | set(self._resisted_moves_for_opponent)):
                m_name = self._move_names.get(s) or self._immune_moves_for_opponent.get(s) or self._resisted_moves_for_opponent.get(s) or f"Move {s}"
                t_str = f", type: {self._move_types[s]}" if s in self._move_types else ""
                eff_label = "had NO EFFECT" if s in self._immune_moves_for_opponent else "was NOT VERY EFFECTIVE"
                bad_parts.append(f"Move {s} ({m_name}{t_str}) {eff_label}")
            alt_moves = [
                f"Move {s} ({self._move_names[s]}" + (f", type: {self._move_types[s]}" if s in self._move_types else "") + ")"
                for s in sorted(self._move_names)
                if s not in self._zero_pp_slots
                and s not in self._immune_moves_for_opponent
                and s not in self._resisted_moves_for_opponent
                and self._move_names[s] not in ("GROWL", "LEER", "TAIL WHIP", "STRING SHOT", "SAND-ATTACK", "SAND ATTACK", "SMOKESCREEN", "SMOKESCR", "KINESIS", "FLASH", "DEFENSE CURL", "HARDEN", "WITHDRAW")
            ]
            alt_note = (
                f" Switch to a non-resisted damaging move with PP left ({', '.join(alt_moves)}) instead of repeating the resisted move."
                if alt_moves
                else ""
            )
            opp_label = f" against {self._current_opponent_name}" if self._current_opponent_name else " against the current opponent"
            lines.append(f"Note: {'; '.join(bad_parts)}{opp_label}.{alt_note}")
        zero_pp_items = self._format_zero_pp_list()
        if zero_pp_items:
            avail = [
                f"Move {s}" + (f" ({self._move_names[s]})" if s in self._move_names else "")
                for s in (1, 2, 3, 4)
                if s not in self._zero_pp_slots and (not self._move_names or s in self._move_names)
            ]
            avail_note = f" Choose from available moves with PP left: {', '.join(avail)}." if avail else ""
            lines.append(
                f"Note: Moves with 0 PP left (cannot be used — do NOT choose these moves): "
                f"{', '.join(zero_pp_items)}.{avail_note}"
            )
        if self._bag_unavailable:
            lines.append(
                "Note: You already checked the bag in this battle and had no usable item — "
                "do NOT choose bag again; choose fight instead."
            )
        elif futility_info is not None and live_hp is not None:
            obs_dmg, best_heal, healed_hp = futility_info
            lines.append(
                f"Note: Bag healing is futile against the current opponent (incoming damage is ~{obs_dmg} HP/turn "
                f"vs at most +{best_heal} HP healed to {healed_hp}/{live_hp[1]} HP) — do NOT choose bag; "
                f"choose fight to attack immediately."
            )
        if not lines:
            return ""
        return "\nWhat has happened in this battle so far:\n" + "\n".join(lines) + "\n"

    def _instructions_block(self) -> str:
        task = (self._task or "").strip()
        if not task or task.lower() in ("win the current battle", "none"):
            return "Win this battle.\n"
        return f"Your instructions for this battle: \"{task}\"\n"

    PLAYER_HP_PATTERN = re.compile(
        r"(?<!enemy\s)(?<!wild\s)(?<!foe\s)"
        r"(?:\bHP\s*(?:is|at|of|:|=)?\s*(\d{1,3})\s*/\s*(\d{1,3})\b"
        r"|\b(\d{1,3})\s*/\s*(\d{1,3})\s*HP\b"
        r"|\bHP\s*\(\s*(\d{1,3})\s*/\s*(\d{1,3})\s*\))",
        re.IGNORECASE,
    )

    PLAYER_STATUS_TAG_PATTERN = re.compile(
        r"\[Party status seen in (?:battle|menu):\s*(PSN|BRN|PAR|SLP|FRZ|healthy)\]",
        re.IGNORECASE,
    )

    ZERO_PP_TAG_PATTERN = re.compile(
        r"\[Moves at 0 PP:\s*([^\]]+)\]",
        re.IGNORECASE,
    )

    BAG_ITEMS_TAG_PATTERN = re.compile(
        r"\[Bag items seen:\s*([^\]]+)\]",
        re.IGNORECASE,
    )

    @classmethod
    def extract_bag_items_from_text(cls, text: str) -> Optional[Tuple[List[str], bool]]:
        if not text:
            return None
        last_result: Optional[Tuple[List[str], bool]] = None
        for m in cls.BAG_ITEMS_TAG_PATTERN.finditer(text):
            raw = m.group(1).strip()
            all_checked = False
            if re.search(r"\((?:all\s+bag\s+slots\s+checked|end\s+of\s+bag\s+reached)\)", raw, re.IGNORECASE):
                all_checked = True
                raw = re.sub(
                    r"\s*\((?:all\s+bag\s+slots\s+checked|end\s+of\s+bag\s+reached)\)\s*",
                    "",
                    raw,
                    flags=re.IGNORECASE,
                ).strip()
            if not raw or raw.lower() == "none":
                items: List[str] = []
            else:
                items = [
                    re.sub(r"\s*\(x\d+\)\s*$", "", part.strip(), flags=re.IGNORECASE)
                    for part in raw.split(",")
                    if part.strip()
                ]
                items = [it for it in items if it and it.lower() != "none"]
            last_result = (items, all_checked)
        return last_result

    @classmethod
    def extract_player_hp_from_text(cls, text: str) -> Optional[Tuple[int, int]]:
        last: Optional[Tuple[int, int]] = None
        for m in cls.PLAYER_HP_PATTERN.finditer(text or ""):
            g = m.groups()
            cur_s = g[0] or g[2] or g[4]
            max_s = g[1] or g[3] or g[5]
            if cur_s and max_s:
                cur_hp, max_hp = int(cur_s), int(max_s)
                if 0 <= cur_hp <= max_hp and 10 <= max_hp <= 999:
                    last = (cur_hp, max_hp)
        return last

    @classmethod
    def extract_player_status_from_text(cls, text: str) -> Optional[str]:
        if not text:
            return None
        last_tag: Optional[str] = None
        for m in cls.PLAYER_STATUS_TAG_PATTERN.finditer(text):
            val = m.group(1).upper()
            last_tag = "" if val == "HEALTHY" else val
        if last_tag is not None:
            return last_tag
        if re.search(
            r"\b(?:cured\s+of\s+(?:its\s+)?poison|poison\s+faded|healed\s+of\s+(?:its\s+)?(?:poison|burn|paralysis))\b",
            text,
            re.IGNORECASE,
        ):
            return ""
        if re.search(
            r"(?:^|[\n.!])\s*(?!Enemy\b|Wild\b|Foe\b)[A-Za-z0-9_.'♀♂-]+\s+(?:was\s+poisoned|’s\s+hurt\s+by\s+poison|'s\s+hurt\s+by\s+poison)\b",
            text,
            re.IGNORECASE,
        ):
            return "PSN"
        if re.search(
            r"(?:^|[\n.!])\s*(?!Enemy\b|Wild\b|Foe\b)[A-Za-z0-9_.'♀♂-]+\s+(?:was\s+burned|’s\s+hurt\s+by\s+(?:its\s+)?burn|'s\s+hurt\s+by\s+(?:its\s+)?burn)\b",
            text,
            re.IGNORECASE,
        ):
            return "BRN"
        if re.search(
            r"(?:^|[\n.!])\s*(?!Enemy\b|Wild\b|Foe\b)[A-Za-z0-9_.'♀♂-]+\s+(?:is\s+paralyzed|’s\s+fully\s+paralyzed|'s\s+fully\s+paralyzed)\b",
            text,
            re.IGNORECASE,
        ):
            return "PAR"
        return None

    @classmethod
    def extract_zero_pp_moves_from_text(cls, text: str) -> Optional[List[str]]:
        if not text:
            return None
        last_list: Optional[List[str]] = None
        for m in cls.ZERO_PP_TAG_PATTERN.finditer(text):
            raw = m.group(1).strip()
            if not raw or raw.lower() == "none":
                last_list = []
            else:
                last_list = [part.strip() for part in raw.split(",") if part.strip()]
        return last_list

    def _last_seen_player_level(self) -> Optional[int]:
        last_lvl: Optional[int] = None
        for step in self.report.steps:
            if isinstance(step, EnvironmentStepRecord):
                for frame in (step.frame_before, step.frame_after):
                    found_l = read_battle_hud_level(frame) or read_party_menu_level(frame)
                    if found_l is not None and (last_lvl is None or found_l >= last_lvl):
                        last_lvl = found_l
        if last_lvl is not None:
            return last_lvl
        if self._party_state is not None:
            for call in self.report.vlm_call_log:
                if isinstance(call, ExecutorVLMCallRecord) and call.tag in (self.STEP_TAG, self.MOVE_TAG):
                    found_l = self._party_state.extract_player_level_from_text(str(call.response))
                    if found_l is not None and (last_lvl is None or found_l >= last_lvl):
                        last_lvl = found_l
        return last_lvl

    def _last_seen_player_hp(self) -> Optional[Tuple[int, int]]:
        last_hud: Optional[Tuple[int, int]] = None
        for step in self.report.steps:
            if isinstance(step, EnvironmentStepRecord):
                for frame in (step.frame_before, step.frame_after):
                    found_hud = read_battle_hud_hp(frame) or read_party_menu_hp(frame)
                    if found_hud is not None:
                        last_hud = found_hud
        if last_hud is not None:
            return last_hud
        last: Optional[Tuple[int, int]] = None
        for call in self.report.vlm_call_log:
            if isinstance(call, ExecutorVLMCallRecord) and call.tag in (self.STEP_TAG, self.MOVE_TAG):
                found = self.extract_player_hp_from_text(str(call.response))
                if found is not None:
                    last = found
        return last

    def _last_seen_player_status(self, current_frame: Optional[np.ndarray] = None) -> Optional[str]:
        if current_frame is not None:
            st = read_party_menu_status(current_frame)
            if st is None:
                st = read_battle_hud_status(current_frame)
            if st is not None:
                return st
        last_hud_status: Optional[str] = None
        for step in self.report.steps:
            if isinstance(step, EnvironmentStepRecord):
                for frame in (step.frame_before, step.frame_after):
                    found_st = read_party_menu_status(frame)
                    if found_st is None:
                        found_st = read_battle_hud_status(frame)
                    if found_st is not None:
                        last_hud_status = found_st
        if last_hud_status is not None:
            return last_hud_status
        return self.extract_player_status_from_text("\n".join(self.dialogue_text))

    def _reconcile_move_with_reasoning(self, move: Optional[int], output: str) -> Optional[int]:
        if not self._move_names:
            return move
        up_out = (output or "").upper()
        if move is not None and move in self._move_names:
            chosen_name = self._move_names[move].upper().strip()
            if chosen_name and (chosen_name in up_out or chosen_name[:6] in up_out):
                return move
        matched_slots: List[Tuple[int, int]] = []
        for slot, m_name in self._move_names.items():
            if slot == move or slot in self._zero_pp_slots:
                continue
            token = m_name.upper().strip()
            if not token:
                continue
            pos = up_out.rfind(token)
            if pos == -1 and len(token) >= 6:
                pos = up_out.rfind(token[:6])
            if pos != -1:
                matched_slots.append((pos, slot))
        if matched_slots:
            matched_slots.sort()
            return matched_slots[-1][1]
        return move

    def _choose_move(self, output: str, *, prompt_if_missing: bool = False) -> Optional[int]:
        self._scan_move_menu_frame(self._current_frame())
        move = self._reconcile_move_with_reasoning(parse_int(output, "Move", lo=1, hi=4), output)
        for attempt_idx in range(self.MOVE_ATTEMPTS):
            if move is not None and move not in self._zero_pp_slots:
                return move
            if not (prompt_if_missing and attempt_idx == 0 and move is None):
                reason = "move has 0 PP left" if (move is not None and move in self._zero_pp_slots) else "no move number"
                self._record_invalid(output, reason=reason)
            frame = self._current_frame()
            self._scan_move_menu_frame(frame)
            prompt = (self.MOVE_PROMPT.replace("[INSTRUCTIONS]", self._instructions_block())
                      .replace("[HISTORY]", self._history_block())
                      .replace("[PREVIOUS]", output[:self.MAX_PREVIOUS_CHARS]))
            output = self._vlm_call(self.MOVE_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            move = self._reconcile_move_with_reasoning(parse_int(output, "Move", lo=1, hi=4), output)
        if move is not None and move in self._zero_pp_slots:
            for candidate in (1, 2, 3, 4):
                if candidate not in self._zero_pp_slots and (
                    not self._move_names or candidate in self._move_names
                ):
                    return candidate
        if move is None:
            self._record_invalid(output, reason="no move number")
        return move

    def _check_battle_defeat(self, state: AgentState) -> bool:
        if self.battle_lost or is_battle_defeat_text(self.dialogue_text):
            self.battle_lost = True
            return True
        if state == AgentState.FREE_ROAM and self._battle_place is not None:
            gmap = getattr(self._env, "_map", None)
            if gmap is not None and getattr(gmap, "current", None) and getattr(gmap, "pos", None) is not None:
                after_place = gmap.resolve(gmap.segment, gmap.pos)
                if (after_place[0] != self._battle_place[0]
                        and has_player_pokemon_fainted_text(self.dialogue_text)):
                    self.battle_lost = True
                    return True
        return False

    def _record_outcome(self, state) -> int:
        code = LEFT_BATTLE_CODES.get(state, BATTLE_ENDED)
        reason = {
            BATTLE_ENDED: "battle_ended",
            BATTLE_LEFT_TO_DIALOGUE: "left_battle_to_dialogue",
            BATTLE_LEFT_TO_MENU: "left_battle_to_menu",
        }[code]
        if self._check_battle_defeat(state):
            recorder = getattr(self._env, "record_defeat", None)
            if callable(recorder):
                recorder(place=self._battle_place)
            reason = "battle_lost"
        if self._pending_initial_ocr and not self._skip_intro_ocr and self.dialogue_crops:
            self._pending_initial_ocr = False
            init_texts = ocr(self.dialogue_crops, self._vlm, do_merge=True, parameters=self._parameters)
            self.dialogue_text = init_texts + list(self.dialogue_text)
            if is_battle_defeat_text(init_texts):
                self.battle_lost = True
        self.report.termination_reason = reason
        notes_lines = list(self.dialogue_text)
        last_lvl = self._last_seen_player_level()
        if last_lvl is not None:
            if self._party_state is not None:
                self._party_state.record_active_level(last_lvl)
            notes_lines.append(f"[Active Pokemon level seen in battle: {last_lvl}]")
        if reason == "battle_lost":
            if self._party_state is not None:
                self._party_state.record_defeat()
                self._party_state.record_full_heal()
        else:
            last_hp = self._last_seen_player_hp()
            if last_hp is not None:
                if self._party_state is not None:
                    self._party_state.record_battle_hp(last_hp)
                evolved = bool(
                    re.search(r"\bevolved\s+into\b", "\n".join(self.dialogue_text), re.IGNORECASE)
                    and not re.search(r"\bstopped\s+evolving\b", "\n".join(self.dialogue_text), re.IGNORECASE)
                )
                if evolved:
                    notes_lines.append(
                        f"[Party HP seen in battle: {last_hp[0]}/{last_hp[1]} HP "
                        f"(pre-evolution HUD; evolution increased current and max HP equally)]"
                    )
                else:
                    notes_lines.append(f"[Party HP seen in battle: {last_hp[0]}/{last_hp[1]} HP]")
            last_status = self._last_seen_player_status()
            if last_status is not None:
                if self._party_state is not None:
                    self._party_state.record_status(last_status)
                notes_lines.append(f"[Party status seen in battle: {last_status or 'healthy'}]")
            zero_pp_items = self._format_zero_pp_list()
            if zero_pp_items:
                if self._party_state is not None:
                    self._party_state.record_zero_pp(zero_pp_items)
                notes_lines.append(f"[Moves at 0 PP: {', '.join(zero_pp_items)}]")
        move_outcome: Optional[Tuple[str, str, str]] = None
        for step in self.report.steps:
            if isinstance(step, EnvironmentStepRecord):
                for frame in (step.frame_before, step.frame_after):
                    st = read_move_learning_dialogue_status(frame)
                    if st is None:
                        continue
                    s_kind, s_poke, s_move = st
                    if s_kind in ("learned", "knows"):
                        if (
                            move_outcome is not None
                            and move_outcome[0] in ("learned", "knows")
                            and move_outcome[1] == s_poke
                        ):
                            p_kind, _, p_move = move_outcome
                            b_kind = "learned" if "learned" in (p_kind, s_kind) else s_kind
                            if s_move.startswith(p_move):
                                move_outcome = (b_kind, s_poke, s_move)
                            elif p_move.startswith(s_move):
                                move_outcome = (b_kind, s_poke, p_move)
                            else:
                                move_outcome = (s_kind, s_poke, s_move)
                        else:
                            move_outcome = (s_kind, s_poke, s_move)
                        if self._party_state is not None:
                            self._party_state.record_learned_move(move_outcome[2], s_poke)
                    elif s_kind == "forgot" and self._party_state is not None:
                        self._party_state.record_forgotten_move(s_move, s_poke)
        if move_outcome is not None:
            m_kind, m_poke, m_move = move_outcome
            if m_kind == "learned":
                notes_lines.append(f"[Move taught in menu: {m_poke} learned {m_move}]")
            elif m_kind == "knows":
                notes_lines.append(f"[Move already known in menu: {m_poke} knows {m_move}]")
        if self._opened_bag_in_battle:
            all_checked = (
                self._saw_bag_end
                or (self._party_state is not None and self._party_state.bag_complete)
                or (
                    (self._bag_unavailable or self._healing_items_exhausted)
                    and not any(is_healing_item_name(it) for it in self._seen_bag_items)
                )
            )
            if self._party_state is not None and (self._bag_unavailable or self._healing_items_exhausted):
                if not any(is_healing_item_name(it) for it in self._seen_bag_items):
                    self._party_state.mark_healing_bag_empty()
            end_note = " (all bag slots checked)" if all_checked else ""
            items_str = ", ".join(self._seen_bag_items) if self._seen_bag_items else "none"
            notes_lines.append(f"[Bag items seen: {items_str}{end_note}]")
        self.report.notes = "\n".join(notes_lines)
        return code

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        self._exited_to_free_roam = False
        self.battle_lost = False
        self._battle_place = self._current_map_place()
        init_state = self._agent_state()
        init_frame = self._current_frame()
        if init_state == AgentState.IN_MENU and (is_pack_screen(init_frame) or is_party_menu_screen(init_frame)):
            if not is_overworld_pack_menu(init_frame):
                self._saw_battle_state = True
                if is_pack_screen(init_frame):
                    self._in_bag_menu = True
                elif is_party_menu_screen(init_frame):
                    self._in_pokemon_menu = True
        else:
            self._saw_battle_state = init_state == AgentState.IN_BATTLE
        if (
            init_state != AgentState.IN_BATTLE
            and not is_battle_screen(init_frame)
            and not self._is_in_battle_sub_screen(init_frame)
        ):
            return self._fail(BATTLE_NOT_IN_BATTLE, "not_in_battle",
                              f"The player is {init_state.name.lower()}, not in a battle.")

        if not is_battle_decision_screen(self._current_frame()) and not self._has_battle_menu_cursor(self._current_frame()):
            self._begin_scripted("advance through opening battle text")
            intro_records: List[EnvironmentStepRecord] = []
            presses, dismisses = self._advance_until_decision(intro_records)
            if self._last_terminated:
                return self._fail(BATTLE_TERMINATED, "terminated")
            if self._last_truncated:
                return self._fail(BATTLE_TRUNCATED, "truncated")
            if self._skip_intro_ocr and not self._exited_to_free_roam:
                self._pending_initial_ocr = False
                intro_crops = [c for c in captured_dialogue_crops(intro_records) if not is_battle_menu_crop(c)]
                drop_prefix_crops(intro_crops, self.dialogue_crops)
                texts = []
            else:
                texts = self._read_new_text(intro_records, new_turn=not bool(self.dialogue_crops))
            if presses or dismisses or texts:
                entry = "- battle intro" + self._format_auto_suffix(presses, dismisses).replace(", then ", ": ")
                if texts:
                    entry += "\n  text on screen: " + " / ".join(texts)
                self.history.append(entry)
            if self._exited_to_free_roam:
                return self._record_outcome(AgentState.FREE_ROAM)
            post_intro_state = self._agent_state()
            if post_intro_state == AgentState.IN_BATTLE:
                self._saw_battle_state = True
            elif post_intro_state == AgentState.IN_MENU:
                post_intro_frame = self._current_frame()
                menu_exit = self._non_battle_menu_outcome(post_intro_frame)
                if menu_exit is not None or (not self._saw_battle_state and not is_battle_screen(post_intro_frame)):
                    return self._record_outcome(menu_exit or AgentState.IN_MENU)

        consecutive_invalid = 0
        consecutive_failures = 0
        for step in range(self._max_steps):
            state = self._agent_state()
            if state == AgentState.IN_BATTLE:
                self._saw_battle_state = True
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

            frame = self._current_frame()
            if (state == AgentState.IN_DIALOGUE and not self._has_battle_menu_cursor(frame)) or (
                    not is_battle_decision_screen(frame)
                    and is_evolution_screen(frame)):
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
                dialogue_progressed = any(
                    not is_same_battle_frame(r.frame_before, r.frame_after) for r in turn_records
                )
                if (
                    self._agent_state() == AgentState.IN_DIALOGUE
                    and not self._has_battle_menu_cursor(self._current_frame())
                    and (not presses or not dialogue_progressed)
                ):
                    return self._record_outcome(AgentState.IN_DIALOGUE)
                continue

            if state == AgentState.IN_MENU or self._is_in_battle_choice_menu(frame):
                menu_exit = self._non_battle_menu_outcome(frame) if state == AgentState.IN_MENU else None
                if menu_exit is not None:
                    return self._record_outcome(menu_exit)
                if not self._has_battle_menu_cursor(frame):
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
                    menu_exit = self._non_battle_menu_outcome(frame)
                    if menu_exit is not None or (
                        not self._saw_battle_state
                        and not is_battle_screen(frame)
                        and not has_bottom_dialogue_box(frame)
                    ):
                        return self._record_outcome(menu_exit or AgentState.IN_MENU)
                    if not self._has_battle_menu_cursor(frame):
                        progressed = any(
                            not is_same_battle_frame(r.frame_before, r.frame_after) for r in turn_records
                        )
                        if not progressed:
                            wait_rec = self._take_action(WaitAction)
                            if self._last_terminated:
                                return self._fail(BATTLE_TERMINATED, "terminated")
                            if self._last_truncated:
                                return self._fail(BATTLE_TRUNCATED, "truncated")
                        continue

                if is_empty_battle_bag_menu(frame):
                    self._ensure_scripted("close empty battle bag menu")
                    record = self._take_battle_menu_press("back")
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    self._opened_bag_in_battle = True
                    self._saw_bag_end = True
                    self._seen_bag_items.clear()
                    self._bag_item_quantities.clear()
                    if self._party_state is not None:
                        self._party_state.observe_bag_window([], has_more=False, at_top=True, at_bottom=True)
                    self._bag_unavailable = True
                    self._in_bag_menu = False
                    self.history.append(
                        f"- step {step + 1}: bag is empty (only CANCEL); pressed back to return to the battle menu"
                    )
                    continue

                bag_before_info = self._scan_bag_menu_frame(frame)
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
                pre_cursor_row = bag_before_info[2] if bag_before_info is not None else None
                intercepted_cancel = False
                if (
                    press == "confirm"
                    and bag_before_info is not None
                    and pre_cursor_row == "CANCEL"
                    and any(
                        is_healing_item_name(it) or is_status_heal_item_name(it)
                        for it in bag_before_info[0]
                    )
                ):
                    press = "up"
                    intercepted_cancel = True
                if bag_before_info is not None:
                    if press in ("up", "down"):
                        self._bag_scrolled_in_opening = True
                    if self._bag_cursor_idx is not None:
                        n_vis = len(bag_before_info[0])
                        max_idx = n_vis if not bag_before_info[1] else max(0, n_vis - 1)
                        if press == "down":
                            self._bag_cursor_idx = min(max_idx, self._bag_cursor_idx + 1)
                        elif press == "up":
                            self._bag_cursor_idx = max(0, self._bag_cursor_idx - 1)
                was_party_selection = is_party_menu_selection_screen(frame)
                record = self._take_battle_menu_press(press)
                if self._last_terminated:
                    return self._fail(BATTLE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._fail(BATTLE_TRUNCATED, "truncated")
                after_press_frame = self._current_frame()
                bag_after_info = self._scan_bag_menu_frame(after_press_frame)
                if bag_before_info is not None and bag_after_info is not None:
                    if press == "down" and (
                        not bag_after_info[1]
                        or (bag_after_info[0] == bag_before_info[0] and bag_after_info[2] == bag_before_info[2])
                        or is_same_battle_frame(record.frame_before, record.frame_after)
                    ):
                        self._saw_bag_bottom = True
                        if self._party_state is not None:
                            self._party_state.saw_bottom = True
                        self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)
                    elif press == "up" and (
                        (bag_after_info[0] == bag_before_info[0] and bag_after_info[2] == bag_before_info[2])
                        or is_same_battle_frame(record.frame_before, record.frame_after)
                    ):
                        self._saw_bag_top = True
                        if self._party_state is not None:
                            self._party_state.saw_top = True
                        self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)
                if (
                    press == "confirm"
                    and was_party_selection
                    and is_party_menu_screen(after_press_frame)
                    and not has_menu_cursor(after_press_frame)
                ):
                    self._party_effect_pending = True
                turn_records = [record]
                entry = f"- step {step + 1}: pressed {press} in battle menu"
                if intercepted_cancel:
                    entry += " (cursor was on CANCEL, so pressed up instead of confirm to move onto an item)"
                elif press == "confirm" and pre_cursor_row and pre_cursor_row != "CANCEL":
                    self._selected_bag_item = pre_cursor_row
                    entry += f" (selected {pre_cursor_row})"
                auto_item_steps: List[str] = []
                if (
                    press == "confirm"
                    and self._in_bag_menu
                    and self._selected_bag_item
                    and record.action_success != -1
                ):
                    auto_item_steps = self._complete_in_battle_item_application(
                        self._selected_bag_item,
                        turn_records,
                    )
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    if auto_item_steps:
                        after_press_frame = self._current_frame()
                        entry += ", " + " and ".join(auto_item_steps)
                if record.action_success == -1:
                    entry += " (the game did not accept it)"
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0
                presses, dismisses = 0, 0
                if (
                    self._agent_state() != AgentState.IN_MENU
                    or not self._has_battle_menu_cursor(after_press_frame)
                ):
                    presses, dismisses = self._advance_until_decision(turn_records)
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    self._party_effect_pending = False
                    if self._in_bag_menu and (
                        self._agent_state() != AgentState.IN_MENU
                        or is_battle_command_menu(self._current_frame())
                    ):
                        self._in_bag_menu = False
                        bag_before = bag_before_info[:2] if bag_before_info is not None else read_bag_item_list(frame)
                        had_bag_items = bool(bag_before and bag_before[0])
                        if (presses == 0
                                and not auto_item_steps
                                and self._agent_state() == AgentState.IN_BATTLE
                                and is_battle_decision_screen(self._current_frame())):
                            if press == "confirm" and had_bag_items:
                                entry += (
                                    " (cursor was on CANCEL so confirm closed the bag without using an item; "
                                    "items are still available in the bag)"
                                )
                            elif any(is_healing_item_name(it) for it in self._seen_bag_items):
                                entry += " (closed the bag without using an item; healing items are still in the bag)"
                            else:
                                self._bag_unavailable = True
                                entry += " (closed the bag without using an item; do not choose bag again)"
                        elif presses > 0 or auto_item_steps:
                            self._consume_selected_bag_item(self._selected_bag_item)
                        used_menu_bag_item = self._selected_bag_item if (presses > 0 or auto_item_steps) else None
                        self._selected_bag_item = None
                    else:
                        used_menu_bag_item = None
                    if self._in_pokemon_menu and (
                        self._agent_state() != AgentState.IN_MENU
                        or is_battle_command_menu(self._current_frame())
                    ):
                        self._in_pokemon_menu = False
                else:
                    used_menu_bag_item = None
                entry += self._format_auto_suffix(presses, dismisses)
                texts = self._read_new_text(turn_records, new_turn=True)
                if texts:
                    self._update_opponent_tracking(frame=self._current_frame(), texts=texts)
                    entry += "\n  text on screen: " + " / ".join(texts)
                else:
                    self._update_opponent_tracking(frame=self._current_frame())
                if not self._in_bag_menu and (presses > 0 or auto_item_steps):
                    self._record_turn_hp_transition(
                        self._turn_start_hp,
                        turn_records,
                        texts,
                        used_heal_item=used_menu_bag_item,
                    )
                    self._turn_start_hp = None
                self.history.append(entry)
                if self._exited_to_free_roam:
                    return self._record_outcome(AgentState.FREE_ROAM)
                if self._agent_state() == AgentState.IN_MENU:
                    menu_exit = self._non_battle_menu_outcome(self._current_frame())
                    if menu_exit is not None:
                        return self._record_outcome(menu_exit)
                if consecutive_failures >= self.MAX_INVALID:
                    return self._fail(BATTLE_ACTION_FAILED, "actions_not_accepted", self._history_block())
                continue

            frame = self._current_frame()
            if is_no_pp_dialogue_frame(frame):
                self._ensure_scripted("dismiss no-PP prompt")
                dismiss_records: List[EnvironmentStepRecord] = []
                n_dismissed = self._dismiss_no_pp_dialogue(dismiss_records)
                if self._last_terminated:
                    return self._fail(BATTLE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._fail(BATTLE_TRUNCATED, "truncated")
                if n_dismissed:
                    self.history.append(
                        f"- step {step + 1}: dismissed 'No PP left' prompt ({n_dismissed} B press{'es' if n_dismissed > 1 else ''})"
                    )
                    frame = self._current_frame()
            self._scan_move_menu_frame(frame)
            pre_turn_hp = (
                read_battle_hud_hp(frame)
                or read_party_menu_hp(frame)
                or self._last_seen_player_hp()
            )
            move_menu_open = is_move_selection_screen(frame)
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
            overrode_futile_bag = False
            if option == "bag" and pre_turn_hp is not None:
                futility = self._get_bag_healing_futility(pre_turn_hp[0], pre_turn_hp[1])
                if futility is not None:
                    live_status = self._last_seen_player_status(frame)
                    wants_ball = bool(
                        re.search(
                            r"\b(?:catch|capture|poke\s*ball|poké\s*ball|great\s*ball|ultra\s*ball)\b",
                            self._task or "",
                            re.IGNORECASE,
                        )
                    )
                    has_status_cure = bool(
                        live_status
                        and any(is_status_heal_item_for_status(it, live_status) for it in self._seen_bag_items)
                    )
                    if not wants_ball and not has_status_cure:
                        option = "fight"
                        overrode_futile_bag = True

            turn_records = []
            presses, dismisses = 0, 0
            used_move_slot: Optional[int] = None
            used_heal_item: Optional[str] = None
            if move_menu_open and option == "fight":
                entry = f"- step {step + 1}: chose fight"
                move = self._choose_move(output, prompt_if_missing=overrode_futile_bag)
                if move is None:
                    entry += ", but no move number was given"
                else:
                    pre_pp = self._move_pp.get(move)
                    attack = self._take_action(PickAttackAction, option=move)
                    self._scan_record_move_frames(attack)
                    if pre_pp is None:
                        pre_pp = self._move_pp.get(move)
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    turn_records.append(attack)
                    move_label = f"move {move}" + (f" ({self._move_names[move]})" if move in self._move_names else "")
                    entry += f", used {move_label}"
                    if attack.action_success == 1:
                        self._mark_zero_pp_slot(move)
                        n_dismissed = self._dismiss_no_pp_dialogue(turn_records)
                        if self._last_terminated:
                            return self._fail(BATTLE_TERMINATED, "terminated")
                        if self._last_truncated:
                            return self._fail(BATTLE_TRUNCATED, "truncated")
                        entry += " (no PP left — do NOT choose this move again)"
                        if n_dismissed:
                            entry += f", dismissed prompt ({n_dismissed} B press{'es' if n_dismissed > 1 else ''})"
                        consecutive_failures = 0
                    elif attack.action_success == -1:
                        entry += " (the game did not accept it)"
                        consecutive_failures += 1
                    else:
                        if pre_pp is not None and pre_pp[0] == 1:
                            self._mark_zero_pp_slot(move)
                        elif move in self._move_pp and self._move_pp[move][0] > 1:
                            self._move_pp[move] = (self._move_pp[move][0] - 1, self._move_pp[move][1])
                        used_move_slot = move
                        consecutive_failures = 0
                        presses, dismisses = self._advance_until_decision(turn_records)
                        if self._last_terminated:
                            return self._fail(BATTLE_TERMINATED, "terminated")
                        if self._last_truncated:
                            return self._fail(BATTLE_TRUNCATED, "truncated")
            elif move_menu_open and option == "progress":
                entry = (
                    f"- step {step + 1}: chose progress "
                    "(the move selection menu is already open; choose fight and give a Move number 1-4)"
                )
            else:
                if move_menu_open and option in ("bag", "pokemon", "run"):
                    back_rec = self._take_action(MenuAction, menu_action="back")
                    if self._last_terminated:
                        return self._fail(BATTLE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._fail(BATTLE_TRUNCATED, "truncated")
                    turn_records.append(back_rec)
                if option == "progress" and (
                        self._agent_state() == AgentState.IN_DIALOGUE
                        or is_evolution_screen(self._current_frame())):
                    record = self._take_action(LowLevelAction, low_level_action=LowLevelActions.PRESS_BUTTON_A)
                elif option == "progress":
                    record = self._take_action(LowLevelAction, low_level_action=LowLevelActions.PRESS_BUTTON_B)
                else:
                    record = self._take_action(BattleMenuAction, option=option)
                    self._scan_record_move_frames(record)
                if self._last_terminated:
                    return self._fail(BATTLE_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._fail(BATTLE_TRUNCATED, "truncated")
                if option != "fight":
                    turn_records.append(record)
                entry = f"- step {step + 1}: chose {option}"
                if record.action_success in (1, 2):
                    entry += " (could not escape)"
                    consecutive_failures = 0
                elif record.action_success == -1:
                    if option == "progress" and not is_battle_decision_screen(self._current_frame()):
                        consecutive_failures = 0
                    else:
                        entry += " (the game did not accept it)"
                        consecutive_failures += 1
                else:
                    consecutive_failures = 0

                if option == "bag" and record.action_success != -1:
                    self._opened_bag_in_battle = True
                    if is_empty_battle_bag_menu(self._current_frame()):
                        back_rec = self._take_action(MenuAction, menu_action="back")
                        if self._last_terminated:
                            return self._fail(BATTLE_TERMINATED, "terminated")
                        if self._last_truncated:
                            return self._fail(BATTLE_TRUNCATED, "truncated")
                        self._saw_bag_end = True
                        self._seen_bag_items.clear()
                        self._bag_item_quantities.clear()
                        if self._party_state is not None:
                            self._party_state.observe_bag_window([], has_more=False, at_top=True, at_bottom=True)
                        self._bag_unavailable = True
                        self._in_bag_menu = False
                        entry += " (the bag is empty — only CANCEL — so closed it; do not choose bag again)"
                    elif self._agent_state() == AgentState.IN_MENU:
                        self._in_bag_menu = True
                        self._turn_start_hp = pre_turn_hp
                        self._bag_scrolled_in_opening = False
                        self._scan_bag_menu_frame(self._current_frame())
                        bag_note, b_presses, b_dismisses = self._auto_navigate_in_battle_bag(turn_records)
                        if self._last_terminated:
                            return self._fail(BATTLE_TERMINATED, "terminated")
                        if self._last_truncated:
                            return self._fail(BATTLE_TRUNCATED, "truncated")
                        if bag_note:
                            entry += bag_note
                            m_sel = re.search(r"\(selected\s+([A-Z0-9 .'-]+?)(?:,|\))", bag_note)
                            if m_sel:
                                used_heal_item = m_sel.group(1).strip()
                        presses += b_presses
                        dismisses += b_dismisses

                elif option == "pokemon" and record.action_success != -1 and self._agent_state() == AgentState.IN_MENU:
                    self._in_pokemon_menu = True
                if option == "fight":
                    attack = None
                    move = None
                    if record.action_success == 0:
                        move = self._choose_move(output, prompt_if_missing=overrode_futile_bag)
                        if move is None:
                            entry += ", but no move number was given"
                        else:
                            pre_pp = self._move_pp.get(move)
                            attack = self._take_action(PickAttackAction, option=move)
                            self._scan_record_move_frames(attack)
                            if pre_pp is None:
                                pre_pp = self._move_pp.get(move)
                    elif record.action_success == -1:
                        move = parse_int(output, "Move", lo=1, hi=4)
                        if move is not None and move in self._zero_pp_slots:
                            move = self._choose_move(output)
                        if move is not None:
                            pre_pp = self._move_pp.get(move)
                            attack = self._take_action(PickAttackAction, option=move)
                            self._scan_record_move_frames(attack)
                            if pre_pp is None:
                                pre_pp = self._move_pp.get(move)
                            if attack.action_success != -1:
                                consecutive_failures = 0
                                entry = f"- step {step + 1}: chose fight"
                    if attack is not None and move is not None:
                        if self._last_terminated:
                            return self._fail(BATTLE_TERMINATED, "terminated")
                        if self._last_truncated:
                            return self._fail(BATTLE_TRUNCATED, "truncated")
                        turn_records.append(attack)
                        move_label = f"move {move}" + (f" ({self._move_names[move]})" if move in self._move_names else "")
                        entry += f", used {move_label}"
                        if attack.action_success == 1:
                            self._mark_zero_pp_slot(move)
                            n_dismissed = self._dismiss_no_pp_dialogue(turn_records)
                            if self._last_terminated:
                                return self._fail(BATTLE_TERMINATED, "terminated")
                            if self._last_truncated:
                                return self._fail(BATTLE_TRUNCATED, "truncated")
                            entry += " (no PP left — do NOT choose this move again)"
                            if n_dismissed:
                                entry += f", dismissed prompt ({n_dismissed} B press{'es' if n_dismissed > 1 else ''})"
                        elif attack.action_success == -1:
                            entry += " (the game did not accept it)"
                        else:
                            if pre_pp is not None and pre_pp[0] == 1:
                                self._mark_zero_pp_slot(move)
                            elif move in self._move_pp and self._move_pp[move][0] > 1:
                                self._move_pp[move] = (self._move_pp[move][0] - 1, self._move_pp[move][1])
                            used_move_slot = move
                            presses, dismisses = self._advance_until_decision(turn_records)
                            if self._last_terminated:
                                return self._fail(BATTLE_TERMINATED, "terminated")
                            if self._last_truncated:
                                return self._fail(BATTLE_TRUNCATED, "truncated")
                elif option == "progress" or (option == "run" and record.action_success != -1):
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
            if option == "run" and record.action_success == 0 and self._exited_to_free_roam:
                flee_crops = [c for c in captured_dialogue_crops(turn_records) if not is_battle_menu_crop(c)]
                drop_prefix_crops(flee_crops, self.dialogue_crops)
                texts = ["Got away safely!"]
                self.dialogue_text.extend(texts)
            else:
                texts = self._read_new_text(turn_records, new_turn=(option != "progress"))
            if texts:
                eff_note = self._record_move_effectiveness(used_move_slot, texts)
                if eff_note:
                    entry += eff_note
                self._update_opponent_tracking(frame=self._current_frame(), texts=texts)
                entry += "\n  text on screen: " + " / ".join(texts)
            else:
                self._update_opponent_tracking(frame=self._current_frame())
            if not self._in_bag_menu and not self._in_pokemon_menu:
                self._record_turn_hp_transition(
                    pre_turn_hp,
                    turn_records,
                    texts,
                    used_heal_item=used_heal_item,
                )
                self._turn_start_hp = None
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
First decide whether the entire objective (including closing the Pack/Item and Start menus back to the overworld if asked) has already been achieved on screen. If it has not, answer Complete: no and choose the single button press that best moves towards it. On an item list screen after using a healing item or teaching a move on a party Pokemon (even if the item list cursor is invisible), do not press confirm on the item again—press back once to close the Pack/Item menu to the Start menu, and press back again to close the Start menu to the overworld. When teaching a move on the party screen, if asked whether to delete an older move to make room for the new move, select YES (confirm) and then press confirm on the move to forget.

The presses you can make:
- up, down, left, right: move the cursor
- confirm: press A to choose what the cursor is on
- back: press B to go back, or to close the current menu

Respond in exactly this format:
Reasoning: <what is on the screen, and what should be pressed next>
Complete: <yes or no>
Press: <one of up, down, left, right, confirm, back>
[STOP]"""

    MAX_MENU_DIALOGUE_SETTLE_STEPS = 20
    MAX_STALLED_MENU_DISMISSES = 4

    STATUS_HEAL_TASK_PATTERN = re.compile(
        r"\b(?:antidote|cure\s+poison|parlyz\s+heal|awakening|burn\s+heal|ice\s+heal|full\s+heal|status)\b",
        re.IGNORECASE,
    )
    STATUS_HEAL_ITEM_NAMES = (
        "ANTIDOTE",
        "FULL HEAL",
        "FULL RESTORE",
        "PARLYZ HEAL",
        "AWAKENING",
        "BURN HEAL",
        "ICE HEAL",
        "PSNCUREBERRY",
        "PRZCUREBERRY",
        "MINT BERRY",
        "BURNT BERRY",
        "ICE BERRY",
        "MIRACLEBERRY",
    )

    def __init__(
        self,
        env,
        task,
        max_steps: int = 25,
        is_start_menu: bool = False,
        party_state: Optional[PartyInventoryState] = None,
        **kwargs,
    ):
        self.is_start_menu = is_start_menu
        self._party_state: Optional[PartyInventoryState] = party_state
        self.presses: List[str] = []
        self.press_outcomes: List[str] = []
        self.dialogue_crops: List[Any] = []
        self.dialogue_text: str = ""
        self._seen_bag_items: List[str] = list(party_state.items) if party_state is not None else []
        self._bag_item_quantities: Dict[str, Optional[int]] = (
            dict(party_state.quantities) if party_state is not None else {}
        )
        self._opened_bag_in_menu: bool = False
        self._saw_bag_top: bool = False
        self._saw_bag_bottom: bool = False
        self._saw_bag_end: bool = False
        self._bag_scrolled_in_opening: bool = False
        self._bag_cursor_idx: Optional[int] = None
        self._pending_bag_item: Optional[str] = None
        self._active_menu_item: Optional[str] = None
        self._last_consumed_bag_item: Optional[str] = None
        self._confirmed_bag_items: set = set()
        self._last_auto_scrolled_confirmed_count: Optional[int] = None
        self._quantity_adjusted_in_box: bool = False
        self._completed_tm_hm_party_attempt: bool = False
        self._last_move_teaching_outcome: Optional[Tuple[str, str, str]] = None
        self._last_incompatible_tm: Optional[str] = None
        self._overworld_menu_session: bool = bool(is_start_menu)
        self._tm_hm_scrolled_in_opening: bool = False
        super().__init__(env, task, max_steps, is_start_menu=is_start_menu, **kwargs)

    def _effective_menu_state(self, frame: np.ndarray) -> AgentState:
        raw_state = self._agent_state()
        eff = effective_agent_state(raw_state, frame)
        if (
            raw_state == AgentState.IN_MENU
            and eff == AgentState.IN_BATTLE
            and (self._overworld_menu_session or is_overworld_menu_or_ui_screen(frame))
        ):
            return AgentState.IN_MENU
        if (
            eff == AgentState.IN_DIALOGUE
            and (
                self._overworld_menu_session
                or is_overworld_menu_or_ui_screen(frame)
                or is_tm_hm_startup_dialogue_frame(frame)
            )
            and (
                is_overworld_menu_or_ui_screen(frame)
                or is_tm_hm_startup_dialogue_frame(frame)
                or is_gen1_item_list_screen(frame)
                or is_item_use_subbox_screen(frame)
                or is_party_menu_screen(frame)
            )
        ):
            return AgentState.IN_MENU
        return eff

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _scan_bag_frame(self, frame: Optional[np.ndarray]) -> Optional[Tuple[List[str], bool]]:
        if frame is None:
            return None
        info = read_bag_item_list(frame)
        if info is None:
            return None
        self._opened_bag_in_menu = True
        items, has_more = info
        quantities = dict(read_bag_item_quantities(frame) or ())
        if self._last_consumed_bag_item:
            if self._party_state is not None:
                self._party_state.consume_item(self._last_consumed_bag_item, count=0, visible_after=items)
                self._seen_bag_items = list(self._party_state.items)
                self._bag_item_quantities = dict(self._party_state.quantities)
            elif items and self._last_consumed_bag_item not in items and self._last_consumed_bag_item in self._seen_bag_items:
                self._seen_bag_items.remove(self._last_consumed_bag_item)
                self._bag_item_quantities.pop(self._last_consumed_bag_item, None)
            self._last_consumed_bag_item = None
        cursor_idx = read_bag_cursor_row_index(frame, tracked_index=self._bag_cursor_idx)
        cursor_row = read_bag_cursor_row(frame, tracked_index=self._bag_cursor_idx)
        if cursor_row == "CANCEL":
            self._bag_cursor_idx = len(items)
        elif cursor_row in items:
            self._bag_cursor_idx = items.index(cursor_row)
        elif cursor_idx is not None:
            self._bag_cursor_idx = cursor_idx
        max_items_with_cancel = 3 if is_gen1_item_list_screen(frame) else 4
        if not has_more:
            self._saw_bag_bottom = True
            if len(items) < max_items_with_cancel:
                self._saw_bag_top = True
        if (
            not self._bag_scrolled_in_opening
            and cursor_idx == 0
            and bool(items)
            and (
                not self._seen_bag_items
                or items[0] not in self._seen_bag_items
                or self._seen_bag_items.index(items[0]) == 0
            )
        ):
            self._saw_bag_top = True
        if self._party_state is not None:
            self._seen_bag_items = self._party_state.observe_bag_window(
                items,
                quantities=quantities,
                has_more=has_more,
                at_top=self._saw_bag_top,
                at_bottom=self._saw_bag_bottom,
                max_items_with_cancel=max_items_with_cancel,
            )
            self._bag_item_quantities = dict(self._party_state.quantities)
        else:
            self._seen_bag_items = CombatExecutor._merge_visible_bag_items(self._seen_bag_items, items)
            for name in items:
                if name in quantities and quantities[name] is not None:
                    self._bag_item_quantities[name] = quantities[name]
        self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)
        return info

    def _task_wants_status_heal(self) -> bool:
        return bool(self.STATUS_HEAL_TASK_PATTERN.search(self._task or ""))

    def _task_wants_move_teaching(self) -> bool:
        raw = self._task or ""
        if not re.search(r"\b(?:teach|learn)\b", raw, re.IGNORECASE):
            return False
        return bool(
            re.search(
                r"\b(?:hm\s*\d*|tm\s*\d*|tm\s*pocket|pack|bag|item|move|cut|flash|surf|strength|fly|whirlpool|waterfall|rock\s+smash|headbutt)\b",
                raw,
                re.IGNORECASE,
            )
        )

    def _scan_move_learning_frame(self, frame: Optional[np.ndarray]) -> Optional[Tuple[str, str, str]]:
        status_info = read_move_learning_dialogue_status(frame)
        if status_info is None:
            return None
        kind, pokemon, move = status_info
        if kind in ("learned", "knows"):
            if (
                self._last_move_teaching_outcome is not None
                and self._last_move_teaching_outcome[0] in ("learned", "knows")
                and self._last_move_teaching_outcome[1] == pokemon
            ):
                prev_kind, _, prev_move = self._last_move_teaching_outcome
                best_kind = "learned" if "learned" in (prev_kind, kind) else kind
                if move.startswith(prev_move):
                    self._last_move_teaching_outcome = (best_kind, pokemon, move)
                elif prev_move.startswith(move):
                    self._last_move_teaching_outcome = (best_kind, pokemon, prev_move)
                else:
                    self._last_move_teaching_outcome = (kind, pokemon, move)
            else:
                self._last_move_teaching_outcome = (kind, pokemon, move)
            self._completed_tm_hm_party_attempt = True
            if self._party_state is not None:
                self._party_state.record_learned_move(self._last_move_teaching_outcome[2], pokemon)
        elif kind == "forgot":
            if self._party_state is not None:
                self._party_state.record_forgotten_move(move, pokemon)
        elif kind == "did_not_learn":
            if self._last_move_teaching_outcome is None or self._last_move_teaching_outcome[0] == "did_not_learn":
                self._last_move_teaching_outcome = (kind, pokemon, move)
            self._completed_tm_hm_party_attempt = False
        return self._last_move_teaching_outcome

    def _task_bag_clause(self) -> str:
        raw = self._task or ""
        return re.sub(
            r"\b(?:and\s+)?(?:then\s+)?(?:return\s+to\s+[^,;.]+\s+and\s+)?(?:select\s+buy\s+to\s+)?(?:buy|purchase)\b[^,;.]*",
            "",
            raw,
            flags=re.IGNORECASE,
        )

    def _bag_target_matches(self, item_name: Optional[str]) -> bool:
        if not item_name or item_name == "CANCEL" or item_name in self._confirmed_bag_items:
            return False
        up = item_name.strip().upper()
        if len(up) < 2:
            return False
        clause_up = self._task_bag_clause().upper()
        if re.search(r"\b" + re.escape(up) + r"S?\b", clause_up):
            return True
        if self._task_wants_status_heal() and is_status_heal_item_name(up):
            return True
        if (
            re.search(r"\b(?:heal|restore\s+hp|potion)\b", clause_up, re.IGNORECASE)
            and not re.search(r"\bhp\s+up\b", clause_up, re.IGNORECASE)
            and is_healing_item_name(up)
        ):
            return True
        return False

    def _seen_status_heal_item(self) -> bool:
        for item in self._seen_bag_items:
            if is_status_heal_item_name(item):
                return True
        return False

    def _target_quantity_for_item(self, active_item: Optional[str] = None) -> Optional[int]:
        raw = self._task or ""
        if not raw:
            return None
        norm_task = raw.upper().replace("É", "E")
        items_to_try: List[str] = []
        if active_item:
            norm_item = active_item.strip().upper().replace("É", "E")
            if norm_item and norm_item != "CANCEL":
                items_to_try.append(norm_item)
                if norm_item == "POK":
                    items_to_try.append("POKE BALL")
        for item in items_to_try:
            esc = re.escape(item)
            m_to = re.search(
                rf"\b{esc}S?\b[^,;.]{{0,40}}?\b(?:FROM\s+X?\d{{1,2}}\s+)?TO\s+X?(\d{{1,2}})\b",
                norm_task,
            )
            if m_to:
                val = int(m_to.group(1))
                if 1 <= val <= 99:
                    return val
            m_before = re.search(
                rf"\b(\d{{1,2}})(?:\s*(?:TO|-)\s*\d{{1,2}})?\s+{esc}S?\b",
                norm_task,
            )
            if m_before:
                val = int(m_before.group(1))
                if 1 <= val <= 99:
                    return val
            m_after = re.search(
                rf"\b{esc}S?\s*(?:X|\()\s*(\d{{1,2}})\b",
                norm_task,
            )
            if m_after:
                val = int(m_after.group(1))
                if 1 <= val <= 99:
                    return val
        m_to_generic = re.search(
            r"\b(?:FROM\s+X?\d{1,2}\s+TO|QUANTITY\s+TO(?:\s+THE\s+MAXIMUM[^0-9]{0,30})?)\s*\(?X?(\d{1,2})\)?\b",
            norm_task,
        )
        if m_to_generic:
            val = int(m_to_generic.group(1))
            if 1 <= val <= 99:
                return val
        all_qty = re.findall(
            r"\b(?:BUY|PURCHASE|SELL|TOSS)\s+(?:UP\s+TO\s+)?(\d{1,2})(?:\s*(?:TO|-)\s*\d{1,2})?\s+[A-Z0-9][A-Z0-9 ]*\b",
            norm_task,
        )
        if len(all_qty) == 1 and not active_item:
            val = int(all_qty[0])
            if 1 <= val <= 99:
                return val
        return None

    def _auto_adjust_quantity_selector(self) -> None:
        frame = self._get_state()["core"]["current_frame"]
        if not has_quantity_selector_box(frame):
            self._quantity_adjusted_in_box = False
            return
        if self._quantity_adjusted_in_box:
            return
        cur_val = read_quantity_selector_value(frame)
        if cur_val is None:
            return
        target_val = self._target_quantity_for_item(self._active_menu_item)
        self._quantity_adjusted_in_box = True
        if target_val is None or target_val == cur_val:
            return
        if self._current_call is None:
            self._begin_scripted(
                f"adjust quantity selector from x{cur_val:02d} to x{target_val:02d}"
            )
        start_val = cur_val
        steps_taken = 0
        last_dir = "up"
        for _ in range(50):
            if self._last_terminated or self._last_truncated:
                return
            cur_frame = self._get_state()["core"]["current_frame"]
            if not has_quantity_selector_box(cur_frame):
                break
            cur_val = read_quantity_selector_value(cur_frame)
            if cur_val is None or cur_val == target_val:
                break
            up_dist = (target_val - cur_val) % 99
            down_dist = (cur_val - target_val) % 99
            step_dir = "up" if up_dist <= down_dist else "down"
            last_dir = step_dir
            rec = self._take_action(MenuAction, menu_action=step_dir)
            drop_prefix_crops(captured_dialogue_crops([rec]), self.dialogue_crops)
            steps_taken += 1
            if self._last_terminated or self._last_truncated:
                return
            after_val = read_quantity_selector_value(self._get_state()["core"]["current_frame"])
            if after_val == cur_val:
                break
            if after_val == target_val:
                cur_val = after_val
                break
        final_val = read_quantity_selector_value(self._get_state()["core"]["current_frame"])
        if steps_taken > 0 and final_val is not None:
            item_label = f" for {self._active_menu_item}" if self._active_menu_item else ""
            if self.press_outcomes:
                self.press_outcomes[-1] = (
                    f"{self.press_outcomes[-1]} (adjusted quantity{item_label} from x{start_val:02d} to x{final_val:02d} — press confirm to accept x{final_val:02d})"
                )
            else:
                self.presses.append(last_dir)
                self.press_outcomes.append(
                    f"{last_dir} (adjusted quantity{item_label} from x{start_val:02d} to x{final_val:02d} — press confirm to accept x{final_val:02d})"
                )

    def _task_allows_party_item_use(self, item_name: Optional[str]) -> bool:
        if not item_name or not is_party_consumable_item_name(item_name):
            return False
        clause = self._task_bag_clause()
        if re.search(r"\b(?:toss|discard|drop|give|hold)\b", clause, re.IGNORECASE):
            return False
        if re.search(r"\bsell\b", clause, re.IGNORECASE) and not re.search(
            r"\b(?:use|heal|restore|apply)\b", clause, re.IGNORECASE
        ):
            return False
        return True

    def _party_slot_matches_task(
        self,
        selected_item: str,
        members: List[Tuple[int, str, Optional[Tuple[int, int]]]],
        cursor_slot: Optional[int],
    ) -> bool:
        if not members:
            return True
        if len(members) == 1 and (cursor_slot is None or cursor_slot == 0):
            return True
        slot_idx = 0 if cursor_slot is None else cursor_slot
        if slot_idx < 0 or slot_idx >= len(members):
            return False
        _, cur_name, cur_hp = members[slot_idx]
        clause_up = self._task_bag_clause().upper()
        named_members = [
            (idx, name, hp)
            for idx, name, hp in members
            if name and len(name) >= 3 and name.upper() in clause_up
        ]
        if named_members:
            return any(idx == slot_idx for idx, _, _ in named_members)
        if slot_idx == 0:
            if is_healing_item_name(selected_item):
                return cur_hp is None or cur_hp[0] < cur_hp[1]
            return True
        return False

    def _complete_menu_party_item_application(self, selected_item: Optional[str]) -> List[str]:
        auto_steps: List[str] = []
        if not selected_item or selected_item == "CANCEL":
            return auto_steps
        if not self._task_allows_party_item_use(selected_item):
            return auto_steps
        cur_frame = self._get_state()["core"]["current_frame"]
        if is_item_use_subbox_screen(cur_frame):
            if self._current_call is None:
                self._begin_scripted(f"confirm USE on {selected_item}")
            use_rec = self._take_action(MenuAction, menu_action="confirm")
            drop_prefix_crops(captured_dialogue_crops([use_rec]), self.dialogue_crops)
            auto_steps.append("confirmed USE")
            if self._last_terminated or self._last_truncated:
                return auto_steps
            cur_frame = self._get_state()["core"]["current_frame"]
        if is_party_menu_selection_screen(cur_frame):
            members = read_party_menu_members(cur_frame)
            cursor_slot = read_party_menu_cursor_slot(cur_frame)
            if self._party_slot_matches_task(selected_item, members, cursor_slot):
                if self._current_call is None:
                    self._begin_scripted(f"apply {selected_item} to party Pokemon")
                target_rec = self._take_action(MenuAction, menu_action="confirm")
                drop_prefix_crops(captured_dialogue_crops([target_rec]), self.dialogue_crops)
                target_name = (
                    members[cursor_slot or 0][1]
                    if members and (cursor_slot or 0) < len(members)
                    else "party Pokemon"
                )
                auto_steps.append(f"confirmed {target_name} on party screen")
                if self._last_terminated or self._last_truncated:
                    return auto_steps
                after_target = self._get_state()["core"]["current_frame"]
                if self._should_settle_menu_dialogue(cur_frame, after_target, "confirm"):
                    self._settle_menu_dialogue()
                    auto_steps.append("settled effect dialogue")
        return auto_steps

    def _task_allows_tm_hm_use(self) -> bool:
        clause = self._task_bag_clause()
        return not bool(
            re.search(r"\b(?:toss|discard|drop|give|hold|sell|deposit|store)\b", clause, re.IGNORECASE)
        )

    def _task_wants_tm_hm_pocket(self) -> bool:
        raw = self._task or ""
        return bool(
            re.search(
                r"\b(?:tm\s*pocket|tm/hm|hm\s*0?\d|tm\s*\d{2})\b",
                raw,
                re.IGNORECASE,
            )
        )

    NON_DAMAGING_REPLACEABLE_MOVES = (
        "GROWL",
        "LEER",
        "TAIL WHIP",
        "STRING SHOT",
        "DEFENSE CURL",
        "HARDEN",
        "WITHDRAW",
        "SPLASH",
        "TELEPORT",
        "SAND-ATTACK",
        "SAND ATTACK",
        "SMOKESCREEN",
        "SMOKESCR",
        "KINESIS",
        "FLASH",
        "SWEET SCENT",
        "ROAR",
        "WHIRLWIND",
    )
    WEAK_EARLY_NORMAL_MOVES = (
        "TACKLE",
        "SCRATCH",
        "POUND",
    )
    PROTECTED_HM_MOVES = (
        "CUT",
        "FLY",
        "SURF",
        "STRENGTH",
        "WATERFALL",
        "WHIRLPOOL",
    )

    def _select_move_to_forget(self, moves: List[str]) -> Optional[Tuple[int, str]]:
        if not moves:
            return None
        up_task = (self._task or "").upper()
        keep_clause_moves = set()
        for m_keep in re.finditer(
            r"\b(?:KEEP|KEEPING|DO\s+NOT\s+(?:FORGET|OVERWRITE|REPLACE|DELETE)|NOT\s+(?:FORGET|OVERWRITE|REPLACE|DELETE))\b([^.;()]*)",
            up_task,
        ):
            seg = m_keep.group(1)
            for mv in moves:
                if mv and len(mv) >= 3 and re.search(r"\b" + re.escape(mv) + r"\b", seg):
                    keep_clause_moves.add(mv)
        replace_clause_moves: List[Tuple[int, str]] = []
        for m_rep in re.finditer(
            r"\b(?:REPLACE|REPLACING|OVERWRITE|OVERWRITING|FORGET|FORGETTING|DELETE|DELETING)\b([^.;]*)",
            up_task,
        ):
            seg = m_rep.group(1)
            for idx, mv in enumerate(moves):
                if (
                    mv
                    and len(mv) >= 3
                    and mv not in keep_clause_moves
                    and re.search(r"\b" + re.escape(mv) + r"\b", seg)
                ):
                    replace_clause_moves.append((idx, mv))
        if replace_clause_moves:
            return replace_clause_moves[0]
        for idx, mv in enumerate(moves):
            if mv in self.NON_DAMAGING_REPLACEABLE_MOVES and mv not in keep_clause_moves:
                return (idx, mv)
        for idx, mv in enumerate(moves):
            if (
                mv
                and len(mv) >= 3
                and mv not in keep_clause_moves
                and mv not in self.PROTECTED_HM_MOVES
                and re.search(r"\b" + re.escape(mv) + r"\b", up_task)
            ):
                return (idx, mv)
        for idx, mv in enumerate(moves):
            if mv in self.WEAK_EARLY_NORMAL_MOVES and mv not in keep_clause_moves:
                return (idx, mv)
        for idx, mv in enumerate(moves):
            if mv not in self.PROTECTED_HM_MOVES and mv not in keep_clause_moves:
                return (idx, mv)
        return (0, moves[0])

    def _auto_forget_move_on_list(self) -> Optional[str]:
        cur_frame = self._get_state()["core"]["current_frame"]
        forget_info = read_party_move_forget_list(cur_frame)
        if forget_info is None:
            return None
        moves, _ = forget_info
        chosen = self._select_move_to_forget(moves)
        if chosen is None:
            return None
        target_idx, target_move = chosen
        if self._current_call is None:
            self._begin_scripted(f"select move {target_move} to forget")
        for _ in range(6):
            if self._last_terminated or self._last_truncated:
                return None
            cur_frame = self._get_state()["core"]["current_frame"]
            cur_info = read_party_move_forget_list(cur_frame)
            if cur_info is None:
                break
            cur_moves, cur_hi = cur_info
            if cur_hi == target_move:
                break
            if cur_hi in cur_moves and target_move in cur_moves:
                cur_i = cur_moves.index(cur_hi)
                tgt_i = cur_moves.index(target_move)
                step_dir = "down" if tgt_i > cur_i else "up"
            elif target_idx > 0:
                step_dir = "down"
            else:
                break
            rec = self._take_action(MenuAction, menu_action=step_dir)
            drop_prefix_crops(captured_dialogue_crops([rec]), self.dialogue_crops)
        if self._last_terminated or self._last_truncated:
            return None
        confirm_rec = self._take_action(MenuAction, menu_action="confirm")
        drop_prefix_crops(captured_dialogue_crops([confirm_rec]), self.dialogue_crops)
        self._scan_move_learning_frame(confirm_rec.frame_before)
        self._scan_move_learning_frame(confirm_rec.frame_after)
        return target_move

    def _advance_party_move_learning_and_forget(
        self,
        pokemon_name: str,
        tm_label: Optional[str] = None,
    ) -> Optional[str]:
        forgotten_move: Optional[str] = None
        for _ in range(36):
            if self._last_terminated or self._last_truncated:
                return forgotten_move
            cur_frame = self._get_state()["core"]["current_frame"]
            self._scan_move_learning_frame(cur_frame)
            if (
                is_tm_hm_pack_pocket(cur_frame)
                or is_gen1_item_list_screen(cur_frame)
                or is_start_menu_screen(cur_frame)
                or self._effective_menu_state(cur_frame) != AgentState.IN_MENU
            ):
                break
            forget_info = read_party_move_forget_list(cur_frame)
            if forget_info is not None:
                _, cur_hi = forget_info
                if cur_hi is None:
                    wait_rec = self._take_action(WaitAction)
                    drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                    continue
                forgot = self._auto_forget_move_on_list()
                if forgot:
                    forgotten_move = forgot
                continue
            if is_party_menu_screen(cur_frame) and has_party_subbox(cur_frame):
                d_l1, d_l2 = read_dialogue_box_lines(cur_frame)
                d_low = f"{d_l1} {d_l2}".lower()
                if not has_menu_cursor(cur_frame):
                    wait_rec = self._take_action(WaitAction)
                    drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                    continue
                if "stop " in d_low or "abandon " in d_low or "give up" in d_low:
                    no_rec = self._take_action(MenuAction, menu_action="back")
                    drop_prefix_crops(captured_dialogue_crops([no_rec]), self.dialogue_crops)
                    continue
                yes_rec = self._take_action(MenuAction, menu_action="confirm")
                drop_prefix_crops(captured_dialogue_crops([yes_rec]), self.dialogue_crops)
                continue
            if is_party_menu_selection_screen(cur_frame) and not has_bottom_dialogue_box(cur_frame):
                break
            if not has_bottom_dialogue_box(cur_frame) and not is_party_menu_screen(cur_frame):
                wait_rec = self._take_action(WaitAction)
                drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                continue
            adv_rec = self._take_action(MenuAction, menu_action="confirm")
            drop_prefix_crops(captured_dialogue_crops([adv_rec]), self.dialogue_crops)
            self._scan_move_learning_frame(adv_rec.frame_before)
            self._scan_move_learning_frame(adv_rec.frame_after)
            if is_same_battle_frame(adv_rec.frame_before, adv_rec.frame_after):
                wait_rec = self._take_action(WaitAction)
                drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                self._scan_move_learning_frame(wait_rec.frame_after)
        cur_after = self._get_state()["core"]["current_frame"]
        self._scan_move_learning_frame(cur_after)
        if (
            self._last_move_teaching_outcome is not None
            and self._last_move_teaching_outcome[0] in ("learned", "knows")
        ) or (
            forgotten_move is not None
            and not is_party_menu_screen(cur_after)
        ):
            self._completed_tm_hm_party_attempt = True
            if forgotten_move and self._party_state is not None:
                self._party_state.record_forgotten_move(forgotten_move, pokemon_name)
            if tm_label:
                clean_tm = re.sub(r"\s*\(X\d+\)\s*$", "", tm_label.strip().upper())
                if clean_tm.startswith("TM"):
                    self._consume_confirmed_bag_item(clean_tm)
                if (
                    self._party_state is not None
                    and not self._party_state.learned_move_since_defeat
                    and (
                        self._last_move_teaching_outcome is None
                        or self._last_move_teaching_outcome[0] not in ("learned", "knows")
                    )
                ):
                    self._party_state.record_learned_move(clean_tm, pokemon_name)
                    self._last_move_teaching_outcome = ("learned", pokemon_name, clean_tm)
        return forgotten_move

    def _match_tm_hm_target_row(self, rows: Optional[List[str]]) -> Optional[Tuple[int, str]]:
        if not rows or not self._task:
            return None
        up_task = self._task.upper()
        target_codes = set()
        for m in re.finditer(r"\bHM\s*0?(\d)\b", up_task):
            target_codes.add(f"H{m.group(1)}")
            target_codes.add(f"HM0{m.group(1)}")
            target_codes.add(f"HM{m.group(1)}")
        for m in re.finditer(r"\bH(\d)\b", up_task):
            target_codes.add(f"H{m.group(1)}")
            target_codes.add(f"HM0{m.group(1)}")
        for m in re.finditer(r"\bTM\s*(\d{2})\b", up_task):
            target_codes.add(m.group(1))
            target_codes.add(f"TM{m.group(1)}")
        tried_tms = (
            set(self._party_state.tried_or_incompatible_tms)
            if self._party_state is not None
            else set()
        )
        for idx, row in enumerate(rows):
            if row == "CANCEL":
                continue
            clean = re.sub(r"\s*\(X\d+\)\s*$", "", row.strip().upper())
            if clean in tried_tms or clean in self._confirmed_bag_items:
                continue
            parts = clean.split(None, 1)
            code = parts[0] if parts else ""
            move_name = parts[1] if len(parts) > 1 else clean
            if code in target_codes or clean in target_codes:
                return (idx, clean)
            if len(move_name) >= 3 and re.search(r"\b" + re.escape(move_name) + r"\b", up_task):
                return (idx, clean)
        for idx, row in enumerate(rows):
            if row == "CANCEL":
                continue
            clean = re.sub(r"\s*\(X\d+\)\s*$", "", row.strip().upper())
            if clean in tried_tms or clean in self._confirmed_bag_items:
                continue
            if is_tm_hm_item_name(clean):
                return (idx, clean)
        return None

    def _advance_tm_hm_use_to_party_screen(self) -> bool:
        if self._current_call is None:
            self._begin_scripted("advance TM/HM startup prompts to party selection screen")
        for _ in range(20):
            if self._last_terminated or self._last_truncated:
                return False
            cur_frame = self._get_state()["core"]["current_frame"]
            if is_party_menu_screen(cur_frame):
                if not is_party_menu_selection_screen(cur_frame):
                    wait_rec = self._take_action(WaitAction)
                    drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                return True
            if self._effective_menu_state(cur_frame) != AgentState.IN_MENU:
                break
            if (
                not is_tm_hm_pack_pocket(cur_frame)
                and not is_tm_hm_startup_dialogue_frame(cur_frame)
                and not is_gen1_item_list_screen(cur_frame)
                and not has_bottom_dialogue_box(cur_frame)
            ):
                wait_rec = self._take_action(WaitAction)
                drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                continue
            rec = self._take_action(MenuAction, menu_action="confirm")
            drop_prefix_crops(captured_dialogue_crops([rec]), self.dialogue_crops)
        return is_party_menu_screen(self._get_state()["core"]["current_frame"])

    def _auto_select_able_party_member_and_advance(self, tm_label: Optional[str] = None) -> Optional[str]:
        cur_frame = self._get_state()["core"]["current_frame"]
        if not is_party_menu_screen(cur_frame) or has_party_subbox(cur_frame):
            return None
        compat_list = read_party_menu_compatibility(cur_frame)
        if not compat_list:
            return None
        able_slots = [(idx, name) for idx, name, status in compat_list if status == "ABLE"]
        if not able_slots:
            return None
        up_task = (self._task or "").upper()
        named_able = [
            (idx, name)
            for idx, name in able_slots
            if name and len(name) >= 3 and name.upper() in up_task
        ]
        target_slot, target_name = named_able[0] if named_able else able_slots[0]
        if len(able_slots) > 1 and not named_able:
            cursor_slot = read_party_menu_cursor_slot(cur_frame)
            if cursor_slot not in [idx for idx, _ in able_slots]:
                return None
            target_slot = cursor_slot
            target_name = dict(able_slots)[target_slot]
        if self._current_call is None:
            self._begin_scripted(f"select ABLE party Pokemon {target_name}")
        for _ in range(6):
            if self._last_terminated or self._last_truncated:
                return None
            cur_frame = self._get_state()["core"]["current_frame"]
            if not is_party_menu_screen(cur_frame) or has_party_subbox(cur_frame):
                return None
            cursor_slot = read_party_menu_cursor_slot(cur_frame)
            if cursor_slot is None or cursor_slot == target_slot:
                break
            step_dir = "down" if target_slot > cursor_slot else "up"
            rec = self._take_action(MenuAction, menu_action=step_dir)
            drop_prefix_crops(captured_dialogue_crops([rec]), self.dialogue_crops)
        if self._last_terminated or self._last_truncated:
            return None
        confirm_rec = self._take_action(MenuAction, menu_action="confirm")
        drop_prefix_crops(captured_dialogue_crops([confirm_rec]), self.dialogue_crops)
        if self._last_terminated or self._last_truncated:
            return target_name
        if self._task_wants_move_teaching():
            self._advance_party_move_learning_and_forget(
                target_name,
                tm_label=tm_label or self._active_menu_item or self._pending_bag_item,
            )
        else:
            after_confirm = self._get_state()["core"]["current_frame"]
            if self._should_settle_menu_dialogue(cur_frame, after_confirm, "confirm"):
                self._settle_menu_dialogue()
        return target_name

    def _handle_tm_hm_party_screen(self, target_label: str) -> bool:
        cur_party_fr = self._get_state()["core"]["current_frame"]
        compat_list = read_party_menu_compatibility(cur_party_fr)
        able_slots = [(idx, name) for idx, name, st in compat_list if st == "ABLE"]
        learned_slots = [(idx, name) for idx, name, st in compat_list if st == "LEARNED"]
        if able_slots:
            chosen_poke = self._auto_select_able_party_member_and_advance(tm_label=target_label)
            if self._last_terminated or self._last_truncated:
                return True
            if chosen_poke:
                cur_after = self._get_state()["core"]["current_frame"]
                self._scan_move_learning_frame(cur_after)
                if read_party_move_forget_list(cur_after) is not None:
                    self.press_outcomes[-1] = (
                        f"confirm (selected {target_label}, confirmed USE and YES, selected {chosen_poke}, and confirmed YES to delete an older move — choose which move to forget)"
                    )
                else:
                    self.press_outcomes[-1] = (
                        f"confirm (selected {target_label}, confirmed USE and YES, and taught the move to {chosen_poke})"
                    )
                return True
            self.press_outcomes[-1] = (
                f"confirm (selected {target_label} and confirmed USE and YES to open the party selection screen)"
            )
            return True
        if compat_list and not able_slots:
            clean_tm = re.sub(r"\s*\(X\d+\)\s*$", "", (target_label or "").strip().upper())
            if learned_slots:
                _, l_poke = learned_slots[0]
                if self._party_state is not None and clean_tm:
                    self._party_state.record_tried_or_incompatible_tm(clean_tm)
                back_rec = self._take_action(MenuAction, menu_action="back")
                drop_prefix_crops(captured_dialogue_crops([back_rec]), self.dialogue_crops)
                self._completed_tm_hm_party_attempt = True
                self.press_outcomes[-1] = (
                    f"confirm (opened party screen: {l_poke} already knows {clean_tm}; backed out to item menu)"
                )
            else:
                if self._party_state is not None and clean_tm:
                    self._party_state.record_tried_or_incompatible_tm(clean_tm)
                self._last_incompatible_tm = clean_tm
                back_rec = self._take_action(MenuAction, menu_action="back")
                drop_prefix_crops(captured_dialogue_crops([back_rec]), self.dialogue_crops)
                remaining_matched = self._match_tm_hm_target_row(self._seen_bag_items)
                if remaining_matched is None:
                    self._completed_tm_hm_party_attempt = True
                else:
                    self._tm_hm_scrolled_in_opening = False
                self.press_outcomes[-1] = (
                    f"confirm (opened party screen: all party Pokemon are NOT ABLE to learn {clean_tm}; backed out to item menu)"
                )
            return False
        return True

    def _auto_navigate_tm_hm_pocket_and_party(self) -> None:
        if (
            self._completed_tm_hm_party_attempt
            or not self._task_wants_move_teaching()
            or not self._task_allows_tm_hm_use()
        ):
            return
        frame = self._get_state()["core"]["current_frame"]
        if is_party_menu_screen(frame):
            if read_party_move_forget_list(frame) is not None or has_party_subbox(frame):
                self._advance_party_move_learning_and_forget(
                    "party Pokemon",
                    tm_label=self._active_menu_item or self._pending_bag_item,
                )
                return
            if is_party_menu_selection_screen(frame):
                self._handle_tm_hm_party_screen(self._active_menu_item or self._pending_bag_item or "TM")
                return
        if (
            is_pack_screen(frame)
            and not is_tm_hm_pack_pocket(frame)
            and pack_action_box_height(frame) < 4
            and not has_quantity_selector_box(frame)
            and self._task_wants_tm_hm_pocket()
        ):
            if self._current_call is None:
                self._begin_scripted("switch Pack to TM/HM pocket")
            for _ in range(3):
                if self._last_terminated or self._last_truncated:
                    return
                rec = self._take_action(MenuAction, menu_action="left")
                drop_prefix_crops(captured_dialogue_crops([rec]), self.dialogue_crops)
                self.presses.append("left")
                self.press_outcomes.append("left")
                if self._last_terminated or self._last_truncated:
                    return
                frame = self._get_state()["core"]["current_frame"]
                if is_tm_hm_pack_pocket(frame):
                    self._tm_hm_scrolled_in_opening = False
                    self.press_outcomes[-1] = "left (switched Pack to TM/HM pocket)"
                    break
        frame = self._get_state()["core"]["current_frame"]
        if (
            is_tm_hm_pack_pocket(frame)
            and is_pack_item_list_screen(frame)
            and not self._tm_hm_scrolled_in_opening
        ):
            rows = read_tm_hm_pocket_rows(frame)
            matched = self._match_tm_hm_target_row(rows)
            if matched is not None:
                target_idx, target_label = matched
                if self._current_call is None:
                    self._begin_scripted(f"select {target_label} in TM/HM pocket")
                self._tm_hm_scrolled_in_opening = True
                for _ in range(target_idx):
                    if self._last_terminated or self._last_truncated:
                        return
                    down_rec = self._take_action(MenuAction, menu_action="down")
                    drop_prefix_crops(captured_dialogue_crops([down_rec]), self.dialogue_crops)
                    self.presses.append("down")
                    self.press_outcomes.append("down")
                if self._last_terminated or self._last_truncated:
                    return
                self._active_menu_item = target_label
                sel_rec = self._take_action(MenuAction, menu_action="confirm")
                drop_prefix_crops(captured_dialogue_crops([sel_rec]), self.dialogue_crops)
                self.presses.append("confirm")
                self.press_outcomes.append(f"confirm (selected {target_label})")
                if self._last_terminated or self._last_truncated:
                    return
                after_sel = self._get_state()["core"]["current_frame"]
                if is_item_use_subbox_screen(after_sel):
                    use_rec = self._take_action(MenuAction, menu_action="confirm")
                    drop_prefix_crops(captured_dialogue_crops([use_rec]), self.dialogue_crops)
                    if self._last_terminated or self._last_truncated:
                        return
                    if self._advance_tm_hm_use_to_party_screen():
                        if self._last_terminated or self._last_truncated:
                            return
                        self._handle_tm_hm_party_screen(target_label)
            return
        for _ in range(3):
            if self._last_terminated or self._last_truncated or self._completed_tm_hm_party_attempt:
                return
            frame = self._get_state()["core"]["current_frame"]
            if not is_gen1_item_list_screen(frame) or self._tm_hm_scrolled_in_opening:
                break
            info = self._scan_bag_frame(frame)
            if info is None:
                break
            vis_items, has_more = info
            matched_vis = self._match_tm_hm_target_row(vis_items)
            matched_all = self._match_tm_hm_target_row(self._seen_bag_items)
            if matched_vis is None and matched_all is None and not has_more:
                break
            if self._current_call is None:
                self._begin_scripted("select TM/HM item in ITEM list")
            self._tm_hm_scrolled_in_opening = True
            target_label: Optional[str] = None
            for _ in range(16):
                if self._last_terminated or self._last_truncated:
                    return
                cur_frame = self._get_state()["core"]["current_frame"]
                cur_info = self._scan_bag_frame(cur_frame)
                if cur_info is None:
                    break
                cur_vis, cur_more = cur_info
                cur_row = read_bag_cursor_row(cur_frame, tracked_index=self._bag_cursor_idx)
                m_vis = self._match_tm_hm_target_row(cur_vis)
                if m_vis is not None:
                    _, tgt = m_vis
                    if cur_row == tgt:
                        target_label = tgt
                        break
                    if cur_row == "CANCEL":
                        step_dir = "up"
                    elif cur_row in cur_vis:
                        step_dir = "down" if cur_vis.index(tgt) > cur_vis.index(cur_row) else "up"
                    else:
                        step_dir = "down" if cur_more else "up"
                else:
                    m_seen = self._match_tm_hm_target_row(self._seen_bag_items)
                    if m_seen is not None and cur_vis and cur_vis[0] in self._seen_bag_items:
                        _, tgt = m_seen
                        step_dir = (
                            "up"
                            if self._seen_bag_items.index(tgt) < self._seen_bag_items.index(cur_vis[0])
                            else "down"
                        )
                    elif cur_more and not self._saw_bag_bottom:
                        step_dir = "down"
                    else:
                        break
                self._bag_scrolled_in_opening = True
                n_vis = len(cur_vis)
                max_idx = n_vis if not cur_more else max(0, n_vis - 1)
                if self._bag_cursor_idx is not None:
                    if step_dir == "down":
                        self._bag_cursor_idx = min(max_idx, self._bag_cursor_idx + 1)
                    elif step_dir == "up":
                        self._bag_cursor_idx = max(0, self._bag_cursor_idx - 1)
                step_rec = self._take_action(MenuAction, menu_action=step_dir)
                drop_prefix_crops(captured_dialogue_crops([step_rec]), self.dialogue_crops)
                self.presses.append(step_dir)
                self.press_outcomes.append(step_dir)
            if self._last_terminated or self._last_truncated or not target_label:
                return
            self._active_menu_item = target_label
            self._pending_bag_item = target_label
            sel_rec = self._take_action(MenuAction, menu_action="confirm")
            drop_prefix_crops(captured_dialogue_crops([sel_rec]), self.dialogue_crops)
            self.presses.append("confirm")
            self.press_outcomes.append(f"confirm (selected {target_label})")
            if self._last_terminated or self._last_truncated:
                return
            after_sel = self._get_state()["core"]["current_frame"]
            if is_item_use_subbox_screen(after_sel):
                use_rec = self._take_action(MenuAction, menu_action="confirm")
                drop_prefix_crops(captured_dialogue_crops([use_rec]), self.dialogue_crops)
                if self._last_terminated or self._last_truncated:
                    return
                if self._advance_tm_hm_use_to_party_screen():
                    if self._last_terminated or self._last_truncated:
                        return
                    if self._handle_tm_hm_party_screen(target_label):
                        break
                    continue
            break

    def _auto_scroll_bag_for_status_item(self) -> None:
        frame = self._get_state()["core"]["current_frame"]
        info = self._scan_bag_frame(frame)
        if info is None:
            return
        confirmed_count = len(self._confirmed_bag_items)
        if self._last_auto_scrolled_confirmed_count == confirmed_count:
            return
        if confirmed_count > 0 and not any(self._bag_target_matches(it) for it in self._seen_bag_items):
            self._last_auto_scrolled_confirmed_count = confirmed_count
            return
        self._last_auto_scrolled_confirmed_count = confirmed_count
        if self._current_call is None:
            self._begin_scripted("scroll bag to find target item")

        for _ in range(24):
            if self._last_terminated or self._last_truncated:
                return
            cur_frame = self._get_state()["core"]["current_frame"]
            cur_info = self._scan_bag_frame(cur_frame)
            if cur_info is None:
                break
            vis_items, has_more = cur_info
            cursor_row = read_bag_cursor_row(cur_frame, tracked_index=self._bag_cursor_idx)
            vis_targets = [it for it in vis_items if self._bag_target_matches(it)]
            if vis_targets:
                target = vis_targets[0]
                if cursor_row == target:
                    break
                if cursor_row == "CANCEL":
                    step_dir = "up"
                elif cursor_row in vis_items:
                    step_dir = "down" if vis_items.index(target) > vis_items.index(cursor_row) else "up"
                else:
                    step_dir = "down" if has_more else "up"
            else:
                seen_targets = [it for it in self._seen_bag_items if self._bag_target_matches(it)]
                if seen_targets and vis_items and vis_items[0] in self._seen_bag_items:
                    target = seen_targets[0]
                    if self._seen_bag_items.index(target) < self._seen_bag_items.index(vis_items[0]):
                        step_dir = "up"
                    else:
                        step_dir = "down"
                elif not self._saw_bag_bottom:
                    step_dir = "down"
                elif not self._saw_bag_top:
                    step_dir = "up"
                else:
                    break

            self._bag_scrolled_in_opening = True
            n_vis = len(vis_items)
            max_idx = n_vis if not has_more else max(0, n_vis - 1)
            if self._bag_cursor_idx is not None:
                if step_dir == "down":
                    self._bag_cursor_idx = min(max_idx, self._bag_cursor_idx + 1)
                elif step_dir == "up":
                    self._bag_cursor_idx = max(0, self._bag_cursor_idx - 1)
            rec = self._take_action(MenuAction, menu_action=step_dir)
            drop_prefix_crops(captured_dialogue_crops([rec]), self.dialogue_crops)
            if self._last_terminated or self._last_truncated:
                return
            after_fr = self._get_state()["core"]["current_frame"]
            after_info = self._scan_bag_frame(after_fr)
            if after_info is None:
                break
            after_items, after_more = after_info
            after_cursor = read_bag_cursor_row(after_fr, tracked_index=self._bag_cursor_idx)
            if step_dir == "down" and (
                not after_more
                or (after_items == vis_items and after_cursor == cursor_row)
                or is_same_battle_frame(rec.frame_before, rec.frame_after)
            ):
                self._saw_bag_bottom = True
                if self._party_state is not None:
                    self._party_state.saw_bottom = True
                self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)
            elif step_dir == "up" and (
                (after_items == vis_items and after_cursor == cursor_row)
                or is_same_battle_frame(rec.frame_before, rec.frame_after)
            ):
                self._saw_bag_top = True
                if self._party_state is not None:
                    self._party_state.saw_top = True
                self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)

    def _last_seen_party_hp(self) -> Optional[Tuple[int, int]]:
        last_party: Optional[Tuple[int, int]] = None
        for step in self.report.steps:
            if isinstance(step, EnvironmentStepRecord):
                for frame in (step.frame_before, step.frame_after):
                    found_party = read_party_menu_hp(frame)
                    if found_party is not None:
                        last_party = found_party
        if last_party is not None:
            return last_party
        last_vlm: Optional[Tuple[int, int]] = None
        for call in self.report.vlm_call_log:
            if isinstance(call, ExecutorVLMCallRecord) and call.tag == self.STEP_TAG:
                found = CombatExecutor.extract_player_hp_from_text(str(call.response))
                if found is not None:
                    last_vlm = found
        return last_vlm

    def _last_seen_party_status(self) -> Optional[str]:
        if self._saw_status_cure_effect():
            return ""
        last_st: Optional[str] = None
        for step in self.report.steps:
            if isinstance(step, EnvironmentStepRecord):
                for frame in (step.frame_before, step.frame_after):
                    found_st = read_party_menu_status(frame)
                    if found_st is not None:
                        last_st = found_st
        return last_st

    def _saw_status_cure_effect(self) -> bool:
        if self.dialogue_text and re.search(
            r"\b(?:cured\s+of\s+(?:its\s+)?poison|poison\s+faded|healed\s+of\s+(?:its\s+)?(?:poison|burn|paralysis)|woke\s+up|thawed\s+out)\b",
            self.dialogue_text,
            re.IGNORECASE,
        ):
            return True
        for call in self.report.vlm_call_log:
            if isinstance(call, ExecutorVLMCallRecord) and call.tag == self.STEP_TAG:
                if re.search(
                    r"\b(?:cured\s+of\s+(?:its\s+)?poison|poison\s+faded|healed\s+of\s+(?:its\s+)?(?:poison|burn|paralysis)|woke\s+up|thawed\s+out)\b",
                    str(call.response),
                    re.IGNORECASE,
                ):
                    return True
        return False

    def _finish(self, code: int, reason: str) -> int:
        if self.dialogue_crops:
            self.dialogue_text = "\n".join(ocr(self.dialogue_crops, self._vlm, do_merge=True,
                                               parameters=self._parameters))
        for step in self.report.steps:
            if isinstance(step, EnvironmentStepRecord):
                self._scan_move_learning_frame(step.frame_before)
                self._scan_move_learning_frame(step.frame_after)
        if self.dialogue_text:
            if self._party_state is not None:
                self._party_state.update_learned_moves_from_text(self.dialogue_text)
            if self._last_move_teaching_outcome is None or self._last_move_teaching_outcome[0] == "did_not_learn":
                for raw_line in self.dialogue_text.splitlines():
                    line = raw_line.strip()
                    m_learn = re.match(
                        r"^([A-Z][A-Z0-9♀♂.'-]{1,10})\s+(learned|(?:already\s+)?knows)\s+([A-Z][A-Z0-9 -]{2,14})[!.]?$",
                        line,
                    )
                    if m_learn:
                        kind = "learned" if m_learn.group(2).lower() == "learned" else "knows"
                        self._last_move_teaching_outcome = (kind, m_learn.group(1), m_learn.group(3).strip())
        notes_lines = [f"Presses: {', '.join(self.presses) or 'none'}"]
        if self.dialogue_text:
            notes_lines.append(self.dialogue_text)
        saw_heal = self._saw_healing_item_effect()
        saw_status_cure = self._saw_status_cure_effect()
        party_hp = self._last_seen_party_hp()
        if party_hp is not None and (
            saw_heal
            or re.search(r"\b(?:item|bag|potion|restore\s+hp|heal|pokemon|party)\b", self._task or "", re.IGNORECASE)
        ):
            notes_lines.append(f"[Party HP seen in menu: {party_hp[0]}/{party_hp[1]} HP]")
        party_status = self._last_seen_party_status()
        if party_status is not None:
            notes_lines.append(f"[Party status seen in menu: {party_status or 'healthy'}]")
        if self._opened_bag_in_menu:
            all_checked = (
                self._saw_bag_end
                or (self._party_state is not None and self._party_state.bag_complete)
            )
            end_note = " (all bag slots checked)" if all_checked else ""
            items_str = ", ".join(self._seen_bag_items) if self._seen_bag_items else "none"
            notes_lines.append(f"[Bag items seen: {items_str}{end_note}]")
        if self._last_move_teaching_outcome is not None and self._last_move_teaching_outcome[0] in ("learned", "knows"):
            m_kind, m_poke, m_move = self._last_move_teaching_outcome
            if m_kind == "learned":
                notes_lines.append(f"[Move taught in menu: {m_poke} learned {m_move}]")
            else:
                notes_lines.append(f"[Move already known in menu: {m_poke} knows {m_move}]")
        elif self._last_incompatible_tm:
            notes_lines.append(
                f"[TM/HM incompatible with party in menu: {self._last_incompatible_tm} cannot be learned by any current party Pokemon]"
            )
        elif self._task_wants_move_teaching():
            notes_lines.append("[No move was taught in menu: menu closed without teaching the move to any party Pokemon]")
        if (
            self.is_start_menu
            and re.search(r"\b(?:item|bag|potion|restore\s+hp)\b", self._task or "", re.IGNORECASE)
            and re.search(r"\b(?:heal|potion|restore\s+hp)\b", self._task or "", re.IGNORECASE)
            and not saw_heal
            and not saw_status_cure
        ):
            if self._party_state is not None:
                self._party_state.mark_healing_bag_empty()
            notes_lines.append("[No healing item was used in menu: bag had no usable healing item]")
        if (
            self._task_wants_status_heal()
            and not saw_status_cure
            and (self._saw_bag_end or not self._seen_status_heal_item())
            and (self._opened_bag_in_menu or self.is_start_menu)
        ):
            if self._party_state is not None:
                self._party_state.mark_status_heal_bag_empty()
            notes_lines.append("[No status-healing item was used in menu: bag had no usable status-healing item]")
        self.report.termination_reason = reason
        self.report.notes = "\n".join(filter(None, notes_lines))
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
        cur_frame = self._get_state()["core"]["current_frame"]
        bag_info = self._scan_bag_frame(cur_frame)
        lines: List[str] = []
        if self.presses:
            outcomes = self.press_outcomes if len(self.press_outcomes) == len(self.presses) else self.presses
            lines.append(f"Presses made so far, in order: {', '.join(outcomes)}.")
        if bag_info is not None:
            vis_items, has_more = bag_info
            cursor_row = read_bag_cursor_row(cur_frame, tracked_index=self._bag_cursor_idx)
            formatted_rows = [
                f"{it} (x{self._bag_item_quantities[it]})"
                if self._bag_item_quantities.get(it) is not None
                else it
                for it in vis_items
            ]
            if not has_more:
                formatted_rows.append("CANCEL")
            more_parts: List[str] = []
            if has_more:
                more_parts.append("more items below")
            if not self._saw_bag_top:
                more_parts.append("top bag slots above this window not yet checked — press up to scroll up")
            more_note = f" ({'; '.join(more_parts)})" if more_parts else ""
            if formatted_rows:
                lines.append(f"Visible bag rows in order: {', '.join(formatted_rows)}{more_note}.")
            if cursor_row:
                lines.append(f"Current bag cursor row: {cursor_row}.")
        if self._opened_bag_in_menu and self._seen_bag_items:
            if self._saw_bag_end or (self._party_state is not None and self._party_state.bag_complete):
                end_note = " (all bag slots checked)"
            elif self._saw_bag_bottom and not self._saw_bag_top:
                end_note = " (bottom of bag reached, but top bag slots above have NOT been checked yet — press up to scroll up)"
            else:
                end_note = " (more items may be below)"
            lines.append(f"Bag items seen so far: {', '.join(self._seen_bag_items)}{end_note}.")
            if self._task_wants_status_heal() and self._saw_bag_end and not self._seen_status_heal_item():
                lines.append(
                    "Note: All bag slots were checked and there is NO Antidote or status-healing item in the bag."
                )
        tm_hm_rows = read_tm_hm_pocket_rows(cur_frame)
        if tm_hm_rows:
            lines.append(f"Visible TM/HM pocket rows in order: {', '.join(tm_hm_rows)}.")
        if self._completed_tm_hm_party_attempt and (
            is_tm_hm_pack_pocket(cur_frame)
            or is_gen1_item_list_screen(cur_frame)
            or is_start_menu_screen(cur_frame)
        ):
            if self._last_move_teaching_outcome and self._last_move_teaching_outcome[0] in ("learned", "knows"):
                _, m_poke, m_move = self._last_move_teaching_outcome
                lines.append(
                    f"Status: {m_poke} knows {m_move} (TM/HM move teaching on the party screen is finished and the screen has returned to the menu) — do NOT press confirm on the item again; press back to close the Pack/Item and Start menus."
                )
            else:
                lines.append(
                    "Status: TM/HM move teaching on the party screen is finished and the screen has returned to the menu — do NOT press confirm on the item again; press back to close the Pack/Item and Start menus."
                )
        forget_info = read_party_move_forget_list(cur_frame)
        if forget_info is not None:
            f_moves, f_cur = forget_info
            cur_note = f" (currently highlighted: {f_cur})" if f_cur else ""
            lines.append(
                f"Move-forget selection box open — moves in order: {', '.join(f_moves)}{cur_note}. Press up/down to select the move to forget and press confirm to forget it (do NOT press back)."
            )
        elif is_party_menu_screen(cur_frame) and has_party_subbox(cur_frame):
            d_l1, _ = read_dialogue_box_lines(cur_frame)
            if d_l1.lower().startswith("stop ") or d_l1.lower().startswith("abandon "):
                lines.append(
                    "Prompt: 'Stop learning move?' YES/NO box is open — press back (or select NO) to continue teaching the move."
                )
            else:
                lines.append(
                    "Prompt: 'Delete an older move to make room?' YES/NO box is open — press confirm on YES to choose a move to forget."
                )
        if has_quantity_selector_box(cur_frame) and not has_menu_cursor(cur_frame):
            q_val = read_quantity_selector_value(cur_frame)
            if q_val is not None:
                item_label = f" for {self._active_menu_item}" if self._active_menu_item else ""
                lines.append(
                    f"Current quantity selector value{item_label}: x{q_val:02d} (press confirm to accept x{q_val:02d}, or up/down to change)."
                )
        if not lines:
            return ""
        return "\n" + "\n".join(lines) + "\n"

    def _saw_healing_item_effect(self) -> bool:
        if self.dialogue_text and re.search(
            r"\b(?:recovered\s+(?:by\s+)?\d+(?:\s*HP)?|restored\s+\d+\s*HP|HP\s+was\s+restored)\b",
            self.dialogue_text,
            re.IGNORECASE,
        ):
            return True
        for call in self.report.vlm_call_log:
            if isinstance(call, ExecutorVLMCallRecord) and call.tag == self.STEP_TAG:
                if re.search(
                    r"\b(?:recovered\s+(?:by\s+)?\d+(?:\s*HP)?|restored\s+\d+\s*HP|HP\s+was\s+restored)\b",
                    str(call.response),
                    re.IGNORECASE,
                ):
                    return True
        return False

    @staticmethod
    def _has_menu_bottom_box(frame: np.ndarray) -> bool:
        return has_bottom_dialogue_box(frame) or (
            is_party_menu_screen(frame) and has_bottom_location_banner(frame)
        )

    def _should_settle_menu_dialogue(self, before_frame: np.ndarray, after_frame: np.ndarray, press: str) -> bool:
        if press not in ("confirm", "back"):
            return False
        if not (self._has_menu_bottom_box(before_frame) and self._has_menu_bottom_box(after_frame)):
            return False
        if has_menu_cursor(after_frame):
            return False
        q_before = has_quantity_selector_box(before_frame)
        q_after = has_quantity_selector_box(after_frame)
        if not (has_menu_cursor(before_frame) or q_before):
            return False
        if not q_before and q_after:
            return False
        return True

    def _settle_menu_dialogue(self) -> None:
        stalled_dismisses = 0
        for _ in range(self.MAX_MENU_DIALOGUE_SETTLE_STEPS):
            if self._last_terminated or self._last_truncated:
                return
            frame = self._get_state()["core"]["current_frame"]
            self._scan_move_learning_frame(frame)
            state = self._effective_menu_state(frame)
            if state != AgentState.IN_MENU:
                return
            if not self._has_menu_bottom_box(frame) or has_menu_cursor(frame):
                return
            if has_party_subbox(frame) or pack_action_box_height(frame) >= 4:
                wait_rec = self._take_action(WaitAction)
                drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                return
            if is_party_menu_screen(frame) and not has_bottom_dialogue_box(frame):
                wait_rec = self._take_action(WaitAction)
                drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                if self._last_terminated or self._last_truncated:
                    return
                after_wait = self._get_state()["core"]["current_frame"]
                self._scan_move_learning_frame(after_wait)
                state = self._effective_menu_state(after_wait)
                if state != AgentState.IN_MENU or not self._has_menu_bottom_box(after_wait) or has_menu_cursor(after_wait):
                    return
                if not is_same_battle_frame(wait_rec.frame_before, wait_rec.frame_after):
                    stalled_dismisses = 0
                    continue
            dismiss_rec = self._take_action(MenuAction, menu_action="back")
            drop_prefix_crops(captured_dialogue_crops([dismiss_rec]), self.dialogue_crops)
            self._scan_move_learning_frame(dismiss_rec.frame_before)
            self._scan_move_learning_frame(dismiss_rec.frame_after)
            if self._last_terminated or self._last_truncated:
                return
            after_dismiss = self._get_state()["core"]["current_frame"]
            self._scan_move_learning_frame(after_dismiss)
            state = self._effective_menu_state(after_dismiss)
            if state != AgentState.IN_MENU or not self._has_menu_bottom_box(after_dismiss) or has_menu_cursor(after_dismiss):
                return
            if has_party_subbox(after_dismiss) or pack_action_box_height(after_dismiss) >= 4:
                wait_rec = self._take_action(WaitAction)
                drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                return
            if is_same_battle_frame(dismiss_rec.frame_before, dismiss_rec.frame_after):
                wait_rec = self._take_action(WaitAction)
                drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                if self._last_terminated or self._last_truncated:
                    return
                after_wait = self._get_state()["core"]["current_frame"]
                state = self._effective_menu_state(after_wait)
                if state != AgentState.IN_MENU or not self._has_menu_bottom_box(after_wait) or has_menu_cursor(after_wait):
                    return
                if is_same_battle_frame(wait_rec.frame_before, wait_rec.frame_after):
                    stalled_dismisses += 1
                    if stalled_dismisses >= self.MAX_STALLED_MENU_DISMISSES:
                        return
                else:
                    stalled_dismisses = 0
            else:
                stalled_dismisses = 0

    def _auto_close_completed_overworld_menu(self) -> None:
        if not self._overworld_menu_session or not self._completed_tm_hm_party_attempt:
            return
        if any(self._bag_target_matches(it) for it in self._seen_bag_items):
            return
        raw = self._task or ""
        if re.search(r"\b(?:field\s+move|use\s+cut\s+from\s+the\s+party)\b", raw, re.IGNORECASE):
            return
        if self._current_call is None:
            self._begin_scripted("close Pack/Item and Start menus to return to overworld")
        for _ in range(4):
            if self._last_terminated or self._last_truncated:
                return
            cur_frame = self._get_state()["core"]["current_frame"]
            if self._effective_menu_state(cur_frame) != AgentState.IN_MENU:
                return
            if not (
                is_pack_screen(cur_frame)
                or is_gen1_item_list_screen(cur_frame)
                or is_start_menu_screen(cur_frame)
            ):
                return
            rec = self._take_action(MenuAction, menu_action="back")
            drop_prefix_crops(captured_dialogue_crops([rec]), self.dialogue_crops)
            self.presses.append("back")
            self.press_outcomes.append("back (closed menu)")
            if self._last_terminated or self._last_truncated:
                return
            after_frame = self._get_state()["core"]["current_frame"]
            if self._effective_menu_state(after_frame) != AgentState.IN_MENU:
                return
            if is_start_menu_screen(after_frame) and not has_menu_cursor(after_frame):
                for _ in range(2):
                    wait_rec = self._take_action(WaitAction)
                    drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                    if self._last_terminated or self._last_truncated:
                        return
                    after_frame = self._get_state()["core"]["current_frame"]
                    if has_menu_cursor(after_frame) or not is_start_menu_screen(after_frame):
                        break

    def _consume_confirmed_bag_item(self, item_name: Optional[str]) -> None:
        if not item_name or item_name == "CANCEL":
            return
        self._confirmed_bag_items.add(item_name)
        self._last_consumed_bag_item = item_name
        if self._party_state is not None:
            self._party_state.consume_item(item_name)
            self._seen_bag_items = list(self._party_state.items)
            self._bag_item_quantities = dict(self._party_state.quantities)
        else:
            qty = self._bag_item_quantities.get(item_name)
            if qty is not None and qty > 1:
                self._bag_item_quantities[item_name] = qty - 1
            elif qty == 1:
                self._bag_item_quantities.pop(item_name, None)
                if item_name in self._seen_bag_items:
                    self._seen_bag_items.remove(item_name)

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
        init_frame = self._get_state()["core"]["current_frame"]
        if self.is_start_menu or (
            self._agent_state() == AgentState.IN_MENU
            and (is_overworld_menu_or_ui_screen(init_frame) or not is_battle_screen(init_frame))
            and not re.search(r"\bactive\s+battle\s+instructions\b", self._task or "", re.IGNORECASE)
        ):
            self._overworld_menu_session = True

        needs_menu_closed = self.is_start_menu or bool(
            re.search(r"\b(?:close|exit|return\s+to\s+(?:the\s+)?overworld)\b", self._task or "", re.IGNORECASE)
        )
        consecutive_invalid = 0
        for _ in range(self._max_steps):
            frame = self._get_state()["core"]["current_frame"]
            state = self._effective_menu_state(frame)
            if state != AgentState.IN_MENU:
                code, reason = LEFT_MENU_OUTCOMES.get(state, (MENU_OK, "menu_closed"))
                return self._finish(code, reason)
            self._auto_navigate_tm_hm_pocket_and_party()
            if self._last_terminated:
                return self._finish(MENU_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(MENU_TRUNCATED, "truncated")
            if self._completed_tm_hm_party_attempt and needs_menu_closed:
                self._auto_close_completed_overworld_menu()
                if self._last_terminated:
                    return self._finish(MENU_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._finish(MENU_TRUNCATED, "truncated")
                closed_frame = self._get_state()["core"]["current_frame"]
                closed_state = self._effective_menu_state(closed_frame)
                if closed_state != AgentState.IN_MENU:
                    code, reason = LEFT_MENU_OUTCOMES.get(closed_state, (MENU_OK, "menu_closed"))
                    return self._finish(code, reason)
            self._auto_scroll_bag_for_status_item()
            if self._last_terminated:
                return self._finish(MENU_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(MENU_TRUNCATED, "truncated")
            self._auto_adjust_quantity_selector()
            if self._last_terminated:
                return self._finish(MENU_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(MENU_TRUNCATED, "truncated")
            frame = self._get_state()["core"]["current_frame"]
            state = self._effective_menu_state(frame)
            if state != AgentState.IN_MENU:
                code, reason = LEFT_MENU_OUTCOMES.get(state, (MENU_OK, "menu_closed"))
                return self._finish(code, reason)
            self._scan_bag_frame(frame)
            prompt = (self.STEP_PROMPT.replace("[TASK]", self._task.strip())
                      .replace("[HISTORY]", self._history_block()))
            output = self._vlm_call(self.STEP_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            press_raw = (parse_key_value(output, "Press") or "").strip().lower()
            press = next((option for option in MENU_PRESSES if option in press_raw), None)
            if parse_yes_no(output, "Complete") and (self.presses or self.is_start_menu):
                if (self._has_menu_bottom_box(frame)
                        and not has_menu_cursor(frame)
                        and not has_quantity_selector_box(frame)):
                    self._settle_menu_dialogue()
                    if self._last_terminated:
                        return self._finish(MENU_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._finish(MENU_TRUNCATED, "truncated")
                    settled_frame = self._get_state()["core"]["current_frame"]
                    settled_state = self._effective_menu_state(settled_frame)
                    if settled_state != AgentState.IN_MENU:
                        code, reason = LEFT_MENU_OUTCOMES.get(settled_state, (MENU_OK, "menu_closed"))
                        return self._finish(code, reason)
                    frame = settled_frame
                if needs_menu_closed and (
                    is_pack_screen(frame)
                    or is_gen1_item_list_screen(frame)
                    or is_party_menu_screen(frame)
                    or is_start_menu_screen(frame)
                ):
                    press = "back"
                else:
                    return self._finish(MENU_OK, "objective_complete")
            if press is None:
                self._record_invalid(output)
                consecutive_invalid += 1
                if consecutive_invalid >= self.MAX_INVALID:
                    return self._fail(MENU_UNREADABLE, "press_unreadable", output)
                continue
            consecutive_invalid = 0
            bag_before = self._scan_bag_frame(frame)
            pre_bag_cursor = (
                read_bag_cursor_row(frame, tracked_index=self._bag_cursor_idx)
                if bag_before is not None
                else None
            )
            pre_menu_item = read_menu_cursor_item_name(frame, tracked_index=self._bag_cursor_idx)
            if press == "confirm" and pre_menu_item and not has_quantity_selector_box(frame):
                self._active_menu_item = pre_menu_item
            if bag_before is not None:
                if press in ("up", "down"):
                    self._bag_scrolled_in_opening = True
                if self._bag_cursor_idx is not None:
                    n_vis = len(bag_before[0])
                    max_idx = n_vis if not bag_before[1] else max(0, n_vis - 1)
                    if press == "down":
                        self._bag_cursor_idx = min(max_idx, self._bag_cursor_idx + 1)
                    elif press == "up":
                        self._bag_cursor_idx = max(0, self._bag_cursor_idx - 1)
                if press == "confirm" and pre_bag_cursor and pre_bag_cursor != "CANCEL":
                    self._pending_bag_item = pre_bag_cursor
            self.presses.append(press)
            self.press_outcomes.append(press)
            was_party_menu = is_party_menu_screen(frame)
            was_tm_hm_context = (
                is_tm_hm_pack_pocket(frame)
                or is_tm_hm_startup_dialogue_frame(frame)
                or is_tm_hm_item_name(self._active_menu_item)
                or is_tm_hm_item_name(self._pending_bag_item)
            )
            was_tm_hm_use = (
                press == "confirm"
                and was_tm_hm_context
                and (is_item_use_subbox_screen(frame) or is_tm_hm_startup_dialogue_frame(frame))
            )
            record = self._take_action(MenuAction, menu_action=press)
            drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
            if self._last_terminated:
                return self._finish(MENU_TERMINATED, "terminated")
            if self._last_truncated:
                return self._finish(MENU_TRUNCATED, "truncated")
            after_frame = self._get_state()["core"]["current_frame"]
            if (
                press == "confirm"
                and not was_tm_hm_use
                and self._task_allows_tm_hm_use()
                and is_item_use_subbox_screen(after_frame)
                and (
                    is_tm_hm_pack_pocket(after_frame)
                    or is_tm_hm_item_name(self._active_menu_item)
                    or is_tm_hm_item_name(self._pending_bag_item)
                )
            ):
                use_rec = self._take_action(MenuAction, menu_action="confirm")
                drop_prefix_crops(captured_dialogue_crops([use_rec]), self.dialogue_crops)
                if self._last_terminated:
                    return self._finish(MENU_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._finish(MENU_TRUNCATED, "truncated")
                was_tm_hm_use = True
            if was_tm_hm_use:
                reached_party = self._advance_tm_hm_use_to_party_screen()
                if self._last_terminated:
                    return self._finish(MENU_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._finish(MENU_TRUNCATED, "truncated")
                if reached_party:
                    tm_code = self._active_menu_item or self._pending_bag_item or "TM"
                    self._handle_tm_hm_party_screen(tm_code)
                    if self._last_terminated:
                        return self._finish(MENU_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._finish(MENU_TRUNCATED, "truncated")
            after_frame = self._get_state()["core"]["current_frame"]
            if press == "back" and is_start_menu_screen(after_frame) and not has_menu_cursor(after_frame):
                for _ in range(2):
                    wait_rec = self._take_action(WaitAction)
                    drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                    if self._last_terminated or self._last_truncated:
                        break
                    after_frame = self._get_state()["core"]["current_frame"]
                    if has_menu_cursor(after_frame) or not is_start_menu_screen(after_frame):
                        break
            bag_after = self._scan_bag_frame(after_frame)
            if bag_before is not None and bag_after is not None:
                after_bag_cursor = read_bag_cursor_row(after_frame, tracked_index=self._bag_cursor_idx)
                if press == "down" and (
                    not bag_after[1]
                    or (bag_after[0] == bag_before[0] and after_bag_cursor == pre_bag_cursor)
                    or is_same_battle_frame(record.frame_before, record.frame_after)
                ):
                    self._saw_bag_bottom = True
                    if self._party_state is not None:
                        self._party_state.saw_bottom = True
                    self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)
                elif press == "up" and (
                    (bag_after[0] == bag_before[0] and after_bag_cursor == pre_bag_cursor)
                    or is_same_battle_frame(record.frame_before, record.frame_after)
                ):
                    self._saw_bag_top = True
                    if self._party_state is not None:
                        self._party_state.saw_top = True
                    self._saw_bag_end = bool(self._saw_bag_top and self._saw_bag_bottom)
            auto_party_steps: List[str] = []
            if (
                press == "confirm"
                and self._pending_bag_item
                and (
                    is_item_use_subbox_screen(after_frame)
                    or is_party_menu_selection_screen(after_frame)
                )
            ):
                auto_party_steps = self._complete_menu_party_item_application(self._pending_bag_item)
                if self._last_terminated:
                    return self._finish(MENU_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._finish(MENU_TRUNCATED, "truncated")
                if auto_party_steps:
                    after_frame = self._get_state()["core"]["current_frame"]
                    bag_after = self._scan_bag_frame(after_frame)
                    if "settled effect dialogue" in auto_party_steps or is_party_menu_screen(after_frame):
                        self._consume_confirmed_bag_item(self._pending_bag_item)
                        if not is_party_menu_screen(after_frame):
                            if any(self._bag_target_matches(it) for it in self._seen_bag_items):
                                self.press_outcomes[-1] = (
                                    f"{press} (selected {self._pending_bag_item}, {', '.join(auto_party_steps)}, and returned to item menu)"
                                )
                            else:
                                self.press_outcomes[-1] = (
                                    f"{press} (selected {self._pending_bag_item}, {', '.join(auto_party_steps)}, and returned to item menu — press back to close the item and Start menus)"
                                )
                        else:
                            self.press_outcomes[-1] = f"{press} (selected {self._pending_bag_item}, {', '.join(auto_party_steps)})"
                        self._pending_bag_item = None
                    else:
                        self.press_outcomes[-1] = f"{press} (selected {self._pending_bag_item}, {', '.join(auto_party_steps)})"
            if (
                press == "confirm"
                and self._pending_bag_item
                and (
                    is_party_menu_screen(after_frame)
                    or has_quantity_selector_box(after_frame)
                    or (has_bottom_dialogue_box(after_frame) and bag_after is None)
                )
            ):
                self._consume_confirmed_bag_item(self._pending_bag_item)
                self._pending_bag_item = None
            state = self._effective_menu_state(after_frame)
            if state != AgentState.IN_MENU:
                code, reason = LEFT_MENU_OUTCOMES.get(state, (MENU_OK, "menu_closed"))
                return self._finish(code, reason)
            if self._should_settle_menu_dialogue(frame, after_frame, press):
                self._settle_menu_dialogue()
                if self._last_terminated:
                    return self._finish(MENU_TERMINATED, "terminated")
                if self._last_truncated:
                    return self._finish(MENU_TRUNCATED, "truncated")
                after_settle = self._get_state()["core"]["current_frame"]
                self._scan_bag_frame(after_settle)
                if was_party_menu and not is_party_menu_screen(after_settle):
                    if is_tm_hm_pack_pocket(after_settle) or is_tm_hm_item_name(self._active_menu_item):
                        if self._last_move_teaching_outcome and self._last_move_teaching_outcome[0] == "learned":
                            _, m_poke, m_move = self._last_move_teaching_outcome
                            self._completed_tm_hm_party_attempt = True
                            self.press_outcomes[-1] = (
                                f"{press} ({m_poke} learned {m_move}, settled move-learning text, and returned to item menu — do NOT press confirm again; press back to close the Pack/Item and Start menus)"
                            )
                        elif self._last_move_teaching_outcome and self._last_move_teaching_outcome[0] == "knows":
                            _, m_poke, m_move = self._last_move_teaching_outcome
                            self._completed_tm_hm_party_attempt = True
                            self.press_outcomes[-1] = (
                                f"{press} ({m_poke} already knows {m_move}, settled text, and returned to item menu — do NOT press confirm again; press back to close the Pack/Item and Start menus)"
                            )
                        elif (
                            press == "back"
                            or (self._last_move_teaching_outcome and self._last_move_teaching_outcome[0] == "did_not_learn")
                        ):
                            self._completed_tm_hm_party_attempt = False
                            self.press_outcomes[-1] = (
                                f"{press} (returned from party screen to item menu without teaching the move)"
                            )
                        else:
                            self._completed_tm_hm_party_attempt = True
                            self.press_outcomes[-1] = (
                                f"{press} (completed TM/HM move teaching on party Pokemon, settled move-learning text, and returned to item menu — do NOT press confirm again; press back to close the Pack/Item and Start menus)"
                            )
                    elif any(self._bag_target_matches(it) for it in self._seen_bag_items):
                        self.press_outcomes[-1] = (
                            f"{press} (used on party Pokemon, settled healing/effect text, and returned to item menu)"
                        )
                    else:
                        self.press_outcomes[-1] = (
                            f"{press} (used on party Pokemon, settled healing/effect text, and returned to item menu — press back to close the item and Start menus)"
                        )
                elif was_party_menu:
                    self.press_outcomes[-1] = f"{press} (settled party text)"
                else:
                    self.press_outcomes[-1] = f"{press} (settled menu dialogue)"
                if self._completed_tm_hm_party_attempt and needs_menu_closed:
                    self._auto_close_completed_overworld_menu()
                    if self._last_terminated:
                        return self._finish(MENU_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._finish(MENU_TRUNCATED, "truncated")
                    after_settle = self._get_state()["core"]["current_frame"]
                state = self._effective_menu_state(after_settle)
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

DIALOGUE_SETTLE_WAIT = 5
DIALOGUE_MAX_CUTSCENE_WAIT = 30
DIALOGUE_POST_MOTION_WAIT = 5
DIALOGUE_POST_WARP_BANNER_WAIT = 6
DIALOGUE_FANFARE_RETRY_LIMIT = 4
DIALOGUE_BUTTON_SETTLE_WAIT = 2
DIALOGUE_TOP_BOX_PRESSES = 4
DIALOGUE_TOP_BOX_SCROLL_WAITS = 3
FREE_SETTLE_WAIT = 5
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

    def _step_wait_with_crops(self) -> Tuple[np.ndarray, np.ndarray]:
        wait_record = self._take_action(WaitAction)
        drop_prefix_crops(captured_dialogue_crops([wait_record]), self.dialogue_crops)
        return wait_record.frame_before, wait_record.frame_after

    def _wait_for_next_box(self, settle_wait: int = DIALOGUE_SETTLE_WAIT):
        state, _, _ = settle_overworld_free_roam(
            env=self._env,
            get_state_fn=self._agent_state,
            get_frame_fn=self._current_frame,
            is_done_fn=lambda: bool(self._last_terminated or self._last_truncated),
            step_wait_fn=self._step_wait_with_crops,
            settle_wait=settle_wait,
            max_cutscene_wait=DIALOGUE_MAX_CUTSCENE_WAIT,
            post_motion_wait=DIALOGUE_POST_MOTION_WAIT,
            post_warp_banner_wait=DIALOGUE_POST_WARP_BANNER_WAIT,
            include_top_box=True,
            record_dialogue_settled=True,
        )
        return state

    def _settle_state(self):
        return settle_overworld_menu_state(
            get_state_fn=self._agent_state,
            get_frame_fn=self._current_frame,
            is_done_fn=lambda: bool(self._last_terminated or self._last_truncated),
            step_wait_fn=self._step_wait_with_crops,
            min_wait=MENU_SETTLE_WAIT,
            settle_wait=DIALOGUE_SETTLE_WAIT,
            max_cutscene_wait=DIALOGUE_MAX_CUTSCENE_WAIT,
            post_motion_wait=DIALOGUE_POST_MOTION_WAIT,
            require_waited_for_ready_check=True,
        )

    def _settle_after_step(self, settle_wait: int = DIALOGUE_SETTLE_WAIT):
        state = self._agent_state()
        if state == AgentState.FREE_ROAM:
            state = self._wait_for_next_box(settle_wait=settle_wait)
        if state == AgentState.IN_MENU:
            state = self._settle_state()
        return state

    def _capture_current_frame_dialogue_crop(self, frame: Optional[np.ndarray] = None) -> None:
        fr = self._current_frame() if frame is None else frame
        crop = extract_bottom_dialogue_crop(fr)
        if crop is not None:
            drop_prefix_crops([crop], self.dialogue_crops)

    def _settle_top_box_hangup_or_ring(self, max_waits: int = DIALOGUE_MAX_CUTSCENE_WAIT) -> bool:
        progressed = False
        for _ in range(max_waits):
            if (
                self._agent_state() != AgentState.IN_MENU
                or not self._is_top_box_dialogue()
                or self._last_terminated
                or self._last_truncated
            ):
                break
            wait_rec = self._take_action(WaitAction)
            drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
            if wait_rec.frame_changed or not is_same_battle_frame(wait_rec.frame_before, wait_rec.frame_after):
                progressed = True
            cur_frame = self._current_frame()
            if (
                has_bottom_dialogue_box(cur_frame)
                and not is_top_box_hangup_or_blank_bottom_box(cur_frame)
                and is_same_battle_frame(wait_rec.frame_before, wait_rec.frame_after)
            ):
                break
        return progressed

    def _advance_dialogue_box(self, button: str = "a") -> bool:
        if self._agent_state() == AgentState.IN_MENU and self._is_top_box_dialogue():
            menu_action = "back" if button == "b" else "confirm"
            progressed = False
            self._capture_current_frame_dialogue_crop()
            cur_frame = self._current_frame()
            if not has_bottom_dialogue_box(cur_frame) or is_top_box_hangup_or_blank_bottom_box(cur_frame):
                if self._settle_top_box_hangup_or_ring():
                    progressed = True
                if self._last_terminated or self._last_truncated:
                    return True
                if self._agent_state() != AgentState.IN_MENU or not self._is_top_box_dialogue():
                    self._settle_after_step()
                    return True
                if not has_bottom_dialogue_box(self._current_frame()) or is_top_box_hangup_or_blank_bottom_box(
                    self._current_frame()
                ):
                    return progressed
            for _ in range(DIALOGUE_TOP_BOX_PRESSES):
                self._capture_current_frame_dialogue_crop()
                record = self._take_action(MenuAction, menu_action=menu_action)
                drop_prefix_crops(captured_dialogue_crops([record]), self.dialogue_crops)
                if record.frame_changed or not is_same_battle_frame(record.frame_before, record.frame_after):
                    progressed = True
                if self._last_terminated or self._last_truncated:
                    return True
                if self._agent_state() != AgentState.IN_MENU or not self._is_top_box_dialogue():
                    self._settle_after_step()
                    return True
                entered_hangup = False
                for _ in range(DIALOGUE_TOP_BOX_SCROLL_WAITS):
                    if (self._agent_state() != AgentState.IN_MENU
                            or not self._is_top_box_dialogue()
                            or self._last_terminated
                            or self._last_truncated):
                        break
                    wait_rec = self._take_action(WaitAction)
                    drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                    if wait_rec.frame_changed or not is_same_battle_frame(wait_rec.frame_before, wait_rec.frame_after):
                        progressed = True
                    cur_frame = self._current_frame()
                    if not has_bottom_dialogue_box(cur_frame) or is_top_box_hangup_or_blank_bottom_box(cur_frame):
                        entered_hangup = True
                        break
                    if is_same_battle_frame(wait_rec.frame_before, wait_rec.frame_after):
                        break
                if self._last_terminated or self._last_truncated:
                    return True
                if self._agent_state() != AgentState.IN_MENU or not self._is_top_box_dialogue():
                    self._settle_after_step()
                    return True
                if entered_hangup:
                    if self._settle_top_box_hangup_or_ring():
                        progressed = True
                    if self._last_terminated or self._last_truncated:
                        return True
                    if self._agent_state() != AgentState.IN_MENU or not self._is_top_box_dialogue():
                        self._settle_after_step()
                        return True
                    if not has_bottom_dialogue_box(self._current_frame()) or is_top_box_hangup_or_blank_bottom_box(
                        self._current_frame()
                    ):
                        break
            if not progressed:
                btn_rec = self._take_action_string(f"button({button})")
                if btn_rec is not None:
                    drop_prefix_crops(captured_dialogue_crops([btn_rec]), self.dialogue_crops)
                    if btn_rec.frame_changed or not is_same_battle_frame(btn_rec.frame_before, btn_rec.frame_after):
                        progressed = True
                    if self._last_terminated or self._last_truncated:
                        return True
                    if self._agent_state() != AgentState.IN_MENU or not self._is_top_box_dialogue():
                        self._settle_after_step()
                        return True
                    wait_rec = self._take_action(WaitAction)
                    drop_prefix_crops(captured_dialogue_crops([wait_rec]), self.dialogue_crops)
                    if wait_rec.frame_changed or not is_same_battle_frame(wait_rec.frame_before, wait_rec.frame_after):
                        progressed = True
                    if self._last_terminated or self._last_truncated:
                        return True
                    if self._agent_state() != AgentState.IN_MENU or not self._is_top_box_dialogue():
                        self._settle_after_step()
                        return True
            return progressed
        if button != "b" and is_evolution_screen(self._current_frame()):
            progressed = False
            idle_evo = 0
            for _ in range(MAX_BATTLE_DIALOGUE_STEPS):
                before_evo = self._current_frame()
                btn_record = self._take_action_string("button(a)")
                if btn_record is None:
                    return progressed
                drop_prefix_crops(captured_dialogue_crops([btn_record]), self.dialogue_crops)
                if not is_same_battle_frame(before_evo, self._current_frame()):
                    progressed = True
                    idle_evo = 0
                else:
                    idle_evo += 1
                if self._last_terminated or self._last_truncated:
                    return True
                state = self._settle_after_step()
                if self._last_terminated or self._last_truncated or state != AgentState.IN_DIALOGUE:
                    return True
                if not is_evolution_screen(self._current_frame()) or idle_evo >= MAX_IDLE_BATTLE_DIALOGUE_STEPS:
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
    def __init__(
        self,
        env,
        task,
        max_steps: int = 10,
        initial_dialogue_crops: Optional[List[Any]] = None,
        defer_ocr_on_battle: bool = False,
        **kwargs,
    ):
        self.dialogue_crops: List[Any] = list(initial_dialogue_crops) if initial_dialogue_crops else []
        self.dialogue_text: str = ""
        self._defer_ocr_on_battle: bool = bool(defer_ocr_on_battle)
        super().__init__(env, task, max_steps, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _finish(self, code: int, reason: str) -> int:
        if self.dialogue_crops and not (self._defer_ocr_on_battle and reason == "left_dialogue_to_battle"):
            self.dialogue_text = "\n".join(ocr(self.dialogue_crops, self._vlm, do_merge=True,
                                               parameters=self._parameters))
            if is_battle_defeat_text([self.dialogue_text]):
                recorder = getattr(self._env, "record_defeat", None)
                if callable(recorder):
                    recorder()
        self.report.termination_reason = reason
        self.report.notes = self.dialogue_text or None
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
FREE_INTERRUPTED = -3
FREE_BUTTONS = ("a", "b", "start", "up", "down", "left", "right")


class FreeExecutor(_DialogueSettlingExecutor):
    MAX_NEW_TOKENS = 4000
    MAX_INVALID = 3
    STEP_TAG = "free_step"

    FIRST_STEP_PROMPT = """You are playing a Game Boy Pokemon game, pressing one button at a time.

What you have been asked to do: "[TASK]"

Stop when: [TERMINATION]

The image is the current screen.
Choose the single button press that best moves towards what you have been asked to do. Note: a 3-tile-tall bordered dialogue box at the bottom of the screen or a 4-tile-tall bordered dialogue box at the top of the screen means dialogue is open and must be advanced with a or b before walking, whereas a thin 2-tile-tall area/location name banner at the bottom of the screen is a non-blocking overlay that disappears automatically as you walk—do not press a or b to dismiss an area/location name banner when your task is to walk.

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
First decide, from this screen, whether the condition to stop at has been met (a 3-tile-tall bordered dialogue box at the bottom of the screen or a 4-tile-tall bordered dialogue box at the top of the screen—even if blank or mid-text—means dialogue is still open, whereas a thin 2-tile-tall area/location name banner is a non-blocking overlay that does not block walking and should not be dismissed with a or b). If it has not, choose the single button press that best moves towards it. Keep Reasoning concise (at most 3-4 sentences) and never repeat a direction listed as blocked at your current position.

The buttons are: a, b, start, up, down, left, right.

Respond in exactly this format:
Reasoning: <what is on the screen, and whether the stopping condition is met>
Complete: <yes or no>
Press: <one of a, b, start, up, down, left, right>
[STOP]"""

    DEFAULT_TERMINATION = "what you were asked to do has visibly happened on screen"

    MAX_REPEATED_BLOCKED_PRESSES = 2
    MAX_NO_DISPLACEMENT_DIRECTIONAL_PRESSES = 3

    def __init__(self, env, task, max_steps: int = 15, termination: Optional[str] = None, **kwargs):
        self.presses: List[str] = []
        self.press_outcomes: List[str] = []
        self.dialogue_crops: List[Any] = []
        self.dialogue_text: str = ""
        self._blocked_at_place: Dict[Any, List[str]] = {}
        self._zero_disp_at_place: Dict[Any, List[str]] = {}
        self._repeated_blocked_presses: int = 0
        self._no_displacement_directional_presses: int = 0
        self._last_was_turn_at_place: bool = False
        self.termination = (termination or "").strip() or self.DEFAULT_TERMINATION
        super().__init__(env, task, max_steps, termination=self.termination, **kwargs)

    def _fail(self, code: int, reason: str, notes: Optional[str] = None) -> int:
        self.report.termination_reason = reason
        self.report.notes = notes
        return code

    def _history_block(self) -> str:
        if not self.presses:
            return ""
        outcomes = self.press_outcomes if len(self.press_outcomes) == len(self.presses) else self.presses
        lines = [f"Buttons pressed so far, in order: {', '.join(outcomes)}."]
        cur_place = self._current_map_place()
        if cur_place is not None:
            facing = getattr(self._env, "facing", None)
            facing_note = f", facing {facing}" if facing in DIRECTIONS else ""
            lines.append(f"Current position: {cur_place[1]}{facing_note}.")
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
            if is_battle_defeat_text([self.dialogue_text]):
                recorder = getattr(self._env, "record_defeat", None)
                if callable(recorder):
                    recorder()
        self.report.termination_reason = reason
        cur_place = self._current_map_place()
        blocked_here = self._blocked_at_place.get(cur_place, []) if cur_place is not None else []
        blocked_note = f"Blocked moving {', '.join(blocked_here)} at {cur_place[1]}" if blocked_here else ""
        if not blocked_note and self._blocked_at_place:
            parts = [f"{', '.join(dirs)} at {place[1]}" for place, dirs in self._blocked_at_place.items() if dirs]
            if parts:
                blocked_note = f"Blocked moving {'; '.join(parts)}"
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

    def _record_blocked_direction(self, place: Tuple[int, Tuple[int, int]], direction: str) -> None:
        blocked_list = self._blocked_at_place.setdefault(place, [])
        if direction in blocked_list:
            self._repeated_blocked_presses += 1
        else:
            blocked_list.append(direction)
        if getattr(self._current_map_place(), "__eq__", lambda _: False)(place):
            recorder = getattr(self._env, "record_blocked_step", None)
            if callable(recorder):
                try:
                    recorder(direction=direction)
                except Exception:  # noqa: BLE001
                    pass

    def _clear_blocked_direction(self, place: Tuple[int, Tuple[int, int]], direction: str) -> None:
        blocked_list = self._blocked_at_place.get(place)
        if blocked_list and direction in blocked_list:
            blocked_list.remove(direction)
            if not blocked_list:
                self._blocked_at_place.pop(place, None)
        zero_list = self._zero_disp_at_place.get(place)
        if zero_list and direction in zero_list:
            zero_list.remove(direction)
        gmap = getattr(self._env, "_map", None)
        if gmap is not None and isinstance(place, tuple) and len(place) == 2:
            try:
                r_seg, r_pos = gmap.resolve(place[0], place[1])
                if 0 <= r_seg < len(getattr(gmap, "blocked_moves", ())):
                    dirs = gmap.blocked_moves[r_seg].get(r_pos)
                    if dirs and direction in dirs:
                        dirs.discard(direction)
                        if not dirs:
                            gmap.blocked_moves[r_seg].pop(r_pos, None)
            except Exception:  # noqa: BLE001
                pass

    def _is_wipe_or_transition_step(
        self,
        frame_after: Optional[np.ndarray],
        passed_frames: Optional[Any],
        step_status: Optional[str],
    ) -> bool:
        from execution.strategist.pokemon.global_map.registration import has_structure

        if step_status in ("blank", "lost", "not_free_roam"):
            return True
        gmap = getattr(self._env, "_map", None)
        if gmap is not None and (bool(getattr(gmap, "in_battle", False)) or bool(getattr(gmap, "faded", False))):
            return True
        frames_to_check = []
        if passed_frames is not None:
            for pf in passed_frames:
                if pf is not None:
                    frames_to_check.append(pf)
        if frame_after is not None:
            frames_to_check.append(frame_after)
        for f in frames_to_check:
            if not has_structure(f) or is_battle_screen(f):
                return True
            if gmap is not None:
                try:
                    if gmap._wipe_closing(f):
                        return True
                    if getattr(gmap, "pos", None) is not None and gmap._blacks_out(f, gmap.segment, gmap.pos):
                        return True
                except Exception:  # noqa: BLE001
                    pass
        return False

    @staticmethod
    def _is_directional_movement_task(task: str, termination: str = "") -> bool:
        if not task:
            return False
        lowered = task.lower()
        if not re.search(r"\b(?:up|down|left|right|north|south|east|west)\b", lowered):
            return False
        if re.search(
            r"\b(?:press(?:ing)?\s+(?:a|b|start)|button\s+(?:a|b|start)|talk|speak|interact|dialogue|phone|call|menu|dismiss|close|clear|event)\b",
            lowered,
        ):
            return False
        term_lowered = (termination or "").lower()
        if re.search(r"\b(?:dialogue|text\s+box|menu|banner\s+disappears)\b", term_lowered):
            return False
        return True

    def _is_dialogue_interruption_frame(self, frame: Optional[np.ndarray] = None) -> bool:
        fr = self._current_frame() if frame is None else frame
        state = self._agent_state()
        return (
            state == AgentState.IN_DIALOGUE
            or has_bottom_dialogue_box(fr)
            or (state == AgentState.IN_MENU and self._is_top_box_dialogue(fr))
        )

    @staticmethod
    def _repeated_walk_direction(task: str) -> Optional[str]:
        if not task:
            return None
        lowered = task.lower()
        found_dirs = [d for d in DIRECTIONS if re.search(rf"\b{d}\b", lowered)]
        if len(found_dirs) != 1:
            return None
        d = found_dirs[0]
        if re.search(
            rf"\b(?:press(?:ing)?\s+{d}\s+repeatedly|repeatedly\s+press(?:ing)?\s+{d}|walk\s+(?:north|south|east|west|{d})\b.*\bpress(?:ing)?\s+{d}|press(?:ing)?\s+{d}\s+to\s+walk)\b",
            lowered,
        ):
            return d
        return None

    @staticmethod
    def _player_sprite_turned(frame_before: Optional[np.ndarray], frame_after: Optional[np.ndarray]) -> bool:
        if frame_before is None or frame_after is None or frame_before.shape != frame_after.shape:
            return False
        a = (frame_before[:, :, 0] if frame_before.ndim == 3 else frame_before)[64:80, 64:80].astype(int)
        b = (frame_after[:, :, 0] if frame_after.ndim == 3 else frame_after)[64:80, 64:80].astype(int)
        return int((np.abs(a - b) > 40).sum()) >= 20

    def _step_press(self, press: str) -> bool:
        from execution.strategist.pokemon.global_map.registration import register

        state = self._agent_state()
        in_top_box_dialogue = state == AgentState.IN_MENU and self._is_top_box_dialogue()
        if press in ("a", "b") and (state == AgentState.IN_DIALOGUE or in_top_box_dialogue):
            self._advance_dialogue_box(press)
            self.press_outcomes.append(f"{press} (advanced dialogue)")
            return True
        if press in DIRECTIONS and in_top_box_dialogue:
            self._advance_dialogue_box("a")
            self.press_outcomes.append(f"{press} (blocked by open dialogue; advanced dialogue)")
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
            self.press_outcomes.append(press)
            return True
        step_ev = getattr(self._env, "last_step_event", None)
        step_status = getattr(step_ev, "status", None) if step_ev is not None else None
        passed = (
            record.transition_states[-1]["core"].get("transition_passed_frames")
            if record.transition_states
            else (step_ev.info.get("core", {}).get("transition_passed_frames") if (step_ev is not None and step_ev.info) else None)
        )
        scrolled_on_screen = False
        if (
            before_place is not None
            and press in DIRECTIONS
            and record.frame_before is not None
            and record.frame_after is not None
        ):
            if step_status == "moved":
                scrolled_on_screen = True
            else:
                step_reg = register(record.frame_before, record.frame_after, press, 1, passed_frames=passed)
                scrolled_on_screen = step_reg is not None and step_reg[0] != (0, 0)
        mid_place = self._current_map_place() if before_place is not None else None
        displaced = (before_place is not None and mid_place is not None and mid_place != before_place) or scrolled_on_screen
        if self._agent_state() == AgentState.FREE_ROAM:
            if press in ("a", "b"):
                box_or_emote = (
                    has_bottom_dialogue_box(record.frame_after)
                    or self._is_top_box_dialogue(record.frame_after)
                    or has_emote_bubble(record.frame_after)
                )
                cutscene_changed = (
                    (
                        passed is not None
                        and any(is_cutscene_frame_changing(record.frame_before, f) for f in passed)
                    )
                    or is_cutscene_frame_changing(record.frame_before, record.frame_after)
                )
                if (
                    box_or_emote
                    or cutscene_changed
                    or not has_bottom_location_banner(record.frame_before)
                ):
                    self._settle_after_step(settle_wait=FREE_SETTLE_WAIT)
            elif press in DIRECTIONS:
                box_or_emote = (
                    has_bottom_dialogue_box(record.frame_after)
                    or self._is_top_box_dialogue(record.frame_after)
                    or has_emote_bubble(record.frame_after)
                )
                cutscene_changed = not displaced and (
                    (
                        passed is not None
                        and any(
                            is_cutscene_frame_changing(record.frame_before, f, ignore_player_box=True) for f in passed
                        )
                    )
                    or is_cutscene_frame_changing(record.frame_before, record.frame_after, ignore_player_box=True)
                )
                if box_or_emote or cutscene_changed:
                    self._settle_after_step(settle_wait=FREE_SETTLE_WAIT)
        elif self._agent_state() == AgentState.IN_MENU:
            self._settle_state()
        if self._last_terminated or self._last_truncated:
            self.press_outcomes.append(press)
            return True
        after_place = self._current_map_place() if before_place is not None else None
        cur_frame = self._current_frame()
        cur_ev = getattr(self._env, "last_step_event", None)
        cur_status = getattr(cur_ev, "status", None) if cur_ev is not None else step_status
        wipe_or_transition = self._is_wipe_or_transition_step(cur_frame, passed, cur_status)
        still_free = (
            self._agent_state() == AgentState.FREE_ROAM
            and not has_bottom_dialogue_box(cur_frame)
            and not self._is_top_box_dialogue(cur_frame)
            and not is_battle_screen(cur_frame)
            and not wipe_or_transition
        )
        outcome_label = press
        if before_place is not None and after_place is not None and after_place != before_place:
            self._repeated_blocked_presses = 0
            self._no_displacement_directional_presses = 0
            self._last_was_turn_at_place = False
            self._clear_blocked_direction(before_place, press)
            outcome_label = f"{press} (moved {before_place[1]} -> {after_place[1]})"
        elif before_place is not None and scrolled_on_screen:
            self._repeated_blocked_presses = 0
            self._no_displacement_directional_presses = 0
            self._last_was_turn_at_place = False
            self._clear_blocked_direction(before_place, press)
            dx, dy = DIRECTIONS[press]
            est_after = (before_place[1][0] + dx, before_place[1][1] + dy)
            gmap = getattr(self._env, "_map", None)
            if gmap is not None and getattr(gmap, "pos", None) == before_place[1]:
                gmap.pos = est_after
                if 0 <= gmap.segment < len(gmap.visits):
                    gmap.visits[gmap.segment][est_after] += 1
            outcome_label = f"{press} (moved {before_place[1]} -> {est_after})"
        elif before_place is not None and still_free and after_place == before_place:
            zero_list = self._zero_disp_at_place.setdefault(before_place, [])
            blocked_here = self._blocked_at_place.get(before_place, [])
            sprite_turned = self._player_sprite_turned(record.frame_before, record.frame_after)
            is_known_blocked = (
                (self.presses and self.presses[-1] == press)
                or was_facing == press
                or not sprite_turned
                or press in blocked_here
            )
            already_tried = press in zero_list
            if not already_tried:
                zero_list.append(press)
            if is_known_blocked:
                self._no_displacement_directional_presses += 1
                self._last_was_turn_at_place = False
                self._record_blocked_direction(before_place, press)
                outcome_label = f"{press} (blocked at {before_place[1]}, 0 steps)"
            else:
                if already_tried or self._last_was_turn_at_place:
                    self._no_displacement_directional_presses += 1
                self._last_was_turn_at_place = True
                outcome_label = f"{press} (turned to face {press} at {before_place[1]} — press {press} again to step {press})"
        elif was_free and still_free and press in ("a", "b"):
            cur_p = self._current_map_place()
            facing_now = getattr(self._env, "facing", None)
            outcome_label = f"{press} (no dialogue or interaction triggered at {cur_p[1] if cur_p else 'current tile'})"
            if cur_p is not None and facing_now in DIRECTIONS and facing_now in self._blocked_at_place.get(cur_p, ()):
                self._no_displacement_directional_presses += 1
        self.press_outcomes.append(outcome_label)
        return True

    def _execute(self) -> int:
        self._last_terminated = False
        self._last_truncated = False
        consecutive_invalid = 0
        walk_dir = self._repeated_walk_direction(self._task or "")
        initial_frame = self._get_state()["core"]["current_frame"]
        started_in_free_roam = (
            self._agent_state() == AgentState.FREE_ROAM
            and not has_bottom_dialogue_box(initial_frame)
            and not self._is_top_box_dialogue(initial_frame)
            and not is_battle_screen(initial_frame)
        )
        is_directional_walk = started_in_free_roam and self._is_directional_movement_task(
            self._task or "", self.termination
        )
        for _ in range(self._max_steps):
            frame = self._get_state()["core"]["current_frame"]
            if self.presses and (self._effective_state(frame) == AgentState.IN_BATTLE or is_battle_screen(frame)):
                return self._finish(FREE_OK, "termination_met")
            if self.presses and is_directional_walk and self._is_dialogue_interruption_frame(frame):
                self._capture_current_frame_dialogue_crop(frame)
                return self._finish(FREE_INTERRUPTED, "move_interrupted")
            template = self.STEP_PROMPT if self.presses else self.FIRST_STEP_PROMPT
            prompt = (template.replace("[TASK]", self._task.strip())
                      .replace("[TERMINATION]", self.termination)
                      .replace("[HISTORY]", self._history_block()))
            output = self._vlm_call(self.STEP_TAG, texts=prompt, images=[frame],
                                    max_new_tokens=self.MAX_NEW_TOKENS)
            if self.presses and parse_yes_no(output, "Complete"):
                if (
                    walk_dir is not None
                    and len(self.presses) == 1
                    and self._last_was_turn_at_place
                    and self._agent_state() == AgentState.FREE_ROAM
                ):
                    press = walk_dir
                else:
                    return self._finish(FREE_OK, "termination_met")
            else:
                answer = (parse_key_value(output, "Press") or "").strip().lower().strip(".")
                press = answer.split()[0] if answer else ""
            if press not in FREE_BUTTONS:
                if not self.presses:
                    press = self._default_first_press(self._task) or ""
                elif self._repeated_blocked_presses > 0 or self._no_displacement_directional_presses >= 2:
                    return self._finish(FREE_OUT_OF_STEPS, "out_of_steps")
            place_before = self._current_map_place()
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
            after_frame = self._get_state()["core"]["current_frame"]
            if self._effective_state(after_frame) == AgentState.IN_BATTLE or is_battle_screen(after_frame):
                return self._finish(FREE_OK, "termination_met")
            if is_directional_walk and press in DIRECTIONS and self._is_dialogue_interruption_frame(after_frame):
                self._capture_current_frame_dialogue_crop(after_frame)
                return self._finish(FREE_INTERRUPTED, "move_interrupted")
            if (
                walk_dir is not None
                and press == walk_dir
                and len(self.presses) < self._max_steps
                and self._agent_state() == AgentState.FREE_ROAM
                and not has_bottom_dialogue_box(after_frame)
                and not self._is_top_box_dialogue(after_frame)
            ):
                batch_limit = 3 if self._last_was_turn_at_place else 2
                for _ in range(batch_limit):
                    if len(self.presses) >= self._max_steps:
                        break
                    cur_p = self._current_map_place()
                    if cur_p is not None and press in self._blocked_at_place.get(cur_p, ()):
                        break
                    if not self._step_press(press):
                        break
                    self.presses.append(press)
                    if self._last_terminated:
                        return self._finish(FREE_TERMINATED, "terminated")
                    if self._last_truncated:
                        return self._finish(FREE_TRUNCATED, "truncated")
                    after_frame = self._get_state()["core"]["current_frame"]
                    if (
                        self._effective_state(after_frame) != AgentState.FREE_ROAM
                        or is_battle_screen(after_frame)
                        or has_bottom_dialogue_box(after_frame)
                        or self._is_top_box_dialogue(after_frame)
                    ):
                        break
                    new_p = self._current_map_place()
                    if new_p == cur_p and not self._last_was_turn_at_place:
                        break
            if self._effective_state(after_frame) == AgentState.IN_BATTLE or is_battle_screen(after_frame):
                return self._finish(FREE_OK, "termination_met")
            if is_directional_walk and press in DIRECTIONS and self._is_dialogue_interruption_frame(after_frame):
                self._capture_current_frame_dialogue_crop(after_frame)
                return self._finish(FREE_INTERRUPTED, "move_interrupted")
            if (
                self._repeated_blocked_presses >= self.MAX_REPEATED_BLOCKED_PRESSES
                or self._no_displacement_directional_presses >= self.MAX_NO_DISPLACEMENT_DIRECTIONAL_PRESSES
            ):
                return self._finish(FREE_OUT_OF_STEPS, "out_of_steps")
        return self._finish(FREE_OUT_OF_STEPS, "out_of_steps")
