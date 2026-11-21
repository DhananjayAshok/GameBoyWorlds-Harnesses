import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

from execution.game_api import AgentState, WaitAction

from execution.executors import Executor
from execution.perception.pokemon.battle import (has_bottom_dialogue_box, has_modal_preview_box,
                                                 is_cutscene_frame_changing)
from execution.perception.pokemon.navigation import cell_kinds, verbalize_edges
from execution.perception.pokemon.tiles import TileRecognizer, describe_state_screen, verbalize_tiles
from execution.pokemon.executors import (CombatExecutor, DialogueExecutor, FreeExecutor,
                                         InteractionExecutor, MenuExecutor,
                                         BATTLE_ENDED, BATTLE_LEFT_TO_DIALOGUE, BATTLE_LEFT_TO_MENU,
                                         DIALOGUE_CLEARED, DIALOGUE_LEFT_TO_BATTLE, DIALOGUE_LEFT_TO_MENU,
                                         FREE_OK, LOCATION_FAIL_CODES,
                                         MENU_LEFT_TO_BATTLE, MENU_LEFT_TO_DIALOGUE, MENU_OK,
                                         MOVE_OK, MOVE_TARGET_AMBIGUOUS,
                                         OffScreenMoveExecutor, OnScreenMoveExecutor)
from execution.pokemon.prompts import (PLAYTHROUGH_ACHIEVABLE_PROMPT, PLAYTHROUGH_AFTER_BLOCK,
                                       PLAYTHROUGH_ATTEMPTS_BLOCK,
                                       PLAYTHROUGH_BATTLE_INSTRUCTIONS_PROMPT, PLAYTHROUGH_BATTLE_TIPS_BLOCK,
                                       PLAYTHROUGH_CLARIFY_TARGET_PROMPT, PLAYTHROUGH_GUIDANCE_BLOCK,
                                       PLAYTHROUGH_JUDGE_PROMPT, PLAYTHROUGH_LOG_BLOCK,
                                       PLAYTHROUGH_MENU_OBJECTIVE_PROMPT, PLAYTHROUGH_PLAN_PROMPT,
                                       PLAYTHROUGH_REMAINING_SUBGOALS_BLOCK, PLAYTHROUGH_REMAINING_SUBGOALS_LINE,
                                       PLAYTHROUGH_SUBGOAL_DECISION_PROMPT, PLAYTHROUGH_TASK_CHECK_PROMPT,
                                       PLAYTHROUGH_TEXT_BLOCK, PRIMITIVES)
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
    MAX_REPLANS = 2

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
        self.last_executor = None
        self.plan: List[str] = []
        super().__init__(task=task, executor_class=None, env=env, game=game,
                         max_steps=max_steps, **kwargs)
        self._tile_recognizer.set_vlm_call(self._vlm_caller("perception"))

    def _run_config(self) -> dict:
        config = super()._run_config()
        config["guidance"] = self._guidance
        config["battle_tips"] = list(self._battle_tips)
        config["tile_recognizer"] = self._tile_recognizer.name
        config["episode_number"] = self._episode_number
        config["max_executor_calls"] = self._max_steps
        config["max_subgoal_attempts"] = self.MAX_SUBGOAL_ATTEMPTS
        config["max_replans"] = self.MAX_REPLANS
        return config

    def _current_frame(self):
        return self._env.get_info()["core"]["current_frame"]

    def _agent_state(self):
        return self._env.get_info()["pokemon_core"]["agent_state"]

    def _is_in_dialogue(self, frame=None) -> bool:
        state = self._agent_state()
        if state == AgentState.IN_DIALOGUE:
            return True
        if state == AgentState.FREE_ROAM:
            frame = self._current_frame() if frame is None else frame
            return has_bottom_dialogue_box(frame) or has_modal_preview_box(frame)
        return False

    def _perceive(self, frame) -> str:
        state = self._agent_state()
        if state != AgentState.FREE_ROAM:
            self._identified = {}
            self._screen_tiles = describe_state_screen(state)
            return self._screen_tiles
        if has_bottom_dialogue_box(frame) or has_modal_preview_box(frame):
            self._identified = {}
            self._screen_tiles = describe_state_screen(AgentState.IN_DIALOGUE)
            return self._screen_tiles
        self._tile_recognizer.record_tiles(frame, self._episode_number)
        self._identified = self._tile_recognizer.identify_tiles(frame)
        people = self._people_now()
        kinds = cell_kinds(self._identified, getattr(self._env, "ledge_patterns", None),
                           getattr(self._env, "walk_patterns", None), people)
        self._screen_tiles = (verbalize_tiles(self._identified, people)
                              + "\nWalkable cells on each edge of the screen:\n"
                              + verbalize_edges(kinds))
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
        raw_state = AgentState.IN_DIALOGUE if self._is_in_dialogue() else self._agent_state()
        state = raw_state.name.lower().replace("_", " ")
        return (prompt.replace("[GAME]", self._game)
                .replace("[TASK]", self._task)
                .replace("[GUIDANCE_BLOCK]", guidance_block)
                .replace("[LOG_BLOCK]", self._log_block())
                .replace("[SCREEN_TILES]", self._screen_tiles)
                .replace("[AGENT_STATE]", state)
                .replace("[PRIMITIVES]", PRIMITIVES)
                .replace("[PLAN]", plan))

    def call_executor(self, tag: str, task: str, *, hint: Optional[str] = None,
                      allow_self_termination: bool = False,
                      max_steps: Optional[int] = None, **executor_kwargs: Any) -> Any:
        if tag not in self.EXECUTORS:
            log_error(f"Unknown executor tag {tag!r}. Options: {list(self.EXECUTORS)}",
                      self._parameters)
        run_kwargs = dict(self._executor_kwargs)
        run_kwargs.update(executor_kwargs)
        run_kwargs["hint"] = hint
        run_kwargs["allow_self_termination"] = allow_self_termination
        self._current_tag = tag
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
               subgoal_index: int = 0) -> Tuple[bool, str, int]:
        report = result["report"]
        before = report.initial_state["core"]["current_frame"]
        after = report.final_state["core"]["current_frame"]
        after_state = report.final_state["pokemon_core"]["agent_state"]
        if after_state == AgentState.FREE_ROAM and (has_bottom_dialogue_box(after) or has_modal_preview_box(after)):
            after_state = AgentState.IN_DIALOGUE
        if after_state == AgentState.FREE_ROAM:
            self._tile_recognizer.record_tiles(after, self._episode_number)
            after_identified = self._tile_recognizer.identify_tiles(after)
            after_people = self._people_now()
            after_kinds = cell_kinds(after_identified, getattr(self._env, "ledge_patterns", None),
                                     getattr(self._env, "walk_patterns", None), after_people)
            after_tiles = (verbalize_tiles(after_identified, after_people)
                           + "\nWalkable cells on each edge of the screen:\n"
                           + verbalize_edges(after_kinds))
            after_block = PLAYTHROUGH_AFTER_BLOCK.replace("[AFTER_TILES]", after_tiles)
        else:
            after_block = PLAYTHROUGH_AFTER_BLOCK.replace("[AFTER_TILES]", describe_state_screen(after_state))
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
        state = self._agent_state().name.lower().replace("_", " ")
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

    def _combat(self, instructions: str) -> Any:
        task = instructions + (f"\nYour battle tips:\n{self._tips_text()}" if self._battle_tips else "")
        return self.call_executor("combat", task)

    def _battles_lost(self) -> int:
        return self._env.get_info()["pokemon_core"].get("n_battles_lost", 0) - self._losses_at_start

    def _run_action(self, action: str, argument: Optional[str], frame) -> Tuple[Dict[str, Any], str]:
        tag, _, default_task = ACTIONS[action]
        task = argument or default_task
        if tag in MOVE_TAGS:
            result = self._move(tag, task)
            clarifications = 0
            while (action == "MoveTo" and result.get("ambiguous")
                   and clarifications < self.MAX_CLARIFICATIONS):
                clarified = self._clarify_target(task, result["notes"], frame)
                if clarified is None:
                    break
                clarifications += 1
                task = clarified
                result = self._move(tag, task)
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

    def _move(self, tag: str, task: str) -> Any:
        kinds = cell_kinds(self._identified, getattr(self._env, "ledge_patterns", None),
                           getattr(self._env, "walk_patterns", None), self._people_now())
        return self.call_executor(tag, task, screen_tiles=self._screen_tiles,
                                  cell_kinds=kinds,
                                  identified=self._identified)

    def _handle_interruption(self) -> Tuple[str, Dict[str, Any]]:
        frame = self._current_frame()
        state = AgentState.IN_DIALOGUE if self._is_in_dialogue(frame) else self._agent_state()
        if state == AgentState.IN_DIALOGUE:
            handled = self.call_executor("dialogue", "clear the dialogue that interrupted what the player was doing")
        elif state == AgentState.IN_BATTLE:
            handled = self._combat(self._battle_instructions(frame))
            if (handled.get("succeeded")
                    and self._agent_state() == AgentState.IN_DIALOGUE
                    and not self._out_of_calls()
                    and not self._env_done()):
                follow_up = self.call_executor("dialogue", "clear the dialogue that followed the battle")
                combined_text = "\n".join(filter(None, [handled.get("dialogue"), follow_up.get("dialogue")]))
                if combined_text:
                    handled["dialogue"] = combined_text
                handled["report"].final_state = follow_up["report"].final_state
        elif state == AgentState.IN_MENU:
            objective, is_start_menu = self._menu_objective(
                "deal with the menu that opened while the player was doing something else", frame)
            handled = self.call_executor("menu", objective, is_start_menu=is_start_menu)
            if (handled.get("code") == MENU_LEFT_TO_DIALOGUE
                    and self._agent_state() == AgentState.IN_DIALOGUE
                    and not self._out_of_calls()
                    and not self._env_done()):
                follow_up = self.call_executor("dialogue", "clear the dialogue that followed the menu")
                combined_text = "\n".join(filter(None, [handled.get("dialogue"), follow_up.get("dialogue")]))
                if combined_text:
                    follow_up["dialogue"] = combined_text
                handled = follow_up
        else:
            handled = self.call_executor("free", self.CLEAR_EVENT_INSTRUCTION,
                                         termination=self.CLEAR_EVENT_TERMINATION)
        return state.name.lower(), handled

    def _note(self, entry: str) -> None:
        self.report.narrative.append(entry)

    def _out_of_calls(self) -> bool:
        return self._calls >= self._max_steps

    def _env_done(self) -> bool:
        return bool(self._env._emulator.check_if_done())

    def _settle_free_roam(self, ticks: int = 2, max_cutscene_ticks: int = 16, post_motion_ticks: int = 2) -> None:
        waited = 0
        post_motion_remaining = 0
        while (self._agent_state() == AgentState.FREE_ROAM
               and not self._env_done()
               and waited < max_cutscene_ticks):
            before = self._current_frame()
            box_open = has_bottom_dialogue_box(before)
            if waited >= ticks and not box_open and post_motion_remaining <= 0:
                break
            self._env.step_high_level_action(WaitAction)
            waited += 1
            if self._agent_state() != AgentState.FREE_ROAM or self._env_done():
                break
            after = self._current_frame()
            if is_cutscene_frame_changing(before, after) or has_bottom_dialogue_box(after):
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
        handled = self.call_executor("dialogue", "clear the dialogue on screen")
        note = f"Cleared open dialogue ({handled['reason']}){self._dialogue_snippet(handled)}"
        self._note(note)
        if attempts is not None:
            attempts.append(note)
        return handled

    def _pursue(self, subgoal: str, subgoal_index: int = 0) -> Tuple[str, List[str], int]:
        attempts: List[str] = []
        for _ in range(self.MAX_SUBGOAL_ATTEMPTS):
            if self._env_done():
                return "env_done", attempts, subgoal_index
            if self._out_of_calls():
                return "out_of_calls", attempts, subgoal_index
            if self._calls > 0:
                self._settle_free_roam()
            if self._is_in_dialogue():
                before = self._current_frame()
                self._perceive(before)
                handled = self._clear_pending_dialogue(attempts)
                if self._battles_lost():
                    self._note("the player lost a battle and blacked out")
                    return "lost_battle", attempts, subgoal_index
                if self._env_done():
                    return "env_done", attempts, subgoal_index
                if handled and handled.get("succeeded"):
                    complete, summary, reached = self._judge(
                        subgoal, "Dialogue(clear open dialogue)", handled, subgoal_index)
                    self._note(summary)
                    attempts.append(f"Dialogue(): {summary}")
                    if complete:
                        return "complete", attempts, reached
                if self._out_of_calls():
                    return "out_of_calls", attempts, subgoal_index
            frame = self._current_frame()
            self._perceive(frame)
            decision = self._decide(subgoal, attempts, frame)
            if decision is None:
                attempts.append("the reply could not be read, so nothing was done")
                continue
            action, argument = decision
            result, task = self._run_action(action, argument, frame)
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
                    and self._is_in_dialogue()
                    and not self._out_of_calls()
                    and not self._env_done()):
                follow_up = self.call_executor("dialogue", "clear the dialogue that followed the action")
                combined_text = "\n".join(filter(None, [result.get("dialogue"), follow_up.get("dialogue")]))
                if combined_text:
                    result["dialogue"] = combined_text
                result["report"].final_state = follow_up["report"].final_state
                if result["tag"] == "menu":
                    follow_up["report"].initial_state = result["report"].initial_state
                    if combined_text:
                        follow_up["dialogue"] = combined_text
                    result = follow_up

            if not result["succeeded"] and (self._agent_state() != AgentState.FREE_ROAM or self._is_in_dialogue()):
                state, handled = self._handle_interruption()
                handled["report"].initial_state = result["report"].initial_state
                note = (f"{request} was interrupted while the game was in {state}; "
                        f"dealt with it ({handled['reason']}){self._dialogue_snippet(handled)}")
                self._note(note)
                attempts.append(note)
                if self._battles_lost():
                    self._note("the player lost a battle and blacked out")
                    return "lost_battle", attempts, subgoal_index
                if handled.get("succeeded"):
                    complete, summary, reached = self._judge(
                        subgoal, f"{request} (cleared {state})", handled, subgoal_index)
                    self._note(summary)
                    attempts.append(f"{request}: {summary}")
                    if complete:
                        return "complete", attempts, reached
                continue

            if not result["succeeded"]:
                note = f"{request} did not work: {result['reason']}" + (
                    f" ({result['notes']})" if result.get("notes") else "")
                self._note(note)
                attempts.append(note)
                continue

            complete, summary, reached = self._judge(subgoal, request, result, subgoal_index)
            self._note(summary)
            attempts.append(f"{request}: {summary}")
            if complete:
                return "complete", attempts, reached
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

    def _finish(self, status: str, reason: Optional[str] = None, summary: Optional[str] = None) -> dict:
        self.report.digest = self._digest(status, reason, summary)
        return {
            "status": status,
            "reason": reason,
            "summary": summary,
            "plan": list(self.plan),
            "narrative": list(self.report.narrative),
            "executor_calls": self._calls,
        }

    def _evaluate(self) -> Optional[dict]:
        self._settle_free_roam()
        self._clear_pending_dialogue()
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
                    return self._finish("out_of_calls", f"Used all {self._max_steps} executor calls.", summary)
                self._settle_free_roam()
                if self._is_in_dialogue() and not self._out_of_calls() and not self._env_done():
                    self._clear_pending_dialogue()
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
                return self._finish("out_of_calls", f"Used all {self._max_steps} executor calls.")

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
