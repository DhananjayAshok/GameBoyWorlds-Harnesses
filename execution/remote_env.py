import io
import os
import sys
from typing import Any, Dict, Optional, Tuple, Type

import numpy as np
from PIL import Image

from execution.game_api import (ACTION_CLASSES, AgentState, BattleMenuAction, InteractAction, LowLevelAction,
                                LowLevelActions, MenuAction, MoveStepsAction, OpenMenuAction, PassDialogueAction,
                                PickAttackAction, WaitAction)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent_workspace"))
from envclient import Client  # noqa: E402

BUTTONS = {"a": LowLevelActions.PRESS_BUTTON_A, "b": LowLevelActions.PRESS_BUTTON_B,
           "start": LowLevelActions.PRESS_BUTTON_START, "up": LowLevelActions.PRESS_ARROW_UP,
           "down": LowLevelActions.PRESS_ARROW_DOWN, "left": LowLevelActions.PRESS_ARROW_LEFT,
           "right": LowLevelActions.PRESS_ARROW_RIGHT}
BUTTON_NAMES = {button: name for name, button in BUTTONS.items()}
DIRECTIONS = ("up", "down", "left", "right")
BATTLE_OPTIONS = ("fight", "pokemon", "bag", "run", "progress")
MENU_OPTIONS = ("up", "down", "left", "right", "confirm", "back")
LAST_ACTION_FIELDS = ("action_return", "transition_passed_frames", "previous_action_details")


def to_frame(png: bytes) -> np.ndarray:
    frame = np.array(Image.open(io.BytesIO(png)))
    return frame[:, :, None] if frame.ndim == 2 else frame


def to_frames(pngs) -> Optional[np.ndarray]:
    return np.stack([to_frame(png) for png in pngs]) if pngs else None


def parse_action(text: str) -> Tuple[Optional[Type], Optional[dict]]:
    text = text.lower().strip()
    if "(" not in text or ")" not in text:
        return None, None
    name = text.split("(")[0].strip()
    args = text.split("(")[1].split(")")[0].strip()
    if name == "button":
        button = BUTTONS.get(args)
        return (LowLevelAction, {"low_level_action": button}) if button is not None else (None, None)
    if name == "wait":
        return WaitAction, {}
    if name == "interact":
        return InteractAction, {}
    if name == "passdialogue":
        return PassDialogueAction, {}
    if name == "battlemenu":
        return (BattleMenuAction, {"option": args}) if args in BATTLE_OPTIONS else (None, None)
    if name == "pickattack":
        if not args.isnumeric() or not 1 <= int(args) <= 4:
            return None, None
        return PickAttackAction, {"option": int(args)}
    if name == "menu":
        return (MenuAction, {"menu_action": args}) if args in MENU_OPTIONS else (None, None)
    if name == "move":
        found = [d for d in DIRECTIONS if d in args]
        if len(found) != 1:
            return None, None
        steps = args.replace(found[0], "").strip()
        if not steps.isnumeric():
            return None, None
        return MoveStepsAction, {"direction": found[0], "steps": int(steps)}
    if name == "openmenu":
        return OpenMenuAction, {"option": args}
    return None, None


def format_action(action: Type, kwargs: Dict[str, Any]) -> str:
    name = action.__name__
    if name == LowLevelAction.__name__:
        return f"button({BUTTON_NAMES[kwargs['low_level_action']]})"
    if name == MoveStepsAction.__name__:
        return f"move({kwargs['direction']} {int(kwargs['steps'])})"
    if name == BattleMenuAction.__name__:
        return f"battlemenu({kwargs['option']})"
    if name == PickAttackAction.__name__:
        return f"pickattack({int(kwargs['option'])})"
    if name == MenuAction.__name__:
        return f"menu({kwargs['menu_action']})"
    if name == OpenMenuAction.__name__:
        return f"openmenu({kwargs['option']})"
    no_args = {WaitAction.__name__: "wait()", InteractAction.__name__: "interact()",
               PassDialogueAction.__name__: "passdialogue()"}
    if name in no_args:
        return no_args[name]
    raise ValueError(f"{name} has no action string on the environment server")


class RemoteEmulator:
    def __init__(self, env: "RemoteEnvironment") -> None:
        self._env = env
        self.session_path = None

    @property
    def step_count(self) -> int:
        return self._env.steps

    def check_if_done(self) -> bool:
        return self._env.done


class RemoteEnvironment:
    def __init__(self, session, owned: bool = False) -> None:
        self._session = session
        self._owned = owned
        self._emulator = RemoteEmulator(self)
        self.described = session.describe()
        self.game = self.described["game"]
        self._named_saves = bool(self.described.get("named_saves"))
        self._absorb(session.observe(), frame_changed=False)

    @classmethod
    def attach(cls, session_id: Optional[str] = None) -> "RemoteEnvironment":
        session_id = session_id or os.environ.get("ENVCTL_SESSION")
        if not session_id:
            raise RuntimeError("no session to attach to: pass one or set $ENVCTL_SESSION")
        return cls(Client.from_config().attach(session_id))

    @classmethod
    def open(cls, game: str, run: Optional[str] = None) -> "RemoteEnvironment":
        session = Client.from_config().open(task_id=f"gb.{game}.playthrough", run=run)
        try:
            session.reset()
            return cls(session, owned=True)
        except Exception:
            session.close()
            raise

    def _absorb(self, out: dict, *, frame_changed: bool) -> None:
        from execution.perception.pokemon.battle import is_false_overworld_menu_state

        obs = out["obs"]
        signals = obs.get("signals") or {}
        self.steps = int(obs.get("steps") or 0)
        self.done = bool(out.get("done"))
        self.success = out.get("success")
        self._actions = obs.get("actions") or {}
        last_action = {key: value for key, value in getattr(self, "_info", {}).get("core", {}).items()
                       if key in LAST_ACTION_FIELDS}
        current_frame = to_frame(obs["frame"])
        raw_state = AgentState[signals["agent_state"]]
        agent_state = (
            AgentState.FREE_ROAM
            if raw_state == AgentState.IN_MENU and is_false_overworld_menu_state(current_frame)
            else raw_state
        )
        self._info = {"core": {**last_action, "current_frame": current_frame,
                               "frame_changed": frame_changed},
                      "pokemon_core": {"agent_state": agent_state,
                                       "raw_agent_state": raw_state,
                                       "n_battles_lost": int(signals.get("n_battles_lost") or 0)}}

    def get_info(self) -> Dict[str, Dict[str, Any]]:
        return {group: dict(values) if isinstance(values, dict) else values for group, values in self._info.items()}

    def get_observation(self) -> np.ndarray:
        return self._info["core"]["current_frame"]

    def get_action_strings(self, return_all: bool = False) -> Dict[Type, str]:
        unknown = sorted(set(self._actions) - set(ACTION_CLASSES))
        if unknown:
            raise RuntimeError(f"the server offers actions this harness cannot take: {unknown}")
        return {ACTION_CLASSES[name]: text for name, text in self._actions.items()}

    def string_to_high_level_action(self, input_str: str) -> Tuple[Optional[Type], Optional[dict]]:
        return parse_action(input_str)

    def step_str(self, input_str: str):
        action, kwargs = parse_action(input_str)
        if action is None:
            return None, None, None, None, None
        return self.step_high_level_action(action, **kwargs)

    def _emulate_move_steps_in_false_menu(
        self,
        direction: str,
        target_steps: int,
        initial_out: Optional[dict] = None,
    ) -> Tuple[dict, dict, int]:
        from execution.perception.pokemon.battle import (
            has_bottom_dialogue_box,
            has_emote_bubble,
            has_top_dialogue_box,
            is_battle_screen,
            is_blank_or_fade_frame,
            is_false_overworld_menu_state,
        )
        from execution.strategist.pokemon.global_map.registration import _sprite_turned, has_structure, register

        all_pngs = []
        all_ocr = []
        any_frame_changed = False
        taken = 0
        rotated = False
        out = initial_out
        if out is not None and out.get("valid"):
            tr0 = out["obs"].get("transition") or {}
            ar0 = tr0.get("action_return") or {}
            taken = int(ar0.get("n_steps_taken") or 0)
            rotated = bool(ar0.get("rotated"))
            any_frame_changed = bool(tr0.get("frame_changed"))
            if tr0.get("frames"):
                all_pngs.extend(tr0["frames"])
            if tr0.get("ocr"):
                all_ocr.extend(tr0["ocr"])
            cur_frame = to_frame(out["obs"]["frame"])
        else:
            cur_frame = self._info["core"]["current_frame"]

        interrupted = False
        dx, dy = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}.get(direction, (0, 0))
        while taken < target_steps:
            prev_frame = cur_frame
            btn_out = self._session.step(f"button({direction})")
            out = btn_out
            if not btn_out.get("valid"):
                break
            tr = btn_out["obs"].get("transition") or {}
            if tr.get("frame_changed"):
                any_frame_changed = True
            btn_pngs = tr.get("frames") or []
            if btn_pngs:
                all_pngs.extend(btn_pngs)
            all_pngs.append(btn_out["obs"]["frame"])
            if tr.get("ocr"):
                all_ocr.extend(tr["ocr"])
            next_frame = to_frame(btn_out["obs"]["frame"])
            passed_arr = to_frames(btn_pngs) if btn_pngs else None
            warped = (
                not has_structure(next_frame)
                or is_blank_or_fade_frame(next_frame)
                or (passed_arr is not None and any(not has_structure(f) for f in passed_arr))
            )
            step_moved = False
            if warped:
                step_moved = True
            else:
                reg = register(prev_frame, next_frame, direction, 1, passed_frames=passed_arr)
                if reg is not None and (reg[0][0] * dx + reg[0][1] * dy) > 0:
                    step_moved = True
            if step_moved:
                taken += 1
            elif taken == 0 and not rotated and _sprite_turned(prev_frame, next_frame):
                rotated = True
                if target_steps == 1:
                    cur_frame = next_frame
                    break
            else:
                cur_frame = next_frame
                break
            cur_frame = next_frame
            if bool(btn_out.get("done")) or warped:
                break
            sig = btn_out["obs"].get("signals") or {}
            raw_now = AgentState[sig["agent_state"]]
            if (
                (raw_now != AgentState.FREE_ROAM and not (raw_now == AgentState.IN_MENU and is_false_overworld_menu_state(cur_frame)))
                or has_bottom_dialogue_box(cur_frame)
                or has_top_dialogue_box(cur_frame)
                or has_emote_bubble(cur_frame)
                or is_battle_screen(cur_frame)
            ):
                interrupted = True
                break

        action_return = {"n_steps_taken": taken, "rotated": rotated}
        if interrupted:
            action_success = 2
        elif taken == target_steps or (target_steps == 1 and rotated):
            action_success = 1
        else:
            action_success = -1
        synth_transition = {
            "frame_changed": any_frame_changed,
            "action_return": action_return,
            "frames": all_pngs if all_pngs else None,
            "ocr": all_ocr if all_ocr else None,
        }
        return out, synth_transition, action_success

    def step_high_level_action(self, action: Type, **kwargs):
        from execution.perception.pokemon.battle import (
            has_bottom_dialogue_box,
            has_modal_preview_box,
            has_top_dialogue_box,
            is_false_overworld_menu_state,
        )

        is_move = action is MoveStepsAction or (isinstance(action, type) and issubclass(action, MoveStepsAction))
        is_interact = action is InteractAction or (isinstance(action, type) and issubclass(action, InteractAction))
        is_open_menu = action is OpenMenuAction or (isinstance(action, type) and issubclass(action, OpenMenuAction))
        p_core = getattr(self, "_info", {}).get("pokemon_core", {})
        in_false_menu = (
            p_core.get("raw_agent_state") == AgentState.IN_MENU
            and p_core.get("agent_state") == AgentState.FREE_ROAM
        )
        if is_move and in_false_menu:
            direction = kwargs.get("direction", "up")
            steps = int(kwargs.get("steps") or 1)
            out, transition, action_success = self._emulate_move_steps_in_false_menu(direction, steps)
            self._absorb(out, frame_changed=bool(transition.get("frame_changed")))
            action_return = transition.get("action_return")
            states = [{"core": {}, "ocr": {"ocr_regions": {kind: to_frames(crops) for kind, crops in ocr.items()
                                                           if crops}}}
                      for ocr in (transition.get("ocr") or [{}])]
            if action_return is not None:
                states[-1]["core"]["action_return"] = action_return
            self._info["core"].update({
                "action_return": action_return,
                "transition_passed_frames": to_frames(transition.get("frames")),
                "previous_action_details": (action, kwargs, states, action_success, action_return),
            })
            info = self.get_info()
            terminated = self.done and self.success == 1.0
            return self.get_observation(), 0.0, terminated, self.done and not terminated, info

        out = self._session.step(format_action(action, kwargs))
        if not out.get("valid") and p_core.get("agent_state") == AgentState.FREE_ROAM:
            if is_move:
                direction = kwargs.get("direction", "up")
                steps = int(kwargs.get("steps") or 1)
                out, transition, action_success = self._emulate_move_steps_in_false_menu(direction, steps)
                self._absorb(out, frame_changed=bool(transition.get("frame_changed")))
                action_return = transition.get("action_return")
                states = [{"core": {}, "ocr": {"ocr_regions": {kind: to_frames(crops) for kind, crops in ocr.items()
                                                               if crops}}}
                          for ocr in (transition.get("ocr") or [{}])]
                if action_return is not None:
                    states[-1]["core"]["action_return"] = action_return
                self._info["core"].update({
                    "action_return": action_return,
                    "transition_passed_frames": to_frames(transition.get("frames")),
                    "previous_action_details": (action, kwargs, states, action_success, action_return),
                })
                info = self.get_info()
                terminated = self.done and self.success == 1.0
                return self.get_observation(), 0.0, terminated, self.done and not terminated, info
            if is_interact:
                out = self._session.step("button(a)")
                if out.get("valid"):
                    af = to_frame(out["obs"]["frame"])
                    raw_af = AgentState[(out["obs"].get("signals") or {}).get("agent_state")]
                    if (
                        raw_af != AgentState.IN_MENU
                        or not is_false_overworld_menu_state(af)
                        or has_bottom_dialogue_box(af)
                        or has_modal_preview_box(af)
                        or has_top_dialogue_box(af)
                    ):
                        out["obs"]["action_success"] = 1
            elif is_open_menu:
                out = self._session.step("button(start)")
                if out.get("valid"):
                    out["obs"]["action_success"] = 0

        if not out.get("valid"):
            self._absorb(out, frame_changed=False)
            self._info["invalid_action"] = True
            self._info["core"].update({
                "action_return": None,
                "transition_passed_frames": None,
                "previous_action_details": (action, kwargs, [], -1, None),
            })
            return self.get_observation(), 0.0, False, False, self.get_info()

        if is_move and out["obs"].get("action_success") == 2:
            sig = out["obs"].get("signals") or {}
            raw_after = AgentState[sig["agent_state"]]
            frame_after = to_frame(out["obs"]["frame"])
            if raw_after == AgentState.IN_MENU and is_false_overworld_menu_state(frame_after):
                direction = kwargs.get("direction", "up")
                steps = int(kwargs.get("steps") or 1)
                tr0 = out["obs"].get("transition") or {}
                ar0 = tr0.get("action_return") or {}
                taken0 = int(ar0.get("n_steps_taken") or 0)
                if taken0 < steps:
                    out, transition, action_success = self._emulate_move_steps_in_false_menu(
                        direction, steps, initial_out=out
                    )
                    self._absorb(out, frame_changed=bool(transition.get("frame_changed")))
                    action_return = transition.get("action_return")
                    states = [{"core": {}, "ocr": {"ocr_regions": {kind: to_frames(crops) for kind, crops in ocr.items()
                                                                   if crops}}}
                              for ocr in (transition.get("ocr") or [{}])]
                    if action_return is not None:
                        states[-1]["core"]["action_return"] = action_return
                    self._info["core"].update({
                        "action_return": action_return,
                        "transition_passed_frames": to_frames(transition.get("frames")),
                        "previous_action_details": (action, kwargs, states, action_success, action_return),
                    })
                    info = self.get_info()
                    terminated = self.done and self.success == 1.0
                    return self.get_observation(), 0.0, terminated, self.done and not terminated, info
                out["obs"]["action_success"] = 1

        transition = out["obs"].get("transition") or {}
        self._absorb(out, frame_changed=bool(transition.get("frame_changed")))
        action_return = transition.get("action_return")
        states = [{"core": {}, "ocr": {"ocr_regions": {kind: to_frames(crops) for kind, crops in ocr.items()
                                                       if crops}}}
                  for ocr in (transition.get("ocr") or [{}])]
        if action_return is not None:
            states[-1]["core"]["action_return"] = action_return
        self._info["core"].update({
            "action_return": action_return,
            "transition_passed_frames": to_frames(transition.get("frames")),
            "previous_action_details": (action, kwargs, states, out["obs"].get("action_success"), action_return)})
        info = self.get_info()
        terminated = self.done and self.success == 1.0
        return self.get_observation(), 0.0, terminated, self.done and not terminated, info

    def save_custom_state(self, name: str) -> None:
        if self._named_saves:
            self._session.save(name)
        else:
            self._session.checkpoint(name)

    def load_custom_state(self, name: str) -> None:
        if self._named_saves:
            self._absorb(self._session.load(name), frame_changed=True)
            return
        label = self._session.describe().get("label")
        if label != name:
            raise RuntimeError(f"the server resumed from checkpoint {label!r}, not {name!r}: the game and this "
                               "harness's saved memory are out of step")

    def reset(self):
        if not self._owned:
            raise RuntimeError("a remote session is opened and reset by the server's operator, not the harness")
        self._absorb(self._session.reset(), frame_changed=True)
        return self.get_observation(), self.get_info()

    def close(self) -> None:
        if self._owned:
            self._session.close()
            self._owned = False
