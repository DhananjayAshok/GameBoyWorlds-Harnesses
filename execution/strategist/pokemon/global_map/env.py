from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np

from .registration import has_structure

if TYPE_CHECKING:
    from .model import GlobalMap


def _movement(action, kwargs: Dict[str, Any], info: Dict[str, Any],
              global_map: Optional[GlobalMap] = None) -> Tuple[Optional[str], int, Optional[int]]:
    from execution.game_api import LowLevelAction, LowLevelActions, MoveStepsAction

    if action is MoveStepsAction or (isinstance(action, type) and issubclass(action, MoveStepsAction)):
        direction = kwargs.get("direction")
        steps = int(kwargs.get("steps") or 1)
        reported = (info.get("core", {}).get("action_return") or {}).get("n_steps_taken")
        # one extra cell allows for a ledge hop, which moves two cells on one press
        return direction, steps + 1, (int(reported) if reported is not None else None)
    if action is LowLevelAction or (isinstance(action, type) and issubclass(action, LowLevelAction)):
        arrows = {LowLevelActions.PRESS_ARROW_UP: "up", LowLevelActions.PRESS_ARROW_DOWN: "down",
                  LowLevelActions.PRESS_ARROW_LEFT: "left", LowLevelActions.PRESS_ARROW_RIGHT: "right"}
        direction = arrows.get(kwargs.get("low_level_action"))
        if direction is not None:
            max_steps = 1
            if global_map is not None and direction in ("down", "left"):
                try:
                    seg = global_map.current
                    pos = global_map.pos
                    if seg and pos is not None:
                        dx, dy = (0, -1) if direction == "down" else (-1, 0)
                        c = seg.get((pos[0] + dx, pos[1] + dy))
                        if c is not None and (
                            (direction == "down" and (c.kind == "ledge" or c.hashes in global_map.ledge_patterns))
                            or (direction == "left" and (c.kind == "ledge_left" or c.hashes in global_map.left_ledge_patterns))
                        ):
                            max_steps = 2
                except Exception:  # noqa: BLE001
                    pass
            return direction, max_steps, None
    return None, 0, None


@dataclass
class StepEvent:
    """One environment step, after the map has placed its frame: what step listeners are told."""
    action: Any
    kwargs: Dict[str, Any]
    info: Dict[str, Any]
    was_free: bool
    free: bool
    frame: np.ndarray
    identified: Dict[Any, Any]
    status: str
    direction: Optional[str] = None
    max_steps: int = 0
    reported: Optional[int] = None


class MappingEnvironment:
    """Wraps the environment so every step taken through it updates the global map; a map failure ends the run.
    Listeners get a StepEvent after each step the map accepted; one that fails is logged and skipped."""

    def __init__(self, env, global_map: GlobalMap, tile_recognizer, parameters: Optional[dict] = None,
                 listeners: Iterable[Callable[[StepEvent], None]] = ()) -> None:
        self._env = env
        self._map = global_map
        self._tiles = tile_recognizer
        if tile_recognizer is not None and hasattr(tile_recognizer, "canonical_hash"):
            global_map.canonical_hash = tile_recognizer.canonical_hash
        self._parameters = parameters
        self._listeners = list(listeners)
        self.marker = None
        self.facing: Optional[str] = None

    def step_high_level_action(self, action, **kwargs):
        from execution.game_api import AgentState, MenuAction, OpenMenuAction
        from execution.perception.pokemon.battle import is_battle_screen

        before = self._env.get_info()
        before_state = before.get("pokemon_core", {}).get("agent_state")
        was_free = before_state == AgentState.FREE_ROAM
        result = self._env.step_high_level_action(action, **kwargs)
        try:
            info = result[4] or {}
            if (info.get("pokemon_core", {}).get("n_battles_lost", 0)
                    > before.get("pokemon_core", {}).get("n_battles_lost", 0)):
                self._map.record_defeat()
            core = info.get("core", {})
            if not info.get("invalid_action") and core.get("current_frame") is not None:
                frame = core["current_frame"]
                after_state = info.get("pokemon_core", {}).get("agent_state")
                free = after_state == AgentState.FREE_ROAM
                in_battle = (
                    before_state == AgentState.IN_BATTLE
                    or after_state == AgentState.IN_BATTLE
                    or (not free and is_battle_screen(frame))
                )
                is_menu_action = (
                    action in (OpenMenuAction, MenuAction)
                    or (isinstance(action, type) and issubclass(action, (OpenMenuAction, MenuAction)))
                )
                in_menu = (
                    is_menu_action
                    or before_state == AgentState.IN_MENU
                    or after_state == AgentState.IN_MENU
                )
                direction, max_steps, reported = _movement(action, kwargs, info, self._map)
                passed = core.get("transition_passed_frames")
                faded = not in_menu and passed is not None and any(not has_structure(f) for f in passed)
                identified = self._tiles.identify_tiles(frame) if free else {}
                queued = len(self._map.naming_queue)
                status = self._map.on_step(
                    frame, identified, free, direction, max_steps, reported, was_free, faded, in_battle=in_battle
                )
                if free and identified and hasattr(self._tiles, "backfill_masked_tiles"):
                    identified = self._tiles.backfill_masked_tiles(identified, frame, self._map)
                if was_free and direction is not None:
                    self.facing = direction
                elif status in ("relocalised", "new_segment"):
                    self.facing = None
                elif status == "moved" and direction is None:
                    self.facing = None
                for entry in self._map.naming_queue[queued:]:
                    entry["marker"] = self.marker() if self.marker is not None else None
                if self._listeners:
                    event = StepEvent(action, kwargs, info, was_free, free, frame, identified, status, direction,
                                      max_steps, reported)
                    for listener in self._listeners:
                        try:
                            listener(event)
                        except Exception as e:  # noqa: BLE001
                            from utils import log_warn
                            log_warn(f"A step listener failed and was skipped: {type(e).__name__}: {e}",
                                     self._parameters)
        except Exception as e:
            from utils import log_error
            # log_error raises
            name = getattr(action, "__name__", repr(action))
            log_error(f"Global map update failed on {name}({kwargs}): {type(e).__name__}: {e}\n"
                      f"was_free={was_free}, segment={self._map.segment}, pos={self._map.pos}, "
                      f"cells={self._map.n_cells}\n{traceback.format_exc()}", self._parameters)
        return result

    @property
    def ledge_patterns(self) -> set:
        return self._map.ledge_patterns

    @property
    def left_ledge_patterns(self) -> set:
        return self._map.left_ledge_patterns

    @property
    def walk_patterns(self) -> set:
        return self._map.walk_patterns

    @property
    def blocked_cells(self) -> set:
        return self._map.local_blocked_cells()

    def record_blocked_step(
        self,
        direction: Optional[str] = None,
        cell: Optional[Tuple[int, int]] = None,
        source_cell: Optional[Tuple[int, int]] = None,
    ):
        return self._map.record_blocked_step(direction=direction, cell=cell, source_cell=source_cell)

    def record_defeat(self, place: Optional[Tuple[int, Tuple[int, int]]] = None) -> None:
        self._map.record_defeat(place=place)

    def step_str(self, input_str: str):
        action, kwargs = self._env.string_to_high_level_action(input_str)
        if action is None:
            return None, None, None, None, None
        return self.step_high_level_action(action, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self._env, name)
