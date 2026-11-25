import re
from typing import Any, Dict, Optional, Tuple

from execution.executor_action import ExecutorAction, LocateAction, coords_to_string
from execution.perception.pokemon.tiles import verbalize_neighbourhood
from execution.game_api import AgentState
from utils import parse_key_value


class PokemonLocateAction(LocateAction):
    # TODO: cannot be instantiated until `verbalize` and `string_to_kwargs` are implemented.
    pre_described_options = {
        "item": "a pixelated, greyscale Poke Ball sprite, recognizable by its circular shape, white center, black band around the top, and grey body",
        "pokeball": "a pixelated, greyscale Poke Ball sprite, recognizable by its circular shape, white center, black band around the top, and grey body",
        "npc": "a pixelated human-like character sprite",
        "grass": "a pixelated, greyscale patch of grass that resembles wavy dark lines.",
        "sign": "a pixelated, greyscale white signpost with dots on its face",
    }


    def is_valid(self, info, target=None, **kwargs):
        if info["pokemon_core"]["agent_state"] != AgentState.FREE_ROAM:
            return False, "can only locate objects while in free roam"
        return super().is_valid(info, target=target, **kwargs)



class CheckInteractionAction(ExecutorAction):
    """Success codes: -1 target not adjacent, 0 in a cardinal cell or across a 1-tile counter (faceable), 1 only in a diagonal/unaligned cell."""

    MAX_NEW_TOKENS = 4000

    LOCATE_TAG = "check_interaction_locate"

    cardinals = {
        "up": (0, 1),
        "down": (0, -1),
        "left": (-1, 0),
        "right": (1, 0),
    }
    counter_cardinals = {
        "up": (0, 2),
        "down": (0, -2),
        "left": (-2, 0),
        "right": (2, 0),
    }
    neighbours = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

    neighbourhood_block = """This is what has already been recognised in the cells around the player. Use it together with the image:
[NEIGHBOURHOOD]
"""

    _call_pattern = re.compile(r"check_interaction\s*\(\s*target\s*=\s*(.*?)\s*\)\s*$", re.IGNORECASE)

    @classmethod
    def verbalize(cls) -> str:
        return (
            "check_interaction(target=<str>) — check whether the target is directly in front of you (or across a 1-tile counter) and can be interacted with, and if not, whether it is next to you.\n"
            "  target — the object or character to look for (e.g. 'old man npc').\n"
            "  Example: check_interaction(target=old man npc)"
        )

    @classmethod
    def string_to_kwargs(cls, action_str: str) -> Dict[str, Any]:
        match = cls._call_pattern.search(action_str.strip())
        if match is None:
            return {}
        return {"target": match.group(1).strip().strip("'\"")}

    def is_valid(self, info, target: str = None, **kwargs):
        if info["pokemon_core"]["agent_state"] != AgentState.FREE_ROAM:
            return False, "can only check interactions while in free roam"
        if not isinstance(target, str) or not target.strip():
            return False, f"target must be a non-empty string, got {target!r}"
        return True, None

    locate_prompt = """You are playing Pokemon. The image is the current screen, divided into a grid of 16x16 cells. The player is at (0, 0), x increases to the right and y increases upward.

The target to interact with is: [TARGET]
[NEIGHBOURHOOD_BLOCK]
Which cell holds the target? The player can interact with something directly above (0, 1), below (0, -1), to the left (-1, 0), or to the right (1, 0), or with an NPC standing directly behind a 1-tile counter/desk two cells away at (0, 2), (0, -2), (-2, 0), or (2, 0). Prefer one of those cardinal cells if the target covers several cells. If the target is visible nearby on screen (such as an NPC behind a counter at (1, 2) or (-1, 2)), give its (x, y) coordinates. Answer None if the target is not near the player.

Respond in exactly this format:
Reasoning: <where the target is on the screen>
Cell: <(x, y) of the target, or None>
[STOP]"""

    def _locate_prompt(self, target: str, identified) -> str:
        block = ""
        if identified is not None:
            block = self.neighbourhood_block.replace("[NEIGHBOURHOOD]", verbalize_neighbourhood(identified))
        return (self.locate_prompt.replace("[TARGET]", target.strip())
                .replace("[NEIGHBOURHOOD_BLOCK]", block))

    def parse_cell(self, output: str) -> Optional[Tuple[int, int]]:
        from execution.perception.pokemon.navigation import in_bounds

        answer = parse_key_value(output, "Cell") or ""
        match = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", answer)
        if match is None:
            return None
        cell = (int(match.group(1)), int(match.group(2)))
        if cell == (0, 0) or not in_bounds(cell):
            return None
        return cell

    def _execute(self, info, target: str, identified=None, cell_kinds=None, **kwargs) -> Tuple[Dict[str, Any], int]:
        from execution.perception.pokemon.navigation import (
            WALKABLE,
            cell_kinds as compute_cell_kinds,
            in_bounds,
            is_counter_separator_cell,
            neighbours,
            parse_all_cells,
            shortest_path,
        )

        results: Dict[str, Any] = {"cell": None, "found_cells": [], "found_cells_str": coords_to_string([])}
        kinds = cell_kinds if cell_kinds is not None else (compute_cell_kinds(identified) if identified is not None else None)
        explicit = [c for c in parse_all_cells(target) if c in self.cardinals.values() or c in self.counter_cardinals.values()]
        if len(explicit) == 1 and identified is not None:
            ec = explicit[0]
            if ec in self.cardinals.values():
                cell = ec
            elif any((ec[0], ec[1], q) in identified and identified[(ec[0], ec[1], q)].tag == "npc"
                     for q in ("top_left", "top_right", "bottom_left", "bottom_right")):
                cell = ec
            else:
                cell = None
        else:
            cell = None
        if cell is None:
            output = self._vlm_call(
                self.LOCATE_TAG,
                texts=self._locate_prompt(target, identified),
                images=[info["core"]["current_frame"]],
                max_new_tokens=self.MAX_NEW_TOKENS,
            )
            cell = self.parse_cell(output)
        results["cell"] = cell
        if cell is None:
            return results, -1
        results["found_cells"] = [cell]
        results["found_cells_str"] = coords_to_string([cell])
        if cell in self.cardinals.values():
            results["direction"] = next(name for name, offset in self.cardinals.items() if offset == cell)
            return results, 0
        if cell in self.counter_cardinals.values():
            mid = (cell[0] // 2, cell[1] // 2)
            direct_goals = (
                [nb for _, nb in neighbours(cell) if in_bounds(nb) and kinds.get(nb) == WALKABLE]
                if kinds is not None
                else []
            )
            if kinds is None or (
                kinds.get(mid) != WALKABLE
                and (
                    is_counter_separator_cell(mid, kinds, identified)
                    or not direct_goals
                    or shortest_path(kinds, (0, 0), direct_goals) is None
                )
            ):
                results["direction"] = next(name for name, offset in self.counter_cardinals.items() if offset == cell)
                results["across_counter"] = True
                return results, 0
        return results, 1
