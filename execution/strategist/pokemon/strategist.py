from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from execution.artifact import episode_dirs
from execution.game_api import AgentState, WaitAction
from execution.perception.pokemon.battle import (has_bottom_dialogue_box, has_modal_preview_box,
                                                 has_player_pokemon_fainted_text, is_battle_defeat_text,
                                                 is_healing_item_name, is_status_heal_item_name)
from execution.perception.pokemon.tiles import TileRecognizer, describe_state_screen, verbalize_tiles
from execution.pokemon.executors import CombatExecutor
from execution.pokemon.supervisors import PokemonPlayThroughSupervisor
from execution.report import (EnvironmentStepRecord, ExecutorVLMCallRecord, SupervisorReport, SupervisorVLMCallRecord,
                              action_name, per_prompt_token_counts)
from execution.run_record import RunRecord
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import ask, single_value
from execution.strategist.pokemon.goals import (GoalNode, GoalSubtree, PokemonGoalTree, check_goals,
                                                create_subgoals, frontier_text, goal_path_text)
from execution.strategist.pokemon.guide import (NEW_TOWN, StrategyGuide, battle_tips, guide_block, load_seeds,
                                                record_events, record_new_town, reflect_on_defeat, town_question,
                                                update_guide)
from execution.strategist.pokemon.knowledge import (DEFAULT_CATEGORY_DESCRIPTIONS, PokemonKnowledgeTree,
                                                    compact_active_story, has_recorded_knowledge, remember,
                                                    remember_all, tree_entries)
from execution.strategist.pokemon.notepad import ThoughtNotepad
from execution.strategist.pokemon.party_state import PartyInventoryState
from execution.strategist.pokemon.npc_map import with_people
from execution.strategist.pokemon.npcs import PLACED_STATUS, STALE_EPISODES, NPCRegistry
from execution.strategist.pokemon.global_map import (FAILURE_EPISODES, GlobalMap, MappingEnvironment, explore_text,
                                                     exploring, judge_merge, judge_stuck, keep_exploring,
                                                     name_segment, navigate_map, prompt_block, stuck_signs)
from execution.perception.pokemon.navigation import cell_kinds, verbalize_edges
from execution.strategist.report import (REPORT_DETAILS, EpisodeArchive, EpisodeRecord, StrategistReport,
                                         StrategistVLMCallRecord, load_episode_report, save_episode_report,
                                         saved_episode_reports, trim_episode)
from python_scripts.paths import playthrough_dir, playthrough_session_rel
from utils import VLM, load_parameters, log_error, log_info, log_warn
from utils.lm_inference import clean_value, parse_key_value
from utils.parsing import parse_list

PROVENANCE_PREFIX = "provenance"
MAX_BATTLE_REASON = 200
# A door warp leaves grey lines, so a frame counts as black when most, not all, pixels are dark.
BLACK_PIXEL_LEVEL = 10
BLACK_FRACTION = 0.95
MAX_SETTLE_STEPS = 8
FIXED_GOAL_SOURCES = ("initial", "default", "override")


def settle_screen(env) -> None:
    """While the screen is black, let the game run on so an episode never ends mid door-warp."""
    for _ in range(MAX_SETTLE_STEPS):
        frame = np.asarray(env.get_info()["core"]["current_frame"])
        if float(np.mean(frame <= BLACK_PIXEL_LEVEL)) < BLACK_FRACTION:
            return
        obs, *_ = env.step_high_level_action(WaitAction)
        if obs is None:
            return


HEALING_DIGEST_PATTERN = re.compile(
    r"\b(?:fighting\s+fit|fully\s+healed|restored\s+(?:your\s+POK[eé]MON\s+)?to\s+full\s+health"
    r"|healed\s+(?:the\s+|your\s+|all\s+)?(?:POK[eé]MON|party)|take\s+a\s+rest)\b",
    re.IGNORECASE,
)
HEALING_ITEM_BOUGHT_PATTERN = re.compile(
    r"\b(?:here\s+you\s+are!\s*thank\s+you!|bought\s+\d*\s*(?:super\s+|hyper\s+|max\s+)?potions?"
    r"|bought\s+as\s+many\s+potions)\b",
    re.IGNORECASE,
)
HEALING_ITEM_USED_PATTERN = re.compile(
    r"\b(?:recovered(?:\s+by)?|restored)\s+(\d+)(?:\s*HP)?\b",
    re.IGNORECASE,
)
HEALING_BAG_EMPTY_PATTERN = re.compile(
    r"(?:No\s+healing\s+item\s+was\s+used\s+in\s+menu"
    r"|bag\s+is\s+empty\s*(?:\(only\s+CANCEL\)|—\s*only\s+CANCEL)"
    r"|checked\s+all\s+bag\s+slots\s*—\s*no\s+usable\s+healing\s+item\s+in\s+bag"
    r"|closed\s+the\s+bag\s+without\s+using\s+an\s+item;\s*do\s+not\s+choose\s+bag\s+again"
    r"|\ball\s+potions\s+(?:in\s+the\s+bag\s+)?have\s+(?:now\s+)?been\s+used\s+up\b"
    r"|\b(?:used|using)\s+up\s+(?:all\s+)?(?:our\s+|the\s+)?remaining\s+potions\b"
    r"|\bno\s+(?:usable\s+)?healing\s+items\s+(?:left\s+|remaining\s+)?in\s+(?:the\s+)?bag\b)",
    re.IGNORECASE,
)


def is_healing_item_bought_digest(digest: str) -> bool:
    if not digest:
        return False
    if HEALING_ITEM_BOUGHT_PATTERN.search(digest):
        return True
    return bool("here you are!" in digest.lower() and "potion" in digest.lower())


def is_healing_bag_empty_digest(digest: str) -> bool:
    if not digest or is_healing_digest(digest) or is_healing_item_bought_digest(digest):
        return False
    bag_info = CombatExecutor.extract_bag_items_from_text(digest)
    if bag_info is not None:
        bag_items, all_checked = bag_info
        if any(is_healing_item_name(it) for it in bag_items):
            return False
        if all_checked:
            return True
    if HEALING_BAG_EMPTY_PATTERN.search(digest):
        return True
    if "Text seen along the way, in order:" in digest:
        text_sec = digest.split("Text seen along the way, in order:", 1)[1]
        chunks = re.split(r"\n-\s+while\s+doing\s+\"", text_sec)
        for ch in chunks[1:]:
            if "\":" not in ch:
                continue
            obj, body = ch.split("\":", 1)
            body = body.strip()
            if (
                re.search(r"\b(?:item|bag|potion|restore\s+hp)\b", obj, re.IGNORECASE)
                and body.startswith("Presses:")
                and len(body.splitlines()) == 1
                and not HEALING_ITEM_USED_PATTERN.search(body)
            ):
                return True
    return False
BATTLE_CHUNK_PATTERN = re.compile(
    r"\b(?:Wild\s+[A-Z0-9♀♂.'-]+\s+appeared|wants\s+to\s+(?:battle|fight)"
    r"|sent\s+out\s+[A-Z0-9♀♂.'-]+!|used\s+[A-Z0-9♀♂.' -]+!|Got\s+away\s+safely!)",
    re.IGNORECASE,
)
USED_MOVE_PATTERN = re.compile(r"\bused\s+[A-Z0-9♀♂.' -]+!")


def is_healing_digest(digest: str) -> bool:
    return bool(digest and HEALING_DIGEST_PATTERN.search(digest))


def is_real_battle_loss(record: EpisodeRecord) -> bool:
    if getattr(record, "status", "") != "lost_battle":
        return False
    digest = record.supervisor_report.digest if getattr(record, "supervisor_report", None) else ""
    if "Text seen along the way, in order:" not in digest:
        return True
    text_sec = digest.split("Text seen along the way, in order:", 1)[1]
    if not text_sec.strip():
        return True
    return bool(is_battle_defeat_text([text_sec]) or has_player_pokemon_fainted_text([text_sec]))


def fought_and_fled_battles_in_digest(digest: str) -> Tuple[int, int]:
    if not digest or "Text seen along the way, in order:" not in digest:
        return 0, 0
    text_sec = digest.split("Text seen along the way, in order:", 1)[1]
    chunks = re.split(r"\n-\s+while\s+doing\s+\"", text_sec)
    fought, fled = 0, 0
    for ch in chunks[1:]:
        obj, body = ch.split("\":", 1) if "\":" in ch else ("", ch)
        if obj.strip().lower().startswith("clear the dialogue"):
            continue
        if not BATTLE_CHUNK_PATTERN.search(body):
            continue
        if "got away safely" in body.lower() and not USED_MOVE_PATTERN.search(body):
            fled += 1
        else:
            fought += 1
    return fought, fled


def sanitize_false_loss_summary(summary: str) -> str:
    if not summary:
        return summary
    return re.sub(
        r",?\s*(?:before\s+losing|then\s+lost)\s+a\s+(?:subsequent\s+|wild\s+)?battle\s+and\s+"
        r"(?:black(?:ing|ed)|whit(?:ing|ed))\s+out",
        "",
        summary,
        flags=re.IGNORECASE,
    )


NARRATIVE_COORD_RE = re.compile(
    r"\[Area coordinate:\s*\((-?\d+),\s*(-?\d+)\)\s*->\s*\((-?\d+),\s*(-?\d+)\)([^\]]*)\]"
)
BLOCKED_AT_COORD_RE = re.compile(
    r"\bBlocked\s+moving\s+(up|down|left|right)\s+at\s+\((-?\d+),\s*(-?\d+)\)",
    re.IGNORECASE,
)
BLOCKED_DIR_NOTE_RE = re.compile(
    r"\b(?:Blocked|Stopped|Could not finish)\s+moving\s+(up|down|left|right)\b",
    re.IGNORECASE,
)


def extract_blocked_moves_from_narrative(
    narrative: Optional[ List[str] ],
    end_pos: Optional[Tuple[int, int]] = None,
) -> Dict[Tuple[int, int], List[str]]:
    if not narrative:
        return {}
    lines = list(narrative)
    last_new_area_idx = -1
    last_after_pos: Optional[Tuple[int, int]] = None
    for i, line in enumerate(lines):
        m_coord = NARRATIVE_COORD_RE.search(line)
        if m_coord:
            last_after_pos = (int(m_coord.group(3)), int(m_coord.group(4)))
            if "entered new area" in m_coord.group(5).lower():
                last_new_area_idx = i
    if end_pos is not None and last_after_pos is not None and last_after_pos != end_pos:
        return {}
    out: Dict[Tuple[int, int], List[str]] = {}
    for line in lines[last_new_area_idx + 1:]:
        for m_at in BLOCKED_AT_COORD_RE.finditer(line):
            d = m_at.group(1).lower()
            pos = (int(m_at.group(2)), int(m_at.group(3)))
            if d not in out.get(pos, ()):
                out.setdefault(pos, []).append(d)
        if "did not work: move_blocked" in line:
            m_coord = NARRATIVE_COORD_RE.search(line)
            m_dir = BLOCKED_DIR_NOTE_RE.search(line)
            if m_coord and m_dir:
                after_pos = (int(m_coord.group(3)), int(m_coord.group(4)))
                d = m_dir.group(1).lower()
                if d not in out.get(after_pos, ()):
                    out.setdefault(after_pos, []).append(d)
    return out


def rewrite_healed_shifts_in_text(
    text: str,
    gmap: GlobalMap,
) -> str:
    if not text or not gmap.current:
        return text
    shifts = [
        sh for sh in getattr(gmap, "healed_shifts", ())
        if gmap.resolve(sh.get("segment", -1), (0, 0))[0] == gmap.segment
        and (sh.get("ox", 0), sh.get("oy", 0)) != (0, 0)
    ]
    if not shifts:
        return text
    last_sh = shifts[-1]
    ox, oy = int(last_sh["ox"]), int(last_sh["oy"])
    old_pos = tuple(last_sh["old_pos"])
    post_old_positions = {old_pos}
    d_step = int(last_sh.get("from_step", 0))
    if d_step > 0:
        for t, s, p in gmap.trail:
            if t >= d_step and gmap.resolve(s, (0, 0))[0] == gmap.segment:
                rp = gmap.resolve(s, p)[1]
                post_old_positions.add((rp[0] - ox, rp[1] - oy))
    if not any(
        f"({op[0]}, {op[1]})" in text or f"({op[0]},{op[1]})" in text
        for op in post_old_positions
    ):
        return text

    def _shift_narrative_bracket(m: re.Match) -> str:
        p1 = (int(m.group(1)), int(m.group(2)))
        p2 = (int(m.group(3)), int(m.group(4)))
        if p1 not in post_old_positions and p2 not in post_old_positions:
            return m.group(0)
        x1, y1 = p1[0] + ox, p1[1] + oy
        x2, y2 = p2[0] + ox, p2[1] + oy
        rest = m.group(5)
        rest = re.sub(
            r"(episode\s+start\s+)\((-?\d+),\s*(-?\d+)\)",
            lambda ms: f"{ms.group(1)}({int(ms.group(2)) + ox}, {int(ms.group(3)) + oy})",
            rest,
        )
        return f"[Area coordinate: ({x1}, {y1}) -> ({x2}, {y2}){rest}]"

    out = NARRATIVE_COORD_RE.sub(_shift_narrative_bracket, text)
    np_coord = (old_pos[0] + ox, old_pos[1] + oy)
    out = re.sub(
        rf"(`?)\({old_pos[0]},\s*{old_pos[1]}\)(`?)",
        lambda m, npc=np_coord: f"{m.group(1)}({npc[0]}, {npc[1]}){m.group(2)}",
        out,
    )
    return out


def sanitize_stale_notepad_routes(
    text: str,
    gmap: GlobalMap,
    blocked_moves: Optional[Dict[Tuple[int, int], Any]] = None,
) -> str:
    if not text or not gmap.current:
        return text
    from execution.strategist.pokemon.global_map.model import BLOCKED, DIRECTIONS, LEDGE, LEDGE_LEFT
    from execution.strategist.pokemon.global_map.routes import blocked_target_cells, compress_moves, path_in_segment

    seg = gmap.current
    px, py = gmap.pos
    shifts = [
        sh for sh in getattr(gmap, "healed_shifts", ())
        if gmap.resolve(sh.get("segment", -1), (0, 0))[0] == gmap.segment
        and (sh.get("ox", 0), sh.get("oy", 0)) != (0, 0)
    ]
    if shifts:
        last_sh = shifts[-1]
        ox, oy = int(last_sh["ox"]), int(last_sh["oy"])
        old_pos = tuple(last_sh["old_pos"])
        if f"({old_pos[0]}, {old_pos[1]})" in text or f"({old_pos[0]},{old_pos[1]})" in text:
            out = rewrite_healed_shifts_in_text(text, gmap)
            doors = gmap.door_list(gmap.segment)
            rel_door_set = {
                (c[0] - px, c[1] - py)
                for d in doors
                for c in d["cells"]
            }
            frontiers = set(gmap.frontier(gmap.segment))
            rel_frontier_set = {(f[0] - px, f[1] - py) for f in frontiers}
            new_lines = []
            in_edges_section = False
            for line in out.splitlines():
                stripped = line.strip()
                if re.match(r"^-\s*Unexplored\s+edges\b", stripped, re.IGNORECASE):
                    in_edges_section = True
                    line = re.sub(
                        r"\(\d+\s+squares?\)",
                        f"({len(frontiers)} square{'s' if len(frontiers) != 1 else ''})",
                        line,
                        flags=re.IGNORECASE,
                    )
                    new_lines.append(line)
                    continue
                elif stripped.startswith("- ") and not line.startswith("  "):
                    in_edges_section = False

                if in_edges_section and line.startswith("  - "):
                    m_edge = re.search(r"\((-?\d+),\s*(-?\d+)\)", line)
                    if m_edge:
                        ex, ey = int(m_edge.group(1)), int(m_edge.group(2))
                        if (ex, ey) not in rel_frontier_set:
                            if (ex - ox, ey - oy) in rel_frontier_set:
                                nx, ny = ex - ox, ey - oy
                                found = path_in_segment(
                                    gmap, gmap.segment, (px, py), (px + nx, py + ny), blocked_moves=blocked_moves
                                )
                                line = re.sub(
                                    rf"(`?)\({ex},\s*{ey}\)(`?)",
                                    lambda m, x=nx, y=ny: f"{m.group(1)}({x}, {y}){m.group(2)}",
                                    line,
                                    count=1,
                                )
                                if found is not None and found[1]:
                                    n_steps = len(found[1])
                                    line = re.sub(r"\b\d+\s+steps\b", f"{n_steps} steps", line, count=1)
                                    line = re.sub(
                                        r"\((`?)(?:(?:up|down|left|right)\s+\d+(?:,\s*)?)+(`?)\)",
                                        lambda m, mv=compress_moves(found[1]): f"({m.group(1)}{mv}{m.group(2)})",
                                        line,
                                        count=1,
                                    )
                                else:
                                    line = re.sub(r"\b\d+\s+steps\b", f"{abs(nx) + abs(ny)} steps", line, count=1)
                                new_lines.append(line)
                            continue

                m_coords = list(re.finditer(r"(`?)\((-?\d+),\s*(-?\d+)\)(`?)", line))
                if m_coords and line.startswith("  - "):
                    m0 = m_coords[0]
                    rx, ry = int(m0.group(2)), int(m0.group(3))
                    if (
                        (rx, ry) in rel_door_set
                        and (rx + ox, ry + oy) not in rel_door_set
                        and re.search(rf"\({rx + ox},\s*{ry + oy}\)", out)
                        and any(
                            (px + rx, py + ry) in d["cells"] and d.get("leads_to") is not None
                            for d in doors
                        )
                        and not any(
                            w in line.lower()
                            for w in ("center", "gym", "house", "well", "mart", "lab", "cave", "tower", "gate")
                        )
                    ):
                        continue
                    if (rx, ry) not in rel_door_set and (rx - ox, ry - oy) in rel_door_set:
                        nx, ny = rx - ox, ry - oy
                        line = re.sub(
                            rf"(`?)\({rx},\s*{ry}\)(`?)",
                            lambda m, x=nx, y=ny: f"{m.group(1)}({x}, {y}){m.group(2)}",
                            line,
                            count=1,
                        )
                        if len(m_coords) > 1:
                            for m_sec in m_coords[1:]:
                                sx, sy = int(m_sec.group(2)), int(m_sec.group(3))
                                line = re.sub(
                                    rf"(`?)\({sx},\s*{sy}\)(`?)",
                                    lambda m, x=sx - ox, y=sy - oy: f"{m.group(1)}({x}, {y}){m.group(2)}",
                                    line,
                                    count=1,
                                )
                        found = path_in_segment(
                            gmap, gmap.segment, (px, py), (px + nx, py + ny), blocked_moves=blocked_moves
                        )
                        if found is not None and found[1]:
                            n_steps = len(found[1])
                            line = re.sub(r"\b\d+\s+steps\b", f"{n_steps} steps", line, count=1)
                            line = re.sub(
                                r"\((`?)(?:(?:up|down|left|right)\s+\d+(?:,\s*)?)+(`?)\)",
                                lambda m, mv=compress_moves(found[1]): f"({m.group(1)}{mv}{m.group(2)})",
                                line,
                                count=1,
                            )
                        else:
                            line = re.sub(r"\b\d+\s+steps\b", f"{abs(nx) + abs(ny)} steps", line, count=1)
                new_lines.append(line)
            text = "\n".join(new_lines)

    blocked_targets = blocked_target_cells(gmap, gmap.segment, blocked_moves)

    def trace_is_blocked(seq_str: str) -> Tuple[bool, Tuple[int, int]]:
        cx, cy = px, py
        hit_blocked = False
        for part in seq_str.split(","):
            tokens = part.strip().lower().split()
            if len(tokens) != 2 or tokens[0] not in DIRECTIONS or not tokens[1].isdigit():
                return False, (cx, cy)
            d, count = tokens[0], int(tokens[1])
            dx, dy = DIRECTIONS[d]
            for _ in range(count):
                if blocked_moves and d in blocked_moves.get((cx, cy), ()):
                    hit_blocked = True
                nx, ny = cx + dx, cy + dy
                c = seg.get((nx, ny))
                if (nx, ny) in blocked_targets or (c is not None and c.kind in (BLOCKED, LEDGE, LEDGE_LEFT)):
                    hit_blocked = True
                cx, cy = nx, ny
        return hit_blocked, (cx, cy)

    pattern = re.compile(
        r"(?:(mapped\s+route(?:,\s*|\s*\()\s*)(\d+)(\s+moves\s*(?:\):\s*|:\s*)))?"
        r"(\b(?:up|down|left|right)\s+\d+(?:\s*,\s*(?:up|down|left|right)\s+\d+)+)",
        re.IGNORECASE,
    )

    def repl(m: re.Match) -> str:
        prefix, _old_count, mid, seq = m.group(1), m.group(2), m.group(3), m.group(4)
        is_blk, goal = trace_is_blocked(seq)
        if not is_blk:
            return m.group(0)
        goal_cell = seg.get(goal)
        found = (
            path_in_segment(gmap, gmap.segment, (px, py), goal, blocked_moves=blocked_moves)
            if (goal_cell is None or goal_cell.kind not in (BLOCKED, LEDGE, LEDGE_LEFT))
            else None
        )
        if found is not None and found[1]:
            new_seq = compress_moves(found[1])
            new_count = str(len(found[1]))
            if prefix is not None:
                return f"{prefix}{new_count}{mid}{new_seq}"
            return new_seq
        return "route blocked by wall — use live mapped routes below"

    return pattern.sub(repl, text)


def sanitize_false_loss_notepad(text: str) -> str:
    if not text:
        return text
    cleaned_lines = []
    for line in text.splitlines():
        if re.search(r"\b(?:just\s+)?(?:blacked|whited)\s+out\b.*\b(?:respawned|healed)\b", line, re.IGNORECASE):
            continue
        line = re.sub(r"\s+before\s+(?:blacking|whiting)\s+out\b", "", line, flags=re.IGNORECASE)
        line = re.sub(
            r"Exit\s+the\s+[^,]*Pok[eé]mon\s+Center,\s*(?:head|walk)\s+[^,]*to\s+the\s+"
            r"([^,]*tall\s+grass),\s*and\s+fight",
            r"In the \1, fight",
            line,
            flags=re.IGNORECASE,
        )
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def sanitize_stale_notepad_bag_status(text: str, healing_bag_empty: bool) -> str:
    if not text or healing_bag_empty:
        return text
    out = re.sub(
        r",\s*no\s+(?:usable\s+)?healing\s+items\s+in\s+(?:the\s+)?(?:bag|pack)",
        ", healing items available in bag",
        text,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\bno\s+(?:usable\s+)?healing\s+items\s+in\s+(?:the\s+)?(?:bag|pack)\b",
        "healing items available in bag",
        out,
        flags=re.IGNORECASE,
    )
    out = re.sub(
        r"\b(?:the\s+)?(?:bag|pack)\s+has\s+no\s+(?:usable\s+)?healing\s+items\b",
        "healing items available in bag",
        out,
        flags=re.IGNORECASE,
    )
    if out != text:
        out = re.sub(
            r"\bthen\s+heal\s+at\s+the\s+[^,;.\n]*?Pok[eé]mon\s+Center\s+or\s+",
            "then ",
            out,
            flags=re.IGNORECASE,
        )
    return out


def treat_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name)


class PokemonStrategist:
    """Plays one long game: picks a task each episode, runs a supervisor on it, and updates its memory."""

    RECENT_EPISODES = 5
    MAX_DISPATCH_ATTEMPTS = 2
    MAX_EXTRACT_ATTEMPTS = 2
    MAX_NOTEPAD_ATTEMPTS = 3
    MAX_FACTS = 3
    MIN_NAMING_STEPS = 3
    RETURN_EPISODES = 3

    def __init__(
        self,
        env,
        game: str = "pokemon_red",
        name: str = "default",
        max_episodes: int = 20,
        supervisor_max_steps: int = 10,
        strategist_vlm_model: Optional[str] = None,
        strategist_vlm_kind: Optional[str] = None,
        max_new_tokens: int = 4800,
        supervisor_max_new_tokens: int = 4800,
        subgoal_every: int = 5,
        report_detail: str = "strategist",
        resume_run: Optional[str] = None,
        override_goal: Optional[str] = None,
        npc_registry: bool = True,
        parameters: Optional[dict] = None,
        resume_episode: Optional[int] = None,
        record_dir: Optional[str] = None,
        **supervisor_kwargs: Any,
    ) -> None:
        self._env = env
        self._game = game
        self._name = name
        self._max_episodes = max_episodes
        self._supervisor_max_steps = supervisor_max_steps
        self._strategist_vlm_model = strategist_vlm_model
        self._strategist_vlm_kind = strategist_vlm_kind
        self._max_new_tokens = max_new_tokens
        self._supervisor_max_new_tokens = supervisor_max_new_tokens
        self._subgoal_every = subgoal_every
        self._report_detail = report_detail
        self._parameters = load_parameters(parameters)
        if isinstance(subgoal_every, bool) or not isinstance(subgoal_every, int) or subgoal_every <= 0:
            log_error(f"subgoal_every must be a positive integer, got {subgoal_every!r}.", self._parameters)
        if report_detail not in REPORT_DETAILS:
            log_error(f"report_detail must be one of {REPORT_DETAILS}, got {report_detail!r}.",
                      self._parameters)
        # Emulator save states are named <name><episode>, so a trailing digit would make them ambiguous.
        if treat_name(name)[-1:].isdigit():
            log_error(f"Strategist name {name!r} must not end in a digit.", self._parameters)
        self._supervisor_kwargs = supervisor_kwargs
        self._vlm_instance: Optional[VLM] = None

        self._dir = playthrough_dir(self._parameters, game=game, name=name)
        self._resume_run = resume_run
        self._resume_episode = resume_episode
        self._resume_dir = self._dir
        if resume_run is not None and resume_run != name:
            if os.path.exists(self._dir):
                log_error(f"resume_run={resume_run!r} would fork that run's state into {name!r}, "
                          f"but {self._dir} already exists.", self._parameters)
            self._resume_dir = playthrough_dir(self._parameters, game=game, name=resume_run)
        if resume_episode is not None:
            if self._resume_dir == self._dir:
                log_error("resume_episode forks another run at one of its episodes, so it needs resume_run "
                          "naming that run.", self._parameters)
            if resume_episode not in episode_dirs(self._resume_dir):
                log_error(f"Run {resume_run!r} has no episode {resume_episode} under {self._resume_dir!r}.",
                          self._parameters)
        self._record = RunRecord(record_dir, name, self._parameters)

        self.tiles = self._load_or_default(
            TileRecognizer, lambda: TileRecognizer(name=name, game=game, parameters=self._parameters))
        self.goals = self._load_or_default(PokemonGoalTree, PokemonGoalTree)
        self.knowledge = self._load_or_default(PokemonKnowledgeTree, PokemonKnowledgeTree)
        self.notepad = self._load_or_default(ThoughtNotepad, ThoughtNotepad)
        self.global_map = self._load_or_default(GlobalMap, GlobalMap)
        self.global_map.refresh(self.tiles._by_hash)
        self.guide = self._load_or_default(StrategyGuide, StrategyGuide)
        self._had_saved_party_artifact = bool(
            PartyInventoryState.saved_episodes(self._resume_dir, self._resume_episode)
        )
        self.party: PartyInventoryState = self._load_or_default(PartyInventoryState, PartyInventoryState)
        self._npc_registry = npc_registry
        self.npcs = self._load_or_default(NPCRegistry, lambda: NPCRegistry(game, name)) if npc_registry else None
        if self.npcs is not None:
            self.npcs.attach_model(self._vlm_caller)
            self.global_map.attach_npcs(self.npcs)
        self._map_placed = True
        # How people are described in the prompt: see npc_map.WHERE_NOTES.
        self._people_where = "here"

        resumed_from = episode_dirs(self._resume_dir)
        self.start_episode = (max(resumed_from) + 1) if resumed_from else 0
        if resume_episode is not None:
            self.start_episode = resume_episode + 1
        self._override_goal = override_goal
        if override_goal is not None and self.start_episode == 0:
            self.goals.tree = GoalSubtree.create_tree(
                [GoalNode(description=override_goal, source="override")], self.goals.tree.root.description)
        if self.start_episode in episode_dirs(self._dir):
            log_error(f"episode_{self.start_episode} already exists under {self._dir!r}; "
                      "refusing to overwrite it.", self._parameters)
        if self.start_episode > 0:
            self._env.load_custom_state(
                self._emulator_state_name(self._resume_run or name, self.start_episode - 1))
        self.guide.sync_seeds(load_seeds(), self.start_episode)

        self._identified: Optional[Dict[Any, Any]] = None
        self._screen_tiles: Optional[str] = None
        # Episodes run by this process, which is what max_episodes bounds.
        self._episodes_run = 0
        self._ran = False
        self._explore_now = False
        self._blocked_moves: Dict[Tuple[int, Tuple[int, int]], Dict[str, List[str]]] = {}

        self._started_at = datetime.now()
        self._session_dir = self._resolve_session_dir()
        os.makedirs(self._dir, exist_ok=True)

        self.report = StrategistReport(
            game=game,
            strategist_name=self.__class__.__name__,
            init_kwargs=self._run_config(),
            episodes=self._load_prior_episodes(),
        )
        self._write_provenance()
        self.save_all(self.start_episode)

    def _run_config(self) -> dict:
        return {
            "name": self._name,
            "max_episodes": self._max_episodes,
            "supervisor_max_steps": self._supervisor_max_steps,
            "strategist_vlm_model": self._strategist_vlm_model,
            "strategist_vlm_kind": self._strategist_vlm_kind,
            "max_new_tokens": self._max_new_tokens,
            "supervisor_max_new_tokens": self._supervisor_max_new_tokens,
            "subgoal_every": self._subgoal_every,
            "report_detail": self._report_detail,
            "storage": self._dir,
            "resume_run": self._resume_run,
            "resume_episode": self._resume_episode,
            "resume_dir": self._resume_dir,
            "supervisor_kwargs": dict(self._supervisor_kwargs),
            "override_goal": self._override_goal,
            "npc_registry": self._npc_registry,
        }

    def _resolve_session_dir(self) -> Optional[str]:
        # Read off the emulator: its timestamp is minted before this object's and can differ by a second.
        emulator = getattr(self._env, "_emulator", None)
        session_path = getattr(emulator, "session_path", None) if emulator is not None else None
        if not session_path:
            log_warn("The environment exposed no session_path, so this run's provenance "
                     "cannot record where its video was written.", self._parameters)
            return None
        return playthrough_session_rel(session_path, game=self._game, parameters=self._parameters)

    def _write_provenance(self) -> str:
        payload = {
            "started_at": self._started_at.isoformat(timespec="seconds"),
            "game": self._game,
            "strategist": self.__class__.__name__,
            "session_dir": self._session_dir,
            "init_kwargs": self._run_config(),
        }
        stamp = self._started_at.strftime("%Y%m%d_%H%M%S")
        path = self._path(f"{PROVENANCE_PREFIX}_{stamp}.json")
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2, default=str)
        log_info(f"Strategist provenance written to {path}", self._parameters)
        return path

    def _path(self, filename: str) -> str:
        return os.path.join(self._dir, filename)

    def _load_or_default(self, cls, factory: Callable[[], Any]) -> Any:
        saved = cls.saved_episodes(self._resume_dir, self._resume_episode)
        if saved:
            artifact = cls.load(self._resume_dir, saved[-1], parameters=self._parameters)
            if self._resume_dir != self._dir:
                artifact.last_written = None
            return artifact
        return factory()

    @property
    def _fought_since_heal(self) -> int:
        return self.party.fought_since_heal

    @_fought_since_heal.setter
    def _fought_since_heal(self, value: int) -> None:
        self.party.fought_since_heal = int(value)

    @property
    def _fled_since_heal(self) -> int:
        return self.party.fled_since_heal

    @_fled_since_heal.setter
    def _fled_since_heal(self, value: int) -> None:
        self.party.fled_since_heal = int(value)

    @property
    def _last_heal_episode(self) -> Optional[int]:
        return self.party.last_heal_episode

    @_last_heal_episode.setter
    def _last_heal_episode(self, value: Optional[int]) -> None:
        self.party.last_heal_episode = value

    @property
    def _last_battle_hp(self) -> Optional[Tuple[int, int]]:
        return self.party.last_battle_hp

    @_last_battle_hp.setter
    def _last_battle_hp(self, value: Optional[Tuple[int, int]]) -> None:
        self.party.last_battle_hp = value

    @property
    def _last_battle_hp_episode(self) -> Optional[int]:
        return self.party.last_battle_hp_episode

    @_last_battle_hp_episode.setter
    def _last_battle_hp_episode(self, value: Optional[int]) -> None:
        self.party.last_battle_hp_episode = value

    @property
    def _fought_since_last_hp(self) -> int:
        return self.party.fought_since_last_hp

    @_fought_since_last_hp.setter
    def _fought_since_last_hp(self, value: int) -> None:
        self.party.fought_since_last_hp = int(value)

    @property
    def _healing_bag_empty(self) -> bool:
        return self.party.healing_bag_empty

    @_healing_bag_empty.setter
    def _healing_bag_empty(self, value: bool) -> None:
        self.party.healing_bag_empty = bool(value)

    @property
    def _active_pokemon_status(self) -> Optional[str]:
        return self.party.active_pokemon_status

    @_active_pokemon_status.setter
    def _active_pokemon_status(self, value: Optional[str]) -> None:
        self.party.active_pokemon_status = value or None

    @property
    def _status_heal_bag_empty(self) -> bool:
        return self.party.status_heal_bag_empty

    @_status_heal_bag_empty.setter
    def _status_heal_bag_empty(self, value: bool) -> None:
        self.party.status_heal_bag_empty = bool(value)

    @property
    def _zero_pp_moves(self) -> List[str]:
        return self.party.zero_pp_moves

    @_zero_pp_moves.setter
    def _zero_pp_moves(self, value: List[str]) -> None:
        self.party.zero_pp_moves = list(value or [])

    @property
    def _known_bag_items(self) -> List[str]:
        return self.party.items

    @_known_bag_items.setter
    def _known_bag_items(self, value: List[str]) -> None:
        self.party.items = [
            PartyInventoryState.normalize_item_name(x) for x in (value or [])
            if PartyInventoryState.normalize_item_name(x)
        ]

    @property
    def _known_bag_complete(self) -> bool:
        return self.party.bag_complete

    @_known_bag_complete.setter
    def _known_bag_complete(self, value: bool) -> None:
        self.party.bag_complete = bool(value)

    def _update_known_bag_from_digest(self, digest: str) -> Optional[Tuple[List[str], bool]]:
        if not digest:
            return None
        info = self.party.extract_bag_items_with_quantities_from_text(digest)
        if info is not None:
            items, qtys, all_checked = info
            self.party.apply_bag_tag_observation(items, qtys, all_checked, body_text=digest)
            return list(self.party.items), all_checked
        return None

    def _record_party_episode(self, record: EpisodeRecord,
                              strategist_calls: Optional[List[StrategistVLMCallRecord]] = None) -> None:
        self.party.reconstruct_from_episode(record, strategist_calls)

    def _bag_inventory_suffix(self) -> str:
        parts: List[str] = []
        if self._known_bag_items:
            complete_note = (
                " (all bag slots checked — do NOT open the Start menu just to search for an item not listed here)"
                if self._known_bag_complete
                else ""
            )
            parts.append(f"\nBag items currently known: {', '.join(self._known_bag_items)}{complete_note}.")
        if getattr(self.party, "learned_moves", None):
            formatted_moves = self.party.format_learned_moves()
            parts.append(
                f"\nParty moves learned / confirmed known: {', '.join(formatted_moves)} "
                f"(already learned — do NOT instruct the player to open the Pack or ITEM menu to teach {', '.join(self.party.learned_moves)} again)."
            )
        return "".join(parts)

    def _party_status_block(self) -> str:
        fought = self._fought_since_heal
        fled = self._fled_since_heal
        last_heal_ep = self._last_heal_episode
        last_hp = self._last_battle_hp
        last_hp_ep = self._last_battle_hp_episode
        fought_after_hp = self._fought_since_last_hp
        bag_suffix = self._bag_inventory_suffix()
        if (
            fought <= 0
            and (last_hp is None or last_hp[0] >= last_hp[1])
            and not self._active_pokemon_status
            and not self._zero_pp_moves
        ):
            if last_heal_ep is not None:
                return f"\nParty health status: Fully healed (0 battles fought since last heal in episode {last_heal_ep}).{bag_suffix}\n"
            return f"\nParty health status: Fully healed (0 battles fought since last heal).{bag_suffix}\n"
        heal_ref = f" in episode {last_heal_ep}" if last_heal_ep is not None else ""
        parts = [
            f"{fought} battle(s) fought"
            + (f" and {fled} fled" if fled else "")
            + f" without healing since last heal{heal_ref}"
        ]
        if last_hp is not None:
            cur_hp, max_hp = last_hp
            hp_note = f"last battle HP recorded: {cur_hp}/{max_hp} HP in episode {last_hp_ep}"
            if fought_after_hp > 0:
                hp_note += f", followed by {fought_after_hp} more battle(s) without healing"
            parts.append(hp_note)
        if self._active_pokemon_status:
            if self._status_heal_bag_empty:
                parts.append(
                    f"active Pokemon status: {self._active_pokemon_status} "
                    f"(bag has NO Antidote or status-curing item — do NOT ask to use an Antidote)"
                )
            else:
                parts.append(f"active Pokemon status: {self._active_pokemon_status}")
        if self._zero_pp_moves:
            parts.append(
                f"moves at 0 PP (cannot be used until healed at a Pokemon Center): {', '.join(self._zero_pp_moves)}"
            )
        if self._healing_bag_empty:
            parts.append("bag was already checked and has no usable healing items")
        status = "; ".join(parts)
        if last_hp is not None and fought_after_hp <= 0:
            low_hp = last_hp[0] * 2 <= last_hp[1]
        elif last_hp is not None:
            low_hp = (last_hp[0] * 2 <= last_hp[1]) or fought_after_hp >= 3
        else:
            low_hp = fought >= 3
        if low_hp:
            if self._healing_bag_empty:
                warning = (
                    " — party HP is low/weakened and the bag has no healing items; "
                    "enter a Pokemon Center to heal (or buy Potions) before continuing onto a route or trainer battles."
                )
            else:
                warning = (
                    " — party HP is low/weakened; open the Start menu to heal your active Pokemon with a Potion "
                    "from the ITEM menu (or enter a nearby Pokemon Center to heal) before walking further along a "
                    "route/cave or engaging trainer battles."
                )
        elif last_hp is not None and fought_after_hp <= 0 and last_hp[0] >= last_hp[1]:
            warning = " — active Pokemon is at full HP, though the party has fought battles since the last Pokemon Center visit."
        else:
            warning = " — party has taken battle attrition and has not healed at a Pokemon Center yet."
        return f"\nParty health status: {status}{warning}{bag_suffix}\n"

    def _ingest_narrative_blocked_moves(
        self,
        narrative: Optional[List[str]],
        segment: Optional[int] = None,
        end_pos: Optional[Tuple[int, int]] = None,
    ) -> None:
        if not narrative or not self.global_map.current:
            return
        from execution.strategist.pokemon.global_map.model import DIRECTIONS

        seg_id = self.global_map.segment if segment is None else segment
        extracted = extract_blocked_moves_from_narrative(narrative, end_pos=end_pos)
        if not extracted:
            return
        for b_pos, b_dirs in extracted.items():
            for b_dir in b_dirs:
                if b_dir not in DIRECTIONS:
                    continue
                detail = f"blocked moving {b_dir}"
                dir_list = self._blocked_moves.setdefault((seg_id, b_pos), {}).setdefault(b_dir, [])
                if detail not in dir_list:
                    dir_list.append(detail)
                if seg_id == self.global_map.segment:
                    dx, dy = DIRECTIONS[b_dir]
                    self.global_map.record_blocked_step(
                        direction=b_dir,
                        cell=(b_pos[0] + dx, b_pos[1] + dy),
                        source_cell=b_pos,
                    )

    def _current_segment_blocked_moves(self) -> Dict[Tuple[int, int], set]:
        if not self.global_map.current:
            return {}
        cur_seg, _ = self.global_map.resolve(self.global_map.segment, self.global_map.pos)
        extra: Dict[Tuple[int, int], set] = {}
        for (b_seg, b_pos), by_dir in self._blocked_moves.items():
            try:
                r_seg, r_pos = self.global_map.resolve(b_seg, b_pos)
            except Exception:  # noqa: BLE001
                continue
            if r_seg == cur_seg and by_dir:
                extra.setdefault(r_pos, set()).update(by_dir.keys())
        if hasattr(self.global_map, "segment_blocked_moves"):
            return self.global_map.segment_blocked_moves(cur_seg, extra)
        return extra

    def _local_blocked_cells(self) -> set:
        from execution.perception.pokemon.navigation import in_bounds
        from execution.strategist.pokemon.global_map.routes import blocked_target_cells

        local_blocked = set(self.global_map.local_blocked_cells())
        if not self.global_map.current:
            return local_blocked
        seg_blocked_moves = self._current_segment_blocked_moves()
        if not seg_blocked_moves:
            return local_blocked
        px, py = self.global_map.pos
        for target in blocked_target_cells(self.global_map, self.global_map.segment, seg_blocked_moves):
            rel = (target[0] - px, target[1] - py)
            if rel != (0, 0) and in_bounds(rel):
                local_blocked.add(rel)
        return local_blocked

    def _load_prior_episodes(self) -> List[EpisodeRecord]:
        if self.start_episode <= 0:
            return []
        prior_nums = sorted(n for n in saved_episode_reports(self._resume_dir) if n < self.start_episode)
        keep_from = set(prior_nums[-max(self.RECENT_EPISODES, self.MIN_NAMING_STEPS + 2):])
        episodes: List[EpisodeRecord] = []
        false_loss_episodes: set = set()
        for n in prior_nums:
            try:
                archive = load_episode_report(self._resume_dir, n)
                ep = archive.episode
                if ep.status == "lost_battle" and not is_real_battle_loss(ep):
                    false_loss_episodes.add(n)
                    ep.status = "completed"
                    ep.summary = sanitize_false_loss_summary(ep.summary)
                if not self._had_saved_party_artifact:
                    self._record_party_episode(ep, archive.strategist_calls)
                if n in keep_from:
                    episodes.append(ep)
            except Exception as e:  # noqa: BLE001
                log_warn(f"Strategist: could not load episode {n} report from {self._resume_dir!r}: {e}",
                         self._parameters)
        if episodes and self.global_map.current:
            if getattr(self.global_map, "healed_shifts", None):
                for ep in episodes:
                    if ep.summary:
                        ep.summary = rewrite_healed_shifts_in_text(ep.summary, self.global_map)
                    if ep.supervisor_report is not None:
                        if ep.supervisor_report.narrative:
                            ep.supervisor_report.narrative = [
                                rewrite_healed_shifts_in_text(ln, self.global_map)
                                for ln in ep.supervisor_report.narrative
                            ]
                        if ep.supervisor_report.digest:
                            ep.supervisor_report.digest = rewrite_healed_shifts_in_text(
                                ep.supervisor_report.digest, self.global_map
                            )
            last_ep = episodes[-1]
            if last_ep.supervisor_report is not None and last_ep.supervisor_report.narrative:
                self._ingest_narrative_blocked_moves(
                    last_ep.supervisor_report.narrative,
                    segment=self.global_map.segment,
                    end_pos=self.global_map.pos,
                )
                self.global_map.refresh(self.tiles._by_hash)
            if self.notepad.text.strip():
                sanitized = sanitize_stale_notepad_routes(
                    self.notepad.text,
                    self.global_map,
                    self._current_segment_blocked_moves(),
                )
                sanitized = sanitize_stale_notepad_bag_status(
                    sanitized,
                    self._healing_bag_empty,
                )
                if sanitized != self.notepad.text:
                    self.notepad.update(sanitized, episode_number=self.start_episode)
        if false_loss_episodes:
            if (self.start_episode - 1) in false_loss_episodes and self.notepad.text.strip():
                cleaned_notes = sanitize_false_loss_notepad(self.notepad.text)
                if cleaned_notes != self.notepad.text:
                    self.notepad.update(cleaned_notes, episode_number=self.start_episode)
            if self.global_map.defeat_return is not None:
                if self.global_map.defeat_return.get("episode") in false_loss_episodes:
                    self.global_map.defeat_return = None
        return episodes

    @property
    def _vlm(self) -> VLM:
        if self._vlm_instance is None:
            if not self._strategist_vlm_model:
                log_error(f"{self.__class__.__name__} tried to make a VLM call but no "
                          f"strategist_vlm_model was given.", self._parameters)
            self._vlm_instance = VLM(self._strategist_vlm_model, self._strategist_vlm_kind)
        return self._vlm_instance

    def _vlm_call(self, stage: str, **kwargs: Any) -> Any:
        result, records = self._vlm_infer(stage, **kwargs)
        self.report.event_log.extend(records)
        return result

    def _vlm_infer(self, stage: str, **kwargs: Any) -> tuple:
        kwargs.setdefault("max_new_tokens", self._max_new_tokens)
        inferred = self._vlm.infer(**kwargs)
        result, meta = inferred["output"], inferred["meta"]
        texts = kwargs.get("texts")
        images = kwargs.get("images") or []
        records: List[StrategistVLMCallRecord] = []
        if isinstance(texts, list):
            responses = result if isinstance(result, list) else [result] * len(texts)
            token_counts = per_prompt_token_counts(meta, len(texts))
            for index, (prompt, response) in enumerate(zip(texts, responses)):
                call_images = images[index] if index < len(images) and isinstance(images[index], list) else images
                records.append(StrategistVLMCallRecord(
                    stage=stage, images=call_images, prompt=prompt, response=response,
                    input_tokens=token_counts[index][0], output_tokens=token_counts[index][1]))
        else:
            records.append(StrategistVLMCallRecord(
                stage=stage, images=images, prompt=texts, response=result,
                input_tokens=meta["input_tokens"], output_tokens=meta["output_tokens"]))
        return result, records

    def _vlm_caller(self, stage: str):
        def call(**kwargs: Any) -> Any:
            return self._vlm_call(stage, **kwargs)
        return call

    def _current_place(self) -> str:
        return self.global_map.place_label() if self.global_map.current else "unknown"

    def _current_frame(self) -> np.ndarray:
        return self._env.get_info()["core"]["current_frame"]

    def _env_done(self) -> bool:
        return bool(self._env._emulator.check_if_done())

    def _agent_state(self):
        return self._env.get_info()["pokemon_core"]["agent_state"]

    def _perceive(self, frame: np.ndarray, episode_number: int) -> str:
        state = self._agent_state()
        if state != AgentState.FREE_ROAM:
            self._identified = {}
            self._screen_tiles = describe_state_screen(state)
            return self._screen_tiles
        if has_bottom_dialogue_box(frame) or has_modal_preview_box(frame):
            self._identified = {}
            self._screen_tiles = describe_state_screen(AgentState.IN_DIALOGUE)
            return self._screen_tiles
        self.tiles.set_vlm_call(self._vlm_caller("perception"))
        self.tiles.record_tiles(frame, episode_number)
        self.global_map.refresh(self.tiles._by_hash)
        self._identified = self.tiles.identify_tiles(frame, global_map=self.global_map)
        self._screen_tiles = verbalize_tiles(self._identified)
        return self._screen_tiles

    def _recent_block(self) -> str:
        recent = self.report.episodes[-self.RECENT_EPISODES:]
        if not recent:
            return ""
        lines = [f"- episode {e.n}, task \"{e.task}\" "
                 f"[{e.status}]: {e.summary or 'no summary'}" for e in recent]
        return P.STRATEGIST_RECENT_BLOCK.replace("[RECENT]", "\n".join(lines))

    def _episode_lines(self) -> str:
        return "\n".join(f"- episode {e.n}: {e.task} [{e.status}]" + (f": {e.summary}" if e.summary else "")
                         for e in self.report.episodes[-self.RECENT_EPISODES:])

    def _goal_completed_since_building_entry(self) -> bool:
        if not self.guide.events:
            return False
        event = self.guide.events[-1]
        for node in self.goals.tree.nodes():
            if node.status == "complete" and node.frontier_since is not None and node.frontier_since > event.episode:
                return True
        return False

    def _knowledge_context_block(self) -> str:
        if not has_recorded_knowledge(self.knowledge):
            return ""
        current_place = self._current_place() or ""
        place_words = [
            w.lower()
            for w in re.split(r"[^A-Za-z0-9]+", current_place)
            if len(w) >= 3 and w.lower() not in {"the", "and", "area", "overworld", "city", "town", "route", "floor"}
        ]
        place_phrases = [
            m.group(0).lower()
            for m in re.finditer(
                r"\b(?:route\s+\d+|[a-z]+\s+(?:city|town|forest|cave|tower|well|gym|house|lab|island|road|mountain|mt\.?\s*[a-z]+))\b",
                current_place.lower(),
            )
        ]
        lines: List[str] = []
        for path, node in tree_entries(self.knowledge):
            if len(path) < 2:
                continue
            desc = (node.description or "").strip()
            if not desc or (len(path) == 2 and desc in DEFAULT_CATEGORY_DESCRIPTIONS):
                continue
            top = path[1]
            include = False
            if top in ("Active Story", "NPCs", "Game Lore"):
                include = True
            elif top == "Locations" or len(path) >= 2:
                path_str = " > ".join(path).lower()
                text_str = f"{path_str} {desc.lower()}"
                if top not in ("Items", "Pokemon", "Game Control Meta") and (
                    any(ph in text_str for ph in place_phrases) or any(w in path_str for w in place_words)
                ):
                    include = True
            if include:
                rendered_desc = compact_active_story(desc) if top == "Active Story" else desc
                lines.append(f"- {' > '.join(path[1:])}: {rendered_desc}")
        if not lines:
            return ""
        return "\nWhat you know from your knowledge database:\n" + "\n".join(lines) + "\n"

    def _fill(self, prompt: str) -> str:
        notepad_text = self.notepad.text.strip()
        if notepad_text and self.global_map.current:
            notepad_text = sanitize_stale_notepad_routes(
                notepad_text,
                self.global_map,
                self._current_segment_blocked_moves(),
            )
        return (prompt.replace("[GAME]", self._game)
                .replace("[RECENT_BLOCK]", self._recent_block())
                .replace("[MAX_STEPS]", str(self._supervisor_max_steps))
                .replace("[GOAL]", frontier_text(self.goals))
                .replace("[CURRENT]", self._current_place())
                .replace("[PARTY_BLOCK]", self._party_status_block())
                .replace("[NOTEPAD]", notepad_text or "(empty)")
                .replace("[KNOWLEDGE_BLOCK]", self._knowledge_context_block())
                .replace("[GUIDE]", guide_block(self.guide, self.global_map, self.start_episode + self._episodes_run,
                                                goal_completed_since_entry=self._goal_completed_since_building_entry())))

    def _map_opening_screen(self, frame: np.ndarray) -> None:
        free = self._env.get_info()["pokemon_core"]["agent_state"] == AgentState.FREE_ROAM
        status = self.global_map.on_step(frame, self._identified if free else {}, free)
        self._consolidate()
        self.global_map.mark_explored(self.start_episode + self._episodes_run)
        # People are read off the screen only where the map placed it; otherwise they are told as remembered.
        self._map_placed = status in PLACED_STATUS
        self._people_where = ("here" if self._map_placed
                              else "held" if not free and not self.global_map.faded else "away")
        if free and self._map_placed and self._identified:
            self._identified = self.tiles.backfill_masked_tiles(self._identified, frame, self.global_map)
        if self.npcs is not None:
            try:
                self.npcs.on_opening_screen(frame, self._identified if free else {}, free and self._map_placed)
            except Exception as e:  # noqa: BLE001
                log_warn(f"NPC registry skipped the opening screen: {type(e).__name__}: {e}", self._parameters)

    def _people_on_map(self, segment: int) -> list:
        if self.npcs is None:
            return []
        try:
            return self.npcs.map_people(self.global_map, segment)
        except Exception as e:  # noqa: BLE001
            log_warn(f"NPC registry: the people on the map were skipped: {type(e).__name__}: {e}", self._parameters)
            return []

    def _maybe_on_map(self, segment: int) -> list:
        if self.npcs is None:
            return []
        try:
            return self.npcs.unconfirmed(self.global_map, segment)
        except Exception as e:  # noqa: BLE001
            log_warn(f"NPC registry: the unconfirmed people were skipped: {type(e).__name__}: {e}", self._parameters)
            return []

    def _screen_people(self) -> Optional[Dict[Tuple[int, int], str]]:
        if self.npcs is None or not self.global_map.current:
            return None
        return self.npcs.screen_labels(self.global_map)

    def _name_people_on_screen(self) -> None:
        try:
            people = self._screen_people()
            if self._identified:
                kinds = cell_kinds(self._identified, self.global_map.ledge_patterns, self.global_map.walk_patterns,
                                   people, self.global_map.left_ledge_patterns,
                                   self._local_blocked_cells(), global_map=self.global_map)
                self._screen_tiles = (verbalize_tiles(self._identified, people, kinds)
                                      + "\nWalkable cells on each edge of the screen:\n"
                                      + verbalize_edges(kinds, identified=self._identified, people=people, global_map=self.global_map))
        except Exception as e:
            log_warn(f"NPC registry: naming the people on screen was skipped: {type(e).__name__}: {e}",
                     self._parameters)

    def _failure_streak(self) -> Tuple[int, List[str]]:
        streak, failed = 0, []
        for e in reversed(self.report.episodes):
            if e.status == "completed":
                break
            if e.status == "out_of_calls" and self.global_map.new_cells_by_episode.get(e.n, 0) > 0:
                break
            streak += 1
            failed.append(e.task)
        return streak, failed

    def _abandon_failing_plan(self, n: int) -> None:
        """When exploration takes over from a failing plan, abandon the generated goals that led there,
        or the planner walks straight back to them."""
        if not self._explore_now or exploring(self.global_map, n):
            return
        streak, failed = self._failure_streak()
        if streak < FAILURE_EPISODES:
            return
        frontier = self.goals.frontier()
        if frontier is None:
            return
        path = self.goals.path_to(frontier)
        generated = next((path[:i] for i in range(1, len(path) + 1)
                          if self.goals.tree.at(path[:i]).root.source not in FIXED_GOAL_SOURCES),
                         None)
        if generated is None:
            return
        top, leaf = self.goals.tree.at(generated).root.description, frontier.root.description
        tasks = "; ".join(f'"{t}"' for t in list(dict.fromkeys(t for t in failed if t))[:3])
        self.goals.set_status(generated, "abandoned",
                              f"abandoned after {streak} failed episodes in a row ({tasks}); this approach "
                              f"does not work from here", episode_number=n)
        self._remember_all([f"The plan '{top}' (which came down to '{leaf}') failed {streak} episodes in a row "
                            f"and was abandoned: that approach does not work from here, so do not plan it again."], n)
        self.global_map.count("plans_abandoned")
        log_info(f"Global map: the plan {top!r} kept failing; abandoned it and re-planning.", self._parameters)

    def _supervisor_steps(self, report: SupervisorReport, marker: Optional[Tuple[int, int]],
                          left: bool = True) -> str:
        n_notes, n_events = marker if marker is not None else (0, 0)
        lines = []
        if n_notes < self.MIN_NAMING_STEPS and len(self.report.episodes) > 1:
            earlier = self.report.episodes[-2].supervisor_report
            if earlier is not None:
                lines += [f"- (previous episode) {note}" for note in earlier.narrative[-self.MIN_NAMING_STEPS:]]
        lines += [f"- {note}" for note in report.narrative[:n_notes]]
        decisions = [e for e in report.event_log[:n_events]
                     if isinstance(e, SupervisorVLMCallRecord) and e.stage == "decision"]
        if decisions and left:
            action = clean_value(parse_key_value(str(decisions[-1].response), "Action"))
            if action:
                lines.append(f"- (under way when the screen went black) {action}")
        return "\n".join(lines)

    def _judge_merge(self, source: int, target: int, overlap: Dict[str, Any]) -> Optional[bool]:
        gmap = self.global_map
        names = (gmap.info(source).name, gmap.info(target).name)
        same = judge_merge(gmap, source, target, overlap, self._vlm_caller("merge_judge"), self._game,
                           self._parameters)
        verdict = "unanswered" if same is None else "the same place" if same else "different places"
        log_info(f"Global map: judged segments {source} '{names[0]}' and {target} '{names[1]}' {verdict}.",
                 self._parameters)
        return same

    def _consolidate(self) -> None:
        for m in self.global_map.consolidate(judge=self._judge_merge):
            log_info(f"Global map: merged segment {m['source']} into {m['target']} at {m['offset']} "
                     f"({m['matches']} of {m['shared']} squares match).", self._parameters)

    def _name_segments(self, report: SupervisorReport, n: int) -> bool:
        gmap = self.global_map
        self._consolidate()
        gmap.queue_grown()
        queue = gmap.take_naming_queue()
        tasks = self._episode_lines()
        done = set()
        named_changed = False
        for entry in queue:
            left = entry.get("reason", "door") == "door"
            if left:
                segment, mark = gmap.resolve(entry["segment"], entry["exit"])
            else:
                segment = gmap.resolve(entry["segment"], (0, 0))[0]
                mark = gmap.pos if gmap.segment == segment else None
            if segment in done or not gmap.segments[segment] or (left and gmap.info(segment).name is not None):
                continue
            done.add(segment)
            marker = entry["marker"] if left else (len(report.narrative), len(report.event_log))
            before = gmap.info(segment).name
            answer = name_segment(gmap, segment, mark, left, tasks, self._supervisor_steps(report, marker, left),
                                  self._vlm_caller("segment_name"), self._game, n, self._parameters)
            if answer is None:
                continue
            changed, name, description = answer
            if before is None:
                named_changed = True
                log_info(f"Global map: named segment {segment} '{name}': {description}", self._parameters)
            elif name != before:
                named_changed = True
                log_info(f"Global map: renamed segment {segment} '{before}' -> '{name}': {description}",
                         self._parameters)
            elif changed:
                log_info(f"Global map: updated description of segment {segment} '{name}': {description}",
                         self._parameters)
            else:
                log_info(f"Global map: kept the name of segment {segment} '{name}'.", self._parameters)
        return named_changed

    def _global_map_block(self) -> str:
        if not self.global_map.current:
            return ""
        local = (cell_kinds(self._identified, self.global_map.ledge_patterns, self.global_map.walk_patterns,
                            self._screen_people(), self.global_map.left_ledge_patterns,
                            self._local_blocked_cells(), global_map=self.global_map)
                 if self._identified else None)
        self.global_map.prune_routes()
        _, failed = self._failure_streak()
        gmap = self.global_map
        marks, maybe = None, []
        if self.npcs is not None:
            here = gmap.resolve(gmap.segment, gmap.pos)[0]
            marks = [m for m in self._people_on_map(here) if self.npcs._episode - m.last_seen[0] <= STALE_EPISODES]
            maybe = self._maybe_on_map(here)
        block, override = prompt_block(self.global_map, list(self.global_map.new_cells_by_episode), local,
                                       explore=self._explore_now,
                                       episode=self.start_episode + self._episodes_run,
                                       failed_tasks=failed,
                                       blocked_moves=self._current_segment_blocked_moves())
        if override is not None:
            keep_exploring(self.global_map, override["until"])
        if marks is not None:
            block = with_people(block, gmap, marks, self._people_where, maybe)
        if override is not None and override["kind"] == "edge":
            log_info(f"Global map: exploration overrides the task: head to ({override['dx']}, {override['dy']}) "
                     f"{override['sector']} in {self.global_map.area_label(override['area'])}.", self._parameters)
        elif override is not None:
            names = ", ".join(self.global_map.area_label(si) for si in override["areas"])
            log_info(f"Global map: exploration overrides the task: nothing left here, head for one of: {names}.",
                     self._parameters)
        back = self.global_map.defeat_return
        if back is not None and self.start_episode + self._episodes_run - back["episode"] <= self.RETURN_EPISODES:
            block += "\n" + (P.DEFEAT_RETURN_LINE.replace("[EPISODE]", str(back["episode"]))
                             .replace("[WHERE]", self.global_map.defeat_route(back["defeat"])))
        return "\n\n" + block

    def _check_stuck(self, frame: np.ndarray, n: int) -> bool:
        gmap = self.global_map
        if exploring(gmap, n):
            return False
        streak, _ = self._failure_streak()
        signs = stuck_signs(gmap, list(gmap.new_cells_by_episode), streak)
        if signs is None:
            return False
        stuck = judge_stuck(gmap, signs, self._episode_lines(), frame, self._vlm_caller("stuck_check"), self._game,
                            self._parameters)
        gmap.count("stuck_checks")
        gmap.count("stuck_yes", bool(stuck))
        verdict = "unanswered" if stuck is None else "stuck: exploring" if stuck else "not stuck"
        log_info(f"Global map: {signs}; the strategist judged itself {verdict}.", self._parameters)
        return bool(stuck)

    def _people_block(self) -> str:
        if self.npcs is None or not self.global_map.current:
            return ""
        try:
            return self.npcs.people_block(self.global_map, where=self._people_where)
        except Exception as e:
            log_warn(f"NPC registry: the people block was skipped: {type(e).__name__}: {e}", self._parameters)
            return ""

    def _dispatch(self, frame: np.ndarray) -> Tuple[Optional[str], Optional[str]]:
        prompt = (self._fill(P.DISPATCH_PROMPT)
                  .replace("[SCREEN_TILES]", self._screen_tiles or "(nothing was recognised)")
                  .replace("[GLOBAL_MAP]", self._global_map_block())
                  .replace("[PEOPLE]", self._people_block()))
        dispatched = ask(self._vlm_caller("dispatch"), prompt, _parse_dispatch, "Dispatch",
                         self.MAX_DISPATCH_ATTEMPTS, self._parameters, images=[frame])
        return dispatched if dispatched is not None else (None, None)

    def _review(self, digest: str, fallback_summary: Optional[str] = None) -> Tuple[str, Optional[str]]:
        new_doors = self.global_map.new_entrances(self.guide.towns)
        cleaned_fallback = (fallback_summary or "").strip()
        if not cleaned_fallback and digest:
            for line in digest.splitlines():
                if line.startswith("Final check:"):
                    cleaned_fallback = line[len("Final check:"):].strip()
                    break
        if not new_doors and cleaned_fallback:
            self.global_map.mark_entrances_seen()
            return cleaned_fallback, None
        question, answer = town_question(self.guide) if new_doors else ("", "")
        prompt = (self._fill(P.REVIEW_PROMPT).replace("[DIGEST]", digest)
                  .replace("[TOWN_QUESTION]", question).replace("[TOWN_FORMAT]", answer))
        output = self._vlm_call("review", texts=prompt)
        self.global_map.mark_entrances_seen()
        summary = clean_value(parse_key_value(output, "Summary")) or cleaned_fallback
        town = clean_value(parse_key_value(output, "Town")) if new_doors else None
        if new_doors:
            log_info(f"Strategy guide: {new_doors} new door square(s) on the map; asked about a new town, "
                     f"answer: {town}.", self._parameters)
        return summary, None if town is None or town.strip().rstrip(".").lower() == "none" else town

    def _should_extract_knowledge(self, report: SupervisorReport, conversation_digest: str,
                                  closed: List[Tuple[Tuple[str, ...], str]], town_event: Any,
                                  episode_status: Optional[str] = None,
                                  episode_number: Optional[int] = None) -> bool:
        if episode_status == "lost_battle":
            return False
        if not (report.narrative or any(r.notes and r.notes.strip() for r in report.executor_reports)):
            return False
        if conversation_digest.strip():
            if self.npcs is None or episode_number is None or self.npcs.has_noteworthy_conversation(episode_number):
                return True
        if town_event is not None:
            return True
        return False

    @staticmethod
    def _should_update_guide(events: List[Any], episode_status: str) -> bool:
        if any(getattr(e, "trigger", None) == NEW_TOWN for e in events):
            return True
        return episode_status != "completed"

    def _extract_knowledge(self, digest: str, before: np.ndarray, after: np.ndarray, n: int) -> None:
        prompt = (self._fill(P.KNOWLEDGE_EXTRACT_PROMPT)
                  .replace("[DIGEST]", digest)
                  .replace("[MAX_FACTS]", str(self.MAX_FACTS)))
        facts = ask(self._vlm_caller("extract"), prompt, _parse_facts, "Knowledge extraction",
                    self.MAX_EXTRACT_ATTEMPTS, self._parameters, images=[before, after])
        self._remember_all(facts or [], n)

    @staticmethod
    def _battle_log(report: SupervisorReport) -> Tuple[str, List[np.ndarray]]:
        lines, frames = [], []
        for executor in report.executor_reports:
            if executor.executor_name != CombatExecutor.__name__:
                continue
            lines.append(f"Orders for the fight: {executor.task}")
            for call in executor.vlm_call_log:
                if not isinstance(call, ExecutorVLMCallRecord) or call.tag not in ("action", CombatExecutor.STEP_TAG):
                    continue
                response = str(call.response)
                action = clean_value(parse_key_value(response, "Action")) or "?"
                move = clean_value(parse_key_value(response, "Move"))
                reason = " ".join((clean_value(parse_key_value(response, "Reasoning")) or "").split())
                lines.append(f"- {action}" + (f" move {move}" if move and move.lower() != "none" else "")
                             + (f": {reason[:MAX_BATTLE_REASON]}" if reason else ""))
            if executor.notes and executor.notes.strip():
                lines.append(f"Battle text seen: {executor.notes.strip()}")
            steps = [s for s in executor.steps if isinstance(s, EnvironmentStepRecord)]
            if steps:
                frames = [steps[0].frame_before, steps[-1].frame_before]
        return ("\n".join(lines) or "(no battle was recorded)"), frames

    def _reflect_on_defeat(self, report: SupervisorReport, task: str, n: int) -> None:
        gmap = self.global_map
        defeat = gmap.defeats[-1] if gmap.defeats else None
        where = gmap.defeat_route(defeat) if defeat is not None else "not known"
        battle, images = self._battle_log(report)
        answer = reflect_on_defeat(self.guide, task, frontier_text(self.goals), where, report.digest, battle,
                                   images, self._vlm_caller("defeat_reflection"), self._game, n, self._parameters)
        if answer is None:
            log_warn("Defeat reflection gave no usable reply; nothing was changed.", self._parameters)
            return
        place = f" ({gmap.area_label(gmap.resolve(defeat['segment'], defeat['pos'])[0])})" if defeat else ""
        self._remember_all([f"The player lost a battle in episode {n}{place}: {answer['wrong']}"], n)
        if answer["prepare"]:
            frontier = self.goals.frontier()
            if frontier is not None:
                path = self.goals.path_to(frontier)
                details = f"Added after losing a battle in episode {n}: {answer['wrong']}"
                self.goals.add_under(path, GoalNode(description=answer["prepare"], details=details, source="defeat"),
                                     episode_number=n)
                self.goals.edit(path, achievable_without_subgoals=None, episode_number=n)
                new_frontier_path = self.goals.path_to(self.goals.frontier())
                self.goals.set_frontier(new_frontier_path, n, episode_number=n)
                if new_frontier_path is not None:
                    self.goals.edit(new_frontier_path, achievable_without_subgoals=n, episode_number=n)
        if answer["return"] and defeat is not None:
            gmap.return_to_defeat(defeat, n)
        log_info(f"Defeat reflection: {answer['wrong']} Plan: "
                 f"{'PREPARE ' + answer['prepare'] if answer['prepare'] else 'CONTINUE'}; "
                 f"return: {'yes' if answer['return'] else 'no'}; {len(answer['changes'])} tip change(s).",
                 self._parameters)

    def _remember_all(self, facts: List[str], n: int) -> None:
        remember_all(self.knowledge, facts, self._vlm_caller("knowledge"), self._game, n, self._parameters)

    def _explore_context_block(self) -> str:
        from execution.strategist.pokemon.global_map.text import door_lines

        if not self.global_map.current:
            return "(none)"
        self.global_map.mark_explored(self.start_episode + self._episodes_run)
        local = (cell_kinds(self._identified, self.global_map.ledge_patterns, self.global_map.walk_patterns,
                            self._screen_people(), self.global_map.left_ledge_patterns,
                            self._local_blocked_cells(), global_map=self.global_map)
                 if self._identified else None)
        seg_blocked_moves = self._current_segment_blocked_moves()
        lines = list(door_lines(self.global_map, local, blocked_moves=seg_blocked_moves))
        exp = explore_text(self.global_map, local, blocked_moves=seg_blocked_moves)
        if exp and exp != "(none)":
            lines.append(exp)
        return "\n".join(lines) or "(none)"

    def _update_notepad_deterministically(self, summary: str, n: int) -> None:
        notes = self.notepad.text.strip()
        if self.global_map.current and notes:
            notes = sanitize_stale_notepad_routes(
                notes,
                self.global_map,
                self._current_segment_blocked_moves(),
            )
            notes = sanitize_stale_notepad_bag_status(notes, self._healing_bag_empty)
        frontier = self.goals.frontier()
        goal_line = goal_path_text(self.goals.path_to(frontier)) if frontier is not None else "(every goal is finished)"
        place = self._current_place()
        clean_summary = " ".join((summary or "").split()) or "completed routine step."
        latest_line = f"- Latest (episode {n} at {place}): {clean_summary}"
        lines = [
            ln for ln in (notes.splitlines() if notes else [])
            if not re.match(r"^\s*-\s*Latest\s*\(episode\s+\d+\b", ln, re.IGNORECASE)
        ]
        replaced_goal = False
        for idx, ln in enumerate(lines):
            if re.match(r"^\s*-\s*Current\s+Goal\s*:", ln, re.IGNORECASE):
                lines[idx] = f"- Current Goal: {goal_line}."
                replaced_goal = True
                break
        if not replaced_goal and not lines:
            lines.append(f"- Current Goal: {goal_line}.")
        lines.append(latest_line)
        self.notepad.update("\n".join(lines), episode_number=n)
    def _update_notepad(self, digest: str, summary: str, n: int) -> None:
        prompt = (self._fill(P.NOTEPAD_UPDATE_PROMPT)
                  .replace("[DIGEST]", digest)
                  .replace("[SUMMARY]", summary or "(none)")
                  .replace("[EXPLORE]", self._explore_context_block()))
        update = ask(self._vlm_caller("notepad_update"), prompt, _parse_notepad_update, "Notepad update",
                     self.MAX_NOTEPAD_ATTEMPTS, self._parameters)
        if update is None:
            log_warn("No usable notepad update; leaving the notepad as it was.", self._parameters)
            return
        action, notes = update
        if self.global_map.current:
            notes = sanitize_stale_notepad_routes(
                notes,
                self.global_map,
                self._current_segment_blocked_moves(),
            )
        notes = sanitize_stale_notepad_bag_status(notes, self._healing_bag_empty)
        if action == "APPEND":
            self.notepad.add(notes, episode_number=n)
        else:
            self.notepad.update(notes, episode_number=n)

    def _absorb_notepad(self, closed: List[Tuple[Tuple[str, ...], str]], n: int) -> None:
        if self.notepad.text.strip():
            prompt = (self._fill(P.NOTEPAD_EXTRACT_PROMPT)
                      .replace("[CLOSED]", "\n".join(f"- {goal_path_text(path)} [{status}]" for path, status in closed))
                      .replace("[MAX_FACTS]", str(self.MAX_FACTS)))
            facts = ask(self._vlm_caller("notepad_extract"), prompt, _parse_facts, "Notepad extraction",
                        self.MAX_EXTRACT_ATTEMPTS, self._parameters)
            self._remember_all(facts or [], n)
        self.notepad.clear(episode_number=n)

    def save_all(self, episode_number: int) -> None:
        self.party.touch(episode_number)
        artifacts = [self.tiles, self.goals, self.knowledge, self.notepad, self.guide, self.global_map, self.party]
        if self.npcs is not None:
            artifacts.append(self.npcs)
        for artifact in artifacts:
            artifact.save(self._dir, episode_number, parameters=self._parameters)
        state_name = self._emulator_state_name(self._name, episode_number)
        self._env.save_custom_state(state_name)
        memory = [type(artifact).latest_path(self._dir, episode_number) for artifact in artifacts]
        self._record.checkpoint(checkpoint_id=episode_number, episode=episode_number, steps=self._env_steps(),
                                server_save=state_name, memory=[path for path in memory if path])

    def _env_steps(self) -> Optional[int]:
        return getattr(getattr(self._env, "_emulator", None), "step_count", None)

    def _record_supervisor(self, events: List[Any], n: int) -> None:
        for event in events:
            if isinstance(event, SupervisorVLMCallRecord):
                if event.stage == "perception":
                    continue
                self._record.event(episode=n, layer="supervisor", kind="plan", steps=self._env_steps(),
                                   text=f"[{event.stage}] {event.response}")
                continue
            actions = ", ".join(action_name(step) if isinstance(step, EnvironmentStepRecord) else "invalid reply"
                                for step in event.steps)
            hint = (event.init_kwargs or {}).get("hint")
            self._record.event(episode=n, layer="executor", kind="attempt", steps=self._env_steps(),
                               text=f"Leg: {event.task}" + (f"\nHint: {hint}" if hint else "")
                               + f"\nActions: {actions or 'none'}\nEnded: {event.termination_reason}")

    @staticmethod
    def _emulator_state_name(run_name: str, episode_number: int) -> str:
        # Alphanumeric only (save_custom_state rejects anything else); sync_data matches the prefix.
        return f"strategist{treat_name(run_name)}{episode_number}"

    def _stop_reason(self) -> Optional[str]:
        if self._episodes_run >= self._max_episodes:
            return "max_episodes"
        if self.goals.tree.root.status == "complete":
            return "goals_complete"
        if self._env_done():
            return "env_done"
        return None

    def run(self) -> StrategistReport:
        if self._ran:
            log_error(f"{self.__class__.__name__}.run() called twice on one instance. The "
                      "report accumulates, so this would merge two runs into one record.",
                      self._parameters)
        self._ran = True
        progress = tqdm(total=self._max_episodes, desc="episodes", unit="ep")
        while True:
            stop = self._stop_reason()
            if stop is not None:
                self.report.stop_reason = stop
                progress.set_postfix_str(f"stopped: {stop}")
                break
            record = self._run_episode(self.start_episode + self._episodes_run)
            self._episodes_run += 1
            progress.update(1)
            progress.set_postfix_str(record.status)
        progress.close()
        return self.report

    def _plan_goals(self, n: int) -> None:
        subtree = self.goals.frontier()
        if subtree is None:
            return
        path = self.goals.path_to(subtree)
        since = subtree.root.frontier_since
        if path != self.goals.frontier_path or since is None:
            self.goals.set_frontier(path, n, episode_number=n)
            if subtree.root.source in ("subgoal_creation", "defeat"):
                self.goals.edit(path, achievable_without_subgoals=n, episode_number=n)
                return
            unsuccessful_for = 0
        else:
            if (subtree.root.achievable_without_subgoals is not None
                    or subtree.root.source in ("subgoal_creation", "defeat")) and not self._explore_now:
                if subtree.root.achievable_without_subgoals is None:
                    self.goals.edit(path, achievable_without_subgoals=n, episode_number=n)
                return
            unsuccessful_for = n - since
            if unsuccessful_for <= 0 or unsuccessful_for % self._subgoal_every:
                return
        self.global_map.clear_routes()
        local = (cell_kinds(self._identified, self.global_map.ledge_patterns, self.global_map.walk_patterns,
                            self._screen_people(), self.global_map.left_ledge_patterns,
                            self._local_blocked_cells(), global_map=self.global_map)
                 if self._identified else None)
        create_subgoals(self.goals, self.knowledge, self._current_place(),
                        explore_text(self.global_map, local, blocked_moves=self._current_segment_blocked_moves()),
                        self.notepad.text,
                        lambda request: navigate_map(self.global_map, request, self._vlm_caller, self._game,
                                                     self._parameters),
                        self._vlm_caller, self._game, n, unsuccessful_for, self._parameters)

    def _run_episode(self, n: int) -> EpisodeRecord:
        calls_from = len(self.report.event_log)
        frame = self._current_frame()
        self._perceive(frame, n)
        if self.npcs is not None:
            self.npcs.begin_episode(n, self.global_map)
        self._map_opening_screen(frame)
        self._name_people_on_screen()
        self._explore_now = self._check_stuck(frame, n)
        self._abandon_failing_plan(n)
        self._plan_goals(n)
        task, guidance = self._dispatch(frame)

        record = EpisodeRecord(n=n, task=task or "", guidance=guidance)
        self.report.episodes.append(record)

        if task is None:
            record.status = "no_task"
            record.summary = "The strategist could not write a task from this screen."
            self._finish_episode(record, calls_from)
            return record

        self._record.event(episode=n, layer="strategist", kind="intention", steps=self._env_steps(),
                           text=f"Task: {task}" + (f"\nGuidance: {guidance}" if guidance else ""))
        recorded = 0

        def record_so_far(_event=None) -> None:
            nonlocal recorded
            events = supervisor.report.event_log
            self._record_supervisor(events[recorded:], n)
            recorded = len(events)

        listeners = [self.npcs.on_env_step] if self.npcs is not None else []
        env = MappingEnvironment(self._env, self.global_map, self.tiles, self._parameters,
                                 listeners=listeners + [record_so_far])
        step_from = self.global_map.step
        supervisor = PokemonPlayThroughSupervisor(
            task=task, env=env, game=self._game, max_steps=self._supervisor_max_steps,
            tile_recognizer=self.tiles, episode_number=n, guidance=guidance, parameters=self._parameters,
            max_new_tokens=self._supervisor_max_new_tokens, battle_tips=battle_tips(self.guide),
            screen_people=self._screen_people if self.npcs is not None else None,
            party_state=self.party,
            initial_blocked_moves=self._blocked_moves,
            **self._supervisor_kwargs,
        )
        env.marker = lambda: (len(supervisor.report.narrative), len(supervisor.report.event_log))
        supervisor._on_event_logged = record_so_far
        result = supervisor.evaluate()
        report = result["report"]
        supervisor_summary = str(result.get("summary") or "").strip()
        record.supervisor_report = report
        record.status = str(result.get("status", "unknown"))
        self.report.event_log.append(report)
        record_so_far()
        new_blocked_moves = False
        for b_place, by_dir in (getattr(supervisor, "_blocked_moves", None) or {}).items():
            dst_by_dir = self._blocked_moves.setdefault(b_place, {})
            for b_dir, notes in by_dir.items():
                dst_list = dst_by_dir.setdefault(b_dir, [])
                for note in notes:
                    if note not in dst_list:
                        dst_list.append(note)
                        new_blocked_moves = True
        if report is not None and report.narrative:
            if extract_blocked_moves_from_narrative(report.narrative, end_pos=self.global_map.pos):
                new_blocked_moves = True
            self._ingest_narrative_blocked_moves(
                report.narrative,
                segment=self.global_map.segment,
                end_pos=self.global_map.pos,
            )
        # Let a mid-door-warp black screen clear, so the next task starts from the new place.
        settle_screen(self._env)
        named_new_segment = self._name_segments(report, n)
        self._read_conversations()
        new_warps = [w for w in self.global_map.warps if w[0] > step_from]
        events = record_events(self.guide, self.global_map, new_warps, n, self._parameters)

        before, frame = frame, self._current_frame()
        self._perceive(frame, n)
        self.global_map.refresh(self.tiles._by_hash)

        self._record_party_episode(record)
        if record.status == "lost_battle":
            closed = []
        else:
            closed = check_goals(self.goals, report.digest, self.notepad.text, self._vlm_caller, self._game, n,
                                 current=self._current_place(), parameters=self._parameters,
                                 episode_status=record.status, subgoal_every=self._subgoal_every,
                                 explore=self._explore_context_block())
        summary, town = self._review(report.digest, fallback_summary=supervisor_summary)
        record.summary = summary or record.summary
        town_event = record_new_town(self.guide, town, self._current_place(), n, self._parameters)
        if town_event is not None:
            events.append(town_event)
        conversation_digest = self._conversation_digest(n)
        extracted = False
        if self._should_extract_knowledge(report, conversation_digest, closed, town_event, record.status, n):
            self._extract_knowledge(report.digest + conversation_digest, before, frame, n)
            extracted = True
        if closed:
            if any(status == "complete" for _, status in closed):
                self.guide.mark_goal_completed(self.global_map, episode_number=n)
            if extracted or any(len(path) <= 3 for path, _ in closed):
                self.notepad.clear(episode_number=n)
        if (
            record.status == "completed"
            and not extracted
            and not named_new_segment
            and not new_blocked_moves
            and not any(len(path) <= 3 for path, _ in closed)
            and self.notepad.text.strip()
        ):
            self._update_notepad_deterministically(record.summary, n)
        else:
            self._update_notepad(report.digest, summary, n)
        if events and self._should_update_guide(events, record.status):
            update_guide(self.guide, events, report.digest, self._vlm_caller("guide_update"), self._game, n,
                         self._parameters)
        if record.status == "lost_battle":
            self._reflect_on_defeat(report, task, n)

        self._finish_episode(record, calls_from)
        return record

    def _read_conversations(self) -> None:
        if self.npcs is None:
            return
        try:
            self.npcs.prepare_episode_summaries(self.global_map, vlm_caller=self._vlm_caller)
        except Exception as e:
            log_warn(f"NPC registry: reading the conversations failed and was skipped: "
                     f"{type(e).__name__}: {e}", self._parameters)

    def _conversation_digest(self, n: int) -> str:
        if self.npcs is None:
            return ""
        try:
            return self.npcs.digest_lines(n, self.global_map)
        except Exception as e:
            log_warn(f"NPC registry: the conversation digest was skipped: {type(e).__name__}: {e}", self._parameters)
            return ""

    def _archive_episode(self, record: EpisodeRecord, calls_from: int) -> None:
        calls = [event for event in self.report.event_log[calls_from:]
                 if isinstance(event, StrategistVLMCallRecord)]
        archive = EpisodeArchive(
            game=self._game,
            strategist_name=self.__class__.__name__,
            init_kwargs=self._run_config(),
            detail=self._report_detail,
            episode=trim_episode(record, self._report_detail),
            strategist_calls=calls,
        )
        save_episode_report(self._dir, record.n, archive, self._parameters)

    def _finish_episode(self, record: EpisodeRecord, calls_from: int = 0) -> None:
        self.global_map.refresh(self.tiles._by_hash)
        added = self.global_map.end_episode(record.n, judge=self._judge_merge, episode_number=record.n)
        log_info(f"Global map: {added} new squares this episode, {self.global_map.n_cells} in all, "
                 f"{self.global_map.n_areas} area(s).", self._parameters)
        if self.npcs is not None:
            try:
                self.npcs.end_episode(self.global_map, vlm_caller=self._vlm_caller, episode_number=record.n)
            except Exception as e:
                log_warn(f"NPC registry end-of-episode failed and was skipped: {type(e).__name__}: {e}",
                         self._parameters)
        self._record.event(episode=record.n, layer="strategist", kind="outcome", steps=self._env_steps(),
                           text=f"Status: {record.status}. {record.summary}")
        self.save_all(record.n)
        self._archive_episode(record, calls_from)


def _parse_dispatch(output: str) -> Tuple[Any, Optional[str]]:
    task = clean_value(parse_key_value(output, "Task"))
    if task is None:
        return None, "The `Task:` line was missing or empty. It must be one concrete sentence."
    return (task, clean_value(parse_key_value(output, "Guidance"))), None


def _parse_notepad_update(output: str) -> Tuple[Any, Optional[str]]:
    action, problem = single_value(output, "Action")
    action = (action or "").rstrip(".").strip().upper()
    if problem is not None or action not in ("APPEND", "REWRITE"):
        return None, "There must be exactly one `Action:` line, saying APPEND or REWRITE."
    lines = output.splitlines()
    starts = [i for i, line in enumerate(lines) if line.replace("**", "").strip().lower().startswith("notes:")]
    if len(starts) != 1:
        return None, f"There must be exactly one `Notes:` line, found {len(starts)}."
    first = lines[starts[0]].replace("**", "").strip()[len("notes:"):].strip()
    body = [first] + lines[starts[0] + 1:]
    notes = "\n".join(line for line in body if line.strip() != "[STOP]").strip()
    if not notes:
        return None, "The notes under `Notes:` were empty."
    return (action, notes), None


def _parse_facts(output: str) -> Tuple[Any, Optional[str]]:
    if "facts:" not in output.lower():
        return None, "There was no `Facts:` line. Give the facts as a list under it, or write `Facts: None`."
    facts = [fact for fact in (clean_value(item) for item in parse_list(output, "Facts")) if fact]
    if len(facts) > PokemonStrategist.MAX_FACTS:
        return None, f"{len(facts)} facts were given. Write at most {PokemonStrategist.MAX_FACTS}."
    return facts, None
