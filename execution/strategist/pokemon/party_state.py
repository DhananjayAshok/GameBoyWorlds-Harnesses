from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from execution.artifact import StrategistArtifact
from execution.perception.pokemon.battle import (
    has_player_pokemon_fainted_text,
    is_battle_defeat_text,
    is_healing_item_name,
    is_status_heal_item_name,
)
from execution.report import ExecutorVLMCallRecord


class PartyInventoryState(StrategistArtifact):
    BAG_ITEMS_TAG_PATTERN = re.compile(
        r"\[Bag items seen:\s*([^\]]+)\]",
        re.IGNORECASE,
    )
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
    BATTLE_CHUNK_PATTERN = re.compile(
        r"\b(?:Wild\s+[A-Z0-9♀♂.'-]+\s+appeared|wants\s+to\s+(?:battle|fight)"
        r"|sent\s+out\s+[A-Z0-9♀♂.'-]+!|used\s+[A-Z0-9♀♂.' -]+!|Got\s+away\s+safely!)",
        re.IGNORECASE,
    )
    USED_MOVE_PATTERN = re.compile(r"\bused\s+[A-Z0-9♀♂.' -]+!")

    def __init__(
        self,
        *,
        items: Optional[List[str]] = None,
        quantities: Optional[Dict[str, Optional[int]]] = None,
        saw_top: bool = False,
        saw_bottom: bool = False,
        healing_exhausted: bool = False,
        status_heal_exhausted: bool = False,
        last_battle_hp: Optional[Tuple[int, int]] = None,
        last_battle_hp_episode: Optional[int] = None,
        fought_since_heal: int = 0,
        fled_since_heal: int = 0,
        last_heal_episode: Optional[int] = None,
        fought_since_last_hp: int = 0,
        active_pokemon_status: Optional[str] = None,
        poison_step_remainder: int = 0,
        zero_pp_moves: Optional[List[str]] = None,
        learned_moves: Optional[List[str]] = None,
        learned_move_holders: Optional[Dict[str, str]] = None,
    ) -> None:
        self.items: List[str] = []
        self.quantities: Dict[str, Optional[int]] = {}
        for name in (items or ()):
            norm, q = self.parse_item_token(name)
            if norm and norm not in self.items:
                self.items.append(norm)
                if q is not None:
                    self.quantities[norm] = q
        for raw_k, raw_v in (quantities or {}).items():
            norm, q = self.parse_item_token(raw_k)
            if norm:
                if norm not in self.items:
                    self.items.append(norm)
                self.quantities[norm] = raw_v if raw_v is not None else q
        self.saw_top: bool = bool(saw_top)
        self.saw_bottom: bool = bool(saw_bottom)
        self.healing_exhausted: bool = bool(healing_exhausted)
        self.status_heal_exhausted: bool = bool(status_heal_exhausted)
        self.last_battle_hp: Optional[Tuple[int, int]] = last_battle_hp
        self.last_battle_hp_episode: Optional[int] = last_battle_hp_episode
        self.fought_since_heal: int = int(fought_since_heal)
        self.fled_since_heal: int = int(fled_since_heal)
        self.last_heal_episode: Optional[int] = last_heal_episode
        self.fought_since_last_hp: int = int(fought_since_last_hp)
        self.active_pokemon_status: Optional[str] = active_pokemon_status or None
        self.poison_step_remainder: int = int(poison_step_remainder)
        self.zero_pp_moves: List[str] = [m for m in (zero_pp_moves or ()) if m]
        self.learned_moves: List[str] = []
        self.learned_move_holders: Dict[str, str] = {}
        for m in (learned_moves or ()):
            self.record_learned_move(m)
        for raw_m, raw_p in (learned_move_holders or {}).items():
            self.record_learned_move(raw_m, raw_p)
        self.active_move_names: Dict[int, str] = {}
        self.active_move_types: Dict[int, str] = {}
        self.active_pokemon_level: Optional[int] = None
        self.last_defeat_episode: Optional[int] = None
        self.last_defeat_party_level: Optional[int] = None
        self.last_defeat_healing_bag_empty: bool = False
        self.learned_move_since_defeat: bool = False
        self.tried_or_incompatible_tms: List[str] = []

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        if not hasattr(self, "learned_moves") or self.learned_moves is None:
            self.learned_moves = []
        if not hasattr(self, "learned_move_holders") or self.learned_move_holders is None:
            self.learned_move_holders = {}
        if not hasattr(self, "active_move_names") or self.active_move_names is None:
            self.active_move_names = {}
        if not hasattr(self, "active_move_types") or self.active_move_types is None:
            self.active_move_types = {}
        if not hasattr(self, "active_pokemon_level"):
            self.active_pokemon_level = None
        if not hasattr(self, "last_defeat_episode"):
            self.last_defeat_episode = None
        if not hasattr(self, "last_defeat_party_level"):
            self.last_defeat_party_level = None
        if not hasattr(self, "last_defeat_healing_bag_empty"):
            self.last_defeat_healing_bag_empty = False
        if not hasattr(self, "learned_move_since_defeat"):
            self.learned_move_since_defeat = False
        if not hasattr(self, "tried_or_incompatible_tms") or self.tried_or_incompatible_tms is None:
            self.tried_or_incompatible_tms = []

    @staticmethod
    def normalize_move_name(move_name: Optional[str]) -> str:
        if not move_name:
            return ""
        clean = " ".join(str(move_name).strip().strip('!.?"\'').split()).upper()
        if len(clean) < 3 or clean in ("NONE", "CANCEL", "YES", "NO", "USE", "QUIT", "TOSS"):
            return ""
        if not re.match(r"^[A-Z][A-Z0-9 -]{2,14}$", clean):
            return ""
        return clean

    def record_learned_move(
        self,
        move_name: Optional[str],
        pokemon_name: Optional[str] = None,
        episode_number: Optional[int] = None,
    ) -> bool:
        norm_move = self.normalize_move_name(move_name)
        if not norm_move:
            return False
        norm_poke = " ".join(str(pokemon_name or "").strip().strip('!.?"\'').split()).upper()
        if norm_poke in ("", "ENEMY", "WILD", "FOE"):
            norm_poke = ""
        for existing in list(self.learned_moves):
            ex_holder = self.learned_move_holders.get(existing, "")
            if existing != norm_move and (not norm_poke or not ex_holder or ex_holder == norm_poke):
                if norm_move.startswith(existing):
                    idx = self.learned_moves.index(existing)
                    self.learned_moves[idx] = norm_move
                    self.learned_move_holders.pop(existing, None)
                    if norm_poke:
                        self.learned_move_holders[norm_move] = norm_poke
                    elif ex_holder:
                        self.learned_move_holders[norm_move] = ex_holder
                    if episode_number is not None:
                        self.last_written = episode_number
                    return True
                if existing.startswith(norm_move):
                    if norm_poke and not ex_holder:
                        self.learned_move_holders[existing] = norm_poke
                    return False
        is_new = norm_move not in self.learned_moves
        if is_new:
            self.learned_moves.append(norm_move)
            if getattr(self, "last_defeat_episode", None) is not None and (
                episode_number is None or episode_number >= self.last_defeat_episode
            ):
                self.learned_move_since_defeat = True
        if norm_poke:
            self.learned_move_holders[norm_move] = norm_poke
        if episode_number is not None:
            self.last_written = episode_number
        return True

    def record_forgotten_move(
        self,
        move_name: Optional[str],
        pokemon_name: Optional[str] = None,
        episode_number: Optional[int] = None,
    ) -> bool:
        norm_move = self.normalize_move_name(move_name)
        if not norm_move:
            return False
        norm_poke = " ".join(str(pokemon_name or "").strip().strip('!.?"\'').split()).upper()
        removed = False
        for existing in list(self.learned_moves):
            ex_holder = self.learned_move_holders.get(existing, "")
            if existing == norm_move or existing.startswith(norm_move):
                if not norm_poke or not ex_holder or ex_holder == norm_poke:
                    self.learned_moves.remove(existing)
                    self.learned_move_holders.pop(existing, None)
                    removed = True
        if removed and episode_number is not None:
            self.last_written = episode_number
        return removed

    def format_learned_moves(self) -> List[str]:
        out: List[str] = []
        for move in getattr(self, "learned_moves", ()) or ():
            holder = (getattr(self, "learned_move_holders", None) or {}).get(move)
            out.append(f"{holder} knows {move}" if holder else move)
        return out

    def update_learned_moves_from_text(
        self,
        text: Optional[str],
        episode_number: Optional[int] = None,
    ) -> bool:
        if not text:
            return False
        changed = False
        for m_incompat in re.finditer(
            r"\[TM/HM incompatible with party in menu:\s*(TM\s*\d+|HM\s*\d+)\b",
            text,
            re.IGNORECASE,
        ):
            self.record_tried_or_incompatible_tm(m_incompat.group(1), episode_number=episode_number)
        for m_tag in re.finditer(
            r"\[Move (?:taught|already known) in menu:\s*([A-Z0-9♀♂.'-]+)\s+(?:learned|knows)\s+([A-Z][A-Z0-9 -]{2,14})\]",
            text,
            re.IGNORECASE,
        ):
            if self.record_learned_move(m_tag.group(2), m_tag.group(1), episode_number=episode_number):
                changed = True
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith("presses:") or line.lower().startswith("reasoning:"):
                continue
            m_forgot = re.match(
                r"^([A-Z][A-Z0-9♀♂.'-]{1,10})\s+forgo(?:t)?\s+([A-Z][A-Z0-9 -]{2,14})[!.]?$",
                line,
            )
            if m_forgot:
                if self.record_forgotten_move(m_forgot.group(2), m_forgot.group(1), episode_number=episode_number):
                    changed = True
                continue
            m_learn = re.match(
                r"^([A-Z][A-Z0-9♀♂.'-]{1,10})\s+(?:learned|(?:already\s+)?knows)\s+([A-Z][A-Z0-9 -]{2,14})[!.]?$",
                line,
            )
            if m_learn:
                if self.record_learned_move(m_learn.group(2), m_learn.group(1), episode_number=episode_number):
                    changed = True
                continue
            m_field = re.match(
                r"^(?!Enemy\b|Wild\b|Foe\b)([A-Z][A-Z0-9♀♂.'-]{1,10})\s+used\s+"
                r"(CUT|FLASH|SURF|STRENGTH|FLY|WHIRLPOOL|WATERFALL|ROCK SMASH|HEADBUTT|DIG|TELEPORT|SOFTBOILED|MILK DRINK|SWEET SCENT)[!.]?$",
                line,
            )
            if m_field:
                if self.record_learned_move(m_field.group(2), m_field.group(1), episode_number=episode_number):
                    changed = True
        return changed

    def touch(self, episode_number: Optional[int]) -> None:
        if episode_number is not None:
            self.last_written = episode_number

    @staticmethod
    def normalize_item_name(name: Optional[str]) -> str:
        if not name:
            return ""
        clean = re.sub(r"\s*\(x\d+\)\s*$", "", str(name).strip(), flags=re.IGNORECASE)
        clean = clean.strip().upper().replace("É", "E")
        if clean in ("", "NONE", "CANCEL"):
            return ""
        if clean == "POK":
            return "POKE BALL"
        return clean

    @classmethod
    def parse_item_token(cls, token: Optional[str]) -> Tuple[str, Optional[int]]:
        if not token:
            return "", None
        raw = str(token).strip()
        m_qty = re.search(r"\(\s*x\s*(\d+)\s*\)\s*$", raw, re.IGNORECASE)
        qty = int(m_qty.group(1)) if m_qty else None
        return cls.normalize_item_name(raw), qty

    @property
    def bag_complete(self) -> bool:
        return bool(self.saw_top and self.saw_bottom)

    @bag_complete.setter
    def bag_complete(self, value: bool) -> None:
        flag = bool(value)
        self.saw_top = flag
        self.saw_bottom = flag

    @property
    def has_healing_item(self) -> bool:
        return any(is_healing_item_name(it) for it in self.items)

    @property
    def has_status_heal_item(self) -> bool:
        return any(is_status_heal_item_name(it) for it in self.items)

    @property
    def healing_bag_empty(self) -> bool:
        if self.has_healing_item:
            return False
        return bool(self.bag_complete or self.healing_exhausted)

    @healing_bag_empty.setter
    def healing_bag_empty(self, value: bool) -> None:
        if value:
            self.mark_healing_bag_empty()
        else:
            self.healing_exhausted = False
            if self.bag_complete and not self.has_healing_item:
                self.saw_bottom = False

    @property
    def status_heal_bag_empty(self) -> bool:
        if self.has_status_heal_item:
            return False
        return bool(self.bag_complete or self.status_heal_exhausted)

    @status_heal_bag_empty.setter
    def status_heal_bag_empty(self, value: bool) -> None:
        if value:
            self.mark_status_heal_bag_empty()
        else:
            self.status_heal_exhausted = False
            if self.bag_complete and not self.has_status_heal_item:
                self.saw_bottom = False

    def mark_healing_bag_empty(self) -> None:
        self.healing_exhausted = True
        for it in [x for x in list(self.items) if is_healing_item_name(x)]:
            self.items.remove(it)
            self.quantities.pop(it, None)

    def mark_status_heal_bag_empty(self) -> None:
        self.status_heal_exhausted = True
        for it in [x for x in list(self.items) if is_status_heal_item_name(x)]:
            self.items.remove(it)
            self.quantities.pop(it, None)

    @staticmethod
    def _is_reversed_order(seen: List[str], candidate: List[str]) -> bool:
        overlap = [(i, seen.index(name)) for i, name in enumerate(candidate) if name in seen]
        if len(overlap) < 3:
            return False
        agreements = sum(1 for i in range(len(overlap) - 1) if overlap[i][1] < overlap[i + 1][1])
        inversions = sum(1 for i in range(len(overlap) - 1) if overlap[i][1] > overlap[i + 1][1])
        return inversions > agreements

    @classmethod
    def merge_visible_items(cls, seen: List[str], visible: List[str]) -> List[str]:
        clean_seen: List[str] = []
        for s in (seen or ()):
            norm = cls.normalize_item_name(s)
            if norm and norm not in clean_seen:
                clean_seen.append(norm)
        clean_vis: List[str] = []
        for v in (visible or ()):
            norm = cls.normalize_item_name(v)
            if norm and norm not in clean_vis:
                clean_vis.append(norm)
        if not clean_vis:
            return list(clean_seen)
        if not clean_seen:
            return list(clean_vis)
        if cls._is_reversed_order(clean_seen, clean_vis):
            clean_vis = list(reversed(clean_vis))
        overlap_pairs = [(v_idx, clean_seen.index(name)) for v_idx, name in enumerate(clean_vis) if name in clean_seen]
        if not overlap_pairs:
            out = list(clean_seen)
            for name in clean_vis:
                if name not in out:
                    out.append(name)
            return out
        first_v_idx, first_s_idx = overlap_pairs[0]
        prefix_new = [name for name in clean_vis[:first_v_idx] if name not in clean_seen]
        out = clean_seen[:first_s_idx] + prefix_new + clean_seen[first_s_idx:]
        last_pos = out.index(clean_vis[overlap_pairs[-1][0]])
        for name in clean_vis[overlap_pairs[-1][0] + 1:]:
            if name not in out:
                last_pos += 1
                out.insert(last_pos, name)
        for v_idx, name in enumerate(clean_vis):
            if name not in out:
                if v_idx > 0 and clean_vis[v_idx - 1] in out:
                    pos = out.index(clean_vis[v_idx - 1]) + 1
                    out.insert(pos, name)
                else:
                    out.append(name)
        return out

    def observe_bag_window(
        self,
        visible_items: List[str],
        *,
        quantities: Optional[Dict[str, Optional[int]]] = None,
        has_more: bool = True,
        at_top: bool = False,
        at_bottom: bool = False,
        max_items_with_cancel: int = 4,
    ) -> List[str]:
        clean_vis: List[str] = []
        for raw in (visible_items or ()):
            norm, q = self.parse_item_token(raw)
            if norm and norm not in clean_vis:
                clean_vis.append(norm)
                if q is not None:
                    self.quantities[norm] = q
        for raw_k, raw_v in (quantities or {}).items():
            norm, q = self.parse_item_token(raw_k)
            val = raw_v if raw_v is not None else q
            if norm and val is not None:
                self.quantities[norm] = val

        if not has_more:
            at_bottom = True
            if len(clean_vis) < max_items_with_cancel:
                at_top = True

        if not has_more and len(clean_vis) < max_items_with_cancel:
            self.items = list(clean_vis)
            for k in list(self.quantities):
                if k not in self.items:
                    self.quantities.pop(k, None)
            self.saw_top = True
            self.saw_bottom = True
        else:
            if clean_vis and self.items:
                if clean_vis[0] in self.items and clean_vis[-1] in self.items:
                    idx0 = self.items.index(clean_vis[0])
                    idx1 = self.items.index(clean_vis[-1])
                    if idx0 < idx1 and (idx1 - idx0) <= len(clean_vis) + 1:
                        for mid in list(self.items[idx0 + 1:idx1]):
                            if mid not in clean_vis:
                                self.items.remove(mid)
                                self.quantities.pop(mid, None)
                if at_top and clean_vis[0] in self.items:
                    idx0 = self.items.index(clean_vis[0])
                    if 0 < idx0 <= 2:
                        for top_it in list(self.items[:idx0]):
                            if top_it not in clean_vis and (
                                is_healing_item_name(top_it) or is_status_heal_item_name(top_it)
                            ):
                                self.items.remove(top_it)
                                self.quantities.pop(top_it, None)
            self.items = self.merge_visible_items(self.items, clean_vis)
            if at_top:
                self.saw_top = True
            if at_bottom:
                self.saw_bottom = True

        if self.has_healing_item:
            self.healing_exhausted = False
        elif self.bag_complete:
            self.healing_exhausted = True
        if self.has_status_heal_item:
            self.status_heal_exhausted = False
        elif self.bag_complete:
            self.status_heal_exhausted = True
        return list(self.items)

    def consume_item(
        self,
        item_name: Optional[str],
        *,
        count: int = 1,
        visible_after: Optional[List[str]] = None,
    ) -> bool:
        norm = self.normalize_item_name(item_name)
        if not norm:
            return False
        qty = self.quantities.get(norm)
        vis_norm = [self.normalize_item_name(v) for v in (visible_after or ())]
        vis_norm = [v for v in vis_norm if v]
        exhausted_stack = False
        if qty is not None and qty > count:
            self.quantities[norm] = qty - count
            if vis_norm and norm not in vis_norm:
                exhausted_stack = True
        elif qty is not None and qty <= count:
            exhausted_stack = True
        elif vis_norm and norm not in vis_norm:
            exhausted_stack = True
        elif not is_healing_item_name(norm) and not is_status_heal_item_name(norm) and "BALL" not in norm:
            if qty is None:
                exhausted_stack = True
        if exhausted_stack:
            self.quantities.pop(norm, None)
            if norm in self.items:
                self.items.remove(norm)
            if is_healing_item_name(norm) and not self.has_healing_item:
                self.healing_exhausted = True
            if is_status_heal_item_name(norm) and not self.has_status_heal_item:
                self.status_heal_exhausted = True
        return exhausted_stack

    def acquire_item(self, item_name: Optional[str], quantity: Optional[int] = None) -> None:
        norm, parsed_q = self.parse_item_token(item_name)
        if not norm:
            return
        add_q = quantity if quantity is not None else parsed_q
        if norm not in self.items:
            self.items.append(norm)
        if add_q is not None:
            prev = self.quantities.get(norm)
            self.quantities[norm] = (prev + add_q) if prev is not None else add_q
        if is_healing_item_name(norm):
            self.healing_exhausted = False
        if is_status_heal_item_name(norm):
            self.status_heal_exhausted = False

    def remove_item(self, item_name: Optional[str]) -> None:
        norm = self.normalize_item_name(item_name)
        if not norm:
            return
        self.quantities.pop(norm, None)
        if norm in self.items:
            self.items.remove(norm)
        if is_healing_item_name(norm) and not self.has_healing_item and self.bag_complete:
            self.healing_exhausted = True
        if is_status_heal_item_name(norm) and not self.has_status_heal_item and self.bag_complete:
            self.status_heal_exhausted = True

    LEVEL_GROW_PATTERN = re.compile(
        r"\bgrew\s+to\s+level\s+(\d{1,3})\b",
        re.IGNORECASE,
    )
    PLAYER_LEVEL_TAG_PATTERN = re.compile(
        r"\[Active Pokemon level seen in (?:battle|menu):\s*(\d{1,3})\]",
        re.IGNORECASE,
    )
    CALL_PLAYER_LEVEL_PATTERN = re.compile(
        r"(?:\b(?:my|our|active(?:\s+pokemon)?)\s+[A-Za-z0-9_.'♀♂-]+\s*\(\s*Lv\.?\s*(\d{1,3})\s*\)"
        r"|\b(?:my|our|active(?:\s+pokemon)?)\s+(?:Lv\.?|L|level)\s*(\d{1,3})\b)",
        re.IGNORECASE,
    )

    def record_active_level(
        self,
        level: Optional[int],
        episode_number: Optional[int] = None,
    ) -> None:
        if level is None:
            return
        lvl = int(level)
        if 1 <= lvl <= 100:
            if self.active_pokemon_level is None or lvl >= self.active_pokemon_level:
                self.active_pokemon_level = lvl
                if episode_number is not None:
                    self.last_written = episode_number

    def record_defeat(self, episode_number: Optional[int] = None) -> None:
        if episode_number is not None:
            self.last_defeat_episode = int(episode_number)
            self.last_written = int(episode_number)
        self.last_defeat_party_level = self.active_pokemon_level
        self.last_defeat_healing_bag_empty = bool(self.healing_bag_empty)
        self.learned_move_since_defeat = False

    def record_tried_or_incompatible_tm(
        self,
        tm_name: Optional[str],
        episode_number: Optional[int] = None,
    ) -> None:
        norm = self.normalize_item_name(tm_name)
        if not norm or not re.match(r"^(?:TM|HM)\s*\d+$", norm):
            return
        if not hasattr(self, "tried_or_incompatible_tms") or self.tried_or_incompatible_tms is None:
            self.tried_or_incompatible_tms = []
        if norm not in self.tried_or_incompatible_tms:
            self.tried_or_incompatible_tms.append(norm)
            if episode_number is not None:
                self.last_written = int(episode_number)

    def unused_bag_tms(self) -> List[str]:
        tried = set(getattr(self, "tried_or_incompatible_tms", ()) or ())
        out: List[str] = []
        for it in (self.items or ()):
            norm = self.normalize_item_name(it)
            if norm and re.match(r"^TM\s*\d+$", norm) and norm not in tried and norm not in out:
                out.append(norm)
        return out

    def needs_post_defeat_preparation(self) -> bool:
        if self.last_defeat_episode is None:
            return False
        if (
            self.active_pokemon_level is not None
            and self.last_defeat_party_level is not None
            and self.active_pokemon_level > self.last_defeat_party_level
        ):
            return False
        if self.learned_move_since_defeat:
            return False
        if getattr(self, "last_defeat_healing_bag_empty", False) and not self.healing_bag_empty:
            return False
        return True

    @classmethod
    def extract_player_level_from_text(cls, text: str) -> Optional[int]:
        if not text:
            return None
        best: Optional[int] = None
        for m in cls.PLAYER_LEVEL_TAG_PATTERN.finditer(text):
            val = int(m.group(1))
            if 1 <= val <= 100 and (best is None or val >= best):
                best = val
        for m in cls.LEVEL_GROW_PATTERN.finditer(text):
            val = int(m.group(1))
            if 1 <= val <= 100 and (best is None or val >= best):
                best = val
        for m in cls.CALL_PLAYER_LEVEL_PATTERN.finditer(text):
            raw = m.group(1) or m.group(2)
            if raw:
                val = int(raw)
                if 1 <= val <= 100 and (best is None or val >= best):
                    best = val
        return best

    def record_full_heal(self, episode_number: Optional[int] = None, *, preserve_max_hp: bool = False) -> None:
        self.fought_since_heal = 0
        self.fled_since_heal = 0
        if episode_number is not None:
            self.last_heal_episode = episode_number
            self.last_written = episode_number
        if preserve_max_hp and self.last_battle_hp is not None:
            _, max_hp = self.last_battle_hp
            self.last_battle_hp = (max_hp, max_hp)
        else:
            self.last_battle_hp = None
            self.last_battle_hp_episode = None
        self.fought_since_last_hp = 0
        self.active_pokemon_status = None
        self.poison_step_remainder = 0
        self.zero_pp_moves = []

    def record_battle_hp(
        self,
        hp: Optional[Tuple[int, int]],
        episode_number: Optional[int] = None,
        *,
        reset_fought_since_hp: bool = True,
    ) -> None:
        if hp is None:
            return
        cur_hp, max_hp = int(hp[0]), int(hp[1])
        if 0 <= cur_hp <= max_hp and max_hp > 0:
            self.last_battle_hp = (cur_hp, max_hp)
            if episode_number is not None:
                self.last_battle_hp_episode = episode_number
                self.last_written = episode_number
            self.poison_step_remainder = 0
            if reset_fought_since_hp:
                self.fought_since_last_hp = 0

    def record_hp_delta(self, delta: int, episode_number: Optional[int] = None) -> None:
        if self.last_battle_hp is None:
            return
        cur_hp, max_hp = self.last_battle_hp
        self.last_battle_hp = (max(0, min(max_hp, cur_hp + int(delta))), max_hp)
        if episode_number is not None:
            self.last_battle_hp_episode = episode_number
            self.last_written = episode_number

    def record_status(self, status: Optional[str], episode_number: Optional[int] = None) -> None:
        if status is None:
            return
        clean = str(status).strip().upper()
        if clean in ("", "HEALTHY", "NONE"):
            self.active_pokemon_status = None
            self.poison_step_remainder = 0
        else:
            self.active_pokemon_status = clean
        if episode_number is not None:
            self.last_written = episode_number

    def record_overworld_steps(self, steps_walked: int) -> None:
        if self.active_pokemon_status != "PSN" or self.last_battle_hp is None or steps_walked <= 0:
            return
        total = self.poison_step_remainder + int(steps_walked)
        ticks, self.poison_step_remainder = divmod(total, 4)
        if ticks > 0:
            cur_hp, max_hp = self.last_battle_hp
            self.last_battle_hp = (max(1, cur_hp - ticks), max_hp)

    def record_zero_pp(self, moves: Iterable[str], episode_number: Optional[int] = None) -> None:
        for item in (moves or ()):
            clean = str(item).strip()
            if not clean or clean.lower() == "none":
                continue
            m_slot = re.match(r"^Move\s+(\d)\s*(?:\(\s*([^)]+)\s*\))?$", clean, re.IGNORECASE)
            if m_slot:
                slot_num = m_slot.group(1)
                move_name = (m_slot.group(2) or "").strip().upper()
                replaced = False
                for idx, existing in enumerate(list(self.zero_pp_moves)):
                    if existing.upper() == clean.upper():
                        replaced = True
                        break
                    if move_name and existing.upper() == move_name:
                        self.zero_pp_moves[idx] = f"Move {slot_num} ({move_name})"
                        replaced = True
                        break
                    if existing.upper() == f"MOVE {slot_num}" and move_name:
                        self.zero_pp_moves[idx] = f"Move {slot_num} ({move_name})"
                        replaced = True
                        break
                if not replaced:
                    self.zero_pp_moves.append(f"Move {slot_num} ({move_name})" if move_name else f"Move {slot_num}")
            else:
                up = clean.upper()
                if not any(up == ex.upper() or f"({up})" in ex.upper() for ex in self.zero_pp_moves):
                    self.zero_pp_moves.append(up)
        if episode_number is not None:
            self.last_written = episode_number

    @classmethod
    def extract_bag_items_with_quantities_from_text(
        cls,
        text: str,
    ) -> Optional[Tuple[List[str], Dict[str, int], bool]]:
        if not text:
            return None
        last_result: Optional[Tuple[List[str], Dict[str, int], bool]] = None
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
            items: List[str] = []
            qtys: Dict[str, int] = {}
            if raw and raw.lower() != "none":
                for part in raw.split(","):
                    norm, q = cls.parse_item_token(part)
                    if norm and norm not in items:
                        items.append(norm)
                        if q is not None:
                            qtys[norm] = q
            last_result = (items, qtys, all_checked)
        return last_result

    @classmethod
    def extract_bag_items_from_text(cls, text: str) -> Optional[Tuple[List[str], bool]]:
        res = cls.extract_bag_items_with_quantities_from_text(text)
        if res is None:
            return None
        return res[0], res[2]

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

    @classmethod
    def is_healing_digest(cls, digest: str) -> bool:
        return bool(digest and cls.HEALING_DIGEST_PATTERN.search(digest))

    @classmethod
    def is_healing_item_bought_digest(cls, digest: str) -> bool:
        if not digest:
            return False
        if cls.HEALING_ITEM_BOUGHT_PATTERN.search(digest):
            return True
        return bool("here you are!" in digest.lower() and "potion" in digest.lower())

    @classmethod
    def is_healing_bag_empty_digest(cls, digest: str) -> bool:
        if not digest or cls.is_healing_digest(digest) or cls.is_healing_item_bought_digest(digest):
            return False
        bag_info = cls.extract_bag_items_from_text(digest)
        if bag_info is not None:
            bag_items, all_checked = bag_info
            if any(is_healing_item_name(it) for it in bag_items):
                return False
            if all_checked:
                return True
        if cls.HEALING_BAG_EMPTY_PATTERN.search(digest):
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
                    and not cls.HEALING_ITEM_USED_PATTERN.search(body)
                ):
                    return True
        return False

    @classmethod
    def is_real_battle_loss(cls, record: Any) -> bool:
        if getattr(record, "status", "") != "lost_battle":
            return False
        digest = record.supervisor_report.digest if getattr(record, "supervisor_report", None) else ""
        if "Text seen along the way, in order:" not in digest:
            return True
        text_sec = digest.split("Text seen along the way, in order:", 1)[1]
        if not text_sec.strip():
            return True
        return bool(is_battle_defeat_text([text_sec]) or has_player_pokemon_fainted_text([text_sec]))

    @classmethod
    def fought_and_fled_battles_in_digest(cls, digest: str) -> Tuple[int, int]:
        if not digest or "Text seen along the way, in order:" not in digest:
            return 0, 0
        text_sec = digest.split("Text seen along the way, in order:", 1)[1]
        chunks = re.split(r"\n-\s+while\s+doing\s+\"", text_sec)
        fought, fled = 0, 0
        for ch in chunks[1:]:
            obj, body = ch.split("\":", 1) if "\":" in ch else ("", ch)
            if obj.strip().lower().startswith("clear the dialogue"):
                continue
            if not cls.BATTLE_CHUNK_PATTERN.search(body):
                continue
            if "got away safely" in body.lower() and not cls.USED_MOVE_PATTERN.search(body):
                fled += 1
            else:
                fought += 1
        return fought, fled

    def apply_bag_tag_observation(
        self,
        items: List[str],
        quantities: Optional[Dict[str, int]],
        all_checked: bool,
        *,
        task_text: str = "",
        body_text: str = "",
    ) -> None:
        clean_items: List[str] = []
        for raw in (items or ()):
            norm = self.normalize_item_name(raw)
            if norm and norm not in clean_items:
                clean_items.append(norm)
        for raw_k, raw_v in (quantities or {}).items():
            norm = self.normalize_item_name(raw_k)
            if norm and raw_v is not None:
                self.quantities[norm] = int(raw_v)

        if re.search(r"\b(?:buy|purchase)\b", task_text, re.IGNORECASE) and re.search(
            r"\bsell\b", task_text, re.IGNORECASE
        ):
            for m_s in re.finditer(
                r"\bsell\s+(?:the\s+|an?\s+)?([A-Z0-9][A-Z0-9 -]{1,15}?)\b",
                task_text,
                re.IGNORECASE,
            ):
                s_name = self.normalize_item_name(m_s.group(1))
                if s_name in clean_items:
                    s_idx = clean_items.index(s_name)
                    clean_items = [
                        it for idx_it, it in enumerate(clean_items)
                        if it != s_name and (idx_it <= s_idx or it in self.items)
                    ]
                self.remove_item(s_name)

        m_presses = re.search(r"Presses:\s*([^\n]+)", body_text)
        press_tokens = [p.strip().lower() for p in m_presses.group(1).split(",")] if m_presses else []
        if (
            clean_items
            and len(clean_items) >= 5
            and is_healing_item_name(clean_items[0])
            and self.HEALING_ITEM_USED_PATTERN.search(body_text)
            and press_tokens
            and "down" not in press_tokens
            and "up" not in press_tokens
        ):
            exhausted_heal = clean_items.pop(0)
            self.remove_item(exhausted_heal)
            if not any(is_healing_item_name(it) for it in clean_items):
                self.mark_healing_bag_empty()

        if self._is_reversed_order(self.items, clean_items):
            clean_items = list(reversed(clean_items))

        if all_checked:
            if not clean_items:
                self.items = []
                self.quantities.clear()
                self.saw_top = True
                self.saw_bottom = True
                self.healing_exhausted = True
                self.status_heal_exhausted = True
                return
            missing_non_heal = [
                old for old in self.items
                if old not in clean_items
                and not is_healing_item_name(old)
                and not is_status_heal_item_name(old)
            ]
            if missing_non_heal:
                self.items = list(missing_non_heal) + [it for it in clean_items if it not in missing_non_heal]
            else:
                self.items = list(clean_items)
            for it in [x for x in list(self.items) if is_healing_item_name(x) and x not in clean_items]:
                self.remove_item(it)
            for it in [x for x in list(self.items) if is_status_heal_item_name(x) and x not in clean_items]:
                self.remove_item(it)
            self.saw_top = True
            self.saw_bottom = True
            self.healing_exhausted = not self.has_healing_item
            self.status_heal_exhausted = not self.has_status_heal_item
        elif clean_items:
            self.items = self.merge_visible_items(self.items, clean_items)
            if any(is_healing_item_name(it) for it in clean_items):
                self.healing_exhausted = False
            if any(is_status_heal_item_name(it) for it in clean_items):
                self.status_heal_exhausted = False

    def update_from_bag_notes(self, notes: str, task: Optional[str] = None) -> bool:
        info = self.extract_bag_items_with_quantities_from_text(notes)
        if info is None:
            return False
        bag_items, qtys, all_checked = info
        task_str = task or ""
        is_buy_only = bool(
            re.search(r"\b(?:buy|purchase|shop)\b", task_str, re.IGNORECASE)
            and not re.search(r"\bsell\b", task_str, re.IGNORECASE)
        ) or bool(
            re.search(r"\bThat\s+will\s+be\b", notes, re.IGNORECASE)
            and not re.search(r"\bWhat\s+would\s+you\s+like\s+to\s+sell\b", notes, re.IGNORECASE)
        )
        if is_buy_only:
            return False
        self.apply_bag_tag_observation(bag_items, qtys, all_checked, task_text=task_str, body_text=notes)
        return True

    def update_from_dialogue_notes(self, notes: Optional[str], episode_number: Optional[int] = None) -> None:
        if not notes:
            return
        self.update_learned_moves_from_text(notes, episode_number=episode_number)
        if re.search(
            r"\b(?:fighting\s+fit|fully\s+healed|restored\s+(?:your\s+POK[eé]MON\s+)?to\s+full\s+health)\b",
            notes,
            re.IGNORECASE,
        ):
            self.record_full_heal(episode_number=episode_number, preserve_max_hp=True)
        for m_found in re.finditer(
            r"\bYou\s+found\s+([A-Z0-9][A-Z0-9é -]{1,15})!|\bput\s+the\s+([A-Z0-9][A-Z0-9é -]{1,15})\s+in\b",
            notes,
        ):
            raw_found = (m_found.group(1) or m_found.group(2) or "").strip()
            if raw_found and not re.search(r"^\d+\s+EXP\b|^₽|\bAWAY\s+SAFELY\b", raw_found, re.IGNORECASE):
                self.acquire_item(raw_found)
        if re.search(r"\bfound\b.{1,30}!|\bput\s+the\b.{1,30}\bin\b|\bgot\b.{1,25}!", notes, re.IGNORECASE):
            if re.search(
                r"\b(?:potion|super\s+potion|hyper\s+potion|full\s+restore|fresh\s+water|soda\s+pop|lemonade)\b",
                notes,
                re.IGNORECASE,
            ):
                self.healing_exhausted = False
            if re.search(
                r"\b(?:antidote|full\s+heal|full\s+restore|parlyz\s+heal|awakening|burn\s+heal|ice\s+heal|berry)\b",
                notes,
                re.IGNORECASE,
            ):
                self.status_heal_exhausted = False

    def update_from_combat_report(
        self,
        report: Any,
        executor: Optional[Any] = None,
        episode_number: Optional[int] = None,
    ) -> None:
        notes = getattr(report, "notes", None) or ""
        self.update_learned_moves_from_text(notes, episode_number=episode_number)
        self.update_from_bag_notes(notes, getattr(report, "task", None))
        bag_info = self.extract_bag_items_from_text(notes)
        has_remaining_heal = bool(bag_info and any(is_healing_item_name(it) for it in bag_info[0]))
        if (
            (
                getattr(executor, "_bag_unavailable", False)
                or getattr(executor, "_healing_items_exhausted", False)
            )
            and not has_remaining_heal
        ):
            self.mark_healing_bag_empty()
        lvl = self.extract_player_level_from_text(notes)
        if lvl is not None:
            self.record_active_level(lvl, episode_number=episode_number)
        if getattr(report, "termination_reason", None) == "battle_lost":
            self.record_defeat(episode_number=episode_number)
            self.record_full_heal(episode_number=episode_number)
            return
        hp = self.extract_player_hp_from_text(notes)
        if hp is not None:
            self.record_battle_hp(hp, episode_number=episode_number)
        status = self.extract_player_status_from_text(notes)
        if status is not None:
            self.record_status(status, episode_number=episode_number)
        zero_pp = self.extract_zero_pp_moves_from_text(notes)
        if zero_pp is not None:
            self.record_zero_pp(zero_pp, episode_number=episode_number)

    def update_from_menu_report(
        self,
        report: Any,
        episode_number: Optional[int] = None,
    ) -> None:
        notes = getattr(report, "notes", None) or ""
        task_str = getattr(report, "task", None) or ""
        self.update_learned_moves_from_text(notes, episode_number=episode_number)
        if (
            "is not compatible" in notes.lower()
            or "can't learn" in notes.lower()
            or "cannot learn" in notes.lower()
            or "No move was taught in menu" in notes
        ):
            for m_tm in re.finditer(r"\b(TM\s*\d{1,2})\b", task_str + " " + notes, re.IGNORECASE):
                self.record_tried_or_incompatible_tm(m_tm.group(1), episode_number=episode_number)
        had_bag_tag = self.update_from_bag_notes(notes, task_str)
        if "No healing item was used in menu" in notes:
            self.mark_healing_bag_empty()
        if "No status-healing item was used in menu" in notes:
            self.mark_status_heal_bag_empty()
        if re.search(r"\b(?:here\s+you\s+(?:go|are)|thank\s+you)\b", notes, re.IGNORECASE):
            if re.search(r"\bpotion\b", notes + " " + task_str, re.IGNORECASE):
                self.acquire_item("POTION")
            if re.search(
                r"\b(?:antidote|full\s+heal|parlyz\s+heal|awakening|burn\s+heal|ice\s+heal)\b",
                notes + " " + task_str,
                re.IGNORECASE,
            ):
                for m_st in re.finditer(
                    r"\b(antidote|full\s+heal|parlyz\s+heal|awakening|burn\s+heal|ice\s+heal)\b",
                    notes + " " + task_str,
                    re.IGNORECASE,
                ):
                    self.acquire_item(m_st.group(1).upper())
                self.status_heal_exhausted = False
            if re.search(r"\bpok[eé]\s*ball\b", notes + " " + task_str, re.IGNORECASE):
                self.acquire_item("POKE BALL")
        if "grew" in notes and "RARE CANDY" in task_str.upper():
            self.consume_item("RARE CANDY")
        if (
            re.search(r"\b(?:HEALTH|ATTACK|DEFENSE|SPEED|SPECIAL)\s+rose\b", notes, re.IGNORECASE)
            and "HP UP" in task_str.upper()
        ):
            self.consume_item("HP UP")
        status = self.extract_player_status_from_text(notes)
        if status is not None:
            self.record_status(status, episode_number=episode_number)
        menu_hp = self.extract_player_hp_from_text(notes)
        m = self.HEALING_ITEM_USED_PATTERN.search(notes)
        if m:
            if not had_bag_tag and not self.healing_exhausted:
                self.healing_exhausted = False
            if menu_hp is not None:
                self.record_battle_hp(menu_hp, episode_number=episode_number, reset_fought_since_hp=False)
            elif self.last_battle_hp is not None:
                self.record_hp_delta(int(m.group(1)), episode_number=episode_number)
        elif self.is_healing_digest(notes):
            if menu_hp is not None:
                self.record_battle_hp((menu_hp[1], menu_hp[1]), episode_number=episode_number)
            self.record_full_heal(episode_number=episode_number, preserve_max_hp=True)
        elif menu_hp is not None:
            self.record_battle_hp(menu_hp, episode_number=episode_number, reset_fought_since_hp=False)

    def reconstruct_from_episode(
        self,
        record: Any,
        strategist_calls: Optional[List[Any]] = None,
    ) -> None:
        digest = record.supervisor_report.digest if getattr(record, "supervisor_report", None) is not None else ""
        if not self.has_status_heal_item:
            guidance_low = (getattr(record, "guidance", None) or "").lower()
            if "bag has no antidote" in guidance_low or any(
                "bag has NO Antidote or status-curing item" in getattr(c, "prompt", "")
                for c in (strategist_calls or ())
            ):
                self.mark_status_heal_bag_empty()

        had_non_buy_bag_tag = False
        if "Text seen along the way, in order:" in digest:
            text_sec = digest.split("Text seen along the way, in order:", 1)[1]
            chunks = re.split(r"\n-\s+while\s+doing\s+\"", text_sec)
            for ch in chunks[1:]:
                if "\":" not in ch:
                    continue
                obj, body = ch.split("\":", 1)
                is_buy_only = bool(
                    re.search(r"\b(?:buy|purchase|shop)\b", obj, re.IGNORECASE)
                    and not re.search(r"\bsell\b", obj, re.IGNORECASE)
                ) or bool(
                    re.search(r"\bThat\s+will\s+be\b", body, re.IGNORECASE)
                    and not re.search(r"\bWhat\s+would\s+you\s+like\s+to\s+sell\b", body, re.IGNORECASE)
                )
                if not is_buy_only:
                    for m in self.BAG_ITEMS_TAG_PATTERN.finditer(body):
                        info = self.extract_bag_items_with_quantities_from_text(m.group(0))
                        if info is not None:
                            items, qtys, all_checked = info
                            self.apply_bag_tag_observation(
                                items,
                                qtys,
                                all_checked,
                                task_text=obj,
                                body_text=body,
                            )
                            had_non_buy_bag_tag = True
                if "grew" in body and "RARE CANDY" in obj.upper():
                    self.consume_item("RARE CANDY")
                if (
                    re.search(r"\b(?:HEALTH|ATTACK|DEFENSE|SPEED|SPECIAL)\s+rose\b", body, re.IGNORECASE)
                    and "HP UP" in obj.upper()
                ):
                    self.consume_item("HP UP")
        else:
            info = self.extract_bag_items_with_quantities_from_text(digest)
            if info is not None:
                items, qtys, all_checked = info
                self.apply_bag_tag_observation(items, qtys, all_checked, body_text=digest)
                had_non_buy_bag_tag = True

        for m_found in re.finditer(
            r"\bYou\s+found\s+([A-Z0-9][A-Z0-9é -]{1,15})!|\bput\s+the\s+([A-Z0-9][A-Z0-9é -]{1,15})\s+in\b",
            digest,
        ):
            raw_found = (m_found.group(1) or m_found.group(2) or "").strip()
            if raw_found and not re.search(r"^\d+\s+EXP\b|^₽|\bAWAY\s+SAFELY\b", raw_found, re.IGNORECASE):
                self.acquire_item(raw_found)

        if self.is_healing_item_bought_digest(digest):
            if not self.has_healing_item:
                self.acquire_item("POTION")
            else:
                self.healing_exhausted = False
            if re.search(
                r"\b(?:antidote|full\s+heal|parlyz\s+heal|awakening|burn\s+heal|ice\s+heal)\b",
                digest,
                re.IGNORECASE,
            ) and ("here you are!" in digest.lower() or "bought" in digest.lower()):
                self.status_heal_exhausted = False
            if re.search(r"\bpok[eé]\s*balls?\b", digest, re.IGNORECASE):
                self.acquire_item("POKE BALL")
        elif self.is_healing_bag_empty_digest(digest):
            self.mark_healing_bag_empty()

        if "No status-healing item was used in menu" in digest:
            self.mark_status_heal_bag_empty()

        self.update_learned_moves_from_text(digest, episode_number=getattr(record, "n", None))

        lvl = self.extract_player_level_from_text(digest)
        if lvl is None and getattr(record, "supervisor_report", None) is not None:
            for ex in record.supervisor_report.executor_reports:
                if ex.executor_name == "CombatExecutor":
                    for call in ex.vlm_call_log:
                        if isinstance(call, ExecutorVLMCallRecord) and call.tag in (
                            "combat_step",
                            "combat_move",
                            "battle_step",
                            "battle_move",
                        ):
                            found_l = self.extract_player_level_from_text(str(call.response))
                            if found_l is not None:
                                lvl = found_l
        if lvl is None and strategist_calls:
            for c in strategist_calls:
                if c.stage == "defeat_reflection":
                    idx = c.prompt.find("The battle, as the player fought it:")
                    end = c.prompt.find("The first image is the battle screen")
                    if idx != -1:
                        lvl = self.extract_player_level_from_text(
                            c.prompt[idx:end if end != -1 else len(c.prompt)]
                        )
        if lvl is not None:
            self.record_active_level(lvl, episode_number=getattr(record, "n", None))

        if self.is_real_battle_loss(record):
            self.record_defeat(episode_number=getattr(record, "n", None))
            self.record_full_heal(episode_number=record.n)
            return
        if self.is_healing_digest(digest):
            self.record_full_heal(episode_number=record.n)
            return

        fought, fled = self.fought_and_fled_battles_in_digest(digest)
        self.fought_since_heal += fought
        self.fled_since_heal += fled
        m_rec = self.HEALING_ITEM_USED_PATTERN.search(digest)
        if m_rec:
            if not had_non_buy_bag_tag and not self.is_healing_bag_empty_digest(digest) and not self.healing_exhausted:
                self.healing_exhausted = False
            if (
                self.active_pokemon_status == "PSN"
                and re.search(
                    r"\buse\s+(?:a\s+Potion\s+\(and\s+an?\s+Antidote|an?\s+Antidote\s+\(if\s+available\))",
                    digest,
                    re.IGNORECASE,
                )
                and not re.search(r"\b(?:cured\s+of\s+(?:its\s+)?poison|poison\s+faded)\b", digest, re.IGNORECASE)
            ):
                self.mark_status_heal_bag_empty()

        status = self.extract_player_status_from_text(digest)
        if status is not None:
            self.record_status(status, episode_number=record.n)

        zero_pp_found: List[str] = []
        if "Text seen along the way, in order:" in digest:
            text_sec = digest.split("Text seen along the way, in order:", 1)[1]
            chunks = re.split(r"\n-\s+while\s+doing\s+\"", text_sec)
            for ch in chunks[1:]:
                body = ch.split("\":", 1)[1] if "\":" in ch else ch
                chunk_z = self.extract_zero_pp_moves_from_text(body)
                if chunk_z:
                    is_flee_only = "got away safely" in body.lower() and not self.USED_MOVE_PATTERN.search(body)
                    for z_item in chunk_z:
                        has_slot = bool(re.match(r"^Move\s+\d\b", z_item.strip(), re.IGNORECASE))
                        if is_flee_only and not has_slot:
                            continue
                        if z_item not in zero_pp_found:
                            zero_pp_found.append(z_item)
        else:
            chunk_z = self.extract_zero_pp_moves_from_text(digest)
            if chunk_z:
                zero_pp_found.extend(chunk_z)
        if zero_pp_found:
            self.record_zero_pp(zero_pp_found, episode_number=record.n)

        hp = self.extract_player_hp_from_text(digest)
        if hp is None and getattr(record, "supervisor_report", None) is not None:
            for ex in record.supervisor_report.executor_reports:
                if ex.executor_name == "CombatExecutor":
                    for call in ex.vlm_call_log:
                        if isinstance(call, ExecutorVLMCallRecord) and call.tag in (
                            "combat_step",
                            "combat_move",
                        ):
                            found = self.extract_player_hp_from_text(str(call.response))
                            if found is not None:
                                hp = found
        if hp is None and strategist_calls:
            for c in strategist_calls:
                if c.stage == "defeat_reflection":
                    idx = c.prompt.find("The battle, as the player fought it:")
                    end = c.prompt.find("The first image is the battle screen")
                    if idx != -1:
                        hp = self.extract_player_hp_from_text(
                            c.prompt[idx:end if end != -1 else len(c.prompt)]
                        )
        if hp is not None:
            self.record_battle_hp(hp, episode_number=record.n, reset_fought_since_hp=True)
        else:
            if m_rec and self.last_battle_hp is not None and fought == 0:
                self.record_hp_delta(int(m_rec.group(1)), episode_number=record.n)
            self.fought_since_last_hp += fought
        self.touch(getattr(record, "n", None))

    def diff(self, old: "PartyInventoryState") -> Dict[str, Any]:
        return {
            "items": (old.items, self.items),
            "quantities": (old.quantities, self.quantities),
            "bag_complete": (old.bag_complete, self.bag_complete),
            "healing_bag_empty": (old.healing_bag_empty, self.healing_bag_empty),
            "status_heal_bag_empty": (old.status_heal_bag_empty, self.status_heal_bag_empty),
            "last_battle_hp": (old.last_battle_hp, self.last_battle_hp),
            "active_pokemon_status": (old.active_pokemon_status, self.active_pokemon_status),
            "zero_pp_moves": (old.zero_pp_moves, self.zero_pp_moves),
            "learned_moves": (getattr(old, "learned_moves", []), self.learned_moves),
        }
