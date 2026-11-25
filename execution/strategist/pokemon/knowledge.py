from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from execution.artifact import StrategistArtifact, update
from execution.strategist.pokemon import prompts as P
from execution.strategist.pokemon.asking import Parser, ask, field_values
from utils import log_error
from utils.lm_inference import clean_value, parse_key_value

KnowledgePath = Tuple[str, ...]

KNOWLEDGE_ATTEMPTS = 3
RECALL_MAX_BRANCHES = 3


@dataclass
class KnowledgeNode:
    title: str
    description: str


class KnowledgeSubtree:
    # `leaves` holds every child, not only childless ones; the name is part of the saved format.
    def __init__(self, root: KnowledgeNode, leaves: Optional[List["KnowledgeSubtree"]] = None,
                 depth: int = 0) -> None:
        self.root = root
        self.depth = depth
        self.leaves: List["KnowledgeSubtree"] = []
        for leaf in leaves or []:
            self.add(leaf)

    def add(self, child) -> "KnowledgeSubtree":
        node = child if isinstance(child, KnowledgeSubtree) else KnowledgeSubtree(child)
        node._set_depth(self.depth + 1)
        self.leaves.append(node)
        return node

    def _set_depth(self, depth: int) -> None:
        self.depth = depth
        for leaf in self.leaves:
            leaf._set_depth(depth + 1)

    def walk(self) -> Iterator["KnowledgeSubtree"]:
        yield self
        for leaf in self.leaves:
            yield from leaf.walk()

    def nodes(self) -> List[KnowledgeNode]:
        return [tree.root for tree in self.walk()]

    def paths(self, prefix: KnowledgePath = ()) -> Dict[KnowledgePath, KnowledgeNode]:
        path = prefix + (self.root.title,)
        out = {path: self.root}
        for leaf in self.leaves:
            out.update(leaf.paths(path))
        return out

    def at(self, path: KnowledgePath) -> Optional["KnowledgeSubtree"]:
        if not path or path[0] != self.root.title:
            return None
        if len(path) == 1:
            return self
        return next((found for leaf in self.leaves if (found := leaf.at(path[1:])) is not None), None)


class KnowledgeTree(StrategistArtifact):
    def __init__(self, tree: KnowledgeSubtree) -> None:
        self.tree = tree

    def _at(self, path: KnowledgePath) -> KnowledgeSubtree:
        subtree = self.tree.at(path)
        if subtree is None:
            log_error(f"No knowledge at {' > '.join(path)!r}.")
        return subtree

    @update
    def add_under(self, path: KnowledgePath, child) -> KnowledgeSubtree:
        return self._at(path).add(child)

    @update
    def set_description(self, path: KnowledgePath, description: str) -> None:
        self._at(path).root.description = _compact_entry_description(path, description)

    def diff(self, old: "KnowledgeTree") -> Dict[str, Any]:
        new_nodes = self.tree.paths()
        old_nodes = old.tree.paths()
        description_changed = {
            path: (old_nodes[path].description, node.description)
            for path, node in new_nodes.items()
            if path in old_nodes and old_nodes[path].description != node.description
        }
        return {
            "description_changed": description_changed,
            "added": [p for p in new_nodes if p not in old_nodes],
            "removed": [p for p in old_nodes if p not in new_nodes],
        }


class PokemonKnowledgeTree(KnowledgeTree):
    def __init__(self) -> None:
        tree = KnowledgeSubtree(KnowledgeNode(title="All", description="All"))
        tree.add(KnowledgeNode(title="Active Story",
                               description="What is currently happening and what the player is in the middle of."))
        tree.add(KnowledgeNode(title="Game Lore",
                               description="Background world knowledge that does not change as the game is played."))
        tree.add(KnowledgeNode(title="NPCs",
                               description="Who has been met, what they said, and what they want."))
        tree.add(KnowledgeNode(title="Locations",
                               description="Where places are, how they connect, and what is in them."))
        tree.add(KnowledgeNode(title="Items",
                               description="What has been found, what is available, and where."))
        tree.add(KnowledgeNode(title="Pokemon",
                               description="What has been caught, trained, or fought, and how it performed."))
        tree.add(KnowledgeNode(title="Game Control Meta",
                               description="How to issue tasks and guidance to the player effectively: which kinds "
                                           "of instructions worked or failed, and why, so later tasks are "
                                           "phrased and scoped better."))
        super().__init__(tree)


def _path_text(path: KnowledgePath) -> str:
    return " > ".join(path)


def _node_prompt(template: str, game: str, text: str, path: KnowledgePath, node: KnowledgeSubtree) -> str:
    children = "\n".join(f"{i}. {child.root.title}: {child.root.description}"
                         for i, child in enumerate(node.leaves))
    siblings = "\n".join(f"- {child.root.title}" for child in node.leaves)
    return (template.replace("[GAME]", game)
            .replace("[KNOWLEDGE]", text)
            .replace("[QUERY]", text)
            .replace("[PATH]", _path_text(path))
            .replace("[DESCRIPTION]", node.root.description)
            .replace("[CHILDREN]", children)
            .replace("[SIBLINGS]", siblings or "(none)")
            .replace("[LAST]", node.root.title))


def _child_index(choice: str, children: List[KnowledgeSubtree]) -> Optional[int]:
    choice = choice.strip().rstrip(".").strip()
    numbered = re.match(r"(\d+)\s*[.):\-]?\s+(.+)$", choice)
    if numbered and int(numbered.group(1)) < len(children):
        title = children[int(numbered.group(1))].root.title.strip().lower()
        if numbered.group(2).strip().lower() == title:
            choice = numbered.group(1)
    if choice.isdigit():
        return int(choice) if int(choice) < len(children) else None
    matches = [i for i, child in enumerate(children) if child.root.title.strip().lower() == choice.lower()]
    return matches[0] if len(matches) == 1 else None


def _parse_navigation(children: List[KnowledgeSubtree]) -> Parser:
    valid = f"one of the entry numbers 0 to {len(children) - 1}, or NEW"

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        choice = clean_value(parse_key_value(output, "Choice"))
        if choice is None:
            return None, f"The `Choice:` line was missing or empty. It must be {valid}."
        if choice.upper() == "NEW":
            return "NEW", None
        index = _child_index(choice, children)
        if index is None:
            return None, f"{choice!r} is not a valid choice. It must be {valid}."
        return index, None
    return parse


def _parse_recall(children: List[KnowledgeSubtree]) -> Parser:
    valid = (f"at most {RECALL_MAX_BRANCHES} entry numbers from 0 to {len(children) - 1} separated by commas, "
             "or NOT FOUND")

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        choice = clean_value(parse_key_value(output, "Choice"))
        if choice is None:
            return None, f"The `Choice:` line was missing or empty. It must be {valid}."
        if choice.upper().replace("_", " ") == "NOT FOUND":
            return [], None
        indices: List[int] = []
        for part in choice.split(","):
            index = _child_index(part, children)
            if index is None:
                return None, f"{part.strip()!r} is not a valid entry. The choice must be {valid}."
            if index not in indices:
                indices.append(index)
        if len(indices) > RECALL_MAX_BRANCHES:
            return None, f"{len(indices)} entries were chosen. The choice must be {valid}."
        return indices, None
    return parse


def _parse_leaf(output: str) -> Tuple[Any, Optional[str]]:
    choice = clean_value(parse_key_value(output, "Choice"))
    if choice is None or choice.upper() not in ("KNOWN", "UPDATE", "NEW"):
        return None, f"The `Choice:` line must be KNOWN, UPDATE or NEW, got {choice!r}."
    choice = choice.upper()
    description = clean_value(parse_key_value(output, "New Description"))
    if choice == "UPDATE" and description is None:
        return None, "UPDATE needs the full rewritten description on the `New Description:` line."
    return (choice, description), None


def _parse_create(sibling_titles: List[str]) -> Parser:
    taken = {title.strip().lower() for title in sibling_titles}

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        parent_title = clean_value(parse_key_value(output, "Parent Title"))
        parent_description = clean_value(parse_key_value(output, "Parent Description"))
        entry_title = clean_value(parse_key_value(output, "Entry Title"))
        entry_description = clean_value(parse_key_value(output, "Entry Description"))
        if entry_title is None or entry_description is None:
            return None, "Both `Entry Title:` and `Entry Description:` must be filled in."
        if (parent_title is None) != (parent_description is None):
            return None, "`Parent Title:` and `Parent Description:` must both be filled in, or both be None."
        top_title = parent_title or entry_title
        if top_title.strip().lower() in taken:
            return None, f"The title {top_title!r} is already used by an entry there. Choose a different title."
        if parent_title is not None and parent_title.strip().lower() == entry_title.strip().lower():
            return None, "The entry's title must differ from its parent's."
        entry = KnowledgeNode(title=entry_title, description=entry_description)
        if parent_title is None:
            return KnowledgeSubtree(entry), None
        return KnowledgeSubtree(KnowledgeNode(title=parent_title, description=parent_description),
                                [KnowledgeSubtree(entry)]), None
    return parse


def _create(knowledge: KnowledgeTree, knowledge_text: str, path: KnowledgePath, vlm_call: Callable[..., str],
            game: str, episode_number: int, parameters: Optional[dict]) -> Dict[str, Any]:
    at = knowledge.tree.at(path)
    prompt = _node_prompt(P.KNOWLEDGE_CREATE_PROMPT, game, knowledge_text, path, at)
    subtree = ask(vlm_call, prompt, _parse_create([child.root.title for child in at.leaves]),
                  "Knowledge create", KNOWLEDGE_ATTEMPTS, parameters)
    if subtree is None:
        return {"outcome": "failed", "stage": "create", "path": path}
    knowledge.add_under(path, subtree, episode_number=episode_number)
    added = path + tuple(node.title for node in [subtree.root] + [leaf.root for leaf in subtree.leaves])
    return {"outcome": "added", "path": added}


def remember(knowledge: KnowledgeTree, knowledge_text: str, vlm_call: Callable[..., str], game: str,
             episode_number: int, parameters: Optional[dict] = None) -> Dict[str, Any]:
    outcomes = remember_all(knowledge, [knowledge_text], vlm_call, game, episode_number, parameters)
    return outcomes[0] if outcomes else {"outcome": "known", "path": (knowledge.tree.root.title,)}


DEFAULT_CATEGORY_DESCRIPTIONS = {
    "What is currently happening and what the player is in the middle of.",
    "Background world knowledge that does not change as the game is played.",
    "Who has been met, what they said, and what they want.",
    "Where places are, how they connect, and what is in them.",
    "What has been found, what is available, and where.",
    "What has been caught, trained, or fought, and how it performed.",
    ("How to issue tasks and guidance to the player effectively: which kinds "
     "of instructions worked or failed, and why, so later tasks are "
     "phrased and scoped better."),
}


_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\b[A-Z][a-z]\.)(?<!\b[A-Z][a-z]{2}\.)(?<!\b[A-Z][a-z]{3}\.)(?<=[.!?])\s+"
)
_ACTIVE_STORY_PREFIX = "What is currently happening and what the player is in the middle of."


def compact_active_story(description: str, max_sentences: int = 4, max_words: int = 110) -> str:
    text = " ".join((description or "").split())
    if len(text.split()) <= max_words:
        return text
    has_prefix = text.startswith(_ACTIVE_STORY_PREFIX)
    body = text[len(_ACTIVE_STORY_PREFIX):].strip() if has_prefix else text
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(body) if s.strip()]
    kept: List[str] = []
    word_count = 0
    for s in reversed(sentences):
        s_words = len(s.split())
        if kept and (len(kept) >= max_sentences or word_count + s_words > max_words):
            break
        kept.append(s)
        word_count += s_words
    kept.reverse()
    result = " ".join(kept)
    if has_prefix and result:
        return f"{_ACTIVE_STORY_PREFIX} {result}"
    return result or text[:600]


def _compact_entry_description(path: KnowledgePath, description: str) -> str:
    if path == ("All", "Active Story"):
        return compact_active_story(description)
    return description


def tree_entries(knowledge: KnowledgeTree) -> List[Tuple[KnowledgePath, KnowledgeNode]]:
    return list(knowledge.tree.paths().items())


def render_knowledge_tree(knowledge: KnowledgeTree) -> str:
    entries = tree_entries(knowledge)
    return "\n".join(f"[{i}] {_path_text(path)}: {_compact_entry_description(path, node.description)}"
                      for i, (path, node) in enumerate(entries))


def has_recorded_knowledge(knowledge: KnowledgeTree) -> bool:
    for path, node in tree_entries(knowledge):
        if len(path) > 2:
            return True
        if len(path) == 2 and node.description.strip() not in DEFAULT_CATEGORY_DESCRIPTIONS:
            return True
    return False


def _parse_entry_index(raw: str, entries: List[Tuple[KnowledgePath, KnowledgeNode]]) -> Optional[int]:
    token = raw.strip().strip("[]").rstrip(".").strip()
    if token.isdigit():
        idx = int(token)
        return idx if 0 <= idx < len(entries) else None
    numbered = re.match(r"^\[?(\d+)\]?\s*[.):\-]?\s*(.*)$", token)
    if numbered:
        idx = int(numbered.group(1))
        if 0 <= idx < len(entries):
            return idx
    lowered = token.lower()
    matches = [i for i, (path, node) in enumerate(entries)
               if _path_text(path).lower() == lowered or node.title.strip().lower() == lowered]
    return matches[0] if len(matches) == 1 else None


def _parse_batch_changes(entries: List[Tuple[KnowledgePath, KnowledgeNode]],
                         knowledge: KnowledgeTree) -> Parser:
    def parse(output: str) -> Tuple[Any, Optional[str]]:
        values = field_values(output, "Change")
        if not values:
            return None, "Write at least one `Change:` line, or the single line `Change: NONE`."
        if len(values) == 1 and clean_value(values[0]) is None:
            return [], None
        changes = []
        updated_indices = set()
        added_under: Dict[KnowledgePath, set] = {}
        for value in values:
            if clean_value(value) is None:
                continue
            parts = [p.strip() for p in value.split("|")]
            action = parts[0].upper()
            if action == "UPDATE":
                if len(parts) < 3:
                    return None, f"`Change: {value}` must be `UPDATE | <entry number> | <new description>`."
                idx = _parse_entry_index(parts[1], entries)
                if idx is None or idx == 0:
                    return None, f"`{parts[1]}` in `Change: {value}` is not a valid entry number (1 to {len(entries) - 1})."
                description = clean_value(" ".join(parts[2:]))
                if description is None:
                    return None, f"`Change: {value}` has an empty description."
                if idx in updated_indices:
                    return None, f"Entry [{idx}] is updated more than once; combine updates to the same entry into one line."
                updated_indices.add(idx)
                changes.append(("UPDATE", entries[idx][0], description))
            elif action == "ADD":
                if len(parts) < 4:
                    return None, f"`Change: {value}` must be `ADD | <parent entry number> | <entry title> | <entry description>`."
                idx = _parse_entry_index(parts[1], entries)
                if idx is None:
                    return None, f"`{parts[1]}` in `Change: {value}` is not a valid parent entry number (0 to {len(entries) - 1})."
                parent_path = entries[idx][0]
                title = clean_value(parts[2])
                description = clean_value(" ".join(parts[3:]))
                if title is None or description is None:
                    return None, f"`Change: {value}` must provide both a title and a description."
                parent_subtree = knowledge.tree.at(parent_path)
                existing_titles = {child.root.title.strip().lower() for child in (parent_subtree.leaves if parent_subtree else [])}
                pending_titles = added_under.setdefault(parent_path, set())
                if title.strip().lower() in existing_titles:
                    for child in (parent_subtree.leaves if parent_subtree else []):
                        if child.root.title.strip().lower() == title.strip().lower():
                            child_path = parent_path + (child.root.title,)
                            changes.append(("UPDATE", child_path, description))
                            break
                elif title.strip().lower() in pending_titles:
                    return None, f"Title {title!r} is added twice under {_path_text(parent_path)!r}."
                else:
                    pending_titles.add(title.strip().lower())
                    changes.append(("ADD", parent_path, KnowledgeNode(title=title, description=description)))
            else:
                return None, f"Each `Change:` line must start with UPDATE, ADD, or NONE, got {parts[0]!r}."
        return changes, None
    return parse


def remember_all(knowledge: KnowledgeTree, facts: List[str], vlm_call: Callable[..., str], game: str,
                 episode_number: int, parameters: Optional[dict] = None) -> List[Dict[str, Any]]:
    cleaned = [f.strip() for f in facts if f and f.strip()]
    if not cleaned:
        return []
    story_sub = knowledge.tree.at(("All", "Active Story"))
    if story_sub is not None:
        compacted = compact_active_story(story_sub.root.description)
        if compacted != story_sub.root.description:
            story_sub.root.description = compacted
    entries = tree_entries(knowledge)
    prompt = (P.KNOWLEDGE_BATCH_UPDATE_PROMPT
              .replace("[GAME]", game)
              .replace("[TREE]", render_knowledge_tree(knowledge))
              .replace("[FACTS]", "\n".join(f"- {fact}" for fact in cleaned)))
    changes = ask(vlm_call, prompt, _parse_batch_changes(entries, knowledge), "Knowledge update",
                  KNOWLEDGE_ATTEMPTS, parameters)
    if changes is None:
        return [{"outcome": "failed", "stage": "batch_update"}]
    outcomes: List[Dict[str, Any]] = []
    for action, path, payload in changes:
        if action == "UPDATE":
            knowledge.set_description(path, payload, episode_number=episode_number)
            outcomes.append({"outcome": "updated", "path": path})
        elif action == "ADD":
            knowledge.add_under(path, KnowledgeSubtree(payload), episode_number=episode_number)
            outcomes.append({"outcome": "added", "path": path + (payload.title,)})
    return outcomes or [{"outcome": "known", "path": (knowledge.tree.root.title,)}]


def _parse_batch_recall(entries: List[Tuple[KnowledgePath, KnowledgeNode]]) -> Parser:
    valid = f"entry numbers from 1 to {len(entries) - 1} separated by commas, or NOT FOUND"

    def parse(output: str) -> Tuple[Any, Optional[str]]:
        choice = clean_value(parse_key_value(output, "Choice"))
        if choice is None:
            return [], None
        if choice.upper().replace("_", " ") in ("NOT FOUND", "NONE"):
            return [], None
        indices: List[int] = []
        for part in choice.split(","):
            part = part.strip()
            if not part:
                continue
            idx = _parse_entry_index(part, entries)
            if idx is None or idx == 0:
                return None, f"{part!r} is not a valid entry. The choice must be {valid}."
            if idx not in indices:
                indices.append(idx)
        return indices, None
    return parse


def recall_queries(knowledge: KnowledgeTree, queries: List[str], vlm_call: Callable[..., str], game: str,
                   parameters: Optional[dict] = None) -> List[Dict[str, Any]]:
    cleaned = [q.strip() for q in queries if q and q.strip()]
    if not cleaned or not has_recorded_knowledge(knowledge):
        return []
    entries = tree_entries(knowledge)
    prompt = (P.KNOWLEDGE_BATCH_RECALL_PROMPT
              .replace("[GAME]", game)
              .replace("[TREE]", render_knowledge_tree(knowledge))
              .replace("[QUERIES]", "\n".join(f"- {q}" for q in cleaned)))
    indices = ask(vlm_call, prompt, _parse_batch_recall(entries), "Knowledge recall",
                  KNOWLEDGE_ATTEMPTS, parameters)
    found: List[Dict[str, Any]] = []
    for idx in indices or []:
        path, node = entries[idx]
        found.append({"path": path, "title": node.title, "description": node.description})
    return found


def _recall_at(node: KnowledgeSubtree, path: KnowledgePath, query: str, vlm_call: Callable[..., str],
               game: str, parameters: Optional[dict]) -> List[Dict[str, Any]]:
    if not node.leaves:
        if len(path) == 1:
            return []
        return [{"path": path, "title": node.root.title, "description": node.root.description}]
    prompt = (_node_prompt(P.KNOWLEDGE_RECALL_PROMPT, game, query, path, node)
              .replace("[MAX_BRANCHES]", str(RECALL_MAX_BRANCHES)))
    indices = ask(vlm_call, prompt, _parse_recall(node.leaves), "Knowledge recall", KNOWLEDGE_ATTEMPTS,
                  parameters)
    found: List[Dict[str, Any]] = []
    for index in indices or []:
        child = node.leaves[index]
        found += _recall_at(child, path + (child.root.title,), query, vlm_call, game, parameters)
    return found


def recall(knowledge: KnowledgeTree, query: str, vlm_call: Callable[..., str], game: str,
           parameters: Optional[dict] = None) -> List[Dict[str, Any]]:
    return recall_queries(knowledge, [query], vlm_call, game, parameters)
