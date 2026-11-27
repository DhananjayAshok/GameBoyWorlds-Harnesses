from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from .routes import _snap, compress_moves, path_in_segment, plan_route, walked_squares
from .text import (LEGEND, _mapped_paths_from_start, _obstacle_detour_info, ascii_window, blocked_moves_lines,
                   door_lines, elsewhere_lines, frontier_targets, project_route_next_leg, route_lines, sector)

if TYPE_CHECKING:
    from .model import Cell, GlobalMap

# Episodes without a new cell, or failed in a row, before the strategist is asked whether it is stuck.
STALE_EPISODES = 3
FAILURE_EPISODES = 3
# Episodes exploration stays in charge once the strategist judges itself stuck.
EXPLORE_EPISODES = 3


def explore_elsewhere(gmap: GlobalMap) -> List[Dict[str, Any]]:
    groups: Dict[Any, Dict[str, Any]] = {}
    unreachable = []
    for si, seg in enumerate(gmap.segments):
        if not seg or si in gmap.merged_into or si == gmap.segment or not gmap.frontier(si):
            continue
        plan = plan_route(gmap, [si])
        if plan is None or not plan["hops"]:
            unreachable.append(si)
            continue
        first = plan["hops"][0]
        group = groups.setdefault((first["depart"], first["to"]), {"plan": plan, "target": si, "areas": []})
        group["areas"].append((si, plan))
        if plan["cost"] < group["plan"]["cost"]:
            group["plan"], group["target"] = plan, si
    out = sorted(groups.values(), key=lambda g: g["plan"]["cost"])
    for g in out:
        g["areas"].sort(key=lambda a: a[1]["cost"])
    if unreachable:
        out.append({"plan": None, "target": None, "areas": [(si, None) for si in unreachable]})
    return out


def explore_text(
    gmap: GlobalMap,
    local_kinds: Optional[Dict[Cell, str]] = None,
    blocked_moves: Optional[Dict[Cell, Iterable[str]]] = None,
) -> str:
    here, lines = gmap.area_label(gmap.segment), []
    lines += blocked_moves_lines(gmap, blocked_moves)
    edge = len(gmap.frontier())
    doors = gmap.reachable_door_list()
    unentered = sum(door["leads_to"] is None for door in doors)
    if edge:
        if unentered == 0 and len(doors) <= 1 and edge <= 16:
            lines.append(
                f"- {here}, where you are: {edge} unexplored edge square{'s' if edge != 1 else ''} "
                f"(no unentered doors/stairs in this room — finish revealing these remaining {edge} edge "
                f"square{'s' if edge != 1 else ''} before leaving through the exit door/stairs so this branch "
                f"is marked as a fully explored dead-end branch with 0 unexplored edges)."
            )
        else:
            lines.append(f"- {here}, where you are: {edge} unexplored edge square{'s' if edge != 1 else ''}.")
        targets, mass = frontier_targets(gmap, local_kinds, blocked_moves=blocked_moves)
        if targets:
            for t in targets:
                leg_note = f" [next on-screen leg: {t['next_leg']}]" if t.get("next_leg") else ""
                route_note = f" (mapped route, {t['path_steps']} moves: {t['path_moves']}{leg_note})" if t.get("path_moves") else ""
                lines.append(f"  - nearest unexplored edge {t['sector']} at ({t['dx']}, {t['dy']}), {t['steps']} steps{route_note}")
            lines.append("  - most unexplored edge lies: " + ", ".join(f"{s} ({n} squares)" for s, n in mass.most_common(3)) + ".")
    if unentered:
        lines.append(f"- {here}, where you are: {unentered} door(s) or stairs not entered yet.")
    for door in doors:
        leads_to = door["leads_to"]
        if leads_to is None or gmap.resolve(leads_to, (0, 0))[0] == gmap.segment:
            continue
        branch = gmap.branch_exploration(gmap.segment, leads_to)
        if branch["exhausted"]:
            chain = " -> ".join(gmap.area_label(si) for si in branch["segments"])
            lines.append(
                f"- Door/stairs in {here} leading to {chain}: dead-end branch, fully explored "
                f"(0 unexplored edges, 0 unentered doors)."
            )
    groups = explore_elsewhere(gmap)
    if groups:
        lines.append("Other areas with unexplored edges, grouped by the way out of here:")
        lines += elsewhere_lines(gmap, groups)
    return "\n".join(lines) or "(none)"


def exploring(gmap: GlobalMap, episode: Optional[int]) -> bool:
    return episode is not None and gmap.exploring_until is not None and episode <= gmap.exploring_until


def keep_exploring(gmap: GlobalMap, until: int) -> None:
    gmap.exploring_until = until


def stuck_signs(gmap: GlobalMap, recent_episodes: Iterable[int], failure_streak: int) -> Optional[str]:
    stale = gmap.episodes_without_new_cells(list(recent_episodes))
    signs = []
    if stale >= STALE_EPISODES:
        signs.append(f"the last {stale} episodes found nothing new for your map")
    if failure_streak >= FAILURE_EPISODES:
        signs.append(f"the last {failure_streak} episodes did not complete their task")
    return " and ".join(signs) if signs else None


def prompt_block(gmap: GlobalMap, recent_episodes: Iterable[int],
                 local_kinds: Optional[Dict[Cell, str]] = None,
                 explore: bool = False, episode: Optional[int] = None,
                 failed_tasks: Iterable[str] = (),
                 blocked_moves: Optional[Dict[Cell, Iterable[str]]] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
    """The map section of the dispatch prompt and, when exploring, the target that overrides the task.
    Does not change the map: the caller applies override["until"] with `keep_exploring`."""
    recent = list(recent_episodes)
    stale = gmap.episodes_without_new_cells(recent)
    targets, mass = frontier_targets(gmap, local_kinds, blocked_moves=blocked_moves)
    here = gmap.info(gmap.segment)
    lines = [f"You are in the area you named {gmap.place_label()}: {here.description}"] if here.name else []
    if here.explored is not None and not targets:
        lines.append("You have fully explored this area: no edge of its map is left unexplored.")
    lines += [
        "MAP OF EVERYTHING YOU HAVE SEEN IN THIS AREA (north is up, one character per step):",
        ascii_window(gmap, blocked_moves=blocked_moves),
        "Legend: " + LEGEND,
        f"Squares mapped here: {len(gmap.current)} (in {gmap.n_areas} separate area(s) overall). "
        f"New squares found in the last episode: "
        f"{gmap.new_cells_by_episode.get(max(recent), 0) if recent else 0}; "
        f"episodes in a row with nothing new: {stale}.",
    ]
    lines += blocked_moves_lines(gmap, blocked_moves)
    lines += door_lines(gmap, local_kinds, blocked_moves=blocked_moves)
    lines += route_lines(gmap)
    if targets:
        lines.append("The nearest edge of the map you have NOT explored yet in each direction "
                     "(x = steps to the right, y = steps up, measured from you):")
        for t in targets:
            where = "on screen, reachable" if t["on_screen"] else "off screen"
            leg_note = f" [next on-screen leg: {t['next_leg']}]" if t.get("next_leg") else ""
            if t.get("path_moves"):
                route_note = f"{where}; mapped route ({t['path_steps']} moves): {t['path_moves']}{leg_note}"
            elif t.get("path_steps") is None:
                route_note = f"{where}; no direct mapped path yet"
            else:
                route_note = where
            lines.append(f"- ({t['dx']}, {t['dy']}): {t['steps']} steps {t['sector']} ({route_note})")
        lines.append("Most unexplored edge lies: " +
                     ", ".join(f"{s} ({n} squares)" for s, n in mass.most_common(3)) + ".")
        doors_here = gmap.reachable_door_list()
        unentered_here = sum(d["leads_to"] is None for d in doors_here)
        edge_here = len(gmap.frontier())
        nearest_dist = targets[0].get("path_steps") if targets[0].get("path_steps") is not None else targets[0].get("steps")
        if unentered_here == 0 and len(doors_here) <= 1 and 1 <= edge_here <= 16 and nearest_dist is not None and nearest_dist <= 15:
            lines.append(
                f"Important: Only {edge_here} reachable unexplored edge square{'s' if edge_here != 1 else ''} remain in this room "
                f"(nearest at ({targets[0]['dx']}, {targets[0]['dy']}), {targets[0]['steps']} steps {targets[0]['sector']}) "
                f"and there are no other unentered doors or stairs here. Before leaving this room through the exit door/stairs, "
                f"finish revealing those remaining unexplored edge squares (or reveal them first and then take the exit door/stairs) "
                f"so this room reaches 0 unexplored edge squares and is permanently marked as a fully explored dead-end branch."
            )
    else:
        lines.append("Every reachable edge of this area has been explored; the way on is a door, "
                     "stairs or a path you have not tried.")
    px, py = gmap.pos
    walked_here = walked_squares(gmap, gmap.segment)
    reached_here = _mapped_paths_from_start(gmap, blocked_moves=blocked_moves, local_kinds=local_kinds)
    unentered_door_routes: List[Dict[str, Any]] = []
    for door in gmap.reachable_door_list():
        if door.get("leads_to") is not None:
            continue
        goal_cell = min(door["cells"], key=lambda g: abs(g[0] - px) + abs(g[1] - py))
        dx, dy = goal_cell[0] - px, goal_cell[1] - py
        snapped = _snap(gmap, gmap.segment, goal_cell, walked_here)
        found = (
            path_in_segment(gmap, gmap.segment, (px, py), snapped, walked_here, blocked_moves=blocked_moves)
            if snapped is not None
            else None
        )
        p_moves = compress_moves(found[1]) if (found is not None and found[1]) else None
        p_steps = len(found[1]) if found is not None else None
        detour = (
            _obstacle_detour_info(gmap, goal_cell, local_kinds, reached=reached_here, blocked_moves=blocked_moves)
            if p_moves is None
            else None
        )
        unentered_door_routes.append({
            "dx": dx,
            "dy": dy,
            "steps": abs(dx) + abs(dy),
            "sector": sector(dx, dy) if (dx or dy) else "here",
            "path_steps": p_steps,
            "path_moves": p_moves,
            "next_leg": project_route_next_leg(gmap, found[1], local_kinds=local_kinds, goal_rel=(dx, dy)) if (found is not None and found[1]) else None,
            "detour": detour,
        })
    unentered_door_routes.sort(
        key=lambda d: (
            d["path_steps"] is None,
            d["path_steps"] if d["path_steps"] is not None else (
                (d["detour"].get("frontier_steps") or d["steps"]) + (d["detour"].get("frontier_door_dist") or 0)
                if d["detour"] is not None else d["steps"]
            ),
            d["steps"],
        )
    )
    last_new_cells = gmap.new_cells_by_episode.get(max(recent), 0) if recent else 0
    if not explore and exploring(gmap, episode):
        if last_new_cells > 0 or any(d["path_moves"] is not None for d in unentered_door_routes):
            gmap.exploring_until = None
    override = None
    triggered = explore and episode is not None
    # Exploration stays in charge for a few episodes: one is not enough to rule a direction out.
    active = exploring(gmap, episode)
    elsewhere = explore_elsewhere(gmap) if (triggered or active) and not targets and not unentered_door_routes else []
    if (triggered or active) and (targets or unentered_door_routes or elsewhere):
        until = gmap.exploring_until if active else episode + EXPLORE_EPISODES - 1
        left = until - episode + 1
        stuck = (f"You judged that you are stuck, so what you have been trying is not working, and you are exploring "
                 f"({left} episode(s) of exploring left).")
        direct_door = next((d for d in unentered_door_routes if d["path_moves"] is not None), None)
        detour_door = next(
            (d for d in unentered_door_routes if d["detour"] is not None and d["detour"].get("frontier_rel") is not None),
            None,
        )
        if direct_door is not None:
            override = {
                "dx": direct_door["dx"],
                "dy": direct_door["dy"],
                "steps": direct_door["steps"],
                "sector": direct_door["sector"],
                "kind": "edge",
                "area": gmap.segment,
                "until": until,
            }
            lines.append(
                f"{stuck} This episode's task MUST be to walk to the unentered door or stairs in "
                f"{gmap.area_label(gmap.segment)} at ({direct_door['dx']}, {direct_door['dy']}), "
                f"{direct_door['steps']} steps {direct_door['sector']}, and enter it."
            )
        elif detour_door is not None:
            f_rel = detour_door["detour"]["frontier_rel"]
            f_steps = abs(f_rel[0]) + abs(f_rel[1])
            f_sec = sector(f_rel[0], f_rel[1]) if (f_rel[0] or f_rel[1]) else "here"
            matched = next((t for t in targets if (t["dx"], t["dy"]) == f_rel), None)
            if matched is None:
                matched = {
                    "dx": f_rel[0],
                    "dy": f_rel[1],
                    "steps": f_steps,
                    "sector": f_sec,
                }
            override = dict(matched, kind="edge", area=gmap.segment, until=until)
            lines.append(
                f"{stuck} This episode's task MUST be to walk to the unexplored edge of {gmap.area_label(gmap.segment)} "
                f"at ({override['dx']}, {override['dy']}), {override['steps']} steps {override['sector']} "
                f"(approaching the unentered door/stairs at ({detour_door['dx']}, {detour_door['dy']})), "
                f"and continue through to enter that door/stairs."
            )
        elif targets:
            heading = mass.most_common(1)[0][0]
            override = next(t for t in targets if t["sector"] == heading)   # every direction with mass has one
            override = dict(override, kind="edge", area=gmap.segment, until=until)
            lines.append(
                f"{stuck} This episode's task MUST be to walk to the unexplored edge of {gmap.area_label(gmap.segment)} "
                f"at ({override['dx']}, {override['dy']}), {override['steps']} steps {override['sector']}, "
                f"and keep going into the area beyond it.")
        else:
            override = {"kind": "elsewhere", "areas": [si for g in elsewhere for si, _ in g["areas"]],
                        "until": until}
            lines.append(
                f"{stuck} Nothing is left to explore in {gmap.area_label(gmap.segment)}. This episode's task MUST "
                f"be to head for one of the areas below that still has unexplored edges, and explore them. "
                f"Areas that share the way out of here are listed together:")
            lines += elsewhere_lines(gmap, elsewhere)
        failed = [t for t in dict.fromkeys(failed_tasks) if t][:3]
        if failed:
            lines.append("Do NOT set any of these recent tasks again — each of them failed: " +
                         "; ".join(f'"{t}"' for t in failed) + ".")
    return "\n".join(lines), override
