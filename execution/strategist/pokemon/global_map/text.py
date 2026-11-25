from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from .model import BLOCKED, LEDGE, LEDGE_LEFT, OCCUPIED, UNKNOWN, WALKABLE, is_door
from .render import segment_bounds
from .routes import blocked_target_cells, plan_route, render_plan

if TYPE_CHECKING:
    from .model import Cell, GlobalMap, MapCell


LEGEND = ('@ you   : walked (visited floor)   . unvisited walkable   # blocked   = ledge (jump down only)   '
          '< ledge (jump left only)   D door/stairs   i item ball   " tall grass   ~ water   n person   '
          '% not yet recognised   ? never seen')
SECTORS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


def glyph(c: Optional[MapCell], visited: bool = False) -> str:
    if c is None:
        return "?"
    tags = set(c.tags)
    if c.kind == OCCUPIED:
        return "n"
    if is_door(c):
        return "D"
    if "item" in tags:
        return "i"
    if "water" in tags:
        return "~"
    if c.kind == WALKABLE and "tall grass" in tags:
        return '"'
    if c.kind == WALKABLE:
        return ":" if visited else "."
    return {BLOCKED: "#", LEDGE: "=", LEDGE_LEFT: "<", UNKNOWN: "%"}.get(c.kind, "?")


MAX_WINDOW_SPAN = 61


def window_bounds(gmap: GlobalMap, rx: int = 12, ry: int = 7) -> Tuple[int, int, int, int]:
    px, py = gmap.pos
    bounds = segment_bounds(gmap, gmap.segment)
    if bounds is None:
        return px - rx, px + rx, py - ry, py + ry
    bx0, bx1, by0, by1 = bounds
    bx0, bx1 = min(bx0, px - 1), max(bx1, px + 1)
    by0, by1 = min(by0, py - 1), max(by1, py + 1)
    max_w = max(2 * rx + 1, MAX_WINDOW_SPAN)
    max_h = max(2 * ry + 1, MAX_WINDOW_SPAN)
    if bx1 - bx0 + 1 <= max_w:
        x0, x1 = bx0, bx1
    else:
        half_w = max_w // 2
        x0 = max(bx0, min(px - half_w, bx1 - max_w + 1))
        x1 = x0 + max_w - 1
    if by1 - by0 + 1 <= max_h:
        y0, y1 = by0, by1
    else:
        half_h = max_h // 2
        y0 = max(by0, min(py - half_h, by1 - max_h + 1))
        y1 = y0 + max_h - 1
    return x0, x1, y0, y1


def ascii_window(
    gmap: GlobalMap,
    rx: int = 12,
    ry: int = 7,
    blocked_moves: Optional[Dict[Cell, Iterable[str]]] = None,
) -> str:
    seg, (px, py) = gmap.current, gmap.pos
    visits = gmap.visits[gmap.segment] if 0 <= gmap.segment < len(gmap.visits) else {}
    blocked_targets = blocked_target_cells(gmap, gmap.segment, blocked_moves)
    x0, x1, y0, y1 = window_bounds(gmap, rx, ry)
    rows = []
    for y in range(y1, y0 - 1, -1):
        rows.append("".join(
            "@" if (x, y) == (px, py)
            else ("#" if (x, y) in blocked_targets else glyph(seg.get((x, y)), visited=visits.get((x, y), 0) > 0))
            for x in range(x0, x1 + 1)
        ))
    return "\n".join(rows)


def sector(dx: int, dy: int) -> str:
    import math
    return SECTORS[int(round(math.atan2(dy, dx) / (math.pi / 4))) % 8]


def blocked_moves_lines(
    gmap: GlobalMap,
    blocked_moves: Optional[Dict[Cell, Iterable[str]]] = None,
) -> List[str]:
    if not gmap.current:
        return []
    from .model import DIRECTIONS

    eff_moves = (
        gmap.segment_blocked_moves(gmap.segment, blocked_moves)
        if hasattr(gmap, "segment_blocked_moves")
        else (blocked_moves or {})
    )
    if not eff_moves:
        return []
    px, py = gmap.pos
    items = []
    for b_pos in sorted(eff_moves):
        dirs = [d for d in sorted(eff_moves[b_pos]) if d in DIRECTIONS]
        if not dirs:
            continue
        seen_d: List[str] = []
        for d in dirs:
            if d not in seen_d:
                seen_d.append(d)
        dx, dy = b_pos[0] - px, b_pos[1] - py
        here = " [HERE]" if (dx, dy) == (0, 0) else f" (relative ({dx}, {dy}) from you)"
        items.append(f"- at area coordinate {b_pos}{here}: blocked moving {', '.join(seen_d)}")
    if not items:
        return []
    return [
        "Blocked movement directions empirically recorded in this area "
        "(impassable wall/obstacle — never route through these coordinates/directions):",
        *items,
    ]


def _mapped_paths_from_start(
    gmap: GlobalMap,
    blocked_moves: Optional[Dict[Cell, Iterable[str]]] = None,
    local_kinds: Optional[Dict[Cell, str]] = None,
) -> Dict[Cell, Optional[Tuple[Cell, str, str]]]:
    import heapq
    from execution.perception.pokemon.navigation import in_bounds
    from .model import DIRECTIONS
    from .routes import STEP_COSTS

    seg = gmap.current
    start = gmap.pos
    px, py = start
    eff_moves = (
        gmap.segment_blocked_moves(gmap.segment, blocked_moves)
        if hasattr(gmap, "segment_blocked_moves")
        else (blocked_moves or {})
    )
    blocked_targets = blocked_target_cells(gmap, gmap.segment, eff_moves)
    best: Dict[Cell, int] = {start: 0}
    previous: Dict[Cell, Optional[Tuple[Cell, str, str]]] = {start: None}
    heap = [(0, start)]
    while heap:
        cost, cell = heapq.heappop(heap)
        if cost > best[cell]:
            continue
        cell_blocked_dirs = set(eff_moves.get(cell, ())) if eff_moves else set()
        for direction, (dx, dy) in DIRECTIONS.items():
            if direction in cell_blocked_dirs:
                continue
            nxt, move = (cell[0] + dx, cell[1] + dy), "walk"
            if nxt in blocked_targets:
                continue
            if local_kinds is not None:
                rel = (nxt[0] - px, nxt[1] - py)
                if in_bounds(rel) and local_kinds.get(rel) == BLOCKED:
                    continue
            c = seg.get(nxt)
            if c is None:
                continue
            if (c.kind == LEDGE and direction == "down") or (c.kind == LEDGE_LEFT and direction == "left"):
                nxt, move, step = (nxt[0] + dx, nxt[1] + dy), "jump", 2
                if nxt in blocked_targets:
                    continue
                if local_kinds is not None:
                    rel_land = (nxt[0] - px, nxt[1] - py)
                    if in_bounds(rel_land) and local_kinds.get(rel_land) == BLOCKED:
                        continue
                landing = seg.get(nxt)
                if landing is None or landing.kind not in STEP_COSTS:
                    continue
            elif c.kind in STEP_COSTS:
                step = STEP_COSTS[c.kind]
            else:
                continue
            if cost + step < best.get(nxt, cost + step + 1):
                best[nxt], previous[nxt] = cost + step, (cell, direction, move)
                heapq.heappush(heap, (cost + step, nxt))
    return previous


def _reconstruct_moves(
    previous: Dict[Cell, Optional[Tuple[Cell, str, str]]],
    goal: Cell,
) -> Optional[List[Tuple[str, str]]]:
    if goal not in previous:
        return None
    moves: List[Tuple[str, str]] = []
    cell = goal
    while previous[cell] is not None:
        parent, direction, move = previous[cell]
        moves.append((direction, move))
        cell = parent
    return moves[::-1]


def frontier_targets(
    gmap: GlobalMap,
    local_kinds: Optional[Dict[Cell, str]] = None,
    limit: int = 8,
    blocked_moves: Optional[Dict[Cell, Iterable[str]]] = None,
) -> Tuple[List[Dict[str, Any]], Counter]:
    # dx, dy are relative to the player: x right, y up
    from execution.perception.pokemon.navigation import shortest_path, in_bounds
    from .routes import compress_moves

    px, py = gmap.pos
    previous = _mapped_paths_from_start(gmap, blocked_moves=blocked_moves, local_kinds=local_kinds)
    mass: Counter = Counter()
    nearest: Dict[str, Dict[str, Any]] = {}
    for g in gmap.frontier():
        dx, dy = g[0] - px, g[1] - py
        if (dx, dy) == (0, 0):
            continue
        on_screen = local_kinds is not None and in_bounds((dx, dy))
        if on_screen and shortest_path(local_kinds, (0, 0), [(dx, dy)]) is None:
            continue
        moves = _reconstruct_moves(previous, g)
        t = {
            "dx": dx,
            "dy": dy,
            "steps": abs(dx) + abs(dy),
            "sector": sector(dx, dy),
            "on_screen": on_screen,
            "path_steps": len(moves) if moves is not None else None,
            "path_moves": compress_moves(moves) if moves else None,
        }
        mass[t["sector"]] += 1
        cur = nearest.get(t["sector"])
        if cur is None:
            nearest[t["sector"]] = t
        else:
            t_key = (t["path_steps"] is None, t["path_steps"] or t["steps"], t["steps"])
            cur_key = (cur["path_steps"] is None, cur["path_steps"] or cur["steps"], cur["steps"])
            if t_key < cur_key:
                nearest[t["sector"]] = t
    out = sorted(
        nearest.values(),
        key=lambda t: (t["path_steps"] is None, t["path_steps"] or t["steps"], t["steps"]),
    )
    return out[:limit], mass


def segment_ascii(gmap: GlobalMap, segment: int, mark: Optional[Cell] = None) -> str:
    seg, bounds = gmap.segments[segment], segment_bounds(gmap, segment)
    if bounds is None:
        return "(empty)"
    visits = gmap.visits[segment] if 0 <= segment < len(gmap.visits) else {}
    x0, x1, y0, y1 = bounds
    return "\n".join(
        "".join(
            "X" if (x, y) == mark
            else glyph(seg.get((x, y)), visited=visits.get((x, y), 0) > 0)
            for x in range(x0, x1 + 1)
        )
        for y in range(y1, y0 - 1, -1)
    )


def _door_text(gmap: GlobalMap, segment: int) -> str:
    doors = gmap.reachable_door_list(segment)
    leads = [gmap.area_label(d["leads_to"]) for d in doors if d["leads_to"] is not None]
    unentered = sum(1 for d in doors if d["leads_to"] is None)
    parts = [f"one to {name}" for name in leads] + ([f"{unentered} not entered yet"] if unentered else [])
    return ", ".join(parts) or "none known"


def route_lines(gmap: GlobalMap) -> List[str]:
    lines = []
    for r in gmap.routes:
        plan = plan_route(gmap, r["targets"])
        if plan is not None and plan.get("hops"):
            first_to = plan["hops"][0]["to"]
            if gmap.branch_exploration(gmap.segment, first_to)["exhausted"]:
                continue
        lines.append(f"Route you looked up to \"{r['request']}\":\n" + render_plan(gmap, r["targets"], plan))
    return lines


MAX_DOOR_LINES = 8


def _obstacle_detour_info(
    gmap: GlobalMap,
    goal_cell: Cell,
    local_kinds: Optional[Dict[Cell, str]] = None,
    reached: Optional[Dict[Cell, Optional[Tuple[Cell, str, str]]]] = None,
    blocked_moves: Optional[Dict[Cell, Iterable[str]]] = None,
) -> Optional[Dict[str, Any]]:
    from collections import deque
    import heapq
    from execution.perception.pokemon.navigation import in_bounds, shortest_path
    from .model import DIRECTIONS
    from .routes import compress_moves

    seg = gmap.current
    px, py = gmap.pos
    start = (px, py)
    if not seg or start == goal_cell:
        return None
    eff_moves = (
        gmap.segment_blocked_moves(gmap.segment, blocked_moves)
        if hasattr(gmap, "segment_blocked_moves")
        else (blocked_moves or {})
    )
    if reached is None:
        reached = _mapped_paths_from_start(gmap, blocked_moves=eff_moves)
    blocked_targets = blocked_target_cells(gmap, gmap.segment, eff_moves)

    on_screen_reached = {start}
    if local_kinds is not None:
        queue = deque([(0, 0)])
        seen_local = {(0, 0)}
        while queue:
            lx, ly = queue.popleft()
            on_screen_reached.add((px + lx, py + ly))
            for ox, oy in DIRECTIONS.values():
                nl = (lx + ox, ly + oy)
                if in_bounds(nl) and nl not in seen_local and local_kinds.get(nl) == WALKABLE:
                    seen_local.add(nl)
                    queue.append(nl)
    effective_reached = set(reached) | on_screen_reached

    def kind_at(g: Cell) -> str:
        if g in blocked_targets and g != goal_cell:
            return BLOCKED
        if local_kinds is not None:
            lx, ly = g[0] - px, g[1] - py
            if in_bounds((lx, ly)):
                return local_kinds.get((lx, ly), BLOCKED)
        c = seg.get(g)
        return c.kind if c is not None else "unexplored"

    walkable_approaches = [
        (goal_cell[0] + ox, goal_cell[1] + oy)
        for ox, oy in DIRECTIONS.values()
        if kind_at((goal_cell[0] + ox, goal_cell[1] + oy)) == WALKABLE
    ]
    targets = set(walkable_approaches) if walkable_approaches else {goal_cell}
    xs = [g[0] for g in seg] + [start[0], goal_cell[0]]
    ys = [g[1] for g in seg] + [start[1], goal_cell[1]]
    x_min, x_max = min(xs) - 4, max(xs) + 4
    y_min, y_max = min(ys) - 4, max(ys) + 4

    best: Dict[Cell, int] = {start: 0}
    prev: Dict[Cell, Optional[Tuple[Cell, str]]] = {start: None}
    heap = [(0, start)]
    while heap:
        cost, cur = heapq.heappop(heap)
        if cur in targets:
            path: List[Tuple[str, Cell]] = []
            c = cur
            while prev[c] is not None:
                p, d = prev[c]
                path.append((d, c))
                c = p
            path.reverse()
            leave_edge: Optional[Tuple[str, Cell]] = None
            for i, (d, (gx, gy)) in enumerate(path):
                lx, ly = gx - px, gy - py
                if not in_bounds((lx, ly)):
                    prev_gx, prev_gy = start if i == 0 else path[i - 1][1]
                    leave_edge = (d, (prev_gx - px, prev_gy - py))
                    break
            exit_frontier = start
            for _, cell in path:
                if cell in effective_reached:
                    exit_frontier = cell
                else:
                    break
            frontier_rel: Optional[Cell] = None
            frontier_steps: Optional[int] = None
            frontier_moves: Optional[str] = None
            if exit_frontier != start:
                frontier_rel = (exit_frontier[0] - px, exit_frontier[1] - py)
                if exit_frontier in reached:
                    m_list = _reconstruct_moves(reached, exit_frontier)
                    if m_list:
                        frontier_steps = len(m_list)
                        frontier_moves = compress_moves(m_list)
                elif local_kinds is not None and in_bounds(frontier_rel):
                    sp = shortest_path(local_kinds, (0, 0), [frontier_rel])
                    if sp:
                        m_list = [(d, m) for d, m, _ in sp]
                        frontier_steps = len(m_list)
                        frontier_moves = compress_moves(m_list)
            frontier_door_dist = abs(exit_frontier[0] - goal_cell[0]) + abs(exit_frontier[1] - goal_cell[1])
            if leave_edge is None and frontier_rel is None:
                return None
            return {
                "leave_edge": leave_edge,
                "frontier_rel": frontier_rel,
                "frontier_door_dist": frontier_door_dist,
                "frontier_steps": frontier_steps,
                "frontier_moves": frontier_moves,
            }
        if cost > best[cur]:
            continue
        cur_blocked_dirs = set(eff_moves.get(cur, ())) if eff_moves else set()
        for d, (ox, oy) in DIRECTIONS.items():
            if d in cur_blocked_dirs:
                continue
            nxt = (cur[0] + ox, cur[1] + oy)
            if not (x_min <= nxt[0] <= x_max and y_min <= nxt[1] <= y_max):
                continue
            if walkable_approaches and nxt == goal_cell:
                continue
            k = kind_at(nxt)
            if nxt in effective_reached and k == WALKABLE:
                step_cost = 1
            elif nxt in targets or k in (WALKABLE, "unexplored", UNKNOWN):
                step_cost = 4
            else:
                continue
            if cost + step_cost < best.get(nxt, cost + step_cost + 1):
                best[nxt] = cost + step_cost
                prev[nxt] = (cur, d)
                heapq.heappush(heap, (cost + step_cost, nxt))
    return None


def _obstacle_detour_edge(
    gmap: GlobalMap,
    goal_cell: Cell,
    local_kinds: Optional[Dict[Cell, str]] = None,
) -> Optional[Tuple[str, Cell]]:
    info = _obstacle_detour_info(gmap, goal_cell, local_kinds)
    return info.get("leave_edge") if info is not None else None


def _unentered_door_cues(gmap: GlobalMap, door_cells: List[Cell]) -> str:
    seg = gmap.current
    if not seg:
        return ""
    px, py = gmap.pos
    sign_cells = set()
    unmapped_above = False
    for dc in door_cells:
        if seg.get((dc[0], dc[1] + 1)) is None:
            unmapped_above = True
        for ox in range(-3, 4):
            for oy in range(-1, 2):
                if (ox, oy) == (0, 0):
                    continue
                nb = (dc[0] + ox, dc[1] + oy)
                c = seg.get(nb)
                if c is not None and "sign" in c.tags:
                    sign_cells.add(nb)
    cues = []
    if sign_cells:
        nearest = min(sign_cells, key=lambda s: abs(s[0] - px) + abs(s[1] - py))
        cues.append(f"adjacent signpost at ({nearest[0] - px}, {nearest[1] - py})")
    if unmapped_above:
        cues.append("upper roof/facade above doorway not yet seen")
    return ("; " + "; ".join(cues)) if cues else ""


def _leads_to_description(gmap: GlobalMap, leads_to: int) -> str:
    label = gmap.area_label(leads_to)
    branch = gmap.branch_exploration(gmap.segment, leads_to)
    if branch["exhausted"]:
        chain = [gmap.area_label(si) for si in branch["segments"]]
        if len(chain) > 1:
            return (
                f"leads to {label} -> {' -> '.join(chain[1:])} "
                f"(dead-end branch, fully explored — no unentered doors or unexplored exits)"
            )
        return f"leads to {label} (dead-end branch, fully explored — no unentered doors or unexplored exits)"
    if gmap.info(leads_to).explored is not None and not gmap.frontier(leads_to):
        return f"leads to {label}, fully explored"
    return f"leads to {label}"


def door_lines(
    gmap: GlobalMap,
    local_kinds: Optional[Dict[Cell, str]] = None,
    blocked_moves: Optional[Dict[Cell, Iterable[str]]] = None,
) -> List[str]:
    from execution.perception.pokemon.navigation import approach_cells, in_bounds, shortest_path
    from .routes import _snap, compress_moves, path_in_segment, walked_squares

    px, py = gmap.pos
    walked = walked_squares(gmap, gmap.segment)
    reached = _mapped_paths_from_start(gmap, blocked_moves=blocked_moves, local_kinds=local_kinds)
    doors = []
    for door in gmap.reachable_door_list():
        leads_to = door["leads_to"]
        if leads_to is not None and gmap.resolve(leads_to, (0, 0))[0] == gmap.segment:
            continue
        goal_cell = min(door["cells"], key=lambda g: abs(g[0] - px) + abs(g[1] - py))
        dx, dy = goal_cell[0] - px, goal_cell[1] - py
        snapped = _snap(gmap, gmap.segment, goal_cell, walked)
        found = (
            path_in_segment(gmap, gmap.segment, (px, py), snapped, walked, blocked_moves=blocked_moves)
            if snapped is not None
            else None
        )
        path_moves = compress_moves(found[1]) if (found is not None and found[1]) else None
        path_steps = len(found[1]) if found is not None else None
        on_screen_blocked = False
        if local_kinds is not None and in_bounds((dx, dy)) and (dx, dy) != (0, 0):
            goals = [(dx, dy)] if local_kinds.get((dx, dy)) == WALKABLE else list(approach_cells(local_kinds, (dx, dy)))
            if not goals or shortest_path(local_kinds, (0, 0), goals) is None:
                on_screen_blocked = True
        detour_info = (
            _obstacle_detour_info(gmap, goal_cell, local_kinds, reached=reached, blocked_moves=blocked_moves)
            if (path_moves is None or on_screen_blocked)
            else None
        )
        cues = _unentered_door_cues(gmap, door["cells"]) if leads_to is None else ""
        doors.append((abs(dx) + abs(dy), dx, dy, leads_to, path_steps, path_moves, on_screen_blocked, detour_info, cues))
    if not doors:
        return []
    lines = ["Doors and stairs in this area (x = steps to the right, y = steps up, measured from you):"]
    for steps, dx, dy, leads_to, path_steps, path_moves, on_screen_blocked, detour_info, cues in sorted(doors, key=lambda d: d[:3])[:MAX_DOOR_LINES]:
        behind = (_leads_to_description(gmap, leads_to)
                  if leads_to is not None else f"NOT entered yet{cues}")
        if path_moves:
            if on_screen_blocked:
                route_suffix = f" (separated by obstacle barrier on current screen — must route off screen first; mapped route, {path_steps} moves: {path_moves})"
            else:
                route_suffix = f" (mapped route, {path_steps} moves: {path_moves})"
        elif detour_info is not None:
            prefix = "separated by obstacle barrier on current screen — " if on_screen_blocked else ""
            parts = [f"{prefix}no direct mapped path yet"]
            f_rel = detour_info.get("frontier_rel")
            f_moves = detour_info.get("frontier_moves")
            f_steps = detour_info.get("frontier_steps")
            f_dist = detour_info.get("frontier_door_dist")
            if f_rel is not None and f_moves:
                parts.append(
                    f"nearest reachable unexplored edge toward this door is {f_rel} "
                    f"({f_dist} step{'s' if f_dist != 1 else ''} from door; mapped route, {f_steps} moves: {f_moves})"
                )
            leave_edge = detour_info.get("leave_edge")
            if leave_edge is not None:
                d, edge_cell = leave_edge
                parts.append(f"shortest route around mapped obstacles leaves screen {d} via {edge_cell}")
            route_suffix = " (" + "; ".join(parts) + ")"
        elif on_screen_blocked:
            route_suffix = " (separated by obstacle barrier on current screen — no direct mapped path yet)"
        else:
            route_suffix = ""
        lines.append(f"- ({dx}, {dy}), {steps} steps {sector(dx, dy) if steps else 'here'}: {behind}{route_suffix}")
    return lines


def elsewhere_lines(gmap: GlobalMap, groups: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for i, g in enumerate(groups, 1):
        areas = []
        for si, plan in g["areas"]:
            n = len(gmap.frontier(si))
            edge = f"{n} unexplored edge square{'s' if n != 1 else ''}"
            if plan is None:
                areas.append(f"{gmap.area_label(si)} ({edge})")
                continue
            chain = " -> ".join([gmap.area_label(plan["hops"][0]["from"])] + [gmap.area_label(h["to"])
                                                                              for h in plan["hops"]])
            areas.append(f"{gmap.area_label(si)} ({edge}; {chain}, {len(plan['hops'])} door(s))")
        if g["plan"] is None:
            lines.append(f"{i}. No way there is known from here yet: " + "; ".join(areas) + ".")
            continue
        lines.append(f"{i}. " + "; ".join(areas) + ".")
        lines.append("   " + render_plan(gmap, [g["target"]], g["plan"]).splitlines()[-1])
    return lines
