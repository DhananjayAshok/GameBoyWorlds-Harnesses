STRATEGIST_RECENT_BLOCK = """
The last few episodes, oldest first:
[RECENT]
"""

DISPATCH_PROMPT = """You are playing [GAME] and are the strategist: you decide what is worth doing next, and hand one short task at a time to a player who does the actual playing.

[RECENT_BLOCK]
The goal you are working towards, from the broadest goal down to it:
[GOAL]

The player is at: [CURRENT]
[PARTY_BLOCK]
Your notepad:
[NOTEPAD]
[KNOWLEDGE_BLOCK][GUIDE]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES][GLOBAL_MAP][PEOPLE]

Write one task for the player that moves them towards the goal. The player is competent but has no memory of anything before this moment, gets roughly [MAX_STEPS] actions, and can only walk, talk to people, use menus and fight battles. So the task must be something reachable from this screen in that budget: "walk into the Pokemon Center, heal your team at the counter, and exit back outside", not "beat the next gym". Whenever `Party health status` above shows that the party has fought battles without healing or has low/reduced HP (especially after returning from a wild-battle training detour or before heading onto a route, cave, or Gym with trainer battles) and a Pokemon Center entrance or sign is on screen or nearby in the current town, prioritize entering the Pokemon Center and healing at the nurse counter before walking past the Pokemon Center onto the next route, even if your notepad's next step says to walk straight to the route. Whenever you task the player with healing at a Pokemon Center (whether starting inside the Pokemon Center or outside in town within ~15 steps of its entrance) and the next objective is outside in town or on a route, always include walking back south to the exit mat to leave the Pokemon Center back outside in the same `Task:` so healing and exiting complete in one episode. When the player is already on a route or inside a cave, tower, or Gym (away from a Pokemon Center) and `Party health status` says `party HP is low/weakened` (or the active Pokemon's last battle HP is at or below 50% of max HP, or is reduced right after picking up a Potion item ball, and does not state that the bag has no usable healing items), make opening the Start menu to heal with a Potion from the ITEM menu part of `Task:` itself (not just an optional reminder in `Guidance:`); do NOT open the Start menu to use a Potion when the active Pokemon's HP is already above 50% of max HP (unless you just picked up a Potion item ball while below 75% of max HP). Never combine a conditional menu healing check ("heal with a Potion if needed") with initiating a Gym Leader or boss trainer battle in the same task when you do not know whether the bag has any Potions: make checking/healing from the menu its own task first so you can return to a Pokemon Center or Poke Mart if the bag is empty instead of challenging the boss unhealed. Walking to a visible doorway, staircase or boundary exit mat automatically steps through or off it in a single move, so describe exiting or entering a room as one step rather than instructing the player to step onto an exit mat and then press a direction again. State the task in terms of the destination, landmark, or general heading to reach rather than prescribing low-level cardinal detours around immediate on-screen obstacles (such as which side of an NPC, ledge, tree row, or building to step around), and never combine speculative "if it ends in a dead end, walk back..." fallback branches into a single task; the player has a live on-screen pathfinder and will route around local obstacles from the screen and the walkable screen edges listed above. Never claim in the task or guidance that an unexplored path or corridor (shown as `?` on your map and listed as an unexplored edge) or an unentered doorway, ladder, or staircase (listed as `NOT entered yet` under `Doors and stairs in this area`) is a dead end before the player has actually walked through or to the end of it and mapped it. Conversely, never send the player into a doorway, ladder, or staircase annotated as `(dead-end branch, fully explored — no unentered doors or unexplored exits)` under `Doors and stairs in this area`, and if your goal details or notepad assume a pre-planned door/staircase numbering or claim that an unentered door (`NOT entered yet`) is a dead end or that an already-explored dead-end branch is still the route forward, trust the live `Doors and stairs in this area` and unexplored map edges (`?`) over those unverified assumptions. Whenever the player has traversed a multi-screen route or corridor away from town (so previously visited town doors are far away, 40+ steps) and is standing at or near an unentered building doorway or entrance (`NOT entered yet` under `Doors and stairs in this area`, within ~15 steps on or near the current screen) while the active Pokemon's HP is healthy (above 50% of max HP), always prioritize entering that unentered doorway to investigate the building and talk to the person inside before ordering a long multi-screen backtrack along the route toward a previously visited town, Pokemon Center, or Gym—even if your notepad's `Immediate plan` was written to walk back to town after battling trainers on the route. When you are inside an enclosed room or chamber whose only doorway, ladder, or staircase leads back the way you came (with 0 unentered doors in the room) and a small number of reachable unexplored edge squares (`<= 16` squares within ~15 steps) still remain in the current room, never order the player to exit through the door/ladder before revealing those remaining unexplored edge squares: task the player with walking to reveal those remaining unexplored edge squares first (or revealing them and then exiting the room) so the global map marks the room and its parent branch as a fully explored dead-end branch with 0 unexplored edge squares before you backtrack out. Because the player gets [MAX_STEPS] action calls per episode and each off-screen movement leg advances roughly 5 to 6 tiles, do not write myopic single-screen tasks that stop after stepping off just one screen edge when `Doors and stairs in this area` or `The nearest edge of the map you have NOT explored yet` shows a multi-turn mapped route (up to ~30 to 40 moves) to the target door, staircase, or unexplored frontier: write `Task:` to follow the multi-leg corridor path across its turns all the way to the target door, unexplored map edge, or corridor junction (for example, "Walk north up the corridor, turn west around the upper rock wall, and head south to the unexplored opening"), and include the mapped route move sequence in `Guidance:` so the player can traverse several screens in a single episode.

Then write the guidance: everything the player needs and could not work out from the screen. Where the thing they want is, what has already been tried and failed, what to avoid, and the mapped route turns (`mapped route, N moves: ...`) when following a multi-turn route across the area map. Always take `mapped route` move sequences from the live `Doors and stairs in this area` and `The nearest edge of the map you have NOT explored yet` sections above rather than copying an older route from your notepad: if your notepad lists an older route that differs from the live mapped route or passes through a coordinate and direction listed under `Blocked movement directions empirically recorded in this area` (or reported as `move_blocked` / `Blocked moving <dir>` in recent episodes), that old route hits an impassable wall and must be bypassed using the updated live mapped route. Because the player has no memory of earlier episodes, whenever the target door or destination is separated by an obstacle barrier or has no direct mapped path yet, or when your notepad, `Blocked movement directions empirically recorded in this area`, or recent episodes record an off-screen obstacle barrier or dead end discovered in a previous episode in this area (such as which side of a building or wall was blocked), do not leave Guidance as None—tell the player which previously tried coordinate, direction, or side is blocked and which open detour route leads around the barrier toward the destination. Do not prescribe exact tile-by-tile step counts or left/right detours around on-screen obstacles unless verified by the shortest path in the walkable screen edges above, and do not repeat the task in it.

Keep `Reasoning:` concise (at most 2-4 sentences) and grounded strictly in the live screen, area map, doors list, and unexplored edges above; never write multi-paragraph coordinate-math derivations or try to recall external game map layouts or coordinates from pre-trained memory. Respond in exactly this format:
Reasoning: <in 2-4 concise sentences, where the player is on the map and what multi-leg task moves them toward the goal>
Task: <one sentence, concrete, achievable from this screen>
Guidance: <one to four sentences for the player, or None>
[STOP]"""

REVIEW_PROMPT = """You are playing [GAME] and are the strategist. One episode has just finished.

Here is the player's report of the episode:
[DIGEST]

Write one line summarising the episode for your own log.[TOWN_QUESTION]

Keep `Reasoning:` concise (at most 1-2 sentences). Respond in exactly this format:
Reasoning: <in 1-2 concise sentences, what actually happened and what it means>
Summary: <one sentence>[TOWN_FORMAT]
[STOP]"""

REVIEW_TOWN_QUESTION = """

New doors appeared on your map this episode. Say whether the player reached a town or city this episode that it had never been to before; a route, a cave or a building is not a town. Towns and cities the player has already reached: [TOWNS]."""

REVIEW_TOWN_FORMAT = """
Town: <the name of the town or city reached for the first time this episode, or None>"""

KNOWLEDGE_EXTRACT_PROMPT = """You are playing [GAME] and are the strategist. One episode has just finished, and you keep a database of what you know about the game.

Here is the player's report of the episode:
[DIGEST]

The first image is the screen when the episode started, the second is the screen when it ended.

List the facts from this episode that are worth remembering for the rest of the game: what people said, where places are and how they connect, what items and Pokemon were found or used, what worked, what failed and why, and what the story now asks of the player. Leave out the play-by-play of what the player pressed, and anything that will not matter later.

Each fact is stored on its own, away from this report, so it must make sense without it: name the people, places and things it is about rather than saying "he", "there" or "it".

Write at most [MAX_FACTS] facts, one per line. If nothing is worth remembering, write None.

Keep `Reasoning:` concise (at most 2-3 sentences). Respond in exactly this format:
Reasoning: <in 2-3 concise sentences, what this episode taught you>
Facts:
- <fact>
- <fact>
[STOP]"""

KNOWLEDGE_HEADER = """You are playing [GAME] and keep a database of what you know, organised as a tree of topics. Here is some knowledge you wish to remember:
[KNOWLEDGE]
"""

KNOWLEDGE_NAVIGATE_PROMPT = KNOWLEDGE_HEADER + """
You may already have this in your database. First, navigate to the place where it is best stored.

You are at: [PATH]
[DESCRIPTION]

The entries under it:
[CHILDREN]

Choose the entry this knowledge belongs in, or NEW if none of them fit and it needs a new entry here.

Respond in exactly this format:
Reasoning: <which entry fits and why>
Choice: <the number of an entry, or NEW>
[STOP]"""

KNOWLEDGE_LEAF_PROMPT = KNOWLEDGE_HEADER + """
The closest entry in your database is:
[PATH]
[DESCRIPTION]

Decide what to do with it:
KNOWN: the entry already says this. Nothing changes.
UPDATE: the entry is about the same thing but is missing this knowledge or gets it wrong. Rewrite its description so it includes the new knowledge and keeps everything still true.
NEW: the knowledge is about something else. It gets its own entry next to this one.

Respond in exactly this format:
Reasoning: <how the knowledge compares to the entry>
Choice: <KNOWN, UPDATE or NEW>
New Description: <the full rewritten description if UPDATE, otherwise None>
[STOP]"""

KNOWLEDGE_CREATE_PROMPT = KNOWLEDGE_HEADER + """
It will be stored under: [PATH]
[DESCRIPTION]

The entries already there:
[SIBLINGS]

Write the new entry. If it belongs directly under [LAST], write None for both parent lines. If it is the first of a group of related entries that deserves its own heading, give that heading as the parent and the entry will go under it.

No title may repeat one of the entries already there.

Respond in exactly this format:
Reasoning: <what the knowledge is and where it fits>
Parent Title: <a short heading, or None>
Parent Description: <one sentence on what the heading covers, or None>
Entry Title: <a short title>
Entry Description: <the knowledge itself, distilled to what is worth remembering>
[STOP]"""

KNOWLEDGE_RECALL_PROMPT = """You are playing [GAME] and keep a database of what you know, organised as a tree of topics. You want to recall what you know about:
[QUERY]

You are at: [PATH]
[DESCRIPTION]

The entries under it:
[CHILDREN]

Choose up to [MAX_BRANCHES] entries that are most likely to hold something relevant, or NOT FOUND if none of them could.

Respond in exactly this format:
Reasoning: <which entries could be relevant and why>
Choice: <the numbers of at most [MAX_BRANCHES] relevant entries, separated by commas, or NOT FOUND>
[STOP]"""

KNOWLEDGE_BATCH_UPDATE_PROMPT = """You are playing [GAME] and keep a database of what you know, organised as a tree of topics.

Your database right now:
[TREE]

Here is the new knowledge you wish to remember:
[FACTS]

Decide how to update your database so all of this knowledge is stored without duplicating what is already there:
- If an existing entry already covers a fact completely, leave it alone.
- Store location, building, NPC, item, and Pokemon facts under their specific `Locations`, `NPCs`, `Items`, or `Pokemon` entries rather than duplicating them into `[1] All > Active Story`. Only update `[1] All > Active Story` when the overarching active story objective itself changes, and keep `[1] All > Active Story` strictly to the current chapter (at most 3-4 concise sentences, under 100 words), pruning completed earlier-town history that is already stored under `Locations`.
- Keep every entry description concise and distilled (at most 3-5 sentences).
- If an existing entry is about the same topic or heading and should be expanded or corrected, write:
  Change: UPDATE | <entry number> | <the concise rewritten description keeping what is still true and relevant>
- If a fact needs a new entry under an existing entry, write:
  Change: ADD | <parent entry number> | <new short entry title> | <the entry description>
- Do not use `|` inside a title or description. Update or add each entry at most once. If everything is already in your database, write the single line `Change: NONE`.

Keep `Reasoning:` concise (at most 2-3 sentences). Respond in exactly this format:
Reasoning: <in 2-3 concise sentences, how each fact fits into the existing entries>
Change: <UPDATE | <number> | <new description>, or ADD | <parent number> | <title> | <description>>
(one `Change:` line per update or addition, or instead the single line `Change: NONE`)
[STOP]"""

KNOWLEDGE_BATCH_RECALL_PROMPT = """You are playing [GAME] and keep a database of what you know, organised as a tree of topics.

Your database right now:
[TREE]

You want to recall what you know about:
[QUERIES]

Choose every entry number in your database that holds information relevant to answering these questions, or NOT FOUND if none of the entries are relevant.

Respond in exactly this format:
Reasoning: <which entries hold relevant information and why>
Choice: <the numbers of the relevant entries separated by commas, or NOT FOUND>
[STOP]"""

RETRY_BLOCK = """

Your previous reply was:
[PREVIOUS]

That reply could not be used: [PROBLEM] Respond again in the required format."""

GOAL_CHILDREN_BLOCK = """
It already had these subgoals, which are all finished with:
[CHILDREN]
"""

GOAL_UNSUCCESSFUL_BLOCK = """
You have been working towards this goal for [EPISODES] episodes and have not achieved it yet. Your notepad may say what went wrong.
"""

SUBGOAL_QUERY_PROMPT = """You are playing [GAME] and are the strategist: you break long-term goals down into smaller subgoals that a player can work through one at a time.

The goal you are planning for, from the broadest goal down to it:
[GOAL]
[EXTRAS]
Your notepad:
[NOTEPAD]

Before planning, you can look things up in your database of what you know about the game. Write the questions whose answers would help you decide how this goal should be broken down.

Each question is looked up on its own, in a database that does not know where you are, what you are doing or what has just happened. So every question must make sense to someone who knows the game but knows nothing about your situation: name the places, people, items and Pokemon it is about, and never write "here", "this town", "the next gym" or "right now". Keep each question specific and about one thing.
[ROUTE_OPTION]
If the plan is already clear, or looking things up would not help, write None instead.

Keep `Reasoning:` concise (at most 2-3 sentences). Respond in exactly this format:
Reasoning: <in 2-3 concise sentences, what you would need to know to plan this goal>
Query: <one self-contained question>
Query: <one self-contained question>[ROUTE_FORMAT]
([LINE_KINDS], or instead the single line `Queries: None`)
[STOP]"""

SUBGOAL_ROUTE_OPTION = """
You can also look up routes on your map of the places you have been. The player is at: [CURRENT]. For each place you want to know how to reach, write a `Route:` line saying where you want to go: a particular place you know by name, or a kind of place, such as "somewhere to heal your Pokemon". Unlike the questions, routes are always worked out from where the player is now.
"""

SUBGOAL_ROUTE_FORMAT = """
Route: <where you want to go from here>"""

SUBGOAL_CREATE_PROMPT = """You are playing [GAME] and are the strategist: you break long-term goals down into smaller subgoals that a player can work through one at a time.

The goal you are planning for, from the broadest goal down to it:
[GOAL]
[EXTRAS]
Your notepad:
[NOTEPAD]

What you looked up in your database of what you know about the game:
[KNOWLEDGE]

The player is at: [CURRENT]

Routes you looked up on your map:
[ROUTES]

Places on your map that may lead somewhere you have not been yet:
[EXPLORE]

Routes are only true from where the player is now, and the player will move while working through the subgoals. So describe each subgoal by where to get to and what to do there, not by copying the steps of a route. Ground every subgoal description and details strictly in what has been observed on your map and in your database: never assume prior walkthrough knowledge about which numbered door, ladder, or staircase is the true exit versus a dead end or where an unseen door or staircase is located, never label a door or staircase listed as not entered yet as a dead end before it has been entered and mapped, and never route into a door or staircase listed as a dead-end branch that is already fully explored.

Decide whether this goal needs to be broken down. If it is one clear objective the player can go straight for, write None. Otherwise, list the subgoals that together achieve it, in the order they should be done. Each subgoal is something to achieve in the game, not a button to press. When creating a subgoal to heal at a Pokemon Center before continuing outside, include exiting the Pokemon Center back outside in that same healing subgoal's description or details so healing and leaving the building do not split into separate steps.

For each subgoal, give a short description, any details that would help achieve it (or None), and why it is needed for the goal. Separate the three with `|`, and do not use `|` anywhere else. No two subgoals may have the same description, and none may repeat the goal itself or one of its existing subgoals.

Keep `Reasoning:` concise (at most 2-3 sentences) and grounded strictly in the map and database above without external walkthrough coordinate speculation. Respond in exactly this format:
Reasoning: <in 2-3 concise sentences, what the goal needs and whether it must be broken down>
Subgoal: <description> | <details, or None> | <why it is needed>
Subgoal: <description> | <details, or None> | <why it is needed>
(one `Subgoal:` line per subgoal, or instead the single line `Subgoals: None`)
[STOP]"""

GOAL_CHECK_PROMPT = """You are playing [GAME] and are the strategist. One episode has just finished.

The goal to check, from the broadest goal down to it:
[GOAL]

Current location:
[CURRENT]

Your notepad:
[NOTEPAD]
[KNOWLEDGE_BLOCK]
Here is the player's report of the episode:
[DIGEST]

Has this goal been achieved, in this episode or before it? Answer Yes only if the report or your notepad clearly shows it has happened. If the goal combines multiple preparation or excursion steps (such as buying supplies at a shop AND training/catching or completing a route excursion before re-challenging a Gym Leader) and your notepad's `Immediate plan` still lists unfinished route-training, catching, or landmark steps before returning to the Gym, completing only the shopping step does NOT achieve the goal yet (`Achieved: No`). Likewise, arriving outside an unentered building doorway at the end of a route does not achieve a route-excursion or landmark goal until the player has entered the doorway and completed the interaction inside.

Keep `Reasoning:` concise (at most 2-3 sentences). Respond in exactly this format:
Reasoning: <in 2-3 concise sentences, the evidence for and against>
Achieved: <Yes or No>
[STOP]"""

GOAL_ABANDON_OPTION = """
ABANDON: the goal cannot be achieved, or is no longer worth pursuing. Drop it."""

GOAL_DECISION_PROMPT = """You are playing [GAME] and are the strategist. One episode has just finished, and this goal has not been achieved yet:
[GOAL]

Current location:
[CURRENT]

Your notepad:
[NOTEPAD]

Here is the player's report of the episode:
[DIGEST]

Places on your map that may lead somewhere you have not been yet:
[EXPLORE]

Decide what to do with the goal:
CONTINUE: the goal is still right and is on its way; it just needs longer. Nothing changes.
UPDATE: the goal is roughly right but aimed at the wrong thing or described wrongly (including when its `Details:` prescribe a specific numbered door, ladder, or corner from unverified assumptions that turned out to be a dead-end branch or contradict the remaining unentered doors and unexplored edges on your map). Rewrite it so `New Details:` is grounded in the actual remaining unentered doors and unexplored map edges without assuming unverified walkthrough door ordering.[ABANDON_OPTION]

Keep `Reasoning:` concise (at most 2-3 sentences) and grounded strictly in the episode report and map above. Respond in exactly this format:
Reasoning: <in 2-3 concise sentences, how the goal is going and what should happen to it>
Choice: <one of [CHOICES]>
New Description: <the rewritten goal if UPDATE, otherwise None>
New Details: <the rewritten details if UPDATE, or None>
Reason: <why you chose to update or abandon it, or None if CONTINUE>
[STOP]"""

NAVIGATE_SELECT_PROMPT = """You are playing [GAME] and keep a map of the places you have been. You want to go to:
[REQUEST]

You are at: [CURRENT]

The places on your map, nearest first:
[PLACES]

Choose every place that fits where you want to go. The nearest one you know a route to will be used. If none of them fit, write None.

Respond in exactly this format:
Reasoning: <which places fit and why>
Choice: <the numbers of the places that fit, separated by commas, or None>
[STOP]"""

NOTEPAD_UPDATE_PROMPT = """You are playing [GAME] and are the strategist. You keep a notepad of working thoughts about the goal you are pursuing: what has been tried, what worked, what failed, and what to try next. One episode has just finished.

[RECENT_BLOCK]
The goal you are working towards, from the broadest goal down to it:
[GOAL]

The player is at: [CURRENT]
[PARTY_BLOCK]
Your notepad:
[NOTEPAD]
[KNOWLEDGE_BLOCK]
Here is the player's report of the episode:
[DIGEST]

Your one-line summary of the episode: [SUMMARY]

Doors and stairs and unexplored edges on your map right now:
[EXPLORE]

Update the notepad with what this episode taught you about reaching the goal. Choose one:
APPEND: add new notes to the end of the notepad, keeping everything already there.
REWRITE: replace the whole notepad, when parts of it are now wrong, out of date or repeated. Keep everything that is still true and useful.

Ground your notes strictly in the observed map doors and unexplored edges above and the episode report: never invent or repeat unverified walkthrough claims about which numbered door, ladder, or staircase is a dead end before entering it, treat any door or staircase listed as `NOT entered yet` as an unverified candidate path until entered and mapped, record any door or staircase annotated as `(dead-end branch, fully explored — no unentered doors or unexplored exits)` as an already-explored dead end not to be re-entered, and whenever the episode report or `Blocked movement directions empirically recorded in this area` shows `move_blocked` / `Blocked moving <dir>` failures along a wall or corridor coordinate, record that blocked coordinate/direction as an impassable wall and never copy a route into your plan that turns through that blocked coordinate and direction (use the live mapped route above that bypasses the wall). When the episode reaches the end or terminus of a multi-screen route or corridor next to an unentered building doorway or entrance (`NOT entered yet` under `Doors and stairs in this area`, within ~15 steps) while previously visited town doors are far away (40+ steps) and the active Pokemon's HP is healthy (above 50% of max HP), make entering and completing that unentered building doorway the first step of your `Immediate plan` before planning a long backtrack across the route to a previously visited town, Pokemon Center, or Gym. If the current room has no unentered doors and only a few remaining unexplored edge squares (`<= 16` squares), make revealing those remaining unexplored edge squares part of the immediate plan before leaving the room so the branch is marked with 0 unexplored edges. Keep the notes short and useful for the episodes to come. Do not narrate the episode step by step.

Keep `Reasoning:` concise (at most 2-3 sentences) and grounded strictly in the observed map doors, unexplored edges, and episode report without external coordinate-math speculation. Respond in exactly this format:
Reasoning: <in 2-3 concise sentences, what this episode changes about your notes>
Action: <APPEND or REWRITE>
Notes:
<the notes to append, or the whole new notepad>
[STOP]"""

SEGMENT_NAME_PROMPT = """You are playing [GAME] and are the strategist. You keep a map of everything the player has seen, made of separate areas: an area is ground the player walked around in, and the player leaves it through a door, stairs, a cave entrance or anything else that makes the screen go black. [SITUATION]

The image is the whole of that area as seen so far, stitched together from the screens: the blue line is the path the player walked, [MARK_IMAGE]numbered boxes are doors and stairs. Here it is as text too, north up, one character per step[MARK_TEXT]:
[MAP]
Legend: [LEGEND]

The tasks you gave the player recently, oldest first:
[TASKS]

[STEPS_TITLE], oldest first:
[STEPS]

Other areas you have already named:
[NAMED]

[ASK] Use the name the game gives it if a sign, the town map or dialogue said so; otherwise a short descriptive name, such as "the Pokemon Center in the second town" or "the cave north of the first route". Name the whole area, not one building in it. If this area is part of a place you have already named, such as another floor of the same building or another part of the same town, make that clear in its name. The description is for finding your way back later: what kind of place it is, how it is laid out, its notable buildings, people and objects, and its exits.

Also say what kind of area it is: outdoors (a town, city or route under the open sky), building (any floor inside a building, such as a house, a Pokemon Center, a shop, a gym, a lab or a gatehouse), or cave (a cave, forest, tunnel or other enclosed place that is not a building).

Respond in exactly this format:
Reasoning: <what the map and the player's steps show about this area>
Name: [NAME_FORMAT]
Description: [DESCRIPTION_FORMAT]
Kind: <outdoors, building or cave>
[STOP]"""

MERGE_JUDGE_PROMPT = """You are playing [GAME] and are the strategist. You keep a map of everything the player has seen, made of separate areas: an area is ground the player walked around in, and the player leaves it through a door, stairs, a cave entrance or anything else that makes the screen go black. Sometimes the player comes back to a place and the map does not recognise it, so the same place ends up as two areas. Other times two different places simply look alike: the floors of one house, or the same kind of building in two towns.

Two areas on your map look alike:
Area A, "[NAME_A]": [DESCRIPTION_A]
Its doors and stairs: [DOORS_A]
Area B, "[NAME_B]": [DESCRIPTION_B]
Its doors and stairs: [DOORS_B]

The names and descriptions were written when each area was first seen and can be wrong, so trust the images over them.

The first image is area A and the second is area B, each as seen so far; numbered boxes are doors and stairs. The third image lays B over A where they line up best: squares outlined green look the same in both and squares outlined red look different. Squares tinted blue were seen only in B, and squares with neither an outline nor a tint were seen only in A: what one area shows where the other has never looked is not a difference, since that part of the other area is simply unexplored. Lined up this way they share [SHARED] squares and [MATCHES] of them look the same. People move around, so a person standing in a different place is not a difference; furniture, walls, buildings and the shape of the ground are.

Are A and B the same place seen twice, or two different places that look alike? If they are the same place they will be joined into one area named "[NAME_A]".

Respond in exactly this format:
Reasoning: <what in the images and descriptions shows whether they are one place or two>
Decision: <SAME or DIFFERENT>
[STOP]"""

STUCK_PROMPT = """You are playing [GAME] and are the strategist: each episode you set the player one task. Lately [SIGNS].

That can mean you are stuck: going round in circles, trying the same thing that keeps failing, or looking for the way on in the wrong place. But it can also be normal progress that is simply slow: a long conversation, a cutscene, choosing from a menu, a battle, or a task that needs several episodes in one place. Only being stuck calls for a change of approach.

You are in [AREA]. Your map has [AREAS] separate area(s) so far. The image is the screen now.

Your recent episodes, oldest first, with their tasks, how they ended and what happened:
[EPISODES]

Are you stuck? If you answer YES, your next few tasks will be to explore the parts of this area you have not seen yet, instead of what you have been trying.

Respond in exactly this format:
Reasoning: <what the recent episodes show: steady progress, or the same failure again and again>
Stuck: <YES or NO>
[STOP]"""

DEFEAT_REFLECTION_PROMPT = """You are playing [GAME] and are the strategist. The player has just LOST a battle: all of their Pokemon fainted and they blacked out, which sends them back to the last Pokemon Center they used with their Pokemon healed and some of their money gone. The episode ended there.

The task you had set for the episode: [TASK]
Your current goal: [GOAL]
Where the battle happened: [WHERE]

The player's report of the episode:
[DIGEST]

The battle, as the player fought it:
[BATTLE]

[FRAMES]

You keep battle tips: short rules shown to the player at the start of every battle and on every turn of it. Your battle tips now:
[TIPS]

Think about why this battle was lost: the opponent was too strong for the player's team, the player's Pokemon were already hurt when it started, the wrong moves were chosen, a Pokemon should have been switched or healed, or something else. Then decide:
1. What went wrong, in one or two sentences.
2. Whether to change your battle tips, so the next battle goes better. A good tip is short and worth following in every battle, not just against this opponent. You may keep at most [MAX_TIPS] tips.
3. Whether the current goal can simply be tried again as it is (CONTINUE), or the player must first do something else to get ready for it (PREPARE), such as training their Pokemon in nearby grass, buying Potions, healing before setting out, or catching another Pokemon. A PREPARE step is added to your goals before the current one.
4. Whether the player should go back to where the battle happened (YES), for example because a trainer or gym leader there still blocks the way, or not (NO), for example because it was a wild Pokemon on a route.

Respond in exactly this format:
Reasoning: <why the battle was lost, and what that means for what to do next>
Wrong: <what went wrong, in one or two sentences>
Change: <ADD | <tip>, or EDIT | <number> | <new tip>, or REMOVE | <number>>
(at most [MAX_CHANGES] `Change:` lines, or instead the single line `Change: NONE`; only tips marked yours can be edited or removed)
Plan: <CONTINUE, or PREPARE | <the step to add before the current goal>>
Return: <YES or NO>
[STOP]"""

DEFEAT_RETURN_LINE = "You lost a battle in episode [EPISODE] at [WHERE] You decided the player should go back there once ready."

SEGMENT_NAME_NEW = "Name the area and describe it."
SEGMENT_NAME_REVIEW = """You named this area "[NAME]" and described it as: [DESCRIPTION]
Its map has grown from about [BEFORE] to [AFTER] squares since then, so look again. Keep the name and description if they still fit, by answering UNCHANGED. Change them only if the larger map shows the area better: for example if the name was for one building or corner and the area turns out to be a whole town, or the name is wrong."""

GUIDE_UPDATE_PROMPT = """You are playing [GAME] and are the strategist. You keep a strategy guide: short rules you follow every time something particular happens in the game. Each rule is shown to you again whenever its trigger happens, so a good rule is one that will be worth following every time, in any town, not just this once.

The triggers are: [TRIGGERS].

Your strategy guide now:
[RULES]

In the episode that just finished:
[EVENTS]

Here is the player's report of the episode:
[DIGEST]

Decide whether this episode taught you something that belongs in the guide: a habit worth having every time the player enters or leaves a building, or one of your own rules that turned out wrong or could be said better. Most episodes teach nothing new, and then the answer is NONE. Never add a rule that repeats one already there.

Respond in exactly this format:
Reasoning: <what this episode shows about entering or leaving buildings>
Change: ADD | <trigger> | <the rule, one sentence>
Change: EDIT | <number of one of your rules> | <the rewritten rule>
Change: REMOVE | <number of one of your rules>
(at most [MAX_CHANGES] `Change:` lines, or instead the single line `Change: NONE`)
[STOP]"""

NOTEPAD_EXTRACT_PROMPT = """You are playing [GAME] and are the strategist. You have just finished with these goals:
[CLOSED]

This is your notepad of working thoughts from pursuing them. It is about to be cleared:
[NOTEPAD]

List the facts in the notepad that are worth remembering for the rest of the game, long after these goals: where places are and how they connect, what people said, what items and Pokemon were found or used, what worked, what failed and why. Leave out what only mattered for finishing these goals.

Each fact is stored on its own, away from this notepad, so it must make sense without it: name the people, places and things it is about rather than saying "he", "there" or "it".

Write at most [MAX_FACTS] facts, one per line. If nothing is worth remembering, write None.

Keep `Reasoning:` concise (at most 2-3 sentences). Respond in exactly this format:
Reasoning: <in 2-3 concise sentences, what in the notepad will still matter later>
Facts:
- <fact>
- <fact>
[STOP]"""
