PRIMITIVES = """1. MoveTo(description): walk to an object, entity, entrance, staircase or exit mat that is visible on the screen, e.g. MoveTo(wooden signpost at (2, -4)) or MoveTo(exit mat at (-4, -4)). Moving to a doorway, staircase or boundary exit mat automatically steps through it to leave or enter. Only works in free roam.
2. MoveOff(direction description): walk to a walkable cell on the edge of the screen and step off it in that direction, e.g. MoveOff(up, through the gap in the ledge) or MoveOff(left). Only works in free roam when that edge of the screen has walkable cells listed; never use MoveOff towards a wall or boundary where that edge is listed as "none" (if already standing on a boundary exit mat at (0, 0), use Free to press the direction that steps off the mat).
3. Interact(target): interact with an object or entity adjacent to the player or across a 1-tile counter/desk, e.g. to talk to someone or read a sign. If the target is within a couple of steps of an adjacent or counter-front tile, Interact automatically steps into position, turns to face it, and interacts. Name the target, e.g. Interact(wooden signpost) or Interact(nurse at (1, 2)). Only works in free roam.
4. Battle(optional instructions): fight or advance the battle that is on screen (including battle intro or defeat text on the battle screen), optionally with instructions, e.g. Battle() or Battle(use Thundershock and do not switch Pokemon). Only works in battle.
5. Menu(specific instruction): use the menu system to do something specific, e.g. Menu(use a Potion on Pikachu). Only works in the main menu opened with Start from free roam, or in a menu or choice box that is already open (in menu).
6. Dialogue(): press through the dialogue that is on screen until it ends. Only works in dialogue.
7. Free(instruction; stop when: condition): press single buttons (a, b, start, arrows) towards something specific, and say what to stop at. The condition must be something visible on screen, e.g. Free(walk up and down inside the patch of tall grass; stop when: a wild Pokemon appears). Use this when none of the others fit, or during a scripted cutscene between text boxes."""

PLAYTHROUGH_GUIDANCE_BLOCK = """
Guidance for this task:
[GUIDANCE]
"""

PLAYTHROUGH_LOG_BLOCK = """
What has happened so far, in order:
[LOG]
"""

PLAYTHROUGH_AREA_MAP_BLOCK = """
[AREA_MAP]
"""

PLAYTHROUGH_PLAN_PROMPT = """You are playing [GAME] and are planning how to complete a task.

Your task: "[TASK]"
[GUIDANCE_BLOCK][LOG_BLOCK]
The player is currently [AGENT_STATE].
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES][AREA_MAP_BLOCK]

These are the only things the player can be told to do:
[PRIMITIVES]

Break the task into subgoals, in order. Write each subgoal as the state of the game to reach, in plain words, not as one of the calls above: "the player is standing in front of the nurse", never "Interact(nurse)". Each subgoal should be a small step that one of the above can plausibly achieve from the screen it starts on, and should be worth checking off on its own. Write as few as the task needs. Start from where the player is now. Because MoveTo on a doorway, staircase or boundary exit mat automatically steps through or off it to warp into the next room or outside, write leaving or entering through a visible entrance or exit mat as a single subgoal (reaching the new room or outside) rather than splitting it into standing on the mat and stepping off it. If a dialogue box or scripted cutscene is currently playing or about to continue (including when a text box at the bottom of the screen is blank, has only printed its first few characters, or is between consecutive lines), include finishing that dialogue/cutscene before any walking subgoal. If a visible entrance or object is annotated with `[no walkable path on current screen ...]` (for example, because a fence line and a building wall meet at a diagonal corner, which blocks orthogonal movement), include walking off screen around the obstacle barrier before entering the doorway. If the task or guidance suggests a cardinal detour around a local obstacle (such as walking east or west around a ledge, tree row, NPC, or building) that is blocked on the current screen or contradicts the walkable screen edges and shortest paths listed above, trust the live screen and walkable edges to route toward the task's destination rather than following a blocked cardinal direction.
If `Active Pokemon last seen battle HP:` above is marked `LOW` (at or below 50% max HP) and the healing bag has not been confirmed empty, or if "Your task" or "Guidance for this task" asks to heal your active Pokemon with a Potion from the ITEM menu while HP is low, include an initial subgoal to heal the active Pokemon with a Potion from the Start menu and close the menu before walking along a route/cave or talking to a trainer. Treat other parts of "Guidance for this task" as tactical advice and lookahead context rather than extra mandatory goals: only plan subgoals for what "Your task" asks (plus any low-HP menu heal when HP is LOW), and do not append extra landmarks, trainers, or exits mentioned only in guidance. An episode has a budget of at most 10 actions (including any battles along the way), so when "Your task" asks for open-ended progression along a route or corridor (such as "continue heading <direction>" or "follow the path as it turns toward..."), scope the movement subgoals to the concrete turns or 1-2 screen transitions needed to make clear forward progress along that path rather than requiring an entire multi-screen route to be traversed in a single subgoal. When a corridor or path continues off screen to an intersection or bend that is not yet visible on the current screen, never guess or pre-commit to a specific turn direction (such as "turns north" or "turns west") in a subgoal unless "Your task" or "Guidance for this task" explicitly specifies that turn direction or the area map above already shows the open continuation; instead, write the subgoal as reaching the end of the corridor or following the open continuation once the turn comes into view. If "Your task" contains a conditional fallback or alternative branch (such as "walk <direction> along the path, or if it ends in a dead end / is blocked, then..."), plan only the primary path up to the point where you can see on screen whether that condition holds; never include the fallback branch as sequential subgoals upfront unless the current screen already shows that the primary path is blocked or at a dead end.

Respond in exactly this format:
Reasoning: <what the screen shows and how you intend to get the task done>
Subgoals:
- <first subgoal>
- <second subgoal>
...
[STOP]"""

PLAYTHROUGH_SUBGOAL_DECISION_PROMPT = """You are playing [GAME] and deciding what the player should do next.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
Your plan:
[PLAN]

You are working on this subgoal: "[SUBGOAL]"
[LOG_BLOCK][ATTEMPTS_BLOCK]
The player is currently [AGENT_STATE].
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES][AREA_MAP_BLOCK]

You can make the player do exactly one of the following:
[PRIMITIVES]

Pick the single action that makes the most progress on the subgoal from this screen and matches the current game state ([AGENT_STATE]). Always check the bottom of the screen for a dialogue box frame—even if it is blank or has only printed its first few letters—and check whether your previous movement attempt was just interrupted by dialogue without the player moving. While a dialogue box is visible or a multi-part scripted conversation/cutscene is still playing between text boxes, choose Dialogue() or Free(...) to advance it rather than MoveTo or MoveOff. When navigating around a local obstacle (a ledge, tree row, NPC, or building), trust the current screen, the area map and visited leg coordinates above, and the walkable cells and shortest paths on each edge of the screen over any cardinal step direction suggested in the task or guidance if that direction leads into a dead end or blocked barrier. Never call MoveTo on an entrance or object annotated with `[no walkable path on current screen ...]` (two obstacle tiles `#` touching diagonally at a corner block orthogonal movement completely, with no gap to squeeze through), and never use Free(...) to press directional buttons toward a target after MoveTo returned `no_path`; instead, use MoveOff along the open walkable screen edge indicated by that door's `nearest reachable unexplored edge toward this door` and `shortest route around mapped obstacles leaves screen <direction> via (x, y)` in the `Doors and stairs in this area` list above (never follow an unrelated frontier route in `Nearest unexplored map edges in this area` that heads toward a distant sector away from the target door). Never use MoveOff towards an edge or alcove annotated as a mapped closed dead-end pocket, never repeat a blocked movement direction from a coordinate listed under blocked movement directions above (or use Free to press into that same blocked direction), and never reverse back into a dead-end coordinate boundary you already reached earlier in this episode unless the task specifically requires an item or person inside that pocket; when navigating out of a pocket or around ledges and 1x1 trees (dark circular tree sprites on grass are impassable obstacles `#`, not walkable tall grass `"`), follow the mapped route toward the reachable unexplored map edges (`?`) or the task's destination rather than a short closed alcove. When in `free roam` and `Active Pokemon last seen battle HP:` above is marked `LOW` (or the recent log shows a battle just reduced your active Pokemon's HP to 50% or less of max HP) and the healing bag has not already been checked and found empty, immediately choose `Menu(open the ITEM menu and use a Potion on your active Pokemon to restore its HP, then close the menu)` before calling `MoveOff`, `MoveTo`, or `Interact` toward another trainer or deeper into a route or cave—even if the current subgoal is a movement or interaction subgoal. When the task is to search for or locate a specific building in a town (such as a Gym or other named building) and an unentered building entrance (`NOT entered yet`) is visible near the edge of the screen with an `adjacent signpost` or with `upper roof/facade above doorway not yet seen`, step closer to or read that adjacent signpost to identify the building before walking past it off screen.

Respond in exactly this format:
Reasoning: <what you see, and why this action is the best next step>
Action: <exactly one action from the list, written as a call, e.g. MoveTo(door of the Poke Mart)>
[STOP]"""

PLAYTHROUGH_ATTEMPTS_BLOCK = """
What you already tried for this subgoal, and what came of it:
[ATTEMPTS]
"""

PLAYTHROUGH_REMAINING_SUBGOALS_BLOCK = """
Later subgoals in your plan:
[REMAINING_SUBGOALS]
If this action also already achieved one or more of these later subgoals (for example, moving to an exit mat also stepped off it and exited the building, or walking down an aisle also triggered and cleared an NPC's dialogue), give the highest plan number that is now complete in `Reached subgoal:`. Never mark a later walking or navigation subgoal as reached just because a battle or dialogue ended; only mark a later movement subgoal as reached if the overworld screen shows the player actually travelled to that subgoal's destination.
"""

PLAYTHROUGH_REMAINING_SUBGOALS_LINE = """
Reached subgoal: <the highest subgoal number from the plan that is now complete, or None>"""

PLAYTHROUGH_JUDGE_PROMPT = """You are playing [GAME] and are checking what the player just did.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
Your plan:
[PLAN]

You are working on subgoal [SUBGOAL_NUMBER]: "[SUBGOAL]"
[REMAINING_SUBGOALS_BLOCK]
You asked the player to: [REQUEST]

The first image is the screen before, and the second image is the screen after. The player is always drawn in the centre of the screen, so when the player walks, the rest of the screen shifts the other way.

Before, this had been recognised on the screen, with the player at (0, 0):
[SCREEN_TILES]
[AFTER_BLOCK][AREA_MAP_BLOCK][TEXT_BLOCK]
The game reported: [EXECUTOR_REPORT] (current game state: [AGENT_STATE]).
That report can be wrong about where the player ended up, so judge from the images.

Say whether the subgoal is now finished, and write a short note for the log of this playthrough. The note is the only record that survives, so say what actually happened and anything learned about the game or the map. Note that finishing a battle or clearing an interrupting dialogue only returns the player to free roam at their current position—it does not complete a walking or navigation subgoal unless the overworld screens show that the player also reached the subgoal's destination. When judging a subgoal to heal a Pokemon from the menu (even if phrased conditionally as "heal ... with a Potion if its HP is reduced"), check the on-screen text and executor report: if the report says `[No healing item was used in menu: bag had no usable healing item]` or shows only menu cursor/back presses without a healing recovery message (`recovered by ...` / `HP was restored`), the Pokemon was NOT healed—if a later subgoal in the plan is to talk to or battle a Gym Leader or trainer while the Pokemon's HP is reduced, answer `Subgoal complete: no` and explicitly note in `Summary` that the bag has no healing items so the supervisor can reconsider instead of starting the trainer battle unhealed. When judging a subgoal to exit a dead-end pocket or walk past a ledge/tree barrier, check the area map, mapped closed-pocket edge annotations, and mapped routes to unexplored edges above: do not mark the subgoal complete while the player is still inside the pocket west/south of the bounding ledge or tree barrier.

Respond in exactly this format:
Reasoning: <compare the two screens and the text>
Subgoal complete: <yes or no>[REMAINING_SUBGOALS_LINE]
Summary: <one or two sentences for the log>
[STOP]"""

PLAYTHROUGH_TEXT_BLOCK = """
Text that appeared on screen during it:
[TEXT]
"""

PLAYTHROUGH_AFTER_BLOCK = """
After it, this is recognised on the screen, with the player at (0, 0):
[AFTER_TILES]
"""

PLAYTHROUGH_ACHIEVABLE_PROMPT = """You are playing [GAME] and are stuck.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
Your plan:
[PLAN]

You have been trying this subgoal and it is not working: "[SUBGOAL]"
[LOG_BLOCK]
What you tried for this subgoal:
[ATTEMPTS]

The player is currently [AGENT_STATE].
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES][AREA_MAP_BLOCK]

Decide whether to carry on with a new plan or to give up. Give up only if the task cannot be done from here at all, for example because it needs something the player does not have and cannot get (including when the task requires healing with a Potion before challenging a Gym Leader or boss trainer, the party is weakened from earlier battles, and opening the bag just showed there are no healing items in the bag).

If you carry on, write a fresh plan for what is left, starting from the current screen. It may drop the subgoal that failed, or reach it another way. Plan only for what remains of "Your task" (do not add extra goals from guidance, and keep open-ended directional progression scoped to 1-2 screen transitions). Check the area map and visited leg coordinates above so your new plan does not send the player back into an already-visited dead-end boundary or closed pocket, and do not guess a specific turn direction for an off-screen intersection before that turn is visible.

Respond in exactly this format:
Reasoning: <why it is not working, and what to do about it>
Decision: <replan or quit>
Reason: <if quitting, why the task cannot be done; otherwise None>
Subgoals:
- <first subgoal of the new plan, if replanning>
- <second subgoal>
...
[STOP]"""

PLAYTHROUGH_TASK_CHECK_PROMPT = """You are playing [GAME] and have worked through your plan.

Your task: "[TASK]"
[GUIDANCE_BLOCK][LOG_BLOCK]
The player is currently [AGENT_STATE].
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES][AREA_MAP_BLOCK]

Decide whether "Your task" is now complete.
- Judge completion against "Your task" itself. Treat "Guidance for this task" as tactical advice and lookahead context, not as mandatory extra requirements: if the player has done what "Your task" asks, answer yes even if further landmarks, trainers, or exits mentioned only in guidance have not been reached yet.
- If "Your task" is a multi-part or open-ended directional traversal (such as "walk <direction> past <landmark>, then turn <direction> to exit onto <route>", "walk <direction> to the fork and take the <side> path to continue <direction>", or "walk <direction> along the path past <person/landmark> toward <destination>"): check every stage of "Your task" against the current screen, the area map, and the remaining action budget (`episode action budget used`). Do NOT answer yes after a single MoveOff leg in the final turn direction if the player has only just entered the exit corridor, reached the start of the fork, or stopped in tall grass after fleeing a wild encounter while open walkable screen edges or mapped routes to unexplored edges (`?`) continue in the task's direction and 3 or more action calls remain in the episode budget; answer no so the player continues advancing along that path in this episode. Similarly, never answer yes if the player is currently stopped directly below/beside an obstacle row (trees or ledges) that blocks the task's forward direction when an adjacent open lane on the current screen continues in that direction.
- If "Your task" includes a conditional fallback ("or if it ends in a dead end / is blocked, walk back..."), check the current screen: if the primary path did not end in a dead end, the fallback is not needed; if the player did hit a dead end on screen, the fallback movement must also be completed before answering yes.

Respond in exactly this format:
Reasoning: <what was achieved, and whether it adds up to the task>
Task complete: <yes or no>
Summary: <one or two sentences describing how the playthrough went>
[STOP]"""
PLAYTHROUGH_CLARIFY_TARGET_PROMPT = """You are playing [GAME] and deciding what the player should do next.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

You asked the player to move to "[TARGET]", but that description matches more than one thing on the screen:
[CANDIDATES]

Rewrite the description so that it matches exactly one thing on the screen: the one you meant. Include its coordinates and whatever sets it apart from the others.

Respond in exactly this format:
Reasoning: <which of the candidates you meant, and why>
Target: <the new description>
[STOP]"""

PLAYTHROUGH_MENU_OBJECTIVE_PROMPT = """You are playing [GAME] and are about to use the game's menus.

Your task: "[TASK]"
[GUIDANCE_BLOCK]
The image is the current game screen. Here is what has been recognised on it:
[SCREEN_TILES]

The player is currently [AGENT_STATE].

You decided to use the menus for this: "[INSTRUCTION]"

Write one clear, specific objective for whoever operates the menus. They see only the screen and your objective, not the task, so name only what should be selected, used, bought, read, or closed in the menu system or choice box that is currently open (or about to be opened)—including completing a shop purchase or bag item use and closing that menu back to the overworld. Do not bundle unrelated overworld walking or future dialogue that will happen after the menu system closes.

Then say whether this needs the main menu, the one opened with the Start button from the overworld, which holds the Pokedex, your Pokemon, the bag and the trainer card. Answer no if it is about a menu that is already open on the screen, such as a PC, a shop, a yes/no choice or a battle menu.

Respond in exactly this format:
Reasoning: <what the task needs from the menus>
Objective: <one sentence, specific>
Start menu: <yes or no>
[STOP]"""

PLAYTHROUGH_BATTLE_TIPS_BLOCK = """
Your battle tips, learned from earlier battles; follow them where they apply:
[TIPS]
"""

PLAYTHROUGH_BATTLE_INSTRUCTIONS_PROMPT = """You are playing [GAME] and a battle has just started.

Your task: "[TASK]"
[GUIDANCE_BLOCK][BATTLE_TIPS_BLOCK]
The image is the current battle screen.

Say how this battle should be fought, for whoever gives the orders in it. Keep it short: which Pokemon to use, which kind of move to favour, whether to use a Potion from the bag or switch Pokemon, whether to try to run, or whether to just win normally. In a trainer battle (where running is impossible), always check your active Pokemon's current/max HP on the battle screen or in `Active Pokemon last seen battle HP:` above: if your active Pokemon starts the battle at or below 50% of its max HP (or if its HP drops below half during the fight and it cannot safely knock out the opponent first) and the healing bag is not known to be empty, explicitly instruct the player to open the bag and use a Potion on their active Pokemon (or switch to a healthy Pokemon) before or during attacking so they do not black out. In a wild battle (even when the task or guidance asks to fight or train on wild Pokemon for EXP), always check the wild Pokemon's species and type against your active Pokemon's type and current HP: if your active Pokemon has a severe type disadvantage against the wild Pokemon (such as a Fire-type starter facing a wild Water-type or Rock/Ground-type Pokemon that can use super-effective moves) and you have no safe switch-in, or if your active Pokemon's HP is low, instruct the player to run away immediately rather than staying in and risking a blackout (and always instruct them to run during a wild battle if the wild opponent uses a super-effective move or drops your active Pokemon's HP below half when it cannot knock out the opponent on the next hit). Answer None if there is nothing special to say and the battle should simply be won.

Respond in exactly this format:
Reasoning: <what the battle screen shows and what it means for the task>
Instructions: <one short sentence, or None>
[STOP]"""
