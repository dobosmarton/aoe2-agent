# Actions and execution

Every input the agent sends goes through `apps/agent/src/executor.py`. Catalog actions from the actor and tool calls from the LLM executor become validated `Action` models, are checked against the per-game `ActionLedger`, resolve their targets against fresh detections, and reach the game through pyautogui. A purchase stays pending until later HUD readings show that it happened.

## Action model

`apps/agent/src/models.py` defines 12 action types, joined in the `Action` union. The union is plain on purpose, so Pydantic emits `anyOf` (OpenAI strict mode rejects the `oneOf` a discriminator produces).

| Type | Fields | Handler does |
|---|---|---|
| `click` | x/y, `target_id` or `target_class`; `building_key`, `menu`, `auto_placement` | Left click; a placement click opens a pending building |
| `right_click` | x/y, `target_id` or `target_class` | Unit order with the profile's command button |
| `press` | `key`, `modifiers`, `rescan`, `selection_only` | Key or chord, optionally waits for a fresh view |
| `build` | `menu` (`q`, `w`, `v`), `building_key` | Select a builder, open the menu, place, press `h` |
| `research` | `tech` | Go to the building, verify selection, press the research key |
| `queue_villager` | none | `h`, verify Town Center, `q` |
| `train_unit` | `unit` (6 military units) | Go to the building, verify selection, press the unit key |
| `assign_idle` | `resource` | `.`, verify villager, right-click a resource in the refreshed view |
| `drag`, `scroll`, `wait`, `detect` | coordinates, clicks, `ms`, none | Raw input, delay, or wait for a fresh frame |

Numeric and length bounds are `field_validator`s instead of `Field` bounds, which keeps them out of the JSON schema and the constrained-decoding grammar small: x up to 7680, y up to 4320, `wait` up to 5000 ms, `key` 1 to 20 characters, `building_key` 1 to 2. `PressAction` refuses `escape`, `esc`, `f10` and `f3`, which open the game menu or pause. A point action needs x and y, a `target_id`, a `target_class`, or (for `click`) `auto_placement`. `validate_action` returns `None` for an invalid dict and `validate_actions` drops those silently.

## Executor tool surface

`apps/agent/src/providers/action_tools.py` lists 14 tools in `_ACTION_TOOLS`, written in Anthropic's schema shape (the OpenAI conversion is in [LLM providers](llm-providers.md)). Building, research and unit options in the descriptions are generated from the action catalog.

| Tool | Runs as |
|---|---|
| `click`, `right_click`, `drag`, `wait`, `scroll`, `detect` | The action of the same type |
| `press` | `press`, after `ExecutorProvider` refuses any key outside the navigation set |
| `build` | `build`, after `build_rejection` |
| `research`, `queue_villager`, `train_unit`, `assign_idle` | The action of the same type |
| `send_villager`, `send_all_idle` | `assign_idle` for the resource kind of `target_class`; both assign one villager, and an unknown class is refused |

`_RECOVERY_TOOLS` is the subset `build`, `research`, `queue_villager`, `train_unit`, `assign_idle`, `wait`. When a tool other than `wait` fails without leaving a ledger outcome, the provider calls `record_failure` so the failure counts toward recovery.

## Layout of executor.py

The file is about 3900 lines of module-level state and functions, in this order.

| Section | Holds |
|---|---|
| Module state | pyautogui settings (`FAILSAFE` off, `PAUSE` 0.02 s), window offset, input scale, platform, modifier map, command button, the detected-entity cache, refresh callbacks |
| Entity cache | `set_detected_entities`, `set_ground_frame`, and setters for the refresh callbacks that `game_loop` points at the frame pipe |
| Build gates and ledger | Settle and expiry constants, `Tech`, the pending dataclasses, `ActionOutcome`, `ActionLedger`, `ledger_policy_state`, `observe_hud`, `observe_age`, the settlement functions, `build_rejection`, `villager_queue_rejection`, `blocked_actions` |
| Coordinate resolution | `_previewed_placement` and the preview hover, `_resolve_coords`, `_fresh_spatial_cache`, builder villager selection |
| macOS input | Modifier mapping, blocked system chords, held clicks, `set_absolute_zoom` |
| Handlers | Placement geometry, `build_steps`, `research_steps`, `research_rejection`, and one `_handle_*` per action type, registered in `_ACTION_HANDLERS` |
| Public API | `execute_action`, `execute_actions` |

## Execution path

`execute_actions` runs a batch in order and stops at the first failure. On macOS it refuses the whole batch when the game is not the front window; on Windows it calls `ensure_game_focused` and retries once after 0.5 s. A click given only as raw x/y after a camera-moving press in the same batch is refused with `STALE_COORDS_DETAIL`.

`execute_action` validates the dict, re-reads window offset, input scale and platform, checks again that a Mac game is in front, and dispatches to the handler. Afterwards it moves the cursor to the window centre so the camera does not edge-scroll, and sleeps `config.action_delay` (0.05 s). A missing parameter or any exception returns a failed `ActionResult` with the reason.

## Checks before a spending key

The named handlers re-check feasibility against `ledger_policy_state()` immediately before the key that spends, using the same `eligible` function as the [catalog](decision-loops.md#the-feasible-action-catalog). Research, villager and unit handlers check both before and after navigating to their building.

| Handler | Refuses when |
|---|---|
| `build` (`_rejection_reason`) | Unknown menu binding; build deferred after a failed refresh or placement step; class suppressed by the circuit breaker; a house already pending; class paid and awaiting completion; a unique building confirmed or pending; house at cap 200; prerequisite not confirmed (farm needs a mill, archery range and stable a barracks); wood minus reserved wood below cost; a lumber or mining camp with no visible resource; not eligible |
| `research` (`_research_rejection_reason`) | Unknown tech; researched; paid; age advance paid; blocked for 50 s after a missed payment; pending; building not confirmed; no resource reading; unaffordable; not eligible |
| `queue_villager` (`villager_queue_rejection`) | Population unknown; population plus pending units at cap; food unknown; food minus reserved food under 50 |
| `train_unit`, `assign_idle` | Not eligible |

The handler then registers the commitment (`_note_pending_placement`, `_note_pending_research`, `_note_pending_training`). That needs a HUD reading taken after the last spending input (`resources_current`) with the cost resources read; otherwise the action fails with "no resource baseline". Navigation keys carry `rescan` and `selection_only`, which wait for a post-input HUD and selected-unit read. `_selection_rejection` then requires the selected unit to match and to have been read at the current input revision, so a go-to key that selected nothing cannot route the next key into the wrong panel.

`apps/agent/src/preflight.py` is a separate read-only qualification tool for a game profile. It clicks nothing and sends no keys; it checks window and capture size, HUD calibration and reading, the selected object, the profile's verification flags, and on macOS the permissions, keyboard layout, colour profile, menu-bar band, open menus and a cursor round trip. It exits non-zero when a blocking check fails.

```bash
uv run --package gameplay-agent python -m gameplay_agent.preflight --profile apps/agent/profiles/magyars_highland_4v4.macos.example.json
```

## Target resolution and click authorization

`_resolve_coords` turns a point action into screenshot pixels or refuses it:

1. The ledger's spatial view must be valid; a camera key or rescan press invalidates it until a fresh frame arrives.
2. `_fresh_spatial_cache` allows boxes up to 1.0 s old (`MAX_SPATIAL_AGE_SECONDS`). A static target (a building, a resource, a herdable animal, an automatic placement) may use boxes up to 2.0 s old if they were seen at the current camera revision.
3. `auto_placement` resolves at click time (see [building placement](#building-placement)).
4. A `target_id` carries the input revision, camera revision, capture id, class and centre it was bound to. Any change, a class supported only by track history, or a missing box refuses the click. A villager is clicked at 35% of its box height, on the body above any animal at its feet.
5. `target_class: villager` takes the most confident fresh villager (0.65 or higher) at the current camera revision, preferring one whose click point is clear of buildings and animals. Other classes take the first detected entity of that class.
6. Literal x/y is accepted; `(0, 0)` is refused as a placeholder.

`right_click` also refuses points in the HUD margins: 160 px at the top, 280 px at the bottom and 40 px at the sides (`UI_MARGIN_*`).

## Building placement

`placement_candidates` lists the spots for a building, best first. A drop-off camp takes ring points 150 to 270 px around its resource (lumber camp on trees, mining camp on gold or stone, mill on berries when visible). Any other building is judged by its footprint: `_footprint` scales the building's edge in tiles (2 for a house, mill or camp, 3 for a farm and most others, 4 for a market, university or siege workshop) from the detected Town Center's width, which is 4 tiles, or from the reference Town Center width when none is in view. Ring points at 280, 400, 520 and 640 px around the Town Center are kept when they are explored and the whole footprint ends above the bottom margin. They sort first by whether they lie near an offset the game refused before, then by whether the footprint overlaps the box of a building, tree or mine (units do not block, since they walk away), then by clutter. The list is the clear ring points, then the clear footprint nearest the anchor (the Town Center, or the view centre without one) on a grid over the whole play area, then the rest of the ring.

In a game, `game_loop` registers a capture hook (`set_preview_capture_fn`), and `_previewed_placement` lets the game's own placement preview pick the spot. With the building on the cursor, it rests the cursor on the selection panel (a quarter of the way across, inside the bottom margin) for a comparison frame, then hovers up to 8 candidates, capturing a frame drawn 0.15 s after each move, and clicks the first spot the preview accepts. `read_preview` (`apps/agent/src/placement_preview.py`) compares the two frames over the building's preview region and returns a `PreviewReading`: the strong-red share without and with the preview (red at least 170, green and blue at most 90) and the share of pixels where a channel moved by more than 40.

| `PreviewKind` | Buildings | Region read | Refused when red rises by | Drawn when changed share reaches |
|---|---|---|---|---|
| `SPRITE` | All but the farm | The footprint and the building's height above it | 0.01 | 0.08 |
| `OUTLINE` | Farm | The footprint alone, widened by 1.15 | 0.001 | not checked |

A spot is accepted when the preview is drawn and not refused; the thresholds in `_CALIBRATION` were measured in a live Mac game. If no hovered spot showed a preview, the click fails with "no <building> on the cursor"; if every spot was red, it fails with "no valid ground in view". Neither notes a placement, so neither counts toward the class suppression. Without a capture hook (scenario runner, tests) `default_build_placement` clicks a camp's best ring point or the first clear spot, and fails when nothing in view is clear.

When a build step after the builder selection fails, `_handle_build` runs the sequence's closing `h`, so no building is left on the cursor for the next map click, and defers the build for 5 s.

## Input on Windows and macOS

On Windows the handlers call pyautogui directly and the batch focuses the game first. On macOS `game_loop` prepares the game once before the clocks start: it requires Screen Recording and Accessibility permission, applies the keyboard layout, requires a game profile, refuses to start while a menu dims the screen, applies the profile's modifier map, and sets an absolute zoom (30 steps out, then `zoom_steps` in). After that the agent never brings the game forward; switching to another app withholds all input.

| macOS behaviour | Code |
|---|---|
| The active layout is read through Carbon and pyautogui's key table is remapped so each character types what the game expects; the run stops if any of `a`-`z`, `0`-`9`, `.`, `,` cannot be typed | `apps/agent/src/keyboard_macos.py` (`use_layout_keys`) |
| Modifiers macOS reports as held are released before every key | `apps/agent/src/keyboard_macos.py` (`release_stuck_modifiers`) |
| Chords that quit, hide or leave the game (Cmd+Q, Cmd+Tab, Ctrl+arrows and others) and F11 are refused | `executor.py` (`_mac_chord`) |
| Clicks hold the button for 40 ms | `executor.py` (`_mouse_click`) |
| `scroll` is refused, so the detector always sees one zoom | `executor.py` (`_handle_scroll`) |
| Frame pixels are divided by the input scale (2 on Retina) | `apps/agent/src/window.py` (`to_input_point`) |

Key bindings for catalog actions live in `apps/agent/src/policy/catalog.py`:

| Building or menu | Go to | Keys |
|---|---|---|
| Economic build menu | `q` | `q` house, `w` mill, `e` mining camp, `r` lumber camp, `a` farm, `s` blacksmith, `d` market, `f` monastery |
| Military build menu | `w` | `q` barracks, `w` archery range, `e` stable, `r` siege workshop |
| Town Center | `h` | `q` villager, `z` age advance, `a` loom, `s` wheelbarrow |
| Barracks, archery range, stable | Ctrl+B, Ctrl+A, Ctrl+L | `q` and `w` per building's units |
| Mill, lumber camp, mining camp | Ctrl+I, Ctrl+Z, Ctrl+G | `q` for the technology |

`apps/agent/src/prompts/hotkeys.md` is appended to the executor's system prompt as the model's hotkey reference (see [prompts and context](prompts-and-context.md)). It lists the monastery under the military menu and the market under `v`, which disagrees with the catalog; the `build` tool accepts only catalog bindings (`_MENU_BUILDINGS`) and refuses any other as an unknown menu binding. `apps/agent/profiles/` holds two example `GameProfile` files, both on Highland (`magyars_highland_4v4.example.json` for Windows, `magyars_highland_4v4.macos.example.json` for macOS), selected with `AOE2_GAME_PROFILE`. A profile fixes platform and capture size (3024x1672 on Windows, 3024x1898 on macOS), `modifier_map`, `zoom_steps`, `command_button`, the hotkey profile name and the roster, and records whether roster, hotkeys and ownership colours were verified.

## Input lock and revisions

`LoopContext.input_lock` makes sure only one loop types at a time: the actor holds it around one batch, and the deliberate loop holds it around each tool call. The ledger counts inputs. `note_input` bumps `input_revision`; camera keys (`h`, `.`, `,`), rescans, drags, scrolls and zoom also bump `camera_revision`; placements and spending keys bump `economic_revision`, and selection clicks `selection_revision`. Perception stamps each frame with these revisions, and the actor and the `ToolExecutionGate` discard any decision whose frame predates the ledger's current input revision.

## Action ledger and reconciliation

One `ActionLedger` exists per game. `game_loop` binds it to a `ContextVar` with `bind_ledger`, and executor functions reach it through `current_ledger()`. Each commitment gets an operation id and a stream of `ActionOutcome` records with status `pending`, `purchased`, `confirmed`, `failed`, `uncertain` or `cancelled`. `reservations()` sums the cost of pending placements, research and training, and eligibility subtracts it from the stock.

Perception calls `observe_hud` and `observe_age` for every HUD reading (`sync_world_state` in `apps/agent/src/loops/perceive.py`). Settlement compares each reading with the baseline captured at the click, and spend already attributed to other commitments is excluded.

| Commitment | Purchased when | Confirmed when | Fails or turns uncertain when |
|---|---|---|---|
| Building | Its wood cost leaves the HUD within the first 20 readings; houses and unique buildings then block a rebuild | An entity id of that class that was not in view at the click appears at least 25 s after payment, not owned by another player; a house when the population cap rises by 5 | Wood never dropped after 6 s and 3 readings, and never rose by the building's cost or more, which gathering could use to hide the payment (refused, offset remembered); undecided 30 s (50 s for a house) is uncertain, 40 s later failed; paid but unseen after 60 s (house) or 240 s is failed and dropped |
| Technology | Its cost leaves the HUD | Never observed; stays paid | Cost never left: blocked for 50 s; undecided 30 s is uncertain |
| Age advance | Its cost leaves the HUD | The age reading changes; a reading of the target age within 60 s of pressing its advance is the HUD's label for the research and is ignored | As technology |
| Villager or unit | Its cost leaves the HUD, or a delivery proves it | The villager count rises (population when the villager digit is unreadable and no military is queued), or population rises for military | Undecided 30 s is uncertain; dropped 15 s later |

Three missing settlements in a row for one building class suppress that class for 50 s. A sighting alone never makes a building count: classes seen in 3 or more frames without purchase evidence are reported to the executor as unverified. A purchase interrupted before its input went out is `cancelled`; after the input it is `uncertain`. `finalize_unverified` marks every unresolved commitment uncertain when the game ends.

The ledger feeds decisions through `ledger_policy_state`: pending and paid actions, timed suppressions, reservations, stalls and idle readings become `PolicyState` fields that the catalog filters on. `blocked_actions` and the known-buildings line carry the same facts into the executor's context.

## Pending idle-worker assignments

`assign_idle` presses `.` with a rescan, verifies a villager is selected, and picks a target with `safe_gather_target` near the screen centre in the post-jump view. It binds the right click to that entity's id, capture and coordinates, and stores one `_PendingAssignment` with the idle count and resource worker count from before. While it is pending, no other `assign_*` action is eligible and the idle-worker obligation is paused.

`_settle_pending_assignment` confirms the assignment when the HUD worker count for that resource rises and the idle reading fell or was unreadable. After 12 s without that evidence it records `uncertain`, frees the slot, and suppresses that resource's assignment for 20 s. A failed refresh before the click suppresses it for 5 s.

## How failures feed recovery

A `failed` outcome calls `record_failure`, which increments `failure_streak` and keeps the last five reasons in `recent_failures`; `purchased` and `confirmed` reset the streak, and `uncertain` adds a reason without counting. The actor and the tool loop record a failure themselves when an action fails without touching the ledger. Three failures in a row trigger a recovery turn in the [deliberate loop](decision-loops.md#the-deliberate-loop), and `recent_failures` goes into both the TypeSafe request and the executor context. For each failed or uncertain outcome `save_action_evidence` saves the current screenshot to `logs/action_screenshots/action_<id>_<status>.jpg`.

## Offline scenario runner

`apps/agent/src/scenario_runner.py` replays the production perceive, act and execute path without the game. A fixture in `apps/agent/src/scenarios/production/` gives an initial observed state (resources, population, villagers, worker counts, age, buildings, entities) and a list of steps, each a catalog action plus the observation that follows it. `castle_imperial_army.yaml` is the only fixture; the `memory`, `multi_turn` and `strategist` directories are empty.

```yaml
steps:
  - action: queue_villager
    after: {resources: {food: 150}, population: 5, villagers: 4}
  - action: assign_food
    after: {resources: {food: 180}, worker_counts: {food: 1}, idle_count: 2}
```

The runner uses the real `perceive_once`, `act_once`, executor handlers and ledger. It replaces pyautogui with `_ScriptedInput`, uses a null window backend and a scripted frame source, answers every policy request with the step's action at confidence 1.0, and drives the controller from a virtual clock. A step's `after` is applied only when the expected spending input fires: a placement click on clear ground with a villager selected, the research key for a pending research, a right click on the pending assignment's resource, or the unit key with the right building selected. A step fails when the action was not in the feasible candidates, when no spending input was issued, or when the ledger recorded a failure.

```bash
just eval apps/agent/src/scenarios/production/castle_imperial_army.yaml
just eval-all
```
