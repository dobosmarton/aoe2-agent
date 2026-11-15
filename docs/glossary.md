# Glossary

Terms used across the wiki, with the page that explains each one.

## Gameplay agent

| Term | Meaning | Page |
|---|---|---|
| ActionLedger | Per-game record of HUD baselines, pending commitments, reservations and settled outcomes, kept in `apps/agent/src/executor.py`. | [Actions and execution](agent/actions-and-execution.md) |
| ActionOutcome | One ledger record, with status pending, purchased, confirmed, failed, uncertain or cancelled. | [Actions and execution](agent/actions-and-execution.md) |
| Allocation focus | TypeSafe's `balanced`, `food`, `wood`, `gold` or `stone` choice. At confidence 0.65 or above it adds one villager slot to the per-age mix. | [Decision loops](agent/decision-loops.md) |
| camera_revision | Ledger counter bumped by camera inputs and by detected pans or resets. A box from another revision cannot be clicked. | [Perception](agent/perception.md) |
| capture_id | Increasing number of a native capture within one `GameSource`. Boxes and HUD readings each record the capture they came from. | [Perception](agent/perception.md) |
| ChatWire | The protocol each LLM vendor adapter implements. It holds every vendor detail and no game logic. | [LLM providers](agent/llm-providers.md) |
| class_agreement | The winning label's share of the tracker's confidence-weighted label vote. It is not a calibrated probability. | [Tracking](agent/tracking.md) |
| Deterministic fallback | `select_fallback`, the ordered preferences the actor uses when TypeSafe advice times out, fails or is rejected. When it would only `wait`, TypeSafe's low-confidence pick runs instead. | [Decision loops](agent/decision-loops.md) |
| EntitySnapshot | Immutable published entity: track ID, smoothed and raw label, class agreement, box and capture stamps. | [Perception](agent/perception.md) |
| Feasible catalog | The `ActionSpec` entries in `apps/agent/src/policy/catalog.py`, filtered by `eligible()` against the current `PolicyState`. | [Decision loops](agent/decision-loops.md) |
| Footprint | A building's ground in frame pixels, its edge in tiles scaled from the detected Town Center's width (4 tiles). Placement ranks a spot whose footprint overlaps a building, tree or mine below every clear one. | [Actions and execution](agent/actions-and-execution.md) |
| FramePipe | Single-writer channel from the perception loop to the act and deliberate loops. It holds the latest `Perception` and wakes waiters on each publication. | [Perception](agent/perception.md) |
| Input revision | Ledger counter bumped by every input. A decision computed from an older frame is discarded. | [Actions and execution](agent/actions-and-execution.md) |
| Memory | A Markdown rule in `apps/agent/memories/NNN_title.md`, written after a game by `MemoryChain` and loaded into the executor's cached system prompt. | [Prompts and context](agent/prompts-and-context.md) |
| Obligation | Idle-worker or Town Center production work that `AgentController` forces once it has been observable for 5 s. TypeSafe cannot decline it. | [Decision loops](agent/decision-loops.md) |
| Perception | The frozen frame the loops share: screenshot, entities, HUD readings and separate spatial and HUD capture IDs, versions and times. | [Perception](agent/perception.md) |
| Placement preview | The game's drawing of the building on the cursor, red where it cannot go. The agent hovers up to 8 spots and clicks the first one whose preview is drawn and not red (`apps/agent/src/placement_preview.py`). | [Actions and execution](agent/actions-and-execution.md) |
| PolicyAdvice | TypeSafe's answer for one frame: an action choice and an allocation focus, each with a confidence, tied to the source tick. | [Decision loops](agent/decision-loops.md) |
| PolicyState | Frozen snapshot of observed HUD facts plus unsettled ledger commitments, read by the catalog and by TypeSafe. | [Decision loops](agent/decision-loops.md) |
| RecoveryBudget | Allows three recovery cycles without food or villager progress, then stops the game with `essential_progress_unverified`. | [Actions and execution](agent/actions-and-execution.md) |
| screentrack | Rust tracker library with Python bindings, the default tracker backend (`AOE2_TRACKER=native`). | [Tracking](agent/tracking.md) |
| Shadow tracker | `AOE2_TRACKER=shadow`: runs the Python and Rust trackers together, acts on Python's output and logs the differences. | [Tracking](agent/tracking.md) |
| Spatial refresh | A post-input capture that an action requests through `frame_refresh`, with a 3 s timeout. | [Perception](agent/perception.md) |
| Strategist brief | Text from `apps/agent/src/strategist_brief.py` in each strategist request: the situation from the agent's records, the gap to the next age and the actions the current age allows. | [Prompts and context](agent/prompts-and-context.md) |
| System blocks | The executor's two cacheable system prompt parts: `core.md`, `hotkeys.md` and memories, then the current age's prompt. | [Prompts and context](agent/prompts-and-context.md) |
| tactical_handoff | Catalog action, feasible only with an own army visible, that asks the deliberate loop to run the LLM executor. | [Decision loops](agent/decision-loops.md) |
| TypeSafe System One | The routine policy model that picks one feasible catalog action per frame. | [Decision loops](agent/decision-loops.md) |
| Wire | The adapter that selects a vendor: `openai`, `zen` or `anthropic`, set with `AOE2_LLM_WIRE`. | [LLM providers](agent/llm-providers.md) |

## Detection

| Term | Meaning | Page |
|---|---|---|
| Checked sidecar | An X-AnyLabeling per-image JSON with `checked: true`, accepted as review evidence. | [Labeling](detection/labeling.md) |
| Composite | A real frame with game sprites pasted onto it at real screen scale (`training/generate_composites.py`). | [Training](detection/training.md) |
| detector_busy | The detection server's 503 reply while an inference is running. The agent discards that frame. | [Serving and evaluation](detection/serving-and-evaluation.md) |
| Independent test set | 40 frames from 10 agent sessions that no training manifest uses, built by `labeling/build_test_set.py`. | [Labeling](detection/labeling.md) |
| IoU | Intersection over union of two boxes, used for NMS, matching and accuracy. | [YOLO primer](appendix/01-yolo-and-object-detection.md) |
| Label alias | A reviewer label finer than a class that trains as an existing class, such as ibex as deer. | [Labeling](detection/labeling.md) |
| Letterbox | Aspect-preserving resize onto a grey canvas before inference. | [Entity detector](detection/detector.md) |
| NMS | Non-maximum suppression: dropping boxes that overlap a higher-scoring box of the same object. | [YOLO primer](appendix/01-yolo-and-object-detection.md) |
| One-to-many head | The YOLO26 export that outputs raw predictions and needs host-side NMS. The served model uses it. | [Entity detector](detection/detector.md) |
| RequiredRemoteDetector | Remote detector client with no local fallback, used for recorded experiments. | [Entity detector](detection/detector.md) |
| SAHI | Sliced inference over image tiles, available in the library and server. No agent path calls it. | [Entity detector](detection/detector.md) |
| SLD | The AoE2: DE sprite file format that the extractor decodes. | [Sprite extraction](detection/sprite-extraction.md) |
| Synthetic render | A scene of sprites on terrain with automatic labels (`training/generate_training_data.py`). | [Training](detection/training.md) |

## Arena, web and autoresearch

| Term | Meaning | Page |
|---|---|---|
| Bradley-Terry rating | Log-strength per profile fitted from pairwise wins, centred on zero. | [Ranking](arena/ranking.md) |
| Cold path | `/events` serving a closed run by replaying its DuckDB log. | [Web architecture](web/architecture.md) |
| EventBroker | The protocol every arena event passes through, with in-process and Redis Streams implementations chosen by `ARENA_BROKER_BACKEND`. | [Event broker](arena/event-broker.md) |
| EventEnvelope | A `(run_id, seq, event)` triple that the broker hands to consumers. | [Event broker](arena/event-broker.md) |
| Experiment gate | `just experiment-gate`: passes only if the ledger has a scored game at the current commit. | [Autoresearch](autoresearch/overview.md) |
| Experiment ledger | `experiments/results.tsv`, one row per scored game. | [Autoresearch](autoresearch/overview.md) |
| Fork | A child run created by `POST /forks` from a parent run's state at a chosen turn, optionally with a `MutationPatch`. | [Web architecture](web/architecture.md) |
| GameScore | The autoresearch 0 to 1 composite: age 0.40, age speed 0.25, economy 0.20, action success 0.10, survival 0.05. | [Autoresearch](autoresearch/overview.md) |
| Live path | `/events` streaming an open run straight from the broker. | [Web architecture](web/architecture.md) |
| MutationPatch | Optional overrides for a fork's resources, population, pop cap and age. Unknown fields are rejected. | [Web architecture](web/architecture.md) |
| Pareto frontier | The prompts that no other prompt beats on every score component. | [Autoresearch](autoresearch/overview.md) |
| Reaper | The API task that drops a closed fork's broker buffer after a grace period. | [Web architecture](web/architecture.md) |
| Scenario | A named starting `WorldState` for ranking: balanced, food-poor, wood-poor or late-start. | [Ranking](arena/ranking.md) |
| score_valid | False when the ledger holds uncertain operations or the game profile is missing or unverified. | [Autoresearch](autoresearch/overview.md) |
| Seq | Per-run sequence number the broker assigns to each event, giving a total order for replay and resume. | [Event broker](arena/event-broker.md) |
| SSE | Server-sent events: one-way HTTP streaming that the dashboard uses to read run events. | [Web architecture](web/architecture.md) |
| WorldState | The synthetic arena's game state dataclass in `packages/core/src/world_state.py`. | [World sim](arena/world-sim.md) |
