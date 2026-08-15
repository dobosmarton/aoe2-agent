# Glossary

One-line definitions for every technical term used in the tutorial, with deep links to the deep dive that explains it in context. If you hit an unfamiliar name in a chapter, this is the fastest way to recover.

Entries are organized alphabetically. Most are followed by a short link of the form **→ [Chapter X §Y](path)** that points to the deep dive, sidebar, or appendix where the term is properly introduced.

---

## A

- **Adapter (LLM)** — One `ChatWire` implementation plus the name that selects it. Three exist: `openai`, `zen` and `anthropic`, chosen with `AOE2_LLM_WIRE`. → [Chapter 4 §4.2](./part2-llm-integration/04-provider-pattern.md).
- **Action-effect verification** — After an entity-affecting action (a build/placement or a camera move), the agent re-detects and confirms the expected change, recording `CONFIRMED …` or the exact phrase `no visible change`; repeated misses feed the stuck-loop detector. → [Chapter 2 §2.1 Step 11](./part1-architecture/02-game-loop-pipeline.md).
- **Anchor box** — A YOLO-style detector predicts each object's bounding box as an *offset* from a pre-defined anchor of a given aspect ratio, rather than predicting raw coordinates. → [Appendix A — YOLO and object detection](./appendix/01-yolo-and-object-detection.md).
- **Asyncio** — Python's cooperative-multitasking concurrency primitive. The agent's game loop is built on a single asyncio event loop so the strategist can think in the background while the executor is acting. → [Chapter 1 §Async-first architecture](./part1-architecture/01-system-overview.md).

## B

- **Bradley-Terry** — A statistical model that turns *pairwise win/loss outcomes* into a single skill rating per competitor by maximum likelihood. → [Appendix C — Bradley-Terry MLE and bootstrap CIs](./appendix/03-bradley-terry-and-bootstrap.md).
- **Backpressure** — When a producer is faster than its consumer, backpressure is the mechanism that slows the producer (or drops events) so memory doesn't blow up. → [Chapter 15 §Backpressure semantics](./part6-evaluation-arena/15-event-broker.md).
- **Bootstrap (statistical)** — A resampling technique that estimates the uncertainty of a statistic by repeatedly sampling with replacement from the observed data. We use it to put confidence intervals on Bradley-Terry skill estimates. → [Appendix C](./appendix/03-bradley-terry-and-bootstrap.md).

## C

- **ChatWire** — The `Protocol` that every LLM transport satisfies: four methods covering a tool turn, a structured-output call, and two exception classifiers. It holds everything vendor-specific and no game logic, so one executor serves every model. → [Chapter 4 §4.1](./part2-llm-integration/04-provider-pattern.md).
- **Cache hit / cache miss (LLM)** — Whether the provider was able to reuse its internal KV-cache for a prefix of your request. Visible in `usage.cache_read_input_tokens`. → [Chapter 5 — Prompt caching callout](./part2-llm-integration/05-prompt-engineering.md).
- **`cache_control`** — Anthropic's marker for "this block of the prompt can be cached." → [Chapter 5 — Prompt caching callout](./part2-llm-integration/05-prompt-engineering.md).
- **COCO format** — A JSON-based bounding-box / segmentation label format with explicit category IDs. The default format CVAT exports. → [Chapter 9 — COCO vs YOLO callout](./part3-entity-detection/09-labeling-and-active-learning.md).
- **Columnar storage** — Storing tables one *column at a time* instead of one row at a time. Massively faster for analytical scans because you only read the columns you query. The basis of DuckDB, Parquet, and ClickHouse. → [Chapter 16 — DuckDB internals](./part6-evaluation-arena/16-duckdb-persister-and-replay.md).
- **Composite tool** — In our agent, a tool that internally chains several primitive actions (press, click, rescan) into one API roundtrip. → [Chapter 5 §5.3](./part2-llm-integration/05-prompt-engineering.md).
- **Confidence interval (CI)** — A range estimate around a statistic that contains the true value with a stated probability under repeated sampling. → [Appendix C](./appendix/03-bradley-terry-and-bootstrap.md).

## D

- **DuckDB** — An embedded analytical (OLAP) database. Like SQLite for analytics: zero-config, file-based, columnar, vectorized. We use it as the cold-path event log. → [Chapter 16](./part6-evaluation-arena/16-duckdb-persister-and-replay.md).
- **DXT1 / BC1** — A GPU-friendly *block-compression* texture format that encodes every 4×4 pixel block into 8 bytes by storing two endpoint colors plus a 2-bit index per pixel. AoE2's SLD sprite format uses it. → [Chapter 11 — DXT1 deep dive](./part4-game-knowledge/11-sprite-extraction.md).
- **Determinism (LLM context)** — Whether the same prompt yields the same output. Surprisingly hard with LLMs: even `temperature=0` plus a fixed seed isn't fully deterministic across hardware due to BF16 floating-point order. → [Chapter 18 — Determinism deep dive](./part6-evaluation-arena/18-synthetic-world-sim.md).

## E

- **Event broker** — A piece of infrastructure that decouples event producers from consumers via pub/sub, queues, or streams. → [Appendix B — Event brokers and Redis Streams](./appendix/02-event-brokers-and-redis-streams.md).

## H

- **Hungarian algorithm** — A polynomial-time algorithm for assigning N predicted tracks to N detected entities optimally given a cost matrix (typically IoU or pixel distance). The standard pairing step in multi-object trackers. → [Chapter 2 — Kalman + Hungarian deep dive](./part1-architecture/02-game-loop-pipeline.md).

## I

- **IoU (Intersection over Union)** — A similarity score for two boxes: `area(A ∩ B) / area(A ∪ B)`. Used everywhere in detection — for NMS, for matching tracks to detections, for measuring mAP. → [Appendix A](./appendix/01-yolo-and-object-detection.md).

## K

- **Kalman filter** — A recursive estimator that maintains a state distribution (mean + covariance) over time, predicting forward each step and correcting against noisy observations. We use a constant-velocity Kalman per tracked entity. → [Chapter 2 — Kalman deep dive](./part1-architecture/02-game-loop-pipeline.md).

## M

- **mAP / mAP50 / mAP50-95** — Mean Average Precision: the standard detection metric. *mAP50* averages precision across classes at IoU ≥ 0.5; *mAP50-95* averages across 10 IoU thresholds from 0.5 to 0.95 in 0.05 steps. → [Appendix A](./appendix/01-yolo-and-object-detection.md).
- **MLE (Maximum Likelihood Estimation)** — Choosing parameter values that make the observed data most probable under the model. Bradley-Terry skills are fit by MLE. → [Appendix C](./appendix/03-bradley-terry-and-bootstrap.md).
- **MM algorithm (Minorization–Maximization)** — The iterative fitting algorithm we use for Bradley-Terry. Each step bounds the log-likelihood from below by a simpler surrogate, then maximizes the surrogate. Hunter (2004). → [Appendix C](./appendix/03-bradley-terry-and-bootstrap.md).

## N

- **NMS (Non-Maximum Suppression)** — The post-processing step that takes a detector's raw box predictions, sorts by confidence, and greedily drops any box whose IoU with a higher-confidence box exceeds a threshold. → [Appendix A](./appendix/01-yolo-and-object-detection.md).

## O

- **OpenCode Zen** — A gateway that speaks the OpenAI Chat Completions shape, reaching GPT-5.6 Luna, Kimi and GLM behind one key. Selected with `AOE2_LLM_WIRE=zen`, which supplies its endpoint automatically. → [Chapter 4 §4.2](./part2-llm-integration/04-provider-pattern.md).
- **OLAP vs OLTP** — *OLAP* (analytical) workloads read lots of rows and few columns to compute aggregates; *OLTP* (transactional) workloads read/write a few rows touching many columns. DuckDB is OLAP; Postgres/SQLite are primarily OLTP. → [Chapter 16](./part6-evaluation-arena/16-duckdb-persister-and-replay.md).
- **OPRO** — Optimization by PROmpting: a paper from DeepMind where one LLM proposes new prompts and another LLM scores them, iteratively converging on better prompts. Related to our autoresearch loop. → [Chapter 22 — Prompt optimization landscape](./part8-autoresearch/22-autoresearch-overview.md).

## P

- **Pareto frontier** — The set of candidates that are *non-dominated*: none is beaten on every score component at once. The autoresearch tournament keeps a frontier over the five score components so a candidate strong on a single axis isn't culled for a weak composite. → [Chapter 22 — How a tournament runs](./part8-autoresearch/22-autoresearch-overview.md).
- **Prompt caching** — Provider-side reuse of the KV-cache for a stable prefix of your prompt. → [Chapter 5 — Prompt caching callout](./part2-llm-integration/05-prompt-engineering.md).

## R

- **Reactive tier** — A deterministic, no-LLM layer that runs routine economy upkeep every turn (queue villagers below the age cap, reassign idle villagers to the nearest resource); it cedes combat to the LLM by returning nothing on alarm. → [Chapter 2 §2.1 Step 9](./part1-architecture/02-game-loop-pipeline.md).
- **Redis Streams** — A Redis data structure (`XADD` / `XREAD`) that behaves like a durable, replayable, consumer-group-aware log. Our cross-process broker backend. → [Appendix B](./appendix/02-event-brokers-and-redis-streams.md).
- **Reflective prompt mutation** — Proposing prompt edits by reasoning over full game *traces* (turn-by-turn reasoning, actions, verification) plus the per-component score breakdown, rather than over summary metrics alone. → [Chapter 23 — The mutator](./part8-autoresearch/23-prompt-mutation-and-memory.md).
- **RTC (turn pipelining)** — Request-to-completion overlap: a routine turn launches the *next* executor plan as a background task and executes the previous turn's committed head while it computes, hiding the LLM roundtrip. → [Chapter 2 §2.1 Step 9](./part1-architecture/02-game-loop-pipeline.md).

## S

- **SAHI (Slicing Aided Hyper Inference)** — A technique for detecting small objects in large images by tiling the input, running the detector on each tile, and stitching results. Implemented (including an *adaptive* variant that only tiles likely regions) but **disabled** (`adaptive_sahi=False`): tiling a retina screenshot into 640 crops shows the model objects ~2.4× larger than its training scale, which *lowers* real F1, so the agent runs a single pass at `imgsz=1280` (v9's training resolution) instead. → [Chapter 7 §7.4](./part3-entity-detection/07-detector-architecture.md).
- **SSE (Server-Sent Events)** — A one-way HTTP streaming protocol where the server pushes `text/event-stream` chunks to the client. Simpler than WebSocket for fan-out telemetry. → [Chapter 19 — SSE callout](./part7-arena-web/19-web-architecture.md).
- **Structured output** — Constraining an LLM to emit a parseable, schema-validated response (typically JSON). → [Chapter 5 — Structured output callout](./part2-llm-integration/05-prompt-engineering.md).

## T

- **Tool use / function calling** — An API pattern where the LLM is given a set of "tools" with JSON-schema arguments; the model emits a tool call and the host runs it and feeds back the result. → [Chapter 4 — Agentic tool loops deep dive](./part2-llm-integration/04-provider-pattern.md).

## V

- **Vectorized execution** — Processing a batch (vector) of rows at a time through each operator, instead of one row at a time. The reason DuckDB and modern OLAP engines are 10–100× faster than row-at-a-time engines on analytical queries. → [Chapter 16](./part6-evaluation-arena/16-duckdb-persister-and-replay.md).

## W

- **Wire** — Chapter 4's term for a vendor transport: the message envelope, tool-schema shape, usage field names, stop-reason spelling and exception classes for one API. Swapping vendor is a wire swap, not a second executor. → [Chapter 4](./part2-llm-integration/04-provider-pattern.md).
- **WireName** — The `Literal["anthropic", "openai", "zen"]` in `config.py` that is the single source for the valid adapter set. `config._parse_wire` validates against it and the factories dispatch over it with `assert_never`, so a new adapter fails the typecheck until every arm handles it. → [Chapter 4 §4.2](./part2-llm-integration/04-provider-pattern.md).

## Y

- **YOLO** — *You Only Look Once*: a family of single-shot object detectors that predict all boxes in one forward pass. We use YOLO26n (NMS-free). → [Appendix A](./appendix/01-yolo-and-object-detection.md).
