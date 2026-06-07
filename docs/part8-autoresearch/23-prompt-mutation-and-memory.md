# Chapter 23 — Prompt Mutation and Memory

This chapter zooms into the two mechanical pieces of the autoresearch loop: the mutator that proposes prompt edits, and the memory chain that turns game results into reusable rules.

<aside class="prereqs">

[Chapter 22 — Autoresearch Overview](./22-autoresearch-overview.md). This chapter zooms into the two pieces (mutator and memory chain) that chapter 22 introduces.

</aside>

## The mutator

`apps/autoresearch/src/prompt_mutator.py:50` — `PromptMutator`.

### The system prompt

`prompt_mutator.py:26` (`MUTATOR_SYSTEM`) is the load-bearing part. It frames the LLM as "an expert AoE2 strategist optimizing a system prompt", names the five scoring dimensions and their weights, and lays down four constraints:

- Change **at most 5 lines** of the prompt.
- Do **not** modify `## Output Format` or `## Game State Detection`. These are the contract between the agent and the executor; an edit here breaks parsing.
- Be **specific** ("always build 2 houses before population reaches 10", not "build more houses"). Vague edits are unmeasurable.
- Each change targets **one specific weakness**. Bundled edits make accept/reject signal-less.

Each edit is a small JSON object: `{description, old_text, new_text, rationale}`, and `old_text` must exist verbatim in the current prompt (the mutator does `.replace(old_text, new_text, 1)` — no regex, no fuzzy match).

The proposer actually runs under `REFLECTIVE_MUTATOR_SYSTEM` (`prompt_mutator.py:56`) — `MUTATOR_SYSTEM` plus a short reflection addendum instructing the model to diagnose what failed turn-to-turn and target the weakest component with trace evidence.

`propose_changes(current_prompt, recent_traces, component_breakdown, failure_modes, n)` (`prompt_mutator.py:75`) is the entry point the tournament calls. It builds a reflective user message — turn-by-turn excerpts from the recent `GameTrace`s (via `format_trace_excerpt`) plus the five-component score breakdown — requests **N distinct edits as a JSON array**, then `_parse_changes` extracts the array via `extract_json_array` (`json_utils.py`) and drops any element missing the required keys (`_is_valid_change`). With no prior traces (a first run) it substitutes a placeholder line and still proposes. There is no separate single-candidate method — a one-edit run is just `n=1`.

### The protection check

Even with the prompt constraint, the mutator code defensively re-checks (in `apply_change`, `prompt_mutator.py:135`): if `old_text` falls inside a `PROTECTED_SECTIONS` span, the change is rejected as `change_in_protected_section`. The check walks from the section header to the next `\n## `, so it's robust to where the LLM positioned its anchor.

### Revert

`PromptMutator.revert()` is `git checkout -- prompts/system.md`. No diff parsing, no partial undo — atomic file-level revert. This is why every accepted change is its own commit: revert resolution is "go back to the file as of the most recent commit". In a tournament **every trial game reverts to baseline immediately after it finishes** (in a `finally`, so a crashed game never leaves the prompt dirty), which is what lets candidates be compared against the same starting text instead of stacking. Only the winner is re-applied and committed at the end.

### Failure modes

The tournament handles these before spending a game:

| Outcome | What happens |
|---|---|
| Mutator API call failed / unparseable | `propose_changes` returns `[]` → the tournament logs `tournament_no_candidates` and exits without playing. |
| Malformed array element (missing keys) | Dropped by `_parse_changes`; the remaining candidates still race. |
| `old_text` not in prompt | Candidate filtered out by `_candidate_applies` before any game; the rest still race. |
| Change overlaps protected section | `apply_change` refuses the edit (`change_in_protected_section`) at trial time. |

None of these waste a game run — candidate filtering happens before `run_game` is called.

## The memory chain

`apps/autoresearch/src/memory_chain.py:99` — `MemoryChain`. Two responsibilities: extracting memories after a game, and loading them as context for the next game.

### Extraction

`extract_memories(memory, score, game_id, model)` (`memory_chain.py:107`):

1. `_build_game_summary` (`memory_chain.py:283`) — builds a text summary from `AgentMemory.working_memory`. Metrics, then "Turn-by-turn summary (last 10 turns)" with reasoning truncated to 150 chars and a compact action summary.
2. Sends the summary to Haiku with `EXTRACTION_SYSTEM` (`memory_chain.py:56`). The system prompt is unusually strict:
   - First person, present/future tense.
   - Imperative, not diagnostic. ("I should X. Last game, Y happened — that's why.")
   - Specific thresholds. ("Build a house when population reaches pop_cap minus 3", not "manage housing better".)
   - One rule per note.
   - No turn numbers in the rule.
   - At most 3 notes; prefer 0 or 1 if nothing genuinely new happened.
3. Parses the JSON response (`_parse_observations`, `memory_chain.py:317`), drops entries with empty content (a known failure mode that produced 0-byte files before the guard was added).
4. **Light client-side dedup**: any observation whose sanitized title matches an existing on-disk title is skipped. Prevents the directory from accumulating duplicate "stuck_at_population_cap"-type rules.
5. Writes each surviving observation to `memories/NNN_<sanitized_title>.md` with frontmatter (`type`, `title`, `game_id`, `applies_when`, `score_impact`, `created`).

### The frontmatter schema

| Field | Type | Used by |
|---|---|---|
| `type` | `strategy \| economy \| military \| detection \| failure` | UI / human review; not used by the loader. |
| `title` | snake_case | Surfaced in the `[applied: title]` reasoning prefix (the agent emits this when it follows a memory rule). |
| `game_id` | experiment id | Attribution for `git blame`-style auditing. |
| `applies_when` | free-text trigger | Surfaced as `(when: ...)` prefix to the memory line. The agent matches it against current state. |
| `score_impact` | `negative \| positive \| neutral` | Ranks memories at load-time. |
| `created` | ISO 8601 | Tiebreak within an impact tier (newer wins). |

### Loading

`load_memories(max_tokens=800)` (`memory_chain.py:179`) builds the `## Notes to Myself from Previous Games` block injected into the agent's context. Three sort stages:

1. Drop empty bodies; build a list of `_MemoryEntry`.
2. Sort by `created` descending (lexicographic on ISO 8601 — that's why the timestamp format matters).
3. Stable-sort by `_IMPACT_RANK` (negative=0, positive=1, neutral=2) — so within each impact tier the order from step 2 is preserved.
4. Cap at `_MAX_MEMORIES = 20`, then trim by token budget (1 token ≈ 4 chars).

The block has a precedence header: **"when a memory rule conflicts with a rule in `core.md` or the age-specific section, follow the MEMORY."** Memories reflect concrete evidence from past games; the defaults are pre-game heuristics. The agent is told to apply any rule whose trigger matches its current state, and when two memories conflict, prefer the more specific trigger.

### Why this shape works

Three properties make the memory chain self-correcting rather than self-amplifying:

- **Human-reviewable.** Every file is plain markdown. Bad rules are `rm`-able.
- **Trigger-gated.** `applies_when` keeps a rule out of unrelated contexts. "Build a house when pop nears cap" doesn't fire when the agent is in a battle.
- **Negative-first.** The loader ranks traps-to-avoid above patterns-to-repeat. If the dir is full of mediocre advice but contains one clear "do not do X", the do-not-do-X surfaces.

### Known gaps

- **No semantic dedup.** Two memories saying the same thing in different words both load.
- **No half-life.** Memories from 100 games ago compete equally with last week's. The 20-cap and recency tiebreak limit the damage but don't eliminate it.
- **Title sanitization is greedy.** `re.sub(r"[^a-z0-9_]", "_", ...)` collapses `_` runs in inconvenient ways. Filename uniqueness is preserved (numeric prefix) but readability suffers.

These are explicitly accepted trade-offs — the file-based design is what makes the system tractable and reviewable. A more clever store would also be more opaque.

<details class="deep-dive">
<summary>Deep dive — Episodic vs semantic memory in agents (and where RAG would slot in)</summary>

Cognitive science distinguishes two memory systems that have direct analogues in LLM-agent design:

- **Episodic memory** — what happened, when, in what order. "On Tuesday I queued villagers and ran out of food by turn 18." Indexed by event, retrievable by time or causal chain. Decays naturally.
- **Semantic memory** — abstracted rules and facts pulled out of many episodes. "Don't queue villagers when food < 500." Indexed by concept, retrievable by context, durable.

Our memory chain has **both**:

- *Working memory* (`AgentMemory.working_memory`, a 10-turn deque in `apps/agent/src/memory.py`) is the agent's *episodic* memory during a game — the last few turns, available as context for the executor's next decision. Bounded, recency-biased, lossy.
- *The `memories/*.md` directory* is the agent's *semantic* memory across games — the rules extracted from past episodes, ranked and loaded as context at the start of every new game. Persistent, hand-reviewable, deduplicated by title.

The chapter's `EXTRACTION_SYSTEM` prompt is the **episodic → semantic transformation**: it asks the LLM to summarize a game's turn-by-turn working memory into one or two reusable rules. "Last game on turn 14–18 I queued three villagers and that delayed age-up by 4 turns" becomes "I should stop queueing villagers when food drops below 500 in Dark Age." The episodic detail (turns 14–18, *this* specific game) is preserved in the body for context but doesn't drive retrieval — the `applies_when` trigger does.

**Where RAG would slot in.** Right now retrieval is a sort + token-budget cap. A vector-DB-backed retrieval would replace the sort with a semantic similarity query: "given my current state X, what memories are most relevant?" That's a real upgrade because (a) it handles the *no semantic dedup* gap honestly (two memories saying the same thing in different words would collide in embedding space), and (b) it makes the rules indexed by what they're *about*, not when they were written. The trade-off is opacity — debugging "why did this memory get retrieved?" becomes harder than `cat` and `grep`.

**Where vector embeddings won't help.** The *applies_when* gating is doing most of the work right now. Memories with sharp triggers ("when food < 500 AND age == Dark Age") don't suffer from retrieval noise because they don't fire when irrelevant — embedding similarity wouldn't change that. Where embeddings would help is in the long tail of vaguer memories ("don't expand too early") that *do* over-fire.

**The framework people invoke at this point** is the classic RAG triad: (1) chunk your knowledge, (2) embed and index, (3) at query time, embed the query and pull the top-K chunks. The simpler our memory file is, the more natural that pipeline becomes. We've kept the schema simple (frontmatter + body, one rule per file) explicitly so that bolting on RAG later is a small change — not a memory-store rewrite.

</details>

## Where this code touches the rest of the system

- Reads `prompts/system.md` — the executor system prompt the real game agent uses.
- Writes git commits to the main repo — `prompts/system.md` changes show up in `git log`.
- Writes `memories/*.md` — read at agent startup by `ClaudeProvider` (the executor LLM context builder).
- Writes `experiments/results.tsv` — the canonical ledger; not source-controlled (it's per-machine experiment state).

## Related reading

- [Chapter 22 — Autoresearch Overview](./22-autoresearch-overview.md) — the orchestrator and scoring.
- [Chapter 6 — Context Injection](../part2-llm-integration/06-context-injection.md) — how memories are spliced into the executor's context.
- [`docs/design/autoresearch-plan.md`](../design/autoresearch-plan.md) — the original 5-phase plan (frozen historical).
