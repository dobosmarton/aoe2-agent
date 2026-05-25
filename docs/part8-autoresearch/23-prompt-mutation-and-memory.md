# Chapter 23 — Prompt Mutation and Memory

This chapter zooms into the two mechanical pieces of the autoresearch loop: the mutator that proposes prompt edits, and the memory chain that turns game results into reusable rules.

## The mutator

`apps/autoresearch/src/prompt_mutator.py:50` — `PromptMutator`.

### The system prompt

`prompt_mutator.py:20` (`MUTATOR_SYSTEM`) is the load-bearing part. It frames the LLM as "an expert AoE2 strategist optimizing a system prompt", names the five scoring dimensions and their weights, and lays down four constraints:

- Change **at most 5 lines** of the prompt.
- Do **not** modify `## Output Format` or `## Game State Detection`. These are the contract between the agent and the executor; an edit here breaks parsing.
- Be **specific** ("always build 2 houses before population reaches 10", not "build more houses"). Vague edits are unmeasurable.
- Each change targets **one specific weakness**. Bundled edits make accept/reject signal-less.

The output schema is a small JSON object: `{description, old_text, new_text, rationale}`. `old_text` must exist verbatim in the current prompt (the mutator does `.replace(old_text, new_text, 1)` — no regex, no fuzzy match).

### The protection check

Even with the prompt constraint, the mutator code defensively re-checks (`prompt_mutator.py:122`): if `old_text` falls inside a `PROTECTED_SECTIONS` span, the change is rejected as `change_in_protected_section`. The check walks from the section header to the next `\n## `, so it's robust to where the LLM positioned its anchor.

### Revert

`PromptMutator.revert()` (`prompt_mutator.py:142`) is `git checkout -- prompts/system.md`. No diff parsing, no partial undo — atomic file-level revert. This is why every accepted change is its own commit: revert resolution is "go back to the file as of the most recent commit", and the orchestrator commits the revert separately so the timeline reads naturally in `git log`.

### Failure modes

The orchestrator handles three:

| Outcome | Logged as | What happens |
|---|---|---|
| Mutator API call failed | `mutation_failed` | Skip experiment, run with current prompt (treated as a baseline). |
| `old_text` not in prompt | `change_not_applicable` | Skip, don't commit. |
| Change overlaps protected section | `change_in_protected_section` | Skip, don't commit. |

None of these waste a game run — they short-circuit before `run_game` is called.

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

## Where this code touches the rest of the system

- Reads `prompts/system.md` — the executor system prompt the real game agent uses.
- Writes git commits to the main repo — `prompts/system.md` changes show up in `git log`.
- Writes `memories/*.md` — read at agent startup by `ClaudeProvider` (the executor LLM context builder).
- Writes `experiments/results.tsv` — the canonical ledger; not source-controlled (it's per-machine experiment state).

## Related reading

- [Chapter 22 — Autoresearch Overview](./22-autoresearch-overview.md) — the orchestrator and scoring.
- [Chapter 6 — Context Injection](../part2-llm-integration/06-context-injection.md) — how memories are spliced into the executor's context.
- [`docs/design/autoresearch-plan.md`](../design/autoresearch-plan.md) — the original 5-phase plan (frozen historical).
