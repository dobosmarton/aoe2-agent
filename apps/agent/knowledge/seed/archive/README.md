# Archived memories — the only cross-game memory the agent ever wrote

9 files, recovered from `logs/2026_04_24/memories/` and `logs/2026_04_25/`.
These are the complete output of `memory_chain.py` to date. The live
`memories/` directory is empty.

The originals sit here verbatim. **Do not import them into the registry as they
stand** — see the conversion notes below.

## The finding worth reading first

**The memory system independently rediscovered 4 rules that were later hardcoded
by hand.** Compare:

| The memory wrote | Later hardcoded as |
| --- | --- |
| "build a house the moment population reaches pop_cap minus 5" | `executor._HOUSE_HEADROOM_MAX = 4` (F-45) |
| "age up to Feudal before population reaches 25" | `reactive._age_up_actions` (F-26) |
| "repeated press(h), press(z) … ineffective at advancing the age-up" | The greyed-button gate (F-26) and the UI-context leak (F-27) |
| "switch location or cancel after 2 failed build() attempts" | `executor._MISSING_STREAK_LIMIT = 3` (T-530) |

The loop found the right answers months before a human wrote them into Python.
What it lacked was a way to **act** on them and a way to **measure** whether they
helped. That gap is what `ADAPTIVE-AGENT-PLAN.md` closes.

This is the strongest evidence available that the seeded arm of the 6.3 bootstrap
experiment is worth running.

## Why these cannot be imported as they stand

**The 2026_04_24 batch (5 usable files)** predates the current schema:
- No `title`, so `[applied: …]` attribution cannot match them.
- No `applies_when`, so there is no trigger to evaluate.
- They cite absolute turn numbers ("turns 31-39", "From turn 30 onwards"), which
  `memory_chain.EXTRACTION_SYSTEM` now explicitly forbids: *"No turn numbers in
  the rule. The rule must apply purely from the next game's observable state."*
- Third-person and diagnostic ("The agent reached population cap…") rather than
  the imperative first person the current prompt requires.

**`006_missing_feudal_age_target.md` is 0 bytes.** It is kept as evidence of the
bug that `_parse_observations` now guards against — empty content is dropped
before a file is written.

**The 2026_04_25 batch (3 files)** already uses the current schema: `title`,
`applies_when`, first person, imperative, no turn numbers. These convert almost
directly.

## Conversion status

| File | Converts to | Note |
| --- | --- | --- |
| `2026_04_24_001` | superseded by `house_when_headroom_gone` | Its threshold (cap minus 4+) is less precise than the shipped rule. |
| `2026_04_24_002` | superseded by `age_up_feudal` | Correct instinct; the shipped rule adds the two-building gate it lacked. |
| `2026_04_24_003` | superseded by safety floor | It correctly *suspected* the h/z sequence but misdiagnosed the cause. The real cause was the greyed button (F-26), not a hotkey mapping error. **A memory can be right about the symptom and wrong about the mechanism.** |
| `2026_04_24_004` | superseded by `house_when_headroom_gone` | Duplicate of 001. |
| `2026_04_24_005` | superseded by `house_when_headroom_gone` | Duplicate of 001. |
| `2026_04_24_006` | dropped | 0 bytes. |
| `2026_04_25_001` | superseded by `house_when_headroom_gone` | Its "minus 5" is outside the executor's allow band of 4; the shipped rule uses 2. |
| `2026_04_25_002` | superseded by `age_up_feudal` | Adds a population trigger the shipped rule does not have. Worth testing as a variant. |
| `2026_04_25_003` | superseded by `stop_paying_for_a_build_that_keeps_vanishing` | Proposed 2 attempts; the shipped circuit breaker uses 3. Worth ablating. |

**Every one of the 9 is superseded.** None enters the registry as a new rule.

Three carry a variant worth testing against the shipped threshold once Phase 6.1
can ablate: the population trigger in `2026_04_25_002`, and the retry counts in
`2026_04_25_003` and `2026_04_24_001`.

## The other lesson

5 of the 9 memories say the same thing about housing. `memory_chain` deduplicates
by **title** only, so five differently-titled notes about one problem all
survived.

`ADAPTIVE-AGENT-PLAN.md` 4.2 replaces title matching with trigger-overlap
detection for exactly this reason. These files are the evidence that title
deduplication is not enough.
