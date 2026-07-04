# Autoresearch for AoE2 Agent — Continuous Improvement Plan

**Status:** **PARTIALLY SHIPPED** — Phase 0 + Phase 1 (prompt-mutation loop with git-revert + memory chain) live in `autoresearch/`. Phases 2–5 unbuilt. Frozen historical plan; for current state see [Part 8 — Autoresearch](../part8-autoresearch/22-autoresearch-overview.md).
**Original location:** repo root `AUTORESEARCH_PLAN.md` (moved 2026-05-24).

> Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): let an LLM autonomously experiment in a tight loop — modify → evaluate → keep/revert → repeat. This plan adapts that pattern to continuously improve the AoE2 game-playing agent.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Current Agent Architecture](#2-current-agent-architecture)
3. [Autoresearch Concept](#3-autoresearch-concept-how-it-maps-to-aoe2)
4. [Bug Fixes (prerequisite)](#4-bug-fixes-prerequisite)
5. [Phase 0: Foundation (COMPLETED)](#5-phase-0-foundation-completed)
6. [Phase 1: Prompt Optimization Loop](#6-phase-1-prompt-optimization-loop)
7. [Phase 2: Context Tuning + Strategy Mining](#7-phase-2-context-tuning--strategy-mining)
8. [Phase 3: Automated Game Restart](#8-phase-3-automated-game-restart)
9. [Phase 4: Detection Active Learning](#9-phase-4-detection-active-learning)
10. [Phase 5: Training Pipeline Improvements](#10-phase-5-training-pipeline-improvements)
11. [Scoring System](#11-scoring-system)
12. [File Reference](#12-file-reference)
13. [Cost Estimates](#13-cost-estimates)

---

## 1. Background & Motivation

### The Problem

The AoE2 agent can play the game — it captures screenshots, perceives them locally (YOLO entity detection + OCR of the resource bar), sends that as text to Claude, receives actions, and executes them via pyautogui. But **it never learns from its gameplay**. Every game starts from the same system prompt with the same strategy. There is no feedback loop from game outcomes back to the agent's behavior.

### The Autoresearch Pattern (Karpathy)

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) demonstrates a powerful pattern for autonomous improvement:

1. An LLM agent has **one file** it can modify (`train.py`)
2. It proposes a change and commits it
3. It runs a **fixed-budget evaluation** (5 minutes of GPU training)
4. It measures **one clear metric** (`val_bpb` — validation bits per byte)
5. If the metric improved → keep the commit. If worse → `git reset`
6. Loop forever (~100 experiments overnight)

**Key insight**: The magic is in the constraints — one file, one metric, fixed budget, git-based accept/reject.

### How This Maps to AoE2

| Autoresearch | AoE2 Agent |
|---|---|
| `train.py` (file to modify) | `prompts/system.md` (system prompt) |
| `val_bpb` (metric) | Composite game score (survival + population + age + economy) |
| 5-min GPU training | 20-min game vs Easiest AI |
| LLM proposes code change | LLM proposes prompt change |
| `git reset` on failure | `git checkout -- prompts/system.md` |
| ~100 experiments/night | ~24 experiments/night (games are slower) |

---

## 2. Current Agent Architecture

```
Screenshot → YOLO Detection (60 classes) + resource-bar OCR → Entity + Resource Context (text) → Claude → JSON Actions → pyautogui
     ↑                                                                                          |
     └────────────────────────────── 2s loop delay ────────────────────────────────────────────┘
```

### Key Files

| File | Purpose |
|---|---|
| `gameplay_agent/game_loop.py` | Core capture→detect→think→act cycle (2-second loop) |
| `gameplay_agent/providers/claude.py` | Sends screenshot + context to Claude, parses JSON response |
| `gameplay_agent/memory.py` | Turn history, game state tracking, cumulative metrics |
| `gameplay_agent/models.py` | Pydantic models for actions and observations |
| `gameplay_agent/executor.py` | Translates actions to pyautogui calls |
| `gameplay_agent/screen.py` | Screenshot capture via mss |
| `gameplay_agent/window.py` | AoE2 window detection and focus |
| `prompts/system.md` | System prompt with game rules, hotkeys, output format |
| `detection/inference/detector.py` | YOLO26n entity detection (60 classes; current served model v9, real F1 ≈ 0.67 single-pass @1280 — the metric of record, not synthetic mAP) |

### Data Flow Per Turn

1. `capture_screenshot()` → JPEG bytes + dimensions
2. `detector.detect(screenshot)` → list of `DetectedEntity` (id, class, bbox, confidence)
3. `memory.get_context_for_llm()` → game state + recent turns as text
4. Entity context formatted as `sheep_0: sheep at (640,380) [92%]`
5. `provider.get_actions(screenshot, context, width, height)` → Claude API call
6. Response parsed via `messages.parse()` into `LLMResponse` (Pydantic model)
7. `memory.create_turn(reasoning, actions, observations)` → updates game state
8. `execute_actions(actions)` → pyautogui clicks/keypresses, returns success_count

---

## 3. Autoresearch Concept: How It Maps to AoE2

We define **four parallel improvement loops**, each with its own "file to modify", "metric to optimize", and "evaluation budget":

| Loop | What Gets Modified | Metric | Eval Time | Cadence |
|---|---|---|---|---|
| 1. Prompt Optimization | `prompts/system.md` | Composite game score | 20 min/game | Every game |
| 2. Strategy Mining | `data/strategy.db` → injected context | Win rate | 0 (piggybacks) | Every 3 games |
| 3. Context Tuning | `autoresearch/context_config.yaml` | Action success rate | 2 min/test | Between games |
| 4. Detection Learning | YOLO model weights | mAP50 + action success | 2 hrs + 3 games | Weekly |

---

## 4. Bug Fixes (prerequisite)

> Absorbed from IMPROVEMENT_PLAN.md Part 1. These are standalone bug fixes that should be addressed before or alongside autoresearch work.

> **Status: ALL DONE.** All items below have been implemented:
> - 4.1 Entity ID persistence (IoU tracking) — `_assign_persistent_ids()` in `detector.py`
> - 4.2 NMS for PyTorch — unified `_nms()` in `detect()` for all backends
> - 4.3 Window offset per-action — re-fetch in `execute_action()` instead of `execute_actions()`
> - 4.4 Debug print cleanup — replaced with `logger.debug()` calls
> - 4.5 Action verification — pre/post detection comparison in `game_loop.py`
> - Additionally: structured output via `messages.parse()` replaced custom JSON parsing in `claude.py`

### 4.1 Entity ID Persistence — IoU-Based Tracking ✅

**Severity**: HIGH
**File**: `detection/inference/detector.py`

**Problem**: `_reset_counters()` clears all entity ID counters at the start of every detection cycle. Entity IDs like `sheep_0` are regenerated from scratch each frame. The LLM targets `sheep_0` in turn N, but by turn N+1 a completely different sheep may be assigned `sheep_0`.

**Fix**: Add `_previous_detections` cache. After each detection cycle, match new detections to previous ones by IoU overlap. If IoU > 0.4, reuse the old entity ID. If no match, assign a new ID with an incrementing global counter (never reset).

```python
# New fields in EntityDetector.__init__():
self._previous_detections: list[DetectedEntity] = []
self._global_id_counter: int = 0

def _assign_persistent_ids(self, new_detections: list[DetectedEntity]) -> list[DetectedEntity]:
    """Match new detections to previous frame by IoU, preserving IDs."""
    used_prev = set()
    result = []
    for new_det in new_detections:
        best_iou, best_prev = 0.0, None
        for i, prev_det in enumerate(self._previous_detections):
            if i in used_prev or prev_det.class_name != new_det.class_name:
                continue
            iou = self._compute_iou(new_det.bbox, prev_det.bbox)
            if iou > best_iou:
                best_iou, best_prev = iou, (i, prev_det)
        if best_prev and best_iou > 0.4:
            used_prev.add(best_prev[0])
            new_det.id = best_prev[1].id
        else:
            new_det.id = f"{new_det.class_name}_{self._global_id_counter}"
            self._global_id_counter += 1
        result.append(new_det)
    self._previous_detections = result
    return result
```

Call `_assign_persistent_ids()` at the end of `detect()` instead of `_reset_counters()` at the beginning.

### 4.2 NMS Missing in PyTorch Backend ✅

**Severity**: MEDIUM
**File**: `detection/inference/detector.py`

**Problem**: `_nms()` method defined but never called for the PyTorch inference path. Only the ONNX path applies NMS. This means PyTorch detections can include duplicate overlapping boxes.

**Fix**: After the PyTorch results loop, add:
```python
entities = self._nms(entities, iou_threshold=0.5)
```

### 4.3 Window Offset Race Condition ✅

**Severity**: MEDIUM
**File**: `gameplay_agent/executor.py`

**Problem**: Window rect is fetched once at the start of action batch execution. If the game window moves during the batch, all subsequent coordinate translations are wrong.

**Fix**: Re-fetch window rect before each individual action:
```python
window_rect = self.window.get_game_window_rect()  # Fresh fetch per action
```

### 4.4 ONNX Debug Print Spam ✅

**Severity**: LOW
**File**: `detection/inference/detector.py`

**Problem**: Multiple `print("DEBUG:...")` statements left in production code.

**Fix**: Replace all with `log.debug()` using the existing structlog logger.

### 4.5 Action Verification Enhancement ✅

**Severity**: MEDIUM
**Files**: `gameplay_agent/game_loop.py`, `gameplay_agent/memory.py`

**Current state**: Phase 0 tracks `success_count` from `execute_actions()` return value. This is a basic count — it doesn't tell the LLM *what* succeeded or failed.

**Enhancement**: Capture a post-action screenshot, compare pre/post entity states, and inject verification text into the next turn's LLM context:

```python
# After execute_actions():
post_screenshot = capture_screenshot()
post_entities = detector.detect(post_screenshot) if detector else []

verification = _verify_actions(pre_entities, post_entities, actions)
memory.last_verification = verification

# In memory.get_context_for_llm():
if self.last_verification:
    parts.append(f"## Last Turn Results\n{self.last_verification}")
```

Verification text example:
```
- Sent villager_2 to gold_mine_0: SUCCESS (villager moved 45px toward gold)
- Built house (press Q): UNCERTAIN (no new house detected yet)
```

---

## 5. Phase 0: Foundation (COMPLETED)

> Status: **DONE**. All items below are implemented and tested.

### What Was Built

#### 4.1 Game State Detection (`gameplay_agent/models.py`)

Added `game_state` field to the `Observations` Pydantic model:

```python
class Observations(BaseModel):
    resources: dict[str, int] = Field(default_factory=dict)
    population: str = ""
    age: str = ""
    idle_tc: bool = False
    under_attack: bool = False
    game_state: Literal["playing", "victory", "defeat", "menu"] = "playing"  # NEW
    events: list[str] = Field(default_factory=list)
```

The LLM reports game state in every response. The game loop checks it and stops on victory/defeat.

**Design decision**: We use the LLM's reported game state rather than template matching or pixel heuristics — the executor already emits an observation (resources, population, age, events) every turn, so a victory/defeat signal rides the same channel without extra perception code. (Perception is local: YOLO entities + resource-bar OCR as text; no image is sent to the model.)

#### 4.2 Cumulative Metrics (`gameplay_agent/memory.py`)

Added to `AgentMemory.__init__()`:

```python
# Cumulative metrics for autoresearch scoring
self.total_food_gathered: int = 0      # Highest food value observed
self.peak_population: int = 0          # Highest population reached
self.total_actions: int = 0            # All actions sent to executor
self.successful_actions: int = 0       # Actions that succeeded
self.highest_age: str = "Dark Age"     # Best age advancement
self.game_start_time: datetime | None = None  # Set on first turn
self.game_end_reason: str = ""         # "victory", "defeat", "timeout", "interrupted"
```

Updated in these methods:
- `add_turn()` → starts timer, counts actions, tracks food
- `update_from_observations()` → tracks peak population, highest age
- `record_action_results(success_count, total)` → increments successful_actions
- `get_metrics_snapshot()` → returns dict of all metrics for scoring
- `reset()` → clears all counters for new game

#### 4.3 Game-Over Detection + Time Budget (`gameplay_agent/game_loop.py`)

The `game_loop()` function was updated:

```python
async def game_loop(
    provider: BaseLLMProvider,
    max_iterations: int | None = None,
    memory: AgentMemory | None = None,
    use_detection: bool = True,
    time_budget: float | None = None,    # NEW: seconds limit
) -> AgentMemory:                        # NEW: returns memory with metrics
```

After each LLM response, two new checks:

```python
# 5b. Check for game-over via LLM observations
game_state = observations.get("game_state", "playing")
if game_state in ("victory", "defeat"):
    memory.game_end_reason = game_state
    break

# 5c. Check time budget
if time_budget and memory.get_game_duration_seconds() >= time_budget:
    memory.game_end_reason = "timeout"
    break
```

Action success is tracked after execution:

```python
if actions:
    success_count = await execute_actions(actions)
    memory.record_action_results(success_count, len(actions))
```

On exit (including errors/interrupts), final metrics are logged and memory is returned.

#### 4.4 Composite Scoring (`autoresearch/metrics.py`)

```python
@dataclass
class GameScore:
    composite: float      # 0.0 - 1.0 overall score
    survival: float       # component: time survived
    population: float     # component: peak pop
    age: float           # component: age advancement
    economy: float       # component: food gathered
    action_success: float # component: action success rate
    raw_metrics: dict    # original metrics snapshot

def compute_score(metrics: dict) -> GameScore:
    """Converts AgentMemory.get_metrics_snapshot() into a GameScore."""
```

**Weights** (must sum to 1.0):
| Component | Weight | Normalization Cap |
|---|---|---|
| Survival time | 0.30 | 1200 seconds (20 min) |
| Peak population | 0.25 | 50 villagers |
| Age advancement | 0.20 | Dark=0, Feudal=0.33, Castle=0.66, Imperial=1.0 |
| Economy (food) | 0.15 | 5000 food gathered |
| Action success rate | 0.10 | success_count / total_actions |

#### 4.5 Experiment Ledger (`autoresearch/experiment_log.py`)

TSV file at `experiments/results.tsv` tracking all experiments:

```
experiment_id  timestamp                loop    change_description  composite_score  survival  population  age  economy  action_success  game_end_reason  turn_count  accepted  git_sha
exp_0001       2026-03-15T22:00:00+00:00  manual  baseline          0.4500           0.8000    0.3000      0.0  0.2000   0.5000          timeout          450         true      abc1234
```

**Key functions**:
- `log_experiment(experiment_id, loop, description, score, accepted, git_sha)` → appends row
- `get_recent_experiments(n=5)` → reads last N experiments as list of dicts
- `get_best_score(loop=None)` → best composite score from accepted experiments
- `get_next_experiment_id()` → auto-increments `exp_NNNN`
- `get_git_sha()` → current short SHA

#### 4.6 Game Runner (`autoresearch/game_runner.py`)

CLI wrapper that runs a game and logs results:

```bash
# Run a 20-minute game with metrics collection
python -m autoresearch.game_runner --time-budget 1200 --description "baseline"

# Run with turn limit instead
python -m autoresearch.game_runner --max-iterations 500

# Specify experiment ID
python -m autoresearch.game_runner --experiment-id exp_0001 --description "added sheep priority"
```

**Key functions**:
- `run_game(time_budget, max_iterations, use_detection)` → runs game, returns `{metrics, score}`
- `run_and_log(experiment_id, loop, description, ...)` → runs game + logs to TSV

#### 4.7 System Prompt Update (`prompts/system.md`)

Added `game_state` to the output format example and a new section:

```markdown
## Game State Detection
Set `game_state` in observations:
- `"playing"` — normal gameplay (default)
- `"victory"` — you see a victory screen or "You are victorious" message
- `"defeat"` — you see a defeat screen or "You have been defeated" message
- `"menu"` — you see the main menu, loading screen, or lobby (not in a game)
```

#### 4.8 Configuration (`autoresearch/config.yaml`)

```yaml
game:
  time_budget: 1200        # seconds per game (20 min)
  max_iterations: null     # turn limit (null = use time_budget only)

prompt_loop:
  enabled: true
  epsilon: 0.02            # accept if score >= best - epsilon
  max_line_changes: 5
  mutator_model: "claude-haiku-4-5-20251001"

scoring:
  survival_weight: 0.30
  population_weight: 0.25
  age_weight: 0.20
  economy_weight: 0.15
  action_success_weight: 0.10
```

### Verification (Phase 0)

Run this to verify everything works:

```bash
python -c "
from gameplay_agent.models import Observations
from gameplay_agent.memory import AgentMemory
from autoresearch.metrics import compute_score
from autoresearch.experiment_log import get_next_experiment_id

# Test game_state field
obs = Observations(game_state='victory')
assert obs.game_state == 'victory'

# Test cumulative metrics
mem = AgentMemory()
mem.create_turn(reasoning='test', actions=[{'type': 'press', 'key': 'h'}],
    observations={'population': '5/10', 'age': 'Feudal Age', 'resources': {'food': 300}})
snapshot = mem.get_metrics_snapshot()
assert snapshot['peak_population'] == 5
assert snapshot['highest_age'] == 'Feudal Age'

# Test scoring
score = compute_score(snapshot)
assert 0 <= score.composite <= 1

print('Phase 0 OK')
"
```

---

## 6. Phase 1: Prompt Optimization Loop

> Status: **NOT STARTED**. This is the next phase to implement.

### Overview

This is the direct autoresearch analog. An LLM proposes changes to the system prompt, a game is played, and the change is accepted or reverted based on the composite score.

### 5.1 Create `autoresearch/prompt_mutator.py`

**Purpose**: Given the current prompt and experiment history, propose a targeted change.

**Implementation details**:

```python
import anthropic
from pathlib import Path

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "system.md"

# Sections the mutator must NOT modify (output format, game state detection)
PROTECTED_SECTIONS = ["## Output Format", "## Game State Detection"]


class PromptMutator:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def read_current_prompt(self) -> str:
        return PROMPT_FILE.read_text()

    def propose_change(
        self,
        current_prompt: str,
        recent_experiments: list[dict],
        failure_modes: list[str],
    ) -> dict:
        """Ask LLM to propose a prompt modification.

        Args:
            current_prompt: Full text of prompts/system.md
            recent_experiments: Last 5 experiments from experiment_log
            failure_modes: Specific failures from most recent game (e.g.,
                "agent got population-capped 3 times",
                "agent never advanced to Feudal Age")

        Returns:
            {
                "description": "Added sheep-gathering priority to Dark Age",
                "old_text": "existing text to replace",
                "new_text": "replacement text",
                "rationale": "why this should improve the score"
            }
        """
        # Build context for the mutator LLM
        experiment_summary = self._format_experiments(recent_experiments)
        failure_summary = "\n".join(f"- {f}" for f in failure_modes) if failure_modes else "None identified"

        system = """You are an expert AoE2 strategist optimizing a system prompt for an AI agent.
Your goal: propose a SMALL, targeted change to the prompt that will improve the agent's game score.

Rules:
- Change at most 5 lines
- Do NOT modify the "## Output Format" or "## Game State Detection" sections
- Focus on strategy, priorities, decision-making heuristics
- Be specific (e.g., "always build 2 houses before advancing" not "build more houses")
- Return JSON with: description, old_text (exact text to replace), new_text (replacement), rationale"""

        user = f"""Current prompt:
```
{current_prompt}
```

Recent experiment results:
{experiment_summary}

Known failure modes from recent games:
{failure_summary}

Propose ONE targeted change to improve the agent's performance."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Parse JSON from response
        # ... (extract JSON from response.content[0].text)

    def apply_change(self, old_text: str, new_text: str) -> bool:
        """Apply the proposed change to prompts/system.md.

        Returns True if the change was applied successfully.
        Validates that old_text exists in the prompt and that
        protected sections are not modified.
        """
        current = self.read_current_prompt()
        if old_text not in current:
            return False

        modified = current.replace(old_text, new_text, 1)

        # Verify protected sections unchanged
        for section in PROTECTED_SECTIONS:
            if section in current:
                # Extract section content and verify it's unchanged
                pass

        PROMPT_FILE.write_text(modified)
        return True

    def revert(self) -> None:
        """Revert prompt to last git-committed version."""
        import subprocess
        subprocess.run(
            ["git", "checkout", "--", str(PROMPT_FILE)],
            cwd=PROMPT_FILE.parent.parent,
        )

    def _format_experiments(self, experiments: list[dict]) -> str:
        lines = []
        for exp in experiments:
            status = "KEPT" if exp.get("accepted") == "true" else "REVERTED"
            lines.append(
                f"  {exp.get('experiment_id')}: score={exp.get('composite_score')} "
                f"[{status}] — {exp.get('change_description')}"
            )
        return "\n".join(lines) or "No previous experiments"
```

**Key design decisions**:
- Uses Haiku (cheap) for mutations, not Sonnet — the mutator doesn't need vision
- Protected sections prevent the mutator from breaking the output format
- `old_text`/`new_text` approach ensures targeted changes (not full rewrites)
- `revert()` uses `git checkout` to undo changes cleanly

### 5.2 Create `autoresearch/orchestrator.py`

**Purpose**: Main loop that coordinates prompt mutation, game running, and accept/reject decisions.

**Implementation details**:

```python
import subprocess
import time
from pathlib import Path

from .experiment_log import (
    get_best_score, get_next_experiment_id, get_recent_experiments, log_experiment
)
from .game_runner import run_game
from .metrics import compute_score
from .prompt_mutator import PromptMutator

REPO_ROOT = Path(__file__).parent.parent
EPSILON = 0.02  # Accept if score >= best - epsilon


class Orchestrator:
    def __init__(self):
        self.mutator = PromptMutator()
        self.best_score = get_best_score(loop="prompt")

    def git_commit(self, message: str) -> str:
        """Commit current changes and return short SHA."""
        subprocess.run(["git", "add", "prompts/system.md"], cwd=REPO_ROOT)
        subprocess.run(["git", "commit", "-m", message], cwd=REPO_ROOT)
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return result.stdout.strip()

    def git_revert_prompt(self) -> None:
        """Revert prompts/system.md to previous commit."""
        subprocess.run(
            ["git", "checkout", "HEAD~1", "--", "prompts/system.md"],
            cwd=REPO_ROOT,
        )
        subprocess.run(
            ["git", "commit", "-m", "[autoresearch] revert: prompt change rejected"],
            cwd=REPO_ROOT,
        )

    async def run_experiment(self, time_budget: float = 1200) -> dict:
        """Run one full experiment cycle: mutate → play → score → accept/reject.

        Returns dict with experiment_id, score, accepted, description.
        """
        experiment_id = get_next_experiment_id()
        recent = get_recent_experiments(5)

        # 1. Propose a prompt change
        current_prompt = self.mutator.read_current_prompt()
        # Extract failure modes from most recent game's low-scoring components
        failure_modes = self._extract_failure_modes(recent)

        change = self.mutator.propose_change(current_prompt, recent, failure_modes)
        description = change["description"]

        # 2. Apply the change
        success = self.mutator.apply_change(change["old_text"], change["new_text"])
        if not success:
            # Change couldn't be applied (old_text not found)
            return {"experiment_id": experiment_id, "error": "change_not_applicable"}

        # 3. Commit the change
        sha = self.git_commit(f"[autoresearch] {experiment_id}: {description}")

        # 4. Run the game
        result = await run_game(time_budget=time_budget)
        score = result["score"]

        # 5. Accept or reject
        accepted = score.composite >= self.best_score - EPSILON

        if accepted:
            self.best_score = max(self.best_score, score.composite)
        else:
            self.git_revert_prompt()

        # 6. Log result
        log_experiment(
            experiment_id=experiment_id,
            loop="prompt",
            change_description=description,
            score=score,
            accepted=accepted,
            git_sha=sha if accepted else None,
        )

        return {
            "experiment_id": experiment_id,
            "score": score.composite,
            "accepted": accepted,
            "description": description,
        }

    async def run_loop(self, max_experiments: int | None = None, time_budget: float = 1200):
        """Run the autonomous experiment loop.

        Human must start each game manually (Phase 1).
        Orchestrator mutates prompt between games.

        Args:
            max_experiments: Stop after N experiments (None = run forever)
            time_budget: Seconds per game
        """
        count = 0
        while max_experiments is None or count < max_experiments:
            print(f"\n{'='*60}")
            print(f"Experiment {count + 1} — Best score: {self.best_score:.4f}")
            print(f"{'='*60}")

            # Wait for human to start game
            print("Start a new game in AoE2, then press Enter...")
            input()

            result = await self.run_experiment(time_budget=time_budget)

            if "error" in result:
                print(f"Error: {result['error']}")
                continue

            status = "ACCEPTED" if result["accepted"] else "REJECTED"
            print(f"\n{status}: {result['description']}")
            print(f"Score: {result['score']:.4f}")

            count += 1

    def _extract_failure_modes(self, recent: list[dict]) -> list[str]:
        """Identify failure patterns from recent experiments."""
        modes = []
        if not recent:
            return modes

        latest = recent[-1]
        if float(latest.get("population", 0)) < 0.2:
            modes.append("Population stayed very low — agent may not be queueing villagers")
        if float(latest.get("age", 0)) == 0:
            modes.append("Agent never advanced past Dark Age")
        if float(latest.get("economy", 0)) < 0.1:
            modes.append("Very little food gathered — agent may not be assigning villagers to food")
        if float(latest.get("action_success", 0)) < 0.3:
            modes.append("Low action success rate — many actions may be failing")
        return modes
```

**Usage**:

```bash
# Run the orchestrator (human starts each game manually)
python -c "
import asyncio
from autoresearch.orchestrator import Orchestrator
asyncio.run(Orchestrator().run_loop(max_experiments=5, time_budget=1200))
"
```

### 5.3 Git Branching Strategy

All experiments run on a dedicated branch:

```bash
# Before first run
git checkout -b autoresearch/prompt-optimization

# Each experiment:
# 1. mutator writes change to prompts/system.md
# 2. git commit -m "[autoresearch] exp_0001: Added sheep-gathering priority"
# 3. Game plays...
# 4a. If accepted: commit stays, branch advances
# 4b. If rejected: git checkout HEAD~1 -- prompts/system.md + commit revert

# After N successful experiments, merge to main
git checkout main
git merge autoresearch/prompt-optimization
```

### 5.4 Acceptance Criteria (Phase 1)

- [ ] `prompt_mutator.py` can propose, apply, and revert prompt changes
- [ ] `orchestrator.py` runs the full experiment cycle end-to-end
- [ ] After 5 manual experiments, `experiments/results.tsv` has 5 entries with valid scores
- [ ] At least 1 experiment shows an accepted improvement over baseline
- [ ] Git log shows proper commit/revert history

---

## 7. Phase 2: Context Tuning + Strategy Mining

> Status: **NOT STARTED**.

### 6.1 Context Tuning Loop

**Purpose**: A/B test which context configuration produces the best action success rate.

#### Create `autoresearch/context_config.yaml`

```yaml
# Parameters to tune via A/B testing
max_entities: 15              # How many detected entities to pass to LLM
working_memory_turns: 3       # How many recent turns to include
entity_sort_order: "confidence"  # "confidence" | "distance_to_center" | "class_priority"
include_dynamic_context: true # Whether to inject game knowledge DB context
```

#### Create `autoresearch/context_tuner.py`

```python
class ContextTuner:
    """A/B tests context configuration parameters."""

    PARAMETERS = {
        "max_entities": [10, 15, 20, 25],
        "working_memory_turns": [2, 3, 5],
        "entity_sort_order": ["confidence", "distance_to_center", "class_priority"],
    }

    def generate_variant(self, current_config: dict) -> dict:
        """Change one parameter at a time from current config."""
        # Pick a random parameter, pick a random value != current
        ...

    async def run_ab_test(self, config_a: dict, config_b: dict, turns: int = 50) -> dict:
        """Run 50 turns with config_a, then 50 with config_b. Compare action success rate."""
        ...
```

#### Modify `gameplay_agent/game_loop.py` — Read Context Config

In the entity context building section (lines 121-129), make the entity limit configurable:

```python
# Current (hardcoded):
for entity in detected_entities[:15]:

# New (from config):
from autoresearch.context_config import get_context_config
ctx_config = get_context_config()
max_entities = ctx_config.get("max_entities", 15)
sort_order = ctx_config.get("entity_sort_order", "confidence")

# Sort entities based on configured order
if sort_order == "confidence":
    sorted_entities = sorted(detected_entities, key=lambda e: e.confidence, reverse=True)
elif sort_order == "distance_to_center":
    cx, cy = width // 2, height // 2
    sorted_entities = sorted(detected_entities, key=lambda e: abs(e.center[0]-cx) + abs(e.center[1]-cy))
elif sort_order == "class_priority":
    PRIORITY = {"town_center": 0, "villager": 1, "sheep": 2, ...}
    sorted_entities = sorted(detected_entities, key=lambda e: PRIORITY.get(e.class_name, 99))

for entity in sorted_entities[:max_entities]:
    ...
```

Also make working memory depth configurable in `memory.get_context_for_llm()`:

```python
# Current (hardcoded):
recent_turns = list(self.working_memory)[-3:]

# New (from config):
memory_depth = ctx_config.get("working_memory_turns", 3)
recent_turns = list(self.working_memory)[-memory_depth:]
```

### 6.2 Strategy Mining Loop

**Purpose**: Learn which action patterns correlate with good game outcomes, and inject those patterns into the LLM context.

#### Create `gameplay_agent/strategy_db.py`

```python
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "strategy.db"


class StrategyDB:
    """SQLite database for game recordings and mined strategy patterns."""

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                timestamp TEXT,
                composite_score REAL,
                end_reason TEXT,   -- victory/defeat/timeout
                turn_count INTEGER,
                prompt_sha TEXT    -- which prompt version was used
            );

            CREATE TABLE IF NOT EXISTS turns (
                game_id TEXT,
                turn_number INTEGER,
                timestamp TEXT,
                reasoning TEXT,
                actions TEXT,       -- JSON array
                resources TEXT,     -- JSON dict
                population INTEGER,
                age TEXT,
                game_state TEXT,    -- playing/victory/defeat
                PRIMARY KEY (game_id, turn_number),
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );

            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT,  -- human-readable pattern
                condition TEXT,    -- when to apply (e.g., "Dark Age, first 5 minutes")
                action TEXT,       -- what to do (e.g., "queue villagers continuously")
                success_rate REAL, -- win rate when pattern is followed
                sample_count INTEGER,
                confidence TEXT,   -- low/medium/high
                created_at TEXT,
                last_updated TEXT
            );
        """)

    def log_turn(self, game_id: str, turn_number: int, reasoning: str,
                 actions: list, resources: dict, population: int, age: str):
        """Log a single turn's data."""
        import json
        self.conn.execute(
            "INSERT OR REPLACE INTO turns VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, 'playing')",
            (game_id, turn_number, reasoning, json.dumps(actions),
             json.dumps(resources), population, age)
        )
        self.conn.commit()

    def log_game(self, game_id: str, score: float, end_reason: str,
                 turn_count: int, prompt_sha: str):
        """Log a completed game."""
        self.conn.execute(
            "INSERT OR REPLACE INTO games VALUES (?, datetime('now'), ?, ?, ?, ?)",
            (game_id, score, end_reason, turn_count, prompt_sha)
        )
        self.conn.commit()

    def get_proven_patterns(self, min_confidence: str = "medium") -> list[dict]:
        """Get patterns with sufficient confidence for injection into LLM context."""
        conf_order = {"low": 0, "medium": 1, "high": 2}
        min_level = conf_order.get(min_confidence, 1)

        rows = self.conn.execute(
            "SELECT description, condition, action, success_rate, confidence "
            "FROM patterns WHERE sample_count >= 3 ORDER BY success_rate DESC"
        ).fetchall()

        return [
            {"description": r[0], "condition": r[1], "action": r[2],
             "success_rate": r[3], "confidence": r[4]}
            for r in rows
            if conf_order.get(r[4], 0) >= min_level
        ]
```

#### Create `autoresearch/strategy_analyzer.py`

```python
class StrategyAnalyzer:
    """Analyzes game recordings to extract winning strategy patterns."""

    def __init__(self):
        self.db = StrategyDB()
        self.client = anthropic.Anthropic()

    def analyze_recent_games(self, n: int = 3) -> list[dict]:
        """Compare the last N games and extract strategy patterns.

        Sends turn-by-turn data from wins vs losses to an LLM,
        asks it to identify what the winning games did differently.
        """
        # Fetch last N games with their turns
        # Build comparison prompt
        # Ask LLM to identify patterns
        # Store patterns in strategy.db
        ...
```

#### Modify `gameplay_agent/game_loop.py` — Per-Turn Logging

After `memory.create_turn()`, add:

```python
# Log turn to strategy DB (if available)
if strategy_db:
    strategy_db.log_turn(
        game_id=game_id,
        turn_number=iteration,
        reasoning=reasoning,
        actions=actions,
        resources=observations.get("resources", {}),
        population=memory.game_state.population,
        age=memory.game_state.current_age,
    )
```

#### Modify `gameplay_agent/providers/claude.py` — Inject Strategy Patterns

In `_get_dynamic_context()` or a new method, inject proven patterns:

```python
def _get_strategy_context(self) -> str:
    """Inject proven strategy patterns from strategy DB."""
    if not self._strategy_db:
        return ""

    patterns = self._strategy_db.get_proven_patterns(min_confidence="medium")
    if not patterns:
        return ""

    lines = ["## Proven Strategy Patterns"]
    for p in patterns[:5]:  # Limit to top 5
        lines.append(f"- When {p['condition']}: {p['action']} (success rate: {p['success_rate']:.0%})")
    return "\n".join(lines)
```

### 6.3 Acceptance Criteria (Phase 2)

- [ ] `context_config.yaml` is loaded and affects entity sorting + memory depth
- [ ] A/B test runner can compare two configs on 50-turn segments
- [ ] `strategy.db` has tables for games, turns, and patterns
- [ ] Per-turn logging populates the turns table during gameplay
- [ ] After 3+ games, strategy analyzer produces at least 1 pattern
- [ ] Proven patterns appear in LLM context during games

---

## 8. Phase 3: Automated Game Restart

> Status: **NOT STARTED**. Enables true overnight autonomy.

### 7.1 Research AoE2:DE Menu Hotkeys

Before implementation, research which menu transitions can be done via keyboard:
- `Enter` — confirm dialogs, start game
- `Escape` — go back, cancel
- Arrow keys — navigate menu items
- Tab — cycle between fields

Document which transitions REQUIRE mouse clicks (there will likely be some).

### 7.2 Create `gameplay_agent/menu_navigator.py`

```python
import pyautogui
import cv2
import numpy as np
from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent.parent / "autoresearch" / "templates"


class MenuNavigator:
    """Navigate AoE2:DE menus using hotkeys + template matching."""

    def find_button(self, screenshot: np.ndarray, template_name: str) -> tuple[int, int] | None:
        """Find a button on screen using template matching.

        Args:
            screenshot: Current screen as numpy array
            template_name: Name of template file (e.g., "start_game_button.png")

        Returns:
            (x, y) center of matched button, or None if not found
        """
        template_path = TEMPLATES_DIR / template_name
        if not template_path.exists():
            return None

        template = cv2.imread(str(template_path))
        result = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        if max_val > 0.8:  # Confidence threshold
            h, w = template.shape[:2]
            return (max_loc[0] + w // 2, max_loc[1] + h // 2)
        return None

    def start_standard_game(self, difficulty: str = "easiest", map_type: str = "arabia"):
        """Navigate from main menu to starting a Standard Game.

        Sequence (mix of hotkeys and template-matched clicks):
        1. Click "Single Player" button
        2. Click "Standard Game" button
        3. Set difficulty (dropdown or arrows)
        4. Set map type
        5. Click "Start Game" / press Enter
        6. Wait for loading screen to finish
        """
        ...

    def handle_game_over(self):
        """After game ends, navigate back to menu.

        Sequence:
        1. Detect victory/defeat screen
        2. Press Enter or click "Continue" to dismiss
        3. Wait for stats screen
        4. Press Escape or click "Exit" to return to menu
        """
        ...

    def wait_for_game_load(self, timeout: int = 60):
        """Wait until the game is fully loaded (HUD visible)."""
        ...
```

### 7.3 Capture Template Images

Manually capture reference images for buttons:

```
autoresearch/templates/
  single_player_button.png
  standard_game_button.png
  start_game_button.png
  continue_button.png     # Victory/defeat screen
  exit_button.png          # Stats screen
```

**Capture process**: Take a screenshot of each button at the game's native resolution, crop tightly around the button.

### 7.4 Modify Orchestrator for Auto-Restart

```python
# In orchestrator.py run_loop():
async def run_loop_autonomous(self, max_experiments: int, time_budget: float = 1200):
    """Fully autonomous loop — no human intervention needed."""
    navigator = MenuNavigator()

    for i in range(max_experiments):
        # 1. Mutate prompt
        ...

        # 2. Start a new game
        navigator.start_standard_game(difficulty="easiest")
        navigator.wait_for_game_load()

        # 3. Run game
        result = await run_game(time_budget=time_budget)

        # 4. Handle game over
        navigator.handle_game_over()

        # 5. Accept/reject
        ...
```

### 7.5 Acceptance Criteria (Phase 3)

- [ ] `menu_navigator.py` can reliably start a Standard Game from main menu
- [ ] Template matching finds buttons at > 80% reliability
- [ ] Game-over → menu → new game cycle works end-to-end
- [ ] Orchestrator runs 3+ games without human intervention
- [ ] Watchdog detects game crashes and recovers

---

## 9. Phase 4: Detection Active Learning

> Status: **NOT STARTED**. Semi-automated, weekly cadence.

### 9.1 Error Capture During Gameplay

#### Create `gameplay_agent/error_capture.py`

Three dedicated capture methods, each saving both screenshot and structured metadata:

```python
from dataclasses import dataclass, asdict
from pathlib import Path
import json, time

CAPTURE_DIR = Path(__file__).parent.parent / "detection" / "error_captures"

@dataclass
class CapturedError:
    timestamp: float
    error_type: str          # "detection_miss", "action_failed", "low_confidence"
    screenshot_path: str
    action_attempted: dict
    detection_state: list    # Entities detected at the time
    game_state: dict         # Resources, pop, age
    confidence_scores: list
    notes: str

class ErrorCapture:
    """Captures problematic screenshots during gameplay for active learning."""

    def __init__(self):
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        self.errors: list[CapturedError] = []

    def capture_detection_miss(self, screenshot_bytes, entities, action, game_state):
        """Action targeted a detected entity but nothing happened."""
        ts = time.time()
        path = CAPTURE_DIR / f"det_miss_{ts:.0f}.jpg"
        path.write_bytes(screenshot_bytes)
        self._save(CapturedError(
            timestamp=ts, error_type="detection_miss", screenshot_path=str(path),
            action_attempted=action, detection_state=[vars(e) for e in entities],
            game_state=game_state, confidence_scores=[e.confidence for e in entities],
            notes=f"Action on {action.get('target_id')} had no effect"
        ))

    def capture_low_confidence(self, screenshot_bytes, entities, threshold=0.25):
        """Auto-save when any detection has low confidence."""
        low = [e for e in entities if e.confidence < threshold and e.confidence > 0.05]
        if not low:
            return
        ts = time.time()
        path = CAPTURE_DIR / f"low_conf_{ts:.0f}.jpg"
        path.write_bytes(screenshot_bytes)
        # Also save YOLO-format labels for prelabeling
        label_path = path.with_suffix(".txt")
        with open(label_path, "w") as f:
            for e in entities:
                f.write(f"{e.class_id} {e.x_center} {e.y_center} {e.width} {e.height}\n")
        self._save(CapturedError(
            timestamp=ts, error_type="low_confidence", screenshot_path=str(path),
            action_attempted={}, detection_state=[], game_state={},
            confidence_scores=[e.confidence for e in low],
            notes=f"Low confidence: {[f'{e.class_name}={e.confidence:.2f}' for e in low]}"
        ))

    def capture_action_failure(self, screenshot_bytes, action, entities, game_state):
        """Action execution returned 0 successes."""
        ts = time.time()
        path = CAPTURE_DIR / f"action_fail_{ts:.0f}.jpg"
        path.write_bytes(screenshot_bytes)
        self._save(CapturedError(
            timestamp=ts, error_type="action_failed", screenshot_path=str(path),
            action_attempted=action, detection_state=[], game_state=game_state,
            confidence_scores=[], notes="No observable change after action execution"
        ))

    def _save(self, error: CapturedError):
        self.errors.append(error)
        meta = CAPTURE_DIR / f"error_{error.timestamp:.0f}.json"
        with open(meta, "w") as f:
            json.dump(asdict(error), f, indent=2)

    def get_capture_count(self) -> int:
        return len(list(CAPTURE_DIR.glob("*.jpg")))
```

### 9.2 Integrate into Game Loop

In `gameplay_agent/game_loop.py`, after action execution:

```python
# After actions are executed:
if error_capture:
    error_capture.check_and_capture(
        screenshot_bytes=screenshot,
        detected_entities=detected_entities,
        actions=actions,
        action_success_count=success_count,
    )
```

### 9.3 Detection Retrain Trigger

#### Create `autoresearch/detection_loop.py`

```python
class DetectionLoop:
    """Manages the error-capture-to-retrain cycle."""

    CAPTURE_THRESHOLD = 50  # Trigger retrain after this many captures

    def should_trigger(self) -> bool:
        """Check if enough error captures have accumulated."""
        capture_dir = Path("detection/error_captures")
        if not capture_dir.exists():
            return False
        return len(list(capture_dir.glob("*.jpg"))) >= self.CAPTURE_THRESHOLD

    def prepare_for_labeling(self):
        """Pre-label error captures with current model for CVAT review."""
        # Run detection/labeling/prelabel.py on error_captures/
        ...

    def trigger_retrain(self):
        """Merge new labels into training data and retrain YOLO."""
        # 1. Convert CVAT exports to YOLO format
        # 2. Merge with existing training data
        # 3. Run detection/training/train_yolo.py
        # 4. Compare new model mAP50 with current
        # 5. If improved, deploy to detection/inference/models/
        ...
```

### 9.4 Acceptance Criteria (Phase 4)

- [ ] `error_capture.py` saves screenshots meeting capture conditions
- [ ] After 50+ captures, `detection_loop.py` triggers the retrain workflow
- [ ] Pre-labeling works on captured screenshots
- [ ] Retrained model is compared against current model
- [ ] New model is deployed only if mAP50 improves

---

## 10. Phase 5: Training Pipeline Improvements

> Absorbed from IMPROVEMENT_PLAN.md Part 2. These improve the YOLO detection model quality independent of the autoresearch loops.

### 10.1 Missing Sprite Extractions

**Files**: `detection/extraction/extract_sprites.py`, `detection/training/config/classes.yaml`

7 classes defined in `classes.yaml` (60 total) have zero synthetic training data:

| Class | ID | Action |
|---|---|---|
| farm | 16 | Skip — flat terrain overlay, rely on real screenshots only |
| krepost | 29 | Search for `b_*_krepost_*_x1.sld`, add to SPRITE_CONFIGS |
| galley | 56 | Search for `u_ship_galley_*_x1.sld`, add with z_order=3 |
| fire_galley | 57 | Search for `u_ship_fire_galley_*_x1.sld`, add similarly |
| siege_tower | 58 | Search for `u_siege_tower_*_x1.sld`, add with z_order=3 |
| goose | 59 | Search for animal goose SLDs, add with z_order=0 |

### 10.2 Synthetic Data Quality

**File**: `detection/training/generate_training_data.py`

**a) Realistic Fog of War**: Current implementation uses random semi-transparent black patches. Real AoE2 has gradient fog from edges. Replace with edge-based gradient fog using PIL alpha compositing.

**b) Unit Clustering**: Currently places 0-3 scattered individuals. Real games have military formations (5-20 units close together) and villager clusters around resources. Add `cluster_mode=True` to military unit SPRITE_CONFIGS with configurable cluster sizes.

**c) Externalize SPRITE_CONFIGS**: Move 250+ lines of hardcoded Python dicts to `detection/training/config/sprite_configs.yaml`. Allows tuning without code changes.

**d) Multiprocessing**: Use `multiprocessing.Pool` for image generation. Currently single-threaded (~30 min for 10k images).

### 10.3 Training Hyperparameters

**File**: `detection/training/train_yolo.py`

Add missing hyperparameters:
```python
lr0=0.01, lrf=0.01, warmup_epochs=3.0, warmup_momentum=0.8,
weight_decay=0.0005, cos_lr=True, box=7.5, cls=0.5
```

Consider training at `imgsz=1280` on A100 for better small-entity detection (current: 640 with 2x downscale of 1280x720 game images).

### 10.4 Active Learning: Class-Diverse Batch Selection

**File**: `detection/labeling/active_learning.py`

Current batch selection sorts by uncertainty score and takes top-N. This can select 20 images all containing only villagers.

Fix: Stratified selection ensuring each batch covers underrepresented classes:
```python
def prepare_diverse_batch(self, scored_images, batch_size, detections_by_image):
    for img_path, score in scored:
        classes_in_image = {d.class_name for d in detections_by_image[img_path]}
        rarity_bonus = sum(1.0 / (class_counts[c] + 1) for c in classes_in_image)
        adjusted_score = score + rarity_bonus * 5
    ...
```

### 10.5 Acceptance Criteria (Phase 5)

- [ ] Missing sprites extracted and added to SPRITE_CONFIGS
- [ ] Fog of war uses gradient edges instead of random patches
- [ ] Military units placed in clusters in synthetic data
- [ ] SPRITE_CONFIGS moved to YAML
- [ ] Generation parallelized with multiprocessing
- [ ] mAP50 improves after retraining with these changes

---

## 11. Scoring System

### Composite Score Formula

```
score = (
  0.30 * min(survival_time / 1200, 1.0)           # 20 min cap
  0.25 * min(peak_population / 50, 1.0)            # 50 pop cap
  0.20 * age_score                                  # 0.0 / 0.33 / 0.66 / 1.0
  0.15 * min(total_food_gathered / 5000, 1.0)      # 5000 food cap
  0.10 * action_success_rate                        # successes / total
)
```

### Score Interpretation

| Score | Meaning |
|---|---|
| 0.00 - 0.10 | Agent barely functional (crashes, no actions) |
| 0.10 - 0.25 | Agent acts but ineffectively (random clicks) |
| 0.25 - 0.40 | Agent performs basic tasks (some villager production) |
| 0.40 - 0.60 | Competent Dark Age play (villagers + houses + gathering) |
| 0.60 - 0.80 | Advances ages, builds economy |
| 0.80 - 1.00 | Full game competency |

### Accept/Reject Threshold

```
accepted = (score >= best_score - epsilon)
```

Where `epsilon = 0.02`. This means:
- A change that improves score by any amount is accepted
- A change that makes score up to 2% worse is ALSO accepted (noise tolerance)
- A change that makes score > 2% worse is rejected

---

## 12. File Reference

### Existing Files (Modified in Phase 0)

| File | Line(s) | What Changed |
|---|---|---|
| `gameplay_agent/models.py:160` | Added `game_state: Literal[...]` to `Observations` |
| `gameplay_agent/memory.py:36-66` | Added `AGE_SCORES` dict and cumulative metrics to `AgentMemory` |
| `gameplay_agent/memory.py:68-86` | Updated `add_turn()` with timer, action count, food tracking |
| `gameplay_agent/memory.py:97-115` | Updated `update_from_observations()` with peak pop, highest age |
| `gameplay_agent/memory.py:171-200` | Added `record_action_results()`, `get_game_duration_seconds()`, `get_metrics_snapshot()` |
| `gameplay_agent/memory.py:202-214` | Updated `reset()` to clear cumulative metrics |
| `gameplay_agent/game_loop.py:28-47` | Added `time_budget` param, changed return type to `AgentMemory` |
| `gameplay_agent/game_loop.py:151-162` | Added game-over detection + time budget checks |
| `gameplay_agent/game_loop.py:165-167` | Added `memory.record_action_results()` call |
| `gameplay_agent/game_loop.py:183-196` | Added error handling for game_end_reason + final metrics log |
| `prompts/system.md:48,64-70` | Added `game_state` field + Game State Detection section |

### New Files (Created in Phase 0)

| File | Purpose |
|---|---|
| `autoresearch/__init__.py` | Package init |
| `autoresearch/metrics.py` | `GameScore` dataclass + `compute_score()` function |
| `autoresearch/experiment_log.py` | TSV ledger management (`log_experiment`, `get_recent_experiments`, `get_best_score`) |
| `autoresearch/game_runner.py` | CLI game runner (`run_game`, `run_and_log`, `main`) |
| `autoresearch/config.yaml` | Global configuration (time budget, scoring weights, loop settings) |
| `experiments/results.tsv` | Experiment ledger (TSV, auto-created with header) |

### Files to Modify (Bug Fixes)

| File | Fix | Severity |
|---|---|---|
| `detection/inference/detector.py` | ✅ Entity ID persistence (IoU tracking) + NMS for all backends + debug print cleanup | HIGH/MED/LOW |
| `gameplay_agent/executor.py` | ✅ Re-fetch window rect per action | MEDIUM |
| `gameplay_agent/game_loop.py` | ✅ Post-action screenshot verification | MEDIUM |
| `gameplay_agent/memory.py` | ✅ Add `last_verification` field | MEDIUM |
| `gameplay_agent/providers/claude.py` | ✅ Structured output via `messages.parse()` (replaced custom JSON parsing) | HIGH |

### Files to Create (Future Phases)

| File | Phase | Purpose |
|---|---|---|
| `autoresearch/prompt_mutator.py` | 1 | LLM-driven prompt modification |
| `autoresearch/orchestrator.py` | 1 | Main experiment loop with git integration |
| `autoresearch/context_tuner.py` | 2 | A/B testing context parameters |
| `autoresearch/context_config.yaml` | 2 | Tunable context parameters |
| `autoresearch/strategy_analyzer.py` | 2 | Post-game strategy pattern extraction |
| `gameplay_agent/strategy_db.py` | 2 | SQLite DB for game recordings + patterns |
| `gameplay_agent/menu_navigator.py` | 3 | Hotkey + template-based menu navigation |
| `autoresearch/templates/` | 3 | Reference images for menu buttons |
| `gameplay_agent/error_capture.py` | 4 | Captures problematic gameplay screenshots (3 capture methods + CapturedError metadata) |
| `autoresearch/detection_loop.py` | 4 | Manages error-capture-to-retrain cycle |
| `detection/training/config/sprite_configs.yaml` | 5 | Externalized sprite configuration (from Python dicts) |

---

## 13. Cost Estimates

### Per Game

| Item | Cost |
|---|---|
| ~600 LLM turns (Sonnet) @ $0.003/turn | ~$1.80 |
| Prompt mutation (Haiku, 1 call) | ~$0.02 |
| Strategy analysis (Haiku, 1 call) | ~$0.05 |

### Per Overnight Run (8 hours)

| Scenario | Games | Cost |
|---|---|---|
| 20-min games, Sonnet gameplay | ~24 | ~$43 |
| 20-min games, Haiku gameplay | ~24 | ~$5 |
| 10-min games, Haiku gameplay | ~48 | ~$10 |
| Context tuning only (50-turn tests) | ~100 tests | ~$3 |

### Recommended Starting Configuration

Use **Sonnet** for the first 5 baseline games to establish reliable scoring, then switch to **Haiku** for bulk overnight experiments. Final validation of the best prompt should always use Sonnet.
