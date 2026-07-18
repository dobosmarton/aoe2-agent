"""Cross-game memory chain — persistent observations that improve over time.

After each game, the agent extracts 1-3 useful observations into individual
markdown files in memories/. Future games load these fragments into the LLM
context, enabling learning from experience.

Memory files are human-readable and reviewable — delete any bad ones.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import anthropic
import structlog
from core.json_utils import extract_json_object

if TYPE_CHECKING:
    from autoresearch.metrics import GameScore
    from gameplay_agent.memory import AgentMemory

log = structlog.stdlib.get_logger()

MEMORIES_DIR = Path(__file__).parent.parent / "memories"

_NUMBERED_FILENAME_RE = re.compile(r"\d+_(.+)\.md$")


def _read_memory_file(path: Path) -> str:
    """Read a memory file, tolerating stray non-UTF-8 bytes.

    One cp1252 em-dash (0x97) in a single file made every load raise and
    silently zeroed the whole memory feature for 13 straight VM runs (T-536,
    run 12 F-44). Bad bytes become U+FFFD instead of an exception — one
    mangled character beats losing every memory.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def _resolve_memory_title(meta: dict, file: Path) -> str:
    """Resolve the snake_case title used in `[applied: title]` reasoning tags.

    Title field was added to memory frontmatter on 2026-04-25. Older files
    fall back to the suffix of a `NNNN_<title>.md` filename, then to `f.stem`.
    """
    title = meta.get("title")
    if title:
        return title
    match = _NUMBERED_FILENAME_RE.match(file.name)
    return match.group(1) if match else file.stem


@dataclass
class _MemoryEntry:
    """One ranked memory fragment loaded for context assembly."""

    rank: int
    created: str
    title: str
    applies_when: str
    content: str


EXTRACTION_SYSTEM = """You just finished an Age of Empires II game. Write 1-3 first-person notes to
your future self — rules you should follow next game, based on what happened in this one.

CRITICAL voice rules — every note's `content` MUST follow these:
- **First person, present/future tense.** Start with "I should...", "I must...", or
  "Next game, I will...". Never write "the agent", "the AI", or "you". You are
  writing to YOURSELF for the next game.
- **Imperative, not diagnostic.** Write the rule, then one short sentence of
  evidence. "I should X. Last game, Y happened — that's why."
- **Specific thresholds, not vague advice.** Bad: "I should manage housing better."
  Good: "I should build a house the moment population reaches pop_cap minus 3."
- **One rule per note.** Don't bundle. If you have three lessons, write three notes.
- **No turn numbers in the rule.** "In turns 30-39 I..." is meaningless next game.
  The rule must apply purely from the next game's observable state.

Use the `applies_when` field to state the trigger condition the agent should match
against its current state — e.g. "Dark Age AND pop >= pop_cap - 3", "food < 50",
"any age". Keep it short and grep-friendly.

Types:
- **strategy**: Build orders, age-up timing, win conditions
- **economy**: Villager allocation, gather rates, food management
- **military**: Unit composition, attack/defense timing
- **detection**: Vision/coordinate issues
- **failure**: Specific traps to avoid that hurt my score

Respond with JSON only:
{
  "observations": [
    {
      "type": "strategy|economy|military|detection|failure",
      "title": "short_snake_case",
      "content": "I should [rule]. Last game, [one-sentence evidence].",
      "applies_when": "trigger condition in plain English",
      "score_impact": "positive|negative|neutral"
    }
  ]
}

Reject the urge to write more than 3 notes — quality over quantity. If nothing
genuinely new happened, write 0 or 1 notes."""


class MemoryChain:
    """Manages persistent cross-game memory fragments."""

    def __init__(self, memories_dir: Path | str = MEMORIES_DIR) -> None:
        self.memories_dir = Path(memories_dir)
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.client = anthropic.Anthropic()

    def extract_memories(
        self,
        memory: AgentMemory,
        score: GameScore,
        game_id: str,
        model: str = "claude-haiku-4-5-20251001",
    ) -> list[Path]:
        """Extract memory fragments from a completed game.

        Args:
            memory: AgentMemory instance (with working_memory still populated)
            score: GameScore from the game
            game_id: Experiment ID for attribution

        Returns:
            List of paths to created memory files.
        """
        # Build game summary from turn history
        game_summary = self._build_game_summary(memory, score)

        if not game_summary.strip():
            log.info("memory_extraction_skipped", reason="no turns to analyze")
            return []

        try:
            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                system=EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": game_summary}],
            )
            block = response.content[0]
            if not isinstance(block, anthropic.types.TextBlock):
                log.error("memory_extraction_unexpected_block", block_type=type(block).__name__)
                return []
            observations = self._parse_observations(block.text)
        except Exception as e:
            log.error("memory_extraction_failed", error=str(e))
            return []

        if not observations:
            log.info("memory_extraction_empty", game_id=game_id)
            return []

        # Light dedup: skip any observation whose sanitized title is already on disk.
        # Cheap client-side filter — keeps the dir from accumulating duplicate
        # "stuck_at_population_cap"-type rules across dozens of games.
        existing_titles = self._existing_titles()

        created = []
        next_num = self._next_file_number()
        for obs in observations:
            safe_title = re.sub(r"[^a-z0-9_]", "_", (obs.get("title") or "observation").lower())[
                :50
            ]
            if safe_title in existing_titles:
                log.info("memory_dedup_skipped", title=safe_title, type=obs.get("type"))
                continue
            path = self._save_memory(obs, game_id, next_num)
            if path:
                created.append(path)
                existing_titles.add(safe_title)
                next_num += 1
                log.info("memory_created", path=path.name, type=obs.get("type"))

        return created

    # Hard cap on how many memories to load — prevents the dir bloating
    # gameplay context as games accumulate. negative > positive > neutral.
    _MAX_MEMORIES: ClassVar[int] = 20
    _IMPACT_RANK: ClassVar[dict[str, int]] = {"negative": 0, "positive": 1, "neutral": 2}

    def load_memories(self, max_tokens: int = 800) -> str:
        """Load memory fragments into a first-person context string.

        Memories are ranked: negative `score_impact` first (traps to avoid),
        then positive (patterns to repeat), then neutral. Within each tier,
        most recently created first. Caps at `_MAX_MEMORIES` then trims by
        token budget.

        Args:
            max_tokens: Approximate token budget (1 token ~ 4 chars)

        Returns:
            Formatted string starting with `## Notes to Myself…`, or empty.
        """
        files = list(self.memories_dir.glob("*.md"))
        if not files:
            return ""

        entries: list[_MemoryEntry] = []
        for f in files:
            text = _read_memory_file(f)
            meta = self._parse_frontmatter(text)
            content = self._strip_frontmatter(text).strip()
            if not content:
                continue
            entries.append(
                _MemoryEntry(
                    rank=self._IMPACT_RANK.get(meta.get("score_impact", "neutral"), 2),
                    created=meta.get("created", ""),
                    title=_resolve_memory_title(meta, f),
                    applies_when=meta.get("applies_when", "").strip(),
                    content=content,
                )
            )

        if not entries:
            return ""

        # Sort by impact tier ascending (negative first), then created descending
        # (newest first). created is an ISO 8601 string so lexicographic sort works.
        entries.sort(key=lambda e: e.created, reverse=True)
        entries.sort(key=lambda e: e.rank)
        ordered = entries[: self._MAX_MEMORIES]

        header = (
            "## Notes to Myself from Previous Games\n"
            "Each bullet is a rule I wrote for myself after finishing a game, based "
            "on what actually happened.\n"
            "\n"
            "**Precedence: when a memory rule conflicts with a rule in core.md or "
            "the age-specific section, follow the MEMORY.** Memories reflect "
            "concrete evidence from my own games; the defaults are pre-game "
            "heuristics. If two memories conflict, prefer the one whose "
            "`(when: …)` trigger is more specific or matches my current state "
            "more tightly.\n"
            "\n"
            "I should apply any rule whose trigger matches my current state.\n"
        )
        char_budget = max_tokens * 4
        lines: list[str] = []
        total_chars = len(header)

        for entry in ordered:
            # First line of content keeps the bullet compact; multi-sentence
            # content is preserved verbatim after the trigger prefix.
            when_prefix = (
                f"(when: {entry.applies_when}) "
                if entry.applies_when and entry.applies_when != "any"
                else ""
            )
            # `[title]` makes the snake_case identifier visible to the model so it
            # can emit the `[applied: title]` reasoning prefix per prompts/core.md.
            line = f"- [{entry.title}] {when_prefix}{entry.content}"
            if total_chars + len(line) + 1 > char_budget:
                break
            lines.append(line)
            total_chars += len(line) + 1

        if not lines:
            return ""

        return header + "\n".join(lines)

    def list_memories(self) -> list[dict]:
        """List all memory fragments with metadata."""
        result = []
        for f in sorted(self.memories_dir.glob("*.md")):
            text = _read_memory_file(f)
            meta = self._parse_frontmatter(text)
            content = self._strip_frontmatter(text).strip()
            title = _resolve_memory_title(meta, f)
            result.append(
                {
                    "file": f.name,
                    "title": title,
                    "type": meta.get("type", "unknown"),
                    "game_id": meta.get("game_id", "unknown"),
                    "applies_when": meta.get("applies_when", ""),
                    "score_impact": meta.get("score_impact", "neutral"),
                    "content": content,
                }
            )
        return result

    def _build_game_summary(self, memory: AgentMemory, score: GameScore) -> str:
        """Build a text summary of the game for the extraction LLM."""
        parts = []

        # Metrics
        metrics = memory.get_metrics_snapshot()
        parts.append(f"Game Result: {metrics['game_end_reason'] or 'unknown'}")
        parts.append(
            f"Score: {score.composite:.4f} (survival={score.survival:.2f}, pop={score.population:.2f}, age={score.age:.2f}, economy={score.economy:.2f}, actions={score.action_success:.2f})"
        )
        parts.append(f"Duration: {metrics['survival_time']:.0f}s, Turns: {metrics['turn_count']}")
        parts.append(
            f"Peak Population: {metrics['peak_population']}, Highest Age: {metrics['highest_age']}"
        )
        parts.append("")

        # Turn history
        turns = list(memory.working_memory)
        if not turns:
            return ""

        parts.append("Turn-by-turn summary (last 10 turns):")
        for t in turns:
            action_summary = ", ".join(
                f"{a.get('type', '?')}({a.get('key', a.get('target_id', ''))})"
                for a in t.actions[:4]
            )
            parts.append(f"  Turn {t.iteration}: {t.reasoning[:150]}")
            parts.append(f"    Actions: {action_summary}")
            if t.observed_resources:
                parts.append(f"    Resources: {t.observed_resources}")

        return "\n".join(parts)

    def _parse_observations(self, text: str) -> list[dict[str, object]]:
        """Parse LLM response into observation dicts.

        Drops entries with empty content (was producing 0-byte files —
        see exp_0011's `006_missing_feudal_age_target.md`).
        """
        data = extract_json_object(text)
        if data is None:
            log.warning("memory_parse_failed", text=text[:200])
            return []

        raw_value = data.get("observations")
        raw: list[object] = raw_value if isinstance(raw_value, list) else []
        valid = [obs for obs in raw if isinstance(obs, dict) and (obs.get("content") or "").strip()]
        if len(valid) < len(raw):
            log.warning("memory_parse_dropped_empty", dropped=len(raw) - len(valid))
        return valid

    def _next_file_number(self) -> int:
        """Get the next available file number."""
        existing = list(self.memories_dir.glob("*.md"))
        if not existing:
            return 1
        numbers = []
        for f in existing:
            match = re.match(r"(\d+)_", f.name)
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def _existing_titles(self) -> set[str]:
        """Return the set of sanitized titles already saved on disk.

        Reads from frontmatter `title:` first, falls back to filename suffix.
        """
        titles: set[str] = set()
        for f in self.memories_dir.glob("*.md"):
            meta = self._parse_frontmatter(_read_memory_file(f))
            title = meta.get("title")
            if not title:
                # Filename pattern: NNN_<title>.md
                match = re.match(r"\d+_(.+)\.md$", f.name)
                if match:
                    title = match.group(1)
            if title:
                titles.add(title)
        return titles

    def _save_memory(self, obs: dict, game_id: str, file_num: int) -> Path | None:
        """Save a single observation as a markdown file."""
        mem_type = obs.get("type", "strategy")
        title = obs.get("title", "observation")
        content = (obs.get("content") or "").strip()
        score_impact = obs.get("score_impact", "neutral")
        applies_when = (obs.get("applies_when") or "any").strip()

        if not content:
            return None

        # Sanitize title for filename
        safe_title = re.sub(r"[^a-z0-9_]", "_", title.lower())[:50]
        filename = f"{file_num:03d}_{safe_title}.md"
        path = self.memories_dir / filename

        now = datetime.now(UTC).isoformat(timespec="seconds")
        file_content = f"""---
type: {mem_type}
title: {safe_title}
game_id: {game_id}
applies_when: {applies_when}
score_impact: {score_impact}
created: {now}
---

{content}
"""
        path.write_text(file_content, encoding="utf-8")
        return path

    def _read_memory(self, path: Path) -> str | None:
        """Read a memory file and return just the content (no frontmatter)."""
        text = _read_memory_file(path)
        content = self._strip_frontmatter(text).strip()
        return content if content else None

    def _parse_frontmatter(self, text: str) -> dict:
        """Parse YAML-like frontmatter from a memory file."""
        match = re.match(r"^---\n(.+?)\n---", text, re.DOTALL)
        if not match:
            return {}
        meta = {}
        for line in match.group(1).split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        return meta

    def _strip_frontmatter(self, text: str) -> str:
        """Remove frontmatter from text."""
        match = re.match(r"^---\n.+?\n---\n?", text, re.DOTALL)
        if match:
            return text[match.end() :]
        return text
