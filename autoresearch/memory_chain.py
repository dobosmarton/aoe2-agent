"""Cross-game memory chain — persistent observations that improve over time.

After each game, the agent extracts 1-3 useful observations into individual
markdown files in memories/. Future games load these fragments into the LLM
context, enabling learning from experience.

Memory files are human-readable and reviewable — delete any bad ones.
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import anthropic
import structlog

log = structlog.get_logger()

MEMORIES_DIR = Path(__file__).parent.parent / "memories"

EXTRACTION_SYSTEM = """You are analyzing a completed Age of Empires II game played by an AI agent.
Your job: extract 1-3 concise, actionable observations that would help the agent play better next time.

Each observation should be:
- Specific and actionable (not vague like "play better")
- Based on what actually happened in this game
- Something the agent can apply in future games

Types of observations:
- **strategy**: Build orders, resource priorities, timing decisions
- **economy**: Food gathering, villager management, age-up timing
- **military**: Unit composition, attack timing, defense
- **detection**: Entity recognition issues, coordinate problems
- **failure**: What went wrong and how to avoid it

Respond with JSON only:
{
  "observations": [
    {
      "type": "strategy|economy|military|detection|failure",
      "title": "short_snake_case_title",
      "content": "The actionable observation in 1-3 sentences.",
      "score_impact": "positive|negative|neutral",
      "turn_range": "1-15"
    }
  ]
}"""


class MemoryChain:
    """Manages persistent cross-game memory fragments."""

    def __init__(self, memories_dir: Path | str = MEMORIES_DIR):
        self.memories_dir = Path(memories_dir)
        self.memories_dir.mkdir(parents=True, exist_ok=True)
        self.client = anthropic.Anthropic()

    def extract_memories(
        self,
        memory,
        score,
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
            observations = self._parse_observations(response.content[0].text)
        except Exception as e:
            log.error("memory_extraction_failed", error=str(e))
            return []

        if not observations:
            log.info("memory_extraction_empty", game_id=game_id)
            return []

        # Save each observation as a separate file
        created = []
        next_num = self._next_file_number()
        for obs in observations:
            path = self._save_memory(obs, game_id, next_num)
            if path:
                created.append(path)
                next_num += 1
                log.info("memory_created", path=path.name, type=obs.get("type"))

        return created

    def load_memories(self, max_tokens: int = 800) -> str:
        """Load memory fragments into a context string for LLM injection.

        Args:
            max_tokens: Approximate token budget (1 token ~ 4 chars)

        Returns:
            Formatted string with learned observations, or empty string.
        """
        files = sorted(self.memories_dir.glob("*.md"))
        if not files:
            return ""

        memories = []
        for f in files:
            content = self._read_memory(f)
            if content:
                memories.append(content)

        if not memories:
            return ""

        # Build context string, respecting token budget
        header = "## Learned Observations (from previous games)\n"
        char_budget = max_tokens * 4  # rough chars-to-tokens
        lines = []
        total_chars = len(header)

        for mem in memories:
            line = f"- {mem}"
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
            text = f.read_text()
            meta = self._parse_frontmatter(text)
            content = self._strip_frontmatter(text).strip()
            result.append({
                "file": f.name,
                "type": meta.get("type", "unknown"),
                "game_id": meta.get("game_id", "unknown"),
                "content": content,
            })
        return result

    def _build_game_summary(self, memory, score) -> str:
        """Build a text summary of the game for the extraction LLM."""
        parts = []

        # Metrics
        metrics = memory.get_metrics_snapshot()
        parts.append(f"Game Result: {metrics['game_end_reason'] or 'unknown'}")
        parts.append(f"Score: {score.composite:.4f} (survival={score.survival:.2f}, pop={score.population:.2f}, age={score.age:.2f}, economy={score.economy:.2f}, actions={score.action_success:.2f})")
        parts.append(f"Duration: {metrics['survival_time']:.0f}s, Turns: {metrics['turn_count']}")
        parts.append(f"Peak Population: {metrics['peak_population']}, Highest Age: {metrics['highest_age']}")
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

    def _parse_observations(self, text: str) -> list[dict]:
        """Parse LLM response into observation dicts."""
        from .json_utils import extract_json_object

        data = extract_json_object(text)
        if data is not None:
            return data.get("observations", [])

        log.warning("memory_parse_failed", text=text[:200])
        return []

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

    def _save_memory(self, obs: dict, game_id: str, file_num: int) -> Path | None:
        """Save a single observation as a markdown file."""
        mem_type = obs.get("type", "strategy")
        title = obs.get("title", "observation")
        content = obs.get("content", "")
        score_impact = obs.get("score_impact", "neutral")
        turn_range = obs.get("turn_range", "unknown")

        if not content:
            return None

        # Sanitize title for filename
        safe_title = re.sub(r"[^a-z0-9_]", "_", title.lower())[:50]
        filename = f"{file_num:03d}_{safe_title}.md"
        path = self.memories_dir / filename

        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        file_content = f"""---
type: {mem_type}
game_id: {game_id}
turn_range: {turn_range}
score_impact: {score_impact}
created: {now}
---

{content}
"""
        path.write_text(file_content)
        return path

    def _read_memory(self, path: Path) -> str | None:
        """Read a memory file and return just the content (no frontmatter)."""
        text = path.read_text()
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
            return text[match.end():]
        return text
