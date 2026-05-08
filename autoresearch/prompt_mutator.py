"""Prompt mutator — LLM-driven system prompt modification for autoresearch.

Uses a cheap model (Haiku) to propose targeted changes to prompts/system.md,
apply them, and revert on failure.
"""

import subprocess
from pathlib import Path

import anthropic
import structlog

log = structlog.get_logger()

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "system.md"

# Sections the mutator must NOT modify
PROTECTED_SECTIONS = ["## Output Format", "## Game State Detection"]

MUTATOR_SYSTEM = """You are an expert Age of Empires II strategist optimizing a system prompt for an AI game-playing agent.

Your goal: propose a SMALL, targeted change to the prompt that will improve the agent's game performance score.

The score is a weighted composite of:
- Survival time (30%) — how long the agent survives
- Peak population (25%) — highest population reached
- Age advancement (20%) — Dark → Feudal → Castle → Imperial
- Economy (15%) — total food gathered
- Action success rate (10%) — fraction of actions that had observable effect

Rules:
- Change at most 5 lines of the prompt
- Do NOT modify the "## Output Format" or "## Game State Detection" sections
- Focus on strategy, priorities, and decision-making heuristics
- Be specific (e.g., "always build 2 houses before population reaches 10" not "build more houses")
- Each change should target ONE specific weakness

Respond with JSON only:
{
  "description": "Short description of the change",
  "old_text": "Exact text to find and replace (must exist in the prompt)",
  "new_text": "Replacement text",
  "rationale": "Why this should improve the score"
}

If proposing to ADD new text (not replace), set old_text to a line that exists in the prompt
and set new_text to that same line PLUS your addition after it."""


class PromptMutator:
    """Proposes, applies, and reverts changes to the system prompt."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001") -> None:
        self.client = anthropic.Anthropic()
        self.model = model

    def read_current_prompt(self) -> str:
        """Read the current system prompt."""
        return PROMPT_FILE.read_text()

    def propose_change(
        self,
        current_prompt: str,
        recent_experiments: list[dict],
        failure_modes: list[str],
    ) -> dict | None:
        """Ask LLM to propose a prompt modification.

        Args:
            current_prompt: Full text of prompts/system.md
            recent_experiments: Last 5 experiments from experiment_log
            failure_modes: Specific failures from most recent game

        Returns:
            Dict with description, old_text, new_text, rationale. None on failure.
        """
        experiment_summary = self._format_experiments(recent_experiments)
        failure_summary = (
            "\n".join(f"- {f}" for f in failure_modes) if failure_modes else "None identified yet"
        )

        user_msg = f"""Current system prompt:
```
{current_prompt}
```

Recent experiment results:
{experiment_summary}

Known failure modes from recent games:
{failure_summary}

Propose ONE targeted change to improve the agent's game performance."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=MUTATOR_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            block = response.content[0]
            if not isinstance(block, anthropic.types.TextBlock):
                log.error("prompt_mutator_unexpected_block", block_type=type(block).__name__)
                return None
            return self._parse_response(block.text)
        except Exception as e:
            log.error("prompt_mutator_error", error=str(e))
            return None

    def apply_change(self, old_text: str, new_text: str) -> bool:
        """Apply the proposed change to prompts/system.md.

        Returns True if the change was applied successfully.
        """
        current = self.read_current_prompt()

        if old_text not in current:
            log.warning("change_not_applicable", old_text=old_text[:100])
            return False

        # Verify protected sections won't be modified
        for section in PROTECTED_SECTIONS:
            section_start = current.find(section)
            if section_start == -1:
                continue
            old_start = current.find(old_text)
            if old_start == -1:
                continue
            # Check if old_text overlaps with protected section
            next_section = current.find("\n## ", section_start + len(section))
            section_end = next_section if next_section != -1 else len(current)
            if section_start <= old_start < section_end:
                log.warning("change_in_protected_section", section=section)
                return False

        modified = current.replace(old_text, new_text, 1)
        PROMPT_FILE.write_text(modified)
        log.info("prompt_change_applied", old_len=len(old_text), new_len=len(new_text))
        return True

    def revert(self) -> None:
        """Revert prompt to last git-committed version."""
        try:
            subprocess.run(
                ["git", "checkout", "--", str(PROMPT_FILE)],
                cwd=PROMPT_FILE.parent.parent,
                capture_output=True,
            )
            log.info("prompt_reverted")
        except Exception as e:
            log.error("prompt_revert_failed", error=str(e))

    def _format_experiments(self, experiments: list[dict]) -> str:
        if not experiments:
            return "No previous experiments — this is the first run (baseline)."

        lines = []
        for exp in experiments:
            status = "KEPT" if exp.get("accepted") == "true" else "REVERTED"
            lines.append(
                f"  {exp.get('experiment_id', '?')}: score={exp.get('composite_score', '?')} "
                f"[{status}] — {exp.get('change_description', '?')}"
            )
        return "\n".join(lines)

    def _parse_response(self, text: str) -> dict | None:
        """Extract JSON from the LLM response."""
        from .json_utils import extract_json_object

        return extract_json_object(text)
