"""Offline tests for `autoresearch.memory_chain.MemoryChain.load_memories()`.

Pure-pytest, no LLM, no API key required. Plants memory files in a temp dir
and asserts the cap (20 max) and the ranking (negative score_impact first,
then positive, then neutral; newest within tier).

Run as:
    pytest tests/test_memory_chain.py -v
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CAP = 20  # mirrors MemoryChain._MAX_MEMORIES; if it drifts, the assertion fails
GENEROUS_TOKEN_BUDGET = 100_000  # high enough that the cap, not the budget, governs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def memories_dir():
    path = Path(tempfile.mkdtemp(prefix="memory_chain_test_"))
    yield path
    shutil.rmtree(path, ignore_errors=True)


def _write_memory(
    memories_dir: Path,
    *,
    file_num: int,
    title: str,
    score_impact: str = "neutral",
    applies_when: str = "any",
    content: str = "I should follow this rule.",
    created: str = "2026-04-25T10:00:00+00:00",
) -> Path:
    path = memories_dir / f"{file_num:03d}_{title}.md"
    path.write_text(
        f"---\n"
        f"type: economy\n"
        f"title: {title}\n"
        f"game_id: test\n"
        f"applies_when: {applies_when}\n"
        f"score_impact: {score_impact}\n"
        f"created: {created}\n"
        f"---\n\n{content}\n"
    )
    return path


def _bullet_count(rendered: str) -> int:
    """Count rendered memory bullets in a load_memories() output."""
    return rendered.count("\n- ")


def _bullets(rendered: str) -> str:
    """Return only the bullet portion of load_memories() output (header stripped).

    Useful for assertions that should NOT match substrings that appear in the
    explanatory header (e.g. `(when:` appears in the precedence note).
    """
    first_bullet = rendered.find("\n- ")
    return rendered[first_bullet:] if first_bullet != -1 else ""


# ---------------------------------------------------------------------------
# Cap tests
# ---------------------------------------------------------------------------


def test_load_memories_caps_at_20(memories_dir: Path):
    """25 input files → exactly 20 bullets in output (the rest get dropped)."""
    from gameplay_agent.memory_chain import MemoryChain

    for i in range(25):
        _write_memory(
            memories_dir,
            file_num=i,
            title=f"rule_{i}",
            score_impact=["negative", "positive", "neutral"][i % 3],
        )

    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    assert _bullet_count(rendered) == CAP


def test_load_memories_under_cap_returns_all(memories_dir: Path):
    """5 input files → 5 bullets, no truncation."""
    from gameplay_agent.memory_chain import MemoryChain

    for i in range(5):
        _write_memory(memories_dir, file_num=i, title=f"rule_{i}")

    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    assert _bullet_count(rendered) == 5


# ---------------------------------------------------------------------------
# Ranking tests
# ---------------------------------------------------------------------------


def test_negative_impact_ranks_before_positive(memories_dir: Path):
    """Negative-impact memories (traps to avoid) appear before positive ones."""
    from gameplay_agent.memory_chain import MemoryChain

    _write_memory(
        memories_dir,
        file_num=1,
        title="positive_rule",
        score_impact="positive",
        content="I should do positive things.",
    )
    _write_memory(
        memories_dir,
        file_num=2,
        title="negative_rule",
        score_impact="negative",
        content="I should avoid negative things.",
    )

    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    assert rendered.index("avoid negative things") < rendered.index("do positive things")


def test_positive_impact_ranks_before_neutral(memories_dir: Path):
    """Positive-impact memories appear before neutral ones."""
    from gameplay_agent.memory_chain import MemoryChain

    _write_memory(
        memories_dir,
        file_num=1,
        title="neutral_rule",
        score_impact="neutral",
        content="I should do neutral things.",
    )
    _write_memory(
        memories_dir,
        file_num=2,
        title="positive_rule",
        score_impact="positive",
        content="I should do positive things.",
    )

    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    assert rendered.index("do positive things") < rendered.index("do neutral things")


def test_within_tier_newer_first(memories_dir: Path):
    """Within the same score_impact tier, newer `created` dates rank first."""
    from gameplay_agent.memory_chain import MemoryChain

    _write_memory(
        memories_dir,
        file_num=1,
        title="older_rule",
        score_impact="negative",
        content="I should follow the older rule.",
        created="2026-01-01T10:00:00+00:00",
    )
    _write_memory(
        memories_dir,
        file_num=2,
        title="newer_rule",
        score_impact="negative",
        content="I should follow the newer rule.",
        created="2026-04-25T10:00:00+00:00",
    )

    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    assert rendered.index("follow the newer rule") < rendered.index("follow the older rule")


def test_cap_keeps_negatives_drops_neutrals(memories_dir: Path):
    """When 25 files cross all three tiers, the 5 dropped ones are neutrals."""
    from gameplay_agent.memory_chain import MemoryChain

    for i in range(10):
        _write_memory(memories_dir, file_num=i, title=f"neg_{i}", score_impact="negative")
    for i in range(10):
        _write_memory(memories_dir, file_num=10 + i, title=f"pos_{i}", score_impact="positive")
    for i in range(5):
        _write_memory(memories_dir, file_num=20 + i, title=f"neu_{i}", score_impact="neutral")

    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    # All 10 negatives + all 10 positives = 20 (the cap). Neutrals are dropped.
    assert _bullet_count(rendered) == CAP
    for i in range(10):
        assert (
            f"neg_{i}" not in rendered or "I should follow this rule." in rendered
        )  # negatives kept
    assert "neu_0" not in rendered  # neutrals dropped
    assert "neu_4" not in rendered


# ---------------------------------------------------------------------------
# Trigger-prefix rendering tests
# ---------------------------------------------------------------------------


def test_any_trigger_omits_when_prefix(memories_dir: Path):
    """A memory with applies_when=any renders WITHOUT a `(when: …)` prefix."""
    from gameplay_agent.memory_chain import MemoryChain

    _write_memory(
        memories_dir,
        file_num=1,
        title="rule",
        applies_when="any",
        content="I should do this always.",
    )
    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    assert "(when:" not in _bullets(rendered)


def test_specific_trigger_renders_when_prefix(memories_dir: Path):
    """A memory with a specific applies_when renders the `(when: ...)` prefix."""
    from gameplay_agent.memory_chain import MemoryChain

    _write_memory(
        memories_dir,
        file_num=1,
        title="rule",
        applies_when="Dark Age AND food < 50",
        content="I should plant farms.",
    )
    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    assert "(when: Dark Age AND food < 50)" in _bullets(rendered)


def test_bullet_starts_with_bracketed_title(memories_dir: Path):
    """Each rendered bullet exposes its snake_case title in `[brackets]`.

    The model relies on this to emit the `[applied: title]` reasoning prefix
    described in prompts/core.md (Telemetry: Tag Applied Memories). If the
    title isn't surfaced, the model has no way to honestly tag.
    """
    from gameplay_agent.memory_chain import MemoryChain

    _write_memory(
        memories_dir,
        file_num=1,
        title="build_house_at_pop_cap_minus_5",
        applies_when="Dark Age AND pop >= pop_cap - 5",
        content="I should build a house immediately.",
    )
    _write_memory(
        memories_dir,
        file_num=2,
        title="rule_with_no_trigger",
        applies_when="any",
        content="I should always do this.",
    )
    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    bullets = _bullets(rendered)
    # Title bracketed at the bullet start, before the `(when: ...)` prefix.
    assert "- [build_house_at_pop_cap_minus_5] (when: Dark Age" in bullets
    # `applies_when=any` still suppresses the (when: ...) part but keeps the title.
    assert "- [rule_with_no_trigger] I should always do this." in bullets


def test_title_falls_back_to_filename_when_frontmatter_missing(memories_dir: Path):
    """Older memory files lacking `title:` frontmatter still get a bracketed title.

    Mirrors list_memories()' fallback: title comes from the `NNN_<title>.md`
    filename pattern. Protects forward-compatibility for memories created
    before the title field was introduced.
    """
    from gameplay_agent.memory_chain import MemoryChain

    # Hand-write a frontmatter block with NO title field — pre-2026-04-25 layout.
    path = memories_dir / "007_legacy_rule_no_title_field.md"
    path.write_text(
        "---\n"
        "type: economy\n"
        "game_id: legacy\n"
        "applies_when: any\n"
        "score_impact: negative\n"
        "created: 2026-01-01T00:00:00+00:00\n"
        "---\n\n"
        "I should follow the legacy rule.\n"
    )

    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    assert "- [legacy_rule_no_title_field]" in _bullets(rendered)


# ---------------------------------------------------------------------------
# Empty-content rejection (defensive — defense-in-depth alongside _save_memory)
# ---------------------------------------------------------------------------


def test_whitespace_only_content_is_dropped(memories_dir: Path):
    """A memory with only whitespace as content should not appear in output."""
    from gameplay_agent.memory_chain import MemoryChain

    _write_memory(memories_dir, file_num=1, title="empty", content="   ")
    _write_memory(memories_dir, file_num=2, title="real", content="I should do something.")

    rendered = MemoryChain(memories_dir=memories_dir).load_memories(
        max_tokens=GENEROUS_TOKEN_BUDGET,
    )
    assert "do something" in rendered
    assert _bullet_count(rendered) == 1
