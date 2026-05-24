"""Test-environment scaffolding for scenario runs.

Three side-effect surfaces the runner has to neutralize so a fixture run
doesn't touch the user's machine:

  - `memories/` — backed up + replaced with fixture-supplied memories
  - `executor.execute_action` — patched to a no-op so pyautogui never fires
  - `executor._detected_entities` — seeded from the fixture so the agent's
    `target_class` / `target_id` resolution finds something to click

Each helper is a context manager (`_isolate_memories_dir`, `_mock_executor`)
or a one-shot setter (`_seed_detected_entities`); the runner composes them
inside `with ...` blocks for each variant.
"""

from __future__ import annotations

import contextlib
import shutil
from typing import TYPE_CHECKING

from gameplay_agent.context_builder import _entity_dict

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from gameplay_agent.executor import ActionResult


@contextlib.contextmanager
def _isolate_memories_dir(fixture_memories: list[dict]) -> Iterator[None]:
    """Back up existing memories/, plant fixture memories, restore on exit.

    Refuses to run if an orphan `<memories>_eval_backup` directory exists
    from a crashed prior run — its contents are the user's real memories
    and we cannot tell which of the two dirs is canonical without asking.
    """
    from gameplay_agent.memory_chain import MEMORIES_DIR

    backup_dir = MEMORIES_DIR.with_name(MEMORIES_DIR.name + "_eval_backup")
    if backup_dir.exists():
        raise RuntimeError(
            f"Found orphan eval backup at {backup_dir}. A prior evaluation "
            f"run crashed before restoring your real memories. Inspect both "
            f"{backup_dir} and {MEMORIES_DIR}, move the canonical contents "
            f"back to {MEMORIES_DIR}, then delete the other. Refusing to "
            f"proceed to avoid silent data loss."
        )

    had_existing = MEMORIES_DIR.exists()
    if had_existing:
        shutil.move(str(MEMORIES_DIR), str(backup_dir))
    MEMORIES_DIR.mkdir(parents=True, exist_ok=True)

    for index, memory in enumerate(fixture_memories, start=1):
        _write_fixture_memory(MEMORIES_DIR, memory, index)

    try:
        yield
    finally:
        shutil.rmtree(MEMORIES_DIR, ignore_errors=True)
        if had_existing and backup_dir.exists():
            shutil.move(str(backup_dir), str(MEMORIES_DIR))
        else:
            MEMORIES_DIR.mkdir(parents=True, exist_ok=True)


def _write_fixture_memory(memories_dir: Path, memory: dict, index: int) -> None:
    """Write a single fixture memory file with frontmatter."""
    title = memory.get("title", f"fixture_memory_{index}")
    applies_when = memory.get("applies_when", "any")
    score_impact = memory.get("score_impact", "negative")
    mem_type = memory.get("type", "economy")
    content = memory.get("content", "I should follow this rule.")
    path = memories_dir / f"{index:03d}_{title}.md"
    path.write_text(
        f"---\n"
        f"type: {mem_type}\n"
        f"title: {title}\n"
        f"game_id: fixture\n"
        f"applies_when: {applies_when}\n"
        f"score_impact: {score_impact}\n"
        f"created: 2026-04-25T00:00:00+00:00\n"
        f"---\n\n{content}\n"
    )


@contextlib.contextmanager
def _mock_executor() -> Iterator[None]:
    """Patch execute_action in both the canonical module and the import in claude.py.

    The executor's tool loop calls execute_action for every action; without
    mocking it would invoke pyautogui (real clicks). We replace it with a
    success-returning no-op so the LLM's behavior loop is preserved.
    """
    import gameplay_agent.executor as ex
    import gameplay_agent.providers.claude as claude_mod

    real_canonical = ex.execute_action
    real_in_claude = claude_mod.execute_action

    async def fake_execute_action(action_dict: dict) -> ActionResult:
        return ex.ActionResult(success=True, detail="ok (eval)")

    ex.execute_action = fake_execute_action
    claude_mod.execute_action = fake_execute_action

    try:
        yield
    finally:
        ex.execute_action = real_canonical
        claude_mod.execute_action = real_in_claude


def _seed_detected_entities(entities: list[dict]) -> None:
    """Push fixture entities into executor module state so target_class resolution works."""
    import gameplay_agent.executor as ex

    ex._detected_entities = [_entity_dict(entity, index) for index, entity in enumerate(entities)]
