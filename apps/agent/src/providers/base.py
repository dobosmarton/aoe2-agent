"""Shared LLM provider types for AoE2 Agent."""

from typing import TypedDict


class LLMResult(TypedDict, total=False):
    """Structured payload returned by `ClaudeProvider.get_actions`.

    `total=False` makes every key Optional — the provider may omit
    fields when nothing of interest happened that turn (e.g. an empty
    response). Consumers should `.get(...)` with a default rather than
    indexing directly.
    """

    reasoning: str
    actions: list[dict[str, object]]
    observations: dict[str, object] | None
    actions_already_executed: bool
    success_count: int
