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
    # True when the executor produced NO usable turn — every LLM path (single-
    # shot and its tool-loop fallback) failed and this is a safe-wait no-op.
    # Drives the executor-outage alarm and the llm_error_rate metric (T-533):
    # run 12 logged 90 such turns yet still wrote accepted=true, so the outage
    # was invisible in results.tsv.
    error: bool
