"""Boundary for providers that make bounded policy judgments."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ..policy.advice import PolicyAdvice, PolicyRequest


class PolicyAdvisorError(RuntimeError):
    """An expected external-service failure while requesting policy advice."""


class PolicyAdvisor(Protocol):
    """The small interface used by the policy loop."""

    async def advise(self, request: PolicyRequest) -> PolicyAdvice: ...

    async def aclose(self) -> None: ...


__all__ = ["PolicyAdvisor", "PolicyAdvisorError"]
