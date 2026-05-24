"""core — pure types and protocols shared across all packages.

Zero I/O. Zero async. The single dependency is pydantic. Every other
package in the workspace consumes types from here; nothing here imports
from another workspace package. Breaking that rule reintroduces the
back-edges the broker rewrite was designed to eliminate.
"""

from __future__ import annotations

from core.entity import DetectedEntity
from core.event_log import (
    SCHEMA_VERSION,
    ActionPayload,
    ActionResultPayload,
    Event,
    EventRow,
    EventSink,
    ForkPayload,
    LlmPromptPayload,
    LlmResponsePayload,
    MetricPayload,
    NullEventSink,
    ObservationPayload,
    Payload,
    TurnStartPayload,
    WorldMutationPayload,
    WorldStateSnapshot,
)
from core.world_state import AGE_SEQUENCE, WorldState

__all__ = [
    "AGE_SEQUENCE",
    "SCHEMA_VERSION",
    "ActionPayload",
    "ActionResultPayload",
    "DetectedEntity",
    "Event",
    "EventRow",
    "EventSink",
    "ForkPayload",
    "LlmPromptPayload",
    "LlmResponsePayload",
    "MetricPayload",
    "NullEventSink",
    "ObservationPayload",
    "Payload",
    "TurnStartPayload",
    "WorldMutationPayload",
    "WorldState",
    "WorldStateSnapshot",
]
