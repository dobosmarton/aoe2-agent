"""Shared test factories for detected-entity dicts.

The serialized entity shape (`class`/`id`/`center`/`confidence`) is consumed via
`entity_utils.extract_attrs` by the reactive tier and the villager-job model;
tests build it through this one factory instead of per-file copies.
"""

from __future__ import annotations


def make_entity(cls: str, center: tuple[float, float] = (0.0, 0.0), eid: str | None = None) -> dict:
    """A minimal serialized detected entity, id defaulting to `<class>_0`."""
    return {"class": cls, "id": eid or f"{cls}_0", "center": center, "confidence": 0.9}
