"""Broker selection — the single point where `ARENA_BROKER_BACKEND` is read.

`make_broker()` decides between the in-process broker (default, no
external deps) and `RedisStreamsBroker` (Phase C, cross-process). Every
callsite that needs a broker constructs it through this factory so the
env-var read happens in exactly one place — adding a third impl later
is a one-arm match extension here, nothing else.

The Redis import is local to the `"redis"` branch so a `make_broker()`
call with the default backend never imports the `redis` package. That
matters because `redis` is an optional extra: a slim install
(`pip install -e .`) must keep working without it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Final

from evaluation.event_broker import InProcessEventBroker

if TYPE_CHECKING:
    from evaluation.event_broker import EventBroker


_BACKEND_ENV: Final = "ARENA_BROKER_BACKEND"
_REDIS_URL_ENV: Final = "REDIS_URL"
_DEFAULT_REDIS_URL: Final = "redis://localhost:6379/0"


def make_broker() -> EventBroker:
    """Construct the broker selected by `ARENA_BROKER_BACKEND`.

    Returns `InProcessEventBroker()` when the env var is unset or
    `"inprocess"`. Returns `RedisStreamsBroker` when set to `"redis"`,
    reading the connection URL from `REDIS_URL` (default:
    `redis://localhost:6379/0`). Any other value raises `ValueError`
    — silently falling back would hide deployment misconfigurations.
    """
    backend = os.environ.get(_BACKEND_ENV, "inprocess").strip().lower()
    if backend == "inprocess":
        return InProcessEventBroker()
    if backend == "redis":
        # Local import: keeps `redis` out of the slim-install dependency
        # graph (see module docstring).
        from redis.asyncio import Redis

        from evaluation.redis_broker import RedisStreamsBroker

        url = os.environ.get(_REDIS_URL_ENV, _DEFAULT_REDIS_URL)
        return RedisStreamsBroker(Redis.from_url(url))
    raise ValueError(f"unknown {_BACKEND_ENV}={backend!r}; expected 'inprocess' or 'redis'")


__all__ = ["make_broker"]
