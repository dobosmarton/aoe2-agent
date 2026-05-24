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
_REDIS_PASSWORD_ENV: Final = "REDIS_PASSWORD"


def _resolved_redis_url() -> str:
    """Pick the connection URL, preferring `REDIS_URL` and otherwise
    smart-defaulting against `REDIS_PASSWORD`.

    Priority:
      1. `REDIS_URL` if set — the operator has explicit control.
      2. `redis://:<REDIS_PASSWORD>@localhost:6379/0` if `REDIS_PASSWORD`
         is set — the compose-stack case (`docker-compose.yml` Redis runs
         with AUTH; `env.example` populates `REDIS_PASSWORD`). Without
         this fallback, users who set `ARENA_BROKER_BACKEND=redis` from
         a populated `.env` would hit `redis.exceptions.AuthenticationError`
         on first publish — a silent foot-gun.
      3. `redis://localhost:6379/0` — bare unauthenticated localhost.
         Right for ad-hoc dev with a `redis-server` you launched yourself.

    Done as a function (not a module-level Final) so `REDIS_PASSWORD` is
    resolved at `make_broker()` call time, not import time. Otherwise a
    test that monkeypatches `REDIS_PASSWORD` would still see the
    pre-import value.
    """
    explicit = os.environ.get(_REDIS_URL_ENV)
    if explicit:
        return explicit
    password = os.environ.get(_REDIS_PASSWORD_ENV)
    if password:
        return f"redis://:{password}@localhost:6379/0"
    return "redis://localhost:6379/0"


def make_broker() -> EventBroker:
    """Construct the broker selected by `ARENA_BROKER_BACKEND`.

    Returns `InProcessEventBroker()` when the env var is unset or
    `"inprocess"`. Returns `RedisStreamsBroker` when set to `"redis"`,
    resolving the connection URL via `_resolved_redis_url()` (see that
    helper for the `REDIS_URL` → `REDIS_PASSWORD` → bare-localhost
    fallback chain). Any other value raises `ValueError` — silently
    falling back would hide deployment misconfigurations.
    """
    backend = os.environ.get(_BACKEND_ENV, "inprocess").strip().lower()
    if backend == "inprocess":
        return InProcessEventBroker()
    if backend == "redis":
        # Local import: keeps `redis` out of the slim-install dependency
        # graph (see module docstring).
        from redis.asyncio import Redis

        from evaluation.redis_broker import RedisStreamsBroker

        return RedisStreamsBroker(Redis.from_url(_resolved_redis_url()))
    raise ValueError(f"unknown {_BACKEND_ENV}={backend!r}; expected 'inprocess' or 'redis'")


__all__ = ["make_broker"]
