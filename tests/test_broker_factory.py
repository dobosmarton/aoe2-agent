"""Tests for `evaluation.broker_factory.make_broker()`.

`make_broker()` is the *only* place `ARENA_BROKER_BACKEND` is read, so it's
also the place a deployment misconfiguration becomes visible. Without these
tests, the factory was uncovered: a typo in the env var (`"redus"`) would
crash at first use rather than at startup, and an uninstalled `redis`
package would raise a confusing `ImportError` from inside a CLI.

Every test patches the env via `monkeypatch.setenv` so process-wide state
isn't leaked across tests. The redis branch is exercised through `fakeredis`
(no real connection), keeping these in the fast `pytest -q` tier.
"""

from __future__ import annotations

import pytest

from evaluation.broker_factory import make_broker
from evaluation.event_broker import InProcessEventBroker


def test_make_broker_default_returns_inprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ARENA_BROKER_BACKEND", raising=False)
    broker = make_broker()
    assert isinstance(broker, InProcessEventBroker)


def test_make_broker_inprocess_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARENA_BROKER_BACKEND", "inprocess")
    broker = make_broker()
    assert isinstance(broker, InProcessEventBroker)


def test_make_broker_case_and_whitespace_tolerance(monkeypatch: pytest.MonkeyPatch) -> None:
    # The factory normalizes via `.strip().lower()`. Verifying both ends of
    # that pipeline so a future refactor that drops one half (or both) breaks
    # loudly here rather than failing some user's deploy at 2am.
    monkeypatch.setenv("ARENA_BROKER_BACKEND", "  InProcess  ")
    assert isinstance(make_broker(), InProcessEventBroker)


def test_make_broker_redis_returns_protocol_satisfying_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The redis branch imports `redis.asyncio.Redis` lazily — patching
    # `Redis.from_url` to return a `FakeRedis` instance lets us exercise
    # the whole wiring (env-var read → Redis client construction →
    # RedisStreamsBroker(client)) without a real Redis server.
    fakeredis = pytest.importorskip("fakeredis.aioredis")
    from redis import asyncio as redis_asyncio

    from evaluation.redis_broker import RedisStreamsBroker

    monkeypatch.setenv("ARENA_BROKER_BACKEND", "redis")
    monkeypatch.setenv("REDIS_URL", "redis://test-host:6379/0")
    monkeypatch.setattr(
        redis_asyncio.Redis,
        "from_url",
        lambda _url, **_kw: fakeredis.FakeRedis(),
    )

    broker = make_broker()
    assert isinstance(broker, RedisStreamsBroker)
    # Structural duck-type check — `EventBroker` is a Protocol but not
    # `@runtime_checkable`, so `isinstance(broker, EventBroker)` would
    # itself raise. We verify the surface API at the attribute level:
    # any callsite that needs an `EventBroker` calls these methods, and
    # if the factory ever returned something missing one, the assert
    # would fire here instead of as a confusing `AttributeError` in prod.
    for method in ("open_run", "close_run", "is_open", "publish", "stream", "reap"):
        assert callable(getattr(broker, method)), f"RedisStreamsBroker is missing {method!r}"


def test_make_broker_unknown_backend_raises_value_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ARENA_BROKER_BACKEND", "kafka")
    with pytest.raises(ValueError, match="unknown ARENA_BROKER_BACKEND='kafka'"):
        make_broker()


def test_make_broker_unknown_backend_lists_valid_choices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The error message is part of the operator-facing contract — make sure
    # it actually names the valid options so a misconfigured deploy can be
    # self-corrected from the traceback alone.
    monkeypatch.setenv("ARENA_BROKER_BACKEND", "bogus")
    with pytest.raises(ValueError, match=r"'inprocess'.*'redis'"):
        make_broker()
