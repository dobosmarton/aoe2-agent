"""A purchase is settled by its own evidence, and never blocks rebuilding forever."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gameplay_agent import executor as ex

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[float]]:
    now = [1000.0]
    monkeypatch.setattr(ex, "_now", lambda: now[0])
    ledger = ex.ActionLedger()
    token = ex.bind_ledger(ledger)
    try:
        yield now
    finally:
        ex.unbind_ledger(token)


def _house(placed_at: float) -> ex._PendingPlacement:
    return ex._PendingPlacement(
        building_class="house",
        wood_cost=25,
        wood_before=100,
        noted_at_snapshot=1,
        operation_id=6,
        cap_before=5,
        settle_deadline=placed_at + ex._HOUSE_SETTLE_SECONDS,
    )


def _statuses() -> list[tuple[str, str]]:
    return [(o.action, o.status) for o in ex.current_ledger().outcomes]


def test_a_drop_before_the_deadline_pays_for_the_house(clock: list[float]) -> None:
    ex.current_ledger().pending_placements = [_house(clock[0])]
    clock[0] += 5
    ex._settle_pending_placements(70, 5)
    assert ("build_house", "purchased") in _statuses()
    assert "house" in ex.current_ledger().building_purchases


def test_a_later_purchases_spend_does_not_pay_for_an_unpaid_house(clock: list[float]) -> None:
    # Live: a house placed at 100 wood never spent; minutes later a lumber camp
    # took wood from 170 to 70, and 100 - 70 >= 25 "confirmed" the house.
    ex.current_ledger().pending_placements = [_house(clock[0])]
    for _ in range(ex._PLACEMENT_WOOD_READINGS):
        clock[0] += 0.5
        ex._settle_pending_placements(100, 5)  # Fresh readings, no payment.
    ex._settle_pending_placements(70, 5)
    assert ("build_house", "purchased") not in _statuses()
    assert "house" not in ex.current_ledger().building_purchases


def test_a_first_reading_after_a_slow_refresh_still_pays(clock: list[float]) -> None:
    ex.current_ledger().pending_placements = [_house(clock[0])]
    clock[0] += ex._HOUSE_SETTLE_SECONDS + 5  # Perception stalled, then caught up.
    ex._settle_pending_placements(70, 5)
    assert ("build_house", "purchased") in _statuses()


def test_an_unseen_placement_is_abandoned_so_it_can_be_retried(clock: list[float]) -> None:
    ex.current_ledger().pending_placements = [_house(clock[0])]
    clock[0] += ex._HOUSE_SETTLE_SECONDS + ex._PLACEMENT_ABANDON_SECONDS + 1
    ex._settle_pending_placements(100, 5)
    assert ex.current_ledger().pending_placements == []
    assert ("build_house", "failed") in _statuses()


def test_a_house_completed_late_is_still_confirmed(clock: list[float]) -> None:
    ex.current_ledger().pending_placements = [_house(clock[0])]
    clock[0] += ex._HOUSE_SETTLE_SECONDS + 10
    ex._settle_pending_placements(100, 10)  # The cap rose by a whole house.
    assert ("build_house", "confirmed") in _statuses()


def test_a_purchase_never_completed_stops_blocking_a_rebuild(clock: list[float]) -> None:
    ledger = ex.current_ledger()
    ledger.building_purchases["house"] = ex._PurchasedBuilding(6, frozenset(), 5, clock[0])
    clock[0] += ex._HOUSE_COMPLETION_SECONDS - 1
    ex._expire_unfinished_purchases()
    assert "house" in ledger.building_purchases
    clock[0] += 2
    ex._expire_unfinished_purchases()
    assert "house" not in ledger.building_purchases
    assert ("build_house", "failed") in _statuses()


def test_other_buildings_get_longer_to_complete(clock: list[float]) -> None:
    ledger = ex.current_ledger()
    ledger.building_purchases["mill"] = ex._PurchasedBuilding(7, frozenset(), 0, clock[0])
    clock[0] += ex._HOUSE_COMPLETION_SECONDS + 1
    ex._expire_unfinished_purchases()
    assert "mill" in ledger.building_purchases
