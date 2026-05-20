"""Tests for arena/prompts.py — variant registry sanity checks."""

from __future__ import annotations

import pytest

from arena.prompts import PROMPTS, get_prompt


def test_all_expected_variants_registered() -> None:
    assert set(PROMPTS) == {"bare", "strategy"}


def test_get_prompt_returns_variant_content() -> None:
    assert get_prompt("strategy").startswith("You are an Age of Empires 2 economy strategist.")


def test_get_prompt_raises_on_unknown_variant() -> None:
    with pytest.raises(KeyError):
        get_prompt("nonexistent")


def test_bare_omits_strategy_notes() -> None:
    assert "FOUR prerequisites" not in get_prompt("bare")


def test_strategy_includes_prereq_enumeration() -> None:
    assert "FOUR prerequisites" in get_prompt("strategy")
