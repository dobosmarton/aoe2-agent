"""Unit tests for the D2 dataset-level rebalancing of the synthetic generator.

Pure-logic tests — no sprites, no image generation. They pin the two helpers
(`effective_count_range`, `scale_bounds`) and the per-class policy baked into
`SPRITE_CONFIGS`.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from detection.training.generate_training_data import (
    SPRITE_CONFIGS,
    SpriteConfig,
    effective_count_range,
    scale_bounds,
)


def _config(**overrides: object) -> SpriteConfig:
    base: dict[str, object] = {
        "class_id": 99,
        "class_name": "test",
        "sprite_patterns": [],
    }
    base.update(overrides)
    return SpriteConfig(**base)  # type: ignore[arg-type]  # test-only kwargs splat


class TestEffectiveCountRange:
    def test_unit_weight_leaves_range_unchanged(self) -> None:
        config = _config(count_range=(4, 12), oversample_weight=1.0)
        assert effective_count_range(config) == (4, 12)

    def test_weight_scales_both_bounds(self) -> None:
        config = _config(count_range=(1, 3), oversample_weight=2.0)
        assert effective_count_range(config) == (2, 6)

    def test_fractional_weight_rounds(self) -> None:
        config = _config(count_range=(0, 1), oversample_weight=2.5)
        assert effective_count_range(config) == (0, 2)


class TestScaleBounds:
    def test_normal_draw_uses_scale_range(self) -> None:
        config = _config(scale_range=(0.8, 1.2), distant_scale_range=(0.18, 0.30))
        assert scale_bounds(config, distant=False) == (0.8, 1.2)

    def test_distant_draw_uses_distant_range(self) -> None:
        config = _config(scale_range=(0.8, 1.2), distant_scale_range=(0.18, 0.30))
        assert scale_bounds(config, distant=True) == (0.18, 0.30)


class TestRebalancePolicy:
    """Policy applied to the shipped `SPRITE_CONFIGS`."""

    def test_rare_unique_is_oversampled_and_distant(self) -> None:
        unique_cavalry = next(c for c in SPRITE_CONFIGS if c.class_id == 51)
        assert unique_cavalry.oversample_weight == 2.5
        assert unique_cavalry.distant_fraction == 0.25

    def test_plain_unit_is_distant_but_not_oversampled(self) -> None:
        villager = next(c for c in SPRITE_CONFIGS if c.class_id == 30)
        assert villager.oversample_weight == 1.0
        assert villager.distant_fraction == 0.25

    def test_building_is_neither_oversampled_nor_distant(self) -> None:
        house = next(c for c in SPRITE_CONFIGS if c.class_id == 10)
        assert house.oversample_weight == 1.0
        assert house.distant_fraction == 0.0


def test_sprite_config_is_immutable() -> None:
    config = _config()
    with pytest.raises(FrozenInstanceError):
        config.oversample_weight = 5.0  # type: ignore[misc]  # asserting immutability
