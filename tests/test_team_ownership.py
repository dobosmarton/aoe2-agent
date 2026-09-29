"""A team roster, not non-blue color, establishes hostility."""

import numpy as np
from detection.inference.ownership import Owner, classify_entity


def _red_unit() -> np.ndarray:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[20:35, 20:60] = (245, 20, 20)
    return image


def test_unqualified_nonblue_unit_remains_unknown() -> None:
    owner, _confidence = classify_entity(_red_unit(), (20, 20, 60, 60))
    assert owner is Owner.UNKNOWN


def test_roster_maps_same_color_to_allied_or_enemy_relationship() -> None:
    image = _red_unit()
    allied, _ = classify_entity(image, (20, 20, 60, 60), {"red": Owner.ALLY})
    enemy, _ = classify_entity(image, (20, 20, 60, 60), {"red": Owner.ENEMY})
    assert allied is Owner.ALLY
    assert enemy is Owner.ENEMY
