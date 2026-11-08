"""Composites put sprites at the real game scale on visible ground, never on labels."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
import pytest
from detection.extraction.sld_extractor import direction_frame_indices
from detection.training.generate_composites import (
    Background,
    Group,
    compose,
    crop_to_alpha,
    in_playfield,
    on_visible_ground,
    overlap,
)
from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path


def test_directional_files_are_sampled_in_every_direction() -> None:
    indices = direction_frame_indices(960, 16)

    assert indices is not None
    assert len(indices) == 32
    assert indices[:4] == [0, 30, 60, 90]
    assert indices[-1] == 930


@pytest.mark.parametrize("frame_count", [1, 4, 31])
def test_short_files_keep_the_fixed_frames(frame_count: int) -> None:
    assert direction_frame_indices(frame_count, 16) is None


def test_label_box_is_the_visible_sprite() -> None:
    sprite = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
    sprite.paste((200, 10, 10, 255), (5, 7, 15, 27))

    cropped = crop_to_alpha(sprite)

    assert cropped is not None
    assert cropped.size == (10, 20)


def test_hud_and_unexplored_ground_are_rejected() -> None:
    size = (3024, 1672)
    luminance = np.full((1672, 3024), 120, dtype=np.float32)
    luminance[:, :1000] = 0

    assert not in_playfield((1500, 20, 1530, 60), size)  # resource bar
    assert not in_playfield((1500, 1500, 1530, 1540), size)  # bottom panel
    assert in_playfield((1500, 600, 1530, 640), size)
    assert not on_visible_ground(luminance, (500, 600, 530, 640))
    assert on_visible_ground(luminance, (1500, 600, 1530, 640))


def test_composite_keeps_real_labels_and_pastes_beside_them(tmp_path: Path) -> None:
    image = tmp_path / "frame.jpg"
    label = tmp_path / "frame.txt"
    Image.new("RGB", (3024, 1672), (120, 140, 90)).save(image)
    label.write_text("9 0.500000 0.450000 0.200000 0.300000\n")
    sprite = Image.new("RGBA", (20, 44), (180, 40, 40, 255))
    classes = (*(f"class_{index}" for index in range(30)), "villager")

    canvas, lines, counts = compose(
        Background(image, label, "sha"), {"villager": [sprite]}, classes, random.Random(3), 1.0
    )

    real_box = (1209.6, 501.6, 1814.4, 1003.2)
    pasted = [[float(value) for value in line.split()] for line in lines[1:]]
    assert lines[0].split()[0] == "9"
    assert counts["villager"] == len(pasted) > 0
    for class_id, cx, cy, bw, bh in pasted:
        box = (
            (cx - bw / 2) * 3024,
            (cy - bh / 2) * 1672,
            (cx + bw / 2) * 3024,
            (cy + bh / 2) * 1672,
        )
        assert class_id == 30
        assert overlap(box, real_box) <= 0.15
        assert in_playfield(box, canvas.size)
        assert 20 * 0.92 <= box[2] - box[0] <= 20 * 1.08 * 1.09 + 1


def test_fine_sprite_label_is_written_as_its_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "frame.jpg"
    label = tmp_path / "frame.txt"
    Image.new("RGB", (3024, 1672), (120, 140, 90)).save(image)
    label.write_text("")
    monkeypatch.setattr(
        "detection.training.generate_composites.plan_groups",
        lambda rng, available: [Group("ibex", 4, 1.5)],
    )
    sprite = Image.new("RGBA", (40, 36), (200, 190, 170, 255))

    _, lines, counts = compose(
        Background(image, label, "sha"),
        {"ibex": [sprite]},
        ("tree", "gold_mine", "stone_mine", "berry_bush", "relic", "deer"),
        random.Random(1),
        1.0,
        {"ibex": "deer"},
    )

    assert counts["ibex"] == len(lines) > 0
    assert {line.split()[0] for line in lines} == {"5"}
