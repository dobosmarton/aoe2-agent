"""The enlargement test's crops and scoring. Running it needs the models and the test set."""

from __future__ import annotations

import math
import sys
from itertools import pairwise
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from detector_scale_probe import (
    Guess,
    Shown,
    Tally,
    Truth,
    bootstrap_gap,
    crops_per_axis,
    score,
    scored_somewhere,
    shown_in,
    tile_frame,
)

WIDTH, HEIGHT = 3024, 1672
EVERYWHERE = (-math.inf, -math.inf, math.inf, math.inf)
SCALES = (1.25, 1.5, 1.7, 2.0)


@pytest.mark.parametrize(("scale", "count"), [(1.25, 2), (1.5, 2), (1.7, 3), (2.0, 3)])
def test_crop_count_per_axis(scale: float, count: int) -> None:
    assert crops_per_axis(scale) == count


@pytest.mark.parametrize("scale", SCALES)
def test_crops_have_the_scaled_size_and_overlap_by_half(scale: float) -> None:
    tiles = tile_frame(WIDTH, HEIGHT, scale)
    for tile in tiles:
        assert tile.right - tile.left == round(WIDTH / scale)
        assert tile.bottom - tile.top == round(HEIGHT / scale)
        assert tile.left >= 0 and tile.right <= WIDTH
        assert tile.top >= 0 and tile.bottom <= HEIGHT
    lefts = sorted({tile.left for tile in tiles})
    assert lefts[0] == 0 and lefts[-1] + round(WIDTH / scale) == WIDTH
    assert max(b - a for a, b in pairwise(lefts)) <= round(WIDTH / scale) / 2 + 1


@pytest.mark.parametrize("scale", SCALES)
def test_cores_split_the_frame_and_sit_inside_their_crop(scale: float) -> None:
    tiles = tile_frame(WIDTH, HEIGHT, scale)
    for x in range(0, WIDTH, 37):
        for y in range(0, HEIGHT, 41):
            owners = [t for t in tiles if t.core[0] <= x < t.core[2] and t.core[1] <= y < t.core[3]]
            assert len(owners) == 1
            owner = owners[0]
            assert owner.left <= x < owner.right and owner.top <= y < owner.bottom


def test_scale_one_is_the_whole_frame() -> None:
    (tile,) = tile_frame(WIDTH, HEIGHT, 1.0)
    assert (tile.left, tile.top, tile.right, tile.bottom) == (0, 0, WIDTH, HEIGHT)
    assert tile.core == EVERYWHERE


def test_a_box_cut_by_an_inner_crop_edge_is_shown_clipped_and_not_scored() -> None:
    (tile,) = [t for t in tile_frame(WIDTH, HEIGHT, 2.0) if t.left == 0 and t.top == 0]
    truth = Truth(3, (tile.right - 20.0, 100.0, tile.right + 20.0, 140.0))
    (shown,) = shown_in([truth], tile, WIDTH, HEIGHT)
    assert shown.box == (tile.right - 20.0, 100.0, float(tile.right), 140.0)
    assert not shown.scored


def test_a_box_past_the_frame_edge_is_not_cut() -> None:
    (tile,) = [t for t in tile_frame(WIDTH, HEIGHT, 2.0) if t.left == 0 and t.top == 0]
    truth = Truth(3, (-5.0, 100.0, 30.0, 140.0))
    (shown,) = shown_in([truth], tile, WIDTH, HEIGHT)
    assert shown.box == truth.box
    assert shown.scored


def test_every_small_box_is_scored_by_exactly_one_crop() -> None:
    truths = [
        Truth(0, (x, y, x + 120.0, y + 90.0))
        for x in range(0, WIDTH - 120, 97)
        for y in range(0, HEIGHT - 90, 83)
    ]
    for scale in SCALES:
        tiles = tile_frame(WIDTH, HEIGHT, scale)
        for truth in truths:
            scoring = [
                s for tile in tiles for s in shown_in([truth], tile, WIDTH, HEIGHT) if s.scored
            ]
            assert len(scoring) == 1
        assert all(scored_somewhere(truths, tiles, WIDTH, HEIGHT))


def test_a_box_too_large_for_its_crop_is_never_scored() -> None:
    tiles = tile_frame(WIDTH, HEIGHT, 2.0)
    # Centred in the first crop's core, wider than any crop's reach.
    huge = Truth(0, (300.0, 200.0, 1700.0, 800.0))
    assert scored_somewhere([huge], tiles, WIDTH, HEIGHT) == [False]


def test_scoring_hits_misses_and_false_guesses() -> None:
    shown = [Shown(1, (0.0, 0.0, 10.0, 10.0), True), Shown(2, (20.0, 20.0, 30.0, 30.0), True)]
    guesses = [Guess(1, (0.0, 0.0, 10.0, 9.0), 0.9), Guess(1, (50.0, 50.0, 60.0, 60.0), 0.8)]
    tally = score(guesses, shown, EVERYWHERE)
    assert tally.micro() == (1, 1, 1)
    assert tally.missed == {2: 1}


def test_the_most_confident_guess_takes_the_truth() -> None:
    shown = [Shown(1, (0.0, 0.0, 10.0, 10.0), True)]
    guesses = [Guess(1, (0.0, 0.0, 10.0, 10.0), 0.4), Guess(1, (0.0, 0.0, 10.0, 8.0), 0.9)]
    assert score(guesses, shown, EVERYWHERE).micro() == (1, 1, 0)


def test_guesses_on_ignored_truths_count_nowhere() -> None:
    ignored = Shown(1, (0.0, 0.0, 10.0, 10.0), False)
    matched = Guess(1, (0.0, 0.0, 10.0, 10.0), 0.9)
    wrong_class_on_top = Guess(2, (1.0, 1.0, 9.0, 9.0), 0.8)
    assert score([matched, wrong_class_on_top], [ignored], EVERYWHERE).micro() == (0, 0, 0)


def test_unmatched_guesses_outside_the_core_count_nowhere() -> None:
    core = (0.0, 0.0, 100.0, 100.0)
    inside, outside = (
        Guess(1, (10.0, 10.0, 20.0, 20.0), 0.9),
        Guess(1, (110.0, 10.0, 120.0, 20.0), 0.9),
    )
    assert score([inside, outside], [], core).micro() == (0, 1, 0)


def _frame(hits: int, false: int, missed: int) -> Tally:
    tally = Tally()
    tally.hits[0], tally.false[0], tally.missed[0] = hits, false, missed
    return tally


def test_bootstrap_of_identical_runs_is_zero() -> None:
    frames = [_frame(5, 1, 2), _frame(3, 0, 4), _frame(8, 2, 1)]
    assert bootstrap_gap(frames, frames, rounds=200) == (0.0, 0.0)


def test_bootstrap_brackets_a_uniform_gain() -> None:
    before = [_frame(5, 5, 5) for _ in range(10)]
    after = [_frame(9, 1, 1) for _ in range(10)]
    low, high = bootstrap_gap(before, after, rounds=200)
    assert low == pytest.approx(high) == pytest.approx(0.9 - 0.5)
