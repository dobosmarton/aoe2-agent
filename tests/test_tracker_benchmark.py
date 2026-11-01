"""The benchmark's metrics, against hand-computed answers and scenes with known outcomes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from gameplay_agent.entity_snapshot import EntitySnapshot

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "tracker_benchmark"))

import contenders
import scorecard
import sequences
from scorecard import ScoredFrame, TruthObject

_BOX = (0.0, 0.0, 10.0, 10.0)


def _prediction(identifier: str, label: str = "villager") -> EntitySnapshot:
    return EntitySnapshot(identifier, label, _BOX, (5.0, 5.0), 0.8)


def _frame(at: float, found: dict[str, str], *, visible: tuple[str, ...] = ()) -> ScoredFrame:
    """A frame where each truth object in `found` was detected under the given ID."""
    predictions = {truth: _prediction(identifier) for truth, identifier in found.items()}
    names = {*found, *visible}
    truths = tuple(TruthObject(name, "villager", _BOX) for name in sorted(names))
    return ScoredFrame(at, tuple(predictions.values()), truths, predictions)


def test_idf1_credits_each_object_with_its_best_id() -> None:
    frames = [
        _frame(0.0, {"a": "x", "b": "z"}),
        _frame(0.2, {"a": "x", "b": "z"}),
        _frame(0.4, {"a": "x"}),
        _frame(0.6, {"a": "y"}),
    ]
    assert scorecard.idf1(frames) == pytest.approx(5 / 6)


def test_idf1_cannot_credit_one_id_to_two_objects() -> None:
    frames = [_frame(0.0, {"a": "p"}), _frame(0.2, {"a": "p"}), _frame(0.4, {"b": "p"})]
    assert scorecard.idf1(frames) == pytest.approx(2 / 3)


def test_idf1_is_zero_when_nothing_was_found() -> None:
    assert scorecard.idf1([_frame(0.0, {}, visible=("a",))]) == 0.0


def test_id_switches_are_per_thousand_consecutive_pairs() -> None:
    frames = [_frame(0.0, {"a": "x"}), _frame(0.2, {"a": "x"}), _frame(0.4, {"a": "y"})]
    assert scorecard.id_switches(frames) == pytest.approx(500.0)


def test_a_missed_detection_is_not_a_switch() -> None:
    frames = [_frame(0.0, {"a": "x"}), _frame(0.2, {}), _frame(0.4, {"a": "y"})]
    assert scorecard.id_switches(frames) == 0.0


def test_target_survives_when_the_id_still_names_the_same_object() -> None:
    frames = [_frame(0.0, {"a": "x"}), _frame(1.0, {"a": "x"})]
    outcome = scorecard.target_outcomes(frames, 1.0)
    assert (outcome.survived, outcome.wrong, outcome.lost) == (1, 0, 0)


def test_target_is_lost_when_the_id_disappears() -> None:
    frames = [_frame(0.0, {"a": "x"}), _frame(1.0, {"a": "y"})]
    outcome = scorecard.target_outcomes(frames, 1.0)
    assert (outcome.survived, outcome.wrong, outcome.lost) == (0, 0, 1)


def test_target_is_wrong_when_the_id_moved_to_another_object() -> None:
    frames = [_frame(0.0, {"a": "x", "b": "y"}), _frame(1.0, {"a": "y", "b": "x"})]
    outcome = scorecard.target_outcomes(frames, 1.0)
    assert (outcome.survived, outcome.wrong, outcome.lost) == (0, 2, 0)


def test_target_ignores_objects_the_detector_no_longer_finds() -> None:
    frames = [_frame(0.0, {"a": "x"}), _frame(1.0, {}, visible=("a",))]
    assert scorecard.target_outcomes(frames, 1.0).total == 0


def test_target_only_counts_entities_the_llm_saw() -> None:
    found = {f"o{index:02d}": f"id{index}" for index in range(25)}
    frames = [_frame(0.0, found), _frame(1.0, found)]
    assert scorecard.target_outcomes(frames, 1.0).total == scorecard.SUMMARY_LIMIT


def test_targets_split_mobile_units_from_static_objects() -> None:
    tree = TruthObject("t", "tree", _BOX)
    prediction = _prediction("x", "tree")
    frames = [ScoredFrame(at, (prediction,), (tree,), {"t": prediction}) for at in (0.0, 1.0)]
    assert scorecard.target_outcomes(frames, 1.0, mobile=False).total == 1
    assert scorecard.target_outcomes(frames, 1.0, mobile=True).total == 0


def test_label_scores_count_accuracy_and_flips() -> None:
    wrong = _prediction("x", "sheep")
    right = _prediction("x", "villager")
    truth = (TruthObject("a", "villager", _BOX),)
    frames = [
        ScoredFrame(0.0, (right,), truth, {"a": right}),
        ScoredFrame(0.2, (wrong,), truth, {"a": wrong}),
        ScoredFrame(0.4, (right,), truth, {"a": right}),
        ScoredFrame(0.6, (right,), truth, {"a": right}),
    ]
    accuracy, flips = scorecard.label_scores(frames)
    assert accuracy == pytest.approx(0.75)
    assert flips == pytest.approx(2 / 3 * 1000)


def test_recovery_is_the_time_until_an_id_holds() -> None:
    ids = ["a", "b", "c", "d", "d", "d", "d", "d"]
    frames = [_frame(index * 0.2, {"o": name}) for index, name in enumerate(ids)]
    assert scorecard.recovery_seconds(frames, "o", 0.2) == pytest.approx(0.4)


def test_recovery_is_none_when_identity_never_holds() -> None:
    frames = [_frame(index * 0.2, {"o": f"id{index}"}) for index in range(8)]
    assert scorecard.recovery_seconds(frames, "o", 0.0) is None


def test_matching_is_one_to_one_and_best_first() -> None:
    near = EntitySnapshot("n", "v", (0.0, 0.0, 10.0, 10.0), (5.0, 5.0), 0.8)
    far = EntitySnapshot("f", "v", (2.0, 0.0, 12.0, 10.0), (7.0, 5.0), 0.8)
    truths = (TruthObject("a", "v", (0.0, 0.0, 10.0, 10.0)),)
    assert scorecard.match_frame((far, near), truths) == {"a": near}


def test_first_difference_tolerates_rounding_but_not_decisions() -> None:
    same = EntitySnapshot("a", "v", _BOX, (5.0, 5.0), 0.8, class_agreement=0.5)
    close = EntitySnapshot("a", "v", _BOX, (5.0, 5.0), 0.8, class_agreement=0.5 + 1e-12)
    other = EntitySnapshot("b", "v", _BOX, (5.0, 5.0), 0.8, class_agreement=0.5)
    assert scorecard.first_difference((same,), (close,)) is None
    assert scorecard.first_difference((same,), (other,)) == "detection 0 id"
    assert scorecard.first_difference((same,), ()) == "1 versus 0 detections"


def test_bootstrap_recovers_a_known_ratio() -> None:
    slower = [10.0 * (index + 1) for index in range(200)]
    faster = [float(index + 1) for index in range(200)]
    low, estimate, high = scorecard.bootstrap_ratio(slower, faster)
    assert (low, estimate, high) == pytest.approx((10.0, 10.0, 10.0))


def test_a_bootstrap_needs_paired_frames() -> None:
    with pytest.raises(ValueError, match="same frames"):
        scorecard.bootstrap_ratio([1.0, 2.0], [1.0])


def test_tile_width_reads_zoom_from_buildings() -> None:
    farm = sequences.Annotation((0.0, 0.0, 300.0, 200.0), "farm")
    house = sequences.Annotation((0.0, 0.0, 200.0, 150.0), "house")
    assert sequences.tile_width((farm, house)) == pytest.approx(100.0)
    assert sequences.tile_width(()) == sequences.DEFAULT_TILE_WIDTH


def test_hidden_or_hud_covered_objects_leave_the_truth() -> None:
    assert sequences.view_box((10.0, 200.0, 50.0, 260.0), (0, 0), 1000, 1000) is not None
    assert sequences.view_box((10.0, 900.0, 50.0, 960.0), (0, 0), 1000, 1000) is None
    assert sequences.view_box((-60.0, 200.0, 10.0, 260.0), (0, 0), 1000, 1000) is None


@pytest.fixture(scope="module")
def scenes(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    return sequences.write_targeted(tmp_path_factory.mktemp("targeted"))


def _scored(name: str, directory: Path) -> list[ScoredFrame]:
    records, _report = contenders.replay(contenders.contender(name), directory)
    return scorecard.score_frames(records, scorecard.read_truth(directory))


@pytest.mark.parametrize("scene", ["duplicate_box", "occlusion", "crossing", "camera_jump"])
def test_the_oracle_scores_perfectly_on_every_scene(scenes: dict[str, Path], scene: str) -> None:
    frames = _scored("oracle", scenes[scene])
    assert scorecard.idf1(frames) == 1.0
    assert scorecard.target_outcomes(frames, 1.0).survival == 1.0


def test_no_tracker_keeps_no_identity(scenes: dict[str, Path]) -> None:
    frames = _scored("raw", scenes["occlusion"])
    assert scorecard.target_outcomes(frames, 1.0).survival == 0.0


def test_one_duplicate_box_heals_in_the_python_tracker(scenes: dict[str, Path]) -> None:
    frames = _scored("python", scenes["duplicate_box"])
    disturbed = 1.0 + sequences.DUPLICATE_FRAME * sequences.FRAME_SECONDS
    recovered = scorecard.recovery_seconds(frames, "gt_0", disturbed)
    assert recovered is not None
    assert recovered <= 1.0


def test_the_iou_baseline_shrugs_off_a_duplicate_box(scenes: dict[str, Path]) -> None:
    frames = _scored("iou", scenes["duplicate_box"])
    disturbed = 1.0 + sequences.DUPLICATE_FRAME * sequences.FRAME_SECONDS
    assert scorecard.recovery_seconds(frames, "gt_0", disturbed) == 0.0


def test_screentrack_reproduces_the_python_decisions(scenes: dict[str, Path]) -> None:
    pytest.importorskip("screentrack")
    for directory in scenes.values():
        python, _ = contenders.replay(contenders.contender("python"), directory)
        native, _ = contenders.replay(contenders.contender("screentrack"), directory)
        assert len(python) == len(native)
        for left, right in zip(python, native, strict=True):
            assert scorecard.first_difference(left.entities, right.entities) is None
