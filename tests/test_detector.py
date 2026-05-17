"""Smoke tests for `EntityDetector` in mock mode.

Mock mode skips model loading entirely (`use_mock=True` short-circuits
`_load_model`) and `_mock_detect` seeds `random` with 42, so the same
screenshot dimensions always produce the same detections. This lets us
exercise the public surface without YOLO weights or the `ultralytics`
dependency.

These are smoke tests — coverage gate before refactoring detector.py into
postprocess/mock/sahi sub-modules. They guard call shape and ID semantics,
not numerical accuracy of inference.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

import detection.inference.detector as detector_mod
from detection.inference.detector import DetectedEntity, EntityDetector, get_detector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_screenshot(width: int = 1920, height: int = 1080) -> bytes:
    """Return JPEG-encoded bytes for a solid-color image."""
    img = Image.new("RGB", (width, height), color=(20, 30, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    return buf.getvalue()


def _make_image(width: int = 1920, height: int = 1080) -> Image.Image:
    return Image.new("RGB", (width, height), color=(20, 30, 40))


@pytest.fixture
def det() -> EntityDetector:
    """Detector in mock mode with the Kalman tracker disabled.

    Disabling the tracker forces ID assignment through
    `_assign_persistent_ids`, which has stable, predictable semantics.
    """
    d = EntityDetector(use_mock=True)
    d.tracker = None
    return d


@pytest.fixture(autouse=True)
def _reset_singleton() -> None:
    """Wipe `get_detector`'s module-level singleton between tests."""
    detector_mod._instance = None


# ---------------------------------------------------------------------------
# DetectedEntity dataclass
# ---------------------------------------------------------------------------


def test_detected_entity_to_dict_shape():
    e = DetectedEntity(
        id="sheep_0",
        class_name="sheep",
        bbox=(10.0, 20.0, 30.0, 40.0),
        center=(20.0, 30.0),
        confidence=0.85,
        area=400,
    )
    d = e.to_dict()
    assert d == {
        "id": "sheep_0",
        "class": "sheep",
        "bbox": [10.0, 20.0, 30.0, 40.0],
        "center": (20.0, 30.0),
        "confidence": 0.85,
    }


# ---------------------------------------------------------------------------
# Constructor / module loading
# ---------------------------------------------------------------------------


def test_init_mock_mode_does_not_load_model():
    d = EntityDetector(use_mock=True)
    assert d.use_mock is True
    assert d.model is None
    assert d.onnx_session is None
    assert d.backend is None


def test_init_default_class_thresholds_present():
    d = EntityDetector(use_mock=True)
    # CLASS_THRESHOLDS is dict-like with at least the threshold for some classes
    assert isinstance(d.class_thresholds, dict)
    assert len(d.class_thresholds) > 0


# ---------------------------------------------------------------------------
# Detection methods (mock mode)
# ---------------------------------------------------------------------------


def test_detect_returns_entities(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    assert len(entities) > 0
    assert all(isinstance(e, DetectedEntity) for e in entities)
    assert all(0.0 <= e.confidence <= 1.0 for e in entities)


def test_detect_includes_town_center(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    classes = {e.class_name for e in entities}
    assert "town_center" in classes


def test_detect_accepts_pil_image(det: EntityDetector):
    entities = det.detect(_make_image())
    assert len(entities) > 0


def test_detect_fast_returns_entities(det: EntityDetector):
    entities = det.detect_fast(_make_screenshot())
    assert len(entities) > 0


def test_detect_fast_multi_returns_entities(det: EntityDetector):
    entities = det.detect_fast_multi(_make_screenshot())
    assert len(entities) > 0


def test_detect_adaptive_force_full_falls_back_to_detect(det: EntityDetector):
    forced = det.detect_adaptive(_make_screenshot(), force_full=True)
    assert len(forced) > 0


def test_detect_adaptive_no_prior_state_falls_back_to_detect(det: EntityDetector):
    # _previous_entities is empty on a fresh detector → adaptive path defers to detect()
    assert det._previous_entities == []
    entities = det.detect_adaptive(_make_screenshot())
    assert len(entities) > 0


def test_detect_to_dict_list_shape(det: EntityDetector):
    rows = det.detect_to_dict_list(_make_screenshot())
    assert len(rows) > 0
    sample = rows[0]
    assert set(sample.keys()) == {"id", "class", "bbox", "center", "confidence"}


def test_detect_handles_small_screenshot(det: EntityDetector):
    # mock_detect places entities relative to image dimensions
    entities = det.detect(_make_screenshot(width=320, height=240))
    assert len(entities) > 0


# ---------------------------------------------------------------------------
# NMS / IDs
# ---------------------------------------------------------------------------


def test_detect_assigns_unique_ids(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    ids = [e.id for e in entities]
    assert len(ids) == len(set(ids))


def test_detect_ids_are_strings(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    assert all(isinstance(e.id, str) and e.id for e in entities)


def test_persistent_ids_carry_forward_overlapping_bboxes(det: EntityDetector):
    """Same-class overlapping bboxes preserve the previous-frame ID."""
    first = [
        DetectedEntity(
            id="placeholder",
            class_name="sheep",
            bbox=(100, 100, 140, 130),
            center=(120, 115),
            confidence=0.9,
        )
    ]
    second = [
        DetectedEntity(
            id="placeholder",
            class_name="sheep",
            bbox=(102, 101, 142, 132),  # >40% IoU with first
            center=(122, 116),
            confidence=0.92,
        )
    ]
    det._assign_persistent_ids(first)
    first_id = first[0].id
    det._assign_persistent_ids(second)
    assert second[0].id == first_id


def test_persistent_ids_assign_fresh_id_when_no_overlap(det: EntityDetector):
    first = [
        DetectedEntity(
            id="placeholder",
            class_name="sheep",
            bbox=(100, 100, 140, 130),
            center=(120, 115),
            confidence=0.9,
        )
    ]
    second = [
        DetectedEntity(
            id="placeholder",
            class_name="sheep",
            bbox=(800, 800, 840, 830),  # no overlap with first
            center=(820, 815),
            confidence=0.85,
        )
    ]
    det._assign_persistent_ids(first)
    det._assign_persistent_ids(second)
    assert second[0].id != first[0].id


def test_persistent_ids_do_not_share_across_classes(det: EntityDetector):
    """A villager that overlaps a previous sheep should get a fresh ID, not the sheep's."""
    first = [
        DetectedEntity(
            id="placeholder",
            class_name="sheep",
            bbox=(100, 100, 140, 130),
            center=(120, 115),
            confidence=0.9,
        )
    ]
    second = [
        DetectedEntity(
            id="placeholder",
            class_name="villager",
            bbox=(102, 101, 142, 132),
            center=(122, 116),
            confidence=0.9,
        )
    ]
    det._assign_persistent_ids(first)
    det._assign_persistent_ids(second)
    assert second[0].id.startswith("villager_")


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def test_find_entity_by_id_hit(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    target = entities[0]
    found = det.find_entity_by_id(entities, target.id)
    assert found is target


def test_find_entity_by_id_miss(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    found = det.find_entity_by_id(entities, "definitely_not_an_id")
    assert found is None


def test_find_entities_by_class_filters_correctly(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    sheep = det.find_entities_by_class(entities, "sheep")
    assert all(e.class_name == "sheep" for e in sheep)


def test_find_entities_by_class_sorts_by_confidence_desc(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    sheep = det.find_entities_by_class(entities, "sheep")
    if len(sheep) >= 2:
        confidences = [e.confidence for e in sheep]
        assert confidences == sorted(confidences, reverse=True)


def test_find_entities_by_class_empty_when_no_match(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    nothing = det.find_entities_by_class(entities, "totally_made_up_class")
    assert nothing == []


def test_find_nearest_entity_picks_closest(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    # Pick a target point right on the first entity's center
    target = entities[0]
    nearest = det.find_nearest_entity(entities, target.center)
    assert nearest is target


def test_find_nearest_entity_with_class_filter(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    nearest_sheep = det.find_nearest_entity(entities, (960, 540), class_filter="sheep")
    if nearest_sheep is not None:
        assert nearest_sheep.class_name == "sheep"


def test_find_nearest_entity_returns_none_for_empty_list(det: EntityDetector):
    assert det.find_nearest_entity([], (100, 100)) is None


def test_find_nearest_entity_returns_none_when_class_filter_misses(det: EntityDetector):
    entities = det.detect(_make_screenshot())
    assert det.find_nearest_entity(entities, (100, 100), class_filter="not_a_class") is None


# ---------------------------------------------------------------------------
# Singleton factory
# ---------------------------------------------------------------------------


def test_get_detector_returns_entity_detector():
    d = get_detector(use_mock=True)
    assert isinstance(d, EntityDetector)
    assert d.use_mock is True


def test_get_detector_returns_singleton():
    a = get_detector(use_mock=True)
    b = get_detector(use_mock=True)
    assert a is b


# ---------------------------------------------------------------------------
# Schema contract: synthetic render must match real DetectedEntity schema.
#
# Risk 4 mitigation per docs/design/synthetic-arena-analysis.md line 380.
# Parametrized over both `mock_detect` (existing) and `mock_detect_from_world`
# (Phase 1, new) so any schema drift between the two surfaces fails loudly.
# ---------------------------------------------------------------------------


_CONTRACT_SCREENSHOT_WIDTH = 1920
_CONTRACT_SCREENSHOT_HEIGHT = 1080


def _build_id_factory():
    counters: dict[str, int] = {}

    def make_id(class_name: str) -> str:
        index = counters.get(class_name, 0)
        counters[class_name] = index + 1
        return f"{class_name}_{index}"

    return make_id


def _dark_age_world_state():
    from evaluation.world_sim import WorldState

    return WorldState(
        food=200.0,
        wood=200.0,
        gold=0.0,
        stone=0.0,
        population=5,
        pop_cap=25,
        age="Dark Age",
        buildings=["mill"],
        villager_queue=[],
        age_up_ticks_remaining=0,
        turn=0,
    )


@pytest.fixture
def contract_screenshot() -> bytes:
    return _make_screenshot(_CONTRACT_SCREENSHOT_WIDTH, _CONTRACT_SCREENSHOT_HEIGHT)


@pytest.fixture(params=["mock_detect", "mock_detect_from_world"])
def contract_entities(request, contract_screenshot) -> list[DetectedEntity]:
    from detection.inference.mock import mock_detect, mock_detect_from_world

    if request.param == "mock_detect":
        return mock_detect(contract_screenshot, _build_id_factory())
    return mock_detect_from_world(contract_screenshot, _build_id_factory(), _dark_age_world_state())


def test_every_id_is_non_empty_string(contract_entities):
    bad = [e for e in contract_entities if not (isinstance(e.id, str) and e.id)]
    assert bad == []


def test_every_class_name_is_in_canonical_classlist(contract_entities):
    from detection.inference.detector import _load_default_classes

    canonical = set(_load_default_classes())
    unknown = [e.class_name for e in contract_entities if e.class_name not in canonical]
    assert unknown == []


def test_every_bbox_is_well_ordered(contract_entities):
    malformed = [
        e for e in contract_entities if not (e.bbox[0] < e.bbox[2] and e.bbox[1] < e.bbox[3])
    ]
    assert malformed == []


def test_every_bbox_is_within_image_dims(contract_entities):
    out_of_bounds = [
        e
        for e in contract_entities
        if not (
            0 <= e.bbox[0] <= _CONTRACT_SCREENSHOT_WIDTH
            and 0 <= e.bbox[2] <= _CONTRACT_SCREENSHOT_WIDTH
        )
        or not (
            0 <= e.bbox[1] <= _CONTRACT_SCREENSHOT_HEIGHT
            and 0 <= e.bbox[3] <= _CONTRACT_SCREENSHOT_HEIGHT
        )
    ]
    assert out_of_bounds == []


def test_every_center_is_inside_its_bbox(contract_entities):
    bad = [
        e
        for e in contract_entities
        if not (e.bbox[0] <= e.center[0] <= e.bbox[2] and e.bbox[1] <= e.center[1] <= e.bbox[3])
    ]
    assert bad == []


def test_every_confidence_is_in_unit_interval(contract_entities):
    bad = [e for e in contract_entities if not (0.0 <= e.confidence <= 1.0)]
    assert bad == []


def test_every_area_is_positive(contract_entities):
    bad = [e for e in contract_entities if not e.area > 0]
    assert bad == []


def test_every_to_dict_has_canonical_keys(contract_entities):
    canonical = {"id", "class", "bbox", "center", "confidence"}
    mismatched = [e for e in contract_entities if set(e.to_dict()) != canonical]
    assert mismatched == []


def test_output_is_sorted_by_class_then_confidence(contract_entities):
    sort_keys = [(e.class_name, -e.confidence) for e in contract_entities]
    assert sort_keys == sorted(sort_keys)


def test_all_ids_are_unique(contract_entities):
    ids = [e.id for e in contract_entities]
    assert len(set(ids)) == len(ids)


def test_population_15_yields_seven_more_villagers_than_population_8(contract_screenshot):
    from dataclasses import replace

    from detection.inference.mock import mock_detect_from_world

    base = _dark_age_world_state()
    state_8 = replace(base, population=8)
    state_15 = replace(base, population=15)

    entities_8 = mock_detect_from_world(contract_screenshot, _build_id_factory(), state_8)
    entities_15 = mock_detect_from_world(contract_screenshot, _build_id_factory(), state_15)

    villagers_8 = sum(1 for e in entities_8 if e.class_name == "villager")
    villagers_15 = sum(1 for e in entities_15 if e.class_name == "villager")
    assert villagers_15 - villagers_8 == 7
