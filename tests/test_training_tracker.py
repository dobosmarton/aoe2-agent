"""Ingest + repository + API tests for the training tracker.

A tiny on-disk fixture (2 raw screenshots, one labeled with a duplicated copy,
one synthetic image, a 6-class schema) exercises the whole read path end-to-end.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from core import DetectedEntity
from fastapi.testclient import TestClient
from PIL import Image
from training_api import ingest, prelabel_pending
from training_api.classes import ClassCatalog
from training_api.config import TrackerConfig, load_config
from training_api.db import connect, init_schema
from training_api.domain import ImageFilter
from training_api.geometry import BBox
from training_api.repository import SqliteTrackerRepository

if TYPE_CHECKING:
    from collections.abc import Iterator

_DATASET_DIRNAME = "training_data_test"
_VERSION_NAME = "vtest"
_CLASSES_YAML = "classes:\n" + "".join(f"  - id: {i}\n    name: class_{i}\n" for i in range(6))
# Labeled screenshot has class 0 (once) and class 5 (once); synthetic img has 0 twice.
_REAL_LABEL = "0 0.5 0.5 0.2 0.2\n5 0.25 0.25 0.1 0.1\n"
_SYNTH_LABEL = "0 0.5 0.5 0.1 0.1\n0 0.3 0.3 0.1 0.1\n"


def _write_png(path: Path) -> None:
    Image.new("RGB", (40, 30), color=(10, 20, 30)).save(path)


def _build_fixture(tmp_path: Path) -> TrackerConfig:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _write_png(raw_dir / "Screenshot A.png")  # will be labeled
    _write_png(raw_dir / "Screenshot B.png")  # stays unlabeled

    dataset_dir = tmp_path / "data" / _DATASET_DIRNAME
    for split in ("train", "val"):
        (dataset_dir / split / "images").mkdir(parents=True)
        (dataset_dir / split / "labels").mkdir(parents=True)
    labels = dataset_dir / "train" / "labels"
    (labels / "real_Screenshot A.txt").write_text(_REAL_LABEL)
    (labels / "real_Screenshot A__dup1.txt").write_text(_REAL_LABEL)  # dup ignored
    (labels / "img_00000.txt").write_text(_SYNTH_LABEL)

    classes_yaml = tmp_path / "classes.yaml"
    classes_yaml.write_text(_CLASSES_YAML)

    return TrackerConfig(
        db_path=tmp_path / "tracker.db",
        raw_images_dir=raw_dir,
        dataset_root=tmp_path / "data",
        classes_yaml=classes_yaml,
        thumb_cache_dir=tmp_path / "thumbs",
        cors_origins=("http://localhost:5173",),
    )


@pytest.fixture
def seeded_config(tmp_path: Path) -> TrackerConfig:
    config = _build_fixture(tmp_path)
    ingest.seed(config, dataset_dirname=_DATASET_DIRNAME, version_name=_VERSION_NAME)
    return config


def _repo(config: TrackerConfig) -> SqliteTrackerRepository:
    conn = connect(config.db_path)
    init_schema(conn)
    return SqliteTrackerRepository(conn, ClassCatalog(config.classes_yaml))


# -- ingest ------------------------------------------------------------------


def test_ingest_report_counts(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    report = ingest.seed(config, dataset_dirname=_DATASET_DIRNAME, version_name=_VERSION_NAME)
    assert report.raw_images == 2
    assert report.labeled_images == 1
    assert report.unmatched_dataset_stems == 0
    assert report.synth_images == 1


def test_ingest_is_idempotent(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    ingest.seed(config, dataset_dirname=_DATASET_DIRNAME, version_name=_VERSION_NAME)
    ingest.seed(config, dataset_dirname=_DATASET_DIRNAME, version_name=_VERSION_NAME)
    coverage = _repo(config).coverage()
    assert coverage.total_images == 2  # not duplicated on re-run
    assert coverage.labeled_images == 1


def test_canonical_real_stem_strips_prefix_and_dup() -> None:
    assert ingest.canonical_real_stem("real_Screenshot A__dup6") == "Screenshot A"
    assert ingest.canonical_real_stem("img_00000") is None


# -- repository --------------------------------------------------------------


def test_coverage_matches_fixture(seeded_config: TrackerConfig) -> None:
    coverage = _repo(seeded_config).coverage()
    assert (coverage.total_images, coverage.labeled_images, coverage.unlabeled_images) == (2, 1, 1)
    by_id = {c.class_id: c for c in coverage.classes}
    assert by_id[0].real_instances == 1  # dup copy did not double-count
    assert by_id[0].synth_instances == 2
    assert by_id[5].real_instances == 1
    assert 1 in coverage.zero_real_class_ids  # class 1 never labeled


def test_list_images_labeled_filter(seeded_config: TrackerConfig) -> None:
    repo = _repo(seeded_config)
    labeled = repo.list_images(ImageFilter(labeled=True))
    unlabeled = repo.list_images(ImageFilter(labeled=False))
    assert labeled.total == 1
    assert unlabeled.total == 1
    assert unlabeled.items[0].record.path.endswith("Screenshot B.png")


def test_dataset_detail_splits(seeded_config: TrackerConfig) -> None:
    repo = _repo(seeded_config)
    summary = repo.list_datasets()[0]
    detail = repo.get_dataset(summary.id)
    assert detail is not None
    assert detail.summary.name == _VERSION_NAME
    assert detail.n_train_images == 1
    assert detail.n_val_images == 0


# -- API ---------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    config = _build_fixture(tmp_path)
    env: dict[str, str] = {
        "TRAINING_API_DB": str(config.db_path),
        "TRAINING_API_RAW_IMAGES": str(config.raw_images_dir),
        "TRAINING_API_DATASET_ROOT": str(config.dataset_root),
        "TRAINING_API_CLASSES_YAML": str(config.classes_yaml),
        "TRAINING_API_THUMB_CACHE": str(config.thumb_cache_dir),
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    ingest.seed(load_config(env), dataset_dirname=_DATASET_DIRNAME, version_name=_VERSION_NAME)
    from training_api.server import app

    # Enter the context so FastAPI's lifespan installs repo/catalog on app.state.
    with TestClient(app) as test_client:
        yield test_client


def test_api_classes(client: TestClient) -> None:
    body = client.get("/classes").json()
    assert len(body) == 6
    assert body[0] == {"id": 0, "name": "class_0"}


def test_api_stats(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["total_images"] == 2
    assert body["labeled_images"] == 1
    assert 1 in body["zero_real_class_ids"]


def test_api_images_filter_and_detail(client: TestClient) -> None:
    unlabeled = client.get("/images", params={"labeled": False}).json()
    assert unlabeled["total"] == 1

    labeled = client.get("/images", params={"labeled": True}).json()
    image_id = labeled["items"][0]["image"]["id"]
    detail = client.get(f"/images/{image_id}").json()
    class_ids = {a["class_id"] for a in detail["annotations"]}
    assert class_ids == {0, 5}
    assert detail["annotations"][0]["geom_type"] == "bbox"


def test_api_missing_image_is_404(client: TestClient) -> None:
    assert client.get("/images/9999").status_code == 404


def test_api_images_survives_concurrent_requests(client: TestClient) -> None:
    """Concurrent reads must not share one SQLite connection.

    Routes run their reads through `asyncio.to_thread`, so a process-wide
    connection gets driven by several threadpool threads at once and sqlite3
    raises `SQLITE_MISUSE` from its shared prepared-statement cache. The
    dashboard triggers this on every load (React StrictMode double-mounts).
    """
    with ThreadPoolExecutor(max_workers=8) as pool:
        codes = [
            future.result().status_code
            for future in [pool.submit(client.get, "/images") for _ in range(16)]
        ]
    assert set(codes) == {200}


def test_thumbnail_never_renders_into_the_served_path(
    client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The render must land on a temp file, never on the path being served.

    Two concurrent requests for the same image both see a cache miss and both
    render; if either writes straight to the destination, a reader can be handed
    a half-written JPEG. Asserting *where the bytes are written* tests that
    invariant deterministically — racing the threads instead would only fail on
    an unlucky interleaving, and with a 40x30 fixture it never would.
    """
    written: list[Path] = []
    real_save = Image.Image.save

    def spy(self: Image.Image, fp: object, *args: object, **kwargs: object) -> None:
        written.append(Path(str(fp)))
        real_save(self, fp, *args, **kwargs)  # pyright: ignore[reportUnknownArgumentType]

    monkeypatch.setattr(Image.Image, "save", spy)

    image_id = client.get("/images").json()["items"][0]["image"]["id"]
    thumbs = tmp_path / "thumbs"
    (thumbs / f"{image_id}.jpg").unlink(missing_ok=True)

    response = client.get(f"/thumbs/{image_id}")

    assert response.status_code == 200
    assert response.content.startswith(b"\xff\xd8")  # complete JPEG
    assert written, "the thumbnail was never rendered"
    assert all(p.suffix == ".tmp" for p in written), (
        f"rendered directly into a served path: {written}"
    )
    assert list(thumbs.glob("*.tmp")) == [], "temp file left behind"


def test_api_thumbnail_and_raw(client: TestClient) -> None:
    listing = client.get("/images").json()["items"][0]
    image_id = listing["image"]["id"]

    thumb = client.get(f"/thumbs/{image_id}")
    assert thumb.status_code == 200
    assert thumb.headers["content-type"] == "image/jpeg"
    assert len(thumb.content) > 0
    assert client.get(f"/thumbs/{image_id}").content == thumb.content  # cache hit is stable

    raw = client.get(f"/raw/{image_id}")
    assert raw.status_code == 200
    assert len(raw.content) > 0

    assert client.get("/thumbs/99999").status_code == 404


# -- API write path (annotation review) --------------------------------------


def _unlabeled_image_id(client: TestClient) -> int:
    body = client.get("/images", params={"labeled": False}).json()
    image_id = body["items"][0]["image"]["id"]
    assert isinstance(image_id, int)
    return image_id


def _bbox_body(class_id: int, **extra: object) -> dict[str, object]:
    return {"class_id": class_id, "geom_type": "bbox", "coords": [4, 6, 8, 8], **extra}


def test_api_create_annotation_labels_image(client: TestClient) -> None:
    image_id = _unlabeled_image_id(client)
    response = client.post(f"/images/{image_id}/annotations", json=_bbox_body(2))
    assert response.status_code == 201
    created = response.json()
    assert created["id"] is not None
    assert (created["class_id"], created["source"], created["status"]) == (2, "human", "approved")
    assert created["coords"] == [4, 6, 8, 8]

    detail = client.get(f"/images/{image_id}").json()
    assert [a["class_id"] for a in detail["annotations"]] == [2]
    assert client.get("/stats").json()["labeled_images"] == 2  # B is now labeled too


def test_api_create_unknown_class_is_422(client: TestClient) -> None:
    image_id = _unlabeled_image_id(client)
    assert client.post(f"/images/{image_id}/annotations", json=_bbox_body(99)).status_code == 422


def test_api_create_missing_image_is_404(client: TestClient) -> None:
    assert client.post("/images/9999/annotations", json=_bbox_body(0)).status_code == 404


def test_api_create_bad_coords_is_422(client: TestClient) -> None:
    image_id = _unlabeled_image_id(client)
    body = {"class_id": 0, "geom_type": "bbox", "coords": [1, 2, 3]}  # bbox needs 4
    assert client.post(f"/images/{image_id}/annotations", json=body).status_code == 422


def test_api_approve_keeps_model_provenance(client: TestClient) -> None:
    """Approving an unedited model box records model/approved — the 'model was
    right' case must stay distinct from a hand-corrected one."""
    image_id = _unlabeled_image_id(client)
    created = client.post(
        f"/images/{image_id}/annotations",
        json=_bbox_body(2, source="model", status="pending"),
    ).json()

    patched = client.patch(f"/annotations/{created['id']}", json={"status": "approved"})
    assert patched.status_code == 200
    assert (patched.json()["status"], patched.json()["source"]) == ("approved", "model")


def test_api_reclassify_flips_source_to_human(client: TestClient) -> None:
    image_id = _unlabeled_image_id(client)
    created = client.post(
        f"/images/{image_id}/annotations",
        json=_bbox_body(2, source="model", status="pending"),
    ).json()

    patched = client.patch(f"/annotations/{created['id']}", json={"class_id": 3}).json()
    assert (patched["class_id"], patched["source"], patched["status"]) == (3, "human", "pending")


def test_api_patch_geometry_updates_box(client: TestClient) -> None:
    """A full geom_type+coords PATCH re-boxes the annotation and records the edit
    as a human correction (source flips), but leaves it pending for approval."""
    image_id = _unlabeled_image_id(client)
    created = client.post(
        f"/images/{image_id}/annotations",
        json=_bbox_body(2, source="model", status="pending"),
    ).json()

    patched = client.patch(
        f"/annotations/{created['id']}",
        json={"geom_type": "bbox", "coords": [10, 12, 5, 6]},
    )
    assert patched.status_code == 200
    body = patched.json()
    assert body["coords"] == [10, 12, 5, 6]
    assert (body["source"], body["status"]) == ("human", "pending")


def test_api_patch_partial_geometry_is_422(client: TestClient) -> None:
    image_id = _unlabeled_image_id(client)
    created = client.post(f"/images/{image_id}/annotations", json=_bbox_body(2)).json()
    # coords without geom_type is an ambiguous half-edit.
    assert (
        client.patch(f"/annotations/{created['id']}", json={"coords": [1, 2, 3, 4]}).status_code
        == 422
    )


def test_api_patch_missing_annotation_is_404(client: TestClient) -> None:
    assert client.patch("/annotations/999999", json={"status": "approved"}).status_code == 404


def test_api_delete_annotation(client: TestClient) -> None:
    image_id = _unlabeled_image_id(client)
    created = client.post(f"/images/{image_id}/annotations", json=_bbox_body(2)).json()

    assert client.delete(f"/annotations/{created['id']}").status_code == 204
    assert client.get(f"/images/{image_id}").json()["annotations"] == []
    assert client.delete(f"/annotations/{created['id']}").status_code == 404  # already gone


# -- prelabel_pending --------------------------------------------------------


def _det(class_name: str, bbox: tuple[float, float, float, float], conf: float) -> DetectedEntity:
    return DetectedEntity(
        id="d", class_name=class_name, bbox=bbox, center=(0.0, 0.0), confidence=conf
    )


class _FakeDetector:
    """Returns a fixed prediction list regardless of the image (the only real
    screenshot in the prelabel queue is the unlabeled one)."""

    def __init__(self, detections: list[DetectedEntity]) -> None:
        self._detections = detections

    def detect(self, screenshot: Image.Image) -> list[DetectedEntity]:
        _ = screenshot
        return list(self._detections)


def _queue_ids(config: TrackerConfig) -> tuple[int, int]:
    """(unlabeled B id, labeled A id) captured before any prelabeling runs."""
    repo = _repo(config)
    (b_id,) = [i.record.id for i in repo.list_images(ImageFilter(labeled=False)).items]
    (a_id,) = [i.record.id for i in repo.list_images(ImageFilter(labeled=True)).items]
    assert b_id is not None and a_id is not None
    return b_id, a_id


def test_detections_to_annotations_converts_bbox(seeded_config: TrackerConfig) -> None:
    catalog = ClassCatalog(seeded_config.classes_yaml)
    converted = prelabel_pending.detections_to_annotations(
        7, [_det("class_2", (4.0, 6.0, 12.0, 14.0), 0.9)], catalog, 0.25
    )
    (ann,) = converted.annotations
    assert (ann.image_id, ann.class_id, ann.source, ann.status) == (7, 2, "model", "pending")
    assert ann.geometry == BBox(x=4.0, y=6.0, w=8.0, h=8.0)  # (x1,y1,x2,y2) -> (x,y,w,h)


def test_prelabel_writes_pending_boxes(seeded_config: TrackerConfig) -> None:
    b_id, _ = _queue_ids(seeded_config)
    fake = _FakeDetector([_det("class_2", (4, 6, 12, 14), 0.9), _det("class_3", (1, 1, 5, 5), 0.8)])

    report = prelabel_pending.run(seeded_config, fake, min_conf=0.25)

    assert (report.images_processed, report.boxes_written) == (1, 2)  # only the unlabeled image
    labeled = _repo(seeded_config).get_image(b_id)
    assert labeled is not None
    assert {a.class_id for a in labeled.annotations} == {2, 3}
    assert all(a.source == "model" and a.status == "pending" for a in labeled.annotations)


def test_prelabel_skips_low_conf_and_unknown_class(seeded_config: TrackerConfig) -> None:
    fake = _FakeDetector(
        [
            _det("class_2", (1, 1, 3, 3), 0.10),  # below threshold
            _det("no_such_class", (1, 1, 3, 3), 0.99),  # not in the 6-class schema
            _det("class_4", (2, 2, 6, 6), 0.99),  # kept
        ]
    )
    report = prelabel_pending.run(seeded_config, fake, min_conf=0.25)
    assert (report.boxes_written, report.skipped_low_conf, report.skipped_unknown_class) == (
        1,
        1,
        1,
    )


def test_prelabel_is_idempotent_and_preserves_human_labels(seeded_config: TrackerConfig) -> None:
    b_id, a_id = _queue_ids(seeded_config)
    fake = _FakeDetector([_det("class_2", (4, 6, 12, 14), 0.9), _det("class_3", (1, 1, 5, 5), 0.9)])

    prelabel_pending.run(seeded_config, fake, min_conf=0.25)
    prelabel_pending.run(seeded_config, fake, min_conf=0.25)  # re-run replaces, never appends

    repo = _repo(seeded_config)
    b_after = repo.get_image(b_id)
    a_after = repo.get_image(a_id)
    assert b_after is not None and a_after is not None
    assert len(b_after.annotations) == 2  # 2, not 4 — pending set was replaced
    assert {a.class_id for a in a_after.annotations} == {0, 5}  # human labels untouched
    assert all(a.source == "human" and a.status == "approved" for a in a_after.annotations)


def test_prelabel_hands_off_once_a_box_is_approved(seeded_config: TrackerConfig) -> None:
    """Approving any box takes the image out of the queue, so a re-run never
    wipes a review in progress or resurrects a rejected box."""
    b_id, _ = _queue_ids(seeded_config)
    fake = _FakeDetector([_det("class_2", (4, 6, 12, 14), 0.9), _det("class_3", (1, 1, 5, 5), 0.9)])
    prelabel_pending.run(seeded_config, fake, min_conf=0.25)

    repo = _repo(seeded_config)
    before = repo.get_image(b_id)
    assert before is not None
    repo.update_annotation(replace(before.annotations[0], status="approved"))

    report = prelabel_pending.run(seeded_config, fake, min_conf=0.25)

    assert report.images_processed == 0  # B left the queue; A was never in it
    after = _repo(seeded_config).get_image(b_id)
    assert after is not None
    statuses = sorted(a.status for a in after.annotations)
    assert statuses == ["approved", "pending"]  # both boxes intact
