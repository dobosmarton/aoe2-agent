"""Ingest + repository + API tests for the training tracker.

A tiny on-disk fixture (2 raw screenshots, one labeled with a duplicated copy,
one synthetic image, a 6-class schema) exercises the whole read path end-to-end.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from training_api import ingest
from training_api.classes import ClassCatalog
from training_api.config import TrackerConfig, load_config
from training_api.db import connect, init_schema
from training_api.domain import ImageFilter
from training_api.repository import SqliteTrackerRepository

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

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
