"""Unit tests for villager job inference + motion-robust selection.

Pure functions over fake entity dicts — no detector / tracker / pyautogui.
"""

from __future__ import annotations

from gameplay_agent.villager_roles import (
    IDLE,
    VillagerRoleModel,
    infer_jobs,
    job_counts,
    select_worker,
)


def _ent(cls: str, center: tuple[float, float], vid: str | None = None) -> dict:
    return {"class": cls, "id": vid or f"{cls}_0", "center": center, "confidence": 0.9}


# ---------------------------------------------------------------------------
# Job inference
# ---------------------------------------------------------------------------


def test_infer_jobs_by_proximity() -> None:
    entities = [
        _ent("tree", (100, 100)),
        _ent("villager", (110, 110), "villager_0"),  # on the tree → wood
        _ent("sheep", (900, 900)),
        _ent("villager", (905, 905), "villager_1"),  # on the sheep → food
        _ent("villager", (500, 500), "villager_2"),  # near nothing → idle
    ]
    jobs = infer_jobs(entities)
    assert jobs == {"villager_0": "wood", "villager_1": "food", "villager_2": IDLE}


def test_mining_camp_tags_gold() -> None:
    entities = [_ent("mining_camp", (0, 0)), _ent("villager", (20, 20), "v")]
    assert infer_jobs(entities)["v"] == "gold"


def test_job_counts_includes_zero_kinds() -> None:
    counts = job_counts({"a": "wood", "b": "wood", "c": "food"})
    assert counts["wood"] == 2
    assert counts["food"] == 1
    assert counts["gold"] == 0 and counts["stone"] == 0 and counts[IDLE] == 0


def test_role_model_smooths_single_frame_flip() -> None:
    model = VillagerRoleModel(window=5)
    wood_frame = [_ent("tree", (100, 100)), _ent("villager", (110, 110), "v")]
    food_frame = [_ent("sheep", (100, 100)), _ent("villager", (110, 110), "v")]
    for _ in range(3):
        model.update(wood_frame)
    model.update(food_frame)  # one noisy frame
    # Majority over the window is still wood — the flip doesn't win.
    assert model.update(wood_frame)["v"] == "wood"


def test_role_model_forgets_vanished_villager() -> None:
    model = VillagerRoleModel()
    model.update([_ent("tree", (0, 0)), _ent("villager", (10, 10), "v")])
    model.update([_ent("tree", (0, 0))])  # villager gone
    assert "v" not in model.counts() or model.counts().get("wood", 0) == 0
    assert model.update([]) == {} or "v" not in model.update([])


# ---------------------------------------------------------------------------
# Worker selection
# ---------------------------------------------------------------------------


def test_select_worker_none_when_job_absent() -> None:
    entities = [_ent("sheep", (0, 0)), _ent("villager", (10, 10), "v")]  # v is food
    assert select_worker(entities, "wood") is None


def test_select_worker_prefers_stationary() -> None:
    entities = [
        _ent("tree", (100, 100)),
        _ent("villager", (110, 110), "moving"),
        _ent("villager", (120, 120), "still"),
    ]
    velocities = {"moving": (30.0, 0.0), "still": (0.5, 0.0)}
    sel = select_worker(entities, "wood", velocities=velocities)
    assert sel is not None
    assert sel.click == (120, 120)  # the stationary worker


def test_select_worker_box_covers_cluster() -> None:
    entities = [
        _ent("tree", (100, 100)),
        _ent("villager", (100, 100), "a"),
        _ent("villager", (200, 160), "b"),
    ]
    sel = select_worker(entities, "wood", box_pad=40)
    assert sel is not None
    x0, y0, x1, y1 = sel.box
    assert x0 == 60 and y0 == 60  # min - pad
    assert x1 == 240 and y1 == 200  # max + pad
