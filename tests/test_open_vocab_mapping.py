"""Unit tests for open-vocab prompt building and label mapping.

These run against the real bundled `classes.yaml` (no model, no network).
"""

from __future__ import annotations

from detection.labeling.open_vocab_mapping import (
    build_class_prompts,
    map_open_vocab_label,
)


def test_prompts_cover_all_60_classes() -> None:
    prompts = build_class_prompts()
    assert len(prompts) == 60
    assert all(len(phrases) >= 1 for phrases in prompts.values())


def test_unique_class_prompts_include_civ_examples() -> None:
    # unique_cavalry (id 51) lists Cataphract/Boyar/... in classes.yaml examples.
    prompts = build_class_prompts()
    assert "Cataphract" in prompts[51]
    assert "cavalry" not in prompts[51]  # humanised "unique cavalry", not bare "cavalry"


def test_humanised_name_drops_line_suffix() -> None:
    # scout_line (id 33) humanises to "scout".
    assert "scout" in build_class_prompts()[33]


def test_map_label_is_case_insensitive_and_maps_examples() -> None:
    assert map_open_vocab_label("Cataphract") == 51
    assert map_open_vocab_label("cataphract") == 51
    assert map_open_vocab_label("scout") == 33


def test_unknown_label_maps_to_none() -> None:
    assert map_open_vocab_label("trebuchet_on_fire") is None
