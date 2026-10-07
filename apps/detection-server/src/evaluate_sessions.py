"""Score manually reviewed, session-disjoint gameplay captures.

The original 32-frame development evaluator stays unchanged. This wrapper
selects an explicit split and then calls its exact inference/scoring functions.
Final-test scoring requires an explicit provenance attestation at release time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detection.labeling.class_mapping import load_classes_yaml
from detection.labeling.session_dataset import (
    CAPTURE_MANIFEST,
    REVIEWED_MANIFEST,
    Split,
    list_sessions,
    load_reviewed_labels,
    sha256_file,
)

from .evaluate import Case, build_report, collect_samples, load_cpu_model


def evaluate_sessions(
    model: Path,
    root: Path,
    split: Split,
    *,
    attest_isolated_final_test: bool = False,
) -> dict[str, object]:
    """Require reviewed labels and identical capture/label bytes before scoring."""
    if split == "final_test" and not attest_isolated_final_test:
        raise ValueError("Final test is sealed; attest training isolation only at the release gate")
    selected = [
        (directory, session) for directory, session in list_sessions(root) if session.split == split
    ]
    if not selected:
        raise ValueError(f"No {split} sessions in {root}")

    cases: list[Case] = []
    capture_hashes: dict[str, str] = {}
    label_hashes: dict[str, str] = {}
    for directory, session in selected:
        reviewed = load_reviewed_labels(directory, session)
        labels_by_image = {item.image: item for item in reviewed.labels}
        for frame in session.frames:
            label = labels_by_image[frame.image]
            cases.append(Case(directory / frame.image, directory / label.label))
        capture_hashes[session.session_id] = sha256_file(directory / CAPTURE_MANIFEST)
        label_hashes[session.session_id] = sha256_file(directory / REVIEWED_MANIFEST)

    state = load_cpu_model(model)
    classes = load_classes_yaml()
    if tuple(classes[index] for index in sorted(classes)) != state.class_names:
        raise ValueError("Model/server classes do not match the dataset schema")
    samples = collect_samples(state, cases)
    report = build_report(model, cases, state, samples)
    report.update(
        {
            "evaluation_kind": "session_held_out_real_frame_test"
            if split == "final_test"
            else "session_development",
            "independent_test_set": split == "final_test",
            "independence_basis": "declared session split, cross-split byte checks, "
            "and operator attestation that final sessions were excluded from training, "
            "backgrounds, thresholds, and model selection",
            "split": split,
            "session_ids": [session.session_id for _, session in selected],
            "session_capture_sha256": capture_hashes,
            "session_reviewed_labels_sha256": label_hashes,
            "session_evaluator_sha256": sha256_file(Path(__file__)),
        }
    )
    return report


class _Args(argparse.Namespace):
    model: Path
    root: Path
    split: Split
    attest_isolated_final_test: bool
    output: Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("development", "final_test"), required=True)
    parser.add_argument("--attest-isolated-final-test", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(namespace=_Args())
    report = evaluate_sessions(
        args.model,
        args.root,
        args.split,
        attest_isolated_final_test=args.attest_isolated_final_test,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    micro = report["micro"]
    if isinstance(micro, dict):
        print(f"Scored {report['images']} frames; micro F1={micro['f1']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
