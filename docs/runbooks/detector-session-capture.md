# Phase 2: capture and label independent gameplay sessions

The [detector-quality plan](detector-quality-retraining.md) needs new, native
gameplay images before retraining. This runbook uses **one session per match**,
not one session per screenshot or camera view. A session's development or
final-test assignment is chosen at capture time and retained across later
age/zoom batches. The old January validation set and previous agent logs are
development material only; do not reuse them as the new final test.

The tools are ready locally. No new Windows gameplay has been captured or
human-reviewed yet, so phase 2's data and annotation exit gate is **pending**.

## 1. Plan the matches before recording

Record several independent matches, not adjacent frames from one opening.
Development sessions should cover opening TCs, sheep and berries, later farms,
housing/construction, forests, cavalry and mixed armies, team colors, fog,
empty views, misleading backgrounds, different terrain, and the normal and edge
zoom positions. Scenario-editor captures are useful for development gaps, but
not as final-test evidence.

Reserve separate **normal gameplay** matches for the final test before looking
at their predictions. Keep every capture or replay view from one match in the
same split. Never use final-test frames as training backgrounds, synthetic
templates, sprite/scale calibration, threshold tuning, or model-selection
examples. Note the game build, graphics preset, UI scale, map, civilization,
capture resolution, game stage, and zoom setting. Do not silently invent an
unknown zoom; describe the actual slider position or wheel-step offset.

The final set must contain enough *labeled* sheep, berry bushes, farms,
villagers, TCs, houses, mills, and cavalry for a meaningful class estimate.
There is no magic frame count: inspect support and keep capturing new sessions
when a critical class is represented by only a handful of objects.

## 2. Capture native PNGs on the Windows VM

Start AoE2, select the intended match/replay view, and run the command from the
Windows repository checkout. Replace `<game build>` with the version shown by
the game. The command fails if it cannot find the exact AoE2 window; it does
not silently capture the entire desktop. It runs as a separate process, not
inside the gameplay agent.

First use `--count 1` and inspect that PNG: it should contain the intended
game-window pixels at the expected resolution, with no host desktop or cut-off
game area. Continue the **same session ID** with `--append` for the remaining
frames; do not rename batches from one match into separate sessions.

```powershell
uv run --no-sync python -m gameplay_agent.capture_dataset --session-id dev_20261005_match01 --split development --source replay --map Highland --civilization Magyars --game-version "<game build>" --graphics-preset high --ui-scale 100% --game-stage dark_age --zoom default --count 20 --interval 15
```

To add later Castle-age or zoomed views of **that same match**, repeat with all
the same match-level settings, change the stage/zoom, and add `--append`. The
tool checks that the split, source, map, civilization, build, graphics preset,
UI scale, and window resolution have not changed. Finish all capture batches
before prelabeling or annotating. Use a new match and ID if resolution or game
settings change. Use `--split final_test --source live_game` (or a normal-match
replay) for the reserved test sessions; staged scenarios are rejected there.

The default output root is
`packages/detection/src/real_screenshots/quality_sessions/`. Use `--root` to
write to a VM-shared folder if desired. The root is ignored by Git. Copy or
sync **whole session directories** to the same root on the Mac without
re-encoding PNGs or editing `capture.json`; hashes detect altered bytes.

Each session contains lossless `images/frame_*.png` and `capture.json` with
capture timestamps, per-frame stage/zoom, metadata, and image SHA-256. Short
intervals can produce near duplicates; pan or vary gameplay deliberately,
then inspect the audit's duplicate warnings.

## 3. Prelabel development only, then correct in CVAT

On the Mac, with synced sessions and the local reference ONNX model present,
run this from the repository root. The `--root` option precedes the subcommand.

```bash
uv run --no-sync python -m detection_server.session_tools \
  --root packages/detection/src/real_screenshots/quality_sessions \
  prelabel --session-id dev_20261005_match01 \
  --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx
```

Create one CVAT task per session with the session's PNGs. For development,
import that session's `prelabels.coco.json` as COCO 1.0 suggestions. The file
uses the **current served ONNX** decoder and 60-class mapping, not the older
v6 PyTorch default in the legacy prelabel script. Its boxes are suggestions,
not ground truth. Add missed objects and correct wrong classes and boxes.

For final-test tasks, **do not prelabel**. Upload the PNGs and annotate from
scratch so reference-model mistakes do not influence the held-out labels.
The prelabel command enforces this split restriction.

Use one annotation convention throughout:

- Annotate every visible, identifiable instance of the 60 classes, including
  negatives/background views with **zero** objects. Keep an empty annotation
  file for a verified empty image; missing labels are not empty scenes.
- Draw a tight box around the object's **visible artwork** inside the image.
  For partial occlusion, box the visible portion; do not infer hidden extent.
  Exclude shadows, health bars, selection circles, and cursor/UI decorations.
  Skip a fully hidden or unidentifiable fragment rather than guessing a class.
- Use the same class for animation and upgrade variants as defined in
  `classes.yaml`. Review dense woods and crowded units for missed instances.
  If an existing training batch used a different box convention, audit it
  before mixing it into the next training set.

Export the corrected CVAT task as **COCO 1.0**, including all images. Import
the corrected export, inspect a QA gallery, and seal it *only after a human has
checked all visible objects*. The importer rejects missing images, unknown
classes, and out-of-bounds boxes. It creates empty YOLO label files for
genuinely empty frames. If QA finds mistakes, correct the CVAT task and
re-import with `--replace-unsealed`; the prior labels are moved to
`annotation_history/` rather than deleted. Sealed labels cannot be replaced.

```bash
uv run --no-sync python -m detection_server.session_tools \
  --root packages/detection/src/real_screenshots/quality_sessions \
  import-coco --session-id dev_20261005_match01 --coco /path/to/corrected.json
```

Repeat the import and review process for final-test sessions **without** the
prelabel step. The seal records label hashes and the reviewer; evaluation
rejects unsealed, incomplete, or later-edited annotations.

## 4. Audit coverage and inspect labels

```bash
uv run --no-sync python -m detection_server.session_tools \
  --root packages/detection/src/real_screenshots/quality_sessions \
  qa-gallery --output-dir tmp/evaluations/qa-gallery
```

Inspect every gallery sample alongside the original image for missed sheep,
small units, overlapping trees, wrong classes, and box extent. Re-export and
re-import corrections before sealing. Use a new gallery output directory after
each revision. Then seal every reviewed session and run the final audit:

```bash
uv run --no-sync python -m detection_server.session_tools \
  --root packages/detection/src/real_screenshots/quality_sessions \
  seal --session-id dev_20261005_match01 --reviewer "your-name"

uv run --no-sync python -m detection_server.session_tools \
  --root packages/detection/src/real_screenshots/quality_sessions \
  audit --output tmp/evaluations/session-qa.json
```

The audit reports per-split sessions, frames, class support, source/map/build,
zoom/stage and resolution coverage, unreviewed sessions, identical captures, near-duplicate
cross-split pairs, and a first/middle/last sample from each session. Exact
image duplication across development and final test fails validation;
perceptual near matches are **warnings for human review**, not automatic
deletions.
Record reviewer, date, findings, and any corrected/re-exported session next to
the audit report. Machine validation cannot prove labels are complete.

The phase-2 exit gate is: new session-disjoint captures; meaningful critical-
class support across zoom and game stage; all intended evaluation labels
human-reviewed and hash-sealed; no unexplained cross-split duplicates; and a
documented visual QA pass. Until then, no new generalization number is valid.

## 5. Development scoring now; final scoring later

The new wrapper accepts reviewed session manifests and uses the exact frozen
phase-1 inference and scorer. Development scoring is safe for iteration:

```bash
uv run --no-sync python -m detection_server.evaluate_sessions \
  --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx \
  --root packages/detection/src/real_screenshots/quality_sessions \
  --split development --output tmp/evaluations/session-development.json
```

Do **not** score the final-test sessions now. At the later release gate, after
freezing the candidate and confirming the test sessions were excluded from all
training and tuning, the evaluator requires the explicit
`--attest-isolated-final-test` flag. Score the reference and candidate on the
same test frames only then. The code checks manifest and image hashes and
session-disjointness inside this dataset; the operator must verify that no
separate training source reused those matches.
