# Streaming perception

The live agent now has three top-level loops: perception, actor, and deliberate
strategy/tactics. Perception captures native game-window pixels at a 100 ms
start-to-start target. One latest-frame slot each feeds single-frame detection
(200 ms target) and HUD reading (500 ms target). An input-dependent refresh is
prioritized, publishes its post-input observation to the same frame channel,
and has the existing three-second timeout. A slow HUD read does not
hold a detector result back. Frames are never queued without a bound.

The native capture's immutable RGB bytes back read-only NumPy views. Live HUD,
selection, and ownership readers use those pixels; encoded-image adapters for
files and test fixtures decode once and call the same readers. A HUD pass
prepares its small RGB/grayscale band once while retaining full capture
geometry for calibration. Full detection refreshes encode one JPEG for the
detector and reuse it for the published screenshot, requested-refresh evidence,
and sampled diagnostics. Selection-only refreshes skip detection. Every tenth
successfully published spatial frame is sampled, rather than every tenth
capture; encoding and diagnostic writing do not run on the capture lane.

Every spatial publication contains **fresh detector boxes**. The per-game
temporal component associates at most three observations per entity for at most
one second. It may stabilize an ID or class label, but it never publishes a
predicted-only box or treats historical agreement as a calibrated probability.
Appearance descriptors are prepared once per detection off the event loop.
Association uses horizontal movement bounds to prune candidate pairs, then
the same distance/appearance cost, ambiguity threshold, deterministic order,
and greedy match as before. Tracker state changes only when a compatible worker
result is published; late results cannot update history.
Click authorization still requires a current, compatible raw detector label,
capture ID, camera revision, and fresh bounding box. HUD and spatial versions,
capture IDs, and timestamps stay separate when their observations are merged.

Camera motion is translation-only. The HUD/minimap are excluded, and known
mobile-object boxes plus the sampled cursor position are masked. Six
downsampled terrain regions are checked;
at least three must agree within two analysis pixels, with at least 60% aligned
overlap. A jump, zoom, geometry change, gap over one second, or failed alignment
resets tracks. The first fresh post-reset detection publishes immediately.

The detection server admits one inference at a time across its regular and
tiled endpoints. Excess requests return a typed `detector_busy` response; a
disconnected client cannot free capacity before its worker finishes. Busy
responses skip that frame, without triggering local fallback. Recorded runs
still fail if their configured detector is unavailable; interactive runs keep
their configured local fallback. The detector adapters are stateless and do not
allocate persistent IDs.

## Capture and replay

On the Windows VM, a short sequence can be recorded without involving the
agent's input loop:

```powershell
uv run python -m gameplay_agent.capture_sequence --output logs/sequence-01 --count 50 --interval 0.2
```

It saves native PNGs and `sequence.jsonl` containing monotonic capture times,
window coordinates, cursor samples, and provisional image-derived camera
events. The reviewed replay recomputes camera motion with detector boxes masked.
An optional
`--input-events events.jsonl` records known input events alongside the captures;
each line must contain `at_monotonic` and `kind` (for example `camera_jump` or
`zoom`) on the same VM clock. Without that file, input events are **unknown**,
not inferred from camera motion.

After reviewing the sequence and writing `labels.jsonl`, cache each image's
single-frame detections once, then compare raw and temporal output without
running inference again:

```powershell
uv run python -m gameplay_agent.sequence_replay --sequence logs/sequence-01 --detections logs/sequence-01/raw.jsonl --labels logs/sequence-01/labels.jsonl --infer
uv run python -m gameplay_agent.sequence_replay --sequence logs/sequence-01 --detections logs/sequence-01/raw.jsonl --labels logs/sequence-01/labels.jsonl
```

Use `--infer` only for the first command: it requires the configured local
model weights and refuses to overwrite an existing cache. Each labels line is
`{"capture_id": 0, "objects": [{"track_id": "sheep-1", "class_name": "sheep", "bbox": [x0, y0, x1, y1]}]}`.
The report scores IoU ≥ 0.5 per class, class flips, temporal ID switches,
camera resets/pans, and association ambiguity. Add `--attempts attempts.jsonl`
to score reviewed clicks; each line needs `capture_id`, `target_class`, `x`,
and `y` in window-relative pixels. Without an action trace, wrong-target
attempts remain `null` rather than a misleading zero.

## Diagnostics and qualification

Runtime logs separate native capture, JPEG encoding, detector duration, camera
analysis, appearance preparation, candidate generation, one-to-one matching,
ownership classification, and publication age. Sequence replay reports p50/p95
for camera analysis, descriptor preparation, candidate generation, matching,
association, and preparation-plus-association. The latter is the comparable
temporal total when evaluating a change that moves preparation to a worker.
The remote client also reports HTTP round trip, server processing, and their
difference (transport plus serialization overhead). Requested-refresh latency
is logged separately. Sampled frames use a bounded background writer;
operation evidence has priority, and dropped diagnostics are reported.

Local warmed-up benchmarks on the same Mac used five repetitions per synthetic
case at 1920×1080. Values below are median milliseconds for **descriptor
preparation plus association** across three consecutive frames; boxes were
64×64 pixels, with overlapping centers in the crowded cases. These stress
association, not detector inference or gameplay accuracy.

| Layout | Objects | Previous frame 2 / 3 | Current frame 2 / 3 |
| --- | ---: | ---: | ---: |
| Separated | 20 | 3.08 / 3.16 | 3.09 / 3.30 |
| Separated | 40 | 7.23 / 6.96 | 6.13 / 6.14 |
| Separated | 60 | 10.84 / 11.28 | 9.11 / 9.16 |
| Crowded | 20 | 13.83 / 43.20 | 3.68 / 4.23 |
| Crowded | 40 | 187.70 / 691.05 | 8.67 / 11.75 |
| Crowded | 60 | 944.53 / 3403.13 | 15.36 / 21.56 |

On the 3024×1964 HUD fixture,
a separate 20-run conversion benchmark gave p50/p95 of 58.42/59.55 ms for an
emulated prior full-image-to-PNG-to-decode preparation path versus 0.07/0.15 ms
for the native HUD band and grayscale conversion. Two JPEG encodes took
37.33/37.70 ms versus 18.86/20.29 ms for one. OCR recognition, capture,
inference, transport, and publication were excluded from those conversion
measurements. These synthetic and isolated results do not establish end-to-end
latency on the Windows VM.

These are **targets**, not observed results: at least five fresh detections/s,
capture-to-publication p95 at most 250 ms, and camera/temporal overhead p95 at
most 20 ms. Track updates are not fresh detector invocations. No Windows/Mac
stationary/pan/jump/zoom/crowded replay or reviewed sequence score has been
measured for this change yet. In particular, local tests do not establish
improved detector accuracy or real-time performance. If the targets are missed,
profile the logged stages before changing cadence or safety limits.
The synthetic crowded 60-object third-frame median above already exceeds the
20 ms camera/temporal objective even before camera estimation, so that workload
needs live profiling; no safety check or inference resolution was relaxed.
