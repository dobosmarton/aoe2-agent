# Streaming perception

The live agent now has three top-level loops: perception, actor, and deliberate
strategy/tactics. Perception captures native game-window pixels at a 100 ms
start-to-start target. One latest-frame slot each feeds single-frame detection
(200 ms target) and HUD reading (500 ms target). An input-dependent refresh is
prioritized, publishes its post-input observation to the same frame channel,
and has the existing three-second timeout. A slow HUD read does not
hold a detector result back. Frames are never queued without a bound.

Every spatial publication contains **fresh detector boxes**. The per-game
temporal component associates at most three observations per entity for at most
one second. It may stabilize an ID or class label, but it never publishes a
predicted-only box or treats historical agreement as a calibrated probability.
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
analysis, ownership classification, temporal association, and publication age.
The remote client also reports HTTP round trip, server processing, and their
difference (transport plus serialization overhead). Requested-refresh latency
is logged separately. Sampled frames use a bounded background writer;
operation evidence has priority, and dropped diagnostics are reported.

These are **targets**, not observed results: at least five fresh detections/s,
capture-to-publication p95 at most 250 ms, and camera/temporal overhead p95 at
most 20 ms. Track updates are not fresh detector invocations. No Windows/Mac
stationary/pan/jump/zoom/crowded replay or reviewed sequence score has been
measured for this change yet. In particular, local tests do not establish
improved detector accuracy or real-time performance. If the targets are missed,
profile the logged stages before changing cadence or safety limits.
