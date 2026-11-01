# Tracker benchmark

**Faster: no. Better: no.**

> **Don't read this run's timing as a verdict.** The VM was running at about 230% CPU, so the "Faster: no" headline and the `overhead`, `spec_target` and `stability` failures aren't reliable. See [Reading this run](#reading-this-run). Quality scores don't depend on load.

Run on arm64 at commit `7d2244e`, over 12 generated scenes from real annotated frames, with the real detector. Units move as pasted sprites, so recorded footage is still needed for the final word.

## Gates

| Result | Gate | Question | Evidence |
|---|---|---|---|
| pass | parity | Same decisions as Python? | 0 differing frames (first: none) |
| **FAIL** | overhead | Inside the perception budget? | camera 18.1 + tracker 4.62 = 22.7 ms; Python 135.6 ms; limit 20 ms |
| pass | speed | Faster on real frames? | low 32x (at least 26x); typical 37x (at least 32x); high 26x (at least 23x); peak 25x (at least 21x) |
| miss | spec_target | Meets the original goal? | 2.83 ms in the high band; limit 2 ms |
| **FAIL** | target_survival | Do IDs shown to the LLM still work later? | 86.9% against 88.8% for the baseline |
| pass | wrong_target | Could an ID point at the wrong object? | 0.00% against 0.00% for the baseline |
| pass | recovery | Does one bad frame heal? | recovers in 0.2 s |
| pass | labels | Does smoothing help? | accuracy 93.46% against 93.45%; flips 3.9 against 4.2 per 1,000 |
| pass | economy_view | Is the agent's picture steadier? | job count error 2.94 against 2.94 |
| **FAIL** | event_loop | Is the agent less blocked? | p99 lag 0.52 ms against 0.34 ms; idle loop 0.53 ms |
| **FAIL** | stability | Does it hold up over a game? | p95 drift +27.8%; memory growth 0.0 MB |

## Agent outcomes

| Contender | Survival 1 s | Survival 2 s | Survival 5 s | Wrong target | IDF1 | IDF1 mobile | IDF1 static | Switches per 1,000 | Label accuracy | Flips per 1,000 | Job error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.0% | 0.0% | 0.0% | 0.00% | 3.3% | 4.1% | 2.9% | 1000 | 93.45% | 4.2 | 2.94 |
| oracle | 100.0% | 100.0% | 100.0% | 0.00% | 100.0% | 100.0% | 100.0% | 0 | 100.00% | 0.0 | 2.86 |
| iou | 94.5% | 88.8% | 73.1% | 0.05% | 79.9% | 64.8% | 86.7% | 82 | 93.45% | 4.2 | 2.94 |
| python | 92.9% | 86.9% | 70.8% | 0.00% | 78.1% | 64.1% | 84.4% | 71 | 93.46% | 3.9 | 2.94 |
| screentrack | 92.9% | 86.9% | 70.8% | 0.00% | 78.1% | 64.1% | 84.4% | 71 | 93.46% | 3.9 | 2.94 |

## Milliseconds per frame at steady state

| Band | Python prepare p95 | Python associate p95 | Python total p95 | screentrack prepare p95 | screentrack associate p95 | screentrack total p95 |
|---|---|---|---|---|---|---|
| low | 75.05 | 0.71 | 75.37 | 2.00 | 0.18 | 2.39 |
| typical | 109.30 | 1.68 | 110.66 | 2.65 | 0.27 | 3.05 |
| high | 73.19 | 2.76 | 74.32 | 2.33 | 0.46 | 2.83 |
| peak | 109.94 | 3.98 | 117.52 | 3.79 | 0.72 | 4.62 |

## Benchmark controls

| Result | Control | Evidence |
|---|---|---|
| pass | oracle is perfect | IDF1 1.000, survival 1.000 |
| pass | raw scores lowest | survival 0.000 |
| pass | A/A is unbiased | 0.90x, 0.93x, 0.91x, 0.87x |
| pass | injected slowdown is flagged | 38.7 ms against 20 ms |

Camera analysis p95 was 18.1 ms.

## Reading this run

- **Timing is unreliable.** The Windows VM was left running, using about 230% CPU, with a load average of 7-12. Code that didn't change got 1.40-1.56x slower than the 2026-10-31 run in `screentrack` prepare, and camera analysis got 1.32x slower. This needs a rerun on a quiet machine.
- **The fix shows no sign of costing time.** `screentrack` associate slowed by 1.29-1.50x, no more than the unchanged stage. Python associate fell by up to 41% even under load, because there are far fewer tracks to compare.
- **`target_survival` is close but real.** 86.9% against 88.8%. The recorded gap below is wider.
- **`event_loop` still needs its rule replaced.** `screentrack` measured 0.52 ms against 0.53 ms for an idle loop, so it adds nothing measurable.
- **`recovery` now passes:** one duplicate box heals in 0.2 s.

## Recorded footage

One human-played match, recorded on the Mac at 15 fps. Each clip is 60 s, replayed to the tracker at 5 Hz, with about 45 detections per frame, of which about 21 are moving units. Raw scores are in `docs/tracker-benchmark-results/2026-11-01-recorded-match-01.json` (before the fix) and `2026-11-01-recorded-match-01-fixed.json`.

| Contender | Still camera, 2 s survival | Moving camera, 2 s survival | Wrong target, still / moving | Town Center IDs, still clip |
|---|---|---|---|---|
| `python` and `screentrack`, before the fix | 0.0% | 1.5% | 0.0% / 0.0% | 296 in 300 frames |
| `python` and `screentrack`, fixed | 80.1% | 66.0% | 0.2% / 0.1% | 1 |
| `iou` | 94.2% | 92.0% | 4.4% / 1.8% | 1 |

Both trackers differ on 0 frames on both clips, before and after the fix.

- **Cause:** `_ambiguous_candidates` rejected an identity when any two of its candidates were close in cost, not just its best two. In a crowded frame most identities have such a pair, so 97% of detections were rejected from the first second. A stale duplicate could also keep one object ambiguous forever.
- **Both changes were needed.** Alone, the top-two rule reached 66.0% and 47.8%, and dropping stale duplicates reached 23.8% and 22.6%.

How far to trust each number:

- **Firm:** the Town Center count, which needs no truth, and static survival on the still clip, where nothing moves and the camera holds.
- **Rough:** moving-unit scores and the moving clip. Truth is linked automatically and hasn't had its hand audit. Camera cuts split the moving clip's truth into 734 tracks, which can also count kept IDs as wrong, so `iou`'s wrong-target figures may be too high. Truth linking resembles `iou`, which may flatter its moving-unit score.
- **Colour:** the recording passes through a colour table fitted on paused frame pairs. The rule set before that check failed even for the VM's own frame moved by 1 px, so it was replaced by "no worse than harmless changes". The corrected recording meets that on five of six measures and misses the sixth by one object, with 7 confidences moving over 0.1 against a worst case of 6.
- **Scope:** one match, played by a person, at the 5 Hz target. The agent reached about 0.08 fresh detections per second on the VM on 2026-11-01, and tracking resets when frames are over 1 s apart, so these results only apply once perception is faster.

### Recording another match

1. On the Mac, set Displays > Preset to Internet & Web (sRGB). The colour table only applies to this mode.
2. Run the game in the VM, and leave the VMware window where it is. Record from Terminal.app, which needs Screen Recording permission:

   ```bash
   ffmpeg -y -f avfoundation -capture_cursor 0 -framerate 15 -pixel_format nv12 -i "Capture screen 0:none" -t 600 -c:v prores_videotoolbox -profile:v standard ~/aoe2-capture/match-02.mov
   ```

3. Press F3 to pause, then capture the same frame both ways: `python -m gameplay_agent.capture_sequence --output logs\pixel-check-N --count 3 --interval 1` in the VM, and the command above with `-t 5` on the Mac. Use `frame_00002.png`, because the first frame can be half size.
4. Fit a table with `recorded.py colour`, using that pair and any earlier ones, cut clips with `recorded.py prepare`, and rate them with `recorded.py score`.
