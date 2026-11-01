# Tracker benchmark

**Faster: yes. Better: no.**

Run on arm64 at commit `a77c135`, over 12 generated scenes from real annotated frames, with the real detector. Units move as pasted sprites, so recorded footage is still needed for the final word.

## Gates

| Result | Gate | Question | Evidence |
|---|---|---|---|
| pass | parity | Same decisions as Python? | 0 differing frames (first: none) |
| pass | overhead | Inside the perception budget? | camera 13.0 + tracker 2.48 = 15.5 ms; Python 107.9 ms; limit 20 ms |
| pass | speed | Faster on real frames? | low 62x (at least 60x); typical 56x (at least 54x); high 45x (at least 40x); peak 35x (at least 33x) |
| pass | spec_target | Meets the original goal? | 1.48 ms in the high band; limit 2 ms |
| **FAIL** | target_survival | Do IDs shown to the LLM still work later? | 86.9% against 88.8% for the baseline |
| pass | wrong_target | Could an ID point at the wrong object? | 0.00% against 0.00% for the baseline |
| pass | recovery | Does one bad frame heal? | recovers in 0.2 s |
| pass | labels | Does smoothing help? | accuracy 93.46% against 93.45%; flips 3.9 against 4.2 per 1,000 |
| pass | economy_view | Is the agent's picture steadier? | job count error 2.94 against 2.94 |
| pass | event_loop | Does the agent stay responsive? | p99 lag 1.09 ms against 1.00 ms for an idle loop; Python 0.22 ms |
| pass | stability | Does it hold up over a game? | p95 drift +3.2%; memory growth 0.0 MB |

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
| low | 60.42 | 0.29 | 60.60 | 0.91 | 0.08 | 0.98 |
| typical | 93.93 | 0.95 | 94.93 | 1.56 | 0.15 | 1.71 |
| high | 65.31 | 1.86 | 66.74 | 1.30 | 0.22 | 1.48 |
| peak | 84.77 | 2.35 | 86.95 | 2.11 | 0.33 | 2.48 |

## Benchmark controls

| Result | Control | Evidence |
|---|---|---|
| pass | oracle is perfect | IDF1 1.000, survival 1.000 |
| pass | raw scores lowest | survival 0.000 |
| pass | A/A is unbiased | 0.99x, 1.01x, 0.94x, 1.01x |
| pass | injected slowdown is flagged | 37.2 ms against 20 ms |

Camera analysis p95 was 13.0 ms.

## Reading this run

This is the clean rerun. An earlier run the same day had the VM using about 230% CPU, which pushed `overhead`, `spec_target` and `stability` into false failures. With the VM suspended, Python's unchanged prepare stage matched the 2026-10-31 run within 5%, and the A/A control read 0.94-1.01x.

- **The fix made association faster.** Against 2026-10-31, `screentrack` associate p95 fell by 29-35% and Python's by 48-65%, because far fewer stale tracks remain to compare.
- **Part of the larger speedup is noise.** `screentrack` prepare is unchanged code, yet it read 8-29% lower than on 2026-10-31. Treat these speedups and the older ones as the same order of magnitude, about 35-60x.
- **`target_survival` is close but real.** 86.9% against 88.8%. The recorded gap below is wider.
- **`event_loop` uses a new rule.** It now requires p99 lag within 25% of an idle loop, because Python reads faster than doing nothing, so the old comparison could never be passed. The rule was changed after this run, with agreement, and applied to its saved measurements: 1.09 ms against 1.00 ms is 1.09x. Lag in the VM is unmeasured.
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
