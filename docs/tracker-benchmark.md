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

One human-played match, recorded on the Mac at 15 fps. Each clip is 60 s, replayed to the tracker at 5 Hz, with about 45 detections per frame. Scores before the first fix are in `docs/tracker-benchmark-results/2026-11-01-recorded-match-01.json`, and those after it in `2026-11-01-recorded-match-01-fixed.json`.

**This match is for tuning, so these aren't a verdict.** The held-out test uses matches the tracker has never seen, with the rules in the `[recorded]` table of `gates.toml`.

Share of IDs still correct after 2 s, on the current truth:

| Contender | Still camera | Moving camera | Moving units, moving camera | After 5 s, still / moving |
|---|---|---|---|---|
| After the first fix (`screentrack` today) | 80.5% | 65.4% | 53.7% | 78.8% / 53.2% |
| New rules, Python only | 94.2% | 94.0% | 97.0% | 93.7% / 86.6% |
| `iou` | 94.7% | 92.5% | 88.1% | 92.3% / 81.4% |

Before the first fix, on older truth, the clips scored 0.0% and 1.5%.

Remaining wrong objects on these clips are 12 events, none on a reviewed unit: they involve farms, camps and farm-sized boxes.

**Why IDs broke:** almost every break was an ambiguity rejection, and in about 85% of those the rival sat on the same spot under another label. The detector labels one spot twice on nearly every frame, for example `farm`+`mill` on 300 of 300 frames, and the two tracks looked identical to the matcher.

**The rules that fixed it:**

- **Overlap in the cost:** `1 - IoU` replaces `distance / allowed`, because a tracked object barely moves between frames.
- **Label penalty:** a label change costs 1.0. That stops two labels tying, while a flickering label can still continue its track.
- **Same-label stale pruning:** a second label that skips a frame gets its ID back.
- **Adaptive memory:** tracks live for `max(1 s, 3 × frame gap)`, and only a camera reset clears them. On the still clip, with the camera held, 92.8% survive 10 s gaps, where the older rule scored 0%. Too few moving-unit targets reached long gaps to judge them.

**Tested and dropped:** velocity prediction, a ratio test and voted labels didn't help.

### How the truth is built

- **Static objects** link automatically, and looked right throughout the audit.
- **Moving units** can't be trusted automatically: 15 of 48 random tracks failed review by eye, from swaps, farm ghosts, static objects or silhouettes that couldn't be checked. So they count only if reviewed, and the 33 that passed are in `docs/tracker-benchmark-results/verified/`. The review shows no tracker output, so it can't favour one.
- **Linking stops rather than guesses.** A broken track only skips an object when scoring, while a wrong join blames a tracker for being right.
- **Wrong target** now means a different object. A second label on the same spot is counted separately, because a click lands in the same place.

### Detector finding

The detector labels whole farms `villager`, at about 0.8 confidence. Boxes over 80 px make up about a third of `villager` detections, around 6 per frame. Truth treats them as `not_a_unit`. These phantom villagers probably skew the agent's worker counts too, which needs its own look.

### How far to trust this

- **Colour:** the recording passes through a colour table fitted on paused frame pairs. The rule set before that check failed even for the VM's own frame moved by 1 px, so it was replaced by "no worse than harmless changes". The corrected recording meets that on five of six measures and misses the sixth by one object.
- **Scope:** one match, played by a person, at the 5 Hz target. The agent reached about 0.08 fresh detections per second on the VM on 2026-11-01, and `apps/agent/src/loops/source.py` still resets at gaps over 1 s, so the agent won't see this until perception is faster.

### Held-out result

The rules in `gates.toml` were fixed before matches 2 and 3 existed, and the reviewed tracks were committed before any tracker was scored on them. Python tracker at `17ac2d1` against the IoU matcher, pooled over four clips and 7,947 targets:

| Check | Python | IoU | Result |
|---|---|---|---|
| Still the same object after 2 s | 93.7% | 94.4% | Fail |
| Wrong object | 0.34% (27) | 0.05% (4) | Fail |
| Label accuracy | 86.84% | 86.88% | Fail |
| Label flips | 16.9 | 17.1 | Pass |

Labels count as one check, so the tracker failed all three. Generated scenes, recovery and speed were not run, because the result was already decided. Per the plan, the rules were not ported to Rust, and these matches are now spent: a retest needs a new match.

The result differs by object type. Reviewed units survived better than with IoU on three clips and tied on the fourth (100% against 95.2%, 97.0% and 73.3%, then 94.1% each), but those are only 20 units. Static objects make up most targets, and they did slightly worse on the VM clips (97.0% against 98.6%, and 94.3% against 96.3%). Full figures are in `docs/tracker-benchmark-results/2026-11-01-recorded-heldout.json`.

**Match 3 is weak evidence.** It is the Mac game, recorded at about 9.5 fps. The detector found only 14-19 objects per frame in a crowded match, misplaced many building boxes, and missed most massed units, so only 11 tracks were long enough to review and 6 passed. The moving clip also has 25 camera cuts, which is why it has no 5 s targets. Playing Magyars on Arabia did not fix the buildings, so an unfamiliar civ was not the cause.

### After the held-out test

These figures are development evidence, not a retest. Matches 2 and 3 are spent, and the rule below came from studying match 2's failures.

Most static losses on match 2 had one cause. When a static object's box was missed for a frame or two, the Python tracker moved the object's ID onto a nearby box. It allows matches without overlap so it can follow units through slow frames. A farm's ID went to a villager standing in it, and the villager walked off with it. Trees and farms passed their IDs to neighbours. IoU can't do this, because it needs 30% overlap. Two farms caused 36 of the 50 losses on the still clip.

The fix in `temporal_entities.py`:

- A label change needs 50% overlap, because a new label means the detector changed its mind about the same box.
- A pair with a static label needs 10% overlap. 30% was too strict: it broke matches the tracker keeps through camera pans.
- Units still match without overlap.

Scored as the held-out verdict, on the same four clips:

| Check | Before | After | IoU |
|---|---|---|---|
| Still the same object after 2 s | 93.7% | 94.7% | 94.4% |
| Wrong object | 0.34% (27) | 0.11% (9) | 0.05% (4) |
| Label accuracy | 86.84% | 86.89% | 86.88% |
| Label flips | 16.9 | 16.6 | 17.1 |

Elsewhere:

- **Match 3:** wrong objects fell from 17 to 5, although its failures were never studied.
- **Match 1:** survival went from 94.1% to 94.3%, against 93.9% for IoU.
- **Generated scenes:** unchanged at 88.89%.
- **Units:** survival is unchanged on every clip.

Still open:

- **Wrong objects:** 9 against 4, all of them static.
- **Match 2's moving clip:** static objects still trail IoU, 95.0% against 96.3%. The losses come mostly from close calls the tracker refuses to decide.
- **Match 1's "same spot" losses:** most of what remains there happens when the detector puts two labels on one box. Python keeps one ID per label, while the truth gives the box to one object arbitrarily. A click lands in the same place either way, so this may be a scoring question rather than a tracker fault. The rules stay unchanged until that is decided.

The Rust port still waits for a new held-out match.

### Recording another match

Record the Mac game from Terminal.app, which needs Screen Recording permission. The fragmented format keeps frame times if a recording dies: in a test, killing the recording lost only the last unwritten fragment, about two seconds, rather than the whole file. Ten minutes takes about 10 GB at the rate below, so start with at least 30 GB free, because swap also uses the disk.

1. On the Mac, set Displays > Preset to Internet & Web (sRGB).
2. Record with this command. Try it first with `-t 30` while the game is busy, and expect about 9.5 fps, with nearly every gap at 67 or 133 ms.

   ```bash
   ffmpeg -y -f avfoundation -capture_cursor 0 -framerate 15 -pixel_format nv12 -i "Capture screen 0:none" -t 600 -c:v prores_videotoolbox -profile:v standard -movflags +frag_keyframe+empty_moov -min_frag_duration 2000000 match.mov
   ```

3. Cut clips with `recorded.py prepare --top 66`. Leave out `--colour`, because no paired capture exists for the Mac game yet. Limits follow each clip's own frame rate, so VM clips rebuild unchanged.
4. Draw review strips with `recorded.py strips`, and save the tracks that pass as `verified.json` beside the clip.
5. Rebuild with `recorded.py prepare`, then judge held-out matches once with `recorded.py verdict`.
