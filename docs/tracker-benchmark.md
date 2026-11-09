# Tracker benchmark

**Faster: yes. Better: no.**

Run on arm64 at commit `1cca928`, over 12 generated scenes from real annotated frames, with the real detector. Units move as pasted sprites, so recorded footage is still needed for the final word.

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

The rules in `gates.toml` were fixed before matches 2 and 3 existed, and the reviewed tracks were committed before any tracker was scored on them. Python tracker at `b8752f3` against the IoU matcher, pooled over four clips and 7,947 targets:

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

These figures are development evidence, not a retest. Matches 2 and 3 are spent, and both changes below came from studying their failures.

**Overlap.** Most static losses on match 2 had one cause. When a static object's box was missed for a frame or two, the Python tracker moved the object's ID onto a nearby box. It allows matches without overlap so it can follow units through slow frames. A farm's ID went to a villager standing in it, and the villager walked off with it. Trees and farms passed their IDs to neighbours. IoU can't do this, because it needs 30% overlap. Two farms caused 36 of the 50 losses on the still clip. The fix:

- A label change needs 50% overlap, because a new label means the detector changed its mind about the same box.
- A pair with a static label needs 10% overlap. 30% was too strict: it broke matches the tracker keeps through camera pans.
- Units still match without overlap.

**Memory.** Most of the wrong objects left after that had one cause too. A static object's box was missed just as a neighbour of the same kind came back into view. The tracker had forgotten the neighbour after a second, so the neighbour looked new, and the missed object's track took it at 10-36% overlap. The truth never forgets static objects, so it knew the neighbour. Static tracks now live for 5 s, while units keep the one-second memory. Memories of 3 s and 10 s scored within 0.1 points of 5 s.

Scored as the held-out verdict, on the same four clips:

| Check | Before | With overlap | With memory | IoU |
|---|---|---|---|---|
| Still the same object after 2 s | 93.7% | 94.7% | 96.5% | 94.4% |
| Wrong object | 0.34% (27) | 0.11% (9) | 0.06% (5) | 0.05% (4) |
| Label accuracy | 86.84% | 86.89% | 86.89% | 86.88% |
| Label flips | 16.9 | 16.6 | 16.6 | 17.1 |

With both changes:

- **Match 1:** survival is 94.5% against 93.9% for IoU, with 5 wrong objects against 8.
- **Match 2:** static survival is 98.5% against 97.8%, with 1 wrong object against 4.
- **Match 3:** survival is 93.8% against 90.1%. Its 4 wrong objects, against none for IoU, are the only gap left.
- **All six clips:** 10 wrong objects against 12.
- **Generated scenes:** 88.95% against 88.82%.
- **Units:** survival is unchanged on every clip.

A stricter overlap for static objects while the camera is still cut wrong objects to 3. It also dropped match 1's survival to 92.9%, so it was left out.

Still open: match 3's wrong objects. This is the Mac recording, where the detector misplaces building boxes, so they probably need the detector fixed first.

Match 1's "same spot" losses are decided. Most of what remains there happens when the detector puts two labels on one box: Python keeps one ID per label, while the truth gives the box to one object arbitrarily. A click lands in the same place either way, so this could have been scored as a pass. On 2026-11-07 the rule was kept as it is: these still count as not surviving, and `verdict` now prints their share on a separate line.

The Rust port still waits for a new held-out match.

### Rules for match 4

Fixed on 2026-11-07, before match 4 was recorded. None of them change once the recording exists.

- **Tracker:** the Python tracker (`--name python`) as of `e0d8bf9`, "remember static objects for five seconds". `apps/agent/src/temporal_entities.py` stays unchanged until the verdict: its git blob must still be `0f29ad1c`, which survives rebases (`git rev-parse HEAD:apps/agent/src/temporal_entities.py`).
- **Checks:** the `[recorded]` rules in `gates.toml`, unchanged and pooled over every clip. Survival after 2 s must be at least IoU's, wrong objects at most IoU's, and label accuracy and flips at least as good as raw detections.
- **Same spot:** counts as not surviving. Its `info` line can't pass or fail anything.
- **Detector:** `aoe2_yolo_v12_nms736`, the model the agent plays with, fixed on 2026-11-09 before recording. `recorded.py prepare` refuses to detect a new clip unless the detection server's `/health` names it, and saves that answer as `detector.json` beside the clip.
- **Recording:** the Mac game, after the probe has set the zoom so a Town Center is 0.9-1.15x its width on the VM. Recording at the default zoom would test the tracker on detector output already known to be degraded.
- **Clips:** two 60 s clips, as for matches 1-3: the minute with the least camera motion and the minute with the most, starting on a 5 s grid and not overlapping. They are picked from pixel change alone, before any detector or tracker runs on the match, with `recorded.py pick`: each window scores the mean absolute grey change between consecutive frames, over the game area shrunk 8x. The still clip is the lowest-scoring window, and the moving clip the highest-scoring one that doesn't overlap it. The method was written down on 2026-11-09, after recording and before it ran.
- **Truth:** unit tracks are reviewed from `recorded.py strips`, which show no tracker output, and `verified.json` is committed before any tracker is scored on the clips.
- **Once:** `verdict` runs once. A fail spends match 4 like matches 2 and 3. A pass lets the rules move to screentrack, where the speed, parity and generated-scene gates then run on the port.

### Match 4 result

Run once on 2026-11-09, by the rules above. Python against IoU, pooled over both clips:

| Check | Python | IoU or raw | Result |
|---|---|---|---|
| Still the same object after 2 s | 99.6% | 98.7% | Pass |
| Wrong object | 0.06% (6 of 5,143) | 0.00% | Fail |
| Label accuracy | 99.66% | 99.67% | Fail |
| Label flips | 0.7 | 0.7 | Pass |

Labels count as one check, so the tracker failed two of three, and match 4 is spent. Units did far better than with IoU: 100% against 41.7% survival on the still clip, and 97.1% against 73.3% on the moving one. IDF1 was 89.6 against 78.8, and 92.7 against 84.5. The verdict's output is in `logs/tracker-benchmark/recorded/match-04-verdict.txt`, which is not committed.

Both fails came from single events:

- **Wrong object:** all 6 were one farm ID. Two touching farms' boxes shifted between frames, so the farm's own box was contested by a duplicate track and refused as ambiguous. The ID then took its second choice, the next farm, and held it while it was sampled every 0.2 s.
- **Labels:** one object-frame in about 19,000. Smoothing held "goose" on a deer for one frame longer than the raw detector. The other 62 label misses, mostly a house read as a mill, are the detector's own, and raw detections share them.

### After match 4

Development evidence, not a retest: the fix and the match 5 rules both came from studying matches 3 and 4.

**Fix.** An ID whose first choice is refused because another track also claims it now sits out the frame, instead of falling back to its second choice. A regression test rebuilds the two farms from the recorded boxes. With the fix, match 4 has no wrong objects, and matches 1-3 are unchanged except match 2's survival, which is 0.1 points lower.

**Match 3's two remaining swaps** have other causes:

- A house's box was relabelled "stable" while the building beside it got both "stable" and "house". The house's ID took the neighbour's "house" box at 16% overlap, because the label penalty outweighs nearly any overlap. Halving the penalty fixed both of match 3's swaps, but cost match 1 0.3 points of survival and match 3 0.03 points of label accuracy, so it was left out.
- Low-confidence boxes around 1,000 px wide covered a cluster of buildings, with labels flickering between farm, archery range and Town Center. That is detector output to filter before tracking, not a tracker rule.

### Rules for match 5

Fixed on 2026-11-09, before match 5 is recorded. Everything not listed here is unchanged from match 4.

**Two checks change, because both of match 4's fails were decided by single events.**

- **Wrong object counts swaps.** A swap is one published ID pointing at one other object, and sampling it again while it lasts doesn't add a second. Counting samples measured how long a swap lasted rather than how many happened. The check passes if Python has no more swaps than IoU plus one per 60 s clip. IoU almost never swaps, because it gives up on IDs instead, so "no more than IoU" had meant "none at all". The sample rate is still printed, as `info`.
- **Label accuracy may trail raw detections by 0.05 points,** about one object-frame in 2,000. Flips still get no tolerance.

Both are in the `[recorded]` table of `gates.toml` (`wrong_incidents_per_clip`, `label_accuracy_tolerance`). They were written after two failed matches, so only match 5 can judge them. With today's tracker, matches 1-4 all pass them, and match 3 sits exactly at its limit with 2 swaps. Those matches shaped both the fix and the rules, so that says nothing either way.

The other rules stay as they were:

- **Tracker:** the Python tracker as of `17f1250`. `apps/agent/src/temporal_entities.py` stays unchanged until the verdict: its git blob must be `c712bf45`.
- **Same, as for match 4:** the detector `aoe2_yolo_v12_nms736`, the Mac game at the probe's zoom, two 60 s clips picked by `recorded.py pick`, reviewed truth committed before any tracker is scored, and one `verdict` run.

### Recording another match

Record the Mac game from Terminal.app, which needs Screen Recording permission. The fragmented format keeps frame times if a recording dies: in a test, killing the recording lost only the last unwritten fragment, about two seconds, rather than the whole file. Ten minutes takes about 10 GB at the rate below, so start with at least 30 GB free, because swap also uses the disk.

1. On the Mac, set Displays > Preset to Internet & Web (sRGB).
2. Record with this command. Try it first with `-t 30` while the game is busy, and expect about 9.5 fps, with nearly every gap at 67 or 133 ms.

   ```bash
   ffmpeg -y -f avfoundation -capture_cursor 0 -framerate 15 -pixel_format nv12 -i "Capture screen 0:none" -t 600 -c:v prores_videotoolbox -profile:v standard -movflags +frag_keyframe+empty_moov -min_frag_duration 2000000 match.mov
   ```

3. Cut clips with `recorded.py prepare --top 66`. Leave out `--colour`, because no paired capture exists for the Mac game yet. Limits follow each clip's own frame rate, so VM clips rebuild unchanged.
4. Draw review strips with `recorded.py strips`, and save the tracks that pass as `verified.json` beside the clip.
5. Rebuild with `recorded.py prepare`, then judge held-out matches once with `recorded.py verdict --name python`, following the latest rules above.
