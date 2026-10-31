# Tracker benchmark

**Faster: yes. Better: no.**

Run on arm64 at commit `8185584`, over 12 generated scenes from real annotated frames, with the real detector. Units move as pasted sprites, so recorded footage is still needed for the final word.

## Gates

| Result | Gate | Question | Evidence |
|---|---|---|---|
| pass | parity | Same decisions as Python? | 0 differing frames (first: none) |
| pass | overhead | Inside the perception budget? | camera 13.7 + tracker 3.11 = 16.8 ms; Python 108.7 ms; limit 20 ms |
| pass | speed | Faster on real frames? | low 43x (at least 35x); typical 50x (at least 49x); high 38x (at least 34x); peak 30x (at least 25x) |
| pass | spec_target | Meets the original goal? | 1.86 ms in the high band; limit 2 ms |
| **FAIL** | target_survival | Do IDs shown to the LLM still work later? | 18.8% against 88.8% for the baseline |
| pass | wrong_target | Could an ID point at the wrong object? | 0.00% against 0.00% for the baseline |
| **FAIL** | recovery | Does one bad frame heal? | never recovers |
| pass | labels | Does smoothing help? | no change: accuracy 93.45% against 93.45%; flips 4.2 against 4.2 per 1,000 |
| **FAIL** | economy_view | Is the agent's picture steadier? | job count error 2.94 against 2.94 |
| **FAIL** | event_loop | Is the agent less blocked? | p99 lag 1.58 ms against 0.79 ms |
| pass | stability | Does it hold up over a game? | p95 drift -69.9%; memory growth 0.0 MB |

## Agent outcomes

| Contender | Survival 1 s | Survival 2 s | Survival 5 s | Wrong target | IDF1 | IDF1 mobile | IDF1 static | Switches per 1,000 | Label accuracy | Flips per 1,000 | Job error |
|---|---|---|---|---|---|---|---|---|---|---|---|
| raw | 0.0% | 0.0% | 0.0% | 0.00% | 3.3% | 4.1% | 2.9% | 1000 | 93.45% | 4.2 | 2.94 |
| oracle | 100.0% | 100.0% | 100.0% | 0.00% | 100.0% | 100.0% | 100.0% | 0 | 100.00% | 0.0 | 2.86 |
| iou | 94.5% | 88.8% | 73.1% | 0.05% | 79.9% | 64.8% | 86.7% | 82 | 93.45% | 4.2 | 2.94 |
| python | 19.7% | 18.8% | 16.5% | 0.00% | 19.8% | 15.6% | 21.7% | 812 | 93.45% | 4.2 | 2.94 |
| screentrack | 19.7% | 18.8% | 16.5% | 0.00% | 19.8% | 15.6% | 21.7% | 812 | 93.45% | 4.2 | 2.94 |
| candidate | 57.8% | 52.2% | 39.4% | 0.00% | 46.5% | 36.6% | 50.9% | 445 | 93.44% | 4.3 | 2.94 |

## Milliseconds per frame at steady state

| Band | Python prepare p95 | Python associate p95 | Python total p95 | screentrack prepare p95 | screentrack associate p95 | screentrack total p95 |
|---|---|---|---|---|---|---|
| low | 63.59 | 0.66 | 64.14 | 1.28 | 0.12 | 1.55 |
| typical | 93.18 | 2.26 | 95.01 | 1.70 | 0.21 | 1.89 |
| high | 67.68 | 3.59 | 70.68 | 1.57 | 0.31 | 1.86 |
| peak | 86.63 | 6.78 | 92.94 | 2.71 | 0.51 | 3.11 |

## Benchmark controls

| Result | Control | Evidence |
|---|---|---|
| pass | oracle is perfect | IDF1 1.000, survival 1.000 |
| pass | raw scores lowest | survival 0.000 |
| pass | A/A is unbiased | 1.01x, 0.93x, 0.94x, 0.99x |
| pass | injected slowdown is flagged | 38.6 ms against 20 ms |

Camera analysis p95 was 13.7 ms.

## Reading this run

Rechecks after the run, with the same code and scenes:

- **`event_loop` isn't a real problem, and the rule needs replacing.** An idle event loop shows a p99 lag of 1.2-1.6 ms, and screentrack measured 1.39 ms over six alternating runs, so it adds no measurable lag. Python measured 0.48 ms, lower than idle, because its long-running worker keeps the CPU awake.
- **`economy_view` is a tie.** The job count error is 2.9396 against 2.9375, a difference of 0.002 per frame.
- **`labels` passes only as a tie.** Smoothing changed nothing measurable in these scenes.
- **The stability drift figure is noise.** Three repeat runs gave -63%, +8% and -53%. Memory growth was 0.0 MB each time, which is the reliable part.

The two real failures are `target_survival` and `recovery`.
