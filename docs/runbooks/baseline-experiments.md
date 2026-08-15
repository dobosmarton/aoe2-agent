# Runbook: Record the P0.1 Baseline (experiments/results.tsv)

**Goal:** 3–5 full games with the current stack recorded in the ledger, so every
subsequent change (IMPROVEMENT-PLAN.md P1–P4) is measured against a baseline
instead of vibes.

## Prerequisites

- Windows VM with AoE2 DE running in a skirmish game (see the VM setup notes),
  repo synced to the commit you want to baseline.
- Detection server running on the Mac host (`just server --model <path-to-aoe2_yolo_v9.onnx>`)
  and `AOE2_DETECTION_HOST` pointing at it.
- `AOE2_LLM_API_KEY` set.
- `AOE2_OCR_BACKEND=template` set (T-202): millisecond digit reads via the
  harvested `templates/3024x1672/` glyphs instead of ~2-4 s of RapidOCR per
  field; age text stays on RapidOCR, sampled every 5 ticks. Only set this where
  digit templates exist for the capture resolution — the template backend
  refuses to run without them (the default `rapidocr` needs no assets).

## Run

```bash
# 3 games, 20 minutes each (adjust n / add --time-budget as needed)
just experiment-baseline 3 --time-budget 1200
```

Each game appends a row to `experiments/results.tsv` (composite score, survival,
population, age, economy, action-success rate, end reason, turn count, git SHA)
and, when the memory chain is enabled, extracts cross-game memories (P4.1).

## Verify

```bash
just experiment-gate          # requires a row at HEAD — should PASS now
```

The ledger (`experiments/results.tsv`) is deliberately NOT committed — it is
machine-local, and the VM that runs the games is its source of truth. Run the
gate on that machine; copy the file (e.g. into a run's `logs/` folder) when a
snapshot needs to be shared.

## Ongoing discipline

Before merging any behavior-affecting change, run at least one game with a
description naming the change:

```bash
just experiment "P1.3 villager job classes"
just experiment-gate          # merge gate: fails without a row at HEAD
```

While the ledger is being bootstrapped, `just experiment-gate --any` only
requires a non-empty ledger.

## Free side quests during baseline games

- **P0.2:** the games produce screenshots under `logs/` — harvest frames across
  ages/biomes for the ≥200-image frozen real eval set (label in CVAT,
  bootstrap with `prelabel.py --open-vocab`).
- **P4.1:** hand-review the first few extracted memories in `memories/` to tune
  the extraction prompt.
