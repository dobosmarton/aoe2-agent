# Record baseline experiments

Record scored games for the current commit in `experiments/results.tsv`, so later changes can be compared against them and `just experiment-gate` passes. How games are scored and what makes a run invalid is in [autoresearch](../autoresearch/overview.md).

## Before you start

- AoE2 is running a skirmish on the machine the agent controls, and the repo there is at the commit you want to record.
- A detector is reachable: `AOE2_DETECTION_HOST` points at a running detection server (`just server --model <model.onnx or .mlpackage>`, port 8420), or the local model named by `AOE2_DETECTION_MODEL` is installed. Without one the run fails at start. See [detection serving](../detection/serving-and-evaluation.md).
- `AOE2_LLM_API_KEY` and `TYPESAFE_API_KEY` are set.
- `AOE2_GAME_PROFILE` points at a profile JSON for this setup with `roster_verified`, `hotkeys_verified` and `ownership_verified` all `true`. Examples are in `apps/agent/profiles/` (`magyars_highland_4v4.example.json` for Windows, `magyars_highland_4v4.macos.example.json` for macOS). Without a verified profile every run is invalid.

## Record

```bash
just experiment-baseline 3 --time-budget 1200
```

This runs `python -m autoresearch.game_runner` three times with descriptions `baseline run 1/3` to `3/3`. Start a fresh game before each run. Each run prints its scores and appends one ledger row.

A run that prints `Run invalid for scoring` was logged with `accepted=false` and git SHA `(reverted)`. It does not count for the gate. Check that the profile is verified, then look in the run's log for actions that ended `uncertain`.

## Verify

```bash
just experiment-gate
```

`PASS: N game(s) recorded at <sha>` means the ledger has a valid row at HEAD. `just experiment-gate --sha <sha>` checks another commit, and `--any` only requires a non-empty ledger.

The ledger is ignored by git and exists only on the machine that played the games, so run the gate there. Copy `experiments/results.tsv` elsewhere if someone else needs the numbers.

## For each later change

```bash
just experiment "<what changed>"
just experiment-gate
```

Each valid game can add memory files in `apps/agent/memories/`. Read them; delete any that give bad advice (see [mutation and memory](../autoresearch/mutation-and-memory.md)).
