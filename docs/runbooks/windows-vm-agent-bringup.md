# Windows VM agent bring-up

The everyday procedure for starting the agent on a VM that is already set up: the game and agent in the Windows VM, the detection server on the macOS host. First-time setup is in [deployment](./deployment.md).

## Fast path

On the Mac:

```bash
cd aoe2-llm-arena/agent
just server --model packages/detection/src/inference/models/aoe2_yolo_v12_nms736.onnx
```

The VM-facing address can change between sessions. If the VM cannot reach the server, rediscover it as described in [deployment](./deployment.md#on-the-macos-host) and update `AOE2_DETECTION_HOST` in the VM's `.env`.

On the VM, in Command Prompt:

```cmd
cd %USERPROFILE%\aoe2-llm-arena\agent
curl http://<host-ip>:8420/health
```

Start a skirmish as player 1 (blue) on a mostly green map such as Highland, wait for the Town Center, then:

```cmd
uv run --package gameplay-agent aoe2-agent
```

The start worked when the log shows `detector_initialized mode=remote`, `game_loop_start`, then repeating `policy_advice_outcome` and `act_execution_outcome` lines, and `strategist_goals_updated` soon after the first frame. Stop with Ctrl+C.

## Recording a game

`aoe2-agent` plays but records nothing in the experiment ledger. To record a game, run the experiment wrapper, which plays the same game through `autoresearch.game_runner` and appends a row to `experiments/results.tsv`:

```cmd
just experiment "what changed since the last row"
```

It takes `--time-budget SECONDS`, `--max-iterations N` (perception frames) and `--overlay`. Its detector has no local fallback, so a detection outage ends the game with `detector_unavailable`. Unless `AOE2_GAME_PROFILE` names a profile with the roster, hotkeys and ownership colours verified, the runner logs `recorded_profile_unqualified` and exits with "Run invalid for scoring". Leave `AOE2_SAVE_SCREENSHOTS` on: the frames saved under `logs/screenshots/` are what `scripts/calibrate_resource_bar.py` builds a HUD calibration from. Drop `--overlay` for timed runs. The ledger and scoring are in [baseline experiments](./baseline-experiments.md) and [autoresearch](../autoresearch/overview.md).

## Symptoms

| Symptom | Cause | Fix |
|---|---|---|
| `missing_api_key` and exit status 1 | `AOE2_LLM_API_KEY` or `TYPESAFE_API_KEY` is empty in the process. A stray `apps/agent/.env` shadows the root `.env`, because the agent reads the nearest file at or above `apps/agent/src/` | Set the key named in the message, or delete the stray file |
| `ValueError: unknown AOE2_LLM_WIRE=...` at startup | The wire name is not `openai`, `zen` or `anthropic` | Fix the value; case and surrounding spaces are ignored |
| `detector_initialized` shows `mode=onnx` or `mode=mock`, not `mode=remote` | `AOE2_DETECTION_HOST` is empty, so the VM runs the detector itself; `mock` means no weights at all | Set `AOE2_DETECTION_HOST` in the root `.env` |
| `remote_detection_failed`, then `falling_back_to_local_detector` or `remote_detection_unavailable_empty_result` | The server is down, bound elsewhere, or behind the macOS firewall, or the host address changed. The agent rechecks `/health` every 30 s | `curl` the health URL from the VM, rediscover the address, allow incoming connections on the Mac, and keep the server on its default `--host 0.0.0.0` |
| `game_not_found` and the run ends | No window titled `Age of Empires II: Definitive Edition` | Start the game before the agent |
| Repeated `could_not_focus_game`, then `focus_lost_giving_up` | The agent could not bring the game forward 15 times in a row, 1 s apart | Click the game window once, run the agent from Command Prompt, and keep other windows off the game |
| `ocr_autodetect` on every HUD read, or `ocr_no_calibration_autodetect_failed` | The capture size has no calibration; only 3024x1672, 3024x1898 and 3024x1964 ship one | Size the game area to 3024x1672, or build a calibration from saved frames with `scripts/calibrate_resource_bar.py` |
| `tracker_native_unavailable` | `screentrack` is not installed, usually because the venv uses ARM64 Python | Recreate the venv on x64 Python 3.11: `uv sync --python <x64 python.exe>` |
| `warning: Using incompatible environment (.venv) due to --no-sync` | `--no-sync` reused a venv built for another Python; the repo pins 3.11 | Drop `--no-sync`, or run `uv sync` first |
| Every executor call logs `llm_api_error` with a 400 response | `AOE2_TEMPERATURE` is set while the wire serves `gpt-5.6`, which accepts only its default | Unset `AOE2_TEMPERATURE` |
| `policy_advice_outcome status=timed_out reason=provider_timeout` on most frames | TypeSafe answers slower than `AOE2_POLICY_TIMEOUT` (2.0 s); the fallback picks every action | Check the VM's network, or raise `AOE2_POLICY_TIMEOUT` |
| `game_profile_unusable` | The profile's `platform` does not match the machine | Use a profile with `"platform": "windows"` |
| `api_cost` reports $0.00 and `pricing_unknown_model` appears | The model name is missing from `apps/agent/src/providers/pricing.py` | Add the exact name there ([LLM providers](../agent/llm-providers.md#cost-accounting)) |
