# Deployment

First-time setup, from a clean checkout to a running agent, for the two topologies the code supports: the game and agent in a Windows VM with the detection server on the macOS host, and the native macOS game with the agent on the same Mac. The [agent overview](../agent/overview.md) explains how the pieces fit together.

## Prerequisites

| Machine | Needs |
|---|---|
| macOS host (both topologies) | Apple Silicon for the CoreML execution provider, [uv](https://docs.astral.sh/uv/), [just](https://github.com/casey/just) |
| Windows VM | AoE2:DE, uv, an x64 build of Python 3.11, and a network route to the host |
| Native macOS game | AoE2:DE for Mac, full screen on the built-in 3024x1964 Retina panel |
| Credentials | `AOE2_LLM_API_KEY` for the chosen wire (OpenAI by default) and `TYPESAFE_API_KEY` |

The repo pins Python 3.11 in `.python-version`. On a Windows VM running on ARM64, use the x64 interpreter anyway: the `screentrack` wheel exists for Windows AMD64 only, and without it the agent falls back to the slower Python tracker.

## Install

On every machine that runs Python code:

```bash
git clone <repo-url> aoe2-llm-arena
cd aoe2-llm-arena/agent
uv sync                      # same as: just install
```

`uv sync` installs every workspace member editable into `.venv/`, plus the dev tools. On Windows, point it at the x64 interpreter if uv would otherwise pick another one: `uv sync --python C:\path\to\x64\python.exe`. The agent needs no JavaScript; `bun install` is only for the dashboard and landing apps ([running the UI locally](../web/running-locally.md)).

## Detector weights

Weights are gitignored, except the old `aoe2_yolo_v5` files. The served model is `aoe2_yolo_v12_nms736.onnx`, published as `model.onnx` at tag `v12` of the `martondobos/aoe2-entity-detector` model registry. Put it on the machine that runs detection (the Mac in both topologies; the VM needs none):

```bash
curl -L -o packages/detection/src/inference/models/aoe2_yolo_v12_nms736.onnx \
  https://huggingface.co/martondobos/aoe2-entity-detector/resolve/v12/model.onnx
shasum -a 256 packages/detection/src/inference/models/aoe2_yolo_v12_nms736.onnx
```

The hash must equal `model_sha256` in `packages/detection/registry/aoe2-entity-detector/inference-contract.json` (`838c1a10...a209`). Without the file, an in-process detector loads the newest bundled weights instead (v5) and logs a warning. Model details are in [detector](../detection/detector.md) and [serving and evaluation](../detection/serving-and-evaluation.md).

## Environment file

```bash
cp .env.example .env
```

Fill in the values for your topology and leave the arena entries alone unless you run the arena stack. The agent loads the nearest `.env` at or above `apps/agent/src/`, which is this root file unless a stray `apps/agent/.env` exists. A variable exported in the shell overrides the file.

| Variable | Windows VM | Native macOS |
|---|---|---|
| `AOE2_LLM_API_KEY` | Required | Required |
| `TYPESAFE_API_KEY` | Required | Required |
| `AOE2_DETECTION_HOST` | `http://<host-ip>:8420` | Empty for in-process detection, or `http://127.0.0.1:8420` with a local server |
| `AOE2_GAME_PROFILE` | Optional | Required |

Every variable is listed in the [agent overview](../agent/overview.md#configuration).

## Windows VM with a host detection server

### On the macOS host

Start the server. It binds `0.0.0.0:8420` by default:

```bash
just server --model packages/detection/src/inference/models/aoe2_yolo_v12_nms736.onnx
curl -s http://localhost:8420/health
# {"status":"ok","model_path":"...","backend":"onnx_coreml","num_classes":65}
```

Find the address the VM can reach. The interface name and subnet depend on the hypervisor and its history, so discover it instead of reusing an old value:

```bash
for i in $(ifconfig -l); do
  case $i in vmnet*|bridge*) echo "$i $(ifconfig $i | awk '/inet /{print $2}')";; esac
done
arp -an | grep bridge      # the VM's address shows which bridge it sits on
```

The Mac is `.1` on the VM's subnet; `ipconfig` on Windows shows that subnet too. If macOS asks whether Python may accept incoming connections, allow it.

### On the VM

Install as above, write `.env` with `AOE2_DETECTION_HOST=http://<host-ip>:8420`, then check the route from Command Prompt:

```cmd
curl http://<host-ip>:8420/health
```

### Game settings

The agent finds the window by its exact title, `Age of Empires II: Definitive Edition`. Make the game area 3024x1672: the shipped HUD calibration (`apps/agent/src/resource_ocr_assets/calibration.3024x1672.yaml`) and the Windows profile expect that size, and any other size makes the agent locate the resource bar from each frame. Play as player 1, blue, because the ownership classifier treats blue as the agent's own colour. Pick a mostly green map with a temperate or highland look, such as Highland: the detector works best on grassland, and on snow it misses sheep and trees.

A profile is optional on Windows. To use one, copy `apps/agent/profiles/magyars_highland_4v4.example.json`, record the real lobby (the `map_name` field takes any map name), and set `AOE2_GAME_PROFILE` to it.

### First run

Start a skirmish, wait until the Town Center is on screen, then from Command Prompt in the repo:

```cmd
uv run --package gameplay-agent aoe2-agent
```

`just agent` runs the same command. Before the first frame the agent scrolls to zoom in, then presses `,` and `g` to send the scout exploring. `--iterations N` stops after N perception frames.

## Native macOS game

1. In System Settings > Privacy & Security, allow Terminal.app under Screen Recording and under Accessibility, and run the agent from Terminal.app.
2. Select a keyboard layout that types `a` to `z`, `0` to `9`, `.` and `,` without Option. The agent remaps keys to the active layout at startup and stops with `keyboard_layout_cannot_type` if it cannot.
3. Set the display colour profile to Internet & Web (sRGB), because the detector learned sRGB colours. Preflight reports this as a warning, not a failure.
4. Copy `apps/agent/profiles/magyars_highland_4v4.macos.example.json`. Keep `platform: macos` and `capture_height: 1898`, and set `command_button` (the example uses `left`), `zoom_steps` (steps in from full zoom-out, 0 to 30) and `modifier_map` to match the game's bindings. Point `AOE2_GAME_PROFILE` at the copy.
5. Put the weights in place and leave `AOE2_DETECTION_HOST` empty to detect in-process, or run `just server` on the same Mac and point the variable at `http://127.0.0.1:8420`.
6. Start a game full screen on a mostly green map, such as Highland, select a villager or the Town Center, close every menu, and run preflight from Terminal.app:

   ```bash
   uv run --package gameplay-agent python -m gameplay_agent.preflight --profile my-macos-profile.json
   ```

   It prints each check as JSON and sends no clicks or keys, but it moves the cursor to the middle of the game and reads it back. It exits 1 while any blocking check fails, including the roster, hotkey and ownership-colour flags of an unverified profile; fix the window, HUD, permission, popup and geometry checks before a first run. `scripts/macos_probe.py` (`info`, `capture`, `snap`, `input`) probes capture and input in more depth and writes to `logs/mac-probe/`.
7. Run `just agent` from Terminal.app. The agent brings the game forward once, refuses to start if a menu dims the screen, and sets the profile's zoom. After that it sends input only while the game is the front window, so switching to another app stops it.

## Verification

A healthy start logs these events, on the console and in `logs/runs/<UTC start>.jsonl`:

| Event | Expect |
|---|---|
| `run_log_file` | Path of the run's JSON-lines file |
| `provider_created` | `wire` and executor `model` |
| `game_profile_loaded` | The profile path, or `None`, and the command button |
| `tracker_backend` | `backend=native` |
| `detector_initialized` | `mode=remote server=http://...` on the VM; `mode=onnx` in-process on the Mac. `mode=mock` means no weights were found |
| `game_loop_start` | `detection=True` |
| `wire_warmed` | Seconds spent importing the SDK |
| `hud_observation_published` | Repeats as the HUD is read |
| `policy_advice_outcome` | `status=applied` most of the time; `timed_out`, `fallback` or `rejected` show why the fallback chose |
| `act_execution_outcome` | `success=True` for most actions |
| `strategist_goals_updated` | A goal count, soon after the first frame and then about every 25 s |

`loop_latency` gives per-loop timings with a breakdown by phase. Stop the agent with Ctrl+C. To record a scored game in the experiment ledger, see [Windows VM agent bring-up](./windows-vm-agent-bringup.md#recording-a-game).
