# Deployment Guide: Mac + Windows VM Setup

Run the detection server on macOS (Apple Silicon) and the gameplay agent on a Windows VM. The agent sends screenshots to the Mac over HTTP for fast YOLO inference via CoreML/Neural Engine. (Screenshots leave the VM only for YOLO detection; the strategist reads the resource bar locally via OCR and sends Claude a text-only prompt — no image is sent to any LLM.)

```
┌─────────────────────────────┐         HTTP          ┌─────────────────────────────┐
│       macOS Host            │◄───────────────────────│       Windows VM            │
│                             │                        │                             │
│  Detection Server (:8420)   │  POST /detect          │  AoE2:DE (game)             │
│  CoreML / ONNX model        │  POST /detect/sahi     │  Gameplay Agent             │
│  ~15ms per tile inference   │  GET  /health          │  Screenshots → HTTP → Mac   │
│                             │                        │  Actions → pyautogui        │
└─────────────────────────────┘                        └─────────────────────────────┘
```

---

## Prerequisites

| Machine | Requirements |
|---------|-------------|
| **Mac** | Python 3.11+, Apple Silicon recommended |
| **Windows VM** | Python 3.11+ (x64 installer, NOT ARM64), AoE2:DE installed, VMware Fusion or similar |
| **Both** | Network connectivity between host and VM |
| **API Key** | Model API key for the selected adapter (`AOE2_LLM_API_KEY`) |

---

## Part 1: macOS Host — Detection Server

### 1.1 Clone and install

```bash
cd ~/Projects/home/aoe2-llm-arena/agent

# Create a venv (if not done already)
python3 -m venv venv
source venv/bin/activate

# Install server dependencies
pip install -r server/requirements.txt

# Optional: install CoreML support (recommended on Apple Silicon)
pip install coremltools
```

### 1.2 Verify model file

The ONNX model should be at:
```
detection/inference/models/aoe2_yolo_v9.onnx
```

If you have the `.pt` weights and want CoreML (faster on Apple Silicon):
```bash
# Export to CoreML (optional, ONNX works fine)
just export-coreml detection/inference/models/aoe2_yolo_v9.pt
```

### 1.3 Start the server

```bash
# Using justfile
just server --model detection/inference/models/aoe2_yolo_v9.onnx

# Or directly
python -m detection_server --model detection/inference/models/aoe2_yolo_v9.onnx --host 0.0.0.0 --port 8420
```

For CoreML model:
```bash
just server --model detection/inference/models/aoe2_yolo_v9.mlpackage
```

You should see:
```
INFO:     Model loaded: onnx_cpu (or coreml)
INFO:     Uvicorn running on http://0.0.0.0:8420
```

### 1.4 Verify the server is running

```bash
curl http://localhost:8420/health
```

Expected response:
```json
{"backend": "onnx_coreml", "classes": 60, "model": "aoe2_yolo_v9.onnx"}
```

### 1.5 Find your Mac's IP address

The VM needs to reach the Mac. Find the IP depending on your VM software:

**VMware Fusion** — the host is typically reachable at `192.168.64.1` from the VM. Verify:
```bash
# On the Mac
ifconfig vmnet8 | grep inet
# or
ifconfig bridge100 | grep inet
```

**Alternative** — check your Mac's local network IP:
```bash
ipconfig getifaddr en0
```

Note this IP (e.g., `192.168.64.1`). You'll need it in Part 2.

---

## Part 2: Windows VM — Game Agent

### 2.1 Transfer the code

Option A — Git clone:
```cmd
git clone <repo-url> aoe2-llm-arena
cd aoe2-llm-arena\agent
```

Option B — ZIP transfer:
```bash
# On Mac: create a zip of the agent directory
cd ~/Projects/home/aoe2-llm-arena
zip -r agent.zip agent/ -x "agent/venv/*" "agent/logs/*" "agent/.superset/*" "agent/*.tar.gz" "agent/*.pt" "agent/*.zip"

# Transfer to VM (replace VM_IP with your VM's IP)
scp agent.zip user@VM_IP:~/
```

Then on the VM:
```cmd
cd %USERPROFILE%
mkdir aoe2-llm-arena
cd aoe2-llm-arena
tar -xf %USERPROFILE%\agent.zip
cd agent
```

### 2.2 Set up Python environment

> **Important**: Use the **Python x64 installer** on Windows ARM64 VMs. ARM64 Python lacks many wheel packages.

```cmd
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` lives at the repo root and is generated from `uv.lock`
(`uv export --no-hashes --format requirements-txt --no-emit-project --all-packages --no-dev`).
Regenerate it after any dependency change, or the VM installs a stale set — the
`openai` SDK was missing from it for the whole first day of the adapter work.

If you get torch/scipy conflicts:
```cmd
pip install scipy numpy --force-reinstall
pip install -r requirements.txt
```

### 2.3 Set environment variables

**Command Prompt:**
```cmd
set AOE2_LLM_API_KEY=your-key-here
set AOE2_DETECTION_HOST=http://192.168.64.1:8420
```

**PowerShell:**
```powershell
$env:AOE2_LLM_API_KEY = "your-key-here"
$env:AOE2_DETECTION_HOST = "http://192.168.64.1:8420"
```

Replace `192.168.64.1` with your Mac's IP from step 1.5.

Optional tuning:
```cmd
set AOE2_LLM_WIRE=openai
set AOE2_LOOP_DELAY=0.3
set AOE2_EXECUTOR_EFFORT=low
set AOE2_STRATEGIST_INTERVAL=10
set AOE2_SAVE_SCREENSHOTS=true
```

`AOE2_LLM_WIRE` picks the adapter: `openai` (default, api.openai.com), `zen` (OpenCode Zen)
or `anthropic`. Supply the matching key in `AOE2_LLM_API_KEY`. A name outside that set stops
the agent at startup with `ValueError: unknown AOE2_LLM_WIRE=...` rather than falling back.

### 2.4 Verify connectivity to the Mac server

```cmd
curl http://192.168.64.1:8420/health
```

If this fails:
- Check the server is running on the Mac (step 1.3)
- Check firewall — allow incoming connections on port 8420
- Try pinging the Mac from the VM: `ping 192.168.64.1`

### 2.5 Start AoE2

1. Launch **Age of Empires II: Definitive Edition**
2. Start a **Single Player** → **Skirmish** match
3. Pick your civilization, set opponent to AI
4. Start the game and wait until you see your Town Center

### 2.6 Run the agent

```cmd
cd aoe2-llm-arena\agent
venv\Scripts\activate
python -m gameplay_agent
```

Or with options:
```cmd
:: Run limited iterations
python -m gameplay_agent --iterations 50

:: Single test iteration (no action execution)
python -m gameplay_agent --test
```

You should see logs like:
```
detector_initialized         mode=remote server=http://192.168.64.1:8420
game_loop_start              detection=True executor_model=gpt-5.6-luna strategist_model=gpt-5.6-terra
iteration_start              iteration=1
screenshot_captured           width=1920 height=1080
detection_complete           entity_count=12
strategist_goals_updated     turn=1 goal_count=4
llm_response                 iteration=1 action_count=3
actions_executed             iteration=1 total=3 successful=3
```

---

## Part 3: Troubleshooting

### Server won't start

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named 'server'` | Run from the `agent/` directory: `cd agent && python -m detection_server ...` |
| `onnxruntime` import error on macOS | `pip install onnxruntime` (not `onnxruntime-gpu`) |
| CoreML model fails to load | Fall back to ONNX: `--model path/to/model.onnx` |

### Agent can't connect to server

| Problem | Fix |
|---------|-----|
| `Connection refused` | Verify server is running and bound to `0.0.0.0` (not `127.0.0.1`) |
| `Connection timed out` | Check firewall on Mac — allow port 8420. On macOS: System Settings → Network → Firewall |
| Wrong IP | Re-check with `ifconfig` on Mac. VMware uses `vmnet8` or `bridge100` |

### Agent can't find the game window

| Problem | Fix |
|---------|-----|
| `game_not_found` | AoE2 must be running and visible. Don't minimize it |
| `could_not_focus_game` | Click the game window once, then restart the agent |
| Coordinates are off | Make sure the game runs **windowed** or the agent captures the right monitor |

### Detection quality issues

| Problem | Fix |
|---------|-----|
| Few entities detected | The agent zooms in on turn 1. If entities are tiny, they may be too far away |
| False positives | Per-class thresholds in `packages/detection/src/inference/thresholds.py` can be tuned |
| Slow detection | Use CoreML model on Mac for ~15ms/tile vs ~100ms/tile with ONNX CPU |

### Agent falls back to local detection

The remote detector logs `remote_detector_unavailable` and falls back to local ONNX. This is slower but works. Check server connectivity to fix.

---

## Part 4: Configuration Reference

### Environment Variables

| Variable | Default | Where | Purpose |
|----------|---------|-------|---------|
| `AOE2_LLM_API_KEY` | — | VM | Model API key for the selected adapter (required) |
| `AOE2_LLM_WIRE` | `openai` | VM | Adapter: `openai`, `zen` (OpenCode Zen) or `anthropic`. An unknown name raises at startup |
| `AOE2_LLM_BASE_URL` | `""` | VM | Endpoint override; empty uses the adapter's own |
| `AOE2_DETECTION_HOST` | `""` | VM | Detection server URL, e.g. `http://192.168.64.1:8420` |
| `AOE2_MODEL` | `gpt-5.6-luna` | VM | Executor LLM model (fast; runs every turn) |
| `AOE2_EXECUTOR_EFFORT` | `low` | VM | Executor effort (`low`/`medium`/`high`) |
| `AOE2_STRATEGIST_MODEL` | `gpt-5.6-terra` | VM | Strategist LLM model (strong; runs every 3-10 turns) |
| `AOE2_STRATEGIST_INTERVAL` | `10` | VM | Run strategist every N turns |
| `AOE2_LOOP_DELAY` | `0.3` | VM | Seconds between game loop iterations |
| `AOE2_SAVE_SCREENSHOTS` | `true` | VM | Save screenshots to `logs/` |

The 3 model defaults follow `AOE2_LLM_WIRE`, because a model name belongs to its vendor. `AOE2_LLM_WIRE=anthropic` defaults the executor to `claude-haiku-4-5` and the strategist to `claude-sonnet-5`. `config._MODELS_BY_WIRE` holds the table.

### Server CLI Flags

```
python -m detection_server --model PATH --host HOST --port PORT
```

| Flag | Default | Purpose |
|------|---------|---------|
| `--model` | (required) | Path to `.onnx` or `.mlpackage` model |
| `--host` | `0.0.0.0` | Bind address |
| `--port` | `8420` | Bind port |

### Agent CLI Flags

```
python -m gameplay_agent [--test] [--iterations N] [--overlay]
```

| Flag | Purpose |
|------|---------|
| `--test` | Single iteration, no action execution |
| `--iterations N` | Stop after N iterations |
| `--overlay` | Show live overlay on the game window: entity detection boxes (colored by class) **and** the resource-bar OCR reading regions (gold boxes over wood/food/gold/stone/population) for verifying calibration |

---

## Quick Start Cheatsheet

**Mac (Terminal 1):**
```bash
cd ~/Projects/home/aoe2-llm-arena/agent
source venv/bin/activate
just server --model detection/inference/models/aoe2_yolo_v9.onnx
```

**Windows VM (Command Prompt):**
```cmd
cd aoe2-llm-arena\agent
venv\Scripts\activate
set AOE2_LLM_API_KEY=your-key-here
set AOE2_DETECTION_HOST=http://192.168.64.1:8420
python -m gameplay_agent
```

---

## Beyond the real-game tier

This guide covers the Mac + Windows VM setup for the **real-game agent**. If you also want to bring up the synthetic-arena tier (compose stack with Langfuse / Redis / Postgres / MinIO / ClickHouse, the arena CLI for offline evaluation, or the web UI for replay and fork), see:

- [Synthetic Arena infrastructure](../README.md#synthetic-arena-infrastructure-optional) in the root README — env-var template and `just arena-infra-up`.
- [Runbook: Redis broker operations](./runbooks/redis-broker-ops.md) — bringing up Redis, rotating the password, inspecting streams.
- [Runbook: Windows VM agent bring-up](./runbooks/windows-vm-agent-bringup.md) — fast-path version of this guide plus a symptom matrix.
- [Chapter 14 — Arena Overview](./part6-evaluation-arena/14-arena-overview.md) and [Chapter 21 — Running the UI Locally](./part7-arena-web/21-running-the-ui-locally.md) — what to run once the infra is up.
