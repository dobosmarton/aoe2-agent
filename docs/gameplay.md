Topology: macOS host runs the detection server; Windows VM runs AoE2 + the agent. Screenshots cross the network
  only for YOLO; the strategist OCRs the bar locally (text-only to Claude).

  Phase 0 — One-time prerequisites

  - [ ] Mac: uv installed, repo at ~/Projects/home/aoe2-llm-arena/agent, just install run (uv syncs all workspace
  members).
  - [ ] Windows VM: Python x64 (not ARM64), AoE2:DE installed, network to the Mac. Get the code on the VM and
  install deps (uv uv sync, or pip from the generated requirements.txt if uv is fiddly on the VM).
  - [ ] Anthropic API key ready (sk-ant-…).
  - [ ] VM is current: git pull on the VM so it has the OCR feature commit (2d09f58) — without it the strategist
  still expects the old path.

  Phase 1 — Mac: start the detection server

  - [ ] Start it: just server --model packages/detection/src/inference/models/aoe2_yolo_v5.onnx
  (use your local v6 export if you have one — the agent's detection_imgsz=640 matches v6's training scale.)
  - [ ] Health check: curl http://localhost:8420/health → expect {"backend": …, "classes": 60, "model": …}. Note
  which model/classes it reports.
  - [ ] Find the Mac IP the VM uses: ipconfig getifaddr en0 (or VMware: ifconfig vmnet8 | grep inet, usually
  192.168.64.1).
  - [ ] macOS firewall allows inbound on 8420.

  Phase 2 — Windows VM: configure the agent

  - [ ] Set env (Command Prompt):
    - [ ] $env:AOE2_LLM_API_KEY=sk-ant-…
    - [ ] $env:AOE2_DETECTION_HOST=http://<MAC_IP>:8420
    - [ ] $env:AOE2_SAVE_SCREENSHOTS=true  ← leave on; this run also harvests training frames to logs/
    - [ ] (optional) $env:AOE2_STRATEGIST_INTERVAL=10, $env:AOE2_EXECUTOR_EFFORT="low", $env:AOE2_OCR_BACKEND="rapidocr"
  - [ ] Verify connectivity from the VM: curl http://<MAC_IP>:8420/health (if it fails: firewall / server bound to
  0.0.0.0 / ping <MAC_IP>).

  Phase 3 — AoE2 game setup (matters for OCR)

  - [ ] Launch AoE2:DE → Single Player → Skirmish vs an AI opponent.
  - [ ] Resolution: prefer the validated 3024×1672 (windowed) so the committed hand-calibration applies. Any size
  works via runtime auto-calibration — you'll just confirm it in Phase 4.
  - [ ] Keep the window visible and unminimized (the agent finds it by title and focuses it).
  - [ ] Start the match; wait until your Town Center is on screen.

  Phase 4 — Launch the agent + first-minute sanity checks

  - [ ] Run it: just agent  (or uv run --package gameplay-agent aoe2-agent). First time, prefer a smoke run: just
  agent --test (one iteration, no clicks), then just agent --iterations 50.
  - [ ] Watch the logs for a healthy startup:
    - [ ] detector_initialized mode=remote server=http://<MAC_IP>:8420 (not falling back to local)
    - [ ] screenshot_captured width=… height=… ← confirm it's your game resolution
    - [ ] ocr_readings food=… wood=… gold=… stone=… population=… ← the OCR check — values must match the on-screen
  bar. (If you see ocr_autodetect … it derived geometry live; if numbers look wrong, that's the signal to
  recalibrate.)
    - [ ] strategist_goals_updated turn=1 goal_count=… and actions_executed … successful=…
  - [ ] Stop anytime with Ctrl-C.

  Phase 5 — After the game (turn it into data)

  - [ ] Frames are in logs/ on the VM (timestamped JPEGs). Pull them back to the Mac for the detection-retrain
  pipeline.
  - [ ] Note where it struggled (economy vs combat) — that's your baseline and tells you which classes/scenarios to
  capture next.
