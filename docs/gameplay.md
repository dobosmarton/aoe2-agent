# Run the screen-controlled agent

The Windows VM runs AoE2:DE, game capture, HUD reading, and input. The macOS host runs the remote detection server. Selected captures cross the network for object detection; local HUD readings and model decisions stay on the VM.

1. Install the Python workspace on both machines with `uv sync`. Set `AOE2_LLM_API_KEY` and `TYPESAFE_API_KEY` on Windows.
2. On the Mac, start the detector with the checkpoint used by the agent: `just server --model packages/detection/src/inference/models/aoe2_yolo_v9.onnx`. Check `http://localhost:8420/health`.
3. On Windows, set `AOE2_DETECTION_HOST=http://<MAC_IP>:8420` and confirm that endpoint is reachable from the VM.
4. Start a supported game setup, bring the game window to the foreground, and run the read-only profile qualification: `uv run --no-sync python -m gameplay_agent.preflight --profile my-4v4-profile.json`. See [agent README](../apps/agent/README.md) for the required roster, HUD, and binding checks.
5. Start `just agent` (or run a bounded recorded experiment via `autoresearch.game_runner`). Watch the log for detector availability, valid HUD readings, and confirmed action effects. Stop with Ctrl-C if the readings or targets do not match the screen.

The agent publishes spatial detections and HUD readings independently. A camera-moving command invalidates old coordinates; a dependent click waits for a fresh post-movement detection. There is no unchanged-frame prediction cache. See [streaming perception](streaming-perception.md) and [game loop pipeline](part1-architecture/02-game-loop-pipeline.md) for the current flow.
