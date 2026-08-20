# Runbook — Windows VM Agent Bring-up

The real-game tier runs on a Windows VM with AoE2:DE; the macOS host runs the detection server. This is the abbreviated, accumulated-experience version. For the full first-time walkthrough see [`docs/deployment-guide.md`](../deployment-guide.md).

## Prereqs (one time)

- VMware Fusion (or similar) with a Windows 10/11 VM, AoE2:DE installed, mouse capture enabled.
- **Python x64 installer on the VM**, not ARM64. ARM64 Python lacks wheels we need; you'll lose half an afternoon discovering this.
- macOS host has the detection server set up per `docs/deployment-guide.md` Part 1.

## Bring-up sequence

### On the macOS host

```bash
cd ~/Projects/home/aoe2-llm-arena/agent
source venv/bin/activate

# Start detection server — needs to be on 0.0.0.0 for the VM to reach it
just server --model detection/inference/models/aoe2_yolo_v9.onnx
# INFO: Uvicorn running on http://0.0.0.0:8420

# In another shell, find the VM-facing IP. DISCOVER it — do not assume a
# value from an earlier setup. The interface name and subnet change with the
# hypervisor, its version, and how many VMs have ever been created.
for i in $(ifconfig -l); do
  case $i in vmnet*|bridge*) echo "$i $(ifconfig $i | awk '/inet /{print $2}')";; esac
done
# bridge100 192.168.99.1
# bridge101 172.16.216.1     <- this one, if the VM is on 172.16.216.x
```

Two ways to tell which bridge the VM is on:

```bash
arp -an | grep -E '192\.168\.99\.|172\.16\.216\.'
# ? (172.16.216.133) at 0:c:29:89:77:70 on bridge101   <- the VM
```

Or read the VM's own IP with `ipconfig` on Windows and match the subnet. The
Mac is `.1` on that subnet. Confirm before moving on:

```bash
curl -s http://<that-ip>:8420/health
# {"status":"ok","backend":"onnx_coreml","num_classes":60}
```

### On the VM (Command Prompt)

```cmd
cd %USERPROFILE%\aoe2-llm-arena\agent
venv\Scripts\activate

set AOE2_LLM_API_KEY=your-key-here    :: must match AOE2_LLM_WIRE (default: openai)
set AOE2_DETECTION_HOST=http://192.168.64.1:8420
:: Optional knobs
set AOE2_STRATEGIST_INTERVAL=10
set AOE2_SAVE_SCREENSHOTS=true

:: Sanity: can the VM reach the Mac?
curl http://192.168.64.1:8420/health
:: {"backend": "onnx_cpu", "classes": 60, "model": "aoe2_yolo_v9.onnx"}
```

### Start the game and the agent

1. Launch AoE2:DE.
2. **Single Player → Skirmish → Standard Game.** Pick civ, set AI opponent, start.
3. Wait for the Town Center to be visible (skip the intro).
4. Switch to Command Prompt:
   ```cmd
   python -m gameplay_agent
   ```

You should see structured logs like:

```
detector_initialized   mode=remote server=http://192.168.64.1:8420
game_loop_start        detection=True executor_model=gpt-5.6-luna strategist_model=gpt-5.6-terra
iteration_start        iteration=1
screenshot_captured    width=1920 height=1080
detection_complete     entity_count=12
strategist_goals_updated  turn=1 goal_count=4
llm_response           iteration=1 action_count=3
actions_executed       iteration=1 total=3 successful=3
```

If you see those five lines, the bring-up worked. If you don't, jump to the symptom matrix below.

## Recording a game

`python -m gameplay_agent` plays but writes **no ledger row**. To record one, run
the experiment wrapper instead — it plays the same game and appends to
`experiments/results.tsv`:

```cmd
just experiment "what changed since the last row"
```

Flags: `--max-iterations N`, `--time-budget SECONDS`, `--overlay`.

Leave `AOE2_SAVE_SCREENSHOTS` on. `save_screenshot` (`screen.py:51`) writes bytes
that `capture_screenshot` already encoded — measured at 3.4 ms per frame against
a `capture_ms` of 2000-6000, so turning it off buys nothing and costs you the
frames `log_to_scenario.py` builds fixtures from.

What does make `capture_ms` seconds long is grabbing and encoding a 3024x1672
region inside a VM. The lever there is the resolution, and a smaller one needs a
matching `calibration.<W>x<H>.yaml` first — see the coordinate row below.

Drop `--overlay` for a timed run: it draws a window and hides it again every
turn. Its cost has not been measured.

## Symptom matrix

These are accumulated failure modes from many bring-up attempts:

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'detection'` after `pip install` | Editable install missed the `packages/detection/src/` directory because `pyproject.toml` excludes it; you ran `pip install` from the wrong dir | `cd agent` and run `pip install -e .` from the project root. |
| Agent starts but `detector_initialized` shows `mode=local`, not `remote` | `AOE2_DETECTION_HOST` not set or unreachable | `printenv AOE2_DETECTION_HOST` on the VM; `curl` the URL; check Mac firewall. |
| `game_not_found` on first iteration | AoE2 window not detected | Click the AoE2 window once. Don't minimize it. Run the agent from Command Prompt, not from inside an IDE that might steal focus. |
| `could_not_focus_game` | Focus race | Add a 2-second `time.sleep` between starting AoE2 and the agent. Easier: focus the AoE2 window manually, then `Win+R`, switch to Command Prompt, hit enter. |
| Coordinates clearly off (clicks land in the wrong place) | Game is fullscreen at unexpected resolution, or DPI scaling is on | Run AoE2 in windowed mode at 1920×1080. Turn off Windows DPI scaling for AoE2. **Check first that `apps/agent/src/resource_ocr_assets/calibration.<W>x<H>.yaml` exists for the target resolution** — only 3024×1672 and 3024×1964 ship one, and without a match every turn pays a full-width OCR scan. |
| Agent picks the wrong screen on multi-monitor VM | `mss` picks monitor 1 by default | Pass `--monitor 0` (primary), or set `AOE2_MONITOR_INDEX` if you've wired it up. |
| Detection works on Mac but VM gets `Connection refused` | Server bound to `127.0.0.1` instead of `0.0.0.0` | Restart the server with `--host 0.0.0.0` (it's the default for `just server`, but easy to override and forget). |
| Detection works once, then connection drops repeatedly | macOS firewall is challenging the server | System Settings → Network → Firewall → allow incoming for the Python binary. |
| `Invalid schema for response_format` on every turn, `llm_error_rate=1.0` | A model field emits an open object (`dict[str, int]`), which OpenAI strict mode rejects | Fixed 2026-08-20. `tests/test_models.py` now fails on any new one — run `just check` before a VM run. |
| `warning: Using incompatible environment (.venv) due to --no-sync` | `--no-sync` reused a venv built for a different Python; the project pins 3.11 | Drop `--no-sync`, or run `uv sync` first. |
| OCR dominates the turn (`ocr_ms` in the tens of seconds) | The resolution has no `calibration.<W>x<H>.yaml`, so auto-detect scans the full-width band every turn | Add a calibration for that resolution, or run at one that has one. |

## Variables you might want to tune

| Env var | Default | When to change |
|---|---|---|
| `AOE2_MODEL` | `gpt-5.6-luna` | Executor: fast, runs every turn. Pin to a dated snapshot for reproducibility (autoresearch runs). |
| `AOE2_EXECUTOR_EFFORT` | `low` | `medium`/`high` for deeper executor reasoning at higher latency. |
| `AOE2_STRATEGIST_MODEL` | `gpt-5.6-terra` | Strategist: strong, runs every 3-10 turns. Same pinning advice. |
| `AOE2_STRATEGIST_INTERVAL` | `10` | Lower (e.g. 5) for tighter goal updates; higher (20+) to save strategist cost. |
| `AOE2_LOOP_DELAY` | `0.3` | Slow CPU? Bump to 1.0. Fast CPU and you want more turns/min? 0.1. |
| `AOE2_SAVE_SCREENSHOTS` | `true` | `false` if disk is filling up. |
| `AOE2_TEMPERATURE` | `0.0` | Raise for output diversity at reproducibility cost. |
| `AOE2_SEED` | unset (OS entropy) | Set an int to make `executor.py`'s build-retry jitter deterministic. Doesn't affect the LLM (the SDK doesn't accept `seed=`). |

## Stopping cleanly

`Ctrl+C` in the Command Prompt running the agent. The shutdown handler closes the wire's HTTP client and flushes any open files. Don't `Ctrl+C` twice — the second one will terminate before the cleanup completes and might leave an orphan connection (harmless, but ugly).

## Related

- [`docs/deployment-guide.md`](../deployment-guide.md) — first-time setup (env, install, model export).
- [Chapter 1 — System Overview](../part1-architecture/01-system-overview.md) — what each env var actually does.
