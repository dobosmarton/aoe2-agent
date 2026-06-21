# Resource-bar OCR — assets & how to run

This directory holds the **resolution-specific** inputs the local resource-bar
reader (`gameplay_agent/resource_ocr.py`) can use: per-field pixel geometry
(*calibration*) and per-digit *glyph templates*. These are data; the reader is
code. They're kept out of `vision_fixtures/` on purpose (that dir is globbed as
strategist fixtures and shared with the strategist vision test).

The reader is the strategist's HUD source of truth — it replaced Claude vision.
**At runtime it needs no files here:** `autodetect_calibration` localizes the bar
from the live frame, so any capture resolution works out of the box. The assets in
this directory are **optional overrides**: a hand-tuned `calibration.<W>x<H>.yaml`
takes precedence over auto-detection for a resolution you want to pin, and glyph
templates add a lone-single-digit fallback.

## Verify the machinery (no real data)

```bash
.venv/bin/python -m gameplay_agent.resource_ocr --selftest   # inner OCR machinery
.venv/bin/python -m pytest tests/test_resource_ocr.py -q      # full public path
```

The test suite also asserts the **no-YAML auto path** reads the 7 real
`vision_fixtures/` frames (Dark→Imperial) as well as the hand calibration does.

## Pinning a resolution (optional) — the calibration tool

Runtime auto-detection handles any resolution. Pin one only if you want a
hand-verified, eyeball-able box set (e.g. the bundled 3024×1964 / 3024×1672
calibrations). Generate it semi-automatically:

```bash
# a few game screenshots at the target resolution (any HUD-visible frames):
.venv/bin/python scripts/calibrate_resource_bar.py --screenshots 'logs/screenshots/*.jpg'
```

It runs RapidOCR over the top of the frame, classifies the detections (population
anchors the main row; villager sub-counts are excluded), writes
`calibration.<W>x<H>.yaml`, saves `calibration_preview_<W>x<H>.png` (boxes drawn —
**eyeball it**), and prints a **self-check readout** (the readings it gets on your
screenshots — sanity-check them against the actual game). Nudge any off box in the
YAML and re-run. The same `_detect`/`_extract`/`_assign` logic powers runtime
`autodetect_calibration`, so a pinned YAML and the auto path agree by construction.

Optional digit templates (lone-single-digit fallback only — RapidOCR handles
multi-digit fine without them): add `--template-frame IMG --template-values
"wood food gold stone population"` (on-screen order) for one frame; they land in
`templates/<W>x<H>/` as `0.png` … `9.png`, `slash.png`.

## Calibration YAML format

See `calibration.example.yaml`. Keys: `width`, `height`, `template_dir` (relative
to the YAML), and `fields` mapping wood/food/gold/stone/population (optionally
`age`) to `[x0, y0, x1, y1]` boxes — inclusive-exclusive, in full-screenshot
pixels. `calibration_for(w, h)` loads the file matching a screenshot's resolution.

## Adding regression fixtures → `vision_fixtures/`

Drop real screenshots spanning the four ages and varied values (include 0s and
3–4 digit numbers) into `apps/agent/src/vision_fixtures/` with a sibling
`<name>.yaml`:

```yaml
name: dark_age_turn8
screenshot: dark_age_turn8.jpg          # relative to this fixture file
expected:
  wood: {min: 145, max: 155}            # range = tolerance check
  food: 245                             # bare int = exact match
  gold: 0
  stone: 200
  population: "8/15"                    # string = exact
  age: "Dark Age"                       # string = exact
```

Label `expected` from what you SEE in the image — that's ground truth. These
become permanent strategist-vision + auto-calibration regression fixtures.
