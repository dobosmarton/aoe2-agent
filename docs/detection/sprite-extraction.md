# Sprite extraction

The synthetic training data is built from the game's own sprites. `packages/detection/src/extraction/` decodes Age of Empires II: DE `.sld` files into transparent PNGs, one set per detector class, and the generators in `packages/detection/src/training/` paste those PNGs onto backgrounds. `packages/detection/src/docs/SLD_FORMAT.md` has the format notes the decoder is built on.

| Module | Does |
|---|---|
| `sld_extractor.py` | Parses one `.sld` file and decodes its main graphics layer (`SLDExtractor`, `extract_sprite`, `extract_multiple_frames`) |
| `extract_sprites.py` | Maps game file names to detector classes and extracts a sprite set (`SPRITE_CONFIG`, `extract_sprites`) |
| `capture_replay.py` | Saves screenshots of a running game every few seconds with `mss`, for backgrounds or labelling |

## SLD layout as the decoder reads it

All fields are little-endian.

| Part | Size | Fields |
|---|---|---|
| File header | 16 bytes | `SLDX` signature, version, frame count, three unknown fields |
| Frame header | 12 bytes | canvas width and height, hotspot x and y, frame-type flags, unknown byte, frame index |
| Main layer | variable | content length (u32), bounds x1, y1, x2, y2 (u16), 2 flag bytes, command count (u16), command pairs, DXT1 blocks |

The frame-type byte says which layers follow, in this order: `0x01` main graphics, `0x02` shadow, `0x04` unknown, `0x08` damage mask, `0x10` player colour mask.

The main layer covers only its bounds, not the canvas, so the PNG is `x2 - x1` by `y2 - y1` pixels. Those pixels form a grid of 4x4 blocks in row-major order. Each command pair is `(skip, draw)`: advance `skip` blocks, leaving them transparent, then decode `draw` blocks of 8 bytes each. After the layer, the parser moves to the next 4-byte boundary using the size it computed itself.

The decoder skips every other layer. It aligns to 4 bytes, reads an optional `0x0000` marker before the unknown, damage and player layers (taken as a marker only when the following u32 is between 0 and 1,000,000), then jumps by `content_length - 4`, since the length field counts itself.

## DXT1 decompression

`decode_dxt1_block` turns 8 bytes into 16 RGBA pixels. Bytes 0 to 3 hold two RGB565 colours, widened to 8 bits per channel by shifting and repeating the top bits. If `color0 > color1` the palette has four opaque colours, the two stored and two interpolated at one and two thirds. Otherwise it has three colours (the stored two and their midpoint) plus transparent. Bytes 4 to 7 hold sixteen 2-bit palette indices.

`decode_bc4_block` (single-channel BC4) exists, but no layer is decoded with it.

## Player colour

The player-colour mask layer is skipped, so extracted units keep the blue base colour of their sprites. With `--player-colors`, `apply_player_color` recolours any pixel where B > 100, B > 1.5 R and B > 1.2 G to one of the eight `PLAYER_COLORS` (blue, red, green, yellow, cyan, purple, gray, orange), scaled by the pixel's luminance. Each saved PNG gets one randomly chosen colour. Only classes in `PLAYER_COLOR_CLASSES` are recoloured: units, siege, ships and owned livestock (sheep, goat, cow, buffalo, horse).

## Extracting a sprite set

`SPRITE_CONFIG` has 66 entries of `(class name, glob patterns, maximum variants, description)`. They cover every class in the [taxonomy](detector.md#class-taxonomy) except `farm`, plus the fine labels `ibex` and `elephant`. Patterns match game file names such as `u_vil_male_villager_idle*_x1.sld`. Building patterns also drop files whose names contain `destruction`, `rubble`, `shadow`, `constr`, `foundation`, `attackup`, `defenseup`, `bothup` or `_snow_`. When more files match than the variant limit, the list is shuffled with seed 42 before it is cut.

```bash
uv run --package detection python -m detection.extraction.extract_sprites \
  --game-dir game_graphics --output tmp/sprites --multi-frame --player-colors --directions 16
```

Relative `--game-dir` and `--output` paths resolve against `packages/detection/src`. `--classes` limits the run to named classes and `--show-config` prints the table.

| Mode | Frames taken | File name |
|---|---|---|
| default | frame 0 | `<class>_<NN>.png` |
| `--multi-frame` | `--frames`, default 0, 4, 8, 12 | `<class>_<NN>_f<frame>.png` |
| `--multi-frame --directions N` | two frames from each of N facing blocks | `<class>_<NN>_f<frame>.png` |

Unit and animal files store each facing direction as a contiguous block of frames (the game has 16), so fixed indices like 0, 4, 8, 12 all come from one direction. `direction_frame_indices` takes the first and middle frame of each block instead, and falls back to `--frames` for files that are too short.

`SLDExtractor.extract_all` stops at the first frame it cannot parse and keeps the frames before it. `extract_sprites` reports found, extracted and failed counts per class. All output lands in one flat directory.

`sld_extractor.py` also runs on its own: `python -m detection.extraction.sld_extractor input.sld [output.png]` writes frame 0, and `--batch <input_dir> <output_dir> [patterns...]` writes frame 0 of every matching file.

## How the generators use the sprites

| Generator | Sprite directory | Which files it loads |
|---|---|---|
| `generate_training_data.py` (synthetic scenes) | `--sprites`, default `tmp/sprites_v6` at the repository root | Per class, the glob `<class>_*.png` from its `SpriteConfig`; it configures IDs 0 to 59 except `farm` (16) |
| `generate_composites.py` (sprites on real frames) | `--sprites`, required | Names matching `<class>_<NN>_f<frame>.png` only, for every schema class and alias label; each is cropped to its alpha and kept if both sides are at least 4 px |

The composite generator loads `ibex` and `elephant` sprites under their own names and writes the class they map to (`deer`, `boar`) into the label file. The synthetic generator reuses `PLAYER_COLORS` for the selection outlines and health bars it draws over some placed sprites. How these datasets feed a training run is in [training](training.md).
