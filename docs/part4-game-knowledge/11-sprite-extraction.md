# Chapter 11: Sprite Extraction

Training data generation requires individual sprite images for each entity class. These are extracted from AoE2:DE's proprietary SLD file format -- a GPU-compressed sprite format used by the game engine.

## 11.1 SLD File Format

SLD (Sprite Layer Data) files use the `SLDX` signature and store multi-layer sprite data with GPU texture compression.

### File Structure

```
Header:
  - Signature: "SLDX" (4 bytes)
  - Version: uint16
  - Num frames: uint16

Frame Headers (repeated):
  - Width, height: uint16 each
  - Hotspot X, Y: int16 each (anchor point for game positioning)
  - Layer flags: uint8 bitfield
  - Per-layer: content_length + compressed pixel data
```

### Layer Flags (Bitfield)

| Bit | Layer | Compression | Purpose |
|-----|-------|-------------|---------|
| 0 | Main graphics | DXT1 (BC1) | Unit/building appearance |
| 1 | Shadow | BC4 | Shadow overlay |
| 2 | Unknown | -- | Unused in extraction |
| 3 | Damage | DXT1 | Damaged variant |
| 4 | Player color | DXT1 | Team color overlay |

The extractor reads the main graphics layer (bit 0) and optionally the player color layer (bit 4).

## 11.2 DXT1 (BC1) Decompression

DXT1 is a lossy texture compression format that encodes 4x4 pixel blocks into 8 bytes:

### Block Layout (8 bytes)

```
Bytes 0-1: Color 0 (RGB565)
Bytes 2-3: Color 1 (RGB565)
Bytes 4-7: 2-bit index table (16 pixels, 4x4)
```

### Color Palette Generation

Two reference colors are stored as RGB565 (5 bits red, 6 bits green, 5 bits blue):

```python
# RGB565 to RGB888 conversion
r = ((c >> 11) & 0x1F) * 255 // 31
g = ((c >> 5) & 0x3F) * 255 // 63
b = (c & 0x1F) * 255 // 31
```

Four palette colors are derived:

| Mode | Color 0 | Color 1 | Color 2 | Color 3 |
|------|---------|---------|---------|---------|
| Opaque (c0 > c1) | c0 | c1 | 2/3*c0 + 1/3*c1 | 1/3*c0 + 2/3*c1 |
| Transparent (c0 <= c1) | c0 | c1 | 1/2*c0 + 1/2*c1 | transparent (alpha=0) |

Each pixel in the 4x4 block uses a 2-bit index to select one of these 4 colors.

## 11.3 BC4 Shadow Decompression

Shadows use BC4 compression -- single-channel (alpha) with 8 bytes per 4x4 block:

```
Bytes 0-1: Two reference alpha values
Bytes 2-7: 3-bit index table (16 pixels)
```

Eight alpha levels are interpolated between the two reference values. The extractor uses these as shadow intensity masks.

## 11.4 Command Array (Run-Length Encoding)

SLD sprites are sparse -- most of the bounding box is transparent. A command array encodes skip/draw pairs:

```
For each row:
  - Skip N transparent pixels
  - Draw M opaque pixels (from compressed data)
  - Repeat until row width reached
```

This avoids storing and decompressing transparent regions, significantly reducing file size for small sprites on large canvases.

## 11.5 Player Color Recoloring

AoE2 uses 8 team colors. The base sprites use blue as the default player color, and the game recolors them at runtime.

The extractor performs luminance-preserving hue shift:

1. Identify blue-range pixels in the player color layer (hue 180-260)
2. Compute luminance from original pixel
3. Map to target team color while preserving luminance
4. Blend with main graphics layer

8 team colors: Blue, Red, Green, Yellow, Cyan, Purple, Gray, Orange.

For training data, sprites are extracted in 2-3 random player colors to teach the model that the same unit can appear in different colors.

## 11.6 Batch Extraction

`detection/extraction/extract_sprites.py` defines 46 sprite categories with glob patterns:

```python
("villager", [
    "u_vil_male_villager_idle*_x1.sld",
    "u_vil_female_villager_idle*_x1.sld", ...
], 6, "Worker units"),

("knight_line", [
    "u_cav_knight_idle*_x1.sld",
    "u_cav_cavalier_idle*_x1.sld",
    "u_cav_paladin_idle*_x1.sld",
], 4, "Knight, Cavalier, Paladin"),
```

Each category specifies:
- Class name (matching the detection taxonomy)
- Glob patterns for SLD files in `game_graphics/`
- Number of variants to extract (4-6 per class)
- Description

Animation frames `[0, 4, 8, 12]` are extracted to capture idle and walking poses.

## 11.7 Output

Extracted sprites are saved as RGBA PNG files in `tmp/sprites/{class_name}/`:

```
tmp/sprites/
├── villager/
│   ├── villager_0_blue.png
│   ├── villager_0_red.png
│   ├── villager_1_blue.png
│   └── ...
├── sheep/
│   ├── sheep_0.png
│   └── ...
└── town_center/
    ├── town_center_0.png
    └── ...
```

These PNGs are consumed by `generate_training_data.py` (see [Chapter 8](../part3-entity-detection/08-training-pipeline.md)) to composite synthetic training images.

> **Key Insight**: The SLD format is not publicly documented by Microsoft. The implementation is reverse-engineered from the openage project (open-source AoE engine), with additional AoE2:DE-specific discoveries around optional 2-byte markers before certain layers and content_length semantics (includes its own size). Some building variants with unusual layer configurations cause parsing failures and are skipped.

---

## Summary

- SLD files use DXT1/BC4 GPU texture compression with run-length encoded command arrays
- 4x4 pixel blocks with 4-color palettes (DXT1) or 8-level alpha (BC4)
- Player color recoloring via luminance-preserving hue shift
- 46 sprite categories extracted with multiple animation frames and team colors
- Output: RGBA PNGs consumed by the synthetic data generator

## Related Topics

- [Chapter 8: Training Pipeline](../part3-entity-detection/08-training-pipeline.md) -- how sprites become training data
- [Chapter 7: Detector Architecture](../part3-entity-detection/07-detector-architecture.md) -- the 59-class taxonomy these sprites map to
