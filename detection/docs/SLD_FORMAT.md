# SLD Sprite Extractor for Age of Empires II: Definitive Edition

A Python implementation for extracting sprite graphics from AoE2:DE's SLD (SLDX) file format.

## Overview

Age of Empires II: Definitive Edition uses the SLD format for storing unit and building sprites. This extractor parses these files and outputs PNG images with transparency, suitable for YOLO training data generation or other purposes.

## File Format Specification

The SLD format specification is based on research from the [openage project](https://github.com/SFTtech/openage/blob/master/doc/media/sld-files.md), with additional reverse-engineering for AoE2:DE-specific variations.

### File Structure Overview

```
┌─────────────────────────────────────────┐
│           SLD File Header (16 bytes)    │
├─────────────────────────────────────────┤
│         Frame 0 Header (12 bytes)       │
│         Frame 0 Layers (variable)       │
├─────────────────────────────────────────┤
│         Frame 1 Header (12 bytes)       │
│         Frame 1 Layers (variable)       │
├─────────────────────────────────────────┤
│                  ...                    │
└─────────────────────────────────────────┘
```

### SLD File Header (16 bytes)

| Offset | Size | Type   | Description |
|--------|------|--------|-------------|
| 0      | 4    | char[] | Signature: "SLDX" |
| 4      | 2    | uint16 | Version (typically 4) |
| 6      | 2    | uint16 | Number of frames |
| 8      | 2    | uint16 | Unknown (always 0x0000) |
| 10     | 2    | uint16 | Unknown (always 0x0010) |
| 12     | 4    | uint32 | Unknown (always 0x000000FF) |

### Frame Header (12 bytes)

| Offset | Size | Type   | Description |
|--------|------|--------|-------------|
| 0      | 2    | uint16 | Canvas width |
| 2      | 2    | uint16 | Canvas height |
| 4      | 2    | int16  | Hotspot X |
| 6      | 2    | int16  | Hotspot Y |
| 8      | 1    | uint8  | Frame type (layer flags) |
| 9      | 1    | uint8  | Unknown |
| 10     | 2    | uint16 | Frame index |

### Frame Type Flags (Bit Field)

The frame type byte indicates which layers are present:

| Bit | Mask | Layer |
|-----|------|-------|
| 0   | 0x01 | Main graphics (DXT1 compressed) |
| 1   | 0x02 | Shadow layer |
| 2   | 0x04 | Unknown layer |
| 3   | 0x08 | Damage mask |
| 4   | 0x10 | Player color mask |

Example: `0x17` = `0b00010111` = Main + Shadow + Unknown + Player

### Layer Structure

Each layer follows this general structure:

```
┌─────────────────────────────────────────┐
│  [Optional] 2-byte marker (0x0000)      │  ← Only for unknown/damage/player layers
├─────────────────────────────────────────┤
│  Content Length (4 bytes, uint32)       │  ← Includes itself in the count
├─────────────────────────────────────────┤
│  Layer Header (10 bytes for graphics)   │
│  - x1, y1, x2, y2 (4 × uint16)          │
│  - flag1, unknown (2 × uint8)           │
├─────────────────────────────────────────┤
│  Number of Commands (2 bytes, uint16)   │  ← Count of command pairs
├─────────────────────────────────────────┤
│  Command Array (N × 2 bytes)            │
│  - skip_count (uint8)                   │
│  - draw_count (uint8)                   │
├─────────────────────────────────────────┤
│  Compressed Pixel Data (DXT1 blocks)    │
└─────────────────────────────────────────┘
```

**Important Notes:**
- `content_length` includes the 4-byte length field itself
- Layer bounds define the actual sprite dimensions: `width = x2 - x1`, `height = y2 - y1`
- Some layers (unknown, damage, player) have an optional 2-byte `0x0000` marker before content_length

## DXT1 (BC1) Compression

The main graphics layer uses DXT1 (also known as BC1) block compression, a format commonly used in video games for texture compression.

### DXT1 Block Structure (8 bytes per 4×4 pixel block)

```
┌─────────────────────────────────────────┐
│  Color 0 (2 bytes, RGB565)              │
│  Color 1 (2 bytes, RGB565)              │
│  Indices (4 bytes, 16 × 2-bit values)   │
└─────────────────────────────────────────┘
```

### RGB565 Format

Each color is stored as a 16-bit value:
- Red: bits 15-11 (5 bits)
- Green: bits 10-5 (6 bits)
- Blue: bits 4-0 (5 bits)

### Color Interpolation

The decoder generates a 4-color palette from the two stored colors:

**If color0 > color1 (4-color opaque mode):**
```
palette[0] = color0
palette[1] = color1
palette[2] = (2 × color0 + color1) / 3
palette[3] = (color0 + 2 × color1) / 3
```

**If color0 ≤ color1 (3-color + transparent mode):**
```
palette[0] = color0
palette[1] = color1
palette[2] = (color0 + color1) / 2
palette[3] = transparent (0, 0, 0, 0)
```

### Index Lookup

The 4-byte index field contains 16 2-bit indices (one per pixel in the 4×4 block), packed in little-endian order. Each index selects a color from the 4-color palette.

## Command Array (Run-Length Encoding)

The command array implements a form of run-length encoding for sparse sprites:

```python
for skip_count, draw_count in commands:
    block_index += skip_count  # Skip transparent blocks
    for _ in range(draw_count):
        decode_and_write_dxt1_block(block_index)
        block_index += 1
```

This significantly reduces file size for sprites with large transparent areas.

## Implementation Details

### Layer Parsing Strategy

The extractor uses different strategies for different layers:

1. **Main Layer**: Full DXT1 decompression with block-by-block decoding
2. **Shadow/Unknown/Damage/Player Layers**: Skipped using content_length

### Handling Format Variations

AoE2:DE files have some variations not documented in the openage spec:

1. **Optional 2-byte Marker**: Some layers have a `0x0000` marker before content_length. The extractor auto-detects this by checking if the first 2 bytes are zero.

2. **Content Length Semantics**: The `content_length` field includes itself (4 bytes) in the count, so actual layer data is `content_length - 4` bytes.

3. **Alignment**: Layers are aligned to 4-byte boundaries, but this is implicitly handled by `content_length`.

### Coordinate System

- **Canvas**: The full frame dimensions (e.g., 200×200)
- **Layer Bounds**: The actual sprite area within the canvas (e.g., 84,76 to 124,104)
- **Hotspot**: The anchor point for positioning the sprite in-game

The extractor outputs only the layer bounds (the actual visible sprite), not the full canvas.

## Usage

### Command Line

```bash
# From agent/ directory:

# Extract single sprite
python -m detection.extraction.sld_extractor input.sld output.png

# Extract specific frame
python -m detection.extraction.sld_extractor input.sld output.png --frame 5

# Batch extract
python -m detection.extraction.sld_extractor --batch input_dir/ output_dir/ [patterns...]
```

### Python API

```python
from detection.extraction.sld_extractor import SLDExtractor, extract_sprite

# Simple extraction
extract_sprite("sheep.sld", "sheep.png")

# Advanced usage
extractor = SLDExtractor("villager.sld")
frames = extractor.extract_all()

for frame in frames:
    print(f"Frame {frame.index}: {frame.width}x{frame.height}")
    extractor.save_as_png(frame, f"frame_{frame.index}.png")
```

### Data Classes

```python
@dataclass
class ExtractedFrame:
    index: int                    # Frame number
    width: int                    # Layer width (not canvas)
    height: int                   # Layer height (not canvas)
    hotspot: Tuple[int, int]      # (x, y) anchor point
    pixels: bytes                 # RGBA pixel data
```

## Known Limitations

1. **Partial Layer Support**: Only the main graphics layer is fully decoded. Shadow, damage, and player color layers are skipped.

2. **Some Building Variants Fail**: Certain civilization-specific buildings have unusual layer formats that cause parsing errors.

3. **Animation Frames**: Later animation frames sometimes fail to parse due to cumulative offset errors. Frame 0 (idle pose) is most reliable.

4. **No Player Color**: Player color masks are not applied, so units appear in their base colors.

## File Locations

AoE2:DE sprite files are located at:
```
Steam/steamapps/common/AoE2DE/resources/_common/drs/graphics/
```

Common naming patterns:
- Units: `u_<category>_<unit>_<animation>_x1.sld`
- Buildings: `b_<civ>_<building>_age<N>_x1.sld`
- Animals: `a_<type>_<animation>_x1.sld`

## Dependencies

- Python 3.8+
- Pillow (PIL) for PNG output

```bash
pip install Pillow
```

## References

- [openage SLD Format Documentation](https://github.com/SFTtech/openage/blob/master/doc/media/sld-files.md)
- [DXT1/BC1 Compression](https://docs.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression)
- [AoE2 Modding Community](https://github.com/Tapsa/AGE) - Advanced Genie Editor

## License

This extractor is part of the AoE2 LLM Arena project and is intended for research and educational purposes.
