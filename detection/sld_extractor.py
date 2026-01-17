#!/usr/bin/env python3
"""
SLD file extractor for AoE2:DE sprite files.

SLD format specification from openage project:
https://github.com/SFTtech/openage/blob/master/doc/media/sld-files.md

SLD uses DXT1 (BC1) compression for main graphics and DXT4 (BC4) for shadows.
"""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Tuple
import io


@dataclass
class SLDHeader:
    """SLD file header."""
    signature: bytes      # "SLDX"
    version: int         # Usually 4
    num_frames: int      # Number of animation frames
    unknown1: int
    unknown2: int
    unknown3: int


@dataclass
class SLDFrameHeader:
    """SLD frame header."""
    canvas_width: int
    canvas_height: int
    hotspot_x: int
    hotspot_y: int
    frame_type: int      # Bit field for layer presence
    unknown1: int
    frame_index: int


@dataclass
class ExtractedFrame:
    """Extracted frame data."""
    index: int
    width: int
    height: int
    hotspot: Tuple[int, int]
    pixels: bytes  # RGBA pixel data


def decode_dxt1_block(block_data: bytes) -> List[Tuple[int, int, int, int]]:
    """
    Decode a single 4x4 DXT1/BC1 compressed block.

    DXT1 format:
    - 2 bytes: color0 (RGB565)
    - 2 bytes: color1 (RGB565)
    - 4 bytes: 16 2-bit indices (4x4 pixels)

    Returns list of 16 RGBA tuples.
    """
    if len(block_data) < 8:
        return [(0, 0, 0, 0)] * 16

    # Read two 16-bit colors in RGB565 format
    color0_565 = struct.unpack('<H', block_data[0:2])[0]
    color1_565 = struct.unpack('<H', block_data[2:4])[0]

    # Convert RGB565 to RGB888
    def rgb565_to_rgb888(c):
        r = ((c >> 11) & 0x1F) << 3
        g = ((c >> 5) & 0x3F) << 2
        b = (c & 0x1F) << 3
        # Extend to full 8-bit range
        r = r | (r >> 5)
        g = g | (g >> 6)
        b = b | (b >> 5)
        return (r, g, b)

    c0 = rgb565_to_rgb888(color0_565)
    c1 = rgb565_to_rgb888(color1_565)

    # Build color lookup table
    colors = [None] * 4
    colors[0] = (*c0, 255)  # RGBA
    colors[1] = (*c1, 255)

    if color0_565 > color1_565:
        # 4-color mode: interpolate 2 more colors
        colors[2] = (
            (2 * c0[0] + c1[0]) // 3,
            (2 * c0[1] + c1[1]) // 3,
            (2 * c0[2] + c1[2]) // 3,
            255
        )
        colors[3] = (
            (c0[0] + 2 * c1[0]) // 3,
            (c0[1] + 2 * c1[1]) // 3,
            (c0[2] + 2 * c1[2]) // 3,
            255
        )
    else:
        # 3-color + transparent mode
        colors[2] = (
            (c0[0] + c1[0]) // 2,
            (c0[1] + c1[1]) // 2,
            (c0[2] + c1[2]) // 2,
            255
        )
        colors[3] = (0, 0, 0, 0)  # Transparent

    # Read 4 bytes of indices (16 2-bit values)
    indices = struct.unpack('<I', block_data[4:8])[0]

    # Decode 16 pixels
    pixels = []
    for i in range(16):
        idx = (indices >> (i * 2)) & 0x3
        pixels.append(colors[idx])

    return pixels


def decode_bc4_block(block_data: bytes) -> List[int]:
    """
    Decode a single 4x4 BC4 compressed block (single channel).

    BC4 format:
    - 1 byte: alpha0
    - 1 byte: alpha1
    - 6 bytes: 16 3-bit indices

    Returns list of 16 alpha values.
    """
    if len(block_data) < 8:
        return [0] * 16

    alpha0 = block_data[0]
    alpha1 = block_data[1]

    # Build alpha lookup table
    alphas = [0] * 8
    alphas[0] = alpha0
    alphas[1] = alpha1

    if alpha0 > alpha1:
        # 8-alpha mode
        for i in range(2, 8):
            alphas[i] = ((8 - i) * alpha0 + (i - 1) * alpha1) // 7
    else:
        # 6-alpha + 0 + 255 mode
        for i in range(2, 6):
            alphas[i] = ((6 - i) * alpha0 + (i - 1) * alpha1) // 5
        alphas[6] = 0
        alphas[7] = 255

    # Read 6 bytes of indices (16 3-bit values packed)
    index_bits = int.from_bytes(block_data[2:8], 'little')

    pixels = []
    for i in range(16):
        idx = (index_bits >> (i * 3)) & 0x7
        pixels.append(alphas[idx])

    return pixels


class SLDExtractor:
    """Extract sprites from AoE2:DE SLD files."""

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.data = self.file_path.read_bytes()
        self.offset = 0
        self.header: Optional[SLDHeader] = None
        self.frames: List[ExtractedFrame] = []

    def _read(self, fmt: str) -> tuple:
        """Read and unpack data at current offset."""
        size = struct.calcsize(fmt)
        result = struct.unpack(fmt, self.data[self.offset:self.offset + size])
        self.offset += size
        return result

    def _read_bytes(self, n: int) -> bytes:
        """Read n bytes at current offset."""
        result = self.data[self.offset:self.offset + n]
        self.offset += n
        return result

    def parse_header(self) -> SLDHeader:
        """Parse SLD file header."""
        sig, ver, frames, u1, u2, u3 = self._read('< 4s 4H I')

        if sig != b'SLDX':
            raise ValueError(f"Invalid SLD signature: {sig}")

        self.header = SLDHeader(
            signature=sig,
            version=ver,
            num_frames=frames,
            unknown1=u1,
            unknown2=u2,
            unknown3=u3
        )
        return self.header

    def parse_frame(self, frame_idx: int) -> Optional[ExtractedFrame]:
        """Parse a single frame."""
        # Read frame header (12 bytes)
        w, h, hx, hy, ftype, u1, fidx = self._read('< 4H 2B H')

        frame_header = SLDFrameHeader(
            canvas_width=w,
            canvas_height=h,
            hotspot_x=hx,
            hotspot_y=hy,
            frame_type=ftype,
            unknown1=u1,
            frame_index=fidx
        )

        # Check which layers are present (bits from LSB)
        has_main = (ftype & 0x01) != 0      # Bit 0: Main graphics
        has_shadow = (ftype & 0x02) != 0    # Bit 1: Shadow
        has_unknown = (ftype & 0x04) != 0   # Bit 2: Unknown
        has_damage = (ftype & 0x08) != 0    # Bit 3: Damage mask
        has_player = (ftype & 0x10) != 0    # Bit 4: Player color

        pixels = None
        layer_width = w
        layer_height = h

        # Process main graphics layer (DXT1)
        if has_main:
            pixels, layer_width, layer_height = self._parse_dxt1_layer(w, h)

        # Skip other layers - use content_length for all since format varies
        # Note: unknown and player layers may have a 2-byte 0x0000 marker before content_length
        if has_shadow:
            self._skip_layer(use_content_length=True)
        if has_unknown:
            self._skip_layer(use_content_length=True, check_marker=True)
        if has_damage:
            self._skip_layer(use_content_length=True, check_marker=True)
        if has_player:
            self._skip_layer(use_content_length=True, check_marker=True)

        if pixels is None:
            return None

        return ExtractedFrame(
            index=frame_idx,
            width=layer_width,
            height=layer_height,
            hotspot=(hx, hy),
            pixels=pixels
        )

    def _parse_dxt1_layer(self, width: int, height: int) -> Tuple[bytes, int, int]:
        """Parse DXT1 compressed main graphics layer.

        Returns:
            Tuple of (pixel_data, layer_width, layer_height)
        """
        # Read content length (4 bytes) - note: we calculate actual end position ourselves
        content_length = self._read('< I')[0]
        layer_start = self.offset

        # Layer header: 10 bytes
        # x1, y1, x2, y2 (4 * uint16) + flag1, unknown (2 * uint8)
        x1, y1, x2, y2 = self._read('< 4H')
        flag1, unk_byte = self._read('< 2B')

        layer_width = x2 - x1
        layer_height = y2 - y1

        # Command array length (2 bytes) - this is the NUMBER of command pairs, not bytes
        num_commands = self._read('< H')[0]

        # Read commands (2 bytes each: skip_count, draw_count)
        commands = []
        total_draw_blocks = 0
        for _ in range(num_commands):
            skip_count, draw_count = self._read('< 2B')
            commands.append((skip_count, draw_count))
            total_draw_blocks += draw_count

        # Calculate number of 4x4 blocks
        blocks_wide = (layer_width + 3) // 4
        blocks_high = (layer_height + 3) // 4
        total_blocks = blocks_wide * blocks_high

        # Create output image buffer (RGBA)
        img_data = bytearray(layer_width * layer_height * 4)

        # Process blocks
        block_idx = 0
        for skip_count, draw_count in commands:
            # Skip blocks (leave as transparent)
            block_idx += skip_count

            # Draw blocks
            for _ in range(draw_count):
                if block_idx >= total_blocks:
                    break

                bx = block_idx % blocks_wide
                by = block_idx // blocks_wide

                # Read and decode DXT1 block (8 bytes)
                block_data = self._read_bytes(8)
                pixels = decode_dxt1_block(block_data)

                # Write pixels to image
                for py in range(4):
                    for px in range(4):
                        ix = bx * 4 + px
                        iy = by * 4 + py
                        if ix < layer_width and iy < layer_height:
                            pixel_idx = py * 4 + px
                            r, g, b, a = pixels[pixel_idx]
                            offset = (iy * layer_width + ix) * 4
                            img_data[offset:offset + 4] = bytes([r, g, b, a])

                block_idx += 1

        # Calculate actual layer size and align to 4-byte boundary
        actual_size = 10 + 2 + (num_commands * 2) + (total_draw_blocks * 8)
        aligned_size = (actual_size + 3) & ~3  # Align to 4 bytes
        self.offset = layer_start + aligned_size

        return bytes(img_data), layer_width, layer_height

    def _skip_layer(self, is_mask_layer: bool = False, use_content_length: bool = False,
                    check_marker: bool = False):
        """Skip a layer we don't need.

        Args:
            is_mask_layer: If True, uses 2-byte mask header instead of 10-byte graphics header
            use_content_length: If True, trust content_length field instead of calculating
            check_marker: If True, check for optional 2-byte 0x0000 marker before content_length
        """
        # Ensure we're 4-byte aligned before reading layer
        if self.offset % 4 != 0:
            self.offset = (self.offset + 3) & ~3

        if check_marker:
            # Check if there's a 0x0000 marker before content_length
            # Only treat as marker if next 4 bytes look like a reasonable content_length
            marker = struct.unpack('< H', self.data[self.offset:self.offset+2])[0]
            if marker == 0:
                potential_len = struct.unpack('< I', self.data[self.offset+2:self.offset+6])[0]
                # Only skip marker if the resulting content_length is reasonable (<1MB)
                if 0 < potential_len < 1000000:
                    self._read('< H')  # Skip the marker

        content_length = self._read('< I')[0]
        layer_start = self.offset

        if use_content_length:
            # Some layers have different formats, just skip by content_length
            # Note: content_length includes the 4-byte length field itself, so subtract 4
            self.offset = layer_start + content_length - 4
            # Ensure 4-byte alignment after skip
            if self.offset % 4 != 0:
                self.offset = (self.offset + 3) & ~3
            return

        if is_mask_layer:
            # Mask Layer Header: 2 bytes (flag1, unknown)
            flag1, unk_byte = self._read('< 2B')
            header_size = 2
        else:
            # Graphics Layer Header: 10 bytes
            x1, y1, x2, y2 = self._read('< 4H')
            flag1, unk_byte = self._read('< 2B')
            header_size = 10

        num_commands = self._read('< H')[0]

        # Count draw blocks
        total_draw_blocks = 0
        for _ in range(num_commands):
            skip_count, draw_count = self._read('< 2B')
            total_draw_blocks += draw_count

        # Skip pixel data
        self.offset += total_draw_blocks * 8

        # Align to 4-byte boundary
        actual_size = header_size + 2 + (num_commands * 2) + (total_draw_blocks * 8)
        aligned_size = (actual_size + 3) & ~3  # Align to 4 bytes
        self.offset = layer_start + aligned_size

    def extract_all(self) -> List[ExtractedFrame]:
        """Extract all frames from the SLD file."""
        self.offset = 0
        self.parse_header()

        self.frames = []
        for i in range(self.header.num_frames):
            try:
                frame = self.parse_frame(i)
                if frame:
                    self.frames.append(frame)
            except Exception as e:
                print(f"Warning: Failed to parse frame {i}: {e}")
                break

        return self.frames

    def extract_first_frame(self) -> Optional[ExtractedFrame]:
        """Extract just the first frame (useful for idle sprites)."""
        self.offset = 0
        self.parse_header()

        if self.header.num_frames > 0:
            return self.parse_frame(0)
        return None

    def save_as_png(self, frame: ExtractedFrame, output_path: str):
        """Save extracted frame as PNG."""
        try:
            from PIL import Image

            img = Image.frombytes('RGBA', (frame.width, frame.height), frame.pixels)
            img.save(output_path)
            return True
        except ImportError:
            print("PIL not available. Install with: pip install Pillow")
            return False


def extract_sprite(sld_path: str, output_path: str, frame_index: int = 0) -> bool:
    """
    Extract a sprite from an SLD file.

    Args:
        sld_path: Path to .sld file
        output_path: Output .png path
        frame_index: Which animation frame to extract (0 = first/idle)

    Returns:
        True if successful
    """
    try:
        extractor = SLDExtractor(sld_path)
        frames = extractor.extract_all()

        if frame_index < len(frames):
            return extractor.save_as_png(frames[frame_index], output_path)
        else:
            print(f"Frame {frame_index} not found (file has {len(frames)} frames)")
            return False
    except Exception as e:
        print(f"Error extracting {sld_path}: {e}")
        return False


def batch_extract(input_dir: str, output_dir: str, patterns: List[str] = None):
    """
    Batch extract sprites matching patterns.

    Args:
        input_dir: Directory containing .sld files
        output_dir: Output directory for .png files
        patterns: List of filename patterns to match (e.g., ["sheep", "villager"])
    """
    from pathlib import Path

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sld_files = list(input_path.glob("*.sld"))

    if patterns:
        sld_files = [f for f in sld_files if any(p in f.name.lower() for p in patterns)]

    extracted = 0
    for sld_file in sld_files:
        output_file = output_path / f"{sld_file.stem}.png"
        if extract_sprite(str(sld_file), str(output_file)):
            extracted += 1
            print(f"Extracted: {sld_file.name} -> {output_file.name}")

    print(f"\nExtracted {extracted}/{len(sld_files)} sprites")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python sld_extractor.py <input.sld> [output.png]")
        print("       python sld_extractor.py --batch <input_dir> <output_dir> [patterns...]")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        if len(sys.argv) < 4:
            print("Usage: python sld_extractor.py --batch <input_dir> <output_dir> [patterns...]")
            sys.exit(1)
        patterns = sys.argv[4:] if len(sys.argv) > 4 else None
        batch_extract(sys.argv[2], sys.argv[3], patterns)
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else input_file.replace('.sld', '.png')
        extract_sprite(input_file, output_file)
