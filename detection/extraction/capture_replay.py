#!/usr/bin/env python3
"""
Screenshot Capture Utility for AoE2 Training Data

Captures screenshots from the game at regular intervals for use as
training data backgrounds or for annotation.

Usage:
    python capture_from_replay.py                    # Capture 200 screenshots
    python capture_from_replay.py --count 300        # Capture 300 screenshots
    python capture_from_replay.py --interval 3       # Every 3 seconds
    python capture_from_replay.py --output my_shots  # Custom output directory

Instructions:
1. Start AoE2 DE and load a replay or start a game
2. Run this script
3. The script will capture screenshots at regular intervals
4. Use captured screenshots as backgrounds for training data generation
   or annotate them with CVAT/LabelImg for real training data

AoE2 DE Replays Location:
    Windows: C:\\Users\\<user>\\Games\\Age of Empires 2 DE\\<steam_id>\\savegame

Tips for good training data:
- Capture different game stages (Dark Age through Imperial)
- Include different situations (base building, combat, gathering)
- Capture with fog of war visible in some shots
- Use different map types and civilizations
- Avoid UI-heavy frames (menus, tech trees open)
"""

import argparse
import time
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Capture screenshots from AoE2 for training data"
    )
    parser.add_argument(
        "--output", "-o",
        default="detection/real_screenshots/raw",
        help="Output directory for screenshots"
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=200,
        help="Number of screenshots to capture (default: 200)"
    )
    parser.add_argument(
        "--interval", "-i",
        type=float,
        default=5.0,
        help="Seconds between captures (default: 5.0)"
    )
    parser.add_argument(
        "--prefix", "-p",
        default="replay",
        help="Filename prefix (default: replay)"
    )
    parser.add_argument(
        "--monitor",
        type=int,
        default=1,
        help="Monitor number to capture (default: 1 = primary)"
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="JPEG quality 1-100 (default: 90)"
    )
    parser.add_argument(
        "--format",
        choices=["png", "jpg"],
        default="png",
        help="Output format (default: png)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=5.0,
        help="Initial delay before starting capture (default: 5.0)"
    )

    args = parser.parse_args()

    # Try to import mss for screen capture
    try:
        import mss
        import mss.tools
    except ImportError:
        print("Error: mss not installed. Install with: pip install mss")
        return 1

    # Resolve output path
    script_dir = Path(__file__).parent.parent  # agent/
    output_dir = script_dir / args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("AoE2 Screenshot Capture Utility")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Screenshots: {args.count}")
    print(f"Interval: {args.interval}s")
    print(f"Format: {args.format}")
    print("=" * 60)

    print(f"\nStarting capture in {args.delay} seconds...")
    print("Switch to AoE2 window now!")
    print("Press Ctrl+C to stop early.\n")

    time.sleep(args.delay)

    # Initialize screen capture
    with mss.mss() as sct:
        # Select monitor
        if args.monitor > len(sct.monitors) - 1:
            print(f"Warning: Monitor {args.monitor} not found, using primary")
            monitor = sct.monitors[1]
        else:
            monitor = sct.monitors[args.monitor]

        print(f"Capturing from monitor: {monitor['width']}x{monitor['height']}")
        print()

        captured = 0
        try:
            for i in range(args.count):
                # Capture screenshot
                screenshot = sct.grab(monitor)

                # Generate filename
                if args.format == "png":
                    filename = f"{args.prefix}_{i:04d}.png"
                    filepath = output_dir / filename
                    mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(filepath))
                else:
                    filename = f"{args.prefix}_{i:04d}.jpg"
                    filepath = output_dir / filename
                    # Convert to PIL and save as JPEG
                    try:
                        from PIL import Image
                        img = Image.frombytes('RGB', screenshot.size, screenshot.rgb)
                        img.save(str(filepath), 'JPEG', quality=args.quality)
                    except ImportError:
                        # Fallback to PNG if PIL not available
                        png_path = filepath.with_suffix('.png')
                        mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(png_path))
                        print(f"Warning: PIL not available, saved as PNG instead")

                captured += 1
                elapsed = i * args.interval
                remaining = (args.count - i - 1) * args.interval

                print(f"[{i+1:4d}/{args.count}] Captured: {filename} "
                      f"(elapsed: {elapsed:.0f}s, remaining: ~{remaining:.0f}s)")

                if i < args.count - 1:
                    time.sleep(args.interval)

        except KeyboardInterrupt:
            print(f"\n\nCapture interrupted by user.")

    print("\n" + "=" * 60)
    print("CAPTURE COMPLETE")
    print("=" * 60)
    print(f"Screenshots captured: {captured}")
    print(f"Output directory: {output_dir}")
    print()
    print("Next steps:")
    print("1. Review screenshots and remove bad ones (menus, loading screens)")
    print("2. Annotate with CVAT (https://cvat.ai/) or LabelImg")
    print("3. Or use as backgrounds: python generate_training_data.py -r " + str(output_dir))

    return 0


if __name__ == "__main__":
    exit(main())
