"""CLI entry point for the arena web server: `python -m arena.web`."""

from __future__ import annotations

import argparse
import logging


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="AoE2 Arena Web (event-log replay)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("arena.web.server:app", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
