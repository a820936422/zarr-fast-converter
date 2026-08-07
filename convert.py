#!/usr/bin/env python3
"""Run the source-data converter from a checkout.

This is the module-one entry point. The Zarr rechunking module will use a
separate ``rechunk.py`` entry point.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fast_nc_zarr.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
