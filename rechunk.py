#!/usr/bin/env python3
"""Run the Zarr v3 rechunking and lossless recompression module."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fast_nc_zarr.rechunking.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
