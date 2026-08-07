#!/usr/bin/env python3
"""Run the xESMF based Zarr resampler from a source checkout."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fast_nc_zarr.resampling.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
