#!/usr/bin/env python3
"""Launch the PySide6 GUI for the fast Zarr converter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from fast_nc_zarr.gui.app import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
