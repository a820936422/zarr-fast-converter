from __future__ import annotations

from pathlib import Path

from ..rechunking.inspection import format_inspection
from .grid import format_grid_report, inspect_grid
from .models import ResampleInspection


def inspect_resample_input(path: str | Path) -> ResampleInspection:
    info, grid = inspect_grid(path)
    report = format_inspection(info) + "\n\n" + format_grid_report(grid)
    return ResampleInspection(info=info, grid=grid, report=report)
