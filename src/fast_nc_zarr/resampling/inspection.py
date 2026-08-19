from __future__ import annotations

from pathlib import Path

import xarray as xr

from ..rechunking.inspection import dataset_info_from_dataset, format_inspection
from .grid import format_grid_report, inspect_dataset_grid, inspect_grid
from .models import ResampleInspection


def inspect_resample_input(path: str | Path) -> ResampleInspection:
    info, grid = inspect_grid(path)
    report = format_inspection(info) + "\n\n" + format_grid_report(grid)
    return ResampleInspection(info=info, grid=grid, report=report)


def inspect_resample_dataset(dataset: xr.Dataset) -> ResampleInspection:
    """Inspect an in-memory xarray Dataset as the resampling source.

    Used by the single-pass fusion path, where the conversion output is a
    lazy in-memory Dataset instead of an on-disk intermediate Zarr store.
    The caller owns ``dataset``; it is not closed here.
    """

    info = dataset_info_from_dataset(dataset, Path("<fused-in-memory>"))
    grid = inspect_dataset_grid(dataset, info)
    report = format_inspection(info) + "\n\n" + format_grid_report(grid)
    return ResampleInspection(info=info, grid=grid, report=report)
