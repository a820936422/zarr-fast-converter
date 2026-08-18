#!/usr/bin/env python3
"""Reproducible conservative resampling A/B benchmark on synthetic Zarr v3 data.

This script is part of the v1.7.7 P1 work: it measures conservative and
conservative_normed resampling on a small regular grid and records wall-clock,
logical/physical bytes, throughput and resolved worker settings. It is intended
as a repeatable micro-benchmark; real-data A/B runs can reuse the same metrics
contract by pointing at a larger source store.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import xarray as xr

from fast_nc_zarr.resampling.engine import run_resample
from fast_nc_zarr.resampling.models import ResampleConfig

DEFAULT_METHODS = ("conservative", "conservative_normed")


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _make_source(
    root: Path,
    *,
    time_size: int,
    lat_size: int,
    lon_size: int,
) -> Path:
    source = root / "source.zarr"
    lat = np.linspace(89.0, -89.0, lat_size, dtype="float32")
    lon = np.linspace(-179.0, 179.0, lon_size, dtype="float32")
    values = np.arange(time_size * lat_size * lon_size, dtype="float32").reshape(
        time_size, lat_size, lon_size
    )
    dataset = xr.Dataset(
        {"value": (("time", "lat", "lon"), values, {"units": "test"})},
        coords={
            "time": np.arange(time_size, dtype="int64"),
            "lat": lat,
            "lon": lon,
        },
    )
    dataset.to_zarr(
        source,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding={"value": {"chunks": (1, max(2, lat_size // 2), max(2, lon_size // 2))}},
    )
    dataset.close()
    return source


def _run_case(
    source: Path,
    output_root: Path,
    method: str,
    resolution: float,
    repeats: int,
) -> dict[str, object]:
    output = output_root / f"{method}.zarr"
    times = []
    last_metrics: dict[str, object] = {}
    for repeat in range(repeats):
        if output.exists():
            shutil.rmtree(output)
        started = time.perf_counter()
        metrics = run_resample(
            ResampleConfig(
                source,
                output,
                resolution=resolution,
                method=method,
                skipna=True,
                validate=False,
                compute_workers=1,
                space_workers=1,
                tune_budget=0.0,
            ),
            progress=False,
        )
        elapsed = time.perf_counter() - started
        times.append(float(metrics["elapsed"]))
        last_metrics = metrics
        if repeat == 0:
            last_metrics = {**metrics, "repeat_0_wall_seconds": elapsed}
    return {
        "method": method,
        "resolution": resolution,
        "repeats": repeats,
        "elapsed_seconds": times,
        "mean_elapsed_seconds": float(np.mean(times)),
        "logical_bytes": int(last_metrics["logical_bytes"]),
        "physical_bytes": int(last_metrics["physical_bytes"]),
        "throughput_mib_s": float(last_metrics["throughput_mib_s"]),
        "physical_throughput_mib_s": float(last_metrics["physical_throughput_mib_s"]),
        "space_workers": int(last_metrics["space_workers"]),
        "compute_workers_per_space_worker": int(
            last_metrics["compute_workers_per_space_worker"]
        ),
        "owner_buffer_peak_bytes": int(
            last_metrics["owner_buffer"]["peak_bytes"]
        ),
        "output_path": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/tmp/fast-nc-zarr-conservative-ab"))
    parser.add_argument("--time-size", type=int, default=2)
    parser.add_argument("--source-lat-size", type=int, default=12)
    parser.add_argument("--source-lon-size", type=int, default=16)
    parser.add_argument("--resolution", type=float, default=20.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="comma-separated resampling methods to compare",
    )
    args = parser.parse_args()
    if min(args.time_size, args.source_lat_size, args.source_lon_size) < 1 or args.repeats < 1:
        parser.error("all sizes and repeats must be positive")
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    if not methods:
        parser.error("at least one method is required")

    root = args.output_root.expanduser().resolve()
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)
    source = _make_source(
        root,
        time_size=args.time_size,
        lat_size=args.source_lat_size,
        lon_size=args.source_lon_size,
    )
    cases = [
        _run_case(source, root, method, args.resolution, args.repeats)
        for method in methods
    ]
    payload = {
        "source": str(source),
        "source_shape": [args.time_size, args.source_lat_size, args.source_lon_size],
        "resolution": args.resolution,
        "repeats": args.repeats,
        "cases": cases,
        "production_default_changed": False,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
