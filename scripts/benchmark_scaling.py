#!/usr/bin/env python3
"""Scaling benchmark for the conversion write path (v1.8.0).

Measures conversion throughput / peak RSS at increasing worker counts on a
synthetic or user-supplied NetCDF source.  The report is written as JSON to
``--output-root/scaling-report.json`` and the script exits non-zero when any
run fails to produce positive throughput (used by the ``scaling-check`` gate).

Real-data runs: point ``--input`` at a NetCDF source directory (dimension or
filename mode).  Synthetic mode generates a small reproducible NetCDF tree so
the gate can run on any machine without external datasets.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from fast_nc_zarr.inspection import inspect_dataset
from fast_nc_zarr.models import ConversionPlan
from fast_nc_zarr.selection import make_selection
from fast_nc_zarr.validation import validate_output
from fast_nc_zarr.writer import direct_write

DEFAULT_WORKERS = (1, 2, 4, 8, 12)
SMOKE_WORKERS = (1, 2)
SMOKE_SHAPE = (2, 4, 8)
FULL_SHAPE = (24, 180, 360)
CHUNK_TIME = 4
CHUNK_LAT = 45
CHUNK_LON = 45


def _rss_bytes() -> int:
    try:
        import psutil

        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        seen: set[int] = set()
        total = 0
        for process in processes:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                total += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total
    except (ImportError, OSError):
        return 0


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _synthetic_source(root: Path, shape: tuple[int, int, int]) -> Path:
    """Write one dimension-mode NetCDF file with canonical time/lat/lon."""
    import netCDF4

    source = root / "source"
    source.mkdir(parents=True, exist_ok=True)
    path = source / "synthetic.nc"
    if path.exists():
        return source
    time_count, lat_count, lon_count = shape
    lat = np.linspace(60.0, -60.0, lat_count, dtype="float32")
    lon = np.linspace(-180.0, 180.0 - 360.0 / lon_count, lon_count, dtype="float32")
    values = (
        np.arange(time_count * lat_count * lon_count, dtype="float32").reshape(shape)
        % 1000
    )
    with netCDF4.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("time", time_count)
        dataset.createDimension("lat", lat_count)
        dataset.createDimension("lon", lon_count)
        time_var = dataset.createVariable("time", "i4", ("time",))
        time_var.units = "days since 2000-01-01 00:00:00"
        time_var.calendar = "standard"
        time_var[:] = np.arange(time_count, dtype="int32")
        lat_var = dataset.createVariable("lat", "f4", ("lat",))
        lat_var.units = "degrees_north"
        lat_var[:] = lat
        lon_var = dataset.createVariable("lon", "f4", ("lon",))
        lon_var.units = "degrees_east"
        lon_var[:] = lon
        data_var = dataset.createVariable(
            "value",
            "f4",
            ("time", "lat", "lon"),
            fill_value=np.float32(-9999.0),
        )
        data_var.units = "g m-2 d-1"
        data_var[:] = values
    return source


def _run_one(
    inventory,
    selection,
    output: Path,
    workers: int,
    compression: str,
) -> dict[str, object]:
    output.parent.mkdir(parents=True, exist_ok=True)
    plan = ConversionPlan(
        "chunk",
        workers,
        CHUNK_TIME,
        CHUNK_LAT,
        CHUNK_LON,
        task_batch=1,
        compression=compression,
        compression_level=1,
        shuffle="noshuffle",
    )
    samples = [_rss_bytes()]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.05):
            samples.append(_rss_bytes())

    monitor_thread = threading.Thread(target=monitor, name="scaling-benchmark-rss", daemon=True)
    started = time.perf_counter()
    monitor_thread.start()
    try:
        metrics = direct_write(
            inventory,
            selection,
            output,
            plan,
            progress=False,
        )
    finally:
        stop.set()
        monitor_thread.join(timeout=2.0)
    elapsed = max(time.perf_counter() - started, 1e-9)
    logical_bytes = int(metrics["logical_bytes"])
    physical_bytes = _tree_bytes(output)
    return {
        "workers": workers,
        "plan": asdict(plan),
        "elapsed_seconds": round(elapsed, 6),
        "logical_bytes": logical_bytes,
        "physical_bytes": physical_bytes,
        "throughput_mib_s": round(logical_bytes / 1024**2 / elapsed, 4),
        "peak_rss_bytes": max(samples, default=0),
        "chunks_written": int(metrics.get("chunks_written", 0)),
        "source_opens": int(metrics.get("source_opens", 0)),
    }


def benchmark(
    source: Path,
    output_root: Path,
    workers: tuple[int, ...],
    *,
    compression: str = "zstd",
    validate: bool = True,
    synthetic: bool = False,
    synthetic_shape: tuple[int, int, int] = FULL_SHAPE,
) -> dict[str, object]:
    source = source.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    if synthetic:
        source = _synthetic_source(output_root / ".scaling-source", synthetic_shape)
    inventory = inspect_dataset(source, workers=min(2, max(1, len(workers))), progress=False)
    selection = make_selection(
        inventory,
        variables=tuple(inventory.variables),
    )
    runs: list[dict[str, object]] = []
    for worker_count in workers:
        output = output_root / f"run-w{worker_count}.zarr"
        shutil.rmtree(output, ignore_errors=True)
        runs.append(_run_one(inventory, selection, output, worker_count, compression))
        if validate:
            validate_output(inventory, selection, output, points=5)
        shutil.rmtree(output, ignore_errors=True)
    report = {
        "tool": "benchmark_scaling.py",
        "source": str(source),
        "source_mode": inventory.source_mode,
        "synthetic": synthetic,
        "shape": list(selection.shape),
        "variables": list(selection.variables),
        "workers": list(workers),
        "compression": compression,
        "runs": runs,
    }
    report_path = output_root / "scaling-report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return report


def _parse_workers(value: str) -> tuple[int, ...]:
    workers = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not workers or any(item < 1 for item in workers):
        raise ValueError("workers 必须是正整数列表")
    return workers


def _parse_shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(item) for item in value.split(","))
    if len(parts) != 3 or any(item < 1 for item in parts):
        raise ValueError("shape 必须是 t,lat,lon 三个正整数")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="NetCDF source directory")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", default=",".join(str(item) for item in DEFAULT_WORKERS))
    parser.add_argument("--compression", default="zstd", help="codec name (zstd/lz4/none)")
    parser.add_argument("--no-validate", action="store_true", help="skip output validation")
    parser.add_argument("--synthetic", action="store_true", help="generate a synthetic NetCDF source")
    parser.add_argument("--shape", default=",".join(str(item) for item in FULL_SHAPE))
    parser.add_argument("--smoke", action="store_true", help="tiny run used by scaling-check gate")
    args = parser.parse_args()
    try:
        workers = _parse_workers(args.workers)
        shape = _parse_shape(args.shape)
    except ValueError as exc:
        parser.error(str(exc))
    if args.smoke:
        workers = SMOKE_WORKERS
        shape = SMOKE_SHAPE
        args.synthetic = True
    if args.synthetic is False and args.input is None:
        parser.error("必须提供 --input 或 --synthetic")
    try:
        report = benchmark(
            args.input or args.output_root,
            args.output_root,
            workers,
            compression=args.compression,
            validate=not args.no_validate,
            synthetic=args.synthetic,
            synthetic_shape=shape,
        )
    except Exception as exc:  # noqa: BLE001 - gate must fail loudly
        print(f"scaling benchmark failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    if any(float(run["throughput_mib_s"]) <= 0 for run in report["runs"]):
        print("scaling benchmark failed: 存在非正吞吐的 run", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
