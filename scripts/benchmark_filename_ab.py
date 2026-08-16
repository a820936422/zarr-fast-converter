#!/usr/bin/env python3
"""Compare filename chunk-owner writes with a partial-region baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import threading
import time

import numpy as np

from fast_nc_zarr.filename_mode import (
    _initialize_filename_zarr,
    _normalized_filename_dataset,
    _prepare_filename_data,
    filename_direct_write,
    inspect_filename_inventory,
    scan_filename_times,
    source_path_by_time,
)
from fast_nc_zarr.models import ConversionPlan, Selection, VariableTransform


def _tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


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


def _blocks(shape: tuple[int, int, int], chunks: tuple[int, int, int]):
    _, ny, nx = shape
    return tuple(
        (y0, min(ny, y0 + max(1, chunks[1])), x0, min(nx, x0 + max(1, chunks[2])))
        for y0 in range(0, ny, max(1, chunks[1]))
        for x0 in range(0, nx, max(1, chunks[2]))
    )


def _partial_region_write(
    inventory,
    selection: Selection,
    output: Path,
    plan: ConversionPlan,
    *,
    transforms: dict[str, VariableTransform],
) -> dict[str, float | int]:
    import zarr

    effective = _initialize_filename_zarr(
        inventory, selection, output, plan, transforms, {}, None
    )
    group = zarr.open_group(output, mode="r+")
    source_map = source_path_by_time(inventory)
    selected_keys = inventory.time_keys[selection.time_start : selection.time_stop]
    blocks = _blocks(selection.shape, plan.chunks)
    source_opens = 0
    region_writes = 0
    logical_bytes = 0
    samples = [_rss_bytes()]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.05):
            samples.append(_rss_bytes())

    monitor_thread = threading.Thread(target=monitor, name="filename-ab-baseline-rss", daemon=True)
    started = time.perf_counter()
    monitor_thread.start()
    def write_regions(index: int, dataset) -> None:
        nonlocal logical_bytes, region_writes
        for y0, y1, x0, x1 in blocks:
            for name in selection.variables:
                dtype, fill = effective[name]
                shape = (1, y1 - y0, x1 - x0)
                data = np.empty(shape, dtype=dtype)
                if dataset is None:
                    data.fill(fill)
                else:
                    raw = dataset[name].isel(
                        lat=slice(selection.lat_start + y0, selection.lat_start + y1),
                        lon=slice(selection.lon_start + x0, selection.lon_start + x1),
                    ).values
                    data[0] = _prepare_filename_data(
                        raw,
                        inventory.variables[name],
                        transforms.get(name),
                        dtype,
                        fill,
                    )
                group[name][index : index + 1, y0:y1, x0:x1] = data
                logical_bytes += int(data.nbytes)
                region_writes += 1

    try:
        for index, key in enumerate(selected_keys):
            source_path = source_map.get(key)
            if source_path is None:
                write_regions(index, None)
            else:
                with _normalized_filename_dataset(
                    Path(source_path), inventory.source_engine
                ) as (dataset, _):
                    source_opens += 1
                    write_regions(index, dataset)
    finally:
        stop.set()
        monitor_thread.join(timeout=2.0)
    elapsed = max(time.perf_counter() - started, 1e-9)
    return {
        "elapsed": elapsed,
        "logical_bytes": logical_bytes,
        "throughput_mib_s": logical_bytes / 1024**2 / elapsed,
        "peak_rss": max(samples, default=0),
        "source_opens": source_opens,
        "region_writes": region_writes,
        "write_mode": "partial-region-baseline",
    }


def _run_case(inventory, selection: Selection, root: Path, plan: ConversionPlan) -> dict[str, object]:
    transforms: dict[str, VariableTransform] = {}
    owner_output = root / "chunk-owner.zarr"
    baseline_output = root / "partial-region.zarr"
    owner_output.parent.mkdir(parents=True, exist_ok=True)
    owner_started = time.perf_counter()
    owner_metrics = filename_direct_write(
        inventory,
        selection,
        owner_output,
        plan,
        transforms=transforms,
        validate=False,
        progress=False,
    )
    owner_metrics = {**owner_metrics, "elapsed": max(time.perf_counter() - owner_started, 1e-9)}
    baseline_metrics = _partial_region_write(
        inventory,
        selection,
        baseline_output,
        plan,
        transforms=transforms,
    )
    # The production writer already validated its write contract in tests; compare all arrays here.
    import xarray as xr

    with (
        xr.open_zarr(owner_output, consolidated=False, chunks=None, mask_and_scale=False) as owner,
        xr.open_zarr(baseline_output, consolidated=False, chunks=None, mask_and_scale=False) as baseline,
    ):
        if set(owner.variables) != set(baseline.variables):
            raise RuntimeError("A/B 输出变量集合不一致")
        for name in owner.variables:
            np.testing.assert_equal(owner[name].values, baseline[name].values)
    owner_physical = _tree_bytes(owner_output)
    baseline_physical = _tree_bytes(baseline_output)
    logical_bytes = int(owner_metrics["logical_bytes"])
    owner_writes = int(owner_metrics.get("chunk_owner_writes", 0))
    baseline_writes = int(baseline_metrics["region_writes"])
    return {
        "plan": asdict(plan),
        "owner": {
            **owner_metrics,
            "physical_bytes": owner_physical,
            "region_writes": owner_writes,
            "write_amplification": 1.0,
            "source_opens": len(
                [key for key in inventory.time_keys[selection.time_start : selection.time_stop] if key in source_path_by_time(inventory)]
            ),
        },
        "baseline": {
            **baseline_metrics,
            "physical_bytes": baseline_physical,
            "write_amplification": baseline_writes / max(owner_writes, 1),
        },
        "parity": True,
        "logical_bytes": logical_bytes,
        "physical_delta_bytes": baseline_physical - owner_physical,
    }


def benchmark(
    source: Path,
    output_root: Path,
    workers: tuple[int, ...],
    chunk_cases: tuple[tuple[int, int, int], ...],
    codecs: tuple[str, ...],
    *,
    time_start: int = 0,
    time_stop: int | None = None,
    lat_start: int = 0,
    lat_stop: int | None = None,
    lon_start: int = 0,
    lon_stop: int | None = None,
) -> dict[str, object]:
    source = source.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    scan = scan_filename_times(source)
    inventory = inspect_filename_inventory(scan, workers=1, progress=False)
    resolved_time_stop = len(inventory.times) if time_stop is None else int(time_stop)
    resolved_lat_stop = len(inventory.lat_values) if lat_stop is None else int(lat_stop)
    resolved_lon_stop = len(inventory.lon_values) if lon_stop is None else int(lon_stop)
    bounds = (
        int(time_start), resolved_time_stop,
        int(lat_start), resolved_lat_stop,
        int(lon_start), resolved_lon_stop,
    )
    if not (
        0 <= bounds[0] < bounds[1] <= len(inventory.times)
        and 0 <= bounds[2] < bounds[3] <= len(inventory.lat_values)
        and 0 <= bounds[4] < bounds[5] <= len(inventory.lon_values)
    ):
        raise ValueError("选择窗口必须位于 inventory 范围内且每个轴至少包含一个元素")
    selection = Selection(
        tuple(inventory.variables),
        bounds[0], bounds[1], bounds[2], bounds[3], bounds[4], bounds[5],
    )
    results: list[dict[str, object]] = []
    for codec in codecs:
        for chunks in chunk_cases:
            for worker_count in workers:
                plan = ConversionPlan(
                    "chunk",
                    worker_count,
                    chunks[0],
                    chunks[1],
                    chunks[2],
                    task_batch=chunks[0],
                    compression=codec,
                    compression_level=1,
                    shuffle="noshuffle",
                )
                case_root = output_root / f"{codec}-t{chunks[0]}-y{chunks[1]}-x{chunks[2]}-w{worker_count}"
                shutil.rmtree(case_root, ignore_errors=True)
                results.append(_run_case(inventory, selection, case_root, plan))
    return {
        "source": str(source),
        "source_mode": inventory.source_mode,
        "time_count": len(inventory.times),
        "missing_time_count": len(inventory.missing_time_keys),
        "shape": list(selection.shape),
        "selection": asdict(selection),
        "cases": results,
        "production_default_changed": False,
    }


def _parse_workers(value: str) -> tuple[int, ...]:
    workers = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not workers or any(item < 1 for item in workers):
        raise ValueError("workers 必须是正整数列表")
    return workers


def _parse_chunks(value: str) -> tuple[tuple[int, int, int], ...]:
    result = []
    for raw in value.split(";"):
        values = tuple(int(item) for item in raw.split(","))
        if len(values) != 3 or any(item < 1 for item in values):
            raise ValueError("chunks 必须是 t,y,x 三元组列表")
        result.append(values)
    return tuple(result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="filename-time NetCDF/HDF directory")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", default="1,2,4")
    parser.add_argument("--chunks", default="1,8,8;2,8,8;4,16,16")
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-stop", type=int)
    parser.add_argument("--lat-start", type=int, default=0)
    parser.add_argument("--lat-stop", type=int)
    parser.add_argument("--lon-start", type=int, default=0)
    parser.add_argument("--lon-stop", type=int)
    parser.add_argument("--codecs", default="none,zstd")
    args = parser.parse_args()
    try:
        workers = _parse_workers(args.workers)
        chunks = _parse_chunks(args.chunks)
    except ValueError as exc:
        parser.error(str(exc))
    codecs = tuple(item.strip() for item in args.codecs.split(",") if item.strip())
    print(json.dumps(
        benchmark(
            args.source,
            args.output_root,
            workers,
            chunks,
            codecs,
            time_start=args.time_start,
            time_stop=args.time_stop,
            lat_start=args.lat_start,
            lat_stop=args.lat_stop,
            lon_start=args.lon_start,
            lon_stop=args.lon_stop,
        ),
        ensure_ascii=False,
        indent=2,
        default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
