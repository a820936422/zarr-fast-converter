from __future__ import annotations

import shutil
import tempfile
import time
import os
from dataclasses import replace
from pathlib import Path
from typing import Callable

import numpy as np

from .models import BenchmarkResult, ConversionPlan, Inventory, OutputLayout, Selection
from .selection import selected_logical_bytes
from .writer import direct_write, fsync_tree

COMPRESSION_SAFETY = 1.25


def drop_source_page_cache(inventory: Inventory, selection: Selection) -> None:
    """Evict sampled source pages so later candidates do not get a warm-cache advantage."""
    advise = getattr(os, "posix_fadvise", None)
    advice = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or advice is None:
        return
    selected_keys = set(
        inventory.time_keys[selection.time_start : selection.time_stop]
    )
    paths = {
        record.path
        for record in inventory.files
        if selected_keys.intersection(record.time_keys)
    }
    for path in paths:
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                advise(descriptor, 0, 0, advice)
            finally:
                os.close(descriptor)
        except OSError:
            pass


def representative_selection(
    selection: Selection,
    candidates: list[ConversionPlan],
    *,
    bytes_per_cell: int | None = None,
    max_sample_mib: int = 1024,
) -> Selection:
    nt, ny, nx = selection.shape
    file_mode = any(item.strategy == "file" for item in candidates)
    if file_mode:
        sample_t = min(nt, max(8, max(item.workers * item.task_batch * 2 for item in candidates)))
    else:
        sample_t = min(nt, max(item.chunk_time for item in candidates))
    sample_y = min(ny, max(item.chunk_lat for item in candidates) * 2)
    sample_x = min(nx, max(item.chunk_lon for item in candidates) * 2)
    if bytes_per_cell and sample_y and sample_x:
        # Keep a large-file probe bounded.  Without this cap a candidate with
        # chunk_time=64 could materialize many GiB before the tuner has even
        # compared a second plan.
        cells_per_time = sample_y * sample_x * bytes_per_cell
        max_t = max(1, max_sample_mib * 1024**2 // cells_per_time)
        sample_t = min(sample_t, max_t)
    return Selection(
        selection.variables,
        selection.time_start,
        selection.time_start + sample_t,
        selection.lat_start,
        selection.lat_start + sample_y,
        selection.lon_start,
        selection.lon_start + sample_x,
    )


def representative_selections(
    selection: Selection,
    candidates: list[ConversionPlan],
    *,
    max_samples: int = 3,
    bytes_per_cell: int | None = None,
    max_sample_mib: int = 1024,
) -> list[Selection]:
    """Build small, stratified samples from the beginning/middle/end.

    A single leading sample can have a very different compression ratio from
    the rest of a remote-sensing time series.  Keeping the spatial window
    fixed makes candidates comparable while the temporal strata capture
    seasonal and yearly changes at a modest I/O cost.
    """
    sample = representative_selection(
        selection,
        candidates,
        bytes_per_cell=bytes_per_cell,
        max_sample_mib=max_sample_mib,
    )
    total = selection.shape[0]
    sample_t = sample.shape[0]
    if total <= sample_t or max_samples <= 1:
        return [sample]
    starts = [
        selection.time_start,
        selection.time_start + max(0, (total - sample_t) // 2),
        selection.time_stop - sample_t,
    ]
    result: list[Selection] = []
    seen: set[tuple[int, int]] = set()
    for start in starts[:max_samples]:
        start = max(selection.time_start, min(start, selection.time_stop - sample_t))
        key = (start, start + sample_t)
        if key in seen:
            continue
        seen.add(key)
        result.append(
            Selection(
                selection.variables,
                start,
                start + sample_t,
                sample.lat_start,
                sample.lat_stop,
                sample.lon_start,
                sample.lon_stop,
            )
        )
    return result or [sample]


def _sample_output_layout(
    layout: OutputLayout,
    sample: Selection,
) -> OutputLayout:
    """Project full-product shapes onto a tune sample without changing storage choices."""

    sizes = dict(zip(("time", "lat", "lon"), sample.shape))
    return replace(
        layout,
        variables=tuple(
            replace(
                item,
                shape=tuple(
                    sizes.get(dim, size)
                    for dim, size in zip(item.dims, item.shape)
                ),
            )
            for item in layout.variables
        ),
    )


def tune(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    candidates: list[ConversionPlan],
    *,
    budget_seconds: float = 60.0,
    progress: bool = True,
    writer: Callable = direct_write,
    writer_kwargs: dict | None = None,
    logical_bytes_fn: Callable[[Inventory, Selection], int] = selected_logical_bytes,
    max_samples: int = 3,
    fixed_layout: bool = False,
    minimum_candidates: int = 1,
) -> tuple[ConversionPlan, list[BenchmarkResult]]:
    if len(candidates) == 1:
        return candidates[0], []
    output.parent.mkdir(parents=True, exist_ok=True)
    bytes_per_cell = sum(
        np.dtype(inventory.variables[name].dtype).itemsize
        for name in selection.variables
    )
    samples = representative_selections(
        selection,
        candidates,
        max_samples=max_samples,
        bytes_per_cell=bytes_per_cell,
    )
    tune_root = Path(tempfile.mkdtemp(prefix=".fast-nc-zarr-tune-", dir=output.parent))
    results = []
    aggregates = {
        index: {
            "plan": plan,
            "logical": 0,
            "physical": 0,
            "write_elapsed": 0.0,
            "durable_elapsed": 0.0,
            "cpu_weighted": 0.0,
            "peak_rss": 0,
            "samples": 0,
        }
        for index, plan in enumerate(candidates, 1)
    }
    writer_kwargs = writer_kwargs or {}
    minimum_candidates = min(
        len(candidates), max(1, int(minimum_candidates))
    )
    tuning_started = time.perf_counter()
    if progress:
        print(
            f"开始实测 {len(candidates)} 组候选；分层样本 {len(samples)} 组，"
            f"样本 shape={samples[0].shape}，"
            f"总预算约 {budget_seconds:g} 秒。"
        )
    try:
        # Round-robin the temporal strata: every candidate gets a chance at
        # the first sample before an early candidate consumes the whole budget
        # on its middle/end samples.
        active = list(range(1, len(candidates) + 1))
        completed_any = False
        stopped = False
        for sample_round, sample in enumerate(samples, 1):
            round_indices = active if sample_round > 1 else list(range(1, len(candidates) + 1))
            next_active: list[int] = []
            for index in round_indices:
                if (
                    completed_any
                    and time.perf_counter() - tuning_started >= budget_seconds
                    and (sample_round > 1 or index > minimum_candidates)
                ):
                    stopped = True
                    break
                plan = candidates[index - 1]
                trial_output = tune_root / f"trial-{index}-{sample_round}.zarr"
                adjusted = plan
                if not fixed_layout:
                    adjusted = replace(
                        plan,
                        chunk_time=min(plan.chunk_time, sample.shape[0]),
                        chunk_lat=min(plan.chunk_lat, sample.shape[1]),
                        chunk_lon=min(plan.chunk_lon, sample.shape[2]),
                    )
                trial_writer_kwargs = writer_kwargs
                layout = writer_kwargs.get("output_layout")
                if fixed_layout and isinstance(layout, OutputLayout):
                    trial_writer_kwargs = dict(writer_kwargs)
                    trial_writer_kwargs["output_layout"] = _sample_output_layout(
                        layout, sample
                    )
                drop_source_page_cache(inventory, sample)
                started = time.perf_counter()
                try:
                    metrics = writer(
                        inventory,
                        sample,
                        trial_output,
                        adjusted,
                        progress=False,
                        **trial_writer_kwargs,
                    )
                    write_elapsed = float(metrics.get("elapsed", time.perf_counter() - started))
                    fsync_tree(trial_output)
                    durable_elapsed = time.perf_counter() - started
                    logical = int(metrics["logical_bytes"])
                    physical = sum(
                        path.stat().st_size
                        for path in trial_output.rglob("*")
                        if path.is_file()
                    )
                    aggregate = aggregates[index]
                    aggregate["logical"] += logical
                    aggregate["physical"] += physical
                    aggregate["write_elapsed"] += write_elapsed
                    aggregate["durable_elapsed"] += durable_elapsed
                    aggregate["cpu_weighted"] += float(metrics.get("average_cpu", 0.0)) * write_elapsed
                    aggregate["peak_rss"] = max(aggregate["peak_rss"], int(metrics.get("peak_rss", 0)))
                    aggregate["samples"] += 1
                    completed_any = True
                    next_active.append(index)
                except Exception as exc:
                    if progress:
                        print(
                            f"  [{index}/{len(candidates)} 样本 {sample_round}/{len(samples)}] "
                            f"{plan.label()} -> 失败：{exc}"
                        )
                finally:
                    shutil.rmtree(trial_output, ignore_errors=True)
            if stopped:
                if progress:
                    print("达到调参时间预算，停止新增候选。")
                break
            active = next_active
            if not active:
                break

        for index, aggregate in aggregates.items():
            completed_samples = int(aggregate["samples"])
            if not completed_samples:
                continue
            logical_total = int(aggregate["logical"])
            physical_total = int(aggregate["physical"])
            write_elapsed_total = float(aggregate["write_elapsed"])
            durable_elapsed_total = float(aggregate["durable_elapsed"])
            result = BenchmarkResult(
                plan=aggregate["plan"],
                elapsed=durable_elapsed_total,
                logical_bytes=logical_total,
                physical_bytes=physical_total,
                durable_mib_s=logical_total / max(durable_elapsed_total, 1e-9) / 1024**2,
                average_cpu=float(aggregate["cpu_weighted"]) / max(write_elapsed_total, 1e-9),
                peak_rss=int(aggregate["peak_rss"]),
                logical_mib_s=logical_total / max(write_elapsed_total, 1e-9) / 1024**2,
                physical_mib_s=physical_total / max(write_elapsed_total, 1e-9) / 1024**2,
                compression_ratio=physical_total / max(logical_total, 1),
                sample_count=completed_samples,
            )
            results.append(result)
            if progress:
                print(
                    f"  [{index}/{len(candidates)}] {result.plan.label()} -> "
                    f"逻辑 {result.logical_mib_s:.1f} MiB/s，"
                    f"物理 {result.physical_mib_s:.1f} MiB/s，"
                    f"耐久 {result.durable_mib_s:.1f} MiB/s，"
                    f"压缩率 {result.compression_ratio:.3f}，"
                    f"样本 {result.sample_count}，CPU {result.average_cpu:.0f}%"
                )
    finally:
        shutil.rmtree(tune_root, ignore_errors=True)
    if not results:
        raise RuntimeError("所有自动调参候选都执行失败。")
    full_logical = logical_bytes_fn(inventory, selection)
    free = shutil.disk_usage(output.parent).free
    feasible = [
        item
        for item in results
        if full_logical
            * (item.physical_bytes / max(item.logical_bytes, 1))
            * COMPRESSION_SAFETY
            <= free * 0.95
    ]
    if not feasible:
        smallest = min(
            results,
            key=lambda item: item.physical_bytes / max(item.logical_bytes, 1),
        )
        estimate = (
            full_logical
            * smallest.physical_bytes
            / max(smallest.logical_bytes, 1)
            * COMPRESSION_SAFETY
        )
        raise OSError(
            f"所有实测压缩方案的预计输出都超过磁盘安全容量；"
            f"最小方案约 {estimate / 1024**3:.1f} GiB，"
            f"可用 {free / 1024**3:.1f} GiB。"
        )
    # Production writes only perform the final durability step after all
    # workers finish, so optimize the end-to-end write speed while retaining
    # durable_mib_s for observability and safety reporting.
    best = max(feasible, key=lambda item: item.logical_mib_s)
    if progress:
        print(
            f"自动调参选择：{best.plan.label()}，"
            f"逻辑 {best.logical_mib_s:.1f} MiB/s，"
            f"耐久 {best.durable_mib_s:.1f} MiB/s"
        )
    return best.plan, results
