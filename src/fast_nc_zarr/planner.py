from __future__ import annotations

import math
import os
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from .hardware import load_cached_profile
from .models import ConversionPlan, Inventory, OutputLayout, Selection
from .performance_model import rank_candidates
from .system import EffectiveResourceBudget, effective_resource_budget, storage_profile

MIB = 1024**2

def output_layout_plan_chunks(
    selection: Selection,
    output_layout: OutputLayout,
) -> tuple[int, int, int] | None:
    """Return a conservative scheduler chunk shape for a fixed variable layout."""
    shapes: list[tuple[int, int, int]] = []
    by_source = {item.source_name: item for item in output_layout.variables}
    for name in selection.variables:
        item = by_source.get(name)
        if item is None:
            continue
        chunks_by_dim = dict(zip(item.dims, item.chunks))
        if all(dim in chunks_by_dim for dim in ("time", "lat", "lon")):
            shapes.append(
                tuple(int(chunks_by_dim[dim]) for dim in ("time", "lat", "lon"))
            )
    if not shapes:
        return None
    return tuple(
        min(size, max(shape[axis] for shape in shapes))
        for axis, size in enumerate(selection.shape)
    )


def output_layout_max_chunk_bytes(
    selection: Selection,
    output_layout: OutputLayout,
) -> int:
    """Return the largest encoded variable chunk's decoded byte footprint."""
    maximum = 0
    by_source = {item.source_name: item for item in output_layout.variables}
    for name in selection.variables:
        item = by_source.get(name)
        if item is None:
            continue
        cells = math.prod(int(chunk) for chunk in item.chunks)
        maximum = max(maximum, cells * np.dtype(item.dtype).itemsize)
    return maximum


def workload_kind(inventory: Inventory) -> str:
    sizes = sorted(item.size_bytes for item in inventory.files)
    p90 = sizes[min(len(sizes) - 1, int(len(sizes) * 0.9))]
    if (
        # A genuinely metadata-heavy workload has thousands of independent
        # files.  Using only a 32 MiB size cutoff classified CSIF/GOSIF as
        # "many small files" even though their file count is moderate and a
        # chunk-oriented plan is worth testing.
        len(inventory.files) >= 4096
        and inventory.median_file_bytes <= 64 * MIB
        and inventory.median_times_per_file <= 2
    ):
        return "many-small-files"
    if (
        inventory.median_file_bytes >= 256 * MIB
        or p90 >= 512 * MIB
        or inventory.median_times_per_file >= 64
    ):
        return "large-files"
    return "balanced"


def direct_compatible(inventory: Inventory, selection: Selection) -> tuple[bool, str]:
    incompatible = [
        name for name in selection.variables if not inventory.variables[name].direct_compatible
    ]
    if incompatible:
        return False, "以下变量不是数值型 time/lat/lon 三维变量：" + ", ".join(incompatible)
    return True, "所选变量支持直接 chunk 写入"


def storage_aware_initial_workers(
    budget: EffectiveResourceBudget,
    kind: str,
    *,
    same_device: bool,
) -> int:
    """Return a storage-aware *initial* worker hint for automatic plans.

    This is a heuristic starting point, not a hard ceiling.  HDD/network
    devices start with fewer concurrent workers to avoid seek and round-trip
    contention, but the tuning stage still explores the full safe worker
    range so a faster HDD/network device can select higher concurrency.
    SSD and unknown media keep the CPU/memory ceiling as the initial hint.
    Explicit ``max_workers`` remains authoritative in callers.
    """
    ceiling = max(1, int(budget.worker_ceiling))
    source = budget.source_storage
    medium = source.medium if source is not None else "unknown"
    if medium == "hdd":
        if kind == "many-small-files":
            initial = 8
        else:
            initial = 4
    elif medium == "network":
        initial = 2 if same_device else 4
    else:
        initial = ceiling
    return max(1, min(ceiling, initial))


def _source_medium(budget: EffectiveResourceBudget, fallback: Path) -> str:
    source = budget.source_storage
    return source.medium if source is not None else "unknown"


def _common_native(inventory: Inventory, dim: str, fallback: int) -> int:
    values = []
    for spec in inventory.variables.values():
        if not spec.native_chunks or dim not in spec.dims:
            continue
        values.append(spec.native_chunks[spec.dims.index(dim)])
    return Counter(values).most_common(1)[0][0] if values else fallback


def _aligned(value: int, native: int, limit: int) -> int:
    value = min(max(1, value), limit)
    if native <= 1:
        return value
    return min(limit, max(native, int(round(value / native)) * native))


def spatial_chunks(
    inventory: Inventory,
    selection: Selection,
    chunk_time: int,
    target_mib: int,
) -> tuple[int, int]:
    _, ny, nx = selection.shape
    itemsize = max(inventory.variables[name].itemsize for name in selection.variables)
    cells = max(1, target_mib * MIB // max(1, chunk_time * itemsize))
    ratio = nx / max(ny, 1)
    y = max(1, int(math.sqrt(cells / max(ratio, 1e-9))))
    x = max(1, int(cells / y))
    native_y = _common_native(inventory, "lat", min(256, ny))
    native_x = _common_native(inventory, "lon", min(512, nx))
    return _aligned(y, native_y, ny), _aligned(x, native_x, nx)


def initial_plan(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    *,
    reserve_gib: float = 2.0,
    resource_budget: EffectiveResourceBudget | None = None,
) -> ConversionPlan:
    budget = resource_budget or effective_resource_budget(
        source=inventory.input_dir,
        output=output,
        reserve_memory_bytes=int(max(0.0, float(reserve_gib)) * 1024**3),
    )
    compatible, reason = direct_compatible(inventory, selection)
    if not compatible:
        chunk_time = min(selection.shape[0], 16)
        cy, cx = spatial_chunks(inventory, selection, chunk_time, 32)
        largest_item = max(inventory.variables[name].itemsize for name in selection.variables)
        chunk_bytes = chunk_time * cy * cx * largest_item
        memory_workers = max(
            1,
            budget.memory_budget_bytes // max(256 * MIB, chunk_bytes * 4),
        )
        kind = workload_kind(inventory)
        same_device = "source+output" in budget.same_device_roles
        storage_workers = min(
            budget.worker_ceiling,
            storage_aware_initial_workers(budget, kind, same_device=same_device),
        )
        return ConversionPlan(
            "dask",
            min(storage_workers, int(memory_workers)),
            chunk_time,
            cy,
            cx,
            rationale=(reason, "回退路径按统一有效资源预算限制 chunk 和进程数"),
        )

    kind = workload_kind(inventory)
    source = budget.source_storage or storage_profile(inventory.input_dir)
    destination = budget.output_storage or storage_profile(output)
    same_device = source.device != "unknown" and source.device == destination.device
    cpus = max(1, int(budget.worker_ceiling))
    storage_workers = min(
        cpus,
        storage_aware_initial_workers(budget, kind, same_device=same_device),
    )
    nt, _, _ = selection.shape
    rationale = [f"输入形态：{kind}"]
    if source.medium:
        rationale.append(f"源存储 profile：{source.medium}/{source.filesystem}")
    if same_device:
        rationale.append("源和目标同设备；自动计划采用存储感知 worker 上限")
    if storage_workers < cpus:
        rationale.append(f"存储感知 worker 上限：{storage_workers}")

    if kind == "many-small-files":
        chunk_time = 1
        strategy = "file"
        target_mib = 4
        workers = storage_workers
        batch = 4
    elif kind == "large-files":
        native_t = _common_native(inventory, "time", 16)
        chunk_time = min(nt, max(8, min(64, native_t)))
        strategy = "chunk"
        target_mib = 64
        workers = storage_workers
        batch = 4 if source.medium == "hdd" else 1
    else:
        typical_t = max(1, inventory.median_times_per_file)
        chunk_time = min(nt, max(1, min(32, typical_t)))
        strategy = "chunk"
        target_mib = 32
        workers = storage_workers
        batch = 4 if source.medium == "hdd" else 1

    cy, cx = spatial_chunks(inventory, selection, chunk_time, target_mib)
    chunk_bytes = chunk_time * cy * cx * max(
        inventory.variables[name].itemsize for name in selection.variables
    )
    memory_workers = max(1, budget.memory_budget_bytes // max(128 * MIB, chunk_bytes * 3))
    workers = max(1, min(workers, int(memory_workers)))
    return ConversionPlan(
        strategy,
        workers,
        chunk_time,
        cy,
        cx,
        task_batch=batch,
        rationale=tuple(rationale),
    )


def resolve_conversion_plan(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    *,
    plan: ConversionPlan | None = None,
    chunks: tuple[int, int, int] | None = None,
    max_workers: int | None = None,
    reserve_gib: float = 2.0,
    resource_budget: EffectiveResourceBudget | None = None,
) -> ConversionPlan:
    """Resolve all non-benchmark conversion overrides in one place.

    Preview and execution must agree on task ownership.  In particular, a
    multi-time external chunk cannot retain the generic one-time-per-task
    file strategy, while filename-time workers must own the whole time chunk
    to prevent concurrent partial writes.
    """

    resolved = plan or initial_plan(
        inventory,
        selection,
        output,
        reserve_gib=reserve_gib,
        resource_budget=resource_budget,
    )
    if chunks is not None:
        if len(chunks) != 3 or any(int(value) <= 0 for value in chunks):
            raise ValueError("转换 chunks 必须是三个正整数。")
        chunk_time = min(int(chunks[0]), selection.shape[0])
        chunk_lat = min(int(chunks[1]), selection.shape[1])
        chunk_lon = min(int(chunks[2]), selection.shape[2])
        strategy = resolved.strategy
        task_batch = resolved.task_batch
        if inventory.source_mode == "filename":
            task_batch = chunk_time
        elif strategy == "file" and chunk_time > 1:
            strategy = "chunk"
            task_batch = 1
        rationale = resolved.rationale
        marker = "使用外部编排器提供的输出 chunks。"
        if marker not in rationale:
            rationale += (marker,)
        resolved = replace(
            resolved,
            strategy=strategy,
            chunk_time=chunk_time,
            chunk_lat=chunk_lat,
            chunk_lon=chunk_lon,
            task_batch=task_batch,
            rationale=rationale,
        )
    if max_workers is not None:
        if int(max_workers) <= 0:
            raise ValueError("max_workers 必须是正整数。")
        resolved = replace(resolved, workers=min(resolved.workers, int(max_workers)))
    return resolved


def fixed_layout_candidate_plans(
    inventory: Inventory,
    selection: Selection,
    base: ConversionPlan,
    *,
    max_workers: int | None = None,
    reserve_gib: float = 2.0,
    worker_chunk_bytes: int | None = None,
    resource_budget: EffectiveResourceBudget | None = None,
) -> list[ConversionPlan]:
    """Vary execution concurrency without changing a supplied storage layout."""

    if base.strategy == "dask":
        return [base]
    if max_workers is not None and int(max_workers) <= 0:
        raise ValueError("max_workers 必须是正整数。")

    budget = resource_budget or effective_resource_budget(
        source=inventory.input_dir,
        reserve_memory_bytes=int(max(0.0, float(reserve_gib)) * 1024**3),
    )
    cpu_limit = int(budget.worker_ceiling)
    if max_workers is not None:
        cpu_limit = min(cpu_limit, int(max_workers))
    largest_item = max(
        inventory.variables[name].itemsize for name in selection.variables
    )
    chunk_bytes = (
        min(base.chunk_time, selection.shape[0])
        * min(base.chunk_lat, selection.shape[1])
        * min(base.chunk_lon, selection.shape[2])
        * largest_item
    )
    if worker_chunk_bytes is not None:
        chunk_bytes = max(chunk_bytes, max(1, int(worker_chunk_bytes)))
    estimated_worker = max(128 * MIB, chunk_bytes * 3)
    memory_limit = max(1, budget.memory_budget_bytes // estimated_worker)
    worker_limit = max(1, min(cpu_limit, int(memory_limit)))

    # Fixed layouts must expose every safe worker count to the benchmark;
    # storage profiles are context, not a static cap.
    worker_values = set(range(1, worker_limit + 1))

    _, ny, nx = selection.shape
    spatial_chunks = math.ceil(ny / base.chunk_lat) * math.ceil(nx / base.chunk_lon)
    maximum_batch = max(1, min(16, spatial_chunks))
    batch_values = {1, min(maximum_batch, max(1, base.task_batch)), maximum_batch}
    batch_values.update(
        value for value in (2, 4, 8, 16) if value <= maximum_batch
    )

    candidates = [
        replace(
            base,
            workers=workers,
            task_batch=min(maximum_batch, max(1, base.task_batch)),
        )
        for workers in sorted(worker_values)
    ]
    batching_workers = min(max(1, base.workers), worker_limit)
    candidates.extend(
        replace(base, workers=batching_workers, task_batch=batch)
        for batch in sorted(batch_values)
    )

    unique: list[ConversionPlan] = []
    seen: set[tuple[int, int]] = set()
    for item in candidates:
        key = (item.workers, item.task_batch)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def candidate_plans(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    *,
    max_workers: int | None = None,
    reserve_gib: float = 2.0,
    resource_budget: EffectiveResourceBudget | None = None,
) -> list[ConversionPlan]:
    base = initial_plan(
        inventory,
        selection,
        output,
        reserve_gib=reserve_gib,
        resource_budget=resource_budget,
    )
    if base.strategy == "dask":
        return [base]
    budget = resource_budget or effective_resource_budget(
        source=inventory.input_dir,
        output=output,
        reserve_memory_bytes=int(max(0.0, float(reserve_gib)) * 1024**3),
    )
    cpu_limit = int(budget.worker_ceiling)
    if max_workers is not None:
        cpu_limit = min(cpu_limit, int(max_workers))
    worker_values = list(range(1, max(1, cpu_limit) + 1))

    kind = workload_kind(inventory)
    source_medium = _source_medium(budget, inventory.input_dir)
    if kind == "many-small-files":
        time_values = [1]
        targets = [2, 4, 8, 16]
        batches = [1, 2, 4, 8, 16]
    elif kind == "large-files":
        native_t = _common_native(inventory, "time", base.chunk_time)
        time_values = sorted({min(selection.shape[0], value) for value in (native_t, 16, 32, 64) if value})
        targets = [32, 64, 128, 256]
        batches = [1, 4] if source_medium == "hdd" else [1]
    else:
        time_values = sorted({min(selection.shape[0], value) for value in (1, 8, 16, 32)})
        targets = [16, 32, 64, 128]
        batches = [1, 4] if source_medium == "hdd" else [1]

    candidates: list[ConversionPlan] = [
        replace(base, workers=workers) for workers in worker_values
    ]
    # Filename-time products with one slice per file can benefit from a
    # file-centric plan even when the general balanced profile starts with a
    # chunk plan.  Keep this after the worker candidate so a short tune budget
    # still evaluates the most important parallelism change first.
    if (
        inventory.source_mode == "filename"
        and kind == "balanced"
        and inventory.median_times_per_file <= 2
    ):
        candidates.append(
            replace(
                base,
                strategy="file",
                chunk_time=1,
                task_batch=max(2, base.task_batch),
            )
        )
    first_y, first_x = spatial_chunks(
        inventory, selection, base.chunk_time, targets[0]
    )
    candidates.append(replace(base, chunk_lat=first_y, chunk_lon=first_x))
    candidates.extend(
        [
            replace(base, compression="lz4", shuffle="shuffle"),
            replace(base, compression="zstd", shuffle="shuffle"),
            replace(base, compression="zstd", shuffle="bitshuffle"),
        ]
    )
    # Use a compact, orthogonal candidate set. Exhaustive cross products make
    # tuning more expensive than the conversion for small datasets.
    for target in targets:
        cy, cx = spatial_chunks(inventory, selection, base.chunk_time, target)
        candidates.append(replace(base, chunk_lat=cy, chunk_lon=cx))
    for chunk_time in time_values:
        cy, cx = spatial_chunks(inventory, selection, chunk_time, 32 if kind != "many-small-files" else 4)
        candidates.append(replace(base, chunk_time=chunk_time, chunk_lat=cy, chunk_lon=cx))
    for workers in worker_values:
        candidates.append(replace(base, workers=workers))
    for batch in batches:
        candidates.append(replace(base, task_batch=batch))

    unique = []
    seen = set()
    for item in candidates:
        largest_item = max(inventory.variables[name].itemsize for name in selection.variables)
        estimated_worker = max(
            128 * MIB,
            item.chunk_time * item.chunk_lat * item.chunk_lon * largest_item * 3,
        )
        memory_workers = max(1, budget.memory_budget_bytes // estimated_worker)
        item = replace(item, workers=min(item.workers, int(memory_workers)))
        key = (
            item.strategy,
            item.workers,
            item.chunk_time,
            item.chunk_lat,
            item.chunk_lon,
            item.task_batch,
            item.compression,
            item.compression_level,
            item.shuffle,
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)
    # Optional performance-model reordering: when a cached HardwareProfile is
    # available and FAST_NC_ZARR_PERF_MODEL=1, order candidates by estimated
    # wall time (fastest first) without dropping any candidate.  This lets a
    # short tuning budget evaluate promising plans earlier.
    if os.environ.get("FAST_NC_ZARR_PERF_MODEL") == "1":
        profile = load_cached_profile((inventory.input_dir, output))
        if profile is not None:
            unique = [
                estimate.plan
                for estimate in rank_candidates(
                    unique, inventory, selection, profile
                )
            ]
    # Never truncate the 1..worker-ceiling parallelism sweep; the budget
    # controls how many trials run, while the manifest must expose all safe
    # worker candidates.  Only orthogonal layout candidates are compacted.
    limit = 14 if kind != "large-files" else 12
    if len(worker_values) >= limit:
        return unique
    return unique[:limit]
