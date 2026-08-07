from __future__ import annotations

import math
from collections import Counter
from dataclasses import replace
from pathlib import Path

from .models import ConversionPlan, Inventory, Selection
from .system import available_memory, physical_cpu_count, storage_profile

MIB = 1024**2


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
) -> ConversionPlan:
    compatible, reason = direct_compatible(inventory, selection)
    if not compatible:
        chunk_time = min(selection.shape[0], 16)
        cy, cx = spatial_chunks(inventory, selection, chunk_time, 32)
        largest_item = max(inventory.variables[name].itemsize for name in selection.variables)
        chunk_bytes = chunk_time * cy * cx * largest_item
        memory_workers = max(
            1,
            available_memory(reserve_gib) // max(256 * MIB, chunk_bytes * 4),
        )
        return ConversionPlan(
            "dask",
            min(physical_cpu_count(), 4, int(memory_workers)),
            chunk_time,
            cy,
            cx,
            rationale=(reason, "回退路径同样按可用内存限制 chunk 和进程数"),
        )

    kind = workload_kind(inventory)
    source = storage_profile(inventory.input_dir)
    destination = storage_profile(output)
    same_device = source.device != "unknown" and source.device == destination.device
    cpus = physical_cpu_count()
    nt, _, _ = selection.shape
    rationale = [f"输入形态：{kind}"]
    if source.rotational:
        rationale.append("源位于机械硬盘")
    if same_device:
        rationale.append("源和目标位于同一文件系统，限制并发随机 I/O")

    if kind == "many-small-files":
        chunk_time = 1
        strategy = "file"
        target_mib = 4
        workers = min(cpus, 2 if same_device and source.rotational else 4)
        batch = 2 if source.rotational else 4
    elif kind == "large-files":
        native_t = _common_native(inventory, "time", 16)
        chunk_time = min(nt, max(8, min(64, native_t)))
        strategy = "chunk"
        target_mib = 64
        workers = min(cpus, 3 if same_device and source.rotational else cpus)
        batch = 1
    else:
        typical_t = max(1, inventory.median_times_per_file)
        chunk_time = min(nt, max(1, min(32, typical_t)))
        strategy = "chunk"
        target_mib = 32
        workers = min(cpus, 3 if same_device and source.rotational else cpus)
        batch = 1

    cy, cx = spatial_chunks(inventory, selection, chunk_time, target_mib)
    chunk_bytes = chunk_time * cy * cx * max(
        inventory.variables[name].itemsize for name in selection.variables
    )
    memory_limit = available_memory(reserve_gib)
    memory_workers = max(1, memory_limit // max(128 * MIB, chunk_bytes * 3))
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


def candidate_plans(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    *,
    max_workers: int | None = None,
    reserve_gib: float = 2.0,
) -> list[ConversionPlan]:
    base = initial_plan(inventory, selection, output, reserve_gib=reserve_gib)
    if base.strategy == "dask":
        return [base]
    cpu_limit = min(max_workers or physical_cpu_count(), physical_cpu_count())
    source = storage_profile(inventory.input_dir)
    destination = storage_profile(output)
    same_rotational = (
        source.rotational is True
        and source.device != "unknown"
        and source.device == destination.device
    )
    worker_values = {1, base.workers, min(cpu_limit, 2), min(cpu_limit, 4), cpu_limit}
    if not same_rotational:
        worker_values.discard(1)
    worker_values = sorted(value for value in worker_values if value >= 1)

    kind = workload_kind(inventory)
    if kind == "many-small-files":
        time_values = [1]
        targets = [2, 4, 8, 16]
        batches = [1, 2, 4, 8, 16]
    elif kind == "large-files":
        native_t = _common_native(inventory, "time", base.chunk_time)
        time_values = sorted({min(selection.shape[0], value) for value in (native_t, 16, 32, 64) if value})
        targets = [32, 64, 128, 256]
        batches = [1]
    else:
        time_values = sorted({min(selection.shape[0], value) for value in (1, 8, 16, 32)})
        targets = [16, 32, 64, 128]
        batches = [1]

    candidates: list[ConversionPlan] = [base]
    candidates.append(replace(base, workers=max(worker_values)))
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
        memory_workers = max(1, available_memory(reserve_gib) // estimated_worker)
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
    # Balanced and many-small profiles need a few more orthogonal candidates
    # (especially task batching); the tune budget still limits how many are
    # actually benchmarked.  Large-file profiles stay compact because each
    # trial can touch much more data.
    limit = 14 if kind != "large-files" else 12
    return unique[:limit]
