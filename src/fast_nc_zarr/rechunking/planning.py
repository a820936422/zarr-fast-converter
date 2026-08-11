from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import psutil

from ..system import worker_ceiling
from .models import ChunkPlan, DatasetInfo, Strategy



MIN_TARGET_MIB = 32.0
MAX_TARGET_MIB = 256.0
DEFAULT_TARGET_MIB = 128.0


def default_workers() -> int:
    return max(1, int(worker_ceiling()))


def _effective_target(
    requested_mib: float,
    workers: int,
) -> tuple[float, tuple[str, ...]]:
    if not math.isfinite(requested_mib) or requested_mib <= 0:
        raise ValueError("目标 chunk 大小必须是正数 MiB。")
    workers = max(1, int(workers))
    available_mib = psutil.virtual_memory().available / 1024**2
    # Reserve most available memory for xarray/Dask, source cache and the OS.
    memory_cap = available_mib * 0.10 / workers
    target = min(float(requested_mib), MAX_TARGET_MIB)
    rationale = [f"请求目标 {requested_mib:.1f} MiB"]
    if requested_mib > MAX_TARGET_MIB:
        rationale.append(f"按安全上限限制为 {MAX_TARGET_MIB:.0f} MiB")
    if memory_cap < target:
        target = max(MIN_TARGET_MIB, memory_cap)
        rationale.append(
            f"依据可用内存 {available_mib:.0f} MiB 和 {workers} 个 worker "
            f"调整为 {target:.1f} MiB"
        )
    target = max(MIN_TARGET_MIB, target)
    return target, tuple(rationale)


def _rounded_dimension(value: float, length: int, base: int = 16) -> int:
    if length <= 0:
        raise ValueError("输入 Zarr 的维度长度必须为正数。")
    candidate = min(length, max(1, int(value)))
    if candidate >= base * 2:
        candidate = max(base, (candidate // base) * base)
    return min(length, max(1, candidate))


def _spatial_tile(area: float, nlat: int, nlon: int) -> tuple[int, int]:
    aspect = nlat / nlon
    lat = math.sqrt(max(1.0, area) * aspect)
    lon = max(1.0, area / max(lat, 1.0))
    lat_chunk = _rounded_dimension(lat, nlat)
    lon_chunk = _rounded_dimension(lon, nlon)
    # If one side reached the full dimension, use the remaining area for the other.
    if lat_chunk >= nlat and lon_chunk < nlon:
        lon_chunk = _rounded_dimension(area / nlat, nlon)
    if lon_chunk >= nlon and lat_chunk < nlat:
        lat_chunk = _rounded_dimension(area / nlon, nlat)
    return lat_chunk, lon_chunk


def _max_itemsize(info: DatasetInfo) -> int:
    values = [variable.dtype.itemsize for variable in info.data_variables if variable.ndim == 3]
    if not values:
        raise ValueError("没有可用于规划的三维数据变量。")
    return max(values)


def _validate_custom(
    custom_chunks: Sequence[int] | None,
    shape: tuple[int, int, int],
) -> tuple[int, int, int]:
    if custom_chunks is None or len(custom_chunks) != 3:
        raise ValueError("自定义分块必须是包含 3 个元素的列表，例如 [10, 300, 300]。")
    values: list[int] = []
    for value, dimension in zip(custom_chunks, shape):
        if isinstance(value, bool) or int(value) != value or int(value) <= 0:
            raise ValueError("自定义 chunk 必须全部是正整数。")
        value = int(value)
        if value > dimension:
            raise ValueError(f"自定义 chunk {value} 不能超过对应维度长度 {dimension}。")
        values.append(value)
    return tuple(values)  # type: ignore[return-value]


def plan_chunks(
    info: DatasetInfo,
    strategy: Strategy,
    *,
    target_mib: float = DEFAULT_TARGET_MIB,
    workers: int | None = None,
    custom_chunks: Sequence[int] | None = None,
) -> ChunkPlan:
    if strategy not in {"time", "space", "custom"}:
        raise ValueError("分块策略必须是 time、space 或 custom。")
    workers = default_workers() if workers is None else max(1, int(workers))
    shape = info.shape
    target, rationale = _effective_target(target_mib, workers)
    if strategy == "custom":
        chunks = _validate_custom(custom_chunks, shape)
        rationale = rationale + ("使用用户提供的自定义 chunk。",)
    else:
        itemsize = _max_itemsize(info)
        target_bytes = target * 1024**2
        ntime, nlat, nlon = shape
        if strategy == "time":
            area = target_bytes / max(1, ntime * itemsize)
            lat_chunk, lon_chunk = _spatial_tile(area, nlat, nlon)
            chunks = (ntime, lat_chunk, lon_chunk)
            rationale = rationale + (
                "时间维度固定为完整长度，空间块按目标未压缩大小求解。",
            )
        else:
            spatial_bytes = nlat * nlon * itemsize
            time_chunk = _rounded_dimension(
                target_bytes / max(1, spatial_bytes),
                ntime,
                base=4,
            )
            chunks = (time_chunk, nlat, nlon)
            rationale = rationale + (
                "纬度和经度固定为完整长度，时间块按目标未压缩大小求解。",
            )

    estimated_chunk_bytes = int(np.prod(chunks, dtype=np.int64)) * _max_itemsize(info)
    estimated_chunks = {}
    for variable in info.data_variables:
        variable_chunks = tuple(
            min(chunk, size)
            for chunk, size in zip(chunks, variable.shape)
        )
        estimated_chunks[variable.name] = math.prod(
            math.ceil(size / chunk)
            for size, chunk in zip(variable.shape, variable_chunks)
        )
    return ChunkPlan(
        strategy=strategy,
        chunks=chunks,
        target_mib=target,
        estimated_chunk_bytes=estimated_chunk_bytes,
        estimated_chunks=estimated_chunks,
        rationale=rationale,
    )


def format_plan(plan: ChunkPlan, info: DatasetInfo) -> str:
    lines = [
        "========== 重分块计划 ==========",
        f"策略：{plan.strategy}",
        f"目标 chunk：{plan.target_mib:.1f} MiB（未压缩）",
        f"统一维度 chunks(time, lat, lon)：{plan.chunks}",
        f"最大 dtype 对应的单 chunk：{plan.estimated_chunk_bytes / 1024**2:.1f} MiB",
    ]
    for reason in plan.rationale:
        lines.append(f"  - {reason}")
    lines.append("变量 chunk 数量：")
    for variable in info.data_variables:
        lines.append(
            f"  {variable.name}: chunks={plan.chunks_for(variable)}, "
            f"预计 {plan.estimated_chunks[variable.name]} 个"
        )
    return "\n".join(lines)
