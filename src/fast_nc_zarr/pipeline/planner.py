from __future__ import annotations

from hashlib import blake2b
import math
from pathlib import Path
from typing import Literal

import numpy as np

from ..models import CodecSpec, Inventory, OutputLayout, VariableOutputLayout
from ..planner import resolve_conversion_plan
from ..rechunking.compression import make_compression_plan
from ..resampling.grid import GridInfo, _axis_bounds, build_target_grid
from ..resampling.autotune import resolve_auto_time_block
from ..resampling.models import TargetGrid
from ..rechunking.models import ChunkPlan, CompressionPlan, DatasetInfo, VariableInfo
from ..rechunking.planning import plan_chunks
from ..resampling.engine import plan_resample
from ..resampling.inspection import inspect_resample_input
from ..resampling.models import ResampleConfig
from ..resampling.replacements import parse_replacement_rules
from .models import (
    OperationDecision,
    PipelineConfig,
    PipelinePlan,
    SourceReadWindow,
    ZarrPipelinePlan,
)
from ..selection import make_selection


def _grid_info(inventory: Inventory) -> GridInfo:
    lat = np.asarray(inventory.lat_values, dtype="float64")
    lon = np.asarray(inventory.lon_values, dtype="float64")
    if lat.size < 2 or lon.size < 2:
        raise ValueError("一条龙 V1 要求源 lat/lon 至少各有两个格点。")
    lat_diff = np.diff(lat)
    lon_diff = np.diff(lon)
    for name, values in (("lat", lat), ("lon", lon)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"源 {name} 坐标包含非有限值。")
    if not (np.all(lat_diff > 0) or np.all(lat_diff < 0)):
        raise ValueError("源 lat 坐标必须严格单调。")
    if not (np.all(lon_diff > 0) or np.all(lon_diff < 0)):
        raise ValueError("源 lon 坐标必须严格单调。")
    lat_resolution = float(np.median(np.abs(lat_diff)))
    lon_resolution = float(np.median(np.abs(lon_diff)))
    if not np.allclose(np.abs(lat_diff), lat_resolution, rtol=1e-5, atol=1e-10):
        raise ValueError("源 lat 坐标不是规则网格。")
    if not np.allclose(np.abs(lon_diff), lon_resolution, rtol=1e-5, atol=1e-10):
        raise ValueError("源 lon 坐标不是规则网格。")
    return GridInfo(
        path=inventory.input_dir,
        lat=lat,
        lon=lon,
        lat_bounds=_axis_bounds(lat),
        lon_bounds=_axis_bounds(lon),
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
        lat_descending=bool(lat[0] > lat[-1]),
        lon_descending=bool(lon[0] > lon[-1]),
        lat_uniform=True,
        lon_uniform=True,
    )


def _axis_window(
    values: np.ndarray,
    bounds: np.ndarray,
    low: float,
    high: float,
    method: str,
    *,
    periodic: bool = False,
) -> tuple[int, int, str]:
    """Return a contiguous source index window for one target axis.

    Conservative methods use source cell footprints.  Point/stencil methods
    retain enough source centers around target centers.  If an entire target
    axis is outside the source, retain the nearest source cell so xESMF can
    represent the no-extrapolation result as missing rather than making the
    conversion selection empty.
    """

    del periodic  # Periodic longitude is handled by the resampler itself.
    ascending = bool(values[0] < values[-1])
    order = np.arange(values.size) if ascending else np.arange(values.size - 1, -1, -1)
    centers = values[order]
    edges = np.asarray(bounds, dtype="float64")
    if not ascending:
        edges = edges[::-1]
    low = float(min(low, high))
    high = float(max(low, high))
    tolerance = max(1e-10, float(np.median(np.abs(np.diff(centers)))) * 1e-7)

    if method in {"conservative", "conservative_normed"}:
        overlaps = (edges[:-1] < high - tolerance) & (edges[1:] > low + tolerance)
        selected = np.flatnonzero(overlaps)
        description = "按源像元外边界与目标范围的面积交叠扩展"
    else:
        # The target extent can be much larger than a single target cell.  A
        # pair of boundary searches is enough to retain all centers in the
        # interior; the extra stencil width protects each edge.
        target_centers = np.asarray([low, high], dtype="float64")
        left = int(np.searchsorted(centers, target_centers[0], side="left"))
        right = int(np.searchsorted(centers, target_centers[1], side="right"))
        stencil = {"bilinear": 1, "patch": 2}.get(method, 0)
        left -= stencil
        right += stencil
        selected = np.arange(max(0, left), min(centers.size, right))
        description = f"按 {method} 插值 stencil 扩展 {stencil} 层"

    if selected.size == 0:
        # Keep the closest source cell for an entirely uncovered axis.  The
        # target coverage mask and xESMF no-extrapolation semantics still
        # determine the final missing values.
        nearest = int(np.argmin(np.abs(centers - (low + high) / 2.0)))
        selected = np.asarray([nearest], dtype=int)
        description += "；目标范围完全超出源范围，保留最近源格点用于缺测输出"
    if selected.size == 1 and centers.size > 1:
        # xESMF requires at least two coordinate points even when the target
        # is completely outside the source.  Retain an adjacent real cell;
        # the no-extrapolation mask still makes the result missing.
        only = int(selected[0])
        neighbour = only + 1 if only + 1 < centers.size else only - 1
        selected = np.asarray(sorted((only, neighbour)), dtype=int)
        description += "；为构造有效 xESMF 网格补留相邻源格点"
    start_order = int(selected.min())
    stop_order = int(selected.max()) + 1
    physical = np.sort(order[start_order:stop_order])
    return int(physical[0]), int(physical[-1]) + 1, description


def _axis_values_bounds(values: np.ndarray, start: int, stop: int, full_bounds: np.ndarray) -> tuple[float, float]:
    selected_bounds = np.asarray(full_bounds[start : stop + 1], dtype="float64")
    return float(np.min(selected_bounds)), float(np.max(selected_bounds))


def _inventory_id(inventory: Inventory) -> str:
    digest = blake2b(digest_size=16)
    digest.update(str(inventory.input_dir.resolve()).encode())
    for record in inventory.files:
        digest.update(str(record.path.resolve()).encode())
        digest.update(str(record.size_bytes).encode())
        digest.update(str(record.mtime_ns).encode())
    digest.update(np.asarray(inventory.times).tobytes())
    return digest.hexdigest()


def _exact_slice(values: np.ndarray, target: np.ndarray) -> tuple[int, int] | None:
    if target.size == 0 or target.size > values.size:
        return None
    for start in range(values.size - target.size + 1):
        if np.allclose(
            values[start : start + target.size],
            target,
            rtol=0.0,
            atol=1e-8,
        ):
            return start, start + target.size
    return None


def _final_layout(
    inventory: Inventory,
    selection,
    target,
    config,
    *,
    needs_resample: bool,
    baseline_chunks: tuple[int, int, int],
):
    """Derive the final chunk/codec plan without creating a temporary store."""

    dimensions = {
        "time": int(selection.shape[0]),
        "lat": int(target.lat.size),
        "lon": int(target.lon.size),
    }
    variables = []
    for name in selection.variables:
        spec = inventory.variables[name]
        if set(spec.dims) != {"time", "lat", "lon"}:
            continue
        dtype = np.dtype(spec.dtype)
        transform = config.conversion.variable_transforms.get(name)
        if transform is not None and transform.scale_factor is not None and dtype.kind not in "fc":
            dtype = np.dtype("float32" if dtype.itemsize <= 4 else "float64")
        if needs_resample:
            if not np.issubdtype(dtype, np.floating):
                dtype = np.dtype("float64")
            elif config.resampling.compute_dtype == "float32":
                dtype = np.dtype("float32")
        shape = tuple(dimensions[dim] for dim in ("time", "lat", "lon"))
        variables.append(
            VariableInfo(
                name=name,
                dims=("time", "lat", "lon"),
                shape=shape,
                dtype=dtype,
                chunks=shape,
                is_coord=False,
            )
        )
    if not variables:
        raise ValueError("一条龙至少需要一个 time/lat/lon 三维数值变量。")
    info = DatasetInfo(
        path=Path(config.general.output).expanduser().resolve(),
        dimensions=dimensions,
        variables=tuple(variables),
        attrs={},
        zarr_format=3,
    )
    if config.operations.rechunk:
        chunk_plan = plan_chunks(
            info,
            config.chunking.strategy,
            target_mib=config.chunking.target_mib,
            workers=config.chunking.workers,
            custom_chunks=config.chunking.custom_chunks,
        )
    else:
        chunks = tuple(
            min(int(size), max(1, int(chunk)))
            for size, chunk in zip(info.shape, baseline_chunks)
        )
        max_itemsize = max(variable.dtype.itemsize for variable in info.data_variables)
        chunk_plan = ChunkPlan(
            strategy="custom",
            chunks=chunks,
            target_mib=(int(np.prod(chunks, dtype=np.int64)) * max_itemsize) / 1024**2,
            estimated_chunk_bytes=int(np.prod(chunks, dtype=np.int64)) * max_itemsize,
            estimated_chunks={
                variable.name: math.prod(
                    math.ceil(size / chunk)
                    for size, chunk in zip(variable.shape, chunks)
                )
                for variable in info.data_variables
            },
            rationale=("未请求重分块，使用终端写入器的标准 chunks。",),
        )
    compression = (
        _pipeline_compression_plan(config)
        if config.operations.recompress
        else CompressionPlan(
            "none",
            None,
            "未请求重压缩；数据变量使用转换基线 Zstd level 1 codec。",
            codec="none",
        )
    )
    return chunk_plan, compression


def _pipeline_compression_plan(config: PipelineConfig) -> CompressionPlan:
    options = config.compression
    return make_compression_plan(
        options.profile,
        codec=options.codec,
        level=options.level,
        shuffle=options.shuffle,
    )


def _converted_dtype(inventory, name: str, config: PipelineConfig) -> np.dtype:
    """Return the dtype that conversion, rather than raw metadata, writes."""

    dtype = np.dtype(inventory.variables[name].dtype)
    transform = config.conversion.variable_transforms.get(name)
    if (
        transform is not None
        and transform.scale_factor is not None
        and dtype.kind not in "fc"
    ):
        return np.dtype("float32" if dtype.itemsize <= 4 else "float64")
    return dtype


def _codec_spec(dtype: np.dtype, compression) -> CodecSpec | None:
    if not compression.enabled:
        # The existing conversion engine writes this codec before a
        # compression-preserving final rechunk.
        return CodecSpec("blosc", level=1, cname="zstd", shuffle="noshuffle")
    if compression.codec == "zstd":
        return CodecSpec("zstd", level=int(compression.level or 0))
    if compression.codec == "gzip":
        return CodecSpec("gzip", level=int(compression.level or 0))
    if compression.shuffle != "auto":
        shuffle = compression.shuffle
    elif np.issubdtype(dtype, np.integer):
        shuffle = "bitshuffle"
    elif np.issubdtype(dtype, np.floating):
        shuffle = "shuffle"
    else:
        shuffle = "noshuffle"
    return CodecSpec(
        "blosc",
        level=int(compression.level or 1),
        cname=compression.codec.removeprefix("blosc-"),
        shuffle=shuffle,
    )


def _output_layout(
    inventory: Inventory,
    selection,
    target,
    config: PipelineConfig,
    chunk_plan,
    compression,
    *,
    needs_resample: bool,
) -> OutputLayout:
    axis_reversals: tuple[Literal["lat", "lon"], ...] = ()
    if not needs_resample:
        selected_axes = {
            "lat": np.asarray(
                inventory.lat_values[selection.lat_start : selection.lat_stop]
            ),
            "lon": np.asarray(
                inventory.lon_values[selection.lon_start : selection.lon_stop]
            ),
        }
        target_axes = {"lat": np.asarray(target.lat), "lon": np.asarray(target.lon)}
        reversals = []
        for name in ("lat", "lon"):
            source_values = selected_axes[name]
            target_values = target_axes[name]
            same_size = source_values.shape == target_values.shape
            if same_size and np.allclose(
                source_values, target_values, rtol=1e-7, atol=1e-10
            ):
                continue
            if same_size and np.allclose(
                source_values[::-1], target_values, rtol=1e-7, atol=1e-10
            ):
                reversals.append(name)
                continue
            raise ValueError(
                f"无需重采样时，目标 {name} 坐标必须与所选源坐标同序或完全逆序。"
            )
        axis_reversals = tuple(reversals)
    shape = (selection.shape[0], int(target.lat.size), int(target.lon.size))
    layouts = []
    for name in selection.variables:
        dtype = _converted_dtype(inventory, name, config)
        if needs_resample:
            if not np.issubdtype(dtype, np.floating):
                dtype = np.dtype("float64")
            elif config.resampling.compute_dtype == "float32":
                dtype = np.dtype("float32")
        output_name = config.conversion.variable_names.get(name, name)
        layouts.append(
            VariableOutputLayout(
                source_name=name,
                output_name=output_name,
                dims=("time", "lat", "lon"),
                shape=shape,
                dtype=str(dtype),
                chunks=tuple(min(size, chunk) for size, chunk in zip(shape, chunk_plan.chunks)),
                codec=_codec_spec(dtype, compression),
            )
        )
    coordinate_codec = CodecSpec("zstd", level=1)
    coordinate_values = {
        "time": inventory.times[selection.time_start : selection.time_stop],
        "lat": target.lat,
        "lon": target.lon,
    }
    dim_chunks = dict(zip(("time", "lat", "lon"), chunk_plan.chunks))
    for name, values in coordinate_values.items():
        layouts.append(
            VariableOutputLayout(
                source_name=name,
                output_name=name,
                dims=(name,),
                shape=(len(values),),
                dtype=str(np.asarray(values).dtype),
                chunks=(min(len(values), dim_chunks[name]),),
                codec=coordinate_codec,
                is_coord=True,
            )
        )
    return OutputLayout(tuple(layouts), axis_reversals=axis_reversals)


def _selected_source_grid(grid: GridInfo, selection) -> TargetGrid:
    lat = np.asarray(grid.lat[selection.lat_start : selection.lat_stop], dtype="float64")
    lon = np.asarray(grid.lon[selection.lon_start : selection.lon_stop], dtype="float64")
    lat_bounds = np.asarray(
        grid.lat_bounds[selection.lat_start : selection.lat_stop + 1],
        dtype="float64",
    )
    lon_bounds = np.asarray(
        grid.lon_bounds[selection.lon_start : selection.lon_stop + 1],
        dtype="float64",
    )
    if lat[0] < lat[-1]:
        lat = lat[::-1]
        lat_bounds = lat_bounds[::-1]
    if lon[0] > lon[-1]:
        lon = lon[::-1]
        lon_bounds = lon_bounds[::-1]
    return TargetGrid(
        lat=lat,
        lon=lon,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        lat_resolution=grid.lat_resolution,
        lon_resolution=grid.lon_resolution,
        extent="custom",
    )


def _resampling_conversion_chunks(
    inventory: Inventory,
    selection,
    grid: GridInfo,
    target: TargetGrid,
    config: PipelineConfig,
) -> tuple[int, int, int]:
    options = config.resampling
    requested_time = options.time_block
    if requested_time == "auto":
        requested_time = min(64, selection.shape[0])
    requested_tile = 256 if options.tile_size == "auto" else options.tile_size
    stencil = 2 if options.method in {"bilinear", "patch"} else 1
    source_lat_chunk = int(
        np.ceil(float(requested_tile) * options.resolution / grid.lat_resolution)
    ) + stencil
    source_lon_chunk = int(
        np.ceil(float(requested_tile) * options.resolution / grid.lon_resolution)
    ) + stencil
    chunks = (
        min(int(requested_time), selection.shape[0]),
        min(max(1, source_lat_chunk), selection.shape[1]),
        min(max(1, source_lon_chunk), selection.shape[2]),
    )
    if options.time_block != "auto":
        return chunks
    converted_grid = GridInfo(
        path=inventory.input_dir,
        lat=np.asarray(inventory.lat_values[selection.lat_start : selection.lat_stop], dtype="float64"),
        lon=np.asarray(inventory.lon_values[selection.lon_start : selection.lon_stop], dtype="float64"),
        lat_bounds=np.asarray(grid.lat_bounds[selection.lat_start : selection.lat_stop + 1], dtype="float64"),
        lon_bounds=np.asarray(grid.lon_bounds[selection.lon_start : selection.lon_stop + 1], dtype="float64"),
        lat_resolution=grid.lat_resolution,
        lon_resolution=grid.lon_resolution,
        lat_descending=grid.lat_descending,
        lon_descending=grid.lon_descending,
        lat_uniform=True,
        lon_uniform=True,
    )
    converted_info = DatasetInfo(
        path=inventory.input_dir,
        dimensions=dict(zip(("time", "lat", "lon"), selection.shape)),
        variables=tuple(
            VariableInfo(
                name=name,
                dims=("time", "lat", "lon"),
                shape=selection.shape,
                dtype=_converted_dtype(inventory, name, config),
                chunks=chunks,
                is_coord=False,
            )
            for name in selection.variables
        ),
        attrs={},
        zarr_format=3,
    )
    resolved_time = resolve_auto_time_block(
        converted_info,
        converted_grid,
        target,
        method=options.method,
        skipna=options.skipna,
        compute_dtype=options.compute_dtype,
    )
    return resolved_time, chunks[1], chunks[2]


def _zarr_inspection_id(info: DatasetInfo) -> str:
    digest = blake2b(digest_size=16)
    digest.update(str(info.path.resolve()).encode())
    metadata = info.path / "zarr.json"
    if metadata.is_file():
        stat = metadata.stat()
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def _zarr_target_info(info: DatasetInfo, resample_plan) -> DatasetInfo:
    dimensions = dict(info.dimensions)
    dimensions.update(resample_plan.target.dimensions)
    variables = []
    for item in info.variables:
        shape = tuple(dimensions.get(dim, size) for dim, size in zip(item.dims, item.shape))
        dtype = item.dtype
        if not item.is_coord and {"lat", "lon"}.issubset(item.dims):
            if not np.issubdtype(dtype, np.floating):
                dtype = np.dtype("float64")
            elif resample_plan.compute_dtype == "float32":
                dtype = np.dtype("float32")
        variables.append(
            VariableInfo(
                name=item.name,
                dims=item.dims,
                shape=shape,
                dtype=np.dtype(dtype),
                chunks=resample_plan.output_chunks.get(item.name, item.chunks),
                is_coord=item.is_coord,
                attrs=item.attrs,
                compressors=item.compressors,
            )
        )
    return DatasetInfo(info.path, dimensions, tuple(variables), info.attrs, info.zarr_format)


def _existing_chunks(info: DatasetInfo) -> tuple[int, int, int]:
    reference = next(
        (item for item in info.data_variables if set(item.dims) == {"time", "lat", "lon"}),
        None,
    )
    if reference is None:
        raise ValueError("输入 Zarr 没有可处理的 time/lat/lon 三维数据变量。")
    mapping = dict(zip(reference.dims, reference.chunks))
    return tuple(int(mapping[name]) for name in ("time", "lat", "lon"))


def _replacement_rules(config: PipelineConfig):
    options = config.resampling
    before = parse_replacement_rules(
        options.before_conditions,
        options.before_results,
    )
    after = parse_replacement_rules(
        options.after_conditions,
        options.after_results,
    )
    if (before.rules or after.rules) and not config.operations.resample:
        raise ValueError("采样前后替换规则仅在选择重采样时有效。")
    return before, after


def build_zarr_pipeline_plan(inspection, config: PipelineConfig) -> ZarrPipelinePlan:
    if getattr(inspection, "kind", None) != "zarr" or inspection.dataset_info is None:
        raise ValueError("现有 Zarr 流程要求已完成 Zarr 输入检查。")
    operations = config.operations
    before_replacements, after_replacements = _replacement_rules(config)
    if not (operations.resample or operations.rechunk or operations.recompress):
        raise ValueError("现有 Zarr 输入至少需要选择重采样、重分块或重压缩中的一项。")
    info = inspection.zarr_info
    resample_plan = None
    target_info = info
    if operations.resample:
        resample_inspection = inspect_resample_input(inspection.path)
        options = config.resampling
        resample_plan = plan_resample(
            ResampleConfig(
                input=inspection.path,
                output=config.general.output,
                resolution=options.resolution,
                method=options.method,
                skipna=options.skipna,
                na_thres=options.na_thres,
                compute_dtype=options.compute_dtype,
                extent="custom",
                target_lat_bounds=(config.general.lat_min, config.general.lat_max),
                target_lon_bounds=(config.general.lon_min, config.general.lon_max),
                target_lat_descending=True,
                target_lon_descending=False,
                tile_size=options.tile_size,
                time_block=options.time_block,
                compute_workers=options.compute_workers,
                space_workers=options.space_workers,
                temporary_dir=config.general.temporary_dir,
                before_replacements=before_replacements,
                after_replacements=after_replacements,
                statistics_policy=options.statistics_policy,
            ),
            resample_inspection,
        )
        target_info = _zarr_target_info(info, resample_plan)
    if operations.rechunk:
        chunk_plan = plan_chunks(
            target_info,
            config.chunking.strategy,
            target_mib=config.chunking.target_mib,
            workers=config.chunking.workers,
            custom_chunks=config.chunking.custom_chunks,
        )
    else:
        chunks = _existing_chunks(target_info)
        chunk_plan = ChunkPlan(
            strategy="custom",
            chunks=chunks,
            target_mib=0.0,
            estimated_chunk_bytes=0,
            estimated_chunks={},
            rationale=("未请求重分块，保持上游输出 chunks。",),
        )
    compression = (
        _pipeline_compression_plan(config)
        if operations.recompress
        else CompressionPlan(
            "none", None, "未请求重压缩，保持上游 codec。", codec="none"
        )
    )
    storage_requested = operations.rechunk or operations.recompress
    decisions = (
        OperationDecision("conversion", False, "not_requested", "现有 Zarr 输入跳过转换。"),
        OperationDecision(
            "resampling",
            operations.resample,
            "executed_as_stage" if operations.resample else "not_requested",
            "执行 Zarr 空间重采样。" if operations.resample else "用户未请求重采样。",
        ),
        OperationDecision(
            "rechunking",
            operations.rechunk,
            "executed_as_stage" if operations.rechunk else "not_requested",
            "由最终化阶段应用目标 chunks。" if operations.rechunk else "用户未请求重分块。",
        ),
        OperationDecision(
            "recompression",
            operations.recompress,
            "executed_as_stage" if operations.recompress else "not_requested",
            "由最终化阶段应用目标 codec。" if operations.recompress else "用户未请求重压缩。",
        ),
    )
    return ZarrPipelinePlan(
        inspection_id=_zarr_inspection_id(info),
        input_info=info,
        needs_resample=bool(operations.resample),
        resample_plan=resample_plan,
        final_chunk_plan=chunk_plan,
        final_compression=compression,
        final_chunks=chunk_plan.chunks,
        direct_finalization=not storage_requested,
        finalization_required=storage_requested,
        operation_decisions=decisions,
    )


def build_pipeline_plan(inspection, config: PipelineConfig) -> PipelinePlan | ZarrPipelinePlan:
    """Validate user intent and derive a write-minimizing execution plan."""

    inspection_kind = getattr(inspection, "kind", None)
    before_replacements, after_replacements = _replacement_rules(config)
    requested_kind = config.input.kind
    if requested_kind not in {"auto", "raw", "zarr"}:
        raise ValueError("input_kind 必须是 auto、raw 或 zarr。")
    if requested_kind == "raw" and inspection_kind != "source":
        raise ValueError("input_kind=raw 要求已完成原始数据检查。")
    if requested_kind == "zarr" or (requested_kind == "auto" and inspection_kind == "zarr"):
        return build_zarr_pipeline_plan(inspection, config)
    if inspection_kind != "source" or inspection.inventory is None:
        raise ValueError("一条龙模块必须使用已完成的数据检查结果。")
    inventory = inspection.source_inventory
    general = config.general
    operations = config.operations
    selected_names = config.conversion.variables or tuple(inventory.variables)
    unknown = sorted(set(selected_names) - set(inventory.variables))
    if unknown:
        raise ValueError("未知变量：" + ", ".join(unknown))
    unsupported = [
        name
        for name in selected_names
        if set(inventory.variables[name].dims) != {"time", "lat", "lon"}
        or len(inventory.variables[name].dims) != 3
    ]
    if unsupported:
        raise ValueError(
            "一条龙 v1.6.0 仅支持 time/lat/lon 三维数据变量："
            + ", ".join(unsupported)
        )
    if not (-90 <= general.lat_min < general.lat_max <= 90):
        raise ValueError("纬度范围必须满足 -90 <= min < max <= 90。")
    if not (-180 <= general.lon_min < general.lon_max <= 180):
        raise ValueError("经度范围必须在 -180..180 且严格递增。")

    time_bounds = (
        (general.time_start, general.time_end)
        if general.time_start is not None or general.time_end is not None
        else None
    )
    grid = _grid_info(inventory)
    same_grid = True
    if operations.resample:
        options = config.resampling
        if not np.isfinite(float(options.resolution)) or options.resolution <= 0:
            raise ValueError("选择重采样时，目标分辨率必须是正数。")
        if options.method not in {
            "bilinear",
            "conservative",
            "conservative_normed",
            "patch",
            "nearest_s2d",
            "nearest_d2s",
        }:
            raise ValueError("不支持的重采样方法。")
        if not 0 <= float(options.na_thres) <= 1:
            raise ValueError("na_thres 必须位于 0 到 1 之间。")
        target = build_target_grid(
            grid,
            options.resolution,
            extent="custom",
            lat_bounds=(general.lat_min, general.lat_max),
            lon_bounds=(general.lon_min, general.lon_max),
            lat_descending=True,
            lon_descending=False,
        )
        exact_lat = _exact_slice(grid.lat, target.lat) if grid.lat_descending else None
        exact_lon = _exact_slice(grid.lon, target.lon) if not grid.lon_descending else None
        same_grid = bool(
            np.isclose(grid.lat_resolution, options.resolution, rtol=1e-5, atol=1e-10)
            and np.isclose(grid.lon_resolution, options.resolution, rtol=1e-5, atol=1e-10)
            and exact_lat is not None
            and exact_lon is not None
        )
        if same_grid:
            lat_start, lat_stop = exact_lat
            lon_start, lon_stop = exact_lon
            lat_reason = lon_reason = "源网格与目标网格完全对齐，不增加 halo"
        else:
            lat_start, lat_stop, lat_reason = _axis_window(
                grid.lat, grid.lat_bounds, general.lat_min, general.lat_max, options.method
            )
            lon_start, lon_stop, lon_reason = _axis_window(
                grid.lon,
                grid.lon_bounds,
                general.lon_min,
                general.lon_max,
                options.method,
                periodic=grid.periodic,
            )
        lat_bounds = _axis_values_bounds(grid.lat, lat_start, lat_stop, grid.lat_bounds)
        lon_bounds = _axis_values_bounds(grid.lon, lon_start, lon_stop, grid.lon_bounds)
        source_selection = make_selection(
            inventory,
            time_bounds=time_bounds,
            lat_bounds=list(lat_bounds),
            lon_bounds=list(lon_bounds),
            variables=list(selected_names),
        )
        halo_description = f"纬度：{lat_reason}；经度：{lon_reason}"
    else:
        source_selection = make_selection(
            inventory,
            time_bounds=time_bounds,
            lat_bounds=[general.lat_min, general.lat_max],
            lon_bounds=[general.lon_min, general.lon_max],
            variables=list(selected_names),
        )
        lat_start, lat_stop = source_selection.lat_start, source_selection.lat_stop
        lon_start, lon_stop = source_selection.lon_start, source_selection.lon_stop
        lat_bounds = _axis_values_bounds(grid.lat, lat_start, lat_stop, grid.lat_bounds)
        lon_bounds = _axis_values_bounds(grid.lon, lon_start, lon_stop, grid.lon_bounds)
        target = _selected_source_grid(grid, source_selection)
        halo_description = "未请求重采样，按源网格格点裁剪，不增加 halo"

    if (source_selection.lat_start, source_selection.lat_stop) != (lat_start, lat_stop):
        raise ValueError("源纬度读取窗口无法映射为连续源索引。")
    if (source_selection.lon_start, source_selection.lon_stop) != (lon_start, lon_stop):
        raise ValueError("源经度读取窗口无法映射为连续源索引。")
    source_window = SourceReadWindow(
        lat_start=lat_start,
        lat_stop=lat_stop,
        lon_start=lon_start,
        lon_stop=lon_stop,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        method=config.resampling.method if operations.resample else "none",
        halo_description=halo_description,
    )

    has_replacements = bool(before_replacements.rules or after_replacements.rules)
    needs_resample = operations.resample and (not same_grid or has_replacements)
    if needs_resample:
        conversion_chunks = _resampling_conversion_chunks(
            inventory, source_selection, grid, target, config
        )
    else:
        conversion_chunks = resolve_conversion_plan(
            inventory,
            source_selection,
            Path(general.output).expanduser().resolve(),
            max_workers=config.conversion.max_workers,
            reserve_gib=config.conversion.reserve_memory_gib,
        ).chunks
    baseline_chunks = tuple(
        min(size, chunk)
        for size, chunk in zip(
            (source_selection.shape[0], target.lat.size, target.lon.size),
            conversion_chunks,
        )
    )

    source_west, source_east, source_south, source_north = grid.source_extent
    requested_west, requested_east = general.lon_min, general.lon_max
    requested_south, requested_north = general.lat_min, general.lat_max
    tolerance = max(
        1e-10,
        (config.resampling.resolution if operations.resample else grid.lat_resolution) * 1e-7,
    )
    outside = []
    if requested_south < source_south - tolerance:
        outside.append(f"南侧 {requested_south:g}° 以下")
    if requested_north > source_north + tolerance:
        outside.append(f"北侧 {requested_north:g}° 以上")
    if requested_west < source_west - tolerance:
        outside.append(f"西侧 {requested_west:g}° 以西")
    if requested_east > source_east + tolerance:
        outside.append(f"东侧 {requested_east:g}° 以东")
    coverage_warning = None
    if outside:
        consequence = (
            "未执行外推，源数据不存在的目标格点将保持缺测值。"
            if operations.resample
            else "未请求重采样，超出源网格的范围不会出现在输出中。"
        )
        coverage_warning = (
            "输出范围超出源数据覆盖范围（"
            + "、".join(outside)
            + "）；"
            + consequence
        )

    final_chunk_plan, final_compression = _final_layout(
        inventory,
        source_selection,
        target,
        config,
        needs_resample=needs_resample,
        baseline_chunks=baseline_chunks,
    )
    output_layout = _output_layout(
        inventory,
        source_selection,
        target,
        config,
        final_chunk_plan,
        final_compression,
        needs_resample=needs_resample,
    )
    terminal_label = "重采样阶段" if needs_resample else "转换阶段"
    direct_layout = all(
        inventory.variables[name].direct_compatible for name in selected_names
    )
    storage_requested = operations.rechunk or operations.recompress
    finalization_required = storage_requested and not direct_layout
    fused = "fused_into_resampling" if needs_resample else "fused_into_conversion"
    decisions = (
        OperationDecision(
            "conversion",
            True,
            "executed_as_stage",
            "原始数据必须转换为 Zarr。",
        ),
        OperationDecision(
            "resampling",
            operations.resample,
            (
                "executed_as_stage"
                if needs_resample
                else "satisfied_as_noop"
                if operations.resample
                else "not_requested"
            ),
            (
                "目标网格与源网格不同或包含采样替换规则，执行 xESMF 重采样。"
                if needs_resample
                else "目标网格与源网格等价，以恒等转换满足请求。"
                if operations.resample
                else "用户未请求重采样。"
            ),
        ),
        OperationDecision(
            "rechunking",
            operations.rechunk,
            (
                "executed_as_stage"
                if finalization_required and operations.rechunk
                else fused
                if operations.rechunk
                else "not_requested"
            ),
            (
                "终端写入器无法直接满足目标 chunks，使用最终化阶段。"
                if finalization_required and operations.rechunk
                else f"目标 chunks 已融合到{terminal_label}写出。"
                if operations.rechunk
                else "用户未请求重分块，使用标准 chunks。"
            ),
        ),
        OperationDecision(
            "recompression",
            operations.recompress,
            (
                "executed_as_stage"
                if finalization_required and operations.recompress
                else fused
                if operations.recompress
                else "not_requested"
            ),
            (
                "终端写入器无法直接满足目标 codec，使用最终化阶段。"
                if finalization_required and operations.recompress
                else f"目标 codec 已融合到{terminal_label}写出。"
                if operations.recompress
                else "用户未请求重压缩，使用转换基线 codec。"
            ),
        ),
    )
    return PipelinePlan(
        inspection_id=_inventory_id(inventory),
        target_grid=target,
        source_read_window=source_window,
        source_selection=source_selection,
        needs_resample=needs_resample,
        coverage_warning=coverage_warning,
        conversion_chunks=conversion_chunks,
        final_chunks=final_chunk_plan.chunks,
        final_chunk_plan=final_chunk_plan,
        final_compression=final_compression,
        output_layout=output_layout,
        direct_finalization=not finalization_required,
        finalization_required=finalization_required,
        operation_decisions=decisions,
    )
