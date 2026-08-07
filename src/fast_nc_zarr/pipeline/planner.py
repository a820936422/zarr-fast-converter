from __future__ import annotations

from hashlib import blake2b
from pathlib import Path

import numpy as np

from ..models import CodecSpec, Inventory, OutputLayout, VariableOutputLayout
from ..rechunking.compression import make_compression_plan
from ..resampling.grid import GridInfo, _axis_bounds, build_target_grid
from ..resampling.autotune import resolve_auto_time_block
from ..rechunking.models import DatasetInfo, VariableInfo
from ..rechunking.planning import plan_chunks
from .models import PipelineConfig, PipelinePlan, SourceReadWindow
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


def _final_layout(inventory: Inventory, selection, target, config, *, needs_resample: bool):
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
    chunk_plan = plan_chunks(
        info,
        config.finalization.strategy,
        target_mib=config.finalization.target_mib,
        workers=config.finalization.workers,
        custom_chunks=config.finalization.custom_chunks,
    )
    return chunk_plan, make_compression_plan(config.finalization.compression)


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
    if np.issubdtype(dtype, np.integer):
        shuffle = "bitshuffle"
    elif np.issubdtype(dtype, np.floating):
        shuffle = "shuffle"
    else:
        shuffle = "noshuffle"
    return CodecSpec(
        "blosc",
        level=int(compression.level or 1),
        cname="zstd",
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
    coordinate_codec = (
        CodecSpec("zstd", level=1) if compression.enabled else None
    )
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
    return OutputLayout(tuple(layouts))


def build_pipeline_plan(inspection, config: PipelineConfig) -> PipelinePlan:
    """Validate user intent and derive the source window/target grid."""

    if getattr(inspection, "kind", None) != "source" or inspection.inventory is None:
        raise ValueError("一条龙模块必须使用已完成的数据检查结果。")
    inventory = inspection.source_inventory
    general = config.general
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
            "一条龙 V1 仅支持 time/lat/lon 三维数据变量："
            + ", ".join(unsupported)
        )
    if not np.isfinite(float(general.resolution)) or general.resolution <= 0:
        raise ValueError("目标分辨率必须是正数。")
    if not (-90 <= general.lat_min < general.lat_max <= 90):
        raise ValueError("目标纬度范围必须满足 -90 <= min < max <= 90。")
    if not (-180 <= general.lon_min < general.lon_max <= 180):
        raise ValueError("V1 目标经度范围必须在 -180..180 且严格递增。")
    if config.resampling.method not in {
        "bilinear", "conservative", "conservative_normed", "patch", "nearest_s2d", "nearest_d2s"
    }:
        raise ValueError("不支持的重采样方法。")
    if not 0 <= float(config.resampling.na_thres) <= 1:
        raise ValueError("na_thres 必须位于 0 到 1 之间。")

    grid = _grid_info(inventory)
    target = build_target_grid(
        grid,
        general.resolution,
        extent="custom",
        lat_bounds=(general.lat_min, general.lat_max),
        lon_bounds=(general.lon_min, general.lon_max),
        # Final V1 canonical orientation.
        lat_descending=True,
        lon_descending=False,
    )
    method = config.resampling.method
    exact_lat = _exact_slice(grid.lat, target.lat) if grid.lat_descending else None
    exact_lon = _exact_slice(grid.lon, target.lon) if not grid.lon_descending else None
    if (
        np.isclose(grid.lat_resolution, general.resolution, rtol=1e-5, atol=1e-10)
        and np.isclose(grid.lon_resolution, general.resolution, rtol=1e-5, atol=1e-10)
        and exact_lat is not None
        and exact_lon is not None
    ):
        lat_start, lat_stop = exact_lat
        lon_start, lon_stop = exact_lon
        lat_reason = "源网格与目标网格完全对齐，不增加 halo"
        lon_reason = "源网格与目标网格完全对齐，不增加 halo"
    else:
        lat_start, lat_stop, lat_reason = _axis_window(
            grid.lat,
            grid.lat_bounds,
            general.lat_min,
            general.lat_max,
            method,
        )
        lon_start, lon_stop, lon_reason = _axis_window(
            grid.lon,
            grid.lon_bounds,
            general.lon_min,
            general.lon_max,
            method,
            periodic=grid.periodic,
        )
    lat_bounds = _axis_values_bounds(grid.lat, lat_start, lat_stop, grid.lat_bounds)
    lon_bounds = _axis_values_bounds(grid.lon, lon_start, lon_stop, grid.lon_bounds)
    source_window = SourceReadWindow(
        lat_start=lat_start,
        lat_stop=lat_stop,
        lon_start=lon_start,
        lon_stop=lon_stop,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        method=method,
        halo_description=f"纬度：{lat_reason}；经度：{lon_reason}",
    )
    source_selection = make_selection(
        inventory,
        time_bounds=(general.time_start, general.time_end)
        if general.time_start is not None or general.time_end is not None
        else None,
        lat_bounds=list(lat_bounds),
        lon_bounds=list(lon_bounds),
        variables=list(config.conversion.variables) or None,
    )
    # Selection bounds are inclusive and may include duplicate edge values
    # due to floating point display.  Assert that the selected index window is
    # exactly the planned source window.
    if (source_selection.lat_start, source_selection.lat_stop) != (lat_start, lat_stop):
        raise ValueError("源纬度读取窗口无法映射为连续源索引。")
    if (source_selection.lon_start, source_selection.lon_stop) != (lon_start, lon_stop):
        raise ValueError("源经度读取窗口无法映射为连续源索引。")

    same_resolution = np.isclose(grid.lat_resolution, general.resolution, rtol=1e-5, atol=1e-10) and np.isclose(
        grid.lon_resolution, general.resolution, rtol=1e-5, atol=1e-10
    )
    same_orientation = grid.lat_descending and not grid.lon_descending
    same_grid = (
        same_resolution
        and same_orientation
        and exact_lat is not None
        and exact_lon is not None
    )
    conversion_chunks = None
    if not same_grid:
        requested_time = config.resampling.time_block
        if requested_time == "auto":
            # GeoTIFF products normally have one time point per file.  Their
            # file cadence must not force the downstream xESMF engine into a
            # one-step-at-a-time loop.  Keep the intermediate time chunk at a
            # bounded vectorization batch; the resampling autotuner applies a
            # second memory check before execution.
            requested_time = min(64, source_selection.shape[0])
        requested_tile = config.resampling.tile_size
        if requested_tile == "auto":
            # Two source chunks per side feed the 512-cell default resampling
            # macro tile.  This keeps individual conversion writes moderate
            # while still making the later source-window reads chunk-aligned.
            requested_tile = 256
        stencil = 2 if method in {"bilinear", "patch"} else 1
        source_lat_chunk = int(
            np.ceil(float(requested_tile) * general.resolution / grid.lat_resolution)
        ) + stencil
        source_lon_chunk = int(
            np.ceil(float(requested_tile) * general.resolution / grid.lon_resolution)
        ) + stencil
        conversion_chunks = (
            min(int(requested_time), source_selection.shape[0]),
            min(max(1, source_lat_chunk), source_selection.shape[1]),
            min(max(1, source_lon_chunk), source_selection.shape[2]),
        )
        if config.resampling.time_block == "auto":
            # Run the same bounded time-batch model before conversion.  This
            # makes source-crop.zarr's time chunk equal to the batch that the
            # resampling engine will actually use, rather than first writing
            # an optimistic chunk and then creating an avoidable merge store.
            converted_grid = GridInfo(
                path=inventory.input_dir,
                lat=np.asarray(
                    inventory.lat_values[
                        source_selection.lat_start : source_selection.lat_stop
                    ],
                    dtype="float64",
                ),
                lon=np.asarray(
                    inventory.lon_values[
                        source_selection.lon_start : source_selection.lon_stop
                    ],
                    dtype="float64",
                ),
                lat_bounds=np.asarray(
                    grid.lat_bounds[
                        source_selection.lat_start : source_selection.lat_stop + 1
                    ],
                    dtype="float64",
                ),
                lon_bounds=np.asarray(
                    grid.lon_bounds[
                        source_selection.lon_start : source_selection.lon_stop + 1
                    ],
                    dtype="float64",
                ),
                lat_resolution=grid.lat_resolution,
                lon_resolution=grid.lon_resolution,
                lat_descending=grid.lat_descending,
                lon_descending=grid.lon_descending,
                lat_uniform=True,
                lon_uniform=True,
            )
            converted_info = DatasetInfo(
                path=inventory.input_dir,
                dimensions={
                    "time": source_selection.shape[0],
                    "lat": source_selection.shape[1],
                    "lon": source_selection.shape[2],
                },
                variables=tuple(
                    VariableInfo(
                        name=name,
                        dims=("time", "lat", "lon"),
                        shape=source_selection.shape,
                        dtype=_converted_dtype(inventory, name, config),
                        chunks=conversion_chunks,
                        is_coord=False,
                    )
                    for name in selected_names
                ),
                attrs={},
                zarr_format=3,
            )
            resolved_time = resolve_auto_time_block(
                converted_info,
                converted_grid,
                target,
                method=method,
                skipna=config.resampling.skipna,
                compute_dtype=config.resampling.compute_dtype,
            )
            conversion_chunks = (
                resolved_time,
                conversion_chunks[1],
                conversion_chunks[2],
            )
    source_west, source_east, source_south, source_north = grid.source_extent
    target_west, target_east, target_south, target_north = target.spatial_extent
    outside = []
    tolerance = max(1e-10, general.resolution * 1e-7)
    if target_south < source_south - tolerance:
        outside.append(f"南侧 {target_south:g}° 以下")
    if target_north > source_north + tolerance:
        outside.append(f"北侧 {target_north:g}° 以上")
    if target_west < source_west - tolerance:
        outside.append(f"西侧 {target_west:g}° 以西")
    if target_east > source_east + tolerance:
        outside.append(f"东侧 {target_east:g}° 以东")
    coverage_warning = None
    if outside:
        coverage_warning = (
            "目标范围超出源数据覆盖范围（" + "、".join(outside) + "）；"
            "未执行外推，源数据不存在的目标格点将保持缺测值。"
        )
    final_chunk_plan, final_compression = _final_layout(
        inventory,
        source_selection,
        target,
        config,
        needs_resample=not same_grid,
    )
    output_layout = _output_layout(
        inventory,
        source_selection,
        target,
        config,
        final_chunk_plan,
        final_compression,
        needs_resample=not same_grid,
    )
    direct_finalization = all(
        inventory.variables[name].direct_compatible for name in selected_names
    )
    if direct_finalization and same_grid:
        conversion_chunks = final_chunk_plan.chunks
    return PipelinePlan(
        inspection_id=_inventory_id(inventory),
        target_grid=target,
        source_read_window=source_window,
        source_selection=source_selection,
        needs_resample=not same_grid,
        coverage_warning=coverage_warning,
        conversion_chunks=conversion_chunks,
        final_chunks=final_chunk_plan.chunks,
        final_chunk_plan=final_chunk_plan,
        final_compression=final_compression,
        output_layout=output_layout,
        direct_finalization=direct_finalization,
    )
