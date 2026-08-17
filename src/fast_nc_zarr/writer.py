from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterator

import numpy as np

from .metadata import sanitize_cf_references
from .models import (
    CodecSpec,
    ConversionPlan,
    Inventory,
    OutputLayout,
    Selection,
    VariableTransform,
)
from .selection import selected_output_logical_bytes
from .runtime import bounded_process_map

_OUTPUT_GROUP = None
_SOURCE_CACHE = None
_SOURCE_CACHE_LIMIT = 4
_SOURCE_CACHE_HITS = 0
_SOURCE_OPENS = 0
_SOURCE_FINALIZER = None
_SOURCE_ENGINE = "netcdf4"

SOURCE_CACHE_HARD_LIMIT = 64
_SOURCE_FD_RESERVE = 32
TASK_BATCH_HARD_LIMIT = 64


def _effective_task_batch(value: int) -> int:
    return max(1, min(TASK_BATCH_HARD_LIMIT, int(value)))



def source_cache_limit(
    chunk_time: int,
    workers: int,
    *,
    task_batch: int = 1,
    hard_limit: int = SOURCE_CACHE_HARD_LIMIT,
    open_file_limit: int | None = None,
) -> int:
    """Choose a conservative per-worker source-handle LRU capacity."""

    worker_count = max(1, int(workers))
    hard_limit = max(1, int(hard_limit))
    desired = max(1, int(chunk_time)) * max(1, int(task_batch))
    if open_file_limit is None:
        try:
            import resource

            soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft_limit != resource.RLIM_INFINITY and int(soft_limit) >= 0:
                open_file_limit = int(soft_limit)
        except (ImportError, OSError, ValueError):
            open_file_limit = None
    descriptor_limit = hard_limit
    if open_file_limit is not None:
        shared_budget = max(1, int(open_file_limit) - _SOURCE_FD_RESERVE)
        descriptor_limit = max(1, shared_budget // worker_count)
    return max(1, min(hard_limit, desired, descriptor_limit))


def _close_source(source) -> None:
    close = getattr(source, "close", None)
    if close is not None:
        close()


def _close_worker_sources() -> None:
    """Close every retained source handle before a worker exits or is reused."""

    global _OUTPUT_GROUP, _SOURCE_CACHE
    cache = _SOURCE_CACHE
    _SOURCE_CACHE = None
    if cache is not None:
        while cache:
            _, source = cache.popitem(last=False)
            try:
                _close_source(source)
            except Exception:
                # Descriptor cleanup is best effort during process teardown;
                # continue closing the remaining retained sources.
                pass
    _OUTPUT_GROUP = None


def _worker_init(
    output: str,
    source_engine: str = "netcdf4",
    cache_limit: int = 4,
) -> None:
    global _OUTPUT_GROUP, _SOURCE_CACHE, _SOURCE_CACHE_LIMIT
    global _SOURCE_CACHE_HITS, _SOURCE_OPENS, _SOURCE_FINALIZER, _SOURCE_ENGINE
    from collections import OrderedDict
    from multiprocessing.util import Finalize

    import zarr

    if _SOURCE_FINALIZER is not None:
        _SOURCE_FINALIZER.cancel()
    _close_worker_sources()
    _OUTPUT_GROUP = zarr.open_group(output, mode="r+")
    _SOURCE_CACHE = OrderedDict()
    _SOURCE_CACHE_LIMIT = max(1, min(SOURCE_CACHE_HARD_LIMIT, int(cache_limit)))
    _SOURCE_CACHE_HITS = 0
    _SOURCE_OPENS = 0
    _SOURCE_ENGINE = source_engine
    _SOURCE_FINALIZER = Finalize(
        None, _close_worker_sources, exitpriority=10
    )


def _source(path: str):
    global _SOURCE_CACHE, _SOURCE_CACHE_HITS, _SOURCE_OPENS
    existing = _SOURCE_CACHE.get(path)
    if existing is not None:
        _SOURCE_CACHE_HITS += 1
        _SOURCE_CACHE.move_to_end(path)
        return existing
    if _SOURCE_ENGINE == "netcdf4":
        import netCDF4

        source = netCDF4.Dataset(path, mode="r")
        source.set_auto_mask(False)
        source.set_auto_scale(False)
    else:
        import xarray as xr

        source = xr.open_dataset(
            path,
            engine=_SOURCE_ENGINE,
            chunks=None,
            decode_times=False,
            mask_and_scale=False,
        )
    _SOURCE_OPENS += 1
    _SOURCE_CACHE[path] = source
    while len(_SOURCE_CACHE) > _SOURCE_CACHE_LIMIT:
        _, stale = _SOURCE_CACHE.popitem(last=False)
        _close_source(stale)
    return source


def _read_segment(task: ChunkTask, segment: SourceSegment) -> np.ndarray:
    source = _source(segment.path)
    selectors = []
    for dim in task.dims:
        if dim == "time":
            selectors.append(slice(segment.local_start, segment.local_stop))
        elif dim == "lat":
            selectors.append(slice(*task.source_lat))
        elif dim == "lon":
            selectors.append(slice(*task.source_lon))
        else:
            raise RuntimeError(f"直接写入不支持维度 {dim}")
    return np.asarray(source.variables[task.variable][tuple(selectors)])

def _apply_transform(
    data: np.ndarray,
    fill_values: tuple[float, ...] | None,
    scale_factor: float | None,
    add_offset: float | None,
    output_fill: float | int | None,
    output_dtype: str,
) -> np.ndarray:
    raw = np.asarray(data)
    target_dtype = np.dtype(output_dtype)
    needs_mask = bool(fill_values)
    needs_cast = raw.dtype != target_dtype
    needs_scale = scale_factor is not None
    needs_offset = add_offset is not None
    if not needs_mask and not needs_cast and not needs_scale and not needs_offset:
        return raw if raw.flags.c_contiguous else np.ascontiguousarray(raw)

    mask = np.zeros(raw.shape, dtype=bool) if needs_mask else None
    if mask is not None:
        for value in fill_values:
            try:
                if np.isnan(value) and np.issubdtype(raw.dtype, np.floating):
                    mask |= np.isnan(raw)
                    continue
            except TypeError:
                pass
            mask |= raw == value
    result = raw.astype(
        target_dtype,
        copy=needs_cast or needs_scale or needs_offset or mask is not None,
    )
    if needs_scale:
        if mask is not None and mask.any():
            result[~mask] *= scale_factor
        else:
            result *= scale_factor
    if needs_offset:
        if mask is not None and mask.any():
            result[~mask] += add_offset
        else:
            result += add_offset
    if mask is not None and mask.any():
        if output_fill is None:
            raise ValueError("缺少整型变量的输出缺失值。")
        result[mask] = output_fill
    return result if result.flags.c_contiguous else np.ascontiguousarray(result)


def progress_line(
    completed: int,
    total: int,
    logical_bytes: int,
    elapsed: float,
    *,
    prefix: str = "进度",
    cpu: float | None = None,
    rss: int | None = None,
) -> str:
    """Render a compact live progress bar for production writes."""
    total = max(1, total)
    ratio = min(1.0, max(0.0, completed / total))
    width = 24
    filled = int(round(width * ratio))
    bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
    speed = logical_bytes / max(elapsed, 1e-9) / 1024**2
    resource = ""
    if cpu is not None:
        resource += f" | CPU {cpu:5.0f}%"
    if rss is not None:
        resource += f" | RSS {rss / 1024**3:.2f} GiB"
    return f"\r{prefix} [{bar}] {ratio:6.1%} ({completed}/{total}) | {speed:7.1f} MiB/s{resource}"


@dataclass(frozen=True)
class SourceSegment:
    path: str
    local_start: int
    local_stop: int
    destination_start: int
    destination_stop: int


@dataclass(frozen=True)
class ChunkTask:
    variable: str
    dims: tuple[str, ...]
    output_dims: tuple[str, ...]
    dtype: str
    output_ranges: tuple[tuple[int, int], ...]
    source_lat: tuple[int, int]
    source_lon: tuple[int, int]
    segments: tuple[SourceSegment, ...]
    fill_values: tuple[float, ...] | None = None
    scale_factor: float | None = None
    add_offset: float | None = None
    output_fill: float | int | None = None
    source_dtype: str | None = None
    output_variable: str | None = None
    reverse_lat: bool = False
    reverse_lon: bool = False


@dataclass(frozen=True)
class WriteBatchResult:
    logical_bytes: int
    chunks: int
    source_opens: int
    source_cache_hits: int


def make_compressor(name: str, level: int, shuffle: str):
    if name == "none":
        return None
    from zarr.codecs import BloscCodec

    return BloscCodec(cname=name, clevel=level, shuffle=shuffle)


def compressor_from_spec(spec: CodecSpec | None):
    if spec is None:
        return None
    if spec.kind == "zstd":
        from zarr.codecs import ZstdCodec

        return ZstdCodec(level=spec.level)
    if spec.kind == "gzip":
        from zarr.codecs import GzipCodec

        return GzipCodec(level=spec.level)
    return make_compressor(spec.cname or "zstd", spec.level, spec.shuffle)


def _chunk_for_dims(dims: tuple[str, ...], plan: ConversionPlan, sizes: dict[str, int]) -> tuple[int, ...]:
    chunks = {"time": plan.chunk_time, "lat": plan.chunk_lat, "lon": plan.chunk_lon}
    return tuple(min(sizes[dim], chunks.get(dim, sizes[dim])) for dim in dims)


def initialize_zarr(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    plan: ConversionPlan,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
) -> None:
    import dask.array as da
    import xarray as xr

    output = output.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"输出路径存在但不是目录：{output}")
        if any(output.iterdir()):
            raise FileExistsError(f"输出目录非空：{output}；请显式使用覆盖选项。")
    output.mkdir(parents=True, exist_ok=True)
    with xr.open_dataset(
        inventory.reference_file,
        engine=inventory.source_engine,
        chunks=None,
        decode_times=True,
        mask_and_scale=False,
    ) as source:
        source_dimensions = dict(
            zip(("time", "lat", "lon"), inventory.source_dimensions)
        )
        sizes = {
            "time": selection.shape[0],
            "lat": selection.shape[1],
            "lon": selection.shape[2],
        }
        variable_transforms = variable_transforms or {}
        variable_names = variable_names or {}
        variables = {}
        encoding = {}
        default_compressor = make_compressor(
            plan.compression, plan.compression_level, plan.shuffle
        )
        for name in selection.variables:
            original = source[name]
            output_name = variable_names.get(name, name)
            source_dims = inventory.variables[name].dims
            canonical_dims = (
                ("time", "lat", "lon")
                if set(source_dims) == {"time", "lat", "lon"}
                else source_dims
            )
            transform = variable_transforms.get(name)
            output_dtype = np.dtype(inventory.variables[name].dtype)
            if (
                transform is not None
                and (transform.scale_factor is not None or transform.add_offset is not None)
                and output_dtype.kind not in "fc"
            ):
                output_dtype = np.dtype("float32" if output_dtype.itemsize <= 4 else "float64")
            output_fill = None
            if transform is not None and transform.output_fill is not None:
                output_fill = transform.output_fill
            elif output_dtype.kind in "fc":
                output_fill = float("nan")
            elif transform is not None and transform.fill_values:
                output_fill = transform.fill_values[0]
            else:
                for key in ("_FillValue", "missing_value"):
                    value = inventory.variables[name].attrs.get(key)
                    if isinstance(value, (list, tuple, np.ndarray)):
                        value = value[0] if len(value) else None
                    if isinstance(value, (int, float, np.number)):
                        output_fill = value
                        break
            if transform is not None and transform.fill_values and output_fill is None and output_dtype.kind not in "fc":
                raise ValueError(f"整型变量 {name} 没有可用的输出缺失值。")
            if output_dtype.kind in "iu" and output_fill is not None:
                try:
                    if float(output_fill) != float(int(output_fill)):
                        raise ValueError(
                            f"整型变量 {name} 的输出缺失值 {output_fill!r} 不是整数。"
                        )
                except (TypeError, ValueError, OverflowError) as exc:
                    if isinstance(exc, ValueError) and str(exc).startswith("整型变量"):
                        raise
                    raise ValueError(f"整型变量 {name} 的输出缺失值无效：{output_fill!r}") from exc
                limits = np.iinfo(output_dtype)
                if output_fill < limits.min or output_fill > limits.max:
                    raise ValueError(
                        f"整型变量 {name} 的输出缺失值 {output_fill!r} 超出 {output_dtype} 可表示范围。"
                    )
            attrs = original.attrs.copy()
            if transform is not None and transform.fill_values:
                attrs["_FillValue"] = output_fill
                attrs["missing_value"] = output_fill
            if transform is not None and transform.scale_factor is not None:
                attrs["source_scale_factor"] = transform.scale_factor
            if transform is not None and transform.add_offset is not None:
                attrs["source_add_offset"] = transform.add_offset
            if transform is not None and (transform.scale_factor is not None or transform.add_offset is not None):
                attrs.pop("scale_factor", None)
                attrs.pop("add_offset", None)
            shape = tuple(
                sizes.get(dim, int(original.sizes[source_dimensions.get(dim, dim)]))
                for dim in canonical_dims
            )
            layout_item = output_layout.for_source(name) if output_layout else None
            if layout_item is not None:
                if layout_item.output_name != output_name:
                    raise ValueError(f"变量 {name} 的输出名称与最终布局不一致。")
                if layout_item.dims != canonical_dims or layout_item.shape != shape:
                    raise ValueError(f"变量 {name} 的 shape/dims 与最终布局不一致。")
                if np.dtype(layout_item.dtype) != output_dtype:
                    raise ValueError(f"变量 {name} 的 dtype 与最终布局不一致。")
                chunks = layout_item.chunks
                compressor = compressor_from_spec(layout_item.codec)
            else:
                chunks = _chunk_for_dims(canonical_dims, plan, sizes)
                compressor = default_compressor
            variables[output_name] = xr.Variable(
                canonical_dims,
                da.empty(shape, chunks=chunks, dtype=output_dtype),
                attrs=attrs,
            )
            encoding[output_name] = {"chunks": chunks}
            if compressor is not None:
                encoding[output_name]["compressors"] = [compressor]

        source_time_attrs = source[source_dimensions["time"]].attrs.copy()
        source_time_units = source_time_attrs.pop("units", None)
        source_time_calendar = source_time_attrs.pop("calendar", None)
        if source_time_units is not None:
            source_time_attrs["source_time_units"] = source_time_units
        if source_time_calendar is not None:
            source_time_attrs["source_time_calendar"] = source_time_calendar
        lat_values = inventory.lat_values[selection.lat_start : selection.lat_stop]
        lon_values = inventory.lon_values[selection.lon_start : selection.lon_stop]
        if output_layout is not None and "lat" in output_layout.axis_reversals:
            lat_values = lat_values[::-1]
        if output_layout is not None and "lon" in output_layout.axis_reversals:
            lon_values = lon_values[::-1]
        coordinates = {
            "time": xr.Variable(
                ("time",),
                inventory.times[selection.time_start : selection.time_stop],
                attrs=source_time_attrs,
            ),
            "lat": xr.Variable(
                ("lat",),
                lat_values,
                attrs=source[source_dimensions["lat"]].attrs.copy(),
            ),
            "lon": xr.Variable(
                ("lon",),
                lon_values,
                attrs=source[source_dimensions["lon"]].attrs.copy(),
            ),
        }
        for name, coordinate in coordinates.items():
            coordinate_layout = output_layout.for_output(name) if output_layout else None
            coordinate_chunks = (
                coordinate_layout.chunks
                if coordinate_layout is not None
                else (min(len(coordinate), 8192),)
            )
            encoding[name] = {"chunks": coordinate_chunks}
            if coordinate_layout is not None:
                coordinate_compressor = compressor_from_spec(coordinate_layout.codec)
                if coordinate_compressor is not None:
                    encoding[name]["compressors"] = [coordinate_compressor]
        template = sanitize_cf_references(
            xr.Dataset(variables, coords=coordinates, attrs=source.attrs.copy()),
            renames=variable_names,
        )
        delayed = template.to_zarr(
            output,
            mode="w",
            encoding=encoding,
            consolidated=False,
            zarr_format=3,
            compute=False,
            align_chunks=False,
        )
        del delayed
        template.close()




def _write_chunk(task: ChunkTask) -> int:
    time_axis = task.output_dims.index("time")
    shape = tuple(stop - start for start, stop in task.output_ranges)

    def output_order(data):
        if task.dims == task.output_dims:
            return data
        axes = tuple(task.dims.index(dim) for dim in task.output_dims)
        return np.transpose(data, axes=axes)

    if len(task.segments) == 1:
        data = output_order(_read_segment(task, task.segments[0]))
    else:
        data = np.empty(shape, dtype=np.dtype(task.source_dtype or task.dtype))
        for segment in task.segments:
            destination = [slice(None)] * len(shape)
            destination[time_axis] = slice(segment.destination_start, segment.destination_stop)
            data[tuple(destination)] = output_order(_read_segment(task, segment))
    if task.reverse_lat and "lat" in task.output_dims:
        data = np.flip(data, axis=task.output_dims.index("lat"))
    if task.reverse_lon and "lon" in task.output_dims:
        data = np.flip(data, axis=task.output_dims.index("lon"))
    data = _apply_transform(
        data,
        task.fill_values,
        task.scale_factor,
        task.add_offset,
        task.output_fill,
        task.dtype,
    )
    output_selection = tuple(slice(start, stop) for start, stop in task.output_ranges)
    _OUTPUT_GROUP[task.output_variable or task.variable][output_selection] = data
    return int(data.nbytes)


def _write_batch(tasks: tuple[ChunkTask, ...]) -> WriteBatchResult:
    opens_before = _SOURCE_OPENS
    hits_before = _SOURCE_CACHE_HITS
    logical_bytes = sum(_write_chunk(task) for task in tasks)
    return WriteBatchResult(
        logical_bytes=logical_bytes,
        chunks=len(tasks),
        source_opens=_SOURCE_OPENS - opens_before,
        source_cache_hits=_SOURCE_CACHE_HITS - hits_before,
    )


def _time_lookup(inventory: Inventory, selection: Selection) -> list[tuple[str, int]]:
    mapping = {}
    for record in inventory.files:
        for local_index, key in enumerate(record.time_keys):
            mapping[key] = (str(record.path), local_index)
    selected = inventory.time_keys[selection.time_start : selection.time_stop]
    missing = [key for key in selected if key not in mapping]
    if missing:
        raise RuntimeError(f"{len(missing)} 个所选时间点无法映射到源文件。")
    return [mapping[key] for key in selected]


def _segments(lookup: list[tuple[str, int]], start: int, stop: int) -> tuple[SourceSegment, ...]:
    result = []
    cursor = start
    destination = 0
    while cursor < stop:
        path, local_start = lookup[cursor]
        following = cursor + 1
        local_stop = local_start + 1
        while following < stop:
            next_path, next_local = lookup[following]
            if next_path != path or next_local != local_stop:
                break
            following += 1
            local_stop += 1
        length = following - cursor
        result.append(SourceSegment(path, local_start, local_stop, destination, destination + length))
        destination += length
        cursor = following
    return tuple(result)


def _chunk_task(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    lookup: list[tuple[str, int]],
    name: str,
    t0: int,
    t1: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
) -> ChunkTask:
    spec = inventory.variables[name]
    output_dims = (
        ("time", "lat", "lon")
        if set(spec.dims) == {"time", "lat", "lon"}
        else spec.dims
    )
    ranges_by_dim = {"time": (t0, t1), "lat": (y0, y1), "lon": (x0, x1)}
    transform = (variable_transforms or {}).get(name)
    output_dtype = spec.dtype
    if (
        transform is not None
        and (transform.scale_factor is not None or transform.add_offset is not None)
        and np.dtype(output_dtype).kind not in "fc"
    ):
        output_dtype = "float32" if np.dtype(output_dtype).itemsize <= 4 else "float64"
    output_fill = None
    if transform is not None and transform.output_fill is not None:
        output_fill = transform.output_fill
    elif np.dtype(output_dtype).kind in "fc":
        output_fill = float("nan")
    elif transform is not None and transform.fill_values:
        output_fill = transform.fill_values[0]
    if output_fill is None:
        for key in ("_FillValue", "missing_value"):
            value = spec.attrs.get(key)
            if isinstance(value, (list, tuple, np.ndarray)):
                value = value[0] if len(value) else None
            if isinstance(value, (int, float, np.number)):
                output_fill = value
                break
    reverse_lat = output_layout is not None and "lat" in output_layout.axis_reversals
    reverse_lon = output_layout is not None and "lon" in output_layout.axis_reversals
    ny = selection.shape[1]
    nx = selection.shape[2]
    source_y0, source_y1 = (ny - y1, ny - y0) if reverse_lat else (y0, y1)
    source_x0, source_x1 = (nx - x1, nx - x0) if reverse_lon else (x0, x1)
    return ChunkTask(
        variable=name,
        output_variable=(variable_names or {}).get(name, name),
        dims=spec.dims,
        output_dims=output_dims,
        dtype=output_dtype,
        output_ranges=tuple(ranges_by_dim[dim] for dim in output_dims),
        source_lat=(selection.lat_start + source_y0, selection.lat_start + source_y1),
        source_lon=(selection.lon_start + source_x0, selection.lon_start + source_x1),
        segments=_segments(lookup, t0, t1),
        fill_values=transform.fill_values if transform is not None else None,
        scale_factor=transform.scale_factor if transform is not None else None,
        add_offset=transform.add_offset if transform is not None else None,
        output_fill=output_fill,
        source_dtype=spec.dtype,
        reverse_lat=reverse_lat,
        reverse_lon=reverse_lon,
    )


def _variable_chunks(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    name: str,
    output_layout: OutputLayout | None,
) -> tuple[int, int, int]:
    if output_layout is None:
        chunks_by_dim = {
            "time": plan.chunk_time,
            "lat": plan.chunk_lat,
            "lon": plan.chunk_lon,
        }
    else:
        item = output_layout.for_source(name)
        chunks_by_dim = dict(zip(item.dims, item.chunks))
    try:
        chunks = tuple(
            min(size, max(1, int(chunks_by_dim[dim])))
            for dim, size in zip(("time", "lat", "lon"), selection.shape)
        )
    except KeyError as exc:
        raise ValueError(
            f"变量 {name} 的直接写入布局缺少 {exc.args[0]} 维度。"
        ) from exc
    return chunks


def _task_batches(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
    *,
    batch_size: int | None = None,
) -> Iterator[tuple[ChunkTask, ...]]:
    """Keep one variable/time slab local while assigning whole physical chunks."""

    nt, ny, nx = selection.shape
    lookup = _time_lookup(inventory, selection)
    limit = _effective_task_batch(
        plan.task_batch if batch_size is None else batch_size
    )
    for name in selection.variables:
        chunk_time, chunk_lat, chunk_lon = _variable_chunks(
            inventory, selection, plan, name, output_layout
        )
        for t0 in range(0, nt, chunk_time):
            t1 = min(nt, t0 + chunk_time)
            grouped: list[ChunkTask] = []
            for y0 in range(0, ny, chunk_lat):
                y1 = min(ny, y0 + chunk_lat)
                for x0 in range(0, nx, chunk_lon):
                    x1 = min(nx, x0 + chunk_lon)
                    grouped.append(
                        _chunk_task(
                            inventory,
                            selection,
                            plan,
                            lookup,
                            name,
                            t0,
                            t1,
                            y0,
                            y1,
                            x0,
                            x1,
                            variable_transforms,
                            variable_names,
                            output_layout,
                        )
                    )
                    if len(grouped) == limit:
                        yield tuple(grouped)
                        grouped.clear()
            if grouped:
                yield tuple(grouped)


def chunk_tasks(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
) -> Iterator[ChunkTask]:
    for batch in _task_batches(
        inventory,
        selection,
        plan,
        variable_transforms,
        variable_names,
        output_layout,
        batch_size=1,
    ):
        yield batch[0]


def chunk_task_batches(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
) -> Iterator[tuple[ChunkTask, ...]]:
    yield from _task_batches(
        inventory,
        selection,
        plan,
        variable_transforms,
        variable_names,
        output_layout,
    )


def file_tasks(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
) -> Iterator[tuple[ChunkTask, ...]]:
    yield from _task_batches(
        inventory,
        selection,
        plan,
        variable_transforms,
        variable_names,
        output_layout,
    )


def _physical_chunk_count(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    output_layout: OutputLayout | None,
) -> int:
    nt, ny, nx = selection.shape
    total = 0
    for name in selection.variables:
        chunk_time, chunk_lat, chunk_lon = _variable_chunks(
            inventory, selection, plan, name, output_layout
        )
        total += (
            ceil(nt / chunk_time)
            * ceil(ny / chunk_lat)
            * ceil(nx / chunk_lon)
        )
    return total


def _batch_count(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    output_layout: OutputLayout | None,
) -> int:
    nt, ny, nx = selection.shape
    batch_size = _effective_task_batch(plan.task_batch)
    total = 0
    for name in selection.variables:
        chunk_time, chunk_lat, chunk_lon = _variable_chunks(
            inventory, selection, plan, name, output_layout
        )
        spatial = ceil(ny / chunk_lat) * ceil(nx / chunk_lon)
        total += ceil(nt / chunk_time) * ceil(spatial / batch_size)
    return total


def _task_count(
    selection: Selection,
    plan: ConversionPlan,
) -> int:
    nt, ny, nx = selection.shape
    if plan.strategy == "file":
        return ceil(nt / max(1, plan.task_batch))
    if plan.strategy == "chunk":
        return (
            ceil(nt / plan.chunk_time)
            * len(selection.variables)
            * ceil(ny / plan.chunk_lat)
            * ceil(nx / plan.chunk_lon)
        )
    raise ValueError(f"direct_write 不支持策略 {plan.strategy}")


def _monitor(stop: threading.Event, samples: list[tuple[float, int]]) -> None:
    try:
        import psutil

        root = psutil.Process(os.getpid())
        tracked = {root.pid: root}
        while not stop.wait(0.25):
            discovered = [root, *root.children(recursive=True)]
            cpu = 0.0
            rss = 0
            for discovered_process in discovered:
                process = tracked.setdefault(
                    discovered_process.pid, discovered_process
                )
                try:
                    cpu += process.cpu_percent(None)
                    rss += process.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            samples.append((cpu, rss))
    except ImportError:
        return


def direct_write(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    plan: ConversionPlan,
    *,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
    cancel_event=None,
    progress: bool = True,
    progress_callback=None,
) -> dict[str, float | int]:
    initialize_zarr(
        inventory,
        selection,
        output,
        plan,
        variable_transforms,
        variable_names,
        output_layout,
    )
    if plan.strategy == "file":
        tasks = file_tasks(
            inventory, selection, plan, variable_transforms, variable_names, output_layout
        )
    elif plan.strategy == "chunk":
        tasks = chunk_task_batches(
            inventory, selection, plan, variable_transforms, variable_names, output_layout
        )
    else:
        raise ValueError(f"direct_write 不支持策略 {plan.strategy}")
    total_tasks = _task_count(selection, plan)
    total_batches = _batch_count(inventory, selection, plan, output_layout)
    total_logical = selected_output_logical_bytes(
        inventory,
        selection,
        variable_transforms,
        output_layout,
    )
    total_chunks = _physical_chunk_count(inventory, selection, plan, output_layout)
    effective_batch = _effective_task_batch(plan.task_batch)
    worker_count = max(1, min(plan.workers, total_batches))
    cache_chunk_time = max(
        _variable_chunks(inventory, selection, plan, name, output_layout)[0]
        for name in selection.variables
    )
    cache_limit = source_cache_limit(
        cache_chunk_time,
        worker_count,
        task_batch=effective_batch if plan.strategy == "file" else 1,
    )
    pending_limit = worker_count

    stop = threading.Event()
    samples: list[tuple[float, int]] = []
    monitor = threading.Thread(target=_monitor, args=(stop, samples), daemon=True)
    logical_bytes = 0
    chunks_written = 0
    source_opens = 0
    source_cache_hits = 0
    started = time.perf_counter()
    report_every = max(1, total_batches // 100)
    monitor.start()
    if progress:
        print(progress_line(0, total_batches, 0, 0.0), end="", flush=True)
    try:
        results = bounded_process_map(
            _write_batch,
            tasks,
            workers=worker_count,
            initializer=_worker_init,
            initargs=(str(output), inventory.source_engine, cache_limit),
            cancel_event=cancel_event,
            max_pending=pending_limit,
        )
        for completed, result in enumerate(results, 1):
            logical_bytes += result.logical_bytes
            chunks_written += result.chunks
            source_opens += result.source_opens
            source_cache_hits += result.source_cache_hits
            if progress_callback is not None:
                progress_callback(
                    min(total_logical, logical_bytes),
                    total_logical,
                    logical_bytes,
                    f"转换写入：{completed}/{total_batches}",
                )
            if progress and (
                completed == total_batches or completed % report_every == 0
            ):
                elapsed = max(time.perf_counter() - started, 1e-9)
                cpu, rss = samples[-1] if samples else (None, None)
                print(
                    progress_line(
                        completed,
                        total_batches,
                        logical_bytes,
                        elapsed,
                        cpu=cpu,
                        rss=rss,
                    ),
                    end="",
                    flush=True,
                )
        if chunks_written != total_chunks:
            raise RuntimeError(
                f"直接写入仅完成 {chunks_written}/{total_chunks} 个物理 chunks。"
            )
    finally:
        if worker_count == 1:
            _close_worker_sources()
        stop.set()
        monitor.join(timeout=2)
    elapsed = time.perf_counter() - started
    if progress:
        print()
    return {
        "elapsed": elapsed,
        "logical_bytes": logical_bytes,
        "throughput_mib_s": logical_bytes / max(elapsed, 1e-9) / 1024**2,
        "average_cpu": sum(cpu for cpu, _ in samples) / len(samples) if samples else 0.0,
        "peak_rss": max((rss for _, rss in samples), default=0),
        "tasks": total_tasks,
        "task_batches": total_batches,
        "task_batch": effective_batch,
        "chunks_written": chunks_written,
        "planned_chunks": total_chunks,
        "workers": worker_count,
        "scheduler_max_pending": pending_limit,
        "source_cache_limit": cache_limit,
        "source_cache_hits": source_cache_hits,
        "source_cache_misses": source_opens,
        "source_opens": source_opens,
    }


def fsync_tree(root: Path) -> None:
    """Make a benchmark durable without flushing unrelated filesystem data."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
        except OSError:
            pass
    try:
        descriptor = os.open(root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass
