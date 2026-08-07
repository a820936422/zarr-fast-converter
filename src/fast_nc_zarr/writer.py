from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .models import ConversionPlan, Inventory, Selection, VariableTransform
from .runtime import spawn_context

_OUTPUT_GROUP = None
_SOURCE_CACHE = None
_SOURCE_CACHE_LIMIT = 4


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
    output_fill: float | int | None = None
    source_dtype: str | None = None
    output_variable: str | None = None


def make_compressor(name: str, level: int, shuffle: str):
    if name == "none":
        return None
    from zarr.codecs import BloscCodec

    return BloscCodec(cname=name, clevel=level, shuffle=shuffle)


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
        compressor = make_compressor(plan.compression, plan.compression_level, plan.shuffle)
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
            if transform is not None and transform.scale_factor is not None and output_dtype.kind not in "fc":
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
                attrs.pop("scale_factor", None)
                attrs.pop("add_offset", None)
            shape = tuple(
                sizes.get(dim, int(original.sizes[source_dimensions.get(dim, dim)]))
                for dim in canonical_dims
            )
            chunks = _chunk_for_dims(canonical_dims, plan, sizes)
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
        coordinates = {
            "time": xr.Variable(
                ("time",),
                inventory.times[selection.time_start : selection.time_stop],
                attrs=source_time_attrs,
            ),
            "lat": xr.Variable(
                ("lat",),
                inventory.lat_values[selection.lat_start : selection.lat_stop],
                attrs=source[source_dimensions["lat"]].attrs.copy(),
            ),
            "lon": xr.Variable(
                ("lon",),
                inventory.lon_values[selection.lon_start : selection.lon_stop],
                attrs=source[source_dimensions["lon"]].attrs.copy(),
            ),
        }
        for name, coordinate in coordinates.items():
            encoding[name] = {"chunks": (min(len(coordinate), 8192),)}
        template = xr.Dataset(variables, coords=coordinates, attrs=source.attrs.copy())
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


def _worker_init(output: str) -> None:
    global _OUTPUT_GROUP, _SOURCE_CACHE
    from collections import OrderedDict

    import zarr

    _OUTPUT_GROUP = zarr.open_group(output, mode="r+")
    _SOURCE_CACHE = OrderedDict()


def _source(path: str):
    global _SOURCE_CACHE
    import netCDF4

    existing = _SOURCE_CACHE.get(path)
    if existing is not None:
        _SOURCE_CACHE.move_to_end(path)
        return existing
    ds = netCDF4.Dataset(path, mode="r")
    ds.set_auto_mask(False)
    ds.set_auto_scale(False)
    _SOURCE_CACHE[path] = ds
    while len(_SOURCE_CACHE) > _SOURCE_CACHE_LIMIT:
        _, stale = _SOURCE_CACHE.popitem(last=False)
        stale.close()
    return ds


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
    output_fill: float | int | None,
    output_dtype: str,
) -> np.ndarray:
    raw = np.asarray(data)
    mask = np.zeros(raw.shape, dtype=bool)
    if fill_values:
        for value in fill_values:
            try:
                if np.isnan(value) and np.issubdtype(raw.dtype, np.floating):
                    mask |= np.isnan(raw)
                    continue
            except TypeError:
                pass
            mask |= raw == value
    result = raw.astype(np.dtype(output_dtype), copy=True)
    if scale_factor is not None:
        if mask.any():
            result[~mask] *= scale_factor
        else:
            result *= scale_factor
    if mask.any():
        if output_fill is None:
            raise ValueError("缺少整型变量的输出缺失值。")
        result[mask] = output_fill
    return np.ascontiguousarray(result)


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
    data = _apply_transform(
        data,
        task.fill_values,
        task.scale_factor,
        task.output_fill,
        task.dtype,
    )
    output_selection = tuple(slice(start, stop) for start, stop in task.output_ranges)
    _OUTPUT_GROUP[task.output_variable or task.variable][output_selection] = data
    return int(data.nbytes)


def _write_batch(tasks: tuple[ChunkTask, ...]) -> int:
    return sum(_write_chunk(task) for task in tasks)


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
    if transform is not None and transform.scale_factor is not None and np.dtype(output_dtype).kind not in "fc":
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
    return ChunkTask(
        variable=name,
        output_variable=(variable_names or {}).get(name, name),
        dims=spec.dims,
        output_dims=output_dims,
        dtype=output_dtype,
        output_ranges=tuple(ranges_by_dim[dim] for dim in output_dims),
        source_lat=(selection.lat_start + y0, selection.lat_start + y1),
        source_lon=(selection.lon_start + x0, selection.lon_start + x1),
        segments=_segments(lookup, t0, t1),
        fill_values=transform.fill_values if transform is not None else None,
        scale_factor=transform.scale_factor if transform is not None else None,
        output_fill=output_fill,
        source_dtype=spec.dtype,
    )


def chunk_tasks(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
) -> list[ChunkTask]:
    nt, ny, nx = selection.shape
    lookup = _time_lookup(inventory, selection)
    tasks = []
    for t0 in range(0, nt, plan.chunk_time):
        t1 = min(nt, t0 + plan.chunk_time)
        for name in selection.variables:
            for y0 in range(0, ny, plan.chunk_lat):
                y1 = min(ny, y0 + plan.chunk_lat)
                for x0 in range(0, nx, plan.chunk_lon):
                    x1 = min(nx, x0 + plan.chunk_lon)
                    tasks.append(
                        _chunk_task(inventory, selection, plan, lookup, name, t0, t1, y0, y1, x0, x1, variable_transforms, variable_names)
                    )
    return tasks


def file_tasks(
    inventory: Inventory,
    selection: Selection,
    plan: ConversionPlan,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
) -> list[tuple[ChunkTask, ...]]:
    if plan.chunk_time != 1:
        raise ValueError("文件优先策略当前要求 chunk_time=1。")
    nt, ny, nx = selection.shape
    lookup = _time_lookup(inventory, selection)
    time_batches = []
    for t0 in range(nt):
        one_time = []
        for name in selection.variables:
            for y0 in range(0, ny, plan.chunk_lat):
                y1 = min(ny, y0 + plan.chunk_lat)
                for x0 in range(0, nx, plan.chunk_lon):
                    x1 = min(nx, x0 + plan.chunk_lon)
                    one_time.append(
                        _chunk_task(inventory, selection, plan, lookup, name, t0, t0 + 1, y0, y1, x0, x1, variable_transforms, variable_names)
                    )
        time_batches.append(tuple(one_time))
    if plan.task_batch <= 1:
        return time_batches
    grouped = []
    for index in range(0, len(time_batches), plan.task_batch):
        grouped.append(tuple(task for batch in time_batches[index : index + plan.task_batch] for task in batch))
    return grouped


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
    cancel_event=None,
    progress: bool = True,
) -> dict[str, float | int]:
    initialize_zarr(inventory, selection, output, plan, variable_transforms, variable_names)
    if plan.strategy == "file":
        tasks: list = file_tasks(inventory, selection, plan, variable_transforms, variable_names)
    elif plan.strategy == "chunk":
        tasks = chunk_tasks(inventory, selection, plan, variable_transforms, variable_names)
    else:
        raise ValueError(f"direct_write 不支持策略 {plan.strategy}")

    stop = threading.Event()
    samples: list[tuple[float, int]] = []
    monitor = threading.Thread(target=_monitor, args=(stop, samples), daemon=True)
    logical_bytes = 0
    started = time.perf_counter()
    report_every = max(1, len(tasks) // 100)
    monitor.start()
    if progress:
        print(progress_line(0, len(tasks), 0, 0.0), end="", flush=True)
    try:
        executor = ProcessPoolExecutor(
            max_workers=plan.workers,
            mp_context=spawn_context(),
            initializer=_worker_init,
            initargs=(str(output),),
        )
        try:
            worker = _write_batch if plan.strategy == "file" else _write_chunk
            results = executor.map(worker, tasks, chunksize=1)
            for completed, amount in enumerate(results, 1):
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("任务已取消。")
                logical_bytes += amount
                if progress and (completed == len(tasks) or completed % report_every == 0):
                    elapsed = max(time.perf_counter() - started, 1e-9)
                    cpu, rss = samples[-1] if samples else (None, None)
                    print(
                        progress_line(
                            completed,
                            len(tasks),
                            logical_bytes,
                            elapsed,
                            cpu=cpu,
                            rss=rss,
                        ),
                        end="",
                        flush=True,
                    )
        except BaseException:
            terminate = getattr(executor, "terminate_workers", None)
            if terminate is not None:
                terminate()
            else:
                executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
    finally:
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
        "tasks": len(tasks),
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
