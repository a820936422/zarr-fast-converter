from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from hashlib import blake2b
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np

from .models import FileRecord, Inventory, VariableSpec
from .runtime import spawn_context
from .system import physical_cpu_count, storage_profile
from .time_mapping import FilenameField, TimeRule, resolve_file_times

SUFFIXES = {".nc", ".nc4", ".nc3", ".cdf", ".hdf"}
CANONICAL_DIMENSIONS = ("time", "lat", "lon")


class DimensionMappingRequired(ValueError):
    """Raised when a dataset does not use the canonical dimension names."""

    def __init__(self, available: tuple[str, ...]):
        self.available = available
        super().__init__(
            "输入数据未使用标准维度名 time/lat/lon；"
            "请手动指定实际名称（--time-dim、--lat-dim、--lon-dim）。"
            f" 首个文件中的维度：{', '.join(available)}"
        )


def discover_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in SUFFIXES)
    if not files:
        raise FileNotFoundError(f"没有在 {input_dir} 中找到 NetCDF 文件。")
    suffixes = {path.suffix.lower() for path in files}
    if len(suffixes) > 1:
        raise ValueError(
            "同一批次只能使用一种源文件后缀；发现：" + ", ".join(sorted(suffixes))
        )
    return files


def _hash_axis(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    digest = blake2b(digest_size=16)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.view(np.uint8))
    return digest.hexdigest()


def time_key(value: Any) -> str:
    if isinstance(value, np.datetime64):
        return "numpy:" + np.datetime_as_string(value, unit="ns")
    calendar = getattr(value, "calendar", "")
    return f"{type(value).__module__}.{type(value).__name__}:{calendar}:{value!s}"


def _clean_attr(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _normalize_daily_times(values: tuple[Any, ...], path: Path | None = None) -> tuple[np.datetime64, ...]:
    """Normalize source timestamps to midnight dates.

    The converter currently supports one or coarser observations per day.  A
    timestamp with a time-of-day is accepted when it is the only observation
    for that date; multiple timestamps collapsing to the same date indicate a
    sub-daily product and are rejected.
    """
    normalized: list[np.datetime64] = []
    for value in values:
        try:
            stamp = np.datetime64(value, "ns")
            if np.isnat(stamp):
                raise ValueError
            day = stamp.astype("datetime64[D]").astype("datetime64[ns]")
        except (TypeError, ValueError, OverflowError):
            text = str(value)
            if len(text) < 10:
                location = f" in {path.name}" if path is not None else ""
                raise ValueError(f"无法将时间 {value!r}{location} 转换为 YYYY-MM-DD。")
            try:
                day = np.datetime64(text[:10], "D").astype("datetime64[ns]")
            except (TypeError, ValueError, OverflowError) as exc:
                location = f" in {path.name}" if path is not None else ""
                raise ValueError(f"无法将时间 {value!r}{location} 转换为 YYYY-MM-DD。") from exc
        normalized.append(day)
    keys = [time_key(item) for item in normalized]
    if len(set(keys)) != len(keys):
        location = f"：{path.name}" if path is not None else ""
        raise ValueError(
            "当前版本不支持日内尺度数据；同一日期存在多个时间点" + location + "。"
        )
    return tuple(normalized)


def _source_dimensions(path: Path, engine: str) -> tuple[str, str, str]:
    """Infer standard names or report dimensions requiring a user mapping."""
    import xarray as xr

    try:
        with xr.open_dataset(
            path, engine=engine, chunks=None, decode_times=False, mask_and_scale=False
        ) as ds:
            available = tuple(str(item) for item in ds.dims)
    except Exception as exc:
        raise RuntimeError(f"无法读取 {path}：{exc}") from exc
    if set(CANONICAL_DIMENSIONS).issubset(available):
        return CANONICAL_DIMENSIONS
    raise DimensionMappingRequired(available)


def _validate_source_dimensions(
    source_dimensions: tuple[str, str, str], available: set[str]
) -> None:
    if len(source_dimensions) != 3 or len(set(source_dimensions)) != 3:
        raise ValueError("time、纬度和经度的实际维度名称必须互不相同。")
    missing = [name for name in source_dimensions if name not in available]
    if missing:
        raise ValueError("指定的源维度不存在：" + ", ".join(missing) + "。")


def inspect_file(
    path: Path,
    engine: str,
    source_dimensions: tuple[str, str, str] = CANONICAL_DIMENSIONS,
    time_rule: TimeRule | None = None,
    filename_fields: tuple[FilenameField, ...] = (),
) -> FileRecord:
    import xarray as xr

    try:
        ds = xr.open_dataset(
            path,
            engine=engine,
            chunks=None,
            decode_times=time_rule is None,
            mask_and_scale=False,
        )
    except Exception as exc:
        raise RuntimeError(f"无法读取 {path}: {exc}") from exc
    try:
        _validate_source_dimensions(source_dimensions, set(ds.dims))
        source_time, source_lat, source_lon = source_dimensions
        for name, label in (
            (source_time, "时间"),
            (source_lat, "纬度"),
            (source_lon, "经度"),
        ):
            if name not in ds.coords or ds[name].dims != (name,):
                raise ValueError(f"{path.name} 的{label}坐标 {name} 必须是一维坐标。")

        lat = np.asarray(ds[source_lat].values)
        lon = np.asarray(ds[source_lon].values)
        rename_dimensions = dict(zip(source_dimensions, CANONICAL_DIMENSIONS))
        # Keep numpy datetime64 scalars; ndarray.tolist() turns ns-resolution
        # values into integers on some NumPy versions.
        raw_times = tuple(np.asarray(ds[source_time].values))
        if time_rule is None:
            times = _normalize_daily_times(raw_times, path)
        else:
            times = resolve_file_times(
                path.name,
                raw_times,
                {str(key): _clean_attr(value) for key, value in ds[source_time].attrs.items()},
                time_rule,
                filename_fields,
            )
        specs = []
        for name, variable in ds.data_vars.items():
            chunks = variable.encoding.get("chunksizes")
            specs.append(
                VariableSpec(
                    name=name,
                    dims=tuple(rename_dimensions.get(dim, dim) for dim in variable.dims),
                    dtype=str(variable.dtype),
                    shape_without_time=tuple(
                        int(variable.sizes[dim])
                        for dim in variable.dims
                        if dim != source_time
                    ),
                    native_chunks=tuple(map(int, chunks)) if chunks else None,
                    attrs={key: _clean_attr(value) for key, value in variable.attrs.items()},
                )
            )
        return FileRecord(
            path=path,
            size_bytes=path.stat().st_size,
            times=times,
            time_keys=tuple(time_key(item) for item in times),
            lat_hash=_hash_axis(lat),
            lon_hash=_hash_axis(lon),
            lat_size=int(lat.size),
            lon_size=int(lon.size),
            variables=tuple(specs),
        )
    finally:
        ds.close()


def choose_inspection_workers(files: list[Path], requested: int | None = None) -> int:
    """Choose metadata workers while accounting for rotational storage.

    Opening thousands of small HDF/NetCDF files is seek-bound on a mechanical
    disk.  Four concurrent readers, which was the previous default, can make
    the heads repeatedly jump between file streams and be slower than a
    smaller queue.  Keep an explicit user request authoritative, but use at
    most two automatic readers for large collections on rotational media.
    """
    if requested is not None:
        return max(1, min(requested, len(files)))
    cpus = physical_cpu_count()
    profile = storage_profile(files[0])
    median_size = int(np.median([item.stat().st_size for item in files]))
    if profile.rotational and len(files) >= 256 and median_size < 32 * 1024**2:
        return min(2, cpus, len(files))
    return min(8, cpus, len(files))


def _variable_signature(spec: VariableSpec) -> tuple:
    critical_attrs = tuple(
        (name, repr(spec.attrs.get(name)))
        for name in ("_FillValue", "missing_value", "scale_factor", "add_offset", "units")
        if name in spec.attrs
    )
    return spec.name, spec.dims, spec.dtype, spec.shape_without_time, critical_attrs


def _variable_signatures(variables: tuple[VariableSpec, ...]) -> tuple[tuple, ...]:
    """Return a stable file schema signature independent of variable order.

    NetCDF/HDF producers are allowed to emit variables in a different order
    from file to file.  The converter only cares about the variable names and
    their definitions, so comparing the raw iteration order can reject a
    perfectly compatible collection (FLUXSAT is one such product).
    """
    return tuple(sorted((_variable_signature(item) for item in variables), key=lambda item: item[0]))


def _infer_frequency(times: np.ndarray) -> tuple[str, list[str]]:
    if len(times) < 2:
        return "单个时间点", []
    try:
        values = times.astype("datetime64[ns]")
        diffs = np.diff(values).astype("timedelta64[ns]").astype(np.int64)
        positive = diffs[diffs > 0]
        if not len(positive):
            return "无法判断", []
        base = int(np.min(positive))
        day_ns = 86_400_000_000_000
        if base % day_ns == 0:
            days = base // day_ns
            label = "每天" if days == 1 else f"每 {days} 天"
        else:
            label = f"每 {base / 1e9:g} 秒"
        gaps = []
        for index, delta in enumerate(diffs):
            ratio = delta / base
            if ratio > 1.5 and abs(ratio - round(ratio)) < 1e-6:
                gaps.append(
                    f"{values[index]} -> {values[index + 1]}，缺 {round(ratio) - 1} 个时间点"
                )
        return label, gaps
    except (TypeError, ValueError):
        return "非标准日历或不规则时间", []


def inspect_dataset(
    input_dir: Path,
    *,
    recursive: bool = False,
    engine: str = "h5netcdf",
    dimension_names: tuple[str, str, str] | None = None,
    workers: int | None = None,
    progress: bool = True,
    time_rule: TimeRule | None = None,
    filename_fields: tuple[FilenameField, ...] = (),
    cancel_event=None,
) -> Inventory:
    import xarray as xr

    files = discover_files(input_dir, recursive)
    source_dimensions = (
        _source_dimensions(files[0], engine)
        if dimension_names is None
        else tuple(str(item).strip() for item in dimension_names)
    )
    worker_count = choose_inspection_workers(files, workers)
    if progress:
        print(f"发现 {len(files)} 个 NetCDF 文件，使用 {worker_count} 个进程检查元数据……")
    if worker_count == 1:
        records = []
        for path in files:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("任务已取消。")
            records.append(
                inspect_file(path, engine, source_dimensions, time_rule, filename_fields)
            )
    else:
        chunksize = max(1, min(16, len(files) // max(1, worker_count * 8)))
        executor = ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=spawn_context(),
        )
        terminated = False
        try:
            records = []
            for record in executor.map(
                inspect_file,
                files,
                repeat(engine),
                repeat(source_dimensions),
                repeat(time_rule),
                repeat(filename_fields),
                chunksize=chunksize,
            ):
                if cancel_event is not None and cancel_event.is_set():
                    terminate = getattr(executor, "terminate_workers", None)
                    if terminate is not None:
                        terminate()
                    else:
                        executor.shutdown(wait=False, cancel_futures=True)
                    terminated = True
                    raise RuntimeError("任务已取消。")
                records.append(record)
        finally:
            if not terminated:
                executor.shutdown(wait=True)

    reference = records[0]
    reference_variables = _variable_signatures(reference.variables)
    errors = []
    for record in records[1:]:
        if (record.lat_hash, record.lon_hash) != (reference.lat_hash, reference.lon_hash):
            errors.append(f"{record.path.name}: 经纬度网格与首文件不同")
        signature = _variable_signatures(record.variables)
        if signature != reference_variables:
            errors.append(f"{record.path.name}: 变量定义与首文件不同")
        if len(errors) >= 10:
            break
    if errors:
        raise ValueError("输入文件结构不一致：\n  " + "\n  ".join(errors))

    key_to_time: dict[str, Any] = {}
    key_to_source: dict[str, Path] = {}
    for record in records:
        for key, value in zip(record.time_keys, record.times):
            if key in key_to_time:
                raise ValueError(
                    f"时间坐标重复：{value} 同时存在于 {key_to_source[key].name} 和 {record.path.name}"
                )
            key_to_time[key] = value
            key_to_source[key] = record.path
    try:
        ordered = sorted(key_to_time.items(), key=lambda item: item[1])
    except TypeError:
        ordered = sorted(key_to_time.items())
    ordered_keys = tuple(key for key, _ in ordered)
    ordered_times = np.asarray([value for _, value in ordered])

    source_time, source_lat, source_lon = source_dimensions
    with xr.open_dataset(
        reference.path, engine=engine, chunks=None, decode_times=True, mask_and_scale=False
    ) as ds:
        lat = np.asarray(ds[source_lat].values).copy()
        lon = np.asarray(ds[source_lon].values).copy()
    frequency, gaps = _infer_frequency(ordered_times)
    return Inventory(
        input_dir=Path(input_dir).expanduser().resolve(),
        files=records,
        lat_values=lat,
        lon_values=lon,
        times=ordered_times,
        time_keys=ordered_keys,
        variables={item.name: item for item in reference.variables},
        source_engine=engine,
        source_dimensions=(source_time, source_lat, source_lon),
        frequency=frequency,
        gaps=gaps,
        total_bytes=sum(item.size_bytes for item in records),
        source_mode="hybrid" if time_rule is not None and time_rule.is_hybrid else "dimension",
    )


def inventory_summary(info: Inventory) -> str:
    def render_time(value: Any) -> str:
        if isinstance(value, np.datetime64):
            return str(np.datetime_as_string(value, unit="D"))
        return str(value)

    sizes = [item.size_bytes for item in info.files]
    time_counts = Counter(item.time_count for item in info.files)
    missing = (
        f"理论时间轴补齐缺失：{len(info.missing_time_keys)} 个"
        if info.missing_time_keys
        else "理论时间轴补齐缺失：无"
    )
    return "\n".join(
        [
            f"文件：{len(info.files)} 个，{info.total_bytes / 1024**3:.2f} GiB；"
            f"中位文件 {np.median(sizes) / 1024**2:.2f} MiB",
            f"时间：{render_time(info.times[0])} -> {render_time(info.times[-1])}，"
            f"{len(info.times)} 点，{info.frequency}",
            f"每文件时间点分布：{dict(time_counts)}",
            f"纬度：{info.lat_values.min():g} .. {info.lat_values.max():g}，{len(info.lat_values)} 点",
            f"经度：{info.lon_values.min():g} .. {info.lon_values.max():g}，{len(info.lon_values)} 点",
            "源维度映射："
            f"time={info.source_dimensions[0]}, "
            f"lat={info.source_dimensions[1]}, "
            f"lon={info.source_dimensions[2]}",
            "变量：" + ", ".join(info.variables),
            f"时间缺口：{len(info.gaps)} 个" + (f"（首个：{info.gaps[0]}）" if info.gaps else ""),
            missing,
        ]
    )
