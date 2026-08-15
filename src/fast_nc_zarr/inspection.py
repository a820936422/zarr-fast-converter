from __future__ import annotations

from collections import Counter
from itertools import batched
from hashlib import blake2b
import os
from pathlib import Path
from typing import Any

import numpy as np

from .models import FileRecord, Inventory, VariableSpec
from .runtime import bounded_process_map
from .system import effective_resource_budget
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


def _discover_files_with_stats(
    input_dir: Path, recursive: bool = False
) -> list[tuple[Path, os.stat_result]]:
    root = input_dir.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{root}")
    entries: list[tuple[Path, os.stat_result]] = []
    if recursive:
        for path in root.rglob("*"):
            if path.suffix.lower() in SUFFIXES and path.is_file():
                entries.append((path, path.stat()))
    else:
        with os.scandir(root) as iterator:
            for entry in iterator:
                if Path(entry.name).suffix.lower() not in SUFFIXES or not entry.is_file():
                    continue
                entries.append((Path(entry.path), entry.stat()))
    entries.sort(key=lambda item: item[0])
    if not entries:
        raise FileNotFoundError(f"没有在 {root} 中找到 NetCDF 文件。")
    suffixes = {path.suffix.lower() for path, _stat in entries}
    if len(suffixes) > 1:
        raise ValueError(
            "同一批次只能使用一种源文件后缀；发现：" + ", ".join(sorted(suffixes))
        )
    return entries


def discover_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    return [path for path, _stat in _discover_files_with_stats(input_dir, recursive)]


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
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return _clean_attr(value.item())
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return _clean_attr(value.reshape(-1)[0])
        return _clean_attr(value.tolist())
    if isinstance(value, list):
        return [_clean_attr(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clean_attr(item) for item in value)
    return value


_SUPPORTED_CALENDARS = frozenset({"", "standard", "gregorian", "proleptic_gregorian", "julian"})

def _normalize_daily_times(values: tuple[Any, ...], path: Path | None = None) -> tuple[np.datetime64, ...]:
    """Normalize source timestamps to midnight dates.

    The converter currently supports one or coarser observations per day.  A
    timestamp with a time-of-day is accepted when it is the only observation
    for that date; multiple timestamps collapsing to the same date indicate a
    sub-daily product and are rejected.
    """
    normalized: list[np.datetime64] = []
    for value in values:
        calendar = str(getattr(value, "calendar", "")).strip().lower()
        if calendar not in _SUPPORTED_CALENDARS:
            location = f" in {path.name}" if path is not None else ""
            raise ValueError(
                f"当前 native/兼容时间轴不支持 calendar={calendar!r}{location}；"
                "请使用标准 Gregorian calendar 或显式预处理。"
            )
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
    if engine == "h5netcdf":
        try:
            import h5netcdf

            with h5netcdf.File(path, "r") as dataset:
                available = tuple(str(item) for item in dataset.dimensions)
        except Exception:
            available = ()
        if available:
            if set(CANONICAL_DIMENSIONS).issubset(available):
                return CANONICAL_DIMENSIONS
            raise DimensionMappingRequired(available)

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


def _inspect_file_xarray(
    path: Path,
    engine: str,
    source_dimensions: tuple[str, str, str],
    time_rule: TimeRule | None,
    filename_fields: tuple[FilenameField, ...],
    file_stat: os.stat_result | None = None,
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
        stat = file_stat or path.stat()
        return FileRecord(
            path=path,
            size_bytes=stat.st_size,
            times=times,
            time_keys=tuple(time_key(item) for item in times),
            lat_hash=_hash_axis(lat),
            lon_hash=_hash_axis(lon),
            lat_size=int(lat.size),
            lon_size=int(lon.size),
            variables=tuple(specs),
            mtime_ns=stat.st_mtime_ns,
        )
    finally:
        ds.close()


_HDF5_INTERNAL_ATTRS = {
    "CLASS",
    "NAME",
    "REFERENCE_LIST",
    "DIMENSION_LIST",
    "_Netcdf4Coordinates",
    "_Netcdf4Dimid",
}


def _public_hdf5_attrs(dataset) -> dict[str, Any]:
    return {
        str(key): _clean_attr(value)
        for key, value in dataset.attrs.items()
        if str(key) not in _HDF5_INTERNAL_ATTRS
    }


def _inspect_file_h5py(
    path: Path,
    source_dimensions: tuple[str, str, str],
    time_rule: TimeRule | None,
    filename_fields: tuple[FilenameField, ...],
    file_stat: os.stat_result | None = None,
) -> FileRecord:
    """Read ordinary NetCDF4/HDF5 metadata through h5py's thin API."""
    import h5py

    with h5py.File(path, "r") as dataset:
        if any(isinstance(value, h5py.Group) for value in dataset.values()):
            raise ValueError("grouped NetCDF requires the compatibility reader")
        source_time, source_lat, source_lon = source_dimensions
        coordinates = {}
        for name, label in (
            (source_time, "时间"),
            (source_lat, "纬度"),
            (source_lon, "经度"),
        ):
            variable = dataset.get(name)
            if not isinstance(variable, h5py.Dataset) or variable.ndim != 1:
                raise ValueError(f"{path.name} 的{label}坐标 {name} 必须是一维坐标。")
            coordinates[name] = variable

        lat = np.asarray(coordinates[source_lat][...])
        lon = np.asarray(coordinates[source_lon][...])
        time_variable = coordinates[source_time]
        time_attrs = _public_hdf5_attrs(time_variable)
        raw_values = np.asarray(time_variable[...])
        raw_times = tuple(raw_values)
        if time_rule is None:
            times = _normalize_daily_times(
                _decode_h5netcdf_times(raw_values, time_attrs), path
            )
        else:
            times = resolve_file_times(
                path.name,
                raw_times,
                time_attrs,
                time_rule,
                filename_fields,
            )

        coordinate_names = set(source_dimensions)
        for variable in dataset.values():
            if not isinstance(variable, h5py.Dataset):
                continue
            value = _clean_attr(variable.attrs.get("coordinates", ""))
            if value:
                coordinate_names.update(str(value).split())
        rename_dimensions = dict(zip(source_dimensions, CANONICAL_DIMENSIONS))
        specs = []
        for name, variable in dataset.items():
            if name in coordinate_names:
                continue
            if not isinstance(variable, h5py.Dataset):
                raise ValueError("grouped NetCDF requires the compatibility reader")
            dimensions = []
            for axis in variable.dims:
                scales = list(axis.keys())
                if len(scales) != 1:
                    raise ValueError("unlabelled HDF5 dimensions require the compatibility reader")
                dimensions.append(str(scales[0]))
            dimensions_tuple = tuple(dimensions)
            specs.append(
                VariableSpec(
                    name=str(name),
                    dims=tuple(
                        rename_dimensions.get(dim, dim) for dim in dimensions_tuple
                    ),
                    dtype=str(variable.dtype),
                    shape_without_time=tuple(
                        int(size)
                        for dim, size in zip(dimensions_tuple, variable.shape)
                        if dim != source_time
                    ),
                    native_chunks=(
                        tuple(map(int, variable.chunks)) if variable.chunks else None
                    ),
                    attrs=_public_hdf5_attrs(variable),
                )
            )
    stat = file_stat or path.stat()
    return FileRecord(
        path=path,
        size_bytes=stat.st_size,
        times=times,
        time_keys=tuple(time_key(item) for item in times),
        lat_hash=_hash_axis(lat),
        lon_hash=_hash_axis(lon),
        lat_size=int(lat.size),
        lon_size=int(lon.size),
        variables=tuple(specs),
        mtime_ns=stat.st_mtime_ns,
    )


def _decode_h5netcdf_times(values: np.ndarray, attrs: dict[str, Any]) -> tuple[Any, ...]:
    units = attrs.get("units")
    if not units:
        raise ValueError("time 坐标缺少 CF units")
    try:
        import cftime

        decoded = cftime.num2date(
            values,
            units=str(units),
            calendar=str(attrs.get("calendar", "standard")),
            only_use_cftime_datetimes=False,
        )
    except Exception as exc:
        raise ValueError(f"无法解码 CF time 坐标：{exc}") from exc
    return tuple(np.atleast_1d(decoded))


def _inspect_file_h5netcdf(
    path: Path,
    source_dimensions: tuple[str, str, str],
    time_rule: TimeRule | None,
    filename_fields: tuple[FilenameField, ...],
    file_stat: os.stat_result | None = None,
) -> FileRecord:
    """Read HDF5-backed NetCDF metadata without constructing xarray objects."""
    import h5netcdf

    with h5netcdf.File(path, "r") as dataset:
        _validate_source_dimensions(source_dimensions, set(dataset.dimensions))
        source_time, source_lat, source_lon = source_dimensions
        coordinates = {}
        for name, label in (
            (source_time, "时间"),
            (source_lat, "纬度"),
            (source_lon, "经度"),
        ):
            variable = dataset.variables.get(name)
            if variable is None or tuple(variable.dimensions) != (name,):
                raise ValueError(f"{path.name} 的{label}坐标 {name} 必须是一维坐标。")
            coordinates[name] = variable

        lat = np.asarray(coordinates[source_lat][:])
        lon = np.asarray(coordinates[source_lon][:])
        time_variable = coordinates[source_time]
        time_attrs = {
            str(key): _clean_attr(value) for key, value in time_variable.attrs.items()
        }
        raw_values = np.asarray(time_variable[:])
        raw_times = tuple(raw_values)
        if time_rule is None:
            times = _normalize_daily_times(
                _decode_h5netcdf_times(raw_values, time_attrs), path
            )
        else:
            times = resolve_file_times(
                path.name,
                raw_times,
                time_attrs,
                time_rule,
                filename_fields,
            )

        coordinate_names = set(source_dimensions)
        for variable in dataset.variables.values():
            value = variable.attrs.get("coordinates")
            if value:
                coordinate_names.update(str(value).split())
        rename_dimensions = dict(zip(source_dimensions, CANONICAL_DIMENSIONS))
        specs = []
        for name, variable in dataset.variables.items():
            if name in coordinate_names:
                continue
            dimensions = tuple(str(item) for item in variable.dimensions)
            chunks = variable.chunks
            specs.append(
                VariableSpec(
                    name=str(name),
                    dims=tuple(rename_dimensions.get(dim, dim) for dim in dimensions),
                    dtype=str(variable.dtype),
                    shape_without_time=tuple(
                        int(size)
                        for dim, size in zip(dimensions, variable.shape)
                        if dim != source_time
                    ),
                    native_chunks=tuple(map(int, chunks)) if chunks else None,
                    attrs={
                        str(key): _clean_attr(value)
                        for key, value in variable.attrs.items()
                    },
                )
            )
    stat = file_stat or path.stat()
    return FileRecord(
        path=path,
        size_bytes=stat.st_size,
        times=times,
        time_keys=tuple(time_key(item) for item in times),
        lat_hash=_hash_axis(lat),
        lon_hash=_hash_axis(lon),
        lat_size=int(lat.size),
        lon_size=int(lon.size),
        variables=tuple(specs),
        mtime_ns=stat.st_mtime_ns,
    )


def inspect_file(
    path: Path,
    engine: str,
    source_dimensions: tuple[str, str, str] = CANONICAL_DIMENSIONS,
    time_rule: TimeRule | None = None,
    filename_fields: tuple[FilenameField, ...] = (),
    file_stat: os.stat_result | None = None,
) -> FileRecord:
    if engine == "h5netcdf":
        try:
            return _inspect_file_h5py(
                path, source_dimensions, time_rule, filename_fields, file_stat
            )
        except (ImportError, OSError, ValueError, TypeError, KeyError):
            try:
                return _inspect_file_h5netcdf(
                    path, source_dimensions, time_rule, filename_fields, file_stat
                )
            except (ImportError, OSError, ValueError, TypeError, KeyError):
                # Preserve compatibility with unusual NetCDF/HDF layouts that
                # xarray can decode but the direct metadata readers cannot.
                pass
    return _inspect_file_xarray(
        path, engine, source_dimensions, time_rule, filename_fields, file_stat
    )


def _inspect_file_task(task) -> FileRecord:
    return inspect_file(*task)


def _inspect_file_batch(tasks) -> tuple[FileRecord, ...]:
    return tuple(_inspect_file_task(task) for task in tasks)


def choose_inspection_workers(files: list[Path], requested: int | None = None) -> int:
    """Choose metadata workers from the effective resource contract.

    Storage profiles remain benchmark context; they never pre-emptively cap
    the candidate ceiling based only on rotational or network media.
    """

    if not files:
        return 1
    budget = effective_resource_budget(
        source=Path(files[0]),
        requested=(None if requested is None else max(1, int(requested))),
    )
    return max(1, min(len(files), int(budget.worker_ceiling)))


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

def _read_reference_axes(
    path: Path,
    engine: str,
    source_lat: str,
    source_lon: str,
) -> tuple[np.ndarray, np.ndarray]:
    if engine == "h5netcdf":
        try:
            import h5netcdf

            with h5netcdf.File(path, "r") as dataset:
                return (
                    np.asarray(dataset.variables[source_lat][:]).copy(),
                    np.asarray(dataset.variables[source_lon][:]).copy(),
                )
        except (ImportError, OSError, KeyError, TypeError, ValueError):
            pass

    import xarray as xr

    with xr.open_dataset(
        path, engine=engine, chunks=None, decode_times=True, mask_and_scale=False
    ) as dataset:
        return (
            np.asarray(dataset[source_lat].values).copy(),
            np.asarray(dataset[source_lon].values).copy(),
        )


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
    cached_inventory: Inventory | None = None,
    cancel_event=None,
) -> Inventory:
    file_entries = _discover_files_with_stats(input_dir, recursive)
    files = [path for path, _stat in file_entries]
    stats_by_path = {path: stat for path, stat in file_entries}

    source_dimensions = (
        _source_dimensions(files[0], engine)
        if dimension_names is None
        else tuple(str(item).strip() for item in dimension_names)
    )
    cached_by_path = (
        {record.path: record for record in cached_inventory.files}
        if (
            cached_inventory is not None
            and cached_inventory.source_engine == engine
            and cached_inventory.source_dimensions == source_dimensions
            and cached_inventory.source_mode
            == ("hybrid" if time_rule is not None and time_rule.is_hybrid else "dimension")
        )
        else {}
    )
    records_by_path: dict[Path, FileRecord] = {}
    changed_files = []
    for path in files:
        cached = cached_by_path.get(path)
        stat = stats_by_path[path]
        if (
            cached is not None
            and cached.size_bytes == stat.st_size
            and cached.mtime_ns is not None
            and cached.mtime_ns == stat.st_mtime_ns
        ):
            records_by_path[path] = cached
        else:
            changed_files.append(path)
    worker_count = choose_inspection_workers(changed_files or files, workers)
    if progress:
        print(
            f"发现 {len(files)} 个 NetCDF 文件；复用 {len(records_by_path)} 个，"
            f"检查 {len(changed_files)} 个（{worker_count} 个进程）……"
        )
    tasks = (
        (path, engine, source_dimensions, time_rule, filename_fields, stats_by_path[path])
        for path in changed_files
    )
    task_batch_size = max(
        1,
        min(64, (len(changed_files) + max(1, worker_count) * 16 - 1) // (max(1, worker_count) * 16)),
    )
    task_batches = batched(tasks, task_batch_size)
    for batch_records in bounded_process_map(
        _inspect_file_batch,
        task_batches,
        workers=min(worker_count, max(1, len(changed_files))),
        cancel_event=cancel_event,
    ):
        for record in batch_records:
            records_by_path[record.path] = record
    records = [records_by_path[path] for path in files]
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
    lat, lon = _read_reference_axes(reference.path, engine, source_lat, source_lon)
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

def inspect_netcdf_native(path: Path) -> dict[str, Any]:
    """Inspect a standard NetCDF file through the Rust native reader."""
    from ._backend import BackendUnavailableError, resolve_backend

    if resolve_backend("rust", "raw.netcdf.inspect") != "rust":
        raise BackendUnavailableError("Rust NetCDF native inspect is unavailable")
    try:
        native = __import__("fast_nc_zarr._native", fromlist=["inspect_netcdf_json"])
        payload = native.inspect_netcdf_json(str(Path(path).expanduser().resolve()))
    except (AttributeError, ImportError) as exc:
        raise BackendUnavailableError("native extension lacks inspect_netcdf_json") from exc
    import json

    result = json.loads(payload)
    if not isinstance(result, dict):
        raise ValueError("native NetCDF inspect response must be an object")
    return result
