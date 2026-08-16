from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import itertools
from pathlib import Path
import shutil
import time
from typing import Iterable
from uuid import uuid4

import dask
import dask.array as da
import numpy as np
import xarray as xr
import zarr

from .._backend import resolve_backend
from ..metadata import sanitize_cf_references
from ..rechunking.models import DatasetInfo, VariableInfo
from ..publication import preflight_writable, publish_staging
from ..writer import compressor_from_spec
from ..runtime import configure_process_runtime, spawn_context
from ..system import effective_resource_budget
from .autotune import (
    resolve_auto_space_workers,
    resolve_auto_tile_size,
    resolve_owner_buffer_budget,
    resolve_auto_time_block,
)
from .grid import RESAMPLING_METHODS, build_target_grid, output_chunks
from .environment import validate_resampling_environment
from .inspection import inspect_resample_input
from .models import (
    ComputeDType,
    ResampleConfig,
    ResampleInspection,
    ResamplePlan,
    TargetGrid,
)
from .replacements import ReplacementRules, apply_replacement_rules, sample_statistics


class ResampleExecutionError(RuntimeError):
    """Raised when a resampling operation cannot safely complete."""
class _NativeResampleFallback(ResampleExecutionError):
    """Signal that the bounded native buffer bridge cannot represent this source safely."""



COMPUTE_DTYPES = ("source", "float32")
MAX_NATIVE_RESAMPLE_VALUES = 4_000_000


@dataclass(frozen=True)
class _OwnerTask:
    """One spatial output-chunk region and the arrays it exclusively owns."""

    region: tuple[int, int, int, int]
    item_names: tuple[str, ...]


@dataclass(frozen=True)
class _TileMetrics:
    """Timing and ownership data returned by one spatial worker."""

    task: _OwnerTask
    covered: bool
    elapsed: float
    weight_seconds: float = 0.0
    read_seconds: float = 0.0
    regrid_seconds: float = 0.0
    write_seconds: float = 0.0
    time_batches: int = 0
    owner_chunks: int = 0
    owner_buffer_bytes: int = 0
    owner_buffer_peak_bytes: int = 0
    owner_memmap_bytes: int = 0


def _empty_timing() -> dict[str, float]:
    return {"read": 0.0, "regrid": 0.0, "write": 0.0}


def _add_timing(total: dict[str, float], value: dict[str, float]) -> None:
    for name in total:
        total[name] += value[name]


def _format_tile_progress(completed: int, total: int, metrics: _TileMetrics) -> str:
    return (
        f"空间块进度：{completed}/{total} | 本块 {metrics.elapsed:.1f}s"
        f"（权重 {metrics.weight_seconds:.1f}s、读取 {metrics.read_seconds:.1f}s、"
        f"重采样 {metrics.regrid_seconds:.1f}s、写入 {metrics.write_seconds:.1f}s）"
    )


def _human_bytes(value: int | float) -> str:
    value = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(value) < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _is_zarr_v3(path: Path) -> bool:
    metadata = path / "zarr.json"
    if not metadata.is_file():
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("node_type") == "group" and value.get("zarr_format") == 3


def _prepare_target(path: Path, overwrite: bool) -> None:
    if path.is_symlink():
        raise ResampleExecutionError(f"拒绝将输出路径写入符号链接：{path}")
    if path.exists() and not path.is_dir():
        raise ResampleExecutionError(f"输出路径存在但不是目录：{path}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise ResampleExecutionError(
                f"输出目录非空：{path}；请确认覆盖或使用新的输出目录。"
            )
        if not _is_zarr_v3(path):
            raise ResampleExecutionError(
                "拒绝覆盖普通非空目录；只有已识别的 Zarr v3 目录可以覆盖。"
            )
    path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_temporary_root(
    source: Path,
    target: Path,
    temporary_dir: str | Path | None,
) -> Path:
    """Resolve the user-selected root for owner buffers and weights."""

    root = (
        target.parent
        if temporary_dir is None
        else Path(temporary_dir).expanduser().resolve()
    )
    if root == source or root == target:
        raise ResampleExecutionError("临时处理目录不能是输入或输出 Zarr 本身。")
    if root.is_relative_to(source) and root != source:
        raise ResampleExecutionError("临时处理目录不能位于输入 Zarr 内部。")
    if root.is_relative_to(target) and root != target:
        raise ResampleExecutionError("临时处理目录不能位于输出 Zarr 内部。")
    if root.exists() and not root.is_dir():
        raise ResampleExecutionError(f"临时处理路径不是目录：{root}")
    return root


def _publish_staging(staging: Path, target: Path, overwrite: bool) -> None:
    """Publish a completed store while keeping an existing output recoverable."""

    publish_staging(
        staging,
        target,
        "resample",
        overwrite=overwrite,
        require_zarr_v3=True,
    )


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _configure_runtime() -> None:
    """Keep ESMF and numerical libraries from multiplying threads."""

    configure_process_runtime()


def _grid_dataset(
    lat: np.ndarray,
    lon: np.ndarray,
    lat_bounds: np.ndarray,
    lon_bounds: np.ndarray,
    *,
    lat_attrs: dict | None = None,
    lon_attrs: dict | None = None,
) -> xr.Dataset:
    """Create the small xESMF grid description, including cell vertices."""

    lat_attrs = dict(lat_attrs or {})
    lon_attrs = dict(lon_attrs or {})
    lat_attrs.setdefault("standard_name", "latitude")
    lat_attrs.setdefault("units", "degrees_north")
    lon_attrs.setdefault("standard_name", "longitude")
    lon_attrs.setdefault("units", "degrees_east")
    return xr.Dataset(
        {
            "lat_b": xr.DataArray(np.asarray(lat_bounds), dims=("lat_b",)),
            "lon_b": xr.DataArray(np.asarray(lon_bounds), dims=("lon_b",)),
        },
        coords={
            "lat": xr.DataArray(np.asarray(lat), dims=("lat",), attrs=lat_attrs),
            "lon": xr.DataArray(np.asarray(lon), dims=("lon",), attrs=lon_attrs),
        },
    )


def _source_encoding(variable: xr.DataArray, item: VariableInfo) -> dict[str, object]:
    entry: dict[str, object] = {}
    for marker in ("_FillValue", "missing_value"):
        value = variable.encoding.get(marker)
        if value is None and marker in item.attrs:
            value = item.attrs[marker]
        if value is not None:
            entry[marker] = value
    if item.compressors:
        entry["compressors"] = list(item.compressors)
    return entry


def _clean_attrs(variable: xr.DataArray) -> None:
    attrs = dict(variable.attrs)
    attrs.pop("_FillValue", None)
    attrs.pop("missing_value", None)
    variable.attrs = attrs


def _missing_values(item: VariableInfo | None, variable: xr.DataArray) -> tuple[object, ...]:
    values: list[object] = []
    for marker in ("_FillValue", "missing_value"):
        value = variable.encoding.get(marker)
        if value is None:
            value = variable.attrs.get(marker)
        if value is None and item is not None:
            value = item.attrs.get(marker)
        if value is None:
            continue
        array = np.asarray(value).reshape(-1)
        values.extend(array.tolist())
    return tuple(values)


def _mask_missing(variable: xr.DataArray, item: VariableInfo | None) -> xr.DataArray:
    """Turn raw Zarr fill markers into NaN, which is what xESMF understands."""

    result = variable
    for value in _missing_values(item, variable):
        try:
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                # NaN is already the in-memory representation of a NaN fill
                # marker. There is no equality comparison that can mask it.
                continue
            else:
                result = result.where(result != value)
        except (TypeError, ValueError):
            continue
    return result


def _rename_xesmf_dims(variable: xr.DataArray) -> xr.DataArray:
    renames = {}
    for old, new in (("lat_new", "lat"), ("lon_new", "lon")):
        if old in variable.dims and new not in variable.dims:
            renames[old] = new
    return variable.rename(renames) if renames else variable


def _expected_output_dtype(source: xr.DataArray, result: xr.DataArray) -> np.dtype:
    source_dtype = np.dtype(source.dtype)
    if np.issubdtype(source_dtype, np.floating):
        return source_dtype
    return np.dtype(result.dtype)


def _effective_compute_dtype(
    variable: xr.DataArray,
    compute_dtype: ComputeDType,
) -> np.dtype:
    """Return the dtype used for floating-point resampling input.

    The optional float32 mode deliberately applies only to floating variables.
    Integer variables remain on the existing floating-output path; silently
    converting them to a smaller integer type would truncate interpolation
    results or overflow source values.
    """

    dtype = np.dtype(variable.dtype)
    if compute_dtype == "float32" and np.issubdtype(dtype, np.floating):
        return np.dtype("float32")
    return dtype


def _build_output_encoding(
    source: xr.Dataset,
    info: DatasetInfo,
    chunks: dict[str, tuple[int, ...]],
    output: xr.Dataset,
    output_layout=None,
) -> dict[str, dict[str, object]]:
    by_name = {item.name: item for item in info.variables}
    encoding: dict[str, dict[str, object]] = {}
    for name, variable in output.variables.items():
        item = by_name.get(name)
        if item is None:
            continue
        entry = _source_encoding(source[name], item)
        if (
            {"lat", "lon"}.issubset(item.dims)
            and np.issubdtype(variable.dtype, np.floating)
        ):
            # Interpolation represents every missing result as NaN, including
            # integer sources promoted to floating output.  Retaining a
            # source ``missing_value=-9999`` beside Zarr's NaN fill marker is
            # both semantically wrong and rejected by xarray's CF encoder.
            entry["_FillValue"] = np.asarray(np.nan, dtype=variable.dtype).item()
            entry.pop("missing_value", None)
        if variable.ndim:
            entry["chunks"] = chunks[name]
        if output_layout is not None:
            try:
                layout_item = output_layout.for_output(name)
            except KeyError:
                codec = output_layout.coordinate_codec if item.is_coord else None
            else:
                codec = layout_item.codec
            # ``None`` means this layout changes chunks only. Preserve the
            # existing source codec instead of silently writing uncompressed.
            if codec is not None:
                compressor = compressor_from_spec(codec)
                entry["compressors"] = [compressor]
        encoding[name] = entry
    return encoding


def _build_regridder(
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    source_lat_bounds: np.ndarray,
    source_lon_bounds: np.ndarray,
    target: TargetGrid,
    method: str,
    workdir: Path,
    *,
    lat_attrs: dict | None = None,
    lon_attrs: dict | None = None,
    periodic: bool = False,
):
    _configure_runtime()
    import xesmf as xe

    source_grid = _grid_dataset(
        source_lat,
        source_lon,
        source_lat_bounds,
        source_lon_bounds,
        lat_attrs=lat_attrs,
        lon_attrs=lon_attrs,
    )
    target_grid = _grid_dataset(target.lat, target.lon, target.lat_bounds, target.lon_bounds)
    weight_path = workdir / f"weights-{method}-{uuid4().hex}.nc"
    try:
        regridder = xe.Regridder(
            source_grid,
            target_grid,
            method=method,
            periodic=periodic,
            filename=str(weight_path),
            reuse_weights=False,
            unmapped_to_nan=True,
            ignore_degenerate=True,
            input_dims=("lat", "lon"),
        )
    finally:
        # xESMF copies the grid metadata while constructing the regridder;
        # close the temporary xarray datasets even if weight construction
        # fails halfway through.
        source_grid.close()
        target_grid.close()
    return regridder, weight_path


def _tile_ranges(size: int, tile_size: int):
    for start in range(0, int(size), int(tile_size)):
        yield start, min(start + int(tile_size), int(size))






def _source_overlap_window(
    source_bounds: np.ndarray,
    target_bounds: np.ndarray,
    *,
    halo: int = 0,
) -> slice | None:
    """Find a contiguous source-cell window intersecting a target tile."""

    source_bounds = np.asarray(source_bounds, dtype="float64")
    target_bounds = np.asarray(target_bounds, dtype="float64")
    low = float(np.min(target_bounds))
    high = float(np.max(target_bounds))
    cell_low = np.minimum(source_bounds[:-1], source_bounds[1:])
    cell_high = np.maximum(source_bounds[:-1], source_bounds[1:])
    tolerance = max(abs(high - low) * 1e-12, 1e-12)
    overlap = (cell_high > low - tolerance) & (cell_low < high + tolerance)
    indices = np.flatnonzero(overlap)
    if indices.size == 0:
        return None
    start = max(0, int(indices[0]) - halo)
    stop = min(source_bounds.size - 1, int(indices[-1]) + halo + 1)
    return slice(start, stop)


def _source_halo(method: str) -> int:
    """Return the smallest source-cell halo for the xESMF stencil."""

    return {
        "bilinear": 1,
        "nearest_s2d": 1,
        "conservative": 0,
        "conservative_normed": 0,
        "patch": 2,
        # nearest_d2s uses the full source grid and does not call this path.
        "nearest_d2s": 0,
    }.get(method, 1)


def _tile_target(
    target: TargetGrid,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
) -> TargetGrid:
    return TargetGrid(
        lat=target.lat[lat_start:lat_stop],
        lon=target.lon[lon_start:lon_stop],
        lat_bounds=target.lat_bounds[lat_start : lat_stop + 1],
        lon_bounds=target.lon_bounds[lon_start : lon_stop + 1],
        lat_resolution=target.lat_resolution,
        lon_resolution=target.lon_resolution,
        extent=target.extent,
    )


def _shift_target_longitude(target: TargetGrid, shift: float) -> TargetGrid:
    if shift == 0:
        return target
    return TargetGrid(
        lat=target.lat,
        lon=target.lon + shift,
        lat_bounds=target.lat_bounds,
        lon_bounds=target.lon_bounds + shift,
        lat_resolution=target.lat_resolution,
        lon_resolution=target.lon_resolution,
        extent=target.extent,
    )


def _resolve_local_source_window(
    grid: GridInfo,
    tile: TargetGrid,
    method: str,
) -> tuple[TargetGrid, slice | None, slice | None]:
    """Return a local source window and a target tile in the same longitude frame."""

    halo = _source_halo(method)
    lat_slice = _source_overlap_window(
        grid.lat_bounds,
        tile.lat_bounds,
        halo=halo,
    )
    lon_slice = _source_overlap_window(
        grid.lon_bounds,
        tile.lon_bounds,
        halo=halo,
    )
    if lon_slice is not None or not grid.periodic:
        # A target tile whose bounds are already in this longitude frame does
        # not cross the periodic seam merely because it touches the first or
        # last source cell.  Expanding such edge tiles to the whole 360-degree
        # axis makes a local conservative calculation several times larger
        # without changing its result.  A genuinely different longitude frame
        # is handled by the shift path below.
        return tile, lat_slice, lon_slice

    # ``extent="global"`` uses -180..180 even when the source uses 0..360.
    # For a tile that lies wholly outside the source's longitude convention,
    # shift the target by one or more full revolutions before finding its
    # local source window.  Normal source-extent jobs take the fast path above.
    source_center = float(np.mean([np.min(grid.lon_bounds), np.max(grid.lon_bounds)]))
    target_center = float(np.mean([np.min(tile.lon_bounds), np.max(tile.lon_bounds)]))
    nearest_shift = int(round((source_center - target_center) / 360.0))
    for revolution in (nearest_shift, nearest_shift - 1, nearest_shift + 1):
        shifted = _shift_target_longitude(tile, 360.0 * revolution)
        candidate = _source_overlap_window(
            grid.lon_bounds,
            shifted.lon_bounds,
            halo=halo,
        )
        if candidate is not None:
            return shifted, lat_slice, candidate
    return tile, lat_slice, None


class _FullGridTileRegridder:
    """Apply a full-grid weight matrix to one target tile.

    ``nearest_d2s`` assigns source points to destination points globally, so
    independently constructing a regridder for each tile changes the answer.
    xESMF exposes the sparse weights and the low-level application routine;
    slicing only the output rows preserves the full-grid semantics while
    keeping the data computation tile-sized.
    """

    def __init__(self, regridder, lat_slice: slice, lon_slice: slice) -> None:
        self._regridder = regridder
        full_weights = regridder.weights.data.reshape(
            regridder.shape_out + regridder.shape_in
        )
        self._weights = full_weights[lat_slice, lon_slice, :, :]
        self._shape_out = (
            int(lat_slice.stop - lat_slice.start),
            int(lon_slice.stop - lon_slice.start),
        )

    def __call__(
        self,
        data: xr.DataArray,
        *,
        keep_attrs: bool = False,
        skipna: bool,
        na_thres: float,
        output_chunks: dict[str, int] | None = None,
    ) -> xr.DataArray:
        del keep_attrs, output_chunks
        data_dims = tuple(dim for dim in data.dims if dim not in {"lat", "lon"}) + (
            "lat",
            "lon",
        )
        values = self._regridder._regrid(
            data.transpose(*data_dims).data,
            self._weights,
            shape_in=self._regridder.shape_in,
            shape_out=self._shape_out,
            skipna=skipna,
            na_thres=na_thres,
        )
        return xr.DataArray(values, dims=data_dims)


def _resampled_dtype(variable: xr.DataArray) -> np.dtype:
    """Use a floating result for discrete fields instead of truncating interpolation."""

    dtype = np.dtype(variable.dtype)
    return dtype if np.issubdtype(dtype, np.floating) else np.dtype("float64")


def _resampled_output_dtype(
    variable: xr.DataArray,
    compute_dtype: ComputeDType,
) -> np.dtype:
    dtype = np.dtype(variable.dtype)
    if np.issubdtype(dtype, np.floating):
        return _effective_compute_dtype(variable, compute_dtype)
    return _resampled_dtype(variable)


def _build_output_skeleton(
    source: xr.Dataset,
    info: DatasetInfo,
    plan: ResamplePlan,
) -> tuple[xr.Dataset, dict[str, dict[str, object]], set[str]]:
    target = plan.target
    output_variables: dict[str, xr.DataArray] = {}
    spatial_names: set[str] = set()
    for item in info.data_variables:
        source_variable = source[item.name]
        if item.ndim and {"lat", "lon"}.issubset(source_variable.dims):
            shape = tuple(
                target.dimensions.get(dim, int(size))
                for dim, size in zip(source_variable.dims, source_variable.shape)
            )
            attrs = dict(source_variable.attrs)
            attrs.update(
                {
                    "resampling_method": plan.method,
                    "resampling_skipna": bool(plan.skipna),
                    "resampling_compute_dtype": plan.compute_dtype,
                }
            )
            output_variables[item.name] = xr.DataArray(
                da.empty(
                    shape,
                    chunks=plan.output_chunks[item.name],
                    dtype=_resampled_output_dtype(
                        source_variable,
                        plan.compute_dtype,
                    ),
                ),
                dims=source_variable.dims,
                attrs=attrs,
            )
            spatial_names.add(item.name)
        else:
            variable = source_variable.copy(deep=False)
            variable.attrs = dict(source_variable.attrs)
            output_variables[item.name] = variable

    coords: dict[str, xr.DataArray] = {
        "lat": xr.DataArray(
            target.lat,
            dims=("lat",),
            attrs={**dict(source.lat.attrs), "resampling_resolution": target.lat_resolution},
        ),
        "lon": xr.DataArray(
            target.lon,
            dims=("lon",),
            attrs={**dict(source.lon.attrs), "resampling_resolution": target.lon_resolution},
        ),
    }
    for name, coordinate in source.coords.items():
        if name in {"lat", "lon"}:
            continue
        if set(coordinate.dims).intersection({"lat", "lon"}):
            continue
        coords[name] = coordinate.copy(deep=False)

    output = xr.Dataset(output_variables, coords=coords, attrs=dict(source.attrs))
    output.attrs.update(
        {
            "resampling_method": plan.method,
            "resampling_skipna": bool(plan.skipna),
            "resampling_na_thres": float(plan.na_thres),
            "resampling_compute_dtype": plan.compute_dtype,
            "resampling_lat_resolution": float(target.lat_resolution),
            "resampling_lon_resolution": float(target.lon_resolution),
            "source_spatial_extent": tuple(
                float(value) for value in plan.inspection.grid.source_extent
            ),
            "target_spatial_extent": tuple(
                float(value) for value in target.spatial_extent
            ),
            "resampling_extrapolation": "disabled",
            "resampling_before_replacements": json.dumps(
                plan.before_replacements.as_pairs(), ensure_ascii=False
            ),
            "resampling_after_replacements": json.dumps(
                plan.after_replacements.as_pairs(), ensure_ascii=False
            ),
            "resampling_statistics_policy": plan.statistics_policy,
        }
    )
    output = sanitize_cf_references(output)
    for variable in output.variables.values():
        _clean_attrs(variable)
    encoding = _build_output_encoding(
        source,
        info,
        plan.output_chunks,
        output,
        plan.output_layout,
    )
    return output, encoding, spatial_names


def _time_slices(variable: VariableInfo, time_block: int):
    time_index = variable.dims.index("time")
    size = int(variable.shape[time_index])
    # ``time_block`` is a computation batch, not a Zarr storage chunk.  It is
    # intentionally allowed to span the source time chunks so a GeoTIFF stack
    # written as time=1 can still use vectorized xESMF calls.
    step = max(1, min(int(time_block), size))
    return ((start, min(start + step, size)) for start in range(0, size, step))

def _owner_task_groups(
    spatial_items: tuple[VariableInfo, ...],
    plan: ResamplePlan,
) -> tuple[tuple[int, int, tuple[str, ...]], ...]:
    """Group arrays only when their physical spatial chunk grids coincide."""

    grouped: dict[tuple[int, int], list[str]] = {}
    for item in spatial_items:
        chunks = plan.output_chunks[item.name]
        key = (
            int(chunks[item.dims.index("lat")]),
            int(chunks[item.dims.index("lon")]),
        )
        grouped.setdefault(key, []).append(item.name)
    return tuple(
        (lat_chunk, lon_chunk, tuple(names))
        for (lat_chunk, lon_chunk), names in grouped.items()
    )


def _owner_tasks(
    groups: tuple[tuple[int, int, tuple[str, ...]], ...],
    target: TargetGrid,
) -> Iterable[_OwnerTask]:
    """Yield tasks whose regions each cover one physical spatial chunk."""

    for lat_chunk, lon_chunk, item_names in groups:
        for lat_start, lat_stop in _tile_ranges(target.lat.size, lat_chunk):
            for lon_start, lon_stop in _tile_ranges(target.lon.size, lon_chunk):
                yield _OwnerTask(
                    (lat_start, lat_stop, lon_start, lon_stop),
                    item_names,
                )


def _owner_task_count(
    groups: tuple[tuple[int, int, tuple[str, ...]], ...],
    target: TargetGrid,
) -> int:
    return sum(
        ((int(target.lat.size) + lat_chunk - 1) // lat_chunk)
        * ((int(target.lon.size) + lon_chunk - 1) // lon_chunk)
        for lat_chunk, lon_chunk, _item_names in groups
    )


def _owner_item_batch_count(
    item_names: tuple[str, ...],
    items_by_name: dict[str, VariableInfo],
    plan: ResamplePlan,
) -> int:
    total = 0
    for name in item_names:
        item = items_by_name[name]
        time_axis = item.dims.index("time")
        time_size = int(item.shape[time_axis])
        time_chunk = int(plan.output_chunks[name][time_axis])
        for start in range(0, time_size, time_chunk):
            length = min(time_chunk, time_size - start)
            total += (length + int(plan.time_block) - 1) // int(plan.time_block)
    return total

def _owner_total_batch_count(
    groups: tuple[tuple[int, int, tuple[str, ...]], ...],
    target: TargetGrid,
    items_by_name: dict[str, VariableInfo],
    plan: ResamplePlan,
) -> int:
    total = 0
    for lat_chunk, lon_chunk, item_names in groups:
        spatial_count = (
            (int(target.lat.size) + lat_chunk - 1) // lat_chunk
        ) * (
            (int(target.lon.size) + lon_chunk - 1) // lon_chunk
        )
        batches = _owner_item_batch_count(item_names, items_by_name, plan)
        total += spatial_count * batches
    return total


@contextmanager
def _owner_buffer(
    shape: tuple[int, ...],
    dtype: np.dtype,
    workdir: Path,
    heap_budget_bytes: int,
):
    """Allocate one bounded final-chunk buffer, spilling safely to a memmap."""

    dtype = np.dtype(dtype)
    nbytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    values: np.ndarray | np.memmap | None = None
    path: Path | None = None
    try:
        if nbytes <= max(0, int(heap_budget_bytes)):
            try:
                values = np.empty(shape, dtype=dtype)
            except MemoryError:
                values = None
        if values is None:
            path = workdir / f".resample-owner-{uuid4().hex}.bin"
            values = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
        yield values, nbytes, path is not None
    finally:
        try:
            if isinstance(values, np.memmap):
                try:
                    values.flush()
                finally:
                    mapping = getattr(values, "_mmap", None)
                    if mapping is not None:
                        mapping.close()
        finally:
            del values
            if path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass


def _avoids_intermediate_store(info: DatasetInfo, plan: ResamplePlan) -> bool:
    """Return whether owner buffers avoid a former batch-layout Zarr write."""

    for item in info.data_variables:
        chunks = plan.output_chunks.get(item.name)
        if chunks is None or "time" not in item.dims:
            continue
        if "lat" not in item.dims or "lon" not in item.dims:
            continue
        time_axis = item.dims.index("time")
        final_chunk = min(int(chunks[time_axis]), int(item.shape[time_axis]))
        if int(plan.time_block) < final_chunk:
            return True
    return False




def _chunk_regions(
    shape: tuple[int, ...],
    chunks: tuple[int, ...],
):
    starts = [range(0, int(size), max(1, int(chunk))) for size, chunk in zip(shape, chunks)]
    for origin in itertools.product(*starts):
        yield tuple(
            slice(int(start), min(int(start) + int(chunk), int(size)))
            for start, chunk, size in zip(origin, chunks, shape)
        )








def _initialize_output_store(
    source: xr.Dataset,
    info: DatasetInfo,
    plan: ResamplePlan,
    path: Path,
) -> set[str]:
    output, encoding, spatial_names = _build_output_skeleton(source, info, plan)
    try:
        delayed = output.to_zarr(
            path,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding=encoding,
            compute=False,
            safe_chunks=False,
        )
        static = output.drop_vars(spatial_names)
        static.to_zarr(
            path,
            mode="r+",
            consolidated=False,
            compute=True,
            safe_chunks=False,
        )
        static.close()
        del static, delayed
    finally:
        output.close()
    return spatial_names


def _write_region(
    output_group,
    name: str,
    values: np.ndarray,
    dims: tuple[str, ...],
    region: dict[str, slice],
) -> None:
    array = output_group[name]
    array[tuple(region[dim] for dim in dims)] = values


def _fill_missing_tile(
    output_group,
    spatial_items: tuple[VariableInfo, ...],
    plan: ResamplePlan,
    task: _OwnerTask,
    workdir: Path,
) -> dict[str, float | int]:
    """Materialize uncovered final chunks without an unbounded full-time array."""

    lat_start, lat_stop, lon_start, lon_stop = task.region
    write_seconds = 0.0
    time_batches = 0
    owner_chunks = 0
    owner_buffer_bytes = 0
    owner_buffer_peak_bytes = 0
    owner_memmap_bytes = 0
    for item in spatial_items:
        dtype = np.dtype(item.dtype)
        if not np.issubdtype(dtype, np.floating):
            dtype = np.dtype("float64")
        elif plan.compute_dtype == "float32":
            dtype = np.dtype("float32")
        time_axis = item.dims.index("time")
        time_size = int(item.shape[time_axis])
        time_chunk = int(plan.output_chunks[item.name][time_axis])
        for time_start in range(0, time_size, time_chunk):
            time_stop = min(time_start + time_chunk, time_size)
            shape = tuple(
                {
                    "time": time_stop - time_start,
                    "lat": lat_stop - lat_start,
                    "lon": lon_stop - lon_start,
                }[dim]
                for dim in item.dims
            )
            with _owner_buffer(
                shape,
                dtype,
                workdir,
                plan.owner_buffer_budget_bytes,
            ) as (values, nbytes, used_memmap):
                values[...] = np.nan
                write_started = time.perf_counter()
                _write_region(
                    output_group,
                    item.name,
                    values,
                    item.dims,
                    {
                        "time": slice(time_start, time_stop),
                        "lat": slice(lat_start, lat_stop),
                        "lon": slice(lon_start, lon_stop),
                    },
                )
                write_seconds += time.perf_counter() - write_started
            owner_chunks += 1
            owner_buffer_bytes += nbytes
            owner_buffer_peak_bytes = max(owner_buffer_peak_bytes, nbytes)
            if used_memmap:
                owner_memmap_bytes += nbytes
            length = time_stop - time_start
            time_batches += (length + int(plan.time_block) - 1) // int(plan.time_block)
    return {
        "write": write_seconds,
        "time_batches": time_batches,
        "owner_chunks": owner_chunks,
        "owner_buffer_bytes": owner_buffer_bytes,
        "owner_buffer_peak_bytes": owner_buffer_peak_bytes,
        "owner_memmap_bytes": owner_memmap_bytes,
    }


def _resample_tile_variable(
    source: xr.Dataset,
    item: VariableInfo,
    regridder,
    output_group,
    target: TargetGrid,
    lat_start: int,
    lat_stop: int,
    lon_start: int,
    lon_stop: int,
    source_lat_slice: slice,
    source_lon_slice: slice,
    target_tile: TargetGrid | None,
    skipna: bool,
    na_thres: float,
    time_block: int,
    compute_workers: int,
    compute_dtype: ComputeDType,
    before_replacements,
    after_replacements,
    statistics: dict[str, float],
    workdir: Path,
    owner_buffer_budget_bytes: int,
) -> dict[str, float | int]:
    source_variable = source[item.name]
    prepared = _mask_missing(source_variable, item)
    effective_dtype = _effective_compute_dtype(source_variable, compute_dtype)
    if np.dtype(prepared.dtype) != effective_dtype:
        prepared = prepared.astype(effective_dtype)
    expected_dims = tuple(source_variable.dims)
    input_dims = tuple(dim for dim in expected_dims if dim not in {"lat", "lon"}) + (
        "lat",
        "lon",
    )
    target_chunk = target_tile or _tile_target(
        target,
        lat_start,
        lat_stop,
        lon_start,
        lon_stop,
    )
    lat_slice = slice(lat_start, lat_stop)
    lon_slice = slice(lon_start, lon_stop)
    timing = _empty_timing()
    output_chunks = tuple(int(value) for value in output_group[item.name].chunks)
    output_shape = tuple(int(value) for value in output_group[item.name].shape)
    for dimension, start, stop in (
        ("lat", lat_start, lat_stop),
        ("lon", lon_start, lon_stop),
    ):
        axis = expected_dims.index(dimension)
        chunk = output_chunks[axis]
        if start % chunk or (stop != output_shape[axis] and stop % chunk):
            raise ResampleExecutionError(
                f"owner task 非完整物理 chunk 边界：{item.name}/{dimension}={start}:{stop}。"
            )

    result_dtype = _resampled_output_dtype(source_variable, compute_dtype)

    def resample_batch(time_start: int, time_stop: int) -> np.ndarray:
        read_started = time.perf_counter()
        subset = prepared.isel(
            time=slice(time_start, time_stop),
            lat=source_lat_slice,
            lon=source_lon_slice,
        ).transpose(*input_dims)
        if hasattr(subset.data, "compute"):
            with dask.config.set(
                scheduler="threads",
                num_workers=max(1, int(compute_workers)),
            ):
                subset = subset.compute()
        if before_replacements.rules:
            subset = subset.copy(
                data=apply_replacement_rules(
                    np.asarray(subset.data), before_replacements, statistics
                )
            )
        timing["read"] += time.perf_counter() - read_started
        regrid_started = time.perf_counter()
        result = regridder(
            subset,
            keep_attrs=False,
            skipna=skipna,
            na_thres=na_thres,
            output_chunks={"lat": target_chunk.lat.size, "lon": target_chunk.lon.size},
        )
        result = _rename_xesmf_dims(result).transpose(*expected_dims)
        values = result.data
        with dask.config.set(
            scheduler="threads",
            num_workers=max(1, int(compute_workers)),
        ):
            if hasattr(values, "compute"):
                values = values.compute()
        timing["regrid"] += time.perf_counter() - regrid_started
        values = np.asarray(values, dtype=result_dtype)
        if after_replacements.rules:
            values = apply_replacement_rules(values, after_replacements, statistics)
        del subset, result
        return values

    time_axis = expected_dims.index("time")
    time_size = int(item.shape[time_axis])
    final_time_chunk = output_chunks[time_axis]
    owner_chunks = 0
    owner_buffer_bytes = 0
    owner_buffer_peak_bytes = 0
    owner_memmap_bytes = 0
    time_batches = 0
    for chunk_start in range(0, time_size, final_time_chunk):
        chunk_stop = min(chunk_start + final_time_chunk, time_size)
        chunk_length = chunk_stop - chunk_start
        chunk_shape = tuple(
            {
                "time": chunk_length,
                "lat": lat_stop - lat_start,
                "lon": lon_stop - lon_start,
            }[dim]
            for dim in expected_dims
        )
        chunk_bytes = int(np.prod(chunk_shape, dtype=np.int64)) * result_dtype.itemsize
        region = {
            "time": slice(chunk_start, chunk_stop),
            "lat": lat_slice,
            "lon": lon_slice,
        }
        if chunk_length <= int(time_block):
            values = resample_batch(chunk_start, chunk_stop)
            time_batches += 1
            write_started = time.perf_counter()
            _write_region(output_group, item.name, values, expected_dims, region)
            timing["write"] += time.perf_counter() - write_started
            del values
        else:
            with _owner_buffer(
                chunk_shape,
                result_dtype,
                workdir,
                owner_buffer_budget_bytes,
            ) as (buffer, _nbytes, used_memmap):
                for time_start in range(chunk_start, chunk_stop, int(time_block)):
                    time_stop = min(time_start + int(time_block), chunk_stop)
                    values = resample_batch(time_start, time_stop)
                    relative = {
                        "time": slice(time_start - chunk_start, time_stop - chunk_start),
                        "lat": slice(None),
                        "lon": slice(None),
                    }
                    buffer[tuple(relative[dim] for dim in expected_dims)] = values
                    del values
                    time_batches += 1
                write_started = time.perf_counter()
                _write_region(output_group, item.name, buffer, expected_dims, region)
                timing["write"] += time.perf_counter() - write_started
            if used_memmap:
                owner_memmap_bytes += chunk_bytes
        owner_chunks += 1
        owner_buffer_bytes += chunk_bytes
        owner_buffer_peak_bytes = max(owner_buffer_peak_bytes, chunk_bytes)
    return {
        **timing,
        "time_batches": time_batches,
        "owner_chunks": owner_chunks,
        "owner_buffer_bytes": owner_buffer_bytes,
        "owner_buffer_peak_bytes": owner_buffer_peak_bytes,
        "owner_memmap_bytes": owner_memmap_bytes,
    }


_SPACE_WORKER_STATE: dict[str, object] = {}


def _initialize_space_worker(
    source_path: str,
    staging_path: str,
    plan: ResamplePlan,
    spatial_items: tuple[VariableInfo, ...],
    workdir: str,
) -> None:
    """Open one input/output handle per spawned spatial worker."""

    _configure_runtime()
    source = xr.open_zarr(
        source_path,
        consolidated=False,
        chunks={},
        decode_times=False,
        mask_and_scale=False,
    )
    _SPACE_WORKER_STATE.clear()
    _SPACE_WORKER_STATE.update(
        source=source,
        output_group=zarr.open_group(staging_path, mode="r+"),
        plan=plan,
        items_by_name={item.name: item for item in spatial_items},
        workdir=Path(workdir),
    )


def _process_space_tile(task: _OwnerTask) -> _TileMetrics:
    """Process one exclusively owned final spatial-chunk region."""

    source = _SPACE_WORKER_STATE["source"]
    output_group = _SPACE_WORKER_STATE["output_group"]
    plan = _SPACE_WORKER_STATE["plan"]
    items_by_name = _SPACE_WORKER_STATE["items_by_name"]
    workdir = _SPACE_WORKER_STATE["workdir"]
    assert isinstance(source, xr.Dataset)
    assert isinstance(output_group, zarr.Group)
    assert isinstance(plan, ResamplePlan)
    assert isinstance(items_by_name, dict)
    assert isinstance(workdir, Path)
    spatial_items = tuple(items_by_name[name] for name in task.item_names)

    started = time.perf_counter()
    lat_start, lat_stop, lon_start, lon_stop = task.region
    tile = _tile_target(plan.target, lat_start, lat_stop, lon_start, lon_stop)
    target_tile, source_lat_slice, source_lon_slice = _resolve_local_source_window(
        plan.inspection.grid,
        tile,
        plan.method,
    )
    if source_lat_slice is None or source_lon_slice is None:
        missing = _fill_missing_tile(output_group, spatial_items, plan, task, workdir)
        return _TileMetrics(
            task=task,
            covered=False,
            elapsed=time.perf_counter() - started,
            write_seconds=float(missing["write"]),
            time_batches=int(missing["time_batches"]),
            owner_chunks=int(missing["owner_chunks"]),
            owner_buffer_bytes=int(missing["owner_buffer_bytes"]),
            owner_buffer_peak_bytes=int(missing["owner_buffer_peak_bytes"]),
            owner_memmap_bytes=int(missing["owner_memmap_bytes"]),
        )

    grid = plan.inspection.grid
    regridder = None
    weight_path = None
    timing = _empty_timing()
    weight_seconds = 0.0
    time_batches = 0
    owner_chunks = 0
    owner_buffer_bytes = 0
    owner_buffer_peak_bytes = 0
    owner_memmap_bytes = 0
    try:
        lat_start_source = int(source_lat_slice.start)
        lat_stop_source = int(source_lat_slice.stop)
        lon_start_source = int(source_lon_slice.start)
        lon_stop_source = int(source_lon_slice.stop)
        weights_started = time.perf_counter()
        regridder, weight_path = _build_regridder(
            grid.lat[source_lat_slice],
            grid.lon[source_lon_slice],
            grid.lat_bounds[lat_start_source : lat_stop_source + 1],
            grid.lon_bounds[lon_start_source : lon_stop_source + 1],
            target_tile,
            plan.method,
            workdir,
            lat_attrs=dict(source.lat.attrs),
            lon_attrs=dict(source.lon.attrs),
            periodic=bool(
                grid.periodic
                and lon_start_source == 0
                and lon_stop_source == grid.lon.size
            ),
        )
        weight_seconds = time.perf_counter() - weights_started
        for item in spatial_items:
            variable_metrics = _resample_tile_variable(
                source,
                item,
                regridder,
                output_group,
                plan.target,
                lat_start,
                lat_stop,
                lon_start,
                lon_stop,
                source_lat_slice,
                source_lon_slice,
                target_tile,
                plan.skipna,
                plan.na_thres,
                plan.time_block,
                plan.compute_workers,
                plan.compute_dtype,
                plan.before_replacements,
                plan.after_replacements,
                plan.statistics.get(item.name, {}),
                workdir,
                plan.owner_buffer_budget_bytes,
            )
            _add_timing(timing, variable_metrics)
            time_batches += int(variable_metrics["time_batches"])
            owner_chunks += int(variable_metrics["owner_chunks"])
            owner_buffer_bytes += int(variable_metrics["owner_buffer_bytes"])
            owner_buffer_peak_bytes = max(
                owner_buffer_peak_bytes,
                int(variable_metrics["owner_buffer_peak_bytes"]),
            )
            owner_memmap_bytes += int(variable_metrics["owner_memmap_bytes"])
        return _TileMetrics(
            task=task,
            covered=True,
            elapsed=time.perf_counter() - started,
            weight_seconds=weight_seconds,
            read_seconds=timing["read"],
            regrid_seconds=timing["regrid"],
            write_seconds=timing["write"],
            time_batches=time_batches,
            owner_chunks=owner_chunks,
            owner_buffer_bytes=owner_buffer_bytes,
            owner_buffer_peak_bytes=owner_buffer_peak_bytes,
            owner_memmap_bytes=owner_memmap_bytes,
        )
    finally:
        del regridder
        if weight_path is not None:
            try:
                weight_path.unlink()
            except OSError:
                pass


def _execute_serial_tile(
    source: xr.Dataset,
    output_group,
    plan: ResamplePlan,
    task: _OwnerTask,
    workdir: Path,
    spatial_items: tuple[VariableInfo, ...],
    full_regridder=None,
) -> _TileMetrics:
    """Execute one exclusive owner task in the caller."""

    items_by_name = {item.name: item for item in spatial_items}
    task_items = tuple(items_by_name[name] for name in task.item_names)
    started = time.perf_counter()
    lat_start, lat_stop, lon_start, lon_stop = task.region
    tile = _tile_target(plan.target, lat_start, lat_stop, lon_start, lon_stop)
    grid = plan.inspection.grid
    if full_regridder is not None:
        target_tile = tile
        source_lat_slice = slice(0, grid.lat.size)
        source_lon_slice = slice(0, grid.lon.size)
        regridder = _FullGridTileRegridder(
            full_regridder,
            slice(lat_start, lat_stop),
            slice(lon_start, lon_stop),
        )
        weight_path = None
    else:
        target_tile, source_lat_slice, source_lon_slice = _resolve_local_source_window(
            grid,
            tile,
            plan.method,
        )
        if source_lat_slice is None or source_lon_slice is None:
            missing = _fill_missing_tile(output_group, task_items, plan, task, workdir)
            return _TileMetrics(
                task=task,
                covered=False,
                elapsed=time.perf_counter() - started,
                write_seconds=float(missing["write"]),
                time_batches=int(missing["time_batches"]),
                owner_chunks=int(missing["owner_chunks"]),
                owner_buffer_bytes=int(missing["owner_buffer_bytes"]),
                owner_buffer_peak_bytes=int(missing["owner_buffer_peak_bytes"]),
                owner_memmap_bytes=int(missing["owner_memmap_bytes"]),
            )
        lat_start_source = int(source_lat_slice.start)
        lat_stop_source = int(source_lat_slice.stop)
        lon_start_source = int(source_lon_slice.start)
        lon_stop_source = int(source_lon_slice.stop)
        weights_started = time.perf_counter()
        regridder, weight_path = _build_regridder(
            grid.lat[source_lat_slice],
            grid.lon[source_lon_slice],
            grid.lat_bounds[lat_start_source : lat_stop_source + 1],
            grid.lon_bounds[lon_start_source : lon_stop_source + 1],
            target_tile,
            plan.method,
            workdir,
            lat_attrs=dict(source.lat.attrs),
            lon_attrs=dict(source.lon.attrs),
            periodic=bool(
                grid.periodic
                and lon_start_source == 0
                and lon_stop_source == grid.lon.size
            ),
        )
        weight_seconds = time.perf_counter() - weights_started
    timing = _empty_timing()
    time_batches = 0
    owner_chunks = 0
    owner_buffer_bytes = 0
    owner_buffer_peak_bytes = 0
    owner_memmap_bytes = 0
    try:
        for item in task_items:
            variable_metrics = _resample_tile_variable(
                source,
                item,
                regridder,
                output_group,
                plan.target,
                lat_start,
                lat_stop,
                lon_start,
                lon_stop,
                source_lat_slice,
                source_lon_slice,
                target_tile,
                plan.skipna,
                plan.na_thres,
                plan.time_block,
                plan.compute_workers,
                plan.compute_dtype,
                plan.before_replacements,
                plan.after_replacements,
                plan.statistics.get(item.name, {}),
                workdir,
                plan.owner_buffer_budget_bytes,
            )
            _add_timing(timing, variable_metrics)
            time_batches += int(variable_metrics["time_batches"])
            owner_chunks += int(variable_metrics["owner_chunks"])
            owner_buffer_bytes += int(variable_metrics["owner_buffer_bytes"])
            owner_buffer_peak_bytes = max(
                owner_buffer_peak_bytes,
                int(variable_metrics["owner_buffer_peak_bytes"]),
            )
            owner_memmap_bytes += int(variable_metrics["owner_memmap_bytes"])
        return _TileMetrics(
            task=task,
            covered=True,
            elapsed=time.perf_counter() - started,
            weight_seconds=weight_seconds if full_regridder is None else 0.0,
            read_seconds=timing["read"],
            regrid_seconds=timing["regrid"],
            write_seconds=timing["write"],
            time_batches=time_batches,
            owner_chunks=owner_chunks,
            owner_buffer_bytes=owner_buffer_bytes,
            owner_buffer_peak_bytes=owner_buffer_peak_bytes,
            owner_memmap_bytes=owner_memmap_bytes,
        )
    finally:
        del regridder
        if weight_path is not None:
            try:
                weight_path.unlink()
            except OSError:
                pass


def _run_parallel_tiles(
    source_path: Path,
    staging: Path,
    workdir: Path,
    plan: ResamplePlan,
    spatial_items: tuple[VariableInfo, ...],
    tasks: Iterable[_OwnerTask],
    total_tasks: int,
    total_batches: int,
    *,
    cancel_event=None,
    progress: bool,
) -> dict[str, float | int]:
    """Run bounded owner tasks and aggregate progress once per completed task."""

    context = spawn_context()
    completed_batches = 0
    report_interval = max(1, total_batches // 20) if total_batches else 1
    next_report = report_interval
    executor = ProcessPoolExecutor(
        max_workers=max(1, int(plan.space_workers)),
        mp_context=context,
        initializer=_initialize_space_worker,
        initargs=(
            str(source_path),
            str(staging),
            plan,
            spatial_items,
            str(workdir),
        ),
    )
    pending = set()
    task_iter = iter(tasks)
    completed = 0
    cancelled = False
    timing = _empty_timing()
    weight_seconds = 0.0
    tile_elapsed = 0.0
    owner_chunks = 0
    owner_buffer_bytes = 0
    owner_buffer_peak_bytes = 0
    owner_memmap_bytes = 0
    try:
        while pending or not cancelled:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                for future in pending:
                    future.cancel()
                if not pending:
                    raise ResampleExecutionError("任务已取消，未生成输出。")

            while not cancelled and len(pending) < max(1, int(plan.space_workers)) * 2:
                try:
                    task = next(task_iter)
                except StopIteration:
                    break
                pending.add(executor.submit(_process_space_tile, task))

            if not pending:
                if cancelled:
                    raise ResampleExecutionError("任务已取消，未生成输出。")
                break
            done, pending = wait(
                pending,
                timeout=0.5,
                return_when=FIRST_COMPLETED,
            )
            if not done:
                continue
            if cancelled:
                raise ResampleExecutionError("任务已取消，未生成输出。")
            for future in done:
                try:
                    metrics = future.result()
                except Exception as exc:  # noqa: BLE001 - preserve worker context
                    for other in pending:
                        other.cancel()
                    raise ResampleExecutionError(
                        f"空间并行 worker 失败：{exc}"
                    ) from exc
                completed += 1
                completed_batches += metrics.time_batches
                tile_elapsed += metrics.elapsed
                weight_seconds += metrics.weight_seconds
                owner_chunks += metrics.owner_chunks
                owner_buffer_bytes += metrics.owner_buffer_bytes
                owner_buffer_peak_bytes = max(
                    owner_buffer_peak_bytes,
                    metrics.owner_buffer_peak_bytes,
                )
                owner_memmap_bytes += metrics.owner_memmap_bytes
                _add_timing(
                    timing,
                    {
                        "read": metrics.read_seconds,
                        "regrid": metrics.regrid_seconds,
                        "write": metrics.write_seconds,
                    },
                )
                if progress and total_batches and (
                    completed_batches >= next_report
                    or completed_batches >= total_batches
                ):
                    print(
                        "重采样时间批次："
                        f"{min(completed_batches, total_batches)}/{total_batches}",
                        flush=True,
                    )
                    while next_report <= completed_batches:
                        next_report += report_interval
                if progress and (
                    completed == 1
                    or completed == total_tasks
                    or completed % max(1, total_tasks // 20) == 0
                ):
                    print(_format_tile_progress(completed, total_tasks, metrics), flush=True)
            if cancelled:
                raise ResampleExecutionError("任务已取消，未生成输出。")
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    return {
        "tiles": completed,
        "tile_elapsed_seconds": tile_elapsed,
        "weight_seconds": weight_seconds,
        "read_seconds": timing["read"],
        "regrid_seconds": timing["regrid"],
        "write_seconds": timing["write"],
        "time_batches": completed_batches,
        "total_time_batches": total_batches,
        "owner_chunks": owner_chunks,
        "owner_buffer_bytes": owner_buffer_bytes,
        "owner_buffer_peak_bytes": owner_buffer_peak_bytes,
        "owner_memmap_bytes": owner_memmap_bytes,
    }


def plan_resample(
    config: ResampleConfig,
    inspection: ResampleInspection | None = None,
) -> ResamplePlan:
    inspection = inspection or inspect_resample_input(config.input)
    config_path = Path(config.input).expanduser().resolve()
    if inspection.info.path != config_path:
        raise ValueError("重采样检查结果与配置中的输入路径不一致，请重新检查输入。")
    resource_budget = config.resource_budget or effective_resource_budget(
        source=config_path,
        temporary=config.temporary_dir,
        output=Path(config.output).expanduser().resolve().parent,
    )
    if config.method not in RESAMPLING_METHODS:
        raise ValueError(
            "重采样方法必须是：" + ", ".join(RESAMPLING_METHODS)
        )
    if config.compute_dtype not in COMPUTE_DTYPES:
        raise ValueError(
            "计算 dtype 必须是：" + ", ".join(COMPUTE_DTYPES)
        )
    if config.statistics_policy not in {"auto", "sample", "exact"}:
        raise ValueError("统计策略必须是 auto、sample 或 exact。")
    if config.tuning_objective not in {"speed", "balanced", "compact"}:
        raise ValueError("重采样调优目标必须是 speed、balanced 或 compact。")
    if float(config.tune_budget) < 0:
        raise ValueError("重采样调优预算不能为负数。")
    if not np.isfinite(float(config.na_thres)) or not 0 <= float(config.na_thres) <= 1:
        raise ValueError("na_thres 必须位于 0 到 1 之间。")
    if config.tile_size != "auto":
        try:
            manual_tile_size = int(config.tile_size)
        except (TypeError, ValueError) as exc:
            raise ValueError("tile_size 必须是正整数或 auto。") from exc
        if manual_tile_size <= 0:
            raise ValueError("tile_size 必须是正整数或 auto。")
    else:
        manual_tile_size = None
    def target_grid():
        return build_target_grid(
            inspection.grid,
            config.resolution,
            extent=config.extent,
            lat_bounds=config.target_lat_bounds,
            lon_bounds=config.target_lon_bounds,
            lat_descending=config.target_lat_descending,
            lon_descending=config.target_lon_descending,
        )

    if config.time_block == "auto":
        selected_time_block = resolve_auto_time_block(
            inspection.info,
            inspection.grid,
            target_grid(),
            method=config.method,
            skipna=config.skipna,
            compute_dtype=config.compute_dtype,
            resource_budget=resource_budget,
        )
    else:
        try:
            selected_time_block = int(config.time_block)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_block 必须是正整数或 auto。") from exc
        if selected_time_block <= 0:
            raise ValueError("time_block 必须是正整数或 auto。")
    try:
        requested_compute_workers = int(config.compute_workers)
    except (TypeError, ValueError) as exc:
        raise ValueError("compute_workers 必须是正整数。") from exc
    if requested_compute_workers <= 0:
        raise ValueError("compute_workers 必须是正整数。")
    selected_compute_workers = min(
        requested_compute_workers,
        max(1, int(resource_budget.worker_ceiling)),
    )
    if config.space_workers == "auto":
        selected_space_workers = resolve_auto_space_workers(
            compute_workers=selected_compute_workers,
            resource_budget=resource_budget,
        )
    else:
        try:
            selected_space_workers = int(config.space_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError("space_workers 必须是正整数或 auto。") from exc
        if selected_space_workers <= 0:
            raise ValueError("space_workers 必须是正整数或 auto。")
    target = build_target_grid(
        inspection.grid,
        config.resolution,
        extent=config.extent,
        lat_bounds=config.target_lat_bounds,
        lon_bounds=config.target_lon_bounds,
        lat_descending=config.target_lat_descending,
        lon_descending=config.target_lon_descending,
    )
    tuning_trials: list[dict[str, object]] = []
    auto_tile = None
    if config.tile_size == "auto":
        worker_candidates = (
            range(1, selected_space_workers + 1)
            if config.tuning_objective == "compact" and config.space_workers == "auto"
            else range(selected_space_workers, 0, -1)
            if config.space_workers == "auto"
            else (selected_space_workers,)
        )
        selected_candidate = None
        for candidate_workers in worker_candidates:
            candidate = resolve_auto_tile_size(
                inspection.info,
                inspection.grid,
                target,
                method=config.method,
                skipna=config.skipna,
                time_block=selected_time_block,
                compute_workers=selected_compute_workers,
                space_workers=candidate_workers,
                compute_dtype=config.compute_dtype,
                resource_budget=resource_budget,
            )
            tuning_trials.append(
                {
                    "candidate_id": int(candidate_workers),
                    "space_workers": int(candidate_workers),
                    "estimated_aggregate_parallel_peak_bytes": int(candidate.estimated_peak_bytes),
                    "estimated_per_worker_peak_bytes": int(
                        candidate.estimated_peak_bytes / max(1, int(candidate_workers))
                    ),
                    "fits_memory_budget": bool(candidate.fits_budget),
                    "status": "ok" if candidate.fits_budget else "rejected_memory",
                    "reason": candidate.warning,
                }
            )
            if selected_candidate is None and (
                candidate.fits_budget or candidate_workers == 1
            ):
                selected_candidate = (candidate_workers, candidate)
        if selected_candidate is None:
            raise ValueError("没有可用的重采样空间worker候选。")
        selected_space_workers, auto_tile = selected_candidate
        selected_tile_size = auto_tile.tile_size
    else:
        assert manual_tile_size is not None
        selected_tile_size = manual_tile_size
    planned_output_chunks = output_chunks(inspection.info, target)
    if config.output_layout is not None:
        for item in inspection.info.variables:
            try:
                layout_item = config.output_layout.for_output(item.name)
            except KeyError:
                continue
            expected_shape = tuple(
                target.dimensions.get(dim, int(size))
                for dim, size in zip(item.dims, item.shape)
            )
            if layout_item.shape != expected_shape or layout_item.dims != item.dims:
                raise ValueError(f"变量 {item.name} 与最终输出布局的 shape/dims 不一致。")
            if not item.is_coord:
                expected_dtype = np.dtype(item.dtype)
                if {"lat", "lon"}.issubset(item.dims):
                    if not np.issubdtype(expected_dtype, np.floating):
                        expected_dtype = np.dtype("float64")
                    elif config.compute_dtype == "float32":
                        expected_dtype = np.dtype("float32")
                if np.dtype(layout_item.dtype) != expected_dtype:
                    raise ValueError(f"变量 {item.name} 与最终输出布局的 dtype 不一致。")
            planned_output_chunks[item.name] = layout_item.chunks
    owner_buffer_budget_bytes = resolve_owner_buffer_budget(
        space_workers=selected_space_workers,
        reserved_bytes=(
            auto_tile.estimated_peak_bytes if auto_tile is not None else None
        ),
        available_bytes=(
            resource_budget.memory_available_bytes
            if resource_budget is not None
            else None
        ),
        total_bytes=(
            resource_budget.memory_total_bytes
            if resource_budget is not None
            else None
        ),
    )
    return ResamplePlan(
        inspection=inspection,
        target=target,
        tune_budget=float(config.tune_budget),
        method=config.method,
        skipna=config.skipna,
        na_thres=float(config.na_thres),
        output_chunks=planned_output_chunks,
        compute_dtype=config.compute_dtype,
        tile_size=selected_tile_size,
        time_block=selected_time_block,
        time_block_requested=config.time_block,
        compute_workers=selected_compute_workers,
        space_workers=selected_space_workers,
        tile_size_requested=config.tile_size,
        space_workers_requested=config.space_workers,
        tuning_objective=config.tuning_objective,
        tuning_trials=tuple(tuning_trials),
        auto_tile=auto_tile,
        owner_buffer_budget_bytes=owner_buffer_budget_bytes,
        resource_budget=resource_budget,
        output_layout=config.output_layout,
        before_replacements=config.before_replacements,
        after_replacements=config.after_replacements,
        statistics_policy=config.statistics_policy,
    )


def format_plan(plan: ResamplePlan) -> str:
    lines = [
        "========== Zarr 重采样计划 ==========",
        f"方法：{plan.method}",
        f"skipna：{'开启' if plan.skipna else '关闭'}；na_thres={plan.na_thres:g}",
        (
            "计算 dtype："
            + ("保持源浮点 dtype" if plan.compute_dtype == "source" else "浮点变量转 float32")
        ),
        f"目标网格：lat={plan.target.lat.size}, lon={plan.target.lon.size}",
        f"目标分辨率：{plan.target.lat_resolution:g}° × {plan.target.lon_resolution:g}°",
        f"范围模式：{plan.target.extent}",
        (
            f"流式计算：空间块={plan.tile_size}"
            f"（{'自动' if plan.tile_size_requested == 'auto' else '手动'}），"
            f"时间块={plan.time_block}"
            f"（{'自动' if plan.time_block_requested == 'auto' else '手动'}），"
            f"块内线程={plan.compute_workers}，"
            f"空间进程={plan.space_workers}"
            f"（{'自动' if plan.space_workers_requested == 'auto' else '手动'}）"
        ),
        f"owner buffer 单进程内存上限：{_human_bytes(plan.owner_buffer_budget_bytes)}；超出时使用临时 memmap",
        f"采样前替换规则：{plan.before_replacements.as_pairs() or '无'}",
        f"采样后替换规则：{plan.after_replacements.as_pairs() or '无'}",
        f"表达式统计策略：{plan.statistics_policy}",
        (
            "目标边界："
            f"lon={plan.target.spatial_extent[0]:g} .. {plan.target.spatial_extent[1]:g}，"
            f"lat={plan.target.spatial_extent[2]:g} .. {plan.target.spatial_extent[3]:g}"
        ),
    ]
    if plan.auto_tile is not None:
        auto = plan.auto_tile
        lines.extend(
            [
                "自动空间块依据：",
                (
                    f"  可用内存 {_human_bytes(auto.available_bytes)}；"
                    f"自动预算 {_human_bytes(auto.budget_bytes)}；"
                    f"估算峰值 {_human_bytes(auto.estimated_peak_bytes)}"
                ),
                (
                    f"  分辨率比例 lat={auto.ratio_lat:g}、lon={auto.ratio_lon:g}；"
                    f"最坏变量={auto.worst_variable}"
                ),
                (
                    f"  对齐后的源读取窗口={auto.source_window[0]}×{auto.source_window[1]}；"
                    f"源 chunk {_human_bytes(auto.source_chunk_bytes)}；"
                    f"单次源窗口 {_human_bytes(auto.source_batch_bytes)}"
                ),
            ]
        )
        if auto.warning:
            lines.append(f"  ⚠️ {auto.warning}")
    lines.append("输出 chunks：")
    for name, chunks in plan.output_chunks.items():
        lines.append(f"  {name}: {chunks}")
    return "\n".join(lines)


def _is_nan_scalar(value: object) -> bool:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if array.size != 1 or not np.issubdtype(array.dtype, np.floating):
        return False
    return bool(np.isnan(array.reshape(-1)[0]))


def _assert_output_marker_matches(
    marker: str,
    actual: object,
    expected: object,
) -> None:
    """Compare metadata markers after resampling normalization.

    Floating resampling canonicalizes missing values to the Zarr NaN fill
    value and intentionally removes a redundant ``missing_value=NaN`` CF
    attribute. Treat that representation as equivalent to the source
    marker; numeric missing markers and scale/offset remain exact checks.
    """

    if marker == "missing_value" and actual is None and _is_nan_scalar(expected):
        return
    np.testing.assert_equal(actual, expected)


def _validate_output(
    source: xr.Dataset,
    output_path: Path,
    plan: ResamplePlan,
) -> None:
    target = xr.open_zarr(
        output_path,
        consolidated=False,
        chunks=None,
        decode_times=False,
        mask_and_scale=False,
    )
    try:
        expected_dimensions = dict(plan.inspection.info.dimensions)
        expected_dimensions.update(plan.target.dimensions)
        actual_dimensions = {name: int(size) for name, size in target.sizes.items()}
        if actual_dimensions != expected_dimensions:
            raise ResampleExecutionError(
                f"输出维度不符合计划：期望 {expected_dimensions}，实际 {actual_dimensions}"
            )
        if set(source.data_vars) != set(target.data_vars):
            raise ResampleExecutionError("输出变量集合与输入不一致。")
        for item in plan.inspection.info.variables:
            actual = target[item.name]
            expected_chunks = plan.output_chunks[item.name]
            actual_chunks = tuple(int(value) for value in actual.encoding.get("chunks", ()))
            if actual_chunks != expected_chunks:
                raise ResampleExecutionError(
                    f"变量 {item.name} 的 chunks 不符合计划："
                    f"期望 {expected_chunks}，实际 {actual_chunks}"
                )
            if tuple(actual.dims) != tuple(
                "lat" if dim == "lat" else "lon" if dim == "lon" else dim
                for dim in source[item.name].dims
            ):
                raise ResampleExecutionError(f"变量 {item.name} 的维度顺序发生变化。")
            if {"lat", "lon"}.issubset(source[item.name].dims):
                expected_dtype = _resampled_output_dtype(
                    source[item.name],
                    plan.compute_dtype,
                )
                if np.dtype(actual.dtype) != expected_dtype:
                    raise ResampleExecutionError(
                        f"变量 {item.name} 的 dtype 不符合计划："
                        f"期望 {expected_dtype}，实际 {actual.dtype}"
                    )
                source_variable = source[item.name]
                source_has_missing_marker = any(
                    marker in source_variable.encoding or marker in source_variable.attrs
                    for marker in ("_FillValue", "missing_value")
                )
                if np.issubdtype(actual.dtype, np.floating):
                    actual_fill = actual.encoding.get("_FillValue")
                    if actual_fill is None:
                        actual_fill = actual.attrs.get("_FillValue")
                    if source_has_missing_marker and not _is_nan_scalar(actual_fill):
                        raise ResampleExecutionError(
                            f"变量 {item.name} 的浮点输出必须使用 NaN _FillValue，实际 {actual_fill!r}"
                        )
                    if actual_fill is not None and not _is_nan_scalar(actual_fill):
                        raise ResampleExecutionError(
                            f"变量 {item.name} 的浮点输出 _FillValue 无效：{actual_fill!r}"
                        )
                    if source_has_missing_marker and actual.attrs.get("missing_value") is not None:
                        raise ResampleExecutionError(
                            f"变量 {item.name} 的浮点输出不应保留 missing_value。"
                        )
                else:
                    expected_fill = source_variable.encoding.get("_FillValue")
                    if expected_fill is not None:
                        _assert_output_marker_matches(
                            "_FillValue",
                            actual.encoding.get("_FillValue"),
                            expected_fill,
                        )
        np.testing.assert_allclose(target.lat.values, plan.target.lat, rtol=0, atol=1e-10)
        np.testing.assert_allclose(target.lon.values, plan.target.lon, rtol=0, atol=1e-10)
        if "time" in source.coords and "time" in target.coords:
            np.testing.assert_array_equal(target.time.values, source.time.values)
    except (AssertionError, KeyError, ValueError) as exc:
        raise ResampleExecutionError(f"输出结构或坐标校验失败：{exc}") from exc
    finally:
        target.close()


def _statistics_limit(plan: ResamplePlan) -> tuple[int | None, str]:
    if plan.statistics_policy == "exact":
        return None, "exact"
    if plan.statistics_policy == "sample":
        return 250_000, "sample"
    output_logical_bytes = 0
    for item in plan.inspection.info.data_variables:
        shape = tuple(
            plan.target.dimensions.get(dim, int(size))
            for dim, size in zip(item.dims, item.shape)
        )
        dtype = np.dtype(item.dtype)
        if {"lat", "lon"}.issubset(item.dims):
            if not np.issubdtype(dtype, np.floating):
                dtype = np.dtype("float64")
            elif plan.compute_dtype == "float32":
                dtype = np.dtype("float32")
        output_logical_bytes += int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if max(plan.inspection.info.logical_bytes, output_logical_bytes) <= 128 * 1024**2:
        return None, "exact"
    return 250_000, "sample"


def _source_replacement_statistics(
    source: xr.Dataset,
    plan: ResamplePlan,
) -> tuple[dict[str, dict[str, float]], str]:
    if not plan.before_replacements.data_dependent:
        return {}, "not_required"
    items = tuple(
        item
        for item in plan.inspection.info.data_variables
        if {"lat", "lon"}.issubset(item.dims)
    )
    if plan.method == "nearest_d2s":
        lat_slice = slice(0, plan.inspection.grid.lat.size)
        lon_slice = slice(0, plan.inspection.grid.lon.size)
    else:
        _target, lat_slice, lon_slice = _resolve_local_source_window(
            plan.inspection.grid,
            plan.target,
            plan.method,
        )
        if lat_slice is None or lon_slice is None:
            return {}, "not_required"
    variables = {
        item.name: _mask_missing(source[item.name], item).isel(
            lat=lat_slice,
            lon=lon_slice,
        )
        for item in items
    }
    dataset = xr.Dataset(variables)
    maximum, mode = _statistics_limit(plan)
    try:
        return sample_statistics(
            dataset,
            tuple(item.name for item in items),
            maximum_values=maximum,
        ), mode
    finally:
        dataset.close()


def _apply_data_dependent_post_replacements(
    path: Path,
    plan: ResamplePlan,
    *,
    cancel_event=None,
) -> tuple[dict[str, dict[str, float]], str]:
    names = tuple(
        item.name
        for item in plan.inspection.info.data_variables
        if {"lat", "lon"}.issubset(item.dims)
    )
    dataset = xr.open_zarr(
        path,
        consolidated=False,
        chunks={},
        decode_times=False,
        mask_and_scale=True,
    )
    maximum, mode = _statistics_limit(plan)
    try:
        statistics = sample_statistics(dataset, names, maximum_values=maximum)
    finally:
        dataset.close()
    group = zarr.open_group(path, mode="r+")
    for name in names:
        array = group[name]
        for region in _chunk_regions(tuple(array.shape), tuple(array.chunks)):
            if cancel_event is not None and cancel_event.is_set():
                raise ResampleExecutionError("任务已取消，未生成输出。")
            replaced = apply_replacement_rules(
                np.asarray(array[region]),
                plan.after_replacements,
                statistics.get(name, {}),
            )
            array[region] = replaced.astype(array.dtype, copy=False)
    return statistics, mode
def _native_source_has_direct_values(path: Path, plan: ResamplePlan) -> bool:
    try:
        group = zarr.open_group(path, mode="r")
        for item in plan.inspection.info.variables:
            attrs = item.attrs
            if "scale_factor" in attrs or "add_offset" in attrs:
                return False
            for marker in ("_FillValue", "missing_value"):
                if marker in attrs and not _is_nan_scalar(attrs[marker]):
                    return False
        return all(
            not tuple(group[item.name].compressors)
            for item in plan.inspection.info.data_variables
        )
    except (KeyError, OSError, ValueError, TypeError):
        return False




def _native_read_block(
    source: xr.Dataset,
    name: str,
    time_start: int,
    time_stop: int,
    source_lat_slice: slice,
    source_lon_slice: slice,
) -> np.ndarray:
    values = source[name].isel(
        time=slice(time_start, time_stop),
        lat=source_lat_slice,
        lon=source_lon_slice,
    ).values
    values = np.ascontiguousarray(values, dtype="float32")
    if values.ndim != 3 or not np.isfinite(values).all():
        raise _NativeResampleFallback(
            "native regular resampling requires finite three-dimensional float32 blocks"
        )
    return values


def _native_call_batch(
    native,
    blocks: list[np.ndarray],
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str,
) -> np.ndarray:
    first_shape = blocks[0].shape
    if any(block.shape != first_shape for block in blocks):
        raise ResampleExecutionError("native batch blocks have inconsistent shapes")
    values = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=0)
    output = np.empty(
        (values.shape[0], target_lat.size, target_lon.size),
        dtype="float32",
    )
    if hasattr(native, "resample_f32_buffer_into"):
        result_shape = native.resample_f32_buffer_into(
            values,
            list(values.shape),
            source_lat,
            source_lon,
            target_lat,
            target_lon,
            method,
            output,
        )
        return output.reshape(tuple(int(value) for value in result_shape))
    raw_values, result_shape = native.resample_f32_buffer(
        values,
        list(values.shape),
        source_lat,
        source_lon,
        target_lat,
        target_lon,
        method,
    )
    return np.frombuffer(raw_values, dtype="float32").reshape(
        tuple(int(value) for value in result_shape)
    )


def _native_process_tile(
    source: xr.Dataset,
    output_group,
    native,
    plan: ResamplePlan,
    task: _OwnerTask,
    items_by_name: dict[str, VariableInfo],
    source_lat: np.ndarray,
    source_lon: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    workdir: Path,
    cancel_event=None,
) -> dict[str, float | int]:
    started = time.perf_counter()
    lat_start, lat_stop, lon_start, lon_stop = task.region
    tile = _tile_target(plan.target, lat_start, lat_stop, lon_start, lon_stop)
    target_tile, source_lat_slice, source_lon_slice = _resolve_local_source_window(
        plan.inspection.grid,
        tile,
        plan.method,
    )
    metrics: dict[str, float | int] = {
        "elapsed": 0.0,
        "read": 0.0,
        "resample": 0.0,
        "write": 0.0,
        "time_batches": 0,
        "owner_chunks": 0,
        "owner_buffer_bytes": 0,
        "owner_buffer_peak_bytes": 0,
        "owner_memmap_bytes": 0,
    }
    item_names = task.item_names
    if source_lat_slice is None or source_lon_slice is None:
        for name in item_names:
            item = items_by_name[name]
            output_array = output_group[name]
            time_axis = item.dims.index("time")
            time_size = int(item.shape[time_axis])
            time_chunk = int(output_array.chunks[time_axis])
            for time_start in range(0, time_size, time_chunk):
                if cancel_event is not None and cancel_event.is_set():
                    raise ResampleExecutionError("任务已取消，未生成输出。")
                time_stop = min(time_start + time_chunk, time_size)
                shape = (time_stop - time_start, lat_stop - lat_start, lon_stop - lon_start)
                with _owner_buffer(
                    shape,
                    np.dtype("float32"),
                    workdir,
                    plan.owner_buffer_budget_bytes,
                ) as (values, nbytes, used_memmap):
                    values[...] = np.nan
                    write_started = time.perf_counter()
                    _write_region(
                        output_group,
                        name,
                        values,
                        item.dims,
                        {
                            "time": slice(time_start, time_stop),
                            "lat": slice(lat_start, lat_stop),
                            "lon": slice(lon_start, lon_stop),
                        },
                    )
                    metrics["write"] += time.perf_counter() - write_started
                metrics["owner_chunks"] += 1
                metrics["owner_buffer_bytes"] += nbytes
                metrics["owner_buffer_peak_bytes"] = max(
                    int(metrics["owner_buffer_peak_bytes"]), nbytes
                )
                if used_memmap:
                    metrics["owner_memmap_bytes"] += nbytes
                metrics["time_batches"] += 1
        metrics["elapsed"] = time.perf_counter() - started
        return metrics

    source_lat_local = np.ascontiguousarray(
        source_lat[source_lat_slice], dtype="float32"
    )
    source_lon_local = np.ascontiguousarray(
        source_lon[source_lon_slice], dtype="float32"
    )
    target_lat_local = np.ascontiguousarray(target_tile.lat, dtype="float32")
    target_lon_local = np.ascontiguousarray(target_tile.lon, dtype="float32")
    source_area = int(source_lat_local.size) * int(source_lon_local.size)
    target_area = int(target_lat_local.size) * int(target_lon_local.size)
    if source_area <= 0 or target_area <= 0:
        raise ResampleExecutionError("native resampling region has an empty spatial axis")
    method = "nearest" if plan.method.startswith("nearest") else "bilinear"

    grouped: dict[tuple[int, int], list[VariableInfo]] = {}
    for name in item_names:
        item = items_by_name[name]
        output_array = output_group[name]
        time_axis = item.dims.index("time")
        grouped.setdefault(
            (int(item.shape[time_axis]), int(output_array.chunks[time_axis])), []
        ).append(item)

    for group_items in grouped.values():
        batch_count = len(group_items)
        max_input_time = MAX_NATIVE_RESAMPLE_VALUES // (source_area * batch_count)
        max_output_time = MAX_NATIVE_RESAMPLE_VALUES // (target_area * batch_count)
        native_time_block = min(
            max(1, int(plan.time_block)),
            max_input_time,
            max_output_time,
        )
        if native_time_block < 1:
            raise _NativeResampleFallback(
                "native regular resampling region exceeds the bounded typed buffer bridge"
            )
        time_size = int(group_items[0].shape[group_items[0].dims.index("time")])
        time_chunk = int(
            output_group[group_items[0].name].chunks[group_items[0].dims.index("time")]
        )
        for chunk_start in range(0, time_size, time_chunk):
            if cancel_event is not None and cancel_event.is_set():
                raise ResampleExecutionError("任务已取消，未生成输出。")
            chunk_stop = min(chunk_start + time_chunk, time_size)
            chunk_length = chunk_stop - chunk_start
            use_owner_buffer = native_time_block < chunk_length
            stack = ExitStack()
            buffers: dict[str, np.ndarray] = {}
            try:
                if use_owner_buffer:
                    for item in group_items:
                        shape = (
                            chunk_length,
                            lat_stop - lat_start,
                            lon_stop - lon_start,
                        )
                        values, nbytes, used_memmap = stack.enter_context(
                            _owner_buffer(
                                shape,
                                np.dtype("float32"),
                                workdir,
                                plan.owner_buffer_budget_bytes,
                            )
                        )
                        buffers[item.name] = values
                        metrics["owner_buffer_bytes"] += nbytes
                        metrics["owner_buffer_peak_bytes"] += nbytes
                        if used_memmap:
                            metrics["owner_memmap_bytes"] += nbytes
                for block_start in range(chunk_start, chunk_stop, native_time_block):
                    if cancel_event is not None and cancel_event.is_set():
                        raise ResampleExecutionError("任务已取消，未生成输出。")
                    block_stop = min(block_start + native_time_block, chunk_stop)
                    read_started = time.perf_counter()
                    blocks = [
                        _native_read_block(
                            source,
                            item.name,
                            block_start,
                            block_stop,
                            source_lat_slice,
                            source_lon_slice,
                        )
                        for item in group_items
                    ]
                    metrics["read"] += time.perf_counter() - read_started
                    resample_started = time.perf_counter()
                    result = _native_call_batch(
                        native,
                        blocks,
                        source_lat_local,
                        source_lon_local,
                        target_lat_local,
                        target_lon_local,
                        method,
                    )
                    metrics["resample"] += time.perf_counter() - resample_started
                    block_length = block_stop - block_start
                    expected_shape = (
                        block_length * len(group_items),
                        lat_stop - lat_start,
                        lon_stop - lon_start,
                    )
                    if tuple(result.shape) != expected_shape:
                        raise ResampleExecutionError(
                            f"native result shape mismatch: expected {expected_shape}, got {result.shape}"
                        )
                    for index, item in enumerate(group_items):
                        values = result[
                            index * block_length : (index + 1) * block_length
                        ]
                        if use_owner_buffer:
                            buffers[item.name][
                                block_start - chunk_start : block_stop - chunk_start
                            ] = values
                        else:
                            write_started = time.perf_counter()
                            _write_region(
                                output_group,
                                item.name,
                                values,
                                item.dims,
                                {
                                    "time": slice(block_start, block_stop),
                                    "lat": slice(lat_start, lat_stop),
                                    "lon": slice(lon_start, lon_stop),
                                },
                            )
                            metrics["write"] += time.perf_counter() - write_started
                    metrics["time_batches"] += 1
                if use_owner_buffer:
                    for item in group_items:
                        write_started = time.perf_counter()
                        _write_region(
                            output_group,
                            item.name,
                            buffers[item.name],
                            item.dims,
                            {
                                "time": slice(chunk_start, chunk_stop),
                                "lat": slice(lat_start, lat_stop),
                                "lon": slice(lon_start, lon_stop),
                            },
                        )
                        metrics["write"] += time.perf_counter() - write_started
            finally:
                stack.close()
            for item in group_items:
                chunk_shape = (
                    chunk_length,
                    lat_stop - lat_start,
                    lon_stop - lon_start,
                )
                metrics["owner_chunks"] += 1
                if not use_owner_buffer:
                    metrics["owner_buffer_bytes"] += int(
                        np.prod(chunk_shape, dtype=np.int64) * np.dtype("float32").itemsize
                    )
                    metrics["owner_buffer_peak_bytes"] = max(
                        int(metrics["owner_buffer_peak_bytes"]),
                        int(np.prod(chunk_shape, dtype=np.int64) * np.dtype("float32").itemsize),
                    )
    metrics["elapsed"] = time.perf_counter() - started
    return metrics


def _run_native_regular_resample(
    config: ResampleConfig,
    plan: ResamplePlan,
    *,
    cancel_event=None,
) -> dict[str, object]:
    source_path = Path(config.input).expanduser().resolve()
    target_path = Path(config.output).expanduser().resolve()
    if config.before_replacements.data_dependent or config.after_replacements.data_dependent:
        raise ResampleExecutionError(
            "native regular resampling does not support data-dependent replacements"
        )
    _prepare_target(target_path, config.overwrite)
    preflight_writable(target_path.parent, "重采样输出")
    source = xr.open_zarr(
        source_path,
        consolidated=False,
        chunks=None,
        decode_times=False,
        mask_and_scale=True,
    )
    staging = target_path.parent / f".{target_path.name}.native-resample-{uuid4().hex}.tmp"
    started = time.perf_counter()
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise ResampleExecutionError("任务已取消，未生成输出。")
        for item in plan.inspection.info.data_variables:
            if item.dims != ("time", "lat", "lon") or str(item.dtype) != "float32":
                raise ResampleExecutionError(
                    "native regular resampling requires float32 (time, lat, lon) variables"
                )
        _initialize_output_store(source, plan.inspection.info, plan, staging)
        output_group = zarr.open_group(staging, mode="r+")
        source_lat = np.ascontiguousarray(source["lat"].values, dtype="float32")
        source_lon = np.ascontiguousarray(source["lon"].values, dtype="float32")
        target_lat = np.ascontiguousarray(plan.target.lat, dtype="float32")
        target_lon = np.ascontiguousarray(plan.target.lon, dtype="float32")
        native = __import__("fast_nc_zarr._native", fromlist=["resample_f32_buffer"])
        spatial_items = tuple(plan.inspection.info.data_variables)
        items_by_name = {item.name: item for item in spatial_items}
        owner_groups = _owner_task_groups(spatial_items, plan)
        total_tiles = _owner_task_count(owner_groups, plan.target)
        tasks = iter(_owner_tasks(owner_groups, plan.target))
        workers = max(1, min(int(plan.space_workers), total_tiles)) if total_tiles else 1
        timing: dict[str, float | int] = {
            "tiles": 0,
            "tile_elapsed_seconds": 0.0,
            "read_seconds": 0.0,
            "resample_seconds": 0.0,
            "write_seconds": 0.0,
            "time_batches": 0,
            "owner_chunks": 0,
            "owner_buffer_bytes": 0,
            "owner_buffer_peak_bytes": 0,
            "owner_memmap_bytes": 0,
        }

        def run_task(task: _OwnerTask):
            return _native_process_tile(
                source,
                output_group,
                native,
                plan,
                task,
                items_by_name,
                source_lat,
                source_lon,
                target_lat,
                target_lon,
                staging,
                cancel_event,
            )

        def add_metrics(metrics: dict[str, float | int]) -> None:
            timing["tiles"] += 1
            timing["tile_elapsed_seconds"] += float(metrics["elapsed"])
            timing["read_seconds"] += float(metrics["read"])
            timing["resample_seconds"] += float(metrics["resample"])
            timing["write_seconds"] += float(metrics["write"])
            timing["time_batches"] += int(metrics["time_batches"])
            timing["owner_chunks"] += int(metrics["owner_chunks"])
            timing["owner_buffer_bytes"] += int(metrics["owner_buffer_bytes"])
            timing["owner_buffer_peak_bytes"] = max(
                int(timing["owner_buffer_peak_bytes"]),
                int(metrics["owner_buffer_peak_bytes"]),
            )
            timing["owner_memmap_bytes"] += int(metrics["owner_memmap_bytes"])

        if workers == 1:
            for task in tasks:
                add_metrics(run_task(task))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending = set()
                for _ in range(min(workers * 2, total_tiles)):
                    pending.add(executor.submit(run_task, next(tasks)))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        add_metrics(future.result())
                        try:
                            pending.add(executor.submit(run_task, next(tasks)))
                        except StopIteration:
                            pass
        if cancel_event is not None and cancel_event.is_set():
            raise ResampleExecutionError("任务已取消，未生成输出。")
        if config.validate:
            _validate_output(source, staging, plan)
        logical_bytes = sum(
            int(item.shape[0])
            * int(plan.target.dimensions["lat"])
            * int(plan.target.dimensions["lon"])
            * np.dtype("float32").itemsize
            for item in spatial_items
        )
        publish_staging(
            staging,
            target_path,
            "resample-native",
            overwrite=config.overwrite,
            require_zarr_v3=True,
        )
        elapsed = time.perf_counter() - started
        physical_bytes = _directory_size(target_path)
        return {
            "backend": "rust",
            "backend_fallback": False,
            "output": str(target_path),
            "logical_bytes": logical_bytes,
            "physical_bytes": physical_bytes,
            "method": plan.method,
            "wall_seconds": elapsed,
            "throughput_mib_s": logical_bytes / 1024**2 / max(elapsed, 1e-9),
            "physical_throughput_mib_s": physical_bytes / 1024**2 / max(elapsed, 1e-9),
            "used_intermediate": False,
            "logical_write_amplification": 1.0,
            "avoided_intermediate_bytes": logical_bytes,
            "space_workers": workers,
            "tiles": int(timing["tiles"]),
            "time_batches": int(timing["time_batches"]),
            "owner_buffer": {
                "logical_bytes": int(timing["owner_buffer_bytes"]),
                "peak_bytes": int(timing["owner_buffer_peak_bytes"]),
                "memmap_bytes": int(timing["owner_memmap_bytes"]),
                "physical_chunks": int(timing["owner_chunks"]),
            },
            "tile_timing": timing,
        }
    except _NativeResampleFallback:
        raise
    except Exception as exc:
        if cancel_event is not None and cancel_event.is_set():
            raise ResampleExecutionError("任务已取消，未生成输出。") from exc
        if isinstance(exc, ResampleExecutionError):
            raise
        raise ResampleExecutionError(
            f"原生重采样失败；输出临时目录已清理：{staging}\n{exc}"
        ) from exc
    finally:
        source.close()
        shutil.rmtree(staging, ignore_errors=True)


def run_resample(
    config: ResampleConfig,
    plan: ResamplePlan | None = None,
    *,
    cancel_event=None,
    progress: bool = True,
) -> dict[str, object]:
    validate_resampling_environment()
    plan = plan or plan_resample(config)
    requested_plan = plan
    source_path = Path(config.input).expanduser().resolve()
    target_path = Path(config.output).expanduser().resolve()
    if source_path == target_path:
        raise ResampleExecutionError("输入和输出不能是同一个目录。")
    if source_path != plan.inspection.info.path:
        raise ResampleExecutionError("输入检查结果与执行输入路径不一致。")
    native_operation = "resample.nearest" if plan.method.startswith("nearest") else "resample.bilinear"
    if (
        plan.method in {"nearest_s2d", "bilinear"}
        and resolve_backend("auto", native_operation) == "rust"
        and not config.before_replacements.rules
        and not config.after_replacements.rules
        and all(item.dims == ("time", "lat", "lon") and str(item.dtype) == "float32" for item in plan.inspection.info.data_variables)
        and _native_source_has_direct_values(Path(config.input), plan)
    ):
        try:
            return _run_native_regular_resample(config, plan, cancel_event=cancel_event)
        except _NativeResampleFallback:
            pass
    _prepare_target(target_path, config.overwrite)
    preflight_writable(target_path.parent, "重采样输出")
    temporary_root = _resolve_temporary_root(
        source_path,
        target_path,
        config.temporary_dir,
    )
    preflight_writable(temporary_root, "重采样临时")
    staging = target_path.parent / f".{target_path.name}.resample-{uuid4().hex}.tmp"
    workdir = temporary_root / f".{target_path.name}.weights-{uuid4().hex}.tmp"
    started = time.perf_counter()
    source = None
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise ResampleExecutionError("任务已取消。")
        workdir.mkdir(parents=True, exist_ok=False)
        source = xr.open_zarr(
            source_path,
            consolidated=False,
            chunks={},
            decode_times=False,
            mask_and_scale=True,
        )
        before_statistics, before_statistics_mode = _source_replacement_statistics(
            source, requested_plan
        )
        runtime_plan = replace(requested_plan, statistics=before_statistics)
        processing_plan = runtime_plan
        if requested_plan.after_replacements.data_dependent:
            # Data-dependent post rules require statistics from the unmodified
            # resampled product, so they run as a bounded second pass below.
            processing_plan = replace(
                runtime_plan,
                after_replacements=ReplacementRules(),
            )
        if progress:
            print(format_plan(runtime_plan))
            print("初始化最终输出骨架；随后按空间块和时间块计算……")
        spatial_names = _initialize_output_store(
            source,
            runtime_plan.inspection.info,
            runtime_plan,
            staging,
        )
        avoided_intermediate = _avoids_intermediate_store(
            runtime_plan.inspection.info,
            runtime_plan,
        )
        processing_path = staging
        output_group = zarr.open_group(processing_path, mode="r+")
        if cancel_event is not None and cancel_event.is_set():
            raise ResampleExecutionError("任务已取消，未生成输出。")

        grid = runtime_plan.inspection.grid
        spatial_items = tuple(
            item
            for item in runtime_plan.inspection.info.data_variables
            if item.name in spatial_names
        )
        owner_groups = _owner_task_groups(spatial_items, processing_plan)
        total_tiles = _owner_task_count(owner_groups, runtime_plan.target)
        items_by_name = {item.name: item for item in spatial_items}
        total_batches = _owner_total_batch_count(
            owner_groups,
            runtime_plan.target,
            items_by_name,
            processing_plan,
        )

        def tile_tasks():
            return _owner_tasks(owner_groups, runtime_plan.target)

        if progress:
            if avoided_intermediate:
                print(
                    "最终物理 chunks 由单一 task 持有；时间批次先填充有界 "
                    "owner buffer，再各写一次，不创建中转 Zarr。"
                )
            print(
                f"流式计算：{total_tiles} 个物理 chunk owner task，"
                f"空间进程={runtime_plan.space_workers}，"
                f"每个进程使用 {runtime_plan.compute_workers} 个 Dask 线程。"
            )
        full_regridder = None
        full_weight_path = None
        tile_timing: dict[str, float | int] = {
            "tiles": 0,
            "tile_elapsed_seconds": 0.0,
            "weight_seconds": 0.0,
            "read_seconds": 0.0,
            "regrid_seconds": 0.0,
            "write_seconds": 0.0,
            "time_batches": 0,
            "total_time_batches": total_batches,
            "owner_chunks": 0,
            "owner_buffer_bytes": 0,
            "owner_buffer_peak_bytes": 0,
            "owner_memmap_bytes": 0,
        }
        if total_tiles and runtime_plan.method == "nearest_d2s":
            full_regridder, full_weight_path = _build_regridder(
                grid.lat,
                grid.lon,
                grid.lat_bounds,
                grid.lon_bounds,
                runtime_plan.target,
                runtime_plan.method,
                workdir,
                lat_attrs=dict(source.lat.attrs),
                lon_attrs=dict(source.lon.attrs),
                periodic=bool(grid.periodic),
            )
        try:
            # d2s must retain one full-grid weight matrix, so it deliberately
            # stays in the caller. Tiny jobs also avoid spawn overhead.
            if (
                full_regridder is not None
                or total_tiles <= 1
                or runtime_plan.space_workers <= 1
            ):
                for index, task in enumerate(tile_tasks(), start=1):
                    if cancel_event is not None and cancel_event.is_set():
                        raise ResampleExecutionError("任务已取消，未生成输出。")
                    metrics = _execute_serial_tile(
                        source,
                        output_group,
                        processing_plan,
                        task,
                        workdir,
                        spatial_items,
                        full_regridder=full_regridder,
                    )
                    tile_timing["tiles"] += 1
                    tile_timing["tile_elapsed_seconds"] += metrics.elapsed
                    tile_timing["weight_seconds"] += metrics.weight_seconds
                    tile_timing["read_seconds"] += metrics.read_seconds
                    tile_timing["regrid_seconds"] += metrics.regrid_seconds
                    tile_timing["write_seconds"] += metrics.write_seconds
                    tile_timing["time_batches"] += metrics.time_batches
                    tile_timing["owner_chunks"] += metrics.owner_chunks
                    tile_timing["owner_buffer_bytes"] += metrics.owner_buffer_bytes
                    tile_timing["owner_buffer_peak_bytes"] = max(
                        tile_timing["owner_buffer_peak_bytes"],
                        metrics.owner_buffer_peak_bytes,
                    )
                    tile_timing["owner_memmap_bytes"] += metrics.owner_memmap_bytes
                    if progress and (
                        index == 1
                        or index == total_tiles
                        or index % max(1, total_tiles // 20) == 0
                    ):
                        print(_format_tile_progress(index, total_tiles, metrics), flush=True)
            else:
                parallel_plan = replace(
                    processing_plan,
                    space_workers=min(
                        int(runtime_plan.space_workers),
                        total_tiles,
                    ),
                )
                tile_timing = _run_parallel_tiles(
                    source_path,
                    processing_path,
                    workdir,
                    parallel_plan,
                    spatial_items,
                    tile_tasks(),
                    total_tiles,
                    total_batches,
                    cancel_event=cancel_event,
                    progress=progress,
                )
        finally:
            if full_regridder is not None:
                del full_regridder
            if full_weight_path is not None:
                try:
                    full_weight_path.unlink()
                except OSError:
                    pass

        merge_metrics = None

        after_statistics: dict[str, dict[str, float]] = {}
        after_statistics_mode = "not_required"
        if requested_plan.after_replacements.data_dependent:
            if progress:
                print("计算采样后规则统计量，并对最终 staging 执行替换……")
            after_statistics, after_statistics_mode = (
                _apply_data_dependent_post_replacements(
                    staging,
                    requested_plan,
                    cancel_event=cancel_event,
                )
            )

        if progress:
            print(
                "空间块耗时汇总："
                f"累计 {tile_timing['tile_elapsed_seconds']:.1f}s"
                f"（权重 {tile_timing['weight_seconds']:.1f}s、"
                f"读取 {tile_timing['read_seconds']:.1f}s、"
                f"重采样 {tile_timing['regrid_seconds']:.1f}s、"
                f"写入 {tile_timing['write_seconds']:.1f}s；"
                "并行任务累计，非总墙钟时间）。",
                flush=True,
            )

        if cancel_event is not None and cancel_event.is_set():
            raise ResampleExecutionError("任务已取消，未生成输出。")
        if config.validate:
            _validate_output(source, staging, runtime_plan)
        logical_bytes = sum(
            int(
                np.prod(
                    tuple(
                        runtime_plan.target.dimensions.get(dim, int(size))
                        for dim, size in zip(
                            variable.dims,
                            variable.shape,
                        )
                    ),
                    dtype=np.int64,
                )
            )
            * _resampled_output_dtype(
                source[variable.name],
                runtime_plan.compute_dtype,
            ).itemsize
            for variable in runtime_plan.inspection.info.data_variables
        )
        source.close()
        source = None
        _publish_staging(staging, target_path, config.overwrite)
        elapsed = time.perf_counter() - started
        physical_bytes = _directory_size(target_path)
        if progress:
            print(f"重采样完成并通过校验：{target_path}")
        return {
            "elapsed": elapsed,
            "wall_seconds": elapsed,
            "output": str(target_path),
            "physical_bytes": physical_bytes,
            "logical_bytes": logical_bytes,
            "used_intermediate": False,
            "intermediate_logical_bytes": 0,
            "owner_buffer_bytes": int(tile_timing["owner_buffer_bytes"]),
            "avoided_intermediate_bytes": (
                logical_bytes if avoided_intermediate else 0
            ),
            "owner_buffer": {
                "heap_budget_bytes_per_worker": runtime_plan.owner_buffer_budget_bytes,
                "logical_bytes": int(tile_timing["owner_buffer_bytes"]),
                "peak_bytes": int(tile_timing["owner_buffer_peak_bytes"]),
                "memmap_bytes": int(tile_timing["owner_memmap_bytes"]),
                "physical_chunks": int(tile_timing["owner_chunks"]),
            },
            "throughput_mib_s": logical_bytes / 1024**2 / max(elapsed, 1e-9),
            "physical_throughput_mib_s": (
                physical_bytes / 1024**2 / max(elapsed, 1e-9)
            ),
            "logical_write_amplification": 1.0,
            "space_workers": min(runtime_plan.space_workers, total_tiles),
            "compute_workers_per_space_worker": runtime_plan.compute_workers,
            "memory_plan": (
                {
                    "available_bytes": runtime_plan.auto_tile.available_bytes,
                    "budget_bytes": runtime_plan.auto_tile.budget_bytes,
                    "estimated_per_worker_peak_bytes": (
                        runtime_plan.auto_tile.estimated_peak_bytes
                        // max(1, int(runtime_plan.space_workers))
                    ),
                    "estimated_aggregate_parallel_peak_bytes": runtime_plan.auto_tile.estimated_peak_bytes,
                    "fits_budget": runtime_plan.auto_tile.fits_budget,
                    "warning": runtime_plan.auto_tile.warning,
                    "owner_buffer_budget_bytes_per_worker": (
                        runtime_plan.owner_buffer_budget_bytes
                    ),
                    "observed_per_worker_peak_bytes": int(
                        tile_timing["owner_buffer_peak_bytes"]
                    ),
                    "observed_aggregate_parallel_peak_bytes": int(
                        tile_timing["owner_buffer_peak_bytes"]
                    ) * max(1, int(runtime_plan.space_workers)),
                    "owner_memmap_bytes": int(tile_timing["owner_memmap_bytes"]),
                }
                if runtime_plan.auto_tile is not None
                else None
            ),
            "resolved_plan": {
                "tile_size": int(runtime_plan.tile_size),
                "time_block": int(runtime_plan.time_block),
                "time_block_requested": runtime_plan.time_block_requested,
                "compute_workers": int(runtime_plan.compute_workers),
                "space_workers": int(runtime_plan.space_workers),
                "space_workers_requested": runtime_plan.space_workers_requested,
                "method": runtime_plan.method,
                "compute_dtype": runtime_plan.compute_dtype,
                "tuning_objective": runtime_plan.tuning_objective,
                "tune_budget": float(runtime_plan.tune_budget),
            },
            "tuning": {
                "objective": runtime_plan.tuning_objective,
                "budget_seconds": float(runtime_plan.tune_budget),
                "candidate_trials": list(runtime_plan.tuning_trials),
                "selected_candidate_id": int(runtime_plan.space_workers),
                "selection_reason": (
                    "按有效资源和物理chunk ownership筛选空间worker；"
                    "内存安全优先，真实阶段记录实际耗时"
                ),
                "rejected_candidates": [
                    item
                    for item in runtime_plan.tuning_trials
                    if item.get("status") != "ok"
                ],
            },
            "resource_budget": (
                runtime_plan.resource_budget.to_dict()
                if runtime_plan.resource_budget is not None
                else None
            ),
            "temporary_dir": str(temporary_root),
            "merge_timing": merge_metrics,
            "tile_timing": tile_timing,
            "replacement_statistics": {
                "before": before_statistics,
                "before_mode": before_statistics_mode,
                "after": after_statistics,
                "after_mode": after_statistics_mode,
            },
        }
    except Exception as exc:
        if source is not None:
            source.close()
            source = None
        shutil.rmtree(staging, ignore_errors=True)
        if cancel_event is not None and cancel_event.is_set():
            raise ResampleExecutionError("任务已取消，未生成输出。") from exc
        if isinstance(exc, ResampleExecutionError):
            raise
        raise ResampleExecutionError(
            f"重采样失败；输出临时目录已清理：{staging}\n{exc}"
        ) from exc
    finally:
        if source is not None:
            source.close()
        shutil.rmtree(workdir, ignore_errors=True)
