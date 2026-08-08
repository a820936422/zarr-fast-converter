from __future__ import annotations

import json
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
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

from ..rechunking.models import DatasetInfo, VariableInfo
from ..publication import publish_staging
from ..runtime import configure_process_runtime, spawn_context
from ..writer import compressor_from_spec
from .autotune import (
    resolve_auto_space_workers,
    resolve_auto_tile_size,
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


COMPUTE_DTYPES = ("source", "float32")


@dataclass(frozen=True)
class _TileMetrics:
    """Timing data returned by one spatial worker without retaining arrays."""

    task: tuple[int, int, int, int]
    covered: bool
    elapsed: float
    weight_seconds: float = 0.0
    read_seconds: float = 0.0
    regrid_seconds: float = 0.0
    write_seconds: float = 0.0


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
    """Resolve the user-selected root for intermediate stores and weights."""

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
    root.mkdir(parents=True, exist_ok=True)
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
            compressor = compressor_from_spec(codec)
            if compressor is None:
                entry.pop("compressors", None)
            else:
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


def _chunk_boundaries(size: int, chunk: int) -> set[int]:
    size = int(size)
    chunk = max(1, int(chunk))
    return set(range(0, size, chunk)) | {size}


def _aligned_tile_ranges(
    info: DatasetInfo,
    target: TargetGrid,
    tile_size: int,
    chunks_by_name: dict[str, tuple[int, ...]] | None = None,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Build ranges that never split a spatial output Zarr chunk.

    Different variables may have different chunk layouts.  The union of all
    chunk boundaries is therefore used.  Adjacent complete chunks are merged
    until the requested tile-size limit is reached.  A single stored chunk
    larger than that limit is kept intact because splitting it would make
    concurrent workers perform unsafe read-modify-write operations.
    """

    lat_boundaries = {0, int(target.lat.size)}
    lon_boundaries = {0, int(target.lon.size)}
    for variable in info.data_variables:
        if "lat" not in variable.dims or "lon" not in variable.dims:
            continue
        chunks = (
            variable.chunks
            if chunks_by_name is None
            else chunks_by_name.get(variable.name, variable.chunks)
        )
        lat_index = variable.dims.index("lat")
        lon_index = variable.dims.index("lon")
        lat_boundaries.update(
            _chunk_boundaries(target.lat.size, chunks[lat_index])
        )
        lon_boundaries.update(
            _chunk_boundaries(target.lon.size, chunks[lon_index])
        )

    def merge(boundaries: set[int]) -> list[tuple[int, int]]:
        ordered = sorted(boundaries)
        ranges: list[tuple[int, int]] = []
        start_index = 0
        limit = max(1, int(tile_size))
        while start_index < len(ordered) - 1:
            start = ordered[start_index]
            end_index = start_index + 1
            while (
                end_index + 1 < len(ordered)
                and ordered[end_index + 1] - start <= limit
            ):
                end_index += 1
            stop = ordered[end_index]
            ranges.append((start, stop))
            start_index = end_index
        return ranges

    return merge(lat_boundaries), merge(lon_boundaries)


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


def _intermediate_chunks(
    info: DatasetInfo,
    plan: ResamplePlan,
) -> dict[str, tuple[int, ...]]:
    """Build a compute-oriented layout for the ephemeral intermediate store.

    Time chunks follow the vectorized computation batch.  Spatial chunks are
    shared by every resampled variable so workers can own independent tiles
    even when the final Zarr layout contains a much larger spatial chunk.
    """

    result = dict(plan.output_chunks)
    for item in info.data_variables:
        if item.name not in result or "time" not in item.dims:
            continue
        if "lat" not in item.dims or "lon" not in item.dims:
            continue
        time_index = item.dims.index("time")
        chunks = list(result[item.name])
        chunks[time_index] = min(
            int(chunks[time_index]),
            int(plan.time_block),
            int(item.shape[time_index]),
        )
        for dimension in ("lat", "lon"):
            axis = item.dims.index(dimension)
            chunks[axis] = min(
                int(plan.tile_size),
                int(plan.target.dimensions[dimension]),
            )
        result[item.name] = tuple(chunks)
    return result


def _needs_intermediate(
    info: DatasetInfo,
    plan: ResamplePlan,
    intermediate_chunks: dict[str, tuple[int, ...]],
) -> bool:
    for item in info.data_variables:
        if item.name not in intermediate_chunks or "time" not in item.dims:
            continue
        if "lat" not in item.dims or "lon" not in item.dims:
            continue
        if intermediate_chunks[item.name] != plan.output_chunks[item.name]:
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


def _relative_region(
    region: tuple[slice, ...],
    subregion: tuple[slice, ...],
) -> tuple[slice, ...]:
    return tuple(
        slice(
            int(sub.start) - int(full.start),
            int(sub.stop) - int(full.start),
        )
        for full, sub in zip(region, subregion)
    )


def _merge_intermediate_variable(
    intermediate_group,
    final_group,
    name: str,
    temporary_root: Path,
) -> None:
    """Assemble compute chunks and write each final chunk exactly once."""

    source_array = intermediate_group[name]
    target_array = final_group[name]
    for region in _chunk_regions(
        tuple(int(size) for size in target_array.shape),
        tuple(int(chunk) for chunk in target_array.chunks),
    ):
        buffer_shape = tuple(int(part.stop - part.start) for part in region)
        buffer_path = temporary_root / f".resample-buffer-{uuid4().hex}.bin"
        buffer = np.memmap(
            buffer_path,
            mode="w+",
            dtype=np.dtype(target_array.dtype),
            shape=buffer_shape,
        )
        try:
            starts = [
                range(
                    int(part.start),
                    int(part.stop),
                    max(1, int(source_chunk)),
                )
                for part, source_chunk in zip(region, source_array.chunks)
            ]
            for origin in itertools.product(*starts):
                subregion = tuple(
                    slice(
                        int(start),
                        min(
                            int(start) + int(source_chunk),
                            int(part.stop),
                        ),
                    )
                    for start, source_chunk, part in zip(
                        origin,
                        source_array.chunks,
                        region,
                    )
                )
                buffer[_relative_region(region, subregion)] = source_array[subregion]
            buffer.flush()
            target_array[region] = buffer
        finally:
            del buffer
            try:
                buffer_path.unlink()
            except OSError:
                pass


def _merge_intermediate_store(
    intermediate: Path,
    staging: Path,
    spatial_items: tuple[VariableInfo, ...],
    temporary_root: Path,
    *,
    progress: bool,
) -> None:
    source_group = zarr.open_group(intermediate, mode="r")
    target_group = zarr.open_group(staging, mode="r+")
    try:
        for index, item in enumerate(spatial_items, start=1):
            _merge_intermediate_variable(
                source_group,
                target_group,
                item.name,
                temporary_root,
            )
            if progress:
                print(f"中转数据合并：{index}/{len(spatial_items)} 个变量")
    finally:
        source_group.store.close() if hasattr(source_group.store, "close") else None
        target_group.store.close() if hasattr(target_group.store, "close") else None


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
    task: tuple[int, int, int, int],
) -> None:
    """Materialize NaN for a target tile with no source-cell overlap."""

    lat_start, lat_stop, lon_start, lon_stop = task
    for item in spatial_items:
        dtype = np.dtype(item.dtype)
        if not np.issubdtype(dtype, np.floating):
            dtype = np.dtype("float64")
        elif plan.compute_dtype == "float32":
            dtype = np.dtype("float32")
        shape = tuple(
            {
                "time": int(item.shape[item.dims.index("time")]),
                "lat": lat_stop - lat_start,
                "lon": lon_stop - lon_start,
            }[dim]
            for dim in item.dims
        )
        values = np.full(shape, np.nan, dtype=dtype)
        _write_region(
            output_group,
            item.name,
            values,
            item.dims,
            {
                "time": slice(0, int(item.shape[item.dims.index("time")])),
                "lat": slice(lat_start, lat_stop),
                "lon": slice(lon_start, lon_stop),
            },
        )


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
) -> dict[str, float]:
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
    for time_start, time_stop in _time_slices(item, time_block):
        read_started = time.perf_counter()
        subset = prepared.isel(
            time=slice(time_start, time_stop),
            lat=source_lat_slice,
            lon=source_lon_slice,
        ).transpose(*input_dims)
        # Materialize one bounded time batch before calling xESMF.  xESMF then
        # receives a NumPy array with a leading time axis and applies one
        # vectorized sparse operation, rather than constructing and executing
        # a Dask graph for every individual time step.
        if hasattr(subset.data, "compute"):
            with dask.config.set(
                scheduler="threads",
                num_workers=max(1, int(compute_workers)),
            ):
                subset = subset.compute()
        if before_replacements.rules:
            subset = subset.copy(data=apply_replacement_rules(
                np.asarray(subset.data), before_replacements, statistics
            ))
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
        with dask.config.set(scheduler="threads", num_workers=max(1, int(compute_workers))):
            if hasattr(values, "compute"):
                values = values.compute()
        timing["regrid"] += time.perf_counter() - regrid_started
        values = np.asarray(
            values,
            dtype=_resampled_output_dtype(source_variable, compute_dtype),
        )
        if after_replacements.rules:
            values = apply_replacement_rules(values, after_replacements, statistics)
        write_started = time.perf_counter()
        _write_region(
            output_group,
            item.name,
            values,
            expected_dims,
            {
                "time": slice(time_start, time_stop),
                "lat": lat_slice,
                "lon": lon_slice,
            },
        )
        timing["write"] += time.perf_counter() - write_started
        del subset, result, values
    return timing


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
        spatial_items=spatial_items,
        workdir=Path(workdir),
    )


def _process_space_tile(
    task: tuple[int, int, int, int],
) -> _TileMetrics:
    """Process one output-chunk-aligned spatial region in a child process."""

    source = _SPACE_WORKER_STATE["source"]
    output_group = _SPACE_WORKER_STATE["output_group"]
    plan = _SPACE_WORKER_STATE["plan"]
    spatial_items = _SPACE_WORKER_STATE["spatial_items"]
    workdir = _SPACE_WORKER_STATE["workdir"]
    assert isinstance(source, xr.Dataset)
    assert isinstance(output_group, zarr.Group)
    assert isinstance(plan, ResamplePlan)
    assert isinstance(workdir, Path)

    started = time.perf_counter()
    lat_start, lat_stop, lon_start, lon_stop = task
    tile = _tile_target(plan.target, lat_start, lat_stop, lon_start, lon_stop)
    target_tile, source_lat_slice, source_lon_slice = _resolve_local_source_window(
        plan.inspection.grid,
        tile,
        plan.method,
    )
    if source_lat_slice is None or source_lon_slice is None:
        _fill_missing_tile(output_group, spatial_items, plan, task)
        return _TileMetrics(task, False, time.perf_counter() - started)

    grid = plan.inspection.grid
    regridder = None
    weight_path = None
    timing = _empty_timing()
    weight_seconds = 0.0
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
            # A local source window is no longer a complete periodic grid.
            periodic=bool(
                grid.periodic
                and lon_start_source == 0
                and lon_stop_source == grid.lon.size
            ),
        )
        weight_seconds = time.perf_counter() - weights_started
        for item in spatial_items:
            _add_timing(timing, _resample_tile_variable(
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
            ))
        return _TileMetrics(
            task,
            True,
            time.perf_counter() - started,
            weight_seconds,
            timing["read"],
            timing["regrid"],
            timing["write"],
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
    task: tuple[int, int, int, int],
    workdir: Path,
    spatial_items: tuple[VariableInfo, ...],
    full_regridder=None,
) -> _TileMetrics:
    """Execute one tile in the caller, used for d2s and tiny jobs."""

    started = time.perf_counter()
    lat_start, lat_stop, lon_start, lon_stop = task
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
            _fill_missing_tile(output_group, spatial_items, plan, task)
            return _TileMetrics(task, False, time.perf_counter() - started)
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
    try:
        for item in spatial_items:
            _add_timing(timing, _resample_tile_variable(
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
            ))
        return _TileMetrics(
            task,
            True,
            time.perf_counter() - started,
            weight_seconds if full_regridder is None else 0.0,
            timing["read"],
            timing["regrid"],
            timing["write"],
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
    tasks: Iterable[tuple[int, int, int, int]],
    total_tasks: int,
    *,
    cancel_event=None,
    progress: bool,
) -> dict[str, float]:
    """Run bounded, spawn-safe spatial work without submitting all tiles."""

    context = spawn_context()
    executor = ProcessPoolExecutor(
        max_workers=max(1, int(plan.space_workers)),
        mp_context=context,
        initializer=_initialize_space_worker,
        initargs=(str(source_path), str(staging), plan, spatial_items, str(workdir)),
    )
    pending = set()
    task_iter = iter(tasks)
    completed = 0
    cancelled = False
    timing = _empty_timing()
    weight_seconds = 0.0
    tile_elapsed = 0.0
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
                tile_elapsed += metrics.elapsed
                weight_seconds += metrics.weight_seconds
                _add_timing(
                    timing,
                    {
                        "read": metrics.read_seconds,
                        "regrid": metrics.regrid_seconds,
                        "write": metrics.write_seconds,
                    },
                )
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
        "tiles": float(completed),
        "tile_elapsed_seconds": tile_elapsed,
        "weight_seconds": weight_seconds,
        "read_seconds": timing["read"],
        "regrid_seconds": timing["regrid"],
        "write_seconds": timing["write"],
    }


def plan_resample(
    config: ResampleConfig,
    inspection: ResampleInspection | None = None,
) -> ResamplePlan:
    inspection = inspection or inspect_resample_input(config.input)
    config_path = Path(config.input).expanduser().resolve()
    if inspection.info.path != config_path:
        raise ValueError("重采样检查结果与配置中的输入路径不一致，请重新检查输入。")
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
        )
    else:
        try:
            selected_time_block = int(config.time_block)
        except (TypeError, ValueError) as exc:
            raise ValueError("time_block 必须是正整数或 auto。") from exc
        if selected_time_block <= 0:
            raise ValueError("time_block 必须是正整数或 auto。")
    if config.space_workers == "auto":
        selected_space_workers = resolve_auto_space_workers(
            compute_workers=int(config.compute_workers),
        )
    else:
        try:
            selected_space_workers = int(config.space_workers)
        except (TypeError, ValueError) as exc:
            raise ValueError("space_workers 必须是正整数或 auto。") from exc
        if selected_space_workers <= 0:
            raise ValueError("space_workers 必须是正整数或 auto。")
    for name in ("compute_workers",):
        if int(getattr(config, name)) <= 0:
            raise ValueError(f"{name} 必须是正整数。")
    target = build_target_grid(
        inspection.grid,
        config.resolution,
        extent=config.extent,
        lat_bounds=config.target_lat_bounds,
        lon_bounds=config.target_lon_bounds,
        lat_descending=config.target_lat_descending,
        lon_descending=config.target_lon_descending,
    )
    auto_tile = None
    if config.tile_size == "auto":
        worker_candidates = (
            range(selected_space_workers, 0, -1)
            if config.space_workers == "auto"
            else (selected_space_workers,)
        )
        for candidate_workers in worker_candidates:
            candidate = resolve_auto_tile_size(
                inspection.info,
                inspection.grid,
                target,
                method=config.method,
                skipna=config.skipna,
                time_block=selected_time_block,
                compute_workers=int(config.compute_workers),
                space_workers=candidate_workers,
                compute_dtype=config.compute_dtype,
            )
            auto_tile = candidate
            if candidate.fits_budget or candidate_workers == 1:
                selected_space_workers = candidate_workers
                break
        assert auto_tile is not None
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
    return ResamplePlan(
        inspection=inspection,
        target=target,
        method=config.method,
        skipna=config.skipna,
        na_thres=float(config.na_thres),
        output_chunks=planned_output_chunks,
        compute_dtype=config.compute_dtype,
        tile_size=selected_tile_size,
        time_block=selected_time_block,
        time_block_requested=config.time_block,
        compute_workers=int(config.compute_workers),
        space_workers=selected_space_workers,
        tile_size_requested=config.tile_size,
        space_workers_requested=config.space_workers,
        auto_tile=auto_tile,
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
        mask_and_scale=False,
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
            values = np.asarray(array[region])
            replaced = apply_replacement_rules(
                values,
                plan.after_replacements,
                statistics.get(name, {}),
            )
            array[region] = replaced.astype(array.dtype, copy=False)
    return statistics, mode


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
    _prepare_target(target_path, config.overwrite)
    temporary_root = _resolve_temporary_root(
        source_path,
        target_path,
        config.temporary_dir,
    )
    staging = target_path.parent / f".{target_path.name}.resample-{uuid4().hex}.tmp"
    workdir = temporary_root / f".{target_path.name}.weights-{uuid4().hex}.tmp"
    intermediate: Path | None = None
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
            mask_and_scale=False,
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
        intermediate_chunks = _intermediate_chunks(
            runtime_plan.inspection.info,
            runtime_plan,
        )
        use_intermediate = _needs_intermediate(
            runtime_plan.inspection.info,
            runtime_plan,
            intermediate_chunks,
        )
        processing_path = staging
        if use_intermediate:
            intermediate = temporary_root / (
                f".{target_path.name}.intermediate-{uuid4().hex}.tmp"
            )
            processing_plan = replace(
                processing_plan,
                output_chunks=intermediate_chunks,
            )
            _initialize_output_store(
                source,
                runtime_plan.inspection.info,
                processing_plan,
                intermediate,
            )
            processing_path = intermediate
            if progress:
                print(
                    "检测到时间块小于最终输出 time chunk，启用临时中转 Zarr；"
                    "完成后按最终 chunk 一次性合并。"
                )
                print(f"中间处理目录：{temporary_root}")
        output_group = zarr.open_group(processing_path, mode="r+")
        if cancel_event is not None and cancel_event.is_set():
            raise ResampleExecutionError("任务已取消，未生成输出。")

        grid = runtime_plan.inspection.grid
        spatial_items = tuple(
            item
            for item in runtime_plan.inspection.info.data_variables
            if item.name in spatial_names
        )
        if use_intermediate:
            lat_ranges = list(
                _tile_ranges(runtime_plan.target.lat.size, runtime_plan.tile_size)
            )
            lon_ranges = list(
                _tile_ranges(runtime_plan.target.lon.size, runtime_plan.tile_size)
            )
            tile_layout_description = "空间计算块与最终 chunks 解耦"
        else:
            lat_ranges, lon_ranges = _aligned_tile_ranges(
                runtime_plan.inspection.info,
                runtime_plan.target,
                runtime_plan.tile_size,
                processing_plan.output_chunks,
            )
            tile_layout_description = "空间块已对齐最终输出 chunks"
        total_tiles = len(lat_ranges) * len(lon_ranges)

        def tile_tasks():
            return (
                (lat_start, lat_stop, lon_start, lon_stop)
                for lat_start, lat_stop in lat_ranges
                for lon_start, lon_stop in lon_ranges
            )

        if progress:
            print(
                f"流式计算：{total_tiles} 个空间块，"
                f"空间进程={runtime_plan.space_workers}，"
                f"每个进程使用 {runtime_plan.compute_workers} 个 Dask 线程；"
                f"{tile_layout_description}。"
            )
        full_regridder = None
        full_weight_path = None
        tile_timing = {
            "tiles": 0.0,
            "tile_elapsed_seconds": 0.0,
            "weight_seconds": 0.0,
            "read_seconds": 0.0,
            "regrid_seconds": 0.0,
            "write_seconds": 0.0,
        }
        if runtime_plan.method == "nearest_d2s":
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
            # stays in the caller.  Tiny jobs also avoid the spawn overhead.
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

        if use_intermediate:
            if progress:
                print("开始将中转数据合并为最终输出 chunks……")
            _merge_intermediate_store(
                processing_path,
                staging,
                spatial_items,
                temporary_root,
                progress=progress,
            )

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
        if intermediate is not None:
            shutil.rmtree(intermediate, ignore_errors=True)
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
            "used_intermediate": intermediate is not None,
            "intermediate_logical_bytes": logical_bytes if use_intermediate else 0,
            "throughput_mib_s": logical_bytes / 1024**2 / max(elapsed, 1e-9),
            "physical_throughput_mib_s": (
                physical_bytes / 1024**2 / max(elapsed, 1e-9)
            ),
            "logical_write_amplification": 2.0 if use_intermediate else 1.0,
            "space_workers": min(runtime_plan.space_workers, total_tiles),
            "compute_workers_per_space_worker": runtime_plan.compute_workers,
            "temporary_dir": str(temporary_root),
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
            if intermediate is not None:
                shutil.rmtree(intermediate, ignore_errors=True)
            raise ResampleExecutionError("任务已取消，未生成输出。") from exc
        if isinstance(exc, ResampleExecutionError):
            raise
        raise ResampleExecutionError(
            f"重采样失败；输出临时目录已清理：{staging}\n"
            + (
                f"中间目录保留用于排查：{intermediate}\n"
                if intermediate is not None
                else ""
            )
            + str(exc)
        ) from exc
    finally:
        if source is not None:
            source.close()
        shutil.rmtree(workdir, ignore_errors=True)
