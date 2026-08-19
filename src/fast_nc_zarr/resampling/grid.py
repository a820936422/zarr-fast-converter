from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from ..rechunking.inspection import inspect_store
from ..rechunking.models import DatasetInfo
from .models import GridInfo, ResampleExtent, TargetGrid


RESAMPLING_METHODS = (
    "bilinear",
    "conservative",
    "conservative_normed",
    "patch",
    "nearest_s2d",
    "nearest_d2s",
)


class GridInspectionError(ValueError):
    """Raised when a Zarr store is outside the first resampling scope."""


def _axis_bounds(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    differences = np.diff(values)
    bounds = np.empty(values.size + 1, dtype="float64")
    bounds[1:-1] = (values[:-1] + values[1:]) / 2.0
    bounds[0] = values[0] - differences[0] / 2.0
    bounds[-1] = values[-1] + differences[-1] / 2.0
    return bounds

def _axis_is_uniform(values: np.ndarray, resolution: float) -> bool:
    """Accept spacing noise caused by storing global coordinates as float32."""
    values = np.asarray(values, dtype="float64")
    scale = max(float(np.max(np.abs(values))), abs(float(resolution)), 1.0)
    float32_quantization = abs(float(np.spacing(np.float32(scale)))) * 2.0
    return bool(
        np.allclose(
            np.abs(np.diff(values)),
            resolution,
            rtol=1e-5,
            atol=max(resolution * 1e-6, float32_quantization, 1e-10),
        )
    )


def _validate_axis(
    dataset: xr.Dataset,
    name: str,
) -> tuple[np.ndarray, np.ndarray, float, bool, bool]:
    if name not in dataset.coords:
        raise GridInspectionError(f"输入 Zarr 缺少 {name} 坐标。")
    coordinate = dataset[name]
    if coordinate.dims != (name,) or coordinate.ndim != 1:
        raise GridInspectionError(
            f"第一版重采样要求 {name} 是一维坐标，实际维度为 {coordinate.dims}。"
        )
    try:
        values = np.asarray(coordinate.values, dtype="float64")
    except (TypeError, ValueError) as exc:
        raise GridInspectionError(f"{name} 坐标不是数值型。") from exc
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise GridInspectionError(f"{name} 坐标至少需要两个有限数值。")
    differences = np.diff(values)
    if np.any(differences == 0) or not (
        np.all(differences > 0) or np.all(differences < 0)
    ):
        raise GridInspectionError(f"{name} 坐标必须严格单调。")
    resolution = float(np.median(np.abs(differences)))
    uniform = _axis_is_uniform(values, resolution)
    if not uniform:
        raise GridInspectionError(
            f"第一版重采样要求 {name} 为规则网格，检测到非等间距坐标。"
        )
    return values, _axis_bounds(values), resolution, bool(values[0] > values[-1]), uniform


def inspect_dataset_grid(
    dataset: xr.Dataset,
    info: DatasetInfo,
    *,
    path: Path | str | None = None,
) -> GridInfo:
    """Validate a regular grid from an already-open dataset.

    Used both by the on-disk ``inspect_grid`` and by the single-pass fusion
    path, where the source is an in-memory xarray Dataset and therefore has no
    real filesystem path.
    """

    lat, lat_bounds, lat_resolution, lat_descending, lat_uniform = _validate_axis(
        dataset, "lat"
    )
    lon, lon_bounds, lon_resolution, lon_descending, lon_uniform = _validate_axis(
        dataset, "lon"
    )
    invalid = [
        variable.name
        for variable in info.data_variables
        if variable.ndim and variable.dtype.kind not in "biuf"
    ]
    if invalid:
        raise GridInspectionError(
            "重采样只支持数值型数据变量，以下变量类型不支持：" + ", ".join(invalid)
        )
    return GridInfo(
        path=Path(path) if path is not None else Path("<fused-in-memory>"),
        lat=lat,
        lon=lon,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        lat_resolution=lat_resolution,
        lon_resolution=lon_resolution,
        lat_descending=lat_descending,
        lon_descending=lon_descending,
        lat_uniform=lat_uniform,
        lon_uniform=lon_uniform,
    )


def inspect_grid(
    path: str | Path,
    info: DatasetInfo | None = None,
) -> tuple[DatasetInfo, GridInfo]:
    """Reuse the Zarr inspection and additionally validate the source grid."""

    source = Path(path).expanduser().resolve()
    info = info or inspect_store(source)
    try:
        dataset = xr.open_zarr(
            source,
            consolidated=False,
            chunks=None,
            decode_times=False,
            mask_and_scale=False,
        )
    except Exception as exc:
        raise GridInspectionError(f"无法读取 Zarr 空间坐标：{source}") from exc
    try:
        return info, inspect_dataset_grid(dataset, info, path=source)
    finally:
        dataset.close()


def _target_axis(
    source_bounds: np.ndarray,
    resolution: float,
    descending: bool,
) -> tuple[np.ndarray, np.ndarray]:
    low = float(np.min(source_bounds))
    high = float(np.max(source_bounds))
    span = high - low
    count = max(1, int(np.ceil(span / resolution - 1e-10)))
    edges = low + np.arange(count + 1, dtype="float64") * resolution
    centers = (edges[:-1] + edges[1:]) / 2.0
    if descending:
        return centers[::-1], edges[::-1]
    return centers, edges


def _global_axis(
    low: float,
    high: float,
    resolution: float,
    descending: bool,
) -> tuple[np.ndarray, np.ndarray]:
    span = high - low
    count = int(round(span / resolution))
    if count <= 0 or not np.isclose(count * resolution, span, rtol=0.0, atol=1e-8):
        raise GridInspectionError(
            f"全球范围 {span:g} 度不能被目标分辨率 {resolution:g} 整除。"
        )
    edges = low + np.arange(count + 1, dtype="float64") * resolution
    centers = (edges[:-1] + edges[1:]) / 2.0
    if descending:
        return centers[::-1], edges[::-1]
    return centers, edges


def _custom_axis(
    low: float,
    high: float,
    resolution: float,
    descending: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a resolution-aligned target axis covering the requested extent.

    User-provided bounds commonly come from source cell centers (for example
    ``89.975``), while target resolution describes output cell edges.  Expand
    the bounds outward to the nearest resolution edge instead of rejecting a
    valid coverage request merely because its center-derived extent is not an
    exact multiple of the target resolution.
    """

    low = float(low)
    high = float(high)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise GridInspectionError("目标范围必须是有限且严格递增的外边界。")
    lower_steps = low / resolution
    upper_steps = high / resolution
    tolerance = max(1e-10, abs(lower_steps), abs(upper_steps)) * 1e-12
    aligned_low = float(np.floor(lower_steps + tolerance) * resolution)
    aligned_high = float(np.ceil(upper_steps - tolerance) * resolution)
    if aligned_high <= aligned_low:
        raise GridInspectionError("目标范围在当前分辨率下没有有效网格单元。")
    count = int(round((aligned_high - aligned_low) / resolution))
    if count <= 0:
        raise GridInspectionError("目标范围在当前分辨率下没有有效网格单元。")
    edges = aligned_low + np.arange(count + 1, dtype="float64") * resolution
    edges[0] = aligned_low
    edges[-1] = aligned_high
    centers = (edges[:-1] + edges[1:]) / 2.0
    if descending:
        return centers[::-1], edges[::-1]
    return centers, edges


def build_target_grid(
    grid: GridInfo,
    resolution: float,
    *,
    extent: ResampleExtent = "source",
    lat_bounds: tuple[float, float] | None = None,
    lon_bounds: tuple[float, float] | None = None,
    lat_descending: bool | None = None,
    lon_descending: bool | None = None,
) -> TargetGrid:
    if extent not in {"source", "global", "custom"}:
        raise GridInspectionError("目标范围只能是 source、global 或 custom。")
    try:
        resolution = float(resolution)
    except (TypeError, ValueError) as exc:
        raise GridInspectionError("目标分辨率必须是正数。") from exc
    if not np.isfinite(resolution) or resolution <= 0:
        raise GridInspectionError("目标分辨率必须是正数。")

    lat_direction = grid.lat_descending if lat_descending is None else bool(lat_descending)
    lon_direction = grid.lon_descending if lon_descending is None else bool(lon_descending)

    if extent == "source":
        lat, target_lat_bounds = _target_axis(
            grid.lat_bounds, resolution, lat_direction
        )
        lon, target_lon_bounds = _target_axis(
            grid.lon_bounds, resolution, lon_direction
        )
    elif extent == "global":
        lat, target_lat_bounds = _global_axis(-90.0, 90.0, resolution, lat_direction)
        lon, target_lon_bounds = _global_axis(-180.0, 180.0, resolution, lon_direction)
    else:
        if lat_bounds is None or lon_bounds is None:
            raise GridInspectionError(
                "custom 目标范围必须同时提供 lat_bounds 和 lon_bounds。"
            )
        lat, target_lat_bounds = _custom_axis(
            float(min(lat_bounds)),
            float(max(lat_bounds)),
            resolution,
            lat_direction,
        )
        lon, target_lon_bounds = _custom_axis(
            float(min(lon_bounds)),
            float(max(lon_bounds)),
            resolution,
            lon_direction,
        )
    return TargetGrid(
        lat=lat,
        lon=lon,
        lat_bounds=target_lat_bounds,
        lon_bounds=target_lon_bounds,
        lat_resolution=resolution,
        lon_resolution=resolution,
        extent=extent,
    )


def output_chunks(info: DatasetInfo, target: TargetGrid) -> dict[str, tuple[int, ...]]:
    """Preserve each variable's chunk sizes, clipping changed dimensions."""

    target_sizes = target.dimensions
    result: dict[str, tuple[int, ...]] = {}
    for variable in info.variables:
        if not variable.ndim:
            result[variable.name] = ()
            continue
        result[variable.name] = tuple(
            min(int(chunk), int(target_sizes[dim])) if dim in target_sizes else int(chunk)
            for dim, chunk in zip(variable.dims, variable.chunks)
        )
    return result


def format_grid_report(grid: GridInfo, target: TargetGrid | None = None) -> str:
    lines = [
        "========== 空间网格检查 ==========",
        f"输入：{grid.path}",
        f"纬度：{grid.lat.size} 点，分辨率 {grid.lat_resolution:g}°，"
        f"范围 {grid.lat.min():g} .. {grid.lat.max():g}",
        f"经度：{grid.lon.size} 点，分辨率 {grid.lon_resolution:g}°，"
        f"范围 {grid.lon.min():g} .. {grid.lon.max():g}",
        f"纬度方向：{'降序' if grid.lat_descending else '升序'}；"
        f"经度方向：{'降序' if grid.lon_descending else '升序'}",
        f"是否全球周期网格：{'是' if grid.periodic else '否'}",
    ]
    if target is not None:
        lines.extend(
            [
                "",
                "目标网格：",
                f"  纬度：{target.lat.size} 点，分辨率 {target.lat_resolution:g}°",
                f"  经度：{target.lon.size} 点，分辨率 {target.lon_resolution:g}°",
                f"  范围模式：{target.extent}",
                (
                    "  边界："
                    f"lon={target.spatial_extent[0]:g} .. {target.spatial_extent[1]:g}，"
                    f"lat={target.spatial_extent[2]:g} .. {target.spatial_extent[3]:g}"
                ),
            ]
        )
    return "\n".join(lines)
