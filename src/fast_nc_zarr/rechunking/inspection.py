from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import xarray as xr

from .models import DatasetInfo, VariableInfo


REQUIRED_DIMS = ("time", "lat", "lon")


class RechunkInspectionError(ValueError):
    """Raised when an input store is outside the first-version scope."""


def _root_metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "zarr.json"
    if not metadata_path.is_file():
        raise RechunkInspectionError(
            f"输入路径不是可识别的 Zarr v3 根目录（缺少 zarr.json）：{path}"
        )
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RechunkInspectionError(f"无法读取 Zarr 根元数据：{metadata_path}") from exc
    if metadata.get("node_type") != "group":
        raise RechunkInspectionError("输入路径的 zarr.json 不是根 group。")
    return metadata


def _array_chunks(variable: xr.DataArray) -> tuple[int, ...]:
    chunks = variable.encoding.get("chunks")
    if chunks is None:
        return tuple(int(size) for size in variable.shape)
    return tuple(int(value) for value in chunks)


def _compressors(variable: xr.DataArray) -> tuple[Any, ...]:
    compressors = variable.encoding.get("compressors")
    if compressors is None:
        compressor = variable.encoding.get("compressor")
        return () if compressor is None else (compressor,)
    return tuple(compressors)


def inspect_store(path: str | Path) -> DatasetInfo:
    """Read only metadata and validate a complete three-dimensional Zarr v3 store."""

    source = Path(path).expanduser().resolve()
    if not source.is_dir():
        raise RechunkInspectionError(f"输入 Zarr 目录不存在：{source}")
    metadata = _root_metadata(source)
    zarr_format = int(metadata.get("zarr_format", 0))
    if zarr_format != 3:
        raise RechunkInspectionError(
            f"当前重分块器只支持 Zarr v3，输入格式为 v{zarr_format}。"
        )

    try:
        dataset = xr.open_zarr(
            source,
            consolidated=False,
            chunks=None,
            decode_times=False,
            mask_and_scale=False,
        )
    except Exception as exc:
        raise RechunkInspectionError(f"无法打开输入 Zarr：{source}") from exc

    try:
        dimensions = {name: int(size) for name, size in dataset.sizes.items()}
        missing_dims = [name for name in REQUIRED_DIMS if name not in dimensions]
        if missing_dims:
            raise RechunkInspectionError(
                "输入 Zarr 缺少标准维度："
                + ", ".join(missing_dims)
                + "；第一版只支持完整 (time, lat, lon) 数据。"
            )

        variables: list[VariableInfo] = []
        coordinate_names = set(dataset.coords)
        for name, variable in dataset.variables.items():
            dims = tuple(str(dim) for dim in variable.dims)
            if not variable.ndim:
                # Scalar metadata variables do not affect chunk strategy.
                pass
            elif name not in coordinate_names and set(dims) != set(REQUIRED_DIMS):
                raise RechunkInspectionError(
                    f"数据变量 {name!r} 的维度为 {dims}，"
                    "第一版要求每个非标量数据变量同时包含 time/lat/lon。"
                )
            variables.append(
                VariableInfo(
                    name=name,
                    dims=dims,
                    shape=tuple(int(size) for size in variable.shape),
                    dtype=variable.dtype,
                    chunks=_array_chunks(variable),
                    is_coord=name in coordinate_names,
                    attrs=dict(variable.attrs),
                    compressors=_compressors(variable),
                )
            )
        data_variables = tuple(item for item in variables if not item.is_coord)
        if not any(item.ndim == 3 for item in data_variables):
            raise RechunkInspectionError(
                "输入 Zarr 没有包含 time/lat/lon 的三维数据变量。"
            )
        return DatasetInfo(
            path=source,
            dimensions=dimensions,
            variables=tuple(variables),
            attrs=dict(dataset.attrs),
            zarr_format=zarr_format,
        )
    finally:
        dataset.close()


def format_inspection(info: DatasetInfo) -> str:
    """Create a concise human-readable metadata report."""

    lines = [
        "========== Zarr 输入检查 ==========",
        f"路径：{info.path}",
        f"格式：Zarr v{info.zarr_format}",
        "维度：" + ", ".join(f"{name}={size}" for name, size in info.dimensions.items()),
        f"数据变量逻辑大小：{info.logical_bytes / 1024**3:.2f} GiB",
        "",
        "变量：",
    ]
    for variable in info.variables:
        role = "坐标" if variable.is_coord else "数据"
        codec = ", ".join(repr(item) for item in variable.compressors) or "无/未知"
        lines.append(
            f"  {variable.name}: {role}, dims={variable.dims}, shape={variable.shape}, "
            f"dtype={variable.dtype}, chunks={variable.chunks}, kind={variable.kind}"
        )
        lines.append(f"    codec={codec}")
        if variable.attrs:
            keys = ", ".join(str(key) for key in variable.attrs)
            lines.append(f"    attrs: {keys}")
    return "\n".join(lines)
