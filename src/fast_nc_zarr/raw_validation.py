from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from .application.services import SourceInspectionConfig, inspect_source
from .filename_mode import convert_filename, normalize_filename_dataset
from .models import Selection
from .time_mapping import TimeFieldRef, TimeRule, inspect_time_metadata


def _sample_indices(size: int, count: int) -> tuple[int, ...]:
    if size <= 0:
        return ()
    return tuple(sorted({int(value) for value in np.linspace(0, size - 1, min(size, count))}))


def _axis_report(values: np.ndarray, name: str) -> dict[str, Any]:
    numeric = np.asarray(values, dtype="float64")
    if numeric.ndim != 1 or numeric.size < 2:
        raise ValueError(f"{name} 必须是至少含两个点的一维坐标。")
    differences = np.diff(numeric)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} 含有非有限坐标。")
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise ValueError(f"{name} 不是严格单调坐标。")
    step = float(np.median(differences))
    tolerance = max(abs(step) * 1e-6, 1e-10)
    maximum_error = float(np.max(np.abs(differences - step)))
    if maximum_error > tolerance:
        raise ValueError(
            f"{name} 不是规则网格：最大步长偏差 {maximum_error:g} > {tolerance:g}。"
        )
    return {
        "size": int(numeric.size),
        "first": float(numeric[0]),
        "last": float(numeric[-1]),
        "ascending": bool(step > 0),
        "step": step,
        "maximum_step_error": maximum_error,
    }


def _sample_source_values(inventory, sample_files: int) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    file_indices = _sample_indices(len(inventory.files), sample_files)
    variable_names = tuple(
        name
        for name, spec in inventory.variables.items()
        if np.dtype(spec.dtype).kind in "biufc"
    )
    for file_index in file_indices:
        record = inventory.files[file_index]
        dataset, engine = normalize_filename_dataset(record.path, inventory.source_engine)
        try:
            lat_indices = _sample_indices(int(dataset.sizes["lat"]), 3)
            lon_indices = _sample_indices(int(dataset.sizes["lon"]), 3)
            variables = []
            for name in variable_names:
                if name not in dataset or set(dataset[name].dims) != {"lat", "lon"}:
                    continue
                values = np.asarray(
                    dataset[name].isel(lat=list(lat_indices), lon=list(lon_indices)).values
                )
                finite = values[np.isfinite(values)] if values.dtype.kind in "fc" else values
                variables.append(
                    {
                        "name": name,
                        "dtype": str(values.dtype),
                        "sample_count": int(values.size),
                        "finite_count": int(finite.size),
                        "minimum": float(np.min(finite)) if finite.size else None,
                        "maximum": float(np.max(finite)) if finite.size else None,
                    }
                )
            if not variables:
                raise ValueError(f"{record.path.name} 没有可抽样的数值型 lat/lon 变量。")
            results.append(
                {
                    "file_index": file_index,
                    "path": str(record.path),
                    "engine": engine,
                    "variables": variables,
                }
            )
        finally:
            dataset.close()
            del dataset
            gc.collect()
    return results


def _smoke_convert(inventory, output: Path) -> dict[str, Any]:
    variables = tuple(
        name
        for name, spec in inventory.variables.items()
        if spec.direct_compatible
    )
    if not variables:
        raise ValueError("没有可用于转换冒烟测试的三维数值变量。")
    ny = int(inventory.lat_values.size)
    nx = int(inventory.lon_values.size)
    lat_start = max(0, ny // 2 - 16)
    lon_start = max(0, nx // 2 - 16)
    selection = Selection(
        variables=(variables[0],),
        time_start=0,
        time_stop=min(2, int(inventory.times.size)),
        lat_start=lat_start,
        lat_stop=min(ny, lat_start + 32),
        lon_start=lon_start,
        lon_stop=min(nx, lon_start + 32),
    )
    started = time.perf_counter()
    plan, metrics = convert_filename(
        inventory,
        selection,
        output,
        auto_tune=False,
        max_workers=2,
        overwrite=True,
        validate=True,
        progress=False,
    )
    return {
        "output": str(output),
        "selection": asdict(selection),
        "plan": asdict(plan),
        "metrics": metrics,
        "elapsed": time.perf_counter() - started,
    }


def validate_raw_tree(
    input_root: Path,
    *,
    workers: int | None = None,
    sample_files: int = 9,
    smoke_output_root: Path | None = None,
    time_field_overrides: dict[str, int] | None = None,
) -> dict[str, Any]:
    root = input_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"原始数据根目录不存在：{root}")
    datasets = sorted(path for path in root.iterdir() if path.is_dir())
    if not datasets:
        raise FileNotFoundError(f"原始数据根目录下没有数据集目录：{root}")
    if smoke_output_root is not None:
        smoke_output_root = smoke_output_root.expanduser().resolve()
        smoke_output_root.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root),
        "datasets": [],
    }
    time_field_overrides = time_field_overrides or {}
    for source in datasets:
        started = time.perf_counter()
        time_inspection = inspect_time_metadata(source)
        time_rule = time_inspection.suggested_rule
        if source.name in time_field_overrides:
            field_index = time_field_overrides[source.name]
            matching = [
                item for item in time_inspection.filename_fields if item.index == field_index
            ]
            if not matching:
                raise ValueError(
                    f"数据集 {source.name} 不存在文件名字段 {field_index}。"
                )
            time_rule = TimeRule(
                full=TimeFieldRef(source="filename", component="full", index=field_index)
            )
        if time_rule is None:
            raise ValueError(
                f"数据集 {source.name} 的时间字段存在歧义；"
                f"请传入 --time-field {source.name}=字段索引。"
            )
        result = inspect_source(
            SourceInspectionConfig(
                input_dir=source,
                mode="auto",
                workers=workers,
                time_rule=time_rule,
                time_inspection=time_inspection,
            )
        )
        inventory = result.source_inventory
        dataset_report = {
            "name": source.name,
            "status": "passed",
            "mode": result.mode,
            "engine": inventory.source_engine,
            "file_count": len(inventory.files),
            "total_bytes": int(inventory.total_bytes),
            "time_count": int(inventory.times.size),
            "time_start": str(inventory.times[0].astype("datetime64[D]")),
            "time_end": str(inventory.times[-1].astype("datetime64[D]")),
            "frequency": inventory.frequency,
            "missing_time_count": len(inventory.missing_time_keys),
            "latitude": _axis_report(inventory.lat_values, "lat"),
            "longitude": _axis_report(inventory.lon_values, "lon"),
            "variables": [
                {
                    "name": spec.name,
                    "dims": list(spec.dims),
                    "dtype": spec.dtype,
                    "native_chunks": list(spec.native_chunks or ()),
                }
                for spec in inventory.variables.values()
            ],
            "source_samples": _sample_source_values(inventory, sample_files),
            "inspection_elapsed": time.perf_counter() - started,
        }
        if smoke_output_root is not None:
            dataset_report["conversion_smoke"] = _smoke_convert(
                inventory,
                smoke_output_root / f"{source.name}.zarr",
            )
        report["datasets"].append(dataset_report)
    report["status"] = "passed"
    report["file_count"] = sum(item["file_count"] for item in report["datasets"])
    report["total_bytes"] = sum(item["total_bytes"] for item in report["datasets"])
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="对真实 NC/HDF/TIFF 数据集执行全文件元数据和抽样数值正确性验证。"
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--sample-files", type=int, default=9)
    parser.add_argument("--smoke-output-root", type=Path)
    parser.add_argument(
        "--time-field",
        action="append",
        default=[],
        metavar="DATASET=INDEX",
        help="为存在歧义的数据集指定文件名完整日期字段索引。",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_files < 1:
        raise ValueError("--sample-files 必须是正整数。")
    overrides: dict[str, int] = {}
    for value in args.time_field:
        name, separator, index = value.partition("=")
        if not separator or not name.strip():
            raise ValueError("--time-field 必须使用 DATASET=INDEX 格式。")
        try:
            overrides[name.strip()] = int(index)
        except ValueError as exc:
            raise ValueError("--time-field 的 INDEX 必须是整数。") from exc
    report = validate_raw_tree(
        args.input_root,
        workers=args.workers,
        sample_files=args.sample_files,
        smoke_output_root=args.smoke_output_root,
        time_field_overrides=overrides,
    )
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(
        f"真实数据验证通过：{report['file_count']} 个文件，"
        f"{len(report['datasets'])} 个数据集；报告：{destination}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
