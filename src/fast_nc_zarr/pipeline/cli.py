from __future__ import annotations

import argparse
from pathlib import Path

from ..application.services import SourceInspectionConfig, inspect_source
from .engine import preview_pipeline, run_pipeline
from .models import (
    PipelineConfig,
    PipelineConversionOptions,
    PipelineFinalizationOptions,
    PipelineGeneralConfig,
    PipelineResamplingOptions,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-zarr-pipeline",
        description="按检查、转换、重采样、重分块和压缩顺序生成最终 Zarr v3。",
    )
    parser.add_argument("--input", type=Path, required=True, help="源数据目录。")
    parser.add_argument("--output", type=Path, required=True, help="最终 Zarr 目录。")
    parser.add_argument("--temporary-dir", type=Path, help="临时处理根目录。")
    parser.add_argument("--time", nargs=2, metavar=("START", "END"), help="时间起止日期。")
    parser.add_argument("--lat", nargs=2, type=float, metavar=("MIN", "MAX"), required=True)
    parser.add_argument("--lon", nargs=2, type=float, metavar=("MIN", "MAX"), required=True)
    parser.add_argument("--resolution", type=float, required=True, help="目标空间分辨率。")
    parser.add_argument("--variables", nargs="+", help="待处理变量；默认全部。")
    parser.add_argument("--variable-name", action="append", default=[], metavar="SOURCE=OUTPUT")
    parser.add_argument("--method", default="bilinear")
    parser.add_argument("--skipna", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--na-thres", type=float, default=1.0)
    parser.add_argument("--compute-dtype", choices=("source", "float32"), default="source")
    parser.add_argument("--cleanup-intermediate", action="store_true")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--tune-budget", type=float, default=60.0)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--strategy", choices=("time", "space", "custom"), default="time")
    parser.add_argument("--target-mib", type=float, default=128.0)
    parser.add_argument("--custom-chunks", nargs=3, type=int)
    parser.add_argument("--compression", choices=("fast", "balanced", "maximum", "none"), default="balanced")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mode", choices=("auto", "complete", "filename"), default="auto")
    parser.add_argument("--engine", default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--inspect-workers", type=int)
    parser.add_argument(
        "--inspection-cache",
        type=Path,
        help="增量检查快照；再次运行时只重新检查新增或发生变化的源文件。",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-validate", action="store_true")
    return parser


def _parse_names(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("--variable-name 必须使用 SOURCE=OUTPUT 格式。")
        source, output = item.split("=", 1)
        source = source.strip()
        output = output.strip()
        if not source or not output:
            raise ValueError("--variable-name 的源变量名和输出变量名不能为空。")
        result[source] = output
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    time_start, time_end = args.time if args.time else (None, None)
    names = _parse_names(args.variable_name)
    inspection = inspect_source(
        SourceInspectionConfig(
            input_dir=args.input,
            mode=args.mode,
            recursive=args.recursive,
            engine=args.engine,
            workers=args.inspect_workers,
            cache_path=args.inspection_cache,
        )
    )
    print(inspection.report)
    if inspection.warnings:
        print("检查警告：")
        for warning in inspection.warnings:
            print(f"  - {warning}")
    config = PipelineConfig(
        general=PipelineGeneralConfig(
            output=args.output,
            temporary_dir=args.temporary_dir,
            time_start=time_start,
            time_end=time_end,
            lat_min=args.lat[0],
            lat_max=args.lat[1],
            lon_min=args.lon[0],
            lon_max=args.lon[1],
            resolution=args.resolution,
            cleanup_intermediate=args.cleanup_intermediate,
            overwrite=args.overwrite,
        ),
        conversion=PipelineConversionOptions(
            variables=tuple(args.variables or ()),
            variable_names=names,
            auto_tune=not args.no_tune,
            tune_budget=args.tune_budget,
            max_workers=args.max_workers,
        ),
        resampling=PipelineResamplingOptions(
            method=args.method,
            skipna=args.skipna,
            na_thres=args.na_thres,
            compute_dtype=args.compute_dtype,
        ),
        finalization=PipelineFinalizationOptions(
            strategy=args.strategy,
            target_mib=args.target_mib,
            custom_chunks=tuple(args.custom_chunks) if args.custom_chunks else None,
            compression=args.compression,
            workers=args.workers,
        ),
        validate=not args.no_validate,
    )
    plan = preview_pipeline(inspection, config)
    print(
        f"一条龙计划：目标 shape(time, lat, lon)=({len(inspection.source_inventory.times)}, "
        f"{plan.target_grid.lat.size}, {plan.target_grid.lon.size})；"
        f"源读取窗口={plan.source_read_window.lat_shape}x{plan.source_read_window.lon_shape}；"
        f"需要重采样={'是' if plan.needs_resample else '否'}"
    )
    print(f"源窗口依据：{plan.source_read_window.halo_description}")
    print(f"最终 chunks(time, lat, lon)：{plan.final_chunks}")
    if plan.final_compression is not None:
        print(f"最终压缩：{plan.final_compression.description}")
    if plan.coverage_warning:
        print(f"覆盖提醒：{plan.coverage_warning}")
    if args.dry_run:
        return 0
    result = run_pipeline(inspection, config)
    print(result)
    return 0
