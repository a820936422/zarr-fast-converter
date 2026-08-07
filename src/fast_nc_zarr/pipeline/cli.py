from __future__ import annotations

import argparse
from pathlib import Path

from ..application.services import SourceInspectionConfig, inspect_source, inspect_zarr
from .engine import preview_pipeline, run_pipeline
from .models import (
    PipelineChunkingOptions,
    PipelineCompressionOptions,
    PipelineConfig,
    PipelineConversionOptions,
    PipelineGeneralConfig,
    PipelineInput,
    PipelineOperations,
    PipelineResamplingOptions,
    ZarrPipelinePlan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pixi run pipeline",
        description="将原始地理数据转换为 Zarr v3，并按需组合重采样、重分块和重压缩。",
    )
    parser.add_argument("--input", type=Path, required=True, help="源数据目录。")
    parser.add_argument(
        "--input-kind",
        choices=("auto", "raw", "zarr"),
        default="auto",
        help="输入类型；auto 根据检查结果识别，raw 为原始 NC/HDF/TIFF。",
    )
    parser.add_argument("--output", type=Path, required=True, help="最终 Zarr 目录。")
    parser.add_argument("--temporary-dir", type=Path, help="临时处理根目录。")
    parser.add_argument("--time", nargs=2, metavar=("START", "END"), help="时间起止日期。")
    parser.add_argument("--lat", nargs=2, type=float, metavar=("MIN", "MAX"), default=(-90.0, 90.0))
    parser.add_argument("--lon", nargs=2, type=float, metavar=("MIN", "MAX"), default=(-180.0, 180.0))
    parser.add_argument("--resample", action="store_true", help="执行空间重采样。")
    parser.add_argument("--rechunk", action="store_true", help="应用指定的最终 chunks。")
    parser.add_argument("--recompress", action="store_true", help="应用指定的最终压缩方案。")
    parser.add_argument("--resolution", type=float, help="重采样目标空间分辨率。")
    parser.add_argument("--variables", nargs="+", help="待处理变量；默认全部。")
    parser.add_argument("--variable-name", action="append", default=[], metavar="SOURCE=OUTPUT")
    parser.add_argument("--method", default="bilinear")
    parser.add_argument("--skipna", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--na-thres", type=float, default=1.0)
    parser.add_argument("--compute-dtype", choices=("source", "float32"), default="source")
    parser.add_argument("--before-conditions", default="", help="采样前替换条件，逗号分隔。")
    parser.add_argument("--before-results", default="", help="采样前替换结果，逗号分隔。")
    parser.add_argument("--after-conditions", default="", help="采样后替换条件，逗号分隔。")
    parser.add_argument("--after-results", default="", help="采样后替换结果，逗号分隔。")
    parser.add_argument(
        "--statistics-policy",
        choices=("auto", "sample", "exact"),
        default="auto",
        help="替换表达式统计策略。",
    )
    parser.add_argument("--cleanup-intermediate", action="store_true")
    parser.add_argument("--no-tune", action="store_true")
    parser.add_argument("--tune-budget", type=float, default=60.0)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--strategy", choices=("time", "space", "custom"), default="time")
    parser.add_argument("--target-mib", type=float, default=128.0)
    parser.add_argument("--custom-chunks", nargs=3, type=int)
    parser.add_argument("--compression", choices=("fast", "balanced", "maximum"), default="balanced")
    parser.add_argument(
        "--compression-codec",
        choices=("blosc-zstd", "blosc-lz4", "blosc-lz4hc", "blosc-zlib", "zstd", "gzip"),
        help="显式选择 Zarr v3 压缩 codec；未指定时沿用 profile。",
    )
    parser.add_argument("--compression-level", type=int, help="显式压缩等级。")
    parser.add_argument(
        "--compression-shuffle",
        choices=("auto", "noshuffle", "shuffle", "bitshuffle"),
        default="auto",
    )
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
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.resample and args.resolution is None:
        parser.error("--resample 需要同时提供 --resolution。")
    if not args.resample and args.resolution is not None:
        parser.error("--resolution 仅在选择 --resample 时有效。")
    if not args.resample and any(
        (args.before_conditions, args.before_results, args.after_conditions, args.after_results)
    ):
        parser.error("采样前后替换参数仅在选择 --resample 时有效。")
    if (args.compression_codec is not None or args.compression_level is not None) and not args.recompress:
        parser.error("显式压缩 codec/level 仅在选择 --recompress 时有效。")
    time_start, time_end = args.time if args.time else (None, None)
    names = _parse_names(args.variable_name)
    input_path = args.input.expanduser().resolve()
    input_kind = args.input_kind
    if input_kind == "auto":
        input_kind = "zarr" if (input_path / "zarr.json").is_file() else "raw"
    if input_kind == "zarr":
        inspection = inspect_zarr(input_path)
    else:
        inspection = inspect_source(
            SourceInspectionConfig(
                input_dir=input_path,
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
        input=PipelineInput(kind=input_kind),
        general=PipelineGeneralConfig(
            output=args.output,
            temporary_dir=args.temporary_dir,
            time_start=time_start,
            time_end=time_end,
            lat_min=args.lat[0],
            lat_max=args.lat[1],
            lon_min=args.lon[0],
            lon_max=args.lon[1],
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
        operations=PipelineOperations(
            resample=args.resample,
            rechunk=args.rechunk,
            recompress=args.recompress,
        ),
        resampling=PipelineResamplingOptions(
            resolution=args.resolution or 0.1,
            method=args.method,
            skipna=args.skipna,
            na_thres=args.na_thres,
            compute_dtype=args.compute_dtype,
            before_conditions=args.before_conditions,
            before_results=args.before_results,
            after_conditions=args.after_conditions,
            after_results=args.after_results,
            statistics_policy=args.statistics_policy,
        ),
        chunking=PipelineChunkingOptions(
            strategy=args.strategy,
            target_mib=args.target_mib,
            custom_chunks=tuple(args.custom_chunks) if args.custom_chunks else None,
            workers=args.workers,
        ),
        compression=PipelineCompressionOptions(
            profile=args.compression,
            codec=args.compression_codec,
            level=args.compression_level,
            shuffle=args.compression_shuffle,
        ),
        validate=not args.no_validate,
    )
    plan = preview_pipeline(inspection, config)
    if isinstance(plan, ZarrPipelinePlan):
        dimensions = (
            plan.resample_plan.output_dimensions
            if plan.resample_plan is not None
            else plan.input_info.dimensions
        )
        print(
            "统一 Zarr 计划：目标 shape(time, lat, lon)="
            f"({dimensions['time']}, {dimensions['lat']}, {dimensions['lon']})；"
            f"需要重采样={'是' if plan.needs_resample else '否'}"
        )
    else:
        print(
            f"一条龙计划：目标 shape(time, lat, lon)=({plan.source_selection.shape[0]}, "
            f"{plan.target_grid.lat.size}, {plan.target_grid.lon.size})；"
            f"源读取窗口={plan.source_read_window.lat_shape}x{plan.source_read_window.lon_shape}；"
            f"需要重采样={'是' if plan.needs_resample else '否'}"
        )
        print(f"源窗口依据：{plan.source_read_window.halo_description}")
    print(f"最终 chunks(time, lat, lon)：{plan.final_chunks}")
    if plan.final_compression is not None:
        print(f"最终压缩：{plan.final_compression.description}")
    if args.resample:
        before_rules = list(
            zip(
                (item.strip() for item in args.before_conditions.split(",")),
                (item.strip() for item in args.before_results.split(",")),
                strict=True,
            )
        ) if args.before_conditions else []
        after_rules = list(
            zip(
                (item.strip() for item in args.after_conditions.split(",")),
                (item.strip() for item in args.after_results.split(",")),
                strict=True,
            )
        ) if args.after_conditions else []
        print(f"采样前替换：{before_rules or '无'}")
        print(f"采样后替换：{after_rules or '无'}")
        print(f"替换统计策略：{args.statistics_policy}")
    if getattr(plan, "coverage_warning", None):
        print(f"覆盖提醒：{plan.coverage_warning}")
    print("操作决策：")
    for decision in plan.operation_decisions:
        print(f"  {decision.operation}: {decision.disposition} - {decision.reason}")
    if args.dry_run:
        return 0
    result = run_pipeline(inspection, config)
    print(result)
    return 0
