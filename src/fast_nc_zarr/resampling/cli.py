from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .engine import ResampleExecutionError, format_plan, plan_resample, run_resample
from .grid import RESAMPLING_METHODS
from .inspection import inspect_resample_input
from .models import ResampleConfig


def _tile_size_arg(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("空间块边长必须是正整数或 auto。") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("空间块边长必须是正整数或 auto。")
    return parsed


def _space_workers_arg(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("空间进程数必须是正整数或 auto。") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("空间进程数必须是正整数或 auto。")
    return parsed


def _time_block_arg(value: str) -> int | str:
    normalized = value.strip().lower()
    if normalized == "auto":
        return "auto"
    try:
        parsed = int(normalized)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("时间块大小必须是正整数或 auto。") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("时间块大小必须是正整数或 auto。")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-zarr-resample",
        description="使用 xESMF 对 Zarr v3 规则经纬度网格进行空间重采样。",
    )
    parser.add_argument("--input", type=Path, help="输入 Zarr v3 目录。")
    parser.add_argument("--output", type=Path, help="输出 Zarr v3 目录。")
    parser.add_argument("--resolution", type=float, help="目标纬度/经度分辨率，单位为度。")
    parser.add_argument("--method", choices=RESAMPLING_METHODS, default="bilinear")
    parser.add_argument(
        "--compute-dtype",
        choices=("source", "float32"),
        default="source",
        help="浮点数据的计算 dtype；默认 source，可选 float32 以降低内存和输出大小。",
    )
    parser.add_argument(
        "--skipna",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否忽略 NaN/填充值；默认开启，可用 --no-skipna 关闭。",
    )
    parser.add_argument(
        "--extent",
        choices=("source", "global"),
        default="source",
        help="目标范围；默认覆盖输入网格范围，边界不足整格时向外取整。",
    )
    parser.add_argument(
        "--tile-size",
        type=_tile_size_arg,
        default="auto",
        help="流式目标空间块边长；默认 auto，根据 chunks、分辨率、dtype 和可用内存选择。",
    )
    parser.add_argument(
        "--time-block",
        type=_time_block_arg,
        default="auto",
        help="每次计算的时间切片数；默认 auto，优先贴合源时间 chunk 并受内存约束。",
    )
    parser.add_argument(
        "--compute-workers",
        type=int,
        default=2,
        help="单个空间块内部的 Dask 线程数；默认 2。",
    )
    parser.add_argument(
        "--space-workers",
        type=_space_workers_arg,
        default="auto",
        help="并行处理空间块的进程数；默认 auto，最多使用 6 个进程。",
    )
    parser.add_argument(
        "--temporary-dir",
        type=Path,
        help="可选的中间处理目录；大时间 chunk 的中转 Zarr 和权重文件写入此处，成功后自动删除。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 Zarr v3 输出。")
    parser.add_argument("--no-validate", action="store_true", help="跳过输出结构校验。")
    parser.add_argument("--inspect-only", action="store_true", help="只检查输入，不生成计划。")
    parser.add_argument("--dry-run", action="store_true", help="只检查和生成计划，不写数据。")
    parser.add_argument("--quiet", action="store_true", help="减少执行输出。")
    return parser


def _required_path(prompt: str) -> Path:
    value = input(prompt).strip()
    if not value:
        raise ValueError("路径不能为空。")
    return Path(value).expanduser()


def _interactive_args() -> argparse.Namespace:
    return argparse.Namespace(
        input=_required_path("请输入输入 Zarr v3 目录："),
        output=_required_path("请输入输出 Zarr 目录："),
        resolution=float(input("请输入目标空间分辨率（度）：").strip()),
        method=input(
            "请输入方法（bilinear/conservative/conservative_normed/patch/nearest_s2d/nearest_d2s）："
        ).strip()
        or "bilinear",
        compute_dtype=(
            input("浮点数据计算 dtype（source/float32，默认 source）：").strip().lower()
            or "source"
        ),
        skipna=input("是否启用 skipna？[Y/n]：").strip().lower() not in {"n", "no"},
        extent="source",
        tile_size="auto",
        time_block="auto",
        compute_workers=2,
        space_workers="auto",
        temporary_dir=(
            Path(value)
            if (value := input("可选中间处理目录（回车使用输出目录旁临时目录）：").strip())
            else None
        ),
        overwrite=False,
        no_validate=False,
        inspect_only=False,
        dry_run=False,
        quiet=False,
        interactive=True,
    )


def run(args: argparse.Namespace) -> int:
    if args.input is None:
        args.input = _required_path("请输入输入 Zarr v3 目录：")
    if args.output is None and not args.inspect_only:
        args.output = _required_path("请输入输出 Zarr 目录：")
    inspection = inspect_resample_input(args.input)
    print(inspection.report)
    if args.inspect_only:
        return 0
    if args.resolution is None:
        if not getattr(args, "interactive", False):
            raise ValueError("非交互模式必须指定 --resolution。")
        args.resolution = float(input("请输入目标空间分辨率（度）：").strip())
    config = ResampleConfig(
        input=args.input,
        output=args.output,
        resolution=args.resolution,
        method=args.method,
        skipna=args.skipna,
        compute_dtype=args.compute_dtype,
        extent=args.extent,
        tile_size=args.tile_size,
        time_block=args.time_block,
        compute_workers=args.compute_workers,
        space_workers=args.space_workers,
        temporary_dir=args.temporary_dir,
        overwrite=args.overwrite,
        validate=not args.no_validate,
    )
    plan = plan_resample(config, inspection)
    print(format_plan(plan))
    if args.dry_run:
        print("Dry-run 完成，没有写入输出。")
        return 0
    metrics = run_resample(config, plan, progress=not args.quiet)
    print(
        f"输出：{metrics['output']}\n"
        f"耗时：{float(metrics['elapsed']):.1f} 秒\n"
        f"输出物理大小：{int(metrics['physical_bytes']) / 1024**3:.2f} GiB\n"
        f"临时处理目录：{metrics.get('temporary_dir', '未返回')}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        argv = sys.argv[1:] if argv is None else argv
        args = _interactive_args() if not argv else build_parser().parse_args(argv)
        args.interactive = not bool(argv)
        return run(args)
    except KeyboardInterrupt:
        print("\n用户中断；未完成的输出不能视为有效 Zarr。", file=sys.stderr)
        return 130
    except (ResampleExecutionError, ValueError, OSError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
