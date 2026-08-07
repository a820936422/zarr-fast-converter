from __future__ import annotations

import argparse
import ast
import shutil
import sys
from pathlib import Path
from typing import Sequence

from .compression import format_compression_plan, make_compression_plan
from .engine import RechunkExecutionError, next_available_output, run_rechunk
from .inspection import RechunkInspectionError, format_inspection, inspect_store
from .planning import (
    DEFAULT_TARGET_MIB,
    default_workers,
    format_plan,
    plan_chunks,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-zarr-rechunk",
        description="Zarr v3 快速重分块与无损重压缩器。",
    )
    parser.add_argument("--input", type=Path, help="输入 Zarr v3 目录。")
    parser.add_argument("--output", type=Path, help="输出 Zarr 目录。")
    parser.add_argument(
        "--strategy",
        choices=("time", "space", "custom"),
        help="time=时间连续，space=空间连续，custom=自定义。",
    )
    parser.add_argument(
        "--chunks",
        help="自定义 chunks，例如 '[10, 300, 300]'；仅 custom 策略使用。",
    )
    parser.add_argument(
        "--target-chunk-mib",
        type=float,
        default=DEFAULT_TARGET_MIB,
        help=f"自动分块目标大小 MiB，默认 {DEFAULT_TARGET_MIB:g}。",
    )
    parser.add_argument(
        "--compression",
        choices=("none", "fast", "balanced", "maximum"),
        help="压缩方案；默认 none（交互模式会询问）。",
    )
    parser.add_argument(
        "--workers",
        type=int,
        help="并行 worker 上限；程序会依据内存和源/目标磁盘类型自动限制。",
    )
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有 Zarr v3 输出。")
    parser.add_argument("--no-validate", action="store_true", help="跳过输出抽样逐值校验。")
    parser.add_argument("--inspect-only", action="store_true", help="只检查输入，不生成计划。")
    parser.add_argument("--dry-run", action="store_true", help="只检查和生成计划，不写数据。")
    parser.add_argument("--quiet", action="store_true", help="关闭实时进度条。")
    return parser


def _required_path(prompt: str) -> Path:
    while True:
        value = input(prompt).strip()
        if value:
            return Path(value).expanduser()
        print("路径不能为空。")


def _parse_custom(raw: str | None) -> Sequence[int] | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError("自定义 chunks 必须是类似 [10, 300, 300] 的列表。") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError("自定义 chunks 必须是列表或元组。")
    return value


def _interactive_strategy() -> tuple[str, Sequence[int] | None]:
    while True:
        value = input(
            "\n请选择重分块策略（1=时间连续，2=空间连续，3=自定义）："
        ).strip()
        if value == "1":
            return "time", None
        if value == "2":
            return "space", None
        if value == "3":
            raw = input("请输入 [time, lat, lon] 三个 chunk 长度，例如 [10, 300, 300]：")
            return "custom", _parse_custom(raw)
        print("请输入 1、2 或 3。")


def _interactive_compression() -> str:
    answer = input("\n是否启用重压缩？[y/N]：").strip().lower()
    if answer not in {"y", "yes"}:
        return "none"
    while True:
        value = input("请选择压缩方案（1=快速，2=平衡，3=极致）：").strip()
        if value in {"1", "2", "3"}:
            return {"1": "fast", "2": "balanced", "3": "maximum"}[value]
        print("请输入 1、2 或 3。")


def _choose_output(
    requested: Path,
    *,
    interactive: bool,
    overwrite: bool,
) -> tuple[Path, bool]:
    requested = requested.expanduser()
    if not requested.exists():
        return requested, overwrite
    if overwrite:
        return requested, True
    if interactive:
        answer = input(
            f"输出目录 {requested} 已存在，是否覆盖（仅限已有 Zarr v3）？[y/N]："
        ).strip().lower()
        if answer in {"y", "yes"}:
            return requested, True
    alternative = next_available_output(requested)
    print(f"不覆盖现有目录，自动使用新目录：{alternative}")
    return alternative, False


def _print_capacity(path: Path, logical_bytes: int) -> None:
    parent = path.parent if path.parent.exists() else Path.cwd()
    free = shutil.disk_usage(parent).free
    print(
        f"目标文件系统可用空间：{free / 1024**3:.1f} GiB；"
        f"输入变量未压缩逻辑量：{logical_bytes / 1024**3:.1f} GiB"
    )


def interactive_args() -> argparse.Namespace:
    return argparse.Namespace(
        input=_required_path("请输入输入 Zarr v3 目录："),
        output=_required_path("请输入输出 Zarr 目录："),
        strategy=None,
        chunks=None,
        target_chunk_mib=DEFAULT_TARGET_MIB,
        compression=None,
        workers=None,
        overwrite=False,
        no_validate=False,
        inspect_only=False,
        dry_run=False,
        quiet=False,
        interactive=True,
    )


def run(args: argparse.Namespace) -> int:
    interactive = bool(getattr(args, "interactive", False))
    if args.input is None:
        args.input = _required_path("请输入输入 Zarr v3 目录：")
    if args.output is None and not args.inspect_only:
        args.output = _required_path("请输入输出 Zarr 目录：")

    info = inspect_store(args.input)
    print(format_inspection(info))
    if args.inspect_only:
        return 0

    strategy = args.strategy
    custom = _parse_custom(args.chunks)
    if strategy is None:
        if not interactive:
            raise ValueError("非交互模式必须指定 --strategy。")
        strategy, custom = _interactive_strategy()
    elif strategy == "custom" and custom is None and interactive:
        custom = _parse_custom(
            input("请输入 [time, lat, lon] 三个 chunk 长度，例如 [10, 300, 300]：")
        )
    plan = plan_chunks(
        info,
        strategy,
        target_mib=args.target_chunk_mib,
        workers=args.workers or default_workers(),
        custom_chunks=custom,
    )
    print(format_plan(plan, info))

    compression_name = args.compression
    if compression_name is None:
        compression_name = _interactive_compression() if interactive else "none"
    compression = make_compression_plan(compression_name)
    print(format_compression_plan(info, compression))
    _print_capacity(args.output, info.logical_bytes)

    if args.dry_run:
        print("Dry-run 完成，没有写入输出。")
        return 0

    output, overwrite = _choose_output(
        args.output,
        interactive=interactive,
        overwrite=args.overwrite,
    )
    workers = max(1, int(args.workers or default_workers()))
    metrics = run_rechunk(
        args.input,
        output,
        info,
        plan,
        compression,
        workers=workers,
        overwrite=overwrite,
        progress=not args.quiet,
        validate=not args.no_validate,
    )
    print(
        "\n重分块完成并通过校验。\n"
        f"输出：{metrics['output']}\n"
        f"耗时：{float(metrics['elapsed']):.1f} 秒\n"
        f"逻辑吞吐：{float(metrics['throughput_mib_s']):.1f} MiB/s\n"
        f"输出物理大小：{int(metrics['physical_bytes']) / 1024**3:.2f} GiB"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        argv = sys.argv[1:] if argv is None else argv
        args = interactive_args() if not argv else build_parser().parse_args(argv)
        args.interactive = not bool(argv)
        return run(args)
    except KeyboardInterrupt:
        print("\n用户中断；未完成的输出不能视为有效 Zarr。", file=sys.stderr)
        return 130
    except (RechunkInspectionError, RechunkExecutionError, ValueError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
