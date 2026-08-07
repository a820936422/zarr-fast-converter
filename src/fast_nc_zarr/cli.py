from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

from .engine import convert
from .filename_mode import (
    FilenameTimeError,
    convert_filename,
    discover_filename_files,
    filename_logical_bytes,
    inspect_filename_inventory,
    probe_dataset_structure,
    scan_filename_times,
)
from .inspection import DimensionMappingRequired, inspect_dataset, inventory_summary
from .models import VariableTransform
from .planner import candidate_plans, initial_plan
from .selection import make_selection, parse_list, selected_logical_bytes


def _date_label(value) -> str:
    """Render a scanned date as the user-facing ``YYYY-MM-DD`` form."""
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="D"))
    return str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fast-nc-zarr",
        description="自动分析并高速转换具有 time/lat/lon 维度的 NetCDF 数据为 Zarr v3。",
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "complete", "filename"),
        default="auto",
        help="auto：根据首文件自动判断；complete：源文件自带 time；filename：从文件名构建 time。",
    )
    parser.add_argument("--input", type=Path, help="NetCDF 输入目录。")
    parser.add_argument("--output", type=Path, help="Zarr 输出目录（该目录本身就是 Zarr store）。")
    parser.add_argument("--recursive", action="store_true", help="递归搜索 NetCDF 文件。")
    parser.add_argument(
        "--engine", choices=("auto", "h5netcdf", "netcdf4", "rasterio"), default=None
    )
    parser.add_argument("--template", choices=("doy", "ymd"), help="文件名时间模板。")
    parser.add_argument("--year", help="样例文件名中的年份字符串。")
    parser.add_argument("--doy", help="样例文件名中的 DOY 字符串。")
    parser.add_argument("--month", help="样例文件名中的月份字符串。")
    parser.add_argument("--day", help="样例文件名中的日期字符串。")
    parser.add_argument("--step-days", type=int, help="文件名时间轴步长（天）；默认从文件推断。")
    parser.add_argument(
        "--continue-missing",
        action="store_true",
        help="文件名时间存在缺失时补齐理论时间轴并继续写入。",
    )
    parser.add_argument("--inspect-workers", type=int, help="检查阶段进程数；默认自动判断。")
    parser.add_argument("--time-dim", help="源文件时间维度名；默认为 time。")
    parser.add_argument("--lat-dim", help="源文件纬度维度名；默认为 lat。")
    parser.add_argument("--lon-dim", help="源文件经度维度名；默认为 lon。")
    parser.add_argument(
        "--time",
        help='时间范围列表，例如 "[2001-01-01, 2022-12-31]" 或 "[2001, 2010]"。',
    )
    parser.add_argument("--lat", help='纬度范围列表，例如 "[30, 90]"。')
    parser.add_argument("--lon", help='经度范围列表，例如 "[-100, 100]"。')
    parser.add_argument("--variables", nargs="+", help="变量名；默认全部。")
    parser.add_argument("--inspect-only", action="store_true", help="只检查，不转换。")
    parser.add_argument("--dry-run", action="store_true", help="只显示自动计划，不写数据。")
    parser.add_argument("--no-tune", action="store_true", help="关闭转换前的实测自动调参。")
    parser.add_argument("--tune-budget", type=float, default=60.0, help="自动调参时间预算秒数。")
    parser.add_argument("--max-workers", type=int, help="允许使用的最大物理核心数。")
    parser.add_argument("--reserve-memory", type=float, default=2.0, help="为系统保留的内存 GiB。")
    parser.add_argument("--overwrite", action="store_true", help="删除并重建已有非空输出目录。")
    parser.add_argument("--no-validate", action="store_true", help="跳过转换后的抽样逐值校验。")
    parser.add_argument("--quiet", action="store_true", help="减少进度输出。")
    return parser


def _required_path(prompt: str) -> Path:
    while True:
        text = input(prompt).strip()
        if text:
            return Path(text).expanduser()
        print("路径不能为空。")


def _interactive_variables(names: list[str]) -> list[str] | None:
    print("\n可转换变量：")
    for index, name in enumerate(names, 1):
        print(f"  {index}. {name}")
    raw = input("请选择变量编号，例如 [1,2]；直接回车表示全部：")
    selected = parse_list(raw, "变量")
    if selected is None:
        return None
    try:
        indices = [int(value) for value in selected]
    except (TypeError, ValueError) as exc:
        raise ValueError("变量列表中只能填写编号。") from exc
    invalid = [value for value in indices if value < 1 or value > len(names)]
    if invalid:
        raise ValueError(f"变量编号超出范围：{invalid}")
    return [names[value - 1] for value in dict.fromkeys(indices)]


def interactive_args() -> argparse.Namespace:
    print("========== 自适应 NetCDF -> Zarr v3 转换器 ==========")
    input_dir = _required_path("请输入原始数据目录：")
    output = _required_path("请输入输出 Zarr 数据目录：")
    return argparse.Namespace(
        mode="auto",
        input=input_dir,
        output=output,
        recursive=False,
        engine="auto",
        template=None,
        year=None,
        doy=None,
        month=None,
        day=None,
        step_days=None,
        continue_missing=False,
        inspect_workers=None,
        time_dim=None,
        lat_dim=None,
        lon_dim=None,
        time=None,
        lat=None,
        lon=None,
        variables=None,
        inspect_only=False,
        dry_run=False,
        no_tune=False,
        tune_budget=60.0,
        max_workers=None,
        reserve_memory=2.0,
        overwrite=False,
        no_validate=False,
        quiet=False,
        interactive=True,
        variable_transforms=None,
    )


def _provided_dimension_names(args: argparse.Namespace) -> tuple[str, str, str] | None:
    values = (
        getattr(args, "time_dim", None),
        getattr(args, "lat_dim", None),
        getattr(args, "lon_dim", None),
    )
    supplied = tuple(value.strip() if isinstance(value, str) else value for value in values)
    if any(value is not None and value != "" for value in supplied):
        if not all(isinstance(value, str) and value for value in supplied):
            raise ValueError("--time-dim、--lat-dim、--lon-dim 必须同时指定。")
        return supplied  # type: ignore[return-value]
    return None


def _auto_detect_mode(args: argparse.Namespace) -> str:
    files = discover_filename_files(args.input, recursive=args.recursive)
    requested = args.engine or "auto"
    engine, dims, _, has_time, has_space = probe_dataset_structure(files[0], requested)
    print(
        f"自动识别：{len(files)} 个 {files[0].suffix.lower()} 文件，"
        f"首文件维度：{', '.join(dims)}，引擎：{engine}"
    )
    if has_time and has_space:
        args.engine = engine
        return "complete"
    if has_space:
        args.engine = engine
        return "filename"
    raise FilenameTimeError(
        "首文件既没有可识别的时间维度，也没有可识别的纬度/经度或 x/y 空间维度。"
    )


def _interactive_dimension_names(available: tuple[str, ...]) -> tuple[str, str, str]:
    print(
        "\n检测到源文件没有使用标准维度名 time/lat/lon。"
        f"\n首个文件中的维度：{', '.join(available)}"
    )
    labels = (("时间", "time"), ("纬度", "lat"), ("经度", "lon"))
    values = []
    for label, canonical in labels:
        value = input(f"请输入{label}实际维度名称（输出统一命名为 {canonical}）：").strip()
        if not value:
            raise ValueError(f"{label}维度名称不能为空。")
        if value not in available:
            raise ValueError(f"指定的{label}维度 {value} 不在首个文件维度中。")
        values.append(value)
    if len(set(values)) != len(values):
        raise ValueError("时间、纬度和经度维度名称必须互不相同。")
    return tuple(values)  # type: ignore[return-value]


def _parse_fill_values(raw: str, variable: str) -> tuple[float, ...] | None:
    values = parse_list(raw, f"变量 {variable} 的填充值")
    if values is None or not values:
        return None if values is None else ()
    result = []
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"变量 {variable} 的填充值必须是数值。")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"变量 {variable} 的填充值必须是数值列表。") from exc
        if not math.isfinite(number) and not math.isnan(number):
            raise ValueError(f"变量 {variable} 的填充值必须是有限数值或 NaN。")
        result.append(number)
    return tuple(result)


def _parse_scale_value(raw: str, variable: str) -> float | None:
    text = raw.strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"变量 {variable} 的缩放因子必须是单个数值。") from exc
    if not math.isfinite(value):
        raise ValueError(f"变量 {variable} 的缩放因子必须是有限数值。")
    return value


def _interactive_transforms(inventory, variables: list[str] | tuple[str, ...]) -> dict[str, VariableTransform]:
    transforms: dict[str, VariableTransform] = {}
    print("\n========== 参数探测与处理 ==========")
    for name in variables:
        spec = inventory.variables[name]
        print(f"\n变量 {name} 检测到的参数：")
        for key in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
            print(f"  {key} = {spec.attrs.get(key, '未设置')!r}")
        fill = _parse_fill_values(
            input("请输入填充值列表（例如 [-1, 999]）；直接回车表示不替换："), name
        )
        scale = _parse_scale_value(
            input("请输入缩放因子（例如 0.01）；直接回车表示不缩放："), name
        )
        transforms[name] = VariableTransform(fill_values=fill, scale_factor=scale)
    return transforms


def _filename_fields(
    args: argparse.Namespace,
) -> tuple[str | None, tuple[str, ...] | None]:
    template = args.template
    if template is None:
        if any(getattr(args, name, None) is not None for name in ("year", "doy", "month", "day")):
            raise ValueError("手动指定文件名时间字段时必须同时指定 --template。")
        return None, None
    if template == "doy":
        values = (args.year, args.doy)
    else:
        values = (args.year, args.month, args.day)
    if any(value is None or not str(value).strip() for value in values):
        raise ValueError("文件名时间模板所需的年份/DOY/月/日字段不完整。")
    return template, tuple(str(value).strip() for value in values)


def _manual_filename_fields() -> tuple[str, tuple[str, ...]]:
    while True:
        template_choice = input("请选择文件名时间模板（1=年+DOY，2=年+月+日）：").strip()
        if template_choice in {"1", "2"}:
            break
        print("请输入 1 或 2。")
    template = "doy" if template_choice == "1" else "ymd"
    year = input("请输入样例文件名中的年份字符串：").strip()
    if template == "doy":
        return template, (year, input("请输入样例文件名中的 DOY 字符串：").strip())
    return template, (
        year,
        input("请输入样例文件名中的月份字符串：").strip(),
        input("请输入样例文件名中的日期字符串：").strip(),
    )


def run_filename(args: argparse.Namespace) -> int:
    interactive = getattr(args, "interactive", False)
    if args.input is None:
        args.input = _required_path("请输入原始数据目录：")
    if args.output is None and not args.inspect_only:
        args.output = _required_path("请输入输出 Zarr 数据目录：")
    template, fields = _filename_fields(args)
    try:
        scan = scan_filename_times(
            args.input,
            template=template,
            field_values=fields,
            step_days=args.step_days,
            recursive=args.recursive,
        )
    except FilenameTimeError as exc:
        if not interactive or template is not None:
            raise
        print(f"自动识别失败：{exc}")
        template, fields = _manual_filename_fields()
        scan = scan_filename_times(
            args.input,
            template=template,
            field_values=fields,
            step_days=args.step_days,
            recursive=args.recursive,
        )
    annual = ", ".join(f"{year}年={step}天" for year, step in scan.annual_steps)
    scale_label = f"每 {scan.step_days} 天" if scan.step_days else f"年度尺度：{annual}"
    print(
        f"\n样例文件名：{scan.sample_name}\n"
        f"实际时间：{_date_label(scan.actual_times[0])} .. {_date_label(scan.actual_times[-1])}，"
        f"{len(scan.actual_times)} 个；时间尺度：{scale_label}\n"
        f"理论时间：{_date_label(scan.expected_times[0])} .. {_date_label(scan.expected_times[-1])}，"
        f"{len(scan.expected_times)} 个"
    )
    if scan.template == "doy":
        annual_counts = {}
        for value in scan.actual_times:
            year = int(str(np.datetime_as_string(value, unit="D"))[:4])
            annual_counts[year] = annual_counts.get(year, 0) + 1
        print(
            "年度统计："
            + ", ".join(
                f"{year}年={count}个/{dict(scan.annual_steps).get(year, 0)}天"
                for year, count in sorted(annual_counts.items())
            )
        )
    if not scan.step_days and interactive:
        answer = input(
            "不同年份推断出的 DOY 时间尺度不一致，是否确认按年度规则继续？[y/N]："
        ).strip().lower()
        if answer not in {"y", "yes"}:
            print("已取消，未创建输出。")
            return 0
    if scan.missing_times:
        print(f"检测到缺失时间 {len(scan.missing_times)} 个：")
        preview = ", ".join(_date_label(item) for item in scan.missing_times[:20])
        print("  " + preview + (" ……" if len(scan.missing_times) > 20 else ""))
        if interactive:
            answer = input("是否继续并为缺失时间建立空值切片？[y/N]：").strip().lower()
            if answer not in {"y", "yes"}:
                print("已取消，未创建输出。")
                return 0
        elif not args.continue_missing:
            raise FilenameTimeError("存在缺失时间；请确认继续，或使用 --continue-missing。")

    inventory = inspect_filename_inventory(
        scan,
        args.engine or "auto",
        workers=args.inspect_workers,
        progress=not args.quiet,
    )
    print("\n========== 数据检查结果 ==========")
    print(inventory_summary(inventory))
    if args.inspect_only:
        return 0
    if interactive:
        time_bounds = parse_list(
            input(
                f"\n时间范围 {_date_label(inventory.times[0])} .. "
                f"{_date_label(inventory.times[-1])}，"
                "请输入 [开始,结束]（日期可不加引号）；直接回车表示全部："
            ),
            "时间范围",
        )
        lon_bounds = parse_list(
            input(f"经度范围 {inventory.lon_values.min():g} .. {inventory.lon_values.max():g}，请输入 [最小,最大]；直接回车表示全部："),
            "经度范围",
        )
        lat_bounds = parse_list(
            input(f"纬度范围 {inventory.lat_values.min():g} .. {inventory.lat_values.max():g}，请输入 [最小,最大]；直接回车表示全部："),
            "纬度范围",
        )
        variables = _interactive_variables(list(inventory.variables))
    else:
        time_bounds = parse_list(args.time or "", "时间范围")
        lat_bounds = parse_list(args.lat or "", "纬度范围")
        lon_bounds = parse_list(args.lon or "", "经度范围")
        variables = args.variables
    selection = make_selection(
        inventory,
        time_bounds=time_bounds,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        variables=variables,
    )
    transforms = (
        _interactive_transforms(inventory, selection.variables)
        if interactive
        else (getattr(args, "variable_transforms", None) or {})
    )
    plan = initial_plan(inventory, selection, args.output, reserve_gib=args.reserve_memory)
    print("\n========== 转换选择 ==========")
    print(f"shape (time, lat, lon)：{selection.shape}")
    print("变量：" + ", ".join(selection.variables))
    print(f"逻辑未压缩量（理论时间轴）：{filename_logical_bytes(inventory, selection, transforms) / 1024**3:.2f} GiB")
    print("文件名时间模式初始计划：" + plan.label())
    if args.dry_run:
        print("Dry-run 完成，没有创建输出。")
        return 0
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        if not interactive:
            raise FileExistsError("输出目录非空；请传入 --overwrite 后重试。")
        answer = input(f"输出目录 {args.output} 是已有 Zarr，删除并重建？[y/N]：").strip().lower()
        if answer not in {"y", "yes"}:
            print("已取消，未修改输出目录。")
            return 0
        args.overwrite = True
    chosen, metrics = convert_filename(
        inventory,
        selection,
        args.output,
        transforms=transforms,
        plan=plan,
        auto_tune=not args.no_tune,
        tune_budget=args.tune_budget,
        max_workers=args.max_workers,
        reserve_gib=args.reserve_memory,
        overwrite=args.overwrite,
        validate=not args.no_validate,
        progress=not args.quiet,
    )
    print(f"\n转换完成并通过校验。\n输出：{args.output.resolve()}\n最终计划：{chosen.label()}")
    print(f"正式生产写入：{metrics['elapsed']:.1f} 秒，{metrics['throughput_mib_s']:.1f} MiB/s")
    return 0


def run(args: argparse.Namespace) -> int:
    mode = getattr(args, "mode", "auto")
    if mode == "filename":
        return run_filename(args)
    interactive = getattr(args, "interactive", False)
    if args.input is None:
        args.input = _required_path("请输入原始数据目录：")
    if args.output is None and not args.inspect_only:
        args.output = _required_path("请输入输出 Zarr 数据目录：")
    if mode == "auto":
        detected = _auto_detect_mode(args)
        if detected == "filename":
            return run_filename(args)
    if getattr(args, "engine", None) in {None, "auto"}:
        args.engine = "h5netcdf"
    dimension_names = _provided_dimension_names(args)
    try:
        inventory = inspect_dataset(
            args.input,
            recursive=args.recursive,
            engine=args.engine or "h5netcdf",
            dimension_names=dimension_names,
            workers=args.inspect_workers,
            progress=not args.quiet,
        )
    except DimensionMappingRequired as exc:
        if not interactive:
            raise
        dimension_names = _interactive_dimension_names(exc.available)
        inventory = inspect_dataset(
            args.input,
            recursive=args.recursive,
            engine=args.engine or "h5netcdf",
            dimension_names=dimension_names,
            workers=args.inspect_workers,
            progress=not args.quiet,
        )
    print("\n========== 数据检查结果 ==========")
    print(inventory_summary(inventory))
    if args.inspect_only:
        return 0

    if interactive:
        time_bounds = parse_list(
            input(
                f"\n时间范围 {_date_label(inventory.times[0])} .. "
                f"{_date_label(inventory.times[-1])}，"
                "请输入 [开始,结束]（日期可不加引号）；直接回车表示全部："
            ),
            "时间范围",
        )
        lon_bounds = parse_list(
            input(
                f"经度范围 {inventory.lon_values.min():g} .. {inventory.lon_values.max():g}，"
                "请输入 [最小,最大]；直接回车表示全部："
            ),
            "经度范围",
        )
        lat_bounds = parse_list(
            input(
                f"纬度范围 {inventory.lat_values.min():g} .. {inventory.lat_values.max():g}，"
                "请输入 [最小,最大]；直接回车表示全部："
            ),
            "纬度范围",
        )
        variables = _interactive_variables(list(inventory.variables))
    else:
        time_bounds = parse_list(args.time or "", "时间范围")
        lat_bounds = parse_list(args.lat or "", "纬度范围")
        lon_bounds = parse_list(args.lon or "", "经度范围")
        variables = args.variables

    selection = make_selection(
        inventory,
        time_bounds=time_bounds,
        lat_bounds=lat_bounds,
        lon_bounds=lon_bounds,
        variables=variables,
    )
    transforms = (
        _interactive_transforms(inventory, selection.variables)
        if interactive
        else (getattr(args, "variable_transforms", None) or {})
    )
    logical = selected_logical_bytes(inventory, selection)
    plan = initial_plan(inventory, selection, args.output, reserve_gib=args.reserve_memory)
    print("\n========== 转换选择 ==========")
    print(f"shape (time, lat, lon)：{selection.shape}")
    print("变量：" + ", ".join(selection.variables))
    print(f"逻辑未压缩量：{logical / 1024**3:.2f} GiB")
    print("初始计划：" + plan.label())
    for reason in plan.rationale:
        print("  - " + reason)
    if not args.no_tune and plan.strategy != "dask":
        print(f"待实测候选：{len(candidate_plans(inventory, selection, args.output, max_workers=args.max_workers, reserve_gib=args.reserve_memory))} 组")
    if args.dry_run:
        print("Dry-run 完成，没有创建输出。")
        return 0

    if args.output.exists() and not args.output.is_dir():
        raise FileExistsError(f"输出路径存在但不是目录：{args.output}")
    if args.output.exists() and any(args.output.iterdir()) and not args.overwrite:
        if not interactive:
            raise FileExistsError("输出目录非空；请传入 --overwrite 后重试。")
        answer = input(f"输出目录 {args.output} 是已有 Zarr，删除并重建？[y/N]：").strip().lower()
        if answer not in {"y", "yes"}:
            print("已取消，未修改输出目录。")
            return 0
        args.overwrite = True

    chosen, metrics = convert(
        inventory,
        selection,
        args.output,
        auto_tune=not args.no_tune,
        tune_budget=args.tune_budget,
        max_workers=args.max_workers,
        reserve_gib=args.reserve_memory,
        overwrite=args.overwrite,
        validate=not args.no_validate,
        progress=not args.quiet,
        variable_transforms=transforms,
    )
    print("\n转换完成并通过校验。")
    print(f"输出：{args.output.resolve()}")
    print(f"最终计划：{chosen.label()}")
    print(
        f"正式生产写入：{metrics['elapsed']:.1f} 秒，"
        f"{metrics['throughput_mib_s']:.1f} MiB/s（按逻辑数据量；"
        "自动调参结果为包含 fsync 的耐久吞吐）"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    try:
        argv = sys.argv[1:] if argv is None else argv
        args = interactive_args() if not argv else build_parser().parse_args(argv)
        return run(args)
    except KeyboardInterrupt:
        print("\n用户中断。未完成的输出不能视为有效 Zarr。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
