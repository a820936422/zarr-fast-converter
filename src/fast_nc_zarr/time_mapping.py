from __future__ import annotations

"""Inspection and composition of temporal fields from names and time axes.

The converter historically had two exclusive modes: a complete source
``time`` coordinate or a complete date in the filename.  Real products also
use a hybrid representation, for example ``year`` in the filename and DOY
inside the source time coordinate.  This module keeps field discovery and the
user-confirmed rule independent from the actual data writer.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re
from pathlib import Path
from typing import Any, Literal

import numpy as np


TimeComponent = Literal["full", "year", "month", "day", "doy"]
TimeSource = Literal["filename", "time"]

_NUMBER = re.compile(r"\d+")


@dataclass(frozen=True)
class FilenameField:
    index: int
    start: int
    length: int
    sample: str
    values: tuple[str, ...]
    changed: bool

    @property
    def label(self) -> str:
        state = "变化" if self.changed else "稳定/未能验证变化"
        return f"文件名字段 #{self.index} = {self.sample}（{state}）"


@dataclass(frozen=True)
class TimeDimensionInfo:
    exists: bool
    name: str | None
    raw_values: tuple[str, ...]
    decoded_values: tuple[str, ...]
    attrs: dict[str, Any]
    format_label: str

    @property
    def has_decoded_full_dates(self) -> bool:
        return bool(self.decoded_values)


@dataclass(frozen=True)
class TimeFieldRef:
    source: TimeSource
    component: TimeComponent
    index: int = 0


@dataclass(frozen=True)
class TimeFieldOption:
    ref: TimeFieldRef
    label: str
    sample: str


@dataclass(frozen=True)
class TimeRule:
    """The fields selected by the user to build one daily timestamp."""

    full: TimeFieldRef | None = None
    year: TimeFieldRef | None = None
    month: TimeFieldRef | None = None
    day: TimeFieldRef | None = None
    doy: TimeFieldRef | None = None

    @property
    def is_full(self) -> bool:
        return self.full is not None

    @property
    def is_hybrid(self) -> bool:
        refs = [self.full, self.year, self.month, self.day, self.doy]
        return any(ref is not None and ref.source == "filename" for ref in refs) and any(
            ref is not None and ref.source == "time" for ref in refs
        )

    def validate(self) -> None:
        if self.full is not None:
            if any(value is not None for value in (self.year, self.month, self.day, self.doy)):
                raise ValueError("选择完整时间字段后，不能再同时选择年/月/日/DOY字段。")
            if self.full.component != "full":
                raise ValueError("完整时间来源必须是 full 字段。")
            return
        if self.year is None:
            raise ValueError("时间规则至少需要一个年份来源。")
        if self.doy is None and (self.month is None or self.day is None):
            raise ValueError("时间规则需要 DOY，或同时需要月份和日期。")
        if self.doy is not None and (self.month is not None or self.day is not None):
            raise ValueError("DOY 与月份/日期不能同时选择。")


@dataclass(frozen=True)
class TimeInspectionResult:
    input_dir: Path
    files: tuple[Path, ...]
    engine: str
    dimensions: tuple[str, ...]
    coordinates: tuple[str, ...]
    filename_fields: tuple[FilenameField, ...]
    time_dimension: TimeDimensionInfo
    options: tuple[TimeFieldOption, ...]
    suggested_rule: TimeRule | None
    report: str


def _raise_if_cancelled(cancel_event) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("任务已取消。")


def inspect_time_metadata(
    input_dir: Path,
    *,
    recursive: bool = False,
    requested_engine: str = "auto",
    cancel_event=None,
) -> TimeInspectionResult:
    """Read only filenames and the first file's metadata/time coordinate."""

    from .filename_mode import discover_filename_files, probe_dataset_structure

    source = Path(input_dir).expanduser().resolve()
    files = discover_filename_files(source, recursive=recursive)
    _raise_if_cancelled(cancel_event)
    engine, dimensions, coordinates, has_time, _has_space = probe_dataset_structure(
        files[0], requested_engine
    )
    filename_fields = _discover_filename_fields(files, cancel_event=cancel_event)
    _raise_if_cancelled(cancel_event)
    time_info = _inspect_time_dimension(
        files[0], engine, dimensions, has_time, cancel_event=cancel_event
    )
    _raise_if_cancelled(cancel_event)
    options = _make_options(filename_fields, time_info)
    suggested = _suggest_rule(filename_fields, time_info, options)
    report = format_time_inspection(
        source,
        files,
        engine,
        dimensions,
        coordinates,
        filename_fields,
        time_info,
        suggested,
    )
    return TimeInspectionResult(
        input_dir=source,
        files=files,
        engine=engine,
        dimensions=dimensions,
        coordinates=coordinates,
        filename_fields=filename_fields,
        time_dimension=time_info,
        options=options,
        suggested_rule=suggested,
        report=report,
    )


def resolve_file_times(
    filename: str,
    raw_time_values: tuple[Any, ...],
    time_attrs: dict[str, Any],
    rule: TimeRule,
    filename_fields: tuple[FilenameField, ...],
) -> tuple[np.datetime64, ...]:
    """Build normalized daily dates for one source file under a TimeRule."""

    rule.validate()
    values = raw_time_values or (None,)
    result = []
    for raw in values:
        if rule.full is not None:
            parts = {"full": _extract_ref(rule.full, filename, raw, time_attrs, filename_fields)}
        else:
            parts = {
                component: _extract_ref(ref, filename, raw, time_attrs, filename_fields)
                for component, ref in (
                    ("year", rule.year),
                    ("month", rule.month),
                    ("day", rule.day),
                    ("doy", rule.doy),
                )
                if ref is not None
            }
        result.append(_date_from_parts(parts, raw))
    dates = tuple(result)
    keys = [str(value) for value in dates]
    if len(set(keys)) != len(keys):
        raise ValueError(f"文件 {filename} 的时间字段在同一日期重复。")
    return dates


def format_time_inspection(
    input_dir: Path,
    files: tuple[Path, ...],
    engine: str,
    dimensions: tuple[str, ...],
    coordinates: tuple[str, ...],
    filename_fields: tuple[FilenameField, ...],
    time_info: TimeDimensionInfo,
    suggested: TimeRule | None,
) -> str:
    lines = [
        "========== 文件时间维度信息检查 ==========",
        f"目录：{input_dir}",
        f"文件数量：{len(files)}（已按文件名排序）",
        f"首文件：{files[0].name}",
        f"读取引擎：{engine}",
        f"首文件维度：{', '.join(dimensions) or '无'}",
        f"首文件坐标：{', '.join(coordinates) or '无'}",
        "",
        "文件名候选字段：",
    ]
    if filename_fields:
        for field in filename_fields:
            values = ", ".join(field.values[:5])
            if len(field.values) > 5:
                values += " ……"
            lines.append(
                f"  #{field.index}: 位置={field.start}:{field.start + field.length}，"
                f"样例={field.sample}，值={values}，"
                f"{'发生变化' if field.changed else '未观察到变化'}"
            )
    else:
        lines.append("  未发现数字字段。")
    lines.extend(
        [
            "",
            "Time 维度：",
            f"  是否存在：{'是' if time_info.exists else '否'}",
            f"  维度名称：{time_info.name or '无'}",
            f"  原始格式：{time_info.format_label}",
            f"  原始样例：{', '.join(time_info.raw_values) or '无'}",
            f"  解码样例：{', '.join(time_info.decoded_values) or '无'}",
            f"  属性：{_render_attrs(time_info.attrs)}",
            "",
            f"建议规则：{_render_rule(suggested) if suggested else '无法唯一确定，请在下方手动选择。'}",
            "提示：建议规则仅供参考，必须由用户确认后才能进入后续结构检查。",
        ]
    )
    return "\n".join(lines)


def _discover_filename_fields(
    files: tuple[Path, ...], *, cancel_event=None
) -> tuple[FilenameField, ...]:
    sample_matches = list(_NUMBER.finditer(files[0].name))
    fields = []
    for index, match in enumerate(sample_matches):
        start, end = match.span()
        values = []
        valid = True
        _raise_if_cancelled(cancel_event)
        for path in files:
            if len(path.name) < end or not path.name[start:end].isdigit():
                valid = False
                break
            values.append(path.name[start:end])
        if not valid:
            continue
        fields.append(
            FilenameField(
                index=index,
                start=start,
                length=end - start,
                sample=match.group(),
                values=tuple(values),
                changed=len(set(values)) > 1,
            )
        )
    return tuple(fields)


def _inspect_time_dimension(
    path: Path,
    engine: str,
    dimensions: tuple[str, ...],
    has_time: bool,
    *,
    cancel_event=None,
) -> TimeDimensionInfo:
    import xarray as xr

    if not has_time:
        return TimeDimensionInfo(False, None, (), (), {}, "不存在 time 维度")
    _raise_if_cancelled(cancel_event)
    with xr.open_dataset(
        path,
        engine=engine,
        chunks=None,
        decode_times=False,
        mask_and_scale=False,
    ) as raw_ds:
        time_name = _find_time_name(raw_ds, dimensions)
        if time_name is None:
            return TimeDimensionInfo(True, None, (), (), {}, "检测到 time 语义，但未找到可读坐标")
        variable = raw_ds[time_name]
        raw_values = tuple(_render_value(value) for value in np.asarray(variable.values).reshape(-1)[:20])
        attrs = {str(key): _json_value(value) for key, value in variable.attrs.items()}
    decoded_values: tuple[str, ...] = ()
    _raise_if_cancelled(cancel_event)
    try:
        with xr.open_dataset(
            path,
            engine=engine,
            chunks=None,
            decode_times=True,
            mask_and_scale=False,
        ) as decoded_ds:
            if time_name in decoded_ds.variables:
                values = np.asarray(decoded_ds[time_name].values).reshape(-1)[:20]
                decoded = [_date_or_datetime_text(value) for value in values]
                if all(item is not None for item in decoded):
                    decoded_values = tuple(item for item in decoded if item is not None)
    except Exception:
        decoded_values = ()
    _raise_if_cancelled(cancel_event)
    units = str(attrs.get("units", ""))
    if decoded_values:
        format_label = "可解码为完整日期 YYYY-MM-DD"
    elif "doy" in units.lower() or units.lower().strip() in {"day", "days", "day of year"}:
        format_label = "数值 DOY（日序 1..366）"
    elif units:
        format_label = f"数值时间（units={units}）"
    else:
        format_label = "原始数值/字符串，尚未构成完整日期"
    return TimeDimensionInfo(True, time_name, raw_values, decoded_values, attrs, format_label)


def _find_time_name(dataset, dimensions: tuple[str, ...]) -> str | None:
    if "time" in dataset.variables:
        return "time"
    for name, variable in dataset.variables.items():
        attrs = variable.attrs
        if str(attrs.get("standard_name", "")).lower() == "time" or str(attrs.get("axis", "")).upper() == "T":
            return str(name)
    for name in dimensions:
        if name.lower() == "time" and name in dataset.variables:
            return name
    return None


def _make_options(
    fields: tuple[FilenameField, ...],
    time_info: TimeDimensionInfo,
) -> tuple[TimeFieldOption, ...]:
    options: list[TimeFieldOption] = []
    for field in fields:
        possible = _filename_components(field.sample)
        for component in possible:
            ref = TimeFieldRef("filename", component, field.index)
            options.append(
                TimeFieldOption(ref, f"文件名 #{field.index}：{field.sample} → {_component_label(component)}", field.sample)
            )
    if time_info.exists and time_info.name:
        if time_info.decoded_values:
            components = ("full", "year", "month", "day", "doy")
        elif _looks_like_doy(time_info):
            components = ("doy",)
        else:
            components = ()
        sample = time_info.decoded_values[0] if time_info.decoded_values else (time_info.raw_values[0] if time_info.raw_values else "")
        for component in components:
            ref = TimeFieldRef("time", component, 0)
            options.append(
                TimeFieldOption(ref, f"time 维度：{sample} → {_component_label(component)}", sample)
            )
    return tuple(options)


def _suggest_rule(
    fields: tuple[FilenameField, ...],
    time_info: TimeDimensionInfo,
    options: tuple[TimeFieldOption, ...],
) -> TimeRule | None:
    full_time = next((item.ref for item in options if item.ref.source == "time" and item.ref.component == "full"), None)
    if full_time is not None:
        return TimeRule(full=full_time)
    full_filename = [item.ref for item in options if item.ref.source == "filename" and item.ref.component == "full"]
    if len(full_filename) == 1 and not time_info.exists:
        return TimeRule(full=full_filename[0])
    year_filename = next((item.ref for item in options if item.ref.source == "filename" and item.ref.component == "year"), None)
    doy_time = next((item.ref for item in options if item.ref.source == "time" and item.ref.component == "doy"), None)
    if year_filename is not None and doy_time is not None:
        return TimeRule(year=year_filename, doy=doy_time)
    return None


def _filename_components(text: str) -> tuple[TimeComponent, ...]:
    result: list[TimeComponent] = []
    if len(text) == 7 and _valid_doy_text(text):
        result.extend(("full", "year", "doy"))
    elif len(text) == 8 and _valid_ymd_text(text):
        result.extend(("full", "year", "month", "day"))
    elif len(text) == 4 and text.isdigit():
        result.append("year")
    elif len(text) == 3 and text.isdigit() and 1 <= int(text) <= 366:
        result.append("doy")
    return tuple(result)


def _extract_ref(
    ref: TimeFieldRef,
    filename: str,
    raw: Any,
    attrs: dict[str, Any],
    filename_fields: tuple[FilenameField, ...],
) -> str:
    if ref.source == "filename":
        field = next((item for item in filename_fields if item.index == ref.index), None)
        if field is None:
            raise ValueError(f"找不到文件名字段 #{ref.index}。")
        text = filename[field.start : field.start + field.length]
        if ref.component == "full":
            return text
        if ref.component == "year":
            return text[:4]
        if ref.component == "doy":
            return text[4:] if len(text) == 7 else text
        if ref.component == "month":
            return text[4:6]
        if ref.component == "day":
            return text[6:8]
    if ref.component == "full":
        decoded = _decode_time_value(raw, attrs)
        if decoded is None:
            raise ValueError(f"time 值 {raw!r} 无法解码为完整日期。")
        return decoded.isoformat()
    decoded = _decode_time_value(raw, attrs)
    if decoded is not None:
        if ref.component == "year":
            return f"{decoded.year:04d}"
        if ref.component == "month":
            return f"{decoded.month:02d}"
        if ref.component == "day":
            return f"{decoded.day:02d}"
        if ref.component == "doy":
            return f"{decoded.timetuple().tm_yday:03d}"
    if ref.component == "doy":
        return str(int(raw))
    raise ValueError(f"time 值 {raw!r} 无法提取 {_component_label(ref.component)}。")


def _date_from_parts(parts: dict[str, str], raw: Any) -> np.datetime64:
    try:
        if "full" in parts:
            text = parts["full"]
            if len(text) == 7 and _valid_doy_text(text):
                parsed = date(int(text[:4]), 1, 1) + timedelta(days=int(text[4:]) - 1)
            elif len(text) == 8 and _valid_ymd_text(text):
                parsed = date(int(text[:4]), int(text[4:6]), int(text[6:8]))
            else:
                parsed = date.fromisoformat(text[:10])
            return np.datetime64(parsed.isoformat(), "ns")
        year = int(parts["year"])
        if "doy" in parts:
            value = int(parts["doy"])
            parsed = date(year, 1, 1) + timedelta(days=value - 1)
            if parsed.year != year or value < 1:
                raise ValueError
            return np.datetime64(parsed.isoformat(), "ns")
        parsed = date(year, int(parts["month"]), int(parts["day"]))
        return np.datetime64(parsed.isoformat(), "ns")
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"无法用字段 {parts} 构建日期（原始 time={raw!r}）。") from exc


def _decode_time_value(value: Any, attrs: dict[str, Any]) -> date | None:
    if value is None:
        return None
    if isinstance(value, (datetime, date)):
        return value if isinstance(value, date) else value.date()
    if isinstance(value, (str, np.datetime64)):
        try:
            stamp = np.datetime64(value, "D")
            if not np.isnat(stamp):
                return date.fromisoformat(str(stamp))
        except (TypeError, ValueError, OverflowError):
            pass
    units = str(attrs.get("units", ""))
    if " since " not in units:
        return None
    try:
        import netCDF4

        decoded = netCDF4.num2date(
            value,
            units,
            calendar=str(attrs.get("calendar", "standard")),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=False,
        )
        if np.ndim(decoded):
            decoded = np.asarray(decoded).reshape(-1)[0]
        return date(int(decoded.year), int(decoded.month), int(decoded.day))
    except (TypeError, ValueError, OverflowError, ImportError):
        return None


def _valid_doy_text(text: str) -> bool:
    try:
        year, doy = int(text[:4]), int(text[4:])
        parsed = date(year, 1, 1) + timedelta(days=doy - 1)
        return 1 <= doy <= 366 and parsed.year == year
    except (TypeError, ValueError, OverflowError):
        return False


def _valid_ymd_text(text: str) -> bool:
    try:
        date(int(text[:4]), int(text[4:6]), int(text[6:8]))
        return True
    except (TypeError, ValueError, OverflowError):
        return False


def _looks_like_doy(info: TimeDimensionInfo) -> bool:
    units = str(info.attrs.get("units", "")).lower().strip()
    return "doy" in units or units in {"day", "days", "day of year"}


def _component_label(component: TimeComponent) -> str:
    return {"full": "完整时间", "year": "年", "month": "月", "day": "日", "doy": "DOY"}[component]


def _render_rule(rule: TimeRule | None) -> str:
    if rule is None:
        return "未确定"
    if rule.full:
        return f"完整时间来自 {rule.full.source}"
    values = []
    for label, value in (("年", rule.year), ("月", rule.month), ("日", rule.day), ("DOY", rule.doy)):
        if value:
            values.append(f"{label}来自 {value.source}")
    return "，".join(values)


def _render_attrs(attrs: dict[str, Any]) -> str:
    if not attrs:
        return "无"
    return ", ".join(f"{key}={value!r}" for key, value in attrs.items())


def _render_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _date_or_datetime_text(value: Any) -> str | None:
    try:
        if isinstance(value, np.datetime64):
            return str(np.datetime_as_string(value, unit="ns"))
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        text = str(value)
        if len(text) >= 10 and text[4] == "-" and text[7] == "-":
            return text
    except (TypeError, ValueError, OverflowError):
        return None
    return None


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value
