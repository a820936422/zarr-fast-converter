"""Filename-time ingestion for files without a source time dimension.

The first implementation deliberately keeps the source representation simple:
each file contributes one two-dimensional latitude/longitude slice, while the
time coordinate is reconstructed from the filename.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import date, timedelta
import contextlib
import gc
import re
from pathlib import Path
from math import ceil, gcd
import shutil
import threading
import time
from typing import Any, Literal

import numpy as np

from .inspection import (
    _clean_attr,
    _hash_axis,
    _infer_frequency,
    choose_inspection_workers,
    time_key,
)
from .benchmark import COMPRESSION_SAFETY, tune
from .models import (
    ConversionPlan,
    FileRecord,
    Inventory,
    OutputLayout,
    Selection,
    VariableSpec,
    VariableTransform,
)
from .planner import (
    candidate_plans,
    fixed_layout_candidate_plans,
    output_layout_max_chunk_bytes,
    output_layout_plan_chunks,
    resolve_conversion_plan,
)
from .system import EffectiveResourceBudget, effective_resource_budget
from .publication import make_staging_path, preflight_writable, publish_staging, validate_publish_target
from .worker_pool import WorkerPool
from .writer import _monitor, compressor_from_spec, make_compressor, progress_line
from .selection import effective_output_dtype, selected_output_logical_bytes


FILENAME_SUFFIXES = {".nc", ".nc4", ".nc3", ".cdf", ".hdf", ".tif", ".tiff"}
FAST_STRUCTURE_SAMPLE_LIMIT = 32
Template = Literal["doy", "ymd"]


class FilenameTimeError(ValueError):
    """A filename time rule cannot be applied consistently to the inputs."""


class _LowLevelUnsupported(Exception):
    """The direct metadata reader cannot represent this source layout.

    This is deliberately separate from :class:`FilenameTimeError`: a source
    that is valid but not exposed cleanly by the netCDF4 low-level API should
    transparently fall back to the existing xarray normalisation path, while
    a real schema/grid mismatch must still be reported to the user.
    """


_LOW_LEVEL_ATTRS = ("_FillValue", "missing_value", "scale_factor", "add_offset", "units")


@dataclass(frozen=True)
class FilenameRuleCandidate:
    template: Template
    start: int
    length: int
    values: tuple[str, ...]
    dates: tuple[date, ...]

    @property
    def target(self) -> str:
        return self.values[0]


@dataclass(frozen=True)
class FilenameScan:
    input_dir: Path
    files: tuple[Path, ...]
    template: Template
    sample_name: str
    sample_start: int
    sample_length: int
    sample_prefix: str
    sample_suffix: str
    actual_times: tuple[np.datetime64, ...]
    expected_times: tuple[np.datetime64, ...]
    missing_times: tuple[np.datetime64, ...]
    step_days: int
    annual_steps: tuple[tuple[int, int], ...] = ()

    @property
    def actual_keys(self) -> tuple[str, ...]:
        return tuple(time_key(value) for value in self.actual_times)

    @property
    def expected_keys(self) -> tuple[str, ...]:
        return tuple(time_key(value) for value in self.expected_times)


def discover_filename_files(input_dir: Path, recursive: bool = False, cancel_event=None) -> tuple[Path, ...]:
    input_dir = input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    files = []
    for path in iterator:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("任务已取消。")
        if path.is_file() and path.suffix.lower() in FILENAME_SUFFIXES:
            files.append(path)
    files = tuple(sorted(files))
    if not files:
        raise FileNotFoundError(f"没有在 {input_dir} 中找到支持的源文件。")
    suffixes = {path.suffix.lower() for path in files}
    if len(suffixes) > 1:
        raise FilenameTimeError(
            "文件名时间模式一次只处理一种文件格式；发现："
            + ", ".join(sorted(suffixes))
        )
    return files


def engine_for_path(path: Path, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if path.suffix.lower() in {".tif", ".tiff"}:
        return "rasterio"
    if path.suffix.lower() == ".hdf":
        return "netcdf4"
    return "h5netcdf"


def _date_from_parts(template: Template, values: tuple[str, ...]) -> date:
    try:
        if template == "doy":
            year_text, doy_text = values
            year = int(year_text)
            doy = int(doy_text)
            first = date(year, 1, 1)
            result = first + timedelta(days=doy - 1)
            if result.year != year or doy < 1:
                raise ValueError
            return result
        year_text, month_text, day_text = values
        return date(int(year_text), int(month_text), int(day_text))
    except (TypeError, ValueError, OverflowError) as exc:
        label = "年+DOY" if template == "doy" else "年+月+日"
        raise FilenameTimeError(f"{label}日期字段无效：{values}") from exc


def _parts_from_text(template: Template, text: str, lengths: tuple[int, ...]) -> tuple[str, ...]:
    if len(text) != sum(lengths):
        raise FilenameTimeError("样例文件名中的时间字符串长度与输入字段不一致。")
    cursor = 0
    parts = []
    for length in lengths:
        parts.append(text[cursor : cursor + length])
        cursor += length
    return tuple(parts)


def _infer_step_from_doys(doys: list[int]) -> int:
    differences = [right - left for left, right in zip(sorted(doys), sorted(doys)[1:])]
    differences = [value for value in differences if value > 0]
    if not differences:
        return 1
    common = 0
    for value in differences:
        common = gcd(common, value)
    if common > 1:
        return int(common)
    return int(Counter(differences).most_common(1)[0][0])


def _infer_annual_steps(times: list[date]) -> dict[int, int]:
    by_year: dict[int, list[int]] = defaultdict(list)
    for value in times:
        by_year[value.year].append(value.timetuple().tm_yday)
    inferred = {
        year: _infer_step_from_doys(doys)
        for year, doys in sorted(by_year.items())
    }
    # A year containing only one observed file has no within-year interval
    # from which to infer a cadence.  Reuse the dominant cadence from years
    # with at least two observations when available; a lone-year dataset
    # remains a one-day theoretical schedule.
    candidates = [
        step
        for year, step in inferred.items()
        if len(by_year[year]) >= 2 and step > 0
    ]
    if candidates:
        fallback = Counter(candidates).most_common(1)[0][0]
        for year, doys in by_year.items():
            if len(doys) < 2:
                inferred[year] = fallback
    return inferred


def _expected_times(
    actual: list[date],
    template: Template,
    step_days: int,
    annual_steps: dict[int, int] | None = None,
) -> tuple[np.datetime64, ...]:
    if not actual:
        raise FilenameTimeError("没有可用于构建时间轴的文件。")
    if step_days < 1:
        raise FilenameTimeError("时间尺度必须是正整数天数。")
    first, last = min(actual), max(actual)
    values: set[date] = set()
    if template == "doy" and annual_steps:
        # Products such as 4-day/8-day composites usually restart their DOY
        # schedule at the beginning of each year.  Build that annual schedule
        # instead of applying one timedelta across year boundaries.
        base_start_doy = min(item.timetuple().tm_yday for item in actual)
        observed_by_year: dict[int, list[int]] = defaultdict(list)
        for item in actual:
            observed_by_year[item.year].append(item.timetuple().tm_yday)
        for year in range(first.year, last.year + 1):
            observed = observed_by_year.get(year, [])
            start_doy = min(observed) if observed else base_start_doy
            days_in_year = (date(year + 1, 1, 1) - date(year, 1, 1)).days
            if year in {first.year, last.year}:
                # The first and last observed years may intentionally be
                # partial years.  Do not invent dates outside the observed
                # boundary in those two years.
                end_doy = max(observed) if observed else days_in_year
            else:
                # An interior year belongs to the requested temporal span in
                # its entirety.  Limiting it to ``max(observed)`` hides a
                # missing file at the end of an otherwise complete year.  In
                # GLASS-PAR, 2012-12-31 is absent from the source files; it
                # must still be present in the theoretical daily axis.
                end_doy = days_in_year
            year_step = annual_steps.get(year, step_days)
            if year_step < 1:
                raise FilenameTimeError(f"{year} 年的时间尺度无效：{year_step} 天。")
            end_doy = min(end_doy, days_in_year)
            for doy in range(start_doy, end_doy + 1, year_step):
                candidate = date(year, 1, 1) + timedelta(days=doy - 1)
                if first <= candidate <= last:
                    values.add(candidate)
    else:
        cursor = first
        while cursor <= last:
            values.add(cursor)
            cursor += timedelta(days=step_days)
    return tuple(
        np.asarray(sorted(values), dtype="datetime64[ns]")
    )


def _repeated_date_like_positions(names: tuple[str, ...]) -> set[int]:
    """Return changing positions occupied by repeated date-like metadata.

    Some products include both an observation date and a processing date in
    each filename.  A processing date that changes only at a product-version
    boundary must not invalidate the unique observation-date candidate, but a
    second date field that is unique for every file remains ambiguous.
    """
    sample = names[0]
    positions: set[int] = set()
    for match in re.finditer(r"\d+", sample):
        start, end = match.span()
        length = end - start
        if length == 7:
            template, part_lengths = "doy", (4, 3)
        elif length == 8:
            template, part_lengths = "ymd", (4, 2, 2)
        else:
            continue
        values = tuple(name[start:end] for name in names)
        if any(not value.isdigit() for value in values):
            continue
        try:
            dates = tuple(
                _date_from_parts(template, _parts_from_text(template, value, part_lengths))
                for value in values
            )
        except FilenameTimeError:
            continue
        if len(set(dates)) < len(dates):
            positions.update(range(start, end))
    return positions


def _automatic_rule_candidates(files: tuple[Path, ...], cancel_event=None) -> tuple[FilenameRuleCandidate, ...]:
    if not files:
        raise FilenameTimeError("没有可用于推断时间字段的文件。")
    names = tuple(path.name for path in files)
    sample = names[0]
    if any(len(name) != len(sample) for name in names):
        raise FilenameTimeError("文件名长度不一致，无法自动推断时间字段。")
    changed_positions = {
        index
        for index, characters in enumerate(zip(*names))
        if len(set(characters)) > 1
    }
    if not changed_positions:
        raise FilenameTimeError("所有文件名都相同，无法推断时间字段。")
    repeated_date_like_positions = _repeated_date_like_positions(names)
    suffix = files[0].suffix.lower()
    if any(path.suffix.lower() != suffix for path in files):
        raise FilenameTimeError("文件扩展名不一致，无法自动推断时间字段。")
    candidates: list[FilenameRuleCandidate] = []
    for template, length in (("doy", 7), ("ymd", 8)):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("任务已取消。")
        for start in range(0, len(sample) - length + 1):
            end = start + length
            # A date field should be a complete numeric token, not a sliding
            # window inside a longer identifier.
            before = sample[start - 1] if start else ""
            after = sample[end] if end < len(sample) else ""
            if before.isdigit() or after.isdigit():
                continue
            values = tuple(name[start:end] for name in names)
            if any(not value.isdigit() for value in values):
                continue
            if len(set(values)) < 2:
                continue
            if any(name.count(value) != 1 for name, value in zip(names, values)):
                # A date value appearing twice is inherently ambiguous, even
                # when one of the occurrences happens to be constant across
                # the directory.
                continue
            outside_positions = changed_positions.difference(range(start, end))
            if outside_positions and not outside_positions.issubset(repeated_date_like_positions):
                # Ignore only a repeated, valid date-like metadata field (for example
                # a processing date); arbitrary changing tokens remain unsafe to
                # ignore, and a second unique date field remains ambiguous.
                continue
            try:
                dates = tuple(
                    _date_from_parts(template, _parts_from_text(template, value, (4, 3) if template == "doy" else (4, 2, 2)))
                    for value in values
                )
            except FilenameTimeError:
                continue
            if len(set(dates)) != len(dates):
                continue
            candidates.append(
                FilenameRuleCandidate(
                    template=template, start=start, length=length, values=values, dates=dates
                )
            )
    # De-duplicate candidates that arise from identical file layouts.
    unique = {(item.template, item.start, item.length): item for item in candidates}
    return tuple(unique.values())


def scan_filename_times(
    input_dir: Path,
    *,
    template: Template | None = None,
    field_values: tuple[str, ...] | None = None,
    step_days: int | None = None,
    recursive: bool = False,
    cancel_event=None,
) -> FilenameScan:
    files = discover_filename_files(input_dir, recursive, cancel_event=cancel_event)
    sample_name = files[0].name
    if template is None or field_values is None:
        candidates = _automatic_rule_candidates(files, cancel_event=cancel_event)
        if len(candidates) != 1:
            if not candidates:
                raise FilenameTimeError(
                    "无法从文件名自动识别唯一的年+DOY或年+月+日字段；"
                    "请手动指定时间模板和字段。"
                )
            rendered = ", ".join(
                f"{item.template}@{item.start}:{item.length}" for item in candidates
            )
            raise FilenameTimeError(
                "文件名中存在多个可能的时间字段：" + rendered + "；请手动确认。"
            )
        candidate = candidates[0]
        template = candidate.template
        start = candidate.start
        target = candidate.values[0]
        lengths = (4, 3) if template == "doy" else (4, 2, 2)
    else:
        lengths = (4, 3) if template == "doy" else (4, 2, 2)
        if len(field_values) != len(lengths):
            raise FilenameTimeError("时间模板字段数量不正确。")
        if any(not item or not item.isdigit() for item in field_values):
            raise FilenameTimeError("时间字段只能包含数字字符串。")
        if any(len(item) != length for item, length in zip(field_values, lengths)):
            raise FilenameTimeError(
                "时间字段长度不正确：年份通常为 4 位，DOY 为 3 位，月份和日期为 2 位。"
            )
        target = "".join(field_values)
        occurrences = sample_name.count(target)
        if occurrences != 1:
            raise FilenameTimeError(
                f"样例文件名 {sample_name} 中目标字符串 {target} 出现 {occurrences} 次，"
                "无法唯一确定时间字段。"
            )
        start = sample_name.find(target)
    prefix = sample_name[:start]
    suffix = sample_name[start + len(target) :]
    _date_from_parts(template, _parts_from_text(template, target, lengths))

    actual_pairs: list[tuple[Path, date]] = []
    seen: dict[date, Path] = {}
    for path in files:
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("任务已取消。")
        name = path.name
        if path.suffix.lower() != files[0].suffix.lower():
            raise FilenameTimeError(f"文件扩展名不一致：{name}")
        if len(name) != len(sample_name):
            raise FilenameTimeError(
                f"文件名结构与样例不一致：{name}；时间字段位置、长度和文件名总长度必须保持一致。"
            )
        extracted = name[start : start + len(target)]
        if not extracted.isdigit():
            raise FilenameTimeError(f"文件 {name} 的时间字段不是数字：{extracted!r}")
        if name.count(extracted) != 1:
            raise FilenameTimeError(
                f"文件 {name} 中时间字符串 {extracted} 不唯一，无法确定时间字段。"
            )
        parsed = _date_from_parts(template, _parts_from_text(template, extracted, lengths))
        if parsed in seen:
            raise FilenameTimeError(
                f"时间重复：{parsed.isoformat()} 同时对应 {seen[parsed]} 和 {path}；"
                "filename 模式要求每个时间点恰好只有一个源文件，"
                "请删除或重命名其中重复的文件后重试。"
            )
        seen[parsed] = path
        actual_pairs.append((path, parsed))

    actual_pairs.sort(key=lambda item: item[1])
    actual_dates = [item[1] for item in actual_pairs]
    actual_np = tuple(np.datetime64(item.isoformat(), "ns") for item in actual_dates)
    annual_steps = (
        _infer_annual_steps(actual_dates)
        if step_days is None
        else {year: int(step_days) for year in {item.year for item in actual_dates}}
    )
    if any(value < 1 for value in annual_steps.values()):
        raise FilenameTimeError("时间尺度必须是正整数天数。")
    unique_steps = set(annual_steps.values())
    chosen_step = unique_steps.pop() if len(unique_steps) == 1 else 0
    expected = _expected_times(
        actual_dates,
        template,
        chosen_step or min(annual_steps.values()),
        annual_steps if template == "doy" else None,
    )
    actual_keys = {time_key(item) for item in actual_np}
    missing = tuple(item for item in expected if time_key(item) not in actual_keys)
    return FilenameScan(
        input_dir=Path(input_dir).expanduser().resolve(),
        files=tuple(path for path, _ in actual_pairs),
        template=template,
        sample_name=sample_name,
        sample_start=start,
        sample_length=len(target),
        sample_prefix=prefix,
        sample_suffix=suffix,
        actual_times=actual_np,
        expected_times=expected,
        missing_times=missing,
        step_days=chosen_step,
        annual_steps=tuple(sorted(annual_steps.items())),
    )


def _open_dataset(path: Path, engine: str):
    import xarray as xr

    if engine == "rasterio":
        # Importing rioxarray makes the rasterio backend available in
        # installations where the entry point has not yet been loaded.
        try:
            import rioxarray  # noqa: F401
        except ImportError as exc:
            raise FilenameTimeError(
                "读取 TIFF 需要安装 rioxarray 和 rasterio。"
            ) from exc

    options = {
        "engine": engine,
        "chunks": None,
        "decode_times": False,
        "mask_and_scale": False,
    }
    try:
        return xr.open_dataset(path, **options)
    except (TypeError, ValueError):
        # Some optional xarray backends (notably rasterio versions) do not
        # accept all NetCDF-oriented keyword arguments.
        options.pop("mask_and_scale", None)
        options.pop("decode_times", None)
        try:
            return xr.open_dataset(path, **options)
        except (TypeError, ValueError):
            options.pop("chunks", None)
            return xr.open_dataset(path, **options)


def _grid_coordinate_values(ds, dimension: str, axis: Literal["lat", "lon"]):
    """Build coordinates for grid products that omit coordinate variables.

    HDF-EOS grids such as GLASS expose ``YDim:*``/``XDim:*`` dimensions and
    keep their geographic bounds in ``StructMetadata.0``.  Recovering cell
    centers here lets filename-time mode use the same canonical lat/lon path
    as NetCDF and rasterio inputs.
    """
    size = int(ds.sizes[dimension])
    metadata = str(ds.attrs.get("StructMetadata.0", ""))
    upper = re.search(r"UpperLeftPointMtrs=\(([^,]+),([^\)]+)\)", metadata)
    lower = re.search(r"LowerRightMtrs=\(([^,]+),([^\)]+)\)", metadata)
    if upper and lower:
        try:
            x0, y0 = (float(item.strip()) for item in upper.groups())
            x1, y1 = (float(item.strip()) for item in lower.groups())
            scale = 1_000_000.0 if max(abs(x0), abs(x1), abs(y0), abs(y1)) > 1000 else 1.0
            x0, x1, y0, y1 = x0 / scale, x1 / scale, y0 / scale, y1 / scale
            if axis == "lat":
                step = (y1 - y0) / size
                return y0 + (np.arange(size, dtype="float64") + 0.5) * step
            step = (x1 - x0) / size
            return x0 + (np.arange(size, dtype="float64") + 0.5) * step
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    values = _hdf_eos_swath_axis_values(ds, axis)
    if values is not None:
        return values
    # A dimension-only source has no trustworthy physical geolocation.  Keep
    # it usable for index-based conversion while making the fallback explicit
    # through the canonical coordinate values.
    return np.arange(size, dtype="float64")


def _hdf_eos_grid_values(
    metadata: Any, size: int, axis: Literal["lat", "lon"]
) -> np.ndarray | None:
    """Return HDF-EOS cell-centre coordinates from ``StructMetadata.0``.

    The HDF-EOS files used by GLASS do not contain coordinate variables.  The
    xarray path reconstructs them from the grid bounds; the low-level path
    uses the same calculation so that the resulting hashes remain identical.
    ``None`` means that the metadata does not contain usable geographic
    bounds, in which case the caller can use the same index-coordinate
    fallback as :func:`_grid_coordinate_values`.
    """
    text = str(metadata or "")
    upper = re.search(r"UpperLeftPointMtrs=\(([^,]+),([^\)]+)\)", text)
    lower = re.search(r"LowerRightMtrs=\(([^,]+),([^\)]+)\)", text)
    if not (upper and lower):
        return None
    try:
        x0, y0 = (float(item.strip()) for item in upper.groups())
        x1, y1 = (float(item.strip()) for item in lower.groups())
        scale = 1_000_000.0 if max(abs(x0), abs(x1), abs(y0), abs(y1)) > 1000 else 1.0
        x0, x1, y0, y1 = x0 / scale, x1 / scale, y0 / scale, y1 / scale
        if axis == "lat":
            step = (y1 - y0) / size
            return y0 + (np.arange(size, dtype="float64") + 0.5) * step
        step = (x1 - x0) / size
        return x0 + (np.arange(size, dtype="float64") + 0.5) * step
    except (TypeError, ValueError, ZeroDivisionError):
        return None


_SWATH_METADATA_GROUP_RE = re.compile(
    r"GROUP=SwathStructMetadata(.*?)END_GROUP=SwathStructMetadata", re.DOTALL
)
_SWATH_BLOCK_RE = re.compile(r"GROUP=(Swath_\d+)(.*?)END_GROUP=Swath_\d+", re.DOTALL)
_SWATH_DIMENSION_RE = re.compile(r'DimensionName="([^"]+)"\s+Size=\s*(\d+)')
_SWATH_FIELD_RE = re.compile(
    r'(\w+)Name="([^"]+)"\s+DataType=\S+\s+DimList=\(([^)]*)\)'
)


@dataclass(frozen=True)
class HdfEosSwath:
    """Parsed HDF-EOS ``SwathStructMetadata`` block for the first Swath.

    A Swath stores its geographic location as two-dimensional GeoFields
    (``Latitude``/``Longitude``) over along-track/cross-track dimensions,
    unlike a Grid whose bounds live in ``StructMetadata.0``.  The parsed
    structure lets both the low-level netCDF4 scan and the xarray path
    recover 1D latitude/longitude axes from regular (degenerate) swaths.
    """

    name: str
    dimensions: tuple[str, ...]
    data_fields: tuple[str, ...]
    geo_fields: tuple[str, ...]
    lat_field: str
    lon_field: str


def _hdf_eos_swath_structure(metadata: Any) -> HdfEosSwath | None:
    """Parse HDF-EOS Swath structure from ``CoreMetadata.0``.

    Returns ``None`` when the metadata does not contain a usable
    ``SwathStructMetadata`` block with both latitude and longitude
    GeoFields.
    """
    text = str(metadata or "")
    group = _SWATH_METADATA_GROUP_RE.search(text)
    if group is None:
        return None
    block = _SWATH_BLOCK_RE.search(group.group(1))
    if block is None:
        return None
    body = block.group(2)
    dimensions = tuple(
        match.group(1) for match in _SWATH_DIMENSION_RE.finditer(body)
    )
    fields = tuple(
        (
            match.group(1),
            match.group(2),
            tuple(part.strip().strip('"') for part in match.group(3).split(",")),
        )
        for match in _SWATH_FIELD_RE.finditer(body)
    )
    data_fields = tuple(name for kind, name, _dims in fields if kind == "DataField")
    geo_fields = tuple(name for kind, name, _dims in fields if kind == "GeoField")
    if not dimensions or not geo_fields:
        return None
    lat_field = next(
        (name for name in geo_fields if name.lower() in ("latitude", "lat")),
        None,
    )
    lon_field = next(
        (name for name in geo_fields if name.lower() in ("longitude", "lon")),
        None,
    )
    if lat_field is None or lon_field is None:
        return None
    return HdfEosSwath(
        name=block.group(1),
        dimensions=dimensions,
        data_fields=data_fields,
        geo_fields=geo_fields,
        lat_field=lat_field,
        lon_field=lon_field,
    )


def _hdf_eos_swath_axis(
    lat_2d: Any, lon_2d: Any, axis: Literal["lat", "lon"]
) -> tuple[np.ndarray | None, int | None]:
    """Reduce a Swath GeoField pair to a one-dimensional axis.

    A Swath stores its location as two-dimensional Latitude/Longitude
    fields.  Products whose location varies along exactly one dimension
    (a degenerate, regular Swath) can be represented by the canonical 1D
    latitude/longitude axes used by filename mode; genuinely irregular
    Swaths return ``(None, None)`` so callers can fall back or refuse.

    The returned integer is the array axis (0 = first dimension, 1 = second
    dimension) along which the field varies, i.e. the dimension the 1D
    values belong to.
    """
    try:
        lat = np.asarray(lat_2d, dtype="float64")
        lon = np.asarray(lon_2d, dtype="float64")
    except (TypeError, ValueError):
        return None, None
    if lat.ndim != 2 or lon.ndim != 2 or lat.shape != lon.shape:
        return None, None
    field = lat if axis == "lat" else lon
    rows, cols = field.shape
    along_row = float(np.mean(np.abs(np.diff(field, axis=0)))) if rows > 1 else 0.0
    along_col = float(np.mean(np.abs(np.diff(field, axis=1)))) if cols > 1 else 0.0
    dominant = max(along_row, along_col)
    tolerance = max(1e-9, dominant * 1e-6)
    if along_row <= tolerance and along_col > tolerance:
        values = np.asarray(field[0, :], dtype="float64")
        dim_index = 1
    elif along_col <= tolerance and along_row > tolerance:
        values = np.asarray(field[:, 0], dtype="float64")
        dim_index = 0
    else:
        return None, None
    if values.size == 0 or not np.all(np.isfinite(values)):
        return None, None
    magnitude = float(np.max(np.abs(values)))
    if magnitude > 500:
        # Coordinates stored as micro-degrees (or projected metres), like
        # Grid bounds.  A mix of plausible degrees and large fill-like
        # values means the field is not a clean coordinate axis.
        mixed = bool(np.any(np.abs(values) <= 500))
        if mixed or magnitude >= 1e9:
            return None, None
        values = values / 1_000_000.0
    return values, dim_index


def _hdf_eos_swath_axis_values(ds, axis: Literal["lat", "lon"]) -> np.ndarray | None:
    """Recover a 1D Swath axis from ``CoreMetadata.0`` GeoFields.

    Works on both netCDF4 ``Dataset`` objects (low-level scan) and xarray
    ``Dataset`` objects; returns ``None`` when the source is not a regular
    Swath.
    """
    if hasattr(ds, "ncattrs"):
        metadata = (
            ds.getncattr("CoreMetadata.0")
            if "CoreMetadata.0" in ds.ncattrs()
            else ""
        )
    else:
        metadata = str(ds.attrs.get("CoreMetadata.0", ""))
    swath = _hdf_eos_swath_structure(metadata)
    if swath is None or len(swath.dimensions) < 2:
        return None
    try:
        if hasattr(ds, "variables"):
            lat_2d = np.asarray(ds.variables[swath.lat_field][:])
            lon_2d = np.asarray(ds.variables[swath.lon_field][:])
        else:
            lat_2d = np.asarray(ds[swath.lat_field].values)
            lon_2d = np.asarray(ds[swath.lon_field].values)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    values, dim_index = _hdf_eos_swath_axis(lat_2d, lon_2d, axis)
    if values is None or dim_index is None:
        return None
    return values


def _swath_spatial_dimensions(ds) -> tuple[str, str] | None:
    """Map an HDF-EOS Swath's along/cross-track dimensions to lat/lon."""
    if hasattr(ds, "ncattrs"):
        metadata = (
            ds.getncattr("CoreMetadata.0")
            if "CoreMetadata.0" in ds.ncattrs()
            else ""
        )
        variables = ds.variables
    else:
        metadata = str(ds.attrs.get("CoreMetadata.0", ""))
        variables = ds
    swath = _hdf_eos_swath_structure(metadata)
    if swath is None or len(swath.dimensions) < 2:
        return None
    try:
        if hasattr(ds, "variables"):
            lat_2d = np.asarray(variables[swath.lat_field][:])
            lon_2d = np.asarray(variables[swath.lon_field][:])
        else:
            lat_2d = np.asarray(ds[swath.lat_field].values)
            lon_2d = np.asarray(ds[swath.lon_field].values)
    except (KeyError, IndexError, TypeError, ValueError):
        return None
    _lat_values, lat_index = _hdf_eos_swath_axis(lat_2d, lon_2d, "lat")
    _lon_values, lon_index = _hdf_eos_swath_axis(lat_2d, lon_2d, "lon")
    if (
        lat_index is None
        or lon_index is None
        or lat_index == lon_index
        or max(lat_index, lon_index) >= len(swath.dimensions)
    ):
        return None
    return swath.dimensions[lat_index], swath.dimensions[lon_index]


def _low_level_spatial_dimensions(ds) -> tuple[str, str]:
    """Infer the source latitude/longitude dimensions from netCDF4 objects."""
    dimensions = tuple(str(name) for name in ds.dimensions)
    dim_set = set(dimensions)
    if {"lat", "lon"}.issubset(dim_set):
        return "lat", "lon"
    if {"latitude", "longitude"}.issubset(dim_set):
        return "latitude", "longitude"
    if {"y", "x"}.issubset(dim_set):
        return "y", "x"

    ydims = [name for name in dimensions if name.lower().startswith("ydim")]
    xdims = [name for name in dimensions if name.lower().startswith("xdim")]
    if ydims and xdims:
        return ydims[0], xdims[0]

    # Some NetCDF products use neutral dimension names but mark the
    # one-dimensional coordinate variables with CF standard_name/axis.
    candidates: dict[str, str] = {}
    for name, variable in ds.variables.items():
        if tuple(str(item) for item in variable.dimensions) != (str(name),):
            continue
        attrs = set(variable.ncattrs())
        standard = str(variable.getncattr("standard_name")) if "standard_name" in attrs else ""
        axis = str(variable.getncattr("axis")) if "axis" in attrs else ""
        if standard.lower() == "latitude" or axis.upper() == "Y":
            candidates["lat"] = str(name)
        if standard.lower() == "longitude" or axis.upper() == "X":
            candidates["lon"] = str(name)
    if set(candidates) == {"lat", "lon"}:
        return candidates["lat"], candidates["lon"]
    swath_dims = _swath_spatial_dimensions(ds)
    if swath_dims is not None:
        return swath_dims
    raise _LowLevelUnsupported(
        "netCDF4 低层扫描无法识别经纬度维度；回退到 xarray。"
    )


def _low_level_axis_values(ds, dimension: str, axis: Literal["lat", "lon"]) -> np.ndarray:
    """Read a coordinate variable or reconstruct an HDF-EOS/index axis."""
    coordinate = ds.variables.get(dimension)
    if coordinate is not None and tuple(str(item) for item in coordinate.dimensions) == (
        dimension,
    ):
        try:
            return np.asarray(coordinate[:])
        except Exception as exc:  # pragma: no cover - backend-specific
            raise _LowLevelUnsupported("无法读取低层坐标变量；回退到 xarray。") from exc

    size = len(ds.dimensions[dimension])
    metadata = ""
    if "StructMetadata.0" in ds.ncattrs():
        metadata = ds.getncattr("StructMetadata.0")
    values = _hdf_eos_grid_values(metadata, size, axis)
    if values is None:
        # HDF-EOS Swath products keep their location in 2D GeoFields
        # declared through CoreMetadata.0 instead of coordinate variables.
        values = _hdf_eos_swath_axis_values(ds, axis)
        if values is not None:
            swath_dims = _swath_spatial_dimensions(ds)
            if swath_dims is None:
                values = None
            else:
                expected = swath_dims[0] if axis == "lat" else swath_dims[1]
                if expected != dimension:
                    values = None
    return values if values is not None else np.arange(size, dtype="float64")


def _low_level_variable_attrs(variable) -> dict[str, Any]:
    attrs = {}
    for key in _LOW_LEVEL_ATTRS:
        if key in variable.ncattrs():
            attrs[key] = _clean_attr(variable.getncattr(key))
    return attrs


def _low_level_variable_signature(
    variable,
    source_lat: str,
    source_lon: str,
    excluded_names: frozenset[str] = frozenset(),
) -> tuple[Any, ...] | None:
    """Build the same signature as ``_filename_variable_signature``.

    The filename mode only keeps numeric two-dimensional spatial variables.
    Variables containing a singleton band/time dimension are intentionally
    left to xarray, whose normalisation already handles those cases.
    HDF-EOS Swath GeoFields (``excluded_names``) are location metadata, not
    data variables, and must not enter the signature or the output.
    """
    if str(variable.name) in excluded_names:
        return None
    dimensions = tuple(str(item) for item in variable.dimensions)
    if len(dimensions) != 2 or set(dimensions) != {source_lat, source_lon}:
        return None
    try:
        dtype = np.dtype(variable.dtype)
    except TypeError as exc:  # pragma: no cover - unusual netCDF type
        raise _LowLevelUnsupported("发现无法映射的 NetCDF 变量类型；回退到 xarray。") from exc
    if dtype.kind not in "biufc":
        return None
    shape = (
        int(variable.shape[dimensions.index(source_lat)]),
        int(variable.shape[dimensions.index(source_lon)]),
    )
    attrs = _low_level_variable_attrs(variable)
    selected = {
        key: attrs[key]
        for key in _LOW_LEVEL_ATTRS
        if key in attrs
    }
    return (
        str(variable.name),
        dtype.name,
        ("lat", "lon"),
        shape,
        tuple(sorted((str(key), repr(value)) for key, value in selected.items())),
    )


def _inspect_filename_file_low_level(task) -> FileRecord:
    """Inspect a NetCDF4/HDF4 file without constructing an xarray Dataset."""
    (
        path,
        engine,
        expected_signature,
        expected_lat_hash,
        expected_lon_hash,
        expected_lat_size,
        expected_lon_size,
        reference_specs,
        time_value,
    ) = task
    if engine != "netcdf4" or path.suffix.lower() in {".tif", ".tiff"}:
        raise _LowLevelUnsupported("当前引擎不适用 NetCDF4 低层扫描。")
    try:
        from netCDF4 import Dataset
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise _LowLevelUnsupported("未安装 netCDF4；回退到 xarray。") from exc
    try:
        dataset = Dataset(path, mode="r")
    except Exception as exc:
        raise _LowLevelUnsupported("netCDF4 无法打开该文件；回退到 xarray。") from exc

    try:
        # Match the xarray inspection path (mask_and_scale=False).  This is
        # especially important when a coordinate variable itself carries a
        # scale_factor or add_offset: metadata validation must compare the
        # stored source coordinates, not silently transformed values.
        dataset.set_auto_maskandscale(False)
        swath = _hdf_eos_swath_structure(
            dataset.getncattr("CoreMetadata.0")
            if "CoreMetadata.0" in dataset.ncattrs()
            else None
        )
        excluded_names = (
            frozenset({swath.lat_field, swath.lon_field})
            if swath is not None
            else frozenset()
        )
        source_lat, source_lon = _low_level_spatial_dimensions(dataset)
        lat_values = _low_level_axis_values(dataset, source_lat, "lat")
        lon_values = _low_level_axis_values(dataset, source_lon, "lon")
        actual_signature = tuple(
            sorted(
                (
                    signature
                    for variable in dataset.variables.values()
                    for signature in (
                        _low_level_variable_signature(
                            variable,
                            source_lat,
                            source_lon,
                            excluded_names,
                        ),
                    )
                    if signature is not None
                ),
                key=lambda item: item[0],
            )
        )
    finally:
        dataset.close()

    if actual_signature != expected_signature:
        raise FilenameTimeError(f"{path.name} 的变量结构或属性与首文件不一致。")
    if (
        lat_values.size != expected_lat_size
        or lon_values.size != expected_lon_size
        or _hash_axis(lat_values) != expected_lat_hash
        or _hash_axis(lon_values) != expected_lon_hash
    ):
        raise FilenameTimeError(f"{path.name} 的经纬度网格与首文件不一致。")
    return FileRecord(
        path=path,
        size_bytes=(stat := path.stat()).st_size,
        times=(np.datetime64(time_value, "ns"),),
        time_keys=(time_key(np.datetime64(time_value, "ns")),),
        lat_hash=expected_lat_hash,
        lon_hash=expected_lon_hash,
        lat_size=expected_lat_size,
        lon_size=expected_lon_size,
        variables=reference_specs,
        mtime_ns=stat.st_mtime_ns,
    )

def _rasterio_metadata(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[VariableSpec, ...], tuple[Any, ...]]:
    """Read a single-band GeoTIFF schema without constructing xarray."""
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise _LowLevelUnsupported("未安装 rasterio；回退到 xarray。") from exc
    try:
        with rasterio.open(path) as dataset:
            if dataset.count != 1:
                raise _LowLevelUnsupported("多 band raster 需要 xarray 归一化。")
            transform = dataset.transform
            crs_signature = dataset.crs.to_string() if dataset.crs is not None else None
            transform_signature = tuple(round(float(value), 12) for value in transform)
            if not np.isclose(transform.b, 0.0) or not np.isclose(transform.d, 0.0):
                raise _LowLevelUnsupported("旋转 raster 网格需要 xarray 归一化。")
            rows = np.arange(dataset.height, dtype=np.int64)
            cols = np.arange(dataset.width, dtype=np.int64)
            _xs, lat_values = rasterio.transform.xy(
                transform, rows, np.zeros(dataset.height, dtype=np.int64), offset="center"
            )
            lon_values, _ys = rasterio.transform.xy(
                transform, np.zeros(dataset.width, dtype=np.int64), cols, offset="center"
            )
            lat = np.asarray(lat_values, dtype="float64")
            lon = np.asarray(lon_values, dtype="float64")
            attrs: dict[str, Any] = {}
            nodata = dataset.nodata
            if nodata is not None:
                attrs["_FillValue"] = _clean_attr(nodata)
            scale = float(dataset.scales[0]) if dataset.scales else 1.0
            offset = float(dataset.offsets[0]) if dataset.offsets else 0.0
            if scale != 1.0:
                attrs["scale_factor"] = scale
            if offset != 0.0:
                attrs["add_offset"] = offset
            dtype = np.dtype(dataset.dtypes[0])
            shape = (int(dataset.height), int(dataset.width))
    except _LowLevelUnsupported:
        raise
    except Exception as exc:  # pragma: no cover - backend-specific
        raise _LowLevelUnsupported("rasterio 无法读取元数据；回退到 xarray。") from exc
    signature = (
        "band_data",
        dtype.name,
        ("lat", "lon"),
        shape,
        tuple(sorted((str(key), repr(value)) for key, value in attrs.items())),
        crs_signature,
        transform_signature,
    )
    spec = VariableSpec(
        name="band_data",
        dims=("time", "lat", "lon"),
        dtype=dtype.name,
        shape_without_time=shape,
        native_chunks=None,
        attrs=attrs,
    )
    return lat, lon, (spec,), (signature,)


def _inspect_filename_file_rasterio(task) -> FileRecord:
    """Validate a GeoTIFF file using metadata only."""
    (
        path,
        engine,
        expected_signature,
        expected_lat_hash,
        expected_lon_hash,
        expected_lat_size,
        expected_lon_size,
        reference_specs,
        time_value,
    ) = task
    if engine != "rasterio":
        raise _LowLevelUnsupported("当前引擎不适用 rasterio 低层扫描。")
    lat, lon, _specs, actual_signature = _rasterio_metadata(path)
    if actual_signature != expected_signature:
        raise FilenameTimeError(f"{path.name} 的变量结构或属性与首文件不一致。")
    if (
        lat.size != expected_lat_size
        or lon.size != expected_lon_size
        or _hash_axis(lat) != expected_lat_hash
        or _hash_axis(lon) != expected_lon_hash
    ):
        raise FilenameTimeError(f"{path.name} 的经纬度网格与首文件不一致。")
    stat = path.stat()
    return FileRecord(
        path=path,
        size_bytes=stat.st_size,
        times=(np.datetime64(time_value, "ns"),),
        time_keys=(time_key(np.datetime64(time_value, "ns")),),
        lat_hash=expected_lat_hash,
        lon_hash=expected_lon_hash,
        lat_size=expected_lat_size,
        lon_size=expected_lon_size,
        variables=reference_specs,
        mtime_ns=stat.st_mtime_ns,
    )


def _rename_spatial_dims(ds):
    dims = set(ds.dims)
    if {"lat", "lon"}.issubset(dims):
        source_lat, source_lon = "lat", "lon"
    elif {"latitude", "longitude"}.issubset(dims):
        source_lat, source_lon = "latitude", "longitude"
    elif any(name in dims for name in ("lat", "latitude")) and any(
        name in dims for name in ("lon", "longitude")
    ):
        source_lat = next(name for name in ("lat", "latitude") if name in dims)
        source_lon = next(name for name in ("lon", "longitude") if name in dims)
    elif {"y", "x"}.issubset(dims):
        source_lat, source_lon = "y", "x"
    elif (
        any(str(name).lower().startswith("ydim") for name in dims)
        and any(str(name).lower().startswith("xdim") for name in dims)
    ):
        source_lat = next(name for name in dims if str(name).lower().startswith("ydim"))
        source_lon = next(name for name in dims if str(name).lower().startswith("xdim"))
    else:
        candidates = {}
        for name, coordinate in ds.coords.items():
            standard = str(coordinate.attrs.get("standard_name", "")).lower()
            axis = str(coordinate.attrs.get("axis", "")).upper()
            if standard == "latitude" or axis == "Y":
                candidates["lat"] = name
            if standard == "longitude" or axis == "X":
                candidates["lon"] = name
        if set(candidates) != {"lat", "lon"}:
            swath_dims = _swath_spatial_dimensions(ds)
            if swath_dims is not None:
                source_lat, source_lon = swath_dims
            else:
                raise FilenameTimeError(
                    f"无法从源文件识别纬度/经度维度；发现维度：{', '.join(map(str, ds.dims))}"
                )
        else:
            source_lat, source_lon = candidates["lat"], candidates["lon"]

    mapping = {}
    if source_lat != "lat":
        mapping[source_lat] = "lat"
    if source_lon != "lon":
        mapping[source_lon] = "lon"
    if mapping:
        ds = ds.rename(mapping)
    if "lat" not in ds.coords:
        ds = ds.assign_coords(lat=_grid_coordinate_values(ds, "lat", "lat"))
    if "lon" not in ds.coords:
        ds = ds.assign_coords(lon=_grid_coordinate_values(ds, "lon", "lon"))
    swath = _hdf_eos_swath_structure(ds.attrs.get("CoreMetadata.0"))
    swath_fields = {swath.lat_field, swath.lon_field} if swath is not None else set()
    for name in list(ds.data_vars):
        variable = ds[name]
        if name in swath_fields:
            # HDF-EOS Swath GeoFields are location metadata consumed by the
            # coordinate reconstruction above; they are not data variables.
            ds = ds.drop_vars(name)
            continue
        if "band" in variable.dims:
            if variable.sizes["band"] != 1:
                raise FilenameTimeError(
                    f"变量 {name} 含有多个 band；当前文件名时间模式暂只支持单 band。"
                )
            variable = variable.isel(band=0, drop=True)
        if "time" in variable.dims:
            if variable.sizes["time"] != 1:
                raise FilenameTimeError(
                    f"变量 {name} 已含多个 time；文件名时间模式无法确定合并规则。"
                )
            variable = variable.isel(time=0, drop=True)
        if set(variable.dims) == {"lat", "lon"}:
            ds[name] = variable.transpose("lat", "lon")
        else:
            ds = ds.drop_vars(name)
    return ds


def normalize_filename_dataset(path: Path, requested_engine: str = "auto"):
    engine = engine_for_path(path, requested_engine)
    try:
        ds = _open_dataset(path, engine)
    except Exception:
        if requested_engine != "auto" or path.suffix.lower() in {".hdf", ".tif", ".tiff"}:
            raise
        engine = "netcdf4"
        ds = _open_dataset(path, engine)
    try:
        return _rename_spatial_dims(ds), engine
    except Exception:
        ds.close()
        del ds
        if engine == "rasterio":
            gc.collect()
        raise


@contextlib.contextmanager
def _normalized_filename_dataset(path: Path, requested_engine: str = "auto"):
    """Open a filename-mode dataset and fully release rasterio resources.

    The rasterio backend can retain a URI/file manager through the lazy
    xarray array after ``Dataset.close()``.  In a spawned worker this object
    may survive until interpreter shutdown, where rasterio/rioxarray can
    invoke the exception hook with no useful traceback.  Deleting the local
    Dataset reference and collecting cycles before leaving the worker avoids
    that shutdown-time failure while retaining lazy windowed reads.
    """
    dataset, engine = normalize_filename_dataset(path, requested_engine)
    try:
        yield dataset, engine
    finally:
        dataset.close()
        del dataset
        if engine == "rasterio":
            gc.collect()


def probe_dataset_structure(
    path: Path, requested_engine: str = "auto"
) -> tuple[str, tuple[str, ...], tuple[str, ...], bool, bool]:
    """Return backend, dimensions, coordinates, and time/space indicators."""
    engine = engine_for_path(path, requested_engine)
    try:
        ds = _open_dataset(path, engine)
    except Exception:
        if requested_engine != "auto" or path.suffix.lower() in {".hdf", ".tif", ".tiff"}:
            raise
        engine = "netcdf4"
        ds = _open_dataset(path, engine)
    try:
        dims = tuple(str(name) for name in ds.dims)
        coords = tuple(str(name) for name in ds.coords)
        time_like = "time" in dims
        lowered_dims = {name.lower() for name in dims}
        lat_like = any(name in dims for name in ("lat", "latitude", "y")) or any(
            name.startswith("ydim") for name in lowered_dims
        )
        lon_like = any(name in dims for name in ("lon", "longitude", "x")) or any(
            name.startswith("xdim") for name in lowered_dims
        )
        for coordinate in ds.coords.values():
            standard = str(coordinate.attrs.get("standard_name", "")).lower()
            axis = str(coordinate.attrs.get("axis", "")).upper()
            time_like = time_like or standard == "time" or axis == "T"
            lat_like = lat_like or standard == "latitude" or axis == "Y"
            lon_like = lon_like or standard == "longitude" or axis == "X"
        space_like = lat_like and lon_like
        return engine, dims, coords, time_like, space_like
    finally:
        ds.close()
        del ds
        if engine == "rasterio":
            gc.collect()


def _variable_attrs(variable) -> dict[str, Any]:
    attrs = {key: _clean_attr(value) for key, value in variable.attrs.items()}
    for key in ("_FillValue", "missing_value", "scale_factor", "add_offset"):
        if key not in attrs and key in variable.encoding:
            attrs[key] = _clean_attr(variable.encoding[key])
    return attrs
def _filename_dataset_spatial_signature(dataset, engine: str) -> tuple[Any, ...] | None:
    if engine != "rasterio":
        return None
    crs_text = None
    transform_values: tuple[float, ...] | None = None
    try:
        rio = dataset.rio
        crs = rio.crs
        crs_text = crs.to_wkt() if crs is not None else None
        transform_values = tuple(round(float(value), 12) for value in rio.transform())
    except (AttributeError, TypeError, ValueError):
        crs_value = dataset.attrs.get("crs_wkt", dataset.attrs.get("crs"))
        crs_text = str(crs_value) if crs_value not in (None, "") else None
        transform_value = dataset.attrs.get("transform")
        if isinstance(transform_value, (list, tuple, np.ndarray)):
            try:
                transform_values = tuple(round(float(value), 12) for value in transform_value)
            except (TypeError, ValueError):
                transform_values = None
    return ("__spatial__", crs_text, transform_values)


def _append_filename_spatial_signature(
    signatures: tuple[Any, ...], dataset, engine: str
) -> tuple[Any, ...]:
    spatial = _filename_dataset_spatial_signature(dataset, engine)
    return signatures if spatial is None else (*signatures, spatial)


def _filename_variable_signature(variable) -> tuple[Any, ...]:
    """Return processing-relevant metadata that must match across files.

    Raster drivers often add file-specific provenance/statistics attributes
    (for example TIFF software versions or per-file min/max statistics).  Those
    do not change the array schema or the converter's fill/scale semantics and
    must not make an otherwise compatible time series fail inspection.
    """
    all_attrs = _variable_attrs(variable)
    attrs = {
        key: all_attrs[key]
        for key in ("_FillValue", "missing_value", "scale_factor", "add_offset", "units")
        if key in all_attrs
    }
    return (
        str(variable.name),
        str(variable.dtype),
        tuple(str(dim) for dim in variable.dims),
        tuple(int(variable.sizes[dim]) for dim in variable.dims),
        tuple(sorted((str(key), repr(value)) for key, value in attrs.items())),
    )


def _inspect_filename_file_xarray(task) -> FileRecord:
    """Inspect one no-time source file through the xarray fallback path."""
    (
        path,
        engine,
        expected_signature,
        expected_lat_hash,
        expected_lon_hash,
        expected_lat_size,
        expected_lon_size,
        reference_specs,
        time_value,
    ) = task
    with _normalized_filename_dataset(path, engine) as (ds, _):
        current_signature = _append_filename_spatial_signature(
            tuple(
                sorted(
                    (
                        _filename_variable_signature(variable)
                        for variable in ds.data_vars.values()
                        if variable.dims == ("lat", "lon")
                        and np.dtype(variable.dtype).kind in "biufc"
                    ),
                    key=lambda item: item[0],
                )
            ),
            ds,
            engine,
        )
        if current_signature != expected_signature:
            raise FilenameTimeError(f"{path.name} 的变量结构或属性与首文件不一致。")
        if (
            ds.lat.size != expected_lat_size
            or ds.lon.size != expected_lon_size
            or _hash_axis(np.asarray(ds.lat.values)) != expected_lat_hash
            or _hash_axis(np.asarray(ds.lon.values)) != expected_lon_hash
        ):
            raise FilenameTimeError(f"{path.name} 的经纬度网格与首文件不一致。")
        stat = path.stat()
        record = FileRecord(
            path=path,
            size_bytes=stat.st_size,
            times=(np.datetime64(time_value, "ns"),),
            time_keys=(time_key(np.datetime64(time_value, "ns")),),
            lat_hash=expected_lat_hash,
            lon_hash=expected_lon_hash,
            lat_size=expected_lat_size,
            lon_size=expected_lon_size,
            variables=reference_specs,
            mtime_ns=stat.st_mtime_ns,
        )
        del ds
        return record


def _inspect_filename_file(task) -> FileRecord:
    """Inspect one file, preferring direct metadata readers."""
    try:
        return _inspect_filename_file_low_level(task)
    except _LowLevelUnsupported:
        try:
            return _inspect_filename_file_rasterio(task)
        except _LowLevelUnsupported:
            return _inspect_filename_file_xarray(task)
def _fast_filename_record(task) -> FileRecord:
    (
        path,
        _engine,
        _expected_signature,
        expected_lat_hash,
        expected_lon_hash,
        expected_lat_size,
        expected_lon_size,
        reference_specs,
        time_value,
    ) = task
    stat = path.stat()
    timestamp = np.datetime64(time_value, "ns")
    return FileRecord(
        path=path,
        size_bytes=stat.st_size,
        times=(timestamp,),
        time_keys=(time_key(timestamp),),
        lat_hash=expected_lat_hash,
        lon_hash=expected_lon_hash,
        lat_size=expected_lat_size,
        lon_size=expected_lon_size,
        variables=reference_specs,
        mtime_ns=stat.st_mtime_ns,
    )


def _sample_filename_tasks(tasks: list[tuple], limit: int) -> list[tuple]:
    if len(tasks) <= limit:
        return tasks
    indices = {0, len(tasks) - 1}
    for position in range(1, limit - 1):
        indices.add(round(position * (len(tasks) - 1) / (limit - 1)))
    return [tasks[index] for index in sorted(indices)]




def inspect_filename_inventory(
    scan: FilenameScan,
    requested_engine: str = "auto",
    *,
    workers: int | None = None,
    progress: bool = True,
    cached_inventory: Inventory | None = None,
    cancel_event=None,
    progress_callback=None,
    fast_structure_validation: bool = False,
) -> Inventory:
    if progress:
        print(f"扫描到 {len(scan.files)} 个文件，使用文件名时间模式。")
    engine = engine_for_path(scan.files[0], requested_engine)
    rasterio_reference = False
    if engine == "rasterio":
        try:
            lat, lon, reference_specs, expected_signature = _rasterio_metadata(scan.files[0])
        except _LowLevelUnsupported:
            pass
        else:
            rasterio_reference = True
    if not rasterio_reference:
        with _normalized_filename_dataset(scan.files[0], requested_engine) as (
            reference_ds,
            engine,
        ):
            if (
                "lat" not in reference_ds.coords
                or "lon" not in reference_ds.coords
                or reference_ds.lat.dims != ("lat",)
                or reference_ds.lon.dims != ("lon",)
            ):
                raise FilenameTimeError("源文件缺少可用的一维 lat/lon 坐标。")
            lat = np.asarray(reference_ds.lat.values).copy()
            lon = np.asarray(reference_ds.lon.values).copy()
            reference_specs = []
            for name, variable in reference_ds.data_vars.items():
                if variable.dims != ("lat", "lon") or np.dtype(variable.dtype).kind not in "biufc":
                    continue
                reference_specs.append(
                    VariableSpec(
                        name=name,
                        dims=("time", "lat", "lon"),
                        dtype=str(variable.dtype),
                        shape_without_time=(int(variable.sizes["lat"]), int(variable.sizes["lon"])),
                        native_chunks=None,
                        attrs=_variable_attrs(variable),
                    )
                )
            if not reference_specs:
                raise FilenameTimeError("源文件没有可转换的二维 lat/lon 变量。")
            expected_signature = _append_filename_spatial_signature(
                tuple(
                    sorted(
                        (
                            _filename_variable_signature(variable)
                            for variable in reference_ds.data_vars.values()
                            if variable.dims == ("lat", "lon")
                            and np.dtype(variable.dtype).kind in "biufc"
                        ),
                        key=lambda item: item[0],
                    )
                ),
                reference_ds,
                engine,
            )
            del reference_ds
    expected_lat_hash = _hash_axis(lat)
    expected_lon_hash = _hash_axis(lon)
    tasks = tuple(
        (
            path,
            engine,
            expected_signature,
            expected_lat_hash,
            expected_lon_hash,
            int(lat.size),
            int(lon.size),
            tuple(reference_specs),
            value,
        )
        for path, value in zip(scan.files, scan.actual_times)
    )
    cached_by_path = (
        {record.path: record for record in cached_inventory.files}
        if (
            cached_inventory is not None
            and cached_inventory.source_engine == engine
            and cached_inventory.source_mode == "filename"
        )
        else {}
    )
    records_by_path: dict[Path, FileRecord] = {}
    changed_tasks = []
    for task in tasks:
        path = task[0]
        expected_time = np.datetime64(task[-1], "ns")
        cached = cached_by_path.get(path)
        stat = path.stat()
        if (
            cached is not None
            and cached.size_bytes == stat.st_size
            and cached.mtime_ns is not None
            and cached.mtime_ns == stat.st_mtime_ns
            and cached.time_keys == (time_key(expected_time),)
        ):
            records_by_path[path] = cached
        else:
            changed_tasks.append(task)
    fast_mode = (
        fast_structure_validation
        and engine == "netcdf4"
        and len(changed_tasks) > FAST_STRUCTURE_SAMPLE_LIMIT
        and all(task[0].suffix.lower() == ".hdf" for task in changed_tasks)
    )
    validation_tasks = _sample_filename_tasks(changed_tasks, FAST_STRUCTURE_SAMPLE_LIMIT) if fast_mode else changed_tasks
    validation_paths = {task[0] for task in validation_tasks}
    worker_count = choose_inspection_workers(
        [task[0] for task in validation_tasks] or list(scan.files), workers
    )
    if progress:
        method = (
            "NetCDF4 低层元数据优先，失败时回退 xarray"
            if engine == "netcdf4"
            else "rasterio 低层元数据优先，失败时回退 xarray"
            if engine == "rasterio"
            else "xarray"
        )
        if fast_mode:
            print(
                f"复用 {len(records_by_path)} 个文件；快速抽样检查 {len(validation_tasks)} / "
                f"{len(changed_tasks)} 个文件（{worker_count} 个进程，{method}）……"
            )
        else:
            print(
                f"复用 {len(records_by_path)} 个文件；检查 {len(changed_tasks)} 个文件的"
                f"变量、网格和属性（{worker_count} 个进程，{method}）……"
            )
    total_changed = max(1, len(validation_tasks))
    completed_changed = 0
    report_every = max(1, total_changed // 100)
    if progress_callback is not None:
        progress_callback(
            completed_changed,
            total_changed,
            (
                f"快速结构检查：抽样验证 {len(validation_tasks)} / {len(changed_tasks)} 个文件"
                if fast_mode
                else f"准备读取结构：复用 {len(records_by_path)} 个，待检查 {len(changed_tasks)} 个文件"
            ),
        )
    pool = WorkerPool(
        max_workers=min(worker_count, max(1, len(validation_tasks)))
    )
    try:
        for record in pool.map(
            _inspect_filename_file,
            validation_tasks,
            cancel_event=cancel_event,
        ):
            records_by_path[record.path] = record
            completed_changed += 1
            if progress_callback is not None and (completed_changed == len(validation_tasks) or completed_changed % report_every == 0):
                progress_callback(
                    completed_changed,
                    total_changed,
                    f"读取文件结构：{completed_changed}/{len(validation_tasks)}",
                )
    finally:
        pool.close()
    if fast_mode:
        for task in changed_tasks:
            if task[0] not in validation_paths:
                records_by_path[task[0]] = _fast_filename_record(task)
    records = [records_by_path[path] for path in scan.files]

    full_times = np.asarray(scan.expected_times, dtype="datetime64[ns]")
    if scan.template == "doy":
        if scan.step_days:
            frequency, gaps = f"每 {scan.step_days} 天（按年度 DOY）", []
        else:
            rendered = ", ".join(f"{year}年={step}天" for year, step in scan.annual_steps)
            frequency, gaps = f"年度 DOY 尺度不一致（{rendered}）", []
    else:
        frequency, gaps = _infer_frequency(full_times)
    if progress_callback is not None:
        progress_callback(total_changed, total_changed, "结构检查完成")
    missing_keys = tuple(time_key(item) for item in scan.missing_times)
    return Inventory(
        input_dir=scan.input_dir,
        files=records,
        lat_values=lat,
        lon_values=lon,
        times=full_times,
        time_keys=tuple(time_key(item) for item in full_times),
        variables={item.name: item for item in reference_specs},
        source_engine=engine,
        source_dimensions=("filename", "lat", "lon"),
        frequency=frequency,
        gaps=gaps,
        total_bytes=sum(item.size_bytes for item in records),
        missing_time_keys=missing_keys,
        source_mode="filename",
        filename_template=scan.template,
        filename_step_days=scan.step_days or None,
        filename_annual_steps=scan.annual_steps,
    )


def source_path_by_time(inventory: Inventory) -> dict[str, Path]:
    result = {}
    for record in inventory.files:
        for key in record.time_keys:
            result[key] = record.path
    return result


def _output_dtype(spec: VariableSpec, transform: VariableTransform | None) -> np.dtype:
    return effective_output_dtype(spec, transform)


def filename_logical_bytes(
    inventory: Inventory,
    selection: Selection,
    transforms: dict[str, VariableTransform] | None = None,
    output_layout: OutputLayout | None = None,
) -> int:
    """Uncompressed output size, including synthetic theoretical times."""
    return selected_output_logical_bytes(
        inventory,
        selection,
        transforms,
        output_layout,
    )


def _numeric_attr(attrs: dict[str, Any], *names: str) -> float | int | None:
    for name in names:
        value = attrs.get(name)
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != 1:
                continue
            value = value[0]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _missing_output_value(
    spec: VariableSpec,
    transform: VariableTransform | None,
    output_dtype: np.dtype,
    *,
    require_missing: bool = True,
) -> float | int:
    if transform is not None and transform.output_fill is not None:
        value = transform.output_fill
    elif output_dtype.kind in "fc":
        return float("nan")
    elif transform is not None and transform.fill_values:
        value = transform.fill_values[0]
    else:
        value = _numeric_attr(spec.attrs, "_FillValue", "missing_value")
    if value is None:
        if not require_missing:
            return np.asarray(0).astype(output_dtype).item()
        raise FilenameTimeError(
            f"整型变量 {spec.name} 没有可用缺失值；请在转换参数中指定填充值。"
        )
    if output_dtype.kind in "iu":
        try:
            if float(value) != float(int(value)):
                raise FilenameTimeError(
                    f"整型变量 {spec.name} 的缺失值 {value!r} 不是整数。"
                )
        except (TypeError, ValueError, OverflowError) as exc:
            if isinstance(exc, FilenameTimeError):
                raise
            raise FilenameTimeError(
                f"变量 {spec.name} 的缺失值 {value!r} 无效。"
            ) from exc
        limits = np.iinfo(output_dtype)
        if value < limits.min or value > limits.max:
            raise FilenameTimeError(
                f"变量 {spec.name} 的缺失值 {value!r} 超出 {output_dtype} 可表示范围。"
            )
    try:
        return np.asarray(value).astype(output_dtype).item()
    except (TypeError, ValueError, OverflowError) as exc:
        raise FilenameTimeError(
            f"变量 {spec.name} 的缺失值 {value!r} 无法写入 {output_dtype}。"
        ) from exc


def _missing_mask(data: np.ndarray, values: tuple[float, ...] | None) -> np.ndarray:
    if not values:
        return np.zeros(data.shape, dtype=bool)
    mask = np.zeros(data.shape, dtype=bool)
    for value in values:
        try:
            if np.isnan(value) and np.issubdtype(data.dtype, np.floating):
                mask |= np.isnan(data)
                continue
        except TypeError:
            pass
        mask |= data == value
    return mask


def _prepare_filename_data(
    data: np.ndarray,
    spec: VariableSpec,
    transform: VariableTransform | None,
    output_dtype: np.dtype,
    output_fill: float | int,
) -> np.ndarray:
    raw = np.asarray(data)
    mask = _missing_mask(raw, transform.fill_values if transform else None)
    result = raw.astype(output_dtype, copy=True)
    if transform is not None and transform.scale_factor is not None:
        if mask.any():
            result[~mask] *= transform.scale_factor
        else:
            result *= transform.scale_factor
    if mask.any():
        result[mask] = output_fill
    return np.ascontiguousarray(result)


def _transformed_attrs(
    spec: VariableSpec,
    transform: VariableTransform | None,
    output_fill: float | int,
) -> dict[str, Any]:
    attrs = dict(spec.attrs)
    if transform is not None and transform.fill_values:
        attrs["_FillValue"] = output_fill
        attrs["missing_value"] = output_fill
    if transform is not None and transform.scale_factor is not None:
        attrs["source_scale_factor"] = transform.scale_factor
        attrs.pop("scale_factor", None)
        attrs.pop("add_offset", None)
    return attrs


def build_filename_fused_dataset(
    inventory: Inventory,
    selection: Selection,
    *,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    chunks: tuple[int, int, int] | None = None,
    output_layout: OutputLayout | None = None,
    cancel_event=None,
) -> object:
    """Build the single-pass fusion crop for filename mode, without Zarr.

    This mirrors ``filename_direct_write`` + ``_initialize_filename_zarr`` value
    and attribute semantics so the in-memory crop is value-for-value equivalent
    to the on-disk ``source-crop.zarr`` intermediate the disk path would write:
    per-time-slice reads go through the same ``_prepare_filename_data``
    transform (missing mask, output dtype cast, scale factor, output fill) and
    variables carry the same ``_transformed_attrs`` including the fill markers
    the resampling engine uses to mask missing values.

    The returned xarray Dataset is Dask-backed (lazy) so the caller owns its
    lifecycle and must close it.  It is memory-bounded by the fusion crop cap
    already enforced by the pipeline eligibility check.
    """

    import dask.array as da
    import xarray as xr

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("任务已取消。")
    variable_transforms = variable_transforms or {}
    variable_names = variable_names or {}
    source_map = source_path_by_time(inventory)
    selected_keys = inventory.time_keys[selection.time_start : selection.time_stop]
    nt, ny, nx = selection.shape
    lat_values = inventory.lat_values[selection.lat_start : selection.lat_stop]
    lon_values = inventory.lon_values[selection.lon_start : selection.lon_stop]
    reverse_lat = output_layout is not None and "lat" in output_layout.axis_reversals
    reverse_lon = output_layout is not None and "lon" in output_layout.axis_reversals
    if reverse_lat:
        lat_values = lat_values[::-1]
    if reverse_lon:
        lon_values = lon_values[::-1]

    with _normalized_filename_dataset(
        inventory.reference_file, inventory.source_engine
    ) as (ref, _):
        lat_attrs = ref.lat.attrs.copy()
        lon_attrs = ref.lon.attrs.copy()
        ref_attrs = dict(ref.attrs)
        scalar_coords = [
            (name, xr.Variable((), coordinate.values, attrs=coordinate.attrs.copy()))
            for name, coordinate in ref.coords.items()
            if name not in {"time", "lat", "lon"} and not coordinate.dims
        ]

    coords = {
        "time": xr.Variable(
            ("time",),
            inventory.times[selection.time_start : selection.time_stop],
            attrs={"source": "filename"},
        ),
        "lat": xr.Variable(("lat",), lat_values, attrs=lat_attrs),
        "lon": xr.Variable(("lon",), lon_values, attrs=lon_attrs),
    }
    for name, coordinate in scalar_coords:
        coords[name] = coordinate

    variables = {}
    for name in selection.variables:
        spec = inventory.variables[name]
        transform = variable_transforms.get(name)
        output_name = variable_names.get(name, name)
        dtype = _output_dtype(spec, transform)
        fill = _missing_output_value(
            spec,
            transform,
            dtype,
            require_missing=bool(inventory.missing_time_keys),
        )
        block = chunks or (max(1, min(nt, 1)), max(1, min(ny, 1)), max(1, min(nx, 1)))
        arrays = np.empty((nt, ny, nx), dtype=dtype)
        for row in range(nt):
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("任务已取消。")
            source = source_map.get(selected_keys[row])
            if source is None:
                arrays[row].fill(fill)
                continue
            with _normalized_filename_dataset(source, inventory.source_engine) as (ds, _):
                raw = np.asarray(
                    ds[name]
                    .isel(
                        lat=slice(selection.lat_start, selection.lat_start + ny),
                        lon=slice(selection.lon_start, selection.lon_start + nx),
                    )
                    .values
                )
            if reverse_lat:
                raw = np.flip(raw, axis=0)
            if reverse_lon:
                raw = np.flip(raw, axis=1)
            arrays[row] = _prepare_filename_data(raw, spec, transform, dtype, fill)
        variables[output_name] = xr.Variable(
            ("time", "lat", "lon"),
            da.from_array(arrays, chunks=block),
            attrs=_transformed_attrs(spec, transform, fill),
        )

    dataset_attrs = dict(ref_attrs)
    dataset_attrs.update({
        "source_mode": "filename",
        "filename_time_template": inventory.filename_template or "",
        "filename_step_days": inventory.filename_step_days,
        "filename_annual_steps": [list(item) for item in inventory.filename_annual_steps],
        "source_engine": inventory.source_engine,
    })
    ds = xr.Dataset(variables, coords=coords, attrs=dataset_attrs)
    # The on-disk intermediate is reopened by the resampling engine with
    # ``mask_and_scale=True``, which CF-decodes ``_FillValue``/``missing_value``
    # into NaN and ``scale_factor``/``add_offset`` into physical values.  Apply
    # the identical xarray decode here so the in-memory crop is
    # value-for-value equivalent to the reopened disk crop (packed integer
    # sources such as GLASS int16 + scale_factor depend on this).
    return xr.decode_cf(ds)


def _initialize_filename_zarr(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    plan: ConversionPlan,
    transforms: dict[str, VariableTransform],
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
) -> dict[str, tuple[np.dtype, float | int]]:
    import dask.array as da
    import xarray as xr

    with _normalized_filename_dataset(
        inventory.reference_file, inventory.source_engine
    ) as (ref, _):
        nt, ny, nx = selection.shape
        variables = {}
        encoding = {}
        effective: dict[str, tuple[np.dtype, float | int]] = {}
        variable_names = variable_names or {}
        default_compressor = make_compressor(
            plan.compression, plan.compression_level, plan.shuffle
        )
        for name in selection.variables:
            spec = inventory.variables[name]
            output_name = variable_names.get(name, name)
            transform = transforms.get(name)
            dtype = _output_dtype(spec, transform)
            fill = _missing_output_value(
                spec,
                transform,
                dtype,
                require_missing=bool(inventory.missing_time_keys),
            )
            effective[name] = (dtype, fill)
            attrs = _transformed_attrs(spec, transform, fill)
            shape = (nt, ny, nx)
            layout_item = output_layout.for_source(name) if output_layout else None
            if layout_item is not None:
                if layout_item.output_name != output_name:
                    raise ValueError(f"变量 {name} 的输出名称与最终布局不一致。")
                if layout_item.shape != shape or layout_item.dims != ("time", "lat", "lon"):
                    raise ValueError(f"变量 {name} 的 shape/dims 与最终布局不一致。")
                if np.dtype(layout_item.dtype) != dtype:
                    raise ValueError(f"变量 {name} 的 dtype 与最终布局不一致。")
                chunks = layout_item.chunks
                compressor = compressor_from_spec(layout_item.codec)
            else:
                chunks = (
                    min(nt, max(1, plan.chunk_time)),
                    min(ny, max(1, plan.chunk_lat)),
                    min(nx, max(1, plan.chunk_lon)),
                )
                compressor = default_compressor
            variables[output_name] = xr.Variable(
                ("time", "lat", "lon"), da.empty(shape, chunks=chunks, dtype=dtype), attrs=attrs
            )
            encoding[output_name] = {"chunks": chunks}
            if compressor is not None:
                encoding[output_name]["compressors"] = [compressor]
        lat_values = inventory.lat_values[selection.lat_start : selection.lat_stop]
        lon_values = inventory.lon_values[selection.lon_start : selection.lon_stop]
        if output_layout is not None and "lat" in output_layout.axis_reversals:
            lat_values = lat_values[::-1]
        if output_layout is not None and "lon" in output_layout.axis_reversals:
            lon_values = lon_values[::-1]
        coords = {
            "time": xr.Variable(
                ("time",),
                inventory.times[selection.time_start : selection.time_stop],
                attrs={"source": "filename"},
            ),
            "lat": xr.Variable(
                ("lat",),
                lat_values,
                attrs=ref.lat.attrs.copy(),
            ),
            "lon": xr.Variable(
                ("lon",),
                lon_values,
                attrs=ref.lon.attrs.copy(),
            ),
        }
        # Rasterio exposes CRS and affine transform metadata as scalar
        # ``spatial_ref`` coordinates.  Preserve scalar coordinates so a
        # GeoTIFF round-trip remains georeferenced instead of retaining only
        # the numeric pixel axes.
        for name, coordinate in ref.coords.items():
            if name in coords or coordinate.dims:
                continue
            coords[name] = xr.Variable(
                (), coordinate.values, attrs=coordinate.attrs.copy()
            )
            if output_layout is not None:
                coordinate_compressor = compressor_from_spec(
                    output_layout.coordinate_codec
                )
                if coordinate_compressor is not None:
                    encoding[name] = {"compressors": [coordinate_compressor]}
        for name, coordinate in coords.items():
            if coordinate.ndim:
                coordinate_layout = output_layout.for_output(name) if output_layout else None
                coordinate_chunks = (
                    coordinate_layout.chunks
                    if coordinate_layout is not None
                    else (min(len(coordinate), 8192),)
                )
                encoding[name] = {"chunks": coordinate_chunks}
                if coordinate_layout is not None:
                    coordinate_compressor = compressor_from_spec(coordinate_layout.codec)
                    if coordinate_compressor is not None:
                        encoding[name]["compressors"] = [coordinate_compressor]
        attrs = dict(ref.attrs)
        attrs.update({
            "source_mode": "filename",
            "filename_time_template": inventory.filename_template or "",
            "filename_step_days": inventory.filename_step_days,
            "filename_annual_steps": [list(item) for item in inventory.filename_annual_steps],
            "source_engine": inventory.source_engine,
        })
        template = xr.Dataset(variables, coords=coords, attrs=attrs)
        delayed = template.to_zarr(
            output,
            mode="w",
            encoding=encoding,
            consolidated=False,
            zarr_format=3,
            compute=False,
            align_chunks=False,
        )
        del delayed
        template.close()
        del ref
        return effective


def validate_filename_output(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    transforms: dict[str, VariableTransform],
    effective: dict[str, tuple[np.dtype, float | int]],
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
    *,
    points: int = 3,
) -> None:
    """Validate coordinates, a few real source cells, and synthetic gaps."""
    import xarray as xr
    import zarr

    nt, ny, nx = selection.shape
    with xr.open_zarr(output, consolidated=False, chunks=None, mask_and_scale=False) as ds:
        expected = {
            "time": nt,
            "lat": selection.shape[1],
            "lon": selection.shape[2],
        }
        if any(ds.sizes.get(dim) != size for dim, size in expected.items()):
            raise RuntimeError(f"输出维度不符合预期：{dict(ds.sizes)}，期望 {expected}")
        np.testing.assert_equal(ds.time.values, inventory.times[selection.time_start : selection.time_stop])
        expected_lat = inventory.lat_values[selection.lat_start : selection.lat_stop]
        expected_lon = inventory.lon_values[selection.lon_start : selection.lon_stop]
        if output_layout is not None and "lat" in output_layout.axis_reversals:
            expected_lat = expected_lat[::-1]
        if output_layout is not None and "lon" in output_layout.axis_reversals:
            expected_lon = expected_lon[::-1]
        np.testing.assert_equal(ds.lat.values, expected_lat)
        np.testing.assert_equal(ds.lon.values, expected_lon)
        output_names = {
            name: (variable_names or {}).get(name, name)
            for name in selection.variables
        }
        if set(output_names.values()) - set(ds.data_vars):
            raise RuntimeError("输出缺少所选变量。")

    group = zarr.open_group(output, mode="r")
    lookup = source_path_by_time(inventory)
    selected_keys = inventory.time_keys[selection.time_start : selection.time_stop]
    sample_indices = sorted(set(np.linspace(0, nt - 1, min(points, nt), dtype=int).tolist()))
    for index in sample_indices:
        key = selected_keys[index]
        path = lookup.get(key)
        for name in selection.variables:
            output_name = (variable_names or {}).get(name, name)
            dtype, fill = effective[name]
            if path is None:
                actual = group[output_name][index]
                if np.issubdtype(dtype, np.floating) and np.isnan(fill):
                    if not np.isnan(actual).all():
                        raise RuntimeError(f"缺失时间 {key} 的变量 {name} 未填充 NaN。")
                elif not np.all(actual == fill):
                    raise RuntimeError(f"缺失时间 {key} 的变量 {name} 未填充缺失值。")
                continue
            with _normalized_filename_dataset(path, inventory.source_engine) as (ds, _):
                raw = ds[name].isel(
                    lat=slice(selection.lat_start, selection.lat_stop),
                    lon=slice(selection.lon_start, selection.lon_stop),
                ).values
                expected_data = _prepare_filename_data(
                    raw, inventory.variables[name], transforms.get(name), dtype, fill
                )
                if output_layout is not None and "lat" in output_layout.axis_reversals:
                    expected_data = np.flip(expected_data, axis=0)
                if output_layout is not None and "lon" in output_layout.axis_reversals:
                    expected_data = np.flip(expected_data, axis=1)
            del ds
            actual = group[output_name][index]
            np.testing.assert_equal(actual, expected_data)


@dataclass(frozen=True)
class FilenameTimeWriteTask:
    """A batch of reconstructed times and spatial blocks for one worker."""

    time_start: int
    time_stop: int
    entries: tuple[tuple[int, str | None], ...]
    blocks: tuple[tuple[int, int, int, int], ...]


_FILENAME_OUTPUT_GROUP = None
_FILENAME_CONTEXT: dict[str, Any] = {}


def _filename_worker_init(
    output: str,
    engine: str,
    specs: dict[str, VariableSpec],
    transforms: dict[str, VariableTransform],
    effective: dict[str, tuple[str, Any]],
    lat_start: int,
    lon_start: int,
    lat_size: int,
    lon_size: int,
    reverse_lat: bool,
    reverse_lon: bool,
    variable_names: dict[str, str] | None = None,
) -> None:
    """Open one output group per process and retain immutable write context."""
    global _FILENAME_OUTPUT_GROUP, _FILENAME_CONTEXT
    import zarr

    _FILENAME_OUTPUT_GROUP = zarr.open_group(output, mode="r+")
    _FILENAME_CONTEXT = {
        "engine": engine,
        "specs": specs,
        "transforms": transforms,
        "effective": {
            name: (np.dtype(dtype), fill) for name, (dtype, fill) in effective.items()
        },
        "lat_start": lat_start,
        "lon_start": lon_start,
        "lat_size": lat_size,
        "lon_size": lon_size,
        "reverse_lat": reverse_lat,
        "reverse_lon": reverse_lon,
        "variable_names": variable_names or {},
    }


def _filename_write_task(task: FilenameTimeWriteTask) -> int:
    """Read source files and write each complete time/spatial Zarr chunk once."""
    context = _FILENAME_CONTEXT
    group = _FILENAME_OUTPUT_GROUP
    specs = context["specs"]
    transforms = context["transforms"]
    effective = context["effective"]
    variable_names = context["variable_names"]
    names = tuple(specs)
    indices = tuple(index for index, _ in task.entries)
    expected_indices = tuple(range(task.time_start, task.time_stop))
    if indices != expected_indices:
        raise ValueError("文件名写入任务的时间索引必须连续且与 chunk 边界一致。")

    total_bytes = 0
    with contextlib.ExitStack() as stack:
        sources = []
        for _, source_path in task.entries:
            if source_path is None:
                sources.append(None)
                continue
            dataset, _ = stack.enter_context(
                _normalized_filename_dataset(Path(source_path), context["engine"])
            )
            sources.append(dataset)

        for y0, y1, x0, x1 in task.blocks:
            source_y0, source_y1 = (
                (context["lat_size"] - y1, context["lat_size"] - y0)
                if context["reverse_lat"] else (y0, y1)
            )
            source_x0, source_x1 = (
                (context["lon_size"] - x1, context["lon_size"] - x0)
                if context["reverse_lon"] else (x0, x1)
            )
            source_lat = slice(
                context["lat_start"] + source_y0,
                context["lat_start"] + source_y1,
            )
            source_lon = slice(
                context["lon_start"] + source_x0,
                context["lon_start"] + source_x1,
            )
            block_shape = (task.time_stop - task.time_start, y1 - y0, x1 - x0)
            for name in names:
                dtype, fill = effective[name]
                data = np.empty(block_shape, dtype=dtype)
                for row, source in enumerate(sources):
                    if source is None:
                        data[row].fill(fill)
                        continue
                    raw = source[name].isel(lat=source_lat, lon=source_lon).values
                    if context["reverse_lat"]:
                        raw = np.flip(raw, axis=0)
                    if context["reverse_lon"]:
                        raw = np.flip(raw, axis=1)
                    data[row] = _prepare_filename_data(
                        raw, specs[name], transforms.get(name), dtype, fill
                    )
                group[variable_names.get(name, name)][
                    task.time_start : task.time_stop, y0:y1, x0:x1
                ] = data
                total_bytes += int(data.nbytes)
    return total_bytes



def _filename_read_phase(task: FilenameTimeWriteTask) -> dict[str, object]:
    """Read phase for phase-batched writes.

    Opens the source files for one task and extracts raw per-row block
    arrays (after axis reversal).  Writes are deferred to
    ``_filename_write_phase`` so all reads of a window happen before any
    writes, reducing same-device HDD head thrash.
    """
    context = _FILENAME_CONTEXT
    names = tuple(context["specs"])
    blocks: dict[str, list[list[np.ndarray | None]]] = {
        name: [] for name in names
    }
    with contextlib.ExitStack() as stack:
        sources = []
        for _, source_path in task.entries:
            if source_path is None:
                sources.append(None)
                continue
            dataset, _ = stack.enter_context(
                _normalized_filename_dataset(Path(source_path), context["engine"])
            )
            sources.append(dataset)

        for y0, y1, x0, x1 in task.blocks:
            source_y0, source_y1 = (
                (context["lat_size"] - y1, context["lat_size"] - y0)
                if context["reverse_lat"] else (y0, y1)
            )
            source_x0, source_x1 = (
                (context["lon_size"] - x1, context["lon_size"] - x0)
                if context["reverse_lon"] else (x0, x1)
            )
            source_lat = slice(
                context["lat_start"] + source_y0,
                context["lat_start"] + source_y1,
            )
            source_lon = slice(
                context["lon_start"] + source_x0,
                context["lon_start"] + source_x1,
            )
            for name in names:
                rows: list[np.ndarray | None] = []
                for source in sources:
                    if source is None:
                        rows.append(None)
                        continue
                    raw = source[name].isel(lat=source_lat, lon=source_lon).values
                    if context["reverse_lat"]:
                        raw = np.flip(raw, axis=0)
                    if context["reverse_lon"]:
                        raw = np.flip(raw, axis=1)
                    rows.append(raw)
                blocks[name].append(rows)
    return {"task": task, "blocks": blocks}


def _filename_write_phase(payload: dict[str, object]) -> int:
    """Write phase for phase-batched writes (mirror of the write half)."""
    context = _FILENAME_CONTEXT
    group = _FILENAME_OUTPUT_GROUP
    specs = context["specs"]
    transforms = context["transforms"]
    effective = context["effective"]
    variable_names = context["variable_names"]
    task = payload["task"]
    blocks = payload["blocks"]
    names = tuple(specs)
    total_bytes = 0
    for block_index, (y0, y1, x0, x1) in enumerate(task.blocks):
        block_shape = (task.time_stop - task.time_start, y1 - y0, x1 - x0)
        for name in names:
            dtype, fill = effective[name]
            data = np.empty(block_shape, dtype=dtype)
            for row, raw in enumerate(blocks[name][block_index]):
                if raw is None:
                    data[row].fill(fill)
                    continue
                data[row] = _prepare_filename_data(
                    raw, specs[name], transforms.get(name), dtype, fill
                )
            group[variable_names.get(name, name)][
                task.time_start : task.time_stop, y0:y1, x0:x1
            ] = data
            total_bytes += int(data.nbytes)
    return total_bytes


def filename_direct_write(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    plan: ConversionPlan,
    *,
    transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
    cancel_event=None,
    validate: bool = False,
    progress: bool = True,
    progress_callback=None,
    phase_batch: bool = False,
) -> dict[str, float | int]:
    """Parallel writer for files that contribute one reconstructed time slice."""
    transforms = transforms or {}
    variable_names = variable_names or {}
    effective = _initialize_filename_zarr(
        inventory,
        selection,
        output,
        plan,
        transforms,
        variable_names,
        output_layout,
    )
    effective_for_workers = {
        name: (str(dtype), fill) for name, (dtype, fill) in effective.items()
    }
    source_map = source_path_by_time(inventory)
    selected_keys = inventory.time_keys[selection.time_start : selection.time_stop]
    nt, ny, nx = selection.shape
    blocks = tuple(
        (y0, min(ny, y0 + max(1, plan.chunk_lat)), x0, min(nx, x0 + max(1, plan.chunk_lon)))
        for y0 in range(0, ny, max(1, plan.chunk_lat))
        for x0 in range(0, nx, max(1, plan.chunk_lon))
    )
    batch_size = plan.task_batch if plan.strategy == "file" else plan.chunk_time
    batch_size = max(1, batch_size)
    total_tasks = ceil(nt / batch_size)
    total_logical = filename_logical_bytes(
        inventory,
        selection,
        transforms,
        output_layout,
    )
    if total_tasks == 0:
        raise ValueError("没有可写入的时间点。")

    def write_tasks():
        for start in range(0, nt, batch_size):
            stop = min(nt, start + batch_size)
            yield FilenameTimeWriteTask(
                time_start=start,
                time_stop=stop,
                entries=tuple(
                    (
                        index,
                        str(source_map[selected_keys[index]])
                        if selected_keys[index] in source_map
                        else None,
                    )
                    for index in range(start, stop)
                ),
                blocks=blocks,
            )

    worker_count = max(1, min(plan.workers, total_tasks))
    stop = threading.Event()
    samples: list[tuple[float, int]] = []
    monitor = threading.Thread(target=_monitor, args=(stop, samples), daemon=True)
    logical_bytes = 0
    started = time.perf_counter()
    report_every = max(1, total_tasks // 100)
    monitor.start()
    if progress:
        print(
            progress_line(0, total_tasks, 0, 0.0, prefix="文件名时间写入"),
            end="",
            flush=True,
        )
    pool = WorkerPool(
        max_workers=worker_count,
        initializer=_filename_worker_init,
        initargs=(
            str(output),
            inventory.source_engine,
            {name: inventory.variables[name] for name in selection.variables},
            transforms,
            effective_for_workers,
            selection.lat_start,
            selection.lon_start,
            ny,
            nx,
            output_layout is not None and "lat" in output_layout.axis_reversals,
            output_layout is not None and "lon" in output_layout.axis_reversals,
            variable_names,
        ),
    )
    def _report(completed: int, amount: int) -> None:
        nonlocal logical_bytes
        logical_bytes += amount
        if progress_callback is not None:
            progress_callback(
                min(total_logical, logical_bytes),
                total_logical,
                logical_bytes,
                f"文件名时间写入：{completed}/{total_tasks}",
            )
        if progress and (completed == total_tasks or completed % report_every == 0):
            elapsed = max(time.perf_counter() - started, 1e-9)
            cpu, rss = samples[-1] if samples else (None, None)
            print(
                progress_line(
                    completed,
                    total_tasks,
                    logical_bytes,
                    elapsed,
                    prefix="文件名时间写入",
                    cpu=cpu,
                    rss=rss,
                ),
                end="",
                flush=True,
            )

    try:
        if phase_batch:
            # Same-device HDD optimization: process tasks in bounded windows
            # with a read phase followed by a write phase on the same pool.
            # All reads of a window complete before any writes start, which
            # reduces read/write head thrash; memory stays bounded to one
            # window of raw source arrays.
            window_size = max(1, worker_count)
            window: list[dict[str, object]] = []
            completed = 0
            for read_result in pool.map(
                _filename_read_phase,
                write_tasks(),
                cancel_event=cancel_event,
                max_pending=window_size,
            ):
                window.append(read_result)
                if len(window) >= window_size:
                    for amount in pool.map(
                        _filename_write_phase,
                        window,
                        cancel_event=cancel_event,
                    ):
                        completed += 1
                        _report(completed, amount)
                    window = []
            if window:
                for amount in pool.map(
                    _filename_write_phase,
                    window,
                    cancel_event=cancel_event,
                ):
                    completed += 1
                    _report(completed, amount)
        else:
            results = pool.map(
                _filename_write_task,
                write_tasks(),
                cancel_event=cancel_event,
            )
            for completed, amount in enumerate(results, 1):
                _report(completed, amount)
    finally:
        pool.close()
        stop.set()
        monitor.join(timeout=2)
    elapsed = time.perf_counter() - started
    if progress:
        print()
    if validate:
        validate_filename_output(
            inventory,
            selection,
            output,
            transforms,
            effective,
            variable_names,
            output_layout,
        )
    return {
        "elapsed": elapsed,
        "logical_bytes": filename_logical_bytes(
            inventory,
            selection,
            transforms,
            output_layout,
        ),
        "throughput_mib_s": logical_bytes / max(elapsed, 1e-9) / 1024**2,
        "average_cpu": sum(cpu for cpu, _ in samples) / len(samples) if samples else 0.0,
        "peak_rss": max((rss for _, rss in samples), default=0),
        "tasks": total_tasks,
        "chunk_owner_writes": total_tasks * len(blocks) * len(selection.variables),
        "write_mode": "chunk-owner",
        "phase_batch": bool(phase_batch),
    }


def convert_filename(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    *,
    transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    chunks: tuple[int, int, int] | None = None,
    output_layout: OutputLayout | None = None,
    cancel_event=None,
    plan: ConversionPlan | None = None,
    auto_tune: bool = False,
    tune_budget: float = 60.0,
    tuning_objective: str = "balanced",
    resource_budget: EffectiveResourceBudget | None = None,
    max_workers: int | None = None,
    reserve_gib: float = 2.0,
    overwrite: bool = False,
    validate: bool = True,
    progress: bool = True,
    staging_root: Path | None = None,
    phase_batch: bool = False,
    progress_callback=None,
) -> tuple[ConversionPlan, dict[str, float | int]]:
    """Tune and write one 2-D source file per reconstructed time coordinate."""
    output = validate_publish_target(
        output,
        overwrite=overwrite,
        operation="文件名时间转换",
    )
    preflight_writable(output.parent, "文件名时间转换输出")
    if output == inventory.input_dir:
        raise ValueError("输入目录和输出目录不能相同。")
    resource_budget = resource_budget or effective_resource_budget(
        source=inventory.input_dir,
        output=output,
        reserve_memory_bytes=int(max(0.0, float(reserve_gib)) * 1024**3),
        requested=max_workers if not auto_tune else None,
    )
    transforms = transforms or {}
    fixed_layout = chunks is not None or output_layout is not None
    plan_chunks = chunks
    if plan_chunks is None and output_layout is not None:
        plan_chunks = output_layout_plan_chunks(selection, output_layout)
    plan = resolve_conversion_plan(
        inventory,
        selection,
        output,
        plan=plan,
        chunks=plan_chunks,
        max_workers=max_workers if not auto_tune or fixed_layout else None,
        reserve_gib=reserve_gib,
        resource_budget=resource_budget,
    )
    tuning_results = []
    if auto_tune:
        if fixed_layout:
            candidates = fixed_layout_candidate_plans(
                inventory,
                selection,
                plan,
                max_workers=max_workers,
                reserve_gib=reserve_gib,
                worker_chunk_bytes=(
                    output_layout_max_chunk_bytes(selection, output_layout)
                    if output_layout is not None
                    else None
                ),
                resource_budget=resource_budget,
            )
            # Filename tasks advance along time. Keep each physical time chunk
            # under one worker while tuning concurrency; splitting it would
            # reintroduce concurrent read/modify/write of the same Zarr chunk.
            candidates = list(
                {
                    item.workers: replace(item, task_batch=plan.chunk_time)
                    for item in candidates
                }.values()
            )
        else:
            candidates = candidate_plans(
                inventory,
                selection,
                output,
                max_workers=max_workers,
                reserve_gib=reserve_gib,
                resource_budget=resource_budget,
            )
        plan, tuning_results = tune(
            inventory,
            selection,
            output,
            candidates,
            objective=tuning_objective,
            budget_seconds=tune_budget,
            progress=progress,
            writer=filename_direct_write,
            writer_kwargs={
                "transforms": transforms,
                "variable_names": variable_names or {},
                "output_layout": output_layout,
                "cancel_event": cancel_event,
                "validate": False,
            },
            logical_bytes_fn=lambda info, chosen: filename_logical_bytes(
                info,
                chosen,
                transforms,
                output_layout,
            ),
            fixed_layout=fixed_layout,
        )

    if tuning_results:
        selected_result = max(
            (item for item in tuning_results if item.plan == plan),
            key=lambda item: item.logical_mib_s,
            default=max(tuning_results, key=lambda item: item.logical_mib_s),
        )
        compression_ratio = selected_result.physical_bytes / max(
            selected_result.logical_bytes, 1
        )
        estimated_output = int(
            filename_logical_bytes(
                inventory,
                selection,
                transforms,
                output_layout,
            )
            * compression_ratio
            * COMPRESSION_SAFETY
        )
        free = shutil.disk_usage(output.parent).free
        if progress:
            print(
                f"依据实测压缩率估算输出约 {estimated_output / 1024**3:.1f} GiB；"
                f"目标磁盘可用 {free / 1024**3:.1f} GiB。"
            )
        if estimated_output > free * 0.95:
            raise OSError(
                f"预计输出 {estimated_output / 1024**3:.1f} GiB，"
                f"超过目标磁盘安全可用空间 {free * 0.95 / 1024**3:.1f} GiB。"
            )

    if progress:
        print("正式执行计划：" + plan.label())
        for reason in plan.rationale:
            print("  - " + reason)
    staging = make_staging_path(output, "filename-convert", staging_root)
    try:
        metrics = filename_direct_write(
            inventory,
            selection,
            staging,
            plan,
            transforms=transforms,
            variable_names=variable_names,
            output_layout=output_layout,
            cancel_event=cancel_event,
            validate=validate,
            progress=progress,
            progress_callback=progress_callback,
            phase_batch=phase_batch,
        )
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("任务已取消，未发布输出。")
        publish_staging(staging, output, "filename-convert", overwrite=overwrite)
        metrics = dict(metrics)
        successful_trials = [item for item in tuning_results if item.status == "ok"]
        selected_trial = next(
            (item for item in successful_trials if item.plan == plan),
            None,
        )
        metrics["resource_budget"] = resource_budget.to_dict()
        metrics["tuning"] = {
            "objective": str(tuning_objective),
            "near_best_threshold": 0.95,
            "candidate_trials": [item.to_dict() for item in tuning_results],
            "selected_candidate_id": (
                selected_trial.candidate_id if selected_trial is not None else None
            ),
            "selected_plan": {
                "strategy": plan.strategy,
                "workers": plan.workers,
                "chunks": list(plan.chunks),
                "task_batch": plan.task_batch,
                "compression": plan.compression,
                "compression_level": plan.compression_level,
                "shuffle": plan.shuffle,
            },
            "selection_reason": (
                f"{tuning_objective} 目标按实测候选选择"
                if tuning_results
                else "未启用自动调优"
            ),
            "rejected_candidates": [
                {
                    "candidate_id": item.candidate_id,
                    "workers": item.plan.workers,
                    "reason": item.failure,
                }
                for item in tuning_results
                if item.status != "ok"
            ],
        }
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return plan, metrics
