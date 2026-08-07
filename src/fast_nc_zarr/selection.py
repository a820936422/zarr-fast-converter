from __future__ import annotations

import ast
from calendar import monthrange
import re
from typing import Any

import numpy as np

from .models import Inventory, Selection


_UNQUOTED_DATE_TOKEN = re.compile(
    r"\d{4}(?:-\d{1,2}(?:-\d{1,2})?)?"
    r"(?:[T ]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?"
)


def _parse_unquoted_date_list(text: str) -> list[str] | None:
    """Parse the interactive date-list form without requiring quotes.

    ``ast.literal_eval`` correctly parses ``['2001-01-01', '2022-12-31']``
    but rejects the more natural interactive form
    ``[2001-01-01,2022-12-31]`` because the hyphens are interpreted as
    Python subtraction operators.  Only ISO-like date/time tokens are
    accepted here, so malformed general input still receives the normal
    list-format error.
    """
    if len(text) < 2 or text[0] != "[" or text[-1] != "]":
        return None
    body = text[1:-1].strip()
    if not body:
        return []
    values = [item.strip() for item in body.split(",")]
    if all(_UNQUOTED_DATE_TOKEN.fullmatch(item) for item in values):
        return values
    return None


def parse_list(text: str, label: str) -> list[Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        value = _parse_unquoted_date_list(text)
        if value is None:
            raise ValueError(f"{label} 必须使用列表格式，例如 [2001, 2010]。") from exc
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label} 必须是列表。")
    return list(value)


def _date_boundary(value: Any, end: bool) -> np.datetime64:
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        text = text + ("-12-31T23:59:59.999999999" if end else "-01-01")
    elif len(text) == 7 and text[4] == "-":
        year, month = map(int, text.split("-"))
        if end:
            text = f"{year:04d}-{month:02d}-{monthrange(year, month)[1]:02d}T23:59:59.999999999"
        else:
            text += "-01"
    elif end and len(text) == 10:
        text += "T23:59:59.999999999"
    return np.datetime64(text, "ns")


def _time_indices(times: np.ndarray, bounds: list[Any] | None) -> tuple[int, int]:
    if bounds is None:
        return 0, len(times)
    if len(bounds) != 2:
        raise ValueError("时间范围必须包含两个值。")
    if np.issubdtype(times.dtype, np.datetime64):
        start = _date_boundary(bounds[0], False)
        end = _date_boundary(bounds[1], True)
        mask = (times.astype("datetime64[ns]") >= start) & (times.astype("datetime64[ns]") <= end)
    else:
        lower, upper = map(str, bounds)
        rendered = np.asarray([str(item) for item in times])
        mask = (rendered >= lower) & (rendered <= upper)
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError("指定时间范围没有匹配的数据。")
    if indices[-1] - indices[0] + 1 != len(indices):
        raise ValueError("时间选择不是连续区间。")
    return int(indices[0]), int(indices[-1]) + 1


def _axis_indices(values: np.ndarray, bounds: list[Any] | None, label: str) -> tuple[int, int]:
    if bounds is None:
        return 0, len(values)
    if len(bounds) != 2:
        raise ValueError(f"{label}范围必须包含两个值。")
    lower, upper = sorted(map(float, bounds))
    indices = np.flatnonzero((values >= lower) & (values <= upper))
    if not len(indices):
        raise ValueError(f"指定{label}范围没有匹配的数据。")
    if indices[-1] - indices[0] + 1 != len(indices):
        raise ValueError(f"{label}选择必须是连续区间。")
    return int(indices[0]), int(indices[-1]) + 1


def make_selection(
    inventory: Inventory,
    *,
    time_bounds: list[Any] | None = None,
    lat_bounds: list[Any] | None = None,
    lon_bounds: list[Any] | None = None,
    variables: list[str] | tuple[str, ...] | None = None,
) -> Selection:
    if variables is None:
        selected_variables = tuple(inventory.variables)
    else:
        unknown = sorted(set(variables) - set(inventory.variables))
        if unknown:
            raise ValueError("未知变量：" + ", ".join(unknown))
        selected_variables = tuple(dict.fromkeys(variables))
    if not selected_variables:
        raise ValueError("至少选择一个变量。")
    t0, t1 = _time_indices(inventory.times, time_bounds)
    y0, y1 = _axis_indices(inventory.lat_values, lat_bounds, "纬度")
    x0, x1 = _axis_indices(inventory.lon_values, lon_bounds, "经度")
    return Selection(selected_variables, t0, t1, y0, y1, x0, x1)


def selected_logical_bytes(inventory: Inventory, selection: Selection) -> int:
    nt, ny, nx = selection.shape
    total = 0
    sizes = {"time": nt, "lat": ny, "lon": nx}
    for name in selection.variables:
        spec = inventory.variables[name]
        count = 1
        non_time_sizes = iter(spec.shape_without_time)
        for dim in spec.dims:
            if dim == "time":
                count *= nt
            else:
                original_size = next(non_time_sizes)
                count *= sizes.get(dim, original_size)
        total += count * spec.itemsize
    return total
