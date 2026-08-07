from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from zarr.codecs import BloscCodec, ZstdCodec

from .models import CompressionPlan, DatasetInfo, VariableInfo


_LEVELS = {
    "fast": 1,
    "balanced": 4,
    "maximum": 9,
}


def make_compression_plan(profile: str) -> CompressionPlan:
    if profile not in {"none", "fast", "balanced", "maximum"}:
        raise ValueError("压缩方案必须是 none、fast、balanced 或 maximum。")
    if profile == "none":
        return CompressionPlan("none", None, "保留输入变量的现有 Zarr v3 codec")
    descriptions = {
        "fast": "Zstd level 1；优先读取速度和较低 CPU 开销",
        "balanced": "Zstd level 4；压缩率、读取速度和 CPU 开销平衡",
        "maximum": "Zstd level 9；优先压缩率，CPU 开销较高",
    }
    return CompressionPlan(profile, _LEVELS[profile], descriptions[profile])


def _shuffle_for(variable: VariableInfo) -> str:
    if variable.kind == "integer":
        return "bitshuffle"
    if variable.kind == "floating":
        return "shuffle"
    return "noshuffle"


def codec_for(
    variable: VariableInfo,
    plan: CompressionPlan,
    *,
    coordinate: bool = False,
) -> Any:
    """Return a Zarr v3 codec for a variable under a selected profile."""

    if not plan.enabled:
        return None
    if coordinate:
        # Coordinates are small and usually read in full; avoid expensive filters.
        return ZstdCodec(level=1)
    return BloscCodec(
        cname="zstd",
        clevel=int(plan.level or 1),
        shuffle=_shuffle_for(variable),
    )


def type_summary(info: DatasetInfo) -> str:
    counts = Counter(variable.kind for variable in info.data_variables)
    if not counts:
        return "没有三维数据变量"
    return "，".join(f"{kind}={count}" for kind, count in sorted(counts.items()))


def format_compression_plan(info: DatasetInfo, plan: CompressionPlan) -> str:
    lines = [
        "========== 压缩计划 ==========",
        f"方案：{plan.profile}",
        f"说明：{plan.description}",
        f"数据变量类型：{type_summary(info)}",
    ]
    if not plan.enabled:
        lines.append("所有变量保留输入中的 Zarr v3 codec。")
        return "\n".join(lines)
    for variable in info.data_variables:
        if variable.kind in {"integer", "floating"}:
            filter_name = "bitshuffle" if variable.kind == "integer" else "shuffle"
            lines.append(
                f"  {variable.name}: {variable.kind}, dtype={variable.dtype}, "
                f"Zstd level {plan.level or '-'}, {filter_name}"
            )
        else:
            lines.append(f"  {variable.name}: {variable.kind}, 使用安全默认配置")
    return "\n".join(lines)
