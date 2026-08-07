from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

from .models import CompressionPlan, DatasetInfo, VariableInfo


_LEVELS = {
    "fast": 1,
    "balanced": 4,
    "maximum": 9,
}

_CODEC_LEVELS = {
    "blosc-zstd": (0, 9),
    "blosc-lz4": (0, 9),
    "blosc-lz4hc": (0, 9),
    "blosc-zlib": (0, 9),
    "zstd": (-7, 22),
    "gzip": (0, 9),
}


def make_compression_plan(
    profile: str = "balanced",
    *,
    codec: str | None = None,
    level: int | None = None,
    shuffle: str = "auto",
) -> CompressionPlan:
    if profile not in {"none", "fast", "balanced", "maximum"}:
        raise ValueError("压缩方案必须是 none、fast、balanced 或 maximum。")
    selected_codec = codec or ("none" if profile == "none" else "blosc-zstd")
    if selected_codec == "none":
        return CompressionPlan(
            "none", None, "保留输入变量的现有 Zarr v3 codec", codec="none"
        )
    if selected_codec not in _CODEC_LEVELS:
        raise ValueError(
            "压缩 codec 必须是 none、blosc-zstd、blosc-lz4、blosc-lz4hc、"
            "blosc-zlib、zstd 或 gzip。"
        )
    if shuffle not in {"auto", "noshuffle", "shuffle", "bitshuffle"}:
        raise ValueError("shuffle 必须是 auto、noshuffle、shuffle 或 bitshuffle。")
    if selected_codec in {"zstd", "gzip"} and shuffle not in {"auto", "noshuffle"}:
        raise ValueError(f"原生 {selected_codec} codec 不支持 Blosc shuffle。")
    selected_level = int(level if level is not None else _LEVELS.get(profile, 1))
    minimum, maximum = _CODEC_LEVELS[selected_codec]
    if not minimum <= selected_level <= maximum:
        raise ValueError(
            f"{selected_codec} 压缩等级必须位于 {minimum} 到 {maximum}。"
        )
    descriptions = {
        "fast": "Zstd level 1；优先读取速度和较低 CPU 开销",
        "balanced": "Zstd level 4；压缩率、读取速度和 CPU 开销平衡",
        "maximum": "Zstd level 9；优先压缩率，CPU 开销较高",
    }
    description = (
        descriptions[profile]
        if codec is None and level is None
        else f"{selected_codec} level {selected_level}；shuffle={shuffle}"
    )
    return CompressionPlan(
        profile if codec is None and level is None else "custom",
        selected_level,
        description,
        codec=selected_codec,
        shuffle=shuffle,
    )


def _shuffle_for(variable: VariableInfo, requested: str = "auto") -> str:
    if requested != "auto":
        return requested
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
    if plan.codec == "zstd":
        return ZstdCodec(level=int(plan.level or 0))
    if plan.codec == "gzip":
        return GzipCodec(level=int(plan.level or 0))
    cname = plan.codec.removeprefix("blosc-")
    return BloscCodec(
        typesize=max(1, int(variable.dtype.itemsize)),
        cname=cname,
        clevel=int(plan.level or 1),
        shuffle=_shuffle_for(variable, plan.shuffle),
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
        f"Codec：{plan.codec}；level={plan.level if plan.level is not None else '-'}",
        f"说明：{plan.description}",
        f"数据变量类型：{type_summary(info)}",
    ]
    if not plan.enabled:
        lines.append("所有变量保留输入中的 Zarr v3 codec。")
        return "\n".join(lines)
    for variable in info.data_variables:
        if variable.kind in {"integer", "floating"}:
            filter_name = (
                "n/a"
                if plan.codec in {"zstd", "gzip"}
                else _shuffle_for(variable, plan.shuffle)
            )
            lines.append(
                f"  {variable.name}: {variable.kind}, dtype={variable.dtype}, "
                f"{plan.codec} level {plan.level if plan.level is not None else '-'}, {filter_name}"
            )
        else:
            lines.append(f"  {variable.name}: {variable.kind}, 使用安全默认配置")
    return "\n".join(lines)
