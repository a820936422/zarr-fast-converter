from __future__ import annotations

import gc
import itertools
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import zarr
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

from .models import (
    CompressionBenchmarkResult,
    CompressionObjective,
    CompressionPlan,
    CompressionResourceBudget,
    CompressionSelectionReport,
    DatasetInfo,
    VariableInfo,
)

DEFAULT_MAX_COMPRESSION_CANDIDATES = 8
DEFAULT_MAX_SAMPLE_BYTES = 64 * 1024**2
DEFAULT_DISK_SAFETY_FACTOR = 1.25
OBJECTIVES = frozenset(("speed", "balanced", "compact"))

_LEVELS = {"fast": 1, "balanced": 4, "maximum": 9, "auto": 1}
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
    """Build a validated, lossless Zarr v3 compression plan."""

    canonical_profile = {"speed": "fast", "compact": "maximum"}.get(profile, profile)
    if canonical_profile not in {"none", "fast", "balanced", "maximum", "auto"}:
        raise ValueError("压缩方案必须是 none、fast、balanced、maximum 或 auto。")
    selected_codec = codec or ("none" if canonical_profile == "none" else "blosc-zstd")
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
    selected_level = int(level if level is not None else _LEVELS.get(canonical_profile, 1))
    minimum, maximum = _CODEC_LEVELS[selected_codec]
    if not minimum <= selected_level <= maximum:
        raise ValueError(f"{selected_codec} 压缩等级必须位于 {minimum} 到 {maximum}。")
    descriptions = {
        "fast": "Zstd level 1；优先读取速度和较低 CPU 开销",
        "balanced": "Zstd level 4；压缩率、读取速度和 CPU 开销平衡",
        "maximum": "Zstd level 9；优先压缩率，CPU 开销较高",
        "auto": "自动候选；由真实代表性读写基准选择",
    }
    description = (
        descriptions[canonical_profile]
        if codec is None and level is None and shuffle == "auto"
        else f"{selected_codec} level {selected_level}；shuffle={shuffle}"
    )
    return CompressionPlan(
        canonical_profile if codec is None and level is None else "custom",
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
        return ZstdCodec(level=1)
    return _codec_for_dtype(np.dtype(variable.dtype), plan)


def _coerce_dtype(value: Any) -> np.dtype:
    if isinstance(value, np.ndarray):
        return np.dtype(value.dtype)
    return np.dtype(getattr(value, "dtype", value))


def _dtype_shuffle(dtype: np.dtype) -> str:
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.datetime64):
        return "bitshuffle" if dtype.itemsize > 1 else "noshuffle"
    if np.issubdtype(dtype, np.bool_):
        return "bitshuffle"
    if np.issubdtype(dtype, np.floating) or np.issubdtype(dtype, np.complexfloating):
        return "shuffle" if dtype.itemsize > 1 else "noshuffle"
    return "noshuffle"


def generate_compression_candidates(
    dtype: Any,
    chunk_shape: Sequence[int] | None = None,
    *,
    profile: str = "balanced",
    max_candidates: int = DEFAULT_MAX_COMPRESSION_CANDIDATES,
    include_none: bool = False,
    codec: str | None = None,
    level: int | None = None,
    shuffle: str = "auto",
    resource_budget: CompressionResourceBudget | Mapping[str, Any] | None = None,
) -> tuple[CompressionPlan, ...]:
    """Return a bounded, dtype-pruned set of controlled lossless candidates.

    A normal automatic run always compares native Zstd level 1 and Blosc LZ4.
    Explicit codec/level/shuffle values remain hard overrides.
    """

    maximum = max(1, int(max_candidates))
    dt = _coerce_dtype(dtype)
    chunks = tuple(int(item) for item in chunk_shape) if chunk_shape is not None else None
    if chunks is not None and any(item <= 0 for item in chunks):
        raise ValueError("chunk_shape 必须包含正整数。")
    if resource_budget is not None and chunks is not None:
        memory_limit = (
            resource_budget.memory_bytes
            if isinstance(resource_budget, CompressionResourceBudget)
            else resource_budget.get("memory_bytes")
        )
        estimate = int(np.prod(chunks, dtype=np.int64)) * max(1, dt.itemsize) * 3
        if memory_limit is not None and estimate > int(memory_limit):
            maximum = 1
    if profile == "none" and codec is None:
        return (make_compression_plan("none"),)
    if codec is not None:
        return (
            make_compression_plan(
                profile if profile != "none" else "balanced",
                codec=codec,
                level=level,
                shuffle=shuffle,
            ),
        )

    if profile in {"fast", "speed"}:
        configs = (("zstd", 1), ("blosc-lz4", 1), ("blosc-zstd", 1))
    elif profile in {"maximum", "compact"}:
        configs = (
            ("zstd", 1),
            ("blosc-lz4", 1),
            ("blosc-zstd", 3),
            ("zstd", 3),
            ("blosc-zstd", 6),
            ("blosc-lz4", 5),
        )
    else:
        configs = (
            ("zstd", 1),
            ("blosc-lz4", 1),
            ("blosc-zstd", 1),
            ("blosc-zstd", 3),
            ("zstd", 3),
            ("blosc-lz4", 5),
            ("blosc-zstd", 6),
        )
    # Blosc filters need a meaningful element size. Object/variable-width data
    # are pruned to native Zstd rather than guessing a filter layout.
    if dt.hasobject or dt.kind in {"O", "S", "U", "V"}:
        configs = tuple(item for item in configs if item[0] == "zstd")
    dtype_shuffle = _dtype_shuffle(dt) if shuffle == "auto" else shuffle
    result: list[CompressionPlan] = [make_compression_plan("none")] if include_none else []
    for candidate_codec, candidate_level in configs:
        selected_shuffle = "noshuffle" if candidate_codec == "zstd" else dtype_shuffle
        candidate = make_compression_plan(
            "maximum" if candidate_level >= 6 else "fast" if candidate_level == 1 else "balanced",
            codec=candidate_codec,
            level=candidate_level,
            shuffle=selected_shuffle,
        )
        if candidate not in result:
            result.append(candidate)
        if len(result) >= maximum:
            break
    return tuple(result[:maximum])


compression_candidates = generate_compression_candidates
candidate_compression_plans = generate_compression_candidates


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
        filter_name = (
            "n/a" if plan.codec in {"zstd", "gzip"} else _shuffle_for(variable, plan.shuffle)
        )
        lines.append(
            f"  {variable.name}: {variable.kind}, dtype={variable.dtype}, "
            f"{plan.codec} level {plan.level if plan.level is not None else '-'}, {filter_name}"
        )
    return "\n".join(lines)


def _valid(result: CompressionBenchmarkResult) -> bool:
    return bool(result.success and result.verified and result.disk_feasible)


def _metrics(result: CompressionBenchmarkResult) -> dict[str, float]:
    hot = max(float(result.hot_read_mib_s or 0.0), 1e-12)
    cold = max(float(result.cold_read_mib_s or 0.0), 1e-12)
    return {
        "write": max(float(result.write_mib_s or 0.0), 1e-12),
        "durable": max(float(result.durable_mib_s or 0.0), 1e-12),
        "read": max(math.sqrt(hot * cold), 1e-12),
        "bytes": max(float(result.compressed_bytes), 1.0),
        "cpu": max(float(result.average_cpu), 1e-6),
        "rss": max(float(result.peak_rss), 1.0),
    }


def dominates(
    left: CompressionBenchmarkResult,
    right: CompressionBenchmarkResult,
    *,
    tolerance: float = 1e-9,
) -> bool:
    """Return whether ``left`` is no worse in every measured dimension."""

    if not _valid(left) or not _valid(right):
        return False
    a, b = _metrics(left), _metrics(right)
    higher = ("write", "durable", "read")
    lower = ("bytes", "cpu", "rss")
    no_worse = all(a[name] >= b[name] * (1 - tolerance) for name in higher) and all(
        a[name] <= b[name] * (1 + tolerance) for name in lower
    )
    strict = any(a[name] > b[name] * (1 + tolerance) for name in higher) or any(
        a[name] < b[name] * (1 - tolerance) for name in lower
    )
    return no_worse and strict


def pareto_front(
    results: Iterable[CompressionBenchmarkResult],
    *,
    tolerance: float = 1e-9,
) -> tuple[CompressionBenchmarkResult, ...]:
    values = tuple(results)
    return tuple(
        item
        for index, item in enumerate(values)
        if _valid(item)
        and not any(
            other_index != index and dominates(other, item, tolerance=tolerance)
            for other_index, other in enumerate(values)
        )
    )


def pareto_indices(
    results: Iterable[CompressionBenchmarkResult],
    *,
    tolerance: float = 1e-9,
) -> tuple[int, ...]:
    values = tuple(results)
    front = pareto_front(values, tolerance=tolerance)
    return tuple(index for index, item in enumerate(values) if item in front)


_OBJECTIVE_WEIGHTS = {
    "speed": {"write": 0.36, "durable": 0.24, "read": 0.28, "bytes": 0.04, "cpu": 0.05, "rss": 0.03},
    "balanced": {"write": 0.18, "durable": 0.17, "read": 0.25, "bytes": 0.30, "cpu": 0.06, "rss": 0.04},
    "compact": {"write": 0.08, "durable": 0.07, "read": 0.15, "bytes": 0.58, "cpu": 0.07, "rss": 0.05},
}


def relative_log_score(
    result: CompressionBenchmarkResult,
    baseline: CompressionBenchmarkResult,
    *,
    objective: CompressionObjective = "balanced",
) -> float:
    """Return a deterministic relative-baseline logarithmic score."""

    if objective not in OBJECTIVES:
        raise ValueError("objective 必须是 speed、balanced 或 compact。")
    if not _valid(result) or not _valid(baseline):
        return float("-inf")
    candidate_values, baseline_values = _metrics(result), _metrics(baseline)
    score = 0.0
    for name, weight in _OBJECTIVE_WEIGHTS[objective].items():
        ratio = (
            candidate_values[name] / baseline_values[name]
            if name in {"write", "durable", "read"}
            else baseline_values[name] / candidate_values[name]
        )
        score += weight * math.log(max(ratio, 1e-12))
    return float(score)


score_compression_result = relative_log_score


def _baseline_result(
    results: Sequence[CompressionBenchmarkResult],
    baseline: CompressionPlan | CompressionBenchmarkResult | None,
) -> CompressionBenchmarkResult | None:
    if isinstance(baseline, CompressionBenchmarkResult) and _valid(baseline):
        return baseline
    if isinstance(baseline, CompressionPlan):
        match = next((item for item in results if item.plan == baseline and _valid(item)), None)
        if match is not None:
            return match
    return next((item for item in results if _valid(item)), None)


def select_compression_candidate(
    report_or_results: CompressionSelectionReport | Iterable[CompressionBenchmarkResult],
    *,
    objective: CompressionObjective = "balanced",
    baseline: CompressionPlan | CompressionBenchmarkResult | None = None,
) -> CompressionPlan | None:
    """Pareto-filter, score, and select only a losslessly verified candidate."""

    if isinstance(report_or_results, CompressionSelectionReport):
        values = tuple(report_or_results.results)
        baseline = baseline or report_or_results.baseline
    else:
        values = tuple(report_or_results)
    base = _baseline_result(values, baseline)
    front = pareto_front(values)
    if base is None or not front:
        return None
    return max(
        front,
        key=lambda item: (
            relative_log_score(item, base, objective=objective),
            -item.compressed_bytes,
            -item.candidate_index,
        ),
    ).plan


choose_compression = select_compression_candidate
select_compression = select_compression_candidate


def _cancelled(cancel_event: Any) -> bool:
    if cancel_event is None:
        return False
    checker = getattr(cancel_event, "is_set", None)
    if callable(checker):
        return bool(checker())
    return bool(cancel_event() if callable(cancel_event) else cancel_event)


def _shape_dtype(array: Any) -> tuple[tuple[int, ...], np.dtype, Any]:
    if getattr(array, "shape", None) is None or getattr(array, "dtype", None) is None:
        array = np.asarray(array)
    return tuple(int(item) for item in array.shape), np.dtype(array.dtype), array


def _normalise_chunks(
    chunks: Sequence[int] | None, shape: Sequence[int]
) -> tuple[int, ...] | None:
    if chunks is None:
        return None
    result = tuple(int(item) for item in chunks)
    if len(result) != len(shape) or any(item <= 0 for item in result):
        raise ValueError(f"chunk_shape 必须包含 {len(shape)} 个正整数。")
    return tuple(min(item, max(1, int(size))) for item, size in zip(result, shape))


def representative_compression_samples(
    array: Any,
    *,
    chunk_shape: Sequence[int] | None = None,
    max_samples: int = 3,
    max_sample_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
) -> tuple[np.ndarray, ...]:
    """Take bounded beginning/middle/end samples without changing real values."""

    shape, dtype, array = _shape_dtype(array)
    if not shape:
        return (np.asarray(array),)
    chunks = _normalise_chunks(chunk_shape, shape)
    sample_shape = [min(size, chunk) for size, chunk in zip(shape, chunks or shape)]
    while int(np.prod(sample_shape, dtype=np.int64)) * max(dtype.itemsize, 1) > max(1, int(max_sample_bytes)):
        index = max(range(len(sample_shape)), key=sample_shape.__getitem__)
        if sample_shape[index] <= 1:
            break
        sample_shape[index] = max(1, sample_shape[index] // 2)
    count = max(1, min(int(max_samples), 3))
    starts = [0]
    if shape[0] > sample_shape[0] and count > 1:
        starts += [(shape[0] - sample_shape[0]) // 2, shape[0] - sample_shape[0]]
    samples = []
    for start in list(dict.fromkeys(starts))[:count]:
        slices = [slice(0, size) for size in sample_shape]
        slices[0] = slice(start, start + sample_shape[0])
        sample = np.asarray(array[tuple(slices)])
        samples.append(sample if sample.flags.c_contiguous else np.ascontiguousarray(sample))
    return tuple(samples)


def _codec_for_dtype(dtype: np.dtype, plan: CompressionPlan) -> Any:
    if not plan.enabled:
        return None
    if plan.codec == "zstd":
        return ZstdCodec(level=int(plan.level if plan.level is not None else 0))
    if plan.codec == "gzip":
        return GzipCodec(level=int(plan.level if plan.level is not None else 0))
    return BloscCodec(
        typesize=max(dtype.itemsize, 1),
        cname=plan.codec.removeprefix("blosc-"),
        clevel=int(plan.level if plan.level is not None else 1),
        shuffle=plan.shuffle if plan.shuffle != "auto" else _dtype_shuffle(dtype),
    )


def _encoded_bytes(codec: Any, values: np.ndarray) -> bytes:
    if codec is None:
        return np.ascontiguousarray(values).tobytes()
    inner = None
    for name in ("_blosc_codec", "_zstd_codec", "_gzip_codec"):
        inner = getattr(codec, name, None)
        if inner is not None:
            break
    if inner is None:
        raise TypeError(f"无法访问 {type(codec).__name__} 的同步编码器")
    encoded = inner.encode(np.ascontiguousarray(values))
    return encoded if isinstance(encoded, bytes) else bytes(encoded)


def _chunk_views(values: np.ndarray, chunks: tuple[int, ...]) -> Iterable[np.ndarray]:
    starts = [range(0, size, chunk) for size, chunk in zip(values.shape, chunks)]
    for offsets in itertools.product(*starts):
        slices = tuple(
            slice(offset, min(size, offset + chunk))
            for offset, size, chunk in zip(offsets, values.shape, chunks)
        )
        yield values[slices]


def _payload_size(root: Path) -> int:
    metadata = {"zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"}
    return sum(
        item.stat().st_size
        for item in root.rglob("*")
        if item.is_file() and item.name not in metadata
    )


def _fsync_tree(root: Path) -> None:
    for item in root.rglob("*"):
        if item.is_file():
            try:
                with item.open("rb") as handle:
                    os.fsync(handle.fileno())
            except OSError:
                pass


def _drop_cache(root: Path) -> None:
    advise, advice = getattr(os, "posix_fadvise", None), getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or advice is None:
        return
    for item in root.rglob("*"):
        if not item.is_file():
            continue
        try:
            descriptor = os.open(item, os.O_RDONLY)
            try:
                advise(descriptor, 0, 0, advice)
            finally:
                os.close(descriptor)
        except OSError:
            pass


def _resources() -> tuple[float, int]:
    try:
        import psutil

        process = psutil.Process()
        cpu = process.cpu_times()
        return cpu.user + cpu.system, process.memory_info().rss
    except (ImportError, OSError, AttributeError):
        return 0.0, 0


def _benchmark_trial(
    values: np.ndarray,
    plan: CompressionPlan,
    path: Path,
    final_chunks: tuple[int, ...] | None,
) -> dict[str, float | int | bool]:
    chunks = tuple(min(size, chunk) for size, chunk in zip(values.shape, final_chunks or values.shape))
    codec = _codec_for_dtype(values.dtype, plan)
    encode_started = time.perf_counter()
    encoded_size = sum(len(_encoded_bytes(codec, chunk)) for chunk in _chunk_views(values, chunks))
    encode_seconds = time.perf_counter() - encode_started
    cpu_start, rss_start = _resources()
    write_started = time.perf_counter()
    target = zarr.create_array(
        path,
        shape=values.shape,
        chunks=chunks,
        dtype=values.dtype,
        compressors=[] if codec is None else [codec],
        zarr_format=3,
        overwrite=True,
    )
    target[:] = values
    write_seconds = time.perf_counter() - write_started
    _fsync_tree(path)
    durable_seconds = time.perf_counter() - write_started
    compressed_bytes = _payload_size(path)
    reader = zarr.open_array(path, mode="r")
    reader[:]  # warm the candidate before the timed hot decode
    hot_started = time.perf_counter()
    hot = np.asarray(reader[:])
    hot_seconds = time.perf_counter() - hot_started
    del reader
    _drop_cache(path)
    gc.collect()
    reader = zarr.open_array(path, mode="r")
    cold_started = time.perf_counter()
    cold = np.asarray(reader[:])
    cold_seconds = time.perf_counter() - cold_started
    verified = bool(
        np.array_equal(values, hot, equal_nan=True)
        and np.array_equal(values, cold, equal_nan=True)
    )
    cpu_end, rss_end = _resources()
    wall = max(time.perf_counter() - write_started, 1e-12)
    return {
        "logical_bytes": values.nbytes,
        "compressed_bytes": compressed_bytes or encoded_size,
        "encode_seconds": encode_seconds,
        "write_seconds": write_seconds,
        "durable_seconds": durable_seconds,
        "hot_read_seconds": hot_seconds,
        "cold_read_seconds": cold_seconds,
        "decode_seconds": hot_seconds + cold_seconds,
        "average_cpu": max(0.0, 100.0 * (cpu_end - cpu_start) / wall),
        "peak_rss": max(rss_start, rss_end),
        "verified": verified,
    }


def _failed(plan: CompressionPlan, index: int, error: str) -> CompressionBenchmarkResult:
    return CompressionBenchmarkResult(plan=plan, candidate_index=index, error=error)


def _final_report(
    candidates: tuple[CompressionPlan, ...],
    results: Sequence[CompressionBenchmarkResult],
    *,
    objective: CompressionObjective,
    baseline: CompressionPlan,
    cancelled: bool,
    budget_seconds: float | None,
    elapsed_seconds: float,
    max_samples: int,
    disk_free_bytes: int | None,
) -> CompressionSelectionReport:
    values = list(results)
    base = _baseline_result(values, baseline)
    indices = pareto_indices(values)
    if base is not None and indices:
        for index in indices:
            values[index] = replace(
                values[index], score=relative_log_score(values[index], base, objective=objective)
            )
        best = max(
            (values[index] for index in indices),
            key=lambda item: (float(item.score or 0), -item.compressed_bytes, -item.candidate_index),
        )
        selected, selected_index, fallback = best.plan, values.index(best), False
        reason = (
            f"objective={objective}；先做 Pareto 淘汰，再相对基线 {base.plan.label()} "
            f"进行对数评分；选择 {best.plan.label()}，score={float(best.score or 0):.6f}。"
        )
    else:
        selected, selected_index, fallback = None, None, True
        reason = (
            "调优已取消且没有可验证候选；保守回退到报告基线。"
            if cancelled
            else "没有通过无损逐值验证且满足磁盘容量的候选；保守回退到报告基线。"
        )
    return CompressionSelectionReport(
        candidates=candidates,
        results=tuple(values),
        objective=objective,
        baseline=baseline,
        pareto_indices=indices,
        selected=selected,
        selected_index=selected_index,
        selection_reason=reason,
        fallback=fallback,
        cancelled=cancelled,
        budget_seconds=budget_seconds,
        elapsed_seconds=elapsed_seconds,
        max_samples=max_samples,
        disk_free_bytes=disk_free_bytes,
    )


def benchmark_compression_candidates(
    array: Any,
    candidates: Iterable[CompressionPlan] | None = None,
    *,
    chunk_shape: Sequence[int] | None = None,
    output_dir: str | Path | None = None,
    objective: CompressionObjective = "balanced",
    baseline: CompressionPlan | None = None,
    budget_seconds: float | None = 60.0,
    max_samples: int = 3,
    max_sample_bytes: int = DEFAULT_MAX_SAMPLE_BYTES,
    max_candidates: int = DEFAULT_MAX_COMPRESSION_CANDIDATES,
    disk_free_bytes: int | None = None,
    disk_safety_factor: float = DEFAULT_DISK_SAFETY_FACTOR,
    cancel_event: Any = None,
    resource_budget: CompressionResourceBudget | Mapping[str, Any] | None = None,
    sample_arrays: Sequence[Any] | None = None,
    sample_sources: Sequence[tuple[Any, Sequence[int]]] | None = None,
    progress: bool = False,
) -> CompressionSelectionReport:
    """Benchmark real Zarr v3 encoding, durable writes, and hot/cold reads.

    Every physical chunk has one synchronous writer. Candidate failures remain
    in the report and never abort later candidates. The task-private directory
    is created on ``output_dir``'s filesystem and is always removed.
    """

    if objective not in OBJECTIVES:
        raise ValueError("objective 必须是 speed、balanced 或 compact。")
    if sample_arrays is not None and sample_sources is not None:
        raise ValueError("sample_arrays 与 sample_sources 不能同时指定。")
    shape, dtype, array = _shape_dtype(array)
    chunks = _normalise_chunks(chunk_shape, shape)
    source_specs: list[tuple[Any, tuple[int, ...], tuple[int, ...], np.dtype]] = [
        (array, chunks, shape, dtype)
    ]
    if sample_sources is not None:
        source_specs = []
        for source, source_chunks in sample_sources:
            source_shape, source_dtype, normalised_source = _shape_dtype(source)
            source_specs.append(
                (
                    normalised_source,
                    _normalise_chunks(source_chunks, source_shape),
                    source_shape,
                    source_dtype,
                )
            )
        if not source_specs:
            raise ValueError("sample_sources 不能为空。")
    plans = (
        generate_compression_candidates(
            dtype, chunks, max_candidates=max_candidates, resource_budget=resource_budget
        )
        if candidates is None
        else tuple(candidates)[: max(1, int(max_candidates))]
    )
    if not plans:
        plans = (make_compression_plan("fast", codec="zstd", level=1, shuffle="noshuffle"),)
    baseline = baseline or plans[0]
    if baseline not in plans:
        plans = ((baseline,) + plans)[: max(1, int(max_candidates))]
    memory_limit = None
    if isinstance(resource_budget, CompressionResourceBudget):
        disk_free_bytes = disk_free_bytes if disk_free_bytes is not None else resource_budget.disk_free_bytes
        memory_limit = resource_budget.memory_bytes
    elif resource_budget is not None:
        disk_free_bytes = disk_free_bytes if disk_free_bytes is not None else resource_budget.get("disk_free_bytes")
        memory_limit = resource_budget.get("memory_bytes")
    parent = Path(output_dir).expanduser().resolve() if output_dir is not None else Path(tempfile.gettempdir())
    parent.mkdir(parents=True, exist_ok=True)
    if disk_free_bytes is None:
        try:
            disk_free_bytes = shutil.disk_usage(parent).free
        except OSError:
            pass
    root = Path(tempfile.mkdtemp(prefix=".compression-tune-", dir=parent))
    started = time.perf_counter()
    cancelled = False
    results: list[CompressionBenchmarkResult] = []
    try:
        if sample_arrays is not None:
            samples = tuple(
                (np.ascontiguousarray(np.asarray(item)), chunks)
                for item in sample_arrays[: max(1, int(max_samples))]
            )
        else:
            grouped_samples = [
                tuple(
                    (sample, source_chunks)
                    for sample in representative_compression_samples(
                        source,
                        chunk_shape=source_chunks,
                        max_samples=max_samples,
                        max_sample_bytes=max_sample_bytes,
                    )
                )
                for source, source_chunks, _source_shape, _source_dtype in source_specs
            ]
            samples = tuple(
                item
                for round_items in itertools.zip_longest(*grouped_samples)
                for item in round_items
                if item is not None
            )
        full_logical = sum(
            int(np.prod(source_shape, dtype=np.int64)) * max(source_dtype.itemsize, 1)
            for _source, _source_chunks, source_shape, source_dtype in source_specs
        )
        for index, plan in enumerate(plans, 1):
            if _cancelled(cancel_event):
                cancelled = True
                results.append(_failed(plan, index, "cancelled"))
                continue
            if budget_seconds is not None and index > 1 and time.perf_counter() - started >= max(0.0, budget_seconds):
                results.append(_failed(plan, index, "budget exhausted"))
                continue
            if memory_limit is not None:
                working_set = max(
                    int(np.prod(source_chunks, dtype=np.int64))
                    * max(source_dtype.itemsize, 1)
                    * 3
                    for _source, source_chunks, _source_shape, source_dtype in source_specs
                )
                if working_set > int(memory_limit):
                    results.append(_failed(plan, index, "resource memory bound"))
                    continue
            sums = {name: 0.0 for name in (
                "logical_bytes", "compressed_bytes", "encode_seconds", "write_seconds",
                "durable_seconds", "hot_read_seconds", "cold_read_seconds", "decode_seconds",
                "cpu_weighted", "peak_rss",
            )}
            errors: list[str] = []
            completed = 0
            for sample_index, (sample, sample_chunks) in enumerate(samples, 1):
                if _cancelled(cancel_event):
                    cancelled = True
                    errors.append("cancelled")
                    break
                if budget_seconds is not None and completed and time.perf_counter() - started >= max(0.0, budget_seconds):
                    errors.append("budget exhausted")
                    break
                trial = root / f"candidate-{index}-{sample_index}.zarr"
                try:
                    measured = _benchmark_trial(sample, plan, trial, sample_chunks)
                    if not measured["verified"]:
                        errors.append("lossless value verification failed")
                    for name in (
                        "logical_bytes", "compressed_bytes", "encode_seconds", "write_seconds",
                        "durable_seconds", "hot_read_seconds", "cold_read_seconds", "decode_seconds",
                    ):
                        sums[name] += float(measured[name])
                    logical = max(int(measured["logical_bytes"]), 1)
                    sums["cpu_weighted"] += float(measured["average_cpu"]) * logical
                    sums["peak_rss"] = max(sums["peak_rss"], float(measured["peak_rss"]))
                    completed += 1
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                finally:
                    shutil.rmtree(trial, ignore_errors=True)
            if not completed:
                results.append(_failed(plan, index, "; ".join(errors) or "candidate failed"))
                continue
            logical, compressed = int(sums["logical_bytes"]), int(sums["compressed_bytes"])
            predicted = math.ceil(
                full_logical * compressed / max(logical, 1) * max(1.0, disk_safety_factor)
            )
            feasible = disk_free_bytes is None or predicted <= disk_free_bytes
            if not feasible:
                errors.append(f"predicted output {predicted} bytes exceeds free disk {disk_free_bytes} bytes")
            result = CompressionBenchmarkResult(
                plan=plan,
                candidate_index=index,
                logical_bytes=logical,
                compressed_bytes=compressed,
                encode_seconds=sums["encode_seconds"],
                write_seconds=sums["write_seconds"],
                durable_seconds=sums["durable_seconds"],
                hot_read_seconds=sums["hot_read_seconds"],
                cold_read_seconds=sums["cold_read_seconds"],
                decode_seconds=sums["decode_seconds"],
                average_cpu=sums["cpu_weighted"] / max(logical, 1),
                peak_rss=int(sums["peak_rss"]),
                sample_count=completed,
                verified=not errors,
                disk_feasible=feasible,
                success=not errors and feasible,
                error="; ".join(errors) or None,
            )
            results.append(result)
            if progress:
                print(
                    f"压缩候选 {index}/{len(plans)} {plan.label()}："
                    f"写 {result.write_mib_s or 0:.1f} MiB/s，"
                    f"冷读 {result.cold_read_mib_s or 0:.1f} MiB/s，"
                    f"体积 {result.compressed_bytes} bytes，验证={'通过' if result.verified else '失败'}"
                )
    finally:
        shutil.rmtree(root, ignore_errors=True)
    return _final_report(
        tuple(plans),
        results,
        objective=objective,
        baseline=baseline,
        cancelled=cancelled or _cancelled(cancel_event),
        budget_seconds=budget_seconds,
        elapsed_seconds=time.perf_counter() - started,
        max_samples=max(1, int(max_samples)),
        disk_free_bytes=disk_free_bytes,
    )


benchmark_compression = benchmark_compression_candidates
benchmark_compression_candidates_on_array = benchmark_compression_candidates
