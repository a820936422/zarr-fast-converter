from __future__ import annotations

import json
import contextlib
import os
import threading
import shutil
import time
from tempfile import TemporaryDirectory
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np

from ..application.services import (
    ConversionConfig,
    RechunkConfig,
    inspect_resample,
    preview_rechunk,
    run_conversion,
    run_rechunk,
    run_resample,
)
from ..publication import preflight_writable, publish_staging, validate_publish_target
from ..selection import selected_logical_bytes, selected_output_logical_bytes
from ..runtime import ProcessLifecycle
from ..system import effective_resource_budget, runtime_resource_snapshot
from ..resampling.engine import (
    _build_regridder,
    _mask_missing,
    _resolve_local_source_window,
    _tile_target,
)
from ..resampling.grid import _axis_bounds
from ..resampling.environment import validate_resampling_environment
from ..resampling.models import GridInfo, ResampleConfig
from ..resampling.replacements import apply_replacement_rules, parse_replacement_rules
from ..validation import validate_semantic_samples
from .models import (
    MANIFEST_SCHEMA_VERSION,
    PipelineConfig,
    PipelinePaths,
    PipelinePlan,
    ZarrPipelinePlan,
)
from .planner import build_pipeline_plan


class PipelineExecutionError(RuntimeError):
    """Raised when an end-to-end task cannot safely publish its output."""


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _write_manifest(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(path)

def _write_event(path: Path, event: str, payload: dict[str, object] | None = None) -> None:
    """Append a best-effort machine-readable stage event."""

    record = {"timestamp": time.time(), "event": str(event), **(payload or {})}
    try:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            handle.flush()
    except OSError:
        # Telemetry must not turn a valid data product into a failed task.
        return

def _stage_event_payload(
    manifest: dict[str, object],
    stage: str,
    payload: dict[str, object] | None = None,
    *,
    checkpoint: str | None = None,
    terminal_status: str | None = None,
) -> dict[str, object]:
    backend = manifest.get("backend") if isinstance(manifest.get("backend"), dict) else {}
    evidence: dict[str, object] = {
        "stage": stage,
        "requested_backend": backend.get("requested"),
        "resolved_backend": backend.get("resolved"),
        "fallback": backend.get("fallback"),
        "fallback_reason": backend.get("fallback_reason"),
        "stage_checkpoint": checkpoint,
        "manifest": manifest.get("manifest"),
        "terminal_status": terminal_status,
    }
    config = manifest.get("config")
    if isinstance(config, dict):
        budget_values: dict[str, object] = {}
        sections = {
            "tune_budget": ("conversion", "tune_budget"),
            "rechunk_tune_budget": ("chunking", "tune_budget"),
            "compression_tune_budget": ("compression", "tune_budget"),
        }
        for name, (section_name, field_name) in sections.items():
            section = config.get(section_name)
            if isinstance(section, dict) and field_name in section:
                budget_values[name] = section[field_name]
        if budget_values:
            evidence["tuning_budgets"] = budget_values
    evidence.update(payload or {})
    return evidence


def _storage_progress_snapshot(roots: tuple[Path, ...]) -> tuple[int, int]:
    """Return observed bytes and file count without affecting task execution."""
    total_bytes = 0
    total_files = 0
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.expanduser().resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        try:
            if resolved.is_file():
                total_files += 1
                total_bytes += resolved.stat().st_size
                continue
            if not resolved.is_dir():
                continue
            for directory, _subdirectories, filenames in os.walk(resolved):
                for filename in filenames:
                    try:
                        total_bytes += (Path(directory) / filename).stat().st_size
                        total_files += 1
                    except OSError:
                        continue
        except OSError:
            continue
    return total_bytes, total_files


def _rss_bytes() -> int:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, OSError, AttributeError):
        return 0


@contextlib.contextmanager
def _stage_progress(
    event_path: Path,
    manifest: dict[str, object],
    stage: str,
    roots: tuple[Path, ...],
    *,
    checkpoint: str,
    logical_total: int = 0,
    progress_callback=None,
    interval_seconds: float = 10.0,
):
    """Emit logical progress separately from filesystem observations."""

    stop = threading.Event()
    started = time.perf_counter()
    sequence = 0
    current_completed = 0
    current_total = max(0, int(logical_total))
    current_logical_bytes = 0
    succeeded = False

    def emit(
        completed: int,
        status: str,
        eta_seconds: float | None,
        *,
        total: int | None = None,
        logical_bytes: int | None = None,
        message: str | None = None,
    ) -> None:
        nonlocal sequence, current_completed, current_total, current_logical_bytes
        total_value = current_total if total is None else max(0, int(total))
        completed_value = max(0, int(completed))
        logical_value = completed_value if logical_bytes is None else max(0, int(logical_bytes))
        current_total = total_value
        current_completed = min(completed_value, total_value) if total_value else completed_value
        current_logical_bytes = logical_value
        observed_bytes, observed_files = _storage_progress_snapshot(roots)
        elapsed = max(0.0, time.perf_counter() - started)
        logical_rate = logical_value / max(elapsed, 1e-9) / 1024**2
        observed_rate = observed_bytes / max(elapsed, 1e-9) / 1024**2
        effective_eta = eta_seconds
        if (
            effective_eta is None
            and status == "running"
            and logical_value > 0
            and total_value > logical_value
            and elapsed >= 0.2
        ):
            effective_eta = (total_value - logical_value) / max(logical_value / elapsed, 1e-9)
        progress_payload: dict[str, object] = {
            "status": status,
            "heartbeat": status == "running",
            "sequence": sequence,
            "elapsed_seconds": elapsed,
            "completed": current_completed,
            "total": total_value,
            "logical_bytes": logical_value,
            "temporary_bytes": int(observed_bytes),
            "rss_bytes": _rss_bytes(),
            "eta_seconds": effective_eta,
            "observed_bytes": int(observed_bytes),
            "observed_files": int(observed_files),
            "logical_throughput_mib_s": logical_rate,
            "observed_throughput_mib_s": observed_rate,
            "throughput_mib_s": observed_rate,
        }
        if message:
            progress_payload["message"] = message
        payload = _stage_event_payload(
            manifest,
            stage,
            progress_payload,
            checkpoint=checkpoint,
            terminal_status=status,
        )
        _write_event(event_path, "progress", payload)
        if progress_callback is not None:
            progress_callback(stage, payload)
        sequence += 1

    def report(
        completed: int,
        total: int | None = None,
        logical_bytes: int | None = None,
        message: str | None = None,
    ) -> None:
        emit(
            completed,
            "running",
            None,
            total=total,
            logical_bytes=logical_bytes,
            message=message,
        )

    emit(0, "running", None)

    def heartbeat() -> None:
        while not stop.wait(max(0.1, interval_seconds)):
            emit(
                current_completed,
                "running",
                None,
                total=current_total,
                logical_bytes=current_logical_bytes,
            )

    thread = threading.Thread(
        target=heartbeat,
        name=f"pipeline-progress-{stage}",
        daemon=True,
    )
    thread.start()
    try:
        yield report
        succeeded = True
    finally:
        stop.set()
        thread.join(timeout=2.0)
        if succeeded:
            emit(current_total, "completed", 0.0, total=current_total, logical_bytes=current_total)
        else:
            emit(
                current_completed,
                "failed",
                None,
                total=current_total,
                logical_bytes=current_logical_bytes,
            )

def _stage_progress_callback(report, logical_total: int):
    total_value = max(0, int(logical_total))

    def callback(completed: int, total: int, logical_bytes: int | None, message: str | None = None) -> None:
        if logical_bytes is None:
            value = int(total_value * max(0, int(completed)) / max(1, int(total)))
        else:
            value = min(total_value, max(0, int(logical_bytes)))
        report(value, total_value, value, message)

    return callback
def _apply_runtime_compression(
    manifest: dict,
    metrics: dict,
) -> None:
    selected = metrics.get("selected_compression")
    if not isinstance(selected, dict):
        return
    manifest["resolved_compression"] = selected
    layout = manifest.get("output_layout")
    variables = layout.get("variables", []) if isinstance(layout, dict) else []
    codec_name = str(selected.get("codec", ""))
    level = int(selected.get("level") or 0)
    requested_shuffle = str(selected.get("shuffle", "auto"))
    for variable in variables:
        if not isinstance(variable, dict) or variable.get("is_coord"):
            continue
        if codec_name.startswith("blosc-"):
            dtype = np.dtype(variable["dtype"])
            shuffle = requested_shuffle
            if shuffle == "auto":
                shuffle = (
                    "bitshuffle"
                    if np.issubdtype(dtype, np.integer)
                    else "shuffle"
                    if np.issubdtype(dtype, np.floating)
                    else "noshuffle"
                )
            variable["codec"] = {
                "kind": "blosc",
                "level": level,
                "cname": codec_name.removeprefix("blosc-"),
                "shuffle": shuffle,
            }
        else:
            variable["codec"] = {
                "kind": codec_name,
                "level": level,
                "cname": None,
                "shuffle": "noshuffle",
            }
def _apply_backend_metrics(manifest: dict, metrics: dict | None, requested: str) -> None:
    if isinstance(metrics, dict) and "backend" in metrics:
        manifest["backend"] = {
            "requested": requested,
            "resolved": str(metrics.get("backend", "python")),
            "fallback": bool(metrics.get("backend_fallback", False)),
            "fallback_reason": metrics.get("backend_fallback_reason"),
            "protocol_version": metrics.get("protocol_version"),
        }
    elif manifest.get("backend", {}).get("resolved") == "pending":
        manifest["backend"] = {
            "requested": requested,
            "resolved": "python",
            "fallback": requested in {"auto", "rust"},
            "fallback_reason": "该物理流程未调用Rust最终化阶段" if requested != "python" else None,
            "protocol_version": None,
        }


def _pipeline_resource_snapshot(inspection, config: PipelineConfig, paths: PipelinePaths):
    general = config.general
    return runtime_resource_snapshot(
        source=Path(inspection.path),
        temporary=paths.root,
        output=Path(general.output).expanduser().resolve().parent,
        storage_overrides={
            "source": general.source_storage,
            "temporary": general.temporary_storage,
            "output": general.output_storage,
        },
    )


def _print_resource_snapshot(snapshot) -> None:
    storage = snapshot.to_dict()["storage"]
    print(
        "运行资源："
        f"CPU worker 上限={snapshot.cpu.worker_ceiling}；"
        f"有效可用内存={snapshot.memory.effective_available_bytes / 1024**3:.1f} GiB；"
        f"源/临时/输出介质="
        f"{storage['source']['medium']}/{storage['temporary']['medium']}/"
        f"{storage['output']['medium']}"
    )
def _logical_output_bytes(plan: PipelinePlan) -> int:
    if plan.output_layout is None:
        return 0
    return sum(
        int(np.prod(item.shape, dtype=np.int64)) * np.dtype(item.dtype).itemsize
        for item in plan.output_layout.variables
        if not item.is_coord
    )


def _paths(config: PipelineConfig) -> PipelinePaths:
    output = Path(config.general.output).expanduser().resolve()
    base = (
        Path(config.general.temporary_dir).expanduser().resolve()
        if config.general.temporary_dir is not None
        else output.parent
    )
    root = base / "fast-nc-zarr-pipeline" / uuid4().hex
    return PipelinePaths(
        root=root,
        manifest=root / "manifest.json",
        converted=root / "source-crop.zarr",
        resampled=root / "resampled.zarr",
        final_staging=output.parent / f".{output.name}.pipeline-{uuid4().hex}.tmp",
        events=root / "events.jsonl",
    )

def _preflight_paths(paths: PipelinePaths, output_parent: Path) -> dict[str, dict[str, object]]:
    try:
        return {
            "temporary": preflight_writable(paths.root.parent, "一条龙临时"),
            "output": preflight_writable(output_parent, "一条龙输出"),
        }
    except (OSError, ValueError) as exc:
        raise PipelineExecutionError(
            f"一条龙开始前目录可写性预检失败：{exc}"
        ) from exc

def _nearest_existing(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _raw_storage_estimate(inspection, plan: PipelinePlan) -> tuple[int, int]:
    inventory = inspection.source_inventory
    selected_logical = selected_logical_bytes(inventory, plan.source_selection)
    full_selected_logical = sum(
        len(inventory.times)
        * len(inventory.lat_values)
        * len(inventory.lon_values)
        * inventory.variables[name].itemsize
        for name in plan.source_selection.variables
    )
    selected_fraction = selected_logical / max(1, full_selected_logical)
    converted_estimate = max(
        64 * 1024**2,
        int(inventory.total_bytes * selected_fraction * 1.25),
    )
    source_cells = max(1, int(np.prod(plan.source_selection.shape, dtype=np.int64)))
    target_cells = max(
        1,
        plan.source_selection.shape[0]
        * int(plan.target_grid.lat.size)
        * int(plan.target_grid.lon.size),
    )
    final_estimate = max(
        64 * 1024**2,
        int(converted_estimate * target_cells / source_cells),
    )
    temporary_estimate = 0
    if plan.needs_resample or plan.finalization_required:
        temporary_estimate += converted_estimate
    if plan.needs_resample:
        # xESMF may require one additional target-shaped intermediate when a
        # large time chunk cannot be written directly to the final layout.
        temporary_estimate += final_estimate
    if plan.finalization_required:
        temporary_estimate += final_estimate
    return temporary_estimate, final_estimate


def _check_raw_storage_capacity(inspection, plan: PipelinePlan, paths: PipelinePaths) -> None:
    temporary_estimate, final_estimate = _raw_storage_estimate(inspection, plan)
    temporary_base = paths.root.parent
    output_base = Path(paths.final_staging).parent
    temporary_existing = _nearest_existing(temporary_base)
    output_existing = _nearest_existing(output_base)
    temporary_usage = shutil.disk_usage(temporary_existing)
    output_usage = shutil.disk_usage(output_existing)
    same_filesystem = temporary_existing.stat().st_dev == output_existing.stat().st_dev
    required_on_temporary = temporary_estimate + (final_estimate if same_filesystem else 0)
    print("存储规划：")
    print(f"  临时任务根目录：{paths.root}")
    print(f"  转换中间 Zarr：{paths.converted}")
    print(f"  最终输出目录：{Path(paths.final_staging).parent}")
    print(
        f"  预计临时峰值：{temporary_estimate / 1024**3:.1f} GiB；"
        f"预计最终输出：{final_estimate / 1024**3:.1f} GiB"
    )
    print(
        f"  临时文件系统可用：{temporary_usage.free / 1024**3:.1f} GiB；"
        f"输出文件系统可用：{output_usage.free / 1024**3:.1f} GiB"
    )
    if required_on_temporary > temporary_usage.free * 0.9:
        raise PipelineExecutionError(
            "预计临时与最终产物需要 "
            f"{required_on_temporary / 1024**3:.1f} GiB，但临时文件系统仅可用 "
            f"{temporary_usage.free / 1024**3:.1f} GiB（保留 10% 安全余量）。"
        )
    if not same_filesystem and final_estimate > output_usage.free * 0.9:
        raise PipelineExecutionError(
            f"预计最终输出需要 {final_estimate / 1024**3:.1f} GiB，但输出文件系统仅可用 "
            f"{output_usage.free / 1024**3:.1f} GiB（保留 10% 安全余量）。"
        )


def _sample_indices(size: int) -> list[int]:
    if size <= 0:
        return []
    values = {0, size - 1, size // 2, size // 4, (3 * size) // 4}
    return sorted(value for value in values if 0 <= value < size)


def _reference_grid(source, source_path: Path) -> GridInfo:
    """Build the same regular-grid description used by the resampling engine."""

    lat = np.asarray(source.lat.values, dtype="float64")
    lon = np.asarray(source.lon.values, dtype="float64")
    if lat.size < 2 or lon.size < 2:
        raise PipelineExecutionError("数学验证要求源 lat/lon 至少各含两个格点。")
    return GridInfo(
        path=source_path,
        lat=lat,
        lon=lon,
        lat_bounds=_axis_bounds(lat),
        lon_bounds=_axis_bounds(lon),
        lat_resolution=float(np.median(np.abs(np.diff(lat)))),
        lon_resolution=float(np.median(np.abs(np.diff(lon)))),
        lat_descending=bool(lat[0] > lat[-1]),
        lon_descending=bool(lon[0] > lon[-1]),
        lat_uniform=True,
        lon_uniform=True,
    )


def _validation_pairs(target_grid, maximum: int) -> list[tuple[int, int]]:
    """Select deterministic windows across edges, centre and interior."""

    lat_indices = _sample_indices(int(target_grid.lat.size))
    lon_indices = _sample_indices(int(target_grid.lon.size))
    candidates = [(lat, lon) for lat in lat_indices for lon in lon_indices]
    if not candidates:
        return []
    count = min(len(candidates), max(1, int(maximum)))
    positions = np.linspace(0, len(candidates) - 1, count, dtype=int)
    return [candidates[int(position)] for position in dict.fromkeys(positions)]


def _validation_time_indices(size: int) -> list[int]:
    if size <= 0:
        return []
    # A long time series must include the end rather than merely its first
    # three quartile candidates; this catches final-chunk write mistakes.
    return sorted({0, int(size) // 2, int(size) - 1})


def validate_resample_samples(
    source_path: Path,
    output_path: Path,
    target_grid,
    config: PipelineConfig,
    inspection=None,
    *,
    max_samples: int = 6,
    cancel_event=None,
    progress: bool = True,
    replacement_statistics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Check samples with bounded local xESMF reference calculations.

    The reference follows the production engine's source-window and periodic
    rules, but builds one tiny weight matrix per target sample window and
    reuses it for every sampled time.  It must never construct a global-source
    weight matrix merely to validate one output point.
    """

    import xarray as xr
    import xesmf as xe

    # The verified conversion store is the exact production input, including
    # variable renames, fill-value replacement and scale factors.  Reopening
    # the raw collection here would duplicate conversion semantics, can force
    # a full multi-file metadata combine, and previously made validation look
    # stalled after resampling had already completed.
    del inspection
    source = xr.open_zarr(
        source_path,
        consolidated=False,
        chunks=None,
        decode_times=False,
        mask_and_scale=False,
    )
    reference_mode = "converted-source-crop"
    output = xr.open_zarr(
        output_path,
        consolidated=False,
        chunks=None,
        decode_times=False,
        mask_and_scale=False,
    )
    comparisons = 0
    max_error = 0.0
    weights_built = 0
    started = time.perf_counter()
    before_rules = parse_replacement_rules(
        config.resampling.before_conditions,
        config.resampling.before_results,
    )
    after_rules = parse_replacement_rules(
        config.resampling.after_conditions,
        config.resampling.after_results,
    )
    replacement_statistics = replacement_statistics or {}
    before_statistics = replacement_statistics.get("before", {})
    after_statistics = replacement_statistics.get("after", {})
    try:
        grid = _reference_grid(source, source_path)
        pairs = _validation_pairs(target_grid, max_samples)
        item_names = [
            name
            for name in output.data_vars
            if "lat" in output[name].dims and "lon" in output[name].dims
        ]
        time_indices = _validation_time_indices(int(output.sizes.get("time", 1)))
        total = len(pairs) * len(item_names) * len(time_indices)
        if progress:
            print(
                f"重采样数学验证：{len(pairs)} 个局部窗口、"
                f"{len(time_indices)} 个时间点、{len(item_names)} 个变量；"
                "按生产局部窗口规则计算。",
                flush=True,
            )
        with TemporaryDirectory(
            prefix=".fast-nc-zarr-validation-",
            dir=output_path.parent,
        ) as weight_root:
            for window_index, (lat_index, lon_index) in enumerate(pairs, start=1):
                if cancel_event is not None and cancel_event.is_set():
                    raise PipelineExecutionError("任务已取消。")
                lat_start = max(0, lat_index - 1)
                lat_stop = min(int(target_grid.lat.size), lat_index + 2)
                lon_start = max(0, lon_index - 1)
                lon_stop = min(int(target_grid.lon.size), lon_index + 2)
                target_tile = _tile_target(
                    target_grid,
                    lat_start,
                    lat_stop,
                    lon_start,
                    lon_stop,
                )
                local_target, source_lat_slice, source_lon_slice = _resolve_local_source_window(
                    grid,
                    target_tile,
                    config.resampling.method,
                )
                if source_lat_slice is None or source_lon_slice is None:
                    for name in item_names:
                        for time_index in time_indices:
                            actual = np.asarray(
                                output[name].isel(
                                    time=time_index,
                                    lat=lat_index,
                                    lon=lon_index,
                                ).values
                            )
                            if not bool(np.asarray(np.isnan(actual)).all()):
                                raise PipelineExecutionError(
                                    "重采样数学验证失败：无源覆盖目标格点未保持缺测。"
                                )
                            comparisons += 1
                    continue
                lat_source_start = int(source_lat_slice.start)
                lat_source_stop = int(source_lat_slice.stop)
                lon_source_start = int(source_lon_slice.start)
                lon_source_stop = int(source_lon_slice.stop)
                regridder = None
                weight_path = None
                try:
                    regridder, weight_path = _build_regridder(
                        grid.lat[source_lat_slice],
                        grid.lon[source_lon_slice],
                        grid.lat_bounds[lat_source_start : lat_source_stop + 1],
                        grid.lon_bounds[lon_source_start : lon_source_stop + 1],
                        local_target,
                        config.resampling.method,
                        Path(weight_root),
                        lat_attrs=dict(source.lat.attrs),
                        lon_attrs=dict(source.lon.attrs),
                        periodic=bool(
                            grid.periodic
                            and lon_source_start == 0
                            and lon_source_stop == grid.lon.size
                        ),
                    )
                    weights_built += 1
                    for name in item_names:
                        for time_index in time_indices:
                            if cancel_event is not None and cancel_event.is_set():
                                raise PipelineExecutionError("任务已取消。")
                            source_values = source[name].isel(
                                time=time_index,
                                lat=source_lat_slice,
                                lon=source_lon_slice,
                                drop=True,
                            ).transpose("lat", "lon")
                            source_values = _mask_missing(source_values, None)
                            if (
                                config.resampling.compute_dtype == "float32"
                                and np.issubdtype(source_values.dtype, np.floating)
                            ):
                                source_values = source_values.astype("float32")
                            if before_rules.rules:
                                source_values = source_values.copy(
                                    data=apply_replacement_rules(
                                        np.asarray(source_values.data),
                                        before_rules,
                                        before_statistics.get(name, {}),
                                    )
                                )
                            expected = regridder(
                                source_values,
                                keep_attrs=False,
                                skipna=config.resampling.skipna,
                                na_thres=config.resampling.na_thres,
                            )
                            expected_value = np.asarray(expected.values)[
                                lat_index - lat_start,
                                lon_index - lon_start,
                            ]
                            if after_rules.rules:
                                expected_value = apply_replacement_rules(
                                    np.asarray(expected_value),
                                    after_rules,
                                    after_statistics.get(name, {}),
                                )
                            actual_value = np.asarray(
                                output[name].isel(
                                    time=time_index,
                                    lat=lat_index,
                                    lon=lon_index,
                                ).values
                            )
                            expected_missing = bool(np.isnan(expected_value))
                            actual_missing = bool(np.isnan(actual_value))
                            if expected_missing or actual_missing:
                                if expected_missing != actual_missing:
                                    raise PipelineExecutionError(
                                        f"重采样数学验证缺测语义不一致：变量={name}，"
                                        f"time={time_index}，lat={lat_index}，lon={lon_index}"
                                    )
                            else:
                                error = float(abs(float(actual_value) - float(expected_value)))
                                scale = max(1.0, abs(float(expected_value)))
                                tolerance = (
                                    5e-5
                                    if np.dtype(output[name].dtype).itemsize <= 4
                                    else 1e-10
                                )
                                if error > tolerance * scale:
                                    raise PipelineExecutionError(
                                        f"重采样数学验证失败：变量={name}，time={time_index}，"
                                        f"lat={lat_index}，lon={lon_index}，实际={actual_value!r}，"
                                        f"参考={expected_value!r}"
                                    )
                                max_error = max(max_error, error)
                            comparisons += 1
                            if progress and (
                                comparisons == 1
                                or comparisons == total
                                or comparisons % max(1, total // 10) == 0
                            ):
                                print(
                                    f"重采样数学验证进度：{comparisons}/{total}；"
                                    f"窗口 {window_index}/{len(pairs)}",
                                    flush=True,
                                )
                            del source_values, expected
                finally:
                    del regridder
                    if weight_path is not None:
                        try:
                            weight_path.unlink()
                        except OSError:
                            pass
    finally:
        source.close()
        output.close()
    return {
        "comparisons": comparisons,
        "max_absolute_error": max_error,
        "xesmf_version": getattr(xe, "__version__", "unknown"),
        "method": config.resampling.method,
        "skipna": bool(config.resampling.skipna),
        "na_thres": float(config.resampling.na_thres),
        "reference_mode": reference_mode,
        "sample_windows": len(pairs),
        "weights_built": weights_built,
        "elapsed": time.perf_counter() - started,
    }


def preview_pipeline(inspection, config: PipelineConfig) -> PipelinePlan | ZarrPipelinePlan:
    """Build a no-write pipeline plan from an already completed inspection."""

    plan = build_pipeline_plan(inspection, config)
    if plan.needs_resample:
        validate_resampling_environment()
    return plan


def _semantic_validation(
    output: Path,
    config: PipelineConfig,
    *,
    progress: bool,
) -> dict[str, object] | None:
    if not config.validate:
        return None
    result = validate_semantic_samples(output, config.semantic_constraints)
    warnings = [str(item) for item in result.get("warnings", [])]
    if progress:
        for warning in warnings:
            print(f"语义检查提醒：{warning}", flush=True)
    if warnings and config.semantic_constraints:
        raise PipelineExecutionError("显式语义约束校验失败：" + "；".join(warnings))
    return result


def _run_zarr_pipeline(
    inspection,
    config: PipelineConfig,
    plan: ZarrPipelinePlan,
    *,
    cancel_event=None,
    progress: bool = True,
    progress_callback=None,
) -> dict[str, object]:
    paths = _paths(config)
    preflight = _preflight_paths(
        paths,
        Path(config.general.output).expanduser().resolve().parent,
    )
    paths.root.mkdir(parents=True, exist_ok=False)
    _write_event(
        paths.events,
        "started",
        {"job_id": paths.root.name, "stage": "preparation", "requested_backend": getattr(config, "backend", "python"), "resolved_backend": "pending", "fallback": None, "fallback_reason": None, "stage_checkpoint": None, "manifest": str(paths.manifest), "terminal_status": "running"},
    )
    output = Path(config.general.output).expanduser().resolve()
    resources = _pipeline_resource_snapshot(inspection, config, paths)
    resource_budget = effective_resource_budget(resources)
    _write_event(paths.events, "resource", {"stage": "preparation", "resource_budget": resource_budget.to_dict(), "manifest": str(paths.manifest), "terminal_status": "running"})
    if progress:
        _print_resource_snapshot(resources)
    requested_operations = {
        "conversion": False,
        "resampling": bool(config.operations.resample),
        "rechunking": bool(config.operations.rechunk),
        "recompression": bool(config.operations.recompress),
    }
    operation_decisions = {
        item.operation: asdict(item) for item in plan.operation_decisions
    }
    physical_stages = []
    if plan.needs_resample:
        physical_stages.append("resampling")
    if plan.finalization_required:
        physical_stages.append("finalization")
    requested_backend = getattr(config, "backend", "python")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "job_id": paths.root.name,
        "status": "running",
        "source": str(inspection.path),
        "preflight": preflight,
        "input_kind": getattr(inspection, "kind", "zarr"),
        "output": str(output),
        "temporary_root": str(paths.root),
        "manifest": str(paths.manifest),
        "events": str(paths.events),
        "backend": {
            "requested": requested_backend,
            "resolved": "pending",
            "fallback": None,
            "fallback_reason": None,
            "protocol_version": None,
        },
        "resource_snapshot": resources.to_dict(),
        "resource_budget": resource_budget.to_dict(),
        "storage_profiles": {
            "source": resource_budget.source_storage.to_dict(redact_paths=True) if resource_budget.source_storage else None,
            "temporary": resource_budget.temporary_storage.to_dict(redact_paths=True) if resource_budget.temporary_storage else None,
            "output": resource_budget.output_storage.to_dict(redact_paths=True) if resource_budget.output_storage else None,
        },
        "candidate_trials": {},
        "selection": {},
        "resolved_plans": {},
        "worker_memory_semantics": {
            "per_worker_peak_is_inclusive": True,
            "aggregate_parallel_peak_is_per_worker_peak_times_workers": True,
        },
        "cleanup_intermediate": bool(config.general.cleanup_intermediate),
        "needs_resample": bool(plan.needs_resample),
        "direct_finalization": bool(plan.direct_finalization),
        "requested_operations": requested_operations,
        "requested_operation_order": [
            name for name in config.requested_operations if name != "conversion"
        ],
        "operation_decisions": operation_decisions,
        "physical_stages": physical_stages,
        "target_shape": (
            plan.resample_plan.output_dimensions if plan.resample_plan else plan.input_info.dimensions
        ),
        "target_extent": (
            plan.resample_plan.target.spatial_extent if plan.resample_plan else None
        ),
        "config": asdict(config),
        "output_layout": asdict(plan.output_layout) if plan.output_layout else None,
        "resolved_compression": (
            asdict(plan.final_compression) if plan.final_compression is not None else None
        ),
        "stages": {},
        "resume": {"schema_version": 1, "checkpoints": {}},
    }
    lifecycle = ProcessLifecycle("pipeline")
    manifest["worker_lifecycle"] = lifecycle.to_dict()

    def update_lifecycle() -> None:
        lifecycle.sample()
        manifest["worker_lifecycle"] = lifecycle.to_dict()
    _write_manifest(paths.manifest, manifest)
    started = time.perf_counter()
    current = Path(inspection.path).expanduser().resolve()
    resample_metrics = None
    finalization_metrics = None
    validation_metrics = None
    resuming = getattr(inspection, "kind", None) == "temporary"
    current_stage = "preparation"
    try:
        if plan.needs_resample:
            options = config.resampling
            direct_resample_target = not plan.finalization_required
            if direct_resample_target:
                validate_publish_target(
                    output,
                    overwrite=config.general.overwrite,
                    operation="统一流程最终发布",
                    require_zarr_v3=True,
                )
            resample_output = paths.final_staging if direct_resample_target else paths.resampled
            if progress:
                print(f"统一流程阶段 1/{len(physical_stages)}：重采样现有 Zarr")
            current_stage = "resampling"
            with _stage_progress(
                paths.events,
                manifest,
                "resampling",
                (paths.root, resample_output),
                checkpoint="resampled.zarr" if plan.finalization_required else "output",
                logical_total=_logical_output_bytes(plan),
                progress_callback=progress_callback,
            ) as stage_report:
                resample_metrics = run_resample(
                    ResampleConfig(
                        input=current,
                        output=resample_output,
                        resolution=options.resolution,
                        method=options.method,
                        skipna=options.skipna,
                        na_thres=options.na_thres,
                        compute_dtype=options.compute_dtype,
                        extent="custom",
                        target_lat_bounds=(config.general.lat_min, config.general.lat_max),
                        target_lon_bounds=(config.general.lon_min, config.general.lon_max),
                        target_lat_descending=True,
                        target_lon_descending=False,
                        overwrite=False,
                        validate=config.validate,
                        tile_size=options.tile_size,
                        time_block=options.time_block,
                        compute_workers=options.compute_workers,
                        space_workers=options.space_workers,
                        tuning_objective=options.tuning_objective,
                        tune_budget=options.tune_budget,
                        temporary_dir=paths.root,
                        resource_budget=effective_resource_budget(resources),
                        output_layout=plan.output_layout,
                        before_replacements=parse_replacement_rules(
                            options.before_conditions, options.before_results
                        ),
                        after_replacements=parse_replacement_rules(
                            options.after_conditions, options.after_results
                        ),
                        statistics_policy=options.statistics_policy,
                        variable_options=options.variable_options,
                    ),
                    plan.resample_plan.inspection if plan.resample_plan else None,
                    cancel_event=cancel_event,
                    progress_callback=_stage_progress_callback(
                        stage_report,
                        _logical_output_bytes(plan),
                    ),
                )
            current = resample_output
            manifest["stages"]["resampling"] = {
                "status": "validating_math_samples" if resuming else "validated"
                if plan.finalization_required
                else "validated_staging",
                "metrics": resample_metrics,
            }
            manifest["candidate_trials"]["resampling"] = resample_metrics.get("tuning", {})
            manifest["selection"]["resampling"] = resample_metrics.get("tuning", {}).get("selection_reason")
            manifest["resolved_plans"]["resampling"] = resample_metrics.get("resolved_plan", {})
            _write_event(paths.events, "tuning", _stage_event_payload(manifest, "resampling", {"selection": manifest["selection"]["resampling"]}, checkpoint="resampled.zarr" if plan.finalization_required else "output", terminal_status="running"))
            update_lifecycle()
            _write_manifest(paths.manifest, manifest)
            if resuming:
                if plan.resample_plan is None:
                    raise PipelineExecutionError("恢复任务缺少重采样计划。")
                current_stage = "mathematical_validation"
                if progress:
                    print("重采样主体完成，开始恢复任务数学验证……", flush=True)
                with _stage_progress(
                    paths.events,
                    manifest,
                    "mathematical_validation",
                    (paths.root, resample_output),
                    checkpoint="resampled.zarr" if plan.finalization_required else "output",
                    logical_total=0,
                    progress_callback=progress_callback,
                ):
                    validation_metrics = validate_resample_samples(
                        Path(inspection.path),
                        resample_output,
                        plan.resample_plan.target,
                        config,
                        cancel_event=cancel_event,
                        progress=progress,
                        replacement_statistics=resample_metrics.get("replacement_statistics"),
                    )
                manifest["stages"]["resampling"]["mathematical_validation"] = validation_metrics
                manifest["stages"]["resampling"]["status"] = "validated"
                if direct_resample_target:
                    manifest["stages"]["resampling"]["status"] = "validated_staging"
                    current = resample_output
                else:
                    manifest["resume"]["checkpoints"]["resampling"] = {
                        "path": "resampled.zarr",
                        "status": "validated",
                        "dimensions": dict(plan.resample_plan.target.dimensions),
                    }

                update_lifecycle()
                _write_manifest(paths.manifest, manifest)
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineExecutionError("任务已取消。")
        if not plan.finalization_required:
            _apply_backend_metrics(manifest, None, requested_backend)
        if plan.finalization_required:
            chunking = config.chunking
            current_stage = "finalization"
            with _stage_progress(
                paths.events,
                manifest,
                "finalization",
                (paths.root, output),
                checkpoint="output",
                logical_total=_logical_output_bytes(plan),
                progress_callback=progress_callback,
            ) as stage_report:
                finalization_metrics = run_rechunk(
                    RechunkConfig(
                        input=current,
                        output=output,
                        strategy=chunking.strategy,
                        target_mib=chunking.target_mib,
                        custom_chunks=chunking.custom_chunks,
                        compression=(
                            config.compression.profile
                            if config.operations.recompress
                            else "none"
                        ),
                        workers=chunking.workers,
                        overwrite=config.general.overwrite,
                        validate=config.validate,
                        backend=getattr(config, "backend", "python"),
                        rechunk=config.operations.rechunk,
                        recompress=config.operations.recompress,
                        temporary_dir=paths.root,
                        compression_codec=config.compression.codec,
                        compression_level=config.compression.level,
                        storage_overrides={
                            "source": config.general.source_storage,
                            "temporary": config.general.temporary_storage,
                            "output": config.general.output_storage,
                        },
                        compression_shuffle=config.compression.shuffle,
                        compression_objective=config.compression.objective,
                        tune_budget_seconds=chunking.tune_budget,
                        compression_tune_budget=config.compression.tune_budget,
                        resource_budget=effective_resource_budget(resources),
                    ),
                    progress_callback=_stage_progress_callback(
                        stage_report,
                        _logical_output_bytes(plan),
                    ),
                    cancel_event=cancel_event,
                )
            manifest["backend"] = {
                "requested": requested_backend,
                "resolved": str(finalization_metrics.get("backend", "python")),
                "fallback": bool(finalization_metrics.get("backend_fallback", False)),
                "fallback_reason": finalization_metrics.get("backend_fallback_reason"),
                "protocol_version": finalization_metrics.get("protocol_version"),
            }
            manifest["stages"]["finalization"] = {
                "status": "published_and_validated",
                "metrics": finalization_metrics,
            }
            _apply_backend_metrics(manifest, finalization_metrics, requested_backend)
            if plan.needs_resample and config.general.cleanup_intermediate:
                shutil.rmtree(current)
                manifest["stages"]["resampling"]["status"] = "validated_and_cleaned"
        update_lifecycle()
        if plan.finalization_required:
            manifest["candidate_trials"]["finalization"] = finalization_metrics.get("worker_tuning", {})
            manifest["selection"]["finalization"] = {
                "tuning_objective": finalization_metrics.get("tuning_objective"),
                "worker_tuning": finalization_metrics.get("worker_tuning", {}),
            }
            manifest["resolved_plans"]["finalization"] = {
                "resource_budget": finalization_metrics.get("resource_budget"),
                "selected_compression": finalization_metrics.get("selected_compression"),
            }
            _write_event(paths.events, "tuning", _stage_event_payload(manifest, "finalization", {"selection": manifest["selection"]["finalization"]}, checkpoint="output", terminal_status="running"))
        semantic_output = output if plan.finalization_required else current
        semantic_validation = _semantic_validation(semantic_output, config, progress=progress)
        if not plan.finalization_required and plan.needs_resample:
            publish_staging(
                current,
                output,
                "pipeline-resume" if resuming else "pipeline",
                overwrite=config.general.overwrite,
                require_zarr_v3=True,
            )
            current = output
            manifest["stages"]["resampling"]["status"] = "published_as_final"
        manifest["semantic_validation"] = semantic_validation
        final_output_bytes = int(
            (finalization_metrics or resample_metrics or {}).get(
                "logical_bytes", plan.input_info.logical_bytes
            )
        )
        temporary_write_bytes = 0
        if plan.needs_resample:
            temporary_write_bytes += int(
                (resample_metrics or {}).get("intermediate_logical_bytes", 0)
            )
            if plan.finalization_required:
                temporary_write_bytes += int(
                    (resample_metrics or {}).get("logical_bytes", final_output_bytes)
                )
        if plan.finalization_required:
            # Rechunking materializes one complete intermediate before writing
            # final staging. The staging bytes are counted as final output.
            temporary_write_bytes += int(
                (finalization_metrics or {}).get("logical_bytes", final_output_bytes)
            )
        logical_io = {
            "final_output_bytes": final_output_bytes,
            "temporary_write_bytes": temporary_write_bytes,
            "total_write_bytes": final_output_bytes + temporary_write_bytes,
            "write_amplification": (
                (final_output_bytes + temporary_write_bytes)
                / max(1, final_output_bytes)
            ),
            "avoided_finalization_read_bytes": (
                final_output_bytes
                if plan.needs_resample and not plan.finalization_required
                else 0
            ),
            "avoided_finalization_write_bytes": (
                final_output_bytes
                if plan.needs_resample and not plan.finalization_required
                else 0
            ),
        }
        manifest["logical_io"] = logical_io
        manifest["status"] = "succeeded"
        manifest["elapsed"] = time.perf_counter() - started
        lifecycle.finish("completed")
        manifest["worker_lifecycle"] = lifecycle.to_dict()
        _write_event(paths.events, "finished", _stage_event_payload(manifest, "terminal", {"status": "succeeded", "elapsed": manifest["elapsed"]}, checkpoint="output", terminal_status="succeeded"))
        _write_manifest(paths.manifest, manifest)
        return {
            "output": str(output),
            "elapsed": manifest["elapsed"],
            "needs_resample": plan.needs_resample,
            "requested_operations": requested_operations,
            "operation_decisions": operation_decisions,
            "physical_stages": physical_stages,
            "manifest": str(paths.manifest),
            "conversion": None,
            "resampling": resample_metrics,
            "finalization": finalization_metrics,
            "mathematical_validation": validation_metrics,
            "semantic_validation": semantic_validation,
            "logical_io": logical_io,
        }
    except Exception as exc:
        cancelled = cancel_event is not None and cancel_event.is_set()
        terminal_status = "cancelled" if cancelled else "failed"
        lifecycle.finish(terminal_status)
        manifest["worker_lifecycle"] = lifecycle.to_dict()
        manifest["status"] = terminal_status
        manifest["failed_stage"] = current_stage
        manifest["elapsed"] = time.perf_counter() - started
        _write_event(
            paths.events,
            "finished",
            _stage_event_payload(
                manifest,
                current_stage,
                {"status": terminal_status, "error": str(exc)},
                checkpoint=None,
                terminal_status=terminal_status,
            ),
        )
        manifest["error"] = str(exc)
        _write_manifest(paths.manifest, manifest)
        if isinstance(exc, PipelineExecutionError):
            raise
        raise PipelineExecutionError(
            f"现有 Zarr 统一流程失败；任务临时目录保留用于排查：{paths.root}\n{exc}"
        ) from exc


def run_pipeline(
    inspection,
    config: PipelineConfig,
    *,
    cancel_event=None,
    progress: bool = True,
    progress_callback=None,
) -> dict[str, object]:
    plan = build_pipeline_plan(inspection, config)
    if config.backend == "rust" and not getattr(plan, "finalization_required", False):
        raise PipelineExecutionError(
            "当前Rust后端只支持兼容性最终化重分块；源转换和重采样仍使用Python后端。"
        )
    if plan.needs_resample:
        validate_resampling_environment()

    if isinstance(plan, ZarrPipelinePlan):
        return _run_zarr_pipeline(
            inspection,
            config,
            plan,
            cancel_event=cancel_event,
            progress=progress,
            progress_callback=progress_callback,
        )
    paths = _paths(config)
    _check_raw_storage_capacity(inspection, plan, paths)
    preflight = _preflight_paths(paths, paths.final_staging.parent)
    paths.root.mkdir(parents=True, exist_ok=False)
    resources = _pipeline_resource_snapshot(inspection, config, paths)
    resource_budget = effective_resource_budget(resources)
    _write_event(paths.events, "started", {"job_id": paths.root.name, "stage": "preparation", "requested_backend": getattr(config, "backend", "python"), "resolved_backend": "python", "fallback": getattr(config, "backend", "python") in {"auto", "rust"}, "fallback_reason": "raw pipeline conversion remains Python coordinator", "stage_checkpoint": None, "manifest": str(paths.manifest), "terminal_status": "running"})
    _write_event(paths.events, "resource", {"stage": "preparation", "resource_budget": resource_budget.to_dict(), "manifest": str(paths.manifest), "terminal_status": "running"})
    if progress:
        _print_resource_snapshot(resources)
    requested_operations = {
        "conversion": True,
        "resampling": bool(config.operations.resample),
        "rechunking": bool(config.operations.rechunk),
        "recompression": bool(config.operations.recompress),
    }
    operation_decisions = {
        decision.operation: asdict(decision) for decision in plan.operation_decisions
    }
    physical_stages = ["conversion"]
    if plan.needs_resample:
        physical_stages.append("resampling")
    if plan.finalization_required:
        physical_stages.append("finalization")
    requested_backend = getattr(config, "backend", "python")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "job_id": paths.root.name,
        "status": "running",
        "source": str(inspection.path),
        "input_kind": "raw" if getattr(inspection, "kind", None) == "source" else getattr(inspection, "kind", "unknown"),
        "preflight": preflight,
        "output": str(Path(config.general.output).expanduser().resolve()),
        "temporary_root": str(paths.root),
        "manifest": str(paths.manifest),
        "events": str(paths.events),
        "backend": {
            "requested": requested_backend,
            "resolved": "python",
            "fallback": False,
            "fallback_reason": None,
            "protocol_version": None,
        },
        "resource_snapshot": resources.to_dict(),
        "resource_budget": resource_budget.to_dict(),
        "storage_profiles": {
            "source": resource_budget.source_storage.to_dict(redact_paths=True) if resource_budget.source_storage else None,
            "temporary": resource_budget.temporary_storage.to_dict(redact_paths=True) if resource_budget.temporary_storage else None,
            "output": resource_budget.output_storage.to_dict(redact_paths=True) if resource_budget.output_storage else None,
        },
        "candidate_trials": {},
        "selection": {},
        "resolved_plans": {},
        "worker_memory_semantics": {
            "per_worker_peak_is_inclusive": True,
            "aggregate_parallel_peak_is_per_worker_peak_times_workers": True,
        },
        "cleanup_intermediate": bool(config.general.cleanup_intermediate),
        "needs_resample": bool(plan.needs_resample),
        "direct_finalization": bool(plan.direct_finalization),
        "requested_operations": requested_operations,
        "requested_operation_order": list(config.requested_operations),
        "operation_decisions": operation_decisions,
        "physical_stages": physical_stages,
        "output_layout": asdict(plan.output_layout) if plan.output_layout else None,
        "resolved_compression": (
            asdict(plan.final_compression) if plan.final_compression is not None else None
        ),
        "source_read_window": asdict(plan.source_read_window),
        "target_shape": plan.target_grid.dimensions,
        "target_extent": plan.target_grid.spatial_extent,
        "coverage_warning": plan.coverage_warning,
        "config": asdict(config),
        "resume": {"schema_version": 1, "checkpoints": {}},
        "stages": {},
    }
    lifecycle = ProcessLifecycle("pipeline")
    manifest["worker_lifecycle"] = lifecycle.to_dict()

    def update_lifecycle() -> None:
        lifecycle.sample()
        manifest["worker_lifecycle"] = lifecycle.to_dict()
    _write_manifest(paths.manifest, manifest)
    started = time.perf_counter()
    current = paths.converted
    current_stage = "preparation"
    try:
        conversion = config.conversion
        conversion_is_final = not plan.needs_resample and not plan.finalization_required
        resampling_is_final = plan.needs_resample and not plan.finalization_required
        final_target = Path(config.general.output).expanduser().resolve()
        if conversion_is_final or resampling_is_final:
            validate_publish_target(
                final_target,
                overwrite=config.general.overwrite,
                operation="一条龙最终发布",
                require_zarr_v3=True,
            )
        conversion_output = paths.final_staging if conversion_is_final else paths.converted
        conversion_budget = effective_resource_budget(
            resources,
            reserve_memory_bytes=int(
                max(0.0, float(config.conversion.reserve_memory_gib)) * 1024**3
            ),
            requested=(
                config.conversion.max_workers
                if not config.conversion.auto_tune
                else None
            ),
        )
        conversion_config = ConversionConfig(
            output=conversion_output,
            time_start=config.general.time_start,
            time_end=config.general.time_end,
            lat_min=plan.source_read_window.lat_bounds[0],
            lat_max=plan.source_read_window.lat_bounds[1],
            lon_min=plan.source_read_window.lon_bounds[0],
            lon_max=plan.source_read_window.lon_bounds[1],
            variables=conversion.variables,
            variable_names=conversion.variable_names,
            variable_transforms=conversion.variable_transforms,
            auto_tune=conversion.auto_tune,
            tuning_objective=config.conversion.tuning_objective,
            resource_budget=conversion_budget,
            tune_budget=conversion.tune_budget,
            max_workers=conversion.max_workers,
            reserve_memory_gib=conversion.reserve_memory_gib,
            chunks=plan.conversion_chunks,
            output_layout=plan.output_layout if conversion_is_final else None,
            overwrite=False,
            validate=config.validate,
        )
        conversion_logical_total = selected_output_logical_bytes(
            inspection.source_inventory,
            plan.source_selection,
            conversion.variable_transforms,
            plan.output_layout if conversion_is_final else None,
        )
        if progress:
            stage_total = len(physical_stages)
            print(
                f"一条龙阶段 1/{stage_total}：转换源读取窗口 "
                f"lat={plan.source_read_window.lat_bounds}，"
                f"lon={plan.source_read_window.lon_bounds}"
            )
            if plan.coverage_warning:
                print("覆盖提醒：" + plan.coverage_warning)
        current_stage = "conversion"
        with _stage_progress(
            paths.events,
            manifest,
            "conversion",
            (paths.root, conversion_output),
            checkpoint="source-crop.zarr" if not conversion_is_final else "output",
            logical_total=conversion_logical_total,
            progress_callback=progress_callback,
        ) as stage_report:
            conversion_plan, conversion_metrics = run_conversion(
                inspection,
                conversion_config,
                cancel_event=cancel_event,
                progress_callback=_stage_progress_callback(
                    stage_report,
                    conversion_logical_total,
                ),
            )
        manifest["stages"]["conversion"] = {
            "status": "validated_staging" if conversion_is_final else "validated",
            "metrics": conversion_metrics,
            "plan": asdict(conversion_plan),
        }
        current = conversion_output
        manifest["candidate_trials"]["conversion"] = conversion_metrics.get("tuning", {})
        manifest["selection"]["conversion"] = conversion_metrics.get("tuning", {}).get("selection_reason")
        manifest["resolved_plans"]["conversion"] = asdict(conversion_plan)
        _write_event(paths.events, "tuning", _stage_event_payload(manifest, "conversion", {"selection": manifest["selection"]["conversion"]}, checkpoint="source-crop.zarr" if not conversion_is_final else "output", terminal_status="running"))
        if not conversion_is_final:
            manifest["resume"]["checkpoints"]["conversion"] = {
                "path": "source-crop.zarr",
                "status": "validated",
                "dimensions": {
                    "time": plan.source_selection.shape[0],
                    "lat": plan.source_read_window.lat_shape,
                    "lon": plan.source_read_window.lon_shape,
                },
                "variables": [
                    config.conversion.variable_names.get(name, name)
                    for name in plan.source_selection.variables
                ],
            }
        update_lifecycle()
        _write_manifest(paths.manifest, manifest)
        if cancel_event is not None and cancel_event.is_set():
            raise PipelineExecutionError("任务已取消。")

        resample_metrics = None
        validation_metrics = None
        if plan.needs_resample:
            resampling = config.resampling
            resample_output = (
                paths.final_staging
                if resampling_is_final
                else paths.resampled
            )
            resample_config = ResampleConfig(
                input=paths.converted,
                output=resample_output,
                resolution=config.resampling.resolution,
                method=resampling.method,
                skipna=resampling.skipna,
                na_thres=resampling.na_thres,
                compute_dtype=resampling.compute_dtype,
                extent="custom",
                target_lat_bounds=(config.general.lat_min, config.general.lat_max),
                target_lon_bounds=(config.general.lon_min, config.general.lon_max),
                target_lat_descending=True,
                target_lon_descending=False,
                overwrite=False,
                validate=config.validate,
                tuning_objective=resampling.tuning_objective,
                tune_budget=resampling.tune_budget,
                tile_size=resampling.tile_size,
                # The conversion planner has already resolved a safe time
                # batch for its temporary layout.  Do not independently
                # re-auto-tune it after conversion or the source and output
                # time chunks can drift apart.
                time_block=(
                    plan.conversion_chunks[0]
                    if resampling.time_block == "auto" and plan.conversion_chunks is not None
                    else resampling.time_block
                ),
                compute_workers=resampling.compute_workers,
                space_workers=resampling.space_workers,
                resource_budget=effective_resource_budget(resources),
                temporary_dir=paths.root,
                output_layout=plan.output_layout if resampling_is_final else None,
                before_replacements=parse_replacement_rules(
                    resampling.before_conditions, resampling.before_results
                ),
                after_replacements=parse_replacement_rules(
                    resampling.after_conditions, resampling.after_results
                ),
                statistics_policy=resampling.statistics_policy,
                variable_options=resampling.variable_options,
            )
            if progress:
                print(
                    f"一条龙阶段 2/{len(physical_stages)}："
                    "xESMF 重采样到目标网格"
                )
            current_stage = "resampling"
            with _stage_progress(
                paths.events,
                manifest,
                "resampling",
                (paths.root, resample_output),
                checkpoint="resampled.zarr" if not resampling_is_final else "output",
                logical_total=_logical_output_bytes(plan),
                progress_callback=progress_callback,
            ) as stage_report:
                resample_metrics = run_resample(
                    resample_config,
                    cancel_event=cancel_event,
                    progress_callback=_stage_progress_callback(
                        stage_report,
                        _logical_output_bytes(plan),
                    ),
                )
            manifest["stages"]["resampling"] = {
                "status": "validating_math_samples",
                "metrics": resample_metrics,
            }
            update_lifecycle()
            _write_manifest(paths.manifest, manifest)
            if progress:
                print("重采样主体完成，开始局部数学验证……", flush=True)
            current_stage = "mathematical_validation"
            with _stage_progress(
                paths.events,
                manifest,
                "mathematical_validation",
                (paths.root, resample_output),
                checkpoint="resampled.zarr" if not resampling_is_final else "output",
                logical_total=0,
                progress_callback=progress_callback,
            ):
                validation_metrics = validate_resample_samples(
                    paths.converted,
                    resample_output,
                    plan.target_grid,
                    config,
                    inspection=inspection,
                    cancel_event=cancel_event,
                    progress=progress,
                    replacement_statistics=resample_metrics.get("replacement_statistics"),
                )
            manifest["stages"]["resampling"] = {
                "status": "validated",
                "metrics": resample_metrics,
                "mathematical_validation": validation_metrics,
            }
            if not resampling_is_final:
                manifest["resume"]["checkpoints"]["resampling"] = {
                    "path": "resampled.zarr",
                    "status": "validated",
                    "dimensions": dict(plan.target_grid.dimensions),
                }
            if resampling_is_final:
                manifest["stages"]["resampling"]["status"] = "validated_staging"
            manifest["candidate_trials"]["resampling"] = resample_metrics.get("tuning", {})
            manifest["selection"]["resampling"] = resample_metrics.get("tuning", {}).get("selection_reason")
            manifest["resolved_plans"]["resampling"] = resample_metrics.get("resolved_plan", {})
            _write_event(paths.events, "tuning", _stage_event_payload(manifest, "resampling", {"selection": manifest["selection"]["resampling"]}, checkpoint="resampled.zarr" if not resampling_is_final else "output", terminal_status="running"))
            _write_manifest(paths.manifest, manifest)
            current = resample_output
            if config.general.cleanup_intermediate:
                shutil.rmtree(paths.converted)
                manifest["stages"]["conversion"]["status"] = "validated_and_cleaned"
                _write_manifest(paths.manifest, manifest)

        update_lifecycle()
        if not plan.finalization_required:
            rechunk_metrics = {
                "skipped": True,
                "reason": (
                    "resampling_wrote_final_output_layout"
                    if plan.needs_resample
                    else "conversion_wrote_final_output_layout"
                ),
                "avoided_full_store_reads": 1,
                "avoided_full_store_writes": 1,
            }
            _apply_backend_metrics(manifest, rechunk_metrics, requested_backend)
            manifest["stages"]["finalization"] = {
                "status": "not_required_direct_layout",
                "metrics": rechunk_metrics,
            }
        else:
            chunking = config.chunking
            rechunk_config = RechunkConfig(
                input=current,
                output=Path(config.general.output).expanduser().resolve(),
                strategy=chunking.strategy,
                target_mib=chunking.target_mib,
                custom_chunks=chunking.custom_chunks,
                compression=(
                    config.compression.profile
                    if config.operations.recompress
                    else "none"
                ),
                workers=chunking.workers,
                tune_budget_seconds=chunking.tune_budget,
                validate=config.validate,
                backend=getattr(config, "backend", "python"),
                recompress=config.operations.recompress,
                temporary_dir=paths.root,
                compression_codec=config.compression.codec,
                compression_level=config.compression.level,
                compression_shuffle=config.compression.shuffle,
                compression_objective=config.compression.objective,
                compression_tune_budget=config.compression.tune_budget,
                resource_budget=effective_resource_budget(resources),
                storage_overrides={
                    "source": config.general.source_storage,
                    "temporary": config.general.temporary_storage,
                    "output": config.general.output_storage,
                },
            )
            if progress:
                print(
                    f"一条龙阶段 {len(physical_stages)}/{len(physical_stages)}："
                    "执行布局兼容性最终化"
                )
            current_stage = "finalization"
            with _stage_progress(
                paths.events,
                manifest,
                "finalization",
                (paths.root, Path(config.general.output).expanduser().resolve()),
                checkpoint="output",
                logical_total=_logical_output_bytes(plan),
                progress_callback=progress_callback,
            ) as stage_report:
                rechunk_metrics = run_rechunk(
                    rechunk_config,
                    cancel_event=cancel_event,
                    progress_callback=_stage_progress_callback(
                        stage_report,
                        _logical_output_bytes(plan),
                    ),
                )
            _apply_backend_metrics(manifest, rechunk_metrics, requested_backend)
            manifest["stages"]["finalization"] = {
                "status": "published_and_validated",
                "metrics": rechunk_metrics,
            }
            _apply_runtime_compression(manifest, rechunk_metrics)
            if config.general.cleanup_intermediate:
                shutil.rmtree(current, ignore_errors=False)
                manifest["stages"]["resampling" if plan.needs_resample else "conversion"]["status"] = "validated_and_cleaned"
            update_lifecycle()
        manifest["candidate_trials"]["finalization"] = rechunk_metrics.get("worker_tuning", {})
        manifest["selection"]["finalization"] = {
            "tuning_objective": rechunk_metrics.get("tuning_objective"),
            "worker_tuning": rechunk_metrics.get("worker_tuning", {}),
        }
        manifest["resolved_plans"]["finalization"] = {
            "resource_budget": rechunk_metrics.get("resource_budget"),
            "selected_compression": rechunk_metrics.get("selected_compression"),
        }
        _write_event(paths.events, "tuning", _stage_event_payload(manifest, "finalization", {"selection": manifest["selection"]["finalization"]}, checkpoint="output", terminal_status="running"))
        semantic_output = final_target if plan.finalization_required else current
        semantic_validation = _semantic_validation(
            semantic_output,
            config,
            progress=progress,
        )
        if not plan.finalization_required and (conversion_is_final or resampling_is_final):
            publish_staging(
                current,
                final_target,
                "pipeline",
                overwrite=config.general.overwrite,
                require_zarr_v3=True,
            )
            current = final_target
            stage_name = "resampling" if resampling_is_final else "conversion"
            manifest["stages"][stage_name]["status"] = "published_as_final"
        manifest["semantic_validation"] = semantic_validation
        final_logical_bytes = _logical_output_bytes(plan)
        temporary_logical_writes = 0
        if not conversion_is_final:
            temporary_logical_writes += int(conversion_metrics.get("logical_bytes", 0))
        if plan.needs_resample:
            temporary_logical_writes += int(
                resample_metrics.get("intermediate_logical_bytes", 0)
            )
            if not resampling_is_final:
                temporary_logical_writes += int(resample_metrics.get("logical_bytes", 0))
        if plan.finalization_required:
            # The rechunk engine writes one full logical intermediate before
            # publishing its final staging store.
            temporary_logical_writes += int(rechunk_metrics.get("logical_bytes", 0))
        logical_io = {
            "final_output_bytes": final_logical_bytes,
            "temporary_write_bytes": temporary_logical_writes,
            "total_write_bytes": temporary_logical_writes + final_logical_bytes,
            "write_amplification": (
                (temporary_logical_writes + final_logical_bytes)
                / max(1, final_logical_bytes)
            ),
            "avoided_finalization_read_bytes": (
                final_logical_bytes if not plan.finalization_required else 0
            ),
            "avoided_finalization_write_bytes": (
                final_logical_bytes if not plan.finalization_required else 0
            ),
        }
        manifest["logical_io"] = logical_io
        manifest["status"] = "succeeded"
        manifest["elapsed"] = time.perf_counter() - started
        lifecycle.finish("completed")
        manifest["worker_lifecycle"] = lifecycle.to_dict()
        _write_event(paths.events, "finished", _stage_event_payload(manifest, "terminal", {"status": "succeeded", "elapsed": manifest["elapsed"]}, checkpoint="output", terminal_status="succeeded"))
        _write_manifest(paths.manifest, manifest)
        if progress:
            print(f"一条龙处理完成：{config.general.output}")
        return {
            "output": str(Path(config.general.output).expanduser().resolve()),
            "elapsed": manifest["elapsed"],
            "needs_resample": plan.needs_resample,
            "requested_operations": requested_operations,
            "operation_decisions": operation_decisions,
            "physical_stages": physical_stages,
            "manifest": str(paths.manifest),
            "conversion": conversion_metrics,
            "resampling": resample_metrics,
            "mathematical_validation": validation_metrics,
            "semantic_validation": semantic_validation,
            "finalization": rechunk_metrics,
            "logical_io": logical_io,
        }
    except Exception as exc:
        cancelled = cancel_event is not None and cancel_event.is_set()
        terminal_status = "cancelled" if cancelled else "failed"
        lifecycle.finish(terminal_status)
        manifest["worker_lifecycle"] = lifecycle.to_dict()
        manifest["status"] = terminal_status
        manifest["failed_stage"] = current_stage
        manifest["elapsed"] = time.perf_counter() - started
        _write_event(
            paths.events,
            "finished",
            _stage_event_payload(
                manifest,
                current_stage,
                {"status": terminal_status, "error": str(exc)},
                checkpoint=None,
                terminal_status=terminal_status,
            ),
        )
        manifest["error"] = str(exc)
        _write_manifest(paths.manifest, manifest)
        if isinstance(exc, PipelineExecutionError):
            raise
        raise PipelineExecutionError(
            f"一条龙处理失败；任务临时目录保留用于排查：{paths.root}\n{exc}"
        ) from exc
