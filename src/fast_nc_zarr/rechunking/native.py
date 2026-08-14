from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Literal

from .._backend import BackendUnavailableError, resolve_backend
from ..publication import (
    make_staging_path,
    preflight_writable,
    publish_staging,
    validate_publish_target,
)


BackendName = Literal["auto", "python", "rust"]


@dataclass(frozen=True, slots=True)
class RustRechunkPlan:
    """Explicit single-array execution plan for the Rust backend."""

    source: Path
    target: Path
    array_path: str
    target_chunks: tuple[int, ...]
    expected_dtype: str = "float32"
    requested_workers: int = 1
    worker_ceiling: int = 1
    memory_budget_bytes: int = 0
    codec_concurrent_target: int = 1
    codec: str = "none"
    codec_level: int | None = None
    codec_shuffle: str = "auto"
    cancellation_file: Path | None = None
    progress_file: Path | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "source": str(self.source.expanduser().resolve()),
                "target": str(self.target.expanduser().resolve()),
                "array_path": self.array_path,
                "target_chunks": [int(value) for value in self.target_chunks],
                "expected_dtype": self.expected_dtype,
                "requested_workers": max(1, int(self.requested_workers)),
                "worker_ceiling": max(1, int(self.worker_ceiling)),
                "memory_budget_bytes": max(0, int(self.memory_budget_bytes)),
                "codec_concurrent_target": max(1, int(self.codec_concurrent_target)),
                "codec": str(self.codec),
                "codec_level": None if self.codec_level is None else int(self.codec_level),
                "codec_shuffle": str(self.codec_shuffle),
                "cancellation_file": (
                    None
                    if self.cancellation_file is None
                    else str(self.cancellation_file.expanduser().resolve())
                ),
                "progress_file": (
                    None
                    if self.progress_file is None
                    else str(self.progress_file.expanduser().resolve())
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )

@dataclass(frozen=True, slots=True)
class RustMultiRechunkVariablePlan:
    array_path: str
    target_chunks: tuple[int, ...]
    expected_dtype: str


@dataclass(frozen=True, slots=True)
class RustMultiRechunkPlan:
    """Explicit multi-variable Zarr plan for the Rust backend."""

    source: Path
    target: Path
    variables: tuple[RustMultiRechunkVariablePlan, ...]
    requested_workers: int = 1
    worker_ceiling: int = 1
    memory_budget_bytes: int = 0
    codec_concurrent_target: int = 1
    codec: str = "none"
    codec_level: int | None = None
    codec_shuffle: str = "auto"
    cancellation_file: Path | None = None
    progress_file: Path | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "source": str(self.source.expanduser().resolve()),
                "target": str(self.target.expanduser().resolve()),
                "variables": [
                    {
                        "array_path": item.array_path,
                        "target_chunks": [int(value) for value in item.target_chunks],
                        "expected_dtype": item.expected_dtype,
                    }
                    for item in self.variables
                ],
                "requested_workers": max(1, int(self.requested_workers)),
                "worker_ceiling": max(1, int(self.worker_ceiling)),
                "memory_budget_bytes": max(0, int(self.memory_budget_bytes)),
                "codec_concurrent_target": max(1, int(self.codec_concurrent_target)),
                "codec": str(self.codec),
                "codec_level": None if self.codec_level is None else int(self.codec_level),
                "codec_shuffle": str(self.codec_shuffle),
                "cancellation_file": (
                    None
                    if self.cancellation_file is None
                    else str(self.cancellation_file.expanduser().resolve())
                ),
                "progress_file": (
                    None
                    if self.progress_file is None
                    else str(self.progress_file.expanduser().resolve())
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _metadata_signature(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    except (TypeError, ValueError):
        return repr(value)


def _codec_matches_request(
    compressors: tuple[Any, ...],
    codec: str,
    level: int | None,
    shuffle: str,
) -> bool:
    text = " ".join(repr(item) for item in compressors).lower()
    if codec == "zstd":
        matched = "zstdcodec" in text and "blosc" not in text
        if level is not None:
            matched = matched and f"level={int(level)}" in text
        return matched
    if codec == "gzip":
        matched = "gzipcodec" in text
        if level is not None:
            matched = matched and f"level={int(level)}" in text
        return matched
    if codec.startswith("blosc-"):
        cname = codec.removeprefix("blosc-")
        matched = "blosccodec" in text and f"cname='{cname}'" in text
        if level is not None:
            matched = matched and f"clevel={int(level)}" in text
        if shuffle not in {"", "auto"}:
            matched = matched and f"shuffle='{shuffle}'" in text
        return matched
    return False


def _validate_multi_staged_output(
    source_info,
    staged: Path,
    variables: tuple[RustMultiRechunkVariablePlan, ...],
    *,
    validate: bool,
) -> None:
    from .inspection import inspect_store

    output_info = inspect_store(staged)
    if source_info.dimensions != output_info.dimensions:
        raise ValueError(
            f"Rust多变量输出维度发生变化：输入 {source_info.dimensions}，输出 {output_info.dimensions}"
        )
    if _metadata_signature(source_info.attrs) != _metadata_signature(output_info.attrs):
        raise ValueError("Rust多变量输出根属性发生变化")
    source_vars = {item.name: item for item in source_info.variables}
    output_vars = {item.name: item for item in output_info.variables}
    if source_vars.keys() != output_vars.keys():
        raise ValueError("Rust多变量输出变量集合与输入不一致")
    requested_names = {
        item.array_path.strip("/").split("/")[-1] for item in variables
    }
    expected_chunks = {
        item.array_path.strip("/").split("/")[-1]: item.target_chunks for item in variables
    }
    if not requested_names.issubset(source_vars):
        missing = sorted(requested_names - source_vars.keys())
        raise ValueError("Rust多变量计划缺少输入变量：" + ", ".join(missing))
    for name, source_variable in source_vars.items():
        output_variable = output_vars[name]
        if (
            source_variable.dims != output_variable.dims
            or source_variable.shape != output_variable.shape
            or source_variable.dtype != output_variable.dtype
        ):
            raise ValueError(f"Rust输出变量 {name} 的结构发生变化")
        if source_variable.ndim:
            expected = expected_chunks.get(name, source_variable.chunks)
            if output_variable.chunks != expected:
                raise ValueError(
                    f"Rust输出变量 {name} 的 chunks 不符合计划：期望 {expected}，实际 {output_variable.chunks}"
                )
        if _metadata_signature(source_variable.attrs) != _metadata_signature(
            output_variable.attrs
        ):
            raise ValueError(f"Rust输出变量 {name} 的属性发生变化")
        if tuple(map(repr, source_variable.compressors)) != tuple(
            map(repr, output_variable.compressors)
        ):
            raise ValueError(f"Rust输出变量 {name} 的 codec 发生变化")


def run_rust_multi_rechunk(
    plan: RustMultiRechunkPlan,
    *,
    requested_backend: BackendName = "rust",
    source_info=None,
    overwrite: bool = False,
    validate: bool = True,
    cancel_event=None,
) -> dict[str, object]:
    """Execute a multi-variable Rust plan with atomic publication."""

    source = plan.source.expanduser().resolve()
    target = plan.target.expanduser().resolve(strict=False)
    if source == target:
        raise ValueError("输入和输出不能是同一个目录")
    if source in target.parents or target in source.parents:
        raise ValueError("输入和输出 Zarr 不能相互嵌套")
    if not source.is_dir():
        raise ValueError(f"输入 Zarr 目录不存在: {source}")
    if not plan.variables:
        raise BackendUnavailableError("Rust 多变量重分块至少需要一个数据变量")
    if any(item.expected_dtype not in {"float32", "float64"} for item in plan.variables):
        raise BackendUnavailableError("Rust 多变量重分块当前只支持 float32/float64")
    if plan.codec not in {"", "none"}:
        raise BackendUnavailableError("Rust 多变量重分块保留源 codec，不执行新的压缩配置")
    resolved = resolve_backend(requested_backend, "zarr.rechunk_multi")
    if resolved != "rust":
        raise BackendUnavailableError(
            "Rust multi-variable rechunk operation is unavailable; the Python caller must use its normal backend."
        )
    if cancel_event is not None and cancel_event.is_set():
        from .engine import RechunkExecutionError

        raise RechunkExecutionError("任务已取消")
    target = validate_publish_target(
        target, overwrite=overwrite, operation="Rust多变量重分块", require_zarr_v3=True
    )
    preflight_writable(target.parent, "Rust多变量重分块输出")
    staging = make_staging_path(target, "rechunk-rust-multi")
    cancellation_file = (
        plan.cancellation_file
        if plan.cancellation_file is not None
        else make_staging_path(target, "rechunk-rust-multi-cancel")
        if cancel_event is not None
        else None
    )
    progress_file = (
        plan.progress_file
        if plan.progress_file is not None
        else make_staging_path(target, "rechunk-rust-multi-progress")
        if cancel_event is not None
        else None
    )
    remove_cancellation_file = plan.cancellation_file is None
    remove_progress_file = plan.progress_file is None
    started = time.perf_counter()
    execution_plan = RustMultiRechunkPlan(
        source=source,
        target=staging,
        variables=plan.variables,
        requested_workers=plan.requested_workers,
        worker_ceiling=plan.worker_ceiling,
        memory_budget_bytes=plan.memory_budget_bytes,
        codec_concurrent_target=plan.codec_concurrent_target,
        codec=plan.codec,
        codec_level=plan.codec_level,
        codec_shuffle=plan.codec_shuffle,
        cancellation_file=cancellation_file,
        progress_file=progress_file,
    )
    import threading

    watch_stop = threading.Event()
    cancel_watcher = None
    if cancel_event is not None and cancellation_file is not None:

        def _watch_cancel() -> None:
            while not watch_stop.wait(0.05):
                if cancel_event.is_set():
                    cancellation_file.touch(exist_ok=True)
                    return

        cancel_watcher = threading.Thread(target=_watch_cancel, daemon=True)
        cancel_watcher.start()

    def _cleanup() -> None:
        watch_stop.set()
        if cancel_watcher is not None:
            cancel_watcher.join(timeout=1.0)
        if remove_cancellation_file and cancellation_file is not None:
            cancellation_file.unlink(missing_ok=True)
        if remove_progress_file and progress_file is not None:
            progress_file.unlink(missing_ok=True)

    try:
        native = importlib.import_module("fast_nc_zarr._native")
        native_rechunk = getattr(native, "rechunk_multi_json", None)
        if native_rechunk is None:
            raise BackendUnavailableError("native extension lacks rechunk_multi_json")
        metrics = json.loads(native_rechunk(execution_plan.to_json()))
        if cancel_event is not None and cancel_event.is_set():
            from .engine import RechunkExecutionError

            raise RechunkExecutionError("任务已取消")
        if source_info is not None:
            _validate_multi_staged_output(
                source_info,
                staging,
                plan.variables,
                validate=validate,
            )
        if cancel_event is not None and cancel_event.is_set():
            from .engine import RechunkExecutionError

            raise RechunkExecutionError("任务已取消")
        publish_staging(staging, target, "rechunk-rust-multi", overwrite=overwrite, require_zarr_v3=True)
        elapsed = time.perf_counter() - started
        logical_bytes = int(metrics.get("logical_bytes", 0))
        for variable in metrics.get("variables", []):
            if isinstance(variable, dict):
                variable["output"] = str(target)
        metrics.update(
            {
                "backend": "rust",
                "backend_fallback": False,
                "backend_fallback_reason": None,
                "protocol_version": 1,
                "elapsed": elapsed,
                "logical_bytes": logical_bytes,
                "physical_bytes": _directory_size(target),
                "throughput_mib_s": logical_bytes / 1024**2 / max(elapsed, 1e-9),
                "output": str(target),
                "temporary_dir": str(target.parent),
                "requested_workers": plan.requested_workers,
                "worker_ceiling": plan.worker_ceiling,
                "memory_budget_bytes": plan.memory_budget_bytes,
                "worker_tuning": {},
                "tuning_objective": "balanced",
                "selected_compression": {
                    "profile": "none",
                    "codec": "none",
                    "level": None,
                    "shuffle": "auto",
                },
            }
        )
        _cleanup()
        return metrics
    except Exception:
        _cleanup()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_staged_output(
    source_info,
    staged: Path,
    target_chunks: tuple[int, ...],
    *,
    validate: bool,
    array_path: str,
    requested_codec: str,
    requested_codec_level: int | None,
    requested_codec_shuffle: str,
) -> None:
    from .inspection import inspect_store

    output_info = inspect_store(staged)
    if source_info.dimensions != output_info.dimensions:
        raise ValueError(
            f"Rust输出维度发生变化：输入 {source_info.dimensions}，输出 {output_info.dimensions}"
        )
    if _metadata_signature(source_info.attrs) != _metadata_signature(output_info.attrs):
        raise ValueError("Rust输出根属性发生变化")
    source_vars = {item.name: item for item in source_info.variables}
    output_vars = {item.name: item for item in output_info.variables}
    if source_vars.keys() != output_vars.keys():
        raise ValueError("Rust输出变量集合与输入不一致")
    target_name = array_path.strip("/").split("/")[-1]
    for name, source_variable in source_vars.items():
        output_variable = output_vars[name]
        if (
            source_variable.dims != output_variable.dims
            or source_variable.shape != output_variable.shape
            or source_variable.dtype != output_variable.dtype
        ):
            raise ValueError(f"Rust输出变量 {name} 的结构发生变化")
        if source_variable.ndim:
            expected_chunks = (
                target_chunks
                if not source_variable.is_coord and name == target_name
                else source_variable.chunks
            )
            if output_variable.chunks != expected_chunks:
                raise ValueError(
                    f"Rust输出变量 {name} 的 chunks 不符合计划："
                    f"期望 {expected_chunks}，实际 {output_variable.chunks}"
                )
        if _metadata_signature(source_variable.attrs) != _metadata_signature(
            output_variable.attrs
        ):
            raise ValueError(f"Rust输出变量 {name} 的属性发生变化")
        if name == target_name and requested_codec not in {"", "none"}:
            if not _codec_matches_request(
                output_variable.compressors,
                requested_codec,
                requested_codec_level,
                requested_codec_shuffle,
            ):
                raise ValueError(
                    f"Rust输出变量 {name} 未使用请求的 codec {requested_codec}"
                )
        elif tuple(map(repr, source_variable.compressors)) != tuple(
            map(repr, output_variable.compressors)
        ):
            raise ValueError(f"Rust输出变量 {name} 的 codec 发生变化")

def run_rust_rechunk(
    plan: RustRechunkPlan,
    *,
    requested_backend: BackendName = "rust",
    source_info=None,
    overwrite: bool = False,
    validate: bool = True,
    cancel_event=None,
) -> dict[str, object]:
    """Execute a Rust plan with cooperative cancellation and atomic publication."""

    source = plan.source.expanduser().resolve()
    target = plan.target.expanduser().resolve(strict=False)
    if source == target:
        raise ValueError("输入和输出不能是同一个目录")
    if source in target.parents or target in source.parents:
        raise ValueError("输入和输出 Zarr 不能相互嵌套")
    if not source.is_dir():
        raise ValueError(f"输入 Zarr 目录不存在: {source}")
    dtype = str(plan.expected_dtype)
    if dtype not in {"float32", "float64"}:
        raise BackendUnavailableError(f"Rust rechunk currently supports float32/float64 only, got {dtype}")
    native_dtype = "f32" if dtype == "float32" else "f64"
    if dtype == "float64" and plan.codec not in {"", "none"}:
        raise BackendUnavailableError("Rust float64 rechunk preserves the source codec and does not apply a new codec")
    operation = f"zarr.rechunk_{native_dtype}"
    resolved = resolve_backend(requested_backend, operation)
    if plan.cancellation_file is not None or cancel_event is not None:
        resolve_backend(requested_backend, f"zarr.rechunk_{native_dtype}_cancel")
    if resolved != "rust":
        raise BackendUnavailableError(
            "Rust rechunk operation is unavailable; the Python caller must use its normal backend."
        )
    if cancel_event is not None and cancel_event.is_set():
        from .engine import RechunkExecutionError
        raise RechunkExecutionError("任务已取消")
    target = validate_publish_target(
        target, overwrite=overwrite, operation="Rust重分块", require_zarr_v3=True
    )
    preflight_writable(target.parent, "Rust重分块输出")
    staging = make_staging_path(target, "rechunk-rust")
    cancellation_file = (
        plan.cancellation_file
        if plan.cancellation_file is not None
        else make_staging_path(target, "rechunk-rust-cancel")
        if cancel_event is not None
        else None
    )
    progress_file = (
        plan.progress_file
        if plan.progress_file is not None
        else make_staging_path(target, "rechunk-rust-progress")
        if cancel_event is not None
        else None
    )
    remove_cancellation_file = plan.cancellation_file is None
    remove_progress_file = plan.progress_file is None
    started = time.perf_counter()
    execution_plan = RustRechunkPlan(
        source=source,
        target=staging,
        array_path=plan.array_path,
        target_chunks=plan.target_chunks,
        expected_dtype=plan.expected_dtype,
        requested_workers=plan.requested_workers,
        worker_ceiling=plan.worker_ceiling,
        memory_budget_bytes=plan.memory_budget_bytes,
        codec_concurrent_target=plan.codec_concurrent_target,
        codec=plan.codec,
        codec_level=plan.codec_level,
        codec_shuffle=plan.codec_shuffle,
        cancellation_file=cancellation_file,
        progress_file=progress_file,
    )
    import threading
    watch_stop = threading.Event()
    cancel_watcher = None
    if cancel_event is not None and cancellation_file is not None:
        def _watch_cancel() -> None:
            while not watch_stop.wait(0.05):
                if cancel_event.is_set():
                    cancellation_file.touch(exist_ok=True)
                    return
        cancel_watcher = threading.Thread(target=_watch_cancel, daemon=True)
        cancel_watcher.start()

    def _cleanup() -> None:
        watch_stop.set()
        if cancel_watcher is not None:
            cancel_watcher.join(timeout=1.0)
        if remove_cancellation_file and cancellation_file is not None:
            cancellation_file.unlink(missing_ok=True)
        if remove_progress_file and progress_file is not None:
            progress_file.unlink(missing_ok=True)

    try:
        native = importlib.import_module("fast_nc_zarr._native")
        native_rechunk = getattr(native, f"rechunk_{native_dtype}_json", None)
        if native_rechunk is None:
            raise BackendUnavailableError(f"native extension lacks rechunk_{native_dtype}_json")
        metrics = json.loads(native_rechunk(execution_plan.to_json()))
        if cancel_event is not None and cancel_event.is_set():
            from .engine import RechunkExecutionError
            raise RechunkExecutionError("任务已取消")
        if source_info is not None:
            _validate_staged_output(
                source_info,
                staging,
                tuple(int(value) for value in plan.target_chunks),
                validate=validate,
                array_path=plan.array_path,
                requested_codec=plan.codec,
                requested_codec_level=plan.codec_level,
                requested_codec_shuffle=plan.codec_shuffle,
            )
        if cancel_event is not None and cancel_event.is_set():
            from .engine import RechunkExecutionError
            raise RechunkExecutionError("任务已取消")
        publish_staging(
            staging, target, "rechunk-rust", overwrite=overwrite, require_zarr_v3=True
        )
        elapsed = time.perf_counter() - started
        logical_bytes = int(metrics.get("logical_bytes", 0))
        metrics.update(
            {
                "backend": "rust",
                "backend_fallback": False,
                "backend_fallback_reason": None,
                "protocol_version": 1,
                "elapsed": elapsed,
                "logical_bytes": logical_bytes,
                "physical_bytes": _directory_size(target),
                "throughput_mib_s": logical_bytes / 1024**2 / max(elapsed, 1e-9),
                "output": str(target),
                "temporary_dir": str(target.parent),
                "requested_workers": plan.requested_workers,
                "worker_ceiling": plan.worker_ceiling,
                "memory_budget_bytes": plan.memory_budget_bytes,
                "worker_tuning": {},
                "tuning_objective": "balanced",
                "selected_compression": {
                    "profile": "rust" if plan.codec not in {"", "none"} else "none",
                    "codec": plan.codec,
                    "level": plan.codec_level,
                    "shuffle": plan.codec_shuffle,
                },
            }
        )
        _cleanup()
        return metrics
    except Exception:
        _cleanup()
        shutil.rmtree(staging, ignore_errors=True)
        raise



def run_rust_multi_rechunk_for_config(
    config,
    info,
    plan=None,
    *,
    compression=None,
    cancel_event=None,
) -> dict[str, object]:
    """Resolve all compatible data variables and execute the Rust multi plan."""

    variables = tuple(info.data_variables)
    if len(variables) < 2:
        raise BackendUnavailableError("Rust 多变量重分块至少需要两个数据变量")
    if any(variable.ndim != 3 for variable in variables):
        raise BackendUnavailableError("Rust 多变量重分块要求所有数据变量都是三维数组")
    if any(tuple(variable.dims) != ("time", "lat", "lon") for variable in variables):
        raise BackendUnavailableError("Rust 多变量重分块要求数据维度为 (time, lat, lon)")
    if any(str(variable.dtype) not in {"float32", "float64"} for variable in variables):
        raise BackendUnavailableError("Rust 多变量重分块当前只支持 float32/float64")
    if plan is None:
        reference = variables[0]
        base_chunks = tuple(int(value) for value in reference.chunks)
        if getattr(config, "rechunk", True):
            preview_chunks = getattr(config, "custom_chunks", None)
            if preview_chunks is not None:
                base_chunks = tuple(
                    min(int(chunk), int(size))
                    for chunk, size in zip(preview_chunks, reference.shape)
                )
        target_chunks = {
            variable.name: tuple(
                min(int(chunk), int(size))
                for chunk, size in zip(base_chunks, variable.shape)
            )
            for variable in variables
        }
    else:
        target_chunks = {
            variable.name: tuple(int(value) for value in plan.chunks_for(variable))
            for variable in variables
        }
    if compression is None:
        from .compression import make_compression_plan

        compression = make_compression_plan(
            getattr(config, "compression", "none"),
            codec=getattr(config, "compression_codec", None),
            level=getattr(config, "compression_level", None),
            shuffle=getattr(config, "compression_shuffle", "auto"),
        )
    if getattr(compression, "profile", None) == "auto":
        raise BackendUnavailableError("Rust 多变量后端不执行自动压缩候选调优")
    if bool(getattr(compression, "enabled", False)):
        raise BackendUnavailableError("Rust 多变量重分块保留源 codec，不执行新的压缩配置")
    budget = getattr(config, "resource_budget", None)
    if budget is None:
        from ..system import effective_resource_budget

        budget = effective_resource_budget(
            source=Path(config.input),
            output=Path(config.output).parent,
            requested=(None if config.workers == "auto" else max(1, int(config.workers))),
        )
    requested_workers = (
        int(budget.worker_ceiling)
        if config.workers == "auto"
        else min(max(1, int(config.workers)), int(budget.worker_ceiling))
    )
    codec_workers = max(1, min(requested_workers, 2))
    return run_rust_multi_rechunk(
        RustMultiRechunkPlan(
            source=Path(config.input),
            target=Path(config.output),
            variables=tuple(
                RustMultiRechunkVariablePlan(
                    array_path=f"/{variable.name}",
                    target_chunks=target_chunks[variable.name],
                    expected_dtype=str(variable.dtype),
                )
                for variable in variables
            ),
            requested_workers=requested_workers,
            worker_ceiling=int(budget.worker_ceiling),
            memory_budget_bytes=int(budget.memory_budget_bytes),
            codec_concurrent_target=codec_workers,
        ),
        requested_backend="rust",
        source_info=info,
        overwrite=bool(config.overwrite),
        validate=bool(config.validate),
        cancel_event=cancel_event,
    )


def run_rust_rechunk_for_config(
    config,
    info,
    plan=None,
    *,
    compression=None,
    cancel_event=None,
) -> dict[str, object]:
    """Resolve one three-dimensional float32/float64 data variable and execute Rust."""
    if len(info.data_variables) > 1:
        return run_rust_multi_rechunk_for_config(
            config,
            info,
            plan,
            compression=compression,
            cancel_event=cancel_event,
        )
    reference = next(
        (variable for variable in info.data_variables if variable.ndim == 3), None
    )
    if reference is None:
        raise BackendUnavailableError("Rust P3 重分块要求三维数据变量")
    expected_dtype = str(reference.dtype)
    if expected_dtype not in {"float32", "float64"}:
        raise BackendUnavailableError(
            f"Rust P3 rechunk currently supports float32/float64 only, got {reference.dtype}"
        )
    if tuple(reference.dims) != ("time", "lat", "lon"):
        raise BackendUnavailableError("Rust P3 重分块要求数据维度为 (time, lat, lon)")

    if plan is None:
        target_chunks = tuple(int(value) for value in reference.chunks)
        if getattr(config, "rechunk", True):
            preview_chunks = getattr(config, "custom_chunks", None)
            if preview_chunks is not None:
                target_chunks = tuple(
                    min(int(chunk), int(size))
                    for chunk, size in zip(preview_chunks, reference.shape)
                )
    else:
        target_chunks = tuple(int(value) for value in plan.chunks_for(reference))

    if compression is None:
        from .compression import make_compression_plan

        compression = make_compression_plan(
            getattr(config, "compression", "none"),
            codec=getattr(config, "compression_codec", None),
            level=getattr(config, "compression_level", None),
            shuffle=getattr(config, "compression_shuffle", "auto"),
        )
    if getattr(compression, "profile", None) == "auto":
        raise BackendUnavailableError("Rust 后端不执行自动压缩候选调优")
    compression_enabled = bool(getattr(compression, "enabled", False))
    codec = str(getattr(compression, "codec", "none")) if compression_enabled else "none"
    codec_level = getattr(compression, "level", None) if compression_enabled else None
    codec_shuffle = str(getattr(compression, "shuffle", "auto")) if compression_enabled else "auto"
    if expected_dtype == "float64" and codec not in {"", "none"}:
        raise BackendUnavailableError("Rust float64 rechunk preserves the source codec and does not apply a new codec")

    budget = getattr(config, "resource_budget", None)
    if budget is None:
        from ..system import effective_resource_budget

        budget = effective_resource_budget(
            source=Path(config.input),
            output=Path(config.output).parent,
            requested=(None if config.workers == "auto" else max(1, int(config.workers))),
        )
    requested_workers = (
        int(budget.worker_ceiling)
        if config.workers == "auto"
        else min(max(1, int(config.workers)), int(budget.worker_ceiling))
    )
    codec_workers = max(1, min(requested_workers, 2))
    return run_rust_rechunk(
        RustRechunkPlan(
            source=Path(config.input),
            target=Path(config.output),
            array_path=f"/{reference.name}",
            target_chunks=target_chunks,
            expected_dtype=expected_dtype,
            requested_workers=requested_workers,
            worker_ceiling=int(budget.worker_ceiling),
            memory_budget_bytes=int(budget.memory_budget_bytes),
            codec_concurrent_target=codec_workers,
            codec=codec,
            codec_level=codec_level,
            codec_shuffle=codec_shuffle,
        ),
        requested_backend="rust",
        source_info=info,
        overwrite=bool(config.overwrite),
        validate=bool(config.validate),
        cancel_event=cancel_event,
    )
