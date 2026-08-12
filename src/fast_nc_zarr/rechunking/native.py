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
    """Explicit single-array P3 execution plan for the Rust backend."""

    source: Path
    target: Path
    array_path: str
    target_chunks: tuple[int, ...]
    expected_dtype: str = "float32"
    requested_workers: int = 1
    worker_ceiling: int = 1
    memory_budget_bytes: int = 0
    codec_concurrent_target: int = 1

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


def _validate_staged_output(
    source_info,
    staged: Path,
    target_chunks: tuple[int, ...],
    *,
    validate: bool,
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
                target_chunks if not source_variable.is_coord else source_variable.chunks
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
        if tuple(map(repr, source_variable.compressors)) != tuple(
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
    """Execute an explicit P3 plan through staging and atomic publication."""

    source = plan.source.expanduser().resolve()
    target = plan.target.expanduser().resolve(strict=False)
    if source == target:
        raise ValueError("输入和输出不能是同一个目录")
    if source in target.parents or target in source.parents:
        raise ValueError("输入和输出 Zarr 不能相互嵌套")
    if not source.is_dir():
        raise ValueError(f"输入 Zarr 目录不存在: {source}")

    resolved = resolve_backend(requested_backend, "zarr.rechunk_f32")
    if resolved != "rust":
        raise BackendUnavailableError(
            "Rust rechunk operation is unavailable; the Python caller must use its normal backend."
        )
    if cancel_event is not None and cancel_event.is_set():
        from .engine import RechunkExecutionError

        raise RechunkExecutionError("任务已取消")

    target = validate_publish_target(
        target,
        overwrite=overwrite,
        operation="Rust重分块",
        require_zarr_v3=True,
    )
    preflight_writable(target.parent, "Rust重分块输出")
    staging = make_staging_path(target, "rechunk-rust")
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
    )

    try:
        if cancel_event is not None and cancel_event.is_set():
            from .engine import RechunkExecutionError

            raise RechunkExecutionError("任务已取消")
        native = importlib.import_module("fast_nc_zarr._native")
        metrics = json.loads(native.rechunk_f32_json(execution_plan.to_json()))
        if cancel_event is not None and cancel_event.is_set():
            from .engine import RechunkExecutionError

            raise RechunkExecutionError("任务已取消")
        if source_info is not None:
            _validate_staged_output(
                source_info,
                staging,
                tuple(int(value) for value in plan.target_chunks),
                validate=validate,
            )
        publish_staging(
            staging,
            target,
            "rechunk-rust",
            overwrite=overwrite,
            require_zarr_v3=True,
        )
        elapsed = time.perf_counter() - started
        logical_bytes = int(metrics.get("logical_bytes", 0))
        metrics.update(
            {
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
                "selected_compression": {"profile": "none", "codec": "none"},
            }
        )
        return metrics
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise



def run_rust_rechunk_for_config(
    config,
    info,
    plan=None,
    *,
    cancel_event=None,
) -> dict[str, object]:
    """Resolve a supported float32 variable and execute one P3 plan."""

    reference = next(
        (variable for variable in info.data_variables if variable.ndim == 3),
        None,
    )
    if reference is None:
        raise ValueError("Rust P3 rechunk requires a three-dimensional data variable")
    if str(reference.dtype) != "float32":
        raise ValueError(
            f"Rust P3 rechunk currently supports float32 only, got {reference.dtype}"
        )
    if len(info.data_variables) != 1:
        raise ValueError("Rust P3 rechunk requires exactly one data variable")
    if tuple(reference.dims) != ("time", "lat", "lon"):
        raise ValueError("Rust P3 rechunk requires data dimensions (time, lat, lon)")
    if getattr(config, "compression", "none") != "none":
        raise ValueError("Rust P3 rechunk does not yet support codec changes")

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
            expected_dtype="float32",
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
