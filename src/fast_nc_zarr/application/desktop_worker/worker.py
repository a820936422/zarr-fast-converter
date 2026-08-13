from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, is_dataclass
import io
import json
from pathlib import Path
import re
import sys
from threading import Event as ThreadEvent, Thread
import time
from typing import Any, TextIO

from .protocol import (
    Event,
    ProtocolError,
    Request,
    decode_request,
    error_payload,
    write_event,
)

_PERCENT_PROGRESS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%")
_FRACTION_PROGRESS = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])")


def _safe(value: Any) -> Any:
    if is_dataclass(value):
        return _safe(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "item"):
        return _safe(value.item())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    return value


def _path(payload: dict[str, Any], key: str) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty path")
    return Path(value).expanduser().resolve()


def _optional_path(payload: dict[str, Any], key: str) -> Path | None:
    value = payload.get(key)
    if value in (None, ""):
        return None
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a path or null")
    return Path(value).expanduser().resolve()


def _time_ref(value: Any):
    from ...time_mapping import TimeFieldRef

    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("time rule field must be an object")
    return TimeFieldRef(
        source=str(value.get("source", "")),
        component=str(value.get("component", "")),
        index=int(value.get("index", 0)),
    )


def _time_rule(value: Any):
    from ...time_mapping import TimeRule

    if value in (None, {}):
        return None
    if not isinstance(value, dict):
        raise ValueError("time_rule must be an object")
    return TimeRule(
        full=_time_ref(value.get("full")),
        year=_time_ref(value.get("year")),
        month=_time_ref(value.get("month")),
        day=_time_ref(value.get("day")),
        doy=_time_ref(value.get("doy")),
    )


def _source_config(payload: dict[str, Any]):
    from ..services import SourceInspectionConfig

    dimensions = payload.get("source_dimensions")
    source_dimensions = tuple(str(item) for item in dimensions) if dimensions else None
    fields = payload.get("field_values")
    field_values = tuple(str(item) for item in fields) if fields else None
    return SourceInspectionConfig(
        input_dir=_path(payload, "input_dir"),
        mode=str(payload.get("mode", "auto")),
        recursive=bool(payload.get("recursive", False)),
        engine=str(payload.get("engine", "auto")),
        template=payload.get("template"),
        field_values=field_values,
        source_dimensions=source_dimensions,
        workers=int(payload["workers"]) if payload.get("workers") is not None else None,
        time_rule=_time_rule(payload.get("time_rule")),
        cache_path=_optional_path(payload, "cache_path"),
    )


def _time_inspection_payload(result: Any) -> dict[str, Any]:
    return {
        "input_dir": str(result.input_dir),
        "files": [str(item) for item in result.files],
        "engine": result.engine,
        "dimensions": list(result.dimensions),
        "coordinates": list(result.coordinates),
        "filename_fields": [_safe(item) for item in result.filename_fields],
        "time_dimension": _safe(result.time_dimension),
        "options": [_safe(item) for item in result.options],
        "suggested_rule": _safe(result.suggested_rule),
        "report": result.report,
    }


def _inspection(request: Request, cancel_event: ThreadEvent):
    from ..services import inspect_source, inspect_zarr

    if request.command == "inspect_zarr":
        return inspect_zarr(_path(request.payload, "path"))
    payload = dict(request.payload)
    if "source_path" in payload and "input_dir" not in payload:
        payload["input_dir"] = payload["source_path"]
    return inspect_source(_source_config(payload), cancel_event=cancel_event)


def _inspection_from_payload(payload: dict[str, Any], cancel_event: ThreadEvent):
    from ..services import inspect_source, inspect_temporary_pipeline, inspect_zarr, load_inspection_snapshot

    snapshot_path = _optional_path(payload, "inspection_snapshot_path")
    if snapshot_path is not None:
        return load_inspection_snapshot(snapshot_path, validate_files=bool(payload.get("validate_snapshot", True)))
    kind = str(payload.get("inspection_kind", payload.get("input_kind", "auto")))
    if kind == "temporary":
        return inspect_temporary_pipeline(_path(payload, "path"))
    if kind == "zarr":
        return inspect_zarr(_path(payload, "input_dir"))
    return inspect_source(_source_config(payload), cancel_event=cancel_event)


def _pipeline_config(payload: dict[str, Any]):
    from ..pipeline.models import (
        PipelineChunkingOptions,
        PipelineCompressionOptions,
        PipelineConfig,
        PipelineConversionOptions,
        PipelineGeneralConfig,
        PipelineInput,
        PipelineOperations,
        PipelineResamplingOptions,
    )

    output = _path(payload, "output")
    custom_chunks = payload.get("custom_chunks")
    variables = tuple(str(item) for item in payload.get("variables") or ())
    compression = str(payload.get("compression", "auto"))
    return PipelineConfig(
        general=PipelineGeneralConfig(
            output=output,
            temporary_dir=_optional_path(payload, "temporary_dir"),
            time_start=payload.get("time_start"),
            time_end=payload.get("time_end"),
            lat_min=float(payload.get("lat_min", -90.0)),
            lat_max=float(payload.get("lat_max", 90.0)),
            lon_min=float(payload.get("lon_min", -180.0)),
            lon_max=float(payload.get("lon_max", 180.0)),
            cleanup_intermediate=bool(payload.get("cleanup_intermediate", False)),
            overwrite=bool(payload.get("overwrite", False)),
            source_storage=str(payload.get("source_storage", "auto")),
            temporary_storage=str(payload.get("temporary_storage", "auto")),
            output_storage=str(payload.get("output_storage", "auto")),
        ),
        input=PipelineInput(kind=str(payload.get("input_kind", "auto"))),
        conversion=PipelineConversionOptions(
            variables=variables,
            variable_names={str(key): str(value) for key, value in (payload.get("variable_names") or {}).items()},
            auto_tune=bool(payload.get("auto_tune", True)),
            tune_budget=float(payload.get("tune_budget", 60.0)),
            tuning_objective=str(payload.get("tuning_objective", "balanced")),
            max_workers=int(payload["max_workers"]) if payload.get("max_workers") is not None else None,
            reserve_memory_gib=float(payload.get("reserve_memory_gib", 2.0)),
        ),
        operations=PipelineOperations(
            resample=bool(payload.get("resample", False)),
            rechunk=bool(payload.get("rechunk", False)),
            recompress=bool(payload.get("recompress", False)),
        ),
        resampling=PipelineResamplingOptions(
            resolution=float(payload.get("resolution", 0.1)),
            method=str(payload.get("method", "bilinear")),
            skipna=bool(payload.get("skipna", True)),
            na_thres=float(payload.get("na_thres", 1.0)),
            compute_dtype=str(payload.get("compute_dtype", "source")),
            tile_size=payload.get("tile_size", "auto"),
            time_block=payload.get("time_block", "auto"),
            compute_workers=int(payload.get("compute_workers", 2)),
            space_workers=payload.get("space_workers", "auto"),
            tuning_objective=str(payload.get("tuning_objective", "balanced")),
            tune_budget=float(payload.get("tune_budget", 60.0)),
            before_conditions=str(payload.get("before_conditions", "")),
            before_results=str(payload.get("before_results", "")),
            after_conditions=str(payload.get("after_conditions", "")),
            after_results=str(payload.get("after_results", "")),
            statistics_policy=str(payload.get("statistics_policy", "auto")),
        ),
        chunking=PipelineChunkingOptions(
            strategy=str(payload.get("strategy", "time")),
            target_mib=float(payload.get("target_mib", 128.0)),
            custom_chunks=tuple(int(item) for item in custom_chunks) if custom_chunks else None,
            workers=payload.get("workers", "auto"),
        ),
        compression=PipelineCompressionOptions(
            profile=compression,
            codec=payload.get("compression_codec"),
            level=int(payload["compression_level"]) if payload.get("compression_level") is not None else None,
            shuffle=str(payload.get("compression_shuffle", "auto")),
            objective=str(payload.get("compression_objective", "balanced")),
            tune_budget=float(payload.get("compression_tune_budget", 60.0)),
        ),
        backend=str(payload.get("backend", "python")),
        validate=bool(payload.get("validate", True)),
        semantic_constraints=dict(payload.get("semantic_constraints") or {}),
    )


class _EventSink:
    def __init__(self, output: TextIO, request: Request) -> None:
        self.output = output
        self.request = request
        self.sequence = 0

    def emit(self, event: str, payload: dict[str, Any] | None = None, *, stage: str | None = None) -> None:
        write_event(
            self.output,
            Event(
                request_id=self.request.request_id,
                task_id=self.request.task_id,
                sequence=self.sequence,
                event=event,
                stage=stage,
                payload=_safe(payload or {}),
            ),
        )
        self.sequence += 1


class _OutputStream(io.TextIOBase):
    def __init__(self, sink: _EventSink, *, stream_name: str) -> None:
        self.sink = sink
        self.stream_name = stream_name
        self.buffer = ""

    def write(self, value: str) -> int:
        self.buffer += value
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._line(line)
        return len(value)

    def flush(self) -> None:
        if self.buffer:
            self._line(self.buffer)
            self.buffer = ""

    def _line(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        self.sink.emit("log", {"stream": self.stream_name, "message": text}, stage="execution")
        percentages = _PERCENT_PROGRESS.findall(text)
        fractions = _FRACTION_PROGRESS.findall(text)
        if percentages and ("进度" in text or "Completed" in text or "[" in text):
            self.sink.emit(
                "progress",
                {"completed": min(1000, max(0, int(round(float(percentages[-1]) * 10)))), "total": 1000, "message": text},
                stage="execution",
            )
        elif fractions:
            completed, total = map(int, fractions[-1])
            if total > 0 and 0 <= completed <= total:
                self.sink.emit("progress", {"completed": completed, "total": total, "message": text}, stage="execution")


def _monitor_cancellation(path: Path | None, cancel_event: ThreadEvent, stop: ThreadEvent) -> None:
    if path is None:
        return
    while not stop.wait(0.1):
        if path.is_file():
            cancel_event.set()
            return


def _dispatch(request: Request, output: TextIO, cancel_event: ThreadEvent) -> None:
    sink = _EventSink(output, request)
    sink.emit("accepted", stage="transport")
    if request.command == "shutdown":
        sink.emit("finished", {"shutdown": True}, stage="worker")
        return
    if request.command == "cancel_task":
        cancel_event.set()
        sink.emit("cancelled", {"reason": "cancel requested"}, stage="worker")
        return
    sink.emit("started", stage=request.command)
    if request.command == "get_capabilities":
        from ..._backend import rust_capability

        capability = rust_capability()
        sink.emit(
            "finished",
            {
                "backend": "python",
                "native": _safe(capability),
                "operations": ["inspection", "pipeline"],
            },
            stage="capabilities",
        )
        return
    if request.command == "inspect_time_metadata":
        from ...time_mapping import inspect_time_metadata

        result = inspect_time_metadata(
            _path(request.payload, "input_dir"),
            recursive=bool(request.payload.get("recursive", False)),
            requested_engine=str(request.payload.get("engine", "auto")),
            cancel_event=cancel_event,
        )
        payload = _time_inspection_payload(result)
        sink.emit("inspection_ready", payload, stage="time_inspection")
        sink.emit("finished", payload, stage="time_inspection")
        return
    if request.command == "save_inspection_snapshot":
        from ..services import save_inspection_snapshot

        result = _inspection_from_payload(request.payload, cancel_event)
        destination = _path(request.payload, "destination")
        saved = save_inspection_snapshot(result, destination)
        payload = {"snapshot_path": str(saved), "kind": result.kind, "source": str(result.path)}
        sink.emit("finished", payload, stage="inspection")
        return
    if request.command in {"inspect_source", "inspect_zarr"}:
        result = _inspection(request, cancel_event)
        payload = {"kind": result.kind, "path": str(result.path), "report": result.report, "warnings": result.warnings, "snapshot": result.snapshot()}
        sink.emit("inspection_ready", payload, stage="inspection")
        sink.emit("finished", payload, stage="inspection")
        return
    if request.command in {"preview_pipeline", "run_pipeline", "resume_pipeline"}:
        from ..services import inspect_temporary_pipeline, preview_pipeline, run_pipeline

        inspection = _inspection_from_payload(request.payload, cancel_event)
        config = _pipeline_config(request.payload)
        if request.command == "preview_pipeline":
            plan = preview_pipeline(inspection, config)
            payload = {"plan_kind": type(plan).__name__, "plan": _safe(plan)}
            sink.emit("plan_ready", payload, stage="planning")
            sink.emit("finished", payload, stage="planning")
            return
        if request.command == "resume_pipeline":
            inspection = inspect_temporary_pipeline(_path(request.payload, "path"))
        cancel_file = _optional_path(request.payload, "cancellation_file")
        monitor_stop = ThreadEvent()
        monitor = Thread(target=_monitor_cancellation, args=(cancel_file, cancel_event, monitor_stop), daemon=True)
        monitor.start()
        try:
            with redirect_stdout(_OutputStream(sink, stream_name="stdout")), redirect_stderr(_OutputStream(sink, stream_name="stderr")):
                result = run_pipeline(inspection, config, cancel_event=cancel_event, progress=False)
            if cancel_event.is_set():
                sink.emit("cancelled", {"reason": "cancellation requested"}, stage="execution")
            else:
                sink.emit("finished", {"result": _safe(result)}, stage="published")
        except Exception as exc:  # execution errors are returned on the wire
            if cancel_event.is_set():
                sink.emit("cancelled", {"reason": str(exc)}, stage="execution")
            else:
                sink.emit("failed", {"error": error_payload("unknown", str(exc), stage="execution")}, stage="execution")
        finally:
            monitor_stop.set()
            monitor.join(timeout=1)
        return
    raise ProtocolError(f"command is not implemented yet: {request.command}")


def run_worker(input_stream: TextIO = sys.stdin, output_stream: TextIO = sys.stdout) -> int:
    for line in input_stream:
        if not line.strip():
            continue
        request_id = "unknown"
        task_id = None
        try:
            raw = json.loads(line)
            if isinstance(raw, dict):
                request_id = str(raw.get("request_id", request_id))
                task_id = raw.get("task_id")
            request = decode_request(raw)
            _dispatch(request, output_stream, ThreadEvent())
        except Exception as exc:
            write_event(
                output_stream,
                Event(
                    request_id=request_id,
                    task_id=task_id,
                    sequence=0,
                    event="failed",
                    stage="transport",
                    payload={"error": error_payload("worker_protocol_error", str(exc))},
                ),
            )
    return 0


def main() -> int:
    return run_worker()


if __name__ == "__main__":
    raise SystemExit(main())
