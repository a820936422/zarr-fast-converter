from __future__ import annotations

"""GUI-facing orchestration around the existing conversion engines.

The services in this module deliberately contain no Qt code.  A desktop
client, the CLI, and future automation can all use the same checked data
objects and operation configuration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..filename_mode import (
    FilenameScan,
    convert_filename as core_convert_filename,
    discover_filename_files,
    filename_logical_bytes,
    inspect_filename_inventory,
    probe_dataset_structure,
    scan_filename_times,
)
from ..inspection import (
    DimensionMappingRequired,
    inspect_dataset,
    inventory_summary,
)
from ..models import (
    FileRecord,
    Inventory,
    OutputLayout,
    Selection,
    VariableSpec,
    VariableTransform,
)
from ..planner import resolve_conversion_plan
from ..selection import make_selection, selected_logical_bytes
from ..rechunking.compression import make_compression_plan
from ..rechunking.engine import run_rechunk as core_run_rechunk
from ..rechunking.inspection import format_inspection, inspect_store
from ..rechunking.models import ChunkPlan, CompressionPlan, DatasetInfo
from ..rechunking.planning import DEFAULT_TARGET_MIB, plan_chunks
from ..resampling.engine import (
    format_plan as format_resample_plan,
    plan_resample as core_plan_resample,
    run_resample as core_run_resample,
)
from ..resampling.inspection import inspect_resample_input as core_inspect_resample
from ..resampling.models import (
    ResampleConfig,
    ResampleInspection,
    ResamplePlan,
)
from ..engine import convert as core_convert
from ..time_mapping import (
    FilenameField,
    TimeFieldRef,
    TimeInspectionResult,
    TimeRule,
    inspect_time_metadata,
)


SourceMode = Literal["auto", "complete", "filename"]


@dataclass(frozen=True)
class SourceInspectionConfig:
    input_dir: Path
    mode: SourceMode = "auto"
    recursive: bool = False
    engine: str = "auto"
    template: str | None = None
    field_values: tuple[str, ...] | None = None
    source_dimensions: tuple[str, str, str] | None = None
    workers: int | None = None
    time_rule: TimeRule | None = None
    time_inspection: TimeInspectionResult | None = None
    cache_path: Path | None = None


@dataclass
class InspectionResult:
    """A checked source or Zarr input kept by the GUI session."""

    kind: Literal["source", "zarr"]
    path: Path
    report: str
    inventory: Inventory | None = None
    scan: FilenameScan | None = None
    dataset_info: DatasetInfo | None = None
    mode: str | None = None
    warnings: list[str] = field(default_factory=list)
    time_inspection: TimeInspectionResult | None = None
    time_rule: TimeRule | None = None

    @property
    def source_inventory(self) -> Inventory:
        if self.inventory is None:
            raise ValueError("当前检查结果不是源数据 Inventory。")
        return self.inventory

    @property
    def zarr_info(self) -> DatasetInfo:
        if self.dataset_info is None:
            raise ValueError("当前检查结果不是 Zarr DatasetInfo。")
        return self.dataset_info

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-safe, human-readable summary of this inspection."""

        if self.inventory is not None:
            info = self.inventory
            times = [_date_label(value) for value in info.times]
            if self.scan is not None:
                missing_dates = [_date_label(value) for value in self.scan.missing_times]
            else:
                missing_dates = list(info.gaps)
            variables = []
            for spec in info.variables.values():
                variables.append(
                    {
                        "name": spec.name,
                        "dims": list(spec.dims),
                        "dtype": spec.dtype,
                        "shape_without_time": list(spec.shape_without_time),
                        "native_chunks": list(spec.native_chunks or ()),
                        "attrs": _json_safe(spec.attrs),
                    }
                )
            return {
                "schema_version": 3,
                "kind": self.kind,
                "report": self.report,
                "source": {
                    "path": str(self.path),
                    "file_count": len(info.files),
                    "total_bytes": info.total_bytes,
                    "engine": info.source_engine,
                    "mode": info.source_mode,
                    "source_dimensions": list(info.source_dimensions),
                    "filename_template": info.filename_template,
                    "filename_step_days": info.filename_step_days,
                    "filename_annual_steps": [list(item) for item in info.filename_annual_steps],
                },
                "time": {
                    "format": "YYYY-MM-DD",
                    "start": times[0] if times else None,
                    "end": times[-1] if times else None,
                    "count": len(times),
                    "frequency": info.frequency,
                    "missing": missing_dates,
                    "filename_template": info.filename_template,
                    "rule": _rule_snapshot(self.time_rule),
                },
                "dimensions": {
                    "time": len(info.times),
                    "lat": int(info.lat_values.size),
                    "lon": int(info.lon_values.size),
                },
                "coordinates": {
                    "lat_values": _json_safe(info.lat_values),
                    "lon_values": _json_safe(info.lon_values),
                    "lat_min": float(np.nanmin(info.lat_values)),
                    "lat_max": float(np.nanmax(info.lat_values)),
                    "lon_min": float(np.nanmin(info.lon_values)),
                    "lon_max": float(np.nanmax(info.lon_values)),
                },
                "inventory": {
                    "times": times,
                    "time_keys": list(info.time_keys),
                    "gaps": list(info.gaps),
                    "missing_time_keys": list(info.missing_time_keys),
                    "files": [
                        {
                            "path": str(record.path),
                            "size_bytes": record.size_bytes,
                            "mtime_ns": record.mtime_ns,
                            "times": [_date_label(value) for value in record.times],
                            "time_keys": list(record.time_keys),
                            "lat_hash": record.lat_hash,
                            "lon_hash": record.lon_hash,
                            "lat_size": record.lat_size,
                            "lon_size": record.lon_size,
                        }
                        for record in info.files
                    ],
                },
                "variables": variables,
                "warnings": list(self.warnings),
                "created_at": datetime.now(timezone.utc).isoformat(),
            }

        info = self.zarr_info
        return {
            "schema_version": 2,
            "kind": self.kind,
            "source": {"path": str(self.path), "zarr_format": info.zarr_format},
            "dimensions": dict(info.dimensions),
            "variables": [
                {
                    "name": variable.name,
                    "dims": list(variable.dims),
                    "shape": list(variable.shape),
                    "dtype": str(variable.dtype),
                    "chunks": list(variable.chunks),
                    "is_coord": variable.is_coord,
                    "attrs": _json_safe(variable.attrs),
                    "compressors": [repr(item) for item in variable.compressors],
                }
                for variable in info.variables
            ],
            "warnings": list(self.warnings),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }


@dataclass(frozen=True)
class ConversionConfig:
    output: Path
    time_start: str | None = None
    time_end: str | None = None
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    variables: tuple[str, ...] = ()
    variable_names: dict[str, str] = field(default_factory=dict)
    variable_transforms: dict[str, VariableTransform] = field(default_factory=dict)
    auto_tune: bool = False
    tune_budget: float = 60.0
    max_workers: int | None = None
    reserve_memory_gib: float = 2.0
    chunks: tuple[int, int, int] | None = None
    output_layout: OutputLayout | None = None
    overwrite: bool = False
    validate: bool = True


@dataclass(frozen=True)
class ConversionPreview:
    inventory: Inventory
    selection: Selection
    plan: Any
    logical_bytes: int


@dataclass(frozen=True)
class RechunkConfig:
    input: Path
    output: Path
    strategy: str = "time"
    target_mib: float = DEFAULT_TARGET_MIB
    custom_chunks: tuple[int, int, int] | None = None
    compression: str = "none"
    workers: int = 1
    overwrite: bool = False
    validate: bool = True
    # ``compression_only`` is retained for callers of the previous two-page
    # GUI/API.  New callers should select the two independent operations
    # below; both can then be applied in one output-producing pass.
    compression_only: bool = False
    rechunk: bool = True
    recompress: bool | None = None
    temporary_dir: Path | None = None


@dataclass(frozen=True)
class RechunkPreview:
    info: DatasetInfo
    plan: ChunkPlan
    compression: CompressionPlan


@dataclass(frozen=True)
class ResamplePreview:
    inspection: ResampleInspection
    plan: ResamplePlan


def inspect_source(config: SourceInspectionConfig, *, cancel_event=None) -> InspectionResult:
    """Inspect a source directory and resolve its time ingestion mode."""

    source = Path(config.input_dir).expanduser().resolve()
    cached_inventory = None
    if config.cache_path is not None:
        cache_path = Path(config.cache_path).expanduser().resolve()
        if cache_path.is_file():
            try:
                cached = load_inspection_snapshot(cache_path, validate_files=False)
            except (OSError, ValueError, FileNotFoundError):
                cached = None
            if (
                cached is not None
                and cached.path == source
                and _rule_snapshot(cached.time_rule) == _rule_snapshot(config.time_rule)
                and (
                    config.source_dimensions is None
                    or cached.source_inventory.source_dimensions == config.source_dimensions
                )
            ):
                cached_inventory = cached.source_inventory
    files = discover_filename_files(source, recursive=config.recursive)
    requested_engine = config.engine or "auto"
    mode = config.mode
    resolved_engine = requested_engine
    if config.time_rule is not None:
        config.time_rule.validate()
        time_result = config.time_inspection or inspect_time_metadata(
            source,
            recursive=config.recursive,
            requested_engine=requested_engine,
        )
        resolved_engine = time_result.engine
        if (
            config.time_rule.full is not None
            and config.time_rule.full.source == "filename"
            and not time_result.time_dimension.exists
        ):
            scan = _scan_selected_filename_full_time(
                source,
                config.time_rule.full,
                time_result.filename_fields,
                recursive=config.recursive,
            )
            inventory = inspect_filename_inventory(
                scan,
                resolved_engine,
                workers=config.workers,
                progress=True,
                cached_inventory=cached_inventory,
                cancel_event=cancel_event,
            )
            warnings = []
            if scan.missing_times:
                warnings.append(f"理论时间轴缺少 {len(scan.missing_times)} 个日期，转换时将写入空值切片。")
            return _cache_inspection_result(InspectionResult(
                kind="source",
                path=source,
                report=format_source_inventory(inventory, scan),
                inventory=inventory,
                scan=scan,
                mode="filename",
                warnings=warnings,
                time_inspection=time_result,
                time_rule=config.time_rule,
            ), config.cache_path)
        mode = "complete"
        inventory = inspect_dataset(
            source,
            recursive=config.recursive,
            engine=resolved_engine,
            dimension_names=config.source_dimensions,
            workers=config.workers,
            progress=True,
            time_rule=config.time_rule,
            filename_fields=time_result.filename_fields,
            cached_inventory=cached_inventory,
            cancel_event=cancel_event,
        )
        warnings = [
            f"检测到 {len(inventory.gaps)} 个源时间间隔缺口。"
            for _ in [0]
            if inventory.gaps
        ]
        return _cache_inspection_result(InspectionResult(
            kind="source",
            path=source,
            report=format_source_inventory(inventory, time_result=time_result, time_rule=config.time_rule),
            inventory=inventory,
            mode="hybrid" if config.time_rule.is_hybrid else "complete",
            warnings=warnings,
            time_inspection=time_result,
            time_rule=config.time_rule,
        ), config.cache_path)
    if mode == "auto":
        resolved_engine, dims, _coords, has_time, has_space = probe_dataset_structure(
            files[0], requested_engine
        )
        if has_time and has_space:
            mode = "complete"
        elif has_space:
            mode = "filename"
        else:
            raise ValueError(
                "首个文件既没有可识别的 time，也没有可识别的纬度/经度空间坐标。"
            )
    elif resolved_engine == "auto":
        resolved_engine = _engine_for_mode(files[0], mode)

    if mode == "complete":
        if config.source_dimensions is None and config.engine == "auto":
            # Probe already selected the backend.  The inspection function will
            # still request a mapping if the source uses non-standard names.
            pass
        inventory = inspect_dataset(
            source,
            recursive=config.recursive,
            engine=resolved_engine,
            dimension_names=config.source_dimensions,
            workers=config.workers,
            progress=True,
            cached_inventory=cached_inventory,
            cancel_event=cancel_event,
        )
        warnings = [
            f"检测到 {len(inventory.gaps)} 个源时间间隔缺口。"
            for _ in [0]
            if inventory.gaps
        ]
        return _cache_inspection_result(InspectionResult(
            kind="source",
            path=source,
            report=format_source_inventory(inventory),
            inventory=inventory,
            mode="complete",
            warnings=warnings,
        ), config.cache_path)

    if mode != "filename":
        raise ValueError(f"不支持的数据检查模式：{mode}")
    template = None if not config.template or config.template == "auto" else config.template
    scan = scan_filename_times(
        source,
        template=template,
        field_values=config.field_values,
        recursive=config.recursive,
    )
    inventory = inspect_filename_inventory(
        scan,
        resolved_engine,
        workers=config.workers,
        progress=True,
        cached_inventory=cached_inventory,
        cancel_event=cancel_event,
    )
    warnings = []
    if scan.missing_times:
        warnings.append(f"理论时间轴缺少 {len(scan.missing_times)} 个日期，转换时将写入空值切片。")
    if not scan.step_days:
        warnings.append("不同年份的文件名时间间隔不一致，需要用户确认年度时间规则。")
    return _cache_inspection_result(InspectionResult(
        kind="source",
        path=source,
        report=format_source_inventory(inventory, scan),
        inventory=inventory,
        scan=scan,
        mode="filename",
        warnings=warnings,
    ), config.cache_path)


def inspect_zarr(path: Path) -> InspectionResult:
    source = Path(path).expanduser().resolve()
    info = inspect_store(source)
    return InspectionResult(
        kind="zarr",
        path=source,
        report=format_inspection(info),
        dataset_info=info,
    )


def inspect_resample(path: Path) -> ResampleInspection:
    """Inspect a Zarr input for the xESMF resampling page."""

    return core_inspect_resample(path)


def load_inspection_snapshot(
    path: Path, *, validate_files: bool = True
) -> InspectionResult:
    """Load a complete source inspection snapshot for later conversion."""

    snapshot_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"无法读取检查快照：{snapshot_path}") from exc
    if payload.get("kind") != "source":
        raise ValueError("当前只支持导入源数据检查快照。")
    schema_version = int(payload.get("schema_version", 0))
    if schema_version < 2 or "inventory" not in payload:
        raise ValueError(
            "该快照是旧版摘要格式，缺少逐文件索引，不能直接用于转换；"
            "请重新检查源数据并保存新快照。"
        )
    if validate_files and schema_version < 3:
        raise ValueError(
            "该快照缺少文件修改时间，不能安全地直接用于转换；"
            "请重新检查源数据并保存新版快照。"
        )

    source_data = payload.get("source") or {}
    source = Path(str(source_data.get("path", ""))).expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"快照中的源数据目录不存在：{source}")
    coordinates = payload.get("coordinates") or {}
    if "lat_values" not in coordinates or "lon_values" not in coordinates:
        raise ValueError("快照缺少完整空间坐标，无法恢复转换索引。")

    variable_specs: dict[str, VariableSpec] = {}
    for item in payload.get("variables") or []:
        name = str(item.get("name", ""))
        if not name:
            raise ValueError("快照中存在没有名称的变量。")
        variable_specs[name] = VariableSpec(
            name=name,
            dims=tuple(str(value) for value in item.get("dims") or ()),
            dtype=str(item.get("dtype", "")),
            shape_without_time=tuple(
                int(value) for value in item.get("shape_without_time") or ()
            ),
            native_chunks=(
                tuple(int(value) for value in item["native_chunks"])
                if item.get("native_chunks")
                else None
            ),
            attrs=dict(item.get("attrs") or {}),
        )
    if not variable_specs:
        raise ValueError("快照中没有可转换变量。")

    inventory_data = payload.get("inventory") or {}
    files: list[FileRecord] = []
    shared_variables = tuple(variable_specs.values())
    for item in inventory_data.get("files") or []:
        file_path = Path(str(item.get("path", ""))).expanduser().resolve()
        if validate_files and not file_path.is_file():
            raise FileNotFoundError(f"快照中的源文件不存在：{file_path}")
        stored_size = int(item.get("size_bytes", 0))
        stored_mtime = item.get("mtime_ns")
        if validate_files:
            stat = file_path.stat()
            if stored_size != stat.st_size:
                raise ValueError(f"检查快照已过期，文件大小发生变化：{file_path}")
            if stored_mtime is not None and int(stored_mtime) != stat.st_mtime_ns:
                raise ValueError(f"检查快照已过期，文件修改时间发生变化：{file_path}")
        times = tuple(np.datetime64(value, "ns") for value in item.get("times") or ())
        files.append(
            FileRecord(
                path=file_path,
                size_bytes=stored_size,
                times=times,
                time_keys=tuple(str(value) for value in item.get("time_keys") or ()),
                lat_hash=str(item.get("lat_hash", "")),
                lon_hash=str(item.get("lon_hash", "")),
                lat_size=int(item.get("lat_size", len(coordinates["lat_values"]))),
                lon_size=int(item.get("lon_size", len(coordinates["lon_values"]))),
                variables=shared_variables,
                mtime_ns=int(stored_mtime) if stored_mtime is not None else None,
            )
        )
    if not files:
        raise ValueError("快照中没有逐文件索引。")

    source_dimensions = tuple(
        str(value) for value in source_data.get("source_dimensions") or ()
    )
    if len(source_dimensions) != 3:
        raise ValueError("快照中的源维度映射不完整。")
    inventory = Inventory(
        input_dir=source,
        files=files,
        lat_values=np.asarray(coordinates["lat_values"]),
        lon_values=np.asarray(coordinates["lon_values"]),
        times=np.asarray(
            [np.datetime64(value, "ns") for value in inventory_data.get("times") or ()],
            dtype="datetime64[ns]",
        ),
        time_keys=tuple(str(value) for value in inventory_data.get("time_keys") or ()),
        variables=variable_specs,
        source_engine=str(source_data.get("engine", "auto")),
        source_dimensions=source_dimensions,
        frequency=str(payload.get("time", {}).get("frequency", "")),
        gaps=[str(value) for value in inventory_data.get("gaps") or ()],
        total_bytes=int(
            source_data.get("total_bytes", sum(item.size_bytes for item in files))
        ),
        missing_time_keys=tuple(
            str(value) for value in inventory_data.get("missing_time_keys") or ()
        ),
        source_mode=str(source_data.get("mode", "dimension")),
        filename_template=source_data.get("filename_template"),
        filename_step_days=(
            int(source_data["filename_step_days"])
            if source_data.get("filename_step_days") is not None
            else None
        ),
        filename_annual_steps=tuple(
            (int(item[0]), int(item[1]))
            for item in source_data.get("filename_annual_steps") or ()
        ),
    )
    return InspectionResult(
        kind="source",
        path=source,
        report=str(payload.get("report", "")),
        inventory=inventory,
        mode=str(source_data.get("mode", "complete")),
        warnings=[str(value) for value in payload.get("warnings") or ()],
        time_rule=_rule_from_snapshot(payload.get("time", {}).get("rule")),
    )


def preview_conversion(
    inspection: InspectionResult,
    config: ConversionConfig,
) -> ConversionPreview:
    inventory = inspection.source_inventory
    _validate_variable_options(
        inventory,
        config.variables,
        config.variable_names,
        config.variable_transforms,
    )
    selection = make_selection(
        inventory,
        time_bounds=_date_bounds(config.time_start, config.time_end),
        lat_bounds=_numeric_bounds(config.lat_min, config.lat_max),
        lon_bounds=_numeric_bounds(config.lon_min, config.lon_max),
        variables=list(config.variables) or None,
    )
    plan = resolve_conversion_plan(
        inventory,
        selection,
        Path(config.output).expanduser().resolve(),
        reserve_gib=config.reserve_memory_gib,
        chunks=config.chunks,
        max_workers=config.max_workers,
    )
    logical = (
        filename_logical_bytes(inventory, selection, config.variable_transforms)
        if inventory.source_mode == "filename"
        else selected_logical_bytes(inventory, selection)
    )
    return ConversionPreview(inventory, selection, plan, logical)


def run_conversion(
    inspection: InspectionResult,
    config: ConversionConfig,
    *,
    cancel_event=None,
) -> tuple[Any, dict[str, Any]]:
    preview = preview_conversion(inspection, config)
    output = Path(config.output).expanduser().resolve()
    if preview.inventory.source_mode == "filename":
        return core_convert_filename(
            preview.inventory,
            preview.selection,
            output,
            transforms=config.variable_transforms,
            variable_names=config.variable_names,
            chunks=config.chunks,
            output_layout=config.output_layout,
            plan=preview.plan,
            auto_tune=config.auto_tune,
            tune_budget=config.tune_budget,
            max_workers=config.max_workers,
            reserve_gib=config.reserve_memory_gib,
            overwrite=config.overwrite,
            validate=config.validate,
            progress=True,
            cancel_event=cancel_event,
        )
    return core_convert(
        preview.inventory,
        preview.selection,
        output,
        auto_tune=config.auto_tune,
        tune_budget=config.tune_budget,
        max_workers=config.max_workers,
        reserve_gib=config.reserve_memory_gib,
        overwrite=config.overwrite,
        validate=config.validate,
        progress=True,
        variable_transforms=config.variable_transforms,
        variable_names=config.variable_names,
        chunks=config.chunks,
        output_layout=config.output_layout,
        cancel_event=cancel_event,
    )


def preview_rechunk(config: RechunkConfig, info: DatasetInfo | None = None) -> RechunkPreview:
    info = info or inspect_store(config.input)
    rechunk_enabled = bool(config.rechunk)
    if config.compression_only:
        rechunk_enabled = False
    recompress_enabled = (
        True
        if config.compression_only
        else (config.compression != "none" if config.recompress is None else bool(config.recompress))
    )
    if not rechunk_enabled and not recompress_enabled:
        raise ValueError("请至少选择重分块或重压缩中的一项。")

    strategy = config.strategy
    custom = config.custom_chunks
    if not rechunk_enabled:
        strategy = "custom"
        reference = next(
            (variable for variable in info.data_variables if variable.ndim == 3),
            None,
        )
        if reference is None:
            raise ValueError("输入 Zarr 没有可用于重压缩的三维数据变量。")
        custom = tuple(
            min(int(chunk), int(size))
            for chunk, size in zip(reference.chunks, reference.shape)
        )
    plan = plan_chunks(
        info,
        strategy,  # type: ignore[arg-type]
        target_mib=config.target_mib,
        workers=config.workers,
        custom_chunks=custom,
    )
    compression = make_compression_plan(
        config.compression if recompress_enabled else "none"
    )
    return RechunkPreview(info, plan, compression)


def run_rechunk(
    config: RechunkConfig,
    info: DatasetInfo | None = None,
    *,
    cancel_event=None,
) -> dict[str, Any]:
    preview = preview_rechunk(config, info)
    return core_run_rechunk(
        config.input,
        config.output,
        preview.info,
        preview.plan,
        preview.compression,
        workers=max(1, int(config.workers)),
        overwrite=config.overwrite,
        progress=True,
        validate=config.validate,
        cancel_event=cancel_event,
        temporary_dir=config.temporary_dir,
    )


def preview_resample(
    config: ResampleConfig,
    inspection: ResampleInspection | None = None,
) -> ResamplePreview:
    plan = core_plan_resample(config, inspection)
    return ResamplePreview(plan.inspection, plan)


def run_resample(
    config: ResampleConfig,
    inspection: ResampleInspection | None = None,
    *,
    cancel_event=None,
) -> dict[str, Any]:
    preview = preview_resample(config, inspection)
    return core_run_resample(
        config,
        preview.plan,
        cancel_event=cancel_event,
        progress=True,
    )


def preview_pipeline(inspection: InspectionResult, config):
    """Build a no-write plan from a completed raw or Zarr inspection."""

    from ..pipeline.engine import preview_pipeline as core_preview_pipeline

    return core_preview_pipeline(inspection, config)


def run_pipeline(
    inspection: InspectionResult,
    config,
    *,
    cancel_event=None,
    progress: bool = True,
):
    """Run the unified raw/Zarr input pipeline and publish one final Zarr."""

    from ..pipeline.engine import run_pipeline as core_run_pipeline

    return core_run_pipeline(
        inspection,
        config,
        cancel_event=cancel_event,
        progress=progress,
    )


def format_resample_preview(preview: ResamplePreview) -> str:
    return format_resample_plan(preview.plan)


def save_inspection_snapshot(result: InspectionResult, destination: Path) -> Path:
    destination = Path(destination).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(result.snapshot(), ensure_ascii=False, indent=2, default=_json_safe),
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _cache_inspection_result(
    result: InspectionResult, cache_path: Path | None
) -> InspectionResult:
    if cache_path is not None:
        save_inspection_snapshot(result, cache_path)
    return result


def format_source_inventory(
    info: Inventory,
    scan: FilenameScan | None = None,
    *,
    time_result: TimeInspectionResult | None = None,
    time_rule: TimeRule | None = None,
) -> str:
    lines = ["========== 数据检查结果 ==========", inventory_summary(info)]
    lines.extend(
        [
            f"模式：{'源数据 time 维度' if info.source_mode == 'dimension' else ('文件名 + time 组合' if info.source_mode == 'hybrid' else '文件名重建 time')}",
            f"引擎：{info.source_engine}",
            f"标准 shape(time, lat, lon)：({len(info.times)}, {len(info.lat_values)}, {len(info.lon_values)})",
            "",
            "变量：",
        ]
    )
    for spec in info.variables.values():
        attrs = ", ".join(
            f"{key}={spec.attrs[key]!r}"
            for key in ("_FillValue", "missing_value", "scale_factor", "add_offset", "units")
            if key in spec.attrs
        )
        lines.append(
            f"  {spec.name}: dtype={spec.dtype}, dims={spec.dims}, "
            f"shape_without_time={spec.shape_without_time}, "
            f"native_chunks={spec.native_chunks or '未提供'}"
        )
        if attrs:
            lines.append(f"    属性：{attrs}")
    if time_result is not None:
        lines.extend(
            [
                "",
                "已确认时间构建规则：",
                f"  {_render_time_rule(time_rule)}",
                f"  time 原始格式：{time_result.time_dimension.format_label}",
            ]
        )
    if scan is not None:
        lines.extend(
            [
                "",
                f"文件名模板：{'年 + DOY' if scan.template == 'doy' else '年 + 月 + 日'}",
                f"样例文件：{scan.sample_name}",
                f"实际日期：{_date_label(scan.actual_times[0])} .. {_date_label(scan.actual_times[-1])}",
                f"理论日期：{_date_label(scan.expected_times[0])} .. {_date_label(scan.expected_times[-1])}",
                f"缺失日期数量：{len(scan.missing_times)}",
            ]
        )
        if scan.missing_times:
            lines.append(
                "缺失日期："
                + ", ".join(_date_label(item) for item in scan.missing_times[:30])
                + (" ……" if len(scan.missing_times) > 30 else "")
            )
    return "\n".join(lines)


def _render_time_rule(rule: TimeRule | None) -> str:
    if rule is None:
        return "未提供"
    if rule.full is not None:
        return f"完整时间来自 {rule.full.source} 字段 #{rule.full.index}"
    values = []
    for label, ref in (("年", rule.year), ("月", rule.month), ("日", rule.day), ("DOY", rule.doy)):
        if ref is not None:
            values.append(f"{label}来自 {ref.source} 字段 #{ref.index}")
    return "，".join(values)


def _scan_selected_filename_full_time(
    source: Path,
    ref: TimeFieldRef,
    fields: tuple[FilenameField, ...],
    *,
    recursive: bool,
) -> FilenameScan:
    from ..filename_mode import scan_filename_times

    if ref.source != "filename" or ref.component != "full":
        raise ValueError("文件名无 time 维度时，必须选择文件名完整时间字段。")
    field = next((item for item in fields if item.index == ref.index), None)
    if field is None:
        raise ValueError(f"找不到文件名时间字段 #{ref.index}。")
    if field.length == 7:
        template = "doy"
        values = (field.sample[:4], field.sample[4:])
    elif field.length == 8:
        template = "ymd"
        values = (field.sample[:4], field.sample[4:6], field.sample[6:])
    else:
        raise ValueError("文件名完整时间必须是 YYYYDOY 或 YYYYMMDD。")
    return scan_filename_times(
        source,
        template=template,
        field_values=values,
        recursive=recursive,
    )


def _engine_for_mode(path: Path, mode: SourceMode) -> str:
    suffix = path.suffix.lower()
    if mode == "filename" and suffix in {".tif", ".tiff"}:
        return "rasterio"
    if suffix == ".hdf":
        return "netcdf4"
    return "h5netcdf"


def _date_bounds(start: str | None, end: str | None) -> list[str] | None:
    if not start and not end:
        return None
    if not start or not end:
        raise ValueError("开始日期和结束日期必须同时填写。")
    return [start, end]


def _numeric_bounds(lower: float | None, upper: float | None) -> list[float] | None:
    if lower is None and upper is None:
        return None
    if lower is None or upper is None:
        raise ValueError("范围的上下限必须同时填写。")
    return [lower, upper]


def _date_label(value: Any) -> str:
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="D"))
    return str(value)[:10]


def _rule_snapshot(rule: TimeRule | None) -> dict[str, Any] | None:
    if rule is None:
        return None
    result: dict[str, Any] = {}
    for component in ("full", "year", "month", "day", "doy"):
        ref = getattr(rule, component)
        if ref is not None:
            result[component] = {
                "source": ref.source,
                "component": ref.component,
                "index": ref.index,
            }
    return result


def _rule_from_snapshot(value: Any) -> TimeRule | None:
    if not isinstance(value, dict):
        return None
    refs: dict[str, TimeFieldRef | None] = {
        "full": None,
        "year": None,
        "month": None,
        "day": None,
        "doy": None,
    }
    for component in refs:
        item = value.get(component)
        if not isinstance(item, dict):
            continue
        refs[component] = TimeFieldRef(
            source=str(item.get("source")),
            component=str(item.get("component")),
            index=int(item.get("index", 0)),
        )
    return TimeRule(**refs)


def _validate_variable_options(
    inventory: Inventory,
    selected: tuple[str, ...],
    variable_names: dict[str, str],
    transforms: dict[str, VariableTransform],
) -> None:
    selected_names = tuple(selected) if selected else tuple(inventory.variables)
    unknown = sorted(set(selected_names) - set(inventory.variables))
    if unknown:
        raise ValueError("未知变量：" + ", ".join(unknown))
    unknown_options = sorted(
        (set(variable_names) | set(transforms)) - set(selected_names)
    )
    if unknown_options:
        raise ValueError("变量配置包含未选择的变量：" + ", ".join(unknown_options))
    output_names = {
        source: str(variable_names.get(source, source)).strip() or source
        for source in selected_names
    }
    reserved = {"time", "lat", "lon"}
    invalid = [
        name for name in output_names.values()
        if not name or name in reserved or any(char in name for char in "/\\")
    ]
    if invalid:
        raise ValueError("输出变量名无效或与坐标冲突：" + ", ".join(invalid))
    reverse: dict[str, str] = {}
    for source, output in output_names.items():
        previous = reverse.get(output)
        if previous is not None and previous != source:
            raise ValueError(f"多个源变量不能使用同一个输出变量名：{output}")
        reverse[output] = source


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, tuple)):
        return list(value)
    try:
        json.dumps(value)
    except TypeError:
        return repr(value)
    return value
