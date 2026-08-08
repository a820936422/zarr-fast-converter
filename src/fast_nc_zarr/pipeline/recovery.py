from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from ..models import VariableTransform
from ..rechunking.inspection import inspect_store
from ..rechunking.models import DatasetInfo
from .models import (
    PipelineChunkingOptions,
    PipelineCompressionOptions,
    PipelineConfig,
    PipelineConversionOptions,
    PipelineGeneralConfig,
    PipelineInput,
    PipelineOperations,
    PipelineResamplingOptions,
)


class PipelineRecoveryError(ValueError):
    """Raised when a retained pipeline directory has no safe resume point."""


@dataclass(frozen=True)
class PipelineRecovery:
    job_root: Path
    manifest_path: Path
    checkpoint: Path
    checkpoint_stage: str
    info: DatasetInfo
    config: PipelineConfig
    manifest: dict[str, Any]
    report: str


def _json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineRecoveryError(f"无法读取临时任务清单：{path}") from exc
    if not isinstance(payload, dict):
        raise PipelineRecoveryError(f"临时任务清单不是 JSON 对象：{path}")
    return payload


def _candidate_manifests(path: Path) -> list[Path]:
    source = path.expanduser().resolve()
    if source.is_file():
        return [source] if source.name == "manifest.json" else []
    candidates = []
    direct = source / "manifest.json"
    if direct.is_file():
        candidates.append(direct)
    for container in (source, source / "fast-nc-zarr-pipeline"):
        if container.is_dir():
            candidates.extend(item for item in container.glob("*/manifest.json") if item.is_file())
    return list(dict.fromkeys(candidates))


def _transform(value: dict[str, Any]) -> VariableTransform:
    fills = value.get("fill_values")
    return VariableTransform(
        fill_values=tuple(fills) if fills is not None else None,
        scale_factor=value.get("scale_factor"),
        output_fill=value.get("output_fill"),
    )


def _as_auto_int(value: Any) -> int | str:
    return "auto" if value == "auto" else int(value)


def _restore_config(
    payload: dict[str, Any], temporary_base: Path, checkpoint_stage: str
) -> PipelineConfig:
    data = payload.get("config")
    if not isinstance(data, dict):
        raise PipelineRecoveryError("任务清单缺少完整 config，无法恢复执行计划。")
    general = dict(data.get("general") or {})
    conversion = dict(data.get("conversion") or {})
    operations = dict(data.get("operations") or {})
    resampling = dict(data.get("resampling") or {})
    chunking = dict(data.get("chunking") or {})
    compression = dict(data.get("compression") or {})
    output_text = str(general.get("output") or payload.get("output") or "").strip()
    if not output_text:
        raise PipelineRecoveryError("任务清单缺少最终输出路径。")
    remaining_resample = bool(operations.get("resample")) and checkpoint_stage == "conversion"
    remaining_rechunk = bool(operations.get("rechunk"))
    remaining_recompress = bool(operations.get("recompress"))
    if not (remaining_resample or remaining_rechunk or remaining_recompress):
        raise PipelineRecoveryError("该临时任务没有尚未完成的处理阶段。")
    transforms = {
        str(name): _transform(dict(value))
        for name, value in dict(conversion.get("variable_transforms") or {}).items()
    }
    custom = chunking.get("custom_chunks")
    return PipelineConfig(
        input=PipelineInput(kind="zarr"),
        general=PipelineGeneralConfig(
            output=Path(output_text).expanduser(),
            temporary_dir=temporary_base,
            time_start=general.get("time_start"),
            time_end=general.get("time_end"),
            lat_min=float(general.get("lat_min", -90.0)),
            lat_max=float(general.get("lat_max", 90.0)),
            lon_min=float(general.get("lon_min", -180.0)),
            lon_max=float(general.get("lon_max", 180.0)),
            cleanup_intermediate=bool(general.get("cleanup_intermediate", False)),
            overwrite=bool(general.get("overwrite", False)),
        ),
        conversion=PipelineConversionOptions(
            variables=tuple(str(item) for item in conversion.get("variables") or ()),
            variable_names={
                str(name): str(value)
                for name, value in dict(conversion.get("variable_names") or {}).items()
            },
            variable_transforms=transforms,
            auto_tune=bool(conversion.get("auto_tune", True)),
            tune_budget=float(conversion.get("tune_budget", 60.0)),
            max_workers=(
                int(conversion["max_workers"])
                if conversion.get("max_workers") is not None
                else None
            ),
            reserve_memory_gib=float(conversion.get("reserve_memory_gib", 2.0)),
        ),
        operations=PipelineOperations(
            resample=remaining_resample,
            rechunk=remaining_rechunk,
            recompress=remaining_recompress,
        ),
        resampling=PipelineResamplingOptions(
            resolution=float(resampling.get("resolution", 0.1)),
            method=str(resampling.get("method", "bilinear")),
            skipna=bool(resampling.get("skipna", True)),
            na_thres=float(resampling.get("na_thres", 1.0)),
            compute_dtype=str(resampling.get("compute_dtype", "source")),
            tile_size=_as_auto_int(resampling.get("tile_size", "auto")),
            time_block=_as_auto_int(resampling.get("time_block", "auto")),
            compute_workers=int(resampling.get("compute_workers", 2)),
            space_workers=_as_auto_int(resampling.get("space_workers", "auto")),
            before_conditions=str(resampling.get("before_conditions", "")),
            before_results=str(resampling.get("before_results", "")),
            after_conditions=str(resampling.get("after_conditions", "")),
            after_results=str(resampling.get("after_results", "")),
            statistics_policy=str(resampling.get("statistics_policy", "auto")),
        ),
        chunking=PipelineChunkingOptions(
            strategy=str(chunking.get("strategy", "time")),
            target_mib=float(chunking.get("target_mib", 128.0)),
            custom_chunks=tuple(int(item) for item in custom) if custom else None,
            workers=int(chunking.get("workers", 1)),
        ),
        compression=PipelineCompressionOptions(
            profile=str(compression.get("profile", "balanced")),
            codec=compression.get("codec"),
            level=(
                int(compression["level"])
                if compression.get("level") is not None
                else None
            ),
            shuffle=str(compression.get("shuffle", "auto")),
        ),
        validate=bool(data.get("validate", True)),
    )


def _relative_checkpoint(job_root: Path, value: str) -> Path:
    relative = Path(value)
    if relative.is_absolute():
        raise PipelineRecoveryError("检查点路径必须相对于任务目录。")
    candidate = (job_root / relative).resolve()
    if not candidate.is_relative_to(job_root) or candidate.is_symlink():
        raise PipelineRecoveryError("检查点路径越出任务目录或指向符号链接。")
    return candidate


def _checkpoint(payload: dict[str, Any], job_root: Path) -> tuple[str, Path]:
    stages = dict(payload.get("stages") or {})
    resume = dict(payload.get("resume") or {})
    checkpoints = dict(resume.get("checkpoints") or {})
    resampling = dict(stages.get("resampling") or {})
    resampling_checkpoint = dict(checkpoints.get("resampling") or {})
    if str(resampling.get("status", "")).startswith("validated"):
        relative = str(resampling_checkpoint.get("path") or "resampled.zarr")
        candidate = _relative_checkpoint(job_root, relative)
        if candidate.is_dir():
            return "resampling", candidate
    conversion = dict(stages.get("conversion") or {})
    conversion_checkpoint = dict(checkpoints.get("conversion") or {})
    if str(conversion.get("status", "")).startswith("validated"):
        relative = str(conversion_checkpoint.get("path") or "source-crop.zarr")
        candidate = _relative_checkpoint(job_root, relative)
        if candidate.is_dir():
            return "conversion", candidate
    raise PipelineRecoveryError(
        "任务没有已验证且仍存在的 conversion/resampling 中间 Zarr，不能安全续跑。"
    )


def _validate_checkpoint(
    payload: dict[str, Any], stage: str, info: DatasetInfo
) -> None:
    if info.zarr_format != 3:
        raise PipelineRecoveryError("恢复检查点不是 Zarr v3。")
    if stage != "conversion":
        return
    window = dict(payload.get("source_read_window") or {})
    layout = dict(payload.get("output_layout") or {})
    layouts = list(layout.get("variables") or ())
    data_layouts = [item for item in layouts if not item.get("is_coord")]
    if data_layouts and window:
        expected = {
            "time": int(data_layouts[0]["shape"][0]),
            "lat": int(window["lat_stop"]) - int(window["lat_start"]),
            "lon": int(window["lon_stop"]) - int(window["lon_start"]),
        }
        actual = {name: info.dimensions.get(name) for name in expected}
        if actual != expected:
            raise PipelineRecoveryError(
                f"转换检查点维度不完整：期望 {expected}，实际 {actual}。"
            )
        expected_names = {str(item["output_name"]) for item in data_layouts}
        actual_names = {item.name for item in info.data_variables}
        if not expected_names.issubset(actual_names):
            missing = sorted(expected_names - actual_names)
            raise PipelineRecoveryError("转换检查点缺少变量：" + ", ".join(missing))


def inspect_pipeline_recovery(path: Path) -> PipelineRecovery:
    source = path.expanduser().resolve()
    manifests = _candidate_manifests(source)
    if not manifests:
        raise PipelineRecoveryError(
            "未找到 fast-nc-zarr-pipeline/<任务ID>/manifest.json。"
        )
    failures: list[str] = []
    recoveries: list[PipelineRecovery] = []
    for manifest_path in manifests:
        try:
            payload = _json(manifest_path)
            if payload.get("status") not in {"failed", "cancelled"}:
                raise PipelineRecoveryError(
                    f"任务状态为 {payload.get('status')!r}，不是失败或取消任务。"
                )
            job_root = manifest_path.parent.resolve()
            stage, checkpoint = _checkpoint(payload, job_root)
            info = inspect_store(checkpoint)
            _validate_checkpoint(payload, stage, info)
            temporary_base = job_root.parent.parent
            config = _restore_config(payload, temporary_base, stage)
            remaining = []
            if config.operations.resample:
                remaining.append("重采样")
            if config.operations.rechunk:
                remaining.append("重分块")
            if config.operations.recompress:
                remaining.append("重压缩")
            report = (
                "临时处理产物检查通过\n"
                f"任务：{payload.get('job_id', job_root.name)}\n"
                f"原任务状态：{payload.get('status')}\n"
                f"已验证检查点：{stage} -> {checkpoint}\n"
                f"检查点 shape(time, lat, lon)：{info.shape}\n"
                f"待继续阶段：{', '.join(remaining)}\n"
                f"最终输出：{config.general.output}\n"
                f"原错误：{payload.get('error', '无')}"
            )
            recoveries.append(
                PipelineRecovery(
                    job_root,
                    manifest_path,
                    checkpoint,
                    stage,
                    info,
                    config,
                    payload,
                    report,
                )
            )
        except Exception as exc:
            failures.append(f"{manifest_path.parent.name}: {exc}")
    if not recoveries:
        raise PipelineRecoveryError(
            "没有可安全续跑的临时任务：\n  " + "\n  ".join(failures)
        )
    return max(recoveries, key=lambda item: item.manifest_path.stat().st_mtime_ns)


def mark_recovery_succeeded(
    recovery: PipelineRecovery, result: dict[str, Any], *, cleanup: bool
) -> None:
    payload = _json(recovery.manifest_path)
    payload["status"] = "resumed_succeeded"
    payload["resume_result"] = {
        "output": result.get("output"),
        "manifest": result.get("manifest"),
        "resumed_from": recovery.checkpoint_stage,
    }
    if cleanup:
        for item in recovery.job_root.iterdir():
            if item == recovery.manifest_path:
                continue
            if item.is_dir() and not item.is_symlink():
                shutil.rmtree(item)
            else:
                item.unlink()
        payload["retained_checkpoints_cleaned"] = True
    temporary = recovery.manifest_path.with_name("manifest.json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(recovery.manifest_path)
