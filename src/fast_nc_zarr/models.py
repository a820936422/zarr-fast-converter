from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class VariableSpec:
    name: str
    dims: tuple[str, ...]
    dtype: str
    shape_without_time: tuple[int, ...]
    native_chunks: tuple[int, ...] | None
    attrs: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def itemsize(self) -> int:
        return np.dtype(self.dtype).itemsize

    @property
    def direct_compatible(self) -> bool:
        return (
            set(self.dims) == {"time", "lat", "lon"}
            and len(self.dims) == 3
            and np.dtype(self.dtype).kind in "biufc"
        )


@dataclass(frozen=True)
class VariableTransform:
    """User-confirmed raw-value processing for a variable.

    ``fill_values`` are compared against the unscaled source values.  An
    empty tuple means that no replacement was requested; ``None`` is used by
    callers that want to distinguish an omitted option while collecting
    metadata.  ``scale_factor`` and ``add_offset`` are applied only to values
    that are not marked missing.  ``output_fill`` is the marker used for
    integer output arrays.
    """

    fill_values: tuple[float, ...] | None = None
    scale_factor: float | None = None
    add_offset: float | None = None
    output_fill: float | int | None = None


@dataclass(frozen=True)
class CodecSpec:
    """Serializable Zarr v3 compressor description used across stages."""

    kind: Literal["blosc", "zstd", "gzip"]
    level: int
    cname: str | None = None
    shuffle: Literal["noshuffle", "shuffle", "bitshuffle"] = "noshuffle"


@dataclass(frozen=True)
class VariableOutputLayout:
    source_name: str
    output_name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...]
    codec: CodecSpec | None = None
    is_coord: bool = False


@dataclass(frozen=True)
class OutputLayout:
    """Complete final storage contract shared by planning and execution."""

    variables: tuple[VariableOutputLayout, ...]
    axis_reversals: tuple[Literal["lat", "lon"], ...] = ()

    def for_source(self, name: str) -> VariableOutputLayout:
        try:
            return next(item for item in self.variables if item.source_name == name)
        except StopIteration as exc:
            raise KeyError(f"输出布局缺少源变量：{name}") from exc

    def for_output(self, name: str) -> VariableOutputLayout:
        try:
            return next(item for item in self.variables if item.output_name == name)
        except StopIteration as exc:
            raise KeyError(f"输出布局缺少变量：{name}") from exc

    @property
    def coordinate_codec(self) -> CodecSpec | None:
        return next(
            (item.codec for item in self.variables if item.is_coord),
            None,
        )


@dataclass(frozen=True)
class FileRecord:
    path: Path
    size_bytes: int
    times: tuple[Any, ...]
    time_keys: tuple[str, ...]
    lat_hash: str
    lon_hash: str
    lat_size: int
    lon_size: int
    variables: tuple[VariableSpec, ...]
    mtime_ns: int | None = None

    @property
    def time_count(self) -> int:
        return len(self.times)


@dataclass
class Inventory:
    input_dir: Path
    files: list[FileRecord]
    lat_values: np.ndarray
    lon_values: np.ndarray
    times: np.ndarray
    time_keys: tuple[str, ...]
    variables: dict[str, VariableSpec]
    source_engine: str
    # Actual source dimension names in canonical (time, lat, lon) order.
    # VariableSpec.dims and all selections use the canonical names instead.
    source_dimensions: tuple[str, str, str]
    frequency: str
    gaps: list[str]
    total_bytes: int
    # Time keys that were inserted to create a complete theoretical time axis.
    missing_time_keys: tuple[str, ...] = ()
    source_mode: Literal["dimension", "filename", "hybrid"] = "dimension"
    filename_template: str | None = None
    filename_step_days: int | None = None
    filename_annual_steps: tuple[tuple[int, int], ...] = ()

    @property
    def reference_file(self) -> Path:
        return self.files[0].path

    @property
    def median_file_bytes(self) -> int:
        return int(np.median([item.size_bytes for item in self.files]))

    @property
    def median_times_per_file(self) -> int:
        return int(np.median([item.time_count for item in self.files]))


@dataclass(frozen=True)
class Selection:
    variables: tuple[str, ...]
    time_start: int
    time_stop: int
    lat_start: int
    lat_stop: int
    lon_start: int
    lon_stop: int

    @property
    def shape(self) -> tuple[int, int, int]:
        return (
            self.time_stop - self.time_start,
            self.lat_stop - self.lat_start,
            self.lon_stop - self.lon_start,
        )


@dataclass(frozen=True)
class StorageProfile:
    """Storage classification plus the evidence behind it.

    The first four fields retain the original positional API.  ``rotational``
    is the trusted value used by legacy callers, while ``reported_rotational``
    preserves an untrusted kernel/sysfs report (notably for WSL virtual disks).
    """

    path: Path
    device: str
    rotational: bool | None
    filesystem: str
    medium: Literal["ssd", "hdd", "network", "unknown", "virtual_unknown"] = (
        "unknown"
    )
    reported_rotational: bool | None = None
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"
    mountpoint: Path | None = None
    evidence: tuple[str, ...] = ()
    override: Literal["auto", "ssd", "hdd", "network"] = "auto"

    def to_dict(self, *, redact_paths: bool = False) -> dict[str, Any]:
        """Return a JSON-safe summary; runtime manifests can redact host paths."""

        if redact_paths:
            evidence = []
            safe_values = {"classification", "filesystem", "override", "sysfs_rotational"}
            for item in self.evidence:
                parts = item.split(":")
                evidence.append(
                    ":".join(parts[:2])
                    if parts[0] in safe_values and len(parts) > 1
                    else parts[0]
                )
        else:
            evidence = list(self.evidence)
        return {
            "path": "<redacted>" if redact_paths else str(self.path),
            "device": "<redacted>" if redact_paths else self.device,
            "rotational": self.rotational,
            "filesystem": self.filesystem,
            "medium": self.medium,
            "reported_rotational": self.reported_rotational,
            "confidence": self.confidence,
            "mountpoint": (
                "<redacted>"
                if redact_paths and self.mountpoint is not None
                else str(self.mountpoint)
                if self.mountpoint is not None
                else None
            ),
            "evidence": evidence,
            "override": self.override,
        }


@dataclass(frozen=True)
class ConversionPlan:
    strategy: Literal["file", "chunk", "dask"]
    workers: int
    chunk_time: int
    chunk_lat: int
    chunk_lon: int
    task_batch: int = 1
    compression: str = "zstd"
    compression_level: int = 1
    shuffle: str = "noshuffle"
    rationale: tuple[str, ...] = ()
    file_affinity: bool = False

    @property
    def chunks(self) -> tuple[int, int, int]:
        return self.chunk_time, self.chunk_lat, self.chunk_lon

    def label(self) -> str:
        return (
            f"{self.strategy}: workers={self.workers}, "
            f"chunks={self.chunks}, batch={self.task_batch}, "
            f"codec={self.compression}:{self.compression_level}/{self.shuffle}"
        )


@dataclass(frozen=True)
class BenchmarkResult:
    plan: ConversionPlan
    elapsed: float
    logical_bytes: int
    physical_bytes: int
    durable_mib_s: float
    average_cpu: float
    peak_rss: int
    # Production write speed excludes the final fsync used to make a trial
    # durable.  Keep the historical durable metric while exposing the two
    # rates needed to choose a fast plan and to estimate disk usage.
    logical_mib_s: float = 0.0
    physical_mib_s: float = 0.0
    compression_ratio: float = 0.0
    sample_count: int = 1
    status: str = "ok"
    failure: str | None = None
    candidate_id: int | None = None
    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe candidate trial record for manifests."""

        return {
            "plan": {
                "strategy": self.plan.strategy,
                "workers": self.plan.workers,
                "chunks": list(self.plan.chunks),
                "task_batch": self.plan.task_batch,
                "compression": self.plan.compression,
                "compression_level": self.plan.compression_level,
                "shuffle": self.plan.shuffle,
            },
            "elapsed": self.elapsed,
            "logical_bytes": self.logical_bytes,
            "physical_bytes": self.physical_bytes,
            "durable_mib_s": self.durable_mib_s,
            "logical_mib_s": self.logical_mib_s,
            "physical_mib_s": self.physical_mib_s,
            "average_cpu": self.average_cpu,
            "peak_rss": self.peak_rss,
            "compression_ratio": self.compression_ratio,
            "status": self.status,
            "failure": self.failure,
            "candidate_id": self.candidate_id,
            "sample_count": self.sample_count,
        }
