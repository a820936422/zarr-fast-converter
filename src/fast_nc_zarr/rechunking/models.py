from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


Strategy = Literal["time", "space", "custom"]

CompressionProfile = Literal["none", "fast", "balanced", "maximum", "custom", "auto"]
CompressionCodec = Literal[
    "none",
    "blosc-zstd",
    "blosc-lz4",
    "blosc-lz4hc",
    "blosc-zlib",
    "zstd",
    "gzip",
]
ShufflePolicy = Literal["auto", "noshuffle", "shuffle", "bitshuffle"]
CompressionObjective = Literal["speed", "balanced", "compact"]


@dataclass(frozen=True)
class CompressionResourceBudget:
    """Optional, caller-supplied safety limits for compression probing.

    The tuner never assumes a particular machine size.  ``None`` means that a
    limit is not supplied; the benchmark may still use the operating system's
    free-space report when it creates a trial store.
    """

    memory_bytes: int | None = None
    disk_free_bytes: int | None = None
    cpu_count: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        return {
            "memory_bytes": self.memory_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "cpu_count": self.cpu_count,
        }


@dataclass(frozen=True)
class VariableInfo:
    name: str
    dims: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: np.dtype
    chunks: tuple[int, ...]
    is_coord: bool
    attrs: dict[str, Any] = field(default_factory=dict)
    compressors: tuple[Any, ...] = ()

    @property
    def logical_bytes(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64)) * self.dtype.itemsize

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def kind(self) -> str:
        if np.issubdtype(self.dtype, np.integer):
            return "integer"
        if np.issubdtype(self.dtype, np.floating):
            return "floating"
        if np.issubdtype(self.dtype, np.bool_):
            return "boolean"
        if np.issubdtype(self.dtype, np.datetime64):
            return "datetime"
        return "other"


@dataclass(frozen=True)
class DatasetInfo:
    path: Path
    dimensions: dict[str, int]
    variables: tuple[VariableInfo, ...]
    attrs: dict[str, Any]
    zarr_format: int

    @property
    def data_variables(self) -> tuple[VariableInfo, ...]:
        return tuple(variable for variable in self.variables if not variable.is_coord)

    @property
    def coordinates(self) -> tuple[VariableInfo, ...]:
        return tuple(variable for variable in self.variables if variable.is_coord)

    @property
    def logical_bytes(self) -> int:
        return sum(variable.logical_bytes for variable in self.data_variables)

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.dimensions[name] for name in ("time", "lat", "lon"))  # type: ignore[return-value]


@dataclass(frozen=True)
class ChunkPlan:
    strategy: Strategy
    chunks: tuple[int, int, int]
    target_mib: float
    estimated_chunk_bytes: int
    estimated_chunks: dict[str, int]
    rationale: tuple[str, ...] = ()

    @property
    def dim_chunks(self) -> dict[str, int]:
        return dict(zip(("time", "lat", "lon"), self.chunks))

    def chunks_for(self, variable: VariableInfo) -> tuple[int, ...]:
        mapping = self.dim_chunks
        return tuple(min(mapping[dim], size) for dim, size in zip(variable.dims, variable.shape))

    def label(self) -> str:
        return (
            f"{self.strategy}: chunks={self.chunks}, "
            f"target={self.target_mib:.1f} MiB"
        )


@dataclass(frozen=True)
class CompressionPlan:
    profile: CompressionProfile
    level: int | None
    description: str
    codec: CompressionCodec = "blosc-zstd"
    shuffle: ShufflePolicy = "auto"

    @property
    def enabled(self) -> bool:
        return self.codec != "none" and self.profile != "none"


    def to_dict(self) -> dict[str, object]:
        return {
            "profile": self.profile,
            "level": self.level,
            "description": self.description,
            "codec": self.codec,
            "shuffle": self.shuffle,
        }

    def label(self) -> str:
        level = "-" if self.level is None else str(self.level)
        return f"{self.codec} level {level} shuffle={self.shuffle}"


@dataclass(frozen=True)
class CompressionCandidate:
    """A generated, controlled compression choice and its rationale."""

    plan: CompressionPlan
    rationale: str = ""
    dtype: str | None = None
    chunk_shape: tuple[int, ...] | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "rationale": self.rationale,
            "dtype": self.dtype,
            "chunk_shape": list(self.chunk_shape) if self.chunk_shape is not None else None,
        }


@dataclass(frozen=True)
class CompressionBenchmarkResult:
    """Measured lossless compression result for one candidate.

    Rates are MiB/s and bytes are physical payload bytes in the trial Zarr v3
    store.  Failed or unverified candidates remain in the report but are never
    eligible for selection.
    """

    plan: CompressionPlan
    logical_bytes: int = 0
    compressed_bytes: int = 0
    encode_seconds: float = 0.0
    write_seconds: float = 0.0
    durable_seconds: float = 0.0
    hot_read_seconds: float = 0.0
    cold_read_seconds: float = 0.0
    decode_seconds: float = 0.0
    write_mib_s: float | None = None
    durable_mib_s: float | None = None
    hot_read_mib_s: float | None = None
    cold_read_mib_s: float | None = None
    average_cpu: float = 0.0
    peak_rss: int = 0
    sample_count: int = 0
    verified: bool = False
    disk_feasible: bool = True
    success: bool = False
    error: str | None = None
    score: float | None = None
    candidate_index: int = 0

    def __post_init__(self) -> None:
        logical_mib = max(int(self.logical_bytes), 0) / 1024**2
        if self.write_mib_s is None:
            object.__setattr__(
                self,
                "write_mib_s",
                logical_mib / max(float(self.write_seconds), 1e-12)
                if self.write_seconds > 0 and logical_mib > 0
                else 0.0,
            )
        if self.durable_mib_s is None:
            object.__setattr__(
                self,
                "durable_mib_s",
                logical_mib / max(float(self.durable_seconds), 1e-12)
                if self.durable_seconds > 0 and logical_mib > 0
                else 0.0,
            )
        if self.hot_read_mib_s is None:
            object.__setattr__(
                self,
                "hot_read_mib_s",
                logical_mib / max(float(self.hot_read_seconds), 1e-12)
                if self.hot_read_seconds > 0 and logical_mib > 0
                else 0.0,
            )
        if self.cold_read_mib_s is None:
            object.__setattr__(
                self,
                "cold_read_mib_s",
                logical_mib / max(float(self.cold_read_seconds), 1e-12)
                if self.cold_read_seconds > 0 and logical_mib > 0
                else 0.0,
            )

    @property
    def physical_bytes(self) -> int:
        """Compatibility alias used by the older conversion tuner."""

        return int(self.compressed_bytes)

    @property
    def compression_ratio(self) -> float:
        return self.compressed_bytes / max(self.logical_bytes, 1)

    @property
    def verified_lossless(self) -> bool:
        return bool(self.success and self.verified)

    @property
    def read_mib_s(self) -> float:
        rates = [float(self.hot_read_mib_s or 0.0), float(self.cold_read_mib_s or 0.0)]
        positive = [rate for rate in rates if rate > 0]
        return min(positive) if positive else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "plan": self.plan.to_dict(),
            "candidate_index": self.candidate_index,
            "logical_bytes": int(self.logical_bytes),
            "compressed_bytes": int(self.compressed_bytes),
            "physical_bytes": int(self.compressed_bytes),
            "compression_ratio": float(self.compression_ratio),
            "encode_seconds": float(self.encode_seconds),
            "write_seconds": float(self.write_seconds),
            "durable_seconds": float(self.durable_seconds),
            "hot_read_seconds": float(self.hot_read_seconds),
            "cold_read_seconds": float(self.cold_read_seconds),
            "decode_seconds": float(self.decode_seconds),
            "write_mib_s": float(self.write_mib_s or 0.0),
            "durable_mib_s": float(self.durable_mib_s or 0.0),
            "hot_read_mib_s": float(self.hot_read_mib_s or 0.0),
            "cold_read_mib_s": float(self.cold_read_mib_s or 0.0),
            "read_mib_s": float(self.read_mib_s),
            "average_cpu": float(self.average_cpu),
            "peak_rss": int(self.peak_rss),
            "sample_count": int(self.sample_count),
            "verified": bool(self.verified),
            "verified_lossless": bool(self.verified_lossless),
            "disk_feasible": bool(self.disk_feasible),
            "success": bool(self.success),
            "error": self.error,
            "score": None if self.score is None else float(self.score),
        }


# Short spelling retained for callers that use the noun from the public API.
CompressionBenchmark = CompressionBenchmarkResult


@dataclass(frozen=True)
class CompressionSelectionReport:
    """JSON-safe report returned by the compression tuner."""

    candidates: tuple[CompressionPlan, ...] = ()
    results: tuple[CompressionBenchmarkResult, ...] = ()
    objective: CompressionObjective = "balanced"
    baseline: CompressionPlan | None = None
    pareto_indices: tuple[int, ...] = ()
    selected: CompressionPlan | None = None
    selected_index: int | None = None
    selection_reason: str = ""
    fallback: bool = False
    cancelled: bool = False
    budget_seconds: float | None = None
    elapsed_seconds: float = 0.0
    max_samples: int = 0
    disk_free_bytes: int | None = None

    @property
    def chosen(self) -> CompressionPlan | None:
        return self.selected

    @property
    def best(self) -> CompressionPlan | None:
        return self.selected

    def to_dict(self) -> dict[str, object]:
        return {
            "candidates": [item.to_dict() for item in self.candidates],
            "results": [item.to_dict() for item in self.results],
            "objective": self.objective,
            "baseline": self.baseline.to_dict() if self.baseline is not None else None,
            "pareto_indices": list(self.pareto_indices),
            "selected": self.selected.to_dict() if self.selected is not None else None,
            "selected_index": self.selected_index,
            "selection_reason": self.selection_reason,
            "fallback": bool(self.fallback),
            "cancelled": bool(self.cancelled),
            "budget_seconds": self.budget_seconds,
            "elapsed_seconds": float(self.elapsed_seconds),
            "max_samples": int(self.max_samples),
            "disk_free_bytes": self.disk_free_bytes,
        }

    def json_dict(self) -> dict[str, object]:
        """Alias emphasizing that ``to_dict`` is safe for ``json.dumps``."""

        return self.to_dict()