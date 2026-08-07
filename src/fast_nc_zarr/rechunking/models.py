from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np


Strategy = Literal["time", "space", "custom"]
CompressionProfile = Literal["none", "fast", "balanced", "maximum", "custom"]
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
