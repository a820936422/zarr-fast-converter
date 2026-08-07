from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ..models import OutputLayout, Selection, VariableTransform
from ..rechunking.models import ChunkPlan, CompressionPlan
from ..resampling.models import TargetGrid


@dataclass(frozen=True)
class PipelineGeneralConfig:
    """The five common fields plus the V1 temporary-store policy."""

    output: Path
    temporary_dir: Path | None = None
    time_start: str | None = None
    time_end: str | None = None
    lat_min: float = -90.0
    lat_max: float = 90.0
    lon_min: float = -180.0
    lon_max: float = 180.0
    resolution: float = 0.1
    cleanup_intermediate: bool = False
    overwrite: bool = False


@dataclass(frozen=True)
class PipelineConversionOptions:
    variables: tuple[str, ...] = ()
    variable_names: dict[str, str] = field(default_factory=dict)
    variable_transforms: dict[str, VariableTransform] = field(default_factory=dict)
    auto_tune: bool = True
    tune_budget: float = 60.0
    max_workers: int | None = None
    reserve_memory_gib: float = 2.0


@dataclass(frozen=True)
class PipelineResamplingOptions:
    method: str = "bilinear"
    skipna: bool = True
    na_thres: float = 1.0
    compute_dtype: Literal["source", "float32"] = "source"
    tile_size: int | Literal["auto"] = "auto"
    time_block: int | Literal["auto"] = "auto"
    compute_workers: int = 2
    space_workers: int | Literal["auto"] = "auto"


@dataclass(frozen=True)
class PipelineFinalizationOptions:
    strategy: Literal["time", "space", "custom"] = "time"
    target_mib: float = 128.0
    custom_chunks: tuple[int, int, int] | None = None
    compression: Literal["fast", "balanced", "maximum", "none"] = "balanced"
    workers: int = 1


@dataclass(frozen=True)
class PipelineConfig:
    general: PipelineGeneralConfig
    conversion: PipelineConversionOptions = field(default_factory=PipelineConversionOptions)
    resampling: PipelineResamplingOptions = field(default_factory=PipelineResamplingOptions)
    finalization: PipelineFinalizationOptions = field(default_factory=PipelineFinalizationOptions)
    validate: bool = True


@dataclass(frozen=True)
class SourceReadWindow:
    """Source indices retained for conversion before target-grid resampling."""

    lat_start: int
    lat_stop: int
    lon_start: int
    lon_stop: int
    lat_bounds: tuple[float, float]
    lon_bounds: tuple[float, float]
    method: str
    halo_description: str

    @property
    def lat_shape(self) -> int:
        return self.lat_stop - self.lat_start

    @property
    def lon_shape(self) -> int:
        return self.lon_stop - self.lon_start


@dataclass(frozen=True)
class PipelinePlan:
    inspection_id: str
    target_grid: TargetGrid
    source_read_window: SourceReadWindow
    source_selection: Selection
    needs_resample: bool
    coverage_warning: str | None = None
    conversion_chunks: tuple[int, int, int] | None = None
    final_chunks: tuple[int, int, int] | None = None
    final_chunk_plan: ChunkPlan | None = None
    final_compression: CompressionPlan | None = None
    output_layout: OutputLayout | None = None
    direct_finalization: bool = False


@dataclass(frozen=True)
class PipelinePaths:
    root: Path
    manifest: Path
    converted: Path
    resampled: Path
    final_staging: Path
