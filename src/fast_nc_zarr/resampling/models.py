from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from ..models import OutputLayout
from ..rechunking.models import DatasetInfo
from ..system import EffectiveResourceBudget
from .replacements import ReplacementRules


ResampleExtent = Literal["source", "global", "custom"]
TileSize = int | Literal["auto"]
SpaceWorkers = int | Literal["auto"]
TimeBlock = int | Literal["auto"]
ComputeDType = Literal["source", "float32"]

BackendName = Literal["auto", "python", "rust"]

@dataclass(frozen=True)
class ResampleVariableOptions:
    """Resolved resampling controls for one spatial data variable."""

    method: str = "bilinear"
    skipna: bool = True
    na_thres: float = 1.0
    compute_dtype: ComputeDType = "source"



@dataclass(frozen=True)
class GridInfo:
    """Validated one-dimensional regular geographic source grid."""

    path: Path
    lat: np.ndarray
    lon: np.ndarray
    lat_bounds: np.ndarray
    lon_bounds: np.ndarray
    lat_resolution: float
    lon_resolution: float
    lat_descending: bool
    lon_descending: bool
    lat_uniform: bool
    lon_uniform: bool

    @property
    def source_extent(self) -> tuple[float, float, float, float]:
        """Return west, east, south, north cell-edge extent."""

        return (
            float(np.min(self.lon_bounds)),
            float(np.max(self.lon_bounds)),
            float(np.min(self.lat_bounds)),
            float(np.max(self.lat_bounds)),
        )

    @property
    def periodic(self) -> bool:
        west, east, south, north = self.source_extent
        del south, north
        return bool(np.isclose(east - west, 360.0, rtol=0.0, atol=1e-5))


@dataclass(frozen=True)
class TargetGrid:
    """Target cell centers and vertices in source axis orientation."""

    lat: np.ndarray
    lon: np.ndarray
    lat_bounds: np.ndarray
    lon_bounds: np.ndarray
    lat_resolution: float
    lon_resolution: float
    extent: ResampleExtent

    @property
    def dimensions(self) -> dict[str, int]:
        return {"lat": int(self.lat.size), "lon": int(self.lon.size)}

    @property
    def spatial_extent(self) -> tuple[float, float, float, float]:
        return (
            float(np.min(self.lon_bounds)),
            float(np.max(self.lon_bounds)),
            float(np.min(self.lat_bounds)),
            float(np.max(self.lat_bounds)),
        )


@dataclass(frozen=True)
class ResampleConfig:
    input: Path
    output: Path
    resolution: float = 0.25
    method: str = "bilinear"
    backend: BackendName = "auto"
    skipna: bool = True
    na_thres: float = 1.0
    compute_dtype: ComputeDType = "source"
    extent: ResampleExtent = "source"
    # Used by the pipeline when the final target area is neither the complete
    # source grid nor the global grid.  Bounds are pixel outer edges in
    # (min, max) order; the target axes are oriented by the two optional
    # direction flags below.
    target_lat_bounds: tuple[float, float] | None = None
    target_lon_bounds: tuple[float, float] | None = None
    target_lat_descending: bool | None = None
    target_lon_descending: bool | None = None
    overwrite: bool = False
    validate: bool = True
    tile_size: TileSize = "auto"
    time_block: TimeBlock = "auto"
    compute_workers: int = 2
    space_workers: SpaceWorkers = "auto"
    tuning_objective: Literal["speed", "balanced", "compact"] = "balanced"
    tune_budget: float = 60.0
    variable_options: dict[str, ResampleVariableOptions] = field(default_factory=dict)
    resource_budget: EffectiveResourceBudget | None = None
    temporary_dir: Path | None = None
    output_layout: OutputLayout | None = None
    before_replacements: ReplacementRules = field(default_factory=ReplacementRules)
    after_replacements: ReplacementRules = field(default_factory=ReplacementRules)
    statistics_policy: Literal["auto", "sample", "exact"] = "auto"


@dataclass(frozen=True)
class ResampleInspection:
    info: DatasetInfo
    grid: GridInfo
    report: str


@dataclass(frozen=True)
class AutoTileDecision:
    """Explain a memory-derived automatic spatial tile decision."""

    tile_size: int
    available_bytes: int
    budget_bytes: int
    estimated_peak_bytes: int
    source_chunk_bytes: int
    source_batch_bytes: int
    output_bytes: int
    weight_bytes: int
    source_window: tuple[int, int]
    ratio_lat: float
    ratio_lon: float
    worst_variable: str
    fits_budget: bool
    warning: str | None = None


@dataclass(frozen=True)
class ResamplePlan:
    inspection: ResampleInspection
    target: TargetGrid
    method: str
    skipna: bool
    na_thres: float
    output_chunks: dict[str, tuple[int, ...]]
    compute_dtype: ComputeDType = "source"
    tile_size: int = 128
    time_block: int = 4
    time_block_requested: TimeBlock = "auto"
    compute_workers: int = 2
    space_workers: int = 1
    tile_size_requested: TileSize = "auto"
    tuning_objective: Literal["speed", "balanced", "compact"] = "balanced"
    tune_budget: float = 60.0
    variable_options: dict[str, ResampleVariableOptions] = field(default_factory=dict)
    tuning_trials: tuple[dict[str, object], ...] = ()
    space_workers_requested: SpaceWorkers = "auto"
    auto_tile: AutoTileDecision | None = None
    owner_buffer_budget_bytes: int = 0
    resource_budget: EffectiveResourceBudget | None = None
    output_layout: OutputLayout | None = None
    before_replacements: ReplacementRules = field(default_factory=ReplacementRules)
    after_replacements: ReplacementRules = field(default_factory=ReplacementRules)
    statistics_policy: Literal["auto", "sample", "exact"] = "auto"
    statistics: dict[str, dict[str, float]] = field(default_factory=dict)

    def options_for(self, name: str) -> ResampleVariableOptions:
        return self.variable_options.get(
            name,
            ResampleVariableOptions(
                method=self.method,
                skipna=self.skipna,
                na_thres=self.na_thres,
                compute_dtype=self.compute_dtype,
            ),
        )

    @property
    def output_dimensions(self) -> dict[str, int]:
        dimensions = dict(self.inspection.info.dimensions)
        dimensions.update(self.target.dimensions)
        return dimensions
