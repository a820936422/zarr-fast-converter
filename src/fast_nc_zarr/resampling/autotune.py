from __future__ import annotations

"""Memory-aware selection of the streaming spatial tile size."""

from dataclasses import dataclass
import math
import os
import numpy as np

from ..rechunking.models import DatasetInfo, VariableInfo
from .models import AutoTileDecision, ComputeDType, GridInfo, TargetGrid


MIB = 1024**2
GIB = 1024**3
MIN_TILE_SIZE = 16
# 512 is unnecessarily restrictive for a coarse target grid whose stored
# chunks are already aligned.  The detailed memory model below remains the
# actual guardrail; this only exposes a useful larger candidate.
MAX_TILE_SIZE = 1024
MAX_AUTO_TIME_BLOCK = 64
# ESMF retains native coordinate, sparse-weight and temporary work buffers
# outside Python-managed arrays. Production conservative-resampling telemetry
# observed 4.5--4.7 GiB RSS per process for 600 x 1024 target tiles. Reserve
# 4 GiB before the explicit source/output/weight estimates below are added.
ESMF_WORKER_BASELINE_BYTES = 4 * GIB


@dataclass(frozen=True)
class _TileEstimate:
    tile_size: int
    peak_bytes: int
    source_chunk_bytes: int
    source_batch_bytes: int
    output_bytes: int
    weight_bytes: int
    source_window: tuple[int, int]
    worst_variable: str


def _memory_limits() -> tuple[int, int]:
    """Return ``(available, total)`` without making psutil mandatory here."""

    try:
        import psutil

        memory = psutil.virtual_memory()
        return max(256 * MIB, int(memory.available)), max(256 * MIB, int(memory.total))
    except ImportError:  # pragma: no cover - psutil is a project dependency
        fallback = 4 * GIB
        return fallback, fallback


def _memory_budget(available: int, total: int) -> int:
    """Keep the auto-selected working set well below system memory pressure."""

    # ``available`` already accounts for the GUI and the current Python
    # process.  The second cap prevents a mostly idle large machine from
    # giving one regridding task an unnecessarily large working set.
    # ESMF can reserve substantial native memory outside Python RSS.  Keep a
    # large system reserve, while allowing a realistic parallel group on a
    # machine with ample free memory.  The per-worker estimate below is
    # deliberately conservative and remains the controlling limit.
    budget = min(int(available * 0.60), int(total * 0.50))
    return max(256 * MIB, budget)

def resolve_owner_buffer_budget(
    *,
    space_workers: int,
    reserved_bytes: int | None = None,
    available_bytes: int | None = None,
    total_bytes: int | None = None,
) -> int:
    """Return a conservative per-worker heap budget for final-chunk buffers.

    A buffer larger than this limit is backed by a temporary memmap instead of
    allocating more resident memory.  ``reserved_bytes`` is the aggregate
    peak already estimated for all regridding workers; manual plans reserve
    the native ESMF baseline because they do not carry an automatic estimate.
    """

    detected_available, detected_total = _memory_limits()
    available = max(
        256 * MIB,
        int(detected_available if available_bytes is None else available_bytes),
    )
    total = max(256 * MIB, int(detected_total if total_bytes is None else total_bytes))
    workers = max(1, int(space_workers))
    reserved = (
        ESMF_WORKER_BASELINE_BYTES * workers
        if reserved_bytes is None
        else max(0, int(reserved_bytes))
    )
    remaining = max(0, _memory_budget(available, total) - reserved)
    pressure_cap = min(int(available * 0.05), int(total * 0.025))
    return max(0, min(256 * MIB, remaining // workers, pressure_cap // workers))


def resolve_auto_space_workers(
    *,
    compute_workers: int,
    available_bytes: int | None = None,
    total_bytes: int | None = None,
    maximum: int = 6,
) -> int:
    """Choose a bounded number of independent spatial worker processes.

    xESMF/ESMF work is CPU-heavy, so independent processes are much more
    effective than adding more Dask threads to one tile.  The limit is kept
    deliberately modest because every process owns an xESMF regridder and a
    decoded source window.  The memory cap is only a guardrail; the detailed
    tile estimate is still responsible for selecting the tile size.
    """

    detected_available, detected_total = _memory_limits()
    available = max(
        256 * MIB,
        int(detected_available if available_bytes is None else available_bytes),
    )
    total = max(256 * MIB, int(detected_total if total_bytes is None else total_bytes))
    cpu_limit = max(1, min(int(maximum), int(os.cpu_count() or 1)))
    # Keep enough headroom for the controller, filesystem cache and native
    # libraries. The detailed tile model adds decoded arrays and weights; this
    # baseline represents the ESMF process before those explicit buffers.
    per_process = (4.0 + 0.25 * max(0, int(compute_workers) - 1)) * GIB
    memory_limit = max(1, int((available * 0.60) / per_process))
    total_limit = max(1, int((total * 0.50) / per_process))
    return max(1, min(cpu_limit, memory_limit, total_limit))


def _spatial_variables(info: DatasetInfo) -> tuple[VariableInfo, ...]:
    return tuple(
        variable
        for variable in info.data_variables
        if "lat" in variable.dims and "lon" in variable.dims
    )


def _dim_chunk(variable: VariableInfo, name: str) -> int:
    try:
        index = variable.dims.index(name)
        return max(1, int(variable.chunks[index]))
    except (ValueError, IndexError):
        return 1


def _dim_size(variable: VariableInfo, name: str) -> int:
    try:
        return max(1, int(variable.shape[variable.dims.index(name)]))
    except (ValueError, IndexError):
        return 1


def _time_chunk(variable: VariableInfo) -> int:
    return _dim_chunk(variable, "time") if "time" in variable.dims else 1


def _time_size(variable: VariableInfo) -> int:
    return _dim_size(variable, "time") if "time" in variable.dims else 1


def resolve_auto_time_block(
    info: DatasetInfo,
    grid: GridInfo,
    target: TargetGrid,
    *,
    method: str,
    skipna: bool,
    compute_dtype: ComputeDType = "source",
) -> int:
    """Choose a bounded vectorized time batch for one spatial tile.

    A source store with ``time`` chunks of one is common for stacks of GeoTIFF
    files.  Treating that storage detail as the computation-batch upper bound
    turns a multi-year product into thousands of tiny xESMF calls.  A batch
    may therefore span stored time chunks; the execution engine materializes
    the selected batch once before calling xESMF.  The cap keeps that
    materialization safe while still recovering xESMF's vectorized time path.
    """

    spatial_variables = _spatial_variables(info)
    if not spatial_variables:
        return 1
    time_sizes = [
        _time_size(variable)
        for variable in spatial_variables
        if "time" in variable.dims
    ]
    maximum = max(1, min(MAX_AUTO_TIME_BLOCK, min(time_sizes or [1])))
    available, _total = _memory_limits()
    batch_budget = max(256 * MIB, min(512 * MIB, int(available * 0.05)))
    required_per_time: list[int] = []
    for variable in spatial_variables:
        target_lat = min(int(target.lat.size), _dim_chunk(variable, "lat"))
        target_lon = min(int(target.lon.size), _dim_chunk(variable, "lon"))
        source_lat, source_lon = _source_window(
            target_lat,
            target_lon,
            grid,
            target,
            method,
        )
        source_lat = _aligned_size(
            source_lat,
            _dim_chunk(variable, "lat"),
            _dim_size(variable, "lat"),
        )
        source_lon = _aligned_size(
            source_lon,
            _dim_chunk(variable, "lon"),
            _dim_size(variable, "lon"),
        )
        source_bytes = source_lat * source_lon * _working_itemsize(
            variable, compute_dtype
        )
        result_bytes = (
            target_lat
            * target_lon
            * max(
                8 if skipna else 1,
                _result_itemsize(variable, compute_dtype),
            )
            * 2
        )
        required_per_time.append(
            int(source_bytes * (2 if skipna else 1) + result_bytes)
        )
    per_time = max(required_per_time, default=1)
    allowed = max(1, min(maximum, batch_budget // per_time))
    candidates = {1, maximum}
    candidates.update(
        value for value in (2**power for power in range(1, 16)) if value <= maximum
    )
    fitting = [value for value in sorted(candidates) if value <= allowed]
    return int(fitting[-1] if fitting else 1)


def _source_window(
    tile_lat: int,
    tile_lon: int,
    grid: GridInfo,
    target: TargetGrid,
    method: str,
) -> tuple[int, int]:
    """Estimate source cells touched by a target tile.

    The execution engine reads only the local longitude window for a
    periodic source grid.  ``nearest_d2s`` is the exception: its global
    destination-to-source assignment requires the complete source grid.
    """

    if method == "nearest_d2s":
        return int(grid.lat.size), int(grid.lon.size)

    # The execution path uses a method-specific stencil halo.  Keep a minimum
    # two-cell estimate so floating-point edge comparisons and conservative
    # bounds remain covered without overestimating every method by four cells.
    halo = max(
        2,
        {
            "bilinear": 1,
            "nearest_s2d": 1,
            "conservative": 0,
            "conservative_normed": 0,
            "patch": 2,
            "nearest_d2s": 0,
        }.get(method, 1),
    )
    lat_ratio = target.lat_resolution / max(grid.lat_resolution, np.finfo(float).eps)
    lon_ratio = target.lon_resolution / max(grid.lon_resolution, np.finfo(float).eps)
    lat_cells = min(int(grid.lat.size), max(1, int(math.ceil(tile_lat * lat_ratio)) + halo))
    lon_cells = min(int(grid.lon.size), max(1, int(math.ceil(tile_lon * lon_ratio)) + halo))
    return lat_cells, lon_cells


def _aligned_size(size: int, chunk: int, dimension: int) -> int:
    return min(int(dimension), max(int(chunk), int(math.ceil(size / chunk)) * int(chunk)))


def _aligned_target_size(tile_size: int, chunk: int, dimension: int) -> int:
    """Model the largest complete output-chunk block within ``tile_size``.

    A processing tile is allowed to be smaller than the configured square
    size in one direction.  This is important for datasets such as GOSIF,
    whose output longitude chunks are 416 cells wide: a 512-cell tile is in
    practice 512 x 416, not 512 x 832.
    """

    dimension = max(1, int(dimension))
    chunk = max(1, int(chunk))
    tile_size = max(1, int(tile_size))
    if dimension <= tile_size:
        return dimension
    if chunk > tile_size:
        return min(dimension, chunk)
    return max(chunk, min(dimension, (tile_size // chunk) * chunk))


def _effective_dtype(variable: VariableInfo, compute_dtype: ComputeDType) -> np.dtype:
    dtype = np.dtype(variable.dtype)
    if compute_dtype == "float32" and np.issubdtype(dtype, np.floating):
        return np.dtype("float32")
    return dtype


def _working_itemsize(
    variable: VariableInfo,
    compute_dtype: ComputeDType = "source",
) -> int:
    dtype = _effective_dtype(variable, compute_dtype)
    # This is the decoded source-window dtype.  xESMF's output and missing
    # value intermediates are accounted for separately in ``_estimate_tile``;
    # retaining the effective input dtype here lets float32 mode actually
    # reduce the source-window estimate.
    if np.issubdtype(dtype, np.floating):
        return int(dtype.itemsize)
    return 8


def _result_itemsize(
    variable: VariableInfo,
    compute_dtype: ComputeDType = "source",
) -> int:
    dtype = _effective_dtype(variable, compute_dtype)
    if np.issubdtype(dtype, np.floating):
        return int(dtype.itemsize)
    return 8


def _stencil_size(method: str) -> int:
    return {
        "bilinear": 4,
        "conservative": 8,
        "conservative_normed": 8,
        "patch": 9,
        "nearest_s2d": 1,
        "nearest_d2s": 1,
    }.get(method, 8)


def _estimate_tile(
    tile_size: int,
    info: DatasetInfo,
    grid: GridInfo,
    target: TargetGrid,
    method: str,
    skipna: bool,
    time_block: int,
    compute_workers: int,
    space_workers: int,
    compute_dtype: ComputeDType,
) -> _TileEstimate:
    spatial_variables = _spatial_variables(info)
    if not spatial_variables:
        return _TileEstimate(
            tile_size=tile_size,
            peak_bytes=256 * MIB,
            source_chunk_bytes=0,
            source_batch_bytes=0,
            output_bytes=0,
            weight_bytes=0,
            source_window=(0, 0),
            worst_variable="无空间数据变量",
        )

    stencil = _stencil_size(method)

    worst: _TileEstimate | None = None
    for variable in spatial_variables:
        target_lat = _aligned_target_size(
            tile_size,
            _dim_chunk(variable, "lat"),
            target.lat.size,
        )
        target_lon = _aligned_target_size(
            tile_size,
            _dim_chunk(variable, "lon"),
            target.lon.size,
        )
        target_cells = target_lat * target_lon
        # Sparse weights contain a value, source index and destination index;
        # multiply by two for ESMF/xarray bookkeeping and index conversion.
        weight_target_cells = (
            int(target.lat.size) * int(target.lon.size)
            if method == "nearest_d2s"
            else target_cells
        )
        weight_bytes = weight_target_cells * stencil * 24
        source_lat, source_lon = _source_window(
            target_lat,
            target_lon,
            grid,
            target,
            method,
        )
        aligned_lat = _aligned_size(
            source_lat,
            _dim_chunk(variable, "lat"),
            _dim_size(variable, "lat"),
        )
        aligned_lon = _aligned_size(
            source_lon,
            _dim_chunk(variable, "lon"),
            _dim_size(variable, "lon"),
        )
        stored_time = _time_chunk(variable)
        requested_time = min(max(1, int(time_block)), _time_size(variable))
        source_chunk_bytes = (
            stored_time
            * _dim_chunk(variable, "lat")
            * _dim_chunk(variable, "lon")
            * int(np.dtype(variable.dtype).itemsize)
        )
        source_chunk_count = max(
            1,
            int(math.ceil(aligned_lat / _dim_chunk(variable, "lat")))
            * int(math.ceil(aligned_lon / _dim_chunk(variable, "lon"))),
        )
        decoded_source_bytes = source_chunk_bytes * source_chunk_count
        source_batch_bytes = (
            requested_time
            * aligned_lat
            * aligned_lon
            * _working_itemsize(variable, compute_dtype)
        )
        output_bytes = (
            requested_time
            * target_cells
            * max(
                8 if skipna else 1,
                _result_itemsize(variable, compute_dtype),
            )
        )

        # A worker can hold one decoded Zarr chunk, a source window, a
        # missing-value mask and a result at the same time.  The final factor
        # covers Dask task objects and ESMF temporary buffers.  This is an
        # intentionally conservative estimate, but it models concurrent
        # decoded chunks rather than multiplying the entire source window by
        # every source chunk it intersects.
        per_worker = (
            # ESMF owns native allocations that are not represented by the
            # Dask/Zarr arrays below.  A fixed baseline makes automatic tile
            # selection match observed worker RSS much more closely.
            ESMF_WORKER_BASELINE_BYTES
            + decoded_source_bytes
            + source_batch_bytes
            * (1.25 if skipna else 1.0)
            * (1.0 + 0.25 * max(0, int(compute_workers) - 1))
            + output_bytes * 2
            + weight_bytes
        )
        peak_bytes = int(
            per_worker
            * max(1, int(space_workers))
            * 1.1
        )
        estimate = _TileEstimate(
            tile_size=tile_size,
            peak_bytes=peak_bytes,
            source_chunk_bytes=decoded_source_bytes,
            source_batch_bytes=source_batch_bytes,
            output_bytes=output_bytes,
            weight_bytes=weight_bytes,
            source_window=(aligned_lat, aligned_lon),
            worst_variable=variable.name,
        )
        if worst is None or estimate.peak_bytes > worst.peak_bytes:
            worst = estimate
    assert worst is not None
    return worst


def _candidate_sizes(target: TargetGrid) -> tuple[int, ...]:
    maximum = max(1, int(target.lat.size), int(target.lon.size))
    values = {
        min(maximum, value)
        for value in range(MIN_TILE_SIZE, MAX_TILE_SIZE + 1)
        if value & (value - 1) == 0
    }
    values.add(maximum if maximum < MIN_TILE_SIZE else min(maximum, MAX_TILE_SIZE))
    return tuple(sorted(values))


def resolve_auto_tile_size(
    info: DatasetInfo,
    grid: GridInfo,
    target: TargetGrid,
    *,
    method: str,
    skipna: bool,
    time_block: int,
    compute_workers: int,
    space_workers: int = 1,
    compute_dtype: ComputeDType = "source",
    available_bytes: int | None = None,
    total_bytes: int | None = None,
) -> AutoTileDecision:
    """Choose the largest candidate whose estimated working set fits budget."""

    detected_available, detected_total = _memory_limits()
    available = max(
        256 * MIB,
        int(detected_available if available_bytes is None else available_bytes),
    )
    total = max(256 * MIB, int(detected_total if total_bytes is None else total_bytes))
    budget = _memory_budget(available, total)
    candidates = _candidate_sizes(target)
    estimates = [
        _estimate_tile(
            candidate,
            info,
            grid,
            target,
            method,
            skipna,
            time_block,
            compute_workers,
            space_workers,
            compute_dtype,
        )
        for candidate in candidates
    ]
    fitting = [estimate for estimate in estimates if estimate.peak_bytes <= budget]
    selected = fitting[-1] if fitting else estimates[0]
    ratio_lat = target.lat_resolution / max(grid.lat_resolution, np.finfo(float).eps)
    ratio_lon = target.lon_resolution / max(grid.lon_resolution, np.finfo(float).eps)
    fits = selected.peak_bytes <= budget
    warning = None
    if not fits:
        warning = (
            "即使使用最小自动空间块，估算峰值仍超过自动内存预算；"
            "建议先使用 Zarr 优化模块减小源 time/空间 chunk，或降低空间进程/块内线程数。"
        )
    return AutoTileDecision(
        tile_size=selected.tile_size,
        available_bytes=available,
        budget_bytes=budget,
        estimated_peak_bytes=selected.peak_bytes,
        source_chunk_bytes=selected.source_chunk_bytes,
        source_batch_bytes=selected.source_batch_bytes,
        output_bytes=selected.output_bytes,
        weight_bytes=selected.weight_bytes,
        source_window=selected.source_window,
        ratio_lat=float(ratio_lat),
        ratio_lon=float(ratio_lon),
        worst_variable=selected.worst_variable,
        fits_budget=fits,
        warning=warning,
    )
