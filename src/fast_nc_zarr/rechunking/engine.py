from __future__ import annotations

import json
import gc
import itertools
import os
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Iterable, Iterator
from uuid import uuid4

import dask.array as dask_array
import numpy as np
import psutil
import xarray as xr
import zarr

from ..metadata import sanitize_cf_references
from .compression import codec_for, make_compression_plan
from .inspection import inspect_store
from .models import ChunkPlan, CompressionPlan, DatasetInfo
from ..publication import publish_staging
from ..runtime import bounded_process_map
from ..system import storage_profile


class RechunkExecutionError(RuntimeError):
    """Raised when a rechunk operation cannot safely complete."""


def _is_zarr_v3(path: Path) -> bool:
    metadata = path / "zarr.json"
    if not metadata.is_file():
        return False
    try:
        value = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("node_type") == "group" and value.get("zarr_format") == 3


def next_available_output(path: str | Path) -> Path:
    """Return a non-existing sibling path with a deterministic suffix."""

    candidate = Path(path).expanduser()
    if not candidate.exists():
        return candidate
    if candidate.suffix == ".zarr":
        stem = candidate.with_suffix("")
        suffix = candidate.suffix
    else:
        stem = candidate
        suffix = ""
    index = 1
    while True:
        alternative = stem.with_name(f"{stem.name}_rechunked_{index}{suffix}")
        if not alternative.exists():
            return alternative
        index += 1


def _prepare_target(path: Path, overwrite: bool) -> None:
    if path.is_symlink():
        raise RechunkExecutionError(f"拒绝将输出路径写入符号链接：{path}")
    if path.exists() and not path.is_dir():
        raise RechunkExecutionError(f"输出路径存在但不是目录：{path}")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        return
    if any(path.iterdir()):
        if not overwrite:
            raise RechunkExecutionError(
                f"输出目录非空：{path}；请确认覆盖或使用新的输出目录。"
            )
        if not _is_zarr_v3(path):
            raise RechunkExecutionError(
                "拒绝覆盖普通非空目录；只有已识别的 Zarr v3 目录可以覆盖。"
            )
    path.parent.mkdir(parents=True, exist_ok=True)


def _resolve_temporary_root(
    source: Path,
    target: Path,
    temporary_dir: str | Path | None,
) -> Path:
    """Resolve and create the directory used for intermediate Zarr stores."""

    root = (
        target.parent
        if temporary_dir is None
        else Path(temporary_dir).expanduser().resolve()
    )
    if root == source or root == target:
        raise RechunkExecutionError("临时处理目录不能是输入或输出 Zarr 本身。")
    if root.exists() and not root.is_dir():
        raise RechunkExecutionError(f"临时处理路径不是目录：{root}")
    root.mkdir(parents=True, exist_ok=True)
    return root


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def _encoding(
    dataset: xr.Dataset,
    info: DatasetInfo,
    plan: ChunkPlan,
    compression: CompressionPlan,
    *,
    shards: dict[str, tuple[int, ...]] | None = None,
) -> dict[str, dict[str, object]]:
    by_name = {variable.name: variable for variable in info.variables}
    encoding: dict[str, dict[str, object]] = {}
    for name, variable in dataset.variables.items():
        item = by_name[name]
        entry: dict[str, object] = {}
        # xarray's CF encoder expects ``_FillValue``.  The source marker may
        # already be in encoding (after cleaning) or remain in the inspection
        # attrs; copy only explicit markers, never Zarr's implicit integer
        # ``fill_value=0`` default.
        fill_value = variable.encoding.get("_FillValue")
        if fill_value is None:
            # ``fill_value`` is Zarr's internal default and is often zero for
            # integer arrays even when the source has no missing-value
            # convention.  Only copy it when the source explicitly declared
            # an xarray/CF _FillValue marker.
            if "_FillValue" in item.attrs:
                fill_value = item.attrs["_FillValue"]
        if fill_value is not None:
            # Keep xarray's serialization marker out of attrs/encoding
            # conflicts when coordinates are written in the first pass.
            entry["_FillValue"] = fill_value
        missing_value = variable.encoding.get("missing_value")
        if missing_value is not None:
            entry["missing_value"] = missing_value
        if item.ndim:
            entry["chunks"] = plan.chunks_for(item)
            # Zarr v3 sharding keeps the logical chunk grid unchanged while
            # packing several small chunks into one physical file.  This is
            # used only for the intermediate store; the final store remains
            # a regular chunked Zarr array for broad reader compatibility.
            if shards is not None and item.name in shards:
                entry["shards"] = shards[item.name]
        if compression.enabled:
            codec = codec_for(item, compression, coordinate=item.is_coord)
            if codec is not None:
                entry["compressors"] = [codec]
        elif item.compressors:
            entry["compressors"] = list(item.compressors)
        encoding[name] = entry
    return encoding


def _rechunk_limit_bytes(info: DatasetInfo, workers: int) -> int:
    """Choose a bounded Dask intermediate block size.

    The source store may already contain a chunk larger than the requested
    target.  That source chunk is the irreducible read unit, but subsequent
    intermediate blocks must not grow without a limit.
    """

    source_bytes = 0
    for variable in info.data_variables:
        if variable.ndim != 3:
            continue
        source_bytes = max(
            source_bytes,
            int(np.prod(variable.chunks, dtype=np.int64)) * variable.dtype.itemsize,
        )
    available = psutil.virtual_memory().available
    # Keep the aggregate Dask working set well below available memory.
    budget = int(available * 0.05 / max(1, workers))
    return max(source_bytes, min(256 * 1024**2, max(64 * 1024**2, budget)))


def _safe_workers(info: DatasetInfo, requested: int) -> int:
    """Cap concurrent region tasks using the irreducible source chunk size."""

    requested = max(1, int(requested))
    source_bytes = max(
        (
            int(np.prod(variable.chunks, dtype=np.int64)) * variable.dtype.itemsize
            for variable in info.data_variables
            if variable.ndim == 3
        ),
        default=1,
    )
    available = max(1, int(psutil.virtual_memory().available))
    # Reserve most memory for the OS, xarray and compression buffers.  Eight
    # workers is a practical upper bound on this threaded pipeline; the
    # memory-derived cap additionally protects users who pass a very large
    # --workers value on low-memory machines.
    memory_workers = max(1, int(available * 0.10 / source_bytes))
    cpu_cap = min(os.cpu_count() or 1, 8)
    return max(1, min(requested, cpu_cap, memory_workers))


_CONSERVATIVE_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "smb",
        "smb2",
        "smb3",
        "sshfs",
    }
)


def _conservative_filesystem(filesystem: str) -> bool:
    value = str(filesystem).strip().lower()
    return value in _CONSERVATIVE_FILESYSTEMS or value.startswith("fuse")


def _parallel_workers(
    source: Path,
    target_parent: Path,
    requested: int,
) -> tuple[int, str]:
    """Choose a bounded process count from shared storage profiles.

    Network and userspace filesystems are deliberately conservative: adding
    processes there often increases metadata round trips rather than useful
    throughput.  Local SSDs retain the CPU-derived allowance, while rotational
    media keep the existing seek-avoidance caps.
    """

    requested = max(1, int(requested))
    source_profile = storage_profile(source)
    target_profile = storage_profile(target_parent)
    same_device = (
        source_profile.device != "unknown"
        and target_profile.device != "unknown"
        and source_profile.device == target_profile.device
    )
    base = min(requested, 8, os.cpu_count() or 1)
    conservative = tuple(
        profile.filesystem
        for profile in (source_profile, target_profile)
        if _conservative_filesystem(profile.filesystem)
    )
    if conservative:
        filesystems = ", ".join(dict.fromkeys(conservative))
        return max(1, min(base, 2)), (
            f"检测到保守文件系统（{filesystems}），限制为 2 个进程"
        )
    if same_device and (
        source_profile.rotational is True or target_profile.rotational is True
    ):
        return max(1, min(base, 2)), "源和目标位于同一块机械硬盘，限制为 2 个进程"
    if source_profile.rotational is True or target_profile.rotational is True:
        return max(1, min(base, 3)), "检测到机械硬盘，限制为 3 个进程"
    if same_device:
        return max(1, min(base, 4)), "源和目标同设备，限制为 4 个进程"
    return max(1, base), "源和目标设备可并行，按内存和 CPU 使用进程"


def _json_signature(value: object) -> str:
    def default(item: object) -> object:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        raise TypeError(f"无法序列化 {type(item).__name__}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=True,
        default=default,
    )


def _codec_signature(codec: object) -> tuple[str, str] | None:
    serializer = getattr(codec, "to_dict", None)
    if not callable(serializer):
        return None
    try:
        payload = serializer()
        signature = _json_signature(payload)
    except (TypeError, ValueError):
        return None
    codec_type = f"{type(codec).__module__}.{type(codec).__qualname__}"
    return codec_type, signature


def _compression_matches_source(
    info: DatasetInfo,
    compression: CompressionPlan,
) -> bool:
    """Return true only when every requested compressor is provably identical."""

    if not compression.enabled:
        return True
    for variable in info.variables:
        desired = codec_for(variable, compression, coordinate=variable.is_coord)
        if desired is None or len(variable.compressors) != 1:
            return False
        expected = _codec_signature(desired)
        actual = _codec_signature(variable.compressors[0])
        if expected is None or actual is None or expected != actual:
            return False
    return True


def _chunks_match_plan(
    info: DatasetInfo,
    plan: ChunkPlan,
    *,
    data_only: bool,
) -> bool:
    variables = info.data_variables if data_only else info.variables
    return all(
        variable.chunks == plan.chunks_for(variable)
        for variable in variables
        if variable.ndim
    )


def _metadata_unchanged(dataset: xr.Dataset, info: DatasetInfo) -> bool:
    """Check that CF sanitization did not alter metadata before byte copying."""

    try:
        if _json_signature(dict(dataset.attrs)) != _json_signature(info.attrs):
            return False
        source_variables = {variable.name: variable for variable in info.variables}
        if set(dataset.variables) != set(source_variables):
            return False
        return all(
            _json_signature(dict(dataset[name].attrs))
            == _json_signature(source_variables[name].attrs)
            for name in dataset.variables
        )
    except (TypeError, ValueError):
        # Unknown attribute types must take the decode/encode path; a false
        # negative only costs performance, while a false positive is unsafe.
        return False


def _copy_equivalent_store(
    source: Path,
    staging: Path,
    *,
    cancel_event=None,
) -> None:
    """Copy an equivalent store into staging without sharing mutable inodes."""

    if staging.exists():
        raise RechunkExecutionError(f"快速复制暂存目录已存在：{staging}")

    for root, directories, files in os.walk(source, followlinks=False):
        for name in (*directories, *files):
            entry = Path(root) / name
            if entry.is_symlink():
                raise RechunkExecutionError(
                    f"等价复制拒绝包含符号链接的源 Zarr：{entry}"
                )

    def copy_file(source_file: str, target_file: str) -> str:
        if cancel_event is not None and cancel_event.is_set():
            raise RechunkExecutionError("任务已取消。")
        # copy2 always creates an independent destination file.  In
        # particular, never use hard links for mutable Zarr chunk payloads.
        return shutil.copy2(source_file, target_file)

    if cancel_event is not None and cancel_event.is_set():
        raise RechunkExecutionError("任务已取消。")
    shutil.copytree(source, staging, copy_function=copy_file)


def _direct_region_chunks(info: DatasetInfo) -> dict[str, tuple[int, ...]]:
    return {
        variable.name: tuple(int(value) for value in variable.chunks)
        for variable in info.data_variables
        if variable.ndim == 3
    }


def _stage2_safe_workers(
    info: DatasetInfo,
    plan: ChunkPlan,
    region_chunks: dict[str, tuple[int, ...]],
    compression: CompressionPlan,
    requested: int,
) -> tuple[int, int]:
    """Bound stage-2 concurrency by decoded regions and codec peak buffers."""

    requested = max(1, int(requested))
    codec_workers = max(1, min(2, requested))
    peak_bytes = 1
    for variable in info.data_variables:
        if variable.ndim != 3:
            continue
        region = region_chunks.get(variable.name, variable.chunks)
        region_bytes = int(
            np.prod(
                [
                    min(int(size), int(chunk))
                    for size, chunk in zip(variable.shape, region)
                ],
                dtype=np.int64,
            )
        ) * int(variable.dtype.itemsize)
        final_bytes = int(
            np.prod(plan.chunks_for(variable), dtype=np.int64)
        ) * int(variable.dtype.itemsize)
        # Dask may transiently retain both decoded inputs and the assembled
        # region.  Each codec lane additionally needs an input and a worst-case
        # encoded output buffer for one final physical chunk.
        has_codec = compression.enabled or bool(variable.compressors)
        codec_bytes = (
            2 * final_bytes * codec_workers if has_codec else final_bytes
        )
        peak_bytes = max(peak_bytes, 2 * region_bytes + codec_bytes)
    available = max(1, int(psutil.virtual_memory().available))
    memory_workers = max(1, int(available * 0.25) // peak_bytes)
    return max(1, min(requested, memory_workers)), peak_bytes


_FUSED_CODEC_PIPELINE = "zarr.core.codec_pipeline.FusedCodecPipeline"
_BATCHED_CODEC_PIPELINE = "zarr.core.codec_pipeline.BatchedCodecPipeline"
_INTERMEDIATE_SHARD_THRESHOLD = 1_000_000


def _bounded_rechunk_dataset(
    dataset: xr.Dataset,
    info: DatasetInfo,
    plan: ChunkPlan,
    *,
    workers: int,
) -> xr.Dataset:
    """Create a Dask-backed view with explicit intermediate rechunk limits."""

    result = dataset.copy(deep=False)
    # xarray exposes Zarr fill values in attrs when mask_and_scale=False.
    # Move them to encoding before writing, otherwise the CF encoder sees the
    # same key in both places and aborts.
    for name in result.variables:
        variable = result[name]
        attrs = dict(variable.attrs)
        encoding = dict(variable.encoding)
        # xarray's CF coders treat these two attributes as serialization
        # markers.  Keeping them in attrs while writing causes a hard error
        # ("Key ... already exists in attrs").  Move them to encoding; the
        # encoder writes the same metadata back to the Zarr array.
        for marker in ("_FillValue", "missing_value"):
            if marker in attrs:
                value = attrs.pop(marker)
                encoding.setdefault(marker, value)
        variable.attrs = attrs
        variable.encoding = encoding
    limit = _rechunk_limit_bytes(info, workers)
    for variable in info.data_variables:
        if variable.name not in result.variables:
            continue
        if variable.ndim != 3:
            continue
        data = result[variable.name].data
        target_chunks = plan.chunks_for(variable)
        if hasattr(data, "rechunk"):
            result[variable.name].data = data.rechunk(
                target_chunks,
                threshold=1,
                block_size_limit=limit,
                method="tasks",
            )
        else:
            result[variable.name].data = dask_array.from_array(
                data,
                chunks=variable.chunks,
                lock=False,
            ).rechunk(
                target_chunks,
                threshold=1,
                block_size_limit=limit,
                method="tasks",
            )
    return result


def _clean_for_region(variable: xr.DataArray) -> xr.DataArray:
    """Make a data block safe for xarray's CF/Zarr encoders."""

    result = variable.copy(deep=False)
    attrs = dict(result.attrs)
    for marker in ("_FillValue", "missing_value"):
        if marker in attrs:
            value = attrs.pop(marker)
            encoding = dict(result.encoding)
            encoding.setdefault(marker, value)
            result.encoding = encoding
    result.attrs = attrs
    return result


def _initialize_store(
    dataset: xr.Dataset,
    info: DatasetInfo,
    plan: ChunkPlan,
    compression: CompressionPlan,
    path: Path,
    *,
    shards: dict[str, tuple[int, ...]] | None = None,
) -> None:
    """Create a Zarr v3 store and all array metadata without writing data."""

    metadata_names = [
        variable.name
        for variable in info.variables
        if variable.is_coord or variable.ndim != 3
    ]
    metadata = (
        _bounded_rechunk_dataset(dataset[metadata_names], info, plan, workers=1)
        if metadata_names
        else xr.Dataset(attrs=dict(dataset.attrs))
    )
    metadata_encoding = _encoding(
        metadata,
        info,
        plan,
        compression,
        shards=shards,
    )
    metadata.to_zarr(
        path,
        mode="w",
        consolidated=False,
        compute=True,
        encoding=metadata_encoding,
        zarr_format=3,
    )
    del metadata, metadata_encoding
    gc.collect()

    for variable in info.data_variables:
        if variable.ndim != 3:
            continue
        source_variable = _clean_for_region(dataset[variable.name])
        # Keep the placeholder as one lazy chunk.  The encoding below defines
        # the real Zarr chunks; using the full target chunk grid here would
        # construct hundreds of thousands of unnecessary Dask tasks merely to
        # create metadata for the intermediate store.
        template_data = dask_array.empty(
            variable.shape,
            chunks=variable.shape,
            dtype=variable.dtype,
        )
        template = xr.Dataset(
            {
                variable.name: xr.DataArray(
                    template_data,
                    dims=variable.dims,
                    attrs=dict(source_variable.attrs),
                )
            },
            attrs=dict(dataset.attrs),
        )
        template[variable.name].encoding = dict(source_variable.encoding)
        variable_encoding = _encoding(
            template,
            info,
            plan,
            compression,
            shards=shards,
        )
        delayed = template.to_zarr(
            path,
            mode="a",
            consolidated=False,
            compute=False,
            encoding={variable.name: variable_encoding[variable.name]},
            zarr_format=3,
        )
        # Only the synchronously-created metadata is needed.  The empty data
        # graph must never be computed.
        del delayed, template, template_data, source_variable, variable_encoding
        gc.collect()


def _chunk_regions(variable: xr.DataArray) -> Iterator[dict[str, slice]]:
    """Chunk-region helper that avoids repeated index searches for large arrays."""

    data = variable.data
    chunks = getattr(data, "chunks", None)
    if chunks is None:
        chunks = tuple((int(size),) for size in variable.shape)
    starts = [
        tuple(int(value) for value in np.cumsum((0,) + tuple(axis_chunks[:-1])))
        for axis_chunks in chunks
    ]
    for indices in itertools.product(*(range(len(axis)) for axis in chunks)):
        yield {
            dim: slice(
                starts[axis][index],
                starts[axis][index] + int(chunks[axis][index]),
            )
            for axis, (dim, index) in enumerate(zip(variable.dims, indices))
        }


def _source_chunk_indices(variable: xr.DataArray) -> Iterator[tuple[int, ...]]:
    """Return the physical chunk coordinates in the same order as regions.

    ``xarray.DataArray.isel`` rebuilds an indexing graph for every source
    block.  Native Zarr inputs are backed by Dask arrays, so selecting the
    block through ``.blocks`` avoids that repeated graph construction and
    reads exactly one physical source chunk.
    """

    chunks = getattr(variable.data, "chunks", None)
    if chunks is None:
        chunks = tuple((int(size),) for size in variable.shape)
    return (
        tuple(int(index) for index in indices)
        for indices in itertools.product(*(range(len(axis)) for axis in chunks))
    )


def _array_chunk_count(variable: xr.DataArray) -> int:
    chunks = getattr(variable.data, "chunks", None)
    if chunks is None:
        return 1
    return int(np.prod([len(axis) for axis in chunks], dtype=np.int64))


def _chunk_layout(
    variable: xr.DataArray,
) -> tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    """Return chunk lengths and cumulative starts for a data array."""

    chunks = getattr(variable.data, "chunks", None)
    if chunks is None:
        chunks = tuple((int(size),) for size in variable.shape)
    normalized = tuple(
        tuple(int(value) for value in axis_chunks) for axis_chunks in chunks
    )
    starts = tuple(
        tuple(int(value) for value in np.cumsum((0,) + axis_chunks[:-1]))
        for axis_chunks in normalized
    )
    return normalized, starts


def _region_for_chunk_indices(
    dims: tuple[str, ...],
    chunks: tuple[tuple[int, ...], ...],
    starts: tuple[tuple[int, ...], ...],
    indices: tuple[int, ...],
) -> dict[str, slice]:
    return {
        dim: slice(
            starts[axis][index],
            starts[axis][index] + chunks[axis][index],
        )
        for axis, (dim, index) in enumerate(zip(dims, indices))
    }


def _write_intermediate_source_chunk(
    array: zarr.Array,
    block: np.ndarray,
    source_region: dict[str, slice],
    variable: VariableInfo,
) -> None:
    """Write one complete source block in a single Zarr selection.

    The previous implementation submitted every overlapping target chunk as a
    separate ``Array.__setitem__`` call.  For a FLUXSAT source chunk this is
    roughly 276 Python/Zarr calls, each with its own event-loop and filesystem
    bookkeeping.  Zarr can split one larger selection into chunk writes itself
    and its fused codec pipeline can process that batch concurrently.  Keeping
    the whole source block in one call removes most of that overhead while
    preserving the one-source-chunk-read invariant.
    """

    # ``source_region`` follows the variable's dimension order.  The source
    # and intermediate arrays have identical dimensions, so the block maps
    # directly onto this selection even when a variable is not ordered
    # ``(time, lat, lon)``.
    selection = tuple(source_region[dim] for dim in variable.dims)
    array[selection] = block


def _set_array_region_with_fallback(
    array: zarr.Array,
    store: Path,
    path: str,
    selection: tuple[slice, ...],
    block: np.ndarray,
    cache: dict[str, zarr.Array],
    codec_workers: int,
) -> zarr.Array:
    """Write with FusedCodecPipeline and reopen with Batched on incompatibility."""

    try:
        array[selection] = block
        return array
    except Exception as fused_error:
        # Some third-party Zarr v3 codecs do not implement the synchronous
        # fused path.  Reopen only the affected array with the standard
        # pipeline and retry the complete selection.  The selection is owned
        # exclusively by this task, so a retry cannot race another writer.
        try:
            with zarr.config.set(
                {
                    "codec_pipeline.path": _BATCHED_CODEC_PIPELINE,
                    "codec_pipeline.max_workers": max(1, int(codec_workers)),
                }
            ):
                fallback = zarr.open_array(store=store, path=path, mode="r+")
                fallback[selection] = block
            cache[path] = fallback
            return fallback
        except Exception:
            raise fused_error


# These process-local handles are initialized once per worker.  Opening the
# source store for every chunk would erase most of the benefit of parallelism.
_STAGE1_SOURCE: xr.Dataset | None = None
_STAGE1_INTERMEDIATE: Path | None = None
_STAGE1_ARRAYS: dict[str, zarr.Array] = {}
_STAGE1_CODEC_WORKERS = 1


def _init_stage1_worker(
    source: str,
    intermediate: str,
    codec_workers: int,
) -> None:
    global _STAGE1_SOURCE, _STAGE1_INTERMEDIATE, _STAGE1_ARRAYS
    global _STAGE1_CODEC_WORKERS
    _STAGE1_SOURCE = xr.open_zarr(
        source,
        consolidated=False,
        chunks={},
        decode_times=False,
        mask_and_scale=False,
    )
    _STAGE1_INTERMEDIATE = Path(intermediate)
    _STAGE1_ARRAYS = {}
    _STAGE1_CODEC_WORKERS = max(1, int(codec_workers))


def _stage1_time_task(task: tuple[str, int]) -> dict[str, int | float | str]:
    """Process one complete source time chunk without overlapping writers."""

    if _STAGE1_SOURCE is None or _STAGE1_INTERMEDIATE is None:
        raise RuntimeError("阶段 1 worker 尚未初始化。")
    variable_name, time_index = task
    variable = _STAGE1_SOURCE[variable_name]
    chunks, starts = _chunk_layout(variable)
    time_axis = variable.dims.index("time")
    spatial_axes = [axis for axis in range(variable.ndim) if axis != time_axis]
    source_data = variable.data
    with zarr.config.set(
        {
            "codec_pipeline.path": _FUSED_CODEC_PIPELINE,
            "codec_pipeline.max_workers": _STAGE1_CODEC_WORKERS,
        }
    ):
        array = _STAGE1_ARRAYS.get(variable_name)
        if array is None:
            array = zarr.open_array(
                store=_STAGE1_INTERMEDIATE,
                path=variable_name,
                mode="r+",
            )
            _STAGE1_ARRAYS[variable_name] = array
        started = time.perf_counter()
        processed_bytes = 0
        source_chunks = 0
        spatial_ranges = [range(len(chunks[axis])) for axis in spatial_axes]
        for spatial_indices in itertools.product(*spatial_ranges):
            indices = [0] * variable.ndim
            indices[time_axis] = int(time_index)
            for axis, index in zip(spatial_axes, spatial_indices):
                indices[axis] = int(index)
            indices_tuple = tuple(indices)
            region = _region_for_chunk_indices(
                tuple(str(dim) for dim in variable.dims),
                chunks,
                starts,
                indices_tuple,
            )
            block = source_data.blocks[indices_tuple]
            if hasattr(block, "compute"):
                block = block.compute(scheduler="synchronous")
            block = np.asarray(block)
            processed_bytes += int(block.nbytes)
            selection = tuple(region[dim] for dim in variable.dims)
            _STAGE1_ARRAYS[variable_name] = _set_array_region_with_fallback(
                array,
                _STAGE1_INTERMEDIATE,
                variable_name,
                selection,
                block,
                _STAGE1_ARRAYS,
                _STAGE1_CODEC_WORKERS,
            )
            source_chunks += 1
            del block
    return {
        "variable": variable_name,
        "time_index": int(time_index),
        "source_chunks": source_chunks,
        "bytes": processed_bytes,
        "elapsed": time.perf_counter() - started,
    }


_STAGE2_INTERMEDIATE: xr.Dataset | None = None
_STAGE2_STAGING: Path | None = None
_STAGE2_ARRAYS: dict[str, zarr.Array] = {}
_STAGE2_CODEC_WORKERS = 1


def _init_stage2_worker(
    intermediate: str,
    staging: str,
    codec_workers: int,
) -> None:
    global _STAGE2_INTERMEDIATE, _STAGE2_STAGING, _STAGE2_ARRAYS
    global _STAGE2_CODEC_WORKERS
    _STAGE2_INTERMEDIATE = xr.open_zarr(
        intermediate,
        consolidated=False,
        chunks={},
        decode_times=False,
        mask_and_scale=False,
    )
    _STAGE2_STAGING = Path(staging)
    _STAGE2_ARRAYS = {}
    _STAGE2_CODEC_WORKERS = max(1, int(codec_workers))


def _stage2_region_task(
    task: tuple[str, tuple[int, ...], tuple[int, ...]],
) -> dict[str, int | float | str]:
    """Merge one final chunk; each task owns a disjoint output region."""

    if _STAGE2_INTERMEDIATE is None or _STAGE2_STAGING is None:
        raise RuntimeError("阶段 2 worker 尚未初始化。")
    variable_name, starts, stops = task
    variable = _STAGE2_INTERMEDIATE[variable_name]
    selection = tuple(slice(int(start), int(stop)) for start, stop in zip(starts, stops))
    started = time.perf_counter()
    block = variable.data[selection]
    if hasattr(block, "compute"):
        block = block.compute(scheduler="synchronous")
    block = np.asarray(block)
    with zarr.config.set(
        {
            "codec_pipeline.path": _FUSED_CODEC_PIPELINE,
            "codec_pipeline.max_workers": _STAGE2_CODEC_WORKERS,
        }
    ):
        array = _STAGE2_ARRAYS.get(variable_name)
        if array is None:
            array = zarr.open_array(
                store=_STAGE2_STAGING,
                path=variable_name,
                mode="r+",
            )
            _STAGE2_ARRAYS[variable_name] = array
        _STAGE2_ARRAYS[variable_name] = _set_array_region_with_fallback(
            array,
            _STAGE2_STAGING,
            variable_name,
            selection,
            block,
            _STAGE2_ARRAYS,
            _STAGE2_CODEC_WORKERS,
        )
    processed_bytes = int(block.nbytes)
    del block
    return {
        "variable": variable_name,
        "bytes": processed_bytes,
        "elapsed": time.perf_counter() - started,
    }


def _run_process_tasks(
    tasks: Iterable[tuple],
    total: int,
    *,
    workers: int,
    initializer: object,
    initargs: tuple[object, ...],
    task_function: object,
    progress: bool,
    stage_label: str,
    cancel_event=None,
) -> None:
    """Run bounded process tasks and report aggregate throughput."""

    started = time.perf_counter()
    processed_bytes = 0
    processed_chunks = 0
    # Keep progress output useful without turning hundreds of process
    # completions into another serialization bottleneck.
    progress_interval = max(1, min(32, total // 20))
    results = bounded_process_map(
        task_function,  # type: ignore[arg-type]
        tasks,
        workers=workers,
        initializer=initializer,  # type: ignore[arg-type]
        initargs=initargs,
        cancel_event=cancel_event,
    )
    try:
        for index, result in enumerate(results, start=1):
            processed_bytes += int(result.get("bytes", 0))
            processed_chunks += int(result.get("source_chunks", 1))
            if progress and (
                index == 1 or index == total or index % progress_interval == 0
            ):
                elapsed = max(time.perf_counter() - started, 1e-9)
                rate = processed_bytes / 1024**2 / elapsed
                print(
                    f"{stage_label}并行任务 {index}/{total} 完成；"
                    f"处理源 chunk {processed_chunks}；吞吐 {rate:.1f} MiB/s",
                    flush=True,
                )
    except RuntimeError as exc:
        if str(exc) == "任务已取消。":
            raise RechunkExecutionError(str(exc)) from exc
        raise


def _stage1_time_tasks(info: DatasetInfo) -> Iterator[tuple[str, int]]:
    for variable in info.data_variables:
        if variable.ndim != 3:
            continue
        time_axis = variable.dims.index("time")
        yield from (
            (variable.name, index)
            for index in range(
                (int(variable.shape[time_axis]) + int(variable.chunks[time_axis]) - 1)
                // int(variable.chunks[time_axis])
            )
        )


def _stage1_time_task_count(info: DatasetInfo) -> int:
    return sum(
        (int(variable.shape[variable.dims.index("time")])
         + int(variable.chunks[variable.dims.index("time")]) - 1)
        // int(variable.chunks[variable.dims.index("time")])
        for variable in info.data_variables
        if variable.ndim == 3
    )


def _stage1_source_chunk_count(info: DatasetInfo) -> int:
    total = 0
    for variable in info.data_variables:
        if variable.ndim != 3:
            continue
        count = 1
        for size, chunk in zip(variable.shape, variable.chunks):
            count *= (int(size) + int(chunk) - 1) // int(chunk)
        total += count
    return total


def _intermediate_chunk_count(
    info: DatasetInfo,
    intermediate_chunks: tuple[int, int, int],
) -> int:
    """Estimate the number of logical chunks in the intermediate arrays."""

    mapping = dict(zip(("time", "lat", "lon"), intermediate_chunks))
    total = 0
    for variable in info.data_variables:
        if variable.ndim != 3:
            continue
        count = 1
        for dim, size in zip(variable.dims, variable.shape):
            count *= (int(size) + int(mapping[dim]) - 1) // int(mapping[dim])
        total += count
    return total


def _stage2_tasks(
    info: DatasetInfo,
    plan: ChunkPlan,
    intermediate_chunks: tuple[int, int, int],
    region_chunks: dict[str, tuple[int, ...]] | None = None,
) -> Iterator[tuple[str, tuple[int, ...], tuple[int, ...]]]:
    for variable in info.data_variables:
        if variable.ndim != 3:
            continue
        chunks = (
            region_chunks.get(variable.name, intermediate_chunks)
            if region_chunks is not None
            else intermediate_chunks
        )
        for region in _group_regions(variable, plan, chunks):
            starts = tuple(int(region[dim].start or 0) for dim in variable.dims)
            stops = tuple(
                int(region[dim].stop or size)
                for dim, size in zip(variable.dims, variable.shape)
            )
            yield variable.name, starts, stops


def _populate_intermediate_parallel(
    source_path: Path,
    info: DatasetInfo,
    intermediate: Path,
    *,
    workers: int,
    progress: bool,
    stage_label: str = "阶段 1/2：",
    cancel_event=None,
) -> None:
    total_tasks = _stage1_time_task_count(info)
    if total_tasks == 0:
        raise RechunkExecutionError("没有可并行处理的三维数据变量。")
    codec_workers = max(1, min(2, workers))
    _run_process_tasks(
        _stage1_time_tasks(info),
        total_tasks,
        workers=workers,
        initializer=_init_stage1_worker,
        initargs=(str(source_path), str(intermediate), codec_workers),
        task_function=_stage1_time_task,
        progress=progress,
        stage_label=stage_label,
        cancel_event=cancel_event,
    )


def _populate_final_parallel(
    intermediate: Path,
    staging: Path,
    info: DatasetInfo,
    plan: ChunkPlan,
    intermediate_chunks: tuple[int, int, int],
    region_chunks: dict[str, tuple[int, ...]] | None = None,
    *,
    workers: int,
    progress: bool,
    cancel_event=None,
) -> None:
    total_tasks = _stage2_task_count(info, plan, intermediate_chunks, region_chunks)
    if total_tasks == 0:
        raise RechunkExecutionError("没有可并行合并的三维数据变量。")
    codec_workers = max(1, min(2, workers))
    _run_process_tasks(
        _stage2_tasks(info, plan, intermediate_chunks, region_chunks),
        total_tasks,
        workers=workers,
        initializer=_init_stage2_worker,
        initargs=(str(intermediate), str(staging), codec_workers),
        task_function=_stage2_region_task,
        progress=progress,
        stage_label="阶段 2/2：",
        cancel_event=cancel_event,
    )


def _populate_intermediate(
    source: xr.Dataset,
    info: DatasetInfo,
    intermediate: Path,
    *,
    workers: int,
    progress: bool,
    source_path: Path | None = None,
    stage_label: str = "阶段 1/2：",
    parallel: bool = False,
    cancel_event=None,
) -> None:
    """Read each physical source chunk once and scatter it to an intermediate store."""

    if parallel:
        if source_path is None:
            raise RechunkExecutionError("并行阶段 1 缺少输入 Zarr 路径。")
        _populate_intermediate_parallel(
            source_path,
            info,
            intermediate,
            workers=workers,
            progress=progress,
            stage_label=stage_label,
            cancel_event=cancel_event,
        )
        return

    data_variables = [variable for variable in info.data_variables if variable.ndim == 3]
    # One batched Array assignment is substantially faster than hundreds of
    # independent assignments.  Limit Zarr's internal codec pool explicitly so
    # the batch remains bounded on high-core machines and cannot create an
    # unbounded number of compressed buffers.
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        with zarr.config.set(
            {
                "codec_pipeline.path": _FUSED_CODEC_PIPELINE,
                "codec_pipeline.max_workers": max(1, int(workers)),
            }
        ):
            for variable_index, variable in enumerate(data_variables, start=1):
                source_variable = source[variable.name]
                regions = _chunk_regions(source_variable)
                source_indices = _source_chunk_indices(source_variable)
                total = _array_chunk_count(source_variable)
                progress_interval = max(1, min(32, total // 100))
                array = zarr.open_array(store=intermediate, path=variable.name, mode="r+")
                if progress:
                    print(
                        f"{stage_label}写入块 {variable.name} "
                        f"（{variable_index}/{len(data_variables)}），共 {total} 个源 chunk"
                )
                source_data = source_variable.data
                variable_started = time.perf_counter()
                processed_bytes = 0
                for index, (region, indices) in enumerate(
                    zip(regions, source_indices), start=1
                ):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RechunkExecutionError("任务已取消。")
                    if hasattr(source_data, "blocks"):
                        block = source_data.blocks[indices]
                    else:
                        block = source_variable.isel(region).data
                    if hasattr(block, "compute"):
                        # Each ``.blocks`` selection is one physical source
                        # chunk.  A synchronous Dask scheduler avoids creating
                        # a fresh one-thread pool for every chunk.
                        block = block.compute(scheduler="synchronous")
                    block = np.asarray(block)
                    processed_bytes += int(block.nbytes)
                    _write_intermediate_source_chunk(
                        array,
                        block,
                        region,
                        variable,
                    )
                    if progress and (
                        index == 1
                        or index == total
                        or index % progress_interval == 0
                    ):
                        elapsed = max(time.perf_counter() - variable_started, 1e-9)
                        rate = processed_bytes / 1024**2 / elapsed
                        print(
                            f"  源 chunk {index}/{total} 完成；"
                            f"阶段吞吐 {rate:.1f} MiB/s",
                            flush=True,
                        )
                    del block
                    # The source block is already bounded by the physical
                    # source chunk.  Reference counting releases the ndarray;
                    # defer cyclic-GC traversal until the variable is done.
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()


def _populate_final_direct(
    source: xr.Dataset,
    source_path: Path,
    info: DatasetInfo,
    staging: Path,
    *,
    workers: int,
    progress: bool,
    parallel: bool,
    cancel_event=None,
) -> None:
    """Decode each source physical chunk and encode that same final chunk once."""

    _populate_intermediate(
        source,
        info,
        staging,
        workers=workers,
        progress=progress,
        source_path=source_path,
        parallel=parallel,
        cancel_event=cancel_event,
        stage_label="单阶段：",
    )


def _intermediate_time_chunk(info: DatasetInfo, plan: ChunkPlan) -> int:
    # For space/custom strategies the final time chunk is already bounded;
    # using it in the intermediate store lets stage 2 read each intermediate
    # chunk once.  Time-contiguous output is the special case: its full time
    # chunk is too large, so retain the source time chunk and merge in stage 2.
    if plan.strategy != "time":
        return max(1, int(plan.chunks[0]))
    values = [
        int(variable.chunks[variable.dims.index("time")])
        for variable in info.data_variables
        if variable.ndim == 3 and variable.chunks
    ]
    if not values:
        raise RechunkExecutionError("无法从源数据确定中间 time chunk。")
    result = values[0]
    for value in values[1:]:
        result = int(np.gcd(result, value))
    return max(1, result)


def _intermediate_chunks(
    info: DatasetInfo,
    plan: ChunkPlan,
    workers: int,
) -> tuple[int, int, int]:
    """Choose source-time chunks aligned with the final spatial chunks.

    Keeping the spatial dimensions equal to the final chunks is intentional:
    source chunks often cross target boundaries, and larger intermediate
    chunks would force repeated read-modify-write operations on the same
    intermediate chunk.  The source-time dimension remains small for the
    time-contiguous strategy and equals the final time chunk for the other
    strategies.
    """

    time_chunk = _intermediate_time_chunk(info, plan)
    return (
        time_chunk,
        int(plan.chunks[1]),
        int(plan.chunks[2]),
    )


def _intermediate_shards(
    info: DatasetInfo,
    intermediate_chunks: tuple[int, int, int],
    final_chunks: tuple[int, int, int] | None = None,
) -> dict[str, tuple[int, ...]]:
    """Choose bounded Zarr v3 shard shapes for the intermediate store.

    The intermediate logical chunks are deliberately small so that a source
    chunk can be scattered without retaining a large in-memory block.  A
    regular Zarr v3 directory would therefore create hundreds of thousands of
    tiny files.  Sharding preserves the logical chunk grid while packing a few
    neighbouring chunks into one physical shard file.

    Time is never coalesced here.  Stage 1 assigns a complete source-time
    chunk to one worker, so keeping the shard time edge equal to the logical
    time edge guarantees that two workers never update the same shard.  Spatial
    edges are expanded by up to four logical chunks, subject to a conservative
    16 MiB uncompressed shard budget.  A second 256 MiB budget bounds the
    stage-2 read region after the final time edge is applied.  These budgets
    limit read-modify-write amplification and keep worker buffers bounded in
    both stages.
    """

    target_bytes = 16 * 1024**2
    # A stage-2 task reads one shard-sized spatial region over its final time
    # edge.  Keep that read block bounded as well; otherwise a 4x4 spatial
    # shard that is cheap in stage 1 could become a multi-gigabyte block when
    # the final time chunk is the complete time axis.
    final_time = int(final_chunks[0]) if final_chunks is not None else int(
        intermediate_chunks[0]
    )
    stage2_target_bytes = 256 * 1024**2
    dim_names = ("time", "lat", "lon")
    result: dict[str, tuple[int, ...]] = {}
    for variable in info.data_variables:
        if variable.ndim != 3:
            continue
        logical_by_dim = dict(zip(dim_names, intermediate_chunks))
        factors = {"time": 1, "lat": 4, "lon": 4}
        for dim in ("lat", "lon"):
            chunk = int(logical_by_dim[dim])
            size = int(info.dimensions[dim])
            # A shard edge must be an integer multiple of the logical chunk
            # edge and cannot needlessly extend beyond the dimension grid.
            factors[dim] = min(
                factors[dim],
                max(1, (size + chunk - 1) // chunk),
            )

        def shard_bytes() -> int:
            return int(
                np.prod(
                    [
                        int(logical_by_dim[dim]) * int(factors[dim])
                        for dim in dim_names
                    ],
                    dtype=np.int64,
                )
            ) * int(variable.dtype.itemsize)

        def stage2_bytes() -> int:
            return int(
                final_time
                * int(logical_by_dim["lat"])
                * int(factors["lat"])
                * int(logical_by_dim["lon"])
                * int(factors["lon"])
            ) * int(variable.dtype.itemsize)

        # Reduce the larger spatial factor first until the shard is within
        # budget or both spatial factors are already one.
        while (
            (shard_bytes() > target_bytes or stage2_bytes() > stage2_target_bytes)
            and max(factors["lat"], factors["lon"]) > 1
        ):
            if factors["lat"] >= factors["lon"] and factors["lat"] > 1:
                factors["lat"] = max(1, factors["lat"] // 2)
            elif factors["lon"] > 1:
                factors["lon"] = max(1, factors["lon"] // 2)

        shard = tuple(
            int(logical_by_dim[dim]) * int(factors[dim])
            for dim in variable.dims
        )
        if shard != tuple(int(logical_by_dim[dim]) for dim in variable.dims):
            result[variable.name] = shard
    return result


def _stage2_region_chunks(
    info: DatasetInfo,
    intermediate_chunks: tuple[int, int, int],
    final_chunks: tuple[int, int, int],
) -> dict[str, tuple[int, ...]]:
    """Group logical intermediate chunks into bounded stage-2 read regions.

    This grouping is useful even when the intermediate store is not sharded:
    one task can read several adjacent logical chunks and emit the matching
    final chunks in one Zarr selection.  Keeping the grouping independent from
    the physical shard layout also lets regular and sharded stores use the
    same bounded stage-2 task geometry.
    """

    stage2_target_bytes = 256 * 1024**2
    dim_names = ("time", "lat", "lon")
    final_time = int(final_chunks[0])
    result: dict[str, tuple[int, ...]] = {}
    for variable in info.data_variables:
        if variable.ndim != 3:
            continue
        logical_by_dim = dict(zip(dim_names, intermediate_chunks))
        factors = {"time": 1, "lat": 4, "lon": 4}
        for dim in ("lat", "lon"):
            chunk = int(logical_by_dim[dim])
            size = int(info.dimensions[dim])
            factors[dim] = min(
                factors[dim],
                max(1, (size + chunk - 1) // chunk),
            )

        def read_bytes() -> int:
            return int(
                final_time
                * int(logical_by_dim["lat"])
                * int(factors["lat"])
                * int(logical_by_dim["lon"])
                * int(factors["lon"])
            ) * int(variable.dtype.itemsize)

        while read_bytes() > stage2_target_bytes and max(
            factors["lat"], factors["lon"]
        ) > 1:
            if factors["lat"] >= factors["lon"] and factors["lat"] > 1:
                factors["lat"] = max(1, factors["lat"] // 2)
            elif factors["lon"] > 1:
                factors["lon"] = max(1, factors["lon"] // 2)

        result[variable.name] = tuple(
            int(logical_by_dim[dim]) * int(factors[dim])
            for dim in variable.dims
        )
    return result


def _group_regions(
    variable: VariableInfo,
    plan: ChunkPlan,
    intermediate_chunks: tuple[int, int, int],
) -> Iterator[dict[str, slice]]:
    mapping = dict(zip(("time", "lat", "lon"), plan.chunks))
    intermediate_mapping = dict(
        zip(("time", "lat", "lon"), intermediate_chunks)
    )
    starts: list[range] = []
    for dim, size in zip(variable.dims, variable.shape):
        chunk = mapping[dim] if dim == "time" else intermediate_mapping[dim]
        starts.append(range(0, int(size), int(chunk)))
    for offsets in itertools.product(*starts):
        yield {
            dim: slice(
                start,
                min(
                    start
                    + (
                        mapping[dim]
                        if dim == "time"
                        else intermediate_mapping[dim]
                    ),
                    int(size),
                ),
            )
            for dim, size, start in zip(variable.dims, variable.shape, offsets)
        }


def _group_region_count(
    variable: VariableInfo,
    plan: ChunkPlan,
    intermediate_chunks: tuple[int, int, int],
) -> int:
    final = dict(zip(("time", "lat", "lon"), plan.chunks))
    intermediate = dict(zip(("time", "lat", "lon"), intermediate_chunks))
    return int(np.prod([
        (int(size) + int(final[dim] if dim == "time" else intermediate[dim]) - 1)
        // int(final[dim] if dim == "time" else intermediate[dim])
        for dim, size in zip(variable.dims, variable.shape)
    ], dtype=np.int64))


def _stage2_task_count(
    info: DatasetInfo,
    plan: ChunkPlan,
    intermediate_chunks: tuple[int, int, int],
    region_chunks: dict[str, tuple[int, ...]] | None = None,
) -> int:
    return sum(
        _group_region_count(
            variable,
            plan,
            region_chunks.get(variable.name, intermediate_chunks)
            if region_chunks is not None
            else intermediate_chunks,
        )
        for variable in info.data_variables
        if variable.ndim == 3
    )


def _write_final_group(
    array: zarr.Array,
    block: np.ndarray,
    group_region: dict[str, slice],
    variable: VariableInfo,
    final_chunks: tuple[int, ...],
) -> None:
    """Write one aligned group of final chunks in one call.

    ``_group_regions`` yields regions whose edges are multiples of the final
    chunk grid (including edge chunks).  A group may contain several final
    chunks, but it never overlaps another task, so Zarr can split this one
    selection without read-modify-write races.
    """

    selection = tuple(group_region[dim] for dim in variable.dims)
    array[selection] = block


def _populate_final_from_intermediate(
    intermediate: xr.Dataset,
    info: DatasetInfo,
    staging: Path,
    plan: ChunkPlan,
    intermediate_chunks: tuple[int, int, int],
    region_chunks: dict[str, tuple[int, ...]] | None = None,
    *,
    workers: int,
    progress: bool,
    cancel_event=None,
) -> None:
    """Merge each spatially aligned intermediate block into final Zarr chunks."""

    data_variables = [variable for variable in info.data_variables if variable.ndim == 3]
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        with zarr.config.set(
            {
                "codec_pipeline.path": _FUSED_CODEC_PIPELINE,
                "codec_pipeline.max_workers": max(1, min(2, int(workers))),
            }
        ):
            for variable_index, variable in enumerate(data_variables, start=1):
                source_variable = _clean_for_region(intermediate[variable.name])
                source_data = source_variable.data
                chunks = (
                    region_chunks.get(variable.name, intermediate_chunks)
                    if region_chunks is not None
                    else intermediate_chunks
                )
                regions = _group_regions(variable, plan, chunks)
                array = zarr.open_array(store=staging, path=variable.name, mode="r+")
                chunks_by_dim = dict(zip(("time", "lat", "lon"), plan.chunks))
                final_chunks = tuple(chunks_by_dim[dim] for dim in variable.dims)
                total = _group_region_count(variable, plan, chunks)
                progress_interval = max(1, min(32, total // 100))
                if progress:
                    print(
                        f"阶段 2/2：最终合并 {variable.name} "
                        f"（{variable_index}/{len(data_variables)}），共 {total} 个块"
                    )
                variable_started = time.perf_counter()
                processed_bytes = 0
                for index, region in enumerate(regions, start=1):
                    if cancel_event is not None and cancel_event.is_set():
                        raise RechunkExecutionError("任务已取消。")
                    selection = tuple(region[dim] for dim in variable.dims)
                    # Slice the underlying Dask array directly.  Going through
                    # ``DataArray.isel`` here rebuilds xarray index metadata
                    # for every final chunk (thousands of times on FLUXSAT).
                    block = source_data[selection]
                    if hasattr(block, "compute"):
                        block = block.compute(
                            scheduler="threads", num_workers=max(1, workers)
                        )
                    block = np.asarray(block)
                    processed_bytes += int(block.nbytes)
                    _write_final_group(
                        array,
                        block,
                        region,
                        variable,
                        final_chunks,
                    )
                    if progress and (
                        index == 1
                        or index == total
                        or index % progress_interval == 0
                    ):
                        elapsed = max(time.perf_counter() - variable_started, 1e-9)
                        rate = processed_bytes / 1024**2 / elapsed
                        print(
                            f"  合并块 {index}/{total} 完成；"
                            f"阶段吞吐 {rate:.1f} MiB/s",
                            flush=True,
                        )
                    del block
    finally:
        if gc_was_enabled:
            gc.enable()
        gc.collect()


def _validate_structure(
    source: DatasetInfo,
    output: DatasetInfo,
    plan: ChunkPlan,
    compression: CompressionPlan,
) -> None:
    if source.dimensions != output.dimensions:
        raise RechunkExecutionError(
            f"输出维度发生变化：输入 {source.dimensions}，输出 {output.dimensions}"
        )
    source_vars = {item.name: item for item in source.variables}
    output_vars = {item.name: item for item in output.variables}
    if source_vars.keys() != output_vars.keys():
        raise RechunkExecutionError("输出变量集合与输入不一致。")
    for name, variable in source_vars.items():
        actual = output_vars[name]
        if variable.dims != actual.dims or variable.shape != actual.shape:
            raise RechunkExecutionError(f"变量 {name} 的维度或 shape 发生变化。")
        if variable.dtype != actual.dtype:
            raise RechunkExecutionError(f"变量 {name} 的 dtype 发生变化。")
        if variable.ndim:
            expected = plan.chunks_for(variable)
            if actual.chunks != expected:
                raise RechunkExecutionError(
                    f"变量 {name} 的 chunks 不符合计划："
                    f"期望 {expected}，实际 {actual.chunks}"
                )
        desired_codec = codec_for(variable, compression, coordinate=variable.is_coord)
        expected_compressors = (
            (desired_codec,)
            if compression.enabled and desired_codec is not None
            else variable.compressors
        )
        if len(actual.compressors) != len(expected_compressors):
            raise RechunkExecutionError(f"变量 {name} 的 codec 数量不符合计划。")
        for expected_codec, actual_codec in zip(
            expected_compressors, actual.compressors
        ):
            expected_signature = _codec_signature(expected_codec)
            actual_signature = _codec_signature(actual_codec)
            if (
                expected_signature is None
                or actual_signature is None
                or expected_signature != actual_signature
            ):
                raise RechunkExecutionError(f"变量 {name} 的 codec 不符合计划。")


def _sample_slices(info: DatasetInfo) -> tuple[object, ...]:
    ntime, nlat, nlon = info.shape
    time_values = sorted({0, max(0, ntime // 2), max(0, ntime - 1)})
    lat_start = max(0, nlat // 2 - 1)
    lon_start = max(0, nlon // 2 - 1)
    return (
        {"time": time_values, "lat": slice(lat_start, min(nlat, lat_start + 2)),
         "lon": slice(lon_start, min(nlon, lon_start + 2))},
        {"time": [0], "lat": slice(0, min(nlat, 2)),
         "lon": slice(0, min(nlon, 2))},
    )


def _validate_samples(source_path: Path, output_path: Path, info: DatasetInfo) -> None:
    source = xr.open_zarr(
        source_path,
        consolidated=False,
        chunks=None,
        decode_times=False,
        mask_and_scale=False,
    )
    output = xr.open_zarr(
        output_path,
        consolidated=False,
        chunks=None,
        decode_times=False,
        mask_and_scale=False,
    )
    try:
        for indexer in _sample_slices(info):
            for variable in info.data_variables:
                expected = source[variable.name].isel(indexer).values
                actual = output[variable.name].isel(indexer).values
                if np.issubdtype(variable.dtype, np.floating):
                    np.testing.assert_allclose(
                        actual, expected, equal_nan=True, rtol=0, atol=0
                    )
                else:
                    np.testing.assert_array_equal(actual, expected)
    except (AssertionError, KeyError, ValueError) as exc:
        raise RechunkExecutionError(f"输出抽样校验失败：{exc}") from exc
    finally:
        source.close()
        output.close()


def run_rechunk(
    source: str | Path,
    output: str | Path,
    info: DatasetInfo,
    plan: ChunkPlan,
    compression: CompressionPlan,
    *,
    workers: int = 1,
    overwrite: bool = False,
    progress: bool = True,
    validate: bool = True,
    cancel_event=None,
    temporary_dir: str | Path | None = None,
) -> dict[str, float | int | str]:
    """Rechunk and optionally recompress a complete Zarr v3 store."""

    source_path = Path(source).expanduser().resolve()
    target = Path(output).expanduser().resolve()
    if source_path == target:
        raise RechunkExecutionError("输入和输出不能是同一个目录。")
    if source_path in target.parents or target in source_path.parents:
        raise RechunkExecutionError("输入和输出 Zarr 不能相互嵌套。")
    if source_path != info.path:
        raise RechunkExecutionError("输入检查结果与执行输入路径不一致。")
    _prepare_target(target, overwrite)
    temporary_root = _resolve_temporary_root(source_path, target, temporary_dir)
    # The final staging store stays beside the requested output.  Publication
    # remains an atomic rename for every execution path, including byte copy.
    staging = target.parent / f".{target.name}.rechunk-{uuid4().hex}.tmp"
    intermediate = temporary_root / f".{target.name}.intermediate-{uuid4().hex}.tmp"
    started = time.perf_counter()
    execution_path = "two_stage"
    dataset = None
    intermediate_dataset = None
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise RechunkExecutionError("任务已取消。")
        if progress:
            print(f"中间处理目录：{temporary_root}")
            if temporary_root != target.parent:
                print("最终输出暂存并直接写入输出目录所在磁盘，校验通过后改名发布。")
        dataset = xr.open_zarr(
            source_path,
            consolidated=False,
            chunks={},
            decode_times=False,
            mask_and_scale=False,
        )
        sanitize_cf_references(dataset)
        if not info.data_variables:
            raise RechunkExecutionError("没有可写入的数据变量。")
        data3d = [variable for variable in info.data_variables if variable.ndim == 3]
        if not data3d:
            raise RechunkExecutionError("没有可写入的三维数据变量。")

        copy_equivalent = (
            _chunks_match_plan(info, plan, data_only=False)
            and _compression_matches_source(info, compression)
            and _metadata_unchanged(dataset, info)
        )
        direct_chunks = _chunks_match_plan(info, plan, data_only=True)

        if copy_equivalent:
            execution_path = "copy"
            if progress:
                print(
                    "chunks、codec 和 metadata 与计划等价；"
                    "直接复制独立 chunk 文件到暂存目录。"
                )
            dataset.close()
            dataset = None
            _copy_equivalent_store(
                source_path,
                staging,
                cancel_event=cancel_event,
            )
        elif direct_chunks:
            execution_path = "single_stage"
            safe_workers = _safe_workers(info, workers)
            effective_workers, device_reason = _parallel_workers(
                source_path,
                target.parent,
                safe_workers,
            )
            effective_workers, worker_peak_bytes = _stage2_safe_workers(
                info,
                plan,
                _direct_region_chunks(info),
                compression,
                effective_workers,
            )
            if progress:
                print(
                    "源与目标物理 chunks 相同；使用单阶段逐 chunk "
                    "decode→final encode，跳过中间 Zarr。"
                )
                print(
                    f"并行 I/O 策略：{device_reason}；实际进程数="
                    f"{effective_workers}；估算每 worker 峰值="
                    f"{worker_peak_bytes / 1024**2:.1f} MiB。"
                )
            _initialize_store(dataset, info, plan, compression, staging)
            direct_task_count = _stage1_time_task_count(info)
            source_chunk_count = _stage1_source_chunk_count(info)
            parallel_direct = (
                effective_workers > 1
                and direct_task_count >= effective_workers
                and source_chunk_count >= effective_workers * 2
            )
            _populate_final_direct(
                dataset,
                source_path,
                info,
                staging,
                workers=effective_workers,
                progress=progress,
                parallel=parallel_direct,
                cancel_event=cancel_event,
            )
            dataset.close()
            dataset = None
        else:
            safe_workers = _safe_workers(info, workers)
            stage1_workers, stage1_reason = _parallel_workers(
                source_path,
                temporary_root,
                safe_workers,
            )
            intermediate_chunks = _intermediate_chunks(info, plan, stage1_workers)
            intermediate_plan = replace(plan, chunks=intermediate_chunks)
            intermediate_count = _intermediate_chunk_count(info, intermediate_chunks)
            if intermediate_count >= _INTERMEDIATE_SHARD_THRESHOLD:
                intermediate_shards = _intermediate_shards(
                    info,
                    intermediate_chunks,
                    plan.chunks,
                )
            else:
                # Grouped stage-2 reads remove call overhead for ordinary
                # stores; reserve sharding for metadata-dominated grids.
                intermediate_shards = {}
            stage2_region_chunks = _stage2_region_chunks(
                info,
                intermediate_chunks,
                plan.chunks,
            )
            stage2_storage_workers, stage2_reason = _parallel_workers(
                temporary_root,
                target.parent,
                safe_workers,
            )
            stage2_workers, stage2_peak_bytes = _stage2_safe_workers(
                info,
                plan,
                stage2_region_chunks,
                compression,
                stage2_storage_workers,
            )
            intermediate_compression = (
                make_compression_plan("fast") if compression.enabled else compression
            )
            if progress:
                print(
                    "使用两阶段源 chunk 对齐重分块：源 chunk 读取一次；"
                    f"最终块分批写入（{len(data3d)} 个三维变量）。"
                )
                print(
                    f"阶段 1 I/O：{stage1_reason}；实际进程数={stage1_workers}。"
                )
                print(
                    f"阶段 2 I/O：{stage2_reason}；实际进程数={stage2_workers}；"
                    f"估算每 worker 峰值={stage2_peak_bytes / 1024**2:.1f} MiB。"
                )
                print(
                    f"阶段 1 中间 chunks(time, lat, lon)：{intermediate_chunks}；"
                    f"中间 codec：{intermediate_compression.profile}"
                )
                print(
                    f"阶段 1 预计逻辑中间 chunk 数：{intermediate_count:,}；"
                    f"sharding 阈值：{_INTERMEDIATE_SHARD_THRESHOLD:,}"
                )
                if intermediate_shards:
                    shard_text = ", ".join(
                        f"{name}={shape}"
                        for name, shape in intermediate_shards.items()
                    )
                    print(
                        "阶段 1 启用 Zarr v3 sharding，减少中间小文件："
                        f" {shard_text}"
                    )
                intermediate_by_dim = dict(
                    zip(("time", "lat", "lon"), intermediate_chunks)
                )
                grouped_text = ", ".join(
                    f"{variable.name}={stage2_region_chunks[variable.name]}"
                    for variable in data3d
                    if stage2_region_chunks.get(variable.name)
                    != tuple(intermediate_by_dim[dim] for dim in variable.dims)
                )
                if grouped_text:
                    print(f"阶段 2 读取区域批处理：{grouped_text}")

            stage1_task_count = (
                _stage1_time_task_count(info) if plan.strategy == "time" else 0
            )
            stage1_source_chunk_count = _stage1_source_chunk_count(info)
            output_chunk_count = sum(plan.estimated_chunks.values())
            # Stage 1 can use time-slice owners only when intermediate time
            # chunks align with source chunks.  Stage 2 tasks always own
            # disjoint final physical chunks.
            parallel_stage1 = (
                plan.strategy == "time"
                and stage1_workers > 1
                and (
                    stage1_task_count >= max(6, stage1_workers * 2)
                    or stage1_source_chunk_count >= max(8, stage1_workers * 4)
                )
            )
            parallel_stage2 = (
                stage2_workers > 1
                and output_chunk_count >= max(8, stage2_workers * 4)
            )
            if progress and parallel_stage1:
                print("阶段 1 使用按源 time chunk 隔离的多进程写入。")
            if progress and parallel_stage2:
                print("阶段 2 使用按最终物理 chunk 单 owner 的多进程合并。")

            _initialize_store(
                dataset,
                info,
                intermediate_plan,
                intermediate_compression,
                intermediate,
                shards=intermediate_shards,
            )
            _initialize_store(dataset, info, plan, compression, staging)
            stage1_started = time.perf_counter()
            _populate_intermediate(
                dataset,
                info,
                intermediate,
                workers=stage1_workers,
                progress=progress,
                source_path=source_path,
                parallel=parallel_stage1,
                cancel_event=cancel_event,
            )
            if cancel_event is not None and cancel_event.is_set():
                raise RechunkExecutionError("任务已取消。")
            if progress:
                print(
                    f"阶段 1/2 完成，耗时 "
                    f"{time.perf_counter() - stage1_started:.1f} 秒"
                )
            dataset.close()
            dataset = None

            stage2_started = time.perf_counter()
            if parallel_stage2:
                _populate_final_parallel(
                    intermediate,
                    staging,
                    info,
                    plan,
                    intermediate_chunks,
                    region_chunks=stage2_region_chunks,
                    workers=stage2_workers,
                    progress=progress,
                    cancel_event=cancel_event,
                )
            else:
                intermediate_dataset = xr.open_zarr(
                    intermediate,
                    consolidated=False,
                    chunks={},
                    decode_times=False,
                    mask_and_scale=False,
                )
                _populate_final_from_intermediate(
                    intermediate_dataset,
                    info,
                    staging,
                    plan,
                    intermediate_chunks,
                    region_chunks=stage2_region_chunks,
                    workers=stage2_workers,
                    progress=progress,
                    cancel_event=cancel_event,
                )
                intermediate_dataset.close()
                intermediate_dataset = None
            if cancel_event is not None and cancel_event.is_set():
                raise RechunkExecutionError("任务已取消。")
            if progress:
                print(
                    f"阶段 2/2 完成，耗时 "
                    f"{time.perf_counter() - stage2_started:.1f} 秒"
                )

        output_info = inspect_store(staging)
        _validate_structure(info, output_info, plan, compression)
        if validate:
            _validate_samples(source_path, staging, info)

        if cancel_event is not None and cancel_event.is_set():
            raise RechunkExecutionError("任务已取消。")
        publish_staging(
            staging,
            target,
            "rechunk",
            overwrite=overwrite,
            require_zarr_v3=True,
        )
        shutil.rmtree(intermediate, ignore_errors=True)
        elapsed = time.perf_counter() - started
        physical_bytes = _directory_size(target)
        return {
            "elapsed": elapsed,
            "logical_bytes": info.logical_bytes,
            "physical_bytes": physical_bytes,
            "throughput_mib_s": info.logical_bytes / 1024**2 / max(elapsed, 1e-9),
            "output": str(target),
            "temporary_dir": str(temporary_root),
            "execution_path": execution_path,
            "avoided_intermediate_bytes": (
                info.logical_bytes if execution_path != "two_stage" else 0
            ),
        }
    except Exception as exc:
        if dataset is not None:
            dataset.close()
        if intermediate_dataset is not None:
            intermediate_dataset.close()
        if cancel_event is not None and cancel_event.is_set():
            # UUID-scoped partial stores are never published on cancellation.
            shutil.rmtree(staging, ignore_errors=True)
            shutil.rmtree(intermediate, ignore_errors=True)
            raise RechunkExecutionError("任务已取消，未生成输出。") from exc
        if isinstance(exc, RechunkExecutionError):
            raise
        raise RechunkExecutionError(
            f"重分块失败；临时目录保留用于排查：{staging}\n"
            f"中间目录保留用于排查：{intermediate}\n{exc}"
        ) from exc
