from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from .benchmark import COMPRESSION_SAFETY, tune
from .models import ConversionPlan, Inventory, Selection, VariableTransform
from .planner import candidate_plans, initial_plan
from .selection import selected_logical_bytes
from .validation import validate_output
from .writer import direct_write, make_compressor, remove_output


def _canonicalize_dimensions(ds, source_dimensions: tuple[str, str, str]):
    """Rename source dimensions and their coordinate variables to canonical names."""
    canonical = ("time", "lat", "lon")
    temporary = {}
    final = {}
    existing = set(ds.dims) | set(ds.variables)
    for source, target in zip(source_dimensions, canonical):
        if source == target:
            continue
        candidate = f"__fast_nc_zarr_{target}"
        while candidate in existing or candidate in temporary.values():
            candidate += "_"
        temporary[source] = candidate
        final[candidate] = target
    if not temporary:
        return ds
    return ds.rename(temporary).rename(final)


def _dask_write(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    plan: ConversionPlan,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    chunks: tuple[int, int, int] | None = None,
    cancel_event=None,
    *,
    progress: bool = True,
) -> dict:
    import dask
    import xarray as xr
    from contextlib import nullcontext

    paths = [record.path for record in inventory.files]
    ds = xr.open_mfdataset(
        paths,
        engine=inventory.source_engine,
        combine="by_coords",
        parallel=False,
        chunks={},
        data_vars="all",
        coords="minimal",
        compat="override",
        join="exact",
        combine_attrs="override",
        decode_times=True,
        mask_and_scale=False,
    )
    try:
        ds = _canonicalize_dimensions(ds, inventory.source_dimensions)
        ds = ds[list(selection.variables)].isel(
            time=slice(selection.time_start, selection.time_stop),
            lat=slice(selection.lat_start, selection.lat_stop),
            lon=slice(selection.lon_start, selection.lon_stop),
        )
        for name in selection.variables:
            if set(ds[name].dims) == {"time", "lat", "lon"}:
                ds[name] = ds[name].transpose("time", "lat", "lon")
        # Inventory inspection normalizes all source timestamps to daily
        # dates.  Replace the backend's original coordinate so the Dask path
        # cannot accidentally write hours/minutes into the output Zarr.
        ds = ds.assign_coords(
            time=inventory.times[selection.time_start : selection.time_stop]
        )
        # The normalized output time is datetime64 even when the source uses
        # raw DOY values such as ``units=day``.  Retaining those CF encoding
        # attrs on a new datetime coordinate makes xarray try to encode the
        # coordinate twice.  Preserve them under source_* metadata instead.
        time_attrs = dict(ds["time"].attrs)
        source_time_units = time_attrs.pop("units", None)
        source_time_calendar = time_attrs.pop("calendar", None)
        if source_time_units is not None:
            time_attrs["source_time_units"] = source_time_units
        if source_time_calendar is not None:
            time_attrs["source_time_calendar"] = source_time_calendar
        ds["time"].attrs = time_attrs
        variable_transforms = variable_transforms or {}
        for name, transform in variable_transforms.items():
            if name not in ds.data_vars:
                continue
            variable = ds[name]
            mask = None
            if transform.fill_values:
                finite_values = []
                has_nan = False
                for value in transform.fill_values:
                    try:
                        has_nan = has_nan or bool(np.isnan(value))
                    except TypeError:
                        finite_values.append(value)
                    else:
                        if not np.isnan(value):
                            finite_values.append(value)
                mask = variable.isin(finite_values) if finite_values else xr.zeros_like(variable, dtype=bool)
                if has_nan:
                    mask = mask | variable.isnull()
            if transform.scale_factor is not None:
                if variable.dtype.kind not in "fc":
                    variable = variable.astype("float32" if variable.dtype.itemsize <= 4 else "float64")
                variable = variable * transform.scale_factor
                if mask is not None:
                    variable = variable.where(~mask)
            elif mask is not None:
                replacement = transform.output_fill
                if replacement is None:
                    replacement = (
                        np.nan
                        if variable.dtype.kind in "fc"
                        else transform.fill_values[0]
                    )
                variable = variable.where(~mask, other=replacement)
            ds[name] = variable
            attrs = dict(ds[name].attrs)
            if transform.fill_values:
                replacement = transform.output_fill
                if replacement is None:
                    replacement = np.nan if variable.dtype.kind in "fc" else transform.fill_values[0]
                attrs["_FillValue"] = replacement
                attrs["missing_value"] = replacement
            if transform.scale_factor is not None:
                attrs["source_scale_factor"] = transform.scale_factor
                attrs.pop("scale_factor", None)
                attrs.pop("add_offset", None)
            ds[name].attrs = attrs
        variable_names = variable_names or {}
        rename_map = {
            source: output
            for source in selection.variables
            for output in (variable_names.get(source, source),)
            if output != source
        }
        if rename_map:
            ds = ds.rename(rename_map)
        compressor = make_compressor(plan.compression, plan.compression_level, plan.shuffle)
        chunk_map = {"time": plan.chunk_time, "lat": plan.chunk_lat, "lon": plan.chunk_lon}
        encoding = {}
        for name, variable in ds.variables.items():
            if variable.ndim == 0:
                continue
            preferred = variable.encoding.get("preferred_chunks") or {}
            encoding[name] = {
                "chunks": tuple(
                    min(
                        variable.sizes[dim],
                        chunk_map.get(dim, preferred.get(dim, variable.sizes[dim])),
                    )
                    for dim in variable.dims
                )
            }
            if name in ds.data_vars and compressor is not None:
                encoding[name]["compressors"] = [compressor]
        delayed = ds.to_zarr(
            output,
            mode="w",
            encoding=encoding,
            zarr_format=3,
            consolidated=False,
            compute=False,
            align_chunks=True,
        )
        started = time.perf_counter()
        progress_context = nullcontext()
        if progress:
            from dask.diagnostics import ProgressBar

            progress_context = ProgressBar()
        with progress_context:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("任务已取消。")
            with dask.config.set(scheduler="processes", num_workers=plan.workers):
                dask.compute(delayed)
        elapsed = time.perf_counter() - started
        logical = selected_logical_bytes(inventory, selection)
        return {
            "elapsed": elapsed,
            "logical_bytes": logical,
            "throughput_mib_s": logical / max(elapsed, 1e-9) / 1024**2,
        }
    finally:
        ds.close()


def convert(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    *,
    auto_tune: bool = True,
    tune_budget: float = 60.0,
    max_workers: int | None = None,
    reserve_gib: float = 2.0,
    overwrite: bool = False,
    validate: bool = True,
    progress: bool = True,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    chunks: tuple[int, int, int] | None = None,
    cancel_event=None,
) -> tuple[ConversionPlan, dict]:
    output = output.expanduser().resolve()
    if output == inventory.input_dir:
        raise ValueError("输入目录和输出目录不能相同。")
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"输出路径存在但不是目录：{output}")
        if any(output.iterdir()):
            if not overwrite:
                raise FileExistsError(f"输出目录非空：{output}")
            remove_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    plan = initial_plan(inventory, selection, output, reserve_gib=reserve_gib)
    if chunks is not None:
        from dataclasses import replace

        if len(chunks) != 3 or any(int(value) <= 0 for value in chunks):
            raise ValueError("转换 chunks 必须是三个正整数。")
        plan = replace(
            plan,
            # ``file`` tasks deliberately own one time slice.  The pipeline
            # may request a larger time chunk for vectorized resampling, in
            # which case switch to chunk-owned writes instead of allowing
            # several workers to update one Zarr time chunk independently.
            strategy=(
                "chunk"
                if plan.strategy == "file"
                and min(int(chunks[0]), selection.shape[0]) > 1
                else plan.strategy
            ),
            task_batch=(
                1
                if plan.strategy == "file"
                and min(int(chunks[0]), selection.shape[0]) > 1
                else plan.task_batch
            ),
            chunk_time=min(int(chunks[0]), selection.shape[0]),
            chunk_lat=min(int(chunks[1]), selection.shape[1]),
            chunk_lon=min(int(chunks[2]), selection.shape[2]),
            rationale=plan.rationale + ("使用外部编排器提供的源临时 chunks。",),
        )
    tuning_results = []
    if auto_tune and chunks is None and plan.strategy != "dask":
        candidates = candidate_plans(
            inventory,
            selection,
            output,
            max_workers=max_workers,
            reserve_gib=reserve_gib,
        )
        plan, tuning_results = tune(
            inventory,
            selection,
            output,
            candidates,
            budget_seconds=tune_budget,
            progress=progress,
        )
    elif max_workers is not None:
        from dataclasses import replace

        plan = replace(plan, workers=min(plan.workers, max_workers))

    if tuning_results:
        selected_result = max(
            (item for item in tuning_results if item.plan == plan),
            key=lambda item: item.logical_mib_s,
            default=max(tuning_results, key=lambda item: item.logical_mib_s),
        )
        compression_ratio = selected_result.physical_bytes / max(
            selected_result.logical_bytes, 1
        )
        estimated_output = int(
            selected_logical_bytes(inventory, selection)
            * compression_ratio
            * COMPRESSION_SAFETY
        )
        free = shutil.disk_usage(output.parent).free
        if progress:
            print(
                f"依据实测压缩率估算输出约 {estimated_output / 1024**3:.1f} GiB；"
                f"目标磁盘可用 {free / 1024**3:.1f} GiB。"
            )
        if estimated_output > free * 0.95:
            raise OSError(
                f"预计输出 {estimated_output / 1024**3:.1f} GiB，"
                f"超过目标磁盘安全可用空间 {free * 0.95 / 1024**3:.1f} GiB。"
            )

    if progress:
        print("正式执行计划：" + plan.label())
        for reason in plan.rationale:
            print("  - " + reason)
    if plan.strategy == "dask":
        metrics = _dask_write(
            inventory,
            selection,
            output,
            plan,
            variable_transforms=variable_transforms,
            variable_names=variable_names,
            chunks=chunks,
            cancel_event=cancel_event,
            progress=progress,
        )
    else:
        metrics = direct_write(
            inventory,
            selection,
            output,
            plan,
            variable_transforms=variable_transforms,
            variable_names=variable_names,
            cancel_event=cancel_event,
            progress=progress,
        )
    if validate:
        validate_output(
            inventory,
            selection,
            output,
            variable_transforms=variable_transforms,
            variable_names=variable_names,
        )
    return plan, metrics
