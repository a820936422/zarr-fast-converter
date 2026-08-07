from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import Inventory, OutputLayout, Selection, VariableTransform


def _lookup(inventory: Inventory, selection: Selection) -> list[tuple[Path, int]]:
    mapping = {}
    for record in inventory.files:
        for local_index, key in enumerate(record.time_keys):
            mapping[key] = (record.path, local_index)
    return [mapping[key] for key in inventory.time_keys[selection.time_start : selection.time_stop]]


def validate_output(
    inventory: Inventory,
    selection: Selection,
    output: Path,
    *,
    points: int = 3,
    variable_transforms: dict[str, VariableTransform] | None = None,
    variable_names: dict[str, str] | None = None,
    output_layout: OutputLayout | None = None,
) -> None:
    import netCDF4
    import xarray as xr
    import zarr

    nt, ny, nx = selection.shape
    expected_sizes = {"time": nt, "lat": ny, "lon": nx}
    with xr.open_zarr(output, consolidated=False, chunks=None, mask_and_scale=False) as ds:
        for dim, size in expected_sizes.items():
            if ds.sizes.get(dim) != size:
                raise RuntimeError(f"输出 {dim}={ds.sizes.get(dim)}，期望 {size}")
        expected_lat = inventory.lat_values[selection.lat_start : selection.lat_stop]
        expected_lon = inventory.lon_values[selection.lon_start : selection.lon_stop]
        if output_layout is not None and "lat" in output_layout.axis_reversals:
            expected_lat = expected_lat[::-1]
        if output_layout is not None and "lon" in output_layout.axis_reversals:
            expected_lon = expected_lon[::-1]
        np.testing.assert_equal(ds.lat.values, expected_lat)
        np.testing.assert_equal(ds.lon.values, expected_lon)
        np.testing.assert_equal(
            ds.time.values,
            inventory.times[selection.time_start : selection.time_stop],
        )
        output_names = {
            name: (variable_names or {}).get(name, name)
            for name in selection.variables
        }
        if set(output_names.values()) - set(ds.data_vars):
            raise RuntimeError("输出缺少所选变量。")

    group = zarr.open_group(output, mode="r")
    lookup = _lookup(inventory, selection)
    indices = sorted({0, nt // 2, nt - 1})
    if points > 3 and nt > 3:
        indices = sorted(set(np.linspace(0, nt - 1, points, dtype=int).tolist()))
    for name in selection.variables:
        output_name = (variable_names or {}).get(name, name)
        spec = inventory.variables[name]
        transform = (variable_transforms or {}).get(name)
        for t_index in indices:
            y_index = min(ny - 1, (t_index * 104729 + 1) % ny)
            x_index = min(nx - 1, (t_index * 130363 + 3) % nx)
            source_path, local_t = lookup[t_index]
            with netCDF4.Dataset(source_path, mode="r") as source:
                source.set_auto_mask(False)
                source.set_auto_scale(False)
                source_indices = []
                for dim in spec.dims:
                    if dim == "time":
                        source_indices.append(local_t)
                    elif dim == "lat":
                        source_y = (
                            ny - 1 - y_index
                            if output_layout is not None
                            and "lat" in output_layout.axis_reversals
                            else y_index
                        )
                        source_indices.append(selection.lat_start + source_y)
                    elif dim == "lon":
                        source_x = (
                            nx - 1 - x_index
                            if output_layout is not None
                            and "lon" in output_layout.axis_reversals
                            else x_index
                        )
                        source_indices.append(selection.lon_start + source_x)
                expected = source.variables[name][tuple(source_indices)]
            output_indices = tuple(
                {"time": t_index, "lat": y_index, "lon": x_index}[dim]
                for dim in ("time", "lat", "lon")
                if dim in spec.dims
            )
            if transform is not None:
                raw = np.asarray(expected)
                mask = np.zeros(raw.shape, dtype=bool)
                if transform.fill_values:
                    for value in transform.fill_values:
                        try:
                            if np.isnan(value) and np.issubdtype(raw.dtype, np.floating):
                                mask |= np.isnan(raw)
                                continue
                        except TypeError:
                            pass
                        mask |= raw == value
                out_dtype = raw.dtype
                if transform.scale_factor is not None and raw.dtype.kind not in "fc":
                    out_dtype = np.dtype("float32" if raw.dtype.itemsize <= 4 else "float64")
                expected = raw.astype(out_dtype, copy=True)
                if transform.scale_factor is not None:
                    expected[~mask] *= transform.scale_factor
                if mask.any():
                    if transform.output_fill is not None:
                        expected[mask] = transform.output_fill
                    elif np.issubdtype(out_dtype, np.floating):
                        expected[mask] = np.nan
                    elif transform.fill_values:
                        expected[mask] = transform.fill_values[0]
            actual = group[output_name][tuple(output_indices)]
            if transform is not None and np.issubdtype(np.asarray(expected).dtype, np.floating):
                np.testing.assert_allclose(actual, expected, equal_nan=True)
            else:
                np.testing.assert_equal(actual, expected)
