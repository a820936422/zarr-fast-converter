from __future__ import annotations

from pathlib import Path

import numpy as np

from .models import Inventory, OutputLayout, Selection, VariableTransform

def validate_semantic_samples(
    output: Path,
    constraints: dict[str, dict[str, float | bool]] | None = None,
    *,
    points_per_axis: int = 3,
) -> dict[str, object]:
    """Report bounded value-domain checks without modifying scientific data."""

    import zarr

    group = zarr.open_group(output, mode="r")
    requested = constraints or {}
    checks: dict[str, dict[str, object]] = {}
    warnings: list[str] = []
    for name in group.array_keys():
        array = group[name]
        dimensions = tuple(array.metadata.dimension_names or ())
        if not {"time", "lat", "lon"}.issubset(dimensions):
            continue
        dtype = np.dtype(array.dtype)
        if dtype.kind not in "iuf":
            continue
        indices = {
            dimension: np.unique(
                np.linspace(
                    0,
                    int(array.shape[axis]) - 1,
                    min(max(1, points_per_axis), int(array.shape[axis])),
                    dtype=int,
                )
            )
            for axis, dimension in enumerate(dimensions)
        }
        selection = tuple(indices[dimension] for dimension in dimensions)
        values = np.asarray(array.oindex[selection], dtype="float64")
        finite = values[np.isfinite(values)]
        attrs = dict(array.attrs)
        variable_constraints = dict(requested.get(name) or {})
        descriptor = " ".join(
            str(attrs.get(key, "")) for key in ("standard_name", "long_name")
        ).lower()
        inferred_nonnegative = "standard_error" in descriptor or "uncertainty" in descriptor
        nonnegative = bool(variable_constraints.get("nonnegative", inferred_nonnegative))
        minimum = (
            float(variable_constraints["min"])
            if "min" in variable_constraints
            else float(attrs["valid_min"])
            if "valid_min" in attrs
            else None
        )
        maximum = (
            float(variable_constraints["max"])
            if "max" in variable_constraints
            else float(attrs["valid_max"])
            if "valid_max" in attrs
            else None
        )
        violations = []
        if finite.size and nonnegative and np.any(finite < 0):
            violations.append(f"发现 {int(np.count_nonzero(finite < 0))} 个负值")
        if finite.size and minimum is not None and np.any(finite < minimum):
            violations.append(f"存在小于 {minimum:g} 的值")
        if finite.size and maximum is not None and np.any(finite > maximum):
            violations.append(f"存在大于 {maximum:g} 的值")
        if violations:
            warnings.append(f"{name}: " + "；".join(violations))
        checks[name] = {
            "samples": int(values.size),
            "finite": int(finite.size),
            "minimum": float(finite.min()) if finite.size else None,
            "maximum": float(finite.max()) if finite.size else None,
            "nonnegative": nonnegative,
            "required_min": minimum,
            "required_max": maximum,
            "violations": violations,
        }
    close = getattr(group.store, "close", None)
    if close is not None:
        close()
    return {
        "status": "warning" if warnings else "passed",
        "checks": checks,
        "warnings": warnings,
    }


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
                if (
                    transform.scale_factor is not None or transform.add_offset is not None
                ) and raw.dtype.kind not in "fc":
                    out_dtype = np.dtype("float32" if raw.dtype.itemsize <= 4 else "float64")
                expected = raw.astype(out_dtype, copy=True)
                if transform.scale_factor is not None:
                    expected[~mask] *= transform.scale_factor
                if transform.add_offset is not None:
                    expected[~mask] += transform.add_offset
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
