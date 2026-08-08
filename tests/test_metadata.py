from __future__ import annotations

import numpy as np
import xarray as xr

from fast_nc_zarr.metadata import sanitize_cf_references
from fast_nc_zarr.validation import validate_semantic_samples


def test_sanitize_cf_references_rewrites_and_filters_dependencies() -> None:
    dataset = xr.Dataset(
        {
            "gpp": (
                ("time",),
                np.asarray([1.0], dtype="float32"),
                {
                    "ancillary_variables": "uncertainty missing_quality uncertainty",
                    "grid_mapping": "crs",
                    "cell_measures": "area: cell_area volume: missing_volume",
                    "formula_terms": "a: coefficient b: missing_term",
                },
            ),
            "uncertainty_out": (("time",), np.asarray([0.1], dtype="float32")),
            "cell_area": (("time",), np.asarray([1.0], dtype="float32")),
            "coefficient": (("time",), np.asarray([2.0], dtype="float32")),
        },
        coords={
            "time": (
                ("time",),
                np.asarray([0], dtype="int64"),
                {"bounds": "time_bnds", "coordinates": "lat lon missing_coord"},
            ),
            "lat": (("time",), np.asarray([30.0])),
            "lon": (("time",), np.asarray([120.0])),
        },
    )

    result = sanitize_cf_references(
        dataset,
        renames={"uncertainty": "uncertainty_out"},
    )

    assert result is dataset
    assert result.gpp.attrs["ancillary_variables"] == "uncertainty_out"
    assert "grid_mapping" not in result.gpp.attrs
    assert result.gpp.attrs["cell_measures"] == "area: cell_area"
    assert result.gpp.attrs["formula_terms"] == "a: coefficient"
    assert "bounds" not in result.time.attrs
    assert result.time.attrs["coordinates"] == "lat lon"


def test_semantic_samples_warn_without_modifying_data(tmp_path) -> None:
    output = tmp_path / "semantic.zarr"
    values = np.asarray(
        [[[-0.25, 0.5], [1.0, 2.0]], [[0.0, 0.5], [1.0, 3.0]]],
        dtype="float32",
    )
    dataset = xr.Dataset(
        {
            "uncertainty": (
                ("time", "lat", "lon"),
                values,
                {"standard_name": "gross_primary_productivity standard_error"},
            )
        },
        coords={"time": [0, 1], "lat": [1.0, 0.0], "lon": [10.0, 11.0]},
    )
    dataset.to_zarr(output, mode="w", consolidated=False, zarr_format=3)
    dataset.close()

    report = validate_semantic_samples(output)

    assert report["status"] == "warning"
    assert report["checks"]["uncertainty"]["violations"]
    with xr.open_zarr(output, consolidated=False, chunks=None) as unchanged:
        np.testing.assert_equal(unchanged.uncertainty.values, values)
