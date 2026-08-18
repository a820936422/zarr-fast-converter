"""P1 verification tests for the v1.7.7 optimization plan.

These tests close the remaining P1 gaps that can be verified without external
full-scale datasets:

- CF auxiliary metadata reference sanitization (bounds, climatology, geometry,
  grid_mapping, cell_measures, formula_terms, coordinates).
- ENOSPC during atomic publication preserves an existing target.
- Native multi-variable region reads remain parity with Python/xarray.
"""

from __future__ import annotations

import errno
import importlib
import json
import os
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

from fast_nc_zarr.metadata import sanitize_cf_references
from fast_nc_zarr.publication import publish_staging

try:
    _NATIVE = importlib.import_module("fast_nc_zarr._native")
    _RUST_ZARR_READY = True
except (ImportError, ModuleNotFoundError):
    _NATIVE = None
    _RUST_ZARR_READY = False

ROOT = Path("/tmp/codex_test/fast_nc_zarr_p1_verification")


def _zarr_marker(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group"}),
        encoding="utf-8",
    )


class P1VerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)
        ROOT.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_cf_reference_sanitization_covers_all_reference_attributes(self) -> None:
        dataset = xr.Dataset(
            {
                "value": (
                    ("time", "lat", "lon"),
                    np.zeros((1, 2, 2), dtype="float32"),
                    {
                        "grid_mapping": "crs",
                        "coordinates": "lat lon missing_coord",
                        "ancillary_variables": "value_status",
                        "cell_measures": "area: cell_area",
                        "formula_terms": "a: A b: B",
                    },
                ),
                "time_bnds": (("time", "nv"), np.zeros((1, 2), dtype="float64")),
                "clim": (("time", "nv"), np.zeros((1, 2), dtype="float64")),
                "geom": ((), 0, {"long_name": "geometry container"}),
                "crs": ((), 0, {"long_name": "Coordinate Reference System"}),
                "cell_area": (("lat", "lon"), np.ones((2, 2), dtype="float32")),
                "value_status": (("time", "lat", "lon"), np.zeros((1, 2, 2), dtype="uint8")),
                "A": (("time",), np.zeros(1, dtype="float32")),
                "B": (("time",), np.zeros(1, dtype="float32")),
            },
            coords={
                "time": np.asarray([0], dtype="int64"),
                "lat": np.asarray([1.0, 0.0], dtype="float32"),
                "lon": np.asarray([0.0, 1.0], dtype="float32"),
            },
        )
        dataset["time"].attrs["bounds"] = "time_bnds"
        dataset["time"].attrs["climatology"] = "clim"
        dataset["time"].attrs["geometry"] = "geom"

        sanitized = sanitize_cf_references(dataset)
        self.assertEqual(sanitized["value"].attrs["grid_mapping"], "crs")
        self.assertEqual(sanitized["value"].attrs["coordinates"], "lat lon")
        self.assertNotIn("missing_coord", sanitized["value"].attrs["coordinates"])
        self.assertEqual(sanitized["value"].attrs["ancillary_variables"], "value_status")
        self.assertEqual(sanitized["value"].attrs["cell_measures"], "area: cell_area")
        self.assertEqual(sanitized["value"].attrs["formula_terms"], "a: A b: B")
        self.assertEqual(sanitized["time"].attrs["bounds"], "time_bnds")
        self.assertEqual(sanitized["time"].attrs["climatology"], "clim")
        self.assertEqual(sanitized["time"].attrs["geometry"], "geom")

    def test_cf_reference_sanitization_drops_absent_references(self) -> None:
        dataset = xr.Dataset(
            {
                "value": (
                    ("time", "lat", "lon"),
                    np.zeros((1, 2, 2), dtype="float32"),
                    {
                        "grid_mapping": "missing_crs",
                        "coordinates": "missing_a missing_b",
                        "cell_measures": "area: missing_area",
                        "formula_terms": "a: missing_a b: missing_b",
                    },
                ),
            },
            coords={
                "time": np.asarray([0], dtype="int64"),
                "lat": np.asarray([1.0, 0.0], dtype="float32"),
                "lon": np.asarray([0.0, 1.0], dtype="float32"),
            },
        )
        sanitized = sanitize_cf_references(dataset)
        self.assertNotIn("grid_mapping", sanitized["value"].attrs)
        self.assertNotIn("coordinates", sanitized["value"].attrs)
        self.assertNotIn("cell_measures", sanitized["value"].attrs)
        self.assertNotIn("formula_terms", sanitized["value"].attrs)

    def test_publish_enospc_restores_existing_target(self) -> None:
        target = ROOT / "output.zarr"
        staging = ROOT / ".output.zarr.test.tmp"
        _zarr_marker(target)
        (target / "old").write_text("old", encoding="utf-8")
        _zarr_marker(staging)
        (staging / "new").write_text("new", encoding="utf-8")

        original_replace = os.replace
        calls = 0

        def enospc_on_new_store(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError(errno.ENOSPC, "No space left on device")
            return original_replace(source, destination)

        with patch("fast_nc_zarr.publication.os.replace", side_effect=enospc_on_new_store):
            with self.assertRaisesRegex(OSError, "No space left on device"):
                publish_staging(staging, target, "test")

        self.assertTrue((target / "old").is_file())
        self.assertFalse((target / "new").exists())
        self.assertTrue(staging.is_dir())

    @unittest.skipUnless(_RUST_ZARR_READY, "Rust Zarr native extension is not built")
    def test_native_multi_variable_region_parity(self) -> None:
        source = ROOT / "multi-region.zarr"
        values_a = np.arange(2 * 4 * 4, dtype="float32").reshape(2, 4, 4)
        values_b = np.arange(2 * 4 * 4, dtype="float32").reshape(2, 4, 4) + 100
        dataset = xr.Dataset(
            {
                "a": (("time", "lat", "lon"), values_a),
                "b": (("time", "lat", "lon"), values_b),
            },
            coords={
                "time": np.asarray([0, 1], dtype="int64"),
                "lat": np.asarray([3.5, 2.5, 1.5, 0.5], dtype="float32"),
                "lon": np.asarray([0.5, 1.5, 2.5, 3.5], dtype="float32"),
            },
        )
        dataset.to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={
                "a": {"chunks": (1, 2, 2)},
                "b": {"chunks": (1, 2, 2)},
            },
        )
        dataset.close()

        for name, expected in (("a", values_a), ("b", values_b)):
            with self.subTest(variable=name):
                region = _NATIVE.read_region_f32(
                    str(source), f"/{name}", [0, 1, 1], [2, 2, 3]
                )
                np.testing.assert_array_equal(region, expected[0:2, 1:3, 1:4].ravel())


if __name__ == "__main__":
    unittest.main()
