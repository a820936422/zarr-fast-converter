from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest

import numpy as np
import xarray as xr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.raw_validation import _axis_report, main, validate_raw_tree  # noqa: E402
from fast_nc_zarr.metadata import sanitize_cf_references  # noqa: E402
from fast_nc_zarr.validation import validate_semantic_samples  # noqa: E402


ROOT = Path("/tmp/codex_test/fast_nc_zarr_raw_validation_tests")


def write_daily_slice(path: Path, value: float) -> None:
    dataset = xr.Dataset(
        {
            "value": (
                ("lat", "lon"),
                np.full((3, 4), value, dtype="float32"),
                {"units": "1"},
            )
        },
        coords={
            "lat": np.asarray([30.0, 20.0, 10.0], dtype="float32"),
            "lon": np.asarray([100.0, 110.0, 120.0, 130.0], dtype="float32"),
        },
    )
    dataset.to_netcdf(path, engine="h5netcdf")
    dataset.close()


class RawValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)
        source = ROOT / "input" / "daily-product"
        source.mkdir(parents=True)
        write_daily_slice(source / "product.2001001.nc", 1.0)
        write_daily_slice(source / "product.2001002.nc", 2.0)

    def tearDown(self) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_tree_report_and_conversion_smoke_cover_real_entrypoint_contract(self) -> None:
        smoke_root = ROOT / "smoke"

        report = validate_raw_tree(
            ROOT / "input",
            workers=1,
            sample_files=2,
            smoke_output_root=smoke_root,
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(report["file_count"], 2)
        self.assertEqual(len(report["datasets"]), 1)
        item = report["datasets"][0]
        self.assertEqual(item["name"], "daily-product")
        self.assertEqual(item["time_start"], "2001-01-01")
        self.assertEqual(item["time_end"], "2001-01-02")
        self.assertFalse(item["latitude"]["ascending"])
        self.assertTrue(item["longitude"]["ascending"])
        self.assertEqual(len(item["source_samples"]), 2)
        self.assertTrue((smoke_root / "daily-product.zarr" / "zarr.json").is_file())

    def test_cli_atomically_writes_json_report(self) -> None:
        destination = ROOT / "reports" / "raw-validation.json"

        status = main(
            [
                "--input-root",
                str(ROOT / "input"),
                "--output",
                str(destination),
                "--workers",
                "1",
                "--sample-files",
                "1",
            ]
        )

        self.assertEqual(status, 0)
        payload = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "passed")
        sampled = payload["datasets"][0]["source_samples"][0]["variables"][0]
        self.assertEqual(sampled["name"], "value")
        self.assertFalse(destination.with_name(f".{destination.name}.tmp").exists())

    def test_axis_report_accepts_float32_regular_grid(self) -> None:
        lat = np.linspace(-89.975, 89.975, 3600, dtype="float32")
        lon = np.linspace(-179.975, 179.975, 7200, dtype="float32")
        report = _axis_report(lat, "lat")
        self.assertEqual(report["size"], 3600)
        self.assertTrue(report["ascending"])
        self.assertAlmostEqual(report["step"], 0.05, places=5)
        report_lon = _axis_report(lon, "lon")
        self.assertEqual(report_lon["size"], 7200)

    def test_invalid_time_override_does_not_replace_existing_report(self) -> None:
        destination = ROOT / "raw-validation.json"
        destination.write_text("preserve", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "不存在文件名字段"):
            main(
                [
                    "--input-root",
                    str(ROOT / "input"),
                    "--output",
                    str(destination),
                    "--workers",
                    "1",
                    "--time-field",
                    "daily-product=99",
                ]
            )

        self.assertEqual(destination.read_text(encoding="utf-8"), "preserve")


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

    result = sanitize_cf_references(dataset, renames={"uncertainty": "uncertainty_out"})

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


if __name__ == "__main__":
    unittest.main()
