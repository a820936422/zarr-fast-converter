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

from fast_nc_zarr.raw_validation import main, validate_raw_tree  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
