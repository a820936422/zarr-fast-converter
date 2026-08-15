from __future__ import annotations

import shutil
from pathlib import Path
import sys
import threading
import unittest

import numpy as np
import xarray as xr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.inspection import _normalize_daily_times, inspect_dataset  # noqa: E402
from fast_nc_zarr.application.services import (  # noqa: E402
    ConversionConfig,
    SourceInspectionConfig,
    inspect_source,
    run_conversion,
)
from fast_nc_zarr.time_mapping import (  # noqa: E402
    TimeRule,
    inspect_time_metadata,
    resolve_file_times,
)


ROOT = Path("/tmp/codex_test/fast_nc_zarr_time_mapping_tests")


class TimeMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)
        (ROOT / "filename").mkdir(parents=True)
        (ROOT / "filename-processing").mkdir(parents=True)
        (ROOT / "hybrid").mkdir(parents=True)
        (ROOT / "complete").mkdir(parents=True)
        lat = np.asarray([10, 20], dtype="float32")
        lon = np.asarray([30, 40], dtype="float32")

        filename_data = xr.Dataset(
            {"value": (("lat", "lon"), np.ones((2, 2), dtype="float32"))},
            coords={"lat": lat, "lon": lon},
        )
        filename_data.to_netcdf(ROOT / "filename" / "product_2001001.nc", engine="h5netcdf")
        for name in (
            "GLASS14B01.V10.A2001001.2023068.hdf",
            "GLASS14B01.V10.A2001009.2023068.hdf",
            "GLASS14B01.V10.A2001017.2025133.hdf",
        ):
            filename_data.to_netcdf(ROOT / "filename-processing" / name, engine="h5netcdf")
        filename_data.close()

        for year in (2001, 2002):
            hybrid_data = xr.Dataset(
                {
                    "value": (
                        ("time", "lat", "lon"),
                        np.ones((2, 2, 2), dtype="float32"),
                    )
                },
                coords={"time": np.asarray([1, 9], dtype="int32"), "lat": lat, "lon": lon},
            )
            hybrid_data.time.attrs.update({"units": "day", "long_name": "day of year"})
            hybrid_data.to_netcdf(ROOT / "hybrid" / f"lai_8-day_0.1_{year}.nc", engine="h5netcdf")
            hybrid_data.close()

        complete_data = xr.Dataset(
            {
                "value": (
                    ("time", "lat", "lon"),
                    np.ones((2, 2, 2), dtype="float32"),
                )
            },
            coords={
                "time": np.asarray([0, 1], dtype="int32"),
                "lat": lat,
                "lon": lon,
            },
        )
        complete_data.time.attrs["units"] = "days since 2001-01-01 00:00:00"
        complete_data.to_netcdf(ROOT / "complete" / "SMrz_2001.nc", engine="h5netcdf")
        complete_data.close()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_filename_full_time_is_discovered_without_time_dimension(self) -> None:
        result = inspect_time_metadata(ROOT / "filename", requested_engine="h5netcdf")
        full = [item for item in result.options if item.ref.component == "full"]
        self.assertEqual(len(full), 1)
        self.assertEqual(full[0].sample, "2001001")
        self.assertFalse(result.time_dimension.exists)
        dates = resolve_file_times(
            "product_2001001.nc",
            (),
            {},
            TimeRule(full=full[0].ref),
            result.filename_fields,
        )
        self.assertEqual(str(dates[0])[:10], "2001-01-01")

    def test_repeated_processing_date_suggests_unique_filename_date(self) -> None:
        result = inspect_time_metadata(ROOT / "filename-processing", requested_engine="h5netcdf")
        self.assertIsNotNone(result.suggested_rule)
        self.assertEqual(result.suggested_rule.full.source, "filename")
        self.assertEqual(result.suggested_rule.full.index, 3)

    def test_time_metadata_reports_progress_phases(self) -> None:
        progress: list[tuple[int, int, str]] = []
        inspect_time_metadata(
            ROOT / "filename",
            requested_engine="h5netcdf",
            progress_callback=lambda completed, total, message: progress.append((completed, total, message)),
        )
        self.assertGreaterEqual(len(progress), 4)
        self.assertEqual(progress[0][:2], (0, 4))
        self.assertEqual(progress[-1][:2], (4, 4))
        self.assertIn("时间字段候选", progress[-1][2])

    def test_source_time_full_date_is_suggested(self) -> None:
        result = inspect_time_metadata(ROOT / "complete", requested_engine="h5netcdf")
        self.assertIsNotNone(result.suggested_rule)
        self.assertEqual(result.suggested_rule.full.source, "time")

    def test_time_metadata_inspection_honours_preexisting_cancellation(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()

        with self.assertRaisesRegex(RuntimeError, "任务已取消"):
            inspect_time_metadata(
                ROOT / "complete",
                requested_engine="h5netcdf",
                cancel_event=cancel_event,
            )

    def test_filename_year_and_time_doy_build_hybrid_inventory(self) -> None:
        result = inspect_time_metadata(ROOT / "hybrid", requested_engine="h5netcdf")
        self.assertIsNotNone(result.suggested_rule)
        inventory = inspect_dataset(
            ROOT / "hybrid",
            engine="h5netcdf",
            workers=1,
            progress=False,
            time_rule=result.suggested_rule,
            filename_fields=result.filename_fields,
        )
        self.assertEqual(inventory.source_mode, "hybrid")
        self.assertEqual(
            [str(value)[:10] for value in inventory.times],
            ["2001-01-01", "2001-01-09", "2002-01-01", "2002-01-09"],
        )

    def test_hybrid_time_rule_can_write_normalized_zarr(self) -> None:
        result = inspect_time_metadata(ROOT / "hybrid", requested_engine="h5netcdf")
        source = inspect_source(
            SourceInspectionConfig(
                ROOT / "hybrid",
                engine="h5netcdf",
                workers=1,
                time_rule=result.suggested_rule,
                time_inspection=result,
            )
        )
        output = ROOT / "hybrid-output.zarr"
        run_conversion(
            source,
            ConversionConfig(output, auto_tune=False, max_workers=1, validate=True),
        )
        with xr.open_zarr(output, consolidated=False, chunks=None, mask_and_scale=False) as dataset:
            np.testing.assert_equal(
                dataset.time.values,
                np.asarray(
                    [
                        "2001-01-01",
                        "2001-01-09",
                        "2002-01-01",
                        "2002-01-09",
                    ],
                    dtype="datetime64[ns]",
                ),
            )


    def test_nonstandard_calendar_is_rejected_explicitly(self) -> None:
        class NonStandardDate:
            calendar = "360_day"

            def __str__(self) -> str:
                return "2001-01-01"

        with self.assertRaisesRegex(ValueError, "不支持 calendar"):
            _normalize_daily_times((NonStandardDate(),), ROOT / "calendar.nc")

if __name__ == "__main__":

    unittest.main()
