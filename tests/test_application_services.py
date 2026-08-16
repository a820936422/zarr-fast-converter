from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr._backend import rust_capability  # noqa: E402
from fast_nc_zarr.application.services import (  # noqa: E402
    ConversionConfig,
    RechunkConfig,
    SourceInspectionConfig,
    inspect_source,
    inspect_zarr,
    load_inspection_snapshot,
    preview_conversion,
    preview_rechunk,
    run_conversion,
    run_rechunk,
    save_inspection_snapshot,
)
from fast_nc_zarr.application.desktop_worker.worker import _pipeline_config  # noqa: E402
from fast_nc_zarr.models import VariableTransform  # noqa: E402


ROOT = Path("/tmp/codex_test/fast_nc_zarr_services")
_RUST_ZARR_READY = rust_capability().supported and "zarr.rechunk_f32" in rust_capability().operations


class ApplicationServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)
        (ROOT / "source").mkdir(parents=True)
        lat = np.asarray([30, 20, 10], dtype="float32")
        lon = np.asarray([100, 110, 120, 130], dtype="float32")
        for index in range(2):
            dataset = xr.Dataset(
                {"value": (("time", "lat", "lon"), np.full((1, 3, 4), index, dtype="float32"))},
                coords={
                    "time": np.asarray([np.datetime64("2001-01-01") + np.timedelta64(index, "D")]),
                    "lat": lat,
                    "lon": lon,
                },
            )
            dataset.to_netcdf(ROOT / "source" / f"day-{index}.nc", engine="h5netcdf")
            dataset.close()
        dataset = xr.Dataset(
            {"value": (("time", "lat", "lon"), np.arange(24, dtype="float32").reshape(2, 3, 4))},
            coords={"time": np.arange(2), "lat": lat, "lon": lon},
        )
        dataset.to_zarr(ROOT / "input.zarr", zarr_format=3, mode="w", consolidated=False)
        dataset.close()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_source_service_and_conversion_preview(self) -> None:
        result = inspect_source(SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1))
        preview = preview_conversion(
            result,
            ConversionConfig(ROOT / "output.zarr", time_start="2001-01-01", time_end="2001-01-02"),
        )
        self.assertEqual(result.mode, "complete")
        self.assertEqual(preview.selection.shape, (2, 3, 4))

    def test_source_snapshot_round_trip_restores_conversion_inventory(self) -> None:
        result = inspect_source(SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1))
        snapshot = ROOT / "source-snapshot.json"
        save_inspection_snapshot(result, snapshot)
        restored = load_inspection_snapshot(snapshot)
        preview = preview_conversion(
            restored,
            ConversionConfig(ROOT / "restored-output.zarr", time_start="2001-01-01", time_end="2001-01-02"),
        )
        self.assertEqual(restored.source_inventory.source_mode, "dimension")
        self.assertEqual(preview.selection.shape, (2, 3, 4))

    def test_source_snapshot_rejects_modified_file(self) -> None:
        result = inspect_source(SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1))
        snapshot = ROOT / "stale-source-snapshot.json"
        save_inspection_snapshot(result, snapshot)
        source = ROOT / "source" / "day-0.nc"
        original = source.stat().st_mtime_ns
        os.utime(source, ns=(original + 1_000_000, original + 1_000_000))
        try:
            with self.assertRaisesRegex(ValueError, "快照已过期"):
                load_inspection_snapshot(snapshot)
        finally:
            os.utime(source, ns=(original, original))

    def test_conversion_supports_variable_rename_and_transform(self) -> None:
        result = inspect_source(SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1))
        output = ROOT / "renamed-output.zarr"
        run_conversion(
            result,
            ConversionConfig(
                output,
                time_start="2001-01-01",
                time_end="2001-01-02",
                variable_names={"value": "renamed_value"},
                variable_transforms={
                    "value": VariableTransform(fill_values=(0,), scale_factor=2, output_fill=-7)
                },
                max_workers=1,
                validate=True,
            ),
        )
        with xr.open_zarr(output, consolidated=False, chunks=None, mask_and_scale=False) as dataset:
            self.assertIn("renamed_value", dataset.data_vars)
            self.assertNotIn("value", dataset.data_vars)
            expected = np.stack([np.full((3, 4), -7, dtype="float32"), np.full((3, 4), 2, dtype="float32")])
            np.testing.assert_equal(dataset.renamed_value.values, expected)

    def test_zarr_service_and_rechunk_preview(self) -> None:
        result = inspect_zarr(ROOT / "input.zarr")
        preview = preview_rechunk(RechunkConfig(ROOT / "input.zarr", ROOT / "output.zarr", workers=1), result.zarr_info)
        self.assertEqual(result.zarr_info.zarr_format, 3)
        self.assertEqual(preview.plan.strategy, "time")

    def test_rechunk_tune_budget_is_forwarded_to_engine(self) -> None:
        result = inspect_zarr(ROOT / "input.zarr")
        config = RechunkConfig(
            ROOT / "input.zarr",
            ROOT / "budget-output.zarr",
            workers=1,
            tune_budget_seconds=12.5,
        )
        with patch(
            "fast_nc_zarr.application.services.core_run_rechunk",
            return_value={"backend": "python"},
        ) as core_run:
            run_rechunk(config, result.zarr_info)
        self.assertEqual(core_run.call_args.kwargs["tune_budget_seconds"], 12.5)

    def test_desktop_payload_restores_rechunk_tune_budget(self) -> None:
        config = _pipeline_config(
            {
                "output": str(ROOT / "payload-output.zarr"),
                "rechunk_tune_budget": 17.5,
            }
        )
        self.assertEqual(config.chunking.tune_budget, 17.5)

    @unittest.skipUnless(_RUST_ZARR_READY, "Rust Zarr native extension is not built")
    def test_zarr_service_rust_backend_publishes_valid_output(self) -> None:
        output = ROOT / "rust-service-output.zarr"
        result = inspect_zarr(ROOT / "input.zarr")
        metrics = run_rechunk(
            RechunkConfig(
                ROOT / "input.zarr",
                output,
                strategy="custom",
                custom_chunks=(1, 3, 4),
                workers=1,
                backend="rust",
                validate=True,
            ),
            result.zarr_info,
        )
        self.assertEqual(metrics["execution_path"], "rust-streaming-target-chunk")
        self.assertEqual(Path(metrics["output"]), output)
        self.assertEqual(metrics["target_chunks"], [1, 3, 4])
        self.assertGreaterEqual(metrics["resolved_workers"], 1)
        self.assertGreater(metrics["peak_bytes_per_worker"], 0)
        with xr.open_zarr(output, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False) as dataset:
            np.testing.assert_array_equal(dataset["value"].values, np.arange(24, dtype="float32").reshape(2, 3, 4))


if __name__ == "__main__":
    unittest.main()
