from __future__ import annotations

import shutil
import json
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import xarray as xr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.application.services import SourceInspectionConfig, inspect_source  # noqa: E402
from fast_nc_zarr.pipeline.engine import preview_pipeline, run_pipeline  # noqa: E402
from fast_nc_zarr.pipeline.models import (  # noqa: E402
    PipelineConfig,
    PipelineConversionOptions,
    PipelineFinalizationOptions,
    PipelineGeneralConfig,
    PipelineResamplingOptions,
)
from fast_nc_zarr.pipeline.planner import build_pipeline_plan  # noqa: E402


ROOT = Path("/tmp/codex_test/fast_nc_zarr_pipeline_tests")


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)
        ROOT.mkdir(parents=True)
        (ROOT / "canonical").mkdir()
        times = np.asarray(["2001-01-01", "2001-01-09"], dtype="datetime64[ns]")
        lat = np.asarray([0.025 + 0.05 * index for index in range(8)], dtype="float64")
        lon = np.asarray([-0.175 + 0.05 * index for index in range(8)], dtype="float64")
        values = np.arange(times.size * lat.size * lon.size, dtype="float32").reshape(
            times.size, lat.size, lon.size
        )
        dataset = xr.Dataset(
            {"value": (("time", "lat", "lon"), values)},
            coords={"time": times, "lat": lat, "lon": lon},
            attrs={"title": "pipeline test"},
        )
        dataset.to_netcdf(ROOT / "input.nc", engine="h5netcdf")
        dataset.close()
        canonical = xr.Dataset(
            {
                "value": (
                    ("time", "lat", "lon"),
                    np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4),
                )
            },
            coords={
                "time": times,
                "lat": np.asarray([0.25, 0.15, 0.05], dtype="float64"),
                "lon": np.asarray([-0.15, -0.05, 0.05, 0.15], dtype="float64"),
            },
        )
        canonical.to_netcdf(ROOT / "canonical" / "input.nc", engine="h5netcdf")
        canonical.close()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def _config(self, output: Path, *, resolution: float = 0.1) -> PipelineConfig:
        return PipelineConfig(
            general=PipelineGeneralConfig(
                output=output,
                temporary_dir=ROOT / "temporary",
                time_start="2001-01-01",
                time_end="2001-01-09",
                lat_min=0.1,
                lat_max=0.3,
                lon_min=-0.1,
                lon_max=0.1,
                resolution=resolution,
            ),
            conversion=PipelineConversionOptions(
                auto_tune=False,
                max_workers=1,
            ),
            resampling=PipelineResamplingOptions(
                method="bilinear",
                compute_workers=1,
                space_workers=1,
                tile_size=2,
                time_block=1,
            ),
            finalization=PipelineFinalizationOptions(
                strategy="time",
                target_mib=1,
                compression="fast",
                workers=1,
            ),
        )

    def test_source_window_expands_beyond_target_boundary(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT,
                mode="complete",
                engine="h5netcdf",
                workers=1,
            )
        )
        plan = build_pipeline_plan(inspection, self._config(ROOT / "planned.zarr"))
        self.assertTrue(plan.needs_resample)
        self.assertLess(plan.source_read_window.lat_bounds[0], 0.1)
        self.assertLess(plan.source_read_window.lon_bounds[0], -0.1)
        self.assertEqual(plan.target_grid.lat[0], 0.25)
        np.testing.assert_allclose(plan.target_grid.lat[-1], 0.15)
        np.testing.assert_allclose(plan.target_grid.lon[0], -0.05)
        np.testing.assert_allclose(plan.target_grid.lon[-1], 0.05)

    def test_pipeline_writes_canonical_target_zarr(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT,
                mode="complete",
                engine="h5netcdf",
                workers=1,
            )
        )
        output = ROOT / "result.zarr"
        plan = preview_pipeline(inspection, self._config(output))
        self.assertEqual(plan.conversion_chunks, (1, 6, 6))
        result = run_pipeline(inspection, self._config(output), progress=False)
        self.assertTrue(result["needs_resample"])
        with xr.open_zarr(output, consolidated=False, chunks=None, decode_times=False) as dataset:
            self.assertEqual(dataset["value"].dims, ("time", "lat", "lon"))
            self.assertEqual(dataset.sizes["lat"], 2)
            self.assertEqual(dataset.sizes["lon"], 2)
            self.assertGreater(float(dataset["value"].isel(time=0).mean()), 0.0)
            self.assertTrue(np.all(np.diff(dataset.lat.values) < 0))
            self.assertTrue(np.all(np.diff(dataset.lon.values) > 0))
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        math_validation = manifest["stages"]["resampling"]["mathematical_validation"]
        self.assertEqual(math_validation["reference_mode"], "converted-source-crop")
        self.assertGreater(math_validation["comparisons"], 0)
        self.assertGreater(math_validation["weights_built"], 0)
        self.assertLessEqual(math_validation["sample_windows"], 6)
        converted = Path(manifest["temporary_root"]) / "source-crop.zarr"
        with xr.open_zarr(converted, consolidated=False, chunks=None, decode_times=False) as dataset:
            self.assertEqual(dataset["value"].encoding["chunks"], (1, 6, 6))

    def test_auto_pipeline_conversion_time_chunk_uses_resampling_batch(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT,
                mode="complete",
                engine="h5netcdf",
                workers=1,
            )
        )
        base = self._config(ROOT / "auto-layout.zarr")
        config = replace(
            base,
            resampling=replace(
                base.resampling,
                tile_size="auto",
                time_block="auto",
            ),
        )
        plan = preview_pipeline(inspection, config)
        # The source's native storage is irrelevant here: the selected two
        # times become one bounded xESMF batch and one temporary time chunk.
        self.assertEqual(plan.conversion_chunks, (2, 6, 6))

    def test_grid_equivalent_pipeline_skips_resampling(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT / "canonical",
                mode="complete",
                engine="h5netcdf",
                workers=1,
            )
        )
        config = self._config(ROOT / "direct-result.zarr")
        config = PipelineConfig(
            general=PipelineGeneralConfig(
                output=config.general.output,
                temporary_dir=config.general.temporary_dir,
                time_start=config.general.time_start,
                time_end=config.general.time_end,
                lat_min=0.1,
                lat_max=0.3,
                lon_min=-0.1,
                lon_max=0.1,
                resolution=0.1,
            ),
            conversion=config.conversion,
            resampling=config.resampling,
            finalization=config.finalization,
        )
        plan = preview_pipeline(inspection, config)
        self.assertFalse(plan.needs_resample)
        result = run_pipeline(inspection, config, progress=False)
        self.assertFalse(result["needs_resample"])
        with xr.open_zarr(config.general.output, consolidated=False, chunks=None, decode_times=False) as dataset:
            self.assertEqual(dataset["value"].dims, ("time", "lat", "lon"))
            np.testing.assert_allclose(dataset.lat.values, [0.25, 0.15])
            np.testing.assert_allclose(dataset.lon.values, [-0.05, 0.05])

    def test_cleanup_removes_only_validated_intermediate_stores(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT,
                mode="complete",
                engine="h5netcdf",
                workers=1,
            )
        )
        config = self._config(ROOT / "cleanup-result.zarr")
        config = replace(
            config,
            general=replace(config.general, cleanup_intermediate=True),
        )
        result = run_pipeline(inspection, config, progress=False)
        root = Path(json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))["temporary_root"])
        self.assertFalse((root / "source-crop.zarr").exists())
        self.assertFalse((root / "resampled.zarr").exists())
        self.assertTrue(config.general.output.exists())

    def test_uncovered_target_tiles_are_nan_without_extrapolation(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT,
                mode="complete",
                engine="h5netcdf",
                workers=1,
            )
        )
        base = self._config(ROOT / "uncovered-result.zarr")
        config = replace(
            base,
            general=replace(
                base.general,
                lat_min=-0.4,
                lat_max=-0.2,
                lon_min=10.0,
                lon_max=10.2,
            ),
        )
        run_pipeline(inspection, config, progress=False)
        with xr.open_zarr(config.general.output, consolidated=False, chunks=None, decode_times=False) as dataset:
            self.assertTrue(np.isnan(dataset["value"].values).all())


if __name__ == "__main__":
    unittest.main()
