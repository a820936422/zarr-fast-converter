from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np
import xarray as xr
from zarr.codecs import BloscCodec

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.resampling.engine import (  # noqa: E402
    ResampleExecutionError,
    _mask_missing,
    _resolve_local_source_window,
    _tile_target,
    _time_slices,
    format_plan,
    plan_resample,
    run_resample,
)
from fast_nc_zarr.resampling.autotune import resolve_auto_tile_size  # noqa: E402
from fast_nc_zarr.resampling.grid import (  # noqa: E402
    GridInspectionError,
    RESAMPLING_METHODS,
    build_target_grid,
    inspect_grid,
)
from fast_nc_zarr.resampling.inspection import inspect_resample_input  # noqa: E402
from fast_nc_zarr.resampling.models import GridInfo, ResampleConfig  # noqa: E402
from fast_nc_zarr.resampling.replacements import parse_replacement_rules  # noqa: E402


ROOT = Path("/tmp/codex_test/fast_nc_zarr_resampling_tests")


class ResamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(ROOT, ignore_errors=True)
        ROOT.mkdir(parents=True)
        values = np.arange(2 * 4 * 4, dtype="float32").reshape(2, 4, 4)
        values[0, 1, 1] = np.nan
        reordered = np.transpose(values, (1, 0, 2))
        dataset = xr.Dataset(
            {
                "value": (
                    ("time", "lat", "lon"),
                    values,
                    {"units": "test"},
                ),
                "reordered": (("lat", "time", "lon"), reordered),
            },
            coords={
                "time": np.arange(2, dtype="int64"),
                "lat": np.asarray([3.5, 2.5, 1.5, 0.5], dtype="float32"),
                "lon": np.asarray([0.5, 1.5, 2.5, 3.5], dtype="float32"),
            },
            attrs={"title": "resampling test"},
        )
        dataset.to_zarr(
            ROOT / "input.zarr",
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={
                "value": {
                    "chunks": (1, 2, 2),
                    "compressors": [BloscCodec(cname="zstd", clevel=1, shuffle="shuffle")],
                },
                "reordered": {
                    "chunks": (2, 1, 2),
                    "compressors": [BloscCodec(cname="zstd", clevel=1, shuffle="shuffle")],
                },
            },
        )
        dataset.close()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_missing_value_attribute_is_masked_when_fill_encoding_is_nan(self) -> None:
        variable = xr.DataArray(
            np.asarray([[1.0, -9999.0]], dtype="float32"),
            dims=("lat", "lon"),
            attrs={"missing_value": -9999.0},
        )
        variable.encoding["_FillValue"] = np.nan
        masked = _mask_missing(variable, None)
        self.assertEqual(float(masked.values[0, 0]), 1.0)
        self.assertTrue(np.isnan(masked.values[0, 1]))

    def test_grid_inspection_and_target_resolution(self) -> None:
        info, grid = inspect_grid(ROOT / "input.zarr")
        self.assertEqual(info.shape, (2, 4, 4))
        self.assertEqual(grid.lat_resolution, 1.0)
        self.assertEqual(grid.lon_resolution, 1.0)
        self.assertTrue(grid.lat_descending)
        target = build_target_grid(grid, 2.0)
        self.assertEqual(target.dimensions, {"lat": 2, "lon": 2})
        self.assertTrue(target.lat[0] > target.lat[-1])

    def test_plan_lists_xesmf_methods_and_preserves_chunks(self) -> None:
        inspection = inspect_resample_input(ROOT / "input.zarr")
        self.assertEqual(
            RESAMPLING_METHODS,
            (
                "bilinear",
                "conservative",
                "conservative_normed",
                "patch",
                "nearest_s2d",
                "nearest_d2s",
            ),
        )
        plan = plan_resample(
            ResampleConfig(
                ROOT / "input.zarr",
                ROOT / "planned.zarr",
                resolution=2.0,
                method="bilinear",
            ),
            inspection,
        )
        self.assertEqual(plan.output_chunks["value"], (1, 2, 2))
        self.assertEqual(plan.output_chunks["reordered"], (2, 1, 2))
        self.assertEqual(plan.tile_size_requested, "auto")
        self.assertEqual(plan.time_block_requested, "auto")
        # Input time chunks are one, but the auto planner is allowed to batch
        # multiple stored chunks for one vectorized xESMF call.
        self.assertEqual(plan.time_block, 2)
        self.assertIsNotNone(plan.auto_tile)
        self.assertIn("自动空间块依据", format_plan(plan))

    def test_auto_tile_uses_resolution_chunks_dtype_and_memory_budget(self) -> None:
        inspection = inspect_resample_input(ROOT / "input.zarr")
        target = build_target_grid(inspection.grid, 2.0)
        decision = resolve_auto_tile_size(
            inspection.info,
            inspection.grid,
            target,
            method="bilinear",
            skipna=True,
            time_block=1,
            compute_workers=1,
            available_bytes=512 * 1024**2,
            total_bytes=1024 * 1024**2,
        )
        self.assertEqual(decision.tile_size, 2)
        self.assertEqual(decision.source_window, (4, 4))
        self.assertEqual(decision.ratio_lat, 2.0)
        self.assertEqual(decision.ratio_lon, 2.0)
        self.assertIn(decision.worst_variable, {"value", "reordered"})
        self.assertGreater(decision.source_chunk_bytes, 0)

    def test_time_batches_can_span_stored_time_chunks(self) -> None:
        inspection = inspect_resample_input(ROOT / "input.zarr")
        value = next(item for item in inspection.info.data_variables if item.name == "value")
        self.assertEqual(value.chunks[0], 1)
        self.assertEqual(list(_time_slices(value, 2)), [(0, 2)])

    def test_periodic_edge_tile_keeps_a_local_longitude_window(self) -> None:
        grid = GridInfo(
            path=ROOT / "periodic.zarr",
            lat=np.asarray([67.5, 22.5, -22.5, -67.5]),
            lon=np.asarray([-135.0, -45.0, 45.0, 135.0]),
            lat_bounds=np.asarray([90.0, 45.0, 0.0, -45.0, -90.0]),
            lon_bounds=np.asarray([-180.0, -90.0, 0.0, 90.0, 180.0]),
            lat_resolution=45.0,
            lon_resolution=90.0,
            lat_descending=True,
            lon_descending=False,
            lat_uniform=True,
            lon_uniform=True,
        )
        target = build_target_grid(grid, 90.0, extent="source")
        edge_tile = _tile_target(target, 0, 1, 0, 1)
        _target, lat_slice, lon_slice = _resolve_local_source_window(
            grid,
            edge_tile,
            "conservative",
        )
        self.assertIsNotNone(lat_slice)
        self.assertIsNotNone(lon_slice)
        self.assertEqual(lon_slice.start, 0)
        self.assertLess(lon_slice.stop, grid.lon.size)

    def test_manual_tile_size_skips_auto_estimate(self) -> None:
        inspection = inspect_resample_input(ROOT / "input.zarr")
        plan = plan_resample(
            ResampleConfig(
                ROOT / "input.zarr",
                ROOT / "manual.zarr",
                resolution=2.0,
                tile_size=1,
            ),
            inspection,
        )
        self.assertEqual(plan.tile_size, 1)
        self.assertEqual(plan.tile_size_requested, 1)
        self.assertIsNone(plan.auto_tile)

    def test_bilinear_resampling_preserves_chunks_codec_and_time(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "bilinear.zarr"
        inspection = inspect_resample_input(source)
        config = ResampleConfig(source, output, resolution=2.0, method="bilinear")
        metrics = run_resample(config, plan_resample(config, inspection), progress=False)
        self.assertGreater(metrics["physical_bytes"], 0)
        timing = metrics["tile_timing"]
        self.assertEqual(int(timing["tiles"]), 1)
        self.assertGreaterEqual(timing["read_seconds"], 0.0)
        self.assertGreaterEqual(timing["regrid_seconds"], 0.0)
        self.assertGreaterEqual(timing["write_seconds"], 0.0)
        with xr.open_zarr(output, consolidated=False, chunks=None, mask_and_scale=False) as dataset:
            self.assertEqual(dataset["value"].dims, ("time", "lat", "lon"))
            self.assertEqual(dataset["value"].encoding["chunks"], (1, 2, 2))
            self.assertEqual(dataset["value"].encoding["compressors"][0].cname, "zstd")
            self.assertEqual(dataset["reordered"].dims, ("lat", "time", "lon"))
            self.assertEqual(dataset["reordered"].encoding["chunks"], (2, 1, 2))
            self.assertEqual(
                dataset["reordered"].encoding["compressors"][0].cname,
                "zstd",
            )
            self.assertEqual(dataset.sizes["lat"], 2)
            self.assertEqual(dataset.sizes["lon"], 2)
            np.testing.assert_array_equal(dataset.time.values, np.arange(2))
            self.assertTrue(np.isfinite(dataset.value.values).any())

    def test_before_and_after_literal_replacements_are_fused_into_tiles(self) -> None:
        output = ROOT / "replacement-literal.zarr"
        metrics = run_resample(
            ResampleConfig(
                ROOT / "input.zarr",
                output,
                resolution=1.0,
                method="nearest_s2d",
                space_workers=1,
                before_replacements=parse_replacement_rules("<5", "5"),
                after_replacements=parse_replacement_rules(">25", "25"),
            ),
            progress=False,
        )
        with xr.open_zarr(output, consolidated=False, chunks=None) as dataset:
            values = dataset.value.values
            self.assertGreaterEqual(float(np.nanmin(values)), 5.0)
            self.assertLessEqual(float(np.nanmax(values)), 25.0)
            self.assertTrue(np.isnan(values).any())
        self.assertEqual(
            metrics["replacement_statistics"]["before_mode"],
            "not_required",
        )

    def test_data_dependent_after_replacement_uses_unmodified_output_statistics(self) -> None:
        output = ROOT / "replacement-statistic.zarr"
        metrics = run_resample(
            ResampleConfig(
                ROOT / "input.zarr",
                output,
                resolution=1.0,
                method="nearest_s2d",
                space_workers=1,
                after_replacements=parse_replacement_rules(">median", "median"),
                statistics_policy="exact",
            ),
            progress=False,
        )
        statistics = metrics["replacement_statistics"]["after"]
        with xr.open_zarr(output, consolidated=False, chunks=None) as dataset:
            self.assertLessEqual(
                float(np.nanmax(dataset.value.values)),
                float(statistics["value"]["median"]),
            )
            self.assertEqual(
                json.loads(dataset.attrs["resampling_after_replacements"]),
                [[">median", "median"]],
            )
        self.assertEqual(metrics["replacement_statistics"]["after_mode"], "exact")

    def test_float32_mode_changes_float_output_dtype(self) -> None:
        source = ROOT / "float64-input.zarr"
        output = ROOT / "float32-output.zarr"
        values = np.arange(2 * 4 * 4, dtype="float64").reshape(2, 4, 4)
        dataset = xr.Dataset(
            {"value": (("time", "lat", "lon"), values)},
            coords={
                "time": np.arange(2, dtype="int64"),
                "lat": np.asarray([3.5, 2.5, 1.5, 0.5], dtype="float32"),
                "lon": np.asarray([0.5, 1.5, 2.5, 3.5], dtype="float32"),
            },
        )
        dataset.to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={"value": {"chunks": (1, 2, 2)}},
        )
        dataset.close()
        inspection = inspect_resample_input(source)
        plan = plan_resample(
            ResampleConfig(
                source,
                output,
                resolution=2.0,
                compute_dtype="float32",
                space_workers=1,
            ),
            inspection,
        )
        self.assertEqual(plan.compute_dtype, "float32")
        run_resample(
            ResampleConfig(
                source,
                output,
                resolution=2.0,
                compute_dtype="float32",
                space_workers=1,
            ),
            plan,
            progress=False,
        )
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            self.assertEqual(result.value.dtype, np.dtype("float32"))
            self.assertEqual(result.value.encoding["chunks"], (1, 2, 2))

    def test_time_intermediate_is_merged_once_and_cleaned(self) -> None:
        source = ROOT / "large-time-chunk.zarr"
        output = ROOT / "large-time-output.zarr"
        temporary = ROOT / "resample-temporary"
        values = np.arange(4 * 4 * 4, dtype="float32").reshape(4, 4, 4)
        dataset = xr.Dataset(
            {"value": (("time", "lat", "lon"), values)},
            coords={
                "time": np.arange(4, dtype="int64"),
                "lat": np.asarray([3.5, 2.5, 1.5, 0.5], dtype="float32"),
                "lon": np.asarray([0.5, 1.5, 2.5, 3.5], dtype="float32"),
            },
        )
        dataset.to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={"value": {"chunks": (4, 2, 2)}},
        )
        dataset.close()
        config = ResampleConfig(
            source,
            output,
            resolution=2.0,
            time_block=1,
            space_workers=1,
            temporary_dir=temporary,
        )
        metrics = run_resample(config, progress=False)
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            self.assertEqual(result.value.encoding["chunks"], (4, 2, 2))
            self.assertEqual(result.value.shape, (4, 2, 2))
            self.assertTrue(np.isfinite(result.value.values).all())
        self.assertTrue(temporary.is_dir())
        self.assertEqual(metrics["temporary_dir"], str(temporary.resolve()))
        self.assertEqual(list(temporary.glob(".*.tmp")), [])
        self.assertEqual(list(temporary.glob(".resample-buffer-*.bin")), [])

    def test_spatial_compute_tiles_are_decoupled_from_final_chunks(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "spatial-intermediate-output.zarr"
        temporary = ROOT / "spatial-intermediate-temporary"
        config = ResampleConfig(
            source,
            output,
            resolution=1.0,
            method="nearest_s2d",
            tile_size=2,
            time_block=2,
            compute_workers=1,
            space_workers=1,
            temporary_dir=temporary,
        )
        plan = plan_resample(config)
        final_chunks = dict(plan.output_chunks)
        final_chunks["value"] = (2, 4, 4)
        final_chunks["reordered"] = (4, 2, 4)
        metrics = run_resample(
            config,
            replace(plan, output_chunks=final_chunks),
            progress=False,
        )

        self.assertTrue(metrics["used_intermediate"])
        self.assertEqual(int(metrics["tile_timing"]["tiles"]), 4)
        self.assertEqual(metrics["logical_write_amplification"], 2.0)
        self.assertGreater(metrics["throughput_mib_s"], 0.0)
        self.assertGreater(metrics["physical_throughput_mib_s"], 0.0)
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            self.assertEqual(result.value.encoding["chunks"], (2, 4, 4))
            self.assertEqual(result.reordered.encoding["chunks"], (4, 2, 4))
            self.assertTrue(np.isfinite(result.value.values).any())
            self.assertTrue(np.isfinite(result.reordered.values).any())
        self.assertEqual(list(temporary.glob(".*.tmp")), [])
        self.assertEqual(list(temporary.glob(".resample-buffer-*.bin")), [])

    def test_conservative_resampling_uses_derived_bounds(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "conservative.zarr"
        config = ResampleConfig(source, output, resolution=2.0, method="conservative")
        metrics = run_resample(config, progress=False)
        self.assertGreater(metrics["physical_bytes"], 0)
        with xr.open_zarr(output, consolidated=False, chunks=None) as dataset:
            self.assertEqual(dataset.value.shape, (2, 2, 2))

    def test_space_workers_write_complete_output_chunks(self) -> None:
        output = ROOT / "parallel.zarr"
        config = ResampleConfig(
            ROOT / "input.zarr",
            output,
            resolution=1.0,
            method="bilinear",
            tile_size=2,
            time_block=1,
            compute_workers=1,
            space_workers=2,
        )
        run_resample(config, progress=False)
        with xr.open_zarr(output, consolidated=False, chunks=None) as dataset:
            self.assertEqual(dataset.value.shape, (2, 4, 4))
            self.assertEqual(dataset.value.encoding["chunks"], (1, 2, 2))
            self.assertTrue(np.isfinite(dataset.value.values).any())

    def test_streaming_small_tiles_preserves_nearest_d2s_semantics(self) -> None:
        output = ROOT / "nearest-d2s-stream.zarr"
        run_resample(
            ResampleConfig(
                ROOT / "input.zarr",
                output,
                resolution=2.0,
                method="nearest_d2s",
                tile_size=1,
                time_block=1,
            ),
            progress=False,
        )
        with xr.open_zarr(output, consolidated=False, chunks=None) as dataset:
            np.testing.assert_allclose(
                dataset.value.values,
                np.asarray(
                    [
                        [[5.0 / 3.0, 4.5], [10.5, 12.5]],
                        [[18.5, 20.5], [26.5, 28.5]],
                    ],
                    dtype="float32",
                ),
            )
            self.assertEqual(dataset.value.encoding["chunks"], (1, 2, 2))

    def test_global_target_plan_and_boundary_rounding_are_explicit(self) -> None:
        inspection = inspect_resample_input(ROOT / "input.zarr")
        plan = plan_resample(
            ResampleConfig(
                ROOT / "input.zarr",
                ROOT / "global.zarr",
                resolution=3.0,
                extent="global",
            ),
            inspection,
        )
        self.assertEqual(plan.target.dimensions, {"lat": 60, "lon": 120})
        self.assertEqual(plan.target.spatial_extent, (-180.0, 180.0, -90.0, 90.0))

        source_plan = plan_resample(
            ResampleConfig(
                ROOT / "input.zarr",
                ROOT / "rounded.zarr",
                resolution=3.0,
            ),
            inspection,
        )
        self.assertEqual(source_plan.target.spatial_extent, (0.0, 6.0, 0.0, 6.0))

    def test_no_skipna_and_existing_nonempty_output_protection(self) -> None:
        output = ROOT / "no-skipna.zarr"
        config = ResampleConfig(
            ROOT / "input.zarr",
            output,
            resolution=2.0,
            skipna=False,
        )
        run_resample(config, progress=False)
        with xr.open_zarr(output, consolidated=False, chunks=None) as dataset:
            self.assertFalse(bool(dataset.attrs["resampling_skipna"]))

        blocked = ROOT / "blocked.zarr"
        blocked.mkdir()
        (blocked / "do-not-delete.txt").write_text("keep", encoding="utf-8")
        with self.assertRaises(ResampleExecutionError):
            run_resample(
                ResampleConfig(ROOT / "input.zarr", blocked, resolution=2.0),
                progress=False,
            )
        self.assertTrue((blocked / "do-not-delete.txt").is_file())

    def test_irregular_grid_is_rejected_before_xesmf(self) -> None:
        path = ROOT / "irregular.zarr"
        dataset = xr.Dataset(
            {"value": (("time", "lat", "lon"), np.ones((1, 3, 2), dtype="float32"))},
            coords={
                "time": [0],
                "lat": [0.5, 1.5, 3.0],
                "lon": [0.5, 1.5],
            },
        )
        dataset.to_zarr(path, mode="w", zarr_format=3, consolidated=False)
        dataset.close()
        with self.assertRaises(GridInspectionError):
            inspect_resample_input(path)


if __name__ == "__main__":
    unittest.main()
