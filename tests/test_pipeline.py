from __future__ import annotations

from contextlib import redirect_stdout
import io
import shutil
import json
from dataclasses import replace
from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
from collections import namedtuple

import numpy as np
import xarray as xr
import zarr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.application.services import (  # noqa: E402
    SourceInspectionConfig,
    inspect_source,
    inspect_temporary_pipeline,
    inspect_zarr,
    run_pipeline as run_pipeline_service,
)
from fast_nc_zarr.pipeline.engine import (  # noqa: E402
    PipelineExecutionError,
    preview_pipeline,
    run_pipeline,
)
from fast_nc_zarr.pipeline.cli import main as pipeline_main  # noqa: E402
from fast_nc_zarr.pipeline.models import (  # noqa: E402
    PipelineChunkingOptions,
    PipelineCompressionOptions,
    PipelineConfig,
    PipelineConversionOptions,
    PipelineGeneralConfig,
    PipelineInput,
    PipelineOperations,
    PipelineResamplingOptions,
)
from fast_nc_zarr.pipeline.planner import build_pipeline_plan  # noqa: E402
from fast_nc_zarr.models import ConversionPlan, VariableSpec  # noqa: E402
from fast_nc_zarr.resampling.environment import ResamplingEnvironmentError  # noqa: E402


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
                ),
                "quality": (
                    ("time", "lat", "lon"),
                    np.arange(2 * 3 * 4, dtype="int16").reshape(2, 3, 4),
                ),
            },
            coords={
                "time": times,
                "lat": np.asarray([0.25, 0.15, 0.05], dtype="float64"),
                "lon": np.asarray([-0.15, -0.05, 0.05, 0.15], dtype="float64"),
            },
        )
        canonical.to_netcdf(ROOT / "canonical" / "input.nc", engine="h5netcdf")
        canonical.to_zarr(ROOT / "canonical-input.zarr", mode="w", consolidated=False)
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
            ),
            conversion=PipelineConversionOptions(
                auto_tune=False,
                max_workers=1,
            ),
            operations=PipelineOperations(
                resample=True,
                rechunk=True,
                recompress=True,
            ),
            resampling=PipelineResamplingOptions(
                resolution=resolution,
                method="bilinear",
                compute_workers=1,
                space_workers=1,
                tile_size=2,
                time_block=1,
            ),
            chunking=PipelineChunkingOptions(
                strategy="time",
                target_mib=1,
                workers=1,
            ),
            compression=PipelineCompressionOptions(profile="fast"),
        )

    def test_zarr_planner_covers_all_nonempty_operation_combinations(self) -> None:
        inspection = inspect_zarr(ROOT / "canonical-input.zarr")
        for resample, rechunk, recompress in product((False, True), repeat=3):
            if not (resample or rechunk or recompress):
                continue
            with self.subTest(resample=resample, rechunk=rechunk, recompress=recompress):
                base = self._config(ROOT / "zarr-plan.zarr")
                config = replace(
                    base,
                    input=PipelineInput(kind="zarr"),
                    operations=PipelineOperations(resample, rechunk, recompress),
                )
                plan = preview_pipeline(inspection, config)
                self.assertFalse(plan.decision("conversion").requested)
                self.assertEqual(plan.needs_resample, resample)
                self.assertEqual(
                    plan.finalization_required,
                    (rechunk or recompress) and not resample,
                )
                self.assertEqual(plan.decision("rechunking").requested, rechunk)
                self.assertEqual(plan.decision("recompression").requested, recompress)


    def test_auto_compression_reserves_runtime_benchmark_finalization(self) -> None:
        inspection = inspect_zarr(ROOT / "canonical-input.zarr")
        base = self._config(ROOT / "zarr-auto-compression.zarr")
        config = replace(
            base,
            input=PipelineInput(kind="zarr"),
            operations=PipelineOperations(resample=True, recompress=True),
            compression=PipelineCompressionOptions(
                profile="auto", objective="balanced", tune_budget=5
            ),
        )

        plan = preview_pipeline(inspection, config)

        self.assertTrue(plan.finalization_required)
        self.assertEqual(plan.final_compression.profile, "auto")
        self.assertEqual(
            plan.decision("recompression").disposition,
            "executed_as_stage",
        )
        self.assertIn("真实代表性数据", plan.decision("recompression").reason)
    def test_zarr_pipeline_executes_rechunk_only(self) -> None:
        inspection = inspect_zarr(ROOT / "canonical-input.zarr")
        base = self._config(ROOT / "zarr-rechunked.zarr")
        config = replace(
            base,
            input=PipelineInput(kind="zarr"),
            operations=PipelineOperations(rechunk=True),
            validate=False,
        )
        result = run_pipeline(inspection, config, progress=False)
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["input_kind"], "zarr")
        self.assertEqual(manifest["physical_stages"], ["finalization"])
        self.assertTrue((ROOT / "zarr-rechunked.zarr" / "zarr.json").is_file())

    def test_zarr_pipeline_executes_resample_then_recompress(self) -> None:
        inspection = inspect_zarr(ROOT / "canonical-input.zarr")
        base = self._config(ROOT / "zarr-resampled-compressed.zarr", resolution=0.2)
        config = replace(
            base,
            input=PipelineInput(kind="zarr"),
            operations=PipelineOperations(resample=True, recompress=True),
            general=replace(base.general, cleanup_intermediate=True),
            validate=False,
        )
        result = run_pipeline(inspection, config, progress=False)
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["physical_stages"], ["resampling"])
        self.assertEqual(
            manifest["stages"]["resampling"]["status"],
            "published_as_final",
        )
        self.assertEqual(
            manifest["operation_decisions"]["recompression"]["disposition"],
            "fused_into_resampling",
        )
        self.assertGreaterEqual(manifest["logical_io"]["write_amplification"], 1.0)
        self.assertGreater(
            manifest["logical_io"]["avoided_finalization_read_bytes"],
            0,
        )
        self.assertEqual(
            manifest["logical_io"]["avoided_finalization_read_bytes"],
            manifest["logical_io"]["avoided_finalization_write_bytes"],
        )
        self.assertTrue((ROOT / "zarr-resampled-compressed.zarr" / "zarr.json").is_file())

    def test_replacement_rules_prevent_identity_resample_from_becoming_noop(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT / "canonical",
                mode="complete",
                engine="h5netcdf",
                workers=1,
            )
        )
        base = self._config(ROOT / "identity-with-replacement.zarr")
        config = replace(
            base,
            operations=PipelineOperations(resample=True),
            resampling=replace(
                base.resampling,
                before_conditions="<5",
                before_results="5",
            ),
        )
        plan = preview_pipeline(inspection, config)
        self.assertTrue(plan.needs_resample)
        self.assertEqual(plan.decision("resampling").disposition, "executed_as_stage")

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

    def test_planner_covers_all_optional_operation_combinations(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        for resample, rechunk, recompress in product((False, True), repeat=3):
            with self.subTest(
                resample=resample,
                rechunk=rechunk,
                recompress=recompress,
            ):
                base = self._config(ROOT / "combination-plan.zarr")
                config = replace(
                    base,
                    operations=PipelineOperations(
                        resample=resample,
                        rechunk=rechunk,
                        recompress=recompress,
                    ),
                )
                plan = preview_pipeline(inspection, config)
                self.assertEqual(plan.needs_resample, resample)
                self.assertEqual(plan.decision("resampling").requested, resample)
                self.assertEqual(plan.decision("rechunking").requested, rechunk)
                self.assertEqual(plan.decision("recompression").requested, recompress)
                self.assertEqual(plan.final_compression.enabled, recompress)
                self.assertFalse(plan.finalization_required)
                terminal = "fused_into_resampling" if resample else "fused_into_conversion"
                self.assertEqual(
                    plan.decision("rechunking").disposition,
                    terminal if rechunk else "not_requested",
                )
                self.assertEqual(
                    plan.decision("recompression").disposition,
                    terminal if recompress else "not_requested",
                )

    def test_resampling_parameters_are_ignored_when_operation_is_not_selected(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        base = self._config(ROOT / "ignored-resampling-options.zarr")
        invalid_options = replace(base.resampling, resolution=-1, method="invalid")
        conversion_only = replace(
            base,
            operations=PipelineOperations(),
            resampling=invalid_options,
        )
        plan = preview_pipeline(inspection, conversion_only)
        self.assertFalse(plan.needs_resample)
        self.assertEqual(plan.decision("resampling").disposition, "not_requested")
        with self.assertRaisesRegex(ValueError, "目标分辨率"):
            preview_pipeline(
                inspection,
                replace(
                    conversion_only,
                    operations=PipelineOperations(resample=True),
                ),
            )

    def test_conversion_only_plan_supports_single_source_grid_cell(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        base = self._config(ROOT / "single-cell.zarr")
        config = replace(
            base,
            operations=PipelineOperations(),
            general=replace(
                base.general,
                lat_min=0.12,
                lat_max=0.13,
                lon_min=-0.08,
                lon_max=-0.07,
            ),
        )
        plan = preview_pipeline(inspection, config)
        self.assertEqual(plan.target_grid.dimensions, {"lat": 1, "lon": 1})
        self.assertEqual(len(plan.target_grid.lat_bounds), 2)
        self.assertEqual(len(plan.target_grid.lon_bounds), 2)

    def test_conversion_only_normalizes_source_axis_orientation(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        base = self._config(ROOT / "conversion-only-orientation.zarr")
        config = replace(base, operations=PipelineOperations())
        result = run_pipeline(inspection, config, progress=False)
        self.assertEqual(result["physical_stages"], ["conversion"])
        with xr.open_zarr(
            config.general.output,
            consolidated=False,
            chunks=None,
            decode_times=False,
        ) as dataset:
            np.testing.assert_allclose(dataset.lat.values, [0.275, 0.225, 0.175, 0.125])
            np.testing.assert_allclose(dataset.lon.values, [-0.075, -0.025, 0.025, 0.075])
            self.assertEqual(float(dataset["value"].isel(time=0, lat=0, lon=0)), 42.0)

    def test_executor_manifest_covers_all_identity_grid_combinations(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT / "canonical", mode="complete", engine="h5netcdf", workers=1
            )
        )
        for index, (resample, rechunk, recompress) in enumerate(
            product((False, True), repeat=3)
        ):
            with self.subTest(
                resample=resample,
                rechunk=rechunk,
                recompress=recompress,
            ):
                base = self._config(ROOT / f"combination-{index}.zarr")
                config = replace(
                    base,
                    operations=PipelineOperations(
                        resample=resample,
                        rechunk=rechunk,
                        recompress=recompress,
                    ),
                    validate=False,
                )
                result = run_pipeline(inspection, config, progress=False)
                manifest = json.loads(
                    Path(result["manifest"]).read_text(encoding="utf-8")
                )
                self.assertEqual(manifest["schema_version"], 6)
                self.assertTrue(Path(manifest["events"]).is_file())
                event_types = {
                    json.loads(line)["event"]
                    for line in Path(manifest["events"]).read_text(encoding="utf-8").splitlines()
                }
                self.assertTrue({"started", "resource", "finished"}.issubset(event_types))
                self.assertTrue(manifest["preflight"]["temporary"]["writable"])
                self.assertIn("worker_ceiling", manifest["resource_budget"])
                self.assertEqual(manifest["input_kind"], "raw")
                self.assertEqual(
                    manifest["requested_operation_order"],
                    [
                        name
                        for name, enabled in (
                            ("conversion", True),
                            ("resampling", resample),
                            ("rechunking", rechunk),
                            ("recompression", recompress),
                        )
                        if enabled
                    ],
                )
                self.assertEqual(manifest["physical_stages"], ["conversion"])
                self.assertIn("conversion", manifest["candidate_trials"])
                self.assertEqual(
                    manifest["requested_operations"],
                    {
                        "conversion": True,
                        "resampling": resample,
                        "rechunking": rechunk,
                        "recompression": recompress,
                    },
                )
                expected_resampling = "satisfied_as_noop" if resample else "not_requested"
                self.assertEqual(
                    manifest["operation_decisions"]["resampling"]["disposition"],
                    expected_resampling,
                )
                self.assertEqual(result["logical_io"]["write_amplification"], 1.0)
                self.assertTrue(config.general.output.exists())

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
        self.assertTrue(plan.direct_finalization)
        result = run_pipeline(inspection, self._config(output), progress=False)
        self.assertTrue(result["needs_resample"])
        self.assertGreater(result["resampling"]["logical_bytes"], 0)
        self.assertGreater(result["mathematical_validation"]["comparisons"], 0)
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
        self.assertEqual(
            manifest["stages"]["finalization"]["status"],
            "not_required_direct_layout",
        )
        self.assertFalse((Path(manifest["temporary_root"]) / "resampled.zarr").exists())
        with xr.open_zarr(converted, consolidated=False, chunks=None, decode_times=False) as dataset:
            self.assertEqual(dataset["value"].encoding["chunks"], (1, 6, 6))
        group = zarr.open_group(output, mode="r")
        value_layout = plan.output_layout.for_source("value")
        self.assertEqual(group["value"].chunks, value_layout.chunks)
        self.assertIn("shuffle", repr(group["value"].compressors).lower())

    def test_raw_pipeline_replacements_pass_mathematical_validation(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        base = self._config(ROOT / "replacement-validated.zarr")
        config = replace(
            base,
            operations=PipelineOperations(resample=True),
            resampling=replace(
                base.resampling,
                before_conditions="<10",
                before_results="10",
                after_conditions=">40",
                after_results="40",
                statistics_policy="exact",
            ),
        )

        result = run_pipeline(inspection, config, progress=False)
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        validation = manifest["stages"]["resampling"]["mathematical_validation"]
        self.assertGreater(validation["comparisons"], 0)
        with xr.open_zarr(
            config.general.output,
            consolidated=False,
            chunks=None,
            decode_times=False,
        ) as dataset:
            self.assertGreaterEqual(float(np.nanmin(dataset["value"].values)), 10.0)
            self.assertLessEqual(float(np.nanmax(dataset["value"].values)), 40.0)

    def test_pipeline_cli_dry_run_accepts_replacement_flags(self) -> None:
        output = ROOT / "cli-replacements.zarr"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = pipeline_main(
                [
                    "--input",
                    str(ROOT / "canonical"),
                    "--output",
                    str(output),
                    "--mode",
                    "complete",
                    "--engine",
                    "h5netcdf",
                    "--inspect-workers",
                    "1",
                    "--time",
                    "2001-01-01",
                    "2001-01-09",
                    "--lat",
                    "0.1",
                    "0.3",
                    "--lon",
                    "-0.1",
                    "0.1",
                    "--resample",
                    "--resolution",
                    "0.1",
                    "--before-conditions",
                    "<5,>20",
                    "--before-results",
                    "5,20",
                    "--after-conditions",
                    ">median",
                    "--after-results",
                    "median",
                    "--statistics-policy",
                    "exact",
                    "--no-tune",
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("resampling: executed_as_stage", stdout.getvalue())
        self.assertIn("采样前替换：[('<5', '5'), ('>20', '20')]", stdout.getvalue())
        self.assertIn("采样后替换：[('>median', 'median')]", stdout.getvalue())
        self.assertIn("替换统计策略：exact", stdout.getvalue())
        self.assertFalse(output.exists())

    def test_pipeline_cli_dry_run_accepts_explicit_codec_and_level(self) -> None:
        output = ROOT / "cli-compression.zarr"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            result = pipeline_main(
                [
                    "--input",
                    str(ROOT / "canonical"),
                    "--output",
                    str(output),
                    "--mode",
                    "complete",
                    "--engine",
                    "h5netcdf",
                    "--inspect-workers",
                    "1",
                    "--time",
                    "2001-01-01",
                    "2001-01-09",
                    "--lat",
                    "0.1",
                    "0.3",
                    "--lon",
                    "-0.1",
                    "0.1",
                    "--recompress",
                    "--compression-codec",
                    "gzip",
                    "--compression-level",
                    "3",
                    "--no-tune",
                    "--dry-run",
                ]
            )
        self.assertEqual(result, 0)
        self.assertIn("最终压缩：gzip level 3；shuffle=auto", stdout.getvalue())
        self.assertFalse(output.exists())

    def test_resampling_without_storage_operations_uses_baseline_layout(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        base = self._config(ROOT / "resample-baseline-layout.zarr")
        config = replace(
            base,
            operations=PipelineOperations(resample=True),
        )
        plan = preview_pipeline(inspection, config)
        result = run_pipeline(inspection, config, progress=False)
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(manifest["physical_stages"], ["conversion", "resampling"])
        self.assertEqual(
            manifest["operation_decisions"]["rechunking"]["disposition"],
            "not_requested",
        )
        self.assertEqual(
            manifest["operation_decisions"]["recompression"]["disposition"],
            "not_requested",
        )
        group = zarr.open_group(config.general.output, mode="r")
        self.assertEqual(group["value"].chunks, plan.output_layout.for_source("value").chunks)
        self.assertIn("zstd", repr(group["value"].compressors).lower())

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
            ),
            conversion=config.conversion,
            operations=config.operations,
            resampling=config.resampling,
            chunking=config.chunking,
            compression=config.compression,
        )
        plan = preview_pipeline(inspection, config)
        self.assertFalse(plan.needs_resample)
        self.assertTrue(plan.direct_finalization)
        self.assertEqual(plan.conversion_chunks, plan.final_chunks)
        result = run_pipeline(inspection, config, progress=False)
        self.assertFalse(result["needs_resample"])
        with xr.open_zarr(config.general.output, consolidated=False, chunks=None, decode_times=False) as dataset:
            self.assertEqual(dataset["value"].dims, ("time", "lat", "lon"))
            np.testing.assert_allclose(dataset.lat.values, [0.25, 0.15])
            np.testing.assert_allclose(dataset.lon.values, [-0.05, 0.05])
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        self.assertEqual(
            manifest["stages"]["finalization"]["status"],
            "not_required_direct_layout",
        )
        self.assertEqual(manifest["semantic_validation"]["status"], "passed")
        self.assertFalse((Path(manifest["temporary_root"]) / "source-crop.zarr").exists())
        group = zarr.open_group(config.general.output, mode="r")
        value_layout = plan.output_layout.for_source("value")
        self.assertEqual(group["value"].chunks, value_layout.chunks)
        self.assertIn("shuffle", repr(group["value"].compressors).lower())
        self.assertIn("bitshuffle", repr(group["quality"].compressors).lower())
        self.assertIn("zstd", repr(group["lat"].compressors).lower())
        self.assertEqual(result["logical_io"]["temporary_write_bytes"], 0)
        self.assertEqual(result["logical_io"]["write_amplification"], 1.0)
        self.assertGreater(result["logical_io"]["avoided_finalization_read_bytes"], 0)

    def test_float32_quantized_regular_grid_is_resampling_noop(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT / "canonical", mode="complete", engine="h5netcdf", workers=1
            )
        )
        inspection.source_inventory.lat_values = np.asarray(
            [60.25, 60.15, 60.05], dtype="float32"
        )
        # Cross zero to cover origin/step quantization where the local ULP is
        # much smaller than the accumulated float32 coordinate error.
        inspection.source_inventory.lon_values = np.asarray(
            [-0.15, -0.05, 0.05, 0.15], dtype="float32"
        )
        base = self._config(ROOT / "float32-grid-noop.zarr")
        config = replace(
            base,
            general=replace(
                base.general,
                lat_min=60.0,
                lat_max=60.3,
                lon_min=-0.2,
                lon_max=0.2,
            ),
            operations=PipelineOperations(resample=True),
        )

        plan = preview_pipeline(inspection, config)

        self.assertFalse(plan.needs_resample)
        self.assertIsNone(plan.coverage_warning)
        self.assertEqual(
            plan.decision("resampling").disposition,
            "satisfied_as_noop",
        )
        self.assertEqual(
            np.dtype(plan.output_layout.for_output("lat").dtype),
            np.dtype("float32"),
        )

    def test_real_and_half_pixel_grid_offsets_still_resample(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT / "canonical", mode="complete", engine="h5netcdf", workers=1
            )
        )
        inspection.source_inventory.lat_values = np.asarray(
            [60.25, 60.15, 60.05], dtype="float32"
        )
        inspection.source_inventory.lon_values = np.asarray(
            [120.05, 120.15, 120.25, 120.35], dtype="float32"
        )
        base = self._config(ROOT / "shifted-grid.zarr")

        for offset in (0.01, 0.05):
            with self.subTest(offset=offset):
                config = replace(
                    base,
                    general=replace(
                        base.general,
                        lat_min=60.0 + offset,
                        lat_max=60.3 + offset,
                        lon_min=120.0,
                        lon_max=120.4,
                    ),
                    operations=PipelineOperations(resample=True),
                )
                plan = preview_pipeline(inspection, config)
                self.assertTrue(plan.needs_resample)
                self.assertEqual(
                    plan.decision("resampling").disposition,
                    "executed_as_stage",
                )

    def test_smaller_conversion_tasks_force_storage_finalization(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT / "canonical", mode="complete", engine="h5netcdf", workers=1
            )
        )
        base = self._config(ROOT / "unsafe-direct-layout.zarr")
        config = replace(
            base,
            operations=PipelineOperations(rechunk=True),
            chunking=PipelineChunkingOptions(
                strategy="custom",
                custom_chunks=(2, 2, 2),
                workers=2,
            ),
        )
        conversion_plan = ConversionPlan(
            strategy="chunk",
            workers=2,
            chunk_time=1,
            chunk_lat=1,
            chunk_lon=1,
        )

        with patch(
            "fast_nc_zarr.pipeline.planner.resolve_conversion_plan",
            return_value=conversion_plan,
        ):
            plan = preview_pipeline(inspection, config)

        self.assertEqual(plan.conversion_chunks, (1, 1, 1))
        self.assertEqual(plan.final_chunks, (2, 2, 2))
        self.assertTrue(plan.finalization_required)
        self.assertFalse(plan.direct_finalization)
        self.assertEqual(plan.decision("rechunking").disposition, "executed_as_stage")
        self.assertIn("每个物理 chunk 仅由一个并行任务写入", plan.decision("rechunking").reason)

    def test_aligned_conversion_tasks_still_fuse_storage_layout(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(
                ROOT / "canonical", mode="complete", engine="h5netcdf", workers=1
            )
        )
        base = self._config(ROOT / "safe-direct-layout.zarr")
        config = replace(
            base,
            operations=PipelineOperations(rechunk=True),
            chunking=PipelineChunkingOptions(
                strategy="custom",
                custom_chunks=(1, 1, 1),
                workers=2,
            ),
        )
        conversion_plan = ConversionPlan(
            strategy="chunk",
            workers=2,
            chunk_time=2,
            chunk_lat=2,
            chunk_lon=2,
        )

        with patch(
            "fast_nc_zarr.pipeline.planner.resolve_conversion_plan",
            return_value=conversion_plan,
        ):
            plan = preview_pipeline(inspection, config)

        self.assertEqual(plan.conversion_chunks, (2, 2, 2))
        self.assertEqual(plan.final_chunks, (1, 1, 1))
        self.assertFalse(plan.finalization_required)
        self.assertTrue(plan.direct_finalization)
        self.assertEqual(
            plan.decision("rechunking").disposition,
            "fused_into_conversion",
        )

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

    def test_resample_math_failure_does_not_replace_existing_output(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        output = ROOT / "protected-result.zarr"
        sentinel = xr.Dataset({"sentinel": (("x",), np.asarray([7], dtype="int16"))})
        sentinel.to_zarr(output, mode="w", zarr_format=3, consolidated=False)
        sentinel.close()
        base = self._config(output)
        config = replace(base, general=replace(base.general, overwrite=True))

        with patch(
            "fast_nc_zarr.pipeline.engine.validate_resample_samples",
            side_effect=RuntimeError("simulated math validation failure"),
        ):
            with self.assertRaisesRegex(Exception, "simulated math validation failure"):
                run_pipeline(inspection, config, progress=False)

        with xr.open_zarr(output, consolidated=False, chunks=None) as preserved:
            self.assertEqual(int(preserved["sentinel"].values[0]), 7)

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

    def test_default_variables_exclude_bounds_and_crs_metadata(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        inventory = inspection.source_inventory
        inventory.variables["time_bnds"] = VariableSpec(
            "time_bnds", ("time", "nv"), "datetime64[ns]", (2,), None
        )
        inventory.variables["crs"] = VariableSpec("crs", (), "|S1", (), None)
        plan = build_pipeline_plan(inspection, self._config(ROOT / "metadata-plan.zarr"))
        self.assertEqual(plan.source_selection.variables, ("value",))

        config = replace(
            self._config(ROOT / "metadata-explicit.zarr"),
            conversion=PipelineConversionOptions(
                variables=("value", "time_bnds"), auto_tune=False, max_workers=1
            ),
        )
        with self.assertRaisesRegex(ValueError, "辅助坐标、边界或 CRS"):
            build_pipeline_plan(inspection, config)

    def test_esmf_preflight_fails_before_creating_intermediate_store(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        temporary = ROOT / "preflight-temporary"
        config = replace(
            self._config(ROOT / "preflight-output.zarr"),
            general=replace(self._config(ROOT / "unused.zarr").general, temporary_dir=temporary),
        )
        with patch(
            "fast_nc_zarr.pipeline.engine.validate_resampling_environment",
            side_effect=ResamplingEnvironmentError("ESMF 配置错误"),
        ):
            with self.assertRaisesRegex(ResamplingEnvironmentError, "ESMF 配置错误"):
                run_pipeline(inspection, config, progress=False)
        self.assertFalse(temporary.exists())

    def test_capacity_preflight_fails_before_conversion(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        temporary = ROOT / "capacity-temporary"
        base = self._config(ROOT / "capacity-output.zarr")
        config = replace(base, general=replace(base.general, temporary_dir=temporary))
        usage = namedtuple("usage", "total used free")(1024**3, 0, 1024)
        with (
            patch("fast_nc_zarr.pipeline.engine.validate_resampling_environment"),
            patch("fast_nc_zarr.pipeline.engine.shutil.disk_usage", return_value=usage),
            patch("fast_nc_zarr.pipeline.engine.run_conversion") as conversion,
        ):
            with self.assertRaisesRegex(PipelineExecutionError, "预计临时与最终产物"):
                run_pipeline(inspection, config, progress=False)
        conversion.assert_not_called()
        self.assertFalse(temporary.exists())

    def test_resampling_failure_records_stage_and_retains_conversion(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        temporary = ROOT / "failed-stage-temporary"
        base = self._config(ROOT / "failed-stage-output.zarr")
        config = replace(base, general=replace(base.general, temporary_dir=temporary))
        with (
            patch("fast_nc_zarr.pipeline.engine.validate_resampling_environment"),
            patch(
                "fast_nc_zarr.pipeline.engine.run_resample",
                side_effect=RuntimeError("worker failed"),
            ),
        ):
            with self.assertRaisesRegex(PipelineExecutionError, "worker failed"):
                run_pipeline(inspection, config, progress=False)
        roots = list((temporary / "fast-nc-zarr-pipeline").iterdir())
        self.assertEqual(len(roots), 1)
        manifest = json.loads((roots[0] / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["failed_stage"], "resampling")
        self.assertEqual(
            manifest["resume"]["checkpoints"]["conversion"]["path"],
            "source-crop.zarr",
        )
        self.assertTrue((roots[0] / "source-crop.zarr" / "zarr.json").is_file())
        self.assertFalse(config.general.output.exists())

    def test_failed_conversion_checkpoint_resumes_remaining_pipeline(self) -> None:
        inspection = inspect_source(
            SourceInspectionConfig(ROOT, mode="complete", engine="h5netcdf", workers=1)
        )
        temporary = ROOT / "resume-temporary"
        base = self._config(ROOT / "resume-output.zarr")
        config = replace(
            base,
            general=replace(
                base.general,
                temporary_dir=temporary,
                cleanup_intermediate=True,
            ),
        )
        with (
            patch("fast_nc_zarr.pipeline.engine.validate_resampling_environment"),
            patch(
                "fast_nc_zarr.pipeline.engine.run_resample",
                side_effect=RuntimeError("simulated worker failure"),
            ),
        ):
            with self.assertRaisesRegex(PipelineExecutionError, "simulated worker failure"):
                run_pipeline(inspection, config, progress=False)
        original_job = next((temporary / "fast-nc-zarr-pipeline").iterdir())
        legacy_manifest_path = original_job / "manifest.json"
        legacy_payload = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        legacy_payload["source"] = "/old-machine/source"
        legacy_payload["temporary_root"] = "/old-machine/temp/job"
        legacy_payload["config"]["general"]["temporary_dir"] = "/old-machine/temp"
        legacy_manifest_path.write_text(
            json.dumps(legacy_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        recovered = inspect_temporary_pipeline(temporary)
        self.assertEqual(recovered.kind, "temporary")
        self.assertTrue(recovered.path.is_relative_to(temporary))
        self.assertEqual(recovered.recovery.config.general.temporary_dir, temporary)
        self.assertEqual(recovered.zarr_info.shape, (2, 6, 6))
        self.assertTrue(recovered.recovery.config.operations.resample)
        self.assertTrue(recovered.recovery.config.operations.rechunk)
        self.assertTrue(recovered.recovery.config.operations.recompress)
        result = run_pipeline_service(
            recovered,
            recovered.recovery.config,
            progress=False,
        )
        self.assertTrue((Path(result["output"]) / "zarr.json").is_file())
        self.assertGreater(result["mathematical_validation"]["comparisons"], 0)
        original_manifest = json.loads(
            recovered.recovery.manifest_path.read_text(encoding="utf-8")
        )
        self.assertEqual(original_manifest["status"], "resumed_succeeded")
        self.assertEqual(original_manifest["schema_version"], 6)
        self.assertNotIn("error", original_manifest)
        self.assertNotIn("failed_stage", original_manifest)
        self.assertEqual(
            original_manifest["error_history"][-1]["message"],
            "simulated worker failure",
        )
        self.assertEqual(
            original_manifest["error_history"][-1]["resolved_by_job"],
            Path(result["manifest"]).parent.name,
        )
        self.assertTrue(original_manifest["retained_checkpoints_cleaned"])
        self.assertFalse(recovered.recovery.checkpoint.exists())


if __name__ == "__main__":
    unittest.main()
