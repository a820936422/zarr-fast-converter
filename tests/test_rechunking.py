from __future__ import annotations

import shutil
import sys
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.models import StorageProfile  # noqa: E402
from fast_nc_zarr.rechunking.compression import make_compression_plan  # noqa: E402
from fast_nc_zarr.rechunking.engine import (  # noqa: E402
    RechunkExecutionError,
    _intermediate_shards,
    _parallel_workers,
    _stage2_benchmark_regions,
    _stage2_benchmark_tasks,
    _stage2_safe_workers,
    run_rechunk,
)
from fast_nc_zarr.rechunking.inspection import (  # noqa: E402
    RechunkInspectionError,
    inspect_store,
)
from fast_nc_zarr.rechunking.planning import plan_chunks  # noqa: E402
from fast_nc_zarr.pipeline.models import PipelineChunkingOptions  # noqa: E402
from fast_nc_zarr.rechunking.autotune import (  # noqa: E402
    WorkerTrial,
    benchmark_worker_candidates,
    select_worker_trial,
    worker_candidates,
)



ROOT = Path("/tmp/codex_test/fast_nc_zarr_rechunk_tests")


class RechunkingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(ROOT, ignore_errors=True)
        ROOT.mkdir(parents=True)
        dataset = xr.Dataset(
            {
                "float_value": (
                    ("time", "lat", "lon"),
                    np.arange(6 * 8 * 10, dtype="float32").reshape(6, 8, 10),
                    {"units": "1"},
                ),
                "integer_value": (
                    ("time", "lat", "lon"),
                    np.arange(6 * 8 * 10, dtype="int16").reshape(6, 8, 10),
                    {"flag": "quality"},
                ),
            },
            coords={
                "time": np.arange(6, dtype="int64"),
                "lat": np.linspace(40, -40, 8, dtype="float32"),
                "lon": np.linspace(-180, 180, 10, dtype="float32"),
            },
            attrs={"title": "rechunk test"},
        )
        dataset.to_zarr(
            ROOT / "input.zarr",
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={
                "float_value": {"chunks": (2, 4, 5)},
                "integer_value": {"chunks": (2, 4, 5)},
                "time": {"chunks": (2,)},
                "lat": {"chunks": (4,)},
                "lon": {"chunks": (5,)},
            },
        )
        dataset.close()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_inspection_and_strategy_plans(self) -> None:
        info = inspect_store(ROOT / "input.zarr")
        self.assertEqual(info.shape, (6, 8, 10))
        self.assertEqual(len(info.data_variables), 2)
        self.assertEqual(plan_chunks(info, "time", target_mib=32).chunks, (6, 8, 10))
        self.assertEqual(plan_chunks(info, "space", target_mib=32).chunks, (6, 8, 10))
        self.assertEqual(
            plan_chunks(info, "custom", custom_chunks=(2, 4, 5)).chunks,
            (2, 4, 5),
        )

    def test_intermediate_shards_preserve_logical_chunks(self) -> None:
        info = inspect_store(ROOT / "input.zarr")
        shards = _intermediate_shards(info, (2, 4, 5))
        self.assertEqual(shards["float_value"], (2, 8, 10))
        self.assertEqual(shards["integer_value"], (2, 8, 10))

    def test_equivalent_store_reports_progress_completion(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "equivalent-progress.zarr"
        info = inspect_store(source)
        plan = plan_chunks(info, "custom", custom_chunks=(2, 4, 5), target_mib=32)
        records: list[tuple[int, int, int | None, str | None]] = []

        metrics = run_rechunk(
            source,
            output,
            info,
            plan,
            make_compression_plan("none"),
            workers=1,
            progress=False,
            progress_callback=lambda completed, total, logical_bytes, message: records.append(
                (completed, total, logical_bytes, message)
            ),
        )

        self.assertEqual(metrics["execution_path"], "copy")
        self.assertEqual(records, [(1, 1, None, "等价复制完成")])

    def test_equivalent_store_uses_independent_chunk_copy(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "equivalent-copy.zarr"
        info = inspect_store(source)
        plan = plan_chunks(
            info,
            "custom",
            custom_chunks=(2, 4, 5),
            target_mib=32,
        )

        metrics = run_rechunk(
            source,
            output,
            info,
            plan,
            make_compression_plan("none"),
            workers=4,
            progress=False,
        )

        self.assertEqual(metrics["execution_path"], "copy")
        self.assertEqual(metrics["avoided_intermediate_bytes"], info.logical_bytes)
        source_metadata = {
            path.relative_to(source): path.read_bytes()
            for path in source.rglob("zarr.json")
        }
        output_metadata = {
            path.relative_to(output): path.read_bytes()
            for path in output.rglob("zarr.json")
        }
        self.assertEqual(output_metadata, source_metadata)
        source_chunks = [
            path
            for path in (source / "float_value").rglob("*")
            if path.is_file() and path.name != "zarr.json"
        ]
        self.assertTrue(source_chunks)
        for source_chunk in source_chunks:
            output_chunk = output / source_chunk.relative_to(source)
            self.assertTrue(output_chunk.is_file())
            self.assertNotEqual(
                (source_chunk.stat().st_dev, source_chunk.stat().st_ino),
                (output_chunk.stat().st_dev, output_chunk.stat().st_ino),
            )
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            np.testing.assert_array_equal(
                result["float_value"].values,
                np.arange(6 * 8 * 10, dtype="float32").reshape(6, 8, 10),
            )
            self.assertEqual(result.attrs["title"], "rechunk test")

    def test_equivalent_copy_rejects_source_symlinks(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "symlink-copy.zarr"
        info = inspect_store(source)
        plan = plan_chunks(
            info,
            "custom",
            custom_chunks=(2, 4, 5),
            target_mib=32,
        )
        link = source / "external-link"
        link.symlink_to(source / "zarr.json")
        self.addCleanup(link.unlink, missing_ok=True)

        with self.assertRaisesRegex(RechunkExecutionError, "符号链接"):
            run_rechunk(
                source,
                output,
                info,
                plan,
                make_compression_plan("none"),
                workers=1,
                progress=False,
            )

        self.assertFalse(output.exists())

    def test_codec_change_uses_single_stage_physical_chunks(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "codec-only.zarr"
        info = inspect_store(source)
        plan = plan_chunks(
            info,
            "custom",
            custom_chunks=(2, 4, 5),
            target_mib=32,
        )

        metrics = run_rechunk(
            source,
            output,
            info,
            plan,
            make_compression_plan("fast"),
            workers=2,
            progress=False,
        )

        self.assertEqual(metrics["execution_path"], "single_stage")
        self.assertEqual(metrics["avoided_intermediate_bytes"], info.logical_bytes)
        result_info = inspect_store(output)
        for variable in result_info.variables:
            self.assertEqual(variable.chunks, plan.chunks_for(variable))
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            codec = result["float_value"].encoding["compressors"][0]
            self.assertEqual(codec.cname, "zstd")
            self.assertEqual(codec.clevel, 1)
            np.testing.assert_array_equal(
                result["integer_value"].values,
                np.arange(6 * 8 * 10, dtype="int16").reshape(6, 8, 10),
            )
            self.assertEqual(result.attrs["title"], "rechunk test")

    def test_auto_compression_benchmarks_real_chunks_and_publishes_selection(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "auto-compression.zarr"
        info = inspect_store(source)
        plan = plan_chunks(info, "custom", custom_chunks=(2, 4, 5))

        metrics = run_rechunk(
            source,
            output,
            info,
            plan,
            make_compression_plan("auto"),
            workers=1,
            progress=False,
            compression_objective="balanced",
            compression_tune_budget_seconds=5,
        )

        self.assertIsNotNone(metrics["compression_tuning"])
        self.assertNotEqual(metrics["selected_compression"]["profile"], "auto")
        self.assertTrue(
            all(item["sample_count"] >= 2 for item in metrics["compression_tuning"]["results"] if item["success"])
        )
        self.assertTrue(output.is_dir())
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            np.testing.assert_array_equal(
                result["integer_value"].values,
                np.arange(6 * 8 * 10, dtype="int16").reshape(6, 8, 10),
            )

    def test_auto_compression_aborts_when_no_candidate_is_safe(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "unsafe-auto-compression.zarr"
        info = inspect_store(source)
        plan = plan_chunks(info, "custom", custom_chunks=(2, 4, 5))
        failed_report = SimpleNamespace(selected=None, cancelled=False)

        with patch(
            "fast_nc_zarr.rechunking.engine.benchmark_compression_candidates",
            return_value=failed_report,
        ):
            with self.assertRaisesRegex(RechunkExecutionError, "没有通过无损验证"):
                run_rechunk(
                    source,
                    output,
                    info,
                    plan,
                    make_compression_plan("auto"),
                    workers=1,
                    progress=False,
                )
        self.assertFalse(output.exists())

    def test_failed_equivalent_copy_preserves_existing_target(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "protected-target.zarr"
        shutil.copytree(source, output)
        before = (output / "zarr.json").read_bytes()
        info = inspect_store(source)
        plan = plan_chunks(info, "custom", custom_chunks=(2, 4, 5))

        with patch(
            "fast_nc_zarr.rechunking.engine._copy_equivalent_store",
            side_effect=OSError("simulated copy failure"),
        ):
            with self.assertRaises(RechunkExecutionError):
                run_rechunk(
                    source,
                    output,
                    info,
                    plan,
                    make_compression_plan("none"),
                    overwrite=True,
                    workers=1,
                    progress=False,
                )

        self.assertEqual((output / "zarr.json").read_bytes(), before)
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            np.testing.assert_array_equal(
                result["integer_value"].values,
                np.arange(6 * 8 * 10, dtype="int16").reshape(6, 8, 10),
            )

    def test_worker_ceiling_uses_resources_not_filesystem_type(self) -> None:
        network_source = StorageProfile(Path("/source"), "network-a", False, "9p")
        network_target = StorageProfile(Path("/target"), "network-b", False, "ext4")
        with (
            patch(
                "fast_nc_zarr.rechunking.engine.storage_profile",
                side_effect=[network_source, network_target],
            ),
            patch("fast_nc_zarr.rechunking.engine.os.cpu_count", return_value=16),
        ):
            network_workers, reason = _parallel_workers(
                Path("/source"), Path("/target"), 8, worker_ceiling=8
            )
        self.assertEqual(network_workers, 8)
        self.assertIn("9p", reason)
        self.assertIn("未静态限制", reason)

        ssd_source = StorageProfile(Path("/source"), "ssd-a", False, "ext4")
        ssd_target = StorageProfile(Path("/target"), "ssd-b", False, "ext4")
        with (
            patch(
                "fast_nc_zarr.rechunking.engine.storage_profile",
                side_effect=[ssd_source, ssd_target],
            ),
            patch("fast_nc_zarr.rechunking.engine.os.cpu_count", return_value=16),
        ):
            ssd_workers, _ = _parallel_workers(
                Path("/source"), Path("/target"), 8, worker_ceiling=8
            )
        self.assertEqual(ssd_workers, 8)

        capped, _ = _parallel_workers(
            Path("/source"),
            Path("/target"),
            8,
            source_profile=network_source,
            target_profile=network_target,
            worker_ceiling=3,
        )
        self.assertEqual(capped, 3)

    def test_stage2_benchmark_uses_final_time_chunk_geometry(self) -> None:
        info = inspect_store(ROOT / "input.zarr")
        plan = plan_chunks(info, "custom", custom_chunks=(2, 4, 5))
        regions = {
            variable.name: tuple(1 if dim == "time" else chunk for dim, chunk in zip(variable.dims, variable.chunks))
            for variable in info.data_variables
        }
        samples = _stage2_benchmark_regions(info, plan, regions)
        tasks = _stage2_benchmark_tasks(info, plan, samples, 8)

        for variable in info.data_variables:
            time_axis = variable.dims.index("time")
            self.assertEqual(samples[variable.name][time_axis], 2)
        for variable_name, starts, _stops in tasks:
            variable = next(item for item in info.data_variables if item.name == variable_name)
            for dim, start in zip(variable.dims, starts):
                self.assertEqual(start % plan.dim_chunks[dim], 0)

        info = inspect_store(ROOT / "input.zarr")
        plan = plan_chunks(info, "custom", custom_chunks=(2, 4, 5))
        regions = {
            variable.name: variable.chunks
            for variable in info.data_variables
            if variable.ndim == 3
        }
        with patch(
            "fast_nc_zarr.rechunking.engine.psutil.virtual_memory"
        ) as virtual_memory:
            virtual_memory.return_value.available = 1
            memory_workers, peak_bytes = _stage2_safe_workers(
                info,
                plan,
                regions,
                make_compression_plan("fast"),
                8,
            )
        self.assertEqual(memory_workers, 1)
        self.assertGreater(peak_bytes, 0)

    def test_rechunk_and_type_specific_lossless_compression(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "balanced.zarr"
        info = inspect_store(source)
        plan = plan_chunks(info, "custom", custom_chunks=(3, 4, 5), target_mib=32)
        metrics = run_rechunk(
            source,
            output,
            info,
            plan,
            make_compression_plan("balanced"),
            workers=1,
            progress=False,
        )
        self.assertGreater(metrics["physical_bytes"], 0)
        self.assertEqual(metrics["execution_path"], "two_stage")
        self.assertEqual(metrics["avoided_intermediate_bytes"], 0)
        result = inspect_store(output)
        self.assertEqual(result.dimensions, info.dimensions)
        for variable in result.data_variables:
            self.assertEqual(variable.chunks, (3, 4, 5))
        with xr.open_zarr(output, consolidated=False, chunks=None) as dataset:
            np.testing.assert_array_equal(
                dataset["integer_value"].values,
                np.arange(6 * 8 * 10, dtype="int16").reshape(6, 8, 10),
            )
            self.assertEqual(dataset.attrs["title"], "rechunk test")

    def test_explicit_codec_level_and_dtype_shuffle_round_trip(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "lz4-level3.zarr"
        info = inspect_store(source)
        plan = plan_chunks(info, "custom", custom_chunks=(3, 4, 5))
        compression = make_compression_plan(
            "balanced",
            codec="blosc-lz4",
            level=3,
            shuffle="auto",
        )
        self.assertEqual(compression.profile, "custom")
        self.assertEqual(compression.codec, "blosc-lz4")
        run_rechunk(
            source,
            output,
            info,
            plan,
            compression,
            workers=1,
            progress=False,
        )
        with xr.open_zarr(output, consolidated=False, chunks=None) as dataset:
            float_codec = dataset.float_value.encoding["compressors"][0]
            integer_codec = dataset.integer_value.encoding["compressors"][0]
            self.assertEqual(float_codec.cname, "lz4")
            self.assertEqual(float_codec.clevel, 3)
            self.assertEqual(float_codec.shuffle, "shuffle")
            self.assertEqual(integer_codec.shuffle, "bitshuffle")
            np.testing.assert_array_equal(
                dataset.integer_value.values,
                np.arange(6 * 8 * 10, dtype="int16").reshape(6, 8, 10),
            )

    def test_custom_temporary_directory_is_used_for_intermediate_data(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "temporary-output.zarr"
        temporary = ROOT / "fast-temporary"
        info = inspect_store(source)
        plan = plan_chunks(info, "custom", custom_chunks=(3, 4, 5))
        metrics = run_rechunk(
            source,
            output,
            info,
            plan,
            make_compression_plan("balanced"),
            workers=1,
            progress=False,
            temporary_dir=temporary,
        )
        self.assertEqual(metrics["temporary_dir"], str(temporary.resolve()))
        self.assertTrue(output.is_dir())
        self.assertFalse(list(temporary.glob("*.tmp")))
        self.assertFalse(list(temporary.glob(".*.tmp")))

    def test_time_strategy_parallel_pipeline(self) -> None:
        source = ROOT / "input.zarr"
        output = ROOT / "parallel_time.zarr"
        info = inspect_store(source)
        plan = plan_chunks(info, "time", target_mib=32, workers=2)
        run_rechunk(
            source,
            output,
            info,
            plan,
            make_compression_plan("fast"),
            workers=2,
            progress=False,
        )
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            np.testing.assert_array_equal(
                result["float_value"].values,
                np.arange(6 * 8 * 10, dtype="float32").reshape(6, 8, 10),
            )
            np.testing.assert_array_equal(
                result["integer_value"].values,
                np.arange(6 * 8 * 10, dtype="int16").reshape(6, 8, 10),
            )

    def test_incomplete_input_is_rejected(self) -> None:
        dataset = xr.Dataset(
            {"value": (("lat", "lon"), np.zeros((2, 3), dtype="float32"))},
            coords={"lat": [1, 2], "lon": [3, 4, 5]},
        )
        path = ROOT / "two_dimensional.zarr"
        dataset.to_zarr(path, mode="w", consolidated=False, zarr_format=3)
        dataset.close()
        with self.assertRaises(RechunkInspectionError):
            inspect_store(path)

    def test_source_and_target_chunk_boundaries_may_differ(self) -> None:
        values = np.arange(6 * 8 * 10, dtype="float32").reshape(6, 8, 10)
        dataset = xr.Dataset(
            {"value": (("time", "lat", "lon"), values)},
            coords={"time": range(6), "lat": range(8), "lon": range(10)},
        )
        source = ROOT / "boundary_input.zarr"
        output = ROOT / "boundary_output.zarr"
        dataset.to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={"value": {"chunks": (2, 5, 7)}},
        )
        dataset.close()
        info = inspect_store(source)
        plan = plan_chunks(info, "custom", custom_chunks=(3, 4, 5))
        run_rechunk(
            source,
            output,
            info,
            plan,
            make_compression_plan("fast"),
            workers=2,
            progress=False,
        )
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            np.testing.assert_array_equal(result["value"].values, values)

    def test_permuted_data_variable_dimensions(self) -> None:
        """The batched source write must preserve a non-standard dim order."""

        values = np.arange(8 * 10 * 6, dtype="float32").reshape(8, 10, 6)
        dataset = xr.Dataset(
            {
                "value": (
                    ("lat", "lon", "time"),
                    values,
                )
            },
            coords={
                "time": range(6),
                "lat": range(8),
                "lon": range(10),
            },
        )
        source = ROOT / "permuted_input.zarr"
        output = ROOT / "permuted_output.zarr"
        dataset.to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={"value": {"chunks": (5, 7, 2)}},
        )
        dataset.close()
        info = inspect_store(source)
        plan = plan_chunks(info, "custom", custom_chunks=(3, 4, 2))
        run_rechunk(
            source,
            output,
            info,
            plan,
            make_compression_plan("fast"),
            workers=2,
            progress=False,
        )
        with xr.open_zarr(output, consolidated=False, chunks=None) as result:
            np.testing.assert_array_equal(result["value"].values, values)
            self.assertEqual(result["value"].dims, ("lat", "lon", "time"))

class WorkerAutotuneTests(unittest.TestCase):
    def test_pipeline_chunking_defaults_to_auto(self) -> None:
        self.assertEqual(PipelineChunkingOptions().workers, "auto")

    def test_candidates_include_every_safe_parallel_option(self) -> None:
        self.assertEqual(worker_candidates(1), (1,))
        self.assertEqual(worker_candidates(6), (1, 2, 3, 4, 5, 6))

    def test_candidates_with_initial_places_initial_first_but_keeps_full_range(self) -> None:
        self.assertEqual(
            worker_candidates(6, initial_workers=4),
            (4, 1, 2, 3, 5, 6),
        )
        self.assertEqual(
            worker_candidates(6, initial_workers=8),
            (6, 1, 2, 3, 4, 5),
        )

    def test_cpu_bound_sample_selects_two_or_more(self) -> None:
        trials = tuple(
            WorkerTrial(
                workers=workers,
                status="ok",
                elapsed_seconds=1.0,
                logical_bytes=int(rate * 1024**2),
                throughput_mib_s=rate,
            )
            for workers, rate in ((1, 10.0), (2, 22.0), (4, 23.0))
        )
        selected, _reason = select_worker_trial(trials)
        self.assertEqual(selected, 2)

    def test_failed_candidates_do_not_block_higher_parallel_options(self) -> None:
        def runner(workers: int) -> dict[str, float | int]:
            if workers != 1:
                raise OSError("simulated failure")
            return {"logical_bytes": 10 * 1024**2, "elapsed_seconds": 1.0}

        report = benchmark_worker_candidates(
            "stage2",
            (1, 2, 4),
            runner,
            safe_ceiling=4,
            storage_reason="local",
            sample_tasks=2,
            sample_logical_bytes=20 * 1024**2,
            budget_seconds=10.0,
        )
        self.assertEqual(report.selected_workers, 1)
        self.assertEqual(report.trials[1].status, "failed")
        self.assertEqual(report.trials[2].status, "failed")
        self.assertEqual(report.rejected_candidates, (2, 4))
