from __future__ import annotations

import shutil
import sys
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.rechunking.compression import make_compression_plan  # noqa: E402
from fast_nc_zarr.rechunking.engine import (  # noqa: E402
    _intermediate_shards,
    run_rechunk,
)
from fast_nc_zarr.rechunking.inspection import (  # noqa: E402
    RechunkInspectionError,
    inspect_store,
)
from fast_nc_zarr.rechunking.planning import plan_chunks  # noqa: E402


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
