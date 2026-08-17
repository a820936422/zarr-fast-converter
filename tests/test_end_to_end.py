from __future__ import annotations

import shutil
import sys
import unittest
import os
from unittest.mock import patch
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.inspection import (  # noqa: E402
    DimensionMappingRequired,
    inspect_dataset,
)
from fast_nc_zarr.cli import _date_label  # noqa: E402
from fast_nc_zarr.engine import convert  # noqa: E402
from fast_nc_zarr.benchmark import representative_selection, tune  # noqa: E402
from fast_nc_zarr.models import CodecSpec, ConversionPlan, OutputLayout, Selection, VariableOutputLayout
from fast_nc_zarr.selection import make_selection, parse_list  # noqa: E402
from fast_nc_zarr.validation import validate_output  # noqa: E402
from fast_nc_zarr.writer import (  # noqa: E402
    SOURCE_CACHE_HARD_LIMIT,
    chunk_tasks,
    direct_write,
    source_cache_limit,
)


ROOT = Path("/tmp/codex_test/fast_nc_zarr_tests")


def write_fixture(
    path: Path,
    times: np.ndarray,
    offset: int,
    dimension_names: tuple[str, str, str] = ("time", "lat", "lon"),
    include_scalar: bool = False,
) -> None:
    time_dim, lat_dim, lon_dim = dimension_names
    lat = np.linspace(60, -60, 12, dtype="float32")
    lon = np.linspace(-170, 170, 16, dtype="float32")
    shape = (len(times), len(lat), len(lon))
    values = np.arange(np.prod(shape), dtype="float32").reshape(shape) + offset
    quality = (values.astype("int16") % 17).astype("int16")
    variables = {
            "temperature": ((time_dim, lat_dim, lon_dim), values, {"units": "K", "_FillValue": np.float32(-9999)}),
            "quality": ((time_dim, lat_dim, lon_dim), quality, {"_FillValue": np.int16(-99)}),
            "permuted": (
                (lon_dim, time_dim, lat_dim),
                values.transpose(2, 0, 1),
                {"_FillValue": np.float32(-9999)},
            ),
    }
    if include_scalar:
        variables["scalar"] = ((), np.float32(offset))
    ds = xr.Dataset(
        variables,
        coords={time_dim: times, lat_dim: lat, lon_dim: lon},
        attrs={"title": "converter fixture"},
    )
    encoding = {
        "temperature": {"chunksizes": (1, 4, 4), "zlib": True, "complevel": 1},
        "quality": {"chunksizes": (1, 4, 4), "zlib": True, "complevel": 1},
        "permuted": {"chunksizes": (4, 1, 4), "zlib": True, "complevel": 1},
    }
    ds.to_netcdf(path, engine="h5netcdf", encoding=encoding)


def write_cf_auxiliary_fixture(
    path: Path,
    day: int,
    *,
    bounds_dtype: str = "float64",
    bounds_units: str | None = None,
    value_units: str = "g m-2 d-1",
) -> None:
    bounds_attrs = {"long_name": "Start and End Time for Each Time Slice"}
    if bounds_units is not None:
        bounds_attrs["units"] = bounds_units
    dataset = xr.Dataset(
        {
            "value": (
                ("time", "lat", "lon"),
                np.full((1, 2, 3), day, dtype="float32"),
                {
                    "units": value_units,
                    "grid_mapping": "crs",
                    "coordinates": "time lat lon",
                },
            ),
            "time_bnds": (
                ("time", "nv"),
                np.asarray([[day, day + 1]], dtype=bounds_dtype),
                bounds_attrs,
            ),
            "crs": ((), np.float64(0), {"long_name": "Coordinate Reference System"}),
        },
        coords={
            "time": (
                "time",
                np.asarray([day + 0.5], dtype="float64"),
                {
                    "units": "days since 2000-01-01 00:00:00",
                    "bounds": "time_bnds",
                },
            ),
            "lat": np.asarray([0.25, -0.25], dtype="float32"),
            "lon": np.asarray([-0.5, 0.0, 0.5], dtype="float32"),
        },
    )
    dataset.to_netcdf(path, engine="h5netcdf")
    dataset.close()


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)
        (ROOT / "small").mkdir(parents=True)
        (ROOT / "large").mkdir(parents=True)
        (ROOT / "aliased").mkdir(parents=True)
        start = np.datetime64("2001-01-01", "D")
        for index in range(8):
            write_fixture(
                ROOT / "small" / f"day-{index:03d}.nc",
                np.asarray([start + np.timedelta64(index, "D")]),
                index * 1000,
            )
        write_fixture(ROOT / "large" / "year-a.nc", start + np.arange(6), 0)
        write_fixture(ROOT / "large" / "year-b.nc", start + np.arange(6, 12), 10000)
        write_fixture(
            ROOT / "aliased" / "year.nc",
            start + np.arange(4),
            20000,
            ("time", "latitude", "longitude"),
            include_scalar=True,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_parse_unquoted_iso_date_range(self) -> None:
        self.assertEqual(
            parse_list("[2001-01-01,2022-12-31]", "时间范围"),
            ["2001-01-01", "2022-12-31"],
        )

    def test_date_label_uses_day_precision(self) -> None:
        value = np.datetime64("2001-01-01T12:34:56", "ns")
        self.assertEqual(_date_label(value), "2001-01-01")

    def test_file_strategy_with_selection(self) -> None:
        inventory = inspect_dataset(ROOT / "small", workers=2, progress=False)
        selection = make_selection(
            inventory,
            time_bounds=["2001-01-02", "2001-01-07"],
            lat_bounds=[-30, 60],
            lon_bounds=[-100, 170],
            variables=["temperature"],
        )
        output = ROOT / "small-output.zarr"
        plan = ConversionPlan("file", 2, 1, 4, 8, task_batch=2)
        metrics = direct_write(inventory, selection, output, plan, progress=False)
        self.assertGreater(metrics["logical_bytes"], 0)
        validate_output(inventory, selection, output)
        with xr.open_zarr(output, chunks=None, mask_and_scale=False, consolidated=False) as actual:
            self.assertEqual(tuple(actual.temperature.shape), selection.shape)
            self.assertEqual(actual.temperature.attrs["_FillValue"], -9999)

    def test_chunk_strategy_crosses_file_boundary(self) -> None:
        inventory = inspect_dataset(ROOT / "large", workers=2, progress=False)
        selection = make_selection(
            inventory, variables=["temperature", "quality", "permuted"]
        )
        output = ROOT / "large-output.zarr"
        plan = ConversionPlan("chunk", 2, 4, 6, 8, task_batch=4)
        metrics = direct_write(inventory, selection, output, plan, progress=False)
        planned = list(chunk_tasks(inventory, selection, plan))
        owners = {
            (task.output_variable or task.variable, task.output_ranges)
            for task in planned
        }
        self.assertEqual(len(owners), len(planned))
        self.assertEqual(metrics["chunks_written"], len(planned))
        self.assertEqual(metrics["planned_chunks"], len(planned))
        self.assertLess(metrics["task_batches"], metrics["planned_chunks"])
        self.assertEqual(metrics["workers"], 2)
        self.assertGreater(metrics["source_opens"], 0)
        self.assertGreater(metrics["source_cache_hits"], 0)
        self.assertLessEqual(metrics["source_cache_limit"], SOURCE_CACHE_HARD_LIMIT)
        self.assertEqual(
            source_cache_limit(10_000, 8, open_file_limit=64),
            4,
        )
        validate_output(inventory, selection, output, points=5)
        with xr.open_zarr(
            output, chunks=None, mask_and_scale=False, consolidated=False
        ) as actual:
            self.assertEqual(actual.sizes["time"], 12)
            self.assertEqual(
                set(actual.data_vars), {"temperature", "quality", "permuted"}
            )

    def test_fixed_chunks_autotune_preserves_complete_layout(self) -> None:
        inventory = inspect_dataset(ROOT / "large", workers=1, progress=False)
        selection = make_selection(inventory, variables=["temperature"])
        shape = selection.shape
        forced_chunks = (3, 6, 8)
        codec = CodecSpec("blosc", level=1, cname="zstd", shuffle="shuffle")
        layout = OutputLayout(
            variables=(
                VariableOutputLayout(
                    "temperature",
                    "temperature",
                    ("time", "lat", "lon"),
                    shape,
                    "float32",
                    forced_chunks,
                    codec=codec,
                ),
                VariableOutputLayout(
                    "time",
                    "time",
                    ("time",),
                    (shape[0],),
                    str(inventory.times.dtype),
                    (shape[0],),
                    codec=codec,
                    is_coord=True,
                ),
                VariableOutputLayout(
                    "lat",
                    "lat",
                    ("lat",),
                    (shape[1],),
                    str(inventory.lat_values.dtype),
                    (shape[1],),
                    codec=codec,
                    is_coord=True,
                ),
                VariableOutputLayout(
                    "lon",
                    "lon",
                    ("lon",),
                    (shape[2],),
                    str(inventory.lon_values.dtype),
                    (shape[2],),
                    codec=codec,
                    is_coord=True,
                ),
            )
        )
        observed = {}

        def capture_tune(*args, **kwargs):
            observed["candidates"] = args[3]
            observed["kwargs"] = kwargs
            return tune(*args, **kwargs)

        tuned_output = ROOT / "fixed-layout-tuned.zarr"
        with patch("fast_nc_zarr.engine.tune", side_effect=capture_tune):
            chosen, metrics = convert(
                inventory,
                selection,
                tuned_output,
                chunks=forced_chunks,
                output_layout=layout,
                auto_tune=True,
                tune_budget=5,
                max_workers=3,
                progress=False,
            )

        candidates = observed["candidates"]
        self.assertGreater(len({item.workers for item in candidates}), 2)
        self.assertTrue(all(item.workers <= 3 for item in candidates))
        invariant_layouts = {
            (
                item.strategy,
                item.chunks,
                item.compression,
                item.compression_level,
                item.shuffle,
                item.rationale,
            )
            for item in candidates
        }
        self.assertEqual(len(invariant_layouts), 1)
        self.assertEqual(observed["kwargs"]["writer_kwargs"]["output_layout"], layout)
        self.assertTrue(observed["kwargs"]["fixed_layout"])
        self.assertEqual(chosen.chunks, forced_chunks)
        self.assertEqual(metrics["chunks_written"], metrics["planned_chunks"])

        reference_output = ROOT / "fixed-layout-reference.zarr"
        convert(
            inventory,
            selection,
            reference_output,
            chunks=forced_chunks,
            output_layout=layout,
            auto_tune=False,
            max_workers=1,
            progress=False,
        )
        with (
            xr.open_zarr(
                tuned_output,
                chunks=None,
                mask_and_scale=False,
                consolidated=False,
            ) as tuned,
            xr.open_zarr(
                reference_output,
                chunks=None,
                mask_and_scale=False,
                consolidated=False,
            ) as reference,
        ):
            self.assertEqual(tuned.temperature.encoding["chunks"], forced_chunks)
            np.testing.assert_array_equal(
                tuned.temperature.values,
                reference.temperature.values,
            )

    def test_tuner_measures_and_cleans_trials(self) -> None:
        inventory = inspect_dataset(ROOT / "large", workers=1, progress=False)
        selection = make_selection(inventory, variables=["temperature"])
        candidates = [
            ConversionPlan("chunk", 1, 3, 6, 8),
            ConversionPlan("chunk", 2, 4, 6, 8),
        ]
        chosen, results = tune(
            inventory,
            selection,
            ROOT / "tuned-output.zarr",
            candidates,
            budget_seconds=10,
            progress=False,
        )
        self.assertIn(chosen, candidates)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(item.physical_bytes > 0 for item in results))
        self.assertFalse(list(ROOT.glob(".fast-nc-zarr-tune-*")))

    def test_tuner_caps_large_probe_size(self) -> None:
        selection = Selection(
            ("temperature", "quality", "permuted"),
            0,
            1000,
            0,
            3600,
            0,
            7200,
        )
        sample = representative_selection(
            selection,
            [ConversionPlan("chunk", 4, 64, 2160, 4320)],
            bytes_per_cell=9,
        )
        self.assertLessEqual(sample.shape[0] * sample.shape[1] * sample.shape[2] * 9, 1024 * 1024**2)

    def test_inspection_ignores_variable_iteration_order(self) -> None:
        folder = ROOT / "reordered"
        folder.mkdir()
        with xr.open_dataset(ROOT / "large" / "year-a.nc", engine="h5netcdf") as source:
            first = source[["temperature", "quality", "permuted"]].load()
        with xr.open_dataset(ROOT / "large" / "year-b.nc", engine="h5netcdf") as source:
            second = source[["permuted", "quality", "temperature"]].load()
        first.to_netcdf(folder / "a.nc", engine="h5netcdf")
        second.to_netcdf(folder / "b.nc", engine="h5netcdf")
        first.close()
        second.close()
        inventory = inspect_dataset(folder, engine="h5netcdf", workers=1, progress=False)
        self.assertEqual(set(inventory.variables), {"temperature", "quality", "permuted"})

    def test_inspection_ignores_cf_auxiliary_schema_variants(self) -> None:
        folder = ROOT / "cf-auxiliary-variants"
        folder.mkdir()
        write_cf_auxiliary_fixture(folder / "a.nc", 0)
        write_cf_auxiliary_fixture(
            folder / "b.nc", 1, bounds_units="days since 2000-01-01"
        )
        write_cf_auxiliary_fixture(folder / "c.nc", 2, bounds_dtype="int64")

        inventory = inspect_dataset(folder, engine="h5netcdf", workers=1, progress=False)

        self.assertEqual(set(inventory.variables), {"value"})
        self.assertEqual(len(inventory.times), 3)

        xarray_inventory = inspect_dataset(
            folder, engine="netcdf4", workers=1, progress=False
        )
        self.assertEqual(set(xarray_inventory.variables), {"value"})

    def test_inspection_rejects_science_variable_schema_changes(self) -> None:
        folder = ROOT / "science-schema-mismatch"
        folder.mkdir()
        write_cf_auxiliary_fixture(folder / "a.nc", 0)
        write_cf_auxiliary_fixture(folder / "b.nc", 1, value_units="kg m-2 d-1")

        with self.assertRaisesRegex(ValueError, "变量定义与首文件不同"):
            inspect_dataset(folder, engine="h5netcdf", workers=1, progress=False)

    def test_h5netcdf_inspection_avoids_xarray_dataset_construction(self) -> None:
        with patch(
            "fast_nc_zarr.inspection._inspect_file_xarray",
            side_effect=AssertionError("unexpected xarray fallback"),
        ):
            inventory = inspect_dataset(
                ROOT / "small", engine="h5netcdf", workers=1, progress=False
            )

        self.assertEqual(len(inventory.files), 8)
        self.assertEqual(
            set(inventory.variables), {"temperature", "quality", "permuted"}
        )
        self.assertEqual(inventory.variables["temperature"].native_chunks, (1, 4, 4))

    def test_inspection_cache_reopens_only_changed_files(self) -> None:
        first = inspect_dataset(ROOT / "small", workers=1, progress=False)
        changed = ROOT / "small" / "day-003.nc"
        original_mtime = changed.stat().st_mtime_ns
        changed.touch()
        if changed.stat().st_mtime_ns == original_mtime:
            os.utime(changed, ns=(original_mtime + 1_000_000, original_mtime + 1_000_000))

        from fast_nc_zarr import inspection as inspection_module

        inspected = []
        original = inspection_module.inspect_file

        def record_call(path, *args, **kwargs):
            inspected.append(path)
            return original(path, *args, **kwargs)

        with patch.object(inspection_module, "inspect_file", side_effect=record_call):
            second = inspect_dataset(
                ROOT / "small",
                workers=1,
                progress=False,
                cached_inventory=first,
            )

        self.assertEqual(inspected, [changed])
        self.assertEqual(len(second.files), len(first.files))

    def test_nonstandard_dimensions_require_mapping_and_write_canonical_zarr(self) -> None:
        with self.assertRaises(DimensionMappingRequired):
            inspect_dataset(ROOT / "aliased", workers=1, progress=False)

        inventory = inspect_dataset(
            ROOT / "aliased",
            dimension_names=("time", "latitude", "longitude"),
            workers=1,
            progress=False,
        )
        self.assertEqual(inventory.source_dimensions, ("time", "latitude", "longitude"))
        self.assertEqual(inventory.variables["temperature"].dims, ("time", "lat", "lon"))
        selection = make_selection(inventory, variables=["temperature", "permuted"])
        output = ROOT / "aliased-output.zarr"
        direct_write(
            inventory,
            selection,
            output,
            ConversionPlan("chunk", 1, 2, 6, 8),
            progress=False,
        )
        validate_output(inventory, selection, output)
        with xr.open_zarr(output, chunks=None, consolidated=False) as actual:
            self.assertEqual(set(actual.sizes), {"time", "lat", "lon"})
            self.assertEqual(set(actual.data_vars), {"temperature", "permuted"})

    def test_nonstandard_dimensions_use_canonical_names_in_dask_fallback(self) -> None:
        inventory = inspect_dataset(
            ROOT / "aliased",
            dimension_names=("time", "latitude", "longitude"),
            workers=1,
            progress=False,
        )
        selection = make_selection(inventory, variables=["temperature", "scalar"])
        output = ROOT / "aliased-dask-output.zarr"
        plan, _ = convert(
            inventory,
            selection,
            output,
            auto_tune=False,
            validate=True,
            progress=False,
        )
        self.assertEqual(plan.strategy, "dask")
        with xr.open_zarr(output, chunks=None, consolidated=False) as actual:
            self.assertEqual(set(actual.sizes), {"time", "lat", "lon"})
            self.assertEqual(set(actual.data_vars), {"temperature", "scalar"})

    def test_dask_fallback_consumes_output_layout(self) -> None:
        inventory = inspect_dataset(
            ROOT / "aliased",
            dimension_names=("time", "latitude", "longitude"),
            workers=1,
            progress=False,
        )
        selection = make_selection(inventory, variables=["temperature", "scalar"])
        output = ROOT / "aliased-dask-layout-output.zarr"
        shape = selection.shape
        codec = CodecSpec("blosc", level=1, cname="zstd")
        layout = OutputLayout(
            variables=(
                VariableOutputLayout(
                    "temperature", "temperature", ("time", "lat", "lon"),
                    shape, "float32", (2, 6, 8), codec=codec,
                ),
                VariableOutputLayout(
                    "time", "time", ("time",), (shape[0],),
                    str(inventory.times.dtype), (shape[0],), codec=codec, is_coord=True,
                ),
                VariableOutputLayout(
                    "lat", "lat", ("lat",), (shape[1],),
                    str(inventory.lat_values.dtype), (shape[1],), codec=codec, is_coord=True,
                ),
                VariableOutputLayout(
                    "lon", "lon", ("lon",), (shape[2],),
                    str(inventory.lon_values.dtype), (shape[2],), codec=codec, is_coord=True,
                ),
            )
        )
        plan, _ = convert(
            inventory,
            selection,
            output,
            auto_tune=False,
            output_layout=layout,
            validate=True,
            progress=False,
        )
        self.assertEqual(plan.strategy, "dask")
        with xr.open_zarr(output, chunks=None, consolidated=False) as actual:
            self.assertEqual(actual.temperature.encoding["chunks"], (2, 6, 8))
            self.assertEqual(tuple(actual.temperature.dims), ("time", "lat", "lon"))


if __name__ == "__main__":
    unittest.main()
