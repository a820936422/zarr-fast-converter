from __future__ import annotations

import os
import shutil
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import xarray as xr

from fast_nc_zarr.filename_mode import (
    FilenameTimeError,
    convert_filename,
    inspect_filename_inventory,
    scan_filename_times,
)
from fast_nc_zarr.inspection import choose_inspection_workers
from fast_nc_zarr.models import (
    ConversionPlan,
    OutputLayout,
    VariableOutputLayout,
    VariableTransform,
)
from fast_nc_zarr.selection import make_selection


ROOT = Path("/tmp/codex_test/fast_nc_zarr_filename_tests")


class FilenameModeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)
        ROOT.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_annual_doy_inference_does_not_use_cross_year_gap(self) -> None:
        folder = ROOT / "csif"
        folder.mkdir()
        for name in (
            "OCO2.SIF.clear.inst.2001365.v2.nc",
            "OCO2.SIF.clear.inst.2002001.v2.nc",
            "OCO2.SIF.clear.inst.2002005.v2.nc",
        ):
            (folder / name).touch()
        scan = scan_filename_times(folder)
        self.assertEqual(scan.template, "doy")
        self.assertEqual(scan.sample_start, 20)
        self.assertEqual(scan.annual_steps, ((2001, 4), (2002, 4)))
        self.assertEqual(scan.step_days, 4)
        self.assertEqual(len(scan.expected_times), 3)
        self.assertFalse(scan.missing_times)

    def test_interior_year_is_not_truncated_by_missing_final_day(self) -> None:
        folder = ROOT / "interior-year"
        folder.mkdir()
        for name in (
            "product_2001001.nc",
            "product_2002001.nc",
            "product_2002002.nc",
            "product_2003001.nc",
        ):
            (folder / name).touch()
        scan = scan_filename_times(folder)
        self.assertEqual(len(scan.expected_times), 367)
        self.assertIn(
            np.datetime64("2002-12-31", "ns"),
            scan.expected_times,
        )
        self.assertIn(
            np.datetime64("2002-12-31", "ns"),
            scan.missing_times,
        )

    def test_automatic_workers_follow_effective_resource_ceiling(self) -> None:
        files = [Path(f"/media/source/file-{index}.nc") for index in range(8008)]
        with patch(
            "fast_nc_zarr.inspection.effective_resource_budget",
            return_value=SimpleNamespace(worker_ceiling=16),
        ):
            self.assertEqual(choose_inspection_workers(files), 16)

    def test_multiple_changing_date_fields_are_not_guessed(self) -> None:
        folder = ROOT / "ambiguous"
        folder.mkdir()
        for name in ("a.2001001.2001001.nc", "a.2001005.2001005.nc"):
            (folder / name).touch()
        with self.assertRaises(FilenameTimeError):
            scan_filename_times(folder)

    def test_repeated_processing_date_does_not_mask_observation_doy(self) -> None:
        folder = ROOT / "glass-processing-date"
        folder.mkdir()
        for name in (
            "GLASS14B01.V10.A2001001.2023068.hdf",
            "GLASS14B01.V10.A2001009.2023068.hdf",
            "GLASS14B01.V10.A2001017.2025133.hdf",
        ):
            (folder / name).touch()
        scan = scan_filename_times(folder)
        self.assertEqual(scan.template, "doy")
        self.assertEqual(scan.sample_start, 16)
        self.assertEqual(scan.sample_length, 7)
        self.assertEqual(
            [str(value)[:10] for value in scan.actual_times],
            ["2001-01-01", "2001-01-09", "2001-01-17"],
        )

    def test_manual_rule_allows_non_time_tokens_to_change(self) -> None:
        folder = ROOT / "manual"
        folder.mkdir()
        for name in ("a_2001001_v.nc", "a_2001005_w.nc"):
            (folder / name).touch()
        with self.assertRaises(FilenameTimeError):
            scan_filename_times(folder)
        scan = scan_filename_times(
            folder, template="doy", field_values=("2001", "001")
        )
        self.assertEqual(len(scan.actual_times), 2)

    def test_inventory_and_gap_conversion(self) -> None:
        folder = ROOT / "data"
        folder.mkdir()
        lat = np.asarray([10, 20], dtype="float32")
        lon = np.asarray([30, 40], dtype="float32")
        for index, doy in enumerate(("001", "005", "013")):
            dataset = xr.Dataset(
                {
                    "value": (
                        ("latitude", "longitude"),
                        np.asarray([[index, -99], [index + 1, index + 2]], dtype="int16"),
                        {"missing_value": -99, "units": "1"},
                    )
                },
                coords={"latitude": lat, "longitude": lon},
            )
            dataset.to_netcdf(folder / f"product_2001{doy}.nc", engine="h5netcdf")
            dataset.close()

        scan = scan_filename_times(folder)
        inventory = inspect_filename_inventory(scan, workers=2, progress=False)
        self.assertEqual(inventory.source_mode, "filename")
        self.assertEqual(len(inventory.missing_time_keys), 1)
        selection = make_selection(inventory)
        output = ROOT / "gap-output.zarr"
        convert_filename(
            inventory,
            selection,
            output,
            transforms={"value": VariableTransform(fill_values=(-99,), scale_factor=2)},
            variable_names={"value": "renamed_value"},
            plan=ConversionPlan("file", 1, 1, 2, 2),
            progress=False,
        )
        with xr.open_zarr(output, consolidated=False, chunks=None, mask_and_scale=False) as dataset:
            np.testing.assert_equal(
                dataset.time.values,
                np.asarray(
                    [
                        "2001-01-01",
                        "2001-01-05",
                        "2001-01-09",
                        "2001-01-13",
                    ],
                    dtype="datetime64[ns]",
                ),
            )
            self.assertTrue(np.isnan(dataset.renamed_value.values[2]).all())
            self.assertEqual(dataset.sizes["lat"], 2)
            self.assertEqual(dataset.sizes["lon"], 2)

    def test_filename_auto_tuning_and_parallel_write(self) -> None:
        folder = ROOT / "tuned"
        folder.mkdir()
        lat = np.asarray([10, 20], dtype="float32")
        lon = np.asarray([30, 40], dtype="float32")
        for index, doy in enumerate(("001", "005", "009", "013")):
            dataset = xr.Dataset(
                {
                    "value": (
                        ("latitude", "longitude"),
                        np.full((2, 2), index, dtype="float32"),
                    )
                },
                coords={"latitude": lat, "longitude": lon},
            )
            dataset.to_netcdf(folder / f"product_2001{doy}.nc", engine="h5netcdf")
            dataset.close()
        scan = scan_filename_times(folder)
        inventory = inspect_filename_inventory(scan, workers=1, progress=False)
        output = ROOT / "tuned-output.zarr"
        plan, metrics = convert_filename(
            inventory,
            make_selection(inventory),
            output,
            auto_tune=True,
            tune_budget=1,
            validate=True,
            progress=False,
        )
        self.assertIn(plan.strategy, {"file", "chunk"})
        self.assertGreater(metrics["tasks"], 0)

    def test_filename_inventory_ignores_file_specific_metadata(self) -> None:
        folder = ROOT / "metadata-variation"
        folder.mkdir()
        lat = np.asarray([10, 20], dtype="float32")
        lon = np.asarray([30, 40], dtype="float32")
        for index, doy in enumerate(("001", "005")):
            dataset = xr.Dataset(
                {
                    "value": (
                        ("latitude", "longitude"),
                        np.full((2, 2), index, dtype="float32"),
                        {
                            "units": "1",
                            "scale_factor": 1.0,
                            "TIFFTAG_SOFTWARE": f"writer-{index}",
                            "STATISTICS_MEAN": float(index),
                        },
                    )
                },
                coords={"latitude": lat, "longitude": lon},
            )
            dataset.to_netcdf(folder / f"product_2001{doy}.nc", engine="h5netcdf")
            dataset.close()
        scan = scan_filename_times(folder)
        inventory = inspect_filename_inventory(scan, workers=1, progress=False)
        self.assertEqual(inventory.variables["value"].attrs["scale_factor"], 1.0)

    def test_filename_conversion_honours_external_chunks(self) -> None:
        folder = ROOT / "linked-chunks"
        folder.mkdir()
        lat = np.asarray([10, 20, 30], dtype="float32")
        lon = np.asarray([30, 40, 50, 60], dtype="float32")
        for index, doy in enumerate(("001", "005", "009")):
            dataset = xr.Dataset(
                {
                    "value": (
                        ("latitude", "longitude"),
                        np.full((3, 4), index, dtype="float32"),
                    )
                },
                coords={"latitude": lat, "longitude": lon},
            )
            dataset.to_netcdf(folder / f"product_2001{doy}.nc", engine="h5netcdf")
            dataset.close()
        scan = scan_filename_times(folder)
        inventory = inspect_filename_inventory(scan, workers=1, progress=False)
        output = ROOT / "linked-chunks-output.zarr"
        with patch(
            "fast_nc_zarr.filename_mode.tune",
            side_effect=lambda *args, **kwargs: (args[3][0], []),
        ) as tune_mock:
            plan, _ = convert_filename(
                inventory,
                make_selection(inventory),
                output,
                chunks=(2, 2, 3),
                auto_tune=True,
                progress=False,
            )
        candidates = tune_mock.call_args.args[3]
        self.assertTrue(tune_mock.call_args.kwargs["fixed_layout"])
        self.assertGreater(len({item.workers for item in candidates}), 1)
        self.assertTrue(all(item.chunks == (2, 2, 3) for item in candidates))
        self.assertTrue(
            all(item.task_batch == item.chunk_time for item in candidates)
        )
        self.assertEqual(plan.chunks, (2, 2, 3))
        with xr.open_zarr(output, consolidated=False, chunks=None, decode_times=False) as dataset:
            self.assertEqual(dataset["value"].encoding["chunks"], (2, 2, 3))

    def test_filename_conversion_applies_axis_reversals(self) -> None:
        folder = ROOT / "axis-reversals"
        folder.mkdir()
        lat = np.asarray([10, 20, 30], dtype="float32")
        lon = np.asarray([60, 50, 40, 30], dtype="float32")
        source_values = np.arange(12, dtype="float32").reshape(3, 4)
        for index, doy in enumerate(("001", "005")):
            dataset = xr.Dataset(
                {"value": (("latitude", "longitude"), source_values + index * 100)},
                coords={"latitude": lat, "longitude": lon},
            )
            dataset.to_netcdf(folder / f"product_2001{doy}.nc", engine="h5netcdf")
            dataset.close()

        inventory = inspect_filename_inventory(
            scan_filename_times(folder), workers=1, progress=False
        )
        selection = make_selection(inventory)
        shape = selection.shape
        output_layout = OutputLayout(
            variables=(
                VariableOutputLayout(
                    "value", "value", ("time", "lat", "lon"), shape,
                    "float32", (1, 2, 3),
                ),
                VariableOutputLayout(
                    "time", "time", ("time",), (shape[0],),
                    str(inventory.times.dtype), (shape[0],), is_coord=True,
                ),
                VariableOutputLayout(
                    "lat", "lat", ("lat",), (shape[1],),
                    str(inventory.lat_values.dtype), (shape[1],), is_coord=True,
                ),
                VariableOutputLayout(
                    "lon", "lon", ("lon",), (shape[2],),
                    str(inventory.lon_values.dtype), (shape[2],), is_coord=True,
                ),
            ),
            axis_reversals=("lat", "lon"),
        )
        output = ROOT / "axis-reversals-output.zarr"
        convert_filename(
            inventory,
            selection,
            output,
            plan=ConversionPlan("file", 1, 1, 2, 3),
            output_layout=output_layout,
            validate=True,
            progress=False,
        )
        with xr.open_zarr(
            output, consolidated=False, chunks=None, decode_times=False
        ) as dataset:
            np.testing.assert_equal(dataset.lat.values, lat[::-1])
            np.testing.assert_equal(dataset.lon.values, lon[::-1])
            np.testing.assert_equal(
                dataset.value.isel(time=0).values,
                source_values[::-1, ::-1],
            )

    def test_netcdf4_low_level_metadata_scan_for_hdf_eos_grid(self) -> None:
        try:
            from netCDF4 import Dataset
        except ImportError:  # pragma: no cover - optional test dependency
            self.skipTest("netCDF4 未安装")
        folder = ROOT / "low-level-hdf"
        folder.mkdir()
        metadata = (
            'GROUP=GridStructure\n'
            'UpperLeftPointMtrs=(-180000000,90000000)\n'
            'LowerRightMtrs=(180000000,-90000000)\n'
        )
        for index, doy in enumerate(("001", "005")):
            path = folder / f"product_2001{doy}.hdf"
            with Dataset(path, "w", format="NETCDF4") as dataset:
                dataset.createDimension("YDim:TEST", 2)
                dataset.createDimension("XDim:TEST", 3)
                dataset.setncattr("StructMetadata.0", metadata)
                variable = dataset.createVariable(
                    "value", "i4", ("YDim:TEST", "XDim:TEST"), fill_value=-1
                )
                variable.setncattr("scale_factor", 0.01)
                variable.setncattr("units", "1")
                variable[:] = index

        scan = scan_filename_times(folder)
        inventory = inspect_filename_inventory(
            scan, requested_engine="netcdf4", workers=1, progress=False
        )
        self.assertEqual(inventory.source_engine, "netcdf4")
        self.assertEqual(inventory.lat_values.shape, (2,))
        self.assertEqual(inventory.lon_values.shape, (3,))
        self.assertEqual(inventory.variables["value"].attrs["_FillValue"], -1)
        self.assertEqual(inventory.variables["value"].attrs["scale_factor"], 0.01)

    def test_rasterio_fallback_signature_includes_crs_and_transform(self) -> None:
        from fast_nc_zarr import filename_mode

        class FakeCrs:
            def __init__(self, value: str) -> None:
                self.value = value

            def to_wkt(self) -> str:
                return self.value

        def dataset(crs: str, transform: tuple[float, ...]) -> SimpleNamespace:
            return SimpleNamespace(
                attrs={},
                rio=SimpleNamespace(crs=FakeCrs(crs), transform=lambda: transform),
            )

        variables = (("band_data", "float32", ("lat", "lon"), (2, 3), ()),)
        reference = filename_mode._append_filename_spatial_signature(
            variables, dataset("EPSG:4326", (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)), "rasterio"
        )
        different_crs = filename_mode._append_filename_spatial_signature(
            variables, dataset("EPSG:3857", (1.0, 0.0, 0.0, 0.0, -1.0, 2.0)), "rasterio"
        )
        different_transform = filename_mode._append_filename_spatial_signature(
            variables, dataset("EPSG:4326", (2.0, 0.0, 0.0, 0.0, -1.0, 2.0)), "rasterio"
        )
        self.assertNotEqual(reference, different_crs)
        self.assertNotEqual(reference, different_transform)

    def test_rasterio_low_level_metadata_scan_avoids_xarray(self) -> None:
        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:  # pragma: no cover - optional dependency
            self.skipTest("rasterio 未安装")
        folder = ROOT / "rasterio-low-level"
        folder.mkdir()
        for index, doy in enumerate(("001", "009")):
            path = folder / f"product_2001{doy}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=3,
                count=1,
                dtype="uint16",
                transform=from_origin(-180, 90, 1, 1),
            ) as dataset:
                dataset.write(np.full((1, 2, 3), index, dtype="uint16"))
        scan = scan_filename_times(folder)
        with patch(
            "fast_nc_zarr.filename_mode._open_dataset",
            side_effect=AssertionError("rasterio metadata scan unexpectedly opened xarray"),
        ):
            inventory = inspect_filename_inventory(
                scan, requested_engine="rasterio", workers=1, progress=False
            )
        self.assertEqual(inventory.variables["band_data"].dtype, "uint16")
        np.testing.assert_allclose(inventory.lat_values, [89.5, 88.5])
        np.testing.assert_allclose(inventory.lon_values, [-179.5, -178.5, -177.5])

    def test_rasterio_dataset_cleanup_in_subprocess(self) -> None:
        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:  # pragma: no cover - optional test dependency
            self.skipTest("rasterio 未安装")

        folder = ROOT / "rasterio-cleanup"
        folder.mkdir()
        path = folder / "product_2001001.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=2,
            width=3,
            count=1,
            dtype="uint16",
            transform=from_origin(0, 2, 1, 1),
        ) as dataset:
            dataset.write(np.arange(6, dtype="uint16").reshape(1, 2, 3))

        script = """
import gc
import sys
from pathlib import Path
from fast_nc_zarr.filename_mode import _normalized_filename_dataset

with _normalized_filename_dataset(Path(sys.argv[1]), "auto") as (dataset, _):
    dataset["band_data"].isel(lat=slice(0, 2), lon=slice(0, 3)).values
del dataset
gc.collect()
"""
        environment = os.environ.copy()
        project_src = str(Path(__file__).resolve().parents[1] / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            item for item in (project_src, environment.get("PYTHONPATH", "")) if item
        )
        result = subprocess.run(
            [sys.executable, "-c", script, str(path)],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Error in sys.excepthook", result.stderr)

    def test_rasterio_filename_conversion_worker_cleanup(self) -> None:
        try:
            import rasterio
            from rasterio.transform import from_origin
        except ImportError:  # pragma: no cover - optional test dependency
            self.skipTest("rasterio 未安装")

        folder = ROOT / "rasterio-conversion"
        folder.mkdir()
        for doy, offset in (("001", 0), ("002", 10)):
            path = folder / f"product_2001{doy}.tif"
            with rasterio.open(
                path,
                "w",
                driver="GTiff",
                height=2,
                width=3,
                count=1,
                dtype="uint16",
                transform=from_origin(0, 2, 1, 1),
            ) as dataset:
                dataset.write(
                    (np.arange(6, dtype="uint16") + offset).reshape(1, 2, 3)
                )

        scan = scan_filename_times(folder)
        inventory = inspect_filename_inventory(
            scan, requested_engine="rasterio", workers=1, progress=False
        )
        output = ROOT / "rasterio-conversion-output.zarr"
        convert_filename(
            inventory,
            make_selection(inventory),
            output,
            plan=ConversionPlan("file", 1, 1, 2, 3),
            auto_tune=False,
            progress=False,
        )


if __name__ == "__main__":
    unittest.main()
