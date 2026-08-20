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
    _hdf_eos_grid_values,
    _hdf_eos_swath_axis,
    _hdf_eos_swath_structure,
    convert_filename,
    filename_direct_write,
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

_SWATH_CORE_METADATA = (
    "GROUP=SwathStructMetadata\n"
    "  GROUP=Swath_1\n"
    "    OBJECT=SwathStructure\n"
    "      GROUP=Dimension\n"
    "        OBJECT=Dimension_1\n"
    "          DimensionName=\"AlongTrack\"\n"
    "          Size=2\n"
    "        END_OBJECT=Dimension_1\n"
    "        OBJECT=Dimension_2\n"
    "          DimensionName=\"CrossTrack\"\n"
    "          Size=3\n"
    "        END_OBJECT=Dimension_2\n"
    "      END_GROUP=Dimension\n"
    "      GROUP=DataField\n"
    "        OBJECT=DataField_1\n"
    "          DataFieldName=\"EVI\"\n"
    "          DataType=DFNT_INT16\n"
    "          DimList=(\"AlongTrack\",\"CrossTrack\")\n"
    "        END_OBJECT=DataField_1\n"
    "      END_GROUP=DataField\n"
    "      GROUP=GeoField\n"
    "        OBJECT=GeoField_1\n"
    "          GeoFieldName=\"Latitude\"\n"
    "          DataType=DFNT_FLOAT32\n"
    "          DimList=(\"AlongTrack\",\"CrossTrack\")\n"
    "        END_OBJECT=GeoField_1\n"
    "        OBJECT=GeoField_2\n"
    "          GeoFieldName=\"Longitude\"\n"
    "          DataType=DFNT_FLOAT32\n"
    "          DimList=(\"AlongTrack\",\"CrossTrack\")\n"
    "        END_OBJECT=GeoField_2\n"
    "      END_GROUP=GeoField\n"
    "    END_OBJECT=SwathStructure\n"
    "  END_GROUP=Swath_1\n"
    "END_GROUP=SwathStructMetadata\n"
)


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
        progress: list[tuple[int, int, int | None, str | None]] = []
        _, metrics = convert_filename(
            inventory,
            selection,
            output,
            transforms={"value": VariableTransform(fill_values=(-99,), scale_factor=2)},
            variable_names={"value": "renamed_value"},
            plan=ConversionPlan("file", 1, 1, 2, 2),
            progress=False,
            progress_callback=lambda completed, total, logical_bytes, message: progress.append(
                (completed, total, logical_bytes, message)
            ),
        )
        self.assertEqual(metrics["logical_bytes"], 4 * 2 * 2 * np.dtype("float32").itemsize)
        self.assertEqual(progress[-1][:3], (metrics["logical_bytes"],) * 3)
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

    def test_hdf_eos_grid_filename_conversion_reconstructs_coordinates(self) -> None:
        from netCDF4 import Dataset

        folder = ROOT / "low-level-hdf-convert"
        folder.mkdir()
        metadata = (
            "GROUP=GridStructure\n"
            "UpperLeftPointMtrs=(-180000000,90000000)\n"
            "LowerRightMtrs=(180000000,-90000000)\n"
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
                variable[:] = index

        scan = scan_filename_times(folder)
        inventory = inspect_filename_inventory(
            scan, requested_engine="netcdf4", workers=1, progress=False
        )
        np.testing.assert_allclose(inventory.lat_values, [45.0, -45.0])
        np.testing.assert_allclose(inventory.lon_values, [-120.0, 0.0, 120.0])

        output = ROOT / "low-level-hdf-convert-output.zarr"
        convert_filename(
            inventory,
            make_selection(inventory),
            output,
            plan=ConversionPlan("file", 1, 1, 2, 3),
            auto_tune=False,
            progress=False,
            validate=True,
        )
        with xr.open_zarr(
            output, consolidated=False, chunks=None, decode_times=False
        ) as result:
            np.testing.assert_allclose(result["lat"].values, [45.0, -45.0])
            np.testing.assert_allclose(result["lon"].values, [-120.0, 0.0, 120.0])
            self.assertEqual(result["value"].shape, (2, 2, 3))

    def test_hdf_eos_grid_values_parses_meters_and_degrees(self) -> None:
        meters = (
            "GROUP=GridStructure\n"
            "UpperLeftPointMtrs=(-180000000,90000000)\n"
            "LowerRightMtrs=(180000000,-90000000)\n"
        )
        lat = _hdf_eos_grid_values(meters, 2, "lat")
        lon = _hdf_eos_grid_values(meters, 3, "lon")
        assert lat is not None and lon is not None
        np.testing.assert_allclose(lat, [45.0, -45.0])
        np.testing.assert_allclose(lon, [-120.0, 0.0, 120.0])

        degrees = (
            "GROUP=GridStructure\n"
            "UpperLeftPointMtrs=(-90,45)\n"
            "LowerRightMtrs=(90,-45)\n"
        )
        lat_deg = _hdf_eos_grid_values(degrees, 2, "lat")
        assert lat_deg is not None
        np.testing.assert_allclose(lat_deg, [22.5, -22.5])

    def test_hdf_eos_grid_values_falls_back_without_bounds(self) -> None:
        self.assertIsNone(_hdf_eos_grid_values("GROUP=GridStructure\n", 4, "lat"))
        self.assertIsNone(
            _hdf_eos_grid_values(
                "UpperLeftPointMtrs=(not-a-number,90)\n"
                "LowerRightMtrs=(180,-90)\n",
                4,
                "lon",
            )
        )

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

    def test_phase_batch_write_matches_normal_write(self) -> None:
        folder = ROOT / "phase-batch"
        folder.mkdir()
        lat = np.asarray([10.0, 20.0, 30.0], dtype="float32")
        lon = np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float32")
        for index, doy in enumerate(("001", "005")):
            dataset = xr.Dataset(
                {
                    "value": (
                        ("latitude", "longitude"),
                        np.arange(12, dtype="float32").reshape(3, 4) + index * 10,
                    )
                },
                coords={"latitude": lat, "longitude": lon},
            )
            dataset.to_netcdf(folder / f"product_2001{doy}.nc", engine="h5netcdf")
            dataset.close()

        inventory = inspect_filename_inventory(
            scan_filename_times(folder), workers=1, progress=False
        )
        selection = make_selection(inventory)
        plan = ConversionPlan("file", 2, 1, 2, 3, task_batch=1)
        normal = ROOT / "phase-batch-normal.zarr"
        phased = ROOT / "phase-batch-phased.zarr"
        filename_direct_write(inventory, selection, normal, plan, progress=False)
        filename_direct_write(
            inventory,
            selection,
            phased,
            plan,
            progress=False,
            phase_batch=True,
        )
        with (
            xr.open_zarr(normal, consolidated=False, chunks=None, decode_times=False) as left,
            xr.open_zarr(phased, consolidated=False, chunks=None, decode_times=False) as right,
        ):
            np.testing.assert_array_equal(left["value"].values, right["value"].values)
            np.testing.assert_array_equal(left["lat"].values, right["lat"].values)
            np.testing.assert_array_equal(left["lon"].values, right["lon"].values)


    def test_hdf_eos_swath_structure_parses_core_metadata(self) -> None:
        metadata = (
            "GROUP=INVENTORYMETADATA\n"
            "  GROUP=ECSMETADATA\n"
            "  END_GROUP=ECSMETADATA\n"
            "END_GROUP=INVENTORYMETADATA\n"
            "GROUP=SwathStructMetadata\n"
            "  GROUP=Swath_1\n"
            "    OBJECT=SwathStructure\n"
            "      GROUP=Dimension\n"
            "        OBJECT=Dimension_1\n"
            "          DimensionName=\"AlongTrack\"\n"
            "          Size=2\n"
            "        END_OBJECT=Dimension_1\n"
            "        OBJECT=Dimension_2\n"
            "          DimensionName=\"CrossTrack\"\n"
            "          Size=3\n"
            "        END_OBJECT=Dimension_2\n"
            "      END_GROUP=Dimension\n"
            "      GROUP=DataField\n"
            "        OBJECT=DataField_1\n"
            "          DataFieldName=\"EVI\"\n"
            "          DataType=DFNT_INT16\n"
            "          DimList=(\"AlongTrack\",\"CrossTrack\")\n"
            "        END_OBJECT=DataField_1\n"
            "      END_GROUP=DataField\n"
            "      GROUP=GeoField\n"
            "        OBJECT=GeoField_1\n"
            "          GeoFieldName=\"Latitude\"\n"
            "          DataType=DFNT_FLOAT32\n"
            "          DimList=(\"AlongTrack\",\"CrossTrack\")\n"
            "        END_OBJECT=GeoField_1\n"
            "        OBJECT=GeoField_2\n"
            "          GeoFieldName=\"Longitude\"\n"
            "          DataType=DFNT_FLOAT32\n"
            "          DimList=(\"AlongTrack\",\"CrossTrack\")\n"
            "        END_OBJECT=GeoField_2\n"
            "      END_GROUP=GeoField\n"
            "    END_OBJECT=SwathStructure\n"
            "  END_GROUP=Swath_1\n"
            "END_GROUP=SwathStructMetadata\n"
        )
        swath = _hdf_eos_swath_structure(metadata)
        assert swath is not None
        self.assertEqual(swath.name, "Swath_1")
        self.assertEqual(swath.dimensions, ("AlongTrack", "CrossTrack"))
        self.assertEqual(swath.data_fields, ("EVI",))
        self.assertEqual(swath.geo_fields, ("Latitude", "Longitude"))
        self.assertEqual(swath.lat_field, "Latitude")
        self.assertEqual(swath.lon_field, "Longitude")

    def test_hdf_eos_swath_structure_ignores_non_swath_metadata(self) -> None:
        self.assertIsNone(_hdf_eos_swath_structure(""))
        self.assertIsNone(
            _hdf_eos_swath_structure(
                "GROUP=GridStructure\nUpperLeftPointMtrs=(-90,45)\n"
            )
        )
        # GeoFields without a latitude/longitude pair are not usable.
        self.assertIsNone(
            _hdf_eos_swath_structure(
                "GROUP=SwathStructMetadata\n"
                "  GROUP=Swath_1\n"
                "    GROUP=GeoField\n"
                "      OBJECT=GeoField_1\n"
                "        GeoFieldName=\"Radiance\"\n"
                "        DataType=DFNT_FLOAT32\n"
                "        DimList=(\"AlongTrack\",\"CrossTrack\")\n"
                "      END_OBJECT=GeoField_1\n"
                "    END_GROUP=GeoField\n"
                "  END_GROUP=Swath_1\n"
                "END_GROUP=SwathStructMetadata\n"
            )
        )

    def test_hdf_eos_swath_axis_reconstructs_regular_axes(self) -> None:
        # Latitude varies along rows (along-track), longitude along columns
        # (cross-track): the canonical degenerate-swath layout.
        lat_2d = np.tile(np.array([30.0, 40.0]), (3, 1)).T
        lon_2d = np.tile(np.array([100.0, 120.0, 140.0]), (2, 1))
        lat_values, lat_index = _hdf_eos_swath_axis(lat_2d, lon_2d, "lat")
        lon_values, lon_index = _hdf_eos_swath_axis(lat_2d, lon_2d, "lon")
        assert lat_values is not None and lon_values is not None
        np.testing.assert_allclose(lat_values, [30.0, 40.0])
        np.testing.assert_allclose(lon_values, [100.0, 120.0, 140.0])
        self.assertEqual(lat_index, 0)
        self.assertEqual(lon_index, 1)

        # Micro-degree storage is normalised like Grid bounds.
        lat_micro = np.tile(np.array([3.0e7, 4.0e7]), (3, 1)).T
        lon_micro = np.tile(np.array([1.0e8, 1.2e8, 1.4e8]), (2, 1))
        lat_values, _ = _hdf_eos_swath_axis(lat_micro, lon_micro, "lat")
        lon_values, _ = _hdf_eos_swath_axis(lat_micro, lon_micro, "lon")
        assert lat_values is not None and lon_values is not None
        np.testing.assert_allclose(lat_values, [30.0, 40.0])
        np.testing.assert_allclose(lon_values, [100.0, 120.0, 140.0])

    def test_hdf_eos_swath_axis_rejects_irregular_fields(self) -> None:
        rng = np.random.default_rng(7)
        # Both directions vary: a genuinely irregular swath.
        lat_2d = rng.normal(30.0, 1.0, size=(4, 5))
        lon_2d = rng.normal(100.0, 1.0, size=(4, 5))
        self.assertEqual(_hdf_eos_swath_axis(lat_2d, lon_2d, "lat"), (None, None))
        self.assertEqual(_hdf_eos_swath_axis(lat_2d, lon_2d, "lon"), (None, None))

        # A constant field carries no geolocation.
        constant = np.full((4, 5), 45.0)
        self.assertEqual(_hdf_eos_swath_axis(constant, constant, "lat"), (None, None))

        # Fill-like values mixed into a plausible-degree axis are rejected.
        mixed = np.tile(np.array([30.0, 40.0]), (3, 1)).T
        mixed[1, 1] = -9999.0
        lon_ok = np.tile(np.array([100.0, 120.0, 140.0]), (2, 1))
        self.assertEqual(_hdf_eos_swath_axis(mixed, lon_ok, "lat"), (None, None))

    def test_hdf_eos_swath_axis_same_dimension_falls_back_to_index(self) -> None:
        from netCDF4 import Dataset
        from fast_nc_zarr.filename_mode import _low_level_axis_values

        folder = ROOT / "degenerate-swath"
        folder.mkdir()
        path = folder / "swath_2001001.hdf"
        metadata = (
            "GROUP=SwathStructMetadata\n"
            "  GROUP=Swath_1\n"
            "    OBJECT=SwathStructure\n"
            "      GROUP=Dimension\n"
            "        OBJECT=Dimension_1\n"
            "          DimensionName=\"YDim:T\"\n"
            "          Size=2\n"
            "        END_OBJECT=Dimension_1\n"
            "        OBJECT=Dimension_2\n"
            "          DimensionName=\"XDim:T\"\n"
            "          Size=3\n"
            "        END_OBJECT=Dimension_2\n"
            "      END_GROUP=Dimension\n"
            "      GROUP=GeoField\n"
            "        OBJECT=GeoField_1\n"
            "          GeoFieldName=\"Latitude\"\n"
            "          DataType=DFNT_FLOAT32\n"
            "          DimList=(\"YDim:T\",\"XDim:T\")\n"
            "        END_OBJECT=GeoField_1\n"
            "        OBJECT=GeoField_2\n"
            "          GeoFieldName=\"Longitude\"\n"
            "          DataType=DFNT_FLOAT32\n"
            "          DimList=(\"YDim:T\",\"XDim:T\")\n"
            "        END_OBJECT=GeoField_2\n"
            "      END_GROUP=GeoField\n"
            "    END_OBJECT=SwathStructure\n"
            "  END_GROUP=Swath_1\n"
            "END_GROUP=SwathStructMetadata\n"
        )
        with Dataset(path, "w", format="NETCDF4") as dataset:
            dataset.createDimension("YDim:T", 2)
            dataset.createDimension("XDim:T", 3)
            dataset.setncattr("CoreMetadata.0", metadata)
            latitude = dataset.createVariable(
                "Latitude", "f8", ("YDim:T", "XDim:T")
            )
            # Both fields vary along the first dimension only, so the
            # lat/lon mapping is ambiguous and must fall back cleanly.
            latitude[:] = np.tile(np.array([30.0, 40.0]), (3, 1)).T
            longitude = dataset.createVariable(
                "Longitude", "f8", ("YDim:T", "XDim:T")
            )
            longitude[:] = np.tile(np.array([100.0, 120.0]), (3, 1)).T
        with Dataset(path, mode="r") as dataset:
            np.testing.assert_array_equal(
                _low_level_axis_values(dataset, "YDim:T", "lat"),
                np.arange(2, dtype="float64"),
            )

    def _write_swath_files(self, folder: Path, metadata: str) -> None:
        from netCDF4 import Dataset

        for index, doy in enumerate(("001", "005")):
            path = folder / f"swath_2001{doy}.hdf"
            with Dataset(path, "w", format="NETCDF4") as dataset:
                dataset.createDimension("AlongTrack", 2)
                dataset.createDimension("CrossTrack", 3)
                dataset.setncattr("CoreMetadata.0", metadata)
                latitude = dataset.createVariable(
                    "Latitude", "f8", ("AlongTrack", "CrossTrack")
                )
                latitude[:] = np.tile(np.array([30.0, 40.0]), (3, 1)).T
                longitude = dataset.createVariable(
                    "Longitude", "f8", ("AlongTrack", "CrossTrack")
                )
                longitude[:] = np.tile(np.array([100.0, 120.0, 140.0]), (2, 1))
                evi = dataset.createVariable(
                    "EVI", "i2", ("AlongTrack", "CrossTrack"), fill_value=-9999
                )
                evi[:] = index

    def test_hdf_eos_swath_low_level_scan_reconstructs_coordinates(self) -> None:
        folder = ROOT / "low-level-swath"
        folder.mkdir()
        self._write_swath_files(folder, _SWATH_CORE_METADATA)

        scan = scan_filename_times(folder)
        inventory = inspect_filename_inventory(
            scan, requested_engine="netcdf4", workers=1, progress=False
        )
        self.assertEqual(inventory.source_engine, "netcdf4")
        np.testing.assert_allclose(inventory.lat_values, [30.0, 40.0])
        np.testing.assert_allclose(inventory.lon_values, [100.0, 120.0, 140.0])
        self.assertEqual(tuple(inventory.variables), ("EVI",))
        self.assertEqual(inventory.variables["EVI"].attrs["_FillValue"], -9999)

    def test_hdf_eos_swath_filename_conversion_reconstructs_coordinates(self) -> None:
        folder = ROOT / "low-level-swath-convert"
        folder.mkdir()
        self._write_swath_files(folder, _SWATH_CORE_METADATA)

        scan = scan_filename_times(folder)
        inventory = inspect_filename_inventory(
            scan, requested_engine="netcdf4", workers=1, progress=False
        )
        output = ROOT / "low-level-swath-convert-output.zarr"
        convert_filename(
            inventory,
            make_selection(inventory),
            output,
            plan=ConversionPlan("file", 1, 1, 2, 3),
            auto_tune=False,
            progress=False,
            validate=True,
        )
        with xr.open_zarr(
            output, consolidated=False, chunks=None, decode_times=False
        ) as result:
            np.testing.assert_allclose(result["lat"].values, [30.0, 40.0])
            np.testing.assert_allclose(result["lon"].values, [100.0, 120.0, 140.0])
            self.assertEqual(result["EVI"].shape, (2, 2, 3))
            np.testing.assert_array_equal(
                result["EVI"].isel(time=0).values,
                np.zeros((2, 3), dtype="int16"),
            )
            # GeoFields are location metadata, not output variables.
            self.assertNotIn("Latitude", result.data_vars)
            self.assertNotIn("Longitude", result.data_vars)

    def test_hdf_eos_swath_xarray_normalization_drops_geofields(self) -> None:
        from fast_nc_zarr.filename_mode import _normalized_filename_dataset

        folder = ROOT / "xarray-swath"
        folder.mkdir()
        self._write_swath_files(folder, _SWATH_CORE_METADATA)

        with _normalized_filename_dataset(
            folder / "swath_2001001.hdf", "netcdf4"
        ) as (ds, engine):
            self.assertEqual(engine, "netcdf4")
            np.testing.assert_allclose(ds.lat.values, [30.0, 40.0])
            np.testing.assert_allclose(ds.lon.values, [100.0, 120.0, 140.0])
            self.assertEqual(tuple(ds.data_vars), ("EVI",))
            self.assertEqual(tuple(ds.EVI.dims), ("lat", "lon"))


if __name__ == "__main__":
    unittest.main()
