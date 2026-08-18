from __future__ import annotations

import importlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr
import zarr

from fast_nc_zarr._backend import resolve_backend, rust_capability
from fast_nc_zarr.inspection import inspect_netcdf_native
from fast_nc_zarr.rechunking.native import (
    RustMultiRechunkPlan,
    RustMultiRechunkVariablePlan,
    RustRechunkPlan,
    run_rust_multi_rechunk,
    run_rust_rechunk,
)


_CAPABILITY = rust_capability()
_RUST_NETCDF_READY = _CAPABILITY.supported and "raw.netcdf.inspect" in _CAPABILITY.operations
_RUST_RESAMPLE_READY = _CAPABILITY.supported and {"resample.nearest", "resample.bilinear"}.issubset(_CAPABILITY.operations)

_RUST_ZARR_READY = _CAPABILITY.supported and {
    "zarr.inspect",
    "zarr.read_chunk_f32",
    "zarr.read_region_f32",
    "zarr.write_f32",
    "zarr.rechunk_f32",
    "zarr.rechunk_f32_codec",
    "zarr.rechunk_f32_cancel",
    "zarr.rechunk_multi",
}.issubset(_CAPABILITY.operations)
class NativePreparationTests(unittest.TestCase):
    def test_capability_probe_is_json_safe(self) -> None:
        capability = rust_capability()
        self.assertIn(capability.name, {"rust", "python"})
        self.assertGreaterEqual(capability.protocol_version, 0)
        self.assertIsInstance(capability.operations, tuple)
        json.dumps(
            {
                "name": capability.name,
                "protocol_version": capability.protocol_version,
                "operations": capability.operations,
                "supported": capability.supported,
            }
        )
    def test_capability_matrix_matches_supported_operations(self) -> None:
        capability = rust_capability()
        supported = {
            item.operation for item in capability.capabilities if item.supported
        }
        self.assertEqual(supported, set(capability.operations))
        if capability.supported:
            detail = capability.operation("raw.netcdf.convert")
            self.assertIsNotNone(detail)
            self.assertTrue(detail.supported)
            self.assertIsNone(detail.reason)
            f64 = capability.operation("zarr.write_f64")
            self.assertIsNotNone(f64)
            self.assertTrue(f64.supported)

    def test_legacy_capability_payload_without_matrix_remains_compatible(self) -> None:
        from fast_nc_zarr import _backend

        operations = ("probe", "zarr.inspect")
        parsed = _backend._parse_capabilities({"operations": list(operations)}, operations)
        self.assertEqual(tuple(item.operation for item in parsed), operations)
        self.assertTrue(all(item.supported for item in parsed))

    def test_auto_backend_falls_back_without_rust_operation(self) -> None:
        expected = "rust" if _RUST_ZARR_READY else "python"
        self.assertEqual(resolve_backend("auto", "rechunk"), expected)

    def test_python_backend_is_always_selectable(self) -> None:
        self.assertEqual(resolve_backend("python", "rechunk"), "python")

    def test_auto_backend_resolves_native_rechunk_capability(self) -> None:
        expected = "rust" if _RUST_ZARR_READY else "python"
        self.assertEqual(resolve_backend("auto", "zarr.rechunk_f32"), expected)

    def test_standard_raw_and_resample_operations_resolve_native_capabilities(self) -> None:
        netcdf_expected = "rust" if _RUST_NETCDF_READY else "python"
        resample_expected = "rust" if _RUST_RESAMPLE_READY else "python"
        self.assertEqual(resolve_backend("auto", "raw.netcdf.inspect"), netcdf_expected)
        self.assertEqual(resolve_backend("auto", "raw.netcdf.convert"), netcdf_expected)
        self.assertEqual(resolve_backend("auto", "resample.nearest"), resample_expected)
        self.assertEqual(resolve_backend("auto", "resample.bilinear"), resample_expected)



@unittest.skipUnless(_RUST_NETCDF_READY, "Rust NetCDF native extension is not built")
class RustNetcdfInspectTests(unittest.TestCase):
    def test_inspect_standard_netcdf4_subset(self) -> None:
        import netCDF4

        with tempfile.TemporaryDirectory(prefix="fast-nc-zarr-netcdf-") as directory:
            path = Path(directory) / "sample.nc"
            with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
                dataset.createDimension("time", 2)
                dataset.createDimension("lat", 2)
                dataset.createDimension("lon", 3)
                dataset.title = "native smoke"
                time = dataset.createVariable("time", "f8", ("time",))
                time.units = "days since 2000-01-01"
                dataset.createVariable("lat", "f4", ("lat",))[:] = [10, 20]
                dataset.createVariable("lon", "f4", ("lon",))[:] = [100, 110, 120]
                value = dataset.createVariable("value", "f4", ("time", "lat", "lon"))
                value.units = "K"
                value[:] = np.zeros((2, 2, 3), dtype="float32")
            summary = inspect_netcdf_native(path)
            self.assertTrue(summary["supported_subset"])
            self.assertEqual(summary["dimensions"][0]["name"], "time")
            variable = next(item for item in summary["variables"] if item["name"] == "value")
            self.assertEqual(variable["dtype"], "float32")
            self.assertEqual(variable["shape"], [2, 2, 3])

    def test_convert_standard_netcdf4_to_zarr(self) -> None:
        import netCDF4

        with tempfile.TemporaryDirectory(prefix="fast-nc-zarr-convert-") as directory:
            path = Path(directory) / "sample.nc"
            target = Path(directory) / "sample.zarr"
            values = np.arange(12, dtype="float32").reshape(2, 2, 3)
            with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
                for name, size in (("time", 2), ("lat", 2), ("lon", 3)):
                    dataset.createDimension(name, size)
                time = dataset.createVariable("time", "i2", ("time",))
                time.units = "hours since 2000-01-01"
                time[:] = [0, 1]
                dataset.createVariable("lat", "i4", ("lat",))[:] = [10, 20]
                dataset.createVariable("lon", "i4", ("lon",))[:] = [100, 110, 120]
                value = dataset.createVariable(
                    "value", "f4", ("time", "lat", "lon"), fill_value=np.nan
                )
                value.long_name = "relative humidity"
                value[:] = values
            native = importlib.import_module("fast_nc_zarr._native")
            metrics = json.loads(native.convert_netcdf_json(str(path), str(target)))
            self.assertEqual(metrics["variables"], ["time", "lat", "lon", "value"])
            with xr.open_zarr(
                target, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
            ) as result:
                np.testing.assert_array_equal(result["time"].values, [0, 1])
                np.testing.assert_array_equal(result["lat"].values, [10, 20])
                np.testing.assert_array_equal(result["lon"].values, [100, 110, 120])
                self.assertEqual(result["time"].dtype, np.dtype("int16"))
                self.assertEqual(result["lat"].dtype, np.dtype("int32"))
                self.assertEqual(result["lon"].dtype, np.dtype("int32"))
                self.assertEqual(result["value"].attrs["long_name"], "relative humidity")

    def test_convert_preserves_packed_float_semantics(self) -> None:
        import netCDF4

        with tempfile.TemporaryDirectory(prefix="fast-nc-zarr-packed-netcdf-") as directory:
            path = Path(directory) / "packed.nc"
            target = Path(directory) / "packed.zarr"
            raw = np.asarray([1.0, 2.0, -9999.0, 4.0], dtype="float32").reshape(1, 2, 2)
            with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
                dataset.createDimension("time", 1)
                dataset.createDimension("lat", 2)
                dataset.createDimension("lon", 2)
                dataset.createVariable("time", "i2", ("time",))[:] = [0]
                dataset.createVariable("lat", "f4", ("lat",))[:] = [0, 1]
                dataset.createVariable("lon", "f4", ("lon",))[:] = [0, 1]
                value = dataset.createVariable(
                    "value", "f4", ("time", "lat", "lon"), fill_value=-9999.0
                )
                value.set_auto_maskandscale(False)
                value.scale_factor = 0.1
                value.add_offset = 273.15
                value[:] = raw
            native = importlib.import_module("fast_nc_zarr._native")
            metrics = json.loads(native.convert_netcdf_json(str(path), str(target)))
            self.assertEqual(metrics["variables"], ["time", "lat", "lon", "value"])
            expected = raw.astype("float64") * 0.1 + 273.15
            expected[0, 1, 0] = np.nan
            with xr.open_zarr(
                target,
                consolidated=False,
                chunks=None,
                decode_times=False,
                mask_and_scale=False,
            ) as result:
                np.testing.assert_allclose(
                    result["value"].values,
                    expected.astype("float32"),
                    equal_nan=True,
                )
                self.assertNotIn("scale_factor", result["value"].attrs)
                self.assertNotIn("add_offset", result["value"].attrs)
                self.assertEqual(result["value"].attrs["source_scale_factor"], 0.1)
                self.assertEqual(result["value"].attrs["source_add_offset"], 273.15)
            with xr.open_zarr(
                target,
                consolidated=False,
                chunks=None,
                decode_times=False,
                mask_and_scale=True,
            ) as decoded_result:
                np.testing.assert_allclose(
                    decoded_result["value"].values,
                    expected.astype("float32"),
                    equal_nan=True,
                )

@unittest.skipUnless(_RUST_ZARR_READY, "Rust Zarr native extension is not built")
class RustZarrCrossBackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.native = importlib.import_module("fast_nc_zarr._native")

        cls.tempdir = tempfile.TemporaryDirectory(prefix="fast-nc-zarr-rust-zarr-")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tempdir.cleanup()

    def test_rust_write_is_readable_by_python(self) -> None:
        root = f"{self.tempdir.name}/rust-output.zarr"
        values = np.arange(4 * 3 * 2, dtype="float32").reshape(4, 3, 2)
        self.native.write_f32_array(
            root,
            "/value",
            [4, 3, 2],
            [2, 2, 1],
            values.ravel().tolist(),
        )
        summary = json.loads(self.native.inspect_array_json(root, "/value"))
        self.assertEqual(summary["shape"], [4, 3, 2])
        self.assertEqual(summary["chunk_shape"], [2, 2, 1])
        self.assertEqual(summary["data_type"], "float32")
        with xr.open_zarr(root, consolidated=False, chunks=None, decode_times=False) as dataset:
            np.testing.assert_array_equal(dataset["value"].values, values)

    def test_python_write_is_readable_by_rust(self) -> None:
        root = f"{self.tempdir.name}/python-output.zarr"
        values = np.arange(4 * 3 * 2, dtype="float32").reshape(4, 3, 2)
        dataset = xr.Dataset({"value": (("time", "lat", "lon"), values)})
        dataset.to_zarr(
            root,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={"value": {"chunks": (2, 2, 1)}},
        )
        dataset.close()
        summary = json.loads(self.native.inspect_array_json(root, "/value"))
        self.assertEqual(summary["shape"], [4, 3, 2])
        chunk = self.native.read_chunk_f32(root, "/value", [0, 0, 0])
        region = self.native.read_region_f32(root, "/value", [1, 1, 0], [2, 2, 2])
        np.testing.assert_array_equal(region, values[1:3, 1:3, :].ravel())
        np.testing.assert_array_equal(chunk, [0.0, 2.0, 6.0, 8.0])

    def test_compressed_python_zarr_region_is_decoded_by_rust(self) -> None:
        from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

        values = np.arange(4 * 5 * 6, dtype="float32").reshape(4, 5, 6)
        expected = values[1:3, 1:4, 1:5].ravel()
        for label, codec in (
            ("zstd", ZstdCodec(level=1)),
            ("blosc-zstd", BloscCodec(cname="zstd", clevel=1, shuffle="shuffle")),
            ("blosc-lz4", BloscCodec(cname="lz4", clevel=1, shuffle="shuffle")),
            ("gzip", GzipCodec(level=1)),
        ):
            with self.subTest(codec=label):
                root = Path(self.tempdir.name) / f"compressed-{label}.zarr"
                xr.Dataset({"value": (("time", "lat", "lon"), values)}).to_zarr(
                    root,
                    mode="w",
                    consolidated=False,
                    zarr_format=3,
                    encoding={
                        "value": {
                            "chunks": (2, 3, 4),
                            "compressors": [codec],
                        }
                    },
                )
                region = self.native.read_region_f32(
                    str(root), "/value", [1, 1, 1], [2, 3, 4]
                )
                np.testing.assert_array_equal(region, expected)


    def test_committed_parity_fixtures_cover_codec_fill_and_layout(self) -> None:
        from fast_nc_zarr.resampling.engine import run_resample
        from fast_nc_zarr.resampling.models import ResampleConfig

        fixture_root = Path(__file__).resolve().parent / "fixtures" / "parity"
        float_source = fixture_root / "float-source.zarr"
        mixed_source = fixture_root / "mixed-source.zarr"
        with xr.open_zarr(
            float_source, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
        ) as dataset:
            self.assertEqual(dataset["value"].encoding["chunks"], (1, 2, 3))
            self.assertEqual(dataset["temperature"].encoding["chunks"], (1, 2, 3))
            np.testing.assert_array_equal(dataset["lat"].values, [3.5, 2.5, 1.5, 0.5])
            self.assertTrue(np.isnan(dataset["value"].values).any())
        value_summary = json.loads(self.native.inspect_array_json(str(float_source), "/value"))
        temperature_summary = json.loads(
            self.native.inspect_array_json(str(float_source), "/temperature")
        )
        self.assertEqual(value_summary["data_type"], "float32")
        self.assertEqual(value_summary["codecs"][1]["name"], "blosc")
        self.assertEqual(temperature_summary["codecs"][1]["name"], "zstd")
        region = self.native.read_region_f32(str(float_source), "/value", [0, 1, 1], [2, 2, 3])
        with xr.open_zarr(
            float_source, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
        ) as dataset:
            np.testing.assert_array_equal(region, dataset["value"].values[0:2, 1:3, 1:4].ravel())

        mixed_target = Path(self.tempdir.name) / "mixed-fixture-rechunk.zarr"
        metrics = run_rust_multi_rechunk(
            RustMultiRechunkPlan(
                source=mixed_source,
                target=mixed_target,
                variables=(
                    RustMultiRechunkVariablePlan(
                        "/value", (2, 4, 5), "float32", dimension_names=("time", "lat", "lon")
                    ),
                    RustMultiRechunkVariablePlan(
                        "/quality", (1, 4, 5), "int16", dimension_names=("time", "lat", "lon")
                    ),
                ),
                requested_workers=1,
                worker_ceiling=1,
                memory_budget_bytes=1024 * 1024,
            )
        )
        self.assertEqual(metrics["backend"], "rust")
        with xr.open_zarr(
            mixed_target, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
        ) as dataset:
            self.assertEqual(dataset["quality"].dtype, np.dtype("int16"))
            self.assertEqual(dataset["quality"].values[0, 2, 3], -9999)
            self.assertTrue(np.isnan(dataset["value"].values).any())

        resampled = Path(self.tempdir.name) / "float-fixture-resampled.zarr"
        resample_metrics = run_resample(
            ResampleConfig(
                float_source,
                resampled,
                resolution=2.0,
                method="bilinear",
                skipna=True,
                space_workers=1,
            ),
            progress=False,
        )
        self.assertEqual(resample_metrics["backend"], "rust")
        with xr.open_zarr(resampled, consolidated=False, chunks=None, mask_and_scale=False) as dataset:
            self.assertEqual(dataset["value"].dtype, np.dtype("float32"))
            self.assertEqual(dataset["value"].encoding["chunks"], (1, 2, 3))

    def test_float64_rust_write_is_readable_by_python_and_rust(self) -> None:
        root = f"{self.tempdir.name}/rust-float64-output.zarr"
        values = (np.arange(4 * 3 * 2, dtype="float64") / 10).reshape(4, 3, 2)
        self.native.write_f64_array(
            root,
            "/value",
            [4, 3, 2],
            [2, 2, 1],
            values.ravel().tolist(),
        )
        summary = json.loads(self.native.inspect_array_json(root, "/value"))
        self.assertEqual(summary["data_type"], "float64")
        chunk = self.native.read_chunk_f64(root, "/value", [0, 0, 0])
        region = self.native.read_region_f64(root, "/value", [1, 1, 0], [2, 2, 2])
        np.testing.assert_array_equal(chunk, values[:2, :2, :1].ravel())
        np.testing.assert_array_equal(region, values[1:3, 1:3, :].ravel())
        with xr.open_zarr(root, consolidated=False, chunks=None, decode_times=False) as dataset:
            self.assertEqual(dataset["value"].dtype, np.dtype("float64"))
            np.testing.assert_array_equal(dataset["value"].values, values)
    def test_float64_rust_rechunk_is_lossless(self) -> None:
        source = f"{self.tempdir.name}/float64-rechunk-source.zarr"
        target = f"{self.tempdir.name}/float64-rechunk-target.zarr"
        values = (np.arange(4 * 3 * 2, dtype="float64") / 10).reshape(4, 3, 2)
        xr.Dataset(
            {"value": (("time", "lat", "lon"), values)},
            coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(2)},
        ).to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={"value": {"chunks": (2, 2, 1)}},
        )
        metrics = run_rust_rechunk(
            RustRechunkPlan(
                source=Path(source),
                target=Path(target),
                array_path="/value",
                target_chunks=(1, 3, 2),
                expected_dtype="float64",
                requested_workers=2,
                worker_ceiling=2,
                memory_budget_bytes=1024 * 1024,
            )
        )
        self.assertEqual(metrics["logical_bytes"], values.nbytes)
    def test_auto_multi_variable_rechunk_uses_native_when_available(self) -> None:
        from fast_nc_zarr.application.services import RechunkConfig, run_rechunk as run_service_rechunk

        source = Path(self.tempdir.name) / "multi-source.zarr"
        target = Path(self.tempdir.name) / "multi-target.zarr"
        values = np.arange(2 * 3 * 4, dtype="float32").reshape(2, 3, 4)
        xr.Dataset(
            {
                "value": (("time", "lat", "lon"), values),
                "quality": (("time", "lat", "lon"), values + 100),
            },
            coords={"time": np.arange(2), "lat": np.arange(3), "lon": np.arange(4)},
        ).to_zarr(source, mode="w", consolidated=False, zarr_format=3)
        metrics = run_service_rechunk(
            RechunkConfig(
                input=source,
                output=target,
                target_mib=1,
                workers=1,
                backend="auto",
                compression="none",
            )
        )
        expected_backend = "rust" if _RUST_ZARR_READY else "python"
        self.assertEqual(metrics["backend"], expected_backend)
        self.assertEqual(bool(metrics["backend_fallback"]), expected_backend == "python")
        with xr.open_zarr(target, consolidated=False, chunks=None, decode_times=False) as dataset:
            np.testing.assert_array_equal(dataset["value"].values, values)
            np.testing.assert_array_equal(dataset["quality"].values, values + 100)

    def test_rust_multi_variable_rechunk_is_lossless_and_atomic(self) -> None:
        source = Path(self.tempdir.name) / "multi-mixed-source.zarr"
        target = Path(self.tempdir.name) / "multi-mixed-target.zarr"
        values = np.arange(4 * 3 * 2, dtype="float32").reshape(4, 3, 2)
        quality = (np.arange(4 * 3 * 2, dtype="float64") / 10).reshape(4, 3, 2)
        xr.Dataset(
            {
                "value": (("time", "lat", "lon"), values),
                "quality": (("time", "lat", "lon"), quality),
            },
            coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(2)},
        ).to_zarr(source, mode="w", consolidated=False, zarr_format=3)
        metrics = run_rust_multi_rechunk(
            RustMultiRechunkPlan(
                source=source,
                target=target,
                variables=(
                    RustMultiRechunkVariablePlan("/value", (1, 3, 2), "float32"),
                    RustMultiRechunkVariablePlan("/quality", (2, 1, 2), "float64"),
                ),
                requested_workers=2,
                worker_ceiling=2,
                memory_budget_bytes=1024 * 1024,
            )
        )
        self.assertEqual(metrics["backend"], "rust")
        self.assertEqual(len(metrics["variables"]), 2)
        self.assertEqual(metrics["output"], str(target))
        with xr.open_zarr(target, consolidated=False, chunks=None, decode_times=False) as dataset:
            self.assertEqual(dataset["value"].dtype, np.dtype("float32"))
            self.assertEqual(dataset["quality"].dtype, np.dtype("float64"))
            np.testing.assert_array_equal(dataset["value"].values, values)
            np.testing.assert_array_equal(dataset["quality"].values, quality)
        self.assertFalse(any(target.parent.glob(f".{target.name}.native-multi-*.tmp")))

    def test_rust_multi_variable_rechunk_pre_cancelled_does_not_publish(self) -> None:
        source = Path(self.tempdir.name) / "multi-cancel-source.zarr"
        target = Path(self.tempdir.name) / "multi-cancel-target.zarr"
        cancellation = Path(self.tempdir.name) / "multi-cancel.request"
        values = np.arange(2 * 2 * 2, dtype="float32").reshape(2, 2, 2)
        xr.Dataset({"value": (("time", "lat", "lon"), values)}).to_zarr(
            source, mode="w", consolidated=False, zarr_format=3
        )
        cancellation.touch()
        with self.assertRaises(Exception):
            run_rust_multi_rechunk(
                RustMultiRechunkPlan(
                    source=source,
                    target=target,
                    variables=(RustMultiRechunkVariablePlan("/value", (1, 2, 2), "float32"),),
                    cancellation_file=cancellation,
                )
            )
        self.assertFalse(target.exists())
        self.assertFalse(any(target.parent.glob(f".{target.name}.native-multi-*.tmp")))

    def test_rust_multi_integer_fill_cf_and_explicit_codec(self) -> None:
        source = Path(self.tempdir.name) / "multi-integer-source.zarr"
        target = Path(self.tempdir.name) / "multi-integer-target.zarr"
        values = np.array([[[1, -9999], [3, 4]], [[5, 6], [-9999, 8]]], dtype="int16")
        quality = np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype="uint32")
        xr.Dataset(
            {
                "value": (
                    ("time", "lat", "lon"),
                    values,
                    {
                        "units": "K",
                        "scale_factor": 0.1,
                        "add_offset": 273.15,
                        "standard_name": "air_temperature",
                    },
                ),
                "quality": (
                    ("time", "lat", "lon"),
                    quality,
                    {"long_name": "quality flag", "units": "1"},
                ),
            },
            coords={"time": np.arange(2), "lat": np.arange(2), "lon": np.arange(2)},
        ).to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={
                "value": {"chunks": (1, 2, 2), "_FillValue": -9999},
                "quality": {"chunks": (2, 1, 2), "_FillValue": 0},
            },
        )
        metrics = run_rust_multi_rechunk(
            RustMultiRechunkPlan(
                source=source,
                target=target,
                variables=(
                    RustMultiRechunkVariablePlan(
                        "/value", (2, 1, 2), "int16", dimension_names=("time", "lat", "lon")
                    ),
                    RustMultiRechunkVariablePlan(
                        "/quality", (1, 2, 2), "uint32", dimension_names=("time", "lat", "lon")
                    ),
                ),
                requested_workers=2,
                worker_ceiling=2,
                memory_budget_bytes=1024 * 1024,
                codec="zstd",
                codec_level=1,
            )
        )
        self.assertEqual(metrics["backend"], "rust")
        self.assertEqual(metrics["selected_compression"]["codec"], "zstd")
        with (
            xr.open_zarr(
                source, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
            ) as source_dataset,
            xr.open_zarr(
                target, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
            ) as dataset,
        ):
            self.assertEqual(dataset["value"].dtype, np.dtype("int16"))
            self.assertEqual(dataset["quality"].dtype, np.dtype("uint32"))
            np.testing.assert_array_equal(dataset["value"].values, values)
            np.testing.assert_array_equal(dataset["quality"].values, quality)
            for name in ("units", "scale_factor", "add_offset", "standard_name"):
                self.assertEqual(dataset["value"].attrs[name], source_dataset["value"].attrs[name])
        source_summary = json.loads(self.native.inspect_array_json(str(source), "/value"))
        target_summary = json.loads(self.native.inspect_array_json(str(target), "/value"))
        self.assertEqual(source_summary["fill_value"], target_summary["fill_value"])
        self.assertEqual(source_summary["attributes"], target_summary["attributes"])

    def test_rust_multi_rechunk_rejects_coordinate_array_plan(self) -> None:
        source = Path(self.tempdir.name) / "multi-coordinate-source.zarr"
        target = Path(self.tempdir.name) / "multi-coordinate-target.zarr"
        xr.Dataset(
            {"value": (("time", "lat", "lon"), np.zeros((2, 2, 2), dtype="float32"))},
            coords={"time": np.arange(2), "lat": np.arange(2), "lon": np.arange(2)},
        ).to_zarr(source, mode="w", consolidated=False, zarr_format=3)
        with self.assertRaises(Exception):
            run_rust_multi_rechunk(
                RustMultiRechunkPlan(
                    source=source,
                    target=target,
                    variables=(
                        RustMultiRechunkVariablePlan(
                            "/lat", (2,), "int64", is_coordinate=True, dimension_names=("lat",)
                        ),
                    ),
                )
            )
        self.assertFalse(target.exists())

    def test_rust_rechunk_uses_bounded_parallel_workers(self) -> None:
        source = f"{self.tempdir.name}/parallel-source.zarr"
        target = f"{self.tempdir.name}/parallel-target.zarr"
        values = np.arange(8 * 4 * 4, dtype="float32").reshape(8, 4, 4)
        dataset = xr.Dataset(
            {"value": (("time", "lat", "lon"), values)},
            coords={"time": np.arange(8), "lat": np.arange(4), "lon": np.arange(4)},
        )
        dataset.to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={"value": {"chunks": (2, 2, 2)}},
        )
        dataset.close()
        metrics = run_rust_rechunk(
            RustRechunkPlan(
                source=Path(source),
                target=Path(target),
                array_path="/value",
                target_chunks=(1, 2, 2),
                requested_workers=4,
                worker_ceiling=4,
                memory_budget_bytes=1024 * 1024,
                codec_concurrent_target=2,
            )
        )
        self.assertGreaterEqual(metrics["resolved_workers"], 1)
        self.assertLessEqual(metrics["resolved_workers"], 4)
        self.assertEqual(metrics["codec_concurrent_target"], 2)
        self.assertEqual(metrics["memory_budget_bytes"], 1024 * 1024)
        self.assertGreater(metrics["peak_bytes_per_worker"], 0)
        with xr.open_zarr(target, consolidated=False, chunks=None, decode_times=False) as result:
            np.testing.assert_array_equal(result["value"].values, values)
    def test_rust_codec_only_rechunk_is_lossless(self) -> None:
        source = f"{self.tempdir.name}/codec-source.zarr"
        target = f"{self.tempdir.name}/codec-target.zarr"
        values = np.arange(4 * 3 * 2, dtype="float32").reshape(4, 3, 2)
        xr.Dataset(
            {"value": (("time", "lat", "lon"), values)},
            coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(2)},
        ).to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={"value": {"chunks": (2, 2, 1)}},
        )
        metrics = run_rust_rechunk(
            RustRechunkPlan(
                source=Path(source),
                target=Path(target),
                array_path="/value",
                target_chunks=(2, 2, 1),
                codec="zstd",
                codec_level=1,
            )
        )
        self.assertEqual(metrics["execution_path"], "rust-codec-target-chunk")
        self.assertEqual(repr(zarr.open_array(target, path="value", mode="r").compressors[0]).split("(", 1)[0], "ZstdCodec")
        with xr.open_zarr(target, consolidated=False, chunks=None, decode_times=False) as result:
            np.testing.assert_array_equal(result["value"].values, values)

    def test_rust_progress_file_reaches_completion(self) -> None:
        source = f"{self.tempdir.name}/progress-source.zarr"
        target = f"{self.tempdir.name}/progress-target.zarr"
        progress = Path(self.tempdir.name) / "progress.json"
        values = np.ones((2, 2, 2), dtype="float32")
        xr.Dataset({"value": (("time", "lat", "lon"), values)}).to_zarr(
            source, mode="w", consolidated=False, zarr_format=3
        )
        run_rust_rechunk(
            RustRechunkPlan(
                source=Path(source),
                target=Path(target),
                array_path="/value",
                target_chunks=(1, 2, 2),
                progress_file=progress,
            )
        )
        self.assertEqual(json.loads(progress.read_text())["total"], 2)
        with xr.open_zarr(target, consolidated=False, chunks=None, decode_times=False) as result:
            np.testing.assert_array_equal(result["value"].values, values)
            self.assertEqual(result["value"].encoding["chunks"], (1, 2, 2))
        progress.unlink()
        np.testing.assert_array_equal(values, np.ones((2, 2, 2), dtype="float32"))

    def test_rust_rechunk_pre_cancelled_does_not_publish(self) -> None:
        import threading

        source = f"{self.tempdir.name}/cancel-source.zarr"
        target = f"{self.tempdir.name}/cancel-target.zarr"
        xr.Dataset(
            {"value": (("time", "lat", "lon"), np.zeros((2, 2, 2), dtype="float32"))}
        ).to_zarr(source, mode="w", consolidated=False, zarr_format=3)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaisesRegex(RuntimeError, "任务已取消"):
            run_rust_rechunk(
                RustRechunkPlan(
                    source=Path(source),
                    target=Path(target),
                    array_path="/value",
                    target_chunks=(1, 2, 2),
                ),
                cancel_event=cancelled,
            )
        self.assertFalse(Path(target).exists())

    def test_rust_rechunk_changes_chunks_without_changing_values(self) -> None:
        source = f"{self.tempdir.name}/rechunk-source.zarr"
        target = f"{self.tempdir.name}/rechunk-target.zarr"
        values = np.arange(4 * 3 * 2, dtype="float32").reshape(4, 3, 2)
        dataset = xr.Dataset(
            {"value": (("time", "lat", "lon"), values)},
            coords={"time": np.arange(4), "lat": np.arange(3), "lon": np.arange(2)},
            attrs={"title": "rust rechunk"},
        )
        dataset.to_zarr(
            source,
            mode="w",
            consolidated=False,
            zarr_format=3,
            encoding={"value": {"chunks": (2, 2, 1)}},
        )
        dataset.close()
        metrics = run_rust_rechunk(
            RustRechunkPlan(
                source=Path(source),
                target=Path(target),
                array_path="/value",
                target_chunks=(1, 3, 2),
            )
        )
        self.assertGreaterEqual(metrics["resolved_workers"], 1)
        self.assertEqual(metrics["output"], target)
        with xr.open_zarr(target, consolidated=False, chunks=None, decode_times=False) as result:
            np.testing.assert_array_equal(result["value"].values, values)
            np.testing.assert_array_equal(result["time"].values, np.arange(4))
            self.assertEqual(result.attrs["title"], "rust rechunk")
        group = zarr.open_group(target, mode="r")
        self.assertEqual(group["value"].chunks, (1, 3, 2))

    def test_rust_rechunk_rejects_non_float32_source(self) -> None:
        source = f"{self.tempdir.name}/rechunk-int-source.zarr"
        target = f"{self.tempdir.name}/rechunk-int-target.zarr"
        dataset = xr.Dataset(
            {"value": (("time", "lat", "lon"), np.zeros((2, 2, 2), dtype="int16"))},
            coords={"time": [0, 1], "lat": [0, 1], "lon": [0, 1]},
        )
        dataset.to_zarr(source, mode="w", consolidated=False, zarr_format=3)
        dataset.close()
        with self.assertRaisesRegex(RuntimeError, "float32"):
            run_rust_rechunk(
                RustRechunkPlan(
                    source=Path(source),
                    target=Path(target),
                    array_path="/value",
                    target_chunks=(1, 2, 2),
                )
            )

@unittest.skipUnless(_RUST_RESAMPLE_READY, "Rust resampling extension is not built")
class RustResamplingTests(unittest.TestCase):
    def test_nearest_and_bilinear_regular_grid(self) -> None:
        native = importlib.import_module("fast_nc_zarr._native")
        request = {
            "values": [0.0, 1.0, 2.0, 3.0],
            "shape": [1, 2, 2],
            "source_lat": [0.0, 1.0],
            "source_lon": [0.0, 1.0],
            "target_lat": [0.5],
            "target_lon": [0.5],
            "method": "bilinear",
        }
        bilinear = json.loads(native.resample_f32_json(json.dumps(request)))
        self.assertEqual(bilinear["shape"], [1, 1, 1])
        self.assertAlmostEqual(bilinear["values"][0], 1.5)
        request["method"] = "nearest"
        nearest = json.loads(native.resample_f32_json(json.dumps(request)))
        self.assertEqual(nearest["values"], [0.0])

    def test_typed_buffer_missing_values_respect_skipna_threshold(self) -> None:
        native = importlib.import_module("fast_nc_zarr._native")
        values = np.asarray([np.nan, 2.0, 4.0, 6.0], dtype="float32").reshape(1, 2, 2)
        kwargs = (
            values,
            list(values.shape),
            np.asarray([0.0, 1.0], dtype="float32"),
            np.asarray([0.0, 1.0], dtype="float32"),
            np.asarray([0.5], dtype="float32"),
            np.asarray([0.5], dtype="float32"),
            "bilinear",
        )
        raw_values, shape = native.resample_f32_buffer_options(*kwargs, True, 1.0)
        result = np.frombuffer(raw_values, dtype="float32").reshape(shape)
        self.assertAlmostEqual(float(result[0, 0, 0]), 4.0)
        raw_values, shape = native.resample_f32_buffer_options(*kwargs, True, 0.0)
        strict = np.frombuffer(raw_values, dtype="float32").reshape(shape)
        self.assertTrue(np.isnan(strict[0, 0, 0]))

    def test_typed_buffer_resampling_matches_json_contract(self) -> None:
        native = importlib.import_module("fast_nc_zarr._native")
        values = np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float32").reshape(1, 2, 2)
        raw_values, shape = native.resample_f32_buffer(
            values,
            list(values.shape),
            np.asarray([0.0, 1.0], dtype="float32"),
            np.asarray([0.0, 1.0], dtype="float32"),
            np.asarray([0.5], dtype="float32"),
            np.asarray([0.5], dtype="float32"),
            "bilinear",
        )
        result = np.frombuffer(raw_values, dtype="float32").reshape(shape)
        self.assertEqual(tuple(shape), (1, 1, 1))
        self.assertFalse(result.flags.writeable)
        self.assertAlmostEqual(float(result[0, 0, 0]), 1.5)
    def test_writable_typed_buffer_resampling_fills_output(self) -> None:
        native = importlib.import_module("fast_nc_zarr._native")
        values = np.asarray([0.0, 1.0, 2.0, 3.0], dtype="float32").reshape(1, 2, 2)
        output = np.empty((1, 1, 1), dtype="float32")
        shape = native.resample_f32_buffer_into(
            values,
            list(values.shape),
            np.asarray([0.0, 1.0], dtype="float32"),
            np.asarray([0.0, 1.0], dtype="float32"),
            np.asarray([0.5], dtype="float32"),
            np.asarray([0.5], dtype="float32"),
            "bilinear",
            output,
        )
        self.assertEqual(tuple(shape), (1, 1, 1))
        self.assertAlmostEqual(float(output[0, 0, 0]), 1.5)

if __name__ == "__main__":
    unittest.main()
