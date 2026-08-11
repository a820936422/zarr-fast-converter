from __future__ import annotations

import importlib
import json
import tempfile
import unittest

import numpy as np
import xarray as xr

from fast_nc_zarr._backend import resolve_backend, rust_capability


_CAPABILITY = rust_capability()
_RUST_ZARR_READY = _CAPABILITY.supported and {
    "zarr.inspect",
    "zarr.read_chunk_f32",
    "zarr.read_region_f32",
    "zarr.write_f32",
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

    def test_auto_backend_falls_back_without_rust_operation(self) -> None:
        self.assertEqual(resolve_backend("auto", "rechunk"), "python")

    def test_python_backend_is_always_selectable(self) -> None:
        self.assertEqual(resolve_backend("python", "rechunk"), "python")


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


if __name__ == "__main__":
    unittest.main()
