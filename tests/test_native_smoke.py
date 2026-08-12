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
from fast_nc_zarr.rechunking.native import RustRechunkPlan, run_rust_rechunk


_CAPABILITY = rust_capability()
_RUST_ZARR_READY = _CAPABILITY.supported and {
    "zarr.inspect",
    "zarr.read_chunk_f32",
    "zarr.read_region_f32",
    "zarr.write_f32",
    "zarr.rechunk_f32",
    "zarr.rechunk_f32_codec",
    "zarr.rechunk_f32_cancel",
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
        expected = "rust" if _RUST_ZARR_READY else "python"
        self.assertEqual(resolve_backend("auto", "rechunk"), expected)

    def test_python_backend_is_always_selectable(self) -> None:
        self.assertEqual(resolve_backend("python", "rechunk"), "python")

    def test_auto_backend_resolves_native_rechunk_capability(self) -> None:
        expected = "rust" if _RUST_ZARR_READY else "python"
        self.assertEqual(resolve_backend("auto", "zarr.rechunk_f32"), expected)


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
        metrics = run_rust_rechunk(
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

if __name__ == "__main__":
    unittest.main()
