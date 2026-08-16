from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import numpy as np

from fast_nc_zarr.engine import convert
from fast_nc_zarr.models import FileRecord, Inventory, Selection, VariableSpec
from fast_nc_zarr.publication import (
    preflight_writable,
    publish_staging,
    validate_publish_target,
)
from fast_nc_zarr.runtime import (
    ProcessLifecycle,
    bounded_process_map,
    configure_process_runtime,
    shutdown_process_executor,
    spawn_context,
)


ROOT = Path("/tmp/codex_test/fast_nc_zarr_runtime_publication_tests")


def _zarr_marker(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "zarr.json").write_text(
        json.dumps({"zarr_format": 3, "node_type": "group"}),
        encoding="utf-8",
    )



def _raise_from_worker(value):
    raise RuntimeError(f"worker failure: {value}")

class RuntimePublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)
        ROOT.mkdir(parents=True)

    def tearDown(self) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_spawn_context_applies_one_native_thread_budget(self) -> None:
        with patch.dict(os.environ, {"FAST_NC_ZARR_THREADS_PER_WORKER": "2"}):
            self.assertEqual(configure_process_runtime(), 2)
            self.assertEqual(spawn_context().get_start_method(), "spawn")
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                self.assertEqual(os.environ[name], "2")

    def test_bounded_process_map_uses_direct_serial_path(self) -> None:
        state = []

        def initialize(prefix):
            state.append(prefix)

        results = list(
            bounded_process_map(
                lambda value: value * 2,
                range(4),
                workers=1,
                initializer=initialize,
                initargs=("ready",),
            )
        )

        self.assertEqual(state, ["ready"])
        self.assertEqual(results, [0, 2, 4, 6])


    def test_process_lifecycle_records_terminal_state(self) -> None:
        lifecycle = ProcessLifecycle("test")
        lifecycle.finish("completed")
        payload = lifecycle.to_dict()

        self.assertEqual(payload["label"], "test")
        self.assertEqual(payload["parent_pid"], os.getpid())
        self.assertEqual(payload["exit_reason"], "completed")
        self.assertIsNotNone(payload["ended_at"])
        self.assertLessEqual(payload["started_at"], payload["ended_at"])
        self.assertIsInstance(payload["child_pids"], list)
        self.assertIsInstance(payload["active_child_pids"], list)

    def test_bounded_process_map_terminates_failed_workers(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "worker failure"):
            list(
                bounded_process_map(
                    _raise_from_worker,
                    range(4),
                    workers=2,
                )
            )
    def test_executor_fallback_terminates_private_processes(self) -> None:
        class Process:
            def __init__(self) -> None:
                self.terminated = False

            def terminate(self) -> None:
                self.terminated = True

        class Executor:
            def __init__(self) -> None:
                self.process = Process()
                self._processes = {1: self.process}
                self.shutdown_called = False

            def shutdown(self, *, wait, cancel_futures) -> None:
                self.shutdown_called = bool(wait and cancel_futures)

        executor = Executor()
        shutdown_process_executor(executor, terminate=True)
        self.assertTrue(executor.process.terminated)
        self.assertTrue(executor.shutdown_called)
    def test_publish_replaces_validated_target_and_removes_backup(self) -> None:

        target = ROOT / "output.zarr"
        staging = ROOT / ".output.zarr.test.tmp"
        _zarr_marker(target)
        (target / "old").write_text("old", encoding="utf-8")
        _zarr_marker(staging)
        (staging / "new").write_text("new", encoding="utf-8")

        publish_staging(staging, target, "test")

        self.assertFalse(staging.exists())
        self.assertTrue((target / "new").is_file())
        self.assertFalse((target / "old").exists())
        self.assertFalse(list(ROOT.glob(".output.zarr.test-backup-*")))

    def test_publish_failure_restores_existing_target(self) -> None:
        target = ROOT / "output.zarr"
        staging = ROOT / ".output.zarr.test.tmp"
        _zarr_marker(target)
        (target / "old").write_text("old", encoding="utf-8")
        _zarr_marker(staging)
        original_replace = os.replace
        calls = 0

        def fail_new_store(source, destination):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("simulated publication failure")
            return original_replace(source, destination)

        with patch("fast_nc_zarr.publication.os.replace", side_effect=fail_new_store):
            with self.assertRaisesRegex(OSError, "simulated"):
                publish_staging(staging, target, "test")

        self.assertTrue((target / "old").is_file())
        self.assertTrue(staging.is_dir())

    def test_target_validation_rejects_symlink_and_plain_directory(self) -> None:
        plain = ROOT / "plain"
        plain.mkdir()
        (plain / "user.txt").write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "普通非空目录"):
            validate_publish_target(plain, overwrite=True, operation="测试")

        link = ROOT / "linked-output"
        link.symlink_to(plain, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "符号链接"):
            validate_publish_target(link, overwrite=True, operation="测试")
    def test_preflight_probes_nested_directory_and_cleans_probe(self) -> None:
        requested = ROOT / "new" / "nested"
        result = preflight_writable(requested, "测试")
        self.assertTrue(result["writable"])
        self.assertTrue(requested.is_dir())
        self.assertEqual(list(requested.glob("*.probe")), [])

    def test_preflight_rejects_file_as_directory(self) -> None:
        file_path = ROOT / "not-a-directory"
        file_path.write_text("x", encoding="utf-8")
        with self.assertRaises(NotADirectoryError):
            preflight_writable(file_path / "child", "测试")

    def test_conversion_failure_keeps_existing_store_untouched(self) -> None:
        input_dir = ROOT / "input"
        input_dir.mkdir()
        source = input_dir / "day.nc"
        source.touch()
        spec = VariableSpec("value", ("time", "lat", "lon"), "float32", (2, 2), None)
        record = FileRecord(
            source,
            1,
            (np.datetime64("2001-01-01", "ns"),),
            ("2001-01-01",),
            "lat",
            "lon",
            2,
            2,
            (spec,),
        )
        inventory = Inventory(
            input_dir=input_dir,
            files=[record],
            lat_values=np.asarray([1.0, 0.0]),
            lon_values=np.asarray([0.0, 1.0]),
            times=np.asarray([np.datetime64("2001-01-01", "ns")]),
            time_keys=("2001-01-01",),
            variables={"value": spec},
            source_engine="h5netcdf",
            source_dimensions=("time", "lat", "lon"),
            frequency="daily",
            gaps=[],
            total_bytes=1,
        )
        selection = Selection(("value",), 0, 1, 0, 2, 0, 2)
        target = ROOT / "existing.zarr"
        _zarr_marker(target)
        (target / "sentinel").write_text("original", encoding="utf-8")

        def fail_write(_inventory, _selection, staging, *_args, **_kwargs):
            _zarr_marker(Path(staging))
            raise RuntimeError("simulated write failure")

        with patch("fast_nc_zarr.engine.direct_write", side_effect=fail_write):
            with self.assertRaisesRegex(RuntimeError, "simulated write failure"):
                convert(
                    inventory,
                    selection,
                    target,
                    auto_tune=False,
                    overwrite=True,
                    validate=False,
                    progress=False,
                )

        self.assertEqual((target / "sentinel").read_text(encoding="utf-8"), "original")
        self.assertFalse(list(ROOT.glob(".existing.zarr.convert-*.tmp")))


if __name__ == "__main__":
    unittest.main()
