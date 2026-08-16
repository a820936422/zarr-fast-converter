from __future__ import annotations

import tomllib
import json
import os
import math
import tempfile
import numpy as np
import xarray as xr
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from fast_nc_zarr.application.desktop_worker.worker import _pipeline_config, _safe  # noqa: E402


class ProtocolContractTests(unittest.TestCase):
    def test_request_event_error_and_capability_schemas_are_valid_json(self) -> None:
        names = (
            "request-v1.schema.json",
            "event-v1.schema.json",
            "error-v1.schema.json",
            "capability-v1.schema.json",
        )
        for name in names:
            payload = json.loads((PROJECT / "contracts" / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["$schema"], "https://json-schema.org/draft/2020-12/schema")
        for name in ("request-v1.schema.json", "event-v1.schema.json"):
            payload = json.loads((PROJECT / "contracts" / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["properties"]["protocol_version"]["const"], 1)

    def test_capability_fixture_has_consistent_supported_operations(self) -> None:
        payload = json.loads(
            (PROJECT / "contracts/fixtures/capability-v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["protocol_version"], 1)
        self.assertEqual(
            set(payload["operations"]),
            {item["operation"] for item in payload["capabilities"] if item["supported"]},
        )
        convert = next(
            item for item in payload["capabilities"] if item["operation"] == "raw.netcdf.convert"
        )
        self.assertTrue(convert["supported"])
        self.assertIsNone(convert["reason"])
        inspect = next(
            item for item in payload["capabilities"] if item["operation"] == "raw.netcdf.inspect"
        )
        self.assertTrue(inspect["supported"])
        self.assertIsNone(inspect["reason"])

    def test_request_and_event_schemas_keep_protocol_version_one(self) -> None:
        for name in ("request-v1.schema.json", "event-v1.schema.json"):
            payload = json.loads((PROJECT / "contracts" / name).read_text(encoding="utf-8"))
            self.assertEqual(payload["properties"]["protocol_version"]["const"], 1)

    def test_worker_capability_fixture_matches_contract(self) -> None:
        request = (PROJECT / "contracts/fixtures/get-capabilities.request.json").read_text(
            encoding="utf-8"
        )
        expected = [
            line.strip()
            for line in (
                PROJECT / "contracts/fixtures/get-capabilities.events.jsonl"
            ).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result = subprocess.run(
            [sys.executable, "-m", "fast_nc_zarr.application.desktop_worker"],
            cwd=PROJECT,
            input=request,
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
        )
        actual = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        self.assertEqual([json.loads(item)["event"] for item in actual], ["accepted", "started", "finished"])
        self.assertEqual(json.loads(actual[-1])["payload"]["backend"], "python")
        self.assertEqual(len(expected), 2)
    def test_project_metadata_declares_runtime_dependencies(self) -> None:
        metadata = tomllib.loads((PROJECT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = set(metadata["project"]["dependencies"])
        for requirement in (
            "dask>=2026.7.1,<2027",
            "h5netcdf>=1.8.1,<2",
            "netCDF4>=1.7.4,<2",
            "numpy>=2.4.6,<3",
            "rioxarray>=0.23,<0.24",
            "xarray>=2026.7,<2027",
            "zarr>=3.3,<4",
        ):
            self.assertIn(requirement, dependencies)
        self.assertEqual(
            metadata["project"]["optional-dependencies"]["resampling"],
            ["xesmf>=0.9.2,<0.10"],
        )


class DesktopWorkerTests(unittest.TestCase):
    def test_worker_safe_serializes_multidimensional_arrays(self) -> None:
        self.assertEqual(_safe(np.arange(4, dtype="int32").reshape(2, 2)), [[0, 1], [2, 3]])

    def test_pipeline_request_forwards_rust_resource_snapshot(self) -> None:
        request = {
            "protocol_version": 1,
            "request_id": "resource-request",
            "task_id": "resource-task",
            "command": "run_pipeline",
            "payload": {
                "resource_snapshot": {
                    "capturedAtMs": 1,
                    "logicalCpus": 2,
                    "memoryTotalBytes": 3,
                    "memoryAvailableBytes": 4,
                }
            },
        }
        result = subprocess.run(
            [sys.executable, "-m", "fast_nc_zarr.application.desktop_worker"],
            cwd=PROJECT,
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
        )
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual(events[2]["event"], "resource")
        self.assertEqual(events[2]["payload"]["logicalCpus"], 2)
        self.assertEqual(events[-1]["event"], "failed")

    def test_pipeline_payload_preserves_variable_and_resampling_rules(self) -> None:
        config = _pipeline_config(
            {
                "output": "/tmp/restored-controls.zarr",
                "temporary_dir": "/tmp/processing",
                "time_start": "2001-01-01",
                "time_end": "2001-01-08",
                "lat_min": 0.1,
                "lat_max": 0.3,
                "lon_min": -0.1,
                "lon_max": 0.1,
                "input_kind": "raw",
                "variables": ["value"],
                "variable_names": {"value": "renamed_value"},
                "variable_transforms": {
                    "value": {
                        "fill_values": [-999, "nan"],
                        "scale_factor": 2,
                        "add_offset": 1,
                        "output_fill": -7,
                    }
                },
                "resample": True,
                "skipna": False,
                "na_thres": 0.25,
                "compute_dtype": "float32",
                "before_conditions": "<0",
                "before_results": "0",
                "after_conditions": ">100",
                "after_results": "100",
                "compression": "balanced",
                "compression_codec": "zstd",
                "compression_level": 5,
                "compression_shuffle": "noshuffle",
                "compression_objective": "compact",
                "compression_tune_budget": 30,
                "strategy": "custom",
                "custom_chunks": [4, 128, 256],
            }
        )
        transform = config.conversion.variable_transforms["value"]
        self.assertEqual(config.conversion.variable_names["value"], "renamed_value")
        self.assertEqual(transform.fill_values[0], -999)
        self.assertTrue(math.isnan(transform.fill_values[1]))
        self.assertEqual(transform.scale_factor, 2)
        self.assertEqual(config.resampling.compute_dtype, "float32")
        self.assertFalse(config.resampling.skipna)
        self.assertEqual(config.resampling.before_conditions, "<0")
        self.assertEqual(config.resampling.after_results, "100")
        self.assertEqual(config.general.temporary_dir, Path("/tmp/processing"))
        self.assertEqual(config.chunking.strategy, "custom")
        self.assertEqual(config.chunking.custom_chunks, (4, 128, 256))
        self.assertEqual(config.general.time_start, "2001-01-01")
        self.assertEqual(config.general.time_end, "2001-01-08")
        self.assertEqual(config.general.lat_min, 0.1)
        self.assertEqual(config.general.lat_max, 0.3)
        self.assertEqual(config.general.lon_min, -0.1)
        self.assertEqual(config.general.lon_max, 0.1)
        self.assertEqual(config.compression.profile, "balanced")
        self.assertEqual(config.compression.codec, "zstd")
        self.assertEqual(config.compression.level, 5)
        self.assertEqual(config.compression.shuffle, "noshuffle")
        self.assertEqual(config.compression.objective, "compact")
        self.assertEqual(config.compression.tune_budget, 30)

    def test_pipeline_payload_parses_variable_resampling_overrides(self) -> None:
        config = _pipeline_config(
            {
                "output": "/tmp/variable-resampling.zarr",
                "variable_resampling": {
                    "a1": {"method": "conservative", "skipna": True, "na_thres": 0.75},
                    "a2": {"method": "bilinear", "skipna": False, "na_thres": 0.25, "compute_dtype": "float32"},
                },
            }
        )
        self.assertEqual(config.resampling.variable_options["a1"].method, "conservative")
        self.assertTrue(config.resampling.variable_options["a1"].skipna)
        self.assertEqual(config.resampling.variable_options["a2"].method, "bilinear")
        self.assertFalse(config.resampling.variable_options["a2"].skipna)
        self.assertEqual(config.resampling.variable_options["a2"].compute_dtype, "float32")

    def test_run_pipeline_publishes_output_after_worker_launch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="desktop-pipeline-") as raw:
            root = Path(raw)
            source = root / "source"
            output = root / "result.zarr"
            source.mkdir()
            dataset = xr.Dataset(
                {"value": (("time", "lat", "lon"), np.ones((1, 2, 2), dtype="float32"))},
                coords={
                    "time": np.asarray(["2001-01-01"], dtype="datetime64[ns]"),
                    "lat": [10.0, 20.0],
                    "lon": [30.0, 40.0],
                },
            )
            dataset.to_netcdf(source / "sample.nc", engine="h5netcdf")
            dataset.close()
            environment = {**os.environ, "PYTHONPATH": str(PROJECT / "src")}

            inspect = subprocess.run(
                [sys.executable, "-m", "fast_nc_zarr.application.desktop_worker"],
                cwd=PROJECT,
                input=json.dumps(
                    {
                        "protocol_version": 1,
                        "request_id": "inspect-launch",
                        "command": "inspect_source",
                        "payload": {
                            "input_dir": str(source),
                            "mode": "complete",
                            "engine": "h5netcdf",
                            "validation_mode": "fast",
                        },
                    }
                )
                + "\n",
                text=True,
                capture_output=True,
                check=True,
                env=environment,
            )
            inspection_events = [json.loads(line) for line in inspect.stdout.splitlines() if line.strip()]
            snapshot = inspection_events[-1]["payload"]["inspection_snapshot_path"]

            run = subprocess.run(
                [sys.executable, "-m", "fast_nc_zarr.application.desktop_worker"],
                cwd=PROJECT,
                input=json.dumps(
                    {
                        "protocol_version": 1,
                        "request_id": "run-launch",
                        "task_id": "run-launch-task",
                        "command": "run_pipeline",
                        "payload": {
                            "input_dir": str(source),
                            "input_kind": "raw",
                            "inspection_kind": "source",
                            "inspection_snapshot_path": snapshot,
                            "validate_snapshot": False,
                            "variables": ["value"],
                            "output": str(output),
                            "temporary_dir": str(root / "temporary"),
                            "backend": "auto",
                            "validate": True,
                            "resample": False,
                            "rechunk": False,
                            "recompress": False,
                        },
                    }
                )
                + "\n",
                text=True,
                capture_output=True,
                check=True,
                env=environment,
            )
            run_events = [json.loads(line) for line in run.stdout.splitlines() if line.strip()]
            self.assertEqual(run_events[-1]["event"], "finished")
            progress_events = [event for event in run_events if event["event"] == "progress"]
            self.assertTrue(progress_events)
            completed = [event for event in progress_events if event["payload"].get("status") == "completed"]
            self.assertTrue(completed)
            self.assertTrue(all("logical_bytes" in event["payload"] for event in progress_events))
            self.assertTrue(all("temporary_bytes" in event["payload"] for event in progress_events))
            self.assertTrue((output / "zarr.json").is_file())

    def test_invalid_request_returns_structured_failure(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "fast_nc_zarr.application.desktop_worker"],
            cwd=PROJECT,
            input='{"protocol_version":99,"request_id":"bad","command":"nope","payload":{}}\n',
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
        )
        event = json.loads(result.stdout)
        self.assertEqual(event["event"], "failed")
        self.assertEqual(event["payload"]["error"]["kind"], "worker_protocol_error")


if __name__ == "__main__":
    unittest.main()
