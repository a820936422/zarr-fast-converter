from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


PROJECT = Path(__file__).resolve().parents[1]


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


class DesktopWorkerTests(unittest.TestCase):
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
