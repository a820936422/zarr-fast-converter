from __future__ import annotations

import json
from pathlib import Path
import os
import subprocess
import sys
import unittest

PROJECT = Path(__file__).resolve().parents[1]


class ProtocolTests(unittest.TestCase):
    def test_fixture_request_produces_terminal_event(self) -> None:
        request = (PROJECT / "contracts/fixtures/get-capabilities.request.json").read_text()
        result = subprocess.run(
            [sys.executable, "-m", "fast_nc_zarr.application.desktop_worker"],
            cwd=PROJECT,
            input=request + "\n",
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "PYTHONPATH": str(PROJECT / "src")},
        )
        events = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertEqual([event["event"] for event in events], ["accepted", "started", "finished"])
        self.assertEqual(events[-1]["payload"]["backend"], "python")
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
