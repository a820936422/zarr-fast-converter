from __future__ import annotations

import json
from pathlib import Path
import os
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
        unavailable = next(
            item for item in payload["capabilities"] if item["operation"] == "raw.netcdf.convert"
        )
        self.assertFalse(unavailable["supported"])
        self.assertTrue(unavailable["reason"])

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
        request = (PROJECT / "contracts/fixtures/get-capabilities.request.json").read_text(encoding="utf-8")
        expected = [
            line.strip()
            for line in (PROJECT / "contracts/fixtures/get-capabilities.events.jsonl").read_text(encoding="utf-8").splitlines()
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
        self.assertEqual(len(actual), 3)
        self.assertEqual(
            [json.loads(item)["event"] for item in actual],
            ["accepted", "started", "finished"],
        )
        self.assertEqual(json.loads(actual[-1])["payload"]["backend"], "python")
        self.assertEqual(len(expected), 2)


if __name__ == "__main__":
    unittest.main()
