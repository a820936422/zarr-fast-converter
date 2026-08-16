#!/usr/bin/env python3
"""Validate checked-in IPC schemas and protocol fixtures."""

from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "contracts"


def validate(schema_name: str, instance: object, label: str) -> None:
    schema = json.loads((CONTRACTS / schema_name).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.path))
    if errors:
        details = "; ".join(
            f"{label}{list(error.path)}: {error.message}" for error in errors
        )
        raise SystemExit(details)


def main() -> int:
    request = json.loads(
        (CONTRACTS / "fixtures/get-capabilities.request.json").read_text(encoding="utf-8")
    )
    validate("request-v1.schema.json", request, "request")

    event_lines = [
        line
        for line in (CONTRACTS / "fixtures/get-capabilities.events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    for index, line in enumerate(event_lines):
        validate("event-v1.schema.json", json.loads(line), f"event[{index}]")

    capability = json.loads(
        (CONTRACTS / "fixtures/capability-v1.json").read_text(encoding="utf-8")
    )
    validate("capability-v1.schema.json", capability, "capability")
    validate(
        "error-v1.schema.json",
        {
            "kind": "worker_protocol_error",
            "message": "invalid JSONL event",
            "retryable": False,
            "stage": "transport",
            "details": {},
            "manifest": None,
        },
        "error",
    )
    print(f"contract schema validation passed: {len(event_lines)} event fixture(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
