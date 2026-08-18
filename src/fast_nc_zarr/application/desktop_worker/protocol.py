from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, TextIO

PROTOCOL_VERSION = 1
MAX_MESSAGE_BYTES = 1_048_576
MAX_JSON_DEPTH = 32
MAX_COLLECTION_ITEMS = 100_000
COMMANDS = frozenset(
    {
        "get_capabilities",
        "inspect_source",
        "inspect_zarr",
        "inspect_time_metadata",
        "save_inspection_snapshot",
        "preview_pipeline",
        "run_pipeline",
        "resume_pipeline",
        "shutdown",
    }
)
EVENTS = frozenset(
    {
        "accepted",
        "started",
        "inspection_ready",
        "plan_ready",
        "progress",
        "resource",
        "log",
        "finished",
        "failed",
        "cancelled",
    }
)
TERMINAL_EVENTS = frozenset({"finished", "failed", "cancelled"})
ERROR_KINDS = frozenset(
    {
        "invalid_request",
        "path_not_found",
        "permission_denied",
        "input_invalid",
        "backend_unavailable",
        "worker_start_failed",
        "worker_protocol_error",
        "cancelled",
        "resource_budget_exceeded",
        "validation_failed",
        "publication_failed",
        "unknown",
    }
)


class ProtocolError(ValueError):
    """Raised when a wire message violates protocol v1."""


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    command: str
    payload: dict[str, Any]
    task_id: str | None = None


@dataclass(frozen=True, slots=True)
class Event:
    request_id: str
    sequence: int
    event: str
    payload: dict[str, Any]
    task_id: str | None = None
    stage: str | None = None
def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(f"{name} must be an object")
    return value


def _validate_limits(value: Any, *, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise ProtocolError("JSON nesting depth exceeds protocol limit")
    if isinstance(value, dict):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ProtocolError("JSON object has too many fields")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ProtocolError("JSON object keys must be strings")
            _validate_limits(item, depth=depth + 1)
    elif isinstance(value, list):
        if len(value) > MAX_COLLECTION_ITEMS:
            raise ProtocolError("JSON array has too many items")
        for item in value:
            _validate_limits(item, depth=depth + 1)


def _decode_json(value: str | bytes) -> Any:
    if len(value) > MAX_MESSAGE_BYTES:
        raise ProtocolError("JSON message exceeds protocol byte limit")
    parsed = json.loads(value)
    _validate_limits(parsed)
    return parsed


def decode_request(value: str | bytes | dict[str, Any]) -> Request:
    raw = _object(_decode_json(value) if isinstance(value, (str, bytes)) else value, name="request")
    _validate_limits(raw)
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported request protocol version")
    request_id = raw.get("request_id")
    command = raw.get("command")
    payload = raw.get("payload")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("request_id must be a non-empty string")
    if command not in COMMANDS:
        raise ProtocolError(f"unsupported command: {command!r}")
    return Request(request_id, command, _object(payload, name="payload"), raw.get("task_id"))


def decode_event(value: str | bytes | dict[str, Any]) -> Event:
    raw = _object(json.loads(value) if isinstance(value, (str, bytes)) else value, name="event")
    if raw.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("unsupported event protocol version")
    request_id = raw.get("request_id")
    sequence = raw.get("sequence")
    event = raw.get("event")
    payload = raw.get("payload")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("request_id must be a non-empty string")
    if not isinstance(sequence, int) or sequence < 0:
        raise ProtocolError("sequence must be a non-negative integer")
    if event not in EVENTS:
        raise ProtocolError(f"unsupported event: {event!r}")
    return Event(request_id, sequence, event, _object(payload, name="payload"), raw.get("task_id"), raw.get("stage"))


def encode_event(event: Event) -> str:
    return json.dumps(
        {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": event.request_id,
            "task_id": event.task_id,
            "sequence": event.sequence,
            "event": event.event,
            "stage": event.stage,
            "payload": event.payload,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


def write_event(stream: TextIO, event: Event) -> None:
    stream.write(encode_event(event) + "\n")
    stream.flush()


def error_payload(
    kind: str,
    message: str,
    *,
    retryable: bool = False,
    stage: str | None = None,
    details: dict[str, Any] | None = None,
    manifest: str | None = None,
) -> dict[str, Any]:
    if kind not in ERROR_KINDS:
        kind = "unknown"
    return {
        "kind": kind,
        "message": message,
        "retryable": retryable,
        "stage": stage,
        "details": details or {},
        "manifest": manifest,
    }
