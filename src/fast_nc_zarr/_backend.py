from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from typing import Literal


BackendName = Literal["auto", "python", "rust"]


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested native backend is unavailable."""


@dataclass(frozen=True, slots=True)
class OperationCapability:
    operation: str
    supported: bool
    reason: str | None = None
    limitations: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class BackendCapability:
    name: str
    protocol_version: int
    crate_version: str | None
    operations: tuple[str, ...]
    supported: bool
    reason: str | None = None
    capabilities: tuple[OperationCapability, ...] = ()

    def operation(self, operation: str) -> OperationCapability | None:
        return next(
            (item for item in self.capabilities if item.operation == operation),
            None,
        )


def _fallback_capabilities(operations: tuple[str, ...]) -> tuple[OperationCapability, ...]:
    return tuple(OperationCapability(operation, True) for operation in operations)


def _parse_capabilities(
    payload: dict[str, object], operations: tuple[str, ...]
) -> tuple[OperationCapability, ...]:
    raw = payload.get("capabilities")
    if raw is None:
        return _fallback_capabilities(operations)
    if not isinstance(raw, list):
        raise ValueError("capabilities must be a list")
    parsed: list[OperationCapability] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("each capability must be an object")
        operation = item.get("operation")
        supported = item.get("supported")
        reason = item.get("reason")
        limitations = item.get("limitations", [])
        if not isinstance(operation, str) or not operation:
            raise ValueError("capability operation must be a non-empty string")
        if operation in seen:
            raise ValueError(f"duplicate capability operation: {operation}")
        if not isinstance(supported, bool):
            raise ValueError(f"capability supported must be boolean: {operation}")
        if reason is not None and not isinstance(reason, str):
            raise ValueError(f"capability reason must be string or null: {operation}")
        if not isinstance(limitations, list) or not all(
            isinstance(value, str) and value for value in limitations
        ):
            raise ValueError(f"capability limitations must be non-empty strings: {operation}")
        seen.add(operation)
        parsed.append(
            OperationCapability(
                operation=operation,
                supported=supported,
                reason=reason,
                limitations=tuple(limitations),
            )
        )
    supported_operations = {item.operation for item in parsed if item.supported}
    if supported_operations != set(operations):
        raise ValueError("capabilities supported operations do not match operations")
    return tuple(parsed)


def rust_capability() -> BackendCapability:
    try:
        native = importlib.import_module("fast_nc_zarr._native")
    except (ImportError, ModuleNotFoundError) as exc:
        return BackendCapability(
            name="rust",
            protocol_version=0,
            crate_version=None,
            operations=(),
            supported=False,
            reason=f"native extension unavailable: {exc}",
        )

    try:
        payload = json.loads(native.capability_json())
        if not isinstance(payload, dict):
            raise ValueError("capability response must be an object")
        raw_operations = payload["operations"]
        if not isinstance(raw_operations, list) or not all(
            isinstance(item, str) and item for item in raw_operations
        ):
            raise ValueError("operations must be a list of non-empty strings")
        operations = tuple(raw_operations)
        capabilities = _parse_capabilities(payload, operations)
        protocol_version = int(payload["protocol_version"])
        crate_version_value = payload.get("crate_version")
        if crate_version_value is not None and not isinstance(crate_version_value, str):
            raise ValueError("crate_version must be string or null")
        crate_version = crate_version_value
    except (AttributeError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return BackendCapability(
            name="rust",
            protocol_version=0,
            crate_version=None,
            operations=(),
            supported=False,
            reason=f"invalid native capability response: {exc}",
        )

    return BackendCapability(
        name=str(payload.get("backend", "rust")),
        protocol_version=protocol_version,
        crate_version=crate_version,
        operations=operations,
        supported=protocol_version == 1,
        reason=None if protocol_version == 1 else "unsupported backend protocol",
        capabilities=capabilities,
    )


def resolve_backend(requested: BackendName, operation: str) -> str:
    if requested == "python":
        return "python"
    if requested not in {"auto", "rust"}:
        raise ValueError(f"unsupported backend selection: {requested}")
    capability = rust_capability()

    operation_id = {
        "rechunk": "zarr.rechunk_f32",
        "rechunk_f32": "zarr.rechunk_f32",
        "rechunk_f64": "zarr.rechunk_f64",
    }.get(operation, operation)
    detail = capability.operation(operation_id)
    supported = capability.supported and operation_id in capability.operations
    if requested == "rust" and not supported:
        reason = (
            detail.reason
            if detail is not None and detail.reason
            else capability.reason
            or f"operation is not supported: {operation}"
        )
        raise BackendUnavailableError(reason)
    return "rust" if supported else "python"
