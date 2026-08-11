from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from typing import Literal


BackendName = Literal["auto", "python", "rust"]


class BackendUnavailableError(RuntimeError):
    """Raised when an explicitly requested native backend is unavailable."""


@dataclass(frozen=True, slots=True)
class BackendCapability:
    name: str
    protocol_version: int
    crate_version: str | None
    operations: tuple[str, ...]
    supported: bool
    reason: str | None = None


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
        operations = tuple(str(item) for item in payload["operations"])
        protocol_version = int(payload["protocol_version"])
        crate_version = str(payload["crate_version"])
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
    )


def resolve_backend(requested: BackendName, operation: str) -> str:
    if requested == "python":
        return "python"
    if requested not in {"auto", "rust"}:
        raise ValueError(f"unsupported backend selection: {requested}")

    capability = rust_capability()
    supported = capability.supported and operation in capability.operations
    if requested == "rust" and not supported:
        reason = capability.reason or f"operation is not supported: {operation}"
        raise BackendUnavailableError(reason)
    return "rust" if supported else "python"
