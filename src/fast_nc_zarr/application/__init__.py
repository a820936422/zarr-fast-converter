"""Application services shared by the CLI and Tauri compatibility runtime."""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .services import (  # noqa: F401
        ConversionConfig,
        ConversionPreview,
        InspectionResult,
        RechunkConfig,
        RechunkPreview,
        ResampleConfig,
        ResamplePreview,
        SourceInspectionConfig,
    )

_SERVICE_EXPORTS = {
    "ConversionConfig",
    "ConversionPreview",
    "InspectionResult",
    "RechunkConfig",
    "RechunkPreview",
    "ResampleConfig",
    "ResamplePreview",
    "SourceInspectionConfig",
    "inspect_source",
    "inspect_time_metadata",
    "inspect_zarr",
    "inspect_resample",
    "load_inspection_snapshot",
    "preview_conversion",
    "preview_rechunk",
    "preview_resample",
    "run_conversion",
    "run_rechunk",
    "run_resample",
    "save_inspection_snapshot",
}

__all__ = sorted(_SERVICE_EXPORTS)


def __getattr__(name: str):
    if name not in _SERVICE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".services", __name__), name)
    globals()[name] = value
    return value
