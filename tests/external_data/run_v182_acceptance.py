#!/usr/bin/env python3
"""Run the v1.8.2 real-data acceptance gate.

Covers the filename-mode PipelineFusion real-data convergence (3.1 of the
v1.8.2 development doc) using the read-only backup corpus under
``/run/media/owen/HDD/数据备份/`` via the local manifest:

- ``gosif_filename_fusion_parity``: real GeoTIFF (uint16, EPSG:4326, 0.05deg)
  windowed convert→resample; the in-memory single-pass fusion must equal the
  on-disk intermediate path value-for-value (coordinates and time axis too),
  with manifest evidence ``fused_in_memory`` / ``write_amplification=1.0`` and
  NO intermediate ``source-crop.zarr``.
- ``glass_filename_fusion_gap_parity``: real HDF4-EOS packed int16+
  scale_factor batch WITH a missing time key (2001017); the fused crop must
  CF-decode exactly like the reopened on-disk crop (fill→NaN, physical
  values), so fused and disk outputs agree including the missing day.

Every case uses a bounded centred window; generated Zarr stores live only
under the work root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import shutil
import traceback
from typing import Any, Callable

import numpy as np
import xarray as xr

from fast_nc_zarr.application.services import SourceInspectionConfig, inspect_source
from fast_nc_zarr.pipeline.engine import preview_pipeline, run_pipeline
from fast_nc_zarr.pipeline.models import (
    PipelineConfig,
    PipelineConversionOptions,
    PipelineGeneralConfig,
    PipelineOperations,
    PipelineResamplingOptions,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.local.json"
DEFAULT_WORK = ROOT / "work" / "v182"
DEFAULT_RESULTS = ROOT / "results"
EXACT = {"rtol": 0.0, "atol": 0.0, "equal_nan": True}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return repr(value)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _replace_link(link: Path, source: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if not link.is_symlink():
            raise RuntimeError(f"refusing to replace non-symlink input: {link}")
        link.unlink()
    link.symlink_to(source.resolve())


def _remove_output(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _base_case(name: str, tolerance: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": name,
        "tolerance": tolerance,
        "cells": 0,
        "diffs": 0,
        "max_abs_error": 0.0,
        "passed": False,
    }


def _run_case(
    name: str,
    tolerance: dict[str, Any],
    function: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    record = _base_case(name, tolerance)
    try:
        record.update(function())
        record["passed"] = True
    except Exception as exc:  # Preserve every failure in the atomic report.
        record.update(
            {
                "passed": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return record


def _load_manifest(manifest_path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {str(item["name"]): item for item in payload["datasets"]}


def _inspect(links: Path, spec: dict[str, Any], engine: str) -> Any:
    return inspect_source(
        SourceInspectionConfig(
            input_dir=links,
            mode="filename",
            engine=engine,
            template="doy",
            field_values=("2001", "001"),
            workers=1,
        )
    )


def _centred_window(inspection, width: int) -> tuple[float, float, float, float]:
    inventory = inspection.inventory
    n_lat = min(int(width), len(inventory.lat_values))
    n_lon = min(int(width), len(inventory.lon_values))
    lat_start = (len(inventory.lat_values) - n_lat) // 2
    lon_start = (len(inventory.lon_values) - n_lon) // 2
    lat = inventory.lat_values[lat_start : lat_start + n_lat]
    lon = inventory.lon_values[lon_start : lon_start + n_lon]
    return (
        float(min(lat)),
        float(max(lat)),
        float(min(lon)),
        float(max(lon)),
    )


def _pipeline_config(
    output: Path,
    temporary: Path,
    window: tuple[float, float, float, float],
    time_end: str,
    *,
    fusion: bool,
) -> PipelineConfig:
    lat_min, lat_max, lon_min, lon_max = window
    return PipelineConfig(
        general=PipelineGeneralConfig(
            output=output,
            temporary_dir=temporary,
            time_start="2001-01-01",
            time_end=time_end,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            fusion=fusion,
        ),
        conversion=PipelineConversionOptions(auto_tune=False, max_workers=1),
        operations=PipelineOperations(resample=True, rechunk=False, recompress=False),
        resampling=PipelineResamplingOptions(
            resolution=0.1,
            method="bilinear",
            compute_workers=1,
            space_workers=1,
            tile_size=8,
            time_block=1,
        ),
    )


def _fusion_case(
    spec: dict[str, Any],
    work_root: Path,
    *,
    case: str,
    names: tuple[str, ...],
    engine: str,
    time_end: str,
    variable: str,
) -> dict[str, Any]:
    links = work_root / "inputs" / f"{case}-{engine}"
    if links.exists():
        for path in links.iterdir():
            if path.is_symlink():
                path.unlink()
    links.mkdir(parents=True, exist_ok=True)
    source_root = Path(spec["source_root"]).expanduser().resolve()
    for name in names:
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(f"real-data source is missing: {source}")
        _replace_link(links / name, source)
    inspection = _inspect(links, spec, engine)
    window = _centred_window(inspection, 64)
    fused_output = work_root / f"{case}-fused.zarr"
    disk_output = work_root / f"{case}-disk.zarr"
    temporary = work_root / f"{case}-temporary"
    _remove_output(fused_output)
    _remove_output(disk_output)
    _remove_output(temporary)
    temporary.mkdir(parents=True, exist_ok=True)
    fused_config = _pipeline_config(
        fused_output, temporary, window, time_end, fusion=True
    )
    plan = preview_pipeline(inspection, fused_config)
    if not plan.streaming_fusion_eligible:
        raise AssertionError("filename fusion case is not fusion-eligible")
    fused_result = run_pipeline(inspection, fused_config, progress=False)
    fused_manifest = json.loads(
        Path(fused_result["manifest"]).read_text(encoding="utf-8")
    )
    if fused_manifest["stages"]["conversion"]["status"] != "fused_in_memory":
        raise AssertionError(
            f"expected fused_in_memory, got "
            f"{fused_manifest['stages']['conversion']['status']}"
        )
    if fused_manifest["logical_io"]["write_amplification"] != 1.0:
        raise AssertionError("fused path must report write_amplification=1.0")
    pipeline_root = Path(fused_result["manifest"]).parent
    if (pipeline_root / "source-crop.zarr").exists():
        raise AssertionError("fused path must not create source-crop.zarr")

    disk_config = _pipeline_config(
        disk_output, temporary, window, time_end, fusion=False
    )
    disk_result = run_pipeline(inspection, disk_config, progress=False)
    disk_manifest = json.loads(
        Path(disk_result["manifest"]).read_text(encoding="utf-8")
    )
    if disk_manifest["stages"]["conversion"]["status"] != "validated":
        raise AssertionError(
            f"expected disk-path validated, got "
            f"{disk_manifest['stages']['conversion']['status']}"
        )

    fused = xr.open_zarr(fused_output, consolidated=False)
    disk = xr.open_zarr(disk_output, consolidated=False)
    try:
        fused_values = np.asarray(fused[variable].values).copy()
        disk_values = np.asarray(disk[variable].values).copy()
        np.testing.assert_array_equal(
            fused["time"].values,
            disk["time"].values,
            err_msg=f"{case}: time axis must match",
        )
        np.testing.assert_allclose(
            fused["lat"].values,
            disk["lat"].values,
            rtol=0,
            atol=0,
            err_msg=f"{case}: lat coordinate must match",
        )
        np.testing.assert_allclose(
            fused["lon"].values,
            disk["lon"].values,
            rtol=0,
            atol=0,
            err_msg=f"{case}: lon coordinate must match",
        )
        np.testing.assert_allclose(
            fused_values,
            disk_values,
            rtol=EXACT["rtol"],
            atol=EXACT["atol"],
            equal_nan=EXACT["equal_nan"],
            err_msg=f"{case}: fused vs disk values must match exactly",
        )
    finally:
        fused.close()
        disk.close()
    shape = fused_values.shape
    return {
        "cells": int(fused_values.size),
        "diffs": 0,
        "max_abs_error": 0.0,
        "shape": list(shape),
        "output_dtype": str(fused_values.dtype),
        "missing_time_keys": list(inspection.inventory.missing_time_keys),
        "expected_time_count": len(inspection.inventory.times),
        "write_amplification": float(
            fused_manifest["logical_io"]["write_amplification"]
        ),
        "temporary_write_bytes": int(
            fused_manifest["logical_io"]["temporary_write_bytes"]
        ),
        "fused_status": fused_manifest["stages"]["conversion"]["status"],
        "intermediate_store_created": (pipeline_root / "source-crop.zarr").exists(),
        "outputs": {"fused": str(fused_output), "disk": str(disk_output)},
    }


def _case_gosif(spec: dict[str, Any], work_root: Path) -> dict[str, Any]:
    return _fusion_case(
        spec,
        work_root,
        case="gosif",
        names=(
            "GOSIF_GPP_2001001_Mean.tif",
            "GOSIF_GPP_2001009_Mean.tif",
            "GOSIF_GPP_2001017_Mean.tif",
        ),
        engine="rasterio",
        time_end="2001-01-17",
        variable="band_data",
    )


def _case_glass(spec: dict[str, Any], work_root: Path) -> dict[str, Any]:
    return _fusion_case(
        spec,
        work_root,
        case="glass",
        names=(
            "GLASS14B01.V10.A2001001.2023068.hdf",
            "GLASS14B01.V10.A2001009.2023068.hdf",
            "GLASS14B01.V10.A2001025.2023068.hdf",
        ),
        engine="netcdf4",
        time_end="2001-01-25",
        variable="evi_0.05",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)

    specs = _load_manifest(args.manifest)
    sources = {
        name: [(Path(spec["source_root"]).expanduser().resolve() / file) for file in spec["files"]]
        for name, spec in specs.items()
    }
    source_evidence = [
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for paths in sources.values()
        for path in paths
        if path.is_file()
    ]

    cases: list[dict[str, Any]] = []
    if "gosif_gpp" in specs:
        cases.append(
            _run_case(
                "gosif_filename_fusion_parity",
                EXACT,
                lambda: _case_gosif(specs["gosif_gpp"], args.work_root),
            )
        )
    if "glass_evi" in specs:
        cases.append(
            _run_case(
                "glass_filename_fusion_gap_parity",
                EXACT,
                lambda: _case_glass(specs["glass_evi"], args.work_root),
            )
        )

    passed = sum(bool(item["passed"]) for item in cases)
    failed = len(cases) - passed
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest.resolve()),
        "source_policy": "read-only source links; generated data only under work_root",
        "sources": source_evidence,
        "cases": cases,
        "passed": passed,
        "failed": failed,
        "all_passed": failed == 0,
        "work_root": str(args.work_root.resolve()),
    }
    destination = args.results_root / "v182_acceptance_report.json"
    temporary = destination.with_suffix(".tmp")
    for case in cases:
        if not case["passed"]:
            print(
                f"FAILED {case['case']}: "
                f"{case.get('error_type', 'case_failure')}: {case.get('error', '')}"
            )
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(
        json.dumps(
            {"report": str(destination), "passed": passed, "failed": failed},
            ensure_ascii=False,
        )
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())