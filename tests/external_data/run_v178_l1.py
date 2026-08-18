#!/usr/bin/env python3
"""Run v1.7.8 L1 real-data validation cases against FLUXSATv2 and GLASS-EVI.

This script depends on the local-only ``manifest.local.json`` and on outputs
produced by ``run_samples.py`` (L0).  It performs bounded, read-only checks:

- T1: GLASS HDF-EOS coordinate reconstruction from ``StructMetadata.0``.
- T2: GLASS filename time axis across all 1012 files (8-day, no dup/missing).
- T3: GLASS int16 numerical parity between source and Zarr output.
- T4: FLUXSAT GPP numerical parity between source and Zarr output.
- T5: FLUXSAT two-month time concatenation (no gap/overlap).

All generated results are local-only and not committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import netCDF4
import numpy as np
import zarr

from fast_nc_zarr.application.services import SourceInspectionConfig, inspect_source
from fast_nc_zarr.engine import convert
from fast_nc_zarr.filename_mode import scan_filename_times
from fast_nc_zarr.models import Selection

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.local.json"
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_WORK = ROOT / "work"
DEFAULT_OUTPUTS = DEFAULT_WORK / "outputs"

GLASS_SAMPLE_FILES = (
    "GLASS14B01.V10.A2001001.2023068.hdf",
    "GLASS14B01.V10.A2001009.2023068.hdf",
)
FLUXSAT_SAMPLE_FILES = ("GPP_FluxSat_daily_v2_200101.nc4",)


def _centered_window(size: int, width: int = 32) -> tuple[int, int]:
    width = min(max(1, int(width)), int(size))
    start = max(0, (int(size) - width) // 2)
    return start, start + width


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


def _load_manifest() -> dict[str, Any]:
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    return {str(item["name"]): item for item in manifest["datasets"]}


def _source_root(manifest: dict[str, Any], name: str) -> Path:
    return Path(manifest[name]["source_root"]).expanduser()


def _assert(condition: bool, message: str, results: list[dict[str, Any]], case: str) -> None:
    results.append({"case": case, "passed": bool(condition), "message": message})


def t1_glass_coordinates(manifest: dict[str, Any]) -> dict[str, Any]:
    case = "T1_GLASS_coordinates"
    try:
        root = _source_root(manifest, "glass_evi")
        input_dir = ROOT / "inputs" / "glass_evi"
        inspection = inspect_source(
            SourceInspectionConfig(
                input_dir=input_dir,
                mode="filename",
                engine="netcdf4",
                template="doy",
                field_values=("2001", "001"),
                workers=1,
            )
        )
        inv = inspection.source_inventory
        expected_lat = 89.975 - np.arange(3600, dtype="float64") * 0.05
        expected_lon = -179.975 + np.arange(7200, dtype="float64") * 0.05
        lat_ok = np.allclose(inv.lat_values, expected_lat, atol=1e-9)
        lon_ok = np.allclose(inv.lon_values, expected_lon, atol=1e-9)
        ok = lat_ok and lon_ok
        return {
            "case": case,
            "passed": bool(ok),
            "lat_first": float(inv.lat_values[0]),
            "lat_last": float(inv.lat_values[-1]),
            "lon_first": float(inv.lon_values[0]),
            "lon_last": float(inv.lon_values[-1]),
            "source_root": str(root),
            "message": "GLASS lat/lon reconstructed from StructMetadata.0 match 0.05-deg grid"
            if ok
            else "GLASS coordinate reconstruction mismatch",
        }
    except Exception as exc:  # pragma: no cover - failure report
        return {"case": case, "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def t2_glass_time_axis(manifest: dict[str, Any]) -> dict[str, Any]:
    case = "T2_GLASS_time_axis"
    try:
        root = _source_root(manifest, "glass_evi")
        scan = scan_filename_times(
            root, template="doy", field_values=("2001", "001"), step_days=8
        )
        actual = list(scan.actual_times)
        duplicates = len(actual) - len(set(str(item) for item in actual))
        ok = (
            len(actual) == 1012
            and duplicates == 0
            and len(scan.missing_times) == 0
            and scan.step_days == 8
            and str(actual[0])[:10] == "2001-01-01"
            and str(actual[-1])[:10] == "2022-12-27"
        )
        return {
            "case": case,
            "passed": bool(ok),
            "file_count": len(actual),
            "duplicates": duplicates,
            "missing": len(scan.missing_times),
            "step_days": scan.step_days,
            "first": str(actual[0])[:10],
            "last": str(actual[-1])[:10],
            "source_root": str(root),
            "message": "GLASS filename time axis is complete and unambiguous"
            if ok
            else "GLASS filename time axis check failed",
        }
    except Exception as exc:  # pragma: no cover - failure report
        return {"case": case, "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _raw_compare_glass() -> tuple[bool, int, int]:
    output = DEFAULT_OUTPUTS / "glass_evi.zarr"
    input_dir = ROOT / "inputs" / "glass_evi"
    if not output.exists():
        raise FileNotFoundError(f"missing L0 output: {output}")
    files = sorted(input_dir.glob("*.hdf"))
    if len(files) < 2:
        raise FileNotFoundError("expected two GLASS sample links")
    lat0, lat1 = _centered_window(3600, 32)
    lon0, lon1 = _centered_window(7200, 32)
    z = zarr.open_group(output, mode="r")["evi_0.05"]
    total_diff = 0
    total_cells = 0
    for time_index, path in enumerate(files):
        with netCDF4.Dataset(path) as ds:
            source = np.asarray(ds.variables["evi_0.05"][lat0:lat1, lon0:lon1])
        target = np.asarray(z[time_index, :, :])
        total_cells += target.size
        total_diff += int(np.count_nonzero(target != source))
    return total_diff == 0, total_cells, total_diff


def t3_glass_parity(manifest: dict[str, Any]) -> dict[str, Any]:
    case = "T3_GLASS_numeric_parity"
    try:
        ok, cells, diff = _raw_compare_glass()
        return {
            "case": case,
            "passed": bool(ok),
            "cells": cells,
            "diffs": diff,
            "message": "GLASS int16 values match source exactly" if ok else "GLASS parity mismatch",
        }
    except Exception as exc:  # pragma: no cover - failure report
        return {"case": case, "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def _raw_compare_fluxsat() -> tuple[bool, int, int]:
    output = DEFAULT_OUTPUTS / "fluxsatv2.zarr"
    input_dir = ROOT / "inputs" / "fluxsatv2"
    if not output.exists():
        raise FileNotFoundError(f"missing L0 output: {output}")
    files = sorted(input_dir.glob("*.nc4"))
    if len(files) != 1:
        raise FileNotFoundError("expected one FLUXSAT sample link")
    lat0, lat1 = _centered_window(3600, 32)
    lon0, lon1 = _centered_window(7200, 32)
    time0, time1 = _centered_window(31, 2)
    z = zarr.open_group(output, mode="r")["GPP"]
    with netCDF4.Dataset(files[0]) as ds:
        source = np.asarray(
            ds.variables["GPP"][time0:time1, lat0:lat1, lon0:lon1]
        )
    target = np.asarray(z[:, :, :])
    diff = int(np.count_nonzero(target != source))
    cells = int(target.size)
    return diff == 0, cells, diff


def t4_fluxsat_parity(manifest: dict[str, Any]) -> dict[str, Any]:
    case = "T4_FLUXSAT_numeric_parity"
    try:
        ok, cells, diff = _raw_compare_fluxsat()
        return {
            "case": case,
            "passed": bool(ok),
            "cells": cells,
            "diffs": diff,
            "message": "FLUXSAT GPP values match source exactly" if ok else "FLUXSAT parity mismatch",
        }
    except Exception as exc:  # pragma: no cover - failure report
        return {"case": case, "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def t5_fluxsat_multi_month(manifest: dict[str, Any]) -> dict[str, Any]:
    case = "T5_FLUXSAT_multi_month_concat"
    try:
        root = _source_root(manifest, "fluxsatv2")
        input_dir = DEFAULT_WORK / "fluxsatv2_multi"
        if input_dir.exists():
            shutil.rmtree(input_dir)
        input_dir.mkdir(parents=True)
        for name in ("GPP_FluxSat_daily_v2_200101.nc4", "GPP_FluxSat_daily_v2_200102.nc4"):
            (input_dir / name).symlink_to((root / name).resolve())
        output = DEFAULT_OUTPUTS / "fluxsatv2_multi.zarr"
        if output.exists():
            shutil.rmtree(output)
        inspection = inspect_source(
            SourceInspectionConfig(
                input_dir=input_dir,
                mode="complete",
                engine="netcdf4",
                workers=1,
            )
        )
        inv = inspection.source_inventory
        lat0, lat1 = _centered_window(len(inv.lat_values), 32)
        lon0, lon1 = _centered_window(len(inv.lon_values), 32)
        selection = Selection(
            variables=("GPP",),
            time_start=0,
            time_stop=len(inv.times),
            lat_start=lat0,
            lat_stop=lat1,
            lon_start=lon0,
            lon_stop=lon1,
        )
        convert(
            inv,
            selection,
            output,
            auto_tune=False,
            max_workers=1,
            reserve_gib=0.25,
            overwrite=True,
            validate=True,
            progress=False,
        )
        z = zarr.open_group(output, mode="r")
        times = np.asarray(z["time"][:])
        diffs = np.diff(times.astype("int64"))
        ok = (
            int(len(times)) == 59
            and bool(np.all(diffs == 1))
            and str(inv.times[0])[:10] == "2001-01-01"
            and str(inv.times[-1])[:10] == "2001-02-28"
        )
        return {
            "case": case,
            "passed": bool(ok),
            "time_count": int(len(times)),
            "first": str(inv.times[0])[:10],
            "last": str(inv.times[-1])[:10],
            "consecutive_days": bool(np.all(diffs == 1)),
            "output": str(output),
            "message": "FLUXSAT two-month time axis concatenates with no gap/overlap"
            if ok
            else "FLUXSAT multi-month concatenation check failed",
        }
    except Exception as exc:  # pragma: no cover - failure report
        return {"case": case, "passed": False, "error": f"{type(exc).__name__}: {exc}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    manifest = _load_manifest()
    results = [
        t1_glass_coordinates(manifest),
        t2_glass_time_axis(manifest),
        t3_glass_parity(manifest),
        t4_fluxsat_parity(manifest),
        t5_fluxsat_multi_month(manifest),
    ]
    report = {
        "schema_version": 1,
        "purpose": "v1.7.8 L1 real-data validation (FLUXSATv2 + GLASS-EVI)",
        "manifest": str(args.manifest),
        "results": results,
        "passed": sum(item["passed"] for item in results),
        "failed": sum(not item["passed"] for item in results),
    }
    args.results_root.mkdir(parents=True, exist_ok=True)
    destination = args.results_root / "v178_l1_report.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(json.dumps({"report": str(destination), "passed": report["passed"], "failed": report["failed"]}, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
