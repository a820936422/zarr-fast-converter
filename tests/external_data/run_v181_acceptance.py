#!/usr/bin/env python3
"""Run the v1.8.1 real-data format acceptance gate.

Uses the read-only backup corpus under ``/run/media/owen/HDD/数据备份/`` via the
local manifest.  Every case uses a bounded center window and at most two time
steps; generated Zarr stores live only under the work root.
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

import netCDF4
import numpy as np
import rasterio
from rasterio.windows import Window
import xarray as xr

from fast_nc_zarr.application.services import SourceInspectionConfig, inspect_source
from fast_nc_zarr.engine import convert
from fast_nc_zarr.filename_mode import convert_filename
from fast_nc_zarr.models import Selection

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.local.json"
DEFAULT_WORK = ROOT / "work" / "v181"
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


def _normalized(value: Any) -> Any:
    return json.loads(json.dumps(value, default=_json_default, sort_keys=True))


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _centered(size: int, width: int) -> tuple[int, int]:
    width = min(int(size), int(width))
    start = (int(size) - width) // 2
    return start, start + width


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


def _sources(spec: dict[str, Any]) -> list[Path]:
    source_root = Path(spec["source_root"]).expanduser().resolve()
    return [(source_root / name).resolve() for name in spec["files"]]


def _prepare_links(spec: dict[str, Any], work_root: Path) -> Path:
    links = work_root / "inputs" / str(spec["name"])
    links.mkdir(parents=True, exist_ok=True)
    expected = set(spec["files"])
    for name in spec["files"]:
        source = Path(spec["source_root"]).expanduser().resolve() / name
        if not source.is_file():
            raise FileNotFoundError(f"real-data source is missing: {source}")
        _replace_link(links / name, source)
    for path in links.iterdir():
        if path.name not in expected:
            if not path.is_symlink():
                raise RuntimeError(f"unexpected non-symlink input: {path}")
            path.unlink()
    return links


def _inspect(links: Path, spec: dict[str, Any]) -> Any:
    return inspect_source(
        SourceInspectionConfig(
            input_dir=links,
            mode=spec.get("mode", "auto"),
            engine=spec.get("engine", "auto"),
            template=spec.get("template"),
            field_values=tuple(spec["field_values"]) if spec.get("field_values") else None,
            workers=1,
        )
    )


def _selection(inventory, variables: tuple[str, ...], time_stop: int, width: int) -> Selection:
    lat0, lat1 = _centered(len(inventory.lat_values), width)
    lon0, lon1 = _centered(len(inventory.lon_values), width)
    return Selection(
        variables=variables,
        time_start=0,
        time_stop=min(int(time_stop), len(inventory.times)),
        lat_start=lat0,
        lat_stop=lat1,
        lon_start=lon0,
        lon_stop=lon1,
    )


def _check_coordinates(result, inventory, lat0: int, lat1: int, lon0: int, lon1: int) -> None:
    np.testing.assert_allclose(result.lat.values, inventory.lat_values[lat0:lat1], rtol=0, atol=0)
    np.testing.assert_allclose(result.lon.values, inventory.lon_values[lon0:lon1], rtol=0, atol=0)


def _case_gosif(spec: dict[str, Any], work_root: Path) -> dict[str, Any]:
    links = _prepare_links(spec, work_root)
    inspection = _inspect(links, spec)
    inventory = inspection.source_inventory
    variable = spec["variables"][0]
    width = 32
    lat0, lat1 = _centered(len(inventory.lat_values), width)
    lon0, lon1 = _centered(len(inventory.lon_values), width)
    raw_blocks: list[np.ndarray] = []
    for name in spec["files"]:
        with rasterio.open(links / name) as dataset:
            if dataset.count != 1:
                raise AssertionError(f"GOSIF is not single-band: {dataset.count}")
            raw_blocks.append(
                np.asarray(
                    dataset.read(1, window=Window(lon0, lat0, width, width)),
                    dtype="uint16",
                )
            )
    raw = np.stack(raw_blocks)
    selection = _selection(inventory, (variable,), time_stop=2, width=width)
    output = work_root / "gosif-window.zarr"
    _remove_output(output)
    convert_filename(
        inventory,
        selection,
        output,
        auto_tune=False,
        max_workers=1,
        reserve_gib=0.25,
        overwrite=True,
        validate=True,
        progress=False,
    )
    with xr.open_zarr(
        output, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
    ) as result:
        values = np.asarray(result[variable].values)
        if values.dtype != np.dtype("uint16"):
            raise AssertionError(f"GOSIF dtype mismatch: {values.dtype}")
        np.testing.assert_array_equal(values, raw)
        _check_coordinates(result, inventory, lat0, lat1, lon0, lon1)
        expected_days = (np.asarray(inventory.times) - inventory.times[0]).astype("timedelta64[D]").astype("int64")
        actual_days = np.asarray(result.time.values).astype("int64")
        np.testing.assert_array_equal(actual_days, expected_days)
        attrs = _normalized(result[variable].attrs)
    return {
        "cells": int(raw.size),
        "diffs": 0,
        "max_abs_error": 0.0,
        "source_format": "GeoTIFF",
        "variable_dtype": "uint16",
        "window": {"time": [0, 2], "lat": [lat0, lat1], "lon": [lon0, lon1]},
        "time_values": actual_days.tolist(),
        "attrs": attrs,
        "output": str(output),
    }


def _case_mcd12c1(spec: dict[str, Any], work_root: Path) -> dict[str, Any]:
    links = _prepare_links(spec, work_root)
    inspection = _inspect(links, spec)
    inventory = inspection.source_inventory
    variable = spec["variables"][0]
    width = 32
    lat0, lat1 = _centered(len(inventory.lat_values), width)
    lon0, lon1 = _centered(len(inventory.lon_values), width)
    raw_blocks: list[np.ndarray] = []
    dtype = None
    for name in spec["files"]:
        with netCDF4.Dataset(links / name) as dataset:
            source = dataset.variables[variable]
            if dtype is None:
                dtype = np.dtype(source.dtype).newbyteorder("=")
            source.set_auto_maskandscale(False)
            raw_blocks.append(np.asarray(source[lat0:lat1, lon0:lon1], dtype=dtype))
    raw = np.stack(raw_blocks)
    selection = _selection(inventory, (variable,), time_stop=2, width=width)
    output = work_root / "mcd12c1-window.zarr"
    _remove_output(output)
    convert_filename(
        inventory,
        selection,
        output,
        auto_tune=False,
        max_workers=1,
        reserve_gib=0.25,
        overwrite=True,
        validate=True,
        progress=False,
    )
    with xr.open_zarr(
        output, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
    ) as result:
        values = np.asarray(result[variable].values)
        if np.dtype(values.dtype) != dtype:
            raise AssertionError(f"MCD12C1 dtype mismatch: {values.dtype} != {dtype}")
        np.testing.assert_array_equal(values, raw)
        _check_coordinates(result, inventory, lat0, lat1, lon0, lon1)
        expected_days = (np.asarray(inventory.times) - inventory.times[0]).astype("timedelta64[D]").astype("int64")
        actual_days = np.asarray(result.time.values).astype("int64")
        np.testing.assert_array_equal(actual_days, expected_days)
        attrs = _normalized(result[variable].attrs)
    return {
        "cells": int(raw.size),
        "diffs": 0,
        "max_abs_error": 0.0,
        "source_format": "HDF-EOS Grid",
        "variable_dtype": str(dtype),
        "window": {"time": [0, 2], "lat": [lat0, lat1], "lon": [lon0, lon1]},
        "time_values": actual_days.tolist(),
        "attrs": attrs,
        "output": str(output),
    }


def _case_glass(spec: dict[str, Any], work_root: Path) -> dict[str, Any]:
    links = _prepare_links(spec, work_root)
    inspection = _inspect(links, spec)
    inventory = inspection.source_inventory
    variable = spec["variables"][0]
    width = 32
    lat0, lat1 = _centered(len(inventory.lat_values), width)
    lon0, lon1 = _centered(len(inventory.lon_values), width)
    raw_blocks: list[np.ndarray] = []
    packing: dict[str, Any] | None = None
    for name in spec["files"]:
        with netCDF4.Dataset(links / name) as dataset:
            if dataset.disk_format != "HDF4":
                raise AssertionError(f"not an HDF4 source: {dataset.disk_format}")
            source = dataset.variables[variable]
            source.set_auto_maskandscale(False)
            raw_blocks.append(np.asarray(source[lat0:lat1, lon0:lon1], dtype="int16"))
            current = {
                "scale_factor": float(source.scale_factor),
                "add_offset": float(getattr(source, "add_offset", 0.0)),
                "_FillValue": int(source._FillValue),
            }
            if packing is None:
                packing = current
            elif current != packing:
                raise AssertionError(f"GLASS packing differs between files: {current} != {packing}")
    raw = np.stack(raw_blocks)
    selection = _selection(inventory, (variable,), time_stop=2, width=width)
    output = work_root / "glass-window.zarr"
    _remove_output(output)
    convert_filename(
        inventory,
        selection,
        output,
        auto_tune=False,
        max_workers=1,
        reserve_gib=0.25,
        overwrite=True,
        validate=True,
        progress=False,
    )
    with xr.open_zarr(
        output, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
    ) as result:
        values = np.asarray(result[variable].values)
        np.testing.assert_array_equal(values, raw)
        _check_coordinates(result, inventory, lat0, lat1, lon0, lon1)
        actual_days = np.asarray(result.time.values).astype("int64")
        expected_days = (np.asarray(inventory.times) - inventory.times[0]).astype("timedelta64[D]").astype("int64")
        np.testing.assert_array_equal(actual_days, expected_days)
        attrs = _normalized(result[variable].attrs)
        assert packing is not None
        for name, expected_value in packing.items():
            if name == "add_offset" and name not in attrs:
                continue
            if attrs.get(name) != expected_value:
                raise AssertionError(f"packing attr {name} mismatch: {attrs.get(name)!r}")
    return {
        "cells": int(raw.size),
        "diffs": 0,
        "max_abs_error": 0.0,
        "source_format": "HDF4",
        "variable_dtype": "int16",
        "packing": packing,
        "native_hdf4_supported": False,
        "native_fallback_reason": "真实 HDF4 容器读取由 Python/netCDF4 compatibility backend 完成",
        "window": {"time": [0, 2], "lat": [lat0, lat1], "lon": [lon0, lon1]},
        "time_values": actual_days.tolist(),
        "attrs": attrs,
        "output": str(output),
    }


def _case_fluxsat(spec: dict[str, Any], work_root: Path) -> dict[str, Any]:
    links = _prepare_links(spec, work_root)
    inspection = _inspect(links, spec)
    inventory = inspection.source_inventory
    variable = spec["variables"][0]
    width = 32
    selection = _selection(inventory, (variable,), time_stop=2, width=width)
    output = work_root / "fluxsat-window.zarr"
    _remove_output(output)
    convert(
        inventory,
        selection,
        output,
        auto_tune=False,
        max_workers=1,
        reserve_gib=0.25,
        overwrite=True,
        validate=True,
        progress=False,
    )
    lat0, lat1 = selection.lat_start, selection.lat_stop
    lon0, lon1 = selection.lon_start, selection.lon_stop
    raw_values = None
    with netCDF4.Dataset(links / spec["files"][0]) as dataset:
        source = dataset.variables[variable]
        source.set_auto_maskandscale(False)
        raw_values = np.asarray(source[0:2, lat0:lat1, lon0:lon1], dtype="float32")
        fill = getattr(source, "_FillValue", np.nan)
    with xr.open_zarr(
        output, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
    ) as result:
        values = np.asarray(result[variable].values).copy()
        if values.dtype != np.dtype("float32"):
            raise AssertionError(f"FLUXSAT dtype mismatch: {values.dtype}")
        if np.isfinite(fill):
            values[values == fill] = np.nan
        expected = raw_values.copy()
        if np.isfinite(fill):
            expected[expected == fill] = np.nan
        np.testing.assert_allclose(values, expected, rtol=0, atol=0, equal_nan=True)
        _check_coordinates(result, inventory, lat0, lat1, lon0, lon1)
        attrs = _normalized(result[variable].attrs)
    return {
        "cells": int(raw_values.size),
        "diffs": 0,
        "max_abs_error": 0.0,
        "source_format": "NetCDF4",
        "variable_dtype": "float32",
        "window": {"time": [0, 2], "lat": [lat0, lat1], "lon": [lon0, lon1]},
        "attrs": attrs,
        "output": str(output),
    }


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
        name: paths
        for name, spec in specs.items()
        for paths in [_sources(spec)]
    }
    source_evidence = [
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for paths in sources.values()
        for path in paths
    ]

    cases: list[dict[str, Any]] = []
    if "gosif_gpp" in specs:
        cases.append(
            _run_case(
                "gosif_geotiff_filename_mode",
                EXACT,
                lambda: _case_gosif(specs["gosif_gpp"], args.work_root),
            )
        )
    if "mcd12c1" in specs:
        cases.append(
            _run_case(
                "mcd12c1_hdf_eos_classification",
                EXACT,
                lambda: _case_mcd12c1(specs["mcd12c1"], args.work_root),
            )
        )
    if "glass_evi" in specs:
        cases.append(
            _run_case(
                "glass_hdf4_packed_parity",
                EXACT,
                lambda: _case_glass(specs["glass_evi"], args.work_root),
            )
        )
    if "fluxsatv2" in specs:
        cases.append(
            _run_case(
                "fluxsat_netcdf4_complete",
                EXACT,
                lambda: _case_fluxsat(specs["fluxsatv2"], args.work_root),
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
    destination = args.results_root / "v181_acceptance_report.json"
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
    print(json.dumps({"report": str(destination), "passed": passed, "failed": failed}, ensure_ascii=False))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
