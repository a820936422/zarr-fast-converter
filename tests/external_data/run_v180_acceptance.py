#!/usr/bin/env python3
"""Run the v1.8 real-data HDF4, integer, and conservative backend gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json
from pathlib import Path
import shutil
import traceback
from typing import Any, Callable

import netCDF4
import numpy as np
import xarray as xr

from fast_nc_zarr.application.services import SourceInspectionConfig, inspect_source
from fast_nc_zarr.engine import convert
from fast_nc_zarr.filename_mode import convert_filename
from fast_nc_zarr.models import Selection
from fast_nc_zarr.resampling.engine import run_resample
from fast_nc_zarr.resampling.models import ResampleConfig

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.local.json"
DEFAULT_WORK = ROOT / "work" / "v180"
DEFAULT_RESULTS = ROOT / "results"
GLASS_FILES = (
    "GLASS14B01.V10.A2001001.2023068.hdf",
    "GLASS14B01.V10.A2001009.2023068.hdf",
)
FLUX_FILE = "GPP_FluxSat_daily_v2_200101.nc4"
EXACT = {"rtol": 0.0, "atol": 0.0, "equal_nan": True}
RESAMPLE_TOLERANCE = {"rtol": 5e-5, "atol": 1e-6, "equal_nan": True}


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


def _base_case(name: str, backend: str, method: str, tolerance: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": name,
        "backend": backend,
        "method": method,
        "tolerance": tolerance,
        "cells": 0,
        "diffs": 0,
        "max_abs_error": 0.0,
        "passed": False,
    }


def _run_case(
    name: str,
    backend: str,
    method: str,
    tolerance: dict[str, Any],
    function: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    record = _base_case(name, backend, method, tolerance)
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


def _manifest_sources(manifest_path: Path) -> dict[str, Path]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    specs = {str(item["name"]): item for item in payload["datasets"]}
    glass_root = Path(specs["glass_evi"]["source_root"]).expanduser().resolve()
    flux_root = Path(specs["fluxsatv2"]["source_root"]).expanduser().resolve()
    sources = {
        name: (glass_root / name).resolve()
        for name in GLASS_FILES
    }
    sources[FLUX_FILE] = (flux_root / FLUX_FILE).resolve()
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(f"real-data source is missing: {path}")
    return sources


def _prepare_glass(
    sources: dict[str, Path], work_root: Path, state: dict[str, Any]
) -> dict[str, Any]:
    links = work_root / "inputs" / "glass_evi"
    links.mkdir(parents=True, exist_ok=True)
    expected = set(GLASS_FILES)
    for name in GLASS_FILES:
        _replace_link(links / name, sources[name])
    for path in links.iterdir():
        if path.name not in expected:
            if not path.is_symlink():
                raise RuntimeError(f"unexpected non-symlink GLASS input: {path}")
            path.unlink()

    magic = sources[GLASS_FILES[0]].read_bytes()[:4].hex()
    if magic != "0e031301":
        raise AssertionError(f"unexpected HDF4 magic: {magic}")
    raw_blocks: list[np.ndarray] = []
    packing: dict[str, Any] | None = None
    for path in (sources[name] for name in GLASS_FILES):
        with netCDF4.Dataset(path) as dataset:
            if dataset.disk_format != "HDF4":
                raise AssertionError(f"not an HDF4 source: {dataset.disk_format}")
            if "StructMetadata.0" not in dataset.ncattrs():
                raise AssertionError("StructMetadata.0 is missing")
            variable = dataset.variables["evi_0.05"]
            if np.dtype(variable.dtype).newbyteorder("=") != np.dtype("int16"):
                raise AssertionError(f"GLASS source is not packed int16: {variable.dtype}")
            if "scale_factor" not in variable.ncattrs():
                raise AssertionError("GLASS scale_factor is missing")
            lat0, lat1 = _centered(variable.shape[0], 32)
            lon0, lon1 = _centered(variable.shape[1], 32)
            variable.set_auto_maskandscale(False)
            raw_blocks.append(np.asarray(variable[lat0:lat1, lon0:lon1], dtype="int16"))
            current = {
                "scale_factor": float(variable.scale_factor),
                "add_offset": float(getattr(variable, "add_offset", 0.0)),
                "_FillValue": int(variable._FillValue),
            }
            if packing is None:
                packing = current
            elif current != packing:
                raise AssertionError(f"GLASS packing differs between files: {current} != {packing}")

    inspection = inspect_source(
        SourceInspectionConfig(
            input_dir=links,
            mode="filename",
            engine="netcdf4",
            template="doy",
            field_values=("2001", "001"),
            workers=1,
        )
    )
    inventory = inspection.source_inventory
    lat0, lat1 = _centered(len(inventory.lat_values), 32)
    lon0, lon1 = _centered(len(inventory.lon_values), 32)
    selection = Selection(
        variables=("evi_0.05",),
        time_start=0,
        time_stop=2,
        lat_start=lat0,
        lat_stop=lat1,
        lon_start=lon0,
        lon_stop=lon1,
    )
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
    raw_source = np.stack(raw_blocks)
    with xr.open_zarr(
        output,
        consolidated=False,
        chunks=None,
        decode_times=False,
        mask_and_scale=False,
    ) as result:
        raw_target = np.asarray(result["evi_0.05"].values)
        diffs = int(np.count_nonzero(raw_target != raw_source))
        if diffs:
            raise AssertionError(f"GLASS raw int16 mismatch count: {diffs}")
        np.testing.assert_allclose(result.lat.values, inventory.lat_values[lat0:lat1], rtol=0, atol=0)
        np.testing.assert_allclose(result.lon.values, inventory.lon_values[lon0:lon1], rtol=0, atol=0)
        np.testing.assert_array_equal(result.time.values, [0, 8])
        attrs = _normalized(result["evi_0.05"].attrs)
        assert packing is not None
        for name, expected_value in packing.items():
            if name == "add_offset" and name not in attrs:
                continue
            if attrs.get(name) != expected_value:
                raise AssertionError(f"packing attr {name} mismatch: {attrs.get(name)!r}")

    assert packing is not None
    fill = packing["_FillValue"]
    source_physical = raw_source.astype("float64") * packing["scale_factor"] + packing["add_offset"]
    target_physical = raw_target.astype("float64") * packing["scale_factor"] + packing["add_offset"]
    source_physical[raw_source == fill] = np.nan
    target_physical[raw_target == fill] = np.nan
    np.testing.assert_allclose(source_physical, target_physical, rtol=0, atol=0, equal_nan=True)
    state.update(
        {
            "glass_raw": raw_source,
            "glass_lat": np.asarray(inventory.lat_values[lat0:lat1], dtype="float32"),
            "glass_lon": np.asarray(inventory.lon_values[lon0:lon1], dtype="float32"),
            "glass_packing": packing,
            "glass_attrs": attrs,
        }
    )
    return {
        "cells": int(raw_source.size),
        "diffs": diffs,
        "max_abs_error": 0.0,
        "source_format": "HDF4",
        "source_magic": magic,
        "variable_dtype": "int16",
        "packing": packing,
        "packing_attrs": attrs,
        "window": {"time": [0, 2], "lat": [lat0, lat1], "lon": [lon0, lon1]},
        "time_values": [0, 8],
        "native_hdf4_supported": False,
        "output": str(output),
    }


def _derived_int16(work_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    if "glass_raw" not in state:
        raise RuntimeError("GLASS real-data case did not produce derived input")
    source_dir = work_root / "derived-int16"
    source_dir.mkdir(parents=True, exist_ok=True)
    netcdf_path = source_dir / "glass-derived-int16.nc4"
    raw = state["glass_raw"]
    fill = np.int16(state["glass_packing"]["_FillValue"])
    with netCDF4.Dataset(netcdf_path, "w", format="NETCDF4") as dataset:
        dataset.title = "Derived from GLASS HDF4 real raw values"
        for name, size in (("time", 2), ("lat", raw.shape[1]), ("lon", raw.shape[2])):
            dataset.createDimension(name, size)
        time = dataset.createVariable("time", "i4", ("time",))
        time.units = "days since 2001-01-01"
        time[:] = [0, 8]
        dataset.createVariable("lat", "f4", ("lat",))[:] = state["glass_lat"]
        dataset.createVariable("lon", "f4", ("lon",))[:] = state["glass_lon"]
        value = dataset.createVariable(
            "evi_0.05", "i2", ("time", "lat", "lon"), fill_value=fill
        )
        value.set_auto_maskandscale(False)
        value.long_name = state["glass_attrs"].get("long_name", "GLASS EVI")
        value.valid_range = np.asarray(state["glass_attrs"].get("valid_range", [-2000, 10000]), dtype="int16")
        value[:] = raw

    native_output = work_root / "derived-int16-rust.zarr"
    python_output = work_root / "derived-int16-python.zarr"
    _remove_output(native_output)
    _remove_output(python_output)
    native = importlib.import_module("fast_nc_zarr._native")
    native_metrics = json.loads(native.convert_netcdf_json(str(netcdf_path), str(native_output)))
    inspection = inspect_source(
        SourceInspectionConfig(input_dir=source_dir, mode="complete", engine="netcdf4", workers=1)
    )
    inventory = inspection.source_inventory
    selection = Selection(
        variables=("evi_0.05",),
        time_start=0,
        time_stop=2,
        lat_start=0,
        lat_stop=raw.shape[1],
        lon_start=0,
        lon_stop=raw.shape[2],
    )
    convert(
        inventory,
        selection,
        python_output,
        auto_tune=False,
        max_workers=1,
        reserve_gib=0.25,
        overwrite=True,
        validate=True,
        progress=False,
    )
    with xr.open_zarr(
        native_output, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
    ) as native_result, xr.open_zarr(
        python_output, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
    ) as python_result:
        native_values = np.asarray(native_result["evi_0.05"].values)
        python_values = np.asarray(python_result["evi_0.05"].values)
        np.testing.assert_array_equal(native_values, raw)
        np.testing.assert_array_equal(python_values, raw)
        np.testing.assert_array_equal(native_values, python_values)
        for name in ("time", "lat", "lon"):
            np.testing.assert_array_equal(native_result[name].values, python_result[name].values)
        if native_result["evi_0.05"].dtype != python_result["evi_0.05"].dtype:
            raise AssertionError("derived int16 dtype mismatch")
        native_fill = native_result["evi_0.05"].attrs.get("_FillValue")
        python_fill = python_result["evi_0.05"].attrs.get("_FillValue")
        if int(native_fill) != int(fill) or int(python_fill) != int(fill):
            raise AssertionError(f"derived fill mismatch: {native_fill}, {python_fill}")
        if _normalized(native_result["evi_0.05"].attrs) != _normalized(
            python_result["evi_0.05"].attrs
        ):
            raise AssertionError("derived int16 attrs mismatch")
    return {
        "cells": int(raw.size),
        "diffs": 0,
        "max_abs_error": 0.0,
        "source": str(netcdf_path),
        "source_provenance": [str(path) for path in state.get("glass_sources", ())],
        "native_metrics": native_metrics,
        "native_output": str(native_output),
        "python_output": str(python_output),
        "dtype": "int16",
        "fill_value": int(fill),
    }


def _prepare_fluxsat(source: Path, work_root: Path, state: dict[str, Any]) -> dict[str, Any]:
    with netCDF4.Dataset(source) as dataset:
        if dataset.disk_format not in {"NETCDF4_CLASSIC", "HDF5"}:
            raise AssertionError(f"unexpected FLUXSAT format: {dataset.disk_format}")
        variable = dataset.variables["GPP"]
        if np.dtype(variable.dtype) != np.dtype("float32"):
            raise AssertionError(f"GPP is not float32: {variable.dtype}")
        lat0, lat1 = _centered(variable.shape[1], 512)
        lon0, lon1 = _centered(variable.shape[2], 512)
        variable.set_auto_maskandscale(False)
        raw_values = np.asarray(variable[:8, lat0:lat1, lon0:lon1], dtype="float32")
        fill_value = float(getattr(variable, "_FillValue", np.nan))
        values = raw_values.copy()
        if np.isfinite(fill_value):
            values[raw_values == fill_value] = np.nan
        lat = np.asarray(dataset.variables["lat"][lat0:lat1], dtype="float32")
        lon = np.asarray(dataset.variables["lon"][lon0:lon1], dtype="float32")
        time = np.asarray(dataset.variables["time"][:8], dtype="float32")
        source_format = dataset.disk_format
        attrs = {
            name: getattr(variable, name)
            for name in variable.ncattrs()
            if name != "_FillValue"
        }
    subset = xr.Dataset(
        {"GPP": (("time", "lat", "lon"), values, attrs)},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"source_path": str(source), "source_format": source_format},
    )
    target = work_root / "fluxsat-window.zarr"
    _remove_output(target)
    subset.to_zarr(
        target,
        mode="w",
        consolidated=False,
        zarr_format=3,
        encoding={name: {"compressors": []} for name in ("GPP", "time", "lat", "lon")},
    )
    subset.close()
    state["fluxsat_input"] = target
    state["fluxsat_window"] = {"time": [0, 8], "lat": [lat0, lat1], "lon": [lon0, lon1]}
    state["fluxsat_format"] = source_format
    return {
        "cells": int(values.size),
        "diffs": 0,
        "max_abs_error": 0.0,
        "source_format": source_format,
        "source": str(source),
        "variable": "GPP",
        "dtype": "float32",
        "window": state["fluxsat_window"],
        "output": str(target),
        "zarr_format": 3,
        "compression": "none",
    }
def _fluxsat_resample(
    method: str, backend: str, work_root: Path, state: dict[str, Any]
) -> dict[str, Any]:
    source = state.get("fluxsat_input")
    if source is None:
        raise RuntimeError("FLUXSAT subset case did not produce a Zarr input")
    output = work_root / f"fluxsat-{method}-{backend}.zarr"
    _remove_output(output)
    metrics = run_resample(
        ResampleConfig(
            input=source,
            output=output,
            resolution=0.1,
            method=method,
            backend=backend,
            skipna=False,
            tile_size=256,
            time_block=2,
            compute_workers=1,
            space_workers=1,
            validate=True,
        ),
        progress=False,
    )
    state[f"fluxsat_{method}_{backend}"] = output
    return {
        "cells": int(8 * 256 * 256),
        "diffs": 0,
        "max_abs_error": 0.0,
        "metrics": metrics,
        "output": str(output),
        "source_format": state["fluxsat_format"],
        "window": state["fluxsat_window"],
        "provenance_variable": "GPP",
    }


def _compare_fluxsat(method: str, state: dict[str, Any]) -> dict[str, Any]:
    python_path = state.get(f"fluxsat_{method}_python")
    rust_path = state.get(f"fluxsat_{method}_rust")
    if python_path is None or rust_path is None:
        raise RuntimeError(f"{method} backend outputs are incomplete")
    with xr.open_zarr(
        python_path, consolidated=False, chunks=None, decode_times=False
    ) as python_result, xr.open_zarr(
        rust_path, consolidated=False, chunks=None, decode_times=False
    ) as rust_result:
        for name in ("time", "lat", "lon"):
            np.testing.assert_allclose(
                rust_result[name].values,
                python_result[name].values,
                rtol=0,
                atol=1e-6,
            )
        python_values = np.asarray(python_result["GPP"].values)
        rust_values = np.asarray(rust_result["GPP"].values)
        python_nan = np.isnan(python_values)
        rust_nan = np.isnan(rust_values)
        mask_diffs = int(np.count_nonzero(python_nan != rust_nan))
        if mask_diffs:
            raise AssertionError(f"{method} NaN mask differs in {mask_diffs} cells")
        finite = ~(python_nan | rust_nan)
        errors = np.abs(rust_values[finite].astype("float64") - python_values[finite].astype("float64"))
        max_error = float(errors.max(initial=0.0))
        close = np.isclose(
            rust_values,
            python_values,
            rtol=RESAMPLE_TOLERANCE["rtol"],
            atol=RESAMPLE_TOLERANCE["atol"],
            equal_nan=True,
        )
        diffs = int(np.count_nonzero(~close))
        if diffs:
            raise AssertionError(f"{method} values differ in {diffs} cells; max={max_error}")
        if _normalized(rust_result["GPP"].attrs) != _normalized(python_result["GPP"].attrs):
            raise AssertionError(f"{method} attrs mismatch")
    rust_metrics = state[f"fluxsat_{method}_rust_metrics"]
    if rust_metrics.get("backend") != "rust" or rust_metrics.get("backend_fallback"):
        raise AssertionError(f"{method} did not resolve strict Rust: {rust_metrics}")
    return {
        "cells": int(python_values.size),
        "diffs": diffs,
        "max_abs_error": max_error,
        "nan_mask_diffs": mask_diffs,
        "source_format": state["fluxsat_format"],
        "window": state["fluxsat_window"],
        "provenance_variable": "GPP",
        "python_output": str(python_path),
        "rust_output": str(rust_path),
        "rust_metrics": rust_metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    args = parser.parse_args(argv)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)
    sources = _manifest_sources(args.manifest)
    source_evidence = [
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sources.values()
    ]
    state: dict[str, Any] = {"glass_sources": tuple(sources[name] for name in GLASS_FILES)}
    cases: list[dict[str, Any]] = []
    cases.append(
        _run_case(
            "glass_hdf4_packed_parity",
            "python",
            "filename.convert",
            EXACT,
            lambda: _prepare_glass(sources, args.work_root, state),
        )
    )
    cases.append(
        _run_case(
            "glass_derived_int16_cross_backend",
            "rust+python",
            "raw.netcdf.convert",
            EXACT,
            lambda: _derived_int16(args.work_root, state),
        )
    )
    cases.append(
        _run_case(
            "fluxsat_real_subset",
            "python",
            "zarr.v3.write",
            EXACT,
            lambda: _prepare_fluxsat(sources[FLUX_FILE], args.work_root, state),
        )
    )
    for method in ("conservative", "conservative_normed"):
        for backend in ("python", "rust"):
            case = _run_case(
                f"fluxsat_{method}_{backend}",
                backend,
                method,
                RESAMPLE_TOLERANCE,
                lambda method=method, backend=backend: _fluxsat_resample(
                    method, backend, args.work_root, state
                ),
            )
            cases.append(case)
            if case["passed"]:
                state[f"fluxsat_{method}_{backend}_metrics"] = case["metrics"]
        cases.append(
            _run_case(
                f"fluxsat_{method}_parity",
                "rust+python",
                method,
                RESAMPLE_TOLERANCE,
                lambda method=method: _compare_fluxsat(method, state),
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
    destination = args.results_root / "v180_acceptance_report.json"
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
