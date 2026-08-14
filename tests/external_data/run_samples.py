#!/usr/bin/env python3
"""Run bounded, read-only samples from the external scientific datasets.

The manifest points at the original files; ``inputs`` contains symlinks only.
All generated Zarr stores, logs and JSON reports are intentionally local test
artifacts and are safe to delete after review.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time
import traceback
from typing import Any

import numpy as np

from fast_nc_zarr.application.services import SourceInspectionConfig, inspect_source
from fast_nc_zarr.engine import convert
from fast_nc_zarr.filename_mode import convert_filename
from fast_nc_zarr.models import Selection
from fast_nc_zarr.rechunking.inspection import inspect_store


ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.local.json"
DEFAULT_INPUTS = ROOT / "inputs"
DEFAULT_RESULTS = ROOT / "results"
DEFAULT_WORK = ROOT / "work"
DEFAULT_LOGS = ROOT / "logs"


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


def _prepare_links(spec: dict[str, Any], inputs_root: Path) -> Path:
    source_root = Path(spec["source_root"]).expanduser()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    input_dir = inputs_root / str(spec["name"])
    input_dir.mkdir(parents=True, exist_ok=True)
    expected = set(str(item) for item in spec["files"])
    for name in expected:
        source = source_root / name
        if not source.is_file():
            raise FileNotFoundError(f"sample source does not exist: {source}")
        link = input_dir / name
        if link.exists() or link.is_symlink():
            if not link.is_symlink():
                raise RuntimeError(f"refusing to replace non-link test input: {link}")
            link.unlink()
        link.symlink_to(source.resolve())
    actual = {path.name for path in input_dir.iterdir() if path.is_symlink()}
    if actual != expected:
        raise RuntimeError(f"sample links mismatch for {spec['name']}: {actual} != {expected}")
    return input_dir


def _centered_window(size: int, width: int = 32) -> tuple[int, int]:
    width = min(max(1, int(width)), int(size))
    start = max(0, (int(size) - width) // 2)
    return start, start + width


def _selection(inventory, variables: tuple[str, ...]) -> Selection:
    time_start, time_stop = _centered_window(len(inventory.times), 2)
    lat_start, lat_stop = _centered_window(len(inventory.lat_values), 32)
    lon_start, lon_stop = _centered_window(len(inventory.lon_values), 32)
    return Selection(
        variables=variables,
        time_start=time_start,
        time_stop=time_stop,
        lat_start=lat_start,
        lat_stop=lat_stop,
        lon_start=lon_start,
        lon_stop=lon_stop,
    )


def _inventory_summary(inventory) -> dict[str, Any]:
    return {
        "source_engine": inventory.source_engine,
        "source_mode": inventory.source_mode,
        "source_dimensions": list(inventory.source_dimensions),
        "file_count": len(inventory.files),
        "total_bytes": int(inventory.total_bytes),
        "time_count": int(len(inventory.times)),
        "time_start": str(inventory.times[0]) if len(inventory.times) else None,
        "time_end": str(inventory.times[-1]) if len(inventory.times) else None,
        "latitude": {
            "size": int(inventory.lat_values.size),
            "first": float(inventory.lat_values[0]),
            "last": float(inventory.lat_values[-1]),
        },
        "longitude": {
            "size": int(inventory.lon_values.size),
            "first": float(inventory.lon_values[0]),
            "last": float(inventory.lon_values[-1]),
        },
        "variables": [
            {
                "name": spec.name,
                "dtype": spec.dtype,
                "dims": list(spec.dims),
                "shape_without_time": list(spec.shape_without_time),
                "direct_compatible": spec.direct_compatible,
            }
            for spec in inventory.variables.values()
        ],
    }


def _store_summary(path: Path) -> dict[str, Any]:
    info = inspect_store(path)
    return {
        "zarr_format": info.zarr_format,
        "dimensions": {str(name): int(size) for name, size in info.dimensions.items()},
        "variables": [
            {
                "name": variable.name,
                "dims": list(variable.dims),
                "shape": list(variable.shape),
                "dtype": str(variable.dtype),
                "chunks": list(variable.chunks),
                "is_coord": bool(variable.is_coord),
            }
            for variable in info.variables
        ],
    }


def _run_one(
    spec: dict[str, Any],
    *,
    inputs_root: Path,
    work_root: Path,
    logs_root: Path,
) -> dict[str, Any]:
    name = str(spec["name"])
    log_path = logs_root / f"{name}.log"
    output = work_root / "outputs" / f"{name}.zarr"
    if output.exists():
        shutil.rmtree(output)
    started = time.perf_counter()
    base: dict[str, Any] = {
        "name": name,
        "status": "failed",
        "source_root": str(Path(spec["source_root"]).expanduser()),
        "sample_files": list(spec["files"]),
        "sample_links_only": True,
        "delete_after_test": bool(spec.get("delete_after_test", False)),
        "worker_policy": "fixed single worker; bounded correctness smoke, not a benchmark",
        "log": str(log_path),
    }
    with log_path.open("w", encoding="utf-8") as log, redirect_stdout(log), redirect_stderr(log):
        try:
            input_dir = _prepare_links(spec, inputs_root)
            inspect_kwargs: dict[str, Any] = {
                "input_dir": input_dir,
                "mode": spec["mode"],
                "engine": spec["engine"],
                "workers": 1,
                "recursive": False,
            }
            if spec.get("template"):
                inspect_kwargs["template"] = spec["template"]
            if spec.get("field_values"):
                inspect_kwargs["field_values"] = tuple(spec["field_values"])
            inspection = inspect_source(SourceInspectionConfig(**inspect_kwargs))
            inventory = inspection.source_inventory
            variables = tuple(str(item) for item in spec["variables"])
            missing = [name for name in variables if name not in inventory.variables]
            if missing:
                raise KeyError(f"requested variables are absent: {missing}")
            not_compatible = [
                name for name in variables if not inventory.variables[name].direct_compatible
            ]
            if not_compatible:
                raise ValueError(f"requested variables are not direct-compatible: {not_compatible}")
            selection = _selection(inventory, variables)
            conversion_started = time.perf_counter()
            if inventory.source_mode == "filename":
                plan, metrics = convert_filename(
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
            else:
                plan, metrics = convert(
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
            base.update(
                {
                    "status": "passed",
                    "input_dir": str(input_dir),
                    "inventory": _inventory_summary(inventory),
                    "selection": asdict(selection),
                    "plan": plan.label(),
                    "metrics": metrics,
                    "conversion_elapsed": time.perf_counter() - conversion_started,
                    "output": str(output),
                    "output_summary": _store_summary(output),
                }
            )
        except Exception as exc:  # report every dataset while preserving other samples
            print(traceback.format_exc())
            base.update(
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    base["elapsed"] = time.perf_counter() - started
    return base


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--inputs-root", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--logs-root", type=Path, default=DEFAULT_LOGS)
    args = parser.parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    args.inputs_root.mkdir(parents=True, exist_ok=True)
    args.results_root.mkdir(parents=True, exist_ok=True)
    args.work_root.mkdir(parents=True, exist_ok=True)
    args.logs_root.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    datasets = [
        _run_one(
            spec,
            inputs_root=args.inputs_root,
            work_root=args.work_root,
            logs_root=args.logs_root,
        )
        for spec in manifest["datasets"]
    ]
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "source_data_policy": "read-only; no source file is copied or modified",
        "sample_policy": "two filename samples or one complete-file sample; centered 32x32 window; at most two time steps",
        "datasets": datasets,
        "passed": sum(item["status"] == "passed" for item in datasets),
        "failed": sum(item["status"] != "passed" for item in datasets),
        "elapsed": time.perf_counter() - started,
        "deletable_artifacts": [
            str(args.inputs_root),
            str(args.work_root),
            str(args.logs_root),
            str(args.results_root),
        ],
    }
    destination = args.results_root / "external_sample_report.json"
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    temporary.replace(destination)
    print(json.dumps({"report": str(destination), "passed": report["passed"], "failed": report["failed"]}, ensure_ascii=False))
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
