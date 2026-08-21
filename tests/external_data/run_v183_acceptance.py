#!/usr/bin/env python3
"""Run the v1.8.3 real-data acceptance gate (time-dimension recognition).

Uses the read-only backup corpus under ``/run/media/owen/HDD/数据备份/`` via
the local manifest.  The single case targets the MCD12C1 annual HDF-EOS
classification product:

- ``mcd12c1_auto_time_recognition``: the 24-granule directory must be
  recognised WITHOUT a manually supplied filename template; the filename
  production timestamps (13-digit) must not make the observation field
  ambiguous; the theoretical axis must be one point per year (annual
  product, no invented daily cadence); the data-internal coverage from
  ``CoreMetadata.0`` ``RANGEDATETIME`` must be recovered and consistent with
  the reconstructed time.

Generated data lives only under the work root.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import traceback
from typing import Any, Callable

import numpy as np

from fast_nc_zarr.application.services import SourceInspectionConfig, inspect_source

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "manifest.local.json"
DEFAULT_WORK = ROOT / "work" / "v183"
DEFAULT_RESULTS = ROOT / "results"


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


def _base_case(name: str) -> dict[str, Any]:
    return {"case": name, "passed": False}


def _run_case(name: str, function: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    record = _base_case(name)
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


def _case_mcd12c1(spec: dict[str, Any], work_root: Path) -> dict[str, Any]:
    links = work_root / "inputs" / "mcd12c1"
    if links.exists():
        for path in links.iterdir():
            if path.is_symlink():
                path.unlink()
    links.mkdir(parents=True, exist_ok=True)
    source_root = Path(spec["source_root"]).expanduser().resolve()
    names = sorted(
        path.name for path in source_root.glob("*.hdf") if path.is_file()
    )
    if not names:
        raise FileNotFoundError(f"no HDF granules under {source_root}")
    for name in names:
        _replace_link(links / name, source_root / name)

    # Deliberately NO template/field_values: the whole point is automatic
    # recognition of the observation field among production timestamps.
    result = inspect_source(
        SourceInspectionConfig(
            links,
            mode="filename",
            engine=spec.get("engine", "auto"),
            workers=1,
        )
    )
    inventory = result.inventory
    years = sorted({int(str(value)[:4]) for value in inventory.times})
    if inventory.filename_template != "doy":
        raise AssertionError(
            f"expected automatic doy template, got {inventory.filename_template}"
        )
    if len(inventory.times) != len(years):
        raise AssertionError(
            f"annual product must have one point per year: "
            f"{len(inventory.times)} times over {len(years)} years"
        )
    if inventory.missing_time_keys:
        raise AssertionError(
            f"no gaps expected for contiguous annual granules: "
            f"{inventory.missing_time_keys}"
        )
    if "年度" not in inventory.frequency:
        raise AssertionError(f"expected annual frequency, got {inventory.frequency!r}")
    if inventory.coverage_start != "2001-01-01":
        raise AssertionError(
            f"expected CoreMetadata coverage 2001-01-01, got {inventory.coverage_start!r}"
        )
    if inventory.internal_time_source != "hdf_eos_core_metadata":
        raise AssertionError(
            f"expected hdf_eos_core_metadata source, got {inventory.internal_time_source!r}"
        )
    inconsistent = [
        warning
        for warning in result.warnings
        if "不一致" in warning
    ]
    if inconsistent:
        raise AssertionError(f"coverage mismatch warnings: {inconsistent}")
    if not any("覆盖区间" in line for line in result.report.splitlines()):
        raise AssertionError("inventory report must surface the coverage interval")
    expected_first = inventory.coverage_start
    actual_first = str(inventory.times[0])[:10]
    if actual_first != expected_first:
        raise AssertionError(
            f"filename-reconstructed time {actual_first} != CoreMetadata "
            f"{expected_first}"
        )
    expected_excluded = {
        "Land_Cover_Type_1_Percent",
        "Land_Cover_Type_2_Percent",
        "Land_Cover_Type_3_Percent",
    }
    if set(inventory.excluded_extra_dimension_variables) != expected_excluded:
        raise AssertionError(
            f"expected excluded class-dimension variables "
            f"{sorted(expected_excluded)}, got "
            f"{inventory.excluded_extra_dimension_variables}"
        )
    if not any(
        "额外维度" in warning for warning in result.warnings
    ):
        raise AssertionError("expected excluded-variable warning")
    return {
        "granules": len(names),
        "template": inventory.filename_template,
        "frequency": inventory.frequency,
        "time_count": len(inventory.times),
        "years": years,
        "first_time": str(inventory.times[0])[:10],
        "last_time": str(inventory.times[-1])[:10],
        "missing_time_keys": len(inventory.missing_time_keys),
        "coverage_start": inventory.coverage_start,
        "coverage_end": inventory.coverage_end,
        "internal_time_source": inventory.internal_time_source,
        "excluded_extra_dimension_variables": list(
            inventory.excluded_extra_dimension_variables
        ),
        "inventory_variable_count": len(inventory.variables),
        "warnings": list(result.warnings),
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
    sources = [
        (Path(spec["source_root"]).expanduser().resolve() / file)
        for spec in specs.values()
        for file in spec["files"]
    ]
    source_evidence = [
        {"path": str(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sources
        if path.is_file()
    ]

    cases: list[dict[str, Any]] = []
    if "mcd12c1" in specs:
        cases.append(
            _run_case(
                "mcd12c1_auto_time_recognition",
                lambda: _case_mcd12c1(specs["mcd12c1"], args.work_root),
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
    destination = args.results_root / "v183_acceptance_report.json"
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