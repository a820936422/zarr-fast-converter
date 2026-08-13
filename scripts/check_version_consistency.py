#!/usr/bin/env python3
"""Verify that every runtime/build manifest uses the canonical VERSION value."""

from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> str:
    return str(json.loads(path.read_text(encoding="utf-8"))["version"])


def _read_toml(path: Path, *keys: str) -> str:
    value: object = tomllib.loads(path.read_text(encoding="utf-8"))
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"missing TOML key {'.'.join(keys)} in {path}")
        value = value[key]
    return str(value)


def _read_python_version(path: Path) -> str:
    match = re.search(
        r"^__version__\s*=\s*['\"]([^'\"]+)['\"]\s*$",
        path.read_text(encoding="utf-8"),
        re.MULTILINE,
    )
    if match is None:
        raise ValueError(f"missing __version__ in {path}")
    return match.group(1)


def _read_rust_runtime_version(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r'version:\s*"([^"]+)"', text)
    if match is None:
        raise ValueError(f"missing runtime version in {path}")
    return match.group(1)


def main() -> int:
    expected = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    values = {
        "VERSION": expected,
        "pixi.toml [workspace].version": _read_toml(ROOT / "pixi.toml", "workspace", "version"),
        "pyproject.toml [project].version": _read_toml(ROOT / "pyproject.toml", "project", "version"),
        "Cargo.toml [workspace.package].version": _read_toml(ROOT / "Cargo.toml", "workspace", "package", "version"),
        "apps/desktop/package.json": _read_json(ROOT / "apps/desktop/package.json"),
        "apps/desktop/src-tauri/tauri.conf.json": _read_json(ROOT / "apps/desktop/src-tauri/tauri.conf.json"),
        "apps/desktop/src-tauri/Cargo.toml": _read_toml(ROOT / "apps/desktop/src-tauri/Cargo.toml", "package", "version"),
        "src/fast_nc_zarr/__init__.py": _read_python_version(ROOT / "src/fast_nc_zarr/__init__.py"),
        "apps/desktop/src-tauri/src/lib.rs": _read_rust_runtime_version(ROOT / "apps/desktop/src-tauri/src/lib.rs"),
    }
    mismatches = {name: value for name, value in values.items() if value != expected}
    for name, value in values.items():
        print(f"{name}={value}")
    if mismatches:
        print("version mismatch:", file=sys.stderr)
        for name, value in mismatches.items():
            print(f"  {name}: {value} != {expected}", file=sys.stderr)
        return 1
    print(f"version consistency check passed: {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
