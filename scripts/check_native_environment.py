from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _probe_command(name: str) -> dict[str, Any]:
    path = shutil.which(name)
    if path is None:
        return {"available": False, "path": None, "version": None}
    try:
        result = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        output = (result.stdout or result.stderr).strip().splitlines()
        return {
            "available": result.returncode == 0,
            "path": path,
            "version": output[0] if output else None,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "path": path, "version": str(exc)}


def _probe_import(name: str) -> bool:
    try:
        __import__(name)
    except Exception:
        return False
    return True


def main() -> int:
    command_names = (
        "rustc",
        "cargo",
        "maturin",
        "cmake",
        "ninja",
        "pkg-config",
        "patchelf",
        "gdal-config",
        "nc-config",
    )
    commands = {name: _probe_command(name) for name in command_names}
    python_include = Path(sysconfig.get_path("include"))
    required = {
        "python": sys.version.split()[0],
        "python_header": {
            "available": (python_include / "Python.h").is_file(),
            "path": str(python_include / "Python.h"),
        },
        "numpy": {"available": _probe_import("numpy")},
        "commands": commands,
    }
    optional = {
        "hdf5_h5cc": _probe_command("h5cc"),
        "hdf5_pkg_config": {
            "available": False,
            "reason": "hdf5 discovery is optional until the Rust HDF5 adapter phase",
        },
    }
    required_commands = (
        "rustc",
        "cargo",
        "maturin",
        "cmake",
        "ninja",
        "pkg-config",
        "patchelf",
        "gdal-config",
        "nc-config",
    )
    missing = [
        name
        for name in required_commands
        if not commands[name]["available"]
    ]
    if not required["python_header"]["available"]:
        missing.append("Python.h")
    if not required["numpy"]["available"]:
        missing.append("numpy")

    report = {
        "project_root": str(ROOT),
        "python": required["python"],
        "required": required,
        "optional": optional,
        "missing_required": missing,
        "ready_for_p0": not missing,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
