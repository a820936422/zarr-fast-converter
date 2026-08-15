from __future__ import annotations

import os
from pathlib import Path
import sys


def _configure_frozen_library_path() -> None:
    bundle_root = getattr(sys, "_MEIPASS", None)
    if not bundle_root:
        return
    runtime_lib = Path(bundle_root) / "runtime-libs"
    package_lib = Path(bundle_root) / "fast_nc_zarr"
    paths = [str(path) for path in (runtime_lib, package_lib) if path.is_dir()]
    if not paths:
        return
    existing = os.environ.get("LD_LIBRARY_PATH")
    if existing:
        paths.append(existing)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(paths)


if __name__ == "__main__":
    _configure_frozen_library_path()
    from fast_nc_zarr.application.desktop_worker.worker import main

    raise SystemExit(main())
