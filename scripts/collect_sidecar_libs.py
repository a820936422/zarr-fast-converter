from __future__ import annotations

import argparse
from collections import deque
import filecmp
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


_DEPENDENCY = re.compile(r"^\s*(\S+)\s+=>\s+(\S+)\s+\(")
_NOT_FOUND = re.compile(r"^\s*(\S+)\s+=>\s+not found")


def _module_native_files(module: str) -> list[Path]:
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, ModuleNotFoundError, ValueError):
        return []
    if spec is None or spec.origin is None or spec.origin in {"built-in", "frozen"}:
        return []
    origin = Path(spec.origin)
    if origin.name == "__init__.py":
        return sorted(path for path in origin.parent.rglob("*.so*") if path.is_file())
    return [origin] if origin.is_file() else []


def _ldd_dependencies(path: Path, conda_lib: Path) -> list[tuple[str, Path]]:
    environment = os.environ.copy()
    old_library_path = environment.get("LD_LIBRARY_PATH")
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        item for item in (str(conda_lib), old_library_path) if item
    )
    result = subprocess.run(
        ["ldd", str(path)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0 and "not a dynamic executable" not in output:
        raise RuntimeError(f"ldd failed for {path}: {output.strip()}")

    dependencies: list[tuple[str, Path]] = []
    for line in output.splitlines():
        match = _DEPENDENCY.match(line)
        if match:
            requested, resolved = match.groups()
            candidate = Path(resolved)
        else:
            missing = _NOT_FOUND.match(line)
            if missing:
                requested = missing.group(1)
                candidate = conda_lib / requested
            else:
                continue
        if candidate.is_file():
            dependencies.append((requested, candidate.resolve()))
    return dependencies


def _inside_prefix(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
    except ValueError:
        return False
    return True


def collect(modules: list[str], output: Path) -> list[Path]:
    conda_prefix = Path(os.environ.get("CONDA_PREFIX", sys.prefix)).resolve()
    conda_lib = conda_prefix / "lib"
    if not conda_lib.is_dir():
        raise RuntimeError(f"Conda library directory does not exist: {conda_lib}")

    roots: list[Path] = []
    for module in modules:
        roots.extend(_module_native_files(module))
    roots = list(dict.fromkeys(path.resolve() for path in roots))
    if not roots:
        raise RuntimeError("no native module roots were found")

    output.mkdir(parents=True, exist_ok=True)
    queue = deque(roots)
    visited: set[Path] = set()
    collected: list[Path] = []
    while queue:
        current = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        for requested, dependency in _ldd_dependencies(current, conda_lib):
            if not _inside_prefix(dependency, conda_prefix) or dependency in visited:
                continue
            destination = output / dependency.name
            if destination.exists():
                if not destination.is_file() or not filecmp.cmp(
                    destination, dependency, shallow=False
                ):
                    raise RuntimeError(
                        f"conflicting bundled libraries share the name {dependency.name}"
                    )
            else:
                shutil.copy2(dependency, destination)
                collected.append(destination)
            alias = output / requested
            if alias != destination and not alias.exists():
                alias.symlink_to(destination.name)
                collected.append(alias)
            queue.append(dependency)
    return collected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect Pixi native libraries for the desktop sidecar"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--module", action="append", dest="modules", required=True)
    args = parser.parse_args()
    collected = collect(args.modules, args.output)
    print(f"collected {len(collected)} native libraries into {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
