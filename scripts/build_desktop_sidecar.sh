#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

: "${PYTHON:=python}"
: "${TAURI_TARGET:=${TARGET:-}}" # Optional Rust target triple.
: "${SIDECAR_DIR:=$ROOT/apps/desktop/src-tauri/binaries}"
if [[ -z "$TAURI_TARGET" ]]; then
  TAURI_TARGET="$(rustc -vV | sed -n 's/^host: //p')"
fi

mkdir -p "$SIDECAR_DIR"

# Build the native Python extension first. The desktop worker is intentionally
# launched with the same Python environment as the sidecar.
pixi run native-develop

# Package the Python runtime and application sources as a relocatable sidecar
# when PyInstaller is available. Native builds remain usable without it.
if command -v pyinstaller >/dev/null 2>&1; then
  NATIVE_MODULE="$(python -c 'import fast_nc_zarr._native as n; print(n.__file__)')"
  BUILD_DIR="${PYINSTALLER_BUILD_DIR:-$ROOT/build/fast-nc-zarr-worker}"
  DIST_DIR="${PYINSTALLER_DIST_DIR:-$ROOT/dist}"
  mkdir -p "$BUILD_DIR" "$DIST_DIR"
  pyinstaller --noconfirm --clean --onefile \
    --name fast-nc-zarr-worker \
    --paths src \
    --workpath "$BUILD_DIR" \
    --specpath "$BUILD_DIR" \
    --distpath "$DIST_DIR" \
    --add-binary "$NATIVE_MODULE:fast_nc_zarr" \
    src/fast_nc_zarr/application/desktop_worker/sidecar_main.py
  WORKER="$DIST_DIR/fast-nc-zarr-worker"
else
  echo "pyinstaller is unavailable; using the project Python worker at runtime" >&2
  exit 0
fi

if [[ -n "$TAURI_TARGET" ]]; then
  cp "$WORKER" "$SIDECAR_DIR/fast-nc-zarr-worker-$TAURI_TARGET"
else
  cp "$WORKER" "$SIDECAR_DIR/fast-nc-zarr-worker"
fi
chmod +x "$SIDECAR_DIR"/fast-nc-zarr-worker*
