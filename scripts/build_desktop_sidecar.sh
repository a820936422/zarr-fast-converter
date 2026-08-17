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
if [[ -n "$TAURI_TARGET" ]]; then
  SIDECAR_PATH="$SIDECAR_DIR/fast-nc-zarr-worker-$TAURI_TARGET"
else
  SIDECAR_PATH="$SIDECAR_DIR/fast-nc-zarr-worker"
fi
rm -f "$SIDECAR_PATH" "$SIDECAR_PATH.sha256" "$SIDECAR_PATH.build.json"

# Build the native Python extension first. The desktop worker is intentionally
# launched with the same Python environment as the sidecar.
pixi run native-develop

if ! command -v pyinstaller >/dev/null 2>&1; then
  echo "pyinstaller is required to build the desktop sidecar" >&2
  exit 2
fi

# Package the Python runtime and application sources as a relocatable sidecar.
NATIVE_MODULE="$($PYTHON -c 'import fast_nc_zarr._native as n; print(n.__file__)')"
BUILD_DIR="${PYINSTALLER_BUILD_DIR:-$ROOT/build/fast-nc-zarr-worker}"
DIST_DIR="${PYINSTALLER_DIST_DIR:-$ROOT/dist}"
RUNTIME_LIB_DIR="$BUILD_DIR/runtime-libs"
mkdir -p "$BUILD_DIR" "$DIST_DIR"
rm -rf "$RUNTIME_LIB_DIR"
"$PYTHON" scripts/collect_sidecar_libs.py \
  --output "$RUNTIME_LIB_DIR" \
  --module netCDF4 \
  --module h5py \
  --module rasterio \
  --module numpy \
  --module scipy \
  --module fast_nc_zarr._native
runtime_binary_args=()
for library in "$RUNTIME_LIB_DIR"/*; do
  [[ -f "$library" ]] || continue
  runtime_binary_args+=(--add-binary "$library:runtime-libs")
done
pyinstaller --noconfirm --clean --onefile \
  --name fast-nc-zarr-worker \
  --paths src \
  --workpath "$BUILD_DIR" \
  --specpath "$BUILD_DIR" \
  --distpath "$DIST_DIR" \
  --add-binary "$NATIVE_MODULE:fast_nc_zarr" \
  "${runtime_binary_args[@]}" \
  src/fast_nc_zarr/application/desktop_worker/sidecar_main.py
WORKER="$DIST_DIR/fast-nc-zarr-worker"
test -x "$WORKER"
cp "$WORKER" "$SIDECAR_PATH"
chmod +x "$SIDECAR_PATH"
(cd "$(dirname "$SIDECAR_PATH")" && sha256sum "$(basename "$SIDECAR_PATH")") > "$SIDECAR_PATH.sha256"
"$PYTHON" scripts/check_sidecar.py --path "$SIDECAR_PATH" --write
echo "built and validated sidecar: $SIDECAR_PATH"
