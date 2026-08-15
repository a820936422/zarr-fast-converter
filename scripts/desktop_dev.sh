#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/desktop_linker_env.sh"
export FAST_NC_ZARR_SOURCE_WORKER=1
exec npm --prefix apps/desktop run tauri:dev
