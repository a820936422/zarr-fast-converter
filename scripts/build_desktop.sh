#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Pixi's Linux environment exports a conda linker whose sysroot does not
# contain the host GTK/WebKit development libraries used by the Tauri shell.
# Native/Python builds continue to use the conda toolchain; desktop Rust uses
# the host linker and host GUI development libraries.
source "$ROOT/scripts/desktop_linker_env.sh"

has_bundles_arg=false
bundle_spec="${TAURI_BUNDLES:-}"
for ((index = 1; index <= $#; index++)); do
  arg="${!index}"
  if [[ "$arg" == --bundles=* ]]; then
    has_bundles_arg=true
    bundle_spec="${arg#--bundles=}"
  elif [[ "$arg" == "--bundles" ]]; then
    has_bundles_arg=true
    next=$((index + 1))
    bundle_spec="${!next:-}"
  fi
done

bundle_args=()
if [[ "$bundle_spec" == *rpm* ]] && ! command -v rpmbuild >/dev/null 2>&1; then
  echo "RPM bundling requested but rpmbuild is unavailable; install rpm or use TAURI_BUNDLES=deb." >&2
  exit 2
fi

if [[ -z "$bundle_spec" && "$(uname -s)" == "Linux" ]]; then
  bundle_spec="deb"
  if command -v rpmbuild >/dev/null 2>&1; then
    bundle_spec="deb,rpm"
  fi
fi

if [[ "$has_bundles_arg" == false && "$(uname -s)" == "Linux" ]]; then
  bundle_args=(--bundles "$bundle_spec")
  if [[ "$bundle_spec" != *rpm* ]]; then
    echo "rpmbuild is unavailable; building deb only (set TAURI_BUNDLES=deb,rpm to require RPM)." >&2
  fi
fi

if [[ ${#bundle_args[@]} -gt 0 ]]; then
  exec npm --prefix apps/desktop exec tauri build -- "${bundle_args[@]}" "$@"
else
  exec npm --prefix apps/desktop exec tauri build -- "$@"
fi
