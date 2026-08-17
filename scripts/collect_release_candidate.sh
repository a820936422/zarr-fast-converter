#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${TAURI_PROFILE:-release}"
target="${TAURI_TARGET:-${TARGET:-}}"
if [[ -n "$target" ]]; then
  bundle_root="$ROOT/target/$target/$profile/bundle"
else
  bundle_root="$ROOT/target/$profile/bundle"
fi
python scripts/check_sidecar.py


release_dir="$ROOT/release"
mkdir -p "$release_dir"
shopt -s nullglob
rm -f "$release_dir/SHA256SUMS"
packages=("$bundle_root/deb"/*.deb "$bundle_root/rpm"/*.rpm)
if ((${#packages[@]} == 0)); then
  echo "no release packages found under $bundle_root" >&2
  exit 2
fi
for package in "${packages[@]}"; do
  name="$(basename "$package")"
  cp "$package" "$release_dir/$name"
  (cd "$release_dir" && sha256sum "$name") >> "$release_dir/SHA256SUMS"
done
printf 'collected %d package(s) from %s\n' "${#packages[@]}" "$bundle_root"
