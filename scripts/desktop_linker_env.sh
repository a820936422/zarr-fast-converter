#!/usr/bin/env bash
set -euo pipefail

# Pixi's Linux environment exports a conda linker whose sysroot does not
# contain the host GTK/WebKit development libraries used by the Tauri shell.
# Native/Python builds continue to use the conda toolchain; desktop Rust uses
# the host linker and host GUI development libraries.
if [[ "$(uname -s)" == "Linux" ]]; then
  export CC="${DESKTOP_CC:-/usr/bin/cc}"
  export CXX="${DESKTOP_CXX:-/usr/bin/c++}"
  export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER="${DESKTOP_LINKER:-/usr/bin/cc}"
  export LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/lib:/usr/lib64${LIBRARY_PATH:+:$LIBRARY_PATH}"
fi
