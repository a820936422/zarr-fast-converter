#!/usr/bin/env bash
set -euo pipefail

exec maturin develop --release "$@"
