#!/usr/bin/env python3
"""Print the local hardware profile (CPU/memory/storage micro-benchmarks)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fast_nc_zarr.hardware import build_hardware_profile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--temporary", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-mib", type=int, default=64)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()
    profile = build_hardware_profile(
        source=args.source,
        temporary=args.temporary,
        output=args.output,
        use_cache=not args.no_cache,
        sample_mib=args.sample_mib,
    )
    print(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
