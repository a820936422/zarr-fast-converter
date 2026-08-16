#!/usr/bin/env python3
"""Compare the legacy JSON bridge with the typed native resampling bridge."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np



def _measure(function, repeats: int) -> float:
    function()
    started = time.perf_counter()
    for _ in range(repeats):
        function()
    return (time.perf_counter() - started) / repeats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-size", type=int, default=4)
    parser.add_argument("--source-lat-size", type=int, default=256)
    parser.add_argument("--source-lon-size", type=int, default=256)
    parser.add_argument("--target-lat-size", type=int, default=128)
    parser.add_argument("--target-lon-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or min(
        args.time_size,
        args.source_lat_size,
        args.source_lon_size,
        args.target_lat_size,
        args.target_lon_size,
    ) < 1:
        parser.error("all sizes and repeats must be positive")

    import fast_nc_zarr._native as native

    values = np.linspace(
        0.0,
        1.0,
        args.time_size * args.source_lat_size * args.source_lon_size,
        dtype="float32",
    ).reshape(args.time_size, args.source_lat_size, args.source_lon_size)
    source_lat = np.linspace(-90.0, 90.0, args.source_lat_size, dtype="float32")
    source_lon = np.linspace(-180.0, 180.0, args.source_lon_size, dtype="float32")
    target_lat = np.linspace(-89.5, 89.5, args.target_lat_size, dtype="float32")
    target_lon = np.linspace(-179.5, 179.5, args.target_lon_size, dtype="float32")
    request = {
        "values": values.reshape(-1).tolist(),
        "shape": list(values.shape),
        "source_lat": source_lat.tolist(),
        "source_lon": source_lon.tolist(),
        "target_lat": target_lat.tolist(),
        "target_lon": target_lon.tolist(),
        "method": "bilinear",
    }

    def json_call():
        return json.loads(native.resample_f32_json(json.dumps(request)))

    def buffer_call():
        return native.resample_f32_buffer(
            values,
            list(values.shape),
            source_lat,
            source_lon,
            target_lat,
            target_lon,
            "bilinear",
        )

    json_result = json_call()
    buffer_bytes, buffer_shape = buffer_call()
    buffer_result = np.frombuffer(buffer_bytes, dtype="float32").reshape(buffer_shape)
    np.testing.assert_allclose(
        np.asarray(json_result["values"], dtype="float32").reshape(json_result["shape"]),
        buffer_result,
        equal_nan=True,
    )
    json_seconds = _measure(json_call, args.repeats)
    buffer_seconds = _measure(buffer_call, args.repeats)
    json_request_bytes = len(json.dumps(request).encode("utf-8"))
    typed_input_bytes = sum(
        array.nbytes for array in (values, source_lat, source_lon, target_lat, target_lon)
    )
    print(
        json.dumps(
            {
                "shape": list(values.shape),
                "target_shape": list(buffer_shape),
                "repeats": args.repeats,
                "json_seconds": json_seconds,
                "typed_buffer_seconds": buffer_seconds,
                "speedup": json_seconds / max(buffer_seconds, 1e-12),
                "json_request_bytes": json_request_bytes,
                "typed_input_bytes": typed_input_bytes,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
