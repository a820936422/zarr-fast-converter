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


def _run_case(native, sizes: tuple[int, int, int, int, int], repeats: int, include_json: bool) -> dict[str, object]:
    time_size, source_lat_size, source_lon_size, target_lat_size, target_lon_size = sizes
    values = np.linspace(
        0.0,
        1.0,
        time_size * source_lat_size * source_lon_size,
        dtype="float32",
    ).reshape(time_size, source_lat_size, source_lon_size)
    source_lat = np.linspace(-90.0, 90.0, source_lat_size, dtype="float32")
    source_lon = np.linspace(-180.0, 180.0, source_lon_size, dtype="float32")
    target_lat = np.linspace(-89.5, 89.5, target_lat_size, dtype="float32")
    target_lon = np.linspace(-179.5, 179.5, target_lon_size, dtype="float32")
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

    def writable_buffer_call():
        output = np.empty(
            (time_size, target_lat_size, target_lon_size), dtype="float32"
        )
        shape = native.resample_f32_buffer_into(
            values,
            list(values.shape),
            source_lat,
            source_lon,
            target_lat,
            target_lon,
            "bilinear",
            output,
        )
        return output, shape

    buffer_bytes, buffer_shape = buffer_call()
    buffer_result = np.frombuffer(buffer_bytes, dtype="float32").reshape(buffer_shape)
    writable_result, writable_shape = writable_buffer_call()
    np.testing.assert_allclose(
        writable_result.reshape(writable_shape),
        buffer_result,
        equal_nan=True,
    )
    json_seconds = None
    if include_json:
        json_result = json_call()
        np.testing.assert_allclose(
            np.asarray(json_result["values"], dtype="float32").reshape(json_result["shape"]),
            buffer_result,
            equal_nan=True,
        )
        json_seconds = _measure(json_call, repeats)
    buffer_seconds = _measure(buffer_call, repeats)
    writable_seconds = _measure(writable_buffer_call, repeats)
    typed_input_bytes = sum(
        array.nbytes for array in (values, source_lat, source_lon, target_lat, target_lon)
    )
    result = {
        "shape": list(values.shape),
        "target_shape": list(buffer_shape),
        "repeats": repeats,
        "json_seconds": json_seconds,
        "typed_buffer_seconds": buffer_seconds,
        "writable_typed_buffer_seconds": writable_seconds,
        "writable_vs_typed_speedup": buffer_seconds / max(writable_seconds, 1e-12),
        "json_request_bytes": len(json.dumps(request).encode("utf-8")) if include_json else None,
        "typed_input_bytes": typed_input_bytes,
        "writable_output_bytes": int(writable_result.nbytes),
    }
    if json_seconds is not None:
        result["json_vs_typed_speedup"] = json_seconds / max(buffer_seconds, 1e-12)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-size", type=int, default=4)
    parser.add_argument("--source-lat-size", type=int, default=256)
    parser.add_argument("--source-lon-size", type=int, default=256)
    parser.add_argument("--target-lat-size", type=int, default=128)
    parser.add_argument("--target-lon-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="run small, medium, and large typed-buffer cases",
    )
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

    if args.matrix:
        cases = [
            (2, 64, 64, 32, 32),
            (4, 128, 128, 64, 64),
            (8, 256, 256, 128, 128),
        ]
        results = [
            _run_case(native, case, args.repeats, include_json=index == 0)
            for index, case in enumerate(cases)
        ]
        payload: dict[str, object] = {"matrix": results}
    else:
        case = (
            args.time_size,
            args.source_lat_size,
            args.source_lon_size,
            args.target_lat_size,
            args.target_lon_size,
        )
        payload = _run_case(native, case, args.repeats, include_json=True)
        if payload.get("json_seconds") is not None:
            payload["speedup"] = payload["json_seconds"] / max(
                float(payload["typed_buffer_seconds"]), 1e-12
            )
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
