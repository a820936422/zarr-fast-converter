#!/usr/bin/env python3
"""Measure resampling worker counts with wall time, throughput, and RSS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import threading
import time

from fast_nc_zarr.rechunking.autotune import benchmark_worker_candidates
from fast_nc_zarr.resampling.engine import run_resample
from fast_nc_zarr.resampling.models import ResampleConfig


def _rss_bytes() -> int:
    try:
        import psutil

        root = psutil.Process()
        processes = [root, *root.children(recursive=True)]
        seen: set[int] = set()
        total = 0
        for process in processes:
            if process.pid in seen:
                continue
            seen.add(process.pid)
            try:
                total += int(process.memory_info().rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return total
    except (ImportError, OSError):
        return 0


def _run_trial(
    source: Path,
    trial_root: Path,
    *,
    workers: int,
    resolution: float,
    method: str,
    skipna: bool,
) -> dict[str, float | int]:
    output = trial_root / f"output-{workers}.zarr"
    temporary = trial_root / f"temporary-{workers}"
    shutil.rmtree(output, ignore_errors=True)
    shutil.rmtree(temporary, ignore_errors=True)
    samples = [_rss_bytes()]
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.05):
            samples.append(_rss_bytes())

    thread = threading.Thread(target=monitor, name=f"rss-calibration-{workers}", daemon=True)
    started = time.perf_counter()
    thread.start()
    try:
        metrics = run_resample(
            ResampleConfig(
                input=source,
                output=output,
                resolution=resolution,
                method=method,
                skipna=skipna,
                space_workers=workers,
                temporary_dir=temporary,
            ),
            progress=False,
        )
    finally:
        stop.set()
        thread.join(timeout=2.0)
        samples.append(_rss_bytes())
    elapsed = max(float(metrics.get("wall_seconds", time.perf_counter() - started)), 1e-9)
    logical_bytes = max(0, int(metrics.get("logical_bytes", 0)))
    timing = metrics.get("tile_timing")
    read_seconds = float(timing.get("read_seconds", 0.0)) if isinstance(timing, dict) else 0.0
    write_seconds = float(timing.get("write_seconds", 0.0)) if isinstance(timing, dict) else 0.0
    return {
        "elapsed_seconds": elapsed,
        "logical_bytes": logical_bytes,
        "throughput_mib_s": logical_bytes / 1024**2 / elapsed,
        "peak_rss_bytes": max(samples, default=0),
        "read_seconds": read_seconds,
        "write_seconds": write_seconds,
        "tiles": int(metrics.get("tiles", 0)),
        "backend": str(metrics.get("backend", "python")),
    }


def calibrate(
    source: Path,
    trial_root: Path,
    *,
    max_workers: int = 4,
    resolution: float = 1.0,
    method: str = "bilinear",
    skipna: bool = True,
    budget_seconds: float = 300.0,
    objective: str = "balanced",
) -> dict[str, object]:
    source = source.expanduser().resolve()
    trial_root = trial_root.expanduser().resolve()
    trial_root.mkdir(parents=True, exist_ok=True)
    ceiling = max(1, int(max_workers))
    candidates = tuple(range(1, ceiling + 1))
    measured_trials: list[dict[str, object]] = []

    def runner(workers: int) -> dict[str, float | int]:
        try:
            metrics = _run_trial(
                source,
                trial_root,
                workers=workers,
                resolution=resolution,
                method=method,
                skipna=skipna,
            )
        except Exception as exc:
            measured_trials.append(
                {
                    "workers": workers,
                    "status": "failed",
                    "failure": f"{type(exc).__name__}: {exc}"[:1000],
                }
            )
            raise
        measured_trials.append({"workers": workers, "status": "ok", **metrics})
        return metrics

    report = benchmark_worker_candidates(
        "resampling-rss-calibration",
        candidates,
        runner,
        safe_ceiling=ceiling,
        storage_reason="实测源/目标组合；结果仅用于本机校准，不自动修改生产默认值",
        sample_tasks=0,
        sample_logical_bytes=0,
        budget_seconds=budget_seconds,
        objective=objective,
    )
    payload = report.to_dict()
    payload.update(
        {
            "source": str(source),
            "trial_root": str(trial_root),
            "resolution": float(resolution),
            "method": method,
            "skipna": bool(skipna),
            "calibration_scope": "source/window/target actual run",
            "production_default_changed": False,
            "measured_trials": measured_trials,
            "sample_tasks": max(
                (int(item.get("tiles", 0)) for item in measured_trials if item.get("status") == "ok"),
                default=0,
            ),
            "sample_logical_bytes": max(
                (int(item.get("logical_bytes", 0)) for item in measured_trials if item.get("status") == "ok"),
                default=0,
            ),
        }
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="输入 Zarr v3 目录")
    parser.add_argument("--trial-root", type=Path, required=True, help="校准 trial 根目录")
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--resolution", type=float, default=1.0)
    parser.add_argument("--method", default="bilinear")
    parser.add_argument("--skipna", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--budget-seconds", type=float, default=300.0)
    parser.add_argument("--objective", choices=("speed", "balanced", "compact"), default="balanced")
    args = parser.parse_args()
    if args.max_workers < 1 or args.resolution <= 0 or args.budget_seconds < 0:
        parser.error("max-workers、resolution 和 budget-seconds 必须为正数或零预算")
    print(
        json.dumps(
            calibrate(
                args.source,
                args.trial_root,
                max_workers=args.max_workers,
                resolution=args.resolution,
                method=args.method,
                skipna=args.skipna,
                budget_seconds=args.budget_seconds,
                objective=args.objective,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
