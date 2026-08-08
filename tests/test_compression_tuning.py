from __future__ import annotations

import json
import shutil
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.rechunking.compression import (  # noqa: E402
    benchmark_compression_candidates,
    generate_compression_candidates,
    make_compression_plan,
    pareto_front,
    relative_log_score,
    select_compression_candidate,
)
from fast_nc_zarr.rechunking.models import CompressionBenchmarkResult  # noqa: E402


ROOT = Path("/tmp/codex_test/fast_nc_zarr_compression_tuning")


def measured(
    plan,
    *,
    write: float = 100.0,
    read: float = 100.0,
    size: int = 1_000,
    verified: bool = True,
    feasible: bool = True,
    index: int = 1,
) -> CompressionBenchmarkResult:
    return CompressionBenchmarkResult(
        plan=plan,
        logical_bytes=2_000,
        compressed_bytes=size,
        write_mib_s=write,
        durable_mib_s=write,
        hot_read_mib_s=read,
        cold_read_mib_s=read,
        average_cpu=50.0,
        peak_rss=100_000,
        sample_count=1,
        verified=verified,
        disk_feasible=feasible,
        success=verified and feasible,
        candidate_index=index,
    )


class CompressionScoringTests(unittest.TestCase):
    def test_default_candidates_are_bounded_and_compare_zstd1_with_lz4(self) -> None:
        candidates = generate_compression_candidates(np.dtype("float32"), (4, 8, 8))
        self.assertLessEqual(len(candidates), 8)
        self.assertIn(("zstd", 1), {(item.codec, item.level) for item in candidates})
        self.assertIn(("blosc-lz4", 1), {(item.codec, item.level) for item in candidates})
        self.assertTrue(all(item.shuffle != "bitshuffle" for item in candidates))

    def test_same_write_and_read_speed_smaller_candidate_dominates(self) -> None:
        large_plan = make_compression_plan("balanced", codec="zstd", level=3)
        small_plan = make_compression_plan(
            "balanced", codec="blosc-zstd", level=3, shuffle="shuffle"
        )
        large = measured(large_plan, size=1_200, index=1)
        small = measured(small_plan, size=800, index=2)
        self.assertEqual(pareto_front((large, small)), (small,))

    def test_balanced_score_rewards_both_speed_and_volume(self) -> None:
        baseline_plan = make_compression_plan("fast", codec="zstd", level=1)
        smaller_plan = make_compression_plan(
            "balanced", codec="blosc-zstd", level=3, shuffle="shuffle"
        )
        baseline = measured(baseline_plan, write=100, read=100, size=1_000)
        smaller = measured(smaller_plan, write=100, read=100, size=500, index=2)
        faster = measured(smaller_plan, write=150, read=150, size=1_000, index=3)
        self.assertGreater(
            relative_log_score(smaller, baseline, objective="balanced"), 0.0
        )
        self.assertGreater(
            relative_log_score(faster, baseline, objective="balanced"), 0.0
        )

    def test_failed_lossless_verification_is_never_selected(self) -> None:
        baseline_plan = make_compression_plan("fast", codec="zstd", level=1)
        invalid_plan = make_compression_plan(
            "balanced", codec="blosc-zstd", level=3, shuffle="shuffle"
        )
        baseline = measured(baseline_plan, index=1)
        invalid = measured(invalid_plan, write=1_000, read=1_000, size=10, verified=False, index=2)
        self.assertEqual(
            select_compression_candidate((baseline, invalid), objective="balanced"),
            baseline_plan,
        )


class CompressionBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ROOT.parent.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(ROOT, ignore_errors=True)
        ROOT.mkdir(parents=True)

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_small_array_end_to_end_benchmark_is_lossless_and_json_safe(self) -> None:
        values = np.arange(3 * 8 * 10, dtype="float32").reshape(3, 8, 10)
        candidates = generate_compression_candidates(
            values.dtype, (2, 4, 5), max_candidates=2
        )
        report = benchmark_compression_candidates(
            values,
            candidates,
            chunk_shape=(2, 4, 5),
            output_dir=ROOT,
            objective="balanced",
            budget_seconds=10,
            max_samples=1,
        )
        self.assertEqual(len(report.results), 2)
        self.assertTrue(any(item.verified_lossless for item in report.results))
        self.assertIsNotNone(report.selected)
        json.dumps(report.to_dict())
        self.assertFalse(list(ROOT.glob(".compression-tune-*")))

    def test_disk_infeasible_candidates_are_not_selected(self) -> None:
        values = np.arange(64, dtype="int16").reshape(4, 4, 4)
        report = benchmark_compression_candidates(
            values,
            generate_compression_candidates(values.dtype, values.shape, max_candidates=1),
            chunk_shape=values.shape,
            output_dir=ROOT,
            max_samples=1,
            disk_free_bytes=1,
        )
        self.assertIsNone(report.selected)
        self.assertTrue(report.fallback)
        self.assertFalse(report.results[0].disk_feasible)

    def test_multiple_sources_contribute_to_samples_and_disk_estimate(self) -> None:
        first = np.arange(200, dtype="uint8")
        second = np.arange(200, dtype="uint8")
        plan = make_compression_plan("fast", codec="zstd", level=1)

        def fake_trial(values, candidate, path, chunks):
            del candidate, path, chunks
            return {
                "logical_bytes": values.nbytes,
                "compressed_bytes": values.nbytes // 2,
                "encode_seconds": 0.01,
                "write_seconds": 0.01,
                "durable_seconds": 0.01,
                "hot_read_seconds": 0.01,
                "cold_read_seconds": 0.01,
                "decode_seconds": 0.02,
                "average_cpu": 10.0,
                "peak_rss": 1024,
                "verified": True,
            }

        with patch(
            "fast_nc_zarr.rechunking.compression._benchmark_trial",
            side_effect=fake_trial,
        ):
            report = benchmark_compression_candidates(
                first,
                (plan,),
                chunk_shape=(200,),
                output_dir=ROOT,
                max_samples=1,
                disk_free_bytes=249,
                sample_sources=((first, (200,)), (second, (200,))),
            )

        self.assertEqual(report.results[0].sample_count, 2)
        self.assertEqual(report.results[0].logical_bytes, 400)
        self.assertFalse(report.results[0].disk_feasible)
        self.assertIsNone(report.selected)

    def test_pre_cancelled_budget_returns_failure_report_without_trials(self) -> None:
        cancel = threading.Event()
        cancel.set()
        values = np.arange(32, dtype="int16").reshape(2, 4, 4)
        report = benchmark_compression_candidates(
            values,
            chunk_shape=values.shape,
            output_dir=ROOT,
            max_candidates=2,
            cancel_event=cancel,
        )
        self.assertTrue(report.cancelled)
        self.assertTrue(report.fallback)
        self.assertTrue(all(not item.success for item in report.results))
        self.assertFalse(list(ROOT.glob(".compression-tune-*")))


if __name__ == "__main__":
    unittest.main()
