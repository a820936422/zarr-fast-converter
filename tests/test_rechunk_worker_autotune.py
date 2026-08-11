from __future__ import annotations

import unittest

from fast_nc_zarr.pipeline.models import PipelineChunkingOptions
from fast_nc_zarr.rechunking.autotune import (
    WorkerTrial,
    benchmark_worker_candidates,
    select_worker_trial,
    worker_candidates,
)


class WorkerAutotuneTests(unittest.TestCase):
    def test_pipeline_chunking_defaults_to_auto(self) -> None:
        self.assertEqual(PipelineChunkingOptions().workers, "auto")

    def test_candidates_include_every_safe_parallel_option(self) -> None:
        self.assertEqual(worker_candidates(1), (1,))
        self.assertEqual(worker_candidates(6), (1, 2, 3, 4, 5, 6))

    def test_cpu_bound_sample_selects_two_or_more(self) -> None:
        trials = tuple(
            WorkerTrial(
                workers=workers,
                status="ok",
                elapsed_seconds=1.0,
                logical_bytes=int(rate * 1024**2),
                throughput_mib_s=rate,
            )
            for workers, rate in ((1, 10.0), (2, 22.0), (4, 23.0))
        )
        selected, _reason = select_worker_trial(trials)
        self.assertEqual(selected, 2)

    def test_failed_candidates_do_not_block_higher_parallel_options(self) -> None:
        def runner(workers: int) -> dict[str, float | int]:
            if workers != 1:
                raise OSError("simulated failure")
            return {"logical_bytes": 10 * 1024**2, "elapsed_seconds": 1.0}

        report = benchmark_worker_candidates(
            "stage2",
            (1, 2, 4),
            runner,
            safe_ceiling=4,
            storage_reason="local",
            sample_tasks=2,
            sample_logical_bytes=20 * 1024**2,
            budget_seconds=10.0,
        )
        self.assertEqual(report.selected_workers, 1)
        self.assertEqual(report.trials[1].status, "failed")
        self.assertEqual(report.trials[2].status, "failed")
        self.assertEqual(report.rejected_candidates, (2, 4))


if __name__ == "__main__":
    unittest.main()
