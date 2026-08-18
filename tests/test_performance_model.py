from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.hardware import HardwareProfile, StorageBenchmark  # noqa: E402
from fast_nc_zarr.models import ConversionPlan, Selection  # noqa: E402
from fast_nc_zarr.performance_model import (  # noqa: E402
    estimate_plan,
    prune_candidates,
    rank_candidates,
)


def _inventory() -> SimpleNamespace:
    return SimpleNamespace(
        variables={
            "gpp": SimpleNamespace(dtype="float32"),
        }
    )


def _selection() -> Selection:
    return Selection(
        variables=("gpp",),
        time_start=0,
        time_stop=100,
        lat_start=0,
        lat_stop=1000,
        lon_start=0,
        lon_stop=1000,
    )


def _profile() -> HardwareProfile:
    return HardwareProfile(
        cpu_physical=6,
        cpu_logical=12,
        cpu_effective=12,
        worker_ceiling=6,
        memory_total_bytes=32 * 1024**3,
        memory_available_bytes=20 * 1024**3,
        storage={
            "source": StorageBenchmark(
                "hdd",
                "ext4",
                100.0,
                80.0,
                100.0,
                1,
                0.1,
            ),
            "output": StorageBenchmark(
                "hdd",
                "ext4",
                100.0,
                80.0,
                100.0,
                1,
                0.1,
            ),
        },
    )


class PerformanceModelTests(unittest.TestCase):
    def test_estimate_plan_returns_positive_components(self) -> None:
        plan = ConversionPlan("chunk", 4, 2, 16, 16, task_batch=1)
        estimate = estimate_plan(plan, _inventory(), _selection(), _profile())
        self.assertGreater(estimate.total_seconds, 0)
        self.assertGreater(estimate.read_seconds, 0)
        self.assertGreater(estimate.write_seconds, 0)
        self.assertGreater(estimate.compute_seconds, 0)

    def test_rank_candidates_orders_by_total(self) -> None:
        slow = ConversionPlan("chunk", 1, 1, 8, 8, task_batch=1)
        fast = ConversionPlan("chunk", 6, 2, 16, 16, task_batch=4)
        ranked = rank_candidates([slow, fast], _inventory(), _selection(), _profile())
        self.assertEqual(ranked[0].plan, fast)

    def test_prune_candidates_keeps_fastest_fraction(self) -> None:
        plans = [
            ConversionPlan("chunk", workers, 1, 8, 8, task_batch=1)
            for workers in (1, 2, 4, 6)
        ]
        pruned = prune_candidates(
            plans,
            _inventory(),
            _selection(),
            _profile(),
            keep_ratio=0.5,
            minimum=1,
        )
        self.assertEqual(len(pruned), 2)


if __name__ == "__main__":
    unittest.main()
