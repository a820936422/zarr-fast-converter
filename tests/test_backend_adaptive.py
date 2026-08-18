from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.planner import storage_aware_worker_limit  # noqa: E402
from fast_nc_zarr.resampling.autotune import resolve_auto_space_workers  # noqa: E402

GIB = 1024**3


def _budget(worker_ceiling: int, medium: str = "unknown") -> SimpleNamespace:
    return SimpleNamespace(
        worker_ceiling=worker_ceiling,
        source_storage=SimpleNamespace(medium=medium),
        same_device_roles=(),
    )


class StorageAwareWorkerLimitTests(unittest.TestCase):
    def test_hdd_large_files_caps_workers(self) -> None:
        budget = _budget(12, medium="hdd")
        self.assertEqual(
            storage_aware_worker_limit(budget, "large-files", same_device=True),
            2,
        )
        self.assertEqual(
            storage_aware_worker_limit(budget, "large-files", same_device=False),
            4,
        )

    def test_hdd_balanced_and_many_small_files_allow_more_workers(self) -> None:
        budget = _budget(12, medium="hdd")
        self.assertEqual(
            storage_aware_worker_limit(budget, "balanced", same_device=True),
            4,
        )
        self.assertEqual(
            storage_aware_worker_limit(budget, "many-small-files", same_device=False),
            8,
        )

    def test_ssd_keeps_cpu_ceiling(self) -> None:
        budget = _budget(12, medium="ssd")
        self.assertEqual(
            storage_aware_worker_limit(budget, "large-files", same_device=False),
            12,
        )

    def test_network_is_conservative(self) -> None:
        budget = _budget(12, medium="network")
        self.assertEqual(
            storage_aware_worker_limit(budget, "large-files", same_device=True),
            2,
        )
        self.assertEqual(
            storage_aware_worker_limit(budget, "large-files", same_device=False),
            4,
        )


class ResampleGlobalThreadBudgetTests(unittest.TestCase):
    def test_space_workers_respect_compute_workers_global_budget(self) -> None:
        budget = SimpleNamespace(
            worker_ceiling=8,
            memory_available_bytes=64 * GIB,
            memory_total_bytes=128 * GIB,
        )
        self.assertEqual(
            resolve_auto_space_workers(compute_workers=4, resource_budget=budget),
            2,
        )
        self.assertEqual(
            resolve_auto_space_workers(compute_workers=1, resource_budget=budget),
            8,
        )

    def test_space_workers_never_zero(self) -> None:
        budget = SimpleNamespace(
            worker_ceiling=2,
            memory_available_bytes=64 * GIB,
            memory_total_bytes=128 * GIB,
        )
        self.assertEqual(
            resolve_auto_space_workers(compute_workers=8, resource_budget=budget),
            1,
        )


if __name__ == "__main__":
    unittest.main()
