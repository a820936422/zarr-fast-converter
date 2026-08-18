from __future__ import annotations

import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np  # noqa: E402

from fast_nc_zarr.hardware import HardwareProfile, StorageBenchmark  # noqa: E402
from fast_nc_zarr.models import (  # noqa: E402
    ConversionPlan,
    FileRecord,
    Inventory,
    Selection,
    VariableSpec,
)
from fast_nc_zarr.performance_model import (  # noqa: E402
    estimate_plan,
    keep_full_worker_sweep,
    prune_candidates,
    rank_candidates,
)
from fast_nc_zarr.planner import (  # noqa: E402
    candidate_plans,
    initial_plan,
    performance_model_enabled,
)
from fast_nc_zarr.system import EffectiveResourceBudget, StorageProfile  # noqa: E402


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


def _planner_inventory() -> Inventory:
    spec = VariableSpec(
        "gpp",
        ("time", "lat", "lon"),
        "float32",
        (100, 100),
        (1, 100, 100),
    )
    files = [
        FileRecord(
            Path("/data/a.nc"),
            10 * 1024 * 1024,
            (0, 1, 2, 3),
            ("a",),
            "h1",
            "h2",
            100,
            100,
            (spec,),
        ),
        FileRecord(
            Path("/data/b.nc"),
            10 * 1024 * 1024,
            (4, 5, 6, 7),
            ("a",),
            "h1",
            "h2",
            100,
            100,
            (spec,),
        ),
    ]
    return Inventory(
        input_dir=Path("/data"),
        files=files,
        lat_values=np.arange(100, dtype="float32"),
        lon_values=np.arange(100, dtype="float32"),
        times=np.arange(8, dtype="int64"),
        time_keys=tuple(str(value) for value in range(8)),
        variables={"gpp": spec},
        source_engine="netcdf4",
        source_dimensions=("time", "lat", "lon"),
        frequency="daily",
        gaps=[],
        total_bytes=20 * 1024 * 1024,
    )


def _budget() -> EffectiveResourceBudget:
    storage = StorageProfile(
        Path("/data"),
        "/dev/sda",
        True,
        "ext4",
        "hdd",
        True,
        "high",
        None,
        (),
    )
    return EffectiveResourceBudget(
        cpu_physical=8,
        cpu_logical=16,
        cpu_effective=16,
        memory_available_bytes=32 * 1024**3,
        memory_total_bytes=32 * 1024**3,
        memory_budget_bytes=24 * 1024**3,
        fd_soft_limit=None,
        worker_ceiling=16,
        source_storage=storage,
        output_storage=storage,
        same_device_roles=("source+output",),
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

    def test_performance_model_enabled_by_default(self) -> None:
        previous = os.environ.pop("FAST_NC_ZARR_PERF_MODEL", None)
        try:
            self.assertTrue(performance_model_enabled())
            os.environ["FAST_NC_ZARR_PERF_MODEL"] = "1"
            self.assertTrue(performance_model_enabled())
            os.environ["FAST_NC_ZARR_PERF_MODEL"] = "on"
            self.assertTrue(performance_model_enabled())
            os.environ["FAST_NC_ZARR_PERF_MODEL"] = "0"
            self.assertFalse(performance_model_enabled())
            os.environ["FAST_NC_ZARR_PERF_MODEL"] = "off"
            self.assertFalse(performance_model_enabled())
            os.environ["FAST_NC_ZARR_PERF_MODEL"] = "FALSE"
            self.assertFalse(performance_model_enabled())
        finally:
            if previous is None:
                os.environ.pop("FAST_NC_ZARR_PERF_MODEL", None)
            else:
                os.environ["FAST_NC_ZARR_PERF_MODEL"] = previous

    def test_keep_full_worker_sweep_preserves_all_worker_counts(self) -> None:
        base = ConversionPlan("chunk", 1, 2, 16, 16, task_batch=1)
        worker_values = list(range(1, 7))
        candidates = [
            replace(base, workers=workers) for workers in worker_values
        ]
        candidates.extend(
            [
                replace(base, compression="lz4", shuffle="shuffle"),
                replace(base, chunk_time=1, chunk_lat=8, chunk_lon=8),
                replace(base, task_batch=4),
            ]
        )
        kept = keep_full_worker_sweep(
            candidates,
            base,
            worker_values,
            _inventory(),
            _selection(),
            _profile(),
            non_worker_budget=2,
        )
        sweep_targets = {replace(base, workers=workers) for workers in worker_values}
        kept_workers = sorted(
            plan.workers for plan in kept if plan in sweep_targets
        )
        self.assertEqual(kept_workers, worker_values)
        self.assertLessEqual(len(kept), len(worker_values) + 2)

    def test_keep_full_worker_sweep_orders_worker_sweep_by_estimate(self) -> None:
        base = ConversionPlan("chunk", 1, 2, 16, 16, task_batch=1)
        worker_values = list(range(1, 7))
        candidates = [
            replace(base, workers=workers) for workers in worker_values
        ]
        ranked = rank_candidates(
            candidates, _inventory(), _selection(), _profile()
        )
        kept = keep_full_worker_sweep(
            candidates,
            base,
            worker_values,
            _inventory(),
            _selection(),
            _profile(),
            non_worker_budget=0,
        )
        self.assertEqual(
            [plan.workers for plan in kept],
            [estimate.plan.workers for estimate in ranked],
        )

    def test_keep_full_worker_sweep_budget_caps_other_candidates(self) -> None:
        base = ConversionPlan("chunk", 1, 2, 16, 16, task_batch=1)
        worker_values = [1, 2]
        candidates = [
            replace(base, workers=workers) for workers in worker_values
        ]
        candidates.extend(
            [
                replace(base, compression="lz4", shuffle="shuffle"),
                replace(base, chunk_time=1, chunk_lat=8, chunk_lon=8),
                replace(base, task_batch=4),
            ]
        )
        kept = keep_full_worker_sweep(
            candidates,
            base,
            worker_values,
            _inventory(),
            _selection(),
            _profile(),
            non_worker_budget=1,
        )
        self.assertEqual(len(kept), len(worker_values) + 1)

    def test_candidate_plans_prunes_with_cached_profile_by_default(self) -> None:
        inventory = _planner_inventory()
        selection = Selection(("gpp",), 0, 8, 0, 100, 0, 100)
        budget = _budget()
        profile = _profile()
        previous = os.environ.pop("FAST_NC_ZARR_PERF_MODEL", None)
        try:
            with patch(
                "fast_nc_zarr.planner.load_cached_profile", return_value=profile
            ):
                pruned = candidate_plans(
                    inventory,
                    selection,
                    Path("/out"),
                    max_workers=16,
                    reserve_gib=2.0,
                    resource_budget=budget,
                )
                os.environ["FAST_NC_ZARR_PERF_MODEL"] = "0"
                unpruned = candidate_plans(
                    inventory,
                    selection,
                    Path("/out"),
                    max_workers=16,
                    reserve_gib=2.0,
                    resource_budget=budget,
                )
        finally:
            if previous is None:
                os.environ.pop("FAST_NC_ZARR_PERF_MODEL", None)
            else:
                os.environ["FAST_NC_ZARR_PERF_MODEL"] = previous
        self.assertTrue(set(range(1, 17)).issubset({plan.workers for plan in pruned}))
        self.assertLess(len(pruned), len(unpruned))

    def test_initial_plan_uses_profile_bandwidth_for_storage_workers(self) -> None:
        inventory = _planner_inventory()
        selection = Selection(("gpp",), 0, 8, 0, 100, 0, 100)
        budget = _budget()
        fast_hdd = HardwareProfile(
            cpu_physical=8,
            cpu_logical=16,
            cpu_effective=16,
            worker_ceiling=16,
            memory_total_bytes=32 * 1024**3,
            memory_available_bytes=24 * 1024**3,
            storage={
                "output": StorageBenchmark(
                    "hdd", "ext4", 180.0, 200.0, 120.0, 64, 1.0
                )
            },
        )
        slow_hdd = HardwareProfile(
            cpu_physical=8,
            cpu_logical=16,
            cpu_effective=16,
            worker_ceiling=16,
            memory_total_bytes=32 * 1024**3,
            memory_available_bytes=24 * 1024**3,
            storage={
                "output": StorageBenchmark(
                    "hdd", "ext4", 80.0, 70.0, 100.0, 64, 1.0
                )
            },
        )
        with patch(
            "fast_nc_zarr.planner.load_cached_profile", return_value=fast_hdd
        ):
            fast_plan = initial_plan(
                inventory,
                selection,
                Path("/out"),
                reserve_gib=2.0,
                resource_budget=budget,
            )
        with patch(
            "fast_nc_zarr.planner.load_cached_profile", return_value=slow_hdd
        ):
            slow_plan = initial_plan(
                inventory,
                selection,
                Path("/out"),
                reserve_gib=2.0,
                resource_budget=budget,
            )
        self.assertEqual(fast_plan.workers, 8)
        self.assertEqual(slow_plan.workers, 4)
        self.assertIn("实测带宽", " ".join(fast_plan.rationale))


if __name__ == "__main__":
    unittest.main()
