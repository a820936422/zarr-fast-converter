from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.online_controller import OnlineController  # noqa: E402
from fast_nc_zarr.runtime import apply_cpu_affinity, parse_cpu_affinity  # noqa: E402
from fast_nc_zarr.worker_pool import WorkerPool  # noqa: E402

_POOL_TEST_STATE: dict[str, object] = {}


def _init_pool_test(value: int) -> None:
    _POOL_TEST_STATE["value"] = int(value)


def _pool_test_task(value: int) -> tuple[int, int]:
    return int(_POOL_TEST_STATE.get("value", -1)), int(value) * 2


class CpuAffinityTests(unittest.TestCase):
    def test_parse_cpu_affinity(self) -> None:
        self.assertEqual(parse_cpu_affinity("0-2,5,8-9"), [0, 1, 2, 5, 8, 9])
        self.assertEqual(parse_cpu_affinity(" 4 "), [4])
        self.assertIsNone(parse_cpu_affinity(""))
        self.assertIsNone(parse_cpu_affinity("abc"))

    def test_apply_cpu_affinity_invalid_spec_returns_false(self) -> None:
        self.assertFalse(apply_cpu_affinity("not-a-spec"))


class WorkerPoolTests(unittest.TestCase):
    def test_worker_pool_reuses_executor_across_maps(self) -> None:
        with WorkerPool(max_workers=2) as pool:
            first = list(pool.map(abs, [-1, -2, -3]))
            second = list(pool.map(abs, [4, 5, 6]))
        self.assertEqual(first, [1, 2, 3])
        self.assertEqual(second, [4, 5, 6])

    def test_worker_pool_runs_initializer_in_workers(self) -> None:
        _POOL_TEST_STATE.clear()
        with WorkerPool(
            max_workers=2,
            initializer=_init_pool_test,
            initargs=(7,),
        ) as pool:
            results = list(pool.map(_pool_test_task, [1, 2, 3]))
        self.assertEqual(results, [(7, 2), (7, 4), (7, 6)])

    def test_worker_pool_serial_path_runs_initializer_in_process(self) -> None:
        _POOL_TEST_STATE.clear()
        with WorkerPool(
            max_workers=1,
            initializer=_init_pool_test,
            initargs=(9,),
        ) as pool:
            results = list(pool.map(_pool_test_task, [1, 2, 3]))
        self.assertEqual(results, [(9, 2), (9, 4), (9, 6)])
        self.assertEqual(_POOL_TEST_STATE.get("value"), 9)

    def test_worker_pool_cancel_event_raises(self) -> None:
        cancel_event = threading.Event()
        cancel_event.set()
        with WorkerPool(max_workers=2) as pool:
            with self.assertRaisesRegex(RuntimeError, "任务已取消"):
                list(pool.map(abs, [1, 2, 3], cancel_event=cancel_event))

    def test_worker_pool_submit_and_shutdown(self) -> None:
        pool = WorkerPool(max_workers=2)
        try:
            future = pool.submit(abs, -5)
            self.assertEqual(future.result(timeout=10), 5)
            self.assertTrue(pool.active)
        finally:
            pool.shutdown(terminate=True)
        self.assertFalse(pool.active)

    def test_worker_pool_dynamic_pending_limit_keeps_results(self) -> None:
        calls = []

        def limit() -> int:
            calls.append(1)
            return 1

        with WorkerPool(max_workers=2) as pool:
            results = list(
                pool.map(abs, [-1, -2, -3, -4], pending_limit_fn=limit)
            )
        self.assertEqual(results, [1, 2, 3, 4])
        self.assertGreaterEqual(len(calls), 1)


class OnlineControllerTests(unittest.TestCase):
    def test_io_bound_returns_increase_batch(self) -> None:
        controller = OnlineController(stage="convert", memory_budget_bytes=1024**3)
        action = controller.decide(
            throughput_mib_s=10.0,
            cpu_percent=20.0,
            rss_bytes=100 * 1024**2,
        )
        self.assertEqual(action, "increase_batch")

    def test_memory_pressure_returns_spill_memory(self) -> None:
        controller = OnlineController(stage="convert", memory_budget_bytes=1024**3)
        action = controller.decide(
            throughput_mib_s=10.0,
            cpu_percent=80.0,
            rss_bytes=int(1024**3 * 0.95),
        )
        self.assertEqual(action, "spill_memory")

    def test_cpu_bound_returns_reduce_workers(self) -> None:
        controller = OnlineController(stage="convert", memory_budget_bytes=1024**3)
        action = controller.decide(
            throughput_mib_s=10.0,
            cpu_percent=95.0,
            rss_bytes=100 * 1024**2,
            disk_queue=2.0,
        )
        self.assertEqual(action, "reduce_workers")


if __name__ == "__main__":
    unittest.main()
