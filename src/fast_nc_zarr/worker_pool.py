"""Reusable process pool building block for the v1.7.9 optimization plan.

The full long-lived worker-pool integration is still being wired into the
conversion/resample/rechunk paths.  This module provides the reusable
``WorkerPool`` abstraction plus tests so the remaining integration can build
on a stable primitive.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing.context import BaseContext
from typing import Callable, Iterable, Iterator, TypeVar

from .runtime import shutdown_process_executor, spawn_context

_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


class WorkerPool:
    """A process pool that can be reused across multiple map calls."""

    def __init__(
        self,
        max_workers: int,
        *,
        mp_context: BaseContext | None = None,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self._mp_context = mp_context
        self._executor: ProcessPoolExecutor | None = None

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=self._mp_context or spawn_context(),
            )
        return self._executor

    def map(
        self,
        function: Callable[[_Task], _Result],
        tasks: Iterable[_Task],
        *,
        max_pending: int | None = None,
    ) -> Iterator[_Result]:
        """Yield results while bounding queued tasks, reusing this pool."""
        executor = self._ensure_executor()
        task_iterator = iter(tasks)
        if self.max_workers == 1:
            for task in task_iterator:
                yield function(task)
            return
        pending_limit = (
            max(self.max_workers, int(max_pending))
            if max_pending is not None
            else self.max_workers * 2
        )
        pending: dict[object, int] = {}
        results: dict[int, _Result] = {}
        next_to_submit = 0
        next_to_yield = 0
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < pending_limit:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    exhausted = True
                    break
                future = executor.submit(function, task)
                pending[future] = next_to_submit
                next_to_submit += 1
            if not pending:
                continue
            completed, _ = wait(set(pending), return_when=FIRST_COMPLETED)
            for future in completed:
                index = pending.pop(future)
                results[index] = future.result()
            while next_to_yield in results:
                yield results.pop(next_to_yield)
                next_to_yield += 1

    def close(self) -> None:
        if self._executor is not None:
            shutdown_process_executor(self._executor, terminate=False)
            self._executor = None

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
