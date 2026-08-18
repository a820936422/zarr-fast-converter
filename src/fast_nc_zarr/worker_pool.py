"""Reusable process pool building block for the v1.7.9/v1.8.0 optimization plan.

``WorkerPool`` is the shared process-pool abstraction used by the conversion,
resample, rechunk and pipeline backends.  It replaces ad-hoc
``ProcessPoolExecutor``/``bounded_process_map`` creation so every backend gets
the same bounded, cancellable, ordered-yield behaviour and the same
lifecycle handling.

A pool is created with the worker initializer context for one operation
(``initializer``/``initargs``).  The context is therefore reset per call,
which solves the "worker initialization depends on the output path" problem:
each operation constructs the pool with the paths it needs and closes it in a
``finally`` block, so a stale context can never leak into a later operation.
Pools are also reusable across multiple ``map`` calls inside one operation
(identical worker context), and ``close()`` releases the OS processes when
the operation is idle.
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
        initializer: Callable[..., object] | None = None,
        initargs: tuple[object, ...] = (),
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self._mp_context = mp_context
        self._initializer = initializer
        self._initargs = tuple(initargs)
        self._executor: ProcessPoolExecutor | None = None

    @property
    def active(self) -> bool:
        """Return whether a live executor is currently held by this pool."""
        return self._executor is not None

    def _ensure_executor(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.max_workers,
                mp_context=self._mp_context or spawn_context(),
                initializer=self._initializer,
                initargs=self._initargs,
            )
        return self._executor

    def submit(self, function: Callable[..., _Result], *args: object, **kwargs: object):
        """Submit one task to the underlying executor, reusing this pool."""
        return self._ensure_executor().submit(function, *args, **kwargs)

    def map(
        self,
        function: Callable[[_Task], _Result],
        tasks: Iterable[_Task],
        *,
        max_pending: int | None = None,
        pending_limit_fn: Callable[[], int] | None = None,
        cancel_event=None,
    ) -> Iterator[_Result]:
        """Yield results while bounding queued tasks, reusing this pool.

        Results are yielded in task order.  ``cancel_event`` aborts the map
        with ``RuntimeError("任务已取消。")``; the pool is terminated so no
        stale worker state survives a cancellation.  ``pending_limit_fn`` can
        shrink the in-flight window dynamically (used by the online
        controller to reduce memory pressure).
        """
        task_iterator = iter(tasks)
        if self.max_workers == 1:
            if self._initializer is not None:
                self._initializer(*self._initargs)
            for task in task_iterator:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("任务已取消。")
                yield function(task)
            return

        executor = self._ensure_executor()
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
        terminated = False
        try:
            while pending or not exhausted:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("任务已取消。")
                while not exhausted:
                    current_pending_limit = (
                        pending_limit_fn() if pending_limit_fn is not None else pending_limit
                    )
                    if len(pending) >= current_pending_limit:
                        break
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
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("任务已取消。")
                    index = pending.pop(future)
                    results[index] = future.result()
                while next_to_yield in results:
                    yield results.pop(next_to_yield)
                    next_to_yield += 1
        except BaseException:
            self.shutdown(terminate=True, pending=set(pending))
            terminated = True
            raise
        finally:
            if not terminated:
                # Keep the executor alive for reuse across map calls.
                pass

    def shutdown(self, *, terminate: bool = False, pending: Iterable[object] = ()) -> None:
        """Stop the pool, optionally terminating workers on failure/cancel."""
        if self._executor is not None:
            shutdown_process_executor(
                self._executor, terminate=terminate, pending=pending
            )
            self._executor = None

    def close(self) -> None:
        """Gracefully stop the pool and release worker processes."""
        self.shutdown(terminate=False)

    def __enter__(self) -> "WorkerPool":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
