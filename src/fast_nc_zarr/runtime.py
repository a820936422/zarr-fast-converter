from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from multiprocessing.context import BaseContext
from typing import Callable, Iterable, Iterator, TypeVar


_Task = TypeVar("_Task")
_Result = TypeVar("_Result")


_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMEXPR_MAX_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)


def configure_process_runtime(threads_per_worker: int | None = None) -> int:
    """Apply one explicit native-thread budget before spawning workers.

    The application uses process-level parallelism.  Letting every worker also
    start a full OpenMP/BLAS thread pool causes severe oversubscription and can
    exhaust memory.  ``FAST_NC_ZARR_THREADS_PER_WORKER`` remains the single
    documented escape hatch for workloads that benefit from nested threads.
    """

    requested = (
        os.environ.get("FAST_NC_ZARR_THREADS_PER_WORKER", "1")
        if threads_per_worker is None
        else str(threads_per_worker)
    )
    try:
        threads = int(requested)
    except ValueError as exc:
        raise ValueError("FAST_NC_ZARR_THREADS_PER_WORKER 必须是正整数。") from exc
    if threads < 1:
        raise ValueError("FAST_NC_ZARR_THREADS_PER_WORKER 必须是正整数。")
    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = str(threads)
    os.environ.setdefault("ESMF_RUNTIME_LOG_KIND", "NONE")
    return threads


def spawn_context() -> BaseContext:
    """Return a spawn context after configuring inherited native threads."""

    configure_process_runtime()
    return mp.get_context("spawn")


def bounded_process_map(
    function: Callable[[_Task], _Result],
    tasks: Iterable[_Task],
    *,
    workers: int,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[object, ...] = (),
    cancel_event=None,
    max_pending: int | None = None,
) -> Iterator[_Result]:
    """Yield completed process results while bounding queued task state.

    Large archives can contain tens of thousands of files or chunks.  Eagerly
    creating one Future per item retains every task argument in the parent and
    can consume substantial memory before useful work finishes.  The shared
    runner keeps only a small multiple of the worker count in flight and uses
    a direct serial path when multiprocessing would only add spawn overhead.
    """

    worker_count = max(1, int(workers))
    task_iterator = iter(tasks)
    if worker_count == 1:
        if initializer is not None:
            initializer(*initargs)
        for task in task_iterator:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("任务已取消。")
            yield function(task)
        return

    pending_limit = (
        max(worker_count, int(max_pending))
        if max_pending is not None
        else worker_count * 2
    )
    executor = ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=spawn_context(),
        initializer=initializer,
        initargs=initargs,
    )
    pending = set()
    exhausted = False
    terminated = False
    try:
        while pending or not exhausted:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("任务已取消。")
            while not exhausted and len(pending) < pending_limit:
                try:
                    task = next(task_iterator)
                except StopIteration:
                    exhausted = True
                    break
                pending.add(executor.submit(function, task))
            if not pending:
                continue
            completed, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                if cancel_event is not None and cancel_event.is_set():
                    raise RuntimeError("任务已取消。")
                yield future.result()
    except BaseException:
        terminate = getattr(executor, "terminate_workers", None)
        if terminate is not None:
            terminate()
        else:
            for future in pending:
                future.cancel()
            executor.shutdown(wait=False, cancel_futures=True)
        terminated = True
        raise
    finally:
        if not terminated:
            executor.shutdown(wait=True)
