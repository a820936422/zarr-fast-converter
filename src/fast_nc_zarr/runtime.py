from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing.context import BaseContext


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
