"""Lightweight performance model for candidate plan pruning.

The v1.7.9 plan introduces a cost model that estimates wall time for a
candidate plan before expensive sample tuning.  It is intentionally coarse:
its purpose is to order/prune candidates, while the real sample benchmark
remains the final arbiter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from .hardware import HardwareProfile
from .models import ConversionPlan, Inventory, Selection

DEFAULT_READ_MIB_S = 100.0
DEFAULT_WRITE_MIB_S = 100.0
DEFAULT_COMPUTE_MIB_S_PER_CORE = 80.0
SPAWN_SECONDS_PER_WORKER = 0.5
COMPRESSION_OVERHEAD = 1.2


@dataclass(frozen=True)
class PlanEstimate:
    plan: ConversionPlan
    read_seconds: float
    write_seconds: float
    compute_seconds: float
    spawn_seconds: float
    total_seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.plan.strategy,
            "workers": self.plan.workers,
            "chunks": list(self.plan.chunks),
            "task_batch": self.plan.task_batch,
            "compression": self.plan.compression,
            "compression_level": self.plan.compression_level,
            "shuffle": self.plan.shuffle,
            "read_seconds": self.read_seconds,
            "write_seconds": self.write_seconds,
            "compute_seconds": self.compute_seconds,
            "spawn_seconds": self.spawn_seconds,
            "total_seconds": self.total_seconds,
        }


def _logical_bytes(inventory: Inventory, selection: Selection) -> int:
    total = 0
    for name in selection.variables:
        spec = inventory.variables[name]
        cells = int(selection.shape[0]) * int(selection.shape[1]) * int(selection.shape[2])
        total += cells * int(np_dtype_itemsize(spec.dtype))
    return max(1, total)


def np_dtype_itemsize(dtype: object) -> int:
    import numpy as np

    return int(np.dtype(dtype).itemsize)


def _bandwidth(
    profile: HardwareProfile | None,
    role: str,
    attr: str,
    default: float,
) -> float:
    if profile is None:
        return default
    benchmark = profile.storage.get(role)
    if benchmark is None:
        return default
    value = float(getattr(benchmark, attr))
    return value if math.isfinite(value) and value > 0 else default


def estimate_plan(
    plan: ConversionPlan,
    inventory: Inventory,
    selection: Selection,
    profile: HardwareProfile | None = None,
) -> PlanEstimate:
    """Estimate wall time for one conversion plan on the given hardware."""
    logical = _logical_bytes(inventory, selection)
    read_mib_s = _bandwidth(profile, "source", "seq_read_mib_s", DEFAULT_READ_MIB_S)
    write_mib_s = _bandwidth(profile, "output", "seq_write_mib_s", DEFAULT_WRITE_MIB_S)
    cpu = profile.cpu_effective if profile is not None else 1
    parallel = max(1, min(int(plan.workers), int(cpu)))
    read_seconds = logical / max(read_mib_s * 1024**2, 1e-9)
    write_seconds = logical / max(
        write_mib_s * 1024**2 * parallel, 1e-9
    )
    compute_mib_s = max(1.0, float(cpu) * DEFAULT_COMPUTE_MIB_S_PER_CORE)
    compute_seconds = (
        logical
        * COMPRESSION_OVERHEAD
        / max(compute_mib_s * 1024**2 * parallel, 1e-9)
    )
    spawn_seconds = max(0.0, float(plan.workers) * SPAWN_SECONDS_PER_WORKER)
    total = read_seconds + write_seconds + compute_seconds + spawn_seconds
    return PlanEstimate(
        plan=plan,
        read_seconds=read_seconds,
        write_seconds=write_seconds,
        compute_seconds=compute_seconds,
        spawn_seconds=spawn_seconds,
        total_seconds=total,
    )


def rank_candidates(
    candidates: Sequence[ConversionPlan],
    inventory: Inventory,
    selection: Selection,
    profile: HardwareProfile | None = None,
) -> list[PlanEstimate]:
    """Return candidates sorted by estimated total time (fastest first)."""
    estimates = [
        estimate_plan(plan, inventory, selection, profile) for plan in candidates
    ]
    return sorted(estimates, key=lambda item: item.total_seconds)


def prune_candidates(
    candidates: Iterable[ConversionPlan],
    inventory: Inventory,
    selection: Selection,
    profile: HardwareProfile | None = None,
    *,
    keep_ratio: float = 0.5,
    minimum: int = 1,
) -> list[ConversionPlan]:
    """Keep the fastest estimated fraction of candidates."""
    estimates = rank_candidates(list(candidates), inventory, selection, profile)
    keep = max(minimum, int(math.ceil(len(estimates) * keep_ratio)))
    return [item.plan for item in estimates[:keep]]
