"""Online feedback controller utilities for the v1.7.9 optimization plan.

This module defines the adjustment model and heuristic rules.  It is designed
to be wired into ``direct_write`` / ``run_resample`` in a later integration
step; the rules themselves are unit-tested here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

AdjustmentAction = Literal[
    "none",
    "reduce_workers",
    "increase_batch",
    "reduce_compression",
    "spill_memory",
]


@dataclass(frozen=True)
class AdjustmentEvent:
    """One runtime adjustment decision recorded for observability."""

    timestamp: str
    stage: str
    action: AdjustmentAction
    reason: str
    throughput_mib_s: float
    cpu_percent: float
    rss_bytes: int


@dataclass
class OnlineController:
    """Conservative heuristic controller for long-running backend stages."""

    stage: str
    memory_budget_bytes: int = 0
    cpu_low_threshold: float = 50.0
    cpu_high_threshold: float = 90.0
    rss_high_ratio: float = 0.9
    events: list[AdjustmentEvent] = field(default_factory=list)

    def decide(
        self,
        *,
        throughput_mib_s: float,
        cpu_percent: float,
        rss_bytes: int,
        disk_queue: float | None = None,
    ) -> AdjustmentAction:
        """Return a conservative adjustment action based on one sample."""
        if self.memory_budget_bytes > 0 and rss_bytes > self.memory_budget_bytes * self.rss_high_ratio:
            return "spill_memory"
        if cpu_percent < self.cpu_low_threshold:
            # Low CPU with active throughput usually means I/O bound.  A larger
            # batch improves sequential locality without increasing workers.
            return "increase_batch"
        if cpu_percent > self.cpu_high_threshold and disk_queue is not None and disk_queue < 1.0:
            return "reduce_compression"
        if cpu_percent > self.cpu_high_threshold:
            return "reduce_workers"
        return "none"

    def record(
        self,
        *,
        action: AdjustmentAction,
        reason: str,
        throughput_mib_s: float,
        cpu_percent: float,
        rss_bytes: int,
    ) -> None:
        self.events.append(
            AdjustmentEvent(
                timestamp=datetime.now(timezone.utc).isoformat(),
                stage=self.stage,
                action=action,
                reason=reason,
                throughput_mib_s=throughput_mib_s,
                cpu_percent=cpu_percent,
                rss_bytes=rss_bytes,
            )
        )
