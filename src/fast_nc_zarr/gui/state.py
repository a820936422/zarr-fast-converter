from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GuiSessionState:
    """Single source of truth for cross-page GUI state."""

    current_page: str = "inspection"
    inspection: Any = None
    time_inspection: Any = None
    plan: Any = None
    recovery: Any = None
    task_label: str | None = None
    task_status: str = "idle"

    @property
    def has_inspection(self) -> bool:
        return self.inspection is not None

    @property
    def has_plan(self) -> bool:
        return self.plan is not None

    @property
    def task_running(self) -> bool:
        return self.task_status == "running"

    def set_inspection(self, result: Any) -> None:
        self.inspection = result
        self.time_inspection = None
        self.plan = None
        self.recovery = getattr(result, "recovery", None)

    def set_time_inspection(self, result: Any) -> None:
        self.time_inspection = result

    def set_plan(self, plan: Any) -> None:
        self.plan = plan

    def invalidate_inspection(self) -> None:
        self.inspection = None
        self.time_inspection = None
        self.plan = None
        self.recovery = None

    def start_task(self, label: str) -> None:
        self.task_label = label
        self.task_status = "running"

    def finish_task(self, status: str) -> None:
        self.task_status = status
