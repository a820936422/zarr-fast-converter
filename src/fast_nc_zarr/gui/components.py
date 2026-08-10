from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QGroupBox,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)


class SectionCard(QGroupBox):
    """Card-like group box with an optional helper description."""

    def __init__(self, title: str, description: str = "", parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self.setObjectName("sectionCard")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(2, 8, 2, 2)
        self.body.setSpacing(8)
        if description:
            helper = QLabel(description)
            helper.setObjectName("helperText")
            helper.setWordWrap(True)
            self.body.addWidget(helper)


class StatusBadge(QLabel):
    """Text and color status indicator; color is never the only signal."""

    def __init__(self, text: str = "未开始", status: str = "neutral", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusBadge")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumWidth(72)
        self.set_status(text, status)

    def set_status(self, text: str, status: str = "neutral") -> None:
        if status not in {"success", "warning", "danger", "info", "neutral"}:
            status = "neutral"
        self.setText(text)
        self.setProperty("status", status)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()


class Stepper(QFrame):
    """Compact horizontal workflow stepper."""

    def __init__(self, steps: Iterable[str], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stepper")
        self._labels: list[QLabel] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(6)
        for index, step in enumerate(steps, start=1):
            label = QLabel(f"{index}  {step}")
            label.setObjectName("stepperItem")
            label.setProperty("stepState", "pending")
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setMinimumHeight(28)
            layout.addWidget(label, 1)
            self._labels.append(label)

    def set_current(self, index: int, *, completed_before: bool = True) -> None:
        for position, label in enumerate(self._labels):
            if position < index and completed_before:
                state = "complete"
            elif position == index:
                state = "current"
            else:
                state = "pending"
            label.setProperty("stepState", state)
            self.style().unpolish(label)
            self.style().polish(label)
            label.update()


class MetricCard(QFrame):
    """Small metric card used by the task monitor."""

    def __init__(self, title: str, value: str = "—", detail: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 9, 12, 9)
        layout.setSpacing(2)
        self.title = QLabel(title)
        self.title.setObjectName("metricTitle")
        self.value = QLabel(value)
        self.value.setObjectName("metricValue")
        self.detail = QLabel(detail)
        self.detail.setObjectName("metricDetail")
        self.detail.setWordWrap(True)
        layout.addWidget(self.title)
        layout.addWidget(self.value)
        layout.addWidget(self.detail)

    def set_value(self, value: str, detail: str = "") -> None:
        self.value.setText(value)
        self.detail.setText(detail)


class PlanSummary(QTextBrowser):
    """Structured-looking summary browser with a stable text API."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("summaryCard")
        self.setReadOnly(True)
        self.setOpenExternalLinks(False)
        self.setPlaceholderText("尚未生成处理计划。")

    def set_summary(self, title: str, lines: Iterable[str]) -> None:
        content = [f"========== {title} =========="]
        content.extend(str(line) for line in lines)
        self.setPlainText("\n".join(content))


class ProgressHeader(QFrame):
    """Task header combining label, status and progress."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel("当前没有运行中的任务。")
        self.label.setObjectName("pageSubtitle")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        layout.addWidget(self.label)
        layout.addWidget(self.progress)

    def set_running(self, label: str) -> None:
        self.label.setText(label)
        self.progress.setVisible(True)
        self.progress.setValue(0)

    def set_finished(self, label: str) -> None:
        self.label.setText(label)
        self.progress.setVisible(False)
