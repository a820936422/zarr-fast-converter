from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
from PySide6.QtCore import QDate, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
    QLineEdit,
)
from .. import __version__

from ..application.services import (
    ConversionConfig,
    ConversionPreview,
    InspectionResult,
    RechunkConfig,
    RechunkPreview,
    ResampleConfig,
    ResamplePreview,
    SourceInspectionConfig,
    default_inspection_cache_path,
    inspect_source,
    inspect_zarr,
    inspect_temporary_pipeline,
    inspect_resample,
    preview_conversion,
    preview_rechunk,
    preview_resample,
    run_conversion,
    run_rechunk,
    run_resample,
    format_resample_preview,
    save_inspection_snapshot,
    load_inspection_snapshot,
    preview_pipeline,
    run_pipeline,
)
from ..rechunking.planning import DEFAULT_TARGET_MIB, default_workers
from ..models import VariableTransform
from ..pipeline.models import (
    PipelineChunkingOptions,
    PipelineCompressionOptions,
    PipelineConfig,
    PipelineConversionOptions,
    PipelineGeneralConfig,
    PipelineInput,
    PipelineOperations,
    PipelineResamplingOptions,
    ZarrPipelinePlan,
)
from ..selection import parse_list
from ..time_mapping import TimeInspectionResult, TimeRule, TimeFieldOption, inspect_time_metadata
from .workers import TaskWorker

try:
    import pyqtgraph as pg
except ImportError:  # pragma: no cover - optional GUI enhancement
    pg = None


TaskCallback = Callable[[Any], None]


def _date_text(value: Any) -> str:
    if isinstance(value, np.datetime64):
        return str(np.datetime_as_string(value, unit="D"))
    return str(value)[:10]


def _qdate(value: Any) -> QDate:
    result = QDate.fromString(_date_text(value), Qt.DateFormat.ISODate)
    return result if result.isValid() else QDate.currentDate()


def _human_bytes(value: int | float) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(size) < 1024 or unit == "TiB":
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TiB"


def _button(text: str, slot: Callable[[], None]) -> QPushButton:
    button = QPushButton(text)
    button.clicked.connect(slot)
    return button


def _run_cancelable(cancel_event, operation: Callable[[], Any]) -> Any:
    """Run a short synchronous inspection/preview with cancellation gates."""
    if cancel_event.is_set():
        raise RuntimeError("任务已取消。")
    result = operation()
    if cancel_event.is_set():
        raise RuntimeError("任务已取消。")
    return result


class TaskPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.history: list[dict[str, Any]] = []
        self._active_history: dict[str, Any] | None = None
        layout = QVBoxLayout(self)
        title = QLabel("任务与日志")
        title.setObjectName("pageTitle")
        layout.addWidget(title)
        self.status = QLabel("当前没有运行中的任务。")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        self.resource_label = QLabel("资源：暂无运行数据")
        layout.addWidget(self.resource_label)
        self.cpu_plot = None
        self.memory_plot = None
        self.cpu_curve = None
        self.memory_curve = None
        self.resource_times: list[float] = []
        self.resource_cpu: list[float] = []
        self.resource_memory: list[float] = []
        if pg is not None:
            resource_row = QHBoxLayout()
            self.cpu_plot = pg.PlotWidget(title="CPU 使用率")
            self.cpu_plot.setMinimumHeight(110)
            self.cpu_plot.setMaximumHeight(155)
            self.cpu_plot.setLabel("left", "CPU", units="%")
            self.cpu_plot.setLabel("bottom", "运行时间", units="s")
            self.cpu_curve = self.cpu_plot.plot(pen=pg.mkPen("#3daee9", width=2))
            self.memory_plot = pg.PlotWidget(title="内存使用")
            self.memory_plot.setMinimumHeight(110)
            self.memory_plot.setMaximumHeight(155)
            self.memory_plot.setLabel("left", "RSS", units="GiB")
            self.memory_plot.setLabel("bottom", "运行时间", units="s")
            self.memory_curve = self.memory_plot.plot(pen=pg.mkPen("#e67e22", width=2))
            resource_row.addWidget(self.cpu_plot)
            resource_row.addWidget(self.memory_plot)
            layout.addLayout(resource_row)
        layout.addWidget(QLabel("本次任务涉及的磁盘"))
        self.disk_table = QTableWidget(0, 7)
        self.disk_table.setHorizontalHeaderLabels(
            ("用途", "设备", "挂载点", "已用", "可用", "读取", "写入")
        )
        self.disk_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.disk_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.ResizeToContents
        )
        self.disk_table.horizontalHeader().setSectionResizeMode(
            2, QHeaderView.ResizeMode.Stretch
        )
        for column in (3, 4, 5, 6):
            self.disk_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.disk_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.disk_table.setMaximumHeight(125)
        layout.addWidget(self.disk_table)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("检查和写入任务的运行日志会显示在这里。")
        layout.addWidget(self.log, 1)
        actions = QHBoxLayout()
        self.cancel = _button("请求取消", self._cancel_requested)
        self.cancel.setEnabled(False)
        actions.addWidget(self.cancel)
        actions.addStretch(1)
        actions.addWidget(_button("清空日志", self.log.clear))
        actions.addWidget(_button("清空历史", self.clear_history))
        layout.addLayout(actions)
        self.history_table = QTableWidget(0, 4)
        self.history_table.setHorizontalHeaderLabels(("开始时间", "任务", "状态", "耗时"))
        self.history_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setMaximumHeight(170)
        layout.addWidget(QLabel("任务历史（当前会话）"))
        layout.addWidget(self.history_table)
        self.cancel_callback: Callable[[], None] | None = None

    def append(self, message: str) -> None:
        self.log.appendPlainText(message)

    def started(self, label: str, cancel: Callable[[], None]) -> None:
        self.status.setText(f"运行中：{label}")
        self.progress.setVisible(True)
        self.progress.setValue(0)
        self.progress.setFormat("等待可量化进度…")
        self.cancel.setEnabled(True)
        self.cancel_callback = cancel
        self._active_history = {
            "started": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "label": label,
            "status": "运行中",
            "elapsed": 0.0,
            "started_monotonic": time.perf_counter(),
        }
        self.resource_times.clear()
        self.resource_cpu.clear()
        self.resource_memory.clear()
        self.disk_table.setRowCount(0)
        if self.cpu_curve is not None:
            self.cpu_curve.setData([], [])
            self.memory_curve.setData([], [])
        self.append(f"\n===== {label} =====")


    def update_progress(self, completed: int, total: int, detail: str) -> None:
        if total <= 0:
            return
        value = min(1000, max(0, int(round(completed * 1000 / total))))
        self.progress.setRange(0, 1000)
        self.progress.setValue(value)
        self.progress.setFormat(f"%p%  {detail[:96]}")

    def update_resource(self, sample: dict[str, Any]) -> None:
        elapsed = float(sample.get("elapsed", 0.0))
        cpu = float(sample.get("cpu_machine_percent", sample.get("cpu", 0.0)))
        cpu_cores = float(sample.get("cpu_cores", cpu / 100.0))
        memory = float(sample.get("rss_gib", 0.0))
        self.resource_times.append(elapsed)
        self.resource_cpu.append(cpu)
        self.resource_memory.append(memory)
        if self.cpu_curve is not None:
            self.cpu_curve.setData(self.resource_times, self.resource_cpu)
            self.memory_curve.setData(self.resource_times, self.resource_memory)
        disks = sample.get("disks") or []
        self.disk_table.setRowCount(len(disks))
        for row, disk in enumerate(disks):
            values = (
                str(disk.get("roles", "")),
                str(disk.get("device", "")),
                str(disk.get("mountpoint", "")),
                f"{float(disk.get('used_gib', 0.0)):.1f} GiB",
                f"{float(disk.get('free_gib', 0.0)):.1f} GiB",
                f"{float(disk.get('read_mib_s', 0.0)):.1f} MiB/s",
                f"{float(disk.get('write_mib_s', 0.0)):.1f} MiB/s",
            )
            for column, value in enumerate(values):
                self.disk_table.setItem(row, column, QTableWidgetItem(value))
        self.resource_label.setText(
            f"资源：CPU {cpu_cores:.2f} 核 / 整机 {cpu:.1f}%；RSS {memory:.2f} GiB；"
            f"读取 {float(sample.get('read_mib_s', 0.0)):.1f} MiB/s；"
            f"写入 {float(sample.get('write_mib_s', 0.0)):.1f} MiB/s；"
            f"磁盘 {len(disks)} 个"
        )
        if self._active_history is not None:
            self._active_history["elapsed"] = elapsed

    def _finish_history(self, status: str) -> None:
        if self._active_history is None:
            return
        self._active_history["elapsed"] = max(
            float(self._active_history.get("elapsed", 0.0)),
            time.perf_counter() - float(self._active_history["started_monotonic"]),
        )
        self._active_history["status"] = status
        if self._active_history not in self.history:
            self.history.append(self._active_history)
            row = self.history_table.rowCount()
            self.history_table.insertRow(row)
            values = (
                self._active_history["started"],
                self._active_history["label"],
                status,
                f"{float(self._active_history['elapsed']):.1f} 秒",
            )
            for column, value in enumerate(values):
                self.history_table.setItem(row, column, QTableWidgetItem(str(value)))
        self._active_history = None

    def completed(self, message: str = "任务完成。") -> None:
        self.status.setText(message)
        self.progress.setVisible(False)
        self.cancel.setEnabled(False)
        self.cancel_callback = None
        self._finish_history("完成")

    def failed(self) -> None:
        self.status.setText("任务失败；请查看下方详细日志。")
        self.progress.setVisible(False)
        self.cancel.setEnabled(False)
        self.cancel_callback = None
        self._finish_history("失败")

    def cancelled(self) -> None:
        self.status.setText("任务已取消。")
        self.progress.setVisible(False)
        self.cancel.setEnabled(False)
        self.cancel_callback = None
        self._finish_history("已取消")

    def clear_history(self) -> None:
        self.history.clear()
        self.history_table.setRowCount(0)

    def _cancel_requested(self) -> None:
        if self.cancel_callback is not None:
            self.cancel_callback()
            self.cancel.setEnabled(False)
            self.status.setText("正在取消任务；当前 I/O 块完成后停止……")


class TimeRulePanel(QGroupBox):
    """User confirmation widget for complete or composed time fields."""

    def __init__(self, parent=None) -> None:
        super().__init__("确认时间字段", parent)
        form = QFormLayout(self)
        self.combos: dict[str, QComboBox] = {}
        for component, label in (
            ("full", "完整时间"),
            ("year", "年"),
            ("month", "月"),
            ("day", "日"),
            ("doy", "DOY"),
        ):
            combo = QComboBox()
            combo.addItem("不使用", None)
            self.combos[component] = combo
            form.addRow(label, combo)
        self.description = QLabel("请确认完整时间来源，或组合年/月/日/DOY。")
        self.description.setWordWrap(True)
        form.addRow("规则说明", self.description)
        self.combos["full"].currentIndexChanged.connect(self._full_changed)
        self.setEnabled(False)

    def set_result(self, result: TimeInspectionResult) -> None:
        for combo in self.combos.values():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("不使用", None)
        for component, combo in self.combos.items():
            for option in result.options:
                if option.ref.component == component:
                    combo.addItem(option.label, option.ref)
            combo.blockSignals(False)
        if result.suggested_rule is not None:
            rule = result.suggested_rule
            if rule.full is not None:
                self._select_ref("full", rule.full)
            else:
                for component in ("year", "month", "day", "doy"):
                    ref = getattr(rule, component)
                    if ref is not None:
                        self._select_ref(component, ref)
        self.setEnabled(True)
        self._full_changed()

    def rule(self) -> TimeRule:
        refs = {name: combo.currentData() for name, combo in self.combos.items()}
        return TimeRule(
            full=refs["full"],
            year=refs["year"],
            month=refs["month"],
            day=refs["day"],
            doy=refs["doy"],
        )

    def _select_ref(self, component: str, ref) -> None:
        combo = self.combos[component]
        for index in range(combo.count()):
            if combo.itemData(index) == ref:
                combo.setCurrentIndex(index)
                return

    def _full_changed(self) -> None:
        using_full = self.combos["full"].currentData() is not None
        for component in ("year", "month", "day", "doy"):
            self.combos[component].setEnabled(not using_full)
        if using_full:
            self.description.setText("将直接使用所选完整时间，并统一归一化为 YYYY-MM-DD。")
        else:
            self.description.setText("将使用年 + DOY，或年 + 月 + 日构建完整日期。")


class InspectionPage(QWidget):
    task_requested = Signal(str, object, object)
    result_ready = Signal(object)
    zarr_result_ready = Signal(object)
    result_invalidated = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.result: InspectionResult | None = None
        self.time_result: TimeInspectionResult | None = None
        root = QVBoxLayout(self)
        title = QLabel("数据检查模块")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(QLabel("必须先确认文件名与 time 维度如何共同构建完整日期，之后才能检查数据结构和进入转换。"))

        inputs = QGroupBox("检查输入")
        form = QFormLayout(inputs)
        self.input_kind = QComboBox()
        self.input_kind.addItem("原始 NC / HDF / TIFF", "raw")
        self.input_kind.addItem("现有 Zarr v3", "zarr")
        self.input_kind.addItem("临时处理产物", "temporary")
        form.addRow("输入类型", self.input_kind)
        path_row, self.path = self._path_row("选择源目录", directory=True)
        form.addRow("输入目录", path_row)
        self.engine = QComboBox()
        for label, value in (
            ("自动", "auto"),
            ("h5netcdf", "h5netcdf"),
            ("netCDF4", "netcdf4"),
            ("rasterio", "rasterio"),
        ):
            self.engine.addItem(label, value)
        form.addRow("读取引擎", self.engine)
        self.recursive = QCheckBox("递归扫描子目录")
        form.addRow("扫描选项", self.recursive)
        self.inspect_workers = QSpinBox()
        self.inspect_workers.setRange(0, 64)
        self.inspect_workers.setValue(0)
        self.inspect_workers.setSpecialValueText("自动")
        form.addRow("检查进程数", self.inspect_workers)
        root.addWidget(inputs)

        time_actions = QHBoxLayout()
        self.time_check_button = _button("检查文件时间维度信息", self._request_time_inspection)
        time_actions.addWidget(self.time_check_button)
        time_actions.addStretch(1)
        root.addLayout(time_actions)
        self.time_panel = TimeRulePanel()
        root.addWidget(self.time_panel)

        mapping = QGroupBox("源维度映射（非标准名称时填写）")
        mapping_form = QFormLayout(mapping)
        self.source_time_dim = QLineEdit()
        self.source_time_dim.setPlaceholderText("例如 time")
        self.source_lat_dim = QLineEdit()
        self.source_lat_dim.setPlaceholderText("例如 latitude 或 y")
        self.source_lon_dim = QLineEdit()
        self.source_lon_dim.setPlaceholderText("例如 longitude 或 x")
        mapping_form.addRow("源 time 维度", self.source_time_dim)
        mapping_form.addRow("源纬度维度", self.source_lat_dim)
        mapping_form.addRow("源经度维度", self.source_lon_dim)
        mapping.setEnabled(False)
        self.mapping_group = mapping
        root.addWidget(mapping)

        actions = QHBoxLayout()
        self.confirm_time_button = _button("确认时间规则并检查数据结构", self._request_structure_inspection)
        self.confirm_time_button.setEnabled(False)
        actions.addWidget(self.confirm_time_button)
        self.save_button = _button("保存检查快照", self._save_snapshot)
        self.save_button.setEnabled(False)
        actions.addWidget(self.save_button)
        self.load_button = _button("导入检查快照", self._load_snapshot)
        actions.addWidget(self.load_button)
        actions.addStretch(1)
        root.addLayout(actions)

        self.status = QLabel("尚未检查。")
        root.addWidget(self.status)
        self.report = QTextBrowser()
        self.report.setOpenExternalLinks(False)
        root.addWidget(self.report, 1)
        self.path.textChanged.connect(self._invalidate_time_check)
        self.input_kind.currentIndexChanged.connect(self._input_kind_changed)
        self.engine.currentIndexChanged.connect(self._invalidate_time_check)
        self.recursive.toggled.connect(self._invalidate_time_check)
        self.inspect_workers.valueChanged.connect(self._invalidate_structure_check)
        for edit in (self.source_time_dim, self.source_lat_dim, self.source_lon_dim):
            edit.textChanged.connect(self._invalidate_structure_check)
        for combo in self.time_panel.combos.values():
            combo.currentIndexChanged.connect(self._invalidate_structure_check)

    def _input_kind_changed(self, *_args) -> None:
        self._invalidate_time_check()
        kind = self.input_kind.currentData()
        is_zarr = kind == "zarr"
        is_temporary = kind == "temporary"
        is_processed = is_zarr or is_temporary
        self.time_check_button.setText(
            "检查临时处理产物"
            if is_temporary
            else "检查现有 Zarr"
            if is_zarr
            else "检查文件时间维度信息"
        )
        self.engine.setEnabled(not is_processed)
        self.recursive.setEnabled(not is_processed)
        self.inspect_workers.setEnabled(not is_processed)
        self.time_panel.setVisible(not is_processed)
        self.mapping_group.setVisible(not is_processed)
        self.confirm_time_button.setVisible(not is_processed)

    def _path_row(self, dialog_title: str, *, directory: bool) -> tuple[QWidget, Any]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        button = QPushButton("浏览…")
        if directory:
            button.clicked.connect(lambda: self._choose_directory(edit, dialog_title))
        else:
            button.clicked.connect(lambda: self._choose_file(edit, dialog_title))
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row, edit

    @staticmethod
    def _choose_directory(edit, title: str) -> None:
        value = QFileDialog.getExistingDirectory(edit, title, edit.text() or str(Path.cwd()))
        if value:
            edit.setText(value)

    @staticmethod
    def _choose_file(edit, title: str) -> None:
        value, _ = QFileDialog.getOpenFileName(edit, title, edit.text() or str(Path.cwd()))
        if value:
            edit.setText(value)

    def _request_time_inspection(self) -> None:
        path = self.path.text().strip()
        if not path:
            QMessageBox.warning(self, "缺少输入", "请选择源数据目录。")
            return
        if self.input_kind.currentData() == "temporary":
            self.status.setText("正在检查临时任务清单和已验证的中间 Zarr。")
            self.task_requested.emit(
                "检查临时处理产物",
                lambda cancel_event: _run_cancelable(
                    cancel_event, lambda: inspect_temporary_pipeline(Path(path))
                ),
                self._temporary_inspection_done,
            )
            return
        if self.input_kind.currentData() == "zarr":
            self.status.setText("正在检查现有 Zarr 元数据。")
            self.task_requested.emit(
                "检查现有 Zarr",
                lambda cancel_event: _run_cancelable(
                    cancel_event, lambda: inspect_zarr(Path(path))
                ),
                self._zarr_inspection_done,
            )
            return
        workers = self.inspect_workers.value() or None
        config = SourceInspectionConfig(
            input_dir=Path(path),
            mode="auto",
            recursive=self.recursive.isChecked(),
            engine=self.engine.currentData(),
            workers=workers,
        )
        self.status.setText("正在检查文件名和 time 维度信息，请在任务与日志页面查看进度。")
        self.task_requested.emit(
            "检查文件时间维度信息",
            lambda cancel_event: inspect_time_metadata(
                config.input_dir,
                recursive=config.recursive,
                requested_engine=config.engine,
                cancel_event=cancel_event,
            ),
            self._time_inspection_done,
        )

    def _zarr_inspection_done(self, result: InspectionResult) -> None:
        self.result = result
        self.report.setPlainText(result.report)
        self.status.setText("Zarr 检查完成，请进入处理流程选择操作。")
        self.save_button.setEnabled(False)
        self.zarr_result_ready.emit(result)

    def _temporary_inspection_done(self, result: InspectionResult) -> None:
        self.result = result
        self.report.setPlainText(result.report)
        self.status.setText("临时处理产物可用，请进入处理流程继续执行。")
        self.save_button.setEnabled(False)
        self.zarr_result_ready.emit(result)

    def _time_inspection_done(self, result: TimeInspectionResult) -> None:
        self.time_result = result
        self.report.setPlainText(result.report)
        self.time_panel.set_result(result)
        self.mapping_group.setEnabled(result.time_dimension.exists)
        self.confirm_time_button.setEnabled(True)
        self.status.setText("时间信息检查完成，请确认完整时间来源或组合字段。")

    def _request_structure_inspection(self) -> None:
        if self.time_result is None:
            QMessageBox.warning(self, "尚未检查时间", "请先点击“检查文件时间维度信息”。")
            return
        try:
            rule = self.time_panel.rule()
            rule.validate()
            dimensions = (
                self.source_time_dim.text().strip(),
                self.source_lat_dim.text().strip(),
                self.source_lon_dim.text().strip(),
            )
            if any(dimensions) and not all(dimensions):
                raise ValueError("源 time、纬度和经度维度必须同时填写。")
            config = SourceInspectionConfig(
                input_dir=self.time_result.input_dir,
                mode="auto",
                recursive=self.recursive.isChecked(),
                engine=self.time_result.engine,
                source_dimensions=tuple(dimensions) if all(dimensions) else None,
                workers=self.inspect_workers.value() or None,
                time_rule=rule,
                time_inspection=self.time_result,
                cache_path=default_inspection_cache_path(self.time_result.input_dir),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "时间规则不完整", str(exc))
            return
        self.status.setText("正在检查变量、空间网格和属性，请在任务与日志页面查看进度。")
        self.task_requested.emit(
            "检查数据结构",
            lambda cancel_event: inspect_source(config, cancel_event=cancel_event),
            self._inspection_done,
        )

    def _inspection_done(self, result: InspectionResult) -> None:
        self.result = result
        self.report.setPlainText(result.report)
        if result.warnings:
            self.status.setText("检查完成，但存在警告：" + "；".join(result.warnings))
        else:
            self.status.setText("检查通过，可以进入转换模块。")
        self.save_button.setEnabled(True)
        self.result_ready.emit(result)

    def _invalidate_time_check(self, *_args) -> None:
        if self.time_result is None and self.result is None:
            return
        had_result = self.result is not None
        self.time_result = None
        self.result = None
        self.time_panel.setEnabled(False)
        self.confirm_time_button.setEnabled(False)
        self.mapping_group.setEnabled(False)
        self.save_button.setEnabled(False)
        self.report.clear()
        self.status.setText("检查参数已改变，请重新检查文件时间维度信息。")
        if had_result:
            self.result_invalidated.emit()

    def _invalidate_structure_check(self, *_args) -> None:
        if self.result is None:
            return
        self.result = None
        self.save_button.setEnabled(False)
        self.status.setText("结构检查参数已改变，请重新检查数据结构。")
        self.result_invalidated.emit()

    def _save_snapshot(self) -> None:
        if self.result is None:
            return
        value, _ = QFileDialog.getSaveFileName(
            self,
            "保存检查快照",
            str(self.result.path / "inspection.json"),
            "JSON 文件 (*.json)",
        )
        if not value:
            return
        try:
            save_inspection_snapshot(self.result, Path(value))
            self.status.setText(f"检查快照已保存：{value}")
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "保存失败", str(exc))

    def _load_snapshot(self) -> None:
        value, _ = QFileDialog.getOpenFileName(
            self,
            "导入检查快照",
            str(Path.cwd()),
            "检查快照 (*.json);;JSON 文件 (*.json)",
        )
        if not value:
            return
        try:
            result = load_inspection_snapshot(Path(value))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "导入失败", str(exc))
            return
        self.path.blockSignals(True)
        self.path.setText(str(result.path))
        self.path.blockSignals(False)
        self.time_result = None
        self.time_panel.setEnabled(False)
        self.confirm_time_button.setEnabled(False)
        self.mapping_group.setEnabled(False)
        self.result = result
        self.report.setPlainText(result.report)
        self.save_button.setEnabled(True)
        self.status.setText("检查快照已导入，可以直接进入转换模块。")
        self.result_ready.emit(result)


class ConversionPage(QWidget):
    task_requested = Signal(str, object, object)
    result_ready = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.inspection: InspectionResult | None = None
        self.preview: ConversionPreview | None = None
        root = QVBoxLayout(self)
        title = QLabel("转换模块")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(QLabel("使用已确认的数据检查结果，将 NetCDF/HDF/TIFF 转换为 Zarr v3。"))

        self.input_status = QLabel("请先在数据检查模块完成检查。")
        root.addWidget(self.input_status)
        settings = QGroupBox("转换参数")
        form = QFormLayout(settings)
        self.start_date = QDateEdit()
        self.start_date.setCalendarPopup(True)
        self.end_date = QDateEdit()
        self.end_date.setCalendarPopup(True)
        date_row = QWidget()
        date_layout = QHBoxLayout(date_row)
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("至"))
        date_layout.addWidget(self.end_date)
        form.addRow("时间范围", date_row)

        self.lat_min = self._number_box(-90)
        self.lat_max = self._number_box(90)
        form.addRow("纬度范围", self._range_row(self.lat_min, self.lat_max))
        self.lon_min = self._number_box(-180)
        self.lon_max = self._number_box(180)
        form.addRow("经度范围", self._range_row(self.lon_min, self.lon_max))

        self.output = QLineEdit()
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output, 1)
        output_layout.addWidget(_button("浏览…", self._choose_output))
        form.addRow("输出 Zarr", output_row)
        root.addWidget(settings)

        variables_group = QGroupBox("变量选择与输出设置")
        variables_layout = QVBoxLayout(variables_group)
        self.variables = QTableWidget(0, 6)
        self.variables.setHorizontalHeaderLabels(
            ("选择", "源变量", "输出变量名", "填充值", "缩放因子", "输出填充值")
        )
        self.variables.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        for column in (2, 3, 4, 5):
            self.variables.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch
            )
        self.variables.setMaximumHeight(240)
        self.variables.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        variables_layout.addWidget(self.variables)
        variables_layout.addWidget(
            QLabel(
                "填充值支持逗号分隔的多个数值或 nan；缩放因子为空表示不额外缩放。"
                "输出填充值用于替换匹配到的缺失值；输出变量名为空时使用源变量名。"
            )
        )
        root.addWidget(variables_group)

        advanced = QGroupBox("执行选项")
        advanced_form = QFormLayout(advanced)
        self.auto_tune = QCheckBox("转换前自动实测候选计划")
        advanced_form.addRow("自动调优", self.auto_tune)
        self.tune_budget = QDoubleSpinBox()
        self.tune_budget.setRange(1, 3600)
        self.tune_budget.setValue(60)
        self.tune_budget.setSuffix(" 秒")
        advanced_form.addRow("调优预算", self.tune_budget)
        self.workers = QSpinBox()
        self.workers.setRange(0, 128)
        self.workers.setValue(0)
        self.workers.setSpecialValueText("自动")
        advanced_form.addRow("最大 worker", self.workers)
        self.validate = QCheckBox("写入后执行抽样校验")
        self.validate.setChecked(True)
        advanced_form.addRow("输出校验", self.validate)
        self.overwrite = QCheckBox("允许覆盖已有 Zarr 输出")
        advanced_form.addRow("覆盖策略", self.overwrite)
        root.addWidget(advanced)

        actions = QHBoxLayout()
        actions.addWidget(_button("预览转换计划", self._request_preview))
        actions.addWidget(_button("开始转换", self._request_run))
        actions.addStretch(1)
        root.addLayout(actions)
        self.status = QLabel("等待检查结果。")
        root.addWidget(self.status)
        self.plan_report = QTextBrowser()
        root.addWidget(self.plan_report, 1)

    @staticmethod
    def _number_box(value: float) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(-1_000_000_000, 1_000_000_000)
        box.setDecimals(6)
        box.setValue(value)
        return box

    @staticmethod
    def _range_row(lower, upper) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(lower)
        layout.addWidget(QLabel("至"))
        layout.addWidget(upper)
        return row

    def set_inspection(self, result: InspectionResult) -> None:
        self.inspection = result
        info = result.source_inventory
        self.input_status.setText(
            f"输入已检查：{result.path}；{len(info.files)} 个文件；"
            f"时间 { _date_text(info.times[0]) } .. { _date_text(info.times[-1]) }"
        )
        self.start_date.setDate(_qdate(info.times[0]))
        self.end_date.setDate(_qdate(info.times[-1]))
        self.start_date.setDateRange(_qdate(info.times[0]), _qdate(info.times[-1]))
        self.end_date.setDateRange(_qdate(info.times[0]), _qdate(info.times[-1]))
        self.lat_min.setRange(float(np.nanmin(info.lat_values)), float(np.nanmax(info.lat_values)))
        self.lat_max.setRange(float(np.nanmin(info.lat_values)), float(np.nanmax(info.lat_values)))
        self.lon_min.setRange(float(np.nanmin(info.lon_values)), float(np.nanmax(info.lon_values)))
        self.lon_max.setRange(float(np.nanmin(info.lon_values)), float(np.nanmax(info.lon_values)))
        self.lat_min.setValue(float(np.nanmin(info.lat_values)))
        self.lat_max.setValue(float(np.nanmax(info.lat_values)))
        self.lon_min.setValue(float(np.nanmin(info.lon_values)))
        self.lon_max.setValue(float(np.nanmax(info.lon_values)))
        self.variables.setRowCount(len(info.variables))
        for row, (name, spec) in enumerate(info.variables.items()):
            item = QTableWidgetItem()
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self.variables.setItem(row, 0, item)
            source_item = QTableWidgetItem(name)
            source_item.setToolTip(
                "；".join(f"{key}={value!r}" for key, value in spec.attrs.items())
            )
            self.variables.setItem(row, 1, source_item)
            self.variables.setCellWidget(row, 2, QLineEdit(name))
            fill_edit = QLineEdit()
            source_fill = ", ".join(
                str(spec.attrs[key])
                for key in ("_FillValue", "missing_value")
                if key in spec.attrs
            )
            fill_edit.setPlaceholderText(f"源: {source_fill}" if source_fill else "不处理")
            self.variables.setCellWidget(row, 3, fill_edit)
            scale_edit = QLineEdit()
            scale_edit.setPlaceholderText(
                f"源: {spec.attrs['scale_factor']}"
                if "scale_factor" in spec.attrs
                else "不处理"
            )
            self.variables.setCellWidget(row, 4, scale_edit)
            output_fill = QLineEdit()
            output_fill.setPlaceholderText("默认浮点为 NaN")
            self.variables.setCellWidget(row, 5, output_fill)
        if not self.output.text():
            self.output.setText(str(result.path.parent / f"{result.path.name}.zarr"))
        self.status.setText("检查结果已载入，请选择范围并预览计划。")

    def _choose_output(self) -> None:
        value = QFileDialog.getSaveFileName(
            self, "选择输出 Zarr 目录", self.output.text() or str(Path.cwd())
        )[0]
        if value:
            self.output.setText(value)

    def _selected_variables(self) -> tuple[str, ...]:
        return tuple(
            str(self.variables.item(row, 0).data(Qt.ItemDataRole.UserRole))
            for row in range(self.variables.rowCount())
            if self.variables.item(row, 0).checkState() == Qt.CheckState.Checked
        )

    @staticmethod
    def _parse_number(text: str, label: str) -> float | None:
        value = text.strip()
        if not value:
            return None
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label}必须是数值：{value}") from exc

    @classmethod
    def _parse_fill_values(cls, text: str) -> tuple[float, ...] | None:
        value = text.strip()
        if not value:
            return None
        parts = [
            item.strip()
            for item in value.replace("，", ",").replace(";", ",").split(",")
            if item.strip()
        ]
        if not parts:
            return None
        return tuple(
            parsed
            for parsed in (cls._parse_number(item, "填充值") for item in parts)
            if parsed is not None
        )

    def _variable_settings(self) -> tuple[dict[str, str], dict[str, VariableTransform]]:
        names: dict[str, str] = {}
        transforms: dict[str, VariableTransform] = {}
        for row in range(self.variables.rowCount()):
            item = self.variables.item(row, 0)
            source = str(item.data(Qt.ItemDataRole.UserRole))
            if item.checkState() != Qt.CheckState.Checked:
                continue
            output = self.variables.cellWidget(row, 2).text().strip() or source
            fill_values = self._parse_fill_values(self.variables.cellWidget(row, 3).text())
            scale_factor = self._parse_number(
                self.variables.cellWidget(row, 4).text(), "缩放因子"
            )
            output_fill = self._parse_number(
                self.variables.cellWidget(row, 5).text(), "输出填充值"
            )
            names[source] = output
            if fill_values is not None or scale_factor is not None or output_fill is not None:
                transforms[source] = VariableTransform(
                    fill_values=fill_values,
                    scale_factor=scale_factor,
                    output_fill=output_fill,
                )
        return names, transforms

    def _config(self) -> ConversionConfig:
        if self.inspection is None:
            raise ValueError("请先完成数据检查。")
        if not self.output.text().strip():
            raise ValueError("请选择输出 Zarr 目录。")
        variable_names, variable_transforms = self._variable_settings()
        return ConversionConfig(
            output=Path(self.output.text().strip()),
            time_start=self.start_date.date().toString("yyyy-MM-dd"),
            time_end=self.end_date.date().toString("yyyy-MM-dd"),
            lat_min=self.lat_min.value(),
            lat_max=self.lat_max.value(),
            lon_min=self.lon_min.value(),
            lon_max=self.lon_max.value(),
            variables=self._selected_variables(),
            variable_names=variable_names,
            variable_transforms=variable_transforms,
            auto_tune=self.auto_tune.isChecked(),
            tune_budget=self.tune_budget.value(),
            max_workers=self.workers.value() or None,
            overwrite=self.overwrite.isChecked(),
            validate=self.validate.isChecked(),
        )

    def _request_preview(self) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        self.task_requested.emit(
            "生成转换计划",
            lambda cancel_event: _run_cancelable(
                cancel_event, lambda: preview_conversion(self.inspection, config)
            ),
            self._preview_done,
        )

    def _request_run(self) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        if self.preview is None:
            answer = QMessageBox.question(
                self,
                "尚未预览计划",
                "尚未生成转换计划，是否直接生成计划并开始转换？",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.task_requested.emit(
            "执行 Zarr 转换",
            lambda cancel_event: run_conversion(
                self.inspection, config, cancel_event=cancel_event
            ),
            self._run_done,
        )

    def _preview_done(self, preview: ConversionPreview) -> None:
        self.preview = preview
        self.plan_report.setPlainText(
            "========== 转换计划预览 ==========\n"
            f"shape(time, lat, lon)：{preview.selection.shape}\n"
            f"变量：{', '.join(preview.selection.variables)}\n"
            f"逻辑未压缩量：{_human_bytes(preview.logical_bytes)}\n"
            f"计划：{preview.plan.label()}\n"
            + "\n".join(f"  - {reason}" for reason in preview.plan.rationale)
        )
        self.status.setText("计划已生成，可以开始转换。")

    def _run_done(self, value: tuple[Any, dict[str, Any]]) -> None:
        plan, metrics = value
        self.status.setText(
            f"转换完成：{self.output.text()}；耗时 {float(metrics.get('elapsed', 0)):.1f} 秒。"
        )
        self.plan_report.append(
            "\n========== 转换结果 ==========\n"
            f"最终计划：{plan.label()}\n"
            f"逻辑吞吐：{float(metrics.get('throughput_mib_s', 0)):.1f} MiB/s"
        )


class RechunkPage(QWidget):
    task_requested = Signal(str, object, object)
    result_ready = Signal(object)

    def __init__(self, *, compression_only: bool = False, parent=None) -> None:
        super().__init__(parent)
        self.info = None
        self.preview: RechunkPreview | None = None
        root = QVBoxLayout(self)
        title = QLabel("Zarr 优化模块")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(
            QLabel(
                "对输入 Zarr v3 选择重分块、重压缩，或在一次任务中同时执行两项操作；"
                "最终只生成一份输出 Zarr。"
            )
        )

        input_group = QGroupBox("Zarr 输入")
        form = QFormLayout(input_group)
        self.input = QLineEdit()
        input_row = QWidget()
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.addWidget(self.input, 1)
        input_layout.addWidget(_button("浏览…", self._choose_input))
        input_layout.addWidget(_button("检查", self._request_inspection))
        form.addRow("输入 Zarr v3", input_row)
        self.input_status = QLabel("尚未检查输入。")
        form.addRow("检查状态", self.input_status)
        root.addWidget(input_group)

        operations = QGroupBox("操作选择")
        operations_layout = QHBoxLayout(operations)
        self.rechunk_checkbox = QCheckBox("执行重分块")
        self.rechunk_checkbox.setChecked(not compression_only)
        self.recompress_checkbox = QCheckBox("执行重压缩")
        self.recompress_checkbox.setChecked(compression_only)
        operations_layout.addWidget(self.rechunk_checkbox)
        operations_layout.addWidget(self.recompress_checkbox)
        operations_layout.addStretch(1)
        root.addWidget(operations)

        options = QGroupBox("操作参数")
        options_form = QFormLayout(options)
        self.strategy = QComboBox()
        self.strategy.addItem("时间连续", "time")
        self.strategy.addItem("空间连续", "space")
        self.strategy.addItem("自定义", "custom")
        options_form.addRow("重分块策略", self.strategy)
        self.target = QDoubleSpinBox()
        self.target.setRange(32, 256)
        self.target.setValue(DEFAULT_TARGET_MIB)
        self.target.setSuffix(" MiB")
        options_form.addRow("目标 chunk", self.target)
        chunk_row = QWidget()
        chunk_layout = QHBoxLayout(chunk_row)
        chunk_layout.setContentsMargins(0, 0, 0, 0)
        self.chunk_boxes = []
        for index, label in enumerate(("time", "lat", "lon")):
            box = QSpinBox()
            box.setRange(1, 1_000_000)
            box.setSpecialValueText(label)
            self.chunk_boxes.append(box)
            chunk_layout.addWidget(box)
        options_form.addRow("自定义 chunk", chunk_row)
        self.compression = QComboBox()
        for label, value in (
            ("保留输入 codec", "none"),
            ("快速（Zstd 1）", "fast"),
            ("平衡（Zstd 4）", "balanced"),
            ("极致（Zstd 9）", "maximum"),
        ):
            self.compression.addItem(label, value)
        self.compression.setCurrentIndex(2)
        options_form.addRow("压缩方案", self.compression)
        temporary_row = QWidget()
        temporary_layout = QHBoxLayout(temporary_row)
        temporary_layout.setContentsMargins(0, 0, 0, 0)
        self.temporary_dir = QLineEdit()
        self.temporary_dir.setPlaceholderText("可选；同时勾选两项时建议选择 SSD 目录")
        self.temporary_dir.setToolTip(
            "仅用于阶段间反复读取的中间 Zarr；最终输出仍直接写入输出目录所在磁盘，"
            "任务完成后自动删除。"
        )
        temporary_layout.addWidget(self.temporary_dir, 1)
        temporary_layout.addWidget(_button("浏览…", self._choose_temporary_dir))
        options_form.addRow("中间处理目录", temporary_row)
        self.workers = QSpinBox()
        self.workers.setRange(1, 128)
        self.workers.setValue(default_workers())
        options_form.addRow("worker 上限", self.workers)
        self.validate = QCheckBox("输出后执行抽样校验")
        self.validate.setChecked(True)
        options_form.addRow("输出校验", self.validate)
        self.overwrite = QCheckBox("允许覆盖已有 Zarr 输出")
        options_form.addRow("覆盖策略", self.overwrite)
        root.addWidget(options)

        output_group = QGroupBox("输出")
        output_form = QFormLayout(output_group)
        self.output = QLineEdit()
        self._output_manually_set = False
        self._updating_output = False
        self.output.textChanged.connect(self._output_changed)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output, 1)
        output_layout.addWidget(_button("浏览…", self._choose_output))
        output_form.addRow("输出 Zarr", output_row)
        root.addWidget(output_group)

        actions = QHBoxLayout()
        actions.addWidget(_button("预览计划", self._request_preview))
        actions.addWidget(_button("开始执行", self._request_run))
        actions.addStretch(1)
        root.addLayout(actions)
        self.status = QLabel("请先检查输入 Zarr。")
        root.addWidget(self.status)
        self.report = QTextBrowser()
        root.addWidget(self.report, 1)

        self.rechunk_checkbox.toggled.connect(self._update_controls)
        self.recompress_checkbox.toggled.connect(self._update_controls)
        self.strategy.currentIndexChanged.connect(self._update_controls)
        self._update_controls()

    def _update_controls(self) -> None:
        rechunk = self.rechunk_checkbox.isChecked()
        recompress = self.recompress_checkbox.isChecked()
        self.strategy.setEnabled(rechunk)
        self.target.setEnabled(rechunk)
        custom = rechunk and self.strategy.currentData() == "custom"
        for box in self.chunk_boxes:
            box.setEnabled(custom)
        self.compression.setEnabled(recompress)
        self.temporary_dir.setEnabled(rechunk and recompress)
        self._set_default_output()

    def _output_changed(self, _text: str) -> None:
        if not self._updating_output:
            self._output_manually_set = True

    def _set_default_output(self) -> None:
        if self.info is None or self._output_manually_set:
            return
        suffix = {
            "重分块": "_rechunked.zarr",
            "重压缩": "_recompressed.zarr",
            "重分块 + 重压缩": "_rechunked_recompressed.zarr",
        }.get(self._operation_label(), "_optimized.zarr")
        value = str(self.info.path.with_name(self.info.path.name + suffix))
        self._updating_output = True
        try:
            self.output.setText(value)
        finally:
            self._updating_output = False

    def _operation_label(self) -> str:
        selected = []
        if self.rechunk_checkbox.isChecked():
            selected.append("重分块")
        if self.recompress_checkbox.isChecked():
            selected.append("重压缩")
        return " + ".join(selected) if selected else "未选择操作"

    def _choose_input(self) -> None:
        value = QFileDialog.getExistingDirectory(self, "选择输入 Zarr", self.input.text() or str(Path.cwd()))
        if value:
            self.input.setText(value)

    def _choose_output(self) -> None:
        value = QFileDialog.getSaveFileName(self, "选择输出 Zarr", self.output.text() or str(Path.cwd()))[0]
        if value:
            self.output.setText(value)

    def _choose_temporary_dir(self) -> None:
        value = QFileDialog.getExistingDirectory(
            self,
            "选择临时处理目录（建议选择 SSD）",
            self.temporary_dir.text() or str(Path.cwd()),
        )
        if value:
            self.temporary_dir.setText(value)

    def _request_inspection(self) -> None:
        if not self.input.text().strip():
            QMessageBox.warning(self, "缺少输入", "请选择输入 Zarr v3 目录。")
            return
        self.task_requested.emit(
            "检查 Zarr 输入",
            lambda cancel_event: _run_cancelable(
                cancel_event, lambda: inspect_zarr(Path(self.input.text().strip()))
            ),
            self._inspection_done,
        )

    def _inspection_done(self, result: InspectionResult) -> None:
        self.info = result.zarr_info
        self.input_status.setText(
            f"检查通过：Zarr v{self.info.zarr_format}，"
            f"shape={self.info.shape}，逻辑大小={_human_bytes(self.info.logical_bytes)}"
        )
        self._set_default_output()
        self.report.setPlainText(result.report)
        self.status.setText("输入检查通过，请设置操作参数并预览计划。")
        self.result_ready.emit(result)

    def _config(self) -> RechunkConfig:
        if self.info is None:
            raise ValueError("请先检查输入 Zarr。")
        if not self.output.text().strip():
            raise ValueError("请选择输出 Zarr 目录。")
        rechunk = self.rechunk_checkbox.isChecked()
        recompress = self.recompress_checkbox.isChecked()
        if not rechunk and not recompress:
            raise ValueError("请至少勾选重分块或重压缩中的一项。")
        custom = None
        if self.strategy.currentData() == "custom" and rechunk:
            custom = tuple(box.value() for box in self.chunk_boxes)
        temporary_dir = (
            Path(self.temporary_dir.text().strip())
            if rechunk and recompress and self.temporary_dir.text().strip()
            else None
        )
        return RechunkConfig(
            input=Path(self.input.text().strip()),
            output=Path(self.output.text().strip()),
            strategy=self.strategy.currentData(),
            target_mib=self.target.value(),
            custom_chunks=custom,
            compression=self.compression.currentData() if recompress else "none",
            workers=self.workers.value(),
            overwrite=self.overwrite.isChecked(),
            validate=self.validate.isChecked(),
            rechunk=rechunk,
            recompress=recompress,
            temporary_dir=temporary_dir,
        )

    def _request_preview(self) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        self.task_requested.emit(
            f"生成{self._operation_label()}计划",
            lambda cancel_event: _run_cancelable(
                cancel_event, lambda: preview_rechunk(config, self.info)
            ),
            self._preview_done,
        )

    def _request_run(self) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        self.task_requested.emit(
            f"执行{self._operation_label()}",
            lambda cancel_event: run_rechunk(
                config, self.info, cancel_event=cancel_event
            ),
            self._run_done,
        )

    def _preview_done(self, preview: RechunkPreview) -> None:
        self.preview = preview
        self.report.setPlainText(
            f"========== {self._operation_label()}计划 ==========\n"
            f"策略：{preview.plan.strategy}\n"
            f"目标 chunks(time, lat, lon)：{preview.plan.chunks}\n"
            f"预计单 chunk：{_human_bytes(preview.plan.estimated_chunk_bytes)}\n"
            f"压缩：{preview.compression.description}\n"
            + "\n".join(f"  - {reason}" for reason in preview.plan.rationale)
        )
        self.status.setText("计划已生成，可以执行。")

    def _run_done(self, metrics: dict[str, Any]) -> None:
        self.status.setText(f"执行完成：{self.output.text()}")
        self.report.append(
            "\n========== 执行结果 ==========\n"
            f"耗时：{float(metrics.get('elapsed', 0)):.1f} 秒\n"
            f"逻辑吞吐：{float(metrics.get('throughput_mib_s', 0)):.1f} MiB/s\n"
            f"输出物理大小：{_human_bytes(int(metrics.get('physical_bytes', 0)))}\n"
            f"临时处理目录：{metrics.get('temporary_dir', '未返回')}"
        )


class ResamplePage(QWidget):
    """GUI page for xESMF spatial resampling of an existing Zarr store."""

    task_requested = Signal(str, object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.inspection = None
        self.preview: ResamplePreview | None = None
        root = QVBoxLayout(self)
        title = QLabel("Zarr 重采样模块")
        title.setObjectName("pageTitle")
        root.addWidget(title)
        root.addWidget(
            QLabel(
                "使用 xESMF 对 Zarr v3 的一维规则经纬度网格进行重采样；"
                "输入检查、chunks 和压缩格式会在执行前锁定。目标范围不足一个整除网格时，"
                "程序会向外覆盖到完整目标单元。"
            )
        )

        input_group = QGroupBox("Zarr 输入")
        form = QFormLayout(input_group)
        self.input = QLineEdit()
        input_row = QWidget()
        input_layout = QHBoxLayout(input_row)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.addWidget(self.input, 1)
        input_layout.addWidget(_button("浏览…", self._choose_input))
        input_layout.addWidget(_button("检查", self._request_inspection))
        form.addRow("输入 Zarr v3", input_row)
        self.input_status = QLabel("尚未检查输入。")
        form.addRow("检查状态", self.input_status)
        root.addWidget(input_group)

        options = QGroupBox("重采样参数")
        options_form = QFormLayout(options)
        self.resolution = QDoubleSpinBox()
        self.resolution.setDecimals(6)
        self.resolution.setRange(0.000001, 180.0)
        self.resolution.setValue(0.25)
        self.resolution.setSuffix("°")
        options_form.addRow("目标空间分辨率", self.resolution)
        self.method = QComboBox()
        for label, value in (
            ("双线性 bilinear", "bilinear"),
            ("守恒 conservative", "conservative"),
            ("归一化守恒 conservative_normed", "conservative_normed"),
            ("Patch patch", "patch"),
            ("源到目标最近邻 nearest_s2d", "nearest_s2d"),
            ("目标到源最近邻 nearest_d2s", "nearest_d2s"),
        ):
            self.method.addItem(label, value)
        options_form.addRow("xESMF 方法", self.method)
        self.compute_dtype = QComboBox()
        self.compute_dtype.addItem("保持源浮点 dtype", "source")
        self.compute_dtype.addItem("浮点变量转 float32（推荐）", "float32")
        self.compute_dtype.setToolTip(
            "仅转换浮点数据变量；整数变量仍使用安全的浮点输出路径。"
        )
        options_form.addRow("计算 dtype", self.compute_dtype)
        self.skipna = QCheckBox("忽略 NaN 和输入填充值")
        self.skipna.setChecked(True)
        options_form.addRow("缺失值处理", self.skipna)
        self.extent = QComboBox()
        self.extent.addItem("覆盖输入空间范围", "source")
        self.extent.addItem("全球范围", "global")
        options_form.addRow("目标空间范围", self.extent)
        self.tile_mode = QComboBox()
        self.tile_mode.addItem("自动（推荐）", "auto")
        self.tile_mode.addItem("手动指定", "manual")
        self.tile_mode.currentIndexChanged.connect(self._tile_mode_changed)
        options_form.addRow("流式空间块模式", self.tile_mode)
        self.tile_size = QSpinBox()
        self.tile_size.setRange(16, 2048)
        self.tile_size.setValue(128)
        self.tile_size.setEnabled(False)
        options_form.addRow("手动空间块边长", self.tile_size)
        self.time_block_mode = QComboBox()
        self.time_block_mode.addItem("自动（推荐）", "auto")
        self.time_block_mode.addItem("手动指定", "manual")
        self.time_block_mode.currentIndexChanged.connect(self._time_block_mode_changed)
        options_form.addRow("时间块模式", self.time_block_mode)
        self.time_block = QSpinBox()
        self.time_block.setRange(1, 4096)
        self.time_block.setValue(128)
        self.time_block.setEnabled(False)
        options_form.addRow("手动时间块大小", self.time_block)
        self.compute_workers = QSpinBox()
        self.compute_workers.setRange(1, 8)
        self.compute_workers.setValue(2)
        options_form.addRow("块内计算线程", self.compute_workers)
        self.space_worker_mode = QComboBox()
        self.space_worker_mode.addItem("自动（推荐）", "auto")
        self.space_worker_mode.addItem("手动指定", "manual")
        self.space_worker_mode.currentIndexChanged.connect(self._space_worker_mode_changed)
        options_form.addRow("空间块并行模式", self.space_worker_mode)
        self.space_workers = QSpinBox()
        self.space_workers.setRange(1, 32)
        self.space_workers.setValue(6)
        self.space_workers.setEnabled(False)
        options_form.addRow("空间并行进程数", self.space_workers)
        temporary_row = QWidget()
        temporary_layout = QHBoxLayout(temporary_row)
        temporary_layout.setContentsMargins(0, 0, 0, 0)
        self.temporary_dir = QLineEdit()
        self.temporary_dir.setPlaceholderText("可选；建议选择 SSD 目录")
        self.temporary_dir.setToolTip(
            "大时间 chunk 的中转 Zarr 和 xESMF 权重写入此目录；成功后自动删除，"
            "失败时保留中间目录用于排查。"
        )
        temporary_layout.addWidget(self.temporary_dir, 1)
        temporary_layout.addWidget(_button("浏览…", self._choose_temporary_dir))
        options_form.addRow("中间处理目录", temporary_row)
        self.validate = QCheckBox("输出后执行结构和坐标校验")
        self.validate.setChecked(True)
        options_form.addRow("输出校验", self.validate)
        self.overwrite = QCheckBox("允许覆盖已有 Zarr 输出")
        options_form.addRow("覆盖策略", self.overwrite)
        root.addWidget(options)

        output_group = QGroupBox("输出")
        output_form = QFormLayout(output_group)
        self.output = QLineEdit()
        self._output_manually_set = False
        self._updating_output = False
        self.output.textChanged.connect(self._output_changed)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output, 1)
        output_layout.addWidget(_button("浏览…", self._choose_output))
        output_form.addRow("输出 Zarr", output_row)
        root.addWidget(output_group)

        actions = QHBoxLayout()
        actions.addWidget(_button("预览计划", self._request_preview))
        actions.addWidget(_button("开始执行", self._request_run))
        actions.addStretch(1)
        root.addLayout(actions)
        self.status = QLabel("请先检查输入 Zarr。")
        root.addWidget(self.status)
        self.report = QTextBrowser()
        root.addWidget(self.report, 1)

    def _tile_mode_changed(self, _index: int) -> None:
        self.tile_size.setEnabled(self.tile_mode.currentData() == "manual")

    def _space_worker_mode_changed(self, _index: int) -> None:
        self.space_workers.setEnabled(self.space_worker_mode.currentData() == "manual")

    def _time_block_mode_changed(self, _index: int) -> None:
        self.time_block.setEnabled(self.time_block_mode.currentData() == "manual")

    def _output_changed(self, _text: str) -> None:
        if not self._updating_output:
            self._output_manually_set = True

    def _set_default_output(self) -> None:
        if self.inspection is None or self._output_manually_set:
            return
        value = str(self.inspection.grid.path.with_name(
            self.inspection.grid.path.name + "_resampled.zarr"
        ))
        self._updating_output = True
        try:
            self.output.setText(value)
        finally:
            self._updating_output = False

    def _choose_input(self) -> None:
        value = QFileDialog.getExistingDirectory(
            self, "选择输入 Zarr", self.input.text() or str(Path.cwd())
        )
        if value:
            self.input.setText(value)

    def _choose_output(self) -> None:
        value = QFileDialog.getSaveFileName(
            self, "选择输出 Zarr", self.output.text() or str(Path.cwd())
        )[0]
        if value:
            self.output.setText(value)

    def _choose_temporary_dir(self) -> None:
        value = QFileDialog.getExistingDirectory(
            self,
            "选择重采样中间处理目录（建议 SSD）",
            self.temporary_dir.text() or str(Path.cwd()),
        )
        if value:
            self.temporary_dir.setText(value)

    def _request_inspection(self) -> None:
        if not self.input.text().strip():
            QMessageBox.warning(self, "缺少输入", "请选择输入 Zarr v3 目录。")
            return
        self.task_requested.emit(
            "检查重采样输入",
            lambda cancel_event: _run_cancelable(
                cancel_event, lambda: inspect_resample(Path(self.input.text().strip()))
            ),
            self._inspection_done,
        )

    def _inspection_done(self, result) -> None:
        self.inspection = result
        self.input_status.setText(
            f"检查通过：Zarr v{result.info.zarr_format}；"
            f"当前分辨率 {result.grid.lat_resolution:g}° × "
            f"{result.grid.lon_resolution:g}°"
        )
        self.report.setPlainText(result.report)
        self._set_default_output()
        self.status.setText("输入检查通过，请设置重采样参数并预览计划。")

    def _config(self) -> ResampleConfig:
        if self.inspection is None:
            raise ValueError("请先检查输入 Zarr。")
        if not self.output.text().strip():
            raise ValueError("请选择输出 Zarr 目录。")
        return ResampleConfig(
            input=self.inspection.info.path,
            output=Path(self.output.text().strip()),
            resolution=self.resolution.value(),
            method=self.method.currentData(),
            skipna=self.skipna.isChecked(),
            compute_dtype=self.compute_dtype.currentData(),
            extent=self.extent.currentData(),
            tile_size=(
                "auto"
                if self.tile_mode.currentData() == "auto"
                else self.tile_size.value()
            ),
            time_block=(
                "auto"
                if self.time_block_mode.currentData() == "auto"
                else self.time_block.value()
            ),
            compute_workers=self.compute_workers.value(),
            space_workers=(
                "auto"
                if self.space_worker_mode.currentData() == "auto"
                else self.space_workers.value()
            ),
            temporary_dir=(
                Path(self.temporary_dir.text().strip())
                if self.temporary_dir.text().strip()
                else None
            ),
            overwrite=self.overwrite.isChecked(),
            validate=self.validate.isChecked(),
        )

    def _request_preview(self) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        self.task_requested.emit(
            "生成 Zarr 重采样计划",
            lambda cancel_event: _run_cancelable(
                cancel_event, lambda: preview_resample(config, self.inspection)
            ),
            self._preview_done,
        )

    def _request_run(self) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        self.task_requested.emit(
            "执行 Zarr 重采样",
            lambda cancel_event: run_resample(
                config, self.inspection, cancel_event=cancel_event
            ),
            self._run_done,
        )

    def _preview_done(self, preview: ResamplePreview) -> None:
        self.preview = preview
        self.report.setPlainText(format_resample_preview(preview))
        self.status.setText("重采样计划已生成，可以执行。")

    def _run_done(self, metrics: dict[str, Any]) -> None:
        self.status.setText(f"执行完成：{metrics.get('output', self.output.text())}")
        self.report.append(
            "\n========== 执行结果 ==========\n"
            f"耗时：{float(metrics.get('elapsed', 0)):.1f} 秒\n"
            f"输出物理大小：{_human_bytes(int(metrics.get('physical_bytes', 0)))}\n"
            f"临时处理目录：{metrics.get('temporary_dir', '未返回')}"
        )


class PipelinePage(QWidget):
    """Composable source-to-final-Zarr workflow page."""

    task_requested = Signal(str, object, object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.inspection: InspectionResult | None = None
        self.plan = None
        self.recovery = None
        root = QVBoxLayout(self)
        title = QLabel("统一处理流程")
        title.setObjectName("pageTitle")
        root.addWidget(title)

        operations = QGroupBox("处理流程")
        operations_layout = QHBoxLayout(operations)
        self.conversion_checkbox = QCheckBox("转换")
        self.conversion_checkbox.setChecked(True)
        self.conversion_checkbox.setEnabled(False)
        self.resample_checkbox = QCheckBox("重采样")
        self.rechunk_checkbox = QCheckBox("重分块")
        self.recompress_checkbox = QCheckBox("重压缩")
        for widget in (
            self.conversion_checkbox,
            self.resample_checkbox,
            self.rechunk_checkbox,
            self.recompress_checkbox,
        ):
            operations_layout.addWidget(widget)
        operations_layout.addStretch(1)
        root.addWidget(operations)

        settings = QWidget()
        settings_layout = QVBoxLayout(settings)
        settings_layout.setContentsMargins(0, 0, 0, 0)

        self.general_group = QGroupBox("通用参数")
        general_form = QFormLayout(self.general_group)
        time_row = QHBoxLayout()
        self.time_start = QDateEdit()
        self.time_start.setCalendarPopup(True)
        self.time_end = QDateEdit()
        self.time_end.setCalendarPopup(True)
        time_row.addWidget(self.time_start)
        time_row.addWidget(QLabel("至"))
        time_row.addWidget(self.time_end)
        general_form.addRow("时间范围", time_row)
        self.lat_min = self._double(-90, 90, -90)
        self.lat_max = self._double(-90, 90, 90)
        general_form.addRow("纬度范围", self._pair(self.lat_min, self.lat_max))
        self.lon_min = self._double(-180, 180, -180)
        self.lon_max = self._double(-180, 180, 180)
        general_form.addRow("经度范围", self._pair(self.lon_min, self.lon_max))
        temp_row, self.temporary_dir = self._path_row("选择临时处理目录", directory=True)
        general_form.addRow("临时处理目录", temp_row)
        output_row, self.output = self._path_row("选择最终输出目录", directory=False)
        general_form.addRow("最终输出目录", output_row)
        self.cleanup_intermediate = QCheckBox("下游验证通过后立即删除上游中间 Zarr")
        general_form.addRow("清理策略", self.cleanup_intermediate)
        settings_layout.addWidget(self.general_group)

        self.conversion_group = QGroupBox("Zarr 转换参数")
        conversion_layout = QVBoxLayout(self.conversion_group)
        self.variables = QTableWidget(0, 6)
        self.variables.setHorizontalHeaderLabels(
            ("处理", "源变量", "输出变量名", "填充值", "缩放因子", "输出填充值")
        )
        self.variables.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.variables.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        for index in (3, 4, 5):
            self.variables.horizontalHeader().setSectionResizeMode(
                index, QHeaderView.ResizeMode.Stretch
            )
        self.variables.setMaximumHeight(170)
        conversion_layout.addWidget(self.variables)
        conversion_options = QFormLayout()
        self.auto_tune = QCheckBox("开启自动调参")
        self.auto_tune.setChecked(True)
        conversion_options.addRow("调参", self.auto_tune)
        self.tune_budget = self._double(1, 3600, 60, decimals=1)
        conversion_options.addRow("调参预算（秒）", self.tune_budget)
        self.max_workers = QSpinBox()
        self.max_workers.setRange(0, 256)
        self.max_workers.setSpecialValueText("自动")
        conversion_options.addRow("最大并行核心", self.max_workers)
        conversion_layout.addLayout(conversion_options)
        settings_layout.addWidget(self.conversion_group)

        self.resampling_group = QGroupBox("重采样参数")
        resampling_form = QFormLayout(self.resampling_group)
        self.resolution = self._double(0.000001, 180, 0.1, decimals=6)
        resampling_form.addRow("目标分辨率", self.resolution)
        self.method = QComboBox()
        for value in ("bilinear", "conservative", "conservative_normed", "patch", "nearest_s2d", "nearest_d2s"):
            self.method.addItem(value, value)
        resampling_form.addRow("方法", self.method)
        self.skipna = QCheckBox("忽略缺测并按 xESMF 规则处理")
        self.skipna.setChecked(True)
        resampling_form.addRow("缺测", self.skipna)
        self.na_thres = self._double(0, 1, 1, decimals=3)
        resampling_form.addRow("na_thres", self.na_thres)
        self.compute_dtype = QComboBox()
        self.compute_dtype.addItem("保持源浮点 dtype", "source")
        self.compute_dtype.addItem("浮点转 float32", "float32")
        resampling_form.addRow("计算 dtype", self.compute_dtype)
        self.before_conditions = QLineEdit()
        self.before_conditions.setPlaceholderText("例如 <0, >100")
        resampling_form.addRow("采样前替换值", self.before_conditions)
        self.before_results = QLineEdit()
        self.before_results.setPlaceholderText("例如 0, 100")
        resampling_form.addRow("采样前替换结果", self.before_results)
        self.after_conditions = QLineEdit()
        self.after_conditions.setPlaceholderText("例如 <=median")
        resampling_form.addRow("采样后替换值", self.after_conditions)
        self.after_results = QLineEdit()
        self.after_results.setPlaceholderText("例如 100")
        resampling_form.addRow("采样后替换结果", self.after_results)
        self.statistics_policy = QComboBox()
        self.statistics_policy.addItem("自动（小数据精确，大数据采样）", "auto")
        self.statistics_policy.addItem("确定性采样", "sample")
        self.statistics_policy.addItem("精确全量统计", "exact")
        resampling_form.addRow("表达式统计策略", self.statistics_policy)
        settings_layout.addWidget(self.resampling_group)

        self.chunking_group = QGroupBox("重分块参数")
        chunking_form = QFormLayout(self.chunking_group)
        self.strategy = QComboBox()
        for label, value in (("时间连续", "time"), ("空间连续", "space"), ("自定义", "custom")):
            self.strategy.addItem(label, value)
        chunking_form.addRow("分块模式", self.strategy)
        self.target_mib = self._double(32, 256, 128, decimals=1)
        chunking_form.addRow("目标 chunk（MiB）", self.target_mib)
        custom_row = QHBoxLayout()
        self.custom_chunks = [QSpinBox() for _ in range(3)]
        for box in self.custom_chunks:
            box.setRange(1, 10_000_000)
            custom_row.addWidget(box)
        chunking_form.addRow("自定义 chunks", custom_row)
        self.final_workers = QSpinBox()
        self.final_workers.setRange(1, 256)
        self.final_workers.setValue(1)
        chunking_form.addRow("兼容性最终化 worker", self.final_workers)
        settings_layout.addWidget(self.chunking_group)

        self.compression_group = QGroupBox("重压缩参数")
        compression_form = QFormLayout(self.compression_group)
        self.compression = QComboBox()
        for label, value in (
            ("Blosc / Zstd", "blosc-zstd"),
            ("Blosc / LZ4", "blosc-lz4"),
            ("Blosc / LZ4HC", "blosc-lz4hc"),
            ("Blosc / Zlib", "blosc-zlib"),
            ("原生 Zstd", "zstd"),
            ("原生 Gzip", "gzip"),
        ):
            self.compression.addItem(label, value)
        compression_form.addRow("压缩 codec", self.compression)
        self.compression_level = QSpinBox()
        self.compression_level.setRange(0, 9)
        self.compression_level.setValue(4)
        compression_form.addRow("压缩等级", self.compression_level)
        self.compression_shuffle = QComboBox()
        self.compression_shuffle.addItem("按变量 dtype 自动", "auto")
        self.compression_shuffle.addItem("不使用 shuffle", "noshuffle")
        self.compression_shuffle.addItem("字节 shuffle", "shuffle")
        self.compression_shuffle.addItem("bitshuffle", "bitshuffle")
        compression_form.addRow("Shuffle", self.compression_shuffle)
        self.compression.currentIndexChanged.connect(self._compression_codec_changed)
        settings_layout.addWidget(self.compression_group)

        actions = QHBoxLayout()
        self.preview_button = _button("生成处理计划", self._request_preview)
        self.run_button = _button("开始处理", self._request_run)
        self.run_button.setEnabled(False)
        actions.addWidget(self.preview_button)
        actions.addWidget(self.run_button)
        actions.addStretch(1)
        settings_layout.addLayout(actions)
        self.status = QLabel("等待数据检查结果。")
        settings_layout.addWidget(self.status)
        settings_layout.addStretch(1)

        self.settings_scroll = QScrollArea()
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setWidget(settings)
        self.report = QTextBrowser()
        self.report.setPlaceholderText("尚未生成处理计划。")
        content = QSplitter(Qt.Orientation.Horizontal)
        content.addWidget(self.settings_scroll)
        content.addWidget(self.report)
        content.setStretchFactor(0, 3)
        content.setStretchFactor(1, 2)
        root.addWidget(content, 1)
        self.strategy.currentIndexChanged.connect(self._update_operation_state)
        for checkbox in (
            self.resample_checkbox,
            self.rechunk_checkbox,
            self.recompress_checkbox,
        ):
            checkbox.toggled.connect(self._update_operation_state)
            checkbox.toggled.connect(self._invalidate_plan)
        self._connect_plan_invalidation()
        self._set_enabled(False)

    @staticmethod
    def _double(minimum: float, maximum: float, value: float, *, decimals: int = 3) -> QDoubleSpinBox:
        box = QDoubleSpinBox()
        box.setRange(minimum, maximum)
        box.setDecimals(decimals)
        box.setValue(value)
        return box

    @staticmethod
    def _pair(first: QWidget, second: QWidget) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(first)
        layout.addWidget(QLabel("至"))
        layout.addWidget(second)
        return row

    def _path_row(self, title: str, *, directory: bool):
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        button = QPushButton("浏览…")
        if directory:
            button.clicked.connect(lambda: self._choose_directory(edit, title))
        else:
            button.clicked.connect(lambda: self._choose_directory(edit, title))
        layout.addWidget(edit, 1)
        layout.addWidget(button)
        return row, edit

    @staticmethod
    def _choose_directory(edit, title: str) -> None:
        value = QFileDialog.getExistingDirectory(edit, title, edit.text() or str(Path.cwd()))
        if value:
            edit.setText(value)

    def _connect_plan_invalidation(self) -> None:
        for widget in (
            self.time_start,
            self.time_end,
        ):
            widget.dateChanged.connect(self._invalidate_plan)
        for widget in (
            self.lat_min,
            self.lat_max,
            self.lon_min,
            self.lon_max,
            self.resolution,
            self.tune_budget,
            self.max_workers,
            self.na_thres,
            self.target_mib,
            self.final_workers,
            self.compression_level,
            *self.custom_chunks,
        ):
            widget.valueChanged.connect(self._invalidate_plan)
        for widget in (
            self.method,
            self.compute_dtype,
            self.statistics_policy,
            self.strategy,
            self.compression,
            self.compression_shuffle,
        ):
            widget.currentIndexChanged.connect(self._invalidate_plan)
        for widget in (self.auto_tune, self.skipna, self.cleanup_intermediate):
            widget.toggled.connect(self._invalidate_plan)
        for widget in (
            self.temporary_dir,
            self.output,
            self.before_conditions,
            self.before_results,
            self.after_conditions,
            self.after_results,
        ):
            widget.textChanged.connect(self._invalidate_plan)

    def _compression_codec_changed(self, *_args) -> None:
        codec = self.compression.currentData()
        if codec == "zstd":
            self.compression_level.setRange(-7, 22)
        else:
            self.compression_level.setRange(0, 9)
        native = codec in {"zstd", "gzip"}
        if native and self.compression_shuffle.currentData() not in {"auto", "noshuffle"}:
            self.compression_shuffle.setCurrentIndex(
                self.compression_shuffle.findData("auto")
            )
        self.compression_shuffle.setEnabled(not native)

    def _invalidate_plan(self, *_args) -> None:
        had_plan = self.plan is not None
        self.plan = None
        self.run_button.setEnabled(False)
        if had_plan:
            self.status.setText("参数已变更，请重新生成处理计划。")

    def _set_enabled(self, enabled: bool) -> None:
        self.general_group.setEnabled(enabled)
        self.conversion_group.setEnabled(enabled)
        for widget in (
            self.preview_button,
            self.variables,
            self.auto_tune,
            self.tune_budget,
            self.max_workers,
            self.cleanup_intermediate,
            self.resample_checkbox,
            self.rechunk_checkbox,
            self.recompress_checkbox,
        ):
            widget.setEnabled(enabled)
        self._update_operation_state()

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: object) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _apply_recovery_config(self, config: PipelineConfig) -> None:
        general = config.general
        self.output.setText(str(general.output))
        self.temporary_dir.setText(
            str(general.temporary_dir) if general.temporary_dir is not None else ""
        )
        for widget, value in (
            (self.time_start, general.time_start),
            (self.time_end, general.time_end),
        ):
            date = QDate.fromString(str(value or ""), Qt.DateFormat.ISODate)
            if date.isValid():
                widget.setDate(date)
        self.lat_min.setValue(general.lat_min)
        self.lat_max.setValue(general.lat_max)
        self.lon_min.setValue(general.lon_min)
        self.lon_max.setValue(general.lon_max)
        self.cleanup_intermediate.setChecked(general.cleanup_intermediate)
        operations = config.operations
        self.resample_checkbox.setChecked(operations.resample)
        self.rechunk_checkbox.setChecked(operations.rechunk)
        self.recompress_checkbox.setChecked(operations.recompress)
        resampling = config.resampling
        self.resolution.setValue(resampling.resolution)
        self._set_combo_data(self.method, resampling.method)
        self.skipna.setChecked(resampling.skipna)
        self.na_thres.setValue(resampling.na_thres)
        self._set_combo_data(self.compute_dtype, resampling.compute_dtype)
        self.before_conditions.setText(resampling.before_conditions)
        self.before_results.setText(resampling.before_results)
        self.after_conditions.setText(resampling.after_conditions)
        self.after_results.setText(resampling.after_results)
        self._set_combo_data(self.statistics_policy, resampling.statistics_policy)
        chunking = config.chunking
        self._set_combo_data(self.strategy, chunking.strategy)
        self.target_mib.setValue(chunking.target_mib)
        self.final_workers.setValue(chunking.workers)
        if chunking.custom_chunks is not None:
            for box, value in zip(self.custom_chunks, chunking.custom_chunks):
                box.setValue(value)
        compression = config.compression
        self._set_combo_data(self.compression, compression.codec)
        if compression.level is not None:
            self.compression_level.setValue(compression.level)
        self._set_combo_data(self.compression_shuffle, compression.shuffle)

    def set_inspection(self, result: InspectionResult) -> None:
        self.inspection = result
        self.recovery = result.recovery
        self.plan = None
        self.run_button.setText("开始处理")
        self.run_button.setEnabled(False)
        if result.kind in {"zarr", "temporary"}:
            self.variables.setRowCount(0)
            self.conversion_checkbox.setChecked(False)
            self.conversion_group.setVisible(False)
            self._set_enabled(True)
            if result.kind == "temporary" and result.recovery is not None:
                self._apply_recovery_config(result.recovery.config)
                self.plan = result.recovery
                self.run_button.setText("继续执行")
                self.run_button.setEnabled(True)
                self.report.setPlainText(result.recovery.report)
                self.status.setText("恢复计划已载入，可直接继续执行；修改参数后需重新生成计划。")
            else:
                self.status.setText("现有 Zarr 检查结果已载入，请至少选择一项处理操作。")
            return
        self.conversion_checkbox.setChecked(True)
        self.conversion_group.setVisible(True)
        info = result.source_inventory
        self.variables.setRowCount(0)
        for name in info.variables:
            row = self.variables.rowCount()
            self.variables.insertRow(row)
            spec = info.variables[name]
            supported = spec.direct_compatible
            check = QCheckBox()
            check.setChecked(supported)
            check.setEnabled(supported)
            if not supported:
                check.setToolTip("辅助坐标、边界或 CRS 元数据变量不参与一条龙栅格处理。")
            check.toggled.connect(self._invalidate_plan)
            self.variables.setCellWidget(row, 0, check)
            self.variables.setItem(row, 1, QTableWidgetItem(name))
            output_name = QLineEdit(name)
            output_name.textChanged.connect(self._invalidate_plan)
            self.variables.setCellWidget(row, 2, output_name)
            fill_edit = QLineEdit()
            fill_edit.textChanged.connect(self._invalidate_plan)
            source_fill = ", ".join(
                str(spec.attrs[key])
                for key in ("_FillValue", "missing_value")
                if key in spec.attrs
            )
            fill_edit.setPlaceholderText(f"源: {source_fill}" if source_fill else "不处理")
            self.variables.setCellWidget(row, 3, fill_edit)
            scale_edit = QLineEdit()
            scale_edit.textChanged.connect(self._invalidate_plan)
            scale_edit.setPlaceholderText(
                f"源: {spec.attrs['scale_factor']}"
                if "scale_factor" in spec.attrs
                else "不处理"
            )
            self.variables.setCellWidget(row, 4, scale_edit)
            output_fill = QLineEdit()
            output_fill.textChanged.connect(self._invalidate_plan)
            output_fill.setPlaceholderText("默认浮点为 NaN")
            self.variables.setCellWidget(row, 5, output_fill)
        if len(info.times):
            self.time_start.setDate(_qdate(info.times[0]))
            self.time_end.setDate(_qdate(info.times[-1]))
        self.status.setText("检查结果已载入，请选择处理操作并生成计划。")
        self._set_enabled(True)

    def clear_inspection(self) -> None:
        self.inspection = None
        self.recovery = None
        self.plan = None
        self.run_button.setEnabled(False)
        self.variables.setRowCount(0)
        self.status.setText("检查参数已改变，请返回数据检查页面重新检查。")
        self._set_enabled(False)

    def _update_operation_state(self, *_args) -> None:
        available = self.inspection is not None
        resample = self.resample_checkbox.isChecked()
        rechunk = self.rechunk_checkbox.isChecked()
        recompress = self.recompress_checkbox.isChecked()
        self.resampling_group.setVisible(resample)
        self.chunking_group.setVisible(rechunk)
        self.compression_group.setVisible(recompress)
        self.resampling_group.setEnabled(available and resample)
        self.chunking_group.setEnabled(available and rechunk)
        self.compression_group.setEnabled(available and recompress)
        custom = rechunk and self.strategy.currentData() == "custom"
        for box in self.custom_chunks:
            box.setEnabled(custom and available)

    def _selected_variables(self) -> tuple[str, ...]:
        selected = []
        for row in range(self.variables.rowCount()):
            check = self.variables.cellWidget(row, 0)
            if check is not None and check.isChecked():
                selected.append(self.variables.item(row, 1).text())
        return tuple(selected)

    def _variable_names(self) -> dict[str, str]:
        result = {}
        for row in range(self.variables.rowCount()):
            source = self.variables.item(row, 1).text()
            edit = self.variables.cellWidget(row, 2)
            value = edit.text().strip() if isinstance(edit, QLineEdit) else source
            if value and value != source:
                result[source] = value
        return result

    def _variable_settings(self) -> tuple[dict[str, str], dict[str, VariableTransform]]:
        names: dict[str, str] = {}
        transforms: dict[str, VariableTransform] = {}
        for row in range(self.variables.rowCount()):
            check = self.variables.cellWidget(row, 0)
            if check is None or not check.isChecked():
                continue
            source = self.variables.item(row, 1).text()
            output = self.variables.cellWidget(row, 2).text().strip() or source
            fill_values = ConversionPage._parse_fill_values(
                self.variables.cellWidget(row, 3).text()
            )
            scale_factor = ConversionPage._parse_number(
                self.variables.cellWidget(row, 4).text(), "缩放因子"
            )
            output_fill = ConversionPage._parse_number(
                self.variables.cellWidget(row, 5).text(), "输出填充值"
            )
            if output != source:
                names[source] = output
            if fill_values is not None or scale_factor is not None or output_fill is not None:
                transforms[source] = VariableTransform(
                    fill_values=fill_values,
                    scale_factor=scale_factor,
                    output_fill=output_fill,
                )
        return names, transforms

    def _config(self) -> PipelineConfig:
        if self.inspection is None:
            raise ValueError("请先完成数据检查与时间规则确认。")
        if not self.output.text().strip():
            raise ValueError("请选择最终输出目录。")
        strategy = self.strategy.currentData()
        custom = tuple(box.value() for box in self.custom_chunks) if strategy == "custom" else None
        is_processed = self.inspection.kind in {"zarr", "temporary"}
        recovery_config = self.recovery.config if self.recovery is not None else None
        selected_variables = self._selected_variables()
        if not is_processed and not selected_variables:
            raise ValueError("至少选择一个转换变量。")
        if is_processed and not any(
            (
                self.resample_checkbox.isChecked(),
                self.rechunk_checkbox.isChecked(),
                self.recompress_checkbox.isChecked(),
            )
        ):
            raise ValueError("现有 Zarr 或临时检查点至少需要选择一项处理操作。")
        variable_names, variable_transforms = self._variable_settings()
        recovered_resampling = (
            recovery_config.resampling if recovery_config is not None else None
        )
        return PipelineConfig(
            input=PipelineInput(kind="zarr" if is_processed else "raw"),
            general=PipelineGeneralConfig(
                output=Path(self.output.text().strip()),
                temporary_dir=Path(self.temporary_dir.text().strip()) if self.temporary_dir.text().strip() else None,
                time_start=self.time_start.date().toString(Qt.DateFormat.ISODate),
                time_end=self.time_end.date().toString(Qt.DateFormat.ISODate),
                lat_min=self.lat_min.value(),
                lat_max=self.lat_max.value(),
                lon_min=self.lon_min.value(),
                lon_max=self.lon_max.value(),
                cleanup_intermediate=self.cleanup_intermediate.isChecked(),
                overwrite=(
                    recovery_config.general.overwrite
                    if recovery_config is not None
                    else False
                ),
            ),
            conversion=PipelineConversionOptions(
                variables=selected_variables,
                variable_names=variable_names,
                variable_transforms=variable_transforms,
                auto_tune=self.auto_tune.isChecked(),
                tune_budget=self.tune_budget.value(),
                max_workers=self.max_workers.value() or None,
            ),
            operations=PipelineOperations(
                resample=self.resample_checkbox.isChecked(),
                rechunk=self.rechunk_checkbox.isChecked(),
                recompress=self.recompress_checkbox.isChecked(),
            ),
            resampling=PipelineResamplingOptions(
                resolution=self.resolution.value(),
                method=self.method.currentData(),
                skipna=self.skipna.isChecked(),
                na_thres=self.na_thres.value(),
                compute_dtype=self.compute_dtype.currentData(),
                tile_size=(
                    recovered_resampling.tile_size
                    if recovered_resampling is not None
                    else "auto"
                ),
                time_block=(
                    recovered_resampling.time_block
                    if recovered_resampling is not None
                    else "auto"
                ),
                compute_workers=(
                    recovered_resampling.compute_workers
                    if recovered_resampling is not None
                    else 2
                ),
                space_workers=(
                    recovered_resampling.space_workers
                    if recovered_resampling is not None
                    else "auto"
                ),
                before_conditions=self.before_conditions.text().strip(),
                before_results=self.before_results.text().strip(),
                after_conditions=self.after_conditions.text().strip(),
                after_results=self.after_results.text().strip(),
                statistics_policy=self.statistics_policy.currentData(),
            ),
            chunking=PipelineChunkingOptions(
                strategy=strategy,
                target_mib=self.target_mib.value(),
                custom_chunks=custom,
                workers=self.final_workers.value(),
            ),
            compression=PipelineCompressionOptions(
                profile=(
                    recovery_config.compression.profile
                    if recovery_config is not None
                    else "balanced"
                ),
                codec=self.compression.currentData(),
                level=self.compression_level.value(),
                shuffle=self.compression_shuffle.currentData(),
            ),
            validate=(recovery_config.validate if recovery_config is not None else True),
        )

    def _request_preview(self) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        self.task_requested.emit(
            "生成一条龙计划",
            lambda cancel_event: _run_cancelable(
                cancel_event, lambda: preview_pipeline(self.inspection, config)
            ),
            self._preview_done,
        )

    def _request_run(self) -> None:
        try:
            config = self._config()
        except ValueError as exc:
            QMessageBox.warning(self, "参数不完整", str(exc))
            return
        self.task_requested.emit(
            "执行一条龙处理",
            lambda cancel_event: run_pipeline(self.inspection, config, cancel_event=cancel_event),
            self._run_done,
        )

    def _preview_done(self, plan) -> None:
        self.plan = plan
        self.run_button.setEnabled(True)
        decisions = "\n".join(
            f"{item.operation}: {item.disposition} - {item.reason}"
            for item in plan.operation_decisions
        )
        if isinstance(plan, ZarrPipelinePlan):
            dimensions = (
                plan.resample_plan.output_dimensions
                if plan.resample_plan is not None
                else plan.input_info.dimensions
            )
            self.report.setPlainText(
                "现有 Zarr 处理计划\n"
                f"目标 shape(time, lat, lon)=({dimensions['time']}, {dimensions['lat']}, {dimensions['lon']})\n"
                f"最终 chunks(time, lat, lon)={plan.final_chunks}\n"
                f"需要重采样={'是' if plan.needs_resample else '否'}\n"
                f"需要最终化={'是' if plan.finalization_required else '否'}\n"
                f"最终压缩={plan.final_compression.description if plan.final_compression else '保持'}\n"
                f"采样前替换={plan.resample_plan.before_replacements.as_pairs() if plan.resample_plan else ()}\n"
                f"采样后替换={plan.resample_plan.after_replacements.as_pairs() if plan.resample_plan else ()}\n"
                f"操作决策：\n{decisions}"
            )
        else:
            self.report.setPlainText(
                "处理计划\n"
                f"目标 shape(lat, lon)=({plan.target_grid.lat.size}, {plan.target_grid.lon.size})\n"
                f"目标范围={plan.target_grid.spatial_extent}\n"
                f"最终 chunks(time, lat, lon)={plan.final_chunks}\n"
                f"源读取窗口=lat {plan.source_read_window.lat_bounds}，"
                f"lon {plan.source_read_window.lon_bounds}\n"
                f"halo={plan.source_read_window.halo_description}\n"
                f"需要实际重采样={'是' if plan.needs_resample else '否'}\n"
                f"最终压缩={plan.final_compression.description if plan.final_compression else '保持'}\n"
                f"采样前替换={self.before_conditions.text().strip() or '无'}\n"
                f"采样后替换={self.after_conditions.text().strip() or '无'}\n"
                f"操作决策：\n{decisions}"
                + (f"\n覆盖提醒：{plan.coverage_warning}" if plan.coverage_warning else "")
            )
        self.status.setText("计划已生成，请确认报告后开始处理。")

    def _run_done(self, result) -> None:
        self.status.setText(f"一条龙处理完成：{result.get('output', '')}")
        self.report.append(f"\n执行结果：{result}")


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"快速 Zarr 转换器 v{__version__}")
        self.resize(1280, 820)
        self._apply_modern_theme()
        self.worker: TaskWorker | None = None
        self.source_result: InspectionResult | None = None
        self.zarr_result: InspectionResult | None = None

        self.task_page = TaskPage()
        self.inspection_page = InspectionPage()
        self.conversion_page = ConversionPage()
        self.rechunk_page = RechunkPage()
        self.resample_page = ResamplePage()
        self.pipeline_page = PipelinePage()
        pages = [
            ("数据检查", self.inspection_page),
            ("转换", self.conversion_page),
            ("Zarr 优化", self.rechunk_page),
            ("Zarr 重采样", self.resample_page),
            ("一条龙", self.pipeline_page),
            ("任务与日志", self.task_page),
        ]
        self.navigation = QListWidget()
        self.navigation.setFixedWidth(150)
        self.stack = QStackedWidget()
        self.task_page_index = len(pages) - 1
        for index, (label, page) in enumerate(pages):
            self.navigation.addItem(QListWidgetItem(label))
            if index == 0:
                container = QScrollArea()
                container.setWidgetResizable(True)
                container.setFrameShape(QScrollArea.Shape.NoFrame)
                container.setWidget(page)
                self.stack.addWidget(container)
            else:
                if index in (1, 2, 3):
                    page.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
                self.stack.addWidget(page)
        self.stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.navigation.item(0).setText("数据检查")
        self.navigation.item(4).setText("处理流程")
        self.navigation.item(5).setText("任务中心")
        self.navigation.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        # Conversion depends on a source inspection; Zarr optimization has
        # its own Zarr input and is intentionally usable independently.
        for index in (1, 4):
            self._set_nav_enabled(index, False)
        # v1.4 makes the composable workflow the primary entry point.  The
        # legacy pages stay instantiated for API and automation compatibility,
        # but are removed from the visual navigation surface.
        for index in (1, 2, 3):
            self.navigation.item(index).setHidden(True)

        splitter = QSplitter()
        splitter.addWidget(self.navigation)
        splitter.addWidget(self.stack)
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self._make_menu()

        self.inspection_page.task_requested.connect(self._task_requested)
        self.conversion_page.task_requested.connect(self._task_requested)
        self.rechunk_page.task_requested.connect(self._task_requested)
        self.resample_page.task_requested.connect(self._task_requested)
        self.pipeline_page.task_requested.connect(self._task_requested)
        self.inspection_page.result_ready.connect(self._source_ready)
        self.inspection_page.zarr_result_ready.connect(self._workflow_zarr_ready)
        self.inspection_page.result_invalidated.connect(self._inspection_invalidated)
        self.rechunk_page.result_ready.connect(self._zarr_ready)

    def _apply_modern_theme(self) -> None:
        """Install a restrained Fusion/QSS theme without changing behavior."""
        app = QApplication.instance()
        if app is None:
            return
        app.setStyle("Fusion")
        app.setStyleSheet(
            """
            QWidget { font-size: 13px; color: #111827; }
            QWidget:disabled { color: #4b5563; }
            QMainWindow { background: #f5f7fb; }
            QLabel, QCheckBox, QRadioButton { color: #111827; }
            QListWidget { background: #17212b; color: #f1f5f9; border: 0; padding: 12px 6px; }
            QListWidget::item { padding: 11px 14px; margin: 2px 0; border-radius: 6px; }
            QListWidget::item:selected { background: #1d63c6; color: #ffffff; }
            QListWidget::item:disabled { color: #aebdca; }
            QGroupBox { border: 1px solid #c7d2df; border-radius: 6px; margin-top: 14px; padding: 12px; background: white; color: #111827; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #193b5a; }
            QPushButton { background: #1d63c6; color: #ffffff; border: 0; border-radius: 5px; padding: 7px 14px; }
            QPushButton:hover { background: #174fa0; }
            QPushButton:disabled { background: #cbd5e1; color: #374151; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QTableWidget { border: 1px solid #b8c5d3; border-radius: 4px; background: #ffffff; color: #111827; padding: 4px; }
            QHeaderView::section { background: #e8eef5; color: #172033; border: 0; border-right: 1px solid #c7d2df; border-bottom: 1px solid #c7d2df; padding: 5px; }
            QToolTip { color: #111827; background: #fffbea; border: 1px solid #9aa8b8; }
            QProgressBar { border: 1px solid #c9d4df; border-radius: 4px; text-align: center; background: white; }
            QProgressBar::chunk { background: #27ae60; }
            """
        )

    def _make_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        action = QAction("退出", self)
        action.triggered.connect(self.close)
        file_menu.addAction(action)
        help_menu = self.menuBar().addMenu("帮助")
        about = QAction("关于", self)
        about.triggered.connect(
            lambda: QMessageBox.information(
                self,
                "关于",
                f"快速 Zarr 转换器 v{__version__}\n\n源数据检查、转换、重分块、重压缩和重采样共享同一套核心引擎。",
            )
        )
        help_menu.addAction(about)

    def _inspection_invalidated(self) -> None:
        self.source_result = None
        self.zarr_result = None
        self.pipeline_page.clear_inspection()
        for index in (1, 4):
            self._set_nav_enabled(index, False)
        self.navigation.item(1).setText("转换")
        self.navigation.item(4).setText("处理流程")
        self.navigation.setCurrentRow(0)

    def _source_ready(self, result: InspectionResult) -> None:
        self.source_result = result
        self.conversion_page.set_inspection(result)
        self.pipeline_page.set_inspection(result)
        for index in (1, 4):
            self._set_nav_enabled(index, True)
        self.navigation.item(1).setText("转换 ✓")
        self.navigation.item(4).setText("处理流程 ✓")
        self.navigation.setCurrentRow(4)

    def _workflow_zarr_ready(self, result: InspectionResult) -> None:
        self.zarr_result = result
        self.pipeline_page.set_inspection(result)
        self._set_nav_enabled(4, True)
        self.navigation.item(4).setText("处理流程 ✓")
        self.navigation.setCurrentRow(4)

    def _zarr_ready(self, result: InspectionResult) -> None:
        self.zarr_result = result
        self.navigation.item(2).setText("Zarr 优化 ✓")

    def _set_nav_enabled(self, index: int, enabled: bool) -> None:
        item = self.navigation.item(index)
        flags = item.flags()
        if enabled:
            item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled)
        else:
            item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)

    def _task_storage_paths(self, sender: object) -> tuple[tuple[str, str], ...]:
        paths: list[tuple[str, str]] = []

        def add(role: str, value: object) -> None:
            text = value.text().strip() if isinstance(value, QLineEdit) else str(value or "").strip()
            if text:
                paths.append((role, text))

        if sender is self.inspection_page:
            add("输入", self.inspection_page.path)
        elif sender is self.pipeline_page:
            inspection = self.pipeline_page.inspection
            if inspection is not None and inspection.kind == "source":
                for record in inspection.source_inventory.files:
                    add("输入", record.path)
            elif inspection is not None:
                add("输入", inspection.path)
            add("临时", self.pipeline_page.temporary_dir)
            add("输出", self.pipeline_page.output)
        elif sender is self.rechunk_page:
            add("输入", self.rechunk_page.input)
            add("临时", self.rechunk_page.temporary_dir)
            add("输出", self.rechunk_page.output)
        elif sender is self.resample_page:
            add("输入", self.resample_page.input)
            add("临时", self.resample_page.temporary_dir)
            add("输出", self.resample_page.output)
        elif sender is self.conversion_page:
            if self.source_result is not None:
                for record in self.source_result.source_inventory.files:
                    add("输入", record.path)
            add("输出", self.conversion_page.output)
        return tuple(dict.fromkeys(paths))

    def _task_requested(self, label: str, function: Callable[[], Any], callback: TaskCallback) -> None:
        if self.worker is not None and self.worker.isRunning():
            QMessageBox.warning(self, "已有任务运行", "请等待当前任务完成或先请求取消。")
            return
        self.navigation.setCurrentRow(self.task_page_index)
        worker = TaskWorker(
            function,
            self,
            storage_paths=self._task_storage_paths(self.sender()),
        )
        self.worker = worker
        self.task_page.started(label, worker.request_cancel)
        worker.log.connect(self.task_page.append)
        worker.resource.connect(self.task_page.update_resource)
        worker.progress.connect(self.task_page.update_progress)

        def succeeded(result: Any) -> None:
            try:
                callback(result)
                self.task_page.completed("任务完成。")
            except Exception as exc:  # noqa: BLE001
                self.task_page.append(f"结果处理失败：{exc}")
                self.task_page.failed()

        def failed(details: str) -> None:
            self.task_page.append(details)
            self.task_page.failed()
            summary = next(
                (line.strip() for line in reversed(details.splitlines()) if line.strip()),
                "未知错误",
            )
            QMessageBox.critical(self, "任务失败", summary)

        def cancelled() -> None:
            self.task_page.append("任务已取消。")
            self.task_page.cancelled()

        worker.succeeded.connect(succeeded)
        worker.failed.connect(failed)
        worker.cancelled.connect(cancelled)
        worker.finished.connect(self._task_finished)
        worker.start()

    def _task_finished(self) -> None:
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()
