from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class PathChooserDialog(QDialog):
    """Embedded file dialog with favorites and recent directories."""

    def __init__(
        self,
        settings: Any,
        *,
        role: str,
        dialog_title: str,
        start: str,
        mode: str,
        file_filter: str = "",
        manager_factory: Callable[[Any, QWidget | None], QDialog] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.path_settings = settings
        self.role = role
        self.dialog_title = dialog_title
        self.mode = mode
        self.manager_factory = manager_factory
        self.selected_path = ""
        self._closing = False
        self._ready = False
        self.setWindowTitle(dialog_title)
        self.setMinimumSize(980, 620)
        self.resize(1120, 700)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(8)
        heading = QLabel(
            "选择目录并使用左侧收藏/最近浏览"
            if mode == "directory"
            else "选择文件并使用左侧收藏/最近浏览目录"
        )
        heading.setObjectName("pageSubtitle")
        root.addWidget(heading)

        body = QHBoxLayout()
        body.setSpacing(10)
        panel = QGroupBox("收藏与最近浏览")
        panel.setObjectName("chooserPanel")
        panel.setMinimumWidth(300)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 12, 10, 10)
        panel_layout.setSpacing(7)
        hint = QLabel("点击目录可在右侧打开；双击或使用按钮确认。")
        hint.setObjectName("chooserHint")
        hint.setWordWrap(True)
        panel_layout.addWidget(hint)
        self.search = QLineEdit()
        self.search.setObjectName("chooserSearch")
        self.search.setPlaceholderText("搜索收藏名称或路径")
        self.search.setAccessibleName("搜索收藏和最近目录")
        panel_layout.addWidget(self.search)
        self.location_list = QListWidget()
        self.location_list.setObjectName("chooserLocations")
        self.location_list.setAccessibleName("收藏和最近目录列表")
        self.location_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        panel_layout.addWidget(self.location_list, 1)

        self.choose_location_button = QPushButton(
            "选择此目录" if mode == "directory" else "进入此目录"
        )
        self.choose_location_button.setObjectName("primaryAction")
        self.favorite_current_button = QPushButton("收藏当前目录")
        self.favorite_current_button.setObjectName("secondaryButton")
        self.manage_button = QPushButton("管理收藏")
        self.manage_button.setObjectName("secondaryButton")
        self.clear_recent_button = QPushButton("清除最近浏览")
        self.clear_recent_button.setObjectName("secondaryButton")
        panel_layout.addWidget(self.choose_location_button)
        panel_layout.addWidget(self.favorite_current_button)
        panel_layout.addWidget(self.manage_button)
        panel_layout.addWidget(self.clear_recent_button)
        body.addWidget(panel)

        self.file_dialog = QFileDialog(self)
        self.file_dialog.setObjectName("embeddedFileDialog")
        self.file_dialog.setWindowFlags(Qt.WindowType.Widget)
        self.file_dialog.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.file_dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.file_dialog.setWindowTitle(dialog_title)
        if mode == "directory":
            self.file_dialog.setFileMode(QFileDialog.FileMode.Directory)
            self.file_dialog.setOption(QFileDialog.Option.ShowDirsOnly, True)
        elif mode == "open_file":
            self.file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        else:
            self.file_dialog.setFileMode(QFileDialog.FileMode.AnyFile)
            self.file_dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        if file_filter:
            self.file_dialog.setNameFilter(file_filter)
        directory = settings.directory_for_path(start, mode) or start
        self.file_dialog.setDirectory(directory)
        body.addWidget(self.file_dialog, 1)
        root.addLayout(body, 1)

        self.search.textChanged.connect(self._refresh_locations)
        self.location_list.itemClicked.connect(self._location_clicked)
        self.location_list.itemDoubleClicked.connect(self._location_activated)
        self.location_list.currentItemChanged.connect(self._update_actions)
        self.choose_location_button.clicked.connect(self._choose_location)
        self.favorite_current_button.clicked.connect(self._toggle_current_favorite)
        self.manage_button.clicked.connect(self._open_manager)
        self.clear_recent_button.clicked.connect(self._clear_recent)
        self.file_dialog.directoryEntered.connect(self._directory_entered)
        self.file_dialog.fileSelected.connect(self._file_selected)
        self.file_dialog.filesSelected.connect(self._files_selected)
        self.file_dialog.accepted.connect(self._accepted_from_file_dialog)
        self.file_dialog.rejected.connect(self.reject)
        self._refresh_locations()
        self._ready = True
        self._directory_entered(self.file_dialog.directory().absolutePath(), remember=False)

    def _query(self) -> str:
        return self.search.text().strip().casefold()

    def _matches(self, name: str, path: str) -> bool:
        query = self._query()
        return not query or query in name.casefold() or query in path.casefold()

    def _add_header(self, text: str) -> None:
        item = QListWidgetItem(text)
        item.setForeground(QBrush(QColor("#000000")))
        item_font = item.font()
        item_font.setWeight(QFont.Weight.Bold)
        item.setFont(item_font)
        item.setFlags(
            item.flags()
            & ~Qt.ItemFlag.ItemIsEnabled
            & ~Qt.ItemFlag.ItemIsSelectable
        )
        self.location_list.addItem(item)

    def _add_location(self, prefix: str, name: str, path: str) -> None:
        if not self._matches(name, path):
            return
        available = self.path_settings.directory_available(path)
        state = "可访问" if available else "不可访问"
        item = QListWidgetItem(f"{prefix} {name} · {state}")
        item.setForeground(QBrush(QColor("#000000")))
        item_font = item.font()
        item_font.setWeight(QFont.Weight.Bold)
        item.setFont(item_font)
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path + ("\n当前不可访问，收藏已保留。" if not available else ""))
        self.location_list.addItem(item)

    def _refresh_locations(self) -> None:
        self.location_list.clear()
        favorites = self.path_settings.favorites()
        if favorites:
            self._add_header("收藏目录")
            for favorite in favorites:
                self._add_location("★", favorite.name, favorite.path)

        favorite_ids = {
            self.path_settings._identity(item.path) for item in favorites
        }
        recent = [
            path
            for path in self.path_settings.recent_directories()
            if self.path_settings._identity(path) not in favorite_ids
        ]
        if recent:
            self._add_header("最近浏览")
            for path in recent:
                self._add_location("↻", self.path_settings._default_name(path), path)

        if self.location_list.count() == 0:
            empty = QListWidgetItem("尚无匹配的收藏或最近目录")
            empty.setForeground(QBrush(QColor("#000000")))
            empty.setFlags(empty.flags() & ~Qt.ItemFlag.ItemIsEnabled)
            self.location_list.addItem(empty)
        self._update_actions()

    @staticmethod
    def _item_path(item: QListWidgetItem | None) -> str:
        if item is None:
            return ""
        value = item.data(Qt.ItemDataRole.UserRole)
        return str(value) if value else ""

    def _selected_location(self) -> str:
        return self._item_path(self.location_list.currentItem())

    def _current_directory(self) -> str:
        return self.path_settings.normalize_path(
            self.file_dialog.directory().absolutePath()
        )

    def _directory_entered(self, path: str, *, remember: bool = True) -> None:
        normalized = self.path_settings.normalize_path(path)
        if not normalized:
            return
        if remember and self._ready:
            self.path_settings.remember_directory(self.role, normalized)
        self._update_actions()

    def _location_clicked(self, item: QListWidgetItem) -> None:
        path = self._item_path(item)
        if path and self.path_settings.directory_available(path):
            self.file_dialog.setDirectory(path)
            self._directory_entered(path)

    def _location_activated(self, item: QListWidgetItem) -> None:
        path = self._item_path(item)
        if not path or not self.path_settings.directory_available(path):
            return
        if self.mode == "directory":
            self._finish(path)
        else:
            self.file_dialog.setDirectory(path)

    def _update_actions(self, *_args) -> None:
        current = self._current_directory()
        selected = self._selected_location()
        selected_available = bool(selected) and self.path_settings.directory_available(selected)
        if self.mode == "directory":
            self.choose_location_button.setText("选择此目录" if selected_available else "选择当前目录")
            self.choose_location_button.setEnabled(bool(selected_available or current))
        else:
            self.choose_location_button.setText("进入此目录")
            self.choose_location_button.setEnabled(bool(selected_available or current))
        favorite = self.path_settings.favorite(current) if current else None
        self.favorite_current_button.setText(
            "取消当前收藏" if favorite is not None else "收藏当前目录"
        )
        self.favorite_current_button.setEnabled(bool(current))
        self.clear_recent_button.setEnabled(bool(self.path_settings.recent_directories()))

    def _choose_location(self) -> None:
        current = self._current_directory()
        selected = self._selected_location()
        if self.mode == "directory":
            self._finish(selected if selected and self.path_settings.directory_available(selected) else current)
            return
        files = self.file_dialog.selectedFiles()
        if files:
            self._finish(files[0])
        elif selected and self.path_settings.directory_available(selected):
            self.file_dialog.setDirectory(selected)
        elif current:
            self.file_dialog.setDirectory(current)

    def _toggle_current_favorite(self) -> None:
        current = self._current_directory()
        if not current:
            return
        if self.path_settings.favorite(current) is None:
            self.path_settings.add_favorite(current)
        else:
            self.path_settings.remove_favorite(current)
        self._refresh_locations()
        self._update_actions()

    def _open_manager(self) -> None:
        if self.manager_factory is not None:
            dialog = self.manager_factory(self.path_settings, self)
        else:
            from .path_picker import FavoriteManagerDialog

            dialog = FavoriteManagerDialog(self.path_settings, self)
        dialog.exec()
        self._refresh_locations()

    def _clear_recent(self) -> None:
        self.path_settings.clear_recent_directories()
        self._refresh_locations()

    def _finish(self, path: str) -> None:
        if self._closing or not path:
            return
        normalized = self.path_settings.normalize_path(path)
        if not normalized:
            return
        self.selected_path = normalized
        self.path_settings.remember_selection(self.role, normalized, self.mode)
        self._closing = True
        self.accept()

    def _file_selected(self, path: str) -> None:
        self._finish(path)

    def _files_selected(self, paths: list[str]) -> None:
        if paths:
            self._finish(paths[0])

    def _accepted_from_file_dialog(self) -> None:
        if self._closing:
            return
        files = self.file_dialog.selectedFiles()
        if files:
            self._finish(files[0])
        elif self.mode == "directory":
            self._finish(self._current_directory())
