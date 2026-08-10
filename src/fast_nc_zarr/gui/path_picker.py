from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Literal

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from .path_chooser import PathChooserDialog


PathPickerMode = Literal["directory", "open_file", "save_file"]


@dataclass(frozen=True, slots=True)
class FavoritePath:
    """One named favorite in its persisted display order."""

    path: str
    name: str
    order: int = 0

    def to_dict(self) -> dict[str, str | int]:
        return {"path": self.path, "name": self.name, "order": self.order}


class PathPickerSettings:
    """Non-fatal QSettings storage for favorite and recent directories."""

    ORGANIZATION = "fast-nc-zarr"
    APPLICATION = "快速 Zarr 转换器"
    ROOT_KEY = "pathPicker/v2"
    LEGACY_ROOT_KEY = "pathPicker/v1"
    MAX_RECENT_DIRECTORIES = 12

    def __init__(self, settings: QSettings | None = None) -> None:
        self.settings = settings or QSettings(self.ORGANIZATION, self.APPLICATION)
        self.last_error: str | None = None
        self._migrate_legacy_settings()

    @staticmethod
    def _has_value(value: Any) -> bool:
        return value not in (None, "", [], (), {})

    def _migrate_legacy_settings(self) -> None:
        """Copy v1 settings once without deleting the user's old values."""
        try:
            values: dict[str, Any] = {}
            for suffix in ("favorites", "recentDirectories"):
                current = self.settings.value(self._key(suffix), None)
                legacy = self.settings.value(
                    f"{self.LEGACY_ROOT_KEY}/{suffix}", None
                )
                if not self._has_value(current) and self._has_value(legacy):
                    values[self._key(suffix)] = legacy

            all_keys = getattr(self.settings, "allKeys", None)
            if callable(all_keys):
                prefix = f"{self.LEGACY_ROOT_KEY}/lastDirectory/"
                for key in all_keys():
                    if not str(key).startswith(prefix):
                        continue
                    suffix = str(key)[len(self.LEGACY_ROOT_KEY) + 1 :]
                    target = self._key(suffix)
                    if not self._has_value(self.settings.value(target, None)):
                        values[target] = self.settings.value(key, "")
            if values:
                self._set_values(values)
        except Exception as exc:  # noqa: BLE001 - settings are non-fatal
            self.last_error = str(exc)

    @staticmethod
    def normalize_path(value: str | os.PathLike[str]) -> str:
        raw = os.fspath(value).strip()
        if not raw:
            return ""
        try:
            return str(Path(os.path.expandvars(raw)).expanduser().resolve(strict=False))
        except (OSError, RuntimeError, ValueError):
            try:
                return os.path.abspath(os.path.normpath(os.path.expandvars(raw)))
            except (OSError, ValueError):
                return raw

    @staticmethod
    def _identity(value: str) -> str:
        return os.path.normcase(os.path.normpath(value))

    @staticmethod
    def _default_name(path: str) -> str:
        value = Path(path)
        return value.name or value.anchor or path

    @staticmethod
    def directory_available(path: str) -> bool:
        try:
            return Path(path).is_dir()
        except OSError:
            return False

    def _key(self, suffix: str) -> str:
        return f"{self.ROOT_KEY}/{suffix}"

    @staticmethod
    def _role_fragment(role: str) -> str:
        fragment = "".join(
            character if character.isalnum() or character in "-_." else "_"
            for character in role.strip()
        )
        return fragment or "default"

    def _record_status(self) -> None:
        try:
            status = self.settings.status()
        except Exception as exc:  # noqa: BLE001 - third-party settings backends may fail
            self.last_error = str(exc)
            return
        if status != QSettings.Status.NoError:
            self.last_error = f"QSettings persistence status: {status.name}"

    def _sync(self) -> None:
        try:
            self.settings.sync()
            self._record_status()
        except Exception as exc:  # noqa: BLE001 - persistence must remain non-fatal
            self.last_error = str(exc)

    def _value(self, key: str, default: Any = None) -> Any:
        self._sync()
        try:
            return self.settings.value(key, default)
        except Exception as exc:  # noqa: BLE001 - persistence must remain non-fatal
            self.last_error = str(exc)
            return default

    def _set_values(self, values: dict[str, Any]) -> None:
        try:
            for key, value in values.items():
                self.settings.setValue(key, value)
            self._sync()
        except Exception as exc:  # noqa: BLE001 - persistence must remain non-fatal
            self.last_error = str(exc)

    def _json_value(self, key: str) -> list[Any]:
        raw = self._value(key, "")
        if isinstance(raw, (list, tuple)):
            return list(raw)
        if not isinstance(raw, str) or not raw.strip():
            return []
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            self.last_error = f"Invalid path picker settings at {key}: {exc}"
            return []
        return list(value) if isinstance(value, list) else []

    @staticmethod
    def _json(items: list[Any]) -> str:
        return json.dumps(items, ensure_ascii=False, separators=(",", ":"))

    def favorites(self) -> tuple[FavoritePath, ...]:
        loaded: list[tuple[int, int, FavoritePath]] = []
        seen: set[str] = set()
        for index, raw in enumerate(self._json_value(self._key("favorites"))):
            if isinstance(raw, str):
                path = self.normalize_path(raw)
                name = self._default_name(path) if path else ""
                order = index
            elif isinstance(raw, dict):
                path = self.normalize_path(str(raw.get("path", "")))
                name = str(raw.get("name", "")).strip()
                try:
                    order = int(raw.get("order", index))
                except (TypeError, ValueError):
                    order = index
            else:
                continue
            if not path:
                continue
            identity = self._identity(path)
            if identity in seen:
                continue
            seen.add(identity)
            entry = FavoritePath(path, name or self._default_name(path), order)
            loaded.append((order, index, entry))
        loaded.sort(key=lambda item: (item[0], item[1]))
        return tuple(
            FavoritePath(entry.path, entry.name, order)
            for order, (_, _, entry) in enumerate(loaded)
        )

    def _save_favorites(self, favorites: list[FavoritePath]) -> None:
        normalized = [
            FavoritePath(item.path, item.name, order).to_dict()
            for order, item in enumerate(favorites)
        ]
        self._set_values({self._key("favorites"): self._json(normalized)})

    def favorite(self, path: str | os.PathLike[str]) -> FavoritePath | None:
        normalized = self.normalize_path(path)
        if not normalized:
            return None
        identity = self._identity(normalized)
        return next(
            (item for item in self.favorites() if self._identity(item.path) == identity),
            None,
        )

    def add_favorite(
        self, path: str | os.PathLike[str], *, name: str | None = None
    ) -> FavoritePath | None:
        normalized = self.normalize_path(path)
        if not normalized:
            return None
        favorites = list(self.favorites())
        identity = self._identity(normalized)
        for item in favorites:
            if self._identity(item.path) == identity:
                if name is not None and name.strip() and item.name != name.strip():
                    self.rename_favorite(normalized, name.strip())
                    return self.favorite(normalized)
                return item
        entry = FavoritePath(
            normalized,
            name.strip() if name is not None and name.strip() else self._default_name(normalized),
            len(favorites),
        )
        favorites.append(entry)
        self._save_favorites(favorites)
        return entry

    def remove_favorite(self, path: str | os.PathLike[str]) -> bool:
        normalized = self.normalize_path(path)
        identity = self._identity(normalized)
        favorites = list(self.favorites())
        kept = [item for item in favorites if self._identity(item.path) != identity]
        if len(kept) == len(favorites):
            return False
        self._save_favorites(kept)
        return True

    def rename_favorite(self, path: str | os.PathLike[str], name: str) -> bool:
        normalized = self.normalize_path(path)
        identity = self._identity(normalized)
        cleaned_name = name.strip()
        if not normalized or not cleaned_name:
            return False
        favorites = list(self.favorites())
        changed = False
        updated: list[FavoritePath] = []
        for item in favorites:
            if self._identity(item.path) == identity:
                updated.append(FavoritePath(item.path, cleaned_name, item.order))
                changed = item.name != cleaned_name
            else:
                updated.append(item)
        if changed:
            self._save_favorites(updated)
        return changed

    def move_favorite(self, path: str | os.PathLike[str], offset: int) -> bool:
        normalized = self.normalize_path(path)
        identity = self._identity(normalized)
        favorites = list(self.favorites())
        index = next(
            (position for position, item in enumerate(favorites) if self._identity(item.path) == identity),
            -1,
        )
        target = index + offset
        if index < 0 or target < 0 or target >= len(favorites):
            return False
        favorites[index], favorites[target] = favorites[target], favorites[index]
        self._save_favorites(favorites)
        return True

    def recent_directories(self) -> tuple[str, ...]:
        values: list[str] = []
        seen: set[str] = set()
        for raw in self._json_value(self._key("recentDirectories")):
            if not isinstance(raw, str):
                continue
            path = self.normalize_path(raw)
            if not path:
                continue
            identity = self._identity(path)
            if identity in seen:
                continue
            seen.add(identity)
            values.append(path)
            if len(values) == self.MAX_RECENT_DIRECTORIES:
                break
        return tuple(values)

    def clear_recent_directories(self) -> None:
        self._set_values({self._key("recentDirectories"): self._json([])})

    def remember_directory(self, role: str, path: str | os.PathLike[str]) -> None:
        normalized = self.normalize_path(path)
        if not normalized:
            return
        identity = self._identity(normalized)
        recent = [
            item
            for item in self.recent_directories()
            if self._identity(item) != identity
        ]
        recent.insert(0, normalized)
        recent = recent[: self.MAX_RECENT_DIRECTORIES]
        self._set_values(
            {
                self._key(f"lastDirectory/{self._role_fragment(role)}"): normalized,
                self._key("recentDirectories"): self._json(recent),
            }
        )

    def last_directory(self, role: str) -> str:
        raw = self._value(
            self._key(f"lastDirectory/{self._role_fragment(role)}"), ""
        )
        return self.normalize_path(str(raw)) if raw else ""

    def directory_for_path(self, value: str, mode: PathPickerMode) -> str:
        normalized = self.normalize_path(value)
        if not normalized:
            return ""
        return normalized if mode == "directory" else str(Path(normalized).parent)

    def remember_selection(self, role: str, value: str, mode: PathPickerMode) -> None:
        directory = self.directory_for_path(value, mode)
        if directory:
            self.remember_directory(role, directory)

    def dialog_start(self, role: str, preferred: str, mode: PathPickerMode) -> str:
        normalized = self.normalize_path(preferred)
        if normalized:
            candidate = Path(normalized)
            try:
                if mode == "directory" and candidate.is_dir():
                    return normalized
                if mode != "directory" and candidate.parent.is_dir():
                    return normalized
            except OSError:
                pass

        candidates = (
            self.last_directory(role),
            *self.recent_directories(),
            *(item.path for item in self.favorites()),
        )
        seen: set[str] = set()
        for value in candidates:
            if not value:
                continue
            identity = self._identity(value)
            if identity in seen:
                continue
            seen.add(identity)
            if self.directory_available(value):
                return value
        try:
            return str(Path.cwd())
        except OSError:
            return str(Path.home())


class FavoriteManagerDialog(QDialog):
    """Searchable favorite manager used by every path field."""

    def __init__(self, settings: PathPickerSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.path_settings = settings
        self.selected_path = ""
        self.setWindowTitle("管理路径收藏")
        self.setMinimumSize(620, 420)

        layout = QVBoxLayout(self)
        self.search = QLineEdit()
        self.search.setPlaceholderText("搜索收藏名称或完整路径")
        self.search.setAccessibleName("搜索路径收藏")
        layout.addWidget(self.search)

        self.list = QListWidget()
        self.list.setAccessibleName("路径收藏列表")
        self.list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        layout.addWidget(self.list, 1)

        actions = QHBoxLayout()
        self.use_button = QPushButton("使用")
        self.use_button.setObjectName("secondaryButton")
        self.rename_button = QPushButton("重命名")
        self.rename_button.setObjectName("secondaryButton")
        self.move_up_button = QPushButton("上移")
        self.move_up_button.setObjectName("secondaryButton")
        self.move_down_button = QPushButton("下移")
        self.move_down_button.setObjectName("secondaryButton")
        self.remove_button = QPushButton("删除")
        self.remove_button.setObjectName("dangerButton")
        for button in (
            self.use_button,
            self.rename_button,
            self.move_up_button,
            self.move_down_button,
            self.remove_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        close_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        close_box.rejected.connect(self.reject)
        layout.addWidget(close_box)
        self.search.textChanged.connect(self._refresh)
        self.list.currentItemChanged.connect(self._update_actions)
        self.list.itemDoubleClicked.connect(lambda _item: self._use_selected())
        self.use_button.clicked.connect(self._use_selected)
        self.rename_button.clicked.connect(self._rename_selected)
        self.move_up_button.clicked.connect(lambda: self._move_selected(-1))
        self.move_down_button.clicked.connect(lambda: self._move_selected(1))
        self.remove_button.clicked.connect(self._remove_selected)
        self._refresh()

    def _favorites(self) -> tuple[FavoritePath, ...]:
        query = self.search.text().strip().casefold()
        favorites = self.path_settings.favorites()
        if not query:
            return favorites
        return tuple(
            item
            for item in favorites
            if query in item.name.casefold() or query in item.path.casefold()
        )

    def _refresh(self) -> None:
        self.list.clear()
        for favorite in self._favorites():
            available = self.path_settings.directory_available(favorite.path)
            marker = "可访问" if available else "不可访问"
            item = QListWidgetItem(f"{favorite.name}  ·  {favorite.path}  ·  {marker}")
            item.setData(Qt.ItemDataRole.UserRole, favorite.path)
            item.setToolTip(favorite.path)
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
        self._update_actions()

    def _selected(self) -> FavoritePath | None:
        item = self.list.currentItem()
        if item is None:
            return None
        path = item.data(Qt.ItemDataRole.UserRole)
        return self.path_settings.favorite(str(path)) if path else None

    def _update_actions(self, *_args) -> None:
        favorite = self._selected()
        enabled = favorite is not None
        for button in (
            self.use_button,
            self.rename_button,
            self.move_up_button,
            self.move_down_button,
            self.remove_button,
        ):
            button.setEnabled(enabled)
        if favorite is not None:
            favorites = self.path_settings.favorites()
            self.move_up_button.setEnabled(favorite.order > 0)
            self.move_down_button.setEnabled(favorite.order < len(favorites) - 1)

    def _use_selected(self) -> None:
        favorite = self._selected()
        if favorite is None:
            return
        self.selected_path = favorite.path
        self.accept()

    def _rename_selected(self) -> None:
        favorite = self._selected()
        if favorite is None:
            return
        name, accepted = QInputDialog.getText(
            self, "重命名收藏", "显示名称", QLineEdit.EchoMode.Normal, favorite.name
        )
        if accepted and name.strip():
            self.path_settings.rename_favorite(favorite.path, name)
            self._refresh()

    def _move_selected(self, offset: int) -> None:
        favorite = self._selected()
        if favorite is not None and self.path_settings.move_favorite(favorite.path, offset):
            self._refresh()

    def _remove_selected(self) -> None:
        favorite = self._selected()
        if favorite is not None:
            self.path_settings.remove_favorite(favorite.path)
            self._refresh()


class PathPicker(QWidget):
    """Reusable path editor with browsing and directory status."""

    textChanged = Signal(str)
    textEdited = Signal(str)
    editingFinished = Signal()
    returnPressed = Signal()

    def __init__(
        self,
        *,
        role: str,
        dialog_title: str,
        mode: PathPickerMode = "directory",
        accessible_name: str = "路径",
        settings: PathPickerSettings | QSettings | None = None,
        file_filter: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if mode not in {"directory", "open_file", "save_file"}:
            raise ValueError(f"Unsupported path picker mode: {mode}")
        self.role = role
        self.dialog_title = dialog_title
        self.mode: PathPickerMode = mode
        self.file_filter = file_filter
        self.path_settings = (
            settings
            if isinstance(settings, PathPickerSettings)
            else PathPickerSettings(settings)
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self.line_edit = QLineEdit()
        self.line_edit.setAccessibleName(accessible_name)
        self.path_status = QLabel("未设置")
        self.path_status.setObjectName("pathStatus")
        self.path_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.path_status.setMinimumWidth(58)
        self.browse_button = QPushButton("浏览")
        self.browse_button.setObjectName("pathPickerAuxButton")
        self.browse_button.setAccessibleName(f"浏览{accessible_name}")
        self.browse_button.setToolTip(f"浏览并选择{accessible_name}")
        self.browse_button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.path_status)
        layout.addWidget(self.browse_button)
        self.setFocusProxy(self.line_edit)
        super().setAccessibleName(accessible_name)

        self.line_edit.textChanged.connect(self._text_changed)
        self.line_edit.textEdited.connect(self.textEdited.emit)
        self.line_edit.editingFinished.connect(self._editing_finished)
        self.line_edit.returnPressed.connect(self.returnPressed.emit)
        self.browse_button.clicked.connect(self.browse)
        self._update_path_status()

    @property
    def persistence_error(self) -> str | None:
        return self.path_settings.last_error

    def text(self) -> str:
        return self.line_edit.text()

    def setText(self, value: str) -> None:
        self.line_edit.setText(value)

    def clear(self) -> None:
        self.line_edit.clear()

    def setPlaceholderText(self, value: str) -> None:
        self.line_edit.setPlaceholderText(value)

    def placeholderText(self) -> str:
        return self.line_edit.placeholderText()

    def setReadOnly(self, value: bool) -> None:
        self.line_edit.setReadOnly(value)
        self.browse_button.setEnabled(not value)

    def isReadOnly(self) -> bool:
        return self.line_edit.isReadOnly()

    def selectAll(self) -> None:
        self.line_edit.selectAll()

    def setToolTip(self, value: str) -> None:
        super().setToolTip(value)
        self.line_edit.setToolTip(value)

    def setAccessibleName(self, value: str) -> None:
        super().setAccessibleName(value)
        if hasattr(self, "line_edit"):
            self.line_edit.setAccessibleName(value)
            self.browse_button.setAccessibleName(f"浏览{value}")

    def _text_changed(self, value: str) -> None:
        self._update_path_status(value)
        self.textChanged.emit(value)

    def _editing_finished(self) -> None:
        value = self.text().strip()
        if value:
            self.path_settings.remember_selection(self.role, value, self.mode)
        self.editingFinished.emit()

    def _update_path_status(self, value: str | None = None) -> None:
        text = self.text().strip() if value is None else value.strip()
        if not text:
            status, label = "neutral", "未设置"
        else:
            directory = self.path_settings.directory_for_path(text, self.mode)
            available = self.path_settings.directory_available(directory)
            status, label = ("success", "可访问") if available else ("warning", "待验证")
        self.path_status.setText(label)
        self.path_status.setProperty("status", status)
        self.path_status.style().unpolish(self.path_status)
        self.path_status.style().polish(self.path_status)
        self.path_status.update()

    def _select_path(self, start: str) -> str:
        dialog = PathChooserDialog(
            self.path_settings,
            role=self.role,
            dialog_title=self.dialog_title,
            start=start,
            mode=self.mode,
            file_filter=self.file_filter,
            manager_factory=FavoriteManagerDialog,
            parent=self,
        )
        if dialog.exec() == QDialog.DialogCode.Accepted:
            return dialog.selected_path
        return ""

    def browse(self) -> None:
        start = self.path_settings.dialog_start(self.role, self.text().strip(), self.mode)
        value = self._select_path(start)
        if not value:
            return
        self.setText(value)
        self.path_settings.remember_selection(self.role, value, self.mode)
        self._update_path_status(value)
