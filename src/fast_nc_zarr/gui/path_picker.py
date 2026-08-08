from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Literal

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QMenu,
    QPushButton,
    QWidget,
)


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
    """Non-fatal QSettings storage for favorite and recently used directories."""

    ORGANIZATION = "fast-nc-zarr"
    APPLICATION = "快速 Zarr 转换器"
    ROOT_KEY = "pathPicker/v1"
    MAX_RECENT_DIRECTORIES = 12

    def __init__(self, settings: QSettings | None = None) -> None:
        self.settings = settings or QSettings(self.ORGANIZATION, self.APPLICATION)
        self.last_error: str | None = None

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


class PathPicker(QWidget):
    """Reusable path editor with browsing, favorites, and recent directories."""

    textChanged = Signal(str)
    textEdited = Signal(str)
    editingFinished = Signal()
    returnPressed = Signal()
    favoriteChanged = Signal(str, bool)

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
        self.line_edit = QLineEdit()
        self.line_edit.setAccessibleName(accessible_name)
        self.browse_button = QPushButton("浏览…")
        self.browse_button.setObjectName("pathPickerAuxButton")
        self.browse_button.setAccessibleName(f"浏览{accessible_name}")
        self.browse_button.setToolTip(f"浏览并选择{accessible_name}")
        self.favorites_button = QPushButton("收藏夹")
        self.favorites_button.setObjectName("pathPickerAuxButton")
        self.favorites_button.setAccessibleName(f"打开{accessible_name}收藏夹")
        self.favorites_button.setToolTip("选择收藏或最近使用的目录")
        self.favorites_menu = QMenu(self.favorites_button)
        self.favorites_menu.setAccessibleName(f"{accessible_name}收藏与最近目录")
        self.favorites_button.setMenu(self.favorites_menu)
        self.star_button = QPushButton("☆")
        self.star_button.setObjectName("pathPickerAuxButton")
        self.star_button.setCheckable(True)
        self.star_button.setAccessibleName(f"收藏当前{accessible_name}")
        self.star_button.setToolTip("收藏当前目录")
        for button in (self.browse_button, self.favorites_button, self.star_button):
            button.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        layout.addWidget(self.line_edit, 1)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.favorites_button)
        layout.addWidget(self.star_button)
        self.setFocusProxy(self.line_edit)
        super().setAccessibleName(accessible_name)

        self.line_edit.textChanged.connect(self._text_changed)
        self.line_edit.textEdited.connect(self.textEdited.emit)
        self.line_edit.editingFinished.connect(self._editing_finished)
        self.line_edit.returnPressed.connect(self.returnPressed.emit)
        self.browse_button.clicked.connect(self.browse)
        self.star_button.clicked.connect(self.toggle_favorite)
        self.favorites_menu.aboutToShow.connect(self.rebuild_menu)
        self._update_star()

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
        self.favorites_button.setEnabled(not value)
        self.star_button.setEnabled(not value and bool(self._favorite_target()))

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
            self.favorites_button.setAccessibleName(f"打开{value}收藏夹")
            self.star_button.setAccessibleName(f"收藏当前{value}")

    def _text_changed(self, value: str) -> None:
        self._update_star()
        self.textChanged.emit(value)

    def _editing_finished(self) -> None:
        value = self.text().strip()
        if value:
            self.path_settings.remember_selection(self.role, value, self.mode)
        self.editingFinished.emit()

    def _favorite_target(self) -> str:
        return self.path_settings.directory_for_path(self.text().strip(), self.mode)

    def _update_star(self) -> None:
        target = self._favorite_target()
        favorite = self.path_settings.favorite(target) if target else None
        checked = favorite is not None
        self.star_button.setChecked(checked)
        self.star_button.setText("★" if checked else "☆")
        self.star_button.setEnabled(bool(target) and not self.line_edit.isReadOnly())
        self.star_button.setAccessibleName(
            ("取消收藏当前" if checked else "收藏当前") + self.accessibleName()
        )
        self.star_button.setToolTip("取消收藏当前目录" if checked else "收藏当前目录")

    def toggle_favorite(self, _checked: bool | None = None) -> None:
        target = self._favorite_target()
        if not target:
            self._update_star()
            return
        if self.path_settings.favorite(target) is None:
            self.path_settings.add_favorite(target)
            is_favorite = True
        else:
            self.path_settings.remove_favorite(target)
            is_favorite = False
        self.path_settings.remember_directory(self.role, target)
        self._update_star()
        self.favoriteChanged.emit(target, is_favorite)

    def browse(self) -> None:
        start = self.path_settings.dialog_start(self.role, self.text().strip(), self.mode)
        if self.mode == "directory":
            value = QFileDialog.getExistingDirectory(self, self.dialog_title, start)
        elif self.mode == "open_file":
            value = QFileDialog.getOpenFileName(
                self, self.dialog_title, start, self.file_filter
            )[0]
        else:
            value = QFileDialog.getSaveFileName(
                self, self.dialog_title, start, self.file_filter
            )[0]
        if not value:
            return
        self.setText(value)
        self.path_settings.remember_selection(self.role, value, self.mode)

    def _use_directory(self, directory: str) -> None:
        normalized = self.path_settings.normalize_path(directory)
        if not normalized:
            return
        if self.mode == "directory":
            value = normalized
        else:
            current = self.path_settings.normalize_path(self.text().strip())
            name = Path(current).name if current else ""
            value = str(Path(normalized) / name) if name else normalized
        self.setText(value)
        self.path_settings.remember_directory(self.role, normalized)
        self.line_edit.setFocus(Qt.FocusReason.ShortcutFocusReason)

    @staticmethod
    def _menu_label(name: str, path: str, *, available: bool) -> str:
        marker = "" if available else "（不可访问）"
        return f"{name}{marker} — {path}"

    def _add_directory_action(
        self, menu: QMenu, name: str, path: str, *, recent: bool = False
    ) -> QAction:
        available = self.path_settings.directory_available(path)
        label = self._menu_label(name, path, available=available)
        action = menu.addAction(label)
        action.setToolTip(
            ("最近使用目录：" if recent else "收藏目录：")
            + path
            + ("；当前不可访问，收藏已保留。" if not available else "")
        )
        action.triggered.connect(
            lambda _checked=False, selected=path: self._use_directory(selected)
        )
        return action

    def rebuild_menu(self) -> None:
        self.favorites_menu.clear()
        favorites = self.path_settings.favorites()
        if favorites:
            self.favorites_menu.addSection("收藏目录")
            for favorite in favorites:
                self._add_directory_action(
                    self.favorites_menu, favorite.name, favorite.path
                )
        else:
            empty = self.favorites_menu.addAction("尚无收藏目录")
            empty.setEnabled(False)

        favorite_ids = {
            self.path_settings._identity(item.path) for item in favorites
        }
        recent = [
            path
            for path in self.path_settings.recent_directories()
            if self.path_settings._identity(path) not in favorite_ids
        ]
        if recent:
            self.favorites_menu.addSeparator()
            self.favorites_menu.addSection("最近使用")
            for path in recent:
                self._add_directory_action(
                    self.favorites_menu,
                    self.path_settings._default_name(path),
                    path,
                    recent=True,
                )

        target = self._favorite_target()
        current = self.path_settings.favorite(target) if target else None
        self.favorites_menu.addSeparator()
        toggle_text = "取消收藏当前目录" if current is not None else "收藏当前目录"
        toggle = self.favorites_menu.addAction(toggle_text)
        toggle.setEnabled(bool(target))
        toggle.triggered.connect(self.toggle_favorite)

        if current is not None:
            rename = self.favorites_menu.addAction("重命名当前收藏…")
            rename.triggered.connect(lambda: self._rename_current(current))
            move_up = self.favorites_menu.addAction("上移当前收藏")
            move_up.setEnabled(current.order > 0)
            move_up.triggered.connect(
                lambda: self._move_current(current.path, -1)
            )
            move_down = self.favorites_menu.addAction("下移当前收藏")
            move_down.setEnabled(current.order < len(favorites) - 1)
            move_down.triggered.connect(
                lambda: self._move_current(current.path, 1)
            )

        if self.path_settings.recent_directories():
            clear_recent = self.favorites_menu.addAction("清除最近目录")
            clear_recent.triggered.connect(self.path_settings.clear_recent_directories)

    def _rename_current(self, favorite: FavoritePath) -> None:
        name, accepted = QInputDialog.getText(
            self,
            "重命名收藏",
            "显示名称",
            QLineEdit.EchoMode.Normal,
            favorite.name,
        )
        if accepted and name.strip():
            self.path_settings.rename_favorite(favorite.path, name)
            self.rebuild_menu()

    def _move_current(self, path: str, offset: int) -> None:
        self.path_settings.move_favorite(path, offset)
        self.rebuild_menu()
