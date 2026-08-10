from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.gui.main_window import MainWindow  # noqa: E402
from fast_nc_zarr.gui.path_chooser import PathChooserDialog  # noqa: E402
from fast_nc_zarr.gui.path_picker import (  # noqa: E402
    FavoriteManagerDialog,
    PathPicker,
    PathPickerSettings,
)


class PathPickerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.application = QApplication.instance() or QApplication([])
        self.temporary = tempfile.TemporaryDirectory(
            prefix="fast-nc-zarr-path-picker-"
        )
        self.root = Path(self.temporary.name)
        self.settings_path = self.root / "settings.ini"

    def tearDown(self) -> None:
        self.application.processEvents()
        self.temporary.cleanup()

    def settings(self) -> QSettings:
        return QSettings(
            str(self.settings_path), QSettings.Format.IniFormat
        )

    def picker(
        self,
        store: PathPickerSettings,
        *,
        role: str = "inspection_input",
        mode: str = "directory",
    ) -> PathPicker:
        return PathPicker(
            role=role,
            dialog_title="选择测试目录",
            mode=mode,
            accessible_name="测试目录路径",
            settings=store,
        )

    def test_embedded_chooser_favorite_persists_without_path_buttons(self) -> None:
        favorite = self.root / "favorite"
        favorite.mkdir()
        first_store = PathPickerSettings(self.settings())
        first = PathChooserDialog(
            first_store,
            role="inspection_input",
            dialog_title="选择测试目录",
            start=str(self.root),
            mode="directory",
        )
        first.file_dialog.setDirectory(str(favorite))
        first._toggle_current_favorite()
        self.assertIsNotNone(first_store.favorite(favorite))
        first.close()

        second_store = PathPickerSettings(self.settings())
        second = PathChooserDialog(
            second_store,
            role="inspection_input",
            dialog_title="选择测试目录",
            start=str(favorite),
            mode="directory",
        )
        self.assertTrue(any("favorite" in second.location_list.item(index).text() for index in range(second.location_list.count())))
        second.file_dialog.setDirectory(str(favorite))
        second._toggle_current_favorite()
        self.assertIsNone(second_store.favorite(favorite))
        second.close()

    def test_display_name_order_and_unavailable_favorite_are_retained(self) -> None:
        first_path = self.root / "first"
        second_path = self.root / "second"
        first_path.mkdir()
        second_path.mkdir()
        store = PathPickerSettings(self.settings())
        store.add_favorite(first_path, name="First")
        store.add_favorite(second_path, name="Second")
        self.assertTrue(store.move_favorite(second_path, -1))
        self.assertTrue(store.rename_favorite(second_path, "Fast disk"))
        shutil.rmtree(second_path)

        restored = PathPickerSettings(self.settings())
        favorites = restored.favorites()
        self.assertEqual([item.name for item in favorites], ["Fast disk", "First"])
        self.assertEqual(favorites[0].path, str(second_path))
        dialog = PathChooserDialog(
            restored,
            role="inspection_input",
            dialog_title="选择测试目录",
            start=str(self.root),
            mode="directory",
        )
        texts = [dialog.location_list.item(index).text() for index in range(dialog.location_list.count())]
        self.assertTrue(any("Fast disk" in text and "不可访问" in text for text in texts))
        self.assertIsNotNone(restored.favorite(second_path))
        dialog.close()

    def test_browse_starts_from_available_favorite_and_remembers_role(self) -> None:
        favorite = self.root / "favorite"
        chosen = self.root / "chosen"
        favorite.mkdir()
        chosen.mkdir()
        store = PathPickerSettings(self.settings())
        store.add_favorite(favorite, name="Favorite")
        picker = self.picker(store, role="pipeline_temporary")

        with patch.object(picker, "_select_path", return_value=str(chosen)) as chooser:
            picker.browse()

        self.assertEqual(chooser.call_args.args[0], str(favorite))

        self.assertEqual(picker.text(), str(chosen))
        self.assertEqual(store.last_directory("pipeline_temporary"), str(chosen))
        self.assertEqual(store.recent_directories()[0], str(chosen))
        picker.close()
    def test_v1_settings_migrate_without_deleting_legacy_values(self) -> None:
        favorite = self.root / "legacy-favorite"
        favorite.mkdir()
        settings = self.settings()
        legacy_favorites = json.dumps(
            [{"path": str(favorite), "name": "Legacy", "order": 0}],
            ensure_ascii=False,
        )
        settings.setValue("pathPicker/v1/favorites", legacy_favorites)
        settings.setValue("pathPicker/v1/recentDirectories", json.dumps([str(favorite)]))
        settings.setValue("pathPicker/v1/lastDirectory/inspection_input", str(favorite))
        settings.sync()

        store = PathPickerSettings(settings)

        self.assertEqual(store.ROOT_KEY, "pathPicker/v2")
        self.assertEqual(store.favorites()[0].name, "Legacy")
        self.assertEqual(store.last_directory("inspection_input"), str(favorite))
        self.assertTrue(settings.value("pathPicker/v1/favorites"))
        self.assertTrue(settings.value("pathPicker/v2/favorites"))

    def test_favorite_manager_filters_and_returns_selected_path(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        first.mkdir()
        second.mkdir()
        store = PathPickerSettings(self.settings())
        store.add_favorite(first, name="Fast SSD")
        store.add_favorite(second, name="Archive")
        dialog = FavoriteManagerDialog(store)

        self.assertEqual(dialog.list.count(), 2)
        dialog.search.setText("SSD")
        self.assertEqual(dialog.list.count(), 1)
        dialog._use_selected()
        self.assertEqual(dialog.selected_path, str(first))
        dialog.close()

    def test_path_field_has_only_status_and_browse_controls(self) -> None:
        directory = self.root / "available"
        directory.mkdir()
        picker = self.picker(PathPickerSettings(self.settings()))
        picker.setText(str(directory))
        self.assertEqual(picker.path_status.text(), "可访问")
        self.assertTrue(picker.browse_button.accessibleName())
        self.assertFalse(hasattr(picker, "favorites_button"))
        self.assertFalse(hasattr(picker, "manage_button"))
        self.assertFalse(hasattr(picker, "star_button"))
        picker.close()


    def test_embedded_chooser_shows_favorites_and_recent_directories(self) -> None:
        favorite = self.root / "favorite-in-dialog"
        recent = self.root / "recent-in-dialog"
        favorite.mkdir()
        recent.mkdir()
        store = PathPickerSettings(self.settings())
        store.add_favorite(favorite, name="常用目录")
        store.remember_directory("inspection_input", recent)

        dialog = PathChooserDialog(
            store,
            role="inspection_input",
            dialog_title="选择测试目录",
            start=str(self.root),
            mode="directory",
        )
        texts = [dialog.location_list.item(index).text() for index in range(dialog.location_list.count())]
        self.assertIn("收藏目录", texts)
        self.assertTrue(any("常用目录" in text for text in texts))
        self.assertIn("最近浏览", texts)
        self.assertTrue(any("recent-in-dialog" in text for text in texts))
        dialog._location_activated(dialog.location_list.item(1))
        self.assertEqual(dialog.selected_path, str(favorite))
        dialog.file_dialog.setDirectory(str(recent))
        dialog._toggle_current_favorite()
        self.assertIsNotNone(store.favorite(recent))
        dialog.close()

    def test_main_window_path_fields_keep_line_edit_compatible_contract(self) -> None:
        store = PathPickerSettings(self.settings())
        window = MainWindow(path_settings=store)
        fields = (
            window.inspection_page.path,
            window.conversion_page.output,
            window.rechunk_page.input,
            window.rechunk_page.temporary_dir,
            window.rechunk_page.output,
            window.resample_page.input,
            window.resample_page.temporary_dir,
            window.resample_page.output,
            window.pipeline_page.temporary_dir,
            window.pipeline_page.output,
        )
        self.assertTrue(all(isinstance(field, PathPicker) for field in fields))
        for index, field in enumerate(fields):
            value = str(self.root / f"path-{index}")
            field.setText(value)
            self.assertEqual(field.text(), value)

        page = window.rechunk_page
        page.info = object()
        page.input.setText(str(self.root / "input.zarr"))
        page.output.setText(str(self.root / "output.zarr"))
        config = page._config()
        self.assertEqual(config.input, self.root / "input.zarr")
        self.assertEqual(config.output, self.root / "output.zarr")
        storage_paths = window._task_storage_paths(page)
        self.assertIn(("输入", str(self.root / "input.zarr")), storage_paths)
        self.assertIn(("输出", str(self.root / "output.zarr")), storage_paths)
        window.close()

    def test_settings_write_failure_is_non_fatal(self) -> None:
        class FailingSettings:
            def sync(self) -> None:
                raise OSError("settings are read-only")

            def value(self, _key, default=None):
                return default

            def setValue(self, _key, _value) -> None:
                raise OSError("settings are read-only")

        store = PathPickerSettings(FailingSettings())
        store.add_favorite(self.root / "unwritable")
        store.remember_directory("inspection_input", self.root)
        self.assertIn("read-only", store.last_error)


if __name__ == "__main__":
    unittest.main()
