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

from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.gui.main_window import MainWindow  # noqa: E402
from fast_nc_zarr.gui.path_picker import (  # noqa: E402
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

    def test_keyboard_star_toggle_and_cross_instance_persistence(self) -> None:
        favorite = self.root / "favorite"
        favorite.mkdir()
        first_store = PathPickerSettings(self.settings())
        first = self.picker(first_store)
        first.setText(str(favorite))
        first.show()
        self.application.processEvents()

        self.assertEqual(first.text(), str(favorite))
        self.assertTrue(first.line_edit.accessibleName())
        self.assertTrue(first.browse_button.accessibleName())
        self.assertTrue(first.favorites_button.accessibleName())
        self.assertTrue(first.star_button.accessibleName())
        first.star_button.setFocus(Qt.FocusReason.TabFocusReason)
        QTest.keyClick(first.star_button, Qt.Key.Key_Space)
        self.application.processEvents()

        self.assertTrue(first.star_button.isChecked())
        persisted = first_store.favorite(favorite)
        self.assertIsNotNone(persisted)
        json.dumps(persisted.to_dict())
        first.close()

        second_store = PathPickerSettings(self.settings())
        second = self.picker(second_store)
        second.setText(str(favorite))
        self.assertTrue(second.star_button.isChecked())
        self.assertEqual(second_store.last_directory("inspection_input"), str(favorite))

        second.show()
        second.star_button.setFocus(Qt.FocusReason.TabFocusReason)
        QTest.keyClick(second.star_button, Qt.Key.Key_Space)
        self.application.processEvents()
        self.assertFalse(second.star_button.isChecked())
        self.assertIsNone(PathPickerSettings(self.settings()).favorite(favorite))
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
        picker = self.picker(restored)
        picker.setText(str(second_path))
        picker.rebuild_menu()
        unavailable_actions = [
            action.text()
            for action in picker.favorites_menu.actions()
            if "Fast disk" in action.text()
        ]
        self.assertEqual(len(unavailable_actions), 1)
        self.assertIn("不可访问", unavailable_actions[0])
        self.assertIsNotNone(restored.favorite(second_path))
        picker.close()

    def test_browse_starts_from_available_favorite_and_remembers_role(self) -> None:
        favorite = self.root / "favorite"
        chosen = self.root / "chosen"
        favorite.mkdir()
        chosen.mkdir()
        store = PathPickerSettings(self.settings())
        store.add_favorite(favorite, name="Favorite")
        picker = self.picker(store, role="pipeline_temporary")

        with patch(
            "fast_nc_zarr.gui.path_picker.QFileDialog.getExistingDirectory",
            return_value=str(chosen),
        ) as dialog:
            picker.browse()

        self.assertEqual(dialog.call_args.args[2], str(favorite))
        self.assertEqual(picker.text(), str(chosen))
        self.assertEqual(store.last_directory("pipeline_temporary"), str(chosen))
        self.assertEqual(store.recent_directories()[0], str(chosen))
        picker.close()

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
