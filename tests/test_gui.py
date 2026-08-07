from __future__ import annotations

import os
import math
from pathlib import Path
import shutil
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import xarray as xr
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.application.services import (  # noqa: E402
    ConversionConfig,
    RechunkConfig,
    SourceInspectionConfig,
    inspect_source,
    inspect_zarr,
    load_inspection_snapshot,
    preview_conversion,
    preview_rechunk,
    save_inspection_snapshot,
    run_conversion,
)
from fast_nc_zarr.models import VariableTransform
from fast_nc_zarr.gui.main_window import MainWindow  # noqa: E402
from fast_nc_zarr.gui.workers import resolve_storage_targets  # noqa: E402
from fast_nc_zarr.time_mapping import inspect_time_metadata  # noqa: E402


ROOT = Path("/tmp/codex_test/fast_nc_zarr_gui_tests")


class GuiServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)
        (ROOT / "source").mkdir(parents=True)
        lat = np.asarray([30, 20, 10], dtype="float32")
        lon = np.asarray([100, 110, 120, 130], dtype="float32")
        for index in range(2):
            dataset = xr.Dataset(
                {
                    "value": (
                        ("time", "lat", "lon"),
                        np.full((1, 3, 4), index, dtype="float32"),
                    )
                },
                coords={
                    "time": np.asarray([np.datetime64("2001-01-01") + np.timedelta64(index, "D")]),
                    "lat": lat,
                    "lon": lon,
                },
            )
            dataset.to_netcdf(ROOT / "source" / f"day-{index}.nc", engine="h5netcdf")
            dataset.close()
        dataset = xr.Dataset(
            {
                "value": (
                    ("time", "lat", "lon"),
                    np.arange(24, dtype="float32").reshape(2, 3, 4),
                )
            },
            coords={"time": np.arange(2), "lat": lat, "lon": lon},
        )
        dataset.to_zarr(ROOT / "input.zarr", zarr_format=3, mode="w", consolidated=False)
        dataset.close()

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(ROOT, ignore_errors=True)

    def test_source_service_and_conversion_preview(self) -> None:
        result = inspect_source(
            SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1)
        )
        preview = preview_conversion(
            result,
            ConversionConfig(ROOT / "output.zarr", time_start="2001-01-01", time_end="2001-01-02"),
        )
        self.assertEqual(result.mode, "complete")
        self.assertEqual(preview.selection.shape, (2, 3, 4))

    def test_source_snapshot_round_trip_restores_conversion_inventory(self) -> None:
        result = inspect_source(
            SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1)
        )
        snapshot = ROOT / "source-snapshot.json"
        save_inspection_snapshot(result, snapshot)
        restored = load_inspection_snapshot(snapshot)
        preview = preview_conversion(
            restored,
            ConversionConfig(
                ROOT / "restored-output.zarr",
                time_start="2001-01-01",
                time_end="2001-01-02",
            ),
        )
        self.assertEqual(restored.source_inventory.source_mode, "dimension")
        self.assertEqual(preview.selection.shape, (2, 3, 4))

    def test_source_snapshot_rejects_modified_file(self) -> None:
        result = inspect_source(
            SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1)
        )
        snapshot = ROOT / "stale-source-snapshot.json"
        save_inspection_snapshot(result, snapshot)
        source = ROOT / "source" / "day-0.nc"
        original = source.stat().st_mtime_ns
        os.utime(source, ns=(original + 1_000_000, original + 1_000_000))
        try:
            with self.assertRaisesRegex(ValueError, "快照已过期"):
                load_inspection_snapshot(snapshot)
        finally:
            os.utime(source, ns=(original, original))

    def test_conversion_supports_variable_rename_and_transform(self) -> None:
        result = inspect_source(
            SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1)
        )
        output = ROOT / "renamed-output.zarr"
        run_conversion(
            result,
            ConversionConfig(
                output,
                time_start="2001-01-01",
                time_end="2001-01-02",
                variable_names={"value": "renamed_value"},
                variable_transforms={
                    "value": VariableTransform(
                        fill_values=(0,), scale_factor=2, output_fill=-7
                    )
                },
                max_workers=1,
                validate=True,
            ),
        )
        with xr.open_zarr(output, consolidated=False, chunks=None, mask_and_scale=False) as dataset:
            self.assertIn("renamed_value", dataset.data_vars)
            self.assertNotIn("value", dataset.data_vars)
            expected = np.stack(
                [
                    np.full((3, 4), -7, dtype="float32"),
                    np.full((3, 4), 2, dtype="float32"),
                ]
            )
            np.testing.assert_equal(dataset.renamed_value.values, expected)

    def test_zarr_service_and_rechunk_preview(self) -> None:
        result = inspect_zarr(ROOT / "input.zarr")
        preview = preview_rechunk(
            RechunkConfig(ROOT / "input.zarr", ROOT / "output.zarr", workers=1),
            result.zarr_info,
        )
        self.assertEqual(result.zarr_info.zarr_format, 3)
        self.assertEqual(preview.plan.strategy, "time")

    def test_main_window_has_mvp_pages(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        self.assertEqual(window.navigation.count(), 6)
        self.assertEqual(window.stack.count(), 6)
        self.assertEqual(window.windowTitle(), "快速 Zarr 转换器")
        self.assertEqual(
            [
                window.navigation.item(index).text()
                for index in range(window.navigation.count())
                if not window.navigation.item(index).isHidden()
            ],
            ["数据检查", "处理流程", "任务中心"],
        )
        self.assertFalse(bool(window.navigation.item(1).flags() & Qt.ItemFlag.ItemIsEnabled))
        self.assertTrue(bool(window.navigation.item(2).flags() & Qt.ItemFlag.ItemIsEnabled))
        self.assertEqual(window.resample_page.method.count(), 6)
        window.task_page.update_resource(
            {
                "cpu": 12,
                "rss_gib": 0.5,
                "read_mib_s": 1,
                "write_mib_s": 2,
                "disks": [
                    {
                        "roles": "输入/输出",
                        "device": "/dev/test",
                        "mountpoint": "/data",
                        "used_gib": 10,
                        "free_gib": 20,
                        "percent": 33.3,
                        "read_mib_s": 3,
                        "write_mib_s": 4,
                    }
                ],
            }
        )
        self.assertEqual(window.task_page.disk_table.rowCount(), 1)
        self.assertEqual(window.task_page.disk_table.columnCount(), 7)
        window.close()
        app.processEvents()

    def test_storage_targets_include_only_task_paths_and_merge_roles(self) -> None:
        class Partition:
            device = "/dev/root-test"
            mountpoint = "/"
            fstype = "ext4"

        class FakePsutil:
            class Error(Exception):
                pass

            @staticmethod
            def disk_partitions(all=True):
                return [Partition()]

        targets = resolve_storage_targets(
            FakePsutil,
            (
                ("输入", str(ROOT / "input.zarr")),
                ("输出", str(ROOT / "not-created" / "output.zarr")),
            ),
        )
        self.assertEqual(len(targets), 1)
        self.assertEqual(targets[0]["roles"], "输入/输出")
        self.assertEqual(targets[0]["device"], "/dev/root-test")

    def test_zarr_operation_page_can_combine_rechunk_and_recompression(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        result = inspect_zarr(ROOT / "input.zarr")
        page = window.rechunk_page
        page._inspection_done(result)

        config = page._config()
        self.assertTrue(config.rechunk)
        self.assertFalse(config.recompress)
        self.assertEqual(config.compression, "none")

        page.recompress_checkbox.setChecked(True)
        page.temporary_dir.setText(str(ROOT / "fast-temporary"))
        config = page._config()
        self.assertTrue(config.rechunk)
        self.assertTrue(config.recompress)
        self.assertEqual(config.compression, "balanced")
        self.assertEqual(config.temporary_dir, ROOT / "fast-temporary")
        preview = preview_rechunk(config, result.zarr_info)
        self.assertTrue(preview.compression.enabled)

        page.rechunk_checkbox.setChecked(False)
        config = page._config()
        self.assertFalse(config.rechunk)
        self.assertTrue(config.recompress)
        compression_preview = preview_rechunk(config, result.zarr_info)
        reference = next(item for item in result.zarr_info.data_variables if item.ndim == 3)
        self.assertEqual(compression_preview.plan.chunks, reference.chunks)
        window.close()
        app.processEvents()

    def test_pipeline_page_builds_composable_operation_config(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        result = inspect_source(
            SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1)
        )
        page = window.pipeline_page
        page.set_inspection(result)
        page.output.setText(str(ROOT / "pipeline-output.zarr"))

        self.assertTrue(page.conversion_checkbox.isChecked())
        self.assertFalse(page.conversion_checkbox.isEnabled())
        self.assertFalse(page.resampling_group.isEnabled())
        self.assertFalse(page.chunking_group.isEnabled())
        self.assertFalse(page.compression_group.isEnabled())

        page.resample_checkbox.setChecked(True)
        page.rechunk_checkbox.setChecked(True)
        page.recompress_checkbox.setChecked(True)
        page.before_conditions.setText("<0, >100")
        page.before_results.setText("0, 100")
        config = page._config()
        self.assertTrue(config.operations.resample)
        self.assertTrue(config.operations.rechunk)
        self.assertTrue(config.operations.recompress)
        self.assertEqual(config.resampling.resolution, 0.1)
        self.assertEqual(config.resampling.before_conditions, "<0, >100")
        self.assertEqual(config.resampling.before_results, "0, 100")
        self.assertEqual(config.compression.profile, "balanced")
        self.assertEqual(config.compression.codec, "blosc-zstd")
        self.assertEqual(config.compression.level, 4)
        self.assertTrue(page.resampling_group.isEnabled())
        self.assertTrue(page.chunking_group.isEnabled())
        self.assertTrue(page.compression_group.isEnabled())

        page.plan = object()
        page.run_button.setEnabled(True)
        page.recompress_checkbox.setChecked(False)
        self.assertIsNone(page.plan)
        self.assertFalse(page.run_button.isEnabled())
        window.close()
        app.processEvents()

    def test_pipeline_page_accepts_existing_zarr_input(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        result = inspect_zarr(ROOT / "input.zarr")
        window._workflow_zarr_ready(result)
        page = window.pipeline_page
        page.output.setText(str(ROOT / "zarr-pipeline-output.zarr"))
        self.assertFalse(page.conversion_checkbox.isChecked())
        self.assertFalse(page.conversion_group.isVisible())
        page.recompress_checkbox.setChecked(True)
        config = page._config()
        self.assertEqual(config.input.kind, "zarr")
        self.assertFalse(config.operations.resample)
        self.assertTrue(config.operations.recompress)
        self.assertEqual(window.navigation.currentRow(), 4)
        window.close()
        app.processEvents()

    def test_time_check_is_required_before_source_operation_pages(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        result = inspect_time_metadata(ROOT / "source", requested_engine="h5netcdf")
        window.inspection_page._time_inspection_done(result)
        self.assertTrue(window.inspection_page.confirm_time_button.isEnabled())
        self.assertFalse(bool(window.navigation.item(1).flags() & Qt.ItemFlag.ItemIsEnabled))
        window.close()
        app.processEvents()

    def test_conversion_editor_collects_names_and_transforms(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        result = inspect_source(
            SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1)
        )
        window.inspection_page._inspection_done(result)
        row = 0
        window.conversion_page.variables.cellWidget(row, 2).setText("renamed_value")
        window.conversion_page.variables.cellWidget(row, 3).setText("-99,nan")
        window.conversion_page.variables.cellWidget(row, 4).setText("0.5")
        config = window.conversion_page._config()
        self.assertEqual(config.variable_names["value"], "renamed_value")
        values = config.variable_transforms["value"].fill_values
        self.assertEqual(values[0], -99.0)
        self.assertTrue(math.isnan(values[1]))
        self.assertEqual(config.variable_transforms["value"].scale_factor, 0.5)
        window.close()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
