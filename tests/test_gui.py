from __future__ import annotations

import json
import os
import math
import threading
from pathlib import Path
import shutil
import sys
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import xarray as xr
from PySide6.QtCore import Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import QApplication
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from fast_nc_zarr import __version__  # noqa: E402

from fast_nc_zarr.application.services import (  # noqa: E402
    ConversionConfig,
    InspectionResult,
    RechunkConfig,
    SourceInspectionConfig,
    inspect_source,
    inspect_zarr,
    load_inspection_snapshot,
    preview_conversion,
    preview_rechunk,
    run_conversion,
    run_rechunk,
    save_inspection_snapshot,
)
from fast_nc_zarr._backend import rust_capability  # noqa: E402

_RUST_ZARR_READY = rust_capability().supported and "zarr.rechunk_f32" in rust_capability().operations
from fast_nc_zarr.models import VariableSpec, VariableTransform
from fast_nc_zarr.gui import fonts  # noqa: E402
from fast_nc_zarr.gui.fonts import configure_application_font  # noqa: E402
from fast_nc_zarr.gui.main_window import MainWindow, TaskPage, _run_cancelable  # noqa: E402
from fast_nc_zarr.gui.workers import parse_progress_message, resolve_storage_targets  # noqa: E402
from fast_nc_zarr.time_mapping import inspect_time_metadata  # noqa: E402
from fast_nc_zarr.pipeline.models import (  # noqa: E402
    PipelineChunkingOptions,
    PipelineConfig,
    PipelineGeneralConfig,
    PipelineInput,
    PipelineOperations,
    PipelineResamplingOptions,
)
from fast_nc_zarr.pipeline.recovery import PipelineRecovery  # noqa: E402


ROOT = Path("/tmp/codex_test/fast_nc_zarr_gui_tests")



class FontTests(unittest.TestCase):
    def test_bundled_font_registers_and_is_applied_to_application(self) -> None:
        app = QApplication.instance() or QApplication([])
        self.assertTrue(fonts.FONT_PATH.is_file())

        family = configure_application_font(app)

        self.assertTrue(family)
        self.assertIn(family, QFontDatabase.families())
        self.assertEqual(app.font().family(), family)

    def test_missing_font_resource_reports_its_path(self) -> None:
        app = QApplication.instance() or QApplication([])
        missing = fonts.FONT_PATH.with_name("missing-font.ttf")
        with patch.object(fonts, "FONT_PATH", missing):
            with self.assertRaisesRegex(RuntimeError, str(missing)):
                configure_application_font(app)

    def test_unregistrable_font_resource_reports_its_path(self) -> None:
        app = QApplication.instance() or QApplication([])
        license_path = fonts.FONT_PATH.with_name("OFL.txt")
        with patch.object(fonts, "FONT_PATH", license_path):
            with self.assertRaisesRegex(RuntimeError, str(license_path)):
                configure_application_font(app)


class VersionTests(unittest.TestCase):
    def test_window_title_displays_release_version(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        self.assertEqual(__version__, "1.6.9")
        self.assertIn("v1.6.9", window.windowTitle())
        window.close()
        app.processEvents()


class TaskFeedbackTests(unittest.TestCase):
    def test_progress_messages_drive_determinate_progress_bar(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        window.task_page.started("测试任务", lambda: None)

        parsed = parse_progress_message("空间块进度：3/4 | 本块 0.1s")
        self.assertEqual(parsed[:2], (3, 4))
        window.task_page.update_progress(*parsed)

        self.assertEqual(window.task_page.progress.minimum(), 0)
        self.assertEqual(window.task_page.progress.maximum(), 1000)
        self.assertIsNone(parse_progress_message("目标磁盘使用率达到 95%"))
        self.assertEqual(window.task_page.progress.value(), 750)
        self.assertIn("空间块进度", window.task_page.progress.format())
        window.close()
        app.processEvents()

    def test_short_preview_cancellation_gates_callback_result(self) -> None:
        cancelled = threading.Event()
        cancelled.set()

        with self.assertRaisesRegex(RuntimeError, "任务已取消"):
            _run_cancelable(cancelled, lambda: "stale result")

    def test_task_log_and_resource_events_are_persisted(self) -> None:
        root = ROOT / "task-logs"
        shutil.rmtree(root, ignore_errors=True)
        app = QApplication.instance() or QApplication([])
        page = TaskPage(log_root=root)
        page.started("持久日志测试", lambda: None)
        page.append("worker message")
        page.update_progress(1, 2, "进度：1/2")
        page.update_resource({"elapsed": 1.0, "cpu": 25.0, "rss_gib": 0.5})
        page.completed()

        self.assertIsNotNone(page.active_log_path)
        self.assertIsNotNone(page.active_events_path)
        self.assertIn("worker message", page.active_log_path.read_text(encoding="utf-8"))
        events = [
            json.loads(line)
            for line in page.active_events_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [event["event"] for event in events],
            ["started", "progress", "resource", "finished"],
        )
        page.close()
        app.processEvents()


    def test_event_write_failure_does_not_break_task_completion(self) -> None:
        class FailingHandle:
            def __init__(self) -> None:
                self.closed = False

            def write(self, _value: str) -> None:
                raise OSError("disk full")

            def flush(self) -> None:
                pass

            def close(self) -> None:
                self.closed = True

        root = ROOT / "failing-task-logs"
        shutil.rmtree(root, ignore_errors=True)
        app = QApplication.instance() or QApplication([])
        page = TaskPage(log_root=root)
        page.started("日志容错测试", lambda: None)
        page._close_task_logs()
        failing_log = FailingHandle()
        failing_events = FailingHandle()
        page._log_handle = failing_log
        page._events_handle = failing_events

        page._finish_history("完成")

        self.assertEqual(page.history[-1]["status"], "完成")
        self.assertEqual(page.log_persistence_error, "disk full")
        self.assertTrue(failing_log.closed)
        self.assertTrue(failing_events.closed)
        self.assertIsNone(page._log_handle)
        self.assertIsNone(page._events_handle)
        page.close()
        app.processEvents()


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

    @unittest.skipUnless(_RUST_ZARR_READY, "Rust Zarr native extension is not built")
    def test_zarr_service_rust_backend_publishes_valid_output(self) -> None:
        output = ROOT / "rust-service-output.zarr"
        result = inspect_zarr(ROOT / "input.zarr")
        metrics = run_rechunk(
            RechunkConfig(
                ROOT / "input.zarr",
                output,
                strategy="custom",
                custom_chunks=(1, 3, 4),
                workers=1,
                backend="rust",
                validate=True,
            ),
            result.zarr_info,
        )
        self.assertEqual(metrics["execution_path"], "rust-streaming-target-chunk")
        self.assertEqual(Path(metrics["output"]), output)
        self.assertEqual(metrics["target_chunks"], [1, 3, 4])
        with xr.open_zarr(
            output, consolidated=False, chunks=None, decode_times=False, mask_and_scale=False
        ) as dataset:
            np.testing.assert_array_equal(
                dataset["value"].values,
                np.arange(24, dtype="float32").reshape(2, 3, 4),
            )

    def test_main_window_has_mvp_pages(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        self.assertEqual(window.navigation.count(), 6)
        self.assertEqual(window.stack.count(), 6)
        self.assertEqual(window.windowTitle(), "快速 Zarr 转换器 v1.6.9")
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
        self.assertEqual(window.context_bar.objectName(), "topContext")
        self.assertGreaterEqual(window.navigation.minimumWidth(), 210)
        self.assertEqual(window.task_page.metric_cards["cpu"].value.text(), "—")
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
        self.assertIn("0.12", window.task_page.metric_cards["cpu"].value.text())
        self.assertIn("0.50", window.task_page.metric_cards["memory"].value.text())
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
        self.assertEqual(config.compression.profile, "auto")
        self.assertIsNone(config.compression.codec)
        self.assertIsNone(config.compression.level)
        self.assertEqual(config.compression.objective, "balanced")
        self.assertEqual(config.chunking.workers, "auto")
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

    def test_pipeline_disables_auxiliary_metadata_variables(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        result = inspect_source(
            SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1)
        )
        inventory = result.source_inventory
        inventory.variables["time_bnds"] = VariableSpec(
            "time_bnds", ("time", "nv"), "datetime64[ns]", (2,), None
        )
        inventory.variables["crs"] = VariableSpec("crs", (), "|S1", (), None)
        page = window.pipeline_page
        page.set_inspection(result)
        rows = {
            page.variables.item(row, 1).text(): page.variables.cellWidget(row, 0)
            for row in range(page.variables.rowCount())
        }
        self.assertTrue(rows["value"].isChecked())
        self.assertTrue(rows["value"].isEnabled())
        for name in ("time_bnds", "crs"):
            self.assertFalse(rows[name].isChecked())
            self.assertFalse(rows[name].isEnabled())
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

    def test_temporary_pipeline_restores_plan_and_continue_action(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        zarr = inspect_zarr(ROOT / "input.zarr")
        config = PipelineConfig(
            input=PipelineInput(kind="zarr"),
            general=PipelineGeneralConfig(
                output=ROOT / "resumed-output.zarr",
                temporary_dir=ROOT / "resume-temp",
                lat_min=10,
                lat_max=40,
                lon_min=100,
                lon_max=140,
                cleanup_intermediate=True,
            ),
            operations=PipelineOperations(True, True, True),
            resampling=PipelineResamplingOptions(
                resolution=0.5,
                method="conservative",
                time_block=3,
                compute_workers=1,
                space_workers=2,
            ),
            chunking=PipelineChunkingOptions(strategy="space", target_mib=64, workers=2),
        )
        recovery = PipelineRecovery(
            ROOT / "resume-job",
            ROOT / "resume-job" / "manifest.json",
            ROOT / "input.zarr",
            "conversion",
            zarr.zarr_info,
            config,
            {},
            "临时处理产物检查通过",
        )
        result = InspectionResult(
            kind="temporary",
            path=ROOT / "input.zarr",
            report=recovery.report,
            dataset_info=zarr.zarr_info,
            recovery=recovery,
        )
        window._workflow_zarr_ready(result)
        page = window.pipeline_page
        self.assertEqual(
            window.inspection_page.input_kind.findData("temporary"), 2
        )
        self.assertEqual(page.run_button.text(), "继续执行")
        self.assertTrue(page.run_button.isEnabled())
        restored = page._config()
        self.assertEqual(restored.input.kind, "zarr")
        self.assertEqual(restored.general.output, ROOT / "resumed-output.zarr")
        self.assertEqual(restored.resampling.method, "conservative")
        self.assertEqual(restored.resampling.time_block, 3)
        self.assertEqual(restored.resampling.space_workers, 2)
        self.assertTrue(restored.operations.resample)
        self.assertTrue(restored.operations.rechunk)
        self.assertTrue(restored.operations.recompress)
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

    def test_changed_inspection_parameters_revoke_downstream_result(self) -> None:
        app = QApplication.instance() or QApplication([])
        window = MainWindow()
        result = inspect_source(
            SourceInspectionConfig(ROOT / "source", engine="h5netcdf", workers=1)
        )
        window.inspection_page._inspection_done(result)
        self.assertTrue(bool(window.navigation.item(4).flags() & Qt.ItemFlag.ItemIsEnabled))

        window.inspection_page.inspect_workers.setValue(2)

        self.assertIsNone(window.inspection_page.result)
        self.assertIsNone(window.pipeline_page.inspection)
        self.assertFalse(bool(window.navigation.item(4).flags() & Qt.ItemFlag.ItemIsEnabled))
        self.assertEqual(window.navigation.currentRow(), 0)
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
