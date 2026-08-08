from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import inspect
import os
import re
import time
import threading
import traceback
from pathlib import Path
from typing import Callable, Any

from PySide6.QtCore import QThread, Signal


def _nearest_existing(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _disk_counter_keys(device: str, device_id: int) -> tuple[str, ...]:
    """Return psutil counter names for plain, partitioned and mapped devices."""

    candidates: list[str] = []
    try:
        sys_device = Path(
            f"/sys/dev/block/{os.major(device_id)}:{os.minor(device_id)}"
        ).resolve(strict=True)
        candidates.append(sys_device.name)
        if sys_device.parent.name != "block":
            candidates.append(sys_device.parent.name)
    except (OSError, ValueError):
        pass
    if device:
        candidates.append(Path(device).name)
    return tuple(dict.fromkeys(item for item in candidates if item))


def resolve_storage_targets(
    psutil,
    paths: tuple[tuple[str, str], ...],
) -> list[dict[str, object]]:
    """Resolve only filesystems used by the current task."""

    try:
        partitions = sorted(
            psutil.disk_partitions(all=True),
            key=lambda item: len(str(item.mountpoint or "")),
            reverse=True,
        )
    except psutil.Error:
        partitions = []
    resolved: dict[int, dict[str, object]] = {}
    for role, raw_path in paths:
        if not raw_path:
            continue
        existing = _nearest_existing(Path(raw_path))
        try:
            stat = existing.stat()
        except OSError:
            continue
        match = next(
            (
                item
                for item in partitions
                if existing == Path(item.mountpoint)
                or Path(item.mountpoint) in existing.parents
            ),
            None,
        )
        if match is None:
            continue
        device = str(match.device or "filesystem")
        mountpoint = str(match.mountpoint)
        device_id = int(stat.st_dev)
        entry = resolved.setdefault(
            device_id,
            {
                "device": device,
                "mountpoint": mountpoint,
                "roles": set(),
                "counter_keys": _disk_counter_keys(device, device_id),
            },
        )
        entry["roles"].add(role)
    result = []
    for entry in resolved.values():
        copied = dict(entry)
        copied["roles"] = "/".join(sorted(entry["roles"]))
        result.append(copied)
    return sorted(result, key=lambda item: (str(item["roles"]), str(item["mountpoint"])))


class _SignalStream(io.TextIOBase):
    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self._callback = callback
        self._buffer = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer += text
        while "\n" in self._buffer or "\r" in self._buffer:
            positions = [index for index in (self._buffer.find("\n"), self._buffer.find("\r")) if index >= 0]
            if not positions:
                break
            index = min(positions)
            message = self._buffer[:index].strip()
            self._buffer = self._buffer[index + 1 :]
            if message:
                self._callback(message)
        return len(text)

    def flush(self) -> None:
        if self._buffer.strip():
            self._callback(self._buffer.strip())
        self._buffer = ""


_PERCENT_PROGRESS = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)\s*%")
_FRACTION_PROGRESS = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])")


def parse_progress_message(message: str) -> tuple[int, int, str] | None:
    """Extract the latest concrete percentage or completed/total pair from log output."""
    percentages = _PERCENT_PROGRESS.findall(message)
    if percentages and (
        "进度" in message or "Completed" in message or "[" in message
    ):
        value = min(1000, max(0, int(round(float(percentages[-1]) * 10))))
        return value, 1000, message
    fractions = _FRACTION_PROGRESS.findall(message)
    if fractions:
        completed, total = map(int, fractions[-1])
        if total > 0 and 0 <= completed <= total:
            return completed, total, message
    return None


class TaskWorker(QThread):
    """Run a synchronous application service without blocking Qt's UI thread."""

    log = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    finished_cleanly = Signal()
    resource = Signal(object)
    progress = Signal(int, int, str)
    cancelled = Signal()

    def __init__(
        self,
        function: Callable[[], Any],
        parent=None,
        *,
        storage_paths: tuple[tuple[str, str], ...] = (),
    ) -> None:
        super().__init__(parent)
        self.function = function
        self.cancel_event = threading.Event()
        self.storage_paths = storage_paths

    def request_cancel(self) -> None:
        self.cancel_event.set()
        self.log.emit("已请求取消；当前正在执行的块完成后停止。")

    def _handle_output(self, message: str) -> None:
        self.log.emit(message)
        parsed = parse_progress_message(message)
        if parsed is not None:
            self.progress.emit(*parsed)

    def _call(self) -> Any:
        try:
            parameters = inspect.signature(self.function).parameters
        except (TypeError, ValueError):
            parameters = {}
        if parameters:
            return self.function(self.cancel_event)
        return self.function()

    def _monitor_resources(self, stop: threading.Event, started: float) -> None:
        try:
            import psutil

            root = psutil.Process(os.getpid())
            previous_time = time.perf_counter()
            previous_io: dict[int, tuple[int, int]] = {}
            targets = resolve_storage_targets(psutil, self.storage_paths)
            previous_disk_io: dict[str, tuple[int, int]] = {}
            root.cpu_percent(None)
            while not stop.wait(0.5):
                processes = [root]
                try:
                    processes.extend(root.children(recursive=True))
                except psutil.Error:
                    pass
                cpu = 0.0
                rss = 0
                read_bytes = 0
                write_bytes = 0
                current_io: dict[int, tuple[int, int]] = {}
                for process in processes:
                    try:
                        cpu += process.cpu_percent(None)
                        rss += process.memory_info().rss
                        counters = process.io_counters()
                        current = (counters.read_bytes, counters.write_bytes)
                        current_io[process.pid] = current
                        previous = previous_io.get(process.pid, current)
                        read_bytes += max(0, current[0] - previous[0])
                        write_bytes += max(0, current[1] - previous[1])
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.Error):
                        continue
                now = time.perf_counter()
                elapsed = max(now - previous_time, 1e-6)
                try:
                    disk_counters = psutil.disk_io_counters(perdisk=True) or {}
                except psutil.Error:
                    disk_counters = {}
                disks = []
                for target in targets:
                    device = str(target["device"])
                    mountpoint = str(target["mountpoint"])
                    try:
                        usage = psutil.disk_usage(mountpoint)
                    except (OSError, psutil.Error):
                        continue
                    counter_keys = tuple(target.get("counter_keys", ()))
                    counter_key = next(
                        (key for key in counter_keys if key in disk_counters),
                        "",
                    )
                    counter = disk_counters.get(counter_key)
                    disk_read = disk_write = 0.0
                    if counter is not None:
                        current_disk = (int(counter.read_bytes), int(counter.write_bytes))
                        previous_disk = previous_disk_io.get(counter_key, current_disk)
                        disk_read = max(0, current_disk[0] - previous_disk[0]) / 1024**2 / elapsed
                        disk_write = max(0, current_disk[1] - previous_disk[1]) / 1024**2 / elapsed
                        previous_disk_io[counter_key] = current_disk
                    disks.append(
                        {
                            "roles": target.get("roles", ""),
                            "device": device,
                            "mountpoint": mountpoint,
                            "total_gib": usage.total / 1024**3,
                            "used_gib": usage.used / 1024**3,
                            "free_gib": usage.free / 1024**3,
                            "percent": float(usage.percent),
                            "read_mib_s": disk_read,
                            "write_mib_s": disk_write,
                        }
                    )
                logical_cpus = max(1, int(psutil.cpu_count(logical=True) or 1))
                cpu_cores = cpu / 100.0
                cpu_machine_percent = cpu / logical_cpus
                self.resource.emit(
                    {
                        "elapsed": now - started,
                        "cpu": cpu_machine_percent,
                        "cpu_machine_percent": cpu_machine_percent,
                        "cpu_core_percent": cpu,
                        "cpu_cores": cpu_cores,
                        "logical_cpus": logical_cpus,
                        "rss_gib": rss / 1024**3,
                        "read_mib_s": read_bytes / 1024**2 / elapsed,
                        "write_mib_s": write_bytes / 1024**2 / elapsed,
                        "disks": disks,
                    }
                )
                previous_time = now
                previous_io = current_io
        except ImportError:
            return
        except Exception:
            # Resource monitoring is auxiliary and must never turn a finished
            # data task into a GUI traceback because a mount disappeared or a
            # platform-specific psutil field was unavailable.
            return

    def run(self) -> None:
        stream = _SignalStream(self._handle_output)
        monitor_stop = threading.Event()
        started = time.perf_counter()
        monitor = threading.Thread(
            target=self._monitor_resources,
            args=(monitor_stop, started),
            daemon=True,
        )
        monitor.start()
        try:
            with redirect_stdout(stream), redirect_stderr(stream):
                result = self._call()
            stream.flush()
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                self.succeeded.emit(result)
        except Exception as exc:  # noqa: BLE001 - the GUI must report task failures
            stream.flush()
            if self.cancel_event.is_set():
                self.cancelled.emit()
            else:
                details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
                self.failed.emit(details)
        finally:
            monitor_stop.set()
            monitor.join(timeout=1)
            self.finished_cleanly.emit()
