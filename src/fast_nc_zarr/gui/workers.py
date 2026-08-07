from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import inspect
import os
import time
import threading
import traceback
from typing import Callable, Any

from PySide6.QtCore import QThread, Signal


def _disk_mounts(psutil) -> list[tuple[str, str]]:
    """Discover mounted, non-pseudo filesystems for compact GUI monitoring."""

    pseudo_types = {
        "autofs",
        "cgroup",
        "cgroup2",
        "devpts",
        "devtmpfs",
        "mqueue",
        "proc",
        "pstore",
        "securityfs",
        "sysfs",
        "tmpfs",
        "tracefs",
        "overlay",
        "squashfs",
    }
    mounts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    try:
        partitions = psutil.disk_partitions(all=False)
    except psutil.Error:
        partitions = []
    for partition in partitions:
        device = str(partition.device or "")
        mountpoint = str(partition.mountpoint or "")
        filesystem = str(partition.fstype or "").lower()
        if not mountpoint or filesystem in pseudo_types:
            continue
        key = (device, mountpoint)
        if key in seen:
            continue
        try:
            psutil.disk_usage(mountpoint)
        except (OSError, psutil.Error):
            continue
        seen.add(key)
        mounts.append(key)
    if mounts:
        return mounts
    # Some container/desktop environments hide partition metadata.  Keep a
    # useful fallback for the filesystem hosting the application.
    try:
        mountpoint = os.getcwd()
        psutil.disk_usage(mountpoint)
        return [("filesystem", mountpoint)]
    except (OSError, psutil.Error):
        return []


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


class TaskWorker(QThread):
    """Run a synchronous application service without blocking Qt's UI thread."""

    log = Signal(str)
    succeeded = Signal(object)
    failed = Signal(str)
    finished_cleanly = Signal()
    resource = Signal(object)
    cancelled = Signal()

    def __init__(self, function: Callable[[], Any], parent=None) -> None:
        super().__init__(parent)
        self.function = function
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()
        self.log.emit("已请求取消；当前正在执行的块完成后停止。")

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
            mounts = _disk_mounts(psutil)
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
                disks = []
                for device, mountpoint in mounts:
                    try:
                        usage = psutil.disk_usage(mountpoint)
                    except (OSError, psutil.Error):
                        continue
                    disks.append(
                        {
                            "device": device,
                            "mountpoint": mountpoint,
                            "total_gib": usage.total / 1024**3,
                            "used_gib": usage.used / 1024**3,
                            "free_gib": usage.free / 1024**3,
                            "percent": float(usage.percent),
                        }
                    )
                self.resource.emit(
                    {
                        "elapsed": now - started,
                        "cpu": cpu,
                        "rss_gib": rss / 1024**3,
                        # ``read_bytes``/``write_bytes`` are already the
                        # per-interval deltas accumulated above.  The old
                        # implementation subtracted the removed aggregate
                        # counters ``previous_read``/``previous_write`` here,
                        # which caused the monitor thread to raise NameError.
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
        stream = _SignalStream(self.log.emit)
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
