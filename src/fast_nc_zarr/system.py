from __future__ import annotations

import os
from pathlib import Path

from .models import StorageProfile


def physical_cpu_count() -> int:
    try:
        import psutil

        return int(psutil.cpu_count(logical=False) or psutil.cpu_count() or 1)
    except ImportError:
        return os.cpu_count() or 1


def nearest_existing(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def storage_profile(path: Path) -> StorageProfile:
    """Resolve mount/device information without requiring the output to exist."""
    import psutil

    existing = nearest_existing(path)
    partitions = sorted(
        psutil.disk_partitions(all=True), key=lambda item: len(item.mountpoint), reverse=True
    )
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
        return StorageProfile(existing, "unknown", None, "unknown")

    rotational = None
    try:
        stat = os.stat(existing)
        sys_device = Path("/sys/dev/block") / f"{os.major(stat.st_dev)}:{os.minor(stat.st_dev)}"
        resolved = sys_device.resolve()
        candidates = [resolved / "queue/rotational", resolved.parent / "queue/rotational"]
        for candidate in candidates:
            if candidate.exists():
                rotational = candidate.read_text(encoding="ascii").strip() == "1"
                break
    except OSError:
        pass
    return StorageProfile(existing, match.device, rotational, match.fstype)


def available_memory(reserve_gib: float) -> int:
    try:
        import psutil

        return max(256 * 1024**2, int(psutil.virtual_memory().available - reserve_gib * 1024**3))
    except ImportError:
        return 4 * 1024**3

