"""Hardware profiling utilities for adaptive backend optimization.

The v1.7.9 plan introduces a ``HardwareProfile`` that combines CPU/memory
resource snapshots with measured storage characteristics.  Measurements are
cached under ``~/.cache/fast-nc-zarr/hardware-profiles`` so repeated runs do
not pay the micro-benchmark cost every time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from .system import (
    RuntimeResourceSnapshot,
    runtime_resource_snapshot,
    storage_profile,
)

MIB = 1024**2
CACHE_ROOT = Path.home() / ".cache" / "fast-nc-zarr" / "hardware-profiles"
DEFAULT_SAMPLE_MIB = 64
DEFAULT_RANDOM_IOPS_SAMPLES = 256


@dataclass(frozen=True)
class StorageBenchmark:
    """Measured storage performance for one role (source/temporary/output)."""

    medium: str
    filesystem: str
    seq_read_mib_s: float
    seq_write_mib_s: float
    rand_read_iops: float
    sample_mib: int
    duration_seconds: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StorageBenchmark":
        return cls(**payload)


@dataclass(frozen=True)
class HardwareProfile:
    """Serializable hardware profile used by adaptive planning."""

    cpu_physical: int
    cpu_logical: int
    cpu_effective: int
    worker_ceiling: int
    memory_total_bytes: int
    memory_available_bytes: int
    storage: dict[str, StorageBenchmark]
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, object]:
        return {
            "cpu_physical": self.cpu_physical,
            "cpu_logical": self.cpu_logical,
            "cpu_effective": self.cpu_effective,
            "worker_ceiling": self.worker_ceiling,
            "memory_total_bytes": self.memory_total_bytes,
            "memory_available_bytes": self.memory_available_bytes,
            "storage": {
                role: benchmark.to_dict() for role, benchmark in self.storage.items()
            },
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HardwareProfile":
        return cls(
            cpu_physical=int(payload["cpu_physical"]),
            cpu_logical=int(payload["cpu_logical"]),
            cpu_effective=int(payload["cpu_effective"]),
            worker_ceiling=int(payload["worker_ceiling"]),
            memory_total_bytes=int(payload["memory_total_bytes"]),
            memory_available_bytes=int(payload["memory_available_bytes"]),
            storage={
                role: StorageBenchmark.from_dict(item)
                for role, item in payload.get("storage", {}).items()
            },
            created_at=float(payload.get("created_at", time.time())),
        )


def _cache_key(paths: tuple[Path, ...]) -> str:
    raw = "\0".join(str(path.expanduser().resolve()) for path in paths)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=16).hexdigest()


def _cache_file(paths: tuple[Path, ...]) -> Path:
    return CACHE_ROOT / f"{_cache_key(paths)}.json"


def _measure_seq_write(directory: Path, sample_mib: int) -> tuple[float, float]:
    """Return ``(write_mib_s, duration_seconds)`` for one sequential write."""
    sample = os.urandom(4 * MIB)
    path = directory / f".fast-nc-zarr-hwprobe-{os.getpid()}-{time.time_ns()}.tmp"
    try:
        started = time.perf_counter()
        with path.open("wb", buffering=0) as handle:
            written = 0
            while written < sample_mib * MIB:
                handle.write(sample)
                written += len(sample)
        duration = max(time.perf_counter() - started, 1e-9)
        return (sample_mib * MIB / duration / MIB, duration)
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def _measure_seq_read(path: Path, sample_mib: int) -> tuple[float, float]:
    """Return ``(read_mib_s, duration_seconds)`` for one sequential read."""
    started = time.perf_counter()
    total = 0
    with path.open("rb", buffering=0) as handle:
        while True:
            chunk = handle.read(4 * MIB)
            if not chunk:
                break
            total += len(chunk)
    duration = max(time.perf_counter() - started, 1e-9)
    return (total / duration / MIB, duration)


def _measure_rand_read_iops(directory: Path, samples: int) -> float:
    """Return random 4 KiB read IOPS using a temporary sparse file."""
    path = directory / f".fast-nc-zarr-hwprobe-rand-{os.getpid()}-{time.time_ns()}.tmp"
    try:
        size = max(64 * MIB, samples * 4096)
        with path.open("wb") as handle:
            handle.seek(size - 1)
            handle.write(b"\0")
        offsets = np.random.default_rng(0).integers(
            0, max(1, size // 4096), size=samples
        ).astype("int64")
        started = time.perf_counter()
        with path.open("rb", buffering=0) as handle:
            for offset in offsets:
                handle.seek(int(offset) * 4096)
                handle.read(4096)
        duration = max(time.perf_counter() - started, 1e-9)
        return samples / duration
    finally:
        try:
            path.unlink()
        except OSError:
            pass


def benchmark_storage_path(
    path: Path,
    *,
    sample_mib: int = DEFAULT_SAMPLE_MIB,
    random_iops_samples: int = DEFAULT_RANDOM_IOPS_SAMPLES,
) -> StorageBenchmark | None:
    """Measure one directory's storage characteristics.

    Returns ``None`` when the directory is not writable or the probe cannot
    run (for example read-only network mounts).
    """
    directory = Path(path).expanduser().resolve()
    if not directory.is_dir():
        directory = directory.parent
    if not os.access(directory, os.W_OK):
        return None
    profile = storage_profile(directory)
    try:
        write_mib_s, write_duration = _measure_seq_write(directory, sample_mib)
        probe = directory / f".fast-nc-zarr-hwprobe-{os.getpid()}-{time.time_ns()}.tmp"
        try:
            with probe.open("wb") as handle:
                handle.write(os.urandom(min(sample_mib, 16) * MIB))
            read_mib_s, read_duration = _measure_seq_read(probe, sample_mib)
        finally:
            try:
                probe.unlink()
            except OSError:
                pass
        iops = _measure_rand_read_iops(directory, random_iops_samples)
        return StorageBenchmark(
            medium=profile.medium,
            filesystem=profile.filesystem,
            seq_read_mib_s=round(read_mib_s, 3),
            seq_write_mib_s=round(write_mib_s, 3),
            rand_read_iops=round(iops, 1),
            sample_mib=sample_mib,
            duration_seconds=round(write_duration + read_duration, 3),
        )
    except (OSError, ValueError):
        return None


def load_cached_profile(paths: tuple[Path, ...]) -> HardwareProfile | None:
    cache = _cache_file(paths)
    try:
        payload = json.loads(cache.read_text(encoding="utf-8"))
        return HardwareProfile.from_dict(payload)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def save_cached_profile(paths: tuple[Path, ...], profile: HardwareProfile) -> None:
    cache = _cache_file(paths)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(cache)
    except OSError:
        pass


def build_hardware_profile(
    source: Path | None = None,
    temporary: Path | None = None,
    output: Path | None = None,
    *,
    use_cache: bool = True,
    sample_mib: int = DEFAULT_SAMPLE_MIB,
) -> HardwareProfile:
    """Build a profile from live resource snapshots and storage probes."""
    paths = tuple(
        path for path in (source, temporary, output) if path is not None
    )
    if use_cache and paths:
        cached = load_cached_profile(paths)
        if cached is not None:
            return cached
    snapshot: RuntimeResourceSnapshot = runtime_resource_snapshot(
        source=source,
        temporary=temporary,
        output=output,
    )
    cpu = snapshot.cpu
    memory = snapshot.memory
    storage: dict[str, StorageBenchmark] = {}
    roles = {
        "source": source,
        "temporary": temporary,
        "output": output,
    }
    for role, path in roles.items():
        if path is None:
            continue
        benchmark = benchmark_storage_path(path, sample_mib=sample_mib)
        if benchmark is not None:
            storage[role] = benchmark
    profile = HardwareProfile(
        cpu_physical=cpu.physical_count,
        cpu_logical=cpu.logical_count,
        cpu_effective=cpu.effective_count,
        worker_ceiling=cpu.worker_ceiling,
        memory_total_bytes=memory.effective_total_bytes,
        memory_available_bytes=memory.effective_available_bytes,
        storage=storage,
    )
    if use_cache and paths:
        save_cached_profile(paths, profile)
    return profile
