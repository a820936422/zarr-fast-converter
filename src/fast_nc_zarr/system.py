from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import resource as _resource
import tempfile
from typing import Literal, cast

from .models import StorageProfile

try:
    import psutil as _psutil
except ImportError:  # pragma: no cover - the project environment includes psutil
    _psutil = None


MIB = 1024**2
GIB = 1024**3
FD_PER_WORKER = 64
_PROC_ROOT = Path("/proc")
_SYS_ROOT = Path("/sys")
_CGROUP_ROOT = _SYS_ROOT / "fs/cgroup"

StorageOverride = Literal["auto", "ssd", "hdd", "network"]

_STORAGE_OVERRIDES = frozenset({"auto", "ssd", "hdd", "network"})
_NETWORK_FILESYSTEMS = frozenset(
    {
        "9p",
        "afs",
        "ceph",
        "cifs",
        "davfs",
        "davfs2",
        "drvfs",
        "fuse.sshfs",
        "glusterfs",
        "lustre",
        "nfs",
        "nfs4",
        "smbfs",
    }
)
_WSL_VIRTUAL_FILESYSTEMS = frozenset(
    {"btrfs", "ext2", "ext3", "ext4", "f2fs", "overlay", "xfs"}
)


def _redacted_evidence(values: tuple[str, ...]) -> list[str]:
    safe_values = {
        "physical",
        "logical",
        "affinity",
        "cgroup_cpu_quota",
        "cgroup_cpuset",
        "memory",
        "cgroup_memory_limit",
        "cgroup_memory_current",
        "kernel",
        "kernel_generation",
    }
    result = []
    for item in values:
        parts = item.split(":")
        result.append(
            ":".join(parts[:2])
            if parts[0] in safe_values and len(parts) > 1
            else parts[0]
        )
    return result

@dataclass(frozen=True)
class WSLInfo:
    """Evidence-based WSL detection without relying on one environment flag."""

    is_wsl: bool
    version: int | None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "is_wsl": self.is_wsl,
            "version": self.version,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class CPUResources:
    """Host CPU topology and process/container limits."""

    physical_count: int
    logical_count: int
    affinity_count: int | None
    cgroup_quota_cpus: float | None
    cgroup_cpuset_count: int | None
    effective_count: int
    worker_ceiling: int
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "physical_count": self.physical_count,
            "logical_count": self.logical_count,
            "affinity_count": self.affinity_count,
            "cgroup_quota_cpus": self.cgroup_quota_cpus,
            "cgroup_cpuset_count": self.cgroup_cpuset_count,
            "effective_count": self.effective_count,
            "worker_ceiling": self.worker_ceiling,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class MemoryResources:
    """Host and cgroup memory, including the amount currently usable."""

    total_bytes: int
    available_bytes: int
    cgroup_limit_bytes: int | None
    cgroup_current_bytes: int | None
    effective_total_bytes: int
    effective_available_bytes: int
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "total_bytes": self.total_bytes,
            "available_bytes": self.available_bytes,
            "cgroup_limit_bytes": self.cgroup_limit_bytes,
            "cgroup_current_bytes": self.cgroup_current_bytes,
            "effective_total_bytes": self.effective_total_bytes,
            "effective_available_bytes": self.effective_available_bytes,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class RuntimeResourceSnapshot:
    """Serializable resource boundary used by automatic tuning.

    Storage profiles are optional because CPU-only callers need not invent
    paths.  The temporary profile is populated from the system temp directory
    when :func:`runtime_resource_snapshot` is called without an explicit path.
    """

    cpu: CPUResources
    memory: MemoryResources
    wsl: WSLInfo
    source_storage: StorageProfile | None = None
    temporary_storage: StorageProfile | None = None
    output_storage: StorageProfile | None = None
    fd_soft_limit: int | None = None

    def worker_ceiling(
        self,
        memory_per_worker_bytes: int = 512 * MIB,
        *,
        reserve_memory_bytes: int = 0,
        requested: int | None = None,
    ) -> int:
        """Return a safe CPU/memory upper bound for process workers.

        ``requested`` remains authoritative as a *hard maximum*, but cannot
        exceed the detected process/container boundary.  At least one worker
        is returned so callers can make progress and surface an allocation
        error rather than silently scheduling no work.
        """

        if memory_per_worker_bytes <= 0:
            raise ValueError("memory_per_worker_bytes must be positive")
        if reserve_memory_bytes < 0:
            raise ValueError("reserve_memory_bytes cannot be negative")
        if requested is not None and requested < 1:
            raise ValueError("requested worker count must be positive")

        usable_memory = max(
            0,
            self.memory.effective_available_bytes - reserve_memory_bytes,
        )
        memory_limit = max(1, usable_memory // memory_per_worker_bytes)
        limits = [self.cpu.worker_ceiling, memory_limit]
        if self.fd_soft_limit is not None:
            limits.append(max(1, int(self.fd_soft_limit) // FD_PER_WORKER))
        ceiling = max(1, min(limits))
        if requested is not None:
            ceiling = min(ceiling, requested)
        return ceiling

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe resource summary without host-identifying paths."""

        cpu = self.cpu.to_dict()
        memory = self.memory.to_dict()
        wsl = self.wsl.to_dict()
        cpu["evidence"] = _redacted_evidence(self.cpu.evidence)
        memory["evidence"] = _redacted_evidence(self.memory.evidence)
        wsl["evidence"] = _redacted_evidence(self.wsl.evidence)
        return {
            "cpu": cpu,
            "memory": memory,
            "wsl": wsl,
            "fd_soft_limit": self.fd_soft_limit,
            "storage": {
                "source": (
                    self.source_storage.to_dict(redact_paths=True)
                    if self.source_storage is not None
                    else None
                ),
                "temporary": (
                    self.temporary_storage.to_dict(redact_paths=True)
                    if self.temporary_storage is not None
                    else None
                ),
                "output": (
                    self.output_storage.to_dict(redact_paths=True)
                    if self.output_storage is not None
                    else None
                ),
            },
        }
@dataclass(frozen=True)
class EffectiveResourceBudget:
    """Unified, serializable resource contract shared by all backends."""

    cpu_physical: int
    cpu_logical: int
    cpu_effective: int
    memory_available_bytes: int
    memory_total_bytes: int
    memory_budget_bytes: int
    fd_soft_limit: int | None
    worker_ceiling: int
    source_storage: StorageProfile | None = None
    temporary_storage: StorageProfile | None = None
    output_storage: StorageProfile | None = None
    same_device_roles: tuple[str, ...] = ()
    limit_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a manifest-safe resource contract with redacted paths."""

        return {
            "cpu_physical": self.cpu_physical,
            "cpu_logical": self.cpu_logical,
            "cpu_effective": self.cpu_effective,
            "memory_available_bytes": self.memory_available_bytes,
            "memory_total_bytes": self.memory_total_bytes,
            "memory_budget_bytes": self.memory_budget_bytes,
            "fd_soft_limit": self.fd_soft_limit,
            "worker_ceiling": self.worker_ceiling,
            "source_storage": (
                self.source_storage.to_dict(redact_paths=True)
                if self.source_storage is not None
                else None
            ),
            "temporary_storage": (
                self.temporary_storage.to_dict(redact_paths=True)
                if self.temporary_storage is not None
                else None
            ),
            "output_storage": (
                self.output_storage.to_dict(redact_paths=True)
                if self.output_storage is not None
                else None
            ),
            "same_device_roles": list(self.same_device_roles),
            "limit_reasons": list(self.limit_reasons),
        }


def _same_device_roles(
    source: StorageProfile | None,
    temporary: StorageProfile | None,
    output: StorageProfile | None,
) -> tuple[str, ...]:
    profiles = {
        "source": source,
        "temporary": temporary,
        "output": output,
    }
    roles = []
    items = tuple((name, profile) for name, profile in profiles.items() if profile is not None)
    for index, (left_name, left) in enumerate(items):
        if left.device == "unknown":
            continue
        for right_name, right in items[index + 1 :]:
            if right.device != "unknown" and left.device == right.device:
                roles.append(f"{left_name}+{right_name}")
    return tuple(roles)


def effective_resource_budget(
    snapshot: RuntimeResourceSnapshot | None = None,
    *,
    source: Path | None = None,
    temporary: Path | None = None,
    output: Path | None = None,
    storage_overrides: Mapping[str, StorageOverride | str] | None = None,
    reserve_memory_bytes: int = 0,
    memory_per_worker_bytes: int = 512 * MIB,
    requested: int | None = None,
) -> EffectiveResourceBudget:
    """Resolve one effective CPU/memory/FD/storage contract for a task."""

    resources = snapshot or runtime_resource_snapshot(
        source=source,
        temporary=temporary,
        output=output,
        storage_overrides=storage_overrides,
    )
    memory_budget = max(
        256 * MIB,
        int(resources.memory.effective_available_bytes) - int(reserve_memory_bytes),
    )
    ceiling = resources.worker_ceiling(
        memory_per_worker_bytes,
        reserve_memory_bytes=reserve_memory_bytes,
        requested=requested,
    )
    reasons = [
        f"cpu_effective={resources.cpu.effective_count}",
        f"memory_budget={memory_budget}",
    ]
    if resources.fd_soft_limit is not None:
        reasons.append(f"fd_soft_limit={resources.fd_soft_limit}")
    if requested is not None:
        reasons.append(f"requested={requested}")
    return EffectiveResourceBudget(
        cpu_physical=resources.cpu.physical_count,
        cpu_logical=resources.cpu.logical_count,
        cpu_effective=resources.cpu.effective_count,
        memory_available_bytes=resources.memory.effective_available_bytes,
        memory_total_bytes=resources.memory.effective_total_bytes,
        memory_budget_bytes=memory_budget,
        fd_soft_limit=resources.fd_soft_limit,
        worker_ceiling=ceiling,
        source_storage=resources.source_storage,
        temporary_storage=resources.temporary_storage,
        output_storage=resources.output_storage,
        same_device_roles=_same_device_roles(
            resources.source_storage,
            resources.temporary_storage,
            resources.output_storage,
        ),
        limit_reasons=tuple(reasons),
    )


def _file_descriptor_soft_limit() -> int | None:
    try:
        soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
    except (AttributeError, OSError, ValueError):
        return None
    if soft == _resource.RLIM_INFINITY:
        return None
    return max(0, int(soft))



def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except (OSError, UnicodeError):
        return None


def detect_wsl(*, environ: Mapping[str, str] | None = None) -> WSLInfo:
    """Detect WSL/WSL2 from mocked-friendly environment and ``/proc`` data."""

    environment = os.environ if environ is None else environ
    evidence: list[str] = []
    for name in ("WSL_INTEROP", "WSL_DISTRO_NAME"):
        if environment.get(name):
            evidence.append(f"environment:{name}")

    osrelease = _read_text(_PROC_ROOT / "sys/kernel/osrelease") or ""
    version_text = _read_text(_PROC_ROOT / "version") or ""
    combined = f"{osrelease}\n{version_text}".lower()
    if "microsoft" in combined:
        evidence.append("kernel:microsoft")
    if "wsl" in combined:
        evidence.append("kernel:wsl")

    is_wsl = bool(evidence)
    if not is_wsl:
        return WSLInfo(False, None, ())

    if "wsl2" in combined or "microsoft-standard" in combined:
        version = 2
        evidence.append("kernel_generation:2")
    else:
        version = 1
        evidence.append("kernel_generation:1_or_unknown")
    return WSLInfo(True, version, tuple(dict.fromkeys(evidence)))


def _host_cpu_counts() -> tuple[int, int, tuple[str, ...]]:
    physical: int | None = None
    logical: int | None = None
    evidence: list[str] = []

    if _psutil is not None:
        try:
            reported_physical = _psutil.cpu_count(logical=False)
            reported_logical = _psutil.cpu_count(logical=True)
            if reported_physical:
                physical = int(reported_physical)
                evidence.append("physical:psutil")
            if reported_logical:
                logical = int(reported_logical)
                evidence.append("logical:psutil")
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    if logical is None:
        logical = int(os.cpu_count() or 1)
        evidence.append("logical:os.cpu_count")
    logical = max(1, logical)

    if physical is None:
        physical = logical
        evidence.append("physical:fallback_logical")
    physical = max(1, min(physical, logical))
    return physical, logical, tuple(evidence)


def physical_cpu_count() -> int:
    """Return host physical CPUs, preserving the original public API."""

    return _host_cpu_counts()[0]


def logical_cpu_count() -> int:
    """Return the host logical CPU count before affinity/cgroup limits."""

    return _host_cpu_counts()[1]


def _affinity_cpu_count() -> tuple[int | None, str | None]:
    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            count = len(get_affinity(0))
            if count > 0:
                return count, "affinity:os.sched_getaffinity"
        except (OSError, TypeError, ValueError):
            pass

    if _psutil is not None:
        process_factory = getattr(_psutil, "Process", None)
        if process_factory is not None:
            try:
                count = len(process_factory().cpu_affinity())
                if count > 0:
                    return count, "affinity:psutil"
            except (AttributeError, OSError, TypeError, ValueError):
                pass
    return None, None


def affinity_cpu_count() -> int | None:
    """Return CPUs allowed to this process, if the OS exposes affinity."""

    return _affinity_cpu_count()[0]


def _cgroup_memberships() -> tuple[str | None, dict[str, tuple[str, str]]]:
    text = _read_text(_PROC_ROOT / "self/cgroup")
    unified: str | None = None
    controllers: dict[str, tuple[str, str]] = {}
    if text is None:
        return unified, controllers

    for line in text.splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, names_text, relative = parts
        if not names_text:
            unified = relative
            continue
        names = tuple(name for name in names_text.split(",") if name)
        for name in names:
            controllers[name] = (relative, names_text)
    return unified, controllers


def _append_unique(paths: list[Path], candidate: Path) -> None:
    if candidate not in paths:
        paths.append(candidate)

def _append_with_ancestors(paths: list[Path], candidate: Path) -> None:
    current = candidate
    while current == _CGROUP_ROOT or _CGROUP_ROOT in current.parents:
        _append_unique(paths, current)
        if current == _CGROUP_ROOT:
            break
        current = current.parent


def _cgroup_directories(
    controller: str,
    *,
    unified: bool,
) -> tuple[Path, ...]:
    unified_path, controllers = _cgroup_memberships()
    paths: list[Path] = []
    if unified:
        if unified_path is not None:
            relative = unified_path.lstrip("/")
            if relative:
                _append_with_ancestors(paths, _CGROUP_ROOT / relative)
        _append_unique(paths, _CGROUP_ROOT)
        return tuple(paths)

    membership = controllers.get(controller)
    if membership is not None:
        relative_text, group_name = membership
        relative = relative_text.lstrip("/")
        if relative:
            _append_with_ancestors(paths, _CGROUP_ROOT / group_name / relative)
            _append_with_ancestors(paths, _CGROUP_ROOT / controller / relative)
            _append_with_ancestors(paths, _CGROUP_ROOT / relative)
        _append_with_ancestors(paths, _CGROUP_ROOT / group_name)
    _append_with_ancestors(paths, _CGROUP_ROOT / controller)
    _append_unique(paths, _CGROUP_ROOT)
    return tuple(paths)


def _parse_nonnegative_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed >= 0 else None


def _cpuset_count(value: str) -> int | None:
    count = 0
    if not value.strip():
        return None
    try:
        for item in value.split(","):
            item = item.strip()
            if not item:
                return None
            if "-" in item:
                start_text, stop_text = item.split("-", 1)
                start = int(start_text)
                stop = int(stop_text)
                if start < 0 or stop < start:
                    return None
                count += stop - start + 1
            else:
                if int(item) < 0:
                    return None
                count += 1
    except ValueError:
        return None
    return count or None


def _cgroup_cpu_limits() -> tuple[float | None, int | None, tuple[str, ...]]:
    quota_cpus: float | None = None
    cpuset_count: int | None = None
    evidence: list[str] = []

    v2_found = False
    for directory in _cgroup_directories("cpu", unified=True):
        path = directory / "cpu.max"
        value = _read_text(path)
        if value is None:
            continue
        v2_found = True
        fields = value.split()
        if len(fields) >= 2:
            if fields[0] == "max":
                evidence.append(f"cgroup_cpu_quota:unlimited:{path}")
            else:
                try:
                    quota = int(fields[0])
                    period = int(fields[1])
                    if quota > 0 and period > 0:
                        candidate = quota / period
                        quota_cpus = (
                            candidate
                            if quota_cpus is None
                            else min(quota_cpus, candidate)
                        )
                        evidence.append(f"cgroup_cpu_quota:v2:{path}")
                except ValueError:
                    evidence.append(f"cgroup_cpu_quota:invalid:{path}")
        else:
            evidence.append(f"cgroup_cpu_quota:invalid:{path}")

    if not v2_found:
        for directory in _cgroup_directories("cpu", unified=False):
            quota_path = directory / "cpu.cfs_quota_us"
            period_path = directory / "cpu.cfs_period_us"
            quota_text = _read_text(quota_path)
            period_text = _read_text(period_path)
            if quota_text is None or period_text is None:
                continue
            try:
                quota = int(quota_text)
                period = int(period_text)
                if quota > 0 and period > 0:
                    candidate = quota / period
                    quota_cpus = (
                        candidate
                        if quota_cpus is None
                        else min(quota_cpus, candidate)
                    )
                    evidence.append(f"cgroup_cpu_quota:v1:{quota_path}")
                else:
                    evidence.append(f"cgroup_cpu_quota:unlimited:{quota_path}")
            except ValueError:
                evidence.append(f"cgroup_cpu_quota:invalid:{quota_path}")

    cpuset_directories = (
        *_cgroup_directories("cpuset", unified=True),
        *_cgroup_directories("cpuset", unified=False),
    )
    for filename in ("cpuset.cpus.effective", "cpuset.cpus"):
        for directory in cpuset_directories:
            path = directory / filename
            value = _read_text(path)
            if value is None:
                continue
            parsed = _cpuset_count(value)
            if parsed is not None:
                cpuset_count = (
                    parsed if cpuset_count is None else min(cpuset_count, parsed)
                )
                evidence.append(f"cgroup_cpuset:{path}")

    return quota_cpus, cpuset_count, tuple(evidence)


def cpu_resources() -> CPUResources:
    """Capture CPU topology plus affinity and cgroup constraints."""

    physical, logical, host_evidence = _host_cpu_counts()
    affinity, affinity_evidence = _affinity_cpu_count()
    quota, cpuset, cgroup_evidence = _cgroup_cpu_limits()

    limits = [logical]
    if affinity is not None:
        limits.append(affinity)
    if cpuset is not None:
        limits.append(cpuset)
    if quota is not None:
        limits.append(max(1, math.floor(quota)))
    effective = max(1, min(limits))
    cpu_worker_ceiling = max(1, min(physical, effective))

    evidence = list(host_evidence)
    if affinity_evidence is not None:
        evidence.append(affinity_evidence)
    evidence.extend(cgroup_evidence)
    return CPUResources(
        physical_count=physical,
        logical_count=logical,
        affinity_count=affinity,
        cgroup_quota_cpus=quota,
        cgroup_cpuset_count=cpuset,
        effective_count=effective,
        worker_ceiling=cpu_worker_ceiling,
        evidence=tuple(evidence),
    )


def _host_memory() -> tuple[int, int, tuple[str, ...]]:
    if _psutil is not None:
        try:
            memory = _psutil.virtual_memory()
            total = max(0, int(memory.total))
            available = max(0, min(total, int(memory.available)))
            if total > 0:
                return total, available, ("memory:psutil",)
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total_pages = int(os.sysconf("SC_PHYS_PAGES"))
        available_pages = int(os.sysconf("SC_AVPHYS_PAGES"))
        total = max(0, page_size * total_pages)
        available = max(0, min(total, page_size * available_pages))
        if total > 0:
            return total, available, ("memory:os.sysconf",)
    except (AttributeError, OSError, TypeError, ValueError):
        pass
    return 4 * GIB, 4 * GIB, ("memory:conservative_fallback",)


def _cgroup_memory_values() -> tuple[int | None, int | None, tuple[str, ...]]:
    limit: int | None = None
    current: int | None = None
    evidence: list[str] = []
    v2_found = False
    minimum_remaining: int | None = None
    for directory in _cgroup_directories("memory", unified=True):
        limit_path = directory / "memory.max"
        limit_text = _read_text(limit_path)
        if limit_text is None:
            continue
        v2_found = True
        current_path = directory / "memory.current"
        candidate_current = _parse_nonnegative_int(_read_text(current_path))
        if limit_text == "max":
            evidence.append(f"cgroup_memory_limit:unlimited:{limit_path}")
        else:
            candidate_limit = _parse_nonnegative_int(limit_text)
            evidence.append(
                f"cgroup_memory_limit:{'v2' if candidate_limit is not None else 'invalid'}:{limit_path}"
            )
            if candidate_limit is not None:
                limit = candidate_limit if limit is None else min(limit, candidate_limit)
                candidate_remaining = (
                    max(0, candidate_limit - candidate_current)
                    if candidate_current is not None
                    else candidate_limit
                )
                minimum_remaining = (
                    candidate_remaining
                    if minimum_remaining is None
                    else min(minimum_remaining, candidate_remaining)
                )
        if candidate_current is not None:
            evidence.append(f"cgroup_memory_current:v2:{current_path}")

    if not v2_found:
        for directory in _cgroup_directories("memory", unified=False):
            limit_path = directory / "memory.limit_in_bytes"
            limit_text = _read_text(limit_path)
            if limit_text is None:
                continue
            parsed_limit = _parse_nonnegative_int(limit_text)
            candidate_limit = (
                parsed_limit
                if parsed_limit is not None and parsed_limit < (1 << 60)
                else None
            )
            current_path = directory / "memory.usage_in_bytes"
            candidate_current = _parse_nonnegative_int(_read_text(current_path))
            evidence.append(
                f"cgroup_memory_limit:{'v1' if candidate_limit is not None else 'unlimited'}:{limit_path}"
            )
            if candidate_limit is not None:
                limit = candidate_limit if limit is None else min(limit, candidate_limit)
                candidate_remaining = (
                    max(0, candidate_limit - candidate_current)
                    if candidate_current is not None
                    else candidate_limit
                )
                minimum_remaining = (
                    candidate_remaining
                    if minimum_remaining is None
                    else min(minimum_remaining, candidate_remaining)
                )
            if candidate_current is not None:
                evidence.append(f"cgroup_memory_current:v1:{current_path}")

    if limit is not None and minimum_remaining is not None:
        current = max(0, limit - min(limit, minimum_remaining))
    return limit, current, tuple(evidence)

def memory_resources() -> MemoryResources:
    """Capture host memory and the finite cgroup remainder, when present."""

    total, available, host_evidence = _host_memory()
    cgroup_limit, cgroup_current, cgroup_evidence = _cgroup_memory_values()
    effective_total = min(total, cgroup_limit) if cgroup_limit is not None else total
    effective_available = available
    if cgroup_limit is not None:
        cgroup_remaining = (
            max(0, cgroup_limit - cgroup_current)
            if cgroup_current is not None
            else cgroup_limit
        )
        effective_available = min(effective_available, cgroup_remaining)
    effective_available = min(effective_available, effective_total)
    return MemoryResources(
        total_bytes=total,
        available_bytes=available,
        cgroup_limit_bytes=cgroup_limit,
        cgroup_current_bytes=cgroup_current,
        effective_total_bytes=effective_total,
        effective_available_bytes=effective_available,
        evidence=host_evidence + cgroup_evidence,
    )


def nearest_existing(path: Path) -> Path:
    current = Path(path).expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _normalise_storage_override(override: StorageOverride | str) -> StorageOverride:
    value = str(override).strip().lower()
    if value not in _STORAGE_OVERRIDES:
        choices = ", ".join(sorted(_STORAGE_OVERRIDES))
        raise ValueError(f"storage override must be one of: {choices}")
    return cast(StorageOverride, value)


def _partition_for(path: Path):
    if _psutil is None:
        return None
    try:
        partitions = sorted(
            _psutil.disk_partitions(all=True),
            key=lambda item: len(str(item.mountpoint)),
            reverse=True,
        )
    except (AttributeError, OSError, TypeError, ValueError):
        return None
    return next(
        (
            item
            for item in partitions
            if path == Path(item.mountpoint) or Path(item.mountpoint) in path.parents
        ),
        None,
    )


def _reported_rotational(
    path: Path,
) -> tuple[bool | None, Path | None, Path | None, str | None]:
    try:
        stat = os.stat(path)
        major = os.major(stat.st_dev)
        minor = os.minor(stat.st_dev)
        sys_device = _SYS_ROOT / "dev/block" / f"{major}:{minor}"
        resolved = sys_device.resolve()
        candidates = (resolved / "queue/rotational", resolved.parent / "queue/rotational")
        for candidate in candidates:
            raw = _read_text(candidate)
            if raw is None:
                continue
            if raw == "1":
                return True, candidate, resolved, raw
            if raw == "0":
                return False, candidate, resolved, raw
            return None, candidate, resolved, raw
        return None, None, resolved, None
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        return None, None, None, None


def _is_network_storage(device: str, filesystem: str) -> bool:
    lowered_fs = filesystem.lower()
    return (
        lowered_fs in _NETWORK_FILESYSTEMS
        or lowered_fs.startswith("nfs")
        or lowered_fs.startswith("fuse.sshfs")
        or device.startswith("//")
        or (":" in device and not device.startswith("/dev/"))
    )


def _is_wsl_virtual_storage(
    wsl: WSLInfo,
    *,
    device: str,
    filesystem: str,
    resolved_sys_device: Path | None,
) -> bool:
    if not wsl.is_wsl or _is_network_storage(device, filesystem):
        return False
    if filesystem.lower() in _WSL_VIRTUAL_FILESYSTEMS:
        return True
    if device.startswith("/dev/"):
        return True
    return resolved_sys_device is not None and "virtual" in resolved_sys_device.parts


def _storage_profile(
    path: Path,
    override: StorageOverride,
    wsl: WSLInfo,
) -> StorageProfile:
    existing = nearest_existing(path)
    match = _partition_for(existing)
    mountpoint = Path(match.mountpoint) if match is not None else None
    device = str(match.device) if match is not None and match.device else "unknown"
    filesystem = str(match.fstype) if match is not None and match.fstype else "unknown"
    reported, rotational_path, resolved_device, raw_rotational = _reported_rotational(
        existing
    )

    evidence = [f"nearest_existing:{existing}"]
    if match is not None:
        evidence.extend(
            (
                f"mountpoint:{mountpoint}",
                f"device:{device}",
                f"filesystem:{filesystem}",
            )
        )
    else:
        evidence.append("mountpoint:unresolved")
    if rotational_path is not None:
        evidence.append(f"sysfs_rotational:{raw_rotational}:{rotational_path}")
    if resolved_device is not None:
        evidence.append(f"sysfs_device:{resolved_device}")

    if override != "auto":
        evidence.append(f"user_override:{override}")
        rotational = True if override == "hdd" else False if override == "ssd" else None
        medium = override
        confidence = "high"
    elif _is_network_storage(device, filesystem):
        rotational = None
        medium = "network"
        confidence = "medium"
        evidence.append("classification:network_filesystem")
    elif _is_wsl_virtual_storage(
        wsl,
        device=device,
        filesystem=filesystem,
        resolved_sys_device=resolved_device,
    ):
        # WSL2 presents its VHDX-backed ext4 disk as a rotational Linux block
        # device even when the Windows host uses an SSD.  Preserve that report
        # as evidence, but never turn it into a trusted HDD assertion.
        rotational = None
        medium = "virtual_unknown"
        confidence = "low"
        evidence.append("classification:wsl_virtual_disk_sysfs_untrusted")
    elif reported is not None:
        rotational = reported
        medium = "hdd" if reported else "ssd"
        confidence = (
            "medium"
            if resolved_device is not None and "virtual" in resolved_device.parts
            else "high"
        )
        evidence.append("classification:sysfs_rotational")
    else:
        rotational = None
        medium = "unknown"
        confidence = "unknown"
        evidence.append("classification:insufficient_evidence")

    return StorageProfile(
        path=existing,
        device=device,
        rotational=rotational,
        filesystem=filesystem,
        medium=cast(
            Literal["ssd", "hdd", "network", "unknown", "virtual_unknown"],
            medium,
        ),
        reported_rotational=reported,
        confidence=cast(Literal["high", "medium", "low", "unknown"], confidence),
        mountpoint=mountpoint,
        evidence=tuple(evidence),
        override=override,
    )


def storage_profile(
    path: Path,
    override: StorageOverride | str = "auto",
) -> StorageProfile:
    """Resolve storage media while retaining evidence and legacy fields.

    Automatic classification trusts Linux sysfs on bare metal.  WSL virtual
    block devices deliberately return ``rotational=None`` because their VHDX
    rotational bit does not describe the Windows host medium.  A user override
    is always applied last and therefore has the highest confidence.
    """

    normalised_override = _normalise_storage_override(override)
    return _storage_profile(Path(path), normalised_override, detect_wsl())


def runtime_resource_snapshot(
    source: Path | None = None,
    temporary: Path | None = None,
    output: Path | None = None,
    *,
    storage_overrides: Mapping[str, StorageOverride | str] | None = None,
) -> RuntimeResourceSnapshot:
    """Capture CPU, memory, WSL, and source/temp/output storage evidence."""

    overrides = dict(storage_overrides or {})
    unexpected_roles = set(overrides) - {"source", "temporary", "output"}
    if unexpected_roles:
        roles = ", ".join(sorted(unexpected_roles))
        raise ValueError(f"unknown storage override roles: {roles}")
    normalised_overrides = {
        role: _normalise_storage_override(value) for role, value in overrides.items()
    }

    wsl = detect_wsl()

    def profile(role: str, profile_path: Path | None) -> StorageProfile | None:
        if profile_path is None:
            return None
        override = normalised_overrides.get(role, "auto")
        return _storage_profile(Path(profile_path), override, wsl)

    temporary_path = Path(tempfile.gettempdir()) if temporary is None else temporary
    return RuntimeResourceSnapshot(
        cpu=cpu_resources(),
        memory=memory_resources(),
        wsl=wsl,
        source_storage=profile("source", source),
        temporary_storage=profile("temporary", temporary_path),
        output_storage=profile("output", output),
        fd_soft_limit=_file_descriptor_soft_limit(),
    )


def worker_ceiling(
    snapshot: RuntimeResourceSnapshot | None = None,
    memory_per_worker_bytes: int = 512 * MIB,
    *,
    reserve_memory_bytes: int = 0,
    requested: int | None = None,
) -> int:
    """Convenience wrapper for the snapshot's CPU/memory worker boundary."""

    resources = snapshot if snapshot is not None else runtime_resource_snapshot()
    return resources.worker_ceiling(
        memory_per_worker_bytes,
        reserve_memory_bytes=reserve_memory_bytes,
        requested=requested,
    )


def available_memory(reserve_gib: float) -> int:
    """Return effective available memory after a caller-selected reserve.

    The historical 256 MiB floor remains for compatibility with existing
    planning code; new tuners should use :class:`MemoryResources` directly
    when strict cgroup exhaustion handling is required.
    """

    effective_available = memory_resources().effective_available_bytes
    return max(256 * MIB, int(effective_available - reserve_gib * GIB))
