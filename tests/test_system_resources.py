from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fast_nc_zarr.models import StorageProfile
from fast_nc_zarr import system
from fast_nc_zarr.system import (
    CPUResources,
    MemoryResources,
    RuntimeResourceSnapshot,
    WSLInfo,
)


MIB = 1024**2
GIB = 1024**3


class _FakePsutil:
    def __init__(
        self,
        *,
        physical_cpus: int = 8,
        logical_cpus: int = 16,
        total_memory: int = 16 * GIB,
        available_memory: int = 8 * GIB,
        partitions: tuple[SimpleNamespace, ...] = (),
    ) -> None:
        self.physical_cpus = physical_cpus
        self.logical_cpus = logical_cpus
        self.total_memory = total_memory
        self.available_memory = available_memory
        self.partitions = partitions

    def cpu_count(self, logical: bool = True) -> int:
        return self.logical_cpus if logical else self.physical_cpus

    def virtual_memory(self) -> SimpleNamespace:
        return SimpleNamespace(
            total=self.total_memory,
            available=self.available_memory,
        )

    def disk_partitions(self, *, all: bool) -> tuple[SimpleNamespace, ...]:
        del all
        return self.partitions


class SystemResourceTests(unittest.TestCase):
    def _roots(self, root: Path, osrelease: str) -> tuple[Path, Path, Path]:
        proc_root = root / "proc"
        sys_root = root / "sys"
        cgroup_root = sys_root / "fs/cgroup"
        (proc_root / "sys/kernel").mkdir(parents=True)
        (proc_root / "sys/kernel/osrelease").write_text(
            osrelease,
            encoding="utf-8",
        )
        (proc_root / "version").write_text(
            f"Linux version test {osrelease}",
            encoding="utf-8",
        )
        cgroup_root.mkdir(parents=True)
        return proc_root, sys_root, cgroup_root

    def _storage_context(
        self,
        root: Path,
        *,
        osrelease: str,
        rotational: str,
        filesystem: str = "ext4",
    ):
        proc_root, sys_root, cgroup_root = self._roots(root, osrelease)
        queue = sys_root / "dev/block/8:1/queue"
        queue.mkdir(parents=True)
        (queue / "rotational").write_text(rotational, encoding="ascii")
        fake_psutil = _FakePsutil(
            partitions=(
                SimpleNamespace(
                    mountpoint="/dataset",
                    device="/dev/sda1",
                    fstype=filesystem,
                ),
            )
        )
        return (
            patch.object(system, "_PROC_ROOT", proc_root),
            patch.object(system, "_SYS_ROOT", sys_root),
            patch.object(system, "_CGROUP_ROOT", cgroup_root),
            patch.object(system, "_psutil", fake_psutil),
            patch.object(
                system,
                "nearest_existing",
                return_value=Path("/dataset/source"),
            ),
            patch.object(
                system.os,
                "stat",
                return_value=SimpleNamespace(
                    st_dev=os.makedev(8, 1), st_size=1, st_mtime=1.0
                ),
            ),
            patch.dict(system.os.environ, {}, clear=True),
        )

    def test_wsl2_virtual_ext4_does_not_claim_reported_hdd(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            contexts = self._storage_context(
                Path(temporary),
                osrelease="6.6.87.2-microsoft-standard-WSL2",
                rotational="1",
            )
            with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6]:
                profile = system.storage_profile(Path("/dataset/source"))

        self.assertIsNone(profile.rotational)
        self.assertIs(profile.reported_rotational, True)
        self.assertEqual(profile.medium, "virtual_unknown")
        self.assertEqual(profile.confidence, "low")
        self.assertTrue(
            any("wsl_virtual_disk_sysfs_untrusted" in item for item in profile.evidence)
        )

    def test_bare_metal_keeps_sysfs_rotational_classification(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            contexts = self._storage_context(
                Path(temporary),
                osrelease="6.8.0-generic",
                rotational="1",
            )
            with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6]:
                profile = system.storage_profile(Path("/dataset/source"))

        self.assertIs(profile.rotational, True)
        self.assertIs(profile.reported_rotational, True)
        self.assertEqual(profile.medium, "hdd")
        self.assertEqual(profile.confidence, "high")

    def test_storage_override_has_priority_and_preserves_report(self) -> None:
        from tempfile import TemporaryDirectory

        expected = {
            "auto": (None, "virtual_unknown", "low"),
            "ssd": (False, "ssd", "high"),
            "hdd": (True, "hdd", "high"),
            "network": (None, "network", "high"),
        }
        with TemporaryDirectory() as temporary:
            contexts = self._storage_context(
                Path(temporary),
                osrelease="6.6.87.2-microsoft-standard-WSL2",
                rotational="1",
            )
            with contexts[0], contexts[1], contexts[2], contexts[3], contexts[4], contexts[5], contexts[6]:
                for override, values in expected.items():
                    with self.subTest(override=override):
                        profile = system.storage_profile(
                            Path("/dataset/source"),
                            override=override,
                        )
                        self.assertEqual(
                            (profile.rotational, profile.medium, profile.confidence),
                            values,
                        )
                        self.assertIs(profile.reported_rotational, True)

        with self.assertRaises(ValueError):
            system.storage_profile(Path("/dataset/source"), override="nvme")

    def test_affinity_and_cgroup_each_constrain_cpu_worker_ceiling(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc_root, sys_root, cgroup_root = self._roots(root, "6.8.0-generic")
            (proc_root / "self").mkdir()
            (proc_root / "self/cgroup").write_text("0::/jobs/demo\n", encoding="ascii")
            job = cgroup_root / "jobs/demo"
            job.mkdir(parents=True)
            (job / "cpuset.cpus.effective").write_text("0-7", encoding="ascii")
            (job / "cpu.max").write_text("800000 100000", encoding="ascii")
            fake_psutil = _FakePsutil(physical_cpus=8, logical_cpus=16)
            with (
                patch.object(system, "_PROC_ROOT", proc_root),
                patch.object(system, "_SYS_ROOT", sys_root),
                patch.object(system, "_CGROUP_ROOT", cgroup_root),
                patch.object(system, "_psutil", fake_psutil),
                patch.object(
                    system.os,
                    "sched_getaffinity",
                    return_value={0, 1, 2},
                    create=True,
                ),
            ):
                affinity_limited = system.cpu_resources()
                (job / "cpu.max").write_text("150000 100000", encoding="ascii")
                quota_limited = system.cpu_resources()

        self.assertEqual(affinity_limited.affinity_count, 3)
        self.assertEqual(affinity_limited.cgroup_quota_cpus, 8.0)
        self.assertEqual(affinity_limited.worker_ceiling, 3)
        self.assertEqual(quota_limited.cgroup_quota_cpus, 1.5)
        self.assertEqual(quota_limited.effective_count, 1)
        self.assertEqual(quota_limited.worker_ceiling, 1)

    def test_nested_cgroup_uses_tightest_parent_cpu_and_memory_limits(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc_root, sys_root, cgroup_root = self._roots(root, "6.8.0-generic")
            (proc_root / "self").mkdir()
            (proc_root / "self/cgroup").write_text(
                "0::/jobs/demo/leaf\n", encoding="ascii"
            )
            parent = cgroup_root / "jobs/demo"
            leaf = parent / "leaf"
            leaf.mkdir(parents=True)
            (leaf / "cpu.max").write_text("max 100000", encoding="ascii")
            (parent / "cpu.max").write_text("150000 100000", encoding="ascii")
            (leaf / "cpuset.cpus.effective").write_text("0-7", encoding="ascii")
            (parent / "cpuset.cpus.effective").write_text("0-1", encoding="ascii")
            (leaf / "memory.max").write_text("max", encoding="ascii")
            (leaf / "memory.current").write_text(str(256 * MIB), encoding="ascii")
            (parent / "memory.max").write_text(str(2 * GIB), encoding="ascii")
            (parent / "memory.current").write_text(str(1536 * MIB), encoding="ascii")
            fake_psutil = _FakePsutil(
                total_memory=16 * GIB,
                available_memory=8 * GIB,
            )
            with (
                patch.object(system, "_PROC_ROOT", proc_root),
                patch.object(system, "_SYS_ROOT", sys_root),
                patch.object(system, "_CGROUP_ROOT", cgroup_root),
                patch.object(system, "_psutil", fake_psutil),
                patch.object(
                    system.os,
                    "sched_getaffinity",
                    return_value=set(range(16)),
                    create=True,
                ),
            ):
                cpu = system.cpu_resources()
                memory = system.memory_resources()

        self.assertEqual(cpu.cgroup_quota_cpus, 1.5)
        self.assertEqual(cpu.cgroup_cpuset_count, 2)
        self.assertEqual(cpu.worker_ceiling, 1)
        self.assertEqual(memory.cgroup_limit_bytes, 2 * GIB)
        self.assertEqual(memory.effective_available_bytes, 512 * MIB)

    def test_cgroup_memory_remainder_constrains_workers(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            proc_root, sys_root, cgroup_root = self._roots(root, "6.8.0-generic")
            (proc_root / "self").mkdir()
            (proc_root / "self/cgroup").write_text("0::/jobs/demo\n", encoding="ascii")
            job = cgroup_root / "jobs/demo"
            job.mkdir(parents=True)
            (job / "memory.max").write_text(str(GIB), encoding="ascii")
            (job / "memory.current").write_text(str(768 * MIB), encoding="ascii")
            fake_psutil = _FakePsutil(
                total_memory=16 * GIB,
                available_memory=8 * GIB,
            )
            with (
                patch.object(system, "_PROC_ROOT", proc_root),
                patch.object(system, "_SYS_ROOT", sys_root),
                patch.object(system, "_CGROUP_ROOT", cgroup_root),
                patch.object(system, "_psutil", fake_psutil),
            ):
                memory = system.memory_resources()

        self.assertEqual(memory.total_bytes, 16 * GIB)
        self.assertEqual(memory.available_bytes, 8 * GIB)
        self.assertEqual(memory.effective_total_bytes, GIB)
        self.assertEqual(memory.effective_available_bytes, 256 * MIB)

        snapshot = RuntimeResourceSnapshot(
            cpu=CPUResources(8, 16, 8, None, None, 8, 8),
            memory=memory,
            wsl=WSLInfo(False, None),
        )
        self.assertEqual(
            snapshot.worker_ceiling(
                128 * MIB,
                reserve_memory_bytes=128 * MIB,
            ),
            1,
        )

    def test_snapshot_profiles_all_roles_and_serializes(self) -> None:
        cpu = CPUResources(4, 8, 4, None, None, 4, 4, ("cpu:test",))
        memory = MemoryResources(
            8 * GIB,
            4 * GIB,
            None,
            None,
            8 * GIB,
            4 * GIB,
            ("memory:test",),
        )
        wsl = WSLInfo(False, None)

        def fake_storage(
            path: Path,
            override: system.StorageOverride,
            detected_wsl: WSLInfo,
        ) -> StorageProfile:
            self.assertIs(detected_wsl, wsl)
            return StorageProfile(
                path=path,
                device=f"device:{path.name}",
                rotational=False if override == "ssd" else None,
                filesystem="ext4",
                medium="ssd" if override == "ssd" else "unknown",
                confidence="high" if override == "ssd" else "unknown",
                evidence=(f"role_path:{path}",),
                override=override,
            )

        with (
            patch.object(system, "detect_wsl", return_value=wsl),
            patch.object(system, "cpu_resources", return_value=cpu),
            patch.object(system, "memory_resources", return_value=memory),
            patch.object(system, "_storage_profile", side_effect=fake_storage),
        ):
            snapshot = system.runtime_resource_snapshot(
                source=Path("/source"),
                temporary=Path("/temporary"),
                output=Path("/output"),
                storage_overrides={"output": "ssd"},
            )

        self.assertEqual(snapshot.source_storage.path, Path("/source"))
        self.assertEqual(snapshot.temporary_storage.path, Path("/temporary"))
        self.assertEqual(snapshot.output_storage.override, "ssd")
        self.assertEqual(snapshot.output_storage.medium, "ssd")
        payload = snapshot.to_dict()
        serialized = json.dumps(payload)
        self.assertEqual(payload["storage"]["source"]["path"], "<redacted>")
        self.assertNotIn("/source", serialized)

        legacy = StorageProfile(Path("/legacy"), "/dev/sdb", True, "ext4")
        self.assertIs(legacy.rotational, True)
        json.dumps(legacy.to_dict())


if __name__ == "__main__":
    unittest.main()
