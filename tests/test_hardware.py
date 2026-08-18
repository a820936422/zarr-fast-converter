from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from fast_nc_zarr.hardware import (  # noqa: E402
    HardwareProfile,
    StorageBenchmark,
    affinity_spec_from_profile,
    benchmark_storage_path,
    build_hardware_profile,
    detect_core_capacities,
    detect_numa_nodes,
    detect_performance_efficiency_cores,
    load_cached_profile,
    save_cached_profile,
    storage_initial_workers,
)


class HardwareProfileTests(unittest.TestCase):
    def test_storage_benchmark_roundtrip(self) -> None:
        benchmark = StorageBenchmark(
            medium="ssd",
            filesystem="ext4",
            seq_read_mib_s=100.0,
            seq_write_mib_s=80.0,
            rand_read_iops=5000.0,
            sample_mib=1,
            duration_seconds=0.1,
        )
        restored = StorageBenchmark.from_dict(benchmark.to_dict())
        self.assertEqual(restored, benchmark)

    def test_numa_detection_returns_none_or_tuple(self) -> None:
        nodes = detect_numa_nodes()
        if nodes is not None:
            for node in nodes:
                self.assertGreater(len(node), 0)

    def test_core_capacity_detection_returns_none_or_mapping(self) -> None:
        capacities = detect_core_capacities()
        if capacities is not None:
            self.assertGreater(len(capacities), 0)

    def test_pe_detection_returns_none_or_pair(self) -> None:
        result = detect_performance_efficiency_cores()
        if result is not None:
            performance, efficiency = result
            self.assertTrue(performance)
            self.assertTrue(efficiency)

    def test_profile_roundtrip_with_numa(self) -> None:
        profile = HardwareProfile(
            cpu_physical=6,
            cpu_logical=12,
            cpu_effective=12,
            worker_ceiling=6,
            memory_total_bytes=32 * 1024**3,
            memory_available_bytes=20 * 1024**3,
            storage={},
            numa_nodes=((0, 1, 2, 3), (4, 5, 6, 7)),
        )
        restored = HardwareProfile.from_dict(profile.to_dict())
        self.assertEqual(restored, profile)

    def test_profile_cache_roundtrip(self) -> None:
        profile = HardwareProfile(
            cpu_physical=6,
            cpu_logical=12,
            cpu_effective=12,
            worker_ceiling=6,
            memory_total_bytes=32 * 1024**3,
            memory_available_bytes=20 * 1024**3,
            storage={
                "source": StorageBenchmark(
                    "hdd",
                    "ext4",
                    100.0,
                    80.0,
                    100.0,
                    1,
                    0.1,
                )
            },
        )
        with tempfile.TemporaryDirectory() as tmp:
            paths = (Path(tmp) / "source",)
            paths[0].mkdir()
            save_cached_profile(paths, profile)
            loaded = load_cached_profile(paths)
        self.assertIsNotNone(loaded)
        assert loaded is not None
        self.assertEqual(loaded, profile)

    def test_benchmark_storage_path_runs_on_tmpdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            benchmark = benchmark_storage_path(
                Path(tmp),
                sample_mib=1,
                random_iops_samples=4,
            )
            if benchmark is None:
                self.skipTest("temporary directory is not writable")
            self.assertGreater(benchmark.seq_write_mib_s, 0)
            self.assertGreater(benchmark.seq_read_mib_s, 0)

    def test_build_hardware_profile_uses_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            first = build_hardware_profile(
                source=source,
                use_cache=True,
                sample_mib=1,
            )
            second = build_hardware_profile(
                source=source,
                use_cache=True,
                sample_mib=1,
            )
            self.assertEqual(first.to_dict(), second.to_dict())

    def test_affinity_spec_prefers_performance_cores(self) -> None:
        profile = HardwareProfile(
            cpu_physical=8,
            cpu_logical=16,
            cpu_effective=16,
            worker_ceiling=8,
            memory_total_bytes=32 * 1024**3,
            memory_available_bytes=20 * 1024**3,
            storage={},
            performance_cores=(0, 2, 4, 6, 8, 10, 12, 14),
            efficiency_cores=(1, 3, 5, 7, 9, 11, 13, 15),
        )
        self.assertEqual(affinity_spec_from_profile(profile), "0,2,4,6,8,10,12,14")

    def test_affinity_spec_falls_back_to_all_logical(self) -> None:
        profile = HardwareProfile(
            cpu_physical=6,
            cpu_logical=12,
            cpu_effective=12,
            worker_ceiling=6,
            memory_total_bytes=32 * 1024**3,
            memory_available_bytes=20 * 1024**3,
            storage={},
        )
        self.assertEqual(affinity_spec_from_profile(profile), "0-11")

    def test_storage_initial_workers_fast_hdd_raises_hint(self) -> None:
        profile = HardwareProfile(
            cpu_physical=6,
            cpu_logical=12,
            cpu_effective=12,
            worker_ceiling=12,
            memory_total_bytes=32 * 1024**3,
            memory_available_bytes=20 * 1024**3,
            storage={
                "output": StorageBenchmark(
                    "hdd", "ext4", 180.0, 200.0, 120.0, 64, 1.0
                )
            },
        )
        self.assertEqual(
            storage_initial_workers(profile, "large-files", same_device=True),
            8,
        )

    def test_storage_initial_workers_slow_hdd_stays_conservative(self) -> None:
        profile = HardwareProfile(
            cpu_physical=6,
            cpu_logical=12,
            cpu_effective=12,
            worker_ceiling=12,
            memory_total_bytes=32 * 1024**3,
            memory_available_bytes=20 * 1024**3,
            storage={
                "output": StorageBenchmark(
                    "hdd", "ext4", 80.0, 70.0, 100.0, 64, 1.0
                )
            },
        )
        self.assertEqual(
            storage_initial_workers(profile, "large-files", same_device=True),
            4,
        )

    def test_storage_initial_workers_ssd_uses_ceiling(self) -> None:
        profile = HardwareProfile(
            cpu_physical=6,
            cpu_logical=12,
            cpu_effective=12,
            worker_ceiling=12,
            memory_total_bytes=32 * 1024**3,
            memory_available_bytes=20 * 1024**3,
            storage={
                "output": StorageBenchmark(
                    "ssd", "ext4", 1800.0, 1500.0, 40000.0, 64, 1.0
                )
            },
        )
        self.assertEqual(
            storage_initial_workers(profile, "large-files", same_device=False),
            12,
        )


if __name__ == "__main__":
    unittest.main()
