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
    benchmark_storage_path,
    build_hardware_profile,
    detect_numa_nodes,
    load_cached_profile,
    save_cached_profile,
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


if __name__ == "__main__":
    unittest.main()
