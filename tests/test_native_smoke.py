from __future__ import annotations

import json
import unittest

from fast_nc_zarr._backend import resolve_backend, rust_capability


class NativePreparationTests(unittest.TestCase):
    def test_capability_probe_is_json_safe(self) -> None:
        capability = rust_capability()
        self.assertIn(capability.name, {"rust", "python"})
        self.assertGreaterEqual(capability.protocol_version, 0)
        self.assertIsInstance(capability.operations, tuple)
        json.dumps(
            {
                "name": capability.name,
                "protocol_version": capability.protocol_version,
                "operations": capability.operations,
                "supported": capability.supported,
            }
        )

    def test_auto_backend_falls_back_without_rust_operation(self) -> None:
        self.assertEqual(resolve_backend("auto", "rechunk"), "python")

    def test_python_backend_is_always_selectable(self) -> None:
        self.assertEqual(resolve_backend("python", "rechunk"), "python")


if __name__ == "__main__":
    unittest.main()
