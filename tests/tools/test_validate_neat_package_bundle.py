"""Unit tests for the host-side B4593 Debian bundle validator."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "validate_neat_package_bundle.py"
SPEC = importlib.util.spec_from_file_location(
    "validate_neat_package_bundle", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ValidateNeatPackageBundleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.versions = {
            "libcamera": "2.1.3+neat2",
            "libcamera-dev": "2.1.3+neat2",
            "libcamera-tools": "2.1.3+neat2",
            "simaai-memory-lib": "2.1.1-0neat5",
            "simaai-memory-lib-dev": "2.1.1-0neat5",
        }
        self.architectures = {package: "arm64" for package in self.versions}
        self.dependencies = {
            "libcamera": "libc6",
            "libcamera-dev": "libcamera (= 2.1.3+neat2)",
            "libcamera-tools": "libcamera (= 2.1.3+neat2)",
            "simaai-memory-lib": "libc6",
            "simaai-memory-lib-dev": "simaai-memory-lib (= 2.1.1-0neat5)",
        }
        self.provides = {
            "libcamera": (
                "libcamera (= 2.1.3~pre4593), simaai-libcamera-dmabuf-abi (= 1)"
            ),
            "libcamera-dev": "libcamera-dev (= 2.1.3~pre4593)",
            "libcamera-tools": "libcamera-tools (= 2.1.3~pre4593)",
            "simaai-memory-lib": (
                "simaai-memory-lib (= 2.1.1~pre4593), "
                "simaai-memory-dmabuf-export-abi (= 1)"
            ),
            "simaai-memory-lib-dev": ("simaai-memory-lib-dev (= 2.1.1~pre4593)"),
        }

    def validate(self) -> int:
        return MODULE.validate_platform_overrides(
            ROOT / "deps" / "manifest.json",
            self.versions,
            self.architectures,
            self.dependencies,
            self.provides,
        )

    def test_accepts_reviewed_b4593_override_closure(self) -> None:
        self.assertGreater(self.validate(), 0)

    def test_rejects_missing_libcamera_capability(self) -> None:
        self.provides["libcamera"] = "libcamera (= 2.1.3~pre4593)"
        with self.assertRaisesRegex(SystemExit, "simaai-libcamera-dmabuf-abi"):
            self.validate()

    def test_rejects_wrong_memory_revision(self) -> None:
        self.versions["simaai-memory-lib"] = "2.1.1-0neat3"
        with self.assertRaisesRegex(SystemExit, "Wrong B4593 override version"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
