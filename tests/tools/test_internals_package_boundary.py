"""Core consumes one selected Internals release without reinterpreting it."""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def cmake() -> str:
    return (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")


def build_script() -> str:
    return (ROOT / "build.sh").read_text(encoding="utf-8")


def installer() -> str:
    return (ROOT / "tools/install_neat_framework.sh").read_text(encoding="utf-8")


class InternalsPackageBoundaryTest(unittest.TestCase):
    def test_internals_is_located_without_a_derived_version(self) -> None:
        """platform-version selects an environment, never an Internals release."""
        text = cmake()
        self.assertIn("find_package(NeatInternals CONFIG REQUIRED)", text)
        self.assertNotIn(
            "find_package(NeatInternals ${SIMANEAT_PLATFORM_VERSION}", text
        )

    def test_no_manually_constructed_internals_version_ranges(self) -> None:
        text = cmake()
        for removed in (
            "neat-runtime (>=",
            "neat-runtime (<<",
            "neat-gst-plugins (>=",
            "neat-gst-plugins (<<",
            "neat-internals-dev (>=",
            "neat-internals-dev (<<",
        ):
            self.assertNotIn(removed, text, removed)
        for kept in ('"neat-runtime"', '"neat-gst-plugins"', '"neat-internals-dev"'):
            self.assertIn(kept, text, kept)

    def test_no_configure_time_bundled_memory_requirement(self) -> None:
        """Memory comes from the platform, not from the Internals artifact."""
        text = cmake()
        self.assertNotIn("SIMANEAT_MEMORY_DEV_PACKAGE_VERSION", text)
        self.assertNotIn("SIMANEAT_BUNDLED_MEMORY_DEV_DEBS", text)
        self.assertNotIn("simaai-memory-lib-dev (=", text)
        # It is still named so downstream C++ apps can link on a DevKit.
        self.assertIn('"simaai-memory-lib-dev"', text)

    def test_every_delivered_internals_package_is_forwarded(self) -> None:
        text = build_script()
        self.assertIn('for file in "${NEAT_INTERNALS_DEB_DIR}"/*.deb; do', text)
        # No filename whitelist decides what reaches the artifact.
        for removed in (
            '"${NEAT_INTERNALS_DEB_DIR}"/neat-*.deb',
            "dist/simaai-common*.deb",
            "'simaai-memory-lib_*.deb'",
            "'libcamera_*.deb'",
            "'neat-runtime_*.deb'",
        ):
            self.assertNotIn(removed, text, removed)

    def test_installer_has_no_bundled_memory_transaction(self) -> None:
        text = installer()
        for removed in (
            "SIMAAI_MEMORY",
            "install_local_simaai_memory_transaction",
            "collect_local_simaai_memory_debs",
            "palette_required_simaai_memory_version",
            "verify_memory_transaction_preservation",
        ):
            self.assertNotIn(removed, text, removed)

    def test_installer_keeps_no_internals_package_filename_list(self) -> None:
        text = installer()
        for removed in (
            "'simaai-memory-lib_*.deb'",
            "'libcamera-tools_*.deb'",
            "'neat-common_*.deb'",
            "'neat-internals-dev_*.deb'",
        ):
            self.assertNotIn(removed, text, removed)
        # Core's own artifacts still install last.
        self.assertIn("'sima-neat-*-Linux-core.deb'", text)

    def test_installer_still_repairs_platform_packages_from_the_platform(self) -> None:
        """Memory/libcamera are repaired from the platform repository."""
        text = installer()
        self.assertIn(
            "for package in libcamera libcamera-tools simaai-memory-lib; do", text
        )


if __name__ == "__main__":
    unittest.main()
