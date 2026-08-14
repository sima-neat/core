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
        text = cmake()
        self.assertNotIn("SIMANEAT_MEMORY_DEV_PACKAGE_VERSION", text)
        self.assertNotIn("SIMANEAT_BUNDLED_MEMORY_DEV_DEBS", text)
        self.assertNotIn("simaai-memory-lib-dev (=", text)
        self.assertIn('"simaai-memory-lib-dev"', text)

    def test_every_delivered_internals_package_is_forwarded(self) -> None:
        text = build_script()
        self.assertIn('for file in "${NEAT_INTERNALS_DEB_DIR}"/*.deb; do', text)
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

    def test_installer_consumes_every_manifest_deb(self) -> None:
        text = installer()
        start = text.index("collect_debs_in_install_order()")
        end = text.index("\nsysroot_path()", start)
        function = text[start:end]
        self.assertIn(
            'append_matching_files "${out_array_name}" "${search_dir}" \'*.deb\'',
            function,
        )
        for package in ("neat-runtime", "libcamera", "sima-lmm", "sima-neat"):
            self.assertNotIn(package, function)

    def test_installer_does_not_resolve_platform_package_versions(self) -> None:
        text = installer()
        for removed in (
            "apt_candidate_version",
            "apt_exact_dependency_version",
            "native_modalix_restore_specs",
            "native_modalix_repair_is_required",
        ):
            self.assertNotIn(removed, text)


if __name__ == "__main__":
    unittest.main()
