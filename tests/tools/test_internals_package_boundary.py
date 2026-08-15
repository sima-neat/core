from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def cmake() -> str:
    return (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")


def build_script() -> str:
    return (ROOT / "build.sh").read_text(encoding="utf-8")


def installer() -> str:
    return (ROOT / "tools/install_neat_framework.sh").read_text(encoding="utf-8")


def shell_function(name: str) -> str:
    text = build_script()
    start = text.index(f"{name}() {{")
    return text[start : text.index("\n}\n", start) + 2]


def run_sync(
    artifact: dict[str, str] | str | None,
    consumer_base: str = "2.1.3",
    enabled: str = "ON",
    update_status: int = 0,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact_dir = root / "artifact"
        artifact_dir.mkdir()
        if artifact is not None:
            contents = json.dumps(artifact) if isinstance(artifact, dict) else artifact
            (artifact_dir / "internals-manifest.json").write_text(
                contents, encoding="utf-8"
            )
        consumer = root / "manifest.json"
        consumer.write_text(
            json.dumps({"platform-version": consumer_base}), encoding="utf-8"
        )
        log = root / "sysroot.log"
        script = f"""
set -e
id() {{ echo 0; }}
sysroot() {{
  printf '%s\n' "$*" >> {shlex.quote(str(log))}
  [[ "$1" != update ]] || return {update_status}
}}
{shell_function("run_privileged")}
{shell_function("sync_sysroot_from_internals_manifest")}
ELXR_SDK=ON
NEAT_SYNC_SYSROOT={shlex.quote(enabled)}
NEAT_DEPS_MANIFEST={shlex.quote(str(consumer))}
sync_sysroot_from_internals_manifest {shlex.quote(str(artifact_dir))}
"""
        result = subprocess.run(
            ["bash", "-c", script], check=False, text=True, capture_output=True
        )
        calls = log.read_text(encoding="utf-8").splitlines() if log.exists() else []
        return result, calls


class InternalsPackageBoundaryTest(unittest.TestCase):
    def test_internals_is_located_without_a_derived_version(self) -> None:
        text = cmake()
        self.assertIn("find_package(NeatInternals CONFIG REQUIRED)", text)
        self.assertNotIn(
            "find_package(NeatInternals ${SIMANEAT_PLATFORM_VERSION}", text
        )
        exported_config = (ROOT / "cmake/SimaNeatConfig.cmake.in").read_text(
            encoding="utf-8"
        )
        self.assertIn("find_dependency(NeatInternals CONFIG REQUIRED)", exported_config)
        self.assertNotIn(
            "find_dependency(NeatInternals @SIMANEAT_PLATFORM_VERSION@",
            exported_config,
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

    def test_llima_is_consumed_without_a_version_policy(self) -> None:
        text = cmake()
        self.assertIn("find_package(SimaLMM CONFIG REQUIRED)", text)
        self.assertIn("find_package(SimaLMM CONFIG QUIET)", text)
        for removed in (
            "find_package(SimaLMM ${SIMANEAT_PLATFORM_VERSION}",
            "sima-lmm-core (>=",
            "sima-lmm-core (<<",
            "sima-lmm-dev (>=",
            "sima-lmm-dev (<<",
            "SIMANEAT_DEP_PACKAGE_MIN_VERSION",
            "SIMANEAT_DEP_PACKAGE_MAX_VERSION",
        ):
            self.assertNotIn(removed, text, removed)
        exported_config = (ROOT / "cmake/SimaNeatConfig.cmake.in").read_text(
            encoding="utf-8"
        )
        self.assertIn("find_dependency(SimaLMM CONFIG REQUIRED)", exported_config)
        self.assertNotIn("@SIMANEAT_PLATFORM_VERSION@", exported_config)

    def test_cmake_package_reports_core_release_identity(self) -> None:
        text = cmake()
        self.assertIn("VERSION ${SIMANEAT_PACKAGE_BASE_VERSION}", text)
        self.assertNotIn("VERSION ${SIMANEAT_PLATFORM_VERSION}", text)

    def test_manifest_has_no_platform_package_pin(self) -> None:
        manifest = json.loads((ROOT / "deps/manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("platform-package-version", manifest)

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

    def test_selected_internals_artifact_is_installed_before_build(self) -> None:
        text = build_script()
        start = text.index("ensure_neat_internals()")
        end = text.index("\nensure_neat_llima()", start)
        function = text[start:end]
        self.assertNotIn("return 0", function[: function.index("fetch_neat_internals")])

    def test_vulcan_build_uses_the_internals_sysroot_receipt(self) -> None:
        text = build_script()
        workflow = (ROOT / ".github/workflows/vulcan-ci.yml").read_text(
            encoding="utf-8"
        )
        manifest = json.loads((ROOT / "deps/manifest.json").read_text(encoding="utf-8"))
        sync = text.index('sync_sysroot_from_internals_manifest "${artifact_dir}"')
        install = text.index('collect_plugin_files_from_debs "${artifact_dir}"', sync)
        ensure = text.index("ensure_neat_internals\n", text.index("main()"))
        target_python = text.index("detect_elxr_target_python\n", ensure)

        self.assertIn('[[ "${NEAT_SYNC_SYSROOT:-OFF}" == "ON" ]] || return 0', text)
        self.assertIn('-e NEAT_SYNC_SYSROOT="ON"', workflow)
        self.assertIn("internals-manifest.json", text)
        self.assertIn('sysroot update "${receipt}"', text)
        self.assertIn("Internals artifact is missing internals-manifest.json", text)
        self.assertIn("invalid sysroot-version", text)
        self.assertIn("platform-version does not match the Internals receipt", text)
        self.assertNotIn("sysroot-version", manifest)
        self.assertNotRegex(text, r"\b[0-9]+(?:\.[0-9]+){2}~pre[0-9]+\b")
        self.assertNotRegex(workflow, r"\b[0-9]+(?:\.[0-9]+){2}~pre[0-9]+\b")
        self.assertLess(sync, install)
        self.assertLess(ensure, target_python)

    def test_sysroot_receipt_behavior(self) -> None:
        base = "2.1.3"
        receipt = f"{base}~pre9999"
        result, calls = run_sync({"sysroot-version": receipt}, base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [f"update {receipt}", "status"])

        result, calls = run_sync(None, enabled="OFF")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

        result, calls = run_sync({"sysroot-version": ""}, base)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])

    def test_invalid_sysroot_receipts_fail_closed(self) -> None:
        base = "2.1.3"
        receipt = f"{base}~pre9999"
        cases = (
            (None, base, "missing internals-manifest.json"),
            ("{", base, "Cannot read Internals build receipt"),
            ({}, base, "Cannot read Internals build receipt"),
            (
                {"sysroot-version": "latest"},
                base,
                "Cannot read Internals build receipt",
            ),
            (
                {"sysroot-version": "|latest"},
                base,
                "Cannot read Internals build receipt",
            ),
            (
                {"sysroot-version": "\nmalformed"},
                base,
                "Cannot read Internals build receipt",
            ),
            (
                {"sysroot-version": receipt},
                "2.1.4",
                "Cannot read Internals build receipt",
            ),
        )
        for artifact, consumer_base, message in cases:
            with self.subTest(message=message):
                result, calls = run_sync(artifact, consumer_base)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertEqual(calls, [])

    def test_failed_sysroot_update_is_not_masked(self) -> None:
        base = "2.1.3"
        result, calls = run_sync(
            {"sysroot-version": f"{base}~pre9999"}, base, update_status=23
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Failed to update SDK sysroot", result.stderr)
        self.assertEqual(calls, [f"update {base}~pre9999"])

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

    def test_installer_does_not_rewrite_llima_dependencies(self) -> None:
        text = installer()
        for removed in (
            "NEAT_INSTALLER_RELAX_SIMA_LMM_DEP",
            "maybe_relax_sima_lmm_dep",
            "prepare_debs_for_board_install",
        ):
            self.assertNotIn(removed, text)

    def test_no_bundle_version_resolver(self) -> None:
        self.assertNotIn("validate_neat_package_bundle.py", build_script())
        self.assertFalse((ROOT / "tools/validate_neat_package_bundle.py").exists())


if __name__ == "__main__":
    unittest.main()
