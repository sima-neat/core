from __future__ import annotations

import json
import hashlib
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






class InternalsPackageBoundaryTest(unittest.TestCase):
    def test_scheduler_frontend_is_located_without_a_derived_version(self) -> None:
        text = cmake()
        self.assertIn("find_package(SimaScheduler CONFIG REQUIRED COMPONENTS frontend)", text)
        exported = (ROOT / "cmake/SimaNeatConfig.cmake.in").read_text()
        self.assertIn("find_dependency(SimaScheduler CONFIG COMPONENTS frontend)", exported)
        for contents in (text, exported):
            self.assertNotIn("SimaScheduler ${SIMANEAT_PLATFORM_VERSION}", contents)
            for lower in ("SimaRunner", "SimaKernels", "SimaCodecs", "NeatInternals"):
                self.assertNotIn("find_package(" + lower, contents)
                self.assertNotIn("find_dependency(" + lower, contents)


    def test_internals_abi_headers_have_one_package_owner(self) -> None:
        text = cmake()
        start = text.index("  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/")
        install_headers = text[start : text.index("\n)", start)]
        for header in (
            "ProcessMlaRuntimeConfig.h",
            "SimaPluginStaticManifestAbi.h",
            "SimaPreparedRuntimeAbi.h",
            "SimaTensorSetMetaAbi.h",
        ):
            with self.subTest(header=header):
                self.assertTrue((ROOT / "include/gst" / header).is_file())
                self.assertIn(f'PATTERN "{header}" EXCLUDE', install_headers)
        self.assertFalse((ROOT / "include/gst/SimaaiGstErrors.h").exists())
        self.assertIn('PATTERN "SimaaiGstErrors.h" EXCLUDE', install_headers)
        manifest = json.loads((ROOT / "deps/manifest.json").read_text())
        subprocess.run(["dpkg", "--compare-versions", manifest["package-version"],
                        "ge", "0.6.0"], check=True)
        self.assertFalse((ROOT / "include/gst/SimaCvuCapabilityAbi.h").exists())
        self.assertIn(
            "#include <simaai/runtime/gstreamer/abi/SimaPluginStaticManifestAbi.h>",
            (ROOT / "include/gst/SimaPluginStaticManifestAbi.h").read_text(),
        )
        self.assertNotIn('"neat-internals-dev"', text)

    def test_client_headers_have_no_adhoc_sdk_fallback(self) -> None:
        # The old empty-cache workaround belonged to removed Core header
        # discovery. Producer frontend usage now owns every support include.
        text = cmake()
        self.assertIn("find_package(SimaScheduler CONFIG REQUIRED COMPONENTS frontend)", text)
        for removed in ("find_path(SIMANEAT_NEAT_", "find_library(SIMANEAT_NEAT_",
                        "../internals/include", "../internals/gst_plugins/include",
                        "NeatInternals::"):
            self.assertNotIn(removed, text)


    def test_actual_header_install_leaves_scheduler_contracts_to_scheduler(self) -> None:
        text = cmake()
        start = text.rfind("install(", 0, text.index("DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/"))
        block = text[start:text.index("\n)", start) + 3]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.18)\nproject(HeaderOwnership NONE)\n"
                'set(CMAKE_INSTALL_INCLUDEDIR "include")\n'
                f'set(CMAKE_CURRENT_SOURCE_DIR "{ROOT}")\n' + block)
            subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build")],
                           check=True, capture_output=True)
            subprocess.run(["cmake", "--install", str(root / "build"), "--prefix",
                            str(root / "installed"), "--component", "dev"],
                           check=True, capture_output=True)
            installed = root / "installed/include"
            self.assertTrue((installed / "neat.h").is_file())
            for name in ("SimaaiGstErrors.h", "SimaTensorSetMetaAbi.h", "SimaPreparedRuntimeAbi.h"):
                self.assertFalse((installed / "gst" / name).exists(), name)

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
        for kept in ('"simaai-scheduler"',):
            self.assertIn(kept, text, kept)
        self.assertNotIn('"simaai-kernels-dev"', text)

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

    def test_explicit_llima_artifact_refreshes_packaging_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact"
            cache = root / "cache"
            deps = root / "deps"
            sysroot = root / "sysroot"
            artifact.mkdir()
            cache.mkdir()
            deps.mkdir()
            for component in ("core", "dev", "cli"):
                (artifact / f"sima-lmm-0.4.0+local-Linux-{component}.deb").touch()
                (cache / f"sima-lmm-0.4.0+stale-Linux-{component}.deb").touch()

            policy = {'platform-version': '3.0.0', 'sdk': {'channel': 'develop'},
                      'elxr_version': '3.0.0_master_B2372', 'sysroot-version': '3.0.0~pre2372'}
            (artifact / 'llima-package-manifest.json').write_text(json.dumps(policy))
            (artifact / 'metadata.json').write_text(json.dumps({'resources-checksum': {
                path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in artifact.iterdir()}}))

            script = f"""
set -euo pipefail
REPO_ROOT={shlex.quote(str(ROOT))}
python3() {{
  if [[ "$1" == tools/build_dependencies.py && "$2" == sdk-policy ]]; then
    printf '%s\\n' {shlex.quote(json.dumps(policy))}
  else
    command python3 "$@"
  fi
}}
resolve_neat_llima_ref() {{
  NEAT_LLIMA_REQUESTED_REF=local:latest
}}
ensure_llima_sdk_sysroot_deps() {{ :; }}
modify_sysroot() {{ "$@"; }}
dpkg-deb() {{
  mkdir -p "$3/usr/lib/aarch64-linux-gnu/cmake/SimaLMM"
  touch "$3/usr/lib/aarch64-linux-gnu/cmake/SimaLMM/SimaLMMConfig.cmake"
  touch "$3/usr/lib/aarch64-linux-gnu/libsima_lmm_runtime.so"
}}
{shell_function("ensure_neat_llima")}
NEAT_LLIMA_ARTIFACT_DIR={shlex.quote(str(artifact))}
NEAT_LLIMA_DEB_DIR={shlex.quote(str(cache))}
NEAT_DEPS_DIR={shlex.quote(str(deps))}
SYSROOT={shlex.quote(str(sysroot))}
NEAT_SYSROOT_LLIMA_RECORD=var/lib/sima-neat/llima.sha256
ELXR_SDK=ON
ensure_neat_llima
"""
            result = subprocess.run(
                ["bash", "-c", script], check=False, text=True, capture_output=True
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                sorted(path.name for path in cache.glob("sima-lmm-*.deb")),
                [
                    "sima-lmm-0.4.0+local-Linux-cli.deb",
                    "sima-lmm-0.4.0+local-Linux-core.deb",
                    "sima-lmm-0.4.0+local-Linux-dev.deb",
                ],
            )

    def test_cmake_package_reports_core_release_identity(self) -> None:
        text = cmake()
        self.assertIn("VERSION ${SIMANEAT_PACKAGE_BASE_VERSION}", text)
        self.assertNotIn("VERSION ${SIMANEAT_PLATFORM_VERSION}", text)

    def test_manifest_tracks_only_immediate_execution_dependencies(self) -> None:
        manifest = json.loads((ROOT / "deps/manifest.json").read_text())
        self.assertIn("scheduler", manifest)
        self.assertIn("llima", manifest)  # GenAI intentionally bypasses vision execution.
        for retired_or_transitive in ("internals", "runner", "kernels", "codecs"):
            self.assertNotIn(retired_or_transitive, manifest)

    def test_manifest_has_no_platform_package_pin(self) -> None:
        manifest = json.loads((ROOT / "deps/manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("platform-package-version", manifest)

    def test_no_configure_time_bundled_memory_requirement(self) -> None:
        text = cmake()
        self.assertNotIn("SIMANEAT_MEMORY_DEV_PACKAGE_VERSION", text)
        self.assertNotIn("SIMANEAT_BUNDLED_MEMORY_DEV_DEBS", text)
        self.assertNotIn("simaai-memory-lib-dev (=", text)
        self.assertNotIn('"simaai-memory-lib-dev"', text)

    def test_sdk_build_uses_the_selected_cross_compiler_headers(self) -> None:
        text = build_script()
        self.assertIn('${CXX:-aarch64-linux-gnu-g++}', text)
        self.assertIn('for query in ("-dumpversion", "-dumpmachine"):', text)
        self.assertIn('subprocess.check_output([*compiler, query]', text)
        self.assertIn('-nostdinc++${cpp_header_flags}', text)
        self.assertIn("usr/include/c++/${cross_gcc_major}", text)
        self.assertNotIn("usr/include/c++/12", text)

    def test_customer_package_selection_preserves_runtime_and_public_dev(self) -> None:
        for package in ("neat-internals-dev", "simaai-memory-lib-dev", "sima-lmm-dev",
                        "sima-neat-dev", "neat-runtime", "neat-gst-plugins",
                        "neat-ev74-firmware", "simaai-memory-lib", "sima-lmm-core", "sima-lmm-cli"):
            with self.subTest(package=package):
                result = subprocess.run(
                    ["bash", "-c", f'dpkg-deb() {{ echo {shlex.quote(package)}; }}\n'
                     + shell_function("is_customer_deb") + '\nis_customer_deb unused.deb'],
                    text=True, capture_output=True,
                )
                self.assertEqual(result.returncode, int(package in {
                    "neat-internals-dev", "simaai-memory-lib-dev", "sima-lmm-dev"
                }), result.stderr)


    def test_shared_package_discovery_is_scoped_and_hides_implementation_sdks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "FindPkgConfig.cmake").write_text("""
set(PkgConfig_FOUND TRUE)
function(pkg_check_modules prefix)
  set(${prefix}_INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/public" PARENT_SCOPE)
  set(${prefix}_LIBRARY_DIRS "${CMAKE_SOURCE_DIR}/lib" PARENT_SCOPE)
endfunction()
""")
            (root / "SimaSchedulerConfig.cmake").write_text("""
foreach(component frontend)
  if(NOT TARGET SimaScheduler::${component})
    add_library(SimaScheduler::${component} INTERFACE IMPORTED)
  endif()
endforeach()
""")
            (root / "SimaRunnerConfig.cmake").write_text("set(SimaRunner_FOUND TRUE)\n")
            (root / "nlohmann_jsonConfig.cmake").write_text("set(nlohmann_json_FOUND TRUE)\n")
            kernels_config = root / "SimaKernelsConfig.cmake"
            kernels_config.write_text("""
if(NOT TARGET SimaKernels::executables)
  add_library(SimaKernels::executables INTERFACE IMPORTED)
endif()
""")
            (root / "SimaKernelsConfigVersion.cmake").write_text("""
set(PACKAGE_VERSION "0.1.0")
if(PACKAGE_FIND_VERSION VERSION_EQUAL PACKAGE_VERSION)
  set(PACKAGE_VERSION_EXACT TRUE)
  set(PACKAGE_VERSION_COMPATIBLE TRUE)
endif()
""")
            (root / "SimaNeatTargets.cmake").write_text("""
add_library(SimaNeat::sima_neat INTERFACE IMPORTED)
add_library(SimaNeat::sima_neat_common INTERFACE IMPORTED)
""")
            (root / "CMakeLists.txt").write_text(f"""
cmake_minimum_required(VERSION 3.18)
project(CustomerPackage NONE)
list(PREPEND CMAKE_MODULE_PATH "${{CMAKE_SOURCE_DIR}}")
list(PREPEND CMAKE_PREFIX_PATH "${{CMAKE_SOURCE_DIR}}")
set(CMAKE_DISABLE_FIND_PACKAGE_NeatInternals TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_SimaRunner TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_SimaKernels TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_SimaCodecs TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_SimaLMM TRUE)
set(CMAKE_DISABLE_FIND_PACKAGE_ZLIB TRUE)
set(SIMANEAT_CONFIG_WITH_LLIMA TRUE)
include(CMakePackageConfigHelpers)
configure_package_config_file("{ROOT}/cmake/SimaNeatConfig.cmake.in"
  "${{CMAKE_SOURCE_DIR}}/SimaNeatConfig.cmake" INSTALL_DESTINATION .)
add_subdirectory(first)
add_subdirectory(second)
""")
            for name in ("first", "second"):
                (root / name).mkdir()
                (root / name / "CMakeLists.txt").write_text("""
find_package(SimaNeat CONFIG REQUIRED)
find_package(SimaNeat CONFIG REQUIRED)
get_target_property(includes SimaNeat::sima_neat_common INTERFACE_INCLUDE_DIRECTORIES)
if (NOT "${CMAKE_SOURCE_DIR}/public" IN_LIST includes)
  message(FATAL_ERROR "Missing public dependency includes")
endif()
if (TARGET SimaNeat::sima_neat_static)
  message(FATAL_ERROR "Static target leaked into shared discovery")
endif()
""")
            result = subprocess.run(["cmake", "-S", str(root), "-B", str(root / "build")],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            runtime_config = root / "SimaSchedulerConfig.cmake"
            saved = runtime_config.read_text()
            runtime_config.write_text(saved.replace("foreach(component frontend)", "foreach(component missing)"))
            missing = subprocess.run(["cmake", "-S", str(root), "-B", str(root / "missing-frontend")],
                                     text=True, capture_output=True)
            self.assertNotEqual(missing.returncode, 0)
            self.assertIn("SimaScheduler::frontend", missing.stdout + missing.stderr)
            runtime_config.write_text(saved)
            (root / "FindZLIB.cmake").write_text("set(ZLIB_FOUND TRUE)\n")
            (root / "SimaNeatStaticTargets.cmake").write_text("add_library(SimaNeat::sima_neat_static INTERFACE IMPORTED)\n")
            top = root / "CMakeLists.txt"
            top.write_text(top.read_text().replace("set(CMAKE_DISABLE_FIND_PACKAGE_ZLIB TRUE)", "")
                           .replace("set(SIMANEAT_CONFIG_WITH_LLIMA TRUE)", "set(SIMANEAT_CONFIG_WITH_LLIMA FALSE)"))
            child = root / "first/CMakeLists.txt"
            child.write_text(child.read_text() + "\nfind_package(SimaNeat CONFIG REQUIRED COMPONENTS Static)\n"
                             "if(NOT TARGET SimaNeat::sima_neat_static)\nmessage(FATAL_ERROR \"Missing explicit static target\")\nendif()\n")
            static = subprocess.run(["cmake", "-S", str(root), "-B", str(root / "static-no-lower-dev")],
                                    text=True, capture_output=True)
            self.assertEqual(static.returncode, 0, static.stdout + static.stderr)









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
