"""Installed-export layout and rejection regressions; no target execution."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

CHECKER = Path(__file__).with_name("check_cmake_export.py")
spec = importlib.util.spec_from_file_location("check_cmake_export", CHECKER)
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class CMakeExportTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="cmake-export-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prefix = self.root / "package"

    def export(self, relative="usr/lib/cmake/SimaNeat"):
        directory = self.prefix / relative
        directory.mkdir(parents=True)
        (directory / "SimaNeatTargets.cmake").write_text("# installed targets\n")
        (directory / "SimaNeatTargets-release.cmake").write_text(
            'IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsima_neat.so"\n'
            'IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsima_neat.a"\n'
        )
        (directory / "SimaNeatConfig.cmake").write_text("set(SIMANEAT_WITH_LLIMA FALSE)\n")
        return directory

    def test_supported_layouts(self):
        for relative in ("usr/lib", "usr/lib64", "usr/lib/aarch64-linux-gnu",
                         "usr/lib/x86_64-linux-gnu", "opt/neat/custom/libraries"):
            with self.subTest(relative=relative):
                self.prefix = self.root / relative.replace("/", "-")
                self.export(relative + "/cmake/SimaNeat")
                self.assertEqual(checker.check(self.prefix, []), 2)

    def test_missing_export(self):
        with self.assertRaisesRegex(ValueError, "missing installed"):
            checker.check(self.prefix, [])

    def test_missing_config(self):
        directory = self.export()
        (directory / "SimaNeatConfig.cmake").unlink()
        with self.assertRaisesRegex(ValueError, "missing installed.*Config"):
            checker.check(self.prefix, [])

    def test_duplicate_exports_rejected_even_if_one_is_incomplete(self):
        self.export()
        second = self.export("usr/lib64/cmake/SimaNeat")
        (second / "SimaNeatConfig.cmake").unlink()
        with self.assertRaisesRegex(ValueError, "multiple installed"):
            checker.check(self.prefix, [])

    def test_build_tree_is_not_an_installed_export(self):
        self.export("build")
        with self.assertRaisesRegex(ValueError, "missing installed"):
            checker.check(self.prefix, [])

    def test_absolute_import_still_rejected(self):
        directory = self.export("usr/lib/aarch64-linux-gnu/cmake/SimaNeat")
        (directory / "SimaNeatTargets-release.cmake").write_text(
            'IMPORTED_LOCATION_RELEASE "/builder/libsima_neat.so"\n')
        with self.assertRaisesRegex(ValueError, "non-relocatable"):
            checker.check(self.prefix, [])

    def test_forbidden_path_in_config_still_rejected(self):
        directory = self.export("usr/lib64/cmake/SimaNeat")
        (directory / "SimaNeatConfig.cmake").write_text("# /builder/private\n")
        with self.assertRaisesRegex(ValueError, "build-host path leaked"):
            checker.check(self.prefix, ["/builder/"])

    def test_incomplete_library_locations_still_rejected(self):
        directory = self.export()
        (directory / "SimaNeatTargets-release.cmake").write_text(
            'IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libsima_neat.so"\n')
        with self.assertRaisesRegex(ValueError, "shared and static"):
            checker.check(self.prefix, [])

    def test_llima_checks_still_required(self):
        directory = self.export("usr/lib64/cmake/SimaNeat")
        config = directory / "SimaNeatConfig.cmake"
        config.write_text("set(SIMANEAT_WITH_LLIMA TRUE)\n")
        with self.assertRaisesRegex(ValueError, "consumer-resolved"):
            checker.check(self.prefix, [])
        (directory / "SimaNeatTargets.cmake").write_text("PkgConfig::SIMANEAT_HTTPLIB\n")
        with self.assertRaisesRegex(ValueError, "not rediscovered"):
            checker.check(self.prefix, [])
        config.write_text(config.read_text() +
                          "pkg_check_modules(SIMANEAT_HTTPLIB REQUIRED IMPORTED_TARGET cpp-httplib)\n")
        self.assertEqual(checker.check(self.prefix, []), 2)

    def test_real_cmake_installed_exports(self):
        # Use real install(EXPORT) output, not only hand-written regex fixtures.
        # Install the dev component only: neither compile nor execute target code.
        for tool in ("cmake", "cc"):
            self.assertIsNotNone(shutil.which(tool), f"required test tool missing: {tool}")
        source = self.root / "source"
        source.mkdir()
        (source / "fixture.c").write_text("int fixture(void) { return 0; }\n")
        (source / "SimaNeatConfig.cmake").write_text("set(SIMANEAT_WITH_LLIMA FALSE)\n")
        (source / "CMakeLists.txt").write_text("""
cmake_minimum_required(VERSION 3.16)
project(ExportFixture LANGUAGES C)
include(GNUInstallDirs)
add_library(sima_neat SHARED fixture.c)
add_library(sima_neat_static STATIC fixture.c)
install(TARGETS sima_neat sima_neat_static EXPORT SimaNeatTargets
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT runtime)
install(EXPORT SimaNeatTargets NAMESPACE SimaNeat::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SimaNeat COMPONENT dev)
install(FILES SimaNeatConfig.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/SimaNeat COMPONENT dev)
""")
        for index, (prefix, libdir) in enumerate((
                ("/usr", "lib"), ("/usr", "lib/aarch64-linux-gnu"),
                ("/usr", "lib64"), ("/opt/neat", "custom/libraries"))):
            with self.subTest(prefix=prefix, libdir=libdir):
                build = self.root / f"build-{index}"
                stage = self.root / f"stage-{index}"
                commands = [
                    ["cmake", "-S", str(source), "-B", str(build),
                     f"-DCMAKE_INSTALL_PREFIX={prefix}", f"-DCMAKE_INSTALL_LIBDIR={libdir}"],
                    ["cmake", "--install", str(build), "--component", "dev"],
                    [sys.executable, str(CHECKER), str(stage), "--forbid-prefix", str(build)],
                ]
                for command in commands:
                    result = subprocess.run(command, env={**os.environ, "DESTDIR": str(stage)},
                                            text=True, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT)
                    self.assertEqual(result.returncode, 0, result.stdout)


if __name__ == "__main__":
    unittest.main()
