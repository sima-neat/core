#!/usr/bin/env python3
"""Exercise the installed CMake dependency ordering from a clean consumer."""

import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CMakePackageContract(unittest.TestCase):
  def test_export_dependencies_exist_before_targets_are_loaded(self):
    template = (ROOT / "cmake/SimaPCIeHostConfig.cmake.in").read_text(encoding="utf-8")
    self.assertIn("PkgConfig::SIMAPCIE_GST", (ROOT / "CMakeLists.txt").read_text())
    self.assertLess(
        template.index("pkg_check_modules(SIMAPCIE_GST"),
        template.index('include("${CMAKE_CURRENT_LIST_DIR}/SimaPCIeHostTargets.cmake")'),
    )

    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      package = root / "package"
      scheduler = root / "scheduler"
      source = root / "consumer"
      package.mkdir()
      scheduler.mkdir()
      source.mkdir()

      config = template.replace("@PACKAGE_INIT@", "")
      config = config.replace(
          "@SIMAPCIE_SCHEDULER_CONFIG_COMPONENTS@", "pcie_host pcie_service_headers"
      )
      (package / "SimaPCIeHostConfig.cmake").write_text(config, encoding="utf-8")
      (package / "SimaPCIeHostTargets.cmake").write_text(
          """
if(NOT TARGET PkgConfig::SIMAPCIE_GST)
  message(FATAL_ERROR "GStreamer target was not created before the exported targets")
endif()
add_library(SimaPCIeHost::sima_neat_pcie_host INTERFACE IMPORTED)
""",
          encoding="utf-8",
      )
      (scheduler / "SimaSchedulerConfig.cmake").write_text(
          """
set(SimaScheduler_FOUND TRUE)
foreach(component IN LISTS SimaScheduler_FIND_COMPONENTS)
  set(SimaScheduler_${component}_FOUND TRUE)
endforeach()
""",
          encoding="utf-8",
      )
      (source / "CMakeLists.txt").write_text(
          """
cmake_minimum_required(VERSION 3.16)
project(SimaPCIeHostConsumer LANGUAGES CXX)
find_package(SimaPCIeHost REQUIRED CONFIG)
""",
          encoding="utf-8",
      )

      result = subprocess.run(
          [
              "cmake",
              "-S",
              str(source),
              "-B",
              str(root / "build"),
              "-DSimaPCIeHost_DIR=" + str(package),
              "-DSimaScheduler_DIR=" + str(scheduler),
          ],
          text=True,
          capture_output=True,
      )
      self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
  unittest.main()
