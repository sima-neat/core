#!/usr/bin/env python3
"""Keep the native Windows package owned by Windows Installer."""

from pathlib import Path
import re
import unittest


PACKAGING = Path(__file__).parents[1] / "cmake" / "PackagingWindows.cmake"
ROOT = Path(__file__).parents[1]


class WindowsPackagingContract(unittest.TestCase):
  def test_msi_has_stable_upgrade_and_install_ownership(self):
    source = PACKAGING.read_text(encoding="utf-8")
    self.assertIn('set(CPACK_GENERATOR "WIX")', source)
    self.assertIn('set(SIMAPCIE_WIX_INSTALL_DIRECTORY "SimaPCIeHost" CACHE STRING', source)
    self.assertIn('set(CPACK_WIX_INSTALL_SCOPE perMachine)', source)
    self.assertIn('set(CPACK_WIX_COMPONENT_INSTALL ON)', source)
    self.assertIn('set(CPACK_WIX_CMAKE_PACKAGE_REGISTRY "SimaPCIeHost")', source)
    self.assertIn('set(CPACK_COMPONENTS_ALL Runtime Development PcieHostTutorials)', source)
    self.assertIn('set(CPACK_COMPONENT_PCIEHOSTTUTORIALS_DEPENDS Runtime)', source)
    self.assertIn('"${CMAKE_CURRENT_BINARY_DIR}/LICENSE.txt" COPYONLY', source)
    self.assertNotIn('set(CPACK_GENERATOR "ZIP")', source)
    match = re.search(r'CPACK_WIX_UPGRADE_GUID "([0-9A-F-]+)"', source)
    self.assertIsNotNone(match)
    self.assertRegex(
        match.group(1),
        r"^[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}$",
    )

  def test_msi_installs_process_scoped_activation_tools(self):
    cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    self.assertIn("scripts/windows/sima-pcie-host-env.cmd", cmake)
    self.assertIn("scripts/windows/sima-pcie-host-run.cmd", cmake)
    self.assertIn("scripts/windows/sima-pcie-host-doctor.cmd", cmake)
    self.assertIn("COMPONENT Runtime", cmake)
    self.assertIn("SIMAPCIE_NLOHMANN_INCLUDE_ROOT", cmake)
    self.assertIn("nlohmann-json-LICENSE.MIT", cmake)

    environment = (ROOT / "scripts/windows/sima-pcie-host-env.cmd").read_text(
        encoding="utf-8"
    )
    self.assertIn("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR", environment)
    self.assertIn("GSTREAMER_ROOT_X86_64", environment)
    self.assertIn("CMAKE_PREFIX_PATH", environment)
    self.assertIn("GST_PLUGIN_PATH_1_0", environment)
    self.assertNotIn("setx", environment.lower())

    python_bootstrap = (ROOT / "python/pyneatpcie/__init__.py").read_text(
        encoding="utf-8"
    )
    self.assertIn("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR", python_bootstrap)
    self.assertIn("GSTREAMER_ROOT_X86_64", python_bootstrap)
    self.assertIn("Services\\SiMaMlaDaemon", python_bootstrap)

    doctor = (ROOT / "scripts/windows/sima-pcie-host-doctor.cmd").read_text(
        encoding="utf-8"
    )
    self.assertIn("SiMaMlaDaemon", doctor)
    self.assertIn("gst-inspect-1.0.exe", doctor)


if __name__ == "__main__":
  unittest.main()
