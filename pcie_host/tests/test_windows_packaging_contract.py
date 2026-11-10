#!/usr/bin/env python3
"""Keep the native Windows package owned by Windows Installer."""

from pathlib import Path
import re
import unittest


PACKAGING = Path(__file__).parents[1] / "cmake" / "PackagingWindows.cmake"


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


if __name__ == "__main__":
  unittest.main()
