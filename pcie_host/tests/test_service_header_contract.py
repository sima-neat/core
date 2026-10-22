#!/usr/bin/env python3
"""Host build-selection contracts; no platform service or target execution."""
from pathlib import Path
import os
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ServiceHeaderContract(unittest.TestCase):
    def test_source_and_installed_consumers_request_declared_component(self):
        for relative in ("CMakeLists.txt", "cmake/SimaPCIeHostConfig.cmake.in"):
            with self.subTest(path=relative):
                source = (ROOT / relative).read_text()
                self.assertIn("COMPONENTS pcie_host pcie_service_headers", source)
                self.assertNotIn("find_path(SIMAPCIE_SVC_INCLUDE_DIR", source)
        source = (ROOT / "CMakeLists.txt").read_text()
        self.assertIn("SimaScheduler::pcie_service_headers", source)

    def test_retired_override_rejected_before_dependency_installation(self):
        result = subprocess.run(["bash", str(ROOT / "build.sh"), "--help"],
                                env=dict(os.environ, SIMAPCIE_SVC_INCLUDE_DIR="/unverified/headers"),
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIn("pcie_service_headers", result.stdout)
        self.assertIn("SIMA_SCHEDULER_PCIE_SERVICE_HEADERS_DEB", result.stdout)
        self.assertNotIn("apt-get", result.stdout)
        self.assertNotIn("-DSIMAPCIE_SVC_INCLUDE_DIR=", (ROOT / "build.sh").read_text())


if __name__ == "__main__":
    unittest.main()
