from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUN_CTEST = ROOT / "scripts/ci/pciehost/run_ctest.sh"
RUN_PYTEST = ROOT / "scripts/ci/pciehost/run_pytest.sh"


class PcieCiRunnerScriptTests(unittest.TestCase):
    def test_throughput_regression_exclusions_are_narrow_and_tracked(self) -> None:
        ctest_script = RUN_CTEST.read_text(encoding="utf-8")
        pytest_script = RUN_PYTEST.read_text(encoding="utf-8")

        self.assertIn("TODO(core#1037)", ctest_script)
        self.assertIn("--exclude-regex '^test_tensor_throughput$'", ctest_script)
        self.assertIn("TODO(core#1037)", pytest_script)
        self.assertIn("-k 'not test_tensor_throughput_yolov8'", pytest_script)


if __name__ == "__main__":
    unittest.main()
