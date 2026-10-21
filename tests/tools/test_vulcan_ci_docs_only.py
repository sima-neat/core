from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/vulcan-ci.yml"


def job_block(name: str) -> str:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    match = re.search(
        rf"(?ms)^  {re.escape(name)}:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)",
        workflow,
    )
    if match is None:
        raise AssertionError(f"Workflow job not found: {name}")
    return match.group("body")


class VulcanCiDocsOnlyTests(unittest.TestCase):
    def test_docs_only_configures_without_runtime_but_real_analysis_requires_it(self) -> None:
        # These are real source configurations, not an empty stand-in project
        # or an imported Scheduler stub. Keep host checks out of the SDK's
        # cross-compiler environment; the product build remains unchanged.
        cmake = shutil.which("cmake")
        compiler = shutil.which("c++")
        self.assertIsNotNone(cmake, "Host tooling tests require CMake")
        self.assertIsNotNone(compiler, "Host tooling tests require a C++ compiler")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            env = {"PATH": os.environ["PATH"], "HOME": temporary, "LANG": "C"}
            # Reuse the exact tree in both directions: CMake caches mode options.
            build = root / "reused-build"
            for iteration, docs_only in enumerate((True, False, True)):
                with self.subTest(docs_only=docs_only):
                    result = subprocess.run(
                        [cmake, "-S", str(ROOT), "-B", str(build),
                         "-G", "Unix Makefiles", f"-DCMAKE_CXX_COMPILER={compiler}",
                         "-DSIMANEAT_BUILD_DOCS=ON",
                         f"-DSIMANEAT_DOCS_ONLY_CONFIGURE={'ON' if docs_only else 'OFF'}",
                         f"-DSIMANEAT_CPP_TIDY={'OFF' if docs_only else 'ON'}",
                         "-DSIMANEAT_BUILD_TESTS=OFF",
                         "-DSIMANEAT_BUILD_TUTORIALS=OFF",
                         "-DCMAKE_DISABLE_FIND_PACKAGE_SimaScheduler=TRUE",
                         "-DCMAKE_DISABLE_FIND_PACKAGE_SimaLMM=TRUE"],
                        env=env, text=True, capture_output=True)
                    if docs_only:
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        targets = subprocess.run(
                            [cmake, "--build", str(build), "--target", "help"],
                            env=env, text=True, capture_output=True)
                        self.assertEqual(targets.returncode, 0, targets.stderr)
                        self.assertNotIn("sima_neat", targets.stdout)
                        if iteration == 0:
                            self.assertNotIn("... package", targets.stdout)
                        # CMake may retain a package target from the intervening
                        # failed configure's CPackConfig.cmake. Docs mode never
                        # invokes it; no runtime build targets may survive.
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn("SimaScheduler", result.stderr)
                        self.assertIn("REQUIRED", result.stderr)

    def test_localization_runs_independently_of_docs_only_detection(self) -> None:
        localization = job_block("documentation-localization")

        self.assertNotIn("needs:", localization)
        self.assertNotIn("if:", localization)
        self.assertIn(
            "run: bash scripts/ci/check_documentation_localization.sh", localization
        )

    def test_protected_branch_pushes_are_never_docs_only(self) -> None:
        detect = job_block("detect-changes")

        self.assertIn(
            '[[ "${GITHUB_REF_NAME}" == "develop" || "${GITHUB_REF_NAME}" == "main" ]]',
            detect,
        )
        self.assertIn(
            'echo "Protected branch push; treating changeset as code-impacting."',
            detect,
        )
        self.assertNotIn("github.event.before", detect)

    def test_core_build_skips_docs_only_changes(self) -> None:
        build = job_block("build")

        self.assertIn("- detect-changes", build)
        self.assertIn("needs.detect-changes.outputs.docs_only != 'true'", build)

    def test_pcie_package_matrix_skips_docs_only_changes(self) -> None:
        build_pcie = job_block("build-pciehost-packages")

        self.assertIn("- detect-changes", build_pcie)
        self.assertIn(
            "needs.detect-changes.outputs.docs_only != 'true'", build_pcie
        )

    def test_pcie_cpu_job_preserves_docs_condition_without_card_actions(self) -> None:
        pcie_cpu = job_block("test-pciehost-hardware")

        self.assertIn("needs.detect-changes.outputs.docs_only != 'true'", pcie_cpu)
        self.assertIn("runs-on: [self-hosted, Linux, X64, pcie, 3.0.0-pcie-test]", pcie_cpu)
        self.assertIn("prepare_cpu_package.py", pcie_cpu)
        self.assertIn("run_cpu_package_tests.py", pcie_cpu)
        for physical_action in (
            "Provision PCIe card SSH key", "Prepare PCIe card CI prerequisites",
            "Check PCIe card platform compatibility", "pcie-setup.sh",
            "prepare_card_prerequisites.sh", "install_host_package.sh",
            "install_card_runtime.sh", "verify_card_runtime.sh", "cleanup_queue.sh",
            "run_ctest.sh", "run_pytest.sh", "DEVKIT_PASSWORD", "SIMAPCIE_CARD_HOST",
        ):
            with self.subTest(physical_action=physical_action):
                self.assertNotIn(physical_action, pcie_cpu)


if __name__ == "__main__":
    unittest.main()
