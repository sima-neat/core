from __future__ import annotations

import re
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

    def test_pcie_hardware_reprovisions_ssh_before_card_checks(self) -> None:
        pcie_hardware = job_block("test-pciehost-hardware")

        provision_step = pcie_hardware.index(
            "- name: Provision PCIe card SSH key"
        )
        prerequisites_step = pcie_hardware.index(
            "- name: Prepare PCIe card CI prerequisites"
        )
        compatibility_step = pcie_hardware.index(
            "- name: Check PCIe card platform compatibility"
        )

        self.assertLess(provision_step, prerequisites_step)
        self.assertLess(prerequisites_step, compatibility_step)
        self.assertIn(
            "SIMAPCIE_SSH_PASSWORD: ${{ secrets.DEVKIT_PASSWORD }}",
            pcie_hardware,
        )
        self.assertIn("./pcie_host/scripts/pcie-setup.sh", pcie_hardware)
        self.assertIn(
            "./scripts/ci/pciehost/prepare_card_prerequisites.sh",
            pcie_hardware,
        )
        self.assertIn('--hosts "${SIMAPCIE_CARD_HOST}"', pcie_hardware)
        self.assertIn("--non-interactive", pcie_hardware)
        self.assertIn("--strict", pcie_hardware)


if __name__ == "__main__":
    unittest.main()
