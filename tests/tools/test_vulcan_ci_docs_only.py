from __future__ import annotations

import os
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


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
    def test_devkit_uses_platform_resolved_runner_group(self) -> None:
        resolver = yaml.safe_load(job_block("resolve-test-runner"))
        self.assertEqual(
            resolver["uses"],
            "sima-neat/.github/.github/workflows/vulcan-resolve-modalix-runner.yml@main",
        )
        self.assertEqual({key: resolver["with"][key] for key in ("legacy_runner_group", "versioned_runner_group_prefix")}, {
            "legacy_runner_group": "modalix-devkit",
            "versioned_runner_group_prefix": "modalix-devkit-",
        })
        self.assertEqual(resolver["needs"], ["detect-changes", "resolve-vulcan-config", "build"])
        self.assertEqual(resolver["with"]["manifest_artifact"], "neat-core-dist")
        self.assertEqual(resolver["with"]["manifest_path"], "scheduler-sdk-policy.json")
        self.assertEqual(resolver["with"]["shared_workflow_ref"], "main")
        self.assertIs(resolver["with"]["allow-version-fallback"], False)
        self.assertEqual(resolver["secrets"]["app_id"], "${{ secrets.NEAT_RELEASES_APP_ID }}")
        self.assertIn("== 'standard'", resolver["if"])
        self.assertIn("needs.detect-changes.outputs.docs_only != 'true'", resolver["if"])
        devkit = yaml.safe_load(job_block("test-devkit-shard"))
        self.assertEqual(devkit["runs-on"]["group"],
                         "${{ needs.resolve-test-runner.outputs.modalix_runner_group }}")

    def test_devkit_labels_preserve_capabilities_and_artifact_environment(self) -> None:
        devkit = yaml.safe_load(job_block("test-devkit-shard"))
        env = "needs.resolve-vulcan-config.outputs.vulcan_env"
        self.assertEqual(devkit["runs-on"]["labels"],
                         "${{ matrix.labels }}")
        resolver = yaml.safe_load(job_block("resolve-test-runner"))
        self.assertEqual(json.loads(resolver["with"]["runner-labels"]), [
            "self-hosted", "Linux", "ARM64", "modalix",
            "vulcan-${{ " + env + " == 'prod' && 'production' || " + env + " }}",
        ])
        guard = devkit["steps"][0]
        self.assertEqual(guard["name"], "Verify selected DevKit firmware")
        self.assertEqual(guard["env"]["EXPECTED_ELXR_VERSION"],
                         "${{ needs.resolve-test-runner.outputs.elxr_version }}")
        # Human-readable runner labels must not change the resolved publisher
        # environment, SDK selection, or artifact-acquisition environment.
        self.assertEqual(devkit["env"]["VULCAN_ENV"], "${{ " + env + " }}")
        self.assertEqual(
            yaml.safe_load(job_block("resolve-vulcan-config"))["with"]["vulcan_env"],
            "${{ vars.VULCAN_ENV || 'production' }}",
        )

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

    def test_pcie_hardware_job_runs_for_standard_code_changes(self) -> None:
        hardware = yaml.safe_load(job_block("test-pciehost-hardware"))
        self.assertEqual(hardware["name"], "Test PCIe Host Hardware")
        self.assertEqual(hardware["if"],
                         "${{ !cancelled() && (inputs.build_mode || 'standard') == 'standard' "
                         "&& needs.detect-changes.outputs.docs_only != 'true' "
                         "&& needs.build.result == 'success' && needs.build-pciehost-packages.result == 'success' "
                         "&& ((inputs.validation_only && needs.publish-vulcan-artifacts.result == 'skipped' "
                         "&& needs.publish-pciehost-package.result == 'skipped') "
                         "|| (!inputs.validation_only && needs.publish-vulcan-artifacts.result == 'success' "
                         "&& needs.publish-pciehost-package.result == 'success')) }}")
        self.assertEqual(hardware["runs-on"], ["self-hosted", "Linux", "X64", "pcie", "3.0.0-pcie-test"])
        self.assertEqual(hardware["env"]["SIMAPCIE_CARD_HOST"], "10.0.0.2")
        self.assertEqual(hardware["env"]["DEVKIT_PASSWORD"], "${{ secrets.DEVKIT_PASSWORD }}")
        self.assertEqual(hardware["env"]["SUDO_PASSWORD"], "${{ secrets.PCIEHOST_SUDO_PASSWORD }}")

    def test_pcie_hardware_job_retains_complete_qualification_steps(self) -> None:
        hardware = yaml.safe_load(job_block("test-pciehost-hardware"))
        commands = [step.get("run", "") for step in hardware["steps"]]
        previous = -1
        for script in (
            "preflight.sh", "pcie-setup.sh", "prepare_card_prerequisites.sh",
            "check_card_compatibility.sh", "install_host_package.sh", "install_host_test_deps.sh",
            "install_card_runtime.sh", "extract_extras.sh", "resolve_assets.sh", "verify_card_runtime.sh",
            "prepare_pcie_genai_models.sh", "run_ctest.sh", "run_pytest.sh", "collect_artifacts.sh",
        ):
            with self.subTest(script=script):
                matches = [index for index, command in enumerate(commands) if script in command]
                self.assertEqual(len(matches), 1)
                self.assertGreater(matches[0], previous)
                previous = matches[0]
        cleanup = [step for step in hardware["steps"] if "cleanup_queue.sh" in step.get("run", "")]
        self.assertTrue(any(step.get("if") == "${{ always() }}" for step in cleanup))
        self.assertTrue(any(step.get("uses", "").startswith("actions/upload-artifact@")
                            and step.get("if") == "${{ always() }}" for step in hardware["steps"]))


if __name__ == "__main__":
    unittest.main()
