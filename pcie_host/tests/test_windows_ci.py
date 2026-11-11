#!/usr/bin/env python3
"""Contracts for the native Windows Core PCIe-host producer."""

import importlib.util
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "pciehost" / "build_windows.py"
SCRIPT_DIR = SCRIPT.parent
WORKFLOW = ROOT / ".github" / "workflows" / "pcie-host-windows.yml"
MAIN_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-ci.yml"
sys.path.insert(0, str(SCRIPT_DIR))
SPEC = importlib.util.spec_from_file_location("build_windows", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
from windows_candidate import (  # noqa: E402
    dependency_identity, require_junit_cases, sha256, validate_candidate)
from qualify_windows_hardware import (  # noqa: E402
    ExclusiveLock, RemoteExclusiveLock, card_snapshot, card_stage_cleanup_command, load_upgrade_base,
    parse_performance_log, require_powershell as require_hardware_powershell,
    recover_windows_platform_device, remove_installed_tutorial_facade,
    run_captured_to_file, run_sudo_remote,
    sima_cli_identity, stage_card_candidate, stage_installed_tutorial_facade,
    stage_invalid_wheel_registry, stage_local_registry, stage_path_fixtures,
    stress_queues, validate_card_candidate)
from verify_linux_release_run import verify_run as verify_linux_run  # noqa: E402
from verify_windows_release import verify  # noqa: E402
import prepare_card_qualification_candidate as CARD_PREP  # noqa: E402
import cleanup_windows_hardware as RECOVERY  # noqa: E402
import acquire_windows_dependencies as WINDOWS_DEPENDENCIES  # noqa: E402


def posix_shell():
  shell = shutil.which("sh")
  if shell:
    return shell
  git = shutil.which("git")
  if git:
    git_root = Path(git).resolve().parent.parent
    for candidate in (git_root / "usr" / "bin" / "sh.exe",
                      git_root / "bin" / "sh.exe"):
      if candidate.is_file():
        return str(candidate)
  raise FileNotFoundError("POSIX shell is unavailable for remote-command contract fixtures")


def run_posix(command):
  shell = posix_shell()
  environment = os.environ.copy()
  environment["PATH"] = os.pathsep.join((str(Path(shell).parent), environment["PATH"]))
  environment["WINDOWS_FIXTURE_PYTHON"] = sys.executable
  return subprocess.run(
      [shell, "-c", command], capture_output=True, text=True,
      check=False, env=environment)


def normalized_link_target(path):
  target = os.readlink(path)
  if os.name == "nt" and target.startswith("\\\\?\\"):
    target = target[4:]
  return os.path.normcase(os.path.normpath(target))


class WindowsCiContracts(unittest.TestCase):
  def test_junit_policy_requires_exact_zero_skip_cases(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "tests.xml"
      path.write_text(
          '<testsuites><testsuite tests="2"><testcase name="alpha"/>'
          '<testcase name="beta"/></testsuite></testsuites>', encoding="utf-8")
      self.assertEqual(
          require_junit_cases(path, {"alpha", "beta"}),
          {"tests": 2, "skipped": 0, "failures": 0, "errors": 0})

  def test_junit_policy_rejects_skip_or_changed_case_set(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "tests.xml"
      path.write_text(
          '<testsuites><testsuite tests="2" skipped="1">'
          '<testcase name="alpha"><skipped/></testcase>'
          '<testcase name="other"/></testsuite></testsuites>', encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "required case set"):
        require_junit_cases(path, {"alpha", "beta"})
      path.write_text(
          '<testsuites><testsuite tests="1" skipped="1">'
          '<testcase name="alpha"><skipped/></testcase></testsuite></testsuites>',
          encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "zero-skip pass"):
        require_junit_cases(path, {"alpha"})

  def test_dependency_acquisition_extracts_one_safe_cli_wheel(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      publication = root / "publication"
      publication.mkdir()
      package = publication / "sima-cli-package-test.zip"
      with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("sima_cli-1.0-py3-none-any.whl", b"wheel")
        archive.writestr("windows.bat", b"installer")
      wheel = WINDOWS_DEPENDENCIES.extract_sima_cli_wheel(
          publication, root / "candidate")
      self.assertEqual(wheel.read_bytes(), b"wheel")

  def test_dependency_acquisition_rejects_nested_cli_wheel(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      publication = root / "publication"
      publication.mkdir()
      with zipfile.ZipFile(publication / "sima-cli-package-test.zip", "w") as archive:
        archive.writestr("nested/sima_cli-1.0-py3-none-any.whl", b"wheel")
      with self.assertRaisesRegex(ValueError, "unsafe wheel"):
        WINDOWS_DEPENDENCIES.extract_sima_cli_wheel(
            publication, root / "candidate")

  def test_dependency_acquisition_stages_admitted_actions_overrides(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      scheduler = root / "scheduler"
      cli = root / "cli"
      scheduler.mkdir()
      cli.mkdir()
      (scheduler / "scheduler.msi").write_bytes(b"scheduler")
      (cli / "sima_cli-test.whl").write_bytes(b"cli")
      result = subprocess.run([
          sys.executable, str(SCRIPT_DIR / "acquire_windows_dependencies.py"),
          "--branch", "develop", "--output", str(root / "output"),
          "--scheduler-candidate", str(scheduler),
          "--scheduler-commit", "a" * 40,
          "--sima-cli-candidate", str(cli),
          "--sima-cli-commit", "b" * 40,
      ], text=True, capture_output=True, check=False)
      self.assertEqual(result.returncode, 0, result.stderr)
      selection = json.loads((root / "output" / "selection.json").read_text())
      self.assertEqual(selection["scheduler"]["commit"], "a" * 40)
      self.assertEqual(selection["sima_cli"]["commit"], "b" * 40)
      self.assertEqual(
          (root / "output" / "scheduler" / "scheduler.msi").read_bytes(),
          b"scheduler")
      self.assertEqual(
          (root / "output" / "sima-cli" / "sima_cli-test.whl").read_bytes(),
          b"cli")

  def test_dependency_acquisition_defaults_to_supported_cli_channel(self):
    source = (SCRIPT_DIR / "acquire_windows_dependencies.py").read_text(encoding="utf-8")
    self.assertIn('parser.add_argument("--sima-cli-branch", default="main")', source)

  def test_recovery_requires_known_private_stage_ownership(self):
    with tempfile.TemporaryDirectory() as directory:
      evidence = Path(directory)
      for code, output, expected in ((0, "owned", True), (0, "absent", False),
                                     (1, "", None), (0, "unexpected", None)):
        with self.subTest(code=code, output=output), \
            mock.patch.object(RECOVERY, "ssh_command", side_effect=lambda _args, cmd: [cmd]), \
            mock.patch.object(RECOVERY, "run_captured_to_file", return_value=
                              subprocess.CompletedProcess([], code, output)):
          if expected is None:
            with self.assertRaisesRegex(RuntimeError, "ownership is unknown"):
              RECOVERY.card_stage_owned(SimpleNamespace(), "a" * 32, evidence)
          else:
            self.assertIs(RECOVERY.card_stage_owned(SimpleNamespace(), "a" * 32, evidence),
                          expected)

  def test_private_stage_cleanup_requires_its_creation_receipt(self):
    for owner in (None, "another-run", "this-run"):
      with self.subTest(owner=owner), tempfile.TemporaryDirectory() as directory:
        stage = Path(directory) / "stage with spaces"
        stage.mkdir()
        (stage / "customer-data").write_text("preserve unless owned")
        if owner is not None:
          (stage / ".qualification-owner").write_text(owner)
        result = run_posix(
            card_stage_cleanup_command({"stage": str(stage)}, "this-run"))
        self.assertEqual(result.returncode == 0, owner == "this-run")
        self.assertEqual(stage.exists(), owner != "this-run")

  def test_recovery_without_owned_lock_does_not_mutate_host_or_card(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      args = SimpleNamespace(candidate=root, upgrade_base_candidate=None,
                             evidence=root / "evidence", lock_file=root / "absent.lock",
                             remote_lock_path="/tmp/neat-pcie-card0.lock")
      with mock.patch.object(RECOVERY.sys, "platform", "win32"), \
          mock.patch.object(RECOVERY, "validate_candidate"), \
          mock.patch.object(RECOVERY, "RemoteExclusiveLock") as lock, \
          mock.patch.object(RECOVERY, "stop_card_workers") as stop, \
          mock.patch.object(RECOVERY, "remove_msi") as remove, \
          mock.patch.object(RECOVERY.shutil, "rmtree") as remove_work:
        lock.return_value.owner.return_value = None
        RECOVERY.cleanup(args)
      stop.assert_not_called()
      remove.assert_not_called()
      remove_work.assert_not_called()
      receipt = json.loads((args.evidence / "recovery-cleanup.json").read_text())
      self.assertEqual(receipt["errors"], [])

  def scheduler_candidate(self, root: Path) -> Path:
    msi = root / "scheduler.msi"
    msi.write_bytes(b"scheduler-msi")
    native_junit = root / "native-tests.xml"
    native_junit.write_text(
        '<testsuites><testsuite tests="1"><testcase name="scheduler_native"/>'
        '</testsuite></testsuites>', encoding="utf-8")
    receipt = root / "host-candidate.json"
    receipt.write_text(json.dumps({
        "native_tests": {
            "errors": 0,
            "exit_code": 0,
            "failures": 0,
            "junit_sha256": MODULE.sha256(native_junit),
            "names": ["scheduler_native"],
            "skipped": 0,
            "tests": 1,
        },
        "source_commit": "a" * 40,
    }) + "\n", encoding="utf-8")
    dependencies = root / "windows-dependencies.json"
    dependencies.write_text(json.dumps({
        "gstreamer": {
            "pkg_config_sha256": "e" * 64,
            "runtime_dlls": [
                {"name": "gstreamer-1.0-0.dll", "sha256": "f" * 64},
            ],
            "version_output": "gst-inspect version 1.26.0",
        },
        "platform": {"runtime_dlls": [
            {"name": "sima-platform.dll", "sha256": "d" * 64},
        ]},
    }), encoding="utf-8")
    resources = {
        path.name: MODULE.sha256(path)
        for path in (msi, native_junit, receipt, dependencies)}
    (root / "metadata.json").write_text(json.dumps({
        "artifact": {"architecture": "amd64", "profile": "pcie-host-windows"},
        "commit": "a" * 40,
        "installation": {"msi": msi.name, "type": "windows-msi"},
        "repository": "sima-neat/scheduler",
        "resources": sorted(resources),
        "resources-checksum": resources,
        "version": "a" * 12,
    }), encoding="utf-8")
    return msi

  def test_scheduler_candidate_accepts_exact_inventory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      expected = self.scheduler_candidate(root)
      metadata, actual = MODULE.verify_scheduler_candidate(root, "a" * 40)
      self.assertEqual(actual, expected)
      self.assertEqual(metadata["commit"], "a" * 40)

  def test_scheduler_candidate_rejects_changed_msi(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      msi = self.scheduler_candidate(root)
      msi.write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "checksum differs"):
        MODULE.verify_scheduler_candidate(root, "a" * 40)

  def test_scheduler_candidate_rejects_non_mapping_checksums(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.scheduler_candidate(root)
      metadata_path = root / "metadata.json"
      metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      metadata["resources-checksum"] = metadata["resources"]
      metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "inventory is incomplete"):
        MODULE.verify_scheduler_candidate(root, "a" * 40)

  def test_scheduler_candidate_rejects_unselected_commit(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.scheduler_candidate(root)
      with self.assertRaisesRegex(ValueError, "is not the selected commit"):
        MODULE.verify_scheduler_candidate(root, "b" * 40)

  def test_scheduler_candidate_rejects_skipped_native_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.scheduler_candidate(root)
      native_junit = root / "native-tests.xml"
      native_junit.write_text(
          '<testsuites><testsuite tests="1" skipped="1">'
          '<testcase name="scheduler_native"><skipped/></testcase>'
          '</testsuite></testsuites>', encoding="utf-8")
      receipt_path = root / "host-candidate.json"
      receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
      receipt["native_tests"]["junit_sha256"] = MODULE.sha256(native_junit)
      receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
      metadata_path = root / "metadata.json"
      metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      metadata["resources-checksum"]["native-tests.xml"] = MODULE.sha256(native_junit)
      metadata["resources-checksum"]["host-candidate.json"] = MODULE.sha256(receipt_path)
      metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "zero-skip pass"):
        MODULE.verify_scheduler_candidate(root, "a" * 40)

  def test_workflow_uses_exact_scheduler_artifact_and_restricted_runner(self):
    source = WORKFLOW.read_text(encoding="utf-8")
    self.assertIn("  pull_request:\n", source)
    self.assertIn("if: github.event_name != 'pull_request'", source)
    self.assertIn("github.repository == 'sima-neat/core'", source)
    self.assertIn("scheduler_run_id", source)
    self.assertIn("scheduler_commit", source)
    self.assertIn("--expected-scheduler-commit", source)
    self.assertIn('"Scheduler native Windows PCIe host"', source)
    self.assertIn('".github/workflows/pcie-host-windows.yml"', source)
    self.assertIn("repository: sima-neat/scheduler", source)
    self.assertIn("scheduler-native-windows-amd64", source)
    self.assertIn("pcie-windows-build", source)
    self.assertIn("pcie-windows-hardware", source)
    self.assertNotIn("windows-latest", source)
    self.assertIn("scripts/ci/pciehost/build_windows.py", source)
    self.assertIn("scripts/ci/pciehost/qualify_windows_hardware.py", source)
    self.assertIn("scripts/ci/pciehost/cleanup_windows_hardware.py", source)
    self.assertIn("needs: [hardware, linux_release]", source)
    self.assertIn("linux_run_id", source)
    self.assertIn("Download this run's exact Linux card candidate", source)
    self.assertIn("Download prior run's exact Linux card candidate", source)
    self.assertIn("name: neat-core-dist", source)
    self.assertIn("--card-candidate", source)
    self.assertIn("--expected-card-commit", source)
    self.assertIn("SIMAPCIE_CARD_SUDO_PASSWORD", source)
    self.assertNotIn("SIMAPCIE_EXPECTED_CARD_RUNTIME_SHA256", source)
    self.assertNotIn("SIMAPCIE_EXPECTED_CARD_WORKER_SHA256", source)
    self.assertNotIn("SIMAPCIE_EXPECTED_CARD_GENAI_WORKER_SHA256", source)
    self.assertIn("sima_cli_run_id", source)
    self.assertIn("sima-cli-dist", source)
    self.assertIn("--require-signatures", source)
    self.assertIn('$arguments += "--require-genai"', source)
    self.assertIn('$arguments += "--require-release"', source)
    self.assertIn('if ("${{ inputs.require_signatures }}" -eq "true")', source)
    self.assertIn("Recover card and native installation", source)
    self.assertIn("if: always()", source)
    self.assertIn("upgrade_base_run_id", source)
    self.assertIn("upgrade_base_commit", source)
    self.assertIn("Admit exact successful Windows upgrade baseline", source)
    self.assertIn('"Core native Windows PCIe host"', source)
    self.assertIn("core-windows-upgrade-base", source)
    self.assertIn("--expected-upgrade-base-commit", source)
    self.assertIn("SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH", source)
    self.assertIn("--dependency-runtime-dir", source)
    self.assertIn("scripts/ci/pciehost/acquire_windows_dependencies.py", source)
    self.assertIn("core-pciehost-windows-dependencies", source)
    self.assertIn("linux_current_run", source)
    self.assertIn("require_upgrade", source)
    self.assertEqual(source.count("shell: powershell"), 5)

  def test_main_ci_requires_windows_acceptance_before_promotion(self):
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    self.assertIn("test-pciehost-windows:", source)
    self.assertIn("uses: ./.github/workflows/pcie-host-windows.yml", source)
    self.assertIn("needs.test-pciehost-windows.result == 'success'", source)

  def test_producer_accepts_inbox_windows_powershell(self):
    def which(name):
      return "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe" \
          if name == "powershell.exe" else None

    with mock.patch.object(MODULE.shutil, "which", side_effect=which):
      self.assertEqual(
          MODULE.require_powershell(),
          str(Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe").resolve()),
      )

  def test_hardware_qualification_accepts_inbox_windows_powershell(self):
    def which(name):
      return "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe" \
          if name == "powershell.exe" else None

    with mock.patch("qualify_windows_hardware.shutil.which", side_effect=which):
      self.assertEqual(
          require_hardware_powershell(),
          str(Path("C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe").resolve()),
      )

  def test_signature_path_avoids_powershell_command_line_quoting(self):
    path = Path("C:/Program Files/SimaPCIeHost/bin/plugin.dll")
    output = '{"status":"NotSigned","message":"","thumbprint":null,' \
        '"timestamp_thumbprint":null}'
    with mock.patch.object(MODULE, "run", return_value=output) as run:
      MODULE.authenticode_status(path, "powershell.exe")
    command = run.call_args.args[0]
    environment = run.call_args.kwargs["env"]
    self.assertNotIn(str(path), command)
    self.assertEqual(environment["SIMA_WINDOWS_SIGNATURE_PATH"], str(path))

  def test_producer_records_runtime_inputs_and_repeat_installation(self):
    source = SCRIPT.read_text(encoding="utf-8")
    self.assertIn('"windows-dependencies.json"', source)
    self.assertIn('"repeat_install": "PASS"', source)
    self.assertNotIn("copytree(platform_runtime", source)
    self.assertIn("SUPPORTED_PYTHON = (3, 14)", source)
    self.assertIn('"dependency_runtime"', source)
    self.assertIn('"--output-junit", native_junit', source)
    self.assertIn("require_junit_cases(native_junit, set(test_names))", source)

  def test_hardware_qualification_bounds_blocker_and_injects_process_faults(self):
    source = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    self.assertIn(
        'require_junit_cases(evidence / "python-hardware.xml", set(BASE_PYTHON_TESTS))',
        source)
    self.assertIn('require_junit_cases(evidence / "python-genai.xml", genai_tests)', source)
    self.assertIn('command.extend(["--iterations", "10", "--interval-ms", "0"])', source)
    self.assertIn('completed.append("test_worker_loss_recovery")', source)
    self.assertIn('completed.append("test_client_process_death_recovery")', source)
    self.assertIn('["--max-mean-latency-ms", "12"]', source)
    self.assertIn('["--min-throughput-fps", "440"]', source)
    self.assertIn('"--warmup", "100", "--iterations", "3000"', source)
    self.assertIn('command.append("--terminate-on-pull-timeout")', source)
    self.assertIn('evidence / "client-process-death-worker-cleanup.log"', source)
    self.assertIn('evidence / "client-process-death-platform-recovery.log"', source)
    self.assertIn('evidence / "client-process-death-card-readiness.log"', source)
    self.assertIn('"simaai_pcie_test.exe"', source)
    self.assertIn("Device recovery successful", source)
    self.assertIn("Stop-Service -InputObject $service", source)
    self.assertIn('evidence / "card-candidate-platform-start.log"', source)
    self.assertIn('evidence / "card-stock-platform-recovery.log"', source)
    self.assertIn("Restart-Service -InputObject $service", source)
    self.assertIn("process.kill()", source)
    self.assertIn('"/fa"', source)
    self.assertIn('"gst-inspect-1.0.exe"', source)
    self.assertIn('install_root / "bin" / "gstreamer-1.0" / "gstneatpciehost.dll"', source)
    self.assertIn('work / "native customer downloads with spaces"', source)
    self.assertIn('work / "missing wheel downloads with spaces"', source)
    self.assertIn('work / "invalid wheel downloads with spaces"', source)
    self.assertIn('work / "recovery downloads with spaces"', source)
    self.assertIn("if card_stage_attempted and not cleanup_errors:", source)
    self.assertIn("restore_card_candidate(args, remote_lock_token", source)
    self.assertIn("stdin=subprocess.DEVNULL", source)
    self.assertIn('cli_environment["SIMA_CLI_CHECK_FOR_UPDATE"] = "0"', source)
    self.assertIn('evidence / "card-candidate-dependencies.log"', source)
    cleanup = (SCRIPT_DIR / "cleanup_windows_hardware.py").read_text(encoding="utf-8")
    self.assertIn("shutil.rmtree(work, ignore_errors=True)", cleanup)
    self.assertIn("restore_card_candidate(args, remote_token", cleanup)
    self.assertIn('evidence / "recovery-platform-service.log"', cleanup)

  def test_platform_device_recovery_requires_vendor_tool(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with self.assertRaisesRegex(RuntimeError, "recovery tool is unavailable"):
        recover_windows_platform_device(root, root / "recovery.log")

  def test_platform_device_recovery_uses_provisioned_vendor_tool(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      tool = root / "simaai_pcie_test.exe"
      tool.write_bytes(b"fixture")
      with mock.patch("qualify_windows_hardware.require_powershell",
                      return_value="powershell.exe"), \
           mock.patch("qualify_windows_hardware.run_logged") as run:
        recover_windows_platform_device(root, root / "recovery.log")
      self.assertEqual(
          run.call_args.kwargs["env"]["SIMA_WINDOWS_PLATFORM_RECOVERY_TOOL"],
          str(tool))
      self.assertEqual(run.call_args.kwargs["timeout"], 120)

  def test_wheel_payload_reader_checks_crc(self):
    with tempfile.TemporaryDirectory() as directory:
      wheel = Path(directory) / "test.whl"
      import zipfile
      with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("pyneatpcie/__init__.py", b"")
      self.assertEqual(MODULE.wheel_payload_names(wheel), {"pyneatpcie/__init__.py"})

  def windows_candidate(self, root: Path, *, signed: bool = True,
                        commit: str = "c" * 40, version: str = "0.6.2"):
    scheduler = root / "scheduler.msi"
    core = root / "core.msi"
    wheel = root / "pyneatpcie.whl"
    for path, content in ((scheduler, b"scheduler"), (core, b"core"), (wheel, b"wheel")):
      path.write_bytes(content + commit.encode("ascii"))
    resources = {path.name: sha256(path) for path in (scheduler, core, wheel)}
    qualification_resource = root / "qualification-resource.bin"
    qualification_resource.write_bytes(b"hardware-tests-and-signature-receipt")
    native_junit = root / "native-tests.xml"
    native_junit.write_text(
        '<testsuites><testsuite tests="1"><testcase name="native_contract"/>'
        '</testsuite></testsuites>', encoding="utf-8")
    signature_record = {
        "path": scheduler.name,
        "status": "Valid" if signed else "NotSigned",
        "timestamp_thumbprint": "B" * 40 if signed else None,
        "thumbprint": "A" * 40 if signed else None,
    }
    scheduler_signatures = root / "scheduler-windows-signatures.json"
    scheduler_signatures.write_text(json.dumps({
        "certificate_sha1": "A" * 40 if signed else "",
        "records": [signature_record],
        "release_required": signed,
        "timestamp_url": "https://timestamp.example" if signed else None,
    }), encoding="utf-8")
    dependencies = root / "windows-dependencies.json"
    dependency_data = {
        "gstreamer": {
            "pkg_config_sha256": "e" * 64,
            "runtime_dlls": [
                {"name": "gstreamer-1.0-0.dll", "sha256": "f" * 64},
            ],
            "version_output": "gst-inspect version 1.26.0",
        },
        "platform": {
            "runtime_dlls": [{
                "name": "sima-platform.dll",
                "sha256": "d" * 64,
                "signature": {
                    "status": "Valid" if signed else "NotSigned",
                    "timestamp_thumbprint": "C" * 40 if signed else None,
                    "thumbprint": "D" * 40 if signed else None,
                },
            }],
        },
    }
    scheduler_dependencies = root / "scheduler-windows-dependencies.json"
    scheduler_dependencies.write_text(json.dumps(dependency_data), encoding="utf-8")
    dependency_data["dependency_runtime"] = [{
        "files": [{"name": "opencv_world500.dll", "sha256": "9" * 64}],
    }]
    dependencies.write_text(json.dumps(dependency_data), encoding="utf-8")
    core_signatures = root / "windows-signatures.json"
    core_signatures.write_text(json.dumps({
        "certificate_sha1": "A" * 40 if signed else "",
        "records": [signature_record],
        "release_required": signed,
        "timestamp_url": "https://timestamp.example" if signed else None,
    }), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps({
        "artifact": {"package_path": "pciehost/windows/amd64"},
        "commit": commit,
        "installation": {
            "native-packages": [
                {"owner": "scheduler", "msi": scheduler.name},
                {"owner": "core", "msi": core.name},
            ],
            "python": {"wheel": wheel.name},
            "type": "windows-msi",
        },
        "name": "gh:sima-neat/core/pciehost/windows/amd64",
        "repository": "sima-neat/core",
        "resources": list(resources),
        "resources-checksum": resources,
        "version": version,
    }), encoding="utf-8")
    (root / "host-candidate.json").write_text(json.dumps({
        "core_source_commit": commit,
        "native_tests": {
            "errors": 0,
            "exit_code": 0,
            "failures": 0,
            "junit_sha256": sha256(native_junit),
            "names": ["native_contract"],
            "skipped": 0,
            "tests": 1,
        },
        "profile": "pciehost/windows/amd64",
        "qualification_resources": {
            native_junit.name: sha256(native_junit),
            qualification_resource.name: sha256(qualification_resource),
            scheduler_signatures.name: sha256(scheduler_signatures),
            dependencies.name: sha256(dependencies),
            scheduler_dependencies.name: sha256(scheduler_dependencies),
            core_signatures.name: sha256(core_signatures),
        },
        "repository": "sima-neat/core",
        "scheduler_source_commit": "d" * 40,
    }), encoding="utf-8")
    qualification = root / "qualification.json"
    qualification.write_text(json.dumps({
        "candidate": {"core_commit": commit, "resources": resources},
        "card_baseline": {
            "firmware": "modalix-cvu-fw",
            "genai_workers": "0",
            "machine": "aarch64",
            "platform_release": "3.0.0_master_B2501",
            "platform_sha256": "1" * 64,
            "remoteproc_state": "running",
            "workers": "0",
        },
        "card_candidate": {
            "core_commit": commit,
            "firmware": "neat-pcie-card0-" + "4" * 32,
            "genai_worker_sha256": "5" * 64,
            "runtime_sha256": "2" * 64,
            "worker_sha256": "3" * 64,
        },
        "card_final": {
            "firmware": "modalix-cvu-fw",
            "genai_workers": "0",
            "machine": "aarch64",
            "platform_release": "3.0.0_master_B2501",
            "platform_sha256": "1" * 64,
            "remoteproc_state": "running",
            "workers": "0",
        },
        "cleanup": {
            "card_candidate": "PASS", "core_msi": "PASS", "errors": [],
            "python": "PASS", "remote_lock": "PASS",
            "scheduler_msi": "PASS"},
        "customer_install": {
            "command": "core/pciehost/windows/amd64@3.0",
            "download_path_with_spaces": r"C:\customer downloads\candidate",
            "missing_wheel": {"exit_code": 1, "status": "PASS"},
            "native_only": "PASS",
            "plugin_discovery": "PASS",
            "repair": {"core": "PASS", "scheduler": "PASS"},
            "recovery": "PASS",
            "repeat": "PASS",
            "upgrade": {
                "baseline_commit": "b" * 40,
                "baseline_version": "0.6.1",
                "candidate_version": version,
                "status": "PASS",
            },
            "wheel_failure": {"exit_code": 1, "status": "PASS"},
        },
        "dependencies": {
            "gstreamer": {
                "pkg_config_sha256": "e" * 64,
                "runtime_dlls": [
                    {"name": "gstreamer-1.0-0.dll", "sha256": "f" * 64},
                ],
                "version_output": "gst-inspect version 1.26.0",
            },
            "platform_runtime_dlls": [
                {"name": "sima-platform.dll", "sha256": "d" * 64},
            ],
            "dependency_runtime": [[
                {"name": "opencv_world500.dll", "sha256": "9" * 64},
            ]],
        },
        "path_fixtures": {
            "SIMAPCIE_YOLOV8_MODEL": r"C:\qualification assets\model.tar.gz",
            "SIMAPCIE_TEST_IMAGE": r"C:\qualification assets\image.jpg",
            "SIMAPCIE_BOXDECODE_IMAGE": r"C:\qualification assets\people.jpg",
        },
        "performance": {
            "latency": {"count": 3000, "effective_fps": 91.2, "mean_ms": 10.9},
            "throughput": {
                "fps": 600.0,
                "frames": 3000,
                "input_mib_per_s": 31.2,
                "output_mib_per_s": 0.6,
                "total_ms": 6435.0,
            },
        },
        "host": {"windows_security": {
            "account": r"RUNNER\ci",
            "applocker_rule_collections": "unavailable",
            "architecture": "64-bit",
            "build_number": "26100",
            "caption": "Microsoft Windows 11 Enterprise",
            "is_administrator": True,
            "product_type": 1,
            "smart_app_control_state": None,
            "version": "10.0.26100",
        }, "python": "3.14.7"},
        "result": "PASS",
        "sima_cli": {
            "commit": "a" * 40,
            "sha256": "b" * 64,
            "version": "2.1.20+ci." + "a" * 12,
        },
        "tests": {
            "tutorial": {
                "cpp": ["run_tensor_mode", "run_image_mode", "run_image_boxdecode"],
                "python": ["run_tensor_mode", "run_image_mode", "run_image_boxdecode"],
            },
            "cpp": [
                "test_metadata", "test_tensor_run", "test_tensor_mla_only_run",
                "test_tensor_mla_only_bf16_run", "test_tensor_parallel_queues",
                "test_image_run", "test_image_boxdecode_run", "test_queue_blocker",
                "test_model_resilience", "test_tensor_latency", "test_tensor_throughput",
                "test_worker_loss_recovery", "test_client_process_death_recovery",
            ],
            "python": [
                "test_async_test_runner_cancels_blocked_peer", "test_metadata_yolov8",
                "test_runtime_request_correlation",
                "test_runtime_two_models_and_independent_unload", "test_tensor_run_yolov8",
                "test_tensor_latency_yolov8", "test_tensor_throughput_yolov8",
                "test_tensor_parallel_queues_yolov8", "test_tensor_run_mla_only",
                "test_image_run_yolov8", "test_image_boxdecode_run_yolov8",
            ],
        },
    }), encoding="utf-8")
    return commit, resources, qualification

  def test_candidate_and_release_gate_bind_exact_bytes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, resources, qualification = self.windows_candidate(root)
      candidate = validate_candidate(root)
      self.assertEqual(candidate.resources, resources)
      admission = verify(root, qualification, commit, True)
      self.assertTrue(admission["hardware_qualified"])
      self.assertTrue(admission["host_security_recorded"])
      self.assertTrue(admission["signatures_valid"])
      self.assertEqual(admission["dependency_signatures"]["platform"], ["sima-platform.dll"])
      candidate.core_msi.write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "checksum differs"):
        validate_candidate(root)

  def test_release_gate_rejects_native_skip_receipt(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      receipt_path = root / "host-candidate.json"
      receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
      receipt["native_tests"]["skipped"] = 1
      receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "zero-skip pass"):
        verify(root, qualification, commit, False)

  def test_card_candidate_binds_exact_linux_payload(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit = "c" * 40
      package = root / "sima-neat-Linux-core.deb"
      package.write_bytes(b"card package")
      resources = {package.name: sha256(package)}
      (root / "metadata-all.json").write_text(json.dumps({
          "name": "gh:sima-neat/core",
          "resources": [package.name],
          "resources-checksum": resources,
          "version": "0.6.2+develop." + commit[:12],
      }), encoding="utf-8")
      candidate = validate_card_candidate(root, commit)
      self.assertEqual(candidate.packages, (package,))
      with self.assertRaisesRegex(ValueError, "not tied"):
        validate_card_candidate(root, "d" * 40)
      package.write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "checksum differs"):
        validate_card_candidate(root, commit)

  def test_private_card_candidate_adds_hashed_arm64_dependencies(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source"
      dependencies = root / "dependencies"
      output = root / "output"
      source.mkdir()
      dependencies.mkdir()
      commit = "c" * 40
      package = source / "sima-neat-Linux-core.deb"
      package.write_bytes(b"card package")
      dependency = dependencies / "libfmt10_arm64.deb"
      dependency.write_bytes(b"dependency package")
      resources = {package.name: sha256(package)}
      (source / "metadata-all.json").write_text(json.dumps({
          "resources": [package.name],
          "resources-checksum": resources,
          "version": "0.6.2+develop." + commit[:12],
      }), encoding="utf-8")
      identity = {
          "architecture": "arm64",
          "file": dependency.name,
          "package": "libfmt10",
          "sha256": sha256(dependency),
          "version": "10.1.1",
      }
      with mock.patch.object(CARD_PREP, "deb_identity", return_value=identity):
        CARD_PREP.prepare(source, dependencies, output, commit)
      candidate = validate_card_candidate(output, commit)
      self.assertEqual(
          {path.name for path in candidate.packages}, {package.name, dependency.name})
      receipt = json.loads(
          (output / "windows-card-qualification-dependencies.json").read_text(
              encoding="utf-8"))
      self.assertEqual(receipt["dependencies"], [identity])
      self.assertEqual(receipt["source_core_commit"], commit)

  def test_release_rejects_incomplete_card_candidate_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _resources, qualification = self.windows_candidate(root)
      receipt = json.loads(qualification.read_text(encoding="utf-8"))
      receipt["card_candidate"]["firmware"] = "neat-pcie-card0-"
      qualification.write_text(json.dumps(receipt), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "exact Linux card candidate"):
        verify(root, qualification, commit, False)

  def test_card_sudo_secret_stays_off_command_line_and_log(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      key = root / "id_ed25519"
      key.write_text("fixture", encoding="ascii")
      args = SimpleNamespace(
          card_host="10.0.0.2", card_sudo_password_env="CARD_TEST_SECRET",
          card_user="sima", ssh_key=key)
      completed = subprocess.CompletedProcess([], 0, "")
      log = root / "sudo.log"
      with mock.patch.dict(os.environ, {"CARD_TEST_SECRET": "fixture-secret"}), \
          mock.patch("qualify_windows_hardware.subprocess.run", return_value=completed) as run:
        run_sudo_remote(args, "true", log)
      command = run.call_args.args[0]
      self.assertNotIn("fixture-secret", " ".join(command))
      self.assertEqual(run.call_args.kwargs["input"], "fixture-secret\n")
      self.assertNotIn("fixture-secret", log.read_text(encoding="utf-8"))

  def test_card_candidate_bad_extraction_receipt_removes_private_stage(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit = "c" * 40
      package = root / "sima-neat-Linux-core.deb"
      package.write_bytes(b"card package")
      (root / "metadata-all.json").write_text(json.dumps({
          "resources": [package.name],
          "resources-checksum": {package.name: sha256(package)},
          "version": "0.6.2+develop." + commit[:12],
      }), encoding="utf-8")
      candidate = validate_card_candidate(root, commit)
      key = root / "id_ed25519"
      key.write_text("fixture", encoding="ascii")
      args = SimpleNamespace(
          card_host="10.0.0.2", card_user="sima", include_genai=False, ssh_key=key)
      evidence = root / "evidence"
      evidence.mkdir()

      def write_bad_receipt(_command, log, **_kwargs):
        log.write_text("worker_sha256=not-a-digest\n", encoding="utf-8")

      completed = subprocess.CompletedProcess([], 0, "")
      with mock.patch(
          "qualify_windows_hardware.run_logged", side_effect=write_bad_receipt), \
          mock.patch(
              "qualify_windows_hardware.subprocess.run", return_value=completed) as cleanup:
        with self.assertRaisesRegex(RuntimeError, "receipt is incomplete"):
          stage_card_candidate(args, candidate, "4" * 32, evidence)
      self.assertIn(
          "rm -rf -- /tmp/neat-pcie-card0." + "4" * 32,
          cleanup.call_args.args[0][-1])

  def test_card_candidate_receipt_binds_private_genai_worker(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit = "c" * 40
      package = root / "sima-neat-Linux-core.deb"
      package.write_bytes(b"card package")
      (root / "metadata-all.json").write_text(json.dumps({
          "resources": [package.name],
          "resources-checksum": {package.name: sha256(package)},
          "version": "0.6.2+develop." + commit[:12],
      }), encoding="utf-8")
      candidate = validate_card_candidate(root, commit)
      key = root / "id_ed25519"
      key.write_text("fixture", encoding="ascii")
      args = SimpleNamespace(
          card_host="10.0.0.2", card_user="sima", include_genai=True, ssh_key=key)
      evidence = root / "evidence"
      evidence.mkdir()

      def write_receipt(command, log, **_kwargs):
        if log.name == "card-candidate-extract.log":
          self.assertIn("genai_worker_sha256", command[-1])
          log.write_text(
              f"worker_sha256={'2' * 64}\nruntime_sha256={'3' * 64}\n"
              f"genai_worker_sha256={'4' * 64}\n",
              encoding="utf-8")
        else:
          log.write_text("", encoding="utf-8")

      with mock.patch(
          "qualify_windows_hardware.run_logged", side_effect=write_receipt):
        staged = stage_card_candidate(args, candidate, "5" * 32, evidence)
      self.assertEqual(staged["genai_worker_sha256"], "4" * 64)
      self.assertIn("SIMA_NEAT_PCIE_GENAI_HELPER_PATH=", staged["genai_card_env"])

  def test_installed_tutorial_facade_uses_private_card_candidate(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      args = SimpleNamespace(card_user="sima")
      staged = {
          "root": "/tmp/neat-pcie-card0.fixture/root",
          "worker": "/tmp/neat-pcie-card0.fixture/root/usr/bin/pcie-pipeline-builder",
      }
      with mock.patch(
          "qualify_windows_hardware.run_sudo_remote") as run:
        stage_installed_tutorial_facade(args, staged, root)
      command = run.call_args.args[1]
      self.assertIn("test ! -e /usr/bin/pcie-pipeline-builder", command)
      self.assertIn(staged["worker"], command)
      self.assertIn(
          staged["root"] + "/usr/lib/aarch64-linux-gnu/neat/runtime", command)
      self.assertIn(
          staged["root"] + "/usr/lib/aarch64-linux-gnu/neat/gst-plugins", command)
      self.assertIn('"$@"', command)
      self.assertEqual(run.call_args.args[2], root / "tutorial-card-facade-stage.log")

  def test_installed_tutorial_facade_cleanup_is_exact(self):
    with tempfile.TemporaryDirectory() as directory:
      log = Path(directory) / "cleanup.log"
      with mock.patch(
          "qualify_windows_hardware.run_sudo_remote") as run:
        remove_installed_tutorial_facade(
            SimpleNamespace(), {"root": "/tmp/neat-pcie-card0.fixture/root"}, log)
      command = run.call_args.args[1]
      self.assertIn("readlink /usr/bin/pcie-pipeline-builder", command)
      self.assertIn("rm -- /usr/bin/pcie-pipeline-builder", command)
      self.assertNotIn("rm -rf", command)
      self.assertEqual(run.call_args.args[2], log)

  def test_tutorial_facade_preserves_existing_paths_and_reconciles_partial_stage(self):
    # Execute the actual POSIX commands in a disposable tree, not on a card.
    for existing in ("file", "directory", "dangling-link", "none", "partial"):
      with self.subTest(existing=existing), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        private_root = root / "candidate"
        private_root.mkdir()
        facade = private_root / "tutorial-facade"
        destinations = [root / "bin" / "worker", root / "run" / "pcie", root / "log" / "pcie"]
        links = dict(zip((path.as_posix() for path in destinations),
                         ((facade / name).as_posix() for name in ("worker", "run", "log"))))
        destinations[0].parent.mkdir()
        if existing == "file":
          destinations[0].write_text("stock-worker")
        elif existing == "directory":
          destinations[1].mkdir(parents=True)
          (destinations[1] / "stock-state").write_text("stock")
        elif existing == "dangling-link":
          destinations[0].symlink_to(root / "missing-stock-target")
        staged = {"root": private_root.as_posix(),
                  "worker": (private_root / "real-worker").as_posix()}
        args = SimpleNamespace(card_user=getpass.getuser())

        def execute(_args, command, _log):
          if existing == "partial" and "ln -sT" in command:
            # Simulate loss of the SSH result after one link was created.
            command = command.split("ln -sT", 2)[0] + "ln -sT" + command.split("ln -sT", 2)[1]
            command += " false"
          if os.name == "nt":
            # Git for Windows provides the POSIX filesystem tools used by this
            # fixture, but its minimal distribution does not include chown or
            # install. The disposable tree already belongs to the test user;
            # model ownership as a no-op and preserve install's directory/mode
            # behavior without weakening the generated card command.
            command = """\
chown() { :; }
chmod() { :; }
ln() {
  [ "$1" = -sT ] || return 2
  shift
  "$WINDOWS_FIXTURE_PYTHON" -c \
    'import os, sys; os.symlink(sys.argv[1], sys.argv[2], target_is_directory=os.path.isdir(sys.argv[1]))' \
    "$1" "$2"
}
readlink() {
  "$WINDOWS_FIXTURE_PYTHON" -c \
    'import os, sys; slash=chr(92); print(os.readlink(sys.argv[1]).removeprefix(slash*2+"?"+slash).replace(slash, "/"))' \
    "$1"
}
install() {
  mode=
  while [ "$#" -gt 0 ]; do
    case "$1" in
      -d) shift ;;
      -o|-g) shift 2 ;;
      -m) mode="$2"; shift 2 ;;
      --) shift; break ;;
      -*) return 2 ;;
      *) break ;;
    esac
  done
  mkdir -p -- "$@" || return
  [ -z "$mode" ] || chmod "$mode" "$@"
}
""" + command
          result = run_posix(command)
          if result.returncode:
            raise RuntimeError(result.stderr or "injected disconnect")

        with mock.patch("qualify_windows_hardware.tutorial_facade_links", return_value=links), \
             mock.patch("qualify_windows_hardware.run_sudo_remote", side_effect=execute):
          if existing == "none":
            stage_installed_tutorial_facade(args, staged, root)
            self.assertTrue(all(path.is_symlink() for path in destinations))
          else:
            with self.assertRaises(RuntimeError):
              stage_installed_tutorial_facade(args, staged, root)
          remove_installed_tutorial_facade(args, staged, root / "cleanup.log")
          remove_installed_tutorial_facade(args, staged, root / "cleanup-repeat.log")
        if existing == "file":
          self.assertEqual(destinations[0].read_text(), "stock-worker")
        elif existing == "directory":
          self.assertEqual((destinations[1] / "stock-state").read_text(), "stock")
        elif existing == "dangling-link":
          self.assertEqual(
              normalized_link_target(destinations[0]),
              os.path.normcase(os.path.normpath(root / "missing-stock-target")))
        else:
          self.assertFalse(any(path.exists() or path.is_symlink() for path in destinations))

  def test_candidate_rejects_changed_qualification_resource(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.windows_candidate(root)
      (root / "qualification-resource.bin").write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "qualification resource checksum differs"):
        validate_candidate(root)

  def test_candidate_requires_dependency_signature_receipts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.windows_candidate(root)
      receipt_path = root / "host-candidate.json"
      receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
      del receipt["qualification_resources"]["scheduler-windows-signatures.json"]
      receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "missing required qualification resources"):
        validate_candidate(root)

  def test_candidate_dependency_identity_is_canonical(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.windows_candidate(root)
      identity = dependency_identity(root)
      self.assertEqual(
          identity["platform_runtime_dlls"],
          [{"name": "sima-platform.dll", "sha256": "d" * 64}])
      self.assertEqual(
          identity["gstreamer"]["runtime_dlls"],
          [{"name": "gstreamer-1.0-0.dll", "sha256": "f" * 64}])

  def test_release_gate_rejects_unsigned_candidate(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root, signed=False)
      with self.assertRaisesRegex(ValueError, "not built in signed release mode"):
        verify(root, qualification, commit, True)

  def test_release_gate_requires_real_package_upgrade(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      del value["customer_install"]["upgrade"]
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "real MSI package upgrade"):
        verify(root, qualification, commit, True, require_release=True)
      # Signing can also be checked in a non-release validation run.
      admission = verify(root, qualification, commit, True)
      self.assertFalse(admission["release_gates_required"])

  def test_unsigned_release_requires_upgrade_without_requiring_signatures(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root, signed=False)
      admission = verify(root, qualification, commit, False, require_release=True)
      self.assertTrue(admission["release_gates_required"])
      self.assertFalse(admission["signatures_required"])
      self.assertFalse(admission["signatures_valid"])
      value = json.loads(qualification.read_text(encoding="utf-8"))
      del value["customer_install"]["upgrade"]
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "real MSI package upgrade"):
        verify(root, qualification, commit, False, require_release=True)

  def test_release_gate_requires_partial_wheel_failure_recovery(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      del value["customer_install"]["wheel_failure"]
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "native success, wheel failure, and retry"):
        verify(root, qualification, commit, True)

  def test_upgrade_baseline_must_name_a_different_package_version(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      candidate_root = root / "candidate"
      baseline_root = root / "baseline"
      candidate_root.mkdir()
      baseline_root.mkdir()
      candidate_commit, _, _ = self.windows_candidate(
          candidate_root, commit="c" * 40, version="0.6.2")
      baseline_commit, _, _ = self.windows_candidate(
          baseline_root, commit="b" * 40, version="0.6.1")
      candidate = validate_candidate(candidate_root)
      baseline = load_upgrade_base(candidate, baseline_root, baseline_commit)
      self.assertEqual(baseline.core_commit, baseline_commit)
      self.assertNotEqual(candidate_commit, baseline_commit)
      with self.assertRaisesRegex(ValueError, "does not match"):
        load_upgrade_base(candidate, baseline_root, "a" * 40)
      baseline.metadata["version"] = candidate.metadata["version"]
      (baseline_root / "metadata.json").write_text(
          json.dumps(baseline.metadata), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "newer than the baseline"):
        load_upgrade_base(candidate, baseline_root, baseline_commit)
      baseline.metadata["version"] = "0.6.3"
      (baseline_root / "metadata.json").write_text(
          json.dumps(baseline.metadata), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "newer than the baseline"):
        load_upgrade_base(candidate, baseline_root, baseline_commit)

  def test_release_gate_rejects_unsigned_platform_dependency(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      dependencies_path = root / "windows-dependencies.json"
      dependencies = json.loads(dependencies_path.read_text(encoding="utf-8"))
      dependencies["platform"]["runtime_dlls"][0]["signature"]["status"] = "NotSigned"
      dependencies_path.write_text(json.dumps(dependencies), encoding="utf-8")
      receipt_path = root / "host-candidate.json"
      receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
      receipt["qualification_resources"][dependencies_path.name] = sha256(dependencies_path)
      receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Platform runtime has invalid"):
        verify(root, qualification, commit, True)

  def test_release_gate_rejects_changed_hardware_dependency_cohort(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["dependencies"]["platform_runtime_dlls"][0]["sha256"] = "0" * 64
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "different runtime dependency cohort"):
        verify(root, qualification, commit, True)

  def test_release_gate_rejects_unsigned_scheduler_payload(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      scheduler_path = root / "scheduler-windows-signatures.json"
      scheduler = json.loads(scheduler_path.read_text(encoding="utf-8"))
      scheduler["release_required"] = False
      scheduler_path.write_text(json.dumps(scheduler), encoding="utf-8")
      receipt_path = root / "host-candidate.json"
      receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
      receipt["qualification_resources"][scheduler_path.name] = sha256(scheduler_path)
      receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Scheduler candidate was not built"):
        verify(root, qualification, commit, True)

  def test_release_gate_rejects_missing_genai_matrix(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      with self.assertRaisesRegex(ValueError, "complete Windows GenAI matrix"):
        verify(root, qualification, commit, True, True)

  def test_release_gate_enforces_linux_performance_thresholds(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["performance"]["latency"]["mean_ms"] = 12.1
      value["performance"]["throughput"]["fps"] = 439.9
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Linux PCIe performance gate"):
        verify(root, qualification, commit, False)

  def test_release_gate_rejects_incomplete_vision_matrix(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["tests"]["cpp"].remove("test_tensor_parallel_queues")
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "missing required vision cases"):
        verify(root, qualification, commit, True)

  def test_release_gate_rejects_incomplete_installed_tutorial_matrix(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["tests"]["tutorial"]["python"].remove("run_image_boxdecode")
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "installed C[+][+] and Python tutorials"):
        verify(root, qualification, commit, True)

  def test_release_gate_rejects_server_or_unsupported_python(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["host"]["windows_security"]["product_type"] = 3
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Windows 11 x64 workstation"):
        verify(root, qualification, commit, True)
      value["host"]["windows_security"]["product_type"] = 1
      value["host"]["python"] = "3.13.9"
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "supported CPython 3.14 ABI"):
        verify(root, qualification, commit, True)

  def test_release_gate_requires_unchanged_platform_3_card(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["card_final"]["platform_sha256"] = "0" * 64
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Platform 3.0 card baseline"):
        verify(root, qualification, commit, True)
      value["card_final"]["platform_sha256"] = value["card_baseline"]["platform_sha256"]
      value["card_baseline"]["platform_release"] = "2.1.3_B4837"
      value["card_final"]["platform_release"] = "2.1.3_B4837"
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Platform 3.0 card baseline"):
        verify(root, qualification, commit, True)

  def test_release_gate_requires_structured_performance_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["performance"]["throughput"]["fps"] = 0
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "valid performance measurements"):
        verify(root, qualification, commit, True)

  def test_producer_retains_cpp_hardware_extras_and_signature_receipt(self):
    source = SCRIPT.read_text(encoding="utf-8")
    self.assertIn('"PcieHostExtras"', source)
    self.assertIn('"windows-signatures.json"', source)
    self.assertIn("signing_certificate_sha1", source)

  def test_signing_inputs_fail_before_signtool(self):
    with self.assertRaisesRegex(ValueError, "certificate SHA-1"):
      MODULE.sign_files([], "signtool", "not-a-thumbprint", "https://timestamp.example", False)
    with self.assertRaisesRegex(ValueError, "must use HTTPS"):
      MODULE.sign_files([], "signtool", "A" * 40, "http://timestamp.example", False)

  def test_signing_selects_only_installed_native_outputs(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      build = root / "build"
      stage = root / "stage"
      build.mkdir()
      stage.mkdir()
      selected = build / "customer.dll"
      selected.write_bytes(b"dll")
      (build / "unit-test.exe").write_bytes(b"test")
      (stage / selected.name).write_bytes(b"dll")
      self.assertEqual(MODULE.build_sources_for_installed(build, [stage]), [selected])

  def test_hardware_lock_is_atomic_and_can_be_retained_for_recovery(self):
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "card.lock"
      with ExclusiveLock(path):
        self.assertTrue(path.is_file())
        with self.assertRaisesRegex(RuntimeError, "already exists"):
          with ExclusiveLock(path):
            pass
      self.assertFalse(path.exists())
      with ExclusiveLock(path) as lock:
        lock.retain()
      self.assertTrue(path.is_file())
      path.unlink()
      with ExclusiveLock(path) as lock:
        lock.retain()
        lock.release_on_exit()
      self.assertFalse(path.exists())

  def test_card_side_lock_is_owned_and_released_by_token(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      key = root / "id_ed25519"
      key.write_text("fixture", encoding="ascii")
      args = SimpleNamespace(card_host="10.0.0.2", card_user="sima", ssh_key=key)
      token = "a" * 32
      remote_lock = RemoteExclusiveLock(args, "/tmp/neat-pcie-card0.lock", token)
      completed = subprocess.CompletedProcess([], 0, "")
      with mock.patch(
          "qualify_windows_hardware.run_captured_to_file",
          return_value=completed) as run:
        remote_lock.acquire(root / "acquire.log")
        remote_lock.release(root / "release.log")
      self.assertEqual(run.call_count, 2)
      acquire_command = run.call_args_list[0].args[0][-1]
      release_command = run.call_args_list[1].args[0][-1]
      self.assertIn("mkdir", acquire_command)
      self.assertIn(token, acquire_command)
      self.assertIn('test "$owner" = "$token"', release_command)

      owner = subprocess.CompletedProcess([], 0, f"owner={token}\n")
      with mock.patch(
          "qualify_windows_hardware.run_captured_to_file", return_value=owner):
        self.assertEqual(remote_lock.owner(root / "owner.log"), token)

  def test_card_ssh_capture_uses_a_file_instead_of_a_pipe(self):
    with tempfile.TemporaryDirectory() as directory:
      log = Path(directory) / "ssh.log"

      def complete(_command, **kwargs):
        self.assertIs(kwargs["stdin"], subprocess.DEVNULL)
        self.assertIsNot(kwargs["stdout"], subprocess.PIPE)
        kwargs["stdout"].write("owner=" + "a" * 32 + "\n")
        kwargs["stdout"].flush()
        return subprocess.CompletedProcess([], 0)

      with mock.patch(
          "qualify_windows_hardware.subprocess.run", side_effect=complete):
        completed = run_captured_to_file(["ssh.exe", "host", "true"], log, timeout=30)
      self.assertEqual(completed.returncode, 0)
      self.assertIn("owner=" + "a" * 32, completed.stdout)
      self.assertNotIn("ssh.exe", completed.stdout)
      self.assertIn("ssh.exe", log.read_text(encoding="utf-8"))

  def test_card_snapshot_records_platform_3_build_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      key = root / "id_ed25519"
      key.write_text("fixture", encoding="ascii")
      args = SimpleNamespace(
          card_host="10.0.0.2",
          card_user="sima",
          include_genai=False,
          ssh_key=key,
      )

      def write_snapshot(command, log, **_kwargs):
        text = (
            "machine=aarch64\nplatform_release=3.0.0_master_B2501\n"
            f"platform_sha256={'1' * 64}\nfirmware=modalix-cvu-fw\n"
            "remoteproc_state=running\nworkers=0\ngenai_workers=0\n")
        if "runtime_sha256" in command[-1]:
          text += f"runtime_sha256={'2' * 64}\nworker_sha256={'3' * 64}\n"
        if "genai_worker_sha256" in command[-1]:
          text += f"genai_worker_sha256={'4' * 64}\n"
        log.write_text(text, encoding="utf-8")

      with mock.patch(
          "qualify_windows_hardware.run_logged", side_effect=write_snapshot):
        snapshot = card_snapshot(args, root, "baseline")
      self.assertEqual(snapshot["platform_release"], "3.0.0_master_B2501")
      self.assertEqual(snapshot["platform_sha256"], "1" * 64)
      with mock.patch(
          "qualify_windows_hardware.run_logged", side_effect=write_snapshot):
        active = card_snapshot(
            args, root, "candidate",
            runtime_path="/tmp/candidate/modalix-cvu-fw",
            worker_path="/tmp/candidate/pcie-pipeline-builder",
            expected_runtime_sha256="2" * 64,
            expected_worker_sha256="3" * 64,
            genai_worker_path="/tmp/candidate/neat-pcie-genai-worker",
            expected_genai_worker_sha256="4" * 64)
      self.assertEqual(active["worker_sha256"], "3" * 64)
      self.assertEqual(active["genai_worker_sha256"], "4" * 64)

      def write_old_snapshot(command, log, **kwargs):
        write_snapshot(command, log, **kwargs)
        text = log.read_text(encoding="utf-8").replace(
            "3.0.0_master_B2501", "2.1.3_B4837")
        log.write_text(text, encoding="utf-8")

      with mock.patch(
          "qualify_windows_hardware.run_logged", side_effect=write_old_snapshot):
        with self.assertRaisesRegex(RuntimeError, "not running Modalix Platform 3.0"):
          card_snapshot(args, root, "old-platform")

  def test_hardware_path_fixtures_use_a_directory_with_spaces(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      environment = {}
      for name in ("SIMAPCIE_YOLOV8_MODEL", "SIMAPCIE_TEST_IMAGE",
                   "SIMAPCIE_BOXDECODE_IMAGE"):
        source = root / f"{name}.bin"
        source.write_bytes(name.encode())
        environment[name] = str(source)
      stage_path_fixtures(environment, root / "path fixtures with spaces")
      for value in environment.values():
        self.assertIn("path fixtures with spaces", value)
        self.assertTrue(Path(value).is_file())

  def test_hardware_stress_queues_are_unique_and_bounded(self):
    self.assertEqual(stress_queues({"SIMAPCIE_STRESS_QUEUES": "0 2 5"}), [0, 2, 5])
    with self.assertRaisesRegex(ValueError, "range 0..5"):
      stress_queues({"SIMAPCIE_STRESS_QUEUES": "0 6"})
    with self.assertRaisesRegex(ValueError, "duplicate"):
      stress_queues({"SIMAPCIE_STRESS_QUEUES": "1 1"})

  def test_hardware_performance_logs_become_structured_evidence(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      latency = root / "latency.log"
      latency.write_text(
          "latency results\n  count=1000\n  mean_ms=10.900 stddev_ms=0.100\n"
          "  total_ms=10900.000 effective_fps=91.743\n",
          encoding="utf-8")
      throughput = root / "throughput.log"
      throughput.write_text(
          "throughput results\n  frames=1000 total_ms=6435.000 fps=155.400\n"
          "  input_mib=200 input_mib_per_s=31.200\n"
          "  output_mib=4 output_mib_per_s=0.600\n",
          encoding="utf-8")
      self.assertEqual(parse_performance_log(latency, "latency")["count"], 1000)
      self.assertEqual(parse_performance_log(throughput, "throughput")["fps"], 155.4)
      throughput.write_text("throughput results\n  frames=0\n", encoding="utf-8")
      with self.assertRaisesRegex(RuntimeError, "not positive and finite"):
        parse_performance_log(throughput, "throughput")

  def test_customer_cli_candidate_and_local_registry_are_exact(self):
    import zipfile
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit = "a" * 40
      wheel = root / "sima_cli-2.1.20-py3-none-any.whl"
      with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "sima_cli-2.1.20.dist-info/METADATA",
            f"Name: sima-cli\nVersion: 2.1.20+ci.{commit[:12]}\n")
      self.assertEqual(sima_cli_identity(wheel, commit)["commit"], commit)
      with self.assertRaisesRegex(ValueError, "not tied"):
        sima_cli_identity(wheel, "b" * 40)

      candidate_root = root / "candidate"
      candidate_root.mkdir()
      self.windows_candidate(candidate_root)
      candidate = validate_candidate(candidate_root)
      registry = root / "registry"
      package = stage_local_registry(candidate, registry)
      self.assertEqual(
          (registry / "core" / "3.0" / "latest.tag").read_text().strip(),
          candidate.core_commit)
      self.assertTrue((package / "metadata.json").is_file())
      self.assertEqual(
          {path.name for path in package.iterdir()},
          {"metadata.json", *candidate.resources})

      invalid_registry = root / "invalid-registry"
      invalid_package = stage_invalid_wheel_registry(candidate, invalid_registry)
      invalid_metadata = json.loads(
          (invalid_package / "metadata.json").read_text(encoding="utf-8"))
      invalid_wheel = invalid_package / candidate.wheel.name
      self.assertNotEqual(sha256(invalid_wheel), candidate.resources[candidate.wheel.name])
      self.assertEqual(
          invalid_metadata["resources-checksum"][candidate.wheel.name],
          sha256(invalid_wheel))

  def test_linux_release_gate_binds_success_to_exact_commit(self):
    repository = "sima-neat/core"
    commit = "e" * 40
    run = {
        "conclusion": "success",
        "event": "workflow_dispatch",
        "head_repository": {"full_name": repository},
        "head_sha": commit,
        "html_url": "https://github.example/run/42",
        "id": 42,
        "name": "Vulcan CI",
        "path": ".github/workflows/vulcan-ci.yml",
        "run_attempt": 1,
        "status": "completed",
    }
    self.assertEqual(
        verify_linux_run(run, repository, 42, commit)["head_sha"], commit)
    changed = dict(run, head_sha="f" * 40)
    with self.assertRaisesRegex(ValueError, "head_sha"):
      verify_linux_run(changed, repository, 42, commit)


if __name__ == "__main__":
  unittest.main()
