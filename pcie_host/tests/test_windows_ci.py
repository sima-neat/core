#!/usr/bin/env python3
"""Contracts for the native Windows Core PCIe-host producer."""

import ast
import base64
import contextlib
import io
from functools import partial
import importlib.util
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from urllib.parse import quote, unquote
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "pciehost" / "build_windows.py"
SCRIPT_DIR = SCRIPT.parent
GENAI_TEST_DIR = ROOT / "pcie_host" / "python" / "tests"
WORKFLOW = ROOT / ".github" / "workflows" / "pcie-host-windows.yml"
MAIN_WORKFLOW = ROOT / ".github" / "workflows" / "vulcan-ci.yml"
WORKFLOW_BASH_SKIP = (
    "workflow bash steps run on Linux runners; executed by the Linux contract job")
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(GENAI_TEST_DIR))
SPEC = importlib.util.spec_from_file_location("build_windows", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(MODULE)
from windows_candidate import (  # noqa: E402
    dependency_identity, installed_neat_status, require_junit_cases, sha256,
    validate_candidate)
from windows_hardware_assets import (  # noqa: E402
    GENAI_ASSETS, VISION_ASSETS, asset_manifest_sha256, create_asset_manifest,
    normalize_asset_manifest, verify_asset_manifest)
from qualify_windows_hardware import (  # noqa: E402
    _windows_openssh_batch_line,
    CARD_BUNDLE_ARCHIVE, CARD_DEPLOY, CARD_STAGE_PATTERN,
    ExclusiveLock, RemoteExclusiveLock,
    card_snapshot,
    card_management_addresses,
    create_standard_user_python, customer_process_environment, install_card_candidate,
    load_upgrade_base,
    isolated_sima_cli_environment,
    parse_performance_log, require_powershell as require_hardware_powershell,
    prepare_openssh_secret_directory,
    remove_card_stage,
    require_restored_windows_host_state,
    recover_windows_platform_device,
    remove_qualification_tree,
    run_captured_to_file, run_sudo_remote,
    run_windows_openssh,
    sima_cli_identity,
    stage_invalid_wheel_registry, stage_local_registry, stage_path_fixtures,
    standard_user_wrapper, stress_queues, validate_card_bundle,
    verify_customer_process_activation,
    windows_card_network_identity, windows_platform_state_snapshot)
from standard_user_genai import (  # noqa: E402
    prove_writable_sentinel, read_serve_mapping, run as run_standard_user_genai)
from verify_linux_release_run import verify_run as verify_linux_run  # noqa: E402
from verify_windows_release import verify  # noqa: E402
import qualify_windows_hardware as HARDWARE  # noqa: E402
import prepare_card_qualification_candidate as CARD_PREP  # noqa: E402
import cleanup_windows_hardware as RECOVERY  # noqa: E402
import acquire_windows_dependencies as WINDOWS_DEPENDENCIES  # noqa: E402
import windows_release as RELEASE  # noqa: E402
import standard_user_genai as STANDARD_USER_GENAI  # noqa: E402
from genai_admission_contract import (  # noqa: E402
    is_bounded_competing_session_rejection, is_injected_transport_failure)


def workflow_job(source, name):
  return re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", source)[1]


def job_condition(source, name):
  return re.search(r"(?m)^    if: (.+)$", workflow_job(source, name))[1]


def run_windows_preflight(**inputs):
  step = WORKFLOW.read_text(encoding="utf-8").split(
      "- name: Validate dependency and admission selectors\n", 1)[1]
  script = textwrap.dedent(step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0])
  environment = os.environ.copy()
  environment.update({name: "" for name in (
      "SCHEDULER_RUN_ID", "SCHEDULER_COMMIT", "SIMA_CLI_RUN_ID", "SIMA_CLI_COMMIT",
      "LINUX_RUN_ID", "REQUIRE_UPGRADE", "UPGRADE_BASE_TAG",
      "FIRST_RELEASE_UPGRADE_EXEMPTION", "FIRST_RELEASE_UPGRADE_EXEMPTION_REASON")})
  environment.update({"GITHUB_WORKFLOW": "Core native Windows PCIe host",
                      "LINUX_CURRENT_RUN": "false", **inputs})
  return subprocess.run([shutil.which("bash"), "-e", "-c", script], capture_output=True,
                        text=True, check=False, env=environment)


def fixture_platform_snapshot():
  service = {
      "configuration": {
          "DelayedAutoStart": 0,
          "FailureActions": [128, 81, 1, 0],
          "ImagePath": r'"C:\Program Files\SiMa\simaai_mla_daemon.exe" --service',
          "ObjectName": "LocalSystem",
          "Start": 2,
          "Type": 16,
      },
      "display_name": "SiMa.ai MLA Daemon",
      "name": "SiMaMlaDaemon",
      "path_name": r'"C:\Program Files\SiMa\simaai_mla_daemon.exe" --service',
      "service_type": "Own Process",
      "start_mode": "Auto",
      "start_name": "LocalSystem",
      "state": "Running",
  }
  devices = []
  for name, instance, device_class in (
      ("SiMa.ai MLA Device", r"PCI\VEN_1F06&DEV_3123\4&1", "System"),
      ("SiMa.ai Virtual Ethernet Adapter #2", r"ROOT\SIMA_VETH\0000", "Net"),
  ):
    devices.append({
        "class": device_class,
        "driver_inf": "oem42.inf",
        "driver_provider": "SiMa.ai",
        "driver_signer": "SiMa.ai Corporation",
        "driver_version": "3.0.0.2",
        "friendly_name": name,
        "instance_id": instance,
        "is_signed": True,
        "manufacturer": "SiMa.ai",
        "status": "OK",
    })
  return {"pnp_devices": devices, "service": service}


def card_args(directory, **options):
  key = directory / "id_ed25519"
  key.write_text("fixture", encoding="ascii")
  return SimpleNamespace(card_host="10.0.0.2", card_user="sima", ssh_key=key, **options)


WINDOWS_OPENSSH_HOP_SKIP = (
    "replays the Windows OpenSSH hop into a bash card shell; runs on Linux")
WINDOWS_CMD_SKIP = "runs a batch wrapper under cmd.exe; executed on Windows"
GENAI_CPP_CASES = frozenset({"test_genai_llm_run", "test_genai_vlm_run", "test_genai_asr_run"})
GENAI_PYTHON_CASES = frozenset({
    "test_genai_asr_file_pcm_translation_and_stream",
    "test_genai_bounded_cancellation_soak_and_context_cleanup",
    "test_genai_competing_sessions_are_isolated",
    "test_genai_llm_run_stream_history_and_cancel",
    "test_genai_request_timeout_and_restart_after_failure",
    "test_genai_restart_after_service_loss", "test_genai_restart_after_worker_loss",
    "test_genai_service_loss_recovery", "test_genai_vlm_run_and_stream",
    "test_genai_worker_loss_recovery", "test_genai_workers_drained",
})
FIXTURE_SUDO = """\
#!/bin/sh
# sudo -n reports FIXTURE_SUDO_N_STATUS; sudo -S checks the piped secret.
if [ "$1" = -n ]; then exit "$FIXTURE_SUDO_N_STATUS"; fi
if [ "$1" = -S ]; then
  IFS= read -r secret
  [ "$secret" = "$FIXTURE_SUDO_SECRET" ] || exit 70
  shift 3
fi
[ "$1" = sh ] && [ "$2" = -c ] || exit 71
if [ -n "${FIXTURE_ROOT_SCRIPT:-}" ]; then printf '%s' "$3" > "$FIXTURE_ROOT_SCRIPT"; exit 0; fi
exec "$@"
"""


def powershell51_native_command_line(arguments):
  """Join arguments the way Windows PowerShell 5.1 starts a native command.

  An argument with whitespace outside double quotes is wrapped in double quotes;
  embedded double quotes and backslashes are never escaped.
  """
  joined = []
  for argument in arguments:
    quotes, after_backslash, needs_quotes = 0, False, False
    for character in argument:
      if character == '"' and not after_backslash:
        quotes += 1
      elif character.isspace() and quotes % 2 == 0:
        needs_quotes = True
      after_backslash = character == "\\"
    joined.append(f'"{argument}"' if needs_quotes else argument)
  return " ".join(joined)


def windows_runtime_argv(command_line):
  """Split a command line with the Microsoft C runtime rules ssh.exe uses."""
  arguments, index, length = [], 0, len(command_line)
  while True:
    while index < length and command_line[index] in " \t":
      index += 1
    if index == length:
      return arguments
    argument, quoted = [], False
    while index < length and (quoted or command_line[index] not in " \t"):
      backslashes = 0
      while index < length and command_line[index] == "\\":
        backslashes += 1
        index += 1
      if index < length and command_line[index] == '"':
        argument.append("\\" * (backslashes // 2))
        if backslashes % 2:
          argument.append('"')
        elif quoted and command_line[index + 1:index + 2] == '"':
          argument.append('"')
          index += 1
        else:
          quoted = not quoted
        index += 1
      elif backslashes:
        argument.append("\\" * backslashes)
      else:
        argument.append(command_line[index])
        index += 1
    arguments.append("".join(argument))


class WindowsOpenSshHop:
  """Stand-in for subprocess.Popen that replays the qualifier's OpenSSH wrapper.

  cmd.exe runs the batch line, Windows PowerShell 5.1 relays the argv to ssh.exe,
  ssh.exe joins its remote words, and the card's login shell (bash) runs them
  with the wrapper's input file as stdin and its output file as stdout/stderr.
  """

  real_popen = subprocess.Popen

  def __init__(self, root):
    self.bin = root / "card-bin"
    self.bin.mkdir()
    (self.bin / "sudo").write_text(FIXTURE_SUDO, encoding="utf-8")
    (self.bin / "sudo").chmod(0o755)
    self.card = root / "card-home"
    self.card.mkdir()
    self.environment = {}
    self.pid = 41
    self.returncode = None
    self.line = ""

  def __call__(self, command, **_kwargs):
    self.line = Path(command[-1]).read_bytes().decode("utf-8").split("\r\n")[1]
    return self

  def communicate(self, timeout=None):
    match = re.fullmatch(
        r"powershell\.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand (\S+)"
        r" > (\S+) 2>&1(?: < (\S+))?", self.line.replace("%%", "%"))
    assert match, self.line
    script = base64.b64decode(match[1]).decode("utf-16le")
    array = script.split("$argv=@(", 1)[1].split(");$exe=", 1)[0]
    relayed = [value.replace("''", "'")
               for value in re.findall(r"'((?:[^']|'')*)'", array)]
    argv = windows_runtime_argv(powershell51_native_command_line(relayed))
    index = 1
    while argv[index].startswith("-"):
      index += 2 if argv[index] in ("-i", "-o") else 1
    remote = " ".join(argv[index + 1:])
    stdin = Path(match[3]).read_bytes() if match[3] else b""
    environment = {
        "PATH": os.pathsep.join((str(self.bin), os.environ["PATH"])),
        "SHELL": shutil.which("bash"), **self.environment}
    with self.real_popen(
        [shutil.which("bash"), "-euc", remote], cwd=self.card, env=environment,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT) as card:
      output = card.communicate(stdin, timeout=timeout)[0]
    Path(match[2]).write_bytes(output)
    self.returncode = card.returncode
    return None, None

  def poll(self):
    return self.returncode


def workflow_step(body, name):
  """Return one step of a workflow job, from its name to the next step or job."""
  step = body.split(f"      - name: {name}\n", 1)[1]
  return re.split(r"(?m)^      - |^  [\w-]+:\n", step, maxsplit=1)[0]


def evaluate_actions_expression(expression, values):
  """Evaluate the boolean subset used by the workflow admission contracts."""
  expression = expression.strip()
  if expression.startswith("${{") and expression.endswith("}}"):
    expression = expression[3:-2].strip()
  for token, value in sorted(
      values.items(), key=lambda item: len(item[0]), reverse=True):
    expression = expression.replace(token, repr(value))
  expression = expression.replace("true", "True").replace("false", "False")
  expression = expression.replace("&&", " and ").replace("||", " or ")
  expression = re.sub(r"!(?!=)", " not ", expression)
  return bool(eval(expression, {"__builtins__": {}}, {}))


def fixture_asset_manifest(include_genai=False):
  manifest = {
      "schema_version": 1,
      "revision": "windows-pcie-fixture-r1",
      "include_genai": include_genai,
      "vision": {
          name: {"kind": "file", "sha256": f"{index + 1:x}" * 64, "size": index + 10}
          for index, name in enumerate(VISION_ASSETS)
      },
      "genai": {},
  }
  if include_genai:
    manifest["genai"] = {
        name: {
            "kind": "directory",
            "source_revision": {"repository": "fixture", "revision": f"r{index + 1}"},
            "files": [{
                "path": f"model-{index + 1}.bin",
                "sha256": f"{index + 6:x}" * 64,
                "size": index + 20,
            }],
        }
        for index, name in enumerate(GENAI_ASSETS)
    }
  return normalize_asset_manifest(manifest)


def fixture_asset_environment(root, include_genai=False):
  root.mkdir(parents=True, exist_ok=True)
  environment = {}
  for index, name in enumerate(VISION_ASSETS):
    path = root / f"vision-{index}.bin"
    path.write_bytes(f"vision-{index}".encode("ascii"))
    environment[name] = str(path.resolve())
  if include_genai:
    for index, name in enumerate(GENAI_ASSETS):
      path = root / f"genai-{index}"
      (path / "nested").mkdir(parents=True)
      (path / "z.bin").write_bytes(f"z-{index}".encode("ascii"))
      (path / "nested" / "a.bin").write_bytes(f"a-{index}".encode("ascii"))
      environment[name] = str(path.resolve())
  return environment


class WindowsCiContracts(unittest.TestCase):
  def test_failed_msi_transaction_receipt_binds_preserved_cohort(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      common_data = root / "ProgramData"
      common_data.mkdir()
      install_root = root / "SimaPCIeHost"
      install_root.mkdir()
      (install_root / "owned.dll").write_bytes(b"unchanged cohort")
      scheduler = root / "scheduler.msi"
      core = root / "core.msi"
      fixture = root / "rollback-fixture.msi"
      for path in (scheduler, core, fixture):
        path.write_bytes(path.name.encode("ascii"))
      log = root / "msi-transaction-rollback.log"
      product_codes = {
          scheduler: "{11111111-2222-3333-4444-555555555555}",
          core: "{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE}",
          fixture: MODULE.ROLLBACK_FIXTURE_PRODUCT_CODE,
      }

      def failed_install(_command, **_kwargs):
        (common_data / MODULE.ROLLBACK_FIXTURE_OBSERVATION).write_text(
            "schema_version=1\nfile=present\nregistry=present\n", encoding="ascii")
        log.write_text("FailAfterOwnedState", encoding="utf-8")
        return SimpleNamespace(returncode=1603)

      def product_state(product_code):
        if product_code == MODULE.ROLLBACK_FIXTURE_PRODUCT_CODE:
          return MODULE.MSI_INSTALLSTATE_UNKNOWN
        return MODULE.MSI_INSTALLSTATE_DEFAULT

      with mock.patch.object(MODULE, "msi_product_code", side_effect=product_codes.get), \
           mock.patch.object(MODULE, "msi_product_state", side_effect=product_state), \
           mock.patch.object(MODULE, "registry_fixture_key_exists", return_value=False), \
           mock.patch.object(MODULE.subprocess, "run", side_effect=failed_install):
        receipt = MODULE.qualify_failed_msi_transaction(
            fixture, scheduler, core, install_root, common_data, log)

      self.assertEqual(receipt["status"], "PASS")
      self.assertEqual(receipt["failure"]["observed_exit_code"], 1603)
      self.assertEqual(receipt["mid_transaction"], {"file": "PASS", "registry": "PASS"})
      self.assertEqual(receipt["rollback"]["product_registration"], "PASS")
      self.assertEqual(receipt["preserved_cohort"]["file_inventory"], "PASS")
      self.assertFalse((common_data / MODULE.ROLLBACK_FIXTURE_OBSERVATION).exists())

  def test_native_windows_genai_runs_production_adapter_against_test_only_dll(self):
    cmake = (ROOT / "pcie_host/CMakeLists.txt").read_text(encoding="utf-8")
    self.assertIn("tests/fake_genai_service_windows.cpp", cmake)
    self.assertIn("OUTPUT_NAME SiMaPCIe", cmake)
    self.assertIn("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR=$<TARGET_FILE_DIR:fake_genai_service>", cmake)
    self.assertIn("unit_pcie_host_genai_service_test", cmake)
    self.assertIn("unit_pcie_host_genai_assets_test", cmake)
    self.assertNotIn("install(TARGETS fake_genai_service", cmake)

  def test_genai_failure_classifiers_reject_unrelated_errors(self):
    for message in (
        "GenAI admission rejected: card resource unavailable",
        "another session owns this resource",
        "session limit reached",
        "device busy",
    ):
      with self.subTest(message=message):
        self.assertTrue(is_bounded_competing_session_rejection(message))
    for message in (
        RuntimeError("model configuration is malformed"),
        ValueError("worker readiness timed out"),
        "could not open unrelated file",
        "invalid session identifier",
    ):
      with self.subTest(message=message):
        self.assertFalse(is_bounded_competing_session_rejection(message))

    for message in (
        "GenAI worker stopped responding",
        "PCIe service receive failed: -5",
        "connection reset by peer",
    ):
      with self.subTest(message=message):
        self.assertTrue(is_injected_transport_failure(message))
    for message in (
        "GenAI request timeout",
        "model asset is missing",
        "invalid prompt",
    ):
      with self.subTest(message=message):
        self.assertFalse(is_injected_transport_failure(message))

  def test_windows_openssh_batch_wrapper_redirects_before_python(self):
    line = _windows_openssh_batch_line(
        ["ssh.exe", "card", "printf '%s\\n' value"],
        Path(r"C:\qualification evidence\card.log"),
    )
    self.assertIn("powershell.exe", line)
    encoded = line.split("-EncodedCommand ", 1)[1].split(" > ", 1)[0]
    decoded = base64.b64decode(encoded).decode("utf-16le")
    self.assertIn("'ssh.exe','card','printf ''%s\\n'' value'", decoded)
    self.assertIn(r'"C:\qualification evidence\card.log" 2>&1', line)
    piped = _windows_openssh_batch_line(
        ["ssh.exe", "card", "true"], Path(r"C:\card.log"),
        input_file=Path(r"C:\secret input"))
    self.assertIn(r'< "C:\secret input"', piped)

  def test_windows_openssh_relay_drops_inherited_fd_state_before_running_ssh(self):
    # PowerShell relays ssh.exe output through a pipe; with the server session's
    # inherited fd state, ssh.exe blocks once that pipe fills (a few KB of output).
    line = _windows_openssh_batch_line(["ssh.exe", "card", "true"], Path(r"C:\card.log"))
    encoded = line.split("-EncodedCommand ", 1)[1].split(" > ", 1)[0]
    decoded = base64.b64decode(encoded).decode("utf-16le")
    cleared = decoded.find("$env:c28fc6f98a2c44abbbd89d6a3037d0d9_POSIX_FD_STATE=$null;")
    self.assertGreaterEqual(cleared, 0)
    self.assertLess(cleared, decoded.index("& $exe @rest"))

  def test_same_run_linux_admission_is_not_a_manual_dispatch_override(self):
    workflow = WORKFLOW.read_text(encoding="utf-8")
    dispatch = workflow.split("  workflow_dispatch:\n", 1)[1].split("\npermissions:", 1)[0]
    self.assertNotIn("linux_current_run:", dispatch)
    self.assertIn('"${GITHUB_WORKFLOW}" != "Vulcan CI"', workflow)
    self.assertIn("Same-run Linux admission is reserved", workflow)
    caller = MAIN_WORKFLOW.read_text(encoding="utf-8")
    job = caller.split("  test-pciehost-windows:\n", 1)[1].split("  promote-latest-tag:", 1)[0]
    self.assertIn("linux_current_run: true", job)
    # The card lane runs beside Linux; same-run candidates are never published.
    self.assertNotIn("test-devkit", job.split("    permissions:", 1)[0])
    self.assertNotIn("test-pciehost-hardware", job.split("    permissions:", 1)[0])
    self.assertIn("validation_only: true", job)

  def test_card_offline_bundle_is_assembled_from_this_run(self):
    caller = MAIN_WORKFLOW.read_text(encoding="utf-8")
    producer = caller.split(
        "  build-pcie-card-offline-bundle:\n", 1)[1].split(
            "  publish-vulcan-artifacts:", 1)[0]
    # The Linux offline assembler, fed this run's neat-core-dist.
    self.assertIn("python3 -B pcie_host/scripts/offline/ci.py build --core --validation-only", producer)
    self.assertIn('--local-core-dist "$RUNNER_TEMP/core-dist"', producer)
    self.assertIn("name: neat-core-dist", producer)
    self.assertIn("- build\n", producer)
    self.assertIn("name: core-pcie-card-offline-bundle", producer)
    self.assertIn("name: core-pcie-card-offline-evidence", producer)
    self.assertIn("role-to-assume: ${{ needs.resolve-vulcan-config.outputs.artifact_publisher_role_arn }}",
                  producer)
    offline = (ROOT / ".github/workflows/pcie-offline.yml").read_text(encoding="utf-8")
    for name in ("core-pcie-card-offline-bundle", "core-pcie-card-offline-evidence"):
      self.assertNotIn(name, offline)
    for retired in ("sdk_build_identity.py", "sdk-build-image.json", "SDK_IMAGE_ID",
                    "build_card_qualification_dependencies.py"):
      self.assertNotIn(retired, caller)
      self.assertFalse((SCRIPT_DIR / retired).exists())
    windows_job = caller.split("  test-pciehost-windows:\n", 1)[1].split(
        "  promote-latest-tag:", 1)[0]
    self.assertIn("build-pcie-card-offline-bundle", windows_job)

    workflow = WORKFLOW.read_text(encoding="utf-8")
    prepare = workflow.split("  card_candidate:\n", 1)[1].split(
        "  dependencies:", 1)[0]
    self.assertIn("prepare_card_qualification_candidate.py", prepare)
    self.assertIn("name: core-pcie-card-offline-bundle", prepare)
    self.assertIn('--bundle "${RUNNER_TEMP}/core-card-bundle"', prepare)
    # A release tag's card bundle carries the bare tag version, so both checks need the ref.
    self.assertIn('--ref "${GITHUB_REF_NAME}"', prepare)
    self.assertIn("core-pcie-card-qualification-candidate", prepare)
    hardware = workflow.split("  hardware:\n", 1)[1].split("  publish:", 1)[0]
    self.assertIn("needs: [native, linux_release, card_candidate, resolve-vulcan-config]", hardware)
    self.assertIn("core-pcie-card-qualification-candidate", hardware)
    self.assertNotIn("name: neat-core-dist", hardware)

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

  def test_recovery_reads_the_stage_recorded_under_the_card_lock(self):
    args = SimpleNamespace(remote_lock_path="/tmp/neat-pcie-card0.lock")
    with tempfile.TemporaryDirectory() as directory:
      evidence = Path(directory)
      stage = "/var/tmp/neat-offline-Ab3dE6gH"
      for code, output, expected in ((0, stage, stage), (0, stage + "\r\n", stage),
                                     (0, "absent", None), (1, "", "error"),
                                     (0, "/tmp/elsewhere", "error")):
        with self.subTest(code=code, output=output), \
            mock.patch.object(RECOVERY, "ssh_command", side_effect=lambda _args, cmd: [cmd]), \
            mock.patch.object(RECOVERY, "run_captured_to_file", return_value=
                              subprocess.CompletedProcess([], code, output)):
          if expected == "error":
            with self.assertRaisesRegex(RuntimeError, "offline stage is unknown"):
              RECOVERY.recorded_card_stage(args, evidence)
          else:
            self.assertEqual(RECOVERY.recorded_card_stage(args, evidence), expected)

  def test_msi_removal_waits_while_another_installation_runs(self):
    # 1618 (ERROR_INSTALL_ALREADY_RUNNING) changes nothing; Windows Update can hold it briefly.
    msi, log = Path("C:/cohort/core.msi"), Path("C:/evidence/core.log")
    command = ["msiexec.exe", "/x", str(msi), "/quiet", "/norestart", "/L*v", str(log)]
    for codes, removal, expected in (
        ([1618, 1618, 0], RECOVERY.remove_msi, 0),
        ([1618, 1605], HARDWARE.remove_msi, "ABSENT"),
        ([1618] * 5, RECOVERY.remove_msi, "failed with 1618"),
        ([1618] * 5, HARDWARE.remove_msi, "failed with 1618"),
        ([1603], HARDWARE.remove_msi, "failed with 1603")):
      with self.subTest(codes=codes, removal=removal.__module__), \
          mock.patch.object(HARDWARE.subprocess, "run", side_effect=[
              subprocess.CompletedProcess(command, code) for code in codes]) as run, \
          mock.patch.object(HARDWARE.time, "sleep") as sleep:
        if isinstance(expected, str) and expected.startswith("failed"):
          with self.assertRaisesRegex(RuntimeError, expected):
            removal(msi, log)
        else:
          self.assertEqual(removal(msi, log), expected)
        self.assertEqual(run.call_args_list, [mock.call(command, check=False)] * len(codes))
        self.assertEqual(sleep.call_args_list, [mock.call(30)] * (len(codes) - 1))

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
          mock.patch.object(RECOVERY, "remove_standard_user_leftovers") as remove_user, \
          mock.patch.object(RECOVERY, "remove_msi") as remove, \
          mock.patch.object(RECOVERY.shutil, "rmtree") as remove_work:
        lock.return_value.owner.return_value = None
        RECOVERY.cleanup(args)
      stop.assert_not_called()
      remove_user.assert_not_called()
      remove.assert_not_called()
      remove_work.assert_not_called()
      receipt = json.loads((args.evidence / "recovery-cleanup.json").read_text())
      self.assertEqual(receipt["errors"], [])

  def test_recovery_under_a_dead_runs_lock_removes_its_standard_user(self):
    token = "0db9418bd645b0e4daf1a17494779d23"
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      lock_file = root / "card.lock"
      lock_file.write_text(f"pid=10080 started=now remote_token={token}\n", encoding="utf-8")
      args = SimpleNamespace(candidate=root, upgrade_base_candidate=None,
                             evidence=root / "evidence", lock_file=lock_file,
                             remote_lock_path="/tmp/neat-pcie-card0.lock", work=root / "work")
      with mock.patch.object(RECOVERY.sys, "platform", "win32"), \
          mock.patch.dict(RECOVERY.os.environ, {"ProgramFiles": str(root / "absent")}), \
          mock.patch.object(RECOVERY, "validate_candidate"), \
          mock.patch.object(RECOVERY, "process_is_running", return_value=False), \
          mock.patch.object(RECOVERY, "RemoteExclusiveLock") as lock, \
          mock.patch.object(RECOVERY, "recorded_card_stage", return_value=None), \
          mock.patch.object(RECOVERY, "remove_standard_user_leftovers") as remove_user, \
          mock.patch.object(RECOVERY, "remove_msi", return_value=0):
        lock.return_value.owner.return_value = token
        RECOVERY.cleanup(args)
      remove_user.assert_called_once_with(
          args.evidence.resolve() / "recovery-standard-user.log")
      receipt = json.loads((args.evidence / "recovery-cleanup.json").read_text())
      self.assertEqual(receipt["errors"], [])
      self.assertEqual(receipt["results"]["standard_user"], "removed")
      self.assertFalse(lock_file.exists())
    # The names verify_standard_user_process_activation creates.
    source = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    self.assertIn("$name = 'sna-' + $PID", source)
    self.assertIn("$_.Name -match '^sna-[0-9]+$'", source)
    self.assertIn("-Filter 'SimaPCIeHostAcceptance-*'", source)

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
        "platform": {"version": "3.0.0.2", "runtime_dlls": [
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
        "package-version": "0.6.1",
        "repository": "sima-neat/scheduler",
        "resources": sorted(resources),
        "resources-checksum": resources,
        "version": "a" * 12,
    }), encoding="utf-8")
    return msi

  def test_scheduler_candidate_accepts_exact_inventory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
      expected = self.scheduler_candidate(root)
      metadata, actual = MODULE.verify_scheduler_candidate(root, "a" * 40)
      self.assertEqual(actual, expected)
      self.assertEqual(metadata["commit"], "a" * 40)
      self.assertEqual(metadata["package-version"], "0.6.1")

  def test_scheduler_candidate_requires_distinct_msi_package_version(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.scheduler_candidate(root)
      metadata_path = root / "metadata.json"
      metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
      for value in (None, "a" * 12, "0.6"):
        with self.subTest(value=value):
          if value is None:
            metadata.pop("package-version", None)
          else:
            metadata["package-version"] = value
          metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
          with self.assertRaisesRegex(ValueError, "MSI package version"):
            MODULE.verify_scheduler_candidate(root, "a" * 40)

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

  def test_scheduler_msi_identity_is_read_from_the_msi(self):
    msi = Path("scheduler.msi")
    properties = {
        "UpgradeCode": "{4043eb4e-3e8c-48cc-8bec-56ecb5ee245a}",
        "ProductVersion": "0.6.2",
    }
    with mock.patch.object(MODULE, "msi_property",
                           side_effect=lambda path, name: properties[name]) as read:
      identity = MODULE.scheduler_msi_identity(msi, "0.6.2")
    self.assertEqual(identity, {
        "upgrade_code": "{4043EB4E-3E8C-48CC-8BEC-56ECB5EE245A}",
        "version": "0.6.2",
    })
    self.assertEqual({call.args for call in read.call_args_list},
                     {(msi, "UpgradeCode"), (msi, "ProductVersion")})

  def test_scheduler_msi_identity_fails_when_unreadable_or_different(self):
    valid = {"UpgradeCode": "{4043EB4E-3E8C-48CC-8BEC-56ECB5EE245A}",
             "ProductVersion": "0.6.2"}
    for name, value, error in (
        ("UpgradeCode", "", "invalid UpgradeCode"),
        ("UpgradeCode", "4043EB4E-3E8C-48CC-8BEC-56ECB5EE245A", "invalid UpgradeCode"),
        ("ProductVersion", "0.6.3", "is not its package version"),
        ("ProductVersion", "0.6.2.0", "is not its package version"),
    ):
      properties = {**valid, name: value}
      with self.subTest(name=name, value=value), \
           mock.patch.object(MODULE, "msi_property",
                             side_effect=lambda _path, key, values=properties: values[key]):
        with self.assertRaisesRegex(ValueError, error):
          MODULE.scheduler_msi_identity(Path("scheduler.msi"), "0.6.2")
    with mock.patch.object(MODULE, "msi_property",
                           side_effect=RuntimeError("MSI has no UpgradeCode row")):
      with self.assertRaisesRegex(RuntimeError, "no UpgradeCode row"):
        MODULE.scheduler_msi_identity(Path("scheduler.msi"), "0.6.2")

  def test_producer_passes_scheduler_msi_identity_to_wix(self):
    source = SCRIPT.read_text(encoding="utf-8")
    build = source.split("def build(", 1)[1]
    read = build.index(
        'scheduler_msi_identity(\n      scheduler_msi, scheduler_metadata["package-version"])')
    configure = build.index('"-DSimaScheduler_DIR=" + str(scheduler_config)')
    self.assertLess(read, configure)
    self.assertIn(
        '"-DSIMAPCIE_SCHEDULER_PACKAGE_VERSION=" + scheduler_identity["version"]', build)
    self.assertIn(
        '"-DSIMAPCIE_SCHEDULER_WIX_UPGRADE_CODE=" + scheduler_identity["upgrade_code"]',
        build)
    packaging = (ROOT / "pcie_host" / "cmake" / "PackagingWindows.cmake").read_text(
        encoding="utf-8")
    self.assertIn("set(CPACK_WIX_CANDLE_EXTRA_FLAGS", packaging)

  def test_python_card_cases_count_only_passes(self):
    with tempfile.TemporaryDirectory() as directory:
      junit = Path(directory) / "python.xml"
      junit.write_text(
          '<testsuite><testcase name="ran"/><testcase name="skipped"><skipped/></testcase>'
          '<testcase name="failed"><failure/></testcase>'
          '<testcase name="broken"><error/></testcase>'
          # pytest names a parametrized test's cases name[parameter].
          '<testcase name="both[a]"/><testcase name="both[b]"/>'
          '<testcase name="one[a]"/><testcase name="one[b]"><skipped/></testcase></testsuite>',
          encoding="utf-8")
      self.assertEqual(HARDWARE.junit_outcomes(junit)[0], {"ran", "both"})

  def test_fault_client_leftovers_are_removed_by_process_id(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      # The snapshot names a card PC run left for client processes 5488 and 7332.
      left = root / ("sima-neat-pcie-model-28984716daa5b400"
                     "0000000000001570000000000000000000001acfd561da2c")
      other = root / ("sima-neat-pcie-model-63ba4187e083da4a"
                      "0000000000001ca4000000000000000000001ac2e71d21d4")
      for snapshot in (left, other):
        snapshot.mkdir()
        (snapshot / "model.tar.gz").write_bytes(b"fixture")
      commands = []
      with mock.patch("qualify_windows_hardware.tempfile.gettempdir", return_value=directory), \
           mock.patch("qualify_windows_hardware.run_logged",
                      side_effect=lambda command, *_a, **_k: commands.append(command[-1])):
        HARDWARE.remove_fault_client_leftovers(card_args(root), 5488, root / "log")
      self.assertFalse(left.exists())
      self.assertTrue(other.is_dir())
      self.assertEqual(commands, ["rm -f -- /tmp/sima-neat-pcie-5488-*"])
    # The names this relies on.
    self.assertIn(
        "token << std::hex << std::setfill('0') << std::setw(8) << random() << std::setw(8) "
        "<< random()\n        << std::setw(16) << host_process_id()",
        (ROOT / "pcie_host/src/HostPlatform.cpp").read_text(encoding="utf-8"))
    self.assertIn(
        'std::string(kRemoteModelDir) + "/sima-neat-pcie-" + std::to_string(host_process_id())',
        (ROOT / "pcie_host/src/RemoteRuntime.cpp").read_text(encoding="utf-8"))
    source = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    self.assertEqual(source.count("remove_fault_client_leftovers(\n        args, process.pid,"), 2)

  def test_cleanup_error_does_not_mask_the_qualification_failure(self):
    HARDWARE.raise_first_failure(None, [])
    with self.assertRaisesRegex(RuntimeError, "^hardware cleanup failed: left; locked$"):
      HARDWARE.raise_first_failure(None, ["left", "locked"])
    failure = RuntimeError("command failed with 1; see tutorial.log")
    stderr = io.StringIO()
    with self.assertRaises(RuntimeError) as raised, contextlib.redirect_stderr(stderr):
      HARDWARE.raise_first_failure(failure, ["left"])
    self.assertIs(raised.exception, failure)
    self.assertEqual(stderr.getvalue(), "hardware cleanup failed: left\n")

  def test_workflow_uses_exact_scheduler_artifact_and_restricted_runner(self):
    workflow = WORKFLOW.read_text(encoding="utf-8")
    producer = (SCRIPT_DIR / "run_windows_build.ps1").read_text(encoding="utf-8")
    source = workflow + producer
    # build_windows.py refuses these; hosted runners (actions/setup-python) export PKG_CONFIG_PATH.
    clear = producer.index('foreach ($name in @("CMAKE_TOOLCHAIN_FILE"')
    self.assertLess(producer.index("Enter-VsDevShell"), clear)
    self.assertLess(clear, producer.index('"scripts/ci/pciehost/build_windows.py"'))
    for name in ("CMAKE_TOOLCHAIN_FILE", "CMAKE_GENERATOR_PLATFORM", "CMAKE_PREFIX_PATH", "PKG_CONFIG_PATH"):
      self.assertIn(f'"{name}"', producer[clear:clear + 200])
    self.assertIn("  pull_request:\n", source)
    self.assertIn("if: github.event_name != 'pull_request'", source)
    self.assertIn("github.repository == 'sima-neat/core'", source)
    self.assertIn("scheduler_run_id", source)
    self.assertIn("scheduler_commit", source)
    self.assertIn("--expected-scheduler-commit", source)
    # The reusable Windows workflow cannot be dispatched from a feature ref
    # before it exists on the default branch. Admit the Scheduler root run,
    # which owns the reusable job and the retained artifact under one run ID.
    # The workflow path identifies the run; run-name can replace the run name.
    self.assertNotIn("--workflow-name", source)
    self.assertIn('--workflow-path ".github/workflows/ci.yml"', source)
    self.assertIn(
        '--required-reusable-workflow '
        '"sima-neat/scheduler/.github/workflows/pcie-host-windows.yml"', source)
    self.assertIn(
        '--required-job "Build, test, and publish native Windows PCIe profile / contracts"',
        source)
    self.assertIn(
        '--required-job '
        '"Build, test, and publish native Windows PCIe profile / Native Windows amd64"',
        source)
    self.assertIn('--required-job "Verify Scheduler build and Vulcan publication"', source)
    self.assertIn('--required-artifact "scheduler-native-windows-amd64"', source)
    self.assertIn("repository: sima-neat/scheduler", source)
    # The org's runner script registers default labels only; the group selects the host.
    for label in ("pcie-windows-build", "pcie-windows-hardware"):
      self.assertNotIn(label, source)
      self.assertNotIn(label, (ROOT / ".github" / "actionlint.yaml").read_text(encoding="utf-8"))
    self.assertNotIn("windows-latest", source)
    self.assertIn("scripts/ci/pciehost/build_windows.py", source)
    self.assertIn("python3 pcie_host/tests/test_verify_linux_release_run.py", source)
    self.assertIn("scripts/ci/pciehost/qualify_windows_hardware.py", source)
    self.assertIn("scripts/ci/pciehost/cleanup_windows_hardware.py", source)
    self.assertIn("linux_run_id", source)
    self.assertIn("Download this run's exact Linux candidate", source)
    self.assertIn("Download prior run's exact Linux candidate", source)
    self.assertIn("Bind the offline card bundle to the exact Linux candidate", source)
    self.assertIn("Download exact private card qualification candidate", source)
    self.assertIn("name: neat-core-dist", source)
    self.assertIn("--card-candidate", source)
    self.assertIn("--expected-card-commit", source)
    self.assertEqual(source.count('"--ref", $env:GITHUB_REF_NAME,'), 2)
    self.assertIn("SIMA_WINDOWS_PCIE_ASSET_MANIFEST", source)
    # The asset manifest is written in-job from the resolved assets, not a variable.
    self.assertNotIn("SIMA_WINDOWS_PCIE_GENAI_ASSET_MANIFEST", source)
    self.assertNotIn("SIMA_WINDOWS_PCIE_VISION_ASSET_MANIFEST", source)
    self.assertEqual(source.count('"--asset-manifest"'), 2)
    hardware_job = source.split("  hardware:\n", 1)[1]
    hardware_env = hardware_job.split("    steps:\n", 1)[0]
    self.assertNotIn("SIMAPCIE_CARD_SUDO_PASSWORD", hardware_env)
    # Linux passes DEVKIT_PASSWORD as step environment; a PowerShell pipe to Python lost it.
    self.assertNotIn("--card-sudo-password-stdin", source)
    recovery = source.split("- name: Recover card and native installation", 1)[1].split(
        "- name: Enforce release admission", 1)[0]
    for step in (source.split("- name: Install, test both APIs", 1)[1].split("- name:", 1)[0], recovery):
      self.assertIn("SIMAPCIE_CARD_SUDO_PASSWORD: ${{ secrets.DEVKIT_PASSWORD }}", step)
    self.assertNotIn("SIMAPCIE_EXPECTED_CARD_RUNTIME_SHA256", source)
    self.assertNotIn("SIMAPCIE_EXPECTED_CARD_WORKER_SHA256", source)
    self.assertNotIn("SIMAPCIE_EXPECTED_CARD_GENAI_WORKER_SHA256", source)
    self.assertIn("sima_cli_run_id", source)
    self.assertIn("sima-cli-dist", source)
    self.assertIn("--require-signatures", source)
    self.assertIn('$arguments += @("--require-genai", "--require-release-qualification")',
                  source)
    self.assertIn('$arguments += "--release-qualification"', source)
    self.assertIn('$arguments += "--require-release"', source)
    self.assertIn('if ("$env:REQUIRE_SIGNATURES" -eq "true")', source)
    self.assertIn("if: always()", source)
    self.assertNotIn("upgrade_base_run_id", source)
    self.assertIn("SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH", source)
    self.assertNotIn("SIMA_WINDOWS_PLATFORM_INCLUDE_DIR", source)
    self.assertIn("--dependency-runtime-dir", source)
    self.assertIn("scripts/ci/pciehost/acquire_windows_dependencies.py", source)
    self.assertIn("core-pciehost-windows-dependencies", source)
    self.assertIn("linux_current_run", source)
    self.assertIn("first_release_upgrade_exemption_reason", source)
    self.assertIn("Publishing requires a real upgrade gate or an explicit first-release exemption.", source)
    self.assertIn("windows-upgrade-exemption.json", source)
    publish_job = source.split("  publish:\n", 1)[1]
    self.assertIn(
        "inputs.require_upgrade == true || inputs.first_release_upgrade_exemption == true",
        publish_job,
    )
    self.assertIn(
        r"SIMA_WINDOWS_SHORT_WORK_ROOT: C:\nr\c-${{ github.run_id }}-${{ github.run_attempt }}",
        source,
    )
    native_job = workflow.split("  native:\n", 1)[1].split("  hardware:\n", 1)[0]
    self.assertIn("& ./scripts/ci/pciehost/run_windows_build.ps1", native_job)
    self.assertIn('"--work", $env:SIMA_WINDOWS_SHORT_WORK_ROOT', producer)
    self.assertIn("Remove short Windows producer work root", native_job)
    self.assertEqual(
        (native_job + producer).count(
            "Remove-Item -LiteralPath $env:SIMA_WINDOWS_SHORT_WORK_ROOT -Recurse -Force"
        ),
        2,
    )
    self.assertEqual(workflow.count("shell: powershell"), 11)
    hardware_job = workflow.split("  hardware:\n", 1)[1].split("  publish:\n", 1)[0]
    # Self-hosted Windows keeps the default Restricted script policy, as customer PCs do;
    # run every step like pcie-setup.cmd does instead of provisioning the host.
    self.assertNotIn("        shell: powershell\n", hardware_job)
    self.assertEqual(hardware_job.count("shell: powershell -NoLogo -NoProfile -ExecutionPolicy Bypass"), 7)
    for job in (native_job, hardware_job):
      # MSI install/remove needs an elevated token; fail before checkout, not mid-build.
      steps = job.split("    steps:\n", 1)[1]
      self.assertTrue(steps.startswith("      - name: Require an elevated runner\n"))
      self.assertIn("IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)", steps)
      self.assertIn("exit 1", steps.split("      - uses: actions/checkout@v5", 1)[0])

  def test_workflow_run_bodies_receive_inputs_only_through_env(self):
    # Expression text is substituted into the script before the shell parses
    # it, so an input such as $(...) would execute on the runner. Reject every
    # expression, whatever its spacing or context.
    lines = WORKFLOW.read_text(encoding="utf-8").splitlines()
    leaks = []
    for index, line in enumerate(lines):
      match = re.match(r"^(\s*)(?:- )?run: ?(.*)$", line)
      if not match:
        continue
      indent = len(match.group(1))
      body = [match.group(2)]
      for following in lines[index + 1:]:
        if following.strip() and len(following) - len(following.lstrip()) <= indent:
          break
        body.append(following)
      leaks += [(index + 1, text.strip()) for text in body if "${{" in text]
    self.assertEqual(leaks, [])

  def test_main_ci_requires_windows_acceptance_before_promotion_whenever_it_applies(self):
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    job, condition = partial(workflow_job, source), partial(job_condition, source)
    self.assertIn("uses: ./.github/workflows/pcie-host-windows.yml", source)


    gate = textwrap.dedent(job("validation-complete").split(
        "          python3 - <<'PYGATE'\n", 1)[1].split("\n          PYGATE", 1)[0])
    gate_flag = re.search(r"(?m)^          WINDOWS_PCIE_CI: (.+)$", job("validation-complete"))[1]
    bundle_flag = re.search(r"(?m)^          CARD_BUNDLE: (.+)$", job("validation-complete"))[1]
    for variable in ("false", "true"):
      for hardware in ("success", "skipped"):
        for outcome in ("success", "failure"):
          with self.subTest(windows_pcie=variable, hardware=hardware, outcome=outcome):
            context = {
                "always()": True,
                "needs.detect-changes.outputs.windows_pcie": variable,
                "github.event_name": "push",
                "github.workflow": "Vulcan CI",
                "inputs.offline_source_run_id": "",
                "inputs.validation_only": False,
                "inputs.build_mode": "standard",
                "github.ref_type": "branch",
                "startsWith(github.ref, 'refs/tags/v')": False,
                "needs.detect-changes.outputs.docs_only": "false",
                "needs.test-pciehost-hardware.result": hardware,
            }
            for upstream in (
                "validate-windows-cohort-selectors", "build",
                "build-pcie-card-offline-bundle", "build-pciehost-packages",
                "test-devkit", "publish-pciehost-package"):
              context[f"needs.{upstream}.result"] = "success"
            self.assertEqual(evaluate_actions_expression(
                condition("build-pcie-card-offline-bundle"), context),
                variable == "true")
            runs = evaluate_actions_expression(condition("test-pciehost-windows"), context)
            # Independent of the Linux card result: both lanes run side by side.
            self.assertEqual(runs, variable == "true")
            context["needs.test-pciehost-windows.result"] = outcome if runs else "skipped"
            self.assertEqual(
                evaluate_actions_expression(condition("promote-latest-tag"), context),
                outcome == "success" if runs else variable != "true")

            results = {name: {"result": "success"} for name in re.findall(
                r"- ([\w-]+)", job("validation-complete").split("    steps:", 1)[0])}
            for name in ("publish-vulcan-artifacts", "publish-pciehost-package",
                         "promote-latest-tag"):
              results[name]["result"] = "skipped"
            results["test-pciehost-windows"]["result"] = context[
                "needs.test-pciehost-windows.result"]
            if variable != "true":
              results["build-pcie-card-offline-bundle"]["result"] = "skipped"
            environment = {
                "RESULTS": json.dumps(results), "BUILD_MODE": "standard",
                "WINDOWS_PCIE_CI": str(evaluate_actions_expression(gate_flag, context)).lower(),
                "CARD_BUNDLE": str(evaluate_actions_expression(bundle_flag, context)).lower(),
            }
            with mock.patch.dict(os.environ, environment):
              if outcome == "success" or variable != "true":
                exec(compile(gate, str(MAIN_WORKFLOW), "exec"), {})
              else:
                with self.assertRaisesRegex(SystemExit, "test-pciehost-windows"):
                  exec(compile(gate, str(MAIN_WORKFLOW), "exec"), {})

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_card_lane_runs_automatically_for_windows_host_changes(self):
    # Like the Linux card lane it needs no repository variable or dispatch opt-in.
    # One Windows card runner (~23 min a job) cannot take every push as the two
    # Linux card runners do, so a feature branch push runs it only for changes to
    # the Windows host's inputs; develop, tags and manual runs always run it.
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    self.assertNotIn("ENABLE_WINDOWS_PCIE_CI", source)
    self.assertNotIn("run_windows_pcie", source)
    detect = workflow_job(source, "detect-changes")
    self.assertIn(
        "      windows_pcie: ${{ steps.detect.outputs.windows_pcie }}\n", detect)
    script = textwrap.dedent(workflow_step(
        detect, "Detect docs-only and Windows PCIe host changesets").split(
            "        run: |\n", 1)[1])
    branch_change = lambda *paths: ("push", "branch", "feature/x", paths)
    cases = (
        (branch_change("pcie_host/src/host.cpp"), "false", "true"),
        (branch_change("scripts/ci/pciehost/qualify_windows_hardware.py"), "false", "true"),
        (branch_change(".github/workflows/pcie-host-windows.yml"), "false", "true"),
        (branch_change("src/pipeline/Session.cpp", "deps/manifest.json"), "false", "true"),
        (branch_change("src/pipeline/Session.cpp", ".github/workflows/vulcan-ci.yml"),
         "false", "false"),
        (branch_change("docs/index.md"), "true", "false"),
        (("push", "branch", "develop", ("src/pipeline/Session.cpp",)), "false", "true"),
        (("push", "tag", "v0.7.0", ()), "false", "true"),
        (("workflow_dispatch", "branch", "feature/x", ("src/pipeline/Session.cpp",)),
         "false", "true"),
    )
    bash = shutil.which("bash")
    self.assertIsNotNone(bash)
    with tempfile.TemporaryDirectory() as temporary:
      root = Path(temporary)
      git = root / "git"
      git.write_text(
          '#!/bin/sh\ncase "$1" in\n  merge-base) echo base ;;\n'
          '  diff) cat "$CHANGED_FILES" ;;\nesac\n', encoding="utf-8")
      git.chmod(0o755)
      for (event, ref_type, ref_name, paths), docs_only, windows_pcie in cases:
        with self.subTest(event=event, ref=ref_name, paths=paths):
          (root / "changed").write_text("".join(f"{path}\n" for path in paths))
          output = root / "github-output"
          output.write_text("")
          environment = os.environ.copy()
          environment.update({
              "PATH": f"{root}{os.pathsep}{environment['PATH']}",
              "CHANGED_FILES": str(root / "changed"), "GITHUB_OUTPUT": str(output),
              "GITHUB_EVENT_NAME": event, "GITHUB_REF_TYPE": ref_type,
              "GITHUB_REF_NAME": ref_name})
          result = subprocess.run(
              [bash, "-e", "-c", script], capture_output=True, text=True, check=False,
              env=environment)
          self.assertEqual(result.returncode, 0, result.stderr)
          self.assertEqual(
              dict(line.split("=", 1) for line in output.read_text().splitlines()),
              {"docs_only": docs_only, "windows_pcie": windows_pcie})

  def test_release_tags_build_the_card_bundle_and_other_tags_block_nothing(self):
    # The release dispatch downloads a release tag run's card bundle. A bundle
    # accepts only release tag versions, so other tags never build it and never
    # wait for the Windows lane, which needs it.
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    job, condition = partial(workflow_job, source), partial(job_condition, source)
    gate = textwrap.dedent(job("validation-complete").split(
        "          python3 - <<'PYGATE'\n", 1)[1].split("\n          PYGATE", 1)[0])
    flag = lambda name: re.search(
        rf"(?m)^          {name}: (.+)$", job("validation-complete"))[1]
    for ref in ("refs/heads/develop", "refs/tags/v0.7.0", "refs/tags/3.0.2121"):
      for variable in ("false", "true"):
        for workflow_name in ("Vulcan CI", "Long Tests Weekly"):
          with self.subTest(ref=ref, windows_pcie=variable, workflow=workflow_name):
            release = ref.startswith("refs/tags/v")
            context = {
                "always()": True,
                "needs.detect-changes.outputs.windows_pcie": variable,
                "github.event_name": "push",
                "github.workflow": workflow_name,
                "inputs.offline_source_run_id": "",
                "inputs.validation_only": False,
                "inputs.build_mode": "standard",
                "github.ref_type": "tag" if ref.startswith("refs/tags/") else "branch",
                "startsWith(github.ref, 'refs/tags/v')": release,
                "needs.detect-changes.outputs.docs_only": "false",
            }
            for upstream in (
                "validate-windows-cohort-selectors", "build", "build-pciehost-packages",
                "test-devkit", "test-pciehost-hardware", "publish-pciehost-package"):
              context[f"needs.{upstream}.result"] = "success"
            main = workflow_name == "Vulcan CI"
            lane = main and variable == "true" and (release or not ref.startswith("refs/tags/"))
            bundle = main and (release or lane)
            self.assertEqual(
                evaluate_actions_expression(condition("build-pcie-card-offline-bundle"), context),
                bundle)
            context["needs.build-pcie-card-offline-bundle.result"] = (
                "success" if bundle else "skipped")
            self.assertEqual(
                evaluate_actions_expression(condition("test-pciehost-windows"), context), lane)
            # A skipped lane blocks promotion only where the lane applies.
            context["needs.test-pciehost-windows.result"] = "skipped"
            self.assertEqual(
                evaluate_actions_expression(condition("promote-latest-tag"), context), not lane)
            for name, expected in (("WINDOWS_PCIE_CI", lane), ("CARD_BUNDLE", bundle)):
              self.assertEqual(evaluate_actions_expression(flag(name), context), expected)

            results = {name: {"result": "success"} for name in re.findall(
                r"- ([\w-]+)", job("validation-complete").split("    steps:", 1)[0])}
            for name in ("publish-vulcan-artifacts", "publish-pciehost-package",
                         "promote-latest-tag"):
              results[name]["result"] = "skipped"
            results["test-pciehost-windows"]["result"] = "success" if lane else "skipped"
            results["build-pcie-card-offline-bundle"]["result"] = (
                "success" if bundle else "skipped")
            environment = {"RESULTS": json.dumps(results), "BUILD_MODE": "standard",
                           "WINDOWS_PCIE_CI": str(lane).lower(),
                           "CARD_BUNDLE": str(bundle).lower()}
            with mock.patch.dict(os.environ, environment):
              exec(compile(gate, str(MAIN_WORKFLOW), "exec"), {})
            # Release publication waits for the bundle only where it is built.
            self.assertIn("      - build-pcie-card-offline-bundle\n",
                          job("publish-offline-develop"))
            self.assertIn("startsWith(github.ref, 'refs/tags/v')",
                          condition("publish-offline-develop"))

  def test_release_qualification_can_be_rehearsed_without_publishing(self):
    source = WORKFLOW.read_text(encoding="utf-8")
    call_inputs = source.split("  workflow_call:\n", 1)[1].split("  workflow_dispatch:\n", 1)[0]
    dispatch_inputs = source.split("  workflow_dispatch:\n", 1)[1].split("\npermissions:", 1)[0]
    self.assertRegex(
        dispatch_inputs,
        r"      release_qualification:\n        description: .+\n"
        r"        type: boolean\n        default: false\n")
    self.assertNotIn("release_qualification", call_inputs)
    hardware = workflow_job(source, "hardware")
    release_checks = '"$env:VALIDATION_ONLY" -ne "true" -or "$env:RELEASE_QUALIFICATION" -eq "true"'
    for step, arguments in (
        ("Install, test both APIs, and remove exact cohort",
         '$arguments += "--release-qualification"'),
        ("Enforce release admission",
         '$arguments += @("--require-genai", "--require-release-qualification")'),
    ):
      with self.subTest(step=step):
        body = workflow_step(hardware, step)
        self.assertIn("RELEASE_QUALIFICATION: ${{ inputs.release_qualification }}\n", body)
        self.assertIn(f"if ({release_checks}) {{\n", body)
        self.assertIn(arguments, body)
    # A rehearsal never publishes: publication still requires validation_only=false.
    self.assertIn("inputs.validation_only != true", job_condition(source, "publish"))
    self.assertNotIn("release_qualification", workflow_job(source, "publish"))

  def test_direct_dispatch_resolves_dependencies_where_it_publishes(self):
    # Like Vulcan CI's matrix, resolve from the environment the publish job uses.
    source = WORKFLOW.read_text(encoding="utf-8")
    dependencies = workflow_job(source, "dependencies")
    self.assertIn("    needs: [contracts, linux_release, resolve-vulcan-config]\n", dependencies)
    step = workflow_step(dependencies, "Resolve, verify, and record dependencies")
    self.assertIn(
        "          NEAT_VULCAN_ENV: ${{ needs.resolve-vulcan-config.outputs.vulcan_env }}\n", step)
    self.assertIn('"--environment", $env:NEAT_VULCAN_ENV,', step)
    self.assertNotIn('"--environment", "production"', source)
    self.assertIn("    environment: ${{ needs.resolve-vulcan-config.outputs.vulcan_env }}\n",
                  workflow_job(source, "publish"))

  def test_workflows_calling_main_ci_skip_the_windows_lane_and_accept_the_skip(self):
    # A called workflow sees the top-level caller's name, which the Windows
    # preflight rejects for same-run Linux admission.
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    job, condition = partial(workflow_job, source), partial(job_condition, source)
    self.assertEqual(re.search(r"(?m)^name: (.+)$", source)[1], "Vulcan CI")
    self.assertIn('"${GITHUB_WORKFLOW}" != "Vulcan CI"', WORKFLOW.read_text(encoding="utf-8"))


    callers = {}
    for path in sorted(MAIN_WORKFLOW.parent.glob("*.yml")):
      text = path.read_text(encoding="utf-8")
      if "uses: ./.github/workflows/vulcan-ci.yml" in text:
        callers[path.name] = re.search(r"(?m)^name: (.+)$", text)[1].strip()
    self.assertEqual(set(callers), {"long-tests-weekly.yml", "vulcan-fuzz-nightly.yml"})
    gate_flag = re.search(r"(?m)^          WINDOWS_PCIE_CI: (.+)$", job("validation-complete"))[1]
    for workflow_name in (*callers.values(), "Vulcan CI"):
      with self.subTest(workflow=workflow_name):
        context = {
            "always()": True,
            "needs.detect-changes.outputs.windows_pcie": "true",
            "github.event_name": "push",
            "github.workflow": workflow_name,
            "inputs.offline_source_run_id": "",
            "inputs.validation_only": False,
            "inputs.build_mode": "standard",
            "github.ref_type": "branch",
            "startsWith(github.ref, 'refs/tags/v')": False,
            "needs.detect-changes.outputs.docs_only": "false",
        }
        for upstream in (
            "validate-windows-cohort-selectors", "build",
            "build-pcie-card-offline-bundle", "build-pciehost-packages",
            "test-devkit", "test-pciehost-hardware", "publish-pciehost-package"):
          context[f"needs.{upstream}.result"] = "success"
        called = workflow_name != "Vulcan CI"
        for gated in ("build-pcie-card-offline-bundle", "test-pciehost-windows"):
          self.assertEqual(evaluate_actions_expression(condition(gated), context), not called)
        context["needs.test-pciehost-windows.result"] = "skipped"
        self.assertEqual(
            evaluate_actions_expression(condition("promote-latest-tag"), context), called)
        self.assertEqual(evaluate_actions_expression(gate_flag, context), not called)

  def test_main_ci_exposes_and_forwards_exact_windows_dependency_selectors(self):
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    call_inputs = source.split("  workflow_call:\n", 1)[1].split(
        "  workflow_dispatch:\n", 1)[0]
    dispatch_inputs = source.split("  workflow_dispatch:\n", 1)[1].split(
        "\npermissions:", 1)[0]
    for name in (
        "scheduler_run_id", "scheduler_commit", "sima_cli_run_id", "sima_cli_commit"):
      with self.subTest(input=name):
        expected = f"      {name}:\n"
        self.assertIn(expected, call_inputs)
        self.assertIn(expected, dispatch_inputs)

    job = source.split("  test-pciehost-windows:\n", 1)[1].split(
        "  promote-latest-tag:", 1)[0]
    self.assertIn("- validate-windows-cohort-selectors", job)
    self.assertIn("needs.validate-windows-cohort-selectors.result == 'success'", job)
    for name in (
        "scheduler_run_id", "scheduler_commit", "sima_cli_run_id", "sima_cli_commit"):
      with self.subTest(forwarded=name):
        self.assertIn(f"      {name}: ${{{{ inputs.{name} || '' }}}}\n", job)

    for gated_job in (
        "publish-vulcan-artifacts",
        "publish-pciehost-package",
        "promote-latest-tag",
        "publish-offline-develop",
        "validation-complete",
    ):
      with self.subTest(gated_job=gated_job):
        match = re.search(
            rf"(?ms)^  {re.escape(gated_job)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", source)
        self.assertIsNotNone(match)
        self.assertIn("- validate-windows-cohort-selectors", match[1])
        if gated_job in (
            "publish-vulcan-artifacts", "publish-pciehost-package", "promote-latest-tag"):
          self.assertIn(
              "needs.validate-windows-cohort-selectors.result == 'success'", match[1])

  def test_main_ci_uses_exact_scheduler_run_for_linux_consumers(self):
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    hygiene = source.split("  code-hygiene:\n", 1)[1].split(
        "  resolve-test-runner:\n", 1)[0]
    packages = source.split("  build-pciehost-packages:\n", 1)[1].split(
        "  publish-pciehost-package:\n", 1)[0]

    for body in (hygiene, packages):
      self.assertIn("actions/create-github-app-token@v2", body)
      self.assertIn('repository: sima-neat/scheduler', body)
      self.assertIn('run-id: ${{ inputs.scheduler_run_id }}', body)
      self.assertNotIn("--workflow-name", body)
      self.assertIn('--workflow-path ".github/workflows/ci.yml"', body)
      self.assertIn(
          '--required-reusable-workflow '
          '"sima-neat/scheduler/.github/workflows/pcie-host.yml"', body)
      self.assertIn('--required-job "Verify Scheduler build and Vulcan publication"', body)

    self.assertIn('--required-artifact "scheduler-candidate"', hygiene)
    self.assertIn('--candidate "${RUNNER_TEMP}/scheduler-analysis-candidate"', hygiene)
    self.assertIn('--commit "${SCHEDULER_COMMIT}"', hygiene)

    artifact = 'scheduler-native-${{ matrix.distro }}-${{ matrix.deb_arch }}'
    self.assertIn(f'--required-artifact "{artifact}"', packages)
    self.assertIn(
        '--scheduler-candidate "${RUNNER_TEMP}/scheduler-native-candidate"', packages)
    self.assertIn('--scheduler-commit "${SCHEDULER_COMMIT}"', packages)
    self.assertIn('elif [[ -n "${SCHEDULER_NATIVE_SPEC}" ]]', packages)
    for body in (hygiene, packages):
      self.assertIn("- validate-windows-cohort-selectors", body)
      self.assertIn(
          "needs.validate-windows-cohort-selectors.result == 'success'", body)

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_scheduler_downloads_use_the_admitted_artifact_id(self):
    # Downloading by name and run would accept bytes uploaded after admission.
    main = MAIN_WORKFLOW.read_text(encoding="utf-8")
    windows = WORKFLOW.read_text(encoding="utf-8")
    linux_release = windows.split("  linux_release:\n", 1)[1].split("  card_candidate:\n", 1)[0]
    dependencies = windows.split("  dependencies:\n", 1)[1].split("  native:\n", 1)[0]
    self.assertIn(
        "    outputs:\n"
        "      scheduler_artifact_id: ${{ steps.scheduler-admission.outputs.artifact_id }}\n",
        linux_release)
    self.assertIn("needs: [contracts, linux_release, resolve-vulcan-config]", dependencies)
    step_id = "${{ steps.scheduler-admission.outputs.artifact_id }}"
    cases = (
        (main.split("  code-hygiene:\n", 1)[1],
         "Admit exact Scheduler analysis candidate", "scheduler-analysis-admission.json",
         main.split("  code-hygiene:\n", 1)[1],
         "Download exact Scheduler analysis candidate", step_id),
        (main.split("  build-pciehost-packages:\n", 1)[1],
         "Admit exact native Scheduler candidate", "scheduler-native-admission.json",
         main.split("  build-pciehost-packages:\n", 1)[1],
         "Download exact native Scheduler candidate", step_id),
        (linux_release, "Admit exact successful Scheduler Windows run",
         "scheduler-release-admission.json", dependencies,
         "Download unpublished Scheduler override",
         "${{ needs.linux_release.outputs.scheduler_artifact_id }}"),
    )
    bash = shutil.which("bash")
    self.assertIsNotNone(bash)
    for admit_job, admit_name, receipt, download_job, download_name, artifact_id in cases:
      with self.subTest(download=download_name):
        admission = workflow_step(admit_job, admit_name)
        download = workflow_step(download_job, download_name)
        self.assertIn("        id: scheduler-admission\n", admission)
        self.assertIn("uses: actions/download-artifact@v4", download)
        self.assertIn(f"          artifact-ids: {artifact_id}\n", download)
        self.assertIn("          merge-multiple: true\n", download)
        self.assertNotRegex(download, r"(?m)^          (name|pattern):")
        self.assertEqual(re.search(r"(?m)^        if: (.+)$", download)[1],
                         re.search(r"(?m)^        if: (.+)$", admission)[1])
        # The step publishes exactly the ID its admission receipt recorded.
        script = textwrap.dedent(admission.split("        run: |\n", 1)[1])
        script = script[script.index("artifact_id=$("):]
        with tempfile.TemporaryDirectory() as temporary:
          Path(temporary, receipt).write_text(
              json.dumps({"artifact": {"id": 4242, "digest": "sha256:" + "0" * 64}}),
              encoding="utf-8")
          output = Path(temporary, "github-output")
          environment = os.environ.copy()
          environment.update({"RUNNER_TEMP": temporary, "GITHUB_OUTPUT": str(output)})
          result = subprocess.run(
              [bash, "-e", "-c", script], capture_output=True, text=True, check=False,
              env=environment)
          self.assertEqual(result.returncode, 0, result.stderr)
          self.assertEqual(output.read_text(encoding="utf-8"), "artifact_id=4242\n")
          Path(temporary, receipt).write_text("{}", encoding="utf-8")
          result = subprocess.run(
              [bash, "-e", "-c", script], capture_output=True, text=True, check=False,
              env=environment)
          self.assertNotEqual(result.returncode, 0)

  def test_scheduler_actions_artifacts_are_bounded_before_use(self):
    verifier = (ROOT / "scripts/ci/pciehost/verify_linux_release_run.py").read_text(
        encoding="utf-8")
    acquisition = (ROOT / "tools/ci/acquire_scheduler.py").read_text(encoding="utf-8")
    self.assertIn("MAX_ARTIFACT_SIZE_BYTES", verifier)
    self.assertIn("exceeds the size bound", verifier)
    self.assertIn("MAX_CANDIDATE_RESOURCES", acquisition)
    self.assertIn("MAX_CANDIDATE_BYTES", acquisition)
    self.assertIn("Scheduler candidate contains undeclared resources", acquisition)

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_main_ci_exact_windows_dependency_selectors_fail_closed(self):
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    job = source.split("  validate-windows-cohort-selectors:\n", 1)[1].split(
        "\n  validate-offline:", 1)[0]
    script = textwrap.dedent(job.split("        run: |\n", 1)[1])
    bash = shutil.which("bash")
    self.assertIsNotNone(bash)

    self.assertIn("      VALIDATION_ONLY: ${{ inputs.validation_only || false }}\n", job)

    def validate(scheduler_run="", scheduler_commit="", cli_run="", cli_commit="",
                 validation_only="true"):
      environment = os.environ.copy()
      environment.update({
          "SCHEDULER_RUN_ID": scheduler_run,
          "SCHEDULER_COMMIT": scheduler_commit,
          "SIMA_CLI_RUN_ID": cli_run,
          "SIMA_CLI_COMMIT": cli_commit,
          "VALIDATION_ONLY": validation_only,
      })
      return subprocess.run(
          [bash, "-c", script], capture_output=True, text=True, check=False,
          env=environment)

    self.assertEqual(validate().returncode, 0)
    self.assertEqual(validate(validation_only="false").returncode, 0)
    self.assertEqual(validate("41", "a" * 40, "82", "b" * 40).returncode, 0)
    # A selected run can come from any branch; a publishing run must not use it.
    for selectors in (("41", "a" * 40, "", ""), ("", "", "82", "b" * 40),
                      ("41", "a" * 40, "82", "b" * 40)):
      with self.subTest(publishing_selectors=selectors):
        result = validate(*selectors, validation_only="false")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run selectors require validation_only=true", result.stderr)
    invalid = (
        ("41", "", "", "", "supplied together"),
        ("", "a" * 40, "", "", "supplied together"),
        ("0", "a" * 40, "", "", "positive integer"),
        ("41", "A" * 40, "", "", "full lowercase"),
        ("", "", "82", "short", "full lowercase"),
    )
    for scheduler_run, scheduler_commit, cli_run, cli_commit, error in invalid:
      with self.subTest(error=error):
        result = validate(scheduler_run, scheduler_commit, cli_run, cli_commit)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(error, result.stderr)

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_windows_preflight_rejects_selectors_on_publishing_runs(self):
    # The documented direct dispatches: validation, and first-release publication.
    validation = {"VALIDATION_ONLY": "true"}
    publishing = {
        "VALIDATION_ONLY": "false",
        "FIRST_RELEASE_UPGRADE_EXEMPTION": "true",
        "FIRST_RELEASE_UPGRADE_EXEMPTION_REASON": "first Windows release",
    }
    scheduler = {"SCHEDULER_RUN_ID": "41", "SCHEDULER_COMMIT": "a" * 40}
    sima_cli = {"SIMA_CLI_RUN_ID": "82", "SIMA_CLI_COMMIT": "b" * 40}
    cases = (
        ({}, validation, True),
        ({}, publishing, True),
        (scheduler, validation, True),
        ({**scheduler, **sima_cli}, validation, True),
        (scheduler, publishing, False),
        (sima_cli, publishing, False),
        ({**scheduler, **sima_cli}, publishing, False),
    )
    for selectors, mode, accepted in cases:
      with self.subTest(selectors=sorted(selectors), validation_only=mode["VALIDATION_ONLY"]):
        result = run_windows_preflight(**selectors, **mode)
        if accepted:
          self.assertEqual(result.returncode, 0, result.stderr)
        else:
          self.assertNotEqual(result.returncode, 0)
          self.assertIn("run selectors require validation_only=true", result.stderr)

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_windows_preflight_rejects_each_invalid_input_combination(self):
    exemption = {"FIRST_RELEASE_UPGRADE_EXEMPTION": "true",
                 "FIRST_RELEASE_UPGRADE_EXEMPTION_REASON": "first Windows release"}
    cases = (
        ({"LINUX_CURRENT_RUN": "true"}, "reserved for the Vulcan CI caller"),
        ({"SCHEDULER_RUN_ID": "41"}, "scheduler_run_id and scheduler_commit"),
        ({"SIMA_CLI_COMMIT": "b" * 40}, "sima_cli_run_id and sima_cli_commit"),
        ({"GITHUB_WORKFLOW": "Vulcan CI", "LINUX_CURRENT_RUN": "true", "LINUX_RUN_ID": "7"},
         "mutually exclusive"),
        ({"UPGRADE_BASE_TAG": "v0.6.1"}, "require require_upgrade=true"),
        ({"REQUIRE_UPGRADE": "true"}, "requires upgrade_base_tag"),
        ({"REQUIRE_UPGRADE": "true", "UPGRADE_BASE_TAG": "41"}, "requires upgrade_base_tag"),
        ({"REQUIRE_UPGRADE": "true", **exemption}, "first-release exemption are mutually"),
        ({"FIRST_RELEASE_UPGRADE_EXEMPTION_REASON": "why"}, "reason requires"),
        ({"VALIDATION_ONLY": "false"}, "Publishing requires a real upgrade gate"),
        ({"FIRST_RELEASE_UPGRADE_EXEMPTION": "true"}, "requires a non-empty reason"),
    )
    for inputs, error in cases:
      with self.subTest(error=error):
        result = run_windows_preflight(**{"VALIDATION_ONLY": "true", **inputs})
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(error, result.stderr)

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_main_ci_windows_runs_are_validation_only_and_pass_preflight(self):
    # A push cannot supply an upgrade baseline or first-release exemption, so
    # Vulcan CI validates Windows and only a direct dispatch can publish it.
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    job = source.split("  test-pciehost-windows:\n", 1)[1].split(
        "  promote-latest-tag:", 1)[0]
    admission = job.split("    with:\n", 1)[0]
    self.assertNotIn("refs/heads/develop", admission)
    self.assertNotIn("refs/tags/v", admission)
    forwarded = dict(re.findall(
        r"(?m)^      (\w+): (.+)$", job.split("    with:\n", 1)[1].split("    secrets:", 1)[0]))
    workflow = WORKFLOW.read_text(encoding="utf-8")
    call_inputs = workflow.split("  workflow_call:\n", 1)[1].split("  workflow_dispatch:\n", 1)[0]
    defaults = dict(re.findall(
        r"(?m)^      (\w+):\n(?:^        (?!default:).*\n)*^        default: (.*)$", call_inputs))
    step = workflow.split("- name: Validate dependency and admission selectors\n", 1)[1]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0])
    bash = shutil.which("bash")
    self.assertIsNotNone(bash)
    cases = (
        ("push", "refs/heads/develop"),
        ("push", "refs/tags/v0.7.0"),
        ("workflow_dispatch", "refs/heads/develop"),
        ("workflow_dispatch", "refs/tags/v0.7.0-rc.1"),
        ("push", "refs/heads/feature/windows"),
    )
    for event, ref in cases:
      with self.subTest(event=event, ref=ref):
        context = {
            "startsWith(github.ref, 'refs/tags/v')": ref.startswith("refs/tags/v"),
            "github.event_name": event,
            "github.ref": ref,
            "inputs.validation_only": False,
        }
        environment = os.environ.copy()
        environment["GITHUB_WORKFLOW"] = "Vulcan CI"
        for name, default in defaults.items():
          value = forwarded.get(name, default).strip()
          if re.fullmatch(r"\$\{\{ inputs\.\w+ \|\| '' \}\}|''", value):
            value = ""
          elif value.startswith("${{"):
            value = str(evaluate_actions_expression(value, context)).lower()
          environment[name.upper()] = value
        self.assertEqual(environment["VALIDATION_ONLY"], "true")
        result = subprocess.run(
            [bash, "-c", script], capture_output=True, text=True, check=False,
            env=environment)
        self.assertEqual(result.returncode, 0, result.stderr)

  def test_windows_publisher_independently_requires_final_release_context(self):
    source = WORKFLOW.read_text(encoding="utf-8")
    publish = source.split("  publish:\n", 1)[1]
    expression = re.search(r"(?m)^    if: (.+)$", publish).group(1)
    cases = (
        ("sima-neat/core", "pull_request", "refs/pull/42/merge",
         False, "success", "success", False),
        ("sima-neat/core", "push", "refs/heads/feature/windows",
         False, "success", "success", False),
        ("sima-neat/core", "push", "refs/heads/v0.7.0",
         False, "success", "success", False),
        ("sima-neat/core", "push", "refs/tags/nightly",
         False, "success", "success", False),
        ("fork/core", "push", "refs/heads/develop",
         False, "success", "success", False),
        ("sima-neat/core", "workflow_dispatch", "refs/heads/feature/windows",
         False, "success", "success", False),
        ("sima-neat/core", "workflow_call", "refs/heads/feature/windows",
         False, "success", "success", False),
        ("sima-neat/core", "workflow_call", "refs/heads/develop",
         False, "success", "success", False),
        # Only a Neat release tag publishes; a branch has no customer selector.
        ("sima-neat/core", "push", "refs/heads/develop",
         False, "success", "success", False),
        ("sima-neat/core", "workflow_dispatch", "refs/heads/develop",
         False, "success", "success", False),
        ("sima-neat/core", "push", "refs/tags/v0.7.0",
         False, "success", "success", True),
        ("sima-neat/core", "workflow_dispatch", "refs/tags/v0.7.0-rc.1",
         False, "success", "success", True),
        ("sima-neat/core", "push", "refs/heads/develop",
         True, "success", "success", False),
        ("sima-neat/core", "push", "refs/heads/develop",
         False, "failure", "success", False),
        ("sima-neat/core", "push", "refs/heads/develop",
         False, "success", "failure", False),
    )
    for repository, event, ref, validation, hardware, linux, expected in cases:
      with self.subTest(repository=repository, event=event, ref=ref, validation=validation,
                        hardware=hardware, linux=linux):
        values = {
            "startsWith(github.ref, 'refs/tags/v')": ref.startswith("refs/tags/v"),
            "github.repository": repository,
            "github.event_name": event,
            "github.ref": ref,
            "inputs.validation_only": validation,
            # Keep the upgrade term open so each row isolates its own term.
            "inputs.require_upgrade": True,
            "inputs.first_release_upgrade_exemption": False,
            "needs.hardware.result": hardware,
            "needs.linux_release.result": linux,
        }
        self.assertEqual(evaluate_actions_expression(expression, values), expected)
    first_release_values = {
        "startsWith(github.ref, 'refs/tags/v')": True,
        "github.repository": "sima-neat/core",
        "github.event_name": "workflow_dispatch",
        "github.ref": "refs/tags/v0.7.0",
        "inputs.validation_only": False,
        "inputs.require_upgrade": False,
        "inputs.first_release_upgrade_exemption": True,
        "needs.hardware.result": "success",
        "needs.linux_release.result": "success",
    }
    self.assertTrue(evaluate_actions_expression(expression, first_release_values))
    first_release_values["inputs.first_release_upgrade_exemption"] = False
    self.assertFalse(evaluate_actions_expression(expression, first_release_values))

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_publishing_run_is_rejected_before_building_unless_on_its_release_tag(self):
    source = WORKFLOW.read_text(encoding="utf-8")
    linux_release = source.split("  linux_release:\n", 1)[1].split("  card_candidate:\n", 1)[0]
    step = workflow_step(linux_release, "Require the Neat release tag before building a publication")
    self.assertIn("if: inputs.validation_only != true", step)
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    version = json.loads((ROOT / "deps" / "manifest.json").read_text())["package-version"]
    for ref_type, ref_name, accepted in (
        ("tag", f"v{version}", True),
        ("branch", "develop", False),
        ("branch", f"v{version}", False),
        ("tag", "v0.0.1", False),
        ("tag", "nightly", False),
    ):
      with self.subTest(ref_type=ref_type, ref_name=ref_name):
        result = subprocess.run(
            [shutil.which("bash"), "-c", script], cwd=ROOT, capture_output=True, text=True,
            env={**os.environ, "GITHUB_REF_TYPE": ref_type, "GITHUB_REF_NAME": ref_name})
        self.assertEqual(result.returncode == 0, accepted, result.stderr)
        if not accepted:
          self.assertIn(f"requires the release tag v{version}", result.stderr)

  def test_publication_plan_publishes_only_customer_files_under_the_tag(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, resources, _ = self.windows_candidate(root)
      (root / "core.msi.sha256").write_text(resources["core.msi"] + "  core.msi\n")
      bucket = {"core/v0.6.2/latest.tag": b"cccccccccccc\n"}

      def read(key):
        if key not in bucket:
          raise FileNotFoundError(key)  # what s3_reader raises for an HTTP 404
        return bucket[key]

      prefix, files = RELEASE.publication_plan(read, "v0.6.2", commit, root)
      self.assertEqual(prefix, "core/v0.6.2/cccccccccccc/pciehost/windows/amd64")
      # Receipts and qualification evidence stay in the Actions artifact.
      self.assertEqual([path.name for path in files],
                       [*resources, "core.msi.sha256", "metadata.json"])
      # A failed attempt left files but never exposed metadata.json: publish again.
      bucket.update({f"{prefix}/{path.name}": b"partial" for path in files[:-1]})
      bucket[f"{prefix}/metadata.json.pending"] = b"{}"
      self.assertEqual(RELEASE.publication_plan(read, "v0.6.2", commit, root)[1], files)
      # A published release is re-published only as a no-op, never with other bytes.
      bucket.update({f"{prefix}/{path.name}": path.read_bytes() for path in files})
      self.assertEqual(RELEASE.publication_plan(read, "v0.6.2", commit, root), (prefix, []))
      bucket[f"{prefix}/core.msi"] = b"rebuilt core"
      with self.assertRaisesRegex(ValueError, "already published with a different core.msi"):
        RELEASE.publication_plan(read, "v0.6.2", commit, root)
      for ref, error in (("develop", "requires the release tag v0.6.2"),
                         ("v0.6.1", "requires the release tag v0.6.2")):
        with self.subTest(ref=ref), self.assertRaisesRegex(ValueError, error):
          RELEASE.publication_plan(read, ref, commit, root)
      with self.assertRaisesRegex(ValueError, "not d{40}"):
        RELEASE.publication_plan(read, "v0.6.2", "d" * 40, root)
      # The Linux release of the tag must already have promoted this commit.
      bucket["core/v0.6.2/latest.tag"] = b"aaaaaaaaaaaa\n"
      with self.assertRaisesRegex(ValueError, "not release commit cccccccccccc"):
        RELEASE.publication_plan(read, "v0.6.2", commit, root)

  def test_s3_reader_tells_a_missing_object_from_a_failed_read(self):
    for stderr, error in (
        (b'fatal error: An error occurred (404) when calling the HeadObject operation: '
         b'Key "k" does not exist', FileNotFoundError),
        (b"fatal error: An error occurred (403) when calling the HeadObject operation: "
         b"Forbidden", ValueError)):
      with self.subTest(error=error.__name__), mock.patch.object(
          RELEASE.subprocess, "run", return_value=subprocess.CompletedProcess(
              [], 1, stdout=b"", stderr=stderr)):
        with self.assertRaises(error):
          RELEASE.s3_reader("bucket")("k")

  def test_a_newer_run_never_cancels_a_tag_run_mid_publication(self):
    workflow = WORKFLOW.read_text(encoding="utf-8")
    cancel = re.search(r"(?m)^concurrency:\n(?:  .*\n)*?  cancel-in-progress: (.+)$", workflow)[1]
    # The same rule as Vulcan CI, which publishes the Linux release of the tag.
    self.assertEqual(cancel, re.search(
        r"(?m)^concurrency:\n(?:  .*\n)*?  cancel-in-progress: (.+)$",
        MAIN_WORKFLOW.read_text(encoding="utf-8"))[1])
    for ref_type, cancelled in (("branch", True), ("tag", False)):
      with self.subTest(ref_type=ref_type):
        self.assertEqual(
            evaluate_actions_expression(cancel, {"github.ref_type": ref_type}), cancelled)
    self.assertNotIn("concurrency:", workflow_job(workflow, "publish"))

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_publish_step_writes_the_layout_the_qualifier_stages(self):
    source = WORKFLOW.read_text(encoding="utf-8")
    publish = source.split("  publish:\n", 1)[1]
    script = textwrap.dedent(
        workflow_step(publish, "Publish Windows release files").split("        run: |\n", 1)[1])
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      candidate_root = root / "runner" / "core-windows-candidate"
      candidate_root.mkdir(parents=True)
      commit, _, _ = self.windows_candidate(candidate_root)
      tools = root / "bin"
      tools.mkdir()
      (tools / "aws").write_text(textwrap.dedent("""\
          #!/bin/bash
          set -e
          echo "$*" >> "$FAKE_S3.log"
          [[ "$1 $2" == "s3 cp" ]] || exit 0
          if [[ "$3" == s3://* ]]; then
            key="${3#s3://*/}"
            [[ -f "$FAKE_S3/$key" ]] || { echo "fatal error: An error occurred (404) when \\
          calling the HeadObject operation: Key \\"$key\\" does not exist" >&2; exit 1; }
            cat "$FAKE_S3/$key"; exit
          fi
          mkdir -p "$(dirname "$FAKE_S3/${4#s3://*/}")"
          cp "$3" "$FAKE_S3/${4#s3://*/}"
          """))
      (tools / "aws").chmod(0o755)
      bucket = root / "bucket"
      environment = {
          **os.environ, "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
          "FAKE_S3": str(bucket), "ARTIFACT_BUCKET": "neat-artifacts",
          "CLOUDFRONT_DISTRIBUTION_ID": "E123", "REF_NAME": "v0.6.2", "REF_TYPE": "tag",
          "SSE_KMS_KEY_ID": "", "GITHUB_SHA": commit, "RUNNER_TEMP": str(root / "runner"),
      }

      def publish():
        return subprocess.run([shutil.which("bash"), "-c", script], cwd=ROOT,
                              capture_output=True, text=True, env=environment)

      result = publish()
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("core/v0.6.2/latest.tag", result.stderr)
      self.assertFalse(bucket.exists())

      (bucket / "core" / "v0.6.2").mkdir(parents=True)
      (bucket / "core" / "v0.6.2" / "latest.tag").write_text("cccccccccccc\n")
      result = publish()
      self.assertEqual(result.returncode, 0, result.stderr)
      candidate = validate_candidate(candidate_root)
      staged = root / "qualifier registry"
      stage_local_registry(candidate, staged)
      published = {path.relative_to(bucket).as_posix() for path in bucket.rglob("*")
                   if path.is_file()}
      self.assertEqual(
          published - {f"{RELEASE.package_prefix('v0.6.2', commit)}/metadata.json.pending"},
          {path.relative_to(staged).as_posix() for path in staged.rglob("*") if path.is_file()})
      log = Path(f"{bucket}.log").read_text()
      self.assertIn("--sse aws:kms", log)
      self.assertIn(
          "cloudfront create-invalidation --distribution-id E123 "
          "--paths /core/v0.6.2/cccccccccccc/pciehost/windows/amd64/*", log)
      # metadata.json is exposed last, after the uploaded bytes were verified.
      self.assertTrue(log.splitlines()[-2].endswith(
          "s3://neat-artifacts/core/v0.6.2/cccccccccccc/pciehost/windows/amd64/metadata.json "
          "--content-type application/json --sse aws:kms"))
      fetched = RELEASE.fetch_release(
          lambda key: (bucket / key).read_bytes(), "v0.6.2", commit, root / "fetched")
      self.assertEqual(fetched.resources, candidate.resources)

      # A re-run of the published release uploads nothing; other bytes are refused.
      uploads = log.count("\ns3 cp /")
      result = publish()
      self.assertEqual(result.returncode, 0, result.stderr)
      self.assertIn("already holds this release; nothing to publish", result.stdout)
      self.assertEqual(Path(f"{bucket}.log").read_text().count("\ns3 cp /"), uploads)
      self.assertGreater(uploads, len(candidate.resources))
      published_msi = bucket / RELEASE.package_prefix("v0.6.2", commit) / "core.msi"
      published_msi.write_bytes(b"another build")
      result = publish()
      self.assertNotEqual(result.returncode, 0)
      self.assertIn("already published with a different core.msi", result.stderr)
      self.assertEqual(published_msi.read_bytes(), b"another build")

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_branch_build_publishes_beside_the_branch_ubuntu_packages(self):
    main = MAIN_WORKFLOW.read_text(encoding="utf-8")
    publish = workflow_job(main, "publish-pciehost-package")
    step = workflow_step(publish, "Publish Windows PCIe host packages")
    self.assertIn("if: ${{ github.ref_type == 'branch' }}", step)
    self.assertIn("REF_TYPE: branch", step)
    self.assertIn("if: ${{ github.ref_type == 'branch' }}",
                  workflow_step(publish, "Download this run's Windows PCIe host cohort"))
    script = textwrap.dedent(step.split("        run: |\n", 1)[1])
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      candidate_root = root / "runner" / "core-windows-candidate"
      candidate_root.mkdir(parents=True)
      commit, _, _ = self.windows_candidate(candidate_root)
      tools = root / "bin"
      tools.mkdir()
      (tools / "aws").write_text(textwrap.dedent("""\
          #!/bin/bash
          set -e
          echo "$*" >> "$FAKE_S3.log"
          [[ "$1 $2" == "s3 cp" ]] || exit 0
          if [[ "$3" == s3://* ]]; then cat "$FAKE_S3/${3#s3://*/}"; exit; fi
          mkdir -p "$(dirname "$FAKE_S3/${4#s3://*/}")"
          cp "$3" "$FAKE_S3/${4#s3://*/}"
          """))
      (tools / "aws").chmod(0o755)
      bucket = root / "bucket"
      environment = {
          **os.environ, "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
          "FAKE_S3": str(bucket), "ARTIFACT_BUCKET": "neat-artifacts",
          "CLOUDFRONT_DISTRIBUTION_ID": "", "REF_NAME": "feature/windows-pcie",
          "REF_TYPE": "branch", "SSE_KMS_KEY_ID": "", "GITHUB_SHA": commit,
          "RUNNER_TEMP": str(root / "runner"),
      }
      # No latest.tag yet: promote-latest-tag names the commit after the gates, as on Linux.
      result = subprocess.run([shutil.which("bash"), "-c", script], cwd=ROOT,
                              capture_output=True, text=True, env=environment)
      self.assertEqual(result.returncode, 0, result.stderr)
      prefix = "core/feature%2Fwindows-pcie/cccccccccccc/pciehost/windows/amd64"
      self.assertEqual(RELEASE.package_prefix("feature/windows-pcie", commit), prefix)
      published = {path.relative_to(bucket).as_posix() for path in bucket.rglob("*")
                   if path.is_file()}
      self.assertIn(f"{prefix}/metadata.json", published)
      self.assertTrue(all(name.startswith(prefix + "/") for name in published))
      self.assertTrue(Path(f"{bucket}.log").read_text().splitlines()[-1].endswith(
          f"s3://neat-artifacts/{prefix}/metadata.json --content-type application/json "
          "--sse aws:kms"))
      fetched = RELEASE.fetch_release(
          lambda key: (bucket / key).read_bytes(), "feature/windows-pcie", commit,
          root / "fetched", require_latest=False)
      self.assertEqual(fetched.resources, validate_candidate(candidate_root).resources)
      with self.assertRaisesRegex(ValueError, "branch or tag"):
        RELEASE.publication_plan(
            lambda key: b"", "develop", commit, candidate_root, ref_type="pull_request")

  def test_producer_and_qualifier_accept_inbox_windows_powershell(self):
    inbox = "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    for module, require in ((MODULE, MODULE.require_powershell),
                            (HARDWARE, require_hardware_powershell)):
      with self.subTest(module=module.__name__), mock.patch.object(
          module.shutil, "which", side_effect=lambda name: inbox
          if name == "powershell.exe" else None):
        self.assertEqual(require(), str(Path(inbox).resolve()))

  def test_card_management_address_selects_the_matching_local_link(self):
    self.assertEqual(card_management_addresses("10.0.9.2"), ("10.0.9.2", "10.0.9.1"))
    for invalid in ("card.example", "10.0.9.3", "192.168.1.2", "::1"):
      with self.subTest(invalid=invalid), self.assertRaisesRegex(
          ValueError, "management IPv4|10[.]0[.]N[.]2"):
        card_management_addresses(invalid)

  def test_card_network_identity_binds_the_route_to_one_sima_adapter(self):
    identity = {
        "card_host": "10.0.4.2",
        "local_address": "10.0.4.1",
        "interface_index": 12,
        "interface_guid": "11111111-2222-3333-4444-555555555555",
        "interface_alias": "Ethernet 4",
        "interface_description": "SiMa.ai Virtual Ethernet Adapter #2",
        "interface_status": "Up",
        "mac_address": "00-11-22-33-44-55",
        "pnp_device_id": r"ROOT\SIMA_VETH\0004",
        "destination_prefix": "10.0.4.0/24",
        "next_hop": "0.0.0.0",
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(identity), "")
    with tempfile.TemporaryDirectory() as directory, mock.patch(
        "qualify_windows_hardware.require_powershell", return_value="powershell.exe"
    ), mock.patch(
        "qualify_windows_hardware.subprocess.run", return_value=completed
    ) as run:
      result = windows_card_network_identity(
          "10.0.4.2", Path(directory) / "network.log")

    self.assertEqual(result, identity)
    self.assertNotIn("10.0.4.2", " ".join(map(str, run.call_args.args[0])))
    self.assertEqual(run.call_args.kwargs["env"]["SIMA_PCIE_CARD_HOST"], "10.0.4.2")
    self.assertEqual(run.call_args.kwargs["env"]["SIMA_PCIE_LOCAL_ADDRESS"], "10.0.4.1")
    self.assertIn(
        "([guid]$adapter.InterfaceGuid).ToString('D')",
        Path(require_hardware_powershell.__code__.co_filename).read_text(encoding="utf-8"),
    )

  def test_card_network_identity_rejects_an_unrelated_adapter(self):
    identity = {
        "card_host": "10.0.0.2",
        "local_address": "10.0.0.1",
        "interface_index": 7,
        "interface_guid": "11111111-2222-3333-4444-555555555555",
        "interface_alias": "Ethernet",
        "interface_description": "Generic Ethernet Adapter",
        "interface_status": "Up",
        "mac_address": "00-11-22-33-44-55",
        "pnp_device_id": r"PCI\VEN_1234",
        "destination_prefix": "10.0.0.0/24",
        "next_hop": "0.0.0.0",
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(identity), "")
    with tempfile.TemporaryDirectory() as directory, mock.patch(
        "qualify_windows_hardware.require_powershell", return_value="powershell.exe"
    ), mock.patch(
        "qualify_windows_hardware.subprocess.run", return_value=completed
    ):
      with self.assertRaisesRegex(RuntimeError, "one up SiMa adapter"):
        windows_card_network_identity(
            "10.0.0.2", Path(directory) / "network.log")

  def test_windows_security_snapshot_records_driver_signing_policy(self):
    security = {
        "account": r"RUNNER\ci",
        "applocker_rule_collections": "unavailable",
        "architecture": "64-bit",
        "build_number": "26100",
        "caption": "Microsoft Windows 11 Pro",
        "driver_signing": {
            "hvci_running": True,
            "no_integrity_checks": False,
            "secure_boot": "unknown: Cmdlet not supported on this platform: 0xC0000002",
            "test_signing": "unknown: bcdedit exited 1: Access is denied.",
            "wdac_kernel_policy": "off",
            "wdac_user_mode_policy": "audit",
        },
        "is_administrator": False,
        "product_type": 1,
        "smart_app_control_state": None,
        "version": "10.0.26100",
    }
    with tempfile.TemporaryDirectory() as directory:
      log = Path(directory) / "security.log"
      completed = subprocess.CompletedProcess([], 0, json.dumps(security), "")
      with mock.patch("qualify_windows_hardware.subprocess.run", return_value=completed) as run:
        self.assertEqual(HARDWARE.windows_security_snapshot(log), security)
      script = run.call_args.args[0][-1]
      for query in ("Confirm-SecureBootUEFI", "/enum '{current}'", "testsigning",
                    "nointegritychecks", r"root\Microsoft\Windows\DeviceGuard",
                    "Win32_DeviceGuard", "SecurityServicesRunning"):
        self.assertIn(query, script)
      del security["driver_signing"]["hvci_running"]
      completed = subprocess.CompletedProcess([], 0, json.dumps(security), "")
      with mock.patch("qualify_windows_hardware.subprocess.run", return_value=completed), \
          self.assertRaisesRegex(RuntimeError, "security state was incomplete"):
        HARDWARE.windows_security_snapshot(log)

  def test_windows_platform_snapshot_records_stable_service_and_present_pnp_identity(self):
    snapshot = fixture_platform_snapshot()
    completed = subprocess.CompletedProcess([], 0, json.dumps(snapshot), "")
    with tempfile.TemporaryDirectory() as directory, mock.patch(
        "qualify_windows_hardware.require_powershell", return_value="powershell.exe"
    ), mock.patch(
        "qualify_windows_hardware.subprocess.run", return_value=completed
    ) as run:
      result = windows_platform_state_snapshot(
          Path(directory) / "windows-platform-state.log")

    self.assertEqual(result, snapshot)
    powershell = run.call_args.args[0][-1]
    self.assertIn("Get-PnpDevice -PresentOnly", powershell)
    self.assertEqual(powershell.count("($_.FriendlyName -replace ' #[0-9]+$', '')"), 2)
    self.assertNotIn("ProcessId", powershell)
    require_restored_windows_host_state(snapshot, dict(snapshot))
    changed = json.loads(json.dumps(snapshot))
    changed["service"]["configuration"]["Start"] = 3
    with self.assertRaisesRegex(RuntimeError, "configuration/state changed"):
      require_restored_windows_host_state(snapshot, changed)

  def test_windows_platform_snapshot_rejects_non_present_driver_state(self):
    snapshot = fixture_platform_snapshot()
    snapshot["pnp_devices"][0]["status"] = "Error"
    completed = subprocess.CompletedProcess([], 0, json.dumps(snapshot), "")
    with tempfile.TemporaryDirectory() as directory, mock.patch(
        "qualify_windows_hardware.require_powershell", return_value="powershell.exe"
    ), mock.patch(
        "qualify_windows_hardware.subprocess.run", return_value=completed
    ), self.assertRaisesRegex(RuntimeError, "PnP identity was invalid"):
      windows_platform_state_snapshot(Path(directory) / "platform.log")

  def test_release_signing_leaves_vendor_runtime_unsigned_by_sima(self):
    shipped = [Path("bin/sima-pcie-host.dll"), Path("bin/opencv_world500.dll")]
    self.assertEqual(MODULE.ours(shipped), [Path("bin/sima-pcie-host.dll")])
    valid = {"status": "Valid", "thumbprint": "AB", "timestamp_thumbprint": "CD"}
    MODULE.require_valid_signatures(
        [{"path": "bin/sima-pcie-host.dll", **valid},
         {"path": "bin/OpenCV_World500.dll", "status": "NotSigned"}], "candidate", "ab")
    with self.assertRaisesRegex(RuntimeError, "sima-pcie-host.dll"):
      MODULE.require_valid_signatures(
          [{"path": "bin/sima-pcie-host.dll", "status": "NotSigned"}], "candidate", "ab")

  def test_signature_path_avoids_powershell_command_line_quoting(self):
    path = Path("C:/Program Files/SimaPCIeHost/bin/plugin.dll")
    output = '{"status":"NotSigned","message":"","thumbprint":null,' \
        '"timestamp_thumbprint":null}'
    with mock.patch.object(MODULE, "run", return_value=output) as run:
      MODULE.authenticode_status(path, "powershell.exe")
    command = run.call_args.args[0]
    environment = run.call_args.kwargs["env"]
    self.assertNotIn(str(path), " ".join(map(str, command)))
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
    self.assertIn('cpack_output = work / "cpack"', source)
    self.assertIn('"-B", cpack_output', source)
    self.assertNotIn('"-B", output, "-C", "Release"', source)
    self.assertIn("validate_candidate(output)", source)

  def test_cpp_hardware_tests_mirror_linux_ctest_and_continue_after_a_failure(self):
    ctest = (ROOT / "pcie_host/tests/hardware/CTestTestfile.cmake.in").read_text(
        encoding="utf-8")
    linux_order = re.findall(r"^add_test\((\w+)", ctest, re.MULTILINE)
    linux_timeouts = {
        name: int(timeout)
        for names, timeout in re.findall(
            r"set_tests_properties\(([^)]*?)\s+PROPERTIES\s+TIMEOUT\s+(\d+)", ctest)
        for name in names.split()
    }
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      (root / "hardware-tests/lib/sima-pcie-host/tests").mkdir(parents=True)
      evidence = root / "evidence"
      evidence.mkdir()
      environment = {"SIMAPCIE_MLA_ONLY_MODEL": "mla.tar.gz",
                     "SIMAPCIE_MLA_ONLY_BF16_MODEL": "bf16.tar.gz",
                     "SIMAPCIE_GENAI_TEST_ASSETS": "assets"}
      timeouts = {}

      def logged(command, log, **options):
        timeouts[log.name[len("cpp-"):-len(".log")]] = options["timeout"]
        if log.name == "cpp-test_tensor_latency.log":
          raise RuntimeError("command failed with 1: ERROR: mean latency 14.5 ms is above 12")

      def parallel(_executable, _evidence, _environment, timeout):
        timeouts["test_tensor_parallel_queues"] = timeout

      for release in (False, True):
        report = HARDWARE.TestReport(evidence)
        args = SimpleNamespace(include_genai=True, release_qualification=release)
        output = io.StringIO()
        with mock.patch.object(HARDWARE, "run_logged", side_effect=logged) as run, \
            mock.patch.object(HARDWARE, "run_cpp_parallel_queues", side_effect=parallel), \
            mock.patch.object(HARDWARE, "run_fault_recovery_tests",
                              return_value=["fault"]) as faults, \
            contextlib.redirect_stdout(output):
          completed = HARDWARE.run_cpp_tests(args, root, evidence, environment, report)
        self.assertEqual([row[0] for row in report.rows], linux_order)
        self.assertEqual(faults.called, release)
        self.assertNotIn("test_tensor_latency", completed)
        self.assertEqual(completed[-1] == "fault", release)
      queue_blocker = next(call.args[0] for call in run.call_args_list
                           if call.args[1].name == "cpp-test_queue_blocker.log")
      self.assertEqual([str(value) for value in queue_blocker[1:]], ["--iterations", "10"])
      self.assertEqual({name: timeouts[name] for name in linux_timeouts}, linux_timeouts)
      self.assertEqual(report.failures, [
          "test_tensor_latency: command failed with 1: ERROR: mean latency 14.5 ms is above 12"])
      self.assertIn("  5 test_tensor_latency ... FAILED", output.getvalue())
      self.assertIn("::error title=test_tensor_latency::command failed with 1: ERROR: mean",
                    output.getvalue())
      with self.assertRaisesRegex(RuntimeError,
                                  "1 of 14 hardware tests failed: test_tensor_latency"):
        report.raise_failures()
      summary = root / "summary.md"
      with mock.patch.dict(os.environ, {"GITHUB_STEP_SUMMARY": str(summary)}):
        report.write_summary("FAIL")
      table = summary.read_text(encoding="utf-8")
      self.assertIn("13 of 14 tests passed.", table)
      self.assertIn("| 5 | test_tensor_latency | FAIL |", table)

  def test_failed_command_names_its_error_line_not_only_a_runner_path(self):
    with tempfile.TemporaryDirectory() as directory:
      log = Path(directory) / "cpp-test_metadata.log"
      log.write_text("+ test_metadata.exe\nloading model\nERROR: model has no outputs\n"
                     "teardown complete\n", encoding="utf-8")
      self.assertEqual(HARDWARE.failure_line(log), "ERROR: model has no outputs")
      with mock.patch.object(HARDWARE, "_run_to_log", return_value=1), \
          self.assertRaisesRegex(RuntimeError, r"failed with 1: ERROR: model has no outputs "
                                               r"\(cpp-test_metadata\.log\)"):
        HARDWARE.run_logged(["test_metadata.exe"], log)

  def test_python_failures_are_annotated_from_junit(self):
    with tempfile.TemporaryDirectory() as directory:
      evidence = Path(directory)

      def pytest(argv, log, _env, _cwd, _timeout, *, echo):
        self.assertTrue(echo)
        self.assertEqual(argv[3:5], ["-q", "-rfEs"])
        (evidence / "python-hardware.xml").write_text(
            '<testsuites><testsuite><testcase name="test_metadata_yolov8"/>'
            '<testcase name="test_tensor_latency_yolov8">'
            '<failure message="AssertionError: assert 14.51 &lt; 12.1&#10;more"/>'
            '</testcase></testsuite></testsuites>', encoding="utf-8")
        return 1

      report = HARDWARE.TestReport(evidence)
      args = SimpleNamespace(include_genai=False, release_qualification=False,
                             source_root=evidence, test_timeout_seconds=60)
      output = io.StringIO()
      with mock.patch.object(HARDWARE, "_run_to_log", side_effect=pytest), \
          contextlib.redirect_stdout(output):
        self.assertEqual(
            HARDWARE.run_python_tests(args, Path("python.exe"), evidence, {}, report), [])
      self.assertIn("::error title=test_tensor_latency_yolov8::AssertionError: assert "
                    "14.51 < 12.1\n", output.getvalue())
      self.assertEqual(report.failures, [
          "pcie_host/python/tests: pytest failed with 1: "
          "test_tensor_latency_yolov8: AssertionError: assert 14.51 < 12.1"])

  def test_hardware_qualification_bounds_blocker_and_injects_process_faults(self):
    source = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    customer_recovery = source.split(
        'registry_root = work / "complete registry with spaces"', 1
    )[1].split('result["customer_install"]', 1)[0]
    self.assertIn('"--native-only")', source)
    native_only = source.split('"--native-only")', 1)[1].split("missing_registry_root", 1)[0]
    self.assertIn("managed_python.is_file() and imports_pyneatpcie(managed_python)", native_only)
    self.assertIn("managed_python.is_file()", customer_recovery)
    self.assertNotIn('"--python-executable", python', customer_recovery)
    # Like Linux's run_pytest.sh: the whole pyneatpcie folder in one run, with the
    # hardware and GenAI cases required to pass rather than skip.
    python_tests = source.split("def run_python_tests(", 1)[1].split("\ndef ", 1)[0]
    self.assertIn('"pcie_host/python/tests"],', python_tests)
    self.assertNotIn("test_hardware_smoke.py::", python_tests)
    self.assertIn("required |= GENAI_PYTHON_TESTS", python_tests)
    self.assertIn('raise RuntimeError(f"required Python hardware tests did not pass: {missing}")',
                  python_tests)
    self.assertIn("run_genai_fault_recovery_tests(", source)
    self.assertIn('stop_windows_platform_service(evidence / "genai-service-loss-injection.log")',
                  source)
    self.assertIn('evidence / "genai-worker-loss-injection.log"', source)
    self.assertIn('completed.append("test_genai_workers_drained")', source)
    # Linux's CTest arguments (CTestTestfile.cmake.in).
    self.assertIn('[test_root / "test_queue_blocker.exe", "--iterations", "10"]', source)
    self.assertNotIn("--interval-ms", source)
    self.assertIn('completed.append("test_worker_loss_recovery")', source)
    self.assertIn('completed.append("test_client_process_death_recovery")', source)
    hardware_env = WORKFLOW.read_text(encoding="utf-8").split("  hardware:\n", 1)[1].split(
        "    steps:\n", 1)[0]
    self.assertIn('SIMAPCIE_STRESS_QUEUES: "0 1"', hardware_env)
    self.assertIn('PERF_MAX_MEAN_LATENCY_MS = "16"', source)
    self.assertIn('PERF_MIN_THROUGHPUT_FPS = "290"', source)
    self.assertIn('"--max-mean-latency-ms", PERF_MAX_MEAN_LATENCY_MS]', source)
    self.assertIn('"SIMAPCIE_PERF_MIN_THROUGHPUT_FPS": PERF_MIN_THROUGHPUT_FPS', source)
    self.assertIn('"--warmup", "100", "--iterations", "3000"', source)
    self.assertIn('command.append("--terminate-on-pull-timeout")', source)
    self.assertIn('if returncode != 1 or "pull timed out; terminating the fault-injection client"',
                  source)
    self.assertIn("pull timed out; terminating the fault-injection client",
                  (ROOT / "pcie_host/tests/hardware/test_queue_blocker.cpp").read_text())
    self.assertIn('evidence / "client-process-death-worker-cleanup.log"', source)
    self.assertIn('evidence / "client-process-death-platform-recovery.log"', source)
    self.assertIn('evidence / "client-process-death-card-readiness.log"', source)
    self.assertIn("New-LocalUser", source)
    self.assertIn("Remove-LocalUser", source)
    self.assertIn("standard_user", source)
    self.assertIn("CreateProcessWithLogonW", source)
    self.assertIn('"SimaPCIeHostAcceptance-" + secrets.token_hex(8)', source)
    self.assertIn('evidence / "standard-user-python-install.log"', source)
    self.assertIn('"SIMAPCIE_GENAI_MEDIA_DIRECTORY"', source)
    self.assertIn('"SIMAPCIE_GENAI_MEDIA_SERVE_ROOT"', source)
    workflow = WORKFLOW.read_text(encoding="utf-8")
    # Like Linux, CI sets no custom media root; the qualifier still honours the pair.
    self.assertNotIn("SIMA_WINDOWS_PCIE_GENAI_MEDIA_DIRECTORY", workflow)
    # Like the Linux card job, the lane has no switch that leaves GenAI out.
    self.assertNotIn("include_genai", workflow + MAIN_WORKFLOW.read_text(encoding="utf-8"))
    self.assertEqual(workflow.count('            "--include-genai"\n'), 2)
    for name in ("SIMAPCIE_GENAI_MODELS_PATH", "SIMAPCIE_GENAI_TEST_ASSETS"):
      self.assertIn(f'            "{name}"', workflow)
    self.assertIn("media overrides require directory and service root as a pair", source)
    self.assertIn(
        "install_root, candidate.wheel, wheelhouse, candidate.root, work, evidence", source)
    self.assertIn('[sys.executable, "-m", "venv", "--copies", portable_root]', source)
    self.assertIn('"--force-reinstall", wheel', source)
    driver = (SCRIPT_DIR / "standard_user_genai.py").read_text(encoding="utf-8")
    self.assertIn("prefix not in module.parents", driver)
    self.assertIn("p.genai.ConnectionOptions()", driver)
    self.assertIn("common_application_data().resolve(strict=True)", driver)
    self.assertIn('read_serve_mapping(configuration, options.media_serve_root)', driver)
    self.assertIn('[executable, "asr", assets]', driver)
    # What the temporary user lacks: the card's host keys and a readable key copy; and
    # the system variables the ProgramData known folder and Windows OpenSSH need.
    self.assertIn('shutil.copyfile(payload["known_hosts"], known_hosts)', driver)
    self.assertIn('"*S-1-5-32-544:(OI)(CI)(F)" /Q | Out-Null', source)
    self.assertIn('"PROGRAMDATA", "SYSTEMDRIVE"', driver)
    self.assertNotIn("Path(sys.base_prefix).resolve", source)
    self.assertIn('(private_root, "temporary standard-user private input")', source)
    self.assertIn("TerminateProcess", source)
    self.assertIn("$cleanupErrors", source)
    self.assertIn('"simaai_pcie_test.exe"', source)
    self.assertIn("Device recovery successful", source)
    self.assertIn("Stop-Service -InputObject $service", source)
    self.assertIn('evidence / "card-candidate-platform-start.log"', source)
    self.assertIn('evidence / "card-stage-platform-recovery.log"', source)
    with mock.patch.object(HARDWARE, "require_powershell", return_value="powershell.exe"), \
        mock.patch.object(HARDWARE, "run_logged") as run:
      HARDWARE.restart_windows_platform_service(Path("restart.log"))
    self.assertIn("Restart-Service -InputObject $service", run.call_args.args[0][-1])
    self.assertIn("process.kill()", source)
    self.assertIn('"/fa"', source)
    self.assertIn('"gst-inspect-1.0.exe"', source)
    self.assertIn('install_root / "bin" / "gstreamer-1.0" / "gstneatpciehost.dll"', source)
    self.assertIn('work / "native customer downloads with spaces"', source)
    self.assertIn('work / "missing wheel downloads with spaces"', source)
    self.assertIn('work / "invalid wheel downloads with spaces"', source)
    self.assertIn('work / "recovery downloads with spaces"', source)
    self.assertIn('if card_state.get("stage"):', source)
    self.assertIn('install_attempted=card_state.get("install_started", False))', source)
    self.assertIn("card_install_attempted = True", source)
    # The card keeps the candidate, as on Linux: no firmware restore remains.
    for retired in ("restore_card_candidate", "firmware-broker", "baseline_firmware",
                    "tutorial_facade", "SIMAPCIE_CARD_ENV", "SIMA_NEAT_PCIE_GENAI_CARD_ENV"):
      self.assertNotIn(retired, source)
    self.assertIn("stdin=subprocess.DEVNULL", source)
    self.assertIn("isolated_sima_cli_environment(", source)
    cleanup = (SCRIPT_DIR / "cleanup_windows_hardware.py").read_text(encoding="utf-8")
    self.assertIn("hardware work directory remained after recovery", cleanup)
    self.assertIn("--card-sudo-password-stdin", cleanup)
    self.assertIn("remove_card_stage(args, stage, evidence, install_attempted=True)", cleanup)
    self.assertIn('evidence / "recovery-platform-service.log"', cleanup)

  def test_genai_qualification_requires_platform_media_contract(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      environment = {}
      for name in (
          "SIMAPCIE_YOLOV8_MODEL",
          "SIMAPCIE_MLA_ONLY_MODEL",
          "SIMAPCIE_MLA_ONLY_BF16_MODEL",
          "SIMAPCIE_TEST_IMAGE",
          "SIMAPCIE_BOXDECODE_IMAGE",
      ):
        path = root / name
        path.write_bytes(b"fixture")
        environment[name] = str(path)
      for name in ("SIMAPCIE_GENAI_MODELS_PATH", "SIMAPCIE_GENAI_TEST_ASSETS"):
        path = root / name
        path.mkdir()
        environment[name] = str(path)

      HARDWARE.require_assets(environment, True)
      media = root / "Platform media with spaces"
      media.mkdir()
      environment["SIMAPCIE_GENAI_MEDIA_DIRECTORY"] = str(media.resolve())
      with self.assertRaisesRegex(RuntimeError, "as a pair"):
        HARDWARE.require_assets(environment, True)
      environment["SIMAPCIE_GENAI_MEDIA_SERVE_ROOT"] = "data"
      HARDWARE.require_assets(environment, True)
      del environment["SIMAPCIE_GENAI_MEDIA_DIRECTORY"]
      with self.assertRaisesRegex(RuntimeError, "as a pair"):
        HARDWARE.require_assets(environment, True)

  def test_hardware_asset_manifest_is_deterministic_for_vision_and_genai(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      vision_environment = fixture_asset_environment(root)
      vision = create_asset_manifest(
          vision_environment, False, "vision-assets-r7",
          {"SIMAPCIE_YOLOV8_MODEL": {"repository": "model-zoo", "revision": "abc123"}})
      self.assertFalse(vision["include_genai"])
      self.assertEqual(vision["genai"], {})
      self.assertEqual(
          list(vision["vision"]), sorted(VISION_ASSETS))

      genai_environment = fixture_asset_environment(root / "with-genai", True)
      genai = create_asset_manifest(
          genai_environment, True, "complete-assets-r9",
          {name: {"repository": "genai-assets", "revision": f"rev-{index}"}
           for index, name in enumerate(GENAI_ASSETS)})
      repeated = create_asset_manifest(
          genai_environment, True, "complete-assets-r9",
          {name: {"revision": f"rev-{index}", "repository": "genai-assets"}
           for index, name in reversed(tuple(enumerate(GENAI_ASSETS)))})
      self.assertEqual(genai, repeated)
      self.assertEqual(
          [item["path"] for item in genai["genai"][GENAI_ASSETS[0]]["files"]],
          ["nested/a.bin", "z.bin"])
      self.assertEqual(
          genai["genai"][GENAI_ASSETS[0]]["source_revision"]["revision"],
          "rev-0")

  def test_hardware_asset_manifest_detects_changed_files_and_directory_mutation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      environment = fixture_asset_environment(root, True)
      manifest = create_asset_manifest(environment, True, "assets-r1")
      path = root / "asset-manifest.json"
      path.write_text(json.dumps(manifest), encoding="utf-8")
      self.assertEqual(verify_asset_manifest(path, environment, True), manifest)

      Path(environment["SIMAPCIE_YOLOV8_MODEL"]).write_bytes(b"changed!")
      with self.assertRaisesRegex(ValueError, "differ from the admitted"):
        verify_asset_manifest(path, environment, True)

      environment = fixture_asset_environment(root / "replacement", True)
      manifest = create_asset_manifest(environment, True, "assets-r2")
      path.write_text(json.dumps(manifest), encoding="utf-8")
      models = Path(environment["SIMAPCIE_GENAI_MODELS_PATH"])
      (models / "new-model.bin").write_bytes(b"new")
      with self.assertRaisesRegex(ValueError, "differ from the admitted"):
        verify_asset_manifest(path, environment, True)

  def test_hardware_asset_manifest_rejects_links_escape_and_case_ambiguity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      environment = fixture_asset_environment(root, True)
      models = Path(environment["SIMAPCIE_GENAI_MODELS_PATH"])
      link = models / "linked.bin"
      try:
        link.symlink_to(models / "z.bin")
      except OSError as error:
        self.skipTest(f"symlink fixture unavailable: {error}")
      with self.assertRaisesRegex(ValueError, "link or reparse point"):
        create_asset_manifest(environment, True, "assets-r1")
      link.unlink()

      (models / "CASE.bin").write_bytes(b"upper")
      (models / "case.bin").write_bytes(b"lower")
      case_entries = [
          entry.name for entry in models.iterdir()
          if entry.name.casefold() == "case.bin"
      ]
      if len(case_entries) == 2:
        with self.assertRaisesRegex(ValueError, "case-ambiguous"):
          create_asset_manifest(environment, True, "assets-r1")
      else:
        self.assertEqual(len(case_entries), 1)

      invalid = fixture_asset_manifest(include_genai=True)
      invalid["genai"][GENAI_ASSETS[0]]["files"][0]["path"] = "../escape.bin"
      with self.assertRaisesRegex(ValueError, "unsafe or non-canonical"):
        normalize_asset_manifest(invalid)
      invalid = fixture_asset_manifest(include_genai=True)
      duplicate = dict(invalid["genai"][GENAI_ASSETS[0]]["files"][0])
      invalid["genai"][GENAI_ASSETS[0]]["files"].append(duplicate)
      with self.assertRaisesRegex(ValueError, "duplicate or case-ambiguous"):
        normalize_asset_manifest(invalid)
      invalid = fixture_asset_manifest(include_genai=True)
      duplicate = dict(invalid["genai"][GENAI_ASSETS[0]]["files"][0])
      duplicate["path"] = duplicate["path"].upper()
      invalid["genai"][GENAI_ASSETS[0]]["files"].append(duplicate)
      with self.assertRaisesRegex(ValueError, "case-ambiguous"):
        normalize_asset_manifest(invalid)

  def test_hardware_asset_manifest_is_verified_before_local_card_lock(self):
    source = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    self.assertLess(
        source.index("asset_manifest = verify_asset_manifest("),
        source.index("with ExclusiveLock(args.lock_file"))

  def test_standard_user_wrapper_uses_json_inputs_and_clears_media_overrides(self):
    wrapper = standard_user_wrapper()
    self.assertIn('set "SIMAPCIE_GENAI_MEDIA_DIRECTORY="', wrapper)
    self.assertIn('set "SIMAPCIE_GENAI_MEDIA_SERVE_ROOT="', wrapper)
    self.assertIn('call "%~1" > "%~2" 2>&1', wrapper)
    self.assertIn('call "%~3" "%~4" "%~5" "%~6" "%~7" >> "%~2" 2>&1', wrapper)
    self.assertNotIn("%SIMA_", wrapper)
    self.assertNotIn("if errorlevel", wrapper)
    self.assertEqual(wrapper.count('if not "%errorlevel%"=="0" exit /b'), 2)
    self.assertNotIn("C:\\", wrapper)
    self.assertNotIn("{", wrapper)

  def test_standard_user_wrapper_paths_are_passed_on_the_command_line(self):
    # CreateProcessWithLogonW gives the child the new user's environment, not ours.
    source = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    block = source[source.index("$wrapperArguments = foreach"):
                   source.index("$exitCode = [SimaStandardUserProcess]::Run(")]
    listed, command = block.split("$arguments =")
    self.assertEqual(re.findall(r"\$env:(SIMA_STANDARD_USER_\w+)", listed), [
        "SIMA_STANDARD_USER_DOCTOR", "SIMA_STANDARD_USER_RUN_LOG",
        "SIMA_STANDARD_USER_RUNNER", "SIMA_STANDARD_USER_PYTHON",
        "SIMA_STANDARD_USER_DRIVER", "SIMA_STANDARD_USER_PAYLOAD",
        "SIMA_STANDARD_USER_RECEIPT"])
    self.assertIn("""'/d /c ""' + $env:SIMA_STANDARD_USER_WRAPPER + '"'""", command)
    self.assertIn("($wrapperArguments -join '') + '\"'", command)
    self.assertIn("-gt 1024", command)

  def test_standard_user_step_rejects_a_base_python_standard_users_cannot_read(self):
    source = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    self.assertIn("Join-Path $env:SIMA_STANDARD_USER_PYTHON_ROOT 'pyvenv.cfg'", source)
    self.assertIn("Where-Object { $_ -match '^\\s*home\\s*=' }", source)
    self.assertIn("$standardUserSids = 'S-1-1-0', 'S-1-5-4', 'S-1-5-11', 'S-1-5-32-545'", source)
    self.assertIn("(Get-Acl -LiteralPath $pythonHome).GetAccessRules(", source)
    self.assertIn("$_.FileSystemRights.HasFlag($readAndExecute)", source)
    self.assertIn("with a Python installed for all users", source)
    self.assertLess(source.index("if (-not $pythonReadable)"), source.index("New-LocalUser"))

  def test_standard_user_child_gets_and_returns_session_window_access(self):
    source = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    for name in ("GetProcessWindowStation", "GetThreadDesktop", "OpenWindowStation",
                 "OpenDesktop", "GetUserObjectSecurity", "SetUserObjectSecurity"):
      self.assertIn(f" {name}(", source)
    self.assertIn("ref uint information, byte[] descriptor", source)
    self.assertIn("const uint access = 0x00060000;  // READ_CONTROL | WRITE_DAC", source)
    self.assertIn("if (acl == null) return;", source)
    self.assertIn("grant ? 0x0002037F : 0", source)
    self.assertIn("grant ? 0x000201FF : 0", source)
    self.assertIn('"Win32 error " + error', source)
    grant = "[SimaStandardUserProcess]::GrantSessionWindowAccess([string]$user.SID)"
    self.assertLess(source.index("$created = $true"), source.index("$windowAccessGranted = $true"))
    self.assertLess(source.index("$windowAccessGranted = $true"), source.index(grant))
    self.assertLess(
        source.index(grant), source.index("$exitCode = [SimaStandardUserProcess]::Run("))
    cleanup = source[source.index("} finally {\n  if ($created) {"):]
    self.assertLess(
        cleanup.index("[SimaStandardUserProcess]::RevokeSessionWindowAccess([string]$user.SID)"),
        cleanup.index("Remove-LocalUser -Name $name"))

  @unittest.skipUnless(os.name == "nt", WINDOWS_CMD_SKIP)
  def test_standard_user_wrapper_runs_from_arguments_without_inherited_variables(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory) / "standard user work"
      root.mkdir()
      doctor = root / "doctor stub.cmd"
      runner = root / "runner stub.cmd"
      runner.write_text("@echo off\necho runner %*\nexit /b 0\n", encoding="ascii")
      wrapper = root / "standard-user-activation.cmd"
      wrapper.write_text(standard_user_wrapper(), encoding="utf-8")
      run_log = root / "standard-user-activation.log"
      python, driver, payload, receipt = (
          root / "python.exe", root / "driver.py", root / "payload.json",
          root / "receipt.json")
      paths = [doctor, run_log, runner, python, driver, payload, receipt]
      command_processor = Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"
      command = (f'"{command_processor}" /d /c ""{wrapper}"'
                 + "".join(f' "{path}"' for path in paths) + '"')
      # -1073741502 is 0xC0000142, the exit code of a program whose DLLs failed to load.
      for doctor_exit, expected in ((0, 0), (-1073741502, 98)):
        with self.subTest(doctor_exit=doctor_exit):
          doctor.write_text(
              f"@echo off\necho doctor ran\nexit /b {doctor_exit}\n", encoding="ascii")
          result = subprocess.run(
              command, env={"SystemRoot": os.environ["SystemRoot"]}, cwd=root,
              check=False)
          log = run_log.read_text(encoding="utf-8", errors="replace")
          self.assertEqual(result.returncode, expected, log)
          self.assertIn("doctor ran", log)
          if expected == 0:
            self.assertIn(f'runner "{python}" "{driver}" "{payload}" "{receipt}"', log)

  def test_standard_user_platform_config_binding_is_deterministic(self):
    with tempfile.TemporaryDirectory() as directory:
      config = Path(directory) / "simaai-mla-daemon.conf"
      config.write_text(
          '\ufeff[serve]\ndata = "C:\\ProgramData\\SiMa.ai\\data"\n',
          encoding="utf-8")
      self.assertEqual(
          read_serve_mapping(config, "data"),
          Path(r"C:\ProgramData\SiMa.ai\data"))
      with self.assertRaisesRegex(RuntimeError, r"\[serve\] missing"):
        read_serve_mapping(config, "missing")
      # Platform 3.0 layout: daemon options precede the [recv] and [serve] sections.
      config.write_text(
          "# comment\nmax-clients  = 64\nnotify-queue = 256\n\n"
          "[recv]\nincoming = C:\\ProgramData\\SiMa.ai\\incoming\n\n"
          "[serve]\ndata    = C:\\ProgramData\\SiMa.ai\\data\n"
          "models  = C:\\ProgramData\\SiMa.ai\\models\n",
          encoding="utf-8")
      self.assertEqual(
          read_serve_mapping(config, "data"),
          Path(r"C:\ProgramData\SiMa.ai\data"))
      config.write_text("[serve\ndata=x\n", encoding="utf-8")
      with self.assertRaisesRegex(RuntimeError, "cannot read Platform daemon configuration"):
        read_serve_mapping(config, "data")

  def test_standard_user_non_genai_path_preserves_installed_python_assertions(self):
    with tempfile.TemporaryDirectory() as directory:
      payload = Path(directory) / "payload.json"
      payload.write_text(
          json.dumps({"environment": {}, "include_genai": False}), encoding="utf-8")
      installed = Path(sys.prefix) / "Lib" / "site-packages" / "pyneatpcie" / "__init__.py"
      module = SimpleNamespace(
          __file__=str(installed), Model=object(), Runtime=object(), Tensor=object())
      with mock.patch.dict(sys.modules, {"pyneatpcie": module}), \
          mock.patch.dict(os.environ, {}, clear=True), \
          mock.patch.object(
              STANDARD_USER_GENAI, "is_user_an_administrator", return_value=False):
        self.assertEqual(run_standard_user_genai(payload), {
            "process_is_administrator": False,
            "python": "PASS",
            "standard_user": "PASS",
        })
      with mock.patch.dict(sys.modules, {"pyneatpcie": module}), \
          mock.patch.dict(os.environ, {
              "SIMAPCIE_GENAI_MEDIA_DIRECTORY": r"C:\admin-only",
          }, clear=True), \
          mock.patch.object(
              STANDARD_USER_GENAI, "is_user_an_administrator", return_value=False), \
          self.assertRaisesRegex(RuntimeError, "retained forbidden override"):
        run_standard_user_genai(payload)

  def test_standard_user_sentinel_is_removed_on_success_and_failure(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      prove_writable_sentinel(root, "success")
      self.assertFalse((root / ".neat-standard-user-success").exists())
      with mock.patch.object(Path, "read_text", return_value="changed"):
        with self.assertRaisesRegex(RuntimeError, "contents changed"):
          prove_writable_sentinel(root, "failure")
      self.assertFalse((root / ".neat-standard-user-failure").exists())

  def test_standard_user_private_input_cleanup_removes_nested_key(self):
    with tempfile.TemporaryDirectory() as directory:
      private = Path(directory) / "standard-user-private"
      private.mkdir()
      (private / "card-ssh-key").write_bytes(b"fixture")
      remove_qualification_tree(private, "temporary standard-user private input")
      self.assertFalse(private.exists())
      remove_qualification_tree(private, "temporary standard-user private input")

  def test_customer_process_environment_excludes_ci_runtime_overlays(self):
    environment = customer_process_environment({
        "COMSPEC": r"C:\Windows\System32\cmd.exe",
        "PROGRAMFILES": r"C:\Program Files",
        "SYSTEMROOT": r"C:\Windows",
        "Path": r"C:\producer\bin",
        "CMAKE_PREFIX_PATH": r"C:\producer\prefix",
        "GSTREAMER_ROOT_X86_64": r"C:\producer\gstreamer",
        "GST_PLUGIN_PATH_1_0": r"C:\producer\plugins",
        "PYTHONPATH": r"C:\producer\python",
        "SIMA_PCIE_HOST_LIBRARY_PATH": r"C:\producer\core",
        "SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH": r"C:\producer\runtime",
        "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR": r"C:\producer\platform",
    })
    self.assertEqual(environment["PROGRAMFILES"], r"C:\Program Files")
    self.assertNotIn("producer", environment["PATH"].lower())
    for name in (
        "CMAKE_PREFIX_PATH", "GSTREAMER_ROOT_X86_64", "GST_PLUGIN_PATH_1_0",
        "PYTHONPATH", "SIMA_PCIE_HOST_LIBRARY_PATH",
        "SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH", "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR",
    ):
      self.assertNotIn(name, environment)

  def test_sima_cli_receipts_and_customer_skills_are_confined_to_disposable_profile(self):
    with tempfile.TemporaryDirectory() as directory:
      work = Path(directory) / "work"
      home = work / "customer home"
      wheelhouse = Path(directory) / "wheelhouse"
      with mock.patch.dict(os.environ, {
          "CLAUDE_HOME": r"C:\real-profile\.claude",
          "CODEX_HOME": r"C:\real-profile\.codex",
          "PYTHONPATH": r"C:\producer",
          "SIMA_CLI_HOME": r"C:\real-profile\.sima-cli",
          "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR": r"C:\simaai",
      }, clear=True):
        environment = isolated_sima_cli_environment(home, wheelhouse, work)
      self.assertNotIn("PYTHONPATH", environment)
      # sima-cli runs the installed doctor, which must locate Platform unaided.
      self.assertNotIn("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR", environment)
      for name in ("CLAUDE_HOME", "CODEX_HOME", "SIMA_CLI_HOME", "PIP_CACHE_DIR"):
        self.assertTrue(Path(environment[name]).is_relative_to(work.resolve()))
      self.assertEqual(environment["USERPROFILE"], str(home.resolve()))
      self.assertEqual(environment["HOME"], str(home.resolve()))
      # No console is attached in CI, so the install must not wait for a card password.
      self.assertEqual(environment["SIMAPCIE_SKIP_SETUP"], "1")
      with self.assertRaisesRegex(ValueError, "inside the disposable work tree"):
        isolated_sima_cli_environment(work.parent / "outside", wheelhouse, work)

  def test_standard_user_python_is_fresh_and_installs_exact_candidate(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      portable = root / "fresh-python"
      wheelhouse = root / "wheelhouse"
      wheel = root / "pyneatpcie-0.6.2-cp314-cp314-win_amd64.whl"
      evidence = root / "evidence"
      wheelhouse.mkdir()
      evidence.mkdir()
      wheel.write_bytes(b"fixture")

      def create_venv(command, **kwargs):
        self.assertEqual(
            command, [sys.executable, "-m", "venv", "--copies", portable])
        self.assertTrue(kwargs["check"])
        scripts = portable / "Scripts"
        scripts.mkdir(parents=True)
        (scripts / "python.exe").write_bytes(b"fixture")
        return subprocess.CompletedProcess(command, 0)

      with mock.patch(
          "qualify_windows_hardware.subprocess.run", side_effect=create_venv), \
          mock.patch("qualify_windows_hardware.run_logged") as install:
        python = create_standard_user_python(
            wheel, wheelhouse, portable, evidence)

      self.assertEqual(python, portable / "Scripts" / "python.exe")
      install_command = install.call_args.args[0]
      self.assertEqual(install_command[-1], wheel)
      self.assertIn("--no-index", install_command)
      self.assertIn("--force-reinstall", install_command)
      self.assertEqual(
          install.call_args.args[1], evidence / "standard-user-python-install.log")

  def test_customer_process_activation_runs_doctor_and_python_cleanly(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      doctor = root / "bin" / "sima-pcie-host-doctor.cmd"
      doctor.parent.mkdir()
      doctor.write_text("@exit /b 0\n", encoding="utf-8")
      python = root / "python.exe"
      python.write_bytes(b"fixture")
      base = {
          "ComSpec": r"C:\Windows\System32\cmd.exe",
          "ProgramFiles": r"C:\Program Files",
          "SystemRoot": r"C:\Windows",
          "PATH": r"C:\producer\bin",
          "GSTREAMER_ROOT_X86_64": r"C:\producer\gstreamer",
          "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR": r"C:\producer\platform",
      }
      with mock.patch.dict(os.environ, base, clear=True), \
           mock.patch("qualify_windows_hardware.run_logged") as run:
        verify_customer_process_activation(root, python, root)
      self.assertEqual(run.call_count, 2)
      doctor_call, python_call = run.call_args_list
      self.assertEqual(doctor_call.args[0][1:3], ["/d", "/c"])
      self.assertEqual(doctor_call.args[0][3], doctor)
      self.assertEqual(python_call.args[0][0], python)
      for call in (doctor_call, python_call):
        self.assertNotIn("GSTREAMER_ROOT_X86_64", call.kwargs["env"])
        self.assertNotIn("SIMA_WINDOWS_PLATFORM_RUNTIME_DIR", call.kwargs["env"])

  def test_platform_device_recovery_requires_vendor_tool(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      with self.assertRaisesRegex(RuntimeError, "recovery tool is unavailable"):
        recover_windows_platform_device(root, root / "recovery.log")

  def test_platform_device_recovery_uses_provisioned_vendor_tool(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
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

  def test_genai_fault_harness_injects_worker_and_service_loss_then_restarts(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      evidence = root / "evidence"
      evidence.mkdir()
      args = SimpleNamespace(source_root=root, test_timeout_seconds=900)
      processes = []
      for _ in range(2):
        process = mock.Mock()
        process.poll.return_value = 0
        process.wait.return_value = 0
        processes.append(process)

      def record_fault(_process, log, _marker, _timeout):
        log.write_text(
            "GENAI_FAULT_READY\nGENAI_FAULT_OBSERVED: injected fixture fault\n",
            encoding="utf-8")
      with mock.patch.object(HARDWARE.subprocess, "Popen", side_effect=processes), \
           mock.patch.object(
               HARDWARE, "wait_for_log", side_effect=record_fault) as wait_for_log, \
           mock.patch.object(HARDWARE, "run_logged") as run_logged, \
           mock.patch.object(HARDWARE, "ssh_command", return_value=["ssh.exe", "card"]), \
           mock.patch.object(HARDWARE, "stop_windows_platform_service") as stop_service, \
           mock.patch.object(HARDWARE, "restart_windows_platform_service") as restart_service, \
           mock.patch.object(HARDWARE, "wait_for_no_genai_workers") as wait_for_no_workers:
        report = HARDWARE.TestReport(evidence)
        completed = HARDWARE.run_genai_fault_recovery_tests(
            args, Path("python.exe"), evidence, {"TEST": "1"}, report)
      self.assertEqual(report.failures, [])

      self.assertEqual(completed, [
          "test_genai_worker_loss_recovery", "test_genai_restart_after_worker_loss",
          "test_genai_service_loss_recovery", "test_genai_restart_after_service_loss",
      ])
      self.assertEqual(wait_for_log.call_count, 2)
      stop_service.assert_called_once_with(evidence / "genai-service-loss-injection.log")
      restart_service.assert_called_once_with(evidence / "genai-service-loss-recovery.log")
      self.assertEqual(wait_for_no_workers.call_count, 2)
      commands = [call.args[0] for call in run_logged.call_args_list]
      self.assertIn(["ssh.exe", "card"], commands)
      self.assertEqual(sum(command[-1] == "smoke" for command in commands), 2)

  def test_wheel_payload_reader_checks_crc(self):
    with tempfile.TemporaryDirectory() as directory:
      wheel = Path(directory) / "test.whl"
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
    rollback_log = root / "msi-transaction-rollback.log"
    rollback_log.write_text(
        "Action start: FailAfterOwnedState\nWindows Installer rollback\n",
        encoding="utf-8")
    rollback_receipt = root / "msi-transaction-rollback.json"
    rollback_receipt.write_text(json.dumps({
        "schema_version": 1,
        "status": "PASS",
        "fixture": {
            "custom_action": "FailAfterOwnedState",
            "msi_sha256": "1" * 64,
            "product_code": "{5C5FB61E-7355-4B72-87FD-5FDD4CC2715B}",
        },
        "failure": {"expected_exit_code": 1603, "observed_exit_code": 1603},
        "mid_transaction": {"file": "PASS", "registry": "PASS"},
        "rollback": {
            "install_directory": "PASS",
            "product_registration": "PASS",
            "registry": "PASS",
            "sentinel": "PASS",
        },
        "preserved_cohort": {
            "file_inventory": "PASS",
            "file_inventory_sha256": "2" * 64,
            "installed_consumer": "PASS",
            "plugin_discovery": "PASS",
            "scheduler": {
                "product_code": "{11111111-2222-3333-4444-555555555555}",
                "state_before": "DEFAULT",
                "state_after": "DEFAULT",
            },
            "core": {
                "product_code": "{AAAAAAAA-BBBB-CCCC-DDDD-EEEEEEEEEEEE}",
                "state_before": "DEFAULT",
                "state_after": "DEFAULT",
            },
        },
        "installer_log_sha256": sha256(rollback_log),
    }), encoding="utf-8")
    standard_user_genai = (
        root / "hardware-tests" / "lib" / "sima-pcie-host" / "tests"
        / "test_genai_run.exe")
    standard_user_genai.parent.mkdir(parents=True)
    standard_user_genai.write_bytes(b"exact-candidate-standard-user-genai" + commit.encode("ascii"))
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
        "platform": {"version": "3.0.0.2"},
    }
    scheduler_dependencies = root / "scheduler-windows-dependencies.json"
    scheduler_dependencies.write_text(json.dumps(dependency_data), encoding="utf-8")
    # Scheduler's installed Platform record names the DLLs the card PC must hold.
    platform_record = root / "scheduler-platform-windows.json"
    platform_record.write_text(json.dumps({
        "platform_version": "3.0.0.2",
        "runtime_dlls": {"sima-platform.dll": "d" * 64},
    }), encoding="utf-8")
    dependency_data["dependency_runtime"] = [{
        "files": [{"name": "opencv_world500.dll", "sha256": "9" * 64}],
    }]
    dependencies.write_text(json.dumps(dependency_data), encoding="utf-8")
    qualified_dependencies = dependency_identity(
        root, require_dependency_runtime=True)
    platform_host_state = {
        "pnp_devices": [
            {
                "driver_version": "3.0.0.2",
                "friendly_name": "SiMa.ai MLA Device",
                "instance_id": r"PCI\VEN_1F06&DEV_3123\4&1",
                "status": "OK",
            },
            {
                "driver_version": "3.0.0.2",
                "friendly_name": "SiMa.ai Virtual Ethernet Adapter",
                "instance_id": r"ROOT\SIMA_VETH\0000",
                "status": "OK",
            },
        ],
        "service": {
            "configuration": {"ImagePath": "simaai_mla_daemon.exe", "Start": 2},
            "name": "SiMaMlaDaemon",
            "state": "Running",
        },
    }
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
            standard_user_genai.relative_to(root).as_posix(): sha256(standard_user_genai),
            native_junit.name: sha256(native_junit),
            qualification_resource.name: sha256(qualification_resource),
            rollback_log.name: sha256(rollback_log),
            rollback_receipt.name: sha256(rollback_receipt),
            scheduler_signatures.name: sha256(scheduler_signatures),
            dependencies.name: sha256(dependencies),
            scheduler_dependencies.name: sha256(scheduler_dependencies),
            platform_record.name: sha256(platform_record),
            core_signatures.name: sha256(core_signatures),
        },
        "repository": "sima-neat/core",
        "scheduler_source_commit": "d" * 40,
    }), encoding="utf-8")
    qualification = Path(tempfile.mkdtemp()) / "qualification.json"
    self.addCleanup(shutil.rmtree, qualification.parent)
    asset_manifest_path = qualification.with_name("hardware-asset-manifest.json")
    asset_manifest = fixture_asset_manifest()
    asset_manifest_path.write_text(
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    card = {
        "firmware": "modalix-cvu-fw",
        "genai_workers": "0",
        "machine": "aarch64",
        "platform_release": "3.0.0_master_B2501",
        "platform_sha256": "1" * 64,
        "remoteproc_state": "running",
        "workers": "0",
    }
    network = {
        "card_host": "10.0.0.2",
        "interface_description": "SiMa.ai Virtual Ethernet Adapter",
        "interface_guid": "11111111-2222-3333-4444-555555555555",
        "interface_index": 7,
        "local_address": "10.0.0.1",
        "pnp_device_id": r"ROOT\SIMA_VETH\0000",
    }
    qualification.write_text(json.dumps({
        "assets": {
            "identity_sha256": asset_manifest_sha256(asset_manifest),
            "manifest": asset_manifest,
        },
        "candidate": {"core_commit": commit, "resources": resources},
        "card_baseline": card,
        "card_network_baseline": network,
        "card_candidate": {
            "bundle_sha256": "4" * 64,
            "core_commit": commit,
            "firmware": "modalix-cvu-fw",
            "genai_worker_sha256": "5" * 64,
            "installed": {"sima-neat": f"0.6.2+feature.{commit[:12]}"},
            "runtime_sha256": "2" * 64,
            "worker_sha256": "3" * 64,
        },
        "card_final": card,
        "card_network_final": network,
        "cleanup": {
            "card_stage": "PASS", "core_msi": "PASS", "errors": [],
            "customer_state": "PASS", "host_state": "PASS", "python": "PASS",
            "remote_lock": "PASS", "scheduler_msi": "PASS", "work": "PASS"},
        "customer_install": {
            "command": f"core/pciehost/windows/amd64@v{version}",
            "download_path_with_spaces": r"C:\customer downloads\candidate",
            "missing_wheel": {"exit_code": 1, "status": "PASS"},
            "native_only": "PASS",
            "plugin_discovery": "PASS",
            "process_activation": "PASS",
            "repair": {"core": "PASS", "scheduler": "PASS"},
            "recovery": "PASS",
            "repeat": "PASS",
            "standard_user": {
                "account": "TEST\\sna-1234",
                "common_application_data": r"C:\ProgramData",
                "config_binding": "PASS",
                "connection_options": "PASS",
                "default_media_directory": r"C:\ProgramData\SiMa.ai\data",
                "default_media_serve_root": "data",
                "default_root": "PASS",
                "doctor": "PASS",
                "is_administrator": False,
                "override_absence": "PASS",
                "platform_config": r"C:\ProgramData\SiMa.ai\simaai-mla-daemon.conf",
                "platform_config_media_directory": r"C:\ProgramData\SiMa.ai\data",
                "process_is_administrator": False,
                "python": "PASS",
                "real_genai": "PASS",
                "real_genai_case": "test_genai_asr_run",
                "real_genai_executable_sha256": sha256(standard_user_genai),
                "required_assets_readable": "PASS",
                "sentinel": "PASS",
                "sentinel_cleanup": "PASS",
                "standard_user": "PASS",
            },
            "upgrade": {
                "baseline_commit": "b" * 40,
                "baseline_version": "0.6.1",
                "candidate_version": version,
                "status": "PASS",
            },
            "wheel_failure": {"exit_code": 1, "status": "PASS"},
        },
        "genai_media": {
            "directory": r"C:\ProgramData\SiMa\PCIe GenAI media",
            "serve_root": "data",
        },
        "dependencies": qualified_dependencies,
        "platform_runtime": [{"name": "sima-platform.dll", "sha256": "d" * 64}],
        "host_state_baseline": {
            "dependencies": qualified_dependencies,
            "platform": platform_host_state,
        },
        "host_state_final": {
            "dependencies": qualified_dependencies,
            "platform": platform_host_state,
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
            "driver_signing": {
                "hvci_running": True,
                "no_integrity_checks": False,
                "secure_boot": True,
                "test_signing": False,
                "wdac_kernel_policy": "off",
                "wdac_user_mode_policy": "off",
            },
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

  def enable_genai_qualification(self, qualification: Path) -> dict:
    value = json.loads(qualification.read_text(encoding="utf-8"))
    value["tests"]["cpp"].extend(sorted(GENAI_CPP_CASES))
    value["tests"]["python"].extend(sorted(GENAI_PYTHON_CASES))
    value["genai_admission"] = {
        name: "PASS" for name in sorted(GENAI_CPP_CASES | GENAI_PYTHON_CASES)
    }
    asset_manifest = fixture_asset_manifest(include_genai=True)
    qualification.with_name("hardware-asset-manifest.json").write_text(
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    value["assets"] = {
        "identity_sha256": asset_manifest_sha256(asset_manifest),
        "manifest": asset_manifest,
    }
    qualification.write_text(json.dumps(value), encoding="utf-8")
    return value

  def rewrite_json(self, path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_text(json.dumps(value), encoding="utf-8")

  def rewrite_candidate_json(self, root: Path, name: str, mutate) -> None:
    path = root / name
    self.rewrite_json(path, mutate)
    self.rewrite_json(root / "host-candidate.json", lambda host: host[
        "qualification_resources"].update({name: sha256(path)}))

  def assert_release_rejected(self, message, mutate=None, *, candidate_file=None,
                              expected_commit=None, genai=False, require_signatures=True,
                              **gates):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      if genai:
        self.enable_genai_qualification(qualification)
        gates["require_genai"] = True
      if candidate_file:
        self.rewrite_candidate_json(root, candidate_file, mutate)
      elif mutate:
        self.rewrite_json(qualification, mutate)
      with self.assertRaisesRegex(ValueError, message):
        verify(root, qualification, expected_commit or commit, require_signatures, **gates)

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
      self.assertEqual(set(admission["dependency_signatures"]), {"scheduler"})
      candidate.core_msi.write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "checksum differs"):
        validate_candidate(root)

  def test_release_gate_requires_the_release_tag_install_command(self):
    for command in ("core/pciehost/windows/amd64@3.0", "core/pciehost/windows/amd64@v0.6.1",
                    "core/pciehost/windows/amd64@develop"):
      with self.subTest(command=command):
        self.assert_release_rejected(
            "customer CLI install",
            lambda value: value["customer_install"].update(command=command))

  def test_release_gate_requires_exact_external_asset_manifest(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["assets"]["manifest"]["vision"][VISION_ASSETS[0]]["size"] += 1
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "different hardware asset manifest"):
        verify(root, qualification, commit, False)

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      external = qualification.with_name("hardware-asset-manifest.json")
      value = json.loads(external.read_text(encoding="utf-8"))
      value["revision"] = "unexpected-revision"
      external.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "different hardware asset manifest"):
        verify(root, qualification, commit, False)

  def test_genai_release_requires_genai_asset_manifest_scope(self):
    self.assert_release_rejected(
        "does not admit GenAI assets", require_signatures=False, require_genai=True)

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

  def test_release_gate_accepts_a_numbered_platform_adapter(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)

      def number_adapter(value):
        for state in ("host_state_baseline", "host_state_final"):
          value[state]["platform"]["pnp_devices"][1]["friendly_name"] += " #2"

      self.rewrite_json(qualification, number_adapter)
      self.assertTrue(verify(root, qualification, commit, True)["hardware_qualified"])

  def test_release_gate_binds_qualification_to_the_workflow_candidate(self):
    self.assert_release_rejected("is not workflow commit", expected_commit="0" * 40)
    self.assert_release_rejected(
        "different Core commit", lambda value: value["candidate"].update(core_commit="0" * 40))
    self.assert_release_rejected(
        "different package bytes", lambda value: value["candidate"]["resources"].popitem())

  def test_release_gate_rejects_each_unproven_qualification_record(self):
    rows = (
        ("hardware qualification did not pass", lambda value: value.update(result="FAIL")),
        ("Windows 11 x64 workstation",
         lambda value: value["host"]["windows_security"].update(build_number="19045")),
        ("did not prove clean removal", lambda value: value["cleanup"].update(errors=["left"])),
        ("every customer-owned component", lambda value: value["cleanup"].update(work="FAIL")),
        ("Platform 3.0 card baseline", lambda value: value["card_final"].update(workers="1")),
        ("PCIe performance gate",
         lambda value: value["performance"]["latency"].update(count=2999)),
        ("PCIe performance gate",
         lambda value: value["performance"]["throughput"].update(frames=2999)),
        ("PCIe performance gate",
         lambda value: value["performance"]["latency"].update(mean_ms=16.1)),
        ("PCIe performance gate",
         lambda value: value["performance"]["throughput"].update(fps=289.9)),
    )
    for index, (message, mutate) in enumerate(rows):
      with self.subTest(row=index, message=message):
        self.assert_release_rejected(message, mutate)
    with self.subTest(record="rollback receipt status"):
      self.assert_release_rejected(
          "MSI failed-transaction rollback gate", lambda receipt: receipt.update(status="FAIL"),
          candidate_file="msi-transaction-rollback.json")
    with self.subTest(record="Scheduler publisher"):
      self.assert_release_rejected(
          "different trusted publisher",
          lambda receipt: receipt.update(certificate_sha1="E" * 40),
          candidate_file="scheduler-windows-signatures.json")
    with self.subTest(record="GenAI worker"):
      self.assert_release_rejected(
          "exact Linux GenAI worker",
          lambda value: value["card_candidate"].update(genai_worker_sha256=""), genai=True)

  def card_bundle(self, root, commit, *, source_metadata=b'{"version": "fixture"}',
                  version=None):
    """Write a standalone Core offline card bundle as assemble.py lays it out."""
    tree = root / "tree" / "neat-core-offline"
    (tree / "card" / "payload").mkdir(parents=True)
    (tree / "card" / "payload" / "metadata.json").write_bytes(source_metadata)
    (tree / "card" / "target.json").write_text(json.dumps({"image": "sha256:" + "e" * 64}))
    (tree / "deploy.py").write_text("# card deployment\n")
    files = {path.relative_to(tree).as_posix(): sha256(path)
             for path in sorted(tree.rglob("*")) if path.is_file()}
    (tree / "bundle.json").write_text(json.dumps({
        "schema_version": 1, "core_revision": commit[:12], "files": files, "scope": "core"}))
    bundle = root / "bundle"
    bundle.mkdir()
    archive = bundle / CARD_BUNDLE_ARCHIVE
    with tarfile.open(archive, "w:gz") as output:
      output.add(tree, arcname="neat-core-offline")
    (bundle / "metadata.json").write_text(json.dumps({
        "name": "gh:sima-neat/core/offline",
        "artifact": {"type": "offline", "repository": "core", "package_path": "offline",
                     "distro": "elxr26", "arches": ["arm64"]},
        "version": version or "0.6.2+feature." + commit[:12],
        "installer_revision": commit,
        "resources": [CARD_BUNDLE_ARCHIVE],
        "resources-checksum": {CARD_BUNDLE_ARCHIVE: sha256(archive)},
        "source_metadata_sha256": {"card": hashlib.sha256(source_metadata).hexdigest()},
    }))
    return bundle

  def test_card_bundle_binds_the_exact_core_commit(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
      commit = "c" * 40
      bundle = self.card_bundle(root, commit)
      validated = validate_card_bundle(bundle, commit, "feature")
      self.assertEqual(validated.sha256, sha256(bundle / CARD_BUNDLE_ARCHIVE))
      with self.assertRaisesRegex(ValueError, "not tied"):
        validate_card_bundle(bundle, "d" * 40, "feature")
      metadata = json.loads((bundle / "metadata.json").read_text())
      for field, value, message in (
          ("installer_revision", "d" * 40, "installer revision"),
          ("name", "gh:sima-neat/core", "standalone Core offline card bundle"),
      ):
        with self.subTest(field=field):
          (bundle / "metadata.json").write_text(json.dumps({**metadata, field: value}))
          with self.assertRaisesRegex(ValueError, message):
            validate_card_bundle(bundle, commit, "feature")
      (bundle / "metadata.json").write_text(json.dumps(metadata))
      with (bundle / CARD_BUNDLE_ARCHIVE).open("ab") as archive:
        archive.write(b"changed")
      with self.assertRaisesRegex(ValueError, "checksum differs"):
        validate_card_bundle(bundle, commit, "feature")

  def test_card_bundle_version_follows_the_linux_offline_identity_rule(self):
    # Branch builds suffix the version with the commit; release tags carry the
    # bare tag version (tools/compute_version.sh, offline/ci.py metadata_identity).
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
      commit = "c" * 40
      for ref, version, accepted in (
          ("feature", "0.6.2+feature." + commit[:12], True),
          ("v0.6.2", "0.6.2+v0.6.2." + commit[:12], True),
          ("v0.6.2", "0.6.2", True),
          ("v0.7.0-rc.1", "0.7.0-rc.1", True),
          ("v0.6.2", "0.6.1", False),
          ("feature", "0.6.2", False),
          ("0.6.2", "0.6.2", False),
          ("v0.6.2", commit[:12] + "+0.6.2", False),
      ):
        with self.subTest(ref=ref, version=version):
          case = root / f"{ref}-{version}"
          case.mkdir()
          bundle = self.card_bundle(case, commit, version=version)
          if accepted:
            validate_card_bundle(bundle, commit, ref)
          else:
            with self.assertRaisesRegex(ValueError, "not tied"):
              validate_card_bundle(bundle, commit, ref)
      # The installer revision still binds a release bundle to the selected commit.
      bundle = self.card_bundle(root / "revision", commit, version="0.6.2")
      with self.assertRaisesRegex(ValueError, "installer revision"):
        validate_card_bundle(bundle, "d" * 40, "v0.6.2")

  def test_card_candidate_binds_the_bundle_to_the_linux_candidate(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
      commit = "c" * 40
      source = root / "neat-core-dist"
      source.mkdir()
      (source / "metadata.json").write_bytes(b'{"version": "fixture"}')
      bundle = self.card_bundle(root, commit)
      CARD_PREP.prepare(source, bundle, root / "candidate", commit, "feature")
      receipt = json.loads((root / "candidate" / "windows-card-offline-bundle.json").read_text())
      self.assertEqual(receipt["bundle_sha256"], sha256(bundle / CARD_BUNDLE_ARCHIVE))
      self.assertEqual(receipt["baseline_image"], "sha256:" + "e" * 64)
      self.assertEqual(validate_card_bundle(root / "candidate", commit, "feature").sha256,
                       receipt["bundle_sha256"])

      # A bundle assembled from different Core metadata is refused.
      (source / "metadata.json").write_bytes(b'{"version": "other"}')
      with self.assertRaisesRegex(ValueError, "not assembled from this Linux candidate"):
        CARD_PREP.prepare(source, bundle, root / "other", commit, "feature")
      shutil.rmtree(root / "tree")
      shutil.rmtree(bundle)
      bundle = self.card_bundle(root, commit, source_metadata=b'{"version": "payload"}')
      metadata = json.loads((bundle / "metadata.json").read_text())
      metadata["source_metadata_sha256"] = {"card": sha256(source / "metadata.json")}
      (bundle / "metadata.json").write_text(json.dumps(metadata))
      with self.assertRaisesRegex(ValueError, "payload metadata differs"):
        CARD_PREP.prepare(source, bundle, root / "payload", commit, "feature")

  def test_card_install_runs_the_linux_offline_lifecycle(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory).resolve()
      commit = "c" * 40
      bundle = validate_card_bundle(self.card_bundle(root, commit), commit, "feature")
      args = SimpleNamespace(remote_lock_path="/tmp/neat-pcie-card0.lock",
                             expected_card_commit=commit, ref="feature")
      outputs = {
          "card-offline-space.log": "999999999999\n",
          "card-offline-stage.log": "/var/tmp/neat-offline-Ab3dE6gH\n",
          "card-runtime-packages-candidate.log": "".join(
              f"{name} 0.6.2+feature.{commit[:12]}\n" for name in HARDWARE.CARD_RUNTIME_PACKAGES),
      }
      calls = []

      def logged(command, log, **_kwargs):
        calls.append(("user", command[-1]))
        log.write_text(outputs.get(log.name, ""), encoding="utf-8")

      def privileged(_args, command, log, **_kwargs):
        calls.append(("root", command))

      state = {}
      with mock.patch.object(HARDWARE, "ssh_command", side_effect=lambda _a, cmd: ["ssh", cmd]), \
          mock.patch.object(HARDWARE, "scp_command",
                            side_effect=lambda _a, files, dest: ["scp", *map(str, files), dest]), \
          mock.patch.object(HARDWARE, "run_logged", side_effect=logged), \
          mock.patch.object(HARDWARE, "run_sudo_remote", side_effect=privileged):
        installed = install_card_candidate(args, bundle, root, state)
      stage = "/var/tmp/neat-offline-Ab3dE6gH"
      self.assertEqual(state, {"stage": stage, "install_started": True})
      self.assertEqual(installed["sima-neat"], f"0.6.2+feature.{commit[:12]}")
      self.assertEqual([command for owner, command in calls if owner == "root"], [
          HARDWARE.card_deploy_command(stage, phase)
          for phase in ("preflight", "install", "verify")])
      user = [command for owner, command in calls if owner == "user"]
      self.assertIn("mktemp -d /var/tmp/neat-offline-XXXXXXXX", user[1])
      self.assertIn(f"{stage} > /tmp/neat-pcie-card0.lock/offline_stage", user[2])
      self.assertEqual(user[3], stage + "/")
      self.assertIn(f"= {bundle.sha256};", user[4])
      self.assertIn("gst-inspect-1.0 neatpciesrc", user[5])
      self.assertTrue(user[6].startswith("dpkg-query -W"))
      self.assertEqual(HARDWARE.card_deploy_command(stage, "install"),
                       "timeout --kill-after=30 1500 flock -n /run/neat-offline-install.lock "
                       f"unshare --net python3 -B {stage}/neat-core-offline/deploy.py card install")

      # A release tag's card reports the bare tag version; a branch run's never may.
      outputs["card-runtime-packages-candidate.log"] = "".join(
          f"{name} 0.6.2\n" for name in HARDWARE.CARD_RUNTIME_PACKAGES)
      for ref, accepted in (("v0.6.2", True), ("v0.6.3", False), ("feature", False)):
        with self.subTest(ref=ref), \
            mock.patch.object(HARDWARE, "ssh_command", side_effect=lambda _a, cmd: ["ssh", cmd]), \
            mock.patch.object(HARDWARE, "scp_command", return_value=["scp"]), \
            mock.patch.object(HARDWARE, "run_logged", side_effect=logged), \
            mock.patch.object(HARDWARE, "run_sudo_remote", side_effect=privileged):
          release_args = SimpleNamespace(**{**vars(args), "ref": ref})
          if accepted:
            self.assertEqual(install_card_candidate(release_args, bundle, root, {})["sima-neat"],
                             "0.6.2")
          else:
            with self.assertRaisesRegex(RuntimeError, "exact Core candidate"):
              install_card_candidate(release_args, bundle, root, {})
      outputs["card-runtime-packages-candidate.log"] = "".join(
          f"{name} 0.6.2+feature.{commit[:12]}\n" for name in HARDWARE.CARD_RUNTIME_PACKAGES)

      # A failed preflight never starts the install phase.
      def failing(_args, command, log, **_kwargs):
        if command.endswith(" preflight"):
          raise RuntimeError("preflight failed")
      state = {}
      with mock.patch.object(HARDWARE, "ssh_command", side_effect=lambda _a, cmd: ["ssh", cmd]), \
          mock.patch.object(HARDWARE, "scp_command", return_value=["scp"]), \
          mock.patch.object(HARDWARE, "run_logged", side_effect=logged), \
          mock.patch.object(HARDWARE, "run_sudo_remote", side_effect=failing), \
          self.assertRaisesRegex(RuntimeError, "preflight failed"):
        install_card_candidate(args, bundle, root, state)
      self.assertEqual(state, {"stage": stage})

  def test_card_stage_removal_keeps_an_unknown_install_outcome(self):
    args = SimpleNamespace(remote_lock_path="/tmp/neat-pcie-card0.lock")
    stage = "/var/tmp/neat-offline-Ab3dE6gH"
    with tempfile.TemporaryDirectory() as directory:
      evidence = Path(directory)
      with mock.patch.object(HARDWARE, "run_sudo_remote") as privileged:
        remove_card_stage(args, stage, evidence, install_attempted=False)
      self.assertEqual([call.args[1] for call in privileged.call_args_list], [
          f"rm -rf -- {stage}; rm -f -- /tmp/neat-pcie-card0.lock/offline_stage"])

      def locked(_args, command, _log, **_kwargs):
        if command.startswith("flock -n /run/neat-offline-install.lock"):
          raise RuntimeError("held")
      with mock.patch.object(HARDWARE, "run_sudo_remote", side_effect=locked) as privileged, \
          self.assertRaisesRegex(RuntimeError, "outcome is unknown"):
        remove_card_stage(args, stage, evidence, install_attempted=True)
      self.assertEqual(len(privileged.call_args_list), 1)
      with self.assertRaisesRegex(ValueError, "Invalid remote staging path"):
        remove_card_stage(args, "/var/tmp/other", evidence, install_attempted=False)

  def test_card_lifecycle_constants_match_the_linux_offline_installer(self):
    installer = (ROOT / "pcie_host/scripts/offline/install.py").read_text(encoding="utf-8")
    self.assertIn(", ".join(repr(value) for value in CARD_DEPLOY), installer)
    self.assertIn(f"r'{CARD_STAGE_PATTERN}'", installer)
    self.assertIn("mktemp -d /var/tmp/neat-offline-XXXXXXXX", installer)
    self.assertIn("'timeout', '--kill-after=30', '1500'", installer)

  def test_release_gate_accepts_the_installed_candidate_firmware(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      # The card keeps the candidate, so its firmware name may change.
      self.rewrite_json(qualification, lambda value: value["card_final"].update(
          firmware="candidate-fw"))
      self.assertTrue(verify(root, qualification, commit, True)["hardware_qualified"])
    self.assert_release_rejected(
        "Platform 3.0 card baseline",
        lambda value: value["card_final"].update(platform_sha256="9" * 64))
    self.assert_release_rejected(
        "Platform 3.0 card baseline", lambda value: value["card_final"].update(workers="1"))

  def test_release_rejects_incomplete_card_candidate_identity(self):
    for mutate in (
        lambda value: value["card_candidate"].update(bundle_sha256=""),
        lambda value: value["card_candidate"]["installed"].update({"sima-neat": "0.6.2"}),
        lambda value: value["card_candidate"].pop("installed"),
    ):
      self.assert_release_rejected(
          "exact Linux card candidate", mutate, require_signatures=False)

  def test_release_gate_accepts_the_release_tag_card_version(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      self.rewrite_json(qualification, lambda value: value["card_candidate"]["installed"].update(
          {"sima-neat": "0.6.2"}))
      self.assertTrue(verify(root, qualification, commit, False, ref="v0.6.2")[
          "hardware_qualified"])
      for ref in ("v0.6.3", "feature", "0.6.2"):
        with self.subTest(ref=ref), \
            self.assertRaisesRegex(ValueError, "exact Linux card candidate"):
          verify(root, qualification, commit, False, ref=ref)

  def test_card_sudo_secret_stays_off_command_line_and_log(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      args = card_args(root, card_sudo_password_env="CARD_TEST_SECRET", work=root / "work")
      completed = subprocess.CompletedProcess([], 0, "")
      log = root / "sudo.log"

      def complete(command, output, **kwargs):
        output.write_text("fixture output\n", encoding="utf-8")
        return completed
      with mock.patch.dict(os.environ, {"CARD_TEST_SECRET": "fixture-secret"}), \
          mock.patch(
              "qualify_windows_hardware.run_windows_openssh",
              side_effect=complete) as run:
        run_sudo_remote(args, "true", log)
      command = run.call_args.args[0]
      self.assertNotIn("fixture-secret", " ".join(command))
      self.assertEqual(run.call_args.kwargs["input_value"], "fixture-secret")
      self.assertEqual(
          run.call_args.kwargs["secret_root"], root / "work" / ".openssh-secrets")
      self.assertTrue(run.call_args.kwargs["redact_stdin"])

  def test_windows_openssh_secret_uses_a_temporary_input_file(self):
    with tempfile.TemporaryDirectory() as directory:
      log = Path(directory) / "sudo.log"
      secret_root = Path(directory) / "private"
      observed = {}

      class Process:
        pid = 41
        returncode = 0

        def __enter__(inner_self):
          return inner_self

        def __exit__(inner_self, exc_type, exc_value, traceback):
          del inner_self, exc_type, exc_value, traceback
          return False

        def communicate(inner_self, input=None, timeout=None):
          del input, timeout
          wrappers = list(log.parent.glob("*.openssh.cmd"))
          inputs = list(secret_root.glob("*.openssh-input"))
          self.assertEqual(len(wrappers), 1)
          self.assertEqual(len(inputs), 1)
          observed["wrapper"] = wrappers[0].read_text(encoding="utf-8")
          observed["secret"] = inputs[0].read_bytes()
          return None, None

        def poll(inner_self):
          return inner_self.returncode

      prepare_openssh_secret_directory(secret_root)
      with mock.patch(
          "qualify_windows_hardware.prepare_openssh_secret_directory",
          return_value=secret_root), mock.patch(
              "qualify_windows_hardware.subprocess.Popen", return_value=Process()):
          result = run_windows_openssh(
              ["ssh.exe", "card", "true"], log,
              input_value="fixture-secret", secret_root=secret_root,
              redact_stdin=True)
      self.assertEqual(result.returncode, 0)
      self.assertEqual(observed["secret"], b"fixture-secret\n")
      self.assertNotIn("fixture-secret", observed["wrapper"])
      self.assertFalse(any(secret_root.glob("*.openssh-input")))
      if sys.platform != "win32":
        self.assertEqual(secret_root.stat().st_mode & 0o777, 0o700)
      self.assertNotIn("fixture-secret", log.read_text(encoding="utf-8"))

  def test_card_sudo_stdin_secret_is_reused_from_process_memory(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      args = card_args(
          root, card_sudo_password_stdin=True, card_sudo_password_value="fixture-secret",
          work=root / "work")
      completed = subprocess.CompletedProcess([], 0, "")
      with mock.patch(
          "qualify_windows_hardware.run_windows_openssh",
          return_value=completed) as run:
        run_sudo_remote(args, "true", root / "first.log")
        run_sudo_remote(args, "true", root / "second.log")
      self.assertEqual(run.call_count, 2)
      self.assertEqual(
          [call.kwargs["input_value"] for call in run.call_args_list],
          ["fixture-secret", "fixture-secret"])
      self.assertTrue(all(call.kwargs["redact_stdin"] for call in run.call_args_list))

  def test_windows_openssh_removes_secret_if_process_start_fails(self):
    with tempfile.TemporaryDirectory() as directory:
      log = Path(directory) / "sudo.log"
      with mock.patch(
          "qualify_windows_hardware.subprocess.Popen",
          side_effect=OSError("fixture process failure")):
        with self.assertRaisesRegex(OSError, "fixture process failure"):
          run_windows_openssh(
              ["ssh.exe", "card", "true"], log,
              input_value="fixture-secret", secret_root=Path(directory) / "private",
              redact_stdin=True)
      self.assertFalse(any((Path(directory) / "private").glob("*.openssh-input")))
      self.assertFalse(any(log.parent.glob("*.openssh.cmd")))
      self.assertNotIn("fixture-secret", log.read_text(encoding="utf-8"))

  def windows_hop_args(self, root):
    return card_args(
        root, card_sudo_password_stdin=True, card_sudo_password_value="fixture-secret",
        remote_lock_path="/tmp/neat-pcie-card0.lock", work=root / "work")

  @unittest.skipIf(os.name == "nt", WINDOWS_OPENSSH_HOP_SKIP)
  def test_card_commands_keep_their_quotes_across_the_windows_hop(self):
    remote = (
        "set -eu; trap 'printf \"%s\\n\" \"exit  trap\"' EXIT; "
        "awk '{print $1}' \"card input.txt\"; printf '%s\\n' \"two  spaces\"")
    expected = "alpha\ngamma\ntwo  spaces\nexit  trap\n"
    for sudo_n_status in ("0", "1"):
      with self.subTest(sudo_n_status=sudo_n_status), \
          tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        args = self.windows_hop_args(root)
        hop = WindowsOpenSshHop(root)
        (hop.card / "card input.txt").write_text(
            "alpha beta\ngamma  delta\n", encoding="utf-8")
        hop.environment = {
            "FIXTURE_SUDO_N_STATUS": sudo_n_status,
            "FIXTURE_SUDO_SECRET": "fixture-secret"}
        privileged_log = root / "privileged.log"
        with mock.patch("qualify_windows_hardware.subprocess.Popen", hop):
          try:
            run_sudo_remote(args, remote, privileged_log)
          except RuntimeError as error:
            self.fail(f"{error}: {privileged_log.read_text(encoding='utf-8')}")
          plain = run_captured_to_file(
              HARDWARE.ssh_command(args, remote), root / "plain.log", timeout=30)
        self.assertEqual(
            privileged_log.read_text(encoding="utf-8").split("\n", 1)[1], expected)
        self.assertEqual((plain.returncode, plain.stdout), (0, expected))

  @unittest.skipIf(os.name == "nt", WINDOWS_OPENSSH_HOP_SKIP)
  def test_card_deploy_reaches_root_unchanged_across_the_windows_hop(self):
    command = HARDWARE.card_deploy_command("/var/tmp/neat-offline-Ab3dE6gH", "install")
    for sudo_n_status in ("0", "1"):
      with self.subTest(sudo_n_status=sudo_n_status), \
          tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        args = self.windows_hop_args(root)
        hop = WindowsOpenSshHop(root)
        hop.environment = {
            "FIXTURE_SUDO_N_STATUS": sudo_n_status,
            "FIXTURE_SUDO_SECRET": "fixture-secret",
            "FIXTURE_ROOT_SCRIPT": str(root / "root-script")}
        with mock.patch("qualify_windows_hardware.subprocess.Popen", hop):
          try:
            run_sudo_remote(args, command, root / "card-offline-install.log")
          except RuntimeError as error:
            log = root / "card-offline-install.log"
            self.fail(f"{error}: {log.read_text(encoding='utf-8')}")
        self.assertEqual((root / "root-script").read_text(encoding="utf-8"), command)
        self.assertNotIn("fixture-secret", (root / "card-offline-install.log").read_text(
            encoding="utf-8"))

  def test_candidate_rejects_changed_qualification_resource(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.windows_candidate(root)
      (root / "qualification-resource.bin").write_bytes(b"changed")
      with self.assertRaisesRegex(ValueError, "qualification resource checksum differs"):
        validate_candidate(root)

  def test_candidate_rejects_uninventoried_cpack_scratch(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.windows_candidate(root)
      scratch = root / "_CPack_Packages" / "win64" / "WIX"
      scratch.mkdir(parents=True)
      (scratch / "files.wxs").write_text("unpublished scratch", encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "uninventoried files or directories"):
        validate_candidate(root)

  def test_candidate_accepts_inventoried_hardware_test_tree(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      self.windows_candidate(root)
      hardware_test = root / "hardware-tests" / "bin" / "test.exe"
      hardware_test.parent.mkdir(parents=True)
      hardware_test.write_bytes(b"connected-card test fixture")
      receipt_path = root / "host-candidate.json"
      receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
      receipt["qualification_resources"]["hardware-tests/bin/test.exe"] = sha256(
          hardware_test)
      receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
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
      # As on Linux, the Platform is identified by version, not by its files.
      self.assertEqual(identity["platform_version"], "3.0.0.2")
      self.assertNotIn("platform_runtime_dlls", identity)
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

  def test_release_gate_requires_failed_msi_transaction_rollback(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      admission = verify(root, qualification, commit, False)
      self.assertTrue(admission["msi_transaction_rollback_qualified"])

    mutations = (
        ("failure", "observed_exit_code", 0, "fail through Windows Installer"),
        ("mid_transaction", "file", "FAIL", "transaction-owned state"),
        ("rollback", "product_registration", "FAIL", "registration behind"),
        ("preserved_cohort", "file_inventory", "FAIL", "preserve the installed cohort"),
    )
    for section, key, replacement, message in mutations:
      with self.subTest(section=section, key=key):
        self.assert_release_rejected(
            message, lambda receipt: receipt[section].update({key: replacement}),
            candidate_file="msi-transaction-rollback.json", require_signatures=False)

  def test_release_gate_binds_failed_msi_transaction_log(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      log = root / "msi-transaction-rollback.log"
      log.write_text("changed exact verbose log", encoding="utf-8")
      host_path = root / "host-candidate.json"
      host = json.loads(host_path.read_text(encoding="utf-8"))
      host["qualification_resources"][log.name] = sha256(log)
      host_path.write_text(json.dumps(host), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "bind its verbose installer log"):
        verify(root, qualification, commit, False)

  def test_release_gate_requires_partial_wheel_failure_recovery(self):
    self.assert_release_rejected(
        "native success, wheel failure, and retry",
        lambda value: value["customer_install"].pop("wheel_failure"),
        require_release_qualification=True)

  def test_validation_admission_leaves_windows_only_release_checks_to_releases(self):
    release_only = ("missing_wheel", "native_only", "recovery", "repair", "repeat",
                    "wheel_failure")
    fault_cases = ("test_worker_loss_recovery", "test_client_process_death_recovery")

    def validation_run(value):
      for name in release_only:
        del value["customer_install"][name]
      value["tests"]["cpp"] = [
          name for name in value["tests"]["cpp"] if name not in fault_cases]

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      self.rewrite_json(qualification, validation_run)
      admission = verify(root, qualification, commit, False)
      self.assertFalse(admission["release_qualification_required"])
      with self.assertRaisesRegex(ValueError, "repeated customer CLI flow"):
        verify(root, qualification, commit, False, require_release_qualification=True)
    for name in ("repair", "missing_wheel"):
      with self.subTest(evidence=name):
        self.assert_release_rejected(
            "release qualification did not prove",
            lambda value: value["customer_install"].pop(name),
            require_signatures=False, require_release_qualification=True)
    self.assert_release_rejected(
        "test_client_process_death_recovery",
        lambda value: value["tests"]["cpp"].remove("test_client_process_death_recovery"),
        require_signatures=False, require_release_qualification=True)

  def test_release_gate_requires_customer_process_activation(self):
    self.assert_release_rejected(
        "standard-user process activation",
        lambda value: value["customer_install"].pop("process_activation"))

  def test_release_gate_requires_non_administrator_process_activation(self):
    for field in ("is_administrator", "process_is_administrator"):
      with self.subTest(field=field):
        self.assert_release_rejected(
            "standard-user process activation",
            lambda value: value["customer_install"]["standard_user"].update({field: True}))

  def test_genai_release_rejects_custom_or_admin_only_standard_user_false_passes(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      original = self.enable_genai_qualification(qualification)
      for field in (
          "config_binding", "connection_options", "default_root", "override_absence",
          "real_genai", "required_assets_readable", "sentinel", "sentinel_cleanup",
      ):
        value = json.loads(json.dumps(original))
        value["customer_install"]["standard_user"][field] = "SKIPPED"
        qualification.write_text(json.dumps(value), encoding="utf-8")
        with self.subTest(field=field), self.assertRaisesRegex(
            ValueError, "default-root real GenAI as a standard user"):
          verify(root, qualification, commit, False, require_genai=True)

      value = json.loads(json.dumps(original))
      value["customer_install"]["standard_user"]["is_administrator"] = True
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "standard-user process activation"):
        verify(root, qualification, commit, False, require_genai=True)

      value = json.loads(json.dumps(original))
      del value["customer_install"]["standard_user"]["default_media_directory"]
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "CommonApplicationData"):
        verify(root, qualification, commit, False, require_genai=True)

  def test_genai_release_admits_default_root_without_custom_overrides(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = self.enable_genai_qualification(qualification)
      del value["genai_media"]
      qualification.write_text(json.dumps(value), encoding="utf-8")
      result = verify(root, qualification, commit, False, require_genai=True)
      self.assertTrue(result["genai_qualified"])

  def test_genai_release_binds_standard_user_run_to_candidate_executable(self):
    self.assert_release_rejected(
        "different executable", lambda value: value["customer_install"]["standard_user"].update(
            real_genai_executable_sha256="0" * 64), genai=True, require_signatures=False)

  def test_upgrade_baseline_is_the_prior_release_as_published(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      candidate_root = root / "candidate"
      baseline_root = root / "baseline"
      candidate_root.mkdir()
      baseline_root.mkdir()
      self.windows_candidate(candidate_root, commit="c" * 40, version="0.6.2")
      baseline_commit, _, _ = self.windows_candidate(
          baseline_root, commit="b" * 40, version="0.6.1")
      candidate = validate_candidate(candidate_root)
      # The registry holds only what the publisher uploads: no receipts.
      registry = root / "published"
      stage_local_registry(validate_candidate(baseline_root), registry)
      with HARDWARE.local_registry(registry) as url:
        published = RELEASE.fetch_release(
            RELEASE.https_reader(url), "v0.6.1", baseline_commit, root / "fetched")
        with self.assertRaisesRegex(ValueError, "not release commit aaaaaaaaaaaa"):
          RELEASE.fetch_release(
              RELEASE.https_reader(url), "v0.6.1", "a" * 40, root / "wrong-commit")
        with self.assertRaisesRegex(OSError, "404"):
          RELEASE.fetch_release(
              RELEASE.https_reader(url), "v0.6.0", baseline_commit, root / "missing")
        package = registry / RELEASE.package_prefix("v0.6.1", baseline_commit)
        (package / "core.msi").write_bytes(b"changed after publication")
        with self.assertRaisesRegex(ValueError, "checksum differs: core.msi"):
          RELEASE.fetch_release(
              RELEASE.https_reader(url), "v0.6.1", baseline_commit, root / "tampered")
      self.assertFalse((published.root / "host-candidate.json").exists())
      baseline = load_upgrade_base(candidate, published.root, baseline_commit)
      self.assertEqual(baseline.core_commit, baseline_commit)
      self.assertEqual(HARDWARE.customer_install_target(baseline),
                       "core/pciehost/windows/amd64@v0.6.1")
      with self.assertRaisesRegex(ValueError, "does not match"):
        load_upgrade_base(candidate, published.root, "a" * 40)
      (published.root / "notes.txt").write_text("unlisted")
      with self.assertRaisesRegex(ValueError, "uninventoried"):
        load_upgrade_base(candidate, published.root, baseline_commit)
      (published.root / "notes.txt").unlink()
      for version in (candidate.metadata["version"], "0.6.3"):
        baseline.metadata["version"] = version
        (published.root / "metadata.json").write_text(
            json.dumps(baseline.metadata), encoding="utf-8")
        with self.subTest(version=version), self.assertRaisesRegex(
            ValueError, "newer than the baseline"):
          load_upgrade_base(candidate, published.root, baseline_commit)

  def test_release_gate_rejects_a_different_card_pc_platform_runtime(self):
    self.assert_release_rejected(
        "different Platform runtime",
        lambda value: value["platform_runtime"][0].update(sha256="0" * 64))
    self.assert_release_rejected(
        "different Platform runtime", lambda value: value.pop("platform_runtime"))

  def test_release_gate_rejects_changed_hardware_dependency_cohort(self):
    self.assert_release_rejected(
        "different runtime dependency cohort",
        lambda value: value["dependencies"]["gstreamer"]["runtime_dlls"][0].update(sha256="0" * 64))

  def test_release_gate_rejects_changed_windows_host_baseline(self):
    self.assert_release_rejected(
        "Windows host baseline", lambda value: value["host_state_final"]["platform"]["service"][
            "configuration"].update(Start=3))

  def test_release_gate_rejects_unsigned_scheduler_payload(self):
    self.assert_release_rejected(
        "Scheduler candidate was not built",
        lambda scheduler: scheduler.update(release_required=False),
        candidate_file="scheduler-windows-signatures.json")

  def test_release_gate_rejects_missing_genai_matrix(self):
    def drop_genai(value):
      for language in ("cpp", "python"):
        value["tests"][language] = [
            name for name in value["tests"][language] if not name.startswith("test_genai_")]
      value.pop("genai_admission")

    self.assert_release_rejected("complete Windows GenAI matrix", drop_genai, genai=True)

  def test_release_gate_requires_genai_failure_recovery_and_cleanup_matrix(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = self.enable_genai_qualification(qualification)
      admission = verify(root, qualification, commit, True, True)
      self.assertTrue(admission["genai_qualified"])

      changed = json.loads(qualification.read_text(encoding="utf-8"))
      changed.pop("genai_media")
      qualification.write_text(json.dumps(changed), encoding="utf-8")
      self.assertTrue(verify(root, qualification, commit, True, True)["genai_qualified"])
      changed["genai_media"] = value["genai_media"]
      qualification.write_text(json.dumps(changed), encoding="utf-8")

      for missing in sorted(GENAI_PYTHON_CASES):
        with self.subTest(missing=missing):
          changed = json.loads(qualification.read_text(encoding="utf-8"))
          changed["tests"]["python"].remove(missing)
          qualification.write_text(json.dumps(changed), encoding="utf-8")
          with self.assertRaisesRegex(ValueError, "complete Windows GenAI matrix"):
            verify(root, qualification, commit, True, True)
          changed["tests"]["python"].append(missing)
          qualification.write_text(json.dumps(changed), encoding="utf-8")

      changed = json.loads(qualification.read_text(encoding="utf-8"))
      changed["genai_admission"]["test_genai_service_loss_recovery"] = "FAIL"
      qualification.write_text(json.dumps(changed), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "incomplete Windows GenAI outcomes"):
        verify(root, qualification, commit, True, True)

  def test_release_gate_enforces_the_runner_performance_floor(self):
    def miss_thresholds(value):
      value["performance"]["latency"]["mean_ms"] = 16.1
      value["performance"]["throughput"]["fps"] = 289.9

    self.assert_release_rejected(
        "PCIe performance gate", miss_thresholds, require_signatures=False)

  def test_release_gate_rejects_incomplete_vision_matrix(self):
    self.assert_release_rejected(
        "missing required vision cases",
        lambda value: value["tests"]["cpp"].remove("test_tensor_parallel_queues"))

  def test_release_gate_rejects_incomplete_installed_tutorial_matrix(self):
    self.assert_release_rejected(
        "installed C[+][+] and Python tutorials",
        lambda value: value["tests"]["tutorial"]["python"].remove("run_image_boxdecode"))

  def test_release_gate_rejects_server_or_unsupported_python(self):
    self.assert_release_rejected(
        "Windows 11 x64 workstation",
        lambda value: value["host"]["windows_security"].update(product_type=3))
    self.assert_release_rejected(
        "supported CPython 3.14 ABI", lambda value: value["host"].update(python="3.13.9"))

  def test_release_gate_requires_unchanged_platform_3_card(self):
    self.assert_release_rejected(
        "Platform 3.0 card baseline",
        lambda value: value["card_final"].update(platform_sha256="0" * 64))
    for release in ("2.1.3_B4837", "2.3.0_B1", "1.3.0"):
      with self.subTest(release=release):
        self.assert_release_rejected("Platform 3.0 card baseline", lambda value: [
            value[card].update(platform_release=release)
            for card in ("card_baseline", "card_final")])

  def test_release_gate_requires_a_stable_card_network_adapter(self):
    self.assert_release_rejected(
        "stable Windows adapter", lambda value: value["card_network_final"].update(
            interface_guid="aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"))

  def test_release_gate_requires_structured_performance_evidence(self):
    self.assert_release_rejected(
        "valid performance measurements",
        lambda value: value["performance"]["throughput"].update(fps=0))

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
    # A fixed concurrency group would cancel the older of two queued runs; the
    # single runner and this lock already serialise them.
    hardware = WORKFLOW.read_text(encoding="utf-8").split("  hardware:\n", 1)[1].split(
        "  publish:", 1)[0]
    self.assertNotIn("concurrency:", hardware)

  def test_card_side_lock_is_owned_and_released_by_token(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      args = card_args(root)
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

      # After a failed acquire, retain the local lock only if the card may hold our token.
      for conflicted, probe, held in (
          (False, subprocess.CompletedProcess([], 0, "owner=\n"), False),
          (False, subprocess.CompletedProcess([], 0, f"owner={'b' * 32}\n"), False),
          (False, subprocess.CompletedProcess([], 0, f"owner={token}\n"), True),
          (False, subprocess.CompletedProcess([], 255, ""), True),
          (True, subprocess.CompletedProcess([], 0, f"owner={'b' * 32}\n"), False),
      ):
        with self.subTest(conflicted=conflicted, probe=probe.stdout, code=probe.returncode), \
            mock.patch("qualify_windows_hardware.run_captured_to_file", return_value=probe):
          remote_lock.conflicted = conflicted
          self.assertIs(remote_lock.may_hold_after_failed_acquire(root / "probe.log"), held)

  def test_card_ssh_capture_uses_a_file_instead_of_a_pipe(self):
    with tempfile.TemporaryDirectory() as directory:
      log = Path(directory) / "ssh.log"
      captured = "owner=" + "a" * 32 + "\n"
      with mock.patch(
          "qualify_windows_hardware.run_windows_openssh",
          return_value=subprocess.CompletedProcess([], 0, captured)) as run:
        completed = run_captured_to_file(["ssh.exe", "host", "true"], log, timeout=30)
      self.assertEqual(completed.returncode, 0)
      self.assertIn("owner=" + "a" * 32, completed.stdout)
      self.assertNotIn("ssh.exe", completed.stdout)
      run.assert_called_once_with(["ssh.exe", "host", "true"], log, timeout=30)

  @unittest.skipIf(os.name == "nt", WORKFLOW_BASH_SKIP)
  def test_card_platform_check_applies_the_linux_card_lane_rule(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      manifest = root / "manifest.json"
      manifest.write_text(json.dumps({"platform-version": "3.0.1+B2798"}), encoding="utf-8")
      self.assertEqual(HARDWARE.require_card_platform("3.0.1", manifest), "3.0.1")
      with self.assertRaisesRegex(
          RuntimeError, "platform 3.0.0 is incompatible with package platform 3.0.1"):
        HARDWARE.require_card_platform("3.0.0", manifest)
      with self.assertRaisesRegex(RuntimeError, "does not define DISTRO_VERSION"):
        HARDWARE.require_card_platform("", manifest)
      self.assertIn(
          'require_card_platform(\n          baseline["platform_version"], '
          'args.source_root / "deps" / "manifest.json")',
          (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8"))

      # The card-side read is the Linux script's awk program.
      args = card_args(root, include_genai=False)
      commands = []
      with mock.patch("qualify_windows_hardware.run_logged",
                      side_effect=lambda command, *_a, **_k: commands.append(command[-1])
                      or (_ for _ in ()).throw(RuntimeError("stop"))):
        with self.assertRaisesRegex(RuntimeError, "stop"):
          card_snapshot(args, root, "baseline")
      read = re.search(r"printf 'platform_version='; (awk .*?/etc/buildinfo)", commands[0])[1]
      linux = (ROOT / "scripts/ci/pciehost/check_card_compatibility.sh").read_text(
          encoding="utf-8")
      self.assertIn("DISTRO_VERSION[[:space:]]*", linux)
      for text, expected in (("DISTRO_VERSION = 3.0.1\n", "3.0.1"),
                             ("  DISTRO_VERSION=3.0.1 \nOTHER=1\n", "3.0.1"),
                             ("SIMA_BUILD_VERSION=3.0.0_master\n", "")):
        with self.subTest(buildinfo=text):
          buildinfo = root / "buildinfo"
          buildinfo.write_text(text, encoding="utf-8")
          result = subprocess.run(
              [shutil.which("bash"), "-c", read.replace("/etc/buildinfo", str(buildinfo))],
              capture_output=True, text=True, check=True)
          self.assertEqual(result.stdout.strip(), expected)

  def test_card_snapshot_records_platform_3_build_identity(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      args = card_args(root, include_genai=False)

      def write_snapshot(command, log, **_kwargs):
        text = (
            "machine=aarch64\nplatform_release=3.0.0_master_B2501\n"
            f"platform_sha256={'1' * 64}\nplatform_version=3.0.1\nfirmware=modalix-cvu-fw\n"
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
      self.assertEqual(snapshot["platform_version"], "3.0.1")
      with mock.patch(
          "qualify_windows_hardware.run_logged", side_effect=write_snapshot):
        active = card_snapshot(
            args, root, "candidate",
            runtime_path="/tmp/candidate/modalix-cvu-fw",
            worker_path="/tmp/candidate/pcie-pipeline-builder",
            genai_worker_path="/tmp/candidate/neat-pcie-genai-worker")
      self.assertEqual(active["worker_sha256"], "3" * 64)
      self.assertEqual(active["genai_worker_sha256"], "4" * 64)

      for release in ("2.1.3_B4837", "2.3.0_B1", "1.3.0"):

        def write_old_snapshot(command, log, **kwargs):
          write_snapshot(command, log, **kwargs)
          text = log.read_text(encoding="utf-8").replace("3.0.0_master_B2501", release)
          log.write_text(text, encoding="utf-8")

        with self.subTest(release=release), mock.patch(
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
      # sima-cli resolves @v0.6.2 through core/v0.6.2/latest.tag to the
      # 12-character commit directory that the publisher writes.
      self.assertEqual(
          (registry / "core" / "v0.6.2" / "latest.tag").read_text(), "cccccccccccc\n")
      self.assertEqual(
          package, registry / "core" / "v0.6.2" / "cccccccccccc" / "pciehost" / "windows"
          / "amd64")
      self.assertEqual(HARDWARE.customer_install_target(candidate),
                       "core/pciehost/windows/amd64@v0.6.2")
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

  def test_windows_release_layout_is_the_cli_and_ubuntu_layout(self):
    commit = "0123456789ab" + "c" * 28
    self.assertEqual(RELEASE.release_tag("0.7.0"), "v0.7.0")
    self.assertEqual(RELEASE.customer_target("v0.7.0"), "core/pciehost/windows/amd64@v0.7.0")
    self.assertEqual(RELEASE.latest_tag_key("v0.7.0"), "core/v0.7.0/latest.tag")
    self.assertEqual(RELEASE.package_prefix("v0.7.0", commit),
                     "core/v0.7.0/0123456789ab/pciehost/windows/amd64")
    # Ubuntu and sima-cli quote the whole ref, so a branch is one path segment.
    self.assertEqual(RELEASE.package_prefix("feature/x y", commit),
                     "core/feature%2Fx%20y/0123456789ab/pciehost/windows/amd64")
    for invalid in ("0123456789ab", "C" * 40, ""):
      with self.subTest(commit=invalid), self.assertRaisesRegex(ValueError, "full lowercase"):
        RELEASE.package_prefix("v0.7.0", invalid)
    for version in ("0.7", "0.7.0-rc.1", "3.0"):
      with self.subTest(version=version), self.assertRaisesRegex(ValueError, "major.minor.patch"):
        RELEASE.release_tag(version)

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
        "name": "feature/branch (production)",
        "path": ".github/workflows/vulcan-ci.yml",
        "run_attempt": 1,
        "status": "completed",
    }
    self.assertEqual(
        verify_linux_run(run, repository, 42, commit)["head_sha"], commit)
    changed = dict(run, head_sha="f" * 40)
    with self.assertRaisesRegex(ValueError, "head_sha"):
      verify_linux_run(changed, repository, 42, commit)

  def test_genai_fault_client_keeps_streaming_until_the_fault_lands(self):
    # A short answer used to finish before a service stop took effect.
    class Fault(RuntimeError):
      pass

    class Stream:
      def __init__(self, chunks):
        self.chunks = list(chunks)

      def next(self):
        item = self.chunks.pop(0)
        if isinstance(item, Exception):
          raise item
        return item

    token = SimpleNamespace(is_final=False)
    model = mock.Mock()
    genai = SimpleNamespace(GenAIModel=mock.Mock(return_value=model),
                            GenerationRequest=SimpleNamespace,
                            ConnectionOptions=SimpleNamespace)
    stubs = {
        "pyneatpcie": SimpleNamespace(genai=genai),
        "genai_admission_contract": SimpleNamespace(
            is_injected_transport_failure=lambda error: isinstance(error, Fault)),
    }
    path = ROOT / "pcie_host" / "python" / "tests" / "genai_fault_client.py"
    spec = importlib.util.spec_from_file_location("genai_fault_client", path)
    client = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
      spec.loader.exec_module(client)
    for streams in (
        [Stream([token, token, None]), Stream([token, Fault("injected")])],
        # The fault lands after one stream ended; starting the next one reports it.
        [Stream([token, None]), Fault("injected")]):
      with self.subTest(second=type(streams[1]).__name__):
        model.reset_mock()
        model.stream.side_effect = streams
        output = []
        with mock.patch.object(client, "model_path", return_value="model"), \
             mock.patch.object(client, "connection", return_value=None), \
             mock.patch("builtins.print", lambda *args, **_: output.append(args[0])):
          client.wait_for_injected_failure()
        self.assertEqual(output, ["GENAI_FAULT_READY", "GENAI_FAULT_OBSERVED: injected"])
        self.assertEqual(model.stream.call_count, 2)
        model.close.assert_called_once()

  def test_installed_neat_status_runs_neat_cmd_and_checks_the_host_report(self):
    packages = {"simaai-scheduler-pcie-host": "0.6.2", "sima-pcie-host": "0.6.2"}
    report = {"host": {"packages": {name: {"version": version, "architecture": "amd64"}
                                    for name, version in packages.items()}},
              "cards": [], "errors": []}
    install_root = Path(r"C:\Program Files\SimaPCIeHost")
    for case, code, stdout, render_code, expected in (
        ("pass", 0, json.dumps(report), 0, None),
        ("exit", 1, "", 0, "--json exited with 1"),
        # stderr text is kept apart from the JSON on stdout.
        ("not-json", 0, "Usage: neat [status] [--json]", 0, "did not print one JSON object"),
        ("errors", 0, json.dumps({**report, "errors": ["Unable to read PCI inventory"]}), 0,
         "reported errors"),
        ("version", 0, json.dumps({**report, "host": {"packages": {
            "sima-pcie-host": {"version": "0.6.1"}}}}), 0, "reports host packages"),
        ("render", 0, json.dumps(report), 1, "neat.cmd status exited with 1"),
    ):
      with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
        log = Path(directory) / "neat-status.log"
        completed = [subprocess.CompletedProcess([], code, stdout, "neat: stderr line\n"),
                     subprocess.CompletedProcess([], render_code, "Host\n", "")]
        with mock.patch.dict(os.environ, {"SystemRoot": r"C:\Windows"}), \
            mock.patch("windows_candidate.subprocess.run", side_effect=completed) as run:
          if expected:
            with self.assertRaisesRegex(RuntimeError, expected):
              installed_neat_status(install_root, {"PATH": "x"}, log, packages)
          else:
            self.assertEqual(installed_neat_status(install_root, {"PATH": "x"}, log, packages),
                             report)
        neat = str(install_root / "bin" / "neat.cmd")
        self.assertEqual([call.args[0][1:] for call in run.call_args_list],
                         [["/d", "/c", neat, "--json"], ["/d", "/c", neat, "status"]])
        for call in run.call_args_list:
          self.assertTrue(call.args[0][0].endswith("cmd.exe"))
          self.assertEqual(call.kwargs["env"], {"PATH": "x"})
          # Not from bin: there a neat.cmd that resolves its helper from the current
          # directory still passes.
          self.assertEqual(call.kwargs["cwd"], log.parent)
        self.assertIn("neat: stderr line", log.read_text(encoding="utf-8"))

  def test_hosted_build_and_hardware_qualifier_run_installed_neat_cmd(self):
    producer = SCRIPT.read_text(encoding="utf-8")
    installed = producer.split('    check_consumer(install_root, "installed")\n', 1)[1]
    self.assertTrue(installed.startswith(
        "    # neat.cmd as a customer runs it, on a runner without a card.\n"
        "    neat_status = installed_neat_status(\n"))
    self.assertIn('if neat_status["cards"]:\n      raise ValueError', installed)
    self.assertIn('"neat_status": "PASS",', producer)
    self.assertIn('"neat-status.log",', (SCRIPT_DIR / "windows_candidate.py").read_text(
        encoding="utf-8"))
    qualifier = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    self.assertIn(
        '      result["customer_install"]["process_activation"] = "PASS"\n'
        '      result["customer_install"]["neat_status"] = verify_installed_neat_status(\n',
        qualifier)

  def test_hosted_windows_build_runs_every_windows_only_test_through_ctest(self):
    # The contracts job runs these files on Linux, where Windows-only tests skip. The
    # hosted Windows build of every non-docs Vulcan CI push runs them through CTest.
    windows_only = set()
    for path in (ROOT / "pcie_host" / "tests").glob("test_*.py"):
      for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and any(
            ast.unparse(decorator).startswith("unittest.skipUnless(os.name == 'nt'")
            for decorator in node.decorator_list):
          windows_only.add(path.name)
    self.assertGreaterEqual(len(windows_only), 2)
    # Each registration's enclosing CMake conditions, with else() branches negated.
    registrations, conditions = {}, []
    cmake = (ROOT / "pcie_host" / "CMakeLists.txt").read_text(encoding="utf-8")
    for command, arguments in re.findall(
        r"(?ms)^\s*(if|elseif|else|endif|add_test)\s*\((.*?)\)\s*$", cmake):
      if command == "if":
        conditions.append(arguments.strip())
      elif command in ("elseif", "else"):
        conditions[-1] = f"{command}: {conditions[-1]}"
      elif command == "endif":
        conditions.pop()
      for name in re.findall(r"\$\{CMAKE_CURRENT_SOURCE_DIR\}/tests/(test_\w+\.py)", arguments):
        registrations[name] = list(conditions)
    for name in sorted(windows_only):
      with self.subTest(name=name):
        self.assertEqual(registrations.get(name), ["SIMAPCIE_BUILD_TESTS"])
    producer = SCRIPT.read_text(encoding="utf-8")
    self.assertIn('"-S", PCIE_HOST,', producer)
    self.assertIn('"-DSIMAPCIE_BUILD_TESTS=ON",', producer)
    # Every registered test runs, unfiltered, and must pass without a CTest skip.
    self.assertIn('run([ctest, "--test-dir", build_dir, "--output-on-failure", "--no-tests=error",\n'
                  '         "--output-junit", native_junit], env=environment)', producer)
    self.assertIn("require_junit_cases(native_junit, set(test_names))", producer)
    wrapper = (SCRIPT_DIR / "run_windows_build.ps1").read_text(encoding="utf-8")
    self.assertIn('  "scripts/ci/pciehost/build_windows.py",\n', wrapper)
    self.assertIn("& python @arguments\n", wrapper)
    main = MAIN_WORKFLOW.read_text(encoding="utf-8")
    self.assertIn('\non:\n  push:\n    branches:\n      - "**"\n', main.split("\njobs:\n", 1)[0])
    build = workflow_job(main, "build-pciehost-packages")
    # Not the card lane's gates: this job runs for every non-docs change.
    condition = job_condition(main, "build-pciehost-packages")
    for gate in ("windows_pcie", "refs/tags"):
      self.assertNotIn(gate, condition)
    self.assertIn("          - distro: windows\n            deb_arch: amd64\n", build)
    step = workflow_step(build, "Build, test, package, install, and remove the Windows cohort")
    self.assertIn("        if: ${{ matrix.distro == 'windows' }}\n", step)
    self.assertIn("& ./scripts/ci/pciehost/run_windows_build.ps1", step)

  def test_hardware_qualifier_requires_the_card_and_cohort_in_neat_status(self):
    native = [{"owner": "scheduler", "version": "0.6.2"}, {"owner": "core", "version": "0.6.2"}]
    candidate = SimpleNamespace(metadata={"installation": {"native-packages": native}})
    card_packages = {"sima-neat": "0.6.2.abcdef123456", "simaai-runner": "1.0"}
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      key = root / "key"
      key.write_text("k", encoding="ascii")
      args = SimpleNamespace(card_host="10.0.0.2", card_user="sima", ssh_key=key)
      card = {"address": "10.0.0.2", "interfaces": ["Ethernet 3"], "state": "ok",
              "software": {"packages": {name: {"version": version}
                                        for name, version in card_packages.items()}}}
      for case, cards, expected in (
          ("pass", [card], None),
          ("missing", [], "does not report the card"),
          ("unavailable", [{**card, "state": "unavailable"}], "does not report the card"),
          ("version", [{**card, "software": {"packages": {}}}], "reports card packages"),
      ):
        with self.subTest(case=case), \
            mock.patch.object(HARDWARE, "customer_process_environment",
                              return_value={"PATH": "p", "USERPROFILE": "runner"}), \
            mock.patch.object(HARDWARE, "installed_neat_status", return_value={
                "host": {"packages": {"sima-pcie-host": {}}}, "cards": cards,
                "errors": []}) as status:
          call = partial(HARDWARE.verify_installed_neat_status, args, candidate,
                         Path("install"), root / "home", card_packages, root)
          if expected:
            with self.assertRaisesRegex(RuntimeError, expected):
              call()
          else:
            self.assertEqual(call()["status"], "PASS")
          install_root, environment, log, packages = status.call_args.args
          self.assertEqual(packages, {"simaai-scheduler-pcie-host": "0.6.2",
                                      "sima-pcie-host": "0.6.2"})
          self.assertEqual(environment["USERPROFILE"], str(root / "home"))
          self.assertEqual(environment["NEAT_PCIE_SSH_KEY"], str(key.resolve()))
          self.assertEqual(log, root / "customer-neat-status.log")


class HostedWindowsBuildContracts(unittest.TestCase):
  """The Windows build runs on GitHub-hosted runners; card testing stays self-hosted."""

  INPUTS = SCRIPT_DIR / "windows_build_inputs.json"
  PROVISION = SCRIPT_DIR / "provision_windows_build.ps1"
  PUBLISH_CONDITION = (
      "github.repository == 'sima-neat/core' && (github.event_name == 'push' || "
      "github.event_name == 'workflow_dispatch') && startsWith(github.ref, 'refs/tags/v') && "
      "inputs.validation_only != true && (inputs.require_upgrade == true || "
      "inputs.first_release_upgrade_exemption == true) && needs.hardware.result == 'success' && "
      "needs.linux_release.result == 'success'")

  def setUp(self):
    self.windows = WORKFLOW.read_text(encoding="utf-8")
    self.main = MAIN_WORKFLOW.read_text(encoding="utf-8")
    self.matrix = workflow_job(self.main, "build-pciehost-packages")

  def matrix_steps(self):
    steps = self.matrix.split("    steps:\n", 1)[1]
    return re.split(r"(?m)^(?=      - )", steps)[1:]

  def test_build_jobs_run_on_hosted_windows_without_the_production_environment(self):
    for name in ("dependencies", "native"):
      with self.subTest(job=name):
        job = workflow_job(self.windows, name)
        self.assertIn("    runs-on: windows-2025\n", job)
        self.assertNotRegex(job, r"(?m)^    environment:")
        # Producer inputs come from pinned provisioning, not runner-specific variables.
        self.assertNotRegex(job, r"vars[.](GSTREAMER|SIMA_WINDOWS_(PLATFORM|CMAKE|DEPENDENCY|PYTHON))")
    self.assertIn(
        "    runs-on: ${{ matrix.distro == 'windows' && 'windows-2025' || fromJSON(", self.matrix)
    self.assertIn("          - distro: windows\n            deb_arch: amd64\n", self.matrix)
    self.assertIn("    container: ${{ matrix.distro != 'windows' && format(", self.matrix)
    self.assertIn("matrix.deb_arch) || '' }}\n", self.matrix)
    self.assertIn("    environment: ${{ needs.resolve-vulcan-config.outputs.vulcan_env }}\n",
                  workflow_job(self.windows, "publish"))

  def test_hardware_lane_stays_self_hosted_and_follows_windows_host_changes(self):
    hardware = workflow_job(self.windows, "hardware")
    # Selected like the Linux card job (labels only, including pcie), not by group.
    self.assertIn("    runs-on: [self-hosted, Windows, X64, pcie]\n", hardware)
    self.assertIn("    - pcie\n", (ROOT / ".github" / "actionlint.yaml").read_text(
        encoding="utf-8"))
    elevation = workflow_step(hardware, "Require an elevated runner")
    self.assertIn("sc.exe config <runner service>", elevation)
    self.assertIn("obj= .\\<administrator> password= <password>", elevation)
    self.assertIn("    environment: production\n", hardware)
    self.assertIn("needs.native.result == 'success'", job_condition(self.windows, "hardware"))
    for gated in ("build-pcie-card-offline-bundle", "test-pciehost-windows"):
      self.assertIn("needs.detect-changes.outputs.windows_pcie == 'true'",
                    job_condition(self.main, gated))

  def test_publication_gating_is_unchanged(self):
    self.assertEqual(job_condition(self.windows, "publish"), self.PUBLISH_CONDITION)
    for name in ("Configure AWS credentials", "Publish PCIe host Debian packages",
                 "Retain this run's PCIe host candidate"):
      with self.subTest(step=name):
        self.assertIn("matrix.distro != 'windows'", workflow_step(self.matrix, name))
    windows_steps = [step for step in self.matrix_steps()
                     if "matrix.distro == 'windows'" in step]
    self.assertTrue(windows_steps)
    for step in windows_steps:
      self.assertNotIn("aws ", step)
      self.assertNotIn("configure-aws-credentials", step)

  def test_ubuntu_steps_exclude_only_the_windows_entry(self):
    shared = ("Checkout code", "Mint read-only Scheduler run token")
    for step in self.matrix_steps():
      name = re.match(r"      - (?:name: (.+)|uses: (.+))\n", step)
      label = name[1] or name[2]
      condition = re.search(r"(?m)^        if: (.+)$", step)
      condition = condition[1] if condition else ""
      with self.subTest(step=label):
        if label in shared:
          self.assertNotIn("matrix.distro", condition)
        elif "matrix.distro != 'windows'" in condition:
          self.assertNotIn("matrix.distro == 'windows'", condition)
        else:
          self.assertIn("matrix.distro == 'windows'", condition)
        self.assertNotIn("continue-on-error", step)

  def test_every_pinned_input_is_hash_verified_before_use(self):
    manifest = json.loads(self.INPUTS.read_text(encoding="utf-8"))
    names = [entry["name"] for entry in manifest["files"]]
    self.assertEqual(len(names), len(set(names)))
    self.assertEqual(
        {entry["use"] for entry in manifest["files"]},
        {"gstreamer", "opencv", "nlohmann-json", "wix", "wheelhouse", "sima-cli"})
    for entry in manifest["files"]:
      with self.subTest(input=entry["name"]):
        self.assertRegex(entry["url"], r"^https://[^\s?#]+$")
        self.assertEqual(unquote(entry["url"].rsplit("/", 1)[1]), entry["name"])
        self.assertRegex(entry["sha256"], r"^[0-9a-f]{64}$")
    wheels = {entry["name"] for entry in manifest["files"] if entry["use"] == "sima-cli"}
    cli_wheel = f"sima_cli-{manifest['sima_cli_version']}-py3-none-any.whl"
    self.assertIn(cli_wheel, wheels)
    # The bootstrap CLI is a Vulcan sima-cli publication, as on Linux, pinned to the
    # first build that reads MSI-style Windows package metadata.
    cli_url = next(entry["url"] for entry in manifest["files"] if entry["name"] == cli_wheel)
    branch, _, commit = manifest["sima_cli_publication"].removeprefix("sima-cli@").partition(":")
    self.assertEqual(
        cli_url.rsplit("/", 1)[0],
        f"https://artifacts.neat.sima.ai/sima-cli/{quote(quote(branch, safe=''), safe='')}/{commit}")
    self.assertIn("colorama-0.4.6-py2.py3-none-any.whl", wheels)
    self.assertTrue(any(name.startswith("numpy-") and "cp314-win_amd64" in name.replace(
        "cp314-cp314", "cp314") for name in (
            entry["name"] for entry in manifest["files"] if entry["use"] == "wheelhouse")))

    script = self.PROVISION.read_text(encoding="utf-8")
    # One download helper, which compares Get-FileHash with the pin before returning.
    self.assertEqual(script.count("Invoke-WebRequest"), 1)
    helper = script.split("function Get-PinnedFile", 1)[1].split("\n}\n", 1)[0]
    self.assertIn("Get-FileHash -Algorithm SHA256", helper)
    self.assertIn("-ne $Sha256) { throw", helper)
    self.assertEqual(script.count("Get-PinnedFile "), 1)  # every manifest entry
    self.assertIn('"--no-index", "--find-links"', script)
    for forbidden in ("curl", "wget", "Start-BitsTransfer", "WebClient", "pip download"):
      self.assertNotIn(forbidden, script)

    hosted = [workflow_job(self.windows, "dependencies"), workflow_job(self.windows, "native"),
              *(step for step in self.matrix_steps()
                if "matrix.distro != 'windows'" not in step)]
    for body in hosted:
      for forbidden in ("Invoke-WebRequest", "curl ", "install_sima_cli_main.sh"):
        self.assertNotIn(forbidden, body)
      for action in re.findall(r"uses: (actions/setup-python@\S+)", body):
        self.assertRegex(action, r"@[0-9a-f]{40}$")

  def test_pinned_downloads_retry_transfers_but_never_the_checksum(self):
    # About fifty downloads gate the release build; one transient failure must not fail it.
    script = self.PROVISION.read_text(encoding="utf-8")
    helper = script.split("function Get-PinnedFile", 1)[1].split("\n}\n", 1)[0]
    loop, after = helper.split("for ($attempt = 1; ; $attempt++) {", 1)[1].split("\n  }\n", 1)
    self.assertIn("try { Invoke-WebRequest -Uri $Url -OutFile $Path; break }", loop)
    self.assertIn("if ($attempt -ge 4) { throw }", loop)
    self.assertIn("Start-Sleep -Seconds (10 * $attempt)", loop)
    self.assertNotIn("Get-FileHash", loop)
    self.assertIn("Get-FileHash -Algorithm SHA256", after)
    self.assertIn("-ne $Sha256) { throw", after)

  def test_platform_inputs_come_only_from_the_scheduler_package(self):
    # As on Linux: no Platform download, installation, or file hashes on the build host.
    producer = SCRIPT.read_text(encoding="utf-8")
    scripts = "".join((SCRIPT_DIR / name).read_text(encoding="utf-8") for name in (
        "provision_windows_build.ps1", "run_windows_build.ps1"))
    hosted = "".join((workflow_job(self.windows, "dependencies"),
                      workflow_job(self.windows, "native"), self.matrix, scripts))
    for forbidden in ("SIMA_WINDOWS_PLATFORM", "PLATFORM_SDK", "PLACEHOLDER"):
      self.assertNotIn(forbidden, hosted)
    for forbidden in ("PLATFORM_SDK", "PLACEHOLDER"):
      self.assertNotIn(forbidden, self.windows + self.main)
    for forbidden in ('"--platform-include"', '"--platform-runtime"', "platform_runtime",
                      "[gst_inspect, plugin]"):
      self.assertNotIn(forbidden, producer)
    # The doctor needs the Platform service; hardware qualification runs it.
    self.assertNotRegex(producer, r'"/c",\s*install_root / "bin" / "sima-pcie-host-doctor.cmd"')
    self.assertIn('"platform": {"version": producer_dependencies["platform_version"]}', producer)
    # The record of the Platform Scheduler was built against travels, checksum-bound,
    # from the installed Scheduler package to the card PC's check.
    self.assertIn(
        'Path("share") / "SimaScheduler" / "host" / "platform-windows.json"', producer)
    self.assertIn('"scheduler-platform-windows.json": sha256(platform_record)', producer)
    self.assertIn('"scheduler-platform-windows.json",',
                  (SCRIPT_DIR / "windows_candidate.py").read_text(encoding="utf-8"))
    # The card PC holds the real Platform; the hardware job still names it.
    self.assertIn('"--platform-runtime", $env:SIMA_WINDOWS_PLATFORM_RUNTIME_DIR',
                  workflow_job(self.windows, "hardware"))

  def test_card_pc_platform_runtime_matches_scheduler_record(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      candidate_root = root / "candidate"
      runtime = root / "Platform runtime"
      candidate_root.mkdir()
      runtime.mkdir()
      (runtime / "SiMaPCIe.dll").write_bytes(b"platform")
      (runtime / "SiMaCpp.dll").write_bytes(b"not built against")
      expected = [{"name": "SiMaPCIe.dll", "sha256": sha256(runtime / "SiMaPCIe.dll")}]
      (candidate_root / "scheduler-platform-windows.json").write_text(json.dumps({
          "platform_version": "3.0.0.2",
          "runtime_dlls": {item["name"]: item["sha256"] for item in expected}}),
          encoding="utf-8")
      candidate = SimpleNamespace(root=candidate_root)
      self.assertEqual(HARDWARE.verify_platform_runtime(candidate, runtime), expected)
      (runtime / "SiMaPCIe.dll").write_bytes(b"another Platform build")
      with self.assertRaisesRegex(RuntimeError, "differs from the DLLs Scheduler"):
        HARDWARE.verify_platform_runtime(candidate, runtime)
      (runtime / "SiMaPCIe.dll").unlink()
      with self.assertRaisesRegex(RuntimeError, "differs from the DLLs Scheduler"):
        HARDWARE.verify_platform_runtime(candidate, runtime)
      (candidate_root / "scheduler-platform-windows.json").write_text(
          json.dumps({"platform_version": "3.0.0.2"}), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "no Platform runtime DLLs"):
        HARDWARE.verify_platform_runtime(candidate, runtime)

  def test_card_lane_reuses_the_matrix_cohort_from_vulcan_ci(self):
    for name, path in (
        ("core-pciehost-windows-dependencies", "${{ runner.temp }}/core-windows-dependencies"),
        ("core-pciehost-windows-amd64", "${{ runner.temp }}/core-windows-dist/*")):
      with self.subTest(artifact=name):
        self.assertIn(f"          name: {name}\n          path: {path}\n", self.matrix)
        self.assertIn(f"          name: {name}\n", workflow_job(self.windows, "hardware"))
    self.assertIn("&& inputs.linux_current_run != true",
                  job_condition(self.windows, "dependencies"))
    self.assertIn("linux_current_run: true", workflow_job(self.main, "test-pciehost-windows"))
    condition = job_condition(self.windows, "hardware")
    upstream = {"needs.linux_release.result": "success", "needs.card_candidate.result": "success",
                "needs.resolve-vulcan-config.result": "success", "cancelled()": False}
    for current_run, native, runs in (
        (True, "skipped", True), (False, "skipped", False), (False, "success", True),
        (True, "failure", False), (False, "failure", False)):
      with self.subTest(linux_current_run=current_run, native=native):
        self.assertEqual(evaluate_actions_expression(condition, {
            **upstream, "inputs.linux_current_run": current_run,
            "needs.native.result": native}), runs)
    # The matrix honours the same sima-cli selector the card lane would have.
    resolve = workflow_step(self.matrix, "Resolve the Windows dependency cohort")
    self.assertIn('"--sima-cli-candidate", "$env:RUNNER_TEMP\\sima-cli-override"', resolve)
    for name in ("Admit exact sima-cli candidate", "Download exact sima-cli candidate"):
      self.assertIn("if: inputs.sima_cli_run_id != '' && matrix.distro == 'windows'",
                    workflow_step(self.matrix, name))
    # Without a selector, the card lane's customer CLI is the bootstrap's pinned
    # publication; a direct (release) dispatch keeps sima-cli main.
    self.assertIn('$arguments += @("--sima-cli-branch", $Matches[1], "--sima-cli-spec", $Matches[2])',
                  resolve)
    pattern = re.search(r"-notmatch '([^']+)'", resolve)[1]
    publication = json.loads(
        (SCRIPT_DIR / "windows_build_inputs.json").read_text(encoding="utf-8"))[
            "sima_cli_publication"]
    self.assertEqual(re.fullmatch(pattern, publication).groups(),
                     ("codex/1074-windows-msi-install", "91d5420c7db9"))
    self.assertNotIn("--sima-cli-branch", workflow_job(self.windows, "dependencies"))
    self.assertIn('parser.add_argument("--sima-cli-spec", default="latest")',
                  (SCRIPT_DIR / "acquire_windows_dependencies.py").read_text(encoding="utf-8"))

  def test_card_lane_needs_no_repository_variables(self):
    hardware = workflow_job(self.windows, "hardware")
    environment = hardware.split("    env:\n", 1)[1].split("    steps:\n", 1)[0]
    # Like test-pciehost-hardware: a fixed card address and one optional debug variable.
    self.assertEqual(re.findall(r"vars[.](\w+)", hardware), ["SIMAPCIE_PCIEHOST_CARD_GST_DEBUG"])
    self.assertIn('      SIMAPCIE_CARD_HOST: "10.0.0.2"\n', environment)
    self.assertNotRegex(hardware, r"SIMA_WINDOWS_PCIE_(SSH_KEY|CARD_HOST|YOLOV8|TEST_IMAGE)")
    linux = workflow_job(self.main, "test-pciehost-hardware")
    self.assertIn('      SIMAPCIE_CARD_HOST: "10.0.0.2"\n', linux)
    self.assertIn("SIMAPCIE_CARD_GST_DEBUG: ${{ vars.SIMAPCIE_PCIEHOST_CARD_GST_DEBUG }}", linux)

    # Inputs the build used travel as artifacts from either Windows producer.
    for producer in (self.matrix, workflow_job(self.windows, "native")):
      upload = workflow_step(producer, "Retain this run's Windows card inputs")
      self.assertIn("name: core-pciehost-windows-card-inputs", upload)
      self.assertIn("/windows-inputs/wheelhouse", upload)
      self.assertIn("/windows-inputs/opencv/opencv/build/x64/vc16/bin/*.dll", upload)
    self.assertIn("name: core-pciehost-windows-card-inputs", hardware)
    manifest = json.loads((SCRIPT_DIR / "windows_build_inputs.json").read_text(encoding="utf-8"))
    wheelhouse = {entry["name"].split("-", 1)[0].lower() for entry in manifest["files"]
                  if entry["use"] == "wheelhouse"}
    self.assertTrue({"numpy", "opencv_python", "pytest", "iniconfig", "pluggy",
                     "pillow"} <= wheelhouse)
    self.assertIn('"wheelhouse", "sima-cli")', self.PROVISION.read_text(encoding="utf-8"))

    # The card key: pcie-setup.ps1 as on Linux, password enrollment only when needed.
    ssh = workflow_step(hardware, "Provision PCIe card SSH key")
    self.assertIn("SIMAPCIE_SSH_PASSWORD: ${{ secrets.DEVKIT_PASSWORD }}", ssh)
    self.assertIn('Join-Path $env:USERPROFILE ".ssh\\sima_neat_pcie_ed25519"', ssh)
    self.assertIn("NonInteractive = $true", ssh)
    self.assertIn("scripts/ci/pciehost/enroll_windows_card_key.py", ssh)
    self.assertIn("pcie-setup.ps1 @setup -Strict", ssh)
    # Machine-wide caches: the temporary standard user reads the GenAI models and fixtures.
    assets = workflow_step(hardware, "Resolve PCIe hardware test assets")
    for cache in ("pciehost-assets", "models_genai"):
      self.assertIn(f'(Join-Path $env:ProgramData "sima-neat-ci\\{cache}")', assets)
    self.assertNotIn("$env:USERPROFILE", assets)
    linux_ssh = workflow_step(linux, "Provision PCIe card SSH key")
    self.assertIn('--key "${HOME}/.ssh/sima_neat_pcie_ed25519"', linux_ssh)
    helper = (SCRIPT_DIR / "enroll_windows_card_key.py").read_text(encoding="utf-8")
    remote = ast.literal_eval(re.search(r'REMOTE_SCRIPT = ("""[\s\S]*?""")', helper)[1])
    setup = (ROOT / "pcie_host/scripts/windows/pcie-setup.ps1").read_text(encoding="utf-8")
    expected = re.search(r"\$remoteScript = @'\r?\n([\s\S]*?)\r?\n'@", setup)[1]
    self.assertEqual(remote, expected.replace("\r", "") + "\n")
    self.assertIn("paramiko.RejectPolicy()", helper)

  def test_card_inputs_mirror_the_linux_asset_sources(self):
    spec = importlib.util.spec_from_file_location(
        "prepare_windows_card_inputs", SCRIPT_DIR / "prepare_windows_card_inputs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    resolve = (SCRIPT_DIR / "resolve_assets.sh").read_text(encoding="utf-8")
    resolver = (ROOT / "pcie_host/scripts/resolve_hardware_test_assets.sh").read_text(
        encoding="utf-8")
    genai = (ROOT / "pcie_host/scripts/prepare_pcie_genai_models.sh").read_text(encoding="utf-8")
    # Linux takes the same Model Browser packages through the same helper.
    self.assertEqual(sorted(re.findall(r"model_browser_mpk (\S+) (\S+)\)", resolve)),
                     sorted(module.MODEL_BROWSER_MODELS.values()))
    self.assertIn("scripts/ci/prepare_model_browser_fixture.py", resolve)
    for name in module.MODEL_BROWSER_MODELS:
      self.assertIn(name, resolve)
    self.assertIn(module.TEST_IMAGE_URL, resolver)
    self.assertIn('"${WORKSPACE}/tests/images/people.jpg"', resolver)
    for name, repository, config, revision in module.GENAI_MODELS:
      with self.subTest(model=name):
        self.assertIn(name, genai)
        self.assertIn(config, genai)
        self.assertIn(repository.rsplit("/", 1)[0], genai)
        if revision:
          self.assertIn(revision, genai)
    for name in module.GENAI_TEST_ASSETS:
      self.assertTrue((ROOT / name).is_file(), name)

    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      service = root / "Platform"
      service.mkdir()
      (service / "SiMaPCIe.dll").write_bytes(b"platform")
      # The registry ImagePath, which no display language translates; not sc.exe qc.
      self.assertNotIn('"sc.exe"', (SCRIPT_DIR / "prepare_windows_card_inputs.py").read_text(
          encoding="utf-8"))
      executable = service / "simaai_mla_daemon.exe"
      for image in (f'"{executable}" --service', f"{executable} --service"):
        with self.subTest(image=image), \
            mock.patch.object(module, "service_image_path", return_value=image):
          self.assertEqual(module.installed_platform_runtime(), service)
      with mock.patch.object(module, "service_image_path", return_value=None):
        with self.assertRaisesRegex(RuntimeError, "SiMaMlaDaemon is not installed"):
          module.installed_platform_runtime()

      # Each run downloads the current builds again, as Linux's emptied work directory does.
      stale = root / "assets" / "model-browser" / "yolov8s" / "w8a8" / "old.zip"
      stale.parent.mkdir(parents=True)
      stale.write_bytes(b"old")
      prepare = mock.Mock(side_effect=lambda _cli, model_id, variant, output: (
          output / f"{model_id}_{variant}_mpk.tar.gz"))
      with mock.patch.object(module.prepare_model_browser_fixture, "prepare", prepare), \
          mock.patch.dict(os.environ, {"SIMA_CLI_CHECK_FOR_UPDATE": "1"}):
        models = module.resolve_model_browser_models(root / "assets", Path("sima-cli.exe"))
        self.assertEqual(os.environ["SIMA_CLI_CHECK_FOR_UPDATE"], "0")
      self.assertFalse(stale.exists())
      output = root / "assets" / "model-browser"
      self.assertEqual(prepare.call_args_list, [
          mock.call("sima-cli.exe", model_id, variant, output)
          for model_id, variant in module.MODEL_BROWSER_MODELS.values()])
      self.assertEqual(models["SIMAPCIE_MLA_ONLY_BF16_MODEL"],
                       output / "yolo26n_w16a16_mpk.tar.gz")

      # The written environment names every input the qualifier requires, and the
      # manifest inventories exactly those files.
      inputs = root / "card inputs"
      (inputs / "wheelhouse").mkdir(parents=True)
      (inputs / "opencv/opencv/build/x64/vc16/bin").mkdir(parents=True)
      cache = root / "cache"
      cache.mkdir()
      files = {name: cache / f"{name}.bin" for name in (
          "SIMAPCIE_YOLOV8_MODEL", "SIMAPCIE_MLA_ONLY_MODEL", "SIMAPCIE_MLA_ONLY_BF16_MODEL",
          "SIMAPCIE_TEST_IMAGE")}
      for path in files.values():
        path.write_bytes(path.name.encode())
      gstreamer = root / "gstreamer"
      gstreamer.mkdir()
      env_file = root / "github-env"
      argv = ["prepare", "--workspace", str(ROOT), "--card-inputs", str(inputs),
              "--cache-dir", str(cache), "--genai-models", str(root / "models"),
              "--sima-cli", "sima-cli", "--manifest", str(root / "assets.json"),
              "--revision", "core-1-1", "--env-file", str(env_file)]
      with mock.patch.object(module, "installed_platform_runtime", return_value=service), \
          mock.patch.object(module, "resolve_model_browser_models",
                            return_value={name: files[name]
                                          for name in module.MODEL_BROWSER_MODELS}), \
          mock.patch.object(module, "resolve_test_image",
                            return_value=files["SIMAPCIE_TEST_IMAGE"]), \
          mock.patch.dict(os.environ, {"GSTREAMER_ROOT_X86_64": str(gstreamer)}), \
          mock.patch.object(sys, "argv", argv):
        module.main()
      written = dict(line.split("=", 1) for line in env_file.read_text().splitlines())
      self.assertEqual(written["SIMA_WINDOWS_PLATFORM_RUNTIME_DIR"], str(service))
      self.assertEqual(written["SIMA_WINDOWS_PYTHON_WHEELHOUSE"], str(inputs.resolve() / "wheelhouse"))
      self.assertEqual(written["SIMAPCIE_BOXDECODE_IMAGE"], str(ROOT / "tests/images/people.jpg"))
      self.assertEqual(written["SIMA_WINDOWS_PCIE_ASSET_MANIFEST"],
                       str((root / "assets.json").resolve()))
      assets = json.loads((root / "assets.json").read_text(encoding="utf-8"))
      self.assertFalse(assets["include_genai"])
      self.assertEqual(set(assets["vision"]), set(VISION_ASSETS))

  def test_receipt_records_the_hosted_toolchain(self):
    environment = {"VSCMD_VER": "17.14.4", "VCToolsVersion": "14.44.35207",
                   "WindowsSDKVersion": "10.0.26100.0\\", "ImageOS": "win25",
                   "ImageVersion": "20261110.1"}
    with mock.patch.dict(os.environ, environment, clear=False):
      self.assertEqual(MODULE.toolchain_identity(), {
          "visual_studio": "17.14.4", "msvc_tools": "14.44.35207",
          "windows_sdk": "10.0.26100.0", "runner_image": "win25",
          "runner_image_version": "20261110.1"})
    self.assertIn("**toolchain_identity()", SCRIPT.read_text(encoding="utf-8"))

  def test_windows_build_failure_fails_promotion_and_validation(self):
    # A failed Windows entry fails the matrix job, which publication metadata,
    # promotion, and the validation-only gate all require to succeed.
    self.assertNotIn("continue-on-error", self.matrix)
    self.assertIn("      - build-pciehost-packages\n", workflow_job(self.main, "publish-pciehost-package"))
    self.assertIn("needs.publish-pciehost-package.result == 'success'",
                  job_condition(self.main, "promote-latest-tag"))
    validation = workflow_job(self.main, "validation-complete")
    self.assertIn("      - build-pciehost-packages\n", validation)
    skipped = re.findall(r"skipped(?: \|)?= \{([^}]*)\}", validation)
    self.assertTrue(skipped)
    self.assertTrue(all("build-pciehost-packages" not in names for names in skipped))


if __name__ == "__main__":
  unittest.main()
