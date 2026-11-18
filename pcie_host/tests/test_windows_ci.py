#!/usr/bin/env python3
"""Contracts for the native Windows Core PCIe-host producer."""

import base64
import contextlib
import importlib.util
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
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
    dependency_identity, require_junit_cases, sha256, validate_candidate)
from windows_hardware_assets import (  # noqa: E402
    GENAI_ASSETS, VISION_ASSETS, asset_manifest_sha256, create_asset_manifest,
    normalize_asset_manifest, verify_asset_manifest)
from qualify_windows_hardware import (  # noqa: E402
    _windows_openssh_batch_line,
    ExclusiveLock, RemoteExclusiveLock, activate_card_candidate,
    card_candidate_paths, card_snapshot, card_stage_cleanup_command,
    card_management_addresses,
    create_standard_user_python, customer_process_environment, load_upgrade_base,
    isolated_sima_cli_environment,
    parse_performance_log, require_powershell as require_hardware_powershell,
    prepare_openssh_secret_directory,
    require_restored_windows_host_state,
    recover_windows_platform_device, remove_installed_tutorial_facade,
    remove_qualification_tree,
    restore_card_candidate, run_captured_to_file, run_sudo_remote,
    run_windows_openssh,
    sima_cli_identity, stage_card_candidate, stage_installed_tutorial_facade,
    stage_invalid_wheel_registry, stage_local_registry, stage_path_fixtures,
    standard_user_wrapper, stress_queues, validate_card_candidate,
    verify_customer_process_activation,
    windows_card_network_identity, windows_platform_state_snapshot)
from standard_user_genai import (  # noqa: E402
    prove_writable_sentinel, read_serve_mapping, run as run_standard_user_genai)
from verify_linux_release_run import verify_run as verify_linux_run  # noqa: E402
from verify_windows_release import verify  # noqa: E402
import qualify_windows_hardware as HARDWARE  # noqa: E402
import prepare_card_qualification_candidate as CARD_PREP  # noqa: E402
import build_card_qualification_dependencies as CARD_DEPS  # noqa: E402
import cleanup_windows_hardware as RECOVERY  # noqa: E402
import acquire_windows_dependencies as WINDOWS_DEPENDENCIES  # noqa: E402
import standard_user_genai as STANDARD_USER_GENAI  # noqa: E402
from genai_admission_contract import (  # noqa: E402
    is_bounded_competing_session_rejection, is_injected_transport_failure)


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


WINDOWS_OPENSSH_HOP_SKIP = (
    "replays the Windows OpenSSH hop into a bash card shell; runs on Linux")
WINDOWS_CMD_SKIP = "runs a batch wrapper under cmd.exe; executed on Windows"
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
           mock.patch.object(MODULE, "registry_sentinel_exists", return_value=False), \
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
    self.assertNotIn("fixture-secret", piped)

  def test_same_run_linux_admission_is_not_a_manual_dispatch_override(self):
    workflow = WORKFLOW.read_text(encoding="utf-8")
    dispatch = workflow.split("  workflow_dispatch:\n", 1)[1].split("\npermissions:", 1)[0]
    self.assertNotIn("linux_current_run:", dispatch)
    self.assertIn('"${GITHUB_WORKFLOW}" != "Vulcan CI"', workflow)
    self.assertIn("Same-run Linux admission is reserved", workflow)
    caller = MAIN_WORKFLOW.read_text(encoding="utf-8")
    job = caller.split("  test-pciehost-windows:\n", 1)[1].split("  promote-latest-tag:", 1)[0]
    self.assertIn("linux_current_run: true", job)
    self.assertIn("needs.test-devkit.result == 'success'", job)
    self.assertIn("needs.test-pciehost-hardware.result == 'success'", job)

  def test_card_qualification_dependencies_are_private_and_same_run(self):
    caller = MAIN_WORKFLOW.read_text(encoding="utf-8")
    producer = caller.split(
        "  build-pcie-card-qualification-dependencies:\n", 1)[1].split(
            "  publish-vulcan-artifacts:", 1)[0]
    self.assertIn("--network=none", producer)
    self.assertIn("--source-image", producer)
    self.assertIn("sdk-build-image.json", producer)
    self.assertIn("- build\n", producer)
    self.assertIn("core-pcie-card-qualification-dependencies", producer)
    windows_job = caller.split("  test-pciehost-windows:\n", 1)[1].split(
        "  promote-latest-tag:", 1)[0]
    self.assertIn("build-pcie-card-qualification-dependencies", windows_job)

    workflow = WORKFLOW.read_text(encoding="utf-8")
    prepare = workflow.split("  card_candidate:\n", 1)[1].split(
        "  dependencies:", 1)[0]
    self.assertIn("prepare_card_qualification_candidate.py", prepare)
    self.assertIn("core-pcie-card-qualification-dependencies", prepare)
    self.assertIn("core-pcie-card-qualification-candidate", prepare)
    hardware = workflow.split("  hardware:\n", 1)[1].split("  publish:", 1)[0]
    self.assertIn("needs: [native, linux_release, card_candidate]", hardware)
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
      root = Path(directory)
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

  def test_workflow_uses_exact_scheduler_artifact_and_restricted_runner(self):
    source = WORKFLOW.read_text(encoding="utf-8")
    self.assertIn("  pull_request:\n", source)
    self.assertIn("if: github.event_name != 'pull_request'", source)
    self.assertIn("github.repository == 'sima-neat/core'", source)
    self.assertIn("scheduler_run_id", source)
    self.assertIn("scheduler_commit", source)
    self.assertIn("--expected-scheduler-commit", source)
    # The reusable Windows workflow cannot be dispatched from a feature ref
    # before it exists on the default branch. Admit the Scheduler root run,
    # which owns the reusable job and the retained artifact under one run ID.
    self.assertIn('--workflow-name "Scheduler"', source)
    self.assertIn('--workflow-path ".github/workflows/ci.yml"', source)
    self.assertNotIn('--workflow-name "Scheduler native Windows PCIe host"', source)
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
    self.assertIn("scheduler-native-windows-amd64", source)
    self.assertIn("pcie-windows-build", source)
    self.assertIn("pcie-windows-hardware", source)
    self.assertNotIn("windows-latest", source)
    self.assertIn("scripts/ci/pciehost/build_windows.py", source)
    self.assertIn("python3 pcie_host/tests/test_verify_linux_release_run.py", source)
    self.assertIn("scripts/ci/pciehost/qualify_windows_hardware.py", source)
    self.assertIn("scripts/ci/pciehost/cleanup_windows_hardware.py", source)
    self.assertIn("needs: [hardware, linux_release]", source)
    self.assertIn("linux_run_id", source)
    self.assertIn("Download this run's exact Linux candidate", source)
    self.assertIn("Download prior run's exact Linux candidate", source)
    self.assertIn("Bind Core and SDK dependencies into one private candidate", source)
    self.assertIn("Download exact private card qualification candidate", source)
    self.assertIn("name: neat-core-dist", source)
    self.assertIn("--card-candidate", source)
    self.assertIn("--expected-card-commit", source)
    self.assertIn("SIMA_WINDOWS_PCIE_ASSET_MANIFEST", source)
    self.assertIn("SIMA_WINDOWS_PCIE_GENAI_ASSET_MANIFEST", source)
    self.assertIn("SIMA_WINDOWS_PCIE_VISION_ASSET_MANIFEST", source)
    self.assertEqual(source.count('"--asset-manifest"'), 2)
    hardware_job = source.split("  hardware:\n", 1)[1]
    hardware_env = hardware_job.split("    steps:\n", 1)[0]
    self.assertNotIn("SIMAPCIE_CARD_SUDO_PASSWORD", hardware_env)
    self.assertIn("QUALIFIER_CARD_SUDO_PASSWORD", source)
    self.assertIn("Remove-Item Env:QUALIFIER_CARD_SUDO_PASSWORD", source)
    self.assertIn('"--card-sudo-password-stdin"', source)
    recovery = source.split("- name: Recover card and native installation", 1)[1].split(
        "- name: Enforce release admission", 1)[0]
    self.assertIn("RECOVERY_CARD_SUDO_PASSWORD", recovery)
    self.assertIn("Remove-Item Env:RECOVERY_CARD_SUDO_PASSWORD", recovery)
    self.assertIn('"--card-sudo-password-stdin"', recovery)
    self.assertNotIn("SIMAPCIE_EXPECTED_CARD_RUNTIME_SHA256", source)
    self.assertNotIn("SIMAPCIE_EXPECTED_CARD_WORKER_SHA256", source)
    self.assertNotIn("SIMAPCIE_EXPECTED_CARD_GENAI_WORKER_SHA256", source)
    self.assertIn("sima_cli_run_id", source)
    self.assertIn("sima-cli-dist", source)
    self.assertIn("--require-signatures", source)
    self.assertIn('$arguments += "--require-genai"', source)
    self.assertIn('$arguments += "--require-release"', source)
    self.assertIn('if ("$env:REQUIRE_SIGNATURES" -eq "true")', source)
    self.assertIn("Recover card and native installation", source)
    self.assertIn("if: always()", source)
    self.assertIn("upgrade_base_run_id", source)
    self.assertIn("upgrade_base_commit", source)
    self.assertIn("Admit exact successful Windows upgrade baseline", source)
    self.assertIn('"Core native Windows PCIe host"', source)
    self.assertIn("core-windows-upgrade-base", source)
    self.assertIn("--expected-upgrade-base-commit", source)
    self.assertIn("SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH", source)
    self.assertIn("SIMA_WINDOWS_PLATFORM_INCLUDE_DIR", source)
    self.assertIn('source_branch: "3.0"', source)
    self.assertIn("--dependency-runtime-dir", source)
    self.assertIn("scripts/ci/pciehost/acquire_windows_dependencies.py", source)
    self.assertIn("core-pciehost-windows-dependencies", source)
    self.assertIn("linux_current_run", source)
    self.assertIn("require_upgrade", source)
    self.assertIn("first_release_upgrade_exemption", source)
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
    native_job = source.split("  native:\n", 1)[1].split("  hardware:\n", 1)[0]
    self.assertIn('"--work", $env:SIMA_WINDOWS_SHORT_WORK_ROOT', native_job)
    self.assertIn("Remove short Windows producer work root", native_job)
    self.assertEqual(
        native_job.count(
            "Remove-Item -LiteralPath $env:SIMA_WINDOWS_SHORT_WORK_ROOT -Recurse -Force"
        ),
        2,
    )
    self.assertEqual(source.count("shell: powershell"), 8)
    hardware_job = source.split("  hardware:\n", 1)[1].split("  publish:\n", 1)[0]
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

  def test_main_ci_requires_windows_acceptance_before_promotion_only_when_enabled(self):
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    self.assertIn("uses: ./.github/workflows/pcie-host-windows.yml", source)

    def job(name):
      return re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", source)[1]

    def condition(name):
      return re.search(r"(?m)^    if: (.+)$", job(name))[1]

    gate = textwrap.dedent(job("validation-complete").split(
        "          python3 - <<'PYGATE'\n", 1)[1].split("\n          PYGATE", 1)[0])
    gate_flag = re.search(r"(?m)^          WINDOWS_PCIE_CI: (.+)$", job("validation-complete"))[1]
    for variable in ("", "false", "true"):
      for hardware in ("success", "skipped"):
        for outcome in ("success", "failure"):
          with self.subTest(variable=variable, hardware=hardware, outcome=outcome):
            context = {
                "always()": True,
                "vars.ENABLE_WINDOWS_PCIE_CI": variable,
                "github.workflow": "Vulcan CI",
                "inputs.offline_source_run_id": "",
                "inputs.validation_only": False,
                "inputs.build_mode": "standard",
                "github.ref_type": "branch",
                "needs.detect-changes.outputs.docs_only": "false",
                "needs.test-pciehost-hardware.result": hardware,
            }
            for upstream in (
                "validate-windows-cohort-selectors", "build",
                "build-pcie-card-qualification-dependencies", "build-pciehost-packages",
                "test-devkit", "publish-pciehost-package"):
              context[f"needs.{upstream}.result"] = "success"
            runs = evaluate_actions_expression(condition("test-pciehost-windows"), context)
            self.assertEqual(runs, variable == "true" and hardware == "success")
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
            environment = {
                "RESULTS": json.dumps(results), "BUILD_MODE": "standard",
                "WINDOWS_PCIE_CI": str(evaluate_actions_expression(gate_flag, context)).lower(),
            }
            with mock.patch.dict(os.environ, environment):
              if hardware == "success" and outcome == "success" or variable != "true":
                exec(compile(gate, str(MAIN_WORKFLOW), "exec"), {})
              else:
                with self.assertRaisesRegex(SystemExit, "test-pciehost-windows"):
                  exec(compile(gate, str(MAIN_WORKFLOW), "exec"), {})

  def test_workflows_calling_main_ci_skip_the_windows_lane_and_accept_the_skip(self):
    # A called workflow sees the top-level caller's name, which the Windows
    # preflight rejects for same-run Linux admission.
    source = MAIN_WORKFLOW.read_text(encoding="utf-8")
    self.assertEqual(re.search(r"(?m)^name: (.+)$", source)[1], "Vulcan CI")
    self.assertIn('"${GITHUB_WORKFLOW}" != "Vulcan CI"', WORKFLOW.read_text(encoding="utf-8"))

    def job(name):
      return re.search(rf"(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)", source)[1]

    def condition(name):
      return re.search(r"(?m)^    if: (.+)$", job(name))[1]

    callers = {}
    for path in sorted(MAIN_WORKFLOW.parent.glob("*.yml")):
      text = path.read_text(encoding="utf-8")
      if "uses: ./.github/workflows/vulcan-ci.yml" in text:
        callers[path.name] = re.search(r"(?m)^name: (.+)$", text)[1].strip()
    self.assertEqual(set(callers), {
        "long-tests-weekly.yml", "vulcan-fuzz-nightly.yml", "zero-skip.yml"})
    gate_flag = re.search(r"(?m)^          WINDOWS_PCIE_CI: (.+)$", job("validation-complete"))[1]
    for workflow_name in (*callers.values(), "Vulcan CI"):
      with self.subTest(workflow=workflow_name):
        context = {
            "always()": True,
            "vars.ENABLE_WINDOWS_PCIE_CI": "true",
            "github.workflow": workflow_name,
            "inputs.offline_source_run_id": "",
            "inputs.validation_only": False,
            "inputs.build_mode": "standard",
            "github.ref_type": "branch",
            "needs.detect-changes.outputs.docs_only": "false",
        }
        for upstream in (
            "validate-windows-cohort-selectors", "build",
            "build-pcie-card-qualification-dependencies", "build-pciehost-packages",
            "test-devkit", "test-pciehost-hardware", "publish-pciehost-package"):
          context[f"needs.{upstream}.result"] = "success"
        called = workflow_name != "Vulcan CI"
        self.assertEqual(
            evaluate_actions_expression(condition("test-pciehost-windows"), context), not called)
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
      self.assertIn('--workflow-name "Scheduler"', body)
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
    self.assertIn(
        'scheduler_run_id and scheduler_commit must be supplied together.', packages)
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
    self.assertIn("needs: [contracts, linux_release]", dependencies)
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
    workflow = WORKFLOW.read_text(encoding="utf-8")
    step = workflow.split("- name: Validate dependency and admission selectors\n", 1)[1]
    script = textwrap.dedent(step.split("        run: |\n", 1)[1].split("\n      - name:", 1)[0])
    bash = shutil.which("bash")
    self.assertIsNotNone(bash)
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
        environment = os.environ.copy()
        for name in ("LINUX_CURRENT_RUN", "SCHEDULER_RUN_ID", "SCHEDULER_COMMIT",
                     "SIMA_CLI_RUN_ID", "SIMA_CLI_COMMIT", "LINUX_RUN_ID", "REQUIRE_UPGRADE",
                     "UPGRADE_BASE_RUN_ID", "UPGRADE_BASE_COMMIT",
                     "FIRST_RELEASE_UPGRADE_EXEMPTION",
                     "FIRST_RELEASE_UPGRADE_EXEMPTION_REASON"):
          environment[name] = ""
        environment.update({"GITHUB_WORKFLOW": "Core native Windows PCIe host",
                            "LINUX_CURRENT_RUN": "false", **selectors, **mode})
        result = subprocess.run(
            [bash, "-e", "-c", script], capture_output=True, text=True, check=False,
            env=environment)
        if accepted:
          self.assertEqual(result.returncode, 0, result.stderr)
        else:
          self.assertNotEqual(result.returncode, 0)
          self.assertIn("run selectors require validation_only=true", result.stderr)

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
        ("sima-neat/core", "push", "refs/heads/develop",
         False, "success", "success", True),
        ("sima-neat/core", "workflow_dispatch", "refs/heads/develop",
         False, "success", "success", True),
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
        "startsWith(github.ref, 'refs/tags/v')": False,
        "github.repository": "sima-neat/core",
        "github.event_name": "workflow_dispatch",
        "github.ref": "refs/heads/develop",
        "inputs.validation_only": False,
        "inputs.require_upgrade": False,
        "inputs.first_release_upgrade_exemption": True,
        "needs.hardware.result": "success",
        "needs.linux_release.result": "success",
    }
    self.assertTrue(evaluate_actions_expression(expression, first_release_values))
    first_release_values["inputs.first_release_upgrade_exemption"] = False
    self.assertFalse(evaluate_actions_expression(expression, first_release_values))
    self.assertIn("needs: [hardware, linux_release]", publish)
    self.assertIn('source_branch: "3.0"', publish)

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
    self.assertNotIn("10.0.4.2", run.call_args.args[0])
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
        ("SiMa.ai Virtual Ethernet Adapter", r"ROOT\SIMA_VETH\0000", "Net"),
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
    snapshot = {"pnp_devices": devices, "service": service}
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
    self.assertNotIn("ProcessId", powershell)
    require_restored_windows_host_state(snapshot, dict(snapshot))
    changed = json.loads(json.dumps(snapshot))
    changed["service"]["configuration"]["Start"] = 3
    with self.assertRaisesRegex(RuntimeError, "configuration/state changed"):
      require_restored_windows_host_state(snapshot, changed)

  def test_windows_platform_snapshot_rejects_non_present_driver_state(self):
    snapshot = {
        "pnp_devices": [
            {
                "class": "System", "driver_inf": "oem42.inf",
                "driver_provider": "SiMa.ai", "driver_signer": "SiMa.ai",
                "driver_version": "3.0.0.2", "friendly_name": "SiMa.ai MLA Device",
                "instance_id": r"PCI\VEN_1F06&DEV_3123\4&1", "is_signed": True,
                "manufacturer": "SiMa.ai", "status": "Error",
            },
            {
                "class": "Net", "driver_inf": "oem43.inf",
                "driver_provider": "SiMa.ai", "driver_signer": "SiMa.ai",
                "driver_version": "3.0.0.2",
                "friendly_name": "SiMa.ai Virtual Ethernet Adapter",
                "instance_id": r"ROOT\SIMA_VETH\0000", "is_signed": True,
                "manufacturer": "SiMa.ai", "status": "OK",
            },
        ],
        "service": {
            "configuration": {"ImagePath": "daemon.exe", "Start": 2},
            "display_name": "SiMa.ai MLA Daemon", "name": "SiMaMlaDaemon",
            "path_name": "daemon.exe", "service_type": "Own Process",
            "start_mode": "Auto", "start_name": "LocalSystem", "state": "Running",
        },
    }
    completed = subprocess.CompletedProcess([], 0, json.dumps(snapshot), "")
    with tempfile.TemporaryDirectory() as directory, mock.patch(
        "qualify_windows_hardware.require_powershell", return_value="powershell.exe"
    ), mock.patch(
        "qualify_windows_hardware.subprocess.run", return_value=completed
    ), self.assertRaisesRegex(RuntimeError, "PnP identity was invalid"):
      windows_platform_state_snapshot(Path(directory) / "platform.log")

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
    self.assertIn('cpack_output = work / "cpack"', source)
    self.assertIn('"-B", cpack_output', source)
    self.assertNotIn('"-B", output, "-C", "Release"', source)
    self.assertIn("validate_candidate(output)", source)

  def test_hardware_qualification_bounds_blocker_and_injects_process_faults(self):
    source = (SCRIPT_DIR / "qualify_windows_hardware.py").read_text(encoding="utf-8")
    customer_recovery = source.split(
        'registry_root = work / "complete registry with spaces"', 1
    )[1].split('result["customer_install"]', 1)[0]
    self.assertIn('"--native-only",', source)
    self.assertIn("managed_python.is_file()", customer_recovery)
    self.assertNotIn('"--python-executable", python', customer_recovery)
    self.assertIn(
        'require_junit_cases(evidence / "python-hardware.xml", set(BASE_PYTHON_TESTS))',
        source)
    self.assertIn('require_junit_cases(evidence / "python-genai.xml", genai_tests)', source)
    self.assertIn("run_genai_fault_recovery_tests(", source)
    self.assertIn('stop_windows_platform_service(evidence / "genai-service-loss-injection.log")',
                  source)
    self.assertIn('evidence / "genai-worker-loss-injection.log"', source)
    self.assertIn('completed.append("test_genai_resource_cleanup")', source)
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
    self.assertIn("New-LocalUser", source)
    self.assertIn("Remove-LocalUser", source)
    self.assertIn("standard_user", source)
    self.assertIn("CreateProcessWithLogonW", source)
    self.assertIn('"SimaPCIeHostAcceptance-" + secrets.token_hex(8)', source)
    self.assertIn('evidence / "standard-user-python-install.log"', source)
    self.assertIn('"SIMAPCIE_GENAI_MEDIA_DIRECTORY"', source)
    self.assertIn('"SIMAPCIE_GENAI_MEDIA_SERVE_ROOT"', source)
    workflow = WORKFLOW.read_text(encoding="utf-8")
    self.assertIn("SIMA_WINDOWS_PCIE_GENAI_MEDIA_DIRECTORY", workflow)
    self.assertIn("SIMA_WINDOWS_PCIE_GENAI_MEDIA_SERVE_ROOT", workflow)
    self.assertIn("Required Windows GenAI variable is unset", workflow)
    self.assertIn("media overrides require directory and service root as a pair", workflow)
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
    self.assertNotIn("Path(sys.base_prefix).resolve", source)
    self.assertIn('(private_root, "temporary standard-user private input")', source)
    self.assertIn("TerminateProcess", source)
    self.assertIn("Remove-LocalUser", source)
    self.assertIn("$cleanupErrors", source)
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
    self.assertIn("card_stage_owned = True", source)
    self.assertIn("firmware_activation_attempted = True", source)
    self.assertIn("restore_card_candidate(args, remote_lock_token", source)
    self.assertIn('current=$(cat \\"$remoteproc/firmware\\")', source)
    self.assertNotIn('read -r current < \\"$remoteproc/firmware\\"', source)
    self.assertIn("stdin=subprocess.DEVNULL", source)
    self.assertIn("isolated_sima_cli_environment(", source)
    self.assertIn('evidence / "card-candidate-dependencies.log"', source)
    cleanup = (SCRIPT_DIR / "cleanup_windows_hardware.py").read_text(encoding="utf-8")
    self.assertIn("hardware work directory remained after recovery", cleanup)
    self.assertIn("--card-sudo-password-stdin", cleanup)
    self.assertIn("restore_card_candidate(args, remote_token", cleanup)
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
      doctor.write_text("@echo off\necho doctor PASS\nexit /b 0\n", encoding="ascii")
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
      result = subprocess.run(
          command, env={"SystemRoot": os.environ["SystemRoot"]}, cwd=root,
          check=False)
      log = run_log.read_text(encoding="utf-8", errors="replace")
      self.assertEqual(result.returncode, 0, log)
      self.assertIn("doctor PASS", log)
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
      }, clear=True):
        environment = isolated_sima_cli_environment(home, wheelhouse, work)
      self.assertNotIn("PYTHONPATH", environment)
      for name in ("CLAUDE_HOME", "CODEX_HOME", "SIMA_CLI_HOME", "PIP_CACHE_DIR"):
        self.assertTrue(Path(environment[name]).is_relative_to(work.resolve()))
      self.assertEqual(environment["USERPROFILE"], str(home.resolve()))
      self.assertEqual(environment["HOME"], str(home.resolve()))
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
        completed = HARDWARE.run_genai_fault_recovery_tests(
            args, Path("python.exe"), evidence, {"TEST": "1"})

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
        "platform": {
            "version": "3.0.0.2",
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
            core_signatures.name: sha256(core_signatures),
        },
        "repository": "sima-neat/core",
        "scheduler_source_commit": "d" * 40,
    }), encoding="utf-8")
    qualification = root.parent / f"{root.name}-qualification.json"
    self.addCleanup(qualification.unlink, missing_ok=True)
    asset_manifest_path = qualification.with_name("hardware-asset-manifest.json")
    self.addCleanup(asset_manifest_path.unlink, missing_ok=True)
    asset_manifest = fixture_asset_manifest()
    asset_manifest_path.write_text(
        json.dumps(asset_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    qualification.write_text(json.dumps({
        "assets": {
            "identity_sha256": asset_manifest_sha256(asset_manifest),
            "manifest": asset_manifest,
        },
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
        "card_network_baseline": {
            "card_host": "10.0.0.2",
            "interface_description": "SiMa.ai Virtual Ethernet Adapter",
            "interface_guid": "11111111-2222-3333-4444-555555555555",
            "interface_index": 7,
            "local_address": "10.0.0.1",
            "pnp_device_id": r"ROOT\SIMA_VETH\0000",
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
        "card_network_final": {
            "card_host": "10.0.0.2",
            "interface_description": "SiMa.ai Virtual Ethernet Adapter",
            "interface_guid": "11111111-2222-3333-4444-555555555555",
            "interface_index": 7,
            "local_address": "10.0.0.1",
            "pnp_device_id": r"ROOT\SIMA_VETH\0000",
        },
        "cleanup": {
            "card_candidate": "PASS", "core_msi": "PASS", "errors": [],
            "customer_state": "PASS", "host_state": "PASS", "python": "PASS",
            "remote_lock": "PASS", "scheduler_msi": "PASS", "work": "PASS"},
        "customer_install": {
            "command": "core/pciehost/windows/amd64@3.0",
            "download_path_with_spaces": r"C:\customer downloads\candidate",
            "missing_wheel": {"exit_code": 1, "status": "PASS"},
            "native_only": "PASS",
            "plugin_discovery": "PASS",
            "process_activation": "PASS",
            "repair": {"core": "PASS", "scheduler": "PASS"},
            "recovery": "PASS",
            "repeat": "PASS",
            "rollback": {
                "baseline_commit": "b" * 40,
                "baseline_version": "0.6.1",
                "candidate_removal": {"core": "PASS", "scheduler": "PASS"},
                "candidate_version": version,
                "status": "PASS",
            },
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
    cpp_tests = {
        "test_genai_llm_run", "test_genai_vlm_run", "test_genai_asr_run",
    }
    python_tests = {
        "test_genai_asr_file_pcm_translation_and_stream",
        "test_genai_bounded_cancellation_soak_and_context_cleanup",
        "test_genai_competing_sessions_are_isolated",
        "test_genai_llm_run_stream_history_and_cancel",
        "test_genai_request_timeout_and_restart_after_failure",
        "test_genai_resource_cleanup", "test_genai_restart_after_service_loss",
        "test_genai_restart_after_worker_loss", "test_genai_service_loss_recovery",
        "test_genai_vlm_run_and_stream", "test_genai_worker_loss_recovery",
    }
    value = json.loads(qualification.read_text(encoding="utf-8"))
    value["tests"]["cpp"].extend(sorted(cpp_tests))
    value["tests"]["python"].extend(sorted(python_tests))
    value["genai_admission"] = {
        name: "PASS" for name in sorted(cpp_tests | python_tests)
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
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      with self.assertRaisesRegex(ValueError, "does not admit GenAI assets"):
        verify(root, qualification, commit, False, require_genai=True)

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
      build_identity = source / "sdk-build-image.json"
      build_identity.write_text(json.dumps({
          "core_commit": commit,
          "schema_version": 1,
          "source_image": "sha256:" + "a" * 64,
      }), encoding="utf-8")
      dependency = dependencies / "libfmt10_arm64.deb"
      dependency.write_bytes(b"dependency package")
      producer = {
          "architecture": "arm64",
          "core_commit": commit,
          "packages": [{"file": dependency.name, "sha256": sha256(dependency)}],
          "schema_version": 1,
          "sdk_build_identity_sha256": sha256(build_identity),
          "source_image": "sha256:" + "a" * 64,
      }
      (dependencies / "card-qualification-dependencies.json").write_text(
          json.dumps(producer), encoding="utf-8")
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
      self.assertEqual(receipt["producer"], producer)
      self.assertEqual(
          receipt["source_sdk_build_identity_sha256"], sha256(build_identity))
      self.assertIn(build_identity.name, candidate.resources)

  def test_private_card_candidate_rejects_uninventoried_dependency(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      source = root / "source"
      dependencies = root / "dependencies"
      source.mkdir()
      dependencies.mkdir()
      commit = "c" * 40
      package = source / "sima-neat-Linux-core.deb"
      package.write_bytes(b"card package")
      build_identity = source / "sdk-build-image.json"
      build_identity.write_text(json.dumps({
          "core_commit": commit,
          "schema_version": 1,
          "source_image": "sha256:" + "a" * 64,
      }), encoding="utf-8")
      (source / "metadata-all.json").write_text(json.dumps({
          "resources": [package.name],
          "resources-checksum": {package.name: sha256(package)},
          "version": "0.6.2+develop." + commit[:12],
      }), encoding="utf-8")
      dependency = dependencies / "dependency_arm64.deb"
      dependency.write_bytes(b"dependency package")
      (dependencies / "card-qualification-dependencies.json").write_text(
          json.dumps({
              "architecture": "arm64",
              "core_commit": commit,
              "packages": [],
              "schema_version": 1,
              "sdk_build_identity_sha256": sha256(build_identity),
              "source_image": "sha256:" + "a" * 64,
          }), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "no package inventory"):
        CARD_PREP.prepare(source, dependencies, root / "output", commit)

  def test_card_dependency_builder_emits_bound_debian_package(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      sysroot = root / "sysroot"
      sysroot.mkdir()
      sdk_release = root / "sdk-release"
      sdk_release.write_text("PLATFORM_VERSION=3.0.0.2\n", encoding="utf-8")
      build_identity = root / "sdk-build-image.json"
      build_identity.write_text(json.dumps({
          "core_commit": "c" * 40,
          "schema_version": 1,
          "source_image": "sha256:" + "a" * 64,
      }), encoding="utf-8")
      output = root / "output"

      def copy_fixture(_sysroot, package, name):
        destination = package / CARD_DEPS.LIBRARY_DIRECTORY / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(name.encode("ascii"))
        return {"file": name, "sha256": sha256(destination), "soname": name}

      dpkg_deb = shutil.which("dpkg-deb")

      def build_fixture(command, **_kwargs):
        archive_path = Path(command[-1])
        archive_path.write_bytes(b"format-faithful packaging is covered on Linux")
        return subprocess.CompletedProcess(command, 0, "", "")

      patches = [mock.patch.object(
          CARD_DEPS, "copy_library", side_effect=copy_fixture)]
      if not dpkg_deb:
        patches.append(mock.patch.object(
            CARD_DEPS.subprocess, "run", side_effect=build_fixture))
      with patches[0]:
        with patches[1] if len(patches) == 2 else contextlib.nullcontext():
          archive = CARD_DEPS.build(
              sysroot, sdk_release, build_identity, output,
              "sha256:" + "a" * 64, "c" * 40, 1700000000)
      self.assertTrue(archive.is_file())
      receipt = json.loads(
          (output / "card-qualification-dependencies.json").read_text(
              encoding="utf-8"))
      self.assertEqual(receipt["core_commit"], "c" * 40)
      self.assertEqual(receipt["packages"], [
          {"file": archive.name, "sha256": sha256(archive)}])
      if dpkg_deb:
        listing = subprocess.run(
            [dpkg_deb, "-c", str(archive)], capture_output=True, text=True,
            timeout=30, check=True).stdout
        for name in CARD_DEPS.REQUIRED_LIBRARIES:
          self.assertIn(name, listing)

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
          card_user="sima", ssh_key=key, work=root / "work")
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
      self.assertNotIn("fixture-secret", log.read_text(encoding="utf-8"))

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
      key = root / "id_ed25519"
      key.write_text("fixture", encoding="ascii")
      args = SimpleNamespace(
          card_host="10.0.0.2", card_sudo_password_stdin=True,
          card_sudo_password_value="fixture-secret", card_user="sima", ssh_key=key,
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
    key = root / "id_ed25519"
    key.write_text("fixture", encoding="ascii")
    return SimpleNamespace(
        card_host="10.0.0.2", card_user="sima", ssh_key=key,
        card_sudo_password_stdin=True, card_sudo_password_value="fixture-secret",
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
  def test_card_activation_reaches_root_unchanged_across_the_windows_hop(self):
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
        with mock.patch("qualify_windows_hardware.run_logged"), \
            mock.patch("qualify_windows_hardware.subprocess.Popen", hop), \
            mock.patch("qualify_windows_hardware.run_sudo_remote",
                       wraps=HARDWARE.run_sudo_remote) as privileged:
          try:
            activate_card_candidate(
                args, card_candidate_paths("6" * 32), "modalix-cvu-fw", root)
          except RuntimeError as error:
            log = root / "card-candidate-activate.log"
            self.fail(f"{error}: {log.read_text(encoding='utf-8')}")
        activation = privileged.call_args.args[1]
        self.assertIn("awk '{print $1}'", activation)
        self.assertEqual(
            (root / "root-script").read_text(encoding="utf-8"), activation)

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
              "qualify_windows_hardware.run_captured_to_file",
              return_value=completed) as cleanup:
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

  def test_card_firmware_broker_uses_one_privileged_activation(self):
    with tempfile.TemporaryDirectory() as directory:
      evidence = Path(directory)
      token = "6" * 32
      staged = card_candidate_paths(token)
      key = evidence / "id_ed25519"
      key.write_text("fixture", encoding="ascii")
      args = SimpleNamespace(
          card_host="10.0.0.2", card_user="sima", ssh_key=key,
          remote_lock_path="/tmp/neat-pcie-card0.lock")
      with mock.patch("qualify_windows_hardware.run_logged") as logged, \
          mock.patch("qualify_windows_hardware.run_sudo_remote") as privileged:
        activate_card_candidate(args, staged, "modalix-cvu-fw", evidence)
        restore_card_candidate(args, token, evidence)
      privileged.assert_called_once()
      activation = privileged.call_args.args[1]
      self.assertIn("nohup \"$broker_root\"", activation)
      self.assertNotIn("nohup sh -c", activation)
      self.assertIn("sha256sum", activation)
      self.assertIn("firmware-restore-request", activation)
      self.assertIn("remaining=120", activation)
      broker_stage = logged.call_args_list[1].args[0][-1]
      self.assertIn("base64 -d", broker_stage)
      encoded = broker_stage.split("printf '%s' ", 1)[1].split(" | base64 -d", 1)[0]
      decoded = base64.b64decode(encoded).decode("utf-8")
      self.assertIn("remaining=10800", decoded)
      self.assertIn("trap failed EXIT HUP INT TERM", decoded)
      self.assertEqual(logged.call_args_list[1].args[1],
                       evidence / "card-candidate-broker-stage.log")
      restore = logged.call_args_list[-2].args[0][-1]
      self.assertIn("firmware-restore-request", restore)
      self.assertIn("firmware-restored", restore)
      self.assertNotIn("sudo", restore)
      self.assertEqual(logged.call_args_list[-1].args[1],
                       evidence / "card-candidate-files-cleanup.log")

  def test_card_restore_is_safe_before_activation_or_after_stage_cleanup(self):
    with tempfile.TemporaryDirectory() as directory:
      evidence = Path(directory)
      token = "7" * 32
      key = evidence / "id_ed25519"
      key.write_text("fixture", encoding="ascii")
      args = SimpleNamespace(
          card_host="10.0.0.2", card_user="sima", ssh_key=key,
          remote_lock_path="/tmp/neat-pcie-card0.lock")
      with mock.patch("qualify_windows_hardware.run_logged") as logged:
        restore_card_candidate(args, token, evidence)
      recovery = logged.call_args_list[0].args[0][-1]
      self.assertIn(f"if [ -e /tmp/neat-pcie-card0.{token} ]", recovery)
      self.assertIn(
          "else test ! -e /tmp/neat-pcie-card0.lock/baseline_firmware", recovery)
      self.assertIn(f"test ! -e /lib/firmware/neat-pcie-card0-{token}", recovery)
      cleanup = logged.call_args_list[1].args[0][-1]
      self.assertIn("rm -rf -- /tmp/neat-pcie-card0.", cleanup)

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

  def test_release_gate_requires_real_package_rollback_and_reupgrade(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      del value["customer_install"]["rollback"]
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "roll back and re-upgrade"):
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
      with self.subTest(section=section, key=key), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        commit, _, qualification = self.windows_candidate(root)
        receipt_path = root / "msi-transaction-rollback.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt[section][key] = replacement
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        host_path = root / "host-candidate.json"
        host = json.loads(host_path.read_text(encoding="utf-8"))
        host["qualification_resources"][receipt_path.name] = sha256(receipt_path)
        host_path.write_text(json.dumps(host), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, message):
          verify(root, qualification, commit, False)

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
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      del value["customer_install"]["wheel_failure"]
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "native success, wheel failure, and retry"):
        verify(root, qualification, commit, True)

  def test_release_gate_requires_customer_process_activation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      del value["customer_install"]["process_activation"]
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "standard-user process activation"):
        verify(root, qualification, commit, True)

  def test_release_gate_requires_non_administrator_process_activation(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["customer_install"]["standard_user"]["is_administrator"] = True
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "standard-user process activation"):
        verify(root, qualification, commit, True)
      value["customer_install"]["standard_user"]["is_administrator"] = False
      value["customer_install"]["standard_user"]["process_is_administrator"] = True
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "standard-user process activation"):
        verify(root, qualification, commit, True)

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
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = self.enable_genai_qualification(qualification)
      value["customer_install"]["standard_user"][
          "real_genai_executable_sha256"] = "0" * 64
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "different executable"):
        verify(root, qualification, commit, False, require_genai=True)

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

  def test_release_gate_rejects_changed_windows_host_baseline(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["host_state_final"]["platform"]["service"]["configuration"]["Start"] = 3
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "Windows host baseline"):
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
      value = self.enable_genai_qualification(qualification)
      value["tests"]["cpp"] = [
          name for name in value["tests"]["cpp"] if not name.startswith("test_genai_")]
      value["tests"]["python"] = [
          name for name in value["tests"]["python"] if not name.startswith("test_genai_")]
      value.pop("genai_admission")
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "complete Windows GenAI matrix"):
        verify(root, qualification, commit, True, True)

  def test_release_gate_requires_genai_failure_recovery_and_cleanup_matrix(self):
    cpp_tests = {
        "test_genai_llm_run", "test_genai_vlm_run", "test_genai_asr_run",
    }
    python_tests = {
        "test_genai_asr_file_pcm_translation_and_stream",
        "test_genai_bounded_cancellation_soak_and_context_cleanup",
        "test_genai_competing_sessions_are_isolated",
        "test_genai_llm_run_stream_history_and_cancel",
        "test_genai_request_timeout_and_restart_after_failure",
        "test_genai_resource_cleanup",
        "test_genai_restart_after_service_loss",
        "test_genai_restart_after_worker_loss",
        "test_genai_service_loss_recovery",
        "test_genai_vlm_run_and_stream",
        "test_genai_worker_loss_recovery",
    }
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

      for missing in sorted(python_tests):
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

  def test_release_gate_requires_a_stable_card_network_adapter(self):
    with tempfile.TemporaryDirectory() as directory:
      root = Path(directory)
      commit, _, qualification = self.windows_candidate(root)
      value = json.loads(qualification.read_text(encoding="utf-8"))
      value["card_network_final"]["interface_guid"] = (
          "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee")
      qualification.write_text(json.dumps(value), encoding="utf-8")
      with self.assertRaisesRegex(ValueError, "stable Windows adapter"):
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
      captured = "owner=" + "a" * 32 + "\n"
      with mock.patch(
          "qualify_windows_hardware.run_windows_openssh",
          return_value=subprocess.CompletedProcess([], 0, captured)) as run:
        completed = run_captured_to_file(["ssh.exe", "host", "true"], log, timeout=30)
      self.assertEqual(completed.returncode, 0)
      self.assertIn("owner=" + "a" * 32, completed.stdout)
      self.assertNotIn("ssh.exe", completed.stdout)
      run.assert_called_once_with(["ssh.exe", "host", "true"], log, timeout=30)

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
