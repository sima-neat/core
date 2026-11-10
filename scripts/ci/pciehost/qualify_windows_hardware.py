#!/usr/bin/env python3
"""Qualify one exact Windows PCIe-host cohort on a directly attached card.

The script is intentionally local to the Windows runner. It never drives the
test process through an inbound SSH session, and it owns install, test, evidence,
and reverse-order cleanup as one transaction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile

from windows_candidate import dependency_identity, msi_version, sha256, validate_candidate


PASS_CODES = {0}
MSI_SUCCESS_CODES = {0, 1641, 3010}
BASE_CPP_TESTS = (
    "test_metadata.exe",
    "test_tensor_run.exe",
    "test_tensor_mla_only_run.exe",
    "test_image_run.exe",
    "test_image_boxdecode_run.exe",
    "test_queue_blocker.exe",
    "test_model_resilience.exe",
)
BASE_PYTHON_TESTS = (
    "test_async_test_runner_cancels_blocked_peer",
    "test_metadata_yolov8",
    "test_runtime_request_correlation",
    "test_runtime_two_models_and_independent_unload",
    "test_tensor_run_yolov8",
    "test_tensor_latency_yolov8",
    "test_tensor_throughput_yolov8",
    "test_tensor_parallel_queues_yolov8",
    "test_tensor_run_mla_only",
    "test_image_run_yolov8",
    "test_image_boxdecode_run_yolov8",
)
SUPPORTED_PYTHON = (3, 14)


def write_json(path: Path, value: object) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_windows() -> None:
  if sys.platform != "win32" or platform.machine().upper() != "AMD64":
    raise RuntimeError("Windows PCIe hardware qualification requires Windows AMD64")
  if platform.python_implementation() != "CPython" or sys.version_info[:2] != SUPPORTED_PYTHON:
    raise RuntimeError("Windows PCIe hardware qualification requires CPython 3.14 x64")


def require_fresh_directory(path: Path, description: str) -> Path:
  resolved = path.resolve()
  if resolved.exists():
    raise RuntimeError(f"{description} must be fresh: {resolved}")
  resolved.mkdir(parents=True)
  return resolved


def verify_dependency_cohort(candidate, platform_runtime: Path, gstreamer_root: Path) -> dict:
  expected = dependency_identity(candidate.root)
  platform_root = platform_runtime.resolve(strict=True)
  gstreamer = gstreamer_root.resolve(strict=True)
  pkg_config = gstreamer / "bin" / "pkg-config.exe"
  gst_inspect = gstreamer / "bin" / "gst-inspect-1.0.exe"
  gstreamer_dlls = sorted(
      (gstreamer / "bin").glob("*.dll"), key=lambda path: path.name.lower())
  if not pkg_config.is_file() or not gst_inspect.is_file():
    raise RuntimeError("qualified GStreamer root is missing pkg-config or gst-inspect")
  if not gstreamer_dlls:
    raise RuntimeError("qualified GStreamer root has no runtime DLLs")
  result = subprocess.run(
      [gst_inspect, "--version"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
      text=True, timeout=60, check=False)
  if result.returncode != 0:
    raise RuntimeError("could not read the hardware runner's GStreamer version")
  actual = {
      "gstreamer": {
          "pkg_config_sha256": sha256(pkg_config),
          "runtime_dlls": [
              {"name": path.name, "sha256": sha256(path)} for path in gstreamer_dlls
          ],
          "version_output": result.stdout.strip(),
      },
      "platform_runtime_dlls": [
          {"name": path.name, "sha256": sha256(path)}
          for path in sorted(platform_root.glob("*.dll"), key=lambda path: path.name.lower())
      ],
  }
  if actual != expected:
    raise RuntimeError(
        "hardware runner Platform/GStreamer dependencies differ from the built candidate")
  return actual


def windows_security_snapshot(log: Path) -> dict:
  script = r"""
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$os = Get-CimInstance Win32_OperatingSystem
$smartAppControl = $null
$policyPath = 'HKLM:\SYSTEM\CurrentControlSet\Control\CI\Policy'
if (Test-Path -LiteralPath $policyPath) {
  $smartAppControl = (Get-ItemProperty -LiteralPath $policyPath -ErrorAction SilentlyContinue).VerifiedAndReputablePolicyState
}
$appLocker = 'unavailable'
try {
  $appLocker = @((Get-AppLockerPolicy -Effective -ErrorAction Stop).RuleCollections).Count
} catch {}
[pscustomobject]@{
  account = $identity.Name
  architecture = $os.OSArchitecture
  build_number = $os.BuildNumber
  caption = $os.Caption
  is_administrator = $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
  product_type = $os.ProductType
  smart_app_control_state = $smartAppControl
  applocker_rule_collections = $appLocker
  version = $os.Version
} | ConvertTo-Json -Compress
"""
  result = subprocess.run(
      ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60, check=False)
  log.write_text(
      f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}",
      encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError(f"could not record Windows security state; see {log}")
  try:
    value = json.loads(result.stdout)
  except json.JSONDecodeError as error:
    raise RuntimeError(f"Windows security state was not valid JSON; see {log}") from error
  required = {
      "account", "architecture", "build_number", "caption", "is_administrator",
      "product_type", "smart_app_control_state", "applocker_rule_collections", "version",
  }
  if not isinstance(value, dict) or set(value) != required:
    raise RuntimeError(f"Windows security state was incomplete; see {log}")
  try:
    build_number = int(value["build_number"])
  except (TypeError, ValueError) as error:
    raise RuntimeError(f"Windows build number was invalid; see {log}") from error
  if value["product_type"] != 1 or build_number < 22000:
    raise RuntimeError(
        "direct-card qualification requires a Windows 11 client workstation; "
        f"see {log}")
  if "64" not in str(value["architecture"]):
    raise RuntimeError(f"direct-card qualification requires a 64-bit OS; see {log}")
  return value


class ExclusiveLock:
  def __init__(self, path: Path):
    self.path = path.resolve()
    self.fd: int | None = None
    self.retain_on_exit = False

  def __enter__(self):
    self.path.parent.mkdir(parents=True, exist_ok=True)
    try:
      self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
      raise RuntimeError(
          f"PCIe hardware lock already exists: {self.path}; verify the card is idle before clearing it"
      ) from error
    os.write(self.fd, f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}\n".encode())
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if self.fd is not None:
      os.close(self.fd)
      self.fd = None
    if not self.retain_on_exit:
      self.path.unlink(missing_ok=True)

  def retain(self) -> None:
    self.retain_on_exit = True


def run_logged(command: list[object], log: Path, *, env: dict[str, str] | None = None,
               cwd: Path | None = None, timeout: int = 3600) -> None:
  argv = [str(value) for value in command]
  with log.open("w", encoding="utf-8", errors="replace") as output:
    output.write("+ " + subprocess.list2cmdline(argv) + "\n")
    output.flush()
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
  if result.returncode not in PASS_CODES:
    raise RuntimeError(f"command failed with {result.returncode}; see {log}")


def run_expected_failure(command: list[object], log: Path,
                         *, env: dict[str, str] | None = None,
                         cwd: Path | None = None, timeout: int = 3600) -> int:
  argv = [str(value) for value in command]
  with log.open("w", encoding="utf-8", errors="replace") as output:
    output.write("+ " + subprocess.list2cmdline(argv) + "\n")
    output.flush()
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
  if result.returncode in PASS_CODES:
    raise RuntimeError(f"command unexpectedly succeeded; see {log}")
  return result.returncode


def remove_msi(msi: Path, log: Path) -> str:
  result = subprocess.run(
      ["msiexec.exe", "/x", str(msi), "/quiet", "/norestart", "/L*v", str(log)],
      check=False,
  )
  if result.returncode == 1605:
    return "ABSENT"
  if result.returncode not in MSI_SUCCESS_CODES:
    raise RuntimeError(f"Windows Installer /x failed with {result.returncode}; see {log}")
  if result.returncode != 0:
    raise RuntimeError(f"Windows Installer /x requested a restart ({result.returncode})")
  return "PASS"


def repair_owned_file(msi: Path, installed_file: Path, log: Path) -> str:
  if not installed_file.is_file():
    raise RuntimeError(f"cannot test MSI repair; installed file is missing: {installed_file}")
  expected = sha256(installed_file)
  installed_file.unlink()
  result = subprocess.run(
      ["msiexec.exe", "/fa", str(msi), "/quiet", "/norestart", "/L*v", str(log)],
      check=False,
  )
  if result.returncode not in MSI_SUCCESS_CODES:
    raise RuntimeError(f"Windows Installer /fa failed with {result.returncode}; see {log}")
  if result.returncode != 0:
    raise RuntimeError(f"Windows Installer /fa requested a restart ({result.returncode})")
  if not installed_file.is_file() or sha256(installed_file) != expected:
    raise RuntimeError(f"Windows Installer did not restore its owned file: {installed_file}")
  return "PASS"


def sima_cli_wheel(directory: Path) -> Path:
  wheels = sorted(directory.resolve(strict=True).glob("sima_cli-*.whl"))
  if len(wheels) != 1:
    raise ValueError(f"expected exactly one sima-cli wheel, found {len(wheels)}")
  return wheels[0]


def sima_cli_identity(wheel: Path, expected_commit: str) -> dict[str, str]:
  packages = []
  with zipfile.ZipFile(wheel) as archive:
    bad = archive.testzip()
    if bad:
      raise ValueError(f"sima-cli wheel member failed CRC validation: {bad}")
    packages = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
    if len(packages) != 1:
      raise ValueError("sima-cli wheel has no unique distribution metadata")
    metadata = archive.read(packages[0]).decode("utf-8", errors="strict")
  version = next(
      (line.removeprefix("Version: ").strip()
       for line in metadata.splitlines() if line.startswith("Version: ")), "")
  if not version:
    raise ValueError("sima-cli wheel metadata has no version")
  if expected_commit[:12] not in version:
    raise ValueError(
        f"sima-cli wheel version {version!r} is not tied to expected commit {expected_commit}")
  return {"commit": expected_commit, "sha256": sha256(wheel), "version": version}


def stage_local_registry(candidate, root: Path) -> Path:
  package = root / "core" / "3.0" / candidate.core_commit / "pciehost" / "windows" / "amd64"
  package.mkdir(parents=True)
  shutil.copy2(candidate.root / "metadata.json", package / "metadata.json")
  for name in candidate.resources:
    shutil.copy2(candidate.root / name, package / name)
  latest = root / "core" / "3.0" / "latest.tag"
  latest.parent.mkdir(parents=True, exist_ok=True)
  latest.write_text(candidate.core_commit + "\n", encoding="ascii")
  return package


def stage_invalid_wheel_registry(candidate, root: Path) -> Path:
  """Create a checksum-consistent fixture whose wheel pip must reject."""
  package = stage_local_registry(candidate, root)
  wheel = package / candidate.wheel.name
  wheel.write_bytes(b"invalid wheel fixture for partial-install recovery\n")
  metadata_path = package / "metadata.json"
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  metadata["resources-checksum"][candidate.wheel.name] = sha256(wheel)
  write_json(metadata_path, metadata)
  return package


def load_upgrade_base(candidate, directory: Path | None, expected_commit: str):
  if directory is None:
    return None
  upgrade_base = validate_candidate(directory)
  if upgrade_base.core_commit != expected_commit:
    raise ValueError(
        "upgrade baseline commit does not match --expected-upgrade-base-commit")
  if upgrade_base.core_commit == candidate.core_commit:
    raise ValueError("upgrade baseline and candidate name the same Core commit")
  upgrade_base_version = str(upgrade_base.metadata.get("version", ""))
  candidate_version = str(candidate.metadata.get("version", ""))
  if not upgrade_base_version or not candidate_version:
    raise ValueError("upgrade baseline and candidate must name package versions")
  if msi_version(candidate_version) <= msi_version(upgrade_base_version):
    raise ValueError("upgrade candidate package version must be newer than the baseline")
  if upgrade_base.resources == candidate.resources:
    raise ValueError("upgrade baseline and candidate contain identical package resources")
  return upgrade_base


@contextmanager
def local_registry(root: Path):
  handler = partial(SimpleHTTPRequestHandler, directory=str(root))
  server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
  thread = threading.Thread(target=server.serve_forever, daemon=True)
  thread.start()
  try:
    yield f"http://127.0.0.1:{server.server_port}"
  finally:
    server.shutdown()
    server.server_close()
    thread.join(timeout=10)


def _validate_remote_path(path: str) -> str:
  if not re.fullmatch(r"/[A-Za-z0-9._/+:-]+", path):
    raise ValueError(f"card runtime path is not a safe absolute POSIX path: {path!r}")
  return path


def ssh_command(args: argparse.Namespace, remote: str) -> list[str]:
  return [
      "ssh.exe",
      "-i", str(args.ssh_key.resolve(strict=True)),
      "-o", "BatchMode=yes",
      "-o", "IdentitiesOnly=yes",
      "-o", "StrictHostKeyChecking=yes",
      "-o", "ConnectTimeout=10",
      f"{args.card_user}@{args.card_host}",
      remote,
  ]


def card_snapshot(args: argparse.Namespace, evidence: Path, label: str) -> dict[str, str]:
  runtime_path = _validate_remote_path(args.card_runtime_path)
  worker_path = _validate_remote_path(args.card_worker_path)
  genai_worker_path = _validate_remote_path(args.card_genai_worker_path)
  remote = (
      "set -eu; "
      "printf 'machine='; uname -m; "
      "printf 'platform='; (cat /etc/buildinfo 2>/dev/null || cat /etc/os-release); "
      f"printf 'runtime_sha256='; sha256sum -- {runtime_path} | awk '{{print $1}}'; "
      f"printf 'worker_sha256='; sha256sum -- {worker_path} | awk '{{print $1}}'; "
      + (f"printf 'genai_worker_sha256='; sha256sum -- {genai_worker_path} | awk '{{print $1}}'; "
         if args.include_genai else "") +
      "printf 'firmware='; cat /sys/class/remoteproc/remoteproc1/firmware; "
      "printf 'remoteproc_state='; cat /sys/class/remoteproc/remoteproc1/state; "
      "printf 'workers='; ps -eo args= | awk '$1 ~ /(^|\\/)pcie-pipeline-builder$/ {n++} END {print n+0}'; "
      "printf 'genai_workers='; ps -eo args= | awk '$1 ~ /(^|\\/)neat-pcie-genai-worker$/ {n++} END {print n+0}'"
  )
  log = evidence / f"card-{label}.log"
  run_logged(ssh_command(args, remote), log, timeout=30)
  values: dict[str, str] = {}
  for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
    if "=" in line and not line.startswith("+"):
      name, value = line.split("=", 1)
      if name in {"machine", "runtime_sha256", "worker_sha256", "genai_worker_sha256",
                  "firmware", "remoteproc_state", "workers", "genai_workers"}:
        values[name] = value.strip()
  required = {"machine", "runtime_sha256", "worker_sha256", "firmware",
              "remoteproc_state", "workers", "genai_workers"}
  if args.include_genai:
    required.add("genai_worker_sha256")
  if set(values) != required:
    raise RuntimeError(f"card {label} snapshot is incomplete; see {log}")
  if values["machine"] != "aarch64":
    raise RuntimeError(f"card has unexpected architecture: {values['machine']}")
  if values["runtime_sha256"].lower() != args.expected_card_runtime_sha256.lower():
    raise RuntimeError("card runtime checksum differs from the qualified baseline")
  if values["worker_sha256"].lower() != args.expected_card_worker_sha256.lower():
    raise RuntimeError("card tensor worker checksum differs from the qualified baseline")
  if (args.include_genai and values["genai_worker_sha256"].lower()
      != args.expected_card_genai_worker_sha256.lower()):
    raise RuntimeError("card GenAI worker checksum differs from the qualified baseline")
  if values["remoteproc_state"] != "running":
    raise RuntimeError(f"card remote processor is not running: {values['remoteproc_state']}")
  return values


def stop_card_workers(args: argparse.Namespace, evidence: Path) -> None:
  remote = (
      "pkill -TERM -f '(^|/)[p]cie-pipeline-builder([[:space:]]|$)' 2>/dev/null || true; "
      "pkill -TERM -f '(^|/)[n]eat-pcie-genai-worker([[:space:]]|$)' 2>/dev/null || true; "
      "sleep 2; "
      "pkill -KILL -f '(^|/)[p]cie-pipeline-builder([[:space:]]|$)' 2>/dev/null || true; "
      "pkill -KILL -f '(^|/)[n]eat-pcie-genai-worker([[:space:]]|$)' 2>/dev/null || true"
  )
  run_logged(ssh_command(args, remote), evidence / "card-worker-cleanup.log", timeout=30)


def wait_for_no_card_workers(args: argparse.Namespace, log: Path, timeout: int) -> None:
  remote = (
      "printf 'workers='; "
      "pgrep -fc '(^|/)[p]cie-pipeline-builder([[:space:]]|$)' || true"
  )
  deadline = time.monotonic() + timeout
  with log.open("w", encoding="utf-8", errors="replace") as output:
    while True:
      result = subprocess.run(
          ssh_command(args, remote), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
          text=True, timeout=30, check=False)
      output.write(f"returncode={result.returncode} {result.stdout.strip()}\n")
      output.flush()
      match = re.search(r"workers=(\d+)", result.stdout)
      if result.returncode == 0 and match and int(match.group(1)) == 0:
        return
      if time.monotonic() >= deadline:
        raise RuntimeError(f"card worker did not stop within {timeout} seconds; see {log}")
      time.sleep(1)


def wait_for_log(process: subprocess.Popen, log: Path, marker: str, timeout: int) -> None:
  deadline = time.monotonic() + timeout
  while True:
    if log.is_file():
      text = log.read_text(encoding="utf-8", errors="replace")
      if marker in text:
        return
    returncode = process.poll()
    if returncode is not None:
      raise RuntimeError(
          f"fault-injection client exited with {returncode} before {marker!r}; see {log}")
    if time.monotonic() >= deadline:
      raise RuntimeError(f"fault-injection client did not reach {marker!r}; see {log}")
    time.sleep(0.2)


def run_fault_recovery_tests(args: argparse.Namespace, test_root: Path, evidence: Path,
                             environment: dict[str, str]) -> list[str]:
  blocker = test_root / "test_queue_blocker.exe"
  recovery = test_root / "test_tensor_run.exe"
  completed = []

  def launch(label: str) -> tuple[subprocess.Popen, object, Path]:
    log = evidence / f"cpp-{label}.log"
    stream = log.open("w", encoding="utf-8", errors="replace")
    command = [blocker, "--iterations", "0"]
    stream.write("+ " + subprocess.list2cmdline([str(value) for value in command]) + "\n")
    stream.flush()
    process = subprocess.Popen(
        [str(value) for value in command], cwd=test_root, env=environment,
        stdout=stream, stderr=subprocess.STDOUT, text=True)
    try:
      wait_for_log(process, log, "queue is now occupied", 300)
    except BaseException:
      if process.poll() is None:
        process.kill()
      process.wait(timeout=30)
      stream.close()
      raise
    return process, stream, log

  process, stream, _ = launch("test_worker_loss_recovery")
  try:
    remote = (
        "pkill -KILL -f '(^|/)[p]cie-pipeline-builder([[:space:]]|$)' "
        "2>/dev/null || true"
    )
    run_logged(
        ssh_command(args, remote), evidence / "card-worker-loss-injection.log", timeout=30)
    try:
      returncode = process.wait(timeout=120)
    except subprocess.TimeoutExpired as error:
      raise RuntimeError("client did not terminate after card worker loss") from error
    if returncode == 0:
      raise RuntimeError("client reported success after its card worker was killed")
  finally:
    if process.poll() is None:
      process.kill()
      process.wait(timeout=30)
    stream.close()
  wait_for_no_card_workers(args, evidence / "card-worker-loss-drain.log", 120)
  run_logged(
      [recovery, "--sync-iterations", "1", "--iterations", "0"],
      evidence / "cpp-test_worker_loss_queue_reuse.log", env=environment,
      cwd=test_root, timeout=args.test_timeout_seconds)
  completed.append("test_worker_loss_recovery")

  process, stream, _ = launch("test_client_process_death_recovery")
  try:
    process.kill()
    process.wait(timeout=30)
  finally:
    if process.poll() is None:
      process.kill()
      process.wait(timeout=30)
    stream.close()
  wait_for_no_card_workers(args, evidence / "client-process-death-drain.log", 120)
  run_logged(
      [recovery, "--sync-iterations", "1", "--iterations", "0"],
      evidence / "cpp-test_client_process_death_queue_reuse.log", env=environment,
      cwd=test_root, timeout=args.test_timeout_seconds)
  completed.append("test_client_process_death_recovery")
  return completed


def require_assets(environment: dict[str, str], include_genai: bool) -> None:
  required_files = (
      "SIMAPCIE_YOLOV8_MODEL",
      "SIMAPCIE_MLA_ONLY_MODEL",
      "SIMAPCIE_MLA_ONLY_BF16_MODEL",
      "SIMAPCIE_TEST_IMAGE",
      "SIMAPCIE_BOXDECODE_IMAGE",
  )
  for name in required_files:
    value = environment.get(name, "")
    if not value or not Path(value).is_file():
      raise RuntimeError(f"required Windows PCIe hardware asset is unavailable: {name}")
  if include_genai:
    for name in ("SIMAPCIE_GENAI_MODELS_PATH", "SIMAPCIE_GENAI_TEST_ASSETS"):
      value = environment.get(name, "")
      if not value or not Path(value).is_dir():
        raise RuntimeError(f"required Windows PCIe GenAI asset directory is unavailable: {name}")


def stage_path_fixtures(environment: dict[str, str], root: Path) -> None:
  """Exercise model and media inputs through an absolute Windows path with spaces."""
  root.mkdir(parents=True)
  for name in ("SIMAPCIE_YOLOV8_MODEL", "SIMAPCIE_TEST_IMAGE", "SIMAPCIE_BOXDECODE_IMAGE"):
    source = Path(environment[name]).resolve(strict=True)
    target = root / f"{name.lower()}-{source.name}"
    try:
      os.link(source, target)
    except OSError:
      shutil.copy2(source, target)
    environment[name] = str(target)


def stress_queues(environment: dict[str, str]) -> list[int]:
  values = environment.get("SIMAPCIE_STRESS_QUEUES", "0 1 2 3 4 5").split()
  if not values or any(not re.fullmatch(r"[0-5]", value) for value in values):
    raise ValueError("SIMAPCIE_STRESS_QUEUES must contain queue numbers in the range 0..5")
  queues = [int(value) for value in values]
  if len(set(queues)) != len(queues):
    raise ValueError("SIMAPCIE_STRESS_QUEUES must not contain duplicate queues")
  return queues


def run_cpp_parallel_queues(executable: Path, evidence: Path,
                            environment: dict[str, str], timeout: int) -> None:
  processes: list[tuple[int, subprocess.Popen, object]] = []
  deadline = time.monotonic() + timeout
  iterations = environment.get("SIMAPCIE_STRESS_ITERATIONS", "1000")
  if not re.fullmatch(r"[1-9][0-9]*", iterations):
    raise ValueError("SIMAPCIE_STRESS_ITERATIONS must be a positive integer")
  try:
    for queue in stress_queues(environment):
      log = (evidence / f"cpp-test_tensor_parallel_queue_{queue}.log").open(
          "w", encoding="utf-8", errors="replace")
      command = [str(executable), "--model", environment["SIMAPCIE_YOLOV8_MODEL"],
                 "--queue", str(queue), "--iterations", iterations]
      log.write("+ " + subprocess.list2cmdline(command) + "\n")
      log.flush()
      process = subprocess.Popen(
          command, cwd=executable.parent, env=environment,
          stdout=log, stderr=subprocess.STDOUT, text=True)
      processes.append((queue, process, log))
    failures = []
    for queue, process, _ in processes:
      remaining = max(0.0, deadline - time.monotonic())
      try:
        returncode = process.wait(timeout=remaining)
      except subprocess.TimeoutExpired:
        failures.append(f"queue {queue} exceeded {timeout} seconds")
        break
      if returncode != 0:
        failures.append(f"queue {queue} failed with {returncode}")
    if failures:
      raise RuntimeError("C++ parallel queue qualification failed: " + "; ".join(failures))
  finally:
    for _, process, _ in processes:
      if process.poll() is None:
        process.kill()
    for _, process, log in processes:
      if process.poll() is None:
        process.wait(timeout=30)
      log.close()


def parse_performance_log(path: Path, mode: str) -> dict[str, float | int]:
  text = path.read_text(encoding="utf-8", errors="replace")
  label = f"{mode} results"
  if label not in text:
    raise RuntimeError(f"performance log has no {label!r} section: {path}")
  section = text.split(label, 1)[1]

  def value(name: str) -> float:
    match = re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}=([0-9]+(?:[.][0-9]+)?)", section)
    if not match:
      raise RuntimeError(f"performance log has no {name!r} metric: {path}")
    result = float(match.group(1))
    if not math.isfinite(result) or result <= 0.0:
      raise RuntimeError(f"performance metric {name!r} is not positive and finite: {path}")
    return result

  if mode == "latency":
    return {
        "count": int(value("count")),
        "mean_ms": value("mean_ms"),
        "effective_fps": value("effective_fps"),
    }
  if mode == "throughput":
    return {
        "frames": int(value("frames")),
        "total_ms": value("total_ms"),
        "fps": value("fps"),
        "input_mib_per_s": value("input_mib_per_s"),
        "output_mib_per_s": value("output_mib_per_s"),
    }
  raise ValueError(f"unsupported performance mode: {mode}")


def run_cpp_tests(args: argparse.Namespace, candidate_root: Path, evidence: Path,
                  environment: dict[str, str]) -> list[str]:
  test_root = candidate_root / "hardware-tests" / "lib" / "sima-pcie-host" / "tests"
  if not test_root.is_dir():
    raise RuntimeError(f"candidate contains no C++ hardware extras: {test_root}")
  completed = []
  for name in BASE_CPP_TESTS:
    executable = test_root / name
    if not executable.is_file():
      raise RuntimeError(f"candidate is missing C++ hardware test: {name}")
    command: list[object] = [executable]
    if name == "test_tensor_mla_only_run.exe":
      command.extend(["--model", environment["SIMAPCIE_MLA_ONLY_MODEL"]])
    elif name == "test_image_boxdecode_run.exe":
      command.append("--require-person")
    elif name == "test_queue_blocker.exe":
      command.extend(["--iterations", "10", "--interval-ms", "0"])
    run_logged(command, evidence / f"cpp-{executable.stem}.log", env=environment,
               cwd=test_root, timeout=args.test_timeout_seconds)
    completed.append(executable.stem)

  completed.extend(run_fault_recovery_tests(
      args, test_root, evidence, environment))

  bf16 = test_root / "test_tensor_mla_only_run.exe"
  run_logged(
      [bf16, "--model", environment["SIMAPCIE_MLA_ONLY_BF16_MODEL"]],
      evidence / "cpp-test_tensor_mla_only_bf16_run.log",
      env=environment,
      cwd=test_root,
      timeout=args.test_timeout_seconds,
  )
  completed.append("test_tensor_mla_only_bf16_run")

  parallel = test_root / "test_tensor_run.exe"
  run_cpp_parallel_queues(
      parallel, evidence, environment, args.test_timeout_seconds)
  completed.append("test_tensor_parallel_queues")

  performance = test_root / "test_tensor_perf.exe"
  for mode in ("latency", "throughput"):
    run_logged(
        [performance, "--mode", mode], evidence / f"cpp-test_tensor_{mode}.log",
        env=environment, cwd=test_root, timeout=args.test_timeout_seconds)
    completed.append(f"test_tensor_{mode}")

  if args.include_genai:
    executable = test_root / "test_genai_run.exe"
    assets = Path(environment["SIMAPCIE_GENAI_TEST_ASSETS"])
    for kind in ("llm", "vlm", "asr"):
      run_logged(
          [executable, kind, assets], evidence / f"cpp-test_genai_{kind}_run.log",
          env=environment, cwd=test_root, timeout=args.test_timeout_seconds)
      completed.append(f"test_genai_{kind}_run")
  return completed


def run_python_tests(args: argparse.Namespace, python: Path, evidence: Path,
                     environment: dict[str, str]) -> list[str]:
  tests = [
      f"pcie_host/python/tests/test_hardware_smoke.py::{name}"
      for name in BASE_PYTHON_TESTS
  ]
  run_logged(
      [python, "-m", "pytest", "-vv", "--junitxml", evidence / "python-hardware.xml", *tests],
      evidence / "python-hardware.log",
      env=environment,
      cwd=args.source_root,
      timeout=args.test_timeout_seconds,
  )
  completed = list(BASE_PYTHON_TESTS)
  if args.include_genai:
    run_logged(
        [python, "-m", "pytest", "-vv", "--junitxml", evidence / "python-genai.xml",
         "pcie_host/python/tests/test_genai_hardware.py"],
        evidence / "python-genai.log",
        env=environment,
        cwd=args.source_root,
        timeout=args.test_timeout_seconds,
    )
    completed.extend(("test_genai_llm_run_stream_history_and_cancel",
                      "test_genai_vlm_run_and_stream",
                      "test_genai_asr_file_pcm_translation_and_stream"))
  return completed


def qualify(args: argparse.Namespace) -> None:
  require_windows()
  candidate = validate_candidate(args.candidate)
  upgrade_base = load_upgrade_base(
      candidate, args.upgrade_base_candidate, args.expected_upgrade_base_commit)
  evidence = require_fresh_directory(args.evidence, "evidence directory")
  work = require_fresh_directory(args.work, "hardware work directory")
  wheelhouse = args.python_wheelhouse.resolve(strict=True)
  if not wheelhouse.is_dir():
    raise RuntimeError(f"Python wheelhouse is not a directory: {wheelhouse}")
  cli_wheel = sima_cli_wheel(args.sima_cli_candidate)
  cli_identity = sima_cli_identity(cli_wheel, args.expected_sima_cli_commit)
  dependency_cohort = verify_dependency_cohort(
      candidate, args.platform_runtime, args.gstreamer_root)

  install_root = Path(os.environ["ProgramFiles"]) / "SimaPCIeHost"
  if install_root.exists():
    raise RuntimeError(f"refusing to modify a pre-existing installation: {install_root}")

  result = {
      "candidate": {
          "core_commit": candidate.core_commit,
          "resources": candidate.resources,
          "scheduler_commit": candidate.receipt.get("scheduler_source_commit"),
      },
      "sima_cli": cli_identity,
      "cleanup": {},
      "completed_at_utc": None,
      "result": "FAIL",
      "started_at_utc": datetime.now(timezone.utc).isoformat(),
      "host": {
          "cpu_count": os.cpu_count(),
          "machine": platform.machine(),
          "platform": platform.platform(),
          "python": platform.python_version(),
      },
      "dependencies": dependency_cohort,
      "tests": {"cpp": [], "python": []},
  }
  venv = work / "venv"
  cli_venv = work / "sima-cli-venv"
  failure: BaseException | None = None
  cleanup_errors: list[str] = []

  result["host"]["windows_security"] = windows_security_snapshot(
      evidence / "windows-security-state.log")

  with ExclusiveLock(args.lock_file) as card_lock:
    try:
      baseline = card_snapshot(args, evidence, "baseline")
      result["card_baseline"] = baseline
      if baseline["workers"] != "0" or baseline["genai_workers"] != "0":
        raise RuntimeError("card had PCIe workers before qualification; refusing to disturb them")

      subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
      python = venv / "Scripts" / "python.exe"
      subprocess.run([sys.executable, "-m", "venv", str(cli_venv)], check=True)
      cli_python = cli_venv / "Scripts" / "python.exe"
      cli = cli_venv / "Scripts" / "sima-cli.exe"
      requirements = ["pytest", "numpy", "pillow", *args.test_requirement]
      run_logged(
          [python, "-m", "pip", "install", "--no-index", "--find-links", wheelhouse,
           *requirements],
          evidence / "test-dependencies-install.log",
          timeout=900,
      )
      run_logged(
          [cli_python, "-m", "pip", "install", "--no-index", "--find-links", wheelhouse,
           cli_wheel],
          evidence / "sima-cli-install.log",
          timeout=900,
      )
      if not cli.is_file():
        raise RuntimeError(f"exact sima-cli candidate installed no console entry point: {cli}")

      cli_environment = os.environ.copy()
      cli_environment.pop("PYTHONPATH", None)
      cli_environment["PIP_NO_INDEX"] = "1"
      cli_environment["PIP_FIND_LINKS"] = str(wheelhouse)
      native_download_root = work / "native customer downloads with spaces"
      target = "core/pciehost/windows/amd64@3.0"
      if upgrade_base is not None:
        upgrade_registry_root = work / "upgrade baseline registry with spaces"
        stage_local_registry(upgrade_base, upgrade_registry_root)
        with local_registry(upgrade_registry_root) as registry_url:
          upgrade_base_command = [
              cli, "neat", "--base-url", registry_url, "install", target,
              "--install-dir", native_download_root,
          ]
          run_logged(
              upgrade_base_command, evidence / "sima-cli-upgrade-baseline-install.log",
              env=cli_environment, cwd=work, timeout=1800)
        if not install_root.is_dir():
          raise RuntimeError("sima-cli installed no native upgrade baseline")

      native_registry_root = work / "native only registry with spaces"
      stage_local_registry(candidate, native_registry_root)
      with local_registry(native_registry_root) as registry_url:
        native_command = [
            cli, "neat", "--base-url", registry_url, "install", target,
            "--install-dir", native_download_root,
        ]
        run_logged(
            native_command, evidence / "sima-cli-native-only-install.log",
            env=cli_environment, cwd=work, timeout=1800)
      if not install_root.is_dir():
        raise RuntimeError("sima-cli native-only flow installed no native cohort")
      if subprocess.run(
          [python, "-c", "import pyneatpcie"], stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL, check=False).returncode == 0:
        raise RuntimeError("native-only sima-cli flow unexpectedly installed pyneatpcie")

      missing_registry_root = work / "missing wheel registry with spaces"
      missing_package = stage_local_registry(candidate, missing_registry_root)
      (missing_package / candidate.wheel.name).unlink()
      missing_download_root = work / "missing wheel downloads with spaces"
      with local_registry(missing_registry_root) as registry_url:
        missing_command = [
            cli, "neat", "--base-url", registry_url, "install", target,
            "--install-dir", missing_download_root,
            "--python-executable", python,
        ]
        missing_code = run_expected_failure(
            missing_command, evidence / "sima-cli-missing-wheel.log",
            env=cli_environment, cwd=work, timeout=1800)
      if not install_root.is_dir():
        raise RuntimeError("missing-wheel failure removed the successful native installation")
      if subprocess.run(
          [python, "-c", "import pyneatpcie"], stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL, check=False).returncode == 0:
        raise RuntimeError("missing-wheel flow reported failure after installing pyneatpcie")

      invalid_wheel_registry_root = work / "invalid wheel registry with spaces"
      stage_invalid_wheel_registry(candidate, invalid_wheel_registry_root)
      invalid_wheel_download_root = work / "invalid wheel downloads with spaces"
      with local_registry(invalid_wheel_registry_root) as registry_url:
        invalid_wheel_command = [
            cli, "neat", "--base-url", registry_url, "install", target,
            "--install-dir", invalid_wheel_download_root,
            "--python-executable", python,
        ]
        invalid_wheel_code = run_expected_failure(
            invalid_wheel_command, evidence / "sima-cli-invalid-wheel.log",
            env=cli_environment, cwd=work, timeout=1800)
      if not install_root.is_dir():
        raise RuntimeError("wheel-install failure removed the successful native installation")
      if subprocess.run(
          [python, "-c", "import pyneatpcie"], stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL, check=False).returncode == 0:
        raise RuntimeError("invalid-wheel flow reported failure after installing pyneatpcie")

      registry_root = work / "complete registry with spaces"
      stage_local_registry(candidate, registry_root)
      recovery_download_root = work / "recovery downloads with spaces"
      with local_registry(registry_root) as registry_url:
        command = [
            cli, "neat", "--base-url", registry_url, "install", target,
            "--install-dir", recovery_download_root,
            "--python-executable", python,
        ]
        run_logged(
            command, evidence / "sima-cli-customer-recovery-install.log",
            env=cli_environment, cwd=work, timeout=1800)
        run_logged(
            command, evidence / "sima-cli-customer-repeat-install.log",
            env=cli_environment, cwd=work, timeout=1800)
      result["customer_install"] = {
          "command": target,
          "download_path_with_spaces": str(recovery_download_root),
          "missing_wheel": {"exit_code": missing_code, "status": "PASS"},
          "native_only": "PASS",
          "recovery": "PASS",
          "repeat": "PASS",
          "wheel_failure": {"exit_code": invalid_wheel_code, "status": "PASS"},
      }
      if upgrade_base is not None:
        result["customer_install"]["upgrade"] = {
            "baseline_commit": upgrade_base.core_commit,
            "baseline_version": str(upgrade_base.metadata["version"]),
            "candidate_version": str(candidate.metadata["version"]),
            "status": "PASS",
        }

      result["customer_install"]["repair"] = {
          "core": repair_owned_file(
              candidate.core_msi,
              install_root / "include" / "simaai" / "neat" / "pcie" / "Model.h",
              evidence / "core-msi-repair.log"),
          "scheduler": repair_owned_file(
              candidate.scheduler_msi,
              install_root / "bin" / "gstreamer-1.0" / "gstneatpciehost.dll",
              evidence / "scheduler-msi-repair.log"),
      }

      environment = os.environ.copy()
      environment.pop("PYTHONPATH", None)
      environment.update({
          "SIMA_PCIE_HOST_LIBRARY_PATH": str(install_root / "bin"),
          "SIMAPCIE_CARD_HOST": args.card_host,
          "SIMAPCIE_USER": args.card_user,
          "SIMAPCIE_SSH_KEY": str(args.ssh_key.resolve()),
      })
      environment["PATH"] = os.pathsep.join((
          str(install_root / "bin"),
          str(args.platform_runtime.resolve(strict=True)),
          str(args.gstreamer_root.resolve(strict=True) / "bin"),
          *(str(path.resolve(strict=True) / "bin") for path in args.dependency_prefix),
          environment["PATH"],
      ))
      run_logged(
          [args.gstreamer_root.resolve(strict=True) / "bin" / "gst-inspect-1.0.exe",
           "neatpciehost"],
          evidence / "gst-inspect-neatpciehost.log", env=environment, cwd=work, timeout=60)
      result["customer_install"]["plugin_discovery"] = "PASS"
      require_assets(environment, args.include_genai)
      stage_path_fixtures(environment, work / "customer path fixtures with spaces")
      result["path_fixtures"] = {
          name: environment[name]
          for name in ("SIMAPCIE_YOLOV8_MODEL", "SIMAPCIE_TEST_IMAGE",
                       "SIMAPCIE_BOXDECODE_IMAGE")
      }
      run_logged(
          [python, "-c", "import pyneatpcie as p; assert p.Model and p.Runtime and p.Tensor"],
          evidence / "python-import.log", env=environment, cwd=work, timeout=60)

      result["tests"]["cpp"] = run_cpp_tests(args, candidate.root, evidence, environment)
      result["tests"]["python"] = run_python_tests(args, python, evidence, environment)
      result["performance"] = {
          mode: parse_performance_log(evidence / f"cpp-test_tensor_{mode}.log", mode)
          for mode in ("latency", "throughput")
      }
      result["result"] = "PASS"
    except BaseException as error:
      failure = error
    finally:
      try:
        stop_card_workers(args, evidence)
      except BaseException as error:
        cleanup_errors.append(str(error))
      customer_python = venv / "Scripts" / "python.exe"
      if customer_python.is_file():
        try:
          run_logged([customer_python, "-m", "pip", "uninstall", "-y", "pyneatpcie"],
                     evidence / "pip-uninstall.log", timeout=300)
          result["cleanup"]["python"] = "PASS"
        except BaseException as error:
          cleanup_errors.append(str(error))
      try:
        result["cleanup"]["core_msi"] = remove_msi(
            candidate.core_msi, evidence / "core-msi-uninstall.log")
      except BaseException as error:
        cleanup_errors.append(str(error))
      try:
        result["cleanup"]["scheduler_msi"] = remove_msi(
            candidate.scheduler_msi, evidence / "scheduler-msi-uninstall.log")
      except BaseException as error:
        cleanup_errors.append(str(error))
      if install_root.exists() and upgrade_base is not None:
        for owner, msi in (("core", upgrade_base.core_msi),
                           ("scheduler", upgrade_base.scheduler_msi)):
          try:
            result["cleanup"][f"upgrade_base_{owner}_msi"] = remove_msi(
                msi, evidence / f"upgrade-base-{owner}-msi-uninstall.log")
          except BaseException as error:
            cleanup_errors.append(str(error))
      shutil.rmtree(venv, ignore_errors=True)
      if install_root.exists():
        cleanup_errors.append(f"cohort left files after removal: {install_root}")
      try:
        final = card_snapshot(args, evidence, "final")
        result["card_final"] = final
        baseline = result.get("card_baseline", {})
        unchanged = ("runtime_sha256", "worker_sha256", "genai_worker_sha256",
                     "firmware", "remoteproc_state")
        if (final["workers"] != "0" or final["genai_workers"] != "0"
            or any(name in baseline and final.get(name) != baseline[name] for name in unchanged)):
          cleanup_errors.append("card did not return to its qualified baseline")
      except BaseException as error:
        cleanup_errors.append(str(error))
      result["cleanup"]["errors"] = cleanup_errors
      result["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
      if cleanup_errors:
        result["result"] = "FAIL"
        card_lock.retain()
      write_json(evidence / "hardware-qualification.json", result)

  shutil.rmtree(work, ignore_errors=True)
  if cleanup_errors:
    raise RuntimeError("hardware cleanup failed: " + "; ".join(cleanup_errors))
  if failure is not None:
    raise failure


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--card-host", required=True)
  parser.add_argument("--card-runtime-path", default="/lib/firmware/modalix-cvu-fw")
  parser.add_argument("--card-worker-path", default="/usr/bin/pcie-pipeline-builder")
  parser.add_argument("--card-genai-worker-path", default="/usr/bin/neat-pcie-genai-worker")
  parser.add_argument("--card-user", default="sima")
  parser.add_argument("--dependency-prefix", action="append", default=[], type=Path)
  parser.add_argument("--evidence", type=Path, required=True)
  parser.add_argument("--expected-card-runtime-sha256", required=True)
  parser.add_argument("--expected-card-worker-sha256", required=True)
  parser.add_argument("--expected-card-genai-worker-sha256", default="")
  parser.add_argument("--expected-sima-cli-commit", required=True)
  parser.add_argument("--gstreamer-root", type=Path, required=True)
  parser.add_argument("--include-genai", action="store_true")
  parser.add_argument("--lock-file", type=Path, required=True)
  parser.add_argument("--platform-runtime", type=Path, required=True)
  parser.add_argument("--python-wheelhouse", type=Path, required=True)
  parser.add_argument("--sima-cli-candidate", type=Path, required=True)
  parser.add_argument("--source-root", type=Path, default=Path.cwd())
  parser.add_argument("--ssh-key", type=Path, required=True)
  parser.add_argument("--test-requirement", action="append", default=[])
  parser.add_argument("--test-timeout-seconds", type=int, default=7200)
  parser.add_argument("--upgrade-base-candidate", type=Path)
  parser.add_argument("--expected-upgrade-base-commit", default="")
  parser.add_argument("--work", type=Path, required=True)
  args = parser.parse_args()
  for name in ("expected_card_runtime_sha256", "expected_card_worker_sha256"):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", getattr(args, name)):
      parser.error(f"--{name.replace('_', '-')} must be a SHA-256 digest")
  if not re.fullmatch(r"[0-9a-f]{40}", args.expected_sima_cli_commit):
    parser.error("--expected-sima-cli-commit must be a full lowercase Git commit")
  if (args.upgrade_base_candidate is None) != (not args.expected_upgrade_base_commit):
    parser.error(
        "--upgrade-base-candidate and --expected-upgrade-base-commit must be supplied together")
  if args.upgrade_base_candidate is not None and not re.fullmatch(
      r"[0-9a-f]{40}", args.expected_upgrade_base_commit):
    parser.error("--expected-upgrade-base-commit must be a full lowercase Git commit")
  if args.include_genai and not re.fullmatch(
      r"[0-9a-fA-F]{64}", args.expected_card_genai_worker_sha256):
    parser.error("--expected-card-genai-worker-sha256 is required with --include-genai")
  qualify(args)


if __name__ == "__main__":
  main()
