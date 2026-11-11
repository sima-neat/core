#!/usr/bin/env python3
"""Qualify one exact Windows PCIe-host cohort on a directly attached card.

The script is intentionally local to the Windows runner. It never drives the
test process through an inbound SSH session, and it owns install, test, evidence,
and reverse-order cleanup as one transaction.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import secrets
import shlex
import shutil
import subprocess
import sys
import threading
import time
import zipfile

from windows_candidate import (
    dependency_identity, msi_version, require_junit_cases, sha256, validate_candidate)


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


@dataclass(frozen=True)
class CardCandidate:
  root: Path
  metadata: dict
  packages: tuple[Path, ...]
  resources: dict[str, str]


def validate_card_candidate(directory: Path, expected_commit: str) -> CardCandidate:
  """Validate the exact Linux Core payload that will run privately on the card."""
  root = directory.resolve(strict=True)
  metadata_path = root / "metadata-all.json"
  try:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"card candidate has no valid metadata-all.json: {root}") from error
  if not isinstance(metadata, dict):
    raise ValueError("card candidate metadata must be an object")
  if expected_commit[:12] not in str(metadata.get("version", "")):
    raise ValueError("card candidate version is not tied to the selected Core commit")
  resources = metadata.get("resources-checksum")
  names = metadata.get("resources")
  if (not isinstance(resources, dict) or not resources
      or not isinstance(names, list) or set(names) != set(resources)):
    raise ValueError("card candidate resource inventory is incomplete")
  packages = []
  for name, expected in resources.items():
    if (not isinstance(name, str) or Path(name).name != name
        or not re.fullmatch(r"[A-Za-z0-9._+%:-]+", name)):
      raise ValueError(f"card candidate resource has an unsafe name: {name!r}")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
      raise ValueError(f"card candidate resource has an invalid checksum: {name}")
    path = root / name
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
      raise ValueError(f"card candidate resource checksum differs: {name}")
    if name.endswith(".deb"):
      packages.append(path)
  if not packages:
    raise ValueError("card candidate contains no Debian runtime packages")
  return CardCandidate(root, metadata, tuple(sorted(packages)), dict(resources))


def write_json(path: Path, value: object) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_windows() -> None:
  if sys.platform != "win32" or platform.machine().upper() != "AMD64":
    raise RuntimeError("Windows PCIe hardware qualification requires Windows AMD64")
  if platform.python_implementation() != "CPython" or sys.version_info[:2] != SUPPORTED_PYTHON:
    raise RuntimeError("Windows PCIe hardware qualification requires CPython 3.14 x64")


def require_powershell() -> str:
  """Use PowerShell 7 when provisioned, otherwise Windows PowerShell 5.1."""
  for name in ("pwsh.exe", "powershell.exe"):
    path = shutil.which(name)
    if path:
      return str(Path(path).resolve())
  raise RuntimeError("Windows PCIe hardware qualification requires PowerShell")


def require_fresh_directory(path: Path, description: str) -> Path:
  resolved = path.resolve()
  if resolved.exists():
    raise RuntimeError(f"{description} must be fresh: {resolved}")
  resolved.mkdir(parents=True)
  return resolved


def verify_dependency_cohort(candidate, platform_runtime: Path, gstreamer_root: Path,
                             dependency_runtime_dirs: list[Path]) -> dict:
  expected = dependency_identity(candidate.root, require_dependency_runtime=True)
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
      "dependency_runtime": [],
  }
  for value in dependency_runtime_dirs:
    directory = value.resolve(strict=True)
    dlls = sorted(directory.glob("*.dll"), key=lambda path: path.name.lower())
    if not dlls:
      raise RuntimeError(f"qualified dependency runtime directory has no DLLs: {directory}")
    actual["dependency_runtime"].append([
        {"name": path.name, "sha256": sha256(path)} for path in dlls
    ])
  if actual != expected:
    raise RuntimeError(
        "hardware runner runtime dependencies differ from the built candidate")
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
  def __init__(self, path: Path, remote_token: str = ""):
    self.path = path.resolve()
    self.remote_token = remote_token
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
    record = f"pid={os.getpid()} started={datetime.now(timezone.utc).isoformat()}"
    if self.remote_token:
      record += f" remote_token={self.remote_token}"
    os.write(self.fd, (record + "\n").encode())
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    if self.fd is not None:
      os.close(self.fd)
      self.fd = None
    if not self.retain_on_exit:
      self.path.unlink(missing_ok=True)

  def retain(self) -> None:
    self.retain_on_exit = True

  def release_on_exit(self) -> None:
    self.retain_on_exit = False


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
        stdin=subprocess.DEVNULL,
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
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
  if result.returncode in PASS_CODES:
    raise RuntimeError(f"command unexpectedly succeeded; see {log}")
  return result.returncode


def run_captured_to_file(command: list[object], log: Path, *,
                         timeout: int) -> subprocess.CompletedProcess[str]:
  """Capture a native command without routing its output through a pipe.

  Windows OpenSSH 9.5 can receive the remote exit status and still block while
  closing a redirected pipe.  A regular file handle avoids that client bug and
  also preserves partial output when a command times out.
  """
  argv = [str(value) for value in command]
  with log.open("w", encoding="utf-8", errors="replace") as output:
    output.write("+ " + subprocess.list2cmdline(argv) + "\n")
    output.flush()
    output_offset = output.tell()
    result = subprocess.run(
        argv,
        stdin=subprocess.DEVNULL,
        stdout=output,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
        check=False,
    )
  with log.open("r", encoding="utf-8", errors="replace") as output:
    output.seek(output_offset)
    captured = output.read()
  return subprocess.CompletedProcess(argv, result.returncode, captured, None)


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


def scp_command(args: argparse.Namespace, sources: tuple[Path, ...], destination: str) -> list[str]:
  return [
      "scp.exe",
      "-i", str(args.ssh_key.resolve(strict=True)),
      "-o", "BatchMode=yes",
      "-o", "IdentitiesOnly=yes",
      "-o", "StrictHostKeyChecking=yes",
      "-o", "ConnectTimeout=10",
      *(str(path) for path in sources),
      f"{args.card_user}@{args.card_host}:{destination}",
  ]


def card_candidate_paths(token: str) -> dict[str, str]:
  if not re.fullmatch(r"[0-9a-f]{32}", token):
    raise ValueError("card candidate token must be 128-bit lowercase hex")
  stage = f"/tmp/neat-pcie-card0.{token}"
  return {
      "stage": stage,
      "packages": f"{stage}/packages",
      "root": f"{stage}/root",
      "worker": f"{stage}/root/usr/bin/pcie-pipeline-builder",
      "genai_worker": f"{stage}/root/usr/bin/neat-pcie-genai-worker",
      "firmware_source": (
          f"{stage}/root/usr/share/sima-neat-firmware/modalix-cvu-fw"),
      "firmware_name": f"neat-pcie-card0-{token}",
      "firmware_installed": f"/lib/firmware/neat-pcie-card0-{token}",
      "run": f"{stage}/run",
      "log": f"{stage}/log",
  }


def _card_sudo_password(args: argparse.Namespace) -> str:
  name = args.card_sudo_password_env
  if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
    raise ValueError("card sudo password environment variable name is invalid")
  password = os.environ.get(name, "")
  if "\n" in password or "\r" in password:
    raise ValueError("card sudo password must be one line")
  return password


def run_sudo_remote(args: argparse.Namespace, remote: str, log: Path,
                    *, timeout: int = 120) -> None:
  """Run a fixed card command as root without placing the password in argv or logs."""
  quoted = shlex.quote(remote)
  launcher = (
      "set -eu; if sudo -n true 2>/dev/null; then "
      f"exec sudo sh -c {quoted}; "
      "fi; IFS= read -r secret; test -n \"$secret\"; "
      f"printf '%s\\n' \"$secret\" | exec sudo -S -p '' sh -c {quoted}"
  )
  argv = ssh_command(args, launcher)
  with log.open("w", encoding="utf-8", errors="replace") as output:
    output.write("+ " + subprocess.list2cmdline(argv) + " [stdin redacted]\n")
    output.flush()
    result = subprocess.run(
        argv, input=_card_sudo_password(args) + "\n", stdout=output,
        stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
  if result.returncode != 0:
    raise RuntimeError(f"privileged card command failed with {result.returncode}; see {log}")


def card_stage_cleanup_command(paths: dict[str, str], token: str) -> str:
  """Only remove a private stage whose creation receipt matches this run."""
  stage = shlex.quote(paths["stage"])
  marker = shlex.quote(paths["stage"] + "/.qualification-owner")
  return (
      f"set -eu; if [ -e {stage} ] || [ -L {stage} ]; then "
      f"test ! -L {stage}; test -f {marker}; test ! -L {marker}; "
      f'test "$(cat {marker})" = {shlex.quote(token)}; rm -rf -- {stage}; fi'
  )


def stage_card_candidate(args: argparse.Namespace, candidate: CardCandidate, token: str,
                         evidence: Path) -> dict[str, str]:
  paths = card_candidate_paths(token)
  library_paths = (
      f"{paths['root']}/usr/lib",
      f"{paths['root']}/usr/lib/aarch64-linux-gnu",
      f"{paths['root']}/usr/lib/aarch64-linux-gnu/neat/runtime",
      "/usr/lib/aarch64-linux-gnu",
  )
  plugin_paths = (
      f"{paths['root']}/usr/lib/aarch64-linux-gnu/neat/gst-plugins",
      "/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
  )
  prepare = (
      f"set -eu; umask 077; mkdir {paths['stage']}; "
      f"printf '%s\\n' {token} > {paths['stage']}/.qualification-owner; "
      f"mkdir {paths['packages']} {paths['root']} {paths['run']} {paths['log']}"
  )
  try:
    run_logged(ssh_command(args, prepare), evidence / "card-candidate-prepare.log", timeout=30)
    run_logged(
        scp_command(args, candidate.packages, paths["packages"] + "/"),
        evidence / "card-candidate-transfer.log", timeout=900)
    extract = (
        f"set -eu; for package in {paths['packages']}/*.deb; do "
        f"dpkg-deb -x \"$package\" {paths['root']}; done; "
        f"test -x {paths['worker']}; test -r {paths['firmware_source']}; "
        f"printf 'worker_sha256='; sha256sum {paths['worker']} | awk '{{print $1}}'; "
        f"printf 'runtime_sha256='; sha256sum {paths['firmware_source']} | awk '{{print $1}}'; "
        + (f"test -x {paths['genai_worker']}; printf 'genai_worker_sha256='; "
           f"sha256sum {paths['genai_worker']} | awk '{{print $1}}'"
           if args.include_genai else "true")
    )
    log = evidence / "card-candidate-extract.log"
    run_logged(ssh_command(args, extract), log, timeout=900)
    values = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
      if "=" in line and not line.startswith("+"):
        name, value = line.split("=", 1)
        if name in {"runtime_sha256", "worker_sha256", "genai_worker_sha256"}:
          values[name] = value.strip()
    expected_receipt = {"runtime_sha256", "worker_sha256"}
    if args.include_genai:
      expected_receipt.add("genai_worker_sha256")
    if set(values) != expected_receipt or any(
        not re.fullmatch(r"[0-9a-f]{64}", value) for value in values.values()):
      raise RuntimeError(f"card candidate extraction receipt is incomplete; see {log}")
    library_path = ":".join(library_paths)
    dependency_check = (
        "set -eu; test -x /lib/ld-linux-aarch64.so.1; "
        f"LD_LIBRARY_PATH={library_path} ldd {paths['worker']}; "
        f"! LD_LIBRARY_PATH={library_path} ldd {paths['worker']} | grep -q 'not found'"
    )
    run_logged(
        ssh_command(args, dependency_check),
        evidence / "card-candidate-dependencies.log", timeout=60)
  except BaseException as error:
    try:
      cleanup = run_captured_to_file(
          ssh_command(args, card_stage_cleanup_command(paths, token)),
          evidence / "card-candidate-stage-cleanup.log", timeout=60)
      if cleanup.returncode != 0:
        error.add_note(
            "private card stage cleanup returned "
            f"{cleanup.returncode}; see {evidence / 'card-candidate-stage-cleanup.log'}")
    except BaseException as cleanup_error:
      error.add_note(f"private card stage cleanup also failed: {cleanup_error}")
    raise
  values.update(paths)
  values["card_env"] = " ".join((
      f"SIMA_NEAT_PCIE_HELPER_PATH={paths['worker']}",
      f"SIMA_NEAT_PCIE_RUN_DIR={paths['run']}",
      f"SIMA_NEAT_PCIE_LOG_DIR={paths['log']}",
      "LD_LIBRARY_PATH=" + ":".join(library_paths),
      "GST_PLUGIN_PATH_1_0=" + ":".join(plugin_paths),
  ))
  values["genai_card_env"] = " ".join((
      f"SIMA_NEAT_PCIE_GENAI_HELPER_PATH={paths['genai_worker']}",
      "LD_LIBRARY_PATH=" + ":".join(library_paths),
      "GST_PLUGIN_PATH_1_0=" + ":".join(plugin_paths),
  ))
  return values


def activate_card_candidate(args: argparse.Namespace, staged: dict[str, str],
                            baseline_firmware: str, evidence: Path) -> None:
  if not re.fullmatch(r"[A-Za-z0-9._+-]+", baseline_firmware):
    raise ValueError(f"card baseline firmware name is unsafe: {baseline_firmware!r}")
  lock = _validate_remote_path(args.remote_lock_path)
  remember = (
      f"set -eu; test -r {lock}/owner; umask 077; "
      f"printf '%s\\n' {shlex.quote(baseline_firmware)} > {lock}/baseline_firmware"
  )
  run_logged(
      ssh_command(args, remember), evidence / "card-candidate-baseline-record.log", timeout=30)
  remoteproc = "/sys/class/remoteproc/remoteproc1"
  command = (
      "set -eu; "
      f"source={staged['firmware_source']}; target={staged['firmware_installed']}; "
      f"candidate={staged['firmware_name']}; baseline={shlex.quote(baseline_firmware)}; "
      f"remoteproc={remoteproc}; test -r \"$source\"; test ! -e \"$target\"; "
      "restore() { printf stop > \"$remoteproc/state\" 2>/dev/null || true; "
      "printf '%s' \"$baseline\" > \"$remoteproc/firmware\" 2>/dev/null || true; "
      "printf start > \"$remoteproc/state\" 2>/dev/null || true; rm -f \"$target\"; }; "
      "trap restore EXIT; install -m 0644 \"$source\" \"$target\"; "
      "printf stop > \"$remoteproc/state\"; printf '%s' \"$candidate\" > \"$remoteproc/firmware\"; "
      "printf start > \"$remoteproc/state\"; grep -qx \"$candidate\" \"$remoteproc/firmware\"; "
      "grep -qx running \"$remoteproc/state\"; trap - EXIT"
  )
  run_sudo_remote(args, command, evidence / "card-candidate-activate.log")


def restore_card_candidate(args: argparse.Namespace, token: str, evidence: Path) -> None:
  paths = card_candidate_paths(token)
  lock = _validate_remote_path(args.remote_lock_path)
  remoteproc = "/sys/class/remoteproc/remoteproc1"
  command = (
      "set -eu; "
      f"lock={lock}; target={paths['firmware_installed']}; candidate={paths['firmware_name']}; "
      f"remoteproc={remoteproc}; "
      "if [ -r \"$lock/baseline_firmware\" ]; then "
      "read -r baseline < \"$lock/baseline_firmware\"; "
      "case \"$baseline\" in (*[!A-Za-z0-9._+-]*|'') exit 74;; esac; "
      "read -r current < \"$remoteproc/firmware\"; "
      "if [ \"$current\" = \"$candidate\" ]; then "
      "printf stop > \"$remoteproc/state\"; printf '%s' \"$baseline\" > \"$remoteproc/firmware\"; "
      "printf start > \"$remoteproc/state\"; "
      "elif [ \"$current\" != \"$baseline\" ]; then exit 75; fi; "
      "grep -qx \"$baseline\" \"$remoteproc/firmware\"; grep -qx running \"$remoteproc/state\"; "
      "fi; rm -f \"$target\""
  )
  run_sudo_remote(args, command, evidence / "card-candidate-restore.log")
  cleanup = (
      card_stage_cleanup_command(paths, token) +
      f"; rm -f -- {lock}/baseline_firmware; "
      f"test ! -e {paths['stage']}"
  )
  run_logged(
      ssh_command(args, cleanup), evidence / "card-candidate-files-cleanup.log", timeout=60)


class RemoteExclusiveLock:
  """Atomic card-side lock shared by every host that can SSH to the card."""

  def __init__(self, args: argparse.Namespace, path: str, token: str):
    self.args = args
    self.path = _validate_remote_path(path)
    if not re.fullmatch(r"[0-9a-f]{32}", token):
      raise ValueError("remote PCIe hardware lock token must be 128-bit lowercase hex")
    self.token = token
    self.acquired = False
    self.conflicted = False

  def _run(self, remote: str, log: Path) -> subprocess.CompletedProcess[str]:
    result = run_captured_to_file(ssh_command(self.args, remote), log, timeout=30)
    log.write_text(f"returncode={result.returncode}\n{result.stdout}",
                   encoding="utf-8", errors="replace")
    return result

  def owner(self, log: Path) -> str | None:
    remote = (
        f"set -eu; lock={self.path}; "
        "if [ -d \"$lock\" ]; then "
        "test -r \"$lock/owner\"; printf 'owner='; cat \"$lock/owner\"; "
        "else printf 'owner=\\n'; fi"
    )
    result = self._run(remote, log)
    if result.returncode != 0:
      raise RuntimeError(f"could not inspect card-side PCIe lock; see {log}")
    match = re.fullmatch(r"owner=([0-9a-f]{32})?\s*", result.stdout)
    if not match:
      raise RuntimeError(f"card-side PCIe lock owner is invalid; see {log}")
    return match.group(1) or None

  def acquire(self, log: Path) -> None:
    remote = (
        f"set -eu; lock={self.path}; token={self.token}; "
        "if mkdir \"$lock\" 2>/dev/null; then "
        "trap 'rm -f \"$lock/owner\"; rmdir \"$lock\" 2>/dev/null || true' EXIT; "
        "umask 077; printf '%s\\n' \"$token\" > \"$lock/owner\"; trap - EXIT; "
        "else printf 'lock_owner='; cat \"$lock/owner\" 2>/dev/null || true; exit 73; fi"
    )
    result = self._run(remote, log)
    self.conflicted = result.returncode == 73
    if result.returncode != 0:
      raise RuntimeError(f"card-side PCIe lock is already held or unavailable; see {log}")
    self.acquired = True

  def release(self, log: Path) -> None:
    remote = (
        f"set -eu; lock={self.path}; token={self.token}; "
        "test -r \"$lock/owner\"; read -r owner < \"$lock/owner\"; "
        "test \"$owner\" = \"$token\"; rm -f \"$lock/owner\"; rmdir \"$lock\""
    )
    result = self._run(remote, log)
    if result.returncode != 0:
      raise RuntimeError(f"could not release owned card-side PCIe lock; see {log}")
    self.acquired = False


def card_snapshot(args: argparse.Namespace, evidence: Path, label: str, *,
                  runtime_path: str = "", worker_path: str = "",
                  expected_runtime_sha256: str = "", expected_worker_sha256: str = "",
                  genai_worker_path: str = "", expected_genai_worker_sha256: str = "") \
    -> dict[str, str]:
  if bool(runtime_path) != bool(worker_path):
    raise ValueError("card runtime and worker paths must be supplied together")
  if runtime_path:
    runtime_path = _validate_remote_path(runtime_path)
    worker_path = _validate_remote_path(worker_path)
  if genai_worker_path:
    genai_worker_path = _validate_remote_path(genai_worker_path)
  remote = (
      "set -eu; "
      "printf 'machine='; uname -m; "
      "test -r /etc/buildinfo; "
      "printf 'platform_release='; sed -n 's/^SIMA_BUILD_VERSION[[:space:]]*=[[:space:]]*//p' "
      "/etc/buildinfo | head -n 1 | tr -d '\\r\\n'; printf '\\n'; "
      "printf 'platform_sha256='; sha256sum -- /etc/buildinfo | awk '{print $1}'; "
      + (f"printf 'runtime_sha256='; sha256sum -- {runtime_path} | awk '{{print $1}}'; "
         f"printf 'worker_sha256='; sha256sum -- {worker_path} | awk '{{print $1}}'; "
         if runtime_path else "") +
      (f"printf 'genai_worker_sha256='; sha256sum -- {genai_worker_path} | awk '{{print $1}}'; "
         if genai_worker_path else "") +
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
      if name in {"machine", "platform_release", "platform_sha256", "runtime_sha256",
                  "worker_sha256", "genai_worker_sha256", "firmware", "remoteproc_state",
                  "workers", "genai_workers"}:
        values[name] = value.strip()
  required = {"machine", "platform_release", "platform_sha256", "firmware",
              "remoteproc_state", "workers", "genai_workers"}
  if runtime_path:
    required.update(("runtime_sha256", "worker_sha256"))
  if genai_worker_path:
    required.add("genai_worker_sha256")
  if set(values) != required:
    raise RuntimeError(f"card {label} snapshot is incomplete; see {log}")
  if values["machine"] != "aarch64":
    raise RuntimeError(f"card has unexpected architecture: {values['machine']}")
  if not re.search(r"(?:^|[_ .-])3[.]0(?:[._-]|$)", values["platform_release"]):
    raise RuntimeError(
        f"card is not running Modalix Platform 3.0: {values['platform_release']}")
  if (runtime_path and values["runtime_sha256"].lower()
      != expected_runtime_sha256.lower()):
    raise RuntimeError("card runtime checksum differs from the exact Linux candidate")
  if (worker_path and values["worker_sha256"].lower()
      != expected_worker_sha256.lower()):
    raise RuntimeError("card tensor worker checksum differs from the exact Linux candidate")
  if (genai_worker_path and values["genai_worker_sha256"].lower()
      != expected_genai_worker_sha256.lower()):
    raise RuntimeError("card GenAI worker checksum differs from the exact Linux candidate")
  if values["remoteproc_state"] != "running":
    raise RuntimeError(f"card remote processor is not running: {values['remoteproc_state']}")
  return values


def terminate_card_workers(args: argparse.Namespace, log: Path) -> None:
  remote = (
      "pkill -TERM -f '(^|/)[p]cie-pipeline-builder([[:space:]]|$)' 2>/dev/null || true; "
      "pkill -TERM -f '(^|/)[n]eat-pcie-genai-worker([[:space:]]|$)' 2>/dev/null || true; "
      "sleep 2; "
      "pkill -KILL -f '(^|/)[p]cie-pipeline-builder([[:space:]]|$)' 2>/dev/null || true; "
      "pkill -KILL -f '(^|/)[n]eat-pcie-genai-worker([[:space:]]|$)' 2>/dev/null || true"
  )
  run_logged(ssh_command(args, remote), log, timeout=30)


def stop_card_workers(args: argparse.Namespace, evidence: Path) -> None:
  terminate_card_workers(args, evidence / "card-worker-cleanup.log")


def restart_windows_platform_service(log: Path) -> None:
  command = (
      "$service = Get-Service -Name 'SiMaMlaDaemon' -ErrorAction Stop; "
      "Restart-Service -InputObject $service -Force -ErrorAction Stop; "
      "$service.WaitForStatus("
      "[System.ServiceProcess.ServiceControllerStatus]::Running, "
      "[TimeSpan]::FromSeconds(30)); "
      "if ($service.Status -ne "
      "[System.ServiceProcess.ServiceControllerStatus]::Running) { "
      "throw 'SiMaMlaDaemon did not return to running state' }; "
      "Start-Sleep -Seconds 2"
  )
  run_logged(
      [require_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive",
       "-Command", command], log, timeout=60)


def recover_windows_platform_device(platform_runtime: Path, log: Path) -> None:
  """Remove and rescan the Platform-owned card after an unclean client exit."""
  recovery_tool = platform_runtime.resolve(strict=True) / "simaai_pcie_test.exe"
  if not recovery_tool.is_file():
    raise RuntimeError(
        "Platform device recovery tool is unavailable: "
        f"{recovery_tool}")
  command = r"""
$ErrorActionPreference = 'Stop'
$service = Get-Service -Name 'SiMaMlaDaemon' -ErrorAction Stop
$output = @()
$exitCode = $null
try {
  Stop-Service -InputObject $service -Force -ErrorAction Stop
  $service.WaitForStatus(
      [System.ServiceProcess.ServiceControllerStatus]::Stopped,
      [TimeSpan]::FromSeconds(30))
  if ($service.Status -ne [System.ServiceProcess.ServiceControllerStatus]::Stopped) {
    throw 'SiMaMlaDaemon did not stop before device recovery'
  }
  $output = @(& $env:SIMA_WINDOWS_PLATFORM_RECOVERY_TOOL -R -c 0 2>&1)
  $exitCode = $LASTEXITCODE
} finally {
  Start-Service -InputObject $service -ErrorAction SilentlyContinue
  $service.WaitForStatus(
      [System.ServiceProcess.ServiceControllerStatus]::Running,
      [TimeSpan]::FromSeconds(30))
}
$output | ForEach-Object { Write-Output $_ }
Write-Output "recovery_tool_exit=$exitCode"
if (($output | Out-String) -notmatch '(?m)^Device recovery successful\s*$') {
  throw 'Platform device remove/rescan did not report success'
}
if ($service.Status -ne [System.ServiceProcess.ServiceControllerStatus]::Running) {
  throw 'SiMaMlaDaemon did not return to running state after device recovery'
}
Start-Sleep -Seconds 2
"""
  environment = os.environ.copy()
  environment["SIMA_WINDOWS_PLATFORM_RECOVERY_TOOL"] = str(recovery_tool)
  run_logged(
      [require_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive",
       "-Command", command], log, env=environment, timeout=120)


def wait_for_no_card_workers(args: argparse.Namespace, log: Path, timeout: int) -> None:
  remote = (
      "printf 'workers='; "
      "pgrep -fc '(^|/)[p]cie-pipeline-builder([[:space:]]|$)' || true"
  )
  deadline = time.monotonic() + timeout
  attempt = log.with_name(log.name + ".attempt")
  with log.open("w", encoding="utf-8", errors="replace") as output:
    try:
      while True:
        result = run_captured_to_file(
            ssh_command(args, remote), attempt, timeout=30)
        output.write(f"returncode={result.returncode} {result.stdout.strip()}\n")
        output.flush()
        match = re.search(r"workers=(\d+)", result.stdout)
        if result.returncode == 0 and match and int(match.group(1)) == 0:
          return
        if time.monotonic() >= deadline:
          raise RuntimeError(f"card worker did not stop within {timeout} seconds; see {log}")
        time.sleep(1)
    finally:
      attempt.unlink(missing_ok=True)


def wait_for_card_ssh(args: argparse.Namespace, log: Path, timeout: int) -> None:
  """Wait for card networking to return after the Windows PnP rescan."""
  deadline = time.monotonic() + timeout
  attempt = log.with_name(log.name + ".attempt")
  with log.open("w", encoding="utf-8", errors="replace") as output:
    try:
      while True:
        try:
          result = run_captured_to_file(
              ssh_command(args, "true"), attempt, timeout=15)
        except subprocess.TimeoutExpired:
          result = subprocess.CompletedProcess([], 124, "SSH probe timed out")
        output.write(
            f"returncode={result.returncode} {result.stdout.strip()}\n")
        output.flush()
        if result.returncode == 0:
          return
        if time.monotonic() >= deadline:
          raise RuntimeError(
              f"card SSH did not recover within {timeout} seconds; see {log}")
        time.sleep(1)
    finally:
      attempt.unlink(missing_ok=True)


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

  def launch(label: str, *, terminate_on_pull_timeout: bool = False
             ) -> tuple[subprocess.Popen, object, Path]:
    log = evidence / f"cpp-{label}.log"
    stream = log.open("w", encoding="utf-8", errors="replace")
    command = [blocker, "--iterations", "0"]
    if terminate_on_pull_timeout:
      command.append("--terminate-on-pull-timeout")
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

  process, stream, _ = launch(
      "test_worker_loss_recovery", terminate_on_pull_timeout=True)
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
  terminate_card_workers(args, evidence / "client-process-death-worker-cleanup.log")
  wait_for_no_card_workers(args, evidence / "client-process-death-drain.log", 120)
  recover_windows_platform_device(
      args.platform_runtime,
      evidence / "client-process-death-platform-recovery.log")
  wait_for_card_ssh(
      args, evidence / "client-process-death-card-readiness.log", 120)
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
    command: list[object] = [
        performance, "--mode", mode, "--warmup", "100", "--iterations", "3000"]
    if mode == "latency":
      command.extend(["--max-mean-latency-ms", "12"])
    else:
      command.extend(["--min-throughput-fps", "440"])
    run_logged(
        command, evidence / f"cpp-test_tensor_{mode}.log",
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
  require_junit_cases(evidence / "python-hardware.xml", set(BASE_PYTHON_TESTS))
  completed = list(BASE_PYTHON_TESTS)
  if args.include_genai:
    genai_tests = {
        "test_genai_llm_run_stream_history_and_cancel",
        "test_genai_vlm_run_and_stream",
        "test_genai_asr_file_pcm_translation_and_stream",
    }
    run_logged(
        [python, "-m", "pytest", "-vv", "--junitxml", evidence / "python-genai.xml",
         "pcie_host/python/tests/test_genai_hardware.py"],
        evidence / "python-genai.log",
        env=environment,
        cwd=args.source_root,
        timeout=args.test_timeout_seconds,
    )
    require_junit_cases(evidence / "python-genai.xml", genai_tests)
    completed.extend(sorted(genai_tests))
  return completed


def tutorial_facade_links(staged: dict[str, str]) -> dict[str, str]:
  """Default tutorial paths point only at this locked candidate's private files."""
  root = staged["root"] + "/tutorial-facade"
  return {
      "/usr/bin/pcie-pipeline-builder": root + "/worker",
      "/run/sima-neat/pcie": root + "/run",
      "/var/log/sima-neat/pcie": root + "/log",
  }


def stage_installed_tutorial_facade(args: argparse.Namespace,
                                    staged: dict[str, str], evidence: Path) -> None:
  root = staged["root"]
  worker = staged["worker"]
  library_path = ":".join((
      f"{root}/usr/lib",
      f"{root}/usr/lib/aarch64-linux-gnu",
      f"{root}/usr/lib/aarch64-linux-gnu/neat/runtime",
      "/usr/lib/aarch64-linux-gnu",
  ))
  plugin_path = ":".join((
      f"{root}/usr/lib/aarch64-linux-gnu/neat/gst-plugins",
      "/usr/lib/aarch64-linux-gnu/gstreamer-1.0",
  ))
  wrapper = "\n".join((
      "#!/bin/sh",
      "set -eu",
      f"exec env LD_LIBRARY_PATH={shlex.quote(library_path)} "
      f"GST_PLUGIN_PATH_1_0={shlex.quote(plugin_path)} "
      f"{shlex.quote(worker)} \"$@\"",
      "",
  ))
  links = tutorial_facade_links(staged)
  private = shlex.quote(root + "/tutorial-facade")
  owner = shlex.quote(args.card_user)
  commands = ["set -eu"]
  # Reject existing files, directories AND dangling links before any mutation.
  for destination in links:
    path = shlex.quote(destination)
    parent = shlex.quote(str(PurePosixPath(destination).parent))
    commands.extend((f"test ! -e {path}", f"test ! -L {path}",
                     f"test ! -L {parent}"))
  commands.extend((
      f"mkdir {private}",
      f"chown {owner}:{owner} {private}",
      f"install -d -o {owner} -g {owner} -m 0700 {private}/run {private}/log",
      f"printf %s {shlex.quote(wrapper)} > {private}/worker",
      f"chmod 0755 {private}/worker",
  ))
  for index, (destination, target) in enumerate(links.items()):
    parent = shlex.quote(str(PurePosixPath(destination).parent))
    commands.append(
        f"if [ ! -d {parent} ]; then mkdir {parent}; "
        f"touch {private}/parent-{index}; fi")
    # -T is essential: an existing directory must not receive a nested link.
    commands.append(f"ln -sT {shlex.quote(target)} {shlex.quote(destination)}")
  run_sudo_remote(args, "; ".join(commands), evidence / "tutorial-card-facade-stage.log")


def remove_installed_tutorial_facade(args: argparse.Namespace, staged: dict[str, str],
                                     log: Path) -> None:
  """Reconcile even an uncertain stage without deleting an existing installation."""
  private = shlex.quote(staged["root"] + "/tutorial-facade")
  commands = ["set -eu"]
  for index, (destination, target) in enumerate(tutorial_facade_links(staged).items()):
    path = shlex.quote(destination)
    commands.append(
        f'if [ -L {path} ] && [ "$(readlink {path})" = {shlex.quote(target)} ]; '
        f"then rm -- {path}; fi")
    parent = shlex.quote(str(PurePosixPath(destination).parent))
    commands.append(
        f"if [ -f {private}/parent-{index} ]; then "
        f"rmdir {parent}; rm -- {private}/parent-{index}; fi")
  run_sudo_remote(args, "; ".join(commands), log)


def run_customer_tutorials(args: argparse.Namespace, install_root: Path, python: Path,
                           work: Path, evidence: Path,
                           environment: dict[str, str]) -> dict[str, list[str]]:
  tutorial_name = "024_run_your_first_model_over_pcie"
  source_root = install_root / "share" / "sima-pcie-host" / "tutorials"
  python_root = source_root / tutorial_name
  executable_root = install_root / "lib" / "sima-pcie-host" / "tutorials"
  run_root = work / "installed customer tutorial"
  run_root.mkdir()

  model = Path(environment["SIMAPCIE_YOLOV8_MODEL"])
  try:
    os.link(model, run_root / "yolo_v8s_mpk.tar.gz")
  except OSError:
    shutil.copy2(model, run_root / "yolo_v8s_mpk.tar.gz")
  shutil.copytree(source_root / "assets", run_root / "share" / "sima-pcie-host" /
                  "tutorials" / "assets")

  completed: dict[str, list[str]] = {"cpp": [], "python": []}
  programs = ("run_tensor_mode", "run_image_mode", "run_image_boxdecode")
  for program in programs:
    script = python_root / f"{program}.py"
    if not script.is_file():
      raise RuntimeError(f"installed package is missing Python tutorial: {script}")
    run_logged(
        [python, script], evidence / f"tutorial-024-python-{program}.log",
        env=environment, cwd=run_root, timeout=args.test_timeout_seconds)
    completed["python"].append(program)

    executable = executable_root / f"tutorial_024_{program}.exe"
    if not executable.is_file():
      raise RuntimeError(f"installed package is missing C++ tutorial: {executable}")
    run_logged(
        [executable], evidence / f"tutorial-024-cpp-{program}.log",
        env=environment, cwd=run_root, timeout=args.test_timeout_seconds)
    completed["cpp"].append(program)
  return completed


def qualify(args: argparse.Namespace) -> None:
  require_windows()
  if not args.dependency_runtime_dir:
    raise RuntimeError("hardware qualification requires dependency runtime directories")
  candidate = validate_candidate(args.candidate)
  card_candidate = validate_card_candidate(args.card_candidate, args.expected_card_commit)
  if args.expected_card_commit != candidate.core_commit:
    raise ValueError("Windows and card candidates must name the same Core commit")
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
      candidate, args.platform_runtime, args.gstreamer_root,
      args.dependency_runtime_dir)

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
      "failure": None,
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

  remote_lock_token = secrets.token_hex(16)
  with ExclusiveLock(args.lock_file, remote_lock_token) as card_lock:
    remote_lock = RemoteExclusiveLock(args, args.remote_lock_path, remote_lock_token)
    try:
      remote_lock.acquire(evidence / "card-lock-acquire.log")
    except BaseException:
      # SSH can disconnect after the remote mkdir succeeds. Retain our token for
      # recovery unless the remote command explicitly reported lock contention.
      if not remote_lock.conflicted:
        card_lock.retain()
      raise
    card_lock.retain()
    staged_card: dict[str, str] | None = None
    card_stage_attempted = False
    tutorial_facade_staged = False
    try:
      baseline = card_snapshot(args, evidence, "baseline")
      result["card_baseline"] = baseline
      if baseline["workers"] != "0" or baseline["genai_workers"] != "0":
        raise RuntimeError("card had PCIe workers before qualification; refusing to disturb them")

      card_stage_attempted = True
      staged_card = stage_card_candidate(
          args, card_candidate, remote_lock_token, evidence)
      activate_card_candidate(args, staged_card, baseline["firmware"], evidence)
      active_card = card_snapshot(
          args, evidence, "candidate",
          runtime_path=staged_card["firmware_installed"],
          worker_path=staged_card["worker"],
          expected_runtime_sha256=staged_card["runtime_sha256"],
          expected_worker_sha256=staged_card["worker_sha256"],
          genai_worker_path=staged_card["genai_worker"] if args.include_genai else "",
          expected_genai_worker_sha256=(
              staged_card.get("genai_worker_sha256", "") if args.include_genai else ""))
      if active_card["firmware"] != staged_card["firmware_name"]:
        raise RuntimeError("card did not select the exact Linux candidate firmware")
      result["card_candidate"] = {
          "core_commit": args.expected_card_commit,
          "firmware": active_card["firmware"],
          "runtime_sha256": active_card["runtime_sha256"],
          "worker_sha256": active_card["worker_sha256"],
      }
      if args.include_genai:
        result["card_candidate"]["genai_worker_sha256"] = active_card["genai_worker_sha256"]
      restart_windows_platform_service(
          evidence / "card-candidate-platform-start.log")

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
      cli_environment["SIMA_CLI_CHECK_FOR_UPDATE"] = "0"
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
          "SIMAPCIE_CARD_ENV": staged_card["card_env"],
          "SIMA_NEAT_PCIE_GENAI_CARD_ENV": staged_card["genai_card_env"],
      })
      environment["PATH"] = os.pathsep.join((
          str(install_root / "bin"),
          str(args.platform_runtime.resolve(strict=True)),
          str(args.gstreamer_root.resolve(strict=True) / "bin"),
          *(str(path.resolve(strict=True) / "bin") for path in args.dependency_prefix),
          *(str(path.resolve(strict=True)) for path in args.dependency_runtime_dir),
          environment["PATH"],
      ))
      run_logged(
          [args.gstreamer_root.resolve(strict=True) / "bin" / "gst-inspect-1.0.exe",
           install_root / "bin" / "gstreamer-1.0" / "gstneatpciehost.dll"],
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

      tutorial_facade_staged = True
      stage_installed_tutorial_facade(args, staged_card, evidence)
      result["tests"]["tutorial"] = run_customer_tutorials(
          args, install_root, python, work, evidence, environment)
      remove_installed_tutorial_facade(
          args, staged_card, evidence / "tutorial-card-facade-cleanup.log")
      tutorial_facade_staged = False
      result["tests"]["cpp"] = run_cpp_tests(args, candidate.root, evidence, environment)
      result["tests"]["python"] = run_python_tests(args, python, evidence, environment)
      result["performance"] = {
          mode: parse_performance_log(evidence / f"cpp-test_tensor_{mode}.log", mode)
          for mode in ("latency", "throughput")
      }
      result["result"] = "PASS"
    except BaseException as error:
      failure = error
      result["failure"] = {
          "message": str(error),
          "type": type(error).__name__,
      }
    finally:
      if card_stage_attempted:
        try:
          stop_card_workers(args, evidence)
        except BaseException as error:
          cleanup_errors.append(str(error))
      if tutorial_facade_staged:
        try:
          remove_installed_tutorial_facade(
              args, staged_card, evidence / "tutorial-card-facade-final-cleanup.log")
          result["cleanup"]["tutorial_facade"] = "PASS"
        except BaseException as error:
          cleanup_errors.append(str(error))
      if card_stage_attempted and not cleanup_errors:
        try:
          restore_card_candidate(args, remote_lock_token, evidence)
          result["cleanup"]["card_candidate"] = "PASS"
          restart_windows_platform_service(
              evidence / "card-stock-platform-recovery.log")
          result["cleanup"]["platform_service"] = "PASS"
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
        unchanged = ("platform_release", "platform_sha256", "firmware", "remoteproc_state")
        expected_workers = "0" if card_stage_attempted else baseline.get("workers", "0")
        expected_genai = "0" if card_stage_attempted else baseline.get("genai_workers", "0")
        if (final["workers"] != expected_workers or final["genai_workers"] != expected_genai
            or any(name in baseline and final.get(name) != baseline[name] for name in unchanged)):
          cleanup_errors.append("card did not return to its qualified baseline")
      except BaseException as error:
        cleanup_errors.append(str(error))
      if not cleanup_errors:
        try:
          remote_lock.release(evidence / "card-lock-release.log")
          card_lock.release_on_exit()
          result["cleanup"]["remote_lock"] = "PASS"
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
  parser.add_argument("--card-candidate", type=Path, required=True)
  parser.add_argument("--card-host", required=True)
  parser.add_argument(
      "--card-sudo-password-env", default="SIMAPCIE_CARD_SUDO_PASSWORD")
  parser.add_argument("--card-user", default="sima")
  parser.add_argument("--dependency-prefix", action="append", default=[], type=Path)
  parser.add_argument("--dependency-runtime-dir", action="append", default=[], type=Path)
  parser.add_argument("--evidence", type=Path, required=True)
  parser.add_argument("--expected-card-commit", required=True)
  parser.add_argument("--expected-sima-cli-commit", required=True)
  parser.add_argument("--gstreamer-root", type=Path, required=True)
  parser.add_argument("--include-genai", action="store_true")
  parser.add_argument("--lock-file", type=Path, required=True)
  parser.add_argument("--remote-lock-path", default="/tmp/neat-pcie-card0.lock")
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
  if not re.fullmatch(r"[0-9a-f]{40}", args.expected_card_commit):
    parser.error("--expected-card-commit must be a full lowercase Git commit")
  if not re.fullmatch(r"[0-9a-f]{40}", args.expected_sima_cli_commit):
    parser.error("--expected-sima-cli-commit must be a full lowercase Git commit")
  if (args.upgrade_base_candidate is None) != (not args.expected_upgrade_base_commit):
    parser.error(
        "--upgrade-base-candidate and --expected-upgrade-base-commit must be supplied together")
  if args.upgrade_base_candidate is not None and not re.fullmatch(
      r"[0-9a-f]{40}", args.expected_upgrade_base_commit):
    parser.error("--expected-upgrade-base-commit must be a full lowercase Git commit")
  qualify(args)


if __name__ == "__main__":
  main()
