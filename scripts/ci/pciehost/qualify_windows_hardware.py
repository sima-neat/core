#!/usr/bin/env python3
"""Qualify one exact Windows PCIe-host cohort on a directly attached card.

The script is intentionally local to the Windows runner. It never drives the
test process through an inbound SSH session, and it owns install, test, evidence,
and reverse-order cleanup as one transaction.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time

from windows_candidate import sha256, validate_candidate


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
    "test_metadata_yolov8",
    "test_runtime_request_correlation",
    "test_runtime_two_models_and_independent_unload",
    "test_tensor_run_yolov8",
    "test_tensor_parallel_queues_yolov8",
    "test_tensor_run_mla_only",
    "test_image_run_yolov8",
    "test_image_boxdecode_run_yolov8",
)


def write_json(path: Path, value: object) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def require_windows() -> None:
  if sys.platform != "win32" or platform.machine().upper() != "AMD64":
    raise RuntimeError("Windows PCIe hardware qualification requires Windows AMD64")


def require_fresh_directory(path: Path, description: str) -> Path:
  resolved = path.resolve()
  if resolved.exists():
    raise RuntimeError(f"{description} must be fresh: {resolved}")
  resolved.mkdir(parents=True)
  return resolved


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


def msi_transaction(msi: Path, action: str, log: Path) -> None:
  result = subprocess.run(
      ["msiexec.exe", f"/{action}", str(msi), "/quiet", "/norestart", "/L*v", str(log)],
      check=False,
  )
  if result.returncode not in MSI_SUCCESS_CODES:
    raise RuntimeError(f"Windows Installer /{action} failed with {result.returncode}; see {log}")
  if result.returncode != 0:
    raise RuntimeError(f"Windows Installer /{action} requested a restart ({result.returncode})")


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
    run_logged(command, evidence / f"cpp-{executable.stem}.log", env=environment,
               cwd=test_root, timeout=args.test_timeout_seconds)
    completed.append(executable.stem)

  bf16 = test_root / "test_tensor_mla_only_run.exe"
  run_logged(
      [bf16, "--model", environment["SIMAPCIE_MLA_ONLY_BF16_MODEL"]],
      evidence / "cpp-test_tensor_mla_only_bf16_run.log",
      env=environment,
      cwd=test_root,
      timeout=args.test_timeout_seconds,
  )
  completed.append("test_tensor_mla_only_bf16_run")

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
  evidence = require_fresh_directory(args.evidence, "evidence directory")
  work = require_fresh_directory(args.work, "hardware work directory")
  wheelhouse = args.python_wheelhouse.resolve(strict=True)
  if not wheelhouse.is_dir():
    raise RuntimeError(f"Python wheelhouse is not a directory: {wheelhouse}")

  install_root = Path(os.environ["ProgramFiles"]) / "SimaPCIeHost"
  if install_root.exists():
    raise RuntimeError(f"refusing to modify a pre-existing installation: {install_root}")

  result = {
      "candidate": {
          "core_commit": candidate.core_commit,
          "resources": candidate.resources,
          "scheduler_commit": candidate.receipt.get("scheduler_source_commit"),
      },
      "cleanup": {},
      "completed_at_utc": None,
      "result": "FAIL",
      "started_at_utc": datetime.now(timezone.utc).isoformat(),
      "tests": {"cpp": [], "python": []},
  }
  scheduler_installed = False
  core_installed = False
  venv = work / "venv"
  failure: BaseException | None = None
  cleanup_errors: list[str] = []

  with ExclusiveLock(args.lock_file) as card_lock:
    try:
      baseline = card_snapshot(args, evidence, "baseline")
      result["card_baseline"] = baseline
      if baseline["workers"] != "0" or baseline["genai_workers"] != "0":
        raise RuntimeError("card had PCIe workers before qualification; refusing to disturb them")

      msi_transaction(candidate.scheduler_msi, "i", evidence / "scheduler-msi-install.log")
      scheduler_installed = True
      msi_transaction(candidate.core_msi, "i", evidence / "core-msi-install.log")
      core_installed = True

      subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
      python = venv / "Scripts" / "python.exe"
      requirements = ["pytest", "numpy", "pillow", *args.test_requirement]
      run_logged(
          [python, "-m", "pip", "install", "--no-index", "--find-links", wheelhouse,
           candidate.wheel, *requirements],
          evidence / "pip-install.log",
          timeout=900,
      )

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
      require_assets(environment, args.include_genai)
      run_logged(
          [python, "-c", "import pyneatpcie as p; assert p.Model and p.Runtime and p.Tensor"],
          evidence / "python-import.log", env=environment, cwd=work, timeout=60)

      result["tests"]["cpp"] = run_cpp_tests(args, candidate.root, evidence, environment)
      result["tests"]["python"] = run_python_tests(args, python, evidence, environment)
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
      if core_installed:
        try:
          msi_transaction(candidate.core_msi, "x", evidence / "core-msi-uninstall.log")
          result["cleanup"]["core_msi"] = "PASS"
        except BaseException as error:
          cleanup_errors.append(str(error))
      if scheduler_installed:
        try:
          msi_transaction(candidate.scheduler_msi, "x", evidence / "scheduler-msi-uninstall.log")
          result["cleanup"]["scheduler_msi"] = "PASS"
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
  parser.add_argument("--gstreamer-root", type=Path, required=True)
  parser.add_argument("--include-genai", action="store_true")
  parser.add_argument("--lock-file", type=Path, required=True)
  parser.add_argument("--platform-runtime", type=Path, required=True)
  parser.add_argument("--python-wheelhouse", type=Path, required=True)
  parser.add_argument("--source-root", type=Path, default=Path.cwd())
  parser.add_argument("--ssh-key", type=Path, required=True)
  parser.add_argument("--test-requirement", action="append", default=[])
  parser.add_argument("--test-timeout-seconds", type=int, default=7200)
  parser.add_argument("--work", type=Path, required=True)
  args = parser.parse_args()
  for name in ("expected_card_runtime_sha256", "expected_card_worker_sha256"):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", getattr(args, name)):
      parser.error(f"--{name.replace('_', '-')} must be a SHA-256 digest")
  if args.include_genai and not re.fullmatch(
      r"[0-9a-fA-F]{64}", args.expected_card_genai_worker_sha256):
    parser.error("--expected-card-genai-worker-sha256 is required with --include-genai")
  qualify(args)


if __name__ == "__main__":
  main()
