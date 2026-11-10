#!/usr/bin/env python3
"""Idempotent emergency cleanup for the Windows PCIe hardware CI job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from qualify_windows_hardware import (
    RemoteExclusiveLock, restore_card_candidate, stop_card_workers)
from windows_candidate import validate_candidate


def stale_lock_pid(path: Path) -> int | None:
  try:
    match = re.search(r"\bpid=(\d+)\b", path.read_text(encoding="utf-8"))
  except OSError:
    return None
  return int(match.group(1)) if match else None


def stale_lock_remote_token(path: Path) -> str | None:
  try:
    match = re.search(r"\bremote_token=([0-9a-f]{32})\b", path.read_text(encoding="utf-8"))
  except OSError:
    return None
  return match.group(1) if match else None


def process_is_running(pid: int) -> bool:
  result = subprocess.run(
      ["tasklist.exe", "/FI", f"PID eq {pid}", "/NH", "/FO", "CSV"],
      check=False, capture_output=True, text=True)
  return result.returncode == 0 and f'"{pid}"' in result.stdout


def recover_stale_lock(path: Path) -> None:
  if not path.exists():
    return
  pid = stale_lock_pid(path)
  if pid is None or process_is_running(pid):
    raise RuntimeError(f"refusing to clear an active or unreadable PCIe hardware lock: {path}")
  path.unlink()


def remove_msi(msi: Path, log: Path) -> int:
  result = subprocess.run(
      ["msiexec.exe", "/x", str(msi), "/quiet", "/norestart", "/L*v", str(log)],
      check=False)
  if result.returncode not in {0, 1605}:
    raise RuntimeError(f"Windows Installer cleanup failed with {result.returncode}; see {log}")
  return result.returncode


def cleanup(args: argparse.Namespace) -> None:
  if sys.platform != "win32":
    raise RuntimeError("Windows PCIe cleanup must run on Windows")
  candidate = validate_candidate(args.candidate)
  upgrade_base = (validate_candidate(args.upgrade_base_candidate)
                  if args.upgrade_base_candidate is not None else None)
  evidence = args.evidence.resolve()
  evidence.mkdir(parents=True, exist_ok=True)
  errors = []
  results = {}

  local_lock = args.lock_file.resolve()
  remote_token = None
  if local_lock.exists():
    pid = stale_lock_pid(local_lock)
    remote_token = stale_lock_remote_token(local_lock)
    if pid is None or process_is_running(pid) or remote_token is None:
      raise RuntimeError(
          f"refusing cleanup with an active or unreadable PCIe hardware lock: {local_lock}")
  probe_token = remote_token or ("0" * 32)
  remote_lock = RemoteExclusiveLock(args, args.remote_lock_path, probe_token)
  remote_owner = remote_lock.owner(evidence / "recovery-card-lock-inspect.log")
  if remote_token is None and remote_owner is not None:
    raise RuntimeError("refusing cleanup because the card-side PCIe lock has another owner")
  if remote_token is not None and remote_owner != remote_token:
    raise RuntimeError("refusing cleanup because local and card-side PCIe lock owners differ")

  try:
    stop_card_workers(args, evidence)
    results["card_workers"] = "stopped"
  except BaseException as error:
    errors.append(str(error))
  if remote_token is not None:
    try:
      restore_card_candidate(args, remote_token, evidence)
      results["card_candidate"] = "restored"
    except BaseException as error:
      errors.append(str(error))
  for owner, msi in (("core", candidate.core_msi), ("scheduler", candidate.scheduler_msi)):
    try:
      results[f"{owner}_msi"] = remove_msi(msi, evidence / f"recovery-{owner}-msi.log")
    except BaseException as error:
      errors.append(str(error))
  install_root = Path(os.environ["ProgramFiles"]) / "SimaPCIeHost"
  if install_root.exists() and upgrade_base is not None:
    for owner, msi in (("core", upgrade_base.core_msi),
                       ("scheduler", upgrade_base.scheduler_msi)):
      try:
        results[f"upgrade_base_{owner}_msi"] = remove_msi(
            msi, evidence / f"recovery-upgrade-base-{owner}-msi.log")
      except BaseException as error:
        errors.append(str(error))
  work = args.work.resolve()
  if work == Path(work.anchor) or work == Path.home().resolve():
    errors.append(f"refusing to remove unsafe hardware work directory: {work}")
  else:
    shutil.rmtree(work, ignore_errors=True)
  if install_root.exists():
    errors.append(f"cohort left files after recovery removal: {install_root}")

  if errors:
    results["lock"] = "retained for operator recovery"
  else:
    try:
      if remote_token is not None:
        remote_lock.release(evidence / "recovery-card-lock-release.log")
        results["remote_lock"] = "released"
      recover_stale_lock(args.lock_file.resolve())
      results["lock"] = "released"
    except BaseException as error:
      errors.append(str(error))
  receipt = {"errors": errors, "results": results}
  (evidence / "recovery-cleanup.json").write_text(
      json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
  if errors:
    raise RuntimeError("; ".join(errors))


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--card-host", required=True)
  parser.add_argument(
      "--card-sudo-password-env", default="SIMAPCIE_CARD_SUDO_PASSWORD")
  parser.add_argument("--card-user", default="sima")
  parser.add_argument("--evidence", type=Path, required=True)
  parser.add_argument("--lock-file", type=Path, required=True)
  parser.add_argument("--remote-lock-path", default="/tmp/neat-pcie-card0.lock")
  parser.add_argument("--ssh-key", type=Path, required=True)
  parser.add_argument("--upgrade-base-candidate", type=Path)
  parser.add_argument("--work", type=Path, required=True)
  args = parser.parse_args()
  cleanup(args)


if __name__ == "__main__":
  main()
