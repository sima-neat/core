#!/usr/bin/env python3
"""Fail-closed release gate for a Windows PCIe-host cohort."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from pathlib import PureWindowsPath
import re

from windows_candidate import dependency_identity, msi_version, sha256, validate_candidate


def read_object(path: Path, description: str) -> dict:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"invalid {description}: {path}: {error}") from error
  if not isinstance(value, dict):
    raise ValueError(f"{description} must be a JSON object")
  return value


def verify_signature_receipt(receipt: dict, description: str,
                             expected_thumbprint: str | None = None) -> list[str]:
  records = receipt.get("records")
  if not isinstance(records, list) or not records:
    raise ValueError(f"{description} has no signature records")
  invalid = [
      str(record.get("path")) for record in records
      if (not isinstance(record, dict) or record.get("status") != "Valid"
          or not record.get("thumbprint") or not record.get("timestamp_thumbprint"))
  ]
  if invalid:
    raise ValueError(f"{description} has invalid or untimestamped signatures: {invalid}")
  if expected_thumbprint:
    expected = expected_thumbprint.upper()
    wrong_publishers = [
        str(record.get("path")) for record in records
        if str(record.get("thumbprint", "")).upper() != expected
    ]
    if wrong_publishers:
      raise ValueError(f"{description} has an unexpected signer: {wrong_publishers}")
  return [str(record["path"]) for record in records]


def verify_platform_signatures(candidate_root: Path) -> list[str]:
  dependencies = read_object(
      candidate_root / "windows-dependencies.json", "Windows dependency receipt")
  platform = dependencies.get("platform")
  runtime_dlls = platform.get("runtime_dlls") if isinstance(platform, dict) else None
  if not isinstance(runtime_dlls, list) or not runtime_dlls:
    raise ValueError("Windows dependency receipt has no Platform runtime DLLs")
  invalid = []
  verified = []
  for item in runtime_dlls:
    if not isinstance(item, dict) or not item.get("name"):
      invalid.append("<invalid record>")
      continue
    signature = item.get("signature")
    if (not isinstance(signature, dict) or signature.get("status") != "Valid"
        or not signature.get("thumbprint") or not signature.get("timestamp_thumbprint")):
      invalid.append(str(item["name"]))
      continue
    verified.append(str(item["name"]))
  if invalid:
    raise ValueError(f"Platform runtime has invalid or untimestamped signatures: {invalid}")
  return verified


def verify(candidate_dir: Path, qualification_path: Path, expected_commit: str,
           require_signatures: bool, require_genai: bool = False,
           require_release: bool = False) -> dict:
  candidate = validate_candidate(candidate_dir)
  if candidate.core_commit != expected_commit:
    raise ValueError(
        f"candidate Core commit {candidate.core_commit} is not workflow commit {expected_commit}")

  qualification = read_object(qualification_path, "hardware qualification")
  if qualification.get("result") != "PASS":
    raise ValueError("hardware qualification did not pass")
  host = qualification.get("host")
  security = host.get("windows_security") if isinstance(host, dict) else None
  required_security = {
      "account", "architecture", "build_number", "caption", "is_administrator",
      "product_type", "smart_app_control_state", "applocker_rule_collections", "version",
  }
  if not isinstance(security, dict) or set(security) != required_security:
    raise ValueError("hardware qualification did not record Windows security state")
  try:
    windows_build = int(security["build_number"])
  except (TypeError, ValueError) as error:
    raise ValueError("hardware qualification recorded an invalid Windows build") from error
  if (security.get("product_type") != 1 or windows_build < 22000
      or "64" not in str(security.get("architecture", ""))):
    raise ValueError("hardware qualification did not run on Windows 11 x64 workstation")
  host_python = str(host.get("python", ""))
  if not re.fullmatch(r"3[.]14(?:[.][0-9]+)?", host_python):
    raise ValueError("hardware qualification did not use the supported CPython 3.14 ABI")
  qualified_candidate = qualification.get("candidate")
  if not isinstance(qualified_candidate, dict):
    raise ValueError("hardware qualification has no candidate identity")
  if qualified_candidate.get("core_commit") != candidate.core_commit:
    raise ValueError("hardware qualification covered a different Core commit")
  if qualified_candidate.get("resources") != candidate.resources:
    raise ValueError("hardware qualification covered different package bytes")
  native_tests = candidate.receipt.get("native_tests")
  native_names = native_tests.get("names") if isinstance(native_tests, dict) else None
  if (not isinstance(native_names, list) or not native_names
      or native_tests.get("exit_code") != 0
      or native_tests.get("tests") != len(native_names)
      or native_tests.get("skipped") != 0
      or native_tests.get("failures") != 0
      or native_tests.get("errors") != 0
      or native_tests.get("junit_sha256") != sha256(candidate.root / "native-tests.xml")):
    raise ValueError("Windows native qualification was not an exact zero-skip pass")
  if qualification.get("dependencies") != dependency_identity(
      candidate.root, require_dependency_runtime=True):
    raise ValueError("hardware qualification used a different runtime dependency cohort")
  cleanup = qualification.get("cleanup")
  if not isinstance(cleanup, dict) or cleanup.get("errors") != []:
    raise ValueError("hardware qualification did not prove clean removal")
  expected_cleanup = {
      "card_candidate": "PASS", "core_msi": "PASS", "python": "PASS",
      "remote_lock": "PASS", "scheduler_msi": "PASS",
  }
  if any(cleanup.get(name) != value for name, value in expected_cleanup.items()):
    raise ValueError("hardware qualification did not remove every customer-owned component")
  customer_install = qualification.get("customer_install")
  if (not isinstance(customer_install, dict)
      or customer_install.get("command") != "core/pciehost/windows/amd64@3.0"
      or customer_install.get("native_only") != "PASS"
      or customer_install.get("recovery") != "PASS"
      or customer_install.get("repeat") != "PASS"
      or " " not in str(customer_install.get("download_path_with_spaces", ""))):
    raise ValueError("hardware qualification did not prove the repeated customer CLI flow")
  missing_wheel = customer_install.get("missing_wheel")
  if (not isinstance(missing_wheel, dict) or missing_wheel.get("status") != "PASS"
      or not isinstance(missing_wheel.get("exit_code"), int)
      or missing_wheel["exit_code"] == 0):
    raise ValueError("hardware qualification did not prove missing-wheel failure and recovery")
  wheel_failure = customer_install.get("wheel_failure")
  if (not isinstance(wheel_failure, dict) or wheel_failure.get("status") != "PASS"
      or not isinstance(wheel_failure.get("exit_code"), int)
      or wheel_failure["exit_code"] == 0):
    raise ValueError(
        "hardware qualification did not prove native success, wheel failure, and retry")
  repair = customer_install.get("repair")
  if (not isinstance(repair, dict) or repair.get("core") != "PASS"
      or repair.get("scheduler") != "PASS"
      or customer_install.get("plugin_discovery") != "PASS"
      or customer_install.get("process_activation") != "PASS"):
    raise ValueError(
        "hardware qualification did not prove MSI repair and customer process activation")
  if require_release:
    upgrade = customer_install.get("upgrade")
    if (not isinstance(upgrade, dict) or upgrade.get("status") != "PASS"
        or not re.fullmatch(r"[0-9a-f]{40}", str(upgrade.get("baseline_commit", "")))
        or upgrade.get("baseline_commit") == candidate.core_commit
        or not upgrade.get("baseline_version") or not upgrade.get("candidate_version")
        or upgrade.get("candidate_version") != candidate.metadata.get("version")):
      raise ValueError("release qualification did not prove a real MSI package upgrade")
    try:
      is_newer = msi_version(upgrade["candidate_version"]) > msi_version(
          upgrade["baseline_version"])
    except ValueError as error:
      raise ValueError("release qualification recorded an invalid MSI upgrade version") from error
    if not is_newer:
      raise ValueError("release qualification did not upgrade to a newer MSI package version")
  sima_cli = qualification.get("sima_cli")
  if (not isinstance(sima_cli, dict)
      or not re.fullmatch(r"[0-9a-f]{40}", str(sima_cli.get("commit", "")))
      or not re.fullmatch(r"[0-9a-f]{64}", str(sima_cli.get("sha256", "")))
      or sima_cli["commit"][:12] not in str(sima_cli.get("version", ""))):
    raise ValueError("hardware qualification has no exact sima-cli candidate identity")
  baseline_card = qualification.get("card_baseline")
  final_card = qualification.get("card_final")
  card_identity = {
      "machine", "platform_release", "platform_sha256", "firmware", "remoteproc_state",
  }
  if (not isinstance(baseline_card, dict) or not isinstance(final_card, dict)
      or not card_identity.issubset(baseline_card)
      or not card_identity.issubset(final_card)
      or any(baseline_card[name] != final_card[name] for name in card_identity)
      or baseline_card.get("machine") != "aarch64"
      or not re.search(
          r"(?:^|[_ .-])3[.]0(?:[._-]|$)", str(baseline_card.get("platform_release", "")))
      or final_card.get("workers") != "0" or final_card.get("genai_workers") != "0"):
    raise ValueError("hardware qualification did not preserve its Platform 3.0 card baseline")
  card_candidate = qualification.get("card_candidate")
  if (not isinstance(card_candidate, dict)
      or card_candidate.get("core_commit") != candidate.core_commit
      or not re.fullmatch(r"[0-9a-f]{64}", str(card_candidate.get("runtime_sha256", "")))
      or not re.fullmatch(r"[0-9a-f]{64}", str(card_candidate.get("worker_sha256", "")))
      or not re.fullmatch(
          r"neat-pcie-card0-[0-9a-f]{32}", str(card_candidate.get("firmware", "")))):
    raise ValueError("hardware qualification did not use the exact Linux card candidate")
  tests = qualification.get("tests")
  if (not isinstance(tests, dict) or not tests.get("cpp") or not tests.get("python")):
    raise ValueError("hardware qualification did not run both customer APIs")
  tutorials = tests.get("tutorial")
  tutorial_programs = {"run_tensor_mode", "run_image_mode", "run_image_boxdecode"}
  if (not isinstance(tutorials, dict)
      or not tutorial_programs.issubset(set(tutorials.get("cpp", [])))
      or not tutorial_programs.issubset(set(tutorials.get("python", [])))):
    raise ValueError("hardware qualification did not run installed C++ and Python tutorials")
  required_cpp = {
      "test_metadata", "test_tensor_run", "test_tensor_mla_only_run",
      "test_tensor_mla_only_bf16_run", "test_tensor_parallel_queues",
      "test_image_run", "test_image_boxdecode_run", "test_queue_blocker",
      "test_model_resilience", "test_tensor_latency", "test_tensor_throughput",
      "test_worker_loss_recovery", "test_client_process_death_recovery",
  }
  required_python = {
      "test_async_test_runner_cancels_blocked_peer", "test_metadata_yolov8",
      "test_runtime_request_correlation", "test_runtime_two_models_and_independent_unload",
      "test_tensor_run_yolov8", "test_tensor_latency_yolov8",
      "test_tensor_throughput_yolov8", "test_tensor_parallel_queues_yolov8",
      "test_tensor_run_mla_only", "test_image_run_yolov8",
      "test_image_boxdecode_run_yolov8",
  }
  missing_cpp = sorted(required_cpp - set(tests["cpp"]))
  missing_python = sorted(required_python - set(tests["python"]))
  if missing_cpp or missing_python:
    raise ValueError(
        f"hardware qualification is missing required vision cases: "
        f"C++={missing_cpp}, Python={missing_python}")
  performance = qualification.get("performance")
  latency = performance.get("latency") if isinstance(performance, dict) else None
  throughput = performance.get("throughput") if isinstance(performance, dict) else None

  def positive_metrics(value: object, names: set[str]) -> bool:
    if not isinstance(value, dict) or set(value) != names:
      return False
    return all(
        isinstance(value[name], (int, float)) and not isinstance(value[name], bool)
        and math.isfinite(float(value[name])) and float(value[name]) > 0.0
        for name in names)

  if (not positive_metrics(latency, {"count", "mean_ms", "effective_fps"})
      or not positive_metrics(
          throughput, {"frames", "total_ms", "fps", "input_mib_per_s", "output_mib_per_s"})):
    raise ValueError("hardware qualification did not retain valid performance measurements")
  if (latency["count"] != 3000 or throughput["frames"] != 3000
      or latency["mean_ms"] > 12.0 or throughput["fps"] < 440.0):
    raise ValueError("hardware qualification did not meet the Linux PCIe performance gate")
  path_fixtures = qualification.get("path_fixtures")
  if not isinstance(path_fixtures, dict) or len(path_fixtures) != 3:
    raise ValueError("hardware qualification did not record Windows path fixtures")
  invalid_paths = []
  for name, value in path_fixtures.items():
    path = PureWindowsPath(str(value))
    if not path.is_absolute() or not path.drive or " " not in str(path):
      invalid_paths.append(name)
  if invalid_paths:
    raise ValueError(f"hardware qualification did not exercise absolute paths with spaces: {invalid_paths}")
  if require_genai:
    if not re.fullmatch(
        r"[0-9a-f]{64}", str(card_candidate.get("genai_worker_sha256", ""))):
      raise ValueError("release qualification did not use the exact Linux GenAI worker")
    names = set(tests["cpp"]) | set(tests["python"])
    required_genai = {
        "test_genai_llm_run", "test_genai_vlm_run", "test_genai_asr_run",
        "test_genai_llm_run_stream_history_and_cancel",
        "test_genai_vlm_run_and_stream",
        "test_genai_asr_file_pcm_translation_and_stream",
    }
    if not required_genai.issubset(names):
      raise ValueError("release qualification did not run the complete Windows GenAI matrix")

  signatures = read_object(candidate.root / "windows-signatures.json", "signature receipt")
  records = signatures.get("records")
  if not isinstance(records, list) or not records:
    raise ValueError("signature receipt has no records")
  invalid = [record.get("path") for record in records
             if not isinstance(record, dict) or record.get("status") != "Valid"]
  dependency_signatures = {"platform": [], "scheduler": []}
  if require_signatures:
    if not signatures.get("release_required"):
      raise ValueError("candidate was not built in signed release mode")
    if not signatures.get("certificate_sha1") or not signatures.get("timestamp_url"):
      raise ValueError("signed release receipt has no certificate or timestamp authority")
    expected_thumbprint = str(signatures["certificate_sha1"]).upper()
    verify_signature_receipt(signatures, "release candidate", expected_thumbprint)
    scheduler_signatures = read_object(
        candidate.root / "scheduler-windows-signatures.json",
        "Scheduler signature receipt")
    if not scheduler_signatures.get("release_required"):
      raise ValueError("Scheduler candidate was not built in signed release mode")
    if str(scheduler_signatures.get("certificate_sha1", "")).upper() != expected_thumbprint:
      raise ValueError("Scheduler candidate uses a different trusted publisher")
    dependency_signatures["scheduler"] = verify_signature_receipt(
        scheduler_signatures, "Scheduler candidate", expected_thumbprint)
    dependency_signatures["platform"] = verify_platform_signatures(candidate.root)

  return {
      "core_commit": candidate.core_commit,
      "hardware_qualified": True,
      "host_security_recorded": True,
      "path_qualified": True,
      "customer_install_qualified": True,
      "release_gates_required": require_release,
      "genai_qualified": require_genai,
      "resources": candidate.resources,
      "dependency_signatures": dependency_signatures,
      "signatures_required": require_signatures,
      "signatures_valid": not invalid,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--expected-commit", required=True)
  parser.add_argument("--qualification", type=Path, required=True)
  parser.add_argument("--require-genai", action="store_true")
  parser.add_argument("--require-release", action="store_true",
                      help="Require release lifecycle evidence even for unsigned packages")
  parser.add_argument("--require-signatures", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = verify(
      args.candidate, args.qualification, args.expected_commit,
      args.require_signatures, args.require_genai, args.require_release)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
