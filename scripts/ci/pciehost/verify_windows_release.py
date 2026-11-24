#!/usr/bin/env python3
"""Fail-closed release gate for a Windows PCIe-host cohort."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from pathlib import PureWindowsPath
import re

from windows_candidate import _read_json as read_object
from windows_candidate import (
    dependency_identity, msi_version, scheduler_platform_runtime, sha256, validate_candidate)
from windows_release import core_version_matches, customer_target, release_tag
from qualify_windows_hardware import PERF_MAX_MEAN_LATENCY_MS, PERF_MIN_THROUGHPUT_FPS
from windows_hardware_assets import asset_manifest_sha256, read_asset_manifest


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


def verify_msi_transaction_rollback(candidate_root: Path) -> dict:
  receipt = read_object(
      candidate_root / "msi-transaction-rollback.json",
      "MSI failed-transaction rollback receipt")
  if receipt.get("schema_version") != 1 or receipt.get("status") != "PASS":
    raise ValueError("candidate did not pass the MSI failed-transaction rollback gate")
  fixture = receipt.get("fixture")
  product_code_pattern = r"\{[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}\}"
  if (not isinstance(fixture, dict)
      or fixture.get("custom_action") != "FailAfterOwnedState"
      or not re.fullmatch(product_code_pattern, str(fixture.get("product_code", "")))
      or not re.fullmatch(r"[0-9a-f]{64}", str(fixture.get("msi_sha256", "")))):
    raise ValueError("MSI rollback receipt has an invalid CI fixture identity")
  failure = receipt.get("failure")
  if failure != {"expected_exit_code": 1603, "observed_exit_code": 1603}:
    raise ValueError("MSI rollback fixture did not fail through Windows Installer")
  if receipt.get("mid_transaction") != {"file": "PASS", "registry": "PASS"}:
    raise ValueError("MSI rollback fixture did not observe its transaction-owned state")
  expected_rollback = {
      "install_directory": "PASS",
      "product_registration": "PASS",
      "registry": "PASS",
      "sentinel": "PASS",
  }
  if receipt.get("rollback") != expected_rollback:
    raise ValueError("MSI rollback fixture left files or product registration behind")
  cohort = receipt.get("preserved_cohort")
  if (not isinstance(cohort, dict)
      or cohort.get("file_inventory") != "PASS"
      or not re.fullmatch(r"[0-9a-f]{64}", str(cohort.get("file_inventory_sha256", "")))
      or cohort.get("installed_consumer") != "PASS"):
    raise ValueError("failed MSI transaction did not preserve the installed cohort")
  product_codes = set()
  for owner in ("scheduler", "core"):
    state = cohort.get(owner)
    if (not isinstance(state, dict)
        or state.get("state_before") != "DEFAULT"
        or state.get("state_after") != "DEFAULT"
        or not re.fullmatch(product_code_pattern, str(state.get("product_code", "")))):
      raise ValueError(
          f"failed MSI transaction did not preserve {owner} product registration")
    product_codes.add(state["product_code"])
  if len(product_codes | {fixture["product_code"]}) != 3:
    raise ValueError("MSI rollback receipt reuses a product identity")
  log = candidate_root / "msi-transaction-rollback.log"
  if (not log.is_file() or log.stat().st_size == 0
      or receipt.get("installer_log_sha256") != sha256(log)):
    raise ValueError("MSI rollback receipt does not bind its verbose installer log")
  return receipt


def verify(candidate_dir: Path, qualification_path: Path, expected_commit: str,
           require_signatures: bool, require_genai: bool = False,
           require_release: bool = False,
           asset_manifest_path: Path | None = None,
           require_release_qualification: bool = False, ref: str = "") -> dict:
  candidate = validate_candidate(candidate_dir)
  if candidate.core_commit != expected_commit:
    raise ValueError(
        f"candidate Core commit {candidate.core_commit} is not workflow commit {expected_commit}")
  verify_msi_transaction_rollback(candidate.root)

  qualification = read_object(qualification_path, "hardware qualification")
  if qualification.get("result") != "PASS":
    raise ValueError("hardware qualification did not pass")
  expected_asset_manifest = read_asset_manifest(
      asset_manifest_path
      if asset_manifest_path is not None
      else qualification_path.with_name("hardware-asset-manifest.json"))
  expected_asset_identity = {
      "identity_sha256": asset_manifest_sha256(expected_asset_manifest),
      "manifest": expected_asset_manifest,
  }
  if qualification.get("assets") != expected_asset_identity:
    raise ValueError("hardware qualification used a different hardware asset manifest")
  if require_genai and not expected_asset_manifest["include_genai"]:
    raise ValueError("release qualification asset manifest does not admit GenAI assets")
  host = qualification.get("host")
  security = host.get("windows_security") if isinstance(host, dict) else None
  required_security = {
      "account", "architecture", "build_number", "caption", "driver_signing",
      "is_administrator", "product_type", "smart_app_control_state",
      "applocker_rule_collections", "version",
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
  expected_dependencies = dependency_identity(
      candidate.root, require_dependency_runtime=True)
  if qualification.get("dependencies") != expected_dependencies:
    raise ValueError("hardware qualification used a different runtime dependency cohort")
  if qualification.get("platform_runtime") != scheduler_platform_runtime(candidate.root):
    raise ValueError("hardware qualification used a different Platform runtime")
  cleanup = qualification.get("cleanup")
  if not isinstance(cleanup, dict) or cleanup.get("errors") != []:
    raise ValueError("hardware qualification did not prove clean removal")
  expected_cleanup = {
      "card_stage": "PASS", "core_msi": "PASS", "python": "PASS",
      "customer_state": "PASS", "host_state": "PASS", "remote_lock": "PASS",
      "scheduler_msi": "PASS", "work": "PASS",
  }
  if any(cleanup.get(name) != value for name, value in expected_cleanup.items()):
    raise ValueError("hardware qualification did not remove every customer-owned component")
  host_baseline = qualification.get("host_state_baseline")
  host_final = qualification.get("host_state_final")
  if (not isinstance(host_baseline, dict) or host_baseline != host_final
      or set(host_baseline) != {"dependencies", "platform"}
      or host_baseline.get("dependencies") != expected_dependencies):
    raise ValueError("hardware qualification did not preserve its Windows host baseline")
  host_platform = host_baseline.get("platform")
  service = host_platform.get("service") if isinstance(host_platform, dict) else None
  devices = host_platform.get("pnp_devices") if isinstance(host_platform, dict) else None
  required_devices = {"SiMa.ai MLA Device", "SiMa.ai Virtual Ethernet Adapter"}
  if (not isinstance(service, dict) or service.get("name") != "SiMaMlaDaemon"
      or service.get("state") != "Running"
      or not isinstance(service.get("configuration"), dict)
      or not service["configuration"] or not isinstance(devices, list) or not devices
      or any(not isinstance(device, dict) for device in devices)
      or {re.sub(r" #[0-9]+$", "", str(device.get("friendly_name")))
          for device in devices} != required_devices
      or any(device.get("status") != "OK" for device in devices)):
    raise ValueError("hardware qualification recorded an invalid Windows host baseline")
  customer_install = qualification.get("customer_install")
  if (not isinstance(customer_install, dict)
      or customer_install.get("command") != customer_target(
          release_tag(candidate.metadata.get("version")))
      or " " not in str(customer_install.get("download_path_with_spaces", ""))):
    raise ValueError("hardware qualification did not prove the customer CLI install")
  if require_release_qualification:
    if (customer_install.get("native_only") != "PASS"
        or customer_install.get("recovery") != "PASS"
        or customer_install.get("repeat") != "PASS"):
      raise ValueError("release qualification did not prove the repeated customer CLI flow")
    missing_wheel = customer_install.get("missing_wheel")
    if (not isinstance(missing_wheel, dict) or missing_wheel.get("status") != "PASS"
        or not isinstance(missing_wheel.get("exit_code"), int)
        or missing_wheel["exit_code"] == 0):
      raise ValueError("release qualification did not prove missing-wheel failure and recovery")
    wheel_failure = customer_install.get("wheel_failure")
    if (not isinstance(wheel_failure, dict) or wheel_failure.get("status") != "PASS"
        or not isinstance(wheel_failure.get("exit_code"), int)
        or wheel_failure["exit_code"] == 0):
      raise ValueError(
          "release qualification did not prove native success, wheel failure, and retry")
    repair = customer_install.get("repair")
    if (not isinstance(repair, dict) or repair.get("core") != "PASS"
        or repair.get("scheduler") != "PASS"):
      raise ValueError("release qualification did not prove MSI repair")
  standard_user = customer_install.get("standard_user")
  if (customer_install.get("plugin_discovery") != "PASS"
      or customer_install.get("process_activation") != "PASS"
      or not isinstance(standard_user, dict)
      or standard_user.get("is_administrator") is not False
      or standard_user.get("process_is_administrator") is not False
      or standard_user.get("doctor") != "PASS"
      or standard_user.get("python") != "PASS"
      or standard_user.get("standard_user") != "PASS"):
    raise ValueError("hardware qualification did not prove standard-user process activation")
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
  baseline_network = qualification.get("card_network_baseline")
  final_network = qualification.get("card_network_final")
  network_identity = {
      "card_host", "local_address", "interface_index", "interface_guid",
      "interface_description", "pnp_device_id",
  }
  if (not isinstance(baseline_network, dict) or not isinstance(final_network, dict)
      or not network_identity.issubset(baseline_network)
      or not network_identity.issubset(final_network)
      or any(baseline_network[name] != final_network[name] for name in network_identity)
      or not re.fullmatch(
          r"SiMa[.]ai Virtual Ethernet Adapter(?: #[0-9]+)?",
          str(baseline_network.get("interface_description", "")))):
    raise ValueError("hardware qualification did not bind one stable Windows adapter to the card")
  card_identity = {
      "machine", "platform_release", "platform_sha256", "remoteproc_state",
  }
  if (not isinstance(baseline_card, dict) or not isinstance(final_card, dict)
      or not card_identity.issubset(baseline_card)
      or not card_identity.issubset(final_card)
      or any(baseline_card[name] != final_card[name] for name in card_identity)
      or baseline_card.get("machine") != "aarch64"
      or not re.search(
          r"(?:^|[_ -])3[.]0(?:[._-]|$)", str(baseline_card.get("platform_release", "")))
      or final_card.get("workers") != "0" or final_card.get("genai_workers") != "0"):
    raise ValueError("hardware qualification did not preserve its Platform 3.0 card baseline")
  card_candidate = qualification.get("card_candidate")
  if (not isinstance(card_candidate, dict)
      or card_candidate.get("core_commit") != candidate.core_commit
      or not re.fullmatch(r"[0-9a-f]{64}", str(card_candidate.get("runtime_sha256", "")))
      or not re.fullmatch(r"[0-9a-f]{64}", str(card_candidate.get("worker_sha256", "")))
      or not re.fullmatch(r"[0-9a-f]{64}", str(card_candidate.get("bundle_sha256", "")))
      or not isinstance(card_candidate.get("installed"), dict)
      or not core_version_matches(
          card_candidate["installed"].get("sima-neat", ""), candidate.core_commit, ref)):
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
  }
  if require_release_qualification:
    required_cpp |= {"test_worker_loss_recovery", "test_client_process_death_recovery"}
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
      or latency["mean_ms"] > float(PERF_MAX_MEAN_LATENCY_MS)
      or throughput["fps"] < float(PERF_MIN_THROUGHPUT_FPS)):
    raise ValueError("hardware qualification did not meet the PCIe performance gate")
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
        "test_genai_bounded_cancellation_soak_and_context_cleanup",
        "test_genai_competing_sessions_are_isolated",
        "test_genai_llm_run", "test_genai_vlm_run", "test_genai_asr_run",
        "test_genai_llm_run_stream_history_and_cancel",
        "test_genai_request_timeout_and_restart_after_failure",
        "test_genai_workers_drained",
        "test_genai_restart_after_service_loss",
        "test_genai_restart_after_worker_loss",
        "test_genai_service_loss_recovery",
        "test_genai_vlm_run_and_stream",
        "test_genai_asr_file_pcm_translation_and_stream",
        "test_genai_worker_loss_recovery",
    }
    if not required_genai.issubset(names):
      raise ValueError("release qualification did not run the complete Windows GenAI matrix")
    genai_admission = qualification.get("genai_admission")
    expected_outcomes = {name: "PASS" for name in required_genai}
    if genai_admission != expected_outcomes:
      raise ValueError("release qualification has incomplete Windows GenAI outcomes")
    standard_genai_outcomes = {
        "config_binding", "connection_options", "default_root", "override_absence",
        "real_genai", "required_assets_readable", "sentinel", "sentinel_cleanup",
    }
    if any(standard_user.get(name) != "PASS" for name in standard_genai_outcomes):
      raise ValueError(
          "release qualification did not prove default-root real GenAI as a standard user")
    if (standard_user.get("default_media_serve_root") != "data"
        or standard_user.get("real_genai_case") != "test_genai_asr_run"):
      raise ValueError(
          "release qualification did not use the advertised default GenAI service root")
    default_directory = PureWindowsPath(str(standard_user.get("default_media_directory", "")))
    common_application_data = PureWindowsPath(
        str(standard_user.get("common_application_data", "")))
    platform_config = PureWindowsPath(str(standard_user.get("platform_config", "")))
    configured_directory = PureWindowsPath(
        str(standard_user.get("platform_config_media_directory", "")))
    if (not common_application_data.is_absolute() or not common_application_data.drive
        or default_directory != common_application_data / "SiMa.ai" / "data"
        or platform_config != common_application_data / "SiMa.ai" / "simaai-mla-daemon.conf"
        or configured_directory != default_directory):
      raise ValueError(
          "release qualification did not bind Windows CommonApplicationData GenAI defaults")
    standard_executable = (
        candidate.root / "hardware-tests" / "lib" / "sima-pcie-host" / "tests"
        / "test_genai_run.exe")
    if (not standard_executable.is_file()
        or standard_user.get("real_genai_executable_sha256") != sha256(standard_executable)):
      raise ValueError(
          "release qualification standard-user GenAI run used a different executable")
    media = qualification.get("genai_media")
    if media is not None:
      if not isinstance(media, dict) or set(media) != {"directory", "serve_root"}:
        raise ValueError("release qualification has an invalid custom GenAI media contract")
      media_directory = PureWindowsPath(str(media["directory"]))
      if not media_directory.is_absolute() or not media_directory.drive:
        raise ValueError("release qualification has an invalid Windows GenAI media directory")
      if not isinstance(media["serve_root"], str) or not media["serve_root"]:
        raise ValueError("release qualification has an invalid Windows GenAI media service root")

  signatures = read_object(candidate.root / "windows-signatures.json", "signature receipt")
  records = signatures.get("records")
  if not isinstance(records, list) or not records:
    raise ValueError("signature receipt has no records")
  invalid = [record.get("path") for record in records
             if not isinstance(record, dict) or record.get("status") != "Valid"]
  dependency_signatures = {"scheduler": []}
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

  return {
      "core_commit": candidate.core_commit,
      "hardware_qualified": True,
      "asset_manifest_sha256": expected_asset_identity["identity_sha256"],
      "host_security_recorded": True,
      "path_qualified": True,
      "customer_install_qualified": True,
      "msi_transaction_rollback_qualified": True,
      "release_gates_required": require_release,
      "release_qualification_required": require_release_qualification,
      "genai_qualified": require_genai,
      "resources": candidate.resources,
      "dependency_signatures": dependency_signatures,
      "signatures_required": require_signatures,
      "signatures_valid": not invalid,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--asset-manifest", type=Path, required=True)
  parser.add_argument("--expected-commit", required=True)
  parser.add_argument("--ref", required=True, help="Git ref name of the selected Core commit")
  parser.add_argument("--qualification", type=Path, required=True)
  parser.add_argument("--require-genai", action="store_true")
  parser.add_argument("--require-release", action="store_true",
                      help="Require release lifecycle evidence even for unsigned packages")
  parser.add_argument("--require-release-qualification", action="store_true",
                      help="Require the qualifier's Windows-only release checks")
  parser.add_argument("--require-signatures", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = verify(
      args.candidate, args.qualification, args.expected_commit,
      args.require_signatures, args.require_genai, args.require_release,
      args.asset_manifest, args.require_release_qualification, args.ref)
  args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
