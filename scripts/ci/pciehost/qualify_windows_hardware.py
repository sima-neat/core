#!/usr/bin/env python3
"""Qualify one exact Windows PCIe-host cohort on a directly attached card.

The script is intentionally local to the Windows runner. It never drives the
test process through an inbound SSH session, and it owns install, test, evidence,
and reverse-order cleanup as one transaction.
"""

from __future__ import annotations

import argparse
import base64
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
import hashlib
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
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
    dependency_identity, msi_version, require_junit_cases, sha256, validate_candidate,
    write_json)
from windows_hardware_assets import asset_manifest_sha256, verify_asset_manifest


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
GENAI_ADMISSION_TESTS = {
    "test_genai_asr_file_pcm_translation_and_stream",
    "test_genai_asr_run",
    "test_genai_bounded_cancellation_soak_and_context_cleanup",
    "test_genai_competing_sessions_are_isolated",
    "test_genai_llm_run",
    "test_genai_llm_run_stream_history_and_cancel",
    "test_genai_request_timeout_and_restart_after_failure",
    "test_genai_restart_after_service_loss",
    "test_genai_restart_after_worker_loss",
    "test_genai_service_loss_recovery",
    "test_genai_vlm_run",
    "test_genai_vlm_run_and_stream",
    "test_genai_worker_loss_recovery",
    "test_genai_workers_drained",
}
SUPPORTED_PYTHON = (3, 14)
STANDARD_USER_GENAI_ENVIRONMENT = {
    "PATH",
    "SIMA_NEAT_PCIE_GENAI_CARD_ENV",
    "SIMA_PCIE_HOST_LIBRARY_PATH",
    "SIMAPCIE_CARD_HOST",
    "SIMAPCIE_CARD_ID",
    "SIMAPCIE_GENAI_MODELS_PATH",
    "SIMAPCIE_GENAI_REQUEST_TIMEOUT_MS",
    "SIMAPCIE_GENAI_TEST_ASSETS",
    "SIMAPCIE_READINESS_TIMEOUT_MS",
    "SIMAPCIE_USER",
    "SIMA_TEST_LLIMA_ASR_MODEL",
}


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


def card_management_addresses(card_host: str) -> tuple[str, str]:
  try:
    address = ipaddress.ip_address(card_host)
  except ValueError as error:
    raise ValueError(f"card host must be a literal management IPv4 address: {card_host}") from error
  if address.version != 4:
    raise ValueError("card host must use the supported 10.0.N.2 management address")
  octets = tuple(int(value) for value in str(address).split("."))
  if octets[:2] != (10, 0) or octets[3] != 2:
    raise ValueError("card host must use the supported 10.0.N.2 management address")
  return str(address), f"10.0.{octets[2]}.1"


def windows_card_network_identity(card_host: str, log: Path) -> dict:
  """Bind a card management address to one up SiMa virtual Ethernet adapter."""
  remote, local = card_management_addresses(card_host)
  script = r"""
$ErrorActionPreference = 'Stop'
$addresses = @(Get-NetIPAddress -AddressFamily IPv4 `
  -IPAddress $env:SIMA_PCIE_LOCAL_ADDRESS -ErrorAction Stop)
if ($addresses.Count -ne 1) { throw 'management source address is not unique' }
$selection = @(Find-NetRoute -RemoteIPAddress $env:SIMA_PCIE_CARD_HOST `
  -ErrorAction Stop)
$sources = @($selection | Where-Object {
  $_.CimClass.CimClassName -eq 'MSFT_NetIPAddress'
})
$routes = @($selection | Where-Object {
  $_.CimClass.CimClassName -eq 'MSFT_NetRoute'
})
if ($sources.Count -ne 1 -or $routes.Count -ne 1) {
  throw 'management route selection is not unique'
}
$source = $sources[0]
$route = $routes[0]
$adapters = @(Get-NetAdapter -InterfaceIndex $source.InterfaceIndex -ErrorAction Stop)
if ($adapters.Count -ne 1) { throw 'management adapter is not unique' }
$adapter = $adapters[0]
[ordered]@{
  card_host = $env:SIMA_PCIE_CARD_HOST
  local_address = [string]$source.IPAddress
  interface_index = [int]$source.InterfaceIndex
  interface_guid = ([guid]$adapter.InterfaceGuid).ToString('D')
  interface_alias = [string]$adapter.Name
  interface_description = [string]$adapter.InterfaceDescription
  interface_status = [string]$adapter.Status
  mac_address = [string]$adapter.MacAddress
  pnp_device_id = [string]$adapter.PnPDeviceID
  destination_prefix = [string]$route.DestinationPrefix
  next_hop = [string]$route.NextHop
} | ConvertTo-Json -Compress
"""
  environment = os.environ.copy()
  environment.update({
      "SIMA_PCIE_CARD_HOST": remote,
      "SIMA_PCIE_LOCAL_ADDRESS": local,
  })
  command = [
      require_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive",
      "-Command", script,
  ]
  result = subprocess.run(
      command, text=True, capture_output=True, env=environment, check=False, timeout=60)
  log.write_text(
      "+ " + subprocess.list2cmdline(command) + "\n" + result.stdout + result.stderr,
      encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError(f"could not bind the selected card to a Windows adapter; see {log}")
  try:
    identity = json.loads(result.stdout)
  except json.JSONDecodeError as error:
    raise RuntimeError(f"card network identity was not valid JSON; see {log}") from error
  required = {
      "card_host", "local_address", "interface_index", "interface_guid",
      "interface_alias", "interface_description", "interface_status", "mac_address",
      "pnp_device_id", "destination_prefix", "next_hop",
  }
  if not isinstance(identity, dict) or not required.issubset(identity):
    raise RuntimeError(f"card network identity was incomplete; see {log}")
  if (identity["card_host"] != remote or identity["local_address"] != local
      or not isinstance(identity["interface_index"], int)
      or not re.fullmatch(
          r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
          str(identity["interface_guid"]))
      or not re.fullmatch(
          r"SiMa[.]ai Virtual Ethernet Adapter(?: #[0-9]+)?",
          str(identity["interface_description"]))
      or str(identity["interface_status"]).lower() != "up"
      or not identity["pnp_device_id"]
      or identity["next_hop"] != "0.0.0.0"):
    raise RuntimeError(f"selected card route does not use one up SiMa adapter; see {log}")
  return identity


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
  platform_script = r"""
$names = @('SiMa.ai MLA Device', 'SiMa.ai Virtual Ethernet Adapter')
$matches = @(Get-CimInstance Win32_PnPSignedDriver -ErrorAction SilentlyContinue |
  Where-Object { $_.DeviceName -in $names })
if ($matches.Count -ne $names.Count) { exit 3 }
$versions = @($matches | Select-Object -ExpandProperty DriverVersion -Unique)
if ($versions.Count -ne 1 -or -not $versions[0]) { exit 4 }
[Console]::Out.Write([string]$versions[0])
"""
  platform_result = subprocess.run(
      ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
       "-Command", platform_script],
      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60,
      check=False)
  if platform_result.returncode != 0 or not platform_result.stdout.strip():
    raise RuntimeError("could not read a single active SiMa Platform driver version")
  actual = {
      "platform_version": platform_result.stdout.strip(),
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


def windows_platform_state_snapshot(log: Path) -> dict:
  """Record stable Platform service and present-device identity.

  Process IDs, service control capabilities, and other transition-dependent
  fields are deliberately omitted because the qualifier restarts the service.
  The service registry values are configuration, so they must survive exactly.
  Driver signer fields are identity only; Platform certificate trust remains
  an externally owned release-policy gate.
  """
  script = r"""
$ErrorActionPreference = 'Stop'
$serviceName = 'SiMaMlaDaemon'
$services = @(Get-CimInstance Win32_Service -Filter "Name='$serviceName'")
if ($services.Count -ne 1) { throw 'SiMaMlaDaemon service is not unique' }
$service = $services[0]
$serviceKey = Get-ItemProperty `
  -LiteralPath "HKLM:\SYSTEM\CurrentControlSet\Services\$serviceName" `
  -ErrorAction Stop
$registry = [ordered]@{}
@($serviceKey.PSObject.Properties | Where-Object {
  $_.Name -notlike 'PS*'
} | Sort-Object Name) | ForEach-Object {
  $registry[$_.Name] = $_.Value
}

$deviceNames = @('SiMa.ai MLA Device', 'SiMa.ai Virtual Ethernet Adapter')
# Windows appends " #N" to a repeated adapter name, as the route binding accepts.
$present = @(Get-PnpDevice -PresentOnly -ErrorAction Stop | Where-Object {
  ($_.FriendlyName -replace ' #[0-9]+$', '') -in $deviceNames
})
foreach ($name in $deviceNames) {
  if (@($present | Where-Object {
      ($_.FriendlyName -replace ' #[0-9]+$', '') -eq $name }).Count -lt 1) {
    throw "No present $name was found"
  }
}
$signed = @(Get-CimInstance Win32_PnPSignedDriver -ErrorAction Stop)
$devices = @($present | Sort-Object InstanceId | ForEach-Object {
  $device = $_
  $drivers = @($signed | Where-Object { $_.DeviceID -eq $device.InstanceId })
  if ($drivers.Count -ne 1) {
    throw "Present Platform device has no unique signed driver: $($device.InstanceId)"
  }
  $driver = $drivers[0]
  [ordered]@{
    class = [string]$device.Class
    driver_inf = [string]$driver.InfName
    driver_provider = [string]$driver.DriverProviderName
    driver_signer = [string]$driver.Signer
    driver_version = [string]$driver.DriverVersion
    friendly_name = [string]$device.FriendlyName
    instance_id = [string]$device.InstanceId
    is_signed = [bool]$driver.IsSigned
    manufacturer = [string]$driver.Manufacturer
    status = [string]$device.Status
  }
})

[ordered]@{
  pnp_devices = $devices
  service = [ordered]@{
    configuration = $registry
    display_name = [string]$service.DisplayName
    name = [string]$service.Name
    path_name = [string]$service.PathName
    service_type = [string]$service.ServiceType
    start_mode = [string]$service.StartMode
    start_name = [string]$service.StartName
    state = [string]$service.State
  }
} | ConvertTo-Json -Depth 12 -Compress
"""
  command = [
      require_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive",
      "-Command", script,
  ]
  result = subprocess.run(
      command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
      timeout=60, check=False)
  log.write_text(
      f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}",
      encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError(f"could not record Windows Platform state; see {log}")
  try:
    value = json.loads(result.stdout)
  except json.JSONDecodeError as error:
    raise RuntimeError(f"Windows Platform state was not valid JSON; see {log}") from error
  service_keys = {
      "configuration", "display_name", "name", "path_name", "service_type",
      "start_mode", "start_name", "state",
  }
  device_keys = {
      "class", "driver_inf", "driver_provider", "driver_signer", "driver_version",
      "friendly_name", "instance_id", "is_signed", "manufacturer", "status",
  }
  service = value.get("service") if isinstance(value, dict) else None
  devices = value.get("pnp_devices") if isinstance(value, dict) else None
  if (not isinstance(value, dict) or set(value) != {"pnp_devices", "service"}
      or not isinstance(service, dict)
      or set(service) != service_keys or service["name"] != "SiMaMlaDaemon"
      or not isinstance(service["configuration"], dict)
      or not service["configuration"] or not service["path_name"]
      or not service["start_mode"] or not service["start_name"]
      or service["state"] != "Running"
      or not isinstance(devices, list) or not devices):
    raise RuntimeError(f"Windows Platform state was incomplete; see {log}")
  names = {"SiMa.ai MLA Device", "SiMa.ai Virtual Ethernet Adapter"}
  if (any(not isinstance(device, dict) for device in devices)
      or {re.sub(r" #[0-9]+$", "", str(device.get("friendly_name")))
          for device in devices} != names
      or len({str(device.get("instance_id", "")).lower() for device in devices})
      != len(devices)):
    raise RuntimeError(f"Windows Platform PnP identity was incomplete; see {log}")
  for device in devices:
    if (not isinstance(device, dict) or set(device) != device_keys
        or not device["instance_id"] or not device["driver_version"]
        or not device["driver_inf"] or not device["driver_provider"]
        or not isinstance(device["driver_signer"], str)
        or not isinstance(device["is_signed"], bool)
        or device["status"] != "OK"):
      raise RuntimeError(f"Windows Platform PnP identity was invalid; see {log}")
  return value


def require_restored_windows_host_state(baseline: dict, final: dict) -> None:
  if final != baseline:
    raise RuntimeError(
        "Windows Platform/GStreamer/runtime dependencies, Platform PnP identity, "
        "or SiMaMlaDaemon configuration/state changed during qualification")


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
# Driver-signing policy is recorded, not gated; a query that fails or needs
# elevation records "unknown: <reason>".
$driverSigning = [ordered]@{}
try {
  $driverSigning.secure_boot = [bool](Confirm-SecureBootUEFI -ErrorAction Stop)
} catch {
  $driverSigning.secure_boot = "unknown: $($_.Exception.Message)"
}
$bcdOutput = & "$env:SystemRoot\System32\bcdedit.exe" /enum '{current}' 2>&1
$bcdExit = $LASTEXITCODE
$bcdLines = @($bcdOutput | ForEach-Object { "$_" })
foreach ($pair in @(('test_signing', 'testsigning'),
                    ('no_integrity_checks', 'nointegritychecks'))) {
  $match = $bcdLines | Select-String -Pattern "^$($pair[1])\s+(\S+)" | Select-Object -First 1
  $driverSigning[$pair[0]] = if ($bcdExit -ne 0) {
    "unknown: bcdedit exited ${bcdExit}: $(($bcdLines -join ' ').Trim())"
  } elseif (-not $match) {
    $false
  } elseif ($match.Matches[0].Groups[1].Value -eq 'Yes') {
    $true
  } elseif ($match.Matches[0].Groups[1].Value -eq 'No') {
    $false
  } else {
    "unknown: bcdedit reported $($match.Line.Trim())"
  }
}
try {
  $deviceGuard = Get-CimInstance -Namespace 'root\Microsoft\Windows\DeviceGuard' `
    -ClassName Win32_DeviceGuard -ErrorAction Stop
  $policyStates = @{ 0 = 'off'; 1 = 'audit'; 2 = 'enforced' }
  foreach ($pair in @(('wdac_kernel_policy', 'CodeIntegrityPolicyEnforcementStatus'),
                      ('wdac_user_mode_policy', 'UsermodeCodeIntegrityPolicyEnforcementStatus'))) {
    $status = $deviceGuard.($pair[1])
    $driverSigning[$pair[0]] = if ($null -ne $status -and $policyStates.ContainsKey([int]$status)) {
      $policyStates[[int]$status]
    } else {
      "unknown: $($pair[1]) is '$status'"
    }
  }
  $driverSigning.hvci_running = @($deviceGuard.SecurityServicesRunning) -contains 2
} catch {
  foreach ($name in 'wdac_kernel_policy', 'wdac_user_mode_policy', 'hvci_running') {
    $driverSigning[$name] = "unknown: $($_.Exception.Message)"
  }
}
[pscustomobject]@{
  account = $identity.Name
  architecture = $os.OSArchitecture
  build_number = $os.BuildNumber
  caption = $os.Caption
  driver_signing = $driverSigning
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
      "account", "architecture", "build_number", "caption", "driver_signing",
      "is_administrator", "product_type", "smart_app_control_state",
      "applocker_rule_collections", "version",
  }
  driver_signing = {
      "secure_boot", "test_signing", "no_integrity_checks", "wdac_kernel_policy",
      "wdac_user_mode_policy", "hvci_running",
  }
  if (not isinstance(value, dict) or set(value) != required
      or not isinstance(value["driver_signing"], dict)
      or set(value["driver_signing"]) != driver_signing):
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


def _windows_openssh_batch_line(argv: list[str], output: Path, *,
                                input_file: Path | None = None) -> str:
  """Redirect OpenSSH before Python gives it redirected standard handles."""
  powershell_arguments = ",".join(
      "'" + value.replace("'", "''") + "'" for value in argv)
  powershell = (
      "$env:c28fc6f98a2c44abbbd89d6a3037d0d9_POSIX_FD_STATE=$null;"
      f"$argv=@({powershell_arguments});$exe=$argv[0];"
      "[string[]]$rest=$argv[1..($argv.Length-1)];"
      "& $exe @rest;exit $LASTEXITCODE"
  )
  encoded = base64.b64encode(powershell.encode("utf-16le")).decode("ascii")
  command = (
      "powershell.exe -NoLogo -NoProfile -NonInteractive -EncodedCommand "
      + encoded + " > " + subprocess.list2cmdline([str(output)]) + " 2>&1"
  )
  if input_file is not None:
    command += " < " + subprocess.list2cmdline([str(input_file)])
  command = command.replace("%", "%%")
  return command


def _windows_relay_safe_remote_command(remote: str) -> str:
  """Carry an OpenSSH remote command through Windows PowerShell 5.1 unchanged.

  Windows PowerShell 5.1 does not escape double quotes inside native-command
  arguments, so ssh.exe would receive the command without them.  Base64 and
  this decoder contain no double quote or backslash; the card's login shell
  decodes the exact command and runs it as sshd would, with the same stdin.
  """
  encoded = base64.b64encode(remote.encode("utf-8")).decode("ascii")
  return (f"set -f; IFS=; remote=$(printf %s {encoded} | base64 -d) || exit; "
          "exec $SHELL -c $remote")


def prepare_openssh_secret_directory(path: Path) -> Path:
  """Create a qualification-owned directory whose contents only the runner can read."""
  root = path.resolve()
  root.mkdir(mode=0o700, parents=True, exist_ok=True)
  if sys.platform == "win32":
    script = r"""
$ErrorActionPreference = 'Stop'
$sid = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
& icacls.exe $env:SIMA_OPENSSH_SECRET_ROOT /inheritance:r `
  /grant:r "*$($sid):(OI)(CI)(F)" /Q | Out-Null
if ($LASTEXITCODE -ne 0) { throw "icacls failed: $LASTEXITCODE" }
"""
    environment = os.environ.copy()
    environment["SIMA_OPENSSH_SECRET_ROOT"] = str(root)
    subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
         "-Command", script],
        env=environment, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, timeout=30, check=True)
  else:
    root.chmod(0o700)
  return root


def run_windows_openssh(command: list[object], log: Path, *,
                        input_value: str = "",
                        secret_root: Path | None = None,
                        env: dict[str, str] | None = None,
                        cwd: Path | None = None,
                        timeout: int = 3600,
                        redact_stdin: bool = False) -> subprocess.CompletedProcess[str]:
  """Run Windows OpenSSH through a batch file with direct file redirection.

  Inside a non-terminal Windows OpenSSH server session, ssh.exe/scp.exe inherit
  c28fc6f98a2c44abbbd89d6a3037d0d9_POSIX_FD_STATE, treat a synchronous pipe as
  overlapped, and block once a write to it cannot complete at once.  The
  PowerShell relay re-pipes their output, so it drops that inherited state
  first; the batch wrapper then writes the relayed output to an ordinary file.
  """
  argv = [str(value) for value in command]
  relayed = argv
  if Path(argv[0]).name.lower() == "ssh.exe":
    # ssh_command() puts the remote command last.
    relayed = [*argv[:-1], _windows_relay_safe_remote_command(argv[-1])]
  wrapper = log.with_name(f".{log.name}.openssh.cmd")
  child_output = log.with_name(f".{log.name}.openssh-output.log")
  child_input = None
  if wrapper.exists() or child_output.exists():
    raise RuntimeError(f"stale Windows OpenSSH wrapper state exists beside {log}")
  if "\n" in input_value or "\r" in input_value:
    raise ValueError("Windows OpenSSH input secret must be one line")
  process: subprocess.Popen[str] | None = None
  timed_out: subprocess.TimeoutExpired | None = None
  try:
    if input_value:
      if secret_root is None:
        raise ValueError("Windows OpenSSH secret input requires a private directory")
      secret_directory = prepare_openssh_secret_directory(secret_root)
      child_input = secret_directory / (secrets.token_hex(16) + ".openssh-input")
      child_input.write_bytes(input_value.encode("utf-8") + b"\n")
    wrapper.write_text(
        "@echo off\r\n"
        + _windows_openssh_batch_line(
            relayed, child_output,
            input_file=child_input)
        + "\r\nexit /b %errorlevel%\r\n",
        encoding="utf-8",
    )
    process = subprocess.Popen(
        ["cmd.exe", "/d", "/c", str(wrapper)],
        cwd=cwd,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    try:
      process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
      timed_out = error
      subprocess.run(
          ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
          stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
          stderr=subprocess.DEVNULL, check=False, timeout=30)
      process.communicate(timeout=30)
  finally:
    captured = child_output.read_text(
        encoding="utf-8", errors="replace") if child_output.is_file() else ""
    suffix = " [stdin redacted]" if redact_stdin else ""
    try:
      log.write_text(
          "+ " + subprocess.list2cmdline(argv) + suffix + "\n" + captured,
          encoding="utf-8", errors="replace")
    finally:
      wrapper.unlink(missing_ok=True)
      child_output.unlink(missing_ok=True)
      if child_input is not None:
        child_input.unlink(missing_ok=True)
  if timed_out is not None:
    raise timed_out
  if process is None:
    raise RuntimeError("Windows OpenSSH wrapper did not start")
  return subprocess.CompletedProcess(argv, process.returncode, captured, None)


def _run_to_log(argv: list[str], log: Path, env: dict[str, str] | None,
                cwd: Path | None, timeout: int) -> int:
  with log.open("w", encoding="utf-8", errors="replace") as output:
    output.write("+ " + subprocess.list2cmdline(argv) + "\n")
    output.flush()
    return subprocess.run(
        argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=output,
        stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False).returncode


def run_logged(command: list[object], log: Path, *, env: dict[str, str] | None = None,
               cwd: Path | None = None, timeout: int = 3600) -> None:
  argv = [str(value) for value in command]
  if Path(argv[0]).name.lower() in {"scp.exe", "ssh.exe"}:
    returncode = run_windows_openssh(argv, log, env=env, cwd=cwd, timeout=timeout).returncode
  else:
    returncode = _run_to_log(argv, log, env, cwd, timeout)
  if returncode not in PASS_CODES:
    raise RuntimeError(f"command failed with {returncode}; see {log}")


def imports_pyneatpcie(python: Path) -> bool:
  return subprocess.run(
      [python, "-c", "import pyneatpcie"], stdout=subprocess.DEVNULL,
      stderr=subprocess.DEVNULL, check=False).returncode == 0


def run_expected_failure(command: list[object], log: Path,
                         *, env: dict[str, str] | None = None,
                         cwd: Path | None = None, timeout: int = 3600) -> int:
  returncode = _run_to_log([str(value) for value in command], log, env, cwd, timeout)
  if returncode in PASS_CODES:
    raise RuntimeError(f"command unexpectedly succeeded; see {log}")
  return returncode


def customer_process_environment(
    base: dict[str, str] | None = None) -> dict[str, str]:
  """Return a normal Windows process environment without CI runtime overlays."""
  environment = dict(os.environ if base is None else base)
  system_root = next((
      value for name, value in environment.items()
      if name.upper() == "SYSTEMROOT"
  ), None)
  if not system_root:
    raise RuntimeError("customer process activation requires SystemRoot")
  removed = {
      "CMAKE_PREFIX_PATH",
      "GSTREAMER_ROOT_X86_64",
      "GST_PLUGIN_PATH",
      "GST_PLUGIN_PATH_1_0",
      "PKG_CONFIG",
      "PKG_CONFIG_PATH",
      "PYTHONPATH",
      "SIMA_PCIE_HOST_ENVIRONMENT",
      "SIMA_PCIE_HOST_LIBRARY_PATH",
      "SIMA_PCIE_HOST_ROOT",
      "SIMA_WINDOWS_CMAKE_PREFIX_PATH",
      "SIMA_WINDOWS_DEPENDENCY_RUNTIME_PATH",
      "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR",
  }
  for name in list(environment):
    if name.upper() in removed or name.upper() == "PATH":
      environment.pop(name)
  windows = Path(system_root)
  environment["PATH"] = os.pathsep.join((
      str(windows / "System32"),
      str(windows),
      str(windows / "System32" / "Wbem"),
      str(windows / "System32" / "WindowsPowerShell" / "v1.0"),
      str(windows / "System32" / "OpenSSH"),
  ))
  return environment


def isolated_sima_cli_environment(customer_home: Path, wheelhouse: Path,
                                  work: Path) -> dict[str, str]:
  """Confine sima-cli receipts, managed Python, and agent skills to this run."""
  home = customer_home.resolve()
  disposable = work.resolve()
  if not home.is_relative_to(disposable):
    raise ValueError("sima-cli customer profile must be inside the disposable work tree")
  environment = os.environ.copy()
  environment.pop("PYTHONPATH", None)
  environment.update({
      "CLAUDE_HOME": str(home / ".claude"),
      "CODEX_HOME": str(home / ".codex"),
      "HOME": str(home),
      "PIP_CACHE_DIR": str(home / ".cache" / "pip"),
      "PIP_FIND_LINKS": str(wheelhouse),
      "PIP_NO_INDEX": "1",
      "SIMA_CLI_CHECK_FOR_UPDATE": "0",
      "SIMA_CLI_HOME": str(home / ".sima-cli"),
      "USERPROFILE": str(home),
  })
  # Skill and sima-cli state live under the work tree, so its removal proves none remains.
  return environment


def verify_customer_process_activation(install_root: Path, python: Path,
                                       evidence: Path) -> None:
  """Prove the installed SDK works without producer or CI PATH injection."""
  doctor = install_root / "bin" / "sima-pcie-host-doctor.cmd"
  if not doctor.is_file():
    raise RuntimeError(f"installed Core MSI is missing its diagnostic: {doctor}")
  environment = customer_process_environment()
  command_processor = next((
      value for name, value in environment.items()
      if name.upper() == "COMSPEC"
  ), None)
  if not command_processor:
    system_root = next(
        value for name, value in environment.items()
        if name.upper() == "SYSTEMROOT")
    command_processor = str(Path(system_root) / "System32" / "cmd.exe")
  run_logged(
      [command_processor, "/d", "/c", doctor],
      evidence / "customer-process-doctor.log", env=environment, timeout=60)
  run_logged(
      [python, "-c",
       "import pyneatpcie as p; assert p.Model and p.Runtime and p.Tensor"],
      evidence / "customer-process-python-import.log", env=environment, timeout=60)


def remove_qualification_tree(path: Path, description: str) -> None:
  """Remove one qualification-owned tree and fail if any residue remains."""
  try:
    shutil.rmtree(path)
  except FileNotFoundError:
    return
  if path.exists():
    raise RuntimeError(f"{description} was not removed: {path}")


def standard_user_wrapper() -> str:
  """Return a static wrapper; all paths and child inputs stay out of batch text.

  The paths arrive as arguments: the child gets the new user's environment, not ours.
  Order: doctor, run log, runner, Python, driver, payload, receipt.
  """
  return (
      "@echo off\n"
      "setlocal\n"
      "set \"SIMAPCIE_GENAI_MEDIA_DIRECTORY=\"\n"
      "set \"SIMAPCIE_GENAI_MEDIA_SERVE_ROOT=\"\n"
      "call \"%~1\" > \"%~2\" 2>&1\n"
      "if not \"%errorlevel%\"==\"0\" exit /b 98\n"
      "call \"%~3\" \"%~4\" \"%~5\" \"%~6\" \"%~7\" >> \"%~2\" 2>&1\n"
      "if not \"%errorlevel%\"==\"0\" exit /b 99\n"
      "exit /b 0\n"
  )


def create_standard_user_python(wheel: Path, wheelhouse: Path, portable_root: Path,
                                evidence: Path) -> Path:
  """Create a fresh venv and force the candidate wheel into that environment."""
  subprocess.run(
      [sys.executable, "-m", "venv", "--copies", portable_root], check=True)
  python = portable_root / "Scripts" / "python.exe"
  if not python.is_file():
    raise RuntimeError("fresh standard-user Python environment has no python.exe")
  run_logged(
      [python, "-m", "pip", "install", "--no-index", "--find-links", wheelhouse,
       "--force-reinstall", wheel],
      evidence / "standard-user-python-install.log", timeout=300)
  return python


def verify_standard_user_process_activation(install_root: Path, wheel: Path,
                                            wheelhouse: Path, candidate_root: Path,
                                            work: Path, evidence: Path,
                                            hardware_environment: dict[str, str],
                                            include_genai: bool,
                                            timeout_seconds: int) -> dict:
  """Run the installed diagnostics and Python API as a disposable standard user."""
  doctor = install_root / "bin" / "sima-pcie-host-doctor.cmd"
  runner = install_root / "bin" / "sima-pcie-host-run.cmd"
  if not doctor.is_file():
    raise RuntimeError(f"installed Core MSI is missing its diagnostic: {doctor}")
  if not runner.is_file():
    raise RuntimeError(f"installed Core MSI is missing its launcher: {runner}")
  program_data = os.environ.get("PROGRAMDATA")
  if not program_data:
    raise RuntimeError("standard-user acceptance requires ProgramData")
  portable_root = Path(program_data) / (
      "SimaPCIeHostAcceptance-" + secrets.token_hex(8))
  driver_source = Path(__file__).with_name("standard_user_genai.py")
  if not driver_source.is_file():
    raise RuntimeError(f"standard-user GenAI driver is missing: {driver_source}")
  source_key = Path(hardware_environment.get("SIMAPCIE_SSH_KEY", ""))
  if include_genai and not source_key.is_file():
    raise RuntimeError(f"standard-user GenAI SSH key is unavailable: {source_key}")
  test_root = candidate_root / "hardware-tests" / "lib" / "sima-pcie-host" / "tests"
  executable = test_root / "test_genai_run.exe"
  if include_genai and not executable.is_file():
    raise RuntimeError(f"candidate is missing standard-user GenAI hardware test: {executable}")
  driver = work / "standard-user-genai.py"
  shutil.copy2(driver_source, driver)
  receipt_path = work / "standard-user-genai.json"
  payload_path = work / "standard-user-genai-input.json"
  private_root = work / ("standard-user-private-" + secrets.token_hex(8))
  private_root.mkdir()
  private_key = private_root / "card-ssh-key"
  try:
    if include_genai:
      shutil.copy2(source_key, private_key)
    child_environment = {
        name: hardware_environment[name]
        for name in STANDARD_USER_GENAI_ENVIRONMENT
        if hardware_environment.get(name)
    }
    if include_genai:
      child_environment["SIMAPCIE_SSH_KEY"] = str(private_key)
    write_json(payload_path, {
        "environment": child_environment,
        "executable": str(executable),
        "include_genai": include_genai,
        "timeout_seconds": timeout_seconds,
    })
    wrapper = work / "standard-user-activation.cmd"
    run_log = work / "standard-user-activation.log"
    wrapper.write_text(standard_user_wrapper(), encoding="utf-8")
  except BaseException as error:
    try:
      remove_qualification_tree(private_root, "temporary standard-user private input")
    except BaseException as cleanup_error:
      error.add_note(str(cleanup_error))
    raise
  try:
    python = create_standard_user_python(wheel, wheelhouse, portable_root, evidence)
  except BaseException as error:
    for cleanup_path in (portable_root, private_root):
      try:
        remove_qualification_tree(cleanup_path, "temporary standard-user setup")
      except BaseException as cleanup_error:
        error.add_note(str(cleanup_error))
    raise
  script = r"""
$ErrorActionPreference = 'Stop'
# The disposable environment's launcher runs the base Python named by pyvenv.cfg "home".
$pythonHomeLine = Get-Content -LiteralPath (
    Join-Path $env:SIMA_STANDARD_USER_PYTHON_ROOT 'pyvenv.cfg') |
  Where-Object { $_ -match '^\s*home\s*=' } | Select-Object -First 1
$pythonHome = ($pythonHomeLine -split '=', 2)[1].Trim()
$readAndExecute = [Security.AccessControl.FileSystemRights]::ReadAndExecute
$standardUserSids = 'S-1-1-0', 'S-1-5-4', 'S-1-5-11', 'S-1-5-32-545'
$pythonReadable = @((Get-Acl -LiteralPath $pythonHome).GetAccessRules(
    $true, $true, [Security.Principal.SecurityIdentifier]) | Where-Object {
  $_.AccessControlType -eq 'Allow' -and
  -not $_.PropagationFlags.HasFlag([Security.AccessControl.PropagationFlags]::InheritOnly) -and
  $_.FileSystemRights.HasFlag($readAndExecute) -and
  $standardUserSids -contains $_.IdentityReference.Value
}).Count -ne 0
if (-not $pythonReadable) {
  throw ("standard users cannot read the base Python at $pythonHome; run the qualifier " +
    'with a Python installed for all users, for example under Program Files')
}
$name = 'sna-' + $PID
$profilePath = Join-Path $env:SystemDrive "Users\$name"
$bytes = New-Object byte[] 32
$rng = [Security.Cryptography.RandomNumberGenerator]::Create()
$rng.GetBytes($bytes)
$rng.Dispose()
$plain = [Convert]::ToBase64String($bytes) + 'aA1!'
$secure = ConvertTo-SecureString $plain -AsPlainText -Force
$created = $false
$windowAccessGranted = $false
try {
  $user = New-LocalUser -Name $name -Password $secure -AccountNeverExpires `
    -PasswordNeverExpires -UserMayNotChangePassword
  $created = $true
  $account = "$env:COMPUTERNAME\$name"
  $admin = @(Get-LocalGroupMember -Group 'Administrators' -ErrorAction Stop |
    Where-Object { $_.SID -eq $user.SID }).Count -ne 0
  if ($admin) { throw 'temporary acceptance user unexpectedly belongs to Administrators' }
  & net.exe user $name /passwordreq:yes | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "could not require the temporary password: $LASTEXITCODE" }
  & icacls.exe $env:SIMA_STANDARD_USER_WORK /grant "${account}:(OI)(CI)(M)" /T /Q | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "icacls failed: $LASTEXITCODE" }
  & icacls.exe $env:SIMA_STANDARD_USER_PRIVATE /inheritance:r /grant:r `
    "${account}:(OI)(CI)(RX)" "*S-1-5-18:(OI)(CI)(F)" `
    "*S-1-5-32-544:(OI)(CI)(F)" /T /Q | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "private input icacls failed: $LASTEXITCODE" }
  & icacls.exe $env:SIMA_STANDARD_USER_PYTHON_ROOT /grant "${account}:(OI)(CI)(RX)" /T /Q |
    Out-Null
  if ($LASTEXITCODE -ne 0) { throw "Python icacls failed: $LASTEXITCODE" }
  Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Security.AccessControl;
using System.Security.Principal;
using System.Text;
public static class SimaStandardUserProcess {
  [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
  public struct StartupInfo {
    public int cb; public string reserved; public string desktop; public string title;
    public int x; public int y; public int xSize; public int ySize;
    public int xCountChars; public int yCountChars; public int fillAttribute;
    public int flags; public short showWindow; public short reserved2;
    public IntPtr reserved2Pointer; public IntPtr standardInput;
    public IntPtr standardOutput; public IntPtr standardError;
  }
  [StructLayout(LayoutKind.Sequential)]
  public struct ProcessInformation {
    public IntPtr process; public IntPtr thread; public int processId; public int threadId;
  }
  [DllImport("advapi32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern bool CreateProcessWithLogonW(string user, string domain, string password,
    int logonFlags, string application, StringBuilder command, int creationFlags,
    IntPtr environment, string currentDirectory, ref StartupInfo startup,
    out ProcessInformation process);
  [DllImport("kernel32.dll", SetLastError = true)] static extern uint WaitForSingleObject(
    IntPtr handle, uint milliseconds);
  [DllImport("kernel32.dll", SetLastError = true)] static extern bool GetExitCodeProcess(
    IntPtr process, out uint exitCode);
  [DllImport("kernel32.dll", SetLastError = true)] static extern bool TerminateProcess(
    IntPtr process, uint exitCode);
  [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr handle);
  [DllImport("kernel32.dll")] static extern uint GetCurrentThreadId();
  [DllImport("user32.dll", SetLastError = true)] static extern IntPtr GetProcessWindowStation();
  [DllImport("user32.dll", SetLastError = true)] static extern IntPtr GetThreadDesktop(
    uint threadId);
  [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern bool GetUserObjectInformation(IntPtr handle, int index, StringBuilder name,
    uint length, out uint needed);
  [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern IntPtr OpenWindowStation(string name, bool inherit, uint access);
  [DllImport("user32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  static extern IntPtr OpenDesktop(string name, uint flags, bool inherit, uint access);
  [DllImport("user32.dll")] static extern bool CloseWindowStation(IntPtr handle);
  [DllImport("user32.dll")] static extern bool CloseDesktop(IntPtr handle);
  [DllImport("user32.dll", SetLastError = true)] static extern bool GetUserObjectSecurity(
    IntPtr handle, ref uint information, byte[] descriptor, uint length, out uint needed);
  [DllImport("user32.dll", SetLastError = true)] static extern bool SetUserObjectSecurity(
    IntPtr handle, ref uint information, byte[] descriptor);
  static void Fail(string action) {
    int error = Marshal.GetLastWin32Error();
    throw new Win32Exception(error, "could not " + action + " for the standard-user child: " +
      "Win32 error " + error + ": " + new Win32Exception(error).Message);
  }
  static string ObjectName(IntPtr handle) {
    var name = new StringBuilder(256);
    uint needed;
    if (handle == IntPtr.Zero || !GetUserObjectInformation(handle, 2, name, 512, out needed))
      Fail("name this session's window station or desktop");
    return name.ToString();
  }
  // Removes any ACE for the user, then adds one allow ACE when rights is non-zero.
  static void SetAccess(IntPtr handle, SecurityIdentifier user, int rights) {
    uint information = 4, needed;  // DACL_SECURITY_INFORMATION
    GetUserObjectSecurity(handle, ref information, null, 0, out needed);
    var bytes = new byte[needed];
    if (needed == 0 || !GetUserObjectSecurity(handle, ref information, bytes, needed, out needed))
      Fail("read a window object's DACL");
    var descriptor = new RawSecurityDescriptor(bytes, 0);
    var acl = descriptor.DiscretionaryAcl;
    if (acl == null) return;  // a null DACL already admits every user
    for (int i = acl.Count - 1; i >= 0; i--) {
      var ace = acl[i] as KnownAce;
      if (ace != null && ace.SecurityIdentifier == user) acl.RemoveAce(i);
    }
    if (rights != 0) acl.InsertAce(acl.Count, new CommonAce(
      AceFlags.None, AceQualifier.AccessAllowed, rights, user, false, null));
    descriptor.DiscretionaryAcl = acl;
    bytes = new byte[descriptor.BinaryLength];
    descriptor.GetBinaryForm(bytes, 0);
    if (!SetUserObjectSecurity(handle, ref information, bytes))
      Fail("write a window object's DACL");
  }
  // The child inherits this window station and desktop. In session 0 they admit only the
  // caller, so user32 cannot initialise in the child (0xC0000142). The rights are the
  // specific rights of Microsoft's WINSTA_ALL and DESKTOP_ALL plus READ_CONTROL, without
  // DELETE, WRITE_DAC or WRITE_OWNER.
  static void SetSessionWindowAccess(string sid, bool grant) {
    var user = new SecurityIdentifier(sid);
    const uint access = 0x00060000;  // READ_CONTROL | WRITE_DAC
    var station = OpenWindowStation(ObjectName(GetProcessWindowStation()), false, access);
    if (station == IntPtr.Zero) Fail("open this session's window station");
    try { SetAccess(station, user, grant ? 0x0002037F : 0); }
    finally { CloseWindowStation(station); }
    var desktop = OpenDesktop(
      ObjectName(GetThreadDesktop(GetCurrentThreadId())), 0, false, access);
    if (desktop == IntPtr.Zero) Fail("open this session's desktop");
    try { SetAccess(desktop, user, grant ? 0x000201FF : 0); }
    finally { CloseDesktop(desktop); }
  }
  public static void GrantSessionWindowAccess(string sid) { SetSessionWindowAccess(sid, true); }
  public static void RevokeSessionWindowAccess(string sid) { SetSessionWindowAccess(sid, false); }
  public static uint Run(string user, string domain, string password, string application,
                         string arguments, string currentDirectory, uint timeoutMilliseconds) {
    var startup = new StartupInfo(); startup.cb = Marshal.SizeOf(startup);
    ProcessInformation process;
    var command = new StringBuilder("\"" + application + "\" " + arguments);
    if (!CreateProcessWithLogonW(user, domain, password, 1, application, command,
        0x08000000, IntPtr.Zero, currentDirectory, ref startup, out process))
      throw new Win32Exception(Marshal.GetLastWin32Error());
    try {
      uint wait = WaitForSingleObject(process.process, timeoutMilliseconds);
      if (wait == 258) {
        TerminateProcess(process.process, 124);
        WaitForSingleObject(process.process, 10000);
        throw new TimeoutException("standard-user process timed out");
      }
      if (wait != 0) throw new Win32Exception(Marshal.GetLastWin32Error());
      uint exitCode;
      if (!GetExitCodeProcess(process.process, out exitCode))
        throw new Win32Exception(Marshal.GetLastWin32Error());
      return exitCode;
    } finally { CloseHandle(process.thread); CloseHandle(process.process); }
  }
}
'@
  $windowAccessGranted = $true
  [SimaStandardUserProcess]::GrantSessionWindowAccess([string]$user.SID)
  $wrapperArguments = foreach ($path in @(
      $env:SIMA_STANDARD_USER_DOCTOR, $env:SIMA_STANDARD_USER_RUN_LOG,
      $env:SIMA_STANDARD_USER_RUNNER, $env:SIMA_STANDARD_USER_PYTHON,
      $env:SIMA_STANDARD_USER_DRIVER, $env:SIMA_STANDARD_USER_PAYLOAD,
      $env:SIMA_STANDARD_USER_RECEIPT)) { ' "' + $path + '"' }
  $arguments = '/d /c ""' + $env:SIMA_STANDARD_USER_WRAPPER + '"' +
    ($wrapperArguments -join '') + '"'
  if ($env:ComSpec.Length + 3 + $arguments.Length -gt 1024) {
    throw ('standard-user command line exceeds the 1024 characters ' +
      'CreateProcessWithLogonW accepts; use a shorter work directory')
  }
  $timeoutMilliseconds = [uint32]([uint64]$env:SIMA_STANDARD_USER_TIMEOUT_SECONDS * 1000)
  $exitCode = [SimaStandardUserProcess]::Run(
    $name, $env:COMPUTERNAME, $plain, $env:ComSpec, $arguments,
    $env:SIMA_STANDARD_USER_WORK, $timeoutMilliseconds)
  if ($exitCode -ne 0) {
    $runLog = Join-Path $env:SIMA_STANDARD_USER_WORK 'standard-user-activation.log'
    $details = if (Test-Path -LiteralPath $runLog) {
      Get-Content -LiteralPath $runLog -Raw
    } else { 'no standard-user process log' }
    throw "standard-user activation failed: $exitCode`n$details"
  }
  [pscustomobject]@{
    account = $account
    is_administrator = $false
    doctor = 'PASS'
    python = 'PASS'
  } | ConvertTo-Json -Compress
} finally {
  if ($created) {
    $cleanupErrors = [Collections.Generic.List[string]]::new()
    if ($windowAccessGranted) {
      try { [SimaStandardUserProcess]::RevokeSessionWindowAccess([string]$user.SID) }
      catch { $cleanupErrors.Add("window access removal failed: $($_.Exception.Message)") }
    }
    $profileWaitDeadline = [DateTime]::UtcNow.AddSeconds(5)
    do {
      $profile = Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue |
        Where-Object { $_.SID -eq [string]$user.SID } | Select-Object -First 1
      if ($profile -and $profile.Loaded) { Start-Sleep -Milliseconds 250 }
    } while ($profile -and $profile.Loaded -and
             [DateTime]::UtcNow -lt $profileWaitDeadline)
    if ($profile -and $profile.Loaded) {
      & reg.exe unload "HKU\$([string]$user.SID)" | Out-Null
      if ($LASTEXITCODE -ne 0) {
        $cleanupErrors.Add('temporary user registry hive remained loaded')
      }
      $profile = Get-CimInstance Win32_UserProfile -ErrorAction SilentlyContinue |
        Where-Object { $_.SID -eq [string]$user.SID } | Select-Object -First 1
    }
    if ($profile -and $profile.Loaded) {
      $cleanupErrors.Add('temporary user profile remained loaded')
    } elseif ($profile) {
      try { Remove-CimInstance -InputObject $profile -ErrorAction Stop }
      catch { $cleanupErrors.Add("temporary user profile removal failed: $($_.Exception.Message)") }
    }
    try { Remove-LocalUser -Name $name -ErrorAction Stop }
    catch { $cleanupErrors.Add("temporary user removal failed: $($_.Exception.Message)") }
    if (-not ($profile -and $profile.Loaded)) {
      $profileDeadline = [DateTime]::UtcNow.AddSeconds(30)
      while ((Test-Path -LiteralPath $profilePath) -and
             [DateTime]::UtcNow -lt $profileDeadline) { Start-Sleep -Milliseconds 250 }
      if (Test-Path -LiteralPath $profilePath) {
        $cleanupErrors.Add('temporary user profile was not removed')
      }
    }
    try {
      Remove-Item -LiteralPath $env:SIMA_STANDARD_USER_PYTHON_ROOT -Recurse -Force `
        -ErrorAction Stop
    } catch {
      $cleanupErrors.Add("temporary Python removal failed: $($_.Exception.Message)")
    }
    if ($cleanupErrors.Count -ne 0) { throw ($cleanupErrors -join '; ') }
  }
}
"""
  environment = os.environ.copy()
  environment.update({
      "SIMA_STANDARD_USER_DOCTOR": str(doctor),
      "SIMA_STANDARD_USER_DRIVER": str(driver),
      "SIMA_STANDARD_USER_PAYLOAD": str(payload_path),
      "SIMA_STANDARD_USER_PRIVATE": str(private_root),
      "SIMA_STANDARD_USER_PYTHON": str(python),
      "SIMA_STANDARD_USER_WORK": str(work),
      "SIMA_STANDARD_USER_RECEIPT": str(receipt_path),
      "SIMA_STANDARD_USER_RUN_LOG": str(run_log),
      "SIMA_STANDARD_USER_RUNNER": str(runner),
      "SIMA_STANDARD_USER_TIMEOUT_SECONDS": str(timeout_seconds + 60),
      "SIMA_STANDARD_USER_WRAPPER": str(wrapper),
      "SIMA_STANDARD_USER_PYTHON_ROOT": str(portable_root),
  })
  result = None
  control_error = None
  try:
    result = subprocess.run(
        ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
         "-Command", script],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        timeout=timeout_seconds + 120,
        env=environment, check=False)
  except BaseException as error:
    control_error = error
  finally:
    for cleanup_path, description in (
        (portable_root, "temporary standard-user Python"),
        (private_root, "temporary standard-user private input"),
    ):
      try:
        remove_qualification_tree(cleanup_path, description)
      except BaseException as cleanup_error:
        if control_error is None:
          control_error = cleanup_error
        else:
          control_error.add_note(str(cleanup_error))
  if control_error is not None:
    raise control_error
  if result is None:
    raise RuntimeError("standard-user acceptance control process did not start")
  (evidence / "standard-user-control.log").write_text(
      f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}",
      encoding="utf-8", errors="replace")
  if result.returncode != 0:
    raise RuntimeError("standard-user process activation failed; see acceptance logs")
  try:
    receipt = json.loads(result.stdout)
  except json.JSONDecodeError as error:
    raise RuntimeError("standard-user process activation returned invalid evidence") from error
  if (not isinstance(receipt, dict) or receipt.get("is_administrator") is not False
      or receipt.get("doctor") != "PASS" or receipt.get("python") != "PASS"):
    raise RuntimeError("standard-user process activation evidence is incomplete")
  try:
    child_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise RuntimeError("standard-user process returned no valid child receipt") from error
  if (not isinstance(child_receipt, dict) or child_receipt.get("python") != "PASS"
      or child_receipt.get("process_is_administrator") is not False
      or child_receipt.get("standard_user") != "PASS"):
    raise RuntimeError("standard-user Python evidence is incomplete")
  receipt.update(child_receipt)
  if include_genai:
    required_genai = {
        "config_binding", "connection_options", "default_root", "override_absence",
        "real_genai", "required_assets_readable", "sentinel", "sentinel_cleanup",
    }
    if any(receipt.get(name) != "PASS" for name in required_genai):
      raise RuntimeError("standard-user default-root GenAI evidence is incomplete")
  return receipt


def run_captured_to_file(command: list[object], log: Path, *,
                         timeout: int) -> subprocess.CompletedProcess[str]:
  """Capture Windows OpenSSH output through its batch-file workaround."""
  return run_windows_openssh(command, log, timeout=timeout)


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


def _openssh_options(args: argparse.Namespace) -> list[str]:
  return [
      "-i", str(args.ssh_key.resolve(strict=True)),
      "-o", "BatchMode=yes",
      "-o", "IdentitiesOnly=yes",
      "-o", "StrictHostKeyChecking=yes",
      "-o", "ConnectTimeout=10",
  ]


def ssh_command(args: argparse.Namespace, remote: str) -> list[str]:
  return ["ssh.exe", *_openssh_options(args), f"{args.card_user}@{args.card_host}", remote]


def scp_command(args: argparse.Namespace, sources: tuple[Path, ...], destination: str) -> list[str]:
  return ["scp.exe", *_openssh_options(args), *(str(path) for path in sources),
          f"{args.card_user}@{args.card_host}:{destination}"]


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
  if getattr(args, "card_sudo_password_stdin", False):
    password = getattr(args, "card_sudo_password_value", "")
  else:
    name = args.card_sudo_password_env
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
      raise ValueError("card sudo password environment variable name is invalid")
    password = os.environ.get(name, "")
  if not password or "\n" in password or "\r" in password:
    raise ValueError("card sudo password must be one non-empty line")
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
  result = run_windows_openssh(
      argv, log, input_value=_card_sudo_password(args),
      secret_root=Path(args.work) / ".openssh-secrets",
      timeout=timeout, redact_stdin=True)
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
  request = f"{staged['run']}/firmware-restore-request"
  active = f"{staged['run']}/firmware-active"
  restored = f"{staged['run']}/firmware-restored"
  failure = f"{staged['run']}/firmware-broker-error"
  broker_log = f"{staged['log']}/firmware-broker.log"
  broker_source = f"{staged['run']}/firmware-broker.sh"
  token = Path(staged["stage"]).name.rsplit(".", 1)[-1]
  if not re.fullmatch(r"[0-9a-f]{32}", token):
    raise ValueError("card candidate stage does not end in its lock token")
  broker = (
      "#!/bin/sh\n"
      "set -eu\n"
      f"source={staged['firmware_source']}; target={staged['firmware_installed']}; "
      f"candidate={staged['firmware_name']}; baseline={shlex.quote(baseline_firmware)}; "
      f"remoteproc={remoteproc}; request={request}; active={active}; restored={restored}; "
      f"failure={failure}; token={token}\n"
      "restore() { current=$(cat \"$remoteproc/firmware\"); "
      "if [ \"$current\" = \"$candidate\" ]; then printf stop > \"$remoteproc/state\"; "
      "printf '%s' \"$baseline\" > \"$remoteproc/firmware\"; "
      "printf start > \"$remoteproc/state\"; "
      "elif [ \"$current\" != \"$baseline\" ]; then return 75; fi; "
      "grep -qx \"$baseline\" \"$remoteproc/firmware\"; "
      "grep -qx running \"$remoteproc/state\"; rm -f \"$target\"; }; "
      "failed() { status=$?; trap - EXIT HUP INT TERM; restore >/dev/null 2>&1 || true; "
      "printf '%s\\n' \"$status\" > \"$failure\"; exit \"$status\"; }; "
      "trap failed EXIT HUP INT TERM; test -r \"$source\"; test ! -e \"$target\"; "
      "test ! -e \"$request\"; test ! -e \"$active\"; test ! -e \"$restored\"; "
      "install -m 0644 \"$source\" \"$target\"; "
      "printf stop > \"$remoteproc/state\"; printf '%s' \"$candidate\" > \"$remoteproc/firmware\"; "
      "printf start > \"$remoteproc/state\"; grep -qx \"$candidate\" \"$remoteproc/firmware\"; "
      "grep -qx running \"$remoteproc/state\"; printf '%s\\n' \"$candidate\" > \"$active\"; "
      "remaining=10800; while [ ! -f \"$request\" ]; do "
      "test \"$remaining\" -gt 0; remaining=$((remaining - 1)); sleep 1; done; "
      "test \"$(cat \"$request\")\" = \"$token\"; restore; "
      "printf '%s\\n' \"$baseline\" > \"$restored\"; trap - EXIT HUP INT TERM"
  )
  broker_sha256 = hashlib.sha256(broker.encode("utf-8")).hexdigest()
  encoded_broker = base64.b64encode(broker.encode("utf-8")).decode("ascii")
  stage_broker = (
      f"set -eu; umask 077; test ! -e {broker_source}; "
      f"printf '%s' {encoded_broker} | base64 -d > {broker_source}; "
      f"chmod 0500 {broker_source}; "
      f"test \"$(sha256sum {broker_source} | awk '{{print $1}}')\" = {broker_sha256}"
  )
  run_logged(
      ssh_command(args, stage_broker), evidence / "card-candidate-broker-stage.log", timeout=30)
  broker_root = f"/run/neat-pcie-card0-{token}-firmware-broker"
  command = (
      f"set -eu; test ! -e {request}; test ! -e {active}; test ! -e {failure}; "
      f"test -f {broker_source}; test ! -L {broker_source}; "
      f"test \"$(sha256sum {broker_source} | awk '{{print $1}}')\" = {broker_sha256}; "
      f"broker_root={broker_root}; test ! -e \"$broker_root\"; "
      f"install -m 0500 -o root -g root {broker_source} \"$broker_root\"; "
      "trap 'rm -f \"$broker_root\"' EXIT; "
      f"nohup \"$broker_root\" </dev/null > {broker_log} 2>&1 & pid=$!; "
      "remaining=120; while [ ! -r " + active + " ]; do "
      f"if [ -r {failure} ]; then cat {failure} >&2; exit 76; fi; "
      "kill -0 \"$pid\" 2>/dev/null; test \"$remaining\" -gt 0; "
      "remaining=$((remaining - 1)); sleep 1; done; "
      f"test \"$(cat {active})\" = {staged['firmware_name']}"
  )
  run_sudo_remote(args, command, evidence / "card-candidate-activate.log")


def restore_card_candidate(args: argparse.Namespace, token: str, evidence: Path) -> None:
  paths = card_candidate_paths(token)
  lock = _validate_remote_path(args.remote_lock_path)
  remoteproc = "/sys/class/remoteproc/remoteproc1"
  request = f"{paths['run']}/firmware-restore-request"
  active = f"{paths['run']}/firmware-active"
  restored = f"{paths['run']}/firmware-restored"
  failure = f"{paths['run']}/firmware-broker-error"
  request_restore = (
      "set -eu; "
      f"if [ -e {paths['stage']} ] || [ -L {paths['stage']} ]; then "
      f"test ! -L {paths['stage']}; test -f {paths['stage']}/.qualification-owner; "
      f"test ! -L {paths['stage']}/.qualification-owner; "
      f"test \"$(cat {paths['stage']}/.qualification-owner)\" = {token}; "
      f"if [ -r {lock}/baseline_firmware ]; then "
      f"baseline=$(cat {lock}/baseline_firmware); "
      "case \"$baseline\" in (*[!A-Za-z0-9._+-]*|'') exit 74;; esac; "
      f"if [ -r {active} ] && [ ! -r {restored} ] && [ ! -r {failure} ]; then "
      f"test ! -e {request}; printf '%s\\n' {token} > {request}; "
      "remaining=180; while [ ! -r " + restored + " ]; do "
      f"if [ -r {failure} ]; then break; fi; "
      "test \"$remaining\" -gt 0; remaining=$((remaining - 1)); sleep 1; done; fi; "
      f"if [ -r {restored} ]; then test \"$(cat {restored})\" = \"$baseline\"; fi; "
      f"test ! -e {paths['firmware_installed']}; "
      f"grep -qx \"$baseline\" {remoteproc}/firmware; "
      f"grep -qx running {remoteproc}/state; "
      f"else test ! -e {active}; test ! -e {restored}; test ! -e {failure}; "
      f"test ! -e {paths['firmware_installed']}; fi; "
      f"else test ! -e {lock}/baseline_firmware; "
      f"test ! -e {paths['firmware_installed']}; fi"
  )
  run_logged(
      ssh_command(args, request_restore),
      evidence / "card-candidate-restore.log", timeout=240)
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

  def may_hold_after_failed_acquire(self, log: Path) -> bool:
    """SSH can disconnect after the card stores the token; unknown means held."""
    if self.conflicted:
      return False
    try:
      return self.owner(log) == self.token
    except RuntimeError:
      return True

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
  if not re.search(r"(?:^|[_ -])3[.]0(?:[._-]|$)", values["platform_release"]):
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


def _control_platform_service(cmdlet: str, status: str, failure: str, settle: str,
                              log: Path) -> None:
  command = (
      "$service = Get-Service -Name 'SiMaMlaDaemon' -ErrorAction Stop; "
      f"{cmdlet} -InputObject $service -Force -ErrorAction Stop; "
      "$service.WaitForStatus("
      f"[System.ServiceProcess.ServiceControllerStatus]::{status}, "
      "[TimeSpan]::FromSeconds(30)); "
      "if ($service.Status -ne "
      f"[System.ServiceProcess.ServiceControllerStatus]::{status}) {{ "
      f"throw 'SiMaMlaDaemon did not {failure}' }}{settle}"
  )
  run_logged(
      [require_powershell(), "-NoLogo", "-NoProfile", "-NonInteractive",
       "-Command", command], log, timeout=60)


def restart_windows_platform_service(log: Path) -> None:
  _control_platform_service(
      "Restart-Service", "Running", "return to running state", "; Start-Sleep -Seconds 2", log)


def stop_windows_platform_service(log: Path) -> None:
  _control_platform_service("Stop-Service", "Stopped", "reach stopped state", "", log)


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


def _wait_for_no_card_process(args: argparse.Namespace, log: Path, timeout: int,
                              label: str, process: str, description: str) -> None:
  remote = f"printf '{label}='; pgrep -fc '(^|/)[{process[0]}]{process[1:]}([[:space:]]|$)' || true"
  deadline = time.monotonic() + timeout
  attempt = log.with_name(log.name + ".attempt")
  with log.open("w", encoding="utf-8", errors="replace") as output:
    try:
      while True:
        result = run_captured_to_file(
            ssh_command(args, remote), attempt, timeout=30)
        output.write(f"returncode={result.returncode} {result.stdout.strip()}\n")
        output.flush()
        match = re.search(rf"{label}=(\d+)", result.stdout)
        if result.returncode == 0 and match and int(match.group(1)) == 0:
          return
        if time.monotonic() >= deadline:
          raise RuntimeError(
              f"card {description} did not stop within {timeout} seconds; see {log}")
        time.sleep(1)
    finally:
      attempt.unlink(missing_ok=True)


def wait_for_no_card_workers(args: argparse.Namespace, log: Path, timeout: int) -> None:
  _wait_for_no_card_process(args, log, timeout, "workers", "pcie-pipeline-builder", "worker")


def wait_for_no_genai_workers(args: argparse.Namespace, log: Path, timeout: int) -> None:
  _wait_for_no_card_process(
      args, log, timeout, "genai_workers", "neat-pcie-genai-worker", "GenAI worker")


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


def launch_until_marker(command: list[object], log: Path, cwd: Path,
                        environment: dict[str, str], marker: str, timeout: int
                        ) -> tuple[subprocess.Popen, object, Path]:
  """Start a logged fault-injection client and wait until it reports readiness."""
  argv = [str(value) for value in command]
  stream = log.open("w", encoding="utf-8", errors="replace")
  stream.write("+ " + subprocess.list2cmdline(argv) + "\n")
  stream.flush()
  process = subprocess.Popen(
      argv, cwd=cwd, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True)
  try:
    wait_for_log(process, log, marker, timeout)
  except BaseException:
    if process.poll() is None:
      process.kill()
    process.wait(timeout=30)
    stream.close()
    raise
  return process, stream, log


def run_fault_recovery_tests(args: argparse.Namespace, test_root: Path, evidence: Path,
                             environment: dict[str, str]) -> list[str]:
  blocker = test_root / "test_queue_blocker.exe"
  recovery = test_root / "test_tensor_run.exe"
  completed = []

  def launch(label: str, *, terminate_on_pull_timeout: bool = False
             ) -> tuple[subprocess.Popen, object, Path]:
    command = [blocker, "--iterations", "0"]
    if terminate_on_pull_timeout:
      command.append("--terminate-on-pull-timeout")
    return launch_until_marker(
        command, evidence / f"cpp-{label}.log", test_root, environment,
        "queue is now occupied", 300)

  process, stream, log = launch(
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
  # Only the bounded pull timeout proves recovery; any other exit may be a crash.
  if returncode != 1 or "pull timed out; terminating the fault-injection client" not in \
      log.read_text(encoding="utf-8", errors="replace"):
    raise RuntimeError(f"client did not exit through its pull timeout after worker loss; see {log}")
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


def run_genai_fault_recovery_tests(
    args: argparse.Namespace, python: Path, evidence: Path,
    environment: dict[str, str]) -> list[str]:
  client = args.source_root / "pcie_host/python/tests/genai_fault_client.py"
  completed = []

  def launch(label: str) -> tuple[subprocess.Popen, object, Path]:
    return launch_until_marker(
        [python, client, "wait-for-failure"], evidence / f"python-{label}.log",
        args.source_root, environment, "GENAI_FAULT_READY", 900)

  def require_observed_failure(
      process: subprocess.Popen, stream, log: Path, label: str) -> None:
    try:
      try:
        returncode = process.wait(timeout=120)
      except subprocess.TimeoutExpired as error:
        raise RuntimeError(f"{label} did not terminate after fault injection") from error
      if returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {returncode}")
      stream.flush()
      observed = log.read_text(encoding="utf-8", errors="replace")
      if "GENAI_FAULT_OBSERVED:" not in observed:
        raise RuntimeError(f"{label} exited without recording the injected fault")
    finally:
      if process.poll() is None:
        process.kill()
        process.wait(timeout=30)
      stream.close()

  def recovery_smoke(label: str) -> None:
    run_logged(
        [python, client, "smoke"], evidence / f"python-{label}.log",
        env=environment, cwd=args.source_root, timeout=args.test_timeout_seconds)

  process, stream, log = launch("test_genai_worker_loss_recovery")
  try:
    remote = (
        "pkill -KILL -f '(^|/)[n]eat-pcie-genai-worker([[:space:]]|$)' "
        "2>/dev/null || true"
    )
    run_logged(
        ssh_command(args, remote), evidence / "genai-worker-loss-injection.log", timeout=30)
    require_observed_failure(process, stream, log, "GenAI worker-loss client")
  finally:
    if process.poll() is None:
      process.kill()
      process.wait(timeout=30)
      stream.close()
  wait_for_no_genai_workers(args, evidence / "genai-worker-loss-drain.log", 120)
  recovery_smoke("test_genai_restart_after_worker_loss")
  completed.extend((
      "test_genai_worker_loss_recovery", "test_genai_restart_after_worker_loss"))

  process, stream, log = launch("test_genai_service_loss_recovery")
  service_stop_attempted = False
  try:
    service_stop_attempted = True
    stop_windows_platform_service(evidence / "genai-service-loss-injection.log")
    require_observed_failure(process, stream, log, "GenAI service-loss client")
  finally:
    if process.poll() is None:
      process.kill()
      process.wait(timeout=30)
      stream.close()
    if service_stop_attempted:
      restart_windows_platform_service(evidence / "genai-service-loss-recovery.log")
  wait_for_no_genai_workers(args, evidence / "genai-service-loss-drain.log", 120)
  recovery_smoke("test_genai_restart_after_service_loss")
  completed.extend((
      "test_genai_service_loss_recovery", "test_genai_restart_after_service_loss"))
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
    media_directory = environment.get("SIMAPCIE_GENAI_MEDIA_DIRECTORY", "")
    media_serve_root = environment.get("SIMAPCIE_GENAI_MEDIA_SERVE_ROOT", "")
    if bool(media_directory) != bool(media_serve_root):
      raise RuntimeError(
          "Windows PCIe GenAI media overrides require directory and service root as a pair")
    if (media_directory and (not Path(media_directory).is_absolute()
                             or not Path(media_directory).is_dir())):
      raise RuntimeError(
          "Windows PCIe GenAI custom media override requires an existing absolute directory: "
          "SIMAPCIE_GENAI_MEDIA_DIRECTORY")
    if media_serve_root and "\0" in media_serve_root:
      raise RuntimeError("Windows PCIe GenAI custom media service root is invalid")


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
        "test_genai_bounded_cancellation_soak_and_context_cleanup",
        "test_genai_competing_sessions_are_isolated",
        "test_genai_llm_run_stream_history_and_cancel",
        "test_genai_request_timeout_and_restart_after_failure",
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
    completed.extend(run_genai_fault_recovery_tests(
        args, python, evidence, environment))
    wait_for_no_genai_workers(
        args, evidence / "genai-release-resource-cleanup.log", 120)
    completed.append("test_genai_workers_drained")
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
  asset_environment = os.environ.copy()
  asset_manifest = verify_asset_manifest(
      args.asset_manifest, asset_environment, args.include_genai)

  install_root = Path(os.environ["ProgramFiles"]) / "SimaPCIeHost"
  if install_root.exists():
    raise RuntimeError(f"refusing to modify a pre-existing installation: {install_root}")

  result = {
      "candidate": {
          "core_commit": candidate.core_commit,
          "resources": candidate.resources,
          "scheduler_commit": candidate.receipt.get("scheduler_source_commit"),
      },
      "assets": {
          "identity_sha256": asset_manifest_sha256(asset_manifest),
          "manifest": asset_manifest,
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
  customer_home = work / "customer home with spaces"
  managed_venv = customer_home / "pyneatpcie"
  managed_python = managed_venv / "Scripts" / "python.exe"
  failure: BaseException | None = None
  cleanup_errors: list[str] = []

  result["host"]["windows_security"] = windows_security_snapshot(
      evidence / "windows-security-state.log")
  remote_lock_token = secrets.token_hex(16)
  with ExclusiveLock(args.lock_file, remote_lock_token) as card_lock:
    host_state_baseline = {
        "dependencies": dependency_cohort,
        "platform": windows_platform_state_snapshot(
            evidence / "windows-platform-baseline.log"),
    }
    result["host_state_baseline"] = host_state_baseline
    result["card_network_baseline"] = windows_card_network_identity(
        args.card_host, evidence / "card-network-baseline.log")
    remote_lock = RemoteExclusiveLock(args, args.remote_lock_path, remote_lock_token)
    try:
      remote_lock.acquire(evidence / "card-lock-acquire.log")
    except BaseException:
      if remote_lock.may_hold_after_failed_acquire(evidence / "card-lock-acquire-owner.log"):
        card_lock.retain()
      raise
    card_lock.retain()
    staged_card: dict[str, str] | None = None
    card_stage_attempted = False
    card_stage_owned = False
    firmware_activation_attempted = False
    tutorial_facade_staged = False
    try:
      baseline = card_snapshot(args, evidence, "baseline")
      result["card_baseline"] = baseline
      if baseline["workers"] != "0" or baseline["genai_workers"] != "0":
        raise RuntimeError("card had PCIe workers before qualification; refusing to disturb them")

      card_stage_attempted = True
      staged_card = stage_card_candidate(
          args, card_candidate, remote_lock_token, evidence)
      card_stage_owned = True
      firmware_activation_attempted = True
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

      cli_environment = isolated_sima_cli_environment(
          customer_home, wheelhouse, work)
      native_download_root = work / "native customer downloads with spaces"
      target = "core/pciehost/windows/amd64@3.0"

      def sima_cli_install(registry_url, download_root, log, *options, run=run_logged):
        return run(
            [cli, "neat", "--base-url", registry_url, "install", target,
             "--install-dir", download_root, *options],
            evidence / log, env=cli_environment, cwd=work, timeout=1800)

      if upgrade_base is not None:
        upgrade_registry_root = work / "upgrade baseline registry with spaces"
        stage_local_registry(upgrade_base, upgrade_registry_root)
        with local_registry(upgrade_registry_root) as registry_url:
          sima_cli_install(
              registry_url, native_download_root, "sima-cli-upgrade-baseline-install.log")
        if not install_root.is_dir():
          raise RuntimeError("sima-cli installed no native upgrade baseline")

      native_registry_root = work / "native only registry with spaces"
      stage_local_registry(candidate, native_registry_root)
      with local_registry(native_registry_root) as registry_url:
        sima_cli_install(
            registry_url, native_download_root, "sima-cli-native-only-install.log",
            "--native-only")
      if not install_root.is_dir():
        raise RuntimeError("sima-cli native-only flow installed no native cohort")
      # Without --python-executable the CLI installs into its managed environment.
      if imports_pyneatpcie(python) or (
          managed_python.is_file() and imports_pyneatpcie(managed_python)):
        raise RuntimeError("native-only sima-cli flow unexpectedly installed pyneatpcie")

      missing_registry_root = work / "missing wheel registry with spaces"
      missing_package = stage_local_registry(candidate, missing_registry_root)
      (missing_package / candidate.wheel.name).unlink()
      missing_download_root = work / "missing wheel downloads with spaces"
      with local_registry(missing_registry_root) as registry_url:
        missing_code = sima_cli_install(
            registry_url, missing_download_root, "sima-cli-missing-wheel.log",
            "--python-executable", python, run=run_expected_failure)
      if not install_root.is_dir():
        raise RuntimeError("missing-wheel failure removed the successful native installation")
      if imports_pyneatpcie(python):
        raise RuntimeError("missing-wheel flow reported failure after installing pyneatpcie")

      invalid_wheel_registry_root = work / "invalid wheel registry with spaces"
      stage_invalid_wheel_registry(candidate, invalid_wheel_registry_root)
      invalid_wheel_download_root = work / "invalid wheel downloads with spaces"
      with local_registry(invalid_wheel_registry_root) as registry_url:
        invalid_wheel_code = sima_cli_install(
            registry_url, invalid_wheel_download_root, "sima-cli-invalid-wheel.log",
            "--python-executable", python, run=run_expected_failure)
      if not install_root.is_dir():
        raise RuntimeError("wheel-install failure removed the successful native installation")
      if imports_pyneatpcie(python):
        raise RuntimeError("invalid-wheel flow reported failure after installing pyneatpcie")

      registry_root = work / "complete registry with spaces"
      stage_local_registry(candidate, registry_root)
      recovery_download_root = work / "recovery downloads with spaces"
      with local_registry(registry_root) as registry_url:
        for log in ("sima-cli-customer-recovery-install.log",
                    "sima-cli-customer-repeat-install.log"):
          sima_cli_install(registry_url, recovery_download_root, log)
      if not managed_python.is_file():
        raise RuntimeError(
            "default sima-cli flow did not create the managed pyneatpcie environment")
      run_logged(
          [managed_python, "-m", "pip", "install", "--no-index", "--find-links",
           wheelhouse, *requirements],
          evidence / "managed-python-test-dependencies-install.log",
          timeout=900,
      )
      python = managed_python
      result["customer_install"] = {
          "command": target,
          "download_path_with_spaces": str(recovery_download_root),
          "managed_python": str(managed_python),
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

      verify_customer_process_activation(install_root, python, evidence)
      result["customer_install"]["process_activation"] = "PASS"
      # The installed doctor loads the Scheduler plug-in through the same
      # process-scoped activation entry point customers use.
      result["customer_install"]["plugin_discovery"] = "PASS"

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
      require_assets(environment, args.include_genai)
      result["customer_install"]["standard_user"] = verify_standard_user_process_activation(
          install_root, candidate.wheel, wheelhouse, candidate.root, work, evidence,
          environment, args.include_genai, args.test_timeout_seconds)
      if args.include_genai:
        if environment.get("SIMAPCIE_GENAI_MEDIA_DIRECTORY"):
          result["genai_media"] = {
              "directory": environment["SIMAPCIE_GENAI_MEDIA_DIRECTORY"],
              "serve_root": environment["SIMAPCIE_GENAI_MEDIA_SERVE_ROOT"],
          }
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
      if args.include_genai:
        completed_genai = (
            set(result["tests"]["cpp"]) | set(result["tests"]["python"]))
        if not GENAI_ADMISSION_TESTS.issubset(completed_genai):
          raise RuntimeError("Windows GenAI qualification did not complete every admission case")
        result["genai_admission"] = {
            name: "PASS" for name in sorted(GENAI_ADMISSION_TESTS)
        }
      result["performance"] = {
          mode: parse_performance_log(evidence / f"cpp-test_tensor_{mode}.log", mode)
          for mode in ("latency", "throughput")
      }
      final_asset_manifest = verify_asset_manifest(
          args.asset_manifest, asset_environment, args.include_genai)
      if final_asset_manifest != asset_manifest:
        raise RuntimeError("hardware asset manifest identity changed during qualification")
      if upgrade_base is not None:
        rollback_removal = {
            "core": remove_msi(
                candidate.core_msi, evidence / "rollback-candidate-core-uninstall.log"),
            "scheduler": remove_msi(
                candidate.scheduler_msi,
                evidence / "rollback-candidate-scheduler-uninstall.log"),
        }
        if install_root.exists():
          raise RuntimeError("candidate installation remained before rollback")
        rollback_download_root = work / "rollback downloads with spaces"
        with local_registry(upgrade_registry_root) as registry_url:
          sima_cli_install(
              registry_url, rollback_download_root, "sima-cli-rollback-baseline-install.log")
        rollback_baseline_evidence = evidence / "rollback-baseline"
        rollback_baseline_evidence.mkdir()
        verify_customer_process_activation(
            install_root, managed_python, rollback_baseline_evidence)
        with local_registry(registry_root) as registry_url:
          sima_cli_install(registry_url, rollback_download_root, "sima-cli-rollback-reupgrade.log")
        rollback_candidate_evidence = evidence / "rollback-candidate"
        rollback_candidate_evidence.mkdir()
        verify_customer_process_activation(
            install_root, managed_python, rollback_candidate_evidence)
        result["customer_install"]["rollback"] = {
            "baseline_commit": upgrade_base.core_commit,
            "baseline_version": str(upgrade_base.metadata["version"]),
            "candidate_version": str(candidate.metadata["version"]),
            "candidate_removal": rollback_removal,
            "status": "PASS",
        }
      result["result"] = "PASS"
    except BaseException as error:
      failure = error
      result["failure"] = {
          "message": str(error),
          "type": type(error).__name__,
      }
    finally:
      if firmware_activation_attempted:
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
          result["cleanup"]["card_stage_state"] = (
              "owned" if card_stage_owned else "partial")
          if firmware_activation_attempted:
            restart_windows_platform_service(
                evidence / "card-stock-platform-recovery.log")
            result["cleanup"]["platform_service"] = "PASS"
        except BaseException as error:
          cleanup_errors.append(str(error))
      python_environments = [
          (managed_python, "managed-python-uninstall.log"),
          (venv / "Scripts" / "python.exe", "override-python-uninstall.log"),
      ]
      python_cleanup_errors = []
      for customer_python, log_name in python_environments:
        if customer_python.is_file():
          try:
            run_logged(
                [customer_python, "-m", "pip", "uninstall", "-y", "pyneatpcie"],
                evidence / log_name, timeout=300)
          except BaseException as error:
            python_cleanup_errors.append(str(error))
      cleanup_errors.extend(python_cleanup_errors)
      if not any(path.is_file() for path, _ in python_environments):
        result["cleanup"]["python"] = "not-installed"
      elif not python_cleanup_errors:
        result["cleanup"]["python"] = "PASS"
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
      if install_root.exists():
        cleanup_errors.append(f"cohort left files after removal: {install_root}")
      try:
        final = card_snapshot(args, evidence, "final")
        result["card_final"] = final
        baseline = result.get("card_baseline", {})
        unchanged = ("platform_release", "platform_sha256", "firmware", "remoteproc_state")
        expected_workers = "0" if firmware_activation_attempted else baseline.get("workers", "0")
        expected_genai = (
            "0" if firmware_activation_attempted else baseline.get("genai_workers", "0"))
        if (final["workers"] != expected_workers or final["genai_workers"] != expected_genai
            or any(name in baseline and final.get(name) != baseline[name] for name in unchanged)):
          cleanup_errors.append("card did not return to its qualified baseline")
      except BaseException as error:
        cleanup_errors.append(str(error))
      try:
        network_final = windows_card_network_identity(
            args.card_host, evidence / "card-network-final.log")
        result["card_network_final"] = network_final
        network_baseline = result.get("card_network_baseline", {})
        stable_network_identity = {
            "card_host", "local_address", "interface_index", "interface_guid",
            "interface_description", "pnp_device_id",
        }
        if any(network_final.get(name) != network_baseline.get(name)
               for name in stable_network_identity):
          cleanup_errors.append("card management route changed during qualification")
      except BaseException as error:
        cleanup_errors.append(str(error))
      try:
        host_state_final = {
            "dependencies": verify_dependency_cohort(
                candidate, args.platform_runtime, args.gstreamer_root,
                args.dependency_runtime_dir),
            "platform": windows_platform_state_snapshot(
                evidence / "windows-platform-final.log"),
        }
        result["host_state_final"] = host_state_final
        require_restored_windows_host_state(host_state_baseline, host_state_final)
        result["cleanup"]["host_state"] = "PASS"
      except BaseException as error:
        cleanup_errors.append(str(error))
      try:
        shutil.rmtree(work)
      except FileNotFoundError:
        pass
      except OSError as error:
        cleanup_errors.append(f"hardware work directory cleanup failed: {error}")
      if work.exists():
        cleanup_errors.append(f"hardware work directory remained after cleanup: {work}")
      else:
        result["cleanup"]["work"] = "PASS"
        result["cleanup"]["customer_state"] = "PASS"
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

  if cleanup_errors:
    raise RuntimeError("hardware cleanup failed: " + "; ".join(cleanup_errors))
  if failure is not None:
    raise failure


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--asset-manifest", type=Path, required=True)
  parser.add_argument("--card-candidate", type=Path, required=True)
  parser.add_argument("--card-host", required=True)
  parser.add_argument(
      "--card-sudo-password-env", default="SIMAPCIE_CARD_SUDO_PASSWORD")
  parser.add_argument(
      "--card-sudo-password-stdin", action="store_true",
      help="read one password line once from inherited stdin")
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
  if args.card_sudo_password_stdin:
    args.card_sudo_password_value = sys.stdin.readline().rstrip("\r\n")
    if not args.card_sudo_password_value:
      parser.error("--card-sudo-password-stdin received no password line")
  qualify(args)


if __name__ == "__main__":
  main()
