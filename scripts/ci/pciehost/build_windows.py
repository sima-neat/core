#!/usr/bin/env python3
"""Build and qualify the native Windows Core PCIe-host cohort.

Scheduler's MSI remains an independently produced input. This producer installs
that exact candidate, builds Core against it, and emits an ordered cohort without
copying Scheduler-owned files into the Core MSI or Python wheel.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

from windows_candidate import (
    dependency_identity, require_junit_cases, sha256, validate_candidate, write_json)


ROOT = Path(__file__).resolve().parents[3]
PCIE_HOST = ROOT / "pcie_host"
EXPECTED_MACHINE = "AMD64"
SUPPORTED_PYTHON = (3, 14)
MSI_INSTALLSTATE_UNKNOWN = -1
MSI_INSTALLSTATE_DEFAULT = 5
MSI_GUID = r"\{[0-9A-F]{8}(?:-[0-9A-F]{4}){3}-[0-9A-F]{12}\}"
ROLLBACK_FIXTURE_PRODUCT_CODE = "{5C5FB61E-7355-4B72-87FD-5FDD4CC2715B}"
ROLLBACK_FIXTURE_REGISTRY_KEY = r"Software\SiMa.ai\NeatPcieRollbackFixture"
ROLLBACK_FIXTURE_OBSERVATION = (
    "SiMaNeatMsiRollbackObserved-5C5FB61E-7355-4B72-87FD-5FDD4CC2715B.txt")
# Scheduler's installed record of the Platform it was built against.
SCHEDULER_PLATFORM_RECORD = Path("share") / "SimaScheduler" / "host" / "platform-windows.json"


def run(command: list[object], *, env: dict[str, str] | None = None,
        cwd: Path | None = None, capture: bool = False) -> str:
  argv = [str(value) for value in command]
  print("+", subprocess.list2cmdline(argv), flush=True)
  result = subprocess.run(
      argv,
      check=True,
      cwd=cwd,
      env=env,
      text=True,
      encoding="utf-8" if capture else None,
      errors="replace" if capture else None,
      stdout=subprocess.PIPE if capture else None,
      stderr=subprocess.STDOUT if capture else None,
  )
  return result.stdout.strip() if capture else ""


def require_file(path: Path, description: str) -> Path:
  resolved = path.resolve(strict=True)
  if not resolved.is_file():
    raise ValueError(f"{description} is not a file: {resolved}")
  return resolved


def require_directory(path: Path, description: str) -> Path:
  resolved = path.resolve(strict=True)
  if not resolved.is_dir():
    raise ValueError(f"{description} is not a directory: {resolved}")
  return resolved


def require_program(name: str) -> str:
  path = shutil.which(name)
  if not path:
    raise ValueError(f"required Windows producer program is unavailable: {name}")
  return str(Path(path).resolve())


def require_powershell() -> str:
  """Use PowerShell 7 when provisioned, otherwise Windows PowerShell 5.1."""
  for name in ("pwsh.exe", "powershell.exe"):
    path = shutil.which(name)
    if path:
      return str(Path(path).resolve())
  raise ValueError("required Windows producer program is unavailable: PowerShell")


def validate_windows_host() -> None:
  if sys.platform != "win32":
    raise ValueError("the Windows Core producer must run on Windows")
  if platform.machine().upper() != EXPECTED_MACHINE:
    raise ValueError(f"the Windows Core producer requires AMD64, found {platform.machine()!r}")
  if platform.python_implementation() != "CPython" or sys.version_info[:2] != SUPPORTED_PYTHON:
    expected = ".".join(str(value) for value in SUPPORTED_PYTHON)
    raise ValueError(f"the initial Windows wheel requires CPython {expected} x64")
  if not os.environ.get("VSCMD_VER"):
    raise ValueError("run the producer from an initialized x64 Visual Studio environment")
  if os.environ.get("VSCMD_ARG_TGT_ARCH", "").lower() != "x64":
    raise ValueError("the Visual Studio target architecture must be x64")
  for variable in (
      "CMAKE_TOOLCHAIN_FILE",
      "CMAKE_GENERATOR_PLATFORM",
      "CMAKE_PREFIX_PATH",
      "PKG_CONFIG_PATH",
  ):
    if os.environ.get(variable):
      raise ValueError(f"the Windows Core producer requires a clean environment: {variable}")


def verify_scheduler_candidate(directory: Path, expected_commit: str) -> tuple[dict, Path]:
  root = require_directory(directory, "Scheduler candidate directory")
  metadata_path = require_file(root / "metadata.json", "Scheduler metadata")
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  if metadata.get("repository") != "sima-neat/scheduler":
    raise ValueError("Scheduler candidate has an unexpected repository")
  if metadata.get("artifact") != {
      "architecture": "amd64", "profile": "pcie-host-windows"}:
    raise ValueError("Scheduler candidate is not the Windows amd64 PCIe-host profile")
  if metadata.get("commit") != expected_commit:
    raise ValueError(
        f"Scheduler candidate commit {metadata.get('commit')!r} is not the selected "
        f"commit {expected_commit!r}")
  package_version = metadata.get("package-version")
  if (not isinstance(package_version, str)
      or not re.fullmatch(r"[0-9]+(?:[.][0-9]+){2}", package_version)):
    raise ValueError("Scheduler candidate has no Windows MSI package version")
  resources = metadata.get("resources")
  checksums = metadata.get("resources-checksum")
  if (not isinstance(resources, list) or not resources
      or not isinstance(checksums, dict) or set(resources) != set(checksums)):
    raise ValueError("Scheduler candidate resource inventory is incomplete")
  for name in resources:
    if not isinstance(name, str) or Path(name).name != name:
      raise ValueError(f"invalid Scheduler resource name: {name!r}")
    path = require_file(root / name, "Scheduler candidate resource")
    if sha256(path) != checksums[name]:
      raise ValueError(f"Scheduler candidate checksum differs: {name}")
  receipt_path = require_file(root / "host-candidate.json", "Scheduler host receipt")
  receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
  native_tests = receipt.get("native_tests")
  native_names = native_tests.get("names") if isinstance(native_tests, dict) else None
  native_junit = require_file(root / "native-tests.xml", "Scheduler native JUnit evidence")
  if (receipt.get("source_commit") != expected_commit
      or not isinstance(native_names, list) or not native_names
      or native_tests.get("exit_code") != 0
      or native_tests.get("junit_sha256") != sha256(native_junit)):
    raise ValueError("Scheduler candidate lacks exact native test evidence")
  native_summary = require_junit_cases(native_junit, set(native_names))
  if any(native_tests.get(key) != value for key, value in native_summary.items()):
    raise ValueError("Scheduler candidate native test receipt differs from its JUnit evidence")
  installation = metadata.get("installation", {})
  if installation.get("type") != "windows-msi":
    raise ValueError("Scheduler candidate is not a Windows MSI installation")
  msi_name = installation.get("msi")
  if msi_name not in resources or not str(msi_name).endswith(".msi"):
    raise ValueError("Scheduler metadata does not select its published MSI")
  return metadata, require_file(root / msi_name, "Scheduler MSI")


def inspect_x64(path: Path, dumpbin: str) -> None:
  output = run([dumpbin, "/headers", path], capture=True)
  if not re.search(r"\b8664 machine \(x64\)", output, re.IGNORECASE):
    raise ValueError(f"built artifact is not an x64 PE/COFF file: {path}")


def authenticode_status(path: Path, powershell: str) -> dict[str, str | None]:
  script = (
      "$signature = Get-AuthenticodeSignature "
      "-LiteralPath $env:SIMA_WINDOWS_SIGNATURE_PATH; "
      "$thumbprint = $null; "
      "$timestampThumbprint = $null; "
      "if ($signature.SignerCertificate) {$thumbprint = $signature.SignerCertificate.Thumbprint}; "
      "if ($signature.TimeStamperCertificate) "
      "{$timestampThumbprint = $signature.TimeStamperCertificate.Thumbprint}; "
      "[pscustomobject]@{"
      "status=[string]$signature.Status; "
      "message=[string]$signature.StatusMessage; "
      "thumbprint=$thumbprint; timestamp_thumbprint=$timestampThumbprint} | "
      "ConvertTo-Json -Compress"
  )
  environment = os.environ.copy()
  environment["SIMA_WINDOWS_SIGNATURE_PATH"] = str(path)
  value = json.loads(run(
      [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script],
      env=environment,
      capture=True,
  ))
  return {
      "status": str(value.get("status", "")),
      "message": str(value.get("message", "")),
      "thumbprint": value.get("thumbprint"),
      "timestamp_thumbprint": value.get("timestamp_thumbprint"),
  }


def sign_files(paths: list[Path], signtool: str, certificate_sha1: str,
               timestamp_url: str, machine_store: bool) -> None:
  certificate_sha1 = certificate_sha1.replace(" ", "").upper()
  if not re.fullmatch(r"[0-9A-F]{40}", certificate_sha1):
    raise ValueError("the Windows signing certificate SHA-1 is invalid")
  if not timestamp_url.startswith("https://"):
    raise ValueError("the Windows signing timestamp URL must use HTTPS")
  for path in paths:
    command = [signtool, "sign", "/sha1", certificate_sha1]
    if machine_store:
      command.append("/sm")
    command.extend(["/fd", "SHA256", "/tr", timestamp_url, "/td", "SHA256", path])
    run(command)


def native_files(root: Path) -> list[Path]:
  return [
      path for path in sorted(root.rglob("*"))
      if path.is_file() and path.suffix.lower() in (".dll", ".exe", ".pyd")
  ]


def build_sources_for_installed(build_root: Path, installed_roots: list[Path]) -> list[Path]:
  build_by_name: dict[str, list[Path]] = {}
  for path in native_files(build_root):
    build_by_name.setdefault(path.name.lower(), []).append(path)
  selected = []
  for name in sorted({path.name.lower() for root in installed_roots for path in native_files(root)}):
    matches = build_by_name.get(name, [])
    if len(matches) != 1:
      raise ValueError(f"cannot identify one build source for shipped native file {name}: {matches}")
    selected.append(matches[0])
  if not selected:
    raise ValueError("Windows Core producer installed no native files")
  return selected


def wheel_native_signatures(wheel: Path, powershell: str) -> list[dict]:
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    with zipfile.ZipFile(wheel) as archive:
      native_names = [name for name in archive.namelist() if name.lower().endswith(".pyd")]
      if not native_names:
        raise ValueError("Windows wheel contains no PYD to verify")
      for name in native_names:
        if Path(name).is_absolute() or ".." in Path(name).parts:
          raise ValueError(f"unsafe wheel member: {name}")
        archive.extract(name, root)
    return [
        {"path": name, **authenticode_status(root / name, powershell)}
        for name in native_names
    ]


def require_valid_signatures(records: list[dict], description: str,
                             certificate_sha1: str) -> None:
  expected = certificate_sha1.replace(" ", "").upper()
  invalid = [
      record["path"] for record in records
      if record.get("status") != "Valid"
      or str(record.get("thumbprint", "")).upper() != expected
      or not record.get("timestamp_thumbprint")
  ]
  if invalid:
    raise RuntimeError(f"{description} has missing or invalid Authenticode signatures: {invalid}")


def msi_transaction(msi: Path, action: str, log: Path) -> int:
  result = subprocess.run(
      ["msiexec.exe", f"/{action}", str(msi), "/quiet", "/norestart", "/L*v", str(log)],
      check=False,
  )
  if result.returncode not in (0, 1641, 3010):
    raise RuntimeError(
        f"Windows Installer /{action} failed with {result.returncode}; see {log}")
  return result.returncode


def require_no_restart(code: int, operation: str) -> None:
  if code != 0:
    raise RuntimeError(f"{operation} requested a Windows restart ({code})")


def file_inventory(root: Path) -> dict[str, str]:
  return {
      path.relative_to(root).as_posix(): sha256(path)
      for path in sorted(root.rglob("*"))
      if path.is_file()
  }


def inventory_digest(inventory: dict[str, str]) -> str:
  encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def msi_property(msi: Path, name: str) -> str:
  """Read one Property table value through the Windows Installer database API."""
  if sys.platform != "win32":
    raise ValueError("MSI database inspection requires Windows")
  installer = ctypes.WinDLL("msi", use_last_error=True)
  handle_type = ctypes.c_uint
  database = handle_type()
  view = handle_type()
  record = handle_type()
  installer.MsiOpenDatabaseW.argtypes = [
      ctypes.c_wchar_p, ctypes.c_void_p, ctypes.POINTER(handle_type)]
  installer.MsiOpenDatabaseW.restype = ctypes.c_uint
  installer.MsiDatabaseOpenViewW.argtypes = [
      handle_type, ctypes.c_wchar_p, ctypes.POINTER(handle_type)]
  installer.MsiDatabaseOpenViewW.restype = ctypes.c_uint
  installer.MsiViewExecute.argtypes = [handle_type, handle_type]
  installer.MsiViewExecute.restype = ctypes.c_uint
  installer.MsiViewFetch.argtypes = [handle_type, ctypes.POINTER(handle_type)]
  installer.MsiViewFetch.restype = ctypes.c_uint
  installer.MsiRecordGetStringW.argtypes = [
      handle_type, ctypes.c_uint, ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint)]
  installer.MsiRecordGetStringW.restype = ctypes.c_uint
  installer.MsiCloseHandle.argtypes = [handle_type]
  installer.MsiCloseHandle.restype = ctypes.c_uint

  handles: list[int] = []
  try:
    status = installer.MsiOpenDatabaseW(str(msi), None, ctypes.byref(database))
    if status != 0:
      raise RuntimeError(f"MsiOpenDatabaseW failed for {msi}: {status}")
    handles.append(database.value)
    query = f"SELECT `Value` FROM `Property` WHERE `Property`='{name}'"
    status = installer.MsiDatabaseOpenViewW(database, query, ctypes.byref(view))
    if status != 0:
      raise RuntimeError(f"MsiDatabaseOpenViewW failed for {msi}: {status}")
    handles.append(view.value)
    status = installer.MsiViewExecute(view, 0)
    if status != 0:
      raise RuntimeError(f"MsiViewExecute failed for {msi}: {status}")
    status = installer.MsiViewFetch(view, ctypes.byref(record))
    if status != 0:
      raise RuntimeError(f"MSI has no {name} row: {msi}: {status}")
    handles.append(record.value)
    capacity = ctypes.c_uint(64)
    value = ctypes.create_unicode_buffer(capacity.value + 1)
    status = installer.MsiRecordGetStringW(record, 1, value, ctypes.byref(capacity))
    if status != 0:
      raise RuntimeError(f"MsiRecordGetStringW failed for {msi}: {status}")
    return value.value
  finally:
    for handle in reversed(handles):
      installer.MsiCloseHandle(handle)


def msi_product_code(msi: Path) -> str:
  product_code = msi_property(msi, "ProductCode").upper()
  if not re.fullmatch(MSI_GUID, product_code):
    raise RuntimeError(f"MSI has an invalid ProductCode: {msi}: {product_code!r}")
  return product_code


def scheduler_msi_identity(msi: Path, package_version: str) -> dict[str, str]:
  """Read the identity the Core MSI's launch condition pins from Scheduler's MSI."""
  upgrade_code = msi_property(msi, "UpgradeCode").upper()
  if not re.fullmatch(MSI_GUID, upgrade_code):
    raise ValueError(f"Scheduler MSI has an invalid UpgradeCode: {msi}: {upgrade_code!r}")
  version = msi_property(msi, "ProductVersion")
  if version != package_version:
    raise ValueError(
        f"Scheduler MSI ProductVersion {version!r} is not its package version "
        f"{package_version!r}: {msi}")
  return {"upgrade_code": upgrade_code, "version": version}


def msi_product_state(product_code: str) -> int:
  if sys.platform != "win32":
    raise ValueError("MSI product-state inspection requires Windows")
  installer = ctypes.WinDLL("msi", use_last_error=True)
  installer.MsiQueryProductStateW.argtypes = [ctypes.c_wchar_p]
  installer.MsiQueryProductStateW.restype = ctypes.c_int
  return int(installer.MsiQueryProductStateW(product_code))


def registry_fixture_key_exists() -> bool:
  if sys.platform != "win32":
    raise ValueError("registry inspection requires Windows")
  import winreg  # pylint: disable=import-outside-toplevel
  try:
    with winreg.OpenKey(
        winreg.HKEY_LOCAL_MACHINE, ROLLBACK_FIXTURE_REGISTRY_KEY, 0,
        winreg.KEY_READ | winreg.KEY_WOW64_64KEY):
      return True
  except FileNotFoundError:
    return False


def remove_fixture_registry_key() -> None:
  if sys.platform != "win32":
    return
  import winreg  # pylint: disable=import-outside-toplevel
  try:
    winreg.DeleteKeyEx(
        winreg.HKEY_LOCAL_MACHINE, ROLLBACK_FIXTURE_REGISTRY_KEY,
        access=winreg.KEY_WOW64_64KEY)
  except FileNotFoundError:
    pass


def common_application_data(powershell: str) -> Path:
  value = run([
      powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command",
      "[Environment]::GetFolderPath([Environment+SpecialFolder]::CommonApplicationData)",
  ], capture=True)
  path = Path(value)
  if not value or not path.is_absolute():
    raise RuntimeError(f"Windows CommonApplicationData is invalid: {value!r}")
  return path


def build_msi_rollback_fixture(work: Path, cl: str, wix_root: Path,
                               environment: dict[str, str]) -> Path:
  fixture_source = ROOT / "scripts" / "ci" / "pciehost" / "windows_msi_rollback_fixture"
  source = require_file(fixture_source / "rollback_probe.c", "rollback probe source")
  wxs = require_file(fixture_source / "rollback_fixture.wxs", "rollback fixture WiX source")
  require_file(fixture_source / "rollback-sentinel.txt", "rollback sentinel source")
  candle = require_file(wix_root / "candle.exe", "WiX candle")
  light = require_file(wix_root / "light.exe", "WiX light")
  fixture_work = work / "msi-rollback-fixture"
  fixture_work.mkdir()
  shutil.copy2(fixture_source / "rollback-sentinel.txt", fixture_work)
  probe = fixture_work / "rollback-probe.exe"
  run([
      cl, "/nologo", "/W4", "/WX", "/O2", "/DUNICODE", "/D_UNICODE",
      "/Fo" + str(fixture_work / "rollback-probe.obj"),
      "/Fe" + str(probe), source,
      "/link", "advapi32.lib", "ole32.lib", "shell32.lib",
  ], cwd=fixture_work, env=environment)
  require_file(probe, "built rollback probe")
  wix_object = fixture_work / "rollback-fixture.wixobj"
  run([
      candle, "-nologo", "-arch", "x64",
      "-out", wix_object, wxs,
  ], cwd=fixture_work, env=environment)
  fixture_msi = fixture_work / "rollback-fixture.msi"
  run([light, "-nologo", "-out", fixture_msi, wix_object],
      cwd=fixture_work, env=environment)
  return require_file(fixture_msi, "built rollback fixture MSI")


def qualify_failed_msi_transaction(
    fixture_msi: Path, scheduler_msi: Path, core_msi: Path, install_root: Path,
    common_data: Path, log: Path) -> dict:
  """Prove that Windows Installer atomically rolls back a deferred failure."""
  scheduler_product = msi_product_code(scheduler_msi)
  core_product = msi_product_code(core_msi)
  fixture_product = msi_product_code(fixture_msi)
  if fixture_product != ROLLBACK_FIXTURE_PRODUCT_CODE:
    raise RuntimeError("CI rollback fixture ProductCode changed unexpectedly")
  product_states_before = {
      "scheduler": msi_product_state(scheduler_product),
      "core": msi_product_state(core_product),
      "fixture": msi_product_state(fixture_product),
  }
  if (product_states_before["scheduler"] != MSI_INSTALLSTATE_DEFAULT
      or product_states_before["core"] != MSI_INSTALLSTATE_DEFAULT
      or product_states_before["fixture"] != MSI_INSTALLSTATE_UNKNOWN):
    raise RuntimeError(
        f"MSI rollback fixture has invalid product preconditions: {product_states_before}")

  fixture_root = common_data / "SiMa.ai" / "NeatPcieRollbackFixture"
  sentinel = fixture_root / "rollback-sentinel.txt"
  observation = common_data / ROLLBACK_FIXTURE_OBSERVATION
  if fixture_root.exists() or observation.exists() or registry_fixture_key_exists():
    raise RuntimeError("MSI rollback fixture state was not clean before installation")
  inventory_before = file_inventory(install_root)
  if not inventory_before:
    raise RuntimeError("installed Scheduler/Core cohort is empty before rollback test")

  result = None
  try:
    result = subprocess.run(
        ["msiexec.exe", "/i", str(fixture_msi), "/quiet", "/norestart",
         "/L*v", str(log)],
        check=False,
    )
    observed = observation.read_text(encoding="ascii") if observation.is_file() else ""
    expected_observation = "schema_version=1\nfile=present\nregistry=present\n"
    inventory_after = file_inventory(install_root)
    product_states_after = {
        "scheduler": msi_product_state(scheduler_product),
        "core": msi_product_state(core_product),
        "fixture": msi_product_state(fixture_product),
    }
    failures = []
    if result.returncode != 1603:
      failures.append(f"expected Windows Installer failure 1603, got {result.returncode}")
    if observed != expected_observation:
      failures.append("deferred failure probe did not observe the MSI-owned file and registry")
    if fixture_root.exists() or sentinel.exists():
      failures.append("Windows Installer rollback left the fixture install directory")
    if registry_fixture_key_exists():
      failures.append("Windows Installer rollback left the fixture registry key or value")
    if product_states_after["fixture"] != MSI_INSTALLSTATE_UNKNOWN:
      failures.append("Windows Installer rollback left the fixture product registered")
    if product_states_after["scheduler"] != MSI_INSTALLSTATE_DEFAULT:
      failures.append("failed MSI transaction changed Scheduler product registration")
    if product_states_after["core"] != MSI_INSTALLSTATE_DEFAULT:
      failures.append("failed MSI transaction changed Core product registration")
    if inventory_after != inventory_before:
      failures.append("failed MSI transaction changed the installed Scheduler/Core inventory")
    if not log.is_file() or log.stat().st_size == 0:
      failures.append("failed MSI transaction did not produce a verbose installer log")
    else:
      log_bytes = log.read_bytes()
      action_ascii = b"FailAfterOwnedState"
      action_utf16 = "FailAfterOwnedState".encode("utf-16-le")
      if action_ascii not in log_bytes and action_utf16 not in log_bytes:
        # The action identifier is stable even when the surrounding Windows
        # Installer messages are localized.
        failures.append("installer log did not record the deferred failure action")
    if failures:
      raise RuntimeError("; ".join(failures))
    return {
        "schema_version": 1,
        "status": "PASS",
        "fixture": {
            "custom_action": "FailAfterOwnedState",
            "msi_sha256": sha256(fixture_msi),
            "product_code": fixture_product,
        },
        "failure": {"expected_exit_code": 1603, "observed_exit_code": result.returncode},
        "mid_transaction": {"file": "PASS", "registry": "PASS"},
        "rollback": {
            "install_directory": "PASS",
            "product_registration": "PASS",
            "registry": "PASS",
            "sentinel": "PASS",
        },
        "preserved_cohort": {
            "file_inventory": "PASS",
            "file_inventory_sha256": inventory_digest(inventory_before),
            "scheduler": {
                "product_code": scheduler_product,
                "state_before": "DEFAULT",
                "state_after": "DEFAULT",
            },
            "core": {
                "product_code": core_product,
                "state_before": "DEFAULT",
                "state_after": "DEFAULT",
            },
        },
        "installer_log_sha256": sha256(log),
    }
  finally:
    # Leave the shared runner clean even when the assertion itself exposes a
    # rollback defect. These paths and ProductCode belong only to this fixture.
    if msi_product_state(fixture_product) != MSI_INSTALLSTATE_UNKNOWN:
      subprocess.run(
          ["msiexec.exe", "/x", fixture_product, "/quiet", "/norestart"],
          check=False,
      )
    observation.unlink(missing_ok=True)
    if fixture_root.exists():
      shutil.rmtree(fixture_root)
    remove_fixture_registry_key()


def wheel_payload_names(wheel: Path) -> set[str]:
  with zipfile.ZipFile(wheel) as archive:
    bad = archive.testzip()
    if bad:
      raise ValueError(f"wheel member failed CRC validation: {bad}")
    return set(archive.namelist())


def build(args: argparse.Namespace) -> None:
  validate_windows_host()
  commit = run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture=True)
  if not re.fullmatch(r"[0-9a-f]{40}", commit):
    raise ValueError("invalid Core source commit")
  if run(["git", "-C", ROOT, "status", "--porcelain"], capture=True):
    raise ValueError("publication requires a clean Core source checkout")

  scheduler_metadata, scheduler_msi = verify_scheduler_candidate(
      args.scheduler_candidate, args.expected_scheduler_commit)
  scheduler_identity = scheduler_msi_identity(
      scheduler_msi, scheduler_metadata["package-version"])
  scheduler_dependencies = dependency_identity(args.scheduler_candidate)
  gstreamer_root = require_directory(args.gstreamer_root, "GStreamer MSVC root")
  gstreamer_bin = require_directory(gstreamer_root / "bin", "GStreamer binary directory")
  gstreamer_dlls = sorted(
      gstreamer_bin.glob("*.dll"), key=lambda path: path.name.lower())
  if not gstreamer_dlls:
    raise ValueError(f"GStreamer binary directory contains no DLLs: {gstreamer_bin}")
  pkgconfig_dir = require_directory(
      gstreamer_root / "lib" / "pkgconfig", "GStreamer pkg-config directory")
  gst_inspect = require_file(gstreamer_bin / "gst-inspect-1.0.exe", "gst-inspect")
  pkg_config = require_file(gstreamer_bin / "pkg-config.exe", "GStreamer pkg-config")
  # Like Linux, the build host has no Platform installation and records no
  # Platform files; Scheduler's receipt names the Platform version it targets.
  producer_dependencies = {
      "platform_version": scheduler_dependencies["platform_version"],
      "gstreamer": {
          "pkg_config_sha256": sha256(pkg_config),
          "runtime_dlls": [
              {"name": path.name, "sha256": sha256(path)} for path in gstreamer_dlls
          ],
          "version_output": run([gst_inspect, "--version"], capture=True),
      },
  }
  if producer_dependencies["gstreamer"] != scheduler_dependencies["gstreamer"]:
    raise ValueError("Core and Scheduler Windows producers use different GStreamer dependencies")
  wheelhouse = require_directory(args.python_wheelhouse, "Python dependency wheelhouse")
  wheelhouse_wheels = sorted(wheelhouse.glob("*.whl"))
  if not wheelhouse_wheels:
    raise ValueError(f"Python dependency wheelhouse contains no wheels: {wheelhouse}")
  dependency_prefixes = [
      require_directory(Path(value), "Windows CMake dependency prefix")
      for value in args.dependency_prefix
  ]
  dependency_runtime_dirs = [
      require_directory(Path(value), "Windows dependency runtime directory")
      for value in args.dependency_runtime_dir
  ]
  if not dependency_runtime_dirs:
    raise ValueError("at least one Windows dependency runtime directory is required")
  dependency_runtime_dlls = []
  for directory in dependency_runtime_dirs:
    dlls = sorted(directory.glob("*.dll"), key=lambda path: path.name.lower())
    if not dlls:
      raise ValueError(f"Windows dependency runtime directory contains no DLLs: {directory}")
    dependency_runtime_dlls.append(dlls)

  cmake = require_program("cmake.exe")
  ctest = require_program("ctest.exe")
  cpack = require_program("cpack.exe")
  dumpbin = require_program("dumpbin.exe")
  cl = require_program("cl.exe")
  ninja = require_program("ninja.exe")
  powershell = require_powershell()
  python = require_file(Path(sys.executable), "producer Python")
  if not os.environ.get("WIX"):
    raise ValueError("WIX must name the pinned WiX v3 tool directory")
  wix_root = require_directory(Path(os.environ["WIX"]), "pinned WiX v3 tool directory")

  work = args.work.resolve()
  output = args.output.resolve()
  for path in (work, output):
    if path.exists():
      raise ValueError(f"producer path must be fresh: {path}")
    path.mkdir(parents=True)

  install_root = Path(os.environ["ProgramFiles"]) / "SimaPCIeHost"
  if install_root.exists():
    raise ValueError(f"refusing to modify a pre-existing installation: {install_root}")
  build_dir = work / "build"
  stage = work / "stage"
  relocated = work / "relocated"
  python_stage = work / "python-stage"
  consumer_build_root = work / "consumers"
  consumer_build_root.mkdir()

  environment = os.environ.copy()
  environment["PKG_CONFIG"] = str(pkg_config)
  environment["PKG_CONFIG_PATH"] = str(pkgconfig_dir)
  environment["PATH"] = os.pathsep.join((str(gstreamer_bin), environment["PATH"]))

  scheduler_installed = False
  core_installed = False
  scheduler_owned: dict[str, str] = {}
  venv = work / "customer-venv"
  try:
    scheduler_install_code = msi_transaction(
        scheduler_msi, "i", output / "scheduler-msi-install.log")
    scheduler_installed = True
    require_no_restart(scheduler_install_code, "Scheduler MSI installation")
    scheduler_config = require_file(
        install_root / "lib" / "cmake" / "SimaScheduler" / "SimaSchedulerConfig.cmake",
        "installed Scheduler CMake package",
    ).parent
    plugin = require_file(
        install_root / "bin" / "gstreamer-1.0" / "gstneatpciehost.dll",
        "installed Scheduler plugin",
    )
    inspect_x64(plugin, dumpbin)
    scheduler_owned = file_inventory(install_root)
    if not scheduler_owned:
      raise ValueError("Scheduler MSI installed no files")
    # Hardware qualification checks the card PC's Platform DLLs against this record.
    platform_record = output / "scheduler-platform-windows.json"
    shutil.copy2(require_file(install_root / SCHEDULER_PLATFORM_RECORD,
                              "installed Scheduler Platform record"), platform_record)
    if (json.loads(platform_record.read_text(encoding="utf-8")).get("platform_version")
        != scheduler_dependencies["platform_version"]):
      raise ValueError("Scheduler Platform record and receipt name different Platform versions")

    prefixes = [install_root, gstreamer_root, *dependency_prefixes]
    prefix_path = ";".join(str(path) for path in prefixes)
    environment["CMAKE_PREFIX_PATH"] = prefix_path
    environment["PATH"] = os.pathsep.join((
        str(install_root / "bin"),
        *(str(path) for path in dependency_runtime_dirs),
        environment["PATH"],
    ))
    run([
        cmake,
        "-S", PCIE_HOST,
        "-B", build_dir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=" + str(stage),
        "-DCMAKE_PREFIX_PATH=" + prefix_path,
        "-DSimaScheduler_DIR=" + str(scheduler_config),
        "-DSIMAPCIE_SCHEDULER_PACKAGE_VERSION=" + scheduler_identity["version"],
        "-DSIMAPCIE_SCHEDULER_WIX_UPGRADE_CODE=" + scheduler_identity["upgrade_code"],
        "-DSIMAPCIE_BUILD_TESTS=ON",
        "-DSIMAPCIE_BUILD_HARDWARE_TESTS=ON",
        "-DSIMAPCIE_BUILD_PYTHON=ON",
        "-DPython3_EXECUTABLE=" + str(python),
        "-DPython_EXECUTABLE=" + str(python),
    ], env=environment)
    run([cmake, "--build", build_dir, "--parallel", str(args.jobs)], env=environment)

    tests = json.loads(run([ctest, "--test-dir", build_dir, "--show-only=json-v1"],
                           env=environment, capture=True))
    test_names = [test["name"] for test in tests.get("tests", [])]
    if not test_names:
      raise ValueError("Windows Core producer registered no tests")
    for path in sorted(build_dir.rglob("*")):
      if path.is_file() and path.suffix.lower() in (".dll", ".exe", ".pyd"):
        inspect_x64(path, dumpbin)
    native_junit = output / "native-tests.xml"
    run([ctest, "--test-dir", build_dir, "--output-on-failure", "--no-tests=error",
         "--output-junit", native_junit], env=environment)
    native_test_summary = require_junit_cases(native_junit, set(test_names))

    def install_components() -> None:
      for prefix, component in ((stage, "Runtime"), (stage, "Development"),
                                (stage, "PcieHostTutorials"), (python_stage, "PcieHostPython")):
        run([cmake, "--install", build_dir, "--config", "Release", "--prefix", prefix,
             "--component", component], env=environment)

    install_components()
    if args.signing_certificate_sha1:
      signtool = require_program("signtool.exe")
      sign_files(
          build_sources_for_installed(build_dir, [stage, python_stage]),
          signtool, args.signing_certificate_sha1,
          args.signing_timestamp_url, args.signing_machine_store)
      install_components()
    elif args.require_signatures:
      raise RuntimeError("release mode requires a Windows signing certificate")
    stage.rename(relocated)
    hardware_tests = output / "hardware-tests"
    run([cmake, "--install", build_dir, "--config", "Release", "--prefix", hardware_tests,
         "--component", "PcieHostExtras"], env=environment)
    run([cmake, "--install", build_dir, "--config", "Release", "--prefix", hardware_tests,
         "--component", "PcieHostTutorials"], env=environment)
    hardware_native = native_files(hardware_tests)
    if not hardware_native:
      raise ValueError("Windows Core producer installed no C++ hardware test executables")
    if args.signing_certificate_sha1:
      sign_files(
          hardware_native, signtool, args.signing_certificate_sha1,
          args.signing_timestamp_url, args.signing_machine_store)

    version = json.loads((ROOT / "deps" / "manifest.json").read_text(encoding="utf-8"))[
        "package-version"]
    wheel_text = run([
        python,
        PCIE_HOST / "python" / "build_wheel.py",
        "--stage-dir", python_stage,
        "--output-dir", output,
        "--version", version,
    ], env=environment, capture=True)
    wheel = require_file(Path(wheel_text.splitlines()[-1]), "built pyneatpcie wheel")
    names = wheel_payload_names(wheel)
    if not any(name.endswith(".pyd") for name in names):
      raise ValueError("Windows pyneatpcie wheel contains no compiled extension")
    forbidden = {"simaai_scheduler_pcie_host.dll", "gstneatpciehost.dll"}
    duplicated = sorted(name for name in names if Path(name).name.lower() in forbidden)
    if duplicated:
      raise ValueError(f"wheel duplicates MSI-owned native files: {duplicated}")

    def check_consumer(prefix: Path, label: str) -> None:
      consumer_prefix_path = ";".join((str(prefix), str(install_root), str(gstreamer_root)))
      consumer_environment = environment.copy()
      consumer_environment["PATH"] = os.pathsep.join(
          (str(prefix / "bin"), str(install_root / "bin"), consumer_environment["PATH"]))
      for source, executable in (
          (PCIE_HOST / "tests" / "package_smoke", "sima_pcie_host_package_smoke.exe"),
          (PCIE_HOST / "tests" / "installed_genai_consumer", "installed_genai_consumer.exe"),
      ):
        consumer_build = consumer_build_root / label / source.name
        run([
            cmake,
            "-S", source,
            "-B", consumer_build,
            "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_PREFIX_PATH=" + consumer_prefix_path,
        ], env=environment)
        run([cmake, "--build", consumer_build], env=environment)
        binary = consumer_build / executable
        inspect_x64(binary, dumpbin)
        run([binary], env=consumer_environment)

    check_consumer(relocated, "relocated")

    cpack_output = work / "cpack"
    run([cpack, "--config", build_dir / "CPackConfig.cmake", "-G", "WIX",
         "-B", cpack_output, "-C", "Release"], env=environment)
    core_packages = sorted(cpack_output.glob("sima-pcie-host-*.msi"))
    if len(core_packages) != 1:
      raise ValueError(f"expected exactly one Core MSI, found {len(core_packages)}")
    core_msi = output / core_packages[0].name
    shutil.copy2(core_packages[0], core_msi)
    if args.signing_certificate_sha1:
      sign_files(
          [core_msi], signtool, args.signing_certificate_sha1,
          args.signing_timestamp_url, args.signing_machine_store)
    shutil.copy2(scheduler_msi, output / scheduler_msi.name)
    scheduler_msi = output / scheduler_msi.name
    rollback_fixture_msi = build_msi_rollback_fixture(work, cl, wix_root, environment)
    inspect_x64(
        rollback_fixture_msi.parent / "rollback-probe.exe", dumpbin)

    core_install_code = msi_transaction(core_msi, "i", output / "core-msi-install.log")
    core_installed = True
    require_no_restart(core_install_code, "Core MSI installation")
    after_core_install = file_inventory(install_root)
    changed_scheduler = {
        name: (digest, after_core_install.get(name))
        for name, digest in scheduler_owned.items()
        if after_core_install.get(name) != digest
    }
    if changed_scheduler:
      raise ValueError(f"Core MSI changed Scheduler-owned files: {changed_scheduler}")
    for required in (
        install_root / "include" / "simaai" / "neat" / "pcie" / "Model.h",
        install_root / "lib" / "sima_neat_pcie_host.lib",
        install_root / "include" / "nlohmann" / "json.hpp",
        install_root / "share" / "licenses" / "sima-pcie-host" /
        "nlohmann-json-LICENSE.MIT",
        install_root / "bin" / "sima-pcie-host-env.cmd",
        install_root / "bin" / "sima-pcie-host-run.cmd",
        install_root / "bin" / "sima-pcie-host-doctor.cmd",
        install_root / "bin" / "pcie-setup.cmd",
        install_root / "bin" / "pcie-setup.ps1",
        install_root / "share" / "sima-pcie-host" / "skills" /
        "neat-pcie-application-builder" / "SKILL.md",
        install_root / "share" / "sima-pcie-host" / "skills" /
        "neat-pcie-application-builder" / "playbook.yml",
        install_root / "share" / "sima-pcie-host" / "skills" /
        "neat-pcie-application-builder" / "targets" / "codex" / "agents" /
        "openai.yaml",
    ):
      require_file(required, "installed Core MSI payload")
    check_consumer(install_root, "installed")

    rollback_receipt = qualify_failed_msi_transaction(
        rollback_fixture_msi, scheduler_msi, core_msi, install_root,
        common_application_data(powershell), output / "msi-transaction-rollback.log")
    check_consumer(install_root, "after-failed-msi-transaction")
    rollback_receipt["preserved_cohort"].update({
        "installed_consumer": "PASS",
    })
    write_json(output / "msi-transaction-rollback.json", rollback_receipt)

    activation_environment = os.environ.copy()
    for name in (
        "CMAKE_PREFIX_PATH", "GSTREAMER_ROOT_X86_64", "GST_PLUGIN_PATH",
        "GST_PLUGIN_PATH_1_0", "PKG_CONFIG", "PKG_CONFIG_PATH",
        "SIMA_PCIE_HOST_ENVIRONMENT", "SIMA_PCIE_HOST_LIBRARY_PATH",
        "SIMA_WINDOWS_PLATFORM_RUNTIME_DIR",
    ):
      activation_environment.pop(name, None)
    windows_root = Path(os.environ["SystemRoot"])
    activation_environment["PATH"] = os.pathsep.join((
        str(windows_root / "System32"), str(windows_root),
        str(windows_root / "System32" / "Wbem"),
        str(windows_root / "System32" / "WindowsPowerShell" / "v1.0"),
        str(windows_root / "System32" / "OpenSSH"),
    ))
    # The doctor and plug-in discovery need an installed Platform; hardware
    # qualification runs them on the card PC, as Linux tests Platform only there.
    run([
        "cmd.exe", "/d", "/c",
        install_root / "bin" / "pcie-setup.cmd",
        "-Hosts", "10.0.0.2", "-DryRun",
    ], env=activation_environment)

    signature_records = []
    for owner, root in (("core", relocated), ("python", python_stage),
                        ("hardware-tests", hardware_tests)):
      for path in native_files(root):
        signature_records.append({
            "owner": owner,
            "path": path.relative_to(root).as_posix(),
            **authenticode_status(path, powershell),
        })
    signature_records.extend({
        "owner": "package",
        "path": path.name,
        **authenticode_status(path, powershell),
    } for path in (scheduler_msi, core_msi))
    signature_records.extend({"owner": "python-wheel", **record}
                             for record in wheel_native_signatures(wheel, powershell))
    scheduler_plugin_signature = {
        "owner": "scheduler",
        "path": plugin.relative_to(install_root).as_posix(),
        **authenticode_status(plugin, powershell),
    }
    signature_records.append(scheduler_plugin_signature)
    if args.require_signatures:
      require_valid_signatures(
          signature_records, "release candidate", args.signing_certificate_sha1)
    write_json(output / "windows-signatures.json", {
        "certificate_sha1": args.signing_certificate_sha1.replace(" ", "").upper(),
        "release_required": args.require_signatures,
        "records": signature_records,
        "timestamp_url": args.signing_timestamp_url if args.signing_certificate_sha1 else None,
    })

    before_repeat = file_inventory(install_root)
    for owner, msi in (("Scheduler", scheduler_msi), ("Core", core_msi)):
      require_no_restart(
          msi_transaction(msi, "i", output / f"{owner.lower()}-msi-repeat-install.log"),
          f"{owner} MSI repeat installation")
    after_repeat = file_inventory(install_root)
    if after_repeat != before_repeat:
      raise ValueError("ordered cohort repeat installation changed its file inventory")
    check_consumer(install_root, "repeat-installed")

    run([python, "-m", "venv", venv], env=environment)
    customer_python = venv / "Scripts" / "python.exe"
    run([
        customer_python,
        "-m", "pip", "install",
        "--no-index",
        "--find-links", wheelhouse,
        wheel,
    ], env=environment)
    import_environment = activation_environment.copy()
    import_environment.pop("PYTHONPATH", None)
    run([
        customer_python,
        "-c",
        "from pathlib import Path; import pyneatpcie as p; assert p.__version__; "
        "assert p.Model; assert p.Runtime; "
        "assert p.Tensor; assert p.genai.GenAIModel; assert p.genai.GenerationRequest; "
        "assert p.genai.ConnectionOptions; c = p.genai.ConnectionOptions(); "
        "media = Path(c.media_directory); assert media.is_absolute(); "
        "assert tuple(part.lower() for part in media.parts[-2:]) == ('sima.ai', 'data'); "
        "assert c.media_serve_root == 'data'",
    ], env=import_environment, cwd=Path(os.environ["TEMP"]))

    scheduler_version = scheduler_metadata["package-version"]
    gstreamer_match = re.search(
        r"(?m)^gst-inspect-1[.]0 version ([0-9]+(?:[.][0-9]+){2})$",
        producer_dependencies["gstreamer"]["version_output"])
    if not gstreamer_match:
      raise ValueError("could not read the exact GStreamer version")
    run([
        python,
        PCIE_HOST / "scripts" / "create_windows_metadata.py",
        "--scheduler-msi", scheduler_msi,
        "--scheduler-version", scheduler_version,
        "--platform-version", producer_dependencies["platform_version"],
        "--gstreamer-version", gstreamer_match.group(1),
        "--msi", core_msi,
        "--wheel", wheel,
        "--version", version,
        "--branch", args.branch,
        "--commit", commit,
        "--output", output / "metadata.json",
    ], env=environment)
    shutil.copy2(
        args.scheduler_candidate / "metadata.json", output / "scheduler-metadata.json")
    scheduler_receipt = args.scheduler_candidate / "host-candidate.json"
    if scheduler_receipt.is_file():
      shutil.copy2(scheduler_receipt, output / "scheduler-host-candidate.json")
    scheduler_signatures = require_file(
        args.scheduler_candidate / "windows-signatures.json",
        "Scheduler signature receipt")
    shutil.copy2(scheduler_signatures, output / "scheduler-windows-signatures.json")
    for source_name, output_name in (
        ("windows-dependencies.json", "scheduler-windows-dependencies.json"),
        ("installed-checks.json", "scheduler-installed-checks.json"),
    ):
      source = args.scheduler_candidate / source_name
      if source.is_file():
        shutil.copy2(source, output / output_name)

    for path in (scheduler_msi, core_msi, wheel):
      (output / f"{path.name}.sha256").write_text(
          f"{sha256(path)}  {path.name}\n", encoding="ascii")
    write_json(output / "installed-checks.json", {
        "core_msi_install": "PASS",
        "core_msi_remove": "PENDING_CLEANUP",
        "hardware_qualified": False,
        "installed_consumer": "PASS",
        "python_import": "PASS",
        "relocated_consumer": "PASS",
        "repeat_install": "PASS",
        "transaction_rollback": "PASS",
        "scheduler_msi_install": "PASS",
        "scheduler_msi_remove": "PENDING_CLEANUP",
    })
    write_json(output / "windows-dependencies.json", {
        "cmake_prefixes": [str(path) for path in dependency_prefixes],
        "dependency_runtime": [
            {
                "files": [
                    {"name": path.name, "sha256": sha256(path)} for path in dlls
                ],
            }
            for dlls in dependency_runtime_dlls
        ],
        "gstreamer": {
            "pkg_config_sha256": sha256(pkg_config),
            "root": str(gstreamer_root),
            "runtime_dlls": [
                {"name": path.name, "sha256": sha256(path)} for path in gstreamer_dlls
            ],
            "version_output": run(
                [gst_inspect, "--version"], env=environment, capture=True),
        },
        "platform": {"version": producer_dependencies["platform_version"]},
    })
    write_json(output / "host-candidate.json", {
        "architecture": "amd64",
        "branch": args.branch,
        "compiler": {"path": cl, "version_output": run([cl, "/?"], env=environment, capture=True)},
        "core_source_commit": commit,
        "hardware_qualified": False,
        "native_tests": {
            "exit_code": 0,
            "junit_sha256": sha256(native_junit),
            "names": test_names,
            **native_test_summary,
        },
        "profile": "pciehost/windows/amd64",
        "python": {
            "executable": str(python),
            "version": run([python, "--version"], env=environment, capture=True),
            "wheelhouse": [
                {"name": path.name, "sha256": sha256(path)} for path in wheelhouse_wheels
            ],
        },
        "qualification_resources": {
            **{
                f"hardware-tests/{name}": digest
                for name, digest in file_inventory(hardware_tests).items()
            },
            "scheduler-windows-signatures.json": sha256(
                output / "scheduler-windows-signatures.json"),
            "scheduler-windows-dependencies.json": sha256(
                output / "scheduler-windows-dependencies.json"),
            "scheduler-platform-windows.json": sha256(platform_record),
            "native-tests.xml": sha256(native_junit),
            "msi-transaction-rollback.json": sha256(
                output / "msi-transaction-rollback.json"),
            "msi-transaction-rollback.log": sha256(
                output / "msi-transaction-rollback.log"),
            "windows-dependencies.json": sha256(output / "windows-dependencies.json"),
            "windows-signatures.json": sha256(output / "windows-signatures.json"),
        },
        "repository": "sima-neat/core",
        "scheduler_source_commit": scheduler_metadata.get("commit"),
        "toolchain": {"cmake": cmake, "ninja": ninja, "wix": os.environ["WIX"]},
    })
  finally:
    customer_python = venv / "Scripts" / "python.exe"
    if customer_python.is_file():
      subprocess.run(
          [str(customer_python), "-m", "pip", "uninstall", "-y", "pyneatpcie"],
          check=False,
      )
    cleanup_errors: list[str] = []
    if core_installed:
      try:
        core_remove_code = msi_transaction(core_msi, "x", output / "core-msi-uninstall.log")
        require_no_restart(core_remove_code, "Core MSI removal")
        if scheduler_installed and scheduler_owned:
          after_core_remove = file_inventory(install_root)
          changed_scheduler = {
              name: (digest, after_core_remove.get(name))
              for name, digest in scheduler_owned.items()
              if after_core_remove.get(name) != digest
          }
          if changed_scheduler:
            cleanup_errors.append(
                f"Core MSI removal changed Scheduler-owned files: {changed_scheduler}")
      except Exception as error:  # Preserve cleanup of the remaining owner.
        cleanup_errors.append(str(error))
    if scheduler_installed:
      try:
        scheduler_remove_code = msi_transaction(
            scheduler_msi, "x", output / "scheduler-msi-uninstall.log")
        require_no_restart(scheduler_remove_code, "Scheduler MSI removal")
      except Exception as error:
        cleanup_errors.append(str(error))
    if cleanup_errors:
      raise RuntimeError("; ".join(cleanup_errors))

  if install_root.exists():
    raise RuntimeError(f"Windows cohort left files after removal: {install_root}")
  checks_path = output / "installed-checks.json"
  checks = json.loads(checks_path.read_text(encoding="utf-8"))
  checks["core_msi_remove"] = "PASS"
  checks["scheduler_msi_remove"] = "PASS"
  write_json(checks_path, checks)
  validate_candidate(output)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--branch", required=True)
  parser.add_argument("--dependency-prefix", action="append", default=[])
  parser.add_argument("--dependency-runtime-dir", action="append", default=[])
  parser.add_argument("--expected-scheduler-commit", required=True)
  parser.add_argument("--gstreamer-root", type=Path, required=True)
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--python-wheelhouse", type=Path, required=True)
  parser.add_argument("--scheduler-candidate", type=Path, required=True)
  parser.add_argument("--require-signatures", action="store_true")
  parser.add_argument("--signing-certificate-sha1", default="")
  parser.add_argument("--signing-machine-store", action="store_true")
  parser.add_argument("--signing-timestamp-url", default="https://timestamp.digicert.com")
  parser.add_argument("--work", type=Path, required=True)
  args = parser.parse_args()
  if not re.fullmatch(r"[0-9a-f]{40}", args.expected_scheduler_commit):
    parser.error("--expected-scheduler-commit must be a full lowercase Git commit")
  build(args)


if __name__ == "__main__":
  main()
