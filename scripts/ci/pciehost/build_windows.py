#!/usr/bin/env python3
"""Build and qualify the native Windows Core PCIe-host cohort.

Scheduler's MSI remains an independently produced input. This producer installs
that exact candidate, builds Core against it, and emits an ordered cohort without
copying Scheduler-owned files into the Core MSI or Python wheel.
"""

from __future__ import annotations

import argparse
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

from windows_candidate import dependency_identity


ROOT = Path(__file__).resolve().parents[3]
PCIE_HOST = ROOT / "pcie_host"
EXPECTED_MACHINE = "AMD64"
SUPPORTED_PYTHON = (3, 14)


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
  path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
      "$signature = Get-AuthenticodeSignature -LiteralPath $args[0]; "
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
  value = json.loads(run(
      [powershell, "-NoLogo", "-NoProfile", "-NonInteractive", "-Command", script, path],
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
  platform_runtime = require_directory(args.platform_runtime, "Platform runtime directory")
  platform_dlls = sorted(platform_runtime.glob("*.dll"), key=lambda path: path.name.lower())
  if not platform_dlls:
    raise ValueError(f"Platform runtime directory contains no DLLs: {platform_runtime}")
  producer_dependencies = {
      "gstreamer": {
          "pkg_config_sha256": sha256(pkg_config),
          "runtime_dlls": [
              {"name": path.name, "sha256": sha256(path)} for path in gstreamer_dlls
          ],
          "version_output": run([gst_inspect, "--version"], capture=True),
      },
      "platform_runtime_dlls": [
          {"name": path.name, "sha256": sha256(path)} for path in platform_dlls
      ],
  }
  if producer_dependencies != scheduler_dependencies:
    raise ValueError(
        "Core and Scheduler Windows producers use different Platform/GStreamer dependencies")
  wheelhouse = require_directory(args.python_wheelhouse, "Python dependency wheelhouse")
  wheelhouse_wheels = sorted(wheelhouse.glob("*.whl"))
  if not wheelhouse_wheels:
    raise ValueError(f"Python dependency wheelhouse contains no wheels: {wheelhouse}")
  dependency_prefixes = [
      require_directory(Path(value), "Windows CMake dependency prefix")
      for value in args.dependency_prefix
  ]

  cmake = require_program("cmake.exe")
  ctest = require_program("ctest.exe")
  cpack = require_program("cpack.exe")
  dumpbin = require_program("dumpbin.exe")
  cl = require_program("cl.exe")
  ninja = require_program("ninja.exe")
  powershell = require_program("pwsh.exe")
  python = require_file(Path(sys.executable), "producer Python")
  if not os.environ.get("WIX"):
    raise ValueError("WIX must name the pinned WiX v3 tool directory")

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
  environment["PATH"] = os.pathsep.join(
      (str(gstreamer_bin), str(platform_runtime), environment["PATH"]))

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

    prefixes = [install_root, gstreamer_root, *dependency_prefixes]
    prefix_path = ";".join(str(path) for path in prefixes)
    run([
        cmake,
        "-S", PCIE_HOST,
        "-B", build_dir,
        "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_INSTALL_PREFIX=" + str(stage),
        "-DCMAKE_PREFIX_PATH=" + prefix_path,
        "-DSimaScheduler_DIR=" + str(scheduler_config),
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
    run([ctest, "--test-dir", build_dir, "--output-on-failure", "--no-tests=error"],
        env=environment)

    for component in ("Runtime", "Development"):
      run([cmake, "--install", build_dir, "--config", "Release", "--prefix", stage,
           "--component", component], env=environment)
    run([cmake, "--install", build_dir, "--config", "Release", "--prefix", python_stage,
         "--component", "PcieHostPython"], env=environment)
    if args.signing_certificate_sha1:
      signtool = require_program("signtool.exe")
      sign_files(
          build_sources_for_installed(build_dir, [stage, python_stage]),
          signtool, args.signing_certificate_sha1,
          args.signing_timestamp_url, args.signing_machine_store)
      for component in ("Runtime", "Development"):
        run([cmake, "--install", build_dir, "--config", "Release", "--prefix", stage,
             "--component", component], env=environment)
      run([cmake, "--install", build_dir, "--config", "Release", "--prefix", python_stage,
           "--component", "PcieHostPython"], env=environment)
    elif args.require_signatures:
      raise RuntimeError("release mode requires a Windows signing certificate")
    stage.rename(relocated)
    hardware_tests = output / "hardware-tests"
    run([cmake, "--install", build_dir, "--config", "Release", "--prefix", hardware_tests,
         "--component", "PcieHostExtras"], env=environment)
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
      consumer_build = consumer_build_root / label
      run([
          cmake,
          "-S", PCIE_HOST / "tests" / "package_smoke",
          "-B", consumer_build,
          "-G", "Ninja",
          "-DCMAKE_BUILD_TYPE=Release",
          "-DCMAKE_PREFIX_PATH=" + ";".join((str(prefix), prefix_path)),
      ], env=environment)
      run([cmake, "--build", consumer_build], env=environment)
      binary = consumer_build / "sima_pcie_host_package_smoke.exe"
      inspect_x64(binary, dumpbin)
      consumer_environment = environment.copy()
      consumer_environment["PATH"] = os.pathsep.join(
          (str(prefix / "bin"), str(install_root / "bin"), consumer_environment["PATH"]))
      run([binary], env=consumer_environment)

    check_consumer(relocated, "relocated")
    run([gst_inspect, plugin], env=environment)

    run([cpack, "--config", build_dir / "CPackConfig.cmake", "-G", "WIX",
         "-B", output, "-C", "Release"], env=environment)
    core_packages = sorted(output.glob("sima-pcie-host-*.msi"))
    if len(core_packages) != 1:
      raise ValueError(f"expected exactly one Core MSI, found {len(core_packages)}")
    core_msi = core_packages[0]
    if args.signing_certificate_sha1:
      sign_files(
          [core_msi], signtool, args.signing_certificate_sha1,
          args.signing_timestamp_url, args.signing_machine_store)
    shutil.copy2(scheduler_msi, output / scheduler_msi.name)
    scheduler_msi = output / scheduler_msi.name

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
    ):
      require_file(required, "installed Core MSI payload")
    check_consumer(install_root, "installed")
    run([gst_inspect, plugin], env=environment)

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
    scheduler_repeat_code = msi_transaction(
        scheduler_msi, "i", output / "scheduler-msi-repeat-install.log")
    require_no_restart(scheduler_repeat_code, "Scheduler MSI repeat installation")
    core_repeat_code = msi_transaction(
        core_msi, "i", output / "core-msi-repeat-install.log")
    require_no_restart(core_repeat_code, "Core MSI repeat installation")
    after_repeat = file_inventory(install_root)
    if after_repeat != before_repeat:
      raise ValueError("ordered cohort repeat installation changed its file inventory")
    check_consumer(install_root, "repeat-installed")
    run([gst_inspect, plugin], env=environment)

    run([python, "-m", "venv", venv], env=environment)
    customer_python = venv / "Scripts" / "python.exe"
    run([
        customer_python,
        "-m", "pip", "install",
        "--no-index",
        "--find-links", wheelhouse,
        wheel,
    ], env=environment)
    import_environment = environment.copy()
    import_environment.pop("PYTHONPATH", None)
    import_environment["SIMA_PCIE_HOST_LIBRARY_PATH"] = str(install_root / "bin")
    import_environment["PATH"] = os.pathsep.join(
        (str(install_root / "bin"), import_environment["PATH"]))
    run([
        customer_python,
        "-c",
        "import pyneatpcie as p; assert p.__version__; assert p.Model; assert p.Runtime; assert p.Tensor",
    ], env=import_environment, cwd=Path(os.environ["TEMP"]))

    scheduler_version = str(scheduler_metadata.get("version", ""))
    run([
        python,
        PCIE_HOST / "scripts" / "create_windows_metadata.py",
        "--scheduler-msi", scheduler_msi,
        "--scheduler-version", scheduler_version,
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
        "plugin_load": "PASS",
        "python_import": "PASS",
        "relocated_consumer": "PASS",
        "repeat_install": "PASS",
        "scheduler_msi_install": "PASS",
        "scheduler_msi_remove": "PENDING_CLEANUP",
    })
    write_json(output / "windows-dependencies.json", {
        "cmake_prefixes": [str(path) for path in dependency_prefixes],
        "gstreamer": {
            "pkg_config_sha256": sha256(pkg_config),
            "root": str(gstreamer_root),
            "runtime_dlls": [
                {"name": path.name, "sha256": sha256(path)} for path in gstreamer_dlls
            ],
            "version_output": run(
                [gst_inspect, "--version"], env=environment, capture=True),
        },
        "platform": {
            "runtime_directory": str(platform_runtime),
            "runtime_dlls": [
                {
                    "name": path.name,
                    "sha256": sha256(path),
                    "signature": authenticode_status(path, powershell),
                }
                for path in platform_dlls
            ],
        },
    })
    write_json(output / "host-candidate.json", {
        "architecture": "amd64",
        "branch": args.branch,
        "compiler": {"path": cl, "version_output": run([cl, "/?"], env=environment, capture=True)},
        "core_source_commit": commit,
        "hardware_qualified": False,
        "native_tests": {"exit_code": 0, "names": test_names},
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


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--branch", required=True)
  parser.add_argument("--dependency-prefix", action="append", default=[])
  parser.add_argument("--expected-scheduler-commit", required=True)
  parser.add_argument("--gstreamer-root", type=Path, required=True)
  parser.add_argument("--jobs", type=int, default=4)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--platform-runtime", type=Path, required=True)
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
