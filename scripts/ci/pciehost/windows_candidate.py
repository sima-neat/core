#!/usr/bin/env python3
"""Validate an exact Windows PCIe-host cohort produced by Core CI."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET


_OPTIONAL_EVIDENCE_FILES = {
    "core-msi-install.log",
    "core-msi-repeat-install.log",
    "core-msi-uninstall.log",
    "installed-checks.json",
    "scheduler-host-candidate.json",
    "scheduler-installed-checks.json",
    "scheduler-metadata.json",
    "scheduler-msi-install.log",
    "scheduler-msi-repeat-install.log",
    "scheduler-msi-uninstall.log",
}


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def msi_version(value: object) -> tuple[int, int, int]:
  match = re.fullmatch(r"([0-9]+)[.]([0-9]+)[.]([0-9]+)", str(value))
  if not match:
    raise ValueError(f"Windows candidate version is not numeric major.minor.patch: {value!r}")
  return tuple(int(part) for part in match.groups())


def require_junit_cases(path: Path, expected: set[str]) -> dict[str, int]:
  """Require an exact, passing, zero-skip JUnit case set."""
  try:
    root = ET.parse(path).getroot()
  except (OSError, ET.ParseError) as error:
    raise ValueError(f"invalid JUnit evidence: {path}: {error}") from error
  cases = list(root.iter("testcase"))
  names = [case.get("name", "") for case in cases]
  if not cases or any(not name for name in names) or len(names) != len(set(names)):
    raise ValueError(f"JUnit evidence has missing or duplicate test cases: {path}")
  missing = sorted(expected - set(names))
  unexpected = sorted(set(names) - expected)
  if missing or unexpected:
    raise ValueError(
        f"JUnit evidence differs from the required case set: "
        f"missing={missing}, unexpected={unexpected}")
  skipped = sum(1 for case in cases if case.find("skipped") is not None)
  failures = sum(1 for case in cases if case.find("failure") is not None)
  errors = sum(1 for case in cases if case.find("error") is not None)
  if skipped or failures or errors:
    raise ValueError(
        f"JUnit evidence is not a zero-skip pass: "
        f"skipped={skipped}, failures={failures}, errors={errors}")
  return {"tests": len(cases), "skipped": 0, "failures": 0, "errors": 0}


def dependency_identity(directory: Path, *, require_dependency_runtime: bool = False) -> dict:
  root = directory.resolve(strict=True)
  receipt = _read_json(root / "windows-dependencies.json", "Windows dependency receipt")
  platform = receipt.get("platform")
  runtime_dlls = platform.get("runtime_dlls") if isinstance(platform, dict) else None
  platform_version = platform.get("version") if isinstance(platform, dict) else None
  if (not isinstance(platform_version, str)
      or not re.fullmatch(r"[0-9]+(?:[.][0-9]+){2,3}", platform_version)):
    raise ValueError("Windows dependency receipt has no exact Platform version")
  if not isinstance(runtime_dlls, list) or not runtime_dlls:
    raise ValueError("Windows dependency receipt has no Platform runtime DLLs")
  dlls = []
  for item in runtime_dlls:
    if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
        or Path(item["name"]).name != item["name"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
      raise ValueError("Windows dependency receipt has an invalid Platform runtime DLL")
    dlls.append({"name": item["name"], "sha256": item["sha256"]})
  if len({item["name"].lower() for item in dlls}) != len(dlls):
    raise ValueError("Windows dependency receipt has duplicate Platform runtime DLLs")

  gstreamer = receipt.get("gstreamer")
  version_output = gstreamer.get("version_output") if isinstance(gstreamer, dict) else None
  pkg_config_sha256 = gstreamer.get("pkg_config_sha256") if isinstance(gstreamer, dict) else None
  gstreamer_dlls = gstreamer.get("runtime_dlls") if isinstance(gstreamer, dict) else None
  if (not isinstance(version_output, str) or not version_output.strip()
      or not re.fullmatch(r"[0-9a-f]{64}", str(pkg_config_sha256 or ""))
      or not isinstance(gstreamer_dlls, list) or not gstreamer_dlls):
    raise ValueError("Windows dependency receipt has an invalid GStreamer identity")
  normalized_gstreamer_dlls = []
  for item in gstreamer_dlls:
    if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
        or Path(item["name"]).name != item["name"]
        or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
      raise ValueError("Windows dependency receipt has an invalid GStreamer runtime DLL")
    normalized_gstreamer_dlls.append(
        {"name": item["name"], "sha256": item["sha256"]})
  if (len({item["name"].lower() for item in normalized_gstreamer_dlls})
      != len(normalized_gstreamer_dlls)):
    raise ValueError("Windows dependency receipt has duplicate GStreamer runtime DLLs")
  identity = {
      "platform_version": platform_version,
      "gstreamer": {
          "pkg_config_sha256": pkg_config_sha256,
          "runtime_dlls": sorted(
              normalized_gstreamer_dlls, key=lambda item: item["name"].lower()),
          "version_output": version_output,
      },
      "platform_runtime_dlls": sorted(dlls, key=lambda item: item["name"].lower()),
  }
  runtime_cohorts = receipt.get("dependency_runtime")
  if runtime_cohorts is None and not require_dependency_runtime:
    return identity
  if not isinstance(runtime_cohorts, list) or not runtime_cohorts:
    raise ValueError("Windows dependency receipt has no additional runtime DLL cohort")
  normalized_cohorts = []
  for cohort in runtime_cohorts:
    files = cohort.get("files") if isinstance(cohort, dict) else None
    if not isinstance(files, list) or not files:
      raise ValueError("Windows dependency receipt has an invalid runtime DLL cohort")
    normalized = []
    for item in files:
      if (not isinstance(item, dict) or not isinstance(item.get("name"), str)
          or Path(item["name"]).name != item["name"]
          or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", "")))):
        raise ValueError("Windows dependency receipt has an invalid additional runtime DLL")
      normalized.append({"name": item["name"], "sha256": item["sha256"]})
    if len({item["name"].lower() for item in normalized}) != len(normalized):
      raise ValueError("Windows dependency receipt has duplicate additional runtime DLLs")
    normalized_cohorts.append(sorted(normalized, key=lambda item: item["name"].lower()))
  identity["dependency_runtime"] = normalized_cohorts
  return identity


def _read_json(path: Path, description: str) -> dict:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"invalid {description}: {path}: {error}") from error
  if not isinstance(value, dict):
    raise ValueError(f"{description} must be a JSON object: {path}")
  return value


def _reject_uninventoried_entries(root: Path, allowed_files: set[str]) -> None:
  allowed_directories = set()
  for name in allowed_files:
    relative = Path(name)
    for parent in relative.parents:
      if parent == Path("."):
        break
      allowed_directories.add(parent.as_posix())

  unexpected = []
  for path in root.rglob("*"):
    relative = path.relative_to(root).as_posix()
    if path.is_symlink():
      unexpected.append(relative)
    elif path.is_file() and relative not in allowed_files:
      unexpected.append(relative)
    elif path.is_dir() and relative not in allowed_directories:
      unexpected.append(relative + "/")
  if unexpected:
    ordered = sorted(unexpected)
    preview = ordered[:20]
    suffix = (
        f" (and {len(ordered) - len(preview)} more)"
        if len(ordered) > len(preview) else "")
    raise ValueError(
        f"candidate contains uninventoried files or directories: {preview}{suffix}")


@dataclass(frozen=True)
class WindowsCandidate:
  root: Path
  metadata: dict
  receipt: dict
  scheduler_msi: Path
  core_msi: Path
  wheel: Path
  resources: dict[str, str]

  @property
  def core_commit(self) -> str:
    return str(self.metadata["commit"])


def validate_candidate(directory: Path) -> WindowsCandidate:
  root = directory.resolve(strict=True)
  if not root.is_dir():
    raise ValueError(f"candidate is not a directory: {root}")

  metadata = _read_json(root / "metadata.json", "candidate metadata")
  receipt = _read_json(root / "host-candidate.json", "host candidate receipt")
  if metadata.get("name") != "gh:sima-neat/core/pciehost/windows/amd64":
    raise ValueError("candidate metadata has the wrong package name")
  if metadata.get("repository") != "sima-neat/core":
    raise ValueError("candidate metadata has the wrong repository")
  artifact = metadata.get("artifact")
  if not isinstance(artifact, dict) or artifact.get("package_path") != "pciehost/windows/amd64":
    raise ValueError("candidate metadata has the wrong artifact profile")
  if receipt.get("repository") != "sima-neat/core":
    raise ValueError("candidate receipt has the wrong repository")
  if receipt.get("profile") != "pciehost/windows/amd64":
    raise ValueError("candidate receipt has the wrong profile")
  if metadata.get("commit") != receipt.get("core_source_commit"):
    raise ValueError("candidate metadata and receipt name different Core commits")
  msi_version(metadata.get("version"))
  dependency_identity(root, require_dependency_runtime=True)

  qualification_resources = receipt.get("qualification_resources")
  if not isinstance(qualification_resources, dict) or not qualification_resources:
    raise ValueError("candidate receipt has no qualification resource inventory")
  required_qualification_resources = {
      "scheduler-windows-signatures.json",
      "scheduler-windows-dependencies.json",
      "windows-dependencies.json",
      "windows-signatures.json",
  }
  missing_qualification_resources = sorted(
      required_qualification_resources - set(qualification_resources))
  if missing_qualification_resources:
    raise ValueError(
        "candidate receipt is missing required qualification resources: "
        f"{missing_qualification_resources}")
  for name, expected in qualification_resources.items():
    relative = Path(str(name))
    if (relative.is_absolute() or ".." in relative.parts or not relative.parts
        or not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)):
      raise ValueError(f"invalid qualification resource: {name!r}")
    path = root / relative
    if not path.is_file() or path.is_symlink():
      raise ValueError(f"candidate qualification resource is missing or unsafe: {name}")
    if sha256(path) != expected:
      raise ValueError(f"candidate qualification resource checksum differs: {name}")

  resource_names = metadata.get("resources")
  resources = metadata.get("resources-checksum")
  if (not isinstance(resource_names, list) or not resource_names
      or not isinstance(resources, dict) or set(resource_names) != set(resources)):
    raise ValueError("candidate resource inventory is incomplete")
  for name, expected in resources.items():
    if not isinstance(name, str) or Path(name).name != name:
      raise ValueError(f"invalid candidate resource name: {name!r}")
    if not isinstance(expected, str) or len(expected) != 64:
      raise ValueError(f"invalid candidate checksum: {name}")
    path = root / name
    if not path.is_file() or path.is_symlink():
      raise ValueError(f"candidate resource is missing or unsafe: {name}")
    if sha256(path) != expected.lower():
      raise ValueError(f"candidate resource checksum differs: {name}")

  allowed_files = {
      "host-candidate.json",
      "metadata.json",
      *_OPTIONAL_EVIDENCE_FILES,
      *resources,
      *qualification_resources,
      *(f"{name}.sha256" for name in resources),
  }
  _reject_uninventoried_entries(root, allowed_files)

  installation = metadata.get("installation")
  if not isinstance(installation, dict) or installation.get("type") != "windows-msi":
    raise ValueError("candidate installation is not windows-msi")
  packages = installation.get("native-packages")
  if not isinstance(packages, list) or [item.get("owner") for item in packages] != [
      "scheduler", "core"]:
    raise ValueError("candidate native package order must be Scheduler then Core")
  msi_names = [item.get("msi") for item in packages]
  if any(name not in resources or not str(name).lower().endswith(".msi") for name in msi_names):
    raise ValueError("candidate native package metadata does not select two MSI resources")

  python_package = installation.get("python")
  if not isinstance(python_package, dict):
    raise ValueError("candidate has no Python installation record")
  wheel_name = python_package.get("wheel")
  if wheel_name not in resources or not str(wheel_name).lower().endswith(".whl"):
    raise ValueError("candidate Python metadata does not select a wheel resource")

  return WindowsCandidate(
      root=root,
      metadata=metadata,
      receipt=receipt,
      scheduler_msi=root / msi_names[0],
      core_msi=root / msi_names[1],
      wheel=root / wheel_name,
      resources={str(name): str(digest).lower() for name, digest in resources.items()},
  )
