#!/usr/bin/env python3
"""Validate an exact Windows PCIe-host cohort produced by Core CI."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def _read_json(path: Path, description: str) -> dict:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"invalid {description}: {path}: {error}") from error
  if not isinstance(value, dict):
    raise ValueError(f"{description} must be a JSON object: {path}")
  return value


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

  qualification_resources = receipt.get("qualification_resources")
  if not isinstance(qualification_resources, dict) or not qualification_resources:
    raise ValueError("candidate receipt has no qualification resource inventory")
  required_qualification_resources = {
      "scheduler-windows-signatures.json",
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
