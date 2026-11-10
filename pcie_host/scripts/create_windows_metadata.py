#!/usr/bin/env python3
"""Create sima-cli metadata for the Windows MSI and optional Python wheel."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def metadata(msi: Path, wheel: Path, version: str, branch: str, commit: str,
             scheduler_msi: Path = None, scheduler_version: str = None) -> dict:
  resources = []
  installation = {
      "type": "windows-msi",
      "post-message": (
          "[bold]Native Neat PCIe host installation completed.[/bold]\n"
          "Use --python-executable <path> to install Python support into an application environment."
      ),
  }
  artifact_paths = {}
  if scheduler_msi:
    resources.append(scheduler_msi.name)
    artifact_paths[scheduler_msi.name] = scheduler_msi
    installation["native-packages"] = [
        {
            "owner": "scheduler",
            "msi": scheduler_msi.name,
            "version": scheduler_version or version,
        },
        {"owner": "core", "msi": msi.name, "version": version},
    ]
  else:
    installation["msi"] = msi.name
  resources.append(msi.name)
  artifact_paths[msi.name] = msi
  total_bytes = msi.stat().st_size
  if scheduler_msi:
    total_bytes += scheduler_msi.stat().st_size
  if wheel:
    resources.append(wheel.name)
    artifact_paths[wheel.name] = wheel
    total_bytes += wheel.stat().st_size
    installation["python"] = {
        "wheel": wheel.name,
        "distribution": "pyneatpcie",
        "version": version,
        "import": "pyneatpcie",
    }

  checksums = {resource: sha256(artifact_paths[resource]) for resource in resources}
  total_mebibytes = max(1, (total_bytes + 1024 * 1024 - 1) // (1024 * 1024))
  return {
      "name": "gh:sima-neat/core/pciehost/windows/amd64",
      "version": version,
      "release": branch,
      "description": "SiMa.ai Neat PCIe host for Windows x64",
      "platforms": [{
          "type": "host",
          "os": ["windows"],
          "versions": {"windows": [">=11"]},
          "arch": ["amd64"],
      }],
      "resources": resources,
      "resources-checksum": checksums,
      "selectable-resources": [],
      "installation": installation,
      "artifact": {
          "type": "windows-msi",
          "repository": "core",
          "package_path": "pciehost/windows/amd64",
          "distro": "windows",
          "arches": ["amd64"],
      },
      "repository": "sima-neat/core",
      "branch": branch,
      "commit": commit,
      "published_at_utc": datetime.now(timezone.utc).isoformat(),
      "size": {
          "download": f"{total_mebibytes}MB",
          "install": f"{total_mebibytes}MB",
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--msi", required=True, type=Path)
  parser.add_argument("--scheduler-msi", type=Path)
  parser.add_argument("--scheduler-version")
  parser.add_argument("--wheel", type=Path)
  parser.add_argument("--version", required=True)
  parser.add_argument("--branch", required=True)
  parser.add_argument("--commit", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  for path in (args.msi, args.scheduler_msi, args.wheel):
    if path and not path.is_file():
      parser.error(f"artifact does not exist: {path}")
  args.output.write_text(
      json.dumps(
          metadata(args.msi.resolve(), args.wheel.resolve() if args.wheel else None,
                   args.version, args.branch, args.commit,
                   args.scheduler_msi.resolve() if args.scheduler_msi else None,
                   args.scheduler_version),
          indent=2,
      ) + "\n",
      encoding="utf-8",
  )


if __name__ == "__main__":
  main()
