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


def package_size(path: Path) -> str:
  mebibytes = max(1, (path.stat().st_size + 1024 * 1024 - 1) // (1024 * 1024))
  return f"{mebibytes}MB"


def metadata(msi: Path, wheel: Path, version: str, branch: str, commit: str) -> dict:
  resources = [msi.name]
  installation = {
      "type": "windows-msi",
      "msi": msi.name,
      "post-message": (
          "[bold]Native Neat PCIe host installation completed.[/bold]\n"
          "Use --python-executable <path> to install Python support into an application environment."
      ),
  }
  total_bytes = msi.stat().st_size
  if wheel:
    resources.append(wheel.name)
    total_bytes += wheel.stat().st_size
    installation["python"] = {
        "wheel": wheel.name,
        "distribution": "pyneatpcie",
        "version": version,
        "import": "pyneatpcie",
    }

  checksums = {resource: sha256(msi if resource == msi.name else wheel) for resource in resources}
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
          "install": package_size(msi),
      },
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--msi", required=True, type=Path)
  parser.add_argument("--wheel", type=Path)
  parser.add_argument("--version", required=True)
  parser.add_argument("--branch", required=True)
  parser.add_argument("--commit", required=True)
  parser.add_argument("--output", required=True, type=Path)
  args = parser.parse_args()
  for path in (args.msi, args.wheel):
    if path and not path.is_file():
      parser.error(f"artifact does not exist: {path}")
  args.output.write_text(
      json.dumps(
          metadata(args.msi.resolve(), args.wheel.resolve() if args.wheel else None,
                   args.version, args.branch, args.commit),
          indent=2,
      ) + "\n",
      encoding="utf-8",
  )


if __name__ == "__main__":
  main()
