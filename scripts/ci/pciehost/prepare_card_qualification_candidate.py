#!/usr/bin/env python3
"""Add resolved ARM64 runtime DEBs to a private card qualification candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess

from qualify_windows_hardware import validate_card_candidate, write_json
from windows_candidate import sha256


def deb_identity(path: Path) -> dict[str, str]:
  result = subprocess.run(
      ["dpkg-deb", "-W", "--showformat=${Package}\\n${Version}\\n${Architecture}\\n",
       str(path)],
      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30, check=False)
  fields = result.stdout.splitlines()
  if result.returncode != 0 or len(fields) != 3:
    raise ValueError(f"could not read Debian package identity: {path}")
  package, version, architecture = fields
  if not re.fullmatch(r"[a-z0-9][a-z0-9+.-]+", package) or not version:
    raise ValueError(f"Debian package identity is invalid: {path}")
  if architecture != "arm64":
    raise ValueError(f"card qualification dependency is not ARM64: {path}")
  return {
      "architecture": architecture,
      "file": path.name,
      "package": package,
      "sha256": sha256(path),
      "version": version,
  }


def prepare(source: Path, dependencies: Path, output: Path,
            expected_commit: str) -> None:
  source = source.resolve(strict=True)
  dependencies = dependencies.resolve(strict=True)
  if not dependencies.is_dir():
    raise ValueError(f"card dependency input is not a directory: {dependencies}")
  validate_card_candidate(source, expected_commit)
  dependency_debs = sorted(dependencies.glob("*.deb"), key=lambda path: path.name)
  if not dependency_debs:
    raise ValueError("card dependency input contains no Debian packages")
  if output.resolve().exists():
    raise ValueError(f"private card qualification candidate already exists: {output}")

  identities = [deb_identity(path) for path in dependency_debs]
  package_names = [value["package"] for value in identities]
  if len(package_names) != len(set(package_names)):
    raise ValueError("card dependency input contains duplicate package identities")

  shutil.copytree(source, output)
  metadata_path = output / "metadata-all.json"
  metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
  resources = dict(metadata["resources-checksum"])
  for path, identity in zip(dependency_debs, identities, strict=True):
    if path.name in resources or (output / path.name).exists():
      raise ValueError(f"card dependency duplicates a candidate resource: {path.name}")
    shutil.copy2(path, output / path.name)
    resources[path.name] = identity["sha256"]

  receipt_name = "windows-card-qualification-dependencies.json"
  receipt = {
      "dependencies": identities,
      "purpose": "private non-installing Windows PCIe card qualification",
      "source_core_commit": expected_commit,
      "source_metadata_sha256": sha256(source / "metadata-all.json"),
  }
  write_json(output / receipt_name, receipt)
  resources[receipt_name] = sha256(output / receipt_name)
  metadata["resources-checksum"] = resources
  metadata["resources"] = sorted(resources)
  write_json(metadata_path, metadata)
  validate_card_candidate(output, expected_commit)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--dependencies", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--expected-commit", required=True)
  args = parser.parse_args()
  prepare(args.source, args.dependencies, args.output, args.expected_commit)


if __name__ == "__main__":
  main()
