#!/usr/bin/env python3
"""Bind the standalone Core offline card bundle to the exact Linux candidate.

The Windows card lane installs the candidate on the card the way the Linux
offline installer does. This checks that the bundle was assembled from this
run's neat-core-dist before the self-hosted card job receives it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile

from qualify_windows_hardware import CARD_BUNDLE_ARCHIVE, validate_card_bundle
from windows_candidate import sha256, write_json

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "pcie_host" / "scripts" / "offline"))
import assemble  # noqa: E402


def prepare(source: Path, bundle_directory: Path, output: Path, expected_commit: str) -> None:
  source = source.resolve(strict=True)
  bundle = validate_card_bundle(bundle_directory, expected_commit)
  source_metadata = source / "metadata.json"
  if bundle.metadata.get("source_metadata_sha256") != {"card": sha256(source_metadata)}:
    raise ValueError("card bundle was not assembled from this Linux candidate's metadata")
  with tempfile.TemporaryDirectory() as directory:
    with tarfile.open(bundle.archive, "r:gz") as archive:
      archive.extractall(directory, filter="data")
    root = Path(directory) / "neat-core-offline"
    manifest = assemble.verify_bundle(root)
    if manifest.get("scope") != "core" or manifest.get("core_revision") != expected_commit[:12]:
      raise ValueError("card bundle is not the standalone Core bundle of this commit")
    payload = root / "card" / "payload" / "metadata.json"
    if payload.read_bytes() != source_metadata.read_bytes():
      raise ValueError("card bundle payload metadata differs from this Linux candidate")
    target = json.loads((root / "card" / "target.json").read_text(encoding="utf-8"))
  output.mkdir(parents=True, exist_ok=False)
  shutil.copy2(bundle.archive, output / CARD_BUNDLE_ARCHIVE)
  shutil.copy2(bundle_directory / "metadata.json", output / "metadata.json")
  write_json(output / "windows-card-offline-bundle.json", {
      "schema_version": 1,
      "core_commit": expected_commit,
      "bundle_sha256": bundle.sha256,
      "metadata_sha256": sha256(output / "metadata.json"),
      "source_metadata_sha256": sha256(source_metadata),
      "baseline_image": target.get("image"),
  })


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--source", type=Path, required=True)
  parser.add_argument("--bundle", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--expected-commit", required=True)
  args = parser.parse_args()
  prepare(args.source, args.bundle, args.output, args.expected_commit)


if __name__ == "__main__":
  main()
