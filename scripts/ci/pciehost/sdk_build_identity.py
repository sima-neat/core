#!/usr/bin/env python3
"""Record and verify the immutable SDK image used for one Core build."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}")


def write_identity(path: Path, source_image: str, core_commit: str) -> None:
  if not IMAGE_ID.fullmatch(source_image):
    raise ValueError("SDK source image must be an immutable sha256 image ID")
  if not COMMIT.fullmatch(core_commit):
    raise ValueError("Core commit must be a full lowercase Git commit")
  if path.exists():
    raise ValueError(f"SDK build identity already exists: {path}")
  path.write_text(json.dumps({
      "core_commit": core_commit,
      "schema_version": 1,
      "source_image": source_image,
  }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_identity(path: Path, source_image: str | None,
                      core_commit: str) -> dict[str, object]:
  try:
    value = json.loads(path.read_text(encoding="utf-8"))
  except (OSError, json.JSONDecodeError) as error:
    raise ValueError(f"Core artifact has no valid SDK build identity: {path}") from error
  if (not isinstance(value, dict) or value.get("schema_version") != 1
      or value.get("core_commit") != core_commit
      or not IMAGE_ID.fullmatch(str(value.get("source_image", "")))):
    raise ValueError("Core artifact SDK build identity has an invalid cohort")
  if source_image is not None and value["source_image"] != source_image:
    raise ValueError("qualification dependency SDK image differs from the Core build")
  return value


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  subparsers = parser.add_subparsers(dest="command", required=True)
  record = subparsers.add_parser("record")
  record.add_argument("--output", type=Path, required=True)
  record.add_argument("--source-image", required=True)
  record.add_argument("--core-commit", required=True)
  verify = subparsers.add_parser("verify")
  verify.add_argument("--identity", type=Path, required=True)
  verify.add_argument("--source-image")
  verify.add_argument("--core-commit", required=True)
  args = parser.parse_args()
  if args.command == "record":
    write_identity(args.output, args.source_image, args.core_commit)
  else:
    value = validate_identity(args.identity, args.source_image, args.core_commit)
    print(value["source_image"])


if __name__ == "__main__":
  main()
