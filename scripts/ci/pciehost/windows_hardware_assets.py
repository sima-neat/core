#!/usr/bin/env python3
"""Create and verify deterministic Windows PCIe hardware asset manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat


VISION_ASSETS = (
    "SIMAPCIE_YOLOV8_MODEL",
    "SIMAPCIE_MLA_ONLY_MODEL",
    "SIMAPCIE_MLA_ONLY_BF16_MODEL",
    "SIMAPCIE_TEST_IMAGE",
    "SIMAPCIE_BOXDECODE_IMAGE",
)
GENAI_ASSETS = (
    "SIMAPCIE_GENAI_MODELS_PATH",
    "SIMAPCIE_GENAI_TEST_ASSETS",
)
SHA256_PATTERN = "0123456789abcdef"


def _object_without_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
  value = {}
  for key, item in pairs:
    if key in value:
      raise ValueError(f"asset manifest contains duplicate JSON key: {key}")
    value[key] = item
  return value


def _read_manifest(path: Path) -> dict:
  try:
    status = _absolute_non_link_status(path, "hardware asset manifest")
  except OSError as error:
    raise ValueError(f"hardware asset manifest is unavailable: {path}: {error}") from error
  if not stat.S_ISREG(status.st_mode):
    raise ValueError(f"hardware asset manifest is not a regular file: {path}")
  try:
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_keys)
  except (OSError, UnicodeError, json.JSONDecodeError) as error:
    raise ValueError(f"invalid hardware asset manifest: {path}: {error}") from error
  if not isinstance(value, dict):
    raise ValueError("hardware asset manifest must be a JSON object")
  return value


def _is_sha256(value: object) -> bool:
  return (isinstance(value, str) and len(value) == 64
          and all(character in SHA256_PATTERN for character in value))


def _validate_source_revision(value: object, description: str) -> dict[str, str]:
  if not isinstance(value, dict) or not value:
    raise ValueError(f"{description} source_revision must be a non-empty object")
  allowed = {"repository", "revision", "version"}
  if "revision" not in value or not set(value).issubset(allowed):
    raise ValueError(
        f"{description} source_revision must declare revision and only source identity fields")
  normalized = {}
  for key, item in value.items():
    if (not isinstance(key, str) or not key or not isinstance(item, str)
        or not item or "\0" in key or "\0" in item):
      raise ValueError(
          f"{description} source_revision keys and values must be non-empty strings")
    normalized[key] = item
  return dict(sorted(normalized.items()))


def _validate_relative_path(value: object, description: str) -> str:
  if not isinstance(value, str) or not value or "\0" in value or "\\" in value:
    raise ValueError(f"{description} has an invalid relative path")
  path = PurePosixPath(value)
  if (path.is_absolute() or path.as_posix() != value
      or any(part in ("", ".", "..") for part in path.parts)):
    raise ValueError(f"{description} has an unsafe or non-canonical relative path: {value!r}")
  return value


def normalize_asset_manifest(value: dict) -> dict:
  """Validate and canonicalize a hardware asset manifest."""
  required_top_level = {"schema_version", "revision", "include_genai", "vision", "genai"}
  if set(value) != required_top_level:
    raise ValueError(
        "hardware asset manifest fields differ: "
        f"expected {sorted(required_top_level)}, got {sorted(value)}")
  if value["schema_version"] != 1:
    raise ValueError("hardware asset manifest schema_version must be 1")
  revision = value["revision"]
  if (not isinstance(revision, str) or not revision.strip() or len(revision) > 256
      or "\0" in revision):
    raise ValueError("hardware asset manifest revision must be a non-empty bounded string")
  if not isinstance(value["include_genai"], bool):
    raise ValueError("hardware asset manifest include_genai must be a boolean")

  expected_sections = {
      "vision": set(VISION_ASSETS),
      "genai": set(GENAI_ASSETS) if value["include_genai"] else set(),
  }
  normalized_sections = {}
  for section, expected_names in expected_sections.items():
    entries = value[section]
    if not isinstance(entries, dict) or set(entries) != expected_names:
      raise ValueError(
          f"hardware asset manifest {section} entries differ: "
          f"expected {sorted(expected_names)}, got "
          f"{sorted(entries) if isinstance(entries, dict) else type(entries).__name__}")
    normalized_entries = {}
    for name in sorted(entries):
      entry = entries[name]
      description = f"hardware asset {name}"
      if not isinstance(entry, dict):
        raise ValueError(f"{description} manifest entry must be an object")
      kind = "file" if section == "vision" else "directory"
      allowed = {"kind", "size", "sha256"} if kind == "file" else {"kind", "files"}
      if "source_revision" in entry:
        allowed.add("source_revision")
      if set(entry) != allowed or entry.get("kind") != kind:
        raise ValueError(f"{description} has invalid fields or kind")
      normalized_entry = {"kind": kind}
      if "source_revision" in entry:
        normalized_entry["source_revision"] = _validate_source_revision(
            entry["source_revision"], description)
      if kind == "file":
        if (not isinstance(entry["size"], int) or isinstance(entry["size"], bool)
            or entry["size"] <= 0 or not _is_sha256(entry["sha256"])):
          raise ValueError(f"{description} has an invalid size or SHA256")
        normalized_entry.update({"sha256": entry["sha256"], "size": entry["size"]})
      else:
        files = entry["files"]
        if not isinstance(files, list) or not files:
          raise ValueError(f"{description} files must be a non-empty list")
        normalized_files = []
        seen = set()
        for item in files:
          if not isinstance(item, dict) or set(item) != {"path", "sha256", "size"}:
            raise ValueError(f"{description} has an invalid file record")
          relative = _validate_relative_path(item["path"], description)
          ambiguous = relative.casefold()
          if ambiguous in seen:
            raise ValueError(
                f"{description} has duplicate or case-ambiguous path: {relative}")
          seen.add(ambiguous)
          if (not isinstance(item["size"], int) or isinstance(item["size"], bool)
              or item["size"] < 0 or not _is_sha256(item["sha256"])):
            raise ValueError(f"{description} file {relative!r} has an invalid size or SHA256")
          normalized_files.append({
              "path": relative,
              "sha256": item["sha256"],
              "size": item["size"],
          })
        normalized_entry["files"] = sorted(
            normalized_files, key=lambda item: (item["path"].casefold(), item["path"]))
      normalized_entries[name] = normalized_entry
    normalized_sections[section] = normalized_entries
  return {
      "schema_version": 1,
      "revision": revision,
      "include_genai": value["include_genai"],
      "vision": normalized_sections["vision"],
      "genai": normalized_sections["genai"],
  }


def read_asset_manifest(path: Path) -> dict:
  return normalize_asset_manifest(_read_manifest(path))


def asset_manifest_sha256(manifest: dict) -> str:
  encoded = json.dumps(
      normalize_asset_manifest(manifest), sort_keys=True, separators=(",", ":"),
      ensure_ascii=False).encode("utf-8")
  return hashlib.sha256(encoded).hexdigest()


def _is_reparse_point(status: os.stat_result) -> bool:
  attributes = getattr(status, "st_file_attributes", 0)
  return bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _status_identity(status: os.stat_result) -> tuple[int, int, int, int]:
  return (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)


def _absolute_non_link_status(path: Path, description: str) -> os.stat_result:
  if not path.is_absolute():
    raise ValueError(f"{description} must use an absolute path: {path}")
  current = Path(path.anchor)
  status = current.lstat()
  for part in path.parts[1:]:
    current /= part
    try:
      status = current.lstat()
    except OSError as error:
      raise ValueError(f"{description} is unavailable: {path}: {error}") from error
    if stat.S_ISLNK(status.st_mode) or _is_reparse_point(status):
      raise ValueError(f"{description} path contains a link or reparse point: {current}")
  return status


def _stable_file_identity(path: Path, status: os.stat_result, description: str) -> dict:
  flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
  try:
    descriptor = os.open(path, flags)
  except OSError as error:
    raise ValueError(f"cannot read {description}: {path}: {error}") from error
  digest = hashlib.sha256()
  try:
    opened_status = os.fstat(descriptor)
    if (not stat.S_ISREG(opened_status.st_mode)
        or _status_identity(opened_status) != _status_identity(status)):
      raise ValueError(f"{description} changed while its identity was being read: {path}")
    with os.fdopen(descriptor, "rb") as stream:
      descriptor = -1
      for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
      final_opened_status = os.fstat(stream.fileno())
    final_path_status = path.lstat()
  finally:
    if descriptor >= 0:
      os.close(descriptor)
  if (_status_identity(final_opened_status) != _status_identity(opened_status)
      or _status_identity(final_path_status) != _status_identity(opened_status)
      or stat.S_ISLNK(final_path_status.st_mode) or _is_reparse_point(final_path_status)):
    raise ValueError(f"{description} changed while its identity was being read: {path}")
  return {"sha256": digest.hexdigest(), "size": opened_status.st_size}


def _regular_file_identity(path: Path, description: str) -> dict:
  try:
    status = _absolute_non_link_status(path, description)
  except OSError as error:
    raise ValueError(f"{description} is unavailable: {path}: {error}") from error
  if not stat.S_ISREG(status.st_mode):
    raise ValueError(f"{description} must be a regular non-link file: {path}")
  return {"kind": "file", **_stable_file_identity(path, status, description)}


def _directory_file_inventory(root: Path, description: str) -> list[dict]:
  try:
    root_status = _absolute_non_link_status(root, description)
  except OSError as error:
    raise ValueError(f"{description} is unavailable: {root}: {error}") from error
  if not stat.S_ISDIR(root_status.st_mode):
    raise ValueError(f"{description} must be a regular non-link directory: {root}")

  inventory = []
  seen = set()

  def visit(directory: Path, parts: tuple[str, ...]) -> None:
    try:
      with os.scandir(directory) as directory_entries:
        ordered = sorted(directory_entries, key=lambda entry: (entry.name.casefold(), entry.name))
    except OSError as error:
      raise ValueError(f"cannot inventory {description}: {directory}: {error}") from error
    local_names = set()
    for entry in ordered:
      folded_name = entry.name.casefold()
      if folded_name in local_names:
        raise ValueError(
            f"{description} has duplicate or case-ambiguous entry: "
            f"{'/'.join((*parts, entry.name))}")
      local_names.add(folded_name)
      relative = "/".join((*parts, entry.name))
      _validate_relative_path(relative, description)
      folded_relative = relative.casefold()
      if folded_relative in seen:
        raise ValueError(f"{description} has duplicate or case-ambiguous path: {relative}")
      seen.add(folded_relative)
      child = directory / entry.name
      try:
        status = child.lstat()
      except OSError as error:
        raise ValueError(f"cannot inspect {description} entry {relative}: {error}") from error
      if entry.is_symlink() or _is_reparse_point(status):
        raise ValueError(f"{description} contains a link or reparse point: {relative}")
      if stat.S_ISDIR(status.st_mode):
        visit(child, (*parts, entry.name))
      elif stat.S_ISREG(status.st_mode):
        inventory.append({
            "path": relative,
            **_stable_file_identity(child, status, f"{description} entry {relative}"),
        })
      else:
        raise ValueError(f"{description} contains a non-regular entry: {relative}")

  visit(root, ())
  return sorted(inventory, key=lambda item: (item["path"].casefold(), item["path"]))


def create_asset_manifest(environment: dict[str, str], include_genai: bool, revision: str,
                          source_revisions: dict[str, dict[str, str]] | None = None) -> dict:
  """Inventory the exact assets named by the qualification environment."""
  source_revisions = source_revisions or {}
  allowed_sources = set(VISION_ASSETS) | (set(GENAI_ASSETS) if include_genai else set())
  if not set(source_revisions).issubset(allowed_sources):
    raise ValueError("source revision metadata names an asset outside this manifest")
  value = {
      "schema_version": 1,
      "revision": revision,
      "include_genai": include_genai,
      "vision": {},
      "genai": {},
  }
  for name in VISION_ASSETS:
    path = Path(environment.get(name, ""))
    entry = _regular_file_identity(path, f"hardware asset {name}")
    if name in source_revisions:
      entry["source_revision"] = source_revisions[name]
    value["vision"][name] = entry
  if include_genai:
    for name in GENAI_ASSETS:
      path = Path(environment.get(name, ""))
      entry = {
          "kind": "directory",
          "files": _directory_file_inventory(path, f"hardware asset {name}"),
      }
      if name in source_revisions:
        entry["source_revision"] = source_revisions[name]
      value["genai"][name] = entry
  return normalize_asset_manifest(value)


def verify_asset_manifest(path: Path, environment: dict[str, str], include_genai: bool) -> dict:
  expected = read_asset_manifest(path)
  if expected["include_genai"] != include_genai:
    raise ValueError(
        "hardware asset manifest GenAI scope differs from qualification request")
  source_revisions = {
      name: entry["source_revision"]
      for section in ("vision", "genai")
      for name, entry in expected[section].items()
      if "source_revision" in entry
  }
  actual = create_asset_manifest(
      environment, include_genai, expected["revision"], source_revisions)
  if actual != expected:
    raise ValueError("hardware assets differ from the admitted checksum manifest")
  return expected


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--include-genai", action="store_true")
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--revision", required=True)
  parser.add_argument(
      "--source-revision", action="append", default=[], metavar="ASSET:KEY=VALUE",
      help="bind optional source metadata to an asset (repeatable)")
  args = parser.parse_args()
  revisions: dict[str, dict[str, str]] = {}
  for declaration in args.source_revision:
    match = declaration.partition(":")
    metadata = match[2].partition("=")
    if not match[0] or not match[1] or not metadata[0] or not metadata[1]:
      parser.error("--source-revision must use ASSET:KEY=VALUE")
    revisions.setdefault(match[0], {})[metadata[0]] = metadata[2]
  manifest = create_asset_manifest(os.environ, args.include_genai, args.revision, revisions)
  args.output.write_text(
      json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main()
