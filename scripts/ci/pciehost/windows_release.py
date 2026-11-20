#!/usr/bin/env python3
"""Name, list, verify and fetch a published Windows PCIe-host release.

The publisher, the hardware qualifier's local registry and the upgrade
baseline use these helpers, so they share the layout sima-cli resolves for
``core/pciehost/windows/amd64@<tag>``: ``core/<tag>/latest.tag`` names the
12-character commit and ``core/<tag>/<commit12>/pciehost/windows/amd64/`` holds
``metadata.json`` and its resources, as for the Ubuntu host packages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable
from urllib.parse import quote
from urllib.request import urlopen

from windows_candidate import msi_version, validate_published_package


PACKAGE_PATH = "pciehost/windows/amd64"


def release_tag(version: object) -> str:
  msi_version(version)
  return f"v{version}"


def require_release_ref(ref_type: str, ref_name: str, version: object) -> str:
  """Publication runs only from the Neat release tag of the packaged version."""
  expected = release_tag(version)
  if ref_type != "tag" or ref_name != expected:
    raise ValueError(
        f"Windows publication requires the release tag {expected}; "
        f"this run is on {ref_type or 'unknown'} {ref_name!r}")
  return expected


def short_commit(commit: str) -> str:
  if not re.fullmatch(r"[0-9a-f]{40}", commit):
    raise ValueError(f"expected a full lowercase Git commit: {commit!r}")
  return commit[:12]


def channel(ref: str) -> str:
  if not ref.strip():
    raise ValueError("release ref is empty")
  return f"core/{quote(ref.strip(), safe='')}"


def latest_tag_key(ref: str) -> str:
  return f"{channel(ref)}/latest.tag"


def package_prefix(ref: str, commit: str) -> str:
  return f"{channel(ref)}/{short_commit(commit)}/{PACKAGE_PATH}"


def customer_target(ref: str) -> str:
  return f"core/{PACKAGE_PATH}@{ref}"


def publication_plan(read: Callable[[str], bytes], ref: str, commit: str,
                     directory: Path, *, ref_type: str = "tag") -> tuple[str, list[Path]]:
  """Return the upload prefix and files: resources first, metadata.json last.

  Only what sima-cli downloads is published. A branch build publishes beside
  the branch's Ubuntu packages before promotion, as they do. A tag is published
  only by the qualified release dispatch: ``read`` must already resolve its
  ``latest.tag`` to this commit, because Vulcan CI promotes the tag channel
  after the Linux release and the Windows files join that commit directory.
  """
  package = validate_published_package(directory, allow_unlisted=True)
  if ref_type == "tag":
    require_release_ref("tag", ref, package.metadata["version"])
  elif ref_type != "branch":
    raise ValueError(f"Windows packages publish from a branch or tag, not {ref_type!r}")
  if package.core_commit != commit:
    raise ValueError(f"Windows package names commit {package.core_commit}, not {commit}")
  if ref_type == "tag":
    _require_latest(read, ref, commit)
  names = package.metadata["resources"]
  files = [package.root / name for name in names]
  files += [path for path in (package.root / f"{name}.sha256" for name in names)
            if path.is_file()]
  return package_prefix(ref, commit), [*files, package.root / "metadata.json"]


def _require_latest(read: Callable[[str], bytes], ref: str, commit: str) -> None:
  latest = read(latest_tag_key(ref)).decode("ascii").strip()
  if latest != short_commit(commit):
    raise ValueError(
        f"{latest_tag_key(ref)} names {latest!r}, not release commit {commit[:12]}")


def fetch_release(read: Callable[[str], bytes], ref: str, commit: str, output: Path,
                  *, metadata_name: str = "metadata.json", require_latest: bool = True):
  """Download one published release through ``read(key)`` and verify every byte."""
  output = Path(output)
  output.mkdir(parents=True, exist_ok=False)
  if require_latest:
    _require_latest(read, ref, commit)
  prefix = package_prefix(ref, commit)
  (output / "metadata.json").write_bytes(read(f"{prefix}/{metadata_name}"))
  metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
  checksums = metadata.get("resources-checksum")
  if not isinstance(checksums, dict) or not checksums:
    raise ValueError(f"{prefix}/{metadata_name} has no resource checksums")
  for name in checksums:
    if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
      raise ValueError(f"published resource has an unsafe name: {name!r}")
    (output / name).write_bytes(read(f"{prefix}/{name}"))
  package = validate_published_package(output)
  if package.core_commit != commit:
    raise ValueError(f"{prefix}/{metadata_name} names commit {package.core_commit}")
  return package


def https_reader(base_url: str) -> Callable[[str], bytes]:
  def read(key: str) -> bytes:
    with urlopen(f"{base_url.rstrip('/')}/{key}", timeout=300) as response:
      return response.read()
  return read


def s3_reader(bucket: str) -> Callable[[str], bytes]:
  def read(key: str) -> bytes:
    result = subprocess.run(["aws", "s3", "cp", f"s3://{bucket}/{key}", "-"],
                            check=False, capture_output=True)
    if result.returncode:
      raise ValueError(f"cannot read s3://{bucket}/{key}: "
                       + result.stderr.decode("utf-8", errors="replace").strip())
    return result.stdout
  return read


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  commands = parser.add_subparsers(dest="command", required=True)
  ref_check = commands.add_parser("release-ref", help="reject a run that cannot publish")
  ref_check.add_argument("--ref-type", required=True)
  ref_check.add_argument("--ref-name", required=True)
  ref_check.add_argument("--manifest", type=Path, required=True)
  for name in ("publication", "verify-upload"):
    command = commands.add_parser(name)
    command.add_argument("--bucket", required=True)
    command.add_argument("--ref-type", choices=("branch", "tag"), required=True)
    command.add_argument("--ref", required=True)
    command.add_argument("--commit", required=True)
    command.add_argument("--candidate", type=Path, required=True)
    if name == "verify-upload":
      command.add_argument("--output", type=Path, required=True)
  fetch = commands.add_parser("fetch", help="download and verify a published release")
  fetch.add_argument("--base-url", required=True)
  fetch.add_argument("--ref", required=True)
  fetch.add_argument("--commit", required=True)
  fetch.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()

  if args.command == "release-ref":
    version = json.loads(args.manifest.read_text(encoding="utf-8"))["package-version"]
    print(require_release_ref(args.ref_type, args.ref_name, version))
  elif args.command == "publication":
    prefix, files = publication_plan(
        s3_reader(args.bucket), args.ref, args.commit, args.candidate, ref_type=args.ref_type)
    # First line: destination prefix; then one source file per line.
    print("\n".join([prefix, *map(str, files)]))
  elif args.command == "fetch":
    package = fetch_release(https_reader(args.base_url), args.ref, args.commit, args.output)
    print(json.dumps({"commit": package.core_commit, "resources": package.resources,
                      "version": package.metadata["version"]}, sort_keys=True))
  else:
    expected = validate_published_package(args.candidate, allow_unlisted=True)
    package = fetch_release(s3_reader(args.bucket), args.ref, args.commit, args.output,
                            metadata_name="metadata.json.pending",
                            require_latest=args.ref_type == "tag")
    if package.metadata != expected.metadata:
      raise ValueError("uploaded Windows release differs from the qualified candidate")
    print(json.dumps({"status": "PASS_WINDOWS_PUBLISHED_RELEASE",
                      "resources": len(package.resources)}))


if __name__ == "__main__":
  try:
    main()
  except (OSError, ValueError) as error:
    print(f"error: {error}", file=sys.stderr)
    sys.exit(1)
