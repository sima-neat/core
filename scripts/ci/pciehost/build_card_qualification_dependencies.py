#!/usr/bin/env python3
"""Build the private ARM64 runtime bundle used by Windows card qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

from sdk_build_identity import validate_identity


LIBRARY_DIRECTORY = Path("usr/lib/aarch64-linux-gnu")
REQUIRED_LIBRARIES = (
    "libgstrtspserver-1.0.so.0",
    "libcpp-httplib.so.0.18",
    "libspdlog.so.1.15",
    "libfmt.so.10",
    "libfftw3.so.3",
)
IMAGE_IDENTITY = re.compile(
    r"(?:[a-z0-9][a-z0-9._/-]*@)?sha256:[0-9a-f]{64}")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as source:
    for chunk in iter(lambda: source.read(1024 * 1024), b""):
      digest.update(chunk)
  return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
  path.write_text(
      json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def image_digest(identity: str) -> str:
  if not IMAGE_IDENTITY.fullmatch(identity):
    raise ValueError(
        "SDK image identity must be an immutable sha256 image ID or repository digest")
  return identity.rsplit("sha256:", 1)[1]


def elf_soname(path: Path) -> str:
  header = subprocess.run(
      ["readelf", "-h", str(path)], stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, text=True, timeout=30, check=False)
  if (header.returncode != 0 or "Class:                             ELF64" not in header.stdout
      or "Data:                              2's complement, little endian"
      not in header.stdout
      or "Machine:                           AArch64" not in header.stdout):
    raise ValueError(f"qualification dependency is not an AArch64 ELF: {path}")
  dynamic = subprocess.run(
      ["readelf", "-d", str(path)], stdout=subprocess.PIPE,
      stderr=subprocess.PIPE, text=True, timeout=30, check=False)
  match = re.search(r"\(SONAME\).*\[([^]]+)]", dynamic.stdout)
  if dynamic.returncode != 0 or not match:
    raise ValueError(f"qualification dependency has no ELF SONAME: {path}")
  return match.group(1)


def copy_library(sysroot: Path, package: Path, name: str) -> dict[str, str]:
  if Path(name).name != name:
    raise ValueError(f"unsafe library name: {name!r}")
  source_directory = (sysroot / LIBRARY_DIRECTORY).resolve(strict=True)
  source = source_directory / name
  resolved = source.resolve(strict=True)
  try:
    resolved.relative_to(source_directory)
  except ValueError as error:
    raise ValueError(f"library resolves outside the SDK sysroot: {source}") from error
  soname = elf_soname(resolved)
  if soname != name:
    raise ValueError(
        f"SDK library SONAME differs from its required name: {name} != {soname}")

  destination_directory = package / LIBRARY_DIRECTORY
  destination_directory.mkdir(parents=True, exist_ok=True)
  destination = destination_directory / resolved.name
  if destination.exists() and sha256(destination) != sha256(resolved):
    raise ValueError(f"two SDK libraries resolve to different bytes named {resolved.name}")
  if not destination.exists():
    shutil.copy2(resolved, destination)
  link = destination_directory / name
  if link != destination:
    if link.exists() or link.is_symlink():
      if not link.is_symlink() or os.readlink(link) != destination.name:
        raise ValueError(f"conflicting SDK library link: {link}")
    else:
      link.symlink_to(destination.name)
  return {
      "file": resolved.name,
      "sha256": sha256(resolved),
      "soname": soname,
  }


def normalize_timestamps(root: Path, epoch: int) -> None:
  for path in sorted(root.rglob("*"), reverse=True):
    if not path.is_symlink():
      os.utime(path, (epoch, epoch), follow_symlinks=False)
  os.utime(root, (epoch, epoch), follow_symlinks=False)


def build(sysroot: Path, sdk_release: Path, build_identity: Path, output: Path,
          source_image: str, core_commit: str, source_date_epoch: int) -> Path:
  sysroot = sysroot.resolve(strict=True)
  sdk_release = sdk_release.resolve(strict=True)
  if not sdk_release.is_file():
    raise ValueError(f"SDK release identity is not a file: {sdk_release}")
  digest = image_digest(source_image)
  if not re.fullmatch(r"[0-9a-f]{40}", core_commit):
    raise ValueError("Core commit must be a full lowercase Git commit")
  validate_identity(build_identity, source_image, core_commit)
  if source_date_epoch <= 0:
    raise ValueError("source date epoch must be positive")
  if output.exists() and (not output.is_dir() or any(output.iterdir())):
    raise ValueError(f"output directory must be absent or empty: {output}")
  output.mkdir(parents=True, exist_ok=True)

  package_name = "sima-pcie-card-qualification-dependencies"
  version = f"0+sdk.{digest[:12]}"
  archive = output / f"{package_name}_{version}_arm64.deb"
  with tempfile.TemporaryDirectory(prefix="card-qualification-deps-",
                                   dir=output.parent) as directory:
    package = Path(directory) / "package"
    control = package / "DEBIAN"
    control.mkdir(parents=True)
    libraries = [
        copy_library(sysroot, package, name) for name in REQUIRED_LIBRARIES]
    provenance_directory = (
        package / "usr/share/sima-pcie-card-qualification-dependencies")
    provenance_directory.mkdir(parents=True)
    provenance = {
        "architecture": "arm64",
        "core_commit": core_commit,
        "libraries": libraries,
        "purpose": "private non-installing Windows PCIe card qualification",
        "schema_version": 1,
        "sdk_build_identity_sha256": sha256(build_identity),
        "sdk_release_sha256": sha256(sdk_release),
        "source_image": source_image,
    }
    write_json(provenance_directory / "provenance.json", provenance)
    (control / "control").write_text(
        f"Package: {package_name}\n"
        f"Version: {version}\n"
        "Section: devel\n"
        "Priority: optional\n"
        "Architecture: arm64\n"
        "Maintainer: SiMa.ai <support@sima.ai>\n"
        "Description: Private SDK libraries for Windows PCIe qualification\n"
        " This package is extracted into a run-private directory and is never installed.\n",
        encoding="utf-8")
    normalize_timestamps(package, source_date_epoch)
    environment = os.environ.copy()
    environment["SOURCE_DATE_EPOCH"] = str(source_date_epoch)
    result = subprocess.run(
        ["dpkg-deb", "--build", "--root-owner-group", str(package), str(archive)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120,
        env=environment, check=False)
    if result.returncode != 0 or not archive.is_file():
      raise RuntimeError(
          f"dpkg-deb failed to build the qualification dependency bundle: "
          f"{result.stderr.strip()}")

  receipt = {
      "architecture": "arm64",
      "core_commit": core_commit,
      "packages": [{"file": archive.name, "sha256": sha256(archive)}],
      "purpose": "private non-installing Windows PCIe card qualification",
      "schema_version": 1,
      "sdk_build_identity_sha256": sha256(build_identity),
      "sdk_release_sha256": sha256(sdk_release),
      "source_image": source_image,
  }
  write_json(output / "card-qualification-dependencies.json", receipt)
  return archive


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--sysroot", type=Path, required=True)
  parser.add_argument("--sdk-release", type=Path, required=True)
  parser.add_argument("--build-identity", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--source-image", required=True)
  parser.add_argument("--core-commit", required=True)
  parser.add_argument("--source-date-epoch", type=int, required=True)
  args = parser.parse_args()
  build(args.sysroot, args.sdk_release, args.build_identity, args.output,
        args.source_image,
        args.core_commit, args.source_date_epoch)


if __name__ == "__main__":
  main()
