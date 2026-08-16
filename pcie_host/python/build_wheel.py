#!/usr/bin/env python3
"""Build a minimal wheel from the staged pyneatpcie package."""

from __future__ import annotations

import argparse
import base64
import hashlib
import platform
import sys
import zipfile
from pathlib import Path


def normalize_distribution(name: str) -> str:
  return name.replace("-", "_").replace(".", "_")


def wheel_python_tag() -> str:
  if platform.python_implementation() != "CPython":
    return f"py{sys.version_info.major}"
  return f"cp{sys.version_info.major}{sys.version_info.minor}"


def wheel_abi_tag() -> str:
  if platform.python_implementation() != "CPython":
    return "none"
  return f"cp{sys.version_info.major}{sys.version_info.minor}"


def wheel_platform_tag() -> str:
  machine = platform.machine().lower()
  if machine in {"x86_64", "amd64"}:
    machine = "x86_64"
  elif machine in {"aarch64", "arm64"}:
    machine = "aarch64"
  else:
    machine = machine.replace("-", "_")
  return f"linux_{machine}"


def record_hash(data: bytes) -> str:
  digest = hashlib.sha256(data).digest()
  encoded = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
  return f"sha256={encoded}"


def iter_payload_files(package_root: Path) -> list[Path]:
  return sorted(path for path in package_root.rglob("*") if path.is_file())


def write_wheel(stage_dir: Path, output_dir: Path, version: str) -> Path:
  package_root = stage_dir / "pyneatpcie"
  if not package_root.is_dir():
    raise SystemExit(f"missing staged package: {package_root}")

  dist_name = normalize_distribution("pyneatpcie")
  dist_info = f"{dist_name}-{version}.dist-info"
  tag = f"{wheel_python_tag()}-{wheel_abi_tag()}-{wheel_platform_tag()}"
  wheel_name = f"{dist_name}-{version}-{tag}.whl"
  wheel_path = output_dir / wheel_name
  output_dir.mkdir(parents=True, exist_ok=True)

  entries: list[tuple[str, bytes]] = []
  for path in iter_payload_files(package_root):
    arcname = path.relative_to(stage_dir).as_posix()
    entries.append((arcname, path.read_bytes()))

  metadata = (
      "Metadata-Version: 2.1\n"
      "Name: pyneatpcie\n"
      f"Version: {version}\n"
      "Summary: Python bindings for the SiMa NEAT PCIe host co-processor API\n"
      "Requires-Python: >=3.8\n"
      "Requires-Dist: numpy\n"
  ).encode("utf-8")
  wheel = (
      "Wheel-Version: 1.0\n"
      "Generator: pyneatpcie build_wheel.py\n"
      "Root-Is-Purelib: false\n"
      f"Tag: {tag}\n"
  ).encode("utf-8")

  entries.append((f"{dist_info}/METADATA", metadata))
  entries.append((f"{dist_info}/WHEEL", wheel))

  records: list[list[str]] = []
  with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for arcname, data in entries:
      zf.writestr(arcname, data)
      records.append([arcname, record_hash(data), str(len(data))])

    record_name = f"{dist_info}/RECORD"
    records.append([record_name, "", ""])
    record_lines: list[str] = []
    for row in records:
      # csv.writer wants a real file-like object; keep this dependency-free.
      escaped = []
      for field in row:
        if any(ch in field for ch in "\",\n\r"):
          escaped.append('"' + field.replace('"', '""') + '"')
        else:
          escaped.append(field)
      record_lines.append(",".join(escaped))
    zf.writestr(record_name, "\n".join(record_lines).encode("utf-8") + b"\n")

  return wheel_path


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--stage-dir", type=Path, required=True)
  parser.add_argument("--output-dir", type=Path, required=True)
  parser.add_argument("--version", required=True)
  args = parser.parse_args()

  wheel = write_wheel(args.stage_dir.resolve(), args.output_dir.resolve(), args.version)
  print(wheel)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
