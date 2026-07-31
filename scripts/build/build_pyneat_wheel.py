#!/usr/bin/env python3
"""Assemble a pyneat wheel from an extension built by the main CMake tree."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import sys
import tomllib
import zipfile
from pathlib import Path


def wheel_name_component(value: str) -> str:
    return value.replace("-", "_")


def record_digest(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    return "sha256=" + base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def metadata(project: dict[str, object], license_path: str) -> bytes:
    lines = [
        "Metadata-Version: 2.3",
        f"Name: {project['name']}",
        f"Version: {project['version']}",
        f"Summary: {project['description']}",
        f"Requires-Python: {project['requires-python']}",
        f"License-File: {license_path}",
    ]
    for author in project.get("authors", []):
        if isinstance(author, dict) and author.get("name"):
            lines.append(f"Author: {author['name']}")
    for classifier in project.get("classifiers", []):
        lines.append(f"Classifier: {classifier}")
    for dependency in project.get("dependencies", []):
        lines.append(f"Requires-Dist: {dependency}")
    optional = project.get("optional-dependencies", {})
    if isinstance(optional, dict):
        for extra, dependencies in optional.items():
            lines.append(f"Provides-Extra: {extra}")
            for dependency in dependencies:
                lines.append(f'Requires-Dist: {dependency}; extra == "{extra}"')
    return ("\n".join(lines) + "\n").encode()


def build_wheel(
    project_root: Path,
    extension: Path,
    output_dir: Path,
    version: str,
    python_tag: str,
    abi_tag: str,
    platform_tag: str,
) -> Path:
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text(encoding="utf-8"))
    project = dict(pyproject["project"])
    project["version"] = version
    distribution = wheel_name_component(str(project["name"]))
    wheel_version = wheel_name_component(version)
    dist_info = f"{distribution}-{wheel_version}.dist-info"
    license_archive_path = f"{dist_info}/licenses/LICENSE"
    tag = f"{python_tag}-{abi_tag}-{platform_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    wheel_path = output_dir / f"{distribution}-{wheel_version}-{tag}.whl"

    entries: dict[str, bytes] = {}
    package_root = project_root / "python" / "pyneat"
    for source in sorted(package_root.rglob("*")):
        if source.is_file() and (source.suffix == ".py" or source.name == "py.typed"):
            entries[f"pyneat/{source.relative_to(package_root).as_posix()}"] = source.read_bytes()
    entries[f"pyneat/{extension.name}"] = extension.read_bytes()
    entries[f"{dist_info}/METADATA"] = metadata(project, license_archive_path)
    entries[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: sima-neat build_pyneat_wheel.py\n"
        "Root-Is-Purelib: false\n"
        f"Tag: {tag}\n"
    ).encode()
    entries[f"{dist_info}/top_level.txt"] = b"pyneat\n"
    entries[license_archive_path] = (project_root / "LICENSE").read_bytes()

    record_path = f"{dist_info}/RECORD"
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for archive_path, data in sorted(entries.items()):
        writer.writerow((archive_path, record_digest(data), len(data)))
    writer.writerow((record_path, "", ""))
    entries[record_path] = record_buffer.getvalue().encode()

    timestamp = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for archive_path, data in sorted(entries.items()):
            info = zipfile.ZipInfo(archive_path, timestamp)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, data)
    return wheel_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--python-tag", required=True)
    parser.add_argument("--abi-tag", required=True)
    parser.add_argument("--platform-tag", required=True)
    args = parser.parse_args()

    if not args.extension.is_file():
        parser.error(f"extension does not exist: {args.extension}")

    wheel_path = build_wheel(
        args.project_root,
        args.extension,
        args.output_dir,
        args.version,
        args.python_tag,
        args.abi_tag,
        args.platform_tag,
    )
    print(wheel_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
