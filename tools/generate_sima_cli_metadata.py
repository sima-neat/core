#!/usr/bin/env python3
"""Generate sima-cli metadata.json for a sima-neat artifact set."""
#
# File purpose:
# - Build a sima-cli compatible metadata.json from a prepared artifact directory.
# - Detect core/extras/wheel/internals artifacts and compute user-facing size estimates.
# - Also generate a "metadata-all.json" variant that includes extras by default without prompts.
#
# Expected inputs (via --artifacts-dir):
# - sima-neat-*-Linux-core.deb
# - *extras.tar.gz
# - *.whl
# - optional additional .deb files (treated as neat-internals dependencies)
#
# Output:
# - metadata.json written to --output
# - metadata-all.json written next to --output

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import tarfile
import urllib.parse
import zipfile
from tempfile import TemporaryDirectory
from pathlib import Path
from typing import List

INSTALLER_SCRIPT_NAME = "install_neat_framework.sh"


def _pick_one(files: List[Path], label: str) -> Path:
    """Pick a deterministic artifact candidate (lexicographically first)."""
    if not files:
        raise SystemExit(f"missing required artifact for {label}")
    return sorted(files)[0]


def _version_from_core_deb(core_deb_name: str) -> str:
    """Extract package version from sima-neat core deb filename."""
    match = re.match(r"^sima-neat-(.+)-Linux-core\.deb$", core_deb_name)
    return match.group(1) if match else "unknown"


def _url_safe_name(name: str) -> str:
    """Encode artifact filenames for metadata URLs."""
    return urllib.parse.quote(name, safe="-._~")


def _fmt_size(num_bytes: int) -> str:
    """Format bytes into a small human-readable size string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _directory_size(path: Path) -> int:
    """Sum all file sizes in a directory tree."""
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def _installed_size_from_deb(path: Path) -> int:
    """Estimate installed size of a deb.

    Prefer control metadata (Installed-Size) and fall back to extracting
    payload if metadata is unavailable in the current environment.
    """
    if shutil.which("dpkg-deb"):
        proc = subprocess.run(
            ["dpkg-deb", "-f", str(path), "Installed-Size"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            raw = proc.stdout.strip()
            if raw.isdigit():
                return int(raw) * 1024

    if shutil.which("dpkg-deb"):
        with TemporaryDirectory(prefix="sima-neat-deb-size-") as tmp:
            tmp_path = Path(tmp)
            proc = subprocess.run(["dpkg-deb", "-x", str(path), str(tmp_path)], check=False)
            if proc.returncode == 0:
                return _directory_size(tmp_path)
    return path.stat().st_size


def _extracted_size(path: Path) -> int:
    """Estimate post-install/extracted size for supported artifact types."""
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            return sum(member.size for member in tar.getmembers() if member.isfile())
    if path.suffix.lower() == ".whl":
        with zipfile.ZipFile(path, "r") as whl:
            return sum(info.file_size for info in whl.infolist() if not info.is_dir())
    if path.suffix.lower() == ".deb":
        return _installed_size_from_deb(path)
    return path.stat().st_size


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--release", default="stable")
    parser.add_argument("--board", default="modalix")
    parser.add_argument("--board-version", default="2.0.0")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    if not artifacts_dir.is_dir():
        raise SystemExit(f"artifacts directory not found: {artifacts_dir}")

    # Required package set for a complete installable payload.
    core_deb = _pick_one(list(artifacts_dir.glob("*-Linux-core.deb")), "core deb")
    extras_tar = _pick_one(list(artifacts_dir.glob("*extras.tar.gz")), "extras tar.gz")
    wheel = _pick_one(list(artifacts_dir.glob("*.whl")), "wheel")
    # Any additional debs are treated as neat-internals runtime dependencies.
    internals_debs = sorted(
        p for p in artifacts_dir.glob("*.deb") if p.name != core_deb.name
    )

    installer_script_path = artifacts_dir / INSTALLER_SCRIPT_NAME
    if not installer_script_path.is_file():
        raise SystemExit(f"missing required installer script: {installer_script_path}")

    version = _version_from_core_deb(core_deb.name)
    resources = [
        _url_safe_name(core_deb.name),
        _url_safe_name(wheel.name),
        _url_safe_name(installer_script_path.name),
    ]
    resources.extend(_url_safe_name(p.name) for p in internals_debs)
    extras_resource = _url_safe_name(extras_tar.name)
    selectable_resources = [
        {
            "name": "SiMa NEAT extras (samples/tutorials/tests)",
            "url": extras_resource,
            "resource": extras_resource,
        }
    ]

    # Size fields are user-facing estimates for download/install budgeting.
    all_payload_files = [core_deb, extras_tar, wheel, installer_script_path] + internals_debs
    download_size_bytes = sum(p.stat().st_size for p in all_payload_files)
    install_size_bytes = sum(_extracted_size(p) for p in all_payload_files)

    install_script = f"bash ./{installer_script_path.name}"

    # sima-cli metadata schema payload.
    payload = {
        "name": "sima-neat",
        "version": version,
        "release": args.release,
        "description": "SiMa.ai Neural Edge Acceleration Toolkit",
        "platforms": [
            {
                "type": "board",
                "compatible_with": [args.board],
                "version": args.board_version,
            },
            {
                "type": "palette"
            }
        ],
        "resources": resources,
        "selectable-resources": selectable_resources,
        "size": {
            "download": _fmt_size(download_size_bytes),
            "install": _fmt_size(install_size_bytes),
        },
        "installation": {
            "script": install_script,
            "post-message": "Successfully installed SiMa NEAT package.",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    # "all" variant: install everything without selectable prompts.
    payload_all = copy.deepcopy(payload)
    payload_all["resources"].append(extras_resource)
    payload_all.pop("selectable-resources", None)
    metadata_all_path = output_path.with_name("metadata-all.json")
    metadata_all_path.write_text(
        json.dumps(payload_all, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
