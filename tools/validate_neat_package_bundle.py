#!/usr/bin/env python3
"""Validate versioned dependencies inside a local Neat package bundle."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


def deb_field(deb: Path, field: str) -> str:
    return subprocess.run(
        ["dpkg-deb", "-f", str(deb), field],
        check=False,
        text=True,
        capture_output=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    args = parser.parse_args()

    versions: dict[str, str] = {}
    dependencies: dict[str, str] = {}
    for deb in sorted(args.bundle_dir.glob("*.deb")):
        package = deb_field(deb, "Package")
        version = deb_field(deb, "Version")
        if not package or not version:
            raise SystemExit(f"Invalid Debian package metadata: {deb}")
        versions[package] = version
        dependencies[package] = deb_field(deb, "Depends")

    checked = 0
    for consumer, dependency in (
        ("sima-neat", "sima-lmm-core"),
        ("sima-neat-dev", "sima-lmm-dev"),
    ):
        if consumer not in dependencies:
            continue
        if dependency not in versions:
            raise SystemExit(
                f"Incomplete package bundle: {consumer} is staged but its local dependency "
                f"{dependency} is missing"
            )
        local_version = versions[dependency]
        relations = re.findall(
            rf"(?:^|,\s*){re.escape(dependency)}\s*"
            r"\((<<|<=|=|>=|>>)\s*([^)]+)\)",
            dependencies[consumer],
        )
        for operator, required in relations:
            checked += 1
            result = subprocess.run(
                [
                    "dpkg",
                    "--compare-versions",
                    local_version,
                    operator,
                    required.strip(),
                ],
                check=False,
            )
            if result.returncode != 0:
                raise SystemExit(
                    f"Incompatible package bundle: {consumer} requires "
                    f"{dependency} ({operator} {required.strip()}), but staged "
                    f"{dependency} is {local_version}"
                )

    if checked == 0:
        raise SystemExit("No Core-to-LLiMa version constraints found in package bundle")
    print(f"Validated {checked} Core-to-LLiMa package constraints")


if __name__ == "__main__":
    main()
