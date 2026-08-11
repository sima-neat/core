#!/usr/bin/env python3
"""Validate versioned dependencies inside a local Neat package bundle."""

from __future__ import annotations

import argparse
import json
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


def has_exact_relation(field: str, package: str, version: str) -> bool:
    pattern = re.compile(
        rf"(?:^|,)\s*{re.escape(package)}(?:\:[^\s(,|]+)?\s*"
        rf"\(\s*=\s*{re.escape(version)}\s*\)(?=\s*(?:,|$))"
    )
    return pattern.search(field) is not None


def validate_platform_overrides(
    manifest_path: Path,
    versions: dict[str, str],
    architectures: dict[str, str],
    dependencies: dict[str, str],
    provides: dict[str, str],
) -> int:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    contract = data.get("platform-package-contract")
    if not isinstance(contract, dict):
        raise SystemExit(f"Missing platform-package-contract in {manifest_path}")

    camera = contract.get("libcamera", {})
    memory = contract.get("memory", {})
    checks = 0
    groups = (
        (
            ("libcamera", "libcamera-dev", "libcamera-tools"),
            str(camera.get("package-version", "")),
            str(camera.get("platform-compat-version", "")),
        ),
        (
            ("simaai-memory-lib", "simaai-memory-lib-dev"),
            str(memory.get("package-version", "")),
            str(memory.get("platform-compat-version", "")),
        ),
    )
    for packages, actual, compatible in groups:
        if not actual or not compatible:
            raise SystemExit(
                f"Incomplete platform override contract in {manifest_path}"
            )
        for package in packages:
            if package not in versions:
                raise SystemExit(
                    f"Incomplete package bundle: required B4593 override {package} is missing"
                )
            if versions[package] != actual:
                raise SystemExit(
                    f"Wrong B4593 override version: {package}={versions[package]}, "
                    f"expected {actual}"
                )
            if architectures[package] != "arm64":
                raise SystemExit(
                    f"Wrong B4593 override architecture: {package}="
                    f"{architectures[package]}, expected arm64"
                )
            if not has_exact_relation(provides[package], package, compatible):
                raise SystemExit(
                    f"{package} must provide {package} (= {compatible}); got: "
                    f"{provides[package] or '<none>'}"
                )
            checks += 3

    for package in ("libcamera-dev", "libcamera-tools"):
        if not has_exact_relation(
            dependencies[package], "libcamera", versions["libcamera"]
        ):
            raise SystemExit(
                f"{package} must depend on libcamera (= {versions['libcamera']}); "
                f"got: {dependencies[package] or '<none>'}"
            )
        checks += 1
    if not has_exact_relation(
        dependencies["simaai-memory-lib-dev"],
        "simaai-memory-lib",
        versions["simaai-memory-lib"],
    ):
        raise SystemExit(
            "simaai-memory-lib-dev must depend on simaai-memory-lib "
            f"(= {versions['simaai-memory-lib']}); got: "
            f"{dependencies['simaai-memory-lib-dev'] or '<none>'}"
        )
    checks += 1

    for component, package in ((camera, "libcamera"), (memory, "simaai-memory-lib")):
        capability_name = str(component.get("capability-name", ""))
        capability_version = str(component.get("capability-version", ""))
        if not capability_name or not capability_version:
            raise SystemExit(f"Incomplete capability contract for {package}")
        if not has_exact_relation(
            provides[package], capability_name, capability_version
        ):
            raise SystemExit(
                f"{package} must provide {capability_name} (= {capability_version}); "
                f"got: {provides[package] or '<none>'}"
            )
        checks += 1

    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle_dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    versions: dict[str, str] = {}
    architectures: dict[str, str] = {}
    dependencies: dict[str, str] = {}
    provides: dict[str, str] = {}
    for deb in sorted(args.bundle_dir.glob("*.deb")):
        package = deb_field(deb, "Package")
        version = deb_field(deb, "Version")
        if not package or not version:
            raise SystemExit(f"Invalid Debian package metadata: {deb}")
        if package in versions:
            raise SystemExit(f"Duplicate Debian package in bundle: {package}")
        versions[package] = version
        architectures[package] = deb_field(deb, "Architecture")
        dependencies[package] = deb_field(deb, "Depends")
        provides[package] = deb_field(deb, "Provides")

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
            rf"(?:^|,\s*){re.escape(dependency)}\s*" r"\((<<|<=|=|>=|>>)\s*([^)]+)\)",
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
    platform_checks = 0
    if args.manifest is not None:
        platform_checks = validate_platform_overrides(
            args.manifest,
            versions,
            architectures,
            dependencies,
            provides,
        )
    print(
        f"Validated {checked} Core-to-LLiMa package constraints and "
        f"{platform_checks} platform override checks"
    )


if __name__ == "__main__":
    main()
