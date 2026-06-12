#!/usr/bin/env python3
"""Generate installed package provenance metadata for Neat builds."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


# Policy / packaging-derived constants. These mirror the Debian `Depends:` floors
# and supported-host list; they change rarely and live here (one place, in code)
# so the docs never hardcode them — the Compatibility page renders from the
# emitted `compatibility` block. Keep these in sync with the packaging Depends
# (sima-neat / neat-runtime control files) and the elxr-sdk-host-setup docs.
COMPAT_DEFAULTS = {
    "target": {
        "board": "any Modalix DevKit",
        "arch": "aarch64",
        "glibc_min": "2.34",        # neat-runtime Depends libc6 (>= 2.34)
        "opencv": "4.6",            # sima-neat Depends libopencv-core406
        "gstreamer": "1.0",         # sima-neat Depends libgstreamer1.0-0
        "cxx_std": "20",
    },
    "host_sdk": {
        "os": [
            "Ubuntu 22.04 / 24.04",
            "Windows 11 (WSL2 + Docker Engine inside WSL)",
            "macOS (Colima + Docker)",
        ],
        "sysroot": "/opt/toolchain/aarch64/modalix",
        # Fallbacks so the Compatibility table never renders blank rows when
        # building outside the SDK container (no /etc/sdk-release, no --toolchain).
        # Overridden by read_sdk_release() and --toolchain when those are available.
        "sdk": "2.0.0",
        "elxr": "2.0.0",
        "toolchain": "aarch64-linux-gnu-g++ 12.2.0",
    },
    "pyneat": {
        "platform": "linux_aarch64",
        "python_abi": "CPython 3.11 (cp311)",
    },
    "tools": {"sima_cli": "latest"},
}


def git_value(repo_root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def split_ref(value: str) -> dict[str, str]:
    if not value:
        return {}
    if ":" in value:
        ref, spec = value.split(":", 1)
        return {"ref": ref, "spec": spec}
    return {"ref": value}


def parse_pyproject_floors(repo_root: Path) -> dict[str, str]:
    """Extract pyneat dependency floors from pyproject.toml without a TOML lib
    (the build's python3 may predate tomllib). Regex over the few fields we need."""
    out: dict[str, str] = {}
    pyproject = repo_root / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return out
    # numpy>=1.24,<2  /  torch>=2.3.0  /  requires-python = ">=3.9"
    m = re.search(r'["\']numpy\s*([^"\']+)["\']', text)
    if m:
        out["numpy"] = m.group(1).strip()
    m = re.search(r'["\']torch\s*([^"\']+)["\']', text)
    if m:
        out["torch"] = m.group(1).strip() + " (optional)"
    m = re.search(r'requires-python\s*=\s*["\']\s*>=\s*([0-9.]+)', text)
    if m:
        out["source_python_min"] = m.group(1).strip()
    return out


def read_sdk_release(path: Path) -> dict[str, str]:
    """Parse /etc/sdk-release for `SDK Version` / `eLXr Version`. Empty if absent
    (e.g. building outside the SDK container)."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return out
    def clean_version(raw: str) -> str:
        # "2.0.0_Palette_SDK_neat_main_780365a" -> "2.0.0" for display; fall back to raw.
        m = re.match(r"\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", raw)
        return m.group(1) if m else raw.strip()

    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip()
        if key == "SDK Version":
            out["sdk"] = clean_version(val)
        elif key == "eLXr Version":
            out["elxr"] = clean_version(val)
    return out


def build_compatibility(
    package_version: str,
    platform_version: str,
    repo_root: Path,
    sdk_release_file: Path,
    toolchain: str,
) -> dict:
    """Assemble the compatibility block the Compatibility doc renders from.
    Constants come from COMPAT_DEFAULTS; the rest is derived from real sources."""
    import copy
    compat = copy.deepcopy(COMPAT_DEFAULTS)
    compat["target"]["platform_sw"] = platform_version.split("+", 1)[0]
    # pyneat floors from pyproject.toml
    compat["pyneat"].update(parse_pyproject_floors(repo_root))
    # SDK / eLxr from the container's sdk-release file
    sdk = read_sdk_release(sdk_release_file)
    compat["host_sdk"].update(sdk)
    if toolchain:
        compat["host_sdk"]["toolchain"] = toolchain
    return compat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--platform-version", required=True)
    parser.add_argument("--vulcan-env", default="dev")
    parser.add_argument("--internals-ref", default="")
    parser.add_argument("--llima-ref", default="")
    parser.add_argument("--sdk-release-file", type=Path, default=Path("/etc/sdk-release"),
                        help="Path to the SDK release file (default /etc/sdk-release).")
    parser.add_argument("--toolchain", default="",
                        help="Cross toolchain version string (e.g. 'aarch64-linux-gnu-g++ 12.2.0').")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_path = args.output

    branch = (
        os.environ.get("GITHUB_HEAD_REF")
        or os.environ.get("GITHUB_REF_NAME")
        or git_value(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    )
    if branch == "HEAD":
        branch = os.environ.get("GITHUB_REF_NAME", "")

    full_sha = os.environ.get("GITHUB_SHA") or git_value(repo_root, "rev-parse", "HEAD")
    short_sha = full_sha[:12] if full_sha else ""
    ref_type = os.environ.get("GITHUB_REF_TYPE") or (
        "tag" if git_value(repo_root, "describe", "--tags", "--exact-match") else "branch"
    )
    repository = os.environ.get("GITHUB_REPOSITORY", "sima-neat/core")
    repo_key = repository.split("/", 1)[-1]
    branch_key = quote(branch, safe="") if branch else ""

    vulcan = {
        "environment": args.vulcan_env.strip() or "dev",
        "repository": repo_key,
        "ref": branch,
        "ref_key": branch_key,
        "ref_type": ref_type,
        "spec": short_sha,
    }

    payload = {
        "_note": "Generated by scripts/build/generate_package_buildinfo.py; do not hand-edit.",
        "format_version": 1,
        "package": args.package_name,
        "version": args.package_version,
        "source": "vulcan",
        "vulcan": vulcan,
        "build": {
            "repository": repository,
            "git_ref": branch,
            "git_ref_type": ref_type,
            "git_sha": full_sha,
            "git_short_sha": short_sha,
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "github_run_id": os.environ.get("GITHUB_RUN_ID", ""),
            "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT", ""),
        },
        "dependencies": {},
        "compatibility": build_compatibility(
            args.package_version,
            args.platform_version,
            repo_root,
            args.sdk_release_file,
            args.toolchain,
        ),
    }
    if args.internals_ref:
        payload["dependencies"]["internals"] = split_ref(args.internals_ref)
    if args.llima_ref:
        payload["dependencies"]["llima"] = split_ref(args.llima_ref)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
