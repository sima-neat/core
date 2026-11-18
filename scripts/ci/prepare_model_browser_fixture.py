#!/usr/bin/env python3
"""Download one Model Browser package with sima-cli and extract its MPK for the device tests.

Prints the extracted MPK path on stdout. The registry run_id and ZIP sha256 go to stderr so a
failing test can be traced to the exact package it used.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path


class FixtureError(RuntimeError):
    pass


def mpk_member(model_id: str, variant: str) -> str:
    return f"{model_id}_{variant}_mpk.tar.gz"


def download(sima_cli: str, model_id: str, variant: str, output: Path) -> dict:
    command = [
        sima_cli, "models", "download", "--branch", "main", "--id", model_id,
        "--variant", variant, "--output", str(output), "--json",
    ]
    result = subprocess.run(command, capture_output=True, text=True, errors="replace", check=False)
    if result.returncode != 0:
        tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-5:])
        raise FixtureError(f"sima-cli models download exited {result.returncode}: {tail}")
    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        head = "\n".join(result.stdout.strip().splitlines()[:5])
        raise FixtureError(
            f"sima-cli models download --json did not print JSON ({exc}): {head}"
        ) from exc
    if not isinstance(info, dict) or not info.get("path"):
        raise FixtureError(f"sima-cli models download --json gave no ZIP path: {result.stdout.strip()}")
    return info


def extract_mpk(zip_path: Path, member: str, destination: Path) -> Path:
    try:
        archive = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise FixtureError(f"{zip_path} is not a readable ZIP: {exc}") from exc
    with archive:
        names = archive.namelist()
        if member not in names:
            mpks = sorted(name for name in names if name.endswith("_mpk.tar.gz"))
            raise FixtureError(
                f"{zip_path.name} has no {member}; MPKs in it: {', '.join(mpks) or 'none'}"
            )
        target = destination / member
        with archive.open(member) as source, target.open("wb") as sink:
            shutil.copyfileobj(source, sink)
    if target.stat().st_size == 0:
        raise FixtureError(f"{member} in {zip_path.name} is empty")
    return target


def prepare(sima_cli: str, model_id: str, variant: str, output: Path) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    info = download(sima_cli, model_id, variant, output)
    print(
        f"Model Browser {model_id} {variant} on main: run_id={info.get('run_id')} "
        f"sha256={info.get('sha256')} zip={info['path']}",
        file=sys.stderr,
    )
    return extract_mpk(Path(info["path"]), mpk_member(model_id, variant), output)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--id", required=True, dest="model_id")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sima-cli", default="sima-cli")
    args = parser.parse_args(argv)
    try:
        mpk = prepare(args.sima_cli, args.model_id, args.variant, args.output)
    except FixtureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(mpk)
    return 0


if __name__ == "__main__":
    sys.exit(main())
