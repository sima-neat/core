#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pathlib
import re
import subprocess
import sys
from typing import Iterable

INCLUDE_RE = re.compile(r'^\s*#\s*include\s*[<\"]([^>\"]+)[>\"]')
CPP_EXTS = (".c", ".cc", ".cpp", ".cxx", ".h", ".hpp")


def run_git(args: list[str]) -> list[str]:
    out = subprocess.check_output(["git", *args], text=True)
    return [line.strip() for line in out.splitlines() if line.strip()]


def resolve_base_ref(explicit: str) -> str:
    if explicit:
        return explicit

    github_base = os_env("GITHUB_BASE_REF")
    if github_base:
        remote_ref = f"origin/{github_base}"
        if not git_ref_exists(remote_ref):
            try:
                subprocess.run(
                    ["git", "fetch", "--no-tags", "--depth=1", "origin", f"{github_base}:{remote_ref}"],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
        if git_ref_exists(remote_ref):
            return remote_ref

    # Local hooks should check staged changes by default. Only use commit-range
    # fallback automatically in CI, where no staged index is present.
    if ci_mode() and git_ref_exists("HEAD~1"):
        return "HEAD~1"
    return ""


def os_env(name: str) -> str:
    import os

    return os.environ.get(name, "")


def ci_mode() -> bool:
    ci = os_env("CI").strip().lower()
    if ci and ci not in {"false", "0"}:
        return True
    return os_env("GITHUB_ACTIONS").strip().lower() == "true"


def git_ref_exists(ref: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", ref],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def changed_files(base_ref: str) -> list[str]:
    if base_ref:
        return run_git(["diff", "--name-only", "--diff-filter=ACMRTUXB", f"{base_ref}...HEAD"])
    return run_git(["diff", "--name-only", "--diff-filter=ACMRTUXB", "--cached"])


def candidate_files(mode: str, base_ref: str) -> list[pathlib.Path]:
    names = run_git(["ls-files"]) if mode == "all" else changed_files(base_ref)
    out: list[pathlib.Path] = []
    for name in names:
        p = pathlib.Path(name)
        if not p.is_file():
            continue
        if p.suffix not in CPP_EXTS:
            continue
        out.append(p)
    return out


def check_duplicates(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    seen: dict[str, int] = {}

    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        errors.append(f"{path}:0: unable to read file: {exc}")
        return errors

    for idx, line in enumerate(lines, start=1):
        match = INCLUDE_RE.match(line)
        if not match:
            continue
        header = match.group(1)
        first = seen.get(header)
        if first is not None:
            errors.append(
                f"{path}:{idx}: duplicate include '{header}' (first at line {first})"
            )
        else:
            seen[header] = idx

    return errors


def main(argv: Iterable[str]) -> int:
    parser = argparse.ArgumentParser(description="Detect duplicate includes in C/C++ files")
    parser.add_argument("--all", action="store_true", help="scan all tracked C/C++ files")
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="scan only changed tracked C/C++ files (default)",
    )
    parser.add_argument("--base-ref", default="", help="explicit base ref for changed-only mode")
    args = parser.parse_args(list(argv)[1:])

    mode = "all" if args.all else "changed"
    base = resolve_base_ref(args.base_ref) if mode == "changed" else ""

    files = candidate_files(mode, base)
    if not files:
        print(f"[include-hygiene] no files to check ({mode} mode)")
        return 0

    errors: list[str] = []
    for path in files:
        errors.extend(check_duplicates(path))

    if errors:
        print("[include-hygiene] failed:", file=sys.stderr)
        for item in errors:
            print(item, file=sys.stderr)
        return 1

    print(f"[include-hygiene] OK ({len(files)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
