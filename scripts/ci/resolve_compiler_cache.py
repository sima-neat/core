#!/usr/bin/env python3
"""Resolve the protected or branch-specific Vulcan compiler-cache namespace."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


PROTECTED_BRANCHES = ("develop", "main")


@dataclass(frozen=True)
class Resolution:
    key_prefix: str
    seed_key_prefix: str
    role_arn: str
    rw_mode: str
    base_branch: str


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True, stderr=subprocess.DEVNULL
    ).strip()


def choose_base_branch(distances: dict[str, int]) -> str:
    candidates = [
        (distances[branch], priority, branch)
        for priority, branch in enumerate(PROTECTED_BRANCHES)
        if branch in distances
    ]
    if not candidates:
        raise RuntimeError("cannot determine whether the branch is based on develop or main")
    return min(candidates)[2]


def detect_base_branch(repo: Path) -> str:
    """Choose the protected branch with the closest merge base to HEAD."""
    distances: dict[str, int] = {}
    for branch in PROTECTED_BRANCHES:
        remote_ref = f"origin/{branch}"
        try:
            merge_base = git(repo, "merge-base", "HEAD", remote_ref)
            distance = int(git(repo, "rev-list", "--count", f"{merge_base}..HEAD"))
        except (subprocess.CalledProcessError, ValueError):
            continue
        distances[branch] = distance
    if not distances:
        return ""
    return choose_base_branch(distances)


def resolve_base_branch(requested_base_branch: str, repo: Path) -> str:
    if requested_base_branch == "auto":
        return detect_base_branch(repo)
    if requested_base_branch in PROTECTED_BRANCHES:
        return requested_base_branch
    raise ValueError("cache base branch must be auto, develop, or main")


def resolve(
    *,
    base_key_prefix: str,
    ref_name: str,
    ref_type: str,
    event_name: str,
    reader_role_arn: str,
    protected_writer_role_arn: str,
    branch_writer_role_arn: str,
    requested_base_branch: str,
    repo: Path,
) -> Resolution:
    root = base_key_prefix.rstrip("/")
    protected = ref_type == "branch" and ref_name in PROTECTED_BRANCHES
    direct_branch = ref_type == "branch" and event_name != "pull_request"

    if protected:
        role = protected_writer_role_arn or reader_role_arn
        mode = "READ_WRITE" if protected_writer_role_arn else "READ_ONLY"
        return Resolution(f"{root}/{ref_name}", "", role, mode, ref_name)

    if direct_branch:
        base_branch = resolve_base_branch(requested_base_branch, repo)
        if not base_branch:
            return Resolution("", "", "", "READ_ONLY", "")

        if branch_writer_role_arn:
            encoded_ref = quote(ref_name, safe="")
            protected_prefix = f"{root}/{base_branch}"
            return Resolution(
                f"{protected_prefix}/branches/{encoded_ref}",
                protected_prefix,
                branch_writer_role_arn,
                "READ_WRITE",
                base_branch,
            )
        return Resolution(f"{root}/{base_branch}", "", reader_role_arn, "READ_ONLY", base_branch)

    # Tags and non-direct contexts cannot own a persistent branch cache, but they can
    # still read from the closest compatible protected baseline.
    base_branch = resolve_base_branch(requested_base_branch, repo)
    if not base_branch:
        return Resolution("", "", "", "READ_ONLY", "")
    return Resolution(f"{root}/{base_branch}", "", reader_role_arn, "READ_ONLY", base_branch)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-key-prefix", required=True)
    parser.add_argument("--reader-role-arn", default="")
    parser.add_argument("--protected-writer-role-arn", default="")
    parser.add_argument("--branch-writer-role-arn", default="")
    parser.add_argument("--base-branch", default="auto")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--github-output", type=Path, required=True)
    args = parser.parse_args()

    resolution = resolve(
        base_key_prefix=args.base_key_prefix,
        ref_name=os.environ["GITHUB_REF_NAME"],
        ref_type=os.environ["GITHUB_REF_TYPE"],
        event_name=os.environ["GITHUB_EVENT_NAME"],
        reader_role_arn=args.reader_role_arn,
        protected_writer_role_arn=args.protected_writer_role_arn,
        branch_writer_role_arn=args.branch_writer_role_arn,
        requested_base_branch=args.base_branch,
        repo=args.repo,
    )

    if not resolution.role_arn:
        resolution = Resolution("", "", "", "READ_ONLY", resolution.base_branch)
        print(
            "::warning::Compiler-cache namespace or role is unavailable; "
            "remote caching is disabled.",
            file=sys.stderr,
        )
    elif resolution.rw_mode == "READ_ONLY":
        print(
            "::warning::Writable compiler-cache role is unavailable; using read-only caching.",
            file=sys.stderr,
        )

    print(
        f"Compiler-cache base={resolution.base_branch} mode={resolution.rw_mode} "
        f"prefix={resolution.key_prefix or '<disabled>'}",
        file=sys.stderr,
    )

    with args.github_output.open("a", encoding="utf-8") as output:
        for name, value in (
            ("key_prefix", resolution.key_prefix),
            ("seed_key_prefix", resolution.seed_key_prefix),
            ("role_arn", resolution.role_arn),
            ("rw_mode", resolution.rw_mode),
            ("base_branch", resolution.base_branch),
        ):
            output.write(f"{name}={value}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
