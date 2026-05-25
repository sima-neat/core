#!/usr/bin/env python3
"""Generate installed package provenance metadata for Neat builds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--package-version", required=True)
    parser.add_argument("--vulcan-env", default="dev")
    parser.add_argument("--internals-ref", default="")
    parser.add_argument("--llima-ref", default="")
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
