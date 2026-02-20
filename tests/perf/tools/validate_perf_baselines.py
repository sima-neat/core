#!/usr/bin/env python3
"""Validate perf baseline profile/scenario files with strict schema checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import perf_schema as schema


def parse_args() -> argparse.Namespace:
    repo_root = THIS_DIR.parents[3]
    parser = argparse.ArgumentParser(description="Validate perf baseline schema")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=repo_root / "tests" / "perf" / "baselines" / "v2" / "modalix_default",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile_dir = args.profile_dir.resolve()

    try:
        profile, baselines = schema.validate_baseline_directory(profile_dir)
    except schema.SchemaError as exc:
        print(f"[validate_perf_baselines] FAIL: {exc}")
        return 1

    missing = sorted(set(schema.REQUIRED_SCENARIO_IDS) - set(baselines.keys()))
    extra = sorted(set(baselines.keys()) - set(schema.REQUIRED_SCENARIO_IDS))
    if missing:
        print(
            "[validate_perf_baselines] FAIL: missing required scenarios: "
            + ", ".join(missing)
        )
        return 1
    if extra:
        print(
            "[validate_perf_baselines] FAIL: unknown scenario baselines present: "
            + ", ".join(extra)
        )
        return 1

    print(
        f"[validate_perf_baselines] PASS: profile={profile.modalix_profile_id} "
        f"scenarios={len(baselines)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
