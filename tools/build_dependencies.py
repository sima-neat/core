#!/usr/bin/env python3
"""Delegate source-first dependency preparation to the selected Scheduler.

Core selects its immediate producer. Scheduler owns lower build order, provider
admission, archive composition and the complete transitive provenance record.
This wrapper neither fetches artifacts nor substitutes a failed producer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


def scheduler_coordinator(core: Path, environment: dict[str, str]) -> Path:
    source = environment.get("NEAT_SCHEDULER_SOURCE", "")
    package = environment.get("NEAT_SCHEDULER_PACKAGE_DIR", "")
    if source and package:
        raise ValueError("Set only one of NEAT_SCHEDULER_SOURCE and NEAT_SCHEDULER_PACKAGE_DIR")
    sibling = core.parent / "scheduler"
    if not source and not package:
        if (sibling / "CMakeLists.txt").is_file() and (sibling / "cmake/SimaSchedulerConfig.cmake.in").is_file():
            source = str(sibling)
        elif (sibling / "dist/package/usr").is_dir():
            package = str(sibling / "dist/package")
    if source:
        root = Path(source).resolve()
        if not (root / "CMakeLists.txt").is_file() or not (root / "cmake/SimaSchedulerConfig.cmake.in").is_file():
            raise ValueError(f"NEAT_SCHEDULER_SOURCE is not a Scheduler source root: {root}")
        tool = root / "tools/client_dependencies.py"
    elif package:
        root = Path(package).resolve()
        prefix = root / "usr" if (root / "usr").is_dir() else root
        tool = prefix / "share/SimaScheduler/tools/client_dependencies.py"
        if not tool.is_file():
            raise ValueError(f"Selected Scheduler lacks its client dependency coordinator: {tool}; rebuild that producer")
        info = json.loads((prefix / "share/SimaScheduler/buildinfo.json").read_text())
        identity = {key: value for key, value in info.items() if key != "package_sha256"}
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        entry = info.get("installed_files", {}).get(tool.relative_to(prefix).as_posix(), {})
        if (info.get("component") != "scheduler" or digest != info.get("package_sha256") or
                entry.get("kind") != "file" or entry.get("sha256") != hashlib.sha256(tool.read_bytes()).hexdigest() or
                entry.get("mode") != (tool.stat().st_mode & 0o7777)):
            raise ValueError("Selected Scheduler coordinator differs from its package receipt")
    else:
        raise ValueError("No local Scheduler source/package. Set NEAT_SCHEDULER_SOURCE or NEAT_SCHEDULER_PACKAGE_DIR. TODO: remote artifact pulling is not implemented.")
    if not tool.is_file():
        raise ValueError(f"Selected Scheduler lacks its client dependency coordinator: {tool}; rebuild that producer")
    return tool


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--core", type=Path, required=True)
    args, _ = parser.parse_known_args()
    try:
        tool = scheduler_coordinator(args.core.resolve(), dict(os.environ))
        os.execv(sys.executable, [sys.executable, str(tool), *sys.argv[1:]])
    except (ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
