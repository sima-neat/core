#!/usr/bin/env python3
"""Reject an incompatible fetched Internals artifact before SDK/package changes."""
import json
import sys
from pathlib import Path


def validate(consumer: dict, artifact: dict) -> None:
    if str(consumer.get("platform-version", "")).startswith("2.1."):
        return
    for expected, actual in (
        ("platform-version", "platform-version"),
        ("runtime-profile", "runtime-profile"),
        ("kernel-commit", "kernel-commit"),
        ("expected-internals-sysroot", "sysroot-version"),
    ):
        value = consumer.get(expected)
        if not isinstance(value, str) or not value or artifact.get(actual) != value:
            raise ValueError(f"Internals {actual} does not match the required {expected}")


def main() -> int:
    try:
        consumer = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
        if str(consumer.get("platform-version", "")).startswith("2.1."):
            return 0
        artifact = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        validate(consumer, artifact)
    except (OSError, ValueError, TypeError, AttributeError) as error:
        print(f"ERROR: Incompatible Internals artifact: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
