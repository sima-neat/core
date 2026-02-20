#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys


def check_file(path: pathlib.Path) -> list[str]:
    errors: list[str] = []
    data = path.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
    for idx, line in enumerate(data, start=1):
        logical = line.rstrip("\n\r")
        if "\t" in logical:
            errors.append(f"{path}:{idx}: tab character is not allowed")

        if logical.rstrip(" ") != logical:
            errors.append(f"{path}:{idx}: trailing whitespace")

        stripped = logical.lstrip(" ")
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        indent = len(logical) - len(stripped)
        if indent % 2 != 0:
            errors.append(f"{path}:{idx}: indentation must be a multiple of 2 spaces")
    return errors


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: scripts/check_cmake_style.py <cmake-files...>", file=sys.stderr)
        return 2

    errors: list[str] = []
    for arg in argv[1:]:
        path = pathlib.Path(arg)
        if not path.is_file():
            continue
        errors.extend(check_file(path))

    if errors:
        print("[cmake-style] failed:", file=sys.stderr)
        for item in errors:
            print(item, file=sys.stderr)
        return 1

    print("[cmake-style] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
