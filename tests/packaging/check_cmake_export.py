#!/usr/bin/env python3
"""Check an extracted Core dev package's installed CMake export, not its source."""
import argparse
from pathlib import Path
import re


def check(prefix: Path, forbidden: list[str]) -> int:
    directory = prefix / "usr/lib/cmake/SimaNeat"
    files = sorted(directory.glob("*.cmake"))
    if not files or not (directory / "SimaNeatTargets.cmake").is_file():
        raise ValueError("missing installed SimaNeat CMake export")
    locations = 0
    for path in files:
        content = path.read_text()
        for marker in forbidden:
            if marker and marker in content:
                raise ValueError(f"{path.name}: build-host path leaked: {marker}")
        for value in re.findall(r'IMPORTED_LOCATION(?:_[A-Z]+)?\s+"([^"]+)"', content):
            locations += 1
            if not value.startswith("${_IMPORT_PREFIX}/"):
                raise ValueError(f"{path.name}: non-relocatable imported location: {value}")
    if locations < 2:
        raise ValueError("expected installed shared and static library locations")
    targets = (directory / "SimaNeatTargets.cmake").read_text()
    config = (directory / "SimaNeatConfig.cmake").read_text()
    if 'set(SIMANEAT_WITH_LLIMA TRUE)' in config:
        if "PkgConfig::SIMANEAT_HTTPLIB" not in targets:
            raise ValueError("static dependency must use consumer-resolved httplib target")
        if "pkg_check_modules(SIMANEAT_HTTPLIB REQUIRED IMPORTED_TARGET cpp-httplib)" not in config:
            raise ValueError("httplib target is not rediscovered on the consuming machine")
    return locations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prefix", type=Path)
    parser.add_argument("--forbid-prefix", action="append", default=[])
    args = parser.parse_args()
    try:
        count = check(args.prefix, ["/opt/toolchain/", "/repair/", *args.forbid_prefix])
    except (OSError, ValueError) as error:
        parser.exit(1, f"FAIL: {error}\n")
    print(f"PASS: {count} relocatable installed library locations; no build-host paths")
