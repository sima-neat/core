#!/usr/bin/env python3
"""Prepare accepted codec inputs, without regenerating a benchmark's workload.

The lock pins original encoded streams AND their decoded references. Normal
build/package preparation verifies these bytes or obtains the immutable extras
archive through sima-cli. The existing generator remains an explicit developer
tool for creating a new workload; its independent accuracy recipes are unchanged.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from urllib.parse import unquote, urlsplit

DEFAULT_LOCK = Path(__file__).with_name("codec_perf_fixtures.lock.json")


def digest(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(data)
    return checksum.hexdigest()


def valid_file(path: Path, expected: dict) -> bool:
    return (not path.is_symlink() and path.is_file()
            and path.stat().st_size == expected["size"]
            and digest(path) == expected["sha256"])


def verify_file(path: Path, expected: dict, description: str) -> None:
    """Reject changed containers even if their selected members might match."""
    if path.is_symlink() or not path.is_file():
        raise ValueError(description + " is absent or nonregular: " + str(path))
    actual_size = path.stat().st_size
    actual_sha256 = digest(path)
    if actual_size != expected["size"] or actual_sha256 != expected["sha256"]:
        raise ValueError(
            f"{description} differs from its lock: expected size={expected['size']} "
            f"sha256={expected['sha256']}; actual size={actual_size} "
            f"sha256={actual_sha256}; path={path}")


def load_lock(path: Path) -> dict:
    lock = json.loads(path.read_text())
    if lock.get("schema_version") != 1 or not lock.get("members"):
        raise ValueError("Unsupported or empty codec fixture lock")
    archive = lock["archive"]
    url = urlsplit(archive["url"])
    if (url.scheme != "https" or url.hostname != "artifacts.neat.sima.ai"
            or url.username or url.password or url.query or url.fragment
            or not re.fullmatch(r"[0-9a-f]{40}", lock["source"]["commit"])):
        raise ValueError("Codec fixture lock requires an immutable production artifact")
    paths = set()
    for name, member in lock["members"].items():
        path = PurePosixPath(member["path"])
        if (not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name)
                or path.is_absolute() or ".." in path.parts
                or str(path) != member["path"] or "\\" in member["path"]
                or path.name != name or member["path"] in paths):
            raise ValueError("Unsafe or duplicate codec fixture member")
        paths.add(member["path"])
    for entry in (archive, *lock["members"].values()):
        if (not isinstance(entry["size"], int) or entry["size"] <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])):
            raise ValueError("Codec fixture lock requires positive sizes and SHA-256 hashes")
    return lock


@contextmanager
def exclusive(path: Path):
    # Cold preparation only. Tests consume the completed fixture set after this
    # command succeeds; they never race an in-progress build/install transaction.
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def obtain_archive(lock: dict, cache: Path, cli: str, offline: bool) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    expected = lock["archive"]
    target = cache / (expected["sha256"] + ".tar.gz")
    with exclusive(cache / ".download.lock"):
        if valid_file(target, expected):
            return target
        if offline:
            raise ValueError("Verified codec fixture archive is absent from the offline cache")
        with tempfile.TemporaryDirectory(prefix=".download-", dir=cache) as temporary:
            root = Path(temporary)
            subprocess.run([cli, "download", expected["url"], "-d", str(root)], check=True)
            # CLI versions may preserve or decode the URL's escaped basename.
            encoded = PurePosixPath(urlsplit(expected["url"]).path).name
            candidates = [root / name for name in {encoded, unquote(encoded)}
                          if (root / name).exists()]
            if len(candidates) != 1:
                raise ValueError("Downloaded codec fixture archive is absent or ambiguous: "
                                 + ", ".join(str(path) for path in candidates))
            verify_file(candidates[0], expected, "Downloaded codec fixture archive")
            os.replace(candidates[0], target)
    return target


def stage_members(archive: Path, lock: dict, stage: Path) -> None:
    """Read only locked regular members; never extract an archive's directory tree."""
    verify_file(archive, lock["archive"], "Codec fixture archive")
    wanted = {entry["path"]: (name, entry) for name, entry in lock["members"].items()}
    seen = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            name = member.name.removeprefix("./")
            if name not in wanted:
                continue
            if name in seen or not member.isfile() or member.size != wanted[name][1]["size"]:
                raise ValueError("Duplicate, nonregular or wrong-size codec fixture member: " + name)
            seen.add(name)
            target, expected = wanted[name]
            with source.extractfile(member) as data, (stage / target).open("xb") as output:
                shutil.copyfileobj(data, output)
            if not valid_file(stage / target, expected):
                raise ValueError("Codec fixture member differs from its lock: " + name)
    if seen != set(wanted):
        raise ValueError("Codec fixture archive lacks a complete source/reference pair")


def prepare(output: Path, lock_path: Path = DEFAULT_LOCK, *, archive: Path | None = None,
            cache: Path | None = None, cli: str = "sima-cli", offline: bool = False,
            accuracy: bool = True) -> dict:
    lock = load_lock(lock_path)
    if output.is_symlink():
        raise ValueError("Codec fixture output must not be a symlink")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(output.parent / ("." + output.name + ".prepare.lock")):
        output.mkdir(exist_ok=True)
        intact = all(valid_file(output / name, entry) for name, entry in lock["members"].items())
        if not intact:
            selected = archive if archive is not None else obtain_archive(
                lock, cache or output.parent / ".codec-perf-cache", cli, offline)
            with tempfile.TemporaryDirectory(prefix=".codec-pairs-", dir=output.parent) as temporary:
                stage = Path(temporary)
                stage_members(selected, lock, stage)
                # Validate the entire pair set before replacing any member. Each
                # file replacement is atomic, and serialized preparation repairs
                # any interrupted transaction before permitting test execution.
                for name in lock["members"]:
                    os.replace(stage / name, output / name)
        if accuracy:
            # Do not change or call the no-B generator/recipe. JPEG/IPB tests keep
            # their independent existing recipes and preparation behavior.
            sys.dont_write_bytecode = True
            from make_codec_perf_fixtures import make_accuracy_fixtures
            make_accuracy_fixtures(output, False)
        if not all(valid_file(output / name, entry) for name, entry in lock["members"].items()):
            raise ValueError("Codec fixture set changed during preparation")
    return {"archive_sha256": lock["archive"]["sha256"],
            "source_commit": lock["source"]["commit"],
            "members": {name: {"size": entry["size"], "sha256": entry["sha256"]}
                        for name, entry in lock["members"].items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--archive", type=Path, help="Previously downloaded exact extras archive")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--cli", default="sima-cli")
    parser.add_argument("--offline", action="store_true", help="Do not download missing/corrupt baseline data")
    args = parser.parse_args()
    try:
        report = prepare(args.output_dir, args.lock, archive=args.archive,
                         cache=args.cache_dir, cli=args.cli, offline=args.offline)
        print(json.dumps(report, sort_keys=True))
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.CalledProcessError, tarfile.TarError) as error:
        print("ERROR: " + str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
