#!/usr/bin/env python3
"""Validate the frozen 12-archive EVO corpus through the explicit Core decoder.

The test harness selects exactly one manifest and one ELF from each archive and
passes their explicit paths to the C++ decoder. Archive/model names are used
only as golden-test labels; they are never decoder inputs or inference facts.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import tarfile
import tempfile
from pathlib import Path


EXPECTED_ARCHIVES = {
    "evo50_ev74tess_bf16_mpk.tar.gz": (2, 28, 1, 1),
    "evo50_ev74tess_int8_mpk.tar.gz": (2, 28, 1, 1),
    "evo50_mlatess_bf16_mpk.tar.gz": (2, 28, 1, 1),
    "evo50_mlatess_int8_mpk.tar.gz": (2, 28, 1, 1),
    "evo50_mlatess_multibuff_bf16_mpk.tar.gz": (2, 28, 2, 28),
    "evo50_mlatess_multibuff_int8_mpk.tar.gz": (2, 28, 2, 28),
    "evo50_v2_ev74tess_bf16_mpk.tar.gz": (1, 25, 1, 1),
    "evo50_v2_ev74tess_int8_mpk.tar.gz": (1, 25, 1, 1),
    "evo50_v2_mlatess_bf16_mpk.tar.gz": (1, 25, 1, 1),
    "evo50_v2_mlatess_int8_mpk.tar.gz": (1, 25, 1, 1),
    # Deliberately misleading archive labels: content is proven 1/1.
    "evo50_v2_mlatess_multibuff_bf16_mpk.tar.gz": (1, 25, 1, 1),
    "evo50_v2_mlatess_multibuff_int8_mpk.tar.gz": (1, 25, 1, 1),
}

SUMMARY_RE = re.compile(r"^inputs=(\d+) outputs=(\d+) ifm=(\d+) ofm=(\d+)\s*$")


def member_bytes(archive: tarfile.TarFile, suffix: str) -> bytes:
    members = [m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)]
    if len(members) != 1:
        raise AssertionError(
            f"{archive.name}: expected exactly one regular {suffix} member, got "
            f"{[m.name for m in members]}"
        )
    stream = archive.extractfile(members[0])
    if stream is None:
        raise AssertionError(f"{archive.name}: cannot read {members[0].name}")
    return stream.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validator", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()

    if not args.corpus.is_dir():
        raise AssertionError(f"configured EVO corpus is unavailable: {args.corpus}")

    archives = {path.name: path for path in args.corpus.glob("*.tar.gz")}
    if set(archives) != set(EXPECTED_ARCHIVES):
        missing = sorted(set(EXPECTED_ARCHIVES) - set(archives))
        extra = sorted(set(archives) - set(EXPECTED_ARCHIVES))
        raise AssertionError(f"official corpus mismatch; missing={missing}, extra={extra}")

    manifest_hashes: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="legacy-afe-evo-corpus-") as temp:
        root = Path(temp)
        for index, name in enumerate(sorted(EXPECTED_ARCHIVES)):
            with tarfile.open(archives[name], mode="r:gz") as archive:
                manifest = member_bytes(archive, "_mpk.json")
                elf = member_bytes(archive, ".elf")
            manifest_hashes.add(hashlib.sha256(manifest).hexdigest())

            case = root / str(index)
            case.mkdir()
            manifest_path = case / "manifest.json"
            elf_path = case / "mla.elf"
            manifest_path.write_bytes(manifest)
            elf_path.write_bytes(elf)

            command = [str(args.validator), str(manifest_path), str(elf_path)]
            first = subprocess.run(command, check=True, text=True, capture_output=True)
            second = subprocess.run(command, check=True, text=True, capture_output=True)
            if first.stdout != second.stdout:
                raise AssertionError(f"{name}: decoder summary is not deterministic")
            match = SUMMARY_RE.match(first.stdout)
            if not match:
                raise AssertionError(f"{name}: malformed validator output: {first.stdout!r}")
            actual = tuple(int(item) for item in match.groups())
            expected = EXPECTED_ARCHIVES[name]
            if actual != expected:
                raise AssertionError(f"{name}: expected {expected}, decoded {actual}")
            print(f"PASS {name}: public={actual[0]}/{actual[1]} MLA={actual[2]}/{actual[3]}")

    if len(manifest_hashes) != 10:
        raise AssertionError(
            f"official corpus must contain ten byte-distinct MPK graphs, got {len(manifest_hashes)}"
        )
    print("PASS: 12 official archives decoded deterministically as 10 MPK graphs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
