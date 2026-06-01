#!/usr/bin/env python3
"""Generate deterministic model archive fixtures and checksum manifest."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import io
import json
import shutil
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

FIXED_MTIME = 0
FIXED_UID = 0
FIXED_GID = 0
MAX_OVERSIZED_BYTES = 2 * 1024 * 1024
TAR_BLOCK_SIZE = 512


@dataclass(frozen=True)
class FixtureSpec:
    relpath: str
    intent: str
    entries: List[dict]


def _parse_octal_ascii(raw: bytes) -> int:
    token = raw.decode("ascii", errors="ignore").strip("\x00 ").strip()
    if not token:
        return 0
    return int(token, 8)


def _format_checksum(value: int) -> bytes:
    # POSIX ustar checksum field is 8 bytes: 6 octal digits, NUL, space.
    return f"{value:06o}\0 ".encode("ascii")


def _header_checksum(header_block: bytes) -> int:
    mutable = bytearray(header_block)
    mutable[148:156] = b"        "
    return sum(mutable)


def _header_offsets(tar_bytes: bytes) -> List[int]:
    offsets: List[int] = []
    off = 0
    size = len(tar_bytes)
    while off + TAR_BLOCK_SIZE <= size:
        header = tar_bytes[off : off + TAR_BLOCK_SIZE]
        if header == b"\x00" * TAR_BLOCK_SIZE:
            break
        entry_size = _parse_octal_ascii(header[124:136])
        offsets.append(off)
        payload_blocks = (entry_size + TAR_BLOCK_SIZE - 1) // TAR_BLOCK_SIZE
        off += TAR_BLOCK_SIZE + payload_blocks * TAR_BLOCK_SIZE
    return offsets


def _set_name_field(tar_bytes: bytearray, header_offset: int, name_bytes: bytes) -> None:
    if len(name_bytes) > 100:
        raise ValueError(f"name too long for ustar header: {name_bytes!r}")
    field = name_bytes.ljust(100, b"\x00")
    tar_bytes[header_offset : header_offset + 100] = field


def _recompute_header_checksum(tar_bytes: bytearray, header_offset: int) -> None:
    header = tar_bytes[header_offset : header_offset + TAR_BLOCK_SIZE]
    checksum = _header_checksum(header)
    tar_bytes[header_offset + 148 : header_offset + 156] = _format_checksum(checksum)


def _gzip_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=FIXED_MTIME) as gz:
            gz.write(payload)


def _norm_bytes(value: str | bytes) -> bytes:
    if isinstance(value, bytes):
        return value
    return value.encode("utf-8")


def _tarinfo(name: str, mode: int = 0o644) -> tarfile.TarInfo:
    ti = tarfile.TarInfo(name=name)
    ti.mtime = FIXED_MTIME
    ti.uid = FIXED_UID
    ti.gid = FIXED_GID
    ti.uname = ""
    ti.gname = ""
    ti.mode = mode
    return ti


def _add_entry(tf: tarfile.TarFile, entry: dict) -> None:
    kind = entry.get("type", "file")
    name = entry["name"]
    if kind == "dir":
        ti = _tarinfo(name.rstrip("/") + "/", mode=0o755)
        ti.type = tarfile.DIRTYPE
        ti.size = 0
        tf.addfile(ti)
        return

    if kind == "symlink":
        ti = _tarinfo(name)
        ti.type = tarfile.SYMTYPE
        ti.linkname = entry["linkname"]
        ti.size = 0
        tf.addfile(ti)
        return

    if kind == "hardlink":
        ti = _tarinfo(name)
        ti.type = tarfile.LNKTYPE
        ti.linkname = entry["linkname"]
        ti.size = 0
        tf.addfile(ti)
        return

    data = _norm_bytes(entry.get("data", b""))
    ti = _tarinfo(name)
    ti.size = len(data)
    tf.addfile(ti, io.BytesIO(data))


def _write_tar(entries: Iterable[dict], out_path: Path) -> None:
    entries = sorted(entries, key=lambda e: (e["name"], e.get("type", "file")))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.suffix == ".rawtar":
        with tarfile.open(out_path, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for entry in entries:
                _add_entry(tf, entry)
        return

    # Use deterministic gzip headers for .tar.gz fixtures.
    with out_path.open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=FIXED_MTIME) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
                for entry in entries:
                    _add_entry(tf, entry)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _base_valid_entries() -> List[dict]:
    pipeline = {
        "pipelines": [
            {
                "sequence": [
                    {
                        "sequence_id": 1,
                        "name": "preproc_0",
                        "pluginId": "processcvu",
                        "configPath": "0_preproc.json",
                        "processor": "CVU",
                        "kernel": "preproc",
                        "input": "decoder",
                    },
                    {
                        "sequence_id": 2,
                        "name": "mla_0",
                        "pluginId": "processmla",
                        "configPath": "0_process_mla.json",
                        "processor": "MLA",
                        "kernel": "infer",
                        "input": "preproc_0",
                    },
                ]
            }
        ]
    }

    return [
        {"type": "dir", "name": "etc"},
        {"type": "dir", "name": "lib"},
        {"type": "dir", "name": "share"},
        {
            "type": "file",
            "name": "etc/pipeline_sequence.json",
            "data": json.dumps(pipeline, sort_keys=True, separators=(",", ":")),
        },
        {
            "type": "file",
            "name": "etc/0_preproc.json",
            "data": json.dumps(
                {
                    "node_name": "preproc_0",
                    "input_width": 64,
                    "input_height": 48,
                    "input_img_type": "RGB",
                    "output_width": 64,
                    "output_height": 48,
                    "output_img_type": "RGB",
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        {
            "type": "file",
            "name": "etc/0_process_mla.json",
            "data": json.dumps(
                {
                    "node_name": "mla_0",
                    "input_buffers": [{"name": "preproc_0"}],
                    "output_width": [64],
                    "output_height": [48],
                    "output_depth": [3],
                    "data_type": ["INT8"],
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        },
        {
            "type": "file",
            "name": "share/model.elf",
            "data": b"\x7fELF\x02\x01\x01\x00fixture_elf_payload",
        },
        {
            "type": "file",
            "name": "lib/model.so",
            "data": b"\x7fELF\x02\x01\x01\x00fixture_so_payload",
        },
    ]


def _fixture_specs() -> List[FixtureSpec]:
    valid = _base_valid_entries()

    multi = copy.deepcopy(valid)
    seq = {
        "pipelines": [
            {
                "sequence": [
                    {
                        "sequence_id": 1,
                        "name": "preproc_0",
                        "pluginId": "processcvu",
                        "configPath": "0_preproc.json",
                        "processor": "CVU",
                        "kernel": "preproc",
                        "input": "decoder",
                    },
                    {
                        "sequence_id": 2,
                        "name": "mla_stage_a",
                        "pluginId": "processmla",
                        "configPath": "0_process_mla_a.json",
                        "processor": "MLA",
                        "kernel": "infer",
                        "input": "preproc_0",
                    },
                    {
                        "sequence_id": 3,
                        "name": "mla_stage_b",
                        "pluginId": "processmla",
                        "configPath": "0_process_mla_b.json",
                        "processor": "MLA",
                        "kernel": "infer",
                        "input": "mla_stage_a",
                    },
                ]
            }
        ]
    }
    multi = [e for e in multi if e["name"] != "etc/pipeline_sequence.json" and e["name"] != "etc/0_process_mla.json"]
    multi.extend(
        [
            {
                "type": "file",
                "name": "etc/pipeline_sequence.json",
                "data": json.dumps(seq, sort_keys=True, separators=(",", ":")),
            },
            {
                "type": "file",
                "name": "etc/0_process_mla_a.json",
                "data": json.dumps(
                    {
                        "node_name": "mla_stage_a",
                        "input_buffers": [{"name": "preproc_0"}],
                        "output_width": [64],
                        "output_height": [48],
                        "output_depth": [3],
                        "data_type": ["INT8"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
            {
                "type": "file",
                "name": "etc/0_process_mla_b.json",
                "data": json.dumps(
                    {
                        "node_name": "mla_stage_b",
                        "input_buffers": [{"name": "mla_stage_a"}],
                        "output_width": [64],
                        "output_height": [48],
                        "output_depth": [3],
                        "data_type": ["INT8"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            },
        ]
    )

    missing_pipeline = [e for e in copy.deepcopy(valid) if e["name"] != "etc/pipeline_sequence.json"]

    schema_error = copy.deepcopy(valid)
    for e in schema_error:
        if e["name"] == "etc/pipeline_sequence.json":
            e["data"] = "{ invalid_json "

    unsupported_version = copy.deepcopy(valid)
    unsupported_version.append(
        {
            "type": "file",
            "name": "etc/manifest.json",
            "data": json.dumps({"version": 999, "name": "unsupported"}, sort_keys=True, separators=(",", ":")),
        }
    )

    duplicate_names = copy.deepcopy(valid)
    dup_seq = {
        "pipelines": [
            {
                "sequence": [
                    {
                        "sequence_id": 1,
                        "name": "stage_dup",
                        "pluginId": "processcvu",
                        "configPath": "0_preproc.json",
                        "processor": "CVU",
                        "kernel": "preproc",
                        "input": "decoder",
                    },
                    {
                        "sequence_id": 2,
                        "name": "stage_dup",
                        "pluginId": "processmla",
                        "configPath": "0_process_mla.json",
                        "processor": "MLA",
                        "kernel": "infer",
                        "input": "stage_dup",
                    },
                ]
            }
        ]
    }
    for e in duplicate_names:
        if e["name"] == "etc/pipeline_sequence.json":
            e["data"] = json.dumps(dup_seq, sort_keys=True, separators=(",", ":"))

    unknown_kernel = copy.deepcopy(valid)
    unk_seq = {
        "pipelines": [
            {
                "sequence": [
                    {
                        "sequence_id": 1,
                        "name": "preproc_0",
                        "pluginId": "processcvu",
                        "configPath": "0_preproc.json",
                        "processor": "CVU",
                        "kernel": "preproc",
                        "input": "decoder",
                    },
                    {
                        "sequence_id": 2,
                        "name": "mystage",
                        "pluginId": "processmla",
                        "configPath": "0_process_mla.json",
                        "processor": "MLA",
                        "kernel": "unknown_kernel_type",
                        "input": "preproc_0",
                    },
                ]
            }
        ]
    }
    for e in unknown_kernel:
        if e["name"] == "etc/pipeline_sequence.json":
            e["data"] = json.dumps(unk_seq, sort_keys=True, separators=(",", ":"))

    invalid_dep = copy.deepcopy(valid)
    bad_dep_seq = {
        "pipelines": [
            {
                "sequence": [
                    {
                        "sequence_id": 1,
                        "name": "preproc_0",
                        "pluginId": "processcvu",
                        "configPath": "0_preproc.json",
                        "processor": "CVU",
                        "kernel": "preproc",
                        "input": "nonexistent_stage",
                    },
                    {
                        "sequence_id": 2,
                        "name": "mla_0",
                        "pluginId": "processmla",
                        "configPath": "0_process_mla.json",
                        "processor": "MLA",
                        "kernel": "infer",
                        "input": "preproc_0",
                    },
                ]
            }
        ]
    }
    for e in invalid_dep:
        if e["name"] == "etc/pipeline_sequence.json":
            e["data"] = json.dumps(bad_dep_seq, sort_keys=True, separators=(",", ":"))

    # Use a Windows-style drive-prefix entry (no '..', no leading '/') so the
    # GNU tar listing does not strip-and-warn (which exits the tar -tvzf subprocess
    # non-zero on stricter runners and gets classified as InvalidArchive).
    # ModelArchiveLoader::normalize_entry_path catches the drive
    # prefix at the dedicated check and raises ErrorClass::PathTraversal —
    # same security signal, but reaches the loader's own normalization step.
    path_traversal = copy.deepcopy(valid)
    path_traversal.append({"type": "file", "name": "C:owned.json", "data": "{}"})

    absolute_path = copy.deepcopy(valid)
    absolute_path.append({"type": "file", "name": "/tmp/owned.json", "data": "{}"})

    mixed_sep = copy.deepcopy(valid)
    mixed_sep.append({"type": "file", "name": "..\\evil.json", "data": "{}"})

    symlink_escape = copy.deepcopy(valid)
    symlink_escape.append({"type": "symlink", "name": "etc/link_escape", "linkname": "../../etc/passwd"})

    hardlink_escape = copy.deepcopy(valid)
    hardlink_escape.append({"type": "hardlink", "name": "etc/hard_escape", "linkname": "../../etc/passwd"})

    oversized_entry = copy.deepcopy(valid)
    oversized_entry.append({"type": "file", "name": "etc/huge_blob.json", "data": b"A" * MAX_OVERSIZED_BYTES})

    unicode_traversal_fullwidth_slash = copy.deepcopy(valid)
    unicode_traversal_fullwidth_slash.append(
        {"type": "file", "name": "etc/..／evil.json", "data": "{}"}
    )

    unicode_traversal_division_slash = copy.deepcopy(valid)
    unicode_traversal_division_slash.append(
        {"type": "file", "name": "etc/..∕evil.json", "data": "{}"}
    )

    unicode_traversal_fullwidth_backslash = copy.deepcopy(valid)
    unicode_traversal_fullwidth_backslash.append(
        {"type": "file", "name": "etc/..＼evil.json", "data": "{}"}
    )

    unicode_confusable_dotdot = copy.deepcopy(valid)
    unicode_confusable_dotdot.append(
        {"type": "file", "name": "etc/．．/evil.json", "data": "{}"}
    )

    duplicate_header_same_path = copy.deepcopy(valid)
    duplicate_header_same_path.extend(
        [
            {"type": "file", "name": "etc/dup_header.json", "data": "{\"version\":1}"},
            {"type": "file", "name": "etc/dup_header.json", "data": "{\"version\":2}"},
        ]
    )

    duplicate_header_safe_then_traversal = copy.deepcopy(valid)
    duplicate_header_safe_then_traversal.extend(
        [
            {"type": "file", "name": "etc/dup_mask.json", "data": "{\"safe\":true}"},
            {"type": "file", "name": "./etc/dup_mask.json", "data": "{\"safe\":false}"},
        ]
    )

    json_deep_nesting_256 = copy.deepcopy(valid)
    deep_obj = {"leaf": True}
    for _ in range(256):
        deep_obj = {"n": deep_obj}
    for e in json_deep_nesting_256:
        if e["name"] == "etc/pipeline_sequence.json":
            e["data"] = json.dumps(deep_obj, separators=(",", ":"))

    json_duplicate_keys_top = copy.deepcopy(valid)
    for e in json_duplicate_keys_top:
        if e["name"] == "etc/pipeline_sequence.json":
            e["data"] = '{"pipelines":[],"pipelines":[{"sequence":[{"sequence_id":1,"name":"mla_0","pluginId":"processmla","configPath":"0_process_mla.json","processor":"MLA","kernel":"infer","input":"decoder"}]}]}'

    json_duplicate_keys_nested = copy.deepcopy(valid)
    for e in json_duplicate_keys_nested:
        if e["name"] == "etc/pipeline_sequence.json":
            e["data"] = '{"pipelines":[{"sequence":[{"sequence_id":1,"name":"dup_kernel","pluginId":"processmla","configPath":"0_process_mla.json","processor":"MLA","kernel":"infer","kernel":"unknown_kernel_type","input":"decoder"}]}]}'

    json_huge_numeric_sequence_id = copy.deepcopy(valid)
    for e in json_huge_numeric_sequence_id:
        if e["name"] == "etc/pipeline_sequence.json":
            e["data"] = '{"pipelines":[{"sequence":[{"sequence_id":9223372036854775807,"name":"mla_0","pluginId":"processmla","configPath":"0_process_mla.json","processor":"MLA","kernel":"infer","input":"decoder"}]}]}'

    return [
        FixtureSpec("valid/basic_valid.tar.gz", "baseline valid package (.tar.gz)", valid),
        FixtureSpec("valid/multi_stage_valid.tar.gz", "valid package with multi-stage inference", multi),
        FixtureSpec("invalid/missing_pipeline_sequence.tar.gz", "missing required pipeline_sequence.json", missing_pipeline),
        FixtureSpec("invalid/schema_error_sequence.tar.gz", "invalid JSON in pipeline_sequence.json", schema_error),
        FixtureSpec("invalid/unsupported_version.tar.gz", "manifest version not supported", unsupported_version),
        FixtureSpec("invalid/duplicate_stage_names.tar.gz", "duplicate stage names in sequence", duplicate_names),
        FixtureSpec("invalid/unknown_kernel.tar.gz", "unknown kernel in sequence", unknown_kernel),
        FixtureSpec("invalid/invalid_dependency.tar.gz", "sequence input references unknown stage", invalid_dep),
        FixtureSpec("invalid/path_traversal.tar.gz", "drive-prefix path traversal entry", path_traversal),
        FixtureSpec("invalid/absolute_path.tar.gz", "absolute path entry", absolute_path),
        FixtureSpec("invalid/mixed_separator_traversal.tar.gz", "mixed separator traversal entry", mixed_sep),
        FixtureSpec("invalid/symlink_escape.tar.gz", "symlink escape entry", symlink_escape),
        FixtureSpec("invalid/hardlink_escape.tar.gz", "hardlink escape entry", hardlink_escape),
        FixtureSpec("invalid/oversized_entry.tar.gz", "oversized entry for size-limit checks", oversized_entry),
        FixtureSpec("invalid/unicode_traversal_fullwidth_slash.tar.gz", "unicode fullwidth slash traversal", unicode_traversal_fullwidth_slash),
        FixtureSpec("invalid/unicode_traversal_division_slash.tar.gz", "unicode division slash traversal", unicode_traversal_division_slash),
        FixtureSpec("invalid/unicode_traversal_fullwidth_backslash.tar.gz", "unicode fullwidth backslash traversal", unicode_traversal_fullwidth_backslash),
        FixtureSpec("invalid/unicode_confusable_dotdot.tar.gz", "unicode confusable dot traversal", unicode_confusable_dotdot),
        FixtureSpec("invalid/duplicate_header_same_path.tar.gz", "duplicate tar header for same path", duplicate_header_same_path),
        FixtureSpec("invalid/duplicate_header_safe_then_traversal.tar.gz", "duplicate header using alternate raw path form", duplicate_header_safe_then_traversal),
        FixtureSpec("invalid/json_deep_nesting_256.tar.gz", "pipeline_sequence exceeds JSON depth limit", json_deep_nesting_256),
        FixtureSpec("invalid/json_duplicate_keys_top.tar.gz", "duplicate keys in top-level pipeline_sequence object", json_duplicate_keys_top),
        FixtureSpec("invalid/json_duplicate_keys_nested.tar.gz", "duplicate keys in nested stage entry", json_duplicate_keys_nested),
        FixtureSpec("invalid/json_huge_numeric_sequence_id.tar.gz", "oversized numeric field in sequence entry", json_huge_numeric_sequence_id),
    ]


def _build_fixture_tree(out_root: Path) -> Dict[str, dict]:
    specs = _fixture_specs()
    generated: Dict[str, dict] = {}

    for spec in specs:
        target = out_root / spec.relpath
        _write_tar(spec.entries, target)
        generated[spec.relpath] = {
            "intent": spec.intent,
            "path": spec.relpath,
            "sha256": _sha256(target),
            "size_bytes": target.stat().st_size,
        }

    # Truncated archive fixture derived from valid package.
    trunc_rel = "invalid/truncated_archive.tar.gz"
    src = out_root / "valid/basic_valid.tar.gz"
    dst = out_root / trunc_rel
    raw = src.read_bytes()
    keep = max(32, len(raw) // 2)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(raw[:keep])
    generated[trunc_rel] = {
        "intent": "truncated archive payload",
        "path": trunc_rel,
        "sha256": _sha256(dst),
        "size_bytes": dst.stat().st_size,
    }

    # Header/path mutation fixtures derived from an uncompressed scratch tar.
    scratch_tar = out_root / ".scratch" / "basic_valid.rawtar"
    _write_tar(_base_valid_entries(), scratch_tar)
    tar_bytes = bytearray(scratch_tar.read_bytes())
    offsets = _header_offsets(tar_bytes)
    if not offsets:
        raise RuntimeError("failed to parse headers from scratch tar")

    # invalid/non_utf8_path_bytes.tar.gz
    non_utf8_rel = "invalid/non_utf8_path_bytes.tar.gz"
    non_utf8_tar = bytearray(tar_bytes)
    _set_name_field(non_utf8_tar, offsets[0], b"etc/\xffbad_name.json")
    _recompute_header_checksum(non_utf8_tar, offsets[0])
    non_utf8_path = out_root / non_utf8_rel
    _gzip_bytes(non_utf8_path, bytes(non_utf8_tar))
    generated[non_utf8_rel] = {
        "intent": "tar entry path contains invalid UTF-8 bytes",
        "path": non_utf8_rel,
        "sha256": _sha256(non_utf8_path),
        "size_bytes": non_utf8_path.stat().st_size,
    }

    # invalid/checksum_corrupt_header.tar.gz
    checksum_header_rel = "invalid/checksum_corrupt_header.tar.gz"
    checksum_header_tar = bytearray(tar_bytes)
    header_off = offsets[0]
    checksum_header_tar[header_off + 148 : header_off + 156] = b"000000\0 "
    checksum_header_path = out_root / checksum_header_rel
    _gzip_bytes(checksum_header_path, bytes(checksum_header_tar))
    generated[checksum_header_rel] = {
        "intent": "tar header checksum corruption",
        "path": checksum_header_rel,
        "sha256": _sha256(checksum_header_path),
        "size_bytes": checksum_header_path.stat().st_size,
    }

    # invalid/checksum_corrupt_body.tar.gz (gzip trailer CRC corruption)
    checksum_body_rel = "invalid/checksum_corrupt_body.tar.gz"
    checksum_body_path = out_root / checksum_body_rel
    checksum_body = bytearray((out_root / "valid/basic_valid.tar.gz").read_bytes())
    if len(checksum_body) < 8:
        raise RuntimeError("valid/basic_valid.tar.gz too small for gzip CRC mutation")
    checksum_body[-8] ^= 0xFF
    checksum_body_path.write_bytes(checksum_body)
    generated[checksum_body_rel] = {
        "intent": "gzip payload CRC corruption",
        "path": checksum_body_rel,
        "sha256": _sha256(checksum_body_path),
        "size_bytes": checksum_body_path.stat().st_size,
    }

    shutil.rmtree(out_root / ".scratch", ignore_errors=True)
    return generated


def _manifest_payload(generated: Dict[str, dict]) -> dict:
    fixtures = [generated[k] for k in sorted(generated.keys())]
    return {
        "format_version": 1,
        "generator": "tests/tools/make_model_archive_fixtures.py",
        "fixtures": fixtures,
    }


def _write_manifest(manifest_path: Path, generated: Dict[str, dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _manifest_payload(generated)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _check(root: Path, manifest_path: Path) -> int:
    with tempfile.TemporaryDirectory(prefix="model_archive_fixtures_check_") as td:
        expected_root = Path(td)
        expected = _build_fixture_tree(expected_root)
        expected_payload = _manifest_payload(expected)

    current_payload = _load_manifest(manifest_path)
    if current_payload != expected_payload:
        print("Manifest mismatch. Re-run generator.")
        return 1

    for fixture in current_payload.get("fixtures", []):
        rel = fixture["path"]
        path = root / rel
        if not path.exists():
            print(f"Missing fixture file: {rel}")
            return 1
        actual_hash = _sha256(path)
        if actual_hash != fixture["sha256"]:
            print(f"Checksum mismatch for {rel}: expected {fixture['sha256']} got {actual_hash}")
            return 1
        actual_size = path.stat().st_size
        if actual_size != fixture["size_bytes"]:
            print(f"Size mismatch for {rel}: expected {fixture['size_bytes']} got {actual_size}")
            return 1

    print("Model archive fixture check passed.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate deterministic model archive fixtures")
    parser.add_argument("--root", default="test-assets/model-archive", help="fixture root directory")
    parser.add_argument(
        "--manifest",
        default="test-assets/model-archive/fixtures_manifest.json",
        help="fixture manifest path",
    )
    parser.add_argument("--check", action="store_true", help="verify fixtures and manifest")
    parser.add_argument("--clean", action="store_true", help="remove existing fixture root before generation")
    args = parser.parse_args()

    root = Path(args.root)
    manifest_path = Path(args.manifest)

    if args.check:
        return _check(root, manifest_path)

    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    generated = _build_fixture_tree(root)
    _write_manifest(manifest_path, generated)

    print(f"Generated {len(generated)} fixtures under {root}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
