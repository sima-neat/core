#!/usr/bin/env python3
"""Make the census corpus: each catalogue component package with the contents of its MLA program's
sections set to zero past their first bytes (README.md, tier T0).

The census asks the planner, which reads a package's manifests and the MLA ELF's section table
(names, sizes, and the 16-byte header at the start of each data section that states its extent),
never the program's instructions or weights. Sections zeroed past their first KEEP_BYTES keep
every offset, size and header, so the package loads as before, and the archive compresses to a
few hundred kilobytes: the 280
components fit in tens of megabytes instead of 17 GB, which is what makes the census cheap enough
for every change. Every other file of a package is kept byte for byte. The census on this corpus is
checked against the census on the full packages (qualify.py census --compare-with).

  make_census_corpus.py --to <dir> (--source-dir <dir of <id>.tar.gz> | --source-list <file>) [--jobs N]
  --source-list lines: <component id> <registry zip> <member .tar.gz in the zip>
"""
import argparse
import concurrent.futures
import io
import os
import pathlib
import struct
import subprocess
import sys
import tarfile

MLA_MACHINE = 0x109
KEEP_TYPES = {2, 3, 7, 11}  # SHT_SYMTAB, SHT_STRTAB, SHT_NOTE, SHT_DYNSYM
KEEP_BYTES = 64  # of every other section: its header (a QMLA data section's extent is in its first 16)


def zero_mla_sections(data: bytes) -> bytes:
    """The ELF with every section's contents past its first KEEP_BYTES zeroed, symbol and string
    tables and notes kept whole;
    bytes that are not an MLA ELF (ELF64 little-endian, e_machine 0x109) come back unchanged."""
    if len(data) < 64 or data[:4] != b'\x7fELF' or data[4] != 2 or data[5] != 1:
        return data
    (machine,) = struct.unpack_from('<H', data, 18)
    if machine != MLA_MACHINE:
        return data
    (shoff,) = struct.unpack_from('<Q', data, 40)
    shentsize, shnum, shstrndx = struct.unpack_from('<HHH', data, 58)
    out = bytearray(data)
    for i in range(shnum):
        base = shoff + i * shentsize
        sh_type, = struct.unpack_from('<I', data, base + 4)
        sh_offset, sh_size = struct.unpack_from('<QQ', data, base + 24)
        if i == shstrndx or sh_type in KEEP_TYPES or sh_type == 8:  # 8: SHT_NOBITS has no bytes
            continue
        start, end = sh_offset + KEEP_BYTES, min(len(out), sh_offset + sh_size)
        if start < end:
            out[start:end] = bytes(end - start)
    return bytes(out)


def slim(source: bytes) -> bytes:
    src = tarfile.open(fileobj=io.BytesIO(source), mode='r:gz')
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz', compresslevel=6) as dst:
        for member in src.getmembers():
            if member.isfile():
                data = zero_mla_sections(src.extractfile(member).read())
                info = tarfile.TarInfo(member.name)
                info.size, info.mode, info.mtime = len(data), member.mode, member.mtime
                info.uid = info.gid = 0
                dst.addfile(info, io.BytesIO(data))
            else:
                dst.addfile(member)
    return buf.getvalue()


def one(job):
    cid, kind, a, b, to = job
    target = pathlib.Path(to) / f'{cid}.tar.gz'
    if target.exists():
        return cid, target.stat().st_size, 'kept'
    if kind == 'dir':
        source = pathlib.Path(a).read_bytes()
    else:
        source = subprocess.run(['unzip', '-p', a, b], check=True, capture_output=True).stdout
    data = slim(source)
    tmp = target.with_suffix('.part')
    tmp.write_bytes(data)
    os.replace(tmp, target)
    return cid, len(data), f'{len(source)} -> {len(data)} bytes'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--to', required=True)
    ap.add_argument('--source-dir')
    ap.add_argument('--source-list')
    ap.add_argument('--jobs', type=int, default=8)
    args = ap.parse_args()
    pathlib.Path(args.to).mkdir(parents=True, exist_ok=True)
    jobs = []
    if args.source_dir:
        for p in sorted(pathlib.Path(args.source_dir).glob('*.tar.gz')):
            jobs.append((p.name[:-len('.tar.gz')], 'dir', str(p), None, args.to))
    if args.source_list:
        for line in open(args.source_list):
            w = line.split()
            if len(w) == 3:
                jobs.append((w[0], 'zip', w[1], w[2], args.to))
    total = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        for cid, size, note in ex.map(one, jobs):
            total += size
            print(cid, note, flush=True)
    print(f'{len(jobs)} components, {total} bytes in {args.to}')


if __name__ == '__main__':
    sys.exit(main())
