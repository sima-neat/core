#!/usr/bin/env python3
"""Acquire only Scheduler's immutable native profile, never the device cohort."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys

CORE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE / 'tools/ci'))
from scheduler_dependency import acquire_scheduler


MAX_RESOURCES = 256
MAX_RESOURCE_BYTES = 1024 * 1024 * 1024
MAX_CANDIDATE_BYTES = 2 * 1024 * 1024 * 1024
MAX_METADATA_BYTES = 1024 * 1024


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def local_candidate(source, destination, expected_commit, distro, arch):
    """Copy an explicitly trusted local producer output before executing helpers."""
    if not re.fullmatch(r'[0-9a-f]{40}', expected_commit):
        raise ValueError('Exact Scheduler candidate requires a full lowercase commit')
    source = Path(source).resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError('Local Scheduler candidate must be a real directory')
    entries = list(source.iterdir())
    if (len(entries) > MAX_RESOURCES + 1
            or any(path.is_symlink() or not path.is_file() for path in entries)):
        raise ValueError('Local Scheduler candidate has invalid or excessive entries')
    metadata_path = source / 'metadata.json'
    if (not metadata_path.is_file() or metadata_path.stat().st_size <= 0
            or metadata_path.stat().st_size > MAX_METADATA_BYTES):
        raise ValueError('Invalid local Scheduler metadata')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    names = metadata.get('resources')
    hashes = metadata.get('resources-checksum')
    artifact = metadata.get('artifact')
    if (not isinstance(names, list) or not names or len(names) > MAX_RESOURCES
            or len(names) != len(set(names)) or not isinstance(hashes, dict)
            or set(names) != set(hashes)
            or not {'host-candidate.json', 'native_artifacts.py'} <= set(names)
            or any(not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)
                   for value in hashes.values())
            or metadata.get('repository') != 'sima-neat/scheduler'
            or metadata.get('commit') != expected_commit
            or artifact != {'profile': 'pcie-host', 'distro': distro, 'architecture': arch}):
        raise ValueError('Invalid local Scheduler publication')

    expected_entries = {'metadata.json', *names}
    if {path.name for path in entries} != expected_entries:
        raise ValueError('Local Scheduler candidate contains undeclared resources')
    total = metadata_path.stat().st_size
    for name in names:
        if (not isinstance(name, str) or Path(name).name != name
                or name in ('', '.', '..') or '/' in name or '\\' in name):
            raise ValueError('Unsafe local Scheduler resource')
        path = source / name
        size = path.stat().st_size
        total += size
        if size <= 0 or size > MAX_RESOURCE_BYTES or total > MAX_CANDIDATE_BYTES:
            raise ValueError('Local Scheduler candidate exceeds the size bound')

    # Hash only after entry count and every size are admitted.
    for name in names:
        path = source / name
        if file_digest(path) != hashes[name]:
            raise ValueError('Local Scheduler resource changed: ' + name)

    host_candidate = json.loads((source / 'host-candidate.json').read_text(encoding='utf-8'))
    if (host_candidate.get('schema_version') != 2
            or host_candidate.get('profile') != 'pcie-host'
            or host_candidate.get('distro') != distro
            or host_candidate.get('architecture') != arch
            or host_candidate.get('source_commit') != expected_commit):
        raise ValueError('Local Scheduler host candidate identity differs')

    destination.mkdir()
    for name in names:
        shutil.copy2(source / name, destination / name)
    # A local producer may rebuild resources after source validation. Only
    # import helpers once the staged bytes match the captured publication.
    for name in names:
        path = destination / name
        if (path.is_symlink() or not path.is_file()
                or path.stat().st_size != (source / name).stat().st_size
                or file_digest(path) != hashes[name]):
            raise ValueError('Staged Scheduler resource changed: ' + name)
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', default='sima-cli')
    p.add_argument('--branch', required=True)
    p.add_argument('--spec', help='Explicit revision override; otherwise use deps/manifest.json')
    p.add_argument('--environment', default='production')
    p.add_argument('--distro', choices=('ubuntu22', 'ubuntu24'), required=True)
    p.add_argument('--arch', choices=('amd64', 'arm64'), required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--candidate', type=Path,
                   help='Explicitly trusted local Scheduler producer output (no download)')
    p.add_argument('--commit', help='Full Scheduler commit selected with --candidate')
    args = p.parse_args()
    if bool(args.candidate) != bool(args.commit):
        p.error('--candidate and --commit must be supplied together')
    if args.candidate and args.spec:
        p.error('--candidate and --spec are mutually exclusive')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    candidate = root / 'candidate'
    receipt = (local_candidate(args.candidate, candidate, args.commit, args.distro, args.arch)
               if args.candidate else
               acquire_scheduler(args.cli, CORE, args.branch, args.spec, args.environment, candidate,
                                 package_path=f'pcie-host-{args.distro}-{args.arch}'))
    # Execution begins only after shared transport verifies every publisher hash.
    spec = importlib.util.spec_from_file_location('scheduler_native_artifacts', candidate / 'native_artifacts.py')
    provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
    expected_commit = args.commit if args.candidate else receipt['commit']
    data = provider.stage(candidate, root / 'prefix', args.distro, args.arch, expected_commit)
    multiarch = 'x86_64-linux-gnu' if args.arch == 'amd64' else 'aarch64-linux-gnu'
    selection = {'schema_version': 1, 'profile': 'pcie-host', 'distro': args.distro, 'architecture': args.arch,
                 'branch': receipt['branch'], 'commit': expected_commit,
                 'candidate': str(candidate), 'candidate_sha256': provider.sha(candidate / 'host-candidate.json'),
                 'package_version': data['packages']['runtime']['payload']['control']['Version'],
                 'scheduler_dir': str(root / 'prefix/usr/lib' / multiarch / 'cmake/SimaScheduler')}
    (root / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(selection, sort_keys=True))


if __name__ == '__main__': main()
