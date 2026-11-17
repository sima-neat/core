#!/usr/bin/env python3
"""Acquire only Scheduler's immutable native profile, never the device cohort."""
import argparse
import importlib.util
import json
import hashlib
from pathlib import Path
import shutil
import sys

CORE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE / 'tools/ci'))
from scheduler_dependency import acquire_scheduler


def local_candidate(source, destination):
    """Copy an explicitly trusted local producer output before executing helpers."""
    source = Path(source).resolve()
    metadata = json.loads((source / 'metadata.json').read_text())
    names = metadata['resources']
    hashes = metadata['resources-checksum']
    if (not names or len(names) != len(set(names)) or set(names) != set(hashes)
            or metadata.get('repository') != 'sima-neat/scheduler'):
        raise ValueError('Invalid local Scheduler publication')
    for name in names:
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Unsafe local Scheduler resource')
        path = source / name
        if (path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]):
            raise ValueError('Local Scheduler resource changed: ' + name)
    destination.mkdir()
    for name in names:
        shutil.copy2(source / name, destination / name)
    # A local producer may rebuild resources after source validation. Only
    # import helpers once the staged bytes match the captured publication.
    for name in names:
        path = destination / name
        if (path.is_symlink() or not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != hashes[name]):
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
    args = p.parse_args()
    if args.candidate and args.spec:
        p.error('--candidate and --spec are mutually exclusive')
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    candidate = root / 'candidate'
    receipt = (local_candidate(args.candidate, candidate) if args.candidate else
               acquire_scheduler(args.cli, CORE, args.branch, args.spec, args.environment, candidate,
                                 package_path=f'pcie-host-{args.distro}-{args.arch}'))
    # Execution begins only after shared transport verifies every publisher hash.
    spec = importlib.util.spec_from_file_location('scheduler_native_artifacts', candidate / 'native_artifacts.py')
    provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
    data = provider.stage(candidate, root / 'prefix', args.distro, args.arch, receipt['commit'])
    multiarch = 'x86_64-linux-gnu' if args.arch == 'amd64' else 'aarch64-linux-gnu'
    selection = {'schema_version': 1, 'profile': 'pcie-host', 'distro': args.distro, 'architecture': args.arch,
                 'branch': receipt['branch'], 'commit': receipt['commit'],
                 'candidate': str(candidate), 'candidate_sha256': provider.sha(candidate / 'host-candidate.json'),
                 'package_version': data['packages']['runtime']['payload']['control']['Version'],
                 'scheduler_dir': str(root / 'prefix/usr/lib' / multiarch / 'cmake/SimaScheduler')}
    (root / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(selection, sort_keys=True))


if __name__ == '__main__': main()
