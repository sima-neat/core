#!/usr/bin/env python3
"""Acquire only Scheduler's immutable native profile, never the device cohort."""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

CORE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CORE / 'tools/ci'))
from published_artifact import acquire


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', default='sima-cli')
    p.add_argument('--branch', required=True)
    p.add_argument('--spec', default='latest')
    p.add_argument('--environment', default='production')
    p.add_argument('--distro', choices=('ubuntu22', 'ubuntu24'), required=True)
    p.add_argument('--arch', choices=('amd64', 'arm64'), required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=False)
    candidate = root / 'candidate'
    receipt = acquire(args.cli, 'scheduler', args.branch, args.spec, args.environment, candidate,
                      package_path=f'pcie-host-{args.distro}-{args.arch}')
    # Execution begins only after shared transport verifies every publisher hash.
    spec = importlib.util.spec_from_file_location('scheduler_native_artifacts', candidate / 'native_artifacts.py')
    provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
    data = provider.stage(candidate, root / 'prefix', args.distro, args.arch, receipt['commit'])
    multiarch = 'x86_64-linux-gnu' if args.arch == 'amd64' else 'aarch64-linux-gnu'
    selection = {'schema_version': 1, 'profile': 'pcie-host', 'distro': args.distro, 'architecture': args.arch,
                 'branch': args.branch, 'commit': receipt['commit'],
                 'candidate': str(candidate), 'candidate_sha256': provider.sha(candidate / 'host-candidate.json'),
                 'package_version': data['packages']['runtime']['payload']['control']['Version'],
                 'scheduler_dir': str(root / 'prefix/usr/lib' / multiarch / 'cmake/SimaScheduler')}
    (root / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps(selection, sort_keys=True))


if __name__ == '__main__': main()
