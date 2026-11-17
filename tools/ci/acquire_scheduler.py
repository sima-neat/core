#!/usr/bin/env python3
"""Explicit CI acquisition of Scheduler's complete published client cohort."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

from published_artifact import digest
from scheduler_dependency import acquire_scheduler


MAX_CANDIDATE_RESOURCES = 256
MAX_CANDIDATE_BYTES = 2 * 1024 * 1024 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024


def _safe_resource_name(name):
    return (isinstance(name, str) and name not in ('', '.', '..')
            and Path(name).name == name and '/' not in name and '\\' not in name)


def stage_run_candidate(source, destination, commit, branch):
    """Admit and copy one immutable Scheduler Actions candidate.

    The caller binds the Actions artifact to ``commit`` before this function is
    reached.  This second boundary validates the publisher's content manifest
    before any downloaded Python is imported or executed.
    """
    if not re.fullmatch(r'[0-9a-f]{40}', commit):
        raise ValueError('Exact Scheduler candidate requires a full lowercase commit')
    source = Path(source).resolve(strict=True)
    if not source.is_dir() or source.is_symlink():
        raise ValueError('Scheduler candidate must be a real directory')
    manifest_path = source / 'candidate.json'
    if (manifest_path.is_symlink() or not manifest_path.is_file()
            or manifest_path.stat().st_size > MAX_MANIFEST_BYTES):
        raise ValueError('Invalid Scheduler candidate manifest')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    resources = manifest.get('resources')
    if (manifest.get('schema_version') != 1
            or manifest.get('component') != 'scheduler'
            or manifest.get('source_commit') != commit
            or manifest.get('qualified') is not False
            or not isinstance(resources, dict)
            or not resources
            or len(resources) > MAX_CANDIDATE_RESOURCES):
        raise ValueError('Scheduler candidate identity differs from selected run')

    total = manifest_path.stat().st_size
    admitted = {}
    for name, record in resources.items():
        if (not _safe_resource_name(name) or not isinstance(record, dict)
                or set(record) != {'sha256', 'size'}
                or not isinstance(record['size'], int) or isinstance(record['size'], bool)
                or record['size'] <= 0
                or not re.fullmatch(r'[0-9a-f]{64}', str(record['sha256']))):
            raise ValueError('Invalid Scheduler candidate resource declaration')
        path = source / name
        if (path.is_symlink() or not path.is_file() or path.stat().st_size != record['size']
                or digest(path) != record['sha256']):
            raise ValueError('Scheduler candidate resource changed: ' + name)
        total += record['size']
        if total > MAX_CANDIDATE_BYTES:
            raise ValueError('Scheduler candidate exceeds the admitted size bound')
        admitted[name] = record['sha256']

    observed = {path.name for path in source.iterdir() if path.is_file() and not path.is_symlink()}
    expected = {'candidate.json', *resources}
    if observed != expected or any(path.is_symlink() or not path.is_file() for path in source.iterdir()):
        raise ValueError('Scheduler candidate contains undeclared resources')

    destination.mkdir(parents=True, exist_ok=False)
    shutil.copy2(manifest_path, destination / manifest_path.name)
    for name in resources:
        shutil.copy2(source / name, destination / name)
    if digest(destination / 'candidate.json') != digest(manifest_path):
        raise ValueError('Staged Scheduler candidate manifest changed')
    for name, expected_hash in admitted.items():
        path = destination / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected_hash:
            raise ValueError('Staged Scheduler candidate resource changed: ' + name)
    return {'branch': branch, 'commit': commit, 'resources': admitted}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cli', default='sima-cli')
    p.add_argument('--environment', required=True)
    p.add_argument('--branch', required=True)
    p.add_argument('--candidate', type=Path,
                   help='Exact Scheduler Actions candidate already admitted by run ID')
    p.add_argument('--commit', help='Full Scheduler commit selected with --candidate')
    args = p.parse_args()
    if bool(args.candidate) != bool(args.commit):
        p.error('--candidate and --commit must be supplied together')
    core = args.core.resolve()
    # Explicit local sources/packages are authoritative; this is not fallback.
    if any(value for name, value in os.environ.items()
           if name.startswith(('NEAT_KERNELS_', 'NEAT_RUNNER_', 'NEAT_SCHEDULER_', 'NEAT_CODECS_'))
           and name.endswith(('_SOURCE', '_PACKAGE_DIR'))):
        raise ValueError('Remote CI acquisition cannot override explicit local inputs')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    receipt = (stage_run_candidate(args.candidate, root / 'candidate', args.commit, args.branch)
               if args.candidate else
               acquire_scheduler(args.cli, core, args.branch, None, args.environment, root / 'candidate'))
    bootstrap = root / 'candidate/client_bootstrap.py'
    if receipt['resources'].get(bootstrap.name) != digest(bootstrap):
        raise ValueError('Publisher did not bind Scheduler client bootstrap')
    subprocess.run([sys.executable, str(bootstrap), '--candidate', str(root / 'candidate'),
                    '--commit', receipt['commit'], '--core', str(core), '--output', str(root / 'client')], check=True,
                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})


if __name__ == '__main__':
    main()
