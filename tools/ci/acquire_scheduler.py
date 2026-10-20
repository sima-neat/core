#!/usr/bin/env python3
"""Explicit CI acquisition of Scheduler's complete published client cohort."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from published_artifact import acquire, digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--core', type=Path, default=Path.cwd())
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cli', default='sima-cli')
    p.add_argument('--environment', required=True)
    p.add_argument('--branch', required=True)
    args = p.parse_args()
    core = args.core.resolve()
    # Explicit local sources/packages are authoritative; this is not fallback.
    if any(value for name, value in os.environ.items()
           if name.startswith(('NEAT_KERNELS_', 'NEAT_RUNNER_', 'NEAT_SCHEDULER_', 'NEAT_CODECS_'))
           and name.endswith(('_SOURCE', '_PACKAGE_DIR'))):
        raise ValueError('Remote CI acquisition cannot override explicit local inputs')
    policy = json.loads((core / 'deps/manifest.json').read_text())['scheduler']
    if policy == {'policy': 'snap'} or policy == '':
        branch, spec = args.branch, 'latest'
    elif isinstance(policy, str) and policy:
        branch, _, spec = policy.partition(':')
        spec = spec or 'latest'
    elif isinstance(policy, dict) and not policy.get('policy') and set(policy) <= {'branch', 'ref', 'spec'}:
        branch = policy.get('branch', policy.get('ref', ''))
        spec = policy.get('spec') or 'latest'
        if not branch:
            raise ValueError('Explicit Scheduler policy requires a branch')
    else:
        raise ValueError('Unsupported explicit Scheduler dependency policy')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    receipt = acquire(args.cli, 'scheduler', branch, spec, args.environment, root / 'candidate')
    bootstrap = root / 'candidate/client_bootstrap.py'
    if receipt['resources'].get(bootstrap.name) != digest(bootstrap):
        raise ValueError('Publisher did not bind Scheduler client bootstrap')
    subprocess.run([sys.executable, str(bootstrap), '--candidate', str(root / 'candidate'),
                    '--commit', receipt['commit'], '--core', str(core), '--output', str(root / 'client')], check=True,
                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})


if __name__ == '__main__':
    main()
