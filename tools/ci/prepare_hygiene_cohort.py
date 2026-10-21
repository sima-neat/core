#!/usr/bin/env python3
"""Use this build's exact Scheduler inputs for host-only Core analysis.

The caller first verifies the complete Core dist using verify_core_dist.sh.
Only Scheduler is bootstrapped here; its installed coordinator owns all lower
partitioning, receipt/cohort checks and platform-runtime archive selection.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from materialize_dependencies import extract_bundle
from published_artifact import digest


def shell_exports(text):
    values = {}
    for line in text.splitlines():
        words = shlex.split(line)
        if len(words) != 2 or words[0] != 'export' or '=' not in words[1]:
            raise ValueError('Coordinator returned non-export output')
        key, value = words[1].split('=', 1)
        if not re.fullmatch(r'NEAT_[A-Z_]+', key) or key in values or '\n' in value or '\r' in value:
            raise ValueError('Invalid/duplicate coordinator export')
        values[key] = value
    return values


def coordinator(core, command, env):
    return subprocess.check_output([sys.executable, str(core / 'tools/build_dependencies.py'),
                                    *command, '--core', str(core)], text=True,
                                   env={**env, 'PYTHONDONTWRITEBYTECODE': '1'})


def prepare(args):
    core, dist, output = args.core.resolve(), args.dist.resolve(), args.output.resolve()
    # The CLI also verifies independently when called outside the workflow.
    subprocess.run(['bash', str(core / 'scripts/ci/verify_core_dist.sh'), '--dist', str(dist)], check=True,
                   stdout=sys.stderr)
    output.mkdir(parents=True, exist_ok=False)
    schedulers = []
    cores = []
    for path in sorted(dist.glob('*.deb')):
        role = subprocess.check_output(['dpkg-deb', '-f', str(path), 'Package'], text=True).strip()
        if role == 'simaai-scheduler':
            schedulers.append(path)
        if role == 'sima-neat':
            cores.append(path)
    if len(schedulers) != 1 or len(cores) != 1:
        raise ValueError('Exactly one same-build Core and Scheduler archive required')
    with tempfile.NamedTemporaryFile(suffix='.tar') as stream:
        subprocess.run(['dpkg-deb', '--fsys-tarfile', str(schedulers[0])], stdout=stream, check=True)
        stream.flush()
        extract_bundle(stream.name, output / 'bootstrap')
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith(('NEAT_KERNELS_', 'NEAT_RUNNER_', 'NEAT_SCHEDULER_',
                                          'NEAT_CODECS_', 'NEAT_RUNTIME_COMPAT_'))}
    environment['NEAT_SCHEDULER_PACKAGE_DIR'] = str(output / 'bootstrap')
    selected = shell_exports(coordinator(core, ['select', '--require', '--cohort-dir', str(dist),
                                               '--output', str(output / 'selected')], environment))
    environment.update(selected)
    selected.update(shell_exports(coordinator(core, ['platform-archives', '--platform-package-dir',
                                                     str(dist)], environment)))
    environment.update(selected)
    # Exercise the real wrapper/coordinator command, not removed Python APIs.
    coordinator(core, ['prepare', '--headers-only', '--client-test-support', '--output', str(output / 'audit')], environment)
    selected['CMAKE_PREFIX_PATH'] = str(Path(selected['NEAT_SCHEDULER_PACKAGE_DIR']) / 'usr')
    (output / 'analysis-inputs.json').write_text(json.dumps({
        'schema_version': 1, 'dist_receipt_sha256': digest(dist / 'resolved-deps-manifest.json'),
        'core_archive_sha256': digest(cores[0]), 'environment': selected,
    }, indent=2) + '\n')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', type=Path, default=Path.cwd())
    parser.add_argument('--dist', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    # Retain the existing caller interface; host analysis resolves no remote inputs.
    parser.add_argument('--environment', required=True, help='Accepted for caller compatibility')
    parser.add_argument('--cli', default='sima-cli', help='Accepted for caller compatibility')
    parser.add_argument('--github-env', type=Path, required=True)
    args = parser.parse_args()
    values = prepare(args)
    with args.github_env.open('a') as stream:
        for name, value in values.items():
            if '\n' in value or '\r' in value:
                raise ValueError('Invalid analysis environment value')
            stream.write(name + '=' + value + '\n')


if __name__ == '__main__':
    main()
