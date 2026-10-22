"""Core's Scheduler selection policy, shared by device and PCIe-host acquisition."""
import json
import os
from pathlib import Path
import subprocess
import sys

from published_artifact import PublicationResolutionError, acquire


def current_core_tag(core):
    if os.environ.get('GITHUB_REF_TYPE') == 'tag' and os.environ.get('GITHUB_REF_NAME'):
        return os.environ['GITHUB_REF_NAME']
    # Match Core's LLiMa snap policy, including local exact-tag checkouts.
    try:
        result = subprocess.run(['git', '-C', str(core), 'describe', '--tags', '--exact-match', 'HEAD'],
                                text=True, capture_output=True)
        return result.stdout.strip() if result.returncode == 0 else ''
    except FileNotFoundError:
        return ''


def acquire_scheduler(cli, core, branch, spec, environment, output, *, package_path=''):
    """Try the snap branch then develop; pins, tags and verification stay strict."""
    policy = json.loads((Path(core) / 'deps/manifest.json').read_text())['scheduler']
    snap = policy == {'policy': 'snap'} or policy == ''
    tag = current_core_tag(core) if snap else ''
    if snap:
        if not tag and (not branch or branch == 'HEAD'):
            print('Could not determine current branch for Scheduler snap; using develop.', file=sys.stderr)
            branch = 'develop'
        branch = tag or branch
        selected_spec = 'latest'
    elif isinstance(policy, str) and policy:
        branch, _, selected_spec = policy.partition(':')
        selected_spec = selected_spec or 'latest'
    elif isinstance(policy, dict) and not policy.get('policy') and set(policy) <= {'branch', 'ref', 'spec'}:
        branch = policy.get('branch', policy.get('ref', ''))
        selected_spec = policy.get('spec') or 'latest'
        if not branch:
            raise ValueError('Explicit Scheduler policy requires a branch')
    else:
        raise ValueError('Unsupported explicit Scheduler dependency policy')
    # PCIe's explicit revision override remains strict, even under branch snap.
    selected_spec = spec or selected_spec
    fallback = snap and not tag and selected_spec == 'latest' and branch != 'develop'
    try:
        return acquire(cli, 'scheduler', branch, selected_spec, environment, output, package_path=package_path)
    except PublicationResolutionError:
        if not fallback:
            raise
        repository = 'scheduler' + ('/' + package_path if package_path else '')
        print(f'Could not resolve {repository}@{branch}:{selected_spec}; retrying develop:latest.', file=sys.stderr)
    return acquire(cli, 'scheduler', 'develop', 'latest', environment, output, package_path=package_path)
