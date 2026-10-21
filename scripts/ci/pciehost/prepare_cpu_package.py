#!/usr/bin/env python3
"""Download and privately stage the current native package; never install drivers."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from urllib.parse import quote

CORE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CORE / 'tools/ci'))
from published_artifact import ENVIRONMENT_HOSTS, basename, digest, download, valid_branch
sys.path.insert(0, str(CORE / 'pcie_host/scripts'))
from pcie_delivery import stage, verify_publication_inventory


def prepare(cli, branch, commit, environment, output, prefix):
    environment = {'prod': 'production', 'prd': 'production', 'stg': 'staging'}.get(
        environment, environment)
    if (environment not in ENVIRONMENT_HOSTS or not re.fullmatch(r'[0-9a-f]{40}', commit)
            or not valid_branch(branch)):
        raise ValueError('Invalid immutable native Core selector')
    distro = subprocess.check_output(
        ['bash', '-c', 'source "$1"; simapcie_detect_host_distro', 'host-platform',
         str(CORE / 'pcie_host/scripts/host-platform.sh')], text=True).strip()
    arch = subprocess.check_output(['dpkg', '--print-architecture'], text=True).strip()
    if distro not in ('ubuntu22', 'ubuntu24') or arch != 'amd64':
        raise ValueError('CPU qualification requires the original Ubuntu x64 host')
    package_path = f'pciehost/{distro}/{arch}'
    revision = commit[:12]
    selector = f'core/{package_path}@{branch}:{revision}'
    resolved = subprocess.check_output(
        [str(cli), 'neat', 'download', selector, '--env', environment, '--json'], text=True)
    selected = json.loads(resolved[resolved.index('{'):])
    origin = 'https://' + ENVIRONMENT_HOSTS[environment]
    expected_url = f'{origin}/core/{quote(quote(branch, safe=""), safe="")}/{revision}/{package_path}/metadata.json'
    if (selected.get('repository') != 'core' or selected.get('package_path') != package_path
            or selected.get('ref') != branch or selected.get('environment') != environment
            or selected.get('requested_spec') != revision or selected.get('resolved_spec') != revision
            or selected.get('base_url') != origin or selected.get('metadata_url') != expected_url):
        raise ValueError('sima-cli resolved a different native Core publication')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    download(selected['metadata_url'], output / 'metadata.json')
    metadata = json.loads((output / 'metadata.json').read_text())
    artifact = metadata.get('artifact', {})
    if (metadata.get('repository') != 'sima-neat/core' or metadata.get('branch') != branch
            or metadata.get('commit') != commit or metadata.get('commit_folder') != revision
            or metadata.get('name') != f'gh:sima-neat/pciehost/{distro}/{arch}'
            or artifact.get('repository') != 'core' or artifact.get('package_path') != package_path
            or artifact.get('distro') != distro or artifact.get('arches') != [arch]):
        raise ValueError('Native metadata differs from selected Core commit/host')
    resources = metadata.get('resources')
    checksums = metadata.get('resources-checksum')
    if (not isinstance(resources, list) or not resources
            or any(not isinstance(name, str) for name in resources)
            or len(resources) != len(set(resources)) or not isinstance(checksums, dict)
            or set(checksums) != set(resources)):
        raise ValueError('Invalid native publication inventory')
    for name in resources:
        basename(name)
        if name == 'metadata.json' or not re.fullmatch(r'[0-9a-f]{64}', str(checksums[name])):
            raise ValueError('Invalid native publication resource')
    metadata_hash = digest(output / 'metadata.json')
    subprocess.run([str(cli), 'neat', 'download', selector, '--env', environment,
                    '-d', str(output)], check=True)
    for name, expected in {**checksums, 'metadata.json': metadata_hash}.items():
        path = output / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError('Native download differs from metadata: ' + name)
    selection = json.loads((output / 'scheduler-native-selection.json').read_text())
    if (selection.get('distro') != distro or selection.get('architecture') != arch
            or selection.get('branch') != branch):
        raise ValueError('Selected Scheduler native tuple/branch differs')
    verify_publication_inventory(output, selection, resources)
    # Trusted checkout entry point; owning producer tooling is imported only
    # after every published resource hash passes, then retains its own checks.
    stage(output, prefix)
    return {'status': 'STAGED_CPU_PACKAGE', 'distro': distro, 'architecture': arch,
            'commit': commit, 'resources': len(resources), 'hardware_qualified': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cli', default=os.environ.get('SIMA_CLI_BIN', 'sima-cli'))
    p.add_argument('--branch', required=True)
    p.add_argument('--commit', required=True)
    p.add_argument('--environment', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prefix', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(prepare(args.cli, args.branch, args.commit, args.environment,
                             args.output, args.prefix)))


if __name__ == '__main__': main()
