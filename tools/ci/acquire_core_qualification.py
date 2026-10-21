#!/usr/bin/env python3
"""Acquire Core's immutable published inputs without installing any package.

Core publishes its original CLI metadata, not Scheduler's publisher envelope.
Keep that distinction explicit and leave provider admission to the full verifier.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import quote, urljoin

from published_artifact import ENVIRONMENT_HOSTS, basename, digest, download, valid_branch


METADATA = ('metadata.json', 'metadata-minimal.json', 'metadata-all.json',
            'metadata-pyneat.json', 'metadata-extras.json')


def acquire_core(cli, branch, commit, environment, output, core):
    environment = {'prod': 'production', 'prd': 'production', 'stg': 'staging'}.get(
        environment, environment)
    if (environment not in ENVIRONMENT_HOSTS or not re.fullmatch(r'[0-9a-f]{40}', commit)
            or not valid_branch(branch)):
        raise ValueError('Invalid immutable Core publication selector')
    revision = commit[:12]
    selector = f'core@{branch}:{revision}'
    resolved = subprocess.check_output(
        [str(cli), 'neat', 'download', selector, '--env', environment, '-t', 'all', '--json'],
        text=True)
    selected = json.loads(resolved[resolved.index('{'):])
    origin = 'https://' + ENVIRONMENT_HOSTS[environment]
    # Validate the CLI's canonical URL; never use a guessed alternative origin,
    # mutable latest selector, or another repository's metadata as authority.
    expected_url = f'{origin}/core/{quote(quote(branch, safe=""), safe="")}/{revision}/metadata-all.json'
    if (selected.get('repository') != 'core' or selected.get('package_path', '') != ''
            or selected.get('ref') != branch or selected.get('environment') != environment
            or selected.get('requested_spec') != revision or selected.get('resolved_spec') != revision
            or selected.get('base_url') != origin or selected.get('metadata_url') != expected_url):
        raise ValueError('sima-cli resolved a different Core publication')

    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {}
    for name in METADATA:
        download(urljoin(selected['metadata_url'], name), output / name)
        metadata[name] = json.loads((output / name).read_text())
    identities = {(item.get('name'), item.get('version')) for item in metadata.values()}
    if (len(identities) != 1 or next(iter(identities))[0] not in ('sima-neat', 'gh:sima-neat/core')
            or not isinstance(next(iter(identities))[1], str) or not next(iter(identities))[1]):
        raise ValueError('Core metadata identity differs from selected publication')
    complete = metadata['metadata-all.json']
    resources = complete.get('resources')
    checksums = complete.get('resources-checksum')
    if (not isinstance(resources, list) or not resources
            or any(not isinstance(name, str) for name in resources)
            or len(resources) != len(set(resources)) or not isinstance(checksums, dict)
            or set(checksums) != set(resources)):
        raise ValueError('Invalid Core publication resource inventory')
    for name in resources:
        basename(name)
        if name in METADATA or not re.fullmatch(r'[0-9a-f]{64}', str(checksums[name])):
            raise ValueError('Invalid Core publication resource checksum/name')
    metadata_hashes = {name: digest(output / name) for name in METADATA}
    subprocess.run([str(cli), 'neat', 'download', selector, '--env', environment,
                    '-t', 'all', '-d', str(output)], check=True)
    for name, expected in {**checksums, **metadata_hashes}.items():
        path = output / name
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise ValueError('Core download differs from selected metadata: ' + name)
    # The selected CLI schema, all five variant contracts, actual DEB identities,
    # and authenticated provider receipts remain mandatory and unchanged.
    subprocess.run(['bash', str(Path(core) / 'scripts/ci/verify_core_dist.sh'),
                    '--dist', str(output), '--expected-manifest', str(Path(core) / 'deps/manifest.json')],
                   check=True, env={**os.environ, 'SIMA_CLI_BIN': str(cli)})
    return {'status': 'PASS', 'selector': selector, 'environment': environment,
            'resources': len(resources), 'directory': str(output)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cli', default=os.environ.get('SIMA_CLI_BIN', 'sima-cli'))
    parser.add_argument('--branch', required=True)
    parser.add_argument('--commit', required=True)
    parser.add_argument('--environment', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(acquire_core(args.cli, args.branch, args.commit, args.environment,
                                 args.output, Path(__file__).resolve().parents[2])))


if __name__ == '__main__':
    main()
