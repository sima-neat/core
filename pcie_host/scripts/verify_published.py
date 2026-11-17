#!/usr/bin/env python3
"""Validate the uploaded native closure before exposing its package metadata."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from pcie_delivery import sha, verify, verify_publication_inventory


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bucket', required=True); p.add_argument('--prefix', required=True)
    p.add_argument('--distro', required=True); p.add_argument('--arch', required=True)
    p.add_argument('--commit', required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    base = f's3://{args.bucket}/{args.prefix}/'
    for name in ('metadata.json', 'metadata-extras.json', 'metadata-pyneatpcie.json'):
        subprocess.run(['aws', 's3', 'cp', base + name + '.pending', str(args.output / name)], check=True)
    data = json.loads((args.output / 'metadata.json').read_text())
    if (data.get('commit') != args.commit or data.get('artifact', {}).get('distro') != args.distro
            or data.get('artifact', {}).get('arches') != [args.arch]):
        raise ValueError('uploaded native Core publication identity differs')
    resources = data['resources']; hashes = data['resources-checksum']
    if not resources or len(resources) != len(set(resources)) or set(resources) != set(hashes):
        raise ValueError('uploaded resource inventory differs')
    for name in resources:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+~-]*', name) or not re.fullmatch(r'[0-9a-f]{64}', hashes[name]):
            raise ValueError('invalid published native resource')
        subprocess.run(['aws', 's3', 'cp', base + name, str(args.output / name)], check=True)
        if sha(args.output / name) != hashes[name]: raise ValueError('uploaded resource differs: ' + name)
    selection, _ = verify(args.output)
    # Fallback or an explicit selector may choose a different Scheduler branch.
    # verify() authenticates that selection; only the host tuple must match Core.
    if selection['distro'] != args.distro or selection['architecture'] != args.arch:
        raise ValueError('native Scheduler publication does not match Core tuple')
    verify_publication_inventory(args.output, selection, resources)
    for name, pattern in [('metadata-extras.json', '*-extras.tar.gz'), ('metadata-pyneatpcie.json', 'pyneatpcie-*.whl')]:
        subset = json.loads((args.output / name).read_text())
        resource = next(args.output.glob(pattern)).name
        if (subset['commit'] != args.commit or subset['resources'] != [resource]
                or subset['resources-checksum'] != {resource: hashes[resource]} or subset['installation']['script'] != ':'):
            raise ValueError('download-only native variant changed')
    print(json.dumps({'status': 'PASS_NATIVE_PUBLISHED_CLOSURE', 'resources': len(resources), 'hardware_qualified': False}))


if __name__ == '__main__': main()
