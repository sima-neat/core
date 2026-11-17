#!/usr/bin/env python3
"""Transport explicitly reviewed immutable CI inputs, never guess latest.

Installed receipts are still validated by build_dependencies.py; this is not a
second build graph. Codecs is an independently selected installed provider.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import tarfile
import urllib.parse
import urllib.request

COMPONENTS = ('kernels', 'runner', 'scheduler', 'codecs')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_manifest(data):
    if data.get('schema_version') != 1 or set(data.get('inputs', {})) != set(COMPONENTS):
        raise ValueError('Explicit Kernels/Runner/Scheduler/Codecs cohort required')
    for component, item in data['inputs'].items():
        if item.get('kind') == 'git' and component != 'codecs':
            if not re.fullmatch(r'[0-9a-f]{40}', item.get('commit', '')):
                raise ValueError('Source commits must be immutable full Git SHAs')
            url = item.get('repository', '')
        elif item.get('kind') == 'package-bundle':
            if not re.fullmatch(r'[0-9a-f]{64}', item.get('sha256', '')):
                raise ValueError('Package bundles require reviewed SHA256')
            url = item.get('url', '')
        else:
            raise ValueError('Codecs requires an external installed bundle; unsupported input kind')
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Only explicit credential-free HTTPS locations are allowed')
    return data


def extract_bundle(archive, destination):
    destination = Path(destination)
    if destination.exists():
        raise ValueError('Refuse stale dependency materialization')
    destination.mkdir()
    with tarfile.open(archive) as bundle:
        names = set()
        for entry in bundle.getmembers():
            path = PurePosixPath(entry.name)
            if path.is_absolute() or '..' in path.parts or entry.name in names:
                raise ValueError('Unsafe/duplicate package bundle path')
            names.add(entry.name)
            if not (entry.isfile() or entry.isdir() or entry.issym()):
                raise ValueError('Unsupported package bundle member')
            if entry.issym():
                target = PurePosixPath(entry.linkname)
                if target.is_absolute():
                    raise ValueError('Absolute package bundle symlink')
                parts = list(path.parent.parts)
                for part in target.parts:
                    if part == '..':
                        if not parts: raise ValueError('Escaping package bundle link')
                        parts.pop()
                    elif part not in ('.', ''): parts.append(part)
        links = {PurePosixPath(entry.name) for entry in bundle.getmembers() if entry.issym()}
        if any(any(parent in links for parent in PurePosixPath(entry.name).parents)
               for entry in bundle.getmembers()):
            raise ValueError('Package bundle member traverses symlink')
        for entry in bundle.getmembers():
            if entry.mode & 0o7000:
                raise ValueError('Privileged package bundle mode')
            bundle.extract(entry, destination, filter='data')
            if entry.isfile():
                (destination / entry.name).chmod(entry.mode & 0o777)


def materialize(manifest, output_root):
    data = validate_manifest(json.loads(Path(manifest).read_text()))
    output_root = Path(output_root)
    if output_root.exists():
        raise ValueError('Use a fresh dependency transport directory')
    output_root.mkdir(parents=True)
    env = {}
    receipt = {'schema_version': 1, 'manifest_sha256': digest(manifest), 'inputs': {}}
    for component in COMPONENTS:
        item = data['inputs'][component]
        destination = output_root / component
        if item['kind'] == 'git':
            subprocess.run(['git', 'init', str(destination)], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            subprocess.run(['git', '-C', str(destination), 'remote', 'add', 'origin', item['repository']], check=True)
            subprocess.run(['git', '-C', str(destination), 'fetch', '--depth=1', 'origin', item['commit']], check=True)
            subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', item['commit']], check=True)
            actual = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
            if actual != item['commit']:
                raise ValueError('Materialized source commit mismatch')
            if subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain'], text=True):
                raise ValueError('Materialized source is dirty')
            env['NEAT_' + component.upper() + '_SOURCE'] = str(destination.resolve())
        else:
            archive = output_root / (component + '.tar.gz')
            with urllib.request.urlopen(item['url'], timeout=60) as response, archive.open('xb') as stream:
                while chunk := response.read(1024 * 1024): stream.write(chunk)
            if digest(archive) != item['sha256']:
                raise ValueError('Downloaded bundle checksum mismatch')
            extract_bundle(archive, destination)
            env['NEAT_' + component.upper() + '_PACKAGE_DIR'] = str(destination.resolve())
        receipt['inputs'][component] = {**item, 'path': str(destination.resolve())}
    (output_root / 'materialization-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    env = materialize(args.manifest, args.output_root)
    for component in COMPONENTS:
        for suffix in ('SOURCE', 'PACKAGE_DIR'):
            print('unset NEAT_' + component.upper() + '_' + suffix)
    for key, value in env.items():
        print('export ' + key + '=' + shlex.quote(value))


if __name__ == '__main__':main()
