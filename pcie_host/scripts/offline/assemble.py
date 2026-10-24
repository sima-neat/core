#!/usr/bin/env python3
"""Assemble host/card offline repositories in immutable, clean OCI baselines."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(*args):
    subprocess.run(args, check=True, timeout=3600)


def verify_publication(directory, expected_hash):
    metadata = directory / 'metadata.json'
    if metadata.is_symlink() or sha(metadata) != expected_hash:
        raise ValueError('publication metadata checksum mismatch')
    data = json.loads(metadata.read_text())
    names = data['resources']
    hashes = data['resources-checksum']
    if not names or len(set(names)) != len(names) or set(names) != set(hashes):
        raise ValueError('incomplete publication inventory')
    for name in names:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+~-]*', name):
            raise ValueError('unsafe publication resource')
        path = directory / name
        if path.is_symlink() or not path.is_file() or sha(path) != hashes[name]:
            raise ValueError('publication resource checksum mismatch: ' + name)
    if not any(n.endswith('.deb') for n in names) or not any(n.endswith('.whl') for n in names):
        raise ValueError('native packages and Python wheel required')
    return data


def image(value):
    if not re.fullmatch(r'[A-Za-z0-9./:_-]+@sha256:[0-9a-f]{64}', value):
        raise ValueError('baseline image must be pinned by sha256 digest')
    return value


def assemble(spec, output):
    if set(spec['targets']) != {'host', 'card'}:
        raise ValueError('both host and card targets are required')
    if output.exists() or Path(str(output)+'.tar.gz').exists():
        raise ValueError('output must not exist; incomplete builds must not be reused')
    # Validate both inputs before starting any container or creating the bundle.
    publications = {}
    for target, row in spec['targets'].items():
        image(row['image'])
        expected = 'amd64' if target == 'host' else 'arm64'
        if row['architecture'] != expected:
            raise ValueError('incorrect target architecture')
        if target == 'host' and (row['os_id'], row['os_version']) != ('ubuntu', '24.04'):
            raise ValueError('initial host support is Ubuntu 24.04 amd64')
        publications[target] = verify_publication(Path(row['publication']), row['metadata_sha256'])
    # Both publications must describe the same Core revision, not merely version 0.6.0.
    revisions = [row['core_revision'] for row in spec['targets'].values()]
    if len(set(revisions)) != 1 or not re.fullmatch(r'[0-9a-f]{12,40}', revisions[0]):
        raise ValueError('matching immutable Core revisions are required')
    for target, data in publications.items():
        row = spec['targets'][target]
        identity = str(data.get('commit', '')) + ' ' + str(data.get('version', ''))
        if row['core_revision'] not in identity:
            raise ValueError('Core revision is absent from publication identity')
    output.mkdir(parents=True)
    for target in ('host', 'card'):
        row = spec['targets'][target]
        dest = output / target
        payload = dest / 'payload'
        payload.mkdir(parents=True)
        for name in publications[target]['resources'] + ['metadata.json']:
            shutil.copyfile(Path(row['publication']) / name, payload / name)
        shutil.copyfile(HERE / 'target.py', dest / 'target.py')
        # No host paths or credentials are saved in the distributable manifest.
        (dest / 'target.json').write_text(json.dumps({k: row[k] for k in (
            'image', 'architecture', 'os_id', 'os_version', 'core_revision', 'metadata_sha256')}, indent=2)+'\n')
        run('docker', 'run', '--rm', '--user', '0:0', '--env', 'DEBIAN_FRONTEND=noninteractive', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--platform', 'linux/' + row['architecture'],
            '--mount', f'type=bind,src={dest.resolve()},dst=/bundle',
            '--entrypoint', '/bin/sh', row['image'], '-ec',
            'cp /var/lib/dpkg/status /bundle/baseline.status; '
            'apt-get update; apt-get install -y --no-install-recommends python3 python3-venv python3-pip dpkg-dev; '
            'python3 /bundle/target.py collect')
    files = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'bundle.json').write_text(json.dumps({
        'schema_version': 1, 'core_revision': revisions[0], 'files': files,
        'scope': 'offline package repositories; hardware orchestration not included',
    }, indent=2)+'\n')
    # No success archive is emitted until both network-isolated installs pass.
    validate(output)
    archive_bundle(output)


def archive_bundle(root):
    """Stable archive bytes for an identical locked bundle."""
    with open(str(root)+'.tar.gz', 'wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for path in sorted(root.rglob('*')):
                    info = archive.gettarinfo(path, arcname=str(Path('neat-offline') / path.relative_to(root)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    info.mode = 0o755 if path.is_dir() else 0o644
                    if path.is_file():
                        with path.open('rb') as stream:
                            archive.addfile(info, stream)
                    else:
                        archive.addfile(info)


def verify_bundle(root):
    data = json.loads((root / 'bundle.json').read_text())
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('bundle symlinks are forbidden')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} - {'bundle.json'}
    if actual != set(data['files']):
        raise ValueError('bundle file inventory differs')
    for name, digest in data['files'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe bundle path')
        path = root / name
        if path.is_symlink() or sha(path) != digest:
            raise ValueError('bundle checksum mismatch: ' + name)
    return data


def validate(root):
    verify_bundle(root)
    for target in ('host', 'card'):
        dest = root / target
        row = json.loads((dest / 'target.json').read_text())
        image(row['image'])
        run('docker', 'run', '--rm', '--user', '0:0', '--env', 'DEBIAN_FRONTEND=noninteractive', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--network=none', '--platform', 'linux/' + row['architecture'],
            '--mount', f'type=bind,src={dest.resolve()},dst=/bundle,readonly',
            '--entrypoint', '/bin/sh', row['image'], '-ec',
            # Bootstrap Python from the local repo; the baseline need not contain it.
            'cmp /var/lib/dpkg/status /bundle/baseline.status; '
            'cp -a /bundle /tmp/offline; cd /tmp/offline; '
            'sh ./bootstrap.sh; python3 ./target.py validate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    build = sub.add_parser('assemble')
    build.add_argument('--spec', type=Path, required=True)
    build.add_argument('--output', type=Path, required=True)
    check = sub.add_parser('validate')
    check.add_argument('bundle', type=Path)
    args = parser.parse_args()
    if args.command == 'assemble':
        assemble(json.loads(args.spec.read_text()), args.output.resolve())
    else:
        validate(args.bundle.resolve())


if __name__ == '__main__':
    main()
