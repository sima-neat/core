#!/usr/bin/env python3
"""Prepare manifest-only test fixtures from checksum-pinned Model Registry packages.

Usage: prepare_registry_mpk_fixtures.py --fixtures <pins.json> --root <output>
       --cache <archive-cache> [--offline] [--cli sima-cli]

Only fixture preparation needs the CLI/network. Native tests consume the generated
manifests offline; neither model executables nor sidecar JSON are extracted.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile
import tempfile
from uuid import UUID


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixtures(path):
    document = json.loads(Path(path).read_text())
    if not isinstance(document, dict):
        raise ValueError('Model Registry fixture pins must be a JSON object')
    branch = document.get('branch', '')
    if (not isinstance(branch, str) or not branch or branch.startswith('-')
            or '..' in branch or any(c.isspace() or c in '~^:?*[\\' for c in branch)):
        raise ValueError('Invalid Model Registry fixture branch')
    if type(document.get('staging')) is not bool:
        raise ValueError('Select the Model Registry environment explicitly')
    fixtures = document.get('fixtures')
    if not isinstance(fixtures, list) or not fixtures:
        raise ValueError('At least one pinned model fixture is required')
    names = set()
    for fixture in fixtures:
        if not isinstance(fixture, dict):
            raise ValueError('Model fixture pins must be JSON objects')
        for key in ('model_id', 'variant'):
            if (not isinstance(fixture.get(key), str)
                    or not re.fullmatch('[a-z][a-z0-9_]*', fixture[key])):
                raise ValueError('Invalid model fixture identity')
        name = fixture['model_id'] + '-' + fixture['variant']
        if name in names:
            raise ValueError('Duplicate model fixture: ' + name)
        names.add(name)
        if (not isinstance(fixture.get('sha256'), str)
                or not re.fullmatch('[a-f0-9]{64}', fixture['sha256'])
                or type(fixture.get('size_bytes')) is not int or fixture['size_bytes'] <= 0):
            raise ValueError(name + ': require a reviewed archive SHA256 and positive size')
        if (not isinstance(fixture.get('run_id'), str)
                or str(UUID(fixture['run_id'])) != fixture['run_id']):
            raise ValueError(name + ': invalid Model Registry package run ID')
    return document


@contextmanager
def locked(path):
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def verify_archive(path, fixture):
    if (path.is_symlink() or not path.is_file()
            or path.stat().st_size != fixture['size_bytes'] or sha256(path) != fixture['sha256']):
        raise ValueError(fixture['model_id'] + ': model archive differs from its reviewed pin')


def acquire(document, fixture, cache, *, cli='sima-cli', offline=False):
    """Reuse the normal CLI download and independently check its selected package."""
    archive = cache / (fixture['sha256'] + '.tar.gz')
    with locked(cache / (fixture['sha256'] + '.lock')):
        if archive.exists() or archive.is_symlink():
            verify_archive(archive, fixture)
            return archive
        if offline:
            raise ValueError(fixture['model_id'] + ': pinned archive missing from offline cache')
        with tempfile.TemporaryDirectory(prefix='registry-', dir=cache) as temporary:
            command = [str(cli), 'models', 'download', '--branch', document['branch'],
                       '--id', fixture['model_id'], '--variant', fixture['variant'],
                       '--output', temporary, '--json']
            if document['staging']:
                command.append('--stg')
            cli_env = {**os.environ, 'SIMA_CLI_CHECK_FOR_UPDATE': '0'}
            cli_env.pop('SIMA_MODELS_REGISTRY_BASE_URL', None)
            result = subprocess.run(command, capture_output=True, text=True, timeout=600,
                                    env=cli_env)
            if result.returncode:
                # CLI errors may contain expiring download URLs. Keep them out of
                # test reports; the model identity and command are reproducible.
                raise ValueError(fixture['model_id'] + ': Model Registry download failed')
            selected = json.loads(result.stdout)
            if not isinstance(selected, dict):
                raise ValueError('Model Registry download receipt must be a JSON object')
            if any(selected.get(key) != fixture[key] for key in
                   ('model_id', 'run_id', 'sha256', 'size_bytes')):
                raise ValueError(fixture['model_id'] + ': Model Registry selected a different package')
            if selected.get('variant_id') != fixture['variant']:
                raise ValueError(fixture['model_id'] + ': Model Registry selected a different variant')
            downloaded = Path(selected['path'])
            if not downloaded.is_absolute():
                downloaded = Path.cwd() / downloaded
            if not downloaded.resolve().is_relative_to(Path(temporary).resolve()):
                raise ValueError('Model Registry download left its fixture staging directory')
            verify_archive(downloaded, fixture)
            shutil.move(str(downloaded), archive)
    return archive


def read_manifest(archive):
    """Read one authoritative regular MPK member, never an arbitrary sidecar."""
    with tarfile.open(archive, 'r:gz') as stream:
        candidates = [member for member in stream if
                      PurePosixPath(member.name).name == 'mpk.json'
                      or PurePosixPath(member.name).name.endswith('_mpk.json')]
        if len(candidates) != 1:
            raise ValueError('Model archive must contain exactly one authoritative MPK manifest')
        member = candidates[0]
        path = PurePosixPath(member.name)
        if (not member.isfile() or path.is_absolute() or '..' in path.parts
                or '\\' in member.name or not 0 < member.size <= 8 * 1024 * 1024):
            raise ValueError('Unsafe or oversized MPK manifest member')
        with stream.extractfile(member) as source:
            content = source.read()
        if not isinstance(json.loads(content), dict):
            raise ValueError('The authoritative MPK manifest must be a JSON object')
        return member.name, content


def write_atomic(path, content):
    if path.is_symlink():
        raise ValueError('Refuse a symlink in generated model fixtures')
    if path.is_file() and path.read_bytes() == content:
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.close()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def prepare(fixtures_path, root, cache, *, cli='sima-cli', offline=False):
    document = load_fixtures(fixtures_path)
    root, cache = Path(root), Path(cache)
    root.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    receipts = []
    # Serialize publication, too: an installed extras package must never pick up
    # half of one preparation attempt and half of a concurrent attempt.
    with locked(root / '.prepare.lock'):
        expected = {f['model_id'] + '-' + f['variant'] + '-mpk.json'
                    for f in document['fixtures']}
        if any(path.name not in expected for path in root.glob('*-mpk.json')):
            raise ValueError('Stale model fixtures exist; use a fresh preparation directory')
        prepared = []
        for fixture in document['fixtures']:
            archive = acquire(document, fixture, cache, cli=cli, offline=offline)
            member, content = read_manifest(archive)
            name = fixture['model_id'] + '-' + fixture['variant'] + '-mpk.json'
            prepared.append((root / name, content))
            receipts.append({**fixture, 'path': name, 'manifest_member': member,
                             'manifest_sha256': hashlib.sha256(content).hexdigest(),
                             'manifest_size_bytes': len(content)})
        # No new fixture is published unless every selected archive is valid.
        for path, content in prepared:
            write_atomic(path, content)
        receipt = {'branch': document['branch'], 'staging': document['staging'],
                   'fixtures': receipts}
        write_atomic(root / 'provenance.json', (json.dumps(receipt, indent=2) + '\n').encode())
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--cli', default='sima-cli')
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    prepare(args.fixtures, args.root, args.cache, cli=args.cli, offline=args.offline)


if __name__ == '__main__':
    main()
