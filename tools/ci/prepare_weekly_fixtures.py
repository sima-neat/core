#!/usr/bin/env python3
"""Stage explicitly pinned weekly fixtures before native admission.

Inputs are local paths or HTTPS locations plus SHA256 in SIMA_CI_{NAME} and
SIMA_CI_{NAME}_SHA256. No model alias resolution or fallback download is allowed.
Only the MPK manifest inside model archives is read; no other model JSON is used.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import tarfile
import urllib.parse
import urllib.request


def stage(name, destination, env, *, model=False):
    source = env.get('SIMA_CI_' + name, '')
    expected = env.get('SIMA_CI_' + name + '_SHA256', '')
    if not source or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError(f'{name}: required explicit source and reviewed SHA256 are missing')
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError(f'{name}: refuse an existing fixture destination')
    parsed = urllib.parse.urlparse(source)
    if parsed.scheme:
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError(f'{name}: use a local path or credential-free HTTPS URL')
        stream = urllib.request.urlopen(source, timeout=60)
    else:
        stream = Path(source).open('rb')
    digest = hashlib.sha256()
    with stream, destination.open('xb') as output:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
            output.write(chunk)
    if digest.hexdigest() != expected:
        raise ValueError(f'{name}: fixture checksum mismatch (failed bytes preserved)')
    if model:
        with tarfile.open(destination) as archive:
            manifests = [m for m in archive.getmembers() if m.isfile() and
                         (Path(m.name).name == 'mpk.json' or Path(m.name).name.endswith('_mpk.json'))]
            if len(manifests) != 1:
                raise ValueError(f'{name}: require one authoritative MPK manifest')
            with archive.extractfile(manifests[0]) as manifest:
                if not isinstance(json.load(manifest), dict):
                    raise ValueError(f'{name}: invalid MPK manifest')
    return destination.resolve()


def require_streams(env, profile='all'):
    if profile not in ('none', 'http', 'all'):
        raise ValueError('Unknown stream profile')
    if profile == 'none':
        return
    required = [('HTTP', 'MJPEG')]
    if profile == 'all':
        required += [('RTSP', codec) for codec in ('H264', 'H265', 'MJPEG')]
    missing = []
    for protocol, codec in required:
        prefix = f'SIMANEAT_TEST_{protocol}_{codec}'
        if not (env.get(prefix + '_URL') or env.get(prefix + '_URLS')):
            missing.append(prefix + '_URL(S)')
        if codec == 'MJPEG' and not env.get(prefix + '_FPS', '').isdigit():
            missing.append(prefix + '_FPS')
        elif codec == 'MJPEG' and int(env[prefix + '_FPS']) <= 0:
            missing.append(prefix + '_FPS')
    if missing:
        raise ValueError('Required stream fixtures missing: ' + ', '.join(missing))


FIXTURE_GROUPS = ('mnist', 'ssd300')


def selected_groups(groups):
    groups = tuple(groups)
    if not groups or len(groups) != len(set(groups)) or any(g not in FIXTURE_GROUPS for g in groups):
        raise ValueError('Select each fixture group at most once: mnist, ssd300')
    return groups


def validate_mnist(path, *, images):
    """Check the same IDX contract consumed by the native accuracy test."""
    with Path(path).open('rb') as stream:
        header_size = 16 if images else 8
        header = stream.read(header_size)
        if len(header) != header_size:
            raise ValueError('Truncated MNIST IDX header')
        fields = struct.unpack('>' + 'I' * (header_size // 4), header)
        magic, count = fields[:2]
        if magic != (2051 if images else 2049) or count < 100:
            raise ValueError('Invalid MNIST IDX magic/count (100 samples required)')
        if images and fields[2:] != (28, 28):
            raise ValueError('MNIST images must be 28x28')
        expected_size = header_size + count * (28 * 28 if images else 1)
        if Path(path).stat().st_size != expected_size:
            raise ValueError('MNIST IDX payload length differs from header')
        if not images and any(label > 9 for label in stream.read()):
            raise ValueError('Invalid MNIST label')
    return count


def prepare(output, runtime_manifest, env, *, groups=FIXTURE_GROUPS):
    groups = selected_groups(groups)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    values = {}
    if 'ssd300' in groups:
        values['SIMA_SSD300_TAR'] = str(stage('SSD300_MODEL', output / 'ssd300_heads_mpk.tar.gz', env, model=True))
    if 'mnist' not in groups:
        return values
    values['SIMA_MNIST_MODEL_TAR'] = str(stage('MNIST_MODEL', output / 'mnist.tar.gz', env, model=True))
    runtime_manifest = Path(runtime_manifest)
    manifest = json.loads(runtime_manifest.read_text())
    source_root = (runtime_manifest.parent.parent / manifest['source_root_rel']).resolve()
    cache = source_root / 'tmp/mnist_preproc_channels1'
    files = []
    counts = []
    for name, filename in (('MNIST_IMAGES', 't10k-images-idx3-ubyte'), ('MNIST_LABELS', 't10k-labels-idx1-ubyte')):
        staged = stage(name, output / (filename + '.gz'), env)
        # Read the entire gzip, including CRC, before modifying the shared cache.
        raw = output / filename
        with gzip.open(staged, 'rb') as stream, raw.open('xb') as destination:
            shutil.copyfileobj(stream, destination)
        counts.append(validate_mnist(raw, images=name == 'MNIST_IMAGES'))
        files.extend((staged, raw))
    if counts[0] != counts[1]:
        raise ValueError('MNIST image/label counts differ')
    # Validate every pre-existing entry first. Never let an old raw file override
    # the reviewed compressed bytes, or replace another campaign's fixture.
    for source in files:
        target = cache / source.name
        if target.is_symlink():
            raise ValueError(f'{source.name}: refuse a symlink in the fixture cache')
        if target.exists() and hashlib.sha256(target.read_bytes()).digest() != hashlib.sha256(source.read_bytes()).digest():
            raise ValueError(f'{source.name}: existing cache differs; do not replace another campaign fixture')
    cache.mkdir(parents=True, exist_ok=True)
    for source in files:
        target = cache / source.name
        if not target.exists():
            with source.open('rb') as stream, target.open('xb') as destination:
                shutil.copyfileobj(stream, destination)
    values['SIMA_RUN_MNIST_PREPROC_CHANNELS1'] = '1'
    return values


def export_environment(values, *, github_env=None, environment_json=None):
    """Publish one validated mapping; a local launcher must merge it into env."""
    if any(not isinstance(value, str) or any(c in value for c in ('\n', '\r', '\0'))
           for value in values.values()):
        raise ValueError('Invalid fixture environment value')
    # Validate all values before emitting even the first opt-in. JSON is an
    # environment mapping, not a qualification result or shell program.
    if environment_json:
        with Path(environment_json).open('x') as stream:
            json.dump(values, stream, indent=2, sort_keys=True)
            stream.write('\n')
    if github_env:
        with Path(github_env).open('a') as stream:
            for key, value in sorted(values.items()):
                stream.write(f'{key}={value}\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--runtime-manifest', type=Path)
    parser.add_argument('--github-env', type=Path)
    parser.add_argument('--environment-json', type=Path, help='Write the same validated environment mapping for a local launcher')
    parser.add_argument('--fixture-group', choices=FIXTURE_GROUPS, action='append',
                        help='Explicitly select groups; default is all (unselected groups remain unqualified)')
    parser.add_argument('--stream-profile', choices=('none', 'http', 'all'))
    parser.add_argument('--stream-report', type=Path)
    args = parser.parse_args()
    try:
        if args.stream_profile:
            require_streams(os.environ, args.stream_profile)
            gaps = {'none': ['HTTP MJPEG', 'RTSP H264/H265/MJPEG'], 'http': ['RTSP H264/H265/MJPEG'], 'all': []}[args.stream_profile]
            report = {'profile': args.stream_profile, 'scope': 'codec_stream registrations only',
                      'other_rtsp_workflows_qualified': False, 'excluded_not_qualified': gaps,
                      'prepared_not_executed': args.stream_profile != 'none'}
            if args.stream_report:
                args.stream_report.parent.mkdir(parents=True, exist_ok=True)
                with args.stream_report.open('x') as stream:
                    json.dump(report, stream, indent=2)
            print(json.dumps(report))
            return 0
        if not args.output or not args.runtime_manifest or not (args.github_env or args.environment_json):
            parser.error('model preparation requires --output, --runtime-manifest and an environment export destination')
        groups = args.fixture_group if args.fixture_group is not None else FIXTURE_GROUPS
        values = prepare(args.output, args.runtime_manifest, os.environ, groups=groups)
        export_environment(values, github_env=args.github_env, environment_json=args.environment_json)
        print(json.dumps({'selected_fixture_groups': groups,
                          'excluded_not_qualified': sorted(set(FIXTURE_GROUPS) - set(groups))}))
    except (ValueError, OSError, tarfile.TarError) as exc:
        print(f'PREPARATION_BLOCKED: {exc}', file=sys.stderr)
        return 1
    print('Weekly immutable fixtures staged; runtime assertions remain to be qualified.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
