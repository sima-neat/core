#!/usr/bin/env python3
"""Build offline variants from authenticated, immutable Vulcan publications."""
import argparse
import base64
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import quote

import assemble

HERE = Path(__file__).resolve().parent
TUPLES = [('ubuntu22', 'amd64'), ('ubuntu22', 'arm64'), ('ubuntu24', 'amd64'), ('ubuntu24', 'arm64')]


def run(*args):
    return subprocess.check_output(list(map(str, args)), text=True).strip()


def prefix(ref, sha, distro=None, arch=None):
    if not re.fullmatch(r'[0-9a-f]{40}', sha) or not ref or any(c.isspace() for c in ref):
        raise ValueError('invalid immutable publication identity')
    value = f'core/{quote(ref, safe="")}/{sha[:12]}'
    if distro is not None:
        if (distro, arch) not in TUPLES:
            raise ValueError('unsupported host tuple')
        value += f'/pciehost/{distro}/{arch}'
    return value


def download(bucket, key, path, optional=False):
    result = subprocess.run(['aws', 's3api', 'get-object', '--bucket', bucket,
                             '--key', key, str(path)], capture_output=True, text=True)
    if result.returncode:
        if optional and re.search(r'\((?:NoSuchKey|404)\)', result.stderr):
            return False
        raise RuntimeError('Cannot retrieve '+key+': '+result.stderr)
    return True


def metadata_identity(data, sha, ref, distro=None, arch=None):
    # Native host metadata carries the full identity. Shared card metadata uses
    # a version suffix; its authenticated S3 key binds the source ref/commit.
    if distro:
        matches = data.get('commit') == sha and data.get('branch') == ref
    else:
        matches = (str(data.get('version', '')).endswith('.' + sha[:12])
                   and data.get('commit', sha) == sha and data.get('branch', ref) == ref)
    if not matches:
        raise ValueError('publication is not from the selected CI commit/ref')
    if distro and (data.get('artifact', {}).get('distro') != distro
                   or data.get('artifact', {}).get('arches') != [arch]):
        raise ValueError('native publication tuple differs')
    resources, hashes = data.get('resources', []), data.get('resources-checksum', {})
    if not resources or len(resources) != len(set(resources)) or set(resources) != set(hashes):
        raise ValueError('publication inventory differs')
    for name in resources:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+~-]*', name) or not re.fullmatch(r'[0-9a-f]{64}', hashes[name]):
            raise ValueError('unsafe publication resource or checksum')


def offline_identity(data, sha, ref, distro, arch, inputs):
    metadata_identity(data, sha, ref, distro, arch)
    if (data['resources'] != [assemble.ARCHIVE_NAME]
            or data.get('installation', {}).get('script') != ''
            or data.get('artifact', {}).get('type') != 'offline'
            or data['artifact'].get('package_path') != f'pciehost/{distro}/{arch}'
            or data.get('source_metadata_sha256') != inputs):
        raise ValueError('existing offline publication differs; refusing replacement')


def verify_uploaded(bucket, key, digest):
    data = json.loads(run('aws', 's3api', 'head-object', '--bucket', bucket,
                          '--key', key, '--checksum-mode', 'ENABLED'))
    expected = base64.b64encode(bytes.fromhex(digest)).decode()
    if data.get('ChecksumSHA256') != expected:
        raise ValueError('uploaded object checksum differs: '+key)


def put(bucket, key, path, kms, immutable=False):
    digest = assemble.sha(path)
    command = ['aws', 's3api', 'put-object', '--bucket', bucket, '--key', key,
               '--body', str(path), '--server-side-encryption', 'aws:kms',
               '--checksum-algorithm', 'SHA256', '--checksum-sha256',
               base64.b64encode(bytes.fromhex(digest)).decode(),
               '--content-type', 'application/json' if path.suffix == '.json' else 'application/gzip']
    if kms:
        command += ['--ssekms-key-id', kms]
    if immutable:
        command += ['--if-none-match', '*']
    run(*command)
    verify_uploaded(bucket, key, digest)


def source_metadata(args, work):
    records, paths, inputs = {}, {}, {}
    for role in ('host', 'card'):
        directory = work/(role+'-publication')
        directory.mkdir()
        key = prefix(args.ref, args.sha, args.distro, args.arch) if role == 'host' else prefix(args.ref, args.sha)
        path = directory/'metadata.json'
        download(args.bucket, key+'/metadata.json', path)
        data = json.loads(path.read_text())
        metadata_identity(data, args.sha, args.ref, args.distro if role == 'host' else None, args.arch)
        records[role], paths[role], inputs[role] = data, (key, directory), assemble.sha(path)
    return records, paths, inputs


def existing(args, work, inputs, pending=True):
    base = prefix(args.ref, args.sha, args.distro, args.arch)
    for suffix in ('', '.pending') if pending else ('',):
        path = work/('existing'+suffix+'.json')
        if download(args.bucket, base+'/'+assemble.METADATA_NAME+suffix, path, optional=True):
            data = json.loads(path.read_text())
            offline_identity(data, args.sha, args.ref, args.distro, args.arch, inputs)
            verify_uploaded(args.bucket, base+'/'+assemble.ARCHIVE_NAME,
                            data['resources-checksum'][assemble.ARCHIVE_NAME])
            print('Already validated offline publication: '+base+suffix)
            return path
    return None


def build(args):
    args.work.mkdir(parents=True, exist_ok=False)
    records, paths, inputs = source_metadata(args, args.work)
    if existing(args, args.work, inputs):
        return
    for role, (key, directory) in paths.items():
        for name, digest in records[role]['resources-checksum'].items():
            download(args.bucket, key+'/'+name, directory/name)
            if assemble.sha(directory/name) != digest:
                raise ValueError('publication resource checksum differs: '+name)
        assemble.verify_publication(directory, inputs[role])
    host = paths['host'][1]
    if not (host/'platform_provider.py').is_file() or any(host.glob('simaai-pcie-host*.deb')):
        raise ValueError('Rebuild Core with the corrected Scheduler native producer; Platform provisioning must remain separate.')
    baselines = json.loads((HERE/'baselines.json').read_text())
    context = args.work/'host-baseline'
    context.mkdir()
    for name in ('platform_provider.py', 'platform-source.json', 'platform-source.tar.gz'):
        shutil.copyfile(host/name, context/name)
    shutil.copyfile(HERE/'HostBaseline.Dockerfile', context/'Dockerfile')
    iid = args.work/'host-image-id'
    run('docker', 'build', '--platform', 'linux/'+args.arch, '--iidfile', iid,
        '--build-arg', 'UBUNTU_IMAGE='+baselines['ubuntu'][args.distro], context)
    host_image = assemble.image(iid.read_text().strip())
    card = baselines['card']
    image_archive = args.work/'card-baseline.docker.tar.gz'
    download(args.bucket, card['key'], image_archive)
    if assemble.sha(image_archive) != card['sha256']:
        raise ValueError('card baseline image checksum differs')
    run('docker', 'load', '--input', image_archive)
    if run('docker', 'image', 'inspect', card['image_id'], '--format', '{{.Id}}') != card['image_id']:
        raise ValueError('card baseline image identity differs')
    image_archive.unlink()
    spec = {'installer_revision': args.installer_revision, 'targets': {}}
    for role, (_, directory) in paths.items():
        spec['targets'][role] = {
            'publication': str(directory.resolve()), 'metadata_sha256': inputs[role],
            'core_revision': args.sha[:12],
            'image': host_image if role == 'host' else card['image_id'],
            'architecture': args.arch if role == 'host' else 'arm64',
            'os_id': 'ubuntu' if role == 'host' else card['os_id'],
            'os_version': {'ubuntu22': '22.04', 'ubuntu24': '24.04'}[args.distro] if role == 'host' else card['os_version'],
        }
    (args.work/'spec.json').write_text(json.dumps(spec, indent=2)+'\n')
    root = args.work/'neat-offline'
    assemble.assemble(spec, root)
    path = args.work/assemble.METADATA_NAME
    metadata = json.loads(path.read_text())
    metadata.update(source_metadata_sha256=inputs, installer_revision=args.installer_revision,
                    source_run_id=args.run_id)
    path.write_text(json.dumps(metadata, indent=2)+'\n')
    base = prefix(args.ref, args.sha, args.distro, args.arch)
    # Metadata is staged only after clean-container network-disabled validation.
    # The final metadata is exposed by the aggregate publish job, after all four
    # tuples pass. A failed pre-metadata upload can be safely retried under the
    # workflow's per-source concurrency group; completed publications are immutable.
    put(args.bucket, base+'/'+assemble.ARCHIVE_NAME, args.work/assemble.ARCHIVE_NAME, args.kms)
    put(args.bucket, base+'/'+assemble.METADATA_NAME+'.pending', path, args.kms)


def publish(args):
    args.work.mkdir(parents=True, exist_ok=False)
    ready = []
    for distro, arch in TUPLES:
        args.distro, args.arch = distro, arch
        work = args.work/(distro+'-'+arch)
        work.mkdir()
        _, _, inputs = source_metadata(args, work)
        path = existing(args, work, inputs)
        if path is None:
            raise ValueError('missing validated offline tuple: '+distro+'/'+arch)
        ready.append((prefix(args.ref, args.sha, distro, arch), path))
    # Complete every read/check before publishing any new metadata.
    for base, path in ready:
        if path.name == 'existing.json':
            continue
        put(args.bucket, base+'/'+assemble.METADATA_NAME, path, args.kms, immutable=True)
    if args.distribution:
        run('aws', 'cloudfront', 'create-invalidation', '--distribution-id', args.distribution,
            '--paths', *['/'+base+'/'+assemble.METADATA_NAME for base, _ in ready])
    print('Published all four offline variants.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['build', 'publish'])
    p.add_argument('--bucket', required=True)
    p.add_argument('--ref', required=True)
    p.add_argument('--sha', required=True)
    p.add_argument('--work', required=True, type=Path)
    p.add_argument('--distro', choices=['ubuntu22', 'ubuntu24'])
    p.add_argument('--arch', choices=['amd64', 'arm64'])
    p.add_argument('--installer-revision', default=os.environ.get('GITHUB_SHA', ''))
    p.add_argument('--run-id', type=int)
    p.add_argument('--kms', default='')
    p.add_argument('--distribution', default='')
    args = p.parse_args()
    prefix(args.ref, args.sha)
    if args.command == 'build':
        if (args.distro, args.arch) not in TUPLES or not re.fullmatch(r'[0-9a-f]{40}', args.installer_revision):
            p.error('build requires a host tuple and immutable installer revision')
        build(args)
    else:
        publish(args)


if __name__ == '__main__':
    main()
