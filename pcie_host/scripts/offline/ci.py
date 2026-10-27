#!/usr/bin/env python3
"""Build offline variants from authenticated, immutable Vulcan publications."""
import argparse
import base64
import json
import hashlib
import tarfile
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
               '--content-type', 'application/json' if key.endswith(('.json', '.json.pending')) else 'application/gzip']
    if kms:
        command += ['--ssekms-key-id', kms]
    if immutable:
        command += ['--if-none-match', '*']
    run(*command)
    verify_uploaded(bucket, key, digest)


def is_core(args):
    return getattr(args, 'core', False)


def artifact_prefix(args):
    return prefix(args.ref, args.sha) + '/offline' if is_core(args) else prefix(args.ref, args.sha, args.distro, args.arch)


def source_metadata(args, work):
    records, paths, inputs = {}, {}, {}
    for role in (('card',) if is_core(args) else ('host', 'card')):
        directory = work/(role+'-publication')
        directory.mkdir()
        key = prefix(args.ref, args.sha, args.distro, args.arch) if role == 'host' else prefix(args.ref, args.sha)
        path = directory/'metadata.json'
        download(args.bucket, key+'/metadata.json', path)
        data = json.loads(path.read_text())
        metadata_identity(data, args.sha, args.ref, args.distro if role == 'host' else None, args.arch)
        records[role], paths[role], inputs[role] = data, (key, directory), assemble.sha(path)
    return records, paths, inputs


def load_card_image(archive, expected):
    # containerd may normalize legacy Docker config JSON during import.
    # Authenticate the original config before accepting Docker's imported ID.
    with tarfile.open(archive, 'r:*') as source:
        manifests = json.load(source.extractfile('manifest.json'))
        if len(manifests) != 1:
            raise ValueError('card baseline must contain exactly one image')
        config = source.extractfile(manifests[0]['Config']).read()
        if 'sha256:' + hashlib.sha256(config).hexdigest() != expected:
            raise ValueError('card baseline image identity differs')
    output = run('docker', 'load', '--input', archive)
    references = re.findall(r'^Loaded image(?: ID)?: (.+)$', output, re.MULTILINE)
    if len(references) != 1:
        raise ValueError('Docker did not report exactly one loaded card image')
    return assemble.image(run('docker', 'image', 'inspect', references[0], '--format', '{{.Id}}'))


def card_baseline(args, card, publication):
    """Resolve/provision the selected Scheduler's sysroot outside the bundle."""
    context = args.work/'card-baseline'
    context.mkdir()
    provider = card['sysroot_tools']
    archive = context/'sdk-tools.tar.gz'
    download(args.bucket, provider['key'], archive)
    if assemble.sha(archive) != provider['sha256']:
        raise ValueError('card sysroot tooling checksum differs')
    shutil.copytree(publication, context/'payload', ignore=shutil.ignore_patterns('*-Linux-extras.tar.gz'))
    shutil.copyfile(HERE/'sysroot_baseline.py', context/'sysroot_baseline.py')
    shutil.copyfile(HERE/'CardBaseline.Dockerfile', context/'Dockerfile')
    iid = args.work/'card-image-id'
    # Dockerfile FROM treats a bare sha256 ID as a registry name. Tag only the
    # already verified local image; the tag is never used as a trust anchor.
    base = 'neat-offline-card-base:' + assemble.image(card['image_id']).removeprefix('sha256:')
    run('docker', 'tag', card['image_id'], base)
    run('docker', 'build', '--platform', 'linux/arm64',
        '--iidfile', iid, '--build-arg', 'CARD_IMAGE='+base, context)
    return assemble.image(iid.read_text().strip())


def metadata_names(args):
    return ('metadata.json', 'metadata-minimal.json') if is_core(args) else (assemble.METADATA_NAME,)


def core_identity(data, args, inputs, minimal):
    metadata_identity(data, args.sha, args.ref)
    resources = data['resources']
    extras = [name for name in resources if name.endswith('-Linux-extras.tar.gz')]
    if (assemble.CORE_ARCHIVE_NAME not in resources
            or (resources != [assemble.CORE_ARCHIVE_NAME] if minimal else
                len(resources) != 2 or len(extras) != 1)
            or data.get('installation', {}).get('script') != ''
            or data.get('artifact') != {'type': 'offline', 'repository': 'core',
                                      'package_path': 'offline', 'distro': 'elxr26', 'arches': ['arm64']}
            or data.get('source_metadata_sha256') != inputs):
        raise ValueError('existing standalone Core publication differs')


def existing(args, work, inputs, pending=True):
    base = artifact_prefix(args)
    ready, documents = {}, {}
    for filename in metadata_names(args):
        for suffix in ('', '.pending') if pending else ('',):
            path = work/(filename+suffix)
            if not download(args.bucket, base+'/'+filename+suffix, path, optional=True):
                continue
            data = json.loads(path.read_text())
            if is_core(args):
                core_identity(data, args, inputs, filename == 'metadata-minimal.json')
            else:
                offline_identity(data, args.sha, args.ref, args.distro, args.arch, inputs)
            for name, digest in data['resources-checksum'].items():
                verify_uploaded(args.bucket, base+'/'+name, digest)
            ready[filename], documents[filename] = path, data
            break
    if ready and set(ready) != set(metadata_names(args)):
        if all(path.name.endswith('.pending') for path in ready.values()):
            return None  # Interrupted staging is safe to rebuild before publication.
        raise ValueError('incomplete offline publication; refusing replacement')
    if is_core(args) and ready:
        name = assemble.CORE_ARCHIVE_NAME
        if documents['metadata.json']['resources-checksum'][name] != documents['metadata-minimal.json']['resources-checksum'][name]:
            raise ValueError('Core offline profiles refer to different runtime archives')
    return ready or None


def build(args):
    args.work.mkdir(parents=True, exist_ok=False)
    records, paths, inputs = source_metadata(args, args.work)
    if not args.validation_only and existing(args, args.work, inputs):
        return
    for role, (key, directory) in paths.items():
        for name, digest in records[role]['resources-checksum'].items():
            download(args.bucket, key+'/'+name, directory/name)
            if assemble.sha(directory/name) != digest:
                raise ValueError('publication resource checksum differs: '+name)
        assemble.verify_publication(directory, inputs[role])
    if is_core(args):
        name, digest = assemble.core_extras(records['card'])
        key, directory = paths['card']
        download(args.bucket, key+'/'+name, directory/name)
        if assemble.sha(directory/name) != digest:
            raise ValueError('Core extras archive checksum mismatch')
    baselines = json.loads((HERE/'baselines.json').read_text())
    if not is_core(args):
        host = paths['host'][1]
        if not (host/'platform_provider.py').is_file() or any(host.glob('simaai-pcie-host*.deb')):
            raise ValueError('Rebuild Core with the corrected Scheduler native producer; Platform provisioning must remain separate.')
        context = args.work/'host-baseline'
        context.mkdir()
        for name in ('platform_provider.py', 'platform-source.json'):
            shutil.copyfile(host/name, context/name)
        source = baselines['platform_source']
        download(args.bucket, source['key'], context/'platform-source.tar.gz')
        if assemble.sha(context/'platform-source.tar.gz') != source['sha256']:
            raise ValueError('Platform source baseline checksum differs')
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
    card = dict(card, image_id=load_card_image(image_archive, card['image_id']))
    image_archive.unlink()
    card_image = card_baseline(args, card, paths['card'][1])
    spec = {'installer_revision': args.installer_revision, 'targets': {}}
    if is_core(args):
        spec['scope'] = 'core'
    for role, (_, directory) in paths.items():
        spec['targets'][role] = {
            'publication': str(directory.resolve()), 'metadata_sha256': inputs[role],
            'core_revision': args.sha[:12],
            'image': host_image if role == 'host' else card_image,
            'architecture': args.arch if role == 'host' else 'arm64',
            'os_id': 'ubuntu' if role == 'host' else card['os_id'],
            'os_version': {'ubuntu22': '22.04', 'ubuntu24': '24.04'}[args.distro] if role == 'host' else card['os_version'],
        }
    (args.work/'spec.json').write_text(json.dumps(spec, indent=2)+'\n')
    root = args.work/('neat-core-offline' if is_core(args) else 'neat-offline')
    assemble.assemble(spec, root)
    documents = {}
    for filename in metadata_names(args):
        path = args.work/filename
        metadata = json.loads(path.read_text())
        metadata.update(source_metadata_sha256=inputs, installer_revision=args.installer_revision,
                        source_run_id=args.run_id)
        path.write_text(json.dumps(metadata, indent=2)+'\n')
        documents[filename] = metadata
    if args.validation_only:
        print('Offline validation passed; bundles retained locally without publication.')
        return
    base = artifact_prefix(args)
    # Stage all resources before metadata. The aggregate gate validates all builds
    # and both Core profiles before exposing any new metadata.
    resources = set().union(*(set(data['resources']) for data in documents.values()))
    for name in sorted(resources):
        put(args.bucket, base+'/'+name, args.work/name, args.kms)
    for filename in documents:
        put(args.bucket, base+'/'+filename+'.pending', args.work/filename, args.kms)


def publish(args):
    args.work.mkdir(parents=True, exist_ok=False)
    ready = []
    args.core = False
    for distro, arch in TUPLES:
        args.distro, args.arch = distro, arch
        work = args.work/(distro+'-'+arch)
        work.mkdir()
        _, _, inputs = source_metadata(args, work)
        path = existing(args, work, inputs)
        if path is None:
            raise ValueError('missing validated offline tuple: '+distro+'/'+arch)
        ready.extend((prefix(args.ref, args.sha, distro, arch), filename, record) for filename, record in path.items())
    args.core = True
    work = args.work/'core'
    work.mkdir()
    _, _, inputs = source_metadata(args, work)
    path = existing(args, work, inputs)
    if path is None:
        raise ValueError('missing validated standalone Core bundle')
    ready.extend((artifact_prefix(args), filename, record) for filename, record in path.items())
    # Complete every read/check before publishing any new metadata.
    for base, filename, path in ready:
        if path.name == filename:
            continue
        put(args.bucket, base+'/'+filename, path, args.kms, immutable=True)
    if args.distribution:
        run('aws', 'cloudfront', 'create-invalidation', '--distribution-id', args.distribution,
            '--paths', *['/'+quote(base, safe='/')+'/'+filename for base, filename, _ in ready])
    print('Published standalone Core and all four PCIe offline variants.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['build', 'publish'])
    p.add_argument('--core', action='store_true', help='build standalone Core without PCIe host packages')
    p.add_argument('--validation-only', action='store_true', help='build and validate without reusing or publishing release objects')
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
        if (not args.core and (args.distro, args.arch) not in TUPLES) or not re.fullmatch(r'[0-9a-f]{40}', args.installer_revision):
            p.error('build requires a host tuple and immutable installer revision')
        build(args)
    else:
        if args.validation_only:
            p.error('validation-only cannot publish')
        publish(args)


if __name__ == '__main__':
    main()
