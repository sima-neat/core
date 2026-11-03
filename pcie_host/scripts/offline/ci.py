#!/usr/bin/env python3
"""Build offline variants from authenticated Vulcan publication snapshots."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
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
    # a version suffix or the exact release tag version; its authenticated S3
    # key binds the source ref/commit in both cases.
    if distro:
        matches = data.get('commit') == sha and data.get('branch') == ref
    else:
        version = str(data.get('version', ''))
        release = (re.fullmatch(r'v[0-9]+\.[0-9]+\.[0-9]+(?:[.-][0-9A-Za-z][0-9A-Za-z.-]*)?', ref)
                   and version == ref[1:])
        matches = ((version.endswith('.' + sha[:12]) or release)
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


def resource_name(name, digest):
    suffix = '.tar.gz'
    if not name.endswith(suffix):
        raise ValueError('offline resources must be tar.gz archives')
    return name[:-len(suffix)] + '-' + digest[:8] + suffix


def archive_resource(data, name):
    return next((resource for resource in data['resources']
                 if resource == name or resource == resource_name(
                     name, data['resources-checksum'][resource])), None)


def staging_prefix(args, inputs):
    generation = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return artifact_prefix(args) + '/staging/' + generation


def offline_identity(data, sha, ref, distro, arch, inputs):
    metadata_identity(data, sha, ref, distro, arch)
    if (data['resources'] != [archive_resource(data, assemble.ARCHIVE_NAME)]
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


def put_resource(bucket, key, path, kms):
    try:
        put(bucket, key, path, kms, immutable=True)
    except subprocess.CalledProcessError:
        # A previous run may already have uploaded the same resource. Compare
        # the full digest: a short-hash collision must fail rather than overwrite.
        verify_uploaded(bucket, key, assemble.sha(path))


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


def baseline_dependencies(record, role):
    names = {'platform_provider.py', 'platform-source.json'}
    checksums = record['resources-checksum']
    if not names <= set(checksums):
        raise ValueError('publication lacks baseline provenance: '+role)
    return {name: checksums[name] for name in sorted(names)}


def baseline_config():
    config = json.loads((HERE/'baseline-images.json').read_text())
    if config.get('schema') != 1 or not re.fullmatch(r'[0-9a-f]{64}', config.get('recipe_sha256', '')):
        raise ValueError('unsupported baseline image contract')
    if config.get('repository') != '069171140053.dkr.ecr.us-west-2.amazonaws.com/vulcan/ubuntu-build':
        raise ValueError('unexpected baseline image registry')
    return config


def baseline_identity(role, dependencies):
    return hashlib.sha256(json.dumps({'schema': 1, 'role': role, 'dependencies': dependencies,
        'recipe_sha256': baseline_config()['recipe_sha256']},
        sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def pull_baseline(args, role, record):
    config = baseline_config()
    if role == 'card':
        # The test environment changes with Platform requirements, not with
        # every rebuilt Scheduler/Runner/Kernels package under test.
        digest, key = config['card']['digest'], config['card']['inputs_sha256']
        if (not re.fullmatch(r'sha256:[0-9a-f]{64}', digest)
                or not re.fullmatch(r'[0-9a-f]{64}', key)):
            raise ValueError('invalid pinned card baseline identity')
    else:
        key = baseline_identity(role, baseline_dependencies(record, role))
        tag = config['repository']+':offline-'+role+'-'+key
        result = subprocess.run(['docker', 'manifest', 'inspect', '--verbose', tag], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Cannot resolve clean baseline '+role+'. Publish matching images with Vulcan '
                               'Build reusable Ubuntu images (offline_source_sha='+args.sha+'). '+result.stderr)
        manifest = json.loads(result.stdout)
        digest = manifest.get('Descriptor', {}).get('digest') if isinstance(manifest, dict) else None
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest or ''):
            raise ValueError('baseline must be a single-platform manifest with a digest')
    image = config['repository']+'@'+digest
    print('Pulling Vulcan clean baseline: '+image, flush=True)
    run('docker', 'pull', image)
    info = json.loads(run('docker', 'image', 'inspect', image))[0]
    labels = info.get('Config', {}).get('Labels', {}) or {}
    if any(labels.get('ai.sima.offline.'+name) != value for name, value in
           [('inputs', key), ('recipe', config['recipe_sha256']), ('role', role)]):
        raise ValueError('published baseline provenance differs: '+role)
    if info['Architecture'] != ('arm64' if role == 'card' else role.split('-')[1]):
        raise ValueError('published baseline architecture differs: '+role)
    local_id = assemble.image(info['Id'])
    path = args.work/'baseline-images.json'
    evidence = json.loads(path.read_text()) if path.exists() else {}
    evidence[role] = {'image': image, 'local_id': local_id, 'inputs_sha256': key,
                      'recipe_sha256': config['recipe_sha256']}
    path.write_text(json.dumps(evidence, indent=2)+'\n')
    return local_id


def metadata_names(args):
    return ('metadata.json', 'metadata-minimal.json') if is_core(args) else (assemble.METADATA_NAME,)


def core_identity(data, args, inputs, minimal):
    metadata_identity(data, args.sha, args.ref)
    resources = data['resources']
    extras = [name for name in resources
              if re.search(r'-Linux-extras(?:-[0-9a-f]{8})?\.tar\.gz$', name)]
    if (archive_resource(data, assemble.CORE_ARCHIVE_NAME) is None
            or (resources != [archive_resource(data, assemble.CORE_ARCHIVE_NAME)] if minimal else
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
            key = (staging_prefix(args, inputs) if suffix else base) + '/' + filename
            if not download(args.bucket, key, path, optional=True):
                continue
            data = json.loads(path.read_text())
            recorded_inputs = data.get('source_metadata_sha256')
            if (not isinstance(recorded_inputs, dict) or set(recorded_inputs) != set(inputs)
                    or any(not re.fullmatch(r'[0-9a-f]{64}', str(value))
                           for value in recorded_inputs.values())):
                raise ValueError('offline publication lacks valid source metadata hashes')
            if is_core(args):
                core_identity(data, args, recorded_inputs, filename == 'metadata-minimal.json')
            else:
                offline_identity(data, args.sha, args.ref, args.distro, args.arch, recorded_inputs)
            if data.get('source_metadata_sha256') != inputs:
                print('Source publications changed; rebuilding ' + base, flush=True)
                continue
            for name, digest in data['resources-checksum'].items():
                verify_uploaded(args.bucket, base+'/'+name, digest)
            ready[filename], documents[filename] = path, data
            break
    if ready and set(ready) != set(metadata_names(args)):
        if all(path.name.endswith('.pending') for path in ready.values()):
            return None  # Interrupted staging is safe to rebuild before publication.
        raise ValueError('incomplete offline publication; refusing replacement')
    if is_core(args) and ready:
        name = archive_resource(documents['metadata.json'], assemble.CORE_ARCHIVE_NAME)
        if archive_resource(documents['metadata-minimal.json'], assemble.CORE_ARCHIVE_NAME) != name:
            raise ValueError('Core offline profiles refer to different runtime archives')
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
    if not is_core(args):
        host = paths['host'][1]
        if not (host/'platform_provider.py').is_file() or any(host.glob('simaai-pcie-host*.deb')):
            raise ValueError('Rebuild Core with the corrected Scheduler native producer; Platform provisioning must remain separate.')
        host_image = pull_baseline(args, args.distro+'-'+args.arch, records['host'])
    card_image = pull_baseline(args, 'card', records['card'])
    card = baseline_config()['card']
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
    documents, resource_files = {}, {}
    for filename in metadata_names(args):
        path = args.work/filename
        metadata = json.loads(path.read_text())
        if not args.validation_only:
            resource_files.update({resource_name(name, digest): args.work/name
                                   for name, digest in metadata['resources-checksum'].items()})
            metadata['resources'] = [resource_name(name, metadata['resources-checksum'][name])
                                     for name in metadata['resources']]
            metadata['resources-checksum'] = {resource_name(name, digest): digest
                                              for name, digest in metadata['resources-checksum'].items()}
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
        # Never replace bytes if two full digests share the short filename hash.
        put_resource(args.bucket, base+'/'+name, resource_files[name], args.kms)
    for filename in documents:
        put(args.bucket, staging_prefix(args, inputs)+'/'+filename, args.work/filename, args.kms)


def publish(args):
    args.work.mkdir(parents=True, exist_ok=False)
    ready, snapshots = [], []
    args.core = False
    for distro, arch in TUPLES:
        args.distro, args.arch = distro, arch
        work = args.work/(distro+'-'+arch)
        work.mkdir()
        _, _, inputs = source_metadata(args, work)
        snapshots.append((False, distro, arch, inputs))
        path = existing(args, work, inputs)
        if path is None:
            raise ValueError('missing validated offline tuple: '+distro+'/'+arch)
        ready.extend((prefix(args.ref, args.sha, distro, arch), filename, record) for filename, record in path.items())
    args.core = True
    work = args.work/'core'
    work.mkdir()
    _, _, inputs = source_metadata(args, work)
    snapshots.append((True, None, None, inputs))
    path = existing(args, work, inputs)
    if path is None:
        raise ValueError('missing validated standalone Core bundle')
    ready.extend((artifact_prefix(args), filename, record) for filename, record in path.items())
    # Online publications can be replaced while assembly is running. Recheck
    # every snapshot before exposing any replacement offline metadata.
    for index, (core, distro, arch, expected) in enumerate(snapshots):
        if expected.get('card') != inputs.get('card'):
            raise ValueError('Core publication changed across offline variants; retry assembly')
        args.core, args.distro, args.arch = core, distro, arch
        check = args.work/('source-recheck-' + str(index))
        check.mkdir()
        _, _, current = source_metadata(args, check)
        if current != expected:
            raise ValueError('source publication changed before offline promotion; retry assembly')
    # Complete every read/check before publishing any new metadata.
    for base, filename, path in ready:
        if path.name == filename:
            continue
        put(args.bucket, base+'/'+filename, path, args.kms)
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
