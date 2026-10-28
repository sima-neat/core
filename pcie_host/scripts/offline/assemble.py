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
ARCHIVE_NAME = 'neat-pcie-offline-installer.tar.gz'
CORE_ARCHIVE_NAME = 'neat-core-offline-installer.tar.gz'
METADATA_NAME = 'metadata-offline.json'


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
        return digest.hexdigest()


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
    if not re.fullmatch(r'(?:[A-Za-z0-9./:_-]+@)?sha256:[0-9a-f]{64}', value):
        raise ValueError('baseline image must be pinned by sha256 digest')
    return value


def core_extras(source):
    extras = [row for row in source.get('selectable-resources', [])
              if str(row.get('resource', '')).endswith('-Linux-extras.tar.gz')]
    if len(extras) != 1:
        raise ValueError('Core publication must declare exactly one tutorials/tests extras archive')
    name, digest = extras[0]['resource'], extras[0].get('checksum', '')
    if (not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._+~-]*', name)
            or not re.fullmatch(r'[0-9a-f]{64}', digest)):
        raise ValueError('unsafe Core extras resource or checksum')
    return name, digest


def assemble(spec, output):
    core = spec.get('scope') == 'core'
    roles = ('card',) if core else ('host', 'card')
    archive_name = CORE_ARCHIVE_NAME if core else ARCHIVE_NAME
    if set(spec['targets']) != set(roles):
        raise ValueError('standalone Core requires only the card target' if core else 'both host and card targets are required')
    if output.exists() or (output.parent/archive_name).exists():
        raise ValueError('output must not exist; incomplete builds must not be reused')
    # Validate every input before starting a container or creating the bundle.
    publications = {}
    for target, row in spec['targets'].items():
        image(row['image'])
        if core and (row['os_id'], row['os_version'], row['architecture']) != ('elxr', '26.04', 'arm64'):
            raise ValueError('standalone Core requires eLxr 26.04 ARM64')
        expected = ('amd64', 'arm64') if target == 'host' else ('arm64',)
        if row['architecture'] not in expected:
            raise ValueError('incorrect target architecture')
        if target == 'host' and (row['os_id'] != 'ubuntu' or row['os_version'] not in ('22.04', '24.04')):
            raise ValueError('host support requires Ubuntu 22.04 or 24.04 amd64 or arm64')
        publications[target] = verify_publication(Path(row['publication']), row['metadata_sha256'])
        if core:
            name, digest = core_extras(publications[target])
            extras = Path(row['publication']) / name
            if extras.is_symlink() or not extras.is_file() or sha(extras) != digest:
                raise ValueError('Core extras archive checksum mismatch')
    # Publications must describe one immutable Core revision, not merely version 0.6.0.
    revisions = [row['core_revision'] for row in spec['targets'].values()]
    if len(set(revisions)) != 1 or not re.fullmatch(r'[0-9a-f]{12,40}', revisions[0]):
        raise ValueError('matching immutable Core revisions are required')
    for target, data in publications.items():
        row = spec['targets'][target]
        identity = str(data.get('commit', '')) + ' ' + str(data.get('version', ''))
        if row['core_revision'] not in identity:
            raise ValueError('Core revision is absent from publication identity')
    output.mkdir(parents=True)
    for target in roles:
        row = spec['targets'][target]
        dest = output / target
        payload = dest / 'payload'
        payload.mkdir(parents=True)
        for name in publications[target]['resources'] + ['metadata.json']:
            shutil.copyfile(Path(row['publication']) / name, payload / name)
        shutil.copyfile(HERE / 'target.py', dest / 'target.py')
        # No host paths or credentials are saved in the distributable manifest.
        (dest / 'target.json').write_text(json.dumps(dict(role=target, **{k: row[k] for k in (
            'image', 'architecture', 'os_id', 'os_version', 'core_revision', 'metadata_sha256')}), indent=2)+'\n')
        # EC2 regional archive avoids the public archive bottleneck during collection.
        # Rewrite both legacy sources.list and Ubuntu 24.04 deb822 sources only
        # in this disposable container; security and ARM64 repositories stay intact.
        mirror = (
            "for source in /etc/apt/sources.list /etc/apt/sources.list.d/*.list "
            "/etc/apt/sources.list.d/*.sources; do "
            '[ ! -f "$source" ] || sed -i '
            "'s|http://archive.ubuntu.com/ubuntu|http://us-west-2.ec2.archive.ubuntu.com/ubuntu|g; "
            "s|https://archive.ubuntu.com/ubuntu|http://us-west-2.ec2.archive.ubuntu.com/ubuntu|g' "
            '"$source"; done; '
        ) if target == 'host' and row['architecture'] == 'amd64' else ''
        run('docker', 'run', '--rm', '--user', '0:0', '--env', 'DEBIAN_FRONTEND=noninteractive', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--platform', 'linux/' + row['architecture'],
            '--mount', f'type=bind,src={dest.resolve()},dst=/bundle',
            '--entrypoint', '/bin/sh', row['image'], '-ec',
            mirror + 'cp /var/lib/dpkg/status /bundle/baseline.status; '
            'apt-get update; apt-get install -y --no-install-recommends python3 python3-venv python3-pip dpkg-dev; '
            'python3 /bundle/target.py collect')
    for name in ('assemble.py', 'deploy.py', 'process.py', 'log_progress.py', 'lifecycle.sh'):
        shutil.copyfile(HERE / name, output / name)
    if core:
        shutil.copyfile(HERE / 'core_install.py', output / 'install.py')
        for name in ('sdk_install.py', 'paired_devkit.py', 'progress.py'):
            shutil.copyfile(HERE / name, output / name)
        shutil.copyfile(HERE / 'CORE_INSTALL.md', output / 'INSTALL.md')
        shutil.copyfile(HERE / 'core_install.sh', output / 'install.sh')
        (output / 'install.sh').chmod(0o755)
    else:
        shutil.copyfile(HERE.parent/'pcie-setup.sh', output/'pcie-setup.sh')
        for name in ('install.py', 'discover.py', 'INSTALL.md'):
            shutil.copyfile(HERE / name, output / name)
    files = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'bundle.json').write_text(json.dumps({
        'schema_version': 1, 'core_revision': revisions[0], 'files': files,
        'installer_revision': spec.get('installer_revision', revisions[0]),
        'scope': 'core' if core else 'Neat host/card deployment; Platform system package required',
    }, indent=2)+'\n')
    # No success archive is emitted until every network-isolated install passes.
    validate(output)
    archive = archive_bundle(output, archive_name, 'neat-core-offline' if core else 'neat-offline')
    if core:
        shutil.copyfile(extras, output.parent / extras.name)
        write_core_metadata(output, archive)
    else:
        write_metadata(output, archive)


def archive_bundle(root, archive_name=ARCHIVE_NAME, directory='neat-offline'):
    """Stable archive bytes; identical files share an earlier tar member."""
    contents = {}
    destination = root.parent/archive_name
    with destination.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w') as archive:
                for path in sorted(root.rglob('*')):
                    info = archive.gettarinfo(path, arcname=str(Path(directory) / path.relative_to(root)))
                    info.uid = info.gid = info.mtime = 0
                    info.uname = info.gname = ''
                    info.mode = 0o755 if path.is_dir() or path.name == 'install.sh' else 0o644
                    if path.is_file():
                        # Preserve all publication/repository paths without storing
                        # duplicate bytes. Reset inode-based tarfile detection so
                        # copied and already-hardlinked inputs produce identical tar.
                        info.type, info.linkname = tarfile.REGTYPE, ''
                        info.size = path.stat().st_size
                        key = (info.size, sha(path))
                        if key in contents:
                            info.type, info.linkname = tarfile.LNKTYPE, contents[key]
                            info.size = 0
                            archive.addfile(info)
                        else:
                            contents[key] = info.name
                            with path.open('rb') as stream:
                                archive.addfile(info, stream)
                    else:
                        archive.addfile(info)

    return destination


def write_metadata(root, archive):
    """Publish a download-only variant beside the architecture-specific archive."""
    manifest = verify_bundle(root)
    row = json.loads((root/'host/target.json').read_text())
    source = json.loads((root/'host/payload/metadata.json').read_text())
    arch = row['architecture']
    if arch not in ('amd64', 'arm64') or (row['os_id'] != 'ubuntu' or row['os_version'] not in ('22.04', '24.04')):
        raise ValueError('unsupported offline host tuple')
    if archive.name != ARCHIVE_NAME:
        raise ValueError('incorrect offline archive filename')
    distro = 'ubuntu' + row['os_version'].split('.')[0]
    package_path = f'pciehost/{distro}/{arch}'
    size = archive.stat().st_size
    metadata = {key: source[key] for key in (
        'repository', 'branch', 'commit', 'commit_folder', 'published_at_utc') if key in source}
    metadata.update({
        'name': f'gh:sima-neat/core/{package_path}',
        'version': source.get('version', manifest['core_revision']), 'release': source.get('release', ''),
        'description': f"Offline Neat installer for Ubuntu {row['os_version']} {arch} hosts and Platform 3.0 cards",
        'platforms': [{'type': 'host', 'os': ['linux'], 'arch': [arch]}],
        'resources': [ARCHIVE_NAME], 'resources-checksum': {ARCHIVE_NAME: sha(archive)},
        'selectable-resources': [],
        'installation': {'script': '', 'post-message':
            'Offline installer downloaded. On the prepared PCIe host, run: '
            'cd neat-offline && sudo python3 -B install.py'},
        'artifact': {'type': 'offline', 'repository': 'core',
                     'package_path': package_path, 'distro': distro, 'arches': [arch]},
        'size': {'download': f'{(size + 2**20 - 1)//2**20}MB',
                 'install': f'{(sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) + 2**20 - 1)//2**20}MB'},
    })
    path = archive.parent/METADATA_NAME
    path.write_text(json.dumps(metadata, indent=2)+'\n')
    return path


def write_core_metadata(root, archive):
    manifest = verify_bundle(root)
    row = json.loads((root/'card/target.json').read_text())
    source = json.loads((root/'card/payload/metadata.json').read_text())
    if (row['os_id'], row['os_version'], row['architecture']) != ('elxr', '26.04', 'arm64'):
        raise ValueError('standalone Core requires eLxr 26.04 ARM64')
    metadata = {key: source[key] for key in (
        'repository', 'branch', 'commit', 'commit_folder', 'published_at_utc') if key in source}
    metadata.update({
        'name': 'gh:sima-neat/core/offline',
        'version': source.get('version', manifest['core_revision']),
        'release': source.get('release', ''),
        'description': 'Offline Neat Core installer for the Platform 3.0 Neat SDK and standalone Modalix DevKits',
        # Downloadable on any connected computer; install.py checks the target.
        'platforms': [],
        'resources': [archive.name], 'resources-checksum': {archive.name: sha(archive)},
        'selectable-resources': [],
        'installation': {'script': '', 'post-message':
            'Extract the archive inside the Neat SDK or on the Modalix DevKit, then run: cd neat-core-offline && ./install.sh'},
        'artifact': {'type': 'offline', 'repository': 'core', 'package_path': 'offline',
                     'distro': 'elxr26', 'arches': ['arm64']},
        'size': {'download': f'{(archive.stat().st_size + 2**20 - 1)//2**20}MB'},
    })
    extras_name, extras_hash = core_extras(source)
    extras = archive.parent / extras_name
    if extras.is_symlink() or sha(extras) != extras_hash:
        raise ValueError('Core extras archive checksum mismatch')
    with tarfile.open(extras, 'r:*') as package:
        extras_size = sum(member.size for member in package if member.isfile())
    installed_size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
    paths = {}
    for minimal, filename in ((False, 'metadata.json'), (True, 'metadata-minimal.json')):
        data = dict(metadata)
        resources = [archive.name] if minimal else [archive.name, extras_name]
        hashes = {archive.name: sha(archive), extras_name: extras_hash}
        data['resources'] = resources
        data['resources-checksum'] = {name: hashes[name] for name in resources}
        data['size'] = {
            'download': f'{(sum((archive.parent / name).stat().st_size for name in resources) + 2**20 - 1)//2**20}MB',
            'install': f'{(installed_size + (0 if minimal else extras_size) + 2**20 - 1)//2**20}MB',
        }
        path = archive.parent / filename
        path.write_text(json.dumps(data, indent=2)+'\n')
        paths[filename] = path
    return paths


def verify_bundle(root, target=None):
    data = json.loads((root / 'bundle.json').read_text())
    if any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('bundle symlinks are forbidden')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} - {'bundle.json'}
    expected = {name: digest for name, digest in data['files'].items()
                if target is None or name.startswith(target + '/') or '/' not in name}
    # A card transfer omits host files and the host-only discovery/coordinator.
    if target == 'card' and data.get('scope') != 'core':
        expected = {name: digest for name, digest in expected.items()
                    if name not in ('install.py', 'discover.py', 'pcie-setup.sh')}
    if actual != set(expected):
        raise ValueError('bundle file inventory differs')
    for name, digest in expected.items():
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('unsafe bundle path')
        path = root / name
        if path.is_symlink() or sha(path) != digest:
            raise ValueError('bundle checksum mismatch: ' + name)
    return data


def validate(root):
    manifest = verify_bundle(root)
    for target in (('card',) if manifest.get('scope') == 'core' else ('host', 'card')):
        dest = root / target
        row = json.loads((dest / 'target.json').read_text())
        image(row['image'])
        standalone = manifest.get('scope') == 'core'
        admission = ('; python3 -B -c \'import sys; from pathlib import Path; '
                     'sys.path.insert(0, "/offline-bundle"); from install import environment; '
                     'assert environment(Path("/offline-bundle")) == "devkit"\'' if standalone else '')
        run('docker', 'run', '--rm', '--user', '0:0', '--env', 'DEBIAN_FRONTEND=noninteractive', '--env', 'PYTHONDONTWRITEBYTECODE=1', '--network=none', '--platform', 'linux/' + row['architecture'],
            '--mount', f'type=bind,src={dest.resolve()},dst=/bundle,readonly',
            *(['--mount', f'type=bind,src={root.resolve()},dst=/offline-bundle,readonly'] if standalone else []),
            '--entrypoint', '/bin/sh', row['image'], '-ec',
            # Bootstrap Python from the local repo; the baseline need not contain it.
            'cmp /var/lib/dpkg/status /bundle/baseline.status; '
            'cp -a /bundle /tmp/offline; cd /tmp/offline; '
            'sh ./bootstrap.sh; python3 ./target.py validate'+admission)


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
