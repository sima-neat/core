#!/usr/bin/env python3
"""Read-only Core artifact-set checks; producer receipts and CLI own their schemas.

Run with the Python environment belonging to the selected sima-cli. Expected
inputs are explicit trusted producer paths, never paths discovered in receipts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile
import zipfile
from email.parser import BytesParser


class ContractError(ValueError):
    """A composed distribution is not self-consistent."""


def require(condition, message):
    if not condition:
        raise ContractError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resource(root, name):
    require(isinstance(name, str) and name and '\\' not in name, f'Invalid resource: {name!r}')
    path = PurePosixPath(name)
    require(len(path.parts) == 1 and name not in ('.', '..') and not path.is_absolute(),
            f'Unsafe resource: {name!r}')
    result = root / name
    require(result.is_file() and not result.is_symlink(), f'Missing/nonregular resource: {name}')
    require(result.resolve().parent == root.resolve(), f'Escaping resource: {name}')
    return result


def single(root, pattern):
    paths = list(root.glob(pattern))
    require(len(paths) == 1, f'Expected exactly one {pattern}, got {len(paths)}')
    return resource(root, paths[0].name)


def control(path):
    fields = subprocess.check_output(
        ['dpkg-deb', '-f', str(path), 'Package', 'Version', 'Architecture'], text=True)
    return dict(line.split(': ', 1) for line in fields.splitlines())


def scheduler_platform_description(root):
    """Use the authenticated producer's classification, never infer backend names."""
    delegate = Path(__file__).resolve().parents[1] / 'scheduler_platform_admission.py'
    try:
        result = json.loads(subprocess.check_output(
            [sys.executable, '-B', str(delegate), 'describe', '--directory', str(Path(root).resolve())],
            text=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        raise ContractError('Scheduler Platform classification failed') from error
    require(isinstance(result, dict) and result.get('schema_version') == 1
            and isinstance(result.get('redistributed_provider_controls'), list)
            and isinstance(result.get('platform_packages'), dict),
            'Invalid Scheduler Platform classification')
    return result


def composed_provider_controls(receipt, root):
    """Use the authenticated owner for both Platform and historical providers.

    The actual incoming Scheduler receipt is checked before any classification;
    stripping an untrusted declaration cannot select a weaker legacy path.
    """
    classification = scheduler_platform_description(root)
    incoming = {control(path)['Package'] for path in Path(root).glob('*.deb')}
    require(not incoming.intersection(classification['platform_packages']),
            'Platform-owned provider DEB is present in the install set')
    return classification['redistributed_provider_controls']


def verify(root, *, validate_metadata, expected_manifest=None, expected_receipt=None,
           expected_deb_dirs=()):
    root = Path(root)
    manifest = json.loads(resource(root, 'manifest.json').read_text())
    for key in ('platform-version', 'modelzoo-version', 'abi-version'):
        require(isinstance(manifest.get(key), str) and manifest[key].strip(), f'Missing {key}')
    require(manifest['abi-version'].isdigit() and int(manifest['abi-version']) > 0, 'Invalid ABI version')
    if expected_manifest:
        selected = json.loads(Path(expected_manifest).read_text())
        expected = {key: str(selected.get(key, '')).strip() for key in manifest if key in (
            'platform-version', 'modelzoo-version', 'abi-version')}
        expected['modelzoo-version'] = expected['modelzoo-version'] or expected['platform-version']
        require(all(manifest[k] == v for k, v in expected.items()), 'Compatibility projection differs')
    receipt_path = resource(root, 'resolved-deps-manifest.json')
    receipt = json.loads(receipt_path.read_text())
    if expected_receipt:
        require(receipt_path.read_bytes() == Path(expected_receipt).read_bytes(), 'Backend receipt differs')
    core = single(root, 'sima-neat-*-Linux-core.deb')
    dev = single(root, 'sima-neat-*-Linux-dev.deb')
    extras = single(root, '*-Linux-extras.tar.gz')
    wheel = single(root, '*.whl')
    require(wheel.name.startswith('pyneat-'), 'Expected the public PyNeat wheel')
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        require(len(names) == len(set(names)), 'Duplicate wheel paths')
        metadata_paths = [name for name in names if name.endswith('.dist-info/METADATA')]
        require(len(metadata_paths) == 1, 'Ambiguous wheel distribution metadata')
        metadata_root = metadata_paths[0].split('/')[0]
        require(metadata_root.startswith('pyneat-'), 'Private wheel distribution namespace')
        require(BytesParser().parsebytes(archive.read(metadata_paths[0]))['Name'] == 'pyneat',
                'Expected the public PyNeat distribution')
        for name in names:
            path = PurePosixPath(name)
            require(not path.is_absolute() and '..' not in path.parts and
                    path.parts[0] in ('pyneat', metadata_root), 'Foreign wheel payload')
            if path.name.endswith('.so') or '.so.' in path.name:
                require(path.parent == PurePosixPath('pyneat') and
                        path.name.startswith('_pyneat_core.'), 'Runtime provider bundled in public wheel')
    debs = {p.name: resource(root, p.name) for p in root.glob('*.deb')}
    controls = {}
    for name, path in debs.items():
        fields = control(path)
        require(fields.get('Package') not in controls, 'Duplicate DEB package role')
        require(fields.get('Architecture') in ('arm64', 'all'), f'Incompatible package architecture: {name}')
        controls[fields['Package']] = (name, fields)
    expected_roles = {}
    for component, role in (('kernels', 'simaai-kernels-dev'), ('runner', 'simaai-runner'),
                            ('scheduler', 'simaai-scheduler'), ('runtime_compat', 'simaai-runtime')):
        require(component in receipt and receipt[component].get('debian_version'), f'Missing {component} receipt')
        expected_roles[role] = receipt[component]['debian_version']
    for entry in receipt.get('external_dependencies', {}).get('SimaCodecs', {}).get('debian_packages', []):
        role = entry['Package']
        require(role not in expected_roles, f'Duplicate receipt role: {role}')
        expected_roles[role] = entry['Version']
        require(role in controls and digest(debs[controls[role][0]]) == entry['sha256'],
                f'Codecs archive differs from consumed receipt: {role}')
    test_support = receipt.get('client_test_support')
    if test_support is not None:
        entry = test_support.get('debian_package', {})
        role = 'simaai-scheduler-client-test-dev'
        require(entry.get('Package') == role and entry.get('Architecture') == 'arm64'
                and entry.get('Version') == receipt['scheduler']['debian_version'],
                'Client test-support differs from selected Scheduler version/architecture')
        require(role in controls and controls[role][1]['Version'] == entry['Version']
                and digest(debs[controls[role][0]]) == entry.get('deb_sha256'),
                'Client test-support archive differs from consumed receipt')
        expected_roles[role] = entry['Version']
    provider_roles = set()
    for entry in composed_provider_controls(receipt, root):
        require(isinstance(entry, dict) and all(isinstance(entry.get(key), str) and entry[key]
                for key in ('Package', 'Version', 'Architecture', 'deb_sha256')),
                'Incomplete composed provider control')
        role = entry['Package']
        require(role not in expected_roles, f'Duplicate receipt role: {role}')
        require(role in controls and all(controls[role][1].get(key) == entry[key]
                for key in ('Package', 'Version', 'Architecture')),
                f'Missing/mismatched provider control: {role}')
        require(digest(debs[controls[role][0]]) == entry['deb_sha256'],
                f'Provider archive differs from consumed receipt: {role}')
        expected_roles[role] = entry['Version']
        provider_roles.add(role)
    for role, version in expected_roles.items():
        require(role in controls and controls[role][1]['Version'] == version, f'Missing/mismatched {role}')
    product_roles = {control(core)['Package'], control(dev)['Package']}
    require(product_roles == {'sima-neat', 'sima-neat-dev'}, 'Wrong Core/dev package roles')
    require(control(core)['Version'] == control(dev)['Version'], 'Core/dev version mismatch')
    # These are the existing customer LLiMa roles, not a fixed package count.
    allowed = set(expected_roles) | product_roles | {'sima-lmm-core', 'sima-lmm-cli'}
    require(set(controls) <= allowed, f'Foreign DEB package roles: {set(controls) - allowed}')
    require(all(fields['Architecture'] == 'arm64' for role, (_, fields) in controls.items()
                if role not in provider_roles), 'Non-arm64 product/dependency package')
    for directory in expected_deb_dirs:
        selected = list(Path(directory).glob('*.deb'))
        require(selected, f'Empty expected dependency directory: {directory}')
        for path in selected:
            # The producer intentionally excludes development-only LLiMa payloads.
            if control(path)['Package'] == 'sima-lmm-dev':
                continue
            require(path.name in debs and digest(debs[path.name]) == digest(path),
                    f'Staged dependency differs: {path.name}')
    installables = set(debs) | {wheel.name}
    allowlist_path = resource(root, 'neat-install-manifest.txt')
    allowlist = [s.strip() for s in allowlist_path.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
    require(len(allowlist) == len(set(allowlist)) and set(allowlist) == installables, 'Installer allowlist differs')
    base = installables | {'manifest.json', receipt_path.name, allowlist_path.name,
                           'install_neat_framework.sh', 'install_sentinel.sh', 'scheduler_platform_admission.py'}
    expected_variants = {
        'metadata.json': (base, {extras.name}),
        'metadata-minimal.json': (base, set()),
        'metadata-all.json': (base | {extras.name}, set()),
        'metadata-pyneat.json': ({wheel.name}, set()),
        'metadata-extras.json': ({extras.name}, set()),
    }
    identities = set()
    hashes = {}
    for filename, (expected_resources, expected_selectables) in expected_variants.items():
        metadata = json.loads(resource(root, filename).read_text())
        validate_metadata(metadata)  # Raising API from the selected CLI; no vendored schema.
        identities.add((metadata['name'], metadata['version']))
        resources = metadata['resources']
        selectables = metadata.get('selectable-resources', [])
        selected = [entry['resource'] for entry in selectables]
        require(len(resources) == len(set(resources)) and set(resources) == expected_resources,
                f'{filename}: wrong/duplicate resources')
        require(len(selected) == len(set(selected)) and set(selected) == expected_selectables,
                f'{filename}: wrong/duplicate selectables')
        recorded = metadata.get('resources-checksum', {})
        require(set(recorded) == set(resources), f'{filename}: missing/foreign resource checksum')
        checksums = [(name, recorded[name]) for name in resources]
        for entry in selectables:
            require(not entry.get('url'), f'{filename}: selectable must use local composed artifact')
            checksums.append((entry['resource'], entry.get('checksum')))
        for name, expected_hash in checksums:
            path = resource(root, name)
            if name not in hashes:
                hashes[name] = digest(path)
            require(expected_hash == hashes[name], f'{filename}: checksum mismatch: {name}')
        if filename != 'metadata-extras.json':
            platforms = metadata['platforms']
            for kind in ('board', 'palette'):
                matches = [p for p in platforms if p['type'] == kind]
                require(len(matches) == 1 and matches[0].get('version') == manifest['platform-version'],
                        f'{filename}: missing/mismatched {kind} compatibility')
            require('modalix' in next(p for p in platforms if p['type'] == 'board')['compatible_with'],
                    f'{filename}: Modalix compatibility missing')
        expected_script = ':' if filename in ('metadata-pyneat.json', 'metadata-extras.json') else './install_neat_framework.sh'
        require(metadata.get('installation', {}).get('script') == expected_script, f'{filename}: wrong installer')
    require(identities == {('sima-neat', control(core)['Version'])},
            'Metadata variants disagree with Core package identity')
    with tarfile.open(extras) as archive:
        entries = archive.getmembers()
        members = {str(PurePosixPath(m.name)): m for m in entries}
        for name in members:
            path = PurePosixPath(name)
            require('CMakeFiles' not in path.parts
                    and path.name not in {'CMakeCache.txt', 'cmake_install.cmake'}
                    and path.suffix not in {'.o', '.obj'},
                    f'Extras contains generated build internals: {name}')
        for tree in ('tests', 'tutorials'):
            name = f'lib/sima-neat/{tree}/CTestTestfile.cmake'
            require(sum(str(PurePosixPath(m.name)) == name for m in entries) == 1
                    and members[name].isfile(), f'Extras missing installed {tree} CTest tree')
    return {'status': 'PASS', 'artifacts_verified': len(hashes), 'dependency_roles': sorted(expected_roles)}

# Required roles are policy, not inferred from whichever files happen to be in
# an artifact directory. Optional backend providers still come from the
# independently selected and validated producer receipts above.
PACKAGE_PROFILES = {
    'core': {'sima-neat', 'sima-neat-dev', 'simaai-kernels-dev',
             'simaai-runner', 'simaai-scheduler', 'simaai-runtime'},
}
PACKAGE_PROFILES['core-codecs'] = PACKAGE_PROFILES['core'] | {'neat-codecs', 'neat-codecs-dev'}
PACKAGE_PROFILES['core-codecs-llima'] = PACKAGE_PROFILES['core-codecs'] | {'sima-lmm-core', 'sima-lmm-cli'}


def payload_soname(stream, digest_state):
    """Hash a regular member and inspect ELF identity using the existing binutils tool.

    No archive pathname is extracted. Temporary bytes belong only to this read-only
    inspection; non-ELF payloads do not require a scratch file or an ELF subprocess.
    """
    first = stream.read(1024 * 1024)
    digest_state.update(first)
    if not first.startswith(b'\x7fELF'):
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest_state.update(chunk)
        return None
    with tempfile.NamedTemporaryFile(prefix='core-package-elf-') as scratch:
        scratch.write(first)
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest_state.update(chunk)
            scratch.write(chunk)
        scratch.flush()
        dynamic = subprocess.check_output(['readelf', '-dW', scratch.name], text=True,
                                          env={**os.environ, 'LC_ALL': 'C'})
    names = re.findall(r'\(SONAME\).*?\[([^\]\n]+)\]', dynamic)
    require(len(names) <= 1, 'Ambiguous ELF SONAME in package payload')
    if names:
        require('/' not in names[0] and '\\' not in names[0], 'Invalid ELF SONAME in package payload')
        return names[0]
    return None


def deb_payload(path):
    """Inspect archive bytes without extracting files or running maintainer scripts."""
    payload = {}
    process = subprocess.Popen(['dpkg-deb', '--fsys-tarfile', str(path)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=process.stdout, mode='r|') as archive:
            for member in archive:
                name = PurePosixPath(member.name)
                require(not name.is_absolute() and '..' not in name.parts,
                        f'Unsafe DEB payload path: {member.name}')
                if member.isdir():
                    continue
                require(name.parts and str(name) != '.', 'Empty DEB payload path')
                key = '/' + str(name)
                require(key not in payload, f'Duplicate DEB payload path: {key}')
                if member.isfile():
                    h = hashlib.sha256()
                    with archive.extractfile(member) as stream:
                        soname = payload_soname(stream, h)
                    payload[key] = {'sha256': h.hexdigest()}
                    if soname:
                        payload[key]['soname'] = soname
                elif member.issym():
                    payload[key] = {'symlink': member.linkname}
                elif member.islnk():
                    target = PurePosixPath(member.linkname)
                    require(not target.is_absolute() and '..' not in target.parts,
                            f'Unsafe DEB hard link: {member.linkname}')
                    payload[key] = {'hardlink': '/' + str(target)}
                else:
                    raise ContractError(f'Unsupported DEB payload type: {key}')
        require(process.wait() == 0, f'Cannot inspect DEB payload: {path}')
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
            process.wait()
    for name, entry in payload.items():
        if 'hardlink' in entry:
            target = payload.get(entry['hardlink'], {})
            require('sha256' in target, f'Unresolved DEB hard link: {name}')
            entry['sha256'] = target['sha256']
            if 'soname' in target:
                entry['soname'] = target['soname']
    return payload


def package_inventory(root, *, profile):
    """Project the already verified selected archives, never the installed state.

    Call verify() first with independently selected producer inputs. This
    projection is expected data for the native inventory test, not evidence
    that packages were installed or that their runtime tests passed.
    """
    require(profile in PACKAGE_PROFILES, f'Unknown package profile: {profile}')
    packages = {}
    payload = {}
    for path in sorted(Path(root).glob('*.deb')):
        path = resource(Path(root), path.name)
        fields = control(path)
        name = fields['Package']
        require(name not in packages, f'Duplicate DEB package role: {name}')
        require(fields['Architecture'] in ('arm64', 'all'), f'Incompatible package architecture: {name}')
        packages[name] = {**fields, 'deb_sha256': digest(path)}
        for filename, entry in deb_payload(path).items():
            require(filename not in payload, f'Conflicting payload ownership: {filename}')
            payload[filename] = {'package': name, **entry}
    require(PACKAGE_PROFILES[profile] <= set(packages),
            f'Missing required profile roles: {sorted(PACKAGE_PROFILES[profile] - set(packages))}')
    return {'schema_version': 1, 'scope': 'expected-installed-packages',
            'profile': profile, 'packages': packages, 'payload': payload}


def installed_controls(names):
    command = ['dpkg-query', '-W', '-f=${Package}\\t${Version}\\t${Architecture}\\t${Status}\\n', *sorted(names)]
    try:
        result = subprocess.check_output(command, text=True)
    except subprocess.CalledProcessError as exc:
        raise ContractError('Required selected packages are not installed (private prefixes do not qualify)') from exc
    rows = {}
    for line in result.splitlines():
        name, version, architecture, status = line.split('\t')
        require(name not in rows, f'Ambiguous installed package architecture: {name}')
        rows[name] = {'Package': name, 'Version': version, 'Architecture': architecture, 'Status': status}
    return rows


def installed_owners(paths):
    owners = {}
    paths = sorted(paths)
    for start in range(0, len(paths), 128):
        try:
            result = subprocess.check_output(['dpkg-query', '-S', '--', *paths[start:start + 128]], text=True)
        except subprocess.CalledProcessError as exc:
            raise ContractError('Installed payload has missing dpkg ownership') from exc
        for line in result.splitlines():
            require(': /' in line, f'Ambiguous dpkg ownership record: {line}')
            owner, path = line.split(': ', 1)
            # binary:Package can include an architecture qualifier. Multiple
            # owners and diversions are not an exact selected installation.
            require(',' not in owner and ' ' not in owner and path not in owners,
                    f'Conflicting installed ownership: {line}')
            owners[path] = owner.split(':', 1)[0]
    return owners


def verify_installed(inventory):
    """Read actual root filesystem + dpkg; deliberately no private-root override."""
    require(inventory.get('scope') == 'expected-installed-packages' and inventory.get('schema_version') == 1,
            'Invalid expected package inventory')
    profile = inventory.get('profile')
    require(profile in PACKAGE_PROFILES, 'Unknown package profile')
    packages = inventory['packages']
    require(PACKAGE_PROFILES[profile] <= set(packages), 'Missing required profile roles')
    actual = installed_controls(packages)
    for name, expected in packages.items():
        require(name in actual and actual[name]['Status'] == 'install ok installed'
                and all(actual[name][key] == expected[key] for key in ('Package', 'Version', 'Architecture')),
                f'Installed package identity differs: {name}')
    payload = inventory['payload']
    owners = installed_owners(payload)
    for filename, expected in payload.items():
        path = Path(filename)
        require(owners.get(filename) == expected['package'], f'Installed payload owner differs: {filename}')
        if 'symlink' in expected:
            require(path.is_symlink() and str(path.readlink()) == expected['symlink'],
                    f'Installed symlink differs: {filename}')
        else:
            require(path.is_file() and not path.is_symlink() and digest(path) == expected['sha256'],
                    f'Installed payload differs: {filename}')
            if 'hardlink' in expected:
                require(path.samefile(expected['hardlink']), f'Installed hard link differs: {filename}')
    return {'status': 'PASS', 'scope': 'actual-dpkg-and-payload', 'profile': profile,
            'packages_verified': len(packages), 'payload_files_verified': len(payload)}


# Platform preservation is separate from optional NEAT component selection.
NATIVE_PLATFORM_ROLES = (
    'appcomplex', 'simaai-common', 'simaai-palette-modalix', 'libcamera',
    'libcamera-tools', 'simaai-a65-plat-tests', 'simaai-base-files-modalix',
    'simaai-gst-plugins', 'simaai-hpi-modalix', 'simaai-log', 'simaai-logd',
    'simaai-mlart-modalix', 'simaai-parser', 'simaai-pcie-ep', 'simaai-traced',
    'simaai-socpipeline', 'simaai-trace',
)


def platform_inventory():
    # Platform generations do not all ship the same native package names.
    # Query the database, not a list whose missing historical names make dpkg
    # fail. Keep explicit absence as part of the preservation contract.
    command = ['dpkg-query', '-W', '-f=${Package}\\t${Version}\\t${Architecture}\\t${Status}\\n']
    try:
        result = subprocess.check_output(command, text=True)
    except subprocess.CalledProcessError as exc:
        raise ContractError('Cannot capture native platform package database') from exc
    packages = {}
    for line in result.splitlines():
        fields = line.split('\t')
        require(len(fields) == 4, 'Malformed native platform package record')
        name, version, architecture, status = fields
        if name not in NATIVE_PLATFORM_ROLES:
            continue
        require(name not in packages, f'Ambiguous native platform package architecture: {name}')
        require(status == 'install ok installed', f'Native platform package is not fully installed: {name}')
        packages[name] = {'Package': name, 'Version': version, 'Architecture': architecture, 'Status': status}
    require('simaai-palette-modalix' in packages, 'Native Palette package is not installed')
    # Some platform cohorts do not ship the historical OTA utility. Preserve
    # its real presence/absence; never install or invent one for this test.
    try:
        palette_files = subprocess.check_output(['dpkg-query', '-L', 'simaai-palette-modalix'], text=True)
    except subprocess.CalledProcessError as exc:
        raise ContractError('Cannot inspect Palette file ownership') from exc
    declared_ota = [Path(name) for name in palette_files.splitlines()
                    if Path(name).is_absolute() and Path(name).name == 'simaai-ota']
    command = shutil.which('simaai-ota')
    require(not declared_ota or command is not None,
            'Palette declares OTA but its executable is missing or unavailable')
    ota = {'ota_present': False, 'ota_path': None, 'ota_owner': None, 'ota_sha256': None}
    if command is not None:
        try:
            path = str(Path(command).resolve(strict=True))
        except OSError as exc:
            raise ContractError('OTA command does not resolve to an installed file') from exc
        require(bool(declared_ota) and any(str(candidate.resolve(strict=True)) == path
                                          for candidate in declared_ota),
                'OTA command is not the Palette-declared executable')
        owner = installed_owners([path])
        require(owner.get(path) == 'simaai-palette-modalix', 'OTA command has unexpected ownership')
        require(Path(path).is_file(), 'OTA command is not a regular file')
        ota = {'ota_present': True, 'ota_path': path, 'ota_owner': owner[path], 'ota_sha256': digest(path)}
    return {'schema_version': 2, 'scope': 'native-platform-before-install',
            'packages': packages, 'absent_packages': sorted(set(NATIVE_PLATFORM_ROLES) - set(packages)), **ota}



def verify_platform(before, selected_packages):
    require(before.get('schema_version') == 2 and before.get('scope') == 'native-platform-before-install',
            'Invalid pre-install platform inventory')
    packages = before.get('packages', {})
    absent = before.get('absent_packages')
    require(isinstance(packages, dict) and isinstance(absent, list)
            and all(isinstance(name, str) for name in absent)
            and len(absent) == len(set(absent)) and not set(packages).intersection(absent)
            and set(packages).union(absent) == set(NATIVE_PLATFORM_ROLES),
            'Incomplete platform baseline presence/absence coverage')
    require('simaai-palette-modalix' in packages and all(
        isinstance(value, dict) and value.get('Package') == name
        and value.get('Status') == 'install ok installed'
        and value.get('Version') and value.get('Architecture') for name, value in packages.items()),
        'Invalid installed native platform baseline')
    ota_fields = ('ota_present', 'ota_path', 'ota_owner', 'ota_sha256')
    require(type(before.get('ota_present')) is bool, 'Missing OTA presence authority')
    if before['ota_present']:
        require(isinstance(before.get('ota_path'), str) and Path(before['ota_path']).is_absolute()
                and before.get('ota_owner') == 'simaai-palette-modalix'
                and isinstance(before.get('ota_sha256'), str)
                and re.fullmatch(r'[0-9a-f]{64}', before['ota_sha256']), 'Invalid present OTA baseline')
    else:
        require(all(key in before and before[key] is None for key in ota_fields[1:]),
                'Invalid absent OTA baseline')
    current = platform_inventory()
    for name in NATIVE_PLATFORM_ROLES:
        # A provider explicitly present in the verified selected artifact set
        # may be upgraded; verify_installed independently checks its exact bytes.
        if name not in selected_packages:
            require(current['packages'].get(name) == packages.get(name),
                    f'Unselected platform package changed: {name}')
    require(all(current[key] == before[key] for key in ota_fields),
            'OTA command changed during NEAT installation')
    return {'status': 'PASS', 'scope': 'native-platform-preservation'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dist', type=Path, default=Path('dist'))
    parser.add_argument('--expected-manifest', type=Path)
    parser.add_argument('--expected-receipt', type=Path)
    parser.add_argument('--expected-deb-dir', type=Path, action='append', default=[])
    parser.add_argument('--package-profile', choices=tuple(PACKAGE_PROFILES),
                        help='Required component roles for installed-package qualification')
    parser.add_argument('--write-package-inventory', type=Path,
                        help='Write expected controls/payload from verified archives, not a PASS report')
    parser.add_argument('--verify-installed', action='store_true',
                        help='Additionally compare actual root/dpkg state, never a private prefix')
    parser.add_argument('--capture-platform-baseline', type=Path,
                        help='Capture actual native/OTA state before installation, without inspecting a dist')
    parser.add_argument('--platform-baseline', type=Path,
                        help='Compare actual native/OTA state after --verify-installed')
    args = parser.parse_args()
    if args.capture_platform_baseline:
        if args.verify_installed or args.write_package_inventory or args.platform_baseline:
            parser.error('Platform baseline capture is a separate pre-install action')
        baseline = platform_inventory()
        with args.capture_platform_baseline.open('x') as stream:
            json.dump(baseline, stream, indent=2, sort_keys=True)
            stream.write('\n')
        return
    if args.platform_baseline and not args.verify_installed:
        parser.error('--platform-baseline requires --verify-installed')
    from sima_cli.install.metadata_validator import validate_metadata
    if (args.write_package_inventory or args.verify_installed) and not args.package_profile:
        parser.error('Installed qualification requires explicit --package-profile')
    result = verify(args.dist, validate_metadata=validate_metadata,
                    expected_manifest=args.expected_manifest,
                    expected_receipt=args.expected_receipt,
                    expected_deb_dirs=args.expected_deb_dir)
    result['scope'] = 'artifact-set'
    if args.write_package_inventory or args.verify_installed:
        inventory = package_inventory(args.dist, profile=args.package_profile)
        if args.write_package_inventory:
            with args.write_package_inventory.open('x') as stream:
                json.dump(inventory, stream, indent=2, sort_keys=True)
                stream.write('\n')
        if args.verify_installed:
            result['installed'] = verify_installed(inventory)
            if args.platform_baseline:
                result['platform'] = verify_platform(json.loads(args.platform_baseline.read_text()), inventory['packages'])
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
