#!/usr/bin/env python3
"""Compose separate owning native DEBs; Core never absorbs lower payload files."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_provider(directory):
    spec = importlib.util.spec_from_file_location('scheduler_native', Path(directory) / 'native_artifacts.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def compose(selection_path, output):
    selection = json.loads(Path(selection_path).read_text())
    candidate = Path(selection['candidate'])
    if sha(candidate / 'host-candidate.json') != selection['candidate_sha256']:
        raise ValueError('selected native candidate changed')
    authority = json.loads((candidate / 'host-candidate.json').read_text())
    # The candidate hash is already bound to the publisher selection. Check its
    # helper resource BEFORE importing executable producer-owned Python.
    helper = authority.get('resources', {}).get('native_artifacts.py', {})
    tool = candidate / 'native_artifacts.py'
    if (tool.is_symlink() or not tool.is_file() or sha(tool) != helper.get('sha256')
            or tool.stat().st_size != helper.get('size')):
        raise ValueError('selected native helper changed')
    provider = load_provider(candidate)
    data = provider.verify(candidate, selection['distro'], selection['architecture'], selection['commit'])
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    resources = dict(data['resources'])
    resources['host-candidate.json'] = {'sha256': sha(candidate / 'host-candidate.json'),
                                      'size': (candidate / 'host-candidate.json').stat().st_size}
    for name, record in resources.items():
        target = output / name
        if target.exists() and (target.is_symlink() or sha(target) != record['sha256']):
            raise ValueError('native closure collides with Core distribution: ' + name)
        shutil.copy2(candidate / name, target)
    receipt = {key: selection[key] for key in ('schema_version', 'profile', 'distro', 'architecture', 'branch', 'commit', 'package_version', 'candidate_sha256')}
    receipt['resources'] = resources
    (output / 'scheduler-native-selection.json').write_text(json.dumps(receipt, indent=2) + '\n')
    shutil.copy2(__file__, output / 'pcie_delivery.py')


def verify(directory, runtime=None, development=None, runtime_only=False):
    directory = Path(directory)
    selection = json.loads((directory / 'scheduler-native-selection.json').read_text())
    if selection.get('profile') != 'pcie-host' or sha(directory / 'host-candidate.json') != selection['candidate_sha256']:
        raise ValueError('native selection identity changed')
    for name, record in selection['resources'].items():
        if (Path(name).name != name or (directory / name).is_symlink()
                or not (directory / name).is_file() or sha(directory / name) != record['sha256']
                or (directory / name).stat().st_size != record['size']):
            raise ValueError('selected native resource changed: ' + name)
    provider = load_provider(directory)
    data = provider.verify(directory, selection['distro'], selection['architecture'], selection['commit'])
    arch = selection['architecture']
    version = data['packages']['runtime']['payload']['control']['Version']
    if version != selection['package_version']:
        raise ValueError('native package version changed')
    rt = Path(runtime) if runtime else next(iter(directory.glob(f'sima-pcie-host_*_{arch}.deb')), None)
    dev = Path(development) if development else next(iter(directory.glob(f'sima-pcie-host-dev_*_{arch}.deb')), None)
    if not rt or not rt.is_file() or (not runtime_only and (not dev or not dev.is_file())):
        raise ValueError('complete native Core package pair is required')
    payloads = {role: row['payload'] for role, row in data['packages'].items()}
    for role, path, name, dependency in [('core_runtime', rt, 'sima-pcie-host', 'simaai-scheduler-pcie-host'),
                                         ('core_development', dev, 'sima-pcie-host-dev', 'simaai-scheduler-pcie-host-dev')]:
        if role == 'core_development' and runtime_only: continue
        payload = provider.inspect_deb(path, arch)
        if payload['control']['Package'] != name:
            raise ValueError('Core package role mismatch')
        depends = subprocess.check_output(['dpkg-deb', '-f', str(path), 'Depends'], text=True)
        if f'{dependency} (= {version})' not in [x.strip() for x in depends.split(',')]:
            raise ValueError('Core must pin the selected Scheduler native package')
        payloads[role] = payload
    if not runtime_only and payloads["core_runtime"]["control"]["Version"] != payloads["core_development"]["control"]["Version"]:
        raise ValueError("Core runtime/development versions differ")
    provider.check_ownership(payloads)
    # Platform source is a provenance/build resource, not a Core-owned binary
    # package or an implicit driver-installation request.
    roles = ('runtime',) if runtime_only else ('runtime', 'development')
    packages = [str((directory / data['packages'][role]['resource']).resolve()) for role in roles]
    packages += [str(rt.resolve())] + ([] if runtime_only else [str(dev.resolve())])
    return selection, packages


def stage(directory, destination):
    """Stage installed exports for a native consumer without a system install."""
    selection, packages = verify(directory)
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError('native staging requires a fresh private destination')
    provider = load_provider(directory)
    provider.stage(Path(directory), destination, selection['distro'],
                   selection['architecture'], selection['commit'])
    # verify() returns the owning Core pair last. Scheduler stages its own
    # packages and external userspace provider; Core adds only its own payload.
    for package in packages[-2:]:
        subprocess.run(['dpkg-deb', '-x', package, str(destination)], check=True)
    return selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('compose'); c.add_argument('--selection', required=True); c.add_argument('--output', required=True)
    s = sub.add_parser('stage'); s.add_argument('--directory', required=True); s.add_argument('--destination', required=True)
    for command in ('verify', 'install-list'):
        v = sub.add_parser(command); v.add_argument('--directory', required=True); v.add_argument('--runtime'); v.add_argument('--development'); v.add_argument('--runtime-only', action='store_true')
        if command == 'install-list':
            v.add_argument('--extra-dependency', action='append', default=[])
    args = p.parse_args()
    if args.command == 'compose': compose(args.selection, args.output); return
    if args.command == 'stage':
        selection = stage(args.directory, args.destination)
        print(json.dumps({'status': 'STAGED_NATIVE_PACKAGES', 'selection': selection}))
        return
    selection, packages = verify(args.directory, args.runtime, args.development, args.runtime_only)
    if args.command == 'install-list':
        release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
        distro = {'22.04': 'ubuntu22', '24.04': 'ubuntu24'}.get(release.get('VERSION_ID', '').strip('"'))
        if (release.get('ID', '').strip('"') != 'ubuntu' or distro != selection['distro']
                or subprocess.check_output(['dpkg', '--print-architecture'], text=True).strip() != selection['architecture']):
            raise ValueError('installation host differs from exact native tuple')
        # Scheduler owns external-provider admission. Core supplies the whole
        # transaction so its dependencies cannot silently change Platform/DKMS.
        provider = load_provider(args.directory)
        provider.validate_install_plan(
            args.directory, packages, selection['distro'], selection['architecture'],
            selection['commit'], extra_dependencies=args.extra_dependency)
        print('\n'.join(packages))
    else: print(json.dumps({'status': 'PASS_NATIVE_PACKAGE_CLOSURE', 'packages': packages, 'selection': selection}, sort_keys=True))


if __name__ == '__main__': main()
