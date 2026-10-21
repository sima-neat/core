#!/usr/bin/env python3
"""Authenticate Core's installed Scheduler tooling; delegate Platform policy.

This is a transport boundary, not an ABI or lower-provider classifier. Receipt
paths never select executable code. The incoming Scheduler DEB and its sealed
installed tooling are the only source of the admission implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
import tarfile
import tempfile


class AdmissionError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AdmissionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def declared(receipt):
    return ('PlatformRuntime' in receipt.get('external_dependencies', {}) or
            'platform_runtime_packages' in receipt.get('runner', {}).get('buildinfo', {}))


def extract_scheduler(archive, destination):
    """Extract data only after rejecting traversal, special files and link parents."""
    with tempfile.TemporaryFile() as stream:
        subprocess.run(['dpkg-deb', '--fsys-tarfile', str(archive)], stdout=stream, check=True)
        stream.seek(0)
        with tarfile.open(fileobj=stream) as bundle:
            members = {}
            for member in bundle.getmembers():
                path = PurePosixPath(member.name)
                require(not path.is_absolute() and '..' not in path.parts and '\\' not in member.name,
                        'Unsafe Scheduler archive path')
                if str(path) == '.':
                    require(member.isdir(), 'Invalid Scheduler archive root')
                    continue
                name = str(path)
                require(name not in members and not member.mode & 0o7000,
                        'Duplicate or privileged Scheduler archive member')
                require(member.isfile() or member.isdir() or member.issym(),
                        'Unsupported Scheduler archive member')
                if member.issym():
                    target = posixpath.normpath(str(path.parent / member.linkname))
                    require(member.linkname and not member.linkname.startswith('/')
                            and '\\' not in member.linkname and target != '..' and not target.startswith('../'),
                            'Escaping Scheduler archive link')
                members[name] = member
            links = {PurePosixPath(name) for name, member in members.items() if member.issym()}
            require(not any(any(parent in links for parent in PurePosixPath(name).parents)
                            for name in members), 'Scheduler archive traverses a symlink')
            destination.mkdir()
            for name, member in members.items():
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if member.isdir():
                    target.mkdir(exist_ok=True)
                elif member.issym():
                    target.symlink_to(member.linkname)
                else:
                    with bundle.extractfile(member) as source, target.open('xb') as output:
                        for chunk in iter(lambda: source.read(1024 * 1024), b''):
                            output.write(chunk)
                    target.chmod(member.mode & 0o777)


def authenticate_tool(prefix, selected):
    path = prefix / 'share/SimaScheduler/buildinfo.json'
    require(path.is_file() and not path.is_symlink(), 'Missing Scheduler tooling receipt')
    info = json.loads(path.read_text())
    require(info == selected, 'Incoming Scheduler receipt differs from selected distribution')
    identity = {key: value for key, value in info.items() if key != 'package_sha256'}
    require(info.get('component') == 'scheduler' and info.get('schema_version') == 1
            and canonical(identity) == info.get('package_sha256'), 'Invalid Scheduler tooling receipt identity')
    inventory = info.get('installed_files', {})
    require(isinstance(inventory, dict) and canonical(inventory) == info.get('installed_payload_sha256'),
            'Invalid Scheduler installed inventory digest')
    root = prefix / 'share/SimaScheduler/tools'
    require(root.is_dir() and not root.is_symlink(), 'Missing Scheduler installed tooling')
    expected = {name: item for name, item in inventory.items()
                if name.startswith('share/SimaScheduler/tools/')}
    actual = {}
    for item in root.rglob('*'):
        name = item.relative_to(prefix).as_posix()
        require(not item.is_symlink(), 'Symlinks are not allowed in authenticated Scheduler tooling')
        if item.is_file():
            actual[name] = {'kind': 'file', 'sha256': sha(item), 'mode': item.stat().st_mode & 0o7777}
    require(expected and actual == expected, 'Scheduler tooling differs from authenticated inventory')
    tool = root / 'platform_admission.py'
    require(tool.is_file() and not tool.is_symlink(), 'Selected Scheduler lacks Platform admission tooling')
    return tool


def firmware_control(result):
    """Validate the authenticated owner's firmware declaration, not its policy."""
    firmware = result.get('firmware')
    require(isinstance(firmware, dict) and isinstance(firmware.get('required'), bool)
            and 'debian_package' in firmware, 'Missing or invalid Scheduler firmware classification')
    control = firmware['debian_package']
    if not firmware['required']:
        require(control is None, 'Unexpected firmware control for a non-EV cohort')
        return None
    require(isinstance(control, dict)
            and isinstance(control.get('Package'), str)
            and re.fullmatch(r'[a-z0-9][a-z0-9+.-]*', control['Package'])
            and isinstance(control.get('Version'), str)
            and re.fullmatch(r'[0-9][A-Za-z0-9.+:~\-]*', control['Version'])
            and control.get('Architecture') == 'arm64'
            and isinstance(control.get('deb_sha256'), str)
            and re.fullmatch(r'[0-9a-f]{64}', control['deb_sha256']),
            'Invalid required firmware package identity')
    return control


def invoke(directory, action, *, installed=False, simulation=None, repair_incomplete=False,
           resume_incomplete=False):
    require(not installed or action == 'check', 'Installed verification requires check')
    require(not resume_incomplete or (repair_incomplete and action == 'transaction'),
            'Incomplete resume requires repair transaction admission')
    require(not repair_incomplete or action == 'transaction',
            'Incomplete package repair requires transaction admission')
    if action == 'transaction':
        require(simulation is not None, 'Transaction admission requires an APT simulation')
        simulation = Path(simulation)
        require(simulation.is_file() and not simulation.is_symlink(),
                'APT simulation must be a regular file')
    else:
        require(simulation is None, 'APT simulation requires transaction admission')
    root = Path(directory).resolve(strict=True)
    receipt_path = root / 'resolved-deps-manifest.json'
    require(receipt_path.is_file() and not receipt_path.is_symlink(), 'Missing selected dependency receipt')
    receipt = json.loads(receipt_path.read_text())
    selected = receipt.get('scheduler', {}).get('buildinfo')
    require(isinstance(selected, dict), 'Missing selected Scheduler build receipt')
    archives = []
    for archive in sorted(root.glob('*.deb')):
        require(archive.is_file() and not archive.is_symlink(), 'Nonregular incoming DEB')
        role = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Package'], text=True).strip()
        if role == 'simaai-scheduler':
            archives.append(archive)
    require(len(archives) == 1, 'Exactly one incoming Scheduler DEB is required')
    with tempfile.TemporaryDirectory(prefix='core-scheduler-admission-') as scratch:
        prefix = Path(scratch) / 'package'
        extract_scheduler(archives[0], prefix)
        tool = authenticate_tool(prefix / 'usr', selected)
        command = [sys.executable, '-B', str(tool), action, '--directory', str(root)]
        if installed:
            command.append('--installed')
        if simulation is not None:
            command.extend(['--simulation', str(simulation.resolve(strict=True))])
        if repair_incomplete:
            command.append('--repair-incomplete')
        if resume_incomplete:
            command.append('--resume-incomplete')
        # No inherited source PYTHONPATH may override authenticated helper imports.
        env = {key: value for key, value in os.environ.items() if key not in ('PYTHONPATH', 'PYTHONHOME')}
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        result = json.loads(subprocess.check_output(command, text=True, env=env))
    require(isinstance(result, dict) and result.get('schema_version') == 1,
            'Invalid Scheduler Platform admission response')
    firmware_control(result)
    return result


def optional_admission(directory, action, **options):
    root = Path(directory).resolve(strict=True)
    # The actual incoming owner determines whether this is a split cohort.
    # Unverified top-level receipt flags must not disable authentication.
    schedulers = []
    for archive in root.glob('*.deb'):
        require(archive.is_file() and not archive.is_symlink(), 'Nonregular incoming DEB')
        role = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Package'], text=True).strip()
        if role == 'simaai-scheduler':
            schedulers.append(archive)
    if schedulers:
        return {**invoke(root, action, **options), 'split_cohort': True}
    receipt_path = root / 'resolved-deps-manifest.json'
    if receipt_path.exists():
        require(receipt_path.is_file() and not receipt_path.is_symlink(), 'Invalid dependency receipt')
        receipt = json.loads(receipt_path.read_text())
        require(not declared(receipt) and not receipt.get('scheduler'),
                'Selected split distribution is missing its Scheduler archive')
    # Legacy monolithic distributions have no incoming Scheduler owner.
    require(not options.get('repair_incomplete'),
            'Unhealthy APT state requires an authenticated split Scheduler cohort')
    return {'schema_version': 1, 'platform_packages': {}, 'packages': {}, 'providers': {},
            'split_cohort': False, 'firmware': {'required': False, 'debian_package': None}}


def optional_board_check(directory, *, installed=False):
    return optional_admission(directory, 'check', installed=installed)


def same_providers(before, after):
    for key in ('platform_packages', 'packages', 'providers'):
        require(isinstance(before.get(key), dict) and before.get(key) == after.get(key),
                'Admitted Platform provider identity changed: ' + key)
    firmware_control(before)
    firmware_control(after)
    require(before['firmware'] == after['firmware'], 'Admitted firmware identity changed')


def verify_plan(admission, simulation):
    packages = admission.get('platform_packages')
    require(isinstance(packages, dict), 'Missing admitted Platform package identities')
    for line in Path(simulation).read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] in ('Inst', 'Remv'):
            require(fields[1].split(':', 1)[0] not in packages,
                    'APT would replace or remove an admitted Platform provider: ' + fields[1])


def retirement_plan(result):
    """Render an authenticated proposal for the installer's existing guards.

    This validates transport identities only. Scheduler owns installed-state,
    relationship and file-ownership eligibility; Core retains removal policy.
    """
    require(result.get('status') == 'PASS'
            and result.get('scope') == 'read-only-package-retirement-proposal'
            and result.get('repair_incomplete') is True,
            'Invalid Scheduler retirement proposal')
    retirements = result.get('retirements')
    require(isinstance(retirements, dict), 'Missing Scheduler retirement identities')
    lines, names = [], set()
    for qualified, state in sorted(retirements.items()):
        require(isinstance(qualified, str) and isinstance(state, dict),
                'Invalid retirement identity')
        match = re.fullmatch(r'([a-z0-9][a-z0-9+.-]*)(?::([a-z0-9][a-z0-9-]*))?', qualified)
        version, architecture = state.get('version'), state.get('architecture')
        require(match and match[1] not in names
                and isinstance(version, str) and re.fullmatch(r'[0-9][A-Za-z0-9.+:~\-]*', version)
                and isinstance(architecture, str) and re.fullmatch(r'[a-z0-9][a-z0-9-]*', architecture)
                and (not match[2] or match[2] == architecture)
                and state.get('status') == 'install ok installed',
                'Invalid or ambiguous retirement identity: ' + qualified)
        names.add(match[1])
        lines.append(f'Remv {qualified} [{version}]\n')
    return ''.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('describe', 'check', 'check-if-declared', 'preflight-if-declared',
                                          'transaction-if-declared', 'retirements', 'verify-plan'))
    parser.add_argument('--directory', type=Path, default=Path.cwd())
    parser.add_argument('--installed', action='store_true')
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--simulation', type=Path)
    parser.add_argument('--repair-incomplete', action='store_true')
    parser.add_argument('--resume-incomplete', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    require(not args.resume_incomplete or (args.repair_incomplete
            and args.action == 'transaction-if-declared'),
            'Incomplete resume requires repair transaction admission')
    require(not args.repair_incomplete or args.action == 'transaction-if-declared',
            'Incomplete package repair requires transaction admission')
    if args.action == 'verify-plan':
        require(args.baseline and args.simulation, 'Plan verification requires admission and simulation')
        verify_plan(json.loads(args.baseline.read_text()), args.simulation)
        return
    if args.action == 'retirements':
        require(not args.simulation and not args.installed and not args.baseline,
                'Retirement proposal does not accept a simulation or provider comparison')
        result = invoke(args.directory, 'retirements')
    elif args.action == 'transaction-if-declared':
        require(args.simulation and not args.installed and not args.baseline,
                'Transaction admission requires a simulation, not installed/provider comparison')
        result = optional_admission(args.directory, 'transaction', simulation=args.simulation,
                                    repair_incomplete=args.repair_incomplete,
                                    resume_incomplete=args.resume_incomplete)
    elif args.action == 'preflight-if-declared':
        require(not args.installed and not args.simulation,
                'Preflight does not accept installed verification or a simulation')
        result = optional_admission(args.directory, 'preflight')
    else:
        require(not args.simulation, 'Simulation requires plan or transaction admission')
        result = (optional_board_check(args.directory, installed=args.installed)
                  if args.action == 'check-if-declared'
                  else invoke(args.directory, args.action, installed=args.installed))
    if args.baseline:
        same_providers(json.loads(args.baseline.read_text()), result)
    data = (retirement_plan(result) if args.action == 'retirements'
            else json.dumps(result, indent=2, sort_keys=True) + '\n')
    if args.output:
        with args.output.open('x') as stream:
            stream.write(data)
    else:
        print(data, end='')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print('Scheduler Platform admission failed: ' + str(error), file=sys.stderr)
        raise SystemExit(1)
