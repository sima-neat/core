#!/usr/bin/env python3
"""One-target offline deployment, called by install.py inside a network namespace."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
from zipfile import ZipFile

from assemble import verify_bundle


def run(args, **kwargs):
    return subprocess.run(args, check=True, timeout=1800, **kwargs)


def output(args):
    return run(args, text=True, capture_output=True).stdout.strip()


def configure(repo, state):
    for directory in ('lists/partial', 'cache/partial', 'empty'):
        (state / directory).mkdir(parents=True, exist_ok=True)
    (state / 'sources.list').write_text(f'deb [trusted=yes] file:{repo} ./\n')
    settings = {
        'Dir::Etc::main': '/dev/null', 'Dir::Etc::parts': str(state/'empty'),
        'Dir::Etc::sourcelist': str(state/'sources.list'),
        'Dir::Etc::sourceparts': str(state/'empty'),
        'Dir::Etc::preferences': '/dev/null', 'Dir::Etc::preferencesparts': str(state/'empty'),
        'Dir::State::lists': str(state/'lists'), 'Dir::Cache::archives': str(state/'cache'),
        # The flat bundle repository supplies Packages, not compressed variants.
        'Acquire::IndexTargets::deb::Packages::CompressionTypes': 'uncompressed',
        'Acquire::Languages': 'none', 'APT::Get::List-Cleanup': '0',
        'APT::Get::AllowUnauthenticated': 'false', 'APT::Sandbox::User': 'root',
    }
    (state/'config').write_text(''.join(f'{key} {json.dumps(value)};\n' for key, value in settings.items()))
    os.environ['APT_CONFIG'] = str(state/'config')


@contextmanager
def private_apt(repo):
    previous = os.environ.get('APT_CONFIG')
    with tempfile.TemporaryDirectory(prefix='neat-offline-apt-') as tmp:
        try:
            state = Path(tmp)
            configure(repo, state)
            yield state
        finally:
            # Error-reporting hooks can import python-apt after this scope exits.
            # Never leave APT_CONFIG pointing at a deleted temporary directory.
            if previous is None:
                os.environ.pop('APT_CONFIG', None)
            else:
                os.environ['APT_CONFIG'] = previous


def missing_prerequisites():
    required = ('python3-venv', 'python3-pip', 'gstreamer1.0-tools', 'ca-certificates')
    missing = []
    for package in required:
        result = subprocess.run(['dpkg-query', '-W', '-f=${db:Status-Status}', package],
                                capture_output=True, text=True, timeout=30)
        if result.returncode or result.stdout.strip() != 'installed':
            missing.append(package)
    return missing


def environment(target, role):
    row = json.loads((target/'target.json').read_text())
    release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    actual = (release['ID'].strip('"'), release['VERSION_ID'].strip('"'), output(['dpkg', '--print-architecture']))
    if actual != (row['os_id'], row['os_version'], row['architecture']):
        raise ValueError(f'{role}: incompatible OS/architecture: {actual}')
    expected = json.loads((target/'dependencies.json').read_text())['python_abi']
    import sysconfig
    if sysconfig.get_config_var('SOABI') != expected:
        raise ValueError('Python ABI differs from bundled wheels')
    if role == 'card' and not re.search(r'\b3\.0\.', Path('/etc/buildinfo').read_text()):
        raise ValueError('Card requires Platform 3.0; complete system conversion first')
    audit = output(['dpkg', '--audit'])
    if audit:
        raise ValueError('Repair package database before installation: ' + audit)


def check_space(target):
    # Archives are already staged, potentially on NVMe or tmpfs. Their current
    # allocation is reflected in disk_usage; do not charge it to root again.
    installed = 0
    for package in (target/'repo').glob('*.deb'):
        size = int(output(['dpkg-deb', '-f', str(package), 'Installed-Size']) or '0') * 1024
        if not size:
            # Some producer packages omit Installed-Size. Never treat their
            # installed contents as free; read the archive without extracting it.
            with tempfile.TemporaryFile() as stream:
                run(['dpkg-deb', '--fsys-tarfile', str(package)], stdout=stream)
                stream.seek(0)
                with tarfile.open(fileobj=stream) as archive:
                    size = sum((entry.size + 4095) // 4096 * 4096
                               for entry in archive if entry.isfile())
        installed += size
    for wheel in (target/'wheels').glob('*.whl'):
        with ZipFile(wheel) as archive:
            installed += sum((entry.file_size + 4095) // 4096 * 4096
                             for entry in archive.infolist())
    # Reserve the full dependency cohort even if already installed, plus unpack/
    # replacement and Python staging space, and 128 MiB for metadata/venv growth.
    required = 3 * installed + 128 * 1024**2
    free = shutil.disk_usage('/').free
    if free < required:
        mib = 1024**2
        raise ValueError(f'Insufficient free root filesystem space: {free // mib} MiB available, '
                         f'{(required + mib - 1) // mib} MiB required; '
                         f'free at least {(required - free + mib - 1) // mib} MiB on /. '
                         'Moving the already-staged bundle does not free installation space.')


def execute(root, role, phase):
    verify_bundle(root, 'card' if role == 'card' and not (root/'host').exists() else None)
    target = root/role
    environment(target, role)
    if phase != 'verify':
        check_space(target)
    # Ignore caller configuration, including pip extra-index-url/find-links.
    for key in list(os.environ):
        if key.startswith(('PIP_', 'NEAT_', 'SIMAPCIE_', 'PYNEAT_')) or key in ('SYSROOT', 'ELXR_SDK_RELEASE_FILE'):
            del os.environ[key]
    venv = '/opt/sima/neat/' + ('pyneatpcie' if role == 'host' else 'pyneat')
    os.environ.update({
        'DEBIAN_FRONTEND': 'noninteractive', 'PYTHONDONTWRITEBYTECODE': '1',
        'PIP_CONFIG_FILE': '/dev/null', 'PIP_NO_INDEX': '1', 'PIP_NO_CACHE_DIR': '1',
        'PIP_FIND_LINKS': str(target/'wheels'), 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        'PYNEAT_VENV_DIR': venv, 'NEAT_INSTALLER_APT_UPDATE': 'OFF',
        'NEAT_INSTALLER_INSTALL_CODEX_SKILL': 'OFF', 'NEAT_INSTALLER_INSTALL_CLAUDE_SKILL': 'OFF',
        'SIMAPCIE_INSTALL_CODEX_SKILL': 'OFF', 'SIMAPCIE_INSTALL_CLAUDE_SKILL': 'OFF',
    })
    with private_apt(target/'repo') as state:
        run(['apt-get', 'update'])
        native = sorted((target/'payload').glob('*.deb'))
        prerequisites = missing_prerequisites()
        # Card upgrades may retire legacy Neat packages. The published lifecycle
        # validates every removal and the complete Scheduler transaction below.
        # Both published lifecycles install the explicitly selected bundle versions,
        # even when Debian orders them below an installed development build.
        policy = ['--allow-downgrades', '--no-remove' if role == 'host' else '--reinstall']
        args = ['apt-get', *policy, '--no-install-recommends', '-y', 'install',
                *map(str, native)]
        if phase == 'verify':
            verify(target, role, venv)
            return
        plan = state/'plan.txt'
        with plan.open('w') as stream:
            run([*args, *prerequisites, '--simulate'], stdout=stream)
        # Producer admission checks the whole local transaction before mutations.
        lifecycle = ['bash', str(root/'lifecycle.sh'), role, 'preflight', str(target/'payload'), str(plan)]
        run(lifecycle)
        if phase == 'preflight':
            return
        if prerequisites:
            run(['apt-get', '--no-remove', '--no-install-recommends', '-y', 'install', *prerequisites])
        run(['python3', '-m', 'venv', venv])
        wheels = sorted((target/'payload').glob('*.whl'))
        run([venv+'/bin/python', '-m', 'pip', 'install', '--force-reinstall',
             '--only-binary=:all:', *map(str, wheels)])
        # Prerequisite installation changed dpkg state; admission must inspect
        # a fresh plan rather than treating already-installed prerequisites as new.
        with plan.open('w') as stream:
            run([*args, '--simulate'], stdout=stream)
        lifecycle[3] = 'install'
        run(lifecycle)
        verify(target, role, venv)


def verify(target, role, venv):
    run(['apt-get', 'check'])
    if output(['dpkg', '--audit']):
        raise ValueError('Post-install package audit failed')
    for deb in sorted((target/'payload').glob('*.deb')):
        package = output(['dpkg-deb', '-f', str(deb), 'Package'])
        version = output(['dpkg-deb', '-f', str(deb), 'Version'])
        installed = output(['dpkg-query', '-W', '-f=${db:Status-Status} ${Version}', package])
        if installed != 'installed ' + version:
            raise ValueError('Installed package mismatch: ' + package)
    run([venv+'/bin/python', '-m', 'pip', 'check'])
    run([venv+'/bin/python', '-c', 'import ' + ('pyneatpcie' if role == 'host' else 'pyneat')])
    if role == 'host':
        run(['gst-inspect-1.0', 'neatpciehost'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['host', 'card'])
    parser.add_argument('phase', choices=['preflight', 'install', 'verify'])
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('requires root')
    if any(name != 'lo' for _, name in socket.if_nameindex()):
        parser.error('deployment requires an isolated network namespace; use install.py')
    execute(Path(__file__).resolve().parent, args.role, args.phase)


if __name__ == '__main__':
    main()
