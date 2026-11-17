#!/usr/bin/env python3
"""One-target offline deployment, called by install.py inside a network namespace."""
import argparse
from contextlib import contextmanager
import json
import os
import pwd
from pathlib import Path
import re
import shutil
import shlex
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
    (state / 'sources.list').write_text(f'deb [trusted=yes] {repo.resolve().as_uri()} ./\n')
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


def check_space(target, plan=None):
    # Archives are already staged, potentially on NVMe or tmpfs. Their current
    # allocation is reflected in disk_usage; do not charge it to root again.
    selected = None if plan is None else {
        line.split()[1].split(':')[0] for line in plan.read_text().splitlines()
        if line.startswith('Inst ')
    }
    accounted = set()
    installed = 0
    for package in (target/'repo').glob('*.deb'):
        if selected is not None:
            name = output(['dpkg-deb', '-f', str(package), 'Package'])
            if name not in selected:
                continue
            accounted.add(name)
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
    if selected is not None and selected - accounted:
        raise ValueError('Planned packages missing from offline repository: ' + ', '.join(sorted(selected - accounted)))
    for wheel in (target/'wheels').glob('*.whl'):
        with ZipFile(wheel) as archive:
            installed += sum((entry.file_size + 4095) // 4096 * 4096
                             for entry in archive.infolist())
    # Reserve every package APT will unpack, including explicit reinstalls, plus
    # replacement and Python staging space, and 128 MiB for metadata/venv growth.
    required = 3 * installed + 128 * 1024**2
    free = shutil.disk_usage('/').free
    if free < required:
        mib = 1024**2
        raise ValueError(f'Insufficient free root filesystem space: {free // mib} MiB available, '
                         f'{(required + mib - 1) // mib} MiB required; '
                         f'free at least {(required - free + mib - 1) // mib} MiB on /. '
                         'Moving the already-staged bundle does not free installation space.')


def python_location(role, nvme=Path('/media/nvme')):
    """Use the online installer's defaults for the user who invoked sudo."""
    user = pwd.getpwuid(int(os.environ.get('SUDO_UID', os.getuid())))
    home = Path(user.pw_dir)
    name = 'pyneatpcie' if role == 'host' else 'pyneat'
    path = home/name
    # Keep an existing home environment (including a link) at its current location.
    if path.exists() or path.is_symlink():
        return user, path
    if role == 'card' and nvme.is_dir():
        if user.pw_uid == os.getuid():
            writable = os.access(nvme, os.W_OK)
        else:
            writable = subprocess.run(
                ['test', '-w', str(nvme)], user=user.pw_uid, group=user.pw_gid,
                extra_groups=os.getgrouplist(user.pw_name, user.pw_gid),
                timeout=30).returncode == 0
        if writable:
            path = nvme/name
    return user, path


def finish_python_location(user, path, role):
    # Python installation runs as root because staged bundle files can be private.
    # Restore ownership so subsequent online updates can run as the same user.
    actual = path.resolve()
    for directory, directories, files in os.walk(actual):
        for entry in [Path(directory), *(Path(directory)/name for name in directories + files)]:
            os.chown(entry, user.pw_uid, user.pw_gid, follow_symlinks=False)
    home_path = Path(user.pw_dir)/'pyneat'
    if role == 'card' and path != home_path:
        if home_path.is_symlink() and home_path.resolve() == actual:
            return
        if home_path.exists() or home_path.is_symlink():
            raise ValueError(f'Refusing to replace existing Python environment at {home_path}; '
                             f'the installed environment is at {path}')
        home_path.symlink_to(path)
        os.chown(home_path, user.pw_uid, user.pw_gid, follow_symlinks=False)


def check_python_environment(path, stage):
    """Report the environment and pip's dependency diagnostics without repairing it."""
    command = [str(path/'bin/python'), '-m', 'pip', 'check']
    try:
        result = run(command, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError) as exc:
        details = '\n'.join(value.strip() for value in
                            (getattr(exc, 'stdout', ''), getattr(exc, 'stderr', '')) if value)
        if not details:
            details = str(exc)
        context = ('Existing Python environment is broken; installation stopped before package changes'
                   if stage == 'before installation' else
                   'Python dependency verification failed; installation may have changed packages')
        raise ValueError(f'{context}. Environment: {path} ({stage}).\n{details}\n'
                         f'Repair this environment and rerun {shlex.join(command)}. '
                         'Offline repairs require the matching wheels. To discard its packages and files, '
                         'explicitly rerun the installer with --recreate-venv. '
                         'No automatic recreation or rollback was performed.') from None
    if result and result.stdout:
        print(result.stdout.strip())


def prepare_python_environment(path, recreate=False):
    """Reuse checked environments; only clear when the caller explicitly opts in."""
    if not recreate and (path.exists() or path.is_symlink()):
        return
    command = ['python3', '-m', 'venv']
    if recreate:
        print(f'Recreating Python environment at {path}; existing packages and files will be removed.')
        command.append('--clear')
    # venv rejects a symlink as its root; retain the link and use its target.
    run([*command, str(path.resolve())])


def execute(root, role, phase, recreate_venv=False):
    verify_bundle(root, 'card' if role == 'card' and not (root/'host').exists() else None)
    target = root/role
    environment(target, role)
    # Ignore caller configuration, including pip extra-index-url/find-links.
    for key in list(os.environ):
        if key.startswith(('PIP_', 'NEAT_', 'SIMAPCIE_', 'PYNEAT_')) or key in ('SYSROOT', 'ELXR_SDK_RELEASE_FILE'):
            del os.environ[key]
    user, python_path = python_location(role)
    venv = str(python_path)
    os.environ.update({
        'DEBIAN_FRONTEND': 'noninteractive', 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1',
        'PIP_CONFIG_FILE': '/dev/null', 'PIP_NO_INDEX': '1', 'PIP_NO_CACHE_DIR': '1',
        'PIP_FIND_LINKS': (target/'wheels').resolve().as_uri(), 'PIP_DISABLE_PIP_VERSION_CHECK': '1',
        'HOME': user.pw_dir, 'PYNEAT_VENV_DIR': venv, 'NEAT_INSTALLER_APT_UPDATE': 'OFF',
        'NEAT_INSTALLER_INSTALL_CODEX_SKILL': 'OFF', 'NEAT_INSTALLER_INSTALL_CLAUDE_SKILL': 'OFF',
        'SIMAPCIE_INSTALL_CODEX_SKILL': 'OFF', 'SIMAPCIE_INSTALL_CLAUDE_SKILL': 'OFF',
    })
    if phase != 'verify' and not recreate_venv and (python_path.exists() or python_path.is_symlink()):
        check_python_environment(python_path, 'before installation')
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
        check_space(target, plan)
        # Producer admission checks the whole local transaction before mutations.
        lifecycle = ['bash', str(root/'lifecycle.sh'), role, 'preflight', str(target/'payload'), str(plan)]
        run(lifecycle)
        if phase == 'preflight':
            return
        if prerequisites:
            run(['apt-get', '--no-remove', '--no-install-recommends', '-y', 'install', *prerequisites])
        prepare_python_environment(python_path, recreate_venv)
        wheels = sorted((target/'payload').glob('*.whl'))
        try:
            result = run([venv+'/bin/python', '-m', 'pip', 'install', '--force-reinstall',
                          '--only-binary=:all:', *map(str, wheels)], capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            details = '\n'.join(value.strip() for value in (exc.stdout, exc.stderr) if value)
            raise ValueError(f'Offline Python installation failed in {venv}.\n{details}\n'
                             'The bundle may lack wheels needed by existing application packages. '
                             'Review the dependency errors before retrying; packages may have changed. '
                             'No automatic recreation or rollback was performed.') from None
        if result and result.stdout:
            print(result.stdout.strip())
        # Prerequisite installation changed dpkg state; admission must inspect
        # a fresh plan rather than treating already-installed prerequisites as new.
        with plan.open('w') as stream:
            run([*args, '--simulate'], stdout=stream)
        lifecycle[3] = 'install'
        run(lifecycle)
        verify(target, role, venv)
        finish_python_location(user, python_path, role)
        print('Python environment: '+venv)
        print('Activate with: source '+str(Path(user.pw_dir)/('pyneatpcie' if role == 'host' else 'pyneat'))+'/bin/activate')


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
    check_python_environment(Path(venv), 'after installation or during verification')
    run([venv+'/bin/python', '-c', 'import ' + ('pyneatpcie' if role == 'host' else 'pyneat')])
    if role == 'host':
        run(['gst-inspect-1.0', 'neatpciehost'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role', choices=['host', 'card'])
    parser.add_argument('phase', choices=['preflight', 'install', 'verify'])
    parser.add_argument('--recreate-venv', action='store_true',
                        help='discard existing Python environment packages and files before installation')
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.error('requires root')
    if any(name != 'lo' for _, name in socket.if_nameindex()):
        parser.error('deployment requires an isolated network namespace; use install.py')
    try:
        execute(Path(__file__).resolve().parent, args.role, args.phase, args.recreate_venv)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'Offline deployment failed: {exc}\n')


if __name__ == '__main__':
    main()
