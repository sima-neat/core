#!/usr/bin/env python3
"""Container worker for dependency collection and network-isolated package tests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BOOTSTRAP = ['python3', 'python3-venv', 'python3-pip', 'iproute2', 'openssh-client',
             'gstreamer1.0-tools', 'ca-certificates']


def run(args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def output(args):
    return subprocess.check_output(args, text=True).strip()


def environment(root):
    row = json.loads((root / 'target.json').read_text())
    release = dict(line.split('=', 1) for line in Path('/etc/os-release').read_text().splitlines() if '=' in line)
    if (release['ID'].strip('"'), release['VERSION_ID'].strip('"')) != (row['os_id'], row['os_version']):
        raise ValueError('baseline OS does not match target')
    if output(['dpkg', '--print-architecture']) != row['architecture']:
        raise ValueError('baseline architecture does not match target')
    return row


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(root):
    row = environment(root)
    repo = root / 'repo'
    wheels = root / 'wheels'
    repo.mkdir()
    wheels.mkdir()
    native = sorted((root / 'payload').glob('*.deb'))
    for path in native:
        arch = output(['dpkg-deb', '-f', str(path), 'Architecture'])
        if arch not in (row['architecture'], 'all'):
            raise ValueError('foreign-architecture payload: ' + path.name)
        package = output(['dpkg-deb', '-f', str(path), 'Package'])
        if row['architecture'] == 'amd64' and (package == 'simaai-pcie-host' or 'dkms' in package):
            raise ValueError('Platform/driver package belongs to step-1 provisioning: ' + package)
        shutil.copyfile(path, repo / path.name)
    helper = root / 'payload/pcie_delivery.py'
    if helper.is_file():
        run(['python3', str(helper), 'verify', '--directory', str(root / 'payload')])
    # Use the pristine image's status, not the tool-enriched collection container.
    # Empty download cache prevents successful collection from depending on a CI cache.
    cache = root / 'download-cache'
    (cache / 'partial').mkdir(parents=True)
    run(['apt-get', '-o', f'Dir::State::status={root}/baseline.status',
         '-o', f'Dir::Cache::archives={cache}', '--download-only', '--reinstall',
         '--no-remove', '--no-install-recommends', '-y', 'install',
         *map(str, native), *BOOTSTRAP])
    for path in cache.glob('*.deb'):
        package = output(['dpkg-deb', '-f', str(path), 'Package'])
        if row['architecture'] == 'amd64' and (package == 'simaai-pcie-host' or 'dkms' in package):
            raise ValueError('Dependency resolution would install a Platform/driver package: ' + package)
        dest = repo / path.name
        if dest.exists() and sha(dest) != sha(path):
            raise ValueError('dependency filename collision')
        shutil.copyfile(path, dest)
    shutil.rmtree(cache)
    with (repo / 'Packages').open('w') as stream:
        run(['dpkg-scanpackages', '--multiversion', '.', '/dev/null'], cwd=repo, stdout=stream)
    run(['python3', '-m', 'pip', 'download', '--disable-pip-version-check',
         '--no-cache-dir', '--only-binary=:all:', '--dest', str(wheels),
         *map(str, sorted((root / 'payload').glob('*.whl'))), 'pip'])
    packages = []
    for path in sorted(repo.glob('*.deb')):
        fields = output(['dpkg-deb', '-f', str(path), 'Package', 'Version', 'Architecture'])
        packages.append({'file': path.name, 'control': fields, 'sha256': sha(path)})
    (root / 'dependencies.json').write_text(json.dumps({
        'packages': packages,
        'wheels': {p.name: sha(p) for p in sorted(wheels.glob('*.whl'))},
        'baseline_status_sha256': sha(root / 'baseline.status'),
        'python': output(['python3', '--version']),
        'python_abi': output(['python3', '-c', 'import sysconfig; print(sysconfig.get_config_var("SOABI"))']),
        'apt_index_hashes': {p.name: sha(p) for p in sorted(Path('/var/lib/apt/lists').iterdir())
                             if p.is_file() and p.name != 'lock'},
    }, indent=2)+'\n')
    # The configuration is private to this command: installed OS sources and
    # apt.conf.d hooks are not read. Only file: repository indexes are available.
    (root / 'bootstrap.sh').write_text('''#!/bin/sh
set -eu
root=$(pwd)
mkdir -p "$root/apt/lists/partial" "$root/apt/cache/partial" "$root/apt/empty"
printf 'deb [trusted=yes] file:%s/repo ./\\n' "$root" > "$root/apt/sources.list"
cat > "$root/apt/config" <<EOF
Dir::Etc::main "/dev/null";
Dir::Etc::parts "$root/apt/empty";
Dir::Etc::sourcelist "$root/apt/sources.list";
Dir::Etc::sourceparts "$root/apt/empty";
Dir::Etc::preferences "/dev/null";
Dir::Etc::preferencesparts "$root/apt/empty";
Dir::State::lists "$root/apt/lists";
Dir::Cache::archives "$root/apt/cache";
Acquire::Languages "none";
APT::Get::List-Cleanup "0";
EOF
export APT_CONFIG="$root/apt/config"
apt-get update
apt-get --no-remove --no-install-recommends -y install python3
''')


def validate(root):
    environment(root)
    os.environ['APT_CONFIG'] = str(root / 'apt/config')
    native = sorted((root / 'payload').glob('*.deb'))
    args = ['apt-get', '--no-remove', '--no-install-recommends', '-y', 'install',
            *map(str, native), *BOOTSTRAP]
    run([*args, '--simulate'])
    run(args)
    run(['apt-get', 'check'])
    audit = output(['dpkg', '--audit'])
    if audit:
        raise ValueError('dpkg audit failed: ' + audit)
    run(['python3', '-m', 'venv', '/tmp/offline-venv'])
    # Environment/user pip configuration cannot re-enable indexes or extra links.
    env = {k: v for k, v in os.environ.items() if not k.startswith('PIP_')}
    env['PIP_CONFIG_FILE'] = '/dev/null'
    run(['/tmp/offline-venv/bin/python', '-m', 'pip', '--isolated', 'install',
         '--no-index', '--no-cache-dir', '--only-binary=:all:', '--find-links', str(root / 'wheels'),
         *map(str, sorted((root / 'payload').glob('*.whl')))], env=env)
    run(['/tmp/offline-venv/bin/python', '-m', 'pip', 'check'], env=env)
    for wheel_prefix, module in [('pyneatpcie', 'pyneatpcie'), ('pyneat', 'pyneat')]:
        if list((root / 'payload').glob(wheel_prefix + '-*.whl')):
            run(['/tmp/offline-venv/bin/python', '-c', 'import ' + module], env=env)
    print('PASS: native package transaction and Python dependency installation with Docker --network=none')


if __name__ == '__main__':
    if sys.argv[1] == 'collect':
        collect(Path('/bundle'))
    elif sys.argv[1] == 'validate':
        validate(Path.cwd())
    else:
        raise SystemExit('expected collect or validate')
