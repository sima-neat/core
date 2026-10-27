#!/usr/bin/env python3
"""Provision a disposable card baseline from authenticated Scheduler SDK policy."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
SYSROOT = Path('/opt/toolchain/aarch64/modalix')
PACKAGES = Path('/tmp/offline-sysroot-packages')
REPORTS = Path('/var/lib/neat-offline-ci')


def run(args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def load(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sdk_policy(publication):
    """Read policy only after authenticating the incoming Scheduler and its file."""
    helper = load(publication/'scheduler_platform_admission.py')
    selected = json.loads((publication/'resolved-deps-manifest.json').read_text())['scheduler']['buildinfo']
    archives = helper._scheduler_archives(publication)
    if len(archives) != 1:
        raise ValueError('expected one selected Scheduler archive')
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)/'scheduler'
        helper.extract_scheduler(archives[0], root)
        prefix = root/'usr'
        helper.authenticate_tool(prefix, selected)
        name = 'share/SimaScheduler/dependency-policy.json'
        path = prefix/name
        expected = selected['installed_files'].get(name, {})
        if (path.is_symlink() or not path.is_file() or expected.get('kind') != 'file'
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected.get('sha256')):
            raise ValueError('Scheduler SDK policy differs from selected receipt')
        return json.loads(path.read_text())


def resolve(policy):
    # Reuse the shared firmware resolver; never guess a package version from B<N>.
    resolver = load(HERE/'resolve_sysroot.py')
    declaration = {key: value for key, value in policy.items() if key != 'sysroot-version'}
    firmware, _ = resolver.manifest_selection(declaration)
    result = resolver.resolve(firmware)
    if result['sysroot_version'] != policy.get('sysroot-version'):
        raise ValueError('resolved sysroot differs from selected Scheduler SDK policy')
    if result['channel'] != 'pre-release':
        raise ValueError('selected SDK sysroot updater requires a pre-release firmware policy')
    return result


def native_archives(directory, required, installed):
    """Keep installed reverse dependencies in the same resolved SDK cohort."""
    archives = {}
    for archive in directory.glob('*.deb'):
        package = subprocess.check_output(['dpkg-deb', '-f', str(archive), 'Package'], text=True).strip()
        if package in required or package in installed:
            if package in archives:
                raise ValueError('duplicate resolved sysroot provider: '+package)
            archives[package] = archive
    if required - set(archives):
        raise ValueError('resolved sysroot lacks native providers: '+', '.join(sorted(required-set(archives))))
    return archives


def provision(publication):
    original = {name.split(':', 1)[0] for name in
                Path('/tmp/offline-baseline-packages').read_text().splitlines()}
    tooling = {name.split(':', 1)[0] for name in subprocess.check_output(
        ['dpkg-query', '-W', '-f=${binary:Package}\n'], text=True).splitlines()} - original
    policy = sdk_policy(publication)
    resolution = resolve(policy)
    REPORTS.mkdir(parents=True)
    (REPORTS/'sysroot-resolution.json').write_text(json.dumps(resolution, indent=2)+'\n')
    (REPORTS/'scheduler-sdk-policy.json').write_text(json.dumps(policy, indent=2)+'\n')
    # Install the unmodified SDK entrypoints and shared package classification.
    for source, destination in (
            ('sysroot.sh', '/usr/local/bin/sysroot'),
            ('setup-sdk-sysroot.sh', '/usr/local/bin/setup-sdk-sysroot.sh'),
            ('validate-sysroot-package-versions.sh', '/usr/local/bin/validate-sysroot-package-versions.sh'),
            ('simaai_setup_sdk.py', '/opt/bin/simaai_setup_sdk.py'),
            ('platform-package-patterns.txt', '/usr/local/share/sima-sdk/platform-package-patterns.txt')):
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(HERE/source, target)
        target.chmod(0o755 if target.suffix != '.txt' else 0o644)
    release = REPORTS/'sdk-release'
    release.write_text('Platform Base = '+policy['platform-version']+'\n'
                       'Platform Version = '+policy['platform-version']+'\n')
    source = Path('/etc/apt/sources.list.d/neat-offline-sysroot.list')
    source.write_text('deb [arch=arm64 trusted=yes] https://debian.neat.sima.ai/pre-release agate non-free\n')
    env = {**os.environ, 'SDK_RELEASE_FILE': str(release), 'SDK_SYSROOT_PYTHON': '/usr/bin/python3',
           'SYSROOT_UPDATE_DOWNLOAD_DIR': str(PACKAGES),
           'PRE_RELEASE_PACKAGES_URL': resolution['index_url']}
    run(['sysroot', 'update', '--sysroot', SYSROOT, resolution['sysroot_version']], env=env)
    # The SDK extracts the platform anchor's closure, not the metapackage itself.
    run(['apt-get', 'download', resolution['package']+'='+resolution['sysroot_version']], cwd=PACKAGES)
    # Scheduler owns runtime-provider classification. Use the just-resolved SDK
    # archives to provision those packages in the native validation root too.
    description = REPORTS/'platform-providers.json'
    run(['/usr/bin/python3', publication/'scheduler_platform_admission.py', 'describe',
         '--directory', publication, '--output', description])
    required = set(json.loads(description.read_text())['platform_packages']) | {resolution['package']}
    archives = native_archives(PACKAGES, required, original)
    print('Installing resolved native sysroot providers: '+', '.join(sorted(archives)), flush=True)
    versions = {package: subprocess.check_output(
        ['dpkg-deb', '-f', str(archive), 'Version'], text=True).strip()
        for package, archive in archives.items()}
    # SDK download filenames contain colons (package:arm64.deb). Use ordinary
    # native filenames and explicit versions so APT cannot retain an older
    # provider to satisfy a reverse dependency while considering local archives.
    native = PACKAGES/'native'
    native.mkdir()
    for package, archive in archives.items():
        shutil.copyfile(archive, native/(package+'.deb'))
    run(['apt-get', '--allow-downgrades', '--no-remove', '--no-install-recommends', '-y',
         'install', *sorted(native.glob('*.deb')),
         *[name+'='+versions[name] for name in sorted(archives)]])
    for package, expected in versions.items():
        actual = subprocess.check_output(['dpkg-query', '-W', '-f=${Version}', package], text=True).strip()
        if actual != expected:
            raise ValueError('native provider differs from resolved sysroot: '+package)
    run(['apt-get', 'check'])
    # Collection must see a runtime baseline, not one enriched with the SDK's
    # Python/APT provisioning dependencies. Purge only additions made before
    # sysroot provisioning; dpkg rejects unexpected reverse dependencies.
    if tooling - set(archives):
        run(['dpkg', '--purge', *sorted(tooling-set(archives))])
    run(['apt-get', 'check'])
    if subprocess.check_output(['dpkg', '--audit'], text=True).strip():
        raise ValueError('provisioned card baseline has an incomplete package transaction')
    source.unlink()


if __name__ == '__main__':
    provision(Path(sys.argv[1]).resolve(strict=True))
