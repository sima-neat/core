#!/usr/bin/env python3
"""Install the verified standalone bundle into the Neat SDK ARM64 sysroot."""
import argparse
import os
from pathlib import Path
import socket
import tempfile

from assemble import sha, verify_bundle
from deploy import private_apt, run, output

SYSROOT = Path('/opt/toolchain/aarch64/modalix')


def verify(target, sysroot=SYSROOT):
    # Sysroot installation extracts DEBs; the SDK host's dpkg database and Python
    # ABI describe the host, not the cross-compilation target.
    with tempfile.TemporaryDirectory(prefix='neat-sdk-verify-') as tmp:
        expected = Path(tmp)
        for deb in sorted((target/'payload').glob('*.deb')):
            run(['dpkg-deb', '-x', str(deb), str(expected)])
        for path in expected.rglob('*'):
            installed = sysroot/path.relative_to(expected)
            if path.is_symlink():
                if not installed.is_symlink() or os.readlink(installed) != os.readlink(path):
                    raise ValueError('SDK sysroot symlink mismatch: ' + str(installed))
            elif path.is_file():
                if installed.is_symlink() or not installed.is_file() or sha(installed) != sha(path):
                    raise ValueError('SDK sysroot package content mismatch: ' + str(installed))


def execute(root, phase):
    if verify_bundle(root).get('scope') != 'core':
        raise ValueError('Expected standalone Core bundle')
    # Bundled core_install.py is named install.py in the archive.
    from install import environment
    if environment(root) != 'sdk':
        raise ValueError('SDK sysroot installation requires the Neat eLxr SDK')
    target = root/'card'
    for deb in sorted((target/'payload').glob('*.deb')):
        if output(['dpkg-deb', '-f', str(deb), 'Architecture']) not in ('arm64', 'all'):
            raise ValueError('SDK payload must target ARM64: ' + deb.name)
    if phase == 'verify':
        verify(target)
        return
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('PIP_', 'NEAT_', 'SIMAPCIE_', 'PYNEAT_', 'DEVKIT_'))
           and k not in ('SYSROOT', 'ELXR_SDK_RELEASE_FILE', 'FORCE_INSTALL', 'APT_CONFIG')}
    env.update(SYSROOT=str(SYSROOT), NEAT_INSTALLER_SKIP_DEVKIT_SYNC='ON',
               NEAT_INSTALLER_INSTALL_CODEX_SKILL='OFF', NEAT_INSTALLER_INSTALL_CLAUDE_SKILL='OFF')
    with private_apt(target/'repo') as state:
        with (state/'config').open('a') as config:
            config.write('APT::Architecture "arm64";\nAPT::Architectures { "arm64"; };\n')
        env['APT_CONFIG'] = str(state/'config')
        run(['apt-get', 'update'], env=env)
        # Reuse the producer's SDK extraction, dependencies, symlink repair and
        # artifact cache. Paired DevKit deployment is handled by the offline
        # entry point, so it also uses this bundle and never the online path.
        run(['bash', '-ec', '''
cd "$1"
source ./install_neat_framework.sh
parse_args --local
collect_debs_in_install_order . DEBS
ENV_MODE=elxr-sdk
ensure_platform_compatible
validate_single_sima_neat_package_pair
if [[ "$2" == install ]]; then
  install_for_environment
fi
''', 'offline-sdk', str(target/'payload'), phase], env=env)
    if phase == 'install':
        verify(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['preflight', 'install', 'verify'])
    args = parser.parse_args()
    if os.geteuid() != 0 or any(name != 'lo' for _, name in socket.if_nameindex()):
        parser.error('requires root and an isolated network namespace; use install.sh')
    execute(Path(__file__).resolve().parent, args.phase)
