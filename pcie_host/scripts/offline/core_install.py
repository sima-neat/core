#!/usr/bin/env python3
"""Install standalone Core in the Neat SDK or on a Modalix DevKit offline."""
import argparse
from contextlib import nullcontext
import fcntl
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess

from assemble import verify_bundle
from process import run_host
from paired_devkit import paired_devkit
from progress import Progress


def metadata(path):
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('\"\'')
    return values


def environment(root, system=Path('/')):
    """Admit canonical on-disk identity; caller overrides are not evidence."""
    if platform.system() != 'Linux':
        raise ValueError('Unsupported installation environment: requires the Neat SDK or Modalix DevKit on Linux')
    expected = json.loads((root/'card/payload/manifest.json').read_text()).get('platform-version', '')
    if not isinstance(expected, str) or not expected:
        raise ValueError('Bundle package manifest is missing platform-version')
    expected = expected.split('+', 1)[0]
    sdk = metadata(system/'etc/sdk-release')
    board = metadata(system/'etc/buildinfo')
    if (system/'etc/sdk-release').exists():
        if sdk.get('SDK Type') or not sdk.get('SDK Version'):
            raise ValueError('Unsupported SDK identity in /etc/sdk-release; expected the Neat eLxr SDK')
        if not (system/'opt/toolchain/aarch64/modalix/usr').is_dir():
            raise ValueError('Neat SDK ARM64 sysroot is missing: /opt/toolchain/aarch64/modalix')
        mode = 'sdk'
        actual = sdk.get('Platform Base') or sdk['SDK Version'].split('_', 1)[0]
    elif board.get('MACHINE') == 'modalix':
        mode = 'devkit'
        actual = board.get('DISTRO_VERSION', '')
        if platform.machine() != 'aarch64':
            raise ValueError('Modalix DevKit requires an ARM64 host')
    else:
        raise ValueError('Unsupported installation environment: run inside the Neat eLxr SDK '
                         'or on a provisioned Modalix DevKit; /etc/sdk-release or '
                         '/etc/buildinfo must identify the environment')
    if actual != expected:
        raise ValueError(f'Incompatible {mode} Platform version: bundle requires {expected}, '
                         f'detected {actual or "missing"}; no installation changes were made')
    return mode


def install(root, phase, devkit_only=False, progress=None):
    if os.geteuid() != 0:
        raise ValueError('Run ./install.sh inside the Neat SDK or on the target Modalix DevKit')
    with progress.step('Checking bundle integrity and supported environment') if progress else nullcontext():
        if verify_bundle(root).get('scope') != 'core':
            raise ValueError('Expected a standalone Core offline bundle')
        mode = environment(root)
    if devkit_only and mode != 'devkit':
        raise ValueError('Paired target must be a provisioned Modalix DevKit')
    for command in (('timeout', 'python3') if mode == 'sdk' else ('unshare', 'timeout', 'python3')):
        if not shutil.which(command):
            raise ValueError('Platform prerequisite missing: ' + command)
    # Shared with the PCIe installer to prevent simultaneous runtime upgrades.
    with open('/run/neat-offline-install.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def local(current):
            command = [str(root/'sdk_install.py'), current] if mode == 'sdk' else [str(root/'deploy.py'), 'card', current]
            labels = {
                'preflight': 'Checking SDK compatibility and offline dependencies' if mode == 'sdk' else 'Checking DevKit compatibility and offline dependencies',
                'install': 'Installing Core into SDK sysroot' if mode == 'sdk' else 'Installing Core on Modalix DevKit and verifying installation',
                'verify': 'Verifying SDK sysroot installation' if mode == 'sdk' else 'Verifying Modalix DevKit installation',
            }
            with progress.step(labels[current]) if progress else nullcontext():
                logs = {'stdout': progress.log, 'stderr': subprocess.STDOUT} if progress else {}
                run_host(['timeout', '--foreground', '--kill-after=30', '1500',
                          *([] if mode == 'sdk' else ['unshare', '--net']),
                          'python3', '-B', *command], stdin=subprocess.DEVNULL, **logs)
        if mode == 'sdk':
            with paired_devkit(root, progress=progress) as device:
                if phase == 'install':
                    local('preflight')
                    if device:
                        device.phase('preflight')
                local(phase)
                if device:
                    device.phase(phase)
        else:
            local(phase)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify compatibility and simulate without installing')
    parser.add_argument('--verify', action='store_true', help='verify SDK sysroot contents or DevKit packages and Python imports')
    parser.add_argument('--devkit-only', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check and args.verify:
        parser.error('choose only one of --check and --verify')
    progress = None if args.devkit_only else Progress()
    try:
        install(Path(__file__).resolve().parent,
                'preflight' if args.check else 'verify' if args.verify else 'install', args.devkit_only, progress)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        if progress:
            progress.message('FAIL', 'Offline Core operation failed: '+str(exc))
            progress.message('INFO', 'Read details with: sudo tail -n 60 '+str(progress.path))
        else:
            print('Core offline installation failed: '+str(exc))
        raise SystemExit(1)
    else:
        action = 'preflight' if args.check else 'verification' if args.verify else 'installation'
        if progress:
            progress.message('OK', 'Offline Core '+action+' completed successfully')
            progress.message('INFO', 'Detailed log: '+str(progress.path))
        else:
            print('PASS: standalone Core '+action)
    finally:
        if progress:
            progress.close()


if __name__ == '__main__':
    main()
