#!/usr/bin/env python3
"""Install standalone Core locally using only the verified offline bundle."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import subprocess

from assemble import verify_bundle
from process import run_host


def install(root, phase):
    if os.geteuid() != 0:
        raise ValueError('Run with sudo on the target DevKit')
    if verify_bundle(root).get('scope') != 'core':
        raise ValueError('Expected a standalone Core offline bundle')
    for command in ('unshare', 'timeout', 'python3'):
        if not shutil.which(command):
            raise ValueError('Platform prerequisite missing: ' + command)
    # Shared with the PCIe installer to prevent simultaneous runtime upgrades.
    with open('/run/neat-offline-install.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_host(['timeout', '--foreground', '--kill-after=30', '1500',
                  'unshare', '--net', 'python3', '-B', str(root/'deploy.py'),
                  'card', phase], stdin=subprocess.DEVNULL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='verify compatibility and simulate without installing')
    parser.add_argument('--verify', action='store_true', help='verify the installed bundle versions and Python imports')
    args = parser.parse_args()
    if args.check and args.verify:
        parser.error('choose only one of --check and --verify')
    install(Path(__file__).resolve().parent,
            'preflight' if args.check else 'verify' if args.verify else 'install')
    print('PASS: standalone Core ' + ('preflight' if args.check else 'verification' if args.verify else 'installation'))


if __name__ == '__main__':
    main()
