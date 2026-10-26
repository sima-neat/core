#!/usr/bin/env python3
"""Read-only PCIe host/card inventory for the installed neat command."""
import argparse
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=8, **kwargs).stdout


def fields(path):
    if not path.is_file():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        if '=' in line:
            key, value = line.split('=', 1)
            result[key.strip()] = value.strip().strip('\"\'')
    return result


def software():
    result = subprocess.run(
        ['dpkg-query', '-W', '-f=${Package}\t${Version}\t${Architecture}\t${db:Status-Status}\n'],
        text=True, capture_output=True, timeout=5)
    packages = {}
    for line in result.stdout.splitlines():
        row = line.split('\t')
        if len(row) == 4 and row[3] == 'installed' and row[0].startswith(('sima', 'neat')):
            packages[row[0]] = {'version': row[1], 'architecture': row[2]}
    if result.returncode:
        raise ValueError('Unable to read installed Neat packages: '+result.stderr.strip())
    python = {}
    # Read distribution metadata without loading native modules or starting hardware.
    for name in ('pyneat', 'pyneatpcie'):
        versions = set()
        try:
            versions.add(importlib.metadata.version(name))
        except importlib.metadata.PackageNotFoundError:
            pass
        for venv in (Path.home()/name, Path('/opt/sima/neat')/name, Path('/media/nvme')/name):
            for site in venv.glob('lib/python*/site-packages'):
                for distribution in importlib.metadata.distributions(path=[str(site)]):
                    if distribution.metadata.get('Name', '').lower() == name:
                        versions.add(distribution.version)
        python[name] = {'version': next(iter(versions)) if len(versions) == 1 else None,
                        'error': 'Conflicting installed versions' if len(versions) > 1 else None}
    return {'os': fields(Path('/etc/os-release')), 'board': fields(Path('/etc/buildinfo')),
            'packages': packages, 'python': python}


def inventory(root):
    rows = []
    if not root.exists():
        return rows
    for device in sorted(root.iterdir()):
        if (device/'vendor').read_text().strip().lower() != '0x1f06':
            continue
        net = device/'net'
        rows.append({'bdf': device.name, 'deviceId': (device/'device').read_text().strip(),
                     'interfaces': sorted(p.name for p in net.iterdir()) if net.is_dir() else []})
    return rows


def endpoint(interface):
    addresses = json.loads(command(['ip', '-j', '-4', 'address', 'show', 'dev', interface]))
    local = []
    for entry in addresses:
        if entry.get('ifname') != interface or 'UP' not in entry.get('flags', []):
            raise ValueError('PCIe interface is down or changed')
        for addr in entry.get('addr_info', []):
            if addr.get('family') == 'inet':
                local.append(str(ipaddress.IPv4Address(addr['local'])))
    if len(local) != 1 or not re.fullmatch(r'10\.0\.\d+\.1', local[0]):
        raise ValueError('Expected a provisioned 10.0.N.1 PCIe interface; check PCIe network setup')
    peer = local[0].rsplit('.', 1)[0]+'.2'
    routes = json.loads(command(['ip', '-j', 'route', 'get', peer, 'from', local[0]]))
    if len(routes) != 1 or routes[0].get('dev') != interface:
        raise ValueError('Card route does not use its PCI-owned interface')
    return local[0], peer


def collect(root=Path('/sys/bus/pci/devices'), user='sima', key=None):
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', user):
        raise ValueError('Invalid PCIe SSH user')
    result = {'host': software(), 'cards': [], 'errors': []}
    try:
        before = inventory(root)
    except OSError as exc:
        result['errors'].append('Unable to read PCI inventory: '+str(exc))
        return result
    peers = set()
    for row in before:
        card = dict(row, address=None, state='unavailable', software=None, error=None)
        result['cards'].append(card)
        try:
            if len(row['interfaces']) != 1:
                raise ValueError('Expected one provisioned PCI-owned network interface')
            local, peer = endpoint(row['interfaces'][0])
            card['address'] = peer
            if peer in peers:
                raise ValueError('Duplicate card address; check PCIe network setup')
            peers.add(peer)
            ssh = ['ssh', '-T', '-b', local, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'ConnectTimeout=3', '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=2',
                   '-o', 'ServerAliveCountMax=1']
            if key and key.is_file():
                ssh += ['-i', str(key)]
            # stdin carries only this read-only probe; no installation, sudo,
            # password prompts, host-key enrollment, or network configuration.
            raw = command([*ssh, user+'@'+peer, shlex.join(['python3', '-B', '-', '--probe'])],
                          input=Path(__file__).read_text())
            remote = json.loads(raw)
            if (not isinstance(remote, dict) or not isinstance(remote.get('board'), dict)
                    or remote['board'].get('MACHINE') != 'modalix'):
                raise ValueError('SSH endpoint does not identify a Modalix board')
            if (not isinstance(remote.get('packages'), dict) or not isinstance(remote.get('python'), dict)
                    or not isinstance(remote.get('os'), dict)):
                raise ValueError('Invalid card software report')
            if any(not isinstance(value, dict) or not isinstance(value.get('version'), str)
                   or not isinstance(value.get('architecture'), str) for value in remote['packages'].values()):
                raise ValueError('Invalid card package version report')
            if any(not isinstance(value, dict) or 'version' not in value or 'error' not in value
                   for value in remote['python'].values()):
                raise ValueError('Invalid card Python version report')
            card.update(state='ok', software=remote)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
            card['error'] = detail
    try:
        if inventory(root) != before:
            result['errors'].append('PCI inventory changed during status collection; rerun with stable topology')
    except OSError as exc:
        result['errors'].append('Unable to recheck PCI inventory: '+str(exc))
    return result


def safe(value):
    return ''.join(char if char.isprintable() else ' ' for char in str(value))


def print_software(value, python_name):
    if not value['packages']:
        print('    No Neat packages installed')
    for name, package in sorted(value['packages'].items()):
        print('    '+safe(name)+'  '+safe(package['version'])+'  '+safe(package['architecture']))
    for name, distribution in sorted(value['python'].items()):
        if distribution['version'] or distribution['error'] or name == python_name:
            print('    '+name+'  '+safe(distribution['version'] or distribution['error'] or 'not installed'))


def render(value):
    print('Neat Environment\n  Mode               Ubuntu PCIe host\n\nHost Software')
    release = value['host']['os']
    print('    OS: '+safe(release.get('PRETTY_NAME') or release.get('ID', 'unknown')))
    print_software(value['host'], 'pyneatpcie')
    print('\nPCIe Cards')
    if not value['cards']:
        print('  No SiMa PCIe cards detected')
    for card in value['cards']:
        print('  '+safe(card['bdf'])+'  '+safe(', '.join(card['interfaces']) or 'no interface')
              +'  '+safe(card['address'] or 'no address'))
        if card['software']:
            board = card['software']['board']
            print('    Board: '+safe(board.get('MACHINE', 'unknown'))
                  +'  Platform: '+safe(board.get('SIMA_BUILD_VERSION') or board.get('DISTRO_VERSION', 'unknown')))
            release = card['software'].get('os', {})
            print('    OS: '+safe(release.get('PRETTY_NAME') or release.get('ID', 'unknown')))
            print_software(card['software'], 'pyneat')
        else:
            print('    Software unavailable: '+safe(card['error']))
    for error in value['errors']:
        print('  '+safe(error))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', action='store_true')
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    if args.render:
        render(json.load(sys.stdin))
    elif args.probe:
        print(json.dumps(software()))
    else:
        value = collect(user=os.environ.get('NEAT_PCIE_SSH_USER', 'sima'),
                        key=Path(os.environ.get('NEAT_PCIE_SSH_KEY', str(Path.home()/'.ssh/sima_neat_pcie_ed25519'))))
        print(json.dumps(value))


if __name__ == '__main__':
    main()
