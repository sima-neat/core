#!/usr/bin/env python3
"""Read-only PCIe host/card inventory for the installed neat command."""
import argparse
import importlib.metadata
import ipaddress
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys

WINDOWS = os.name == 'nt'
# MSI product names (CPACK_PACKAGE_NAME) of Core and Scheduler, as Linux lists sima*/neat* DEBs.
WINDOWS_PRODUCTS = ('sima-pcie-host', 'simaai-scheduler-pcie-host')
# The Platform virtual Ethernet driver; Windows numbers later instances "#2", "#3", ...
# A LAN, VPN or Hyper-V adapter at 10.0.N.1 is not a card.
WINDOWS_CARD_ADAPTER = r'SiMa\.ai Virtual Ethernet Adapter( #\d+)?'
WINDOWS_ADAPTERS = (
    "$ErrorActionPreference='Stop'; [Console]::OutputEncoding=New-Object Text.UTF8Encoding $false; "
    "Get-NetAdapter | ForEach-Object { $a=$_; "
    "$ips=@(Get-NetIPAddress -InterfaceIndex $a.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue); "
    "if (-not $ips) { $ips=@($null) }; foreach ($ip in $ips) { [pscustomobject]@{"
    "Alias=$a.Name; Description=$a.InterfaceDescription; Status=[string]$a.Status; Address=$ip.IPAddress} } "
    "} | ConvertTo-Json -Compress")


def command(args, timeout=8, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, timeout=timeout, **kwargs).stdout


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
    python = {name: python_installations(name) for name in ('pyneat', 'pyneatpcie')}
    return {'os': fields(Path('/etc/os-release')), 'board': fields(Path('/etc/buildinfo')),
            'packages': packages, 'python': python}


def registry(path, names, children=False):
    import winreg
    # 64-bit view: the x64 MSIs register there whatever the interpreter's bitness.
    access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY

    def read(key):
        row = {}
        for name in names:
            try:
                row[name] = winreg.QueryValueEx(key, name)[0]
            except OSError:
                pass
        return row
    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path, 0, access) as key:
        if not children:
            return [read(key)]
        rows = []
        for index in range(winreg.QueryInfoKey(key)[0]):
            try:
                with winreg.OpenKey(key, winreg.EnumKey(key, index), 0, access) as child:
                    rows.append(read(child))
            except OSError:
                continue
        return rows


def windows_software():
    try:
        products = registry(r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall',
                            ('DisplayName', 'DisplayVersion'), children=True)
        current = registry(r'SOFTWARE\Microsoft\Windows NT\CurrentVersion',
                           ('DisplayVersion', 'CurrentBuild'))[0]
    except OSError as exc:
        raise ValueError('Unable to read installed Neat products: '+str(exc)) from exc
    packages = {}
    for product in products:
        name = str(product.get('DisplayName', ''))
        if name.startswith(WINDOWS_PRODUCTS):
            packages[name] = {'version': str(product.get('DisplayVersion', '')), 'architecture': 'amd64'}
    # Not ProductName: Windows 11 still stores "Windows 10" there.
    pretty = ' '.join(part for part in ('Windows', platform.release(), current.get('DisplayVersion'),
                                        '(build '+str(current.get('CurrentBuild', '?'))+')') if part)
    release = {'ID': 'windows', 'VERSION_ID': platform.version(), 'PRETTY_NAME': pretty}
    python = {name: python_installations(name) for name in ('pyneat', 'pyneatpcie')}
    return {'os': release, 'board': {}, 'packages': packages, 'python': python}


def python_installations(name):
    # Inspect metadata only: importing native modules can initialize hardware.
    found = {}

    def add(distribution):
        if distribution.metadata.get('Name', '').lower() != name:
            return
        site = Path(distribution.locate_file('')).resolve()
        environment = next((str(parent) for parent in site.parents
                            if (parent/'pyvenv.cfg').is_file()), None)
        found[(str(site), distribution.version)] = {
            'version': distribution.version, 'environment': environment, 'path': str(site)}

    for distribution in importlib.metadata.distributions():
        add(distribution)
    for venv in (Path.home()/name, Path('/opt/sima/neat')/name, Path('/media/nvme')/name):
        for site in sorted([*venv.glob('lib/python*/site-packages'), *venv.glob('Lib/site-packages')]):
            for distribution in importlib.metadata.distributions(path=[str(site)]):
                add(distribution)
    installations = [found[key] for key in sorted(found)]
    versions = {item['version'] for item in installations}
    return {'version': next(iter(versions)) if len(versions) == 1 else None,
            'error': None, 'installations': installations}


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


def windows_inventory():
    # Named deviation: Windows has no sysfs PCI inventory; cards are the Platform's
    # virtual Ethernet adapters, listed by interface and address without a PCI address.
    powershell = Path(os.environ.get('SystemRoot', r'C:\Windows'))/'System32/WindowsPowerShell/v1.0/powershell.exe'
    try:
        rows = json.loads(command([str(powershell), '-NoLogo', '-NoProfile', '-NonInteractive', '-Command',
                                   WINDOWS_ADAPTERS], timeout=30, encoding='utf-8-sig') or '[]')
        adapters = {}
        for row in rows if isinstance(rows, list) else [rows]:
            if row.get('Status') == 'Up' and re.fullmatch(WINDOWS_CARD_ADAPTER, row.get('Description') or ''):
                addresses = adapters.setdefault(row['Alias'], [])
                if row.get('Address'):
                    addresses.append(str(ipaddress.IPv4Address(row['Address'])))
    except subprocess.CalledProcessError as exc:
        raise OSError(exc.stderr.strip() or str(exc)) from exc
    except (ValueError, KeyError, AttributeError, subprocess.SubprocessError) as exc:
        raise OSError(str(exc)) from exc
    return [{'bdf': None, 'deviceId': None, 'interfaces': [alias], 'addresses': sorted(addresses)}
            for alias, addresses in sorted(adapters.items())]


def windows_endpoint(row):
    local = row['addresses']
    if len(local) != 1 or not re.fullmatch(r'10\.0\.\d+\.1', local[0]):
        raise ValueError('Expected a provisioned 10.0.N.1 PCIe interface; check PCIe network setup')
    return local[0], local[0].rsplit('.', 1)[0]+'.2'


def windows_ssh():
    # As HostPlatform.cpp open_ssh_executable: System32 OpenSSH, then absolute PATH entries.
    for folder in [Path(os.environ.get('SystemRoot', r'C:\Windows'))/'System32/OpenSSH',
                   *map(Path, os.environ.get('PATH', '').split(os.pathsep))]:
        if folder.is_absolute() and (folder/'ssh.exe').is_file():
            return str(folder/'ssh.exe')
    raise ValueError('Windows OpenSSH client (ssh.exe) not found; install the OpenSSH Client optional feature')


def collect(root=Path('/sys/bus/pci/devices'), user='sima', key=None):
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', user):
        raise ValueError('Invalid PCIe SSH user')
    result = {'host': windows_software() if WINDOWS else software(), 'cards': [], 'errors': []}
    discover = windows_inventory if WINDOWS else lambda: inventory(root)
    try:
        before = discover()
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
            local, peer = windows_endpoint(row) if WINDOWS else endpoint(row['interfaces'][0])
            card['address'] = peer
            if peer in peers:
                raise ValueError('Duplicate card address; check PCIe network setup')
            peers.add(peer)
            ssh = [windows_ssh() if WINDOWS else 'ssh',
                   '-T', '-b', local, '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
                   '-o', 'ConnectTimeout=3', '-o', 'ConnectionAttempts=1', '-o', 'ServerAliveInterval=2',
                   '-o', 'ServerAliveCountMax=1']
            if key and key.is_file():
                # Windows only: the ssh-agent service commonly offers several keys before the Neat key.
                ssh += ['-i', str(key)] + (['-o', 'IdentitiesOnly=yes'] if WINDOWS else [])
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
            for distribution in remote['python'].values():
                installations = distribution.get('installations', [])
                if (not isinstance(installations, list) or any(
                        not isinstance(item, dict)
                        or not isinstance(item.get('version'), str)
                        or not isinstance(item.get('path'), str)
                        or not (item.get('environment') is None
                                or isinstance(item['environment'], str))
                        for item in installations)):
                    raise ValueError('Invalid card Python installation report')
            card.update(state='ok', software=remote)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
            card['error'] = detail
    try:
        if discover() != before:
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
        installations = distribution.get('installations', [])
        if installations:
            locations = {item['environment'] or item['path'] for item in installations}
            detail = ('Multiple environments' if len(locations) > 1
                      else 'Multiple installed versions' if len(installations) > 1
                      else installations[0]['version'])
            print('    '+safe(name)+'  '+safe(detail))
            for item in installations:
                location = ('venv: '+item['environment'] if item['environment']
                            else 'site-packages: '+item['path'])
                print('      '+safe(item['version'])+'  ('+safe(location)+')')
        elif distribution['version'] or distribution['error'] or name == python_name:
            print('    '+name+'  '+safe(distribution['version'] or distribution['error'] or 'not installed'))


def render(value):
    print('Neat Environment\n  Mode               '+('Windows' if WINDOWS else 'Ubuntu')
          +' PCIe host\n\nHost Software')
    release = value['host']['os']
    print('    OS: '+safe(release.get('PRETTY_NAME') or release.get('ID', 'unknown')))
    print_software(value['host'], 'pyneatpcie')
    print('\nPCIe Cards')
    if not value['cards']:
        print('  No SiMa PCIe cards detected')
    for card in value['cards']:
        parts = ([card['bdf']] if card['bdf'] else []) + [', '.join(card['interfaces']) or 'no interface',
                                                         card['address'] or 'no address']
        print('  '+'  '.join(safe(part) for part in parts))
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
