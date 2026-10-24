#!/usr/bin/env python3
"""Install a verified Neat bundle on the prepared host and every attached PCIe card."""
import argparse
import fcntl
import getpass
import ipaddress
import json
import os
from pathlib import Path
import pwd
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True

from assemble import verify_bundle
import discover

PCI_ROOT = Path('/sys/bus/pci/devices')
COMMON = ('bundle.json', 'assemble.py', 'deploy.py', 'lifecycle.sh', 'INSTALL.md')


def run(args, **kwargs):
    return subprocess.run(args, check=True, timeout=1800, **kwargs)


def endpoints(evidence):
    if evidence['errors'] or not evidence['cards']:
        raise ValueError('All-card discovery failed: ' + '; '.join(evidence['errors']))
    result = []
    addresses = set()
    for card in evidence['cards']:
        probes = card['probes']
        if len(card['interfaces']) != 1 or len(probes) != 1 or len(probes[0]['responses']) != 1:
            raise ValueError('Ambiguous PCIe endpoint: ' + card['bdf'])
        probe = probes[0]
        reply = probe['responses'][0]
        address = str(ipaddress.IPv4Address(reply['source_ip']))
        ip = ipaddress.IPv4Address(address)
        if ip.is_loopback or ip.is_multicast or ip.is_unspecified or address == probe['local_ip']:
            raise ValueError('Invalid discovery source address')
        if address in addresses:
            raise ValueError('Duplicate card IPs: provision unique PCIe addresses before installing')
        if reply['payload'].get('ip') != address:
            raise ValueError('Discovery advertised IP differs from packet source')
        if probe['interface'] != card['interfaces'][0]:
            raise ValueError('Discovery interface differs from PCI inventory')
        addresses.add(address)
        result.append(dict(card, ip=address, interface=probe['interface'], local_ip=probe['local_ip']))
    return result


class Operations:
    def __init__(self, root, user, key, known_hosts, sudo_password=None):
        self.root, self.user = root, user
        self.key, self.known_hosts = key, known_hosts
        self.staged = {}
        self.sudo_password = sudo_password

    def topology(self):
        return discover.inventory(PCI_ROOT)

    def route(self, card):
        result = run(['ip', '-j', '-4', 'route', 'get', card['ip'], 'from', card['local_ip']],
                     capture_output=True, text=True)
        routes = json.loads(result.stdout)
        if len(routes) != 1 or routes[0].get('dev') != card['interface'] or routes[0].get('gateway'):
            raise ValueError('Card route is not direct through its PCI-owned interface')

    def ssh(self, card, command, **kwargs):
        self.route(card)
        try:
            return run(['ssh', '-F', '/dev/null', '-T', '-i', str(self.key), '-b', card['local_ip'],
                        '-o', 'BindInterface='+card['interface'], '-o', 'BatchMode=yes',
                    '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=yes',
                    '-o', 'UserKnownHostsFile='+str(self.known_hosts),
                    '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
                    '-o', 'ServerAliveCountMax=2', self.user+'@'+card['ip'], command], **kwargs)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 255:
                raise ConnectionError('SSH transport lost; remote outcome may be unknown') from exc
            raise

    def stage(self, card):
        required = 2 * sum(p.stat().st_size for p in (self.root/'card').rglob('*') if p.is_file())
        space = self.ssh(card, "python3 -c 'import shutil; print(shutil.disk_usage(\"/var/tmp\").free)'",
                         capture_output=True, text=True)
        if int(space.stdout.strip()) < required:
            raise ValueError('Insufficient card staging space')
        result = self.ssh(card, 'umask 077; mktemp -d /var/tmp/neat-offline-XXXXXXXX',
                          capture_output=True, text=True)
        path = result.stdout.strip()
        import re
        if not re.fullmatch(r'/var/tmp/neat-offline-[A-Za-z0-9]{8}', path):
            raise ValueError('Invalid remote staging path')
        self.staged[card['bdf']] = path
        # Send only the card repository and shared verifier/deployment code.
        with tempfile.TemporaryFile() as stream:
            with tarfile.open(fileobj=stream, mode='w') as archive:
                for name in (*COMMON, 'card'):
                    archive.add(self.root/name, arcname=name)
            stream.seek(0)
            self.ssh(card, 'tar --no-same-owner -xf - -C '+shlex.quote(path), stdin=stream)

    def host(self, phase):
        run(['timeout', '--kill-after=30', '1500', 'unshare', '--net', 'python3', '-B', str(self.root/'deploy.py'), 'host', phase])

    def privileged(self, card, command):
        sudo = ['sudo', '-n'] if self.sudo_password is None else ['sudo', '-S', '-p', '']
        kwargs = {} if self.sudo_password is None else {'input': self.sudo_password+'\n', 'text': True}
        return self.ssh(card, shlex.join(sudo + command), **kwargs)

    def card(self, card, phase):
        command = ['timeout', '--kill-after=30', '1500', 'flock', '-n', '/run/neat-offline-install.lock', 'unshare', '--net', 'python3', '-B',
                   self.staged[card['bdf']]+'/deploy.py', 'card', phase]
        self.privileged(card, command)

    def cleanup(self, card):
        if card['bdf'] in self.staged:
            self.privileged(card, ['rm', '-rf', '--', self.staged[card['bdf']]])


def orchestrate(cards, snapshot, operations, preflight_only=False):
    report = {'host': 'not_attempted', 'cards': [
        {'bdf': c['bdf'], 'ip': c['ip'], 'status': 'not_attempted'} for c in cards], 'errors': []}

    def stable():
        if operations.topology() != snapshot:
            raise ValueError('PCI topology changed; stop and rerun discovery')
        for card in cards:
            operations.route(card)

    try:
        stable()
        report['host'] = 'preflight_failed'
        operations.host('preflight')
        report['host'] = 'preflight_passed'
        for card, row in zip(cards, report['cards']):
            stable()
            row['status'] = 'preflight_failed'
            operations.stage(card)
            operations.card(card, 'preflight')
            row['status'] = 'preflight_passed'
        stable()
        if preflight_only:
            report['host'] = 'preflight_passed'
            return report
        report['host'] = 'install_failed'
        operations.host('install')
        report['host'] = 'installed_verified'
        for card, row in zip(cards, report['cards']):
            stable()
            try:
                operations.card(card, 'install')
                row['status'] = 'installed_verified'
            except (ConnectionError, subprocess.TimeoutExpired) as exc:
                row['status'] = 'outcome_unknown'
                row['error'] = str(exc)
                raise
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row['status'] = 'failed'
                row['error'] = str(exc)
                report['errors'].append(card['bdf']+': '+str(exc))
                report['host'] = 'verification_failed'
                operations.host('verify')
                report['host'] = 'installed_verified'
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report['errors'].append(str(exc))
    finally:
        for card, row in zip(cards, report['cards']):
            if row['status'] == 'outcome_unknown':
                row['retained_staging'] = getattr(operations, 'staged', {}).get(card['bdf'])
                continue
            try:
                operations.cleanup(card)
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                report['errors'].append('Staging cleanup failed for '+card['bdf']+': '+str(exc))
    return report


def main():
    home = Path(pwd.getpwuid(int(os.environ.get('SUDO_UID', os.getuid()))).pw_dir)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-user', default='sima')
    parser.add_argument('--ssh-key', type=Path, default=home/'.ssh/sima_neat_pcie_ed25519')
    parser.add_argument('--known-hosts', type=Path, default=home/'.ssh/known_hosts')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--ask-card-sudo', action='store_true',
                        help='prompt once for the shared card sudo password; never saved')
    args = parser.parse_args()
    if sys.platform != 'linux' or os.geteuid() != 0:
        parser.error('run with sudo on the prepared Linux PCIe host')
    import re
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', args.ssh_user):
        parser.error('invalid SSH user')
    if not args.ssh_key.is_file() or not args.known_hosts.is_file():
        parser.error('provision SSH key access and trusted host keys before installing')
    for command in ('ip', 'ssh', 'unshare', 'timeout'):
        if not shutil.which(command):
            parser.error('system prerequisite missing: ' + command)
    root = Path(__file__).resolve().parent
    if any(c.isspace() for c in str(root)):
        parser.error('extract the bundle into a path without whitespace')
    try:
        with open('/run/neat-offline-install.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            manifest = verify_bundle(root)
            snapshot = discover.inventory(PCI_ROOT)
            evidence = discover.collect(PCI_ROOT, 2.0)
            cards = endpoints(evidence)
            observed = [{k: c[k] for k in ('bdf', 'device_id', 'interfaces')} for c in cards]
            if observed != snapshot:
                raise ValueError('PCI inventory changed before discovery')
            password = getpass.getpass('Card sudo password (all cards): ') if args.ask_card_sudo else None
            operations = Operations(root, args.ssh_user, args.ssh_key.resolve(), args.known_hosts.resolve(), password)
            report = orchestrate(cards, snapshot, operations, args.preflight_only)
            report['core_revision'] = manifest['core_revision']
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report = {'errors': [str(exc)]}
    print(json.dumps(report, indent=2))
    return bool(report['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
