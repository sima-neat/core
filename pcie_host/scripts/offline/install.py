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
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True

from assemble import verify_bundle
import discover
from process import run_host
from log_progress import LogProgress, safe_text

PCI_ROOT = Path('/sys/bus/pci/devices')
COMMON = ('bundle.json', 'assemble.py', 'process.py', 'log_progress.py', 'deploy.py', 'lifecycle.sh', 'INSTALL.md')


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


class Progress(LogProgress):
    def summary(self, report, preflight_only=False):
        rows = report.get('cards', [])
        installed = sum(r['status'] == 'installed_verified' for r in rows)
        passed = sum(r['status'] == 'preflight_passed' for r in rows)
        if preflight_only:
            self.show(f'Preflight: {passed}/{len(rows)} cards passed. No packages were installed.',
                      'error' if report['errors'] else 'ok')
        else:
            self.show(f'Installation: {installed}/{len(rows)} cards installed and verified.',
                      'error' if report['errors'] else 'ok')
        self.show('Host: '+report.get('host', 'not_attempted').replace('_', ' '))
        for row in rows:
            self.show(f"Card {row['bdf']} | virtual IP {row['ip']} | {row['status'].replace('_', ' ')}")
            ports = row.get('ethernet_ports', [])
            if len(ports) == 1:
                self.show(f"  Primary Ethernet: {ports[0]['interface']} | MAC {ports[0]['mac']}")
            elif ports:
                self.show('  Primary Ethernet: undetermined (multiple physical ports)')
                for port in ports:
                    self.show(f"  Ethernet: {port['interface']} | MAC {port['mac']}")
            else:
                self.show('  Primary Ethernet MAC: unavailable')
            if row.get('identity_note'):
                self.show('  '+row['identity_note'])
            if row.get('retained_staging'):
                self.show('  Staging retained: '+row['retained_staging'])
        for error in report['errors']:
            self.show('ERROR: '+error, 'error')
        if report.get('log'):
            self.show('Detailed log: '+report['log'])


def ethernet_ports(value):
    """Only physical Ethernet netdevs observed over authenticated card SSH."""
    ports = json.loads(value)
    if not isinstance(ports, list):
        raise ValueError('invalid Ethernet inventory')
    for port in ports:
        if not isinstance(port, dict) or not re.fullmatch(r'[A-Za-z0-9_.:-]+', port.get('interface', '')) or not re.fullmatch(r'(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}', port.get('mac', '')):
            raise ValueError('invalid Ethernet inventory')
    return sorted(ports, key=lambda p: p['interface'])


class Operations:
    def __init__(self, root, user, key, known_hosts, sudo_password=None, log=None, recreate_venv=False):
        self.root, self.user = root, user
        self.key, self.known_hosts = key, known_hosts
        self.staged = {}
        self.sudo_password = sudo_password
        self.card_passwords = {}
        self.log = log
        self.recreate_venv = recreate_venv

    def topology(self):
        return discover.inventory(PCI_ROOT)

    def route(self, card):
        result = run(['ip', '-j', '-4', 'route', 'get', card['ip'], 'from', card['local_ip']],
                     capture_output=True, text=True)
        routes = json.loads(result.stdout)
        if len(routes) != 1 or routes[0].get('dev') != card['interface'] or routes[0].get('gateway'):
            raise ValueError('Card route is not direct through its PCI-owned interface')

    def ssh_command(self, card, command, interactive=False):
        self.route(card)
        return ['ssh', '-F', '/dev/null', '-T', '-i', str(self.key), '-b', card['local_ip'],
                '-o', 'BindInterface='+card['interface'],
                '-o', 'BatchMode='+('no' if interactive else 'yes'),
                '-o', 'IdentitiesOnly=yes',
                '-o', 'StrictHostKeyChecking='+('ask' if interactive else 'yes'),
                '-o', 'UpdateHostKeys=no',
                '-o', 'UserKnownHostsFile='+str(self.known_hosts),
                '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=2', self.user+'@'+card['ip'], command]

    def ssh(self, card, command, **kwargs):
        try:
            return run(self.ssh_command(card, command), **kwargs)
        except subprocess.CalledProcessError as exc:
            if exc.returncode == 255:
                raise ConnectionError('SSH transport lost; remote outcome may be unknown') from exc
            raise

    def stage(self, card):
        self.authenticate(card)
        # Discovery reports the PCIe virtual MAC, not the physical Ethernet MAC.
        script = """import json
from pathlib import Path
ports = []
for p in sorted(Path('/sys/class/net').iterdir()):
    if (p/'device').exists() and (p/'type').read_text().strip() == '1' and not p.name.startswith('veth'):
        ports.append({'interface': p.name, 'mac': (p/'address').read_text().strip()})
print(json.dumps(ports))
"""
        try:
            result = self.ssh(card, 'python3 -c '+shlex.quote(script), capture_output=True, text=True)
            card['ethernet_ports'] = ethernet_ports(result.stdout)
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            card['identity_note'] = 'Ethernet identity could not be read: '+str(exc)

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
        # Keep timeout and its children in the supervised session/process group.
        # Host deployment is noninteractive; do not expose terminal stdin.
        run_host(['timeout', '--foreground', '--kill-after=30', '1500', 'unshare', '--net', 'python3', '-B', str(self.root/'deploy.py'), 'host', phase,
                  *(['--recreate-venv'] if self.recreate_venv else [])], stdin=subprocess.DEVNULL, **self.output_options())

    def output_options(self):
        return {'stdout': self.log, 'stderr': subprocess.STDOUT} if self.log is not None else {}

    def sudo_probe(self, card, password):
        sudo = ['sudo', '-k', '-n'] if password is None else ['sudo', '-k', '-S', '-p', '']
        kwargs = {'stdin': subprocess.DEVNULL} if password is None else {'input': password+'\n'}
        try:
            self.ssh(card, 'LC_ALL=C '+shlex.join(sudo+['true']),
                     capture_output=True, text=True, **kwargs)
            return True
        except subprocess.CalledProcessError as exc:
            error = exc.stderr or ''
            authentication = ('a password is required', 'incorrect password attempt',
                              'Sorry, try again.')
            if exc.returncode == 1 and any(message in error for message in authentication):
                return False
            raise ValueError('Card '+card['bdf']+': sudo access check failed: '+error.strip()) from None

    def authenticate(self, card):
        identity = card['bdf']
        if identity in self.card_passwords:
            return
        candidates = [self.sudo_password] if self.sudo_password is not None else [None, 'edgeai']
        for password in candidates:
            if self.sudo_probe(card, password):
                self.card_passwords[identity] = password
                return
        # A real terminal is required; do not consume redirected installer input
        # or allow getpass to echo a secret in an unattended job.
        try:
            with open('/dev/tty', 'w') as terminal:
                password = getpass.getpass('Sudo password for card '+identity+': ', stream=terminal)
        except (OSError, EOFError):
            raise ValueError('Card '+identity+': sudo authentication failed; rerun in an interactive terminal') from None
        if not self.sudo_probe(card, password):
            raise ValueError('Card '+identity+': sudo password rejected')
        self.card_passwords[identity] = password

    def privileged(self, card, command):
        self.authenticate(card)
        password = self.card_passwords[card['bdf']]
        sudo = ['sudo', '-n'] if password is None else ['sudo', '-S', '-p', '']
        kwargs = {'stdin': subprocess.DEVNULL} if password is None else {'input': password+'\n', 'text': True}
        # Never retry a deployment command: it may already have changed packages.
        return self.ssh(card, shlex.join(sudo + command), **self.output_options(), **kwargs)

    def card(self, card, phase):
        command = ['timeout', '--kill-after=30', '1500', 'flock', '-n', '/run/neat-offline-install.lock', 'unshare', '--net', 'python3', '-B',
                   self.staged[card['bdf']]+'/deploy.py', 'card', phase,
                   *(['--recreate-venv'] if self.recreate_venv else [])]
        self.privileged(card, command)

    def cleanup(self, card):
        if card['bdf'] in self.staged:
            self.privileged(card, ['rm', '-rf', '--', self.staged[card['bdf']]])


def prepare_ssh(cards, snapshot, operations, account, progress, *, preflight_only=False, create_key=False):
    """Bootstrap only discovered PCIe endpoints, with OpenSSH's explicit trust prompt."""
    def stable(card):
        if operations.topology() != snapshot:
            raise ValueError('PCI topology changed; stop and rerun discovery')
        operations.route(card)

    missing = []
    for card in cards:
        stable(card)
        if operations.key.is_file() and operations.known_hosts.is_file():
            try:
                operations.ssh(card, 'true', stdin=subprocess.DEVNULL, capture_output=True, text=True)
                continue
            except ConnectionError:
                pass
        missing.append(card)
    if not missing:
        return
    if preflight_only:
        raise ValueError('SSH setup required for card(s) '+', '.join(c['bdf'] for c in missing)+
                         '; rerun sudo python3 -B install.py in an interactive terminal without --preflight-only')
    if not operations.key.is_file() and not create_key:
        raise ValueError('SSH key not found: '+str(operations.key)+
                         '; supply an existing key or omit --ssh-key to create the default key')
    # Open the terminal before creating files. Redirected stdin is never used for
    # passwords, and authentication output never enters the diagnostic log/JSON.
    try:
        terminal = open('/dev/tty', 'r+b', buffering=0)
    except OSError:
        raise ValueError('SSH setup requires an interactive terminal; rerun sudo python3 -B install.py interactively') from None
    with terminal:
        if not shutil.which('ssh-keygen'):
            raise ValueError('system prerequisite missing: ssh-keygen (install openssh-client)')
        identity = {'user': account.pw_uid, 'group': account.pw_gid, 'extra_groups': [],
                    'env': dict(os.environ, HOME=account.pw_dir, USER=account.pw_name, LOGNAME=account.pw_name)}
        def local(command, **kwargs):
            return run(command, **identity, **kwargs)

        if not operations.key.is_file():
            progress.show('Creating SSH key '+str(operations.key)+' for '+account.pw_name+'...')
            local(['mkdir', '-p', '-m', '700', str(operations.key.parent)], stdin=subprocess.DEVNULL)
            local(['ssh-keygen', '-q', '-t', 'ed25519', '-N', '', '-f', str(operations.key)],
                  stdin=subprocess.DEVNULL, stdout=terminal, stderr=terminal)
        # Derive from the selected private key, even when its .pub file is missing
        # or stale. Batch deployment requires a key without a passphrase.
        try:
            public = local(['ssh-keygen', '-y', '-P', '', '-f', str(operations.key)],
                           stdin=subprocess.DEVNULL, capture_output=True, text=True).stdout.strip()
        except subprocess.CalledProcessError:
            raise ValueError('Cannot read SSH key '+str(operations.key)+
                             '; use a readable key without a passphrase for unattended card operations') from None
        if len(public.split()) < 2 or '\n' in public:
            raise ValueError('Invalid SSH public key')
        # Ignore comments in the derived key so retries do not append duplicates.
        public = ' '.join(public.split()[:2])
        key = shlex.quote(public)
        command = ('umask 077; mkdir -p "$HOME/.ssh" && '
                   'touch "$HOME/.ssh/authorized_keys" && '
                   'chmod 700 "$HOME/.ssh" && chmod 600 "$HOME/.ssh/authorized_keys" && '
                   '(grep -qF -- '+key+' "$HOME/.ssh/authorized_keys" || '
                   "printf '\\n%s\\n' "+key+' >> "$HOME/.ssh/authorized_keys")')
        local(['mkdir', '-p', '-m', '700', str(operations.known_hosts.parent)], stdin=subprocess.DEVNULL)
        for card in missing:
            stable(card)
            progress.show('Setting up SSH for card '+card['bdf']+' ('+card['ip']+'). '
                          'Verify the host fingerprint when prompted, then enter the card login password if requested.')
            # OpenSSH refuses changed keys and asks before trusting new ones.
            # Run as the invoking user so new credentials remain user-owned.
            local(operations.ssh_command(card, command, interactive=True),
                  stdin=terminal, stdout=terminal, stderr=terminal)
            operations.ssh(card, 'true', stdin=subprocess.DEVNULL, capture_output=True, text=True)
            progress.show('Card '+card['bdf']+' SSH access ready.', 'ok')


def orchestrate(cards, snapshot, operations, preflight_only=False, progress=None):
    notify = progress.show if progress else lambda *args: None
    def perform(message, operation, *args):
        if progress:
            return progress.run(message, operation, *args)
        return operation(*args)

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
        perform('Checking host prerequisites and offline packages...', operations.host, 'preflight')
        notify('Host preflight passed.', 'ok')
        report['host'] = 'preflight_passed'
        for card, row in zip(cards, report['cards']):
            stable()
            row['status'] = 'preflight_failed'
            notify(f"Checking card {card['bdf']} ({card['ip']}) and staging offline packages...")
            operations.stage(card)
            row.update({k: card[k] for k in ('ethernet_ports', 'identity_note') if k in card})
            perform(f"Checking card {card['bdf']} offline compatibility...", operations.card, card, 'preflight')
            notify(f"Card {card['bdf']} preflight passed.", 'ok')
            row['status'] = 'preflight_passed'
        stable()
        if preflight_only:
            report['host'] = 'preflight_passed'
            return report
        report['host'] = 'install_failed'
        perform('Installing and verifying Neat on the host...', operations.host, 'install')
        notify('Host installed and verified.', 'ok')
        report['host'] = 'installed_verified'
        for card, row in zip(cards, report['cards']):
            stable()
            try:
                # Until SSH confirms completion, even a local interruption cannot
                # establish that the privileged remote transaction has stopped.
                row['status'] = 'outcome_unknown'
                perform(f"Installing and verifying card {card['bdf']} ({card['ip']})...", operations.card, card, 'install')
                notify(f"Card {card['bdf']} installed and verified.", 'ok')
                row['status'] = 'installed_verified'
            except (ConnectionError, subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                row['status'] = 'outcome_unknown'
                row['error'] = str(exc) or 'Installation interrupted; remote outcome unknown'
                raise
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                row['status'] = 'failed'
                row['error'] = str(exc)
                report['errors'].append(card['bdf']+': '+str(exc))
                report['host'] = 'verification_failed'
                perform('Rechecking host health...', operations.host, 'verify')
                report['host'] = 'installed_verified'
        stable()
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        report['errors'].append(str(exc) or 'Installation interrupted; check card outcomes before retrying')
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
    account = pwd.getpwuid(int(os.environ.get('SUDO_UID', os.getuid())))
    home = Path(account.pw_dir)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ssh-user', default='sima')
    parser.add_argument('--ssh-key', type=Path, help='existing private key; default: create/reuse ~/.ssh/sima_neat_pcie_ed25519')
    parser.add_argument('--known-hosts', type=Path, default=home/'.ssh/known_hosts')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--recreate-venv', action='store_true',
                        help='discard existing Python environments on host and cards during installation')
    parser.add_argument('--ask-card-sudo', action='store_true',
                        help='prompt once for the shared card sudo password; never saved')
    parser.add_argument('-v', '--verbose', action='store_true', help='stream all installation logs for debugging; also save the full log')
    parser.add_argument('--json', action='store_true', help='emit the final machine-readable report; progress goes to stderr')
    args = parser.parse_args()
    progress = Progress(sys.stderr if args.json else sys.stdout, verbose=args.verbose)
    if sys.platform != 'linux' or os.geteuid() != 0:
        parser.error('run with sudo on the prepared Linux PCIe host')
    import re
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', args.ssh_user):
        parser.error('invalid SSH user')
    create_key = args.ssh_key is None
    args.ssh_key = args.ssh_key or home/'.ssh/sima_neat_pcie_ed25519'
    for command in ('ip', 'ssh', 'unshare', 'timeout'):
        if not shutil.which(command):
            parser.error('system prerequisite missing: ' + command)
    root = Path(__file__).resolve().parent
    if any(c.isspace() for c in str(root)):
        parser.error('extract the bundle into a path without whitespace')
    try:
        with open('/run/neat-offline-install.lock', 'w') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            progress.show('Verifying offline bundle...')
            manifest = verify_bundle(root)
            progress.show('Discovering cards on PCI-owned interfaces...')
            snapshot = discover.inventory(PCI_ROOT)
            evidence = discover.collect(PCI_ROOT, 2.0)
            cards = endpoints(evidence)
            observed = [{k: c[k] for k in ('bdf', 'device_id', 'interfaces')} for c in cards]
            if observed != snapshot:
                raise ValueError('PCI inventory changed before discovery')
            progress.show(f'Found {len(cards)} card(s). All targets must pass preflight before installation.')
            password = getpass.getpass('Card sudo password (all cards): ') if args.ask_card_sudo else None
            with tempfile.NamedTemporaryFile(mode='w+', errors='replace', prefix='neat-offline-', suffix='.log', dir='/var/tmp', delete=False) as log:
                progress.log_path = Path(log.name)
                progress.show('Detailed log: '+log.name)
                operations = Operations(root, args.ssh_user, args.ssh_key.resolve(), args.known_hosts.resolve(), password, log, args.recreate_venv)
                prepare_ssh(cards, snapshot, operations, account, progress,
                            preflight_only=args.preflight_only, create_key=create_key)
                with progress.stream_log():
                    report = orchestrate(cards, snapshot, operations, args.preflight_only, progress)
                report['log'] = log.name
                if report['errors'] and not args.verbose:
                    log.flush()
                    log.seek(max(0, log.tell() - 8192))
                    progress.show('Recent command diagnostics:', 'error')
                    for line in log.read().splitlines()[-20:]:
                        progress.show(line, 'error')
            report['core_revision'] = manifest['core_revision']
    except (OSError, ValueError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        report = {'errors': [str(exc) or 'Installation interrupted']}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        progress.summary(report, args.preflight_only)
    return bool(report['errors'])


if __name__ == '__main__':
    raise SystemExit(main())
