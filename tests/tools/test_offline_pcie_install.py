"""All-card orchestration: no package changes before complete preflight."""
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).resolve().parents[2]/'pcie_host/scripts/offline'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('offline_install', HERE/'install.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)

CARDS = [{'bdf': f'0000:0{i}:00.0', 'ip': f'10.0.{i}.2'} for i in (1, 2, 3)]


class Fake:
    def __init__(self, fail=None, topology_change=False):
        self.events = []
        self.fail = fail
        self.topology_change = topology_change

    def topology(self):
        if self.topology_change and ('host', 'install') in self.events:
            return ['changed']
        return ['original']

    def route(self, card):
        pass

    def event(self, target, phase):
        self.events.append((target, phase))
        if (target, phase) == self.fail:
            raise subprocess.CalledProcessError(1, 'fixture')

    def host(self, phase):
        self.event('host', phase)

    def card(self, card, phase):
        self.event(card['bdf'], phase)

    def stage(self, card):
        self.event(card['bdf'], 'stage')

    def cleanup(self, card):
        self.event(card['bdf'], 'cleanup')


def test_all_preflight_before_host_and_sequential_cards():
    ops = Fake()
    result = installer.orchestrate(CARDS, ['original'], ops)
    assert not result['errors']
    assert ops.events[:8] == [('host', 'preflight'),
        (CARDS[0]['bdf'], 'stage'), (CARDS[0]['bdf'], 'preflight'),
        (CARDS[1]['bdf'], 'stage'), (CARDS[1]['bdf'], 'preflight'),
        (CARDS[2]['bdf'], 'stage'), (CARDS[2]['bdf'], 'preflight'), ('host', 'install')]
    assert ops.events[8:11] == [(c['bdf'], 'install') for c in CARDS]
    assert all(r['status'] == 'installed_verified' for r in result['cards'])


def test_card_preflight_failure_changes_no_packages_and_cleans_staging():
    ops = Fake((CARDS[1]['bdf'], 'preflight'))
    result = installer.orchestrate(CARDS, ['original'], ops)
    assert result['errors']
    assert not any(phase == 'install' for _, phase in ops.events)
    assert len([e for e in ops.events if e[1] == 'cleanup']) == 3


def test_failure_continues_remaining_cards_after_host_health_check():
    ops = Fake((CARDS[0]['bdf'], 'install'))
    result = installer.orchestrate(CARDS, ['original'], ops)
    assert result['errors']
    assert [r['status'] for r in result['cards']] == ['failed', 'installed_verified', 'installed_verified']
    index = ops.events.index((CARDS[0]['bdf'], 'install'))
    assert ops.events[index+1] == ('host', 'verify')


def test_host_failure_stops_card_mutations():
    ops = Fake(('host', 'install'))
    result = installer.orchestrate(CARDS, ['original'], ops)
    assert result['errors']
    assert not any(t != 'host' and p == 'install' for t, p in ops.events)


def test_topology_change_stops_card_mutations():
    ops = Fake(topology_change=True)
    result = installer.orchestrate(CARDS, ['original'], ops)
    assert 'topology' in result['errors'][0]
    assert not any(t != 'host' and p == 'install' for t, p in ops.events)


def test_preflight_only_never_installs():
    ops = Fake()
    result = installer.orchestrate(CARDS, ['original'], ops, True)
    assert result['host'] == 'preflight_passed'
    assert not any(p == 'install' for _, p in ops.events)


def evidence(ip='10.0.0.2'):
    return {'errors': [], 'cards': [{'bdf': '0000:01:00.0', 'interfaces': ['pci0'],
        'probes': [{'interface': 'pci0', 'local_ip': '10.0.0.1',
                    'responses': [{'source_ip': ip, 'payload': {'ip': ip}}]}]}]}


@pytest.mark.parametrize('ip', ['127.0.0.1', '239.1.1.1', '0.0.0.0', '10.0.0.1'])
def test_invalid_discovery_destinations(ip):
    with pytest.raises(ValueError):
        installer.endpoints(evidence(ip))


def test_never_uses_advertised_address_as_destination():
    data = evidence()
    data['cards'][0]['probes'][0]['responses'][0]['payload']['ip'] = '192.168.1.10'
    with pytest.raises(ValueError, match='differs'):
        installer.endpoints(data)


def test_duplicate_card_ips_are_not_silently_deduplicated():
    data = evidence()
    data['cards'].append(data['cards'][0])
    with pytest.raises(ValueError, match='Duplicate'):
        installer.endpoints(data)


def test_ssh_binds_interface_and_requires_trusted_keys(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(installer, 'run', lambda args, **kw: calls.append(args))
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known')
    monkeypatch.setattr(ops, 'route', lambda card: None)
    ops.ssh({'ip': '10.0.0.2', 'local_ip': '10.0.0.1', 'interface': 'pci0'}, 'true')
    assert 'BindInterface=pci0' in calls[0]
    assert 'StrictHostKeyChecking=yes' in calls[0]
    assert calls[0][-2] == 'sima@10.0.0.2'


def test_host_package_commands_run_without_network(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(installer, 'run', lambda args, **kw: calls.append(args))
    installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known').host('install')
    assert calls[0][:5] == ['timeout', '--kill-after=30', '1500', 'unshare', '--net']


def test_remote_package_commands_are_locked_and_network_isolated(monkeypatch, tmp_path):
    commands = []
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known')
    ops.staged['0000:01:00.0'] = '/var/tmp/neat-offline-abcdefgh'
    monkeypatch.setattr(ops, 'ssh', lambda card, command: commands.append(command))
    ops.card({'bdf': '0000:01:00.0'}, 'install')
    assert commands[0].startswith('sudo -n timeout --kill-after=30 1500 flock -n /run/neat-offline-install.lock unshare --net python3 -B')


def test_ssh_rejects_physical_ethernet_route(monkeypatch, tmp_path):
    import json
    def command(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, json.dumps([{'dev': 'eth0'}]))
    monkeypatch.setattr(installer, 'run', command)
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known')
    with pytest.raises(ValueError, match='PCI-owned'):
        ops.route({'ip': '10.0.0.2', 'local_ip': '10.0.0.1', 'interface': 'pci0'})


def test_staged_card_manifest_verification(tmp_path):
    import json
    assembly = sys.modules['assemble']
    files = {}
    for name in ('assemble.py', 'install.py', 'discover.py', 'deploy.py', 'lifecycle.sh', 'INSTALL.md',
                 'card/target.json', 'card/payload/test.deb', 'host/target.json'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
        files[name] = assembly.sha(path)
    (tmp_path/'bundle.json').write_text(json.dumps({'files': files}))
    (tmp_path/'host/target.json').unlink()
    (tmp_path/'install.py').unlink()
    (tmp_path/'discover.py').unlink()
    assembly.verify_bundle(tmp_path, 'card')
    (tmp_path/'card/payload/test.deb').write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        assembly.verify_bundle(tmp_path, 'card')


def test_failed_host_recheck_stops_after_card_failure():
    class BrokenHost(Fake):
        def host(self, phase):
            super().host(phase)
            if phase == 'verify':
                raise ValueError('host unhealthy')
    ops = BrokenHost((CARDS[0]['bdf'], 'install'))
    report = installer.orchestrate(CARDS, ['original'], ops)
    assert 'host unhealthy' in report['errors']
    assert (CARDS[1]['bdf'], 'install') not in ops.events


def test_card_sudo_password_uses_stdin_not_command(monkeypatch, tmp_path):
    captured = []
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known', 'test-secret')
    monkeypatch.setattr(ops, 'ssh', lambda card, command, **kwargs: captured.append((command, kwargs)))
    ops.privileged({}, ['true'])
    command, kwargs = captured[0]
    assert 'test-secret' not in command
    assert 'sudo -S' in command
    assert kwargs['input'] == 'test-secret\n'


def test_transport_loss_stops_instead_of_overlapping_remote_installs():
    class LostConnection(Fake):
        def card(self, card, phase):
            super().card(card, phase)
            if phase == 'install':
                raise ConnectionError('SSH transport lost')
    ops = LostConnection()
    report = installer.orchestrate(CARDS, ['original'], ops)
    assert report['cards'][0]['status'] == 'outcome_unknown'
    assert (CARDS[0]['bdf'], 'cleanup') not in ops.events
    assert (CARDS[1]['bdf'], 'install') not in ops.events
