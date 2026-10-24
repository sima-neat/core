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
    ops.card_passwords['0000:01:00.0'] = None
    monkeypatch.setattr(ops, 'ssh', lambda card, command, **kwargs: commands.append(command))
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
    ops.card_passwords['test'] = 'test-secret'
    ops.privileged({'bdf': 'test'}, ['true'])
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


@pytest.mark.parametrize('results, expected', [([True], [None]), ([False, True], [None, 'edgeai'])])
def test_sudo_fallback_order_and_cache(tmp_path, monkeypatch, results, expected):
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known')
    attempts = []
    def probe(card, password):
        attempts.append(password)
        return results[len(attempts)-1]
    monkeypatch.setattr(ops, 'sudo_probe', probe)
    ops.authenticate(CARDS[0])
    ops.authenticate(CARDS[0])
    assert attempts == expected


def test_sudo_prompts_only_after_default_rejected(tmp_path, monkeypatch):
    import io
    import builtins
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known')
    attempts = []
    def probe(card, password):
        attempts.append(password)
        return password == 'custom-secret'
    monkeypatch.setattr(ops, 'sudo_probe', probe)
    monkeypatch.setattr(builtins, 'open', lambda *args: io.StringIO())
    monkeypatch.setattr(installer.getpass, 'getpass', lambda *args, **kwargs: 'custom-secret')
    ops.authenticate(CARDS[0])
    assert attempts == [None, 'edgeai', 'custom-secret']
    assert ops.card_passwords[CARDS[0]['bdf']] == 'custom-secret'
    assert CARDS[1]['bdf'] not in ops.card_passwords


@pytest.mark.parametrize('message, retry', [('sudo: a password is required', True),
    ('sudo: 1 incorrect password attempt', True), ('user is not in the sudoers file', False),
    ('sudo: command not found', False)])
def test_sudo_probe_only_retries_authentication(tmp_path, monkeypatch, message, retry):
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known')
    def fail(card, command, **kwargs):
        assert 'edgeai' not in command
        assert kwargs['input'] == 'edgeai\n'
        raise subprocess.CalledProcessError(1, command, stderr=message)
    monkeypatch.setattr(ops, 'ssh', fail)
    if retry:
        assert not ops.sudo_probe(CARDS[0], 'edgeai')
    else:
        with pytest.raises(ValueError, match='sudo access check failed'):
            ops.sudo_probe(CARDS[0], 'edgeai')


def test_sudo_transport_failure_does_not_prompt(tmp_path, monkeypatch):
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known')
    def fail(*args, **kwargs):
        raise ConnectionError('transport lost')
    monkeypatch.setattr(ops, 'ssh', fail)
    with pytest.raises(ConnectionError):
        ops.authenticate(CARDS[0])


def test_sudo_no_terminal_fails_without_echoing(tmp_path, monkeypatch):
    import builtins
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known')
    monkeypatch.setattr(ops, 'sudo_probe', lambda *args: False)
    def absent(*args):
        raise OSError('no terminal')
    monkeypatch.setattr(builtins, 'open', absent)
    with pytest.raises(ValueError, match='interactive terminal'):
        ops.authenticate(CARDS[0])


def test_human_summary_preflight_is_not_an_update():
    import io
    stream = io.StringIO()
    report = {'host': 'preflight_passed', 'errors': [], 'cards': [
        dict(CARDS[0], status='preflight_passed', ethernet_ports=[
            {'interface': 'end3', 'mac': '00:11:22:33:44:55'}])]}
    installer.Progress(stream).summary(report, True)
    text = stream.getvalue()
    assert '1/1 cards passed. No packages were installed.' in text
    assert 'virtual IP 10.0.1.2' in text
    assert 'Primary Ethernet: end3 | MAC 00:11:22:33:44:55' in text
    assert '\033' not in text


def test_summary_counts_only_verified_updates_and_displays_failures():
    import io
    stream = io.StringIO()
    report = {'host': 'installed_verified', 'errors': ['card failed'], 'cards': [
        dict(CARDS[0], status='installed_verified'), dict(CARDS[1], status='outcome_unknown')]}
    installer.Progress(stream).summary(report)
    assert '1/2 cards installed and verified' in stream.getvalue()
    assert 'outcome unknown' in stream.getvalue()
    assert 'ERROR: card failed' in stream.getvalue()


def test_color_respects_terminal_and_no_color(monkeypatch):
    import io
    stream = io.StringIO()
    stream.isatty = lambda: True
    monkeypatch.delenv('NO_COLOR', raising=False)
    monkeypatch.setenv('TERM', 'xterm')
    installer.Progress(stream).show('Passed', 'ok')
    assert '\033[32mPassed\033[0m' in stream.getvalue()
    monkeypatch.setenv('NO_COLOR', '')
    assert not installer.Progress(stream).color
    assert installer.safe_text('remote\033[2J\nmessage') == 'remote [2J message'


def test_ethernet_identity_rejects_terminal_injection():
    import json
    with pytest.raises(ValueError, match='inventory'):
        installer.ethernet_ports(json.dumps([{'interface': 'end3\033', 'mac': '00:11:22:33:44:55'}]))


@pytest.mark.parametrize('fails', [False, True])
def test_live_progress_preserves_result_and_failure_and_stops(monkeypatch, fails):
    import io
    import time
    import threading
    stream = io.StringIO()
    stream.isatty = lambda: True
    monkeypatch.setenv('TERM', 'xterm')
    threads = set(threading.enumerate())
    def operation():
        time.sleep(1.1)
        if fails:
            raise ValueError('test failure')
        return 42
    if fails:
        with pytest.raises(ValueError, match='test failure'):
            installer.Progress(stream).run('Installing card...', operation)
    else:
        assert installer.Progress(stream).run('Installing card...', operation) == 42
    assert '00:01 elapsed' in stream.getvalue()
    assert stream.getvalue().endswith('\n')
    assert set(threading.enumerate()) == threads


def test_install_output_tail_is_bounded_and_scoped_to_phase(tmp_path):
    import io
    log = tmp_path/'output.log'
    log.write_text('previous phase\n')
    start = log.stat().st_size
    with log.open('a') as stream:
        stream.write('x'*70000+'\n')
        stream.write('\n'.join(f'Installing package {n}' for n in range(20))+'\n')
        stream.write('dpkg-query: no path found matching pattern /lib/libc.so\n')
        stream.write('Err:3 file:/bundle/repo ./ Packages\n  Method gave a blank filename\n')
        stream.write('\033[31mERROR: real install failure\033[0m\n')
    progress = installer.Progress(io.StringIO())
    progress.log_path = log
    lines = progress.tail(start)
    assert len(lines) == 10
    assert lines[0] == 'Installing package 11'
    assert lines[-1] == 'ERROR: real install failure'
    assert not any('previous' in line or '\033' in line for line in lines)


def test_install_panel_finishes_with_last_ten_lines(tmp_path, monkeypatch):
    import io
    log = tmp_path/'output.log'
    log.write_text('previous phase\n')
    stream = io.StringIO()
    stream.isatty = lambda: True
    monkeypatch.setenv('TERM', 'xterm')
    monkeypatch.setattr(installer.shutil, 'get_terminal_size', lambda fallback: installer.os.terminal_size((80, 24)))
    progress = installer.Progress(stream)
    progress.log_path = log
    def install(phase):
        assert phase == 'install'
        with log.open('a') as output:
            output.write('\n'.join(f'Package {n} installed' for n in range(20))+'\n')
    progress.run('Installing card...', install, 'install')
    output = stream.getvalue()
    assert 'Package 9 installed' not in output
    assert all(f'Package {n} installed' in output for n in range(10, 20))
    assert 'previous phase' not in output
    assert 'elapsed | Installing card' in output
    assert installer.Progress.clip('ab界cd', 5) == 'ab界c'


@pytest.mark.parametrize('fails', [False, True])
def test_verbose_streams_all_output_and_drains_on_failure(tmp_path, fails):
    import io
    log = tmp_path/'output.log'
    log.write_bytes(b'')
    stream = io.StringIO()
    progress = installer.Progress(stream, verbose=True)
    progress.log_path = log
    content = ''.join(f'line {n}\n' for n in range(30)) + 'dpkg-query: diagnostic\nlast partial line'
    def operation():
        log.write_text(content)
        if fails:
            raise ValueError('test failure')
        return 42
    def execute():
        with progress.stream_log():
            return progress.run('Installing...', operation)
    if fails:
        with pytest.raises(ValueError, match='test failure'):
            execute()
    else:
        assert execute() == 42
    assert stream.getvalue() == 'Installing...\n'+content
    assert log.read_text() == content


def test_verbose_output_arrives_before_operation_finishes(tmp_path):
    import io
    import time
    log = tmp_path/'live.log'
    log.write_text('')
    stream = io.StringIO()
    progress = installer.Progress(stream, verbose=True)
    progress.log_path = log
    with progress.stream_log():
        log.write_text('live package output\n')
        deadline = time.monotonic() + 2
        while 'live package output' not in stream.getvalue() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert 'live package output' in stream.getvalue()
