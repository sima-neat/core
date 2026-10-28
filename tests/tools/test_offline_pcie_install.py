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

CARDS = [{'bdf': f'0000:0{i}:00.0', 'ip': f'10.0.{i}.2',
          'local_ip': f'10.0.{i}.1', 'interface': f'pci{i}'} for i in (1, 2, 3)]


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


@pytest.mark.parametrize('change', ['topology', 'route'])
def test_final_card_change_prevents_success_and_preserves_cleanup(change):
    class ChangedAfterLastCard(Fake):
        def topology(self):
            if change == 'topology' and (CARDS[-1]['bdf'], 'install') in self.events:
                return ['changed']
            return super().topology()

        def route(self, card):
            if change == 'route' and (CARDS[-1]['bdf'], 'install') in self.events:
                raise ValueError('PCIe route changed')

    ops = ChangedAfterLastCard()
    report = installer.orchestrate(CARDS, ['original'], ops)
    assert report['errors'] and change in report['errors'][0]
    assert report['host'] == 'installed_verified'
    assert all(row['status'] == 'installed_verified' for row in report['cards'])
    assert [event for event in ops.events if event[1] == 'cleanup'] == [
        (card['bdf'], 'cleanup') for card in CARDS]


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


def test_host_package_commands_run_without_network_or_terminal_input(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(installer, 'run_host', lambda args, **kw: calls.append((args, kw)))
    installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known').host('install')
    args, options = calls[0]
    assert args[:6] == ['timeout', '--foreground', '--kill-after=30', '1500', 'unshare', '--net']
    assert options['stdin'] == subprocess.DEVNULL


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
    for name in ('assemble.py', 'log_progress.py', 'install.py', 'discover.py', 'pcie-setup.sh', 'deploy.py', 'lifecycle.sh', 'INSTALL.md',
                 'card/target.json', 'card/payload/test.deb', 'host/target.json'):
        path = tmp_path/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture')
        files[name] = assembly.sha(path)
    (tmp_path/'bundle.json').write_text(json.dumps({'files': files}))
    (tmp_path/'host/target.json').unlink()
    (tmp_path/'install.py').unlink()
    (tmp_path/'discover.py').unlink()
    (tmp_path/'pcie-setup.sh').unlink()
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
    def absent(*args, **kwargs):
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


@pytest.mark.parametrize('interrupted_index', [0, 1])
def test_interrupted_card_retains_staging_and_stops_remaining_installs(interrupted_index):
    class Interrupted(Fake):
        def __init__(self):
            super().__init__()
            self.staged = {c['bdf']: '/var/tmp/staged-' + str(i) for i, c in enumerate(CARDS)}

        def card(self, card, phase):
            super().card(card, phase)
            if phase == 'install' and card == CARDS[interrupted_index]:
                raise KeyboardInterrupt()

    ops = Interrupted()
    report = installer.orchestrate(CARDS, ['original'], ops)
    assert report['host'] == 'installed_verified'
    assert report['errors'] and 'interrupted' in report['errors'][0]
    for i, (card, row) in enumerate(zip(CARDS, report['cards'])):
        if i == interrupted_index:
            assert row['status'] == 'outcome_unknown'
            assert 'remote outcome unknown' in row['error']
            assert row['retained_staging'] == ops.staged[card['bdf']]
            assert (card['bdf'], 'cleanup') not in ops.events
        else:
            assert (card['bdf'], 'cleanup') in ops.events
            assert row['status'] == ('installed_verified' if i < interrupted_index else 'preflight_passed')
        if i > interrupted_index:
            assert (card['bdf'], 'install') not in ops.events
    assert ('host', 'verify') not in ops.events


@pytest.mark.parametrize('recreate', [False, True])
def test_recreate_option_reaches_both_host_and_card(tmp_path, monkeypatch, recreate):
    calls = []
    monkeypatch.setattr(installer, 'run_host', lambda args, **kwargs: calls.append(args))
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'key', tmp_path/'known', recreate_venv=recreate)
    ops.host('install')
    ops.staged['card'] = '/var/tmp/neat-offline-abcdefgh'
    monkeypatch.setattr(ops, 'privileged', lambda card, command: calls.append(command))
    ops.card({'bdf': 'card'}, 'install')
    assert len(calls) == 2
    assert all(('--recreate-venv' in args) == recreate for args in calls)


@pytest.fixture
def ssh_setup(tmp_path, monkeypatch):
    import io
    import builtins
    from types import SimpleNamespace
    ops = installer.Operations(tmp_path, 'sima', tmp_path/'home/.ssh/key', tmp_path/'home/.ssh/known_hosts')
    account = SimpleNamespace(pw_uid=1234, pw_gid=5678, pw_name='owner', pw_dir=str(tmp_path/'home'))
    monkeypatch.setattr(ops, 'topology', lambda: ['original'])
    monkeypatch.setattr(ops, 'route', lambda card: None)
    calls = []
    terminal = io.StringIO()
    monkeypatch.setattr(builtins, 'open', lambda *args, **kwargs: terminal)
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, '')
    monkeypatch.setattr(installer, 'run', run)
    return SimpleNamespace(ops=ops, account=account, calls=calls, terminal=terminal,
                           progress=installer.Progress(io.StringIO()))


def prepare(setup, **kwargs):
    installer.prepare_ssh(CARDS, ['original'], setup.ops, setup.account, setup.progress, **kwargs)


def test_fresh_host_delegates_every_discovered_card_to_shared_setup(ssh_setup):
    setup = ssh_setup
    prepare(setup, create_key=True)
    provisioning = [(args, kw) for args, kw in setup.calls if args[0] == 'bash']
    assert len(provisioning) == len(CARDS)
    for (args, kw), card in zip(provisioning, CARDS):
        assert args[1] == str(setup.ops.root/'pcie-setup.sh')
        for option, value in {'--hosts': card['ip'], '--user': 'sima',
                              '--key': str(setup.ops.key)}.items():
            assert args[args.index(option)+1] == value
        assert '--strict' in args and '--no-discover' in args
        assert kw['user'] == 1234 and kw['group'] == 5678 and kw['extra_groups'] == []
        assert kw['env']['HOME'] == setup.account.pw_dir
        assert kw['stdin'] is setup.terminal and kw['stderr'] is setup.terminal
        assert 'capture_output' not in kw
    verification = [args for args, kw in setup.calls if args[0] == 'ssh']
    assert [args[-2] for args in verification] == ['sima@'+card['ip'] for card in CARDS]
    assert all('StrictHostKeyChecking=yes' in args and 'UpdateHostKeys=no' in args for args in verification)
    assert not setup.ops.card_passwords


def test_existing_key_without_public_file_is_delegated_without_replacement(ssh_setup):
    setup = ssh_setup
    setup.ops.key.parent.mkdir(parents=True)
    setup.ops.key.write_text('existing private key')
    prepare(setup)
    assert setup.ops.key.read_text() == 'existing private key'
    assert len([args for args, kw in setup.calls if args[0] == 'bash']) == len(CARDS)


def test_working_access_never_opens_terminal_or_provisions(ssh_setup, monkeypatch):
    import builtins
    setup = ssh_setup
    setup.ops.key.parent.mkdir(parents=True)
    setup.ops.key.touch()
    setup.ops.known_hosts.touch()
    monkeypatch.setattr(builtins, 'open', lambda *args: pytest.fail('unexpected terminal prompt'))
    prepare(setup)
    assert len(setup.calls) == len(CARDS)
    assert all('BatchMode=yes' in args for args, kw in setup.calls)


@pytest.mark.parametrize('existing', [False, True])
def test_preflight_does_not_provision_keys_or_trust(ssh_setup, monkeypatch, existing):
    setup = ssh_setup
    if existing:
        setup.ops.key.parent.mkdir(parents=True)
        setup.ops.key.touch()
        setup.ops.known_hosts.touch()
        def fail(*args, **kwargs):
            raise ConnectionError('untrusted host')
        monkeypatch.setattr(setup.ops, 'ssh', fail)
    with pytest.raises(ValueError, match='without --preflight-only'):
        prepare(setup, preflight_only=True, create_key=True)
    assert not setup.calls
    assert setup.ops.key.exists() == existing


def test_no_terminal_cannot_create_credentials(ssh_setup, monkeypatch):
    import builtins
    def absent(*args, **kwargs):
        raise OSError('no terminal')
    monkeypatch.setattr(builtins, 'open', absent)
    with pytest.raises(ValueError, match='interactive terminal'):
        prepare(ssh_setup, create_key=True)
    assert not ssh_setup.calls
    assert not ssh_setup.ops.key.exists()


def test_missing_explicit_key_is_not_silently_created(ssh_setup):
    with pytest.raises(ValueError, match='omit --ssh-key'):
        prepare(ssh_setup)
    assert not ssh_setup.calls


def test_setup_failure_stops_before_package_or_other_card_changes(ssh_setup, monkeypatch):
    setup = ssh_setup
    original = installer.run
    def reject(args, **kwargs):
        if args[0] == 'bash':
            assert '--host-key-checking' not in args
            raise subprocess.CalledProcessError(1, args)
        return original(args, **kwargs)
    monkeypatch.setattr(installer, 'run', reject)
    with pytest.raises(subprocess.CalledProcessError):
        prepare(setup, create_key=True)
    assert not setup.calls
    assert not setup.ops.staged


def test_topology_change_stops_ssh_setup_before_any_mutation(ssh_setup, monkeypatch):
    monkeypatch.setattr(ssh_setup.ops, 'topology', lambda: ['changed'])
    with pytest.raises(ValueError, match='topology changed'):
        prepare(ssh_setup, create_key=True)
    assert not ssh_setup.calls


def test_route_failure_stops_ssh_setup_before_any_mutation(ssh_setup, monkeypatch):
    def fail(card):
        raise ValueError('not direct through its PCI-owned interface')
    monkeypatch.setattr(ssh_setup.ops, 'route', fail)
    with pytest.raises(ValueError, match='PCI-owned'):
        prepare(ssh_setup, create_key=True)
    assert not ssh_setup.calls


def test_failed_batch_verification_stops_remaining_card_setup(ssh_setup, monkeypatch):
    setup = ssh_setup
    def fail(*args, **kwargs):
        raise ConnectionError('new key not authorized')
    monkeypatch.setattr(setup.ops, 'ssh', fail)
    with pytest.raises(ConnectionError):
        prepare(setup, create_key=True)
    assert len(setup.calls) == 1 and setup.calls[0][0][0] == 'bash'
    assert not setup.ops.staged


def test_ssh_setup_accepts_a_real_nonseekable_terminal(ssh_setup, monkeypatch):
    import builtins
    import io
    import os
    master, slave = os.openpty()
    def terminal(path, mode, **kwargs):
        assert path == '/dev/tty'
        return io.open(os.open(os.ttyname(slave), os.O_RDWR | os.O_NOCTTY), mode, **kwargs)
    monkeypatch.setattr(builtins, 'open', terminal)
    try:
        prepare(ssh_setup, create_key=True)
    finally:
        os.close(master)
        os.close(slave)


def test_custom_trust_file_requires_prior_provisioning(ssh_setup):
    ssh_setup.ops.known_hosts = ssh_setup.ops.root/'custom_known_hosts'
    with pytest.raises(ValueError, match='omit --known-hosts'):
        prepare(ssh_setup, create_key=True)
    assert not ssh_setup.calls
