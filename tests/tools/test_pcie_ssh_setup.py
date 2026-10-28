"""Shared online/offline SSH provisioning uses consistent options and trust policy."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[2]/'pcie_host/scripts/pcie-setup.sh'


@pytest.fixture
def setup_commands(tmp_path):
    commands = tmp_path/'bin'
    commands.mkdir()
    log = tmp_path/'commands.jsonl'
    for command in ('ssh', 'ssh-copy-id', 'ssh-keyscan'):
        path = commands/command
        path.write_text(f'#!{sys.executable}\n'+'''import json, os, sys
from pathlib import Path
with open(os.environ['SETUP_LOG'], 'a') as log:
    log.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]])+'\\n')
name = Path(sys.argv[0]).name
sys.exit(int(os.environ.get('SSH_RESULT', '255')) if name == 'ssh' else int(os.environ.get('COPY_RESULT', '0')))
''')
        path.chmod(0o755)
    home = tmp_path/'home'
    home.mkdir()
    env = dict(os.environ, HOME=str(home), PATH=str(commands)+os.pathsep+os.environ['PATH'], SETUP_LOG=str(log))
    env.pop('SIMAPCIE_SSH_PASSWORD', None)
    def run(*options, **updates):
        result = subprocess.run(['bash', str(SCRIPT), '--hosts', '10.0.0.2', '--strict', *options],
                                env=dict(env, **updates), capture_output=True, text=True, timeout=30)
        calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        return result, calls, home
    return run


def test_default_online_setup_creates_credentials_and_accepts_new_keys(setup_commands):
    result, calls, home = setup_commands()
    assert result.returncode == 0, result.stderr
    assert (home/'.ssh/sima_neat_pcie_ed25519').is_file()
    assert (home/'.ssh/sima_neat_pcie_ed25519.pub').is_file()
    assert (home/'.ssh/sima_neat_pcie_ed25519').stat().st_mode & 0o777 == 0o600
    assert [call[0] for call in calls] == ['ssh', 'ssh-copy-id']
    assert 'StrictHostKeyChecking=yes' in calls[0]
    assert 'StrictHostKeyChecking=accept-new' in calls[1]


def test_offline_options_reach_probe_and_key_enrollment(setup_commands, tmp_path):
    known = tmp_path/'trust/known_hosts'
    result, calls, home = setup_commands('--known-hosts', str(known), '--interface', 'pci0',
                                        '--source-ip', '10.0.0.1', '--ssh-config', '/dev/null',
                                        '--host-key-checking', 'ask')
    assert result.returncode == 0, result.stderr
    for call in calls:
        assert 'BindInterface=pci0' in call
        assert 'BindAddress=10.0.0.1' in call
        assert 'UserKnownHostsFile='+str(known) in call
        assert 'UpdateHostKeys=no' in call
        assert call[call.index('-F')+1] == '/dev/null'
    assert 'StrictHostKeyChecking=ask' in calls[-1]
    assert all(call[0] != 'ssh-keyscan' for call in calls)


def test_existing_key_and_missing_public_file_are_reused(setup_commands):
    result, _, home = setup_commands()
    assert result.returncode == 0
    key = home/'.ssh/sima_neat_pcie_ed25519'
    private = key.read_bytes()
    public = Path(str(key)+'.pub')
    expected = public.read_bytes()
    public.unlink()
    result, calls, _ = setup_commands(SSH_RESULT='0')
    assert result.returncode == 0
    assert key.read_bytes() == private
    assert public.read_bytes() == expected
    assert calls[-1][0] == 'ssh'


def test_failed_enrollment_returns_failure_without_scanning_or_erasing_trust(setup_commands, tmp_path):
    known = tmp_path/'known_hosts'
    known.write_text('existing trusted entry\n')
    result, calls, _ = setup_commands('--known-hosts', str(known), '--host-key-checking', 'ask', COPY_RESULT='1')
    assert result.returncode == 1
    assert known.read_text() == 'existing trusted entry\n'
    assert [call[0] for call in calls] == ['ssh', 'ssh-copy-id']


def test_dry_run_never_creates_credentials_or_contacts_targets(setup_commands):
    result, calls, home = setup_commands('--dry-run')
    assert result.returncode == 0
    assert not calls
    assert not (home/'.ssh').exists()


def test_unsupported_host_key_policy_is_rejected_before_mutation(setup_commands):
    result, calls, home = setup_commands('--host-key-checking', 'no')
    assert result.returncode == 2
    assert not calls
    assert not (home/'.ssh').exists()
