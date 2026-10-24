"""Offline target adapter keeps APT/Python configuration private."""
import importlib.util
import json
import os
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parents[2]/'pcie_host/scripts/offline'
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('offline_deploy', HERE/'deploy.py')
deploy = importlib.util.module_from_spec(spec)
spec.loader.exec_module(deploy)


def test_private_apt_sources_and_hooks(tmp_path, monkeypatch):
    monkeypatch.setenv('APT_CONFIG', '/untrusted/config')
    deploy.configure(tmp_path/'repo', tmp_path/'state')
    config = Path(os.environ['APT_CONFIG']).read_text()
    assert 'Dir::Etc::main "/dev/null"' in config
    assert str(tmp_path/'state/empty') in config
    assert (tmp_path/'state/sources.list').read_text() == f'deb [trusted=yes] file:{tmp_path}/repo ./\n'
    assert 'https:' not in config


@pytest.mark.parametrize('role', ['host', 'card'])
def test_preflight_runs_admission_without_package_install(tmp_path, monkeypatch, role):
    calls = []
    (tmp_path/role/'payload').mkdir(parents=True)
    (tmp_path/role/'payload/test.deb').touch()
    monkeypatch.setattr(deploy, 'verify_bundle', lambda *args: None)
    monkeypatch.setattr(deploy, 'environment', lambda *args: None)
    monkeypatch.setattr(deploy, 'run', lambda args, **kwargs: calls.append(args))
    monkeypatch.setenv('PIP_EXTRA_INDEX_URL', 'https://untrusted.invalid')
    monkeypatch.setenv('NEAT_INSTALLER_SKIP_PLATFORM_CHECK', 'ON')
    with monkeypatch.context() as context:
        context.setattr(deploy.os, 'environ', os.environ.copy())
        deploy.execute(tmp_path, role, 'preflight')
        assert 'PIP_EXTRA_INDEX_URL' not in deploy.os.environ
        assert 'NEAT_INSTALLER_SKIP_PLATFORM_CHECK' not in deploy.os.environ
        assert deploy.os.environ['PIP_NO_INDEX'] == '1'
    assert not any('install' in c and '--simulate' not in c for c in calls)
    assert calls[-1][2:4] == [role, 'preflight']
    plan = next(c for c in calls if '--simulate' in c)
    assert ('--no-remove' in plan) == (role == 'host')
    assert ('--allow-downgrades' in plan) == (role == 'card')


def test_integrity_failure_precedes_environment_or_commands(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError('checksum')
    monkeypatch.setattr(deploy, 'verify_bundle', fail)
    monkeypatch.setattr(deploy, 'environment', lambda *args: pytest.fail('must not reach environment'))
    with pytest.raises(ValueError, match='checksum'):
        deploy.execute(tmp_path, 'host', 'install')


def test_card_adapter_retains_admission_and_lifecycle_without_online_sentinel(tmp_path):
    import subprocess
    payload = tmp_path/'payload'
    payload.mkdir()
    (payload/'install_neat_framework.sh').write_text('''
parse_args() { echo parse; }
collect_debs_in_install_order() { DEBS=(test.deb); echo collect; }
detect_env_mode() { echo modalix-board; }
ensure_platform_compatible() { echo platform; }
prepare_board_platform_admission() { echo admission; }
verify_board_platform_plan() { echo plan; }
verify_simulated_package_removals() { echo replacements; }
verify_scheduler_transaction() { echo transaction; }
log() { echo "$*"; }
install_board_sentinel() { echo ONLINE_SENTINEL; exit 77; }
install_for_environment() { install_board_sentinel; echo runtime_lifecycle; }
''')
    result = subprocess.run(['bash', str(HERE/'lifecycle.sh'), 'card', 'install', str(payload), '/tmp/plan'],
                            text=True, capture_output=True, check=True)
    assert 'ONLINE_SENTINEL' not in result.stdout
    assert result.stdout.splitlines()[:4] == ['parse', 'collect', 'platform', 'admission']
    assert 'plan\nreplacements\ntransaction\n' in result.stdout
    assert result.stdout.endswith('runtime_lifecycle\n')


def test_card_admission_failure_never_reaches_runtime_install(tmp_path):
    import subprocess
    payload = tmp_path/'payload'
    payload.mkdir()
    (payload/'install_neat_framework.sh').write_text('''
parse_args() { :; }
collect_debs_in_install_order() { DEBS=(test.deb); }
detect_env_mode() { echo modalix-board; }
ensure_platform_compatible() { return 42; }
prepare_board_platform_admission() { echo admission; }
install_for_environment() { echo MUTATED; }
''')
    result = subprocess.run(['bash', str(HERE/'lifecycle.sh'), 'card', 'install', str(payload), '/tmp/plan'],
                            text=True, capture_output=True)
    assert result.returncode == 42
    assert 'MUTATED' not in result.stdout


def test_card_replacement_rejection_precedes_install(tmp_path):
    import subprocess
    payload = tmp_path/'payload'
    payload.mkdir()
    (payload/'install_neat_framework.sh').write_text("""
parse_args() { :; }
collect_debs_in_install_order() { DEBS=(test.deb); }
detect_env_mode() { echo modalix-board; }
ensure_platform_compatible() { :; }
prepare_board_platform_admission() { :; }
verify_board_platform_plan() { :; }
verify_simulated_package_removals() { return 43; }
verify_scheduler_transaction() { echo TRANSACTION; }
install_for_environment() { echo MUTATED; }
""")
    result = subprocess.run(['bash', str(HERE/'lifecycle.sh'), 'card', 'install', str(payload), '/tmp/plan'],
                            text=True, capture_output=True)
    assert result.returncode == 43
    assert 'MUTATED' not in result.stdout
    assert 'TRANSACTION' not in result.stdout
