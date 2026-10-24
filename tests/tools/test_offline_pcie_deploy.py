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
    assert 'Acquire::IndexTargets::deb::Packages::CompressionTypes "uncompressed";' in config
    assert 'https:' not in config


@pytest.mark.parametrize('role', ['host', 'card'])
def test_preflight_runs_admission_without_package_install(tmp_path, monkeypatch, role):
    calls = []
    (tmp_path/role/'payload').mkdir(parents=True)
    (tmp_path/role/'payload/test.deb').touch()
    monkeypatch.setattr(deploy, 'verify_bundle', lambda *args: None)
    monkeypatch.setattr(deploy, 'environment', lambda *args: None)
    monkeypatch.setattr(deploy, 'missing_prerequisites', lambda: ['python3-venv'])
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
    assert '--allow-downgrades' in plan
    assert ('--reinstall' in plan) == (role == 'card')


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
install_python_environment() { echo SYSTEM_PYTHON; exit 78; }
install_for_environment() { install_board_sentinel; install_python_environment; echo runtime_lifecycle; }
''')
    result = subprocess.run(['bash', str(HERE/'lifecycle.sh'), 'card', 'install', str(payload), '/tmp/plan'],
                            text=True, capture_output=True, check=True)
    assert 'ONLINE_SENTINEL' not in result.stdout
    assert 'SYSTEM_PYTHON' not in result.stdout
    assert 'retaining prepared isolated Python environment' in result.stdout
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


def test_prerequisites_preserve_installed_system_packages(monkeypatch):
    import subprocess
    states = {'python3-venv': (0, ''), 'python3-pip': (1, ''),
              'gstreamer1.0-tools': (0, 'installed'), 'ca-certificates': (0, 'installed')}
    def query(args, **kwargs):
        status, stdout = states[args[-1]]
        return subprocess.CompletedProcess(args, status, stdout=stdout)
    monkeypatch.setattr(deploy.subprocess, 'run', query)
    assert deploy.missing_prerequisites() == ['python3-venv', 'python3-pip']


def test_install_refreshes_plan_after_prerequisites(tmp_path, monkeypatch):
    calls = []
    (tmp_path/'card/payload').mkdir(parents=True)
    (tmp_path/'card/payload/test.deb').touch()
    monkeypatch.setattr(deploy, 'verify_bundle', lambda *args: None)
    monkeypatch.setattr(deploy, 'environment', lambda *args: None)
    monkeypatch.setattr(deploy, 'missing_prerequisites', lambda: ['python3-venv'])
    monkeypatch.setattr(deploy, 'verify', lambda *args: None)
    monkeypatch.setattr(deploy, 'run', lambda args, **kwargs: calls.append(list(args)))
    with monkeypatch.context() as context:
        context.setattr(deploy.os, 'environ', os.environ.copy())
        deploy.execute(tmp_path, 'card', 'install')
    simulations = [i for i, c in enumerate(calls) if '--simulate' in c]
    prerequisite = next(i for i, c in enumerate(calls)
                        if c[0] == 'apt-get' and 'install' in c and '--simulate' not in c)
    assert len(simulations) == 2
    assert simulations[0] < prerequisite < simulations[1]
    assert calls[simulations[1]+1][2:4] == ['card', 'install']
    assert 'python3-venv' in calls[simulations[0]]
    assert 'python3-venv' not in calls[simulations[1]]
    assert '--reinstall' in calls[simulations[1]]


def test_generated_bootstrap_reads_only_uncompressed_package_index(tmp_path):
    spec = importlib.util.spec_from_file_location('offline_target', HERE/'target.py')
    target = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(target)
    target.write_bootstrap(tmp_path)
    script = (tmp_path/'bootstrap.sh').read_text()
    assert 'Acquire::IndexTargets::deb::Packages::CompressionTypes "uncompressed";' in script
    assert 'apt-get update' in script
    assert 'Dir::Etc::parts "$root/apt/empty"' in script


@pytest.mark.parametrize('fails', [False, True])
@pytest.mark.parametrize('previous', [None, '/etc/apt/custom.conf'])
def test_private_apt_restores_environment_before_cleanup(tmp_path, monkeypatch, fails, previous):
    if previous is None:
        monkeypatch.delenv('APT_CONFIG', raising=False)
    else:
        monkeypatch.setenv('APT_CONFIG', previous)
    config = None
    try:
        with deploy.private_apt(tmp_path/'repo'):
            config = Path(os.environ['APT_CONFIG'])
            assert config.is_file()
            if fails:
                raise RuntimeError('simulated APT failure')
    except RuntimeError:
        assert fails
    assert os.environ.get('APT_CONFIG') == previous
    assert not config.exists()
