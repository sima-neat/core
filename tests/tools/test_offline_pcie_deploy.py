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
    assert (tmp_path/'state/sources.list').read_text() == f'deb [trusted=yes] {(tmp_path/"repo").as_uri()} ./\n'
    assert 'Acquire::IndexTargets::deb::Packages::CompressionTypes "uncompressed";' in config
    assert 'https:' not in config


@pytest.mark.parametrize('sdk', [False, True])
def test_apt_reads_and_downloads_from_encoded_bundle_path(tmp_path, sdk):
    import subprocess
    repo = tmp_path/'offline space # café'/'repo'
    repo.mkdir(parents=True)
    package = tmp_path/'package'
    (package/'DEBIAN').mkdir(parents=True)
    (package/'DEBIAN/control').write_text(
        'Package: neat-offline-path-test\nVersion: 1.0\nArchitecture: all\n'
        'Maintainer: Test <test@example.com>\nDescription: Offline path fixture\n')
    deb = repo/'neat-offline-path-test_1.0_all.deb'
    subprocess.run(['dpkg-deb', '--build', str(package), str(deb)], check=True, capture_output=True)
    with (repo/'Packages').open('w') as stream:
        subprocess.run(['dpkg-scanpackages', '.', '/dev/null'], cwd=repo,
                       stdout=stream, stderr=subprocess.PIPE, check=True)
    destination = tmp_path/'download'
    destination.mkdir()
    # Exercise real APT parsing and fetching without changing installed packages.
    with deploy.private_apt(repo) as state:
        if sdk:
            from sdk_install import local_apt_methods
            local_apt_methods(state)
        subprocess.run(['apt-get', 'update'], check=True, capture_output=True, timeout=30)
        subprocess.run(['apt-get', 'download', 'neat-offline-path-test=1.0'], cwd=destination,
                       check=True, capture_output=True, timeout=30)
    assert (destination/deb.name).read_bytes() == deb.read_bytes()


def test_pip_resolves_dependencies_from_encoded_wheelhouse(tmp_path, monkeypatch):
    import subprocess
    from zipfile import ZipFile
    root = tmp_path/'offline space # café'
    wheels = root/'card/wheels'
    wheels.mkdir(parents=True)
    wheel = wheels/'neat_offline_path_test-1.0-py3-none-any.whl'
    dist = 'neat_offline_path_test-1.0.dist-info'
    with ZipFile(wheel, 'w') as archive:
        archive.writestr(dist+'/METADATA', 'Metadata-Version: 2.1\nName: neat-offline-path-test\nVersion: 1.0\n')
        archive.writestr(dist+'/WHEEL', 'Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
        archive.writestr(dist+'/RECORD', '')
    monkeypatch.setattr(deploy, 'verify_bundle', lambda *args: None)
    monkeypatch.setattr(deploy, 'environment', lambda *args: None)
    monkeypatch.setattr(deploy, 'missing_prerequisites', lambda: [])
    monkeypatch.setattr(deploy, 'check_space', lambda *args: None)
    monkeypatch.setattr(deploy, 'run', lambda *args, **kwargs: None)
    with monkeypatch.context() as context:
        context.setattr(deploy.os, 'environ', os.environ.copy())
        deploy.execute(root, 'card', 'preflight')
        env = dict(deploy.os.environ)
    destination = tmp_path/'download'
    # Resolve by name, as pip must do for dependencies of the payload wheels.
    subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--only-binary=:all:',
                    '--dest', str(destination), 'neat-offline-path-test==1.0'], env=env,
                   check=True, capture_output=True, timeout=30)
    assert (destination/wheel.name).read_bytes() == wheel.read_bytes()


@pytest.mark.parametrize('role', ['host', 'card'])
def test_preflight_runs_admission_without_package_install(tmp_path, monkeypatch, role):
    calls = []
    tmp_path = tmp_path/'offline space # café'
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
        assert deploy.os.environ['PIP_FIND_LINKS'] == (tmp_path/role/'wheels').as_uri()
        assert deploy.os.environ['LC_ALL'] == 'C'
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
    import pwd
    monkeypatch.setattr(deploy, 'python_location', lambda role: (pwd.getpwuid(os.getuid()), tmp_path/'venv'))
    monkeypatch.setattr(deploy, 'finish_python_location', lambda *args: None)
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


def test_space_budget_excludes_staged_archives_and_counts_expanded_wheels(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from zipfile import ZipFile
    mib = 1024**2
    (tmp_path/'repo').mkdir()
    (tmp_path/'repo/runtime.deb').write_bytes(b'archive')
    (tmp_path/'payload').mkdir()
    # Duplicates/staging are already allocated, not future root growth.
    (tmp_path/'payload/runtime.deb').write_bytes(b'duplicate' * mib)
    (tmp_path/'wheels').mkdir()
    with ZipFile(tmp_path/'wheels/runtime.whl', 'w') as wheel:
        wheel.writestr('module.py', b'x' * mib)
    monkeypatch.setattr(deploy, 'output', lambda args: '1024')
    required = 3 * (mib + mib) + 128 * mib
    monkeypatch.setattr(deploy.shutil, 'disk_usage', lambda path: SimpleNamespace(free=required))
    deploy.check_space(tmp_path)
    monkeypatch.setattr(deploy.shutil, 'disk_usage', lambda path: SimpleNamespace(free=required-1))
    with pytest.raises(ValueError, match='133 MiB available, 134 MiB required; free at least 1 MiB on /'):
        deploy.check_space(tmp_path)


def test_space_budget_counts_packages_without_installed_size(tmp_path, monkeypatch):
    import io
    import tarfile
    from types import SimpleNamespace
    (tmp_path/'repo').mkdir()
    (tmp_path/'repo/runtime.deb').touch()
    monkeypatch.setattr(deploy, 'output', lambda args: '')
    def contents(args, stdout):
        assert args[0:2] == ['dpkg-deb', '--fsys-tarfile']
        with tarfile.open(fileobj=stdout, mode='w') as archive:
            member = tarfile.TarInfo('./usr/lib/runtime.so')
            member.size = 4097
            archive.addfile(member, io.BytesIO(b'x' * member.size))
    monkeypatch.setattr(deploy, 'run', contents)
    required = 3 * 8192 + 128 * 1024**2
    monkeypatch.setattr(deploy.shutil, 'disk_usage', lambda path: SimpleNamespace(free=required-1))
    with pytest.raises(ValueError, match='Insufficient free root filesystem space'):
        deploy.check_space(tmp_path)
    monkeypatch.setattr(deploy.shutil, 'disk_usage', lambda path: SimpleNamespace(free=required))
    deploy.check_space(tmp_path)


def test_space_budget_rejects_unreadable_package_without_size(tmp_path, monkeypatch):
    import subprocess
    (tmp_path/'repo').mkdir()
    (tmp_path/'repo/runtime.deb').touch()
    monkeypatch.setattr(deploy, 'output', lambda args: '0')
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(2, args[0])
    monkeypatch.setattr(deploy, 'run', fail)
    with pytest.raises(subprocess.CalledProcessError):
        deploy.check_space(tmp_path)


def test_verification_does_not_reserve_installation_space(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(deploy, 'verify_bundle', lambda *args: None)
    monkeypatch.setattr(deploy, 'environment', lambda *args: None)
    monkeypatch.setattr(deploy, 'check_space', lambda *args: pytest.fail('verification must not reserve install space'))
    monkeypatch.setattr(deploy, 'run', lambda *args, **kwargs: None)
    monkeypatch.setattr(deploy, 'missing_prerequisites', lambda: [])
    monkeypatch.setattr(deploy, 'verify', lambda *args: calls.append(args))
    with monkeypatch.context() as context:
        context.setattr(deploy.os, 'environ', os.environ.copy())
        deploy.execute(tmp_path, 'card', 'verify')
    assert len(calls) == 1


def test_space_budget_uses_apt_plan_and_retains_explicit_reinstalls(tmp_path, monkeypatch):
    from types import SimpleNamespace
    mib = 1024**2
    (tmp_path/'repo').mkdir()
    for name in ('runtime', 'dependency'):
        (tmp_path/'repo'/f'{name}.deb').touch()
    plan = tmp_path/'plan.txt'
    plan.write_text('Inst runtime:arm64 [1] (1 offline [arm64])\nConf runtime (1 offline [arm64])\n')
    def metadata(args):
        return Path(args[2]).stem if args[-1] == 'Package' else '1024'
    monkeypatch.setattr(deploy, 'output', metadata)
    monkeypatch.setattr(deploy.shutil, 'disk_usage', lambda path: SimpleNamespace(free=131*mib))
    deploy.check_space(tmp_path, plan)
    # An uninstalled dependency is charged when the solver adds it to the plan.
    plan.write_text(plan.read_text()+'Inst dependency (1 offline [arm64])\n')
    with pytest.raises(ValueError, match='134 MiB required'):
        deploy.check_space(tmp_path, plan)


def test_space_budget_rejects_unaccounted_planned_package(tmp_path):
    plan = tmp_path/'plan.txt'
    plan.write_text('Inst missing-package (1 offline [arm64])\n')
    with pytest.raises(ValueError, match='Planned packages missing.*missing-package'):
        deploy.check_space(tmp_path, plan)


@pytest.mark.parametrize('role,nvme_exists,writable,expected', [
    ('host', True, True, 'home/pyneatpcie'),
    ('card', False, False, 'home/pyneat'),
    ('card', True, False, 'home/pyneat'),
    ('card', True, True, 'nvme/pyneat'),
])
def test_python_paths_match_online_defaults(tmp_path, monkeypatch, role, nvme_exists, writable, expected):
    from types import SimpleNamespace
    user = SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid(), pw_dir=str(tmp_path/'home'))
    monkeypatch.setenv('HOME', '/root')
    monkeypatch.setenv('SUDO_UID', str(user.pw_uid))
    monkeypatch.setattr(deploy.pwd, 'getpwuid', lambda uid: user)
    monkeypatch.setattr(deploy.os, 'access', lambda *args: writable)
    if nvme_exists:
        (tmp_path/'nvme').mkdir()
    assert deploy.python_location(role, tmp_path/'nvme') == (user, tmp_path/expected)


def test_nvme_activation_preserves_existing_home_environment(tmp_path, monkeypatch):
    from types import SimpleNamespace
    user = SimpleNamespace(pw_uid=os.getuid(), pw_gid=os.getgid(), pw_dir=str(tmp_path))
    old = tmp_path/'pyneat'
    old.mkdir()
    (old/'user-package').write_text('keep')
    path = tmp_path/'nvme/pyneat'
    path.mkdir(parents=True)
    monkeypatch.setattr(deploy.os, 'chown', lambda *args, **kwargs: None)
    deploy.finish_python_location(user, path, 'card')
    assert old.is_symlink() and old.resolve() == path
    backups = list(tmp_path.glob('pyneat-before-offline-*/pyneat/user-package'))
    assert len(backups) == 1 and backups[0].read_text() == 'keep'
    deploy.finish_python_location(user, path, 'card')
    assert len(list(tmp_path.glob('pyneat-before-offline-*'))) == 1


def test_sudo_selects_invoking_user_and_checks_nvme_as_that_user(tmp_path, monkeypatch):
    from types import SimpleNamespace
    user = SimpleNamespace(pw_uid=1234, pw_gid=2345, pw_name='installer', pw_dir=str(tmp_path/'home'))
    monkeypatch.setenv('SUDO_UID', '1234')
    monkeypatch.setenv('HOME', '/root')
    monkeypatch.setattr(deploy.os, 'getuid', lambda: 0)
    def account(uid):
        assert uid == 1234
        return user
    monkeypatch.setattr(deploy.pwd, 'getpwuid', account)
    monkeypatch.setattr(deploy.os, 'getgrouplist', lambda *args: [2345, 3456])
    nvme = tmp_path/'nvme'
    nvme.mkdir()
    def writable(args, **kwargs):
        assert args == ['test', '-w', str(nvme)]
        assert kwargs['user'] == 1234 and kwargs['group'] == 2345
        assert kwargs['extra_groups'] == [2345, 3456]
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(deploy.subprocess, 'run', writable)
    assert deploy.python_location('card', nvme)[1] == tmp_path/'home/pyneat'
    assert not (tmp_path/'home').exists()


def test_host_lifecycle_does_not_reopen_prepared_python_environment(tmp_path):
    import subprocess
    (tmp_path/'install_pciehost.sh').write_text('printf "%s\\n" "$@"\n')
    result = subprocess.run(['bash', str(HERE/'lifecycle.sh'), 'host', 'install', str(tmp_path), '/tmp/plan'],
                            text=True, capture_output=True, check=True)
    assert result.stdout.splitlines() == ['--skip-setup']


@pytest.mark.parametrize('symlink', [False, True])
@pytest.mark.parametrize('fails', [False, True])
def test_fresh_environment_preserves_previous_packages(tmp_path, monkeypatch, symlink, fails):
    path = tmp_path/'pyneatpcie'
    original = tmp_path/'original' if symlink else path
    original.mkdir()
    (original/'user-data').write_text('preserve me')
    if symlink:
        path.symlink_to(original.name, target_is_directory=True)
    def create(args):
        Path(args[-1]).mkdir()
        (Path(args[-1])/'new-data').touch()
    monkeypatch.setattr(deploy, 'run', create)
    def install():
        with deploy.fresh_python_environment(path):
            assert not (path/'user-data').exists()
            if fails:
                raise RuntimeError('install failed')
    if fails:
        with pytest.raises(RuntimeError, match='install failed'):
            install()
        assert (path/'user-data').read_text() == 'preserve me'
        assert path.is_symlink() == symlink
        assert not list(tmp_path.glob('pyneatpcie-before-offline-*'))
    else:
        install()
        backups = list(tmp_path.glob('pyneatpcie-before-offline-*'))
        assert len(backups) == 1
        assert (backups[0]/'user-data').read_text() == 'preserve me'
        assert backups[0].is_symlink() == symlink
        assert (path/'new-data').exists()


def test_existing_broken_pytest_does_not_contaminate_offline_environment(tmp_path):
    import json
    import subprocess
    import sys
    path = tmp_path/'pyneatpcie'
    subprocess.run([sys.executable, '-m', 'venv', str(path)], check=True)
    python = str(path/'bin/python')
    site = Path(subprocess.check_output([python, '-c',
        'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    metadata = site/'pytest-9.1.1.dist-info'
    metadata.mkdir()
    (metadata/'METADATA').write_text('Metadata-Version: 2.1\nName: pytest\nVersion: 9.1.1\n'
                                   'Requires-Dist: packaging\nRequires-Dist: pygments\n')
    broken = subprocess.run([python, '-m', 'pip', 'check'], capture_output=True, text=True)
    assert broken.returncode == 1
    assert 'packaging' in broken.stdout and 'pygments' in broken.stdout
    with deploy.fresh_python_environment(path):
        subprocess.run([python, '-m', 'pip', 'check'], check=True, capture_output=True)
        names = json.loads(subprocess.check_output([python, '-m', 'pip', 'list', '--format=json'], text=True))
        assert 'pytest' not in {row['name'] for row in names}
    assert list(tmp_path.glob('pyneatpcie-before-offline-*'))
