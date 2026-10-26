"""Standalone bundles retain integrity, isolation, and aggregate publication gates."""
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parents[2]/'pcie_host/scripts/offline'
sys.path.insert(0, str(HERE))
import assemble
import ci
import core_install
import paired_devkit
import sdk_install


def spec(tmp_path):
    pub = tmp_path/'pub'
    pub.mkdir()
    for name in ('runtime.deb', 'pyneat.whl'):
        (pub/name).write_text(name)
    metadata = {'version': '0.6.0+develop.'+'a'*12,
                'resources': ['runtime.deb', 'pyneat.whl'],
                'resources-checksum': {p.name: assemble.sha(p) for p in pub.iterdir()}}
    extra = pub/'sima-neat-1-Linux-extras.tar.gz'
    import io
    with tarfile.open(extra, 'w:gz') as package:
        entry = tarfile.TarInfo('share/tutorials/README.md')
        entry.size = len(b'tutorials and tests')
        package.addfile(entry, io.BytesIO(b'tutorials and tests'))
    metadata['selectable-resources'] = [{'resource': extra.name, 'checksum': assemble.sha(extra)}]
    (pub/'metadata.json').write_text(json.dumps(metadata))
    return {'scope': 'core', 'targets': {'card': {
        'publication': str(pub), 'metadata_sha256': assemble.sha(pub/'metadata.json'),
        'core_revision': 'a'*12, 'image': 'sha256:'+'b'*64,
        'architecture': 'arm64', 'os_id': 'elxr', 'os_version': '26.04'}}}


def test_core_archive_is_self_contained_and_validated_offline(tmp_path):
    root = tmp_path/'bundle'
    with patch.object(assemble, 'run') as run:
        assemble.assemble(spec(tmp_path), root)
    assert run.call_count == 2
    assert '--network=none' in run.call_args.args
    assert not (root/'host').exists()
    assert not (root/'discover.py').exists()
    assert assemble.verify_bundle(root, 'card')['scope'] == 'core'
    archive = tmp_path/assemble.CORE_ARCHIVE_NAME
    with tarfile.open(archive) as tar:
        assert tar.getmember('neat-core-offline/install.sh').mode == 0o755
        assert tar.getmember('neat-core-offline/process.py').isfile()
        assert tar.getmember('neat-core-offline/sdk_install.py').isfile()
        assert tar.getmember('neat-core-offline/paired_devkit.py').isfile()
    meta = json.loads((tmp_path/'metadata.json').read_text())
    assert meta['resources'] == [assemble.CORE_ARCHIVE_NAME, 'sima-neat-1-Linux-extras.tar.gz']
    minimal = json.loads((tmp_path/'metadata-minimal.json').read_text())
    assert minimal['resources'] == [assemble.CORE_ARCHIVE_NAME]
    assert not (root/'card/payload/sima-neat-1-Linux-extras.tar.gz').exists()
    assert meta['artifact']['package_path'] == 'offline'
    assert meta['resources-checksum'][archive.name] == assemble.sha(archive)
    assert meta['installation']['script'] == ''
    assert meta['platforms'] == []
    assert set(meta['size']) == set(minimal['size']) == {'download', 'install'}
    (root/'install.sh').write_text('corrupted')
    with pytest.raises(ValueError, match='checksum'):
        assemble.verify_bundle(root, 'card')


def test_no_archive_or_metadata_when_offline_validation_fails(tmp_path):
    with patch.object(assemble, 'run'), patch.object(assemble, 'validate', side_effect=ValueError('missing wheel')):
        with pytest.raises(ValueError, match='missing wheel'):
            assemble.assemble(spec(tmp_path), tmp_path/'bundle')
    assert not (tmp_path/assemble.CORE_ARCHIVE_NAME).exists()
    assert not (tmp_path/'metadata.json').exists()
    assert not (tmp_path/'metadata-minimal.json').exists()


def test_missing_core_publishes_none_of_the_ready_pcie_variants(tmp_path):
    args = SimpleNamespace(work=tmp_path/'publish', ref='develop', sha='a'*40,
                           bucket='bucket', kms='', distribution='')
    with patch.object(ci, 'source_metadata', return_value=({}, {}, {})), \
         patch.object(ci, 'existing', side_effect=[{'metadata-offline.json': tmp_path/'metadata-offline.json.pending'}]*4+[None]), \
         patch.object(ci, 'put') as put:
        with pytest.raises(ValueError, match='standalone Core'):
            ci.publish(args)
        put.assert_not_called()


def test_core_source_acquisition_never_downloads_host_metadata(tmp_path):
    args = SimpleNamespace(core=True, ref='develop', sha='a'*40, arch=None, bucket='bucket')
    data = {'version': '0.6.0+develop.'+'a'*12, 'resources': ['runtime.deb'],
            'resources-checksum': {'runtime.deb': 'b'*64}}
    def download(bucket, key, path):
        assert key == 'core/develop/'+'a'*12+'/metadata.json'
        path.write_text(json.dumps(data))
    with patch.object(ci, 'download', side_effect=download):
        records, paths, hashes = ci.source_metadata(args, tmp_path)
    assert set(records) == set(paths) == set(hashes) == {'card'}


def test_installer_rejects_corruption_before_namespace_or_lock(tmp_path):
    with patch.object(core_install.os, 'geteuid', return_value=0), \
         patch.object(core_install, 'verify_bundle', side_effect=ValueError('checksum')), \
         patch.object(core_install, 'run_host') as run:
        with pytest.raises(ValueError, match='checksum'):
            core_install.install(tmp_path, 'install')
        run.assert_not_called()


def test_installer_uses_shared_lock_and_isolated_supervised_process(tmp_path):
    from unittest.mock import mock_open
    with patch.object(core_install.os, 'geteuid', return_value=0), \
         patch.object(core_install, 'verify_bundle', return_value={'scope': 'core'}), \
         patch.object(core_install, 'environment', return_value='devkit'), \
         patch.object(core_install.shutil, 'which', return_value='/bin/tool'), \
         patch('builtins.open', mock_open()) as opened, \
         patch.object(core_install.fcntl, 'flock') as lock, \
         patch.object(core_install, 'run_host') as run:
        core_install.install(tmp_path, 'preflight')
    opened.assert_called_once_with('/run/neat-offline-install.lock', 'w')
    lock.assert_called_once()
    argv = run.call_args.args[0]
    assert argv[-2:] == ['card', 'preflight']
    assert argv[argv.index('unshare')+1] == '--net'
    assert '--foreground' in argv
    assert run.call_args.kwargs['stdin'] == core_install.subprocess.DEVNULL


def environment_fixture(tmp_path, identity):
    root, system = tmp_path/'bundle', tmp_path/'system'
    (root/'card/payload').mkdir(parents=True)
    (root/'card/payload/manifest.json').write_text(json.dumps({'platform-version': '3.0.0'}))
    (system/'etc').mkdir(parents=True)
    if identity == 'devkit':
        (system/'etc/buildinfo').write_text('MACHINE = modalix\nDISTRO_VERSION = 3.0.0\n')
    elif identity == 'sdk':
        (system/'etc/sdk-release').write_text('SDK Version = 3.0.0_Palette_SDK_neat_develop\nPlatform Base = 3.0.0\n')
        (system/'opt/toolchain/aarch64/modalix/usr').mkdir(parents=True)
    return root, system


@pytest.mark.parametrize('identity', ['devkit', 'sdk'])
def test_offline_environment_uses_bundled_manifest_without_download_metadata(tmp_path, identity):
    root, system = environment_fixture(tmp_path, identity)
    with patch.object(core_install.platform, 'machine', return_value='aarch64'), \
         patch.object(core_install.platform, 'system', return_value='Linux'):
        assert core_install.environment(root, system) == identity
    assert not (root/'metadata.json').exists()


@pytest.mark.parametrize('identity', ['devkit', 'sdk'])
def test_offline_environment_rejects_platform_mismatch(tmp_path, identity):
    root, system = environment_fixture(tmp_path, identity)
    (root/'card/payload/manifest.json').write_text(json.dumps({'platform-version': '4.0.0'}))
    with patch.object(core_install.platform, 'machine', return_value='aarch64'), \
         patch.object(core_install.platform, 'system', return_value='Linux'):
        with pytest.raises(ValueError, match='Incompatible.*Platform version'):
            core_install.environment(root, system)


@pytest.mark.parametrize('identity', ['missing', 'other_board', 'unknown_sdk', 'missing_sysroot', 'wrong_arch'])
def test_offline_environment_rejects_unsupported_identity(tmp_path, identity):
    root, system = environment_fixture(tmp_path, 'sdk' if identity == 'missing_sysroot' else 'missing')
    if identity == 'other_board':
        (system/'etc/buildinfo').write_text('MACHINE = other\nDISTRO_VERSION = 3.0.0\n')
    elif identity == 'unknown_sdk':
        (system/'etc/sdk-release').write_text('SDK Type = other-sdk\nSDK Version = 3.0.0\n')
    elif identity == 'missing_sysroot':
        (system/'opt/toolchain/aarch64/modalix/usr').rmdir()
    elif identity == 'wrong_arch':
        (system/'etc/buildinfo').write_text('MACHINE = modalix\nDISTRO_VERSION = 3.0.0\n')
    with patch.object(core_install.platform, 'machine', return_value='x86_64'), \
         patch.object(core_install.platform, 'system', return_value='Linux'):
        with pytest.raises(ValueError):
            core_install.environment(root, system)


@pytest.mark.parametrize('phase', ['preflight', 'install', 'verify'])
def test_environment_rejection_precedes_lock_namespace_and_mutation(tmp_path, phase):
    with patch.object(core_install.os, 'geteuid', return_value=0), \
         patch.object(core_install, 'verify_bundle', return_value={'scope': 'core'}), \
         patch.object(core_install, 'environment', side_effect=ValueError('Unsupported installation environment')), \
         patch.object(core_install, 'run_host') as run, patch('builtins.open') as opened:
        with pytest.raises(ValueError, match='Unsupported installation environment'):
            core_install.install(tmp_path, phase)
        run.assert_not_called()
        opened.assert_not_called()


@pytest.mark.parametrize('phase', ['preflight', 'install', 'verify'])
def test_sdk_routes_to_sysroot_and_paired_devkit_after_preflight(tmp_path, phase):
    from unittest.mock import mock_open, MagicMock
    events = []
    device = MagicMock()
    device.phase.side_effect = lambda value: events.append(('devkit', value))
    def local(args, **kwargs):
        assert args[-2] == str(tmp_path/'sdk_install.py')
        assert 'unshare' not in args
        events.append(('sdk', args[-1]))
    with patch.object(core_install.os, 'geteuid', return_value=0), \
         patch.object(core_install, 'verify_bundle', return_value={'scope': 'core'}), \
         patch.object(core_install, 'environment', return_value='sdk'), \
         patch.object(core_install.shutil, 'which', return_value='/bin/tool'), \
         patch('builtins.open', mock_open()), patch.object(core_install.fcntl, 'flock'), \
         patch.object(core_install, 'paired_devkit') as pairing, \
         patch.object(core_install, 'run_host', side_effect=local):
        pairing.return_value.__enter__.return_value = device
        core_install.install(tmp_path, phase)
    expected = [('sdk', 'preflight'), ('devkit', 'preflight')] if phase == 'install' else []
    assert events == expected+[('sdk', phase), ('devkit', phase)]


def test_paired_preflight_failure_prevents_sdk_installation(tmp_path):
    from unittest.mock import mock_open, MagicMock
    device = MagicMock()
    device.phase.side_effect = ValueError('incompatible paired DevKit')
    with patch.object(core_install.os, 'geteuid', return_value=0), \
         patch.object(core_install, 'verify_bundle', return_value={'scope': 'core'}), \
         patch.object(core_install, 'environment', return_value='sdk'), \
         patch.object(core_install.shutil, 'which', return_value='/bin/tool'), \
         patch('builtins.open', mock_open()), patch.object(core_install.fcntl, 'flock'), \
         patch.object(core_install, 'paired_devkit') as pairing, \
         patch.object(core_install, 'run_host') as run:
        pairing.return_value.__enter__.return_value = device
        with pytest.raises(ValueError, match='paired DevKit'):
            core_install.install(tmp_path, 'install')
    assert [call.args[0][-1] for call in run.call_args_list] == ['preflight']


def test_paired_target_cannot_be_another_sdk(tmp_path):
    with patch.object(core_install.os, 'geteuid', return_value=0), \
         patch.object(core_install, 'verify_bundle', return_value={'scope': 'core'}), \
         patch.object(core_install, 'environment', return_value='sdk'), \
         patch.object(core_install, 'run_host') as run:
        with pytest.raises(ValueError, match='Paired target'):
            core_install.install(tmp_path, 'install', devkit_only=True)
        run.assert_not_called()


def test_generic_host_cannot_use_environment_overrides_as_identity(tmp_path, monkeypatch):
    root, system = environment_fixture(tmp_path, 'missing')
    for key in ('SYSROOT', 'ELXR_SDK_RELEASE_FILE', 'NEAT_BUILDINFO_FILE', 'FORCE_INSTALL', 'NEAT_INSTALLER_SKIP_PLATFORM_CHECK'):
        monkeypatch.setenv(key, str(tmp_path))
    with patch.object(core_install.platform, 'system', return_value='Linux'):
        with pytest.raises(ValueError, match='Unsupported installation environment'):
            core_install.environment(root, system)


def test_paired_deployment_requires_explicit_sdk_pairing(tmp_path, monkeypatch):
    monkeypatch.delenv('DEVKIT_SYNC_DEVKIT_IP', raising=False)
    with patch.object(paired_devkit, 'PairedDevKit') as device:
        with paired_devkit.paired_devkit(tmp_path) as result:
            assert result is None
        device.assert_not_called()


def test_paired_deployment_sends_password_only_on_stdin(monkeypatch):
    monkeypatch.setenv('SUDO_PASSWORD', 'test-secret')
    with patch.object(paired_devkit.os, 'getuid', return_value=0), \
         patch.object(paired_devkit.shutil, 'which', return_value='/usr/bin/ssh'), \
         patch.object(paired_devkit, 'run') as run:
        device = paired_devkit.PairedDevKit('192.0.2.2', 'sima')
        device.directory = '/var/tmp/neat-core-offline-ABCDEFGH'
        device.phase('install')
    assert '--devkit-only' in run.call_args.args[0][-1]
    assert 'test-secret' not in str(run.call_args.args)
    assert run.call_args.kwargs['input'] == 'test-secret\n'


@pytest.mark.parametrize('fails', [False, True])
def test_paired_transfer_uses_same_bundle_and_retains_unknown_installation(tmp_path, monkeypatch, fails, capsys):
    monkeypatch.setenv('DEVKIT_SYNC_DEVKIT_IP', '192.0.2.2')
    monkeypatch.delenv('NEAT_INSTALLER_SKIP_DEVKIT_SYNC', raising=False)
    (tmp_path/'install.py').write_text('offline entry point')
    commands = []
    def ssh(device, command, **kwargs):
        commands.append(command)
        if 'mktemp' in command:
            return SimpleNamespace(stdout='/var/tmp/neat-core-offline-ABCDEFGH\n')
        if 'tar --no-same-owner' in command:
            with tarfile.open(fileobj=kwargs['stdin']) as archive:
                assert archive.extractfile('neat-core-offline/install.py').read() == b'offline entry point'
        if 'sudo' in command and fails:
            raise core_install.subprocess.CalledProcessError(255, ['ssh'])
    with patch.object(paired_devkit.PairedDevKit, 'ssh', ssh), \
         patch.object(paired_devkit.shutil, 'which', return_value='/usr/bin/ssh'):
        if fails:
            with pytest.raises(core_install.subprocess.CalledProcessError):
                with paired_devkit.paired_devkit(tmp_path) as device:
                    device.phase('install')
        else:
            with paired_devkit.paired_devkit(tmp_path) as device:
                device.phase('install')
    assert any(command.startswith('rm -rf') for command in commands) == (not fails)
    assert ('outcome unknown' in capsys.readouterr().out) == fails


def test_sdk_uses_private_arm64_repository_and_disables_online_paired_sync(tmp_path, monkeypatch):
    (tmp_path/'card/payload').mkdir(parents=True)
    (tmp_path/'card/payload/runtime.deb').touch()
    monkeypatch.setenv('FORCE_INSTALL', '1')
    monkeypatch.setenv('NEAT_INSTALLER_SKIP_PLATFORM_CHECK', 'ON')
    monkeypatch.setenv('DEVKIT_SYNC_DEVKIT_IP', '192.0.2.2')
    calls = []
    with patch.dict(sys.modules, {'install': core_install}), \
         patch.object(core_install, 'environment', return_value='sdk'), \
         patch.object(sdk_install, 'verify_bundle', return_value={'scope': 'core'}), \
         patch.object(sdk_install, 'output', return_value='arm64'), \
         patch.object(sdk_install, 'local_apt_methods'), \
         patch.object(sdk_install, 'run', side_effect=lambda args, **kwargs: calls.append((args, kwargs))):
        sdk_install.execute(tmp_path, 'preflight')
    assert len(calls) == 2
    env = calls[-1][1]['env']
    assert env['SYSROOT'] == '/opt/toolchain/aarch64/modalix'
    assert env['NEAT_INSTALLER_SKIP_DEVKIT_SYNC'] == 'ON'
    assert 'FORCE_INSTALL' not in env
    assert 'NEAT_INSTALLER_SKIP_PLATFORM_CHECK' not in env
    assert 'DEVKIT_SYNC_DEVKIT_IP' not in env
    assert 'ensure_platform_compatible' in calls[-1][0][2]


def test_sdk_does_not_require_namespace_capabilities(tmp_path):
    from unittest.mock import mock_open
    from contextlib import nullcontext
    with patch.object(core_install.os, 'geteuid', return_value=0), \
         patch.object(core_install, 'verify_bundle', return_value={'scope': 'core'}), \
         patch.object(core_install, 'environment', return_value='sdk'), \
         patch.object(core_install.shutil, 'which', side_effect=lambda name: None if name == 'unshare' else '/bin/'+name), \
         patch('builtins.open', mock_open()), patch.object(core_install.fcntl, 'flock'), \
         patch.object(core_install, 'paired_devkit', return_value=nullcontext(None)), \
         patch.object(core_install, 'run_host') as run:
        core_install.install(tmp_path, 'preflight')
    assert 'unshare' not in run.call_args.args[0]
    assert str(tmp_path/'sdk_install.py') in run.call_args.args[0]


def test_sdk_apt_exposes_only_local_transport_methods(tmp_path):
    source, state = tmp_path/'system-methods', tmp_path/'private-apt'
    source.mkdir(); state.mkdir()
    for name in ('file', 'copy', 'store', 'http', 'https', 'mirror', 'ssh'):
        (source/name).write_text('method')
    sdk_install.local_apt_methods(state, source)
    assert {path.name for path in (state/'methods').iterdir()} == {'file', 'copy', 'store'}
    assert all(path.is_symlink() for path in (state/'methods').iterdir())
    assert str(state/'methods')+'/' in (state/'config').read_text()


def test_sdk_fails_closed_if_local_apt_methods_are_missing(tmp_path):
    source, state = tmp_path/'system-methods', tmp_path/'private-apt'
    source.mkdir(); state.mkdir()
    with pytest.raises(ValueError, match='SDK prerequisite missing'):
        sdk_install.local_apt_methods(state, source)
    assert not (state/'config').exists()


def test_sdk_apt_blocks_http_even_with_an_online_source(tmp_path):
    import http.server
    import os
    import subprocess
    import threading
    from deploy import private_apt
    requests = []
    class Canary(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_error(404)
        def log_message(self, *args):
            pass
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Canary)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with private_apt(tmp_path/'repo') as state:
            (state/'sources.list').write_text(f'deb [trusted=yes] http://127.0.0.1:{server.server_port}/ ./\n')
            env = {**os.environ, 'LC_ALL': 'C'}
            # Positive control: the regular APT HTTP method can reach the server.
            subprocess.run(['apt-get', 'update'], env=env, capture_output=True, timeout=30)
            assert requests
            requests.clear()
            sdk_install.local_apt_methods(state)
            result = subprocess.run(['apt-get', 'update'], env=env, capture_output=True, text=True, timeout=30)
            assert result.returncode != 0
            assert '/methods/http' in result.stderr
            assert not requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


@pytest.mark.parametrize('field,value', [('os_id', 'ubuntu'), ('os_version', '24.04'), ('architecture', 'amd64')])
def test_unsupported_core_tuple_rejected_before_collection(tmp_path, field, value):
    config = spec(tmp_path)
    config['targets']['card'][field] = value
    with patch.object(assemble, 'run') as run:
        with pytest.raises(ValueError, match='eLxr'):
            assemble.assemble(config, tmp_path/'bundle')
    run.assert_not_called()


def test_core_build_uses_only_card_baseline_and_stages_root_variant(tmp_path):
    args = SimpleNamespace(core=True, work=tmp_path/'work', ref='develop', sha='a'*40,
                           bucket='bucket', installer_revision='a'*40, run_id=123,
                           validation_only=False, kms='', arch=None)
    baseline = json.loads((HERE/'baselines.json').read_text())['card']
    pub = tmp_path/'pub'
    pub.mkdir()
    def download(bucket, key, path):
        assert key in (baseline['key'], 'core-key/sima-neat-1-Linux-extras.tar.gz')
        path.write_text('baseline')
    def build(config, root):
        assert config['scope'] == 'core'
        assert set(config['targets']) == {'card'}
        for filename, resources in (('metadata.json', [assemble.CORE_ARCHIVE_NAME, 'sima-neat-1-Linux-extras.tar.gz']),
                                    ('metadata-minimal.json', [assemble.CORE_ARCHIVE_NAME])):
            (root.parent/filename).write_text(json.dumps({'resources': resources}))
        (root.parent/assemble.CORE_ARCHIVE_NAME).write_text('validated bundle')
    with patch.object(ci, 'source_metadata', return_value=(
             {'card': {'resources-checksum': {}, 'selectable-resources': [{'resource': 'sima-neat-1-Linux-extras.tar.gz', 'checksum': baseline['sha256']}]}}, {'card': ('core-key', pub)}, {'card': 'c'*64})), \
         patch.object(ci, 'existing', return_value=None), \
         patch.object(ci, 'download', side_effect=download), \
         patch.object(ci, 'run', return_value=baseline['image_id']), \
         patch.object(ci.assemble, 'sha', return_value=baseline['sha256']), \
         patch.object(ci.assemble, 'verify_publication'), \
         patch.object(ci.assemble, 'assemble', side_effect=build), \
         patch.object(ci, 'put') as put:
        ci.build(args)
    assert [call.args[1] for call in put.call_args_list] == [
        'core/develop/'+'a'*12+'/offline/'+assemble.CORE_ARCHIVE_NAME,
        'core/develop/'+'a'*12+'/offline/sima-neat-1-Linux-extras.tar.gz',
        'core/develop/'+'a'*12+'/offline/metadata.json.pending',
        'core/develop/'+'a'*12+'/offline/metadata-minimal.json.pending']


def core_metadata(minimal=False, archive_hash='d'*64):
    name = assemble.CORE_ARCHIVE_NAME
    hashes = {name: archive_hash}
    if not minimal:
        hashes['sima-neat-1-Linux-extras.tar.gz'] = 'e'*64
    return {'version': '0.6.0+develop.'+'a'*12, 'resources': list(hashes),
            'resources-checksum': hashes, 'installation': {'script': ''},
            'artifact': {'type': 'offline', 'repository': 'core', 'package_path': 'offline',
                         'distro': 'elxr26', 'arches': ['arm64']},
            'source_metadata_sha256': {'card': 'c'*64}}


@pytest.mark.parametrize('mutation', ['valid', 'missing_minimal', 'different_runtime', 'extras_in_minimal'])
def test_reuse_requires_complete_consistent_core_profiles(tmp_path, mutation):
    args = SimpleNamespace(core=True, ref='develop', sha='a'*40, bucket='bucket')
    def download(bucket, key, path, optional=False):
        assert '/offline/' in key
        if key.endswith('.pending'):
            return False
        minimal = key.endswith('/metadata-minimal.json')
        if minimal and mutation == 'missing_minimal':
            return False
        data = core_metadata(minimal and mutation != 'extras_in_minimal',
                             'f'*64 if minimal and mutation == 'different_runtime' else 'd'*64)
        path.write_text(json.dumps(data))
        return True
    with patch.object(ci, 'download', side_effect=download), patch.object(ci, 'verify_uploaded'):
        if mutation == 'valid':
            assert set(ci.existing(args, tmp_path, {'card': 'c'*64})) == {'metadata.json', 'metadata-minimal.json'}
        else:
            with pytest.raises(ValueError):
                ci.existing(args, tmp_path, {'card': 'c'*64})


def test_all_profiles_verified_before_any_metadata_is_published(tmp_path):
    args = SimpleNamespace(work=tmp_path/'publish', ref='codex/offline', sha='a'*40,
                           bucket='bucket', kms='', distribution='cdn')
    def existing(args, work, inputs):
        return {name: work/(name+'.pending') for name in ci.metadata_names(args)}
    with patch.object(ci, 'source_metadata', return_value=({}, {}, {})), \
         patch.object(ci, 'existing', side_effect=existing), patch.object(ci, 'put') as put, \
         patch.object(ci, 'run') as run:
        ci.publish(args)
    assert put.call_count == 6
    keys = {call.args[1] for call in put.call_args_list}
    base = 'core/codex%2Foffline/'+'a'*12+'/offline/'
    assert base+'metadata.json' in keys
    assert base+'metadata-minimal.json' in keys
    assert all(call.kwargs['immutable'] for call in put.call_args_list)
    assert '/'+base.replace('%2F','%252F')+'metadata-minimal.json' in run.call_args.args


def test_extras_checksum_failure_prevents_collection(tmp_path):
    config = spec(tmp_path)
    (Path(config['targets']['card']['publication'])/'sima-neat-1-Linux-extras.tar.gz').write_text('tampered')
    with patch.object(assemble, 'run') as run:
        with pytest.raises(ValueError, match='extras archive checksum'):
            assemble.assemble(config, tmp_path/'bundle')
        run.assert_not_called()
