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


def spec(tmp_path):
    pub = tmp_path/'pub'
    pub.mkdir()
    for name in ('runtime.deb', 'pyneat.whl'):
        (pub/name).write_text(name)
    metadata = {'version': '0.6.0+develop.'+'a'*12,
                'resources': ['runtime.deb', 'pyneat.whl'],
                'resources-checksum': {p.name: assemble.sha(p) for p in pub.iterdir()}}
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
    meta = json.loads((tmp_path/'metadata-offline.json').read_text())
    assert meta['resources'] == [assemble.CORE_ARCHIVE_NAME]
    assert meta['resources-checksum'][archive.name] == assemble.sha(archive)
    assert meta['installation']['script'] == ''
    (root/'install.sh').write_text('corrupted')
    with pytest.raises(ValueError, match='checksum'):
        assemble.verify_bundle(root, 'card')


def test_no_archive_or_metadata_when_offline_validation_fails(tmp_path):
    with patch.object(assemble, 'run'), patch.object(assemble, 'validate', side_effect=ValueError('missing wheel')):
        with pytest.raises(ValueError, match='missing wheel'):
            assemble.assemble(spec(tmp_path), tmp_path/'bundle')
    assert not (tmp_path/assemble.CORE_ARCHIVE_NAME).exists()
    assert not (tmp_path/'metadata-offline.json').exists()


def test_missing_core_publishes_none_of_the_ready_pcie_variants(tmp_path):
    args = SimpleNamespace(work=tmp_path/'publish', ref='develop', sha='a'*40,
                           bucket='bucket', kms='', distribution='')
    with patch.object(ci, 'source_metadata', return_value=({}, {}, {})), \
         patch.object(ci, 'existing', side_effect=[tmp_path/'existing.pending.json']*4+[None]), \
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
        assert key == baseline['key']
        path.write_text('baseline')
    def build(config, root):
        assert config['scope'] == 'core'
        assert set(config['targets']) == {'card'}
        (root.parent/'metadata-offline.json').write_text('{}')
        (root.parent/assemble.CORE_ARCHIVE_NAME).write_text('validated bundle')
    with patch.object(ci, 'source_metadata', return_value=(
             {'card': {'resources-checksum': {}}}, {'card': ('core-key', pub)}, {'card': 'c'*64})), \
         patch.object(ci, 'existing', return_value=None), \
         patch.object(ci, 'download', side_effect=download), \
         patch.object(ci, 'run', return_value=baseline['image_id']), \
         patch.object(ci.assemble, 'sha', return_value=baseline['sha256']), \
         patch.object(ci.assemble, 'verify_publication'), \
         patch.object(ci.assemble, 'assemble', side_effect=build), \
         patch.object(ci, 'put') as put:
        ci.build(args)
    assert [call.args[1] for call in put.call_args_list] == [
        'core/develop/'+'a'*12+'/'+assemble.CORE_ARCHIVE_NAME,
        'core/develop/'+'a'*12+'/metadata-offline.json.pending']
