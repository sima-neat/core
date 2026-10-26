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
