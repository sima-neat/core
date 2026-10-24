"""Admission and publication failure contracts for offline CI."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parents[2] / 'pcie_host/scripts/offline'

def load(name):
    spec = importlib.util.spec_from_file_location('offline_' + name, HERE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

with patch.dict(sys.modules, {'assemble': load('assemble')}):
    ci = load('ci')
resolver = load('resolve_run')
SHA = 'a' * 40
INPUTS = {'host': 'b' * 64, 'card': 'c' * 64}


def source(ref='develop'):
    return dict(repository={'full_name': 'sima-neat/core'},
                head_repository={'full_name': 'sima-neat/core'},
                path='.github/workflows/vulcan-ci.yml', event='push',
                status='completed', conclusion='success', head_sha=SHA,
                head_branch=ref, id=123)


def jobs():
    return [dict(name=n, conclusion='success') for n in (
        'Publish PCIe Host Package Metadata', 'Publish to Vulcan / Publish Vulcan artifacts')]


@pytest.mark.parametrize('ref', ['develop', 'main', 'release-0.6'])
def test_allowed_branch(ref):
    assert resolver.eligible(source(ref), jobs())['sha'] == SHA


def test_tag_requires_exact_commit():
    assert resolver.eligible(source('v0.6'), jobs()) is None
    assert resolver.eligible(source('v0.6'), jobs(), 'd' * 40) is None
    assert resolver.eligible(source('v0.6'), jobs(), SHA)


@pytest.mark.parametrize('field,value', [
    ('head_repository', {'full_name': 'fork/core'}), ('event', 'pull_request'),
    ('conclusion', 'failure'), ('status', 'in_progress'),
    ('path', '.github/workflows/other.yml'), ('head_branch', 'develop\nenabled=true'),
])
def test_untrusted_or_incomplete_run_rejected(field, value):
    run = source()
    run[field] = value
    assert resolver.eligible(run, jobs()) is None


def test_skipped_or_missing_publications_rejected():
    assert resolver.eligible(source(), []) is None
    rows = jobs()
    rows[0]['conclusion'] = 'skipped'
    assert resolver.eligible(source(), rows) is None


def metadata():
    return {'commit': SHA, 'branch': 'develop', 'resources': [ci.assemble.ARCHIVE_NAME],
            'resources-checksum': {ci.assemble.ARCHIVE_NAME: 'd' * 64},
            'artifact': {'distro': 'ubuntu22', 'arches': ['amd64'], 'type': 'offline',
                         'package_path': 'pciehost/ubuntu22/amd64'},
            'installation': {'script': ''}, 'source_metadata_sha256': INPUTS}


def test_offline_metadata_identity_and_source_drift():
    ci.offline_identity(metadata(), SHA, 'develop', 'ubuntu22', 'amd64', INPUTS)
    with pytest.raises(ValueError):
        ci.offline_identity(metadata(), SHA, 'develop', 'ubuntu22', 'amd64', {})
    data = metadata()
    data['installation']['script'] = 'install.sh'
    with pytest.raises(ValueError):
        ci.offline_identity(data, SHA, 'develop', 'ubuntu22', 'amd64', INPUTS)


@pytest.mark.parametrize('name', ['../payload', '/payload', 'a/b', 'bad\nname'])
def test_unsafe_resource_rejected(name):
    data = metadata()
    data['resources'] = [name]
    data['resources-checksum'] = {name: 'd' * 64}
    with pytest.raises(ValueError):
        ci.metadata_identity(data, SHA, 'develop', 'ubuntu22', 'amd64')


def test_tuple_and_ref_paths():
    assert ci.prefix('release/v1', SHA, 'ubuntu24', 'arm64') == 'core/release%2Fv1/aaaaaaaaaaaa/pciehost/ubuntu24/arm64'
    with pytest.raises(ValueError):
        ci.prefix('develop', SHA, 'ubuntu20', 'amd64')


def test_missing_last_tuple_publishes_nothing(tmp_path):
    args = SimpleNamespace(work=tmp_path/'publish', ref='develop', sha=SHA,
                           bucket='bucket', kms='', distribution='')
    with patch.object(ci, 'source_metadata', return_value=({}, {}, INPUTS)), \
         patch.object(ci, 'existing', side_effect=[tmp_path/'existing.pending.json'] * 3 + [None]), \
         patch.object(ci, 'put') as put:
        with pytest.raises(ValueError, match='missing validated'):
            ci.publish(args)
        put.assert_not_called()


def test_completed_publication_not_replaced(tmp_path):
    args = SimpleNamespace(work=tmp_path/'publish', ref='develop', sha=SHA,
                           bucket='bucket', kms='', distribution='')
    with patch.object(ci, 'source_metadata', return_value=({}, {}, INPUTS)), \
         patch.object(ci, 'existing', return_value=tmp_path/'existing.json'), \
         patch.object(ci, 'put') as put:
        ci.publish(args)
        put.assert_not_called()


def test_local_image_ids_are_immutable():
    assert ci.assemble.image('sha256:' + 'a' * 64)
    with pytest.raises(ValueError):
        ci.assemble.image('ubuntu:24.04')


def test_optional_metadata_does_not_hide_access_denied(tmp_path):
    result = SimpleNamespace(returncode=1, stderr='An error occurred (AccessDenied)')
    with patch.object(ci.subprocess, 'run', return_value=result):
        with pytest.raises(RuntimeError, match='AccessDenied'):
            ci.download('bucket', 'key', tmp_path/'meta', optional=True)
    result.stderr = 'An error occurred (NoSuchKey)'
    with patch.object(ci.subprocess, 'run', return_value=result):
        assert ci.download('bucket', 'key', tmp_path/'meta', optional=True) is False


def test_uploaded_checksum_required():
    with patch.object(ci, 'run', return_value='{}'):
        with pytest.raises(ValueError, match='checksum'):
            ci.verify_uploaded('bucket', 'key', 'a' * 64)


def test_final_metadata_write_is_conditional(tmp_path):
    path = tmp_path/'metadata.json'
    path.write_text('{}')
    with patch.object(ci, 'run') as run, patch.object(ci, 'verify_uploaded'):
        ci.put('bucket', 'key', path, 'kms-key', immutable=True)
        command = run.call_args.args
        assert command[command.index('--if-none-match') + 1] == '*'
        assert '--checksum-sha256' in command


def test_shared_card_publication_version_identity():
    data = metadata()
    del data['commit'], data['branch']
    data['version'] = '0.6.0+develop.' + SHA[:12]
    ci.metadata_identity(data, SHA, 'develop')
    data['version'] = '0.6.0+develop.' + 'b' * 12
    with pytest.raises(ValueError, match='selected CI'):
        ci.metadata_identity(data, SHA, 'develop')


def test_validation_branch_opt_in_does_not_admit_other_branches_or_forks():
    ref = 'codex/offline-pcie-discovery'
    assert resolver.eligible(source(ref), jobs()) is None
    assert resolver.eligible(source(ref), jobs(), validation_ref=ref)
    assert resolver.eligible(source('other'), jobs(), validation_ref=ref) is None
    run = source(ref)
    run['head_repository'] = {'full_name': 'fork/core'}
    assert resolver.eligible(run, jobs(), validation_ref=ref) is None


def test_validation_build_does_not_reuse_or_publish(tmp_path):
    import json
    host = tmp_path/'host'
    card = tmp_path/'card'
    host.mkdir()
    card.mkdir()
    for name in ('platform_provider.py', 'platform-source.json'):
        (host/name).write_text('fixture')
    baselines = json.loads((HERE/'baselines.json').read_text())
    baseline = baselines['card']
    args = SimpleNamespace(work=tmp_path/'work', ref='develop', sha=SHA,
                           bucket='bucket', distro='ubuntu24', arch='amd64',
                           installer_revision=SHA, run_id=123, validation_only=True)

    def command(*argv):
        if argv[:2] == ('docker', 'build'):
            Path(argv[argv.index('--iidfile') + 1]).write_text('sha256:' + 'a' * 64)
        return baseline['image_id'] if argv[:3] == ('docker', 'image', 'inspect') else ''

    def download(bucket, key, path):
        path.write_text('fixture')

    def assemble(spec, root):
        assert set(spec['targets']) == {'host', 'card'}
        (root.parent/ci.assemble.METADATA_NAME).write_text('{}')
        (root.parent/ci.assemble.ARCHIVE_NAME).write_text('validated bundle')

    records = {role: {'resources-checksum': {}} for role in ('host', 'card')}
    paths = {'host': ('host-key', host), 'card': ('card-key', card)}
    with patch.object(ci, 'source_metadata', return_value=(records, paths, INPUTS)), \
         patch.object(ci, 'existing') as existing, patch.object(ci, 'put') as put, \
         patch.object(ci, 'download', side_effect=download), \
         patch.object(ci, 'run', side_effect=command), \
         patch.object(ci.assemble, 'sha', side_effect=lambda p: baselines['platform_source']['sha256'] if p.name == 'platform-source.tar.gz' else baseline['sha256']), \
         patch.object(ci.assemble, 'verify_publication'), \
         patch.object(ci.assemble, 'assemble', side_effect=assemble) as build:
        ci.build(args)
        existing.assert_not_called()
        put.assert_not_called()
        build.assert_called_once()
    assert (args.work/ci.assemble.ARCHIVE_NAME).is_file()


@pytest.mark.parametrize('status', ['queued', 'in_progress', 'waiting', 'pending'])
def test_validation_can_use_ready_publications_while_other_tests_run(status):
    run = source('codex/offline-pcie-discovery')
    run.update(status=status, conclusion=None)
    assert resolver.eligible(run, jobs()) is None
    assert resolver.eligible(run, jobs(), validation_ref=run['head_branch'])
    assert resolver.eligible(run, [], validation_ref=run['head_branch']) is None
    run.update(status='completed', conclusion='failure')
    assert resolver.eligible(run, jobs(), validation_ref=run['head_branch']) is None
