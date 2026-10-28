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
         patch.object(ci, 'existing', side_effect=[{'metadata-offline.json': tmp_path/'metadata-offline.json.pending'}] * 3 + [None]), \
         patch.object(ci, 'put') as put:
        with pytest.raises(ValueError, match='missing validated'):
            ci.publish(args)
        put.assert_not_called()


def test_completed_publication_not_replaced(tmp_path):
    args = SimpleNamespace(work=tmp_path/'publish', ref='develop', sha=SHA,
                           bucket='bucket', kms='', distribution='')
    with patch.object(ci, 'source_metadata', return_value=({}, {}, INPUTS)), \
         patch.object(ci, 'existing', return_value={'metadata-offline.json': tmp_path/'metadata-offline.json'}), \
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
    path = tmp_path/'metadata.json.pending'
    path.write_text('{}')
    with patch.object(ci, 'run') as run, patch.object(ci, 'verify_uploaded'):
        ci.put('bucket', 'offline/metadata.json', path, 'kms-key', immutable=True)
        command = run.call_args.args
        assert command[command.index('--if-none-match') + 1] == '*'
        assert '--checksum-sha256' in command
        assert command[command.index('--content-type') + 1] == 'application/json'


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
    args = SimpleNamespace(work=tmp_path/'work', ref='develop', sha=SHA,
                           bucket='bucket', distro='ubuntu24', arch='amd64',
                           installer_revision=SHA, run_id=123, validation_only=True)

    def download(bucket, key, path):
        path.write_text('fixture')

    def assemble(spec, root):
        assert set(spec['targets']) == {'host', 'card'}
        assert spec['targets']['card']['image'] == 'sha256:' + 'd' * 64
        (root.parent/ci.assemble.METADATA_NAME).write_text('{}')
        (root.parent/ci.assemble.ARCHIVE_NAME).write_text('validated bundle')

    records = {role: {'resources-checksum': {}} for role in ('host', 'card')}
    paths = {'host': ('host-key', host), 'card': ('card-key', card)}
    with patch.object(ci, 'source_metadata', return_value=(records, paths, INPUTS)), \
         patch.object(ci, 'pull_baseline', return_value='sha256:' + 'd' * 64) as pull, \
         patch.object(ci, 'existing') as existing, patch.object(ci, 'put') as put, \
         patch.object(ci, 'download', side_effect=download), \
         patch.object(ci.assemble, 'verify_publication'), \
         patch.object(ci.assemble, 'assemble', side_effect=assemble) as build:
        ci.build(args)
        existing.assert_not_called()
        put.assert_not_called()
        build.assert_called_once()
        assert [call.args[1] for call in pull.call_args_list] == ['ubuntu24-amd64', 'card']
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


def test_manual_publication_is_opt_in_and_reuses_the_validated_matrix():
    import yaml
    workflows = HERE.parents[2]/'.github/workflows'
    caller = yaml.safe_load((workflows/'vulcan-ci.yml').read_text())
    inputs = caller[True]['workflow_dispatch']['inputs']
    assert inputs['offline_publish']['default'] is False
    call = caller['jobs']['validate-offline']
    assert call['with']['validation_only'] == '${{ !inputs.offline_publish }}'
    assert call['with']['publish_branch'] == '${{ inputs.offline_publish || false }}'
    worker = yaml.safe_load((workflows/'pcie-offline.yml').read_text())
    assert worker['jobs']['publish']['needs'] == ['source', 'config', 'build', 'build-core']
    assert worker['jobs']['publish']['if'] == '${{ !inputs.validation_only }}'
    matrix = worker['jobs']['build']['strategy']['matrix']['include']
    assert {(r['distro'], r['arch']) for r in matrix} == set(ci.TUPLES)


def test_offline_assembly_uses_vulcan_native_pools_without_changing_target_matrix():
    import yaml
    worker = yaml.safe_load((HERE.parents[2]/'.github/workflows/pcie-offline.yml').read_text())
    pool = 'vulcan-${{ needs.config.outputs.vulcan_env }}'
    build = worker['jobs']['build']
    assert build['runs-on'] == ['self-hosted', 'linux', '${{ matrix.arch }}', pool, 'ubuntu', 'default']
    assert worker['jobs']['build-core']['runs-on'] == ['self-hosted', 'linux', 'arm64', pool, 'ubuntu', 'default']
    assert {(row['distro'], row['arch']) for row in build['strategy']['matrix']['include']} == set(ci.TUPLES)
    qemu = next(step for step in build['steps'] if step.get('uses') == 'docker/setup-qemu-action@v3')
    assert qemu['if'] == "${{ matrix.arch == 'amd64' }}"


def test_cloudfront_invalidation_encodes_s3_branch_keys(tmp_path):
    args = SimpleNamespace(work=tmp_path/'publish', ref='codex/offline', sha=SHA,
                           bucket='bucket', kms='', distribution='distribution')
    with patch.object(ci, 'source_metadata', return_value=({}, {}, INPUTS)),          patch.object(ci, 'existing', return_value={'metadata-offline.json': tmp_path/'metadata-offline.json'}),          patch.object(ci, 'run') as run:
        ci.publish(args)
        assert '/core/codex%252Foffline/aaaaaaaaaaaa/pciehost/ubuntu24/amd64/metadata-offline.json' in run.call_args.args


@pytest.mark.parametrize('event,ref,mode,validation,manual_source,failed,cancelled,promoted,expected', [
    ('push', 'refs/heads/develop', 'standard', False, '', False, False, True, True),
    ('pull_request', 'refs/heads/develop', 'standard', False, '', False, False, True, False),
    ('workflow_dispatch', 'refs/heads/develop', 'standard', False, '', False, False, True, False),
    ('push', 'refs/heads/main', 'standard', False, '', False, False, True, False),
    ('push', 'refs/heads/develop', 'fuzz', False, '', False, False, True, False),
    ('push', 'refs/heads/develop', 'standard', True, '', False, False, True, False),
    ('push', 'refs/heads/develop', 'standard', False, '123', False, False, True, False),
    ('push', 'refs/heads/develop', 'standard', False, '', True, False, True, False),
    ('push', 'refs/heads/develop', 'standard', False, '', False, True, True, False),
    ('push', 'refs/heads/develop', 'standard', False, '', False, False, False, False),
])
def test_automatic_develop_publication_gate(event, ref, mode, validation, manual_source,
                                           failed, cancelled, promoted, expected):
    import yaml
    caller = yaml.safe_load((HERE.parents[2]/'.github/workflows/vulcan-ci.yml').read_text())
    job = caller['jobs']['publish-offline-develop']
    # Evaluate the actual workflow gate with representative GitHub contexts.
    expression = job['if'][3:-2].strip()
    values = {
        'always()': True, 'github.event_name': event, 'github.ref': ref,
        'inputs.validation_only': validation, 'inputs.offline_source_run_id': manual_source,
        'inputs.build_mode': mode, "contains(needs.*.result, 'failure')": failed,
        "contains(needs.*.result, 'cancelled')": cancelled,
        'needs.documentation-localization.result': 'success',
        'needs.code-hygiene.result': 'success',
        'needs.promote-latest-tag.result': 'success' if promoted else 'skipped',
    }
    for key, value in values.items():
        expression = expression.replace(key, repr(value))
    expression = expression.replace('&&', ' and ').replace('||', ' or ').replace('!', ' not ')
    assert bool(eval(expression, {'__builtins__': {}}, {})) is expected
    assert job['uses'] == './.github/workflows/pcie-offline.yml'
    assert job['with'] == {'source_run_id': "${{ format('{0}', github.run_id) }}",
                           'publish_branch': True, 'validation_only': False}
    # Every standard job gates automatic publication, including optional hardware.
    standard_jobs = set(caller['jobs']) - {'validate-offline', 'publish-offline-develop', 'validation-complete'}
    assert set(job['needs']) == standard_jobs


def test_completed_run_listener_does_not_duplicate_develop_publication():
    import yaml
    worker = yaml.safe_load((HERE.parents[2]/'.github/workflows/pcie-offline.yml').read_text())
    assert "github.event.workflow_run.head_branch != 'develop'" in worker['jobs']['source']['if']


def test_host_baseline_keys_track_only_platform_inputs():
    checksums = {name: 'a'*64 for name in (
        'platform_provider.py', 'platform-source.json', 'sima-neat-core.deb')}
    record = {'resources-checksum': checksums}
    first = ci.baseline_identity('ubuntu24-arm64', ci.baseline_dependencies(record, 'ubuntu24-arm64'))
    checksums['sima-neat-core.deb'] = 'b'*64
    assert ci.baseline_identity('ubuntu24-arm64', ci.baseline_dependencies(record, 'ubuntu24-arm64')) == first
    checksums['platform-source.json'] = 'b'*64
    assert ci.baseline_identity('ubuntu24-arm64', ci.baseline_dependencies(record, 'ubuntu24-arm64')) != first
    with pytest.raises(ValueError, match='lacks baseline provenance'):
        ci.baseline_dependencies({'resources-checksum': {}}, 'ubuntu24-arm64')


def test_core_consumes_registry_images_without_a_baseline_producer():
    import yaml
    workflow = yaml.safe_load((HERE.parents[2]/'.github/workflows/pcie-offline.yml').read_text())
    assert 'prepare-card' not in workflow['jobs']
    assert 'actions/cache' not in (HERE.parents[2]/'.github/workflows/pcie-offline.yml').read_text()
    source = (HERE/'ci.py').read_text()
    assert "run('docker', 'build'" not in source
    assert "run('docker', 'push'" not in source
    assert not list(HERE.glob('*.Dockerfile'))


def baseline_record():
    return {'resources-checksum': {name: 'a'*64 for name in (
        'platform_provider.py', 'platform-source.json')}}


@pytest.mark.parametrize('role', ['card', 'ubuntu24-arm64'])
@pytest.mark.parametrize('fault', [None, 'labels', 'architecture'])
def test_baseline_resolution_pulls_only_digest_and_checks_provenance(tmp_path, role, fault):
    import json
    args = SimpleNamespace(sha=SHA, work=tmp_path)
    config = ci.baseline_config()
    key = (config['card']['inputs_sha256'] if role == 'card' else
           ci.baseline_identity(role, ci.baseline_dependencies(baseline_record(), role)))
    digest = config['card']['digest'] if role == 'card' else 'sha256:'+'b'*64
    local_id = 'sha256:'+'c'*64
    labels = {'ai.sima.offline.'+name: value for name, value in
              [('inputs', key), ('recipe', config['recipe_sha256']), ('role', role)]}
    if fault == 'labels':
        labels['ai.sima.offline.inputs'] = 'd'*64
    info = {'Architecture': 'amd64' if fault == 'architecture' else 'arm64',
            'Config': {'Labels': labels}, 'Id': local_id}
    result = SimpleNamespace(returncode=0, stdout=json.dumps({'Descriptor': {'digest': digest}}))
    with patch.object(ci.subprocess, 'run', return_value=result) as manifest, \
         patch.object(ci, 'run', side_effect=['', json.dumps([info])]) as command:
        if fault:
            with pytest.raises(ValueError, match='provenance|architecture'):
                ci.pull_baseline(args, role, baseline_record())
            assert not (tmp_path/'baseline-images.json').exists()
        else:
            assert ci.pull_baseline(args, role, baseline_record()) == local_id
            evidence = json.loads((tmp_path/'baseline-images.json').read_text())
            assert evidence[role]['image'] == config['repository']+'@'+digest
            assert evidence[role]['inputs_sha256'] == key
        assert command.call_args_list[0].args == ('docker', 'pull', config['repository']+'@'+digest)
        if role == 'card':
            manifest.assert_not_called()
        else:
            manifest.assert_called_once()


def test_missing_host_baseline_fails_without_local_build(tmp_path):
    result = SimpleNamespace(returncode=1, stderr='manifest unknown')
    with patch.object(ci.subprocess, 'run', return_value=result), patch.object(ci, 'run') as docker:
        with pytest.raises(RuntimeError, match='Publish matching images with Vulcan'):
            ci.pull_baseline(SimpleNamespace(sha=SHA, work=tmp_path), 'ubuntu24-arm64', baseline_record())
        docker.assert_not_called()


def test_card_image_pin_is_independent_of_packages_under_test(tmp_path):
    import json
    config = ci.baseline_config()
    info = {'Architecture': 'arm64', 'Id': config['card']['digest'], 'Config': {'Labels': {
        'ai.sima.offline.inputs': config['card']['inputs_sha256'],
        'ai.sima.offline.recipe': config['recipe_sha256'], 'ai.sima.offline.role': 'card'}}}
    with patch.object(ci.subprocess, 'run') as manifest, \
         patch.object(ci, 'run', side_effect=['', json.dumps([info])]*2) as docker:
        for sha in ('a'*40, 'b'*40):
            record = {'resources-checksum': {name: sha for name in (
                'simaai-scheduler.deb', 'resolved-deps-manifest.json', 'scheduler_platform_admission.py')}}
            assert ci.pull_baseline(SimpleNamespace(sha=sha, work=tmp_path), 'card', record) == config['card']['digest']
        pulls = [call.args for call in docker.call_args_list if call.args[:2] == ('docker', 'pull')]
        assert pulls == [('docker', 'pull', config['repository']+'@'+config['card']['digest'])]*2
        manifest.assert_not_called()


@pytest.mark.parametrize('field,value', [('digest', 'latest'), ('inputs_sha256', 'invalid')])
def test_invalid_card_pin_is_rejected_before_docker(tmp_path, field, value):
    config = ci.baseline_config()
    config['card'][field] = value
    with patch.object(ci, 'baseline_config', return_value=config), patch.object(ci, 'run') as docker:
        with pytest.raises(ValueError, match='invalid pinned card baseline'):
            ci.pull_baseline(SimpleNamespace(sha=SHA, work=tmp_path), 'card', {})
        docker.assert_not_called()
