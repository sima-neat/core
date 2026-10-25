"""Immutable CLI transport tests; lower SDK admission remains Scheduler-owned."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from urllib.parse import quote, unquote
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ci'))
import published_artifact as transport
import acquire_core_qualification as core_transport


def test_literal_branch_grammar_matches_git_without_repository(tmp_path):
    git = shutil.which('git')
    assert git, 'Host differential test requires Git'
    names = {'main', 'feature/a65', '@', 'HEAD', 'head', 'foo./bar', 'foo/-bar',
             'éclair/東京', 'foo.locked/bar', 'foo.LOCK', 'a' * 256, '', '.', '/',
             '-topic', '/topic', 'topic/', 'topic//other', '.topic', 'topic/.hidden',
             'topic.lock', 'topic.lock/child', 'topic.', 'topic..child', '@{-1}',
             'topic@{1}', 'topic:main', 'a@b', 'topic#1', 'topic;$value'}
    for number in range(1, 128):
        char = chr(number)
        names.update((char + 'topic', 'top' + char + 'ic', 'topic/' + char + 'child'))
    env = {key: value for key, value in os.environ.items() if not key.startswith('GIT_')}
    for name in sorted(names):
        checked = subprocess.run([git, 'check-ref-format', '--branch', name], cwd=tmp_path,
                                 env=env, capture_output=True)
        assert transport.valid_branch(name) == (checked.returncode == 0), repr(name)
    for name in (None, 12, [], 'topic\0child'):
        assert not transport.valid_branch(name)


@pytest.fixture
def git_free_archive(tmp_path):
    # Match checkout's archive fallback: real entrypoint bytes, no .git or Git
    # executable. Entry points must still reach the selected CLI.
    archive = tmp_path / 'archive'; archive.mkdir()
    for name in ('tools/ci/published_artifact.py', 'tools/ci/acquire_core_qualification.py'):
        target = archive / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    binaries = tmp_path / 'bin'; binaries.mkdir()
    calls = tmp_path / 'cli-calls.json'
    cli = binaries / 'fixture-cli'
    cli.write_text(f'#!{sys.executable}\nimport json, pathlib, sys\n'
                   f'pathlib.Path({str(calls)!r}).write_text(json.dumps(sys.argv[1:]))\n'
                   'sys.exit(29)\n')
    cli.chmod(0o755)
    assert shutil.which('git', path=str(binaries)) is None
    return archive, binaries, calls


@pytest.mark.parametrize('entrypoint', ['publication', 'core'])
@pytest.mark.parametrize('branch,valid', [
    ('feature/core', True), ('@', True), ('éclair/東京', True), ('foo./bar', True),
    ('HEAD', False), ('-topic', False), ('topic.lock/child', False),
    ('topic/.hidden', False), ('topic//child', False), ('topic.', False),
    ('@{-1}', False), ('topic:other', False), ('topic\nchild', False), ('', False),
])
def test_real_archive_entrypoints_without_git(git_free_archive, entrypoint, branch, valid):
    archive, binaries, calls = git_free_archive
    output = archive / 'out'
    if entrypoint == 'publication':
        program = ('import sys; sys.path.insert(0, "tools/ci"); '
                   'from published_artifact import acquire; '
                   'acquire("fixture-cli", "scheduler", sys.argv[1], "a" * 40, '
                   '"production", "out")')
        command = [sys.executable, '-c', program, branch]
    else:
        command = [sys.executable, str(archive / 'tools/ci/acquire_core_qualification.py'),
                   '--cli', 'fixture-cli',
                   '--branch=' + branch, '--commit', 'a' * 40, '--environment', 'prod',
                   '--output', str(output)]
    result = subprocess.run(command, cwd=archive, env={**os.environ, 'PATH': str(binaries)},
                            capture_output=True, text=True)
    assert result.returncode != 0
    assert 'FileNotFoundError' not in result.stderr
    if valid:
        # A real failing CLI must be reached unchanged and its failure retained;
        # this control performs no network access, package install or staging.
        argv = json.loads(calls.read_text())
        assert any('@' + branch + ':' + 'a' * 12 in value for value in argv)
        assert '--json' in argv and 'exit status 29' in result.stderr
    else:
        assert not calls.exists() and not output.exists()
        assert 'ValueError: Invalid' in result.stderr


@pytest.fixture
def publication(tmp_path):
    source = tmp_path / 'published'; source.mkdir()
    component, branch, commit = 'scheduler', 'topic', 'a' * 40
    resources = {'client_bootstrap.py': b'# producer bootstrap', 'runtime.deb': b'archive'}
    def write(name, data):
        path = source / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    for name, data in resources.items(): write(name, data)
    metadata = {'resources': list(resources), 'resources-checksum': {
        name: hashlib.sha256(data).hexdigest() for name, data in resources.items()}}
    write('metadata.json', json.dumps(metadata).encode())
    write('candidate.json', b'{"fixture":true}')
    manifest = {'repository': 'sima-neat/' + component, 'repository_key': component,
                'branch': branch, 'commit': commit, 'artifact_folder': '', 'artifacts': [
                    {'path': p.name, 'sha256': transport.digest(p), 'size': p.stat().st_size}
                    for p in source.iterdir()]}
    selected = {'repository': component, 'ref': branch, 'resolved_spec': commit[:12],
                'environment': 'production', 'metadata_url': 'https://artifacts.neat.sima.ai/x/metadata.json',
                'package_path': ''}
    calls = []
    def fetch(url, path, *, sha256=None, size=None):
        data = json.dumps(manifest).encode() if url.endswith('/manifest.json') else (source / url.rsplit('/', 1)[-1]).read_bytes()
        Path(path).write_bytes(data)
        if sha256 is not None and transport.digest(path) != sha256: raise ValueError('hash differs')
        if size is not None and Path(path).stat().st_size != size: raise ValueError('size differs')
    def output(argv, **kwargs):
        calls.append(argv)
        return 'CLI message\n' + json.dumps(selected)
    original_run = subprocess.run
    def run(argv, **kwargs):
        if argv[0] == 'git': return original_run(argv, **kwargs)
        calls.append(argv)
        out = Path(argv[argv.index('-d') + 1]); out.mkdir()
        if argv[1] == 'download':
            names = [argv[2].rsplit('/', 1)[-1]]
        else: names = json.loads((source / 'metadata.json').read_text())['resources']
        for name in names: shutil.copy2(source / unquote(name), out / name)
        return subprocess.CompletedProcess(argv, 0)
    with mock.patch.object(transport, 'download', side_effect=fetch), \
         mock.patch.object(transport.subprocess, 'check_output', side_effect=output), \
         mock.patch.object(transport.subprocess, 'run', side_effect=run):
        yield tmp_path, source, manifest, selected, calls


def acquire(fixture, **kwargs):
    root, *_ = fixture
    return transport.acquire('fixture-cli', 'scheduler', 'topic', 'latest', 'production', root / 'out', **kwargs)


def test_resolve_once_then_download_immutable(publication):
    receipt = acquire(publication)
    assert receipt['commit'] == 'a' * 40
    assert len(publication[-1]) == 2
    assert publication[-1][1][1:4] == ['neat', 'download', 'scheduler@topic:' + 'a' * 12]
    assert (publication[0] / 'out/candidate.json').is_file()


def test_full_publication_transport_does_not_require_git(publication, monkeypatch):
    monkeypatch.setenv('PATH', str(publication[0]))
    assert acquire(publication)['commit'] == 'a' * 40
    assert len(publication[-1]) == 2


@pytest.mark.parametrize('inherited', [None, '0', '1'])
@pytest.mark.parametrize('resource_pattern', [None, r'runtime\.deb'])
def test_selected_cli_children_disable_updates_without_changing_parent(
        publication, monkeypatch, inherited, resource_pattern):
    if inherited is None:
        monkeypatch.delenv('SIMA_CLI_CHECK_FOR_UPDATE', raising=False)
    else:
        monkeypatch.setenv('SIMA_CLI_CHECK_FOR_UPDATE', inherited)
    monkeypatch.setenv('SIMA_CLI_TRANSPORT_TEST', 'preserved caller value')
    parent = dict(os.environ)
    output, run = transport.subprocess.check_output, transport.subprocess.run
    children = []

    def check_child(argv, kwargs):
        assert kwargs['env'] == {**parent, 'SIMA_CLI_CHECK_FOR_UPDATE': '0'}
        children.append(argv)

    def selected_output(argv, **kwargs):
        check_child(argv, kwargs)
        return output(argv, **kwargs)

    def selected_run(argv, **kwargs):
        if argv[0] == 'git':
            assert 'env' not in kwargs
        else:
            check_child(argv, kwargs)
        return run(argv, **kwargs)

    with mock.patch.object(transport.subprocess, 'check_output', side_effect=selected_output), \
         mock.patch.object(transport.subprocess, 'run', side_effect=selected_run):
        receipt = acquire(publication, resource_pattern=resource_pattern)
    assert receipt['commit'] == 'a' * 40
    assert len(children) == 2
    assert dict(os.environ) == parent


@pytest.mark.parametrize('phase', ['resolution', 'bundle', 'resource'])
def test_selected_cli_failure_is_not_retried_or_hidden(publication, monkeypatch, phase):
    monkeypatch.setenv('SIMA_CLI_CHECK_FOR_UPDATE', '1')
    parent = dict(os.environ)
    run = transport.subprocess.run
    failed = []

    def fail(argv, **kwargs):
        assert kwargs['env'] == {**parent, 'SIMA_CLI_CHECK_FOR_UPDATE': '0'}
        failed.append(argv)
        raise subprocess.CalledProcessError(29, argv)

    def selected_run(argv, **kwargs):
        return run(argv, **kwargs) if argv[0] == 'git' else fail(argv, **kwargs)

    method = 'check_output' if phase == 'resolution' else 'run'
    with mock.patch.object(transport.subprocess, method,
                           side_effect=fail if phase == 'resolution' else selected_run):
        with pytest.raises(subprocess.CalledProcessError) as error:
            acquire(publication, resource_pattern=r'runtime\.deb' if phase == 'resource' else None)
    assert error.value.returncode == 29
    assert len(failed) == 1
    assert not (publication[0] / 'out/acquisition.json').exists()
    assert dict(os.environ) == parent


@pytest.mark.parametrize('length', [12, 20, 40])
def test_pinned_commit_uses_short_lookup_and_preserves_full_authority(publication, length):
    root, _, _, _, calls = publication
    receipt = transport.acquire('fixture-cli', 'scheduler', 'topic', 'a' * length,
                                'production', root / 'out')
    assert calls[0][5] == 'scheduler@topic:' + 'a' * 12
    assert calls[1][3] == 'scheduler@topic:' + 'a' * 12
    assert receipt['commit'] == 'a' * 40


def test_full_commit_with_same_short_prefix_but_different_tail_is_rejected(publication):
    root, _, manifest, _, calls = publication
    manifest['commit'] = 'a' * 12 + 'b' * 28
    with pytest.raises(ValueError, match='Publisher identity'):
        transport.acquire('fixture-cli', 'scheduler', 'topic', 'a' * 40,
                          'production', root / 'out')
    assert calls[0][5] == 'scheduler@topic:' + 'a' * 12
    assert len(calls) == 1
    assert not (root / 'out/download').exists()


@pytest.mark.parametrize('field,value', [('repository', 'other'), ('ref', 'develop'),
                                         ('environment', 'dev'), ('resolved_spec', 'latest')])
def test_different_resolution_fails_before_download(publication, field, value):
    publication[3][field] = value
    with pytest.raises(ValueError, match='different publication'): acquire(publication)
    assert len(publication[-1]) == 1


@pytest.mark.parametrize('field,value', [('commit', 'b' * 40), ('branch', 'develop'),
                                         ('repository_key', 'runner'), ('artifact_folder', 'other')])
def test_publisher_mismatch_fails_before_download(publication, field, value):
    publication[2][field] = value
    with pytest.raises(ValueError, match='Publisher identity'): acquire(publication)
    assert len(publication[-1]) == 1


def test_changed_download_not_executed(publication):
    (publication[1] / 'runtime.deb').write_bytes(b'changed')
    with pytest.raises(ValueError, match='differs from publisher'): acquire(publication)


@pytest.mark.parametrize('distro', ['ubuntu22', 'ubuntu24'])
@pytest.mark.parametrize('arch', ['amd64', 'arm64'])
def test_explicit_native_package_path(publication, distro, arch):
    path = f'pcie-host-{distro}-{arch}'
    publication[2]['artifact_folder'] = path
    publication[3]['package_path'] = path
    result = acquire(publication, package_path=path)
    assert result['package_path'] == path
    assert publication[-1][0][5] == 'scheduler/' + path + '@topic:latest'
    assert publication[-1][1][3] == 'scheduler/' + path + '@topic:' + 'a' * 12


@pytest.mark.parametrize('distro', ['ubuntu22', 'ubuntu24'])
@pytest.mark.parametrize('arch', ['amd64', 'arm64'])
@pytest.mark.parametrize('mismatch', ['device', 'nested', 'other-architecture'])
def test_native_manifest_must_match_selected_profile(publication, distro, arch, mismatch):
    path = f'pcie-host-{distro}-{arch}'
    other_arch = 'arm64' if arch == 'amd64' else 'amd64'
    publication[3]['package_path'] = path
    publication[2]['artifact_folder'] = {
        'device': '',
        'nested': f'pcie-host/{distro}/{arch}',
        'other-architecture': f'pcie-host-{distro}-{other_arch}',
    }[mismatch]
    with pytest.raises(ValueError, match='Publisher identity'):
        acquire(publication, package_path=path)
    # Wrong profiles never reach the CLI payload download or provider execution.
    assert len(publication[-1]) == 1
    assert not (publication[0] / 'out/download').exists()


def test_generic_nested_package_path_remains_supported(publication):
    path = 'pciehost/ubuntu24/amd64'
    publication[2]['artifact_folder'] = path
    publication[3]['package_path'] = path
    result = acquire(publication, package_path=path)
    assert result['package_path'] == path
    assert publication[-1][1][3] == 'scheduler/' + path + '@topic:' + 'a' * 12


def test_nested_publisher_files_are_not_downloaded_as_llima_dev(publication):
    root, source, manifest, _, calls = publication
    name = 'sima-lmm-0.6.0+develop.aaaaaaaaaaaa-Linux-dev.deb'
    (source / name).write_bytes(b'dev')
    metadata = json.loads((source / 'metadata.json').read_text())
    metadata['resources'].append(name)
    metadata['resources-checksum'][name] = transport.digest(source / name)
    (source / 'metadata.json').write_text(json.dumps(metadata))
    manifest['artifacts'] = [entry for entry in manifest['artifacts'] if entry['path'] != 'metadata.json']
    for path in (source / name, source / 'metadata.json'):
        manifest['artifacts'].append({'path': path.name, 'size': path.stat().st_size, 'sha256': transport.digest(path)})
    manifest['artifacts'].append({'path': '_CPack_Packages/huge/file', 'size': 10**12, 'sha256': 'b' * 64})
    result = acquire(publication, resource_pattern=r'sima-lmm-.*-Linux-dev\.deb')
    assert calls[-1][1] == 'download' and calls[-1][2].endswith('/' + quote(name, safe=''))
    assert set(result['resources']) == {'metadata.json', name}
    assert not (root / 'out/runtime.deb').exists()


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'a/../b', 'path\\name'])
def test_unsafe_publisher_path_rejected(publication, name):
    publication[2]['artifacts'].append({'path': name, 'size': 0, 'sha256': 'a' * 64})
    with pytest.raises(ValueError, match='Unsafe publisher'): acquire(publication)


def test_cross_environment_origin_is_rejected(publication):
    publication[3]['metadata_url'] = 'https://artifacts.stg.neat.sima.ai/x/metadata.json'
    with pytest.raises(ValueError, match='selected environment'):
        acquire(publication)


def test_staging_environment_is_preserved(publication):
    root, _, _, selected, calls = publication
    selected['environment'] = 'staging'
    selected['metadata_url'] = 'https://artifacts.stg.neat.sima.ai/x/metadata.json'
    result = transport.acquire('fixture-cli', 'scheduler', 'topic', 'latest', 'stg', root / 'out')
    assert result['environment'] == 'staging'
    assert calls[0][calls[0].index('--env') + 1] == 'staging'


def test_nested_selected_resource_rejected_before_cli_download(publication):
    _, source, manifest, _, calls = publication
    path = source / "nested/runtime.deb"
    path.parent.mkdir(); path.write_bytes(b"nested")
    name = "nested/runtime.deb"
    manifest["artifacts"].append({"path": name, "sha256": transport.digest(path), "size": path.stat().st_size})
    metadata = {"resources": [name], "resources-checksum": {name: transport.digest(path)}}
    (source / "metadata.json").write_text(json.dumps(metadata))
    record = next(item for item in manifest["artifacts"] if item["path"] == "metadata.json")
    record.update(sha256=transport.digest(source / "metadata.json"), size=(source / "metadata.json").stat().st_size)
    with pytest.raises(ValueError, match="Unsafe published resource"):
        acquire(publication)
    assert len(calls) == 1


@pytest.fixture
def core_publication(tmp_path):
    source = tmp_path / 'core-published'; source.mkdir()
    branch, commit = 'feature/core', 'a' * 40
    payloads = {'manifest.json': b'{"abi-version":"6"}', 'core.deb': b'Core package'}
    for name, data in payloads.items(): (source / name).write_bytes(data)
    metadata = {'name': 'gh:sima-neat/core', 'version': '0.6.0+feature-core.' + commit[:12],
                'resources': list(payloads), 'resources-checksum': {
                    name: transport.digest(source / name) for name in payloads}}
    for name in core_transport.METADATA:
        (source / name).write_text(json.dumps(metadata))
    selected = {'repository': 'core', 'package_path': '', 'ref': branch,
                'requested_spec': commit[:12], 'resolved_spec': commit[:12],
                'environment': 'production', 'base_url': 'https://artifacts.neat.sima.ai',
                'metadata_url': 'https://artifacts.neat.sima.ai/core/feature%252Fcore/' + commit[:12] + '/metadata-all.json'}
    calls = []
    def fetch(url, output, **kwargs):
        Path(output).write_bytes((source / url.rsplit('/', 1)[-1]).read_bytes())
    def output(argv, **kwargs):
        calls.append(argv)
        return 'CLI version message\n' + json.dumps(selected)
    original_run = subprocess.run
    def run(argv, **kwargs):
        if argv[0] == 'git': return original_run(argv, **kwargs)
        calls.append(argv)
        if argv[0] == 'fixture-cli':
            destination = Path(argv[argv.index('-d') + 1])
            for name in json.loads((source / 'metadata-all.json').read_text())['resources']:
                path = source / name
                if path.exists(): shutil.copy2(path, destination / name, follow_symlinks=False)
        return subprocess.CompletedProcess(argv, 0)
    with mock.patch.object(core_transport, 'download', side_effect=fetch), \
         mock.patch.object(core_transport.subprocess, 'check_output', side_effect=output), \
         mock.patch.object(core_transport.subprocess, 'run', side_effect=run):
        yield tmp_path, source, selected, calls


def acquire_core(fixture, **kwargs):
    root, *_ = fixture
    return core_transport.acquire_core('fixture-cli', 'feature/core', 'a' * 40,
                                       kwargs.pop('environment', 'production'), root / 'out', ROOT, **kwargs)


def test_full_core_transport_does_not_require_git(core_publication, monkeypatch):
    monkeypatch.setenv('PATH', str(core_publication[0]))
    assert acquire_core(core_publication)['status'] == 'PASS'
    assert len(core_publication[-1]) == 3  # Resolve, download, full verifier.


@pytest.mark.parametrize('environment', ['production', 'prod', 'prd'])
def test_core_uses_original_metadata_not_scheduler_publisher_envelope(core_publication, environment):
    root, source, _, calls = core_publication
    result = acquire_core(core_publication, environment=environment)
    assert result['selector'] == 'core@feature/core:' + 'a' * 12
    assert result['resources'] == 2
    assert calls[0][1:] == ['neat', 'download', result['selector'], '--env', 'production', '-t', 'all', '--json']
    assert calls[1][1:] == ['neat', 'download', result['selector'], '--env', 'production', '-t', 'all', '-d', str(root / 'out')]
    assert calls[2] == ['bash', str(ROOT / 'scripts/ci/verify_core_dist.sh'), '--dist', str(root / 'out'),
                        '--expected-manifest', str(ROOT / 'deps/manifest.json')]
    for name in (*core_transport.METADATA, 'manifest.json', 'core.deb'):
        assert (root / 'out' / name).read_bytes() == (source / name).read_bytes()
    assert not (root / 'out/publication.json').exists()


@pytest.mark.parametrize('field,value', [
    ('repository', 'scheduler'), ('package_path', 'pciehost/ubuntu24/amd64'),
    ('ref', 'develop'), ('environment', 'staging'), ('requested_spec', 'latest'),
    ('resolved_spec', 'b' * 12), ('base_url', 'https://example.org'),
    ('metadata_url', 'https://artifacts.neat.sima.ai/core/develop/' + 'a' * 12 + '/metadata-all.json'),
    ('metadata_url', 'https://artifacts.stg.neat.sima.ai/core/feature%252Fcore/' + 'a' * 12 + '/metadata-all.json'),
])
def test_core_wrong_resolution_never_downloads_payload(core_publication, field, value):
    core_publication[2][field] = value
    with pytest.raises(ValueError, match='different Core publication'):
        acquire_core(core_publication)
    assert len(core_publication[-1]) == 1
    assert not (core_publication[0] / 'out').exists()


@pytest.mark.parametrize('field,value,all_variants', [
    ('name', 'foreign', True), ('name', 'sima-neat', False),
    ('version', '0.6.0+feature-core.' + 'b' * 12, False),
])
def test_core_wrong_or_mixed_metadata_identity_fails_before_payload(core_publication, field, value, all_variants):
    source = core_publication[1]
    for name in core_transport.METADATA if all_variants else ('metadata-extras.json',):
        path = source / name; item = json.loads(path.read_text()); item[field] = value
        path.write_text(json.dumps(item))
    with pytest.raises(ValueError, match='metadata identity'):
        acquire_core(core_publication)
    assert len(core_publication[-1]) == 1


@pytest.mark.parametrize('identity', [('sima-neat', '0.6.0'), ('gh:sima-neat/core', '0.6.0')])
def test_core_release_versions_need_not_encode_commit(core_publication, identity):
    # Production CMake uses exact tags as versions. Commit authority is the
    # immutable CLI selector/URL; the full verifier checks actual DEB versions.
    for name in core_transport.METADATA:
        path = core_publication[1] / name; value = json.loads(path.read_text())
        value['name'], value['version'] = identity
        path.write_text(json.dumps(value))
    assert acquire_core(core_publication)['status'] == 'PASS'


@pytest.mark.parametrize('name', core_transport.METADATA)
def test_core_missing_metadata_variant_fails_before_payload(core_publication, name):
    (core_publication[1] / name).unlink()
    with pytest.raises(FileNotFoundError): acquire_core(core_publication)
    assert len(core_publication[-1]) == 1


@pytest.mark.parametrize('mutation', ['hash', 'missing', 'symlink'])
def test_core_payload_hash_missing_and_symlink_fail_before_verifier(core_publication, mutation):
    path = core_publication[1] / 'core.deb'
    if mutation == 'hash': path.write_bytes(b'changed')
    elif mutation == 'missing': path.unlink()
    else:
        real = path.with_suffix('.target'); path.rename(real); path.symlink_to(real)
    with pytest.raises(ValueError, match='differs from selected metadata'):
        acquire_core(core_publication)
    assert len(core_publication[-1]) == 2


@pytest.mark.parametrize('name', core_transport.METADATA)
def test_core_cli_cannot_replace_selected_metadata_before_owner_execution(core_publication, name):
    original = core_transport.subprocess.run.side_effect
    def overwrite(argv, **kwargs):
        result = original(argv, **kwargs)
        if argv[0] == 'fixture-cli':
            selected = Path(argv[argv.index('-d') + 1]) / name
            value = json.loads(selected.read_text())
            value['version'] = 'changed-by-cli'
            selected.write_text(json.dumps(value))
        return result
    core_transport.subprocess.run.side_effect = overwrite
    with pytest.raises(ValueError, match='differs from selected metadata: ' + name):
        acquire_core(core_publication)
    assert len(core_publication[-1]) == 2


def test_core_expected_inventory_is_generated_by_full_verifier_after_authentication(core_publication):
    output = core_publication[0] / 'expected.json'
    acquire_core(core_publication, package_profile='core-codecs-llima', write_package_inventory=output)
    assert core_publication[-1][-1][-4:] == ['--package-profile', 'core-codecs-llima',
                                           '--write-package-inventory', str(output)]


@pytest.mark.parametrize('options', [{'package_profile': 'core'}, {'write_package_inventory': Path('expected.json')}])
def test_core_expected_inventory_requires_profile_and_output_before_download(core_publication, options):
    with pytest.raises(ValueError, match='explicit package profile'):
        acquire_core(core_publication, **options)
    assert not core_publication[-1]


@pytest.mark.parametrize('name', ['../escape', 'nested/archive.deb', 'metadata.json'])
def test_core_unsafe_payload_cannot_overwrite_authority(core_publication, name):
    path = core_publication[1] / 'metadata-all.json'
    data = json.loads(path.read_text()); data['resources'] = [name]; data['resources-checksum'] = {name: 'a' * 64}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): acquire_core(core_publication)
    assert len(core_publication[-1]) == 1


def test_core_full_verifier_failure_propagates(core_publication):
    original = core_transport.subprocess.run.side_effect
    def run(argv, **kwargs):
        if argv[0] == 'bash': raise subprocess.CalledProcessError(1, argv)
        return original(argv, **kwargs)
    core_transport.subprocess.run.side_effect = run
    with pytest.raises(subprocess.CalledProcessError): acquire_core(core_publication)


def test_core_existing_destination_is_not_reused(core_publication):
    (core_publication[0] / 'out').mkdir()
    with pytest.raises(FileExistsError): acquire_core(core_publication)
    assert len(core_publication[-1]) == 1


def test_core_workflow_only_qualification_transport_uses_publication():
    source = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()
    block = source.split("      - name: Download this build's installed-qualification inputs", 1)[1]
    block = block.split('      - name: Capture native platform', 1)[0]
    published, candidate = block.split("      - name: Download this run's Core validation candidate", 1)
    assert 'if: ${{ !inputs.validation_only }}' in published
    assert 'acquire_core_qualification.py' in published
    assert '--commit "${GITHUB_SHA}"' in published and '--environment "${VULCAN_ENV}"' in published
    assert 'actions/download-artifact' not in published
    assert candidate.count('if: ${{ inputs.validation_only }}') == 2
    assert 'actions/download-artifact@v4' in candidate and 'name: neat-core-dist' in candidate
    assert 'scripts/ci/verify_core_dist.sh --dist .' in candidate
    assert '--expected-manifest deps/manifest.json --expected-commit "$GITHUB_SHA"' in candidate
    assert '--write-package-inventory _work/expected-package-inventory.json' in candidate
    assert 'cleanup_github_artifacts: false' not in source
