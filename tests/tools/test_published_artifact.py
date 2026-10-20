"""Immutable CLI transport tests; lower SDK admission remains Scheduler-owned."""
import hashlib
import importlib.util
import json
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
