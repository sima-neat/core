"""Real model fixtures must be pinned, offline at execution, and fail closed."""
from concurrent.futures import ProcessPoolExecutor
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'tests/tools/prepare_registry_mpk_fixtures.py'
spec = importlib.util.spec_from_file_location('registry_fixtures', SCRIPT)
registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)


def write_archive(path, members):
    with tarfile.open(path, 'w:gz') as stream:
        for name, content in members:
            member = tarfile.TarInfo(name)
            if content is None:
                member.type = tarfile.SYMTYPE
                member.linkname = 'outside.json'
                stream.addfile(member)
            else:
                member.size = len(content)
                stream.addfile(member, io.BytesIO(content))


@pytest.fixture
def fixture(tmp_path):
    source = tmp_path / 'source.tar.gz'
    content = b'{"name":"real_model","plugins":[]}\n'
    write_archive(source, [('model/mpk.json', content), ('etc/ignore.json', b'not JSON'),
                           ('share/model.elf', b'not executable')])
    pin = {'model_id': 'fixture_model', 'variant': 'w8a8',
           'run_id': 'cc3fac2d-d85d-492e-8c29-f56d44c7a301',
           'sha256': registry.sha256(source), 'size_bytes': source.stat().st_size}
    document = {'branch': 'published/snapshot', 'staging': True, 'fixtures': [pin]}
    pins = tmp_path / 'pins.json'
    pins.write_text(json.dumps(document))
    return source, content, pin, document, pins, tmp_path / 'output', tmp_path / 'cache'


def cached(fixture):
    source, content, pin, document, pins, root, cache = fixture
    cache.mkdir()
    archive = cache / (pin['sha256'] + '.tar.gz')
    shutil.copyfile(source, archive)
    return archive


def test_cold_download_uses_standard_cli_and_checks_every_identity(fixture, monkeypatch):
    source, content, pin, document, pins, root, cache = fixture
    calls = []
    monkeypatch.setenv('SIMA_MODELS_REGISTRY_BASE_URL', 'https://not-the-pinned-registry.invalid')

    def download(command, **kwargs):
        calls.append(command)
        assert kwargs['env']['SIMA_CLI_CHECK_FOR_UPDATE'] == '0'
        assert 'SIMA_MODELS_REGISTRY_BASE_URL' not in kwargs['env']
        assert kwargs['timeout'] == 600
        path = Path(command[command.index('--output') + 1]) / 'model.tar.gz'
        shutil.copyfile(source, path)
        selected = {**pin, 'variant_id': pin['variant'], 'path': str(path)}
        return subprocess.CompletedProcess(command, 0, json.dumps(selected), '')

    monkeypatch.setattr(registry.subprocess, 'run', download)
    result = registry.prepare(pins, root, cache)
    assert len(calls) == 1
    assert calls[0][:3] == ['sima-cli', 'models', 'download']
    assert calls[0][3:9] == ['--branch', document['branch'], '--id', pin['model_id'],
                           '--variant', pin['variant']]
    assert calls[0][-1] == '--stg'
    assert (root / 'fixture_model-w8a8-mpk.json').read_bytes() == content
    assert {p.name for p in root.iterdir()} == {
        '.prepare.lock', 'fixture_model-w8a8-mpk.json', 'provenance.json'}
    receipt = result['fixtures'][0]
    assert receipt['manifest_member'] == 'model/mpk.json'
    assert receipt['manifest_sha256'] == hashlib.sha256(content).hexdigest()
    assert receipt['run_id'] == pin['run_id']
    assert (cache / (pin['sha256'] + '.tar.gz')).is_file()
    assert not list(cache.glob('registry-*'))


def test_warm_and_explicit_offline_runs_need_no_cli_or_network(fixture, monkeypatch):
    source, content, pin, document, pins, root, cache = fixture
    cached(fixture)

    def denied(*args, **kwargs):
        pytest.fail('A verified cached fixture must not start a CLI/network operation')

    monkeypatch.setattr(registry.subprocess, 'run', denied)
    registry.prepare(pins, root, cache, offline=True)
    output = root / 'fixture_model-w8a8-mpk.json'
    original_mtime = output.stat().st_mtime_ns
    registry.prepare(pins, root, cache)
    assert output.read_bytes() == content
    assert output.stat().st_mtime_ns == original_mtime


@pytest.mark.parametrize('failure', ['missing', 'corrupt', 'truncated', 'symlink'])
def test_invalid_cache_is_not_replaced_or_silently_downloaded(fixture, monkeypatch, failure):
    source, content, pin, document, pins, root, cache = fixture
    archive = cached(fixture)
    if failure == 'missing':
        archive.unlink()
    elif failure == 'symlink':
        archive.unlink()
        archive.symlink_to(source)
    else:
        data = bytearray(archive.read_bytes())
        if failure == 'corrupt':
            data[-1] ^= 1
        else:
            data = data[:20]
        archive.write_bytes(data)
    monkeypatch.setattr(registry.subprocess, 'run', lambda *a, **k: pytest.fail('unexpected download'))
    with pytest.raises(ValueError, match='offline cache|reviewed pin'):
        registry.prepare(pins, root, cache, offline=True)
    assert not (root / 'provenance.json').exists()


@pytest.mark.parametrize('field', ['model_id', 'variant_id', 'run_id', 'sha256', 'size_bytes',
                                  'outside', 'symlink', 'bytes', 'cli_error'])
def test_download_failure_never_publishes_a_fixture(fixture, monkeypatch, field):
    source, content, pin, document, pins, root, cache = fixture

    def download(command, **kwargs):
        path = Path(command[command.index('--output') + 1]) / 'model.tar.gz'
        shutil.copyfile(source, path)
        selected = {**pin, 'variant_id': pin['variant'], 'path': str(path)}
        if field in selected:
            selected[field] = 0 if field == 'size_bytes' else 'different'
        elif field == 'outside':
            selected['path'] = str(source)
        elif field == 'symlink':
            path.unlink()
            path.symlink_to(source)
        elif field == 'bytes':
            data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
        if field == 'cli_error':
            return subprocess.CompletedProcess(command, 1, '', 'secret download URL')
        return subprocess.CompletedProcess(command, 0, json.dumps(selected), '')

    monkeypatch.setattr(registry.subprocess, 'run', download)
    with pytest.raises(ValueError) as raised:
        registry.prepare(pins, root, cache)
    assert 'secret' not in str(raised.value)
    assert not (root / 'provenance.json').exists()
    assert not list(root.glob('*-mpk.json'))
    assert not (cache / (pin['sha256'] + '.tar.gz')).exists()


@pytest.mark.parametrize('members', [
    [('config.json', b'{}')],
    [('mpk.json', b'{}'), ('other_mpk.json', b'{}')],
    [('mpk.json', b'{}'), ('mpk.json', b'{}')],
    [('../model_mpk.json', b'{}')],
    [('/model/mpk.json', b'{}')],
    [('model\\mpk.json', b'{}')],
    [('mpk.json', None)],
    [('mpk.json', b'not JSON')],
    [('mpk.json', b'[]')],
    [('mpk.json', b'')],
])
def test_missing_ambiguous_unsafe_and_invalid_manifests_fail(tmp_path, members):
    archive = tmp_path / 'fixture.tar.gz'
    write_archive(archive, members)
    with pytest.raises(ValueError):
        registry.read_manifest(archive)


def test_manifest_size_is_bounded(tmp_path):
    archive = tmp_path / 'oversized.tar.gz'
    write_archive(archive, [('mpk.json', b' ' * (8 * 1024 * 1024 + 1))])
    with pytest.raises(ValueError, match='oversized'):
        registry.read_manifest(archive)


@pytest.mark.parametrize('name', ['mpk.json', 'nested/model_mpk.json', './nested/model_mpk.json'])
def test_both_authoritative_manifest_names_are_supported(tmp_path, name):
    archive = tmp_path / 'fixture.tar.gz'
    write_archive(archive, [(name, b'{"plugins":[]}'), ('bad_sidecar.json', b'not JSON')])
    assert registry.read_manifest(archive) == (name, b'{"plugins":[]}')


@pytest.mark.parametrize('change', [
    {'branch': '../wrong'}, {'staging': 'true'}, {'fixtures': []},
    {'model_id': '../wrong'}, {'variant': 'INT8;bad'}, {'run_id': 'latest'},
    {'sha256': 'bad'}, {'size_bytes': 0}, {'size_bytes': True},
    {'model_id': None}, {'sha256': None}, {'run_id': None},
])
def test_pins_must_be_explicit_safe_and_complete(fixture, change):
    source, content, pin, document, pins, root, cache = fixture
    if set(change) & {'branch', 'staging', 'fixtures'}:
        document.update(change)
    else:
        document['fixtures'][0].update(change)
    pins.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        registry.prepare(pins, root, cache, offline=True)
    assert not root.exists()


def test_duplicate_fixture_declarations_fail(fixture):
    source, content, pin, document, pins, root, cache = fixture
    document['fixtures'].append(copy.deepcopy(pin))
    pins.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='Duplicate'):
        registry.load_fixtures(pins)


def test_all_inputs_validate_before_publication_and_symlinks_fail(fixture):
    source, content, pin, document, pins, root, cache = fixture
    cached(fixture)
    second = {**pin, 'model_id': 'second', 'sha256': '0' * 64}
    document['fixtures'].append(second)
    pins.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='offline cache'):
        registry.prepare(pins, root, cache, offline=True)
    assert not list(root.glob('*-mpk.json'))
    document['fixtures'].pop()
    pins.write_text(json.dumps(document))
    (root / 'fixture_model-w8a8-mpk.json').symlink_to(source)
    with pytest.raises(ValueError, match='symlink'):
        registry.prepare(pins, root, cache, offline=True)
    assert source.read_bytes()[:2] == b'\x1f\x8b'


def test_stale_generated_manifests_cannot_hide_a_missing_declaration(fixture):
    source, content, pin, document, pins, root, cache = fixture
    cached(fixture)
    root.mkdir()
    stale = root / 'other_model-w8a8-mpk.json'
    stale.write_bytes(content)
    with pytest.raises(ValueError, match='Stale model fixtures'):
        registry.prepare(pins, root, cache, offline=True)
    assert stale.read_bytes() == content
    assert not (root / 'provenance.json').exists()


def run_preparation(arguments):
    return subprocess.run([sys.executable, str(SCRIPT), *arguments], capture_output=True, text=True)


def test_parallel_offline_preparation_and_relocation(fixture, tmp_path):
    source, content, pin, document, pins, root, cache = fixture
    cached(fixture)
    arguments = ['--fixtures', str(pins), '--root', str(root), '--cache', str(cache), '--offline']
    with ProcessPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run_preparation, [arguments, arguments]))
    assert [(r.returncode, r.stderr) for r in results] == [(0, ''), (0, '')]
    installed = tmp_path / 'installed/lib/sima-neat/test-assets/model-archive/registry/ssd-contract'
    shutil.copytree(root, installed, ignore=shutil.ignore_patterns('*.lock'))
    relocated = tmp_path / 'relocated'
    (tmp_path / 'installed').rename(relocated)
    shutil.rmtree(cache)
    shutil.rmtree(root)
    path = relocated / 'lib/sima-neat/test-assets/model-archive/registry/ssd-contract'
    assert (path / 'fixture_model-w8a8-mpk.json').read_bytes() == content
    assert json.loads((path / 'provenance.json').read_text())['fixtures'][0]['sha256'] == pin['sha256']
    assert not list(path.rglob('*.tar.gz')) and not list(path.rglob('*.elf'))


def test_actual_cmake_preparation_install_and_relocated_native_discovery(fixture, tmp_path):
    source, content, pin, document, pins, root, cache = fixture
    cmake = (ROOT / 'tests/CMakeLists.txt').read_text()
    start = cmake.index('  COMMAND ${Python3_EXECUTABLE}\n'
                        '    ${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare_registry_mpk_fixtures.py')
    build_command = cmake[start:cmake.index('  WORKING_DIRECTORY', start)]
    start = cmake.index('execute_process(\n  COMMAND \\"${Python3_EXECUTABLE}\\"\n'
                        '          \\"${CMAKE_CURRENT_SOURCE_DIR}/tools/prepare_registry_mpk_fixtures.py')
    end = cmake.index('endif()', start) + len('endif()')
    install_code = cmake[start:end]
    start = cmake.index('install(\n  DIRECTORY "${CMAKE_BINARY_DIR}/test-assets/model-archive/"')
    install_directory = cmake[start:cmake.index('\n)', start) + 3]
    source_dir, build = tmp_path / 'cmake-source', tmp_path / 'cmake-build'
    (source_dir / 'tools').mkdir(parents=True)
    (source_dir / 'tools/prepare_registry_mpk_fixtures.py').symlink_to(SCRIPT)
    (source_dir / 'assets/ssd-contract').mkdir(parents=True)
    shutil.copyfile(pins, source_dir / 'assets/ssd-contract/fixtures.json')
    (build / 'model-registry-cache').mkdir(parents=True)
    shutil.copyfile(source, build / 'model-registry-cache' / (pin['sha256'] + '.tar.gz'))
    # The producer's real commands and install rule, not a parallel path convention.
    (source_dir / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.16)
project(RegistryFixtureInstallation NONE)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(SIMA_TEST_MODEL_ARCHIVE_FIXTURE_ROOT "${CMAKE_BINARY_DIR}/test-assets/model-archive")
set(SIMA_TEST_SSD_REGISTRY_FIXTURE_ROOT "${SIMA_TEST_MODEL_ARCHIVE_FIXTURE_ROOT}/registry/ssd-contract")
set(SIMA_TEST_MODEL_REGISTRY_CACHE "${CMAKE_BINARY_DIR}/model-registry-cache")
add_custom_target(prepare ALL
''' + build_command + '\n  VERBATIM)\ninstall(CODE "\n' + install_code
                                            + '\n" COMPONENT extras)\n' + install_directory)
    subprocess.run(['cmake', '-S', str(source_dir), '-B', str(build)], check=True, capture_output=True)
    subprocess.run(['cmake', '--build', str(build)], check=True, capture_output=True)
    installed = tmp_path / 'installed'
    subprocess.run(['cmake', '--install', str(build), '--prefix', str(installed),
                    '--component', 'extras'], check=True, capture_output=True)
    native_root = installed / 'lib/sima-neat'
    runtime_manifest = native_root / 'test-fixtures/runtime_manifest.json'
    runtime_manifest.parent.mkdir(parents=True)
    producer = (ROOT / 'CMakeLists.txt').read_text()
    manifest = producer.split('set(SIMANEAT_TEST_RUNTIME_MANIFEST_INSTALLED_CONTENT [=[\n', 1)[1]
    runtime_manifest.write_text(manifest.split('\n]=])', 1)[0])
    probe_source = tmp_path / 'probe.cpp'
    probe_source.write_text('''#include "asset_utils.h"
#include <iostream>
int main() {
  const auto path = sima_test::test_model_archive_fixture_root_path() /
                    "registry/ssd-contract/fixture_model-w8a8-mpk.json";
  if (!sima_test::is_usable_regular_file(path)) return 1;
  std::ifstream input(path);
  std::cout << path << "\\n" << input.rdbuf();
}
''')
    binary = native_root / 'tests/probe'
    binary.parent.mkdir()
    subprocess.run(['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror',
                    '-I', str(ROOT / 'tests'), str(probe_source), '-o', str(binary)],
                   check=True, capture_output=True)
    description = subprocess.check_output(['file', str(binary)], text=True)
    architecture = {'x86_64': 'x86-64', 'amd64': 'x86-64',
                    'aarch64': 'ARM aarch64', 'arm64': 'ARM aarch64'}
    assert architecture.get(platform.machine().lower(), 'unknown native architecture') in description
    # Remove every acquisition/build input before running the installed probe.
    shutil.rmtree(build)
    shutil.rmtree(source_dir)
    relocated = tmp_path / 'relocated'
    installed.rename(relocated)
    binary = relocated / 'lib/sima-neat/tests/probe'
    result = subprocess.run([str(binary)], cwd=tmp_path, check=True, capture_output=True)
    expected = relocated / 'lib/sima-neat/test-assets/model-archive/registry/ssd-contract'
    assert result.stdout.splitlines()[0].strip(b'"') == str(expected / 'fixture_model-w8a8-mpk.json').encode()
    assert result.stdout.split(b'\n', 1)[1] == content
    assert (expected / 'fixture_model-w8a8-mpk.json').stat().st_mode & 0o444 == 0o444
    assert not list(expected.rglob('*.lock'))
    assert not list(relocated.rglob('*.tar.gz')) and not list(relocated.rglob('*.elf'))


def test_ssd_source_has_only_pins_and_native_assertions_are_preserved():
    path = ROOT / 'tests/assets/ssd-contract'
    pins = registry.load_fixtures(path / 'fixtures.json')
    assert {(f['model_id'], f['variant']) for f in pins['fixtures']} == {
        (f'ssd_mobilenet_v{version}', variant)
        for version in (1, 2, 3) for variant in ('w8a8', 'w16a16')}
    assert not list(path.glob('*-mpk.json'))
    source = (ROOT / 'tests/unit_testing/boxdecode_ssd_mpk_contract.cpp').read_text()
    assert 'test_model_archive_fixture_root_path()' in source
    assert 'test_shared_asset_path("tests/assets/ssd-contract/' not in source
    assert 'one-byte-short input no longer fails closed' in source
    assert 'kTv320Golden' in source and 'kTf300Golden' in source
    assert 'kernel input exceeds authored carrier span' in source
    assert "'models', 'download'" not in source
    workflow = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()
    assert 'tests/tools/test_registry_mpk_fixtures.py' in workflow
    cmake = (ROOT / 'tests/CMakeLists.txt').read_text()
    assert cmake.count('tools/prepare_registry_mpk_fixtures.py') == 2
    assert 'add_dependencies(unit_boxdecode_kernel_recipe_test prepare_test_fixtures)' in cmake
