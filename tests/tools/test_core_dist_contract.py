"""Small artifact-set fixtures; deep dependency receipt tests live elsewhere."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import copy
import re
import shutil
import subprocess
import tarfile
import venv
import zipfile
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('verify_core_dist', ROOT / 'tools/ci/verify_core_dist.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


@pytest.fixture
def cli_launcher_layout(tmp_path):
    tools = tmp_path / 'tools'; tools.mkdir()
    for name in ('dirname', 'readlink'):
        (tools / name).symlink_to(shutil.which(name))
    cache = tmp_path / 'portable cache'
    selected = cache / '.sima-cli/.venv'
    venv.EnvBuilder(with_pip=False).create(selected)
    entry = selected / 'bin/sima-cli'
    entry.write_text('#!/usr/bin/env python3\n')
    entry.chmod(0o755)
    launcher = cache / 'bin/sima-cli'; launcher.parent.mkdir()
    # The installed Vulcan cache launcher's path-relative interpreter binding.
    launcher.write_text('''#!/usr/bin/env bash
set -euo pipefail
wrapper_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cache_root="$(CDPATH= cd -- "${wrapper_dir}/.." && pwd)"
exec "${cache_root}/.sima-cli/.venv/bin/python" "${cache_root}/.sima-cli/.venv/bin/sima-cli" "$@"
''')
    launcher.chmod(0o755)
    env = {key: value for key, value in os.environ.items()
           if key not in ('SIMA_CLI_BIN', 'SIMA_CLI_PYTHON', 'PYTHONPATH', 'PYTHONHOME')}
    env['PATH'] = str(tools)
    return selected, entry, launcher, env


def resolve_cli_python(layout, **overrides):
    _, _, launcher, env = layout
    # Run the actual resolver with a relocated cache path; never create or
    # mutate the real /mnt/vulcan installation in a host test.
    script = (ROOT / 'scripts/ci/verify_core_dist.sh').read_text()
    assert '${1:-/mnt/vulcan/bin/sima-cli}' in script
    resolver = script.split('\nroot=', 1)[0]
    return subprocess.run(['/bin/bash', '-c', resolver + '\nresolve_sima_cli_python "$1"',
                           'test', str(launcher)], env={**env, **overrides}, text=True, capture_output=True)


@pytest.mark.parametrize('selection', ['venv', 'path', 'symlink', 'portable', 'portable_path', 'fallback'])
def test_verifier_selects_python_from_installed_cli_layout(cli_launcher_layout, selection, tmp_path):
    selected, entry, launcher, env = cli_launcher_layout
    if selection == 'symlink':
        alias = tmp_path / 'sima-cli'; alias.symlink_to(entry)
        overrides = {'SIMA_CLI_BIN': str(alias)}
    elif selection in ('path', 'portable_path'):
        directory = entry.parent if selection == 'path' else launcher.parent
        overrides = {'PATH': str(directory) + os.pathsep + env['PATH']}
    elif selection == 'fallback':
        overrides = {}
    else:
        overrides = {'SIMA_CLI_BIN': str(entry if selection == 'venv' else launcher)}
    result = resolve_cli_python(cli_launcher_layout, **overrides)
    assert result.returncode == 0, result.stderr
    python = Path(result.stdout.strip())
    assert python.samefile(selected / 'bin/python')
    # Preserve venv identity, not just the underlying system executable.
    probe = subprocess.run([str(python), '-c', 'import sys; print(sys.prefix)'],
                           env=env, text=True, capture_output=True, check=True)
    assert Path(probe.stdout.strip()).resolve() == selected.resolve()
    if selection != 'fallback':
        help_result = subprocess.run(['/bin/bash', str(ROOT / 'scripts/ci/verify_core_dist.sh'), '--help'],
                                     env={**env, **overrides}, text=True, capture_output=True)
        assert help_result.returncode == 0, help_result.stderr
        assert '--expected-receipt' in help_result.stdout


def test_verifier_honors_explicit_python_without_cli(cli_launcher_layout):
    selected, _, _, env = cli_launcher_layout
    overrides = {'SIMA_CLI_PYTHON': str(selected / 'bin/python'), 'SIMA_CLI_BIN': '/missing/cli'}
    result = resolve_cli_python(cli_launcher_layout, **overrides)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(selected / 'bin/python')
    result = subprocess.run(['/bin/bash', str(ROOT / 'scripts/ci/verify_core_dist.sh'), '--invalid-option'],
                            env={**env, **overrides}, text=True, capture_output=True)
    assert result.returncode == 2  # The real verifier's argparse failure is not masked.
    assert 'unrecognized arguments' in result.stderr


@pytest.mark.parametrize('override', ['SIMA_CLI_BIN', 'SIMA_CLI_PYTHON'])
def test_verifier_missing_explicit_selection_never_falls_back(cli_launcher_layout, override):
    result = resolve_cli_python(cli_launcher_layout, **{override: '/missing/selected'})
    assert result.returncode != 0
    assert f'explicit {override}' in result.stderr
    assert not result.stdout


@pytest.mark.parametrize('missing', ['pyvenv.cfg', 'bin/python', 'bin/sima-cli'])
def test_verifier_rejects_incomplete_portable_cli(cli_launcher_layout, missing):
    selected, _, launcher, _ = cli_launcher_layout
    (selected / missing).unlink()
    result = resolve_cli_python(cli_launcher_layout, SIMA_CLI_BIN=str(launcher))
    assert result.returncode != 0
    assert 'Python belonging to selected sima-cli' in result.stderr
    assert not result.stdout


def test_verifier_missing_cli_has_actionable_error(cli_launcher_layout):
    _, _, launcher, _ = cli_launcher_layout
    launcher.unlink()
    result = resolve_cli_python(cli_launcher_layout)
    assert result.returncode != 0
    assert 'Cannot locate sima-cli on PATH' in result.stderr
    assert not result.stdout


def test_verifier_unknown_explicit_layout_never_uses_other_cache(cli_launcher_layout, tmp_path):
    unrelated = tmp_path / 'other-cli'
    unrelated.write_text('#!/bin/sh\nexit 0\n'); unrelated.chmod(0o755)
    result = resolve_cli_python(cli_launcher_layout, SIMA_CLI_BIN=str(unrelated))
    assert result.returncode != 0
    assert 'Python belonging to selected sima-cli' in result.stderr
    assert not result.stdout


def test_verifier_missing_selected_schema_does_not_use_ambient_python(cli_launcher_layout, tmp_path):
    _, entry, _, env = cli_launcher_layout
    result = subprocess.run(['/bin/bash', str(ROOT / 'scripts/ci/verify_core_dist.sh'), '--dist', str(tmp_path)],
                            env={**env, 'SIMA_CLI_BIN': str(entry)}, text=True, capture_output=True)
    assert result.returncode != 0
    assert "No module named 'sima_cli'" in result.stderr


def fixture(root, metadata_name='sima-neat'):
    debs = {
        'sima-neat-1-Linux-core.deb': 'sima-neat',
        'sima-neat-1-Linux-dev.deb': 'sima-neat-dev',
        'kernels.deb': 'simaai-kernels-dev',
        'runner.deb': 'simaai-runner',
        'scheduler.deb': 'simaai-scheduler',
        'runtime-compat.deb': 'simaai-runtime',
    }
    for name in [*debs, 'pyneat-1-py3-none-any.whl', 'install_neat_framework.sh', 'install_sentinel.sh', 'scheduler_platform_admission.py']:
        (root / name).write_bytes(name.encode())
    with zipfile.ZipFile(root / 'pyneat-1-py3-none-any.whl', 'w') as archive:
        archive.writestr('pyneat-1.dist-info/METADATA', 'Name: pyneat\nVersion: 1\n')
        archive.writestr('pyneat/__init__.py', '# public namespace')
    manifest = {'platform-version': '3.0.0', 'modelzoo-version': '2.1.3', 'abi-version': '4'}
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'resolved-deps-manifest.json').write_text(json.dumps({
        'kernels': {'debian_version': '1'}, 'runner': {'debian_version': '1'},
        'scheduler': {'debian_version': '1'}, 'runtime_compat': {'debian_version': '1'}}))
    (root / 'neat-install-manifest.txt').write_text('\n'.join([*debs, 'pyneat-1-py3-none-any.whl']) + '\n')
    extras = 'sima-neat-1-Linux-extras.tar.gz'
    with tarfile.open(root / extras, 'w:gz') as archive:
        for tree in ('tests', 'tutorials'):
            member = tarfile.TarInfo('./lib/sima-neat/' + tree + '/CTestTestfile.cmake')
            data = b'add_test(fixture /usr/bin/true)\n'
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    base = [*debs, 'pyneat-1-py3-none-any.whl', 'manifest.json', 'resolved-deps-manifest.json',
            'neat-install-manifest.txt', 'install_neat_framework.sh', 'install_sentinel.sh', 'scheduler_platform_admission.py']
    for name, resources, selectables in (
        ('metadata.json', base, [extras]), ('metadata-minimal.json', base, []),
        ('metadata-all.json', base + [extras], []), ('metadata-pyneat.json', ['pyneat-1-py3-none-any.whl'], []),
        ('metadata-extras.json', [extras], []),
    ):
        metadata = {'name': metadata_name, 'version': '1', 'release': '',
                    'platforms': [{'type': 'board', 'version': '3.0.0', 'compatible_with': ['modalix']},
                                  {'type': 'palette', 'version': '3.0.0'}],
                    'resources': resources,
                    'resources-checksum': {p: verifier.digest(root / p) for p in resources},
                    'selectable-resources': [{'resource': p, 'checksum': verifier.digest(root / p)} for p in selectables],
                    'installation': {'script': ':' if name in ('metadata-pyneat.json', 'metadata-extras.json') else './install_neat_framework.sh'}}
        (root / name).write_text(json.dumps(metadata))
    return lambda path: {'Package': debs[Path(path).name], 'Version': '1', 'Architecture': 'arm64'}


def change(root, file, mutate):
    path = root / file
    value = json.loads(path.read_text())
    mutate(value)
    path.write_text(json.dumps(value))


def run(root, **kwargs):
    # Artifact-shape fixtures inject only the producer-owned description. Real
    # DEB extraction/authentication/delegation is exercised independently below.
    description = getattr(verifier.control, 'fixture_description', {
        'schema_version': 1, 'redistributed_provider_controls': [], 'platform_packages': {}})
    with patch.object(verifier, 'scheduler_platform_description', return_value=description) as describe:
        result = verifier.verify(root, validate_metadata=lambda _: None, **kwargs)
        describe.assert_called_once_with(root)
        return result


@pytest.mark.parametrize('metadata_name', ['sima-neat', 'gh:sima-neat/core'])
def test_golden_set_is_read_only(tmp_path, monkeypatch, metadata_name):
    monkeypatch.setenv('NEAT_PACKAGE_NAME', 'gh:foreign/core')
    controls = fixture(tmp_path, metadata_name=metadata_name)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    with patch.object(verifier, 'control', controls):
        assert run(tmp_path)['status'] == 'PASS'
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}


@pytest.mark.parametrize('name', ['foreign', 'gh:foreign/core', 'gh:sima-neat/scheduler'])
def test_metadata_rejects_uniform_foreign_identity_even_from_environment(tmp_path, monkeypatch, name):
    monkeypatch.setenv('NEAT_PACKAGE_NAME', name)
    controls = fixture(tmp_path, metadata_name=name)
    with patch.object(verifier, 'control', controls), pytest.raises(verifier.ContractError) as error:
        run(tmp_path)
    assert 'Metadata variants disagree with Core package identity' in str(error.value)
    assert repr((name, '1')) in str(error.value)


@pytest.mark.parametrize('original,other', [
    ('sima-neat', 'gh:sima-neat/core'), ('gh:sima-neat/core', 'sima-neat')])
@pytest.mark.parametrize('filename', [
    'metadata.json', 'metadata-minimal.json', 'metadata-all.json',
    'metadata-pyneat.json', 'metadata-extras.json'])
def test_metadata_rejects_mixed_allowed_aliases(tmp_path, original, other, filename):
    controls = fixture(tmp_path, metadata_name=original)
    change(tmp_path, filename, lambda value: value.update(name=other))
    with patch.object(verifier, 'control', controls), pytest.raises(
            verifier.ContractError, match='Metadata variants disagree with Core package identity'):
        run(tmp_path)


@pytest.mark.parametrize('metadata_name', ['sima-neat', 'gh:sima-neat/core'])
@pytest.mark.parametrize('uniform', [False, True])
def test_metadata_version_must_match_actual_deb(tmp_path, metadata_name, uniform):
    controls = fixture(tmp_path, metadata_name=metadata_name)
    targets = list(tmp_path.glob('metadata*.json')) if uniform else [tmp_path / 'metadata-pyneat.json']
    for path in targets:
        change(tmp_path, path.name, lambda value: value.update(version='2'))
    with patch.object(verifier, 'control', controls), pytest.raises(verifier.ContractError) as error:
        run(tmp_path)
    assert 'Metadata variants disagree with Core package identity' in str(error.value)
    assert repr((metadata_name, '2')) in str(error.value)
    assert repr((metadata_name, '1')) in str(error.value)


@pytest.mark.parametrize('mutation', [
    lambda r: (r / 'manifest.json').write_text('{}'),
    lambda r: (r / 'runner.deb').unlink(),
    lambda r: (r / 'install_sentinel.sh').unlink(),
    lambda r: (r / 'scheduler_platform_admission.py').unlink(),
    lambda r: (r / 'install_sentinel.sh').write_bytes(b'changed installer'),
    lambda r: (r / 'scheduler.deb').unlink(),
    lambda r: (r / 'runtime-compat.deb').unlink(),
    lambda r: (r / 'pyneat-1-py3-none-any.whl').write_bytes(b'corrupt'),
    lambda r: change(r, 'metadata.json', lambda v: v['resources'].append(v['resources'][0])),
    lambda r: change(r, 'metadata.json', lambda v: v['resources-checksum'].pop('pyneat-1-py3-none-any.whl')),
    lambda r: change(r, 'metadata.json', lambda v: v['resources'].append('../escape')),
    lambda r: (r / 'neat-install-manifest.txt').write_text('foreign.deb\n'),
    lambda r: change(r, 'metadata-pyneat.json', lambda v: v['resources'].append('runner.deb')),
    lambda r: change(r, 'metadata.json', lambda v: v['platforms'].clear()),
    lambda r: change(r, 'metadata-minimal.json', lambda v: v.update(version='2')),
    lambda r: change(r, 'metadata.json', lambda v: v['selectable-resources'][0].update(checksum='0' * 64)),
])
def test_rejects_invalid_set(tmp_path, mutation):
    controls = fixture(tmp_path)
    mutation(tmp_path)
    with patch.object(verifier, 'control', controls), pytest.raises((verifier.ContractError, zipfile.BadZipFile)):
        run(tmp_path)


def test_expected_receipt_and_dependency_are_explicit(tmp_path):
    root = tmp_path / 'dist'; root.mkdir()
    controls = fixture(root)
    expected = tmp_path / 'expected'; expected.mkdir()
    (expected / 'runner.deb').write_bytes(b'changed producer bytes')
    (expected / 'receipt.json').write_text('{}')
    with patch.object(verifier, 'control', controls):
        with pytest.raises(verifier.ContractError, match='receipt differs'):
            run(root, expected_receipt=expected / 'receipt.json')
        with pytest.raises(verifier.ContractError, match='dependency differs'):
            run(root, expected_deb_dirs=[expected])


def test_missing_extras_tree(tmp_path):
    controls = fixture(tmp_path)
    extras = tmp_path / 'sima-neat-1-Linux-extras.tar.gz'
    with tarfile.open(extras, 'w:gz'):
        pass
    for name in ('metadata.json', 'metadata-all.json', 'metadata-extras.json'):
        def update(value):
            if extras.name in value['resources-checksum']:
                value['resources-checksum'][extras.name] = verifier.digest(extras)
            for entry in value['selectable-resources']:
                entry['checksum'] = verifier.digest(extras)
        change(tmp_path, name, update)
    with patch.object(verifier, 'control', controls), pytest.raises(verifier.ContractError, match='CTest tree'):
        run(tmp_path)


@pytest.mark.parametrize('name', [
    'lib/sima-neat/tutorials/CMakeFiles/tutorial.dir/example.cpp.o',
    'lib/sima-neat/tutorials/CMakeFiles',
    'lib/sima-neat/tutorials/CMakeCache.txt',
    'lib/sima-neat/tutorials/cmake_install.cmake',
    'lib/sima-neat/tests/fixture.obj',
])
def test_extras_rejects_generated_build_internals(tmp_path, name):
    controls = fixture(tmp_path)
    extras = tmp_path / 'sima-neat-1-Linux-extras.tar.gz'
    with tarfile.open(extras, 'w:gz') as archive:
        for member_name in ('lib/sima-neat/tests/CTestTestfile.cmake',
                            'lib/sima-neat/tutorials/CTestTestfile.cmake', name):
            member = tarfile.TarInfo(member_name)
            archive.addfile(member, io.BytesIO())
    for metadata in ('metadata.json', 'metadata-all.json', 'metadata-extras.json'):
        def update(value):
            if extras.name in value['resources-checksum']:
                value['resources-checksum'][extras.name] = verifier.digest(extras)
            for entry in value['selectable-resources']:
                entry['checksum'] = verifier.digest(extras)
        change(tmp_path, metadata, update)
    with patch.object(verifier, 'control', controls), pytest.raises(
            verifier.ContractError, match='generated build internals'):
        run(tmp_path)


def test_actual_tutorial_install_rules_preserve_targets_without_build_internals(tmp_path):
    """Exercise the real CMake rules, with tiny native sources, never run tutorials."""
    source = tmp_path / 'source'; source.mkdir()
    tutorials = source / 'tutorials'; tutorials.mkdir()
    rules = (ROOT / 'tutorials/CMakeLists.txt').read_text()
    (tutorials / 'CMakeLists.txt').write_text(rules)
    targets = dict(re.findall(r'add_tutorial\((\w+)\s+(\S+\.cpp)', rules))
    assert targets
    for relative in targets.values():
        path = tutorials / relative; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('extern "C" int dependency(); int main() { return dependency(); }\n')
    for relative in ('cmake/NeatGuardedTests.cmake', 'tools/hardware_test_supervisor.py',
                     'tutorials/build.sh', 'tutorials/run_with_modelzoo.sh'):
        path = source / relative; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, path)
    (tutorials / 'fixture.jpg').write_bytes(b'asset-preservation-control')
    (source / 'dependency.cpp').write_text('extern "C" int dependency() { return 0; }\n')
    (source / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.16)
project(TutorialInstallContract LANGUAGES CXX)
enable_testing()
add_library(sima_neat SHARED dependency.cpp)
add_library(opencv_videoio INTERFACE)
set(CMAKE_INSTALL_RPATH "$ORIGIN/../runtime")
add_subdirectory(tutorials)
''')
    env = os.environ.copy()
    for key in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'CPATH',
                'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                'CMAKE_TOOLCHAIN_FILE', 'CMAKE_PREFIX_PATH', 'DESTDIR'):
        env.pop(key, None)
    build = tmp_path / 'build'; installed = tmp_path / 'installed'
    def command(*args):
        return subprocess.check_output(list(map(str, args)), env=env, text=True,
                                       stderr=subprocess.STDOUT)
    command('cmake', '-S', source, '-B', build, '-DCMAKE_CXX_COMPILER=/usr/bin/c++',
            '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_INSTALL_PREFIX=' + str(installed))
    command('cmake', '--build', build, '--parallel', '4')
    command('cmake', '--install', build, '--component', 'extras')
    runtime = installed / 'lib/sima-neat/tutorials'
    assert {p.name for p in runtime.iterdir()} == set(targets) | {'CTestTestfile.cmake'}
    assert (installed / 'share/sima-neat/tutorials/fixture.jpg').read_bytes() == b'asset-preservation-control'
    for target in targets:
        path = runtime / target
        assert path.read_bytes().startswith(b'\x7fELF')
        assert os.access(path, os.X_OK)
        dynamic = command('readelf', '-d', path)
        assert '$ORIGIN/../runtime' in dynamic
        assert str(tmp_path) not in dynamic
    assert not any(p.name in {'CMakeFiles', 'CMakeCache.txt', 'cmake_install.cmake'}
                   or p.suffix in {'.o', '.obj'} for p in installed.rglob('*'))


def test_raising_cli_validator_is_called(tmp_path):
    controls = fixture(tmp_path)
    def reject(_):
        raise RuntimeError('CLI-owned schema rejection')
    with patch.object(verifier, 'control', controls), patch.object(
            verifier, 'scheduler_platform_description', return_value={
                'schema_version': 1, 'platform_packages': {}, 'redistributed_provider_controls': []}), \
            pytest.raises(RuntimeError, match='CLI-owned'):
        verifier.verify(tmp_path, validate_metadata=reject)


def test_both_entrypoints_share_verifier_without_postseal_overwrite():
    build = (ROOT / 'build.sh').read_text()
    workflow = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()
    assert 'scripts/ci/verify_core_dist.sh' in build
    assert 'scripts/ci/verify_core_dist.sh' in workflow
    assert 'sudo install -m 0644 build/resolved_manifest.json dist/resolved-deps-manifest.json' not in workflow
    assert 'cmp deps/internals-manifest.json' not in workflow


def test_public_wheel_cannot_vendor_runtime(tmp_path):
    controls = fixture(tmp_path)
    with zipfile.ZipFile(tmp_path / 'pyneat-1-py3-none-any.whl', 'a') as archive:
        archive.writestr('pyneat/libsima_runtime.so', b'foreign runtime')
    with patch.object(verifier, 'control', controls), pytest.raises(verifier.ContractError, match='Runtime provider'):
        run(tmp_path)


def test_compatibility_projection_uses_selected_input(tmp_path):
    root = tmp_path / 'dist'; root.mkdir()
    controls = fixture(root)
    expected = tmp_path / 'manifest.json'
    expected.write_text(json.dumps({'platform-version': '2.1.0', 'modelzoo-version': '2.1.3', 'abi-version': '4'}))
    with patch.object(verifier, 'control', controls), pytest.raises(verifier.ContractError, match='projection differs'):
        run(root, expected_manifest=expected)


def provider_fixture(tmp_path, architecture='arm64'):
    """Real small provider DEBs plus the existing artifact-set control doubles."""
    root = tmp_path / 'dist'; root.mkdir()
    base_controls = fixture(root)
    actual_control = verifier.control

    def package(name, arch='arm64', version='1'):
        source = tmp_path / ('producer-' + name)
        (source / 'DEBIAN').mkdir(parents=True, exist_ok=True)
        (source / 'DEBIAN/control').write_text(
            f'Package: {name}\nVersion: {version}\nArchitecture: {arch}\n'
            'Maintainer: Fixture <fixture@example.invalid>\nDescription: provider control fixture\n')
        payload = source / 'usr/share' / name
        payload.mkdir(parents=True, exist_ok=True)
        (payload / 'identity').write_text(name + ':' + version)
        target = root / (name + '.deb')
        subprocess.run(['dpkg-deb', '--build', '--root-owner-group', str(source), str(target)],
                       check=True, capture_output=True)
        return {**actual_control(target), 'deb_sha256': verifier.digest(target)}

    backend = package('fixture-backend', architecture)
    draw = package('fixture-renderer')
    draw_dev = package('fixture-renderer-dev')
    provider = {'component': 'boxrender-provider', 'packages': {'runtime': draw, 'development': draw_dev}}
    receipt_path = root / 'resolved-deps-manifest.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['runner']['buildinfo'] = {
        'external_dependencies': {'ArbitraryBackend': {'loader_name': 'fixture.so', 'debian_provider': backend}},
        'debian_dependencies': [backend]}
    receipt['scheduler']['buildinfo'] = {
        'external_dependencies': {'BoxRender': provider}, 'debian_dependencies': [draw]}
    receipt['external_dependencies'] = {
        'RunnerBackends': {'ArbitraryBackend': {'kind': 'external-platform-runtime',
            'archive': '/must/not/read/this/source/path.deb', 'debian_provider': copy.deepcopy(backend)}},
        'BoxRender': {'kind': 'external-platform-package', 'package': '/must/not/read/this/prefix',
                      'buildinfo': copy.deepcopy(provider)}}
    receipt_path.write_text(json.dumps(receipt))
    refresh_provider_metadata(root)

    def controls(path):
        return actual_control(path) if Path(path).name.startswith('fixture-') else base_controls(path)
    controls.fixture_description = {'schema_version': 1, 'platform_packages': {},
                                    'redistributed_provider_controls': [backend, draw, draw_dev]}
    return root, controls, package


def refresh_provider_metadata(root):
    providers = sorted(p.name for p in root.glob('fixture-*.deb'))
    install = root / 'neat-install-manifest.txt'
    entries = set(install.read_text().splitlines()) | set(providers)
    install.write_text('\n'.join(sorted(entries)) + '\n')
    for name in ('metadata.json', 'metadata-minimal.json', 'metadata-all.json'):
        def update(value):
            value['resources'] = sorted(set(value['resources']) | set(providers))
            value['resources-checksum'] = {p: verifier.digest(root / p) for p in value['resources']}
        change(root, name, update)


@pytest.mark.parametrize('architecture', ['arm64', 'all'])
def test_consumed_provider_archives_are_admitted_read_only(tmp_path, architecture):
    root, controls, _ = provider_fixture(tmp_path, architecture)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}
    with patch.object(verifier, 'control', controls):
        result = run(root)
    assert {'fixture-backend', 'fixture-renderer', 'fixture-renderer-dev'} <= set(result['dependency_roles'])
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in root.iterdir()}


@pytest.mark.parametrize('mutation,diagnostic', [
    (lambda r, p: (r / 'fixture-backend.deb').unlink(), 'provider control'),
    (lambda r, p: (r / 'fixture-renderer-dev.deb').unlink(), 'provider control'),
    (lambda r, p: p('fixture-backend', version='2'), 'provider control'),
    (lambda r, p: p('fixture-backend', arch='amd64'), 'architecture'),
    (lambda r, p: (r / 'fixture-backend.deb').write_bytes((r / 'fixture-backend.deb').read_bytes() + b'changed'), 'archive differs'),
    (lambda r, p: p('fixture-unrecorded'), 'Foreign DEB'),
    (lambda r, p: shutil.copy2(r / 'fixture-backend.deb', r / 'fixture-duplicate.deb'), 'Duplicate DEB package role'),
    (lambda r, p: (r / 'fixture-renderer.deb').write_bytes((r / 'fixture-renderer.deb').read_bytes() + b'changed'), 'archive differs'),
])
def test_rejects_unconsumed_or_changed_provider_archives(tmp_path, mutation, diagnostic):
    root, controls, package = provider_fixture(tmp_path)
    mutation(root, package)
    with patch.object(verifier, 'control', controls), pytest.raises(verifier.ContractError, match=diagnostic):
        run(root)


def inventory_fixture(tmp_path, profile='core'):
    root = tmp_path / 'packages'; root.mkdir()
    for name in verifier.PACKAGE_PROFILES[profile]:
        source = tmp_path / ('source-' + name)
        (source / 'DEBIAN').mkdir(parents=True)
        (source / 'DEBIAN/control').write_text(
            f'Package: {name}\nVersion: 1\nArchitecture: arm64\n'
            'Maintainer: Fixture <fixture@example.invalid>\nDescription: inventory fixture\n')
        payload = source / 'usr/share' / name
        payload.mkdir(parents=True)
        (payload / 'identity').write_text(name + ':1')
        (payload / 'identity-link').symlink_to('identity')
        subprocess.run(['dpkg-deb', '--build', '--root-owner-group', str(source), str(root / (name + '.deb'))],
                       check=True, capture_output=True)
    return root


@pytest.mark.parametrize('profile', ['core', 'core-codecs', 'core-codecs-llima'])
def test_selected_profile_inventory_is_expected_data_from_real_debs(tmp_path, profile):
    root = inventory_fixture(tmp_path, profile)
    inventory = verifier.package_inventory(root, profile=profile)
    assert set(inventory['packages']) == verifier.PACKAGE_PROFILES[profile]
    assert inventory['scope'] == 'expected-installed-packages'
    assert 'status' not in inventory
    assert inventory['payload']['/usr/share/simaai-runner/identity']['sha256'] == hashlib.sha256(b'simaai-runner:1').hexdigest()
    assert inventory['payload']['/usr/share/simaai-runner/identity-link']['symlink'] == 'identity'
    if profile != 'core-codecs-llima':
        assert 'sima-lmm-core' not in inventory['packages']
    assert 'simaai-memory-lib' not in inventory['packages']


def test_profile_cannot_be_satisfied_by_missing_roles(tmp_path):
    root = inventory_fixture(tmp_path)
    with pytest.raises(verifier.ContractError, match='required profile roles'):
        verifier.package_inventory(root, profile='core-codecs')
    (root / 'simaai-runner.deb').unlink()
    with pytest.raises(verifier.ContractError, match='required profile roles'):
        verifier.package_inventory(root, profile='core')


def test_conflicting_archive_ownership_rejected(tmp_path):
    root = inventory_fixture(tmp_path)
    original = verifier.deb_payload
    def collision(path):
        return {**original(path), '/usr/lib/duplicated.so': {'sha256': '0' * 64}}
    with patch.object(verifier, 'deb_payload', collision), pytest.raises(verifier.ContractError, match='Conflicting payload ownership'):
        verifier.package_inventory(root, profile='core')


def fake_installed_fixture(tmp_path):
    # Files are real and independent of the mocked dpkg observation. No package
    # is installed and this unit fixture is not an installed qualification pass.
    names = verifier.PACKAGE_PROFILES['core']
    packages = {name: {'Package': name, 'Version': '1', 'Architecture': 'arm64'} for name in names}
    actual = {name: {**entry, 'Status': 'install ok installed'} for name, entry in packages.items()}
    path = tmp_path / 'selected.so'; path.write_bytes(b'expected binary bytes')
    link = tmp_path / 'selected-link.so'; link.symlink_to(path.name)
    inventory = {'schema_version': 1, 'scope': 'expected-installed-packages', 'profile': 'core',
                 'packages': packages, 'payload': {
                     str(path): {'package': 'simaai-runner', 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()},
                     str(link): {'package': 'simaai-runner', 'symlink': path.name}}}
    owners = {str(path): 'simaai-runner', str(link): 'simaai-runner'}
    return inventory, actual, owners, path, link


def test_installed_check_reads_files_and_dpkg_not_a_receipt_pass(tmp_path):
    inventory, actual, owners, path, link = fake_installed_fixture(tmp_path)
    with patch.object(verifier, 'installed_controls', return_value=actual), patch.object(verifier, 'installed_owners', return_value=owners):
        assert verifier.verify_installed(inventory)['scope'] == 'actual-dpkg-and-payload'
        path.write_bytes(b'altered after package selection')
        with pytest.raises(verifier.ContractError, match='Installed payload differs'):
            verifier.verify_installed(inventory)


@pytest.mark.parametrize('mutation,diagnostic', [
    (lambda i, a, o, p, l: a['simaai-runner'].update(Version='2'), 'identity differs'),
    (lambda i, a, o, p, l: a['simaai-runner'].update(Architecture='amd64'), 'identity differs'),
    (lambda i, a, o, p, l: a['simaai-runner'].update(Status='deinstall ok config-files'), 'identity differs'),
    (lambda i, a, o, p, l: a.pop('simaai-runner'), 'identity differs'),
    (lambda i, a, o, p, l: i['packages'].pop('simaai-runner'), 'required profile roles'),
    (lambda i, a, o, p, l: o.update({str(p): 'other-provider'}), 'owner differs'),
    (lambda i, a, o, p, l: p.unlink(), 'payload differs'),
    (lambda i, a, o, p, l: (l.unlink(), l.symlink_to('other.so')), 'symlink differs'),
    (lambda i, a, o, p, l: (p.unlink(), p.symlink_to(l)), 'payload differs'),
])
def test_installed_failures_cannot_be_hidden_by_expected_inventory(tmp_path, mutation, diagnostic):
    inventory, actual, owners, path, link = fake_installed_fixture(tmp_path)
    mutation(inventory, actual, owners, path, link)
    with patch.object(verifier, 'installed_controls', return_value=actual), patch.object(verifier, 'installed_owners', return_value=owners), pytest.raises(verifier.ContractError, match=diagnostic):
        verifier.verify_installed(inventory)


def test_dpkg_query_parses_actual_status_and_multiarch_owners():
    with patch.object(verifier.subprocess, 'check_output', return_value='simaai-runner\t1\tarm64\tinstall ok installed\n') as command:
        assert verifier.installed_controls(['simaai-runner'])['simaai-runner']['Architecture'] == 'arm64'
        assert command.call_args.args[0][0] == 'dpkg-query'
    with patch.object(verifier.subprocess, 'check_output', return_value='simaai-runner:arm64: /usr/lib/example.so\n'):
        assert verifier.installed_owners(['/usr/lib/example.so']) == {'/usr/lib/example.so': 'simaai-runner'}
    with patch.object(verifier.subprocess, 'check_output', side_effect=subprocess.CalledProcessError(1, ['dpkg-query'])):
        with pytest.raises(verifier.ContractError, match='private prefixes do not qualify'):
            verifier.installed_controls(['simaai-runner'])
    with patch.object(verifier.subprocess, 'check_output', return_value='owner1, owner2: /usr/lib/example.so\n'):
        with pytest.raises(verifier.ContractError, match='Conflicting'):
            verifier.installed_owners(['/usr/lib/example.so'])


def test_platform_preservation_is_separate_and_checks_ota():
    before = {'schema_version': 2, 'scope': 'native-platform-before-install',
              'packages': {name: {'Package': name, 'Version': '1', 'Architecture': 'arm64',
                                  'Status': 'install ok installed'} for name in verifier.NATIVE_PLATFORM_ROLES},
              'absent_packages': [],
              'ota_present': True, 'ota_path': '/opt/sima/bin/simaai-ota', 'ota_sha256': 'a' * 64, 'ota_owner': 'simaai-palette-modalix'}
    current = copy.deepcopy(before)
    with patch.object(verifier, 'platform_inventory', return_value=current):
        assert verifier.verify_platform(before, {})['status'] == 'PASS'
        current['packages']['simaai-mlart-modalix']['Version'] = '2'
        with pytest.raises(verifier.ContractError, match='Unselected platform package changed'):
            verifier.verify_platform(before, {})
        # Explicit selected provider upgrade is covered by verify_installed.
        verifier.verify_platform(before, {'simaai-mlart-modalix': {}})
        current['ota_sha256'] = 'b' * 64
        with pytest.raises(verifier.ContractError, match='OTA command changed'):
            verifier.verify_platform(before, {'simaai-mlart-modalix': {}})


def test_workflow_uses_actual_installed_verification_not_prefix_pass():
    workflow = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()
    assert workflow.index('--capture-platform-baseline') < workflow.index('- name: Install test runtime from Vulcan artifact')
    assert '--verify-installed --platform-baseline' in workflow
    assert 'export SIMA_TEST_PACKAGE_INVENTORY=' in workflow
    assert 'export SIMA_TEST_PLATFORM_BASELINE="${GITHUB_WORKSPACE}/_work/platform-before-install.json"' in workflow
    assert workflow.index('--capture-platform-baseline') < workflow.index('export SIMA_TEST_PLATFORM_BASELINE=')
    assert '--package-profile core-codecs-llima' in workflow
    native = (ROOT / 'tests/unit_testing/unit_000_devkit_package_inventory_test.cpp').read_text()
    assert '{"simaai-memory-lib", "native-sima"}' not in native
    assert 'SIMA_TEST_PACKAGE_INVENTORY' in native
    assert 'sima_neat_version()' in native


def test_installed_workflow_verifies_complete_producer_not_cli_resource_workspace(tmp_path):
    """Exercise the real verifier against both acquisition layouts.

    Remote metadata installation downloads metadata.resources, not the remote
    metadata documents themselves. The complete build artifact must therefore
    be an independent workflow input. Real small DEBs keep this test at the
    filesystem/control boundary rather than only asserting YAML text.
    """
    import shlex
    import yaml
    producer = tmp_path / 'producer'; producer.mkdir()
    original_controls = fixture(producer)
    packages_root = tmp_path / 'real-packages'; packages_root.mkdir()
    packages = inventory_fixture(packages_root, profile='core-codecs-llima')
    for deb in producer.glob('*.deb'):
        package_name = original_controls(deb)['Package']
        shutil.copyfile(packages / (package_name + '.deb'), deb)
    extra_roles = verifier.PACKAGE_PROFILES['core-codecs-llima'] - verifier.PACKAGE_PROFILES['core']
    extra_files = sorted(name + '.deb' for name in extra_roles)
    for filename in extra_files:
        shutil.copyfile(packages / filename, producer / filename)
    receipt_path = producer / 'resolved-deps-manifest.json'
    receipt = json.loads(receipt_path.read_text())
    receipt['external_dependencies'] = {'SimaCodecs': {'debian_packages': [
        {**verifier.control(producer / (name + '.deb')), 'sha256': verifier.digest(producer / (name + '.deb'))}
        for name in ('neat-codecs', 'neat-codecs-dev')]}}
    receipt_path.write_text(json.dumps(receipt))
    allowlist = producer / 'neat-install-manifest.txt'
    allowlist.write_text(allowlist.read_text() + '\n'.join(extra_files) + '\n')
    for metadata in producer.glob('metadata*.json'):
        value = json.loads(metadata.read_text())
        if metadata.name in ('metadata.json', 'metadata-minimal.json', 'metadata-all.json'):
            value['resources'].extend(extra_files)
        value['resources-checksum'] = {name: verifier.digest(producer / name) for name in value['resources']}
        metadata.write_text(json.dumps(value))
    workspace = tmp_path / 'workspace'; workspace.mkdir()
    # This is the CLI's selected all-variant resource layout: full data payload,
    # but no surviving metadata*.json from its TemporaryDirectory.
    selected = json.loads((producer / 'metadata-all.json').read_text())
    for filename in selected['resources']:
        shutil.copyfile(producer / filename, workspace / filename)
    assert not list(workspace.glob('metadata*.json'))
    with pytest.raises(verifier.ContractError, match='metadata.json'):
        run(workspace)

    workflow = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())
    steps = workflow['jobs']['test-devkit']['steps']
    acquisition = next(step for step in steps if step.get('name') == "Download this build's installed-qualification inputs")
    acquisition_command = shlex.split(acquisition['run'].replace('\\\n', ' '))
    assert acquisition_command[:2] == ['python3', 'tools/ci/acquire_core_qualification.py']
    assert acquisition_command[acquisition_command.index('--branch') + 1] == '${GITHUB_HEAD_REF:-${GITHUB_REF_NAME}}'
    assert acquisition_command[acquisition_command.index('--commit') + 1] == '${GITHUB_SHA}'
    assert acquisition_command[acquisition_command.index('--environment') + 1] == '${VULCAN_ENV}'
    artifact_root = workspace / acquisition_command[acquisition_command.index('--output') + 1]
    shutil.copytree(producer, artifact_root)
    inventory_step = next(step for step in steps if step.get('name') == 'Run Installed DevKit Package Inventory Test')
    baseline = next(step for step in steps if step.get('name') == 'Capture native platform before NEAT installation')
    install = next(step for step in steps if step.get('name') == 'Install test runtime from Vulcan artifact')
    assert steps.index(acquisition) < steps.index(baseline) < steps.index(install) < steps.index(inventory_step)
    command = next(line for line in inventory_step['run'].replace('\\\n', ' ').splitlines()
                   if line.strip().startswith('bash scripts/ci/verify_core_dist.sh '))
    argv = shlex.split(command)
    selected_root = workspace / argv[argv.index('--dist') + 1]
    assert selected_root.resolve() == artifact_root.resolve()
    # No mocked controls, no synthesized metadata, and no installed-state data:
    # all exact expected inputs came from this run's independently built artifact.
    assert run(selected_root)['status'] == 'PASS'
    selected_profile = argv[argv.index('--package-profile') + 1]
    expected = verifier.package_inventory(selected_root, profile=selected_profile)
    assert set(expected['packages']) == verifier.PACKAGE_PROFILES[selected_profile]
    (selected_root / 'runner.deb').write_bytes(b'changed after transfer')
    with pytest.raises((verifier.ContractError, subprocess.CalledProcessError)):
        run(selected_root)


def platform_fixture(absent=('appcomplex', 'simaai-common')):
    return {'schema_version': 2, 'scope': 'native-platform-before-install',
            'packages': {name: {'Package': name, 'Version': '1', 'Architecture': 'arm64',
                                'Status': 'install ok installed'}
                         for name in verifier.NATIVE_PLATFORM_ROLES if name not in absent},
            'absent_packages': sorted(absent), 'ota_present': True, 'ota_path': '/opt/sima/bin/simaai-ota', 'ota_sha256': 'a' * 64,
            'ota_owner': 'simaai-palette-modalix'}


def test_platform_capture_records_real_present_and_absent_roles():
    expected = platform_fixture()
    rows = ''.join('\t'.join(row.values()) + '\n' for row in expected['packages'].values())
    # Unrelated multiarch packages must not affect the native-platform projection.
    rows += 'other\t1\tarm64\tinstall ok installed\nother\t1\tarmhf\tinstall ok installed\n'
    with patch.object(verifier.subprocess, 'check_output', side_effect=[rows, '/opt/sima/bin/simaai-ota\n']) as query, \
         patch.object(verifier, 'installed_owners', return_value={'/opt/sima/bin/simaai-ota': 'simaai-palette-modalix'}), \
         patch.object(verifier.Path, 'is_file', return_value=True), patch.object(verifier, 'digest', return_value='a' * 64), \
         patch.object(verifier.shutil, 'which', return_value='/opt/sima/bin/simaai-ota'), \
         patch.object(verifier.Path, 'resolve', return_value=Path('/opt/sima/bin/simaai-ota')):
        assert verifier.platform_inventory() == expected
        assert len(query.call_args_list[0].args[0]) == 3  # full dpkg query, no historical names


@pytest.mark.parametrize('change', ['disappear', 'appear', 'wrong-version', 'wrong-architecture'])
def test_platform_preserves_actual_presence_absence_and_identity(change):
    before = platform_fixture()
    current = copy.deepcopy(before)
    if change == 'disappear':
        current['packages'].pop('simaai-log')
        current['absent_packages'].append('simaai-log')
    elif change == 'appear':
        current['packages']['appcomplex'] = {'Package': 'appcomplex', 'Version': '1', 'Architecture': 'arm64', 'Status': 'install ok installed'}
        current['absent_packages'].remove('appcomplex')
    elif change == 'wrong-version':
        current['packages']['simaai-log']['Version'] = '2'
    else:
        current['packages']['simaai-log']['Architecture'] = 'amd64'
    with patch.object(verifier, 'platform_inventory', return_value=current), pytest.raises(verifier.ContractError, match='Unselected platform package changed'):
        verifier.verify_platform(before, {})


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'overlap', 'palette-absent', 'old-schema'])
def test_platform_baseline_cannot_omit_presence_authority(change):
    before = platform_fixture()
    if change == 'missing': before['absent_packages'].remove('appcomplex')
    elif change == 'duplicate': before['absent_packages'].append('appcomplex')
    elif change == 'overlap': before['absent_packages'].append('simaai-log')
    elif change == 'palette-absent':
        before['packages'].pop('simaai-palette-modalix')
        before['absent_packages'].append('simaai-palette-modalix')
    else: before['schema_version'] = 1
    with pytest.raises(verifier.ContractError): verifier.verify_platform(before, {})


@pytest.mark.parametrize('rows,diagnostic', [
    ('simaai-log\t1\tarm64\tinstall ok half-configured\n', 'not fully installed'),
    ('simaai-log\t1\tarm64\tinstall ok installed\nsimaai-log\t1\tarmhf\tinstall ok installed\n', 'Ambiguous'),
    ('other\t1\tarm64\tinstall ok installed\n', 'Palette'),
    ('malformed\n', 'Malformed'),
])
def test_platform_capture_rejects_incomplete_or_ambiguous_dpkg_state(rows, diagnostic):
    with patch.object(verifier.subprocess, 'check_output', return_value=rows), pytest.raises(verifier.ContractError, match=diagnostic):
        verifier.platform_inventory()


def test_platform_query_failure_is_not_absence():
    with patch.object(verifier.subprocess, 'check_output', side_effect=subprocess.CalledProcessError(1, ['dpkg-query'])), pytest.raises(verifier.ContractError, match='Cannot capture'):
        verifier.platform_inventory()


def test_platform_capture_preserves_supported_ota_absence():
    expected = platform_fixture()
    expected.update(ota_present=False, ota_path=None, ota_owner=None, ota_sha256=None)
    rows = ''.join('\t'.join(row.values()) + '\n' for row in expected['packages'].values())
    with patch.object(verifier.subprocess, 'check_output', return_value=rows), patch.object(verifier.shutil, 'which', return_value=None), patch.object(verifier, 'installed_owners') as owners:
        assert verifier.platform_inventory() == expected
        owners.assert_not_called()
    with patch.object(verifier, 'platform_inventory', return_value=expected):
        assert verifier.verify_platform(expected, {})['status'] == 'PASS'


@pytest.mark.parametrize('change', ['appears', 'disappears', 'path', 'owner', 'hash'])
def test_platform_detects_every_ota_presence_identity_transition(change):
    before = platform_fixture()
    current = copy.deepcopy(before)
    if change == 'appears': before.update(ota_present=False, ota_path=None, ota_owner=None, ota_sha256=None)
    elif change == 'disappears': current.update(ota_present=False, ota_path=None, ota_owner=None, ota_sha256=None)
    elif change == 'path': current['ota_path'] = '/other/bin/simaai-ota'
    elif change == 'owner': current['ota_owner'] = 'not-palette'
    else: current['ota_sha256'] = 'b' * 64
    with patch.object(verifier, 'platform_inventory', return_value=current), pytest.raises(verifier.ContractError, match='OTA command changed'):
        verifier.verify_platform(before, {})


@pytest.mark.parametrize('change', ['missing', 'null-presence', 'false-with-path', 'true-with-no-hash'])
def test_platform_rejects_ambiguous_ota_authority(change):
    before = platform_fixture()
    if change == 'missing': before.pop('ota_present')
    elif change == 'null-presence': before['ota_present'] = None
    elif change == 'false-with-path': before['ota_present'] = False
    else: before['ota_sha256'] = None
    with pytest.raises(verifier.ContractError): verifier.verify_platform(before, {})


def test_platform_capture_rejects_declared_but_missing_ota():
    expected = platform_fixture()
    rows = ''.join('\t'.join(row.values()) + '\n' for row in expected['packages'].values())
    with patch.object(verifier.subprocess, 'check_output', side_effect=[rows, '/opt/sima/bin/simaai-ota\n']), patch.object(verifier.shutil, 'which', return_value=None), pytest.raises(verifier.ContractError, match='declares OTA'):
        verifier.platform_inventory()


def test_platform_capture_rejects_undeclared_ota_on_path():
    expected = platform_fixture()
    rows = ''.join('\t'.join(row.values()) + '\n' for row in expected['packages'].values())
    with patch.object(verifier.subprocess, 'check_output', side_effect=[rows, '/usr/bin/simaai-pdiff\n']), patch.object(verifier.shutil, 'which', return_value='/other/simaai-ota'), patch.object(verifier.Path, 'resolve', return_value=Path('/other/simaai-ota')), pytest.raises(verifier.ContractError, match='not the Palette-declared'):
        verifier.platform_inventory()


def test_client_test_support_is_exact_optional_development_input(tmp_path):
    controls = fixture(tmp_path)
    name = 'simaai-scheduler-client-test-dev.deb'
    (tmp_path / name).write_bytes(b'closed development archive fixture')
    receipt_path = tmp_path / 'resolved-deps-manifest.json'
    receipt = json.loads(receipt_path.read_text())
    entry = {'Package': 'simaai-scheduler-client-test-dev', 'Version': '1', 'Architecture': 'arm64',
             'deb_sha256': verifier.digest(tmp_path / name)}
    receipt['client_test_support'] = {'debian_package': entry}
    receipt_path.write_text(json.dumps(receipt))
    allowlist = tmp_path / 'neat-install-manifest.txt'
    allowlist.write_text(allowlist.read_text() + name + '\n')
    for path in tmp_path.glob('metadata*.json'):
        metadata = json.loads(path.read_text())
        if receipt_path.name not in metadata['resources']:
            continue
        metadata['resources'].append(name)
        metadata['resources-checksum'] = {p: verifier.digest(tmp_path / p) for p in metadata['resources']}
        path.write_text(json.dumps(metadata))
    actual_control = {key: entry[key] for key in ('Package', 'Version', 'Architecture')}
    def control(path):
        if Path(path).name == name:
            return actual_control
        return controls(path)
    with patch.object(verifier, 'control', control):
        assert 'simaai-scheduler-client-test-dev' in run(tmp_path)['dependency_roles']
        for field, invalid in [('Version', 'wrong'), ('Architecture', 'amd64'), ('deb_sha256', '0' * 64)]:
            original = entry[field]
            entry[field] = invalid
            receipt_path.write_text(json.dumps(receipt))
            with pytest.raises(verifier.ContractError, match='Client test-support'):
                run(tmp_path)
            entry[field] = original
        receipt.pop('client_test_support')
        receipt_path.write_text(json.dumps(receipt))
        with pytest.raises(verifier.ContractError, match='Foreign DEB'):
            run(tmp_path)


# Authentication is tested with real incoming DEBs, independently of the small
# artifact-shape doubles above. The stub is a producer entrypoint, not an ABI
# compatibility oracle; Scheduler's own suite owns real provider/ELF semantics.
admission_spec = importlib.util.spec_from_file_location(
    'scheduler_platform_admission', ROOT / 'tools/scheduler_platform_admission.py')
admission = importlib.util.module_from_spec(admission_spec)
admission_spec.loader.exec_module(admission)


def admission_fixture(tmp_path, *, transaction_error=None, empty_platform=False):
    root = tmp_path / 'admission-dist'; root.mkdir()
    source = tmp_path / 'scheduler-source'
    prefix = source / 'usr'
    tool_dir = prefix / 'share/SimaScheduler/tools'; tool_dir.mkdir(parents=True)
    marker = tmp_path / 'executed'
    tool = tool_dir / 'platform_admission.py'
    tool.write_text('''import json, pathlib, sys
from fixture_policy import RESULT
pathlib.Path(''' + repr(str(marker)) + ''').write_text('executed')
pathlib.Path(''' + repr(str(marker.with_suffix('.argv'))) + ''').write_text(json.dumps(sys.argv[1:]))
with pathlib.Path(''' + repr(str(marker.with_suffix('.calls'))) + ''').open('a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1] == 'transaction' and ''' + repr(transaction_error) + ''' is not None:
    raise SystemExit(''' + repr(transaction_error) + ''')
directory = pathlib.Path(sys.argv[sys.argv.index('--directory') + 1])
receipt = json.loads((directory / 'resolved-deps-manifest.json').read_text())
if receipt.get('runner', {}).get('buildinfo', {}).get('platform_runtime_packages') != ['fixture-platform']:
    raise SystemExit('producer rejected missing Platform declaration')
if 'PlatformRuntime' not in receipt.get('external_dependencies', {}):
    raise SystemExit('producer rejected missing Platform evidence')
print(json.dumps(RESULT))
''')
    result = {'schema_version': 1, 'redistributed_provider_controls': [],
              'platform_packages': {'fixture-platform': 'arm64'},
              'packages': {'fixture-platform': {'Package': 'fixture-platform', 'Version': '2',
                                                'Architecture': 'arm64'}},
              'providers': {'libfixture.so': {'package': 'fixture-platform',
                                            'path': '/usr/lib/libfixture.so', 'sha256': 'a' * 64}}}
    if empty_platform:
        for key in ('platform_packages', 'packages', 'providers'):
            result[key] = {}
    (tool_dir / 'fixture_policy.py').write_text('RESULT = ' + repr(result) + '\n')
    files = {}
    for path in tool_dir.iterdir():
        path.chmod(0o644)
        files[path.relative_to(prefix).as_posix()] = {
            'kind': 'file', 'sha256': admission.sha(path), 'mode': 0o644}
    info = {'schema_version': 1, 'component': 'scheduler', 'installed_files': files,
            'installed_payload_sha256': admission.canonical(files)}
    info['package_sha256'] = admission.canonical(info)
    (prefix / 'share/SimaScheduler/buildinfo.json').write_text(json.dumps(info))
    (source / 'DEBIAN').mkdir()
    (source / 'DEBIAN/control').write_text(
        'Package: simaai-scheduler\nVersion: 1\nArchitecture: arm64\n'
        'Maintainer: Test <test@example.invalid>\nDescription: authenticated tooling fixture\n')
    receipt = {'scheduler': {'buildinfo': info},
               'runner': {'buildinfo': {'platform_runtime_packages': ['fixture-platform']}},
               'external_dependencies': {'PlatformRuntime': {'schema_version': 1}}}
    (root / 'resolved-deps-manifest.json').write_text(json.dumps(receipt))
    archive = root / 'scheduler.deb'
    def rebuild():
        subprocess.run(['dpkg-deb', '--build', '--root-owner-group', str(source), str(archive)],
                       check=True, capture_output=True)
    rebuild()
    return root, source, tool, marker, result, rebuild


def test_installed_scheduler_tool_is_authenticated_and_read_only(tmp_path):
    root, source, tool, marker, expected, _ = admission_fixture(tmp_path)
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert admission.invoke(root, 'describe') == expected
    assert marker.read_text() == 'executed'
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    assert not list(source.rglob('__pycache__'))


@pytest.mark.parametrize('repair', [False, True])
def test_transaction_delegates_regular_plan_with_exact_arguments(tmp_path, repair):
    root, _, _, marker, expected, _ = admission_fixture(tmp_path)
    simulation = tmp_path / 'apt simulation.log'; simulation.write_text('Inst fixture (2 local-deb)\n')
    result = admission.optional_admission(root, 'transaction', simulation=simulation,
                                         repair_incomplete=repair)
    assert result == {**expected, 'split_cohort': True}
    arguments = json.loads(marker.with_suffix('.argv').read_text())
    assert arguments == ['transaction', '--directory', str(root), '--simulation', str(simulation),
                         *(['--repair-incomplete'] if repair else [])]


@pytest.mark.parametrize('kind', ['missing', 'directory', 'symlink'])
def test_transaction_rejects_nonregular_simulation_before_import(tmp_path, kind):
    root, _, _, marker, _, _ = admission_fixture(tmp_path)
    simulation = tmp_path / 'simulation'
    if kind == 'directory':
        simulation.mkdir()
    elif kind == 'symlink':
        target = tmp_path / 'target'; target.write_text('')
        simulation.symlink_to(target)
    with pytest.raises(admission.AdmissionError, match='regular file'):
        admission.invoke(root, 'transaction', simulation=simulation)
    assert not marker.exists()


def test_legacy_transaction_cannot_request_incomplete_repair(tmp_path):
    simulation = tmp_path / 'simulation'; simulation.write_text('')
    assert admission.optional_admission(tmp_path, 'transaction', simulation=simulation)['split_cohort'] is False
    with pytest.raises(admission.AdmissionError, match='authenticated split'):
        admission.optional_admission(tmp_path, 'transaction', simulation=simulation, repair_incomplete=True)


@pytest.mark.parametrize('mutation', ['helper', 'sibling', 'mode', 'extra', 'missing', 'symlink', 'receipt', 'inventory_digest'])
@pytest.mark.parametrize('action', ['describe', 'transaction'])
def test_scheduler_tampering_rejected_before_import(tmp_path, mutation, action):
    root, source, tool, marker, _, rebuild = admission_fixture(tmp_path)
    if mutation == 'helper':
        tool.write_text('raise SystemExit("untrusted helper executed")\n')
    elif mutation == 'sibling':
        tool.with_name('fixture_policy.py').write_text('raise SystemExit("untrusted sibling executed")\n')
    elif mutation == 'mode':
        tool.chmod(0o755)
    elif mutation == 'extra':
        tool.with_name('unrecorded.py').write_text('raise SystemExit("unrecorded")\n')
    elif mutation == 'missing':
        tool.unlink()
    elif mutation == 'symlink':
        sibling = tool.with_name('fixture_policy.py')
        target = source / 'usr/share/fixture_policy.py'
        sibling.rename(target)
        sibling.symlink_to('../../fixture_policy.py')
    elif mutation == 'receipt':
        change(root, 'resolved-deps-manifest.json', lambda value: value['scheduler']['buildinfo'].update(component='other'))
    elif mutation == 'inventory_digest':
        receipt = source / 'usr/share/SimaScheduler/buildinfo.json'
        data = json.loads(receipt.read_text()); data['installed_payload_sha256'] = '0' * 64
        data['package_sha256'] = admission.canonical({k: v for k, v in data.items() if k != 'package_sha256'})
        receipt.write_text(json.dumps(data))
        change(root, 'resolved-deps-manifest.json', lambda value: value['scheduler'].update(buildinfo=data))
    rebuild()
    options = {}
    if action == 'transaction':
        simulation = tmp_path / 'simulation'; simulation.write_text('')
        options = {'simulation': simulation, 'repair_incomplete': True}
    with pytest.raises(admission.AdmissionError):
        admission.invoke(root, action, **options)
    assert not marker.exists()


@pytest.mark.parametrize('mutation', ['declaration', 'evidence', 'runner', 'receipt'])
def test_actual_split_owner_cannot_skip_admission_by_stripping_receipt(tmp_path, mutation):
    root, _, _, marker, _, _ = admission_fixture(tmp_path)
    if mutation == 'declaration':
        change(root, 'resolved-deps-manifest.json', lambda value: value['runner']['buildinfo'].clear())
    elif mutation == 'evidence':
        change(root, 'resolved-deps-manifest.json', lambda value: value['external_dependencies'].clear())
    elif mutation == 'runner':
        change(root, 'resolved-deps-manifest.json', lambda value: value.pop('runner'))
    else:
        (root / 'resolved-deps-manifest.json').unlink()
    with pytest.raises((admission.AdmissionError, subprocess.CalledProcessError)):
        admission.optional_board_check(root)
    if mutation == 'receipt':
        assert not marker.exists()


@pytest.mark.parametrize('action', ['Inst', 'Remv'])
@pytest.mark.parametrize('package', ['fixture-platform', 'fixture-platform:arm64'])
def test_apt_cannot_mutate_admitted_platform_provider(tmp_path, action, package):
    path = tmp_path / 'plan'; path.write_text(action + ' ' + package + ' [2] (1 local-deb)\n')
    with pytest.raises(admission.AdmissionError, match='replace or remove'):
        admission.verify_plan({'platform_packages': {'fixture-platform': 'arm64'}}, path)


def test_same_platform_identity_and_product_only_plan_pass(tmp_path):
    _, _, _, _, result, _ = admission_fixture(tmp_path)
    admission.same_providers(result, copy.deepcopy(result))
    plan = tmp_path / 'plan'; plan.write_text('Inst sima-neat [1] (2 local-deb)\n')
    admission.verify_plan(result, plan)
    for key in ('packages', 'providers', 'platform_packages'):
        changed = copy.deepcopy(result); changed[key] = {}
        with pytest.raises(admission.AdmissionError, match='identity changed'):
            admission.same_providers(result, changed)


def test_verifier_uses_genuine_authenticated_scheduler_description(tmp_path):
    root, _, tool, marker, expected, rebuild = admission_fixture(tmp_path)
    assert verifier.scheduler_platform_description(root) == expected
    marker.unlink()
    tool.unlink(); rebuild()
    with pytest.raises(verifier.ContractError, match='classification failed'):
        verifier.scheduler_platform_description(root)
    assert not marker.exists()


def test_delegated_platform_package_cannot_be_redistributed(tmp_path):
    controls = fixture(tmp_path)
    controls.fixture_description = {'schema_version': 1, 'redistributed_provider_controls': [],
                                    'platform_packages': {'simaai-runner': 'arm64'}}
    with patch.object(verifier, 'control', controls), pytest.raises(
            verifier.ContractError, match='Platform-owned provider DEB'):
        run(tmp_path)


def installer_shell(root, body):
    return subprocess.run(['bash', '-c', 'source "$1"; cd "$2"; ' + body,
                           'bash', str(ROOT / 'tools/install_neat_framework.sh'), str(root)],
                          text=True, capture_output=True)


# These tests exercise the real shell/delegate/DEB authentication boundary. The
# sealed producer double owns its yes/no decision; Scheduler's package suite
# tests real dpkg ownership, pending states and APT action admission.
TRANSACTION_SETUP = '''
DEBS=(./scheduler.deb)
refresh_apt_metadata_for_board_install() { :; }
stop_board_runtime_before_install() { :; }
collect_board_heal_specs() { :; }
dpkg_package_database_is_configured() { return 0; }
board_runtime_is_legacy() { return 0; }
complete_board_install_after_packages() { echo COMPLETE; }
remove_installed_local_deb_packages() { echo REMOVE >> commands; }
run_sudo() {
  printf '%s\\n' "$*" >> commands
  if [[ " $* " == *" --simulate "* ]]; then
    echo 'Inst simaai-scheduler (1 local-deb [arm64])'
    echo 'Conf simaai-scheduler (1 local-deb [arm64])'
    return 0
  fi
  if [[ "$1 ${2:-}" == 'apt-get install' ]]; then
    echo APPLY >> commands
    return "${APPLY_STATUS:-0}"
  fi
  [[ "$1 ${2:-}" == 'apt-get check' || "$1" == */ldconfig || "$1" == ldconfig ]]
}
'''


@pytest.mark.parametrize('repair', [False, True])
@pytest.mark.parametrize('allowed', [False, True])
def test_board_applies_only_authenticated_admitted_transaction(tmp_path, repair, allowed):
    root, _, _, marker, _, _ = admission_fixture(
        tmp_path, empty_platform=True, transaction_error=None if allowed else 'foreign pending package')
    result = installer_shell(root, TRANSACTION_SETUP +
                             f'apt_package_database_is_healthy() {{ return {int(repair)}; }}\n'
                             'install_debs_on_board\n')
    assert (result.returncode == 0) == allowed, result.stdout + result.stderr
    commands = (root / 'commands').read_text()
    assert ('APPLY' in commands) == allowed
    assert ('COMPLETE' in result.stdout) == allowed
    arguments = json.loads(marker.with_suffix('.argv').read_text())
    assert arguments[0] == 'transaction'
    assert ('--repair-incomplete' in arguments) == repair
    transactions = [line for line in commands.splitlines() if line.startswith('apt-get install')]
    assert len(transactions) == (2 if allowed else 1)
    for command in transactions:
        assert ('--no-remove' in command) == repair
        assert './scheduler.deb simaai-scheduler:arm64=1' in command
    assert 'REMOVE' not in commands and 'dpkg -i' not in commands


def test_failed_admitted_repair_never_enters_opted_in_maintenance(tmp_path):
    root, _, _, _, _, _ = admission_fixture(tmp_path)
    result = installer_shell(root, TRANSACTION_SETUP + '''
apt_package_database_is_healthy() { return 1; }
NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON
NEAT_INSTALLER_ALLOW_DPKG_FALLBACK=ON
APPLY_STATUS=100
install_debs_on_board
''')
    assert result.returncode != 0
    assert 'Exact split-cohort recovery failed' in result.stderr
    commands = (root / 'commands').read_text()
    assert commands.count('APPLY') == 1
    assert 'REMOVE' not in commands and 'dpkg -i' not in commands
    assert 'COMPLETE' not in result.stdout


@pytest.mark.parametrize('state', [1, 2])
def test_split_pending_state_or_query_error_cannot_use_healthy_transaction(tmp_path, state):
    root, _, _, marker, _, _ = admission_fixture(tmp_path, empty_platform=True)
    result = installer_shell(root, TRANSACTION_SETUP + f'''
apt_package_database_is_healthy() {{ return 0; }}
dpkg_package_database_is_configured() {{ return {state}; }}
install_debs_on_board
''')
    calls = [json.loads(line) for line in marker.with_suffix('.calls').read_text().splitlines()]
    transactions = [call for call in calls if call[0] == 'transaction']
    if state == 1:
        assert result.returncode == 0, result.stdout + result.stderr
        assert len(transactions) == 1 and '--repair-incomplete' in transactions[0]
        commands = (root / 'commands').read_text().splitlines()
        assert all('--no-remove' in line for line in commands if line.startswith('apt-get install'))
    else:
        assert result.returncode != 0
        assert 'Cannot read dpkg package states' in result.stderr
        assert not transactions
        assert not (root / 'commands').exists()


def test_split_maintenance_retry_is_readmitted_and_never_direct_dpkg(tmp_path):
    root, _, _, marker, _, _ = admission_fixture(tmp_path, empty_platform=True)
    result = installer_shell(root, TRANSACTION_SETUP + '''
apt_package_database_is_healthy() { return 0; }
NEAT_INSTALLER_ALLOW_PACKAGE_REMOVAL=ON
NEAT_INSTALLER_ALLOW_DPKG_FALLBACK=ON
APPLY_STATUS=100
install_debs_on_board
''')
    assert result.returncode != 0
    assert 'refusing direct dpkg' in result.stderr
    calls = [json.loads(line) for line in marker.with_suffix('.calls').read_text().splitlines()]
    assert len([call for call in calls if call[0] == 'transaction']) == 2
    commands = (root / 'commands').read_text()
    assert commands.count('APPLY') == 2
    assert commands.count('REMOVE') == 1
    assert commands.count('--simulate') == 2
    assert 'dpkg -i' not in commands
    assert 'COMPLETE' not in result.stdout


def test_unhealthy_legacy_board_still_refuses_before_simulation(tmp_path):
    result = installer_shell(tmp_path, TRANSACTION_SETUP + '''
DEBS=()
apt_package_database_is_healthy() { return 1; }
install_debs_on_board
''')
    assert result.returncode != 0
    assert 'APT package state is unhealthy' in result.stderr
    assert not (tmp_path / 'commands').exists()


@pytest.mark.parametrize('allowed', [False, True])
def test_ros2_uses_same_ownership_gate_without_board_provider_admission(tmp_path, allowed):
    root, _, _, marker, _, _ = admission_fixture(
        tmp_path, transaction_error=None if allowed else 'undeclared file transfer')
    result = installer_shell(root, TRANSACTION_SETUP + '''
apt_package_database_is_healthy() { return 0; }
validate_ros2_sdk_native_host() { :; }
validate_single_sima_neat_package_pair() { :; }
validate_ros2_sdk_deb_architectures() { :; }
validate_ros2_sdk_tvm_runtime() { :; }
install_ros2_sdk_tvm_runtime() { :; }
verify_ros2_sdk_deb_packages_installed() { :; }
repair_global_sima_neat_lib_links() { :; }
verify_global_sima_neat_lib_links() { :; }
install_debs_in_ros2_sdk
''')
    assert (result.returncode == 0) == allowed, result.stdout + result.stderr
    calls = [json.loads(line) for line in marker.with_suffix('.calls').read_text().splitlines()]
    assert [call[0] for call in calls] == ['transaction']
    assert all('--repair-incomplete' not in call for call in calls)
    commands = (root / 'commands').read_text()
    assert ('APPLY' in commands) == allowed
    for line in commands.splitlines():
        if line.startswith('apt-get install'):
            assert '--no-remove' in line and '--fix-broken' not in line


def test_ros2_unhealthy_state_does_not_enter_board_repair(tmp_path):
    root, _, _, marker, _, _ = admission_fixture(tmp_path)
    result = installer_shell(root, TRANSACTION_SETUP + '''
apt_package_database_is_healthy() { return 1; }
validate_ros2_sdk_native_host() { :; }
validate_single_sima_neat_package_pair() { :; }
validate_ros2_sdk_deb_architectures() { :; }
validate_ros2_sdk_tvm_runtime() { :; }
install_debs_in_ros2_sdk
''')
    assert result.returncode != 0
    assert 'Repair the container package database first' in result.stderr
    assert not marker.exists()
    assert not (root / 'commands').exists()


@pytest.mark.parametrize('override', ['FORCE_INSTALL=1', 'NEAT_INSTALLER_SKIP_PLATFORM_CHECK=ON'])
@pytest.mark.parametrize('valid', [False, True])
def test_board_admission_precedes_even_forced_mutation(tmp_path, override, valid):
    root, _, tool, marker, _, rebuild = admission_fixture(tmp_path)
    if not valid:
        tool.unlink(); rebuild()
    result = installer_shell(root, f'''ENV_MODE=modalix-board; {override};
run_sudo() {{ echo MUTATION; }}
ensure_platform_compatible
run_sudo sentinel-or-package-mutation
''')
    assert (result.returncode == 0) == valid, result.stderr
    assert ('MUTATION' in result.stdout) == valid
    assert marker.exists() == valid


@pytest.mark.parametrize('mode', ['elxr-sdk', 'ros2-sdk'])
def test_sdk_mode_does_not_admit_host_as_board(tmp_path, mode):
    root, _, tool, marker, _, rebuild = admission_fixture(tmp_path)
    tool.unlink(); rebuild()
    result = installer_shell(root, f'ENV_MODE={mode}; FORCE_INSTALL=1; ensure_platform_compatible')
    assert result.returncode == 0, result.stderr
    assert not marker.exists()


@pytest.mark.parametrize('action', ['Inst', 'Remv'])
def test_installer_rejects_provider_solver_mutation_before_execution(tmp_path, action):
    root, _, _, _, _, _ = admission_fixture(tmp_path)
    result = installer_shell(root, f'''prepare_board_platform_admission
printf '{action} fixture-platform:arm64 [2] (1 fixture)\\n' > simulation
verify_board_platform_plan simulation
echo MUTATION
''')
    assert result.returncode != 0
    assert 'MUTATION' not in result.stdout


def test_installer_checks_post_install_closure_before_runtime_restart(tmp_path):
    root, _, _, _, _, _ = admission_fixture(tmp_path)
    result = installer_shell(root, '''prepare_board_platform_admission
verify_board_platform_admission --installed
python3 - "$BOARD_PLATFORM_ADMISSION_DIR/before.json" <<'MUTATE'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); v = json.loads(p.read_text())
v['providers']['libfixture.so']['sha256'] = 'b' * 64
p.write_text(json.dumps(v))
MUTATE
activate_board_runtime_after_install() { echo MUTATION; }
complete_board_install_after_packages
''')
    assert result.returncode != 0
    assert 'MUTATION' not in result.stdout
    assert 'identity changed' in result.stderr


def test_packaging_carries_shared_delegate_and_selected_receipt():
    build = (ROOT / 'build.sh').read_text()
    installer = (ROOT / 'tools/install_neat_framework.sh').read_text()
    assert 'dist/resolved-deps-manifest.json dist/neat-install-manifest.txt' in build
    assert '"tools/scheduler_platform_admission.py" dist/' in build
    assert '"${REPO_ROOT}/tools/scheduler_platform_admission.py"' in build
    assert 'for admission_resource in scheduler_platform_admission.py resolved-deps-manifest.json' in installer
    assert 'cp -f ./resolved-deps-manifest.json "${cache_dir}/"' in installer


@pytest.mark.parametrize('removed,allowed', [
    ('fixture-platform:arm64', False), ('unrelated-dependent', False), ('simaai-scheduler', True),
])
@pytest.mark.parametrize('empty_platform', [False, True])
def test_maintenance_removal_is_still_product_only(tmp_path, removed, allowed, empty_platform):
    root, _, _, _, _, _ = admission_fixture(tmp_path, empty_platform=empty_platform)
    result = installer_shell(root, f'''prepare_board_platform_admission
DEBS=(./scheduler.deb)
dpkg-query() {{ printf 'ii'; }}
run_sudo() {{
  if [[ "$*" == *--simulate* ]]; then
    printf 'Remv {removed} [1]\\n'
  else
    echo MUTATION
  fi
}}
remove_installed_local_deb_packages
''')
    assert (result.returncode == 0) == allowed, result.stdout + result.stderr
    assert ('MUTATION' in result.stdout) == allowed


@pytest.mark.parametrize('libdir,runtime_dir,plugin_dir', [
    ('lib', 'lib/neat/runtime', 'lib/neat/gst-plugins'),
    ('lib64', 'lib64/neat/runtime', 'lib64/neat/gst-plugins'),
    ('lib/aarch64-linux-gnu', 'lib/aarch64-linux-gnu/neat/runtime',
     'lib/aarch64-linux-gnu/neat/gst-plugins'),
    ('lib', 'lib/aarch64-linux-gnu/neat/runtime', 'lib/aarch64-linux-gnu/neat/gst-plugins'),
    ('lib/aarch64-linux-gnu', 'lib/neat/runtime', 'lib/neat/gst-plugins'),
    ('lib64', 'lib/x86_64-linux-gnu/neat/runtime', 'lib/x86_64-linux-gnu/neat/gst-plugins'),
    ('lib/core', 'support/runtime', 'plugins/frontend'),
])
def test_actual_core_install_rpath_is_relative_and_shared_only(tmp_path, libdir, runtime_dir, plugin_dir):
    """Compile the production target properties; static clients own their path.

    These are explicit native host doubles, not cross-target execution. Both
    private DSOs are direct dependencies, matching Scheduler's frontend support.
    Core and Scheduler independently select their install directories.
    """
    production = (ROOT / 'CMakeLists.txt').read_text()
    manifest = json.loads((ROOT / 'deps/manifest.json').read_text())
    abi = manifest['abi-version']
    platform_version = manifest['platform-version']
    # Distributed ABI 6 previews used the smaller public ArgMax options/class.
    assert int(abi) > 6
    properties = re.search(r'set_target_properties\(sima_neat PROPERTIES.*?\n\)',
                           production, re.DOTALL).group(0)
    static_properties = re.search(r'set_target_properties\(sima_neat_static PROPERTIES.*?\n\)',
                                  production, re.DOTALL).group(0)
    assert 'INSTALL_RPATH "${SIMANEAT_SCHEDULER_INSTALL_RPATH}"' in properties
    assert 'include(cmake/SchedulerInstallRpath.cmake)' in production
    assert 'simaneat_scheduler_install_rpath(SIMANEAT_SCHEDULER_INSTALL_RPATH)' in production
    assert 'RPATH' not in static_properties
    assert not re.search(r'set\(CMAKE_INSTALL_RPATH', production)
    source = tmp_path / 'source'; source.mkdir()
    (source / 'runtime.cpp').write_text('extern "C" int fixture_plugin();\n'
                                      'extern "C" int fixture_runtime() { return fixture_plugin() - 2; }\n')
    (source / 'plugin.cpp').write_text('extern "C" int fixture_plugin() { return 22; }\n')
    (source / 'core.cpp').write_text('extern "C" int fixture_runtime();\n'
                                    'extern "C" int fixture_plugin();\n'
                                    'extern "C" int fixture_core() { return fixture_runtime() + fixture_plugin(); }\n')
    (source / 'main.cpp').write_text('extern "C" int fixture_core();\n'
                                    'int main() { return fixture_core() == 42 ? 0 : 1; }\n')
    (source / 'CMakeLists.txt').write_text("""cmake_minimum_required(VERSION 3.16)
project(CoreInstalledLoaderFixture LANGUAGES CXX)
include(GNUInstallDirs)
""" + f'''set(SIMANEAT_PLATFORM_VERSION {platform_version})
set(SIMANEAT_ABI_VERSION {abi})
# The namespaced exports retain the selected staging prefix even when a
# transitive dependency has overwritten CMake's ambient PACKAGE_PREFIX_DIR.
# No receipt file is required for this declared-layout calculation.
set(SimaScheduler_BUILD_RECEIPT "{tmp_path}/selected scheduler/share/SimaScheduler/buildinfo.json")
set(SimaScheduler_RUNTIME_DIR "{tmp_path}/selected scheduler/{runtime_dir}")
set(SimaScheduler_PLUGIN_DIR "{tmp_path}/selected scheduler/{plugin_dir}")
set(PACKAGE_PREFIX_DIR "{tmp_path}/unrelated-dependency")
include("{ROOT}/cmake/SchedulerInstallRpath.cmake")
simaneat_scheduler_install_rpath(SIMANEAT_SCHEDULER_INSTALL_RPATH)
set(PROVIDER_RUNTIME_DIR "{runtime_dir}")
set(PROVIDER_PLUGIN_DIR "{plugin_dir}")
''' + """
add_library(fixture_runtime SHARED runtime.cpp)
add_library(fixture_plugin SHARED plugin.cpp)
target_link_libraries(fixture_runtime PRIVATE fixture_plugin)
""" + f'''set_target_properties(fixture_runtime PROPERTIES INSTALL_RPATH
  "$ORIGIN;$ORIGIN/{os.path.relpath(plugin_dir, runtime_dir)}")
''' + """
add_library(sima_neat SHARED core.cpp)
add_library(sima_neat_static STATIC core.cpp)
""" + properties + '\n' + static_properties + """
target_link_libraries(sima_neat PRIVATE fixture_runtime fixture_plugin)
target_link_libraries(sima_neat_static PRIVATE fixture_runtime fixture_plugin)
install(TARGETS sima_neat sima_neat_static DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(TARGETS fixture_runtime DESTINATION ${PROVIDER_RUNTIME_DIR})
install(TARGETS fixture_plugin DESTINATION ${PROVIDER_PLUGIN_DIR})
add_executable(shared_client main.cpp)
target_link_libraries(shared_client PRIVATE sima_neat)
set_target_properties(shared_client PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
add_executable(static_client main.cpp)
target_link_libraries(static_client PRIVATE sima_neat_static)
# This belongs to the final static consumer, not Core's archive/interface.
set_target_properties(static_client PROPERTIES INSTALL_RPATH
  "$ORIGIN/../${PROVIDER_RUNTIME_DIR};$ORIGIN/../${PROVIDER_PLUGIN_DIR}")
add_executable(static_without_path main.cpp)
target_link_libraries(static_without_path PRIVATE sima_neat_static)
install(TARGETS shared_client static_client static_without_path DESTINATION bin)
""")
    env = os.environ.copy()
    for key in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'CPATH',
                'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                'LD_AUDIT', 'CMAKE_TOOLCHAIN_FILE', 'CMAKE_PREFIX_PATH', 'DESTDIR', 'SYSROOT'):
        env.pop(key, None)
    build = tmp_path / 'build'; installed = tmp_path / 'installed'
    def command(*args, check=True):
        return subprocess.run(list(map(str, args)), env=env, text=True, capture_output=True, check=check)
    command('cmake', '-S', source, '-B', build, '-DCMAKE_CXX_COMPILER=/usr/bin/c++',
            '-DCMAKE_INSTALL_LIBDIR=' + libdir, '-DCMAKE_INSTALL_PREFIX=' + str(installed))
    command('cmake', '--build', build, '--parallel', '4')
    command('cmake', '--install', build)
    legacy = tmp_path / 'abi6'
    (legacy / libdir).mkdir(parents=True)
    (legacy / 'bin').mkdir()
    (source / 'legacy.cpp').write_text('extern "C" int fixture_core() { return 42; }\n')
    command('/usr/bin/c++', '-shared', '-fPIC', source / 'legacy.cpp',
            '-Wl,-soname,libsima_neat.so.6', '-o', legacy / libdir / 'libsima_neat.so.6')
    old_client = legacy / 'bin/legacy_client'
    command('/usr/bin/c++', source / 'main.cpp', legacy / libdir / 'libsima_neat.so.6',
            '-Wl,-rpath,$ORIGIN/../' + libdir, '-o', old_client)
    assert 'x86-64' in command('file', old_client).stdout
    command(old_client)  # The old consumer works with its actual old ABI provider.
    shutil.copy2(old_client, installed / 'bin/legacy_client')
    relocated = tmp_path / 'relocated'
    shutil.copytree(installed, relocated, symlinks=True)
    # Remove the original directory name from lookup, proving relocation rather
    # than accidentally finding the original installation or build tree.
    installed.rename(tmp_path / 'original-install-not-at-its-old-path')
    shutil.rmtree(build)
    expected_rpath = ':'.join('$ORIGIN/' + os.path.relpath(directory, libdir)
                              for directory in (runtime_dir, plugin_dir))
    for prefix in (tmp_path / 'original-install-not-at-its-old-path', relocated):
        dynamic = command('readelf', '-d', prefix / libdir / f'libsima_neat.so.{platform_version}').stdout
        assert f'Library soname: [libsima_neat.so.{abi}]' in dynamic
        assert (prefix / libdir / 'libsima_neat.so').readlink().name == f'libsima_neat.so.{abi}'
        assert not (prefix / libdir / 'libsima_neat.so.6').exists()
        assert '[' + expected_rpath + ']' in dynamic
        assert str(tmp_path) not in dynamic
        assert 'libfixture_plugin' in dynamic and 'libfixture_runtime' in dynamic
        runtime_dynamic = command('readelf', '-d', prefix / runtime_dir / 'libfixture_runtime.so').stdout
        assert 'libfixture_plugin' in runtime_dynamic
        needed = command('readelf', '-d', prefix / 'bin/shared_client').stdout
        assert f'Shared library: [libsima_neat.so.{abi}]' in needed
        assert 'Shared library: [libsima_neat.so.6]' in command(
            'readelf', '-d', prefix / 'bin/legacy_client').stdout
        for client in ('shared_client', 'static_client', 'static_without_path', 'legacy_client'):
            executable = prefix / 'bin' / client
            identity = command('file', executable).stdout
            assert 'ELF' in identity and 'x86-64' in identity, identity
            result = command(executable, check=False)
            assert (result.returncode == 0) == (client in ('shared_client', 'static_client')), result.stderr
            if client == 'legacy_client':
                assert 'libsima_neat.so.6' in result.stderr
        static_dynamic = command('readelf', '-d', prefix / 'bin/static_client').stdout
        assert 'Shared library: [libsima_neat' not in static_dynamic
        assert 'libfixture_plugin' in static_dynamic and 'libfixture_runtime' in static_dynamic


@pytest.mark.parametrize('overrides,error', [
    ({'SimaScheduler_BUILD_RECEIPT': ''}, 'absolute SimaScheduler_BUILD_RECEIPT'),
    ({'SimaScheduler_BUILD_RECEIPT': 'share/SimaScheduler/buildinfo.json'},
     'absolute SimaScheduler_BUILD_RECEIPT'),
    ({'SimaScheduler_BUILD_RECEIPT': '/selected/share/other/buildinfo.json'},
     'receipt must use share/SimaScheduler/buildinfo.json'),
    ({'SimaScheduler_RUNTIME_DIR': ''}, 'absolute SimaScheduler_RUNTIME_DIR'),
    ({'SimaScheduler_PLUGIN_DIR': 'lib/neat/gst-plugins'}, 'absolute SimaScheduler_PLUGIN_DIR'),
    ({'SimaScheduler_RUNTIME_DIR': '/selected/../other/runtime'},
     'SimaScheduler_RUNTIME_DIR must remain within'),
    ({'SimaScheduler_PLUGIN_DIR': '/selected-sibling/plugins'},
     'SimaScheduler_PLUGIN_DIR must remain within'),
    ({'CMAKE_INSTALL_LIBDIR': '../lib'}, 'CMAKE_INSTALL_LIBDIR must remain within'),
    ({'CMAKE_INSTALL_LIBDIR': '/other/lib'}, 'CMAKE_INSTALL_LIBDIR must remain within'),
    ({'CMAKE_INSTALL_LIBDIR': ''}, 'CMAKE_INSTALL_LIBDIR must not be empty'),
    ({'CMAKE_INSTALL_LIBDIR': '/core/lib'}, None),
    ({'SimaScheduler_RUNTIME_DIR': '/selected/lib/../lib/neat/runtime'}, None),
])
def test_scheduler_install_rpath_uses_confined_declared_layout(tmp_path, overrides, error):
    variables = {
        'CMAKE_INSTALL_PREFIX': '/core', 'CMAKE_INSTALL_LIBDIR': 'lib',
        'SimaScheduler_BUILD_RECEIPT': '/selected/share/SimaScheduler/buildinfo.json',
        'SimaScheduler_RUNTIME_DIR': '/selected/lib/neat/runtime',
        'SimaScheduler_PLUGIN_DIR': '/selected/lib/neat/gst-plugins',
        'PACKAGE_PREFIX_DIR': '/unrelated-dependency',
        **overrides,
    }
    script = tmp_path / 'layout.cmake'
    script.write_text('\n'.join(f'set({key} "{value}")' for key, value in variables.items()) + f'''
include("{ROOT}/cmake/SchedulerInstallRpath.cmake")
simaneat_scheduler_install_rpath(result)
message(STATUS "selected-rpath=${{result}}")
''')
    result = subprocess.run(['cmake', '-P', str(script)], text=True, capture_output=True)
    assert (result.returncode != 0) == bool(error), result.stdout + result.stderr
    if error:
        assert error in result.stderr
        assert 'selected-rpath=' not in result.stdout
    else:
        assert 'selected-rpath=$ORIGIN/neat/runtime;$ORIGIN/neat/gst-plugins' in result.stdout


@pytest.mark.parametrize('installed_abi', ['6', 'selected'])
def test_install_smoke_requires_selected_manifest_abi(tmp_path, installed_abi):
    manifest = json.loads((ROOT / 'deps/manifest.json').read_text())
    expected = manifest['abi-version']
    selected = expected if installed_abi == 'selected' else installed_abi
    source = tmp_path / 'source/tests/install_smoke'
    source.mkdir(parents=True)
    (tmp_path / 'source/deps').mkdir()
    shutil.copy2(ROOT / 'deps/manifest.json', tmp_path / 'source/deps/manifest.json')
    # Exercise the actual consumer's admission block, not a duplicated check.
    block = (ROOT / 'tests/install_smoke/CMakeLists.txt').read_text().split('add_executable(', 1)[0]
    (source / 'CMakeLists.txt').write_text(block)
    package = tmp_path / 'package'; package.mkdir()
    (package / 'SimaNeatConfig.cmake').write_text(f'set(SimaNeat_ABI_VERSION "{selected}")\n')
    env = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path), 'LANG': 'C'}
    result = subprocess.run(['cmake', '-S', str(source), '-B', str(tmp_path / 'build'),
                             '-DCMAKE_CXX_COMPILER=/usr/bin/c++', f'-DSimaNeat_DIR={package}'],
                            env=env, text=True, capture_output=True)
    assert (result.returncode == 0) == (selected == expected), result.stdout + result.stderr
    if selected != expected:
        assert f"Expected SimaNeat public ABI {expected}, got '{selected}'" in result.stderr
