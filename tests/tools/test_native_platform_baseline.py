"""Exercise the exact native inventory baseline contract on the host, without dpkg mutation."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('platform_inventory_verifier', ROOT / 'tools/ci/verify_core_dist.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def host_elf(path):
    description = subprocess.check_output(['file', str(path)], text=True)
    expected = {'x86_64': 'x86-64', 'amd64': 'x86-64', 'aarch64': 'ARM aarch64', 'arm64': 'ARM aarch64'}
    machine = platform.machine().lower()
    assert machine in expected and expected[machine] in description, description


@pytest.fixture(scope='module')
def probe(tmp_path_factory):
    assert shutil.which('c++') and shutil.which('file'), 'native compiler and file are required'
    root = tmp_path_factory.mktemp('native-platform-baseline')
    source = root / 'probe.cpp'
    source.write_text(r'''
#include "support/package_platform_baseline.h"
#include <iostream>
int main(int argc, char**) {
  try {
    using namespace simaai::neat::test_support;
    if (argc > 1) { std::cout << nlohmann::json(native_platform_roles()); return 0; }
    nlohmann::json input;
    std::cin >> input;
    if (input.contains("resolve")) {
      std::cout << resolve_platform_ota(input.at("palette_files"), input.at("command_path"));
      return 0;
    }
    require_platform_preserved(input.at("before"), input.at("selected"), input.at("current"));
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
''')
    binary = root / 'probe'
    command = ['c++', '-std=c++20', '-Wall', '-Wextra', '-Werror', '-I', str(ROOT / 'tests'),
               str(source), '-o', str(binary)]
    result = subprocess.run(command, text=True, capture_output=True)
    assert result.returncode == 0, (
        f'Native platform probe compile failed: {command!r}\n'
        f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}')
    host_elf(binary)
    return binary


def snapshot(present=False):
    installed = {'simaai-palette-modalix', 'simaai-mlart-modalix'}
    return {'schema_version': 2, 'scope': 'native-platform-before-install',
            'packages': {name: {'Package': name, 'Version': '1', 'Architecture': 'arm64',
                                'Status': 'install ok installed'} for name in installed},
            'absent_packages': sorted(set(verifier.NATIVE_PLATFORM_ROLES) - installed),
            'ota_present': present, 'ota_path': '/usr/sbin/simaai-ota' if present else None,
            'ota_owner': 'simaai-palette-modalix' if present else None,
            'ota_sha256': 'a' * 64 if present else None}


def run(probe, before, current, selected=None):
    host_elf(probe)
    return subprocess.run([str(probe)], input=json.dumps({'before': before, 'current': current,
                          'selected': selected or {}}), text=True, capture_output=True)


def test_python_and_native_track_identical_roles(probe):
    host_elf(probe)
    result = subprocess.check_output([str(probe), 'roles'], text=True)
    assert set(json.loads(result)) == set(verifier.NATIVE_PLATFORM_ROLES)


def test_absent_historical_roles_do_not_become_requirements(probe):
    before = snapshot()
    assert 'appcomplex' in before['absent_packages'] and 'simaai-common' in before['absent_packages']
    result = run(probe, before, copy.deepcopy(before))
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', ['remove', 'version', 'architecture', 'unexpected_install'])
def test_unselected_platform_changes_are_rejected(probe, change):
    before = snapshot(); current = copy.deepcopy(before)
    if change == 'remove':
        del current['packages']['simaai-mlart-modalix']
        current['absent_packages'].append('simaai-mlart-modalix')
    elif change in ('version', 'architecture'):
        current['packages']['simaai-mlart-modalix'][change.title()] = '2' if change == 'version' else 'all'
    else:
        current['absent_packages'].remove('appcomplex')
        current['packages']['appcomplex'] = {'Package': 'appcomplex', 'Version': '1',
                                             'Architecture': 'arm64', 'Status': 'install ok installed'}
    result = run(probe, before, current)
    assert result.returncode == 1 and 'unselected platform package changed' in result.stderr


def test_selected_provider_upgrade_preserves_separate_selected_authority(probe):
    before = snapshot(); current = copy.deepcopy(before)
    current['packages']['simaai-mlart-modalix']['Version'] = '2'
    result = run(probe, before, current, {'simaai-mlart-modalix': {'Version': '2'}})
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('change', ['schema', 'missing_role', 'overlap', 'duplicate_absence',
                                    'identity', 'partial', 'missing_palette', 'bad_hash', 'owner', 'relative_path'])
def test_invalid_preinstall_authority_is_rejected(probe, change):
    before = snapshot(); current = copy.deepcopy(before)
    if change == 'schema': before['schema_version'] = 1
    elif change == 'missing_role': before['absent_packages'].pop()
    elif change == 'overlap': before['absent_packages'].append('simaai-mlart-modalix')
    elif change == 'duplicate_absence': before['absent_packages'].append(before['absent_packages'][0])
    elif change == 'identity': before['packages']['simaai-mlart-modalix']['Package'] = 'other'
    elif change == 'partial': before['packages']['simaai-mlart-modalix']['Status'] = 'install ok unpacked'
    elif change == 'missing_palette':
        del before['packages']['simaai-palette-modalix']; before['absent_packages'].append('simaai-palette-modalix')
    elif change == 'bad_hash': before['ota_sha256'] = 'not-a-hash'
    elif change == 'owner': before['ota_owner'] = 'other'
    elif change == 'relative_path': before['ota_path'] = 'usr/sbin/simaai-ota'
    result = run(probe, before, current)
    assert result.returncode == 1, change


@pytest.mark.parametrize('field,value', [('ota_sha256', 'b' * 64), ('ota_path', '/other/simaai-ota')])
def test_ota_identity_cannot_change_even_when_palette_is_selected(probe, field, value):
    before = snapshot(present=True); current = copy.deepcopy(before); current[field] = value
    result = run(probe, before, current, {'simaai-palette-modalix': {}})
    assert result.returncode == 1 and 'OTA command changed' in result.stderr


@pytest.mark.parametrize('present', [False, True])
def test_ota_presence_changes_in_either_direction_are_rejected(probe, present):
    before = snapshot(present=present); current = snapshot(present=not present)
    result = run(probe, before, current)
    assert result.returncode == 1 and 'OTA command changed' in result.stderr


def test_present_ota_requires_actual_palette_ownership(probe):
    before = snapshot(present=True); current = copy.deepcopy(before)
    current['ota_owner'] = 'unrelated-package'
    result = run(probe, before, current)
    assert result.returncode == 1 and 'OTA ownership differs' in result.stderr


def test_present_ota_is_preserved(probe):
    before = snapshot(present=True)
    result = run(probe, before, copy.deepcopy(before))
    assert result.returncode == 0, result.stderr


def test_canonical_native_lane_requires_independent_baseline():
    native = (ROOT / 'tests/unit_testing/unit_000_devkit_package_inventory_test.cpp').read_text()
    workflow = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()
    assert 'SIMA_TEST_PLATFORM_BASELINE' in native
    assert 'require_platform_preserved(before, inventory.at("packages")' in native
    assert 'require_installed_packages' not in native
    assert 'export SIMA_TEST_PLATFORM_BASELINE=' in workflow
    assert workflow.index('--capture-platform-baseline') < workflow.index('export SIMA_TEST_PLATFORM_BASELINE=')


@pytest.mark.parametrize('state', ['absent', 'present', 'declared_missing', 'declared_not_in_path',
                                   'declared_nonexec', 'declared_directory', 'other_executable', 'declared_symlink'])
def test_actual_ota_resolution_distinguishes_absence_from_damage(probe, tmp_path, state):
    executable = tmp_path / 'simaai-ota'
    files = '/usr/bin/simaai-pdiff\n'
    command = ''
    if state != 'absent':
        files += str(executable) + '\n'
        command = str(executable)
        if state == 'declared_directory':
            executable.mkdir()
        elif state not in ('declared_missing', 'declared_not_in_path'):
            executable.write_text('#!/bin/sh\nexit 0\n')
            executable.chmod(0o644 if state == 'declared_nonexec' else 0o755)
        if state == 'declared_not_in_path':
            executable.write_text('#!/bin/sh\nexit 0\n'); executable.chmod(0o755); command = ''
        if state == 'other_executable':
            other = tmp_path / 'simaai-pdiff'
            other.write_text('#!/bin/sh\nexit 0\n'); other.chmod(0o755); command = str(other)
        if state == 'declared_symlink':
            alias = tmp_path / 'alias'; alias.symlink_to(executable); command = str(alias)
    host_elf(probe)
    result = subprocess.run([str(probe)], input=json.dumps({'resolve': True, 'palette_files': files,
                            'command_path': command}), text=True, capture_output=True)
    assert result.returncode == (0 if state in ('absent', 'present', 'declared_symlink') else 1), result.stderr
    if state == 'absent': assert result.stdout == ''
    if state in ('present', 'declared_symlink'): assert result.stdout == str(executable.resolve())
