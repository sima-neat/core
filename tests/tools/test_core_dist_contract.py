"""Small artifact-set fixtures; deep dependency receipt tests live elsewhere."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import copy
import shutil
import subprocess
import tarfile
import zipfile
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('verify_core_dist', ROOT / 'tools/ci/verify_core_dist.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def fixture(root):
    debs = {
        'sima-neat-1-Linux-core.deb': 'sima-neat',
        'sima-neat-1-Linux-dev.deb': 'sima-neat-dev',
        'kernels.deb': 'simaai-kernels-dev',
        'runner.deb': 'simaai-runner',
        'scheduler.deb': 'simaai-scheduler',
        'runtime-compat.deb': 'simaai-runtime',
    }
    for name in [*debs, 'pyneat-1-py3-none-any.whl', 'install_neat_framework.sh']:
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
            'neat-install-manifest.txt', 'install_neat_framework.sh']
    for name, resources, selectables in (
        ('metadata.json', base, [extras]), ('metadata-minimal.json', base, []),
        ('metadata-all.json', base + [extras], []), ('metadata-pyneat.json', ['pyneat-1-py3-none-any.whl'], []),
        ('metadata-extras.json', [extras], []),
    ):
        metadata = {'name': 'sima-neat', 'version': '1', 'release': '',
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
    return verifier.verify(root, validate_metadata=lambda _: None, **kwargs)


def test_golden_set_is_read_only(tmp_path):
    controls = fixture(tmp_path)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}
    with patch.object(verifier, 'control', controls):
        assert run(tmp_path)['status'] == 'PASS'
    assert before == {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.iterdir()}


@pytest.mark.parametrize('mutation', [
    lambda r: (r / 'manifest.json').write_text('{}'),
    lambda r: (r / 'runner.deb').unlink(),
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


def test_raising_cli_validator_is_called(tmp_path):
    controls = fixture(tmp_path)
    def reject(_):
        raise RuntimeError('CLI-owned schema rejection')
    with patch.object(verifier, 'control', controls), pytest.raises(RuntimeError, match='CLI-owned'):
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
    (lambda r, p: change(r, 'resolved-deps-manifest.json', lambda v: v['external_dependencies']['RunnerBackends'].clear()), 'backend providers differ'),
    (lambda r, p: change(r, 'resolved-deps-manifest.json', lambda v: v['external_dependencies']['RunnerBackends']['ArbitraryBackend']['debian_provider'].update(Version='2')), 'differs from Runner'),
    (lambda r, p: change(r, 'resolved-deps-manifest.json', lambda v: v['runner']['buildinfo']['debian_dependencies'].clear()), 'Runner provider dependency'),
    (lambda r, p: change(r, 'resolved-deps-manifest.json', lambda v: v['external_dependencies'].pop('BoxRender')), 'renderer provider differs'),
    (lambda r, p: change(r, 'resolved-deps-manifest.json', lambda v: v['external_dependencies']['BoxRender']['buildinfo']['packages']['development'].update(Version='2')), 'renderer provider differs'),
    (lambda r, p: (r / 'fixture-renderer.deb').write_bytes((r / 'fixture-renderer.deb').read_bytes() + b'changed'), 'archive differs'),
    (lambda r, p: change(r, 'resolved-deps-manifest.json', lambda v: v['scheduler']['buildinfo']['debian_dependencies'].clear()), 'Scheduler renderer dependency'),
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
    acquisition = next(step for step in steps if step.get('uses', '').startswith('actions/download-artifact@'))
    assert acquisition['with']['name'] == workflow['jobs']['build']['with']['artifact_name']
    artifact_root = workspace / acquisition['with']['path']
    shutil.copytree(producer, artifact_root)
    inventory_step = next(step for step in steps if step.get('name') == 'Run Installed DevKit Package Inventory Test')
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
