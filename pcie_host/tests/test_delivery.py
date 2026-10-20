#!/usr/bin/env python3
"""Core composition tests use a provider seam, not copied Scheduler policy."""
import importlib.util
import io
import json
import os
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('delivery', ROOT / 'scripts/pcie_delivery.py')
delivery = importlib.util.module_from_spec(spec); spec.loader.exec_module(delivery)
spec = importlib.util.spec_from_file_location('native_acquisition', ROOT / 'scripts/acquire_scheduler.py')
acquisition = importlib.util.module_from_spec(spec); spec.loader.exec_module(acquisition)


class Delivery(unittest.TestCase):
    def test_installer_one_transaction_and_runtime_subset(self):
        source = (ROOT / 'scripts/install_pciehost.sh').read_text()
        self.assertIn('pcie_delivery.py', source)
        self.assertIn('closure_args+=(--runtime-only)', source)
        self.assertEqual(source.count('run_sudo apt-get "${apt_install_opts[@]}"'), 1)
        self.assertLess(source.index('package_list="$(python3'), source.index('run_sudo apt-get "${apt_install_opts[@]}"'))
        self.assertIn('closure_args+=(--extra-dependency "$dependency")', source)

    def test_build_consumes_staged_component_without_privileged_install(self):
        source = (ROOT / 'build.sh').read_text()
        block = source.split('GENAI_CMAKE_ARGS=()\n', 1)[1].split('\ncmake -S .', 1)[0]
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            scheduler_dir = directory / 'private prefix/usr/lib/x86_64-linux-gnu/cmake/SimaScheduler'
            selection = directory / 'fixture.json'
            selection.write_text(json.dumps({'scheduler_dir': str(scheduler_dir), 'package_version': '1.2.3'}))
            script = r'''
set -euo pipefail
MAKE_DEB=ON; HOST_DISTRO=ubuntu24; DEB_ARCH=amd64
SCHEDULER_SPEC=latest; SCHEDULER_ENVIRONMENT=production
current_core_branch() { printf 'feature/native'; }
run_privileged() { echo 'unexpected privileged installation' >&2; exit 99; }
python3() {
  if [[ "$1" == scripts/acquire_scheduler.py ]]; then
    mkdir -p "$native_root/selected"
    cp "$fixture" "$native_root/selected/selection.json"
  else
    command python3 "$@"
  fi
}
GENAI_CMAKE_ARGS=()
''' + block + '\nprintf "%s\\n" "${GENAI_CMAKE_ARGS[@]}"\n'
            env = dict(os.environ, TMPDIR=temp, fixture=str(selection))
            result = subprocess.run(['bash', '-c', script], env=env, text=True,
                                    check=True, capture_output=True)
            self.assertEqual(result.stdout.splitlines(), [
                f'-DSimaScheduler_DIR={scheduler_dir}', '-DSIMAPCIE_SCHEDULER_PACKAGE_VERSION=1.2.3'])

    def test_install_list_delegates_complete_plan_before_emitting_paths(self):
        selection = {'distro': 'ubuntu24', 'architecture': 'amd64', 'commit': 'a' * 40}
        packages = ['/dist/scheduler.deb', '/dist/core.deb']
        argv = ['pcie_delivery.py', 'install-list', '--directory', '/dist',
                '--runtime-only', '--extra-dependency', 'python3-venv']
        for failure in (False, True):
            with self.subTest(provider_rejected=failure), patch.object(sys, 'argv', argv), \
                 patch.object(delivery, 'verify', return_value=(selection, packages)), \
                 patch.object(Path, 'read_text', return_value='ID=ubuntu\nVERSION_ID="24.04"\n'), \
                 patch.object(delivery.subprocess, 'check_output', return_value='amd64\n'), \
                 patch.object(delivery, 'load_provider') as provider, \
                 patch.object(sys, 'stdout', io.StringIO()) as output:
                if failure:
                    provider.return_value.validate_install_plan.side_effect = ValueError('Platform mismatch')
                    with self.assertRaisesRegex(ValueError, 'Platform mismatch'):
                        delivery.main()
                    self.assertEqual(output.getvalue(), '')
                else:
                    delivery.main()
                    self.assertEqual(output.getvalue().splitlines(), packages)
                provider.return_value.validate_install_plan.assert_called_once_with(
                    '/dist', packages, 'ubuntu24', 'amd64', 'a' * 40,
                    extra_dependencies=['python3-venv'])

    def test_platform_source_resource_never_becomes_an_install_package(self):
        for runtime_only in (False, True):
            with self.subTest(runtime_only=runtime_only), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                candidate = root / 'host-candidate.json'
                candidate.write_text('{}')
                selection = {'profile': 'pcie-host', 'candidate_sha256': delivery.sha(candidate),
                             'resources': {}, 'distro': 'ubuntu24', 'architecture': 'amd64',
                             'commit': 'a' * 40, 'package_version': '1.0'}
                (root / 'scheduler-native-selection.json').write_text(json.dumps(selection))
                runtime = root / 'sima-pcie-host_0.6.0_amd64.deb'
                development = root / 'sima-pcie-host-dev_0.6.0_amd64.deb'
                runtime.touch(); development.touch()
                provider = Mock()
                provider.verify.return_value = {
                    'packages': {
                        'runtime': {'resource': 'scheduler.deb', 'payload': {'control': {'Version': '1.0'}}},
                        'development': {'resource': 'scheduler-dev.deb', 'payload': {'control': {'Version': '1.0'}}},
                    },
                    'platform_source': {'resource': 'official-platform-source.deb'},
                }
                provider.inspect_deb.side_effect = [
                    {'control': {'Package': 'sima-pcie-host', 'Version': '0.6.0'}},
                    {'control': {'Package': 'sima-pcie-host-dev', 'Version': '0.6.0'}},
                ]
                with patch.object(delivery, 'load_provider', return_value=provider), \
                     patch.object(delivery.subprocess, 'check_output', side_effect=[
                         'simaai-scheduler-pcie-host (= 1.0)',
                         'simaai-scheduler-pcie-host-dev (= 1.0)']):
                    _, packages = delivery.verify(root, runtime_only=runtime_only)
                expected = ['scheduler.deb']
                if not runtime_only: expected.append('scheduler-dev.deb')
                expected.append(runtime.name)
                if not runtime_only: expected.append(development.name)
                self.assertEqual([Path(path).name for path in packages], expected)
                provider.check_ownership.assert_called_once()

    def test_staging_delegates_provider_and_extracts_only_core_pair(self):
        selection = {'distro': 'ubuntu24', 'architecture': 'amd64', 'commit': 'a' * 40}
        packages = ['/dist/scheduler.deb', '/dist/scheduler-dev.deb',
                    '/dist/core.deb', '/dist/core-dev.deb']
        with tempfile.TemporaryDirectory() as temp:
            destination = Path(temp) / 'stage'
            with patch.object(delivery, 'verify', return_value=(selection, packages)), \
                 patch.object(delivery, 'load_provider') as provider, \
                 patch.object(delivery.subprocess, 'run') as extract:
                self.assertEqual(delivery.stage('/dist', destination), selection)
                provider.return_value.stage.assert_called_once_with(
                    Path('/dist'), destination, 'ubuntu24', 'amd64', 'a' * 40)
                self.assertEqual([call.args[0] for call in extract.call_args_list], [
                    ['dpkg-deb', '-x', '/dist/core.deb', str(destination)],
                    ['dpkg-deb', '-x', '/dist/core-dev.deb', str(destination)]])
                provider.return_value.preflight_installed.assert_not_called()
            destination.mkdir()
            with patch.object(delivery, 'verify', return_value=(selection, packages)), \
                 patch.object(delivery, 'load_provider') as provider:
                with self.assertRaisesRegex(ValueError, 'fresh private destination'):
                    delivery.stage('/dist', destination)
                provider.assert_not_called()

    def test_ci_stages_packages_but_hardware_still_installs(self):
        workflow = (ROOT.parent / '.github/workflows/vulcan-ci.yml').read_text()
        smoke = workflow.split('- name: Smoke test staged PCIe host packages and wheel', 1)[1]
        smoke = smoke.split('- name: Configure AWS credentials', 1)[0]
        self.assertIn('pcie_delivery.py stage', smoke)
        self.assertIn('-DCMAKE_PREFIX_PATH="${prefix}/usr"', smoke)
        self.assertIn('mv "${prefix}" "${prefix}-relocated"', smoke)
        self.assertIn('import pyneatpcie', smoke)
        self.assertNotIn('install_pciehost.sh', smoke)
        hardware = workflow.split('  test-pciehost-hardware:', 1)[1]
        self.assertIn('./scripts/ci/pciehost/run_ctest.sh', hardware)

    def test_exact_cmake_dependency_guard(self):
        source = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('simaai-scheduler-pcie-host (= ${SIMAPCIE_SCHEDULER_PACKAGE_VERSION})', source)
        self.assertIn('simaai-scheduler-pcie-host-dev (= ${SIMAPCIE_SCHEDULER_PACKAGE_VERSION})', source)
        self.assertIn('NativePackageGuard.cmake', source)
        self.assertIn('CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14', source)

    def test_acquisition_selects_exact_native_profile_and_stages_verified_commit(self):
        commit = 'a' * 40
        for distro in ('ubuntu22', 'ubuntu24'):
            for arch in ('amd64', 'arm64'):
                with self.subTest(distro=distro, arch=arch), tempfile.TemporaryDirectory() as temp:
                    root = Path(temp) / 'selected'
                    candidate = root / 'candidate'

                    def acquire(*args, **kwargs):
                        # The transport seam represents an already hash-verified
                        # publication; the real entry point must stage its commit.
                        candidate.mkdir()
                        (candidate / 'host-candidate.json').write_text('{}')
                        (candidate / 'native_artifacts.py').write_text(
                            'import hashlib, json\n'
                            'def sha(path):\n'
                            '    return hashlib.sha256(path.read_bytes()).hexdigest()\n'
                            'def stage(candidate, prefix, distro, arch, commit):\n'
                            '    prefix.mkdir()\n'
                            '    (prefix / "arguments.json").write_text(json.dumps([distro, arch, commit]))\n'
                            '    return {"packages": {"runtime": {"payload": {"control": {"Version": "0.1.0"}}}}}\n')
                        return {'commit': commit}

                    argv = ['acquire_scheduler.py', '--cli', 'test-cli', '--branch', 'feature/native',
                            '--spec', 'latest', '--environment', 'production', '--distro', distro,
                            '--arch', arch, '--output', str(root)]
                    with patch.object(sys, 'argv', argv), patch.object(sys, 'stdout', io.StringIO()), \
                         patch.object(acquisition, 'acquire', side_effect=acquire) as transport:
                        acquisition.main()
                    transport.assert_called_once_with(
                        'test-cli', 'scheduler', 'feature/native', 'latest', 'production', candidate,
                        package_path=f'pcie-host-{distro}-{arch}')
                    self.assertEqual(json.loads((root / 'prefix/arguments.json').read_text()),
                                     [distro, arch, commit])
                    selection = json.loads((root / 'selection.json').read_text())
                    self.assertEqual((selection['distro'], selection['architecture'], selection['commit']),
                                     (distro, arch, commit))
                    self.assertEqual(selection['package_version'], '0.1.0')

    def test_selected_environment_reaches_native_acquisition(self):
        source = (ROOT / 'build.sh').read_text()
        parser = source.split('while [[ $# -gt 0 ]]; do', 1)[1].split('\ndone\n\n# Refuse inherited', 1)[0]
        parser = 'while [[ $# -gt 0 ]]; do' + parser + '\ndone\n'
        for environment in ('production', 'staging', 'dev'):
            with self.subTest(environment=environment):
                command = 'set -eu; SCHEDULER_ENVIRONMENT=production; ' + parser + 'printf "%s" "$SCHEDULER_ENVIRONMENT"'
                result = subprocess.run(['bash', '-c', command, 'parser-test', '--scheduler-environment', environment],
                                        check=True, text=True, capture_output=True)
                self.assertEqual(result.stdout, environment)
        self.assertIn('--environment "$SCHEDULER_ENVIRONMENT"', source)

    def test_candidate_hash_checked_before_provider_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); candidate = root / 'candidate'; candidate.mkdir()
            (candidate / 'host-candidate.json').write_text('{}')
            selection = root / 'selection.json'
            selection.write_text(json.dumps({'candidate': str(candidate), 'candidate_sha256': '0' * 64}))
            with patch.object(delivery, 'load_provider') as provider:
                with self.assertRaises(ValueError): delivery.compose(selection, root / 'dist')
                provider.assert_not_called()

    def test_composition_preserves_provider_bytes_and_ownership(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); candidate = root / 'candidate'; candidate.mkdir()
            (candidate / 'native_artifacts.py').write_text('# producer helper test seam\n')
            helper = candidate / 'native_artifacts.py'
            (candidate / 'host-candidate.json').write_text(json.dumps({'resources': {'native_artifacts.py': {'sha256': delivery.sha(helper), 'size': helper.stat().st_size}}}))
            (candidate / 'platform.deb').write_bytes(b'authentic bytes represented by a unit seam')
            resources = {'platform.deb': {'sha256': delivery.sha(candidate / 'platform.deb'), 'size': (candidate / 'platform.deb').stat().st_size}}
            selection = {'schema_version': 1, 'profile': 'pcie-host', 'candidate': str(candidate),
                         'candidate_sha256': delivery.sha(candidate / 'host-candidate.json'), 'distro': 'ubuntu24',
                         'architecture': 'amd64', 'branch': 'feature/native', 'commit': '1' * 40, 'package_version': '1.0'}
            path = root / 'selection.json'; path.write_text(json.dumps(selection))
            with patch.object(delivery, 'load_provider') as provider:
                provider.return_value.verify.return_value = {'resources': resources}
                delivery.compose(path, root / 'out')
                provider.return_value.verify.assert_called_once_with(candidate, 'ubuntu24', 'amd64', '1' * 40)
            self.assertEqual((root / 'out/platform.deb').read_bytes(), (candidate / 'platform.deb').read_bytes())
            self.assertFalse((root / 'out/usr').exists())
            (root / 'out/platform.deb').write_bytes(b'collision')
            with patch.object(delivery, 'load_provider') as provider:
                provider.return_value.verify.return_value = {'resources': resources}
                with self.assertRaises(ValueError): delivery.compose(path, root / 'out')


if __name__ == '__main__': unittest.main()
