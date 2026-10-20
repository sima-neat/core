#!/usr/bin/env python3
"""Core composition tests use a provider seam, not copied Scheduler policy."""
import importlib.util
import io
import json
import subprocess
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

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
        self.assertEqual(source.count('run_sudo apt "${apt_install_opts[@]}"'), 1)
        self.assertLess(source.index('package_list="$(python3'), source.index('run_sudo apt "${apt_install_opts[@]}"'))

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
