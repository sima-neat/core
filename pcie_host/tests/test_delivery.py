#!/usr/bin/env python3
"""Core composition tests use a provider seam, not copied Scheduler policy."""
import ast
import importlib.util
import io
import json
import os
import shutil
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
spec = importlib.util.spec_from_file_location(
    'cpu_package_tests', ROOT.parent / 'scripts/ci/pciehost/run_cpu_package_tests.py')
cpu_tests = importlib.util.module_from_spec(spec); spec.loader.exec_module(cpu_tests)
spec = importlib.util.spec_from_file_location(
    'cpu_package_preparation', ROOT.parent / 'scripts/ci/pciehost/prepare_cpu_package.py')
cpu_prepare = importlib.util.module_from_spec(spec); spec.loader.exec_module(cpu_prepare)


class Delivery(unittest.TestCase):
    def test_dependency_bootstrap_reuses_existing_host_privilege_helper(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); commands = root / 'bin'; commands.mkdir()
            log = root / 'commands.log'
            (commands / 'id').write_text('#!/bin/bash\nprintf "1000\\n"\n')
            (commands / 'sudo').write_text('''#!/bin/bash
printf '%s\\n' "$*" >> "$COMMAND_LOG"
if [[ "$1" == -n ]]; then exit "$REQUIRE_PASSWORD"; fi
if [[ "$1" == -S ]]; then
  IFS= read -r password
  [[ "$password" == "$SUDO_PASSWORD" ]] || exit 99
  printf 'existing host secret received on stdin\\n' >> "$COMMAND_LOG"
fi
''')
            for path in commands.iterdir():
                path.chmod(0o755)
            script = '''set -euo pipefail
source scripts/ci/pciehost/common.sh
[[ ! -e "$COMMAND_LOG" ]]
run_host_sudo ./pcie_host/build.sh --python --install-deps-only
'''
            for password_required in (False, True):
                with self.subTest(password_required=password_required):
                    log.unlink(missing_ok=True)
                    env = {'PATH': str(commands) + ':/usr/bin:/bin', 'HOME': str(root),
                           'GITHUB_WORKSPACE': str(ROOT.parent), 'COMMAND_LOG': str(log),
                           'REQUIRE_PASSWORD': str(int(password_required)),
                           'SUDO_PASSWORD': 'test-only-existing-host-secret'}
                    result = subprocess.run(['bash', '-c', script], cwd=ROOT.parent,
                                            env=env, text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    expected = ['-n true']
                    command = './pcie_host/build.sh --python --install-deps-only'
                    expected += (['-S -p  ' + command, 'existing host secret received on stdin']
                                 if password_required else [command])
                    self.assertEqual(log.read_text().splitlines(), expected)

    def test_dependency_only_mode_uses_existing_bootstrap_and_stops_before_build(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); commands = root / 'bin'; commands.mkdir()
            log = root / 'commands.log'
            for name in ('apt-get', 'apt-cache', 'gcc-14', 'g++-14', 'sudo',
                         'cmake', 'python3', 'sima-cli', 'git', 'rm'):
                script = commands / name
                script.write_text('''#!/bin/bash
printf '%s %s\\n' "${0##*/}" "$*" >> "$COMMAND_LOG"
case "${0##*/}" in
  apt-get|apt-cache) exit 0 ;;
  gcc-14|g++-14) printf '%s\\n' "${FIXTURE_GCC_VERSION:-14}" ;;
  sudo) exec "$@" ;;
  *) echo 'unexpected post-bootstrap command' >&2; exit 99 ;;
esac
''')
                script.chmod(0o755)
            preserved = root / 'existing-build'; preserved.mkdir()
            (preserved / 'keep').write_text('must not clean in dependency-only mode')
            env = {'PATH': str(commands) + ':/usr/bin:/bin', 'HOME': str(root),
                   'COMMAND_LOG': str(log)}
            argv = ['bash', str(ROOT / 'build.sh'), '--python', '--install-deps-only',
                    '--clean', '--build-dir', str(preserved)]
            result = subprocess.run(argv, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((preserved / 'keep').is_file())
            calls = log.read_text()
            self.assertIn('apt-get update', calls)
            self.assertIn('gcc-14 g++-14', calls)
            self.assertIn('python3-venv', calls)
            self.assertIn('gcc-14 -dumpversion', calls)
            self.assertIn('g++-14 -dumpversion', calls)
            executed = {line.split()[0] for line in calls.splitlines()}
            self.assertFalse(executed.intersection({'cmake', 'python3', 'sima-cli', 'git', 'rm'}))
            self.assertNotIn('dkms', calls)
            self.assertNotIn('simaai-pcie', calls)
            result = subprocess.run(argv, env={**env, 'FIXTURE_GCC_VERSION': '13'},
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            log.unlink()
            result = subprocess.run(argv, env={**env, 'CXXFLAGS': '-target-sdk-flag'},
                                    text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(log.exists(), 'SDK environment must fail before dependency installation')

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

    def test_ci_stages_packages_without_device_installation(self):
        workflow = (ROOT.parent / '.github/workflows/vulcan-ci.yml').read_text()
        smoke = workflow.split('- name: Smoke test staged PCIe host packages and wheel', 1)[1]
        smoke = smoke.split('- name: Configure AWS credentials', 1)[0]
        self.assertIn('pcie_delivery.py stage', smoke)
        self.assertIn('-DCMAKE_PREFIX_PATH="${prefix}/usr"', smoke)
        self.assertIn('mv "${prefix}" "${prefix}-relocated"', smoke)
        self.assertIn('import pyneatpcie', smoke)
        self.assertNotIn('install_pciehost.sh', smoke)
        hardware = workflow.split('  test-pciehost-hardware:', 1)[1]
        self.assertIn('scripts/ci/pciehost/prepare_cpu_package.py', hardware)
        self.assertNotIn('./scripts/ci/pciehost/install_host_package.sh', hardware)

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


class CpuPackagePreparation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / 'published'; self.source.mkdir()
        self.commit = 'a' * 40
        self.branch = 'feature/cpu'
        self.selector = f'core/pciehost/ubuntu22/amd64@{self.branch}:{self.commit[:12]}'
        self.selection = {'distro': 'ubuntu22', 'architecture': 'amd64', 'branch': self.branch,
                          'resources': {'native_artifacts.py': {'sha256': 'fixture-only'}}}
        names = ['native_artifacts.py', 'scheduler-native-selection.json', 'pcie_delivery.py',
                 'install_pciehost.sh', 'sima-pcie-host_1_amd64.deb', 'sima-pcie-host-dev_1_amd64.deb',
                 'sima-pcie-host-1-extras.tar.gz', 'pyneatpcie-1.whl']
        for name in names: (self.source / name).write_bytes(b'not executable fixture data')
        (self.source / 'scheduler-native-selection.json').write_text(json.dumps(self.selection))
        self.metadata = {
            'name': 'gh:sima-neat/pciehost/ubuntu22/amd64', 'repository': 'sima-neat/core',
            'branch': self.branch, 'commit': self.commit, 'commit_folder': self.commit[:12],
            'artifact': {'repository': 'core', 'package_path': 'pciehost/ubuntu22/amd64',
                         'distro': 'ubuntu22', 'arches': ['amd64']},
            'resources': names, 'resources-checksum': {
                name: cpu_prepare.digest(self.source / name) for name in names}}
        self.selected = {'repository': 'core', 'package_path': 'pciehost/ubuntu22/amd64',
                         'ref': self.branch, 'environment': 'production',
                         'requested_spec': self.commit[:12], 'resolved_spec': self.commit[:12],
                         'base_url': 'https://artifacts.neat.sima.ai',
                         'metadata_url': f'https://artifacts.neat.sima.ai/core/feature%252Fcpu/{self.commit[:12]}/pciehost/ubuntu22/amd64/metadata.json'}
        self.host_distro = 'ubuntu22'; self.host_arch = 'amd64'
        self.calls = []; self.mutation = None
        def output(argv, **kwargs):
            if argv[0] == 'bash': return self.host_distro + '\n'
            if argv[0] == 'dpkg': return self.host_arch + '\n'
            self.calls.append(argv)
            return 'CLI notice\n' + json.dumps(self.selected)
        original_run = subprocess.run
        def run(argv, **kwargs):
            if argv[0] == 'git': return original_run(argv, **kwargs)
            self.calls.append(argv)
            destination = Path(argv[argv.index('-d') + 1])
            for name in self.metadata['resources']:
                path = self.source / name
                if path.exists(): shutil.copy2(path, destination / name, follow_symlinks=False)
            if self.mutation: self.mutation(destination)
            return subprocess.CompletedProcess(argv, 0)
        def fetch(url, target):
            self.assertEqual(url, self.selected['metadata_url'])
            Path(target).write_text(json.dumps(self.metadata))
        for target, method, replacement in (
                (cpu_prepare.subprocess, 'check_output', output),
                (cpu_prepare.subprocess, 'run', run),
                (cpu_prepare, 'download', fetch)):
            patcher = patch.object(target, method, side_effect=replacement)
            patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(cpu_prepare, 'stage')
        self.stage = patcher.start(); self.addCleanup(patcher.stop)

    def acquire(self):
        return cpu_prepare.prepare('fixture-cli', self.branch, self.commit, 'prod',
                                   self.root / 'distribution', self.root / 'prefix')

    def test_exact_native_download_then_trusted_private_stage_without_install(self):
        result = self.acquire()
        self.assertFalse(result['hardware_qualified'])
        self.assertEqual(result['resources'], 8)
        self.assertEqual(self.calls, [
            ['fixture-cli', 'neat', 'download', self.selector, '--env', 'production', '--json'],
            ['fixture-cli', 'neat', 'download', self.selector, '--env', 'production',
             '-d', str(self.root / 'distribution')]])
        self.stage.assert_called_once_with(self.root / 'distribution', self.root / 'prefix')

    def test_wrong_cli_identity_never_downloads_or_stages(self):
        for key, value in [('repository', 'scheduler'), ('package_path', 'pciehost/ubuntu24/amd64'),
                           ('ref', 'develop'), ('resolved_spec', 'latest'), ('requested_spec', 'b' * 12),
                           ('environment', 'staging'), ('base_url', 'https://example.org'),
                           ('metadata_url', 'https://example.org/metadata.json')]:
            with self.subTest(key=key), patch.dict(self.selected, {key: value}):
                with self.assertRaisesRegex(ValueError, 'different native Core publication'): self.acquire()
        self.stage.assert_not_called()
        self.assertFalse((self.root / 'distribution').exists())

    def test_incompatible_host_never_resolves_native_publication(self):
        for distro, arch in [('elxr', 'amd64'), ('ubuntu22', 'arm64')]:
            with self.subTest(distro=distro, arch=arch):
                self.host_distro, self.host_arch = distro, arch
                with self.assertRaisesRegex(ValueError, 'original Ubuntu x64'): self.acquire()
        self.assertEqual(self.calls, [])
        self.stage.assert_not_called()

    def test_full_commit_collision_tail_is_checked_before_download(self):
        self.metadata['commit'] = self.commit[:12] + 'b' * 28
        with self.assertRaisesRegex(ValueError, 'selected Core commit/host'): self.acquire()
        self.assertEqual(len(self.calls), 1)
        self.stage.assert_not_called()

    def test_changed_downloaded_helper_rejected_before_import(self):
        self.mutation = lambda directory: (directory / 'native_artifacts.py').write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'differs from metadata'): self.acquire()
        self.stage.assert_not_called()

    def test_symlink_download_rejected_before_import(self):
        def mutate(directory):
            helper = directory / 'native_artifacts.py'; helper.unlink()
            helper.symlink_to(self.source / 'native_artifacts.py')
        self.mutation = mutate
        with self.assertRaisesRegex(ValueError, 'differs from metadata'): self.acquire()
        self.stage.assert_not_called()

    def test_changed_metadata_rejected_before_import(self):
        self.mutation = lambda directory: (directory / 'metadata.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'differs from metadata'): self.acquire()
        self.stage.assert_not_called()

    def test_missing_payload_rejected_before_import(self):
        self.mutation = lambda directory: (directory / 'pyneatpcie-1.whl').unlink()
        with self.assertRaisesRegex(ValueError, 'differs from metadata'): self.acquire()
        self.stage.assert_not_called()

    def test_unsafe_resource_rejected_before_cli_download(self):
        self.metadata['resources'] = ['../escape']
        self.metadata['resources-checksum'] = {'../escape': 'a' * 64}
        with self.assertRaisesRegex(ValueError, 'Unsafe published resource'): self.acquire()
        self.assertEqual(len(self.calls), 1)
        self.stage.assert_not_called()

    def test_incomplete_original_inventory_is_not_a_smaller_cpu_package(self):
        name = 'pyneatpcie-1.whl'
        self.metadata['resources'].remove(name); del self.metadata['resources-checksum'][name]
        with self.assertRaisesRegex(ValueError, 'Expected one native Core resource'): self.acquire()
        self.stage.assert_not_called()

    def test_provider_rejection_is_not_replaced_by_ambient_packages(self):
        self.stage.side_effect = ValueError('actual provider rejected')
        with self.assertRaisesRegex(ValueError, 'actual provider rejected'): self.acquire()

    def test_existing_output_is_not_reused(self):
        (self.root / 'distribution').mkdir()
        with self.assertRaises(FileExistsError): self.acquire()
        self.stage.assert_not_called()


class CpuPackageCoverage(unittest.TestCase):
    def test_explicit_models_require_no_authentication_or_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            environment = {}
            for name in cpu_tests.MODEL_VARIABLES:
                path = root / name; path.write_bytes(b'fixture')
                environment[name] = str(path)
            with patch.object(cpu_tests.subprocess, 'run') as run:
                cpu_tests.resolve_models(ROOT.parent, root, environment)
                run.assert_not_called()

    def test_missing_models_require_bounded_existing_auth_check_before_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); environment = {}
            lines = []
            for name in cpu_tests.MODEL_VARIABLES:
                path = root / name; path.write_bytes(b'fixture'); lines.append(f'{name}={path}')
            (root / 'assets.env').write_text('\n'.join(lines))
            with patch.object(cpu_tests.subprocess, 'run') as run:
                cpu_tests.resolve_models(ROOT.parent, root, environment)
            self.assertEqual([call.args[0] for call in run.call_args_list], [
                ['bash', str(ROOT.parent / 'scripts/ci/check_sima_cli_session.sh')],
                ['bash', str(ROOT.parent / 'scripts/ci/pciehost/resolve_assets.sh')]])
            self.assertTrue(all(call.kwargs['check'] for call in run.call_args_list))
            self.assertEqual(set(environment), set(cpu_tests.MODEL_VARIABLES))

    def test_missing_authentication_fails_before_model_download(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(cpu_tests.subprocess, 'run',
                              side_effect=subprocess.CalledProcessError(1, ['auth-check'])) as run:
                with self.assertRaises(subprocess.CalledProcessError):
                    cpu_tests.resolve_models(ROOT.parent, Path(temp), {})
            run.assert_called_once()
            self.assertEqual(run.call_args.args[0][1], str(ROOT.parent / 'scripts/ci/check_sima_cli_session.sh'))

    def test_selection_preserves_cpu_modules_and_only_two_mixed_file_cases(self):
        self.assertEqual(cpu_tests.CPU_TESTS, (
            'test_api_surface.py', 'test_genai_tutorials.py', 'test_genai_values.py',
            'test_metadata.py', 'test_wheel_metadata.py',
            'test_hardware_smoke.py::test_async_test_runner_cancels_blocked_peer',
            'test_hardware_smoke.py::test_metadata_yolov8'))
        for node in cpu_tests.CPU_TESTS:
            self.assertTrue((ROOT / 'python/tests' / node.split('::')[0]).is_file())

    def test_current_selected_source_preserves_all_36_audited_cpu_cases(self):
        # Audit today's coverage, without a fixed-count gate in the CI runner.
        count = 0
        for node in cpu_tests.CPU_TESTS:
            filename, _, selected_function = node.partition('::')
            tree = ast.parse((ROOT / 'python/tests' / filename).read_text())
            functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)
                         and item.name.startswith('test_')
                         and (not selected_function or item.name == selected_function)]
            self.assertTrue(functions, node)
            for function in functions:
                cases = 1
                for decorator in function.decorator_list:
                    if (isinstance(decorator, ast.Call)
                            and isinstance(decorator.func, ast.Attribute)
                            and decorator.func.attr == 'parametrize'):
                        cases *= len(ast.literal_eval(decorator.args[1]))
                count += cases
        self.assertEqual(count, 36)

    def test_workflow_keeps_cpu_coverage_on_develop_host_without_physical_actions(self):
        workflow = (ROOT.parent / '.github/workflows/vulcan-ci.yml').read_text()
        smoke = workflow.split('      - name: Smoke test staged PCIe host packages and wheel\n', 1)[1]
        smoke = smoke.split('\n      - name:', 1)[0]
        self.assertIn('pcie_host/dist/pyneatpcie-*.whl', smoke)
        self.assertNotIn('run_cpu_package_tests.py', smoke)
        self.assertNotIn('pytest', smoke)
        cpu = workflow.split('\n  test-pciehost-hardware:\n', 1)[1].split('\n  promote-latest-tag:', 1)[0]
        self.assertIn('name: Test PCIe Host CPU Coverage', cpu)
        self.assertIn('runs-on: [self-hosted, Linux, X64, pcie, 3.0.0-pcie-test]', cpu)
        self.assertIn('prepare_cpu_package.py', cpu)
        self.assertIn('--commit "${GITHUB_SHA}"', cpu)
        self.assertIn('"${python_venv}/bin/python" scripts/ci/pciehost/run_cpu_package_tests.py', cpu)
        self.assertIn('/usr/bin/python3 -m venv', cpu)
        self.assertIn('source scripts/ci/pciehost/common.sh', cpu)
        self.assertIn('run_host_sudo ./pcie_host/build.sh --python --install-deps-only', cpu)
        self.assertIn('SUDO_PASSWORD: ${{ secrets.PCIEHOST_SUDO_PASSWORD }}', cpu)
        self.assertEqual(cpu.count('secrets.'), 1)
        self.assertEqual(cpu.count('run_host_sudo '), 1)
        self.assertIn('LD_LIBRARY_PATH=', cpu)
        for forbidden in ('preflight.sh', 'pcie-setup.sh', 'prepare_card_prerequisites.sh',
                          'check_card_compatibility.sh', 'install_host_package.sh',
                          'install_host_test_deps.sh', 'install_card_runtime.sh', 'cleanup_queue.sh',
                          'verify_card_runtime.sh', 'prepare_pcie_genai_models.sh', 'run_ctest.sh',
                          'run_pytest.sh', 'DEVKIT_PASSWORD', 'SIMAPCIE_CARD_'):
            with self.subTest(physical_action=forbidden):
                self.assertNotIn(forbidden, cpu)
        self.assertEqual(workflow.count('scripts/ci/pciehost/run_cpu_package_tests.py'), 1)

    def test_models_are_required_even_when_original_tests_would_skip(self):
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / 'actual-model-path'
            model.write_bytes(b'fixture-only path validation, not a model oracle')
            environment = {name: str(model) for name in cpu_tests.MODEL_VARIABLES}
            cpu_tests.validate_models(environment)
            for name in cpu_tests.MODEL_VARIABLES:
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                    cpu_tests.validate_models({**environment, name: '/missing/model'})
            model.write_bytes(b'')
            with self.assertRaisesRegex(ValueError, 'required CPU metadata model'):
                cpu_tests.validate_models(environment)

    def test_installed_tutorials_must_match_current_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            extras = Path(temporary)
            target = extras / 'share/sima-pcie-host/tutorials/029_run_genai_over_pcie'
            target.mkdir(parents=True)
            for name in ('run_llm.py', 'run_vlm.py', 'run_whisper.py'):
                (target / name).write_bytes(
                    (ROOT / 'tutorials/029_run_genai_over_pcie' / name).read_bytes())
            cpu_tests.verify_tutorials(ROOT.parent, extras)
            (target / 'run_vlm.py').write_text('stale installed tutorial')
            with self.assertRaisesRegex(ValueError, 'run_vlm.py'):
                cpu_tests.verify_tutorials(ROOT.parent, extras)

    def test_collection_matches_executed_parameterized_inventory(self):
        collected = cpu_tests.collected_tests(
            'pcie_host/python/tests/test_api_surface.py::test_options_surface\n'
            'pcie_host/python/tests/test_metadata.py::test_load[BF16]\n\n2 tests collected\n')
        self.assertEqual(len(collected), 2)
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / 'report.xml'
            cases = ('<testcase classname="pcie_host.python.tests.test_api_surface" '
                     'name="test_options_surface"/>',
                     '<testcase classname="pcie_host.python.tests.test_metadata" '
                     'name="test_load[BF16]"/>')
            report.write_text('<testsuites><testsuite>' + ''.join(cases) + '</testsuite></testsuites>')
            cpu_tests.verify_python_report(collected, report)
            for changed in (cases[:1], (*cases, cases[0])):
                report.write_text('<testsuites><testsuite>' + ''.join(changed) + '</testsuite></testsuites>')
                with self.subTest(cases=len(changed)), self.assertRaisesRegex(ValueError, 'inventory'):
                    cpu_tests.verify_python_report(collected, report)
            for tag in ('skipped', 'failure', 'error'):
                report.write_text('<testsuites><testsuite>' + cases[0] +
                                 cases[1].replace('/>', f'><{tag}/></testcase>') +
                                 '</testsuite></testsuites>')
                with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'skip, failure or error'):
                    cpu_tests.verify_python_report(collected, report)

    def test_collection_never_accepts_empty_or_duplicate_inventory(self):
        node = 'pcie_host/python/tests/test_metadata.py::test_load'
        for output in ('no tests collected', node + '\n' + node):
            with self.subTest(output=output), self.assertRaisesRegex(ValueError, 'empty or duplicated'):
                cpu_tests.collected_tests(output)

    def test_parameter_ids_preserve_path_and_node_separators(self):
        for parameter in ('/nonexistent/neat-model', 'path/file.py::value[inner]'):
            with self.subTest(parameter=parameter), tempfile.TemporaryDirectory() as temporary:
                node = f'pcie_host/python/tests/test_genai_values.py::test_host_model_path_validation[{parameter}]'
                expected = f'pcie_host.python.tests.test_genai_values.test_host_model_path_validation[{parameter}]'
                collected = cpu_tests.collected_tests(node)
                self.assertEqual(collected, {expected})
                report = Path(temporary) / 'report.xml'
                report.write_text('<testsuite><testcase classname="pcie_host.python.tests.test_genai_values" '
                                  f'name="test_host_model_path_validation[{parameter}]"/></testsuite>')
                cpu_tests.verify_python_report(collected, report)

    def test_cpp_metadata_report_never_accepts_missing_extra_or_skipped_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / 'metadata.xml'
            case = '<testcase name="test_metadata"/>'
            report.write_text('<testsuite>' + case + '</testsuite>')
            cpu_tests.verify_metadata_report(report)
            for changed in ('', case * 2, '<testcase name="test_tensor_run"/>'):
                report.write_text('<testsuite>' + changed + '</testsuite>')
                with self.subTest(cases=changed), self.assertRaisesRegex(ValueError, 'inventory'):
                    cpu_tests.verify_metadata_report(report)
            for tag in ('skipped', 'failure', 'error'):
                report.write_text('<testsuite>' + case.replace('/>', f'><{tag}/></testcase>') + '</testsuite>')
                with self.subTest(tag=tag), self.assertRaisesRegex(ValueError, 'skip, failure or error'):
                    cpu_tests.verify_metadata_report(report)


if __name__ == '__main__': unittest.main()
