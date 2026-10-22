#!/usr/bin/env python3
"""Core composition tests use a provider seam, not copied Scheduler policy."""
import importlib.util
import io
import json
import os
import shutil
import subprocess
from pathlib import Path
import sys
import tarfile
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
spec = importlib.util.spec_from_file_location('native_publication', ROOT / 'scripts/verify_published.py')
publication = importlib.util.module_from_spec(spec); spec.loader.exec_module(publication)


class PublicationInventory(unittest.TestCase):
    def test_complete_inventory_and_each_required_role(self):
        for arch in ('amd64', 'arm64'):
            with tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                selection = {'architecture': arch, 'resources': {'provider.deb': {}}}
                names = ['provider.deb', 'scheduler-native-selection.json', 'pcie_delivery.py',
                         'install_pciehost.sh', f'sima-pcie-host_1_{arch}.deb',
                         f'sima-pcie-host-dev_1_{arch}.deb', 'core-extras.tar.gz', 'pyneatpcie-1.whl']
                for name in names: (directory / name).touch()
                delivery.verify_publication_inventory(directory, selection, names)
                for name in names:
                    with self.subTest(arch=arch, missing=name):
                        (directory / name).unlink()
                        with self.assertRaises(ValueError):
                            delivery.verify_publication_inventory(
                                directory, selection, [item for item in names if item != name])
                        (directory / name).touch()
                for duplicate in (f'sima-pcie-host_2_{arch}.deb', f'sima-pcie-host-dev_2_{arch}.deb',
                                  'other-extras.tar.gz', 'pyneatpcie-2.whl'):
                    with self.subTest(arch=arch, duplicate=duplicate):
                        (directory / duplicate).touch()
                        with self.assertRaisesRegex(ValueError, 'Expected one'):
                            delivery.verify_publication_inventory(directory, selection, names + [duplicate])
                        (directory / duplicate).unlink()
                for invalid in ([], names + [names[0]], names + ['foreign.deb']):
                    with self.subTest(arch=arch, invalid=invalid), self.assertRaises(ValueError):
                        delivery.verify_publication_inventory(directory, selection, invalid)
                with self.assertRaises(ValueError):
                    delivery.verify_publication_inventory(directory, {**selection, 'architecture': 'other'}, names)

    def test_publisher_authentication_shared_inventory_and_variant_checks(self):
        for mutation in ('complete', 'identity', 'hash', 'duplicate', 'missing', 'extra',
                         'scheduler-identity', 'producer', 'variant'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary); source = root / 'source'; source.mkdir()
                selection = {'architecture': 'amd64', 'distro': 'ubuntu22', 'branch': 'feature/cpu',
                             'resources': {'native_artifacts.py': {}}}
                names = ['native_artifacts.py', 'scheduler-native-selection.json', 'pcie_delivery.py',
                         'install_pciehost.sh', 'sima-pcie-host_1_amd64.deb',
                         'sima-pcie-host-dev_1_amd64.deb', 'core-extras.tar.gz', 'pyneatpcie-1.whl']
                if mutation == 'missing': names.remove('pyneatpcie-1.whl')
                if mutation == 'extra': names.append('foreign.deb')
                for name in names: (source / name).write_text('authenticated transport fixture')
                hashes = {name: delivery.sha(source / name) for name in names}
                data = {'commit': 'a' * 40, 'branch': 'feature/cpu',
                        'artifact': {'distro': 'ubuntu22', 'arches': ['amd64']},
                        'resources': names[:], 'resources-checksum': hashes}
                if mutation == 'identity': data['commit'] = 'b' * 40
                if mutation == 'hash': (source / names[0]).write_text('changed bytes')
                if mutation == 'duplicate': data['resources'].append(names[0])
                (source / 'metadata.json.pending').write_text(json.dumps(data))
                for name, resource in [('metadata-extras.json', 'core-extras.tar.gz'),
                                       ('metadata-pyneatpcie.json', 'pyneatpcie-1.whl')]:
                    subset = {'commit': 'b' * 40 if mutation == 'variant' else 'a' * 40,
                              'resources': [resource], 'resources-checksum': {resource: hashes.get(resource)},
                              'installation': {'script': ':'}}
                    (source / (name + '.pending')).write_text(json.dumps(subset))
                if mutation == 'scheduler-identity': selection['branch'] = 'other'
                output = root / 'download'; events = []
                def download(argv, **kwargs):
                    self.assertEqual(argv[:3], ['aws', 's3', 'cp'])
                    shutil.copy2(source / argv[3].rsplit('/', 1)[1], argv[4])
                def verify(directory):
                    events.append('producer')
                    for name, digest in hashes.items():
                        self.assertEqual(delivery.sha(Path(directory) / name), digest)
                    if mutation == 'producer': raise ValueError('producer rejected')
                    return selection, []
                def inventory(directory, selected, resources):
                    events.append('inventory')
                    return delivery.verify_publication_inventory(directory, selected, resources)
                argv = ['verify_published.py', '--bucket', 'fixture', '--prefix', 'native',
                        '--distro', 'ubuntu22', '--arch', 'amd64', '--commit', 'a' * 40,
                        '--output', str(output)]
                with patch.object(sys, 'argv', argv), patch.object(publication.subprocess, 'run', side_effect=download), \
                     patch.object(publication, 'verify', side_effect=verify), \
                     patch.object(publication, 'verify_publication_inventory', side_effect=inventory), \
                     patch('sys.stdout', new_callable=io.StringIO) as report:
                    if mutation == 'complete':
                        publication.main()
                        self.assertEqual(json.loads(report.getvalue())['resources'], len(names))
                    else:
                        with self.assertRaises(ValueError): publication.main()
                if mutation in ('identity', 'hash', 'duplicate'):
                    self.assertEqual(events, [], 'authentication must precede both checkers')
                elif mutation in ('producer', 'scheduler-identity'):
                    self.assertEqual(events, ['producer'])
                else:
                    self.assertEqual(events, ['producer', 'inventory'])


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
            for name in ('apt-get', 'apt-cache', 'gcc-14', 'g++-14', 'add-apt-repository', 'sudo',
                         'cmake', 'python3', 'sima-cli', 'git', 'rm'):
                script = commands / name
                script.write_text('''#!/bin/bash
printf '%s %s\\n' "${0##*/}" "$*" >> "$COMMAND_LOG"
case "${0##*/}" in
  apt-get|apt-cache) exit 0 ;;
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
            self.assertIn('apt-get install -y --no-install-recommends build-essential', calls)
            self.assertIn('python3-venv', calls)
            self.assertNotIn('gcc-14', calls)
            self.assertNotIn('g++-14', calls)
            self.assertNotIn('add-apt-repository', calls)
            self.assertNotIn('software-properties-common', calls)
            executed = {line.split()[0] for line in calls.splitlines()}
            self.assertFalse(executed.intersection({'cmake', 'python3', 'sima-cli', 'git', 'rm'}))
            self.assertNotIn('dkms', calls)
            self.assertNotIn('simaai-pcie', calls)
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

    def test_exact_cmake_dependency_guard(self):
        source = (ROOT / 'CMakeLists.txt').read_text()
        self.assertIn('simaai-scheduler-pcie-host (= ${SIMAPCIE_SCHEDULER_PACKAGE_VERSION})', source)
        self.assertIn('simaai-scheduler-pcie-host-dev (= ${SIMAPCIE_SCHEDULER_PACKAGE_VERSION})', source)
        self.assertIn('NativePackageGuard.cmake', source)
        self.assertIn('set(CMAKE_CXX_STANDARD 20)', source)
        self.assertIn('set(CMAKE_CXX_STANDARD_REQUIRED ON)', source)
        self.assertNotIn('libstdc++6 (>= 14)', source)

    def test_default_native_compiler_reaches_required_scheduler_component(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            package = root / 'scheduler'; package.mkdir()
            (package / 'SimaSchedulerConfig.cmake').write_text(
                'if(NOT SimaScheduler_FIND_COMPONENTS STREQUAL "pcie_host;pcie_service_headers")\n'
                '  message(FATAL_ERROR "wrong Scheduler component selection")\n'
                'endif()\n'
                'message(FATAL_ERROR "native compiler reached required Scheduler components")\n')
            # Exercise real CMake/compiler discovery without inherited SDK flags.
            # The component seam stops configuration before any provider loading.
            result = subprocess.run(
                ['cmake', '-S', str(ROOT), '-B', str(root / 'build'),
                 '-DSimaScheduler_DIR=' + str(package)],
                env={'PATH': '/usr/bin:/bin', 'HOME': str(root)},
                text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('native compiler reached required Scheduler components',
                          result.stdout + result.stderr)
            source = (ROOT / 'build.sh').read_text()
            self.assertNotIn('-DCMAKE_CXX_COMPILER=', source)
            self.assertNotIn('ppa:ubuntu-toolchain-r/test', source)

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

    def test_complete_native_transport_does_not_require_git(self):
        with patch.dict(os.environ, {'PATH': str(self.root)}):
            result = self.acquire()
        self.assertEqual(result['status'], 'STAGED_CPU_PACKAGE')
        self.stage.assert_called_once_with(self.root / 'distribution', self.root / 'prefix')
        self.assertEqual(len(self.calls), 2)

    def test_exact_native_download_then_trusted_private_stage_without_install(self):
        def inventory(directory, selection, resources):
            self.stage.assert_not_called()
            for name, expected in self.metadata['resources-checksum'].items():
                self.assertEqual(cpu_prepare.digest(Path(directory) / name), expected)
            return delivery.verify_publication_inventory(directory, selection, resources)
        with patch.object(cpu_prepare, 'verify_publication_inventory', side_effect=inventory) as shared:
            result = self.acquire()
        shared.assert_called_once_with(self.root / 'distribution', self.selection, self.metadata['resources'])
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
        with patch.object(cpu_prepare, 'verify_publication_inventory') as shared:
            with self.assertRaisesRegex(ValueError, 'differs from metadata'): self.acquire()
        shared.assert_not_called()
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

    def test_shared_inventory_rejection_precedes_producer_staging(self):
        with patch.object(cpu_prepare, 'verify_publication_inventory',
                          side_effect=ValueError('common inventory rejected')) as shared:
            with self.assertRaisesRegex(ValueError, 'common inventory rejected'): self.acquire()
        shared.assert_called_once()
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

    def test_cpu_runner_uses_shared_accounting_with_real_host_processes(self):
        for outcome in ('pass', 'skip', 'missing', 'ctest_skip', 'ctest_failure', 'ctest_missing',
                        'wrong_binary', 'invalid_native'):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary)
                (workspace / 'tools').symlink_to(ROOT.parent / 'tools', target_is_directory=True)
                sources = workspace / 'pcie_host/python/tests'
                sources.mkdir(parents=True)
                record = ('from pathlib import Path\n'
                          f'LOG = Path({str(workspace / "executed")!r})\n'
                          'def record():\n'
                          '    with LOG.open("a") as stream: stream.write("executed\\n")\n')
                expected_nodes = set()
                for node in cpu_tests.CPU_TESTS:
                    name, _, function = node.partition('::')
                    with (sources / name).open('a') as source:
                        source.write(record + f'def {function or "test_cpu"}(): record()\n')
                    expected_nodes.add(
                        f'{(sources / name).relative_to(workspace).as_posix()}::{function or "test_cpu"}')
                with (sources / 'test_hardware_smoke.py').open('a') as source:
                    source.write('def test_requires_card(): assert False, "hardware case must stay unselected"\n')
                parameters = ['/nonexistent/neat-model', 'path/file.py::value[inner]']
                (sources / 'test_genai_values.py').write_text(
                    record + 'import pytest\n' +
                    f'@pytest.mark.parametrize("value", {parameters!r})\n'
                    'def test_values(value): record()\n')
                values_node = (sources / 'test_genai_values.py').relative_to(workspace).as_posix()
                expected_nodes.remove(f'{values_node}::test_cpu')
                expected_nodes.update(f'{values_node}::test_values[{value}]' for value in parameters)
                if outcome == 'skip':
                    (sources / 'test_api_surface.py').write_text(
                        'import pytest\ndef test_cpu(): pytest.skip("required CPU fixture")\n')
                elif outcome == 'missing':
                    (sources / 'conftest.py').write_text(
                        'def pytest_runtest_protocol(item, nextitem):\n'
                        '    if item.path.name == "test_api_surface.py": return True\n')

                payload = workspace / 'payload'
                tutorials = Path('pcie_host/tutorials/029_run_genai_over_pcie')
                for name in ('run_llm.py', 'run_vlm.py', 'run_whisper.py'):
                    for directory in (workspace / tutorials,
                                      payload / 'share/sima-pcie-host/tutorials/029_run_genai_over_pcie'):
                        directory.mkdir(parents=True, exist_ok=True)
                        (directory / name).write_text('# selected tutorial fixture\n')
                installed_tests = payload / 'lib/sima-pcie-host/tests'
                installed_tests.mkdir(parents=True)
                binary = installed_tests / 'test_metadata'
                shutil.copy2('/usr/bin/false' if outcome == 'ctest_failure' else '/usr/bin/true', binary)
                if outcome == 'invalid_native':
                    binary.write_text('not a native ELF executable\n')
                command = ('/usr/bin/true' if outcome == 'wrong_binary' else './test_metadata')
                name = 'other' if outcome == 'ctest_missing' else 'test_metadata'
                registration = f'add_test({name} "{command}")\n'
                if outcome == 'ctest_skip':
                    registration += 'set_tests_properties(test_metadata PROPERTIES SKIP_RETURN_CODE 0)\n'
                (installed_tests / 'CTestTestfile.cmake').write_text(registration)
                distribution = workspace / 'distribution'
                distribution.mkdir()
                with tarfile.open(distribution / 'sima-pcie-host-fixture-Linux-cpu-extras.tar.gz', 'w:gz') as archive:
                    archive.add(payload, arcname='.')
                model = workspace / 'model'; model.write_bytes(b'explicit offline fixture')
                environment = {name: str(model) for name in cpu_tests.MODEL_VARIABLES}
                environment['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
                work = workspace / 'results'
                argv = ['run_cpu_package_tests.py', '--workspace', str(workspace),
                        '--distribution', str(distribution), '--work-directory', str(work)]
                # Only package imports are a seam: pytest, CTest, archive extraction,
                # native ELF inspection and both shared accountants execute normally.
                package = Mock(__file__=str(Path(sys.prefix) / 'lib/pyneatpcie/__init__.py'))
                with patch.object(sys, 'argv', argv), patch.object(sys, 'base_prefix', '/fixture-base'), \
                     patch.object(cpu_tests.importlib, 'import_module', return_value=package), \
                     patch.dict(os.environ, environment):
                    if outcome == 'pass':
                        cpu_tests.main()
                        selection = json.loads((work / 'python-cpu/selection.json').read_text())
                        self.assertEqual(set(selection['selected']), expected_nodes)
                        report = json.loads((work / 'metadata-cpu/coverage.json').read_text())
                        self.assertEqual(report['passed'], 1)
                        executed = (workspace / 'executed').read_text()
                        with self.assertRaisesRegex(ValueError, 'fresh work directory'):
                            cpu_tests.main()
                        self.assertEqual((workspace / 'executed').read_text(), executed)
                    elif outcome in ('skip', 'missing', 'ctest_skip', 'ctest_failure'):
                        with self.assertRaises(subprocess.CalledProcessError):
                            cpu_tests.main()
                        if outcome in ('skip', 'missing'):
                            self.assertFalse((work / 'metadata-cpu').exists())
                    else:
                        with self.assertRaisesRegex(ValueError, {
                                'ctest_missing': 'exactly test_metadata',
                                'wrong_binary': 'selected extras binary',
                                'invalid_native': 'native host'}[outcome]):
                            cpu_tests.main()


if __name__ == '__main__': unittest.main()
