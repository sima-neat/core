"""Native build coordination; no SDK installation, target execution or network."""
from __future__ import annotations

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("build_dependencies", ROOT / "tools/build_dependencies.py")
DEPENDENCIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPENDENCIES)


LAST_PACKAGES = {}


def seal(prefix: Path, name: str, info: dict) -> None:
    path = prefix / "share" / name / "buildinfo.json"
    exclusions = [f"share/{name}/buildinfo.json"]
    if name == "SimaScheduler":
        exclusions += [f"share/{name}/platform-install-manifest.json"]
    info['payload_exclusions'] = exclusions
    info['installed_files'] = {p.relative_to(prefix).as_posix():
        {'kind': 'file', 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'mode': p.stat().st_mode & 0o777}
        for p in prefix.rglob('*') if p.is_file() and p.relative_to(prefix).as_posix() not in exclusions}
    info['installed_payload_sha256'] = hashlib.sha256(json.dumps(info['installed_files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    info.pop('package_sha256', None)
    info['package_sha256'] = hashlib.sha256(json.dumps(info, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    path.write_text(json.dumps(info))


# An installed test-only metadata producer. No compiler, sibling checkout or
# network is used. The production producer is independently tested in Scheduler.
COMPAT_PRODUCER = r'''import argparse,hashlib,json,pathlib,subprocess,tempfile,shutil
p=argparse.ArgumentParser()
for a in ('runner-package','scheduler-package','package-dir'):p.add_argument('--'+a,required=True)
a=p.parse_args(); root=pathlib.Path(a.package_dir);prefix=root/'usr'
prefix.mkdir(parents=True)
children={n:json.loads((pathlib.Path(getattr(a,n+'_package'))/'usr/share'/pkg/'buildinfo.json').read_text()) for n,pkg in [('runner','SimaRunner'),('scheduler','SimaScheduler')]}
for n in ('SimaRuntime','NeatInternals'):
 f=prefix/'lib/cmake'/n/(n+'Config.cmake');f.parent.mkdir(parents=True);f.write_text('# metadata fixture\n')
f=prefix/'share/SimaRuntime/buildinfo.json';f.parent.mkdir(parents=True)
info={'schema_version':1,'component':'runtime-compat','architecture':'aarch64','runner_build':children['runner'],'scheduler_build':children['scheduler'],'payload_exclusions':['share/SimaRuntime/buildinfo.json']}
info['installed_files']={str(p.relative_to(prefix)):{'kind':'file','sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'mode':p.stat().st_mode&0o777} for p in prefix.rglob('*') if p.is_file()}
info['installed_payload_sha256']=hashlib.sha256(json.dumps(info['installed_files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
info['package_sha256']=hashlib.sha256(json.dumps(info,sort_keys=True,separators=(',',':')).encode()).hexdigest();f.write_text(json.dumps(info))
with tempfile.TemporaryDirectory() as t:
 d=pathlib.Path(t);shutil.copytree(prefix,d/'usr');(d/'DEBIAN').mkdir()
 versions={n:subprocess.check_output(['dpkg-deb','-f',str(next(pathlib.Path(getattr(a,n+'_package')).glob('*.deb'))),'Version'],text=True).strip() for n in ('runner','scheduler')}
 (d/'DEBIAN/control').write_text('Depends: '+', '.join('simaai-'+n+' (= '+versions[n]+')' for n in ('runner','scheduler'))+'\nPackage: simaai-runtime\nVersion: 0.2.0~split1+test\nArchitecture: arm64\nMaintainer: fixture\nDescription: metadata fixture\n')
 subprocess.run(['dpkg-deb','--build',str(d),str(root/'compat.deb')],check=True,stdout=subprocess.DEVNULL)
'''


def package(root: Path, component: str, contract: str = "contract") -> None:
    name = DEPENDENCIES.PACKAGE_NAMES[component]
    directory = root / "usr/lib/cmake" / name
    directory.mkdir(parents=True)
    (directory / f"{name}Config.cmake").touch()
    info = {
        "source_sha256": "0123456789ab", "schema_version": 1, "component": component,
        "flags": {"cxx": "", "release": "-O3 -DNDEBUG"},
        "architecture": "aarch64", "build_type": "Release", "sysroot": "/sdk",
        "compiler": {"sha256": "compiler", "version": "12", "target": "aarch64"},
        "toolchain": {"sha256": "toolchain"},
    }
    expected = hashlib.sha256(("kernels/api.h:" + hashlib.sha256(b"// test package contract\n").hexdigest() + "\n").encode()).hexdigest()
    info["contract_sha256" if component == "kernels" else "kernels_contract_sha256"] = expected if contract == "contract" else contract
    if component != "kernels":
        info["kernels_build"] = {key: info[key] for key in ("flags", "build_type", "toolchain")}
    header = root / "usr/include" / component / "api.h"
    header.parent.mkdir(parents=True)
    header.write_text("// test package contract\n")
    if component == "runner":
        info["contract_sha256"] = hashlib.sha256(("runner/api.h:" + hashlib.sha256(header.read_bytes()).hexdigest() + "\n").encode()).hexdigest()
    if component == "scheduler":
        info['runner_build'] = json.loads((LAST_PACKAGES['runner'] / 'usr/share/SimaRunner/buildinfo.json').read_text())
        generator = root/'usr/share/SimaScheduler/tools/compatibility_package.py'
        generator.parent.mkdir(parents=True)
        generator.write_text(COMPAT_PRODUCER)
    receipt = root / "usr/share" / name / "buildinfo.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(info))
    if component != 'kernels':seal(root/'usr', name, info)
    control = root / "control/DEBIAN"
    control.mkdir(parents=True)
    identity = DEPENDENCIES.DEBIAN_ROLES[component]
    dependencies = {'kernels': (), 'runner': ('kernels',), 'scheduler': ('kernels', 'runner')}[component]
    depends = ('Depends: ' + ', '.join(DEPENDENCIES.DEBIAN_ROLES[c] + ' (= 0.1.0)' for c in dependencies) + "\n") if dependencies else ''
    (control / "control").write_text(depends + f"Package: {identity}\nVersion: 0.1.0\nArchitecture: arm64\nMaintainer: test <test@example.invalid>\nDescription: build coordination test\n")
    shutil.copytree(root / "usr", control.parent / "usr")
    subprocess.check_call(["dpkg-deb", "--build", str(control.parent), str(root / f"{component}.deb")], stdout=subprocess.DEVNULL)
    LAST_PACKAGES[component] = root


def source(root: Path, component: str) -> None:
    (root / DEPENDENCIES.PROJECT_FILES[component]).parent.mkdir(parents=True)
    (root / DEPENDENCIES.PROJECT_FILES[component]).touch()
    (root / "CMakeLists.txt").touch()
    if component in ("runner", "scheduler"):
        (root / "build.sh").touch()


def shell_function(name: str) -> str:
    text = (ROOT / "build.sh").read_text()
    start = text.index(f"{name}() {{")
    return text[start:text.index("\n}\n", start) + 2]


class BuildDependenciesTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.core = self.root / "core"
        self.core.mkdir()
        self.addCleanup(self.temporary.cleanup)

    def codecs_fixture(self, control_version='0.1.3', control_arch='arm64', dev_name='neat-codecs-dev', dev_depends='neat-codecs (= 0.1.3)'):
        root = self.root / 'codecs'; prefix = root / 'usr'
        runtime = {'usr/lib/aarch64-linux-gnu/neat/codecs/libfixture.so.0.1.3': b'unit-test library payload only'}
        development = {'usr/include/neat/codecs/codec.h': b'unit-test header',
                       'usr/lib/aarch64-linux-gnu/cmake/SimaCodecs/SimaCodecsConfig.cmake': b'# test config'}
        for name, data in {**runtime, **development}.items():
            path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        info = {'schema_version': 1, 'component': 'codecs', 'version': '0.1.3', 'architecture': 'aarch64',
                'sonames': {'decoder': 'libfixture.so.0.1.3'},
                'files': {name: {'kind': 'file', 'sha256': hashlib.sha256(data).hexdigest()}
                          for name, data in {**runtime, **development}.items()}}
        receipt = prefix / 'share/SimaCodecs/buildinfo.json';receipt.parent.mkdir(parents=True)
        receipt.write_text(json.dumps(info));runtime['usr/share/SimaCodecs/buildinfo.json'] = receipt.read_bytes()
        controls = []
        for role, files in [('runtime', runtime), ('dev', development)]:
            stage = self.root / ('codec-' + role); (stage / 'DEBIAN').mkdir(parents=True)
            name = dev_name if role == 'dev' else 'neat-codecs'
            dependency = f'Depends: {dev_depends}\n' if role == 'dev' else ''
            (stage / 'DEBIAN/control').write_text(f'Package: {name}\nVersion: {control_version}\nArchitecture: {control_arch}\n{dependency}Maintainer: tests\nDescription: external Codec fixture, not hardware qualification\n')
            for filename, data in files.items():
                path = stage / filename;path.parent.mkdir(parents=True, exist_ok=True);path.write_bytes(data)
            archive = root / (name + '.deb')
            subprocess.run(['dpkg-deb', '--root-owner-group', '--build', str(stage), str(archive)], check=True, stdout=subprocess.DEVNULL)
            if role == 'runtime':
                controls.append({'Package':name, 'Version':'0.1.3','Architecture':'arm64',
                                 'deb_sha256':hashlib.sha256(archive.read_bytes()).hexdigest()})
        digest = hashlib.sha256(''.join(f'{p.relative_to(prefix).as_posix()}:{hashlib.sha256(p.read_bytes()).hexdigest()}\n'
                                        for p in sorted(prefix.rglob('*')) if p.is_file()).encode()).hexdigest()
        runner = {'build_configuration': {'SIMA_RUNTIME_CODECS': 'ON'},
                  'external_dependencies': {'SimaCodecs': {'package_sha256': digest,
                        'config_sha256':hashlib.sha256((prefix/'lib/aarch64-linux-gnu/cmake/SimaCodecs/SimaCodecsConfig.cmake').read_bytes()).hexdigest()}},
                  'debian_dependencies': controls}
        return root, runner

    def test_prepare_stages_external_codec_pair_with_native_two_step_order(self):
        root, codec_runner = self.codecs_fixture()
        selected = {}
        for component in DEPENDENCIES.COMPONENTS:
            path = self.root / component;package(path, component)
            selected[component] = {'kind':'package', 'path':str(path)}
        runner_root = self.root / 'scheduler'
        receipt = runner_root / 'usr/share/SimaScheduler/buildinfo.json'
        data = json.loads(receipt.read_text());data.update(codec_runner);seal(runner_root/'usr','SimaScheduler',data)
        shutil.copytree(runner_root/'usr', runner_root/'control/usr', dirs_exist_ok=True)
        subprocess.run(['dpkg-deb','--root-owner-group','--build',str(runner_root/'control'),str(runner_root/'scheduler.deb')],check=True,stdout=subprocess.DEVNULL)
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}):
            result = DEPENDENCIES.prepare(self.core, self.root/'output', selected, 1, False)
        self.assertEqual(len(list((self.root/'output/debs').glob('*.deb'))),6)
        self.assertEqual(set(result['external_dependencies']),{'SimaCodecs'})
        resolved = json.loads((self.root/'output/resolved_manifest.json').read_text())
        self.assertEqual(resolved['external_dependencies']['SimaCodecs']['buildinfo']['version'],'0.1.3')

    def test_external_codecs_matching_version_deb_cannot_hide_wrong_header(self):
        root, runner = self.codecs_fixture()
        (self.root/'codec-dev/usr/include/neat/codecs/codec.h').write_bytes(b'different header')
        subprocess.run(['dpkg-deb','--root-owner-group','--build',str(self.root/'codec-dev'),str(root/'neat-codecs-dev.deb')],check=True,stdout=subprocess.DEVNULL)
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'exactly cover'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_external_codecs_pair_composes_exactly_without_native_build_step(self):
        root, runner = self.codecs_fixture();destination = self.root / 'cache';destination.mkdir()
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR': str(root)}):
            info = DEPENDENCIES.compose_external_codecs(runner, destination)
        self.assertEqual({x['Package'] for x in info['debian_packages']}, {'neat-codecs', 'neat-codecs-dev'})
        self.assertEqual(len(list(destination.glob('*.deb'))), 2)
        self.assertEqual(DEPENDENCIES.COMPONENTS, ('kernels', 'runner', 'scheduler'))

    def test_external_codecs_extra_non_usr_payload_refused(self):
        root, runner = self.codecs_fixture()
        extra = self.root/'codec-dev/etc/unrelated';extra.parent.mkdir(parents=True);extra.write_bytes(b'not owned Codec payload')
        subprocess.run(['dpkg-deb','--root-owner-group','--build',str(self.root/'codec-dev'),str(root/'neat-codecs-dev.deb')],check=True,stdout=subprocess.DEVNULL)
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'outside usr'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_external_codecs_special_mode_difference_refused(self):
        root, runner = self.codecs_fixture()
        (self.root/'codec-dev/usr/include/neat/codecs/codec.h').chmod(0o4644)
        subprocess.run(['dpkg-deb','--root-owner-group','--build',str(self.root/'codec-dev'),str(root/'neat-codecs-dev.deb')],check=True,stdout=subprocess.DEVNULL)
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'exactly cover'):
            DEPENDENCIES.compose_external_codecs(runner, None)
        (self.root/'codec-dev/usr/include/neat/codecs/codec.h').chmod(0o644)
        subprocess.run(['dpkg-deb','--root-owner-group','--build',str(self.root/'codec-dev'),str(root/'neat-codecs-dev.deb')],check=True,stdout=subprocess.DEVNULL)
        (root/'usr/include/neat/codecs/codec.h').chmod(0o4644)
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'exactly cover'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_codec_archive_paths_validated_before_payload_domain_filter(self):
        import io, tarfile
        for name in ('/etc/unrelated', '../etc/unrelated', 'usr/../unrelated', 'usr//unrelated', 'usr/./unrelated'):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode='w') as archive:
                item = tarfile.TarInfo(name);item.size=1;archive.addfile(item, io.BytesIO(b'x'))
            buffer.seek(0)
            process = __import__('unittest.mock').mock.Mock()
            process.stdout=buffer;process.__enter__=lambda x:x;process.__exit__=lambda *args:None
            with patch.object(DEPENDENCIES.subprocess,'Popen',return_value=process), self.assertRaisesRegex(ValueError,'Unsafe dependency archive path'):
                DEPENDENCIES.codec_deb_inventory(Path('mock-untrusted.deb'))

    def test_external_codecs_renamed_sdk_control_role_refused(self):
        root, runner = self.codecs_fixture(dev_name='renamed-codec-sdk')
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'noncanonical dev Package'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_external_codecs_wrong_sdk_runtime_dependency_refused(self):
        root, runner = self.codecs_fixture(dev_depends='neat-codecs (= 3.0.0)')
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'SDK archive must depend exactly'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_external_codecs_accepts_usr_prefix_with_matching_archives_beside_it(self):
        root, runner = self.codecs_fixture()
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root/'usr')}):
            result = DEPENDENCIES.compose_external_codecs(runner, None)
        self.assertEqual(len(result['debian_packages']),2)

    def test_external_codecs_missing_explicit_provider_refused(self):
        root, runner = self.codecs_fixture()
        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(ValueError, 'NEAT_CODECS_PACKAGE_DIR'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_external_codecs_api3_version_control_refused(self):
        root, runner = self.codecs_fixture(control_version='3.0.0')
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'version/architecture'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_external_codecs_wrong_control_architecture_refused(self):
        root, runner = self.codecs_fixture(control_arch='amd64')
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'version/architecture'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_external_codecs_missing_dev_archive_refused(self):
        root, runner = self.codecs_fixture();(root/'neat-codecs-dev.deb').unlink()
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'exactly cover'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_external_codecs_tampered_header_refused(self):
        root, runner = self.codecs_fixture();(root/'usr/include/neat/codecs/codec.h').write_bytes(b'wrong API')
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'consumed Runtime Codec payload'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_external_codecs_runtime_archive_hash_pin_refused(self):
        root, runner = self.codecs_fixture();runner['debian_dependencies'][0]['deb_sha256']='different archive'
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'dependency control receipt'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_codec_disabled_transport_does_not_require_external_provider(self):
        root, runner = self.codecs_fixture();runner['build_configuration']['SIMA_RUNTIME_CODECS']='OFF'
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(DEPENDENCIES.compose_external_codecs(runner, None))

    def test_enabled_codec_transport_requires_consumed_receipt(self):
        for external in ({}, {'SimaCodecs': {}}, {'SimaCodecs': None}):
            with self.subTest(external=external), self.assertRaisesRegex(ValueError, 'lack a consumed'):
                DEPENDENCIES.compose_external_codecs({'build_configuration': {'SIMA_RUNTIME_CODECS': 'ON'},
                                                     'external_dependencies': external}, None)

    def test_codec_archive_unowned_empty_directory_refused(self):
        root, runner = self.codecs_fixture()
        (self.root/'codec-dev/etc').mkdir(mode=0o000)
        subprocess.run(['dpkg-deb','--root-owner-group','--build',str(self.root/'codec-dev'),str(root/'neat-codecs-dev.deb')],check=True,stdout=subprocess.DEVNULL)
        with patch.dict(os.environ, {'NEAT_CODECS_PACKAGE_DIR':str(root)}), self.assertRaisesRegex(ValueError, 'outside usr'):
            DEPENDENCIES.compose_external_codecs(runner, None)

    def test_archive_stream_drained_before_wait_and_reaped_on_refusal(self):
        import io, tarfile
        from unittest.mock import Mock
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            item = tarfile.TarInfo('usr');item.type=tarfile.DIRTYPE;archive.addfile(item)
        raw = buffer.getvalue() + b'\0' * (1024 * 1024)
        stream = io.BytesIO(raw);process = Mock(stdout=stream)
        process.poll.return_value = 0
        def wait():
            self.assertEqual(stream.tell(), len(raw))
            return 0
        process.wait.side_effect = wait
        with patch.object(DEPENDENCIES.subprocess, 'Popen', return_value=process):
            self.assertEqual(DEPENDENCIES.codec_deb_inventory(Path('mock.deb')), {})
        self.assertTrue(stream.closed)
        process = Mock(stdout=io.BytesIO(buffer.getvalue()))
        process.poll.return_value = None
        with patch.object(DEPENDENCIES.subprocess, 'Popen', return_value=process), self.assertRaisesRegex(ValueError, 'reject'):
            with DEPENDENCIES.dependency_archive(Path('mock.deb')):
                raise ValueError('reject')
        process.terminate.assert_called_once()
        process.wait.assert_called_once_with(timeout=5)
        self.assertTrue(process.stdout.closed)

    def test_dependency_deb_unowned_payload_and_special_modes_refused(self):
        root = self.root/'package';package(root, 'kernels')
        info = DEPENDENCIES.receipt(root, 'SimaKernels')
        stage = root/'control'
        (stage/'etc').mkdir()
        subprocess.run(['dpkg-deb','--build',str(stage),str(root/'kernels.deb')],check=True,stdout=subprocess.DEVNULL)
        with self.assertRaisesRegex(ValueError, 'outside owned paths'):
            DEPENDENCIES.verify_deb_receipt(root/'kernels.deb', 'SimaKernels', info, root/'usr')
        (stage/'etc').rmdir()
        (stage/'usr/include/kernels/api.h').chmod(0o4644)
        subprocess.run(['dpkg-deb','--build',str(stage),str(root/'kernels.deb')],check=True,stdout=subprocess.DEVNULL)
        with self.assertRaisesRegex(ValueError, 'payload does not match'):
            DEPENDENCIES.verify_deb_receipt(root/'kernels.deb', 'SimaKernels', info, root/'usr')

    def test_dependency_archive_duplicate_and_unsafe_paths_refused(self):
        import io, tarfile
        from unittest.mock import Mock
        for names, error in ((('usr/x','usr/x'),'Duplicate dependency'),
                             (('usr/../x',),'Unsafe dependency')):
            buffer = io.BytesIO()
            with tarfile.open(fileobj=buffer, mode='w') as archive:
                for name in names:
                    item=tarfile.TarInfo(name);item.size=1;archive.addfile(item,io.BytesIO(b'x'))
            buffer.seek(0);process=Mock(stdout=buffer);process.poll.return_value=0
            with patch.object(DEPENDENCIES.subprocess,'Popen',return_value=process), self.assertRaisesRegex(ValueError,error):
                DEPENDENCIES.verify_deb_receipt(Path('mock.deb'),'SimaKernels',{},self.root)

    def test_runtime_loader_configuration_exact_producer_contract(self):
        root = self.root/'package';package(root,'runner')
        runtime=root/'usr/lib/aarch64-linux-gnu/neat/runtime';runtime.mkdir(parents=True)
        stage=root/'control';shutil.copytree(root/'usr',stage/'usr',dirs_exist_ok=True)
        info=DEPENDENCIES.receipt(root,'SimaRunner')
        loader=stage/'etc/ld.so.conf.d/sima-neat.conf';loader.parent.mkdir(parents=True)
        loader.write_bytes(b'/usr/lib/aarch64-linux-gnu/neat/runtime\n/usr/lib/aarch64-linux-gnu/neat/gst-plugins\n')
        archive=root/'runner.deb'
        subprocess.run(['dpkg-deb','--build',str(stage),str(archive)],check=True,stdout=subprocess.DEVNULL)
        DEPENDENCIES.verify_deb_receipt(archive,'SimaRunner',info,root/'usr')
        loader.write_text('unrelated loader path\n')
        subprocess.run(['dpkg-deb','--build',str(stage),str(archive)],check=True,stdout=subprocess.DEVNULL)
        with self.assertRaisesRegex(ValueError,'payload does not match'):
            DEPENDENCIES.verify_deb_receipt(archive,'SimaRunner',info,root/'usr')

    def test_runtime_receipt_checks_exact_payload_and_modes(self):
        package(self.root / 'runner', 'runner')
        prefix = self.root / 'runner/usr'
        path = prefix / 'share/SimaRunner/buildinfo.json'
        info = json.loads(path.read_text())
        info['payload_exclusions'] = ['share/SimaRunner/buildinfo.json']
        info['installed_files'] = {p.relative_to(prefix).as_posix():
            {'kind': 'file', 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'mode': p.stat().st_mode & 0o777}
            for p in prefix.rglob('*') if p.is_file() and p != path}
        info['installed_payload_sha256'] = hashlib.sha256(json.dumps(info['installed_files'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        info.pop('package_sha256', None)
        info['package_sha256'] = hashlib.sha256(json.dumps(info, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        path.write_text(json.dumps(info))
        DEPENDENCIES.receipt(self.root / 'runner', 'SimaRunner')
        (prefix / 'include/runner/api.h').chmod(0o755)
        with self.assertRaisesRegex(ValueError, 'Runtime payload differs'):
            DEPENDENCIES.receipt(self.root / 'runner', 'SimaRunner')

    def test_cohort_requires_exact_schema2_numerical_payload(self):
        for component in DEPENDENCIES.COMPONENTS:
            package(self.root / component, component)
        packages = {component: {'buildinfo': DEPENDENCIES.receipt(self.root / component,
                    DEPENDENCIES.PACKAGE_NAMES[component]),
                    'prefix': str(self.root / component / 'usr')} for component in DEPENDENCIES.COMPONENTS}
        kernels = packages['kernels']['buildinfo']; runner = packages['runner']['buildinfo']
        kernels.update(schema_version=2, package_sha256='package-v1', installed_payload_sha256='math-v1')
        runner['kernels_build'].update(package_sha256='package-v1', installed_payload_sha256='math-v1')
        packages['scheduler']['buildinfo']['kernels_build'].update(package_sha256='package-v1', installed_payload_sha256='math-v1')
        DEPENDENCIES.validate_cohort(packages)
        kernels['installed_payload_sha256'] = 'math-v2'
        with self.assertRaisesRegex(ValueError, 'different Kernels installed_payload'):
            DEPENDENCIES.validate_cohort(packages)

    def test_all_eight_source_package_combinations_build_once_in_order(self):
        for local in __import__("itertools").chain.from_iterable(__import__("itertools").combinations(DEPENDENCIES.COMPONENTS, n) for n in range(4)):
            with self.subTest(local=local), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                core = root / "core"
                core.mkdir()
                env, calls = {}, []
                # Prebuilt Scheduler consumes the same deterministic Runner fixture
                # even when Runner is selected from source in this combination.
                package(root / "expected-runner", "runner")
                for component in DEPENDENCIES.COMPONENTS:
                    if component in local:
                        source(root / component, component)
                    else:
                        path = root / "prebuilt" / component
                        package(path, component)
                        env[f"NEAT_{component.upper()}_PACKAGE_DIR"] = str(path)

                real_run = subprocess.run
                def build(command, **kwargs):
                    if command[0] == "cmake":
                        self.assertTrue(kwargs["check"])
                        self.assertNotIn("ctest", command)
                        if "-S" in command:
                            calls.append("kernels")
                            self.assertIn("-DSIMA_KERNELS_PROFILE=a65", command)
                        elif "--install" in command:
                            package(Path(kwargs["env"]["DESTDIR"]), "kernels")
                        return subprocess.CompletedProcess(command, 0)
                    if command[0] == __import__("sys").executable and "deb" in command:
                        # The real DEB projection is exercised by producer tests.
                        return subprocess.CompletedProcess(command, 0)
                    if command[0] != "bash":
                        return real_run(command, **kwargs)
                    cwd, check, env = kwargs["cwd"], kwargs["check"], kwargs["env"]
                    component = Path(cwd).name
                    self.assertEqual(check, True)
                    self.assertNotIn("--install", command)
                    self.assertNotIn("ctest", command)
                    calls.append(component)
                    k = Path(command[command.index("--kernels-package") + 1])
                    self.assertTrue((k / "usr/share/SimaKernels/buildinfo.json").is_file())
                    package(Path(command[command.index("--package-dir") + 1]), component)
                with patch.object(DEPENDENCIES.subprocess, "run", side_effect=build):
                    result = DEPENDENCIES.prepare(core, root / "output", DEPENDENCIES.select(core, env), 2, False)
                self.assertEqual(calls, list(local))
                self.assertEqual(len(list((root / "output/debs").glob("*.deb"))), 4)
                self.assertEqual(set(result), {"kernels", "runner", "scheduler", "runtime_compat"})
                for component in DEPENDENCIES.COMPONENTS:
                    self.assertEqual(result[component]["debian_version"], "0.1.0")
                    self.assertIn(f'SIMANEAT_{component.upper()}_DEBIAN_VERSION [==[0.1.0]==]',
                                  (root / "output/dependencies.cmake").read_text())

    def test_dependency_pins_use_actual_deb_control_not_receipt_hash(self):
        selected = {}
        for component in DEPENDENCIES.COMPONENTS:
            directory = self.root / component
            package(directory, component)
            control = directory / "control/DEBIAN/control"
            control.write_text(control.read_text().replace("Version: 0.1.0", "Version: 7:0.1.0+actual-2").replace("(= 0.1.0)", "(= 7:0.1.0+actual-2)"))
            subprocess.run(["dpkg-deb", "--build", str(directory / "control"),
                            str(directory / (component + ".deb"))], check=True,
                           stdout=subprocess.DEVNULL)
            selected[component] = {"kind": "package", "path": str(directory)}
        result = DEPENDENCIES.prepare(self.core, self.root / "pins", selected, 1, False)
        for component in DEPENDENCIES.COMPONENTS:
            item = result[component]
            self.assertEqual(item["debian_version"], "7:0.1.0+actual-2")

    def test_source_wins_over_stale_sibling_package(self):
        source(self.root / "kernels", "kernels")
        package(self.root / "kernels/dist/package", "kernels")
        self.assertEqual(DEPENDENCIES.select(self.core, {})["kernels"]["kind"], "source")

    def test_explicit_package_wins_over_source(self):
        source(self.root / "kernels", "kernels")
        package(self.root / "prebuilt", "kernels")
        selected = DEPENDENCIES.select(self.core, {"NEAT_KERNELS_PACKAGE_DIR": str(self.root / "prebuilt")})
        self.assertEqual(selected["kernels"]["kind"], "package")

    def test_conflicting_overrides_and_invalid_source_rejected(self):
        with self.assertRaisesRegex(ValueError, "only one"):
            DEPENDENCIES.select(self.core, {"NEAT_KERNELS_SOURCE": "/source", "NEAT_KERNELS_PACKAGE_DIR": "/package"})
        with self.assertRaisesRegex(ValueError, "source root"):
            DEPENDENCIES.select(self.core, {"NEAT_RUNNER_SOURCE": str(self.core)})

    def test_missing_dependency_is_actionable_todo_without_fetch(self):
        with patch.object(DEPENDENCIES.subprocess, "run") as run:
            with self.assertRaisesRegex(ValueError, "TODO: remote artifact"):
                DEPENDENCIES.prepare(self.core, self.root / "out", {}, 1, False)
            run.assert_not_called()

    def test_local_failure_stops_without_artifact_fallback(self):
        source(self.root / "kernels", "kernels")
        source(self.root / "runner", "runner")
        source(self.root / "scheduler", "scheduler")
        with patch.object(DEPENDENCIES.subprocess, "run", side_effect=subprocess.CalledProcessError(1, "build")) as run:
            with self.assertRaises(subprocess.CalledProcessError):
                DEPENDENCIES.prepare(self.core, self.root / "out", DEPENDENCIES.select(self.core, {}), 1, False)
            self.assertEqual(run.call_count, 1)

    def test_contract_and_toolchain_mismatch_rejected_not_revision_changes(self):
        packages = {}
        for component in DEPENDENCIES.COMPONENTS:
            path = self.root / component
            package(path, component)
            name = DEPENDENCIES.PACKAGE_NAMES[component]
            packages[component] = {"buildinfo": DEPENDENCIES.receipt(path, name), "prefix": str(path / "usr")}
        packages["kernels"]["buildinfo"]["source_sha256"] = "new math"
        DEPENDENCIES.validate_cohort(packages)
        packages["runner"]["buildinfo"]["kernels_contract_sha256"] = "old"
        with self.assertRaisesRegex(ValueError, "contract"):
            DEPENDENCIES.validate_cohort(packages)
        packages["runner"]["buildinfo"]["kernels_contract_sha256"] = packages["kernels"]["buildinfo"]["contract_sha256"]
        packages["runner"]["buildinfo"]["compiler"]["target"] = "x86"
        with self.assertRaisesRegex(ValueError, "compiler.target"):
            DEPENDENCIES.validate_cohort(packages)

    def test_headers_only_copies_packages_without_target_build(self):
        env = {}
        for component in DEPENDENCIES.COMPONENTS:
            path = self.root / component
            package(path, component)
            env[f"NEAT_{component.upper()}_PACKAGE_DIR"] = str(path)
        with patch.object(DEPENDENCIES.subprocess, "run", wraps=subprocess.run) as run:
            DEPENDENCIES.prepare(self.core, self.root / "out", DEPENDENCIES.select(self.core, env), 1, True)
            self.assertEqual(run.call_count, 1)
            self.assertIn('compatibility_package.py', run.call_args.args[0][1])
        self.assertTrue((self.root / "out/headers/usr/include/runner/api.h").is_file())
        source(self.root / "sources", "kernels")
        with self.assertRaisesRegex(ValueError, "will not compile"):
            DEPENDENCIES.prepare(self.core, self.root / "out2", {"kernels": {"kind": "source", "path": str(self.root / "sources")}, "runner": {"kind": "package", "path": str(self.root / "runner")}, "scheduler": {"kind": "package", "path": str(self.root / "scheduler")}}, 1, True)

    def test_shadow_copies_only_selected_sources_and_rewrites_paths(self):
        source(self.root / "external runner", "runner")
        (self.core / "build.sh").touch()
        (self.root / "external runner/build").mkdir()
        (self.root / "external runner/build/stale").touch()
        shadow = self.root / "shadow"
        script = "set -euo pipefail\n" + shell_function("copy_shadow_source") + "\n" + shell_function("prepare_shadow_workspace")
        script += f'\nREPO_ROOT={shlex.quote(str(self.core))}\nNEAT_RUNNER_SOURCE={shlex.quote(str(self.root / "external runner"))}\n'
        script += f'prepare_shadow_workspace {shlex.quote(str(shadow))}\nprintf "%s" "$NEAT_RUNNER_SOURCE"\n'
        result = subprocess.run(["bash", "-c", script], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, str(shadow / "runner"))
        self.assertTrue((shadow / "runner/CMakeLists.txt").is_file())
        self.assertFalse((shadow / "runner/build").exists())
        self.assertFalse((shadow / "internals").exists())

    def test_stale_deployment_payload_is_rejected_even_with_same_headers(self):
        root = self.root / "package"
        package(root, "kernels")
        info = DEPENDENCIES.receipt(root, "SimaKernels")
        (root / "usr/lib/changed.a").write_text("new library missing from old deb")
        with self.assertRaisesRegex(ValueError, "payload does not match"):
            DEPENDENCIES.verify_deb_receipt(root / "kernels.deb", "SimaKernels", info, root / "usr")

    def test_header_receipt_tampering_is_rejected(self):
        packages = {}
        for component in DEPENDENCIES.COMPONENTS:
            root = self.root / component
            package(root, component)
            name = DEPENDENCIES.PACKAGE_NAMES[component]
            packages[component] = {"prefix": str(root / "usr"), "buildinfo": DEPENDENCIES.receipt(root, name)}
        (self.root / "kernels/usr/include/kernels/api.h").write_text("different ABI")
        with self.assertRaisesRegex(ValueError, "headers do not match"):
            DEPENDENCIES.validate_cohort(packages)

    def test_schema2_payload_mutation_rejected_before_cohort(self):
        root = self.root / "kernel-new"
        package(root, "kernels")
        file = root / "usr/share/SimaKernels/buildinfo.json"
        info = json.loads(file.read_text())
        info["schema_version"] = 2
        info["installed_files"] = {p.relative_to(root / "usr").as_posix():
                                   {"kind": "file", "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "mode": p.stat().st_mode & 0o777}
                                   for p in (root / "usr").rglob("*") if p.is_file() and p != file}
        info["installed_payload_sha256"] = hashlib.sha256(json.dumps(info["installed_files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        file.write_text(json.dumps(info))
        DEPENDENCIES.receipt(root, "SimaKernels")
        (root / "usr/lib/new.a").write_text("unrecorded numerical payload")
        with self.assertRaisesRegex(ValueError, "payload differs"):
            DEPENDENCIES.receipt(root, "SimaKernels")

    def test_scheduler_rejects_stale_runner_or_numerical_inputs(self):
        for component in DEPENDENCIES.COMPONENTS:
            package(self.root / component, component)
        packages = {component: {'buildinfo': DEPENDENCIES.receipt(self.root / component,
                    DEPENDENCIES.PACKAGE_NAMES[component]), 'prefix': str(self.root / component / 'usr')}
                    for component in DEPENDENCIES.COMPONENTS}
        DEPENDENCIES.validate_cohort(packages)
        for field in ('package_sha256', 'installed_payload_sha256', 'contract_sha256'):
            with self.subTest(field=field):
                info = packages['scheduler']['buildinfo']['runner_build']
                previous = info[field]; info[field] = 'stale'
                with self.assertRaisesRegex(ValueError, 'Scheduler consumed'):
                    DEPENDENCIES.validate_cohort(packages)
                info[field] = previous
        packages['scheduler']['buildinfo']['compiler']['target'] = 'x86'
        with self.assertRaisesRegex(ValueError, 'Runner/Scheduler compiler.target'):
            DEPENDENCIES.validate_cohort(packages)

    def test_scheduler_is_required_and_old_combined_runner_is_not_accepted(self):
        for component in ('kernels', 'runner'):
            package(self.root / component, component)
        selected = {component: {'kind': 'package', 'path': str(self.root / component)}
                    for component in ('kernels', 'runner')}
        with self.assertRaisesRegex(ValueError, 'No local scheduler'):
            DEPENDENCIES.prepare(self.core, self.root / 'out', selected, 1, False)
        path = self.root / 'runner/usr/share/SimaRunner/buildinfo.json'
        info = json.loads(path.read_text()); info['component'] = 'runtime'
        path.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, 'Unsupported SimaRunner'):
            DEPENDENCIES.receipt(self.root / 'runner', 'SimaRunner')

    def test_main_keeps_one_final_install_and_deploy(self):
        main = shell_function("main")
        self.assertEqual(main.count("install_artifacts_into_current_environment_if_requested"), 1)
        self.assertEqual(main.count("deploy_artifacts_to_devkit_if_requested"), 1)
        self.assertNotIn("ensure_neat_internals", main)
        self.assertLess(main.index("build_dependencies.py prepare"), main.index("configure_cmake"))


if __name__ == "__main__":
    unittest.main()
