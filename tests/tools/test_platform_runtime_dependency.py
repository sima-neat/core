"""Host-only real-archive tests for forwarding Runner's consumed backend packages."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('core_build_dependencies', ROOT/'tools/build_dependencies.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class PlatformRuntimeDependencyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.archive = self.make_archive('simaai-mlart-modalix', 'libMLArt.so')
        self.control = {'Package': 'simaai-mlart-modalix', 'Version': '2.2.0+test', 'Architecture': 'arm64',
                        'deb_sha256': hashlib.sha256(self.archive.read_bytes()).hexdigest()}
        self.runner = {'external_dependencies': {'MLArt': {'debian_provider': copy.deepcopy(self.control), 'loader_name': 'libMLArt.so'}},
                       'debian_dependencies': [copy.deepcopy(self.control)]}
        self.env = patch.dict(os.environ, {'NEAT_RUNNER_PLATFORM_RUNTIME_DEBS': json.dumps([str(self.archive)])})
        self.env.start(); self.addCleanup(self.env.stop)

    def make_archive(self, name, library):
        stage = self.root/name; (stage/'DEBIAN').mkdir(parents=True)
        (stage/'DEBIAN/control').write_text(f'Package: {name}\nVersion: 2.2.0+test\nArchitecture: arm64\n'
                                          'Maintainer: Test <test@example.invalid>\nDescription: Explicit host fixture\n')
        file = stage/'usr/lib/aarch64-linux-gnu'/library; file.parent.mkdir(parents=True)
        file.write_bytes(b'HOST_ARCHIVE_FIXTURE_NOT_ELF')
        archive = self.root/(name+'.deb')
        subprocess.run(['dpkg-deb', '--build', '--root-owner-group', str(stage), str(archive)],
                       check=True, stdout=subprocess.PIPE)
        return archive

    def test_exact_runtime_deb_composed_without_embedding(self):
        out = self.root/'out'; out.mkdir()
        result = m.compose_external_runtimes(self.runner, out)
        self.assertEqual(result['MLArt']['debian_provider'], self.control)
        self.assertIn('usr/lib/aarch64-linux-gnu/libMLArt.so', result['MLArt']['installed_files'])
        self.assertEqual((out/self.archive.name).read_bytes(), self.archive.read_bytes())
        self.assertEqual(list(out.iterdir()), [out/self.archive.name])

    def test_wrapper_forwards_each_explicit_archive_before_cmake_args(self):
        with patch.dict(os.environ, {'NEAT_RUNNER_CMAKE_ARGS': '-DFIXTURE=ON'}), patch.object(m.subprocess, 'run') as run:
            m.build_runtime_source('runner', self.root/'src', self.root/'package', self.root/'output',
                                   {'kernels': {'package': 'kernels', 'buildinfo': {}}}, 2, False)
        command = run.call_args.args[0]
        i = command.index('--platform-runtime-deb')
        self.assertEqual(command[i+1], str(self.archive)); self.assertLess(i, command.index('--'))

    def test_modified_archive_not_relabelled(self):
        self.archive.write_bytes(self.archive.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'missing exact'):
            m.compose_external_runtimes(self.runner, None)

    def test_missing_archive_rejected(self):
        os.environ['NEAT_RUNNER_PLATFORM_RUNTIME_DEBS'] = '[]'
        with self.assertRaisesRegex(ValueError, 'missing exact'):
            m.compose_external_runtimes(self.runner, None)

    def test_unpinned_runner_dependency_rejected(self):
        self.runner['debian_dependencies'][0]['Version'] = 'older'
        with self.assertRaisesRegex(ValueError, 'lacks its exact'):
            m.compose_external_runtimes(self.runner, None)

    def test_unconsumed_archive_rejected(self):
        extra = self.make_archive('other-runtime', 'libunused.so')
        os.environ['NEAT_RUNNER_PLATFORM_RUNTIME_DEBS'] = json.dumps([str(self.archive), str(extra)])
        with self.assertRaisesRegex(ValueError, 'Unconsumed'):
            m.compose_external_runtimes(self.runner, None)

    def test_paths_with_spaces_and_invalid_json(self):
        p = self.root/'space name.deb'; p.write_bytes(self.archive.read_bytes())
        self.assertEqual(m.platform_runtime_archives({'NEAT_RUNNER_PLATFORM_RUNTIME_DEBS': json.dumps([str(p)])}), [p])
        for value in ('oops', '{}', 'null', '[1]', '[""]'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'JSON array'):
                m.platform_runtime_archives({'NEAT_RUNNER_PLATFORM_RUNTIME_DEBS': value})
        with self.assertRaisesRegex(ValueError, 'distinct'):
            m.platform_runtime_archives({'NEAT_RUNNER_PLATFORM_RUNTIME_DEBS': json.dumps([str(p), str(p)])})

    def test_payload_overlap_with_runner_rejected(self):
        result = m.compose_external_runtimes(self.runner, None)
        prefix = self.root/'runner/usr'; (prefix/'lib/aarch64-linux-gnu').mkdir(parents=True)
        (prefix/'lib/aarch64-linux-gnu/libMLArt.so').touch()
        with self.assertRaisesRegex(ValueError, 'duplicates'):
            m.verify_package_ownership({'runner': {'prefix': str(prefix)},
                                        'external_dependencies': {'RunnerBackends': result}})

    def test_heap_remains_the_existing_sdk_dependency_not_an_optional_engine(self):
        self.runner['external_dependencies']['SimaHeap'] = {'debian_provider': {'Package': 'simaai-heap'}}
        self.assertEqual(set(m.compose_external_runtimes(self.runner, None)), {'MLArt'})

    def test_no_runtime_selected_when_no_optional_provider(self):
        with self.assertRaisesRegex(ValueError, 'no backend providers'):
            m.compose_external_runtimes({}, None)
        os.environ['NEAT_RUNNER_PLATFORM_RUNTIME_DEBS'] = '[]'
        self.assertEqual(m.compose_external_runtimes({}, None), {})


if __name__ == '__main__':
    unittest.main()
