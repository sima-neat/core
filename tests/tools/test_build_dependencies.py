"""Core selects Scheduler; the producer owns lower cohort validation."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('build_dependencies', ROOT/'tools/build_dependencies.py')
DEPENDENCIES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DEPENDENCIES)


def source(root, component):
    # Shell-shadow fixture only; no lower implementation is imported.
    (root/'cmake').mkdir(parents=True)
    (root/'CMakeLists.txt').touch()
    (root/'cmake'/('Sima'+component.title()+'Config.cmake.in')).touch()
    (root/'build.sh').touch()


def shell_function(name):
    text=(ROOT/'build.sh').read_text(); start=text.index(f'{name}() {{')
    return text[start:text.index('\n}\n', start)+2]


class BuildDependenciesTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(); self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name); self.core=self.root/'core'; self.core.mkdir()

    def producer(self):
        root=self.root/'scheduler'; source(root,'scheduler'); (root/'tools').mkdir()
        (root/'tools/client_dependencies.py').write_text('import sys\nprint("delegated", " ".join(sys.argv[1:]))\nsys.exit(23)\n')
        return root

    def package(self):
        root=self.root/'package'; prefix=root/'usr'
        tool=prefix/'share/SimaScheduler/tools/client_dependencies.py';tool.parent.mkdir(parents=True)
        tool.write_text('print("metadata fixture")\n')
        entry={'kind':'file','sha256':hashlib.sha256(tool.read_bytes()).hexdigest(),'mode':tool.stat().st_mode&0o7777}
        info={'component':'scheduler','installed_files':{tool.relative_to(prefix).as_posix():entry}}
        info['package_sha256']=hashlib.sha256(json.dumps(info,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        (prefix/'share/SimaScheduler/buildinfo.json').write_text(json.dumps(info))
        return root,tool

    def test_selects_source_before_stale_package(self):
        root=self.producer(); (root/'dist/package/usr').mkdir(parents=True)
        self.assertEqual(DEPENDENCIES.scheduler_coordinator(self.core,{}),root/'tools/client_dependencies.py')

    def test_explicit_package_and_receipt_are_authoritative(self):
        self.producer(); root,tool=self.package()
        self.assertEqual(DEPENDENCIES.scheduler_coordinator(self.core,{'NEAT_SCHEDULER_PACKAGE_DIR':str(root)}),tool)
        tool.write_text('different producer')
        with self.assertRaisesRegex(ValueError,'receipt'):
            DEPENDENCIES.scheduler_coordinator(self.core,{'NEAT_SCHEDULER_PACKAGE_DIR':str(root)})

    def test_selected_missing_coordinator_never_falls_back(self):
        self.producer(); broken=self.root/'broken'; source(broken,'scheduler')
        with self.assertRaisesRegex(ValueError,'lacks its client'):
            DEPENDENCIES.scheduler_coordinator(self.core,{'NEAT_SCHEDULER_SOURCE':str(broken)})
        with self.assertRaisesRegex(ValueError,'only one'):
            DEPENDENCIES.scheduler_coordinator(self.core,{'NEAT_SCHEDULER_SOURCE':str(broken),'NEAT_SCHEDULER_PACKAGE_DIR':'unused'})

    def test_missing_scheduler_does_not_fetch(self):
        with self.assertRaisesRegex(ValueError, 'No local Scheduler source/package') as error:
            DEPENDENCIES.scheduler_coordinator(self.core, {})
        self.assertIn('NEAT_SCHEDULER_SOURCE', str(error.exception))
        self.assertIn('NEAT_SCHEDULER_PACKAGE_DIR', str(error.exception))
        self.assertIn('tools/ci/acquire_scheduler.py', str(error.exception))

    def test_docs_only_skips_runtime_selection_but_normal_build_requires_it(self):
        # Exercise the real CLI parsing and pre-bootstrap shell selection, with
        # no sibling Scheduler, package override, or fake coordinator.
        tools = self.core / 'tools'
        tools.mkdir()
        (tools / 'build_dependencies.py').write_text(
            (ROOT / 'tools/build_dependencies.py').read_text())
        functions = shell_function('parse_args') + '\n' + shell_function(
            'select_and_normalize_local_artifact_paths')
        script = 'set -euo pipefail\n' + functions + '\n'
        script += f'REPO_ROOT={shlex.quote(str(self.core))}\n'
        script += 'DOCS_ONLY=OFF\nINSTALL_DEPS_ONLY=OFF\nNEAT_LLIMA_ARTIFACT_DIR=\n'
        script += 'parse_args "$@"\nselect_and_normalize_local_artifact_paths\n'
        script += 'printf "selected:%s" "$DOCS_ONLY"\n'
        env = {key: value for key, value in os.environ.items()
               if not key.startswith('NEAT_')}
        for arguments, expected_success in [(['--doc', '--clean'], True), ([], False)]:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    ['bash', '-c', script, 'selection-test', *arguments],
                    cwd=self.core, env=env, text=True, capture_output=True)
                if expected_success:
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stdout, 'selected:ON')
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('No local Scheduler source/package', result.stderr)

    def test_cmake_always_receives_the_selected_docs_only_mode(self):
        values = {name: 'OFF' for name in (
            'BUILD_TESTS', 'BUILD_TUTORIALS', 'BUILD_PYTHON', 'STRICT_WARNINGS',
            'SIMA_ENABLE_ASAN', 'SIMA_ENABLE_UBSAN', 'SIMA_ENABLE_TSAN',
            'SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS', 'BUILD_FUZZ',
            'SKIP_NEAT_LLIMA', 'ELXR_SDK', 'SIMANEAT_SCCACHE_ACTIVE')}
        values.update(BUILD_DIR=str(self.root / 'reused-build'), BUILD_TYPE='Release',
                      BUILD_DOCS='ON', NEAT_PACKAGE_BUILDINFO_JSON='',
                      NEAT_BACKEND_DIR=str(self.root / 'dependencies'))
        script = 'set -euo pipefail\n' + shell_function('configure_cmake') + '\n'
        script += '\n'.join(f'{key}={shlex.quote(value)}' for key, value in values.items())
        # Capture the real wrapper's argv; actual CMake reused-tree behavior is
        # covered in test_vulcan_ci_docs_only.py without provider stubs.
        script += '\ncmake() { printf "%s\\n" "$@"; }\n'
        script += 'DOCS_ONLY="$1"\nconfigure_cmake\n'
        for mode in ('ON', 'OFF'):
            with self.subTest(mode=mode):
                result = subprocess.run(
                    ['bash', '-c', script, 'cmake-arguments-test', mode],
                    text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                arguments = result.stdout.splitlines()
                selected = [arg for arg in arguments
                            if arg.startswith('-DSIMANEAT_DOCS_ONLY_CONFIGURE=')]
                self.assertEqual(selected, [f'-DSIMANEAT_DOCS_ONLY_CONFIGURE={mode}'])

    def test_actual_cli_preserves_arguments_and_failure(self):
        producer=self.producer()
        env=dict(os.environ);env.pop('NEAT_SCHEDULER_PACKAGE_DIR',None);env['NEAT_SCHEDULER_SOURCE']=str(producer)
        args=['prepare','--core',str(self.core),'--output',str(self.root/'out'),'--jobs','3','--clean']
        result=subprocess.run([sys.executable,str(ROOT/'tools/build_dependencies.py'),*args],env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,23,result.stderr)
        self.assertEqual(result.stdout.strip(),'delegated '+' '.join(args))
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

    def test_main_keeps_one_final_install_and_deploy(self):
        main = shell_function("main")
        self.assertEqual(main.count("install_artifacts_into_current_environment_if_requested"), 1)
        self.assertEqual(main.count("deploy_artifacts_to_devkit_if_requested"), 1)
        self.assertNotIn("ensure_neat_internals", main)
        self.assertLess(main.index("build_dependencies.py prepare"), main.index("configure_cmake"))


if __name__ == "__main__":
    unittest.main()
