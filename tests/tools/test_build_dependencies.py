"""Core selects Scheduler; the producer owns lower cohort validation."""
from __future__ import annotations
import hashlib
import importlib.util
import json
import os
import platform
from pathlib import Path
import shlex
import shutil
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


class SdkCppHeadersTest(unittest.TestCase):
    """Compile real standard-library wrappers through the production SDK setup."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.compiler = shutil.which('c++')
        self.assertIsNotNone(self.compiler, 'native C++ compiler is a test prerequisite')
        identity = subprocess.check_output(['file', '-L', self.compiler], text=True)
        machine = {'x86_64': 'x86-64', 'aarch64': 'ARM aarch64'}[platform.machine()]
        self.assertIn(machine, identity, 'never execute a foreign compiler')
        self.major = subprocess.check_output(
            [self.compiler, '-dumpversion'], text=True).strip().split('.')[0]
        self.target = subprocess.check_output(
            [self.compiler, '-dumpmachine'], text=True).strip()
        self.sdk = self.root / 'selected sdk'
        self.cpp = self.sdk / 'usr/include/c++' / self.major
        self.target_headers = self.sdk / 'usr/include' / self.target
        # Distinct real copies reproduce the duplicate #include_next wrapper
        # cohort; symlinks to the compiler defaults would be deduplicated by GCC.
        shutil.copytree(Path('/usr/include/c++') / self.major, self.cpp)
        shutil.copytree(Path('/usr/include') / self.target / 'c++' / self.major,
                        self.target_headers / 'c++' / self.major)
        for name in ('fenv.h', 'stdlib.h', 'math.h', 'stdio.h', 'string.h',
                     'time.h', 'wchar.h', 'locale.h', 'stdint.h'):
            shutil.copyfile(Path('/usr/include') / name, self.sdk / 'usr/include' / name)
        self.probe = self.root / 'wrappers.cpp'
        self.probe.write_text('''#include <cfenv>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <cwchar>
#include <clocale>
#include <cstdint>
#include <vector>
#include <string>
int main() {
  std::vector<std::string> values{"selected SDK"};
  return std::fesetround(FE_TONEAREST) + (values.empty() ? 1 : 0);
}
''')

    def configure(self, sdk_mode='ON', cxx=None, *, real_cmake=False, sdk=None):
        values = {name: 'OFF' for name in (
            'BUILD_TESTS', 'BUILD_TUTORIALS', 'BUILD_PYTHON', 'STRICT_WARNINGS',
            'SIMA_ENABLE_ASAN', 'SIMA_ENABLE_UBSAN', 'SIMA_ENABLE_TSAN',
            'SIMANEAT_SANITIZER_GATE_ONLY_EXTRAS', 'BUILD_FUZZ',
            'SKIP_NEAT_LLIMA', 'SIMANEAT_SCCACHE_ACTIVE', 'BUILD_DOCS', 'DOCS_ONLY')}
        values.update(BUILD_DIR=str(self.root / 'build'), BUILD_TYPE='Release',
                      NEAT_PACKAGE_BUILDINFO_JSON='', NEAT_BACKEND_DIR=str(self.root / 'deps'),
                      ELXR_SDK=sdk_mode, SYSROOT=str(self.sdk if sdk is None else sdk),
                      CXX=self.compiler if cxx is None else cxx,
                      CXXFLAGS='-DKEEP_CALLER_FLAG=1', ELXR_HOST_PYTHON_EXECUTABLE=sys.executable,
                      ELXR_TARGET_PYTHON_INCLUDE_DIR=str(self.sdk / 'usr/include/python3.13'),
                      ELXR_TARGET_PYTHON_VERSION='3.13')
        script = 'set -euo pipefail\n' + shell_function('configure_cmake') + '\n'
        script += '\n'.join(f'{key}={shlex.quote(value)}' for key, value in values.items())
        script += '\nelxr_ext_platform_triplet() { echo aarch64-linux-gnu; }\n'
        if not real_cmake:
            script += 'cmake() { printf "%s\\n" "$CXXFLAGS"; }\n'
        script += 'export CXX CXXFLAGS\nconfigure_cmake\n'
        # Never inherit the surrounding developer container's cross toolchain.
        env = {key: value for key, value in os.environ.items()
               if key in ('PATH', 'HOME', 'LANG')}
        return subprocess.run(['bash', '-c', script], cwd=self.root, env=env,
                              capture_output=True, text=True)

    def compile(self, flags):
        # Compile only: no target executable or floating-point test is run.
        return subprocess.run(
            [self.compiler, '-std=c++20', '-fsyntax-only', '-H', *flags, str(self.probe)],
            capture_output=True, text=True)

    def test_sdk_wrappers_reach_selected_libc_without_second_cpp_cohort(self):
        legacy_flags = ['-isystem', str(self.cpp), '-isystem',
                        str(self.target_headers / 'c++' / self.major)]
        broken = self.compile(legacy_flags)
        self.assertNotEqual(broken.returncode, 0)
        self.assertIn('fenv_t', broken.stderr)
        selected = self.configure()
        self.assertEqual(selected.returncode, 0, selected.stderr)
        result = self.compile(shlex.split(selected.stdout))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(str(self.sdk / 'usr/include/fenv.h'), result.stderr)
        self.assertNotIn(f'/usr/include/c++/{self.major}/fenv.h\n',
                         '\n'.join(line for line in result.stderr.splitlines()
                                   if str(self.sdk) not in line) + '\n')
        # Do not depend on the optional compatibility-header directory.
        shutil.rmtree(self.cpp / 'backward')
        selected = self.configure()
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertNotIn(str(self.cpp / 'backward'), shlex.split(selected.stdout))

    def test_missing_sdk_header_cannot_fall_back_to_compiler_headers(self):
        (self.cpp / 'cfenv').unlink()
        selected = self.configure()
        self.assertEqual(selected.returncode, 0, selected.stderr)
        result = self.compile(shlex.split(selected.stdout))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('cfenv: No such file', result.stderr)
        (self.target_headers / 'c++' / self.major / 'bits/c++config.h').unlink()
        selected = self.configure()
        self.assertNotEqual(selected.returncode, 0)
        self.assertIn('header cohort', selected.stderr)

    def test_native_configuration_does_not_replace_compiler_headers(self):
        selected = self.configure(sdk_mode='OFF')
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(selected.stdout.strip(), '-DKEEP_CALLER_FLAG=1')
        result = self.compile(shlex.split(selected.stdout))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_cxx_quoted_path_and_arguments_are_preserved_without_shell_execution(self):
        wrapper = self.root / 'compiler with spaces'
        calls = self.root / 'compiler-argv.jsonl'
        sentinel = self.root / 'must-not-be-created'
        wrapper.write_text(f'''#!{sys.executable}
import json, os, sys
with open({str(calls)!r}, 'a') as stream:
    stream.write(json.dumps(sys.argv[1:]) + '\\n')
os.execv({self.compiler!r}, [{self.compiler!r}, *sys.argv[1:]])
''')
        wrapper.chmod(0o755)
        arguments = ['-DARGUMENT_WITH_SPACES=two words', f'-DLITERAL=$(touch {sentinel})']
        selected = self.configure(cxx=shlex.join([str(wrapper), *arguments]))
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual([json.loads(line) for line in calls.read_text().splitlines()],
                         [[*arguments, '-dumpversion'], [*arguments, '-dumpmachine']])
        self.assertFalse(sentinel.exists())
        result = self.compile(shlex.split(selected.stdout))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        malformed = self.configure(cxx='"unterminated compiler')
        self.assertNotEqual(malformed.returncode, 0)
        self.assertEqual(malformed.stdout, '')

    def cmake_project(self):
        # Real native --sysroot fixture: copy the selected C++ wrappers above,
        # and make the actual host libc header closure available separately.
        for parent, destination in ((Path('/usr/include'), self.sdk / 'usr/include'),
                                    (Path('/usr/include') / self.target, self.target_headers)):
            for entry in parent.iterdir():
                target = destination / entry.name
                if not target.exists():
                    target.symlink_to(entry)
        (self.root / 'CMakeLists.txt').write_text(
            'cmake_minimum_required(VERSION 3.16)\n'
            'project(SdkCacheProbe LANGUAGES CXX)\n'
            'add_library(wrappers OBJECT wrappers.cpp)\n')
        dependencies = self.root / 'deps'
        dependencies.mkdir()
        (dependencies / 'dependencies.cmake').write_text(
            'set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)\n')

    def seed_cache(self, flags, *, native=False, cxx=None):
        env = {key: value for key, value in os.environ.items()
               if key in ('PATH', 'HOME', 'LANG')}
        env.update(CXX=self.compiler if cxx is None else cxx, CXXFLAGS=flags)
        args = ['cmake', '-S', str(self.root), '-B', str(self.root / 'build'),
                '-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY']
        if not native:
            args.append(f'-DCMAKE_SYSROOT={self.sdk}')
        result = subprocess.run(args, env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def cached_flags(self):
        return next(line.split('=', 1)[1] for line in
                    (self.root / 'build/CMakeCache.txt').read_text().splitlines()
                    if line.startswith('CMAKE_CXX_FLAGS:STRING='))

    def build_probe(self):
        # Object target only: compile actual wrappers, never execute a target.
        return subprocess.run(['cmake', '--build', str(self.root / 'build'), '--verbose'],
                              capture_output=True, text=True)

    def test_nonclean_sdk_cache_migrates_without_losing_caller_flags_or_growing(self):
        self.cmake_project()
        caller = "-DKEEP_CACHED_FLAG=1  '-DQUOTED_VALUE=two words'"
        sentinel = self.root / 'must-not-be-created'
        caller += ' ' + shlex.quote(f'-DLITERAL=$(touch {sentinel})')
        legacy = caller + ' ' + shlex.join([
            '-isystem' + str(self.cpp), '-isystem',
            str(self.target_headers / 'c++' / self.major)])
        self.seed_cache(legacy)
        before = self.build_probe()
        self.assertNotEqual(before.returncode, 0)
        self.assertIn('fenv_t', before.stdout + before.stderr)
        original = self.cached_flags()
        selected = self.configure(real_cmake=True)
        self.assertEqual(selected.returncode, 0, selected.stdout + selected.stderr)
        migrated = self.cached_flags()
        self.assertTrue(migrated.startswith(original + ' '))
        self.assertIn(caller, migrated)
        self.assertNotIn('KEEP_CALLER_FLAG', migrated)
        after = self.build_probe()
        self.assertEqual(after.returncode, 0, after.stdout + after.stderr)
        selected = self.configure(real_cmake=True)
        self.assertEqual(selected.returncode, 0, selected.stdout + selected.stderr)
        self.assertEqual(self.cached_flags(), migrated)
        self.assertEqual(shlex.split(migrated).count('-nostdinc++'), 1)
        self.assertFalse(sentinel.exists())

    def test_fresh_sdk_cmake_preserves_environment_flags(self):
        self.cmake_project()
        selected = self.configure(real_cmake=True)
        self.assertEqual(selected.returncode, 0, selected.stdout + selected.stderr)
        self.assertIn('-DKEEP_CALLER_FLAG=1', self.cached_flags())
        self.assertIn('-nostdinc++', self.cached_flags())
        result = self.build_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nonclean_native_cmake_leaves_cached_flags_untouched(self):
        self.cmake_project()
        self.seed_cache('-DKEEP_CACHED_FLAG=1', native=True)
        original = self.cached_flags()
        selected = self.configure(sdk_mode='OFF', real_cmake=True)
        self.assertEqual(selected.returncode, 0, selected.stdout + selected.stderr)
        self.assertEqual(self.cached_flags(), original)
        result = self.build_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_changed_sdk_or_compiler_rejects_before_changing_cache(self):
        self.cmake_project()
        self.seed_cache('-DKEEP_CACHED_FLAG=1')
        cache = self.root / 'build/CMakeCache.txt'
        original = cache.read_bytes()
        other_sdk = self.root / 'other sdk'
        (other_sdk / 'usr').mkdir(parents=True)
        (other_sdk / 'usr/include').symlink_to(self.sdk / 'usr/include')
        wrapper = self.root / 'different compiler'
        wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(self.compiler)} "$@"\n')
        wrapper.chmod(0o755)
        for arguments, diagnostic in (({'sdk': other_sdk}, 'sysroot differs'),
                                      ({'cxx': shlex.quote(str(wrapper))}, 'compiler differs'),
                                      ({'cxx': shlex.join([self.compiler, '-DNEW_ARGUMENT=1'])},
                                       'compiler differs')):
            with self.subTest(arguments=arguments):
                selected = self.configure(real_cmake=True, **arguments)
                self.assertNotEqual(selected.returncode, 0)
                self.assertIn(diagnostic, selected.stderr)
                self.assertIn('--clean', selected.stderr)
                self.assertEqual(cache.read_bytes(), original)

    def test_preloaded_compiler_identity_and_quoted_arguments_remain_supported(self):
        self.cmake_project()
        wrapper = self.root / 'compiler with spaces'
        wrapper.write_text(f'#!/bin/sh\nexec {shlex.quote(self.compiler)} "$@"\n')
        wrapper.chmod(0o755)
        cxx = shlex.join([str(wrapper), '-DKEEP_COMPILER_ARGUMENT=1'])
        self.seed_cache('-DKEEP_CACHED_FLAG=1', cxx=cxx)
        # Toolchain/preload selection can leave only the generated compiler
        # record, rather than a CMAKE_CXX_COMPILER cache entry (real SDK case).
        cache = self.root / 'build/CMakeCache.txt'
        cache.write_text('\n'.join(line for line in cache.read_text().splitlines()
                                   if line != '//CXX compiler'
                                   and not line.startswith('CMAKE_CXX_COMPILER:')) + '\n')
        selected = self.configure(cxx=cxx, real_cmake=True)
        self.assertEqual(selected.returncode, 0, selected.stdout + selected.stderr)
        result = self.build_probe()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_incompatible_cached_sdk_header_order_is_not_silently_rewritten(self):
        self.cmake_project()
        self.seed_cache(shlex.join(['-isystem', str(self.sdk / 'usr/include')]))
        cache = self.root / 'build/CMakeCache.txt'
        original = cache.read_bytes()
        selected = self.configure(real_cmake=True)
        self.assertNotEqual(selected.returncode, 0)
        self.assertIn('cached SDK header order', selected.stderr)
        self.assertIn('--clean', selected.stderr)
        self.assertEqual(cache.read_bytes(), original)

    def test_changed_or_missing_compiler_record_requires_clean(self):
        self.cmake_project()
        self.seed_cache('-DKEEP_CACHED_FLAG=1')
        record = next((self.root / 'build/CMakeFiles').glob('*/CMakeCXXCompiler.cmake'))
        original_record = record.read_text()
        cache = self.root / 'build/CMakeCache.txt'
        original_cache = cache.read_bytes()
        version_line = next(line for line in original_record.splitlines()
                            if line.startswith('set(CMAKE_CXX_COMPILER_VERSION '))
        changed = original_record.replace(
            version_line, f'set(CMAKE_CXX_COMPILER_VERSION "{int(self.major) + 1}.0.0")')
        for text, diagnostic in ((changed, 'compiler differs'), ('', 'record is incomplete')):
            with self.subTest(diagnostic=diagnostic):
                record.write_text(text)
                selected = self.configure(real_cmake=True)
                self.assertNotEqual(selected.returncode, 0)
                self.assertIn(diagnostic, selected.stderr)
                self.assertIn('--clean', selected.stderr)
                self.assertEqual(cache.read_bytes(), original_cache)

    def test_prior_nonclean_sysroot_update_cannot_hide_old_sdk_headers(self):
        self.cmake_project()
        flags = '-DKEEP_CACHED_FLAG=1 ' + shlex.join([
            '-isystem', str(self.cpp), '-isystem',
            str(self.target_headers / 'c++' / self.major)])
        self.seed_cache(flags)
        other_sdk = self.root / 'later sdk'
        (other_sdk / 'usr').mkdir(parents=True)
        (other_sdk / 'usr/include').symlink_to(self.sdk / 'usr/include')
        # Reproduce the real historical sequence, not a fabricated cache:
        # CMake updates its explicit sysroot but ignores new CXXFLAGS.
        changed = subprocess.run([
            'cmake', '-S', str(self.root), '-B', str(self.root / 'build'),
            f'-DCMAKE_SYSROOT={other_sdk}'], capture_output=True, text=True)
        self.assertEqual(changed.returncode, 0, changed.stdout + changed.stderr)
        self.assertEqual(self.cached_flags(), flags)
        cache = self.root / 'build/CMakeCache.txt'
        original = cache.read_bytes()
        selected = self.configure(sdk=other_sdk, real_cmake=True)
        self.assertNotEqual(selected.returncode, 0)
        self.assertIn('header cohort differs', selected.stderr)
        self.assertIn('--clean', selected.stderr)
        self.assertEqual(cache.read_bytes(), original)

    def test_foreign_cpp_major_rejected_but_unrelated_user_includes_preserved(self):
        self.cmake_project()
        caller_headers = self.root / 'caller headers'
        caller_headers.mkdir()
        flags = '-DKEEP_CACHED_FLAG=1 ' + shlex.join(['-isystem', str(caller_headers)])
        self.seed_cache(flags)
        selected = self.configure(real_cmake=True)
        self.assertEqual(selected.returncode, 0, selected.stdout + selected.stderr)
        self.assertTrue(self.cached_flags().startswith(flags + ' '))
        foreign = self.sdk / 'usr/include/c++' / str(int(self.major) + 1)
        self.seed_cache(flags + ' ' + shlex.join(['-isystem', str(foreign)]))
        # A new CXXFLAGS environment is intentionally ignored by CMake, so set
        # this explicit caller cache option through CMake for the negative case.
        changed = subprocess.run([
            'cmake', '-S', str(self.root), '-B', str(self.root / 'build'),
            '-DCMAKE_CXX_FLAGS=' + flags + ' ' + shlex.join(['-isystem', str(foreign)])],
            capture_output=True, text=True)
        self.assertEqual(changed.returncode, 0, changed.stdout + changed.stderr)
        cache = self.root / 'build/CMakeCache.txt'
        original = cache.read_bytes()
        selected = self.configure(real_cmake=True)
        self.assertNotEqual(selected.returncode, 0)
        self.assertIn('header cohort differs', selected.stderr)
        self.assertIn('--clean', selected.stderr)
        self.assertEqual(cache.read_bytes(), original)


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

    def test_build_dependency_selection_and_failures(self):
        self.producer()  # A partial sibling stack still needs a complete publication.
        script = 'set -euo pipefail\n' + '\n'.join(shell_function(name) for name in (
            'parse_args', 'has_local_backend_inputs', 'select_and_normalize_local_artifact_paths')) + '''
require_sima_cli_neat_install() { :; }
current_core_branch() { echo feature/test; }
python3() {
  echo "$1" >&2
  case "$1" in
    tools/ci/acquire_scheduler.py) return "${TEST_ACQUIRE_EXIT:-0}" ;;
    *) return "${TEST_SELECT_EXIT:-0}" ;;
  esac
}
parse_args "$@"
select_and_normalize_local_artifact_paths
'''
        env = {key: value for key, value in os.environ.items() if key in ('PATH', 'HOME', 'LANG')}
        env.update(REPO_ROOT=str(self.core), WORKSPACE_ROOT=str(self.root), DOCS_ONLY='OFF',
                   INSTALL_DEPS_ONLY='OFF', NEAT_LLIMA_ARTIFACT_DIR='',
                   SIMA_CLI_BIN='sima-cli', NEAT_VULCAN_ENV='production')
        acquire, select = 'tools/ci/acquire_scheduler.py', 'tools/build_dependencies.py'
        for arguments, overrides, code, calls in (
            (['--doc', '--clean'], {}, 0, []),
            ([], {}, 0, [acquire, select]),
            ([], {'NEAT_SCHEDULER_PACKAGE_DIR': '/ci/package'}, 0, [select]),
            ([], {'TEST_ACQUIRE_EXIT': '29'}, 29, [acquire]),
            ([], {'NEAT_RUNNER_SOURCE': '/local', 'TEST_SELECT_EXIT': '37'}, 37, [select]),
        ):
            with self.subTest(arguments=arguments, overrides=overrides):
                staging = self.core / 'deps/published/scheduler'
                staging.mkdir(parents=True, exist_ok=True)
                stale = staging / 'previous-publication'
                stale.touch()
                result = subprocess.run(['bash', '-c', script, 'selection-test', *arguments],
                                        cwd=self.core, env={**env, **overrides}, text=True, capture_output=True)
                self.assertEqual(result.returncode, code, result.stderr)
                self.assertEqual(result.stderr.splitlines(), calls)
                self.assertEqual(stale.exists(), acquire not in calls)
                self.assertEqual(list(staging.parent.iterdir()), [staging])

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
        self.assertLess(main.index("activate_elxr_build_env_if_needed"), main.index("prepare_published_scheduler_sdk"))
        self.assertLess(main.index("prepare_published_scheduler_sdk"), main.index("build_dependencies.py prepare"))
        self.assertLess(main.index("build_dependencies.py prepare"), main.index("configure_cmake"))


if __name__ == "__main__":
    unittest.main()
