#!/usr/bin/env python3
"""Exercise actual CMake registration, runtime profiles and installed tests."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class GuardedCtestTests(unittest.TestCase):
    def test_proxy_and_legacy_commands_have_same_args_labels_deadline(self):
        function='include("'+str(ROOT/'cmake/NeatGuardedTests.cmake')+'")\n'
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.16)\nproject(proxy NONE)\nenable_testing()\nfind_package(Python3 COMPONENTS Interpreter REQUIRED)\n'+function+'''
add_executable(fixture IMPORTED)
set_target_properties(fixture PROPERTIES IMPORTED_LOCATION /usr/bin/true)
simaneat_add_test(NAME full_fixture COMMAND fixture --original-argument)
set_tests_properties(full_fixture PROPERTIES WORKING_DIRECTORY /tmp TIMEOUT 60 LABELS "e2e;perf" ENVIRONMENT "UNCHANGED=1")
''')
            configs=[]
            for enabled in ('OFF','ON'):
                build=root/enabled
                subprocess.run(['cmake','-S',str(root),'-B',str(build),'-DSIMANEAT_GUARDED_HARDWARE_TESTS='+enabled],check=True,capture_output=True)
                value=json.loads(subprocess.check_output(['ctest','--test-dir',str(build),'--show-only=json-v1']))['tests'][0]
                configs.append(value)
            old,new=configs
            self.assertEqual(old['name'],new['name'])
            self.assertEqual(old['command'],['/usr/bin/true','--original-argument'])
            self.assertEqual(new['command'][-2:],old['command'])
            self.assertIn('ctest-proxy',new['command'])
            self.assertEqual(old['properties'],new['properties'])


    def test_tutorial_script_and_target_preserve_argv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'run_with_modelzoo.sh').write_text('#!/bin/sh\nexit 0\n')
            (root/'run_with_modelzoo.sh').chmod(0o755)
            (root/'CMakeLists.txt').write_text('cmake_minimum_required(VERSION 3.16)\nproject(proxy NONE)\nenable_testing()\ninclude("'+str(ROOT/'cmake/NeatGuardedTests.cmake')+'")\n'+'''
add_executable(tutorial IMPORTED)
set_target_properties(tutorial PROPERTIES IMPORTED_LOCATION /usr/bin/true)
simaneat_add_test(NAME model NATIVE_COMMAND COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/run_with_modelzoo.sh resnet /tmp/model.tar.gz /tmp 2.1.3 $<TARGET_FILE:tutorial> --model /tmp/model.tar.gz)
simaneat_add_test(NAME script NATIVE_COMMAND COMMAND /bin/sh -c [=[value="$1"; printf '%s\\n' "$value";]=] sh "argument with spaces")
simaneat_add_test(NAME target COMMAND tutorial --unchanged)
set_tests_properties(model script target PROPERTIES WORKING_DIRECTORY /tmp TIMEOUT 240 LABELS "tutorial;mpk" RESOURCE_LOCK dispatcher)
''')
            configs = []
            for enabled in ('OFF', 'ON'):
                build = root/enabled
                subprocess.run(['cmake','-S',str(root),'-B',str(build),'-DSIMANEAT_GUARDED_HARDWARE_TESTS='+enabled],check=True,capture_output=True)
                configs.append(json.loads(subprocess.check_output(['ctest','--test-dir',str(build),'--show-only=json-v1']))['tests'])
            for old, new in zip(*configs):
                self.assertEqual(old['name'], new['name'])
                self.assertEqual(old['properties'], new['properties'])
                self.assertIn('ctest-proxy', new['command'])
                separator = new['command'].index('--')
                self.assertEqual(old['command'], new['command'][separator+1:])
            self.assertEqual(configs[0][0]['command'][0], str(root/'run_with_modelzoo.sh'))
            self.assertEqual(len(configs[0][1]['command']), 5)
            self.assertIn(';', configs[0][1]['command'][2])
            self.assertTrue((root/'ON/tools/hardware_test_supervisor.py').is_file())

    def test_actual_e2e_variant_loops_use_guard_without_changing_contract(self):
        text = (ROOT/'tests/CMakeLists.txt').read_text()
        loops = []
        for variable in ('encoder_case', 'mode'):
            pattern = r'  foreach\(' + variable + r' (?:IN ITEMS )?[^\n]+\).*?  endforeach\(\)'
            candidates = re.findall(pattern, text, re.DOTALL)
            expected_target = ('video_sender_adaptive_ingress_e2e_test'
                               if variable == 'encoder_case' else 'model_seeded_input_capacity_test')
            matches = [loop for loop in candidates if 'COMMAND ' + expected_target in loop]
            self.assertEqual(len(matches), 1)
            self.assertNotRegex(matches[0], r'(?m)^\s+add_test\(')
            loops.append(matches[0])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.16)
project(ActualE2ERegistrations NONE)
enable_testing()
include("''' + str(ROOT/'cmake/NeatGuardedTests.cmake') + '''")
add_executable(video_sender_adaptive_ingress_e2e_test IMPORTED)
add_executable(model_seeded_input_capacity_test IMPORTED)
set_target_properties(video_sender_adaptive_ingress_e2e_test model_seeded_input_capacity_test
  PROPERTIES IMPORTED_LOCATION /usr/bin/true)
function(simaneat_append_ctest_env name)
  set_property(TEST ${name} APPEND PROPERTY ENVIRONMENT "ORIGINAL_ENVIRONMENT=1")
endfunction()
''' + '\n'.join(loops))
            configs = []
            for enabled in ('OFF', 'ON'):
                build = root/'build'
                subprocess.run(['cmake', '-S', str(root), '-B', str(build),
                                '-DSIMANEAT_GUARDED_HARDWARE_TESTS='+enabled],
                               check=True, capture_output=True)
                configs.append(json.loads(subprocess.check_output(
                    ['ctest', '--test-dir', str(build), '--show-only=json-v1']))['tests'])
            expected = {'encoder_'+case+'_contract_test' for case in ('h264', 'h265', 'mjpeg', 'concurrent')}
            expected |= {'model_seeded_input_capacity_'+mode+'_test' for mode in
                         ('tensor', 'sample', 'defaults', 'unseeded', 'oversize_width', 'oversize_height')}
            self.assertEqual({test['name'] for test in configs[1]}, expected)
            self.assertEqual(len(configs[1]), 10)
            for old, new in zip(*configs):
                self.assertEqual(old['name'], new['name'])
                self.assertEqual(old['properties'], new['properties'])
                self.assertIn('ctest-proxy', new['command'])
                self.assertEqual(old['command'], new['command'][new['command'].index('--')+1:])

    def test_tutorial_sources_install_shared_guard(self):
        text = (ROOT/'tutorials/CMakeLists.txt').read_text()
        self.assertNotRegex(text, r'(?m)^\s+add_test\(')
        self.assertIn('DESTINATION share/sima-neat/tutorials/cmake', text)
        self.assertIn('DESTINATION share/sima-neat/tutorials/tools', text)

    def test_custom_scheduler_tests_use_resolved_profile_and_installed_targets(self):
        """Native loader doubles exercise literal production CMake, not ARM code."""
        tests = (ROOT / 'tests/CMakeLists.txt').read_text()
        production = (ROOT / 'CMakeLists.txt').read_text()
        registrations = tests[tests.index('add_executable(unit_boxdecode_publication_test'):
                              tests.index('simaneat_add_test(\n  NAME unit_afe_publication_ledger_generated')]
        profile = tests[tests.index('option(SIMANEAT_CTEST_FOR_DEVKIT'):
                        tests.index('# Test blacklist (by executable name)')]
        helper = re.search(r'function\(simaneat_append_ctest_env test_name\).*?endfunction\(\)',
                           production, re.DOTALL).group(0)
        target_install = re.search(
            r'if \(PIPELINESESSION_TEST_TARGETS\).*?COMPONENT extras\n\)',
            tests, re.DOTALL).group(0)
        directory_install = re.search(
            r'install\(\n  DIRECTORY \$\{CMAKE_BINARY_DIR\}/tests/\n.*?\n\)',
            tests, re.DOTALL).group(0)
        manifest_install = re.search(
            r'install\(\n  FILES \$\{CMAKE_BINARY_DIR\}/tests/CTestTestfile.cmake\n.*?\n\)',
            tests, re.DOTALL).group(0)
        names = {'unit_boxdecode_publication_test', 'unit_boxdecode_kernel_recipe_test',
                 'unit_boxdecode_geometry_test', 'unit_kernel_runner_integration_test'}
        env = os.environ.copy()
        for key in ('CC', 'CXX', 'CFLAGS', 'CXXFLAGS', 'CPPFLAGS', 'LDFLAGS', 'CPATH',
                    'CPLUS_INCLUDE_PATH', 'LIBRARY_PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD',
                    'LD_AUDIT', 'CMAKE_TOOLCHAIN_FILE', 'CMAKE_PREFIX_PATH', 'DESTDIR', 'SYSROOT'):
            env.pop(key, None)

        def command(*args, check=True):
            return subprocess.run(list(map(str, args)), env=env, text=True,
                                  capture_output=True, check=check)

        # Compare against the running native interpreter rather than assuming
        # x86: the same regression also runs on native AArch64 hosts.
        machine_pattern = r'^\s*Machine:\s*(.+)$'
        native_machine = re.search(machine_pattern,
                                   command('readelf', '-h', sys.executable).stdout,
                                   re.MULTILINE).group(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source'
            test_source = source / 'tests'
            test_source.mkdir(parents=True)
            # Installed Scheduler and extracted Core extras have separate prefixes,
            # as in CI. Test RPATH alone cannot discover the private providers.
            provider = root / 'installed-scheduler'
            plugins = provider / 'gst-plugins'
            runtime = provider / 'runtime'
            plugins.mkdir(parents=True)
            runtime.mkdir()
            for filename in set(re.findall(r'(?:\$\{PROJECT_SOURCE_DIR\}/)?[\w/]+\.cpp',
                                           registrations)):
                path = (source / filename.removeprefix('${PROJECT_SOURCE_DIR}/')
                        if filename.startswith('${PROJECT_SOURCE_DIR}/')
                        else test_source / filename)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('extern "C" int fixture_runtime();\n'
                                'extern "C" int fixture_plugin();\n'
                                'int main() { return fixture_runtime() == 42 && '
                                'fixture_plugin() == 22 ? 0 : 1; }\n'
                                if path.name.startswith('unit_') else '// Native fixture TU.\n')
            (source / 'plugin.cpp').write_text('extern "C" int fixture_plugin() { return 22; }\n')
            (source / 'runtime.cpp').write_text('extern "C" int fixture_plugin();\n'
                                               'extern "C" int fixture_runtime() { '
                                               'return fixture_plugin() + 20; }\n')
            (source / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.16)
project(InstalledCustomTests LANGUAGES CXX)
enable_testing()
add_library(fixture_plugin SHARED plugin.cpp)
add_library(fixture_runtime SHARED runtime.cpp)
target_link_libraries(fixture_runtime PRIVATE fixture_plugin)
set_target_properties(fixture_runtime PROPERTIES INSTALL_RPATH "$ORIGIN/../gst-plugins")
add_library(fixture_frontend INTERFACE)
target_link_libraries(fixture_frontend INTERFACE fixture_runtime fixture_plugin)
add_library(SimaScheduler::frontend ALIAS fixture_frontend)
add_library(SimaScheduler::ClientTestSupport ALIAS fixture_frontend)
add_library(sima_neat INTERFACE)
add_library(sima_neat_internal_headers INTERFACE)
target_link_libraries(sima_neat_internal_headers INTERFACE SimaScheduler::frontend)
''' + f'''install(TARGETS fixture_plugin LIBRARY DESTINATION "{plugins}" COMPONENT extras)
install(TARGETS fixture_runtime LIBRARY DESTINATION "{runtime}" COMPONENT extras)
set(SIMANEAT_CTEST_NEAT_PLUGIN_DIR "{plugins}")
set(SIMANEAT_EXPORTED_NEAT_RUNTIME_DIR_CANDIDATES "{runtime}")
include("{ROOT}/cmake/NeatGuardedTests.cmake")
''' + helper + '\nadd_subdirectory(tests)\n')
            (test_source / 'CMakeLists.txt').write_text(
                registrations + '\n' + profile + '\n' + target_install + '\n' +
                directory_install + '\n' + manifest_install + '\n')
            for devkit in ('OFF', 'ON'):
                with self.subTest(profile=devkit):
                    build = root / ('build-' + devkit)
                    installed = root / ('extras-' + devkit)
                    command('cmake', '-S', source, '-B', build,
                            '-DCMAKE_CXX_COMPILER=/usr/bin/c++',
                            '-DCMAKE_INSTALL_PREFIX=' + str(installed),
                            '-DSIMANEAT_GUARDED_HARDWARE_TESTS=OFF',
                            '-DSIMANEAT_CTEST_FOR_DEVKIT=' + devkit)
                    selected = json.loads(command('ctest', '--test-dir', build / 'tests',
                                                  '--show-only=json-v1').stdout)['tests']
                    self.assertEqual({test['name'] for test in selected}, names)
                    expected_plugins = ('/usr/lib/aarch64-linux-gnu/neat/gst-plugins'
                                        if devkit == 'ON' else str(plugins))
                    expected_runtime = ('/usr/lib/aarch64-linux-gnu/neat/runtime'
                                        if devkit == 'ON' else str(runtime))
                    for test in selected:
                        props = {prop['name']: prop['value'] for prop in test['properties']}
                        self.assertEqual(props['TIMEOUT'], 30)
                        self.assertEqual(set(props['LABELS']),
                                         {'strict', 'unit', 'runtime', 'gstreamer'}
                                         if test['name'] == 'unit_kernel_runner_integration_test'
                                         else {'strict', 'unit', 'contracts'})
                        values = dict(item.split('=', 1) for item in props['ENVIRONMENT'])
                        for key in ('GST_PLUGIN_PATH', 'GST_PLUGIN_PATH_1_0', 'SIMA_GST_PLUGIN_DIR'):
                            self.assertEqual(values[key], expected_plugins)
                        self.assertEqual(values['GST_PLUGIN_SYSTEM_PATH_1_0'], '')
                        self.assertEqual(values['SIMA_CLI_CHECK_FOR_UPDATE'], '0')
                        self.assertIn(expected_plugins, values['LD_LIBRARY_PATH'].split(':'))
                        self.assertIn(expected_runtime, values['LD_LIBRARY_PATH'].split(':'))
                    if devkit == 'ON':
                        continue  # Configure-only: never use DevKit paths for host execution.
                    command('cmake', '--build', build, '--parallel', '2')
                    command('cmake', '--install', build, '--component', 'extras')
                    relocated = root / 'relocated-extras'
                    installed.rename(relocated)
                    installed_tests = relocated / 'lib/sima-neat/tests'
                    manifest = installed_tests / 'CTestTestfile.cmake'
                    manifest.write_text(manifest.read_text().replace(str(build / 'tests'),
                                                                    str(installed_tests)))
                    shutil.rmtree(build)
                    for name in names:
                        executable = installed_tests / name
                        self.assertIn('ELF', command('file', executable).stdout)
                        machine = re.search(machine_pattern,
                                            command('readelf', '-h', executable).stdout,
                                            re.MULTILINE).group(1)
                        self.assertEqual(machine, native_machine)
                        dynamic = command('readelf', '-d', executable).stdout
                        self.assertIn('[$ORIGIN/../..]', dynamic)
                        self.assertNotIn(str(root), dynamic)
                    result = command('ctest', '--test-dir', installed_tests, '--output-on-failure')
                    self.assertIn('100% tests passed', result.stdout)
                    plugin = plugins / 'libfixture_plugin.so'
                    plugin.rename(plugins / 'provider-removed')
                    result = command('ctest', '--test-dir', installed_tests,
                                     '--output-on-failure', check=False)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn('0% tests passed, 4 tests failed out of 4', result.stdout)
                    self.assertIn('error while loading shared libraries: libfixture_plugin.so',
                                  result.stdout)


if __name__=='__main__':unittest.main()
