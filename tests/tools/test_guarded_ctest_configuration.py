#!/usr/bin/env python3
"""Configure-only native fixture: prove proxy preserves argv and properties."""
import json
from pathlib import Path
import re
import subprocess
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


if __name__=='__main__':unittest.main()
