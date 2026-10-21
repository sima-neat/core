"""Host-only registration/result fixtures, not native hardware qualification."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('coverage', ROOT / 'tools/ci/check_test_coverage.py')
coverage = importlib.util.module_from_spec(spec); spec.loader.exec_module(coverage)


def selection(properties=()):
    return {'tests': [{'name': 'fixture', 'command': [sys.executable, str(ROOT / 'tools/hardware_test_supervisor.py'), 'ctest-proxy', '--test-name', 'fixture', '--', '/fixture', '--unchanged'],
                       'properties': [{'name': k, 'value': v} for k, v in properties]}]}


def junit(path, inner='', name='fixture'):
    path.write_text(f'<testsuite><testcase name="{name}" status="run">{inner}</testcase></testsuite>')


@pytest.mark.parametrize('properties', [
    [('DISABLED', True)], [('SKIP_RETURN_CODE', 77)], [('SKIP_REGULAR_EXPRESSION', 'skip')],
    [('ENVIRONMENT', ['SIMA_TEST_IS_LONG=1'])], [('LABELS', ['strict', 'long'])],
])
def test_strict_registration_rejects_escape(properties):
    with pytest.raises(coverage.CoverageError):
        coverage.validate_selection(selection(properties), 'strict', environment={})


def test_long_required_accepts_registration_not_actual_skip(tmp_path):
    names = coverage.validate_selection(selection([('SKIP_RETURN_CODE', 77), ('ENVIRONMENT', ['SIMA_TEST_IS_LONG=1']),
                                                   ('LABELS', ['strict', 'long'])]), 'required', environment={})
    path = tmp_path / 'results.xml'; junit(path)
    assert coverage.validate_results(names, path, 0)['passed'] == 1
    junit(path, '<skipped message="SKIP_RETURN_CODE=77"/>')
    with pytest.raises(coverage.CoverageError, match='did not pass'):
        coverage.validate_results(names, path, 0)


@pytest.mark.parametrize('inventory', [{'tests': []}, {'tests': selection()['tests'] * 2}])
def test_rejects_empty_duplicate_selection(inventory):
    with pytest.raises(coverage.CoverageError):
        coverage.validate_selection(inventory, 'strict', environment={})


def test_rejects_unguarded_and_leaked_environment():
    inventory = selection(); inventory['tests'][0]['command'] = ['/native']
    with pytest.raises(coverage.CoverageError, match='not guarded'):
        coverage.validate_selection(inventory, 'strict', require_guarded=True, environment={})
    with pytest.raises(coverage.CoverageError, match='Leaked long'):
        coverage.validate_selection(selection(), 'strict', environment={'SIMA_TEST_IS_LONG': '1'})


@pytest.mark.parametrize('inner', ['<skipped/>', '<failure/>', '<error/>'])
def test_required_results_reject_nonpass(tmp_path, inner):
    path = tmp_path / 'junit.xml'; junit(path, inner)
    with pytest.raises(coverage.CoverageError):
        coverage.validate_results(['fixture'], path, 0)


def test_result_completeness_exit_and_freshness(tmp_path):
    path = tmp_path / 'junit.xml'
    with pytest.raises(coverage.CoverageError, match='Missing'):
        coverage.validate_results(['fixture'], path, 0)
    junit(path)
    for names, status, start in [(['missing'], 0, None), (['fixture', 'fixture'], 0, None),
                                  (['fixture'], 1, None), (['fixture'], 0, time.time_ns()+10**9)]:
        with pytest.raises(coverage.CoverageError):
            coverage.validate_results(names, path, status, start)


def test_command_argv_and_deadlines_preserved(tmp_path):
    command = ['ctest', '--test-dir', '/installed/tests', '-L', 'strict', '--timeout', '240', '-j', '4']
    calls = []
    def execute(argv, **kwargs):
        calls.append(argv)
        if '--show-only=json-v1' in argv:
            return subprocess.CompletedProcess(argv, 0, json.dumps(selection()), '')
        junit(Path(argv[-1]))
        return subprocess.CompletedProcess(argv, 0)
    with patch.object(coverage.subprocess, 'run', execute):
        coverage.run_ctest(command, tmp_path / 'new', 'strict', True)
    assert calls[0] == command + ['--show-only=json-v1']
    assert calls[1][:-2] == command
    with pytest.raises(FileExistsError):
        coverage.run_ctest(command, tmp_path / 'new', 'strict', True)


@pytest.mark.parametrize('body,expected', [
    ('def test_ok(): assert True', 0),
    ('import pytest\ndef test_skip(): pytest.skip("missing required fixture")', 1),
    ('import pytest\n@pytest.mark.skip(reason="missing")\ndef test_skip(): pass', 1),
    ('def test_failure(): assert False', 1),
    ('def test_ok(): assert True\ndef test_omitted(): assert False', 0),
])
def test_pytest_actual_node_accounting(tmp_path, body, expected):
    (tmp_path / 'test_fixture.py').write_text(body)
    env = {**os.environ, 'PYTHONPATH': str(ROOT / 'tools/ci')}
    args = [sys.executable, '-m', 'pytest', '-q', '-p', 'core_pytest_coverage',
            '--core-coverage-dir', str(tmp_path / 'coverage'), str(tmp_path / 'test_fixture.py')]
    if 'test_omitted' in body:
        args += ['-k', 'not omitted']
    result = subprocess.run(args, env=env, text=True, capture_output=True)
    assert result.returncode == expected, result.stdout + result.stderr
    report = json.loads((tmp_path / 'coverage/coverage.json').read_text())
    assert report['status'] == ('PASS' if expected == 0 else 'FAIL')
    if 'test_omitted' in body:
        assert len(report['deselected']) == 1


def test_manual_gate_reuses_canonical_lane_without_forcing_guarded_execution():
    manual = (ROOT / '.github/workflows/zero-skip.yml').read_text()
    wrapper = (ROOT / 'scripts/ci/run_zero_skip_gate.sh').read_text()
    assert 'uses: ./.github/workflows/vulcan-ci.yml' in manual
    assert 'cmake --build' not in wrapper and '--require-guarded' not in wrapper


def direct_ctest_fixture(tmp_path, *, returncode=0, properties='LABELS "strict" TIMEOUT "17"'):
    """Use real CTest and a resource-free Python script, not a substitute checker."""
    script = tmp_path / 'fixture.py'
    marker = tmp_path / 'executed.json'
    script.write_text(
        'import json, pathlib, sys\n'
        f'pathlib.Path({str(marker)!r}).write_text(json.dumps(sys.argv[1:]))\n'
        f'raise SystemExit({returncode})\n'
    )
    (tmp_path / 'CTestTestfile.cmake').write_text(
        f'add_test(fixture "{sys.executable}" "{script}" "--unchanged" "two words")\n'
        f'set_tests_properties(fixture PROPERTIES {properties})\n'
        f'add_test(not_selected "{sys.executable}" "-c" "raise SystemExit(99)")\n'
    )
    env = dict(os.environ)
    for key in ('SIMA_HARDWARE_SUPERVISOR_SOCKET', 'SIMA_TEST_IS_LONG'):
        env.pop(key, None)
    env.update(COVERAGE_DIR=str(tmp_path / 'coverage'), COVERAGE_PROFILE='strict')
    command = ['ctest', '--test-dir', str(tmp_path), '-R', '^fixture$',
               '--output-on-failure', '--timeout', '23', '-j', '1']
    return env, command, marker


def test_actual_ci_wrapper_runs_direct_registration_and_keeps_accounting(tmp_path):
    env, command, marker = direct_ctest_fixture(tmp_path)
    result = subprocess.run(['bash', str(ROOT / 'scripts/ci/run_zero_skip_gate.sh'), *command],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(marker.read_text()) == ['--unchanged', 'two words']
    output = Path(env['COVERAGE_DIR'])
    assert json.loads((output / 'execution.json').read_text())['argv'] == command
    selected = json.loads((output / 'selection.json').read_text())['tests']
    assert [test['name'] for test in selected] == ['fixture']
    properties = {item['name']: item['value'] for item in selected[0]['properties']}
    assert properties['TIMEOUT'] == 17
    assert json.loads((output / 'coverage.json').read_text()) == {
        'status': 'PASS', 'selected': 1, 'passed': 1, 'skipped': 0,
    }
    marker.unlink()
    reused = subprocess.run(['bash', str(ROOT / 'scripts/ci/run_zero_skip_gate.sh'), *command],
                            env=env, text=True, capture_output=True)
    assert reused.returncode != 0
    assert not marker.exists()


@pytest.mark.parametrize('case,returncode,properties,ran', [
    ('failure', 1, 'LABELS "strict"', True),
    ('actual_skip', 77, 'LABELS "long" SKIP_RETURN_CODE "77"', True),
    ('disabled', 0, 'LABELS "strict" DISABLED "TRUE"', False),
    ('strict_skip_escape', 0, 'LABELS "strict" SKIP_RETURN_CODE "77"', False),
    ('empty_selection', 0, 'LABELS "strict"', False),
])
def test_actual_ci_wrapper_still_rejects_missing_or_nonpass_coverage(
        tmp_path, case, returncode, properties, ran):
    env, command, marker = direct_ctest_fixture(tmp_path, returncode=returncode, properties=properties)
    if case == 'actual_skip':
        env['COVERAGE_PROFILE'] = 'required'
    elif case == 'empty_selection':
        command[command.index('^fixture$')] = '^missing$'
    result = subprocess.run(['bash', str(ROOT / 'scripts/ci/run_zero_skip_gate.sh'), *command],
                            env=env, text=True, capture_output=True)
    assert result.returncode != 0, result.stdout + result.stderr
    assert '[required-coverage] FAIL:' in result.stderr
    assert marker.exists() is ran
    assert not (Path(env['COVERAGE_DIR']) / 'coverage.json').exists()


def test_explicit_guarded_accounting_rejects_direct_registration_without_execution(tmp_path):
    env, command, marker = direct_ctest_fixture(tmp_path)
    result = subprocess.run(
        [sys.executable, str(ROOT / 'tools/ci/check_test_coverage.py'), '--require-guarded',
         '--output-dir', env['COVERAGE_DIR'], '--', *command],
        env=env, text=True, capture_output=True,
    )
    assert result.returncode != 0
    assert 'Native registration is not guarded: fixture' in result.stderr
    assert not marker.exists()


def test_host_label_is_not_native_admission():
    candidate = selection([('LABELS', ['host', 'strict'])])
    candidate['tests'][0]['command'] = ['/native']
    with pytest.raises(coverage.CoverageError, match='not guarded'):
        coverage.validate_selection(candidate, 'required', require_guarded=True, environment={})
    candidate['tests'][0].update(name='unit_python_runtime_bootstrap', command=[sys.executable, str(ROOT / 'tests/tools/test_python_runtime_bootstrap.py')])
    assert coverage.validate_selection(candidate, 'required', require_guarded=True, environment={})
    candidate['tests'][0]['command'].append('--unreviewed')
    with pytest.raises(coverage.CoverageError, match='not guarded'):
        coverage.validate_selection(candidate, 'required', require_guarded=True, environment={})


def test_mixed_required_profile_keeps_nonlong_strict():
    with pytest.raises(coverage.CoverageError, match='skip escape'):
        coverage.validate_selection(selection([('LABELS', ['strict', 'e2e']), ('SKIP_RETURN_CODE', 77)]),
                                    'required', environment={})
    with pytest.raises(coverage.CoverageError, match='Leaked long'):
        coverage.validate_selection(selection([('LABELS', ['strict', 'e2e'])]), 'required',
                                    environment={'SIMA_TEST_IS_LONG': '1'})


def test_required_nonstrict_capability_registration_still_rejects_actual_skip(tmp_path):
    names = coverage.validate_selection(selection([('LABELS', ['e2e']), ('SKIP_RETURN_CODE', 77)]),
                                        'required', environment={})
    path = tmp_path / 'results.xml'; junit(path)
    assert coverage.validate_results(names, path, 0)['passed'] == 1
    junit(path, '<skipped message="unavailable capability"/>')
    with pytest.raises(coverage.CoverageError, match='did not pass'):
        coverage.validate_results(names, path, 0)


@pytest.mark.parametrize('name', [
    'test_codec_fixture_preparation', 'test_perf_schema', 'test_perf_classifier', 'test_perf_comparator',
    'test_perf_matrix_failfast', 'test_graph_run_schema', 'test_graph_run_visualizer',
    'test_perf_encoder_matrix',
])
def test_exact_perf_host_contracts_resolve_registered_working_directory(name, tmp_path):
    script = ROOT / 'tests/perf/tools' / (name + '.py')
    staged = tmp_path / 'perf/tools' / script.name
    staged.parent.mkdir(parents=True); staged.write_bytes(script.read_bytes())
    candidate = selection([('LABELS', ['perf', 'unit', 'host']), ('WORKING_DIRECTORY', str(tmp_path))])
    test = candidate['tests'][0]
    test.update(name=name, command=[sys.executable, 'perf/tools/' + script.name])
    assert coverage.validate_selection(candidate, 'required', require_guarded=True, environment={})
    for command in [[sys.executable, str(staged), '--unreviewed'], ['/not-python', str(staged)]]:
        test['command'] = command
        with pytest.raises(coverage.CoverageError, match='not guarded'):
            coverage.validate_selection(candidate, 'required', require_guarded=True, environment={})
    test['command'] = [sys.executable, str(staged)]
    staged.write_bytes(script.read_bytes() + b'\n# changed\n')
    with pytest.raises(coverage.CoverageError, match='not guarded'):
        coverage.validate_selection(candidate, 'required', require_guarded=True, environment={})


def test_unknown_host_script_and_missing_working_directory_are_not_authority():
    candidate = selection([('LABELS', ['perf', 'host'])])
    test = candidate['tests'][0]
    test.update(name='unknown_host_test', command=[sys.executable, str(ROOT / 'tests/perf/tools/test_perf_schema.py')])
    assert not coverage.is_host_contract(test)
    test.update(name='test_perf_schema', command=[sys.executable, 'perf/tools/test_perf_schema.py'])
    assert not coverage.is_host_contract(test)


def test_nonstrict_label_cannot_exempt_native_command():
    candidate = selection([('LABELS', ['host', 'e2e']), ('SKIP_RETURN_CODE', 77)])
    candidate['tests'][0]['command'] = ['/native']
    with pytest.raises(coverage.CoverageError, match='not guarded'):
        coverage.validate_selection(candidate, 'required', require_guarded=True, environment={})


def test_inventory_workflow_uses_required_accounting_without_changing_selector():
    workflow = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()
    assert 'COVERAGE_DIR=_work/coverage/package-inventory COVERAGE_PROFILE=required' in workflow
    assert '--label-regex "devkit-package-inventory"' in workflow


def kernel_runner_host_selection():
    candidate = selection([('LABELS', ['host', 'strict', 'unit', 'architecture'])])
    candidate['tests'][0].update(
        name='unit_kernel_runner_default_contract',
        command=[sys.executable, str(ROOT / 'tests/tools/check_kernel_runner_default.py'),
                 '--source-root', str(ROOT)],
    )
    return candidate


def test_kernel_runner_host_contract_accepts_registered_source_root_form():
    # Keep the positive fixture tied to the actual build-tree registration.
    registration = (ROOT / 'tests/CMakeLists.txt').read_text().split(
        'simaneat_add_test(NAME unit_kernel_runner_default_contract', 1)[1].split(')', 1)[0]
    assert registration.split() == [
        'COMMAND', '${Python3_EXECUTABLE}',
        '${CMAKE_CURRENT_SOURCE_DIR}/tools/check_kernel_runner_default.py',
        '--source-root', '${PROJECT_SOURCE_DIR}',
    ]
    candidate = kernel_runner_host_selection()
    assert coverage.validate_selection(candidate, 'required', require_guarded=True, environment={}) == [
        'unit_kernel_runner_default_contract']


@pytest.mark.parametrize('change', [
    'wrong_root', 'missing_root', 'relative_root', 'wrong_flag', 'extra_argument',
    'missing_source_root_arguments', 'changed_script', 'wrong_interpreter',
])
def test_kernel_runner_host_contract_rejects_unreviewed_authority(tmp_path, change):
    candidate = kernel_runner_host_selection()
    command = candidate['tests'][0]['command']
    if change == 'wrong_root':
        command[3] = str(tmp_path)
    elif change == 'missing_root':
        command[3] = str(tmp_path / 'not-present')
    elif change == 'relative_root':
        command[3] = os.path.relpath(ROOT)
    elif change == 'wrong_flag':
        command[2] = '--other-root'
    elif change == 'extra_argument':
        command.append('--unreviewed')
    elif change == 'missing_source_root_arguments':
        del command[2:]
    elif change == 'changed_script':
        script = tmp_path / 'check_kernel_runner_default.py'
        script.write_bytes(Path(command[1]).read_bytes() + b'\n# unreviewed change\n')
        command[1] = str(script)
    else:
        interpreter = tmp_path / 'not-python'
        interpreter.write_text('not the approved Python interpreter\n')
        command[0] = str(interpreter)
    with pytest.raises(coverage.CoverageError, match='not guarded'):
        coverage.validate_selection(candidate, 'required', require_guarded=True, environment={})


def test_source_root_arguments_do_not_authorize_other_host_scripts():
    candidate = kernel_runner_host_selection()
    candidate['tests'][0].update(name='unit_python_runtime_bootstrap')
    candidate['tests'][0]['command'][1] = str(ROOT / 'tests/tools/test_python_runtime_bootstrap.py')
    with pytest.raises(coverage.CoverageError, match='not guarded'):
        coverage.validate_selection(candidate, 'required', require_guarded=True, environment={})


def test_encoder_runtime_alias_uses_canonical_guarded_registration():
    cmake = (ROOT / 'tests/CMakeLists.txt').read_text()
    assert 'simaneat_add_test(NAME image_group_encoder_runtime_test' in cmake
    assert '\n  add_test(NAME image_group_encoder_runtime_test' not in cmake
    registration = cmake.split('simaneat_add_test(NAME image_group_encoder_runtime_test', 1)[1].split(')', 1)[0]
    assert registration.split() == [
        'COMMAND', 'unit_image_group_runtime_test', '${CMAKE_SOURCE_DIR}/test.jpg', '--encoder',
    ]


exact_spec = importlib.util.spec_from_file_location(
    'exact_selection', ROOT / 'tools/ci/verify_ctest_exact_selection.py')
exact_selection = importlib.util.module_from_spec(exact_spec)
exact_spec.loader.exec_module(exact_selection)


def exact_lane_regexes():
    import re
    source = (ROOT / 'scripts/ci/run_sanitizer_gate.sh').read_text()
    lanes = {}
    for name in ('asan_ubsan_tests', 'tsan_tests'):
        match = re.search(r'(?ms)^' + name + r'=\(\n(.*?)^\)', source)
        assert match is not None
        lanes[name] = '^(' + '|'.join(match[1].split()) + ')$'
    for workflow in ('test-crash-correctness-nightly', 'test-stress-nightly'):
        source = (ROOT / '.github/workflows' / (workflow + '.yml')).read_text()
        for variable, regex in re.findall(r"(?m)^\s*([A-Z_]*REGEX)='([^'\n]*)'$", source):
            lanes[workflow + ':' + variable] = regex
    assert len(lanes) == 6
    return lanes


@pytest.mark.parametrize('lane,regex', exact_lane_regexes().items())
def test_every_exact_lane_id_is_required_once(lane, regex):
    names = exact_selection.expected_ids(regex)
    inventory = {'tests': [{'name': name} for name in names]}
    assert exact_selection.validate_selection(regex, inventory) == names
    for name in names:
        missing = {'tests': [test for test in inventory['tests'] if test['name'] != name]}
        with pytest.raises(ValueError, match='Exact test IDs differ'):
            exact_selection.validate_selection(regex, missing)
        duplicate = {'tests': inventory['tests'] + [{'name': name}]}
        with pytest.raises(ValueError, match='Exact test IDs differ'):
            exact_selection.validate_selection(regex, duplicate)
        with pytest.raises(ValueError, match='Duplicate requested'):
            exact_selection.expected_ids('^(' + '|'.join(names + [name]) + ')$')


@pytest.mark.parametrize('regex', ['', '.*', '^unit_test$', '^(unit_.*)$', '^(first||last)$'])
def test_exact_lane_rejects_nonliteral_selectors(regex):
    with pytest.raises(ValueError, match='Exact-ID'):
        exact_selection.expected_ids(regex)


def test_exact_lane_rejects_disabled_and_unexpected_ids():
    with pytest.raises(ValueError, match='Disabled requested'):
        exact_selection.validate_selection('^(fixture)$', {'tests': [
            {'name': 'fixture', 'properties': [{'name': 'DISABLED', 'value': True}]}]})
    with pytest.raises(ValueError, match='Exact test IDs differ'):
        exact_selection.validate_selection('^(fixture)$', {'tests': [
            {'name': 'fixture'}, {'name': 'unexpected'}]})


def test_exact_lane_cli_only_collects_registration(tmp_path, monkeypatch, capsys):
    calls = []
    def collect(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, json.dumps({'tests': [{'name': 'fixture'}]}), '')
    monkeypatch.setattr(sys, 'argv', ['verify_ctest_exact_selection.py', '--test-dir', str(tmp_path),
                                     '--regex', '^(fixture)$'])
    monkeypatch.setattr(exact_selection.subprocess, 'run', collect)
    assert exact_selection.main() == 0
    assert calls == [['ctest', '--test-dir', str(tmp_path), '--show-only=json-v1',
                      '--fixture-exclude-any', '.*', '-R', '^(fixture)$']]
    assert json.loads(capsys.readouterr().out)['native_execution'] is False


def test_exact_lane_gate_precedes_each_execution_and_preserves_scope():
    sanitizer = (ROOT / 'scripts/ci/run_sanitizer_gate.sh').read_text()
    guard = 'python3 tools/ci/verify_ctest_exact_selection.py --test-dir "${TEST_DIR}" --regex "${regex}"'
    assert guard in sanitizer
    assert sanitizer.index(guard) < sanitizer.index('ctest --test-dir "${TEST_DIR}"')
    for name, variables in [('test-crash-correctness-nightly', ('CRITICAL_REGEX', 'REPEAT_REGEX')),
                            ('test-stress-nightly', ('STRESS_REGEX', 'REPEAT_REGEX'))]:
        workflow = (ROOT / '.github/workflows' / (name + '.yml')).read_text()
        for variable in variables:
            guard = 'python3 tools/ci/verify_ctest_exact_selection.py --test-dir "${TEST_DIR}" --regex "${' + variable + '}"'
            assert workflow.count(guard) == 1
            execution = workflow.index('-R "${' + variable + '}"')
            assert workflow.index(guard) < execution
        assert '--repeat "until-fail:${REPEAT_COUNT}"' in workflow
        assert "github.event.inputs.repeat_count || '20'" in workflow
    stress = (ROOT / '.github/workflows/test-stress-nightly.yml').read_text()
    assert "github.event.inputs.stress_iters || '120'" in stress


def test_exact_lane_ignores_automatic_fixture_ids_only_during_collection(tmp_path, monkeypatch):
    (tmp_path / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.16)
project(exact_ids NONE)
enable_testing()
add_test(NAME setup COMMAND never_execute_this_test)
add_test(NAME selected COMMAND never_execute_this_test)
add_test(NAME cleanup COMMAND never_execute_this_test)
set_tests_properties(setup PROPERTIES FIXTURES_SETUP example)
set_tests_properties(cleanup PROPERTIES FIXTURES_CLEANUP example)
set_tests_properties(selected PROPERTIES FIXTURES_REQUIRED example)
''')
    build = tmp_path / 'build'
    subprocess.run(['cmake', '-S', str(tmp_path), '-B', str(build)],
                   check=True, capture_output=True)
    command = ['ctest', '--test-dir', str(build), '--show-only=json-v1', '-R', '^(selected)$']
    regular = json.loads(subprocess.check_output(command, text=True))
    assert {test['name'] for test in regular['tests']} == {'setup', 'selected', 'cleanup'}
    monkeypatch.setattr(sys, 'argv', ['verify_ctest_exact_selection.py', '--test-dir', str(build),
                                     '--regex', '^(selected)$'])
    assert exact_selection.main() == 0
    # The preflight changed neither registration nor the real execution command.
    after = json.loads(subprocess.check_output(command, text=True))
    assert {test['name'] for test in after['tests']} == {'setup', 'selected', 'cleanup'}
