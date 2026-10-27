"""Host-only registration/result fixtures, not native hardware qualification."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
from unittest.mock import patch

import pytest
import yaml

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


def genai_workflow_step(name):
    workflow = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())
    return next(step['run'] for step in workflow['jobs']['test-devkit']['steps']
                if step['name'] == name)


@pytest.mark.parametrize('outcome', ['pass', 'skip', 'disabled', 'empty', 'collection_failure'])
def test_isolated_genai_ctest_workflow_requires_each_result(tmp_path, outcome):
    tests = tmp_path / 'extras/tests'
    tests.mkdir(parents=True)
    work = tmp_path / '_work'
    work.mkdir()
    (work / 'extras_lib_dir.txt').write_text(str(tests.parent))
    (work / 'gst_registry_dir.txt').write_text(str(tmp_path / 'registry'))
    for directory in ('scripts', 'tools'):
        (tmp_path / directory).symlink_to(ROOT / directory, target_is_directory=True)
    marker = tmp_path / 'executed.txt'
    script = tmp_path / 'fixture.py'
    script.write_text('import os, pathlib, sys\n'
                      f'with pathlib.Path({str(marker)!r}).open("a") as output:\n'
                      '    output.write(str(os.getpid()) + "\\n")\n'
                      'raise SystemExit(int(sys.argv[1]))\n')
    label = 'other' if outcome == 'empty' else 'llima;long'
    registration = ''
    for index in range(2):
        status = 77 if outcome == 'skip' and index == 1 else 0
        registration += (f'add_test(genai_{index} "{sys.executable}" "{script}" "{status}")\n'
                         f'set_tests_properties(genai_{index} PROPERTIES LABELS "{label}" '
                         'SKIP_RETURN_CODE 77 TIMEOUT 17)\n')
    if outcome == 'disabled':
        registration += 'set_tests_properties(genai_1 PROPERTIES DISABLED TRUE)\n'
    elif outcome == 'collection_failure':
        registration += 'invalid_ctest_command()\n'
    (tests / 'CTestTestfile.cmake').write_text(registration)
    env = dict(os.environ)
    for name in ('LLIMA_MODELS_PATH', 'SIMA_TEST_LLIMA_TEXT_MODEL', 'SIMA_TEST_LLIMA_VLM_MODEL',
                 'SIMA_TEST_LLIMA_ASR_MODEL', 'SIMA_TEST_LLIMA_REASONING_QWEN_MODEL',
                 'SIMA_TEST_LLIMA_REASONING_GEMMA_MODEL'):
        env[name] = 'host-fixture'
    env.pop('SIMA_HARDWARE_SUPERVISOR_SOCKET', None)
    env.pop('SIMA_TEST_IS_LONG', None)
    command = ['bash', '-c', genai_workflow_step('Run GenAI / LLiMa E2E Tests')]
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert (result.returncode == 0) == (outcome == 'pass'), result.stdout + result.stderr
    executed = marker.read_text().splitlines() if marker.exists() else []
    assert len(executed) == {'pass': 2, 'skip': 2, 'disabled': 1}.get(outcome, 0)
    assert len(set(executed)) == len(executed)
    if outcome == 'pass':
        for index in range(2):
            directory = work / 'coverage/genai-cpp' / str(index)
            assert (directory / 'junit.xml').is_file()
            assert json.loads((directory / 'coverage.json').read_text())['passed'] == 1
            recorded = json.loads((directory / 'execution.json').read_text())['argv']
            assert recorded[recorded.index('--tests-regex') + 1] == f'^genai_{index}$'
            assert recorded[recorded.index('--timeout') + 1] == '900'
        repeated = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
        assert repeated.returncode != 0
        assert marker.read_text().splitlines() == executed


@pytest.mark.parametrize('outcome', ['pass', 'skip', 'setup_skip', 'missing', 'empty', 'collection_failure'])
def test_isolated_genai_pytest_workflow_requires_each_result(tmp_path, outcome):
    (tmp_path / '_work').mkdir()
    (tmp_path / 'pytest.ini').write_text('[pytest]\nmarkers = torch_interop: optional capability\n')
    marker = tmp_path / 'executed.txt'
    script = tmp_path / 'test_genai_bindings.py'
    body = ('import os, pathlib, pytest\n'
            f'MARKER = pathlib.Path({str(marker)!r})\n'
            'def record():\n'
            '    with MARKER.open("a") as output: output.write(str(os.getpid()) + "\\n")\n'
            'def test_first():\n'
            '    record()\n'
            '    pathlib.Path("first-completed").touch()\n'
            '@pytest.mark.torch_interop\n'
            'def test_excluded(): pytest.fail("explicit capability exclusion did not apply")\n'
            'def test_second():\n'
            '    record()\n')
    if outcome == 'skip':
        body += '    pytest.skip("DispatcherMLASHM not available")\n'
    elif outcome == 'setup_skip':
        body += ('@pytest.fixture(autouse=True)\n'
                 'def require_dispatcher(request):\n'
                 '    if request.node.name == "test_second": pytest.skip("dispatcher unavailable")\n')
    elif outcome == 'missing':
        body += 'if pathlib.Path("first-completed").exists(): del test_second\n'
    elif outcome == 'empty':
        body = 'def helper(): pass\n'
    elif outcome == 'collection_failure':
        body += 'raise RuntimeError("collection failed")\n'
    script.write_text(body)
    step = genai_workflow_step('Run Python pytest from installed extras')
    start = step.index('"${PYTHON_BIN}" -m pytest --collect-only')
    stop = step.index('# Temporary external RTSP exclusions;', start)
    command = ['bash', '-c', 'set -euo pipefail\nPYTHON_MARKS=(-m "not torch_interop")\n' + step[start:stop]]
    env = {**os.environ, 'PYTHON_BIN': sys.executable, 'genai_test_file': str(script),
           'PYTHONPATH': str(ROOT / 'tools/ci')}
    result = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
    assert (result.returncode == 0) == (outcome == 'pass'), result.stdout + result.stderr
    executed = marker.read_text().splitlines() if marker.exists() else []
    assert len(executed) == {'pass': 2, 'skip': 2, 'setup_skip': 1, 'missing': 1}.get(outcome, 0)
    assert len(set(executed)) == len(executed)
    if outcome in ('pass', 'skip', 'setup_skip'):
        for index in range(2):
            directory = tmp_path / '_work/coverage/genai-python' / str(index)
            assert (directory / 'junit.xml').is_file()
            selection = json.loads((directory / 'selection.json').read_text())
            assert len(selection['selected']) == 1
            assert selection['markexpr'] == 'not torch_interop'
            report = json.loads((directory / 'coverage.json').read_text())
            assert report['status'] == ('FAIL' if index == 1 and outcome != 'pass' else 'PASS')
    if outcome == 'pass':
        repeated = subprocess.run(command, cwd=tmp_path, env=env, text=True, capture_output=True)
        assert repeated.returncode != 0
        assert marker.read_text().splitlines() == executed


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


def test_kernel_runner_contract_has_required_source_gate_not_installed_registration():
    workflow = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())
    steps = workflow['jobs']['build-tooling-tests']['steps']
    step = next(item for item in steps if item['name'] == 'Run build-tooling tests')
    commands = step['run'].replace('\\\n', ' ').splitlines()
    pytest_command = next(line for line in commands if ' -m pytest ' in line)
    assert shlex.split(pytest_command).count('tests/tools/check_kernel_runner_default.py') == 1
    assert 'if' not in step and not step.get('continue-on-error', False)
    for relative in ('CMakeLists.txt', 'tests/CMakeLists.txt'):
        assert 'NAME unit_kernel_runner_default_contract' not in (ROOT / relative).read_text()


def bootstrap_host_selection(tmp_path):
    script = tmp_path / 'test_python_runtime_bootstrap.py'
    script.write_bytes((ROOT / 'tests/tools/test_python_runtime_bootstrap.py').read_bytes())
    subject = tmp_path / 'pyneat_init.py'
    subject.write_bytes((ROOT / 'python/pyneat/__init__.py').read_bytes())
    candidate = selection([('LABELS', ['host', 'strict', 'unit', 'python']), ('TIMEOUT', 10)])
    candidate['tests'][0].update(
        name='unit_python_runtime_bootstrap',
        command=[sys.executable, str(script), '--bootstrap', str(subject)],
    )
    return candidate


def test_bootstrap_host_contract_accepts_exact_installed_subject(tmp_path):
    candidate = bootstrap_host_selection(tmp_path)
    assert coverage.validate_selection(candidate, 'required', require_guarded=True, environment={}) == [
        'unit_python_runtime_bootstrap']


@pytest.mark.parametrize('change', [
    'missing_subject', 'substituted_subject', 'relative_subject', 'wrong_flag',
    'missing_argument', 'extra_argument', 'substituted_script', 'wrong_interpreter',
])
def test_bootstrap_host_contract_rejects_unreviewed_subject(tmp_path, change):
    candidate = bootstrap_host_selection(tmp_path)
    command = candidate['tests'][0]['command']
    if change == 'missing_subject':
        Path(command[3]).unlink()
    elif change == 'substituted_subject':
        Path(command[3]).write_text('def _configure_neat_runtime_environment(): pass\n')
    elif change == 'relative_subject':
        command[3] = 'pyneat_init.py'
    elif change == 'wrong_flag':
        command[2] = '--source-root'
    elif change == 'missing_argument':
        command.pop()
    elif change == 'extra_argument':
        command.append('--unreviewed')
    elif change == 'substituted_script':
        Path(command[1]).write_text('raise SystemExit(0)\n')
    else:
        command[0] = '/unreviewed/python'
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
