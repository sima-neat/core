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


def test_manual_gate_reuses_canonical_guarded_lane():
    manual = (ROOT / '.github/workflows/zero-skip.yml').read_text()
    wrapper = (ROOT / 'scripts/ci/run_zero_skip_gate.sh').read_text()
    assert 'uses: ./.github/workflows/vulcan-ci.yml' in manual
    assert 'cmake --build' not in wrapper and '--require-guarded' in wrapper


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
    'test_perf_schema', 'test_perf_classifier', 'test_perf_comparator',
    'test_perf_matrix_failfast', 'test_graph_run_schema', 'test_graph_run_visualizer',
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
