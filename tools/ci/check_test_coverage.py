#!/usr/bin/env python3
"""Account for the exact canonical CTest selection and its fresh results.

This does not grant hardware admission, alter test argv/deadlines, build tests,
or launch native executables itself. Guarded registrations remain authoritative.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


class CoverageError(ValueError):
    """Required selected coverage was not completed successfully."""


def require(condition, message):
    if not condition:
        raise CoverageError(message)


def is_host_contract(test):
    """Only explicitly reviewed source-only contracts bypass native admission.

    A label or Python-looking command alone is not authority: require the same
    interpreter, exact checked-in script bytes and the reviewed argv shape.
    """
    scripts = {
        'unit_kernel_runner_default_contract': 'tests/tools/check_kernel_runner_default.py',
        'unit_python_runtime_bootstrap': 'tests/tools/test_python_runtime_bootstrap.py',
        'test_codec_fixture_preparation': 'tests/perf/tools/test_codec_fixture_preparation.py',
        'test_perf_schema': 'tests/perf/tools/test_perf_schema.py',
        'test_perf_classifier': 'tests/perf/tools/test_perf_classifier.py',
        'test_perf_comparator': 'tests/perf/tools/test_perf_comparator.py',
        'test_perf_matrix_failfast': 'tests/perf/tools/test_perf_matrix_failfast.py',
        'test_perf_encoder_matrix': 'tests/perf/tools/test_perf_encoder_matrix.py',
        'test_graph_run_schema': 'tests/perf/tools/test_graph_run_schema.py',
        'test_graph_run_visualizer': 'tests/perf/tools/test_graph_run_visualizer.py',
    }
    expected = scripts.get(test.get('name'))
    command = test.get('command', [])
    if expected is None:
        return False
    source_root = Path(__file__).resolve().parents[2]
    if test.get('name') == 'unit_kernel_runner_default_contract':
        # This is a build-tree source gate, not an installed runtime test.
        # Never let a caller redirect its source inspection to another tree.
        if (len(command) != 4 or command[2] != '--source-root'
                or not Path(command[3]).is_absolute()):
            return False
        try:
            if Path(command[3]).resolve(strict=True) != source_root:
                return False
        except OSError:
            return False
    elif len(command) != 2:
        return False
    source = source_root / expected
    script = Path(command[1])
    if not script.is_absolute():
        props = {p['name']: p['value'] for p in test.get('properties', [])}
        directory = props.get('WORKING_DIRECTORY')
        if not isinstance(directory, str) or not Path(directory).is_absolute():
            return False
        script = Path(directory) / script
    try:
        return (Path(command[0]).resolve(strict=True) == Path(sys.executable).resolve(strict=True)
                and script.read_bytes() == source.read_bytes())
    except OSError:
        return False


def is_guarded_registration(test):
    command = test.get('command', [])
    if (len(command) < 7 or command[2:4] != ['ctest-proxy', '--test-name']
            or command[4] != test.get('name') or command[5] != '--'):
        return False
    source = Path(__file__).resolve().parents[2] / 'tools/hardware_test_supervisor.py'
    try:
        return (Path(command[0]).resolve(strict=True) == Path(sys.executable).resolve(strict=True)
                and Path(command[1]).read_bytes() == source.read_bytes())
    except OSError:
        return False


def validate_selection(inventory, profile, *, require_guarded=False, environment=None):
    tests = inventory.get('tests', [])
    names = [test.get('name') for test in tests]
    require(names and all(isinstance(n, str) and n for n in names), 'Empty/invalid mandatory selection')
    require(len(names) == len(set(names)), 'Duplicate selected IDs')
    environment = os.environ if environment is None else environment
    if profile == 'strict':
        require(environment.get('SIMA_TEST_IS_LONG', '0') in ('', '0'), 'Leaked long environment')
    for test in tests:
        props = {p['name']: p['value'] for p in test.get('properties', [])}
        require(not props.get('DISABLED', False), f"Disabled mandatory test: {test['name']}")
        labels = props.get('LABELS', [])
        if profile == 'strict':
            require('long' not in props.get('LABELS', []), f"Long test in strict non-long selection: {test['name']}")
        # Required lanes retain non-strict capability registrations, but their
        # actual skips still fail fresh JUnit accounting. Strict non-long tests
        # must never acquire a skip escape, even inside a mixed required lane.
        if profile == 'strict' or ('strict' in labels and 'long' not in labels):
            require(environment.get('SIMA_TEST_IS_LONG', '0') in ('', '0'),
                    f"Leaked long environment: {test['name']}")
            require(not {'SKIP_RETURN_CODE', 'SKIP_REGULAR_EXPRESSION'} & props.keys(),
                    f"Strict skip escape: {test['name']}")
            env = props.get('ENVIRONMENT', [])
            if isinstance(env, str):
                env = env.split(';')
            require(not any(v.startswith('SIMA_TEST_IS_LONG=') and v.split('=', 1)[1] not in ('', '0') for v in env),
                    f"Leaked long property: {test['name']}")
        if require_guarded:
            require(is_guarded_registration(test) or ('host' in labels and is_host_contract(test)),
                    f"Native registration is not guarded: {test['name']}")
    return names


def validate_results(names, junit, execution_status, started_ns=None):
    path = Path(junit)
    require(path.is_file(), 'Missing fresh JUnit results')
    if started_ns is not None:
        require(path.stat().st_mtime_ns >= started_ns, 'Stale JUnit results')
    root = ET.parse(path).getroot()
    cases = list(root.iter('testcase'))
    actual = [case.get('name') for case in cases]
    require(Counter(actual) == Counter(names),
            f'Selected/result IDs differ: missing={list((Counter(names)-Counter(actual)).elements())}, '
            f'unexpected={list((Counter(actual)-Counter(names)).elements())}')
    bad = [case.get('name') for case in cases if any(case.find(tag) is not None for tag in
            ('skipped', 'failure', 'error')) or case.get('status', 'run') != 'run']
    require(not bad, f'Required tests did not pass: {bad}')
    require(execution_status == 0, f'CTest exited {execution_status}')
    return {'status': 'PASS', 'selected': len(names), 'passed': len(cases), 'skipped': 0}


def run_ctest(command, output, profile, require_guarded):
    require(command and Path(command[0]).name == 'ctest', 'Pass the canonical ctest command after --')
    require(not any(arg.startswith(('--output-junit', '--show-only', '--repeat', '--rerun-failed')) or arg in ('-N', '-S') for arg in command),
            'The accounting wrapper owns JUnit/selection; repeated/script execution needs distinct accounting')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)  # Never reuse stale evidence.
    selected = subprocess.run(command + ['--show-only=json-v1'], text=True, capture_output=True)
    (output / 'collection.stderr').write_text(selected.stderr)
    require(selected.returncode == 0, f'CTest collection failed: {selected.stderr}')
    (output / 'selection.json').write_text(selected.stdout)
    inventory = json.loads(selected.stdout)
    names = validate_selection(inventory, profile, require_guarded=require_guarded)
    started = time.time_ns()
    (output / 'execution.json').write_text(json.dumps({'argv': command, 'profile': profile,
                                                      'started_ns': started}, indent=2) + '\n')
    junit = (output / 'junit.xml').resolve()
    # No timeout, signal handling, test binary substitution, or altered scheduling.
    result = subprocess.run(command + ['--output-junit', str(junit)])
    (output / 'exit-status.txt').write_text(str(result.returncode) + '\n')
    summary = validate_results(names, junit, result.returncode, started)
    (output / 'coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('strict', 'required'), default='strict')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--require-guarded', action='store_true')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        print(json.dumps(run_ctest(command, args.output_dir, args.profile, args.require_guarded)))
    except (CoverageError, OSError, ValueError, ET.ParseError) as exc:
        print(f'[required-coverage] FAIL: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
