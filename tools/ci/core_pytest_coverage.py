"""Opt-in pytest plugin: account for actual selected node IDs, not source grep.

Load with ``-p core_pytest_coverage --core-coverage-dir NEW_DIRECTORY``. Explicit
marker deselections are recorded separately and are never reported as passes.
"""
from collections import Counter
import json
from pathlib import Path

import pytest


def pytest_addoption(parser):
    parser.addoption('--core-coverage-dir', help='Fresh result-accounting directory')


def pytest_configure(config):
    config.addinivalue_line('markers', 'torch_interop: optional pinned Torch integration profile')
    config.addinivalue_line('markers', 'mipi_camera: requires real camera/model fixtures')
    path = config.getoption('--core-coverage-dir')
    if not path:
        raise pytest.UsageError('core_pytest_coverage requires --core-coverage-dir')
    junit = getattr(config.option, 'xmlpath', None)
    if junit and Path(junit).exists():
        raise pytest.UsageError('Refuse stale JUnit result path')
    config._core_coverage_dir = Path(path)
    config._core_coverage_dir.mkdir(parents=True, exist_ok=False)
    config._core_coverage_reports = []
    config._core_coverage_selected = []
    config._core_coverage_deselected = []
    config.pluginmanager.register(_Reports(config), 'core-required-results')


def pytest_deselected(items):
    if items:
        items[0].config._core_coverage_deselected.extend(item.nodeid for item in items)


def pytest_collection_finish(session):
    config = session.config
    selected = [item.nodeid for item in session.items]
    config._core_coverage_selected = selected
    for item in session.items:
        item.user_properties.append(('core_nodeid', item.nodeid))
    (config._core_coverage_dir / 'selection.json').write_text(json.dumps({
        'selected': selected, 'deselected': config._core_coverage_deselected,
        'args': list(config.args), 'markexpr': config.option.markexpr,
        'keyword': config.option.keyword}, indent=2) + '\n')
    if not selected or len(selected) != len(set(selected)):
        raise pytest.UsageError('Empty/duplicate required pytest selection')


class _Reports:
    def __init__(self, config):
        self.config = config

    def pytest_runtest_logreport(self, report):
        self.config._core_coverage_reports.append({
            'nodeid': report.nodeid, 'when': report.when, 'outcome': report.outcome})


def validate_reports(selected, reports, exitstatus):
    if not selected or len(selected) != len(set(selected)):
        return False
    for phase in ('setup', 'call', 'teardown'):
        phase_reports = [r for r in reports if r['when'] == phase]
        if Counter(r['nodeid'] for r in phase_reports) != Counter(selected):
            return False
        if any(r['outcome'] != 'passed' for r in phase_reports):
            return False
    return exitstatus == 0 and len(reports) == 3 * len(selected)


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    config = session.config
    passed = validate_reports(config._core_coverage_selected,
                              config._core_coverage_reports, int(exitstatus))
    (config._core_coverage_dir / 'coverage.json').write_text(json.dumps({
        'status': 'PASS' if passed else 'FAIL', 'pytest_exitstatus': int(exitstatus),
        'reports': config._core_coverage_reports,
        'deselected': config._core_coverage_deselected}, indent=2) + '\n')
    if not passed and int(exitstatus) == 0:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
