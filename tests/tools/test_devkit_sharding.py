"""Scheduling/coverage contracts, with real host-only CTest fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f'tools/ci/{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


planner = load('plan_devkit_shards')
partitioner = load('partition_devkit_e2e')


def runner(i, *, busy=False, status='online', labels=('rtsp', 'firmware')):
    return {'name': f'devkit-{i}', 'busy': busy, 'status': status,
            'labels': [{'name': x} for x in labels]}


@pytest.mark.parametrize('count', range(1, 10))
def test_adapt_to_idle_capacity_and_cover_all_lanes(count):
    plan = planner.make_plan([runner(i) for i in range(count)], ['rtsp', 'firmware'], 'sha')
    assert len(plan['shards']) == min(count, 6)
    assert sorted(t for s in plan['shards'] for t in s['tasks']) == sorted(planner.LANES)
    planner.validate_plan(plan)


def test_busy_fallback_and_exact_firmware_admission():
    plan = planner.make_plan([runner(1, busy=True), runner(2, status='offline'),
                              runner(3, labels=('rtsp', 'wrong-firmware'))],
                             ['RTSP', 'firmware'], 'sha')
    assert len(plan['shards']) == 1
    assert not plan['shards'][0]['idle_at_planning']
    with pytest.raises(ValueError, match='No online'):
        planner.make_plan([runner(1, status='offline')], ['rtsp'], 'sha')


def test_idle_workers_preferred_over_busy():
    plan = planner.make_plan([runner(0, busy=True), runner(1), runner(2)], ['rtsp'], 'sha')
    assert [s['candidate'] for s in plan['shards']] == ['devkit-1', 'devkit-2']


@pytest.mark.parametrize('damage', ['missing', 'duplicate', 'unknown', 'empty', 'duplicate-id'])
def test_corrupt_plan_rejected(damage):
    plan = planner.make_plan([runner(1), runner(2)], ['rtsp'], 'sha')
    if damage == 'missing': plan['shards'][0]['tasks'].pop()
    if damage == 'duplicate': plan['shards'][0]['tasks'].append('fast')
    if damage == 'unknown': plan['shards'][0]['tasks'][0] = 'new-lane'
    if damage == 'empty': plan['shards'][0]['labels'] = []
    if damage == 'duplicate-id': plan['shards'][1]['id'] = plan['shards'][0]['id']
    with pytest.raises(ValueError): planner.validate_plan(plan)


def results(tmp_path):
    plan = planner.make_plan([runner(1), runner(2)], ['rtsp'], 'sha')
    for s in plan['shards']:
        d = tmp_path / s['id']; d.mkdir()
        (d / 'shard-receipt.json').write_text(json.dumps(
            {'id': s['id'], 'plan_id': plan['plan_id'], 'sha': 'sha', 'tasks': s['tasks'], 'result': 'success'}))
    for i in range(2):
        (tmp_path / f'devkit-{i+1}' / 'e2e-partition.json').write_text(json.dumps(
            {'index': i, 'canonical': ['a', 'b'], 'selected': [['a'], ['b']][i]}))
    return plan


@pytest.mark.parametrize('damage', [None, 'missing', 'failed', 'sha', 'tasks', 'overlap', 'inventory'])
def test_aggregate_requires_every_result_and_exact_partition_union(tmp_path, damage):
    plan = results(tmp_path)
    p = tmp_path / 'devkit-1/shard-receipt.json'
    if damage == 'missing': p.unlink()
    elif damage in ('failed', 'sha', 'tasks'):
        data = json.loads(p.read_text())
        data[{'failed': 'result', 'sha': 'sha', 'tasks': 'tasks'}[damage]] = (
            [] if damage == 'tasks' else 'bad')
        p.write_text(json.dumps(data))
    elif damage in ('overlap', 'inventory'):
        p = tmp_path / 'devkit-1/e2e-partition.json'
        data = json.loads(p.read_text())
        data['selected' if damage == 'overlap' else 'canonical'] = ['b']
        p.write_text(json.dumps(data))
    if damage:
        with pytest.raises(ValueError): planner.verify_results(plan, tmp_path)
    else: planner.verify_results(plan, tmp_path)


def test_weighted_partition_is_deterministic_and_complete():
    names = ['a', 'b', 'c', 'd']
    bins = partitioner.partition(names, {'a': 100, 'b': 90, 'c': 10, 'd': 20})
    assert bins == partitioner.partition(list(reversed(names)), {'a': 100, 'b': 90, 'c': 10, 'd': 20})
    assert sorted(n for b in bins for n in b) == names
    assert set(bins[0]).isdisjoint(bins[1])
    with pytest.raises(ValueError): partitioner.partition(['a', 'a'], {})


def test_real_ctest_partition_executes_all_selected_tests(tmp_path):
    # Uses harmless host Python registrations; no aarch64 binary is executed.
    marker = tmp_path / 'executed'
    helper = tmp_path / 'fixture.py'
    helper.write_text('from pathlib import Path\nimport sys\n'
                      f'with Path({str(marker)!r}).open("a") as f: f.write(sys.argv[1]+"\\n")\n')
    (tmp_path / 'CTestTestfile.cmake').write_text(''.join(
        f'add_test(case_{i} "{sys.executable}" "{helper}" "{i}")\n'
        f'set_tests_properties(case_{i} PROPERTIES LABELS "e2e")\n' for i in range(6)))
    for i in range(2):
        env = dict(os.environ, COVERAGE_DIR=str(tmp_path / f'results-{i}'), COVERAGE_PROFILE='required')
        subprocess.run([sys.executable, str(ROOT / 'tools/ci/partition_devkit_e2e.py'),
                        '--index', str(i), '--', 'ctest', '--test-dir', str(tmp_path),
                        '-L', 'e2e', '--no-tests=error', '--output-on-failure'], env=env, check=True)
    assert sorted(marker.read_text().splitlines()) == list('012345')


def test_workflow_guards_and_release_gate():
    jobs = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())['jobs']
    shard = jobs['test-devkit-shard']
    assert shard['strategy']['fail-fast'] is False
    assert jobs['test-devkit']['name'] == 'Test Core on DevKit'
    assert 'test-devkit-shard' in jobs['test-devkit']['needs']
    assert 'test-devkit' in jobs['promote-latest-tag']['needs']
    assert 'test-devkit' in jobs['validation-complete']['needs']
    for step in shard['steps']:
        if step.get('name') in ('Run Tutorial Tests', 'Run Python pytest from installed extras'):
            assert "'python-tutorials'" in step['if']
        if step.get('name') == 'Prepare GenAI models':
            assert "'genai'" in step['if'] and "'python-tutorials'" in step['if']


def test_group_visibility_and_pagination():
    calls = []
    def get(path):
        calls.append(path)
        if '/repositories?' in path:
            return {'repositories': [{'full_name': 'sima-neat/core'}]}
        if '/runners?' in path:
            return {'runners': [runner(1)]}
        return {'runner_groups': [{'name': 'modalix-devkit', 'id': 7, 'visibility': 'selected'}]}
    assert planner.discover_runners(get, 'sima-neat/core', 'modalix-devkit', True)[0]['name'] == 'devkit-1'
    with pytest.raises(ValueError, match='no access'):
        planner.discover_runners(get, 'sima-neat/other', 'modalix-devkit', True)
    with pytest.raises(ValueError, match='public'):
        planner.discover_runners(get, 'sima-neat/core', 'modalix-devkit', False)
    def pages(path):
        return {'runners': [runner(i) for i in range(100)] if path.endswith('&page=1') else [runner(100)]}
    assert len(planner.collect(pages, '/runners', 'runners')) == 101


def test_reruns_replace_only_the_reexecuted_shards_artifact():
    jobs = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())['jobs']
    steps = jobs['test-devkit-shard']['steps']
    upload = next(s for s in steps if s.get('name') == 'Upload selected coverage and outcomes')['with']
    assert upload['overwrite'] is True
    assert 'github.run_attempt' not in upload['name']
    assert 'matrix.shard' in upload['name'] and 'github.run_id' in upload['name']
    for gate in ('publish-offline-develop', 'validation-complete'):
        assert {'plan-devkit-tests', 'test-devkit-shard', 'test-devkit'} <= set(jobs[gate]['needs'])


def test_offline_only_and_fuzz_runs_do_not_require_hardware_shards():
    jobs = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())['jobs']
    gate = jobs['test-devkit']['if']
    assert '!inputs.offline_source_run_id' in gate
    assert "== 'standard'" in gate
    assert "needs.detect-changes.outputs.docs_only != 'true'" in gate


def test_rerun_all_uses_new_plan_generation_and_ignores_removed_shards(tmp_path):
    # Re-run all jobs creates a fresh plan even when the commit is unchanged.
    old = planner.make_plan([runner(i) for i in range(6)], ['rtsp'], 'sha', plan_id='run-1')
    new = planner.make_plan([runner(0)], ['rtsp'], 'sha', plan_id='run-2')
    assert len(old['shards']) == 6 and len(new['shards']) == 1
    assert old['plan_id'] != new['plan_id']
    jobs = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())['jobs']
    upload = next(s for s in jobs['test-devkit-shard']['steps']
                  if s.get('name') == 'Upload selected coverage and outcomes')['with']
    download = next(s for s in jobs['test-devkit']['steps']
                    if 'pattern' in s.get('with', {}))['with']
    assert '${{ matrix.plan_id }}' in upload['name']
    assert download['pattern'] == 'core-selected-coverage-${{ needs.plan-devkit-tests.outputs.plan_id }}-*'
    # A failed-job-only rerun keeps its original matrix plan ID and stable name.
    assert 'github.run_attempt' not in upload['name'] and upload['overwrite'] is True
    plan = results(tmp_path)
    p = tmp_path / 'devkit-1/shard-receipt.json'
    data = json.loads(p.read_text()); data['plan_id'] = 'stale-generation'
    p.write_text(json.dumps(data))
    with pytest.raises(ValueError): planner.verify_results(plan, tmp_path)


def test_planner_exports_one_generation_to_manifest_matrix_and_gate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['planner', '--group', 'devkits', '--labels', '["rtsp"]'])
    for key, value in {'RUNNER_DISCOVERY_TOKEN': 'fixture', 'GITHUB_REPOSITORY': 'sima-neat/core',
                       'GITHUB_SHA': 'sha', 'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2',
                       'GITHUB_OUTPUT': str(tmp_path / 'outputs'),
                       'GITHUB_STEP_SUMMARY': str(tmp_path / 'summary')}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(planner, 'discover_runners', lambda *a: [runner(1)])
    planner.main()
    plan = json.loads((tmp_path / 'devkit-shard-plan.json').read_text())
    output = dict(line.split('=', 1) for line in (tmp_path / 'outputs').read_text().splitlines())
    matrix = json.loads(output['matrix'])
    assert output['plan_id'] == plan['plan_id'] == matrix['include'][0]['plan_id'] == '123-2'
    assert json.loads(matrix['include'][0]['tasks']) == plan['shards'][0]['tasks']
