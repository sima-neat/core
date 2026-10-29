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


def runner(i, *, busy=False, status='online', labels=('firmware', 'regular-only', 'modalix-som-16gig')):
    return {'name': f'devkit-{i}', 'busy': busy, 'status': status,
            'labels': [{'name': x} for x in labels]}


def plan(runners=None, **kwargs):
    return planner.make_plan(runners or [runner(i) for i in range(9)], ['firmware'], 'sha', **kwargs)


@pytest.mark.parametrize('count', range(1, 10))
def test_adapts_to_capacity_and_preserves_every_lane(count):
    p = plan([runner(i) for i in range(count)])
    assert len(p['shards']) <= 7
    assert sorted(t for s in p['shards'] for t in s['tasks']) == sorted(planner.LANES)
    assert all('rtsp' not in s['labels'] for s in p['shards'])
    assert len({s['candidate'] for s in p['shards']}) == min(count, 7)
    planner.validate_plan(p)


def test_standard_lanes_and_memory_admission():
    p = plan([runner(0, labels=('firmware', 'regular-only'))] + [runner(i) for i in range(1, 8)])
    for s in p['shards']:
        if any(t.startswith('genai') for t in s['tasks']):
            assert s['pool'] == 'llm-16gb' and 'modalix-som-16gig' in s['labels']
        assert 'rtsp' not in s['labels']
    with pytest.raises(ValueError, match='llm-16gb'):
        plan([runner(0, labels=('firmware', 'regular-only'))])


@pytest.mark.parametrize('profile', ('http', 'all', 'none'))
def test_weekly_external_sources_only_use_rtsp_pool(profile):
    runners = [runner(i) for i in range(9)] + [runner('rtsp', labels=('firmware', 'rtsp'))]
    p = plan(runners, weekly=True, stream_profile=profile)
    stream = next(s for s in p['shards'] if 'streams' in s['tasks'])
    assert stream['pool'] == ('default' if profile == 'none' else 'rtsp')
    for s in p['shards']:
        if s['pool'] == 'rtsp': assert s['tasks'] == ['streams']
    assert len(p['shards']) <= 8
    if profile != 'none':
        with pytest.raises(ValueError, match='rtsp'):
            plan(weekly=True, stream_profile=profile)


def test_busy_fallback_exact_firmware_and_no_online_runner():
    p = plan([runner(1, busy=True), runner(2, status='offline'),
              runner(3, labels=('wrong-firmware', 'regular-only', 'modalix-som-16gig'))])
    assert all(not s['idle_at_planning'] for s in p['shards'])
    assert {s['candidate'] for s in p['shards']} == {'devkit-1'}
    with pytest.raises(ValueError, match='No safe online'):
        plan([runner(1, status='offline')])
    with pytest.raises(ValueError, match='Base runner'):
        planner.make_plan([runner(1)], ['rtsp'], 'sha')


@pytest.mark.parametrize('status', ('online', 'offline'))
def test_regular_selector_cannot_match_any_rtsp_registration(status):
    # Dropping rtsp from required labels is insufficient: the scheduler could
    # still match a RTSP runner carrying a superset of all candidate labels.
    blocked = runner('rtsp', status=status,
                     labels=('firmware', 'regular-only', 'modalix-som-16gig', 'rtsp'))
    with pytest.raises(ValueError, match='No safe online'):
        plan([runner(1), blocked])
    safe = runner(2, labels=('firmware', 'non-rtsp-pool', 'modalix-som-16gig'))
    assert all(s['candidate'] == safe['name'] for s in plan([runner(1), blocked, safe])['shards'])


def test_unready_runner_exclusion_applies_to_full_selector():
    bad = runner('unready')
    with pytest.raises(ValueError, match='No safe online'):
        plan([runner(1), bad], excluded_names=[bad['name']])


@pytest.mark.parametrize('damage', ('missing', 'duplicate', 'wrong-pool', 'missing-memory', 'rtsp-leak'))
def test_invalid_plan_is_rejected(damage):
    p = plan()
    if damage == 'missing': p['shards'][0]['tasks'].pop()
    if damage == 'duplicate': p['shards'][0]['tasks'].append('genai')
    if damage == 'wrong-pool': p['shards'][0]['pool'] = 'default'
    if damage == 'missing-memory': p['shards'][0]['labels'].remove('modalix-som-16gig')
    if damage == 'rtsp-leak': p['shards'][0]['labels'].append('rtsp')
    with pytest.raises(ValueError): planner.validate_plan(p)


def results(tmp_path):
    p = plan()
    for s in p['shards']:
        d = tmp_path / s['id']; d.mkdir()
        (d / 'shard-receipt.json').write_text(json.dumps(
            {'id': s['id'], 'plan_id': p['plan_id'], 'sha': 'sha', 'tasks': s['tasks'], 'result': 'success'}))
    for i in range(2):
        (tmp_path / p['shards'][i]['id'] / 'e2e-partition.json').write_text(json.dumps(
            {'index': i, 'canonical': ['a', 'b'], 'selected': [['a'], ['b']][i]}))
    return p


@pytest.mark.parametrize('damage', (None, 'missing', 'failed', 'sha', 'tasks', 'overlap', 'inventory', 'generation'))
def test_aggregate_requires_exact_results_and_partition_union(tmp_path, damage):
    p = results(tmp_path)
    path = next(tmp_path.rglob('shard-receipt.json'))
    if damage == 'missing': path.unlink()
    elif damage in ('failed', 'sha', 'tasks', 'generation'):
        data = json.loads(path.read_text())
        data[{'failed': 'result', 'sha': 'sha', 'tasks': 'tasks', 'generation': 'plan_id'}[damage]] = (
            [] if damage == 'tasks' else 'bad')
        path.write_text(json.dumps(data))
    elif damage in ('overlap', 'inventory'):
        path = next(tmp_path.rglob('e2e-partition.json'))
        data = json.loads(path.read_text())
        data['selected' if damage == 'overlap' else 'canonical'] = ['a', 'b', 'extra']
        path.write_text(json.dumps(data))
    if damage:
        with pytest.raises(ValueError): planner.verify_results(p, tmp_path)
    else: planner.verify_results(p, tmp_path)


def test_weighted_partition_is_deterministic_and_complete():
    names = ['a', 'b', 'c', 'd']
    weights = {'a': 100, 'b': 90, 'c': 10, 'd': 20}
    bins = partitioner.partition(names, weights)
    assert bins == partitioner.partition(list(reversed(names)), weights)
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


def test_workflow_guards_and_reruns():
    jobs = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())['jobs']
    shard = jobs['test-devkit-shard']
    assert shard['strategy']['fail-fast'] is False
    assert 'rtsp' not in json.loads(jobs['resolve-test-runner']['with']['runner-labels'])
    assert jobs['test-devkit']['name'] == 'Test Core on DevKit'
    assert '!inputs.offline_source_run_id' in jobs['test-devkit']['if']
    for gate in ('publish-offline-develop', 'validation-complete'):
        assert {'plan-devkit-tests', 'test-devkit-shard', 'test-devkit'} <= set(jobs[gate]['needs'])
    assert 'test-devkit' in jobs['promote-latest-tag']['needs']
    steps = {s.get('name'): s for s in shard['steps']}
    assert "'streams'" in steps['Run declared weekly stream capability profile']['if']
    assert "'genai-python'" in steps['Run Python GenAI E2E tests from installed extras']['if']
    assert "'genai-python'" in steps['Prepare GenAI models']['if']
    assert "'python-tutorials'" not in steps['Prepare GenAI models']['if']
    # Every E2E worker must discover the same canonical selection.
    e2e = steps['Run E2E Tests'], steps['Run E2E Tests (partition 2)']
    assert e2e[0]['env'] == e2e[1]['env']
    shard_0 = e2e[0]['run'].replace('--index 0', '--index 1')
    assert shard_0.replace('e2e-0', 'e2e-1').replace('e2e-tail-0', 'e2e-tail-1') == e2e[1]['run']
    upload = steps['Upload selected coverage and outcomes']['with']
    assert upload['overwrite'] is True and 'github.run_attempt' not in upload['name']
    assert '${{ matrix.plan_id }}' in upload['name']
    download = next(s for s in jobs['test-devkit']['steps'] if 'pattern' in s.get('with', {}))['with']
    assert download['pattern'] == 'core-selected-coverage-${{ needs.plan-devkit-tests.outputs.plan_id }}-*'


def test_group_visibility_and_pagination():
    def get(path):
        if '/repositories?' in path: return {'repositories': [{'full_name': 'sima-neat/core'}]}
        if '/runners?' in path: return {'runners': [runner(1)]}
        return {'runner_groups': [{'name': 'devkits', 'id': 7, 'visibility': 'selected'}]}
    assert planner.discover_runners(get, 'sima-neat/core', 'devkits', True)[0]['name'] == 'devkit-1'
    with pytest.raises(ValueError, match='no access'):
        planner.discover_runners(get, 'sima-neat/other', 'devkits', True)
    with pytest.raises(ValueError, match='public'):
        planner.discover_runners(get, 'sima-neat/core', 'devkits', False)
    def pages(path):
        return {'runners': [runner(i) for i in range(100)] if path.endswith('&page=1') else [runner(100)]}
    assert len(planner.collect(pages, '/runners', 'runners')) == 101


def test_planner_exports_generation_and_capabilities(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['planner', '--group', 'devkits', '--labels', '["firmware"]'])
    for key, value in {'RUNNER_DISCOVERY_TOKEN': 'fixture', 'GITHUB_REPOSITORY': 'sima-neat/core',
                       'GITHUB_SHA': 'sha', 'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2',
                       'RUN_LONG_TESTS': 'false', 'GITHUB_OUTPUT': str(tmp_path / 'outputs'),
                       'GITHUB_STEP_SUMMARY': str(tmp_path / 'summary')}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(planner, 'discover_runners', lambda *a: [runner(i) for i in range(9)])
    planner.main()
    p = json.loads((tmp_path / 'devkit-shard-plan.json').read_text())
    output = dict(line.split('=', 1) for line in (tmp_path / 'outputs').read_text().splitlines())
    matrix = json.loads(output['matrix'])
    assert output['plan_id'] == p['plan_id'] == matrix['include'][0]['plan_id'] == '123-2'
    assert json.loads(matrix['include'][0]['tasks']) == p['shards'][0]['tasks']
    old = plan(plan_id='123-1')
    new = plan([runner(0)], plan_id='123-2')
    assert len(old['shards']) > len(new['shards']) and old['plan_id'] != new['plan_id']


def test_fresh_genai_worker_creates_model_root_before_lock(tmp_path):
    jobs = yaml.safe_load((ROOT / '.github/workflows/vulcan-ci.yml').read_text())['jobs']
    body = next(s['run'] for s in jobs['test-devkit-shard']['steps'] if s.get('name') == 'Prepare GenAI models')
    prefix = body[:body.index('flock ')]
    work = tmp_path / '_work'; work.mkdir()
    extras = tmp_path / 'extras'; helper = extras / 'share/sima-neat/tests/tools/prepare_genai_models.sh'
    helper.parent.mkdir(parents=True); helper.write_text('#!/bin/sh\nexit 0\n'); helper.chmod(0o755)
    (work / 'extras_prefix.txt').write_text(str(extras))
    binary = tmp_path / 'bin'; binary.mkdir()
    (binary / 'mountpoint').write_text('#!/bin/sh\nexit 0\n'); (binary / 'mountpoint').chmod(0o755)
    model_root = tmp_path / 'fresh/models'
    env = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ['PATH'], LLIMA_MODELS_PATH=str(model_root))
    subprocess.run(['bash', '-c', prefix + '\ntouch "$prepare_models_lock"\n'], env=env, cwd=tmp_path, check=True)
    assert (model_root / '.prepare-models.lock').is_file()


def test_default_selectors_do_not_inherit_memory_or_appliance_labels():
    runners = [runner(i, labels=('firmware', 'model-registry', 'modalix-som-16gig', 'appliance-specific'))
               for i in range(9)]
    runners.append(runner('rtsp', labels=('firmware', 'rtsp', 'modalix-som-16gig')))
    p = plan(runners)
    for shard in p['shards']:
        assert 'appliance-specific' not in shard['labels']
        assert 'model-registry' in shard['labels']
        assert ('modalix-som-16gig' in shard['labels']) == (shard['pool'] == 'llm-16gb')
        assert not set(shard['labels']) <= planner.label_set(runners[-1])
