"""Host-only workflow ordering/transport fixtures; never package qualification."""
import os
from pathlib import Path
import re
import subprocess
import textwrap
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / '.github/workflows/vulcan-ci.yml'
HYGIENE_INSTALLER = 'scripts/ci/install_code_hygiene_deps.sh'


def job(name):
    match = re.search(rf'(?ms)^  {re.escape(name)}:\n(.*?)(?=^  [\w-]+:\n|\Z)',
                      WORKFLOW.read_text())
    assert match, name
    return match[1]


def needs(name):
    block = re.search(r'(?m)^    needs:\n((?:      - [\w-]+\n)+)', job(name))
    return re.findall(r'- ([\w-]+)', block[1]) if block else []


def transport():
    body = job('code-hygiene')
    block = body.split("          python3 - <<'PY'\n", 1)[1].split('\n          PY', 1)[0]
    return textwrap.dedent(block)


def test_dependency_dag_and_publication_gate():
    names = re.findall(r'(?m)^  ([\w-]+):$', WORKFLOW.read_text())
    def visit(name, active):
        assert name not in active, f'Cycle: {active} -> {name}'
        for dependency in needs(name):
            visit(dependency, active + [name])
    for name in names:
        visit(name, [])
    assert 'build' in needs('code-hygiene')
    assert 'code-hygiene' not in needs('build')
    assert 'code-hygiene' in needs('publish-vulcan-artifacts')
    gate = job('publish-vulcan-artifacts')
    assert "needs.build.result == 'success'" in gate
    assert "== 'fuzz' || needs.code-hygiene.result == 'success'" in gate
    assert 'publish-vulcan-artifacts' in needs('test-devkit')


def test_same_run_artifact_and_validator_precede_analysis():
    body = job('code-hygiene')
    assert 'uses: actions/download-artifact@v5' in body
    assert 'name: neat-core-dist' in body
    assert 'run-id:' not in body and 'repository:' not in body
    assert body.index('bash scripts/ci/verify_core_dist.sh') < body.index("python3 - <<'PY'")
    # Matched target dependencies come from the verified same-run artifact.
    # The later installer supplies host analysis tools, not a second cohort.
    assert body.index("python3 - <<'PY'") < body.index('./' + HYGIENE_INSTALLER)
    assert body.index('./' + HYGIENE_INSTALLER) < body.index('./scripts/ci/run_code_hygiene.sh')
    assert 'test_ci_hygiene_cohort.py' in job('resolve-compiler-cache')
    assert 'materialize_dependencies.py' not in body
    assert './build.sh' not in body
    compile(transport(), str(WORKFLOW), 'exec')


@pytest.mark.parametrize('profile', ['ready', 'broken-headers-root', 'missing-tool-unprivileged'])
def test_hygiene_installer_only_prepares_host_tools(tmp_path, profile):
    """Exercise the real installer with shell doubles; never install packages."""
    tools = tmp_path / 'bin'
    tools.mkdir()
    calls = tmp_path / 'calls'
    headers = tmp_path / 'headers'

    def tool(name, body='exit 0\n'):
        path = tools / name
        path.write_text('#!/bin/sh\n'
                        'printf "%s %s\\n" "${0##*/}" "$*" >> "$TEST_CALLS"\n' + body)
        path.chmod(0o755)

    for name in ('clang-format', 'clang-tidy', 'run-clang-tidy', 'rg', 'g++'):
        if name != 'g++' or profile != 'missing-tool-unprivileged':
            tool(name)
    tool('clang++', 'while IFS= read -r line; do\n'
                   '  printf "%s\\n" "$line" >> "$TEST_HEADERS"\n'
                   'done\n'
                   f'exit {1 if profile == "broken-headers-root" else 0}\n')
    tool('id', f'echo {1000 if profile == "missing-tool-unprivileged" else 0}\n')
    tool('sudo', 'exit 1\n')
    tool('apt-get')
    result = subprocess.run(
        ['/bin/bash', str(ROOT / HYGIENE_INSTALLER)],
        env={'PATH': str(tools), 'TEST_CALLS': str(calls), 'TEST_HEADERS': str(headers)},
        text=True, capture_output=True, check=False,
    )
    observed = calls.read_text().splitlines()
    clang_probe = 'clang++ -std=c++20 -x c++ - -fsyntax-only'
    if profile == 'ready':
        assert result.returncode == 0, result.stderr
        assert observed == [clang_probe]
        assert 'already installed' in result.stdout
    elif profile == 'broken-headers-root':
        assert result.returncode == 0, result.stderr
        assert observed == [clang_probe, 'id -u', 'apt-get update',
                            'apt-get install -y build-essential gcc-12 g++-12 libstdc++-12-dev '
                            'clang clang-format clang-tidy ripgrep']
    else:
        assert result.returncode == 1
        assert observed == ['id -u', 'sudo -n true']
        assert 'cannot install without passwordless sudo' in result.stderr
        assert not headers.exists()
    if profile != 'missing-tool-unprivileged':
        assert headers.read_text() == '#include <memory>\n'


def execute_fixture(tmp_path, monkeypatch, roles, *, reject_receipt=False):
    """Tiny deliberately non-DEB fixtures exercise transport control flow only."""
    monkeypatch.chdir(tmp_path)
    tools = tmp_path / 'tools'
    tools.mkdir()
    (tools / 'build_dependencies.py').write_text('''
import os
def prepare(core, output, selected, jobs, headers_only):
    assert headers_only is True and jobs == 1
    assert set(selected) == {'kernels', 'runner', 'scheduler', 'runtime_compat'}
    assert all(v['kind'] == 'package' for v in selected.values())
    if os.environ.get('TEST_REJECT_RECEIPT') == '1':
        raise ValueError('test receipt/payload rejection')
    return {k: dict(package=v['path'], config=v['path']+'/usr/lib/cmake/'+k)
            for k,v in selected.items()}
def find_config(root, name):
    return root/'usr/lib/cmake'/name
''')
    dist = tmp_path / '_work/hygiene-dist'
    dist.mkdir(parents=True)
    for index, role in enumerate(roles):
        (dist / f'fixture-{index}.deb').write_text(role)
    output = tmp_path / 'github.env'
    monkeypatch.setenv('GITHUB_ENV', str(output))
    monkeypatch.delenv('NEAT_CODECS_PACKAGE_DIR', raising=False)
    monkeypatch.setenv('TEST_REJECT_RECEIPT', '1' if reject_receipt else '0')
    def control(argv, **kwargs):
        assert argv[:2] == ['dpkg-deb', '-f'] and argv[-1] == 'Package'
        return Path(argv[2]).read_text()
    def extract(argv, **kwargs):
        assert argv[:2] == ['dpkg-deb', '-x'] and kwargs['check'] is True
        (Path(argv[-1]) / 'usr/include').mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(argv, 0)
    with mock.patch('subprocess.check_output', side_effect=control), \
         mock.patch('subprocess.run', side_effect=extract):
        exec(compile(transport(), str(WORKFLOW), 'exec'), {})
    return dict(line.split('=', 1) for line in output.read_text().splitlines())


def test_transport_reuses_installed_contracts_and_selected_headers(tmp_path, monkeypatch):
    env = execute_fixture(tmp_path, monkeypatch, ['simaai-kernels-dev', 'simaai-runner', 'simaai-scheduler', 'simaai-runtime',
                                                 'neat-codecs', 'neat-codecs-dev', 'sima-neat'])
    assert set(env) == {'NEAT_KERNELS_PACKAGE_DIR', 'NEAT_RUNNER_PACKAGE_DIR',
                        'NEAT_SCHEDULER_PACKAGE_DIR', 'NEAT_RUNTIME_COMPAT_PACKAGE_DIR',
                        'NEAT_CODECS_PACKAGE_DIR', 'CMAKE_PREFIX_PATH', 'CMAKE_INCLUDE_PATH'}
    assert env['CMAKE_INCLUDE_PATH'] == env['NEAT_RUNNER_PACKAGE_DIR'] + '/usr/include/simaai'
    assert all(env[k] in env['CMAKE_PREFIX_PATH'] for k in
               ('NEAT_KERNELS_PACKAGE_DIR', 'NEAT_RUNNER_PACKAGE_DIR', 'NEAT_CODECS_PACKAGE_DIR'))
    assert not (tmp_path / '_work/hygiene-dependencies/core').exists()


def test_no_external_codec_provider_is_not_fabricated(tmp_path, monkeypatch):
    env = execute_fixture(tmp_path, monkeypatch, ['simaai-kernels-dev', 'simaai-runner', 'simaai-scheduler', 'simaai-runtime'])
    assert 'NEAT_CODECS_PACKAGE_DIR' not in env


@pytest.mark.parametrize('roles', [
    ['simaai-kernels-dev'],
    ['simaai-kernels-dev', 'simaai-runner', 'simaai-scheduler', 'simaai-runtime', 'simaai-runtime'],
    ['simaai-kernels-dev', 'simaai-runner', 'simaai-scheduler', 'simaai-runtime', 'neat-codecs'],
])
def test_incomplete_or_duplicate_transport_fails_without_env(tmp_path, monkeypatch, roles):
    with pytest.raises(ValueError):
        execute_fixture(tmp_path, monkeypatch, roles)
    assert not (tmp_path / 'github.env').exists()


def test_receipt_rejection_is_not_suppressed(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='receipt/payload rejection'):
        execute_fixture(tmp_path, monkeypatch, ['simaai-kernels-dev', 'simaai-runner', 'simaai-scheduler', 'simaai-runtime'],
                        reject_receipt=True)
    assert not (tmp_path / 'github.env').exists()


GATE_IDS = {
    'unit_model_stage_fragments_test',
    'unit_kernel_runner_integration_test',
    'unit_plugin_failure_messages_test',
}


def gate_block():
    body = job('test-devkit')
    return body.split('      - name: Typed Contract Admission Gate\n', 1)[1].split(
        '      - name: Run Tutorial Tests\n', 1)[0]


def test_typed_admission_gate_maps_current_semantics_and_registrations():
    block = gate_block()
    regex = re.search(r"GATE_REGEX='([^']+)'", block)[1]
    registered = set(re.findall(r'^  (unit_[a-z0-9_]+_test)$',
                               (ROOT / 'tests/CMakeLists.txt').read_text(), re.M))
    # The kernel-runner test uses the shared guarded registration directly.
    registered.update(re.findall(r'simaneat_add_test\(NAME\s+(unit_[a-z0-9_]+_test)\s+COMMAND',
                                 (ROOT / 'tests/CMakeLists.txt').read_text()))
    assert {name for name in registered if re.fullmatch(regex, name)} == GATE_IDS
    assert 'SIMA_STRICT_MANIFEST_NO_JSON_FALLBACK' not in WORKFLOW.read_text()
    assert 'unit_sima_plugin_manifest_strict_' not in block
    assert 'scripts/ci/run_zero_skip_gate.sh' in block
    assert 'selection.json' in block
    for name in GATE_IDS:
        assert (ROOT / 'tests/unit_testing' / (name + '.cpp')).is_file()


@pytest.mark.parametrize('mutation', ['complete', 'missing', 'extra', 'duplicate'])
def test_typed_gate_requires_exact_fresh_core_id_inventory(tmp_path, monkeypatch, mutation):
    import json
    block = gate_block()
    script = textwrap.dedent(block.split("          python3 - <<'PY_GATE'\n", 1)[1].split(
        '\n          PY_GATE', 1)[0])
    names = sorted(GATE_IDS)
    if mutation == 'missing': names.pop()
    if mutation == 'extra': names.append('unrelated_test')
    if mutation == 'duplicate': names[-1] = names[0]
    path = tmp_path / '_work/coverage/manifest/selection.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'tests': [{'name': name} for name in names]}))
    monkeypatch.chdir(tmp_path)
    if mutation == 'complete':
        exec(compile(script, str(WORKFLOW), 'exec'), {})
    else:
        with pytest.raises(SystemExit, match='all three Core-owned'):
            exec(compile(script, str(WORKFLOW), 'exec'), {})


def test_plugin_only_mla_admission_keeps_scheduler_ownership():
    block = gate_block()
    assert 'unit_processmla_requires_explicit_output_contract_test' not in block
    assert 'belongs to Scheduler' in block
    assert 'separately qualified dependency test cohort' in block
    assert not (ROOT / 'tests/unit_testing/unit_processmla_requires_explicit_output_contract_test.cpp').exists()
