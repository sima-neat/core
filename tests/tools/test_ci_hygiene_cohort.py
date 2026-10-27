"""Host-only workflow ordering/transport fixtures; never package qualification."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

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


def run_block(name):
    step = job('code-hygiene').split(f'      - name: {name}\n', 1)[1].split('\n      - ', 1)[0]
    return textwrap.dedent(step.split('        run: |\n', 1)[1])


def test_dependency_dag_and_publication_gate():
    names = re.findall(r'(?m)^  ([\w-]+):$', WORKFLOW.read_text())
    def visit(name, active):
        assert name not in active, f'Cycle: {active} -> {name}'
        for dependency in needs(name):
            visit(dependency, active + [name])
    for name in names:
        visit(name, [])
    assert 'build' not in needs('code-hygiene')
    assert 'code-hygiene' not in needs('build')
    assert 'code-hygiene' in needs('publish-vulcan-artifacts')
    gate = job('publish-vulcan-artifacts')
    assert "needs.build.result == 'success'" in gate
    assert "== 'fuzz' || needs.code-hygiene.result == 'success'" in gate
    assert 'publish-vulcan-artifacts' in needs('test-devkit')


def test_hygiene_uses_vulcan_native_ubuntu_arm64_runner():
    body = job('code-hygiene')
    assert 'environment: ${{ needs.resolve-vulcan-config.outputs.vulcan_env }}' in body
    assert ('runs-on: [self-hosted, linux, arm64, '
            '"vulcan-${{ needs.resolve-vulcan-config.outputs.vulcan_env }}", ubuntu, default]') in body
    assert 'scripts/ci/install_code_hygiene_deps.sh' in body
    assert needs('code-hygiene') == ['resolve-vulcan-config', 'detect-changes']
    assert 'vulcan/ubuntu-build:ubuntu24-arm64-media-' in body
    assert 'HYGIENE_REQUIRE_PREINSTALLED: "1"' in body


def test_build_tooling_runs_alongside_compilation_and_gates_publication():
    assert needs('build-tooling-tests') == ['resolve-vulcan-config']
    assert 'build-tooling-tests' not in needs('resolve-compiler-cache')
    assert 'build-tooling-tests' not in needs('build')
    assert 'Run build-tooling tests' not in job('resolve-compiler-cache')
    assert 'Install native build-tooling prerequisites' not in job('resolve-compiler-cache')
    assert '-m pytest -q --durations=25' in job('build-tooling-tests')
    for name in ('publish-vulcan-artifacts', 'publish-pciehost-package',
                 'publish-offline-develop', 'validation-complete'):
        assert 'build-tooling-tests' in needs(name)
    # PCIe metadata uses Actions' implicit success() dependency gate.
    assert 'always()' not in job('publish-pciehost-package')
    assert 'continue-on-error' not in job('build-tooling-tests')


@pytest.mark.parametrize('mode', ['standard', 'fuzz'])
@pytest.mark.parametrize('outcome', ['success', 'failure', 'cancelled', 'skipped'])
@pytest.mark.parametrize('validation_only', [False, True])
def test_core_publication_requires_tooling_success_in_every_build_mode(mode, outcome, validation_only):
    expression = re.search(r'(?m)^    if: \$\{\{ (.*?) \}\}$',
                           job('publish-vulcan-artifacts'))[1]
    values = {
        'always()': True, 'inputs.validation_only': validation_only,
        'needs.build.result': 'success', 'needs.build-tooling-tests.result': outcome,
        'inputs.build_mode': mode,
        'needs.code-hygiene.result': 'skipped' if mode == 'fuzz' else 'success',
    }
    for key, value in values.items():
        expression = expression.replace(key, repr(value))
    expression = expression.replace('&&', ' and ').replace('||', ' or ').replace('!', ' not ')
    assert bool(eval(expression, {'__builtins__': {}}, {})) is (
        outcome == 'success' and not validation_only)


@pytest.mark.parametrize('outcome', ['success', 'failure', 'cancelled', 'skipped'])
def test_final_validation_requires_tooling_success(monkeypatch, outcome):
    results = {name: {'result': 'success'} for name in needs('validation-complete')}
    for name in ('publish-vulcan-artifacts', 'publish-pciehost-package', 'promote-latest-tag'):
        results[name]['result'] = 'skipped'
    results['build-tooling-tests']['result'] = outcome
    monkeypatch.setenv('RESULTS', json.dumps(results))
    monkeypatch.setenv('BUILD_MODE', 'standard')
    script = textwrap.dedent(job('validation-complete').split(
        "          python3 - <<'PYGATE'\n", 1)[1].split('\n          PYGATE', 1)[0])
    if outcome == 'success':
        exec(compile(script, str(WORKFLOW), 'exec'), {})
    else:
        with pytest.raises(SystemExit, match='build-tooling-tests'):
            exec(compile(script, str(WORKFLOW), 'exec'), {})


@pytest.mark.parametrize(('profile', 'head_ref'), [
    ('ready', 'topic'), ('ready', ''), ('acquisition-failure', 'topic'),
    ('linefeed', 'topic'), ('carriage-return', 'topic'),
])
def test_hygiene_acquires_without_core_build_and_stops_on_failure(tmp_path, profile, head_ref):
    """Run the workflow shell; the acquisition boundary supplies private package paths."""
    acquisition = tmp_path / 'tools/ci/acquire_scheduler.py'
    acquisition.parent.mkdir(parents=True)
    acquisition.write_text(textwrap.dedent('''\
        import argparse, json, os, shlex, sys
        from pathlib import Path
        parser = argparse.ArgumentParser()
        parser.add_argument('--branch', required=True)
        parser.add_argument('--environment', required=True)
        parser.add_argument('--output', required=True)
        args = parser.parse_args()
        Path('acquisition.json').write_text(json.dumps(vars(args)))
        package = str(Path(args.output) / 'scheduler with spaces')
        profile = os.environ['TEST_PROFILE']
        if profile == 'linefeed': package += '\\ninvalid'
        if profile == 'carriage-return': package += '\\rinvalid'
        print('export NEAT_SCHEDULER_PACKAGE_DIR=' + shlex.quote(package))
        if profile == 'acquisition-failure': sys.exit(23)
        print('export NEAT_SCHEDULER_CLIENT_TEST_PACKAGE_DIR=' + shlex.quote(str(Path(args.output) / 'client-test')))
        print('export NEAT_RUNNER_PACKAGE_DIR=/not-a-public-analysis-input')
        '''))
    analysis = tmp_path / 'scripts/ci/run_code_hygiene.sh'
    analysis.parent.mkdir(parents=True)
    analysis.write_text(f'#!{sys.executable}\n' + textwrap.dedent('''\
        import json, os
        from pathlib import Path
        Path('analysis.json').write_text(json.dumps({key: os.environ[key] for key in
            ('NEAT_SCHEDULER_PACKAGE_DIR', 'NEAT_SCHEDULER_CLIENT_TEST_PACKAGE_DIR')}))
        '''))
    analysis.chmod(0o755)
    github_env = tmp_path / 'github-env'
    github_env.write_text('EXISTING=value\n')
    script = (run_block('Acquire published Scheduler for analysis') + '\n' +
              run_block('Run changed-code hygiene'))
    result = subprocess.run(['/bin/bash', '-c', script], cwd=tmp_path, text=True, capture_output=True,
                            env={**os.environ, 'PATH': str(Path(sys.executable).parent) + ':' + os.environ['PATH'],
                                 'TEST_PROFILE': profile, 'GITHUB_HEAD_REF': head_ref, 'GITHUB_REF_NAME': 'develop',
                                 'NEAT_VULCAN_ENV': 'production', 'RUNNER_TEMP': str(tmp_path),
                                 'GITHUB_ENV': str(github_env)})
    output = tmp_path / 'core-hygiene-dependencies'
    assert json.loads((tmp_path / 'acquisition.json').read_text()) == {
        'branch': head_ref or 'develop', 'environment': 'production', 'output': str(output)}
    if profile != 'ready':
        assert result.returncode != 0
        assert not (tmp_path / 'analysis.json').exists()
        assert github_env.read_text() == 'EXISTING=value\n'
        if profile != 'acquisition-failure':
            assert 'Invalid analysis package path' in result.stderr
        return
    assert result.returncode == 0, result.stderr
    expected = {'NEAT_SCHEDULER_PACKAGE_DIR': str(output / 'scheduler with spaces'),
                'NEAT_SCHEDULER_CLIENT_TEST_PACKAGE_DIR': str(output / 'client-test')}
    assert json.loads((tmp_path / 'analysis.json').read_text()) == expected
    assert github_env.read_text() == 'EXISTING=value\n' + ''.join(f'{key}={value}\n' for key, value in expected.items())


@pytest.mark.parametrize('profile', ['ready', 'broken-headers-root', 'missing-tool-unprivileged', 'missing-native-module-root', 'apt-failure', 'preinstalled-required'])
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

    for name in ('clang-format', 'clang-tidy', 'run-clang-tidy', 'rg', 'g++', 'cmake', 'python3', 'git'):
        if name != 'g++' or profile != 'missing-tool-unprivileged':
            tool(name)
    tool('clang++', 'while IFS= read -r line; do\n'
                   '  printf "%s\\n" "$line" >> "$TEST_HEADERS"\n'
                   'done\n'
                   f'exit {1 if profile in ("broken-headers-root", "apt-failure") else 0}\n')
    tool('id', f'echo {1000 if profile == "missing-tool-unprivileged" else 0}\n')
    tool('sudo', 'exit 1\n')
    tool('pkg-config', 'test -z "${SYSROOT}${SDKTARGETSYSROOT}${PKG_CONFIG_SYSROOT_DIR}${PKG_CONFIG_PATH}${PKG_CONFIG_LIBDIR}" || exit 90\n'
         + f'exit {1 if profile in ("missing-native-module-root", "preinstalled-required") else 0}\n')
    tool('apt-get', 'exit 47\n' if profile == 'apt-failure' else 'exit 0\n')
    result = subprocess.run(
        ['/bin/bash', str(ROOT / HYGIENE_INSTALLER)],
        env={'PATH': str(tools), 'TEST_CALLS': str(calls), 'TEST_HEADERS': str(headers),
             'HYGIENE_REQUIRE_PREINSTALLED': '1' if profile == 'preinstalled-required' else '0',
             'SYSROOT': '/poison-sdk', 'SDKTARGETSYSROOT': '/poison-sdk',
             'PKG_CONFIG_SYSROOT_DIR': '/poison-sdk', 'PKG_CONFIG_PATH': '/poison-sdk',
             'PKG_CONFIG_LIBDIR': '/poison-sdk'},
        text=True, capture_output=True, check=False,
    )
    observed = calls.read_text().splitlines()
    clang_probe = 'clang++ -std=c++20 -x c++ - -fsyntax-only'
    module_probe = 'pkg-config --exists gstreamer-1.0 >= 1.22 gstreamer-allocators-1.0 gstreamer-app-1.0 gstreamer-video-1.0 gstreamer-sdp-1.0 gstreamer-rtsp-server-1.0 glib-2.0 opencv4 cpp-httplib yaml-cpp zlib'
    if profile == 'ready':
        assert result.returncode == 0, result.stderr
        assert observed == [module_probe, clang_probe]
        assert 'already installed' in result.stdout
    elif profile in ('broken-headers-root', 'missing-native-module-root'):
        assert result.returncode == 0, result.stderr
        probe = [module_probe] + ([clang_probe] if profile == 'broken-headers-root' else [])
        assert observed == probe + ['id -u', 'apt-get update',
                            'apt-get install -y build-essential gcc-12 g++-12 libstdc++-12-dev '
                            'clang clang-format clang-tidy ripgrep cmake pkg-config python3 git '
                            'libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libgstrtspserver-1.0-dev '
                            'libopencv-dev nlohmann-json3-dev libcpp-httplib-dev libyaml-cpp-dev zlib1g-dev']
    elif profile == 'preinstalled-required':
        assert result.returncode == 1
        assert observed == [module_probe]
        assert 'rebuild the shared media image' in result.stderr
        assert not headers.exists()
    elif profile == 'apt-failure':
        assert result.returncode == 47
        assert observed == [module_probe, clang_probe, 'id -u', 'apt-get update']
    else:
        assert result.returncode == 1
        assert observed == ['id -u', 'sudo -n true']
        assert 'cannot install without passwordless sudo' in result.stderr
        assert not headers.exists()
    assert all(':arm64' not in row and 'dpkg' not in row for row in observed)
    if profile not in ('missing-tool-unprivileged', 'missing-native-module-root', 'preinstalled-required'):
        assert headers.read_text() == '#include <memory>\n#include <nlohmann/json.hpp>\n'


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
