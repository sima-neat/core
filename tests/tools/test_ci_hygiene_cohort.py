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


def load_preparation():
    import importlib.util
    import sys
    sys.path.insert(0, str(ROOT / 'tools/ci'))
    spec = importlib.util.spec_from_file_location('prepare_hygiene_cohort', ROOT / 'tools/ci/prepare_hygiene_cohort.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
    assert body.index('bash scripts/ci/verify_core_dist.sh') < body.index('python3 tools/ci/prepare_hygiene_cohort.py')
    # Matched target dependencies come from the verified same-run artifact.
    # The later installer supplies host analysis tools, not a second cohort.
    assert body.index('python3 tools/ci/prepare_hygiene_cohort.py') < body.index('./' + HYGIENE_INSTALLER)
    assert body.index('./' + HYGIENE_INSTALLER) < body.index('./scripts/ci/run_code_hygiene.sh')
    assert 'test_ci_hygiene_cohort.py' in job('resolve-compiler-cache')
    assert 'materialize_dependencies.py' not in body
    assert './build.sh' not in body
    helper = (ROOT / 'tools/ci/prepare_hygiene_cohort.py').read_text()
    assert 'tools/build_dependencies.py' in helper
    assert "['prepare', '--headers-only'" in helper
    assert 'dependencies.prepare(' not in helper and 'dependencies.find_config(' not in helper


@pytest.mark.parametrize('profile', ['ready', 'broken-headers-root', 'missing-tool-unprivileged', 'missing-native-module-root', 'apt-failure'])
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
         + f'exit {1 if profile == "missing-native-module-root" else 0}\n')
    tool('apt-get', 'exit 47\n' if profile == 'apt-failure' else 'exit 0\n')
    result = subprocess.run(
        ['/bin/bash', str(ROOT / HYGIENE_INSTALLER)],
        env={'PATH': str(tools), 'TEST_CALLS': str(calls), 'TEST_HEADERS': str(headers),
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
    elif profile == 'apt-failure':
        assert result.returncode == 47
        assert observed == [module_probe, clang_probe, 'id -u', 'apt-get update']
    else:
        assert result.returncode == 1
        assert observed == ['id -u', 'sudo -n true']
        assert 'cannot install without passwordless sudo' in result.stderr
        assert not headers.exists()
    assert all(':arm64' not in row and 'dpkg' not in row for row in observed)
    if profile not in ('missing-tool-unprivileged', 'missing-native-module-root'):
        assert headers.read_text() == '#include <memory>\n#include <nlohmann/json.hpp>\n'


def installed_dispatch_fixture(tmp_path):
    """Receipt-bound CLI transport double, not a pretend Scheduler validator.

    Producer's real installed coordinator/DEB integration is separately exercised
    by Scheduler client_cohort_ingestion_test.py and the actual hygiene command.
    This test protects Core's actual exec delegate and its own receipt gate.
    """
    import hashlib
    import json
    import shutil
    core = tmp_path / 'core'
    (core / 'tools').mkdir(parents=True)
    shutil.copy2(ROOT / 'tools/build_dependencies.py', core / 'tools/build_dependencies.py')
    package = tmp_path / 'scheduler'
    tool = package / 'usr/share/SimaScheduler/tools/client_dependencies.py'
    tool.parent.mkdir(parents=True)
    tool.write_text('import sys\nprint("CLI:" + " ".join(sys.argv[1:]))\n')
    info = {'component': 'scheduler', 'installed_files': {
        'share/SimaScheduler/tools/client_dependencies.py': {
            'kind': 'file', 'sha256': hashlib.sha256(tool.read_bytes()).hexdigest(),
            'mode': tool.stat().st_mode & 0o7777}}}
    info['package_sha256'] = hashlib.sha256(json.dumps(info, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    (tool.parent.parent / 'buildinfo.json').write_text(json.dumps(info))
    return core, package, tool


def test_hygiene_calls_real_core_exec_delegate(tmp_path):
    core, package, _ = installed_dispatch_fixture(tmp_path)
    output = load_preparation().coordinator(core, ['prepare', '--headers-only', '--output', str(tmp_path / 'audit')],
                                           {**os.environ, 'NEAT_SCHEDULER_PACKAGE_DIR': str(package)})
    assert output == f'CLI:prepare --headers-only --output {tmp_path / "audit"} --core {core}\n'


def test_real_delegate_rejects_changed_installed_coordinator(tmp_path):
    core, package, tool = installed_dispatch_fixture(tmp_path)
    tool.write_text(tool.read_text() + '# tampered\n')
    with pytest.raises(subprocess.CalledProcessError):
        load_preparation().coordinator(core, ['select', '--require'],
                                       {**os.environ, 'NEAT_SCHEDULER_PACKAGE_DIR': str(package)})


@pytest.mark.parametrize('text', ['import os', 'export OTHER=x', 'export NEAT_X=one\nexport NEAT_X=two',
                                 'export NEAT_X="line\nline"'])
def test_invalid_coordinator_output_is_not_executed(text):
    with pytest.raises(ValueError):
        load_preparation().shell_exports(text)


def test_hygiene_has_no_second_mutable_llima_resolution():
    helper = (ROOT / 'tools/ci/prepare_hygiene_cohort.py').read_text()
    assert "info.get('dependencies', {}).get('llima')" in helper
    assert "llima['ref'], llima['spec']" in helper
    assert 'develop:latest' not in helper and "'latest'" not in helper
    assert "selected['CMAKE_PREFIX_PATH']" in helper
    assert "selected['NEAT_SCHEDULER_PACKAGE_DIR']" in helper


def test_full_core_build_explicitly_requires_same_producer_test_component():
    driver = (ROOT / 'build.sh').read_text()
    assert '[[ "${BUILD_TESTS}" == "ON" ]] && dependency_args+=(--client-test-support)' in driver
    helper = (ROOT / 'tools/ci/prepare_hygiene_cohort.py').read_text()
    assert "['prepare', '--headers-only', '--client-test-support'" in helper
    tidy = (ROOT / 'scripts/run_cpp_tidy.sh').read_text()
    assert '--headers-only --client-test-support' in tidy
    assert '-C "$build_dir/dependencies/dependencies.cmake"' in tidy
    assert './build.sh --install-deps-only' not in tidy
    assert '/opt/toolchain' not in tidy


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
