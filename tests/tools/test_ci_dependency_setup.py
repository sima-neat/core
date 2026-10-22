"""Execute the real workflow's setup shell: nonzero helpers must stop the build."""
import os
from pathlib import Path
import subprocess
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()


@pytest.mark.parametrize('fail', ['acquire', 'sdk'])
def test_setup_failure_never_reaches_build(tmp_path, fail):
    start = WORKFLOW.index('            dependency_root=')
    stop = WORKFLOW.index('            ./build.sh ${CORE_BUILD_ARGS}', start)
    script = 'set -euo pipefail\n' + textwrap.dedent(WORKFLOW[start:stop]) + '\nprintf reached > "$MARKER"\n'
    tools = tmp_path / 'bin'; tools.mkdir()
    for name in ('python3', 'bash'):
        path = tools / name
        body = 'exit 47' if (name == 'python3' and fail == 'acquire') or (name == 'bash' and fail == 'sdk') else 'printf "export NEAT_SCHEDULER_PACKAGE_DIR=/fixture\\n"'
        path.write_text('#!/bin/sh\n' + body + '\n'); path.chmod(0o755)
    result = subprocess.run(['/bin/bash', '-c', script], text=True, capture_output=True,
        env={**os.environ, 'PATH': str(tools), 'GITHUB_SHA': 'a' * 40, 'GITHUB_HEAD_REF': 'topic',
             'GITHUB_REF_NAME': 'topic', 'NEAT_VULCAN_ENV': 'production', 'MARKER': str(tmp_path / 'marker')})
    assert result.returncode == 47, result.stderr
    assert not (tmp_path / 'marker').exists()


def test_no_absent_manifest_or_fake_sysroot_control():
    assert 'ci/dependencies.json' not in WORKFLOW
    assert 'NEAT_SYNC_SYSROOT' not in WORKFLOW
    assert 'source /opt/bin/simaai-init-build-env modalix' in WORKFLOW
    assert 'Expected exactly one running SDK container' in WORKFLOW
    assert 'docker image inspect' in WORKFLOW


def test_native_tooling_prerequisites_precede_compilation():
    assert WORKFLOW.index('Install native build-tooling prerequisites') < WORKFLOW.index('Run build-tooling tests')
    assert 'build-essential file nlohmann-json3-dev python3-venv' in WORKFLOW


@pytest.mark.parametrize('missing_archive', [False, True])
def test_sdk_shell_retains_only_actual_official_cache_exports(tmp_path, missing_archive):
    """Exercise the actual shell below its SDK/coordinator command boundaries."""
    import json
    import shlex
    import sys
    script = (ROOT / 'scripts/ci/prepare_scheduler_sdk.sh').read_text()
    assert 'SYSROOT_UPDATE_DOWNLOAD_DIR' not in script
    assert '--platform-package-dir /tmp/modalix' in script
    cache = tmp_path / 'official-cache'; cache.mkdir()
    provider = cache / 'consumed.deb'; provider.write_bytes(b'exact provider')
    (cache / 'unrelated.deb').write_bytes(b'unrelated SDK package')
    tools = tmp_path / 'tools'; tools.mkdir()
    (tools / 'build_dependencies.py').write_text('''import json, pathlib, shlex, sys
command = sys.argv[1]
if command == 'sdk-policy': print(json.dumps({'sysroot-version':'3.0.0~git202610140145.e5ece50-2121'}))
elif command == 'verify-sdk': print(json.dumps({'status':'verified'}))
else:
    root = pathlib.Path(sys.argv[sys.argv.index('--platform-package-dir')+1])
    p = root / 'consumed.deb'
    if not p.is_file(): raise SystemExit(31)
    print('export NEAT_RUNNER_PLATFORM_RUNTIME_DEBS=' + shlex.quote(json.dumps([str(p)])))
''')
    official = tmp_path / 'official.sh'; official.write_text('#!/bin/sh\nexit 0\n'); official.chmod(0o755)
    env = tmp_path / 'env.sh'; env.write_text('export CXX=cross-g++-14\n')
    script = script.replace('/usr/local/bin/setup-sdk-sysroot.sh', str(official))
    script = script.replace('/opt/bin/simaai-init-build-env', str(env))
    script = script.replace('/tmp/modalix', str(cache))
    # Never grant fixture privileges; source production root/sudo branches are unchanged.
    script = script.replace('[[ $(id -u) == 0 ]]', '[[ 0 == 0 ]]')
    if missing_archive: provider.unlink()
    retained = tmp_path / 'retained'
    result = subprocess.run(['/bin/bash', '-c', script, 'sdk-test', str(retained)], cwd=tmp_path,
                            text=True, capture_output=True)
    if missing_archive:
        assert result.returncode == 31, result.stderr
        assert not list(retained.glob('*.deb'))
    else:
        assert result.returncode == 0, result.stderr
        assert [p.name for p in retained.glob('*.deb')] == ['consumed.deb']
        assert (retained / 'consumed.deb').read_bytes() == provider.read_bytes()
        assert str(retained / 'consumed.deb') in result.stdout
        assert str(cache) not in result.stdout
