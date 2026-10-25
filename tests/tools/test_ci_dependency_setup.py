"""Check the workflow's SDK build setup and the Scheduler SDK preparation shell."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()


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
    assert 'SYSROOT_UPDATE_DOWNLOAD_DIR=$package_dir/downloads' in script
    assert 'sysroot update' in script
    cache = tmp_path / 'official-cache'; cache.mkdir()
    provider = cache / 'consumed.deb'; provider.write_bytes(b'exact provider')
    (cache / 'unrelated.deb').write_bytes(b'unrelated SDK package')
    tools = tmp_path / 'tools'; tools.mkdir()
    (tools / 'build_dependencies.py').write_text('''import json, pathlib, shlex, sys
command = sys.argv[1]
if command == 'sdk-policy': print(json.dumps({'elxr_version':'3.0.0_master_B2372', 'sysroot-version':'3.0.0~pre2372'}))
elif command == 'verify-sdk': print(json.dumps({'status':'verified'}))
else:
    root = pathlib.Path(sys.argv[sys.argv.index('--platform-package-dir')+1])
    p = root / 'consumed.deb'
    if not p.is_file(): raise SystemExit(31)
    print('export NEAT_RUNNER_PLATFORM_RUNTIME_DEBS=' + shlex.quote(json.dumps([str(p)])))
''')
    official = tmp_path / 'official.sh'; official.write_text('#!/bin/sh\nexit 0\n'); official.chmod(0o755)
    env = tmp_path / 'env.sh'; env.write_text('export CXX=cross-g++-14\n')
    script = script.replace('sysroot update', str(official) + ' update').replace('sysroot status', str(official) + ' status')
    script = script.replace('/opt/bin/simaai-init-build-env', str(env))
    script = script.replace('$package_dir/downloads', str(cache))
    (tmp_path / 'deps').mkdir()
    (tmp_path / 'deps/manifest.json').write_text(json.dumps({'elxr_version': '3.0.0_master_B2372'}))
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


@pytest.mark.parametrize('firmware,version', [
    ('3.0.0_develop_B2121', '3.0.0~git202610140145.e5ece50-2121'),
    ('3.0.0_master_B2372', '3.0.0~pre2371'),
    ('3.0.0_master_B2372', '3.0.0~git202610140145.e5ece50-2372'),
])
def test_scheduler_policy_mismatch_fails_before_sdk_update(tmp_path, firmware, version):
    import json
    import sys
    script = (ROOT / 'scripts/ci/prepare_scheduler_sdk.sh').read_text()
    code = script.split("<<'PY_POLICY'\n", 1)[1].split('\nPY_POLICY', 1)[0]
    (tmp_path / 'deps').mkdir()
    (tmp_path / 'deps/manifest.json').write_text(json.dumps({'elxr_version': '3.0.0_master_B2372'}))
    result = subprocess.run([sys.executable, '-c', code,
                             json.dumps({'elxr_version': firmware, 'sysroot-version': version})],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode != 0
    assert 'differs from Core' in result.stderr
