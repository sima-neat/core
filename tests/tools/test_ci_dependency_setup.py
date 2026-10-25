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
    import os
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
if command == 'sdk-policy': print(json.dumps({'elxr_version':'3.0.0_master_B2372', 'sysroot-version':'3.0.0~pre2372', 'platform-version':'3.0.0', 'sdk':{'channel':'develop'}}))
elif command == 'verify-sdk': print(json.dumps({'status':'verified'}))
else:
    root = pathlib.Path(sys.argv[sys.argv.index('--platform-package-dir')+1])
    matches = list(root.rglob('consumed.deb'))
    if len(matches) != 1: raise SystemExit(31)
    p = matches[0]
    print('export NEAT_RUNNER_PLATFORM_RUNTIME_DEBS=' + shlex.quote(json.dumps([str(p)])))
''')
    official = tmp_path / 'official.sh'
    official.write_text('#!/bin/sh\nset -eu\n[ "$SDK_PKG_LIST" = fixture-dev ]\n'
                        'if [ "$1" = update ]; then\n'
                        '  mkdir -p "$SYSROOT_UPDATE_DOWNLOAD_DIR"\n'
                        f'  cp {shlex.quote(str(cache))}/*.deb "$SYSROOT_UPDATE_DOWNLOAD_DIR/"\n'
                        'fi\n')
    official.chmod(0o755)
    env = tmp_path / 'env.sh'; env.write_text('export CXX=cross-g++-14\n')
    script = script.replace('sysroot update', str(official) + ' update').replace('sysroot status', str(official) + ' status')
    script = script.replace('/opt/bin/simaai-init-build-env', str(env))
    script = script.replace('tools/ci/scheduler_sdk_policy.py', str(ROOT / 'tools/ci/scheduler_sdk_policy.py'))
    (tmp_path / 'deps').mkdir()
    (tmp_path / 'deps/manifest.json').write_text(json.dumps({'platform-version': '3.0.0', 'sdk': {'channel': 'develop'}}))
    # Never grant fixture privileges; source production root/sudo branches are unchanged.
    script = script.replace('[[ $(id -u) == 0 ]]', '[[ 0 == 0 ]]')
    if missing_archive: provider.unlink()
    retained = tmp_path / 'retained'
    result = subprocess.run(['/bin/bash', '-c', script, 'sdk-test', str(retained)], cwd=tmp_path,
                            text=True, capture_output=True, env={**os.environ, 'SDK_PKG_LIST': 'fixture-dev'})
    if missing_archive:
        assert result.returncode == 31, result.stderr
        assert not list(retained.glob('*.deb'))
    else:
        assert result.returncode == 0, result.stderr
        selected = retained / 'retained'
        assert [p.name for p in selected.glob('*.deb')] == ['consumed.deb']
        assert (selected / 'consumed.deb').read_bytes() == provider.read_bytes()
        assert str(selected / 'consumed.deb') in result.stdout
        assert (retained / 'downloads/consumed.deb').read_bytes() == provider.read_bytes()
        assert str(retained / 'downloads') not in result.stdout
        assert str(cache) not in result.stdout


@pytest.mark.parametrize('version', ['', '3.0.0~pre2371'])
def test_invalid_scheduler_receipt_stops_before_sdk_mutation(tmp_path, version):
    import json
    tools = tmp_path / 'tools'
    tools.mkdir()
    policy = {'platform-version': '3.0.0', 'sdk': {'channel': 'develop'},
              'elxr_version': '3.0.0_master_B2372', 'sysroot-version': version}
    (tools / 'build_dependencies.py').write_text('print(' + repr(json.dumps(policy)) + ')\n')
    script = (ROOT / 'scripts/ci/prepare_scheduler_sdk.sh').read_text()
    script = script.replace('tools/ci/scheduler_sdk_policy.py', str(ROOT / 'tools/ci/scheduler_sdk_policy.py'))
    marker = tmp_path / 'mutation'
    script = script.replace('sysroot update', f'touch {marker}; false')
    result = subprocess.run(['/bin/bash', '-c', script, 'policy-test', str(tmp_path / 'cache')],
                            cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Scheduler exact sysroot receipt differs' in result.stderr
    assert not marker.exists()
