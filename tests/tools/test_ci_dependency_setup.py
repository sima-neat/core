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


@pytest.mark.parametrize('mode', ['standard', 'fuzz'])
@pytest.mark.parametrize(('image_matches', 'configure_status', 'build_status'), [
    (True, 0, 0), (True, 23, 0), (True, 0, 37), (False, 0, 0),
])
def test_sdk_configuration_precedes_build_and_preserves_failures(
        tmp_path, mode, image_matches, configure_status, build_status):
    """Execute the workflow shell with Docker mocked; never run dpkg or APT."""
    import json
    import re
    import sys
    import textwrap

    script = textwrap.dedent(WORKFLOW.split('      build_command: |\n', 1)[1]
                            .split('\n      post_build_command:', 1)[0])
    values = {"inputs.build_mode || 'standard'": mode,
              'needs.resolve-vulcan-config.outputs.vulcan_env': 'production',
              'github.head_ref': 'feature/sdk', 'github.ref_name': 'feature/sdk',
              'github.ref_type': 'branch', 'github.sha': 'a' * 40}
    script = re.sub(r'\$\{\{\s*(.*?)\s*\}\}', lambda m: values[m[1]], script)
    commands = tmp_path / 'commands.jsonl'
    docker = tmp_path / 'docker'
    docker.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
log = Path(os.environ['COMMANDS'])
with log.open('a') as output:
    output.write(json.dumps(args) + '\\n')
if args == ['ps', '--format', '{{.ID}}']:
    print('selected-sdk')
elif args == ['exec', 'selected-sdk', 'test', '-f', '/etc/sdk-release']:
    pass
elif args == ['images', '--format', '{{.Repository}}:{{.Tag}}']:
    print('ghcr.io/sima-neat/sdk-develop:latest')
elif args[:2] == ['image', 'inspect']:
    print('sha256:' + 'a' * 64)
elif args[0] == 'inspect':
    print('sha256:' + ('a' if os.environ['IMAGE_MATCHES'] == '1' else 'b') * 64)
elif args == ['exec', '--user', 'root', '-e', 'DEBIAN_FRONTEND=noninteractive',
              'selected-sdk', 'dpkg', '--configure', '-a']:
    if int(os.environ['CONFIGURE_STATUS']):
        print('fixture package configuration failed', file=sys.stderr)
    sys.exit(int(os.environ['CONFIGURE_STATUS']))
elif args[0] == 'exec' and args[-3:-1] == ['bash', '-lc']:
    assert args[-4] == 'selected-sdk'
    assert './build.sh ${CORE_BUILD_ARGS}' in args[-1]
    sys.exit(int(os.environ['BUILD_STATUS']))
else:
    raise AssertionError(args)
''')
    docker.chmod(0o755)
    result = subprocess.run(['/bin/bash', '-c', script], cwd=tmp_path,
                            text=True, capture_output=True,
                            env={'PATH': f'{tmp_path}:/usr/bin:/bin', 'HOME': str(tmp_path),
                                 'COMMANDS': str(commands), 'IMAGE_MATCHES': str(int(image_matches)),
                                 'CONFIGURE_STATUS': str(configure_status),
                                 'BUILD_STATUS': str(build_status)})
    calls = [json.loads(line) for line in commands.read_text().splitlines()]
    configuration = [i for i, args in enumerate(calls) if args[-3:] == ['dpkg', '--configure', '-a']]
    build = [i for i, args in enumerate(calls) if args[-3:-1] == ['bash', '-lc']]
    expected_status = (configure_status or build_status) if image_matches else 1
    assert result.returncode == expected_status, result.stderr
    assert len(configuration) == int(image_matches)
    if not image_matches or configure_status:
        assert not build
    else:
        assert len(build) == 1 and configuration[0] < build[0]
    if configure_status and image_matches:
        assert 'fixture package configuration failed' in result.stderr


# overlay None is a fresh SDK image: no overlay, so the sysroot is the image's platform.
@pytest.mark.parametrize(('update', 'image', 'overlay', 'code'), [
    ('OFF', '3.0.0~pre2371', '3.0.0~pre2372', 0), ('ON', '3.0.0~pre2371', '3.0.0~pre2372', 0),
    ('OFF', '3.0.0~pre2372', '3.0.0~pre2371', 0),
    ('OFF', '3.0.0~pre2372', None, 0), ('ON', '3.0.0~pre2372', None, 0),
    ('OFF', '3.0.0~pre2371', None, 0), ('ON', '3.0.0~pre2371', None, 1),
])
def test_sdk_shell_checks_sysroot_and_downloads_exact_archives(tmp_path, update, image, overlay, code):
    """Run the real shell with sysroot, APT and the coordinator mocked."""
    import json
    import os
    import shlex
    calls = tmp_path / 'calls'
    tools = tmp_path / 'tools'; tools.mkdir()
    (tools / 'build_dependencies.py').write_text('''import json, os, pathlib, shlex, sys
command = sys.argv[1]
if command == 'sdk-policy': print(json.dumps({'elxr_version':'3.0.0_master_B2372', 'sysroot-version':'3.0.0~pre2372', 'platform-version':'3.0.0', 'sdk':{'channel':'develop'}}))
elif command == 'verify-sdk': print(json.dumps({'status':'verified'}))
else:
    root = pathlib.Path(sys.argv[sys.argv.index('--platform-package-dir')+1])
    print('export NEAT_RUNNER_PLATFORM_RUNTIME_DEBS=' + shlex.quote(json.dumps(sorted(map(str, root.glob('*.deb'))))))
''')
    coordinator = tmp_path / 'scheduler/usr/share/SimaScheduler/tools'; coordinator.mkdir(parents=True)
    (coordinator / 'client_dependencies.py').write_text('''
def receipt(root, name): return name
def platform_provider_contracts(runner):
    return {'heap': {'debian_provider': {'Package': 'simaai-heap', 'Architecture': 'arm64', 'Version': '3.0~pre2372'}}}
''')
    stubs = tmp_path / 'bin'; stubs.mkdir()
    (stubs / 'sysroot').write_text(f'#!/bin/sh\necho "sysroot $* SDK_PKG_LIST=$SDK_PKG_LIST" >> {calls}\n'
                                   f'[ "$1" = status ] && printf "%s\\n" "SDK image platform:       {image}" '
                                   f'"Sysroot overlay state:    {"active" if overlay else "inactive"}" '
                                   f'"Sysroot overlay revision: {overlay or "none"}"\nexit 0\n')
    (stubs / 'apt-get').write_text(f'#!/bin/sh\necho "apt-get $* APT_CONFIG=$APT_CONFIG" >> {calls}\n'
                                   '[ "$1" = download ] && touch "$(echo "$2" | tr ":=" "__").deb"\nexit 0\n')
    for stub in stubs.iterdir():
        stub.chmod(0o755)
    env_script = tmp_path / 'env.sh'; env_script.write_text('export CXX=cross-g++-14\n')
    script = (ROOT / 'scripts/ci/prepare_scheduler_sdk.sh').read_text()
    script = script.replace('/opt/bin/simaai-init-build-env', str(env_script))
    script = script.replace('tools/ci/scheduler_sdk_policy.py', str(ROOT / 'tools/ci/scheduler_sdk_policy.py'))
    script = script.replace('[[ $(id -u) == 0 ]]', '[[ 0 == 0 ]]')
    (tmp_path / 'deps').mkdir()
    (tmp_path / 'deps/manifest.json').write_text(json.dumps({'platform-version': '3.0.0', 'sdk': {'channel': 'develop'}}))
    cache = tmp_path / 'cache'
    result = subprocess.run(['/bin/bash', '-c', script, 'sdk-test', str(cache)], cwd=tmp_path, text=True,
                            capture_output=True, env={**os.environ, 'PATH': f'{stubs}:{os.environ["PATH"]}',
                                                      'SDK_PKG_LIST': '\tfixture-dev,\t,extra-dev',
                                                      'NEAT_UPDATE_SYSROOT': update,
                                                      'NEAT_SCHEDULER_PACKAGE_DIR': str(tmp_path / 'scheduler'),
                                                      'NEAT_RUNNER_PACKAGE_DIR': str(tmp_path / 'runner')})
    log = calls.read_text().splitlines()
    updates = [line for line in log if line.startswith('sysroot update')]
    assert updates == ([f'sysroot update 3.0.0~pre2372 SDK_PKG_LIST=fixture-dev,extra-dev'] if update == 'ON' else [])
    assert result.returncode == code, result.stderr
    if code:
        assert 'after updating' in result.stderr
        return
    mismatch = (overlay or image) != '3.0.0~pre2372'
    assert ('sudo env SDK_PKG_LIST=fixture-dev,extra-dev sysroot update 3.0.0~pre2372' in result.stderr) == mismatch
    apt_config = str(cache / 'apt/apt.conf')
    assert f'apt-get download simaai-heap:arm64=3.0~pre2372 APT_CONFIG={apt_config}' in log
    assert 'https://debian.neat.sima.ai/pre-release agate' in (cache / 'apt/sources.list').read_text()
    archive = cache / 'downloads/simaai-heap_arm64_3.0~pre2372.deb'
    assert result.stdout.strip() == 'export NEAT_RUNNER_PLATFORM_RUNTIME_DEBS=' + shlex.quote(json.dumps([str(archive)]))
    assert not list(coordinator.rglob('__pycache__'))


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
