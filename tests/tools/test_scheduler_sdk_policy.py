"""Receipt selection and LLiMa compatibility without network or SDK mutation."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ci'))
import scheduler_sdk_policy as policy

CORE = {'platform-version': '3.0.0', 'sdk': {'channel': 'develop'}}


def receipt(firmware='3.0.0_master_B2372', version='3.0.0~pre2372'):
    return dict(CORE, elxr_version=firmware, **{'sysroot-version': version})


@pytest.mark.parametrize('firmware,version', [
    ('3.0.0_master_B2372', '3.0.0~pre2372'),
    ('3.0.0_master_B2400', '3.0.0~pre2400'),
    ('3.0.0_develop_B2121', '3.0.0~git202610140145.e5ece50-2121'),
])
def test_inherit_firmware_and_exact_version_without_core_pin(firmware, version):
    selected = receipt(firmware, version)
    assert policy.validate(selected, CORE) == selected
    assert 'elxr_version' not in json.loads((ROOT / 'deps/manifest.json').read_text())


@pytest.mark.parametrize('update', [
    {'elxr_version': None}, {'elxr_version': 'malformed'},
    {'elxr_version': '3.0.0_master_B2371'}, {'sysroot-version': None},
    {'sysroot-version': '3.0.0~git202610140145.e5ece50-2372'},
    {'platform-version': '2.0.0'}, {'sdk': {'channel': 'official'}},
])
def test_reject_missing_or_conflicting_scheduler_policy(update):
    with pytest.raises(ValueError):
        policy.validate(dict(receipt(), **update), CORE)


def fixture(directory, update=None):
    manifest = dict(CORE, elxr_version='3.0.0_master_B2372')
    manifest.update(update or {})
    (directory / 'llima-package-manifest.json').write_text(json.dumps(manifest))
    for role in ('core', 'dev', 'cli'):
        (directory / f'sima-lmm-0.6.0-Linux-{role}.deb').write_bytes(role.encode())
    metadata = {'resources-checksum': {p.name: policy.digest(p) for p in directory.iterdir()}}
    (directory / 'metadata.json').write_text(json.dumps(metadata))


def test_llima_matching_publication_and_cached_evidence(tmp_path):
    fixture(tmp_path)
    policy.check_llima(receipt(), tmp_path)
    (tmp_path / 'metadata.json').unlink()
    policy.check_llima(receipt(), tmp_path)


def test_llima_rebuilt_in_place_refreshes_cached_metadata(tmp_path):
    fixture(tmp_path)
    policy.check_llima(receipt(), tmp_path)
    cached = tmp_path / 'llima-policy-metadata.json'
    previous = cached.read_bytes()
    root = tmp_path / 'metadata.json'
    metadata = json.loads(root.read_text())
    for package in tmp_path.glob('*.deb'):
        package.write_bytes(package.read_bytes() + b' rebuilt')
        metadata['resources-checksum'][package.name] = policy.digest(package)
    root.write_text(json.dumps(metadata))
    current = root.read_bytes()
    assert cached.read_bytes() == previous != current

    policy.check_llima(receipt(), tmp_path)

    assert root.read_bytes() == cached.read_bytes() == current
    root.unlink()
    policy.check_llima(receipt(), tmp_path)


@pytest.mark.parametrize('resource', [None, 'llima-package-manifest.json',
                                    'sima-lmm-0.6.0-Linux-core.deb',
                                    'sima-lmm-0.6.0-Linux-dev.deb',
                                    'sima-lmm-0.6.0-Linux-cli.deb'])
def test_invalid_current_metadata_does_not_fall_back_to_cache(tmp_path, resource):
    fixture(tmp_path)
    policy.check_llima(receipt(), tmp_path)
    cached = tmp_path / 'llima-policy-metadata.json'
    previous = cached.read_bytes()
    root = tmp_path / 'metadata.json'
    metadata = json.loads(root.read_text())
    if resource is None:
        root.write_text('not JSON')
    else:
        metadata['resources-checksum'][resource] = '0' * 64
        root.write_text(json.dumps(metadata))

    with pytest.raises(ValueError):
        policy.check_llima(receipt(), tmp_path)
    assert cached.read_bytes() == previous


@pytest.mark.parametrize('update', [
    {'elxr_version': '3.0.0_master_B2371'}, {'elxr_version': None},
    {'platform-version': '2.0.0'}, {'sdk': {'channel': 'official'}},
])
def test_reject_llima_incompatible_firmware_or_sdk(tmp_path, update):
    fixture(tmp_path, update)
    with pytest.raises(ValueError, match='differs from selected Scheduler'):
        policy.check_llima(receipt(), tmp_path)


@pytest.mark.parametrize('name', ['llima-package-manifest.json', 'sima-lmm-0.6.0-Linux-core.deb',
                                  'sima-lmm-0.6.0-Linux-dev.deb', 'sima-lmm-0.6.0-Linux-cli.deb'])
@pytest.mark.parametrize('cached_only', [False, True])
def test_reject_swapped_or_missing_publication_bytes(tmp_path, name, cached_only):
    fixture(tmp_path)
    policy.check_llima(receipt(), tmp_path)
    if cached_only:
        (tmp_path / 'metadata.json').unlink()
    target = tmp_path / name
    original = target.read_bytes()
    target.write_bytes(original + b' ')
    with pytest.raises(ValueError, match='differs from'):
        policy.check_llima(receipt(), tmp_path)
    target.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        policy.check_llima(receipt(), tmp_path)


def test_llima_checked_before_installation_and_policy_retained_after_packaging():
    source = (ROOT / 'build.sh').read_text()
    ensure = source.split('ensure_neat_llima() {', 1)[1].split('ensure_neat_llima_headers()', 1)[0]
    assert ensure.index('scheduler_sdk_policy.py') < ensure.index('record_sysroot_files')
    main = source.split('main() {', 1)[1]
    assert main.index('generate_package_metadata_if_requested') < main.index('--output dist/scheduler-sdk-policy.json')


@pytest.mark.parametrize('role', ['core', 'dev', 'cli'])
@pytest.mark.parametrize('root_state', ['valid', 'missing', 'tampered', 'duplicate'])
def test_llima_installs_and_caches_only_validated_root_packages(tmp_path, role, root_state):
    publication = tmp_path / 'local publication'
    publication.mkdir()
    fixture(publication)
    package = publication / f'sima-lmm-0.6.0-Linux-{role}.deb'
    nested = publication / 'build' / package.name
    nested.parent.mkdir()
    nested.write_bytes(b'stale nested package')
    assert str(nested) < str(package)
    if root_state == 'missing':
        package.unlink()
    elif root_state == 'tampered':
        package.write_bytes(b'unpublished root bytes')
    elif root_state == 'duplicate':
        (publication / f'sima-lmm-0.5.0-Linux-{role}.deb').write_bytes(b'duplicate')

    cache = tmp_path / 'cache'
    sysroot = tmp_path / 'sysroot'
    for name in ('cmake/SimaLMM/SimaLMMConfig.cmake', 'libsima_lmm_runtime.so'):
        path = sysroot / 'usr/lib/aarch64-linux-gnu' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    source = (ROOT / 'build.sh').read_text()
    start = source.index('ensure_neat_llima() {')
    function = source[start:source.index('\n}\n', start) + 2]
    # Run the real selection, validation and cache copy, mocking installation.
    script = '''set -euo pipefail
resolve_neat_llima_ref() { NEAT_LLIMA_REQUESTED_REF=develop:fixture; }
fetch_neat_llima_vulcan_artifacts() { echo 'unexpected download' >&2; return 99; }
ensure_llima_sdk_sysroot_deps() { :; }
run_privileged() { echo 'unexpected privileged operation' >&2; return 99; }
record_sysroot_files() { printf '%s\\n' "$@" > "$TEST_TMP/recorded"; }
dpkg-deb() {
  [[ "$1" == -x && "$3" == "$SYSROOT" ]] || return 99
  printf '%s\\n' "$2" >> "$TEST_TMP/extracted"
}
mktemp() { command mktemp -d "$TEST_TMP/work.XXXXXX"; }
python3() {
  if [[ "$1" == tools/build_dependencies.py ]]; then
    printf '%s\\n' "$SCHEDULER_POLICY"
  else
    command python3 "$@"
  fi
}
'''
    result = subprocess.run(['/bin/bash', '-c', script + function + '\nensure_neat_llima\n'],
                            cwd=ROOT, text=True, capture_output=True,
                            env={'PATH': '/usr/bin:/bin', 'TEST_TMP': str(tmp_path),
                                 'PYTHONDONTWRITEBYTECODE': '1',
                                 'NEAT_LLIMA_ARTIFACT_DIR': str(publication),
                                 'NEAT_DEPS_DIR': str(tmp_path), 'NEAT_LLIMA_DEB_DIR': str(cache),
                                 'REPO_ROOT': str(ROOT), 'ELXR_SDK': 'ON', 'SYSROOT': str(sysroot),
                                 'SCHEDULER_POLICY': json.dumps(receipt())})
    if root_state != 'valid':
        assert result.returncode != 0, result.stdout
        if root_state == 'missing':
            assert 'debs were not found' in result.stderr
        else:
            assert 'differs from the firmware publication' in result.stderr
        assert not cache.exists()
        assert not (tmp_path / 'recorded').exists()
        assert not (tmp_path / 'extracted').exists()
        return
    assert result.returncode == 0, result.stderr
    expected = [str(publication / f'sima-lmm-0.6.0-Linux-{r}.deb') for r in ('core', 'dev')]
    assert (tmp_path / 'recorded').read_text().splitlines() == expected
    assert (tmp_path / 'extracted').read_text().splitlines() == expected
    for root_package in publication.glob('*.deb'):
        assert (cache / root_package.name).read_bytes() == root_package.read_bytes()
    policy.check_llima(receipt(), cache)


def test_remote_policy_uses_selected_immutable_llima_commit(tmp_path, monkeypatch):
    fixture(tmp_path)
    manifest = (tmp_path / 'llima-package-manifest.json').read_bytes()
    metadata = (tmp_path / 'metadata.json').read_bytes()
    (tmp_path / 'llima-package-manifest.json').unlink()
    (tmp_path / 'metadata.json').unlink()
    calls = []

    def acquire(cli, component, branch, commit, environment, output, **kwargs):
        calls.append((component, branch, commit, kwargs['resource_pattern']))
        output.mkdir()
        (output / 'llima-package-manifest.json').write_bytes(manifest)
        (output / 'metadata.json').write_bytes(metadata)

    monkeypatch.setattr(policy, 'acquire', acquire)
    monkeypatch.setattr(sys, 'argv', ['policy', '--policy-json', json.dumps(receipt()),
        '--llima-dir', str(tmp_path), '--llima-selector', 'develop:' + 'a' * 40,
        '--environment', 'production', '--output', str(tmp_path / 'scheduler-policy.json')])
    policy.main()
    assert calls == [('llima', 'develop', 'a' * 40, r'llima-package-manifest\.json')]
    assert json.loads((tmp_path / 'scheduler-policy.json').read_text()) == receipt()
