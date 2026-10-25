"""Receipt selection and LLiMa compatibility without network or SDK mutation."""
import json
from pathlib import Path
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
def test_reject_swapped_or_missing_publication_bytes(tmp_path, name):
    fixture(tmp_path)
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
