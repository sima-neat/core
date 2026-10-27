"""Selected Scheduler policy is the only authority for offline CI sysroots."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parents[2]/'pcie_host/scripts/offline'
SPEC = importlib.util.spec_from_file_location('offline_sysroot', HERE/'sysroot_baseline.py')
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


@pytest.mark.parametrize('observed', ['3.0.0~pre2372', '3.0.0~pre2424'])
def test_sysroot_resolution_must_match_selected_scheduler(observed):
    policy = {'elxr_version': '3.0.0_master_B2372', 'sysroot-version': '3.0.0~pre2372'}
    def declaration(value):
        assert value == {'elxr_version': policy['elxr_version']}
        return policy['elxr_version'], None
    resolver = SimpleNamespace(manifest_selection=declaration,
        resolve=lambda firmware: {'sysroot_version': observed, 'channel': 'pre-release'})
    with patch.object(baseline, 'load', return_value=resolver):
        if observed == policy['sysroot-version']:
            assert baseline.resolve(policy)['sysroot_version'] == observed
        else:
            with pytest.raises(ValueError, match='differs from selected Scheduler'):
                baseline.resolve(policy)


@pytest.mark.parametrize('tamper', [False, True])
def test_sdk_policy_is_bound_to_selected_scheduler_inventory(tmp_path, tamper):
    policy = {'sysroot-version': '3.0.0~pre2372'}
    data = json.dumps(policy).encode()
    selected = {'installed_files': {'share/SimaScheduler/dependency-policy.json': {
        'kind': 'file', 'sha256': hashlib.sha256(data).hexdigest()}}}
    (tmp_path/'resolved-deps-manifest.json').write_text(json.dumps({'scheduler': {'buildinfo': selected}}))
    def extract(archive, root):
        destination = root/'usr/share/SimaScheduler/dependency-policy.json'
        destination.parent.mkdir(parents=True)
        destination.write_bytes(b'{}' if tamper else data)
    def authenticate(prefix, info):
        assert info == selected
    helper = SimpleNamespace(_scheduler_archives=lambda root: [root/'scheduler.deb'],
                             extract_scheduler=extract, authenticate_tool=authenticate)
    with patch.object(baseline, 'load', return_value=helper):
        if tamper:
            with pytest.raises(ValueError, match='differs from selected receipt'):
                baseline.sdk_policy(tmp_path)
        else:
            assert baseline.sdk_policy(tmp_path) == policy


def test_native_sysroot_updates_installed_reverse_dependencies_together(tmp_path):
    for name in ('tvm', 'python3-tvm', 'libc6', 'unneeded-dev'):
        (tmp_path/(name+'.deb')).touch()
    with patch.object(baseline.subprocess, 'check_output',
                      side_effect=lambda args, **kwargs: Path(args[2]).stem):
        selected = baseline.native_archives(tmp_path, {'tvm'}, {'python3-tvm', 'libc6'})
    assert set(selected) == {'tvm', 'python3-tvm', 'libc6'}


def test_missing_sysroot_provider_fails_before_native_install(tmp_path):
    with pytest.raises(ValueError, match='lacks native providers: simaai-heap'):
        baseline.native_archives(tmp_path, {'simaai-heap'}, set())
