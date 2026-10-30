"""Official SDK admission and manifest-driven release selection."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ci'))
import scheduler_sdk_policy as policy


def release_policy():
    return {'platform-version': '3.0.0', 'elxr_version': '3.0.0_master_B2499',
            'sdk': {'channel': 'develop', 'sysroot-version': '3.0.0'},
            'sysroot-version': '3.0.0'}


def test_inherit_official_scheduler_pin_without_duplicate_core_firmware():
    selected = release_policy()
    core = json.loads((ROOT / 'deps/manifest.json').read_text())
    assert policy.validate(selected, core) == selected
    for version in ('3.0.0~pre2499', '3.0.1', ''):
        with pytest.raises(ValueError):
            policy.validate(dict(selected, **{'sysroot-version': version}), core)
    assert core['scheduler'] == {'policy': 'snap'}
    assert 'elxr_version' not in core


@pytest.mark.parametrize('bad', [None, 'receipt', 'duplicate-receipt', 'inventory',
                                 'duplicate-inventory', 'firmware', 'overlay'])
def test_release_sdk_requires_matching_receipt_inventory_and_firmware(tmp_path, bad):
    receipt = tmp_path / 'sdk-release'
    receipt.write_text('Platform Version = 3.0.0\nPlatform Base = 3.0.0\nPlatform Channel = release\n')
    inventory = tmp_path / 'var/lib/sima-sdk/sysroot-packages.tsv'
    inventory.parent.mkdir(parents=True)
    inventory.write_text('simaai-palette-modalix\tarm64\t3.0.0\t/usr\n'
                         'simaai-base-files-modalix\tall\t3.0.0\t/etc\n')
    (tmp_path / 'usr/include').mkdir(parents=True)
    firmware = tmp_path / 'etc/packageinfo'
    firmware.parent.mkdir()
    firmware.write_text('SIMA_PACKAGE_VERSION = 3.0.0_master_B2499\n')
    if bad == 'receipt': receipt.write_text(receipt.read_text().replace('release', 'daily'))
    if bad == 'duplicate-receipt': receipt.write_text(receipt.read_text() + 'Platform Base = 3.0.0\n')
    if bad == 'inventory': inventory.write_text(inventory.read_text().replace('3.0.0', '3.0.1'))
    if bad == 'duplicate-inventory': inventory.write_text(inventory.read_text() * 2)
    if bad == 'firmware': firmware.write_text('SIMA_PACKAGE_VERSION = 3.0.0_master_B2372\n')
    if bad == 'overlay': (inventory.parent / 'sysroot-overlay').mkdir()
    def check():
        policy.check_release_sdk('3.0.0', '3.0.0_master_B2499', receipt, tmp_path)
    if bad:
        with pytest.raises(ValueError): check()
    else: check()


def test_devkit_guard_uses_shared_resolved_firmware():
    workflow = (ROOT / '.github/workflows/vulcan-ci.yml').read_text()
    assert 'EXPECTED_ELXR_VERSION: ${{ needs.resolve-test-runner.outputs.devkit_firmware }}' in workflow
    assert 'vulcan-resolve-modalix-runner.yml@main' in workflow
    assert 'manifest_path: scheduler-sdk-policy.json' in workflow
