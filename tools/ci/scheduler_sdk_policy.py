#!/usr/bin/env python3
"""Use the selected Scheduler policy and admit LLiMa from the same firmware cohort."""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile

from published_artifact import acquire, digest

ROOT = Path(__file__).resolve().parents[2]


def validate(policy, core):
    if not isinstance(policy, dict):
        raise ValueError('Scheduler SDK policy must be an object')
    firmware = policy.get('elxr_version')
    match = re.fullmatch(r'(\d+\.\d+\.\d+)_(master|develop)_B([1-9][0-9]*)', firmware) if isinstance(firmware, str) else None
    if not match or match[1] != policy.get('platform-version') or match[1] != core.get('platform-version'):
        raise ValueError('Scheduler firmware does not match the Core platform')
    sdk = policy.get('sdk')
    if not isinstance(sdk, dict) or sdk.get('channel') not in ('develop', 'official') or sdk.get('channel') != core.get('sdk', {}).get('channel'):
        raise ValueError('Scheduler SDK channel differs from Core')
    core_pin = core.get('sdk', {}).get('sysroot-version')
    if core_pin is not None and core_pin != sdk.get('sysroot-version'):
        raise ValueError('Scheduler SDK channel differs from Core')
    base, channel, build = match.groups()
    pin = sdk.get('sysroot-version')
    suffix = ('' if pin == base else '~pre' + build) if channel == 'master' else r'~git[0-9]{12}\.[0-9a-f]{7,40}-' + build
    version = policy.get('sysroot-version')
    if (not isinstance(version, str) or not re.fullmatch(re.escape(base) + suffix, version)
            or (pin is not None and version != pin)):
        raise ValueError('Scheduler exact sysroot receipt differs from its firmware')
    return policy


def check_llima(policy, directory):
    directory = Path(directory)
    manifest_path = directory / 'llima-package-manifest.json'
    # A local publication may be rebuilt in place; its metadata takes precedence.
    metadata_path = directory / 'metadata.json'
    if not metadata_path.exists():
        metadata_path = directory / 'llima-policy-metadata.json'
    manifest = json.loads(manifest_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    checksums = metadata.get('resources-checksum', {})
    if checksums.get(manifest_path.name) != digest(manifest_path):
        raise ValueError('LLiMa firmware declaration differs from its publication')
    for role in ('core', 'dev', 'cli'):
        packages = list(directory.glob(f'sima-lmm-*-Linux-{role}.deb'))
        if len(packages) != 1 or checksums.get(packages[0].name) != digest(packages[0]):
            raise ValueError(f'LLiMa {role} DEB differs from the firmware publication')
    for key in ('platform-version', 'elxr_version'):
        if manifest.get(key) != policy[key]:
            raise ValueError(f'LLiMa {key} differs from selected Scheduler policy; rebuild matching LLiMa')
    sdk = manifest.get('sdk')
    if (not isinstance(sdk, dict) or sdk.get('channel') != policy['sdk']['channel']
            or (sdk.get('sysroot-version') is not None
                and sdk['sysroot-version'] != policy['sysroot-version'])):
        raise ValueError('LLiMa sdk differs from selected Scheduler policy; rebuild matching LLiMa')
    # Preserve root metadata beside download-only/cached DEBs for repeat admission.
    if metadata_path.name != 'llima-policy-metadata.json':
        shutil.copy2(metadata_path, directory / 'llima-policy-metadata.json')


def check_release_sdk(version, firmware, sdk_release, sysroot):
    """Require a prebuilt release SDK; never install or patch release payloads."""
    fields = {}
    for line in sdk_release.read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            if key in fields:
                raise ValueError(f'Duplicate SDK receipt field: {key}')
            fields[key] = value
    if (fields.get('Platform Version') != version or fields.get('Platform Base') != version
            or fields.get('Platform Channel') != 'release'):
        raise ValueError('Cached SDK does not contain the requested release; refresh its snapshot')
    overlay = sysroot / 'var/lib/sima-sdk/sysroot-overlay'
    if overlay.exists() or overlay.is_symlink():
        raise ValueError('Release SDK contains an overlay; use a fresh release snapshot')
    inventory = sysroot / 'var/lib/sima-sdk/sysroot-packages.tsv'
    if inventory.is_symlink():
        raise ValueError('SDK inventory must be a regular file')
    records = [line.split('\t') for line in inventory.read_text().splitlines()]
    for package in ('simaai-palette-modalix', 'simaai-base-files-modalix'):
        versions = [row[2] for row in records if len(row) >= 3 and row[0] == package
                    and row[1] in {'all', 'arm64'}]
        if versions != [version]:
            raise ValueError(f'Cached SDK {package} inventory differs: {versions}')
    if not (sysroot / 'usr/include').is_dir():
        raise ValueError('Cached SDK sysroot headers are missing')
    identities = re.findall(r'^SIMA_PACKAGE_VERSION\s*=\s*(\S+)\s*$',
                            (sysroot / 'etc/packageinfo').read_text(), re.M)
    if identities != [firmware]:
        raise ValueError(f'Cached SDK firmware identity differs: {identities}; expected {firmware}')
    print(f'SKIP sysroot installation: SDK already contains official release {version}; '
          'SDK receipt and Palette/base-files inventory match.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy-json', required=True)
    parser.add_argument('--core-manifest', type=Path, default=ROOT / 'deps/manifest.json')
    parser.add_argument('--check-release-sdk', action='store_true')
    parser.add_argument('--sdk-release', type=Path, default=Path('/etc/sdk-release'))
    parser.add_argument('--sysroot', type=Path, default=Path('/opt/toolchain/aarch64/modalix'))
    parser.add_argument('--field', choices=['sysroot-version'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--llima-dir', type=Path)
    parser.add_argument('--llima-selector')
    parser.add_argument('--environment')
    parser.add_argument('--cli', default='sima-cli')
    args = parser.parse_args()
    policy = validate(json.loads(args.policy_json), json.loads(args.core_manifest.read_text()))
    if args.check_release_sdk:
        check_release_sdk(policy['sysroot-version'], policy['elxr_version'], args.sdk_release, args.sysroot)
    if args.llima_selector:
        if not args.llima_dir or not args.environment:
            parser.error('Published LLiMa requires --llima-dir and --environment')
        branch, commit = args.llima_selector.rsplit(':', 1)
        if not re.fullmatch(r'[0-9a-f]{12,40}', commit):
            raise ValueError('LLiMa compatibility requires its immutable selected commit')
        with tempfile.TemporaryDirectory(prefix='core-llima-policy-') as temporary:
            target = Path(temporary) / 'publication'
            acquire(args.cli, 'llima', branch, commit, args.environment, target,
                    resource_pattern=r'llima-package-manifest\.json')
            shutil.copy2(target / 'llima-package-manifest.json', args.llima_dir)
            shutil.copy2(target / 'metadata.json', args.llima_dir / 'llima-policy-metadata.json')
    if args.llima_dir:
        check_llima(policy, args.llima_dir)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(policy, indent=2) + '\n')
    if args.field:
        print(policy[args.field])


if __name__ == '__main__':
    main()
