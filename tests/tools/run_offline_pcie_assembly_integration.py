#!/usr/bin/env python3
"""Explicit Docker integration test using synthetic Neat payloads and real OS deps.

This proves offline repository mechanics, not Neat or Platform qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
IMAGES = {
    'host': ('ubuntu@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac', 'amd64', 'ubuntu', '24.04'),
    'card': ('debian@sha256:62bd423aeee912639ca8ee681c2b762cb3e25137e0a92b20be455898a9f034ad', 'arm64', 'debian', '13'),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('work', type=Path, help='new Docker-bind-mountable directory')
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=False)
    spec = {'targets': {}}
    for target, (image, arch, os_id, version) in IMAGES.items():
        directory = work/target
        control = directory/'package/DEBIAN'
        control.mkdir(parents=True)
        (control/'control').write_text(
            f'Package: offline-pcie-fixture\nVersion: 1.0\nArchitecture: {arch}\n'
            'Maintainer: Test <test@example.invalid>\nDescription: Offline closure test fixture\n'
            'Depends: libjson-glib-1.0-0\n')
        subprocess.run(['docker', 'run', '--rm', '--platform', 'linux/'+arch,
                        '--mount', f'type=bind,src={directory},dst=/fixture',
                        '--entrypoint', 'dpkg-deb', image, '--build', '/fixture/package',
                        '/fixture/offline-pcie-fixture.deb'], check=True)
        wheel = directory/'offline_pcie_fixture-1.0-py3-none-any.whl'
        with zipfile.ZipFile(wheel, 'w') as archive:
            prefix = 'offline_pcie_fixture-1.0.dist-info/'
            archive.writestr(prefix+'METADATA', 'Metadata-Version: 2.1\nName: offline-pcie-fixture\nVersion: 1.0\nRequires-Dist: packaging==24.2\n')
            archive.writestr(prefix+'WHEEL', 'Wheel-Version: 1.0\nGenerator: integration-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n')
            archive.writestr(prefix+'RECORD', '')
        names = ['offline-pcie-fixture.deb', wheel.name]
        metadata = {'commit': 'abcdef123456', 'resources': names,
                    'resources-checksum': {n: hashlib.sha256((directory/n).read_bytes()).hexdigest() for n in names}}
        (directory/'metadata.json').write_text(json.dumps(metadata))
        spec['targets'][target] = {
            'image': image, 'architecture': arch, 'os_id': os_id, 'os_version': version,
            'core_revision': 'abcdef123456', 'publication': str(directory),
            'metadata_sha256': hashlib.sha256((directory/'metadata.json').read_bytes()).hexdigest(),
        }
    (work/'spec.json').write_text(json.dumps(spec, indent=2))
    subprocess.run(['python3', str(ROOT/'pcie_host/scripts/offline/assemble.py'), 'assemble',
                    '--spec', str(work/'spec.json'), '--output', str(work/'bundle')], check=True)
    # APT may create these beside the alternate dpkg status file. On Linux they
    # are root-owned/unreadable to the assembler; they are never release inputs.
    for target in IMAGES:
        for name in ('lock', 'lock-frontend'):
            if (work/'bundle'/target/name).exists():
                raise AssertionError('APT collection lock leaked into bundle: ' + target+'/'+name)


if __name__ == '__main__':
    main()
