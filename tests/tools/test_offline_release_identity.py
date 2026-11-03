"""Release bundles bind the source revision through the hashed Core receipt."""
import importlib.util
import io
import json
from pathlib import Path
import tarfile
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('release_assembly', ROOT / 'pcie_host/scripts/offline/assemble.py')
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)
SHA = 'a' * 40


def receipt():
    return {'version': '0.6.0', 'build': {
        'repository': 'sima-neat/core', 'git_ref': 'v0.6.0', 'git_ref_type': 'tag',
        'source_status': 'unavailable', 'git_dirty': None, 'git_sha': '',
        'ci_requested_git_sha': SHA}}


def publication(root, role, document=None, duplicate=False, link=False):
    root.mkdir()
    runtime = 'sima-neat-0.6.0-Linux-core.deb' if role == 'card' else 'host.deb'
    # The subprocess seam below supplies the fsys tar stream, just as dpkg-deb
    # does. Hash verification and receipt parsing remain real.
    with tarfile.open(root / runtime, 'w') as archive:
        if document is not None:
            for _ in range(2 if duplicate else 1):
                data = json.dumps(document).encode()
                entry = tarfile.TarInfo('./usr/share/sima-neat/buildinfo.json')
                if link:
                    entry.type = tarfile.SYMTYPE
                    entry.linkname = 'other.json'
                    archive.addfile(entry)
                else:
                    entry.size = len(data)
                    archive.addfile(entry, io.BytesIO(data))
    (root / 'runtime.whl').write_bytes(b'wheel')
    metadata = {'version': '0.6.0', 'resources': [runtime, 'runtime.whl']}
    metadata['resources-checksum'] = {name: assembly.sha(root / name) for name in metadata['resources']}
    if role == 'host':
        metadata['commit'] = SHA
    extra = root / 'sima-neat-0.6.0-Linux-extras.tar.gz'
    with tarfile.open(extra, 'w:gz') as archive:
        entry = tarfile.TarInfo('share/tutorials/README.md')
        entry.size = len(b'extras')
        archive.addfile(entry, io.BytesIO(b'extras'))
    metadata['selectable-resources'] = [{'resource': extra.name, 'checksum': assembly.sha(extra)}]
    (root / 'metadata.json').write_text(json.dumps(metadata))
    return {'publication': str(root), 'metadata_sha256': assembly.sha(root / 'metadata.json'),
            'core_revision': SHA[:12], 'image': 'sha256:' + 'b' * 64,
            'architecture': 'arm64' if role == 'card' else 'amd64',
            'os_id': 'elxr' if role == 'card' else 'ubuntu',
            'os_version': '26.04' if role == 'card' else '22.04'}


def fsys_tar(command, *, stdout, **kwargs):
    assert command[:2] == ['dpkg-deb', '--fsys-tarfile']
    stdout.write(Path(command[2]).read_bytes())


@pytest.mark.parametrize('scope', ['core', 'pcie'])
@pytest.mark.parametrize('source', ['archive', 'clean_git'])
def test_release_assembly_uses_exact_receipt_revision(tmp_path, scope, source):
    document = receipt()
    if source == 'clean_git':
        document['build'].update(source_status='clean', git_dirty=False, git_sha=SHA)
    targets = {'card': publication(tmp_path / 'card', 'card', document)}
    if scope == 'pcie':
        targets['host'] = publication(tmp_path / 'host', 'host')
    output = tmp_path / 'bundle'
    with patch.object(assembly.subprocess, 'run', side_effect=fsys_tar), patch.object(assembly, 'run') as docker:
        assembly.assemble({'scope': scope, 'targets': targets}, output)
    assert docker.call_count == 2 * len(targets)
    assert assembly.verify_bundle(output)['core_revision'] == SHA[:12]


@pytest.mark.parametrize('mutation', ['wrong_commit', 'dirty', 'wrong_version', 'wrong_ref',
                                     'wrong_repository', 'missing', 'duplicate', 'symlink'])
def test_bad_release_receipt_fails_before_any_container(tmp_path, mutation):
    document = receipt()
    if mutation == 'wrong_commit':
        document['build']['ci_requested_git_sha'] = 'c' * 40
    elif mutation == 'dirty':
        document['build'].update(source_status='dirty', git_dirty=True, git_sha=SHA)
    elif mutation == 'wrong_version':
        document['version'] = '0.5.0'
    elif mutation == 'wrong_ref':
        document['build']['git_ref'] = 'v0.5.0'
    elif mutation == 'wrong_repository':
        document['build']['repository'] = 'other/core'
    target = publication(tmp_path / 'card', 'card', None if mutation == 'missing' else document,
                         duplicate=mutation == 'duplicate', link=mutation == 'symlink')
    with patch.object(assembly.subprocess, 'run', side_effect=fsys_tar), patch.object(assembly, 'run') as docker:
        with pytest.raises(ValueError):
            assembly.assemble({'scope': 'core', 'targets': {'card': target}}, tmp_path / 'bundle')
    docker.assert_not_called()
    assert not (tmp_path / 'bundle').exists()
