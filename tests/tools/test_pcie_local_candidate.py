"""Local producer inputs must be complete and checked before helper execution."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('pcie_acquire', ROOT/'pcie_host/scripts/acquire_scheduler.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


COMMIT = 'a' * 40
IDENTITY = (COMMIT, 'ubuntu24', 'amd64')


@pytest.fixture
def candidate(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    resources = {
        'host-candidate.json': json.dumps({
            'schema_version': 2, 'profile': 'pcie-host', 'distro': 'ubuntu24',
            'architecture': 'amd64', 'source_commit': COMMIT}).encode(),
        'native_artifacts.py': b'not executable; fixture producer resource'}
    for name, payload in resources.items():
        (source/name).write_bytes(payload)
    metadata = {'repository': 'sima-neat/scheduler', 'commit': COMMIT, 'branch': 'topic',
                'artifact': {'profile': 'pcie-host', 'distro': 'ubuntu24', 'architecture': 'amd64'},
                'resources': list(resources),
                'resources-checksum': {name: hashlib.sha256(payload).hexdigest()
                                       for name, payload in resources.items()}}
    (source/'metadata.json').write_text(json.dumps(metadata))
    return source, metadata, tmp_path/'copy'


def test_local_candidate_copies_verified_bytes_and_keeps_identity(candidate):
    source, metadata, dest = candidate
    assert module.local_candidate(source, dest, *IDENTITY) == metadata
    assert (dest/'native_artifacts.py').read_bytes() == (source/'native_artifacts.py').read_bytes()


@pytest.mark.parametrize('failure', ['changed', 'symlink', 'missing', 'extra', 'traversal'])
def test_bad_local_candidate_is_rejected_before_staging(candidate, failure):
    source, metadata, dest = candidate
    path = source/'native_artifacts.py'
    if failure == 'changed':
        path.write_text('changed')
    elif failure == 'symlink':
        path.rename(source/'original')
        path.symlink_to(source/'original')
    elif failure == 'missing':
        path.unlink()
    elif failure == 'extra':
        metadata['resources'].append('missing.py')
    else:
        metadata['resources'].append('../native_artifacts.py')
        metadata['resources-checksum']['../native_artifacts.py'] = '0'*64
    (source/'metadata.json').write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        module.local_candidate(source, dest, *IDENTITY)
    assert not dest.exists()


def test_local_rebuild_during_copy_is_rejected_before_helper_import(candidate, monkeypatch):
    source, metadata, dest = candidate
    copy = module.shutil.copy2

    def rebuild_and_copy(path, destination):
        Path(path).write_bytes(b'changed by concurrent local producer rebuild')
        return copy(path, destination)

    monkeypatch.setattr(module.shutil, 'copy2', rebuild_and_copy)
    with pytest.raises(ValueError, match='Staged Scheduler resource changed: host-candidate.json'):
        module.local_candidate(source, dest, *IDENTITY)
