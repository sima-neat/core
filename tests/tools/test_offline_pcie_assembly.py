"""Fail-closed publication, baseline, and bundle integrity contracts."""
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

SPEC = importlib.util.spec_from_file_location(
    'offline_assembly', Path(__file__).resolve().parents[2] / 'pcie_host/scripts/offline/assemble.py')
assembly = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assembly)


def publication(root):
    root.mkdir()
    for name in ('runtime.deb', 'runtime.whl'):
        (root / name).write_text('fixture')
    metadata = {'resources': ['runtime.deb', 'runtime.whl'], 'resources-checksum': {
        name: assembly.sha(root/name) for name in ('runtime.deb', 'runtime.whl')}}
    (root/'metadata.json').write_text(json.dumps(metadata))
    return assembly.sha(root/'metadata.json')


def test_publication_requires_all_hashes_and_payloads(tmp_path):
    root = tmp_path/'publication'
    digest = publication(root)
    assembly.verify_publication(root, digest)
    (root/'runtime.deb').write_text('changed')
    with pytest.raises(ValueError, match='checksum'):
        assembly.verify_publication(root, digest)


def test_metadata_cannot_replace_trust_anchor(tmp_path):
    digest = publication(tmp_path/'pub')
    with pytest.raises(ValueError, match='metadata checksum'):
        assembly.verify_publication(tmp_path/'pub', '0'*64)
    assert digest != '0'*64


def test_metadata_path_traversal_rejected(tmp_path):
    publication(tmp_path/'pub')
    path = tmp_path/'pub/metadata.json'
    data = {'resources': ['../outside'], 'resources-checksum': {'../outside': '0'*64}}
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='unsafe'):
        assembly.verify_publication(path.parent, assembly.sha(path))


@pytest.mark.parametrize('name', ['ubuntu:24.04', 'debian:trixie', 'ubuntu@sha256:no'])
def test_mutable_baselines_rejected(name):
    with pytest.raises(ValueError, match='pinned'):
        assembly.image(name)


def test_both_architectures_required_before_build(tmp_path):
    with patch.object(assembly, 'run') as run:
        with pytest.raises(ValueError, match='both'):
            assembly.assemble({'targets': {'host': {}}}, tmp_path/'bundle')
        run.assert_not_called()


def bundle(root):
    (root/'payload').write_text('original')
    (root/'bundle.json').write_text(json.dumps({'files': {'payload': assembly.sha(root/'payload')}}))


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'corrupt', 'symlink'])
def test_bundle_tamper_fails_before_docker(tmp_path, mutation):
    bundle(tmp_path)
    if mutation == 'missing':
        (tmp_path/'payload').unlink()
    elif mutation == 'extra':
        (tmp_path/'extra').touch()
    elif mutation == 'corrupt':
        (tmp_path/'payload').write_text('changed')
    else:
        (tmp_path/'alias').symlink_to(tmp_path/'payload')
    with patch.object(assembly, 'run') as run:
        with pytest.raises(ValueError):
            assembly.validate(tmp_path)
        run.assert_not_called()


def test_archive_bytes_are_reproducible(tmp_path):
    root = tmp_path/'bundle'
    root.mkdir()
    bundle(root)
    assembly.archive_bundle(root)
    first = (tmp_path/'bundle.tar.gz').read_bytes()
    assembly.archive_bundle(root)
    assert first == (tmp_path/'bundle.tar.gz').read_bytes()


def test_validation_always_disables_network_for_both_targets(tmp_path):
    for target, arch in [('host', 'amd64'), ('card', 'arm64')]:
        dest = tmp_path/target
        dest.mkdir()
        (dest/'target.json').write_text(json.dumps({'architecture': arch, 'image': 'fixture@sha256:'+'a'*64}))
    (tmp_path/'bundle.json').write_text(json.dumps({'files': {
        str(p.relative_to(tmp_path)): assembly.sha(p) for p in tmp_path.rglob('target.json')}}))
    with patch.object(assembly, 'run') as run:
        assembly.validate(tmp_path)
    assert run.call_count == 2
    for call in run.call_args_list:
        assert '--network=none' in call.args
        assert 'readonly' in call.args[call.args.index('--mount')+1]


def test_publication_list_includes_platform_source_archive(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'pcie_delivery', Path(__file__).resolve().parents[2] / 'pcie_host/scripts/pcie_delivery.py')
    delivery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(delivery)
    source = tmp_path/'platform-source.tar.gz'
    source.write_bytes(b'producer-owned archive')
    (tmp_path/'metadata.json').write_text(json.dumps({
        'resources': [source.name], 'resources-checksum': {source.name: assembly.sha(source)}}))
    assert source in delivery.publication_paths(tmp_path)
    source.unlink()
    with pytest.raises(ValueError, match='missing or changed'):
        delivery.publication_paths(tmp_path)


def test_publication_preserves_existing_path_prefixed_sidecars(tmp_path):
    spec = importlib.util.spec_from_file_location(
        'delivery_sidecars', Path(__file__).resolve().parents[2] / 'pcie_host/scripts/pcie_delivery.py')
    delivery = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(delivery)
    resource = tmp_path/'runtime.deb'
    resource.write_bytes(b'fixture')
    digest = assembly.sha(resource)
    (tmp_path/'metadata.json').write_text(json.dumps({
        'resources': [resource.name], 'resources-checksum': {resource.name: digest}}))
    sidecar = tmp_path/'runtime.deb.sha256'
    sidecar.write_text(digest+'  pcie_host/dist/runtime.deb\n')
    assert sidecar in delivery.publication_paths(tmp_path)
    sidecar.write_text('0'*64+'  pcie_host/dist/runtime.deb\n')
    with pytest.raises(ValueError, match='sidecar differs'):
        delivery.publication_paths(tmp_path)


def test_archive_deduplicates_and_preserves_card_transfer(tmp_path):
    import tarfile
    root = tmp_path/'bundle'
    paths = ('card/payload/runtime.deb', 'card/repo/runtime.deb',
             'card/wheels/pip.whl', 'host/wheels/pip.whl')
    for name in paths:
        path = root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b'native' if name.endswith('.deb') else b'wheel')
    (root/'bundle.json').write_text(json.dumps({'files': {
        name: assembly.sha(root/name) for name in paths}}))
    assembly.archive_bundle(root)
    archive_path = Path(str(root)+'.tar.gz')
    original = archive_path.read_bytes()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        links = [m for m in members if m.islnk()]
        assert len(links) == 2
        earlier = set()
        for member in members:
            if member.islnk():
                assert member.linkname in earlier
            earlier.add(member.name)
        archive.extractall(tmp_path/'unpacked', filter='data')
    extracted = tmp_path/'unpacked/neat-offline'
    assembly.verify_bundle(extracted)
    assert (extracted/paths[0]).stat().st_ino == (extracted/paths[1]).stat().st_ino
    # Production staging sends only card/ and common files. Cross-target links
    # must not leave a card transfer referring to omitted host files.
    with tarfile.open(tmp_path/'card.tar', 'w') as archive:
        archive.add(extracted/'bundle.json', arcname='bundle.json')
        archive.add(extracted/'card', arcname='card')
    with tarfile.open(tmp_path/'card.tar') as archive:
        archive.extractall(tmp_path/'card-only', filter='data')
    assembly.verify_bundle(tmp_path/'card-only', 'card')
    assembly.archive_bundle(extracted)
    assert Path(str(extracted)+'.tar.gz').read_bytes() == original
