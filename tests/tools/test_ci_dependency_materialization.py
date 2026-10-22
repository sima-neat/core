#!/usr/bin/env python3
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import tarfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('materialization',ROOT/'tools/ci/materialize_dependencies.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def archive_bytes(name='usr/share/provider/receipt.json',kind=None,link=None):
    buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        entry=tarfile.TarInfo(name);data=b'HOST_TRANSPORT_FIXTURE_NOT_A_PACKAGE'
        if kind is not None:entry.type=kind;entry.linkname=link or '';entry.size=0;archive.addfile(entry)
        else:entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    return buffer.getvalue()


class MaterializationTests(unittest.TestCase):
    def manifest(self, digest='a'*64):
        return {'schema_version':1,'inputs':{key:dict(kind='package-bundle',url='https://example.invalid/'+key+'.tar.gz',sha256=digest)
                                            for key in m.COMPONENTS}}
    def test_manifest_pins_all_four_and_forbids_moving_sources(self):
        good=self.manifest();m.validate_manifest(good)
        for change in ('missing','http','short_sha','codec_source','source_latest'):
            with self.subTest(change=change):
                data=self.manifest()
                if change=='missing':del data['inputs']['codecs']
                if change=='http':data['inputs']['runner']['url']='http://example.invalid/r.tar.gz'
                if change=='short_sha':data['inputs']['runner']['sha256']='abc'
                if change=='codec_source':data['inputs']['codecs']=dict(kind='git',repository='https://example.invalid/codecs.git',commit='b'*40)
                if change=='source_latest':data['inputs']['kernels']=dict(kind='git',repository='https://example.invalid/kernels.git',commit='main')
                with self.assertRaises(ValueError):m.validate_manifest(data)
    def test_pinned_source_transport_is_explicit(self):
        data=self.manifest();data['inputs']['kernels']=dict(kind='git',repository='https://example.invalid/kernels.git',commit='b'*40)
        m.validate_manifest(data)
    def test_complete_materialization_env_and_receipt_without_network(self):
        import hashlib
        payload=archive_bytes();data=self.manifest(hashlib.sha256(payload).hexdigest())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'manifest.json';manifest.write_text(json.dumps(data))
            with mock.patch.object(m.urllib.request,'urlopen',side_effect=lambda *_a,**_k:io.BytesIO(payload)):
                env=m.materialize(manifest,root/'fresh')
            self.assertEqual(set(env),{'NEAT_KERNELS_PACKAGE_DIR','NEAT_RUNNER_PACKAGE_DIR','NEAT_SCHEDULER_PACKAGE_DIR','NEAT_CODECS_PACKAGE_DIR'})
            self.assertTrue(all(Path(value).is_dir() for value in env.values()))
            receipt=json.loads((root/'fresh/materialization-receipt.json').read_text())
            self.assertEqual(receipt['manifest_sha256'],m.digest(manifest))
            with self.assertRaises(ValueError):m.materialize(manifest,root/'fresh')
    def test_checksum_mismatch_does_not_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=Path(tmp)/'manifest.json';manifest.write_text(json.dumps(self.manifest()))
            with mock.patch.object(m.urllib.request,'urlopen',return_value=io.BytesIO(archive_bytes())):
                with self.assertRaises(ValueError):m.materialize(manifest,Path(tmp)/'fresh')
            self.assertFalse((Path(tmp)/'fresh/runner').exists())
            self.assertFalse((Path(tmp)/'fresh/materialization-receipt.json').exists())
    def test_tar_escape_link_and_special_members_rejected(self):
        for payload in (archive_bytes('../outside'),archive_bytes('/absolute'),
                        archive_bytes('evil',tarfile.SYMTYPE,'../../outside'),
                        archive_bytes('fifo',tarfile.FIFOTYPE)):
            with tempfile.TemporaryDirectory() as tmp:
                archive=Path(tmp)/'bundle.tar.gz';archive.write_bytes(payload)
                with self.assertRaises(ValueError):m.extract_bundle(archive,Path(tmp)/'new')
    def test_workflow_passes_portable_inputs_without_external_supervisor(self):
        workflow=(ROOT/'.github/workflows/vulcan-ci.yml').read_text()
        self.assertNotIn('NEAT_CI_DEPENDENCY_MANIFEST',workflow)
        self.assertIn('tools/ci/acquire_scheduler.py',workflow)
        self.assertIn('scripts/ci/prepare_scheduler_sdk.sh',workflow)
        self.assertNotIn('NEAT_GUARDED_HARDWARE_TESTS',workflow)
        self.assertNotIn('SIMA_HARDWARE_SUPERVISOR_SOCKET',workflow)
        self.assertNotIn('kernel-device-qualification-',workflow)
        self.assertNotIn('internals-migration',workflow)


if __name__=='__main__':unittest.main()
