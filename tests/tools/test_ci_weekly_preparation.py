"""Preparation failures must not turn required weekly cases into skips."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('weekly', ROOT / 'tools/ci/prepare_weekly_fixtures.py')
weekly = importlib.util.module_from_spec(spec); spec.loader.exec_module(weekly)


def test_missing_sources_pins_and_streams_block(tmp_path):
    for env in ({}, {'SIMA_CI_MODEL': '/missing'}, {'SIMA_CI_MODEL': '/missing', 'SIMA_CI_MODEL_SHA256': 'bad'}):
        with pytest.raises(ValueError):
            weekly.stage('MODEL', tmp_path / 'new', env)
    with pytest.raises(ValueError, match='RTSP'):
        weekly.require_streams({})
    streams = {f'SIMANEAT_TEST_RTSP_{c}_URL': 'rtsp://fixture' for c in ('H264', 'H265', 'MJPEG')}
    streams.update(SIMANEAT_TEST_RTSP_MJPEG_FPS='30', SIMANEAT_TEST_HTTP_MJPEG_URL='http://fixture', SIMANEAT_TEST_HTTP_MJPEG_FPS='30')
    weekly.require_streams(streams)
    weekly.require_streams({'SIMANEAT_TEST_HTTP_MJPEG_URL': 'http://fixture', 'SIMANEAT_TEST_HTTP_MJPEG_FPS': '30'}, 'http')


def test_model_only_reads_authoritative_mpk(tmp_path):
    archive = tmp_path / 'model.tgz'
    with tarfile.open(archive, 'w:gz') as stream:
        for name, data in [('model/mpk.json', b'{"name":"fixture"}'), ('model/ignore.json', b'not json')]:
            member = tarfile.TarInfo(name); member.size = len(data)
            stream.addfile(member, io.BytesIO(data))
    env = {'SIMA_CI_MODEL': str(archive), 'SIMA_CI_MODEL_SHA256': hashlib.sha256(archive.read_bytes()).hexdigest()}
    assert weekly.stage('MODEL', tmp_path / 'staged.tgz', env, model=True).is_file()
    with pytest.raises(ValueError, match='existing'):
        weekly.stage('MODEL', tmp_path / 'staged.tgz', env, model=True)
    env['SIMA_CI_MODEL_SHA256'] = '0' * 64
    with pytest.raises(ValueError, match='checksum'):
        weekly.stage('MODEL', tmp_path / 'corrupt.tgz', env, model=True)


def test_camera_api_not_hardware_gated():
    text = (ROOT / 'tests/tutorials/test_mipi_camera_tutorial.py').read_text()
    assert 'pytestmark =' not in text
    assert text.count('@pytest.mark.mipi_camera') == 3
    assert '@_hardware_only\ndef test_installed_pyneat' not in text


def preparation_fixture(tmp_path, *, ssd=True):
    import gzip
    import struct
    env = {}
    def pinned(name, payload):
        path = tmp_path / name
        path.write_bytes(payload)
        env['SIMA_CI_' + name] = str(path)
        env['SIMA_CI_' + name + '_SHA256'] = hashlib.sha256(payload).hexdigest()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        member = tarfile.TarInfo('model/mpk.json'); member.size = 2
        archive.addfile(member, io.BytesIO(b'{}'))
    pinned('MNIST_MODEL', buf.getvalue())
    if ssd:
        pinned('SSD300_MODEL', buf.getvalue())
    pinned('MNIST_IMAGES', gzip.compress(struct.pack('>IIII', 2051, 100, 28, 28) + bytes(100 * 28 * 28)))
    pinned('MNIST_LABELS', gzip.compress(struct.pack('>II', 2049, 100) + bytes(range(10)) * 10))
    manifest = tmp_path / 'fixtures/runtime_manifest.json'
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({'source_root_rel': 'source'}))
    return env, manifest, tmp_path / 'source/tmp/mnist_preproc_channels1'


def test_default_requires_ssd_explicit_mnist_does_not(tmp_path):
    env, manifest, cache = preparation_fixture(tmp_path, ssd=False)
    with pytest.raises(ValueError, match='SSD300_MODEL'):
        weekly.prepare(tmp_path / 'all', manifest, env)
    values = weekly.prepare(tmp_path / 'mnist', manifest, env, groups=['mnist'])
    assert values['SIMA_RUN_MNIST_PREPROC_CHANNELS1'] == '1'
    assert 'SIMA_SSD300_TAR' not in values
    assert len(list(cache.iterdir())) == 4


def test_all_groups_and_exports_are_one_mapping(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    env, manifest, _ = preparation_fixture(tmp_path)
    monkeypatch.setattr(weekly.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('Unexpected network access'))
    values = weekly.prepare(tmp_path / 'all', manifest, env)
    github = tmp_path / 'github-env'
    local = tmp_path / 'environment.json'
    weekly.export_environment(values, github_env=github, environment_json=local)
    assert dict(line.split('=', 1) for line in github.read_text().splitlines()) == json.loads(local.read_text()) == values
    child_env = {**os.environ, **json.loads(local.read_text())}
    subprocess.run([sys.executable, '-c', "import os; assert os.environ['SIMA_RUN_MNIST_PREPROC_CHANNELS1'] == '1'"], env=child_env, check=True)
    assert 'SIMA_SSD300_TAR' in values
    # Exact, already-populated gzip and raw caches are reusable without download.
    weekly.prepare(tmp_path / 'again', manifest, env)


@pytest.mark.parametrize('kind', ['gzip_crc', 'raw_cache', 'gzip_cache', 'model', 'labels', 'symlink_cache'])
def test_corrupt_fixture_never_exports_optin(tmp_path, kind):
    import gzip
    import struct
    import subprocess
    import sys
    env, manifest, cache = preparation_fixture(tmp_path)
    if kind in ('raw_cache', 'gzip_cache', 'symlink_cache'):
        cache.mkdir(parents=True)
        target = cache / ('t10k-images-idx3-ubyte.gz' if kind == 'gzip_cache' else 't10k-images-idx3-ubyte')
        if kind == 'symlink_cache':
            target.symlink_to(tmp_path / 'elsewhere')
        else:
            target.write_bytes(b'other campaign')
    else:
        key = 'MNIST_MODEL' if kind == 'model' else ('MNIST_LABELS' if kind == 'labels' else 'MNIST_IMAGES')
        path = Path(env['SIMA_CI_' + key])
        payload = (b'not an archive' if kind == 'model' else
                   gzip.compress(struct.pack('>II', 2049, 100) + bytes([11]) * 100) if kind == 'labels' else
                   path.read_bytes()[:-8] + bytes(8))
        path.write_bytes(payload)
        env['SIMA_CI_' + key + '_SHA256'] = hashlib.sha256(payload).hexdigest()
    github = tmp_path / 'github-env'; local = tmp_path / 'environment.json'
    import os
    result = subprocess.run([sys.executable, str(ROOT / 'tools/ci/prepare_weekly_fixtures.py'),
        '--output', str(tmp_path / 'out'), '--runtime-manifest', str(manifest),
        '--github-env', str(github), '--environment-json', str(local)], env={**os.environ, **env}, capture_output=True, text=True)
    assert result.returncode != 0
    assert not github.exists() and not local.exists()


@pytest.mark.parametrize('groups', [[], ['mnist', 'mnist'], ['unknown']])
def test_invalid_selection_rejected_before_staging(tmp_path, groups):
    with pytest.raises(ValueError, match='fixture group'):
        weekly.prepare(tmp_path / 'out', '/missing', {}, groups=groups)
    assert not (tmp_path / 'out').exists()


def test_export_rejects_newline_before_any_output(tmp_path):
    github = tmp_path / 'github-env'; local = tmp_path / 'environment.json'
    with pytest.raises(ValueError, match='environment value'):
        weekly.export_environment({'SIMA_RUN_MNIST_PREPROC_CHANNELS1': '1', 'PATH': 'bad\nvalue'},
                                  github_env=github, environment_json=local)
    assert not github.exists() and not local.exists()
