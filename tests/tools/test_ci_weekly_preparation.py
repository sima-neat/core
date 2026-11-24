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


def preparation_fixture(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode='w:gz') as archive:
        member = tarfile.TarInfo('model/mpk.json'); member.size = 2
        archive.addfile(member, io.BytesIO(b'{}'))
    path = tmp_path / 'SSD300_MODEL'
    path.write_bytes(buf.getvalue())
    return {'SIMA_CI_SSD300_MODEL': str(path),
            'SIMA_CI_SSD300_MODEL_SHA256': hashlib.sha256(buf.getvalue()).hexdigest()}


def test_default_requires_ssd(tmp_path):
    with pytest.raises(ValueError, match='SSD300_MODEL'):
        weekly.prepare(tmp_path / 'all', {})


def test_all_groups_and_exports_are_one_mapping(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    env = preparation_fixture(tmp_path)
    monkeypatch.setattr(weekly.urllib.request, 'urlopen', lambda *a, **kw: pytest.fail('Unexpected network access'))
    values = weekly.prepare(tmp_path / 'all', env)
    github = tmp_path / 'github-env'
    local = tmp_path / 'environment.json'
    weekly.export_environment(values, github_env=github, environment_json=local)
    assert dict(line.split('=', 1) for line in github.read_text().splitlines()) == json.loads(local.read_text()) == values
    child_env = {**os.environ, **json.loads(local.read_text())}
    subprocess.run([sys.executable, '-c', "import os, pathlib; assert pathlib.Path(os.environ['SIMA_SSD300_TAR']).is_file()"],
                   env=child_env, check=True)


@pytest.mark.parametrize('kind', ['model', 'checksum'])
def test_corrupt_fixture_never_exports_optin(tmp_path, kind):
    import os
    import subprocess
    import sys
    env = preparation_fixture(tmp_path)
    if kind == 'model':
        payload = b'not an archive'
        Path(env['SIMA_CI_SSD300_MODEL']).write_bytes(payload)
        env['SIMA_CI_SSD300_MODEL_SHA256'] = hashlib.sha256(payload).hexdigest()
    else:
        env['SIMA_CI_SSD300_MODEL_SHA256'] = '0' * 64
    github = tmp_path / 'github-env'; local = tmp_path / 'environment.json'
    result = subprocess.run([sys.executable, str(ROOT / 'tools/ci/prepare_weekly_fixtures.py'),
        '--output', str(tmp_path / 'out'),
        '--github-env', str(github), '--environment-json', str(local)], env={**os.environ, **env}, capture_output=True, text=True)
    assert result.returncode != 0
    assert not github.exists() and not local.exists()


@pytest.mark.parametrize('groups', [[], ['ssd300', 'ssd300'], ['unknown']])
def test_invalid_selection_rejected_before_staging(tmp_path, groups):
    with pytest.raises(ValueError, match='fixture group'):
        weekly.prepare(tmp_path / 'out', {}, groups=groups)
    assert not (tmp_path / 'out').exists()


def test_export_rejects_newline_before_any_output(tmp_path):
    github = tmp_path / 'github-env'; local = tmp_path / 'environment.json'
    with pytest.raises(ValueError, match='environment value'):
        weekly.export_environment({'SIMA_SSD300_TAR': '/fixture', 'PATH': 'bad\nvalue'},
                                  github_env=github, environment_json=local)
    assert not github.exists() and not local.exists()
