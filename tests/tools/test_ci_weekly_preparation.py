"""Preparation failures must not turn required weekly cases into skips."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('weekly', ROOT / 'tools/ci/prepare_weekly_fixtures.py')
weekly = importlib.util.module_from_spec(spec); spec.loader.exec_module(weekly)

HTTP = {'SIMANEAT_TEST_HTTP_MJPEG_URL': 'http://fixture', 'SIMANEAT_TEST_HTTP_MJPEG_FPS': '30'}


def test_missing_streams_block():
    with pytest.raises(ValueError, match='RTSP'):
        weekly.require_streams({})
    streams = {f'SIMANEAT_TEST_RTSP_{c}_URL': 'rtsp://fixture' for c in ('H264', 'H265', 'MJPEG')}
    streams.update(SIMANEAT_TEST_RTSP_MJPEG_FPS='30', **HTTP)
    weekly.require_streams(streams)
    weekly.require_streams(HTTP, 'http')
    weekly.require_streams({}, 'none')
    with pytest.raises(ValueError, match='FPS'):
        weekly.require_streams({**HTTP, 'SIMANEAT_TEST_HTTP_MJPEG_FPS': '0'}, 'http')


def test_camera_api_not_hardware_gated():
    text = (ROOT / 'tests/tutorials/test_mipi_camera_tutorial.py').read_text()
    assert 'pytestmark =' not in text
    assert text.count('@pytest.mark.mipi_camera') == 3
    assert '@_hardware_only\ndef test_installed_pyneat' not in text


def run_script(tmp_path, profile, env):
    report = tmp_path / 'report.json'
    result = subprocess.run([sys.executable, str(ROOT / 'tools/ci/prepare_weekly_fixtures.py'),
                             '--stream-profile', profile, '--stream-report', str(report)],
                            env={**{k: v for k, v in os.environ.items() if not k.startswith('SIMANEAT_TEST_')},
                                 **env},
                            capture_output=True, text=True)
    return result, report


def test_stream_report_names_the_excluded_transports(tmp_path):
    result, report = run_script(tmp_path, 'http', HTTP)
    assert result.returncode == 0, result.stderr
    assert json.loads(report.read_text())['excluded_not_qualified'] == ['RTSP H264/H265/MJPEG']


def test_missing_stream_blocks_without_a_report(tmp_path):
    result, report = run_script(tmp_path, 'all', HTTP)
    assert result.returncode != 0
    assert 'PREPARATION_BLOCKED' in result.stderr
    assert not report.exists()
