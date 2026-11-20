"""The C++ tutorial wrapper selects the MPK the same way as the Python tutorial tests."""
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / 'tutorials/run_with_modelzoo.sh'
MODEL_VARIABLES = ('SIMA_NEAT_TUTORIAL_MODEL_RESNET', 'SIMA_NEAT_TUTORIAL_MODEL_YOLO')


@pytest.fixture
def program(tmp_path):
    path = tmp_path / 'tutorial'
    path.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    path.chmod(0o755)
    return path


def run(program, model, default, **variables):
    env = {key: value for key, value in os.environ.items() if key not in MODEL_VARIABLES}
    env.update(variables)
    return subprocess.run(['bash', str(WRAPPER), model, str(default), str(program),
                           '--samples', '10'], env=env, text=True, capture_output=True)


def mpk(path, data=b'mpk'):
    path.write_bytes(data)
    return path


def test_variable_selects_the_mpk_passed_as_model(tmp_path, program):
    selected = mpk(tmp_path / 'resnet_50_w8a8_mpk.tar.gz')
    result = run(program, 'resnet_50', tmp_path / 'missing.tar.gz',
                 SIMA_NEAT_TUTORIAL_MODEL_RESNET=str(selected))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['--model', str(selected), '--samples', '10']


def test_unset_variable_uses_the_registered_default(tmp_path, program):
    default = mpk(tmp_path / 'yolo_v8s.tar.gz')
    result = run(program, 'yolo_v8s', default,
                 SIMA_NEAT_TUTORIAL_MODEL_RESNET=str(tmp_path / 'other.tar.gz'))
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ['--model', str(default), '--samples', '10']


@pytest.mark.parametrize('data', [None, b''], ids=['missing', 'empty'])
def test_missing_or_empty_mpk_fails_and_names_the_variable(tmp_path, program, data):
    default = tmp_path / 'yolo_v8s.tar.gz'
    if data is not None:
        mpk(default, data)
    result = run(program, 'yolo_v8s', default)
    assert result.returncode == 1
    assert result.stdout == ''
    assert 'SIMA_NEAT_TUTORIAL_MODEL_YOLO' in result.stderr
    assert 'build.sh --download-models-only' in result.stderr


def test_variable_naming_a_missing_file_does_not_fall_back(tmp_path, program):
    default = mpk(tmp_path / 'resnet_50.tar.gz')
    result = run(program, 'resnet_50', default,
                 SIMA_NEAT_TUTORIAL_MODEL_RESNET=str(tmp_path / 'missing.tar.gz'))
    assert result.returncode == 1
    assert result.stdout == ''
    assert 'SIMA_NEAT_TUTORIAL_MODEL_RESNET' in result.stderr


def test_unknown_model_is_a_usage_error(tmp_path, program):
    result = run(program, 'mobilenet_v2', mpk(tmp_path / 'mobilenet_v2.tar.gz'))
    assert result.returncode == 2
    assert result.stdout == ''
