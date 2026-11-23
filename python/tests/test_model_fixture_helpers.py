"""Host tests for how model_fixture_helpers resolves an explicitly given model path."""

from __future__ import annotations

import pytest

import model_fixture_helpers as model_fixtures


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
  model_fixtures._MODEL_PATH_CACHE.clear()
  monkeypatch.delenv("SIMA_RESNET50_TAR", raising=False)
  monkeypatch.delenv("SIMA_MODEL_TAR", raising=False)
  yield
  model_fixtures._MODEL_PATH_CACHE.clear()


def _mpk(path):
  path.write_bytes(b"mpk")
  return path


def test_fixture_variable_wins_over_the_generic_one(monkeypatch, tmp_path):
  resnet = _mpk(tmp_path / "resnet.tar.gz")
  monkeypatch.setenv("SIMA_RESNET50_TAR", str(resnet))
  monkeypatch.setenv("SIMA_MODEL_TAR", str(_mpk(tmp_path / "other.tar.gz")))
  assert model_fixtures.resolve_model_tar("SIMA_RESNET50_TAR") == resnet


def test_generic_variable_is_used_when_the_fixture_variable_is_unset(monkeypatch, tmp_path):
  generic = _mpk(tmp_path / "generic.tar.gz")
  monkeypatch.setenv("SIMA_MODEL_TAR", str(generic))
  assert model_fixtures.resolve_model_tar("SIMA_RESNET50_TAR") == generic


def test_a_missing_explicit_path_fails_and_names_the_variable(monkeypatch, tmp_path):
  monkeypatch.setenv("SIMA_RESNET50_TAR", str(tmp_path / "missing.tar.gz"))
  with pytest.raises(pytest.fail.Exception, match="SIMA_RESNET50_TAR=.*missing.tar.gz"):
    model_fixtures.resolve_model_tar("SIMA_RESNET50_TAR")


def test_an_empty_explicit_path_fails(monkeypatch, tmp_path):
  empty = tmp_path / "empty.tar.gz"
  empty.write_bytes(b"")
  monkeypatch.setenv("SIMA_MODEL_TAR", str(empty))
  with pytest.raises(pytest.fail.Exception, match="SIMA_MODEL_TAR="):
    model_fixtures.resolve_model_tar("SIMA_RESNET50_TAR")
