"""Unit tests for the tutorial asset-root lookup in tutorials/common/python_utils.py.

Mirrors the precedence documented in sima_tutorial::find_asset_root() (cpp):
  1. SIMA_NEAT_TUTORIAL_ASSETS env var, if set and exists.
  2. /usr/share/sima-neat/tutorials/assets (DEB install).
  3. /neat-resources/core-src/tutorials/assets (eLxr SDK install).
  4. <repo>/tutorials/assets (source checkout fallback).

These tests exercise each branch without requiring a real install layout —
all filesystem probes are monkeypatched.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_UTILS_PATH = REPO_ROOT / "tutorials" / "common" / "python_utils.py"

_DEB_INSTALL = Path("/usr/share/sima-neat/tutorials/assets")
_SDK_INSTALL = Path("/neat-resources/core-src/tutorials/assets")


@pytest.fixture(scope="module")
def tu():
  # Load the helper module directly from its file path so the test does not
  # depend on tutorials/common/ being on sys.path.
  spec = importlib.util.spec_from_file_location("_tutorial_python_utils_under_test", PYTHON_UTILS_PATH)
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  try:
    yield module
  finally:
    sys.modules.pop(spec.name, None)


def _install_paths_absent(path: Path) -> bool:
  # Force all three installed layouts (DEB + SDK + any env-provided path) to
  # report missing so the lookup falls through to later branches.
  return False


def test_env_var_points_to_existing_dir_wins(tu, tmp_path, monkeypatch):
  override = tmp_path / "custom-assets"
  override.mkdir()
  monkeypatch.setenv("SIMA_NEAT_TUTORIAL_ASSETS", str(override))

  assert tu.asset_root() == override


def test_env_var_points_to_missing_dir_falls_through(tu, tmp_path, monkeypatch):
  missing = tmp_path / "does-not-exist"
  monkeypatch.setenv("SIMA_NEAT_TUTORIAL_ASSETS", str(missing))
  # Force the install probes to miss so we can observe the fallback target
  # without relying on the host filesystem.
  monkeypatch.setattr(Path, "exists", _install_paths_absent)

  assert tu.asset_root() == tu.repo_root() / "tutorials" / "assets"


def test_deb_install_path_returned_when_present(tu, monkeypatch):
  monkeypatch.delenv("SIMA_NEAT_TUTORIAL_ASSETS", raising=False)
  # Only the DEB install path exists; SDK path and everything else is absent.
  monkeypatch.setattr(Path, "exists", lambda self: self == _DEB_INSTALL)

  assert tu.asset_root() == _DEB_INSTALL


def test_sdk_install_path_returned_when_present(tu, monkeypatch):
  monkeypatch.delenv("SIMA_NEAT_TUTORIAL_ASSETS", raising=False)
  # DEB path missing, SDK path present: the loop must advance past DEB.
  monkeypatch.setattr(Path, "exists", lambda self: self == _SDK_INSTALL)

  assert tu.asset_root() == _SDK_INSTALL


def test_source_fallback_when_nothing_installed(tu, monkeypatch):
  monkeypatch.delenv("SIMA_NEAT_TUTORIAL_ASSETS", raising=False)
  monkeypatch.setattr(Path, "exists", _install_paths_absent)

  assert tu.asset_root() == tu.repo_root() / "tutorials" / "assets"
