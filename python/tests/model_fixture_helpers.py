from __future__ import annotations

import os
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH_CACHE: dict[str, Path] = {}

# CI prepares these Model Browser packages before the tests start and sets the variables.
MODEL_FIXTURES = {
    "SIMA_RESNET50_TAR": ("resnet_50", "w8a8"),
    "SIMA_YOLO_TAR": ("yolov8s", "w8a8"),
}


def _existing_file(path: Path) -> Path | None:
  return path if path.is_file() and path.stat().st_size > 0 else None


def resolve_model_tar(name: str) -> Path:
  """Return the model archive named by `name`, else by SIMA_MODEL_TAR. Never downloads."""
  if name in _MODEL_PATH_CACHE:
    return _MODEL_PATH_CACHE[name]

  for env_name in (name, "SIMA_MODEL_TAR"):
    env_value = os.environ.get(env_name, "")
    if env_value:
      env_path = _existing_file(Path(env_value))
      if env_path is None:
        pytest.fail(f"{env_name}={env_value} is not a non-empty file")
      _MODEL_PATH_CACHE[name] = env_path
      return env_path

  model_id, variant = MODEL_FIXTURES[name]
  pytest.fail(
      f"{name} is not set. Set it, or SIMA_MODEL_TAR, to the {model_id}_{variant}_mpk.tar.gz "
      f"from 'sima-cli models download --id {model_id} --variant {variant}'."
  )


def fixture_model_path(name: str) -> Path:
  return resolve_model_tar(name)


def has_strict_mpk_json(path: Path) -> bool:
  try:
    with tarfile.open(path, "r:gz") as tar:
      return any(Path(member.name).name.endswith("_mpk.json") for member in tar.getmembers())
  except tarfile.TarError:
    return False


def strict_model_tar_path(name: str) -> Path:
  path = fixture_model_path(name)
  if not has_strict_mpk_json(path):
    pytest.fail(f"real model fixture lacks required *_mpk.json strict contract: {path}")
  return path
