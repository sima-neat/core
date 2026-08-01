from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_MODEL_PATH_CACHE: dict[str, Path | None] = {}

try:
  from _platform_version_generated import MODELZOO_VERSION as MODELZOO_PLATFORM_VERSION
except Exception:
  manifest = ROOT / "deps" / "manifest.json"
  try:
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    MODELZOO_PLATFORM_VERSION = (
        manifest_data.get("modelzoo-version")
        or manifest_data.get("platform-version")
        or "2.0.0"
    )
  except Exception:
    MODELZOO_PLATFORM_VERSION = "2.0.0"

MODEL_FIXTURES = {
    "SIMA_RESNET50_TAR": {
        "model_name": "resnet_50",
        "names": (
            "resnet_50_mpk.tar.gz",
            "resnet-50_mpk.tar.gz",
        ),
    },
    "SIMA_YOLO_TAR": {
        "model_name": "yolo_v8s",
        "names": (
            "yolo_v8s_mpk.tar.gz",
            "yolo-v8s_mpk.tar.gz",
            "yolov8s_mpk.tar.gz",
            "yolov8_s_mpk.tar.gz",
        ),
    },
}


def candidate_model_dirs() -> list[Path]:
  home = Path.home()
  return [
      ROOT / "tmp",
      ROOT / "assets" / "models",
      ROOT,
      Path.cwd() / "tmp",
      Path.cwd(),
      home / ".simaai",
      home / ".simaai" / "modelzoo",
      home / ".sima" / "modelzoo",
      Path("/data/simaai/modelzoo"),
  ]


def _existing_file(path: Path) -> Path | None:
  return path if path.is_file() and path.stat().st_size > 0 else None


def find_existing_model_tar(names: tuple[str, ...]) -> Path | None:
  for directory in candidate_model_dirs():
    for name in names:
      candidate = _existing_file(directory / name)
      if candidate is not None:
        return candidate
  return None


def resolve_model_tar(name: str) -> Path | None:
  if name in _MODEL_PATH_CACHE:
    return _MODEL_PATH_CACHE[name]

  spec = MODEL_FIXTURES[name]
  for env_name in (name, "SIMA_MODEL_TAR"):
    env_value = os.environ.get(env_name, "")
    if env_value:
      env_path = _existing_file(Path(env_value))
      if env_path is not None:
        _MODEL_PATH_CACHE[name] = env_path
        return env_path

  found = find_existing_model_tar(spec["names"])
  if found is not None:
    _MODEL_PATH_CACHE[name] = found
    return found

  sima_cli = shutil.which("sima-cli")
  if sima_cli is not None:
    result = subprocess.run(
        [sima_cli, "modelzoo", "-v", MODELZOO_PLATFORM_VERSION, "get", spec["model_name"]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
      found = find_existing_model_tar(spec["names"])
      if found is not None:
        _MODEL_PATH_CACHE[name] = found
        return found

  _MODEL_PATH_CACHE[name] = None
  return None


def fixture_model_path(name: str) -> Path:
  path = resolve_model_tar(name)
  if path is not None:
    return path

  spec = MODEL_FIXTURES[name]
  pytest.skip(
      f"missing real model fixture for {name}; set {name} or SIMA_MODEL_TAR, or run "
      f"'sima-cli modelzoo -v {MODELZOO_PLATFORM_VERSION} get {spec['model_name']}'"
  )


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
