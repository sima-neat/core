"""Smoke test: each Python tutorial exits 0 when run with the required --mpk.

Requires these env vars to actually exercise the tutorials (skipped otherwise):
  SIMA_NEAT_TUTORIAL_MPK_RESNET=/path/to/resnet_50_mpk.tar.gz
  SIMA_NEAT_TUTORIAL_MPK_YOLO=/path/to/yolo_v8s_mpk.tar.gz
  SIMA_NEAT_TUTORIAL_RTSP_URL=rtsp://host:port/stream   # chapter 017
"""
from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS_ROOT = REPO_ROOT / "tutorials"
TIMEOUT_SEC = int(os.environ.get("SIMA_TUTORIAL_TIMEOUT_SEC", "180"))

MPK_RESNET = os.environ.get("SIMA_NEAT_TUTORIAL_MPK_RESNET")
MPK_YOLO = os.environ.get("SIMA_NEAT_TUTORIAL_MPK_YOLO")
RTSP_URL = os.environ.get("SIMA_NEAT_TUTORIAL_RTSP_URL")
PYNEAT_AVAILABLE = importlib.util.find_spec("pyneat") is not None

# Which MPK each chapter needs. Keep in sync with the --mpk argparse calls
# in each tutorial; chapters not listed here do not take --mpk.
RESNET_CHAPTERS = {"001", "002", "005", "016"}
YOLO_CHAPTERS = {"004", "006", "007", "013"}
RTSP_CHAPTERS = {"017"}


def _chapter_id(folder: str) -> str:
  return folder[:3]


def _mpk_for(folder: str) -> str | None:
  tid = _chapter_id(folder)
  if tid in RESNET_CHAPTERS:
    return MPK_RESNET
  if tid in YOLO_CHAPTERS:
    return MPK_YOLO
  return None  # no MPK needed


def _tutorial_py_files() -> list[tuple[str, Path]]:
  out: list[tuple[str, Path]] = []
  for d in sorted(TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*/")):
    py = next(d.glob("*.py"), None)
    if py:
      out.append((d.name, py))
  return out


def _readme_models() -> dict[str, str]:
  models: dict[str, str] = {}
  for d in sorted(TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*/")):
    readme = d / "README.md"
    if not readme.exists():
      continue
    match = re.search(r"^\|\s*Model\s*\|\s*([^|]+?)\s*\|", readme.read_text(), re.MULTILINE)
    assert match, f"{readme} does not declare '| Model | ... |' metadata"
    models[_chapter_id(d.name)] = match.group(1).strip()
  return models


def _cmake_models() -> dict[str, str]:
  cmake = (TUTORIALS_ROOT / "CMakeLists.txt").read_text()
  models: dict[str, str] = {}
  for match in re.finditer(
      r"add_tutorial\(\s*tutorial_(\d{3})_[^\s]+\s+.*?\)(?=\n\n|$)",
      cmake,
      re.DOTALL,
  ):
    chapter = match.group(1)
    model_match = re.search(r"^\s*MODEL\s+([^\s)]+)", match.group(0), re.MULTILINE)
    if model_match:
      models[chapter] = model_match.group(1)
  return models


def test_tutorial_model_metadata_matches_cmake_and_smoke_selection() -> None:
  readme_models = _readme_models()
  cmake_models = _cmake_models()
  selected_models = {
      **{chapter: "resnet_50" for chapter in RESNET_CHAPTERS},
      **{chapter: "yolo_v8s" for chapter in YOLO_CHAPTERS},
  }

  for chapter, model in readme_models.items():
    if model == "None":
      assert chapter not in cmake_models, (
          f"chapter {chapter} has README Model None but CMake MODEL "
          f"{cmake_models[chapter]}"
      )
      assert chapter not in selected_models, (
          f"chapter {chapter} has README Model None but smoke test selects "
          f"{selected_models[chapter]}"
      )
      continue

    assert cmake_models.get(chapter) == model, (
        f"chapter {chapter} README Model {model} does not match "
        f"CMake MODEL {cmake_models.get(chapter)}"
    )
    assert selected_models.get(chapter) == model, (
        f"chapter {chapter} README Model {model} does not match "
        f"smoke-test model selection {selected_models.get(chapter)}"
    )


def test_python_usage_comments_use_current_script_names() -> None:
  for _, py_path in _tutorial_py_files():
    match = re.search(r"^\s*python3\s+([^\s]+)", py_path.read_text(), re.MULTILINE)
    assert match, f"{py_path} does not include a python3 usage line"
    assert match.group(1) == py_path.name, (
        f"{py_path} usage references {match.group(1)}, expected {py_path.name}"
    )


@pytest.mark.parametrize(
    "folder,py_path",
    _tutorial_py_files(),
    ids=lambda v: v if isinstance(v, str) else v.name,
)
def test_tutorial_runs(folder: str, py_path: Path) -> None:
  tid = _chapter_id(folder)
  needs_mpk = tid in RESNET_CHAPTERS | YOLO_CHAPTERS
  needs_rtsp = tid in RTSP_CHAPTERS
  mpk = _mpk_for(folder)

  if not PYNEAT_AVAILABLE:
    pytest.skip("pyneat is not importable in this Python environment")
  if needs_mpk and not mpk:
    pytest.skip(f"set SIMA_NEAT_TUTORIAL_MPK_RESNET / MPK_YOLO to run {folder}")
  if needs_rtsp and not RTSP_URL:
    pytest.skip(f"set SIMA_NEAT_TUTORIAL_RTSP_URL to run {folder}")

  cmd = [sys.executable, str(py_path)]
  if mpk:
    cmd += ["--mpk", mpk]
  if needs_rtsp:
    cmd += ["--url", RTSP_URL]

  r = subprocess.run(
      cmd,
      cwd=str(REPO_ROOT),
      capture_output=True,
      text=True,
      timeout=TIMEOUT_SEC,
  )
  assert r.returncode == 0, (
      f"{folder} exited {r.returncode}\n"
      f"--- STDOUT ---\n{r.stdout}\n"
      f"--- STDERR ---\n{r.stderr}"
  )
