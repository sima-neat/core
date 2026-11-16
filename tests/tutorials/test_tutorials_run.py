"""Smoke test: each CTest-enabled Python tutorial exits 0.

Requires these env vars to actually exercise the tutorials (skipped otherwise):
  SIMA_NEAT_TUTORIAL_MODEL_RESNET=/path/to/resnet_50.tar.gz
  SIMA_NEAT_TUTORIAL_MODEL_YOLO=/path/to/yolo_v8s.tar.gz
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
REQUIRE_RUNTIME = os.environ.get("SIMA_NEAT_TUTORIAL_REQUIRE_RUNTIME") == "1"

MODEL_RESNET = os.environ.get("SIMA_NEAT_TUTORIAL_MODEL_RESNET")
MODEL_YOLO = os.environ.get("SIMA_NEAT_TUTORIAL_MODEL_YOLO")
PYNEAT_AVAILABLE = importlib.util.find_spec("pyneat") is not None

MODEL_ENV_BY_NAME = {
    "resnet_50": "SIMA_NEAT_TUTORIAL_MODEL_RESNET",
    "yolo_v8s": "SIMA_NEAT_TUTORIAL_MODEL_YOLO",
}
MODEL_PATH_BY_NAME = {
    "resnet_50": MODEL_RESNET,
    "yolo_v8s": MODEL_YOLO,
}
RTSP_URL = os.environ.get("SIMANEAT_TEST_RTSP_H264_URL")
if not RTSP_URL:
  rtsp_urls = os.environ.get("SIMANEAT_TEST_RTSP_H264_URLS", "")
  RTSP_URL = re.split(r"[ ,;]+", rtsp_urls.strip(), maxsplit=1)[0] if rtsp_urls.strip() else None
SMOKE_ARGS_BY_CHAPTER = {
    "003": ["--samples", "10"],
}
EXTRA_RUNTIME_CHAPTERS = {"018"}


def _chapter_id(folder: str) -> str:
  return folder[:3]


def _tutorial_script_path(folder: Path) -> Path:
  return folder / f"{folder.name[4:]}.py"


def _tutorial_py_files() -> list[tuple[str, Path]]:
  out: list[tuple[str, Path]] = []
  for d in sorted(TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*/")):
    py = _tutorial_script_path(d)
    if py.exists():
      out.append((d.name, py))
  return out


def _ctested_tutorial_py_files() -> list[tuple[str, Path]]:
  ctested_chapters = _ctested_chapters()
  return [
      (folder, path)
      for folder, path in _tutorial_py_files()
      if _chapter_id(folder) in ctested_chapters or _chapter_id(folder) in EXTRA_RUNTIME_CHAPTERS
  ]


def _skip_or_fail(reason: str) -> None:
  if REQUIRE_RUNTIME:
    pytest.fail(reason)
  pytest.skip(reason)


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


def _cmake_tutorial_blocks() -> dict[str, str]:
  cmake = (TUTORIALS_ROOT / "CMakeLists.txt").read_text()
  return {
      match.group(1): match.group(0)
      for match in re.finditer(
          r"add_tutorial\(\s*tutorial_(\d{3})_[^\s]+\s+.*?\)(?=\n\n|$)",
          cmake,
          re.DOTALL,
      )
  }


def _ctested_chapters() -> set[str]:
  return {
      chapter
      for chapter, block in _cmake_tutorial_blocks().items()
      if not re.search(r"^\s*NO_TEST\b", block, re.MULTILINE)
  }


def _cmake_models() -> dict[str, str]:
  models: dict[str, str] = {}
  for chapter, block in _cmake_tutorial_blocks().items():
    model_match = re.search(r"^\s*MODEL\s+([^\s)]+)", block, re.MULTILINE)
    if model_match:
      models[chapter] = model_match.group(1)
  return models


def test_tutorial_model_metadata_matches_cmake_and_smoke_selection() -> None:
  readme_models = _readme_models()
  cmake_models = _cmake_models()
  ctested_chapters = _ctested_chapters()

  for chapter, model in readme_models.items():
    if model == "None":
      assert chapter not in cmake_models, (
          f"chapter {chapter} has README Model None but CMake MODEL "
          f"{cmake_models[chapter]}"
      )
      continue

    if chapter not in ctested_chapters:
      continue

    assert model in MODEL_ENV_BY_NAME, (
        f"chapter {chapter} uses CTest model {model}, but the Python smoke "
        "test has no model env mapping"
    )
    assert cmake_models.get(chapter) == model, (
        f"chapter {chapter} README Model {model} does not match "
        f"CMake MODEL {cmake_models.get(chapter)}"
    )


_STEP_ANCHOR_RE = re.compile(r"\{#step-([a-z0-9][a-z0-9_-]*)\}")
_STEP_MARKER_RE = re.compile(r"^\s*(?://|#)\s*STEP\s+([a-z0-9][a-z0-9_-]*)\s*$", re.MULTILINE)
_STEP_END_RE = re.compile(r"^\s*(?://|#)\s*END STEP(?:\s+[a-z0-9_-]+)?\s*$", re.MULTILINE)


def _walkthrough_step_names(readme_text: str) -> list[str]:
  """The {#step:<name>} anchors declared in a README's Walkthrough section."""
  return _STEP_ANCHOR_RE.findall(readme_text)


def test_walkthrough_segments_pair_across_languages() -> None:
  """For every tutorial with a `## Walkthrough`, each `{#step:<name>}` prose
  anchor must have a matching `STEP <name>` region in BOTH the .cpp and .py,
  with balanced END markers and no duplicate names. Comment-only markers, so
  this never affects whether the program runs."""
  for d in sorted(TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*/")):
    readme = d / "README.md"
    if not readme.exists():
      continue
    readme_text = readme.read_text()
    if not re.search(r"^##\s+Walkthrough\s*$", readme_text, re.MULTILINE):
      continue

    anchors = _walkthrough_step_names(readme_text)
    assert anchors, f"{readme} has a Walkthrough but no {{#step:<name>}} anchors"

    for src in (next(d.glob("*.cpp"), None), _tutorial_script_path(d)):
      assert src is not None, f"{d.name}: missing a .cpp/.py source"
      assert src.exists(), f"{d.name}: missing source {src}"
      text = src.read_text()
      starts = _STEP_MARKER_RE.findall(text)
      assert len(starts) == len(set(starts)), (
          f"{src}: duplicate STEP names {sorted(set(s for s in starts if starts.count(s) > 1))}"
      )
      assert len(starts) == len(_STEP_END_RE.findall(text)), (
          f"{src}: unbalanced STEP / END STEP markers"
      )
      missing = [name for name in anchors if name not in set(starts)]
      assert not missing, (
          f"{src}: walkthrough steps {missing} have no matching STEP marker"
      )


_LLIMA_DEFAULT_MODELS_DIR = "/media/nvme/llima/models"
_FENCED_CODE_RE = re.compile(r"^```[^\n]*\n(.*?)^```", re.MULTILINE | re.DOTALL)
# The only command allowed to name the NVMe default: assigning it to the variable.
_LLIMA_MODELS_PATH_EXPORT_RE = re.compile(r"^#?\s*export\s+LLIMA_MODELS_PATH=")


def _hardcoded_llima_model_paths(readme_text: str) -> list[str]:
  """Command lines in fenced code blocks that hardcode the NVMe models dir,
  whether the invocation is on one line or split with `\\` continuations."""
  return [
      line.strip()
      for block in _FENCED_CODE_RE.findall(readme_text)
      for line in block.splitlines()
      if _LLIMA_DEFAULT_MODELS_DIR in line
      and not _LLIMA_MODELS_PATH_EXPORT_RE.match(line.strip())
  ]


def test_hardcoded_llima_model_paths_detects_single_and_multi_line_commands() -> None:
  readme = (
      "Prose may mention `/media/nvme/llima/models`.\n\n"
      "```bash\n"
      "export LLIMA_MODELS_PATH=/media/nvme/llima/models\n"
      "python3 example.py --model /media/nvme/llima/models/foo\n"
      "./tutorial \\\n"
      "  --model /media/nvme/llima/models/bar\n"
      "./tutorial --model \"$LLIMA_MODELS_PATH/baz\"\n"
      "```\n"
  )
  assert _hardcoded_llima_model_paths(readme) == [
      "python3 example.py --model /media/nvme/llima/models/foo",
      "--model /media/nvme/llima/models/bar",
  ]


def test_llima_tutorials_document_models_path_without_nvme() -> None:
  """Tutorials that `llima pull` models must work on a DevKit without NVMe:
  they explain `LLIMA_MODELS_PATH` and pass model directories through it
  instead of hardcoding the NVMe default (issue #1347)."""
  readmes = sorted(TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*/README.md")) + sorted(
      TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*/i18n/*/README.md")
  )
  checked = 0
  for readme in readmes:
    text = readme.read_text(encoding="utf-8")
    if not re.search(r"^llima pull\s", text, re.MULTILINE):
      continue
    checked += 1
    assert "export LLIMA_MODELS_PATH=" in text, (
        f"{readme} pulls LLiMa models but does not show how to set LLIMA_MODELS_PATH"
    )
    hardcoded = _hardcoded_llima_model_paths(text)
    assert not hardcoded, (
        f"{readme} hardcodes {_LLIMA_DEFAULT_MODELS_DIR} in commands; "
        f"use \"$LLIMA_MODELS_PATH/<model>\" instead: {hardcoded}"
    )
  assert checked, "expected at least one tutorial README that runs `llima pull`"


def test_ctested_python_usage_comments_use_current_script_names() -> None:
  ctested_chapters = _ctested_chapters()
  for _, py_path in _tutorial_py_files():
    if _chapter_id(py_path.parent.name) not in ctested_chapters:
      continue
    match = re.search(r"^\s*python3\s+([^\s]+)", py_path.read_text(), re.MULTILINE)
    assert match, f"{py_path} does not include a python3 usage line"
    assert match.group(1) == py_path.name, (
        f"{py_path} usage references {match.group(1)}, expected {py_path.name}"
    )


@pytest.mark.parametrize(
    "folder,py_path",
    _ctested_tutorial_py_files(),
    ids=lambda v: v if isinstance(v, str) else v.name,
)
def test_tutorial_runs(folder: str, py_path: Path) -> None:
  tid = _chapter_id(folder)
  model_name = _cmake_models().get(tid)
  needs_model = model_name is not None
  model_path = MODEL_PATH_BY_NAME.get(model_name) if needs_model else None

  if not PYNEAT_AVAILABLE:
    _skip_or_fail("pyneat is not importable in this Python environment")
  if needs_model and model_name not in MODEL_ENV_BY_NAME:
    _skip_or_fail(f"{folder} uses unsupported smoke-test model {model_name}")
  if needs_model and not model_path:
    _skip_or_fail(f"set {MODEL_ENV_BY_NAME[model_name]} to run {folder}")
  if tid == "018" and not RTSP_URL:
    _skip_or_fail("set SIMANEAT_TEST_RTSP_H264_URL or SIMANEAT_TEST_RTSP_H264_URLS to run 018")

  cmd = [sys.executable, str(py_path)]
  if model_path:
    cmd += ["--model", model_path]
  if tid == "018":
    cmd += ["--url", RTSP_URL, "--frames", "10"]
  cmd += SMOKE_ARGS_BY_CHAPTER.get(tid, [])

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
  if tid == "018":
    assert "rtsp_timeout" not in r.stdout, (
        f"{folder} timed out while pulling RTSP frames\n"
        f"--- STDOUT ---\n{r.stdout}\n"
        f"--- STDERR ---\n{r.stderr}"
    )
    shape_count = len(re.findall(r"^frame=\d+ shape=", r.stdout, re.MULTILINE))
    assert shape_count == 10, (
        f"{folder} expected 10 decoded frames, got {shape_count}\n"
        f"--- STDOUT ---\n{r.stdout}\n"
        f"--- STDERR ---\n{r.stderr}"
    )
