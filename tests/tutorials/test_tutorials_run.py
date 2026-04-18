"""Smoke test: each Python tutorial exits 0 when run with the required --mpk.

Requires these env vars to actually exercise the tutorials (skipped otherwise):
  SIMA_NEAT_TUTORIAL_MPK_RESNET=/path/to/resnet_50_mpk.tar.gz
  SIMA_NEAT_TUTORIAL_MPK_YOLO=/path/to/yolo_v8s_mpk.tar.gz
  SIMA_NEAT_TUTORIAL_RTSP_URL=rtsp://host:port/stream   # chapter 019
"""
from __future__ import annotations

import os
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

# Which MPK each chapter needs. Keep in sync with the --mpk argparse calls
# in each tutorial; chapters not listed here do not take --mpk.
RESNET_CHAPTERS = {"001", "002", "013"}
YOLO_CHAPTERS = {"004", "005", "006", "007", "012", "015", "018", "019"}
RTSP_CHAPTERS = {"019"}


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
