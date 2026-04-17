"""Shared helpers for v2 tutorials (code-only, no doc dependencies)."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence


def repo_root() -> Path:
  here = Path(__file__).resolve()
  # tutorials/common/python_utils.py -> repo root is parents[2]
  return here.parents[2]


def asset_root() -> Path:
  """Directory containing tutorial sample assets.

  Mirrors sima_tutorial::find_asset_root() in tutorial_common.h. Lookup order:
    1. SIMA_NEAT_TUTORIAL_ASSETS env var, if set and exists.
    2. /usr/share/sima-neat/tutorials/assets (installed DEB layout).
    3. /neat-resources/core-src/tutorials/assets (eLxr SDK layout).
    4. <repo>/tutorials/assets (source checkout fallback).
  """
  env = os.environ.get("SIMA_NEAT_TUTORIAL_ASSETS")
  if env:
    p = Path(env)
    if p.exists():
      return p
  for p in (
      Path("/usr/share/sima-neat/tutorials/assets"),
      Path("/neat-resources/core-src/tutorials/assets"),
  ):
    if p.exists():
      return p
  return repo_root() / "tutorials" / "assets"


def has_flag(argv: list[str], key: str) -> bool:
  return key in argv


def get_arg(argv: list[str], key: str, default: Optional[str] = None) -> Optional[str]:
  for i in range(1, len(argv) - 1):
    if argv[i] == key:
      return argv[i + 1]
  return default


def parse_int(argv: list[str], key: str, default: int) -> int:
  raw = get_arg(argv, key)
  if raw is None:
    return default
  try:
    return int(raw)
  except Exception as exc:
    raise ValueError(f"invalid integer for {key}: {raw}") from exc


def parse_float(argv: list[str], key: str, default: float) -> float:
  raw = get_arg(argv, key)
  if raw is None:
    return default
  try:
    return float(raw)
  except Exception as exc:
    raise ValueError(f"invalid float for {key}: {raw}") from exc


def strict_mode() -> bool:
  return os.getenv("SIMA_RUN_TUTORIALS_FULL") is not None


def first_existing(paths: Iterable[Path]) -> Optional[Path]:
  for path in paths:
    if path.exists():
      return path
  return None


def default_yolo_mpk(root: Path) -> Optional[Path]:
  return first_existing(
      [
          root / "tmp" / "yolo_v8s_mpk.tar.gz",
          root / "tmp" / "yolov8s_mpk.tar.gz",
          root / "tmp" / "yolo_mpk.tar.gz",
      ]
  )


def default_resnet_mpk(root: Path) -> Optional[Path]:
  return first_existing(
      [
          root / "tmp" / "resnet_50_mpk.tar.gz",
          root / "tmp" / "resnet50_mpk.tar.gz",
      ]
  )


def default_image(root: Path) -> Optional[Path]:  # pylint: disable=unused-argument
  # `root` is ignored and retained only for backward compatibility with existing
  # callers. Lookup now flows through asset_root() so DEB/SDK installs resolve
  # the same way the cpp helper does (see tutorials/common/cpp_utils.h).
  return asset_root() / "ilena_488.jpg"


def skip(reason: str) -> int:
  print(f"SKIP: {reason}")
  return 0


def ensure(cond: bool, message: str) -> None:
  if not cond:
    raise RuntimeError(message)


def step(name: str, detail: str = "") -> None:
  if detail:
    print(f"STEP {name}: {detail}")
  else:
    print(f"STEP {name}")


def _emit(tag: str, detail: str = "") -> None:
  if detail:
    print(f"{tag} {detail}")
  else:
    print(tag)


def why(detail: str) -> None:
  _emit("WHY", detail)


def tradeoff(detail: str) -> None:
  _emit("TRADEOFF", detail)


def failure_mode(detail: str) -> None:
  _emit("FAILURE_MODE", detail)


def interpret_output(detail: str) -> None:
  _emit("INTERPRET", detail)


def runtime_fallback(exc: BaseException) -> None:
  msg = str(exc).strip()
  if not msg:
    msg = exc.__class__.__name__
  print(f"runtime_fallback: {msg}")


def check(name: str, cond: bool, detail: str = "") -> None:
  status = "PASS" if cond else "FAIL"
  suffix = f" ({detail})" if detail else ""
  print(f"CHECK {name}: {status}{suffix}")
  if not cond:
    raise RuntimeError(f"check failed: {name}")


_REQUIRED_SIGNATURE_KEYS = (
    "tutorial",
    "lang",
    "flow",
    "run_mode",
    "output_kind",
    "tensor_rank",
    "field_count",
)


def signature(values: dict[str, object]) -> None:
  missing = [key for key in _REQUIRED_SIGNATURE_KEYS if key not in values]
  if missing:
    raise ValueError(f"missing signature key(s): {','.join(missing)}")
  print("SIGNATURE " + json.dumps(values, sort_keys=True, separators=(",", ":")))


def _candidate_build_roots(root: Path) -> list[Path]:
  candidates = [root / "build", root / "build-codex"]
  candidates.extend(sorted(root.glob("build*")))
  seen: set[Path] = set()
  uniq: list[Path] = []
  for cand in candidates:
    if cand in seen:
      continue
    seen.add(cand)
    uniq.append(cand)
  return uniq


def find_cpp_tutorial_binary(name: str, root: Optional[Path] = None) -> Optional[Path]:
  if root is None:
    root = repo_root()

  direct = shutil.which(name)
  if direct:
    return Path(direct)

  for build_root in _candidate_build_roots(root):
    cand = build_root / "tutorials" / name
    if cand.exists():
      return cand
  return None


def run_cpp_tutorial_pair(name: str, args: Optional[Sequence[str]] = None) -> int:
  root = repo_root()
  binary = find_cpp_tutorial_binary(name, root)
  if binary is None:
    if strict_mode():
      raise RuntimeError(f"missing C++ tutorial binary: {name}")
    return skip(f"missing C++ tutorial binary {name}; build tutorials first")

  cmd = [str(binary)]
  if args:
    cmd.extend(args)

  proc = subprocess.run(cmd, cwd=str(root), check=False, text=True, capture_output=True)
  if proc.stdout:
    print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
  if proc.stderr:
    print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")

  if proc.returncode != 0:
    if strict_mode():
      raise RuntimeError(f"C++ paired tutorial failed ({name}) rc={proc.returncode}")
    return skip(f"C++ paired tutorial failed ({name}) rc={proc.returncode}")
  return 0
