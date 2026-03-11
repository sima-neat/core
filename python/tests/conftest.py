from __future__ import annotations

import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
PKG_DIR = PYTHON_ROOT / "pyneat"

if str(PYTHON_ROOT) not in sys.path:
  sys.path.insert(0, str(PYTHON_ROOT))


def _ensure_pyneat_core_in_package() -> None:
  existing = list(PKG_DIR.glob("_pyneat_core*.so"))
  if existing:
    return

  candidates = []
  for build_dir in sorted(REPO_ROOT.glob("build*")):
    candidates.extend(build_dir.glob("python/_pyneat_core*.so"))

  if not candidates:
    raise AssertionError("missing pyneat core .so; build python bindings first")

  src = candidates[0]
  dst = PKG_DIR / src.name
  shutil.copy2(src, dst)


_ensure_pyneat_core_in_package()
