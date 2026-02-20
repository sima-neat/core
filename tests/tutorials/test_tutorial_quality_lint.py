from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tutorial_quality_lint() -> None:
  proc = subprocess.run(
      ["python3", "tools/tutorial_quality_lint.py"],
      cwd=REPO_ROOT,
      check=False,
      text=True,
      capture_output=True,
  )
  assert proc.returncode == 0, f"tutorial quality lint failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
