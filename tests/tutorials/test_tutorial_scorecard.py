from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_tutorial_scorecard_gate() -> None:
  cmd = [
      "python3",
      "tools/tutorial_scorecard.py",
      "--min-score",
      "10.0",
      "--min-per-tutorial",
      "10.0",
  ]
  proc = subprocess.run(cmd, cwd=REPO_ROOT, check=False, text=True, capture_output=True)
  assert proc.returncode == 0, f"scorecard gate failed\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
