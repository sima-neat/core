from pathlib import Path

import pytest


def test_pyneat_wheel_excludes_board_peripheral_service():
  root = Path(__file__).resolve().parents[2]
  if not (root / "pyproject.toml").is_file():
    pytest.skip("source-only packaging contract")

  pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
  cmake = (root / "CMakeLists.txt").read_text(encoding="utf-8")

  assert 'SIMANEAT_BUILD_PERIPHERAL_DAEMON = "OFF"' in pyproject
  assert '"Build the board-local peripheral catalog daemon"\n  OFF)' in cmake
  assert "if (SKBUILD AND SIMANEAT_BUILD_PERIPHERAL_DAEMON)" in cmake
