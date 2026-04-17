from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS_ROOT = REPO_ROOT / "tutorials"
REQUIRED_SIGNATURE_KEYS = {
    "tutorial",
    "lang",
    "flow",
    "run_mode",
    "output_kind",
    "tensor_rank",
    "field_count",
}


def _signature_keys(text: str, suffix: str) -> set[str]:
  keys: set[str] = set()
  if suffix == ".py":
    for match in re.finditer(r"tu\.signature\(\s*(\{.*?\})\s*\)", text, flags=re.S):
      keys.update(re.findall(r'["\']([A-Za-z0-9_]+)["\']\s*:', match.group(1)))
  else:
    for match in re.finditer(r"tutorial_v2::print_signature\(\{(.*?)\}\);", text, flags=re.S):
      keys.update(re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,', match.group(1)))
  return keys


def test_teaching_helpers_exist() -> None:
  py_helper = (TUTORIALS_ROOT / "common" / "python_utils.py").read_text(encoding="utf-8", errors="ignore")
  cpp_helper = (TUTORIALS_ROOT / "common" / "cpp_utils.h").read_text(encoding="utf-8", errors="ignore")

  for marker in ("def runtime_fallback(",):
    assert marker in py_helper, f"missing Python helper: {marker}"

  for marker in (
      "inline void why(",
      "inline void tradeoff(",
      "inline void failure_mode(",
      "inline void interpret_output(",
      "inline void runtime_fallback(",
  ):
    assert marker in cpp_helper, f"missing C++ helper: {marker}"


def test_tutorial_teaching_markers_and_signature_contract() -> None:
  for path in sorted(TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*/*")):
    if path.suffix not in {".py", ".cpp"}:
      continue

    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(REPO_ROOT))

    if path.suffix == ".cpp":
      assert "tutorial_v2::why(" in text, f"missing why marker in {rel}"
      assert "tutorial_v2::tradeoff(" in text, f"missing tradeoff marker in {rel}"
      assert "tutorial_v2::failure_mode(" in text, f"missing failure_mode marker in {rel}"
      assert "tutorial_v2::interpret_output(" in text, f"missing interpret_output marker in {rel}"

    keys = _signature_keys(text, path.suffix)
    missing = sorted(REQUIRED_SIGNATURE_KEYS - keys)
    assert not missing, f"missing signature keys in {rel}: {','.join(missing)}"
