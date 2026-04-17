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
    for match in re.finditer(r'print\("SIGNATURE " \+ json\.dumps\(\s*(\{.*?\})', text, flags=re.S):
      keys.update(re.findall(r'["\']([A-Za-z0-9_]+)["\']\s*:', match.group(1)))
  else:
    for match in re.finditer(r"print_signature\(\{(.*?)\}\);", text, flags=re.S):
      keys.update(re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,', match.group(1)))
  return keys


def test_tutorial_teaching_markers_and_signature_contract() -> None:
  for path in sorted(TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*/*")):
    if path.suffix not in {".py", ".cpp"}:
      continue

    text = path.read_text(encoding="utf-8", errors="ignore")
    rel = str(path.relative_to(REPO_ROOT))

    keys = _signature_keys(text, path.suffix)
    missing = sorted(REQUIRED_SIGNATURE_KEYS - keys)
    assert not missing, f"missing signature keys in {rel}: {','.join(missing)}"
