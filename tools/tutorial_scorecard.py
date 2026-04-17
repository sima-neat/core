#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
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


@dataclass
class FileStats:
  path: Path
  text: str
  step_count: int
  check_count: int
  signature_count: int
  comment_count: int
  ok_marker: bool
  strict_mode_ref: bool
  skip_count: int
  skip_without_strict: bool
  signature_key_count: int
  signature_required_ok: bool


@dataclass
class TutorialScore:
  tid: str
  name: str
  cpp: FileStats
  py: FileStats
  run_reliability: float
  parity: float
  explainability: float
  framework_understanding: float

  @property
  def total(self) -> float:
    return self.run_reliability + self.parity + self.explainability + self.framework_understanding


def _comment_count(lines: Iterable[str], ext: str) -> int:
  count = 0
  for line in lines:
    s = line.lstrip()
    if ext == ".py":
      if s.startswith("#") and not s.startswith("#!"):
        count += 1
    else:
      if s.startswith("//"):
        count += 1
  return count


def _signature_call_keys(text: str, ext: str) -> set[str]:
  keys: set[str] = set()
  if ext == ".py":
    for match in re.finditer(r'print\("SIGNATURE " \+ json\.dumps\(\s*(\{.*?\})', text, flags=re.S):
      keys.update(re.findall(r'["\']([A-Za-z0-9_]+)["\']\s*:', match.group(1)))
  else:
    for match in re.finditer(r"tutorial_v2::print_signature\(\{(.*?)\}\);", text, flags=re.S):
      keys.update(re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,', match.group(1)))
  return keys


def analyze_file(path: Path, tid: str) -> FileStats:
  text = path.read_text(encoding="utf-8", errors="ignore")
  lines = text.splitlines()
  ext = path.suffix

  if ext == ".py":
    step_count = len(re.findall(r'print\(f?"STEP ', text))
    check_count = len(re.findall(r'print\(f?"CHECK ', text))
    signature_count = len(re.findall(r'print\("SIGNATURE " \+ json\.dumps', text))
    ok_marker = bool(re.search(rf"\[OK\]\s+{re.escape(tid)}", text))
    skip_count = len(re.findall(r'print\(f?"SKIP:', text))
    strict_mode_ref = "strict_mode" in text
  else:
    step_count = len(re.findall(r"\btutorial_v2::step\(", text))
    check_count = len(re.findall(r"\btutorial_v2::(check|require)\(", text))
    signature_count = len(re.findall(r"\btutorial_v2::print_signature\(", text))
    ok_marker = bool(re.search(rf"\[OK\]\s+{re.escape(tid)}", text))
    skip_count = len(re.findall(r"return\s+tutorial_v2::skip\(", text))
    strict_mode_ref = "strict_mode" in text

  comment_count = _comment_count(lines, ext)
  sig_keys = _signature_call_keys(text, ext)
  signature_key_count = len(sig_keys)
  signature_required_ok = REQUIRED_SIGNATURE_KEYS.issubset(sig_keys)
  skip_without_strict = skip_count > 0 and not strict_mode_ref

  return FileStats(
      path=path,
      text=text,
      step_count=step_count,
      check_count=check_count,
      signature_count=signature_count,
      comment_count=comment_count,
      ok_marker=ok_marker,
      strict_mode_ref=strict_mode_ref,
      skip_count=skip_count,
      skip_without_strict=skip_without_strict,
      signature_key_count=signature_key_count,
      signature_required_ok=signature_required_ok,
  )


def _score_run_reliability(cpp: FileStats, py: FileStats) -> float:
  score = 0.0
  py_stubbed = py.skip_count > 0 and py.signature_count == 0
  if not py_stubbed:
    score += 1.0

  if not cpp.skip_without_strict and not py.skip_without_strict:
    score += 0.75

  if cpp.check_count >= 1 and py.check_count >= 1:
    score += 0.75

  return score


def _score_parity(cpp: FileStats, py: FileStats) -> float:
  score = 0.0
  if cpp.signature_count >= 1 and py.signature_count >= 1:
    score += 1.0
  if cpp.step_count >= 2 and py.step_count >= 2:
    score += 0.75
  if cpp.ok_marker and py.ok_marker:
    score += 0.75
  return score


def _score_explainability(cpp: FileStats, py: FileStats) -> float:
  score = 0.0
  if cpp.step_count >= 3 and py.step_count >= 3:
    score += 2.5
  return score


def _score_framework_understanding(cpp: FileStats, py: FileStats) -> float:
  score = 0.0
  if cpp.signature_required_ok and py.signature_required_ok:
    score += 1.0
  if cpp.check_count >= 2 and py.check_count >= 2:
    score += 0.75
  if cpp.signature_key_count >= 6 and py.signature_key_count >= 6:
    score += 0.75
  return score


def compute_scores() -> list[TutorialScore]:
  rows: list[TutorialScore] = []
  for chapter_dir in sorted(TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*")):
    tid = chapter_dir.name.split("_", 1)[0]
    cpp_files = sorted(chapter_dir.glob("*.cpp"))
    py_files = sorted(chapter_dir.glob("*.py"))
    if len(cpp_files) != 1 or len(py_files) != 1:
      continue

    cpp = analyze_file(cpp_files[0], tid)
    py = analyze_file(py_files[0], tid)

    row = TutorialScore(
        tid=tid,
        name=chapter_dir.name,
        cpp=cpp,
        py=py,
        run_reliability=_score_run_reliability(cpp, py),
        parity=_score_parity(cpp, py),
        explainability=_score_explainability(cpp, py),
        framework_understanding=_score_framework_understanding(cpp, py),
    )
    rows.append(row)
  return rows


def main() -> int:
  parser = argparse.ArgumentParser(description="Tutorial quality scorecard (0-10).")
  parser.add_argument("--min-score", type=float, default=None, help="Fail if aggregate score < min")
  parser.add_argument(
      "--min-per-tutorial",
      type=float,
      default=None,
      help="Fail if any tutorial score < min",
  )
  parser.add_argument("--json", action="store_true", help="Emit JSON output")
  args = parser.parse_args()

  rows = compute_scores()
  if not rows:
    print("No tutorial rows found", file=sys.stderr)
    return 2

  aggregate = sum(r.total for r in rows) / len(rows)

  out = {
      "aggregate": round(aggregate, 3),
      "count": len(rows),
      "tutorials": [
          {
              "id": r.tid,
              "name": r.name,
              "total": round(r.total, 3),
              "run_reliability": round(r.run_reliability, 3),
              "parity": round(r.parity, 3),
              "explainability": round(r.explainability, 3),
              "framework_understanding": round(r.framework_understanding, 3),
          }
          for r in rows
      ],
  }

  if args.json:
    print(json.dumps(out, indent=2, sort_keys=True))
  else:
    print(f"Tutorial scorecard: {aggregate:.3f}/10 over {len(rows)} tutorials")
    for r in rows:
      print(
          f"  {r.tid} {r.total:.3f} "
          f"(R={r.run_reliability:.2f} P={r.parity:.2f} E={r.explainability:.2f} F={r.framework_understanding:.2f})"
      )

  if args.min_score is not None and aggregate < args.min_score:
    print(
        f"score gate failed: {aggregate:.3f} < required {args.min_score:.3f}",
        file=sys.stderr,
    )
    return 1

  if args.min_per_tutorial is not None:
    below = [r for r in rows if r.total < args.min_per_tutorial]
    if below:
      print(
          "per-tutorial score gate failed: "
          + ", ".join(f"{r.tid}:{r.total:.3f}" for r in below),
          file=sys.stderr,
      )
      return 1

  return 0


if __name__ == "__main__":
  raise SystemExit(main())
