#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


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
class Violation:
  file: str
  rule: str
  detail: str


def _read(path: Path) -> str:
  return path.read_text(encoding="utf-8", errors="ignore")


def _signature_keys(text: str, suffix: str) -> set[str]:
  keys: set[str] = set()
  if suffix == ".py":
    for match in re.finditer(r"tu\.signature\(\s*(\{.*?\})\s*\)", text, flags=re.S):
      keys.update(re.findall(r'["\']([A-Za-z0-9_]+)["\']\s*:', match.group(1)))
  else:
    for match in re.finditer(r"tutorial_v2::print_signature\(\{(.*?)\}\);", text, flags=re.S):
      keys.update(re.findall(r'\{\s*"([A-Za-z0-9_]+)"\s*,', match.group(1)))
  return keys


def lint() -> list[Violation]:
  violations: list[Violation] = []

  py_helper = _read(TUTORIALS_ROOT / "common" / "python_utils.py")
  for marker in ("def why(", "def tradeoff(", "def failure_mode(", "def interpret_output(", "def runtime_fallback("):
    if marker not in py_helper:
      violations.append(
          Violation(
              file="tutorials/common/python_utils.py",
              rule="teaching-helper",
              detail=f"missing helper marker: {marker}",
          )
      )

  if "raise ValueError" not in py_helper:
    violations.append(
        Violation(
            file="tutorials/common/python_utils.py",
            rule="strict-parse",
            detail="parse helpers must raise on invalid CLI values",
        )
    )

  cpp_helper = _read(TUTORIALS_ROOT / "common" / "cpp_utils.h")
  for marker in (
      "inline void why(",
      "inline void tradeoff(",
      "inline void failure_mode(",
      "inline void interpret_output(",
      "inline void runtime_fallback(",
  ):
    if marker not in cpp_helper:
      violations.append(
          Violation(
              file="tutorials/common/cpp_utils.h",
              rule="teaching-helper",
              detail=f"missing helper marker: {marker}",
          )
      )

  if "throw std::invalid_argument" not in cpp_helper:
    violations.append(
        Violation(
            file="tutorials/common/cpp_utils.h",
            rule="strict-parse",
            detail="parse helpers must throw on invalid CLI values",
        )
    )

  for path in sorted(TUTORIALS_ROOT.glob("[0-9][0-9][0-9]_*/*")):
    if path.suffix not in {".py", ".cpp"}:
      continue

    text = _read(path)
    rel = str(path.relative_to(REPO_ROOT))
    tid = path.parent.name.split("_", 1)[0]

    if "TODO" in text:
      violations.append(
          Violation(
              file=rel,
              rule="no-todo",
              detail="tutorial tracks cannot ship unresolved TODO markers",
          )
      )

    if path.suffix == ".py":
      step_count = len(re.findall(r"\btu\.step\(", text))
      check_count = len(re.findall(r"\btu\.(check|ensure)\(", text))
      signature_count = len(re.findall(r"\btu\.signature\(", text))
      why_count = len(re.findall(r"\btu\.why\(", text))
      tradeoff_count = len(re.findall(r"\btu\.tradeoff\(", text))
      failure_count = len(re.findall(r"\btu\.failure_mode\(", text))
      interpret_count = len(re.findall(r"\btu\.interpret_output\(", text))
      raw_fallback_count = len(re.findall(r'print\(\s*f?["\']runtime_fallback:', text))

      if step_count < 3 or check_count < 2 or signature_count < 1:
        violations.append(
            Violation(
                file=rel,
                rule="marker-minimum",
                detail=(
                    f"need step>=3/check>=2/signature>=1; got "
                    f"step={step_count} check={check_count} signature={signature_count}"
                ),
            )
        )

      if why_count < 1 or tradeoff_count < 1 or failure_count < 1 or interpret_count < 1:
        violations.append(
            Violation(
                file=rel,
                rule="teaching-contract",
                detail=(
                    "need why/tradeoff/failure_mode/interpret_output >=1; got "
                    f"why={why_count} tradeoff={tradeoff_count} "
                    f"failure_mode={failure_count} interpret_output={interpret_count}"
                ),
            )
        )

      if raw_fallback_count > 0:
        violations.append(
            Violation(
                file=rel,
                rule="fallback-helper",
                detail="use tu.runtime_fallback(...) instead of raw runtime_fallback print",
            )
        )

      if tid in {"014", "015", "016"} and "tu.skip(" in text:
        violations.append(
            Violation(
                file=rel,
                rule="no-graph-stub",
                detail="graph tutorials must not use skip-as-implementation",
            )
        )

    else:
      step_count = len(re.findall(r"\btutorial_v2::step\(", text))
      check_count = len(re.findall(r"\btutorial_v2::(check|require)\(", text))
      signature_count = len(re.findall(r"\btutorial_v2::print_signature\(", text))
      why_count = len(re.findall(r"\btutorial_v2::why\(", text))
      tradeoff_count = len(re.findall(r"\btutorial_v2::tradeoff\(", text))
      failure_count = len(re.findall(r"\btutorial_v2::failure_mode\(", text))
      interpret_count = len(re.findall(r"\btutorial_v2::interpret_output\(", text))
      raw_fallback_count = len(re.findall(r'std::cout << "runtime_fallback:', text))

      if step_count < 3 or check_count < 2 or signature_count < 1:
        violations.append(
            Violation(
                file=rel,
                rule="marker-minimum",
                detail=(
                    f"need step>=3/check>=2/signature>=1; got "
                    f"step={step_count} check={check_count} signature={signature_count}"
                ),
            )
        )

      if why_count < 1 or tradeoff_count < 1 or failure_count < 1 or interpret_count < 1:
        violations.append(
            Violation(
                file=rel,
                rule="teaching-contract",
                detail=(
                    "need why/tradeoff/failure_mode/interpret_output >=1; got "
                    f"why={why_count} tradeoff={tradeoff_count} "
                    f"failure_mode={failure_count} interpret_output={interpret_count}"
                ),
            )
        )

      if raw_fallback_count > 0:
        violations.append(
            Violation(
                file=rel,
                rule="fallback-helper",
                detail="use tutorial_v2::runtime_fallback(...) instead of raw runtime_fallback print",
            )
        )

    sig_keys = _signature_keys(text, path.suffix)
    missing = sorted(REQUIRED_SIGNATURE_KEYS - sig_keys)
    if missing:
      violations.append(
          Violation(
              file=rel,
              rule="signature-contract",
              detail=f"missing required signature keys: {','.join(missing)}",
          )
      )

  return violations


def main() -> int:
  parser = argparse.ArgumentParser(description="Lint tutorial quality markers and anti-stub rules")
  parser.add_argument("--json", action="store_true", help="Emit JSON diagnostics")
  args = parser.parse_args()

  violations = lint()

  if args.json:
    print(
        json.dumps(
            {
                "ok": not violations,
                "violations": [v.__dict__ for v in violations],
            },
            indent=2,
            sort_keys=True,
        )
    )
  else:
    if not violations:
      print("tutorial quality lint: OK")
    else:
      print(f"tutorial quality lint: {len(violations)} violation(s)")
      for v in violations:
        print(f"  - {v.file}: [{v.rule}] {v.detail}")

  return 0 if not violations else 1


if __name__ == "__main__":
  raise SystemExit(main())
