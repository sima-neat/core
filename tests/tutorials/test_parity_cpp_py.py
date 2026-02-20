from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS_ROOT = REPO_ROOT / "tutorials"
MANIFEST = TUTORIALS_ROOT / "manifest.yaml"
DEFAULT_TIMEOUT_SEC = int(os.getenv("SIMA_TUTORIAL_TIMEOUT_SEC", "90"))


def _text_or_empty(value: str | bytes | None) -> str:
  if value is None:
    return ""
  if isinstance(value, bytes):
    return value.decode("utf-8", errors="replace")
  return value


def _parse_manifest_pairs() -> list[tuple[str, Path, Path]]:
  pairs: list[tuple[str, Path, Path]] = []
  current_id = ""
  cpp_rel = ""
  py_rel = ""

  for raw in MANIFEST.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line.startswith("- id:"):
      if current_id and cpp_rel and py_rel:
        pairs.append((current_id, REPO_ROOT / cpp_rel, REPO_ROOT / py_rel))
      current_id = line.split(":", 1)[1].strip()
      cpp_rel = ""
      py_rel = ""
      continue
    if line.startswith("pair_cpp:"):
      cpp_rel = line.split(":", 1)[1].strip()
    elif line.startswith("pair_py:"):
      py_rel = line.split(":", 1)[1].strip()

  if current_id and cpp_rel and py_rel:
    pairs.append((current_id, REPO_ROOT / cpp_rel, REPO_ROOT / py_rel))

  assert len(pairs) == 18, f"expected 18 tutorial pairs, found {len(pairs)}"
  return pairs


def _ensure_pyneat_core_in_package() -> None:
  pkg_dir = REPO_ROOT / "python" / "pyneat"
  existing = list(pkg_dir.glob("_pyneat_core*.so"))
  if existing:
    return

  candidates = []
  for build_dir in sorted(REPO_ROOT.glob("build*")):
    candidates.extend(build_dir.glob("python/_pyneat_core*.so"))

  if not candidates:
    raise AssertionError("missing pyneat core .so; build python bindings first")

  src = candidates[0]
  dst = pkg_dir / src.name
  shutil.copy2(src, dst)


def _run_cpp_tutorial(cpp_path: Path) -> tuple[int, str]:
  chapter = cpp_path.parent.name
  target = f"tutorial_v2_{chapter}"
  binary = REPO_ROOT / "build" / "tutorials" / target
  if not binary.exists():
    raise AssertionError(f"missing C++ tutorial binary: {binary}")

  env = os.environ.copy()
  env["SIMA_RUN_TUTORIALS_FULL"] = "1"
  try:
    proc = subprocess.run(
        [str(binary)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output
  except subprocess.TimeoutExpired as exc:
    out = _text_or_empty(exc.stdout) + _text_or_empty(exc.stderr)
    return 124, out + f"\nTIMEOUT: C++ tutorial exceeded {DEFAULT_TIMEOUT_SEC}s\n"


def _run_py_tutorial(py_path: Path) -> tuple[int, str]:
  _ensure_pyneat_core_in_package()
  env = os.environ.copy()
  env["SIMA_RUN_TUTORIALS_FULL"] = "1"
  try:
    proc = subprocess.run(
        ["python3", str(py_path)],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        text=True,
        capture_output=True,
        timeout=DEFAULT_TIMEOUT_SEC,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, output
  except subprocess.TimeoutExpired as exc:
    out = _text_or_empty(exc.stdout) + _text_or_empty(exc.stderr)
    return 124, out + f"\nTIMEOUT: Python tutorial exceeded {DEFAULT_TIMEOUT_SEC}s\n"


def _parse_signature_payload(payload: str) -> dict[str, str]:
  payload = payload.strip()
  if not payload:
    return {}
  if payload.startswith("{") and ":" in payload:
    obj = ast.literal_eval(payload)
    return {str(k): str(v) for k, v in obj.items()}
  if payload.startswith("{") and "=" in payload:
    out: dict[str, str] = {}
    inner = payload.strip("{}")
    for part in inner.split(","):
      if "=" not in part:
        continue
      k, v = part.split("=", 1)
      out[k.strip()] = v.strip()
    return out
  return {}


def _signature_from_output(output: str) -> dict[str, str]:
  for line in output.splitlines():
    if line.startswith("SIGNATURE "):
      return _parse_signature_payload(line[len("SIGNATURE ") :])
  return {}


def _signature_from_py_source(py_path: Path) -> dict[str, str]:
  text = py_path.read_text(encoding="utf-8", errors="ignore")
  m = re.search(r"tu\.signature\(\s*(\{.*?\})\s*\)", text, flags=re.S)
  if not m:
    return {}
  obj = ast.literal_eval(m.group(1))
  return {str(k): str(v) for k, v in obj.items()}


def _signature_from_cpp_source(cpp_path: Path) -> dict[str, str]:
  text = cpp_path.read_text(encoding="utf-8", errors="ignore")
  m = re.search(r"tutorial_v2::print_signature\(\{(.*?)\}\);", text, flags=re.S)
  if not m:
    return {}
  out: dict[str, str] = {}
  for key, value in re.findall(r'\{\s*"([^"]+)"\s*,\s*"([^"]*)"\s*\}', m.group(1)):
    out[key] = value
  return out


def _normalized(sig: dict[str, str]) -> dict[str, str]:
  keys = ["tutorial", "run_mode", "output_kind", "tensor_rank", "field_count", "tput_contract"]
  return {k: sig.get(k, "") for k in keys}


def test_parity_cpp_py() -> None:
  mismatches: list[str] = []
  enforce_runtime = os.getenv("SIMA_TUTORIAL_ENFORCE_RUNTIME") == "1"

  for tid, cpp_path, py_path in _parse_manifest_pairs():
    rc_cpp, out_cpp = _run_cpp_tutorial(cpp_path)
    rc_py, out_py = _run_py_tutorial(py_path)

    sig_cpp = _signature_from_output(out_cpp)
    sig_py = _signature_from_output(out_py)

    if not sig_cpp:
      sig_cpp = _signature_from_cpp_source(cpp_path)
    if not sig_py:
      sig_py = _signature_from_py_source(py_path)

    norm_cpp = _normalized(sig_cpp)
    norm_py = _normalized(sig_py)

    if enforce_runtime:
      if rc_cpp != 0:
        mismatches.append(f"{tid}: C++ tutorial failed rc={rc_cpp}")
      if rc_py != 0:
        mismatches.append(f"{tid}: Python tutorial failed rc={rc_py}")
      if "SKIP:" in out_cpp:
        mismatches.append(f"{tid}: C++ tutorial emitted SKIP in strict mode")
      if "SKIP:" in out_py:
        mismatches.append(f"{tid}: Python tutorial emitted SKIP in strict mode")

    if not norm_cpp or not norm_py:
      mismatches.append(f"{tid}: missing comparable signature cpp={norm_cpp} py={norm_py}")
    elif norm_cpp != norm_py:
      mismatches.append(f"{tid}: signature mismatch cpp={norm_cpp} py={norm_py}")

  assert not mismatches, "\n".join(mismatches)
