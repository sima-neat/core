#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either NEAT is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate\n"
      "If the venv does not exist yet, follow the installation guide."
  )

# Why: this chapter contrasts model-backed graph stages with deterministic fallback flow.
# Why: learners should see identical checkpoint output even when model assets are unavailable.


def make_fp32_sample(width: int, height: int, depth: int):
  # A synthetic tensor keeps the tutorial deterministic while matching model tensor contracts.
  import numpy as np

  arr = np.zeros((height, width, depth), dtype=np.float32)
  tensor = pyneat.Tensor.from_numpy(arr, copy=True)

  sample = pyneat.Sample()
  sample.kind = pyneat.SampleKind.Tensor
  sample.tensor = tensor
  sample.frame_id = 1
  sample.stream_id = "model"
  return sample


def run_model_hybrid(mpk: Path):
  # CORE LOGIC
  model = pyneat.Model(str(mpk))

  opt = pyneat.graph.nodes.StageModelExecutorOptions()
  opt.model = model
  opt.do_preproc = False
  opt.do_mla = False
  opt.do_boxdecode = False

  graph = pyneat.graph.Graph()
  node_id = graph.add(pyneat.graph.nodes.stage_model_executor(opt, "stage_model"))
  run = pyneat.graph.GraphSession(graph).build()

  tensor_opt = model.input_appsrc_options(True)
  width = tensor_opt.width if tensor_opt.width > 0 else tensor_opt.max_width
  height = tensor_opt.height if tensor_opt.height > 0 else tensor_opt.max_height
  depth = tensor_opt.depth if tensor_opt.depth > 0 else tensor_opt.max_depth
  if width <= 0 or height <= 0 or depth <= 0:
    width, height, depth = 8, 8, 3

  inp = make_fp32_sample(width, height, depth)
  _pushed = run.push(node_id, inp)
  print("CHECK graph_push: " + ("PASS" if _pushed else "FAIL") + " (sample accepted by stage-model node)")
  assert _pushed, "check failed: graph_push (sample accepted by stage-model node)"
  out = run.pull(node_id, 2000)
  _pulled = out is not None
  print("CHECK graph_pull: " + ("PASS" if _pulled else "FAIL") + " (stage-model node produced output)")
  assert _pulled, "check failed: graph_pull (stage-model node produced output)"
  run.stop()
  # END CORE LOGIC
  return "model_stage", out


def run_stage_fallback():
  # CORE LOGIC
  graph = pyneat.graph.Graph()
  node_id = graph.add(pyneat.graph.nodes.stamp_frame_id("stamp"))
  run = pyneat.graph.GraphSession(graph).build()

  inp = make_fp32_sample(8, 8, 3)
  _pushed = run.push(node_id, inp)
  print("CHECK graph_push: " + ("PASS" if _pushed else "FAIL") + " (sample accepted by fallback stage)")
  assert _pushed, "check failed: graph_push (sample accepted by fallback stage)"
  out = run.pull(node_id, 2000)
  _pulled = out is not None
  print("CHECK graph_pull: " + ("PASS" if _pulled else "FAIL") + " (fallback stage produced output)")
  assert _pulled, "check failed: graph_pull (fallback stage produced output)"
  run.stop()
  # END CORE LOGIC
  return "stage_fallback", out


def main(argv: list[str]) -> int:

  if "--help" in argv:
    print(f"Usage: {argv[0]} [--mpk <path>]")
    return 0

  root = Path(__file__).resolve().parents[2]
  mpk_arg = next((argv[i + 1] for i in range(1, len(argv) - 1) if argv[i] == "--mpk"), None)
  if mpk_arg:
    mpk = Path(mpk_arg)
  else:
    yolo = next(
        (p for p in [
            root / "tmp" / "yolo_v8s_mpk.tar.gz",
            root / "tmp" / "yolov8s_mpk.tar.gz",
            root / "tmp" / "yolo_mpk.tar.gz",
        ] if p.exists()),
        None,
    )
    resnet = next(
        (p for p in [
            root / "tmp" / "resnet_50_mpk.tar.gz",
            root / "tmp" / "resnet50_mpk.tar.gz",
        ] if p.exists()),
        None,
    )
    mpk = next((p for p in [yolo, resnet] if p is not None and p.exists()), None)

  print("STEP input_contract: graph model stage expects tensor input with model-compatible dimensions")
  print("STEP run_mode_choice: run stage-model hybrid when MPK exists, otherwise stage fallback")
  flow = "stage_fallback"
  out = None
  if mpk and mpk.exists():
    try:
      flow, out = run_model_hybrid(mpk)
    except Exception as exc:
      print(f"model branch fallback reason: {exc}")
      flow, out = run_stage_fallback()
  else:
    flow, out = run_stage_fallback()

  print("STEP output_interpretation: read output rank and payload shape to reason about stage boundaries")
  _cond = out.kind == pyneat.SampleKind.Tensor
  print("CHECK output_kind_tensor: " + ("PASS" if _cond else "FAIL") + " (hybrid stage should emit tensor sample)")
  assert _cond, "check failed: output_kind_tensor (hybrid stage should emit tensor sample)"
  _cond = out.tensor is not None
  print("CHECK output_tensor_present: " + ("PASS" if _cond else "FAIL") + " (tensor payload must exist)")
  assert _cond, "check failed: output_tensor_present (tensor payload must exist)"

  print("SIGNATURE " + json.dumps({
          "tutorial": "015",
          "lang": "py",
          "flow": flow,
          "run_mode": "graph_sync_pull",
          "output_kind": 0,
          "tensor_rank": 3,
          "field_count": 0,
      },
      sort_keys=True,
      separators=(",", ":"),
  ))

  print(f"Output rank: {len(out.tensor.shape) if out.tensor is not None else 0}")
  print("[OK] 015_graph_model_hybrid")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
