#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit("pyneat is not importable. Either NEAT is not installed, or the venv is not activated.\nRun: source ~/pyneat/bin/activate\nIf the venv does not exist yet, follow the installation guide.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu

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
  tu.check("graph_push", run.push(node_id, inp), "sample accepted by stage-model node")
  out = run.pull(node_id, 2000)
  tu.check("graph_pull", out is not None, "stage-model node produced output")
  run.stop()
  # END CORE LOGIC
  return "model_stage", out


def run_stage_fallback():
  # CORE LOGIC
  graph = pyneat.graph.Graph()
  node_id = graph.add(pyneat.graph.nodes.stamp_frame_id("stamp"))
  run = pyneat.graph.GraphSession(graph).build()

  inp = make_fp32_sample(8, 8, 3)
  tu.check("graph_push", run.push(node_id, inp), "sample accepted by fallback stage")
  out = run.pull(node_id, 2000)
  tu.check("graph_pull", out is not None, "fallback stage produced output")
  run.stop()
  # END CORE LOGIC
  return "stage_fallback", out


def main(argv: list[str]) -> int:

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--mpk <path>]")
    return 0

  root = tu.repo_root()
  mpk_arg = tu.get_arg(argv, "--mpk")
  mpk = Path(mpk_arg) if mpk_arg else tu.first_existing([tu.default_yolo_mpk(root), tu.default_resnet_mpk(root)])

  tu.step("input_contract", "graph model stage expects tensor input with model-compatible dimensions")
  tu.step("run_mode_choice", "run stage-model hybrid when MPK exists, otherwise stage fallback")
  tu.why("understand the contract first: inputs, run mode, and outputs")
  tu.tradeoff("prefer deterministic samples and stable contracts over production realism")
  tu.failure_mode("runtime/plugin issues should degrade to runtime_fallback without losing observability")
  tu.interpret_output("use CHECK markers plus SIGNATURE fields to validate behavior and parity")

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

  tu.step("output_interpretation", "read output rank and payload shape to reason about stage boundaries")
  tu.check("output_kind_tensor", out.kind == pyneat.SampleKind.Tensor, "hybrid stage should emit tensor sample")
  tu.check("output_tensor_present", out.tensor is not None, "tensor payload must exist")

  tu.signature(
      {
          "tutorial": "015",
          "lang": "py",
          "flow": flow,
          "run_mode": "graph_sync_pull",
          "output_kind": 0,
          "tensor_rank": 3,
          "field_count": 0,
      }
  )

  print(f"Output rank: {len(out.tensor.shape) if out.tensor is not None else 0}")
  print("[OK] 015_graph_model_hybrid")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
