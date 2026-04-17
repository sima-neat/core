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


def make_rgb_sample():
  # We use a deterministic CPU tensor so graph behavior is reproducible.
  import numpy as np

  arr = (np.arange(8 * 8 * 3, dtype=np.uint8) % 255).reshape(8, 8, 3)
  tensor = pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)

  sample = pyneat.Sample()
  sample.kind = pyneat.SampleKind.Tensor
  sample.tensor = tensor
  sample.stream_id = "graph"
  sample.frame_id = -1
  return sample


def run_pipeline_plus_stage():
  # CORE LOGIC
  # Hybrid flow: pipeline node normalizes media contract, stage node stamps frame ids.
  graph = pyneat.graph.Graph()
  pipe = graph.add(pyneat.graph.nodes.pipeline_node(pyneat.nodes.video_convert(), "convert"))
  stamp = graph.add(pyneat.graph.nodes.stamp_frame_id("stamp"))
  graph.connect(pipe, stamp)

  print(pyneat.graph.to_text(graph))
  run = pyneat.graph.GraphSession(graph).build()
  tu.check("graph_push", run.push(pipe, make_rgb_sample()), "sample reached pipeline node")
  out = run.pull(stamp, 2000)
  tu.check("graph_pull", out is not None, "stage sink produced output")
  run.stop()
  # END CORE LOGIC
  return "pipeline_plus_stage", out


def run_stage_only_fallback():
  # CORE LOGIC
  # Fallback still teaches graph push/pull and stage execution when pipeline plugin setup differs.
  graph = pyneat.graph.Graph()
  stamp = graph.add(pyneat.graph.nodes.stamp_frame_id("stamp"))

  print(pyneat.graph.to_text(graph))
  run = pyneat.graph.GraphSession(graph).build()
  tu.check("graph_push", run.push(stamp, make_rgb_sample()), "sample reached stage node")
  out = run.pull(stamp, 2000)
  tu.check("graph_pull", out is not None, "stage sink produced output")
  run.stop()
  # END CORE LOGIC
  return "stage_only_fallback", out


def main(argv: list[str]) -> int:

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]}")
    return 0

  tu.step("input_contract", "build a minimal graph and push one deterministic tensor sample")
  tu.step("run_mode_choice", "prefer pipeline+stage hybrid; fallback to stage-only if needed")
  tu.why("understand the contract first: inputs, run mode, and outputs")
  tu.tradeoff("prefer deterministic samples and stable contracts over production realism")
  tu.failure_mode("runtime/plugin issues should degrade to runtime_fallback without losing observability")
  tu.interpret_output("use CHECK markers plus SIGNATURE fields to validate behavior and parity")

  flow = "pipeline_plus_stage"
  try:
    flow, out = run_pipeline_plus_stage()
  except Exception as exc:
    print(f"fallback reason: {exc}")
    flow, out = run_stage_only_fallback()

  tu.step("output_interpretation", "verify stream/frame metadata survived graph traversal")
  tu.check("stream_id_present", bool(out.stream_id), "stream id should be non-empty")
  tu.check("frame_id_stamped", out.frame_id >= 0, "stamp stage should assign non-negative frame id")

  tu.signature(
      {
          "tutorial": "014",
          "lang": "py",
          "flow": flow,
          "run_mode": "graph_sync_pull",
          "output_kind": 0,
          "tensor_rank": 3,
          "field_count": 0,
      }
  )

  print(f"Output stream={out.stream_id} frame={out.frame_id}")
  print("[OK] 014_graph_basics")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
