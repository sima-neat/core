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
  _pushed = run.push(pipe, make_rgb_sample())
  print("CHECK graph_push: " + ("PASS" if _pushed else "FAIL") + " (sample reached pipeline node)")
  assert _pushed, "check failed: graph_push (sample reached pipeline node)"
  out = run.pull(stamp, 2000)
  _pulled = out is not None
  print("CHECK graph_pull: " + ("PASS" if _pulled else "FAIL") + " (stage sink produced output)")
  assert _pulled, "check failed: graph_pull (stage sink produced output)"
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
  _pushed = run.push(stamp, make_rgb_sample())
  print("CHECK graph_push: " + ("PASS" if _pushed else "FAIL") + " (sample reached stage node)")
  assert _pushed, "check failed: graph_push (sample reached stage node)"
  out = run.pull(stamp, 2000)
  _pulled = out is not None
  print("CHECK graph_pull: " + ("PASS" if _pulled else "FAIL") + " (stage sink produced output)")
  assert _pulled, "check failed: graph_pull (stage sink produced output)"
  run.stop()
  # END CORE LOGIC
  return "stage_only_fallback", out


def main(argv: list[str]) -> int:

  if "--help" in argv:
    print(f"Usage: {argv[0]}")
    return 0

  print("STEP input_contract: build a minimal graph and push one deterministic tensor sample")
  print("STEP run_mode_choice: prefer pipeline+stage hybrid; fallback to stage-only if needed")
  flow = "pipeline_plus_stage"
  try:
    flow, out = run_pipeline_plus_stage()
  except Exception as exc:
    print(f"fallback reason: {exc}")
    flow, out = run_stage_only_fallback()

  print("STEP output_interpretation: verify stream/frame metadata survived graph traversal")
  _cond = bool(out.stream_id)
  print("CHECK stream_id_present: " + ("PASS" if _cond else "FAIL") + " (stream id should be non-empty)")
  assert _cond, "check failed: stream_id_present (stream id should be non-empty)"
  _cond = out.frame_id >= 0
  print("CHECK frame_id_stamped: " + ("PASS" if _cond else "FAIL") + " (stamp stage should assign non-negative frame id)")
  assert _cond, "check failed: frame_id_stamped (stamp stage should assign non-negative frame id)"

  print("SIGNATURE " + json.dumps({
          "tutorial": "014",
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

  print(f"Output stream={out.stream_id} frame={out.frame_id}")
  print("[OK] 014_graph_basics")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
