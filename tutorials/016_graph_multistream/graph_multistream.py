#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit("pyneat is not installed. Follow the installation guide.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu

# Why: this chapter contrasts model-backed graph stages with deterministic fallback flow.
# Why: learners should see identical checkpoint output even when model assets are unavailable.


def make_rgb_sample(stream_id: str, frame_id: int):
  # Per-stream synthetic RGB tensors keep multistream scheduling deterministic.
  import numpy as np

  arr = (np.arange(6 * 8 * 3, dtype=np.uint8) % 255).reshape(6, 8, 3)
  tensor = pyneat.Tensor.from_numpy(arr, copy=True, image_format=pyneat.PixelFormat.RGB)

  sample = pyneat.Sample()
  sample.kind = pyneat.SampleKind.Tensor
  sample.tensor = tensor
  sample.frame_id = frame_id
  sample.stream_id = stream_id
  return sample


def main(argv: list[str]) -> int:

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--streams <n>] [--frames <n>]")
    return 0

  streams = tu.parse_int(argv, "--streams", 8)
  frames = tu.parse_int(argv, "--frames", 4)

  tu.step("input_contract", "each stream/frame pair is tagged so scheduler fairness is observable")
  tu.step("run_mode_choice", "build a strict stage graph: stamp -> scheduler -> fanout -> join")
  tu.why("understand the contract first: inputs, run mode, and outputs")
  tu.tradeoff("prefer deterministic samples and stable contracts over production realism")
  tu.failure_mode("runtime/plugin issues should degrade to runtime_fallback without losing observability")
  tu.interpret_output("use CHECK markers plus SIGNATURE fields to validate behavior and parity")

  # CORE LOGIC
  graph = pyneat.graph.Graph()
  stamp = graph.add(pyneat.graph.nodes.stamp_frame_id("stamp"))

  sched_opt = pyneat.graph.nodes.StreamSchedulerOptions()
  sched_opt.per_stream_queue = 2
  sched_opt.drop_policy = pyneat.graph.nodes.StreamDropPolicy.DropOldest
  sched = graph.add(pyneat.graph.nodes.stream_scheduler(sched_opt, "sched"))

  fan = graph.add(pyneat.graph.nodes.fan_out(["image", "bbox"], "fan"))
  join = graph.add(pyneat.graph.nodes.join_bundle(["image", "bbox"], "join", "bundle"))

  graph.connect(stamp, sched)
  graph.connect(sched, fan)
  graph.connect(fan, join, "image", "image")
  graph.connect(fan, join, "bbox", "bbox")

  print(pyneat.graph.to_text(graph))

  run = pyneat.graph.GraphSession(graph).build()
  for frame in range(frames):
    for sid in range(streams):
      tu.check(
          "graph_push",
          run.push(stamp, make_rgb_sample(str(sid), frame)),
          f"stream={sid} frame={frame}",
      )

  expected = streams * frames
  received = 0
  first_bundle_fields = -1
  for _ in range(expected):
    out = run.pull(join, 2000)
    tu.check("graph_pull", out is not None, "join node produced bundle")
    if first_bundle_fields < 0:
      first_bundle_fields = len(out.fields)
    received += 1

  run.stop()

  tu.step("output_interpretation", "joined bundle cardinality validates multistream graph wiring")
  tu.check("all_outputs_received", received == expected, f"expected={expected}, received={received}")
  tu.check("bundle_has_two_fields", first_bundle_fields == 2, "join should emit image+bbox bundle")
  # END CORE LOGIC

  tu.signature(
      {
          "tutorial": "016",
          "lang": "py",
          "flow": "multistream_stage_graph",
          "run_mode": "graph_sync_pull",
          "output_kind": 1,
          "tensor_rank": -1,
          "field_count": 2,
          "streams": streams,
          "frames": frames,
      }
  )

  print("[OK] 016_graph_multistream")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
