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

  if "--help" in argv:
    print(f"Usage: {argv[0]} [--streams <n>] [--frames <n>]")
    return 0

  def _parse_int(key: str, default: int) -> int:
    raw = next((argv[i + 1] for i in range(1, len(argv) - 1) if argv[i] == key), None)
    if raw is None:
      return default
    try:
      return int(raw)
    except Exception as exc:
      raise ValueError(f"invalid integer for {key}: {raw}") from exc

  streams = _parse_int("--streams", 8)
  frames = _parse_int("--frames", 4)

  print("STEP input_contract: each stream/frame pair is tagged so scheduler fairness is observable")
  print("STEP run_mode_choice: build a strict stage graph: stamp -> scheduler -> fanout -> join")
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
      _pushed = run.push(stamp, make_rgb_sample(str(sid), frame))
      print("CHECK graph_push: " + ("PASS" if _pushed else "FAIL") + f" (stream={sid} frame={frame})")
      assert _pushed, f"check failed: graph_push (stream={sid} frame={frame})"

  expected = streams * frames
  received = 0
  first_bundle_fields = -1
  for _ in range(expected):
    out = run.pull(join, 2000)
    _pulled = out is not None
    print("CHECK graph_pull: " + ("PASS" if _pulled else "FAIL") + " (join node produced bundle)")
    assert _pulled, "check failed: graph_pull (join node produced bundle)"
    if first_bundle_fields < 0:
      first_bundle_fields = len(out.fields)
    received += 1

  run.stop()

  print("STEP output_interpretation: joined bundle cardinality validates multistream graph wiring")
  _cond = received == expected
  print("CHECK all_outputs_received: " + ("PASS" if _cond else "FAIL") + f" (expected={expected}, received={received})")
  assert _cond, f"check failed: all_outputs_received (expected={expected}, received={received})"
  _cond = first_bundle_fields == 2
  print("CHECK bundle_has_two_fields: " + ("PASS" if _cond else "FAIL") + " (join should emit image+bbox bundle)")
  assert _cond, "check failed: bundle_has_two_fields (join should emit image+bbox bundle)"
  # END CORE LOGIC

  print("SIGNATURE " + json.dumps({
          "tutorial": "016",
          "lang": "py",
          "flow": "multistream_stage_graph",
          "run_mode": "graph_sync_pull",
          "output_kind": 1,
          "tensor_rank": -1,
          "field_count": 2,
          "streams": streams,
          "frames": frames,
      },
      sort_keys=True,
      separators=(",", ":"),
  ))

  print("[OK] 016_graph_multistream")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
