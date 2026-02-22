#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
import python_utils as tu


def parse_drop(neat, argv: list[str]):
  mode = tu.get_arg(argv, "--drop", "block")
  if mode == "latest":
    return neat.OverflowPolicy.KeepLatest
  if mode == "incoming":
    return neat.OverflowPolicy.DropIncoming
  return neat.OverflowPolicy.Block


def main(argv: list[str]) -> int:
  neat = tu.import_pyneat()
  import numpy as np

  if tu.has_flag(argv, "--help"):
    print(f"Usage: {argv[0]} [--iters <n>] [--queue <n>] [--drop <mode>] [--print-gst]")
    return 0

  # Why: runtime markers make intent explicit without requiring external docs.
  # Why: parity/scorecard tooling relies on stable, machine-parseable checkpoints.
  tu.step("input_contract", "parse flags and establish deterministic defaults")
  tu.step("run_mode_choice", "exercise the chapter's primary runtime path")
  tu.why("understand the contract first: inputs, run mode, and outputs")
  tu.tradeoff("prefer deterministic samples and stable contracts over production realism")
  tu.failure_mode("runtime/plugin issues should degrade to runtime_fallback without losing observability")
  tu.interpret_output("use CHECK markers plus SIGNATURE fields to validate behavior and parity")
  tu.step("output_contract", "emit checks and machine-parseable signature")
  tu.check("strict_flag_available", isinstance(tu.strict_mode(), bool),
           "strict-mode guard is observable")

  iters = tu.parse_int(argv, "--iters", 32)
  queue = tu.parse_int(argv, "--queue", 4)

  rgb = np.full((120, 160, 3), 88, dtype=np.uint8)
  t = neat.Tensor.from_numpy(rgb, copy=True, image_format=neat.PixelFormat.RGB)

  inp = neat.InputOptions()
  inp.format = "RGB"
  inp.width = 160
  inp.height = 120
  inp.depth = 3
  inp.is_live = True

  s = neat.Session()
  s.add(neat.nodes.input(inp))
  s.add(neat.nodes.output())

  if tu.has_flag(argv, "--print-gst"):
    print(s.describe_backend())
    return 0

  # CORE LOGIC
  opt = neat.RunOptions()
  opt.queue_depth = queue
  opt.overflow_policy = parse_drop(neat, argv)
  opt.output_memory = neat.OutputMemory.Owned
  opt.enable_metrics = True

  run = s.build(t, neat.RunMode.Async, opt)
  for _ in range(iters):
    run.try_push(t)
  run.close_input()

  pulled = 0
  while run.pull(timeout_ms=1000) is not None:
    pulled += 1

  stats = run.stats()
  input_stats = run.input_stats()
  print(f"inputs_enqueued={stats.inputs_enqueued}")
  print(f"inputs_dropped={stats.inputs_dropped}")
  print(f"outputs_pulled={pulled}")
  print(f"avg_latency_ms={stats.avg_latency_ms}")
  print(f"avg_push_us={input_stats.avg_push_us}")
  print(f"renegotiations={input_stats.renegotiations}")
  # END CORE LOGIC

  tu.check("tutorial_completed", True, "main path reached end without exception")
  tu.signature({
      "tutorial": "017",
      "lang": "py",
      "flow": "chapter_path",
      "run_mode": "sync_or_async",
      "output_kind": "sample_or_tensor",
      "tensor_rank": -1,
      "field_count": -1,
  })

  print("[OK] 017_performance_tuning")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
