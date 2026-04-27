#!/usr/bin/env python3
"""Run a Model inside a pyneat.graph via StageModelExecutorOptions.

Usage:
  python3 graph_model_hybrid.py --mpk /path/to/yolo_v8s.tar.gz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )

import numpy as np


def make_fp32_sample(width: int, height: int, depth: int):
  arr = np.zeros((height, width, depth), dtype=np.float32)
  sample = pyneat.Sample()
  sample.kind = pyneat.SampleKind.Tensor
  sample.tensor = pyneat.Tensor.from_numpy(arr, copy=True)
  sample.frame_id = 1
  sample.stream_id = "model"
  return sample


def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--mpk", type=Path, required=True)
  args = ap.parse_args(argv[1:])

  model = pyneat.Model(str(args.mpk))

  stage_opt = pyneat.graph.nodes.StageModelExecutorOptions()
  stage_opt.model = model
  stage_opt.do_preproc = False
  stage_opt.do_mla = False
  stage_opt.do_boxdecode = False

  # CORE LOGIC
  graph = pyneat.graph.Graph()
  node_id = graph.add(pyneat.graph.nodes.stage_model_executor(stage_opt, "stage_model"))
  run = pyneat.graph.GraphSession(graph).build()
  # END CORE LOGIC

  tensor_opt = model.input_appsrc_options(True)
  width = tensor_opt.width if tensor_opt.width > 0 else tensor_opt.max_width
  height = tensor_opt.height if tensor_opt.height > 0 else tensor_opt.max_height
  depth = tensor_opt.depth if tensor_opt.depth > 0 else tensor_opt.max_depth

  run.push(node_id, make_fp32_sample(width, height, depth))
  out = run.pull(node_id, 2000)
  run.stop()

  print(f"output_kind={out.kind} output_rank={len(out.tensor.shape)}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
