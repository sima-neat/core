#!/usr/bin/env python3
"""Read the Sample that a Graph Run returns: kind, tensor, fields.

Usage:
  python3 interpret_model_output.py
"""
from __future__ import annotations

import argparse
import sys

try:
  import pyneat
except ImportError:
  sys.exit(
      "pyneat is not importable. Either Neat is not installed, or the venv is not activated.\n"
      "Run: source ~/pyneat/bin/activate"
  )

import numpy as np


def main(argv: list[str]) -> int:
  argparse.ArgumentParser(description=__doc__).parse_args(argv[1:])

  rgb = np.full((120, 160, 3), 101, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(rgb, copy=True, image_format=pyneat.PixelFormat.RGB)
  input_sample = pyneat.Sample()
  input_sample.kind = pyneat.SampleKind.Tensor
  input_sample.tensor = tensor

  # STEP configure-input
  inp = pyneat.InputOptions()
  inp.format = "RGB"
  inp.width = 160
  inp.height = 120
  inp.depth = 3
  # END STEP

  # STEP compose-graph
  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input(inp))
  graph.add(pyneat.nodes.output())

  # END STEP

  # STEP run-frame
  sample = graph.run([input_sample])
  # END STEP

  # CORE LOGIC
  # STEP inspect-sample
  print(f"sample_kind={sample.kind}")
  print(f"has_tensor={sample.tensor is not None}")
  print(f"num_fields={len(sample.fields)}")
  print(f"output_rank={len(sample.tensor.shape)}")
  # END STEP
  # END CORE LOGIC
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
