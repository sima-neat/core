#!/usr/bin/env python3
"""Compose a Model directly inside a public pyneat.Graph.

Usage:
  python3 embed_model_inside_graph.py --model /path/to/yolo_v8s.tar.gz
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

def main(argv: list[str]) -> int:
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--model", type=Path, required=True)
  args = ap.parse_args(argv[1:])

  # STEP load-model
  model = pyneat.Model(str(args.model))
  # END STEP

  # CORE LOGIC
  # STEP compose-graph
  graph = pyneat.Graph()
  graph.add(pyneat.nodes.input("image"))
  graph.add(model)
  graph.add(pyneat.nodes.output("result"))
  print(graph.describe())
  # END STEP
  # END CORE LOGIC

  # STEP inspect-model
  print("model fragment added to public Graph")
  # END STEP
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
