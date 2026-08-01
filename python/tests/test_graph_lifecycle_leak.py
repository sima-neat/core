"""Regression coverage for Python-owned composite graph teardown."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
from pathlib import Path


_CHILD_COMPLETE = "composite graph lifecycle child completed"
_LEAK_MARKERS = ("nanobind: leaked", "keep_alive records")


def _exercise_composite_graph_lifecycle() -> None:
  import numpy as np
  import pyneat

  seed = pyneat.Tensor.from_numpy(
      np.zeros((24, 32, 3), dtype=np.uint8),
      copy=True,
      image_format=pyneat.PixelFormat.RGB,
      memory=pyneat.TensorMemory.CPU,
  )

  for _ in range(3):
    source = pyneat.Graph("image")
    source.add(pyneat.nodes.input("image"))

    sink = pyneat.Graph("output")
    sink.add(pyneat.nodes.output("output"))

    graph = pyneat.Graph()
    graph.connect(source, sink)

    run = graph.build([seed])
    run.close()

    del run, graph, sink, source
    gc.collect()

  del seed
  gc.collect()
  print(_CHILD_COMPLETE)


def test_composite_graph_close_releases_python_bindings() -> None:
  env = os.environ.copy()
  env.pop("GST_PLUGIN_SYSTEM_PATH_1_0", None)

  result = subprocess.run(
      [sys.executable, str(Path(__file__).resolve()), "--child"],
      check=False,
      capture_output=True,
      text=True,
      timeout=120,
      env=env,
  )

  assert result.returncode == 0, result.stderr
  assert _CHILD_COMPLETE in result.stdout

  stderr = result.stderr.lower()
  leaked = [marker for marker in _LEAK_MARKERS if marker in stderr]
  assert not leaked, f"native Python bindings leaked at interpreter shutdown:\n{result.stderr}"


if __name__ == "__main__":
  if sys.argv[1:] != ["--child"]:
    raise SystemExit("expected --child")
  _exercise_composite_graph_lifecycle()
