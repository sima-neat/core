"""Regression coverage for blocking Python push bindings."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path


_CALL_SHAPES = ("samples", "tensors", "generic", "named")
_CHILD_COMPLETE = "blocking push released the GIL"


def _exercise_blocking_push(call_shape: str) -> None:
  import numpy as np
  import pyneat

  array = np.zeros((24, 32, 3), dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(
      array,
      copy=True,
      image_format=pyneat.PixelFormat.RGB,
      memory=pyneat.TensorMemory.CPU,
  )
  sample = pyneat.Sample()
  sample.kind = pyneat.SampleKind.Tensor
  sample.tensor = tensor

  input_options = pyneat.InputOptions()
  input_options.payload_type = pyneat.PayloadType.Image
  input_options.format = pyneat.Format.RGB
  input_options.width = 32
  input_options.height = 24
  input_options.depth = 3
  input_options.max_width = 32
  input_options.max_height = 24
  input_options.max_depth = 3
  input_options.memory_policy = pyneat.InputMemoryPolicy.SystemMemory

  source = pyneat.Graph("input")
  source.add(pyneat.nodes.input("input", input_options))
  sink = pyneat.Graph("output")
  sink.add(pyneat.nodes.output("output", pyneat.OutputOptions.every_frame(1)))
  graph = pyneat.Graph("blocking_push_gil")
  graph.connect(source, sink)

  options = pyneat.RunOptions()
  options.preset = pyneat.RunPreset.Realtime
  options.queue_depth = 1
  options.overflow_policy = pyneat.OverflowPolicy.Block
  options.output_memory = pyneat.OutputMemory.Owned
  run = graph.build(options)

  def try_push() -> bool:
    if call_shape == "samples":
      return run.try_push_samples(sample)
    if call_shape == "tensors":
      return run.try_push_tensors([tensor])
    if call_shape == "generic":
      return run.try_push(
          [array], layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB
      )
    return run.try_push("input", [sample])

  def push() -> bool:
    if call_shape == "samples":
      return run.push_samples(sample)
    if call_shape == "tensors":
      return run.push_tensors([tensor])
    if call_shape == "generic":
      return run.push(
          [array], layout=pyneat.TensorLayout.HWC, image_format=pyneat.PixelFormat.RGB
      )
    return run.push("input", [sample])

  try:
    consecutive_rejections = 0
    saturation_deadline = time.monotonic() + 5
    while time.monotonic() < saturation_deadline:
      if try_push():
        consecutive_rejections = 0
      else:
        consecutive_rejections += 1
        if consecutive_rejections == 10:
          break
        time.sleep(0.01)
    if consecutive_rejections < 10:
      raise AssertionError("failed to fill the run input queue")

    consumer_started = threading.Event()
    consumer_output: list[object] = []

    def consume_after_delay() -> None:
      consumer_started.set()
      time.sleep(0.2)
      consumer_output.append(run.pull("output", 5000))

    consumer = threading.Thread(target=consume_after_delay)
    consumer.start()
    assert consumer_started.wait(1)

    started = time.monotonic()
    assert push()
    elapsed = time.monotonic() - started

    consumer.join(5)
    assert not consumer.is_alive()
    assert consumer_output and consumer_output[0] is not None
    assert elapsed >= 0.1, "push did not encounter the established backpressure"
  finally:
    run.close()

  print(_CHILD_COMPLETE)


def test_blocking_run_push_releases_gil() -> None:
  env = os.environ.copy()
  env.pop("GST_PLUGIN_SYSTEM_PATH_1_0", None)

  for call_shape in _CALL_SHAPES:
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child", call_shape],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
        env=env,
    )

    assert result.returncode == 0, f"{call_shape}: {result.stderr}"
    assert _CHILD_COMPLETE in result.stdout


if __name__ == "__main__":
  if len(sys.argv) != 3 or sys.argv[1] != "--child" or sys.argv[2] not in _CALL_SHAPES:
    raise SystemExit("expected --child <samples|tensors|generic|named>")
  _exercise_blocking_push(sys.argv[2])
