"""Core encoder and RTP sender lifecycle checks requiring the installed codec runtime."""

import gc
import re
import socket
import subprocess
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pyneat


def _mjpeg_encode_and_passthrough_retention():
  image = np.full((96, 160, 3), 96, dtype=np.uint8)
  tensor = pyneat.Tensor.from_numpy(
      image, copy=True, image_format=pyneat.PixelFormat.RGB,
      memory=pyneat.TensorMemory.CPU)
  seed = pyneat.Sample()
  seed.kind = pyneat.SampleKind.TensorSet
  seed.tensors = [tensor]
  inputs = pyneat.InputOptions()
  inputs.payload_type = pyneat.PayloadType.Image
  inputs.format = pyneat.Format.RGB
  inputs.width, inputs.height, inputs.fps_n, inputs.fps_d = 160, 96, 30, 1
  inputs.memory_policy = pyneat.InputMemoryPolicy.SystemMemory
  inputs.is_live, inputs.do_timestamp = True, False
  encode = pyneat.SimaEncodeOptions()
  encode.type = pyneat.SimaEncodeType.MJPEG
  encode.width, encode.height, encode.num_buffers = 160, 96, 4
  graph = pyneat.Graph("python-encoder")
  graph.add(pyneat.nodes.input(inputs))
  graph.add(pyneat.nodes.sima_encode(encode))
  graph.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(4)))
  options = pyneat.RunOptions()
  options.output_memory = pyneat.OutputMemory.ZeroCopy
  options.startup_preflight = False
  run = graph.build([seed], options)
  frames = []
  retained = None
  try:
    for index in range(6):
      seed.pts_ns = index * 1_000_000_000 // 30
      seed.dts_ns = seed.pts_ns
      seed.duration_ns = 1_000_000_000 // 30
      assert run.push([seed])
      out = run.pull(5000)
      assert out is not None and out.pts_ns == seed.pts_ns
      assert not out.owned, "retention requires an actual native output loan"
      assert "image/jpeg" in out.caps_string
      data = bytes(out.tensors[0].copy_payload_bytes())
      assert data.startswith(b"\xff\xd8") and data.endswith(b"\xff\xd9")
      frames.append(pyneat.make_encoded_sample(data, out.caps_string, pts_ns=out.pts_ns))
      if retained is None:
        retained, retained_data = out, data
      del out
    run.close_input()
    assert run.pull(5000) is None
  finally:
    run.close()
  assert bytes(retained.tensors[0].copy_payload_bytes()) == retained_data

  with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
    receiver.bind(("127.0.0.1", 0))
    receiver.settimeout(5)
    sender_options = pyneat.VideoSenderOptions.passthrough(pyneat.RtspCodec.MJPEG)
    sender_options.video_port_base = receiver.getsockname()[1]
    inputs.payload_type, inputs.format = pyneat.PayloadType.Encoded, pyneat.Format.ENCODED
    inputs.caps_override = frames[0].caps_string
    graph = pyneat.Graph("python-jpeg-sender")
    graph.add(pyneat.nodes.input(inputs))
    graph.add(pyneat.groups.video_sender(sender_options))
    run = graph.build([frames[0]], options)
    timestamps = []
    try:
      for frame in frames:
        assert run.push([frame])
        while True:
          packet, _ = receiver.recvfrom(65536)
          assert len(packet) >= 20 and packet[0] >> 6 == 2
          assert packet[1] & 127 == 26
          if packet[1] & 128:
            timestamps.append(int.from_bytes(packet[4:8], "big"))
            break
      run.close_input()
    finally:
      run.close()
    assert len(timestamps) == len(frames)
    expected_ticks = [frame.pts_ns * 90000 // 1_000_000_000 for frame in frames]
    assert all(abs(((timestamp - timestamps[0]) & 0xffffffff) - expected) <= 1
               for timestamp, expected in zip(timestamps, expected_ticks)), timestamps
    receiver.settimeout(0.2)
    try:
      receiver.recvfrom(65536)
    except socket.timeout:
      pass
    else:
      raise AssertionError("unexpected duplicate RTP output")


def _resources():
  descriptors = Counter()
  for fd in Path("/proc/self/fd").iterdir():
    try:
      target = str(fd.readlink())
    except FileNotFoundError:
      continue
    if re.match(r"/proc/\d+/fd", target):
      continue
    descriptors[re.sub(r"(socket|pipe):\[\d+\]", r"\1", target)] += 1
  dma = subprocess.check_output(
      ["sudo", "-n", "cat", "/sys/kernel/debug/dma_buf/bufinfo"], text=True)
  match = re.search(r"Total (\d+) objects, (\d+) bytes", dma)
  assert match, "DMA resource counters are unavailable"
  return descriptors, len(list(Path("/proc/self/task").iterdir())), match.groups()


def test_mjpeg_encode_sender_retention_and_resources():
  baseline = None
  for _ in range(3):
    _mjpeg_encode_and_passthrough_retention()
    gc.collect()
    fds, threads, dma = _resources()
    if baseline is None:
      baseline = fds, threads, dma
      continue
    growth = fds - baseline[0]
    logger = {path: count for path, count in growth.items()
              if path.endswith("/counter0/count0/count")}
    other = {path: count for path, count in growth.items() if path not in logger}
    if logger:
      warnings.warn(f"Platform logger FD leak remains present: {logger}", RuntimeWarning)
    assert not other, f"new file descriptors survived teardown: {other}"
    assert threads <= baseline[1], "encoder threads survived teardown"
    assert dma == baseline[2], "DMA buffers survived encoder teardown"
