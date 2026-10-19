"""Installed-runtime decoder checks requiring Modalix and packaged fixtures."""

import pyneat


def test_mjpeg_422_i420_retained_output():
  import json
  from pathlib import Path

  here = Path(__file__).resolve()
  manifests = (
      here.parents[2] / "build/test-fixtures/runtime_manifest.json",
      here.parents[4] / "lib/sima-neat/test-fixtures/runtime_manifest.json",
  )
  manifest = next((path for path in manifests if path.is_file()), None)
  assert manifest is not None, "missing installed decoder fixture manifest"
  assets = json.loads(manifest.read_text())
  fixtures = (manifest.parent.parent / assets["codec_perf_h264_fixture_rel"]).parent
  caps = "image/jpeg,width=160,height=96,framerate=10/1"
  frames = []
  references = []
  for index in range(4):
    sample = pyneat.make_encoded_sample(
        (fixtures / f"jpeg422_{index}.jpg").read_bytes(), caps,
        pts_ns=index * 100_000_000)
    sample.frame_id = index
    sample.attributes = {"fixture-frame": str(index)}
    frames.append(sample)
    references.append((fixtures / f"jpeg422_{index}.i420").read_bytes())

  input_options = pyneat.InputOptions()
  input_options.payload_type = pyneat.PayloadType.Encoded
  input_options.caps_override = caps
  input_options.memory_policy = pyneat.InputMemoryPolicy.SystemMemory
  decode = pyneat.SimaDecodeOptions()
  decode.type = pyneat.SimaDecodeType.MJPEG
  decode.out_format = pyneat.Format.I420
  decode.dec_width, decode.dec_height, decode.dec_fps = 160, 96, 10
  graph = pyneat.Graph("jpeg-422-retention")
  graph.add(pyneat.nodes.input(input_options))
  graph.add(pyneat.nodes.sima_decode(decode))
  graph.add(pyneat.nodes.output(pyneat.OutputOptions.every_frame(2)))
  options = pyneat.RunOptions()
  options.output_memory = pyneat.OutputMemory.ZeroCopy
  options.startup_preflight = False
  run = graph.build([frames[0]], options)

  def visible_pixels(sample):
    assert len(sample.tensors) == 1
    tensor = sample.tensors[0]
    assert tensor.is_i420() and len(tensor.planes) == 3
    assert (tensor.width(), tensor.height()) == (160, 96)
    pixels = tensor.contiguous().copy_payload_bytes()
    assert len(pixels) == 160 * 96 * 3 // 2
    return pixels

  retained = None
  try:
    for frame, reference in zip(frames, references):
      assert run.push([frame])
      out = run.pull(5000)
      assert out is not None, "missing decoded JPEG output"
      assert not out.owned
      assert out.pts_ns == frame.pts_ns
      assert out.frame_id == frame.frame_id
      assert out.attributes == frame.attributes
      pixels = visible_pixels(out)
      assert len(reference) == len(pixels)
      assert max(abs(a - b) for a, b in zip(pixels, reference)) <= 3
      if retained is None:
        retained, retained_pixels = out, pixels
      del out
    run.close_input()
  finally:
    run.close()
  assert retained is not None
  assert visible_pixels(retained) == retained_pixels
