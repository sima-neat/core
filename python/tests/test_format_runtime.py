"""Phase 6 slice: pyneat.Format vocabulary (S8) + runtime warm-up / build info."""

from __future__ import annotations

import pytest

import pyneat


def test_format_enum_members():
  fmt = pyneat.Format
  for member in ("Auto", "RGB", "BGR", "GRAY8", "NV12", "I420", "YUYV", "ENCODED", "H264",
                 "ByteStream", "FP32", "INT8", "UINT8", "BF16"):
    assert hasattr(fmt, member), member
  # S8: caps-layer / EV aliases are NOT exposed as enum members.
  for excluded in ("MLA", "BBOX", "ARGMAX", "DETESSDEQUANT", "EVXX_FLOAT32", "EVXX_INT8",
                   "EVXX_BFLOAT16"):
    assert not hasattr(fmt, excluded), excluded


def test_format_fields_use_format_enum():
  """FormatSpec option fields are surfaced as the pyneat.Format enum (issue #337).

  Assignment takes a pyneat.Format value (not a string); reads return the enum.
  """
  rtsp = pyneat.RtspDecodedInputOptions()
  rtsp.out_format = pyneat.Format.NV12
  assert rtsp.out_format == pyneat.Format.NV12
  rtsp.output_caps.format = pyneat.Format.RGB
  assert rtsp.output_caps.format == pyneat.Format.RGB

  video = pyneat.VideoInputGroupOptions()
  video.out_format = pyneat.Format.I420
  assert video.out_format == pyneat.Format.I420

  inp = pyneat.InputOptions()
  inp.format = pyneat.Format.BGR
  assert inp.format == pyneat.Format.BGR

  out = pyneat.OutputTensorOptions()
  out.format = pyneat.Format.FP32
  assert out.format == pyneat.Format.FP32

  # Strings are rejected — callers use the typed enum, not a token.
  with pytest.raises(TypeError):
    rtsp.out_format = "NV12"
  with pytest.raises(TypeError):
    inp.format = "RGB"


def test_format_tag_alias():
  # FormatTag is retained as a canonical alias of the friendly Format.
  assert pyneat.FormatTag is pyneat.Format


def test_format_codec_aliases():
  # Same codec spellings as RtspCodec and SimaDecodeType.
  assert pyneat.Format.AVC == pyneat.Format.H264
  assert pyneat.Format.HEVC == pyneat.Format.H265
  assert pyneat.FormatTag.HEVC == pyneat.Format.H265
  # Aliases stay assignable and keep the canonical serialized spelling.
  rtsp = pyneat.RtspDecodedInputOptions()
  rtsp.out_format = pyneat.Format.HEVC
  assert rtsp.out_format == pyneat.Format.H265
  assert pyneat.advanced.format_tag_name(pyneat.Format.HEVC) == "H265"


def test_format_converters_in_advanced_tier():
  adv = pyneat.advanced
  assert adv.format_tag_name(pyneat.Format.NV12) == "NV12"
  assert adv.format_tag_from_string("NV12") == pyneat.Format.NV12
  assert adv.format_tag_name(pyneat.Format.H265) == "H265"
  assert adv.format_tag_from_string("H265") == pyneat.Format.H265
  assert adv.is_raw_video_format(pyneat.Format.NV12) is True
  assert adv.is_tensor_payload_format(pyneat.Format.FP32) is True
  assert adv.is_raw_video_format(pyneat.Format.FP32) is False
  # Converters are advanced-only, not leaked to the top level.
  assert not hasattr(pyneat, "format_tag_name")


def test_build_info():
  info = pyneat.build_info()
  assert set(info.keys()) >= {"version", "platform_version", "abi_version"}
  assert isinstance(info["version"], str)
  assert info["platform_version"]  # non-empty


def test_prewarm_runtime_callable():
  # Idempotent warm-up; must not raise. (prewarm_runtime_async is intentionally not bound.)
  assert pyneat.prewarm_runtime() is None
  assert not hasattr(pyneat, "prewarm_runtime_async")


def test_make_encoded_sample():
  data = b"\x00\x00\x00\x01\x67"
  sample = pyneat.make_encoded_sample(data, "video/x-h264", pts_ns=1000)
  assert isinstance(sample, pyneat.Sample)
  assert sample.pts_ns == 1000
  assert sample.caps_string == "video/x-h264"
  assert bool(sample)  # carries a payload


def test_sample_from_encoded_wrapper():
  sample = pyneat.Sample.from_encoded(
      b"\x00\x01\x02", "video/x-h264", pts_ns=5, port_name="enc_in"
  )
  assert isinstance(sample, pyneat.Sample)
  assert sample.pts_ns == 5
  assert sample.port_name == "enc_in"
  assert sample.caps_string == "video/x-h264"


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
