"""Phase 6 slice: pyneat.Format vocabulary (S8) + runtime warm-up / build info."""

from __future__ import annotations

import pyneat


def test_format_enum_members():
  fmt = pyneat.Format
  for member in ("Auto", "RGB", "BGR", "GRAY8", "NV12", "I420", "YUYV", "ENCODED", "H264",
                 "ByteStream", "FP32", "INT8", "UINT8", "BF16"):
    assert hasattr(fmt, member), member
  # S8: caps-layer / EV aliases are NOT exposed as enum members (the string parser still accepts them).
  for excluded in ("MLA", "BBOX", "ARGMAX", "DETESSDEQUANT", "EVXX_FLOAT32", "EVXX_INT8",
                   "EVXX_BFLOAT16"):
    assert not hasattr(fmt, excluded), excluded


def test_format_tag_alias():
  # FormatTag is retained as a canonical alias of the friendly Format.
  assert pyneat.FormatTag is pyneat.Format


def test_format_converters_in_advanced_tier():
  adv = pyneat.advanced
  assert adv.format_tag_name(pyneat.Format.NV12) == "NV12"
  assert adv.format_tag_from_string("NV12") == pyneat.Format.NV12
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
