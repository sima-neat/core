"""Phase 4 (plan slice): caps / custom / common-media node factories.

Node factories are build-time descriptions (no pipeline run), so these construct real Node objects
and check graph composition without needing a model or the runtime. Encodes the S5 correction
(nodes.custom returns a connect()-able Node, distinct from the linear-only Graph.custom) and the
S10 decision (format_filter friendly alias over caps_raw; caps_raw / caps_nv12_sys_mem raw parity).
"""

from __future__ import annotations

import model_fixture_helpers as model_fixtures
import pyneat


NEW_FACTORIES = (
    "custom",
    "caps_raw",
    "format_filter",
    "caps_nv12_sys_mem",
    "camera_input",
    "file_input",
    "image_decode",
    "jpeg_decode",
    "video_scale",
    "video_rate",
    "image_freeze",
    "video_track_select",
    "still_image_input",
    "sima_render",
    "sima_argmax",
    "cast",
    "cast_tess",
    "dequant",
    "detess",
    "detess_cast",
)


def test_phase4_node_factories_present():
  for name in NEW_FACTORIES:
    assert hasattr(pyneat.nodes, name), name
  # caps_i420 is deferred (S8/S10) — must not be bound yet.
  assert not hasattr(pyneat.nodes, "caps_i420")


def test_caps_and_media_nodes_construct():
  # Each returns a real Node (build-time description). still_image_input reads an image file at
  # construction, so it is exercised separately.
  built = [
      pyneat.nodes.caps_raw("NV12", width=640, height=640, fps=30),
      pyneat.nodes.caps_raw("RGB"),
      pyneat.nodes.format_filter(format="NV12"),
      pyneat.nodes.caps_nv12_sys_mem(1920, 1080, 30),
      pyneat.nodes.camera_input(),
      pyneat.nodes.file_input("input.mp4"),
      pyneat.nodes.image_decode(),
      pyneat.nodes.jpeg_decode(),
      pyneat.nodes.video_scale(),
      pyneat.nodes.video_rate(),
      pyneat.nodes.image_freeze(),
      pyneat.nodes.image_freeze(num_buffers=1),
      pyneat.nodes.video_track_select(),
      pyneat.nodes.video_track_select(video_pad_index=1),
  ]
  for node in built:
    assert isinstance(node, pyneat.Node)


def test_camera_input_options_roundtrip():
  opt = pyneat.CameraInputOptions()
  assert opt.width == 1920
  assert opt.height == 1080
  assert opt.framerate_num == 30
  assert opt.framerate_den == 1
  assert opt.format == "NV12"
  assert opt.buffer_name == "camera"
  assert opt.insert_queue is True
  assert opt.leaky_queue is True
  assert opt.queue_depth == 2
  assert opt.allow_cpu_fallback is False

  opt.camera_name = "imx477 5-001a"
  opt.width = 1280
  opt.height = 720
  opt.framerate_num = 60
  opt.framerate_den = 1
  opt.format = "NV12"
  opt.buffer_name = "camera0"
  opt.insert_queue = False
  opt.leaky_queue = False
  opt.queue_depth = 4
  opt.allow_cpu_fallback = True

  assert opt.camera_name == "imx477 5-001a"
  assert opt.width == 1280
  assert opt.height == 720
  assert opt.framerate_num == 60
  assert opt.buffer_name == "camera0"
  assert opt.insert_queue is False
  assert opt.leaky_queue is False
  assert opt.queue_depth == 4
  assert opt.allow_cpu_fallback is True
  opt.camera_name = None
  assert opt.camera_name is None
  assert isinstance(pyneat.nodes.camera_input(opt), pyneat.Node)


def test_format_filter_accepts_memory_enum():
  node = pyneat.nodes.format_filter("NV12", memory=pyneat.CapsMemory.Any)
  assert isinstance(node, pyneat.Node)


def test_custom_node_is_connectable_vertex():
  # S5: nodes.custom returns a Node object usable as a graph vertex (connect()-able), which the
  # linear-only Graph.custom method cannot express. Graph.custom remains a fluent method.
  node = pyneat.nodes.custom(
      "videotestsrc num-buffers=1 is-live=false ! "
      "video/x-raw,format=RGB,width=16,height=16,framerate=1/1",
      pyneat.InputRole.Source,
  )
  assert isinstance(node, pyneat.Node)
  graph = pyneat.Graph()
  assert graph.add(node) is graph
  assert "videotestsrc" in graph.describe_backend()
  # The two APIs are distinct surfaces, not duplicates.
  assert callable(pyneat.Graph.custom)


def test_custom_defaults_to_internal_role():
  node = pyneat.nodes.custom("identity")
  assert isinstance(node, pyneat.Node)


def test_still_image_input_signature():
  # Present and bound with plain-int geometry (strong-typedef args wrapped). Not constructed here
  # because the C++ constructor loads/encodes the image file at build time.
  assert hasattr(pyneat.nodes, "still_image_input")


def test_sima_render_and_argmax_nodes():
  # S6: sima_render is bound so the bbox-overlay power isn't stranded by the deferred UdpOutputGroupG.
  render_opt = pyneat.SimaRenderOptions()
  render_opt.transmit = True
  render_opt.silent = False
  assert render_opt.transmit is True
  assert isinstance(pyneat.nodes.sima_render(), pyneat.Node)
  assert isinstance(pyneat.nodes.sima_render(render_opt), pyneat.Node)

  argmax_opt = pyneat.SimaArgMaxOptions()
  argmax_opt.sima_allocator_type = 2
  assert isinstance(pyneat.nodes.sima_argmax(), pyneat.Node)
  assert isinstance(pyneat.nodes.sima_argmax(argmax_opt), pyneat.Node)


def test_cvu_atom_options_roundtrip():
  cast = pyneat.CastOptions()
  cast.direction = pyneat.CastDirection.Fp32ToBf16
  cast.num_buffers = 4
  assert cast.direction == pyneat.CastDirection.Fp32ToBf16
  assert cast.num_buffers == 4

  dq = pyneat.DequantOptions()
  dq.model_managed = True
  dq.q_scale = 0.5
  dq.q_zp = 3
  assert dq.model_managed is True
  assert dq.q_scale == 0.5
  assert dq.q_zp == 3

  for cls in (pyneat.CastTessOptions, pyneat.DetessOptions, pyneat.DetessCastOptions):
    opt = cls()
    opt.num_buffers = 2
    assert opt.num_buffers == 2


def test_cvu_atom_nodes_construct_standalone():
  # The forms that don't need model-managed params construct standalone.
  dq_opt = pyneat.DequantOptions()
  dq_opt.q_scale = 0.5
  dq_opt.q_zp = 0
  built = [
      pyneat.nodes.cast(),
      pyneat.nodes.cast(pyneat.CastOptions()),
      pyneat.nodes.cast_tess(),
      pyneat.nodes.dequant(dq_opt),
      pyneat.nodes.detess(),
  ]
  for node in built:
    assert isinstance(node, pyneat.Node)


def test_dequant_requires_quant_params():
  import pytest

  # Standalone dequant requires explicit quant params unless model-managed — the binding surfaces
  # the C++ validation rather than silently producing a broken node.
  with pytest.raises(RuntimeError, match="q_scale"):
    pyneat.nodes.dequant()


def test_cvu_atom_from_model_constructors_registered():
  # from-Model constructors pull model-managed params (geometry/quant/buffer counts) so atoms like
  # detess_cast — which require model-managed num_buffers — become usable. The exact stage set is
  # route-specific (resnet50 = quanttess/detessdequant), so a given atom may legitimately not be in
  # this model's route; the constructor then raises a clear RuntimeError. What must hold is that the
  # (Model) overload is registered (no TypeError) and, when it succeeds, yields a buildable node.
  path = model_fixtures.strict_model_tar_path("SIMA_RESNET50_TAR")
  model = pyneat.Model(str(path))
  atoms = {
      pyneat.CastTessOptions: pyneat.nodes.cast_tess,
      pyneat.DequantOptions: pyneat.nodes.dequant,
      pyneat.DetessOptions: pyneat.nodes.detess,
      pyneat.DetessCastOptions: pyneat.nodes.detess_cast,
  }
  for opt_cls, factory in atoms.items():
    try:
      opt = opt_cls(model)
    except RuntimeError:
      continue  # stage not in this model's route — overload exists and validated correctly
    except TypeError as exc:  # pragma: no cover - registration regression guard
      raise AssertionError(f"{opt_cls.__name__}(Model) overload not registered: {exc}")
    assert isinstance(factory(opt), pyneat.Node)
