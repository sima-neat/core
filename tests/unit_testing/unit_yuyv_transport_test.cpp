#include "pipeline/Tensor.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/GstDataAdapter.h"
#include "nodes/io/Input.h"
#include <gst/app/gstappsrc.h>
#include <gst/app/gstappsink.h>
#include "pipeline/TensorAdapters.h"
#include "pipeline/FormatSpec.h"
#include "nodes/io/UsbCameraInput.h"
#include "nodes/sima/SimaEncode.h"
#include <gst/video/video.h>
#include <gstsimaaitensorbuffer_meta.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

using namespace simaai::neat;
namespace {
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
void raw_caps_defaults() {
  SampleSpec implicit;
  implicit.kind = SampleMediaKind::RawVideo;
  implicit.media_type = "video/x-raw";
  implicit.format = "NV12";
  implicit.width = 18;
  implicit.height = 10;
  GstCaps* caps = gst_caps_from_string(caps_string_from_spec(implicit).c_str());
  GstVideoInfo info{};
  require(gst_video_info_from_caps(&info, caps) &&
              GST_VIDEO_INFO_INTERLACE_MODE(&info) == GST_VIDEO_INTERLACE_MODE_PROGRESSIVE &&
              info.par_n == 1 && info.par_d == 1,
          "fixture must follow GStreamer's raw-video defaults");
  gst_caps_unref(caps);
  auto explicit_defaults = implicit;
  explicit_defaults.video.interlace_mode = "progressive";
  explicit_defaults.video.par_n = explicit_defaults.video.par_d = 1;
  require(capkey_from_spec(implicit) == capkey_from_spec(explicit_defaults),
          "spelling GStreamer defaults must not trigger renegotiation");
  auto changed = explicit_defaults;
  changed.video.colorimetry = "bt709";
  require(capkey_from_spec(implicit) != capkey_from_spec(changed),
          "unspecified color must not be guessed");
  changed = explicit_defaults;
  changed.video.interlace_mode = "interleaved";
  require(capkey_from_spec(implicit) != capkey_from_spec(changed),
          "interlaced input is not the progressive default");
  changed = explicit_defaults;
  changed.video.par_n = 2;
  require(capkey_from_spec(implicit) != capkey_from_spec(changed),
          "non-square pixels must retain distinct caps identity");
}
void packed_view(bool copy_input, int extra_capacity = 0) {
  constexpr int width = 6, height = 3, pitch = 20, offset = 7;
  constexpr int active_extent = offset + (height - 1) * pitch + width * 2;
  const int extent = active_extent + extra_capacity;
  GstBuffer* buffer = gst_buffer_new_allocate(nullptr, extent, nullptr);
  GstMapInfo source_map{};
  require(gst_buffer_map(buffer, &source_map, GST_MAP_WRITE), "map source fixture");
  for (std::size_t i = 0; i < source_map.size; ++i)
    source_map.data[i] = static_cast<uint8_t>(i);
  gst_buffer_unmap(buffer, &source_map);
  gsize offsets[GST_VIDEO_MAX_PLANES] = {offset};
  gint strides[GST_VIDEO_MAX_PLANES] = {pitch};
  require(gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE, GST_VIDEO_FORMAT_YUY2,
                                         width, height, 1, offsets, strides),
          "video meta");
  GstCaps* caps = gst_caps_from_string(
      "video/x-raw,format=YUY2,width=6,height=3,framerate=30000/"
      "1001,colorimetry=bt709,chroma-site=jpeg,interlace-mode=progressive,pixel-aspect-ratio=1/1");
  GstSample* sample = gst_sample_new(buffer, caps, nullptr, nullptr);
  Tensor tensor = from_gst_sample(sample);
  gst_sample_unref(sample);
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(tensor.semantic.image && tensor.semantic.image->format == ImageType::YUYV,
          "packed semantic format");
  require(tensor.shape == std::vector<int64_t>{height, width, 2}, "storage shape");
  require(tensor.strides_bytes == std::vector<int64_t>{pitch, 2, 1}, "negotiated stride");
  require(tensor.byte_offset == offset && tensor.storage->size_bytes == extent,
          "preserve offset and short final padding");
  require(tensor.storage->holder && tensor.storage->data == nullptr,
          "retained owner without eager CPU mapping");
  require(tensor.semantic.image->color_space == "bt709", "preserve color metadata");
  std::string error;
  require(tensor.validate(&error), error.c_str());
  auto message = sample_from_tensors(TensorList{tensor});
  auto spec = derive_sample_spec_or_throw(message);
  require(spec.format == "YUY2" && spec.depth == 2 && spec.required_bytes_actual == extent,
          "graph handoff must accept packed YUYV active extent");
  require(spec.planes.size() == 1 && spec.planes[0].offset_bytes == offset &&
              spec.planes[0].stride_bytes == pitch &&
              spec.planes[0].size_bytes == active_extent - offset,
          "graph handoff must preserve offset/pitch without requiring final row padding");
  require(spec.video.colorimetry == "bt709" && spec.video.chroma_site == "jpeg" &&
              spec.video.interlace_mode == "progressive" && spec.video.par_n == 1 &&
              spec.video.par_d == 1,
          "graph handoff must retain raw interpretation");
  GstCaps* rebuilt = caps_from_spec(spec);
  const auto* fields = gst_caps_get_structure(rebuilt, 0);
  require(std::string(gst_structure_get_string(fields, "colorimetry")) == "bt709" &&
              std::string(gst_structure_get_string(fields, "chroma-site")) == "jpeg",
          "rebuilt raw caps must retain color");
  GstCaps* serialized = gst_caps_from_string(spec.caps_string.c_str());
  require(gst_caps_is_equal(rebuilt, serialized), "caps string and object must agree");
  gst_caps_unref(serialized);
  auto changed = spec;
  changed.video.colorimetry = "bt601";
  changed.caps_key = capkey_from_spec(changed);
  changed.caps_string = caps_string_from_spec(changed);
  require(changed.caps_key != spec.caps_key, "color changes must invalidate the caps key");
  GstCaps* changed_caps = caps_from_spec(changed);
  require(std::string(gst_structure_get_string(gst_caps_get_structure(changed_caps, 0),
                                               "colorimetry")) == "bt601",
          "global caps cache must not alias different color interpretations");
  gst_caps_unref(changed_caps);
  for (int field = 0; field < 3; ++field) {
    auto other = spec;
    if (field == 0)
      other.video.chroma_site = "mpeg2";
    if (field == 1)
      other.video.interlace_mode = "interleaved";
    if (field == 2)
      other.video.par_n = 2;
    require(capkey_from_spec(other) != spec.caps_key, "raw interpretation is caps identity");
  }
  auto verify_pixels = [&](GstBuffer* output, bool copied) {
    const auto* meta = gst_buffer_get_video_meta(output);
    require(meta && meta->offset[0] == offset && meta->stride[0] == pitch,
            "copy must retain carrier-relative layout");
    GstMapInfo mapped{};
    require(gst_buffer_map(output, &mapped, GST_MAP_READ), "map output pixels");
    require(mapped.size == static_cast<std::size_t>(extent), "carrier size changed");
    for (int y = 0; y < height; ++y)
      for (int x = 0; x < width * 2; ++x) {
        const auto index = offset + y * pitch + x;
        require(mapped.data[index] == static_cast<uint8_t>(index), "copied pixel shifted");
      }
    if (copied)
      for (int i = 0; i < offset; ++i)
        require(mapped.data[i] == 0, "copy prefix must be initialized");
    gst_buffer_unmap(output, &mapped);
  };
  // The generic buffer builder has an independent explicit-copy caller.
  pipeline_internal::GstBufferBuildPolicy copy_policy;
  copy_policy.allow_zero_copy = false;
  copy_policy.require_video_meta = true;
  auto* copy = pipeline_internal::build_gst_buffer_from_tensor(tensor, spec, copy_policy, &error);
  require(copy, error.c_str());
  verify_pixels(copy, true);
  gst_buffer_unref(copy);

  // Exercise actual segmented-graph sharing and explicit-copy handoffs with
  // labeled system memory. This does not prove device admission.
  GstElement* pipeline = gst_pipeline_new("yuyv-handoff");
  GstElement* source = GST_ELEMENT(
      g_object_new(GST_TYPE_APP_SRC, "format", GST_FORMAT_TIME, "is-live", TRUE, nullptr));
  GstElement* sink = GST_ELEMENT(
      g_object_new(GST_TYPE_APP_SINK, "sync", FALSE, "enable-last-sample", FALSE, nullptr));
  gst_bin_add_many(GST_BIN(pipeline), source, sink, nullptr);
  require(gst_element_link(source, sink), "link handoff fixture");
  gst_object_ref(source);
  gst_object_ref(sink);
  gst_app_src_set_caps(GST_APP_SRC(source), rebuilt);
  gst_caps_unref(rebuilt);
  InputOptions input;
  input.memory_policy = InputMemoryPolicy::SystemMemory;
  InputStreamOptions options;
  options.copy_input = copy_input;
  options.require_device_visible_input = false;
  options.allow_graph_internal_zero_copy_input = true;
  options.teardown_policy = pipeline_internal::InputStreamTeardownPolicy::MustReachNull;
  auto stream = InputStream::create(pipeline, source, sink, spec, input, options, {}, nullptr);
  auto* retained = static_cast<GstSample*>(tensor.storage->holder.get());
  auto* original = gst_sample_get_buffer(retained);
  auto* memory = gst_buffer_peek_memory(original, 0);
  require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
          "start handoff fixture");
  require(stream.try_push_message(message), "push YUYV across graph boundary");
  auto* received = gst_app_sink_try_pull_sample(GST_APP_SINK(sink), GST_SECOND);
  require(received, "pull YUYV after graph boundary");
  auto* received_buffer = gst_sample_get_buffer(received);
  auto* received_meta = gst_buffer_get_video_meta(received_buffer);
  require((gst_buffer_peek_memory(received_buffer, 0) == memory) == !copy_input &&
              gst_buffer_get_size(received_buffer) == extent && received_meta &&
              received_meta->offset[0] == offset && received_meta->stride[0] == pitch,
          "graph handoff must honor copy policy and retain authoritative layout");
  verify_pixels(received_buffer, copy_input);
  const auto roundtrip = from_gst_sample(received);
  require(roundtrip.semantic.image->color_space == "bt709", "color survives actual appsrc");
  gst_sample_unref(received);
  auto* new_caps = gst_caps_copy(gst_sample_get_caps(retained));
  gst_caps_set_simple(new_caps, "colorimetry", G_TYPE_STRING, "bt601", nullptr);
  auto* new_sample = gst_sample_new(original, new_caps, nullptr, nullptr);
  auto new_message = sample_from_tensors(TensorList{from_gst_sample(new_sample)});
  gst_sample_unref(new_sample);
  bool color_rejected = false;
  try {
    (void)stream.try_push_message(new_message);
  } catch (const std::exception&) {
    color_rejected = true;
  }
  require(color_rejected, "static graph must reject color changes instead of reusing cached caps");
  new_caps = gst_caps_make_writable(new_caps);
  gst_caps_set_simple(new_caps, "colorimetry", G_TYPE_INT, 123, nullptr);
  new_sample = gst_sample_new(original, new_caps, nullptr, nullptr);
  bool malformed_rejected = false;
  try {
    (void)derive_sample_spec_or_throw(sample_from_tensors(TensorList{from_gst_sample(new_sample)}));
  } catch (const std::invalid_argument&) {
    malformed_rejected = true;
  }
  require(malformed_rejected, "wrong-typed color must not disappear into defaults");
  gst_sample_unref(new_sample);
  gst_caps_unref(new_caps);
  stream.close();
  auto reject_geometry = [&](const Tensor& bad) {
    bool rejected = false;
    try {
      (void)derive_sample_spec_or_throw(sample_from_tensors(TensorList{bad}));
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "malformed YUYV must fail graph admission");
  };
  auto bad = tensor;
  bad.shape[1] = 5;
  reject_geometry(bad);
  bad = tensor;
  bad.byte_offset = offset + extra_capacity + 1;
  reject_geometry(bad);
  bad = tensor;
  bad.strides_bytes[1] = 3;
  reject_geometry(bad);
  tensor.shape[1] = 5;
  require(!tensor.validate(&error), "odd width must fail");
  tensor.shape[1] = width;
  tensor.strides_bytes[0] = 4;
  require(!tensor.validate(&error), "overlapping rows must fail");
}
} // namespace
int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  try {
    require(simaai::gst::tensor_buffer_register_meta(), "register Scheduler transport metadata");
    raw_caps_defaults();
    // Old function-pointer forms remain unambiguous with explicit signatures.
    [[maybe_unused]] auto old_source =
        static_cast<std::shared_ptr<Node> (*)(UsbCameraInputOptions)>(&nodes::UsbCameraInput);
    [[maybe_unused]] auto old_info =
        static_cast<UsbCameraInfo (*)(const UsbCameraInputOptions&)>(&usb_camera_info);
    [[maybe_unused]] auto new_source =
        static_cast<std::shared_ptr<Node> (*)(UsbCameraInputOptions, UsbCameraFormat)>(
            &nodes::UsbCameraInput);
    require(format_tag_from_string("YUY2") == FormatTag::YUYV, "GStreamer alias");
    SimaEncodeOptions encode;
    const auto strict = nodes::SimaEncode(encode, EncoderInputPolicy::RequireDirect);
    require(strict->backend_fragment(0).find("require-direct=true") != std::string::npos,
            "strict encoder property");
    require(nodes::SimaEncode(encode)->backend_fragment(0).find("require-direct=true") ==
                std::string::npos,
            "legacy encoder policy");
    require(strict->backend_fragment(0).find("signal-input-color=true") != std::string::npos,
            "strict encoder must signal the negotiated color");
    encode.type = SimaEncodeType::H265;
    bool rejected = false;
    try {
      (void)nodes::SimaEncode(encode, EncoderInputPolicy::RequireDirect);
    } catch (const std::invalid_argument&) {
      rejected = true;
    }
    require(rejected, "unqualified strict codec must fail");
    packed_view(false);
    packed_view(true);
    packed_view(true, 21);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  std::cout << "PASS YUYV packed transport and additive API contracts\n";
}
