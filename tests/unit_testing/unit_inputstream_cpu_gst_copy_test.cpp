#include "gst/GstSampleAttributes.h"
#include "nodes/io/Input.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "test_main.h"

#include <gstsimaaitensorbuffer_meta.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace {

using namespace simaai::neat;
constexpr const char* kCopyEnv = "SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY";
constexpr int kWidth = 8;
constexpr int kHeight = 4;

class CopyPermission {
public:
  explicit CopyPermission(bool enabled) {
    if (const char* value = std::getenv(kCopyEnv)) {
      previous_ = value;
    }
    require(::setenv(kCopyEnv, enabled ? "1" : "0", 1) == 0, "set copy permission");
  }
  ~CopyPermission() {
    if (previous_) {
      ::setenv(kCopyEnv, previous_->c_str(), 1);
    } else {
      ::unsetenv(kCopyEnv);
    }
  }

private:
  std::optional<std::string> previous_;
};

struct SampleUnref {
  void operator()(GstSample* sample) const {
    if (sample) {
      gst_sample_unref(sample);
    }
  }
};
using SamplePtr = std::unique_ptr<GstSample, SampleUnref>;

struct Layout {
  const char* name;
  GstVideoFormat format;
  int stride;
  std::size_t uv_offset;
  bool video_meta;
  bool split_memory;

  bool nv12() const {
    return format == GST_VIDEO_FORMAT_NV12;
  }
  std::size_t bytes() const {
    return nv12() ? uv_offset + stride * (kHeight / 2) : stride * kHeight;
  }
};

PreprocessRuntimeMeta source_preprocess() {
  PreprocessRuntimeMeta meta;
  meta.original_width = kWidth;
  meta.original_height = kHeight;
  meta.resized_width = kWidth;
  meta.resized_height = kHeight;
  meta.scaled_width = kWidth;
  meta.scaled_height = kHeight;
  meta.resize_mode = "stretch";
  meta.color_in = "BGR";
  meta.color_out = "RGB";
  meta.axis_perm = {0, 1, 2};
  return meta;
}

struct InputFrame {
  Sample sample;
  std::vector<std::uint8_t> bytes;
  GstMemory* memory = nullptr; // Identity only; sample owns the reference.
  std::shared_ptr<std::atomic<bool>> released = std::make_shared<std::atomic<bool>>(false);
};

InputFrame make_frame(const Layout& layout, bool source_roi) {
  InputFrame frame;
  frame.bytes.assign(layout.bytes(), 0xEA);
  for (int plane = 0; plane < (layout.nv12() ? 2 : 1); ++plane) {
    const auto offset = plane == 0 ? 0U : layout.uv_offset;
    const int rows = plane == 0 ? kHeight : kHeight / 2;
    for (int row = 0; row < rows; ++row) {
      for (int col = 0; col < layout.stride; ++col) {
        frame.bytes[offset + row * layout.stride + col] =
            static_cast<std::uint8_t>(17 + plane * 80 + row * 19 + col);
      }
    }
  }

  GstBuffer* buffer = gst_buffer_new();
  require(buffer != nullptr, "create source buffer");
  // Fill individual memories before appending: gst_buffer_map would merge a split fixture.
  const std::vector<std::size_t> ends =
      layout.split_memory ? std::vector<std::size_t>{layout.uv_offset, layout.bytes()}
                          : std::vector<std::size_t>{layout.bytes()};
  std::size_t begin = 0;
  for (const auto end : ends) {
    GstMemory* memory = gst_allocator_alloc(nullptr, end - begin, nullptr);
    require(memory != nullptr, "allocate source memory");
    GstMapInfo map{};
    require(gst_memory_map(memory, &map, GST_MAP_WRITE), "map source memory");
    std::memcpy(map.data, frame.bytes.data() + begin, end - begin);
    gst_memory_unmap(memory, &map);
    gst_buffer_append_memory(buffer, memory);
    begin = end;
  }
  frame.memory = gst_buffer_peek_memory(buffer, 0);
  require(!gst_is_dmabuf_memory(frame.memory), "fixture must be real system memory");
  gst_mini_object_weak_ref(
      GST_MINI_OBJECT(buffer),
      [](gpointer data, GstMiniObject*) {
        std::unique_ptr<std::shared_ptr<std::atomic<bool>>> released(
            static_cast<std::shared_ptr<std::atomic<bool>>*>(data));
        (*released)->store(true);
      },
      new std::shared_ptr<std::atomic<bool>>(frame.released));

  if (layout.video_meta) {
    gsize offsets[GST_VIDEO_MAX_PLANES] = {0, layout.uv_offset};
    gint strides[GST_VIDEO_MAX_PLANES] = {layout.stride, layout.stride};
    require(gst_buffer_add_video_meta_full(buffer, GST_VIDEO_FRAME_FLAG_NONE, layout.format, kWidth,
                                           kHeight, layout.nv12() ? 2U : 1U, offsets,
                                           strides) != nullptr,
            "attach source video layout");
  }
  auto preprocess = source_preprocess();
  if (source_roi) {
    // The enabled-empty override must clear a real source ROI, not merely preserve emptiness.
    preprocess.roi_list_enabled = true;
    preprocess.rois = {{0, 0, 0, 2, 2}};
    preprocess.roi_capacity = preprocess.roi_valid_count = preprocess.roi_input_count = 1;
    preprocess.roi_input_batch_size = 1;
    preprocess.roi_source_width = kWidth;
    preprocess.roi_source_height = kHeight;
    preprocess.roi_source_stride_bytes = layout.stride;
  }
  require(write_simaai_preprocess_meta(buffer, preprocess), "write source preprocess");
  GST_BUFFER_PTS(buffer) = 1;
  GST_BUFFER_DTS(buffer) = 2;
  GST_BUFFER_DURATION(buffer) = 3;
  GstCaps* caps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING,
                                      gst_video_format_to_string(layout.format), "width",
                                      G_TYPE_INT, kWidth, "height", G_TYPE_INT, kHeight, nullptr);
  SamplePtr source(gst_sample_new(buffer, caps, nullptr, nullptr));
  gst_caps_unref(caps);
  gst_buffer_unref(buffer);
  require(source != nullptr, "create source GstSample");
  frame.sample = sample_from_tensors({from_gst_sample(source.get())});
  frame.sample.frame_id = 41;
  frame.sample.input_seq = 43;
  frame.sample.orig_input_seq = 47;
  frame.sample.stream_id = "camera-1213";
  frame.sample.pts_ns = 100000;
  frame.sample.dts_ns = 90000;
  frame.sample.duration_ns = 33000;
  frame.sample.attributes = {{"frame-token", "raw-gst-1213"}};
  return frame;
}

// This exercises the real InputStream dispatcher, allocator and copier, without a worker
// consuming appsink. SystemMemory is an explicit synthetic destination for the host-only
// routing test; only --device requests Ev74 and proves actual DMA-backed publication.
class StreamFixture {
public:
  StreamFixture(const Sample& sample, bool require_device, bool device_allocator) {
    // Canonical CTest disables system-plugin discovery. Use the actual linked app
    // GTypes instead of depending on a cached registry containing the app plugin.
    GstElement* pipeline = gst_pipeline_new("raw-gst-copy");
    GstElement* source = GST_ELEMENT(g_object_new(GST_TYPE_APP_SRC, "name", "source", nullptr));
    sink_ = GST_ELEMENT(g_object_new(GST_TYPE_APP_SINK, "name", "sink", nullptr));
    require(pipeline && source && sink_, "create InputStream pipeline");
    g_object_set(source, "is-live", TRUE, "format", GST_FORMAT_TIME, "block", TRUE, nullptr);
    g_object_set(sink_, "sync", FALSE, "max-buffers", 1U, "drop", FALSE, "enable-last-sample",
                 FALSE, nullptr);
    gst_bin_add_many(GST_BIN(pipeline), source, sink_, nullptr);
    require(gst_element_link(source, sink_), "link InputStream pipeline");
    gst_object_ref(source);
    gst_object_ref(sink_);
    const SampleSpec spec = derive_sample_spec_or_throw(sample);
    GstCaps* caps = caps_from_spec(spec);
    gst_app_src_set_caps(GST_APP_SRC(source), caps);
    gst_caps_unref(caps);
    InputOptions input;
    input.memory_policy =
        device_allocator ? InputMemoryPolicy::Ev74 : InputMemoryPolicy::SystemMemory;
    input.do_timestamp = false;
    InputStreamOptions options;
    options.copy_input = false;
    options.require_device_visible_input = require_device;
    options.teardown_policy = pipeline_internal::InputStreamTeardownPolicy::MustReachNull;
    stream = InputStream::create(pipeline, source, sink_, spec, input, options, {}, nullptr);
    require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
            "start InputStream pipeline");
  }

  SamplePtr pull(GstClockTime timeout = GST_SECOND) {
    return SamplePtr(gst_app_sink_try_pull_sample(GST_APP_SINK(sink_), timeout));
  }

  InputStream stream;

private:
  GstElement* sink_ = nullptr; // Owned by stream.
};

void check_metadata(GstBuffer* buffer, const Sample& expected) {
  Sample restored;
  restore_sample_timing_from_gst_buffer(buffer, &restored);
  require(restored.frame_id == expected.frame_id && restored.pts_ns == expected.pts_ns &&
              restored.dts_ns == expected.dts_ns && restored.duration_ns == expected.duration_ns,
          "explicit Sample timing/frame ID must override source GstBuffer values");
  GstCustomMeta* custom = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  require(custom != nullptr, "missing destination GstSimaMeta");
  const GstStructure* meta = gst_custom_meta_get_structure(custom);
  gint64 input_seq = -1;
  gint64 orig_input_seq = -1;
  const char* stream_id = gst_structure_get_string(meta, "stream-id");
  require(gst_structure_get_int64(meta, "input-seq", &input_seq) &&
              gst_structure_get_int64(meta, "orig-input-seq", &orig_input_seq) &&
              input_seq == expected.input_seq && orig_input_seq == expected.orig_input_seq &&
              stream_id && expected.stream_id == stream_id,
          "Sample sequence/stream identity changed");
  gst_internal::read_attributes(buffer, &restored.attributes);
  require(restored.attributes == expected.attributes, "Sample attributes changed");
}

enum class PreprocessCase { Source, AxisOverride, RoiOverride, EmptyRoiOverride };

void apply_override(Sample& sample, PreprocessCase mode, const Layout& layout) {
  if (mode == PreprocessCase::Source) {
    return;
  }
  auto& meta = *sample.tensors.front().semantic.preprocess;
  meta.axis_perm = {2, 0, 1};
  meta.original_width = 123; // Source geometry remains authoritative unless ROI-list overrides it.
  if (mode == PreprocessCase::RoiOverride || mode == PreprocessCase::EmptyRoiOverride) {
    meta.roi_list_enabled = true;
    meta.roi_input_batch_size = 1;
    meta.roi_source_width = kWidth;
    meta.roi_source_height = kHeight;
    meta.roi_source_stride_bytes = layout.stride;
    meta.roi_pad_value = 114;
    if (mode == PreprocessCase::RoiOverride) {
      meta.rois = {{0, 1, 0, 4, 2}};
    } else {
      meta.rois.clear();
    }
    meta.roi_affines.clear();
    meta.roi_capacity = meta.roi_valid_count = meta.roi_input_count =
        static_cast<int>(meta.rois.size());
  }
}

void check_preprocess(GstBuffer* buffer, PreprocessCase mode) {
  const auto meta = read_simaai_preprocess_meta(buffer);
  require(meta.has_value(), "source preprocess metadata disappeared");
  const bool roi_override =
      mode == PreprocessCase::RoiOverride || mode == PreprocessCase::EmptyRoiOverride;
  require(meta->original_width == (roi_override ? 123 : kWidth), "preprocess precedence changed");
  require(meta->axis_perm == (mode == PreprocessCase::Source ? std::vector<int>{0, 1, 2}
                                                             : std::vector<int>{2, 0, 1}),
          "preprocess axis permutation changed");
  require(meta->roi_list_enabled == roi_override, "ROI-list enable override lost");
  require(meta->rois.size() == (mode == PreprocessCase::RoiOverride ? 1U : 0U),
          "ROI-list override (including enabled-empty) lost");
  if (mode == PreprocessCase::RoiOverride) {
    const auto& roi = meta->rois.front();
    require(roi.batch_index == 0 && roi.x == 1 && roi.y == 0 && roi.width == 4 && roi.height == 2,
            "ROI coordinates changed");
  }
}

void check_payload(GstSample* output, const Layout& layout, const std::vector<std::uint8_t>& bytes,
                   bool copied) {
  GstBuffer* buffer = gst_sample_get_buffer(output);
  const GstVideoMeta* video = gst_buffer_get_video_meta(buffer);
  require(video && video->format == layout.format && video->width == kWidth &&
              video->height == kHeight && video->stride[0] == layout.stride,
          "destination video geometry/row padding changed");
  if (layout.nv12()) {
    require(video->n_planes == 2 && video->stride[1] == layout.stride &&
                video->offset[1] == layout.uv_offset,
            "NV12 global plane offset/stride changed");
  }
  const Tensor tensor = from_gst_sample(output);
  // The actual Tensor mapping owns the DMA READ scope in --device mode.
  const Mapping mapping = tensor.storage->map(MapMode::Read);
  require(mapping.data && mapping.size_bytes >= layout.bytes(), "destination span is too small");
  auto expected = bytes;
  if (copied && layout.nv12()) {
    std::fill(expected.begin() + layout.stride * kHeight, expected.begin() + layout.uv_offset, 0);
  }
  require(std::memcmp(mapping.data, expected.data(), expected.size()) == 0,
          "payload, row padding or inter-plane gap differs");
}

void run_case(const Layout& layout, bool device_allocator, bool require_device,
              PreprocessCase preprocess) {
  std::cout << "case=" << layout.name << " require_device=" << require_device
            << " preprocess=" << static_cast<int>(preprocess) << '\n';
  InputFrame frame = make_frame(layout, preprocess == PreprocessCase::EmptyRoiOverride);
  apply_override(frame.sample, preprocess, layout);
  StreamFixture fixture(frame.sample, require_device, device_allocator);
  if (require_device) {
    CopyPermission disabled(false);
    bool rejected = false;
    try {
      (void)fixture.stream.try_push_message(frame.sample);
    } catch (const std::exception& error) {
      const std::string message = error.what();
      rejected = message.find("CPU-backed") != std::string::npos &&
                 message.find("tensor.cvu()") != std::string::npos &&
                 message.find("OutputPlacement::EV74Compatible") != std::string::npos;
    }
    require(rejected, "flag-off device ingress must reject CPU storage with actionable guidance");
    require(!fixture.pull(0), "rejected CPU input must not reach appsink");
  }
  CopyPermission permission(require_device);
  require(fixture.stream.try_push_message(frame.sample), "raw GstSample push failed");
  SamplePtr output = fixture.pull();
  require(output != nullptr, "raw GstSample pull timed out");
  GstBuffer* buffer = gst_sample_get_buffer(output.get());
  GstMemory* memory = gst_buffer_peek_memory(buffer, 0);
  require((memory != frame.memory) == require_device, "wrong holder-versus-copy selection");
  require(static_cast<bool>(gst_is_dmabuf_memory(memory)) == (require_device && device_allocator),
          "destination memory domain does not match the selected allocator");
  check_metadata(buffer, frame.sample);
  check_preprocess(buffer, preprocess);
  frame.sample = {};
  require(frame.released->load() == require_device,
          "copy should release source; zero-copy should retain its source holder");
  check_payload(output.get(), layout, frame.bytes, require_device);

  if (device_allocator && require_device && !layout.video_meta) {
    // A valid device holder must still forward the exact memory with copy permission OFF.
    CopyPermission disabled(false);
    Sample device_sample = sample_from_tensors({from_gst_sample(output.get())});
    StreamFixture device_fixture(device_sample, true, true);
    require(device_fixture.stream.try_push_message(device_sample), "DMA holder push failed");
    SamplePtr forwarded = device_fixture.pull();
    require(forwarded &&
                gst_buffer_peek_memory(gst_sample_get_buffer(forwarded.get()), 0) == memory,
            "valid DMA holder must remain zero-copy");
    check_payload(forwarded.get(), layout, frame.bytes, false);
  }
}

} // namespace

int main(int argc, char** argv) {
  return sima_test::run_test("unit_inputstream_cpu_gst_copy_test", [&] {
    require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--device"), "expected [--device]");
    const bool device_allocator = argc == 2;
    // This fixture needs appsrc/appsink and the real transport metadata, not unrelated
    // inference factories required by the complete Graph runtime bootstrap.
    gst_init(nullptr, nullptr);
    require(simaai::gst::tensor_buffer_register_meta(), "register Scheduler transport metadata");
    std::cout << (device_allocator
                      ? "real Ev74 DMA allocation mode\n"
                      : "HOST dispatch only: explicit synthetic SystemMemory destination\n");
    const Layout bgr{"BGR-tight", GST_VIDEO_FORMAT_BGR, 24, 0, false, false};
    const Layout nv12{"NV12-tight", GST_VIDEO_FORMAT_NV12, 8, 32, false, false};
    for (const auto& layout : {bgr, nv12}) {
      run_case(layout, device_allocator, true, PreprocessCase::Source);
      run_case(layout, device_allocator, false, PreprocessCase::Source);
    }
    run_case({"BGR-padded", GST_VIDEO_FORMAT_BGR, 32, 0, true, false}, device_allocator, true,
             PreprocessCase::AxisOverride);
    run_case({"NV12-padded-global-offset", GST_VIDEO_FORMAT_NV12, 12, 64, true, false},
             device_allocator, true, PreprocessCase::RoiOverride);
    run_case({"NV12-split-global-offset", GST_VIDEO_FORMAT_NV12, 12, 64, true, true},
             device_allocator, true, PreprocessCase::EmptyRoiOverride);
  });
}
