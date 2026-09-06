#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "gst/GstInit.h"
#include "pipeline/internal/InputStream.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <string>
#include <thread>

namespace {

using namespace simaai::neat;

void check_queue(const char* media, OverflowPolicy policy, bool stop_full) {
  const std::string launch =
      std::string("fakesrc num-buffers=12 sizetype=fixed sizemax=8 filltype=pattern ! ") + media +
      " ! appsink name=mysink sync=false max-buffers=2 drop=false enable-last-sample=false";
  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(launch.c_str(), &error);
  require(error == nullptr && pipeline != nullptr, "encoded queue fixture did not parse");
  GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
  GstPad* pad = gst_element_get_static_pad(sink, "sink");
  struct Buffers {
    std::atomic<guint8> produced{0};
    std::atomic<int> released{0};
  };
  using BufferOwner = std::shared_ptr<Buffers>;
  const auto buffers = std::make_shared<Buffers>();
  gst_pad_add_probe(
      pad, GST_PAD_PROBE_TYPE_BUFFER,
      +[](GstPad*, GstPadProbeInfo* info, gpointer data) {
        const auto& buffers = *static_cast<BufferOwner*>(data);
        GstBuffer* buffer = gst_buffer_make_writable(GST_PAD_PROBE_INFO_BUFFER(info));
        GST_PAD_PROBE_INFO_DATA(info) = buffer;
        const guint8 index = buffers->produced++;
        gst_buffer_fill(buffer, 0, &index, 1);
        GST_BUFFER_PTS(buffer) = index * GST_MSECOND;
        gst_mini_object_weak_ref(
            GST_MINI_OBJECT(buffer),
            +[](gpointer data, GstMiniObject*) {
              std::unique_ptr<BufferOwner> owner(static_cast<BufferOwner*>(data));
              ++(*owner)->released;
            },
            new BufferOwner(buffers));
        return GST_PAD_PROBE_OK;
      },
      new BufferOwner(buffers), +[](gpointer data) { delete static_cast<BufferOwner*>(data); });
  gst_object_unref(pad);

  SampleSpec spec;
  spec.kind = SampleMediaKind::Encoded;
  spec.media_type = media;
  spec.caps_string = media;
  spec.caps_key = capkey_from_spec(spec);
  InputStreamOptions options;
  options.public_output_contract = false;
  options.appsink_max_buffers = 2;
  options.copy_output = false;
  options.timeout_ms = 5000;
  options.prefer_synchronous_teardown = true;
  InputStream stream =
      InputStream::create(pipeline, nullptr, sink, spec, InputOptions{}, options, {}, nullptr);
  require(gst_element_set_state(pipeline, GST_STATE_PAUSED) != GST_STATE_CHANGE_FAILURE,
          "encoded queue fixture did not pause");
  require(gst_element_get_state(pipeline, nullptr, nullptr, GST_SECOND) != GST_STATE_CHANGE_FAILURE,
          "encoded queue fixture failed to preroll");
  RunOptions run_options;
  run_options.preset = RunPreset::Realtime;
  run_options.queue_depth = 1;
  run_options.overflow_policy = policy;
  run_options.output_memory = OutputMemory::ZeroCopy;
  auto core = runtime::RunCore::start_single_pipeline(std::move(stream), run_options, options,
                                                      RunMode::Async);
  try {
    gst_element_set_state(pipeline, GST_STATE_PLAYING);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (core->stats().outputs_ready == 0 && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    require(core->stats().outputs_ready == 1 && core->stats().outputs_dropped == 0,
            "full internal encoded queue must wait without dropping");
    if (!stop_full) {
      for (guint8 index = 0; index < 12; ++index) {
        Sample output;
        PullError error;
        require(core->pull(2000, output, &error) == PullStatus::Ok,
                "internal encoded queue lost a sample");
        require(output.tensors.size() == 1 && output.tensors.front().semantic.encoded.has_value(),
                "fixture did not produce encoded samples");
        const auto bytes = output.tensors.front().copy_payload_bytes();
        require(bytes.size() == 8 && bytes.front() == index && output.pts_ns == index * GST_MSECOND,
                "internal encoded queue changed payload or ordering");
      }
      require(core->stats().outputs_dropped == 0, "internal encoded queue dropped samples");
      Sample output;
      PullError error;
      require(core->pull(2000, output, &error) == PullStatus::Closed,
              "internal encoded queue did not drain to EOS");
    }
    const auto start = std::chrono::steady_clock::now();
    core->stop();
    require(std::chrono::steady_clock::now() - start < std::chrono::seconds(2),
            "full internal encoded queue prevented shutdown");
    core->close();
    core.reset();
    require(buffers->released == buffers->produced, "encoded buffers retained after shutdown");
  } catch (...) {
    if (core) {
      core->close();
    }
    throw;
  }
}

} // namespace

RUN_TEST("unit_internal_encoded_output_queue_test", ([] {
           gst_init_once();
           setenv("SIMA_PIPELINE_OUTPUT_DROP_ON_ZERO_COPY", "1", 1);
           for (const char* media : {"video/x-h264", "video/x-h265", "image/jpeg"}) {
             check_queue(media, OverflowPolicy::KeepLatest, false);
             check_queue(media, OverflowPolicy::Block, false);
             check_queue(media, OverflowPolicy::KeepLatest, true);
           }
         }));
