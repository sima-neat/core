#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "gst/GstInit.h"
#include "nodes/common/Output.h"
#include "nodes/io/CameraInput.h"
#include "nodes/io/RTSPInput.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/internal/InputStreamUtil.h"
#include "pipeline/internal/SampleUtil.h"
#include "pipeline/internal/TensorUtil.h"
#include "dmabuf_test_utils.h"
#include "pipeline/runtime/EdgeRouter.h"
#include "pipeline/runtime/ExecutionGraphRuntime.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <thread>
#include <vector>

#include <gst/gst.h>

namespace {

using NodeList = std::vector<std::shared_ptr<simaai::neat::Node>>;

simaai::neat::runtime::RuntimeSinkQueueMsg sink_message(std::int64_t frame_id,
                                                        std::size_t edge_index) {
  simaai::neat::runtime::RuntimeSinkQueueMsg message;
  message.sample.frame_id = frame_id;
  message.edge_index = edge_index;
  return message;
}

simaai::neat::SampleSpec raw_rgb_spec(int width, int height, int fps) {
  simaai::neat::SampleSpec spec;
  spec.kind = simaai::neat::SampleMediaKind::RawVideo;
  spec.media_type = "video/x-raw";
  spec.format = "RGB";
  spec.width = width;
  spec.height = height;
  spec.depth = 3;
  spec.fps_n = fps;
  spec.fps_d = 1;
  spec.required_bytes_actual = static_cast<std::size_t>(width * height * 3);
  spec.caps_string = simaai::neat::caps_string_from_spec(spec);
  spec.caps_key = simaai::neat::capkey_from_spec(spec);
  return spec;
}

struct AppSinkProperties {
  int max_buffers = -1;
  bool drop = false;
  bool sync = false;
};

AppSinkProperties configured_appsink_properties(const simaai::neat::InputStreamOptions& options) {
  GstElement* sink = gst_element_factory_make("appsink", nullptr);
  require(sink != nullptr, "appsink element is unavailable");

  simaai::neat::session_build_configure_appsink_for_input_stream(sink, options);

  guint max_buffers = 0;
  gboolean drop = FALSE;
  gboolean sync = FALSE;
  g_object_get(G_OBJECT(sink), "max-buffers", &max_buffers, "drop", &drop, "sync", &sync, nullptr);
  gst_object_unref(sink);
  return AppSinkProperties{static_cast<int>(max_buffers), drop != FALSE, sync != FALSE};
}

template <class Predicate> bool wait_until(Predicate&& predicate, int timeout_ms) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return predicate();
}

void require_appsink_properties(const simaai::neat::InputStreamOptions& options,
                                int expected_max_buffers, bool expected_drop, bool expected_sync,
                                const char* context) {
  const AppSinkProperties actual = configured_appsink_properties(options);
  require(actual.max_buffers == expected_max_buffers,
          std::string(context) + ": max-buffers changed");
  require(actual.drop == expected_drop, std::string(context) + ": drop changed");
  require(actual.sync == expected_sync, std::string(context) + ": sync changed");
}

// Replace only the fixture producer's output allocation. No payload is read:
// this proves Core allocation identity/ownership, not decoding or DMA import.
struct DmaOutputFixture {
  std::mutex mutex;
  std::unordered_map<std::uint64_t, sima_test::DmaBufSpan> allocations;
  std::string error;
};

GstPadProbeReturn publish_fixture_dmabuf(GstPad*, GstPadProbeInfo* info, gpointer user_data) {
  auto& fixture = *static_cast<DmaOutputFixture*>(user_data);
  GstBuffer* original = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!original) {
    return GST_PAD_PROBE_OK;
  }
  GstBuffer* buffer = nullptr;
  try {
    buffer = sima_test::make_bookkeeping_dmabuf(gst_buffer_get_size(original));
    gst_buffer_copy_into(
        buffer, original,
        static_cast<GstBufferCopyFlags>(GST_BUFFER_COPY_TIMESTAMPS | GST_BUFFER_COPY_FLAGS), 0, -1);
    {
      std::lock_guard<std::mutex> lock(fixture.mutex);
      fixture.allocations.emplace(GST_BUFFER_PTS(buffer), sima_test::dmabuf_span(buffer));
    }
    GST_PAD_PROBE_INFO_DATA(info) = buffer;
    gst_buffer_unref(original);
    return GST_PAD_PROBE_OK;
  } catch (const std::exception& error) {
    if (buffer) {
      gst_buffer_unref(buffer);
    }
    std::lock_guard<std::mutex> lock(fixture.mutex);
    fixture.error = error.what();
    return GST_PAD_PROBE_DROP;
  }
}

struct RemoveProbe {
  GstPad* pad = nullptr;
  gulong id = 0;
  ~RemoveProbe() {
    if (pad) {
      gst_pad_remove_probe(pad, id);
      gst_object_unref(pad);
    }
  }
};

} // namespace

RUN_TEST(
    "unit_public_output_appsink_contract_test", ([] {
      unsetenv("SIMA_RTSP_ALLOW_BACKPRESSURE");
      unsetenv("SIMA_OUTPUT_MEMORY_DEFAULT");
      setenv("SIMA_INPUTSTREAM_USE_APPSINK_CALLBACKS", "1", 1);
      setenv("SIMA_PIPELINE_OUTPUT_DROP_ON_ZERO_COPY", "1", 1);
      simaai::neat::gst_init_once();

      // Fused ingress stores RTSP producers in branch lists.  Their generic
      // anti-backpressure policy must not replace an explicit EveryFrame
      // contract on the shared public terminal.
      {
        simaai::neat::InputStreamOptions stream_options;
        stream_options.public_output_contract = true;
        stream_options.appsink_max_buffers = 2;
        stream_options.appsink_drop = true;
        stream_options.appsink_sync = true;
        const NodeList consumer = {
            simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(19))};
        const std::vector<NodeList> branches = {
            {simaai::neat::nodes::RTSPInput("rtsp://example.test/live")}};

        require(simaai::neat::graph_build_internal::apply_explicit_public_output_options(
                    stream_options, consumer),
                "fused public Output was not recognized");
        simaai::neat::session_build_maybe_enable_rtsp_appsink_drop(stream_options, consumer,
                                                                   branches);
        require_appsink_properties(stream_options, 19, false, false,
                                   "fused RTSP EveryFrame Output");
      }

      // Latest remains a one-sample dropping terminal even when RunOptions
      // selected a different queue policy.
      {
        simaai::neat::InputStreamOptions stream_options;
        stream_options.public_output_contract = true;
        stream_options.appsink_max_buffers = 31;
        stream_options.appsink_drop = false;
        stream_options.appsink_sync = true;
        const NodeList nodes = {simaai::neat::nodes::RTSPInput("rtsp://example.test/latest"),
                                simaai::neat::nodes::Output(simaai::neat::OutputOptions::Latest())};

        require(simaai::neat::graph_build_internal::apply_explicit_public_output_options(
                    stream_options, nodes),
                "Latest public Output was not recognized");
        simaai::neat::session_build_maybe_enable_rtsp_appsink_drop(stream_options, nodes);
        require_appsink_properties(stream_options, 1, true, false, "Latest Output");
      }

      // A framework-created Output at a graph-internal RTSP boundary is not a
      // public contract.  Keep the generic bounded/drop behavior there.
      {
        simaai::neat::InputStreamOptions stream_options;
        stream_options.public_output_contract = false;
        stream_options.appsink_max_buffers = 0;
        stream_options.appsink_drop = false;
        stream_options.appsink_sync = true;
        const NodeList consumer = {
            simaai::neat::nodes::Output(simaai::neat::OutputOptions::EveryFrame(19))};
        const std::vector<NodeList> branches = {
            {simaai::neat::nodes::RTSPInput("rtsp://example.test/internal")}};

        require(!simaai::neat::graph_build_internal::apply_explicit_public_output_options(
                    stream_options, consumer),
                "internal Output was mistaken for a public contract");
        simaai::neat::session_build_maybe_enable_rtsp_appsink_drop(stream_options, consumer,
                                                                   branches);
        require_appsink_properties(stream_options, 1, true, true, "internal RTSP boundary");
      }

      // The same ownership rule applies to ordinary non-RTSP pipelines; this
      // proves the fix is not special-cased to fused/live topology.
      {
        simaai::neat::OutputOptions output_options;
        output_options.max_buffers = 7;
        output_options.drop = false;
        output_options.sync = true;

        simaai::neat::InputStreamOptions stream_options;
        stream_options.public_output_contract = true;
        stream_options.appsink_max_buffers = 1;
        stream_options.appsink_drop = true;
        stream_options.appsink_sync = false;
        const NodeList nodes = {
            simaai::neat::nodes::CameraInput(simaai::neat::CameraInputOptions{}),
            simaai::neat::nodes::Output(output_options)};

        require(simaai::neat::graph_build_internal::apply_explicit_public_output_options(
                    stream_options, nodes),
                "ordinary public Output was not recognized");
        simaai::neat::session_build_maybe_enable_rtsp_appsink_drop(stream_options, nodes);
        require_appsink_properties(stream_options, 7, false, true, "ordinary non-RTSP Output");
      }

      // A named Output is drained from appsink into GraphSinkQueue.  That
      // second queue must enforce the same policy or it can accumulate
      // stale detections after the correctly-configured appsink.
      {
        using simaai::neat::runtime::FusedEncodedOutputEnqueueResult;
        using simaai::neat::runtime::GraphSinkQueue;
        using simaai::neat::runtime::RuntimeSinkQueueMsg;
        using simaai::neat::runtime::enqueue_graph_sink_output;

        const simaai::neat::OutputOptions every_frame = simaai::neat::OutputOptions::EveryFrame(2);
        GraphSinkQueue every_frame_queue(2);
        require(enqueue_graph_sink_output(every_frame_queue, every_frame, sink_message(1, 11), 0) ==
                    FusedEncodedOutputEnqueueResult::Enqueued,
                "EveryFrame queue rejected its first sample");
        require(enqueue_graph_sink_output(every_frame_queue, every_frame, sink_message(2, 12), 0) ==
                    FusedEncodedOutputEnqueueResult::Enqueued,
                "EveryFrame queue rejected its second sample");
        require(enqueue_graph_sink_output(every_frame_queue, every_frame, sink_message(3, 13), 0) ==
                    FusedEncodedOutputEnqueueResult::Overflow,
                "EveryFrame queue did not backpressure at max_buffers");

        const simaai::neat::OutputOptions latest = simaai::neat::OutputOptions::Latest();
        GraphSinkQueue latest_queue(1);
        require(enqueue_graph_sink_output(latest_queue, latest, sink_message(20, 20), 0) ==
                    FusedEncodedOutputEnqueueResult::Enqueued,
                "Latest queue rejected its first sample");
        require(enqueue_graph_sink_output(latest_queue, latest, sink_message(21, 21), 0) ==
                    FusedEncodedOutputEnqueueResult::ReplacedOldest,
                "Latest queue did not replace its oldest sample");
        RuntimeSinkQueueMsg newest;
        require(latest_queue.pop(newest, 0), "Latest queue lost its replacement sample");
        require(newest.sample.frame_id == 21 && newest.edge_index == 21,
                "Latest queue returned stale data or lost edge identity");

        GraphSinkQueue close_queue(1);
        require(enqueue_graph_sink_output(close_queue, every_frame, sink_message(30, 30), -1) ==
                    FusedEncodedOutputEnqueueResult::Enqueued,
                "close-wakeup queue rejected its first sample");
        std::atomic<FusedEncodedOutputEnqueueResult> blocked_result{
            FusedEncodedOutputEnqueueResult::Overflow};
        std::atomic<bool> blocked_done{false};
        std::thread blocked_producer([&] {
          blocked_result.store(
              enqueue_graph_sink_output(close_queue, every_frame, sink_message(31, 31), -1),
              std::memory_order_release);
          blocked_done.store(true, std::memory_order_release);
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
        const bool producer_blocked = !blocked_done.load(std::memory_order_acquire);
        close_queue.close();
        blocked_producer.join();
        require(producer_blocked,
                "EveryFrame producer did not backpressure while the queue was full");
        require(blocked_result.load(std::memory_order_acquire) ==
                    FusedEncodedOutputEnqueueResult::Closed,
                "closing GraphSinkQueue did not wake its EveryFrame producer");

        // A direct public Input owns an in-flight push until EdgeRouter admits it. If the
        // EveryFrame sink is full, close_input() must cancel that wait before waiting for the
        // public producer count, while preserving the sample already in the queue.
        {
          using simaai::neat::graph::NodeId;
          using simaai::neat::graph::PortId;
          using simaai::neat::runtime::DownstreamTarget;
          using simaai::neat::runtime::EdgePlan;
          using simaai::neat::runtime::EdgeRouterOptions;
          using simaai::neat::runtime::Endpoint;
          using simaai::neat::runtime::ExecutionGraphRuntime;
          using simaai::neat::runtime::PipelineSegmentRuntime;
          using simaai::neat::runtime::RunCore;

          constexpr NodeId kInputNode = 0;
          constexpr NodeId kSinkNode = 1;
          constexpr PortId kPort = 0;
          constexpr std::size_t kEdgeIndex = 0;
          const auto adjacency_key = [](NodeId node, PortId port) {
            return (static_cast<std::uint64_t>(node) << 32U) | static_cast<std::uint64_t>(port);
          };

          RunCore core;
          core.graph_execution_ = std::make_unique<ExecutionGraphRuntime>();
          auto& execution = *core.graph_execution_;
          execution.plan.edges.push_back(
              EdgePlan{.from = kInputNode, .from_port = kPort, .to = kSinkNode, .to_port = kPort});
          execution.adjacency[adjacency_key(kInputNode, kPort)].push_back(
              DownstreamTarget{DownstreamTarget::Kind::GraphSink, kSinkNode, kPort, kEdgeIndex});

          auto input_pipeline = std::make_unique<PipelineSegmentRuntime>();
          input_pipeline->seg.node_ids = {kInputNode};
          input_pipeline->seg.output_edges = {kEdgeIndex};
          input_pipeline->seg.boundary.direct_graph_source = true;
          execution.node_to_pipeline.emplace(kInputNode, 0U);
          execution.pipelines.push_back(std::move(input_pipeline));
          execution.public_ingress_endpoints.push_back(
              Endpoint{Endpoint::Kind::PipelineInput, kInputNode, kPort, 0U});

          auto public_sink = std::make_shared<GraphSinkQueue>(1, &every_frame);
          public_sink->set_producer_count(1);
          execution.sinks.emplace(kSinkNode, public_sink);

          EdgeRouterOptions blocking_options;
          blocking_options.push_timeout_ms = 100;
          blocking_options.request_stop_on_backpressure = true;
          simaai::neat::Sample queued_sample;
          queued_sample.frame_id = 50;
          require(core.graph_begin_public_push(), "direct public input was already closed");
          require(core.graph_push(kInputNode, kPort, true, queued_sample, blocking_options),
                  "direct public input rejected the first sink sample");
          core.graph_end_public_push();

          std::atomic<bool> direct_push_started{false};
          std::atomic<bool> direct_push_done{false};
          std::atomic<bool> direct_push_result{true};
          std::thread direct_push([&] {
            const bool admitted = core.graph_begin_public_push();
            direct_push_started.store(true, std::memory_order_release);
            simaai::neat::Sample blocked_sample;
            blocked_sample.frame_id = 51;
            const bool pushed = admitted && core.graph_push(kInputNode, kPort, true, blocked_sample,
                                                            blocking_options);
            if (admitted) {
              core.graph_end_public_push();
            }
            direct_push_result.store(pushed, std::memory_order_release);
            direct_push_done.store(true, std::memory_order_release);
          });
          const bool direct_push_did_start =
              wait_until([&] { return direct_push_started.load(std::memory_order_acquire); }, 500);
          if (!direct_push_did_start) {
            public_sink->close();
            direct_push.join();
            require(false, "direct public push thread did not start");
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          const bool direct_push_blocked = !direct_push_done.load(std::memory_order_acquire);

          std::atomic<bool> close_done{false};
          std::thread close_input([&] {
            core.graph_close_public_input();
            close_done.store(true, std::memory_order_release);
          });
          const bool close_unblocked =
              wait_until([&] { return close_done.load(std::memory_order_acquire); }, 1000);
          if (!close_unblocked) {
            public_sink->close();
          }
          close_input.join();
          direct_push.join();

          require(direct_push_blocked,
                  "direct EveryFrame push did not block on the full public sink");
          require(close_unblocked, "close_input deadlocked behind a blocked direct public push");
          require(!direct_push_result.load(std::memory_order_acquire),
                  "close_input did not cancel the blocked direct public push");
          RuntimeSinkQueueMsg preserved;
          require(public_sink->pop(preserved, 0),
                  "close_input dropped the sample already queued in the public sink");
          require(preserved.sample.frame_id == 50,
                  "close_input replaced the sample already queued in the public sink");
          require(!public_sink->pop(preserved, 0) && public_sink->closed(),
                  "direct public sink did not close after its queued sample drained");
        }

        // A public push can be inside synchronous lazy-pipeline construction, where no queue can
        // wake it. close_input() bounds that wait and defers producer completion to the guard that
        // eventually leaves the build path.
        {
          simaai::neat::runtime::RunCore core;
          core.graph_execution_ = std::make_unique<simaai::neat::runtime::ExecutionGraphRuntime>();
          require(core.graph_begin_public_push(), "lazy-build fixture input was already closed");
          const auto close_started = std::chrono::steady_clock::now();
          core.graph_close_public_input();
          const auto close_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - close_started);
          require(close_elapsed < std::chrono::milliseconds(1000),
                  "close_input waited indefinitely for a public lazy build");
          require(!core.graph_execution_->public_ingress_completion_forwarded,
                  "close_input forwarded completion while a public build was still active");
          core.graph_end_public_push();
          require(core.graph_execution_->public_ingress_completion_forwarded,
                  "last public build did not forward deferred input completion");
        }

        GraphSinkQueue reserved_latest_queue(1);
        require(enqueue_graph_sink_output(reserved_latest_queue, latest, sink_message(40, 40), 0) ==
                    FusedEncodedOutputEnqueueResult::Enqueued,
                "reserved Latest queue rejected its first sample");
        RuntimeSinkQueueMsg reserved;
        require(reserved_latest_queue.pop_with_restore_reservation(reserved, 0),
                "Latest reservation fixture could not reserve its sample");
        require(enqueue_graph_sink_output(reserved_latest_queue, latest, sink_message(41, 41), 0) ==
                    FusedEncodedOutputEnqueueResult::DroppedIncoming,
                "Latest enqueue treated a public-pull reservation as fatal overflow");
        require(reserved_latest_queue.restore_reserved_front(std::move(reserved)),
                "Latest reservation fixture could not restore its sample");
      }

      // Saturate the real appsink -> InputStream -> RunCore path with standard
      // DMA memory. EveryFrame overrides conflicting Realtime options; Auto
      // preserves producer storage both with and without a CPU-copy default.
      struct SaturationCase {
        const char* name;
        bool explicit_output;
        simaai::neat::RunPreset preset;
        bool stop_while_full;
      };
      for (const auto test_case : {
               SaturationCase{"EveryFrame/Realtime", true, simaai::neat::RunPreset::Realtime,
                              false},
               SaturationCase{"Block/Balanced", false, simaai::neat::RunPreset::Balanced, false},
               SaturationCase{"Block/Reliable", false, simaai::neat::RunPreset::Reliable, false},
               SaturationCase{"Block/stop", false, simaai::neat::RunPreset::Balanced, true},
           }) {
        GError* error = nullptr;
        GstElement* pipeline =
            gst_parse_launch("videotestsrc num-buffers=8 pattern=ball ! "
                             "video/x-raw,format=RGB,width=16,height=16,framerate=100/1 ! "
                             "appsink name=mysink emit-signals=false sync=false max-buffers=2 "
                             "drop=false enable-last-sample=false",
                             &error);
        if (error) {
          const std::string message = error->message ? error->message : "gst_parse_launch";
          g_error_free(error);
          throw std::runtime_error(message);
        }
        require(pipeline != nullptr, "DMA saturation pipeline did not parse");
        GstElement* sink = gst_bin_get_by_name(GST_BIN(pipeline), "mysink");
        require(sink != nullptr, "DMA saturation appsink was not found");

        DmaOutputFixture fixture;
        GstPad* sink_pad = gst_element_get_static_pad(sink, "sink");
        require(sink_pad != nullptr, "DMA saturation appsink has no sink pad");
        const auto probe = gst_pad_add_probe(sink_pad, GST_PAD_PROBE_TYPE_BUFFER,
                                             publish_fixture_dmabuf, &fixture, nullptr);
        RemoveProbe remove_probe{sink_pad, probe};

        simaai::neat::RunOptions run_options;
        run_options.preset = test_case.preset;
        run_options.queue_depth = 1;
        run_options.overflow_policy = test_case.explicit_output
                                          ? simaai::neat::OverflowPolicy::KeepLatest
                                          : simaai::neat::OverflowPolicy::Block;
        run_options.output_memory = test_case.explicit_output ? simaai::neat::OutputMemory::ZeroCopy
                                                              : simaai::neat::OutputMemory::Auto;
        auto stream_options = simaai::neat::session_build_make_stream_options(
            run_options, simaai::neat::RunMode::Async);
        stream_options.explicit_public_output_options = test_case.explicit_output;
        stream_options.appsink_max_buffers = test_case.explicit_output ? 2 : 1;
        stream_options.appsink_drop = false;
        stream_options.appsink_sync = false;
        stream_options.timeout_ms = 5000;
        stream_options.worker_poll_ms = 1;

        simaai::neat::InputStream stream = simaai::neat::InputStream::create(
            pipeline, nullptr, sink, raw_rgb_spec(16, 16, 100), simaai::neat::InputOptions{},
            stream_options, {}, nullptr);

        // PAUSED prevents a finite source outrunning callback installation.
        require(gst_element_set_state(pipeline, GST_STATE_PAUSED) != GST_STATE_CHANGE_FAILURE,
                "DMA saturation pipeline did not pause");
        GstState current_state = GST_STATE_NULL;
        GstState pending_state = GST_STATE_VOID_PENDING;
        require(gst_element_get_state(pipeline, &current_state, &pending_state, 5 * GST_SECOND) !=
                        GST_STATE_CHANGE_FAILURE &&
                    current_state == GST_STATE_PAUSED,
                "DMA saturation pipeline did not reach PAUSED");

        auto core = simaai::neat::runtime::RunCore::start_single_pipeline(
            std::move(stream), run_options, stream_options, simaai::neat::RunMode::Async);
        struct StopCore {
          std::shared_ptr<simaai::neat::runtime::RunCore> core;
          ~StopCore() {
            if (core) {
              try {
                core->stop();
              } catch (...) {
              }
            }
          }
        } stop{core};

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto before = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
        require(gst_element_set_state(pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
                "DMA saturation pipeline did not start");
        const auto output_capacity = static_cast<std::uint64_t>(stream_options.appsink_max_buffers);
        require(wait_until([&] { return core->stats().outputs_ready >= output_capacity; }, 2000),
                "DMA saturation run did not fill its output queue");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const auto saturated = core->stats();
        require(saturated.outputs_ready == output_capacity && saturated.outputs_dropped == 0,
                "DMA Block policy was silently replaced by dropping under queue pressure");
        require(core->pipeline.copy_output_latched.load() == stream_options.copy_output,
                "DMA queue pressure activated the Balanced copy latch");
        {
          std::lock_guard<std::mutex> lock(fixture.mutex);
          require(fixture.error.empty(), "DMA producer fixture failed: " + fixture.error);
          require(fixture.allocations.size() < 8U,
                  std::string(test_case.name) +
                      ": the finite producer ran through a saturated no-drop output");
        }

        if (test_case.stop_while_full) {
          const auto stop_started = std::chrono::steady_clock::now();
          core->stop();
          require(std::chrono::steady_clock::now() - stop_started < std::chrono::seconds(2),
                  "stop failed to wake the blocked DMA output worker");
          continue;
        }

        std::int64_t previous_pts = -1;
        for (int index = 0; index < 8; ++index) {
          simaai::neat::Sample output;
          simaai::neat::PullError pull_error;
          const auto status = core->pull(2000, output, &pull_error);
          if (status != simaai::neat::PullStatus::Ok) {
            const auto stats = core->stats();
            std::lock_guard<std::mutex> lock(fixture.mutex);
            throw std::runtime_error(std::string("DMA saturation ") + test_case.name +
                                     " received=" + std::to_string(index) +
                                     "/8 produced=" + std::to_string(fixture.allocations.size()) +
                                     " ready=" + std::to_string(stats.outputs_ready) +
                                     " dropped=" + std::to_string(stats.outputs_dropped) + ": " +
                                     pull_error.message +
                                     (fixture.error.empty() ? "" : "; producer: " + fixture.error));
          }
          require(output.pts_ns > previous_pts, "DMA saturation outputs were not ordered");
          previous_pts = output.pts_ns;
          require(!output.tensors.empty() &&
                      simaai::neat::pipeline_internal::sample_has_dmabuf_memory(output),
                  "DMA output was materialized instead of retaining producer memory");
          GstBuffer* retained = simaai::neat::pipeline_internal::buffer_from_tensor_holder(
              output.tensors.front().storage->holder);
          require(retained != nullptr, "DMA output lost its retained GstSample");
          const auto actual = sima_test::dmabuf_span(retained);
          gst_buffer_unref(retained);
          std::lock_guard<std::mutex> lock(fixture.mutex);
          require(fixture.error.empty(), "DMA producer fixture failed: " + fixture.error);
          const auto expected = fixture.allocations.find(static_cast<std::uint64_t>(output.pts_ns));
          require(expected != fixture.allocations.end() && actual == expected->second,
                  "DMA output changed backing allocation, offset or span");
        }
        const auto after = simaai::neat::pipeline_internal::snapshot_tensor_io_stats();
        require(after.tensor_copy_count == before.tensor_copy_count &&
                    after.gst_memory_map_calls == before.gst_memory_map_calls,
                "DMA output policy introduced a hidden copy or generic CPU map");
        const simaai::neat::RunStats stats = core->stats();
        require(stats.outputs_ready == 8 && stats.outputs_pulled == 8 && stats.outputs_dropped == 0,
                "DMA saturation run dropped a terminal result");
        simaai::neat::Sample output;
        simaai::neat::PullError pull_error;
        const auto status = core->pull(2000, output, &pull_error);
        require(status == simaai::neat::PullStatus::Closed &&
                    pull_error.code == simaai::neat::error_codes::kSourceEnded,
                std::string(test_case.name) + ": expected EOS after all eight outputs");
      }
    }));
