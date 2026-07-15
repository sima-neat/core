#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "builder/Node.h"
#include "gst/GstInit.h"
#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/io/CameraInput.h"
#include "nodes/io/Input.h"
#include "nodes/io/RTSPInput.h"
#include "nodes/io/UdpOutput.h"
#include "nodes/groups/RtspEncodedInput.h"
#include "nodes/groups/VideoSender.h"
#include "nodes/rtp/H264Depacketize.h"
#include "nodes/sima/H264Packetize.h"
#include "nodes/sima/H264Parse.h"
#include "nodes/sima/SimaDecode.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "test_main.h"
#include "test_utils.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

namespace {

template <typename GraphType>
concept AcceptsReleasedBraceConnect = requires(
    GraphType& graph, const GraphType& from, const GraphType& to) { graph.connect(from, to, {}); };

static_assert(std::is_standard_layout_v<simaai::neat::GraphLinkOptions>);
static_assert(std::is_aggregate_v<simaai::neat::GraphLinkOptions>);
static_assert(
    std::is_same_v<decltype(simaai::neat::GraphLinkOptions::max_inflight_per_stream), int>);
static_assert(std::is_same_v<decltype(simaai::neat::GraphLinkOptions::max_inflight_total), int>);
static_assert(AcceptsReleasedBraceConnect<simaai::neat::Graph>);

// RunAdvancedOptions is passed across the public shared-library boundary. Keep
// its released layout pinned: inserting a new bool into the padding after
// copy_input would preserve offsets while making a new library interpret
// uninitialized padding from an old caller as a feature flag.
struct ReleasedRunAdvancedOptionsLayout {
  bool copy_input = false;
  std::size_t max_input_bytes = 0;
  int sync_num_buffers_override = -1;
  bool prepare_output_cpu_visible = false;
};

template <typename T>
concept AcceptsPaddingBoolAsAggregateMember = requires {
  // This is the precise five-member shape that caused the ABI hazard: a new
  // bool inserted between copy_input and max_input_bytes.
  T{false, false, std::size_t{0}, -1, false};
};

static_assert(std::is_standard_layout_v<simaai::neat::RunAdvancedOptions>);
static_assert(std::is_aggregate_v<simaai::neat::RunAdvancedOptions>);
static_assert(!AcceptsPaddingBoolAsAggregateMember<simaai::neat::RunAdvancedOptions>);
static_assert(sizeof(simaai::neat::RunAdvancedOptions) == sizeof(ReleasedRunAdvancedOptionsLayout));
static_assert(alignof(simaai::neat::RunAdvancedOptions) ==
              alignof(ReleasedRunAdvancedOptionsLayout));
static_assert(offsetof(simaai::neat::RunAdvancedOptions, copy_input) ==
              offsetof(ReleasedRunAdvancedOptionsLayout, copy_input));
static_assert(offsetof(simaai::neat::RunAdvancedOptions, max_input_bytes) ==
              offsetof(ReleasedRunAdvancedOptionsLayout, max_input_bytes));
static_assert(offsetof(simaai::neat::RunAdvancedOptions, sync_num_buffers_override) ==
              offsetof(ReleasedRunAdvancedOptionsLayout, sync_num_buffers_override));
static_assert(offsetof(simaai::neat::RunAdvancedOptions, prepare_output_cpu_visible) ==
              offsetof(ReleasedRunAdvancedOptionsLayout, prepare_output_cpu_visible));

class ScopedUnsetEnv {
public:
  explicit ScopedUnsetEnv(const char* key) : key_(key ? key : "") {
    if (const char* value = std::getenv(key_.c_str())) {
      previous_ = value;
    }
    unsetenv(key_.c_str());
  }

  ~ScopedUnsetEnv() {
    if (previous_.has_value()) {
      setenv(key_.c_str(), previous_->c_str(), 1);
    } else {
      unsetenv(key_.c_str());
    }
  }

  ScopedUnsetEnv(const ScopedUnsetEnv&) = delete;
  ScopedUnsetEnv& operator=(const ScopedUnsetEnv&) = delete;

private:
  std::string key_;
  std::optional<std::string> previous_;
};

class FragmentNode final : public simaai::neat::Node {
public:
  FragmentNode(std::string kind, std::string factory, std::string role, std::string properties = {})
      : kind_(std::move(kind)), factory_(std::move(factory)), role_(std::move(role)),
        properties_(std::move(properties)) {}

  std::string kind() const override {
    return kind_;
  }

  std::string backend_fragment(int node_index) const override {
    return factory_ + " name=" + element_name(node_index) + properties_;
  }

  std::vector<std::string> element_names(int node_index) const override {
    return {element_name(node_index)};
  }

  simaai::neat::NodeCapsBehavior caps_behavior() const override {
    return simaai::neat::NodeCapsBehavior::Dynamic;
  }

private:
  std::string element_name(int node_index) const {
    return "n" + std::to_string(node_index) + "_" + role_;
  }

  std::string kind_;
  std::string factory_;
  std::string role_;
  std::string properties_;
};

std::vector<std::shared_ptr<simaai::neat::Node>> make_consumer_nodes() {
  std::vector<std::shared_ptr<simaai::neat::Node>> nodes;
  // Model-managed routes render multiple hardware stages inside one Node
  // fragment. Keep explicit false/default properties here to prove that the
  // fused renderer rewrites the matching stage segment rather than mistaking
  // ProcessCVU's `async` or `num-buffers` token for ProcessMLA's token.
  nodes.push_back(std::make_shared<FragmentNode>(
      "ModelRoute", "neatprocesscvu", "preproc",
      " async=false num-buffers=4 ! neatprocessmla name=n0_mla async=false num-buffers=4 "
      "defer-output-invalidate=false ! neatboxdecode name=n0_boxdecode ! appsink "
      "name=n0_output"));
  return nodes;
}

simaai::neat::Graph make_composed_consumer_graph() {
  simaai::neat::Graph graph("detector");
  graph.add(simaai::neat::nodes::Input("detector_frame"));
  graph.add(std::make_shared<FragmentNode>(
      "ModelRoute", "neatprocesscvu", "preproc",
      " async=false num-buffers=4 ! neatprocessmla name=n1_mla async=false num-buffers=4 "
      "defer-output-invalidate=false ! neatboxdecode name=n1_boxdecode"));
  graph.add(simaai::neat::nodes::Output("detections"));
  return graph;
}

simaai::neat::Graph make_live_source_graph(int index, std::uint32_t width = 1920,
                                           std::uint32_t height = 1080, std::uint32_t fps = 30) {
  simaai::neat::CameraInputOptions options;
  options.buffer_name = "fused_queue_cam" + std::to_string(index);
  options.width = width;
  options.height = height;
  options.framerate_num = fps;
  simaai::neat::Graph graph("camera" + std::to_string(index));
  graph.add(simaai::neat::nodes::CameraInput(options));
  return graph;
}

std::size_t count_occurrences(const std::string& value, const std::string& token) {
  std::size_t count = 0;
  std::size_t offset = 0;
  while ((offset = value.find(token, offset)) != std::string::npos) {
    ++count;
    offset += token.size();
  }
  return count;
}

void mark_mini_object_finalized(gpointer data, GstMiniObject*) {
  *static_cast<bool*>(data) = true;
}

} // namespace

RUN_TEST(
    "unit_fused_realtime_fast_path_options_test", ([] {
      ScopedUnsetEnv cvu_kill_switch("SIMA_PROCESSCVU_ASYNC");
      ScopedUnsetEnv mla_kill_switch("SIMA_PROCESSMLA_SAFE_ASYNC");
      ScopedUnsetEnv legacy_queue_enable("SIMA_ENABLE_ASYNC_QUEUE2");
      ScopedUnsetEnv legacy_queue_depth("SIMA_ASYNC_QUEUE2_DEPTH");
      ScopedUnsetEnv global_inflight_override("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_GLOBAL");
      ScopedUnsetEnv rtsp_backpressure_override("SIMA_RTSP_ALLOW_BACKPRESSURE");

      simaai::neat::gst_init_once();

      // Fused sources keep producers in per-stream branch node lists. Ensure
      // an RTSP producer there still applies the live-source appsink policy to
      // the shared terminal consumer.
      simaai::neat::InputStreamOptions fused_rtsp_stream_options;
      std::vector<std::vector<std::shared_ptr<simaai::neat::Node>>> fused_rtsp_branches = {
          {simaai::neat::nodes::RTSPInput("rtsp://example.test/live")}};
      simaai::neat::session_build_maybe_enable_rtsp_appsink_drop(
          fused_rtsp_stream_options, make_consumer_nodes(), fused_rtsp_branches);
      require(fused_rtsp_stream_options.appsink_drop,
              "an RTSP node in a fused source branch must make the terminal appsink drop");
      require(fused_rtsp_stream_options.appsink_max_buffers == 4,
              "fused RTSP appsink dropping must preserve a configured queue depth");

      simaai::neat::InputStreamOptions non_rtsp_stream_options;
      std::vector<std::vector<std::shared_ptr<simaai::neat::Node>>> non_rtsp_branches = {
          {simaai::neat::nodes::CameraInput(simaai::neat::CameraInputOptions{})}};
      simaai::neat::session_build_maybe_enable_rtsp_appsink_drop(
          non_rtsp_stream_options, make_consumer_nodes(), non_rtsp_branches);
      require(!non_rtsp_stream_options.appsink_drop,
              "fused non-RTSP branches must preserve the configured appsink policy");

      // Model a shared probe buffer with one external observer reference and
      // one streaming reference in the probe data slot. COW replacement must
      // consume only the slot reference: the observer keeps the original alive
      // and no displaced streaming reference leaks.
      GstBuffer* original_probe_buffer = gst_buffer_new_allocate(nullptr, 16, nullptr);
      require(original_probe_buffer != nullptr, "failed to allocate terminal probe buffer");
      bool original_finalized = false;
      gst_mini_object_weak_ref(GST_MINI_OBJECT_CAST(original_probe_buffer),
                               mark_mini_object_finalized, &original_finalized);
      GstPadProbeInfo terminal_probe_info{};
      terminal_probe_info.type = GST_PAD_PROBE_TYPE_BUFFER;
      terminal_probe_info.data = gst_buffer_ref(original_probe_buffer);
      require(!gst_buffer_is_writable(original_probe_buffer),
              "external and probe-slot references must force copy-on-write");

      GstBuffer* writable_probe_buffer =
          simaai::neat::session_test::make_fused_terminal_probe_buffer_writable_for_test(
              &terminal_probe_info);
      require(writable_probe_buffer != nullptr &&
                  writable_probe_buffer == GST_PAD_PROBE_INFO_BUFFER(&terminal_probe_info),
              "terminal probe helper must install its writable result in the probe data slot");
      require(writable_probe_buffer != original_probe_buffer &&
                  gst_buffer_is_writable(writable_probe_buffer),
              "a shared terminal buffer must be replaced by a distinct writable buffer");
      require(!original_finalized && GST_MINI_OBJECT_REFCOUNT_VALUE(original_probe_buffer) == 1,
              "terminal COW must retain the external original without leaking the displaced "
              "probe-slot reference");
      gst_buffer_unref(original_probe_buffer);
      require(original_finalized,
              "the original terminal buffer must finalize after its observer releases it");
      gst_buffer_unref(writable_probe_buffer);

      simaai::neat::GraphOptions options;
      options.processcvu.async = true;
      options.processmla.async = true;
      options.processmla.output_pool_buffers = 7;
      options.processmla.defer_output_invalidate = true;
      options.async_queue_depth = 4;

      const std::string pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              make_consumer_nodes(), options);
      require_contains(pipeline, "neatlatestbystreammux",
                       "test must exercise the fused realtime renderer");
      require_contains(pipeline, "stream-inflight-limits=\"4\"",
                       "unset per-stream admission must resolve to the public default");
      require_contains(pipeline, "max-inflight-total=4",
                       "single-stream fused admission must preserve the default total guard");
      require_contains(pipeline, "neatprocesscvu name=n0_preproc async=true num-buffers=4",
                       "fused ProcessCVU must receive the public async option");
      require_contains(pipeline, "neatprocessmla name=n0_mla async=true num-buffers=7",
                       "fused pipeline must retain ProcessMLA");
      require(count_occurrences(pipeline, "async=true") == 2U,
              "fused ProcessCVU and ProcessMLA must both receive the public async option");
      require_contains(pipeline, "num-buffers=7",
                       "fused ProcessMLA must receive the public output-pool option");
      require_contains(pipeline, "defer-output-invalidate=true",
                       "fused ProcessMLA must receive the public deferred-cache-sync option");

      const std::string queue = "queue max-size-buffers=4 max-size-bytes=0 max-size-time=0";
      require(count_occurrences(pipeline, queue) == 3U,
              "fused async depth must insert exactly mux-to-CVU, CVU-to-MLA, and "
              "MLA-to-decode queues");
      require_contains(pipeline,
                       "stream-inflight-limits=\"4\" max-inflight-total=4 ! " + queue +
                           " ! neatprocesscvu",
                       "first fused queue must decouple the mux from ProcessCVU");
      require_contains(pipeline, "num-buffers=4 ! " + queue + " ! neatprocessmla",
                       "second fused queue must decouple ProcessCVU from ProcessMLA");
      require_contains(pipeline, "defer-output-invalidate=true ! " + queue + " ! neatboxdecode",
                       "third fused queue must decouple ProcessMLA from decode");
      require_contains(pipeline, "neatboxdecode name=n0_boxdecode ! appsink name=n0_output",
                       "terminal Output must stay directly connected to decode");
      require(pipeline.find("leaky=") == std::string::npos,
              "fused consumer-stage queues must be non-leaky so terminal loan release "
              "cannot be skipped");
      setenv("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_GLOBAL", "3", 1);
      const std::string env_limited_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              make_consumer_nodes(), options);
      require_contains(env_limited_pipeline, "max-inflight-total=3",
                       "fused default admission must honor the realtime global env override");

      simaai::neat::GraphLinkOptions explicit_total_link;
      explicit_total_link.max_inflight_total = 2;
      const std::string explicit_total_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              make_consumer_nodes(), options, {explicit_total_link});
      require_contains(explicit_total_pipeline, "max-inflight-total=2",
                       "an explicit fused total cap must take precedence over the env override");
      unsetenv("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_GLOBAL");

      const auto require_replacing_drop_rejected = [&](const std::string& fragment,
                                                       const char* reason) {
        std::vector<std::shared_ptr<simaai::neat::Node>> dropping_nodes;
        dropping_nodes.push_back(std::make_shared<FragmentNode>(
            "DroppingModelRoute", "neatprocesscvu", "drop_preproc", fragment));
        bool rejected = false;
        try {
          (void)simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              dropping_nodes, options);
        } catch (const std::invalid_argument&) {
          rejected = true;
        }
        require(rejected, reason);
      };
      require_replacing_drop_rejected(
          " ! queue name=explicit_leaky max-size-buffers=1 leaky=2 ! appsink "
          "name=drop_output",
          "buffer-replacing fused consumers must reject numeric leaky queues before terminal");
      require_replacing_drop_rejected(
          " ! queue name=spaced_leaky max-size-buffers=1 leaky = downstream ! appsink "
          "name=spaced_drop_output",
          "buffer-replacing fused consumers must reject spaced enum leaky assignments");
      require_replacing_drop_rejected(
          " ! queue name=reassigned_leaky leaky=0 leaky=2 ! appsink "
          "name=reassigned_drop_output",
          "buffer-replacing fused consumers must honor the last repeated property assignment");
      require_replacing_drop_rejected(
          " ! valve name=spaced_valve drop = true ! appsink name=valve_output",
          "buffer-replacing fused consumers must reject spaced valve-drop assignments");
      require_replacing_drop_rejected(
          " ! videorate name=explicit_rate ! appsink name=rate_output",
          "buffer-replacing fused consumers must reject pre-terminal videorate drops");
      std::vector<std::shared_ptr<simaai::neat::Node>> zero_drop_nodes;
      zero_drop_nodes.push_back(std::make_shared<FragmentNode>(
          "NonDroppingModelRoute", "neatprocesscvu", "zero_drop_preproc",
          " ! identity drop-probability = 0.0 ! queue leaky = 0 ! appsink "
          "name=zero_drop_output"));
      const std::string zero_drop_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              zero_drop_nodes, options);
      require_contains(zero_drop_pipeline, "drop-probability = 0.0",
                       "explicit non-dropping values must remain valid in replacing chains");
      std::vector<std::shared_ptr<simaai::neat::Node>> identity_drop_nodes;
      identity_drop_nodes.push_back(std::make_shared<FragmentNode>(
          "IdentityDropRoute", "identity", "identity_drop",
          " drop-probability=1.0 ! appsink name=identity_drop_output"));
      const std::string identity_drop_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              identity_drop_nodes, options);
      require_contains(identity_drop_pipeline, "drop-probability=1.0",
                       "identity-preserving fused chains may retain guard-backed drop handling");

      simaai::neat::GraphOptions no_consumer_queues = options;
      no_consumer_queues.async_queue_depth = 0;
      const std::string no_queue_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              make_consumer_nodes(), no_consumer_queues);
      require(no_queue_pipeline.find("queue max-size-buffers=") == std::string::npos,
              "async_queue_depth=0 must preserve the fused single-chain behavior");

      const auto reordered_timing =
          simaai::neat::session_test::find_fused_decoder_timing_match_for_test({100U, 200U, 300U},
                                                                               300U);
      require(reordered_timing.has_value() && *reordered_timing == 2U,
              "decoder timing restoration must match a reordered output by PTS, not FIFO age");
      const auto dropped_au_timing =
          simaai::neat::session_test::find_fused_decoder_timing_match_for_test({100U, 200U, 300U},
                                                                               200U);
      require(dropped_au_timing.has_value() && *dropped_au_timing == 1U,
              "a dropped encoded AU must not shift timing onto the next decoder output");
      require(!simaai::neat::session_test::find_fused_decoder_timing_match_for_test(
                   {100U, 200U, 300U}, 400U)
                   .has_value(),
              "an uncorrelated decoder output must retain its own timestamps instead of guessing");

      using simaai::neat::graph::NodeId;
      using simaai::neat::graph::PortId;
      require(simaai::neat::runtime::session_test::fused_realtime_source_segment_eligible_for_test(
                  false),
              "an unfused private live source must remain eligible for fused lowering");
      require(!simaai::neat::runtime::session_test::fused_realtime_source_segment_eligible_for_test(
                  true),
              "recursive fused fan-in must stay on the explicit transport path until nested "
              "source branches can be preserved");
      require(simaai::neat::runtime::session_test::fused_realtime_destinations_share_port_for_test(
                  {{NodeId{10U}, PortId{20U}}, {NodeId{10U}, PortId{20U}}}),
              "same-port realtime fan-in must remain eligible for fused lowering");
      require(!simaai::neat::runtime::session_test::fused_realtime_destinations_share_port_for_test(
                  {{NodeId{10U}, PortId{20U}}, {NodeId{10U}, PortId{21U}}}),
              "distinct consumer ports must remain on the non-fused runtime path");
      require(!simaai::neat::runtime::session_test::fused_realtime_destinations_share_port_for_test(
                  {{NodeId{10U}, PortId{20U}}, {NodeId{11U}, PortId{20U}}}),
              "distinct consumer nodes must not be collapsed merely because port ids match");
      const auto distinct_fallback_ids =
          simaai::neat::runtime::session_test::resolve_unique_fused_stream_ids_for_test({"", ""});
      require(distinct_fallback_ids.has_value() &&
                  *distinct_fallback_ids == std::vector<std::string>{"stream0", "stream1"},
              "empty fused stream ids must resolve to stable ordinal fallbacks");
      require(!simaai::neat::runtime::session_test::resolve_unique_fused_stream_ids_for_test(
                   {"stream1", ""})
                   .has_value(),
              "an explicit stream id must not collide with another branch's ordinal fallback");
      require(!simaai::neat::runtime::session_test::resolve_unique_fused_stream_ids_for_test(
                   {"same", "same"})
                   .has_value(),
              "duplicate explicit fused stream ids must be rejected before plan mutation");

      simaai::neat::GraphOptions outer_options;
      outer_options.async_queue_depth = 3;

      // Customers do not need a fusion build mode, a realtime-specific
      // connect method, or a feature switch for the common live fan-in case.
      // Two ordinary live-source connect() calls are promoted to the default
      // latest-by-stream policy and ordinary build() compilation fuses them.
      simaai::neat::Graph default_live_app("default_live_fan_in", outer_options);
      auto default_live_detector = make_composed_consumer_graph();
      default_live_app.connect(make_live_source_graph(400), default_live_detector);
      default_live_app.connect(make_live_source_graph(401), default_live_detector);
      const auto default_live_plan =
          simaai::neat::runtime::compile_public_graph(default_live_app, simaai::neat::RunOptions{});
      const auto default_live_fused = std::find_if(
          default_live_plan.pipeline_segments.begin(), default_live_plan.pipeline_segments.end(),
          [](const auto& segment) { return segment.fused_realtime_ingress.has_value(); });
      require(default_live_fused != default_live_plan.pipeline_segments.end() &&
                  default_live_fused->fused_realtime_ingress->branches.size() == 2U,
              "ordinary connect()/build() live fan-in must select fused lowering automatically");
      for (const auto& branch : default_live_fused->fused_realtime_ingress->branches) {
        require(branch.link_options.policy ==
                        simaai::neat::GraphLinkPolicy::RealtimeLatestByStream &&
                    branch.link_options.max_inflight_per_stream == -1 &&
                    branch.link_options.max_inflight_total == -1,
                "automatic live fan-in must use safe framework admission defaults without "
                "customer setup");
      }

      simaai::neat::Graph composed_app("composed_fused_queue_app", outer_options);
      auto composed_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        auto source = make_live_source_graph(stream);
        simaai::neat::GraphLinkOptions link;
        link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        link.queue_depth = 1;
        link.stream_id = "fused_queue_stream" + std::to_string(stream);
        link.max_inflight_per_stream = stream == 0 ? 1 : 8;
        link.max_inflight_total = stream == 0 ? 2 : 8;
        composed_app.connect(source, composed_detector, link);
      }
      simaai::neat::RunOptions composed_run_options;
      composed_run_options.queue_depth = 1;
      const auto composed_plan =
          simaai::neat::runtime::compile_public_graph(composed_app, composed_run_options);
      const simaai::neat::runtime::PipelineSegmentPlan* fused_segment = nullptr;
      for (const auto& segment : composed_plan.pipeline_segments) {
        if (segment.fused_realtime_ingress.has_value()) {
          fused_segment = &segment;
          break;
        }
      }
      require(fused_segment != nullptr,
              "actual composed live graph must lower to fused realtime ingress");
      require(fused_segment->route_options.async_queue_depth == 3,
              "fused segment must preserve outer GraphOptions::async_queue_depth");
      require(fused_segment->fused_realtime_ingress->branches.size() == 2U,
              "fused segment must preserve both source links");
      const auto& first_fused_edge = composed_plan.edges.at(
          fused_segment->fused_realtime_ingress->branches.front().edge_index);
      std::vector<simaai::neat::GraphLinkOptions> fused_link_options;
      std::string rendered_per_stream_limits;
      bool found_stream0 = false;
      bool found_stream1 = false;
      for (const auto& branch : fused_segment->fused_realtime_ingress->branches) {
        const auto& fused_edge = composed_plan.edges.at(branch.edge_index);
        require(fused_edge.to == first_fused_edge.to &&
                    fused_edge.to_port == first_fused_edge.to_port,
                "positive fused topology must be a true same-destination-port fan-in");
        if (branch.stream_id == "fused_queue_stream0") {
          require(branch.link_options.max_inflight_per_stream == 1 &&
                      branch.link_options.max_inflight_total == 2,
                  "stream 0 must retain its stricter per-stream and total admission caps");
          found_stream0 = true;
        } else if (branch.stream_id == "fused_queue_stream1") {
          require(branch.link_options.max_inflight_per_stream == 8 &&
                      branch.link_options.max_inflight_total == 8,
                  "stream 1 must retain its independent per-stream and total admission caps");
          found_stream1 = true;
        } else {
          require(false, "fused ingress must retain each public stream id");
        }
        require(branch.link_options.policy == simaai::neat::GraphLinkPolicy::RealtimeLatestByStream,
                "fused ingress must preserve the latest-by-stream policy");
        if (!rendered_per_stream_limits.empty()) {
          rendered_per_stream_limits += ',';
        }
        rendered_per_stream_limits += std::to_string(branch.link_options.max_inflight_per_stream);
        fused_link_options.push_back(branch.link_options);
      }
      require(found_stream0 && found_stream1,
              "fan-in normalization must preserve both heterogeneous link contracts");
      const std::string composed_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              fused_segment->nodes, fused_segment->route_options, fused_link_options);
      require_contains(composed_pipeline,
                       "stream-inflight-limits=\"" + rendered_per_stream_limits + "\"",
                       "fused mux must receive public per-link admission limits");
      require_contains(composed_pipeline, "max-inflight-total=2",
                       "fused mux must apply the strictest public mux-wide admission limit");
      const std::string composed_queue =
          "queue max-size-buffers=3 max-size-bytes=0 max-size-time=0";
      require(count_occurrences(composed_pipeline, composed_queue) == 3U,
              "actual composed fused graph must render the three selected stage queues");
      require(composed_pipeline.find(composed_queue + " ! appsink") == std::string::npos,
              "actual composed fused graph must not queue before its terminal Output");

      const std::string no_output_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              fused_segment->nodes, fused_segment->route_options, fused_link_options,
              /*enable_terminal_loans=*/false);
      require_contains(no_output_pipeline, "stream-inflight-limits=\"0,0\"",
                       "a fused graph without terminal Output must not take unreleasable "
                       "per-stream loans");
      require_contains(no_output_pipeline, "max-inflight-total=0",
                       "a fused graph without terminal Output must disable its total loan gate");

      // Partial aggregate initialization remains source-compatible and leaves admission caps at
      // their framework-default sentinels.
      simaai::neat::Graph default_options_app("default_graph_link_options", outer_options);
      auto default_options_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        simaai::neat::GraphLinkOptions default_link{
            simaai::neat::GraphLinkPolicy::RealtimeLatestByStream,
            1,
            "default_stream" + std::to_string(stream),
        };
        default_options_app.connect(make_live_source_graph(300 + stream), default_options_detector,
                                    default_link);
      }
      const auto default_options_plan =
          simaai::neat::runtime::compile_public_graph(default_options_app, composed_run_options);
      const simaai::neat::runtime::PipelineSegmentPlan* default_options_segment = nullptr;
      for (const auto& segment : default_options_plan.pipeline_segments) {
        if (segment.fused_realtime_ingress.has_value()) {
          default_options_segment = &segment;
          break;
        }
      }
      require(default_options_segment != nullptr,
              "default GraphLinkOptions path must remain eligible for automatic fusion");
      std::vector<simaai::neat::GraphLinkOptions> default_options_links;
      for (const auto& branch : default_options_segment->fused_realtime_ingress->branches) {
        require(branch.link_options.max_inflight_per_stream == -1 &&
                    branch.link_options.max_inflight_total == -1,
                "partial GraphLinkOptions initialization must retain default admission values");
        default_options_links.push_back(branch.link_options);
      }
      const std::string default_options_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              default_options_segment->nodes, default_options_segment->route_options,
              default_options_links);
      require_contains(default_options_pipeline, "stream-inflight-limits=\"4,4\"",
                       "GraphLinkOptions must resolve the framework per-stream default");

      // Public encoded fan-out stays expressed as ordinary graph topology. The
      // compiler must absorb source+decoder into each fused ingress branch while
      // preserving the encoded named Outputs as graph-scoped, ref-counted taps.
      simaai::neat::Graph encoded_fanout_app("encoded_output_fused_app", outer_options);
      auto encoded_fanout_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        simaai::neat::nodes::groups::RtspEncodedInputOptions source_options;
        source_options.url = "rtsp://example.test/stream" + std::to_string(stream);
        // Exercise the public defaults: queues remain enabled and H.264 caps
        // are repaired from stream metadata (with a complete fallback).
        source_options.fallback_h264_fps = 20;
        source_options.fallback_h264_width = 1280;
        source_options.fallback_h264_height = 720;
        auto source = simaai::neat::nodes::groups::RtspEncodedInput(source_options);

        simaai::neat::Graph decoder("decoder" + std::to_string(stream));
        simaai::neat::H264ParseOptions decoder_parser;
        decoder_parser.enforce_caps = true;
        decoder_parser.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
        decoder_parser.stream_format = simaai::neat::H264ParseOptions::StreamFormat::ByteStream;
        decoder.add(simaai::neat::nodes::H264Parse(decoder_parser));
        decoder.add(simaai::neat::nodes::SimaDecode());
        // Match the application topology exactly: the named raw boundary is
        // required for ordinary connect() inference, but automatic fusion
        // must normalize it away rather than materializing a decoded-frame
        // appsink/appsrc pair.
        decoder.add(simaai::neat::nodes::Output("detector_frame"));
        encoded_fanout_app.connect(source, decoder);
        simaai::neat::GraphLinkOptions realtime;
        realtime.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        realtime.queue_depth = 1;
        realtime.stream_id = "encoded_stream" + std::to_string(stream);
        realtime.max_inflight_per_stream = 1;
        realtime.max_inflight_total = 2;
        encoded_fanout_app.connect(decoder, encoded_fanout_detector, realtime);

        simaai::neat::Graph encoded_output("encoded_branch" + std::to_string(stream));
        encoded_output.add(simaai::neat::nodes::Output(
            "encoded_h264_" + std::to_string(stream),
            simaai::neat::OutputOptions::EveryFrame(/*max_buffers=*/120)));
        simaai::neat::GraphLinkOptions encoded_link;
        encoded_link.stream_id = "encoded_output_stream" + std::to_string(stream);
        encoded_fanout_app.connect(source, encoded_output, encoded_link);
      }
      const auto encoded_fanout_plan =
          simaai::neat::runtime::compile_public_graph(encoded_fanout_app, composed_run_options);
      const simaai::neat::runtime::PipelineSegmentPlan* encoded_fused_segment = nullptr;
      for (const auto& segment : encoded_fanout_plan.pipeline_segments) {
        if (segment.fused_realtime_ingress.has_value()) {
          require(encoded_fused_segment == nullptr,
                  "encoded fan-out topology must create one shared fused consumer");
          encoded_fused_segment = &segment;
        }
      }
      require(encoded_fused_segment != nullptr,
              "ordinary encoded fan-out topology must be eligible for automatic fusion");
      require(encoded_fused_segment->fused_realtime_ingress->branches.size() == 2U,
              "encoded fused topology must preserve both streams");
      require(encoded_fanout_plan.named_outputs.find("detector_frame") ==
                  encoded_fanout_plan.named_outputs.end(),
              "fused decoder boundary must not remain as a public raw-frame Output");
      std::size_t consumed_segments = 0;
      for (const auto& segment : encoded_fanout_plan.pipeline_segments) {
        consumed_segments += segment.consumed_by_fused_realtime_ingress ? 1U : 0U;
      }
      require(consumed_segments == 6U,
              "encoded streams must absorb each source, decoder, and named Output");
      std::size_t consumed_fanouts = 0;
      for (const auto& stage : encoded_fanout_plan.stage_nodes) {
        consumed_fanouts += stage.consumed_by_fused_realtime_ingress ? 1U : 0U;
      }
      require(consumed_fanouts == 2U,
              "generated encoded fan-out stages must not also materialize in graph runtime");

      for (std::size_t stream = 0;
           stream < encoded_fused_segment->fused_realtime_ingress->branches.size(); ++stream) {
        const auto& branch = encoded_fused_segment->fused_realtime_ingress->branches[stream];
        require(branch.encoded_output.has_value(),
                "fused encoded branch must retain its graph-owned Output plan");
        require(branch.encoded_output->options.max_buffers == 120 &&
                    !branch.encoded_output->options.drop,
                "fused encoded Output must preserve EveryFrame queue options");
        require(branch.stream_id.rfind("encoded_stream", 0) == 0,
                "fused encoded branch must preserve its configured stream id");
        const std::string stream_suffix =
            branch.stream_id.substr(std::string("encoded_stream").size());
        require(branch.encoded_output->stream_id == "encoded_output_stream" + stream_suffix,
                "fused encoded Output must preserve the source-to-Output edge stream id (got '" +
                    branch.encoded_output->stream_id + "')");
        const auto named = encoded_fanout_plan.named_outputs.find("encoded_h264_" + stream_suffix);
        require(named != encoded_fanout_plan.named_outputs.end() &&
                    named->second.node == branch.encoded_output->sink_node,
                "each fused encoded Output must remain pullable by its public name");
        require(std::any_of(branch.nodes.begin(), branch.nodes.end(),
                            [](const auto& node) {
                              return dynamic_cast<const simaai::neat::H264Depacketize*>(
                                         node.get()) != nullptr;
                            }),
                "fused encoded branch must retain H.264 depacketization");
        require(std::any_of(branch.nodes.begin(), branch.nodes.end(),
                            [](const auto& node) {
                              return dynamic_cast<const simaai::neat::SimaDecode*>(node.get()) !=
                                     nullptr;
                            }),
                "fused encoded branch must retain hardware decode");
        require(branch.encoded_split_node_index.has_value() &&
                    *branch.encoded_split_node_index < branch.nodes.size() &&
                    branch.nodes[*branch.encoded_split_node_index]->kind() == "H264Parse",
                "the encoded Output tap boundary must remain before decoder-side H.264 parsing");
      }

      const std::string encoded_fanout_pipeline =
          simaai::neat::session_test::render_fused_realtime_pipeline_for_test(
              *encoded_fused_segment->fused_realtime_ingress, encoded_fused_segment->nodes,
              encoded_fused_segment->route_options);
      require(count_occurrences(encoded_fanout_pipeline, "name=neat_encoded_output_tap_b") == 2U,
              "each named encoded Output must materialize exactly one private tap");
      for (std::size_t stream = 0; stream < 2U; ++stream) {
        const std::string branch_suffix = "_b" + std::to_string(stream);
        const std::string tap_name = "name=neat_encoded_output_tap" + branch_suffix;
        const std::size_t tap = encoded_fanout_pipeline.find(tap_name);
        const std::size_t source_caps = encoded_fanout_pipeline.find("_h264_caps" + branch_suffix);
        const std::size_t decoder_caps = encoded_fanout_pipeline.find(
            "_h264_caps" + branch_suffix, tap == std::string::npos ? 0U : tap + tap_name.size());
        require(source_caps != std::string::npos && tap != std::string::npos &&
                    decoder_caps != std::string::npos && source_caps < tap && tap < decoder_caps,
                "the named encoded Output tap must be inserted at the public boundary, not at a "
                "source or decoder capsfilter");
      }

      // Attach the production probe to a synthetic branch containing H.264
      // capsfilters on both sides of the public boundary. One AU must dispatch
      // once: scanning *_h264_caps would incorrectly dispatch it twice.
      simaai::neat::runtime::FusedRealtimeIngress probe_ingress;
      simaai::neat::runtime::FusedRealtimeIngressBranch probe_branch;
      probe_branch.stream_id = "detector-stream";
      probe_branch.encoded_output.emplace();
      probe_branch.encoded_output->stream_id = "public-encoded-stream";
      probe_ingress.branches.push_back(std::move(probe_branch));
      simaai::neat::GraphOptions probe_options;
      const std::string probe_tap =
          simaai::neat::session_test::fused_encoded_output_tap_name_for_test(probe_options, 0U);
      const std::string probe_caps =
          "video/x-h264,parsed=(boolean)true,stream-format=(string)byte-stream,"
          "alignment=(string)au";
      const std::string probe_pipeline_text =
          "appsrc name=encoded_probe_src is-live=false format=time caps=\"" + probe_caps +
          "\" ! capsfilter name=n0_h264_caps_b0 caps=\"" + probe_caps +
          "\" ! identity name=" + probe_tap +
          " silent=true ! capsfilter name=n1_h264_caps_b0 caps=\"" + probe_caps +
          "\" ! appsink name=encoded_probe_sink sync=false";
      GError* probe_parse_error = nullptr;
      GstElement* probe_pipeline =
          gst_parse_launch(probe_pipeline_text.c_str(), &probe_parse_error);
      const std::string probe_parse_message =
          probe_parse_error && probe_parse_error->message ? probe_parse_error->message : "";
      if (probe_parse_error) {
        g_error_free(probe_parse_error);
      }
      require(probe_pipeline != nullptr,
              "failed to parse encoded Output probe fixture: " + probe_parse_message);
      std::atomic<std::size_t> dispatch_count{0U};
      std::atomic<bool> correct_stream{false};
      const std::size_t attached =
          simaai::neat::session_test::attach_fused_encoded_output_probe_for_test(
              probe_pipeline, probe_ingress, probe_options,
              [&](const simaai::neat::Sample& sample) {
                correct_stream.store(sample.stream_id == "public-encoded-stream",
                                     std::memory_order_relaxed);
                dispatch_count.fetch_add(1U, std::memory_order_relaxed);
              });
      require(attached == 1U, "one named encoded Output must attach exactly one dispatch probe");
      GstElement* probe_src = gst_bin_get_by_name(GST_BIN(probe_pipeline), "encoded_probe_src");
      GstElement* probe_sink = gst_bin_get_by_name(GST_BIN(probe_pipeline), "encoded_probe_sink");
      require(probe_src != nullptr && probe_sink != nullptr,
              "encoded Output probe fixture is missing appsrc/appsink");
      require(gst_element_set_state(probe_pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
              "encoded Output probe fixture failed to enter PLAYING");
      const std::array<std::uint8_t, 8> au{0U, 0U, 0U, 1U, 0x65U, 0x88U, 0x84U, 0x21U};
      GstBuffer* probe_buffer = gst_buffer_new_allocate(nullptr, au.size(), nullptr);
      require(probe_buffer != nullptr &&
                  gst_buffer_fill(probe_buffer, 0U, au.data(), au.size()) == au.size(),
              "failed to allocate encoded Output probe AU");
      GST_BUFFER_PTS(probe_buffer) = 1234U;
      GST_BUFFER_DURATION(probe_buffer) = 50000000U;
      require(gst_app_src_push_buffer(GST_APP_SRC(probe_src), probe_buffer) == GST_FLOW_OK,
              "failed to push encoded Output probe AU");
      GstSample* probe_sample = gst_app_sink_try_pull_sample(GST_APP_SINK(probe_sink), GST_SECOND);
      require(probe_sample != nullptr, "encoded Output probe AU did not reach the public boundary");
      gst_sample_unref(probe_sample);
      require(dispatch_count.load(std::memory_order_relaxed) == 1U &&
                  correct_stream.load(std::memory_order_relaxed),
              "one AU must dispatch exactly once with the public encoded stream id");
      (void)gst_app_src_end_of_stream(GST_APP_SRC(probe_src));
      gst_element_set_state(probe_pipeline, GST_STATE_NULL);
      gst_object_unref(probe_src);
      gst_object_unref(probe_sink);
      gst_object_unref(probe_pipeline);

      // Clock synchronization, combine policies, and realtime scheduling on
      // the encoded Output edge require the ordinary segmented runtime. The
      // optimizer must decline those shapes rather than approximating them in
      // a source-pad probe.
      const auto require_unsupported_encoded_output_falls_back =
          [&](simaai::neat::OutputOptions output_options,
              simaai::neat::GraphLinkOptions encoded_link, const char* reason) {
            simaai::neat::Graph app(std::string("encoded_output_fallback_") + reason,
                                    outer_options);
            auto detector = make_composed_consumer_graph();
            for (int stream = 0; stream < 2; ++stream) {
              simaai::neat::nodes::groups::RtspEncodedInputOptions source_options;
              source_options.url = "rtsp://example.test/fallback" + std::to_string(stream);
              source_options.insert_queue = false;
              source_options.auto_caps_from_stream = false;
              source_options.h264_fps = 20;
              source_options.h264_width = 1280;
              source_options.h264_height = 720;
              auto source = simaai::neat::nodes::groups::RtspEncodedInput(source_options);

              simaai::neat::Graph decoder("fallback_decoder" + std::to_string(stream));
              decoder.add(simaai::neat::nodes::SimaDecode());
              decoder.add(simaai::neat::nodes::Output("fallback_detector_frame"));
              simaai::neat::GraphLinkOptions detector_link;
              detector_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
              detector_link.stream_id = "fallback_stream" + std::to_string(stream);
              app.connect(source, decoder);
              app.connect(decoder, detector, detector_link);

              simaai::neat::Graph output("unsupported_encoded_output" + std::to_string(stream));
              output.add(simaai::neat::nodes::Output("unsupported_h264_" + std::to_string(stream),
                                                     output_options));
              app.connect(source, output, encoded_link);
            }

            const auto plan =
                simaai::neat::runtime::compile_public_graph(app, composed_run_options);
            require(std::none_of(plan.pipeline_segments.begin(), plan.pipeline_segments.end(),
                                 [](const auto& segment) {
                                   return segment.fused_realtime_ingress.has_value();
                                 }),
                    std::string("encoded Output fusion must fall back for ") + reason);
          };
      require_unsupported_encoded_output_falls_back(simaai::neat::OutputOptions::Clocked(4), {},
                                                    "clocked output");
      simaai::neat::OutputOptions combined_output = simaai::neat::OutputOptions::EveryFrame(4);
      combined_output.combine_policy = simaai::neat::CombinePolicy::RoundRobin;
      require_unsupported_encoded_output_falls_back(combined_output, {}, "combined output");
      simaai::neat::GraphLinkOptions realtime_encoded_output;
      realtime_encoded_output.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      require_unsupported_encoded_output_falls_back(simaai::neat::OutputOptions::EveryFrame(4),
                                                    realtime_encoded_output,
                                                    "realtime encoded-output link");

      // Encoded fusion concatenates the source and decoder fragments. Until
      // that fused boundary can reproduce arbitrary link semantics, any
      // non-default source-to-decoder contract must retain the segmented path.
      const auto require_nondefault_decoder_link_falls_back =
          [&](simaai::neat::GraphLinkOptions decoder_link, const char* reason) {
            simaai::neat::Graph app(std::string("decoder_link_fallback_") + reason, outer_options);
            auto detector = make_composed_consumer_graph();
            for (int stream = 0; stream < 2; ++stream) {
              simaai::neat::nodes::groups::RtspEncodedInputOptions source_options;
              source_options.url = "rtsp://example.test/decoder-link" + std::to_string(stream);
              source_options.insert_queue = false;
              source_options.auto_caps_from_stream = false;
              source_options.h264_fps = 20;
              source_options.h264_width = 1280;
              source_options.h264_height = 720;
              auto source = simaai::neat::nodes::groups::RtspEncodedInput(source_options);

              simaai::neat::Graph decoder("contract_decoder" + std::to_string(stream));
              decoder.add(simaai::neat::nodes::SimaDecode());
              decoder.add(simaai::neat::nodes::Output("contract_detector_frame"));
              auto stream_decoder_link = decoder_link;
              if (!stream_decoder_link.stream_id.empty()) {
                stream_decoder_link.stream_id += std::to_string(stream);
              }
              app.connect(source, decoder, stream_decoder_link);

              simaai::neat::GraphLinkOptions detector_link;
              detector_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
              detector_link.stream_id = "contract_stream" + std::to_string(stream);
              app.connect(decoder, detector, detector_link);

              simaai::neat::Graph output("contract_encoded_output" + std::to_string(stream));
              output.add(simaai::neat::nodes::Output("contract_h264_" + std::to_string(stream),
                                                     simaai::neat::OutputOptions::EveryFrame(4)));
              app.connect(source, output);
            }

            const auto plan =
                simaai::neat::runtime::compile_public_graph(app, composed_run_options);
            require(std::none_of(plan.pipeline_segments.begin(), plan.pipeline_segments.end(),
                                 [](const auto& segment) {
                                   return segment.fused_realtime_ingress.has_value();
                                 }),
                    std::string("encoded fusion must preserve the segmented path for ") + reason);
          };
      simaai::neat::GraphLinkOptions realtime_decoder_link;
      realtime_decoder_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      realtime_decoder_link.queue_depth = 1;
      realtime_decoder_link.max_inflight_per_stream = 1;
      realtime_decoder_link.max_inflight_total = 2;
      require_nondefault_decoder_link_falls_back(realtime_decoder_link,
                                                 "realtime source-decoder link");
      simaai::neat::GraphLinkOptions identified_decoder_link;
      identified_decoder_link.stream_id = "decoder_input_";
      require_nondefault_decoder_link_falls_back(identified_decoder_link,
                                                 "identified source-decoder link");

      simaai::neat::Graph shared_encoded_output_app("shared_encoded_output_fallback",
                                                    outer_options);
      auto shared_output_detector = make_composed_consumer_graph();
      simaai::neat::Graph shared_encoded_output("shared_encoded_output");
      shared_encoded_output.add(
          simaai::neat::nodes::Output("shared_h264", simaai::neat::OutputOptions::EveryFrame(8)));
      for (int stream = 0; stream < 2; ++stream) {
        simaai::neat::nodes::groups::RtspEncodedInputOptions source_options;
        source_options.url = "rtsp://example.test/shared" + std::to_string(stream);
        source_options.insert_queue = false;
        source_options.auto_caps_from_stream = false;
        source_options.h264_fps = 20;
        source_options.h264_width = 1280;
        source_options.h264_height = 720;
        auto source = simaai::neat::nodes::groups::RtspEncodedInput(source_options);

        simaai::neat::Graph decoder("shared_output_decoder" + std::to_string(stream));
        decoder.add(simaai::neat::nodes::SimaDecode());
        decoder.add(simaai::neat::nodes::Output("shared_output_detector_frame"));
        simaai::neat::GraphLinkOptions detector_link;
        detector_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        detector_link.stream_id = "shared_output_stream" + std::to_string(stream);
        shared_encoded_output_app.connect(source, decoder);
        shared_encoded_output_app.connect(decoder, shared_output_detector, detector_link);

        simaai::neat::GraphLinkOptions output_link;
        output_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        output_link.stream_id = "shared_encoded_stream" + std::to_string(stream);
        shared_encoded_output_app.connect(source, shared_encoded_output, output_link);
      }
      const auto shared_encoded_output_plan = simaai::neat::runtime::compile_public_graph(
          shared_encoded_output_app, composed_run_options);
      require(std::none_of(
                  shared_encoded_output_plan.pipeline_segments.begin(),
                  shared_encoded_output_plan.pipeline_segments.end(),
                  [](const auto& segment) { return segment.fused_realtime_ingress.has_value(); }),
              "an encoded Output segment with multiple producers must remain segmented");

      // The durable App16 topology connects the encoded producer directly to
      // VideoSender. Automatic fusion must render that sink behind an encoded
      // tee rather than materializing an appsink/appsrc transport or CPU copy.
      simaai::neat::Graph direct_video_app("direct_encoded_video_fused_app", outer_options);
      auto direct_video_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        simaai::neat::nodes::groups::RtspEncodedInputOptions source_options;
        source_options.url = "rtsp://example.test/direct" + std::to_string(stream);
        source_options.insert_queue = false;
        source_options.auto_caps_from_stream = false;
        source_options.h264_fps = 20;
        source_options.h264_width = 1280;
        source_options.h264_height = 720;
        auto source = simaai::neat::nodes::groups::RtspEncodedInput(source_options);

        simaai::neat::Graph decoder("direct_decoder" + std::to_string(stream));
        decoder.add(simaai::neat::nodes::SimaDecode());
        decoder.add(simaai::neat::nodes::Output("direct_detector_frame"));

        simaai::neat::GraphLinkOptions realtime;
        realtime.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        realtime.queue_depth = 1;
        realtime.stream_id = "direct_stream" + std::to_string(stream);
        realtime.max_inflight_per_stream = 1;
        realtime.max_inflight_total = 2;
        direct_video_app.connect(source, decoder);
        direct_video_app.connect(decoder, direct_video_detector, realtime);

        auto video_options =
            simaai::neat::nodes::groups::VideoSenderOptions::H264RtpUdpFromEncoded();
        video_options.host = "127.0.0.1";
        video_options.channel = stream;
        auto video_sender = simaai::neat::nodes::groups::VideoSender(video_options);
        video_sender.set_name("direct_video_sender_" + std::to_string(stream));
        simaai::neat::GraphLinkOptions video_link;
        if (stream == 1) {
          video_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        }
        direct_video_app.connect(source, video_sender, video_link);
      }
      const auto direct_video_plan =
          simaai::neat::runtime::compile_public_graph(direct_video_app, composed_run_options);
      const simaai::neat::runtime::PipelineSegmentPlan* direct_video_fused = nullptr;
      for (const auto& segment : direct_video_plan.pipeline_segments) {
        if (segment.fused_realtime_ingress.has_value()) {
          require(direct_video_fused == nullptr,
                  "direct encoded VideoSender topology must create one fused consumer");
          direct_video_fused = &segment;
        }
      }
      require(direct_video_fused != nullptr &&
                  direct_video_fused->fused_realtime_ingress->branches.size() == 2U,
              "direct encoded VideoSender topology must fuse both streams");
      const auto direct_detections = direct_video_plan.named_outputs.find("detections");
      require(direct_detections != direct_video_plan.named_outputs.end(),
              "direct VideoSender fusion must preserve the detector Output");
      require(direct_video_plan.output_endpoints.size() == 1U &&
                  direct_video_plan.default_output.has_value() &&
                  direct_video_plan.default_output->node == direct_detections->second.node,
              "consumed VideoSender UDP sinks must not become phantom pull endpoints");
      for (const auto& branch : direct_video_fused->fused_realtime_ingress->branches) {
        require(!branch.encoded_output.has_value(),
                "direct VideoSender fusion must not synthesize a public encoded Output");
        require(branch.encoded_sink_nodes.size() == 3U &&
                    branch.encoded_sink_nodes[0]->kind() == "H264Parse" &&
                    branch.encoded_sink_nodes[1]->kind() == "H264Packetize" &&
                    branch.encoded_sink_nodes[2]->kind() == "UdpOutput",
                "direct VideoSender fusion must retain the parser, packetizer, and sink");
        require(branch.encoded_split_node_index.has_value() &&
                    *branch.encoded_split_node_index < branch.nodes.size() &&
                    branch.nodes[*branch.encoded_split_node_index]->kind() == "SimaDecode",
                "standard direct VideoSender fusion must tee at the source/decoder boundary");
        const auto expected_video_policy =
            branch.stream_id == "direct_stream1"
                ? simaai::neat::GraphLinkPolicy::RealtimeLatestByStream
                : simaai::neat::GraphLinkPolicy::Default;
        require(branch.encoded_sink_link_options.policy == expected_video_policy,
                "direct VideoSender fusion must retain the encoded edge policy");
      }
      std::size_t direct_consumed_segments = 0;
      for (const auto& segment : direct_video_plan.pipeline_segments) {
        direct_consumed_segments += segment.consumed_by_fused_realtime_ingress ? 1U : 0U;
      }
      require(direct_consumed_segments == 6U,
              "each direct video stream must absorb its source, decoder, and sender segment");

      const std::string direct_video_pipeline =
          simaai::neat::session_test::render_fused_realtime_pipeline_for_test(
              *direct_video_fused->fused_realtime_ingress, direct_video_fused->nodes,
              direct_video_fused->route_options);
      const std::string lossless_decoder_queue =
          " ! queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! neatdecoder";
      const std::string leaky_video_queue =
          "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream ! "
          "h264parse";
      const std::string lossless_video_queue =
          "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 ! h264parse";
      require(count_occurrences(direct_video_pipeline, lossless_decoder_queue) == 2U,
              "a queue-less encoded prefix must retain one lossless decoder queue per stream");
      require(count_occurrences(direct_video_pipeline, lossless_video_queue) == 1U,
              "a default VideoSender edge must retain lossless one-AU backpressure");
      require(count_occurrences(direct_video_pipeline, leaky_video_queue) == 1U,
              "a realtime-latest VideoSender edge must retain one-AU replacement");
      require(count_occurrences(
                  direct_video_pipeline,
                  "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream") ==
                  1U,
              "latest mux input must not add a second decoded-EV buffer per stream");

      // RtspEncodedInput normally ends its encoded prefix in the framework's
      // typed Queue. That queue already provides the tee task boundary, so a
      // second lossless queue on every decoder branch only adds scheduler and
      // buffering overhead at 24/48 streams.
      auto typed_queue_ingress = *direct_video_fused->fused_realtime_ingress;
      for (auto& branch : typed_queue_ingress.branches) {
        const std::size_t split = *branch.encoded_split_node_index;
        branch.nodes.insert(branch.nodes.begin() + static_cast<std::ptrdiff_t>(split),
                            simaai::neat::nodes::Queue());
        branch.encoded_split_node_index = split + 1U;
        require(dynamic_cast<const simaai::neat::Queue*>(branch.nodes[split].get()) != nullptr,
                "the optimized prefix test must use the framework's typed Queue");
      }
      const std::string typed_queue_pipeline =
          simaai::neat::session_test::render_fused_realtime_pipeline_for_test(
              typed_queue_ingress, direct_video_fused->nodes, direct_video_fused->route_options);
      require(count_occurrences(typed_queue_pipeline, lossless_decoder_queue) == 0U,
              "a typed prefix Queue must suppress the redundant lossless decoder queue");
      require(count_occurrences(typed_queue_pipeline, ". ! neatdecoder") == 2U,
              "each typed-Queue prefix tee must feed its decoder directly");
      require(count_occurrences(typed_queue_pipeline, lossless_video_queue) == 1U &&
                  count_occurrences(typed_queue_pipeline, leaky_video_queue) == 1U,
              "suppressing the decoder queue must preserve each encoded edge policy");

      // A custom Node is allowed to use the label \"Queue\". Do not infer its
      // scheduling or loss behavior from kind() alone.
      auto custom_queue_ingress = *direct_video_fused->fused_realtime_ingress;
      for (auto& branch : custom_queue_ingress.branches) {
        const std::size_t split = *branch.encoded_split_node_index;
        branch.nodes.insert(branch.nodes.begin() + static_cast<std::ptrdiff_t>(split),
                            std::make_shared<FragmentNode>("Queue", "queue", "customer_queue"));
        branch.encoded_split_node_index = split + 1U;
      }
      const std::string custom_queue_pipeline =
          simaai::neat::session_test::render_fused_realtime_pipeline_for_test(
              custom_queue_ingress, direct_video_fused->nodes, direct_video_fused->route_options);
      require(count_occurrences(custom_queue_pipeline, lossless_decoder_queue) == 2U,
              "a kind-only custom Queue must retain the lossless decoder queue fallback");
      require(count_occurrences(direct_video_pipeline, "rtph264pay name=neat_fused_pay_") == 2U,
              "each fused direct VideoSender branch must render its own RTP packetizer name");
      require(direct_video_pipeline.find("rtph264pay name=pay0") == std::string::npos,
              "fused direct VideoSender branches must not retain the fixed pay0 name");
      const std::size_t first_pay = direct_video_pipeline.find("name=neat_fused_pay_");
      const std::size_t second_pay = direct_video_pipeline.find(
          "name=neat_fused_pay_", first_pay + std::string("name=neat_fused_pay_").size());
      const std::size_t first_pay_end = direct_video_pipeline.find(' ', first_pay);
      const std::size_t second_pay_end = direct_video_pipeline.find(' ', second_pay);
      require(first_pay != std::string::npos && second_pay != std::string::npos &&
                  direct_video_pipeline.substr(first_pay, first_pay_end - first_pay) !=
                      direct_video_pipeline.substr(second_pay, second_pay_end - second_pay),
              "fused direct VideoSender RTP packetizer names must be unique");
      GError* direct_video_parse_error = nullptr;
      GstElement* direct_video_parsed =
          gst_parse_launch(direct_video_pipeline.c_str(), &direct_video_parse_error);
      const std::string direct_video_parse_message =
          direct_video_parse_error && direct_video_parse_error->message
              ? direct_video_parse_error->message
              : std::string{};
      if (direct_video_parse_error) {
        g_error_free(direct_video_parse_error);
      }
      if (direct_video_parsed) {
        gst_object_unref(direct_video_parsed);
      }
      require(direct_video_parsed != nullptr && direct_video_parse_message.empty(),
              "the full fused direct VideoSender pipeline must parse without element-name "
              "collisions: " +
                  direct_video_parse_message);

      // Kind-based recognition must preserve a customer-configured parser's
      // caps/header behavior exactly.
      simaai::neat::Graph custom_video_app("custom_encoded_video_fused_app", outer_options);
      auto custom_video_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        simaai::neat::nodes::groups::RtspEncodedInputOptions source_options;
        source_options.url = "rtsp://example.test/custom" + std::to_string(stream);
        source_options.insert_queue = false;
        source_options.auto_caps_from_stream = false;
        source_options.h264_fps = 20;
        source_options.h264_width = 1280;
        source_options.h264_height = 720;
        auto source = simaai::neat::nodes::groups::RtspEncodedInput(source_options);

        simaai::neat::Graph decoder("custom_decoder" + std::to_string(stream));
        simaai::neat::H264ParseOptions decoder_parser_options;
        decoder_parser_options.config_interval = -1;
        decoder.add(simaai::neat::nodes::H264Parse(decoder_parser_options));
        decoder.add(simaai::neat::nodes::SimaDecode());
        decoder.add(simaai::neat::nodes::Output("custom_detector_frame"));
        simaai::neat::GraphLinkOptions realtime;
        realtime.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        realtime.stream_id = "custom_stream" + std::to_string(stream);
        custom_video_app.connect(source, decoder);
        custom_video_app.connect(decoder, custom_video_detector, realtime);

        simaai::neat::H264ParseOptions parser_options;
        parser_options.config_interval = 1;
        parser_options.enforce_caps = true;
        parser_options.alignment = simaai::neat::H264ParseOptions::Alignment::AU;
        parser_options.stream_format = simaai::neat::H264ParseOptions::StreamFormat::ByteStream;
        simaai::neat::Graph custom_sender("custom_sender" + std::to_string(stream));
        custom_sender.add(simaai::neat::nodes::H264Parse(parser_options));
        custom_sender.add(
            simaai::neat::nodes::H264Packetize(simaai::neat::H264Packetize::PayloadType(96),
                                               simaai::neat::H264Packetize::ConfigInterval(5)));
        simaai::neat::UdpOutputOptions udp_options;
        udp_options.port = 9200 + stream;
        custom_sender.add(simaai::neat::nodes::UdpOutput(udp_options));
        custom_video_app.connect(source, custom_sender);
      }
      const auto custom_video_plan =
          simaai::neat::runtime::compile_public_graph(custom_video_app, composed_run_options);
      const auto custom_video_fused = std::find_if(
          custom_video_plan.pipeline_segments.begin(), custom_video_plan.pipeline_segments.end(),
          [](const auto& segment) { return segment.fused_realtime_ingress.has_value(); });
      require(custom_video_fused != custom_video_plan.pipeline_segments.end(),
              "custom encoded VideoSender-shaped topology must remain eligible for fusion");
      for (const auto& branch : custom_video_fused->fused_realtime_ingress->branches) {
        require(branch.encoded_sink_nodes.size() == 3U &&
                    branch.encoded_sink_nodes.front()->kind() == "H264Parse",
                "fusion must preserve a custom encoded parser with distinct semantics");
        require(branch.encoded_split_node_index.has_value() &&
                    *branch.encoded_split_node_index + 1U < branch.nodes.size() &&
                    branch.nodes[*branch.encoded_split_node_index]->kind() == "H264Parse" &&
                    branch.nodes[*branch.encoded_split_node_index + 1U]->kind() == "SimaDecode",
                "fusion must tee before decoder-fragment preprocessing rather than moving the "
                "public fan-out boundary to SimaDecode");
      }

      int unsafe_stream_case = 0;
      for (const std::string& unsafe_stream_id :
           std::vector<std::string>{"camera,0", " camera0", "camera0 "}) {
        simaai::neat::Graph unsafe_stream_app(
            "unsafe_fused_stream_id_" + std::to_string(unsafe_stream_case), outer_options);
        auto unsafe_stream_detector = make_composed_consumer_graph();
        for (int stream = 0; stream < 2; ++stream) {
          simaai::neat::GraphLinkOptions link;
          link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
          link.stream_id = stream == 0 ? unsafe_stream_id : "safe-camera1";
          unsafe_stream_app.connect(make_live_source_graph(500 + unsafe_stream_case * 2 + stream),
                                    unsafe_stream_detector, link);
        }
        const auto unsafe_stream_plan =
            simaai::neat::runtime::compile_public_graph(unsafe_stream_app, composed_run_options);
        require(std::none_of(
                    unsafe_stream_plan.pipeline_segments.begin(),
                    unsafe_stream_plan.pipeline_segments.end(),
                    [](const auto& segment) { return segment.fused_realtime_ingress.has_value(); }),
                "fusion must fall back instead of changing a CSV-unsafe stream id");
        require(std::any_of(unsafe_stream_plan.edges.begin(), unsafe_stream_plan.edges.end(),
                            [&](const auto& edge) { return edge.stream_id == unsafe_stream_id; }),
                "segmented fallback must preserve the exact public stream id");
        ++unsafe_stream_case;
      }

      const auto require_duplicate_stream_plan_falls_back = [&](const char* name,
                                                                const std::array<std::string, 2>&
                                                                    ids) {
        simaai::neat::Graph duplicate_stream_app(std::string("duplicate_fused_stream_id_") + name,
                                                 outer_options);
        auto duplicate_stream_detector = make_composed_consumer_graph();
        for (std::size_t stream = 0; stream < ids.size(); ++stream) {
          simaai::neat::GraphLinkOptions link;
          link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
          link.stream_id = ids[stream];
          duplicate_stream_app.connect(make_live_source_graph(550 + static_cast<int>(stream)),
                                       duplicate_stream_detector, link);
        }
        const auto duplicate_stream_plan =
            simaai::neat::runtime::compile_public_graph(duplicate_stream_app, composed_run_options);
        require(std::none_of(
                    duplicate_stream_plan.pipeline_segments.begin(),
                    duplicate_stream_plan.pipeline_segments.end(),
                    [](const auto& segment) { return segment.fused_realtime_ingress.has_value(); }),
                std::string("fusion must decline duplicate effective stream ids: ") + name);
        require(
            std::none_of(duplicate_stream_plan.edges.begin(), duplicate_stream_plan.edges.end(),
                         [](const auto& edge) { return edge.consumed_by_fused_realtime_ingress; }),
            std::string("duplicate-id fallback must leave the segmented plan intact: ") + name);
        for (const std::string& configured_id : ids) {
          if (configured_id.empty()) {
            continue;
          }
          require(
              std::any_of(duplicate_stream_plan.edges.begin(), duplicate_stream_plan.edges.end(),
                          [&](const auto& edge) {
                            return edge.link_options.policy ==
                                       simaai::neat::GraphLinkPolicy::RealtimeLatestByStream &&
                                   edge.stream_id == configured_id;
                          }),
              std::string("segmented fallback must preserve the configured stream id: ") + name);
        }
      };
      require_duplicate_stream_plan_falls_back(
          "explicit_duplicate", std::array<std::string, 2>{"shared-camera", "shared-camera"});

      // Public composition assigns deterministic ids to otherwise unnamed
      // live fan-in edges. Rebuild the same topology with one edge explicitly
      // claiming the other edge's generated id and verify the compiled plan
      // stays segmented rather than collapsing the two branches.
      simaai::neat::Graph generated_id_baseline("generated_id_baseline", outer_options);
      auto generated_id_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        simaai::neat::GraphLinkOptions link;
        link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        generated_id_baseline.connect(make_live_source_graph(570 + stream), generated_id_detector,
                                      link);
      }
      const auto generated_id_plan =
          simaai::neat::runtime::compile_public_graph(generated_id_baseline, composed_run_options);
      const auto generated_id_fused = std::find_if(
          generated_id_plan.pipeline_segments.begin(), generated_id_plan.pipeline_segments.end(),
          [](const auto& segment) { return segment.fused_realtime_ingress.has_value(); });
      require(generated_id_fused != generated_id_plan.pipeline_segments.end() &&
                  generated_id_fused->fused_realtime_ingress->branches.size() == 2U,
              "unnamed live fan-in baseline must resolve two generated stream ids");
      const std::string generated_collision_id =
          generated_id_fused->fused_realtime_ingress->branches.front().stream_id;
      require_duplicate_stream_plan_falls_back(
          "explicit_generated_collision", std::array<std::string, 2>{generated_collision_id, ""});

      simaai::neat::Graph heterogeneous_caps_app("heterogeneous_caps_fallback", outer_options);
      auto heterogeneous_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        simaai::neat::GraphLinkOptions link;
        link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        link.stream_id = "heterogeneous_stream" + std::to_string(stream);
        heterogeneous_caps_app.connect(
            make_live_source_graph(600 + stream, 1280U, 720U, stream == 0 ? 20U : 10U),
            heterogeneous_detector, link);
      }
      const auto heterogeneous_caps_plan =
          simaai::neat::runtime::compile_public_graph(heterogeneous_caps_app, composed_run_options);
      require(std::none_of(
                  heterogeneous_caps_plan.pipeline_segments.begin(),
                  heterogeneous_caps_plan.pipeline_segments.end(),
                  [](const auto& segment) { return segment.fused_realtime_ingress.has_value(); }),
              "known heterogeneous branch framerates must fall back to segmented fan-in");
      require(
          std::none_of(heterogeneous_caps_plan.edges.begin(), heterogeneous_caps_plan.edges.end(),
                       [](const auto& edge) { return edge.consumed_by_fused_realtime_ingress; }),
          "caps fallback must not partially consume realtime edges");

      simaai::neat::Graph single_source_app("single_source_not_fused", outer_options);
      auto single_source = make_live_source_graph(99);
      auto single_detector = make_composed_consumer_graph();
      simaai::neat::GraphLinkOptions single_link;
      single_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      single_link.max_inflight_per_stream = 1;
      single_source_app.connect(single_source, single_detector, single_link);
      const auto single_source_plan =
          simaai::neat::runtime::compile_public_graph(single_source_app, composed_run_options);
      for (const auto& segment : single_source_plan.pipeline_segments) {
        require(!segment.fused_realtime_ingress.has_value(),
                "automatic fusion must leave ineligible one-source topology unchanged");
      }

      simaai::neat::GraphOptions synchronous;
      synchronous.processcvu.async = false;
      synchronous.processmla.async = false;
      synchronous.processmla.defer_output_invalidate = false;
      const std::string synchronous_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              make_consumer_nodes(), synchronous);
      require(synchronous_pipeline.find("async=true") == std::string::npos,
              "explicit public synchronous options must not enable fused async stages");
      require_contains(synchronous_pipeline, "defer-output-invalidate=false",
                       "explicit public cache-sync option must be preserved by fused rendering");
    }));
