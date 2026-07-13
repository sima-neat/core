#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "builder/Node.h"
#include "gst/GstInit.h"
#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/io/CameraInput.h"
#include "nodes/io/Input.h"
#include "nodes/io/RTSPInput.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/graph/internal/GraphBuildInternal.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

// GraphLinkOptions is passed by reference across the public shared-library boundary.  Keep the
// exact v0.2.x aggregate layout so a new library never reads beyond an object constructed by an
// already-deployed caller.  Realtime admission limits live in the separately named derived type.
struct ReleasedGraphLinkOptionsLayout {
  simaai::neat::GraphLinkPolicy policy = simaai::neat::GraphLinkPolicy::Default;
  int queue_depth = 16;
  std::string stream_id;
};

template <typename GraphType>
concept AcceptsReleasedBraceConnect = requires(
    GraphType& graph, const GraphType& from, const GraphType& to) { graph.connect(from, to, {}); };

static_assert(std::is_standard_layout_v<simaai::neat::GraphLinkOptions>);
static_assert(std::is_aggregate_v<simaai::neat::GraphLinkOptions>);
static_assert(sizeof(simaai::neat::GraphLinkOptions) == sizeof(ReleasedGraphLinkOptionsLayout));
static_assert(alignof(simaai::neat::GraphLinkOptions) == alignof(ReleasedGraphLinkOptionsLayout));
static_assert(offsetof(simaai::neat::GraphLinkOptions, policy) ==
              offsetof(ReleasedGraphLinkOptionsLayout, policy));
static_assert(offsetof(simaai::neat::GraphLinkOptions, queue_depth) ==
              offsetof(ReleasedGraphLinkOptionsLayout, queue_depth));
static_assert(offsetof(simaai::neat::GraphLinkOptions, stream_id) ==
              offsetof(ReleasedGraphLinkOptionsLayout, stream_id));
static_assert(
    std::is_base_of_v<simaai::neat::GraphLinkOptions, simaai::neat::RealtimeGraphLinkOptions>);
static_assert(sizeof(simaai::neat::RealtimeGraphLinkOptions) >
              sizeof(simaai::neat::GraphLinkOptions));
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

simaai::neat::Graph make_live_source_graph(int index) {
  simaai::neat::CameraInputOptions options;
  options.buffer_name = "fused_queue_cam" + std::to_string(index);
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
      require(pipeline.find("block-when-pending") == std::string::npos,
              "default latest links must not opt into producer backpressure");

      setenv("SIMA_GRAPH_REALTIME_CREDIT_MAX_INFLIGHT_GLOBAL", "3", 1);
      const std::string env_limited_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              make_consumer_nodes(), options);
      require_contains(env_limited_pipeline, "max-inflight-total=3",
                       "fused default admission must honor the realtime global env override");

      simaai::neat::RealtimeGraphLinkOptions explicit_total_link;
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

      simaai::neat::RealtimeGraphLinkOptions latest_ingress;
      latest_ingress.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      const std::string latest_ingress_queue =
          simaai::neat::session_test::render_fused_realtime_ingress_queue_for_test(latest_ingress);
      require_contains(latest_ingress_queue, "leaky=downstream",
                       "default latest ingress must retain its non-blocking leaky queue");
      simaai::neat::RealtimeGraphLinkOptions every_frame_ingress;
      every_frame_ingress.policy = simaai::neat::GraphLinkPolicy::RealtimeEveryFrameByStream;
      const std::string every_frame_ingress_queue =
          simaai::neat::session_test::render_fused_realtime_ingress_queue_for_test(
              every_frame_ingress);
      require(every_frame_ingress_queue.find("leaky=") == std::string::npos,
              "every-frame ingress queue must backpressure instead of dropping upstream");

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

      simaai::neat::GraphOptions outer_options;
      outer_options.async_queue_depth = 3;
      simaai::neat::Graph composed_app("composed_fused_queue_app", outer_options);
      auto composed_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        auto source = make_live_source_graph(stream);
        simaai::neat::RealtimeGraphLinkOptions link;
        link.policy = simaai::neat::GraphLinkPolicy::RealtimeEveryFrameByStream;
        link.queue_depth = 1;
        link.stream_id = "fused_queue_stream" + std::to_string(stream);
        link.max_inflight_per_stream = stream == 0 ? 1 : 8;
        link.max_inflight_total = stream == 0 ? 2 : 8;
        composed_app.connect(source, composed_detector, link);
      }
      simaai::neat::RunOptions composed_run_options;
      composed_run_options.queue_depth = 1;
      const auto default_plan =
          simaai::neat::runtime::compile_public_graph(composed_app, composed_run_options);
      for (const auto& segment : default_plan.pipeline_segments) {
        require(!segment.fused_realtime_ingress.has_value(),
                "the ABI-compatible build path must not silently opt old callers into fusion");
      }
      const auto composed_plan = simaai::neat::runtime::compile_public_graph(
          composed_app, composed_run_options, std::nullopt, true);
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
      std::vector<simaai::neat::RealtimeGraphLinkOptions> fused_link_options;
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
        require(branch.link_options.policy ==
                    simaai::neat::GraphLinkPolicy::RealtimeEveryFrameByStream,
                "fused ingress must preserve the every-frame backpressure policy");
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
      require_contains(composed_pipeline, "block-when-pending=true",
                       "every-frame public links must enable bounded mux backpressure");
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

      // Exercise the released three-member object through the unchanged public connect symbol.
      // The new library must synthesize default admission values instead of reading a tail that an
      // old caller did not allocate.
      simaai::neat::Graph released_options_app("released_graph_link_options", outer_options);
      auto released_options_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        simaai::neat::GraphLinkOptions released_link{
            simaai::neat::GraphLinkPolicy::RealtimeEveryFrameByStream,
            1,
            "released_stream" + std::to_string(stream),
        };
        released_options_app.connect(make_live_source_graph(300 + stream),
                                     released_options_detector, released_link);
      }
      const auto released_options_plan = simaai::neat::runtime::compile_public_graph(
          released_options_app, composed_run_options, std::nullopt, true);
      const simaai::neat::runtime::PipelineSegmentPlan* released_options_segment = nullptr;
      for (const auto& segment : released_options_plan.pipeline_segments) {
        if (segment.fused_realtime_ingress.has_value()) {
          released_options_segment = &segment;
          break;
        }
      }
      require(released_options_segment != nullptr,
              "released GraphLinkOptions path must remain eligible for explicit fusion");
      std::vector<simaai::neat::RealtimeGraphLinkOptions> released_options_links;
      for (const auto& branch : released_options_segment->fused_realtime_ingress->branches) {
        require(branch.link_options.max_inflight_per_stream == -1 &&
                    branch.link_options.max_inflight_total == -1,
                "released GraphLinkOptions must acquire safe default admission values");
        released_options_links.push_back(branch.link_options);
      }
      const std::string released_options_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              released_options_segment->nodes, released_options_segment->route_options,
              released_options_links);
      require_contains(released_options_pipeline, "stream-inflight-limits=\"4,4\"",
                       "released GraphLinkOptions must resolve the framework per-stream default");

      simaai::neat::Graph mixed_policy_app("mixed_realtime_policy_app", outer_options);
      auto mixed_detector = make_composed_consumer_graph();
      simaai::neat::RealtimeGraphLinkOptions latest_link;
      latest_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      mixed_policy_app.connect(make_live_source_graph(200), mixed_detector, latest_link);
      simaai::neat::RealtimeGraphLinkOptions every_frame_link;
      every_frame_link.policy = simaai::neat::GraphLinkPolicy::RealtimeEveryFrameByStream;
      bool rejected_mixed_policy = false;
      try {
        mixed_policy_app.connect(make_live_source_graph(201), mixed_detector, every_frame_link);
      } catch (const std::runtime_error&) {
        rejected_mixed_policy = true;
      }
      require(rejected_mixed_policy,
              "one fan-in must reject mixed latest and every-frame producer policies");

      simaai::neat::Graph single_source_app("single_source_not_fused", outer_options);
      auto single_source = make_live_source_graph(99);
      auto single_detector = make_composed_consumer_graph();
      simaai::neat::RealtimeGraphLinkOptions single_link;
      single_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      single_link.max_inflight_per_stream = 1;
      single_source_app.connect(single_source, single_detector, single_link);
      const auto single_source_plan = simaai::neat::runtime::compile_public_graph(
          single_source_app, composed_run_options, std::nullopt, true);
      for (const auto& segment : single_source_plan.pipeline_segments) {
        require(!segment.fused_realtime_ingress.has_value(),
                "explicit fused build must leave ineligible one-source topology unchanged");
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
