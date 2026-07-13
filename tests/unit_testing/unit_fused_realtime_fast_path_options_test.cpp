#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "builder/Node.h"
#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/io/CameraInput.h"
#include "nodes/io/Input.h"
#include "pipeline/Graph.h"
#include "pipeline/GraphOptions.h"
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

} // namespace

RUN_TEST(
    "unit_fused_realtime_fast_path_options_test", ([] {
      ScopedUnsetEnv cvu_kill_switch("SIMA_PROCESSCVU_ASYNC");
      ScopedUnsetEnv mla_kill_switch("SIMA_PROCESSMLA_SAFE_ASYNC");
      ScopedUnsetEnv legacy_queue_enable("SIMA_ENABLE_ASYNC_QUEUE2");
      ScopedUnsetEnv legacy_queue_depth("SIMA_ASYNC_QUEUE2_DEPTH");

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
      simaai::neat::GraphLinkOptions latest_ingress;
      latest_ingress.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      const std::string latest_ingress_queue =
          simaai::neat::session_test::render_fused_realtime_ingress_queue_for_test(latest_ingress);
      require_contains(latest_ingress_queue, "leaky=downstream",
                       "default latest ingress must retain its non-blocking leaky queue");
      simaai::neat::GraphLinkOptions every_frame_ingress;
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

      simaai::neat::GraphOptions outer_options;
      outer_options.async_queue_depth = 3;
      simaai::neat::Graph composed_app("composed_fused_queue_app", outer_options);
      auto composed_detector = make_composed_consumer_graph();
      for (int stream = 0; stream < 2; ++stream) {
        auto source = make_live_source_graph(stream);
        simaai::neat::GraphLinkOptions link;
        link.policy = simaai::neat::GraphLinkPolicy::RealtimeEveryFrameByStream;
        link.queue_depth = 1;
        link.stream_id = "fused_queue_stream" + std::to_string(stream);
        link.max_inflight_per_stream = 1;
        link.max_inflight_total = 2;
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
      std::vector<simaai::neat::GraphLinkOptions> fused_link_options;
      for (const auto& branch : fused_segment->fused_realtime_ingress->branches) {
        require(branch.link_options.max_inflight_per_stream == 1,
                "fused ingress must preserve each public per-link admission limit");
        require(branch.link_options.max_inflight_total == 2,
                "fused ingress must preserve the public total admission limit");
        require(branch.link_options.policy ==
                    simaai::neat::GraphLinkPolicy::RealtimeEveryFrameByStream,
                "fused ingress must preserve the every-frame backpressure policy");
        fused_link_options.push_back(branch.link_options);
      }
      const std::string composed_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              fused_segment->nodes, fused_segment->route_options, fused_link_options);
      require_contains(composed_pipeline, "stream-inflight-limits=\"1,1\"",
                       "fused mux must receive public per-link admission limits");
      require_contains(composed_pipeline, "max-inflight-total=2",
                       "fused mux must receive the public mux-wide admission limit");
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

      simaai::neat::Graph mixed_policy_app("mixed_realtime_policy_app", outer_options);
      auto mixed_detector = make_composed_consumer_graph();
      simaai::neat::GraphLinkOptions latest_link;
      latest_link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
      mixed_policy_app.connect(make_live_source_graph(200), mixed_detector, latest_link);
      simaai::neat::GraphLinkOptions every_frame_link;
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
      simaai::neat::GraphLinkOptions single_link;
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
