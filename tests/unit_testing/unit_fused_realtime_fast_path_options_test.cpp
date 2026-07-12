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

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

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
      ScopedUnsetEnv queue_placement("SIMA_FUSED_REALTIME_CONSUMER_QUEUE_PLACEMENT");

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
      require_contains(pipeline, "stream-ids=\"stream0\" ! " + queue + " ! neatprocesscvu",
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
        link.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
        link.queue_depth = 1;
        link.stream_id = "fused_queue_stream" + std::to_string(stream);
        link.max_inflight_per_stream = 1;
        composed_app.connect(source, composed_detector, link);
      }
      simaai::neat::RunOptions composed_run_options;
      composed_run_options.queue_depth = 1;
      composed_run_options.advanced.fuse_realtime_source_branches = true;
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
      const std::string composed_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              fused_segment->nodes, fused_segment->route_options);
      const std::string composed_queue =
          "queue max-size-buffers=3 max-size-bytes=0 max-size-time=0";
      require(count_occurrences(composed_pipeline, composed_queue) == 3U,
              "actual composed fused graph must render the three selected stage queues");
      require(composed_pipeline.find(composed_queue + " ! appsink") == std::string::npos,
              "actual composed fused graph must not queue before its terminal Output");

      setenv("SIMA_FUSED_REALTIME_CONSUMER_QUEUE_PLACEMENT", "post-mla", 1);
      simaai::neat::GraphOptions post_mla_only = options;
      post_mla_only.async_queue_depth = 1;
      const std::string post_mla_pipeline =
          simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
              make_consumer_nodes(), post_mla_only);
      unsetenv("SIMA_FUSED_REALTIME_CONSUMER_QUEUE_PLACEMENT");
      const std::string depth_one_queue =
          "queue max-size-buffers=1 max-size-bytes=0 max-size-time=0";
      require(count_occurrences(post_mla_pipeline, depth_one_queue) == 1U,
              "post-MLA diagnostic placement must insert exactly one queue");
      require_contains(post_mla_pipeline,
                       "defer-output-invalidate=true ! " + depth_one_queue + " ! neatboxdecode",
                       "post-MLA diagnostic queue must decouple MLA from decode");
      require(post_mla_pipeline.find("stream-ids=\"stream0\" ! " + depth_one_queue) ==
                  std::string::npos,
              "post-MLA placement must not insert a mux-to-CVU queue");
      require(post_mla_pipeline.find(depth_one_queue + " ! neatprocessmla") == std::string::npos,
              "post-MLA placement must not insert a CVU-to-MLA queue");
      require_contains(post_mla_pipeline,
                       "neatboxdecode name=n0_boxdecode ! appsink name=n0_output",
                       "post-MLA placement must not queue before terminal Output");

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
