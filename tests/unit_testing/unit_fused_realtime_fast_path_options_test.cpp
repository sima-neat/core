#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "builder/Node.h"
#include "pipeline/GraphOptions.h"
#include "pipeline/graph/internal/GraphTestHooks.h"
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
      "defer-output-invalidate=false ! neatboxdecode name=n0_boxdecode"));
  return nodes;
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

RUN_TEST("unit_fused_realtime_fast_path_options_test", ([] {
           ScopedUnsetEnv cvu_kill_switch("SIMA_PROCESSCVU_ASYNC");
           ScopedUnsetEnv mla_kill_switch("SIMA_PROCESSMLA_SAFE_ASYNC");

           simaai::neat::GraphOptions options;
           options.processcvu.async = true;
           options.processmla.async = true;
           options.processmla.output_pool_buffers = 7;
           options.processmla.defer_output_invalidate = true;

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

           simaai::neat::GraphOptions synchronous;
           synchronous.processcvu.async = false;
           synchronous.processmla.async = false;
           synchronous.processmla.defer_output_invalidate = false;
           const std::string synchronous_pipeline =
               simaai::neat::session_test::render_fused_realtime_consumer_pipeline_for_test(
                   make_consumer_nodes(), synchronous);
           require(synchronous_pipeline.find("async=true") == std::string::npos,
                   "explicit public synchronous options must not enable fused async stages");
           require_contains(
               synchronous_pipeline, "defer-output-invalidate=false",
               "explicit public cache-sync option must be preserved by fused rendering");
         }));
