#include "graph/GraphBuild.h"
#include "internal/GraphRunState.h"
#include "pipeline/internal/UxLogging.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "pipeline/runtime/RunCore.h"

#include <stdexcept>
#include <string>
#include <utility>

namespace simaai::neat::graph {
namespace {

simaai::neat::runtime::GraphRuntimeOptions make_graph_runtime_options(const GraphRunOptions& opt) {
  simaai::neat::runtime::GraphRuntimeOptions out;
  out.edge_queue = opt.edge_queue;
  out.push_timeout_ms = opt.push_timeout_ms;
  out.pull_timeout_ms = opt.pull_timeout_ms;
  out.verbose = opt.verbose;
  out.pipeline = opt.pipeline;
  out.power_monitor = opt.power_monitor;
  return out;
}

} // namespace

GraphRun build(Graph graph, const GraphRunOptions& opt) {
  if (opt.push_timeout_ms < 0) {
    throw std::invalid_argument("graph::build: GraphRunOptions.push_timeout_ms must be >= 0");
  }
  if (opt.pull_timeout_ms < 0) {
    throw std::invalid_argument("graph::build: GraphRunOptions.pull_timeout_ms must be >= 0");
  }

  pipeline_internal::ux::ScopedVerboseContext verbose_ctx(opt.verbose);
  pipeline_internal::ux::ProgressReporter progress(opt.verbose, 3);

  progress.step("Compiling graph...");
  auto plan = simaai::neat::runtime::compile_graph_run_plan(graph, opt);
  progress.detail("segments=" + std::to_string(plan.pipeline_segments.size()) +
                  " stages=" + std::to_string(plan.stage_nodes.size()));

  progress.step("Starting graph runtime...");
  simaai::neat::runtime::RunCoreStartOptions start_opt;
  start_opt.run_options = opt.pipeline;
  start_opt.mode = RunMode::Async;
  start_opt.graph_options = make_graph_runtime_options(opt);
  start_opt.graph_verbose_guard = pipeline_internal::ux::acquire_runtime_verbosity(opt.verbose);

  auto state = std::make_shared<GraphRun::State>();
  state->core = simaai::neat::runtime::RunCore::start(std::move(plan), std::move(start_opt));

  progress.done("Graph ready");
  return GraphRun(std::move(state));
}
} // namespace simaai::neat::graph
