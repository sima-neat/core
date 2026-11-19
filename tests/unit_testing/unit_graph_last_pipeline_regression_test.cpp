#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/io/Input.h"
#include "graphs/Fragments.h"
#include "pipeline/Graph.h"
#include "pipeline/runtime/RunCore.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

namespace {

using simaai::neat::Graph;
using simaai::neat::ImageSpec;
using simaai::neat::Run;
using simaai::neat::RunOptions;
using simaai::neat::Tensor;
using simaai::neat::TensorList;
namespace nodes = simaai::neat::nodes;

Tensor seed_tensor() {
  return make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x21);
}

// The launch string the Run's only pipeline segment was built with.
const std::string& only_segment_string(const Run& run, const std::string& where) {
  const auto core = simaai::neat::run_internal::core(run);
  const auto& exec = core->graph_execution();
  require(exec.pipelines.size() == 1U, where + ": expected a single pipeline segment, got " +
                                           std::to_string(exec.pipelines.size()));
  require(exec.pipelines.front() != nullptr, where + ": missing segment runtime");
  return exec.pipelines.front()->last_pipeline;
}

// A source graph with a named Output publishes its source segment's string.
void source_build_publishes_launch_string() {
  Graph g;
  g.custom("videotestsrc is-live=true num-buffers=3 pattern=black",
           simaai::neat::InputRole::Source);
  g.add(nodes::CapsRaw("RGB", 64, 48, 30, simaai::neat::CapsMemory::SystemMemory));
  g.add(nodes::Output("out", simaai::neat::OutputOptions::EveryFrame(4)));
  Run run = g.build();
  const std::string seg = only_segment_string(run, "named source build");
  require(!seg.empty(), "named source build: segment string is empty (test premise)");
  require_contains(g.last_pipeline(), "videotestsrc", "named source build: last_pipeline()");
  require(g.last_pipeline() == seg, "named source build: last_pipeline() differs from the segment "
                                    "string\n graph:   " +
                                        g.last_pipeline() + "\n segment: " + seg);
}

// A seedless build defers pipeline creation, so a string from an earlier build must not survive,
// and the lazy segment build on the first push must not fill it in.
void seedless_named_build_clears_stale_value() {
  Graph g;
  g.add(nodes::Input("image"));
  g.add(nodes::Queue());
  g.add(nodes::Output("out"));

  {
    Run seeded = g.build(TensorList{seed_tensor()});
    require_contains(g.last_pipeline(), "appsrc", "seedless: earlier seeded build");
    seeded.close();
  }

  Run run = g.build(RunOptions{});
  require(g.last_pipeline().empty(),
          "seedless: last_pipeline() kept a stale string: " + g.last_pipeline());
  require(run.push("image", TensorList{seed_tensor()}), "seedless: push failed");
  require(run.pull("out", 5000).has_value(), "seedless: pull timed out: " + run.last_error());
  require(g.last_pipeline().empty(),
          "seedless: a later push must not fill last_pipeline(): " + g.last_pipeline());
}

// Several pipeline segments have no single launch string, so the value is empty.
void multi_segment_build_reports_empty() {
  Graph source("image");
  source.add(nodes::Input("image"));
  source.add(nodes::Queue());
  Graph branch = simaai::neat::graphs::Branch("image", {"leg_a", "leg_b"});
  Graph leg_a("leg_a");
  leg_a.add(nodes::Input("leg_a"));
  leg_a.add(nodes::Queue());
  leg_a.add(nodes::Output("out_a"));
  Graph leg_b("leg_b");
  leg_b.add(nodes::Input("leg_b"));
  leg_b.add(nodes::Queue());
  leg_b.add(nodes::Output("out_b"));
  Graph app("last_pipeline_fanout");
  app.connect(source, branch);
  app.connect(branch, leg_a);
  app.connect(branch, leg_b);

  Run run = app.build(RunOptions{});
  const auto core = simaai::neat::run_internal::core(run);
  require(core->graph_execution().pipelines.size() > 1U,
          "multi-segment build: expected several pipeline segments (test premise)");
  require(app.last_pipeline().empty(),
          "multi-segment build: last_pipeline() should be empty, got: " + app.last_pipeline());
}

} // namespace

RUN_TEST("unit_graph_last_pipeline_regression_test", ([] {
           source_build_publishes_launch_string();
           seedless_named_build_clears_stale_value();
           multi_segment_build_reports_empty();
         }));
