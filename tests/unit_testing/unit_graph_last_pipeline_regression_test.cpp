#ifndef SIMA_NEAT_INTERNAL
#define SIMA_NEAT_INTERNAL 1
#endif

#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/io/Input.h"
#include "graphs/Fragments.h"
#include "pipeline/Graph.h"
#include "pipeline/NeatError.h"
#include "pipeline/runtime/RunCore.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using simaai::neat::Graph;
using simaai::neat::ImageSpec;
using simaai::neat::Run;
using simaai::neat::RunOptions;
using simaai::neat::Sample;
using simaai::neat::Tensor;
using simaai::neat::TensorList;
namespace nodes = simaai::neat::nodes;

Tensor seed_tensor() {
  return make_color_tensor(64, 48, ImageSpec::PixelFormat::RGB, 0x21);
}

Graph named_graph() {
  Graph g("last_pipeline_named");
  g.add(nodes::Input("in"));
  g.add(nodes::Queue());
  g.add(nodes::Output("out"));
  return g;
}

Graph unnamed_graph() {
  Graph g("last_pipeline_unnamed");
  g.add(nodes::Input());
  g.add(nodes::Queue());
  g.add(nodes::Output());
  return g;
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

void require_seed_launch_string(const std::string& s, const std::string& where) {
  require(!s.empty(), where + ": last_pipeline() is empty after a successful build");
  require_contains(s, "appsrc", where);
  require_contains(s, "appsink", where);
  require_contains(s, "format=RGB", where);
  require_contains(s, "width=64", where);
  require_contains(s, "height=48", where);
}

void push_pull_one(Run& run, const std::string& in, const std::string& out,
                   const std::string& where) {
  const bool named = !in.empty();
  const TensorList frame{seed_tensor()};
  require(named ? run.push(in, frame) : run.push(frame), where + ": push failed");
  auto got = named ? run.pull(out, 5000) : run.pull(5000);
  require(got.has_value(), where + ": pull timed out: " + run.last_error());
}

// A graph that never built reports an empty string.
void empty_before_any_build() {
  Graph named = named_graph();
  require(named.last_pipeline().empty(), "named graph before build should be empty");
  Graph unnamed = unnamed_graph();
  require(unnamed.last_pipeline().empty(), "unnamed graph before build should be empty");
}

// A seeded build of a named-endpoint graph publishes the string its pipeline was launched with.
void seeded_build_publishes_launch_string() {
  Graph g = named_graph();
  Run run = g.build(TensorList{seed_tensor()});
  const std::string seg = only_segment_string(run, "named seeded build");
  require(!seg.empty(), "named seeded build: segment string is empty (test premise)");
  require_seed_launch_string(g.last_pipeline(), "named seeded build");
  require(g.last_pipeline() == seg, "named seeded build: last_pipeline() differs from the segment "
                                    "string\n graph:   " +
                                        g.last_pipeline() + "\n segment: " + seg);
}

void seeded_sample_build_publishes_launch_string() {
  Graph g = named_graph();
  Run run = g.build(simaai::neat::sample_from_tensors(TensorList{seed_tensor()}));
  require_seed_launch_string(g.last_pipeline(), "named seeded Sample build");
  require(g.last_pipeline() == only_segment_string(run, "named seeded Sample build"),
          "named seeded Sample build: last_pipeline() differs from the segment string");
}

// The value stays put while the Run is used, after close, and after the Run is destroyed.
void launch_string_persists_across_run_activity() {
  Graph g = named_graph();
  std::string published;
  {
    Run run = g.build(TensorList{seed_tensor()});
    published = g.last_pipeline();
    require_seed_launch_string(published, "persist: after build");
    push_pull_one(run, "in", "out", "persist");
    require(g.last_pipeline() == published, "persist: value changed after push/pull");
    run.close();
    require(g.last_pipeline() == published, "persist: value changed after close");
  }
  require(g.last_pipeline() == published, "persist: value changed after the Run was destroyed");
}

// A one-shot run on a named graph builds a runner, and the string that runner was launched with
// is published whether or not the call itself goes on to return an output.
void run_publishes_launch_string() {
  Graph g = named_graph();
  try {
    (void)g.run(TensorList{seed_tensor()});
  } catch (const std::exception& e) {
    std::cerr << "note: Graph::run threw after building its runner: " << e.what() << "\n";
  }
  require_seed_launch_string(g.last_pipeline(), "run: after first run");
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
  require(!g.last_pipeline().empty(),
          "named source build: last_pipeline() is empty after a successful build");
  require_contains(g.last_pipeline(), "videotestsrc", "named source build: last_pipeline()");
  require(g.last_pipeline() == seg, "named source build: last_pipeline() differs from the segment "
                                    "string\n graph:   " +
                                        g.last_pipeline() + "\n segment: " + seg);
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

// A seedless build launches no pipeline, so a string from an earlier build must not survive,
// and the lazy segment build on the first push must not fill it in.
void seedless_named_build_clears_stale_value() {
  Graph g;
  g.add(nodes::Input("image"));
  g.add(nodes::Queue());
  g.add(nodes::Output("out"));

  {
    Run seeded = g.build(TensorList{seed_tensor()});
    require_seed_launch_string(g.last_pipeline(), "seedless: earlier seeded build");
    seeded.close();
  }

  Run run = g.build(RunOptions{});
  require(g.last_pipeline().empty(),
          "seedless: last_pipeline() kept a stale string: " + g.last_pipeline());
  push_pull_one(run, "image", "out", "seedless");
  require(g.last_pipeline().empty(),
          "seedless: a later push must not fill last_pipeline(): " + g.last_pipeline());
}

simaai::neat::runtime::ExecutionGraphPlan
named_plan(simaai::neat::runtime::RunCoreStartOptions& so) {
  using namespace simaai::neat;
  Graph graph;
  graph.add(nodes::Input("image"));
  graph.add(nodes::Output("output", OutputOptions::EveryFrame(4)));

  const Sample seed = sample_from_tensors(TensorList{seed_tensor()});
  RunOptions run_options;
  run_options.queue_depth = 4;
  auto plan = runtime::compile_public_graph(graph, run_options, seed);
  so.run_options = run_options;
  so.mode = RunMode::Async;
  so.seed = seed;
  so.graph_options = runtime::graph_runtime_options_from_run_options(run_options);
  return plan;
}

// A start that throws leaves the caller's string untouched.
void failed_start_leaves_value_unchanged() {
  using namespace simaai::neat;
  const std::string sentinel = "previous launch string";
  runtime::RunCoreStartOptions so;
  auto plan = named_plan(so);
  std::string probe = sentinel;
  so.last_pipeline = &probe;
  so.after_pipeline_start_for_test = [] { throw std::runtime_error("injected start failure"); };
  bool threw = false;
  try {
    (void)runtime::RunCore::start(std::move(plan), std::move(so));
  } catch (const NeatError&) {
    threw = true;
  }
  require(threw, "failed start: injected failure did not throw");
  require(probe == sentinel, "failed start: last_pipeline changed to: " + probe);
}

// A start that succeeds hands the segment's launch string to the caller.
void successful_start_publishes_value() {
  using namespace simaai::neat;
  runtime::RunCoreStartOptions so;
  auto plan = named_plan(so);
  std::string probe = "previous launch string";
  so.last_pipeline = &probe;
  auto core = runtime::RunCore::start(std::move(plan), std::move(so));
  require(core != nullptr, "successful start: no RunCore");
  const auto& exec = core->graph_execution();
  require(exec.pipelines.size() == 1U, "successful start: expected a single segment");
  const std::string seg = exec.pipelines.front()->last_pipeline;
  require(!seg.empty(), "successful start: segment string is empty (test premise)");
  require(probe == seg, "successful start: last_pipeline was not published; got: " + probe);
  core->close();
}

// Unnamed endpoints take the simple path, which already publishes its string.
void unnamed_graph_publishes_launch_string() {
  Graph g = unnamed_graph();
  std::string published;
  {
    Run run = g.build(TensorList{seed_tensor()});
    require(run.input_names().empty(), "unnamed: expected the simple path (no named inputs)");
    published = g.last_pipeline();
    require_seed_launch_string(published, "unnamed: after build");
    push_pull_one(run, "", "", "unnamed");
    require(g.last_pipeline() == published, "unnamed: value changed after push/pull");
    run.close();
    require(g.last_pipeline() == published, "unnamed: value changed after close");
  }
  require(g.last_pipeline() == published, "unnamed: value changed after the Run was destroyed");
}

// The Graph may be moved or destroyed while its Run is alive; the Run never writes back to it.
void graph_can_die_before_run() {
  auto graph = std::make_unique<Graph>(named_graph());
  Run run = graph->build(TensorList{seed_tensor()});
  auto moved = std::make_unique<Graph>(std::move(*graph));
  graph.reset();
  const std::string published = moved->last_pipeline();
  require_seed_launch_string(published, "moved graph");
  push_pull_one(run, "in", "out", "moved graph");
  moved.reset();
  push_pull_one(run, "in", "out", "destroyed graph");
  run.close();
}

} // namespace

// Runs every case even when an earlier one fails, so each broken behaviour is reported.
RUN_TEST("unit_graph_last_pipeline_regression_test", ([] {
           struct Case {
             const char* name;
             void (*fn)();
           };
           const Case cases[] = {
               {"empty_before_any_build", empty_before_any_build},
               {"unnamed_graph_publishes_launch_string", unnamed_graph_publishes_launch_string},
               {"failed_start_leaves_value_unchanged", failed_start_leaves_value_unchanged},
               {"successful_start_publishes_value", successful_start_publishes_value},
               {"seeded_build_publishes_launch_string", seeded_build_publishes_launch_string},
               {"seeded_sample_build_publishes_launch_string",
                seeded_sample_build_publishes_launch_string},
               {"launch_string_persists_across_run_activity",
                launch_string_persists_across_run_activity},
               {"run_publishes_launch_string", run_publishes_launch_string},
               {"source_build_publishes_launch_string", source_build_publishes_launch_string},
               {"multi_segment_build_reports_empty", multi_segment_build_reports_empty},
               {"seedless_named_build_clears_stale_value", seedless_named_build_clears_stale_value},
               {"graph_can_die_before_run", graph_can_die_before_run},
           };
           std::string failures;
           for (const auto& c : cases) {
             try {
               c.fn();
               std::cerr << "[CASE PASS] " << c.name << "\n";
             } catch (const std::exception& e) {
               std::cerr << "[CASE FAIL] " << c.name << ": " << e.what() << "\n";
               failures += std::string(c.name) + "; ";
             }
           }
           require(failures.empty(), "failing cases: " + failures);
         }));
