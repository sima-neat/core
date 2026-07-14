#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/io/CameraInput.h"
#include "nodes/io/Input.h"
#include "pipeline/runtime/ExecutionGraphPlan.h"
#include "test_main.h"
#include "test_utils.h"

#include <functional>
#include <string>

namespace {

void require_throws_contains(const std::function<void()>& fn, const std::string& needle,
                             const std::string& label) {
  try {
    fn();
  } catch (const std::exception& e) {
    require_contains(std::string(e.what()), needle, label);
    return;
  }
  throw std::runtime_error(label + ": expected exception containing '" + needle + "'");
}

simaai::neat::Graph input_fragment() {
  simaai::neat::Graph g;
  g.add(simaai::neat::nodes::Input());
  return g;
}

simaai::neat::Graph output_fragment() {
  simaai::neat::Graph g;
  g.add(simaai::neat::nodes::Output());
  return g;
}

simaai::neat::Graph push_passthrough_fragment(const std::string& input_name,
                                              const std::string& output_name) {
  simaai::neat::Graph g;
  g.add(simaai::neat::nodes::Input(input_name));
  g.add(simaai::neat::nodes::Output(output_name));
  g.connect(input_name, output_name);
  return g;
}

simaai::neat::Graph linked_passthrough_fragment(const std::string& input_name,
                                                const std::string& output_name,
                                                const simaai::neat::RealtimeMuxByStream& link) {
  simaai::neat::Graph in;
  in.add(simaai::neat::nodes::Input(input_name));

  simaai::neat::Graph out;
  out.add(simaai::neat::nodes::Output(output_name));

  simaai::neat::Graph g;
  g.connect(in, out, link);
  return g;
}

simaai::neat::Graph live_camera_source_fragment(const std::string& buffer_name) {
  simaai::neat::CameraInputOptions opt;
  opt.buffer_name = buffer_name;
  simaai::neat::Graph g;
  g.add(simaai::neat::nodes::CameraInput(opt));
  return g;
}

simaai::neat::Graph mixed_live_and_finite_source_fragment() {
  simaai::neat::CameraInputOptions live_opt;
  live_opt.buffer_name = "mixed_live_camera";

  simaai::neat::Graph g;
  g.connect(simaai::neat::nodes::CameraInput(live_opt), simaai::neat::nodes::Output("live"));
  g.connect(simaai::neat::nodes::Custom("videotestsrc num-buffers=1 pattern=black",
                                        simaai::neat::InputRole::Source),
            simaai::neat::nodes::Output("finite"));
  return g;
}

} // namespace

RUN_TEST("graph_migration_phase3_negative_diagnostics_test", [] {
  {
    simaai::neat::runtime::ExecutionGraphPlan plan;
    simaai::neat::runtime::PipelineSegmentPlan segment;
    segment.id = 7;
    segment.input_edges = {0};
    simaai::neat::runtime::FragmentBoundaryHints hints;
    simaai::neat::InputOptions options;
    options.max_width = 1920;
    options.max_height = 1080;
    options.max_depth = 3;
    hints.ingress_inputs.push_back(options);
    segment.boundary_hints = std::move(hints);
    plan.pipeline_segments.push_back(std::move(segment));

    simaai::neat::runtime::EdgePlan edge;
    edge.spec.width = 2048;
    edge.spec.height = 1080;
    edge.spec.depth = 3;
    edge.spec_complete = true;
    plan.edges.push_back(std::move(edge));

    require_throws_contains(
        [&] { simaai::neat::runtime::validate_static_connected_input_capacities(plan); },
        "input width 2048 exceeds configured capacity 1920",
        "known connected source shape should fail at compile preflight");
    require_throws_contains(
        [&] { simaai::neat::runtime::validate_static_connected_input_capacities(plan); },
        "Model::Options::preprocess.input_max_width",
        "connected shape preflight should provide the public model fix");
  }

  {
    simaai::neat::runtime::ExecutionGraphPlan plan;
    plan.port_names.push_back("small");
    plan.port_names.push_back("large");
    simaai::neat::runtime::PipelineSegmentPlan segment;
    segment.id = 8;
    segment.input_edges = {0};
    simaai::neat::runtime::FragmentBoundaryHints hints;
    hints.ingress_endpoint_names.push_back("small");
    hints.ingress_endpoint_names.push_back("large");
    simaai::neat::InputOptions small;
    small.max_width = 640;
    small.max_height = 480;
    small.max_depth = 3;
    simaai::neat::InputOptions large = small;
    large.max_width = 1920;
    large.max_height = 1080;
    hints.ingress_inputs.push_back(small);
    hints.ingress_inputs.push_back(large);
    segment.boundary_hints = std::move(hints);
    plan.pipeline_segments.push_back(std::move(segment));

    simaai::neat::runtime::EdgePlan edge;
    edge.to_port = 1;
    edge.spec.width = 1280;
    edge.spec.height = 720;
    edge.spec.depth = 3;
    edge.spec_complete = true;
    plan.edges.push_back(std::move(edge));

    simaai::neat::runtime::validate_static_connected_input_capacities(plan);
  }

  {
    simaai::neat::runtime::ExecutionGraphPlan plan;
    plan.port_names.push_back("large");
    plan.port_names.push_back("small");
    simaai::neat::runtime::PipelineSegmentPlan segment;
    segment.id = 9;
    segment.input_edges = {0};
    simaai::neat::runtime::FragmentBoundaryHints hints;
    hints.ingress_endpoint_names.push_back("large");
    hints.ingress_endpoint_names.push_back("small");
    simaai::neat::InputOptions large;
    large.max_width = 1920;
    large.max_height = 1080;
    large.max_depth = 3;
    simaai::neat::InputOptions small = large;
    small.max_width = 640;
    small.max_height = 480;
    hints.ingress_inputs.push_back(large);
    hints.ingress_inputs.push_back(small);
    segment.boundary_hints = std::move(hints);
    plan.pipeline_segments.push_back(std::move(segment));

    simaai::neat::runtime::EdgePlan edge;
    edge.to_port = 1;
    edge.spec.width = 800;
    edge.spec.height = 480;
    edge.spec.depth = 3;
    edge.spec_complete = true;
    plan.edges.push_back(std::move(edge));

    require_throws_contains(
        [&] { simaai::neat::runtime::validate_static_connected_input_capacities(plan); },
        "input width 800 exceeds configured capacity 640",
        "each connected source shape should use its matching ingress capacity");
  }

  {
    simaai::neat::Graph empty;
    auto sink = output_fragment();
    simaai::neat::Graph app;
    require_throws_contains([&] { app.connect(empty, sink); }, "empty Graph fragments",
                            "empty source fragment should fail closed");
  }

  {
    auto source = input_fragment();
    simaai::neat::Graph empty;
    simaai::neat::Graph app;
    require_throws_contains([&] { app.connect(source, empty); }, "empty Graph fragments",
                            "empty destination fragment should fail closed");
  }

  {
    simaai::neat::Graph app;
    app.add(simaai::neat::nodes::Input("image"));
    app.add(simaai::neat::nodes::Output("classes"));
    require_throws_contains([&] { app.connect("", "classes"); }, "endpoint name must not be empty",
                            "empty source endpoint name should fail closed");
  }

  {
    simaai::neat::Graph app;
    app.add(simaai::neat::nodes::Input("image"));
    app.add(simaai::neat::nodes::Output("classes"));
    require_throws_contains([&] { app.connect("image", ""); }, "endpoint name must not be empty",
                            "empty destination endpoint name should fail closed");
  }

  {
    auto source = input_fragment();
    auto sink_a = output_fragment();
    auto sink_b = output_fragment();
    simaai::neat::Graph app;
    app.connect(source, sink_a);
    app.connect(source, sink_b);
    auto tail = output_fragment();
    require_throws_contains([&] { app.add(tail); }, "Graph::add after branching is ambiguous",
                            "Graph::add connected fragment should fail closed");
  }

  {
    auto left = push_passthrough_fragment("left_in", "left_out");
    auto right = push_passthrough_fragment("right_in", "right_out");
    auto sink = push_passthrough_fragment("image", "classes");
    simaai::neat::Graph app;
    app.connect(left, sink);
    require_throws_contains([&] { app.connect(right, sink); }, "already connected",
                            "default duplicate app-pushed fan-in should stay explicit");
  }

  {
    auto left = live_camera_source_fragment("left_camera");
    auto right = live_camera_source_fragment("right_camera");
    auto sink = push_passthrough_fragment("image", "classes");
    simaai::neat::Graph app;
    app.connect(left, sink);
    app.connect(right, sink);
  }

  {
    simaai::neat::RealtimeMuxByStream valid_realtime;
    valid_realtime.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    valid_realtime.max_inflight_per_stream = 4;
    valid_realtime.max_inflight_total = 16;

    simaai::neat::RealtimeMuxByStream invalid_realtime = valid_realtime;
    invalid_realtime.max_inflight_per_stream = 0;

    auto left = live_camera_source_fragment("valid_realtime_camera");
    auto right = live_camera_source_fragment("invalid_realtime_camera");
    auto sink = push_passthrough_fragment("image", "classes");
    simaai::neat::Graph app;
    app.connect(left, sink, valid_realtime);
    require_throws_contains([&] { app.connect(right, sink, invalid_realtime); },
                            "max_inflight_per_stream",
                            "fan-in merge should reject invalid per-stream inflight cap");
  }

  {
    simaai::neat::RealtimeMuxByStream valid_realtime;
    valid_realtime.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    valid_realtime.max_inflight_per_stream = 4;
    valid_realtime.max_inflight_total = 16;

    simaai::neat::RealtimeMuxByStream invalid_realtime = valid_realtime;
    invalid_realtime.max_inflight_total = -2;

    auto left = live_camera_source_fragment("valid_realtime_total_camera");
    auto right = live_camera_source_fragment("invalid_realtime_total_camera");
    auto sink = push_passthrough_fragment("image", "classes");
    simaai::neat::Graph app;
    app.connect(left, sink, valid_realtime);
    require_throws_contains([&] { app.connect(right, sink, invalid_realtime); },
                            "max_inflight_total",
                            "fan-in merge should reject invalid total inflight cap");
  }

  {
    simaai::neat::RealtimeMuxByStream valid_inner;
    valid_inner.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    valid_inner.max_inflight_per_stream = 4;

    simaai::neat::RealtimeMuxByStream invalid_outer = valid_inner;
    invalid_outer.max_inflight_per_stream = 0;

    auto source = live_camera_source_fragment("compiler_merge_invalid_realtime_camera");
    auto sink = linked_passthrough_fragment("image", "classes", valid_inner);
    simaai::neat::Graph app;
    app.connect(source, sink, invalid_outer);
    require_throws_contains(
        [&] { (void)simaai::neat::runtime::compile_public_graph(app, simaai::neat::RunOptions{}); },
        "max_inflight_per_stream",
        "compiler boundary merge should reject invalid per-stream inflight cap");
  }

  {
    simaai::neat::RealtimeMuxByStream valid_inner;
    valid_inner.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    valid_inner.max_inflight_total = 16;

    simaai::neat::RealtimeMuxByStream invalid_outer = valid_inner;
    invalid_outer.max_inflight_total = -2;

    auto source = live_camera_source_fragment("compiler_merge_invalid_total_camera");
    auto sink = linked_passthrough_fragment("image", "classes", valid_inner);
    simaai::neat::Graph app;
    app.connect(source, sink, invalid_outer);
    require_throws_contains(
        [&] { (void)simaai::neat::runtime::compile_public_graph(app, simaai::neat::RunOptions{}); },
        "max_inflight_total", "compiler boundary merge should reject invalid total inflight cap");
  }

  {
    simaai::neat::RealtimeMuxByStream inner;
    inner.policy = simaai::neat::GraphLinkPolicy::RealtimeLatestByStream;
    inner.max_inflight_per_stream = 16;
    inner.max_inflight_total = 16;

    simaai::neat::RealtimeMuxByStream outer = inner;
    outer.max_inflight_per_stream = 4;
    outer.max_inflight_total = 4;

    auto source = live_camera_source_fragment("compiler_merge_outer_override_camera");
    auto sink = linked_passthrough_fragment("image", "classes", inner);
    simaai::neat::Graph app;
    app.connect(source, sink, outer);

    const auto plan = simaai::neat::runtime::compile_public_graph(app, simaai::neat::RunOptions{});
    bool saw_realtime_override = false;
    for (const auto& edge : plan.edges) {
      if (edge.link_options.policy != simaai::neat::GraphLinkPolicy::RealtimeLatestByStream) {
        continue;
      }
      require(edge.link_options.max_inflight_per_stream == 4,
              "compiler boundary merge should let outer per-stream cap override inner cap");
      require(edge.link_options.max_inflight_total == 4,
              "compiler boundary merge should let outer total cap override inner cap");
      saw_realtime_override = true;
    }
    require(saw_realtime_override, "compiler boundary merge test should produce a realtime edge");
  }

  {
    auto left = live_camera_source_fragment("camera");
    auto right = live_camera_source_fragment("camera");
    auto sink = push_passthrough_fragment("image", "classes");
    simaai::neat::Graph app;
    app.connect(left, sink);
    app.connect(right, sink);
    const auto report = app.validate();
    require(!report.error_code.empty(),
            "duplicate CameraInput buffer_name should fail connected graph validation");
    require_contains(report.repro_note, "duplicate source buffer_name",
                     "duplicate CameraInput buffer_name diagnostic");
  }

  {
    simaai::neat::Graph app;
    app.connect(simaai::neat::nodes::CameraInput(), simaai::neat::nodes::Output("left_preview"));
    app.connect(simaai::neat::nodes::CameraInput(), simaai::neat::nodes::Output("right_preview"));
    const auto report = app.validate();
    require(report.error_code.empty(),
            "independent CameraInput paths may share the public default buffer_name");
  }

  {
    auto mixed = mixed_live_and_finite_source_fragment();
    auto sink = push_passthrough_fragment("finite", "classes");
    simaai::neat::Graph app;
    app.connect(mixed, sink);
    require_throws_contains([&] { app.connect(mixed, sink); }, "already connected",
                            "finite branch in a mixed live/non-live imported fragment should not "
                            "auto-promote duplicate fan-in");
  }
});
