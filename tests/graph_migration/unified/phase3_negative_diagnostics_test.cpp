#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Caps.h"
#include "nodes/common/Output.h"
#include "nodes/io/CameraInput.h"
#include "nodes/io/Input.h"
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
    auto mixed = mixed_live_and_finite_source_fragment();
    auto sink = push_passthrough_fragment("finite", "classes");
    simaai::neat::Graph app;
    app.connect(mixed, sink);
    require_throws_contains([&] { app.connect(mixed, sink); }, "already connected",
                            "finite branch in a mixed live/non-live imported fragment should not "
                            "auto-promote duplicate fan-in");
  }
});
