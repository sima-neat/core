#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
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
    auto source = input_fragment();
    auto sink = output_fragment();
    simaai::neat::Graph connected;
    connected.connect(source, sink);
    simaai::neat::Graph receiver;
    require_throws_contains([&] { receiver.add(connected); }, "not linear",
                            "linear splice of connected graph should fail closed");
  }
});
