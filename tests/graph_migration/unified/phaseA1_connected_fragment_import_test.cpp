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
    const std::string what = e.what();
    if (what.find(needle) == std::string::npos) {
      throw std::runtime_error(label + ": expected exception containing '" + needle +
                               "', got: " + what);
    }
    return;
  }
  throw std::runtime_error(label + ": expected exception containing '" + needle + "'");
}

simaai::neat::Graph input_graph(const std::string& name) {
  simaai::neat::Graph g(name);
  g.add(simaai::neat::nodes::Input(name));
  return g;
}

simaai::neat::Graph output_graph(const std::string& name) {
  simaai::neat::Graph g(name);
  g.add(simaai::neat::nodes::Output(name));
  return g;
}

simaai::neat::Graph passthrough_graph() {
  simaai::neat::Graph g("route");
  g.add(simaai::neat::nodes::Input("in"));
  g.add(simaai::neat::nodes::Output("out"));
  g.connect("in", "out");
  return g;
}

simaai::neat::Graph named_route_graph() {
  simaai::neat::Graph g("route");
  g.add(simaai::neat::nodes::Input("image"));
  g.add(simaai::neat::nodes::Output("classes"));
  g.connect("image", "classes");
  return g;
}

} // namespace

RUN_TEST("graph_migration_phaseA1_connected_fragment_import_test", [] {
  {
    simaai::neat::Graph source("image");
    source.add(simaai::neat::nodes::Input("raw"));
    auto route = named_route_graph();
    auto sink = output_graph("classes");

    simaai::neat::Graph app;
    app.connect(source, route);
    app.connect(route, sink);

    simaai::neat::Run run = app.build();
    require(
        run.push("raw", simaai::neat::TensorList{graph_phase3_test::make_rgb_tensor(32, 24, 0x51)}),
        "graph-name endpoint inference push failed: " + run.last_error());
    simaai::neat::TensorList tensors = run.pull_tensors("classes", 5000);
    graph_phase3_test::require_nonempty_tensor_output(tensors,
                                                      "graph-name endpoint inference output");
    run.close();
  }

  {
    auto source = input_graph("image");
    auto route = passthrough_graph();
    auto sink = output_graph("classes");

    simaai::neat::Graph app;
    app.connect(source, route);
    app.connect(route, sink);

    simaai::neat::Run run = app.build();
    const bool pushed = run.push(
        "image", simaai::neat::TensorList{graph_phase3_test::make_rgb_tensor(32, 24, 0x52)});
    require(pushed, "connected fragment import push failed: " + run.last_error());
    simaai::neat::TensorList tensors = run.pull_tensors("classes", 5000);
    graph_phase3_test::require_nonempty_tensor_output(tensors,
                                                      "connected fragment import pull_tensors");
    run.close();
  }

  {
    auto source = input_graph("image");
    auto route = passthrough_graph();
    auto sink = output_graph("classes");

    simaai::neat::Graph app;
    app.connect(source, route);
    route.add(simaai::neat::nodes::Output("late"));
    require_throws_contains([&] { app.connect(route, sink); }, "modified after it was imported",
                            "mutated connected fragment should fail version check");
  }

  {
    auto left = input_graph("left");
    auto right = input_graph("right");
    simaai::neat::Graph join("combined");
    join.add(simaai::neat::nodes::Input("left"));
    join.add(simaai::neat::nodes::Input("right"));
    join.add(simaai::neat::nodes::Output("combined"));
    join.connect("left", "combined");
    join.connect("right", "combined");
    auto sink = output_graph("combined");

    simaai::neat::Graph app;
    app.connect(left, join);
    app.connect(right, join);
    app.connect(join, sink);
    require_throws_contains([&] { (void)app.build(); }, "multiple producers",
                            "multi-input fragment should match by graph/input names before "
                            "CombinePolicy lowering");
  }

  {
    simaai::neat::Graph branch("branch");
    branch.add(simaai::neat::nodes::Input("image"));
    branch.add(simaai::neat::nodes::Output("preview"));
    branch.add(simaai::neat::nodes::Output("model"));
    branch.connect("image", "preview");
    branch.connect("image", "model");

    auto sink = output_graph("classes");
    simaai::neat::Graph app;
    require_throws_contains([&] { app.connect(branch, sink); }, "cannot infer endpoints",
                            "ambiguous connected fragment output should fail closed");
  }
});
