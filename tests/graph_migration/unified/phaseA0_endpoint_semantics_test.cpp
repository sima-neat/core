#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
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

} // namespace

RUN_TEST("graph_migration_phaseA0_endpoint_semantics_test", [] {
  {
    simaai::neat::Graph app;
    app.add(simaai::neat::nodes::Input("image"));
    app.add(simaai::neat::nodes::Output("classes"));
    app.connect("image", "classes");

    simaai::neat::Run run = app.build();
    require(run.push("image",
                     simaai::neat::TensorList{graph_phase3_test::make_rgb_tensor(32, 24, 0x31)}),
            "local endpoint TensorList push failed");
    simaai::neat::TensorList tensors = run.pull_tensors("classes", 5000);
    graph_phase3_test::require_nonempty_tensor_output(tensors, "local endpoint pull_tensors");
    run.close();
  }

  {
    simaai::neat::Graph app;
    app.add(simaai::neat::nodes::Input("image"));
    app.add(simaai::neat::nodes::Queue());
    app.add(simaai::neat::nodes::Output("classes"));
    require_throws_contains([&] { app.connect("image", "classes"); }, "non-boundary node",
                            "endpoint promotion should not prune real linear work");
  }

  {
    simaai::neat::Graph app;
    app.add(simaai::neat::nodes::Input("image"));
    app.add(simaai::neat::nodes::Input("image"));
    app.add(simaai::neat::nodes::Output("classes"));
    require_throws_contains([&] { app.connect("image", "classes"); }, "ambiguous",
                            "duplicate endpoint name should fail closed");
  }

  {
    simaai::neat::Graph app;
    app.add(simaai::neat::nodes::Input("left"));
    app.add(simaai::neat::nodes::Input("right"));
    app.add(simaai::neat::nodes::Output("combined"));
    app.connect("left", "combined");
    app.connect("right", "combined");
    require_throws_contains([&] { (void)app.build(); }, "multiple producers",
                            "public endpoint fan-in should fail until CombinePolicy lowering");
  }

  {
    simaai::neat::Graph app;
    app.add(simaai::neat::nodes::Input("image"));
    app.add(simaai::neat::nodes::Output("preview"));
    app.add(simaai::neat::nodes::Output("model"));
    app.connect("image", "preview");
    app.connect("image", "model");
    require_throws_contains([&] { app.add(simaai::neat::nodes::Output("late")); },
                            "Graph::add after endpoint branching is ambiguous",
                            "add after endpoint branching should fail closed");
  }
});
