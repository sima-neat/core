#include "graph_migration/common/phase3_graph_test_utils.h"
#include "model/Model.h"
#include "nodes/common/Output.h"
#include "nodes/common/Queue.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"

#include <memory>
#include <type_traits>
#include <utility>

namespace {

void require_roundtrip(simaai::neat::Graph& app, const std::string& input,
                       const std::string& output, const std::string& label) {
  simaai::neat::Run run = app.build();
  require(
      run.push(input, simaai::neat::TensorList{graph_phase3_test::make_rgb_tensor(32, 24, 0x61)}),
      label + ": push failed: " + run.last_error());
  simaai::neat::TensorList tensors = run.pull_tensors(output, 5000);
  graph_phase3_test::require_nonempty_tensor_output(tensors, label + ": pull_tensors");
  run.close();
}

} // namespace

RUN_TEST("graph_migration_phaseA2_connect_overload_facade_test", [] {
  static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().connect(
                                   std::declval<const simaai::neat::Model&>(),
                                   std::declval<const simaai::neat::Graph&>())),
                               simaai::neat::Graph&>);
  static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().connect(
                                   std::declval<const simaai::neat::Graph&>(),
                                   std::declval<const simaai::neat::Model&>())),
                               simaai::neat::Graph&>);
  static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().connect(
                                   std::declval<const simaai::neat::Model&>(),
                                   std::declval<const simaai::neat::Model&>())),
                               simaai::neat::Graph&>);
  static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().connect(
                                   std::declval<const simaai::neat::Model&>(),
                                   std::declval<std::shared_ptr<simaai::neat::Node>>())),
                               simaai::neat::Graph&>);
  static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().connect(
                                   std::declval<std::shared_ptr<simaai::neat::Node>>(),
                                   std::declval<const simaai::neat::Model&>())),
                               simaai::neat::Graph&>);
  static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().connect(
                                   std::declval<const simaai::neat::Graph&>(),
                                   std::declval<std::shared_ptr<simaai::neat::Node>>())),
                               simaai::neat::Graph&>);
  static_assert(std::is_same_v<decltype(std::declval<simaai::neat::Graph&>().connect(
                                   std::declval<std::shared_ptr<simaai::neat::Node>>(),
                                   std::declval<const simaai::neat::Graph&>())),
                               simaai::neat::Graph&>);

  {
    auto input = simaai::neat::nodes::Input("image");
    auto queue = simaai::neat::nodes::Queue();
    auto output = simaai::neat::nodes::Output("classes");

    simaai::neat::Graph app;
    app.connect(input, queue);
    app.connect(queue, output);
    require_roundtrip(app, "image", "classes", "Node->Node reuse");
  }

  {
    simaai::neat::Graph source("image");
    source.add(simaai::neat::nodes::Input("image"));
    auto queue = simaai::neat::nodes::Queue();
    auto output = simaai::neat::nodes::Output("classes");

    simaai::neat::Graph app;
    app.connect(source, queue);
    app.connect(queue, output);
    require_roundtrip(app, "image", "classes", "Graph->Node->Node");
  }

  {
    auto input = simaai::neat::nodes::Input("image");
    auto queue = simaai::neat::nodes::Queue();
    simaai::neat::Graph sink("classes");
    sink.add(simaai::neat::nodes::Output("classes"));

    simaai::neat::Graph app;
    app.connect(input, queue);
    app.connect(queue, sink);
    require_roundtrip(app, "image", "classes", "Node->Node->Graph");
  }
});
