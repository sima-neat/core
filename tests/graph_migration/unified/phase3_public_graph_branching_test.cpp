#include "graph_migration/common/phase3_graph_test_utils.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

namespace {

simaai::neat::Graph make_input(const std::string& name = {}) {
  simaai::neat::Graph g(name);
  g.add(simaai::neat::nodes::Input());
  return g;
}

simaai::neat::Graph make_named_input_node(const std::string& node_name,
                                          const std::string& graph_name = {}) {
  simaai::neat::Graph g(graph_name);
  g.add(simaai::neat::nodes::Input(node_name));
  return g;
}

simaai::neat::Graph make_output(const std::string& name = {}) {
  simaai::neat::Graph g(name);
  g.add(simaai::neat::nodes::Output());
  return g;
}

simaai::neat::Graph make_named_output_node(const std::string& node_name,
                                           const std::string& graph_name = {}) {
  simaai::neat::Graph g(graph_name);
  g.add(simaai::neat::nodes::Output(node_name));
  return g;
}

} // namespace

RUN_TEST("graph_migration_phase3_public_graph_branching_test", [] {
  {
    auto input = make_input("image");
    auto sink_a = make_output("left");
    auto sink_b = make_output("right");
    simaai::neat::Graph app;
    app.connect(input, sink_a);
    app.connect(input, sink_b);

    simaai::neat::Run run = app.build();
    simaai::neat::Sample ignored;
    simaai::neat::PullError err;
    const simaai::neat::PullStatus st = run.pull(1, ignored, &err);
    require(st == simaai::neat::PullStatus::Error,
            "multi-output public Graph should fail default pull closed");
    require_contains(err.message, "no unambiguous default output",
                     "multi-output diagnostic should name default output ambiguity");

    require(run.push(simaai::neat::Sample{graph_phase3_test::make_tensor_sample(1, "branching")}),
            "multi-output public Graph should accept default push from the single input");
    auto pulled_a = run.pull("left", 5000);
    auto pulled_b = run.pull("right", 5000);
    require(pulled_a.has_value(), "named pull for first public Graph output timed out");
    require(pulled_b.has_value(), "named pull for second public Graph output timed out");
    require(pulled_a->stream_id == "branching", "first branch stream_id mismatch");
    require(pulled_b->stream_id == "branching", "second branch stream_id mismatch");
    run.stop();
  }

  {
    auto input = make_input();
    auto sink = make_output();
    simaai::neat::Graph app;
    app.connect(input, sink);

    simaai::neat::Run run = app.build();
    require(run.push("input",
                     simaai::neat::Sample{graph_phase3_test::make_tensor_sample(1, "auto-names")}),
            "auto-named input push failed");
    auto pulled = run.pull("output", 5000);
    require(pulled.has_value(), "auto-named output pull timed out");
    require(pulled->stream_id == "auto-names", "auto-named branch stream_id mismatch");

    bool unknown_threw = false;
    try {
      (void)run.pull("missing", 1);
    } catch (const std::exception& e) {
      unknown_threw = true;
      require_contains(std::string(e.what()), "Available outputs",
                       "unknown named output diagnostic should list available outputs");
      require_contains(std::string(e.what()), "output",
                       "unknown named output diagnostic should include auto output name");
    }
    require(unknown_threw, "unknown named output should throw");
    run.stop();
  }

  {
    auto input = make_named_input_node("camera", "ignored_graph_input_name");
    auto sink = make_named_output_node("labels", "ignored_graph_output_name");
    simaai::neat::Graph app;
    app.connect(input, sink);

    simaai::neat::Run run = app.build();
    require(run.push("camera", simaai::neat::Sample{graph_phase3_test::make_tensor_sample(
                                   2, "explicit-node")}),
            "explicit Input(name) push failed");
    auto pulled = run.pull("labels", 5000);
    require(pulled.has_value(), "explicit Output(name) pull timed out");
    require(pulled->stream_id == "explicit-node", "explicit node endpoint stream_id mismatch");

    bool graph_base_input_threw = false;
    try {
      (void)run.push("ignored_graph_input_name",
                     simaai::neat::Sample{graph_phase3_test::make_tensor_sample(2, "wrong-name")});
    } catch (const std::exception& e) {
      graph_base_input_threw = true;
      require_contains(std::string(e.what()), "Available inputs",
                       "explicit Input(name) should override graph input base name");
    }
    require(graph_base_input_threw, "graph input base name should not alias explicit Input(name)");
    run.stop();
  }

  {
    auto input_a = make_input("image_a");
    auto input_b = make_input("image_b");
    auto sink_a = make_output("classes_a");
    auto sink_b = make_output("classes_b");
    simaai::neat::Graph app;
    app.connect(input_a, sink_a);
    app.connect(input_b, sink_b);

    simaai::neat::Run run = app.build();
    bool threw = false;
    try {
      (void)run.push(
          simaai::neat::Sample{graph_phase3_test::make_tensor_sample(0, "ambiguous-input")});
    } catch (const std::exception& e) {
      threw = true;
      require_contains(std::string(e.what()), "no unambiguous default input",
                       "multi-input diagnostic should name default input ambiguity");
    }
    require(threw, "multi-input public Graph should fail default push closed");
    bool unknown_input_threw = false;
    try {
      (void)run.push("missing",
                     simaai::neat::Sample{graph_phase3_test::make_tensor_sample(0, "missing")});
    } catch (const std::exception& e) {
      unknown_input_threw = true;
      require_contains(std::string(e.what()), "Available inputs",
                       "unknown named input diagnostic should list available inputs");
      require_contains(std::string(e.what()), "image_a",
                       "unknown named input diagnostic should include first input name");
      require_contains(std::string(e.what()), "image_b",
                       "unknown named input diagnostic should include second input name");
    }
    require(unknown_input_threw, "unknown named input should throw");
    require(
        run.push("image_a", simaai::neat::Sample{graph_phase3_test::make_tensor_sample(0, "a")}),
        "named push to first public Graph input failed");
    require(
        run.push("image_b", simaai::neat::Sample{graph_phase3_test::make_tensor_sample(0, "b")}),
        "named push to second public Graph input failed");
    auto pulled_a = run.pull("classes_a", 5000);
    auto pulled_b = run.pull("classes_b", 5000);
    require(pulled_a.has_value(), "named pull from first output timed out");
    require(pulled_b.has_value(), "named pull from second output timed out");
    require(pulled_a->stream_id == "a", "first named branch stream_id mismatch");
    require(pulled_b->stream_id == "b", "second named branch stream_id mismatch");
    run.stop();
  }

  {
    auto input = make_input("image");
    auto sink_a = make_output("classes");
    auto sink_b = make_output("classes");
    simaai::neat::Graph app;
    app.connect(input, sink_a);
    app.connect(input, sink_b);

    bool threw = false;
    try {
      (void)app.build();
    } catch (const std::exception& e) {
      threw = true;
      require_contains(std::string(e.what()), "used by more than one output",
                       "duplicate named output diagnostic should identify duplicate outputs");
    }
    require(threw, "duplicate user-provided output names should fail closed at build");
  }

  {
    auto input = make_input("image");
    auto sink_a = make_named_output_node("classes");
    auto sink_b = make_named_output_node("classes");
    simaai::neat::Graph app;
    app.connect(input, sink_a);
    app.connect(input, sink_b);

    bool threw = false;
    try {
      (void)app.build();
    } catch (const std::exception& e) {
      threw = true;
      require_contains(std::string(e.what()), "used by more than one output",
                       "duplicate explicit Output(name) diagnostic should identify duplicates");
    }
    require(threw, "duplicate explicit Output(name) values should fail closed at build");
  }
});
