#include "graphs/Fragments.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "pipeline/ErrorCodes.h"
#include "test_main.h"
#include "test_utils.h"

#include <string>

namespace {

void require_contains_local(const std::string& haystack, const std::string& needle,
                            const std::string& label) {
  if (haystack.find(needle) == std::string::npos) {
    throw std::runtime_error(label + ": expected to find '" + needle + "' in:\n" + haystack);
  }
}

} // namespace

RUN_TEST("graph_migration_phaseA4_connected_introspection_test", [] {
  {
    simaai::neat::Graph input("image");
    input.add(simaai::neat::nodes::Input("image"));
    simaai::neat::Graph output("classes");
    output.add(simaai::neat::nodes::Output("classes"));

    simaai::neat::Graph app("app");
    app.connect(input, output);

    const std::string public_description = app.describe();
    require_contains_local(public_description, "Graph \"app\"", "connected describe graph name");
    require_contains_local(public_description, "input=\"image\"", "connected describe input");
    require_contains_local(public_description, "output=\"classes\"", "connected describe output");
    require_contains_local(public_description, "endpoint image -> classes",
                           "connected describe endpoint edge");

    const std::string backend = app.describe_backend(false);
    require_contains_local(backend, "ExecutionGraphPlan", "connected describe_backend plan");
    require_contains_local(backend, "mode: connected", "connected describe_backend mode");
    require_contains_local(backend, "named_inputs:", "connected describe_backend named inputs");
    require_contains_local(backend, "image", "connected describe_backend image name");
    require_contains_local(backend, "named_outputs:", "connected describe_backend named outputs");
    require_contains_local(backend, "classes", "connected describe_backend classes name");

    const simaai::neat::GraphReport report = app.validate();
    require(report.error_code.empty(),
            "connected validate should pass, got " + report.error_code + ": " + report.repro_note);
    require_contains_local(report.pipeline_string, "ExecutionGraphPlan",
                           "connected validate report plan");
  }

  {
    simaai::neat::Graph join;
    join.add(simaai::neat::nodes::Input("left"));
    join.add(simaai::neat::nodes::Input("right"));
    join.add(simaai::neat::nodes::Output("combined"));
    join.connect("left", "combined");
    join.connect("right", "combined");

    const simaai::neat::GraphReport report = join.validate();
    require(report.error_code == simaai::neat::error_codes::kPipelineShape,
            "connected validate should fail with pipeline.shape for missing CombinePolicy");
    require_contains_local(report.repro_note, "CombinePolicy",
                           "connected validate CombinePolicy diagnostic");
  }

  {
    simaai::neat::Graph join = simaai::neat::graphs::Combine({"left", "right"}, "combined",
                                                             simaai::neat::CombinePolicy::ByFrame);
    const std::string public_description = join.describe();
    require_contains_local(public_description, "combine=ByFrame",
                           "connected describe should expose CombinePolicy");
    const simaai::neat::GraphReport report = join.validate();
    require(report.error_code.empty(), "Combine ByFrame validate should pass, got " +
                                           report.error_code + ": " + report.repro_note);
  }
});
