#include "nodes/sima/Preproc.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "test_main.h"

#include <memory>
#include <vector>

namespace {

simaai::neat::PreprocOptions make_binding_preproc_options() {
  simaai::neat::PreprocOptions opt;
  opt.model_managed_contract = true;
  opt.set_input_shape({720, 1280, 3});
  opt.input_img_type = "RGB";
  opt.set_output_shape({640, 640, 3});
  opt.scaled_width = 640;
  opt.scaled_height = 640;
  opt.output_img_type = "RGB";
  opt.output_dtype = "EVXX_INT8";
  opt.tessellate = true;
  opt.single_output_handoff = true;
  opt.set_slice_shape({32, 128, 3});
  opt.q_scale = 0.25;
  opt.q_zp = 7;
  return opt;
}

} // namespace

RUN_TEST("unit_processcvu_binding_table_uses_runtime_outputs_test", ([] {
  using namespace simaai::neat;

  std::vector<std::shared_ptr<Node>> nodes_to_compile = {nodes::Preproc(make_binding_preproc_options())};
  pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled =
      compile_node_contracts(nodes_to_compile, ContractCompileInput{}, &diagnostics);
  require(diagnostics.errors.empty(), "compiled contract should not emit errors");

  const auto manifest_opt =
      render_manifest_from_compiled_contracts(compiled, ContractCompileInput{}, &diagnostics);
  require(manifest_opt.has_value(), "compiled contract should render a manifest");
  require(manifest_opt->stages.size() == 1U, "manifest should contain one stage");
  const auto& stage = manifest_opt->stages.front();

  // The runtime now publishes both the RGB and tessellated outputs; the
  // single_output_handoff selection only narrows the exposed routes below.
  require(stage.processcvu.default_output_names ==
              std::vector<std::string>{"output_rgb_image", "output_tessellated_image"},
          "runtime should expose both runtime outputs");
  require(stage.output_order.size() == 1U,
          "exposed output order should stay projected to the selected handoff");
  require(stage.output_order.front().cm_output_name == "output_tessellated_image",
          "rendered exposed route should match the selected tessellated handoff");
  require(!stage.logical_outputs.empty(),
          "exposed logical outputs should be present");
  // logical_output_index points into the runtime's full logical-output list
  // (rgb + tess). The selected handoff is the second runtime output.
  require(stage.output_order.front().logical_output_index == 1,
          "exposed route should index the tessellated runtime logical output");
  require(!stage.physical_outputs.empty(),
          "runtime should publish at least one physical output");
}));
