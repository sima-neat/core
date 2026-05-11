#include "builder/NodeGroup.h"
#include "nodes/sima/Preproc.h"
#include "pipeline/internal/RenderedMlaContractQuery.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "test_main.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace {

namespace rendered_stage_query = simaai::neat::pipeline_internal::rendered_stage_query;

simaai::neat::PreprocOptions make_equivalence_preproc_options() {
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

RUN_TEST("session_stage_contract_equivalence_test", ([] {
  using namespace simaai::neat;

  auto node = nodes::Preproc(make_equivalence_preproc_options());
  NodeGroup group(std::vector<std::shared_ptr<Node>>{node});

  pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled =
      compile_node_contracts(group.nodes(), ContractCompileInput{}, &diagnostics);
  require(diagnostics.errors.empty(), "compiled contracts should not emit errors");

  const auto manifest_opt =
      render_manifest_from_compiled_contracts(compiled, ContractCompileInput{}, &diagnostics);
  require(manifest_opt.has_value(), "compiled contracts should render a manifest");
  require(manifest_opt->stages.size() == 1U, "manifest should contain one stage");

  const auto rendered_info = rendered_stage_query::preproc_output_info(group);
  const auto& stage = manifest_opt->stages.front();
  require(rendered_info.primary_output_name == stage.processcvu.primary_output_name,
          "stage helper and rendered manifest should agree on primary output");
  require(rendered_info.primary_route_slot == 0,
          "stage helper should keep a dense exposed route slot");
  require(rendered_info.output_memory_order == stage.processcvu.default_output_names,
          "stage helper and rendered manifest should agree on runtime output order");
  const auto routed_output = std::find_if(
      stage.logical_outputs.begin(), stage.logical_outputs.end(), [&](const auto& logical_output) {
        return logical_output.logical_index == stage.output_order.front().logical_output_index;
      });
  require(routed_output != stage.logical_outputs.end() &&
              routed_output->logical_name == rendered_info.primary_output_name,
          "rendered manifest should preserve routed identity for the projected exposed output");
  require(stage.logical_outputs.front().shape == std::vector<std::int64_t>({640, 640, 3}) &&
              stage.logical_outputs.front().size_bytes == 640U * 640U * 3U,
          "rendered manifest should expose the preproc tessellated handoff with semantic shape");
  // The rendered manifest now stores per-output logical shapes instead of scalar output_width/height.
  // The primary output (index 0 in runtime_output_logical_shapes for the selected output) carries
  // the same geometry that rendered_info.logical_dims was populated from.
  require(!stage.processcvu.runtime_output_logical_shapes.empty(),
          "rendered manifest should expose runtime output logical shapes");
  const auto& primary_shape = stage.processcvu.runtime_output_logical_shapes.front();
  require(primary_shape.size() >= 2 &&
              rendered_info.logical_dims.height == primary_shape[0] &&
              rendered_info.logical_dims.width == primary_shape[1],
          "stage helper and rendered manifest should agree on primary output width and height");
}));
