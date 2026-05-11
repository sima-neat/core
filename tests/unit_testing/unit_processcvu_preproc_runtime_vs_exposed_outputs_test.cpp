#include "nodes/sima/Preproc.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "test_main.h"

#include <algorithm>
#include <memory>

namespace {

simaai::neat::PreprocOptions make_split_preproc_options() {
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

RUN_TEST("unit_processcvu_preproc_runtime_vs_exposed_outputs_test", ([] {
  using namespace simaai::neat;

  auto node = nodes::Preproc(make_split_preproc_options());
  pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
  const auto compiled =
      compile_node_contracts(std::vector<std::shared_ptr<Node>>{node}, ContractCompileInput{},
                             &diagnostics);
  require(diagnostics.errors.empty(), "preproc contract compile failed");
  require(compiled.stages.size() == 1U, "preproc compile should emit one stage");
  require(compiled.stages.front().processcvu.has_value(), "preproc should compile to processcvu");

  const auto& contract = *compiled.stages.front().processcvu;
  // Runtime contract publishes both the rgb and tessellated outputs; the
  // exposed view narrows to the selected tessellated handoff.
  const auto tess_runtime_it =
      std::find_if(contract.runtime_contract.logical_outputs.begin(),
                   contract.runtime_contract.logical_outputs.end(),
                   [](const auto& logical) { return logical.logical_name == "output_tessellated_image"; });
  require(tess_runtime_it != contract.runtime_contract.logical_outputs.end(),
          "runtime contract should retain the tessellated logical output");
  require(tess_runtime_it->shape == std::vector<std::int64_t>({640, 640, 3}),
          "tessellated handoff should preserve the semantic output shape");
  require(tess_runtime_it->size_bytes == 640U * 640U * 3U,
          "tessellated handoff should preserve the packed MLA ingress byte size");
  require(contract.exposed_view.exposed_logical_outputs.size() == 1U,
          "exposed view should project one logical output");
  require(contract.exposed_view.exposed_output_order.size() == 1U,
          "exposed view should project one output route");
  require(contract.exposed_view.exposed_output_order.front().logical_output_index ==
              tess_runtime_it->logical_index,
          "projected exposed route should preserve selected runtime logical index");
  require(contract.exposed_view.primary_output_name == "output_tessellated_image",
          "primary exposed output should follow tessellated handoff");
}));
