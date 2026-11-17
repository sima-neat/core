#include <neat.h>
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/CanonicalKernelLowering.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "test_utils.h"

#include <array>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

namespace n = simaai::neat;
namespace si = n::pipeline_internal::sima;
namespace client = simaai::scheduler::client;

namespace {
void check_contract(int width, int height) {
  n::MetoakDepthOptions options;
  options.width = width;
  options.height = height;
  options.num_buffers = 4;
  n::MetoakDepth node(options);
  require(n::nodes::MetoakDepth(options)->kind() == "MetoakDepth", "public factory");
  require(node.element_names(2) == std::vector<std::string>{"n2_metoak_depth"}, "stable name");
  require(options.summary().find("graph_id=") == std::string::npos, "no legacy dispatch ID");
  const auto definition = node.contract_definition();
  require(definition.inputs.size() == 6 && definition.outputs.size() == 3, "fixed arity");
  require(definition.inputs[3].port_id == "disp_src" &&
              definition.inputs[4].port_id == "bf_mm_src" &&
              definition.outputs[1].port_id == "depth_dst",
          "canonical names");

  n::ContractCompileInput input;
  n::CompiledNodeContract compiled;
  std::string error;
  require(node.compile_node_contract(input, &compiled, &error), error);
  require(compiled.native_execution_capacity == 4U, "explicit buffer depth");
  require(compiled.processcvu.has_value(), "compiled semantic contract");
  const auto& runtime = compiled.processcvu->runtime_contract;
  const std::array<std::string, 6> in_types{"UINT8", "UINT8", "UINT8", "UINT16", "FP32", "FP32"};
  const std::array<std::string, 3> out_types{"UINT8", "UINT16", "FP32"};
  require(runtime.logical_inputs.size() == 6 && runtime.logical_outputs.size() == 3,
          "all ports retained");
  for (std::size_t i = 0; i < in_types.size(); ++i)
    require(runtime.logical_inputs[i].dtype == in_types[i], "heterogeneous input dtype");
  for (std::size_t i = 0; i < out_types.size(); ++i)
    require(runtime.logical_outputs[i].dtype == out_types[i], "heterogeneous output dtype");
  require(compiled.processcvu->exposed_view.primary_output_name == "depth_dst",
          "depth is primary, not exclusive");

  n::InputOptions source;
  source.payload_type = n::PayloadType::Tensor;
  source.memory_policy = n::InputMemoryPolicy::Ev74;
  source.width = source.max_width = width;
  source.height = source.max_height = height;
  const std::vector<std::shared_ptr<n::Node>> nodes{
      n::nodes::Input(source), n::nodes::MetoakDepth(options), n::nodes::Output()};
  si::ManifestBuildDiagnostics diagnostics;
  const auto contracts = n::compile_node_contracts(nodes, input, &diagnostics);
  require(contracts.fully_renderable && contracts.stages.size() == 1 && diagnostics.errors.empty(),
          "Input/MetoakDepth/Output contract compilation");
  auto manifest = n::render_manifest_from_compiled_contracts(contracts, input, &diagnostics);
  require(manifest.has_value() && diagnostics.errors.empty(), "manifest render");
  auto failure = si::prepare_kernel_stage_recipes(&*manifest, {}, {}, {}, {});
  require(!failure, failure ? failure->field + ": " + failure->message : "recipe");
  si::finalize_pipeline_storage(*manifest);
  failure = si::finalize_stage_publications(&*manifest);
  require(!failure, failure ? failure->field + ": " + failure->message : "publication");
  const auto& stage = manifest->stages.front();
  require(stage.frame_layout && stage.tensor_publication && stage.native_kernel,
          "finalized typed stage");
  require(stage.frame_layout->inputs.size() == 6, "all public input views are bound");
  for (std::size_t i = 0; i < stage.frame_layout->inputs.size(); ++i) {
    const auto& route = stage.frame_layout->inputs[i];
    const auto* binding = std::get_if<client::PhysicalBinding>(&route.source);
    require(
        binding && binding->pad == 0 && binding->binding_index == i && binding->byte_offset == 0 &&
            route.view_offset == 0 && binding->size_bytes == runtime.logical_inputs[i].size_bytes,
        "public tensor offsets must come from each frame's binding table, not allocation starts");
  }
  const auto& member = stage.native_kernel->members.front();
  require(member.inputs.size() == 6 && member.outputs.size() == 3, "typed member arity");
  auto inspection = client::inspect_kernel(member.executable);
  require(std::holds_alternative<client::KernelInspection>(inspection), "kernel inspection");
  const auto& checked = std::get<client::KernelInspection>(inspection);
  require(checked.inputs()[3].dtype == client::DType::UInt16 &&
              checked.outputs()[1].dtype == client::DType::UInt16,
          "unsigned numerical authority");
  require(checked.inputs()[4].shape == std::vector<std::int64_t>{1} &&
              checked.inputs()[5].shape == std::vector<std::int64_t>{3},
          "calibration ranks");
  require(stage.logical_outputs.size() == 3, "RGB and XYZ must not disappear behind primary depth");

  auto bad = stage;
  bad.processcvu.input_tensors[3] =
      si::AuthoredTensorDescriptor(bad.processcvu.input_tensors[3].descriptor(), "INT16");
  require(std::holds_alternative<client::Error>(si::kernel_lowering::check_visual_stage(bad)),
          "signed disparity must be rejected, not reinterpreted");
  bad = stage;
  bad.processcvu.resolved_exec_backend = "A65";
  require(std::holds_alternative<client::Error>(si::kernel_lowering::check_visual_stage(bad)),
          "no CPU fallback");
  bad = stage;
  bad.processcvu.batch_size = 2;
  require(std::holds_alternative<client::Error>(si::kernel_lowering::check_visual_stage(bad)),
          "batch one");
}
void invalid_options() {
  for (const auto [w, h] :
       std::array<std::pair<int, int>, 6>{{{0, 8}, {7, 8}, {9, 8}, {8, 9}, {2050, 8}, {8, 1538}}}) {
    n::MetoakDepthOptions opt;
    opt.width = w;
    opt.height = h;
    bool rejected = false;
    try {
      (void)n::MetoakDepth(opt).contract_definition();
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    require(rejected, "invalid geometry rejected before device use");
  }
  n::MetoakDepthOptions opt;
  opt.width = 8;
  opt.height = 8;
  opt.y_name = "other";
  bool rejected = false;
  try {
    (void)n::MetoakDepth(opt).contract_definition();
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  require(rejected, "preserve canonical-name contract");
}
} // namespace
int main() {
  try {
    for (const auto [w, h] :
         std::array<std::pair<int, int>, 5>{{{8, 8}, {10, 14}, {18, 10}, {640, 360}, {2048, 1536}}})
      check_contract(w, h);
    invalid_options();
    std::cout << "PASS MetoakDepth API, typed contracts, publication and negative controls\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
