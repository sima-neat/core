#include "gst/GstHelpers.h"
#include "builder/NodeContractProvider.h"
#include "nodes/sima/Dequant.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/contract/ContractFacts.h"
#include "pipeline/internal/sima/StaticSpecBuilders.h"
#include "pipeline/internal/sima/stagesemantics/DequantStageSemantics.h"

#include "test_utils.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

void require_not_contains(const std::string& haystack, const std::string& needle,
                          const std::string& msg) {
  if (haystack.find(needle) != std::string::npos) {
    throw std::runtime_error(msg + " (found unexpected: " + needle + ")");
  }
}

} // namespace

int main() {
  try {
    require(simaai::neat::element_exists("neatdequant"), "required plugin missing (neatdequant)");

    {
      simaai::neat::DequantOptions opt;
      simaai::neat::pipeline_internal::sima::stagesemantics::DequantCanonicalFacts facts;
      facts.input_name = "MLA_0_0";
      facts.output_name = "bbox_0";
      facts.input_dtype = "INT8";
      facts.output_dtype = "FLOAT32";
      facts.layout = "HWC";
      simaai::neat::pipeline_internal::sima::QuantStaticSpec quant;
      quant.scales = {0.125};
      quant.zero_points = {-7};
      facts.input_quant = quant;

      simaai::neat::CompiledRuntimeContract upstream;
      upstream.logical_outputs.push_back(
          simaai::neat::pipeline_internal::sima::specbuilders::build_logical_output_static_spec(
              0, 0, 0, 0, 0, {80, 80, 64}, "INT8", "HWC", "MLA_0_0", "MLA_0_0", "MLA_0_0", 0, 0,
              quant));
      upstream.physical_outputs.push_back(
          simaai::neat::pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
              0, 0,
              simaai::neat::pipeline_internal::sima::specbuilders::
                  tensor_size_bytes_from_shape_dtype({80, 80, 64}, "INT8"),
              simaai::neat::pipeline_internal::sima::DeviceKind::Mla, "MLA_0_0"));
      std::string error;
      const auto compiled_contract = simaai::neat::pipeline_internal::sima::stagesemantics::
          build_dequant_compiled_contract_from_upstream(upstream, facts, &error);
      require(error.empty(), "model-managed dequant compiled contract error: " + error);
      opt.model_managed = true;
      opt.compiled_contract =
          std::make_shared<const simaai::neat::CompiledDequantContract>(compiled_contract);
      auto node = simaai::neat::nodes::Dequant(opt);
      const std::string frag = node->backend_fragment(7);
      require_contains(frag, "neatdequant name=n7_dequant", "model-managed fragment name mismatch");
      require_contains(frag, "stage-id=n7_dequant", "model-managed fragment stage-id missing");
      require_not_contains(frag,
                           "q-scale=", "model-managed fragment must not emit standalone quant");
      auto* provider = dynamic_cast<const simaai::neat::NodeContractProvider*>(node.get());
      require(provider != nullptr, "model-managed dequant should expose a contract provider");
      simaai::neat::ContractCompileInput input;
      simaai::neat::pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled_pipeline =
          simaai::neat::compile_node_contracts({node}, input, &diagnostics);
      require(compiled_pipeline.stages.size() == 1U,
              "model-managed dequant should compile to exactly one stage");
      const auto& compiled = compiled_pipeline.stages.front();
      require(compiled.dequant.has_value(),
              "model-managed dequant should compile to a dequant contract");
      require(!compiled.processcvu.has_value(),
              "model-managed dequant should not compile as processcvu");
      require(compiled.plugin_kind == "dequant",
              "model-managed dequant compiled plugin kind mismatch");
      require(compiled.dequant->runtime_contract.logical_inputs.size() == 1U,
              "model-managed dequant should expose one logical input");
      require(compiled.dequant->runtime_contract.logical_outputs.size() == 1U,
              "model-managed dequant should expose one logical output");
      const auto& logical_input = compiled.dequant->runtime_contract.logical_inputs.front();
      const auto& logical_output = compiled.dequant->runtime_contract.logical_outputs.front();
      require(logical_input.logical_name == "MLA_0_0",
              "model-managed dequant input alias mismatch");
      require(logical_input.shape == std::vector<std::int64_t>({80, 80, 64}),
              "model-managed dequant input shape mismatch");
      require(logical_input.quant.has_value(),
              "model-managed dequant should preserve input quant facts");
      require(logical_input.quant->scales == std::vector<double>{0.125},
              "model-managed dequant quant scale mismatch");
      require(logical_input.quant->zero_points == std::vector<std::int64_t>{-7},
              "model-managed dequant quant zero-point mismatch");
      require(logical_output.logical_name == "bbox_0",
              "model-managed dequant output alias mismatch");
      require(logical_output.shape == std::vector<std::int64_t>({80, 80, 64}),
              "model-managed dequant output shape mismatch");
      require(logical_output.dtype == "FLOAT32", "model-managed dequant output dtype mismatch");
    }

    {
      simaai::neat::DequantOptions opt;
      simaai::neat::CompiledDequantContract compiled_contract;
      compiled_contract.runtime_contract.plugin_kind = "dequant";

      simaai::neat::pipeline_internal::sima::QuantStaticSpec quant;
      quant.scales = {0.125};
      quant.zero_points = {-7};
      const std::vector<std::int64_t> shape = {80, 80, 64};
      const std::uint64_t input_size =
          simaai::neat::pipeline_internal::sima::specbuilders::tensor_size_bytes_from_shape_dtype(
              shape, "INT8");
      const std::uint64_t output_size =
          simaai::neat::pipeline_internal::sima::specbuilders::tensor_size_bytes_from_shape_dtype(
              shape, "FLOAT32");
      std::uint64_t output_offset = 0U;

      for (int i = 0; i < 6; ++i) {
        const std::string input_name = "MLA_0_" + std::to_string(i);
        const std::string output_name = "bbox_" + std::to_string(i);

        auto logical_input =
            simaai::neat::pipeline_internal::sima::specbuilders::build_logical_input_static_spec(
                i, i, i, shape, "INT8", "HWC", input_name, "input_tensor", input_name, 0, 0,
                simaai::neat::pipeline_internal::sima::TensorMaterializationKind::Direct, quant);
        logical_input.size_bytes = input_size;
        compiled_contract.runtime_contract.logical_inputs.push_back(std::move(logical_input));
        compiled_contract.runtime_contract.input_bindings.push_back(
            simaai::neat::pipeline_internal::sima::specbuilders::build_input_binding_static_spec(
                0, i, "input_tensor", input_name, i, i, i, input_size, 0, true));
        compiled_contract.runtime_contract.physical_inputs.push_back(
            simaai::neat::pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
                i, i, input_size, simaai::neat::pipeline_internal::sima::DeviceKind::Mla,
                input_name, i, 0));
        compiled_contract.runtime_contract.logical_outputs.push_back(
            simaai::neat::pipeline_internal::sima::specbuilders::build_logical_output_static_spec(
                i, i, 0, i, i, shape, "FLOAT32", "HWC", output_name, output_name, "output_tensor",
                static_cast<std::int64_t>(output_offset), output_size));
        compiled_contract.runtime_contract.output_order.push_back(
            simaai::neat::pipeline_internal::sima::specbuilders::build_output_route_static_spec(
                i, i, i, "output_tensor", "output_tensor"));
        output_offset += output_size;
      }

      compiled_contract.runtime_contract.physical_outputs.push_back(
          simaai::neat::pipeline_internal::sima::specbuilders::build_physical_buffer_static_spec(
              0, 0, output_offset, simaai::neat::pipeline_internal::sima::DeviceKind::Cpu,
              "output_tensor"));

      opt.model_managed = true;
      opt.compiled_contract =
          std::make_shared<const simaai::neat::CompiledDequantContract>(compiled_contract);
      auto node = simaai::neat::nodes::Dequant(opt);
      auto* provider = dynamic_cast<const simaai::neat::NodeContractProvider*>(node.get());
      require(provider != nullptr,
              "multi-input model-managed dequant should expose a contract provider");
      simaai::neat::ContractCompileInput input;
      simaai::neat::pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled_pipeline =
          simaai::neat::compile_node_contracts({node}, input, &diagnostics);
      require(diagnostics.errors.empty(),
              "multi-input model-managed dequant should compile without manifest diagnostics");
      require(compiled_pipeline.stages.size() == 1U,
              "multi-input model-managed dequant should compile to exactly one stage");
      const auto& compiled = compiled_pipeline.stages.front();
      require(compiled.dequant.has_value(),
              "multi-input model-managed dequant should compile to a dequant contract");
      require(compiled.dequant->runtime_contract.logical_inputs.size() == 6U,
              "multi-input model-managed dequant should preserve logical input count");
      require(compiled.dequant->runtime_contract.input_bindings.size() == 6U,
              "multi-input model-managed dequant should preserve binding count");
      require(compiled.dequant->runtime_contract.physical_inputs.size() == 6U,
              "multi-input model-managed dequant should preserve physical input count");
      require(compiled.dequant->runtime_contract.logical_outputs.size() == 6U,
              "multi-input model-managed dequant should preserve logical output count");
      for (int i = 0; i < 6; ++i) {
        const std::string input_name = "MLA_0_" + std::to_string(i);
        require(compiled.dequant->runtime_contract.logical_inputs[i].physical_index == i,
                "multi-input model-managed dequant should preserve logical input physical index");
        require(compiled.dequant->runtime_contract.logical_inputs[i].segment_name == input_name,
                "multi-input model-managed dequant should preserve logical input segment name");
        require(compiled.dequant->runtime_contract.input_bindings[i].src_physical_output_index == i,
                "multi-input model-managed dequant should preserve source physical output index");
        require(compiled.dequant->runtime_contract.input_bindings[i].source_segment_name ==
                    input_name,
                "multi-input model-managed dequant should preserve source segment name");
        require(compiled.dequant->runtime_contract.physical_inputs[i].physical_index == i,
                "multi-input model-managed dequant should preserve physical input index");
        require(compiled.dequant->runtime_contract.physical_inputs[i].source_physical_index == i,
                "multi-input model-managed dequant should preserve physical input source index");
        require(compiled.dequant->runtime_contract.physical_inputs[i].segment_name == input_name,
                "multi-input model-managed dequant should preserve physical input segment name");
      }
    }

    {
      simaai::neat::DequantOptions opt;
      opt.element_name = "dq0";
      opt.model_managed = false;
      opt.q_scale = 0.125;
      opt.q_zp = -7;
      auto node = simaai::neat::nodes::Dequant(opt);
      const std::string frag = node->backend_fragment(3);
      require_contains(frag, "neatdequant name=dq0",
                       "standalone dequant fragment should use neatdequant");
      require_contains(frag, "q-scale=0.125", "standalone dequant fragment should emit q-scale");
      require_contains(frag, "q-zp=-7", "standalone dequant fragment should emit q-zp");
      require_not_contains(
          frag, "stage-id=", "standalone dequant fragment must not emit processcvu stage-id");
      auto* provider = dynamic_cast<const simaai::neat::NodeContractProvider*>(node.get());
      require(provider != nullptr, "standalone dequant should expose a contract provider");
      simaai::neat::ContractCompileInput input;
      simaai::neat::pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled_pipeline =
          simaai::neat::compile_node_contracts({node}, input, &diagnostics);
      require(diagnostics.errors.empty(),
              "standalone dequant should compile without manifest diagnostics");
      require(compiled_pipeline.stages.size() == 1U,
              "standalone dequant should compile to exactly one stage");
      const auto& compiled = compiled_pipeline.stages.front();
      require(compiled.dequant.has_value(),
              "standalone dequant should compile to a dequant contract");
      require(!compiled.processcvu.has_value(),
              "standalone dequant should not compile as processcvu");
      require(compiled.plugin_kind == "dequant",
              "standalone dequant compiled plugin kind mismatch");
      require(compiled.dequant->runtime_contract.plugin_kind == "dequant",
              "standalone dequant runtime plugin kind mismatch");
      require(compiled.dequant->runtime_contract.logical_inputs.size() == 1U,
              "standalone dequant should expose one logical input");
      require(compiled.dequant->runtime_contract.logical_inputs.front().quant.has_value(),
              "standalone dequant should preserve input quant facts");
      require(compiled.dequant->runtime_contract.logical_inputs.front().quant->scales ==
                  std::vector<double>{0.125},
              "standalone dequant quant scale mismatch");
      require(compiled.dequant->runtime_contract.logical_inputs.front().quant->zero_points ==
                  std::vector<std::int64_t>{-7},
              "standalone dequant quant zero-point mismatch");
    }

    std::cout << "[OK] unit_dequant_node_fragment_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
