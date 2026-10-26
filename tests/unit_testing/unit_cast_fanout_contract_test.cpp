#include "nodes/sima/Cast.h"
#include "pipeline/Tensor.h"
#include "pipeline/internal/contract/ContractCompiler.h"
#include "pipeline/internal/sima/ContractRender.h"
#include "pipeline/internal/sima/StageMemoryPolicy.h"
#include "pipeline/internal/sima/StageRecipePreparation.h"
#include "pipeline/internal/TerminalOutputContractQuery.h"
#include "pipeline/internal/TensorMath.h"
#include "test_main.h"
#include "test_utils.h"

#include <simaai/scheduler/client/frame_layout.h>

namespace {

simaai::neat::Tensor make_bf16_head(int logical_index, const std::vector<int64_t>& shape,
                                    int64_t physical_offset, const std::string& name) {
  using namespace simaai::neat;
  Tensor tensor;
  tensor.dtype = TensorDType::BFloat16;
  tensor.layout = TensorLayout::CHW;
  tensor.shape = shape;
  tensor.strides_bytes = simaai::neat::pipeline_internal::contiguous_strides_bytes(shape, 2U);

  std::size_t bytes = 2U;
  for (const auto dim : shape) {
    bytes *= static_cast<std::size_t>(dim);
  }
  tensor.storage = make_cpu_owned_storage(bytes);
  tensor.route.logical_index = logical_index;
  tensor.route.route_slot = logical_index;
  tensor.route.physical_index = 0;
  tensor.route.memory_index = 0;
  tensor.route.physical_byte_offset = physical_offset;
  tensor.route.name = name;
  tensor.route.backend_name = name;
  tensor.route.segment_name = name;
  return tensor;
}

} // namespace

RUN_TEST(
    "unit_cast_fanout_contract_test", ([] {
      using namespace simaai::neat;

      {
        auto strided = make_bf16_head(0, {1, 2, 2}, 4, "strided");
        strided.strides_bytes = {16, 8, 2};
        strided.storage = make_cpu_owned_storage(32);
        auto node = nodes::Cast(CastOptions{.direction = CastDirection::Bf16ToFp32});
        ContractCompileInput seed;
        seed.processcvu_requested_run_target = "A65";
        seed.ingress.ingress_sample = sample_from_tensors(TensorList{strided});
        pipeline_internal::sima::ManifestBuildDiagnostics diag;
        const auto contracts = compile_node_contracts({node}, seed, &diag);
        require(diag.errors.empty() && contracts.stages.size() == 1 &&
                    contracts.stages.front().transport.has_value(),
                "strided Cast compile");
        const auto& cvu = *contracts.stages.front().transport->processcvu_payload;
        require(cvu.input_tensors.front().descriptor().layout_kind == SIMA_EV_LAYOUT_STRIDED &&
                    cvu.input_tensors.front().descriptor().storage.addr == 4 &&
                    cvu.input_tensors.front().descriptor().layout.strided.strides_bytes[1] == 8 &&
                    cvu.output_tensors.front().descriptor().storage.addr == 8 &&
                    cvu.output_tensors.front().descriptor().layout.strided.strides_bytes[1] == 16,
                "Cast descriptor lost strided offsets");
        const auto& runtime = contracts.stages.front().transport->runtime_contract;
        require(runtime.physical_inputs.front().size_bytes >= 16 &&
                    runtime.physical_outputs.front().size_bytes >= 32 &&
                    runtime.input_bindings.size() == 1,
                "Cast allocation ignores addressed strided extent");
        const auto manifest = render_manifest_from_compiled_contracts(contracts, seed, &diag);
        require(manifest && diag.errors.empty() &&
                    manifest->stages.front().processcvu.resolved_exec_backend == "A65",
                "standalone Cast did not honor A65 target");
        std::string terminal_error;
        const auto terminal =
            pipeline_internal::terminal_output_contract::build_output_override_from_manifest(
                *manifest, {}, &terminal_error);
        require(terminal && terminal->outputs.size() == 1 &&
                    terminal->outputs.front().dtype == TensorDType::Float32,
                "Cast public terminal override lost authoritative dtype: " + terminal_error);
        strided.strides_bytes = {16, -8, 2};
        seed.ingress.ingress_sample = sample_from_tensors(TensorList{strided});
        pipeline_internal::sima::ManifestBuildDiagnostics bad;
        (void)compile_node_contracts({node}, seed, &bad);
        require(!bad.errors.empty(), "negative Cast stride accepted");
      }
      Tensor head0 = make_bf16_head(0, {4, 2, 2}, 0, "head_0");
      Tensor head1 = make_bf16_head(1, {4, 1, 2}, 32, "head_1");
      Tensor head2 = make_bf16_head(2, {2, 1, 2}, 48, "head_2");
      Tensor head3 = make_bf16_head(3, {5, 2, 2}, 56, "head_3");
      Tensor head4 = make_bf16_head(4, {5, 1, 2}, 96, "head_4");
      Tensor head5 = make_bf16_head(5, {3, 1, 2}, 116, "head_5");
      const Sample ingress =
          sample_from_tensors(TensorList{head0, head1, head2, head3, head4, head5});

      auto cast = simaai::neat::nodes::Cast(
          CastOptions{.direction = CastDirection::Bf16ToFp32, .element_name = "post_cast"});

      ContractCompileInput input;
      input.ingress.ingress_sample = ingress;
      pipeline_internal::sima::ManifestBuildDiagnostics diagnostics;
      const auto compiled = compile_node_contracts({cast}, input, &diagnostics);

      require(diagnostics.errors.empty(),
              "compile_node_contracts should succeed for multi-output Cast");
      require(compiled.stages.size() == 1U,
              "multi-output Cast should compile to exactly one stage");
      require(compiled.stages.front().transport.has_value(),
              "Cast should compile as a transport contract");

      const auto fragment = cast->backend_fragment(0);
      require(fragment.find("neatkernelrunner name=") != std::string::npos,
              "standalone Cast must use Runtime without an opt-in");
      require(fragment.find("num-buffers=") == std::string::npos &&
                  fragment.find("stage-id=") == std::string::npos,
              "Cast capacity and identity must use typed preparation, not legacy properties");
      const auto& transport = *compiled.stages.front().transport;
      require(transport.payload_kind == pipeline_internal::sima::StagePayloadKind::ProcessCvu &&
                  transport.processcvu_payload &&
                  transport.processcvu_payload->graph_name == "cast",
              "standalone Cast lacks a canonical semantic execution payload");
      auto rendered = render_manifest_from_compiled_contracts(compiled, input, &diagnostics);
      require(rendered && diagnostics.errors.empty() && rendered->stages.size() == 1,
              "standalone Cast lacks a renderable semantic contract");
      const auto preparation = pipeline_internal::sima::prepare_kernel_stage_recipes(
          &*rendered, std::nullopt, {}, {}, {});
      require(!preparation, preparation ? preparation->field + ": " + preparation->message : "");
      pipeline_internal::sima::finalize_pipeline_storage(*rendered);
      require(rendered->stages.front().native_kernel &&
                  rendered->stages.front().frame_arena_role ==
                      pipeline_internal::sima::FrameArenaRole::Allocate &&
                  rendered->stages.front().frame_layout &&
                  rendered->stages.front().frame_layout->ownership ==
                      simaai::scheduler::client::FrameOwnership::Allocate &&
                  rendered->stages.front().frame_layout->outputs.size() == 6U,
              "standalone Cast lacks checked direct arena placement");
      const auto& runtime = transport.runtime_contract;
      require(runtime.logical_outputs.size() == 6U,
              "Cast should publish all six logical outputs from the upstream contract");
      require(runtime.logical_outputs[0].logical_index == 0 &&
                  runtime.logical_outputs[0].byte_offset == 0 &&
                  runtime.logical_outputs[0].size_bytes == 64U &&
                  runtime.logical_outputs[0].dtype == "FP32",
              "Cast should convert the first BF16 head to FP32 bytes and dtype");
      require(runtime.logical_outputs[1].logical_index == 1 &&
                  runtime.logical_outputs[1].byte_offset == 64 &&
                  runtime.logical_outputs[1].size_bytes == 32U,
              "Cast should preserve logical ordering and double the second head byte facts");
      require(runtime.logical_outputs[2].logical_index == 2 &&
                  runtime.logical_outputs[2].byte_offset == 96 &&
                  runtime.logical_outputs[2].size_bytes == 16U,
              "Cast should preserve the third logical head");
      require(runtime.logical_outputs[3].logical_index == 3 &&
                  runtime.logical_outputs[3].byte_offset == 112 &&
                  runtime.logical_outputs[3].size_bytes == 80U,
              "Cast should preserve the fourth logical head");
      require(runtime.logical_outputs[4].logical_index == 4 &&
                  runtime.logical_outputs[4].byte_offset == 192 &&
                  runtime.logical_outputs[4].size_bytes == 40U,
              "Cast should preserve the fifth logical head");
      require(runtime.logical_outputs[5].logical_index == 5 &&
                  runtime.logical_outputs[5].byte_offset == 232 &&
                  runtime.logical_outputs[5].size_bytes == 24U,
              "Cast should preserve the sixth logical head");
      require(runtime.output_order.size() == 6U,
              "Cast should publish output ordering for every logical output");
    }));
