#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/sima/static_contract/PhysicalExecutionPlan.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
namespace internal {
struct ModelFragmentPlanSource;
}

struct CompiledRuntimeContract {
  std::string plugin_kind;
  // Checked image semantics are independent of later executable preparation.
  std::shared_ptr<const ::simaai::scheduler::client::TensorStageImagePublication> image_publication;
  std::vector<pipeline_internal::sima::LogicalInputStaticSpec> logical_inputs;
  std::vector<pipeline_internal::sima::InputBindingStaticSpec> input_bindings;
  std::vector<pipeline_internal::sima::PhysicalBufferStaticSpec> physical_inputs;
  std::vector<pipeline_internal::sima::PhysicalBufferStaticSpec> physical_outputs;
  std::vector<pipeline_internal::sima::LogicalTensorStaticSpec> logical_outputs;
  std::vector<pipeline_internal::sima::StageOutputRoute> output_order;
  std::vector<pipeline_internal::sima::QuantStaticSpec> output_quant;
  std::vector<std::string> required_preprocess_meta_fields;
  std::optional<::simaai::scheduler::client::AllocationPolicy> allocation_policy;
  std::optional<std::size_t> reuse_input_pad;
  std::shared_ptr<const ::simaai::scheduler::client::FrameLayout> frame_layout;
  // Mirrors MlaStaticContract::consumer_keeps_distinct_physical_inputs. True
  // when the stage's compiled binary expects N>1 distinct physical input
  // segments (native multi-IFM dispatch). Used downstream to gate
  // physical-segment coalescing helpers.
  bool consumer_keeps_distinct_physical_inputs = false;
  // Optional .elf-derived I/O symbol names (Phase 4). Empty = legacy
  // synthesized "ifmN" / "ofmN" naming. See MlaStaticContractExtractor.h.
  std::vector<std::string> elf_ifm_symbol_names;
  std::vector<std::string> elf_ofm_symbol_names;
};

struct CompiledExposedView {
  std::vector<pipeline_internal::sima::StageOutputRoute> exposed_output_order;
  std::vector<pipeline_internal::sima::LogicalTensorStaticSpec> exposed_logical_outputs;
  std::string primary_output_name;
};

struct CompiledProcessCvuContract {
  // Immutable model/command provenance for contextual fragment compilation.
  // Runtime payloads never consult this build-time source.
  std::shared_ptr<const internal::ModelFragmentPlanSource> model_execution_source;
  std::vector<pipeline_internal::sima::static_contract::PhysicalCommandId> physical_command_ids;
  pipeline_internal::sima::ProcessCvuStagePayload payload;
  CompiledRuntimeContract runtime_contract;
  CompiledExposedView exposed_view;
  std::optional<pipeline_internal::sima::static_contract::PhysicalCommandRole>
      physical_command_role;
  bool preproc_single_output_handoff = false;
};

struct CompiledMlaContract {
  pipeline_internal::sima::ProcessMlaStagePayload payload;
  std::vector<pipeline_internal::sima::TensorStaticSpec> inputs;
  std::vector<pipeline_internal::sima::PhysicalBufferStaticSpec> dispatcher_physical_outputs;
  CompiledRuntimeContract runtime_contract;
};

struct CompiledBoxDecodeContract {
  pipeline_internal::sima::BoxDecodeStagePayload payload;
  CompiledRuntimeContract runtime_contract;
};

struct CompiledKernelContract {
  CompiledRuntimeContract runtime_contract;
  std::shared_ptr<const pipeline_internal::sima::KernelStageRecipe> recipe;
};

struct CompiledDequantContract {
  CompiledRuntimeContract runtime_contract;
};

struct CompiledTransportContract {
  std::string plugin_kind;
  std::string kernel_kind;
  CompiledRuntimeContract runtime_contract;
  pipeline_internal::sima::StagePayloadKind payload_kind =
      pipeline_internal::sima::StagePayloadKind::None;
  std::optional<pipeline_internal::sima::ProcessCvuStagePayload> processcvu_payload;
  bool model_managed_stage = false;
  std::shared_ptr<const ::simaai::scheduler::client::TvmProgram> tvm_program;
};

} // namespace simaai::neat
