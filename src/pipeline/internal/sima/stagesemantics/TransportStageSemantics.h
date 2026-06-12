#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "contracts/NodeContractDefinition.h"
#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"

#include <string>

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

struct TransportCanonicalFacts {
  std::string plugin_kind;
  std::string kernel_kind;
  bool model_managed_stage = false;
  pipeline_internal::sima::StagePayloadKind payload_kind =
      pipeline_internal::sima::StagePayloadKind::None;
  std::optional<pipeline_internal::sima::ProcessCvuStagePayload> processcvu_payload;
  std::optional<CompiledRuntimeContract> runtime_contract;
};

enum class TransportCastDirection : std::uint8_t {
  Bf16ToFp32 = 0,
  Fp32ToBf16 = 1,
};

CompiledTransportContract
build_transport_compiled_contract_from_facts(const TransportCanonicalFacts& facts);
CompiledRuntimeContract build_transport_runtime_contract_from_processcvu_compiled(
    const CompiledProcessCvuContract& compiled);

std::optional<TransportCastDirection>
infer_cast_direction_from_upstream_contract(const CompiledRuntimeContract& upstream);
CompiledRuntimeContract
build_cast_runtime_contract_from_external_input(int width, int height, int depth,
                                                const std::string& layout,
                                                TransportCastDirection direction, std::string* err);
CompiledRuntimeContract
build_cast_runtime_contract_from_upstream(const CompiledRuntimeContract& upstream,
                                          TransportCastDirection direction, std::string* err);

bool build_transport_node_contract(const std::string& node_kind, const std::string& element_name,
                                   const std::string& logical_stage_id,
                                   const NodeContractDefinition& definition,
                                   const CompiledTransportContract& compiled,
                                   CompiledNodeContract* out, std::string* error_message = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
