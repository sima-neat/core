#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/contract/CompiledNodeContract.h"
#include "pipeline/internal/contract/PluginCompiledContracts.h"
#include "pipeline/internal/sima/CompiledProcessCvuContractQuery.h"
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat {
struct PreprocOptions;
}

namespace simaai::neat::internal {
enum class ExecutionStageKind : std::uint8_t;
}

namespace simaai::neat::pipeline_internal::sima::stagesemantics {

enum class ProcessCvuOutputRepresentation : std::uint8_t {
  DenseTensor = 0,
  PackedBlob = 1,
  PackedTensor = 2,
};

struct ProcessCvuCanonicalInputFact {
  int logical_index = -1;
  int physical_index = -1;
  std::string physical_name;
  std::string logical_name;
  std::vector<std::int64_t> shape;
  std::uint64_t size_bytes = 0;
  std::string dtype;
  std::string layout;
  std::int64_t byte_offset = 0;
  TensorMaterializationKind materialization_kind = TensorMaterializationKind::Direct;
  std::optional<QuantStaticSpec> quant;
};

struct ProcessCvuCanonicalBindingFact {
  int sink_pad_index = 0;
  int local_logical_input_index = -1;
  int src_logical_output_index = -1;
  int src_output_slot = -1;
  int src_physical_output_index = -1;
  std::uint64_t src_physical_size_bytes = 0;
  std::int64_t src_physical_byte_offset = 0;
  bool required = true;
  std::string cm_input_name;
  std::string source_segment_name;
};

struct ProcessCvuCanonicalOutputFact {
  ProcessCvuOutputRepresentation representation = ProcessCvuOutputRepresentation::DenseTensor;
  int logical_index = -1;
  int physical_index = -1;
  int output_slot = -1;
  int tensor_index = -1;
  std::string physical_name;
  std::string logical_name;
  std::vector<std::int64_t> shape;
  std::string dtype;
  std::string layout;
  std::int64_t byte_offset = 0;
  std::uint64_t size_bytes = 0;
  std::optional<QuantStaticSpec> quant;
};

struct ProcessCvuCanonicalRouteFact {
  int output_slot = -1;
  int logical_output_index = -1;
  int tensor_index = -1;
  std::string cm_output_name;
  std::string segment_name;
};

struct ProcessCvuCanonicalFacts {
  std::vector<std::string> physical_input_names;
  std::vector<std::string> physical_output_names;
  std::vector<ProcessCvuCanonicalInputFact> inputs;
  std::vector<ProcessCvuCanonicalBindingFact> input_bindings;
  std::vector<ProcessCvuCanonicalOutputFact> outputs;
  std::vector<ProcessCvuCanonicalRouteFact> output_order;
  std::vector<std::string> published_output_names;
  std::string primary_output_name;
  bool preserve_physical_outputs = false;
};

struct ProcessCvuCanonicalCompileInputs {
  ProcessCvuStagePayload payload;
  ProcessCvuCanonicalFacts facts;
};

std::string canonical_processcvu_graph_family(const std::string& graph_family);

ProcessCvuCanonicalCompileInputs build_processcvu_compile_inputs_from_options(
    const ::simaai::neat::PreprocOptions& opt);

CompiledProcessCvuContract build_processcvu_compiled_contract_from_options(
    const ::simaai::neat::PreprocOptions& opt);

CompiledProcessCvuContract build_processcvu_compiled_contract_from_facts(
    const ProcessCvuStagePayload& payload,
    const ProcessCvuCanonicalFacts& facts);

CompiledProcessCvuContract build_processcvu_compiled_contract(
    const ProcessCvuCanonicalCompileInputs& inputs);


CompiledProcessCvuContract build_processcvu_mpk_preadapter_compiled_contract_for_stage_kind(
    const MpkContract& contract,
    ::simaai::neat::internal::ExecutionStageKind stage_kind,
    const std::optional<std::string>& exact_stage_name_or_id = std::nullopt,
    const std::optional<std::string>& canonical_handoff_segment_name = std::nullopt);
CompiledProcessCvuContract build_processcvu_mpk_compiled_contract_for_stage_kind(
    const MpkContract& contract,
    ::simaai::neat::internal::ExecutionStageKind stage_kind,
    const std::optional<std::string>& exact_stage_name_or_id = std::nullopt,
    const std::optional<std::string>& canonical_handoff_segment_name = std::nullopt,
    const std::optional<bool>& preproc_single_output_handoff = std::nullopt,
    const std::string& input_format = {},
    int input_depth = 0,
    int max_input_width = 0,
    int max_input_height = 0,
    bool normalize = false,
    const std::vector<float>& mean = {},
    const std::vector<float>& stddev = {});

std::string resolve_preproc_primary_output_name(
    const std::vector<std::string>& runtime_output_names,
    bool tessellate,
    const std::string& requested_primary_output_name = {});

std::uint64_t processcvu_dtype_size_bytes_from_token(const std::string& raw_dtype);

std::uint64_t processcvu_tensor_size_bytes_from_spec(const TensorStaticSpec& tensor);

bool build_processcvu_node_contract(const std::string& node_kind,
                                    const std::string& element_name,
                                    const std::string& logical_stage_id,
                                    const NodeContractDefinition& definition,
                                    const CompiledProcessCvuContract& compiled,
                                    CompiledNodeContract* out,
                                    std::string* err = nullptr);

} // namespace simaai::neat::pipeline_internal::sima::stagesemantics
