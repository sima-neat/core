#include "pipeline/internal/sima/CanonicalGraphStageContract.h"
#include <array>
#include <iostream>
#include <stdexcept>

namespace {
namespace ps = simaai::neat::pipeline_internal::sima;
void require(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
ps::MpkTensorContract tensor(const char* name, const char* dtype, std::uint64_t bytes) {
  ps::MpkTensorContract result;
  result.tensor_index = result.physical_index = result.source_physical_index = 0;
  result.name = result.segment_name = name;
  result.dtype = dtype;
  result.dtype_source = ps::DTypeSource::ExplicitMpk;
  result.logical_shape = {2, 3, 33};
  result.size_bytes = bytes;
  return result;
}
ps::StageStaticSpec stage(std::uint64_t input_bytes) {
  ps::StageStaticSpec result;
  result.element_name = result.logical_stage_id = "canonical-projection";
  sima_ev_tensor_desc input{}, output{};
  input.dtype = SIMA_EV_DTYPE_INT8;
  output.dtype = SIMA_EV_DTYPE_FP32;
  input.storage.nbytes = input_bytes;
  output.storage.nbytes = 792;
  result.processcvu.input_tensors.emplace_back(input, "INT8");
  result.processcvu.output_tensors.emplace_back(output, "FP32");
  return result;
}
void inferred_extents() {
  const std::array inputs{tensor("input", "INT8", 0)};
  const std::array outputs{tensor("output", "FP32", 0)};
  const std::array<std::string, 1> names{"published"};
  // Exact extents already authored by Core's model descriptor synthesis. The
  // projection must retain compact tile rounding, padded HWC and CBlock sizes.
  for (auto extent : {208U, 288U}) {
    auto candidate = stage(extent);
    std::string error;
    require(ps::complete_graph_kernel_contract(&candidate, inputs, outputs, "", "", "slot", names,
                                               &error),
            error.c_str());
    require(candidate.logical_inputs.front().size_bytes == extent &&
                candidate.physical_inputs.front().size_bytes == extent &&
                candidate.input_bindings.front().src_physical_size_bytes == extent,
            "exact inferred storage extent was not preserved through every route");
    require(candidate.logical_outputs.front().size_bytes == 792 &&
                candidate.physical_outputs.front().size_bytes == 792,
            "dense FP32 output extent changed");
    require(candidate.logical_outputs.front().logical_name == "published" &&
                candidate.output_order.front().cm_output_name == "published" &&
                candidate.input_bindings.front().cm_input_name == "slot",
            "graph names/routes were not preserved");
    require(candidate.logical_outputs.front().stride_bytes ==
                std::vector<std::int64_t>{396, 132, 4},
            "dense logical strides changed");
  }
}
void carrier_authority_and_atomic_failure() {
  std::array inputs{tensor("input", "INT8", 288)};
  const std::array outputs{tensor("output", "FP32", 792)};
  auto candidate = stage(288);
  ps::PhysicalBufferStaticSpec physical;
  physical.physical_index = 0;
  physical.source_physical_index = 7;
  physical.size_bytes = 4096;
  physical.source_byte_offset = 512;
  physical.segment_name = "retained-parent";
  physical.address_source = ps::PhysicalAddressSource::RuntimePhysicalBinding;
  candidate.physical_inputs.push_back(physical);
  std::string error;
  require(ps::complete_graph_kernel_contract(&candidate, inputs, outputs, "", "", "", {}, &error),
          error.c_str());
  const auto& retained = candidate.physical_inputs.front();
  require(retained.size_bytes == 4096 && retained.source_physical_index == 7 &&
              retained.source_byte_offset == 512 &&
              retained.address_source == ps::PhysicalAddressSource::RuntimePhysicalBinding,
          "graph completion replaced authoritative upstream carrier coordinates");
  require(candidate.input_bindings.front().src_physical_output_index == 7 &&
              candidate.input_bindings.front().src_physical_byte_offset == 0 &&
              candidate.input_bindings.front().src_physical_size_bytes == retained.size_bytes &&
              candidate.input_bindings.front().source_segment_name == "retained-parent",
          "inferred binding mixed MPK source identity with compiler-owned physical coordinates");
  auto bad = stage(288);
  physical.size_bytes = 128;
  bad.physical_inputs.push_back(physical);
  require(!ps::complete_graph_kernel_contract(&bad, inputs, outputs, "", "", "", {}, &error),
          "undersized authoritative carrier accepted");
  require(!bad.processcvu.canonical_contract && bad.logical_inputs.empty() &&
              bad.logical_outputs.empty() && bad.physical_inputs.front().size_bytes == 128,
          "failed completion partially mutated the stage");
  inputs.front().source_physical_index = -1;
  bad = stage(288);
  require(!ps::complete_graph_kernel_contract(&bad, inputs, outputs, "", "", "", {}, &error),
          "missing upstream physical authority was invented");
}
void shared_offsets_and_conflicts() {
  std::array inputs{tensor("first", "INT8", 198), tensor("second", "INT8", 198)};
  inputs[0].segment_name = inputs[1].segment_name = "parent";
  inputs[1].tensor_index = 1;
  inputs[0].source_byte_offset = inputs[1].source_byte_offset = 128;
  inputs[1].byte_offset = 256;
  const std::array outputs{tensor("output", "FP32", 792)};
  auto candidate = stage(198);
  candidate.processcvu.input_tensors.push_back(candidate.processcvu.input_tensors.front());
  std::string error;
  require(ps::complete_graph_kernel_contract(&candidate, inputs, outputs, "", "", "", {}, &error),
          error.c_str());
  require(candidate.physical_inputs.size() == 1 &&
              candidate.physical_inputs.front().size_bytes == 454 &&
              candidate.logical_inputs[1].byte_offset == 256,
          "shared-parent logical views lost their independent offsets");
  require(candidate.physical_inputs.front().source_byte_offset == 128 &&
              candidate.input_bindings[0].src_physical_byte_offset == 0 &&
              candidate.input_bindings[1].src_physical_byte_offset == 0 &&
              candidate.input_bindings[0].src_physical_size_bytes == 454 &&
              candidate.input_bindings[1].src_physical_size_bytes == 454,
          "inferred routes must bind the shared carrier while logical views retain their offsets");
  inputs[1].source_physical_index = 1;
  auto bad = stage(198);
  bad.processcvu.input_tensors.push_back(bad.processcvu.input_tensors.front());
  require(!ps::complete_graph_kernel_contract(&bad, inputs, outputs, "", "", "", {}, &error),
          "conflicting identities for a shared physical carrier accepted");
  inputs[1].source_physical_index = 0;
  inputs[1].byte_offset = INT64_MAX;
  inputs[1].size_bytes = UINT64_MAX;
  require(!ps::complete_graph_kernel_contract(&bad, inputs, outputs, "", "", "", {}, &error),
          "logical offset plus storage extent overflow accepted");
}
void canonical_dtype_and_name_authority() {
  const std::array inputs{tensor("model-input", "INT8", 198)};
  const std::array outputs{tensor("model-output", "FP32", 792)};
  const std::array<std::string, 1> names{"compiled-output"};
  for (const auto* stale : {"", "stale-alias"}) {
    auto candidate = stage(198);
    auto& policy = candidate.processcvu;
    policy.input_dtype = policy.output_dtype = policy.out_dtype = stale;
    policy.default_input_name = policy.primary_output_name = stale;
    policy.default_output_names = {stale};
    policy.runtime_output_dtype_list = {stale};
    std::string error;
    require(ps::complete_graph_kernel_contract(&candidate, inputs, outputs, "INT8", "FP32",
                                               "compiled-input", names, &error),
            error.c_str());
    const auto& resolved = candidate.processcvu;
    require(resolved.canonical_contract && resolved.input_dtype == "INT8" &&
                resolved.output_dtype == "FP32" && resolved.out_dtype == "FP32" &&
                resolved.runtime_output_dtype_list == std::vector<std::string>{"FP32"},
            "canonical MPK dtype authority was hidden by stale scalar aliases");
    require(resolved.default_input_name == "compiled-input" &&
                resolved.default_output_names == std::vector<std::string>{"compiled-output"} &&
                resolved.primary_output_name == "compiled-output",
            "graph runtime slots/publication names were not resolved from compiled authority");
  }
}

} // namespace
int main() {
  try {
    inferred_extents();
    carrier_authority_and_atomic_failure();
    shared_offsets_and_conflicts();
    canonical_dtype_and_name_authority();
    std::cout << "canonical graph storage/publication contracts PASS\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
