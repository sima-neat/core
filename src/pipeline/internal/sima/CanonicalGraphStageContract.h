#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif
#include "pipeline/internal/sima/MpkContract.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include <span>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima {
// Complete absent graph carrier/publication fields from selected, authoritative
// MPK tensors. Existing compiler routes and allocation authority are never
// replaced. Descriptors in stage.processcvu have already been synthesized by
// Core. This is cold model projection, not a prepared Scheduler ABI.
// Failure is atomic: the supplied stage is unchanged.
bool complete_graph_kernel_contract(StageStaticSpec* stage,
                                    std::span<const MpkTensorContract> inputs,
                                    std::span<const MpkTensorContract> outputs,
                                    std::string_view canonical_input_dtype,
                                    std::string_view canonical_output_dtype,
                                    std::string_view input_slot,
                                    std::span<const std::string> output_slots, std::string* error);
} // namespace simaai::neat::pipeline_internal::sima
