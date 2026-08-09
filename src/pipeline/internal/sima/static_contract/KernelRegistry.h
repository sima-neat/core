#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/internal/sima/static_contract/ModelExecutionPlan.h"

#include <cstddef>
#include <optional>
#include <string_view>

namespace simaai::neat::pipeline_internal::sima::static_contract {

struct KernelDescriptor {
  std::string_view contract_version;
  std::string_view processor;
  std::string_view kernel;
  OpKind kind = OpKind::PassThrough;
  std::size_t minimum_inputs = 0;
  std::size_t maximum_inputs = 0;
  std::size_t minimum_outputs = 0;
  std::size_t maximum_outputs = 0;
};

// Exact, versioned lookup only. There is deliberately no case folding,
// substring matching, suffix stripping, processor aliasing, or default entry.
std::optional<KernelDescriptor> lookup_exact_kernel(std::string_view contract_version,
                                                    std::string_view processor,
                                                    std::string_view kernel) noexcept;

bool exact_kernel_arity_is_valid(const KernelDescriptor& descriptor, std::size_t input_count,
                                 std::size_t output_count) noexcept;

} // namespace simaai::neat::pipeline_internal::sima::static_contract
