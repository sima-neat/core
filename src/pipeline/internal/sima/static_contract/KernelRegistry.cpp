#define SIMA_NEAT_INTERNAL 1
#include "pipeline/internal/sima/static_contract/KernelRegistry.h"

#include <array>
#include <limits>

namespace simaai::neat::pipeline_internal::sima::static_contract {
namespace {

constexpr std::size_t kMany = std::numeric_limits<std::size_t>::max();

// `cast_transform` is the frozen legacy spelling. `cast` is a separately
// registered newer spelling; lookup never derives one from the other.
constexpr std::array<KernelDescriptor, 26> kRegistry = {{
    {"2.0.0", "EV74", "cast_transform", OpKind::Cast, 1, 1, 1, 1},
    {"2.0.0", "EV74", "cast", OpKind::Cast, 1, 1, 1, 1},
    {"2.0.0", "EV74", "quantization_transform", OpKind::Quantize, 1, 1, 1, 1},
    {"2.0.0", "EV74", "tessellation_transform", OpKind::Tessellate, 1, 1, 1, 1},
    {"2.0.0", "EV74", "pack_transform", OpKind::Pack, 2, kMany, 1, 1},
    {"2.0.0", "MLA", "", OpKind::Mla, 1, kMany, 1, kMany},
    {"2.0.0", "EV74", "unpack_transform", OpKind::Unpack, 1, 1, 1, kMany},
    {"2.0.0", "EV74", "slice_transform", OpKind::Slice, 1, 1, 1, 1},
    {"2.0.0", "EV74", "reshape_transform", OpKind::Reshape, 1, 1, 1, 1},
    {"2.0.0", "EV74", "detessellation_transform", OpKind::Detessellate, 1, 1, 1, 1},
    {"2.0.0", "EV74", "dequantization_transform", OpKind::Dequantize, 1, 1, 1, 1},
    {"2.0.0", "EV74", "pass_through", OpKind::PassThrough, 1, kMany, 1, kMany},
    {"2.1.0", "EV74", "cast_transform", OpKind::Cast, 1, 1, 1, 1},
    {"2.1.0", "EV74", "cast", OpKind::Cast, 1, 1, 1, 1},
    {"2.1.0", "EV74", "quantization_transform", OpKind::Quantize, 1, 1, 1, 1},
    {"2.1.0", "EV74", "tessellation_transform", OpKind::Tessellate, 1, 1, 1, 1},
    {"2.1.0", "EV74", "pack_transform", OpKind::Pack, 2, kMany, 1, 1},
    {"2.1.0", "MLA", "", OpKind::Mla, 1, kMany, 1, kMany},
    {"2.1.0", "EV74", "unpack_transform", OpKind::Unpack, 1, 1, 1, kMany},
    {"2.1.0", "EV74", "slice_transform", OpKind::Slice, 1, 1, 1, 1},
    {"2.1.0", "EV74", "reshape_transform", OpKind::Reshape, 1, 1, 1, 1},
    {"2.1.0", "EV74", "batch_flatten_transform", OpKind::Reshape, 1, 1, 1, 1},
    {"2.1.0", "EV74", "detessellation_transform", OpKind::Detessellate, 1, 1, 1, 1},
    {"2.1.0", "EV74", "dequantization_transform", OpKind::Dequantize, 1, 1, 1, 1},
    {"2.1.0", "A65", "", OpKind::HostTvm, 1, kMany, 1, kMany},
    {"2.1.0", "EV74", "pass_through", OpKind::PassThrough, 1, kMany, 1, kMany},
}};

} // namespace

std::optional<KernelDescriptor> lookup_exact_kernel(const std::string_view contract_version,
                                                    const std::string_view processor,
                                                    const std::string_view kernel) noexcept {
  for (const auto& descriptor : kRegistry) {
    if (descriptor.contract_version == contract_version && descriptor.processor == processor &&
        descriptor.kernel == kernel) {
      return descriptor;
    }
  }
  return std::nullopt;
}

bool exact_kernel_arity_is_valid(const KernelDescriptor& descriptor, const std::size_t input_count,
                                 const std::size_t output_count) noexcept {
  if (input_count < descriptor.minimum_inputs || input_count > descriptor.maximum_inputs ||
      output_count < descriptor.minimum_outputs || output_count > descriptor.maximum_outputs) {
    return false;
  }
  if ((descriptor.kind == OpKind::PassThrough && input_count != output_count) ||
      (descriptor.kind == OpKind::Unpack && input_count != 1U)) {
    return false;
  }
  return true;
}

} // namespace simaai::neat::pipeline_internal::sima::static_contract
