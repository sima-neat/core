#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorTypes.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {
struct Sample;
struct Tensor;

namespace pipeline_internal {

// Opt-in policy for an ordered public tensor boundary. Names assert roles when
// supplied; they never reorder tensors or override physical routing metadata.
struct PublicInputTensorSpec {
  std::string name;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<std::int64_t> shape;
};

struct OrderedPublicInputContract {
  std::vector<PublicInputTensorSpec> inputs;
};

void validate_ordered_public_input(const OrderedPublicInputContract& contract,
                                   const Sample& sample);
void validate_ordered_public_input(const OrderedPublicInputContract& contract,
                                   const Tensor& tensor);

} // namespace pipeline_internal
} // namespace simaai::neat
