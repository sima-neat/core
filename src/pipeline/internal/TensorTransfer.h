#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorCore.h"

#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

simaai::neat::Tensor transfer_to_device(const simaai::neat::Tensor& src,
                                        const simaai::neat::Device& target,
                                        const std::vector<simaai::neat::Segment>* required_segments,
                                        const std::vector<std::string>* required_segment_names);

simaai::neat::Tensor transfer_to_cpu(const simaai::neat::Tensor& src);

} // namespace simaai::neat::pipeline_internal
