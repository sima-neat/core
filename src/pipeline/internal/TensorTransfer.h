#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorCore.h"

#include <cstdint>
#include <memory>
#include <vector>

typedef struct _GstSample GstSample;

namespace simaai::neat::pipeline_internal {

simaai::neat::Tensor
transfer_to_device(const simaai::neat::Tensor& src, const simaai::neat::Device& target,
                   const std::vector<simaai::neat::Segment>* required_segments);

simaai::neat::Tensor transfer_to_cpu(const simaai::neat::Tensor& src);

// The storage of a tensor over `sample`, a GstSample whose buffer is one standard DMA-BUF memory,
// placed on `target` with the SiMa target flags `target_flags`: what a TensorMemory::EV74 or MLA
// tensor's storage is. Empty when the sample has no buffer.
std::shared_ptr<simaai::neat::Storage>
make_driver_dmabuf_storage(GstSample* sample, const simaai::neat::Device& target,
                           std::uint64_t target_flags);

} // namespace simaai::neat::pipeline_internal
