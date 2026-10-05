#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorCore.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

typedef struct _GstBuffer GstBuffer;

namespace simaai::neat::pipeline_internal {

struct TransferPoolStats {
  std::size_t hits = 0;
  std::size_t misses = 0;
  std::size_t entries = 0;
};

TransferPoolStats tensor_transfer_pool_stats();

GstBuffer* acquire_segment_pool_buffer(std::uint64_t target_flags, std::uint64_t mem_flags,
                                       const std::vector<simaai::neat::Segment>& segments,
                                       std::string* err);

simaai::neat::Tensor transfer_to_device(const simaai::neat::Tensor& src,
                                        const simaai::neat::Device& target,
                                        const std::vector<simaai::neat::Segment>* required_segments,
                                        const std::vector<std::string>* required_segment_names);

simaai::neat::Tensor transfer_to_cpu(const simaai::neat::Tensor& src);

} // namespace simaai::neat::pipeline_internal
