#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"

#include <string>

namespace simaai::neat::pipeline_internal {

// These are optional source labels, not consumer input identities or physical
// carrier admission. Inspecting a label must never compile a stage or map data.
inline std::string tensor_source_segment_name(const Tensor& tensor) {
  if (!tensor.route.segment_name.empty()) {
    return tensor.route.segment_name;
  }
  if (tensor.storage && !tensor.storage->sima_segments.empty()) {
    int memory_index = tensor.route.memory_index;
    if (memory_index < 0) {
      memory_index = tensor.route.physical_index;
    }
    std::size_t segment_index = 0U;
    if (memory_index >= 0 &&
        static_cast<std::size_t>(memory_index) < tensor.storage->sima_segments.size()) {
      segment_index = static_cast<std::size_t>(memory_index);
    }
    return tensor.storage->sima_segments[segment_index].name;
  }
  return {};
}

inline std::string sample_source_segment_name(const Sample& sample) {
  if (sample_has_tensor_list(sample)) {
    const std::string tensor_name = tensor_source_segment_name(sample.tensors.front());
    if (!tensor_name.empty()) {
      return tensor_name;
    }
  }
  if (!sample.segment_name.empty()) {
    return sample.segment_name;
  }
  if (sample.kind == SampleKind::Bundle && !sample.fields.empty()) {
    return sample_source_segment_name(sample.fields.front());
  }
  return {};
}

inline std::string tensor_wire_source_segment_name(const Tensor& tensor,
                                                   const std::string& wire_name) {
  const std::string tensor_name = tensor_source_segment_name(tensor);
  return tensor_name.empty() ? wire_name : tensor_name;
}

} // namespace simaai::neat::pipeline_internal
