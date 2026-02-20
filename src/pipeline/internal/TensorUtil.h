#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorCore.h"

typedef struct _GstSample GstSample;
typedef struct _GstBuffer GstBuffer;

namespace simaai::neat::pipeline_internal {

std::shared_ptr<simaai::neat::Storage> make_gst_sample_storage(GstSample* sample);

// Get a ref-counted GstBuffer from a tensor holder (e.g. GstSample holder).
// The returned buffer must be unref'd if it is not pushed into appsrc.
GstBuffer* buffer_from_tensor_holder(const std::shared_ptr<void>& holder);

// Drop GstSample holder references (including map_fn) to allow buffers to return to pools.
// Returns true if a holder or map_fn was cleared.
bool drop_tensor_holder(const simaai::neat::Tensor& tensor);

// Map tensor payload for CPU access. Uses device-aware mapping for GstSample-backed
// tensors when available, and falls back to Tensor::map for other cases.
simaai::neat::Mapping map_tensor_view(const simaai::neat::Tensor& tensor,
                                      simaai::neat::MapMode mode);

// Copy a specific GstBuffer memory index into an owned simaai::neat::Tensor.
simaai::neat::Tensor copy_tensor_from_sample_memory(const simaai::neat::Tensor& ref,
                                                    int memory_index, bool keep_holder = true);
std::shared_ptr<void> holder_from_tensor(const simaai::neat::Tensor& tensor);

} // namespace simaai::neat::pipeline_internal
