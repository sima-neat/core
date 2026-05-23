#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "nodes/io/Input.h"

#include <memory>
#include <string>

typedef struct _GstBuffer GstBuffer;

namespace simaai::neat::pipeline_internal {

std::shared_ptr<void> make_sample_holder_from_bundle(const Sample& bundle, std::string* err,
                                                     bool allow_zero_copy = true);
Sample canonicalize_tensor_transport_sample(const Sample& sample);
Sample sample_from_tensors_for_input(const TensorList& tensors, const InputOptions& opt);
Sample collapse_single_tensor_sample(Sample sample);
void attach_tensor_set_meta_from_tensors(GstBuffer* buffer, const TensorList& tensors);

bool build_bundled_input_gst_buffer(const TensorList& tensors, GstBuffer** out_buffer,
                                    std::string* err);

} // namespace simaai::neat::pipeline_internal
