#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/GraphOptions.h"
#include "nodes/io/Input.h"

#include <memory>
#include <functional>
#include <string>

typedef struct _GstBuffer GstBuffer;

namespace simaai::neat::pipeline_internal {

class HolderLoanGate;
using HolderLoanGatePtr = std::shared_ptr<HolderLoanGate>;

std::shared_ptr<void> make_sample_holder_from_bundle(const Sample& bundle, std::string* err,
                                                     bool allow_zero_copy = true);
Sample canonicalize_tensor_transport_sample(const Sample& sample);
Sample sample_from_tensors_for_input(const TensorList& tensors, const InputOptions& opt);
Sample collapse_single_tensor_sample(Sample sample);
void attach_tensor_set_meta_from_tensors(GstBuffer* buffer, const TensorList& tensors);

bool sample_has_device_gstsample_producer_lifetime(const Sample& sample,
                                                   bool require_expired = false);
std::string cross_run_zero_copy_sample_error(const char* where);
bool sample_has_device_gstsample_holder(const Sample& sample);
void attach_holder_release_to_sample(const Sample& sample, std::function<void()> on_release);
void mark_sample_producer_stream_lifetime(Sample& sample, std::shared_ptr<void> lifetime_token);
bool attach_zero_copy_loan_to_sample(const Sample& sample, const HolderLoanGatePtr& gate,
                                     std::string* err = nullptr);
bool sample_has_transferable_zero_copy_loan(const Sample& sample, std::string* reason = nullptr);
void attach_zero_copy_loans_to_gst_buffer(GstBuffer* buffer, const Sample& sample);
void attach_zero_copy_loans_from_holder_to_gst_buffer(GstBuffer* buffer,
                                                      const std::shared_ptr<void>& holder);
int count_distinct_device_gstsample_holders(const Sample& sample);

bool build_bundled_input_gst_buffer(const TensorList& tensors, GstBuffer** out_buffer,
                                    std::string* err);

} // namespace simaai::neat::pipeline_internal
