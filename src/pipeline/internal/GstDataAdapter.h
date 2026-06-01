#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorCore.h"
#include "pipeline/GraphOptions.h"

#include <cstddef>
#include <cstdint>
#include <string>

typedef struct _GstBuffer GstBuffer;
typedef struct _GstCaps GstCaps;
typedef struct _GstSample GstSample;

namespace simaai::neat {
struct SampleSpec;
} // namespace simaai::neat

namespace simaai::neat::pipeline_internal {

struct GstBufferBuildPolicy {
  bool allow_zero_copy = true;
  bool require_video_meta = false;
  bool allow_appsrc_pool = false;
  bool require_contiguous = false;
  bool allow_device_memory = false;
};

std::size_t tensor_plane_bytes_tight(const simaai::neat::Plane& plane, TensorDType dtype);
std::size_t tensor_bytes_tight(const simaai::neat::Tensor& input);

bool copy_tensor_payload_to(const simaai::neat::Tensor& tensor, uint8_t* dst, std::size_t dst_bytes,
                            std::string* err);

bool canonicalize_sample_spec(SampleSpec* spec, std::string* err);

bool derive_field_spec(const Sample& field, SampleSpec* out, std::string* err);

GstBuffer* build_gst_buffer_from_tensor(const simaai::neat::Tensor& tensor, const SampleSpec& spec,
                                        const GstBufferBuildPolicy& policy, std::string* err);

GstSample* build_gst_sample_from_tensor(const simaai::neat::Tensor& tensor, const SampleSpec& spec,
                                        const GstBufferBuildPolicy& policy, GstCaps* override_caps,
                                        std::string* err);

GstBuffer* build_copy_buffer_from_tensor(const simaai::neat::Tensor& tensor, const SampleSpec& spec,
                                         std::string* err);

GstBuffer* buffer_from_holder_if_gstsample(const simaai::neat::Tensor& tensor, std::string* err);

void copy_custom_meta(GstBuffer* dst, GstBuffer* src, const char* meta_name);

bool validate_buffer_video_meta(GstBuffer* buffer, const SampleSpec& spec, std::string* err);

bool resolve_encoded_payload_bytes(const simaai::neat::Tensor& tensor, const SampleSpec& spec,
                                   std::size_t* out_bytes, std::string* err);

bool attach_video_meta(GstBuffer** buffer, const SampleSpec& spec, std::string* err);
bool apply_tensor_size(GstBuffer** buffer, const SampleSpec& spec, std::string* err);

bool wrap_cpu_dense_zero_copy(const simaai::neat::Tensor& tensor, GstBuffer** out,
                              std::string* err);

} // namespace simaai::neat::pipeline_internal
