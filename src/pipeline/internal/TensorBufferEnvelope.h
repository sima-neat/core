#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/SessionOptions.h"
#include "pipeline/internal/OutputTensorOverride.h"

#include "gst/SimaTensorSetMetaAbi.h"
#include "gst/SimaPluginStaticManifestAbi.h"
#include "gstsimaaitensorbuffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

typedef struct _GstBuffer GstBuffer;
typedef struct _GstCaps GstCaps;
typedef struct _GstSample GstSample;

namespace simaai::neat::pipeline_internal {

struct TensorBufferQuantDescriptor {
  int granularity = 0;
  int axis = -1;
  std::vector<double> scales;
  std::vector<std::int64_t> zero_points;
};

struct TensorBufferTensorDescriptor {
  int logical_index = -1;
  int physical_index = -1;
  int backend_output_index = -1;
  int route_slot = -1;
  int memory_index = -1;
  std::string logical_name;
  std::string backend_name;
  std::string segment_name;
  std::int64_t byte_offset = 0;
  std::uint64_t size_bytes = 0;
  int dtype = SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1;
  int layout = SIMA_TENSOR_SET_LAYOUT_UNKNOWN_V1;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> stride_bytes;
  std::optional<TensorBufferQuantDescriptor> quant;
};

struct TensorBufferView {
  std::shared_ptr<void> holder;
  GstSample* sample = nullptr;
  GstBuffer* buffer = nullptr;
  GstCaps* caps = nullptr;
  std::string stage_key;
  std::vector<TensorBufferTensorDescriptor> tensors;
};

bool tensor_buffer_view_from_tensors(const TensorList& tensors, TensorBufferView* out,
                                     std::string* err);
bool tensor_buffer_view_from_sample(const Sample& sample, TensorBufferView* out, std::string* err);
bool tensor_buffer_descriptor_from_sample(GstSample* sample, TensorBufferView* out,
                                          std::string* err);

std::shared_ptr<void> tensor_to_gst_envelope_holder(const Tensor& tensor, std::string* err);
std::shared_ptr<void> tensor_list_to_gst_envelope_holder(const TensorList& tensors,
                                                         const Sample& envelope_meta,
                                                         std::string* err,
                                                         bool allow_zero_copy = true);
std::shared_ptr<void> sample_to_gst_envelope_holder(const Sample& sample, std::string* err,
                                                    bool allow_zero_copy = true);

} // namespace simaai::neat::pipeline_internal

namespace simaai::neat {

Sample sample_from_gst_envelope(GstSample* sample, const char* where, bool copy_output,
                                const std::optional<OutputTensorOverride>* override_opt);

} // namespace simaai::neat
