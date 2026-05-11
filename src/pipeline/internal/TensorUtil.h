#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TensorCore.h"

#include <cstddef>
#include <cstdint>
#include <string>

typedef struct _GstSample GstSample;
typedef struct _GstBuffer GstBuffer;

namespace simaai::neat::pipeline_internal {

struct TensorIoStats {
  std::uint64_t tensor_copy_count = 0;
  std::uint64_t tensor_copy_bytes = 0;
  std::uint64_t tensor_view_count = 0;
  std::uint64_t gst_memory_map_calls = 0;
  std::uint64_t holder_fast_path_hits = 0;
  std::uint64_t bundle_projection_count = 0;
  std::uint64_t packed_view_reuse_hits = 0;
  std::uint64_t packed_view_reuse_opportunities = 0;

  double packed_view_reuse_ratio() const {
    if (packed_view_reuse_opportunities == 0) {
      return 1.0;
    }
    return static_cast<double>(packed_view_reuse_hits) /
           static_cast<double>(packed_view_reuse_opportunities);
  }
};

std::shared_ptr<simaai::neat::Storage> make_gst_sample_storage(GstSample* sample);
std::shared_ptr<simaai::neat::Storage>
make_gst_sample_storage_with_segments(GstSample* sample,
                                      const std::vector<simaai::neat::Segment>& segments);
TensorIoStats snapshot_tensor_io_stats();
void reset_tensor_io_stats();
void record_tensor_holder_fast_path_hit();
void record_tensor_bundle_projection(std::size_t field_count);
void record_tensor_packed_view_reuse(std::size_t logical_output_count,
                                     std::size_t unique_memory_count,
                                     bool materialize_output);
const char* storage_kind_name(simaai::neat::StorageKind kind);

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
simaai::neat::Tensor tensor_view_from_sample_memory(const simaai::neat::Tensor& ref,
                                                    int memory_index, bool keep_holder = true);
std::shared_ptr<void> holder_from_tensor(const simaai::neat::Tensor& tensor);

// Copy `bytes` from `src` into the simaai segment memory pointed to by `segment`
// (an opaque simaai_memory_t*). Maps the segment, memcpys, flushes cache, unmaps.
// Used by the bundled-input multi-IO path to materialize user CPU tensors into
// device-accessible segments addressed via the SimaaiSegmentAllocator.
bool copy_into_simaai_segment_memory(void* segment, const void* src, std::size_t bytes,
                                     std::string* err);

} // namespace simaai::neat::pipeline_internal
