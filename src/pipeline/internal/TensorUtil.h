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

namespace simaai::neat {
struct Sample;
}

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
                                      std::vector<simaai::neat::Segment> segments);
// Rebind admitted names to this frame's live carrier extents without admitting
// descriptors, mapping payloads, or acquiring a CPU synchronization epoch.
enum class TensorBufferBindingResult { Current, IncompatibleCarrierCount, Invalid };
TensorBufferBindingResult tensor_buffer_current_segments(
    GstBuffer* buffer, const std::vector<simaai::neat::Segment>& admitted_segments,
    std::vector<simaai::neat::Segment>* current_segments, std::string* err);
TensorIoStats snapshot_tensor_io_stats();
void reset_tensor_io_stats();
void record_tensor_holder_fast_path_hit();
void record_tensor_bundle_projection(std::size_t field_count);
void record_tensor_packed_view_reuse(std::size_t logical_output_count,
                                     std::size_t unique_memory_count, bool materialize_output);
const char* storage_kind_name(simaai::neat::StorageKind kind);

// Get a ref-counted GstBuffer from a tensor holder (e.g. GstSample holder).
// The returned buffer must be unref'd if it is not pushed into appsrc.
GstBuffer* buffer_from_tensor_holder(const std::shared_ptr<void>& holder);

// Drop GstSample holder references (including map_fn) to allow buffers to return to pools.
// Returns true if a holder or map_fn was cleared.
bool drop_tensor_holder(const simaai::neat::Tensor& tensor);

// Map tensor payload for CPU access. Uses device-aware mapping for GstSample-backed
// tensors when available, and falls back to the storage mapping for other cases.
simaai::neat::Mapping map_tensor_view(const simaai::neat::Tensor& tensor,
                                      simaai::neat::MapMode mode);

// A bound, dense numerical rank-zero Tensor describes one dtype element. Empty
// shape alone does not turn an unbound/default Tensor or an explicitly opaque
// carrier into a scalar. Geometry/bounds remain separate admission checks: a
// malformed scalar must fail them rather than fall back to whole-carrier access.
inline bool is_bound_dense_scalar(const simaai::neat::Tensor& tensor) noexcept {
  return tensor.storage && tensor.is_dense() && tensor.shape.empty() && !tensor.semantic.encoded &&
         !tensor.semantic.byte_stream && !tensor.semantic.tess && !tensor.semantic.text;
}

// Internal checked dense READ for shape-bounded consumers. Returns the exact
// reachable tensor span (including positive stride gaps), with byte_offset
// applied once. Invalid/unsupported geometry or insufficient authority returns
// an empty Mapping. Does not change public Tensor::map/view semantics or bypass
// the existing custom/selected-segment mapping contract.
simaai::neat::Mapping map_dense_tensor_read(const simaai::neat::Tensor& tensor);

// Materialize public output storage once per carrier, retaining tensor views and
// metadata but no producer buffer/pool loan. Auto may preserve actual DMA-BUFs
// and certified framework terminal host carriers, selected per tensor span.
void materialize_owned_output(Sample& sample, bool preserve_framework = false);

// True when the tensor's storage is a copy made by materialize_owned_output.
bool tensor_is_owned_copy(const simaai::neat::Tensor& tensor);

// Copy a specific GstBuffer memory index into an owned simaai::neat::Tensor.
simaai::neat::Tensor copy_tensor_from_sample_memory(const simaai::neat::Tensor& ref,
                                                    int memory_index, bool keep_holder = true);
simaai::neat::Tensor tensor_view_from_sample_memory(const simaai::neat::Tensor& ref,
                                                    int memory_index, bool keep_holder = true);
simaai::neat::Tensor tensor_view_from_sample_segment(const simaai::neat::Tensor& ref,
                                                     const std::string& segment_name,
                                                     int fallback_memory_index,
                                                     bool keep_holder = true);
std::shared_ptr<void> holder_from_tensor(const simaai::neat::Tensor& tensor);

} // namespace simaai::neat::pipeline_internal
