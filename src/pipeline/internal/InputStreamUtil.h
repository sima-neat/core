#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pipeline/TensorTypes.h"
#include "pipeline/TensorCore.h"

#include <gst/gst.h>

namespace cv {
class Mat;
} // namespace cv

namespace simaai::neat {

struct InputOptions;
struct Sample;
struct SampleSpec;

struct Status {
  std::string message;
};

template <typename T, typename E> class Expected {
public:
  Expected(const T& value) : value_(value) {}
  Expected(T&& value) : value_(std::move(value)) {}
  Expected(const E& error) : error_(error) {}
  Expected(E&& error) : error_(std::move(error)) {}

  explicit operator bool() const {
    return value_.has_value();
  }
  bool ok() const {
    return value_.has_value();
  }

  const T& value() const {
    return *value_;
  }
  T& value() {
    return *value_;
  }
  const E& error() const {
    return *error_;
  }
  E& error() {
    return *error_;
  }

private:
  std::optional<T> value_;
  std::optional<E> error_;
};

enum class SampleMediaKind {
  RawVideo,
  Tensor,
  Encoded,
};

struct PlaneInfo {
  simaai::neat::PlaneRole role = simaai::neat::PlaneRole::Unknown;
  int width = -1;
  int height = -1;
  int64_t stride_bytes = 0;
  int64_t offset_bytes = 0;
  size_t size_bytes = 0;
};

struct CapKey {
  SampleMediaKind kind = SampleMediaKind::Tensor;
  std::string media_type;
  std::string format;
  int width = -1;
  int height = -1;
  int fps_n = 0;
  int fps_d = 1;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<int64_t> shape;
  std::size_t caps_hash = 0;

  bool operator==(const CapKey& other) const {
    if (kind != other.kind)
      return false;
    switch (kind) {
    case SampleMediaKind::RawVideo:
      return media_type == other.media_type && format == other.format && width == other.width &&
             height == other.height && fps_n == other.fps_n && fps_d == other.fps_d;
    case SampleMediaKind::Tensor:
      return dtype == other.dtype && shape == other.shape;
    case SampleMediaKind::Encoded:
      return caps_hash == other.caps_hash;
    }
    return false;
  }
  bool operator!=(const CapKey& other) const {
    return !(*this == other);
  }

  std::string to_string() const;
};

struct CapKeyHash {
  std::size_t operator()(const CapKey& key) const;
};

struct TensorCompatDims {
  int width = -1;
  int height = -1;
  int depth = -1;
};

std::vector<int64_t> tensor_shape_from_compat_dims(int width, int height, int depth,
                                                   TensorLayout layout);
TensorCompatDims tensor_compat_dims_from_shape(const std::vector<int64_t>& shape,
                                               TensorLayout layout);

CapKey capkey_from_spec(const SampleSpec& spec);
std::string caps_string_from_spec(const SampleSpec& spec);

struct SampleSpec {
  SampleMediaKind kind = SampleMediaKind::Tensor;
  std::string media_type;
  std::string format;
  TensorDType dtype = TensorDType::UInt8;
  TensorLayout layout = TensorLayout::Unknown;
  std::vector<int64_t> shape;
  // Raw-video compatibility only. Tensor truth is carried in `shape`.
  int width = -1;
  int height = -1;
  int depth = -1;
  bool tensor_envelope_transport = false;
  int fps_n = 0;
  int fps_d = 1;
  std::vector<PlaneInfo> planes;
  size_t required_bytes_actual = 0;
  CapKey caps_key;
  std::string caps_string;
};

Expected<SampleSpec, Status> derive_sample_spec_or_error(const Sample& sample);
SampleSpec derive_sample_spec_or_throw(const Sample& sample);
SampleSpec derive_tensor_spec_or_throw(const simaai::neat::Tensor& input, const InputOptions& opt,
                                       const char* where);
simaai::neat::Tensor tensor_from_cv_mat(const cv::Mat& mat, const InputOptions& opt,
                                        const char* where);

struct InputBufferPoolGuard {
  std::unique_ptr<GstBufferPool, void (*)(GstBufferPool*)> pool{nullptr, +[](GstBufferPool*) {}};
};

struct StreamIdOverride {
  std::optional<std::string> value;
  StreamIdOverride() = default;
  StreamIdOverride(std::nullopt_t) : value(std::nullopt) {}
  StreamIdOverride(std::optional<std::string> v) : value(std::move(v)) {}
};

struct BufferNameOverride {
  std::optional<std::string> value;
  BufferNameOverride() = default;
  BufferNameOverride(std::nullopt_t) : value(std::nullopt) {}
  BufferNameOverride(std::optional<std::string> v) : value(std::move(v)) {}
};

struct SampleTimingOverrides {
  std::optional<int64_t> frame_id;
  std::optional<uint64_t> pts_ns;
  std::optional<uint64_t> dts_ns;
  std::optional<uint64_t> duration_ns;

  bool empty() const {
    return !frame_id.has_value() && !pts_ns.has_value() && !dts_ns.has_value() &&
           !duration_ns.has_value();
  }
};

SampleTimingOverrides sample_timing_overrides_from_sample(const Sample& sample);

GstCaps* caps_from_spec(const SampleSpec& spec);
GstBuffer* allocate_input_buffer(size_t bytes, const InputOptions& opt,
                                 InputBufferPoolGuard& guard);
int64_t next_input_frame_id();
bool maybe_add_simaai_meta(GstBuffer* buffer, int64_t frame_id, const InputOptions& opt);
void maybe_update_simaai_meta_name(GstBuffer* buffer, const std::string& name);
void dump_buffer_memories(GstBuffer* buffer, const char* label);
void dump_sima_meta(GstBuffer* buffer, const char* label);
void dump_sima_meta_full(GstBuffer* buffer, const char* label);
void debug_input_buffer_release(GstBuffer* buffer, const char* where);
void track_input_pool_acquire(GstBufferPool* pool, GstBuffer* buffer, size_t bytes,
                              const char* where, double wait_ms, bool ok);
void track_input_pool_release(GstBuffer* buffer, const char* where);
GstBuffer*
attach_simaai_meta_inplace(GstBuffer* buffer, const InputOptions& opt, InputBufferPoolGuard& guard,
                           const char* label,
                           const std::optional<int64_t>& frame_id_override = std::nullopt,
                           const StreamIdOverride& stream_id_override = {},
                           const BufferNameOverride& buffer_name_override = {});
bool update_simaai_meta_fields(
    GstBuffer* buffer, const std::optional<int64_t>& frame_id_override,
    const std::optional<int64_t>& input_seq_override,
    const std::optional<int64_t>& orig_input_seq_override,
    const std::optional<std::string>& stream_id_override,
    const std::optional<std::string>& buffer_name_override,
    const std::optional<uint64_t>& timestamp_override = std::nullopt,
    const std::optional<std::string>& origin_stage_id_override = std::nullopt,
    const std::optional<int>& origin_output_slot_override = std::nullopt);
bool write_sample_timing_to_gst_buffer(GstBuffer* buffer, const SampleTimingOverrides& timing);
void restore_sample_timing_from_gst_buffer(GstBuffer* buffer, Sample* out);
bool write_simaai_preprocess_meta(GstBuffer* buffer, const PreprocessRuntimeMeta& meta);
// Merge `axis_perm` (and only that field) onto an existing GstSimaMeta on
// `buffer`. Adds GstSimaMeta if absent. The plugin path no longer writes
// preproc_axis_perm because that field is framework-owned semantic
// (the user's resolved layout_convert.perm); the framework merges it onto
// the buffer's existing meta at handoff time without overwriting any
// plugin-authored geometry/affine/flag fields.
bool merge_simaai_preprocess_axis_perm(GstBuffer* buffer, const std::vector<int>& axis_perm);
std::optional<PreprocessRuntimeMeta> read_simaai_preprocess_meta(GstBuffer* buffer);
bool has_simaai_preprocess_meta(GstBuffer* buffer);
bool copy_simaai_preprocess_meta(GstBuffer* dst, GstBuffer* src, std::string* err = nullptr);
std::optional<std::string>
validate_simaai_preprocess_meta_required_fields(GstBuffer* buffer,
                                                const std::vector<std::string>& required_fields,
                                                PreprocessRuntimeMeta* out_meta = nullptr);
bool apply_simaai_preprocess_meta_template(GstBuffer* buffer, const InputOptions& opt,
                                           int input_width, int input_height);

} // namespace simaai::neat
