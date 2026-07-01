#include "InputStreamInternal.h"

#include "gst/SimaTensorSetMetaAbi.h"
#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/CapsBridge.h"

#include <unordered_set>

namespace simaai::neat {
bool inputstream_restore_preproc_buffer_meta_enabled();
bool inputstream_fast_static_decode_enabled();
bool inputstream_fast_minimal_output_meta_enabled();
bool inputstream_fast_static_preprocess_meta_enabled();

static std::string image_format_name(simaai::neat::ImageSpec::PixelFormat fmt) {
  switch (fmt) {
  case simaai::neat::ImageSpec::PixelFormat::RGB:
    return "RGB";
  case simaai::neat::ImageSpec::PixelFormat::BGR:
    return "BGR";
  case simaai::neat::ImageSpec::PixelFormat::GRAY8:
    return "GRAY8";
  case simaai::neat::ImageSpec::PixelFormat::NV12:
    return "NV12";
  case simaai::neat::ImageSpec::PixelFormat::I420:
    return "I420";
  case simaai::neat::ImageSpec::PixelFormat::UNKNOWN:
    return "";
  }
  return "";
}

static std::string tensor_format_name(const simaai::neat::Tensor& t) {
  if (t.semantic.image.has_value()) {
    const std::string fmt = image_format_name(t.semantic.image->format);
    if (!fmt.empty())
      return fmt;
  }
  if (t.semantic.tess.has_value()) {
    if (!t.semantic.tess->format.empty())
      return t.semantic.tess->format;
  }
  return "";
}

void maybe_restore_cached_preprocess_meta(InputStream::State& st, GstSample* sample) {
  const bool restore_buffer_meta = inputstream_restore_preproc_buffer_meta_enabled();
  if (!restore_buffer_meta) {
    return;
  }
  const bool debug = pipeline_internal::env_bool("SIMA_PREPROC_META_TRACE", false);
  if (!sample) {
    if (debug) {
      std::fprintf(stderr, "[PREPROC_META_TRACE] pull-buffer restore skip=no_sample\n");
    }
    return;
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    if (debug) {
      std::fprintf(stderr, "[PREPROC_META_TRACE] pull-buffer restore skip=no_buffer\n");
    }
    return;
  }

  if (st.src_opt.preprocess_meta.has_value() && !st.src_opt.preprocess_meta->axis_perm.empty()) {
    merge_simaai_preprocess_axis_perm(buffer, st.src_opt.preprocess_meta->axis_perm);
  }

  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    if (debug) {
      std::fprintf(stderr,
                   "[PREPROC_META_TRACE] pull-buffer restore skip=no_sima_meta has_preproc=%d\n",
                   has_simaai_preprocess_meta(buffer) ? 1 : 0);
    }
    return;
  }

  gint64 orig_input_seq = -1;
  gint64 input_seq = -1;
  const bool has_orig_input_seq =
      (gst_structure_get_int64(s, "orig-input-seq", &orig_input_seq) == TRUE);
  const bool has_input_seq = (gst_structure_get_int64(s, "input-seq", &input_seq) == TRUE);
  const int64_t key =
      (has_orig_input_seq && orig_input_seq >= 0)
          ? static_cast<int64_t>(orig_input_seq)
          : ((has_input_seq && input_seq >= 0) ? static_cast<int64_t>(input_seq) : -1);
  if (key < 0) {
    if (debug) {
      std::fprintf(stderr,
                   "[PREPROC_META_TRACE] pull-buffer restore skip=no_key has_input=%d input=%lld "
                   "has_orig=%d orig=%lld cache_size=%zu\n",
                   has_input_seq ? 1 : 0, static_cast<long long>(input_seq),
                   has_orig_input_seq ? 1 : 0, static_cast<long long>(orig_input_seq),
                   st.preprocess_meta_by_input_seq.size());
    }
    return;
  }

  std::lock_guard<std::mutex> lock(st.preprocess_meta_mu);
  const auto it = st.preprocess_meta_by_input_seq.find(key);
  if (it == st.preprocess_meta_by_input_seq.end()) {
    if (debug) {
      std::fprintf(
          stderr,
          "[PREPROC_META_TRACE] pull-buffer restore skip=cache_miss key=%lld cache_size=%zu\n",
          static_cast<long long>(key), st.preprocess_meta_by_input_seq.size());
    }
    return;
  }

  (void)write_simaai_preprocess_meta(buffer, it->second);
  if (debug) {
    std::fprintf(
        stderr,
        "[PREPROC_META_TRACE] pull-buffer restore wrote key=%lld orig=%dx%d resized=%dx%d\n",
        static_cast<long long>(key), it->second.original_width, it->second.original_height,
        it->second.resized_width, it->second.resized_height);
  }
}

bool sample_already_has_preprocess_semantic(const Sample& sample) {
  if (sample_has_tensor_list(sample) && !sample.tensors.empty()) {
    return sample.tensors.front().semantic.preprocess.has_value();
  }
  if (sample.tensor.has_value()) {
    return sample.tensor->semantic.preprocess.has_value();
  }
  if (sample_is_multi_output(sample)) {
    for (const auto& field : sample.fields) {
      if (sample_already_has_preprocess_semantic(field)) {
        return true;
      }
    }
  }
  return false;
}

void maybe_restore_cached_preprocess_meta_on_sample(InputStream::State& st, Sample* sample) {
  const bool debug = pipeline_internal::env_bool("SIMA_PREPROC_META_TRACE", false);
  if (!sample) {
    if (debug) {
      std::fprintf(stderr, "[PREPROC_META_TRACE] pull-sample restore skip=no_sample\n");
    }
    return;
  }
  if (inputstream_fast_static_preprocess_meta_enabled() &&
      sample_already_has_preprocess_semantic(*sample)) {
    return;
  }
  const int64_t key = sample->orig_input_seq >= 0
                          ? sample->orig_input_seq
                          : (sample->input_seq >= 0 ? sample->input_seq : -1);
  if (key < 0) {
    if (debug) {
      std::fprintf(stderr,
                   "[PREPROC_META_TRACE] pull-sample restore skip=no_key sample_input=%lld "
                   "sample_orig=%lld\n",
                   static_cast<long long>(sample->input_seq),
                   static_cast<long long>(sample->orig_input_seq));
    }
    return;
  }

  std::optional<PreprocessRuntimeMeta> cached;
  {
    std::lock_guard<std::mutex> lock(st.preprocess_meta_mu);
    const auto it = st.preprocess_meta_by_input_seq.find(key);
    if (it == st.preprocess_meta_by_input_seq.end()) {
      if (debug) {
        std::fprintf(
            stderr,
            "[PREPROC_META_TRACE] pull-sample restore skip=cache_miss key=%lld cache_size=%zu\n",
            static_cast<long long>(key), st.preprocess_meta_by_input_seq.size());
      }
      return;
    }
    cached = it->second;
  }
  if (debug) {
    std::fprintf(
        stderr,
        "[PREPROC_META_TRACE] pull-sample restore found key=%lld orig=%dx%d resized=%dx%d\n",
        static_cast<long long>(key), cached->original_width, cached->original_height,
        cached->resized_width, cached->resized_height);
  }

  const bool restore_buffer_meta = inputstream_restore_preproc_buffer_meta_enabled();
  auto apply_to_tensor = [&](Tensor& tensor) {
    if (!tensor.semantic.preprocess.has_value()) {
      tensor.semantic.preprocess = cached;
    }
    if (!restore_buffer_meta) {
      return;
    }

    const std::shared_ptr<void> holder = pipeline_internal::holder_from_tensor(tensor);
    if (!holder) {
      return;
    }
    GstBuffer* buffer = pipeline_internal::buffer_from_tensor_holder(holder);
    if (!buffer) {
      return;
    }
    if (has_simaai_preprocess_meta(buffer)) {
      gst_buffer_unref(buffer);
      return;
    }
    if (write_simaai_preprocess_meta(buffer, *cached)) {
      gst_buffer_unref(buffer);
      return;
    }
    gst_buffer_unref(buffer);

    if (!tensor.storage || tensor.storage->kind != simaai::neat::StorageKind::GstSample) {
      return;
    }
    auto* sample = static_cast<GstSample*>(holder.get());
    if (!sample || !GST_IS_SAMPLE(sample)) {
      return;
    }
    GstBuffer* source_buffer = gst_sample_get_buffer(sample);
    if (!source_buffer) {
      return;
    }
    GstBuffer* cloned_buffer = gst_buffer_copy_deep(source_buffer);
    if (!cloned_buffer) {
      return;
    }
    if (!write_simaai_preprocess_meta(cloned_buffer, *cached)) {
      gst_buffer_unref(cloned_buffer);
      return;
    }
    GstCaps* caps = gst_sample_get_caps(sample);
    GstSample* cloned_sample =
        gst_sample_new(cloned_buffer, caps ? gst_caps_ref(caps) : nullptr, nullptr, nullptr);
    gst_buffer_unref(cloned_buffer);
    if (!cloned_sample) {
      return;
    }

    Tensor replaced = simaai::neat::from_gst_sample(cloned_sample);
    gst_sample_unref(cloned_sample);
    replaced.route = tensor.route;
    replaced.semantic = tensor.semantic;
    replaced.dtype = tensor.dtype;
    replaced.layout = tensor.layout;
    replaced.shape = tensor.shape;
    replaced.strides_bytes = tensor.strides_bytes;
    replaced.byte_offset = tensor.byte_offset;
    replaced.planes = tensor.planes;
    replaced.read_only = tensor.read_only;
    tensor = std::move(replaced);
  };

  if (sample_has_tensor_list(*sample)) {
    for (auto& tensor : sample->tensors) {
      apply_to_tensor(tensor);
    }
  }
  if (sample->tensor.has_value()) {
    apply_to_tensor(*sample->tensor);
  }
  if (sample_is_multi_output(*sample)) {
    for (auto& field : sample->fields) {
      if (sample_has_tensor_list(field)) {
        for (auto& tensor : field.tensors) {
          apply_to_tensor(tensor);
        }
      } else if (field.tensor.has_value()) {
        apply_to_tensor(*field.tensor);
      }
    }
  }
}

constexpr const char* kSampleMetaName = "GstSimaSampleMeta";
constexpr const char* kTensorSetMetaName = SIMA_TENSOR_SET_META_NAME;

bool sample_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_SAMPLE_DEBUG", false);
}

bool sample_bytes_enabled() {
  return pipeline_internal::env_bool("SIMA_SAMPLE_BYTES", false);
}

bool sample_storage_debug_enabled() {
  return sample_debug_enabled() || pipeline_internal::env_bool("SIMA_STAGE_DEBUG", false) ||
         pipeline_internal::env_bool("SIMA_INPUTSTREAM_HOLDER_DEBUG", false);
}

bool tensor_set_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_TENSOR_SET_DEBUG", false);
}

bool inputstream_restore_preproc_buffer_meta_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached != 0;
  }
  cached =
      pipeline_internal::env_bool("SIMA_INPUTSTREAM_RESTORE_PREPROC_BUFFER_META", false) ? 1 : 0;
  return cached != 0;
}

bool inputstream_fast_static_decode_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached != 0;
  }
  cached = pipeline_internal::env_bool("SIMA_INPUTSTREAM_FAST_STATIC_DECODE", true) ? 1 : 0;
  return cached != 0;
}

bool inputstream_fast_minimal_output_meta_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached != 0;
  }
  cached = pipeline_internal::env_bool("SIMA_INPUTSTREAM_FAST_MINIMAL_OUTPUT_META", true) ? 1 : 0;
  return cached != 0;
}

bool inputstream_fast_static_preprocess_meta_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached != 0;
  }
  // Do not assume preprocess semantics are static across frames by default.
  // The YOLO path can have per-input original/resize metadata; preserving the
  // first cached tensor semantic made the fast TensorSet decode cheaper, but it
  // produced stale preprocess values and broke boxdecode correctness. Keep this
  // as an explicit experiment-only opt-in.
  cached = pipeline_internal::env_bool("SIMA_INPUTSTREAM_FAST_STATIC_PREPROC_META", false) ? 1 : 0;
  return cached != 0;
}

using InputStreamDecodeProfileClock = std::chrono::steady_clock;

struct InputStreamDecodeProfileSample {
  double restore_buffer_meta_ms = 0.0;
  double envelope_ms = 0.0;
  double restore_sample_meta_ms = 0.0;
  double apply_port_ms = 0.0;
  double total_decode_ms = 0.0;

  double tensor_sig_ms = 0.0;
  double tensor_cache_lock_ms = 0.0;
  double tensor_cache_instantiate_ms = 0.0;
  double tensor_slow_ms = 0.0;
  double tensor_cache_store_ms = 0.0;

  double fast_make_storage_ms = 0.0;
  double fast_sample_copy_template_ms = 0.0;
  double fast_sample_meta_ms = 0.0;
  double fast_tensor_clear_ms = 0.0;
  double fast_tensor_loop_ms = 0.0;
  double fast_tensor_view_ms = 0.0;

  double slow_descriptor_ms = 0.0;
  double slow_base_ms = 0.0;
  double slow_tensor_loop_ms = 0.0;
};

thread_local InputStreamDecodeProfileSample* g_inputstream_decode_profile = nullptr;

double inputstream_decode_profile_ms(const InputStreamDecodeProfileClock::duration& duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

bool inputstream_decode_profile_enabled() {
  static int cached = -1;
  if (cached != -1) {
    return cached != 0;
  }
  cached = pipeline_internal::env_bool("SIMA_INPUTSTREAM_DECODE_PROFILE", false) ? 1 : 0;
  return cached != 0;
}

int inputstream_decode_profile_interval() {
  static int cached = -1;
  if (cached > 0) {
    return cached;
  }
  cached = 120;
  const char* raw = std::getenv("SIMA_INPUTSTREAM_DECODE_PROFILE_EVERY");
  if (!raw || !*raw) {
    raw = std::getenv("SIMA_RUNTIME_PROFILE_EVERY");
  }
  if (raw && *raw) {
    const int parsed = std::atoi(raw);
    if (parsed > 0) {
      cached = parsed;
    }
  }
  return cached;
}

struct InputStreamDecodeProfileScope {
  InputStreamDecodeProfileSample* previous = nullptr;
  explicit InputStreamDecodeProfileScope(InputStreamDecodeProfileSample* sample)
      : previous(g_inputstream_decode_profile) {
    g_inputstream_decode_profile = sample;
  }
  ~InputStreamDecodeProfileScope() {
    g_inputstream_decode_profile = previous;
  }
};

void log_inputstream_decode_profile(const InputStreamDecodeProfileSample& sample) {
  if (!inputstream_decode_profile_enabled()) {
    return;
  }
  struct Accumulator {
    uint64_t samples = 0;
    InputStreamDecodeProfileSample sum;
    InputStreamDecodeProfileSample max;
  };
  static std::mutex mutex;
  static Accumulator acc;
  const auto add_field = [](double& dst, double value) { dst += value; };
  const auto max_field = [](double& dst, double value) { dst = std::max(dst, value); };
  std::lock_guard<std::mutex> lock(mutex);
  acc.samples++;
#define SIMA_INPUTSTREAM_DECODE_PROFILE_FOR_EACH(F)                                                \
  F(restore_buffer_meta_ms);                                                                       \
  F(envelope_ms);                                                                                  \
  F(restore_sample_meta_ms);                                                                       \
  F(apply_port_ms);                                                                                \
  F(total_decode_ms);                                                                              \
  F(tensor_sig_ms);                                                                                \
  F(tensor_cache_lock_ms);                                                                         \
  F(tensor_cache_instantiate_ms);                                                                  \
  F(tensor_slow_ms);                                                                               \
  F(tensor_cache_store_ms);                                                                        \
  F(fast_make_storage_ms);                                                                         \
  F(fast_sample_copy_template_ms);                                                                 \
  F(fast_sample_meta_ms);                                                                          \
  F(fast_tensor_clear_ms);                                                                         \
  F(fast_tensor_loop_ms);                                                                          \
  F(fast_tensor_view_ms);                                                                          \
  F(slow_descriptor_ms);                                                                           \
  F(slow_base_ms);                                                                                 \
  F(slow_tensor_loop_ms)
#define SIMA_INPUTSTREAM_DECODE_ADD(field) add_field(acc.sum.field, sample.field)
#define SIMA_INPUTSTREAM_DECODE_MAX(field) max_field(acc.max.field, sample.field)
  SIMA_INPUTSTREAM_DECODE_PROFILE_FOR_EACH(SIMA_INPUTSTREAM_DECODE_ADD);
  SIMA_INPUTSTREAM_DECODE_PROFILE_FOR_EACH(SIMA_INPUTSTREAM_DECODE_MAX);
#undef SIMA_INPUTSTREAM_DECODE_ADD
#undef SIMA_INPUTSTREAM_DECODE_MAX
  if ((acc.samples % static_cast<uint64_t>(inputstream_decode_profile_interval())) != 0U) {
    return;
  }
  const double n = static_cast<double>(acc.samples);
  std::fprintf(stderr,
               "[runtime-profile] component=inputstream_decode samples=%llu "
               "avg_ms{restore_buffer_meta=%.4f,envelope=%.4f,restore_sample_meta=%.4f,apply_port=%"
               ".4f,total_decode=%.4f,"
               "tensor_sig=%.4f,tensor_cache_lock=%.4f,tensor_cache_instantiate=%.4f,tensor_slow=%."
               "4f,tensor_cache_store=%.4f,"
               "fast_make_storage=%.4f,fast_sample_copy_template=%.4f,fast_sample_meta=%.4f,fast_"
               "tensor_clear=%.4f,fast_tensor_loop=%.4f,fast_tensor_view=%.4f,"
               "slow_descriptor=%.4f,slow_base=%.4f,slow_tensor_loop=%.4f} "
               "max_ms{total_decode=%.4f,envelope=%.4f,tensor_sig=%.4f,tensor_cache_instantiate=%."
               "4f,tensor_slow=%.4f,"
               "fast_make_storage=%.4f,fast_tensor_loop=%.4f,slow_descriptor=%.4f,slow_base=%.4f,"
               "slow_tensor_loop=%.4f}\n",
               static_cast<unsigned long long>(acc.samples), acc.sum.restore_buffer_meta_ms / n,
               acc.sum.envelope_ms / n, acc.sum.restore_sample_meta_ms / n,
               acc.sum.apply_port_ms / n, acc.sum.total_decode_ms / n, acc.sum.tensor_sig_ms / n,
               acc.sum.tensor_cache_lock_ms / n, acc.sum.tensor_cache_instantiate_ms / n,
               acc.sum.tensor_slow_ms / n, acc.sum.tensor_cache_store_ms / n,
               acc.sum.fast_make_storage_ms / n, acc.sum.fast_sample_copy_template_ms / n,
               acc.sum.fast_sample_meta_ms / n, acc.sum.fast_tensor_clear_ms / n,
               acc.sum.fast_tensor_loop_ms / n, acc.sum.fast_tensor_view_ms / n,
               acc.sum.slow_descriptor_ms / n, acc.sum.slow_base_ms / n,
               acc.sum.slow_tensor_loop_ms / n, acc.max.total_decode_ms, acc.max.envelope_ms,
               acc.max.tensor_sig_ms, acc.max.tensor_cache_instantiate_ms, acc.max.tensor_slow_ms,
               acc.max.fast_make_storage_ms, acc.max.fast_tensor_loop_ms,
               acc.max.slow_descriptor_ms, acc.max.slow_base_ms, acc.max.slow_tensor_loop_ms);
#undef SIMA_INPUTSTREAM_DECODE_PROFILE_FOR_EACH
}

void log_sample_tensor_state(const char* where, const char* label,
                             const simaai::neat::Tensor& tensor) {
  if (!sample_storage_debug_enabled())
    return;
  const auto* storage = tensor.storage.get();
  const void* holder_ptr = (storage && storage->holder) ? storage->holder.get() : nullptr;
  std::fprintf(
      stderr,
      "[SAMPLE][tensor] %s %s storage=%p kind=%d holder=%p read_only=%d planes=%zu shape=%s\n",
      where ? where : "unknown", label ? label : "tensor", static_cast<const void*>(storage),
      storage ? static_cast<int>(storage->kind) : -1, holder_ptr, tensor.read_only ? 1 : 0,
      tensor.planes.size(), tensor.debug_string().c_str());
}

static std::optional<Sample> bundle_from_sample_meta(GstSample* sample, const char* where,
                                                     bool copy_output);
static std::optional<Sample> tensor_set_from_meta(GstSample* sample, const char* where,
                                                  bool copy_output, InputStream::State* st);
static std::optional<Sample> tensor_set_from_meta_slow(GstSample* sample, const char* where,
                                                       bool copy_output);

TensorDType tensor_dtype_from_tensor_set_descriptor(gint dtype) {
  switch (dtype) {
  case SIMA_TENSOR_SET_DTYPE_UINT8_V1:
    return TensorDType::UInt8;
  case SIMA_TENSOR_SET_DTYPE_INT8_V1:
    return TensorDType::Int8;
  case SIMA_TENSOR_SET_DTYPE_UINT16_V1:
    return TensorDType::UInt16;
  case SIMA_TENSOR_SET_DTYPE_INT16_V1:
    return TensorDType::Int16;
  case SIMA_TENSOR_SET_DTYPE_BF16_V1:
    return TensorDType::BFloat16;
  case SIMA_TENSOR_SET_DTYPE_INT32_V1:
    return TensorDType::Int32;
  case SIMA_TENSOR_SET_DTYPE_FP32_V1:
    return TensorDType::Float32;
  case SIMA_TENSOR_SET_DTYPE_FP64_V1:
    return TensorDType::Float64;
  case SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1:
  default:
    return TensorDType::UInt8;
  }
}

TensorLayout tensor_layout_from_tensor_set_descriptor(gint layout) {
  switch (layout) {
  case SIMA_TENSOR_SET_LAYOUT_HW_V1:
    return TensorLayout::HW;
  case SIMA_TENSOR_SET_LAYOUT_HWC_V1:
    return TensorLayout::HWC;
  case SIMA_TENSOR_SET_LAYOUT_CHW_V1:
    return TensorLayout::CHW;
  case SIMA_TENSOR_SET_LAYOUT_NHWC_V1:
    return TensorLayout::HWC;
  case SIMA_TENSOR_SET_LAYOUT_NCHW_V1:
    return TensorLayout::CHW;
  case SIMA_TENSOR_SET_LAYOUT_UNKNOWN_V1:
  default:
    return TensorLayout::Unknown;
  }
}

std::size_t tensor_dtype_bytes_from_tensor_set_descriptor(gint dtype) {
  switch (dtype) {
  case SIMA_TENSOR_SET_DTYPE_UINT16_V1:
  case SIMA_TENSOR_SET_DTYPE_INT16_V1:
  case SIMA_TENSOR_SET_DTYPE_BF16_V1:
    return 2U;
  case SIMA_TENSOR_SET_DTYPE_INT32_V1:
  case SIMA_TENSOR_SET_DTYPE_FP32_V1:
    return 4U;
  case SIMA_TENSOR_SET_DTYPE_FP64_V1:
    return 8U;
  case SIMA_TENSOR_SET_DTYPE_UINT8_V1:
  case SIMA_TENSOR_SET_DTYPE_INT8_V1:
  case SIMA_TENSOR_SET_DTYPE_UNKNOWN_V1:
  default:
    return 1U;
  }
}

bool tensor_set_descriptor_uses_byte_addressed_shape(
    const pipeline_internal::TensorBufferTensorDescriptor& descriptor,
    std::size_t* out_elem_bytes = nullptr) {
  if (descriptor.shape.empty()) {
    return false;
  }
  const std::size_t elem_bytes = tensor_dtype_bytes_from_tensor_set_descriptor(descriptor.dtype);
  if (elem_bytes <= 1U || descriptor.size_bytes == 0U) {
    return false;
  }
  std::size_t flat_extent = 1U;
  for (const auto dim : descriptor.shape) {
    if (dim <= 0) {
      return false;
    }
    if (flat_extent > (std::numeric_limits<std::size_t>::max() / static_cast<std::size_t>(dim))) {
      return false;
    }
    flat_extent *= static_cast<std::size_t>(dim);
  }
  if (out_elem_bytes) {
    *out_elem_bytes = elem_bytes;
  }
  return flat_extent == descriptor.size_bytes;
}

void normalize_tensor_set_descriptor_shape(
    const pipeline_internal::TensorBufferTensorDescriptor& descriptor,
    std::vector<int64_t>* out_shape, std::vector<int64_t>* out_strides_bytes) {
  if (!out_shape || !out_strides_bytes) {
    return;
  }
  *out_shape = descriptor.shape;
  *out_strides_bytes = descriptor.stride_bytes;

  std::size_t elem_bytes = 0U;
  if (!tensor_set_descriptor_uses_byte_addressed_shape(descriptor, &elem_bytes)) {
    return;
  }

  const auto elem_i64 = static_cast<int64_t>(elem_bytes);
  if (!out_shape->empty() && out_shape->back() > 0 && elem_i64 > 0 &&
      (out_shape->back() % elem_i64) == 0) {
    out_shape->back() /= elem_i64;
    return;
  }
}

Tensor
apply_tensor_set_descriptor(const Tensor& base,
                            const pipeline_internal::TensorBufferTensorDescriptor& descriptor,
                            const std::string& stage_key, bool materialize_output) {
  Tensor out = base;
  const bool gst_sample_backed =
      base.storage && base.storage->kind == simaai::neat::StorageKind::GstSample;
  if (gst_sample_backed) {
    if (materialize_output) {
      out = pipeline_internal::copy_tensor_from_sample_memory(base, descriptor.memory_index,
                                                              /*keep_holder=*/false);
    } else if (descriptor.memory_index >= 0) {
      out = pipeline_internal::tensor_view_from_sample_memory(base, descriptor.memory_index,
                                                              /*keep_holder=*/true);
    }
  }
  std::vector<int64_t> normalized_shape;
  std::vector<int64_t> normalized_strides_bytes;
  normalize_tensor_set_descriptor_shape(descriptor, &normalized_shape, &normalized_strides_bytes);

  out.shape.clear();
  out.strides_bytes.clear();
  for (const auto dim : normalized_shape) {
    out.shape.push_back(dim);
  }
  for (const auto stride : normalized_strides_bytes) {
    out.strides_bytes.push_back(stride);
  }
  out.byte_offset = out.byte_offset + descriptor.byte_offset;
  out.dtype = tensor_dtype_from_tensor_set_descriptor(descriptor.dtype);
  out.layout = tensor_layout_from_tensor_set_descriptor(descriptor.layout);
  out.planes.clear();
  out.route.logical_index = descriptor.logical_index;
  out.route.physical_index = descriptor.physical_index;
  out.route.memory_index = descriptor.memory_index;
  out.route.physical_byte_offset = descriptor.byte_offset;
  if (!stage_key.empty()) {
    out.route.stage_key = stage_key;
  }
  out.route.backend_output_index = descriptor.backend_output_index;
  out.route.route_slot = descriptor.route_slot;
  out.route.name = descriptor.logical_name;
  out.route.backend_name = descriptor.backend_name;
  out.route.segment_name = descriptor.segment_name;
  out.semantic.quant.reset();
  if (descriptor.quant.has_value()) {
    QuantSpec quant;
    quant.axis = descriptor.quant->axis;
    quant.scales.reserve(descriptor.quant->scales.size());
    quant.zero_points.reserve(descriptor.quant->zero_points.size());
    for (const auto scale : descriptor.quant->scales) {
      quant.scales.push_back(static_cast<float>(scale));
    }
    for (const auto zero_point : descriptor.quant->zero_points) {
      quant.zero_points.push_back(static_cast<int32_t>(zero_point));
    }
    if (!quant.scales.empty()) {
      quant.scale = quant.scales.front();
    }
    if (!quant.zero_points.empty()) {
      quant.zero_point = quant.zero_points.front();
    }
    out.semantic.quant = std::move(quant);
  }
  if (out.route.name.empty()) {
    out.route.name = "output" + std::to_string(descriptor.logical_index);
  }
  if (out.route.segment_name.empty()) {
    out.route.segment_name = out.route.name;
  }
  if (materialize_output) {
    // GstSample-backed tensor-set descriptors already materialize the full backing
    // memory segment above via copy_tensor_from_sample_memory(). Running an extra
    // clone here would repack padded dense tensors into tight contiguous storage
    // and destroy the published physical stride metadata.
    if (!gst_sample_backed) {
      out = out.clone();
    } else if (out.is_dense() && out.is_contiguous()) {
      const std::size_t bytes = out.dense_bytes_tight();
      if (bytes > 0U) {
        const Mapping src = out.map_read();
        auto storage = simaai::neat::make_cpu_owned_storage(bytes);
        const Mapping dst = storage->map(simaai::neat::MapMode::Write);
        if (!src.data || src.size_bytes < bytes || !dst.data || dst.size_bytes < bytes) {
          throw std::runtime_error("apply_tensor_set_descriptor: contiguous slice copy failed");
        }
        std::memcpy(dst.data, src.data, bytes);
        out.storage = std::move(storage);
        out.device = {simaai::neat::DeviceType::CPU, 0};
        out.byte_offset = 0;
        out.read_only = false;
      }
    }
  }
  return out;
}

static void fill_output_meta_from_sample(GstSample* sample, Sample* out) {
  if (!sample || !out)
    return;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer)
    return;

  restore_sample_timing_from_gst_buffer(buffer, out);

  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s)
    return;

  gint64 frame_id = -1;
  gint64 buffer_offset = -1;
  gint origin_output_slot = -1;
  gst_structure_get_int64(s, "frame-id", &frame_id);
  gst_structure_get_int64(s, "buffer-offset", &buffer_offset);
  gst_structure_get_int(s, "origin_output_slot", &origin_output_slot);
  gint64 input_seq = -1;
  gint64 orig_input_seq = -1;
  const bool has_input_seq = (gst_structure_get_int64(s, "input-seq", &input_seq) == TRUE);
  const bool has_orig_input_seq =
      (gst_structure_get_int64(s, "orig-input-seq", &orig_input_seq) == TRUE);
  const char* stream_id = gst_structure_get_string(s, "stream-id");
  const char* buffer_name = gst_structure_get_string(s, "buffer-name");
  const char* orig_stream_id = gst_structure_get_string(s, "orig-stream-id");
  gboolean sample_frame_valid = FALSE;
  const bool has_sample_frame =
      gst_structure_get_boolean(s, "sample-frame-id-valid", &sample_frame_valid) == TRUE;
  if (has_sample_frame) {
    gint64 sample_frame_id = -1;
    if (sample_frame_valid == TRUE &&
        gst_structure_get_int64(s, "sample-frame-id", &sample_frame_id) == TRUE &&
        sample_frame_id >= 0) {
      out->frame_id = sample_frame_id;
    } else {
      out->frame_id = -1;
    }
  } else if (frame_id >= 0) {
    out->frame_id = frame_id;
  }
  if (has_input_seq && input_seq >= 0) {
    out->input_seq = input_seq;
  } else if (has_orig_input_seq && orig_input_seq >= 0) {
    out->input_seq = orig_input_seq;
  }
  if (has_orig_input_seq && orig_input_seq >= 0) {
    out->orig_input_seq = orig_input_seq;
  } else if (out->input_seq >= 0) {
    out->orig_input_seq = out->input_seq;
  }
  if (orig_stream_id && *orig_stream_id) {
    if (!stream_id || !*stream_id || (buffer_name && std::string(stream_id) == buffer_name)) {
      out->stream_id = orig_stream_id;
    } else {
      out->stream_id = stream_id;
    }
  } else if (stream_id) {
    out->stream_id = stream_id;
  }
  if (buffer_name)
    out->stream_label = buffer_name;
  if (buffer_name)
    out->segment_name = buffer_name;
  if (origin_output_slot >= 0) {
    out->route_slot = origin_output_slot;
  }
  if (buffer_offset >= 0 && buffer_offset <= std::numeric_limits<int>::max()) {
    out->output_index = static_cast<int>(buffer_offset);
    out->logical_output_index = static_cast<int>(buffer_offset);
    if (out->route_slot < 0) {
      out->route_slot = static_cast<int>(buffer_offset);
    }
  }
}

static void fill_output_meta_minimal_from_sample(GstSample* sample, Sample* out) {
  if (!sample || !out)
    return;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer)
    return;

  restore_sample_timing_from_gst_buffer(buffer, out);

  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s)
    return;

  gint64 frame_id = -1;
  gboolean sample_frame_valid = FALSE;
  const bool has_sample_frame =
      gst_structure_get_boolean(s, "sample-frame-id-valid", &sample_frame_valid) == TRUE;
  gint64 sample_frame_id = -1;
  if (has_sample_frame) {
    if (sample_frame_valid == TRUE &&
        gst_structure_get_int64(s, "sample-frame-id", &sample_frame_id) == TRUE &&
        sample_frame_id >= 0) {
      out->frame_id = sample_frame_id;
    } else {
      out->frame_id = -1;
    }
  } else if (gst_structure_get_int64(s, "frame-id", &frame_id) == TRUE && frame_id >= 0) {
    out->frame_id = frame_id;
  }

  gint64 input_seq = -1;
  gint64 orig_input_seq = -1;
  const bool has_input_seq = (gst_structure_get_int64(s, "input-seq", &input_seq) == TRUE);
  const bool has_orig_input_seq =
      (gst_structure_get_int64(s, "orig-input-seq", &orig_input_seq) == TRUE);
  if (has_input_seq && input_seq >= 0) {
    out->input_seq = input_seq;
  } else if (has_orig_input_seq && orig_input_seq >= 0) {
    out->input_seq = orig_input_seq;
  }
  if (has_orig_input_seq && orig_input_seq >= 0) {
    out->orig_input_seq = orig_input_seq;
  } else if (out->input_seq >= 0) {
    out->orig_input_seq = out->input_seq;
  }
}

static Sample output_from_sample_stream_inner(GstSample* sample, const char* where,
                                              bool copy_output, bool keep_holder_for_tensor_copy) {
  const auto copy_video_to_cpu = [](GstSample* s) -> simaai::neat::Tensor {
    if (!s)
      throw std::runtime_error("copy_video_to_cpu: null sample");
    const auto make_plane_local = [](simaai::neat::PlaneRole role, int64_t h, int64_t w,
                                     int64_t stride_bytes, int64_t offset_bytes,
                                     int64_t elem_bytes) -> simaai::neat::Plane {
      simaai::neat::Plane plane;
      plane.role = role;
      plane.shape = {h, w};
      plane.strides_bytes = {stride_bytes, elem_bytes};
      plane.byte_offset = offset_bytes;
      return plane;
    };
    GstCaps* caps = gst_sample_get_caps(s);
    if (!caps)
      throw std::runtime_error("copy_video_to_cpu: missing caps");
    GstVideoInfo info;
    std::memset(&info, 0, sizeof(info));
    if (!gst_video_info_from_caps(&info, caps)) {
      throw std::runtime_error("copy_video_to_cpu: gst_video_info_from_caps failed");
    }
    GstBuffer* buffer = gst_sample_get_buffer(s);
    if (!buffer)
      throw std::runtime_error("copy_video_to_cpu: missing buffer");

    GstVideoFrame frame;
    std::memset(&frame, 0, sizeof(frame));
    if (!gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
      throw std::runtime_error("copy_video_to_cpu: gst_video_frame_map failed");
    }

    const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&info);
    const int w = GST_VIDEO_INFO_WIDTH(&info);
    const int h = GST_VIDEO_INFO_HEIGHT(&info);
    if (w <= 0 || h <= 0) {
      gst_video_frame_unmap(&frame);
      throw std::runtime_error("copy_video_to_cpu: invalid dimensions");
    }

    simaai::neat::Tensor out;
    out.dtype = simaai::neat::TensorDType::UInt8;
    out.device = {simaai::neat::DeviceType::CPU, 0};
    out.read_only = false;

    simaai::neat::ImageSpec image;
    if (fmt == GST_VIDEO_FORMAT_NV12) {
      image.format = simaai::neat::ImageSpec::PixelFormat::NV12;
    } else if (fmt == GST_VIDEO_FORMAT_I420) {
      image.format = simaai::neat::ImageSpec::PixelFormat::I420;
    } else if (fmt == GST_VIDEO_FORMAT_RGB) {
      image.format = simaai::neat::ImageSpec::PixelFormat::RGB;
    } else if (fmt == GST_VIDEO_FORMAT_BGR) {
      image.format = simaai::neat::ImageSpec::PixelFormat::BGR;
    } else if (fmt == GST_VIDEO_FORMAT_GRAY8) {
      image.format = simaai::neat::ImageSpec::PixelFormat::GRAY8;
    } else {
      gst_video_frame_unmap(&frame);
      throw std::runtime_error("copy_video_to_cpu: unsupported video format");
    }
    out.semantic.image = image;

    const int64_t elem = 1;
    if (fmt == GST_VIDEO_FORMAT_NV12) {
      const size_t y_sz = static_cast<size_t>(w) * static_cast<size_t>(h);
      const size_t uv_sz = y_sz / 2;
      auto storage = simaai::neat::make_cpu_owned_storage(y_sz + uv_sz);
      auto map = storage->map(simaai::neat::MapMode::Write);
      if (!map.data || map.size_bytes < (y_sz + uv_sz)) {
        gst_video_frame_unmap(&frame);
        throw std::runtime_error("copy_video_to_cpu: map failed");
      }
      uint8_t* dst = static_cast<uint8_t*>(map.data);
      const uint8_t* y = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const uint8_t* uv = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
      const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
      const int uv_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);
      {
        for (int r = 0; r < h; ++r) {
          std::memcpy(dst + static_cast<size_t>(r) * static_cast<size_t>(w),
                      y + static_cast<size_t>(r) * static_cast<size_t>(y_stride),
                      static_cast<size_t>(w));
        }
      }
      uint8_t* dst_uv = dst + y_sz;
      const int uv_h = h / 2;
      {
        for (int r = 0; r < uv_h; ++r) {
          std::memcpy(dst_uv + static_cast<size_t>(r) * static_cast<size_t>(w),
                      uv + static_cast<size_t>(r) * static_cast<size_t>(uv_stride),
                      static_cast<size_t>(w));
        }
      }
      out.storage = std::move(storage);
      out.shape = {h, w};
      out.layout = simaai::neat::TensorLayout::HW;
      out.strides_bytes = {static_cast<int64_t>(w), elem};
      out.planes.push_back(make_plane_local(simaai::neat::PlaneRole::Y, h, w, w, 0, elem));
      out.planes.push_back(make_plane_local(simaai::neat::PlaneRole::UV, h / 2, w, w,
                                            static_cast<int64_t>(y_sz), elem));
    } else if (fmt == GST_VIDEO_FORMAT_I420) {
      const size_t y_sz = static_cast<size_t>(w) * static_cast<size_t>(h);
      const size_t u_sz = y_sz / 4;
      const size_t v_sz = u_sz;
      auto storage = simaai::neat::make_cpu_owned_storage(y_sz + u_sz + v_sz);
      auto map = storage->map(simaai::neat::MapMode::Write);
      if (!map.data || map.size_bytes < (y_sz + u_sz + v_sz)) {
        gst_video_frame_unmap(&frame);
        throw std::runtime_error("copy_video_to_cpu: map failed");
      }
      uint8_t* dst = static_cast<uint8_t*>(map.data);
      const uint8_t* y = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const uint8_t* u = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 1));
      const uint8_t* v = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 2));
      const int y_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
      const int u_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 1);
      const int v_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 2);
      {
        for (int r = 0; r < h; ++r) {
          std::memcpy(dst + static_cast<size_t>(r) * static_cast<size_t>(w),
                      y + static_cast<size_t>(r) * static_cast<size_t>(y_stride),
                      static_cast<size_t>(w));
        }
        uint8_t* dst_u = dst + y_sz;
        uint8_t* dst_v = dst + y_sz + u_sz;
        const int uv_h = h / 2;
        const int uv_w = w / 2;
        for (int r = 0; r < uv_h; ++r) {
          std::memcpy(dst_u + static_cast<size_t>(r) * static_cast<size_t>(uv_w),
                      u + static_cast<size_t>(r) * static_cast<size_t>(u_stride),
                      static_cast<size_t>(uv_w));
          std::memcpy(dst_v + static_cast<size_t>(r) * static_cast<size_t>(uv_w),
                      v + static_cast<size_t>(r) * static_cast<size_t>(v_stride),
                      static_cast<size_t>(uv_w));
        }
      }
      out.storage = std::move(storage);
      out.shape = {h, w};
      out.layout = simaai::neat::TensorLayout::HW;
      out.strides_bytes = {static_cast<int64_t>(w), elem};
      out.planes.push_back(make_plane_local(simaai::neat::PlaneRole::Y, h, w, w, 0, elem));
      out.planes.push_back(make_plane_local(simaai::neat::PlaneRole::U, h / 2, w / 2, w / 2,
                                            static_cast<int64_t>(y_sz), elem));
      out.planes.push_back(make_plane_local(simaai::neat::PlaneRole::V, h / 2, w / 2, w / 2,
                                            static_cast<int64_t>(y_sz + u_sz), elem));
    } else {
      const int channels = (fmt == GST_VIDEO_FORMAT_GRAY8) ? 1 : 3;
      const size_t row_bytes = static_cast<size_t>(w) * static_cast<size_t>(channels);
      const size_t total = row_bytes * static_cast<size_t>(h);
      auto storage = simaai::neat::make_cpu_owned_storage(total);
      auto map = storage->map(simaai::neat::MapMode::Write);
      if (!map.data || map.size_bytes < total) {
        gst_video_frame_unmap(&frame);
        throw std::runtime_error("copy_video_to_cpu: map failed");
      }
      uint8_t* dst = static_cast<uint8_t*>(map.data);
      const uint8_t* src = static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0));
      const int src_stride = GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0);
      {
        for (int r = 0; r < h; ++r) {
          std::memcpy(dst + static_cast<size_t>(r) * row_bytes,
                      src + static_cast<size_t>(r) * static_cast<size_t>(src_stride), row_bytes);
        }
      }
      out.storage = std::move(storage);
      if (channels == 1) {
        out.shape = {h, w};
        out.layout = simaai::neat::TensorLayout::HW;
        out.strides_bytes = {static_cast<int64_t>(row_bytes), elem};
      } else {
        out.shape = {h, w, channels};
        out.layout = simaai::neat::TensorLayout::HWC;
        out.strides_bytes = {static_cast<int64_t>(row_bytes), static_cast<int64_t>(channels) * elem,
                             elem};
      }
    }

    gst_video_frame_unmap(&frame);
    return out;
  };
  const auto normalize_format = [](const std::string& fmt) {
    std::string out;
    out.reserve(fmt.size());
    for (char c : fmt) {
      out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
    }
    if (out == "YUV420P" || out == "YUV420")
      return std::string("I420");
    if (out == "GRAY")
      return std::string("GRAY8");
    return out;
  };

  Sample out;
  GstCaps* out_caps = gst_sample_get_caps(sample);
  const GstStructure* st = out_caps ? gst_caps_get_structure(out_caps, 0) : nullptr;
  const char* media = st ? gst_structure_get_name(st) : nullptr;

  out.caps_string = pipeline_internal::gst_caps_to_string_safe(out_caps);
  out.media_type = media ? media : "";
  out.payload_type = payload_type_from_media_type(out.media_type);
  const EncodedSpec::Codec encoded_codec = caps_to_codec(out.caps_string);
  const bool is_encoded = encoded_codec != EncodedSpec::Codec::UNKNOWN;
  if (is_encoded) {
    out.payload_type = PayloadType::Encoded;
  }

  fill_output_meta_from_sample(sample, &out);

  const bool is_raw = media && (std::string(media).rfind("video/x-raw", 0) == 0 ||
                                std::string(media) == "application/vnd.simaai.tensor");
  if (sample_storage_debug_enabled()) {
    std::fprintf(stderr, "[SAMPLE] %s media=%s is_raw=%d copy_output=%d caps=%s\n",
                 where ? where : "unknown", media ? media : "<none>", is_raw ? 1 : 0,
                 copy_output ? 1 : 0, out.caps_string.c_str());
  }

  if (is_raw) {
    simaai::neat::Tensor neat = simaai::neat::from_gst_sample(sample);
    log_sample_tensor_state(where, "from_gst_sample", neat);
    if (pipeline_internal::env_bool("SIMA_NEAT_CAPS_TRACE", false)) {
      const auto constraint = pipeline_internal::tensor_constraint_from_caps(out_caps);
      std::fprintf(stderr, "[NEAT_CAPS] caps=%s constraint=%s tensor=%s\n", out.caps_string.c_str(),
                   pipeline_internal::tensor_constraint_debug_string(constraint).c_str(),
                   neat.debug_string().c_str());
    }
    out.kind = SampleKind::TensorSet;
    out.payload_tag = normalize_format(tensor_format_name(neat));
    out.format = out.payload_tag;
    if (copy_output) {
      bool copied = false;
      const bool is_tensor_media = media && std::string(media) == "application/vnd.simaai.tensor";
      if (sample_storage_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s tensor_media=%d\n", where ? where : "unknown",
                     is_tensor_media ? 1 : 0);
      }
      if (is_tensor_media && neat.storage &&
          neat.storage->kind == simaai::neat::StorageKind::GstSample && neat.planes.empty()) {
        try {
          const int memory_index = (out.memory_index >= 0)
                                       ? out.memory_index
                                       : ((out.output_index >= 0) ? out.output_index : -1);
          out.tensors = TensorList{pipeline_internal::copy_tensor_from_sample_memory(
              neat, memory_index, keep_holder_for_tensor_copy)};
          out.owned = true;
          copied = true;
          if (!out.tensors.empty()) {
            log_sample_tensor_state(where, "copy_tensor_from_sample_memory", out.tensors.front());
          }
        } catch (const std::exception& e) {
          if (pipeline_internal::env_bool("SIMA_GRAPH_OUTPUT_COPY_DEBUG", false)) {
            std::fprintf(stderr, "[GRAPH_OUTPUT] copy_tensor_from_sample_memory failed: %s\n",
                         e.what());
          }
        }
      }
      if (!copied) {
        out.tensors = TensorList{neat.clone()};
        out.owned = true;
        if (!out.tensors.empty()) {
          log_sample_tensor_state(where, "clone_fallback", out.tensors.front());
        }
      }
    } else {
      out.tensors = TensorList{neat};
      out.owned = false;
      if (!out.tensors.empty()) {
        log_sample_tensor_state(where, "zero_copy_output", out.tensors.front());
      }
    }
    return out;
  }

  if (is_encoded && !copy_output) {
    simaai::neat::Tensor neat = simaai::neat::from_gst_sample(sample);
    log_sample_tensor_state(where, "encoded_zero_copy_output", neat);
    out.kind = SampleKind::TensorSet;
    out.tensors = TensorList{std::move(neat)};
    out.owned = false;
    return out;
  }

  // Encoded or non-raw payload: copy bytes and wrap in encoded Sample.
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    throw std::runtime_error("InputStream::pull: missing buffer");
  }
  GstMapInfo map{};
  if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    throw std::runtime_error("InputStream::pull: failed to map buffer");
  }
  std::vector<uint8_t> bytes;
  if (map.size > 0) {
    bytes.resize(static_cast<size_t>(map.size));
    std::memcpy(bytes.data(), map.data, bytes.size());
  }
  gst_buffer_unmap(buffer, &map);

  if (sample_storage_debug_enabled()) {
    std::fprintf(
        stderr, "[SAMPLE] %s non_raw_path bytes=%zu media=%s caps=%s (make_encoded_sample)\n",
        where ? where : "unknown", bytes.size(), out.media_type.c_str(), out.caps_string.c_str());
  }

  Sample enc = make_encoded_sample(std::move(bytes), out.caps_string, out.pts_ns, out.dts_ns,
                                   out.duration_ns);
  enc.payload_type = payload_type_from_media_type(out.media_type);
  enc.media_type = out.media_type;
  enc.frame_id = out.frame_id;
  enc.stream_id = out.stream_id;
  enc.stream_label = out.stream_label;
  enc.output_index = out.output_index;
  enc.logical_output_index = out.logical_output_index;
  enc.memory_index = out.memory_index;
  enc.route_slot = out.route_slot;
  enc.segment_name = out.segment_name;
  enc.input_seq = out.input_seq;
  enc.orig_input_seq = out.orig_input_seq;
  enc.owned = out.owned;
  if (const auto preprocess = read_simaai_preprocess_meta(buffer); preprocess.has_value()) {
    if (!enc.tensors.empty() && !enc.tensors.front().semantic.preprocess.has_value()) {
      enc.tensors.front().semantic.preprocess = *preprocess;
    }
    if (enc.tensor.has_value() && !enc.tensor->semantic.preprocess.has_value()) {
      enc.tensor->semantic.preprocess = *preprocess;
    }
  }
  if (sample_storage_debug_enabled() && !enc.tensors.empty()) {
    log_sample_tensor_state(where, "non_raw_encoded_tensor", enc.tensors.front());
  }
  return enc;
}

Sample output_from_sample_stream(GstSample* sample, const char* where, bool copy_output,
                                 const std::optional<OutputTensorOverride>* override_opt,
                                 InputStream::State* st) {
  const bool has_override =
      override_opt && override_opt->has_value() && !override_opt->value().outputs.empty();
  if (auto tensor_set =
          tensor_set_from_meta(sample, where, has_override ? false : copy_output, st)) {
    Sample out = std::move(*tensor_set);
    if (has_override) {
      out = apply_output_tensor_override(out, **override_opt, copy_output);
    }
    return out;
  }
  if (auto bundle = bundle_from_sample_meta(sample, where, copy_output)) {
    Sample out = std::move(*bundle);
    if (has_override) {
      out = apply_output_tensor_override(out, **override_opt, copy_output);
    }
    return out;
  }
  Sample out = output_from_sample_stream_inner(sample, where, has_override ? false : copy_output,
                                               /*keep_holder_for_tensor_copy=*/has_override);
  if (has_override) {
    out = apply_output_tensor_override(out, **override_opt, copy_output);
  }
  if (pipeline_internal::env_bool("SIMA_SAMPLE_FORCE_BUNDLE", false)) {
    if (!sample_is_multi_output(out)) {
      Sample forced;
      forced.kind = SampleKind::Bundle;
      forced.owned = out.owned;
      forced.frame_id = out.frame_id;
      forced.stream_id = out.stream_id;
      forced.input_seq = out.input_seq;
      forced.orig_input_seq = out.orig_input_seq;
      forced.fields.emplace_back(std::move(out));
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: forced bundle with 1 field\n", where);
      }
      return forced;
    }
  }
  return out;
}

Sample output_from_sample_stream(GstSample* sample, const char* where, bool copy_output,
                                 const std::optional<OutputTensorOverride>* override_opt) {
  return output_from_sample_stream(sample, where, copy_output, override_opt, nullptr);
}

Sample sample_from_gst_envelope(GstSample* sample, const char* where, bool copy_output,
                                const std::optional<OutputTensorOverride>* override_opt,
                                InputStream::State* st) {
  return output_from_sample_stream(sample, where, copy_output, override_opt, st);
}

Sample sample_from_gst_envelope(GstSample* sample, const char* where, bool copy_output,
                                const std::optional<OutputTensorOverride>* override_opt) {
  return sample_from_gst_envelope(sample, where, copy_output, override_opt, nullptr);
}

static std::optional<Sample> bundle_from_sample_meta(GstSample* sample, const char* where,
                                                     bool copy_output) {
  if (!sample)
    return std::nullopt;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer)
    return std::nullopt;
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, kSampleMetaName);
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s)
    return std::nullopt;

  const GValue* list_val = gst_structure_get_value(s, "fields");
  if (!list_val || !GST_VALUE_HOLDS_LIST(list_val))
    return std::nullopt;
  const guint field_count = gst_value_list_get_size(list_val);

  Sample out;
  out.kind = SampleKind::Bundle;
  out.owned = copy_output;

  GstCaps* out_caps = gst_sample_get_caps(sample);
  const GstStructure* out_st = out_caps ? gst_caps_get_structure(out_caps, 0) : nullptr;
  const char* media = out_st ? gst_structure_get_name(out_st) : nullptr;
  out.caps_string = pipeline_internal::gst_caps_to_string_safe(out_caps);
  out.media_type = media ? media : "";
  out.payload_type = payload_type_from_media_type(out.media_type);
  fill_output_meta_from_sample(sample, &out);

  if (sample_debug_enabled()) {
    std::fprintf(stderr, "[SAMPLE] %s: bundle meta fields=%u\n", where, field_count);
  }

  out.fields.reserve(field_count);
  std::vector<int> projected_memory_indices;
  projected_memory_indices.reserve(field_count);
  for (guint i = 0; i < field_count; ++i) {
    const GValue* entry_val = gst_value_list_get_value(list_val, i);
    if (!entry_val || !GST_VALUE_HOLDS_STRUCTURE(entry_val)) {
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: field[%u] invalid entry\n", where, i);
      }
      continue;
    }
    const GstStructure* entry = static_cast<const GstStructure*>(g_value_get_boxed(entry_val));
    if (!entry)
      continue;

    const char* field_name = gst_structure_get_string(entry, "name");
    const char* buffer_name = gst_structure_get_string(entry, "buffer-name");
    const char* caps_str = gst_structure_get_string(entry, "caps");
    const char* segment_name = gst_structure_get_string(entry, "segment-name");
    gint logical_output_index = -1;
    gint memory_index = -1;
    gint route_slot = -1;
    gst_structure_get_int(entry, "logical-output-index", &logical_output_index);
    gst_structure_get_int(entry, "memory-index", &memory_index);
    gst_structure_get_int(entry, "route-slot", &route_slot);

    const GValue* buf_val = gst_structure_get_value(entry, "buffer");
    if (!buf_val || !GST_VALUE_HOLDS_BUFFER(buf_val)) {
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: field[%u] missing buffer\n", where, i);
      }
      continue;
    }
    GstBuffer* field_buf = gst_value_get_buffer(buf_val);
    if (!field_buf)
      continue;

    GstCaps* field_caps = nullptr;
    if (caps_str && *caps_str) {
      field_caps = gst_caps_from_string(caps_str);
      if (!field_caps && sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: field[%u] caps parse failed caps=%s\n", where, i,
                     caps_str);
      }
    }
    if (!field_caps) {
      field_caps = gst_sample_get_caps(sample);
      if (field_caps)
        gst_caps_ref(field_caps);
    }
    if (!field_caps) {
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: field[%u] missing caps\n", where, i);
      }
      continue;
    }

    GstSample* field_sample = gst_sample_new(field_buf, field_caps, nullptr, nullptr);
    gst_caps_unref(field_caps);
    if (!field_sample)
      continue;

    Sample field_out = output_from_sample_stream_inner(field_sample, where, copy_output, false);
    gst_sample_unref(field_sample);

    if (field_name && *field_name) {
      field_out.stream_label = field_name;
      if (field_out.port_name.empty()) {
        field_out.port_name = field_name;
      }
    } else if (buffer_name && *buffer_name) {
      field_out.stream_label = buffer_name;
      if (field_out.port_name.empty()) {
        field_out.port_name = buffer_name;
      }
    }
    if (segment_name && *segment_name) {
      field_out.segment_name = segment_name;
    } else if (buffer_name && *buffer_name) {
      field_out.segment_name = buffer_name;
    }
    if (logical_output_index >= 0) {
      field_out.logical_output_index = logical_output_index;
      field_out.output_index = logical_output_index;
    } else if (field_out.output_index >= 0) {
      field_out.logical_output_index = field_out.output_index;
    }
    if (memory_index >= 0) {
      field_out.memory_index = memory_index;
      if (std::find(projected_memory_indices.begin(), projected_memory_indices.end(),
                    memory_index) == projected_memory_indices.end()) {
        projected_memory_indices.push_back(memory_index);
      }
    }
    if (route_slot >= 0) {
      field_out.route_slot = route_slot;
    }

    if (out.frame_id < 0 && field_out.frame_id >= 0) {
      out.frame_id = field_out.frame_id;
    }
    if (out.stream_id.empty() && !field_out.stream_id.empty()) {
      out.stream_id = field_out.stream_id;
    }
    if (out.input_seq < 0 && field_out.input_seq >= 0) {
      out.input_seq = field_out.input_seq;
    }
    if (out.orig_input_seq < 0 && field_out.orig_input_seq >= 0) {
      out.orig_input_seq = field_out.orig_input_seq;
    }

    if (sample_debug_enabled() || sample_bytes_enabled()) {
      const char* name =
          !field_out.stream_label.empty()
              ? field_out.stream_label.c_str()
              : (field_out.segment_name.empty() ? "field" : field_out.segment_name.c_str());
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: field[%u] name=%s caps=%s\n", where, i, name,
                     field_out.caps_string.empty() ? "<none>" : field_out.caps_string.c_str());
      }
      if (sample_bytes_enabled()) {
        const size_t buf_bytes = static_cast<size_t>(gst_buffer_get_size(field_buf));
        size_t tensor_bytes = 0;
        if (sample_has_tensor_list(field_out) && !field_out.tensors.empty()) {
          tensor_bytes = pipeline_internal::tensor_bytes_tight(field_out.tensors.front());
        }
        std::fprintf(stderr,
                     "[SAMPLE] %s: field[%u] name=%s buffer-bytes=%zu tensor-bytes=%zu copy=%s\n",
                     where, i, name, buf_bytes, tensor_bytes, copy_output ? "true" : "false");
      }
    }

    out.fields.emplace_back(std::move(field_out));
  }

  if (out.fields.empty()) {
    if (sample_debug_enabled()) {
      std::fprintf(stderr, "[SAMPLE] %s: bundle meta had no valid fields\n", where);
    }
    return std::nullopt;
  }

  pipeline_internal::record_tensor_bundle_projection(out.fields.size());
  pipeline_internal::record_tensor_packed_view_reuse(out.fields.size(),
                                                     projected_memory_indices.size(), copy_output);

  return out;
}

static std::optional<Sample> tensor_set_from_meta_slow(GstSample* sample, const char* where,
                                                       bool copy_output) {
  if (!sample) {
    return std::nullopt;
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    return std::nullopt;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, kTensorSetMetaName);
  if (!meta) {
    if (tensor_set_debug_enabled() || sample_debug_enabled()) {
      std::fprintf(stderr, "[tensor-set][debug] %s: tensor-set meta missing\n",
                   where ? where : "<unknown>");
    }
    return std::nullopt;
  }

  pipeline_internal::TensorBufferView tensor_buffer_view;
  std::string tensor_buffer_err;
  const auto descriptor_start = InputStreamDecodeProfileClock::now();
  if (!pipeline_internal::tensor_buffer_descriptor_from_sample(sample, &tensor_buffer_view,
                                                               &tensor_buffer_err) ||
      tensor_buffer_view.tensors.empty()) {
    if (g_inputstream_decode_profile) {
      g_inputstream_decode_profile->slow_descriptor_ms +=
          inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - descriptor_start);
    }
    if (tensor_set_debug_enabled() || sample_debug_enabled()) {
      std::fprintf(stderr, "[tensor-set][debug] %s: descriptor_from_sample failed err=%s\n",
                   where ? where : "<unknown>",
                   tensor_buffer_err.empty() ? "<empty>" : tensor_buffer_err.c_str());
    }
    return std::nullopt;
  }
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->slow_descriptor_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - descriptor_start);
  }

  const auto base_start = InputStreamDecodeProfileClock::now();
  Sample base = output_from_sample_stream_inner(sample, where, false, /*keep_holder=*/true);
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->slow_base_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - base_start);
  }
  if (base.kind != SampleKind::TensorSet || base.tensors.empty()) {
    return std::nullopt;
  }
  const Tensor& base_tensor = base.tensors.front();
  if (tensor_set_debug_enabled()) {
    GstBuffer* debug_buffer = gst_sample_get_buffer(sample);
    const guint memory_count = debug_buffer ? gst_buffer_n_memory(debug_buffer) : 0U;
    const std::size_t buffer_size =
        debug_buffer ? static_cast<std::size_t>(gst_buffer_get_size(debug_buffer)) : 0U;
    std::fprintf(stderr,
                 "[tensor-set][debug] where=%s tensors=%u buffer=%p memories=%u buffer_size=%zu "
                 "storage=%p storage_size=%zu segments=%zu\n",
                 where ? where : "<unknown>",
                 static_cast<unsigned>(tensor_buffer_view.tensors.size()),
                 static_cast<void*>(debug_buffer), static_cast<unsigned>(memory_count), buffer_size,
                 static_cast<const void*>(base_tensor.storage.get()),
                 base_tensor.storage ? base_tensor.storage->size_bytes : 0U,
                 base_tensor.storage ? base_tensor.storage->sima_segments.size() : 0U);
    if (base_tensor.storage) {
      for (std::size_t si = 0; si < base_tensor.storage->sima_segments.size(); ++si) {
        const auto& segment = base_tensor.storage->sima_segments[si];
        std::fprintf(stderr, "[tensor-set][debug]   seg[%zu] name=%s size=%zu\n", si,
                     segment.name.c_str(), segment.size_bytes);
      }
    }
    GstMapInfo debug_map{};
    if (debug_buffer && gst_buffer_map(debug_buffer, &debug_map, GST_MAP_READ)) {
      const std::size_t sample_count = std::min<std::size_t>(8U, debug_map.size);
      std::fprintf(stderr, "[tensor-set][debug]   buffer_head=");
      for (std::size_t bi = 0; bi < sample_count; ++bi) {
        if (bi > 0U) {
          std::fprintf(stderr, ",");
        }
        std::fprintf(stderr, "%u", static_cast<unsigned>(debug_map.data[bi]));
      }
      std::fprintf(stderr, "\n");
      gst_buffer_unmap(debug_buffer, &debug_map);
    }
  }

  Sample out;
  out.kind = SampleKind::TensorSet;
  out.owned = copy_output;
  out.caps_string = base.caps_string;
  out.payload_type = base.payload_type;
  out.media_type = base.media_type;
  out.payload_tag = base.payload_tag;
  out.format = base.format;
  out.frame_id = base.frame_id;
  out.stream_id = base.stream_id;
  out.stream_label = base.stream_label;
  out.output_index = base.output_index;
  out.logical_output_index = base.logical_output_index;
  out.memory_index = base.memory_index;
  out.route_slot = base.route_slot;
  out.segment_name = base.segment_name;
  out.input_seq = base.input_seq;
  out.orig_input_seq = base.orig_input_seq;
  out.pts_ns = base.pts_ns;
  out.dts_ns = base.dts_ns;
  out.duration_ns = base.duration_ns;
  out.tensors.reserve(tensor_buffer_view.tensors.size());
  std::unordered_set<int> unique_memory_indices;
  const auto tensor_loop_start = InputStreamDecodeProfileClock::now();
  for (const auto& descriptor : tensor_buffer_view.tensors) {
    if (tensor_set_debug_enabled()) {
      const bool has_quant = descriptor.quant.has_value();
      const double first_scale =
          (has_quant && !descriptor.quant->scales.empty()) ? descriptor.quant->scales.front() : 0.0;
      const long long first_zp = (has_quant && !descriptor.quant->zero_points.empty())
                                     ? static_cast<long long>(descriptor.quant->zero_points.front())
                                     : 0LL;
      std::fprintf(
          stderr,
          "[tensor-set][debug]   desc logical=%d physical=%d backend=%d route_slot=%d memory=%d "
          "byte_offset=%zu size=%zu name=%s backend_name=%s segment=%s has_quant=%d scale=%.9f "
          "zp=%lld\n",
          descriptor.logical_index, descriptor.physical_index, descriptor.backend_output_index,
          descriptor.route_slot, descriptor.memory_index,
          static_cast<size_t>(descriptor.byte_offset), descriptor.size_bytes,
          descriptor.logical_name.c_str(), descriptor.backend_name.c_str(),
          descriptor.segment_name.c_str(), has_quant ? 1 : 0, first_scale, first_zp);
    }
    Tensor tensor = apply_tensor_set_descriptor(base_tensor, descriptor,
                                                tensor_buffer_view.stage_key, copy_output);
    if (tensor_set_debug_enabled()) {
      std::fprintf(stderr,
                   "[tensor-set][debug]     tensor route logical=%d memory=%d slot=%d name=%s "
                   "segment=%s shape=",
                   tensor.route.logical_index, tensor.route.memory_index, tensor.route.route_slot,
                   tensor.route.name.c_str(), tensor.route.segment_name.c_str());
      for (std::size_t i = 0; i < tensor.shape.size(); ++i) {
        std::fprintf(stderr, "%s%lld", i == 0 ? "" : "x", static_cast<long long>(tensor.shape[i]));
      }
      std::fprintf(stderr, "\n");
      try {
        const Mapping mapping = tensor.map_read();
        const auto* bytes = static_cast<const uint8_t*>(mapping.data);
        const std::size_t sample_count =
            (bytes && mapping.size_bytes > 0U) ? std::min<std::size_t>(8U, mapping.size_bytes) : 0U;
        std::fprintf(stderr, "[tensor-set][debug]     tensor head=");
        for (std::size_t bi = 0; bi < sample_count; ++bi) {
          if (bi > 0U) {
            std::fprintf(stderr, ",");
          }
          std::fprintf(stderr, "%u", static_cast<unsigned>(bytes[bi]));
        }
        if (sample_count == 0U) {
          std::fprintf(stderr, "<empty>");
        }
        std::fprintf(stderr, " size=%zu\n", mapping.size_bytes);
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[tensor-set][debug]     tensor map failed: %s\n", e.what());
      }
    }
    out.tensors.emplace_back(std::move(tensor));
    if (descriptor.memory_index >= 0) {
      unique_memory_indices.insert(descriptor.memory_index);
    }
  }
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->slow_tensor_loop_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - tensor_loop_start);
  }
  if (out.segment_name.empty() && !out.tensors.empty()) {
    out.segment_name = out.tensors.front().route.segment_name;
  }
  if (out.stream_label.empty() && !out.tensors.empty()) {
    out.stream_label = out.tensors.front().route.name;
  }

  pipeline_internal::record_tensor_packed_view_reuse(out.tensors.size(),
                                                     unique_memory_indices.size(), copy_output);
  if (sample_debug_enabled()) {
    std::fprintf(stderr, "[SAMPLE] %s: tensor-set meta tensors=%zu\n", where,
                 tensor_buffer_view.tensors.size());
  }
  return out;
}

std::uint64_t tensor_set_decode_hash_bytes(std::uint64_t h, const void* data, std::size_t size) {
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  const auto* bytes = static_cast<const std::uint8_t*>(data);
  for (std::size_t i = 0; i < size; ++i) {
    h ^= static_cast<std::uint64_t>(bytes[i]);
    h *= kFnvPrime;
  }
  return h;
}

std::uint64_t tensor_set_decode_hash_u64(std::uint64_t h, std::uint64_t value) {
  return tensor_set_decode_hash_bytes(h, &value, sizeof(value));
}

bool tensor_set_decode_hash_gbytes_field(const GstStructure* s, const char* field,
                                         std::uint64_t* hash) {
  if (!s || !field || !hash) {
    return false;
  }
  const GValue* value = gst_structure_get_value(s, field);
  *hash = tensor_set_decode_hash_bytes(*hash, field, std::strlen(field));
  if (!value) {
    *hash = tensor_set_decode_hash_u64(*hash, 0);
    return true;
  }
  if (!G_VALUE_HOLDS(value, G_TYPE_BYTES)) {
    return false;
  }
  GBytes* bytes = static_cast<GBytes*>(g_value_get_boxed(value));
  if (!bytes) {
    *hash = tensor_set_decode_hash_u64(*hash, 0);
    return true;
  }
  gsize size = 0;
  const void* data = g_bytes_get_data(bytes, &size);
  *hash = tensor_set_decode_hash_u64(*hash, static_cast<std::uint64_t>(size));
  if (data && size > 0) {
    *hash = tensor_set_decode_hash_bytes(*hash, data, static_cast<std::size_t>(size));
  }
  return true;
}

bool tensor_set_decode_hash_strv_field(const GstStructure* s, const char* field,
                                       std::uint64_t* hash) {
  if (!s || !field || !hash) {
    return false;
  }
  const GValue* value = gst_structure_get_value(s, field);
  *hash = tensor_set_decode_hash_bytes(*hash, field, std::strlen(field));
  if (!value) {
    *hash = tensor_set_decode_hash_u64(*hash, 0);
    return true;
  }
  if (!G_VALUE_HOLDS(value, G_TYPE_STRV)) {
    return false;
  }
  auto* raw = static_cast<const gchar* const*>(g_value_get_boxed(value));
  if (!raw) {
    *hash = tensor_set_decode_hash_u64(*hash, 0);
    return true;
  }
  for (guint i = 0; raw[i] != nullptr; ++i) {
    const char* item = raw[i];
    const std::size_t len = std::strlen(item);
    *hash = tensor_set_decode_hash_u64(*hash, static_cast<std::uint64_t>(len));
    *hash = tensor_set_decode_hash_bytes(*hash, item, len);
  }
  return true;
}

bool read_tensor_set_output_decode_signature(GstSample* sample, bool copy_output,
                                             TensorSetOutputDecodeSignature* out) {
  if (!sample || !out) {
    return false;
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    return false;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, kTensorSetMetaName);
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    return false;
  }
  guint version = 0;
  guint tensor_count = 0;
  guint descriptor_size = 0;
  gst_structure_get_uint(s, SIMA_TENSOR_SET_META_FIELD_VERSION, &version);
  gst_structure_get_uint(s, SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT, &tensor_count);
  gst_structure_get_uint(s, SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE, &descriptor_size);
  if (version != SIMA_TENSOR_SET_META_VERSION || tensor_count == 0U ||
      descriptor_size != sizeof(SimaTensorDescriptorV2)) {
    return false;
  }
  constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
  std::uint64_t hash = kFnvOffset;
  if (!tensor_set_decode_hash_gbytes_field(s, SIMA_TENSOR_SET_META_FIELD_DESCRIPTORS, &hash) ||
      !tensor_set_decode_hash_gbytes_field(s, SIMA_TENSOR_SET_META_FIELD_QUANT_SCALES, &hash) ||
      !tensor_set_decode_hash_gbytes_field(s, SIMA_TENSOR_SET_META_FIELD_QUANT_ZERO_POINTS,
                                           &hash) ||
      !tensor_set_decode_hash_strv_field(s, SIMA_TENSOR_SET_META_FIELD_NAME_TABLE, &hash) ||
      !tensor_set_decode_hash_strv_field(s, SIMA_TENSOR_SET_META_FIELD_SEGMENT_TABLE, &hash)) {
    return false;
  }

  TensorSetOutputDecodeSignature sig;
  sig.tensor_count = tensor_count;
  sig.descriptor_size = descriptor_size;
  sig.memory_count = gst_buffer_n_memory(buffer);
  sig.copy_output = copy_output;
  sig.meta_hash = hash;
  if (GstCaps* caps = gst_sample_get_caps(sample)) {
    if (gchar* caps_text = gst_caps_to_string(caps)) {
      sig.caps_hash = g_str_hash(caps_text);
      g_free(caps_text);
    }
  }
  if (const char* stage_key = gst_structure_get_string(s, SIMA_TENSOR_SET_META_FIELD_STAGE_KEY)) {
    sig.stage_key = stage_key;
  }
  *out = std::move(sig);
  return true;
}

bool tensor_set_output_decode_fast_key_matches(GstSample* sample, bool copy_output,
                                               const TensorSetOutputDecodeSignature& cached) {
  if (!inputstream_fast_static_decode_enabled()) {
    return false;
  }
  if (!sample) {
    return false;
  }
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    return false;
  }
  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, kTensorSetMetaName);
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s) {
    return false;
  }
  guint version = 0;
  guint tensor_count = 0;
  guint descriptor_size = 0;
  gst_structure_get_uint(s, SIMA_TENSOR_SET_META_FIELD_VERSION, &version);
  gst_structure_get_uint(s, SIMA_TENSOR_SET_META_FIELD_TENSOR_COUNT, &tensor_count);
  gst_structure_get_uint(s, SIMA_TENSOR_SET_META_FIELD_DESCRIPTOR_SIZE, &descriptor_size);
  if (version != SIMA_TENSOR_SET_META_VERSION || tensor_count != cached.tensor_count ||
      descriptor_size != cached.descriptor_size ||
      gst_buffer_n_memory(buffer) != cached.memory_count || copy_output != cached.copy_output) {
    return false;
  }
  const char* stage_key = gst_structure_get_string(s, SIMA_TENSOR_SET_META_FIELD_STAGE_KEY);
  const std::string_view stage_view = stage_key ? std::string_view(stage_key) : std::string_view();
  return stage_view == cached.stage_key;
}

void clear_dynamic_sample_fields_for_decode_template(Sample* sample) {
  if (!sample) {
    return;
  }
  sample->frame_id = -1;
  sample->input_seq = -1;
  sample->orig_input_seq = -1;
  sample->pts_ns = -1;
  sample->dts_ns = -1;
  sample->duration_ns = -1;
}

void strip_tensor_storage_for_decode_template(Tensor* tensor) {
  if (!tensor) {
    return;
  }
  tensor->storage.reset();
  if (!inputstream_fast_static_preprocess_meta_enabled()) {
    tensor->semantic.preprocess.reset();
  }
}

bool tensor_is_device_gstsample_holder(const Tensor& tensor) {
  if (!tensor.storage || tensor.storage->kind != simaai::neat::StorageKind::GstSample ||
      !tensor.storage->holder) {
    return false;
  }
  return tensor.device.type != simaai::neat::DeviceType::CPU ||
         tensor.storage->device.type != simaai::neat::DeviceType::CPU ||
         tensor.storage->sima_mem_target_flags != 0 || !tensor.storage->sima_segments.empty();
}

bool sample_has_device_gstsample_holder(const Sample& sample) {
  if (sample_has_tensor_list(sample)) {
    for (const auto& tensor : sample.tensors) {
      if (tensor_is_device_gstsample_holder(tensor)) {
        return true;
      }
    }
  }
  if (sample.tensor.has_value() && tensor_is_device_gstsample_holder(*sample.tensor)) {
    return true;
  }
  for (const auto& field : sample.fields) {
    if (sample_has_device_gstsample_holder(field)) {
      return true;
    }
  }
  return false;
}

void attach_holder_loan_release_to_sample(Sample& sample,
                                          const pipeline_internal::HolderLoanGatePtr& gate) {
  if (!gate) {
    return;
  }
  std::vector<const void*> seen_storage;
  auto attach_tensor = [&](Tensor& tensor) {
    if (!tensor_is_device_gstsample_holder(tensor)) {
      return;
    }
    const void* key = tensor.storage.get();
    if (std::find(seen_storage.begin(), seen_storage.end(), key) != seen_storage.end()) {
      return;
    }
    seen_storage.push_back(key);
    auto original_holder = tensor.storage->holder;
    tensor.storage->holder =
        std::shared_ptr<void>(original_holder.get(), [original_holder, gate](void*) mutable {
          original_holder.reset();
          gate->release();
        });
  };
  auto walk = [&](auto&& self, Sample& s) -> void {
    if (sample_has_tensor_list(s)) {
      for (auto& tensor : s.tensors) {
        attach_tensor(tensor);
      }
    }
    if (s.tensor.has_value()) {
      attach_tensor(*s.tensor);
    }
    for (auto& field : s.fields) {
      self(self, field);
    }
  };
  walk(walk, sample);
}

std::optional<Sample>
instantiate_tensor_set_from_cached_decode(GstSample* sample, const char* where,
                                          const CachedTensorSetOutputDecode& cache) {
  if (!sample || !cache.valid || cache.tensor_templates.empty()) {
    return std::nullopt;
  }

  std::shared_ptr<simaai::neat::Storage> base_storage;
  try {
    const auto make_storage_start = InputStreamDecodeProfileClock::now();
    base_storage =
        pipeline_internal::make_gst_sample_storage_with_segments(sample, cache.runtime_segments);
    if (g_inputstream_decode_profile) {
      g_inputstream_decode_profile->fast_make_storage_ms +=
          inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - make_storage_start);
    }
  } catch (const std::exception& e) {
    if (tensor_set_debug_enabled() || sample_debug_enabled()) {
      std::fprintf(stderr, "[tensor-set][debug] %s: fast base storage failed: %s\n",
                   where ? where : "<unknown>", e.what());
    }
    return std::nullopt;
  }
  if (!base_storage) {
    return std::nullopt;
  }

  const auto copy_template_start = InputStreamDecodeProfileClock::now();
  Sample out = cache.sample_template;
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->fast_sample_copy_template_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - copy_template_start);
  }
  clear_dynamic_sample_fields_for_decode_template(&out);
  const auto sample_meta_start = InputStreamDecodeProfileClock::now();
  if (inputstream_fast_minimal_output_meta_enabled()) {
    fill_output_meta_minimal_from_sample(sample, &out);
  } else {
    fill_output_meta_from_sample(sample, &out);
  }
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->fast_sample_meta_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - sample_meta_start);
  }
  const auto tensor_clear_start = InputStreamDecodeProfileClock::now();
  out.tensors = cache.tensor_templates;
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->fast_tensor_clear_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - tensor_clear_start);
  }
  int first_memory_index = -1;
  std::size_t unique_memory_count = 0U;
  bool multiple_memory_indices = false;

  const bool single_memory_buffer = cache.signature.memory_count == 1U;
  const auto tensor_loop_start = InputStreamDecodeProfileClock::now();
  for (Tensor& tensor : out.tensors) {
    tensor.storage = base_storage;
    tensor.device = base_storage->device;
    tensor.read_only = true;

    // Common tensor-set output case: one GstMemory/segment-backed parent with
    // N logical tensor views.  Reuse the one current-sample storage and the
    // cached per-tensor byte_offset/shape/stride metadata instead of creating
    // six per-memory Storage objects and re-querying segment tables every frame.
    if (!single_memory_buffer && tensor.route.memory_index > 0) {
      try {
        const auto tensor_view_start = InputStreamDecodeProfileClock::now();
        Tensor memory_view = pipeline_internal::tensor_view_from_sample_memory(
            tensor, tensor.route.memory_index, /*keep_holder=*/true);
        if (g_inputstream_decode_profile) {
          g_inputstream_decode_profile->fast_tensor_view_ms += inputstream_decode_profile_ms(
              InputStreamDecodeProfileClock::now() - tensor_view_start);
        }
        tensor.storage = std::move(memory_view.storage);
        tensor.device = tensor.storage ? tensor.storage->device : tensor.device;
      } catch (const std::exception& e) {
        if (tensor_set_debug_enabled() || sample_debug_enabled()) {
          std::fprintf(stderr, "[tensor-set][debug] %s: fast tensor view failed: %s\n",
                       where ? where : "<unknown>", e.what());
        }
        return std::nullopt;
      }
    }

    if (tensor.route.memory_index >= 0) {
      if (first_memory_index < 0) {
        first_memory_index = tensor.route.memory_index;
        unique_memory_count = 1U;
      } else if (tensor.route.memory_index != first_memory_index) {
        multiple_memory_indices = true;
      }
    }
  }
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->fast_tensor_loop_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - tensor_loop_start);
  }

  if (out.segment_name.empty() && !out.tensors.empty()) {
    out.segment_name = out.tensors.front().route.segment_name;
  }
  if (out.stream_label.empty() && !out.tensors.empty()) {
    out.stream_label = out.tensors.front().route.name;
  }
  if (multiple_memory_indices) {
    std::unordered_set<int> unique_memory_indices;
    unique_memory_indices.reserve(out.tensors.size());
    for (const Tensor& tensor : out.tensors) {
      if (tensor.route.memory_index >= 0) {
        unique_memory_indices.insert(tensor.route.memory_index);
      }
    }
    unique_memory_count = unique_memory_indices.size();
  }
  pipeline_internal::record_tensor_packed_view_reuse(out.tensors.size(), unique_memory_count,
                                                     out.owned);
  return out;
}

static std::optional<Sample> tensor_set_from_meta(GstSample* sample, const char* where,
                                                  bool copy_output, InputStream::State* st) {
  if (copy_output || !st || tensor_set_debug_enabled() || sample_debug_enabled() ||
      sample_bytes_enabled()) {
    return tensor_set_from_meta_slow(sample, where, copy_output);
  }

  {
    const auto cache_lock_start = InputStreamDecodeProfileClock::now();
    std::lock_guard<std::mutex> lock(st->output_decode_mu);
    if (g_inputstream_decode_profile) {
      g_inputstream_decode_profile->tensor_cache_lock_ms +=
          inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - cache_lock_start);
    }
    if (st->tensor_set_output_decode_cache.valid &&
        tensor_set_output_decode_fast_key_matches(sample, copy_output,
                                                  st->tensor_set_output_decode_cache.signature)) {
      const auto instantiate_start = InputStreamDecodeProfileClock::now();
      if (auto fast = instantiate_tensor_set_from_cached_decode(
              sample, where, st->tensor_set_output_decode_cache)) {
        if (g_inputstream_decode_profile) {
          g_inputstream_decode_profile->tensor_cache_instantiate_ms +=
              inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() -
                                            instantiate_start);
        }
        return fast;
      }
      if (g_inputstream_decode_profile) {
        g_inputstream_decode_profile->tensor_cache_instantiate_ms +=
            inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - instantiate_start);
      }
      st->tensor_set_output_decode_cache.valid = false;
    }
  }

  TensorSetOutputDecodeSignature sig;
  const auto sig_start = InputStreamDecodeProfileClock::now();
  if (!read_tensor_set_output_decode_signature(sample, copy_output, &sig)) {
    if (g_inputstream_decode_profile) {
      g_inputstream_decode_profile->tensor_sig_ms +=
          inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - sig_start);
    }
    const auto slow_start = InputStreamDecodeProfileClock::now();
    auto slow = tensor_set_from_meta_slow(sample, where, copy_output);
    if (g_inputstream_decode_profile) {
      g_inputstream_decode_profile->tensor_slow_ms +=
          inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - slow_start);
    }
    return slow;
  }
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->tensor_sig_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - sig_start);
  }

  {
    const auto cache_lock_start = InputStreamDecodeProfileClock::now();
    std::lock_guard<std::mutex> lock(st->output_decode_mu);
    if (g_inputstream_decode_profile) {
      g_inputstream_decode_profile->tensor_cache_lock_ms +=
          inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - cache_lock_start);
    }
    if (st->tensor_set_output_decode_cache.valid &&
        st->tensor_set_output_decode_cache.signature == sig) {
      const auto instantiate_start = InputStreamDecodeProfileClock::now();
      if (auto fast = instantiate_tensor_set_from_cached_decode(
              sample, where, st->tensor_set_output_decode_cache)) {
        if (g_inputstream_decode_profile) {
          g_inputstream_decode_profile->tensor_cache_instantiate_ms +=
              inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() -
                                            instantiate_start);
        }
        return fast;
      }
      if (g_inputstream_decode_profile) {
        g_inputstream_decode_profile->tensor_cache_instantiate_ms +=
            inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - instantiate_start);
      }
      st->tensor_set_output_decode_cache.valid = false;
    }
  }

  const auto slow_start = InputStreamDecodeProfileClock::now();
  auto slow = tensor_set_from_meta_slow(sample, where, copy_output);
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->tensor_slow_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - slow_start);
  }
  if (!slow || !sample_has_tensor_list(*slow)) {
    return slow;
  }

  CachedTensorSetOutputDecode cache;
  cache.valid = true;
  cache.signature = std::move(sig);
  cache.sample_template = *slow;
  clear_dynamic_sample_fields_for_decode_template(&cache.sample_template);
  cache.sample_template.tensors.clear();
  if (!slow->tensors.empty() && slow->tensors.front().storage) {
    cache.runtime_segments = slow->tensors.front().storage->sima_segments;
  }
  cache.tensor_templates = slow->tensors;
  for (auto& tensor : cache.tensor_templates) {
    strip_tensor_storage_for_decode_template(&tensor);
  }
  {
    const auto cache_store_start = InputStreamDecodeProfileClock::now();
    std::lock_guard<std::mutex> lock(st->output_decode_mu);
    st->tensor_set_output_decode_cache = std::move(cache);
    if (g_inputstream_decode_profile) {
      g_inputstream_decode_profile->tensor_cache_store_ms +=
          inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - cache_store_start);
    }
  }
  return slow;
}

Sample decode_sample_from_inputstream_state(InputStream::State& st, GstSample* sample,
                                            const char* where) {
  const bool decode_profile_enabled = inputstream_decode_profile_enabled();
  InputStreamDecodeProfileSample decode_profile{};
  InputStreamDecodeProfileScope decode_profile_scope(decode_profile_enabled ? &decode_profile
                                                                            : nullptr);
  const auto decode_start = InputStreamDecodeProfileClock::now();

  const auto restore_buffer_meta_start = InputStreamDecodeProfileClock::now();
  maybe_restore_cached_preprocess_meta(st, sample);
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->restore_buffer_meta_ms += inputstream_decode_profile_ms(
        InputStreamDecodeProfileClock::now() - restore_buffer_meta_start);
  }

  const auto envelope_start = InputStreamDecodeProfileClock::now();
  Sample out =
      sample_from_gst_envelope(sample, where, st.opt.copy_output, &st.opt.output_override, &st);
  pipeline_internal::mark_sample_producer_stream_lifetime(out, st.lifetime_token);
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->envelope_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - envelope_start);
  }

  const auto restore_sample_meta_start = InputStreamDecodeProfileClock::now();
  maybe_restore_cached_preprocess_meta_on_sample(st, &out);
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->restore_sample_meta_ms += inputstream_decode_profile_ms(
        InputStreamDecodeProfileClock::now() - restore_sample_meta_start);
  }

  const auto apply_port_start = InputStreamDecodeProfileClock::now();
  apply_default_port_name(out, st.src_opt);
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->apply_port_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - apply_port_start);
    g_inputstream_decode_profile->total_decode_ms =
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - decode_start);
  }

  if (decode_profile_enabled) {
    log_inputstream_decode_profile(decode_profile);
  }
  return out;
}

Sample InputStream::pull(int timeout_ms) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::pull: stream is closed");
  }
  if (!state_->appsink) {
    throw std::runtime_error("InputStream::pull: appsink not available (no Output)");
  }
  const int timeout = (timeout_ms >= 0) ? timeout_ms : state_->opt.timeout_ms;
  const bool timings = state_->timing_enabled;
  std::chrono::steady_clock::time_point t_wait_start{};
  if (timings)
    t_wait_start = std::chrono::steady_clock::now();
  auto sample_opt = pipeline_internal::try_pull_sample_sliced(
      state_->pipeline, state_->appsink, timeout, state_->diag, "InputStream::pull");
  std::chrono::steady_clock::time_point t_wait_end{};
  if (timings)
    t_wait_end = std::chrono::steady_clock::now();
  if (!sample_opt.has_value()) {
    if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_DEBUG", false)) {
      std::fprintf(stderr, "[DBG] InputStream::pull timeout\n%s", diagnostics_summary().c_str());
    }
    throw std::runtime_error("InputStream::pull: timeout waiting for output");
  }
  GstSample* sample = sample_opt.value();
  const std::int64_t inflight = state_->inflight.load(std::memory_order_relaxed);
  if (inflight > 0) {
    state_->inflight.fetch_sub(1, std::memory_order_relaxed);
  }
  const bool decode_profile_enabled = inputstream_decode_profile_enabled();
  std::chrono::steady_clock::time_point t_decode_start{};
  if (timings || decode_profile_enabled)
    t_decode_start = std::chrono::steady_clock::now();
  InputStreamDecodeProfileSample decode_profile{};
  InputStreamDecodeProfileScope decode_profile_scope(decode_profile_enabled ? &decode_profile
                                                                            : nullptr);
  const auto restore_buffer_meta_start = InputStreamDecodeProfileClock::now();
  maybe_restore_cached_preprocess_meta(*state_, sample);
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->restore_buffer_meta_ms += inputstream_decode_profile_ms(
        InputStreamDecodeProfileClock::now() - restore_buffer_meta_start);
  }
  const auto envelope_start = InputStreamDecodeProfileClock::now();
  Sample out = sample_from_gst_envelope(sample, "InputStream::pull", state_->opt.copy_output,
                                        &state_->opt.output_override, state_.get());
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->envelope_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - envelope_start);
  }
  const auto restore_sample_meta_start = InputStreamDecodeProfileClock::now();
  maybe_restore_cached_preprocess_meta_on_sample(*state_, &out);
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->restore_sample_meta_ms += inputstream_decode_profile_ms(
        InputStreamDecodeProfileClock::now() - restore_sample_meta_start);
  }
  const auto apply_port_start = InputStreamDecodeProfileClock::now();
  apply_default_port_name(out, state_->src_opt);
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->apply_port_ms +=
        inputstream_decode_profile_ms(InputStreamDecodeProfileClock::now() - apply_port_start);
  }
  if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_POOL_DEBUG", false)) {
    auto log_tensor_storage = [](const simaai::neat::Sample& s) {
      if (s.kind == SampleKind::TensorSet && !s.tensors.empty()) {
        std::size_t gst_tensors = 0;
        for (const auto& tensor : s.tensors) {
          if (tensor.storage && tensor.storage->kind == simaai::neat::StorageKind::GstSample) {
            gst_tensors++;
          }
        }
        const auto& first = s.tensors.front();
        const bool has_holder = first.storage && static_cast<bool>(first.storage->holder);
        std::fprintf(
            stderr,
            "[INPUTSTREAM] out tensors=%zu gstsample_tensors=%zu first_storage_kind=%d holder=%s\n",
            s.tensors.size(), gst_tensors,
            first.storage ? static_cast<int>(first.storage->kind) : -1,
            has_holder ? "true" : "false");
        return;
      }
      if (!s.tensor || !s.tensor->storage)
        return;
      const bool has_holder = static_cast<bool>(s.tensor->storage->holder);
      std::fprintf(stderr, "[INPUTSTREAM] out tensor storage_kind=%d holder=%s\n",
                   static_cast<int>(s.tensor->storage->kind), has_holder ? "true" : "false");
    };
    if (sample_is_multi_output(out)) {
      std::size_t gst_fields = 0;
      for (const auto& field : out.fields) {
        if (field.tensor && field.tensor->storage &&
            field.tensor->storage->kind == simaai::neat::StorageKind::GstSample) {
          gst_fields++;
        }
      }
      std::fprintf(stderr, "[INPUTSTREAM] out bundle fields=%zu gstsample_fields=%zu\n",
                   out.fields.size(), gst_fields);
      if (!out.fields.empty()) {
        log_tensor_storage(out.fields.front());
      }
    } else {
      log_tensor_storage(out);
    }
  }
  std::chrono::steady_clock::time_point t_decode_end{};
  if (timings || decode_profile_enabled)
    t_decode_end = std::chrono::steady_clock::now();
  if (g_inputstream_decode_profile) {
    g_inputstream_decode_profile->total_decode_ms =
        inputstream_decode_profile_ms(t_decode_end - t_decode_start);
  }
  const bool pool_dbg = pipeline_internal::env_bool("SIMA_INPUTSTREAM_POOL_DEBUG", false);
  GstBuffer* buf = gst_sample_get_buffer(sample);
  const guint sample_ref = GST_MINI_OBJECT_REFCOUNT_VALUE(sample);
  const guint buf_ref = buf ? GST_MINI_OBJECT_REFCOUNT_VALUE(buf) : 0u;
  const gboolean buf_writable = buf ? gst_buffer_is_writable(buf) : FALSE;
  const bool weak_dbg = pipeline_internal::env_bool("SIMA_INPUTSTREAM_WEAKREF_DEBUG", false);
  if (weak_dbg) {
    auto* sample_tag = new WeakRefTag{next_weak_id(), "sample"};
    gst_mini_object_weak_ref(GST_MINI_OBJECT(sample), mini_object_weak_notify, sample_tag);
    if (buf) {
      auto* buf_tag = new WeakRefTag{next_weak_id(), "buffer"};
      gst_mini_object_weak_ref(GST_MINI_OBJECT(buf), mini_object_weak_notify, buf_tag);
    }
  }

  if (pool_dbg) {
    std::fprintf(stderr,
                 "[INPUTSTREAM] pre unref sample_refcnt=%u buffer=%p refcnt=%u "
                 "pool=%p buf_writable=%s\n",
                 sample_ref, static_cast<void*>(buf), buf_ref,
                 buf ? static_cast<void*>(buf->pool) : nullptr, buf_writable ? "true" : "false");
  }

  gst_sample_unref(sample);
  if (decode_profile_enabled) {
    log_inputstream_decode_profile(decode_profile);
  }
  if (timings) {
    state_->pull_count.fetch_add(1, std::memory_order_relaxed);
    state_->pull_wait_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_wait_end - t_wait_start)
                .count()),
        std::memory_order_relaxed);
    state_->decode_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_decode_end - t_decode_start)
                .count()),
        std::memory_order_relaxed);
  }
  return out;
}

void InputStream::pull_and_discard(int timeout_ms) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::pull_and_discard: stream is closed");
  }
  if (!state_->appsink) {
    throw std::runtime_error("InputStream::pull_and_discard: appsink not available (no Output)");
  }
  const int timeout = (timeout_ms >= 0) ? timeout_ms : state_->opt.timeout_ms;
  const bool timings = state_->timing_enabled;
  std::chrono::steady_clock::time_point t_wait_start{};
  if (timings)
    t_wait_start = std::chrono::steady_clock::now();
  auto sample_opt = pipeline_internal::try_pull_sample_sliced(
      state_->pipeline, state_->appsink, timeout, state_->diag, "InputStream::pull_and_discard");
  std::chrono::steady_clock::time_point t_wait_end{};
  if (timings)
    t_wait_end = std::chrono::steady_clock::now();
  if (!sample_opt.has_value()) {
    if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_DEBUG", false)) {
      std::fprintf(stderr, "[DBG] InputStream::pull_and_discard timeout\n%s",
                   diagnostics_summary().c_str());
    }
    throw std::runtime_error("InputStream::pull_and_discard: timeout waiting for output");
  }

  GstSample* sample = sample_opt.value();
  const std::int64_t inflight = state_->inflight.load(std::memory_order_relaxed);
  if (inflight > 0) {
    state_->inflight.fetch_sub(1, std::memory_order_relaxed);
  }
  const bool pool_dbg = pipeline_internal::env_bool("SIMA_INPUTSTREAM_POOL_DEBUG", false);
  const bool weak_dbg = pipeline_internal::env_bool("SIMA_INPUTSTREAM_WEAKREF_DEBUG", false);
  GstBuffer* buf = gst_sample_get_buffer(sample);
  GstBuffer* buf_ref = nullptr;
  if (weak_dbg) {
    auto* sample_tag = new WeakRefTag{next_weak_id(), "discard_sample"};
    gst_mini_object_weak_ref(GST_MINI_OBJECT(sample), mini_object_weak_notify, sample_tag);
    if (buf) {
      auto* buf_tag = new WeakRefTag{next_weak_id(), "discard_buffer"};
      gst_mini_object_weak_ref(GST_MINI_OBJECT(buf), mini_object_weak_notify, buf_tag);
    }
  }
  if (pool_dbg && buf) {
    const guint sample_ref = GST_MINI_OBJECT_REFCOUNT_VALUE(sample);
    const guint buf_refcnt = GST_MINI_OBJECT_REFCOUNT_VALUE(buf);
    buf_ref = gst_buffer_ref(buf);
    std::fprintf(stderr,
                 "[INPUTSTREAM] discard pre unref sample=%p sample_refcnt=%u buffer=%p "
                 "buf_refcnt=%u pool=%p\n",
                 static_cast<void*>(sample), sample_ref, static_cast<void*>(buf), buf_refcnt,
                 static_cast<void*>(buf->pool));
  }
  gst_sample_unref(sample);
  if (pool_dbg && buf_ref) {
    const guint after = GST_MINI_OBJECT_REFCOUNT_VALUE(buf_ref);
    std::fprintf(stderr,
                 "[INPUTSTREAM] discard after sample_unref buffer=%p buf_refcnt_with_probe=%u "
                 "pool=%p\n",
                 static_cast<void*>(buf_ref), after, static_cast<void*>(buf_ref->pool));
    gst_buffer_unref(buf_ref);
  }

  if (timings) {
    state_->pull_count.fetch_add(1, std::memory_order_relaxed);
    state_->pull_wait_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_wait_end - t_wait_start)
                .count()),
        std::memory_order_relaxed);
  }
}

Sample InputStream::push_and_pull_holder(const std::shared_ptr<void>& holder, int timeout_ms) {
  push_holder(holder);
  return pull(timeout_ms);
}
} // namespace simaai::neat
