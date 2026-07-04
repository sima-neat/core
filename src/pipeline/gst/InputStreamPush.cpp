#include "InputStreamInternal.h"
#include "gst/SimaTensorSetMetaAbi.h"
#include "pipeline/internal/SampleUtil.h"

namespace simaai::neat {

namespace {

struct SeqOverrides {
  std::optional<int64_t> input_seq;
  std::optional<int64_t> orig_input_seq;
};

struct MessageMetaOverrides {
  std::optional<int64_t> frame_id;
  std::optional<std::string> stream_id;
  std::optional<std::string> stream_label;
};

SeqOverrides next_seq_overrides(InputStream::State& st) {
  SeqOverrides out;
  out.input_seq = std::optional<int64_t>(st.next_input_seq.fetch_add(1, std::memory_order_relaxed));
  out.orig_input_seq = out.input_seq;
  return out;
}

SeqOverrides seq_overrides_for_message(InputStream::State& st, const Sample& msg) {
  SeqOverrides out;
  out.input_seq =
      (msg.input_seq >= 0)
          ? std::optional<int64_t>(msg.input_seq)
          : std::optional<int64_t>(st.next_input_seq.fetch_add(1, std::memory_order_relaxed));
  out.orig_input_seq =
      (msg.orig_input_seq >= 0) ? std::optional<int64_t>(msg.orig_input_seq) : out.input_seq;
  return out;
}

MessageMetaOverrides message_meta_overrides(const Sample& msg) {
  MessageMetaOverrides out;
  out.frame_id = (msg.frame_id >= 0) ? std::optional<int64_t>(msg.frame_id) : std::nullopt;
  out.stream_id = msg.stream_id.empty() ? std::nullopt : std::optional<std::string>(msg.stream_id);
  const bool tensor_payload = sample_has_tensor_list(msg);
  if (!tensor_payload) {
    if (!msg.stream_label.empty()) {
      out.stream_label = msg.stream_label;
    } else if (!msg.port_name.empty()) {
      out.stream_label = msg.port_name;
    }
  }
  return out;
}

bool cpu_owned_zero_copy_input_enabled() {
  const char* raw = std::getenv("SIMA_INPUTSTREAM_CPU_ZC");
  return raw && *raw && std::strcmp(raw, "0") != 0;
}

bool allow_inputstream_cpu_to_device_copy() {
  const char* raw = std::getenv("SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY");
  return raw && *raw && std::strcmp(raw, "0") != 0;
}

bool tensor_requires_cpu_to_device_copy_for_push(const simaai::neat::Tensor& tensor) {
  if (!tensor.storage) {
    return false;
  }
  if (tensor.storage->kind == simaai::neat::StorageKind::CpuOwned ||
      tensor.storage->kind == simaai::neat::StorageKind::CpuExternal) {
    return true;
  }
  // Device-created tensors are GstSample-backed too.  Treat a GstSample as
  // CPU-backed only when it has no SiMa allocator target; otherwise appsrc can
  // forward the original device-visible GstMemory without a hidden memcpy.
  if (tensor.storage->kind == simaai::neat::StorageKind::GstSample) {
    if (!tensor.storage->sima_segments.empty()) {
      return false;
    }
    return tensor.storage->sima_mem_target_flags == 0 &&
           tensor.device.type == simaai::neat::DeviceType::CPU;
  }
  return tensor.device.type == simaai::neat::DeviceType::CPU;
}

// Set-complete scan: true if ANY tensor in the sample (tensor-list, single
// tensor, or nested bundle field) is CPU-backed and would need a copy to be
// pushed into a device-visible route. The previous inline guards only checked
// the front tensor, which misses mixed CPU/device tensor sets.
bool sample_has_cpu_backed_tensor_for_device_push(const simaai::neat::Sample& sample) {
  if (sample_has_tensor_list(sample)) {
    for (const auto& t : sample.tensors) {
      if (tensor_requires_cpu_to_device_copy_for_push(t)) {
        return true;
      }
    }
  }
  if (sample.tensor.has_value() && tensor_requires_cpu_to_device_copy_for_push(*sample.tensor)) {
    return true;
  }
  for (const auto& field : sample.fields) {
    if (sample_has_cpu_backed_tensor_for_device_push(field)) {
      return true;
    }
  }
  return false;
}

// Single source of truth for the actionable guidance shown when a CPU-backed
// tensor reaches a device-visible route. Reused by the raw-path guard and the
// tensor-envelope failure path so both surface identical guidance.
const char* device_visible_push_guard_message() {
  return "CPU-backed Tensor pushed into a device-visible EV74/DMS route. Create the Tensor "
         "with TensorMemory::EV74 (or TensorMemory::MLA for DMS routes) so the copy is owned by "
         "Tensor creation, or set SIMA_ALLOW_INPUTSTREAM_CPU_TO_EV74_COPY=1 to allow the slow "
         "compatibility copy in runner.push().";
}

// Shared fail-fast guard. Throws the actionable message when the route demands
// device-visible input, the compat-copy escape is off, and any tensor in the
// pushed sample is CPU-backed.
void enforce_device_visible_push_or_throw(bool require_device_visible,
                                          const simaai::neat::Sample& sample, const char* where) {
  if (!require_device_visible || allow_inputstream_cpu_to_device_copy()) {
    return;
  }
  if (sample_has_cpu_backed_tensor_for_device_push(sample)) {
    throw std::runtime_error(std::string(where) + ": " + device_visible_push_guard_message());
  }
}

bool allow_cross_run_gstsample_push() {
  return pipeline_internal::env_bool("SIMA_ALLOW_CROSS_RUN_GSTSAMPLE_PUSH", false);
}

void enforce_live_gstsample_producer_or_throw(const simaai::neat::Sample& sample, const char* where,
                                              bool allow_graph_internal_zero_copy_input) {
  if (allow_cross_run_gstsample_push()) {
    static std::atomic<bool> warned{false};
    bool expected = false;
    if (warned.compare_exchange_strong(expected, true)) {
      std::fprintf(stderr,
                   "[WARN] SIMA_ALLOW_CROSS_RUN_GSTSAMPLE_PUSH=1 is enabling unsafe legacy "
                   "zero-copy frame transfer between running graphs; frame memory may be stale or "
                   "recycled.\n");
    }
    return;
  }
  if (pipeline_internal::sample_has_device_gstsample_producer_lifetime(sample,
                                                                       /*require_expired=*/false)) {
    if (allow_graph_internal_zero_copy_input &&
        !pipeline_internal::sample_has_device_gstsample_producer_lifetime(
            sample,
            /*require_expired=*/true)) {
      return;
    }
    std::string reason;
    if (!pipeline_internal::sample_has_transferable_zero_copy_loan(sample, &reason)) {
      throw std::runtime_error(pipeline_internal::cross_run_zero_copy_sample_error(where) +
                               (reason.empty() ? std::string{} : " Reason: " + reason + "."));
    }
  }
}

const Tensor* first_tensor_for_preprocess_meta(const Sample& sample) {
  if (sample_has_tensor_list(sample) && !sample.tensors.empty()) {
    return &sample.tensors.front();
  }
  if (sample.tensor.has_value()) {
    return &*sample.tensor;
  }
  for (const auto& field : sample.fields) {
    if (const Tensor* nested = first_tensor_for_preprocess_meta(field)) {
      return nested;
    }
  }
  return nullptr;
}

void cache_preprocess_meta(InputStream::State& st, const Sample& msg,
                           const std::optional<int64_t>& input_seq_override,
                           const std::optional<int64_t>& orig_input_seq_override) {
  const bool debug = pipeline_internal::env_bool("SIMA_PREPROC_META_TRACE", false);
  const Tensor* tensor = first_tensor_for_preprocess_meta(msg);
  if (!tensor) {
    if (debug) {
      std::fprintf(stderr, "[PREPROC_META_TRACE] cache skip=no_tensor\n");
    }
    return;
  }
  const std::shared_ptr<void> holder = pipeline_internal::holder_from_tensor(*tensor);
  if (!holder) {
    if (debug) {
      std::fprintf(stderr,
                   "[PREPROC_META_TRACE] cache skip=no_holder semantic=%d storage_kind=%d\n",
                   tensor->semantic.preprocess.has_value() ? 1 : 0,
                   tensor->storage ? static_cast<int>(tensor->storage->kind) : -1);
    }
    return;
  }
  GstBuffer* buffer = pipeline_internal::buffer_from_tensor_holder(holder);
  if (!buffer) {
    if (debug) {
      std::fprintf(stderr,
                   "[PREPROC_META_TRACE] cache skip=no_buffer semantic=%d storage_kind=%d\n",
                   tensor->semantic.preprocess.has_value() ? 1 : 0,
                   tensor->storage ? static_cast<int>(tensor->storage->kind) : -1);
    }
    return;
  }
  auto meta = read_simaai_preprocess_meta(buffer);
  const bool buffer_meta = meta.has_value();
  gst_buffer_unref(buffer);
  if (!meta.has_value() && tensor->semantic.preprocess.has_value()) {
    meta = tensor->semantic.preprocess;
  }
  if (!meta.has_value()) {
    if (debug) {
      std::fprintf(stderr,
                   "[PREPROC_META_TRACE] cache skip=no_meta buffer_meta=%d semantic=%d key_in=%lld "
                   "key_orig=%lld storage_kind=%d\n",
                   buffer_meta ? 1 : 0, tensor->semantic.preprocess.has_value() ? 1 : 0,
                   static_cast<long long>(input_seq_override.value_or(-1)),
                   static_cast<long long>(orig_input_seq_override.value_or(-1)),
                   tensor->storage ? static_cast<int>(tensor->storage->kind) : -1);
    }
    return;
  }

  const int64_t key = orig_input_seq_override.has_value()
                          ? *orig_input_seq_override
                          : (input_seq_override.has_value() ? *input_seq_override : -1);
  if (key < 0) {
    if (debug) {
      std::fprintf(stderr,
                   "[PREPROC_META_TRACE] cache skip=no_key buffer_meta=%d semantic=%d key_in=%lld "
                   "key_orig=%lld\n",
                   buffer_meta ? 1 : 0, tensor->semantic.preprocess.has_value() ? 1 : 0,
                   static_cast<long long>(input_seq_override.value_or(-1)),
                   static_cast<long long>(orig_input_seq_override.value_or(-1)));
    }
    return;
  }

  std::lock_guard<std::mutex> lock(st.preprocess_meta_mu);
  const bool inserted =
      st.preprocess_meta_by_input_seq.find(key) == st.preprocess_meta_by_input_seq.end();
  st.preprocess_meta_by_input_seq[key] = *meta;
  if (debug) {
    std::fprintf(stderr,
                 "[PREPROC_META_TRACE] cache stored key=%lld buffer_meta=%d semantic=%d orig=%dx%d "
                 "resized=%dx%d cache_size=%zu\n",
                 static_cast<long long>(key), buffer_meta ? 1 : 0,
                 tensor->semantic.preprocess.has_value() ? 1 : 0, meta->original_width,
                 meta->original_height, meta->resized_width, meta->resized_height,
                 st.preprocess_meta_by_input_seq.size());
  }
  if (inserted) {
    st.preprocess_meta_order.push_back(key);
  }
  constexpr std::size_t kMaxCachedPreprocessMeta = 256U;
  while (st.preprocess_meta_order.size() > kMaxCachedPreprocessMeta) {
    const int64_t evict = st.preprocess_meta_order.front();
    st.preprocess_meta_order.pop_front();
    st.preprocess_meta_by_input_seq.erase(evict);
  }
}

void validate_spec_with_limits(const InputStream::State& st, const SampleSpec& spec,
                               const char* where) {
  if (spec.kind != SampleMediaKind::RawVideo && spec.kind != SampleMediaKind::Tensor)
    return;
  if (spec.tensor_envelope_transport)
    return;

  const char* tag = where ? where : "InputStream::try_push";
  const int max_w = st.shape_limits.max_width;
  const int max_h = st.shape_limits.max_height;
  const int max_d = st.shape_limits.max_depth;

  auto fail_if_over = [&](const char* field, int value, int max_val) {
    if (value <= 0 || max_val <= 0 || value <= max_val)
      return;
    std::ostringstream oss;
    oss << tag << ": " << field << " exceeds effective max (" << value << " > " << max_val << ")";
    throw std::invalid_argument(oss.str());
  };

  fail_if_over("width", spec.width, max_w);
  fail_if_over("height", spec.height, max_h);
  if (spec.depth > 0) {
    fail_if_over("depth", spec.depth, max_d);
  }
}

std::function<void(GstBuffer**)>
make_prepare_for_spec(const SampleSpec& spec, const char* where,
                      GstBuffer* source_preproc_meta_buffer = nullptr,
                      const TensorList* tensor_set_meta_tensors = nullptr,
                      std::optional<PreprocessRuntimeMeta> tensor_preprocess_meta = std::nullopt) {
  return [spec, where, source_preproc_meta_buffer, tensor_set_meta_tensors,
          tensor_preprocess_meta](GstBuffer** buf) {
    if (!buf || !*buf) {
      return;
    }
    if (spec.kind == SampleMediaKind::RawVideo) {
      apply_video_meta_or_throw(buf, spec, where);
    } else if (spec.kind == SampleMediaKind::Tensor) {
      apply_tensor_size_or_throw(buf, spec, where);
      if (tensor_set_meta_tensors && !tensor_set_meta_tensors->empty() &&
          !gst_buffer_get_custom_meta(*buf, SIMA_TENSOR_SET_META_NAME)) {
        pipeline_internal::attach_tensor_set_meta_from_tensors(*buf, *tensor_set_meta_tensors);
      }
    } else {
      return;
    }

    if (source_preproc_meta_buffer && has_simaai_preprocess_meta(source_preproc_meta_buffer)) {
      std::string copy_err;
      if (!copy_simaai_preprocess_meta(*buf, source_preproc_meta_buffer, &copy_err)) {
        throw std::runtime_error(std::string(where ? where : "InputStream::make_prepare_for_spec") +
                                 ": failed to preserve preprocess metadata: " + copy_err);
      }
      // Plan 1: framework owns preproc_axis_perm. Plugin-authored geometry was
      // copied above; merge the user's resolved perm onto it without
      // overwriting other fields.
      if (tensor_preprocess_meta.has_value() && !tensor_preprocess_meta->axis_perm.empty()) {
        merge_simaai_preprocess_axis_perm(*buf, tensor_preprocess_meta->axis_perm);
      }
      return;
    }

    if (!tensor_preprocess_meta.has_value()) {
      return;
    }

    if (!write_simaai_preprocess_meta(*buf, *tensor_preprocess_meta)) {
      throw std::runtime_error(std::string(where ? where : "InputStream::make_prepare_for_spec") +
                               ": failed to apply tensor preprocess metadata");
    }
  };
}

SampleSpec derive_mat_spec_or_throw(const cv::Mat& contiguous, const InputOptions& relaxed) {
  SampleSpec spec;
  const bool float_input = contiguous.depth() == CV_32F && contiguous.channels() > 0;
  const std::string relaxed_media = resolve_input_media_type(relaxed);
  spec.media_type = relaxed_media.empty()
                        ? (float_input ? "application/vnd.simaai.tensor" : "video/x-raw")
                        : relaxed_media;
  spec.required_bytes_actual = 0;
  if (spec.media_type == "video/x-raw") {
    spec.kind = SampleMediaKind::RawVideo;
    std::string fmt = upper_copy(relaxed.format);
    if (fmt.empty()) {
      fmt = (contiguous.channels() == 1) ? "GRAY8" : "BGR";
    }
    if (fmt == "GRAY")
      fmt = "GRAY8";
    if (fmt != "RGB" && fmt != "BGR" && fmt != "GRAY8") {
      throw std::invalid_argument("InputStream::try_push: unsupported video format");
    }
    if ((fmt == "RGB" || fmt == "BGR") && contiguous.type() != CV_8UC3) {
      throw std::invalid_argument("InputStream::try_push: expected CV_8UC3");
    }
    if (fmt == "GRAY8" && contiguous.type() != CV_8UC1) {
      throw std::invalid_argument("InputStream::try_push: expected CV_8UC1");
    }
    if (fmt == "NV12" || fmt == "I420") {
      throw std::invalid_argument(
          "InputStream::try_push: planar video requires simaai::neat::Tensor planes");
    }
    spec.format = fmt;
    spec.width = contiguous.cols;
    spec.height = contiguous.rows;
    spec.depth = contiguous.channels();
    PlaneInfo plane;
    plane.role = simaai::neat::PlaneRole::Unknown;
    plane.width = spec.width;
    plane.height = spec.height;
    plane.stride_bytes = static_cast<int64_t>(contiguous.step[0]);
    plane.offset_bytes = 0;
    plane.size_bytes = static_cast<size_t>(plane.stride_bytes) * static_cast<size_t>(spec.height);
    if (plane.size_bytes == 0) {
      throw std::invalid_argument("InputStream::try_push: invalid video byte size");
    }
    spec.required_bytes_actual = plane.size_bytes;
    spec.planes = {plane};
    spec.dtype = TensorDType::UInt8;
  } else if (spec.media_type == "application/vnd.simaai.tensor") {
    spec.kind = SampleMediaKind::Tensor;
    if (contiguous.depth() != CV_32F || contiguous.channels() <= 0) {
      throw std::invalid_argument(
          "InputStream::try_push: tensor input must be a non-empty CV_32F matrix");
    }
    const std::string requested_format =
        normalize_caps_format_for_media("application/vnd.simaai.tensor", relaxed.format.str());
    const std::string requested_up = upper_copy(requested_format);
    const bool fp32_like = requested_up.empty() || requested_up == "FP32" ||
                           requested_up == "FLOAT32" || requested_up == "EVXX_FLOAT32";
    if (!fp32_like) {
      throw std::invalid_argument(
          "InputStream::try_push: cv::Mat tensor input supports FP32/EVXX_FLOAT32 only; "
          "pass simaai::neat::Tensor for BF16 or quantized tensor ingress");
    }
    spec.required_bytes_actual = contiguous.total() * contiguous.elemSize();
    spec.dtype = TensorDType::Float32;
    spec.format = requested_format.empty() ? std::string("FP32") : requested_format;
    spec.width = contiguous.cols;
    spec.height = contiguous.rows;
    spec.depth = contiguous.channels();
    spec.layout = TensorLayout::Unknown;
    if (spec.depth > 1) {
      spec.shape = {spec.height, spec.width, spec.depth};
    } else {
      spec.shape = {spec.height, spec.width};
    }
  } else {
    throw std::invalid_argument("InputStream::try_push: unsupported media_type");
  }

  spec.fps_n = relaxed.fps_n;
  spec.fps_d = relaxed.fps_d;
  spec.caps_key = capkey_from_spec(spec);
  spec.caps_string = caps_string_from_spec(spec);
  return spec;
}

std::function<void(GstBuffer**)>
make_prepare_for_mat_spec(const SampleSpec& spec, const cv::Mat& contiguous, size_t input_bytes) {
  if (spec.kind == SampleMediaKind::RawVideo) {
    SampleSpec meta_spec = spec;
    meta_spec.width = contiguous.cols;
    meta_spec.height = contiguous.rows;
    meta_spec.depth = contiguous.channels();
    meta_spec.planes[0].stride_bytes = static_cast<int64_t>(contiguous.step[0]);
    meta_spec.planes[0].size_bytes = input_bytes;
    meta_spec.required_bytes_actual = input_bytes;
    return [meta_spec](GstBuffer** buf) {
      apply_video_meta_or_throw(buf, meta_spec, "InputStream::try_push(mat)");
    };
  }
  if (spec.kind == SampleMediaKind::Tensor) {
    SampleSpec size_spec = spec;
    size_spec.required_bytes_actual = input_bytes;
    return [size_spec](GstBuffer** buf) {
      apply_tensor_size_or_throw(buf, size_spec, "InputStream::try_push(mat)");
    };
  }
  return {};
}

std::function<size_t(uint8_t*, size_t)>
make_tensor_copy_fill(const simaai::neat::Tensor& input, size_t input_bytes, const char* where) {
  return [&input, input_bytes, where](uint8_t* dst, size_t dst_bytes) -> size_t {
    const size_t copy_bytes = std::min(input_bytes, dst_bytes);
    if (!input.storage || copy_bytes == 0)
      return 0;
    std::string copy_err;
    if (!pipeline_internal::copy_tensor_payload_to(input, dst, copy_bytes, &copy_err)) {
      const char* tag = where ? where : "InputStream::try_push";
      throw std::runtime_error(std::string(tag) + ": " +
                               (copy_err.empty() ? std::string("tensor copy failed") : copy_err));
    }
    return copy_bytes;
  };
}

void validate_holder_video_meta_or_throw(const InputStream::State& st, GstBuffer* buf) {
  if (!st.current_key.has_value() || st.current_key->kind != SampleMediaKind::RawVideo) {
    return;
  }
  const CapKey& key = *st.current_key;
  SampleSpec spec;
  spec.kind = SampleMediaKind::RawVideo;
  spec.media_type = "video/x-raw";
  spec.format = key.format;
  spec.width = key.width;
  spec.height = key.height;
  std::string spec_err;
  pipeline_internal::canonicalize_sample_spec(&spec, &spec_err);
  std::string meta_err;
  if (!pipeline_internal::validate_buffer_video_meta(buf, spec, &meta_err)) {
    throw std::runtime_error("InputStream::try_push_holder: " +
                             (meta_err.empty() ? std::string("video meta mismatch") : meta_err));
  }
}

void apply_holder_spec_and_meta_or_throw(
    GstBuffer** buffer, const SampleSpec& spec, const MessageMetaOverrides& meta,
    const std::optional<int64_t>& input_seq_override,
    const std::optional<int64_t>& orig_input_seq_override,
    const SampleTimingOverrides& timing_override, const InputOptions& src_opt,
    InputBufferPoolGuard& guard, const char* where,
    const std::optional<PreprocessRuntimeMeta>& tensor_preprocess_meta = std::nullopt) {
  GstBuffer* const original = buffer ? *buffer : nullptr;
  const bool source_has_preproc_meta = has_simaai_preprocess_meta(original);

  if (spec.kind == SampleMediaKind::RawVideo) {
    apply_video_meta_or_throw(buffer, spec, where);
  } else if (spec.kind == SampleMediaKind::Tensor) {
    apply_tensor_size_or_throw(buffer, spec, where);
  }

  if (original && *buffer && *buffer != original && source_has_preproc_meta) {
    std::string copy_err;
    if (!copy_simaai_preprocess_meta(*buffer, original, &copy_err)) {
      throw std::runtime_error(std::string(where ? where : "InputStream::apply_holder_spec") +
                               ": failed to preserve preprocess metadata: " + copy_err);
    }
  }

  *buffer = attach_simaai_meta_inplace(*buffer, src_opt, guard, where, meta.frame_id,
                                       StreamIdOverride{meta.stream_id},
                                       BufferNameOverride{meta.stream_label});
  if (!*buffer) {
    throw std::runtime_error(std::string(where ? where : "InputStream::push") +
                             ": failed to attach GstSimaMeta");
  }
  if (!update_simaai_meta_fields(*buffer, meta.frame_id, input_seq_override,
                                 orig_input_seq_override, meta.stream_id, meta.stream_label,
                                 timing_override.pts_ns)) {
    throw std::runtime_error(std::string(where ? where : "InputStream::apply_holder_spec") +
                             ": failed to write GstSimaMeta fields");
  }
  if (!write_sample_timing_to_gst_buffer(*buffer, timing_override)) {
    throw std::runtime_error(std::string(where ? where : "InputStream::apply_holder_spec") +
                             ": failed to write sample timing metadata");
  }
  if (!has_simaai_preprocess_meta(*buffer) && tensor_preprocess_meta.has_value()) {
    if (!write_simaai_preprocess_meta(*buffer, *tensor_preprocess_meta)) {
      throw std::runtime_error(std::string(where ? where : "InputStream::apply_holder_spec") +
                               ": failed to apply tensor preprocess metadata");
    }
  }
  if (!has_simaai_preprocess_meta(*buffer) && spec.width > 0 && spec.height > 0) {
    (void)apply_simaai_preprocess_meta_template(*buffer, src_opt, spec.width, spec.height);
  }
}

bool push_holder_buffer_with_appsrc(InputStream::State& st, GstBuffer* buffer, const char* where,
                                    const char* push_fail_unref_tag, bool record_timings,
                                    bool log_refcount_on_push, const Sample* fail_msg,
                                    const SampleSpec* fail_spec,
                                    const std::optional<int64_t>& input_seq_override,
                                    const std::optional<int64_t>& orig_input_seq_override) {
  std::chrono::steady_clock::time_point t_push_start{};
  if (record_timings || inputstream_push_timing_enabled()) {
    t_push_start = std::chrono::steady_clock::now();
  }
  if (log_refcount_on_push) {
    log_push_refcount(where, buffer);
  }
  if (fail_msg) {
    pipeline_internal::attach_zero_copy_loans_to_gst_buffer(buffer, *fail_msg);
  }
  if (holder_debug_enabled()) {
    const bool has_tensor_set =
        gst_buffer_get_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME) != nullptr;
    std::fprintf(stderr,
                 "[HOLDER] push_holder_buffer_with_appsrc before_push where=%s buffer=%p bytes=%zu "
                 "mems=%u has_tensor_set=%d\n",
                 where ? where : "<null>", static_cast<void*>(buffer),
                 static_cast<std::size_t>(gst_buffer_get_size(buffer)), gst_buffer_n_memory(buffer),
                 has_tensor_set ? 1 : 0);
  }
  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(st.appsrc), buffer);
  if (holder_debug_enabled()) {
    std::fprintf(stderr, "[HOLDER] push_holder_buffer_with_appsrc after_push where=%s ret=%d\n",
                 where ? where : "<null>", static_cast<int>(ret));
  }
  const auto t_push_end = std::chrono::steady_clock::now();

  if (record_timings) {
    st.push_count.fetch_add(1, std::memory_order_relaxed);
    st.push_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start)
                .count()),
        std::memory_order_relaxed);
  }
  if (inputstream_push_timing_enabled()) {
    const auto push_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start).count();
    std::fprintf(
        stderr,
        "[INPUTSTREAM_PUSH_TIMING] %s holder_push_buffer_ns=%lld bytes=%zu ret=%d record=%d\n",
        where ? where : "InputStream::push_holder_buffer_with_appsrc",
        static_cast<long long>(push_ns),
        static_cast<std::size_t>(buffer ? gst_buffer_get_size(buffer) : 0U), static_cast<int>(ret),
        record_timings ? 1 : 0);
  }
  if (ret != GST_FLOW_OK) {
    release_input_buffer_on_push_fail(buffer, push_fail_unref_tag);
    if (record_timings) {
      st.push_failures.fetch_add(1, std::memory_order_relaxed);
    }
    const char* fail_where = where;
    std::string where_detail;
    if (fail_msg && fail_spec && push_fail_detail_enabled()) {
      where_detail = push_fail_context(where, *fail_msg, *fail_spec, st.src_opt, input_seq_override,
                                       orig_input_seq_override);
      fail_where = where_detail.c_str();
    }
    return handle_appsrc_push_fail(st, fail_where, ret);
  }
  st.inflight.fetch_add(1, std::memory_order_relaxed);
  return true;
}

GstSample* holder_as_gstsample(const std::shared_ptr<void>& holder) {
  if (!holder) {
    return nullptr;
  }
  auto* sample = static_cast<GstSample*>(holder.get());
  return (sample && GST_IS_SAMPLE(sample)) ? sample : nullptr;
}

bool push_holder_sample_with_appsrc(InputStream::State& st, GstSample* sample, GstBuffer* buffer,
                                    const char* where, bool record_timings, const Sample* fail_msg,
                                    const SampleSpec* fail_spec,
                                    const std::optional<int64_t>& input_seq_override,
                                    const std::optional<int64_t>& orig_input_seq_override) {
  std::chrono::steady_clock::time_point t_push_start{};
  if (record_timings || inputstream_push_timing_enabled()) {
    t_push_start = std::chrono::steady_clock::now();
  }
  if (holder_debug_enabled()) {
    const bool has_tensor_set =
        buffer && gst_buffer_get_custom_meta(buffer, SIMA_TENSOR_SET_META_NAME) != nullptr;
    std::fprintf(stderr,
                 "[HOLDER] push_holder_sample_with_appsrc before_push where=%s sample=%p buffer=%p "
                 "bytes=%zu mems=%u has_tensor_set=%d\n",
                 where ? where : "<null>", static_cast<void*>(sample), static_cast<void*>(buffer),
                 static_cast<std::size_t>(buffer ? gst_buffer_get_size(buffer) : 0U),
                 buffer ? gst_buffer_n_memory(buffer) : 0U, has_tensor_set ? 1 : 0);
  }
  if (fail_msg) {
    pipeline_internal::attach_zero_copy_loans_to_gst_buffer(buffer, *fail_msg);
  }
  GstFlowReturn ret = gst_app_src_push_sample(GST_APP_SRC(st.appsrc), sample);
  if (holder_debug_enabled()) {
    std::fprintf(stderr, "[HOLDER] push_holder_sample_with_appsrc after_push where=%s ret=%d\n",
                 where ? where : "<null>", static_cast<int>(ret));
  }
  const auto t_push_end = std::chrono::steady_clock::now();

  if (record_timings) {
    st.push_count.fetch_add(1, std::memory_order_relaxed);
    st.push_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start)
                .count()),
        std::memory_order_relaxed);
  }
  if (inputstream_push_timing_enabled()) {
    const auto push_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start).count();
    std::fprintf(
        stderr,
        "[INPUTSTREAM_PUSH_TIMING] %s holder_push_sample_ns=%lld bytes=%zu ret=%d record=%d\n",
        where ? where : "InputStream::push_holder_sample_with_appsrc",
        static_cast<long long>(push_ns),
        static_cast<std::size_t>(buffer ? gst_buffer_get_size(buffer) : 0U), static_cast<int>(ret),
        record_timings ? 1 : 0);
  }
  if (ret != GST_FLOW_OK) {
    if (record_timings) {
      st.push_failures.fetch_add(1, std::memory_order_relaxed);
    }
    const char* fail_where = where;
    std::string where_detail;
    if (fail_msg && fail_spec && push_fail_detail_enabled()) {
      where_detail = push_fail_context(where, *fail_msg, *fail_spec, st.src_opt, input_seq_override,
                                       orig_input_seq_override);
      fail_where = where_detail.c_str();
    }
    return handle_appsrc_push_fail(st, fail_where, ret);
  }
  st.inflight.fetch_add(1, std::memory_order_relaxed);
  return true;
}

bool sima_meta_fields_need_update(GstBuffer* buffer,
                                  const std::optional<int64_t>& frame_id_override,
                                  const std::optional<int64_t>& input_seq_override,
                                  const std::optional<int64_t>& orig_input_seq_override,
                                  const std::optional<std::string>& stream_id_override,
                                  const std::optional<std::string>& buffer_name_override,
                                  const SampleTimingOverrides& timing_override) {
  GstCustomMeta* meta = buffer ? gst_buffer_get_custom_meta(buffer, "GstSimaMeta") : nullptr;
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s)
    return true;

  auto int64_mismatch = [s](const char* field, int64_t expected) {
    gint64 current = 0;
    return gst_structure_get_int64(s, field, &current) != TRUE ||
           current != static_cast<gint64>(expected);
  };
  auto uint64_mismatch = [s](const char* field, uint64_t expected) {
    guint64 current = 0;
    return gst_structure_get_uint64(s, field, &current) != TRUE ||
           current != static_cast<guint64>(expected);
  };
  auto boolean_mismatch = [s](const char* field, bool expected) {
    gboolean current = FALSE;
    return gst_structure_get_boolean(s, field, &current) != TRUE || (current == TRUE) != expected;
  };
  auto string_mismatch = [s](const char* field, const std::string& expected) {
    const char* current = gst_structure_get_string(s, field);
    return !current || expected != current;
  };

  if (frame_id_override.has_value() && int64_mismatch("frame-id", *frame_id_override))
    return true;
  if (input_seq_override.has_value() && int64_mismatch("input-seq", *input_seq_override))
    return true;
  if (orig_input_seq_override.has_value()) {
    if (int64_mismatch("orig-input-seq", *orig_input_seq_override))
      return true;
  } else if (input_seq_override.has_value() &&
             int64_mismatch("orig-input-seq", *input_seq_override)) {
    return true;
  }
  if (stream_id_override.has_value()) {
    if (string_mismatch("stream-id", *stream_id_override) ||
        string_mismatch("orig-stream-id", *stream_id_override)) {
      return true;
    }
  }
  if (buffer_name_override.has_value() && string_mismatch("buffer-name", *buffer_name_override))
    return true;
  if (timing_override.pts_ns.has_value() && uint64_mismatch("timestamp", *timing_override.pts_ns))
    return true;
  if (boolean_mismatch("sample-frame-id-valid", timing_override.frame_id.has_value()))
    return true;
  if (timing_override.frame_id.has_value() &&
      int64_mismatch("sample-frame-id", *timing_override.frame_id))
    return true;
  if (boolean_mismatch("sample-pts-valid", timing_override.pts_ns.has_value()))
    return true;
  if (timing_override.pts_ns.has_value() &&
      uint64_mismatch("sample-pts-ns", *timing_override.pts_ns))
    return true;
  if (boolean_mismatch("sample-dts-valid", timing_override.dts_ns.has_value()))
    return true;
  if (timing_override.dts_ns.has_value() &&
      uint64_mismatch("sample-dts-ns", *timing_override.dts_ns))
    return true;
  if (boolean_mismatch("sample-duration-valid", timing_override.duration_ns.has_value()))
    return true;
  if (timing_override.duration_ns.has_value() &&
      uint64_mismatch("sample-duration-ns", *timing_override.duration_ns))
    return true;
  return false;
}

bool push_holder_transport(InputStream::State& st, const std::shared_ptr<void>& holder,
                           const char* where, bool record_timings,
                           const std::optional<int64_t>& frame_id_override,
                           const std::optional<int64_t>& input_seq_override,
                           const std::optional<int64_t>& orig_input_seq_override,
                           const std::optional<std::string>& stream_id_override,
                           const std::optional<std::string>& buffer_name_override,
                           const SampleTimingOverrides& timing_override,
                           const Sample* fail_msg = nullptr,
                           const SampleSpec* fail_spec = nullptr) {
  GstBuffer* buf = pipeline_internal::buffer_from_tensor_holder(holder);
  if (!buf) {
    throw std::runtime_error(std::string(where ? where : "InputStream::push_holder_transport") +
                             ": missing GstBuffer");
  }
  BufferUnrefGuard holder_guard(&buf, "InputStream::push_holder_transport:buffer_unref");
  const bool preproc_trace = pipeline_internal::env_bool("SIMA_PREPROC_META_TRACE", false);
  if (preproc_trace) {
    std::fprintf(stderr,
                 "[PREPROC_META_TRACE] holder before meta where=%s has_sima_meta=%d has_preproc=%d "
                 "key_in=%lld key_orig=%lld mems=%u size=%zu\n",
                 where ? where : "<null>", gst_buffer_get_custom_meta(buf, "GstSimaMeta") ? 1 : 0,
                 has_simaai_preprocess_meta(buf) ? 1 : 0,
                 static_cast<long long>(input_seq_override.value_or(-1)),
                 static_cast<long long>(orig_input_seq_override.value_or(-1)),
                 static_cast<unsigned>(gst_buffer_n_memory(buf)),
                 static_cast<std::size_t>(gst_buffer_get_size(buf)));
  }
  if (!gst_buffer_get_custom_meta(buf, "GstSimaMeta")) {
    buf = attach_simaai_meta_inplace(buf, st.src_opt, st.pool_guard, where, frame_id_override,
                                     StreamIdOverride{stream_id_override},
                                     BufferNameOverride{buffer_name_override});
    if (!buf) {
      throw std::runtime_error(std::string(where ? where : "InputStream::push_holder_transport") +
                               ": failed to attach GstSimaMeta");
    }
  }
  const bool need_meta_update = sima_meta_fields_need_update(
      buf, frame_id_override, input_seq_override, orig_input_seq_override, stream_id_override,
      buffer_name_override, timing_override);
  if (need_meta_update) {
    if (!gst_buffer_is_writable(buf)) {
      buf = gst_buffer_make_writable(buf);
      if (!buf) {
        throw std::runtime_error(std::string(where ? where : "InputStream::push_holder_transport") +
                                 ": failed to make holder buffer writable for GstSimaMeta update");
      }
    }
    if (!update_simaai_meta_fields(buf, frame_id_override, input_seq_override,
                                   orig_input_seq_override, stream_id_override,
                                   buffer_name_override, timing_override.pts_ns)) {
      throw std::runtime_error(std::string(where ? where : "InputStream::push_holder_transport") +
                               ": failed to write GstSimaMeta fields");
    }
  }
  if (!write_sample_timing_to_gst_buffer(buf, timing_override)) {
    throw std::runtime_error(std::string(where ? where : "InputStream::push_holder_transport") +
                             ": failed to write sample timing metadata");
  }
  if (preproc_trace) {
    GstCustomMeta* meta = gst_buffer_get_custom_meta(buf, "GstSimaMeta");
    GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
    gint64 input_seq = -1;
    gint64 orig_input_seq = -1;
    const bool has_input_seq = s && gst_structure_get_int64(s, "input-seq", &input_seq);
    const bool has_orig_input_seq =
        s && gst_structure_get_int64(s, "orig-input-seq", &orig_input_seq);
    std::fprintf(stderr,
                 "[PREPROC_META_TRACE] holder after meta where=%s has_sima_meta=%d has_preproc=%d "
                 "has_input_seq=%d input_seq=%lld has_orig=%d orig=%lld\n",
                 where ? where : "<null>", meta ? 1 : 0, has_simaai_preprocess_meta(buf) ? 1 : 0,
                 has_input_seq ? 1 : 0, static_cast<long long>(input_seq),
                 has_orig_input_seq ? 1 : 0, static_cast<long long>(orig_input_seq));
  }
  if (st.current_key.has_value()) {
    const CapKey& key = *st.current_key;
    if (!has_simaai_preprocess_meta(buf) && key.width > 0 && key.height > 0 &&
        st.src_opt.preprocess_meta.has_value()) {
      if (!apply_simaai_preprocess_meta_template(buf, st.src_opt, key.width, key.height)) {
        throw std::runtime_error(
            std::string(where ? where : "InputStream::push_holder_transport") +
            ": Cannot attach preprocessing metadata to this zero-copy frame. Box decoding and "
            "other post-processing stages need the original frame size to produce correct results. "
            "Copy the frame before pushing it, or make sure preprocessing metadata is provided by "
            "the source graph.");
      }
    }
  }
  if (std::getenv("SIMA_JPEG_ZC_TRACE")) {
    GstVideoMeta* vmeta = gst_buffer_get_video_meta(buf);
    GstCustomMeta* smeta = gst_buffer_get_custom_meta(buf, "GstSimaMeta");
    GstStructure* ss = smeta ? gst_custom_meta_get_structure(smeta) : nullptr;
    const char* bname = ss ? gst_structure_get_string(ss, "buffer-name") : nullptr;
    gint64 bid = -1;
    if (ss)
      gst_structure_get_int64(ss, "buffer-id", &bid);
    std::fprintf(stderr,
                 "[JPEG_ZC_TRACE] holder_push where=%s size=%zu mems=%u appsrc_buffer_name=%s "
                 "cur_key=%s spec=%s vmeta=%dx%d fmt=%d n_planes=%u sima_name=%s bid=%lld "
                 "has_preproc=%d\n",
                 where ? where : "<null>", static_cast<std::size_t>(gst_buffer_get_size(buf)),
                 static_cast<unsigned>(gst_buffer_n_memory(buf)), st.src_opt.buffer_name.c_str(),
                 st.current_key.has_value() ? st.current_key->to_string().c_str() : "<none>",
                 fail_spec ? fail_spec->caps_key.to_string().c_str() : "<none>",
                 vmeta ? static_cast<int>(vmeta->width) : -1,
                 vmeta ? static_cast<int>(vmeta->height) : -1,
                 vmeta ? static_cast<int>(vmeta->format) : -1,
                 vmeta ? static_cast<unsigned>(vmeta->n_planes) : 0U, bname ? bname : "<none>",
                 static_cast<long long>(bid), has_simaai_preprocess_meta(buf) ? 1 : 0);
    dump_sima_meta_full(buf, "JPEG_ZC_TRACE");
    dump_buffer_memories(buf, "JPEG_ZC_TRACE");
  }
  dump_buffer_memories(buf, where ? where : "InputStream::push_holder_transport");
  validate_holder_video_meta_or_throw(st, buf);
  pipeline_internal::attach_zero_copy_loans_from_holder_to_gst_buffer(buf, holder);

  if (GstSample* sample = holder_as_gstsample(holder)) {
    const bool has_tensor_set =
        buf && gst_buffer_get_custom_meta(buf, SIMA_TENSOR_SET_META_NAME) != nullptr;
    if (has_tensor_set) {
      if (holder_debug_enabled()) {
        std::fprintf(stderr,
                     "[HOLDER] push_holder_transport tensor-set buffer path where=%s sample=%p "
                     "buffer=%p bytes=%zu mems=%u\n",
                     where ? where : "<null>", static_cast<void*>(sample), static_cast<void*>(buf),
                     static_cast<std::size_t>(gst_buffer_get_size(buf)),
                     static_cast<unsigned>(gst_buffer_n_memory(buf)));
      }
      holder_guard.release();
      return push_holder_buffer_with_appsrc(
          st, buf, where ? where : "InputStream::push_holder_transport",
          "InputStream::push_holder_transport:push_fail", record_timings,
          /*log_refcount_on_push=*/false, fail_msg, fail_spec, input_seq_override,
          orig_input_seq_override);
    }
    GstSample* sample_to_push = sample;
    GstSample* writable_sample = nullptr;
    if (gst_sample_get_buffer(sample) != buf) {
      writable_sample =
          gst_sample_new(buf, gst_sample_get_caps(sample), gst_sample_get_segment(sample), nullptr);
      sample_to_push = writable_sample ? writable_sample : sample;
    }
    const bool pushed =
        push_holder_sample_with_appsrc(st, sample_to_push, buf, where, record_timings, fail_msg,
                                       fail_spec, input_seq_override, orig_input_seq_override);
    if (writable_sample) {
      gst_sample_unref(writable_sample);
    }
    return pushed;
  }

  holder_guard.release();
  return push_holder_buffer_with_appsrc(
      st, buf, where ? where : "InputStream::push_holder_transport",
      "InputStream::push_holder_transport:push_fail", record_timings,
      /*log_refcount_on_push=*/false, fail_msg, fail_spec, input_seq_override,
      orig_input_seq_override);
}

std::optional<bool> admit_copy_payload_nonpush(
    InputStream::State& st, CapsDecision decision, const char* where, const SampleSpec& spec,
    const std::function<size_t(uint8_t*, size_t)>& fill,
    const std::optional<int64_t>& frame_id_override,
    const std::optional<int64_t>& input_seq_override,
    const std::optional<int64_t>& orig_input_seq_override,
    const std::optional<std::string>& stream_id_override,
    const std::optional<std::string>& buffer_name_override,
    const SampleTimingOverrides& timing_override, const std::function<void(GstBuffer**)>& prepare) {
  if (decision != CapsDecision::Queue && decision != CapsDecision::Flush) {
    return std::nullopt;
  }
  ensure_alloc_for_bytes(st, spec.required_bytes_actual, where);
  const bool record_timings = (decision == CapsDecision::Flush) && st.timing_enabled;
  BuiltBuffer pending = build_buffer_with_fill(
      st, where, fill, spec.required_bytes_actual, frame_id_override, input_seq_override,
      orig_input_seq_override, stream_id_override, buffer_name_override, timing_override, prepare,
      record_timings, "build_buffer_with_fill", false, spec.width, spec.height);
  queue_pending_buffer(st, pending, spec, where);
  if (decision == CapsDecision::Queue) {
    st.dropped_frames.fetch_add(1, std::memory_order_relaxed);
    return true;
  }
  return flush_pending_buffer(st, where);
}

bool try_push_message_encoded(InputStream::State& st, const Sample& msg,
                              const simaai::neat::Tensor& input, const SampleSpec& spec,
                              const MessageMetaOverrides& meta,
                              const std::optional<int64_t>& input_seq_override,
                              const std::optional<int64_t>& orig_input_seq_override,
                              const SampleTimingOverrides& timing_override) {
  if (!input.storage) {
    throw std::invalid_argument("InputStream::try_push_message: encoded Sample missing storage");
  }

  if (!st.opt.copy_input && input.storage->kind == simaai::neat::StorageKind::GstSample &&
      input.storage->holder) {
    return push_holder_transport(
        st, input.storage->holder, "InputStream::try_push_message(encoded_holder)",
        /*record_timings=*/st.timing_enabled, meta.frame_id, input_seq_override,
        orig_input_seq_override, meta.stream_id, meta.stream_label, timing_override, &msg, &spec);
  }

  const bool timings = st.timing_enabled;
  size_t nbytes = 0;
  std::string bytes_err;
  if (!pipeline_internal::resolve_encoded_payload_bytes(input, spec, &nbytes, &bytes_err)) {
    throw std::runtime_error(std::string("InputStream::try_push_message: ") +
                             (bytes_err.empty() ? "encoded payload invalid" : bytes_err));
  }

  std::chrono::steady_clock::time_point t_alloc_start{};
  if (timings)
    t_alloc_start = std::chrono::steady_clock::now();
  GstBuffer* buf = gst_buffer_new_allocate(nullptr, nbytes, nullptr);
  if (!buf) {
    throw std::runtime_error("InputStream::try_push_message: encoded buffer allocation failed");
  }
  std::chrono::steady_clock::time_point t_alloc_end{};
  if (timings)
    t_alloc_end = std::chrono::steady_clock::now();

  GstMapInfo map{};
  std::chrono::steady_clock::time_point t_map_start{};
  if (timings)
    t_map_start = std::chrono::steady_clock::now();
  if (!gst_buffer_map(buf, &map, GST_MAP_WRITE)) {
    release_input_buffer(buf, "InputStream::try_push_message:encoded_map_fail");
    throw std::runtime_error("InputStream::try_push_message: encoded buffer map failed");
  }
  std::chrono::steady_clock::time_point t_map_end{};
  if (timings)
    t_map_end = std::chrono::steady_clock::now();
  if (nbytes > 0) {
    std::string copy_err;
    if (!pipeline_internal::copy_tensor_payload_to(input, static_cast<uint8_t*>(map.data), nbytes,
                                                   &copy_err)) {
      gst_buffer_unmap(buf, &map);
      release_input_buffer(buf, "InputStream::try_push_message:encoded_copy_fail");
      throw std::runtime_error(std::string("InputStream::try_push_message: ") +
                               (copy_err.empty() ? "encoded copy failed" : copy_err));
    }
  }
  gst_buffer_unmap(buf, &map);
  std::chrono::steady_clock::time_point t_copy_end{};
  if (timings)
    t_copy_end = std::chrono::steady_clock::now();

  attach_required_meta(buf, st.src_opt, st.pool_guard, "InputStream::try_push_message(encoded)");
  update_simaai_meta_fields(buf, meta.frame_id, input_seq_override, orig_input_seq_override,
                            meta.stream_id, meta.stream_label, timing_override.pts_ns);
  if (!write_sample_timing_to_gst_buffer(buf, timing_override)) {
    release_input_buffer(buf, "InputStream::try_push_message:encoded_timing_fail");
    throw std::runtime_error(
        "InputStream::try_push_message(encoded): failed to write sample timing metadata");
  }

  std::chrono::steady_clock::time_point t_push_start{};
  if (timings)
    t_push_start = std::chrono::steady_clock::now();
  log_push_refcount("InputStream::try_push_message(encoded)", buf);
  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(st.appsrc), buf);
  const auto t_push_end = std::chrono::steady_clock::now();

  if (timings) {
    st.push_count.fetch_add(1, std::memory_order_relaxed);
    st.alloc_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_alloc_end - t_alloc_start)
                .count()),
        std::memory_order_relaxed);
    st.map_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_map_end - t_map_start).count()),
        std::memory_order_relaxed);
    st.copy_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_copy_end - t_map_end).count()),
        std::memory_order_relaxed);
    st.push_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start)
                .count()),
        std::memory_order_relaxed);
  }
  if (ret != GST_FLOW_OK) {
    release_input_buffer_on_push_fail(buf, "InputStream::try_push_message:encoded_push_fail");
    if (timings) {
      st.push_failures.fetch_add(1, std::memory_order_relaxed);
    }
    const char* where = "InputStream::try_push_message(encoded)";
    std::string where_detail;
    if (push_fail_detail_enabled()) {
      where_detail = push_fail_context(where, msg, spec, st.src_opt, input_seq_override,
                                       orig_input_seq_override);
      where = where_detail.c_str();
    }
    handle_appsrc_push_fail(st, where, ret);
    return false;
  }
  st.inflight.fetch_add(1, std::memory_order_relaxed);
  const auto push_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end.time_since_epoch()).count();
  st.last_push_ns.store(static_cast<std::int64_t>(push_ns), std::memory_order_relaxed);
  return true;
}

struct HolderFastPathResult {
  std::optional<bool> handled;
  bool holder_failed = false;
  std::string holder_err;
  std::string holder_fail_reason;
};

struct CpuZeroCopyFastPathResult {
  std::optional<bool> handled;
  bool failed = false;
  std::string reason;
};

HolderFastPathResult try_push_message_holder_fastpath(
    InputStream::State& st, const Sample& msg, const simaai::neat::Tensor& input,
    const SampleSpec& spec, CapsDecision decision, const MessageMetaOverrides& meta,
    const std::optional<int64_t>& input_seq_override,
    const std::optional<int64_t>& orig_input_seq_override,
    const SampleTimingOverrides& timing_override,
    const std::optional<PreprocessRuntimeMeta>& tensor_preprocess_meta) {
  HolderFastPathResult out;

  if (holder_debug_enabled() && input.storage->holder) {
    auto* raw = input.storage->holder.get();
    auto* sample = static_cast<GstSample*>(raw);
    GstBuffer* samp_buf =
        (sample && GST_IS_SAMPLE(sample)) ? gst_sample_get_buffer(sample) : nullptr;
    const int buf_ref = samp_buf ? GST_MINI_OBJECT_REFCOUNT_VALUE(samp_buf) : -1;
    std::fprintf(stderr, "[HOLDER] push_message holder=%p is_sample=%d buffer=%p buf_ref=%d\n", raw,
                 (sample && GST_IS_SAMPLE(sample)) ? 1 : 0, static_cast<void*>(samp_buf), buf_ref);
  }

  const int max_inflight = holder_max_inflight();
  const std::int64_t inflight = st.inflight.load(std::memory_order_relaxed);
  const bool holder_blocked =
      (max_inflight > 0 && inflight >= static_cast<std::int64_t>(max_inflight));
  if (holder_blocked) {
    out.holder_failed = true;
    out.holder_fail_reason = "holder inflight limit";
    out.holder_err = out.holder_fail_reason;
  }

  GstBuffer* holder_buf =
      holder_blocked ? nullptr
                     : pipeline_internal::buffer_from_holder_if_gstsample(input, &out.holder_err);
  if (!holder_buf && holder_debug_enabled()) {
    std::fprintf(stderr, "[HOLDER] push_message holder_buf=null err=%s\n",
                 out.holder_err.empty() ? "<none>" : out.holder_err.c_str());
  }

  const bool nonpush_decision =
      (decision == CapsDecision::Queue || decision == CapsDecision::Flush);
  if (holder_buf && nonpush_decision) {
    BufferUnrefGuard holder_guard(&holder_buf, "InputStream::try_push_message:holder_guard");
    try {
      apply_holder_spec_and_meta_or_throw(&holder_buf, spec, meta, input_seq_override,
                                          orig_input_seq_override, timing_override, st.src_opt,
                                          st.pool_guard, "InputStream::try_push_message(holder)",
                                          tensor_preprocess_meta);
    } catch (const std::exception& e) {
      out.holder_fail_reason = e.what();
      out.holder_failed = true;
      holder_buf = nullptr;
    }
    if (holder_buf) {
      BuiltBuffer pending;
      pending.buffer = gst_buffer_ref(holder_buf);
      pipeline_internal::attach_zero_copy_loans_to_gst_buffer(pending.buffer, msg);
      pipeline_internal::attach_zero_copy_loans_from_holder_to_gst_buffer(
          pending.buffer, input.storage ? input.storage->holder : std::shared_ptr<void>{});
      queue_pending_buffer(st, pending, spec, "InputStream::try_push_message(holder)");
      holder_guard.release();
      release_input_buffer(holder_buf, "InputStream::try_push_message:holder_queue_unref");
      if (decision == CapsDecision::Queue) {
        st.dropped_frames.fetch_add(1, std::memory_order_relaxed);
        out.handled = true;
        return out;
      }
      out.handled = flush_pending_buffer(st, "InputStream::try_push_message(holder)");
      return out;
    }
  }

  if (holder_buf) {
    BufferUnrefGuard holder_guard(&holder_buf, "InputStream::try_push_message:holder_guard");
    try {
      apply_holder_spec_and_meta_or_throw(&holder_buf, spec, meta, input_seq_override,
                                          orig_input_seq_override, timing_override, st.src_opt,
                                          st.pool_guard, "InputStream::try_push_message(holder)",
                                          tensor_preprocess_meta);
    } catch (const std::exception& e) {
      out.holder_fail_reason = e.what();
      out.holder_failed = true;
      holder_buf = nullptr;
    }
    if (holder_buf) {
      if (std::getenv("SIMA_JPEG_ZC_TRACE")) {
        GstVideoMeta* vmeta = gst_buffer_get_video_meta(holder_buf);
        GstCustomMeta* smeta = gst_buffer_get_custom_meta(holder_buf, "GstSimaMeta");
        GstStructure* ss = smeta ? gst_custom_meta_get_structure(smeta) : nullptr;
        const char* bname = ss ? gst_structure_get_string(ss, "buffer-name") : nullptr;
        gint64 bid = -1;
        gint64 input_seq = -1;
        if (ss) {
          gst_structure_get_int64(ss, "buffer-id", &bid);
          gst_structure_get_int64(ss, "input-seq", &input_seq);
        }
        std::fprintf(
            stderr,
            "[JPEG_ZC_TRACE] fastpath holder size=%zu mems=%u appsrc_buffer_name=%s "
            "cur_key=%s spec=%s vmeta=%dx%d fmt=%d n_planes=%u sima_name=%s bid=%lld "
            "input_seq=%lld has_preproc=%d holder_ref=%ld buffer_ref=%d\n",
            static_cast<std::size_t>(gst_buffer_get_size(holder_buf)),
            static_cast<unsigned>(gst_buffer_n_memory(holder_buf)), st.src_opt.buffer_name.c_str(),
            st.current_key.has_value() ? st.current_key->to_string().c_str() : "<none>",
            spec.caps_key.to_string().c_str(), vmeta ? static_cast<int>(vmeta->width) : -1,
            vmeta ? static_cast<int>(vmeta->height) : -1,
            vmeta ? static_cast<int>(vmeta->format) : -1,
            vmeta ? static_cast<unsigned>(vmeta->n_planes) : 0U, bname ? bname : "<none>",
            static_cast<long long>(bid), static_cast<long long>(input_seq),
            has_simaai_preprocess_meta(holder_buf) ? 1 : 0,
            input.storage && input.storage->holder ? input.storage->holder.use_count() : 0L,
            GST_MINI_OBJECT_REFCOUNT_VALUE(holder_buf));
        dump_sima_meta_full(holder_buf, "JPEG_ZC_TRACE_FASTPATH");
        dump_buffer_memories(holder_buf, "JPEG_ZC_TRACE_FASTPATH");
      }
      holder_guard.release();
      out.handled = push_holder_buffer_with_appsrc(
          st, holder_buf, "InputStream::try_push_message(holder)",
          "InputStream::try_push_message:holder_push_fail", /*record_timings=*/false,
          /*log_refcount_on_push=*/true, &msg, &spec, input_seq_override, orig_input_seq_override);
      if (out.handled.value_or(false)) {
        maybe_drop_holder_after_push(input, "InputStream::try_push_message(holder)");
      }
      return out;
    }
  }

  return out;
}

CpuZeroCopyFastPathResult try_push_message_cpu_owned_zero_copy_fastpath(
    InputStream::State& st, const Sample& msg, const simaai::neat::Tensor& input,
    const SampleSpec& spec, CapsDecision decision, const MessageMetaOverrides& meta,
    const std::optional<int64_t>& input_seq_override,
    const std::optional<int64_t>& orig_input_seq_override,
    const SampleTimingOverrides& timing_override, const std::function<void(GstBuffer**)>& prepare) {
  CpuZeroCopyFastPathResult out;

  // Keep v1 surgical: only bypass the memcpy for stable caps/push frames and
  // dense Tensor payloads.  Queue/Flush caps-transition handling stays on the
  // existing copy path, which already preserves the pending-buffer semantics.
  if (decision != CapsDecision::Push || spec.kind != SampleMediaKind::Tensor) {
    return out;
  }

  const bool timing = inputstream_push_timing_enabled();
  const auto t0 =
      timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};

  GstBuffer* buf = nullptr;
  std::string wrap_err;
  if (!pipeline_internal::wrap_cpu_dense_zero_copy(input, &buf, &wrap_err) || !buf) {
    out.failed = true;
    out.reason = wrap_err.empty() ? "cpu zero-copy wrap failed" : wrap_err;
    return out;
  }
  BufferUnrefGuard guard(&buf, "InputStream::try_push_message:cpu_zc_guard");

  const auto t_wrap =
      timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  try {
    if (prepare) {
      prepare(&buf);
    }

    buf = attach_simaai_meta_inplace(
        buf, st.src_opt, st.pool_guard, "InputStream::try_push_message(cpu_zc)", meta.frame_id,
        StreamIdOverride{meta.stream_id}, BufferNameOverride{meta.stream_label});
    if (!buf) {
      throw std::runtime_error(
          "InputStream::try_push_message(cpu_zc): failed to attach GstSimaMeta");
    }
    if (!update_simaai_meta_fields(buf, meta.frame_id, input_seq_override, orig_input_seq_override,
                                   meta.stream_id, meta.stream_label, timing_override.pts_ns)) {
      throw std::runtime_error(
          "InputStream::try_push_message(cpu_zc): failed to write GstSimaMeta fields");
    }
    if (!write_sample_timing_to_gst_buffer(buf, timing_override)) {
      throw std::runtime_error(
          "InputStream::try_push_message(cpu_zc): failed to write sample timing metadata");
    }
    if (!has_simaai_preprocess_meta(buf) && spec.width > 0 && spec.height > 0) {
      (void)apply_simaai_preprocess_meta_template(buf, st.src_opt, spec.width, spec.height);
    }
  } catch (const std::exception& e) {
    out.failed = true;
    out.reason = e.what();
    return out;
  } catch (...) {
    out.failed = true;
    out.reason = "unknown cpu zero-copy metadata exception";
    return out;
  }

  const auto t_meta =
      timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  guard.release();
  const bool pushed = push_holder_buffer_with_appsrc(
      st, buf, "InputStream::try_push_message(cpu_zc)",
      "InputStream::try_push_message:cpu_zc_push_fail",
      /*record_timings=*/st.timing_enabled, /*log_refcount_on_push=*/false, &msg, &spec,
      input_seq_override, orig_input_seq_override);
  const auto t_end =
      timing ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  if (timing) {
    const auto wrap_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_wrap - t0).count();
    const auto meta_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_meta - t_wrap).count();
    const auto push_ns =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_meta).count();
    const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t0).count();
    std::fprintf(stderr,
                 "[INPUTSTREAM_PUSH_TIMING] cpu_zc wrap_ns=%lld meta_ns=%lld push_ns=%lld "
                 "total_ns=%lld bytes=%zu ok=%d\n",
                 static_cast<long long>(wrap_ns), static_cast<long long>(meta_ns),
                 static_cast<long long>(push_ns), static_cast<long long>(total_ns),
                 static_cast<std::size_t>(spec.required_bytes_actual), pushed ? 1 : 0);
  }
  out.handled = pushed;
  return out;
}

} // namespace

void InputStream::push(const cv::Mat& input) {
  if (!try_push(input)) {
    throw_push_failed_with_last_error("InputStream::push", state_);
  }
}

bool InputStream::try_push(const cv::Mat& input) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::try_push: stream is closed");
  }
  if (input.empty()) {
    throw std::invalid_argument("InputStream::try_push: input frame is empty");
  }
  InputOptions relaxed = state_->src_opt;
  relaxed.width = -1;
  relaxed.height = -1;
  relaxed.depth = -1;
  cv::Mat contiguous = input;
  if (!contiguous.isContinuous()) {
    contiguous = input.clone();
  }
  SampleSpec spec = derive_mat_spec_or_throw(contiguous, relaxed);
  validate_spec_with_limits(*state_, spec, "InputStream::try_push(mat)");
  const CapsDecision decision =
      maybe_update_caps_for_spec(*state_, spec, "InputStream::try_push(mat)");

  const size_t input_bytes = spec.required_bytes_actual;
  const std::function<void(GstBuffer**)> prepare =
      make_prepare_for_mat_spec(spec, contiguous, input_bytes);
  const SeqOverrides seq = next_seq_overrides(*state_);
  const auto fill = [&](uint8_t* dst, size_t dst_bytes) -> size_t {
    const size_t copy_bytes = std::min(input_bytes, dst_bytes);
    if (copy_bytes > 0) {
      std::memcpy(dst, contiguous.data, copy_bytes);
    }
    return copy_bytes;
  };
  if (auto admitted = admit_copy_payload_nonpush(
          *state_, decision, "InputStream::try_push(mat)", spec, fill, std::nullopt, seq.input_seq,
          seq.orig_input_seq, std::nullopt, std::nullopt, SampleTimingOverrides{}, prepare);
      admitted.has_value()) {
    return *admitted;
  }

  return push_with_fill("InputStream::try_push(mat)", fill, input_bytes, std::nullopt,
                        seq.input_seq, seq.orig_input_seq, std::nullopt, std::nullopt,
                        SampleTimingOverrides{}, prepare, spec.width, spec.height);
}

void InputStream::push(const simaai::neat::Tensor& input) {
  if (!try_push(input)) {
    throw_push_failed_with_last_error("InputStream::push", state_);
  }
}

bool InputStream::try_push(const simaai::neat::Tensor& input) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::try_push: stream is closed");
  }
  if (input.semantic.encoded.has_value()) {
    throw std::invalid_argument(
        "InputStream::try_push: encoded tensors require Sample (caps/timestamps)");
  }
  return try_push_message(sample_from_tensors(TensorList{input}));
}

void InputStream::push_message(const Sample& msg) {
  if (!try_push_message(msg)) {
    if (state_ && (state_->stop_requested.load() || state_->teardown_started.load())) {
      return;
    }
    throw_push_failed_with_last_error("InputStream::push_message", state_);
  }
}

bool InputStream::try_push_message(const Sample& msg) {
  const bool inputstream_top_timing = inputstream_push_timing_enabled();
  const auto inputstream_top_start = inputstream_top_timing
                                         ? std::chrono::steady_clock::now()
                                         : std::chrono::steady_clock::time_point{};
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::try_push_message: stream is closed");
  }
  if (!state_->appsrc) {
    throw std::runtime_error("InputStream::try_push_message: appsrc not available (no Input)");
  }

  const Sample transport_msg = pipeline_internal::canonicalize_tensor_transport_sample(msg);
  if (pipeline_internal::env_bool("SIMA_SAMPLE_TIMING_DEBUG", false)) {
    std::fprintf(
        stderr,
        "[SAMPLE_TIMING] inputstream_try_push incoming_kind=%d transport_kind=%d "
        "incoming_frame=%lld transport_frame=%lld incoming_pts=%lld transport_pts=%lld "
        "incoming_dts=%lld transport_dts=%lld incoming_dur=%lld transport_dur=%lld "
        "stream_id=%s\n",
        static_cast<int>(msg.kind), static_cast<int>(transport_msg.kind),
        static_cast<long long>(msg.frame_id), static_cast<long long>(transport_msg.frame_id),
        static_cast<long long>(msg.pts_ns), static_cast<long long>(transport_msg.pts_ns),
        static_cast<long long>(msg.dts_ns), static_cast<long long>(transport_msg.dts_ns),
        static_cast<long long>(msg.duration_ns), static_cast<long long>(transport_msg.duration_ns),
        transport_msg.stream_id.c_str());
  }
  const bool tensor_like_message =
      sample_has_tensor_list(transport_msg) || transport_msg.kind == SampleKind::Bundle;
  const bool allow_zero_copy_transport = !state_->opt.copy_input;
  if (!tensor_like_message) {
    throw std::runtime_error("InputStream::try_push_message: missing tensor");
  }

  auto st = state_;
  const SeqOverrides seq = seq_overrides_for_message(*st, transport_msg);
  const MessageMetaOverrides meta = message_meta_overrides(transport_msg);
  verify_buffer_name_override(st->src_opt, meta.stream_label, "InputStream::try_push_message");
  const simaai::neat::Tensor* input_tensor =
      sample_has_tensor_list(transport_msg) && !transport_msg.tensors.empty()
          ? &transport_msg.tensors.front()
          : nullptr;
  const SampleTimingOverrides timing_override = sample_timing_overrides_from_sample(transport_msg);
  cache_preprocess_meta(*st, transport_msg, seq.input_seq, seq.orig_input_seq);
  SampleSpec spec = derive_sample_spec_or_throw(transport_msg);
  const bool use_tensor_envelope_transport = spec.tensor_envelope_transport;
  // Fail-fast (set-complete) guard: covers every transport kind below,
  // including the tensor-envelope branch which previously had no guard.
  enforce_device_visible_push_or_throw(st->opt.require_device_visible_input, transport_msg,
                                       "InputStream::try_push_message");
  enforce_live_gstsample_producer_or_throw(transport_msg, "InputStream::try_push_message",
                                           st->opt.allow_graph_internal_zero_copy_input);
  if (spec.kind == SampleMediaKind::RawVideo) {
    spec.fps_n = st->src_opt.fps_n;
    spec.fps_d = st->src_opt.fps_d;
    spec.caps_key = capkey_from_spec(spec);
    spec.caps_string = caps_string_from_spec(spec);
  }
  validate_spec_with_limits(*st, spec, "InputStream::try_push_message");
  const CapsDecision decision =
      maybe_update_caps_for_spec(*state_, spec, "InputStream::try_push_message");
  if (spec.kind == SampleMediaKind::Encoded) {
    if (decision != CapsDecision::Push) {
      throw std::runtime_error("InputStream::try_push_message: encoded caps change not supported");
    }
    if (!input_tensor) {
      throw std::runtime_error("InputStream::try_push_message: encoded sample missing tensor");
    }
    return try_push_message_encoded(*st, msg, *input_tensor, spec, meta, seq.input_seq,
                                    seq.orig_input_seq, timing_override);
  }

  if (use_tensor_envelope_transport) {
    if (decision != CapsDecision::Push) {
      throw std::runtime_error(
          "InputStream::try_push_message: tensor envelope caps change not supported");
    }
    Sample envelope = msg;
    if (meta.frame_id.has_value()) {
      envelope.frame_id = *meta.frame_id;
    }
    if (seq.input_seq.has_value()) {
      envelope.input_seq = *seq.input_seq;
    }
    if (seq.orig_input_seq.has_value()) {
      envelope.orig_input_seq = *seq.orig_input_seq;
    }
    if (meta.stream_id.has_value()) {
      envelope.stream_id = *meta.stream_id;
    }
    if (meta.stream_label.has_value() && envelope.stream_label.empty()) {
      envelope.stream_label = *meta.stream_label;
    }
    const auto inputstream_before_envelope = inputstream_top_timing
                                                 ? std::chrono::steady_clock::now()
                                                 : std::chrono::steady_clock::time_point{};
    std::string err;
    auto holder =
        pipeline_internal::sample_to_gst_envelope_holder(envelope, &err, allow_zero_copy_transport);
    const auto inputstream_after_envelope = inputstream_top_timing
                                                ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
    if (!holder) {
      std::string detail =
          err.empty() ? "InputStream::try_push_message: tensor envelope conversion failed" : err;
      // The low-level envelope builder reports holder/GstSample-shaped errors
      // (e.g. "holder: storage is not GstSample") when a CPU-backed tensor can't
      // back a device envelope. Append actionable guidance so the failure is
      // self-explanatory regardless of whether require_device_visible_input was
      // armed for this route.
      if (sample_has_cpu_backed_tensor_for_device_push(transport_msg)) {
        detail += " [" + std::string(device_visible_push_guard_message()) + "]";
      }
      throw std::runtime_error(detail);
    }
    const bool ok = push_holder_transport(
        *st, holder, "InputStream::try_push_message(holder_envelope)",
        /*record_timings=*/st->timing_enabled, meta.frame_id, seq.input_seq, seq.orig_input_seq,
        meta.stream_id, meta.stream_label, timing_override, &transport_msg, &spec);
    if (inputstream_top_timing) {
      const auto inputstream_end = std::chrono::steady_clock::now();
      const auto pre_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                              inputstream_before_envelope - inputstream_top_start)
                              .count();
      const auto envelope_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   inputstream_after_envelope - inputstream_before_envelope)
                                   .count();
      const auto push_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                               inputstream_end - inputstream_after_envelope)
                               .count();
      const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                inputstream_end - inputstream_top_start)
                                .count();
      std::fprintf(stderr,
                   "[INPUTSTREAM_PUSH_TIMING] try_push_message_tensor_envelope pre_ns=%lld "
                   "envelope_ns=%lld push_transport_ns=%lld total_ns=%lld ok=%d\n",
                   static_cast<long long>(pre_ns), static_cast<long long>(envelope_ns),
                   static_cast<long long>(push_ns), static_cast<long long>(total_ns), ok ? 1 : 0);
    }
    return ok;
  }

  if (!input_tensor) {
    throw std::runtime_error("InputStream::try_push_message: missing tensor payload");
  }
  const simaai::neat::Tensor& input = *input_tensor;
  if (!input.storage) {
    throw std::invalid_argument(
        "InputStream::try_push_message: simaai::neat::Tensor missing storage");
  }
  const std::optional<PreprocessRuntimeMeta> tensor_preprocess_meta = input.semantic.preprocess;

  HolderFastPathResult holder_result;
  if (allow_zero_copy_transport) {
    holder_result = try_push_message_holder_fastpath(*st, msg, input, spec, decision, meta,
                                                     seq.input_seq, seq.orig_input_seq,
                                                     timing_override, tensor_preprocess_meta);
  }
  if (holder_result.handled.has_value()) {
    return *holder_result.handled;
  }

  if (holder_result.holder_failed && holder_debug_enabled()) {
    std::fprintf(stderr, "[HOLDER] push_message holder path failed err=%s\n",
                 holder_result.holder_fail_reason.empty()
                     ? "<unknown>"
                     : holder_result.holder_fail_reason.c_str());
  }
  if (allow_zero_copy_transport && input.storage->kind == simaai::neat::StorageKind::GstSample) {
    if (holder_result.holder_failed) {
      throw std::runtime_error(
          "InputStream::try_push_message: Cannot use this zero-copy frame as input because its "
          "frame metadata could not be prepared safely. Copy the frame before pushing it, or keep "
          "the producer and consumer in one graph. Details: " +
          (holder_result.holder_fail_reason.empty() ? std::string("holder metadata failed")
                                                    : holder_result.holder_fail_reason));
    }
    throw std::runtime_error("InputStream::try_push_message: " +
                             (holder_result.holder_err.empty() ? std::string("missing GstBuffer")
                                                               : holder_result.holder_err));
  }

  const size_t input_bytes = spec.required_bytes_actual;
  GstBuffer* source_preproc_meta_buffer = nullptr;
  if (input.storage && input.storage->holder) {
    source_preproc_meta_buffer =
        pipeline_internal::buffer_from_tensor_holder(input.storage->holder);
  }
  const TensorList* tensor_set_meta_tensors =
      sample_has_tensor_list(transport_msg) ? &transport_msg.tensors : nullptr;
  const std::function<void(GstBuffer**)> prepare =
      make_prepare_for_spec(spec, "InputStream::try_push_message", source_preproc_meta_buffer,
                            tensor_set_meta_tensors, tensor_preprocess_meta);

  if (allow_zero_copy_transport && cpu_owned_zero_copy_input_enabled()) {
    CpuZeroCopyFastPathResult cpu_zc_result = try_push_message_cpu_owned_zero_copy_fastpath(
        *st, msg, input, spec, decision, meta, seq.input_seq, seq.orig_input_seq, timing_override,
        prepare);
    if (cpu_zc_result.handled.has_value()) {
      maybe_drop_holder_after_push(input, "InputStream::try_push_message(cpu_zc)");
      return *cpu_zc_result.handled;
    }
    if (cpu_zc_result.failed && holder_debug_enabled()) {
      std::fprintf(stderr, "[HOLDER] push_message cpu_zc fallback to copy err=%s\n",
                   cpu_zc_result.reason.empty() ? "<unknown>" : cpu_zc_result.reason.c_str());
    }
  }

  // Device-visibility guard already enforced set-completely at the top of
  // try_push_message (covers this copy path too).

  const auto fill = make_tensor_copy_fill(input, input_bytes, "InputStream::try_push_message");

  if (auto admitted = admit_copy_payload_nonpush(
          *st, decision, "InputStream::try_push_message", spec, fill, meta.frame_id, seq.input_seq,
          seq.orig_input_seq, meta.stream_id, meta.stream_label, timing_override, prepare);
      admitted.has_value()) {
    return *admitted;
  }

  ensure_alloc_for_bytes(*state_, spec.required_bytes_actual, "InputStream::try_push_message");

  const char* where = "InputStream::try_push_message";
  std::string where_detail;
  if (push_fail_detail_enabled()) {
    where_detail =
        push_fail_context(where, msg, spec, st->src_opt, seq.input_seq, seq.orig_input_seq);
    where = where_detail.c_str();
  }
  const bool pushed = push_with_fill(where, fill, input_bytes, meta.frame_id, seq.input_seq,
                                     seq.orig_input_seq, meta.stream_id, meta.stream_label,
                                     timing_override, prepare, spec.width, spec.height);
  if (pushed) {
    maybe_drop_holder_after_push(input, "InputStream::try_push_message(copy)");
  }
  return pushed;
}

void InputStream::push_holder(const std::shared_ptr<void>& holder) {
  if (!try_push_holder(holder)) {
    throw_push_failed_with_last_error("InputStream::push_holder", state_);
  }
}

bool InputStream::try_push_holder(const std::shared_ptr<void>& holder) {
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::try_push_holder: stream is closed");
  }
  if (!state_->appsrc) {
    throw std::runtime_error("InputStream::try_push_holder: appsrc not available (no Input)");
  }
  if (!holder) {
    throw std::invalid_argument("InputStream::try_push_holder: missing holder");
  }

  auto st = state_;
  const SeqOverrides seq = next_seq_overrides(*st);
  return push_holder_transport(*st, holder, "InputStream::try_push_holder", st->timing_enabled,
                               std::nullopt, seq.input_seq, seq.orig_input_seq, std::nullopt,
                               std::nullopt, SampleTimingOverrides{});
}
} // namespace simaai::neat
