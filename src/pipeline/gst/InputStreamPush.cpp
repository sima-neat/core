#include "InputStreamInternal.h"

namespace simaai::neat {

namespace {

struct SeqOverrides {
  std::optional<int64_t> input_seq;
  std::optional<int64_t> orig_input_seq;
};

struct MessageMetaOverrides {
  std::optional<int64_t> frame_id;
  std::optional<std::string> stream_id;
  std::optional<std::string> port_name;
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
  out.port_name = msg.port_name.empty() ? std::nullopt : std::optional<std::string>(msg.port_name);
  return out;
}

void validate_spec_with_limits(const InputStream::State& st, const SampleSpec& spec,
                               const char* where) {
  if (spec.kind != SampleMediaKind::RawVideo && spec.kind != SampleMediaKind::Tensor)
    return;

  const char* tag = where ? where : "InputStream::try_push";
  const int max_w = st.shape_limits.max_width;
  const int max_h = st.shape_limits.max_height;
  const int max_d = st.shape_limits.max_depth;

  auto fail_if_over = [&](const char* field, int value, int max_val) {
    if (value <= 0 || max_val <= 0 || value <= max_val)
      return;
    std::ostringstream oss;
    oss << tag << ": " << field << " exceeds effective max (" << value << " > " << max_val
        << ")";
    throw std::invalid_argument(oss.str());
  };

  fail_if_over("width", spec.width, max_w);
  fail_if_over("height", spec.height, max_h);
  if (spec.depth > 0) {
    fail_if_over("depth", spec.depth, max_d);
  }
}

std::function<void(GstBuffer*)> make_prepare_for_spec(const SampleSpec& spec, const char* where) {
  if (spec.kind == SampleMediaKind::RawVideo) {
    return [spec, where](GstBuffer* buf) { apply_video_meta_or_throw(&buf, spec, where); };
  }
  if (spec.kind == SampleMediaKind::Tensor) {
    return [spec, where](GstBuffer* buf) { apply_tensor_size_or_throw(&buf, spec, where); };
  }
  return {};
}

SampleSpec derive_mat_spec_or_throw(const cv::Mat& contiguous, const InputOptions& relaxed) {
  SampleSpec spec;
  const bool float_input = (contiguous.type() == CV_32FC1 || contiguous.type() == CV_32FC3);
  spec.media_type = relaxed.media_type.empty()
                        ? (float_input ? "application/vnd.simaai.tensor" : "video/x-raw")
                        : relaxed.media_type;
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
    if (contiguous.type() != CV_32FC1 && contiguous.type() != CV_32FC3) {
      throw std::invalid_argument("InputStream::try_push: tensor input must be CV_32FC1/CV_32FC3");
    }
    spec.required_bytes_actual = contiguous.total() * contiguous.elemSize();
    spec.dtype = TensorDType::Float32;
    spec.format = "FP32";
    spec.width = contiguous.cols;
    spec.height = contiguous.rows;
    spec.depth = contiguous.channels();
    spec.layout = (spec.depth > 1) ? TensorLayout::HWC : TensorLayout::HW;
    if (spec.layout == TensorLayout::HWC) {
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

std::function<void(GstBuffer*)>
make_prepare_for_mat_spec(const SampleSpec& spec, const cv::Mat& contiguous, size_t input_bytes) {
  if (spec.kind == SampleMediaKind::RawVideo) {
    SampleSpec meta_spec = spec;
    meta_spec.width = contiguous.cols;
    meta_spec.height = contiguous.rows;
    meta_spec.depth = contiguous.channels();
    meta_spec.planes[0].stride_bytes = static_cast<int64_t>(contiguous.step[0]);
    meta_spec.planes[0].size_bytes = input_bytes;
    meta_spec.required_bytes_actual = input_bytes;
    return [meta_spec](GstBuffer* buf) {
      apply_video_meta_or_throw(&buf, meta_spec, "InputStream::try_push(mat)");
    };
  }
  if (spec.kind == SampleMediaKind::Tensor) {
    SampleSpec size_spec = spec;
    size_spec.required_bytes_actual = input_bytes;
    return [size_spec](GstBuffer* buf) {
      apply_tensor_size_or_throw(&buf, size_spec, "InputStream::try_push(mat)");
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

void apply_holder_spec_and_meta_or_throw(GstBuffer** buffer, const SampleSpec& spec,
                                         const MessageMetaOverrides& meta,
                                         const std::optional<int64_t>& input_seq_override,
                                         const std::optional<int64_t>& orig_input_seq_override,
                                         const std::optional<uint64_t>& ts_override,
                                         const char* where) {
  if (spec.kind == SampleMediaKind::RawVideo) {
    apply_video_meta_or_throw(buffer, spec, where);
  } else if (spec.kind == SampleMediaKind::Tensor) {
    apply_tensor_size_or_throw(buffer, spec, where);
  }
  update_simaai_meta_fields(*buffer, meta.frame_id, input_seq_override, orig_input_seq_override,
                            meta.stream_id, meta.port_name, ts_override);
}

bool push_holder_buffer_with_appsrc(InputStream::State& st, GstBuffer* buffer, const char* where,
                                    const char* push_fail_unref_tag, bool record_timings,
                                    bool log_refcount_on_push, const Sample* fail_msg,
                                    const SampleSpec* fail_spec,
                                    const std::optional<int64_t>& input_seq_override,
                                    const std::optional<int64_t>& orig_input_seq_override) {
  std::chrono::steady_clock::time_point t_push_start{};
  if (record_timings) {
    t_push_start = std::chrono::steady_clock::now();
  }
  if (log_refcount_on_push) {
    log_push_refcount(where, buffer);
  }
  GstFlowReturn ret = gst_app_src_push_buffer(GST_APP_SRC(st.appsrc), buffer);
  const auto t_push_end = std::chrono::steady_clock::now();

  if (record_timings) {
    st.push_count.fetch_add(1, std::memory_order_relaxed);
    st.push_ns.fetch_add(
        static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(t_push_end - t_push_start)
                .count()),
        std::memory_order_relaxed);
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

std::optional<bool>
admit_copy_payload_nonpush(InputStream::State& st, CapsDecision decision, const char* where,
                           const SampleSpec& spec,
                           const std::function<size_t(uint8_t*, size_t)>& fill,
                           const std::optional<int64_t>& frame_id_override,
                           const std::optional<int64_t>& input_seq_override,
                           const std::optional<int64_t>& orig_input_seq_override,
                           const std::optional<std::string>& stream_id_override,
                           const std::optional<std::string>& buffer_name_override,
                           const std::optional<uint64_t>& timestamp_override,
                           const std::function<void(GstBuffer*)>& prepare) {
  if (decision != CapsDecision::Queue && decision != CapsDecision::Flush) {
    return std::nullopt;
  }
  ensure_alloc_for_bytes(st, spec.required_bytes_actual, where);
  const bool record_timings = (decision == CapsDecision::Flush) && st.timing_enabled;
  BuiltBuffer pending = build_buffer_with_fill(
      st, where, fill, spec.required_bytes_actual, frame_id_override, input_seq_override,
      orig_input_seq_override, stream_id_override, buffer_name_override, timestamp_override,
      prepare, record_timings, "build_buffer_with_fill", false);
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
                              const std::optional<uint64_t>& ts_override) {
  if (!input.storage) {
    throw std::invalid_argument("InputStream::try_push_message: encoded Sample missing storage");
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

  if (msg.pts_ns >= 0) {
    GST_BUFFER_PTS(buf) = static_cast<GstClockTime>(msg.pts_ns);
  }
  if (msg.dts_ns >= 0) {
    GST_BUFFER_DTS(buf) = static_cast<GstClockTime>(msg.dts_ns);
  }
  if (msg.duration_ns >= 0) {
    GST_BUFFER_DURATION(buf) = static_cast<GstClockTime>(msg.duration_ns);
  }

  attach_required_meta(buf, st.src_opt, st.pool_guard, "InputStream::try_push_message(encoded)");
  update_simaai_meta_fields(buf, meta.frame_id, input_seq_override, orig_input_seq_override,
                            meta.stream_id, meta.port_name, ts_override);

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

HolderFastPathResult
try_push_message_holder_fastpath(InputStream::State& st, const Sample& msg,
                                 const simaai::neat::Tensor& input, const SampleSpec& spec,
                                 CapsDecision decision, const MessageMetaOverrides& meta,
                                 const std::optional<int64_t>& input_seq_override,
                                 const std::optional<int64_t>& orig_input_seq_override,
                                 const std::optional<uint64_t>& ts_override) {
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
                                          orig_input_seq_override, ts_override,
                                          "InputStream::try_push_message(holder)");
    } catch (const std::exception& e) {
      out.holder_fail_reason = e.what();
      out.holder_failed = true;
      holder_buf = nullptr;
    }
    if (holder_buf) {
      BuiltBuffer pending;
      pending.buffer = gst_buffer_ref(holder_buf);
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
                                          orig_input_seq_override, ts_override,
                                          "InputStream::try_push_message(holder)");
    } catch (const std::exception& e) {
      out.holder_fail_reason = e.what();
      out.holder_failed = true;
      holder_buf = nullptr;
    }
    if (holder_buf) {
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
  const std::function<void(GstBuffer*)> prepare =
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
          seq.orig_input_seq, std::nullopt, std::nullopt, std::nullopt, prepare);
      admitted.has_value()) {
    return *admitted;
  }

  return push_with_fill("InputStream::try_push(mat)", fill, input_bytes, std::nullopt,
                        seq.input_seq, seq.orig_input_seq, std::nullopt, std::nullopt, std::nullopt,
                        prepare);
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
  InputOptions relaxed = state_->src_opt;
  relaxed.width = -1;
  relaxed.height = -1;
  relaxed.depth = -1;
  SampleSpec spec = derive_tensor_spec_or_throw(input, relaxed, "InputStream::try_push");
  validate_spec_with_limits(*state_, spec, "InputStream::try_push(neat)");
  if (!input.storage) {
    throw std::invalid_argument("InputStream::try_push: simaai::neat::Tensor missing storage");
  }
  const CapsDecision decision =
      maybe_update_caps_for_spec(*state_, spec, "InputStream::try_push(neat)");
  if (decision == CapsDecision::Push) {
    ensure_alloc_for_bytes(*state_, spec.required_bytes_actual, "InputStream::try_push(neat)");
  }

  const size_t input_bytes = spec.required_bytes_actual;
  const std::function<void(GstBuffer*)> prepare =
      make_prepare_for_spec(spec, "InputStream::try_push(neat)");

  const SeqOverrides seq = next_seq_overrides(*state_);
  const auto fill = make_tensor_copy_fill(input, input_bytes, "InputStream::try_push");
  if (auto admitted = admit_copy_payload_nonpush(
          *state_, decision, "InputStream::try_push(neat)", spec, fill, std::nullopt, seq.input_seq,
          seq.orig_input_seq, std::nullopt, std::nullopt, std::nullopt, prepare);
      admitted.has_value()) {
    return *admitted;
  }

  return push_with_fill("InputStream::try_push(neat)", fill, input_bytes, std::nullopt,
                        seq.input_seq, seq.orig_input_seq, std::nullopt, std::nullopt, std::nullopt,
                        prepare);
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
  if (!state_ || !state_->pipeline) {
    throw std::runtime_error("InputStream::try_push_message: stream is closed");
  }
  if (!state_->appsrc) {
    throw std::runtime_error("InputStream::try_push_message: appsrc not available (no Input)");
  }

  if (msg.kind == SampleKind::Bundle) {
    std::string err;
    auto holder = pipeline_internal::make_sample_holder_from_bundle(msg, &err);
    if (!holder) {
      throw std::runtime_error(
          err.empty() ? "InputStream::try_push_message: bundle to sample failed" : err);
    }
    return try_push_holder(holder);
  }

  if (msg.kind != SampleKind::Tensor || !msg.tensor.has_value()) {
    throw std::runtime_error("InputStream::try_push_message: missing tensor");
  }

  auto st = state_;
  const SeqOverrides seq = seq_overrides_for_message(*st, msg);
  const MessageMetaOverrides meta = message_meta_overrides(msg);
  verify_buffer_name_override(st->src_opt, meta.port_name, "InputStream::try_push_message");
  const simaai::neat::Tensor& input = *msg.tensor;
  const std::optional<uint64_t> ts_override =
      msg.pts_ns >= 0 ? std::optional<uint64_t>(static_cast<uint64_t>(msg.pts_ns)) : std::nullopt;
  SampleSpec spec = derive_sample_spec_or_throw(msg);
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
    return try_push_message_encoded(*st, msg, input, spec, meta, seq.input_seq, seq.orig_input_seq,
                                    ts_override);
  }

  if (!input.storage) {
    throw std::invalid_argument(
        "InputStream::try_push_message: simaai::neat::Tensor missing storage");
  }

  const HolderFastPathResult holder_result = try_push_message_holder_fastpath(
      *st, msg, input, spec, decision, meta, seq.input_seq, seq.orig_input_seq, ts_override);
  if (holder_result.handled.has_value()) {
    return *holder_result.handled;
  }

  if (holder_result.holder_failed && holder_debug_enabled()) {
    std::fprintf(stderr, "[HOLDER] push_message fallback to copy err=%s\n",
                 holder_result.holder_fail_reason.empty()
                     ? "<unknown>"
                     : holder_result.holder_fail_reason.c_str());
  }
  if (input.storage->kind == simaai::neat::StorageKind::GstSample && !holder_result.holder_failed) {
    throw std::runtime_error("InputStream::try_push_message: " +
                             (holder_result.holder_err.empty() ? std::string("missing GstBuffer")
                                                               : holder_result.holder_err));
  }

  const size_t input_bytes = spec.required_bytes_actual;
  const std::function<void(GstBuffer*)> prepare =
      make_prepare_for_spec(spec, "InputStream::try_push_message");
  const auto fill = make_tensor_copy_fill(input, input_bytes, "InputStream::try_push_message");

  if (auto admitted = admit_copy_payload_nonpush(
          *st, decision, "InputStream::try_push_message", spec, fill, meta.frame_id, seq.input_seq,
          seq.orig_input_seq, meta.stream_id, meta.port_name, ts_override, prepare);
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
  const bool pushed =
      push_with_fill(where, fill, input_bytes, meta.frame_id, seq.input_seq, seq.orig_input_seq,
                     meta.stream_id, meta.port_name, ts_override, prepare);
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
  const bool timings = st->timing_enabled;

  // This path reuses the original GstBuffer so plugin metadata (GstSimaMeta)
  // and layout assumptions survive standalone stage boundaries.
  GstBuffer* buf = pipeline_internal::buffer_from_tensor_holder(holder);
  if (!buf) {
    throw std::runtime_error("InputStream::try_push_holder: missing GstBuffer");
  }
  const SeqOverrides seq = next_seq_overrides(*st);
  update_simaai_meta_fields(buf, std::nullopt, seq.input_seq, seq.orig_input_seq, std::nullopt,
                            std::nullopt, std::nullopt);
  dump_buffer_memories(buf, "InputStream::try_push_holder");
  validate_holder_video_meta_or_throw(*st, buf);
  return push_holder_buffer_with_appsrc(
      *st, buf, "InputStream::try_push_holder", "InputStream::try_push_holder:push_fail", timings,
      /*log_refcount_on_push=*/false, nullptr, nullptr, std::nullopt, std::nullopt);
}
} // namespace simaai::neat
