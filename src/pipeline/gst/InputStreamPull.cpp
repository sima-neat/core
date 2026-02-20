#include "InputStreamInternal.h"

#include "pipeline/EncodedSampleUtil.h"
#include "pipeline/TensorAdapters.h"
#include "pipeline/internal/CapsBridge.h"

namespace simaai::neat {
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

constexpr const char* kSampleMetaName = "GstSimaSampleMeta";

bool sample_debug_enabled() {
  return pipeline_internal::env_bool("SIMA_SAMPLE_DEBUG", false);
}

bool sample_bytes_enabled() {
  return pipeline_internal::env_bool("SIMA_SAMPLE_BYTES", false);
}

static std::optional<Sample> bundle_from_sample_meta(GstSample* sample, const char* where,
                                                     bool copy_output);

static void fill_output_meta_from_sample(GstSample* sample, Sample* out) {
  if (!sample || !out)
    return;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer)
    return;

  const GstClockTime pts = GST_BUFFER_PTS(buffer);
  const GstClockTime dts = GST_BUFFER_DTS(buffer);
  const GstClockTime dur = GST_BUFFER_DURATION(buffer);
  if (pts != GST_CLOCK_TIME_NONE)
    out->pts_ns = static_cast<int64_t>(pts);
  if (dts != GST_CLOCK_TIME_NONE)
    out->dts_ns = static_cast<int64_t>(dts);
  if (dur != GST_CLOCK_TIME_NONE)
    out->duration_ns = static_cast<int64_t>(dur);

  GstCustomMeta* meta = gst_buffer_get_custom_meta(buffer, "GstSimaMeta");
  GstStructure* s = meta ? gst_custom_meta_get_structure(meta) : nullptr;
  if (!s)
    return;

  gint64 frame_id = -1;
  gint64 buffer_offset = -1;
  gst_structure_get_int64(s, "frame-id", &frame_id);
  gst_structure_get_int64(s, "buffer-offset", &buffer_offset);
  gint64 input_seq = -1;
  gint64 orig_input_seq = -1;
  const bool has_input_seq = (gst_structure_get_int64(s, "input-seq", &input_seq) == TRUE);
  const bool has_orig_input_seq =
      (gst_structure_get_int64(s, "orig-input-seq", &orig_input_seq) == TRUE);
  const char* stream_id = gst_structure_get_string(s, "stream-id");
  const char* buffer_name = gst_structure_get_string(s, "buffer-name");
  const char* orig_stream_id = gst_structure_get_string(s, "orig-stream-id");
  if (frame_id >= 0)
    out->frame_id = frame_id;
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
    out->port_name = buffer_name;
  if (buffer_offset >= 0 && buffer_offset <= std::numeric_limits<int>::max()) {
    out->output_index = static_cast<int>(buffer_offset);
  }
}

static Sample output_from_sample_stream_inner(GstSample* sample, const char* where,
                                              bool copy_output) {
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
      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * static_cast<size_t>(w),
                    y + static_cast<size_t>(r) * static_cast<size_t>(y_stride),
                    static_cast<size_t>(w));
      }
      uint8_t* dst_uv = dst + y_sz;
      const int uv_h = h / 2;
      for (int r = 0; r < uv_h; ++r) {
        std::memcpy(dst_uv + static_cast<size_t>(r) * static_cast<size_t>(w),
                    uv + static_cast<size_t>(r) * static_cast<size_t>(uv_stride),
                    static_cast<size_t>(w));
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
      for (int r = 0; r < h; ++r) {
        std::memcpy(dst + static_cast<size_t>(r) * row_bytes,
                    src + static_cast<size_t>(r) * static_cast<size_t>(src_stride), row_bytes);
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

  fill_output_meta_from_sample(sample, &out);

  const bool is_raw = media && (std::string(media).rfind("video/x-raw", 0) == 0 ||
                                std::string(media) == "application/vnd.simaai.tensor");

  if (is_raw) {
    simaai::neat::Tensor neat = simaai::neat::from_gst_sample(sample);
    if (pipeline_internal::env_bool("SIMA_NEAT_CAPS_TRACE", false)) {
      const auto constraint = pipeline_internal::tensor_constraint_from_caps(out_caps);
      std::fprintf(stderr, "[NEAT_CAPS] caps=%s constraint=%s tensor=%s\n", out.caps_string.c_str(),
                   pipeline_internal::tensor_constraint_debug_string(constraint).c_str(),
                   neat.debug_string().c_str());
    }
    out.kind = SampleKind::Tensor;
    out.payload_tag = normalize_format(tensor_format_name(neat));
    out.format = out.payload_tag;
    if (copy_output) {
      bool copied = false;
      const bool is_tensor_media = media && std::string(media) == "application/vnd.simaai.tensor";
      if (is_tensor_media && neat.storage &&
          neat.storage->kind == simaai::neat::StorageKind::GstSample && neat.planes.empty()) {
        try {
          out.tensor = pipeline_internal::copy_tensor_from_sample_memory(neat, 0, false);
          out.owned = true;
          copied = true;
        } catch (const std::exception& e) {
          if (pipeline_internal::env_bool("SIMA_GRAPH_OUTPUT_COPY_DEBUG", false)) {
            std::fprintf(stderr, "[GRAPH_OUTPUT] copy_tensor_from_sample_memory failed: %s\n",
                         e.what());
          }
        }
      }
      if (!copied) {
        out.tensor = neat.clone();
        out.owned = true;
      }
    } else {
      out.tensor = neat;
      out.owned = false;
    }
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

  Sample enc = make_encoded_sample(std::move(bytes), out.caps_string, out.pts_ns, out.dts_ns,
                                   out.duration_ns);
  enc.media_type = out.media_type;
  enc.frame_id = out.frame_id;
  enc.stream_id = out.stream_id;
  enc.port_name = out.port_name;
  enc.output_index = out.output_index;
  enc.input_seq = out.input_seq;
  enc.orig_input_seq = out.orig_input_seq;
  enc.owned = out.owned;
  return enc;
}

Sample output_from_sample_stream(GstSample* sample, const char* where, bool copy_output,
                                 const std::optional<OutputTensorOverride>* override_opt) {
  if (auto bundle = bundle_from_sample_meta(sample, where, copy_output)) {
    Sample out = *bundle;
    if (override_opt && override_opt->has_value()) {
      out = apply_output_tensor_override(out, **override_opt);
    }
    return out;
  }
  Sample out = output_from_sample_stream_inner(sample, where, copy_output);
  if (override_opt && override_opt->has_value()) {
    out = apply_output_tensor_override(out, **override_opt);
  }
  if (pipeline_internal::env_bool("SIMA_SAMPLE_FORCE_BUNDLE", false)) {
    if (out.kind != SampleKind::Bundle) {
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
  fill_output_meta_from_sample(sample, &out);

  if (sample_debug_enabled()) {
    std::fprintf(stderr, "[SAMPLE] %s: bundle meta fields=%u\n", where, field_count);
  }

  out.fields.reserve(field_count);
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

    Sample field_out = output_from_sample_stream_inner(field_sample, where, copy_output);
    gst_sample_unref(field_sample);

    if (field_name && *field_name) {
      field_out.port_name = field_name;
    } else if (buffer_name && *buffer_name) {
      field_out.port_name = buffer_name;
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
      const char* name = field_out.port_name.empty() ? "field" : field_out.port_name.c_str();
      if (sample_debug_enabled()) {
        std::fprintf(stderr, "[SAMPLE] %s: field[%u] name=%s caps=%s\n", where, i, name,
                     field_out.caps_string.empty() ? "<none>" : field_out.caps_string.c_str());
      }
      if (sample_bytes_enabled()) {
        const size_t buf_bytes = static_cast<size_t>(gst_buffer_get_size(field_buf));
        size_t tensor_bytes = 0;
        if (field_out.tensor.has_value()) {
          tensor_bytes = pipeline_internal::tensor_bytes_tight(*field_out.tensor);
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
  std::chrono::steady_clock::time_point t_decode_start{};
  if (timings)
    t_decode_start = std::chrono::steady_clock::now();
  Sample out = output_from_sample_stream(sample, "InputStream::pull", state_->opt.copy_output,
                                         &state_->opt.output_override);
  apply_default_port_name(out, state_->src_opt);
  if (pipeline_internal::env_bool("SIMA_INPUTSTREAM_POOL_DEBUG", false)) {
    auto log_tensor_storage = [](const simaai::neat::Sample& s) {
      if (!s.tensor || !s.tensor->storage)
        return;
      const bool has_holder = static_cast<bool>(s.tensor->storage->holder);
      std::fprintf(stderr, "[INPUTSTREAM] out tensor storage_kind=%d holder=%s\n",
                   static_cast<int>(s.tensor->storage->kind), has_holder ? "true" : "false");
    };
    if (out.kind == SampleKind::Bundle) {
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
  if (timings)
    t_decode_end = std::chrono::steady_clock::now();
  const bool pool_dbg = pipeline_internal::env_bool("SIMA_INPUTSTREAM_POOL_DEBUG", false);
  GstBuffer* buf = gst_sample_get_buffer(sample);
  GstBufferPool* pool =
      (buf && buf->pool && GST_IS_BUFFER_POOL(buf->pool) && is_simaai_pool(buf->pool)) ? buf->pool
                                                                                       : nullptr;
  const guint sample_ref = GST_MINI_OBJECT_REFCOUNT_VALUE(sample);
  const guint buf_ref = buf ? GST_MINI_OBJECT_REFCOUNT_VALUE(buf) : 0u;
  const gboolean buf_writable = buf ? gst_buffer_is_writable(buf) : FALSE;
  const bool force_release = pipeline_internal::env_bool("SIMA_FORCE_POOL_RELEASE", false) &&
                             state_->opt.copy_output && state_->opt.appsink_max_buffers <= 1 &&
                             pool != nullptr;
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
                 "pool=%p buf_writable=%s force_release=%s\n",
                 sample_ref, static_cast<void*>(buf), buf_ref,
                 buf ? static_cast<void*>(buf->pool) : nullptr, buf_writable ? "true" : "false",
                 force_release ? "true" : "false");
  }

  GstBuffer* pool_buf = nullptr;
  if (force_release && buf) {
    pool_buf = gst_buffer_ref(buf);
  }

  if (force_release && state_->appsink) {
    maybe_drop_appsink_last_sample(state_->appsink);
  }

  gst_sample_unref(sample);

  if (pool_buf && pool) {
    const guint refcnt = GST_MINI_OBJECT_REFCOUNT_VALUE(pool_buf);
    const gboolean writable = gst_buffer_is_writable(pool_buf);
    if (pool_dbg) {
      std::fprintf(stderr,
                   "[INPUTSTREAM] force pool release buffer=%p pool=%p type=%s "
                   "refcnt=%u writable=%s\n",
                   static_cast<void*>(pool_buf), static_cast<void*>(pool), G_OBJECT_TYPE_NAME(pool),
                   refcnt, writable ? "true" : "false");
    }
    gst_buffer_pool_release_buffer(pool, pool_buf);
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

Sample InputStream::push_and_pull(const cv::Mat& input, int timeout_ms) {
  push(input);
  return pull(timeout_ms);
}

Sample InputStream::push_and_pull(const simaai::neat::Tensor& input, int timeout_ms) {
  push(input);
  return pull(timeout_ms);
}

Sample InputStream::push_and_pull_holder(const std::shared_ptr<void>& holder, int timeout_ms) {
  push_holder(holder);
  return pull(timeout_ms);
}
} // namespace simaai::neat
