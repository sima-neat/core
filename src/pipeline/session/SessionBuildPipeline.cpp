/**
 * @file SessionBuildPipeline.cpp
 * @brief Shared pipeline state-change helpers for Session build paths.
 */

#include "SessionDetail.h"
#include "internal/SessionBuildInternal.h"

#include "nodes/io/RTSPInput.h"
#include "nodes/rtp/H264CapsFixup.h"
#include "pipeline/ErrorCodes.h"
#include "pipeline/SessionError.h"
#include "pipeline/internal/sima/SimaPluginStaticManifest.h"
#include "pipeline/internal/sima/SimaPluginStaticManifestResolver.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>
#include <mutex>
#include <sstream>
#include <string_view>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gst/sdp/sdp.h>

namespace simaai::neat {

namespace {

struct StateChangeTask {
  GstElement* pipeline = nullptr;
  GstState target = GST_STATE_NULL;
  GstStateChangeReturn result = GST_STATE_CHANGE_FAILURE;
  bool done = false;
  std::mutex mu;
  std::condition_variable cv;
};

GstStateChangeReturn set_state_with_timeout(GstElement* pipeline, GstState target, int timeout_ms,
                                            bool* timed_out) {
  if (timed_out)
    *timed_out = false;
  if (timeout_ms <= 0) {
    return gst_element_set_state(pipeline, target);
  }

  auto task = std::make_shared<StateChangeTask>();
  task->pipeline = pipeline;
  task->target = target;

  gst_object_ref(pipeline);
  std::thread([task]() {
    task->result = gst_element_set_state(task->pipeline, task->target);
    {
      std::lock_guard<std::mutex> lk(task->mu);
      task->done = true;
    }
    task->cv.notify_one();
    gst_object_unref(task->pipeline);
  }).detach();

  std::unique_lock<std::mutex> lk(task->mu);
  if (!task->cv.wait_for(lk, std::chrono::milliseconds(timeout_ms), [&]() { return task->done; })) {
    if (timed_out)
      *timed_out = true;
    return GST_STATE_CHANGE_FAILURE;
  }
  return task->result;
}

} // namespace

void set_state_or_throw(GstElement* pipeline, GstState target, const char* where,
                        const std::shared_ptr<DiagCtx>& diag) {
  if (!pipeline) {
    session_build_throw_session_error_simple(error_codes::kPipelineShape,
                                             std::string(where) + ": pipeline is null");
  }

  while (true) {
    const int timeout_ms = env_int("SIMA_STATE_CHANGE_TIMEOUT_MS", 15000);
    bool timed_out = false;
    GstStateChangeReturn r = set_state_with_timeout(pipeline, target, timeout_ms, &timed_out);
    if (timed_out) {
      maybe_dump_dot(pipeline, std::string(where) + "_set_state_timeout");
      if (env_bool("SIMA_DISPATCHER_TRACE", false)) {
        std::fprintf(stderr, "[TRACE] %s: set_state TIMEOUT after %d ms\n", where, timeout_ms);
      }
      session_build_dump_pipeline_string_force(
          diag, std::string(where).append("_set_state_timeout").c_str());
      session_build_dump_flow_snapshot(diag,
                                       std::string(where).append("_set_state_timeout").c_str());
      SessionReport rep = diag ? diag->snapshot_basic() : SessionReport{};
      rep.error_code = error_codes::kCaps;
      rep.repro_note = std::string(where) + ": set_state timed out after " +
                       std::to_string(timeout_ms) + " ms.\n" + boundary_summary_local(diag);
      throw SessionError(session_build_decorate_with_error_code(rep.error_code, rep.repro_note),
                         std::move(rep));
    }
    if (r == GST_STATE_CHANGE_FAILURE) {
      try {
        session_build_throw_if_bus_error_local(pipeline, diag, where);
      } catch (const SessionError& e) {
        if (env_bool("SIMA_DISPATCHER_TRACE", false)) {
          std::fprintf(stderr, "[TRACE] %s: SessionError: %s\n", where, e.what());
        }
        session_build_dump_pipeline_string_force(
            diag, std::string(where).append("_set_state_failure").c_str());
        session_build_dump_flow_snapshot(diag,
                                         std::string(where).append("_set_state_failure").c_str());
        throw;
      } catch (const std::exception& e) {
        if (env_bool("SIMA_DISPATCHER_TRACE", false)) {
          std::fprintf(stderr, "[TRACE] %s: exception: %s\n", where, e.what());
        }
      }
      maybe_dump_dot(pipeline, std::string(where) + "_set_state_failure");
      if (env_bool("SIMA_DISPATCHER_TRACE", false)) {
        std::fprintf(stderr, "[TRACE] %s: set_state FAILURE\n", where);
      }
      session_build_dump_pipeline_string_force(
          diag, std::string(where).append("_set_state_failure").c_str());
      session_build_dump_flow_snapshot(diag,
                                       std::string(where).append("_set_state_failure").c_str());
      SessionReport rep = diag ? diag->snapshot_basic() : SessionReport{};
      rep.error_code = error_codes::kCaps;
      rep.repro_note =
          std::string(where) + ": failed to set state.\n" + boundary_summary_local(diag);
      throw SessionError(session_build_decorate_with_error_code(rep.error_code, rep.repro_note),
                         std::move(rep));
    }

    GstState cur = GST_STATE_VOID_PENDING;
    GstState pend = GST_STATE_VOID_PENDING;
    gst_element_get_state(pipeline, &cur, &pend, 2 * GST_SECOND);

    try {
      session_build_throw_if_bus_error_local(pipeline, diag, where);
    } catch (const SessionError& e) {
      if (env_bool("SIMA_DISPATCHER_TRACE", false)) {
        std::fprintf(stderr, "[TRACE] %s: SessionError: %s\n", where, e.what());
      }
      session_build_dump_pipeline_string_force(
          diag, std::string(where).append("_set_state_error").c_str());
      session_build_dump_flow_snapshot(diag, std::string(where).append("_set_state_error").c_str());
      throw;
    } catch (const std::exception& e) {
      if (env_bool("SIMA_DISPATCHER_TRACE", false)) {
        std::fprintf(stderr, "[TRACE] %s: exception: %s\n", where, e.what());
      }
      session_build_dump_pipeline_string_force(
          diag, std::string(where).append("_set_state_exception").c_str());
      session_build_dump_flow_snapshot(diag,
                                       std::string(where).append("_set_state_exception").c_str());
      throw;
    }
    session_build_drain_bus_into_diag(pipeline, diag);

    if (env_bool("SIMA_INPUTSTREAM_DEBUG", false)) {
      std::fprintf(stderr, "[DBG] %s state cur=%s pending=%s\n", where,
                   gst_element_state_get_name(cur), gst_element_state_get_name(pend));
    }
    return;
  }
}

namespace {
static GstElement* parse_pipeline_or_throw(const std::string& pipeline_string, const char* where) {
  GError* err = nullptr;
  GstElement* pipeline = gst_parse_launch(pipeline_string.c_str(), &err);
  const bool had_pipeline = (pipeline != nullptr);
  const bool parse_error = (err != nullptr) || !had_pipeline || !GST_IS_BIN(pipeline);

  std::string msg;
  if (err && err->message) {
    msg = err->message;
  }
  if (err)
    g_error_free(err);

  if (parse_error) {
    if (had_pipeline) {
      gst_object_unref(pipeline);
    }
    if (msg.empty()) {
      msg = had_pipeline ? "parser returned non-bin root element" : "unknown";
    }
    session_build_throw_session_error_simple(
        error_codes::kParseLaunch,
        std::string(where ? where : "Session::build") + ": gst_parse_launch failed: " + msg +
            "\nPipeline:\n" + pipeline_string,
        "Validate pipeline fragments and plugin availability (gst-inspect-1.0).", pipeline_string);
  }

  {
    using namespace simaai::neat::pipeline_internal::sima;
    ManifestBuildDiagnostics manifest_diag;
    const SimaPluginStaticManifest manifest =
        resolve_manifest_from_pipeline(pipeline_string, /*session_id=*/"", &manifest_diag);

    for (const auto& warning : manifest_diag.warnings) {
      if (env_bool("SIMA_MANIFEST_DEBUG", false)) {
        std::fprintf(stderr, "[DBG] %s: manifest warning: %s\n", where ? where : "Session::build",
                     warning.c_str());
      }
    }
    if (!manifest_diag.errors.empty()) {
      std::ostringstream oss;
      oss << (where ? where : "Session::build") << ": static manifest resolution failed:\n";
      for (const auto& error : manifest_diag.errors) {
        oss << "  - " << error << '\n';
      }
      gst_object_unref(pipeline);
      session_build_throw_session_error_simple(error_codes::kPipelineShape, oss.str(),
                                               "Fix manifest field ownership/precedence issues or "
                                               "disable strict manifest mode for transition flows.",
                                               pipeline_string);
    }

    if (!manifest.stages.empty()) {
      std::string attach_error;
      if (!attach_manifest_context(pipeline, manifest, &attach_error)) {
        gst_object_unref(pipeline);
        session_build_throw_session_error_simple(
            error_codes::kPipelineShape,
            std::string(where ? where : "Session::build") +
                ": failed to attach sima static manifest context: " + attach_error,
            "Ensure model/static config files are readable and context injection is enabled.",
            pipeline_string);
      }
    }
  }

  return pipeline;
}

static int parse_sdp_fps_value(const char* value) {
  if (!value || !*value)
    return 0;
  char* end = nullptr;
  const double num = std::strtod(value, &end);
  if (end == value)
    return 0;
  double fps = num;
  if (*end == '/') {
    char* end_den = nullptr;
    const double den = std::strtod(end + 1, &end_den);
    if (end_den != end + 1 && den > 0.0) {
      fps = num / den;
    }
  }
  if (fps <= 0.0)
    return 0;
  const double rounded = std::round(fps);
  if (std::fabs(fps - rounded) > 0.1)
    return 0;
  return static_cast<int>(rounded);
}

static int parse_sdp_fps(const GstSDPMessage* sdp) {
  if (!sdp)
    return 0;
  const char* session_fps = gst_sdp_message_get_attribute_val(sdp, "framerate");
  int fps = parse_sdp_fps_value(session_fps);
  if (fps > 0)
    return fps;

  const guint media_count = gst_sdp_message_medias_len(sdp);
  for (guint i = 0; i < media_count; ++i) {
    const GstSDPMedia* media = gst_sdp_message_get_media(sdp, i);
    if (!media)
      continue;
    const char* media_type = gst_sdp_media_get_media(media);
    if (media_type && std::strcmp(media_type, "video") != 0)
      continue;
    const char* media_fps = gst_sdp_media_get_attribute_val(media, "framerate");
    fps = parse_sdp_fps_value(media_fps);
    if (fps > 0)
      return fps;
  }

  return 0;
}

struct H264CapsFixupCtx {
  std::string element_name;
  std::string rtsp_element_name;
  int fallback_fps = -1;
  int fallback_width = -1;
  int fallback_height = -1;
  std::atomic<int> sdp_fps{0};
  std::atomic<int> derived_fps{0};
  std::atomic<int> derived_width{0};
  std::atomic<int> derived_height{0};
  GstClockTime last_pts = GST_CLOCK_TIME_NONE;
  int fps_samples = 0;
  GstClockTime fps_total_delta = 0;
  bool logged = false;
  bool error_reported = false;
  bool derived_logged = false;
  bool derived_dims_logged = false;
  bool sprop_logged = false;
};

static bool h264_sdp_dump_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  const char* env = std::getenv("SIMA_H264_SDP_DUMP");
  if (env && *env && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 &&
      std::strcmp(env, "no") != 0) {
    cached = 1;
  } else {
    cached = 0;
  }
  return cached == 1;
}

static bool h264_sps_fixup_stream_enabled() {
  static int cached = -1;
  if (cached >= 0) {
    return cached == 1;
  }
  const char* env = std::getenv("SIMA_H264_SPS_FIXUP_STREAM");
  if (env && *env && std::strcmp(env, "0") != 0 && std::strcmp(env, "false") != 0 &&
      std::strcmp(env, "no") != 0) {
    cached = 1;
  } else {
    cached = 0;
  }
  return cached == 1;
}

struct H264BitReader {
  const uint8_t* data = nullptr;
  size_t size = 0;
  size_t bitpos = 0;

  bool read_bits(unsigned n, uint32_t* out) {
    if (!out)
      return false;
    if (n == 0) {
      *out = 0;
      return true;
    }
    if (bitpos + n > size * 8)
      return false;
    uint32_t val = 0;
    for (unsigned i = 0; i < n; ++i) {
      const size_t idx = bitpos + i;
      const size_t byte = idx / 8;
      const unsigned shift = 7 - static_cast<unsigned>(idx % 8);
      val = (val << 1) | ((data[byte] >> shift) & 0x01);
    }
    bitpos += n;
    *out = val;
    return true;
  }

  bool read_ue(uint32_t* out) {
    if (!out)
      return false;
    unsigned zeros = 0;
    while (true) {
      uint32_t bit = 0;
      if (!read_bits(1, &bit))
        return false;
      if (bit == 1)
        break;
      ++zeros;
      if (zeros > 31)
        return false;
    }
    if (zeros == 0) {
      *out = 0;
      return true;
    }
    uint32_t rest = 0;
    if (!read_bits(zeros, &rest))
      return false;
    *out = ((1u << zeros) - 1u) + rest;
    return true;
  }

  bool read_se(int32_t* out) {
    if (!out)
      return false;
    uint32_t ue = 0;
    if (!read_ue(&ue))
      return false;
    int32_t val = (ue & 1) ? static_cast<int32_t>((ue + 1) / 2) : -static_cast<int32_t>(ue / 2);
    *out = val;
    return true;
  }
};

static bool h264_skip_scaling_list(H264BitReader& br, int size) {
  int last_scale = 8;
  int next_scale = 8;
  for (int i = 0; i < size; ++i) {
    if (next_scale != 0) {
      int32_t delta = 0;
      if (!br.read_se(&delta))
        return false;
      next_scale = (last_scale + delta + 256) % 256;
    }
    last_scale = (next_scale == 0) ? last_scale : next_scale;
  }
  return true;
}

static std::vector<uint8_t> h264_ebsp_to_rbsp(const uint8_t* data, size_t size) {
  std::vector<uint8_t> out;
  if (!data || size == 0)
    return out;
  out.reserve(size);
  int zero_count = 0;
  for (size_t i = 0; i < size; ++i) {
    const uint8_t b = data[i];
    if (zero_count >= 2 && b == 0x03) {
      zero_count = 0;
      continue;
    }
    out.push_back(b);
    if (b == 0x00) {
      ++zero_count;
    } else {
      zero_count = 0;
    }
  }
  return out;
}

static bool h264_parse_sps_dimensions(const uint8_t* data, size_t size, int* out_w, int* out_h) {
  if (!data || size == 0 || !out_w || !out_h)
    return false;

  H264BitReader br;
  br.data = data;
  br.size = size;

  uint32_t profile_idc = 0;
  uint32_t tmp = 0;
  if (!br.read_bits(8, &profile_idc))
    return false;
  if (!br.read_bits(8, &tmp))
    return false; // constraint flags + reserved
  if (!br.read_bits(8, &tmp))
    return false; // level_idc
  if (!br.read_ue(&tmp))
    return false; // seq_parameter_set_id

  int chroma_format_idc = 1;
  bool separate_colour_plane_flag = false;
  if (profile_idc == 100 || profile_idc == 110 || profile_idc == 122 || profile_idc == 244 ||
      profile_idc == 44 || profile_idc == 83 || profile_idc == 86 || profile_idc == 118 ||
      profile_idc == 128 || profile_idc == 138 || profile_idc == 144) {
    if (!br.read_ue(&tmp))
      return false;
    chroma_format_idc = static_cast<int>(tmp);
    if (chroma_format_idc == 3) {
      uint32_t sep = 0;
      if (!br.read_bits(1, &sep))
        return false;
      separate_colour_plane_flag = (sep != 0);
    }
    if (!br.read_ue(&tmp))
      return false; // bit_depth_luma_minus8
    if (!br.read_ue(&tmp))
      return false; // bit_depth_chroma_minus8
    if (!br.read_bits(1, &tmp))
      return false; // qpprime_y_zero_transform_bypass_flag
    uint32_t scaling = 0;
    if (!br.read_bits(1, &scaling))
      return false;
    if (scaling) {
      const int count = (chroma_format_idc != 3) ? 8 : 12;
      for (int i = 0; i < count; ++i) {
        uint32_t present = 0;
        if (!br.read_bits(1, &present))
          return false;
        if (present) {
          const int size = (i < 6) ? 16 : 64;
          if (!h264_skip_scaling_list(br, size))
            return false;
        }
      }
    }
  }

  if (!br.read_ue(&tmp))
    return false; // log2_max_frame_num_minus4
  uint32_t pic_order_cnt_type = 0;
  if (!br.read_ue(&pic_order_cnt_type))
    return false;
  if (pic_order_cnt_type == 0) {
    if (!br.read_ue(&tmp))
      return false; // log2_max_pic_order_cnt_lsb_minus4
  } else if (pic_order_cnt_type == 1) {
    if (!br.read_bits(1, &tmp))
      return false; // delta_pic_order_always_zero_flag
    int32_t se = 0;
    if (!br.read_se(&se))
      return false; // offset_for_non_ref_pic
    if (!br.read_se(&se))
      return false; // offset_for_top_to_bottom_field
    uint32_t num_ref = 0;
    if (!br.read_ue(&num_ref))
      return false;
    for (uint32_t i = 0; i < num_ref; ++i) {
      if (!br.read_se(&se))
        return false;
    }
  }

  if (!br.read_ue(&tmp))
    return false; // max_num_ref_frames
  if (!br.read_bits(1, &tmp))
    return false; // gaps_in_frame_num_value_allowed_flag

  uint32_t pic_width_in_mbs_minus1 = 0;
  uint32_t pic_height_in_map_units_minus1 = 0;
  if (!br.read_ue(&pic_width_in_mbs_minus1))
    return false;
  if (!br.read_ue(&pic_height_in_map_units_minus1))
    return false;

  uint32_t frame_mbs_only_flag = 0;
  if (!br.read_bits(1, &frame_mbs_only_flag))
    return false;
  if (!frame_mbs_only_flag) {
    if (!br.read_bits(1, &tmp))
      return false; // mb_adaptive_frame_field_flag
  }
  if (!br.read_bits(1, &tmp))
    return false; // direct_8x8_inference_flag

  uint32_t frame_cropping_flag = 0;
  if (!br.read_bits(1, &frame_cropping_flag))
    return false;
  uint32_t crop_left = 0, crop_right = 0, crop_top = 0, crop_bottom = 0;
  if (frame_cropping_flag) {
    if (!br.read_ue(&crop_left))
      return false;
    if (!br.read_ue(&crop_right))
      return false;
    if (!br.read_ue(&crop_top))
      return false;
    if (!br.read_ue(&crop_bottom))
      return false;
  }

  int width = static_cast<int>(pic_width_in_mbs_minus1 + 1) * 16;
  int height = static_cast<int>(pic_height_in_map_units_minus1 + 1) * 16;
  if (!frame_mbs_only_flag)
    height *= 2;

  int crop_unit_x = 1;
  int crop_unit_y = 2 - static_cast<int>(frame_mbs_only_flag);
  if (chroma_format_idc == 1) {
    crop_unit_x = 2;
    crop_unit_y = 2 * (2 - static_cast<int>(frame_mbs_only_flag));
  } else if (chroma_format_idc == 2) {
    crop_unit_x = 2;
    crop_unit_y = 2 - static_cast<int>(frame_mbs_only_flag);
  } else if (chroma_format_idc == 3 && !separate_colour_plane_flag) {
    crop_unit_x = 1;
    crop_unit_y = 2 - static_cast<int>(frame_mbs_only_flag);
  }

  if (frame_cropping_flag) {
    width -= static_cast<int>((crop_left + crop_right) * crop_unit_x);
    height -= static_cast<int>((crop_top + crop_bottom) * crop_unit_y);
  }

  if (width <= 0 || height <= 0)
    return false;
  *out_w = width;
  *out_h = height;
  return true;
}

static bool h264_extract_sprop_sps(const char* fmtp, std::string* out_sps_b64) {
  if (!fmtp || !out_sps_b64)
    return false;
  const char* key = "sprop-parameter-sets=";
  const char* pos = std::strstr(fmtp, key);
  if (!pos)
    return false;
  pos += std::strlen(key);
  while (*pos == ' ')
    ++pos;
  const char* end = pos;
  while (*end && *end != ';' && *end != '\r' && *end != '\n')
    ++end;
  if (end == pos)
    return false;
  std::string value(pos, static_cast<size_t>(end - pos));
  const size_t comma = value.find(',');
  if (comma != std::string::npos) {
    value = value.substr(0, comma);
  }
  while (!value.empty() && value.back() == ' ')
    value.pop_back();
  if (value.empty())
    return false;
  *out_sps_b64 = value;
  return true;
}

static bool h264_parse_sps_from_sprop(const char* fmtp, int* out_w, int* out_h) {
  if (!fmtp || !out_w || !out_h)
    return false;
  std::string sps_b64;
  if (!h264_extract_sprop_sps(fmtp, &sps_b64))
    return false;

  gsize raw_len = 0;
  guchar* raw = g_base64_decode(sps_b64.c_str(), &raw_len);
  if (!raw || raw_len <= 1) {
    if (raw)
      g_free(raw);
    return false;
  }

  std::vector<uint8_t> rbsp = h264_ebsp_to_rbsp(raw + 1, raw_len - 1);
  g_free(raw);
  if (rbsp.empty())
    return false;

  return h264_parse_sps_dimensions(rbsp.data(), rbsp.size(), out_w, out_h);
}

static bool h264_parse_sps_from_sdp_text(const char* text, int* out_w, int* out_h) {
  if (!text || !out_w || !out_h)
    return false;
  const char* key = "sprop-parameter-sets=";
  const char* pos = std::strstr(text, key);
  if (!pos)
    return false;
  return h264_parse_sps_from_sprop(pos, out_w, out_h);
}

static const uint8_t* h264_find_start_code(const uint8_t* p, const uint8_t* end, int* out_len) {
  if (!p || !end || p >= end)
    return nullptr;
  for (const uint8_t* cur = p; cur + 3 < end; ++cur) {
    if (cur[0] == 0x00 && cur[1] == 0x00) {
      if (cur[2] == 0x01) {
        if (out_len)
          *out_len = 3;
        return cur;
      }
      if (cur + 3 < end && cur[2] == 0x00 && cur[3] == 0x01) {
        if (out_len)
          *out_len = 4;
        return cur;
      }
    }
  }
  return nullptr;
}

static void h264_caps_fixup_try_derive_dims(GstPad* pad, GstPadProbeInfo* info,
                                            H264CapsFixupCtx* ctx) {
  if (!pad || !info || !ctx)
    return;
  if (!h264_sps_fixup_stream_enabled())
    return;
  if (ctx->derived_width.load() > 0 && ctx->derived_height.load() > 0)
    return;

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf)
    return;

  GstMapInfo map{};
  if (!gst_buffer_map(buf, &map, GST_MAP_READ))
    return;
  const uint8_t* data = map.data;
  const uint8_t* end = map.data + map.size;

  int sc_len = 0;
  const uint8_t* start = h264_find_start_code(data, end, &sc_len);
  while (start) {
    const uint8_t* nal_start = start + sc_len;
    if (nal_start >= end)
      break;
    int next_len = 0;
    const uint8_t* next = h264_find_start_code(nal_start, end, &next_len);
    const uint8_t* nal_end = next ? next : end;
    const size_t nal_size = static_cast<size_t>(nal_end - nal_start);
    if (nal_size > 1) {
      const uint8_t nal_type = nal_start[0] & 0x1F;
      if (nal_type == 7) {
        std::vector<uint8_t> rbsp = h264_ebsp_to_rbsp(nal_start + 1, nal_size - 1);
        int w = 0;
        int h = 0;
        if (!rbsp.empty() && h264_parse_sps_dimensions(rbsp.data(), rbsp.size(), &w, &h)) {
          int expected = 0;
          if (ctx->derived_width.compare_exchange_strong(expected, w)) {
            ctx->derived_height.store(h);
          } else if (ctx->derived_height.load() <= 0) {
            ctx->derived_height.store(h);
          }
          gst_buffer_unmap(buf, &map);

          GstCaps* caps = gst_pad_get_current_caps(pad);
          if (caps && !gst_caps_is_any(caps) && !gst_caps_is_empty(caps)) {
            GstCaps* new_caps = gst_caps_copy(caps);
            gst_caps_unref(caps);
            bool changed = false;
            const guint n = gst_caps_get_size(new_caps);
            for (guint i = 0; i < n; ++i) {
              GstStructure* s = gst_caps_get_structure(new_caps, i);
              if (!s)
                continue;
              const char* media = gst_structure_get_name(s);
              if (!media || std::strcmp(media, "video/x-h264") != 0)
                continue;
              int cw = 0;
              int ch = 0;
              if (!gst_structure_get_int(s, "width", &cw) || cw <= 0) {
                gst_structure_set(s, "width", G_TYPE_INT, w, nullptr);
                changed = true;
              }
              if (!gst_structure_get_int(s, "height", &ch) || ch <= 0) {
                gst_structure_set(s, "height", G_TYPE_INT, h, nullptr);
                changed = true;
              }
            }
            if (changed) {
              GstEvent* ev = gst_event_new_caps(new_caps);
              gst_caps_unref(new_caps);
              gst_pad_push_event(pad, ev);
            } else {
              gst_caps_unref(new_caps);
            }
          } else if (caps) {
            gst_caps_unref(caps);
          }

          if (!ctx->derived_dims_logged) {
            std::fprintf(stderr, "[rtsp] %s: derived dimensions=%dx%d (sps)\n",
                         ctx->element_name.c_str(), w, h);
            ctx->derived_dims_logged = true;
          }
          return;
        }
      }
    }
    start = next;
  }

  gst_buffer_unmap(buf, &map);
}

static void h264_caps_fixup_on_sdp(GstElement* /*src*/, GstSDPMessage* sdp, gpointer user_data) {
  auto* ctx = reinterpret_cast<H264CapsFixupCtx*>(user_data);
  if (!ctx || !sdp)
    return;

  if (h264_sdp_dump_enabled()) {
    gchar* text = gst_sdp_message_as_text(sdp);
    if (text) {
      std::fprintf(stderr, "[rtsp] %s: SDP:\n%s\n", ctx->element_name.c_str(), text);
      g_free(text);
    }
  }

  if (ctx->derived_width.load() <= 0 || ctx->derived_height.load() <= 0) {
    bool sprop_found = false;
    bool sprop_parsed = false;
    const int media_count = gst_sdp_message_medias_len(sdp);
    for (int i = 0; i < media_count; ++i) {
      const GstSDPMedia* media = gst_sdp_message_get_media(sdp, i);
      if (!media)
        continue;
      const char* media_type = gst_sdp_media_get_media(media);
      if (media_type && std::strcmp(media_type, "video") != 0)
        continue;
      const int attr_count = gst_sdp_media_attributes_len(media);
      for (int a = 0; a < attr_count; ++a) {
        const GstSDPAttribute* attr = gst_sdp_media_get_attribute(media, a);
        if (!attr || !attr->key || !attr->value)
          continue;
        if (std::strcmp(attr->key, "fmtp") != 0)
          continue;
        if (std::strstr(attr->value, "sprop-parameter-sets=") != nullptr) {
          sprop_found = true;
        }
        int w = 0;
        int h = 0;
        if (h264_parse_sps_from_sprop(attr->value, &w, &h)) {
          ctx->derived_width.store(w);
          ctx->derived_height.store(h);
          sprop_parsed = true;
          if (!ctx->derived_dims_logged) {
            std::fprintf(stderr, "[rtsp] %s: derived dimensions=%dx%d (sdp-sps)\n",
                         ctx->element_name.c_str(), w, h);
            ctx->derived_dims_logged = true;
          }
          a = attr_count; // break outer loop
          break;
        }
      }
      if (ctx->derived_width.load() > 0 && ctx->derived_height.load() > 0) {
        break;
      }
    }

    if (ctx->derived_width.load() <= 0 || ctx->derived_height.load() <= 0) {
      gchar* text = gst_sdp_message_as_text(sdp);
      if (text) {
        int w = 0;
        int h = 0;
        if (h264_parse_sps_from_sdp_text(text, &w, &h)) {
          ctx->derived_width.store(w);
          ctx->derived_height.store(h);
          sprop_found = true;
          sprop_parsed = true;
          if (!ctx->derived_dims_logged) {
            std::fprintf(stderr, "[rtsp] %s: derived dimensions=%dx%d (sdp-sps)\n",
                         ctx->element_name.c_str(), w, h);
            ctx->derived_dims_logged = true;
          }
        }
        g_free(text);
      }
    }

    if (!ctx->sprop_logged) {
      const char* state =
          sprop_found ? (sprop_parsed ? "present" : "present (parse failed)") : "missing";
      std::fprintf(stderr, "[rtsp] %s: sprop-parameter-sets %s\n", ctx->element_name.c_str(),
                   state);
      ctx->sprop_logged = true;
    }
  }

  const int fps = parse_sdp_fps(sdp);
  if (fps <= 0)
    return;
  int expected = 0;
  ctx->sdp_fps.compare_exchange_strong(expected, fps);
}

static bool h264_caps_fixup_apply(GstPad* pad, GstStructure* s, H264CapsFixupCtx* ctx,
                                  bool* out_changed, int* out_log_fps,
                                  const char** out_log_source) {
  if (!s || !ctx)
    return false;

  bool changed = false;
  const char* fps_source = nullptr;
  int applied_fps = 0;

  int width = 0;
  if (!gst_structure_get_int(s, "width", &width) || width <= 0) {
    int derived = ctx->derived_width.load();
    if (derived > 0) {
      gst_structure_set(s, "width", G_TYPE_INT, derived, nullptr);
      changed = true;
    } else if (ctx->fallback_width > 0) {
      gst_structure_set(s, "width", G_TYPE_INT, ctx->fallback_width, nullptr);
      changed = true;
    }
  }

  int height = 0;
  if (!gst_structure_get_int(s, "height", &height) || height <= 0) {
    int derived = ctx->derived_height.load();
    if (derived > 0) {
      gst_structure_set(s, "height", G_TYPE_INT, derived, nullptr);
      changed = true;
    } else if (ctx->fallback_height > 0) {
      gst_structure_set(s, "height", G_TYPE_INT, ctx->fallback_height, nullptr);
      changed = true;
    }
  }

  int num = 0;
  int den = 0;
  const bool has_fps =
      gst_structure_get_fraction(s, "framerate", &num, &den) && den == 1 && num > 0;
  if (!has_fps) {
    applied_fps = ctx->sdp_fps.load();
    if (applied_fps > 0) {
      fps_source = "sdp";
    } else if (ctx->derived_fps.load() > 0) {
      applied_fps = ctx->derived_fps.load();
      fps_source = "derived";
    } else if (ctx->fallback_fps > 0) {
      applied_fps = ctx->fallback_fps;
      fps_source = "fallback";
    }
    if (applied_fps <= 0) {
      if (out_changed)
        *out_changed = changed;
      if (out_log_fps)
        *out_log_fps = 0;
      if (out_log_source)
        *out_log_source = "missing";
      return true;
    }
    gst_structure_set(s, "framerate", GST_TYPE_FRACTION, applied_fps, 1, nullptr);
    changed = true;
  }

  if (out_changed)
    *out_changed = changed;
  if (out_log_fps)
    *out_log_fps = has_fps ? num : applied_fps;
  if (out_log_source) {
    *out_log_source = fps_source ? fps_source : (has_fps ? "caps" : "missing");
  }
  return true;
}

static void h264_caps_fixup_try_derive_fps(GstPad* pad, GstPadProbeInfo* info,
                                           H264CapsFixupCtx* ctx) {
  if (!pad || !info || !ctx)
    return;
  if (ctx->derived_fps.load() > 0)
    return;

  GstBuffer* buf = GST_PAD_PROBE_INFO_BUFFER(info);
  if (!buf)
    return;

  h264_caps_fixup_try_derive_dims(pad, info, ctx);

  GstClockTime pts = GST_BUFFER_PTS(buf);
  if (!GST_CLOCK_TIME_IS_VALID(pts)) {
    pts = GST_BUFFER_DTS(buf);
  }
  if (!GST_CLOCK_TIME_IS_VALID(pts))
    return;

  if (ctx->last_pts != GST_CLOCK_TIME_NONE && pts > ctx->last_pts) {
    const GstClockTime delta = pts - ctx->last_pts;
    ctx->fps_total_delta += delta;
    ctx->fps_samples += 1;
  }
  ctx->last_pts = pts;

  if (ctx->fps_samples < 5)
    return;

  const double avg_delta =
      static_cast<double>(ctx->fps_total_delta) / static_cast<double>(ctx->fps_samples);
  if (avg_delta <= 0.0)
    return;

  const int fps = static_cast<int>(std::lround(GST_SECOND / avg_delta));
  if (fps <= 0 || fps > 240)
    return;

  int expected = 0;
  if (!ctx->derived_fps.compare_exchange_strong(expected, fps))
    return;

  GstCaps* caps = gst_pad_get_current_caps(pad);
  if (!caps || gst_caps_is_any(caps) || gst_caps_is_empty(caps)) {
    if (caps)
      gst_caps_unref(caps);
    return;
  }

  GstCaps* new_caps = gst_caps_copy(caps);
  gst_caps_unref(caps);

  bool changed = false;
  const guint n = gst_caps_get_size(new_caps);
  for (guint i = 0; i < n; ++i) {
    GstStructure* s = gst_caps_get_structure(new_caps, i);
    if (!s)
      continue;
    const char* media = gst_structure_get_name(s);
    if (!media || std::strcmp(media, "video/x-h264") != 0)
      continue;
    int num = 0;
    int den = 0;
    const bool has_fps =
        gst_structure_get_fraction(s, "framerate", &num, &den) && den == 1 && num > 0;
    if (!has_fps) {
      gst_structure_set(s, "framerate", GST_TYPE_FRACTION, fps, 1, nullptr);
      changed = true;
    }
  }

  if (!changed) {
    gst_caps_unref(new_caps);
    return;
  }

  GstEvent* new_ev = gst_event_new_caps(new_caps);
  gst_caps_unref(new_caps);
  gst_pad_push_event(pad, new_ev);

  if (!ctx->derived_logged) {
    std::fprintf(stderr, "[rtsp] %s: derived framerate=%d (stream)\n", ctx->element_name.c_str(),
                 fps);
    ctx->derived_logged = true;
  }
}

static GstPadProbeReturn h264_caps_fixup_cb(GstPad* pad, GstPadProbeInfo* info,
                                            gpointer user_data) {
  auto* ctx = reinterpret_cast<H264CapsFixupCtx*>(user_data);
  if (!ctx)
    return GST_PAD_PROBE_OK;

  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) != 0) {
    h264_caps_fixup_try_derive_fps(pad, info, ctx);
    return GST_PAD_PROBE_OK;
  }

  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM) != 0) {
    GstQuery* query = GST_PAD_PROBE_INFO_QUERY(info);
    if (!query || GST_QUERY_TYPE(query) != GST_QUERY_ACCEPT_CAPS) {
      return GST_PAD_PROBE_OK;
    }

    GstCaps* caps = nullptr;
    gst_query_parse_accept_caps(query, &caps);
    if (!caps || gst_caps_get_size(caps) < 1)
      return GST_PAD_PROBE_OK;

    GstCaps* new_caps = gst_caps_copy(caps);
    GstStructure* s = gst_caps_get_structure(new_caps, 0);
    if (!s) {
      gst_caps_unref(new_caps);
      return GST_PAD_PROBE_OK;
    }

    const char* media = gst_structure_get_name(s);
    if (!media || std::strcmp(media, "video/x-h264") != 0) {
      gst_caps_unref(new_caps);
      return GST_PAD_PROBE_OK;
    }

    bool changed = false;
    int log_fps = 0;
    const char* log_source = nullptr;
    if (!h264_caps_fixup_apply(pad, s, ctx, &changed, &log_fps, &log_source)) {
      gst_caps_unref(new_caps);
      return GST_PAD_PROBE_OK;
    }

    if (!changed) {
      gst_caps_unref(new_caps);
      return GST_PAD_PROBE_OK;
    }

    if (!ctx->logged) {
      int log_width = 0;
      int log_height = 0;
      gst_structure_get_int(s, "width", &log_width);
      gst_structure_get_int(s, "height", &log_height);
      std::fprintf(stderr, "[rtsp] %s: H264 caps missing fields; applying %dx%d@%d (%s)\n",
                   ctx->element_name.c_str(), log_width, log_height, log_fps, log_source);
      ctx->logged = true;
    }

    GstQuery* new_query = gst_query_new_accept_caps(new_caps);
    gboolean ok = gst_pad_peer_query(pad, new_query);
    gboolean result = FALSE;
    if (ok) {
      gst_query_parse_accept_caps_result(new_query, &result);
    }
    gst_query_set_accept_caps_result(query, result);
    gst_query_unref(new_query);
    gst_caps_unref(new_caps);
    return GST_PAD_PROBE_HANDLED;
  }

  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) == 0)
    return GST_PAD_PROBE_OK;

  GstEvent* ev = GST_PAD_PROBE_INFO_EVENT(info);
  if (!ev || GST_EVENT_TYPE(ev) != GST_EVENT_CAPS)
    return GST_PAD_PROBE_OK;

  GstCaps* caps = nullptr;
  gst_event_parse_caps(ev, &caps);
  if (!caps || gst_caps_get_size(caps) < 1)
    return GST_PAD_PROBE_OK;

  GstCaps* new_caps = gst_caps_copy(caps);
  GstStructure* s = gst_caps_get_structure(new_caps, 0);
  if (!s) {
    gst_caps_unref(new_caps);
    return GST_PAD_PROBE_OK;
  }

  const char* media = gst_structure_get_name(s);
  if (!media || std::strcmp(media, "video/x-h264") != 0) {
    gst_caps_unref(new_caps);
    return GST_PAD_PROBE_OK;
  }

  bool changed = false;
  int log_fps = 0;
  const char* log_source = nullptr;
  if (!h264_caps_fixup_apply(pad, s, ctx, &changed, &log_fps, &log_source)) {
    gst_caps_unref(new_caps);
    gst_event_unref(ev);
    return GST_PAD_PROBE_DROP;
  }

  if (!changed) {
    gst_caps_unref(new_caps);
    return GST_PAD_PROBE_OK;
  }

  if (!ctx->logged) {
    int log_width = 0;
    int log_height = 0;
    gst_structure_get_int(s, "width", &log_width);
    gst_structure_get_int(s, "height", &log_height);
    std::fprintf(stderr, "[rtsp] %s: H264 caps missing fields; applying %dx%d@%d (%s)\n",
                 ctx->element_name.c_str(), log_width, log_height, log_fps, log_source);
    ctx->logged = true;
  }

  GstEvent* new_ev = gst_event_new_caps(new_caps);
  gst_caps_unref(new_caps);
  gst_event_unref(ev);
  GST_PAD_PROBE_INFO_DATA(info) = new_ev;
  return GST_PAD_PROBE_OK;
}

static std::string find_rtsp_input_name_for_fixup(const std::vector<std::shared_ptr<Node>>& nodes,
                                                  size_t fixup_index,
                                                  const NameTransform& name_transform) {
  if (fixup_index == 0)
    return {};
  for (size_t i = fixup_index; i-- > 0;) {
    const auto* rtsp = dynamic_cast<const RTSPInput*>(nodes[i].get());
    if (!rtsp)
      continue;
    const auto names =
        apply_name_transform(name_transform, nodes[i]->element_names(static_cast<int>(i)));
    if (!names.empty())
      return names[0];
  }
  return {};
}

static std::string
find_h264_capsfilter_name_for_fixup(const std::vector<std::shared_ptr<Node>>& nodes,
                                    size_t fixup_index, const NameTransform& name_transform) {
  if (fixup_index == 0)
    return {};
  for (size_t i = fixup_index; i-- > 0;) {
    if (nodes[i]->kind() != "H264Depacketize")
      continue;
    const auto names =
        apply_name_transform(name_transform, nodes[i]->element_names(static_cast<int>(i)));
    if (!names.empty())
      return names.back();
  }
  return {};
}

static H264CapsFixupCtx* make_h264_caps_fixup_ctx(GstElement* pipeline, const H264CapsFixup& fix,
                                                  const std::string& element_name,
                                                  const std::string& rtsp_element_name) {
  auto* ctx = new H264CapsFixupCtx();
  ctx->element_name = element_name;
  ctx->fallback_fps = fix.fallback_fps();
  ctx->fallback_width = fix.fallback_width();
  ctx->fallback_height = fix.fallback_height();
  ctx->rtsp_element_name = rtsp_element_name;
  if (!ctx->rtsp_element_name.empty()) {
    GstElement* rtspsrc = gst_bin_get_by_name(GST_BIN(pipeline), ctx->rtsp_element_name.c_str());
    if (rtspsrc) {
      g_signal_connect(rtspsrc, "on-sdp", G_CALLBACK(h264_caps_fixup_on_sdp), ctx);
      gst_object_unref(rtspsrc);
    }
  }
  return ctx;
}

struct RtspDebugCtx {
  std::string element_name;
  std::atomic<bool> stop{false};
  std::thread poller;
  bool poller_started = false;
};

static bool rtsp_stats_debug_enabled() {
  return env_bool("SIMA_RTSP_STATS_DEBUG", false);
}

static int rtsp_stats_worker_poll_ms() {
  return env_int("SIMA_RTSP_STATS_POLL_MS", 0);
}

static GQuark rtsp_debug_quark() {
  static GQuark q = g_quark_from_static_string("sima-rtsp-debug");
  return q;
}

static void stop_rtsp_poller(RtspDebugCtx* ctx) {
  if (!ctx)
    return;
  ctx->stop.store(true, std::memory_order_release);
  if (ctx->poller_started && ctx->poller.joinable()) {
    if (ctx->poller.get_id() == std::this_thread::get_id()) {
      ctx->poller.detach();
    } else {
      ctx->poller.join();
    }
  }
}

static RtspDebugCtx* ensure_rtsp_debug_ctx(GstElement* src, const std::string& name) {
  if (!src)
    return nullptr;
  auto* ctx = static_cast<RtspDebugCtx*>(g_object_get_qdata(G_OBJECT(src), rtsp_debug_quark()));
  if (ctx)
    return ctx;
  ctx = new RtspDebugCtx();
  ctx->element_name = name;
  g_object_set_qdata_full(
      G_OBJECT(src), rtsp_debug_quark(), ctx, +[](gpointer p) {
        auto* ctx = static_cast<RtspDebugCtx*>(p);
        stop_rtsp_poller(ctx);
        delete ctx;
      });
  return ctx;
}

static void rtsp_log_stats(GstElement* src, const RtspDebugCtx* ctx, const char* tag) {
  if (!rtsp_stats_debug_enabled() || !src || !ctx)
    return;
  GstState state = GST_STATE_NULL;
  GstState pending = GST_STATE_VOID_PENDING;
  gst_element_get_state(src, &state, &pending, 0);

  GstStructure* stats = nullptr;
  const GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(src), "stats");
  if (pspec) {
    g_object_get(src, "stats", &stats, nullptr);
  }
  if (stats) {
    gchar* s = gst_structure_to_string(stats);
    std::fprintf(stderr, "[rtsp] %s name=%s state=%s pending=%s stats=%s\n", tag ? tag : "stats",
                 ctx->element_name.c_str(), gst_element_state_get_name(state),
                 gst_element_state_get_name(pending), s ? s : "<null>");
    g_free(s);
    gst_structure_free(stats);
  } else {
    std::fprintf(stderr, "[rtsp] %s name=%s state=%s pending=%s stats=<none>\n",
                 tag ? tag : "stats", ctx->element_name.c_str(), gst_element_state_get_name(state),
                 gst_element_state_get_name(pending));
  }
}

static void rtsp_timeout_cb(GstElement* src, guint64 timeout, gpointer /*user_data*/) {
  if (!rtsp_stats_debug_enabled())
    return;
  auto* ctx = static_cast<RtspDebugCtx*>(g_object_get_qdata(G_OBJECT(src), rtsp_debug_quark()));
  if (!ctx)
    return;
  std::fprintf(stderr, "[rtsp] timeout name=%s timeout_ns=%" G_GUINT64_FORMAT "\n",
               ctx->element_name.c_str(), timeout);
  rtsp_log_stats(src, ctx, "timeout");
}

static void rtsp_connection_closed_cb(GstElement* src, gpointer /*user_data*/) {
  if (!rtsp_stats_debug_enabled())
    return;
  auto* ctx = static_cast<RtspDebugCtx*>(g_object_get_qdata(G_OBJECT(src), rtsp_debug_quark()));
  if (!ctx)
    return;
  std::fprintf(stderr, "[rtsp] connection-closed name=%s\n", ctx->element_name.c_str());
  rtsp_log_stats(src, ctx, "connection-closed");
}

static bool rtsp_try_connect_signal(GstElement* src, const char* name,
                                    const std::vector<GType>& params, GCallback cb) {
  if (!src || !name || !cb)
    return false;
  const guint id = g_signal_lookup(name, G_OBJECT_TYPE(src));
  if (id == 0)
    return false;
  GSignalQuery query;
  g_signal_query(id, &query);
  if (query.n_params != params.size())
    return false;
  for (guint i = 0; i < query.n_params; ++i) {
    if (query.param_types[i] != params[i])
      return false;
  }
  g_signal_connect(src, name, cb, nullptr);
  return true;
}

static void maybe_start_rtsp_stats_poller(GstElement* src, RtspDebugCtx* ctx) {
  if (!src || !ctx)
    return;
  const int worker_poll_ms = rtsp_stats_worker_poll_ms();
  if (worker_poll_ms <= 0)
    return;
  if (ctx->poller_started)
    return;
  const GParamSpec* pspec = g_object_class_find_property(G_OBJECT_GET_CLASS(src), "stats");
  if (!pspec)
    return;
  ctx->poller_started = true;
  ctx->stop.store(false, std::memory_order_release);
  GstElement* src_ref = GST_ELEMENT(gst_object_ref(src));
  ctx->poller = std::thread([src_ref, ctx, worker_poll_ms]() {
    while (!ctx->stop.load(std::memory_order_acquire)) {
      rtsp_log_stats(src_ref, ctx, "poll");
      const int step = std::max(1, worker_poll_ms);
      for (int waited = 0; waited < step; waited += 50) {
        if (ctx->stop.load(std::memory_order_acquire))
          break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
      }
    }
    gst_object_unref(src_ref);
  });
}

static void rtsp_pad_added(GstElement* src, GstPad* pad, gpointer /*user_data*/) {
  if (!rtsp_stats_debug_enabled())
    return;
  auto* ctx = static_cast<RtspDebugCtx*>(g_object_get_qdata(G_OBJECT(src), rtsp_debug_quark()));
  if (!ctx)
    return;
  const char* pad_name = gst_pad_get_name(pad);
  GstCaps* caps = gst_pad_get_current_caps(pad);
  if (!caps)
    caps = gst_pad_query_caps(pad, nullptr);
  gchar* caps_str = caps ? gst_caps_to_string(caps) : nullptr;
  std::fprintf(stderr, "[rtsp] pad-added name=%s pad=%s caps=%s\n", ctx->element_name.c_str(),
               pad_name ? pad_name : "<null>", caps_str ? caps_str : "<none>");
  if (caps)
    gst_caps_unref(caps);
  g_free(caps_str);
  rtsp_log_stats(src, ctx, "pad-added");
  maybe_start_rtsp_stats_poller(src, ctx);
}

static void rtsp_pad_removed(GstElement* src, GstPad* pad, gpointer /*user_data*/) {
  if (!rtsp_stats_debug_enabled())
    return;
  auto* ctx = static_cast<RtspDebugCtx*>(g_object_get_qdata(G_OBJECT(src), rtsp_debug_quark()));
  if (!ctx)
    return;
  const char* pad_name = gst_pad_get_name(pad);
  std::fprintf(stderr, "[rtsp] pad-removed name=%s pad=%s\n", ctx->element_name.c_str(),
               pad_name ? pad_name : "<null>");
  rtsp_log_stats(src, ctx, "pad-removed");
}

static void attach_rtsp_debug(GstElement* pipeline, const std::vector<std::shared_ptr<Node>>& nodes,
                              const NameTransform& name_transform) {
  if (!pipeline || !rtsp_stats_debug_enabled())
    return;
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto* rtsp = dynamic_cast<const RTSPInput*>(nodes[i].get());
    if (!rtsp)
      continue;
    const auto names =
        apply_name_transform(name_transform, nodes[i]->element_names(static_cast<int>(i)));
    if (names.empty())
      continue;
    GstElement* rtspsrc = gst_bin_get_by_name(GST_BIN(pipeline), names[0].c_str());
    if (!rtspsrc)
      continue;
    RtspDebugCtx* ctx = ensure_rtsp_debug_ctx(rtspsrc, names[0]);
    (void)ctx;
    g_signal_connect(rtspsrc, "pad-added", G_CALLBACK(rtsp_pad_added), nullptr);
    g_signal_connect(rtspsrc, "pad-removed", G_CALLBACK(rtsp_pad_removed), nullptr);
    rtsp_try_connect_signal(rtspsrc, "timeout", {G_TYPE_UINT64}, G_CALLBACK(rtsp_timeout_cb));
    rtsp_try_connect_signal(rtspsrc, "connection-closed", {},
                            G_CALLBACK(rtsp_connection_closed_cb));
    rtsp_log_stats(rtspsrc, ctx, "attach");
    maybe_start_rtsp_stats_poller(rtspsrc, ctx);
    gst_object_unref(rtspsrc);
  }
}

static void attach_h264_caps_fixups(GstElement* pipeline,
                                    const std::vector<std::shared_ptr<Node>>& nodes,
                                    const NameTransform& name_transform) {
  if (!pipeline)
    return;

  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto* fix = dynamic_cast<const H264CapsFixup*>(nodes[i].get());
    if (!fix)
      continue;

    const auto names =
        apply_name_transform(name_transform, nodes[i]->element_names(static_cast<int>(i)));
    if (names.empty())
      continue;

    GstElement* elem = gst_bin_get_by_name(GST_BIN(pipeline), names[0].c_str());
    if (!elem)
      continue;

    GstPad* src = gst_element_get_static_pad(elem, "src");
    if (!src) {
      gst_object_unref(elem);
      continue;
    }

    const std::string rtsp_name = find_rtsp_input_name_for_fixup(nodes, i, name_transform);
    auto* ctx = make_h264_caps_fixup_ctx(pipeline, *fix, names[0], rtsp_name);
    gst_pad_add_probe(
        src,
        static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM |
                                     GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM |
                                     GST_PAD_PROBE_TYPE_BUFFER),
        h264_caps_fixup_cb, ctx,
        +[](gpointer p) { delete reinterpret_cast<H264CapsFixupCtx*>(p); });

    gst_object_unref(src);
    gst_object_unref(elem);

    const std::string upstream_caps = find_h264_capsfilter_name_for_fixup(nodes, i, name_transform);
    if (upstream_caps.empty() || upstream_caps == names[0])
      continue;

    GstElement* upstream_elem = gst_bin_get_by_name(GST_BIN(pipeline), upstream_caps.c_str());
    if (!upstream_elem)
      continue;

    GstPad* upstream_src = gst_element_get_static_pad(upstream_elem, "src");
    if (!upstream_src) {
      gst_object_unref(upstream_elem);
      continue;
    }

    auto* upstream_ctx = make_h264_caps_fixup_ctx(pipeline, *fix, upstream_caps, rtsp_name);
    gst_pad_add_probe(
        upstream_src,
        static_cast<GstPadProbeType>(GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM |
                                     GST_PAD_PROBE_TYPE_QUERY_DOWNSTREAM),
        h264_caps_fixup_cb, upstream_ctx,
        +[](gpointer p) { delete reinterpret_cast<H264CapsFixupCtx*>(p); });

    gst_object_unref(upstream_src);
    gst_object_unref(upstream_elem);
  }
}

// =====================================================================================
// Bus/meta plumbing + improved error diagnostics (parse_error + DOT dumps)
// =====================================================================================

} // namespace

GstElement* session_build_parse_pipeline_or_throw(const std::string& pipeline_string,
                                                  const char* where) {
  return parse_pipeline_or_throw(pipeline_string, where);
}

void session_build_attach_rtsp_debug(GstElement* pipeline,
                                     const std::vector<std::shared_ptr<Node>>& nodes,
                                     const NameTransform& name_transform) {
  attach_rtsp_debug(pipeline, nodes, name_transform);
}

void session_build_attach_h264_caps_fixups(GstElement* pipeline,
                                           const std::vector<std::shared_ptr<Node>>& nodes,
                                           const NameTransform& name_transform) {
  attach_h264_caps_fixups(pipeline, nodes, name_transform);
}

} // namespace simaai::neat
