// src/pipeline/internal/CapsBridge.cpp
#include "pipeline/internal/CapsBridge.h"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <algorithm>
#include <cstring>
#include <cctype>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace simaai::neat::pipeline_internal {
namespace {

simaai::neat::ImageSpec::PixelFormat pixel_format_from_gst(GstVideoFormat fmt) {
  switch (fmt) {
  case GST_VIDEO_FORMAT_RGB:
    return simaai::neat::ImageSpec::PixelFormat::RGB;
  case GST_VIDEO_FORMAT_BGR:
    return simaai::neat::ImageSpec::PixelFormat::BGR;
  case GST_VIDEO_FORMAT_GRAY8:
    return simaai::neat::ImageSpec::PixelFormat::GRAY8;
  case GST_VIDEO_FORMAT_NV12:
    return simaai::neat::ImageSpec::PixelFormat::NV12;
  case GST_VIDEO_FORMAT_I420:
    return simaai::neat::ImageSpec::PixelFormat::I420;
  default:
    return simaai::neat::ImageSpec::PixelFormat::UNKNOWN;
  }
}

TensorDType dtype_from_tensor_format(const std::string& fmt) {
  std::string s;
  s.reserve(fmt.size());
  for (char c : fmt) {
    s.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  if (s == "DETESS")
    return TensorDType::UInt16;
  if (s == "DETESSDEQUANT" || s == "FP32")
    return TensorDType::Float32;
  if (s == "EVXX_INT8" || s == "EV74_INT8" || s == "INT8")
    return TensorDType::Int8;
  if (s == "EVXX_BFLOAT16" || s == "BF16" || s == "BFLOAT16")
    return TensorDType::BFloat16;
  if (s == "UINT8")
    return TensorDType::UInt8;
  return TensorDType::UInt8;
}

bool find_indexed_int_field(const GstStructure* st, const char* base_name, int* out) {
  if (!st || !base_name || !out)
    return false;
  if (gst_structure_get_int(st, base_name, out))
    return true;

  const std::string prefix = std::string(base_name) + "__";
  int best_idx = std::numeric_limits<int>::max();
  int best_val = 0;
  bool found = false;

  const int n = gst_structure_n_fields(st);
  for (int i = 0; i < n; ++i) {
    const char* name = gst_structure_nth_field_name(st, i);
    if (!name)
      continue;
    const std::string_view field(name);
    if (field.rfind(prefix, 0) != 0)
      continue;
    const std::string_view suffix = field.substr(prefix.size());
    bool numeric = !suffix.empty();
    for (char c : suffix) {
      if (!std::isdigit(static_cast<unsigned char>(c))) {
        numeric = false;
        break;
      }
    }
    if (!numeric)
      continue;
    int value = 0;
    if (!gst_structure_get_int(st, name, &value))
      continue;
    const int idx = std::stoi(std::string(suffix));
    if (idx < best_idx) {
      best_idx = idx;
      best_val = value;
      found = true;
    }
  }

  if (found)
    *out = best_val;
  return found;
}

bool parse_shape_csv(const std::string& raw, std::vector<int64_t>* out) {
  if (!out)
    return false;
  out->clear();
  std::stringstream ss(raw);
  std::string token;
  while (std::getline(ss, token, ',')) {
    token.erase(std::remove_if(token.begin(), token.end(),
                               [](unsigned char c) { return std::isspace(c) != 0; }),
                token.end());
    if (token.empty())
      return false;
    try {
      const long long value = std::stoll(token);
      if (value <= 0)
        return false;
      out->push_back(static_cast<int64_t>(value));
    } catch (...) {
      return false;
    }
  }
  return !out->empty();
}

bool find_tensor_shape(const GstStructure* st, std::vector<int64_t>* out) {
  if (!st || !out)
    return false;

  gint rank_i = 0;
  if (gst_structure_get_int(st, "rank", &rank_i) && rank_i > 0) {
    const guint rank = static_cast<guint>(rank_i);
    if (const char* shape_csv = gst_structure_get_string(st, "shape")) {
      std::vector<int64_t> parsed;
      if (parse_shape_csv(shape_csv, &parsed) && parsed.size() == static_cast<std::size_t>(rank)) {
        *out = std::move(parsed);
        return true;
      }
      return false;
    }

    out->clear();
    out->reserve(rank);
    for (guint i = 0; i < rank; ++i) {
      const std::string key = "dim" + std::to_string(i);
      gint dim_i = 0;
      if (!gst_structure_get_int(st, key.c_str(), &dim_i) || dim_i <= 0) {
        out->clear();
        return false;
      }
      out->push_back(static_cast<int64_t>(dim_i));
    }
    return true;
  }

  int w = 0;
  int h = 0;
  int d = 0;
  find_indexed_int_field(st, "width", &w);
  find_indexed_int_field(st, "height", &h);
  find_indexed_int_field(st, "depth", &d);
  if (w > 0 && h > 0 && d > 0) {
    *out = {h, w, d};
    return true;
  }
  if (w > 0 && h > 0) {
    *out = {h, w};
    return true;
  }
  if (w > 0) {
    *out = {w};
    return true;
  }
  return false;
}

const char* dtype_name(TensorDType dtype) {
  switch (dtype) {
  case TensorDType::UInt8:
    return "UInt8";
  case TensorDType::Int8:
    return "Int8";
  case TensorDType::UInt16:
    return "UInt16";
  case TensorDType::Int16:
    return "Int16";
  case TensorDType::Int32:
    return "Int32";
  case TensorDType::BFloat16:
    return "BFloat16";
  case TensorDType::Float32:
    return "Float32";
  case TensorDType::Float64:
    return "Float64";
  }
  return "Unknown";
}

const char* image_format_name(simaai::neat::ImageSpec::PixelFormat fmt) {
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
    return "UNKNOWN";
  }
  return "UNKNOWN";
}

} // namespace

simaai::neat::TensorConstraint tensor_constraint_from_caps(GstCaps* caps) {
  simaai::neat::TensorConstraint out;
  if (!caps)
    return out;

  const GstStructure* st = gst_caps_get_structure(caps, 0);
  const char* media = st ? gst_structure_get_name(st) : nullptr;
  if (!media)
    return out;

  if (std::string(media) == "application/vnd.simaai.tensor") {
    if (const char* dt = gst_structure_get_string(st, "dtype")) {
      out.dtypes.push_back(dtype_from_tensor_format(dt));
    } else {
      const char* fmt = gst_structure_get_string(st, "format");
      const std::string fmt_str = fmt ? fmt : "";
      out.dtypes.push_back(dtype_from_tensor_format(fmt_str));
    }

    if (find_tensor_shape(st, &out.shape)) {
      out.rank = static_cast<int>(out.shape.size());
    }
    return out;
  }

  if (std::string(media).rfind("video/x-raw", 0) == 0) {
    GstVideoInfo info;
    std::memset(&info, 0, sizeof(info));
    if (!gst_video_info_from_caps(&info, caps)) {
      return out;
    }

    const int w = GST_VIDEO_INFO_WIDTH(&info);
    const int h = GST_VIDEO_INFO_HEIGHT(&info);
    const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&info);
    const simaai::neat::ImageSpec::PixelFormat pixel = pixel_format_from_gst(fmt);
    out.dtypes.push_back(TensorDType::UInt8);
    if (pixel != simaai::neat::ImageSpec::PixelFormat::UNKNOWN) {
      out.image_format = pixel;
    }

    if (pixel == simaai::neat::ImageSpec::PixelFormat::NV12 ||
        pixel == simaai::neat::ImageSpec::PixelFormat::I420) {
      out.rank = 2;
      out.shape = {h, w};
      out.allow_composite = true;
      return out;
    }

    if (pixel == simaai::neat::ImageSpec::PixelFormat::GRAY8) {
      out.rank = 3;
      out.shape = {h, w, 1};
      out.allow_composite = false;
      return out;
    }

    if (pixel == simaai::neat::ImageSpec::PixelFormat::RGB ||
        pixel == simaai::neat::ImageSpec::PixelFormat::BGR) {
      out.rank = 3;
      out.shape = {h, w, 3};
      out.allow_composite = false;
      return out;
    }

    out.rank = (w > 0 && h > 0) ? 2 : -1;
    if (w > 0 && h > 0)
      out.shape = {h, w};
    return out;
  }

  return out;
}

std::string tensor_constraint_debug_string(const simaai::neat::TensorConstraint& constraint) {
  std::ostringstream ss;
  ss << "{rank=" << constraint.rank;
  if (!constraint.shape.empty()) {
    ss << " shape=";
    for (size_t i = 0; i < constraint.shape.size(); ++i) {
      if (i)
        ss << "x";
      ss << constraint.shape[i];
    }
  }
  if (!constraint.dtypes.empty()) {
    ss << " dtypes=[";
    for (size_t i = 0; i < constraint.dtypes.size(); ++i) {
      if (i)
        ss << ",";
      ss << dtype_name(constraint.dtypes[i]);
    }
    ss << "]";
  }
  if (constraint.image_format.has_value()) {
    ss << " image=" << image_format_name(*constraint.image_format);
  }
  ss << " composite=" << (constraint.allow_composite ? "true" : "false");
  ss << "}";
  return ss.str();
}

} // namespace simaai::neat::pipeline_internal
