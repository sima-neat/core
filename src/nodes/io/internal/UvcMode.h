#pragma once

#include "nodes/sima/ImageTransform.h"
#include <gst/video/video.h>
#include <linux/videodev2.h>
#include <cstdint>
#include <string>

namespace simaai::neat::internal::uvc {
// V4L2 intervals are reciprocal frame rates. Widen before multiplication: a
// step test can contain three independent 32-bit factors. No floating-point
// rounding or integer-fps approximation participates in mode admission.
inline bool interval_contains(const v4l2_frmivalenum& interval, std::uint32_t fps_num,
                              std::uint32_t fps_den) {
  if (!fps_num || !fps_den)
    return false;
  const auto valid = [](v4l2_fract f) { return f.numerator && f.denominator; };
  const auto equal = [](v4l2_fract a, v4l2_fract b) {
    return std::uint64_t(a.numerator) * b.denominator == std::uint64_t(b.numerator) * a.denominator;
  };
  const v4l2_fract requested{fps_den, fps_num};
  if (interval.type == V4L2_FRMIVAL_TYPE_DISCRETE)
    return valid(interval.discrete) && equal(requested, interval.discrete);
  if (interval.type != V4L2_FRMIVAL_TYPE_STEPWISE && interval.type != V4L2_FRMIVAL_TYPE_CONTINUOUS)
    return false;
  const auto& s = interval.stepwise;
  if (!valid(s.min) || !valid(s.max) ||
      std::uint64_t(fps_den) * s.min.denominator < std::uint64_t(fps_num) * s.min.numerator ||
      std::uint64_t(fps_den) * s.max.denominator > std::uint64_t(fps_num) * s.max.numerator)
    return false;
  if (interval.type == V4L2_FRMIVAL_TYPE_CONTINUOUS)
    return true;
  if (!valid(s.step))
    return false;
  const __uint128_t delta =
      __uint128_t(fps_den) * s.min.denominator - __uint128_t(fps_num) * s.min.numerator;
  return (delta * s.step.denominator) %
             (__uint128_t(fps_num) * s.min.denominator * s.step.numerator) ==
         0;
}

inline bool yuyv_layout_valid(const v4l2_pix_format& pix) {
  const std::uint64_t active = std::uint64_t(pix.width) * 2;
  return pix.bytesperline >= active &&
         pix.sizeimage >= std::uint64_t(pix.height - 1) * pix.bytesperline + active;
}

struct YuyvColor {
  ImageColorOptions color;
  std::string colorimetry;
  const char* error = nullptr; ///< Why the format's color metadata is unsupported.
};

// V4L2 defaults describe matrix/range; never equate DEFAULT to full-range 601.
inline YuyvColor yuyv_color(const v4l2_pix_format& pix) {
  YuyvColor out;
  const auto colorspace = pix.colorspace == V4L2_COLORSPACE_DEFAULT
                              ? V4L2_MAP_COLORSPACE_DEFAULT(pix.width <= 720 && pix.height <= 576,
                                                            pix.width <= 1920 && pix.height <= 1080)
                              : pix.colorspace;
  const auto encoding = pix.ycbcr_enc == V4L2_YCBCR_ENC_DEFAULT
                            ? V4L2_MAP_YCBCR_ENC_DEFAULT(colorspace)
                            : pix.ycbcr_enc;
  const auto range = pix.quantization == V4L2_QUANTIZATION_DEFAULT
                         ? V4L2_MAP_QUANTIZATION_DEFAULT(false, colorspace, encoding)
                         : pix.quantization;
  const auto transfer = pix.xfer_func == V4L2_XFER_FUNC_DEFAULT
                            ? V4L2_MAP_XFER_FUNC_DEFAULT(colorspace)
                            : pix.xfer_func;
  if ((encoding != V4L2_YCBCR_ENC_601 && encoding != V4L2_YCBCR_ENC_709) ||
      (range != V4L2_QUANTIZATION_FULL_RANGE && range != V4L2_QUANTIZATION_LIM_RANGE) ||
      (transfer != V4L2_XFER_FUNC_709 && transfer != V4L2_XFER_FUNC_SRGB)) {
    out.error = "has unsupported YUYV matrix/range/transfer metadata";
    return out;
  }
  GstVideoColorimetry color{};
  color.matrix =
      encoding == V4L2_YCBCR_ENC_601 ? GST_VIDEO_COLOR_MATRIX_BT601 : GST_VIDEO_COLOR_MATRIX_BT709;
  color.range = range == V4L2_QUANTIZATION_FULL_RANGE ? GST_VIDEO_COLOR_RANGE_0_255
                                                      : GST_VIDEO_COLOR_RANGE_16_235;
  color.transfer =
      transfer == V4L2_XFER_FUNC_709 ? GST_VIDEO_TRANSFER_BT709 : GST_VIDEO_TRANSFER_SRGB;
  switch (colorspace) {
  case V4L2_COLORSPACE_SMPTE170M:
    color.primaries = GST_VIDEO_COLOR_PRIMARIES_SMPTE170M;
    break;
  case V4L2_COLORSPACE_REC709:
  case V4L2_COLORSPACE_SRGB:
  case V4L2_COLORSPACE_JPEG:
    color.primaries = GST_VIDEO_COLOR_PRIMARIES_BT709;
    break;
  default:
    out.error = "has unsupported YUYV color primaries";
    return out;
  }
  gchar* text = gst_video_colorimetry_to_string(&color);
  if (!text) {
    out.error = "cannot describe YUYV color metadata";
    return out;
  }
  out.colorimetry = text;
  g_free(text);
  out.color.matrix =
      encoding == V4L2_YCBCR_ENC_601 ? ImageColorMatrix::Bt601 : ImageColorMatrix::Bt709;
  out.color.range =
      range == V4L2_QUANTIZATION_FULL_RANGE ? ImageYuvRange::Full : ImageYuvRange::Limited;
  return out;
}
} // namespace simaai::neat::internal::uvc
