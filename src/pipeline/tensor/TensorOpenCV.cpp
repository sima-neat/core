// TensorOpenCV.cpp
//
// OpenCV adapter helpers for Tensor.
//
// This file provides:
//  - from_cv_mat(): wrap a cv::Mat as a Tensor (zero-copy wrapper via CpuExternal storage)
//  - map_cv_mat_view(): expose a dense UInt8 image tensor as a cv::Mat view (zero-copy mapping)
//  - to_cv_mat_copy(): convenience conversion/copy to cv::Mat (handles NV12/I420 -> BGR)
//
// Key semantics / invariants (confirmed by TensorCore.h + repo call sites):
//  - Tensor::map_read() returns a mapping whose `data` pointer already includes
//  Tensor::byte_offset.
//  - read-only enforcement happens at Tensor::map() level using Tensor::read_only.
//    make_cpu_external_storage() does NOT enforce read_only in its map_fn; it is an intent flag
//    only.
//  - For dense tensors, strides_bytes are byte-strides per dimension. For HWC uint8 images,
//    typical strides are {row_step_bytes, pixel_bytes, channel_bytes}.
//
// TODO(repo-policy): If you want BGR<->RGB or BGR/RGB<->GRAY8 conversions for dense tensors,
// add explicit cv::cvtColor conversion paths in to_cv_mat_copy() (today we only support:
//   - direct view when formats match
//   - NV12/I420 -> BGR conversion
//   - otherwise cpu().contiguous() + direct view when formats match)

#include "pipeline/TensorOpenCV.h"
#include "pipeline/internal/TensorMath.h"

#if defined(SIMA_WITH_OPENCV)
#include <opencv2/imgproc.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat {

using simaai::neat::pipeline_internal::safe_add;
using simaai::neat::pipeline_internal::safe_mul;

namespace {

//==============================================================================
// Small helpers
//==============================================================================

struct CvImageTraits {
  int channels = -1;
  simaai::neat::TensorLayout layout = simaai::neat::TensorLayout::Unknown;
  int cv_type = -1;
};

std::optional<CvImageTraits> traits_for_format(ImageSpec::PixelFormat fmt) {
  CvImageTraits t;
  switch (fmt) {
  case ImageSpec::PixelFormat::GRAY8:
    t.channels = 1;
    t.layout = simaai::neat::TensorLayout::HW;
    t.cv_type = CV_8UC1;
    return t;
  case ImageSpec::PixelFormat::RGB:
  case ImageSpec::PixelFormat::BGR:
    t.channels = 3;
    t.layout = simaai::neat::TensorLayout::HWC;
    t.cv_type = CV_8UC3;
    return t;
  default:
    break;
  }
  return std::nullopt;
}

/**
 * Validate that the provided cv::Mat is a supported input image for wrapping.
 * Requirements:
 *  - non-empty
 *  - CV_8U depth
 *  - channel count matches the requested pixel format
 */
void validate_cv_mat_or_throw(const cv::Mat& mat, ImageSpec::PixelFormat fmt) {
  if (mat.empty() || mat.data == nullptr) {
    throw std::runtime_error("from_cv_mat: empty cv::Mat");
  }

  const auto traits = traits_for_format(fmt);
  if (!traits.has_value()) {
    throw std::runtime_error("from_cv_mat: unsupported pixel format");
  }

  if (mat.depth() != CV_8U) {
    throw std::runtime_error("from_cv_mat: only CV_8U is supported");
  }

  if (mat.channels() != traits->channels) {
    throw std::runtime_error("from_cv_mat: channel count mismatch");
  }

  // Optional stricter check: ensure actual cv::Mat type matches expected.
  // (Depth + channels already implies CV_8UC1 / CV_8UC3.)
  if (traits->cv_type > 0 && mat.type() != traits->cv_type) {
    throw std::runtime_error("from_cv_mat: cv::Mat type mismatch");
  }
}

/**
 * Compute tight required byte-span for an image with row-step:
 *   end = step*(h-1) + row_bytes
 *
 * Returns 0 on overflow or invalid geometry.
 */
std::size_t required_span_bytes_checked(std::size_t step_bytes, std::size_t row_bytes,
                                        std::size_t rows) {
  if (rows == 0)
    return 0;
  if (row_bytes == 0)
    return 0;
  if (step_bytes == 0)
    return 0;

  std::size_t term = 0;
  if (!safe_mul((rows - 1), step_bytes, &term))
    return 0;

  std::size_t end = 0;
  if (!safe_add(term, row_bytes, &end))
    return 0;
  return end;
}

} // namespace

//==============================================================================
// Tensor <-> OpenCV adapters
//==============================================================================

Tensor Tensor::from_cv_mat(const cv::Mat& mat, ImageSpec::PixelFormat fmt, bool read_only) {
  return simaai::neat::from_cv_mat(mat, fmt, read_only);
}

/**
 * Wrap a cv::Mat as a Tensor without copying.
 *
 * Storage:
 *  - CpuExternal, holding a shared_ptr<cv::Mat> to keep the underlying buffer alive.
 *
 * Strides:
 *  - For GRAY8 (HW): strides_bytes = {row_step_bytes, elem_bytes}
 *  - For RGB/BGR (HWC): strides_bytes = {row_step_bytes, pixel_bytes, channel_bytes}
 *
 * NOTE:
 *  - This wrapper preserves cv::Mat padding via row stride (step[0]).
 *  - read_only is enforced by Tensor::map() (throws on Write/ReadWrite if read_only=true).
 */
Tensor from_cv_mat(const cv::Mat& mat, ImageSpec::PixelFormat fmt, bool read_only) {
  validate_cv_mat_or_throw(mat, fmt);

  // Keep the Mat header (and its refcounted backing store) alive.
  auto holder = std::make_shared<cv::Mat>(mat);

  const std::size_t rows = static_cast<std::size_t>(holder->rows);
  const std::size_t step0 = static_cast<std::size_t>(holder->step[0]);

  // IMPORTANT: cv::Mat may be an ROI (submatrix). In that case,
  // holder->data points to the ROI origin, but holder->step[0] still refers
  // to the parent row stride. Using step*rows can overrun the parent buffer
  // when the ROI starts at x>0. Use the tight span: step*(rows-1) + row_bytes.
  const std::size_t row_bytes =
      static_cast<std::size_t>(holder->cols) * static_cast<std::size_t>(holder->elemSize());
  const std::size_t bytes = required_span_bytes_checked(step0, row_bytes, rows);
  if (bytes == 0) {
    throw std::runtime_error("from_cv_mat: invalid byte span");
  }

  Tensor out;
  out.storage = make_cpu_external_storage(holder->data, bytes, holder, read_only);
  out.dtype = simaai::neat::TensorDType::UInt8;
  out.device = {DeviceType::CPU, 0};
  out.byte_offset = 0;
  out.read_only = read_only;

  const int h = holder->rows;
  const int w = holder->cols;
  const int c = holder->channels();
  const int64_t row_step = static_cast<int64_t>(holder->step[0]);

  // For CV_8U, elemSize1() == 1 byte per channel.
  const int64_t channel_bytes = static_cast<int64_t>(holder->elemSize1());
  const int64_t pixel_bytes = static_cast<int64_t>(holder->elemSize()); // channels * elemSize1()

  if (fmt == ImageSpec::PixelFormat::GRAY8) {
    out.shape = {h, w};
    out.strides_bytes = {row_step, channel_bytes}; // channel_bytes == 1 for CV_8U
    out.layout = simaai::neat::TensorLayout::HW;
  } else {
    out.shape = {h, w, c};
    out.strides_bytes = {row_step, pixel_bytes, channel_bytes};
    out.layout = simaai::neat::TensorLayout::HWC;
  }

  ImageSpec image;
  image.format = fmt;
  // TODO(optional): set image.color_space if you have a canonical string for BGR/RGB/GRAY8.
  out.semantic.image = image;

  return out;
}

/**
 * Map a dense UInt8 image tensor as a cv::Mat view (no copy).
 *
 * This only succeeds when:
 *  - semantic.image.format == desired
 *  - dtype == UInt8
 *  - is_dense()
 *  - layout matches desired (HW for GRAY8, HWC for RGB/BGR)
 *
 * The returned CvMatView owns a Mapping keepalive/unmap to ensure the
 * backing memory remains valid while the returned cv::Mat exists.
 */
std::optional<CvMatView> Tensor::map_cv_mat_view(ImageSpec::PixelFormat desired) const {
  if (!semantic.image.has_value())
    return std::nullopt;
  if (semantic.image->format != desired)
    return std::nullopt;
  if (!is_dense())
    return std::nullopt;
  if (dtype != simaai::neat::TensorDType::UInt8)
    return std::nullopt;

  const auto traits = traits_for_format(desired);
  if (!traits.has_value())
    return std::nullopt;

  if (layout != traits->layout)
    return std::nullopt;

  const int h = height();
  const int w = width();
  if (h <= 0 || w <= 0)
    return std::nullopt;

  // Shape sanity checks.
  if (desired != ImageSpec::PixelFormat::GRAY8) {
    if (shape.size() < 3 || shape[2] != traits->channels)
      return std::nullopt;
  } else {
    if (shape.size() < 2)
      return std::nullopt;
  }

  Mapping mapping = map_read();
  if (!mapping.data)
    return std::nullopt;

  // Determine row step in bytes.
  std::size_t step = 0;
  if (!strides_bytes.empty()) {
    const int64_t s0 = strides_bytes[0];
    if (s0 <= 0)
      return std::nullopt;
    step = static_cast<std::size_t>(s0);
  } else {
    // Fallback: assume tightly packed rows.
    // dtype is UInt8 => 1 byte/channel.
    const std::size_t row_bytes =
        static_cast<std::size_t>(w) * static_cast<std::size_t>(traits->channels);
    step = row_bytes;
  }

  // Validate the stride is large enough for a row.
  const std::size_t row_bytes =
      static_cast<std::size_t>(w) * static_cast<std::size_t>(traits->channels);
  if (step < row_bytes)
    return std::nullopt;

  // Validate mapping bounds (when mapping.size_bytes is known).
  if (mapping.size_bytes > 0) {
    const std::size_t required =
        required_span_bytes_checked(step, row_bytes, static_cast<std::size_t>(h));
    if (required == 0 || required > mapping.size_bytes)
      return std::nullopt;
  }

  if (traits->cv_type <= 0)
    return std::nullopt;

  cv::Mat mat(h, w, traits->cv_type, mapping.data, step);

  CvMatView view;
  view.mapping = std::move(mapping);
  view.mat = std::move(mat);
  return view;
}

/**
 * Convert tensor into an owning cv::Mat.
 *
 * Supported desired formats:
 *  - BGR, RGB, GRAY8
 *
 * Conversion rules:
 *  - If the tensor is already a matching dense UInt8 image, returns a clone of a mapped view.
 *  - If desired is BGR and the tensor is NV12/I420, packs YUV and uses cv::cvtColor to BGR.
 *  - Otherwise falls back to cpu().contiguous() and requires an exact format match.
 *
 * TODO(repo-policy): If you want RGB<->BGR or GRAY conversions for dense tensors,
 * implement cv::cvtColor conversion paths here.
 */
cv::Mat Tensor::to_cv_mat_copy(ImageSpec::PixelFormat desired) const {
  if (desired != ImageSpec::PixelFormat::BGR && desired != ImageSpec::PixelFormat::RGB &&
      desired != ImageSpec::PixelFormat::GRAY8) {
    throw std::runtime_error("to_cv_mat_copy: unsupported desired format");
  }

  // Fast path: direct view -> clone.
  if (auto view = map_cv_mat_view(desired); view.has_value()) {
    return view->mat.clone();
  }

  // Special-case: NV12/I420 -> BGR.
  if (desired == ImageSpec::PixelFormat::BGR && (is_nv12() || is_i420())) {
    const int w = width();
    const int h = height();
    if (w <= 0 || h <= 0) {
      throw std::runtime_error("to_cv_mat_copy: invalid NV12/I420 dimensions");
    }

    std::vector<uint8_t> yuv = is_nv12() ? copy_nv12_contiguous() : copy_i420_contiguous();

    // OpenCV expects a single-plane Mat of size (h + h/2) x w for these formats.
    cv::Mat yuv_mat(h + h / 2, w, CV_8UC1, yuv.data());

    cv::Mat bgr;
    const int code = is_nv12() ? cv::COLOR_YUV2BGR_NV12 : cv::COLOR_YUV2BGR_I420;
    cv::cvtColor(yuv_mat, bgr, code);
    return bgr; // owning output
  }

  // Fallback: bring to CPU + make dense contiguous and attempt direct view.
  // NOTE: This still requires semantic.image.format == desired.
  Tensor cpu_tensor = cpu().contiguous();
  auto cpu_view = cpu_tensor.map_cv_mat_view(desired);
  if (!cpu_view.has_value()) {
    throw std::runtime_error("to_cv_mat_copy: tensor layout/format mismatch");
  }
  return cpu_view->mat.clone();
}

} // namespace simaai::neat
#endif // SIMA_WITH_OPENCV
