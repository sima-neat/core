/**
 * @file
 * @ingroup tensors
 * @brief OpenCV helpers for Tensor.
 *
 * Provides a `cv::Mat` to `Tensor` adapter, available only when the framework
 * is built with OpenCV support (`SIMA_WITH_OPENCV`). Returned tensors can
 * either reference the Mat's data (zero-copy, read-only) or take ownership of
 * the bytes; see the function-level docs for which mode is selected.
 *
 * @see TensorAdapters.h for the GStreamer-side adapter.
 */
#pragma once

#include "pipeline/TensorCore.h"

#if defined(SIMA_WITH_OPENCV)
namespace simaai::neat {

/**
 * @brief Wrap or copy a `cv::Mat` as a `Tensor`.
 *
 * @param mat       Source matrix (continuous data layout assumed; otherwise copied).
 * @param fmt       Pixel format describing @p mat's channel order and semantic.
 * @param read_only If true, the returned tensor references @p mat's data (caller
 *                  must keep @p mat alive); if false, the tensor owns a copy.
 * @return Tensor with shape and dtype derived from @p mat.
 * @deprecated Use the TensorMemory overload. The default from_cv_mat(mat, fmt)
 *             now creates an EV74-placed tensor; pass TensorMemory::CPU/A65
 *             explicitly when CPU placement is intended.
 */
[[deprecated("Use from_cv_mat(mat, fmt, TensorMemory::EV74/CPU/MLA); "
             "from_cv_mat(mat, fmt) defaults to EV74 placement.")]] Tensor
from_cv_mat(const cv::Mat& mat, ImageSpec::PixelFormat fmt, bool read_only);

/**
 * @brief Wrap a `cv::Mat` as a zero-copy CPU tensor view.
 *
 * The returned tensor references the Mat buffer and preserves ROI row stride.
 * Keep the source Mat/storage alive through the returned tensor lifetime. Use
 * the TensorMemory overload of `from_cv_mat` when an owned CPU copy or device
 * placement is intended.
 */
Tensor from_cv_mat_view(const cv::Mat& mat,
                        ImageSpec::PixelFormat fmt = ImageSpec::PixelFormat::BGR,
                        bool read_only = true);

Tensor from_cv_mat(const cv::Mat& mat, ImageSpec::PixelFormat fmt = ImageSpec::PixelFormat::BGR,
                   TensorMemory memory = TensorMemory::EV74);
Tensor from_cv_mat(const cv::Mat& mat, TensorMemory memory);

} // namespace simaai::neat
#endif
