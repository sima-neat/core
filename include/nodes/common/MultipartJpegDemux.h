/**
 * @file
 * @ingroup nodes_common
 * @brief `MultipartJpegDemux` Node — extracts parts from multipart MJPEG streams.
 */
#pragma once

#include "builder/Node.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace simaai::neat {

/// Maximum number of distinct header names one capture allowlist may select.
inline constexpr std::size_t kMultipartHeaderCaptureMaxHeaders = 64U;
/// Maximum length, in bytes, of a single selected header name.
inline constexpr std::size_t kMultipartHeaderCaptureMaxNameBytes = 128U;
/// Maximum length, in bytes, of one header line inside a multipart part header block.
inline constexpr std::size_t kMultipartHeaderCaptureMaxLineBytes = 8U * 1024U;
/// Maximum total size, in bytes, of one multipart part header block.
inline constexpr std::size_t kMultipartHeaderCaptureMaxBlockBytes = 64U * 1024U;

/**
 * @brief Selects which multipart part headers are captured as per-frame attributes.
 *
 * The list is an allowlist. Leaving it empty disables capture entirely and keeps the
 * stock demux topology and behavior unchanged.
 *
 * Configured names and emitted keys are normalized to ASCII lowercase, so matching is
 * case-insensitive; duplicates collapse after normalization. Within one part, a repeated
 * selected header keeps its last value. A header that is absent from a part is omitted;
 * a header present with an empty value is preserved as an empty string. Only surrounding
 * SP/HTAB is trimmed — values are never otherwise reinterpreted.
 *
 * Configuration is rejected rather than truncated when it exceeds
 * `kMultipartHeaderCaptureMaxHeaders` names or `kMultipartHeaderCaptureMaxNameBytes` per
 * name, or when a name is not a valid HTTP token.
 *
 * @see SampleAttributes
 * @ingroup nodes_common
 */
struct MultipartHeaderCaptureOptions {
  /// Header names to capture. Empty (the default) disables capture.
  std::vector<std::string> headers;

  /// True when capture is requested.
  bool enabled() const noexcept {
    return !headers.empty();
  }
};

/// Construction options for `MultipartJpegDemux`.
struct MultipartJpegDemuxOptions {
  std::string boundary;       ///< Optional multipart boundary override; empty = auto-detect.
  bool single_stream = false; ///< If true, assume the multipart content type is stable.
  /// Optional per-part header capture. Empty (the default) keeps the stock topology.
  MultipartHeaderCaptureOptions header_capture;
};

/**
 * @brief Normalize and validate a header-capture allowlist.
 *
 * Lowercases each configured name, trims surrounding SP/HTAB, removes duplicates, and
 * returns the result in a stable sorted order.
 *
 * @param opt Requested capture configuration.
 * @return Normalized, deduplicated header names (empty when capture is disabled).
 * @throws std::invalid_argument if a name is not a valid HTTP token, is empty, exceeds
 *         `kMultipartHeaderCaptureMaxNameBytes`, or if more than
 *         `kMultipartHeaderCaptureMaxHeaders` distinct names remain after normalization.
 * @ingroup nodes_common
 */
std::vector<std::string>
normalize_multipart_header_capture(const MultipartHeaderCaptureOptions& opt);

/**
 * @brief Demux multipart HTTP streams into per-part buffers.
 *
 * With header capture disabled, this wraps GStreamer's `multipartdemux`; place
 * `JpegParse` after it to normalize each MJPEG part. With header capture enabled,
 * this node already emits parsed `image/jpeg` frames and must connect directly to
 * the decoder because `JpegParse` does not preserve the captured attributes.
 *
 * @ingroup nodes_common
 */
class MultipartJpegDemux final : public Node {
public:
  explicit MultipartJpegDemux(MultipartJpegDemuxOptions opt = {});

  std::string kind() const override {
    return "MultipartJpegDemux";
  }
  NodeCapsBehavior caps_behavior() const override {
    return NodeCapsBehavior::Dynamic;
  }

  std::string backend_fragment(int node_index) const override;
  std::vector<std::string> element_names(int node_index) const override;

  const MultipartJpegDemuxOptions& options() const {
    return opt_;
  }

  /// Normalized, deduplicated capture allowlist; empty when capture is disabled.
  const std::vector<std::string>& capture_headers() const {
    return capture_;
  }

private:
  MultipartJpegDemuxOptions opt_;
  std::vector<std::string> capture_;
};

} // namespace simaai::neat

namespace simaai::neat::nodes {
/// Convenience factory for a `MultipartJpegDemux` Node.
std::shared_ptr<simaai::neat::Node> MultipartJpegDemux(MultipartJpegDemuxOptions opt = {});
} // namespace simaai::neat::nodes
