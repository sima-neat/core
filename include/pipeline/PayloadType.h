/**
 * @file
 * @ingroup pipeline
 * @brief Public payload-family enum shared by inputs, samples, and contracts.
 */
#pragma once

#include <string>
#include <string_view>

namespace simaai::neat {

/**
 * @brief Semantic family of a payload flowing through a Graph.
 *
 * `PayloadType` is intentionally neutral: the same enum describes an input option,
 * a pushed/pulled `Sample`, and a boundary contract. GStreamer media strings are an
 * internal transport detail derived from this enum at the adapter boundary.
 */
enum class PayloadType {
  Auto = 0, ///< Infer from tensor/sample metadata when possible.
  Image,    ///< Decoded image pixels (`video/x-raw` internally).
  Tensor,   ///< Tensor payload (`application/vnd.simaai.tensor` internally).
  Encoded,  ///< Encoded video/byte-stream payloads such as H.264, H.265, or JPEG.
};

/// Compatibility alias for code written during the InputOptions migration.
using InputType [[deprecated("use PayloadType")]] = PayloadType;

/// Convert an internal/legacy caps media string to the public payload family.
inline PayloadType payload_type_from_media_type(std::string_view media_type) {
  if (media_type == "video/x-raw") {
    return PayloadType::Image;
  }
  if (media_type == "application/vnd.simaai.tensor") {
    return PayloadType::Tensor;
  }
  if (media_type == "video/x-h264" || media_type == "video/x-h265" || media_type == "image/jpeg") {
    return PayloadType::Encoded;
  }
  return PayloadType::Auto;
}

/// Legacy helper name kept as a source-compatible bridge.
inline PayloadType input_type_from_media_type(std::string_view media_type) {
  return payload_type_from_media_type(media_type);
}

/// Convert a payload family to the canonical internal media type when one exists.
inline std::string media_type_from_payload_type(PayloadType payload_type) {
  switch (payload_type) {
  case PayloadType::Image:
    return "video/x-raw";
  case PayloadType::Tensor:
    return "application/vnd.simaai.tensor";
  case PayloadType::Encoded:
    return "video/x-h264";
  case PayloadType::Auto:
  default:
    return "";
  }
}

} // namespace simaai::neat
