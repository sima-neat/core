/**
 * @file
 * @ingroup pipeline
 * @brief Typed media/pixel/tensor format tags for public options.
 *
 * `FormatTag` is the framework-wide enum identifying every payload format the
 * pipeline understands: raw video (RGB/BGR/GRAY8/NV12/I420/YUYV), encoded
 * video (H.264, generic ENCODED), tensor element types (FP32/INT8/UINT8/BF16
 * and their EVXX_ aliases), opaque byte-stream payloads, and a few
 * pipeline-internal payload kinds (MLA, BBOX, ARGMAX, DETESSDEQUANT).
 * `FormatSpec` is a tiny wrapper that converts
 * implicitly to/from string forms used in caps strings and config files.
 *
 * @see Tensor.h for the underlying `Sample`/`Tensor` types.
 * @see Run.h for how format tags appear on `RunOptions` and stream caps.
 */
#pragma once

#include <algorithm>
#include <cctype>
#include <ostream>
#include <string>

namespace simaai::neat {

/**
 * @brief Identifies a media or tensor payload format.
 *
 * Used in caps strings and option fields throughout the pipeline. `Auto` is
 * the unset sentinel (let the framework pick or sniff). The EVXX_ variants
 * are aliases preferred by the EV74 caps surface.
 *
 * @ingroup pipeline
 * @see FormatSpec
 */
enum class FormatTag {
  Auto = 0,      ///< Unset / framework decides.
  RGB,           ///< Packed RGB, 8 bits per channel.
  BGR,           ///< Packed BGR, 8 bits per channel (OpenCV default).
  GRAY8,         ///< Single-plane 8-bit grayscale.
  NV12,          ///< YUV 4:2:0, Y plane + interleaved UV plane.
  I420,          ///< YUV 4:2:0, three planes (Y, U, V).
  YUYV,          ///< YUV 4:2:2 packed (Y0 U Y1 V).
  ENCODED,       ///< Generic encoded payload (codec from caps).
  H264,          ///< H.264 access unit / NAL stream.
  ByteStream,    ///< Opaque byte stream; downstream interprets bytes by contract.
  MLA,           ///< MLA-tessellated tensor payload.
  BBOX,          ///< Decoded bounding-box byte stream.
  ARGMAX,        ///< Argmax/segmentation map.
  DETESSDEQUANT, ///< Detessellated + dequantized tensor payload.
  FP32,          ///< IEEE-754 32-bit float tensor.
  INT8,          ///< Signed 8-bit integer tensor.
  UINT8,         ///< Unsigned 8-bit integer tensor.
  BF16,          ///< bfloat16 tensor.
  EVXX_FLOAT32,  ///< EV-side alias for FP32.
  EVXX_INT8,     ///< EV-side alias for INT8.
  EVXX_BFLOAT16, ///< EV-side alias for BF16.
};

/// @brief Stable, canonical string token for @p tag (empty string for `Auto`).
inline const char* format_tag_name(FormatTag tag) {
  switch (tag) {
  case FormatTag::Auto:
    return "";
  case FormatTag::RGB:
    return "RGB";
  case FormatTag::BGR:
    return "BGR";
  case FormatTag::GRAY8:
    return "GRAY8";
  case FormatTag::NV12:
    return "NV12";
  case FormatTag::I420:
    return "I420";
  case FormatTag::YUYV:
    return "YUYV";
  case FormatTag::ENCODED:
    return "ENCODED";
  case FormatTag::H264:
    return "H264";
  case FormatTag::ByteStream:
    return "BYTESTREAM";
  case FormatTag::MLA:
    return "MLA";
  case FormatTag::BBOX:
    return "BBOX";
  case FormatTag::ARGMAX:
    return "ARGMAX";
  case FormatTag::DETESSDEQUANT:
    return "DETESSDEQUANT";
  case FormatTag::FP32:
    return "FP32";
  case FormatTag::INT8:
    return "INT8";
  case FormatTag::UINT8:
    return "UINT8";
  case FormatTag::BF16:
    return "BF16";
  case FormatTag::EVXX_FLOAT32:
    return "EVXX_FLOAT32";
  case FormatTag::EVXX_INT8:
    return "EVXX_INT8";
  case FormatTag::EVXX_BFLOAT16:
    return "EVXX_BFLOAT16";
  }
  return "";
}

/// @brief `std::string` form of the canonical format token.
inline std::string format_tag_to_string(FormatTag tag) {
  return std::string(format_tag_name(tag));
}

/// @brief ASCII upper-case copy of @p value (does not touch non-ASCII bytes).
inline std::string upper_copy_ascii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
  return value;
}

/// @brief True iff @p value names BF16 (any of `BF16`, `BFLOAT16`, `EVXX_BF16`, `EVXX_BFLOAT16`).
inline bool tensor_format_is_bf16_alias(std::string value) {
  const std::string up = upper_copy_ascii(std::move(value));
  return up == "BF16" || up == "BFLOAT16" || up == "EVXX_BF16" || up == "EVXX_BFLOAT16";
}

/// @brief True iff @p value names FP32 (`FP32`, `FLOAT32`, `EVXX_FLOAT32`).
inline bool tensor_format_is_fp32_alias(std::string value) {
  const std::string up = upper_copy_ascii(std::move(value));
  return up == "FP32" || up == "FLOAT32" || up == "EVXX_FLOAT32";
}

/// @brief True iff @p value names INT8 (`INT8`, `EVXX_INT8`).
inline bool tensor_format_is_int8_alias(std::string value) {
  const std::string up = upper_copy_ascii(std::move(value));
  return up == "INT8" || up == "EVXX_INT8";
}

/// @brief True iff @p value names INT16 (`INT16`, `EVXX_INT16`).
inline bool tensor_format_is_int16_alias(std::string value) {
  const std::string up = upper_copy_ascii(std::move(value));
  return up == "INT16" || up == "EVXX_INT16";
}

/// @brief True iff @p value names INT32 (`INT32`, `EVXX_INT32`).
inline bool tensor_format_is_int32_alias(std::string value) {
  const std::string up = upper_copy_ascii(std::move(value));
  return up == "INT32" || up == "EVXX_INT32";
}

/**
 * @brief Normalize tensor format aliases to the canonical `EVXX_*` token.
 *
 * Maps user-facing or legacy spellings (FP32/FLOAT32, BF16/BFLOAT16, INT8/16/32)
 * onto the EV-side caps form. Unknown values are passed through unchanged.
 */
inline std::string normalize_tensor_caps_format(std::string value) {
  if (tensor_format_is_fp32_alias(value)) {
    return "EVXX_FLOAT32";
  }
  if (tensor_format_is_bf16_alias(value)) {
    return "EVXX_BFLOAT16";
  }
  if (tensor_format_is_int8_alias(value)) {
    return "EVXX_INT8";
  }
  if (tensor_format_is_int16_alias(value)) {
    return "EVXX_INT16";
  }
  if (tensor_format_is_int32_alias(value)) {
    return "EVXX_INT32";
  }
  return value;
}

/**
 * @brief Normalize a caps-format string conditionally on its media type.
 *
 * Tensor caps (`application/vnd.simaai.tensor`) are normalized to canonical
 * EVXX_ tokens; other media types (raw video, encoded video) are returned
 * unchanged.
 *
 * @param media_type GStreamer-style media type (e.g., `application/vnd.simaai.tensor`).
 * @param format     Format string to potentially normalize.
 * @return Normalized format if applicable, otherwise @p format unchanged.
 */
inline std::string normalize_caps_format_for_media(std::string media_type, std::string format) {
  if (format.empty()) {
    return format;
  }
  const std::string media_up = upper_copy_ascii(std::move(media_type));
  if (media_up == "APPLICATION/VND.SIMAAI.TENSOR") {
    return normalize_tensor_caps_format(std::move(format));
  }
  return format;
}

/// @brief Parse a string token to a `FormatTag`; unknown values map to `Auto`.
inline FormatTag format_tag_from_string(const std::string& value) {
  const std::string up = upper_copy_ascii(value);
  if (up.empty())
    return FormatTag::Auto;
  if (up == "RGB")
    return FormatTag::RGB;
  if (up == "BGR")
    return FormatTag::BGR;
  if (up == "GRAY" || up == "GRAY8")
    return FormatTag::GRAY8;
  if (up == "NV12")
    return FormatTag::NV12;
  if (up == "I420" || up == "YUV420")
    return FormatTag::I420;
  if (up == "YUYV")
    return FormatTag::YUYV;
  if (up == "ENCODED")
    return FormatTag::ENCODED;
  if (up == "H264")
    return FormatTag::H264;
  if (up == "BYTESTREAM" || up == "BYTE_STREAM" || up == "BYTE-STREAM" || up == "RAW_BYTES" ||
      up == "RAW-BYTES" || up == "OPAQUE_BYTES" || up == "OPAQUE-BYTES" || up == "OCTET_STREAM" ||
      up == "OCTET-STREAM")
    return FormatTag::ByteStream;
  if (up == "MLA")
    return FormatTag::MLA;
  if (up == "BBOX")
    return FormatTag::BBOX;
  if (up == "ARGMAX")
    return FormatTag::ARGMAX;
  if (up == "DETESSDEQUANT")
    return FormatTag::DETESSDEQUANT;
  if (up == "FP32")
    return FormatTag::FP32;
  if (up == "INT8")
    return FormatTag::INT8;
  if (up == "UINT8")
    return FormatTag::UINT8;
  if (up == "BF16" || up == "BFLOAT16")
    return FormatTag::BF16;
  if (up == "EVXX_BF16")
    return FormatTag::EVXX_BFLOAT16;
  if (up == "EVXX_FLOAT32")
    return FormatTag::EVXX_FLOAT32;
  if (up == "EVXX_INT8")
    return FormatTag::EVXX_INT8;
  if (up == "EVXX_BFLOAT16")
    return FormatTag::EVXX_BFLOAT16;
  return FormatTag::Auto;
}

/// @brief True iff @p tag names a raw (uncompressed) video format.
inline bool is_raw_video_format(FormatTag tag) {
  return tag == FormatTag::RGB || tag == FormatTag::BGR || tag == FormatTag::GRAY8 ||
         tag == FormatTag::NV12 || tag == FormatTag::I420 || tag == FormatTag::YUYV;
}

/// @brief True iff @p tag names a tensor payload (MLA, BBOX, ARGMAX, dtype kinds, EVXX aliases).
inline bool is_tensor_payload_format(FormatTag tag) {
  return tag == FormatTag::ByteStream || tag == FormatTag::MLA || tag == FormatTag::BBOX ||
         tag == FormatTag::ARGMAX || tag == FormatTag::DETESSDEQUANT || tag == FormatTag::FP32 ||
         tag == FormatTag::INT8 || tag == FormatTag::UINT8 || tag == FormatTag::BF16 ||
         tag == FormatTag::EVXX_FLOAT32 || tag == FormatTag::EVXX_INT8 ||
         tag == FormatTag::EVXX_BFLOAT16;
}

/**
 * @brief Thin wrapper around `FormatTag` with implicit string conversions.
 *
 * Used in option structs and caps fields where a format may arrive as either a
 * `FormatTag` enum value or a string token. Implicit conversions to/from
 * `std::string` keep call sites compact; the `empty()` and `operator bool()`
 * predicates both treat `Auto` as "unset".
 *
 * @ingroup pipeline
 * @see FormatTag
 */
struct FormatSpec {
  FormatTag tag = FormatTag::Auto; ///< Underlying tag (defaults to unset).

  /// @brief Default-construct as unset (`Auto`).
  constexpr FormatSpec() = default;
  /// @brief Construct from a `FormatTag` value.
  constexpr FormatSpec(FormatTag value) : tag(value) {}
  /// @brief Parse from a string token (case-insensitive); unknown tokens become `Auto`.
  FormatSpec(const std::string& value) : tag(format_tag_from_string(value)) {}
  /// @brief Parse from a C string token; null pointer treated as empty.
  FormatSpec(const char* value)
      : tag(format_tag_from_string(value ? std::string(value) : std::string{})) {}

  /// @brief Assign from a `FormatTag`.
  FormatSpec& operator=(FormatTag value) {
    tag = value;
    return *this;
  }
  /// @brief Assign from a parsed string token.
  FormatSpec& operator=(const std::string& value) {
    tag = format_tag_from_string(value);
    return *this;
  }
  /// @brief Assign from a parsed C string token.
  FormatSpec& operator=(const char* value) {
    tag = format_tag_from_string(value ? std::string(value) : std::string{});
    return *this;
  }

  /// @brief True iff the tag is `Auto` (unset).
  bool empty() const {
    return tag == FormatTag::Auto;
  }
  /// @brief Equality compares the underlying tag.
  bool operator==(const FormatSpec& other) const {
    return tag == other.tag;
  }
  /// @brief Inequality compares the underlying tag.
  bool operator!=(const FormatSpec& other) const {
    return tag != other.tag;
  }
  /// @brief Canonical string form of the tag.
  std::string str() const {
    return format_tag_to_string(tag);
  }
  /// @brief True iff the tag is set (anything other than `Auto`).
  explicit operator bool() const {
    return tag != FormatTag::Auto;
  }
  /// @brief Implicit conversion to canonical string token.
  operator std::string() const {
    return str();
  }
};

/// @brief Stream-insert the canonical string form of @p spec.
inline std::ostream& operator<<(std::ostream& os, const FormatSpec& spec) {
  os << spec.str();
  return os;
}

} // namespace simaai::neat
