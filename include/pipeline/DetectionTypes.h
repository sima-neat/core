/**
 * @file
 * @ingroup pipeline
 * @brief Detection result types and helpers for decoding model outputs.
 *
 * Defines the structured output types produced after the BoxDecode postprocess
 * stage runs. `Box` is a single detection (axis-aligned bbox + class + score);
 * `BoxDecodeResult` bundles the parsed list with the raw byte payload it came
 * from. The free functions parse byte-encoded BBOX tensors back into typed
 * `Box` records for application code.
 *
 * @see BoxDecodeType for the decode-family selection.
 * @see StageRun.h for the BoxDecode stage entry point.
 */
#pragma once

#include "pipeline/TensorCore.h"
#include "pipeline/Tensor.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief One axis-aligned detection produced by BoxDecode.
 *
 * Coordinates are in the decoded image's pixel space (integer pixel units stored
 * as `float`). `class_id == -1` is the unset sentinel; valid detections always
 * carry a non-negative class id.
 *
 * @ingroup pipeline
 * @see BoxDecodeResult
 */
struct Box {
  float x1 = 0.0f;    ///< Top-left corner X, in pixels.
  float y1 = 0.0f;    ///< Top-left corner Y, in pixels.
  float x2 = 0.0f;    ///< Bottom-right corner X, in pixels.
  float y2 = 0.0f;    ///< Bottom-right corner Y, in pixels.
  float score = 0.0f; ///< Class confidence in [0, 1].
  int class_id = -1;  ///< Predicted class index; -1 means unset.
};

/**
 * @brief Parsed BoxDecode output paired with its raw byte buffer.
 *
 * `boxes` holds the typed detections; `raw` retains the underlying BBOX-format
 * byte payload so callers can re-parse, log, or forward the encoded form.
 *
 * @ingroup pipeline
 */
struct BoxDecodeResult {
  std::vector<Box> boxes;   ///< Parsed detections.
  std::vector<uint8_t> raw; ///< Source bytes the boxes were parsed from.
};

/// List form used by public stage APIs; even single-image decode results travel as a list.
using BoxDecodeResultList = std::vector<BoxDecodeResult>;

/**
 * @brief Parse a packed BBOX byte payload into typed `Box` records.
 *
 * @param bytes Raw bytes (the BBOX caps payload).
 * @param img_w Original image width used to clamp/scale coordinates.
 * @param img_h Original image height used to clamp/scale coordinates.
 * @param expected_topk Number of detections expected (advisory; see @p strict).
 * @param strict If true, mismatches against @p expected_topk throw; if false, parse best-effort.
 * @return Parsed detections (size may be less than @p expected_topk).
 */
std::vector<Box> parse_bbox_bytes(const std::vector<uint8_t>& bytes, int img_w, int img_h,
                                  int expected_topk, bool strict);

/**
 * @brief Decode a BBOX-format Tensor into a `BoxDecodeResult`.
 *
 * Convenience wrapper that maps the tensor, parses bytes via `parse_bbox_bytes`,
 * and retains the raw byte view alongside the typed boxes.
 *
 * @param tensor Tensor whose payload is a BBOX byte stream.
 * @param img_w  Original image width used to clamp/scale coordinates.
 * @param img_h  Original image height used to clamp/scale coordinates.
 * @param expected_topk Number of detections expected (advisory; see @p strict).
 * @param strict If true, mismatches against @p expected_topk throw.
 * @return Parsed detections plus retained raw bytes.
 */
BoxDecodeResult decode_bbox_tensor(const simaai::neat::Tensor& tensor, int img_w, int img_h,
                                   int expected_topk, bool strict);

/// Number of columns in a decoded-boxes tensor: x1, y1, x2, y2, score, class_id.
inline constexpr int64_t kDecodedBoxColumns = 6;

/**
 * @brief Build a decoded-boxes Tensor from parsed `Box` records.
 *
 * Produces a dense CPU `float32` tensor of shape `[boxes.size(), 6]`. Columns are
 * `x1, y1, x2, y2, score, class_id` (class_id stored as float; exact for the integer range).
 */
simaai::neat::Tensor boxes_to_tensor(const std::vector<Box>& boxes);

/**
 * @brief Decode a list of BBOX-format tensors into a list of decoded-boxes tensors.
 *
 * Positional, 1:1 — output `[i]` is the decode of input `[i]`. Each output is a dense CPU
 * `float32` tensor of shape `[num_detections, 6]` (columns `x1, y1, x2, y2, score, class_id`).
 *
 * Every input tensor must be BBOX-format (`semantic.detection.format == "BBOX"`); a tensor
 * that is not raises (no silent skip).
 *
 * @param bbox_tensors The BBOX-format tensors (e.g. a model's run output).
 * @param img_w Optional source width to clamp coordinates to (0 = no clamp).
 * @param img_h Optional source height to clamp coordinates to (0 = no clamp).
 * @param top_k Optional cap on detections per tensor (0 = no cap).
 * @param strict When true, malformed buffers throw instead of best-effort parsing.
 * @return A `TensorList` of the same length as @p bbox_tensors.
 */
simaai::neat::TensorList decode_bbox(const simaai::neat::TensorList& bbox_tensors, int img_w = 0,
                                     int img_h = 0, int top_k = 0, bool strict = false);

/**
 * @brief Tag a tensor as carrying a detection-decoder wire format.
 *
 * Sets `tensor.semantic.detection = DetectionSpec{.format = format}` so downstream
 * consumers (and `decode_bbox_tensor`, `pyneat.decode_bbox`) can recognize the payload
 * via the type-honest `DetectionSpec` slot instead of overloading `TessSpec::format`.
 *
 * @param tensor Tensor to mutate.
 * @param format Wire-format token (e.g., `"BBOX"`).
 */
void tag_detection_format(simaai::neat::Tensor& tensor, std::string format);

/**
 * @brief Read the detection-format token from a tensor, if present.
 *
 * Checks `tensor.semantic.detection->format` first. Falls back to the legacy
 * `tensor.semantic.tess->format` location for back-compat with un-migrated producers;
 * the fallback is scheduled for removal once all producers tag via `tag_detection_format`.
 *
 * @return The format token, or empty string if neither slot carries one.
 */
std::string read_detection_format(const simaai::neat::Tensor& tensor);

} // namespace simaai::neat

// Forward decl from Sample.h to avoid an extra include in this header.
namespace simaai::neat {
struct Sample;

/**
 * @brief Walk a Sample tree and tag every recognized BBOX-payload Tensor with
 *        `semantic.detection.format = "BBOX"`.
 *
 * Called as a runtime finalizer at every chokepoint where detection-stage outputs
 * exit the framework into user code (e.g., `Run::pull_tensors_strict`,
 * `Postprocess`). Normalises the type metadata so consumers can rely on the
 * type-honest `DetectionSpec` slot instead of inspecting `payload_tag`,
 * `Sample::format`, or the legacy `TessSpec::format` overload.
 */
void tag_detection_format_in_sample(simaai::neat::Sample& sample);

} // namespace simaai::neat
