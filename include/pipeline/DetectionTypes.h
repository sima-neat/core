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
  float x1 = 0.0f;     ///< Top-left corner X, in pixels.
  float y1 = 0.0f;     ///< Top-left corner Y, in pixels.
  float x2 = 0.0f;     ///< Bottom-right corner X, in pixels.
  float y2 = 0.0f;     ///< Bottom-right corner Y, in pixels.
  float score = 0.0f;  ///< Class confidence in [0, 1].
  int class_id = -1;   ///< Predicted class index; -1 means unset.
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
  std::vector<Box> boxes;     ///< Parsed detections.
  std::vector<uint8_t> raw;   ///< Source bytes the boxes were parsed from.
};

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

} // namespace simaai::neat
