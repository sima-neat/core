/**
 * @file
 * @ingroup pipeline
 * @brief Helpers for constructing and inspecting encoded-media samples.
 *
 * Encoded samples wrap compressed payloads (e.g., H.264 NAL units) in the
 * framework's `Sample` type so they can flow through the same pipeline as raw
 * frames or tensors. This header exposes a caps-string codec sniff and a
 * factory that wraps an owned byte buffer plus timing metadata into a `Sample`.
 *
 * @see GraphOptions.h for `EncodedSpec::Codec`.
 * @see Tensor.h for the underlying `Sample` type.
 */
#pragma once

#include "pipeline/GraphOptions.h"
#include "pipeline/TensorCore.h"

#include <cstdint>
#include <string>
#include <vector>

namespace simaai::neat {

/**
 * @brief Map a GStreamer-style caps string to an `EncodedSpec::Codec` value.
 *
 * @param caps_string GStreamer caps string (e.g., `"video/x-h264, ..."`).
 * @return Matching codec enum, or the codec-unspecified sentinel if no match.
 */
simaai::neat::EncodedSpec::Codec caps_to_codec(const std::string& caps_string);

/**
 * @brief Build an encoded `Sample` from raw bytes and caps metadata.
 *
 * The returned `Sample` owns @p bytes (moved in) and carries the supplied caps
 * string plus optional PTS/DTS/duration timestamps. Pass `-1` to leave a
 * timestamp unset.
 *
 * @param bytes       Encoded payload (e.g., H.264 access unit or NAL).
 * @param caps_string Caps string describing the codec/profile.
 * @param pts_ns      Presentation timestamp in nanoseconds, or -1 if unknown.
 * @param dts_ns      Decode timestamp in nanoseconds, or -1 if unknown.
 * @param duration_ns Frame duration in nanoseconds, or -1 if unknown.
 * @return A Sample wrapping the encoded payload with the supplied caps.
 */
Sample make_encoded_sample(std::vector<uint8_t> bytes, std::string caps_string, int64_t pts_ns = -1,
                           int64_t dts_ns = -1, int64_t duration_ns = -1);

} // namespace simaai::neat
