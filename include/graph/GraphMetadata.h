/**
 * @file
 * @ingroup graph
 * @brief Stream metadata helpers for graph samples.
 */
#pragma once

#include "graph/Graph.h"
#include "pipeline/GraphOptions.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace simaai::neat::graph {

/**
 * @brief Defaults applied to a `Sample` by the `StreamMetadata` stage.
 *
 * Each non-empty field is filled into the matching `Sample` field when missing. The
 * `fill_*` flags control whether the executor should auto-assign `stream_id`,
 * `input_seq`, and `orig_input_seq` when not present.
 *
 * @see ensure_stream_metadata
 * @see simaai::neat::graph::nodes::StreamMetadata
 * @ingroup graph
 */
struct StreamMetadataDefaults {
  std::string stream_id;   ///< Default stream id; used if the sample has none and `fill_stream_id`.
  std::string caps_string; ///< Default GStreamer caps string.
  std::string stream_label;        ///< Default stream label.
  std::string port_name;           ///< Default port name.
  std::string media_type;          ///< Default media type (e.g., `"video/x-raw"`).
  std::string format;              ///< Default format (e.g., `"NV12"`).
  std::string payload_tag;         ///< Default payload tag.
  bool fill_stream_id = true;      ///< Auto-assign `stream_id` when missing.
  bool fill_input_seq = true;      ///< Auto-assign `input_seq` when missing.
  bool fill_orig_input_seq = true; ///< Auto-assign `orig_input_seq` when missing.
};

/**
 * @brief Apply `StreamMetadataDefaults` to a sample, filling in missing fields.
 *
 * Fills `stream_id`, sequence numbers, caps string, labels, media type, format, and payload
 * tag on `sample` from `defaults` where the sample lacks values. The optional `next_seq`
 * map is used to mint per-stream `input_seq` values when neither the sample nor `defaults`
 * supply one.
 *
 * @param sample   Sample to mutate in place.
 * @param defaults Defaults source.
 * @param next_seq Optional per-stream sequence counter, keyed by stream id.
 */
inline void ensure_stream_metadata(Sample& sample, const StreamMetadataDefaults& defaults,
                                   std::unordered_map<std::string, int64_t>* next_seq = nullptr) {
  if (sample.stream_id.empty() && defaults.fill_stream_id) {
    if (!defaults.stream_id.empty()) {
      sample.stream_id = defaults.stream_id;
    } else {
      sample.stream_id = "stream0";
    }
  }

  if (defaults.fill_input_seq && sample.input_seq < 0) {
    if (sample.orig_input_seq >= 0) {
      sample.input_seq = sample.orig_input_seq;
    } else if (next_seq) {
      auto& next = (*next_seq)[sample.stream_id];
      sample.input_seq = next++;
    } else {
      sample.input_seq = 0;
    }
  }

  if (defaults.fill_orig_input_seq && sample.orig_input_seq < 0) {
    sample.orig_input_seq = sample.input_seq;
  }

  if (sample.caps_string.empty() && !defaults.caps_string.empty()) {
    sample.caps_string = defaults.caps_string;
  }
  if (sample.stream_label.empty()) {
    if (!defaults.stream_label.empty()) {
      sample.stream_label = defaults.stream_label;
    } else if (!defaults.port_name.empty()) {
      sample.stream_label = defaults.port_name;
    }
  }
  if (sample.port_name.empty() && !defaults.port_name.empty()) {
    sample.port_name = defaults.port_name;
  }
  if (sample.media_type.empty() && !defaults.media_type.empty()) {
    sample.media_type = defaults.media_type;
  }
  if (sample.payload_type == PayloadType::Auto && !sample.media_type.empty()) {
    sample.payload_type = payload_type_from_media_type(sample.media_type);
  }
  if (sample.format.empty() && !defaults.format.empty()) {
    sample.format = defaults.format;
  }
  if (sample.payload_tag.empty() && !defaults.payload_tag.empty()) {
    sample.payload_tag = defaults.payload_tag;
  }
}

} // namespace simaai::neat::graph
