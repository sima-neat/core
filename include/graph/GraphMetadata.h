/**
 * @file
 * @ingroup graph
 * @brief Stream metadata helpers for graph samples.
 */
#pragma once

#include "graph/Graph.h"
#include "pipeline/SessionOptions.h"

#include <cstdint>
#include <string>
#include <unordered_map>

namespace simaai::neat::graph {

struct StreamMetadataDefaults {
  std::string stream_id;
  std::string caps_string;
  std::string port_name;
  std::string media_type;
  std::string format;
  std::string payload_tag;
  bool fill_stream_id = true;
  bool fill_input_seq = true;
  bool fill_orig_input_seq = true;
};

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
  if (sample.port_name.empty() && !defaults.port_name.empty()) {
    sample.port_name = defaults.port_name;
  }
  if (sample.media_type.empty() && !defaults.media_type.empty()) {
    sample.media_type = defaults.media_type;
  }
  if (sample.format.empty() && !defaults.format.empty()) {
    sample.format = defaults.format;
  }
  if (sample.payload_tag.empty() && !defaults.payload_tag.empty()) {
    sample.payload_tag = defaults.payload_tag;
  }
}

} // namespace simaai::neat::graph
