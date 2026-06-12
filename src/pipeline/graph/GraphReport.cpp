#include "pipeline/GraphReport.h"

#include <sstream>

namespace simaai::neat {
namespace {

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 16);
  for (char c : s) {
    switch (c) {
    case '\\':
      out += "\\\\";
      break;
    case '"':
      out += "\\\"";
      break;
    case '\b':
      out += "\\b";
      break;
    case '\f':
      out += "\\f";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(c) < 0x20) {
        const char* hex = "0123456789abcdef";
        out += "\\u00";
        out += hex[(static_cast<unsigned char>(c) >> 4) & 0xF];
        out += hex[static_cast<unsigned char>(c) & 0xF];
      } else {
        out += c;
      }
      break;
    }
  }
  return out;
}

void append_quoted(std::ostringstream& oss, const std::string& s) {
  oss << '"' << json_escape(s) << '"';
}

} // namespace

std::string GraphReport::to_json() const {
  std::ostringstream oss;
  oss << "{";

  oss << "\"pipeline_string\":";
  append_quoted(oss, pipeline_string);
  oss << ",";

  oss << "\"error_code\":";
  append_quoted(oss, error_code);
  oss << ",";

  oss << "\"nodes\":[";
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (i)
      oss << ",";
    const auto& n = nodes[i];
    oss << "{";
    oss << "\"index\":" << n.index << ",";
    oss << "\"kind\":";
    append_quoted(oss, n.kind);
    oss << ",";
    oss << "\"user_label\":";
    append_quoted(oss, n.user_label);
    oss << ",";
    oss << "\"backend_fragment\":";
    append_quoted(oss, n.backend_fragment);
    oss << ",";
    oss << "\"elements\":[";
    for (std::size_t j = 0; j < n.elements.size(); ++j) {
      if (j)
        oss << ",";
      append_quoted(oss, n.elements[j]);
    }
    oss << "]";
    oss << "}";
  }
  oss << "],";

  oss << "\"bus\":[";
  for (std::size_t i = 0; i < bus.size(); ++i) {
    if (i)
      oss << ",";
    const auto& b = bus[i];
    oss << "{";
    oss << "\"type\":";
    append_quoted(oss, b.type);
    oss << ",";
    oss << "\"src\":";
    append_quoted(oss, b.src);
    oss << ",";
    oss << "\"detail\":";
    append_quoted(oss, b.detail);
    oss << ",";
    oss << "\"wall_time_us\":" << b.wall_time_us;
    oss << "}";
  }
  oss << "],";

  oss << "\"boundaries\":[";
  for (std::size_t i = 0; i < boundaries.size(); ++i) {
    if (i)
      oss << ",";
    const auto& b = boundaries[i];
    oss << "{";
    oss << "\"boundary_name\":";
    append_quoted(oss, b.boundary_name);
    oss << ",";
    oss << "\"after_node_index\":" << b.after_node_index << ",";
    oss << "\"before_node_index\":" << b.before_node_index << ",";
    oss << "\"in_buffers\":" << b.in_buffers << ",";
    oss << "\"out_buffers\":" << b.out_buffers << ",";
    oss << "\"last_in_pts_ns\":" << b.last_in_pts_ns << ",";
    oss << "\"last_out_pts_ns\":" << b.last_out_pts_ns << ",";
    oss << "\"last_in_wall_us\":" << b.last_in_wall_us << ",";
    oss << "\"last_out_wall_us\":" << b.last_out_wall_us;
    oss << "}";
  }
  oss << "],";

  oss << "\"caps_dump\":";
  append_quoted(oss, caps_dump);
  oss << ",";

  oss << "\"dot_paths\":[";
  for (std::size_t i = 0; i < dot_paths.size(); ++i) {
    if (i)
      oss << ",";
    append_quoted(oss, dot_paths[i]);
  }
  oss << "],";

  oss << "\"repro_gst_launch\":";
  append_quoted(oss, repro_gst_launch);
  oss << ",";

  oss << "\"repro_env\":";
  append_quoted(oss, repro_env);
  oss << ",";

  oss << "\"repro_note\":";
  append_quoted(oss, repro_note);
  oss << ",";

  oss << "\"has_build_adaptation\":" << (has_build_adaptation ? "true" : "false") << ",";
  oss << "\"build_adaptation\":{";
  oss << "\"shape_policy\":";
  append_quoted(oss, build_adaptation.shape_policy);
  oss << ",";
  oss << "\"dynamic_capability\":";
  append_quoted(oss, build_adaptation.dynamic_capability);
  oss << ",";

  oss << "\"seed_width\":" << build_adaptation.seed_width << ",";
  oss << "\"seed_height\":" << build_adaptation.seed_height << ",";
  oss << "\"seed_depth\":" << build_adaptation.seed_depth << ",";
  oss << "\"seed_width_origin\":";
  append_quoted(oss, build_adaptation.seed_width_origin);
  oss << ",";
  oss << "\"seed_height_origin\":";
  append_quoted(oss, build_adaptation.seed_height_origin);
  oss << ",";
  oss << "\"seed_depth_origin\":";
  append_quoted(oss, build_adaptation.seed_depth_origin);
  oss << ",";

  oss << "\"max_width\":" << build_adaptation.max_width << ",";
  oss << "\"max_height\":" << build_adaptation.max_height << ",";
  oss << "\"max_depth\":" << build_adaptation.max_depth << ",";
  oss << "\"max_width_origin\":";
  append_quoted(oss, build_adaptation.max_width_origin);
  oss << ",";
  oss << "\"max_height_origin\":";
  append_quoted(oss, build_adaptation.max_height_origin);
  oss << ",";
  oss << "\"max_depth_origin\":";
  append_quoted(oss, build_adaptation.max_depth_origin);
  oss << ",";

  oss << "\"max_input_bytes_guard\":" << build_adaptation.max_input_bytes_guard << ",";
  oss << "\"byte_guard_origin\":";
  append_quoted(oss, build_adaptation.byte_guard_origin);
  oss << ",";
  oss << "\"allow_ingress_cvu_format_renegotiation\":"
      << (build_adaptation.allow_ingress_cvu_format_renegotiation ? "true" : "false") << ",";

  oss << "\"actions\":[";
  for (std::size_t i = 0; i < build_adaptation.actions.size(); ++i) {
    if (i)
      oss << ",";
    const auto& a = build_adaptation.actions[i];
    oss << "{";
    oss << "\"target\":";
    append_quoted(oss, a.target);
    oss << ",";
    oss << "\"applied\":" << (a.applied ? "true" : "false") << ",";
    oss << "\"detail\":";
    append_quoted(oss, a.detail);
    oss << ",";
    oss << "\"reason\":";
    append_quoted(oss, a.reason);
    oss << "}";
  }
  oss << "]";

  oss << "}";

  oss << "}";
  return oss.str();
}

} // namespace simaai::neat
