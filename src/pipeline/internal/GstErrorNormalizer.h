#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <gst/gst.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace simaai::neat::pipeline_internal {

struct DiagnosticFact {
  std::string label;
  std::string value;
};

struct RawGstError {
  std::string source_name;
  std::string factory_name;
  std::string domain_name;
  int code = 0;
  std::string message;
  std::string debug;
  std::map<std::string, std::string> details;
  std::int64_t wall_time_us = 0;
};

struct NormalizedDiagnostic {
  std::string error_code;
  std::string diagnostic_id;
  std::string title;
  std::string explanation;
  std::string stage;
  std::string source;
  std::vector<DiagnosticFact> facts;
  std::vector<std::string> actions;
  RawGstError raw;
};

RawGstError parse_gst_error_message(GstMessage* message);
std::string sanitize_gst_diagnostic_text(std::string text);
NormalizedDiagnostic classify_gst_error(RawGstError raw);
NormalizedDiagnostic classify_gst_parse_error(const GError* error,
                                              const std::string& pipeline_string);

std::string render_diagnostic_body(const NormalizedDiagnostic& diagnostic,
                                   bool include_debug_details);
int diagnostic_priority(const NormalizedDiagnostic& diagnostic);

} // namespace simaai::neat::pipeline_internal
