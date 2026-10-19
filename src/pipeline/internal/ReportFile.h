/**
 * @file
 * @brief YAML and JSON report files shared by benchmark and telemetry reports (not part of the
 * public API).
 */
#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include "pipeline/TelemetryMonitor.h"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <string>

namespace simaai::neat::pipeline_internal {

/// Report contents, in the key order they are written.
using ReportDocument = nlohmann::ordered_json;

/// The telemetry block of a report. Throws `std::invalid_argument`, prefixed with `context`, for
/// a negative count or a non-finite value.
ReportDocument telemetry_document(const Telemetry& telemetry, const std::string& context);

/// `document` as YAML text that starts with the `header` comment lines.
std::string report_yaml(const ReportDocument& document, const std::string& header,
                        const std::string& context);

/**
 * @brief Write `document` to `file_path`: JSON for a `.json` filename, otherwise YAML that
 * starts with `yaml_header`.
 *
 * Creates missing parent directories and atomically replaces an existing file. Throws, prefixed
 * with `context`, when the path names no file or the write fails.
 */
void save_report(const std::filesystem::path& file_path, const ReportDocument& document,
                 const std::string& yaml_header, const std::string& context);

} // namespace simaai::neat::pipeline_internal
