/**
 * @file
 * @brief List the models the host serves over PCIe (the `pcie-genai --list`
 *        command) and look one up by name.
 *
 * The models live on the host, under a directory the simaai-mla-daemon config
 * names in its [serve] section (by default `models = /srv/simaai/models`). The
 * card cannot enumerate that folder over PCIe -- its transport only pulls a
 * file by exact name -- so listing is done here, on the host, by reading the
 * local directory. Everything here is pure (no PCIe, no SSH) so it can be
 * unit-tested against config text and a temporary directory.
 */
#pragma once

#include "genai/GenAITypes.h"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace simaai::neat::pcie::genai::tools {

/// One model folder found under the serve root.
struct ModelListing {
  std::string name;              ///< subfolder name = the value to pass to --model
  std::string type;              ///< model_type from devkit/vlm_config.json, or "" if unknown
  std::uintmax_t size_bytes = 0; ///< total size of the folder on disk
};

/// Path of a [serve] root by name, parsed from simaai-mla-daemon.conf text.
/// Returns nullopt when the [serve] section or the name is absent. Takes the
/// text (not a path) so it can be tested without a config file on disk.
std::optional<std::string> serve_root_path(const std::string& conf_text,
                                           const std::string& serve_name);

/// The model_type field of a devkit/vlm_config.json body, or "" if the JSON
/// cannot be parsed or has no string model_type.
std::string model_type_from_vlm_config(const std::string& json_text);

/// The model subfolders under serve_root, sorted by name. A subfolder counts
/// as a model when it has both a devkit/ and an elf_files/ directory (the same
/// layout `llima run` requires). An unreadable or missing serve_root yields an
/// empty list, never throws.
std::vector<ModelListing> list_models_in(const std::filesystem::path& serve_root);

/// The human-readable text `--list` prints: a table of the models plus the
/// serve-root path and a hint on where to copy new models. Pure so the exact
/// wording is testable.
std::string format_model_listing(const std::vector<ModelListing>& models,
                                 const std::string& serve_root);

// The one-line run summary printed to stderr after an answer, plus a second
// line warning the user when token notifications were dropped in transit.
// No trailing newline; the caller adds one.
// dropped_events comes from GenAIModel::last_run_dropped_events().
std::string format_final_stats(const simaai::neat::genai::GenerationMetrics& metrics,
                               std::uint32_t dropped_events, const std::string& finish_reason);

} // namespace simaai::neat::pcie::genai::tools
