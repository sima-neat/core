/**
 * @file
 * @ingroup builder
 * @brief Helpers for picking the downstream CPU/processor domain for a Node.
 *
 * Modalix Nodes can target multiple processor domains (APU, CVU, MLA). This
 * header provides the small enum + helpers the Builder uses to convert
 * domain hints (from plugin ids, JSON, or explicit caller intent) into the
 * `next_cpu` integer/string tags that the runtime plugins read.
 */
#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace simaai::neat {

/**
 * @brief Processor domain a Node's downstream stage runs on.
 * @ingroup builder
 */
enum class NextCpuDomain {
  APU,    ///< Application processor (general-purpose CPU).
  CVU,    ///< Computer-Vision Unit (EV74 vector cores).
  MLA,    ///< Machine-Learning Accelerator.
  Unknown ///< Domain could not be determined.
};

/**
 * @brief Encoded `next_cpu` value pair used by plugin JSON.
 *
 * Plugins consume both the integer and string forms; both are kept here so
 * callers can stamp them consistently.
 *
 * @ingroup builder
 */
struct NextCpuValue {
  int cpu_int = 0;             ///< Integer domain code.
  const char* cpu_str = "APU"; ///< String domain label.
};

/// @brief Convert a `NextCpuDomain` to its integer/string `NextCpuValue` pair.
NextCpuValue next_cpu_value(NextCpuDomain domain);

/// @brief Stable string label for a `NextCpuDomain` (e.g., for diagnostics).
const char* next_cpu_domain_name(NextCpuDomain domain);

/// @brief Parse a `NextCpuDomain` from its string label; unknown labels return `Unknown`.
NextCpuDomain next_cpu_domain_from_string(const std::string& value);

/// @brief Infer `NextCpuDomain` from a backend plugin id (e.g., `"sima_mla"` -> `MLA`).
NextCpuDomain next_cpu_domain_from_plugin_id(const std::string& plugin_id);

/**
 * @brief Stamp `next_cpu` fields into a Node's JSON config for the given domain.
 *
 * Updates the conventional locations (`root`, `params`, optionally `memory`)
 * so downstream plugins read consistent values. Each `force_*` flag controls
 * whether to overwrite a pre-existing value at that location.
 *
 * @param j             JSON config to mutate in place.
 * @param domain        Target processor domain.
 * @param force_root    Overwrite root-level `next_cpu` if present.
 * @param force_params  Overwrite `params.next_cpu` if present.
 * @param force_memory  Overwrite `memory.next_cpu` if present (default false).
 * @return True if any field was changed.
 */
bool apply_next_cpu_json(nlohmann::json& j, NextCpuDomain domain, bool force_root = true,
                         bool force_params = true, bool force_memory = false);

} // namespace simaai::neat
