#pragma once

#include <nlohmann/json.hpp>

#include <string>

namespace simaai::neat {

enum class NextCpuDomain { APU, CVU, MLA, Unknown };

struct NextCpuValue {
  int cpu_int = 0;
  const char* cpu_str = "APU";
};

NextCpuValue next_cpu_value(NextCpuDomain domain);
const char* next_cpu_domain_name(NextCpuDomain domain);
NextCpuDomain next_cpu_domain_from_string(const std::string& value);
NextCpuDomain next_cpu_domain_from_plugin_id(const std::string& plugin_id);

bool apply_next_cpu_json(nlohmann::json& j, NextCpuDomain domain, bool force_root = true,
                         bool force_params = true, bool force_memory = false);

} // namespace simaai::neat
