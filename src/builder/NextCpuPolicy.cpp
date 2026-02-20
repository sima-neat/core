#include "builder/NextCpuPolicy.h"

#include "pipeline/internal/EnvUtil.h"

#include <algorithm>

namespace simaai::neat {
namespace {

std::string upper_copy(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back(static_cast<char>(std::toupper(static_cast<unsigned char>(c))));
  }
  return out;
}

enum class RootCpuType { String, Integer };

RootCpuType root_cpu_type(const nlohmann::json& j) {
  if (j.contains("next_cpu")) {
    if (j["next_cpu"].is_number_integer())
      return RootCpuType::Integer;
    if (j["next_cpu"].is_string())
      return RootCpuType::String;
  }
  if (j.contains("simaai__params") && j["simaai__params"].is_object()) {
    const auto& params = j["simaai__params"];
    if (params.contains("next_cpu") && params["next_cpu"].is_number_integer()) {
      return RootCpuType::Integer;
    }
  }
  return RootCpuType::String;
}

} // namespace

NextCpuValue next_cpu_value(NextCpuDomain domain) {
  switch (domain) {
  case NextCpuDomain::CVU:
    return {1, "CVU"};
  case NextCpuDomain::MLA:
    return {2, "MLA"};
  case NextCpuDomain::APU:
  default:
    return {0, "APU"};
  }
}

const char* next_cpu_domain_name(NextCpuDomain domain) {
  switch (domain) {
  case NextCpuDomain::CVU:
    return "CVU";
  case NextCpuDomain::MLA:
    return "MLA";
  case NextCpuDomain::APU:
    return "APU";
  case NextCpuDomain::Unknown:
  default:
    return "Unknown";
  }
}

NextCpuDomain next_cpu_domain_from_string(const std::string& value) {
  const std::string up = upper_copy(value);
  if (up == "CVU" || up == "1")
    return NextCpuDomain::CVU;
  if (up == "MLA" || up == "2")
    return NextCpuDomain::MLA;
  if (up == "APU" || up == "A65" || up == "0")
    return NextCpuDomain::APU;
  return NextCpuDomain::Unknown;
}

NextCpuDomain next_cpu_domain_from_plugin_id(const std::string& plugin_id) {
  const std::string up = upper_copy(plugin_id);
  if (up.find("PROCESSMLA") != std::string::npos)
    return NextCpuDomain::MLA;
  if (up.find("PROCESSCVU") != std::string::npos)
    return NextCpuDomain::CVU;
  return NextCpuDomain::Unknown;
}

bool apply_next_cpu_json(nlohmann::json& j, NextCpuDomain domain, bool force_root,
                         bool force_params, bool force_memory) {
  if (domain == NextCpuDomain::Unknown)
    return false;

  bool changed = false;
  const NextCpuValue v = next_cpu_value(domain);

  if (force_root || j.contains("next_cpu")) {
    const RootCpuType root_type = root_cpu_type(j);
    if (root_type == RootCpuType::Integer) {
      if (!j.contains("next_cpu") || !j["next_cpu"].is_number_integer() ||
          j["next_cpu"].get<int>() != v.cpu_int) {
        j["next_cpu"] = v.cpu_int;
        changed = true;
      }
    } else {
      if (!j.contains("next_cpu") || !j["next_cpu"].is_string() ||
          j["next_cpu"].get<std::string>() != v.cpu_str) {
        j["next_cpu"] = v.cpu_str;
        changed = true;
      }
    }
  }

  if (force_params || (j.contains("simaai__params") && j["simaai__params"].is_object())) {
    if (!j.contains("simaai__params") || !j["simaai__params"].is_object()) {
      j["simaai__params"] = nlohmann::json::object();
      changed = true;
    }
    nlohmann::json& params = j["simaai__params"];
    if (!params.contains("next_cpu") || !params["next_cpu"].is_number_integer() ||
        params["next_cpu"].get<int>() != v.cpu_int) {
      params["next_cpu"] = v.cpu_int;
      changed = true;
    }
    params["next_cpu_manual"] = 1;
  }

  if (force_memory || (j.contains("memory") && j["memory"].is_object())) {
    if (!j.contains("memory") || !j["memory"].is_object()) {
      j["memory"] = nlohmann::json::object();
      changed = true;
    }
    nlohmann::json& mem = j["memory"];
    if (!mem.contains("next_cpu") || !mem["next_cpu"].is_number_integer() ||
        mem["next_cpu"].get<int>() != v.cpu_int) {
      mem["next_cpu"] = v.cpu_int;
      changed = true;
    }
  }

  return changed;
}

} // namespace simaai::neat
