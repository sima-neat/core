#pragma once

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <cstdio>
#include <string>

namespace simaai::neat {

inline bool wire_debug_enabled() {
  const char* env = std::getenv("SIMA_WIRE_DEBUG");
  return env && *env && std::string(env) != "0";
}

inline bool set_input_buffer_names(nlohmann::json& j, const std::string& name) {
  if (name.empty())
    return false;
  bool changed = false;

  if (!j.contains("input_buffers") || !j["input_buffers"].is_array()) {
    j["input_buffers"] = nlohmann::json::array();
  }
  {
    auto& arr = j["input_buffers"];
    if (arr.empty()) {
      arr.push_back(nlohmann::json::object());
    }
    for (auto& entry : arr) {
      if (!entry.is_object())
        entry = nlohmann::json::object();
      entry["name"] = name;
      changed = true;
    }
  }

  if (!j.contains("buffers") || !j["buffers"].is_object()) {
    j["buffers"] = nlohmann::json::object();
  }
  {
    auto& buffers = j["buffers"];
    if (!buffers.contains("input") || !buffers["input"].is_array()) {
      buffers["input"] = nlohmann::json::array();
    }
    auto& arr = buffers["input"];
    if (arr.empty()) {
      arr.push_back(nlohmann::json::object());
    }
    for (auto& entry : arr) {
      if (!entry.is_object())
        entry = nlohmann::json::object();
      entry["name"] = name;
      changed = true;
    }
  }

  if (changed && wire_debug_enabled()) {
    const auto in_sz = (j.contains("input_buffers") && j["input_buffers"].is_array())
                           ? j["input_buffers"].size()
                           : 0;
    const auto buf_sz = (j.contains("buffers") && j["buffers"].is_object() &&
                         j["buffers"].contains("input") && j["buffers"]["input"].is_array())
                            ? j["buffers"]["input"].size()
                            : 0;
    std::fprintf(stderr,
                 "[WIRE] set_input_buffer_names name=%s input_buffers=%zu buffers.input=%zu\n",
                 name.c_str(), static_cast<size_t>(in_sz), static_cast<size_t>(buf_sz));
  }
  return changed;
}

inline bool set_input_buffer_name_if_exists(nlohmann::json& j, const std::string& name) {
  if (name.empty())
    return false;
  if (!j.contains("input_buffers") || !j["input_buffers"].is_array()) {
    if (wire_debug_enabled()) {
      std::fprintf(stderr, "[WIRE] set_input_buffer_name_if_exists skip=no_input_buffers name=%s\n",
                   name.c_str());
    }
    return false;
  }
  auto& arr = j["input_buffers"];
  if (arr.empty()) {
    if (wire_debug_enabled()) {
      std::fprintf(stderr,
                   "[WIRE] set_input_buffer_name_if_exists skip=empty_input_buffers name=%s\n",
                   name.c_str());
    }
    return false;
  }
  bool changed = false;
  for (auto& entry : arr) {
    if (!entry.is_object())
      continue;
    if (arr.size() == 1) {
      auto it = entry.find("name");
      if (it == entry.end() || (it->is_string() && it->get<std::string>().empty())) {
        if (wire_debug_enabled()) {
          std::fprintf(stderr, "[WIRE] set_input_buffer_name_if_exists skip=single_empty name=%s\n",
                       name.c_str());
        }
        continue;
      }
    }
    entry["name"] = name;
    changed = true;
  }
  if (wire_debug_enabled()) {
    std::fprintf(stderr,
                 "[WIRE] set_input_buffer_name_if_exists name=%s changed=%d input_buffers=%zu\n",
                 name.c_str(), changed ? 1 : 0, static_cast<size_t>(arr.size()));
  }
  return changed;
}

} // namespace simaai::neat
