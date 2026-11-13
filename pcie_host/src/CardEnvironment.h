#pragma once

#include <algorithm>
#include <cctype>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace simaai::neat::pcie::internal {

inline bool is_card_environment_name_char(const char ch) {
  return std::isalnum(static_cast<unsigned char>(ch)) || ch == '_';
}

inline std::vector<std::string> parse_card_environment(const std::string& raw) {
  std::vector<std::string> result;
  std::istringstream stream(raw);
  std::string token;
  while (stream >> token) {
    const std::size_t equals = token.find('=');
    if (equals == std::string::npos || equals == 0) {
      throw std::runtime_error("card_env entries must be NAME=VALUE assignments");
    }
    for (std::size_t index = 0; index < equals; ++index) {
      const char ch = token[index];
      if ((index == 0 && std::isdigit(static_cast<unsigned char>(ch))) ||
          !is_card_environment_name_char(ch)) {
        throw std::runtime_error("card_env contains invalid environment variable name: " +
                                 token.substr(0, equals));
      }
    }
    result.push_back(std::move(token));
  }
  return result;
}

inline bool card_environment_contains(const std::vector<std::string>& environment,
                                      const std::string& name) {
  const std::string prefix = name + "=";
  return std::any_of(environment.begin(), environment.end(),
                     [&prefix](const std::string& entry) { return entry.rfind(prefix, 0) == 0; });
}

inline std::optional<std::string>
card_environment_value(const std::vector<std::string>& environment, const std::string& name) {
  const std::string prefix = name + "=";
  std::optional<std::string> value;
  for (const auto& entry : environment) {
    if (entry.rfind(prefix, 0) == 0) {
      value = entry.substr(prefix.size());
    }
  }
  return value;
}

inline std::string card_environment_path_or(const std::vector<std::string>& environment,
                                            const char* name, const char* fallback) {
  const auto value = card_environment_value(environment, name);
  if (!value.has_value()) {
    return fallback;
  }
  if (value->empty() || value->front() != '/' || value->find('\n') != std::string::npos ||
      value->find('\r') != std::string::npos) {
    throw std::invalid_argument(std::string(name) + " must be an absolute POSIX path");
  }
  return *value;
}

} // namespace simaai::neat::pcie::internal
