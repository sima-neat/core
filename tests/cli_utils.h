#pragma once

#include <string>

namespace sima_test {

inline bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

inline bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

inline bool parse_int_arg(int argc, char** argv, const std::string& key, int& out) {
  std::string raw;
  if (!get_arg(argc, argv, key, raw))
    return false;
  out = std::stoi(raw);
  return true;
}

inline bool parse_float_arg(int argc, char** argv, const std::string& key, float& out) {
  std::string raw;
  if (!get_arg(argc, argv, key, raw))
    return false;
  out = std::stof(raw);
  return true;
}

inline bool parse_double_arg(int argc, char** argv, const std::string& key, double& out) {
  std::string raw;
  if (!get_arg(argc, argv, key, raw))
    return false;
  out = std::stod(raw);
  return true;
}

} // namespace sima_test
