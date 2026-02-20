#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <cstdlib>
#include <cstring>
#include <string>

namespace simaai::neat::pipeline_internal {

inline bool env_bool(const char* key, bool def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return def_val;
  if (!std::strcmp(v, "1") || !std::strcmp(v, "true") || !std::strcmp(v, "TRUE") ||
      !std::strcmp(v, "yes") || !std::strcmp(v, "YES") || !std::strcmp(v, "on") ||
      !std::strcmp(v, "ON")) {
    return true;
  }
  if (!std::strcmp(v, "0") || !std::strcmp(v, "false") || !std::strcmp(v, "FALSE") ||
      !std::strcmp(v, "no") || !std::strcmp(v, "NO") || !std::strcmp(v, "off") ||
      !std::strcmp(v, "OFF")) {
    return false;
  }
  return def_val;
}

inline int env_int(const char* key, int def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return def_val;
  return std::atoi(v);
}

inline double env_double(const char* key, double def_val) {
  const char* v = std::getenv(key);
  if (!v || !*v)
    return def_val;
  return std::strtod(v, nullptr);
}

inline std::string env_str(const char* key, const std::string& def_val = "") {
  const char* v = std::getenv(key);
  if (!v)
    return def_val;
  return std::string(v);
}

inline bool env_int(const char* key, int* out) {
  if (!out)
    return false;
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  char* end = nullptr;
  long val = std::strtol(v, &end, 10);
  if (!end || *end != '\0')
    return false;
  *out = static_cast<int>(val);
  return true;
}

inline bool env_double(const char* key, double* out) {
  if (!out)
    return false;
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  char* end = nullptr;
  double val = std::strtod(v, &end);
  if (!end || *end != '\0')
    return false;
  *out = val;
  return true;
}

inline bool env_string(const char* key, std::string* out) {
  if (!out)
    return false;
  const char* v = std::getenv(key);
  if (!v || !*v)
    return false;
  *out = v;
  return true;
}

} // namespace simaai::neat::pipeline_internal
