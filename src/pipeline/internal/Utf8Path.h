#pragma once
#ifndef SIMA_NEAT_INTERNAL
#error "Internal header. Not part of the public API."
#endif

#include <filesystem>
#include <string>

namespace simaai::neat::pipeline_internal {

// Paths cross internal string interfaces as UTF-8. MSVC's narrow path conversions use the ANSI
// code page instead and throw for a name outside it, such as a non-ASCII profile or temporary
// directory, so Windows converts explicitly. POSIX paths are already the native narrow bytes.
inline std::filesystem::path utf8_path(const std::string& path) {
#ifdef _WIN32
  return std::filesystem::path(std::u8string(path.begin(), path.end()));
#else
  return std::filesystem::path(path);
#endif
}

inline std::string utf8_string(const std::filesystem::path& path) {
#ifdef _WIN32
  const std::u8string utf8 = path.u8string();
  return std::string(utf8.begin(), utf8.end());
#else
  return path.string();
#endif
}

} // namespace simaai::neat::pipeline_internal
