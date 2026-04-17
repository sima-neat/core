#pragma once

// Tutorial helpers: argument parsing, skip handling, and asset discovery.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sima_tutorial {

inline bool has_flag(int argc, char** argv, const std::string& key) {
  for (int i = 1; i < argc; ++i) {
    if (key == argv[i])
      return true;
  }
  return false;
}

inline bool get_arg(int argc, char** argv, const std::string& key, std::string& out) {
  for (int i = 1; i + 1 < argc; ++i) {
    if (key == argv[i]) {
      out = argv[i + 1];
      return true;
    }
  }
  return false;
}

inline bool wants_help(int argc, char** argv) {
  return has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h");
}

inline bool wants_print_gst(int argc, char** argv) {
  return has_flag(argc, argv, "--print-gst");
}

inline void print_common_flags(std::ostream& os) {
  os << "  --help               Show this help message\n";
  os << "  --print-gst          Print the gst-launch string and exit\n";
}

inline int skip(const std::string& reason) {
  std::cout << "SKIP: " << reason << "\n";
  return 0;
}

inline bool exists_or_skip(const std::filesystem::path& path, const std::string& label) {
  if (!std::filesystem::exists(path)) {
    std::cout << "SKIP: missing " << label << " at " << path.string() << "\n";
    return false;
  }
  return true;
}

inline void require(bool ok, const std::string& msg) {
  if (!ok)
    throw std::runtime_error(msg);
}

inline std::filesystem::path find_repo_root() {
  std::filesystem::path cur = std::filesystem::current_path();
  for (int i = 0; i < 6; ++i) {
    if (std::filesystem::exists(cur / "CMakeLists.txt") &&
        std::filesystem::exists(cur / "include") && std::filesystem::exists(cur / "tests")) {
      return cur;
    }
    if (!cur.has_parent_path())
      break;
    cur = cur.parent_path();
  }
  return std::filesystem::current_path();
}

// Returns the directory containing tutorial sample assets.
// Lookup order:
//   1. SIMA_NEAT_TUTORIAL_ASSETS env var, if set and exists.
//   2. /usr/share/sima-neat/tutorials/assets (installed DEB layout).
//   3. /neat-resources/core-src/tutorials/assets (eLxr SDK layout).
//   4. <repo>/tutorials/assets (source checkout fallback).
inline std::filesystem::path find_asset_root() {
  namespace fs = std::filesystem;
  if (const char* env = std::getenv("SIMA_NEAT_TUTORIAL_ASSETS")) {
    fs::path p{env};
    if (fs::exists(p))
      return p;
  }
  for (const fs::path& p : {
           fs::path{"/usr/share/sima-neat/tutorials/assets"},
           fs::path{"/neat-resources/core-src/tutorials/assets"},
       }) {
    if (fs::exists(p))
      return p;
  }
  return find_repo_root() / "tutorials" / "assets";
}

inline std::string join_args(int argc, char** argv) {
  std::string out;
  for (int i = 0; i < argc; ++i) {
    if (i > 0)
      out += " ";
    out += argv[i];
  }
  return out;
}

} // namespace sima_tutorial
