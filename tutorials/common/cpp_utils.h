#pragma once

#include "tutorial_common.h"

#include <array>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tutorial_v2 {

using sima_tutorial::exists_or_skip;
using sima_tutorial::find_repo_root;
using sima_tutorial::get_arg;
using sima_tutorial::has_flag;
using sima_tutorial::join_args;
using sima_tutorial::print_common_flags;
using sima_tutorial::require;
using sima_tutorial::skip;
using sima_tutorial::wants_help;
using sima_tutorial::wants_print_gst;

inline int parse_int_arg(int argc, char** argv, const std::string& key, int def) {
  std::string value;
  if (!get_arg(argc, argv, key, value)) {
    return def;
  }
  try {
    return std::stoi(value);
  } catch (...) {
    throw std::invalid_argument("invalid integer for " + key + ": " + value);
  }
}

inline float parse_float_arg(int argc, char** argv, const std::string& key, float def) {
  std::string value;
  if (!get_arg(argc, argv, key, value)) {
    return def;
  }
  try {
    return std::stof(value);
  } catch (...) {
    throw std::invalid_argument("invalid float for " + key + ": " + value);
  }
}

inline bool strict_mode() {
  return std::getenv("SIMA_RUN_TUTORIALS_FULL") != nullptr;
}

inline std::filesystem::path first_existing(
    const std::initializer_list<std::filesystem::path>& candidates) {
  for (const auto& candidate : candidates) {
    if (std::filesystem::exists(candidate)) {
      return candidate;
    }
  }
  return {};
}

inline std::filesystem::path default_yolo_mpk(const std::filesystem::path& root) {
  return first_existing({
      root / "tmp" / "yolo_v8s_mpk.tar.gz",
      root / "tmp" / "yolov8s_mpk.tar.gz",
      root / "tmp" / "yolo_mpk.tar.gz",
  });
}

inline std::filesystem::path default_resnet_mpk(const std::filesystem::path& root) {
  return first_existing({
      root / "tmp" / "resnet_50_mpk.tar.gz",
      root / "tmp" / "resnet50_mpk.tar.gz",
  });
}

inline std::filesystem::path default_image(const std::filesystem::path& root) {
  return first_existing({
      root / "tmp" / "coco_sample.jpg",
      root / "tests" / "assets" / "preproc_dynamic" / "ilena_488.jpg",
      root / "test.jpg",
  });
}

inline void print_header(const std::string& title) {
  std::cout << "== " << title << " ==\n";
}

inline void step(const std::string& name, const std::string& detail = {}) {
  if (detail.empty()) {
    std::cout << "STEP " << name << "\n";
  } else {
    std::cout << "STEP " << name << ": " << detail << "\n";
  }
}

inline void emit(const std::string& tag, const std::string& detail = {}) {
  if (detail.empty()) {
    std::cout << tag << "\n";
  } else {
    std::cout << tag << " " << detail << "\n";
  }
}

inline void why(const std::string& detail) {
  emit("WHY", detail);
}

inline void tradeoff(const std::string& detail) {
  emit("TRADEOFF", detail);
}

inline void failure_mode(const std::string& detail) {
  emit("FAILURE_MODE", detail);
}

inline void interpret_output(const std::string& detail) {
  emit("INTERPRET", detail);
}

inline void runtime_fallback(const std::exception& e) {
  const char* what = e.what();
  std::cout << "runtime_fallback: " << (what ? what : "unknown") << "\n";
}

inline void runtime_fallback(const std::string& reason) {
  std::cout << "runtime_fallback: " << reason << "\n";
}

inline void check(const std::string& name, bool cond, const std::string& detail = {}) {
  std::cout << "CHECK " << name << ": " << (cond ? "PASS" : "FAIL");
  if (!detail.empty()) {
    std::cout << " (" << detail << ")";
  }
  std::cout << "\n";
  if (!cond) {
    throw std::runtime_error("check failed: " + name);
  }
}

inline bool has_signature_key(const std::vector<std::pair<std::string, std::string>>& values,
                              const std::string& key) {
  for (const auto& kv : values) {
    if (kv.first == key) {
      return true;
    }
  }
  return false;
}

inline void print_signature(const std::vector<std::pair<std::string, std::string>>& values) {
  static constexpr std::array<const char*, 7> kRequired = {
      "tutorial",
      "lang",
      "flow",
      "run_mode",
      "output_kind",
      "tensor_rank",
      "field_count",
  };
  for (const char* key : kRequired) {
    if (!has_signature_key(values, key)) {
      throw std::invalid_argument(std::string("missing signature key: ") + key);
    }
  }

  std::ostringstream oss;
  oss << "SIGNATURE {";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) {
      oss << ",";
    }
    oss << values[i].first << "=" << values[i].second;
  }
  oss << "}";
  std::cout << oss.str() << "\n";
}

inline std::string yes_no(bool v) {
  return v ? "yes" : "no";
}

} // namespace tutorial_v2
