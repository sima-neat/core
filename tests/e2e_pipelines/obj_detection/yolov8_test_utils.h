#pragma once

#include "e2e_pipelines/e2e_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>

namespace sima_yolov8_test {

namespace fs = std::filesystem;

inline bool step_log_enabled() {
  static int enabled = -1;
  if (enabled >= 0)
    return enabled != 0;
  const char* env = std::getenv("SIMA_STEP_LOG");
  if (!env || !*env) {
    enabled = 1;
    return true;
  }
  std::string v(env);
  for (char& c : v) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (v == "0" || v == "false" || v == "no" || v == "off") {
    enabled = 0;
    return false;
  }
  enabled = 1;
  return true;
}

inline void step_log(const char* label) {
  if (!step_log_enabled())
    return;
  std::cout << "[STEP] " << label << std::endl;
}

inline void append_note(std::string& note, const std::string& part) {
  if (part.empty())
    return;
  if (!note.empty())
    note += ";";
  note += part;
}

inline std::string sanitize_note(const std::string& s) {
  std::string out;
  out.reserve(s.size());
  for (char c : s) {
    out.push_back((c == '\n' || c == '\r') ? '|' : c);
  }
  return out;
}

inline int env_int(const char* name, int def) {
  const char* val = std::getenv(name);
  if (!val || !*val)
    return def;
  return std::atoi(val);
}

inline bool env_bool(const char* name, bool def) {
  const char* val = std::getenv(name);
  if (!val || !*val)
    return def;
  std::string v(val);
  for (char& c : v) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (v == "0" || v == "false" || v == "no" || v == "off")
    return false;
  if (v == "1" || v == "true" || v == "yes" || v == "on")
    return true;
  return def;
}

inline std::filesystem::path default_people_image_path(const fs::path& root = {}) {
  const fs::path base = root.empty() ? fs::current_path() : root;
  return base / "tmp" / "yolov8s_people.jpg";
}

inline std::filesystem::path resolve_people_image_path(const fs::path& root = {}) {
  return default_people_image_path(root);
}

inline cv::Mat load_people_image_or_skip(const fs::path& root = {}) {
  fs::path img_path = resolve_people_image_path(root);
  if (!fs::exists(img_path)) {
    const fs::path downloaded = sima_e2e::ensure_coco_sample(root);
    if (downloaded.empty() || !fs::exists(downloaded)) {
      skip_long_test_exception("Missing yolov8 test image and failed to download zidane sample. "
                               "Set SIMA_COCO_URL to override.");
    }
    img_path = downloaded;
  }
  cv::Mat img = cv::imread(img_path.string(), cv::IMREAD_COLOR);
  require(!img.empty(), "Failed to read image: " + img_path.string());
  return img;
}

inline std::string resolve_yolov8s_tar_or_skip(const fs::path& root = {}) {
  const std::string tar_gz = sima_e2e::resolve_yolov8s_tar(root);
  if (tar_gz.empty()) {
    skip_long_test_exception("Failed to locate yolo_v8s MPK tarball. Set SIMA_YOLO_TAR or run "
                             "sima-cli modelzoo get yolo_v8s.");
  }
  return tar_gz;
}

} // namespace sima_yolov8_test
