#pragma once

#include "e2e_pipelines/e2e_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

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
    skip_long_test_exception(
        "Failed to locate yolo_v8s MPK tarball. Set SIMA_MODEL_TAR (or SIMA_YOLO_TAR) or run "
        "sima-cli modelzoo get yolo_v8s.");
  }
  return tar_gz;
}

// Canonical YOLOv8n variant set used by the matrix tests
// (preproc_yolov8_matrix_test, yolov8_variant_route_matrix_test). Mirrors
// tmp/run_yolov8_matrix.sh so on-disk staging matches CI staging.
inline const std::vector<std::string>& yolov8n_drive_variant_stems() {
  static const std::vector<std::string> stems = {
      "yolov8n_A_BF16_W_INT8_MLATess", "yolov8n_A_BF16_W_INT8_mpk", "yolov8n_A_W_BF16_MLATess",
      "yolov8n_A_W_BF16_mpk",          "yolov8n_A_W_INT8_MLATess",  "yolov8n_A_W_int8_mpk",
  };
  return stems;
}

inline std::string yolov8n_variant_base_url() {
  if (const char* env = std::getenv("SIMA_YOLOV8N_VARIANTS_BASE_URL"); env && *env)
    return env;
  if (const char* env = std::getenv("SIMA_YOLOV8_VARIANTS_BASE_URL"); env && *env)
    return env;
  return "https://docs.sima.ai/pkg_downloads/SDK2.0.0/models/modalix";
}

// Ensure every YOLOv8n variant tarball is staged under
// <root>/tmp/yolov8n_drive/. The matrix tests are mandatory in CI and must
// NOT skip when the variants are missing: fetch them on demand via
// sima_test::download_file (which prefers sima-cli for the OAuth-gated
// docs.sima.ai endpoint and falls back to curl/wget). Each tarball is
// validated with `tar -tzf` after download; corrupt downloads are removed
// so the next retry isn't poisoned. Returns the variants directory.
inline fs::path ensure_yolov8n_drive_variants(const fs::path& root) {
  const fs::path drive_dir = root / "tmp" / "yolov8n_drive";
  std::error_code ec;
  fs::create_directories(drive_dir, ec);
  require(!ec, "ensure_yolov8n_drive_variants: failed to create " + drive_dir.string() +
                   " (" + ec.message() + ")");

  std::string base_url = yolov8n_variant_base_url();
  while (!base_url.empty() && base_url.back() == '/')
    base_url.pop_back();

  for (const auto& stem : yolov8n_drive_variant_stems()) {
    const fs::path tar_path = drive_dir / (stem + ".tar.gz");
    if (sima_test::is_listable_tar_gz(tar_path))
      continue;
    sima_test::purge_unlistable_tar_gz(tar_path);

    const std::string url = base_url + "/" + stem + ".tar.gz";
    const bool ok = sima_test::download_file(url, tar_path);
    require(ok, "ensure_yolov8n_drive_variants: failed to download " + url + " -> " +
                    tar_path.string() +
                    " (run `sima-cli login` if the docs.sima.ai endpoint requires OAuth, or "
                    "set SIMA_YOLOV8N_VARIANTS_BASE_URL to a mirror)");
    require(sima_test::is_listable_tar_gz(tar_path),
            "ensure_yolov8n_drive_variants: downloaded archive is not a readable tar.gz: " +
                tar_path.string());
  }
  return drive_dir;
}

} // namespace sima_yolov8_test
