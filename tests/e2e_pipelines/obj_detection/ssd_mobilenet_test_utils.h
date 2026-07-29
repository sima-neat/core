#pragma once

#include "e2e_pipelines/e2e_utils.h"
#include "e2e_pipelines/obj_detection/obj_detection_utils.h"
#include "test_utils.h"

#include <opencv2/imgcodecs.hpp>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace sima_ssd_mobilenet_test {

namespace fs = std::filesystem;

// Full download URL for the compiled SSD-MobileNetV2 pack. Overridable via env for mirrors;
// defaults to the SDK2.1.2 modalix model download (OAuth-gated docs.sima.ai endpoint that
// sima_test::download_file resolves via sima-cli), mirroring yolov8n_variant_base_url().
inline std::string ssd_mobilenet_pack_url() {
  if (const char* env = std::getenv("SIMA_SSD_MOBILENET_URL"); env && *env)
    return env;
  return "https://docs.sima.ai/pkg_downloads/SDK2.1.2/models/modalix/"
         "ssd_mobilenet_v2_heads_mpk.tar.gz";
}

// Compiled SSD-MobileNetV2-COCO pack (12 grouped conv heads, on-device SSD boxdecode). The archive
// is not part of the modelzoo, so it is taken from an explicit env var or on-disk staging when
// present, and otherwise fetched on demand. Like the yolov8n variant matrix, this test is mandatory
// in CI and must NOT skip when the pack is missing: sima_test::download_file prefers sima-cli for
// the OAuth-gated docs.sima.ai endpoint (falls back to curl/wget), and the archive is validated
// with `tar -tzf`. Point SIMA_SSD_MOBILENET_TAR at a local pack, or SIMA_SSD_MOBILENET_URL at a
// mirror, to override the download.
inline std::string resolve_ssd_mobilenet_tar_or_skip(const fs::path& root = {}) {
  const auto exists_no_throw = [](const fs::path& p) {
    std::error_code ec;
    const bool ok = fs::exists(p, ec);
    return ok && !ec;
  };

  // Fast paths: explicit env var, then conventional on-disk locations.
  for (const char* var : {"SIMA_SSD_MOBILENET_TAR", "SIMA_SSD_TAR"}) {
    const char* value = std::getenv(var);
    if (value && *value && exists_no_throw(value))
      return value;
  }

  const fs::path base = root.empty() ? fs::current_path() : root;
  const fs::path fallbacks[] = {
      base / "tmp" / "ssd_mobilenet_v2_heads_mpk.tar.gz",
      base / "ssd_mobilenet_v2_heads_mpk.tar.gz",
  };
  for (const auto& candidate : fallbacks) {
    if (exists_no_throw(candidate))
      return candidate.string();
  }

  // Not staged locally: fetch on demand. Prefer <root>/tmp; fall back to a writable scratch dir
  // when root is a build-host path that doesn't exist on the test runner (same approach as
  // ensure_yolov8n_drive_variants / ensure_coco_sample).
  fs::path dest_dir = base / "tmp";
  std::error_code ec;
  fs::create_directories(dest_dir, ec);
  if (ec) {
    fs::path scratch;
    try {
      scratch = fs::temp_directory_path();
    } catch (...) {
      scratch = fs::path("/tmp");
    }
    dest_dir = scratch / "sima-ssd-mobilenet";
    ec.clear();
    fs::create_directories(dest_dir, ec);
    if (ec) {
      dest_dir = fs::path("/tmp") / "sima-ssd-mobilenet";
      ec.clear();
      fs::create_directories(dest_dir, ec);
    }
  }
  require(!ec, "resolve_ssd_mobilenet_tar_or_skip: failed to create " + dest_dir.string() + " (" +
                   ec.message() + ")");
  const sima_test::ScopedFileLock lock(dest_dir / ".download.lock");

  const fs::path tar_path = dest_dir / "ssd_mobilenet_v2_heads_mpk.tar.gz";
  if (sima_test::is_listable_tar_gz(tar_path))
    return tar_path.string();
  sima_test::purge_unlistable_tar_gz(tar_path);

  const std::string url = ssd_mobilenet_pack_url();
  const bool ok = sima_test::download_file(url, tar_path);
  require(ok, "resolve_ssd_mobilenet_tar_or_skip: failed to download " + url + " -> " +
                  tar_path.string() +
                  " (run `sima-cli login` if the docs.sima.ai endpoint requires OAuth, set "
                  "SIMA_SSD_MOBILENET_URL to a mirror, or SIMA_SSD_MOBILENET_TAR to a local pack)");
  require(sima_test::is_listable_tar_gz(tar_path),
          "resolve_ssd_mobilenet_tar_or_skip: downloaded archive is not a readable tar.gz: " +
              tar_path.string());
  return tar_path.string();
}

// The same COCO sample ("zidane") the YOLOv8 e2e tests use, downloaded on demand.
inline cv::Mat load_coco_people_image_or_skip(const fs::path& root = {}) {
  const fs::path image_path = sima_e2e::ensure_coco_sample(root);
  std::error_code ec;
  if (image_path.empty() || !fs::exists(image_path, ec) || ec) {
    skip_long_test_exception("Missing SSD test image and failed to download COCO sample. Set "
                             "SIMA_COCO_URL to override.");
  }
  cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  require(!img.empty(), "Failed to read image: " + image_path.string());
  return img;
}

// Golden detections produced by the bit-exact host SSD decode (== TensorFlow full model) on the
// zidane sample, in original-frame pixels. class_id follows the MS-COCO 90-index map that this
// model emits (1 = person). Only the two strong person detections are asserted: the on-device
// int8 pipeline stays well within IoU tolerance of these, while weaker classes (e.g. the tie at
// ~0.42) are near the int8 score floor and intentionally left out to keep the check stable.
inline std::vector<objdet::ExpectedBox> expected_ssd_people_boxes() {
  return {
      {737.0f, 54.0f, 1147.0f, 699.0f, 1},
      {96.0f, 194.0f, 1105.0f, 707.0f, 1},
  };
}

} // namespace sima_ssd_mobilenet_test
