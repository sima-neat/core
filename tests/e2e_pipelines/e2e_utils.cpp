#include "e2e_pipelines/e2e_utils.h"

namespace sima_e2e {

bool download_file(const std::string& url, const std::filesystem::path& out_path) {
  return sima_test::download_file(url, out_path);
}

std::string resolve_yolov8s_tar(const std::filesystem::path& root) {
  return sima_test::resolve_yolov8s_tar(root);
}

std::filesystem::path ensure_coco_sample(const std::filesystem::path& root) {
  return sima_test::ensure_coco_sample(root);
}

} // namespace sima_e2e
