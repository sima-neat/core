#pragma once

#include <filesystem>
#include <string>

#include "asset_utils.h"

namespace sima_e2e {

bool download_file(const std::string& url, const std::filesystem::path& out_path);
std::string resolve_yolov8s_tar(const std::filesystem::path& root = {});
std::filesystem::path ensure_coco_sample(const std::filesystem::path& root = {});

} // namespace sima_e2e
