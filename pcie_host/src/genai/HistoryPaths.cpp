#include "genai/HistoryPaths.h"

#include <nlohmann/json.hpp>

#include <filesystem>

namespace simaai::neat::pcie::genai::internal {

namespace {
// True if `path` is `name`, ends with "/<name>", or has the same file name.
// The card's PcieFileProvider drops the subfolder ("pcie-genai/"), so a pulled
// image lands at <recv root>/<file name>. File names are unique per run
// ("h<pid>-<run>-<index><ext>"), so the file name alone identifies the image.
bool names_match(const std::string& path, const std::string& name) {
  if (path == name) {
    return true;
  }
  if (path.size() > name.size() &&
      path.compare(path.size() - name.size(), name.size(), name) == 0 &&
      path[path.size() - name.size() - 1] == '/') {
    return true;
  }
  const std::string file = std::filesystem::path(name).filename().string();
  return !file.empty() && std::filesystem::path(path).filename().string() == file;
}
} // namespace

std::string map_history_image_paths(const std::string& history_json,
                                    const std::map<std::string, std::string>& sent) {
  nlohmann::ordered_json j =
      nlohmann::ordered_json::parse(history_json, nullptr, /*allow_exceptions=*/false);
  if (j.is_discarded() || !j.is_array()) {
    return history_json;
  }
  for (auto& message : j) {
    if (!message.is_object() || !message.contains("content") || !message["content"].is_array()) {
      continue; // e.g. a plain-string system/assistant message
    }
    for (auto& item : message["content"]) {
      if (!item.is_object() || !item.contains("type") || item["type"] != "image" ||
          !item.contains("image") || !item["image"].is_string()) {
        continue;
      }
      const std::string path = item["image"].get<std::string>();
      for (const auto& [name, host_path] : sent) {
        if (names_match(path, name)) {
          item["image"] = host_path;
          break;
        }
      }
    }
  }
  return j.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
}

} // namespace simaai::neat::pcie::genai::internal
