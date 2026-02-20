#include "pipeline/internal/TempJsonFileUtil.h"

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <unistd.h>

namespace simaai::neat::pipeline_internal {

std::string make_temp_json_path(const std::string& dir, const std::string& prefix,
                                const char* error_label) {
  const char* label = (error_label && *error_label) ? error_label : "TempJsonFile";
  std::string root = dir.empty() ? "/tmp" : dir;
  std::error_code ec;
  std::filesystem::create_directories(root, ec);
  if (ec) {
    throw std::runtime_error(std::string(label) + ": failed to create config dir: " + root);
  }

  const std::string file_prefix = prefix.empty() ? "sima_temp_json" : prefix;
  const std::string templ = (std::filesystem::path(root) / (file_prefix + "_XXXXXX.json")).string();
  std::vector<char> path_buf(templ.begin(), templ.end());
  path_buf.push_back('\0');

  constexpr int kJsonSuffixLen = 5; // ".json"
  const int fd = mkstemps(path_buf.data(), kJsonSuffixLen);
  if (fd < 0) {
    std::ostringstream oss;
    oss << label << ": mkstemps failed";
    const char* reason = std::strerror(errno);
    if (reason && *reason) {
      oss << ": " << reason;
    }
    throw std::runtime_error(oss.str());
  }
  close(fd);
  return std::string(path_buf.data());
}

} // namespace simaai::neat::pipeline_internal
