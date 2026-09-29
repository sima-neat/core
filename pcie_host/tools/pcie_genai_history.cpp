#include "pcie_genai_history.h"

#include <cstdlib>

namespace simaai::neat::pcie::genai::tools {

namespace fs = std::filesystem;

namespace {
constexpr const char* kHistoryFilename = ".pcie_genai_history";
} // namespace

fs::path history_file_path() {
  const char* home = std::getenv("HOME");
  if (home == nullptr || *home == '\0') {
    home = ".";
  }
  return fs::path(home) / kHistoryFilename;
}

bool should_record_history(std::string_view line, std::string_view previous) {
  if (line.empty()) {
    return false;
  }
  return line != previous;
}

} // namespace simaai::neat::pcie::genai::tools
