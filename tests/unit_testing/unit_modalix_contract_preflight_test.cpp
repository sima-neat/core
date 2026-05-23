#include "gst/GstHelpers.h"
#include "gst/GstInit.h"
#include "test_main.h"

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

bool tool_available(const std::string& tool) {
  const std::string cmd = "command -v " + tool + " >/dev/null 2>&1";
  return std::system(cmd.c_str()) == 0;
}

fs::path find_repo_root() {
  std::error_code ec;
  fs::path path = fs::current_path(ec);
  if (ec) {
    return fs::current_path();
  }
  while (!path.empty()) {
    if (fs::exists(path / "tests" / "assets" / "decoder" / "dynamic_caps.h264", ec) && !ec) {
      return path;
    }
    const fs::path parent = path.parent_path();
    if (parent == path)
      break;
    path = parent;
  }
  return fs::current_path();
}

} // namespace

RUN_TEST("unit_modalix_contract_preflight_test", ([] {
           simaai::neat::gst_init_once();

           const std::vector<std::string> required_tools = {
               "sima-cli",
           };

           std::vector<std::string> missing_tools;
           for (const auto& tool : required_tools) {
             if (!tool_available(tool)) {
               missing_tools.push_back(tool);
             }
           }

           require(missing_tools.empty(), [&] {
             std::ostringstream oss;
             oss << "Modalix preflight: missing required tools:";
             for (const auto& tool : missing_tools) {
               oss << " " << tool;
             }
             return oss.str();
           }());

           const std::vector<const char*> required_elements = {
               "appsrc",     "appsink", "filesrc",  "h264parse",   "videoconvert",
               "videoscale", "queue",   "identity", "neatdecoder",
           };

           std::vector<std::string> missing_elements;
           for (const char* elem : required_elements) {
             if (!simaai::neat::element_exists(elem)) {
               missing_elements.emplace_back(elem);
             }
           }

           require(missing_elements.empty(), [&] {
             std::ostringstream oss;
             oss << "Modalix preflight: missing required GStreamer elements:";
             for (const auto& elem : missing_elements) {
               oss << " " << elem;
             }
             return oss.str();
           }());

           const fs::path repo_root = find_repo_root();
           const fs::path decoder_fixture =
               repo_root / "tests" / "assets" / "decoder" / "dynamic_caps.h264";

           require(fs::exists(decoder_fixture),
                   "Modalix preflight: missing decoder dynamic_caps.h264 fixture");
         }));
