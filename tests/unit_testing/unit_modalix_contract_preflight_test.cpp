#include "asset_utils.h"
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
               "appsrc",       "appsink",    "filesrc", "h264parse", "h265parse",
               "videoconvert", "videoscale", "queue",   "identity",  "neatdecoder",
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

           const fs::path decoder_fixture = sima_test::test_decoder_fixture_path();

           require(fs::exists(decoder_fixture),
                   "Modalix preflight: missing decoder fixture " + decoder_fixture.string());
         }));
