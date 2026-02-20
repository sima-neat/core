#include "pipeline/Session.h"
#include "runtime_test_utils.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string tmp_json_path(const std::string& name) {
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "sima_neat_session_io";
  std::error_code ec;
  fs::create_directories(dir, ec);
  return (dir / name).string();
}

void write_text(const std::string& path, const std::string& text) {
  std::ofstream out(path, std::ios::binary);
  if (!out.is_open()) {
    throw std::runtime_error("failed to open temp file for write: " + path);
  }
  out << text;
}

} // namespace

RUN_TEST("unit_session_io_roundtrip_test", ([] {
           using namespace simaai::neat;

           Session session;
           session.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
           session.custom("identity name=roundtrip_quote_test silent=true");
           session.custom(
               "appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true");

           const std::string save_path = tmp_json_path("roundtrip.json");
           session.save(save_path);

           Session loaded = Session::load(save_path);
           const std::string original = session.describe_backend(false);
           const std::string restored = loaded.describe_backend(false);
           require(!original.empty(), "unit_session_io_roundtrip_test: original pipeline is empty");
           require(original == restored, "unit_session_io_roundtrip_test: save/load mismatch");

           // malformed JSON should fail deterministically
           const std::string malformed = tmp_json_path("roundtrip_malformed.json");
           write_text(malformed, "{ broken json ");
           require(sima_test::throws_with([&]() { (void)Session::load(malformed); }, "JSON"),
                   "unit_session_io_roundtrip_test: malformed JSON should fail");

           // missing required fields should fail deterministically
           const std::string missing_nodes = tmp_json_path("roundtrip_missing_nodes.json");
           write_text(missing_nodes, "{\"version\":1}\n");
           require(sima_test::throws_with([&]() { (void)Session::load(missing_nodes); },
                                          "nodes' array"),
                   "unit_session_io_roundtrip_test: missing nodes should fail");
         }));
