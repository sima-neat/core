#include "pipeline/Session.h"
#include "pipeline/ErrorCodes.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::string tmp_json_path(const std::string& name) {
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "sima_neat_tests";
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

RUN_TEST(
    "unit_session_io_negative_test", ([] {
      using namespace simaai::neat;

      // Positive save/load roundtrip with escaped content.
      Session session;
      session.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
      session.custom("identity name=io_quote_test silent=true");
      session.custom("appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true");

      const std::string save_path = tmp_json_path("session_io_roundtrip.json");
      session.save(save_path);
      Session loaded = Session::load(save_path);

      const std::string original = session.describe_backend(false);
      const std::string restored = loaded.describe_backend(false);
      require(!original.empty(), "saved pipeline backend string is empty");
      require(original == restored, "save/load backend mismatch");

      // io.open: load() missing file.
      const std::string missing_file = tmp_json_path("session_io_missing_input.json");
      {
        std::error_code ec;
        std::filesystem::remove(missing_file, ec);
      }
      require_session_error([&]() { (void)Session::load(missing_file); }, error_codes::kIoOpen,
                            "io.open", "failed to open file for reading");

      // io.open: save() should fail when target path is a directory.
      const std::string tmp_dir =
          (std::filesystem::temp_directory_path() / "sima_neat_tests").string();
      {
        std::error_code ec;
        std::filesystem::create_directories(tmp_dir, ec);
      }
      require_session_error([&]() { session.save(tmp_dir); }, error_codes::kIoOpen, "io.open",
                            "failed to open file for writing");

      const std::string invalid_root = tmp_json_path("session_io_invalid_root.json");
      write_text(invalid_root, "[]\n");
      require_session_error([&]() { (void)Session::load(invalid_root); }, error_codes::kIoParse,
                            "io.parse", "invalid JSON root");

      const std::string missing_nodes = tmp_json_path("session_io_missing_nodes.json");
      write_text(missing_nodes, "{\"version\":1}\n");
      require_session_error([&]() { (void)Session::load(missing_nodes); }, error_codes::kIoParse,
                            "io.parse", "missing required 'nodes' array");

      const std::string invalid_node_entry = tmp_json_path("session_io_invalid_node_entry.json");
      write_text(invalid_node_entry, "{\"nodes\":[5]}\n");
      require_session_error([&]() { (void)Session::load(invalid_node_entry); },
                            error_codes::kIoParse, "io.parse", "node entry must be object");

      const std::string missing_fragment = tmp_json_path("session_io_missing_fragment.json");
      write_text(missing_fragment,
                 "{\"nodes\":[{\"kind\":\"Gst\",\"label\":\"x\",\"elements\":[]}]}\n");
      require_session_error([&]() { (void)Session::load(missing_fragment); }, error_codes::kIoParse,
                            "io.parse", "node fragment is empty");

      // io.parse: parser diagnostics should include byte offset + nearby token context.
      const std::string malformed_json = tmp_json_path("session_io_malformed.json");
      write_text(malformed_json, "{\"nodes\":[{\"kind\":\"Gst\" \"fragment\":\"identity\"}]}\n");
      require_session_error([&]() { (void)Session::load(malformed_json); }, error_codes::kIoParse,
                            "offset=", "near='");
    }));
