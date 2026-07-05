#include "pipeline/ErrorCodes.h"
#include "pipeline/Graph.h"
#include "nodes/common/Output.h"
#include "nodes/io/Input.h"
#include "test_main.h"
#include "test_utils.h"

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

std::string tmp_json_path(const std::string& name) {
  namespace fs = std::filesystem;
  fs::path dir = fs::temp_directory_path() / "sima_neat_graph_io_contract";
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

std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("failed to open temp file for read: " + path);
  }
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

std::string io_case(const char* id, const std::string& msg) {
  return std::string("SESSION_IO_CASE=") + id + " " + msg;
}

} // namespace

RUN_TEST(
    "unit_graph_io_contract_test", ([] {
      using namespace simaai::neat;

      Graph graph;
      graph.custom("videotestsrc num-buffers=1 pattern=black", InputRole::Source);
      graph.custom("identity name=io_quote_test silent=true");
      graph.custom("appsink name=mysink emit-signals=false sync=false max-buffers=1 drop=true");

      const std::string save_path = tmp_json_path("graph_io_roundtrip.json");
      graph.save(save_path);
      Graph loaded = Graph::load(save_path);

      const std::string original = graph.describe_backend(false);
      const std::string restored = loaded.describe_backend(false);
      require(!original.empty(),
              io_case("roundtrip_non_empty", "saved pipeline should be non-empty"));
      require(original == restored, io_case("roundtrip_match", "save/load backend mismatch"));

      Graph input_graph;
      InputOptions input_opt;
      input_opt.payload_type = PayloadType::Image;
      input_opt.format = FormatTag::RGB;
      input_opt.width = 16;
      input_opt.height = 16;
      input_opt.memory_policy = InputMemoryPolicy::Ev74;
      const auto input_node = nodes::Input(input_opt);
      const auto output_a = nodes::Output("out_a");
      const auto output_b = nodes::Output("out_b");
      input_graph.connect(input_node, output_a);
      input_graph.connect(input_node, output_b);

      const std::string memory_policy_path = tmp_json_path("graph_io_memory_policy.json");
      input_graph.save(memory_policy_path);
      const std::string saved_memory_policy_json = read_text(memory_policy_path);
      require_contains(saved_memory_policy_json, "\"memory_policy\":1",
                       io_case("memory_policy_saved", "input memory policy should be serialized"));
      require(saved_memory_policy_json.find("\"use_simaai_pool\"") == std::string::npos,
              io_case("deprecated_simaai_pool_omitted",
                      "deprecated input pool flag should not be serialized"));

      const Graph loaded_input = Graph::load(memory_policy_path);
      const std::string memory_policy_roundtrip_path =
          tmp_json_path("graph_io_memory_policy_roundtrip.json");
      loaded_input.save(memory_policy_roundtrip_path);
      require_contains(
          read_text(memory_policy_roundtrip_path), "\"memory_policy\":1",
          io_case("memory_policy_roundtrip", "input memory policy should survive save/load"));

      std::string legacy_memory_policy_json = saved_memory_policy_json;
      const std::string memory_policy_key = "\"memory_policy\":1";
      const std::size_t memory_policy_pos = legacy_memory_policy_json.find(memory_policy_key);
      require(memory_policy_pos != std::string::npos,
              io_case("legacy_simaai_pool_fixture", "memory policy fixture key missing"));
      legacy_memory_policy_json.replace(memory_policy_pos, memory_policy_key.size(),
                                        "\"use_simaai_pool\":false");
      const std::string legacy_memory_policy_path =
          tmp_json_path("graph_io_legacy_simaai_pool.json");
      write_text(legacy_memory_policy_path, legacy_memory_policy_json);
      const Graph loaded_legacy_input = Graph::load(legacy_memory_policy_path);
      const std::string legacy_roundtrip_path =
          tmp_json_path("graph_io_legacy_simaai_pool_roundtrip.json");
      loaded_legacy_input.save(legacy_roundtrip_path);
      const std::string legacy_roundtrip_json = read_text(legacy_roundtrip_path);
      require_contains(legacy_roundtrip_json, "\"memory_policy\":3",
                       io_case("legacy_simaai_pool_memory_policy",
                               "legacy pool flag should convert to SystemMemory policy"));
      require(legacy_roundtrip_json.find("\"use_simaai_pool\"") == std::string::npos,
              io_case("legacy_simaai_pool_not_reserialized",
                      "deprecated input pool flag should not be reserialized"));

      Graph stream_source("stream_source");
      stream_source.add(nodes::Input("image"));
      stream_source.add(nodes::Output("frame"));
      Graph stream_sink("stream_sink");
      stream_sink.add(nodes::Input("frame"));
      stream_sink.add(nodes::Output("classes"));

      Graph stream_app("graph_io_default_link_stream_id");
      GraphLinkOptions stream_link;
      stream_link.stream_id = "persisted-stream-0";
      stream_app.connect(stream_source, stream_sink, stream_link);

      const std::string stream_link_path = tmp_json_path("graph_io_default_link_stream_id.json");
      stream_app.save(stream_link_path);
      const std::string stream_link_json = read_text(stream_link_path);
      require_contains(
          stream_link_json, "\"link_stream_id\":\"persisted-stream-0\"",
          io_case("default_link_stream_id_saved", "default-link stream id should be serialized"));
      require(stream_link_json.find("\"link_policy\"") == std::string::npos,
              io_case("default_link_policy_omitted",
                      "default-link stream id should not require a non-default policy"));

      const Graph loaded_stream_app = Graph::load(stream_link_path);
      const std::string stream_link_roundtrip_path =
          tmp_json_path("graph_io_default_link_stream_id_roundtrip.json");
      loaded_stream_app.save(stream_link_roundtrip_path);
      const std::string stream_link_roundtrip_json = read_text(stream_link_roundtrip_path);
      require_contains(stream_link_roundtrip_json, "\"link_stream_id\":\"persisted-stream-0\"",
                       io_case("default_link_stream_id_roundtrip",
                               "default-link stream id should survive Graph::load/save"));
      require(stream_link_roundtrip_json.find("\"link_policy\"") == std::string::npos,
              io_case("default_link_policy_roundtrip_omitted",
                      "default-link stream id roundtrip should keep default policy implicit"));

      const std::string missing_file = tmp_json_path("graph_io_missing_input.json");
      {
        std::error_code ec;
        std::filesystem::remove(missing_file, ec);
      }
      require_neat_error([&]() { (void)Graph::load(missing_file); }, error_codes::kIoOpen,
                         "io.open", "failed to open file for reading");

      const std::string tmp_dir =
          (std::filesystem::temp_directory_path() / "sima_neat_graph_io_contract").string();
      {
        std::error_code ec;
        std::filesystem::create_directories(tmp_dir, ec);
      }
      require_neat_error([&]() { graph.save(tmp_dir); }, error_codes::kIoOpen, "io.open",
                         "failed to open file for writing");

      const std::string invalid_root = tmp_json_path("graph_io_invalid_root.json");
      write_text(invalid_root, "[]\n");
      require_neat_error([&]() { (void)Graph::load(invalid_root); }, error_codes::kIoParse,
                         "io.parse", "invalid JSON root");

      const std::string missing_nodes = tmp_json_path("graph_io_missing_nodes.json");
      write_text(missing_nodes, "{\"version\":1}\n");
      require_neat_error([&]() { (void)Graph::load(missing_nodes); }, error_codes::kIoParse,
                         "io.parse", "missing required 'nodes' array");

      const std::string invalid_node_entry = tmp_json_path("graph_io_invalid_node_entry.json");
      write_text(invalid_node_entry, "{\"nodes\":[5]}\n");
      require_neat_error([&]() { (void)Graph::load(invalid_node_entry); }, error_codes::kIoParse,
                         "io.parse", "node entry must be object");

      const std::string missing_fragment = tmp_json_path("graph_io_missing_fragment.json");
      write_text(missing_fragment,
                 "{\"nodes\":[{\"kind\":\"Gst\",\"label\":\"x\",\"elements\":[]}]}\n");
      require_neat_error([&]() { (void)Graph::load(missing_fragment); }, error_codes::kIoParse,
                         "io.parse", "node fragment is empty");

      const std::string malformed_json = tmp_json_path("graph_io_malformed.json");
      write_text(malformed_json, "{\"nodes\":[{\"kind\":\"Gst\" \"fragment\":\"identity\"}]}\n");
      require_neat_error([&]() { (void)Graph::load(malformed_json); }, error_codes::kIoParse,
                         "offset=", "near='");
    }));
