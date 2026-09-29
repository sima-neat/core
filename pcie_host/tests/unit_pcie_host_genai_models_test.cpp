#include "pcie_genai_models.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

namespace fs = std::filesystem;
using simaai::neat::pcie::genai::tools::format_final_stats;
using simaai::neat::pcie::genai::tools::format_model_listing;
using simaai::neat::pcie::genai::tools::list_models_in;
using simaai::neat::pcie::genai::tools::model_type_from_vlm_config;
using simaai::neat::pcie::genai::tools::ModelListing;
using simaai::neat::pcie::genai::tools::serve_root_path;

namespace {

void require(const bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void write_file(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path) << text;
}

// A model folder as `llima run` expects it: devkit/vlm_config.json + elf_files/.
void make_model(const fs::path& root, const std::string& name, const std::string& model_type,
                std::uintmax_t bytes) {
  write_file(root / name / "devkit" / "vlm_config.json",
             "{\"model_type\": \"" + model_type + "\"}");
  write_file(root / name / "elf_files" / "model.elf", std::string(bytes, 'x'));
}

} // namespace

int main() {
  try {
    // --- serve_root_path: only the named [serve] entry, section-aware -------
    const std::string conf = "# comment\n"
                             "[recv]\n"
                             "incoming = /srv/simaai/incoming\n"
                             "models   = /srv/simaai/recv-not-this\n"
                             "\n"
                             "[serve]\n"
                             "data    = /srv/simaai/data\n"
                             "models  = /srv/simaai/models\n";
    require(serve_root_path(conf, "models") == "/srv/simaai/models",
            "reads the [serve] models path, not the [recv] one");
    require(serve_root_path(conf, "data") == "/srv/simaai/data", "reads another serve name");
    require(!serve_root_path(conf, "absent").has_value(), "absent name -> nullopt");
    require(!serve_root_path("no sections here\n", "models").has_value(),
            "no [serve] section -> nullopt");
    require(serve_root_path("[serve]\nmodels=/x\n", "models") == "/x",
            "tolerates no spaces around '='");

    // --- model_type_from_vlm_config ----------------------------------------
    require(model_type_from_vlm_config("{\"model_type\": \"llm-llama3.2\"}") == "llm-llama3.2",
            "extracts model_type");
    require(model_type_from_vlm_config("{\"other\": 1}").empty(), "missing key -> empty");
    require(model_type_from_vlm_config("not json").empty(), "bad json -> empty");

    // --- list_models_in: temp dir ------------------------------------------
    const fs::path root =
        fs::temp_directory_path() / ("pcie_genai_models_test_" + std::to_string(::getpid()));
    fs::remove_all(root);
    make_model(root, "Llama-3.2-3B-Instruct-a16w4", "llm-llama3.2", 2048);
    make_model(root, "Another-Model", "llm-qwen", 1024);
    fs::create_directories(root / "not-a-model" / "devkit"); // no elf_files/ -> skipped
    write_file(root / "stray.txt", "x");                     // a file, not a folder -> skipped

    const std::vector<ModelListing> models = list_models_in(root);
    require(models.size() == 2, "only the two well-formed model folders are listed");
    require(models[0].name == "Another-Model", "sorted by name");
    require(models[1].name == "Llama-3.2-3B-Instruct-a16w4", "sorted by name (2)");
    require(models[1].type == "llm-llama3.2", "type read from vlm_config.json");
    require(models[1].size_bytes >= 2048, "size covers the folder contents");

    require(list_models_in(root / "does-not-exist").empty(), "missing root -> empty, no throw");

    // --- format_model_listing ----------------------------------------------
    const std::string text = format_model_listing(models, "/srv/simaai/models");
    require(text.find("Another-Model") != std::string::npos, "listing shows each model name");
    require(text.find("Llama-3.2-3B-Instruct-a16w4") != std::string::npos, "listing shows names");
    require(text.find("/srv/simaai/models") != std::string::npos,
            "listing shows the serve-root path to copy into");

    const std::string empty_text = format_model_listing({}, "/srv/simaai/models");
    require(empty_text.find("/srv/simaai/models") != std::string::npos,
            "empty listing still shows where to copy models");

    // --- format_final_stats: run summary + drop warning --------------------
    {
      using simaai::neat::genai::GenerationMetrics;
      GenerationMetrics m;
      m.generated_tokens = 293;
      m.time_to_first_token_s = 0.09;
      m.tokens_per_second = 26.73;
      m.dropped_events = 0;
      const std::string line = format_final_stats(m, "stop");
      require(line == "[stop | 293 tokens | TTFT 0.09 s | 26.73 tok/s | dropped 0]",
              "clean run: one summary line, dropped 0, no warning");

      GenerationMetrics d = m;
      d.dropped_events = 2;
      const std::string with_warn = format_final_stats(d, "stop");
      require(with_warn.find("| dropped 2]") != std::string::npos, "the count is in the summary");
      require(with_warn.find("WARNING") != std::string::npos, "a non-zero count prints a warning");
    }

    fs::remove_all(root);
    std::cout << "[PASS] pcie-genai model listing\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[FAIL] " << e.what() << "\n";
    return 1;
  }
}
